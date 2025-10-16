import os, math, copy, torch, time
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import wandb

from typing import Dict, Optional, Tuple
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from src.dataset import UBPDataset
from src.train_utils import (
    _make_loss,
    _make_optimizer,
    _make_scheduler,
    dice_coefficient,
)


class Trainer:
    """Trainer with fp16, best-only checkpoint, W&B logging, and early stopping."""

    def __init__(
        self,
        exp_id: str,
        fold_num: int,
        model: nn.Module,
        train_dataset: UBPDataset,
        val_dataset: UBPDataset,
        arguments: Dict,
    ):
        """Initialize trainer and resources."""
        self.exp_id = str(exp_id)
        self.fold_num = int(fold_num)
        self.model = model
        self.args = arguments or {}

        self.device = str(
            self.args.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        self.epochs = int(self.args.get("epochs", 100))
        self.batch_size = int(self.args.get("batch_size", 16))
        self.half_precision = bool(self.args.get("half_precision", True))

        self.ignore_empty = bool(self.args.get("ignore_empty", False))
        self.num_workers = int(self.args.get("num_workers", 2))
        self.pin_memory = bool(self.args.get("pin_memory", True))
        self.grad_clip = float(self.args.get("grad_clip", 0.0))
        self.compute_dice = bool(self.args.get("metric_dice", True))

        # early stopping
        self.patience = int(self.args.get("patience", 10))
        self.restore_best_weights = True
        self._epochs_since_improve = 0
        self._best_state_dict = None

        # checkpoints
        self.save_dir = str(self.args.get("save_dir", "checkpoints"))
        os.makedirs(self.save_dir, exist_ok=True)
        self.best_filename = f"{self.exp_id}_fold_{self.fold_num}.pth"
        self.save_weights_only = bool(self.args.get("save_weights_only", True))
        self.best_path = os.path.join(self.save_dir, self.best_filename)

        # monitor
        self.monitor = str(self.args.get("monitor", "val_loss"))
        self.monitor_mode = str(
            self.args.get(
                "monitor_mode", "min" if self.monitor == "val_loss" else "max"
            )
        ).lower()
        self.best_metric = math.inf if self.monitor_mode == "min" else -math.inf

        # wandb
        self.use_wandb = bool(self.args.get("use_wandb", False))
        if self.use_wandb:
            try:
                api_key = os.getenv("WANDB_API_KEY")
                wandb.login(key=api_key)

                name = f"{self.exp_id}_fold_{fold_num}"
                wandb.init(
                    project="ubpd",
                    group=self.exp_id,
                    name=name,
                    config=self.args,
                    reinit=True,
                )
                wandb.watch(
                    self.model,
                    log="gradients",
                    log_freq=int(self.args.get("wandb_log_freq", 200)),
                )
            except Exception as e:
                print(f"[wandb] Disabled: {e}")

        # data loaders
        collate_fn = self.args.get("collate_fn", None)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.get("val_batch_size", self.batch_size),
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

        # loss/optim/sched
        num_classes = getattr(getattr(self.model, "outc", None), "conv", None)
        num_classes = (
            num_classes.out_channels
            if isinstance(num_classes, nn.Conv2d)
            else int(self.args.get("num_classes", 1))
        )
        self.criterion = _make_loss(self.args, num_classes)
        self.model.to(self.device)
        if (
            isinstance(self.criterion, nn.BCEWithLogitsLoss)
            and self.criterion.pos_weight is not None
        ):
            self.criterion.pos_weight = self.criterion.pos_weight.to(self.device)

        self.optimizer = _make_optimizer(self.model.parameters(), self.args)
        self.sched_type = self.args.get("scheduler", "none").lower()
        steps_per_epoch = len(self.train_loader) if len(self.train_loader) > 0 else None
        self.scheduler = _make_scheduler(
            self.optimizer, self.args, steps_per_epoch=steps_per_epoch
        )

        # AMP scaler
        self.scaler = GradScaler(enabled=self.half_precision and self.device == "cuda")

    def _is_better(self, value: float) -> bool:
        """Return True if current value improves best."""
        return (
            (value < self.best_metric)
            if self.monitor_mode == "min"
            else (value > self.best_metric)
        )

    def _save_best(self, epoch: int, metrics: Dict[str, float]):
        """Save best-only checkpoint and optionally log to wandb."""
        if self.save_weights_only:
            torch.save(self.model.state_dict(), self.best_path)
        else:
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "args": self.args,
                    "metrics": metrics,
                },
                self.best_path,
            )
        if self.use_wandb:
            try:
                art_name = self.args.get(
                    "wandb_artifact_name", os.path.splitext(self.best_filename)[0]
                )
                art = wandb.Artifact(art_name, type="model")
                art.add_file(self.best_path)
                wandb.log_artifact(art)
            except Exception as e:
                print(f"[wandb] Artifact save failed: {e}")

    def train(self) -> Dict:
        """Run training with early stopping and best-only saving, logging to W&B."""
        history = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": []}
        stopped_early = False

        total_time_start = time.perf_counter()
        total_train_time = 0.0
        total_validation_time = 0.0

        for epoch in range(1, self.epochs + 1):
            # ---- Train ----
            t0 = time.perf_counter()
            tr_loss, tr_dice = self.train_one_epoch(epoch)
            train_time = time.perf_counter() - t0
            total_train_time += train_time

            # ---- Validate ----
            t1 = time.perf_counter()
            vl_loss, vl_dice = self.validate_one_epoch(epoch)
            validation_time = time.perf_counter() - t1
            total_validation_time += validation_time

            # ---- Bookkeeping ----
            history["train_loss"].append(tr_loss)
            history["val_loss"].append(vl_loss)
            history["train_dice"].append(tr_dice)
            history["val_dice"].append(vl_dice)

            # Scheduler step
            if self.scheduler:
                if self.sched_type == "onecycle":
                    pass
                elif self.sched_type == "plateau":
                    metric = (
                        vl_dice
                        if self.args.get("plateau_mode", "min") == "max"
                        else vl_loss
                    )
                    self.scheduler.step(metric)
                else:
                    self.scheduler.step()

            # Check best / early stopping
            current = {
                "train_loss": tr_loss,
                "val_loss": vl_loss,
                "train_dice": tr_dice,
                "val_dice": vl_dice,
            }
            monitored_val = current[self.monitor]
            improved = self._is_better(monitored_val)
            if improved:
                self.best_metric = monitored_val
                self._epochs_since_improve = 0
                self._best_state_dict = copy.deepcopy(self.model.state_dict())
                self._save_best(epoch, current)
            else:
                self._epochs_since_improve += 1

            # ---- W&B logging (exact fields requested) ----
            if self.use_wandb:
                current_lr = float(self.optimizer.param_groups[0]["lr"])
                total_time = time.perf_counter() - total_time_start
                # Names mapped to your requested keys:
                epoch_loss = tr_loss
                avg_train_dice = tr_dice
                avg_val_loss = vl_loss
                avg_val_dice = vl_dice
                wandb.log(
                    {
                        "epoch": epoch,  # or epoch+1 if you prefer 1-based
                        "learning_rate": current_lr,
                        "total_time": total_time,
                        "train/loss": epoch_loss,
                        "train/dice_mean": avg_train_dice,
                        "validation/loss": avg_val_loss,
                        "validation/dice_mean": avg_val_dice,
                        "train/time": train_time,
                        "train/time_total": total_train_time,
                        "validation/time": validation_time,
                        "validation/time_total": total_validation_time,
                    }
                )

            # ---- Console print ----
            print(
                f"Epoch {epoch:03d}/{self.epochs} | "
                f"lr={self.optimizer.param_groups[0]['lr']:.3e} "
                f"train_loss={tr_loss:.4f} val_loss={vl_loss:.4f} "
                f"train_dice={tr_dice:.4f} val_dice={vl_dice:.4f} "
                f"{'[BEST]' if improved else ''} "
                f"(patience {self._epochs_since_improve}/{self.patience})"
            )

            if self._epochs_since_improve >= self.patience:
                print(
                    f"⏹️ Early stopping triggered (no improvement for {self.patience} epochs)."
                )
                stopped_early = True
                break

        if self.restore_best_weights and self._best_state_dict is not None:
            self.model.load_state_dict(self._best_state_dict)

        if self.use_wandb:
            if stopped_early:
                wandb.summary["early_stopped"] = True
            wandb.finish()
        return history

    def validate(self) -> Tuple[float, float]:
        """Run a full validation pass."""
        return self.validate_one_epoch(epoch=None)

    def train_one_epoch(self, epoch: Optional[int] = None) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        running_loss = running_metric = 0.0
        n_batches = 0

        for images, targets in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)

            with autocast(
                device_type=self.device,
                dtype=torch.float16,
                enabled=self.half_precision and self.device == "cuda",
            ):
                logits = self.model(images)
                if logits.shape[1] == 1 and targets.dim() == 3:
                    loss = (
                        self.criterion(logits.squeeze(1), targets.float())
                        if isinstance(self.criterion, nn.BCEWithLogitsLoss)
                        else self.criterion(logits, targets)
                    )
                else:
                    loss = self.criterion(logits, targets)

            self.scaler.scale(loss).backward()
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler and self.sched_type == "onecycle":
                self.scheduler.step()

            running_loss += loss.item()
            if self.compute_dice:
                with torch.no_grad():
                    running_metric += dice_coefficient(
                        logits,
                        targets,
                        include_background=False,
                        ignore_empty=self.ignore_empty,
                    ).item()
            n_batches += 1

        return running_loss / max(1, n_batches), (
            running_metric / max(1, n_batches) if self.compute_dice else float("nan")
        )

    def validate_one_epoch(self, epoch: Optional[int] = None) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        running_loss = running_metric = 0.0
        n_batches = 0
        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                with autocast(
                    device_type=self.device,
                    dtype=torch.float16,
                    enabled=self.half_precision and self.device == "cuda",
                ):
                    logits = self.model(images)
                    if logits.shape[1] == 1 and targets.dim() == 3:
                        loss = (
                            self.criterion(logits.squeeze(1), targets.float())
                            if isinstance(self.criterion, nn.BCEWithLogitsLoss)
                            else self.criterion(logits, targets)
                        )
                    else:
                        loss = self.criterion(logits, targets)

                running_loss += loss.item()
                if self.compute_dice:
                    running_metric += dice_coefficient(
                        logits,
                        targets,
                        include_background=False,
                        ignore_empty=self.ignore_empty,
                    ).item()
                n_batches += 1
        return running_loss / max(1, n_batches), (
            running_metric / max(1, n_batches) if self.compute_dice else float("nan")
        )
