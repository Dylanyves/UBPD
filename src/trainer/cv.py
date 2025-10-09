from .base import BaseTrainer
from src.train_utils import Criterion, Scheduler, Optimizer, dice_coef
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast, GradScaler
from sklearn.model_selection import GroupKFold
import torch
import wandb
import copy


class CVTrainer(BaseTrainer):
    def run(self):
        # Save initial weights so each fold starts from the same initialization
        init_state = copy.deepcopy(self.model.state_dict())
        from torch.utils.data import random_split, Subset

        n_samples = len(self.dataset)
        fold_size = n_samples // 5
        lengths = [fold_size] * 4 + [n_samples - fold_size * 4]
        fold_metrics = []
        for fold in range(5):
            # Get splits for this fold
            splits = random_split(
                self.dataset,
                lengths,
                generator=torch.Generator().manual_seed(
                    self.params.get("seed", 111) + fold
                ),
            )
            val_dataset = splits[fold]
            train_indices = [i for i in range(5) if i != fold]
            train_datasets = [splits[i] for i in train_indices]
            # Concat train datasets
            from torch.utils.data import ConcatDataset

            train_dataset = ConcatDataset(train_datasets)
            print(
                f"\n{'='*80}\n📦 Fold {fold+1}/5 | Train={len(train_dataset)} | Val={len(val_dataset)}"
            )
            self.model.load_state_dict(init_state)
            self.model.to(self.device)
            optimizer = Optimizer(
                self.params.get("optimizer", "adam"),
                self.model.parameters(),
                lr=self.params.get("learning_rate", 1e-3),
                weight_decay=self.params.get("weight_decay", 1e-3),
            )
            scheduler = Scheduler(self.params.get("scheduler", None), optimizer)
            scaler = GradScaler(enabled=self.params.get("half_precision", False))
            criterion = Criterion(self.params.get("criterion", "ce"))
            best_val_loss = float("inf")
            best_epoch = 0
            early_stopping_counter = 0
            best_state = None
            train_loader = DataLoader(
                train_dataset, batch_size=self.params["batch_size"], shuffle=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.params["batch_size"], shuffle=False
            )
            if self.log_to_wandb:
                run_name = f"{self.exp_name}-fold{fold}"
                wandb.init(
                    project="UBPD", name=run_name, config={**self.params, "fold": fold}
                )
                wandb.watch(self.model, log="all")
            for epoch in range(1, self.params["max_epoch"] + 1):
                train_loss, train_dice = self.train_one_epoch(
                    train_loader, criterion, optimizer, scaler
                )
                val_loss, val_dice = self.validate_one_epoch(val_loader, criterion)
                if scheduler:
                    if isinstance(
                        scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                    ):
                        scheduler.step(val_loss)
                    else:
                        scheduler.step()
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"[Fold {fold+1}] [{epoch}/{self.params['max_epoch']}] [lr:{current_lr:.6f}] Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Train Dice={train_dice:.4f}, Val Dice={val_dice:.4f}"
                )
                if self.log_to_wandb:
                    wandb.log(
                        {
                            "fold": fold,
                            "epoch": epoch,
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            "train_dice": train_dice,
                            "val_dice": val_dice,
                            "lr": optimizer.param_groups[0]["lr"],
                        }
                    )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = copy.deepcopy(self.model.state_dict())
                    best_epoch = epoch
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                if self.params.get(
                    "early_stopping", True
                ) and early_stopping_counter >= self.params.get("patience", 5):
                    print(f"[Fold {fold}] Early stopping at epoch {epoch}")
                    break
            if best_state:
                out_path = f"./data/models/{self.exp_name}_fold{fold}_best.pth"
                torch.save(best_state, out_path)
                print(
                    f"✅ [Fold {fold}] Best model saved (epoch {best_epoch}) -> {out_path}"
                )
            fold_metrics.append(
                {
                    "fold": fold,
                    "best_epoch": best_epoch,
                    "best_val_loss": float(best_val_loss),
                }
            )
            if self.log_to_wandb:
                wandb.summary["best_val_loss"] = float(best_val_loss)
                wandb.summary["best_epoch"] = best_epoch
                wandb.finish()
        print("\nCV summary (best val loss per fold):")
        for m in fold_metrics:
            print(
                f"  Fold {m['fold']}: loss={m['best_val_loss']:.5f} @ epoch {m['best_epoch']}"
            )
        mean_loss = sum(m["best_val_loss"] for m in fold_metrics) / len(fold_metrics)
        print(f"➡️  Mean best val loss across folds: {mean_loss:.5f}")

    def _get_groups(self):
        if hasattr(self.dataset, "patient_ids"):
            return list(self.dataset.patient_ids)
        if hasattr(self.dataset, "groups"):
            return list(self.dataset.groups)
        if hasattr(self.dataset, "json_files"):
            ids = []
            import re

            for f in self.dataset.json_files:
                s = str(f)
                m = re.search(r"\d+", s)
                ids.append(m.group(0) if m else s.split("_")[0])
            return ids
        raise AttributeError(
            "Dataset must expose 'patient_ids' or 'groups' or 'json_files' to derive groups."
        )

    def _make_loaders_from_indices(self, train_idx, val_idx):
        train_ds = Subset(self.dataset, train_idx)
        val_ds = Subset(self.dataset, val_idx)
        train_loader = DataLoader(
            train_ds, batch_size=self.params["batch_size"], shuffle=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.params["batch_size"], shuffle=False
        )
        return train_loader, val_loader

    def train_one_epoch(self, loader, criterion, optimizer, scaler):
        self.model.train()
        total_loss, total_dice = 0.0, 0.0
        for images, masks in loader:
            images = images.to(self.device)
            masks = masks.squeeze(1).long().to(self.device)
            optimizer.zero_grad()
            with autocast(
                device_type=self.device,
                enabled=self.params.get("half_precision", False),
            ):
                outputs = self.model(images)
                loss = criterion(outputs, masks)
            if self.params.get("half_precision", False):
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            preds = outputs.argmax(dim=1)
            total_loss += loss.item()
            total_dice += dice_coef(preds, masks).item()
        n = len(loader)
        return total_loss / n, total_dice / n

    def validate_one_epoch(self, loader, criterion):
        self.model.eval()
        total_loss, total_dice = 0.0, 0.0
        with torch.no_grad():
            for images, masks in loader:
                images = images.to(self.device)
                masks = masks.squeeze(1).long().to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, masks)
                preds = outputs.argmax(dim=1)
                total_loss += loss.item()
                total_dice += dice_coef(preds, masks).item()
        n = len(loader)
        return total_loss / n, total_dice / n
