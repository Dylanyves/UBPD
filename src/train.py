import torch
import copy
import wandb
import matplotlib.pyplot as plt
import re

from torch.utils.data import DataLoader, random_split, Subset
from torch.amp import autocast, GradScaler
from sklearn.model_selection import GroupKFold
from torchvision import transforms
from torchvision.transforms import InterpolationMode


from src.train_utils import Criterion, Scheduler, Optimizer, dice_coef
from src.dataset import UBPDatasetTest
from src.evaluate import Evaluate


class Training:
    def __init__(
        self,
        exp_name,
        model,
        dataset,
        split_strategy,
        params,
        device,
        log_to_wandb=False,
    ):
        self.exp_name = exp_name
        self.model = model.to(device)
        self.dataset = dataset
        self.split_strategy = split_strategy.lower()
        self.params = params
        self.device = device
        self.log_to_wandb = log_to_wandb

        # Instantiate components using __new__ factories
        self.criterion = Criterion(params.get("criterion", "ce"))
        self.optimizer = Optimizer(
            params.get("optimizer", "adam"),
            self.model.parameters(),
            lr=params.get("learning_rate", 1e-3),
            weight_decay=params.get("weight_decay", 1e-3),
        )
        self.scheduler = Scheduler(params.get("scheduler", None), self.optimizer)
        self.scaler = GradScaler(enabled=params.get("half_precision", False))

        # Tracking
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.early_stopping_counter = 0

    # --------------------------------------------------------------------------
    def prepare_dataloader(self):
        """Prepare train/val splits."""
        if self.split_strategy == "random":
            generator = torch.Generator().manual_seed(111)
            train_size = int(0.8 * len(self.dataset))
            val_size = len(self.dataset) - train_size
            train_dataset, val_dataset = random_split(
                self.dataset, [train_size, val_size], generator=generator
            )

        elif self.split_strategy == "gkf":
            groups = [int(f.split("_")[0]) for f in self.dataset.json_files]
            gkf = GroupKFold(n_splits=5)
            train_idx, val_idx = list(gkf.split(groups, groups, groups))[0]
            train_groups = [groups[i] for i in train_idx]
            val_groups = [groups[i] for i in val_idx]

            print("=" * 80)
            print(
                f"{len(train_idx)} images on training \nTrain group ids: {sorted(set(train_groups))}"
            )
            print()
            print(
                f"{len(val_idx)} images on validation \nVal group ids: {sorted(set(val_groups))}"
            )
            print("=" * 80)

            train_dataset = torch.utils.data.Subset(self.dataset, train_idx)
            val_dataset = torch.utils.data.Subset(self.dataset, val_idx)

        else:
            raise ValueError(f"Unknown split strategy: {self.split_strategy}")

        train_loader = DataLoader(
            train_dataset, batch_size=self.params["batch_size"], shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.params["batch_size"], shuffle=False
        )
        return train_loader, val_loader

    # --------------------------------------------------------------------------
    def train_one_epoch(self, loader):
        self.model.train()
        total_loss, total_dice = 0.0, 0.0
        for images, masks in loader:
            images = images.to(self.device)
            masks = masks.squeeze(1).long().to(self.device)  # ✅ fix dtype + shape
            self.optimizer.zero_grad()

            with autocast(
                device_type=self.device,
                enabled=self.params.get("half_precision", False),
            ):
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

            if self.params.get("half_precision", False):
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            # Multiclass prediction -> argmax
            preds = outputs.argmax(dim=1)  # (B,H,W)
            total_loss += loss.item()
            total_dice += dice_coef(preds, masks).item()

        n = len(loader)
        return total_loss / n, total_dice / n

    # --------------------------------------------------------------------------
    def validate_one_epoch(self, loader):
        self.model.eval()
        total_loss, total_dice = 0.0, 0.0
        with torch.no_grad():
            for images, masks in loader:
                images = images.to(self.device)
                masks = (
                    masks.squeeze(1).long().to(self.device)
                )  # <-- ensure Long & (B,H,W)

                outputs = self.model(images)  # (B,C,H,W)
                loss = self.criterion(outputs, masks)

                preds = outputs.argmax(dim=1)  # (B,H,W)
                total_loss += loss.item()
                total_dice += dice_coef(preds, masks).item()

        n = len(loader)
        return total_loss / n, total_dice / n

        # --------------------------------------------------------------------------

    def _get_groups(self):
        """
        Returns a list of group (patient) ids with len == len(dataset).
        Change this to match how your dataset stores patient IDs.
        Priority:
          1) dataset.patient_ids
          2) dataset.groups
          3) parse from dataset.json_files (e.g., '123_*.json' -> 123)
        """
        if hasattr(self.dataset, "patient_ids"):
            return list(self.dataset.patient_ids)
        if hasattr(self.dataset, "groups"):
            return list(self.dataset.groups)
        if hasattr(self.dataset, "json_files"):
            ids = []
            for f in self.dataset.json_files:
                s = str(f)
                m = re.search(r"\d+", s)  # grab first number as patient id
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

    def _unique_sorted(self, seq):
        """Return unique values as a sorted list; works for str/int patient ids."""
        try:
            return sorted(set(seq), key=lambda x: (str(type(x)), x))
        except TypeError:
            return sorted(set(map(str, seq)))

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    def train(self):
        # -------------------------
        # RANDOM 80/20 single split
        # -------------------------
        if self.split_strategy != "gkf":
            train_loader, val_loader = self.prepare_dataloader()

            if self.log_to_wandb:
                wandb.init(project="UBPD", name=self.exp_name, config=self.params)
                wandb.watch(self.model, log="all")

            best_state = None
            for epoch in range(1, self.params["max_epoch"] + 1):
                train_loss, train_dice = self.train_one_epoch(train_loader)
                val_loss, val_dice = self.validate_one_epoch(val_loader)

                if self.scheduler:
                    if isinstance(
                        self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                    ):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()

                print(
                    f"[{epoch}/{self.params['max_epoch']}] "
                    f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                    f"Train Dice={train_dice:.4f}, Val Dice={val_dice:.4f}"
                )

                if self.log_to_wandb:
                    wandb.log(
                        {
                            "epoch": epoch,
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            "train_dice": train_dice,
                            "val_dice": val_dice,
                            "lr": self.optimizer.param_groups[0]["lr"],
                        }
                    )

                # Early stopping
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    best_state = self.model.state_dict()
                    self.best_epoch = epoch
                    self.early_stopping_counter = 0
                else:
                    self.early_stopping_counter += 1

                if self.params.get(
                    "early_stopping", True
                ) and self.early_stopping_counter >= self.params.get("patience", 5):
                    print(f"Early stopping at epoch {epoch}")
                    break

            if best_state:
                torch.save(best_state, f"./data/models/{self.exp_name}_best.pth")
                print(f"✅ Best model saved (epoch {self.best_epoch})")
                # 🔹 Load the best weights back into the model for evaluation
                self.model.load_state_dict(best_state)
                self.model.eval()

                # 🔹 Run test evaluation for this fold
                print(f"🔎 [Fold {fold}] Evaluating on test set...")
                self.evaluate_on_test(self.model)

            if self.log_to_wandb:
                wandb.finish()
            return  # ---- end non-GKF path ----

        # -------------------------
        # 5-fold GroupKFold training
        # -------------------------
        # Save initial weights so each fold starts from the same initialization
        init_state = copy.deepcopy(self.model.state_dict())

        groups = self._get_groups()
        gkf = GroupKFold(n_splits=5)

        fold_metrics = []  # collect per-fold best results

        for fold, (train_idx, val_idx) in enumerate(
            gkf.split(X=groups, y=groups, groups=groups), 1
        ):
            print("\n" + "=" * 80)
            print(f"📦 Fold {fold}/5 | Train={len(train_idx)} | Val={len(val_idx)}")
            print(f"Train groups: {sorted(set([groups[i] for i in train_idx]))}")
            print(f"Val   groups: {sorted(set([groups[i] for i in val_idx]))}")
            print("=" * 80)

            # Reset model + optimizer/scheduler/scaler + trackers for this fold
            self.model.load_state_dict(init_state)
            self.model.to(self.device)
            # re-create optimizer & scheduler to reset their states
            self.optimizer = Optimizer(
                self.params.get("optimizer", "adam"),
                self.model.parameters(),
                lr=self.params.get("learning_rate", 1e-3),
                weight_decay=self.params.get("weight_decay", 1e-3),
            )
            self.scheduler = Scheduler(
                self.params.get("scheduler", None), self.optimizer
            )
            self.scaler = GradScaler(enabled=self.params.get("half_precision", False))
            self.best_val_loss = float("inf")
            self.best_epoch = 0
            self.early_stopping_counter = 0

            train_loader, val_loader = self._make_loaders_from_indices(
                train_idx, val_idx
            )

            # W&B (one run per fold)
            if self.log_to_wandb:
                run_name = f"{self.exp_name}-fold{fold}"
                wandb.init(
                    project="UBPD", name=run_name, config={**self.params, "fold": fold}
                )
                wandb.watch(self.model, log="all")

            best_state = None
            for epoch in range(1, self.params["max_epoch"] + 1):
                train_loss, train_dice = self.train_one_epoch(train_loader)
                val_loss, val_dice = self.validate_one_epoch(val_loader)

                if self.scheduler:
                    if isinstance(
                        self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                    ):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()

                print(
                    f"[Fold {fold}] [{epoch}/{self.params['max_epoch']}] "
                    f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                    f"Train Dice={train_dice:.4f}, Val Dice={val_dice:.4f}"
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
                            "lr": self.optimizer.param_groups[0]["lr"],
                        }
                    )

                # Early stopping (per fold)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    best_state = copy.deepcopy(self.model.state_dict())
                    self.best_epoch = epoch
                    self.early_stopping_counter = 0
                else:
                    self.early_stopping_counter += 1

                if self.params.get(
                    "early_stopping", True
                ) and self.early_stopping_counter >= self.params.get("patience", 5):
                    print(f"[Fold {fold}] Early stopping at epoch {epoch}")
                    break

            # Save best model per fold
            if best_state:
                out_path = f"./data/models/{self.exp_name}_fold{fold}_best.pth"
                torch.save(best_state, out_path)
                print(
                    f"✅ [Fold {fold}] Best model saved (epoch {self.best_epoch}) -> {out_path}"
                )
                # 🔹 Load the best weights back into the model for evaluation
                self.model.load_state_dict(best_state)
                self.model.eval()

                # 🔹 Run test evaluation for this fold
                print(f"🔎 [Fold {fold}] Evaluating on test set...")
                self.evaluate_on_test(self.model)

            # Log the best result for this fold
            fold_metrics.append(
                {
                    "fold": fold,
                    "best_epoch": self.best_epoch,
                    "best_val_loss": float(self.best_val_loss),
                }
            )

            if self.log_to_wandb:
                wandb.summary["best_val_loss"] = float(self.best_val_loss)
                wandb.summary["best_epoch"] = self.best_epoch
                wandb.finish()

        # After all folds
        print("\nCV summary (best val loss per fold):")
        for m in fold_metrics:
            print(
                f"  Fold {m['fold']}: loss={m['best_val_loss']:.5f} @ epoch {m['best_epoch']}"
            )
        mean_loss = sum(m["best_val_loss"] for m in fold_metrics) / len(fold_metrics)
        print(f"➡️  Mean best val loss across folds: {mean_loss:.5f}")

    def evaluate_on_test(self, model):

        TARGET_SIZE = (512, 512)

        image_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize(TARGET_SIZE, antialias=True)]
        )

        mask_transform = transforms.Compose(
            [
                transforms.Resize(TARGET_SIZE, interpolation=InterpolationMode.NEAREST),
                transforms.PILToTensor(),
            ]
        )

        # TODO: changing the classes will result in CUDA error
        include_classes = [1, 2, 3, 4]
        out_channels = len(include_classes) + 1
        image_dir = "./data/dataset/images"
        json_dir = "./data/dataset/labels/json_train"

        test_dataset = UBPDatasetTest(
            image_dir,
            json_dir,
            transform=image_transform,
            target_transform=mask_transform,
            include_classes=include_classes,
        )

        test_loader = DataLoader(
            test_dataset, batch_size=4, shuffle=False, num_workers=2
        )
        evaluator = Evaluate(
            model=model, dataloader=test_loader, device="cuda", num_classes=out_channels
        )
        mean_dice, per_class = evaluator.evaluate_dice_score()
        print(f"\nMean Dice (no background): {mean_dice:.4f}")

        for cid, (name, score) in per_class.items():
            print(f"Class {cid:>2} ({name:<10}): Dice = {score:.4f}")
