import torch
import wandb
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler
from sklearn.model_selection import GroupKFold

from src.train_utils import Criterion, Scheduler, Optimizer, dice_coef

class Training:
    def __init__(self, exp_name, model, dataset, split_strategy, params, device, log_to_wandb=False):
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
            weight_decay=params.get("weight_decay", 1e-3)
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
            train_dataset, val_dataset = random_split(self.dataset, [train_size, val_size], generator=generator)

        elif self.split_strategy == "gkf":
            groups = [int(f.split("_")[0]) for f in self.dataset.json_files]
            gkf = GroupKFold(n_splits=5)
            train_idx, val_idx = list(gkf.split(groups, groups, groups))[0]
            train_groups = [groups[i] for i in train_idx]
            val_groups = [groups[i] for i in val_idx]
            
            print("=" * 80)
            print(f"{len(train_idx)} images on training \nTrain group ids: {sorted(set(train_groups))}")
            print()
            print(f"{len(val_idx)} images on validation \nVal group ids: {sorted(set(val_groups))}")
            print("=" * 80)

            train_dataset = torch.utils.data.Subset(self.dataset, train_idx)
            val_dataset = torch.utils.data.Subset(self.dataset, val_idx)

        else:
            raise ValueError(f"Unknown split strategy: {self.split_strategy}")

        train_loader = DataLoader(train_dataset, batch_size=self.params["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.params["batch_size"], shuffle=False)
        return train_loader, val_loader

    # --------------------------------------------------------------------------
    def train_one_epoch(self, loader):
        self.model.train()
        total_loss, total_dice = 0.0, 0.0
        for images, masks in loader:
            images = images.to(self.device)
            masks = masks.squeeze(1).long().to(self.device)  # ✅ fix dtype + shape
            self.optimizer.zero_grad()

            with autocast(device_type=self.device, enabled=self.params.get("half_precision", False)):
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
            preds = outputs.argmax(dim=1)                       # (B,H,W)
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
                masks  = masks.squeeze(1).long().to(self.device)   # <-- ensure Long & (B,H,W)
    
                outputs = self.model(images)                       # (B,C,H,W)
                loss    = self.criterion(outputs, masks)
    
                preds = outputs.argmax(dim=1)                      # (B,H,W)
                total_loss += loss.item()
                total_dice += dice_coef(preds, masks).item()
    
        n = len(loader)
        return total_loss / n, total_dice / n


    # --------------------------------------------------------------------------
    def train(self):
        train_loader, val_loader = self.prepare_dataloader()

        if self.log_to_wandb:
            wandb.init(project="UBPD", name=self.exp_name, config=self.params)
            wandb.watch(self.model, log="all")

        best_state = None
        for epoch in range(1, self.params["max_epoch"] + 1):
            train_loss, train_dice = self.train_one_epoch(train_loader)
            val_loss, val_dice = self.validate_one_epoch(val_loader)

            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            print(f"[{epoch}/{self.params['max_epoch']}] "
                  f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                  f"Train Dice={train_dice:.4f}, Val Dice={val_dice:.4f}")

            if self.log_to_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_dice": train_dice,
                    "val_dice": val_dice,
                    "lr": self.optimizer.param_groups[0]['lr']
                })

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_state = self.model.state_dict()
                self.best_epoch = epoch
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1

            if self.params.get("early_stopping", True) and self.early_stopping_counter >= self.params.get("patience", 5):
                print(f"Early stopping at epoch {epoch}")
                break

        if best_state:
            torch.save(best_state, f"./data/models/{self.exp_name}_best.pth")
            print(f"✅ Best model saved (epoch {self.best_epoch})")

        if self.log_to_wandb:
            wandb.finish()