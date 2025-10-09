from .base import BaseTrainer
from src.train_utils import Criterion, Scheduler, Optimizer, dice_coef
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler
import torch
import wandb


class SingleTrainer(BaseTrainer):
    def run(self):
        # Prepare dataloaders
        generator = torch.Generator().manual_seed(self.params.get("seed", 111))
        train_size = int(0.8 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        train_dataset, val_dataset = random_split(
            self.dataset, [train_size, val_size], generator=generator
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self.params["batch_size"], shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.params["batch_size"], shuffle=False
        )

        # Setup
        criterion = Criterion(self.params.get("criterion", "ce"))
        optimizer = Optimizer(
            self.params.get("optimizer", "adam"),
            self.model.parameters(),
            lr=self.params.get("learning_rate", 1e-3),
            weight_decay=self.params.get("weight_decay", 1e-3),
        )
        scheduler = Scheduler(self.params.get("scheduler", None), optimizer)
        scaler = GradScaler(enabled=self.params.get("half_precision", False))
        best_val_loss = float("inf")
        best_epoch = 0
        early_stopping_counter = 0
        best_state = None

        if self.log_to_wandb:
            wandb.init(project="UBPD", name=self.exp_name, config=self.params)
            wandb.watch(self.model, log="all")

        for epoch in range(1, self.params["max_epoch"] + 1):
            train_loss, train_dice = self.train_one_epoch(
                train_loader, criterion, optimizer, scaler
            )
            val_loss, val_dice = self.validate_one_epoch(val_loader, criterion)

            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            print(
                f"[{epoch}/{self.params['max_epoch']}] Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Train Dice={train_dice:.4f}, Val Dice={val_dice:.4f}"
            )

            if self.log_to_wandb:
                wandb.log(
                    {
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
                best_state = self.model.state_dict()
                best_epoch = epoch
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if self.params.get(
                "early_stopping", True
            ) and early_stopping_counter >= self.params.get("patience", 5):
                print(f"Early stopping at epoch {epoch}")
                break

        if best_state:
            torch.save(best_state, f"./data/models/{self.exp_name}_best.pth")
            print(f"✅ Best model saved (epoch {best_epoch})")

        if self.log_to_wandb:
            wandb.finish()

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
