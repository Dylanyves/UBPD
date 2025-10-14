import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, Optional


def _make_optimizer(params, args: Dict):
    """Build optimizer from args."""
    opt_name = args.get("optimizer", "adamw").lower()
    lr = float(args.get("lr", 1e-3))
    wd = float(args.get("weight_decay", 1e-4))
    if opt_name == "sgd":
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=float(args.get("momentum", 0.9)),
            weight_decay=wd,
            nesterov=True,
        )
    elif opt_name == "adam":
        b1, b2 = args.get("betas", (0.9, 0.999))
        return torch.optim.Adam(params, lr=lr, betas=(b1, b2), weight_decay=wd)
    else:  # adamw default
        b1, b2 = args.get("betas", (0.9, 0.999))
        return torch.optim.AdamW(params, lr=lr, betas=(b1, b2), weight_decay=wd)


def _make_scheduler(optimizer, args: Dict, steps_per_epoch: Optional[int] = None):
    """Build LR scheduler from args."""
    name = args.get("scheduler", "none").lower()
    epochs = int(args.get("epochs", 20))
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "step":
        step_size = int(args.get("step_size", max(1, epochs // 3)))
        gamma = float(args.get("gamma", 0.1))
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    if name == "onecycle":
        if steps_per_epoch is None:
            return None
        max_lr = float(args.get("lr", 1e-3))
        pct = float(args.get("onecycle_pct", 0.3))
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            pct_start=pct,
        )
    if name == "plateau":
        mode = args.get("plateau_mode", "min")  # "min" for loss, "max" for metric
        factor = float(args.get("plateau_factor", 0.1))  # LR decay factor
        patience = int(args.get("plateau_patience", 5))  # epochs to wait
        threshold = float(args.get("plateau_threshold", 1e-4))
        cooldown = int(args.get("plateau_cooldown", 0))
        min_lr = float(args.get("plateau_min_lr", 0.0))
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
            cooldown=cooldown,
            min_lr=min_lr,
            verbose=False,
        )
    return None


def _make_loss(args: Dict, num_classes: int) -> nn.Module:
    """Build loss from args."""
    loss_name = args.get("loss", "bce" if num_classes == 1 else "ce").lower()
    if loss_name in ("bce", "bcelogits", "bcewithlogits"):
        pos_weight = args.get("pos_weight", None)
        pw = torch.tensor(pos_weight) if pos_weight is not None else None
        return nn.BCEWithLogitsLoss(pos_weight=pw)
    if loss_name in ("ce", "crossentropy", "cross_entropy"):
        ignore_index = int(args.get("ignore_index", 255))
        return nn.CrossEntropyLoss(ignore_index=ignore_index)
    return nn.MSELoss()


def dice_coefficient(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
    include_background: bool = False,
    ignore_empty: bool = True,
) -> torch.Tensor:
    """Compute mean Dice with options to exclude background and/or ignore empty classes."""
    B, C, H, W = logits.shape
    if C == 1:
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()  # [B,1,H,W]
        tgt = (
            targets.float().unsqueeze(1) if targets.dim() == 3 else targets.float()
        )  # [B,1,H,W]
    else:
        labels = torch.argmax(logits, dim=1)  # [B,H,W]
        preds = F.one_hot(labels, num_classes=C).permute(0, 3, 1, 2).float()
        tgt = F.one_hot(targets.long(), num_classes=C).permute(0, 3, 1, 2).float()
        if not include_background:
            preds = preds[:, 1:, ...]
            tgt = tgt[:, 1:, ...]
    inter = (preds * tgt).sum(dim=(2, 3))
    denom = preds.sum(dim=(2, 3)) + tgt.sum(dim=(2, 3))
    dice_per = (2 * inter + eps) / (denom + eps)
    if ignore_empty:
        valid = denom > 0
        dice_per = torch.where(valid, dice_per, torch.nan)
        return torch.nanmean(dice_per)
    else:
        return dice_per.mean()


def plot_history(
    history: dict,
    figsize: tuple = (12, 8),
    title: str | None = None,
    save_path: str | None = None,
    show: bool = True,
):
    """Plot training history with two subplots: (1) train & val loss, (2) train & val dice."""
    # Defensive extraction
    tl = list(history.get("train_loss", []))
    vl = list(history.get("val_loss", []))
    td = list(history.get("train_dice", []))
    vd = list(history.get("val_dice", []))

    n_epochs = max(len(tl), len(vl), len(td), len(vd), 0)
    epochs = np.arange(1, n_epochs + 1)

    plt.style.use("seaborn-darkgrid")
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [1, 1]}
    )

    # Loss plot
    if len(tl) > 0:
        ax1.plot(
            epochs[: len(tl)],
            tl,
            label="train loss",
            color="#1f77b4",
            marker="o",
            linewidth=2,
            alpha=0.9,
        )
    if len(vl) > 0:
        ax1.plot(
            epochs[: len(vl)],
            vl,
            label="val loss",
            color="#ff7f0e",
            marker="s",
            linewidth=2,
            alpha=0.95,
        )
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Loss per epoch", fontsize=13, weight="bold")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.35)

    # Mark best val loss if available
    if len(vl) > 0:
        try:
            best_idx = int(np.nanargmin(vl)) + 1
            best_val = float(vl[best_idx - 1])
            ax1.axvline(best_idx, color="#2ca02c", linestyle="--", alpha=0.6)
            ax1.text(
                best_idx + 0.2,
                ax1.get_ylim()[1] - 0.05 * (ax1.get_ylim()[1] - ax1.get_ylim()[0]),
                f"best val @ {best_idx}\n{best_val:.4f}",
                color="#2ca02c",
                fontsize=10,
            )
        except Exception:
            pass

    # Dice plot
    if len(td) > 0:
        ax2.plot(
            epochs[: len(td)],
            td,
            label="train dice",
            color="#17becf",
            marker="o",
            linewidth=2,
            alpha=0.9,
        )
    if len(vd) > 0:
        ax2.plot(
            epochs[: len(vd)],
            vd,
            label="val dice",
            color="#d62728",
            marker="s",
            linewidth=2,
            alpha=0.95,
        )
    ax2.set_ylabel("Dice", fontsize=12)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_title("Dice per epoch", fontsize=13, weight="bold")
    ax2.legend(loc="lower right")
    ax2.grid(alpha=0.35)

    if title:
        fig.suptitle(title, fontsize=16, weight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        try:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        except Exception as e:
            print(f"Failed to save figure to {save_path}: {e}")

    if show:
        try:
            plt.show()
        except Exception:
            # In headless envs, show may fail — just return the figure
            pass

    return fig
