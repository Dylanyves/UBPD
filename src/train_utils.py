import torch.optim as optim
import torch.nn as nn


import torch
import torch.nn.functional as F

@torch.no_grad()
def dice_coef(preds, targets, num_classes=5, eps=1e-6, include_background=True):
    """
    preds: logits [B,C,H,W] (float) OR hard labels [B,H,W] (long)
    targets: int labels [B,H,W] (long)
    """
    if preds.ndim == 4 and preds.dtype.is_floating_point:
        # logits -> probabilities
        probs = torch.softmax(preds, dim=1)                         # [B,C,H,W]
    elif preds.ndim == 3 and preds.dtype in (torch.long, torch.int64):
        # hard labels -> one-hot then treat as 'probs'
        probs = F.one_hot(preds, num_classes=num_classes).permute(0,3,1,2).float()
    else:
        raise ValueError("preds must be logits [B,C,H,W] (float) or hard labels [B,H,W] (long)")

    one_hot = F.one_hot(targets.long(), num_classes=num_classes).permute(0,3,1,2).float()

    if not include_background:
        probs   = probs[:, 1:, ...]
        one_hot = one_hot[:, 1:, ...]

    dims = (0,2,3)
    inter = (probs * one_hot).sum(dim=dims)
    denom = probs.sum(dim=dims) + one_hot.sum(dim=dims)
    dice_per_class = (2*inter + eps) / (denom + eps)
    return dice_per_class.mean()


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        num = 2 * (preds * targets).sum(dim=(2,3)) + self.eps
        den = preds.sum(dim=(2,3)) + targets.sum(dim=(2,3)) + self.eps
        return 1 - (num / den).mean()

class Criterion:
    def __new__(cls, name):
        name = name.lower()

        # Binary segmentation losses
        if name == "bce":
            return nn.BCEWithLogitsLoss()

        elif name == "dice":
            return DiceLoss()

        elif name == "bcedice":
            return lambda pred, mask: (
                nn.BCEWithLogitsLoss()(pred, mask) + DiceLoss()(pred, mask)
            ) / 2

        # Multi-class segmentation losses
        elif name in ["ce", "crossentropy", "cross_entropy"]:
            return nn.CrossEntropyLoss()

        elif name in ["cedice", "crossentropy_dice"]:
            # Combined CE + Dice for multi-class segmentation
            return lambda pred, mask: (
                nn.CrossEntropyLoss()(pred, mask)
                + DiceLoss(mode="multiclass")(pred, mask)
            ) / 2

        raise ValueError(f"❌ Unknown criterion: {name}")



class Optimizer:
    def __new__(cls, name, model_params, lr, weight_decay=1e-3):
        name = name.lower()
        if name == "adam":
            return optim.Adam(model_params, lr=lr, weight_decay=weight_decay)
        elif name == "adamw":
            return optim.AdamW(model_params, lr=lr, weight_decay=weight_decay)
        elif name == "rmsprop":
            return optim.RMSprop(model_params, lr=lr, alpha=0.9, weight_decay=weight_decay)
        raise ValueError(f"Unknown optimizer: {name}")



class Scheduler:
    def __new__(cls, name, optimizer, **kwargs):
        if not name:
            return None
        name = name.lower()
        if name == "rrlonp":
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=5
            )
        elif name == "ca":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=50, eta_min=1e-6
            )
        raise ValueError(f"Unknown scheduler: {name}")