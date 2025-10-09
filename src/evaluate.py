import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader
from collections import OrderedDict


def _colorize_mask(mask_np, class_colors, num_classes):
    """
    mask_np: [H,W] int (0..C-1)
    class_colors: dict {class_id: (r,g,b)} in [0..1]
    returns RGB float image [H,W,3]
    """
    h, w = mask_np.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for cid in range(num_classes):
        color = class_colors.get(cid, (0, 0, 0))
        rgb[mask_np == cid] = color
    return rgb


def _per_sample_mean_dice(
    logits,
    targets,
    num_classes: int,
    include_background: bool = False,
    eps: float = 1e-6,
):
    """
    logits:  [B,C,H,W] float
    targets: [B,H,W]   long
    Returns: [B] per-sample mean Dice (averaged over classes, bg optionally excluded)
    """
    probs = torch.softmax(logits, dim=1)  # [B,C,H,W]
    one_hot = (
        F.one_hot(targets.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
    )  # [B,C,H,W]

    if not include_background:
        probs = probs[:, 1:, ...]
        one_hot = one_hot[:, 1:, ...]

    # per-sample, per-class dice -> [B,C]
    dims = (2, 3)
    inter = (probs * one_hot).sum(dim=dims)  # [B,C]
    denom = probs.sum(dim=dims) + one_hot.sum(dim=dims)  # [B,C]
    dice_pc = (2.0 * inter + eps) / (denom + eps)  # [B,C]
    return dice_pc.mean(dim=1)  # [B]


# ----------------------------
# Multiclass Dice (aggregated)
# ----------------------------
def _accumulate_inter_union(logits, targets, num_classes: int, ignore_index=None):
    """
    Accumulate (per-class) intersection and union across a batch for stable micro-averaging.
    logits: [B, C, H, W] float
    targets: [B, H, W] long
    """
    if ignore_index is not None:
        valid = targets != ignore_index
        targets = targets.clone()
        targets[~valid] = 0  # safe for one_hot; will be zeroed out later
    else:
        valid = None

    probs = torch.softmax(logits, dim=1)  # [B,C,H,W]
    one_hot = (
        F.one_hot(targets.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
    )

    if valid is not None:
        m = valid.unsqueeze(1).float()
        probs = probs * m
        one_hot = one_hot * m

    # sum over batch and spatial dims
    dims = (0, 2, 3)
    inter = (probs * one_hot).sum(dim=dims)  # [C]
    denom = probs.sum(dim=dims) + one_hot.sum(dim=dims)  # [C]
    return inter, denom


# ----------------------------
# Evaluator
# ----------------------------
class Evaluate:
    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: torch.device | str = "cuda",
        num_classes: int = 5,  # 0 bg + 1..4 landmarks
        include_background: bool = False,
        ignore_index: int | None = None,
        class_names: dict[int, str] | None = None,
    ):
        self.model = model.to(device).eval()
        self.loader = dataloader
        self.device = device
        self.num_classes = num_classes
        self.include_background = include_background
        self.ignore_index = ignore_index
        # Optional pretty printing for classes
        self.class_names = class_names or {
            0: "background",
            1: "artery",
            2: "vein",
            3: "muscle",
            4: "nerve",
        }

    @torch.no_grad()
    def evaluate_dice_score(self):
        """
        Returns:
            mean_dice (float): mean Dice over selected classes
            per_class (OrderedDict[int, float]): dice per class id
        """
        total_inter = torch.zeros(
            self.num_classes, dtype=torch.float64, device=self.device
        )
        total_denom = torch.zeros(
            self.num_classes, dtype=torch.float64, device=self.device
        )

        for images, masks in self.loader:
            images = images.to(self.device)  # [B,1,H,W] or [B,3,H,W]
            masks = masks.long().to(self.device)  # [B,H,W]

            logits = self.model(images)  # [B,C,H,W] (logits)
            inter, denom = _accumulate_inter_union(
                logits, masks, self.num_classes, self.ignore_index
            )
            total_inter += inter
            total_denom += denom

        eps = 1e-6
        dice_per_class = (2.0 * total_inter + eps) / (total_denom + eps)  # [C]

        # include/exclude background in the mean
        start_c = 0 if self.include_background else 1
        selected = dice_per_class[start_c:]
        mean_dice = float(selected.mean().item())

        # package per-class nicely
        per_class = OrderedDict()
        for c in range(self.num_classes):
            name = self.class_names.get(c, f"class_{c}")
            per_class[c] = (name, float(dice_per_class[c].item()))

        return mean_dice, per_class

    @torch.no_grad()
    def visualize_ranked(
        self,
        image_name,
        save_dir: str | None = None,
        alpha: float = 0.45,
        class_colors: dict[int, tuple] | None = None,
    ):
        """
        Save 6 images: top 2, worst 2, average 2 predictions. Plot title shows dice score and image id.
        """
        device = self.device
        C = self.num_classes

        # Default pleasant colors (0=bg)
        class_colors = class_colors or {
            0: (0.00, 0.00, 0.00),  # background - black
            1: (0.90, 0.10, 0.10),  # artery     - red
            2: (0.10, 0.40, 0.90),  # vein       - blue
            3: (0.30, 0.80, 0.30),  # muscle     - green
            4: (0.95, 0.85, 0.20),  # nerve      - yellow
        }

        # Collect per-sample scores and tensors (on CPU for plotting)
        samples = []  # list of dicts: {dice, img, gt, pred, idx}
        global_idx = 0
        for batch in self.loader:
            # batch could be (imgs, masks) or (imgs, masks, meta)
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                images, masks, meta = batch
            else:
                images, masks = batch
                meta = None

            images = images.to(device)
            masks = masks.long().to(device)

            logits = self.model(images)
            dice_b = _per_sample_mean_dice(
                logits, masks, C, include_background=self.include_background
            )  # [B]
            preds = logits.argmax(dim=1)  # [B,H,W] long

            # move to cpu numpy for plotting
            for i in range(images.size(0)):
                img_np = images[i].detach().cpu().numpy()
                # squeeze channel to [H,W], normalize 0..1 for display
                if img_np.ndim == 3:
                    if img_np.shape[0] == 1:
                        img_np = img_np[0]
                    else:
                        img_np = img_np.mean(axis=0)
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

                gt_np = masks[i].detach().cpu().numpy().astype(np.int32)
                pred_np = preds[i].detach().cpu().numpy().astype(np.int32)
                # Try to get <p_id>_<img_id> from meta if available, else try to parse from filename
                img_id = None
                if meta is not None and hasattr(meta, "__getitem__") and len(meta) > i:
                    meta_item = meta[i]
                    if (
                        isinstance(meta_item, dict)
                        and "p_id" in meta_item
                        and "img_id" in meta_item
                    ):
                        img_id = f"{meta_item['p_id']}_{meta_item['img_id']}"
                    elif isinstance(meta_item, (tuple, list)) and len(meta_item) >= 2:
                        img_id = f"{meta_item[0]}_{meta_item[1]}"
                    elif isinstance(meta_item, str):
                        # Try to parse from string like '23_21.jpg'
                        import re

                        m = re.match(r"(\d+)_(\d+)", meta_item)
                        if m:
                            img_id = f"{m.group(1)}_{m.group(2)}"
                        else:
                            img_id = meta_item
                    else:
                        img_id = str(meta_item)
                if img_id is None:
                    # Try to get from dataset if possible (e.g., self.loader.dataset)
                    # Fallback: try to get from dataset.json_files if available
                    dataset = getattr(self.loader, "dataset", None)
                    if dataset is not None and hasattr(dataset, "json_files"):
                        try:
                            fname = dataset.json_files[global_idx]
                            import re

                            m = re.match(r"(\d+)_(\d+)", str(fname))
                            if m:
                                img_id = f"{m.group(1)}_{m.group(2)}"
                            else:
                                img_id = str(fname)
                        except Exception:
                            img_id = f"img_{global_idx}"
                    else:
                        img_id = f"img_{global_idx}"
                samples.append(
                    {
                        "dice": float(dice_b[i].item()),
                        "image": img_np,
                        "gt": gt_np,
                        "pred": pred_np,
                        "img_id": img_id,
                    }
                )
                global_idx += 1

        if len(samples) == 0:
            print("No samples collected for visualization.")
            return

        # Rank by dice
        samples_sorted = sorted(samples, key=lambda x: x["dice"])
        n = len(samples_sorted)
        k = min(2, n)

        # Indices for worst 2, best 2, and average 2
        worst_idxs = list(range(0, k))
        best_idxs = list(range(n - k, n))
        mid_center = n // 2
        if n >= 2:
            if n % 2 == 0:
                mid_idxs = [mid_center - 1, mid_center]
            else:
                mid_idxs = [max(0, mid_center - 1), min(n - 1, mid_center + 1)]
        else:
            mid_idxs = list(range(n))

        selected = [
            ("worst", worst_idxs),
            ("average", mid_idxs),
            ("best", best_idxs),
        ]

        import os

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        for group, idxs in selected:
            for si in idxs:
                s = samples_sorted[si]
                img = s["image"]
                gt = s["gt"]
                pr = s["pred"]
                sc = s["dice"]
                img_id = s["img_id"]

                # Color masks
                gt_rgb = _colorize_mask(gt, class_colors, C)
                pr_rgb = _colorize_mask(pr, class_colors, C)
                img_rgb = np.stack([img, img, img], axis=-1)

                # Overlays
                gt_overlay = (1 - alpha) * img_rgb + alpha * gt_rgb
                pr_overlay = (1 - alpha) * img_rgb + alpha * pr_rgb

                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                axes[0].imshow(gt_overlay)
                axes[0].set_title(
                    f"GT Overlay\nDice={sc:.3f} | ID={img_id}",
                    fontsize=11,
                    weight="bold",
                )
                axes[0].axis("off")
                axes[1].imshow(pr_overlay)
                axes[1].set_title(
                    f"Prediction Overlay\nDice={sc:.3f} | ID={img_id}",
                    fontsize=11,
                    weight="bold",
                )
                axes[1].axis("off")
                plt.suptitle(
                    f"{group.capitalize()} | Dice={sc:.3f} | ID={img_id}",
                    fontsize=14,
                    weight="bold",
                    y=0.995,
                )
                plt.tight_layout()
                if save_dir:
                    out_path = os.path.join(
                        save_dir, f"{image_name}_{group}_{img_id}.png"
                    )
                    plt.savefig(out_path, dpi=150, bbox_inches="tight")
                    print(f"Saved: {out_path}")
                else:
                    plt.show()
