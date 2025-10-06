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
        color = class_colors.get(cid, (0,0,0))
        rgb[mask_np == cid] = color
    return rgb

def _per_sample_mean_dice(logits, targets, num_classes: int, include_background: bool = False, eps: float = 1e-6):
    """
    logits:  [B,C,H,W] float
    targets: [B,H,W]   long
    Returns: [B] per-sample mean Dice (averaged over classes, bg optionally excluded)
    """
    probs = torch.softmax(logits, dim=1)  # [B,C,H,W]
    one_hot = F.one_hot(targets.long(), num_classes=num_classes).permute(0,3,1,2).float()  # [B,C,H,W]

    if not include_background:
        probs = probs[:, 1:, ...]
        one_hot = one_hot[:, 1:, ...]

    # per-sample, per-class dice -> [B,C]
    dims = (2, 3)
    inter = (probs * one_hot).sum(dim=dims)                       # [B,C]
    denom = probs.sum(dim=dims) + one_hot.sum(dim=dims)           # [B,C]
    dice_pc = (2.0 * inter + eps) / (denom + eps)                 # [B,C]
    return dice_pc.mean(dim=1)                                    # [B]


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
        valid = (targets != ignore_index)
        targets = targets.clone()
        targets[~valid] = 0  # safe for one_hot; will be zeroed out later
    else:
        valid = None

    probs = torch.softmax(logits, dim=1)                        # [B,C,H,W]
    one_hot = F.one_hot(targets.long(), num_classes=num_classes).permute(0,3,1,2).float()

    if valid is not None:
        m = valid.unsqueeze(1).float()
        probs = probs * m
        one_hot = one_hot * m

    # sum over batch and spatial dims
    dims = (0, 2, 3)
    inter = (probs * one_hot).sum(dim=dims)                     # [C]
    denom = probs.sum(dim=dims) + one_hot.sum(dim=dims)         # [C]
    return inter, denom

# ----------------------------
# Evaluator
# ----------------------------
class Evaluate:
    def __init__(self,
                 model: torch.nn.Module,
                 dataloader: DataLoader,
                 device: torch.device | str = "cuda",
                 num_classes: int = 5,                 # 0 bg + 1..4 landmarks
                 include_background: bool = False,
                 ignore_index: int | None = None,
                 class_names: dict[int, str] | None = None):
        self.model = model.to(device).eval()
        self.loader = dataloader
        self.device = device
        self.num_classes = num_classes
        self.include_background = include_background
        self.ignore_index = ignore_index
        # Optional pretty printing for classes
        self.class_names = class_names or {0: "background", 1: "artery", 2: "vein", 3: "muscle", 4: "nerve"}

    @torch.no_grad()
    def evaluate_dice_score(self):
        """
        Returns:
            mean_dice (float): mean Dice over selected classes
            per_class (OrderedDict[int, float]): dice per class id
        """
        total_inter = torch.zeros(self.num_classes, dtype=torch.float64, device=self.device)
        total_denom = torch.zeros(self.num_classes, dtype=torch.float64, device=self.device)

        for images, masks in self.loader:
            images = images.to(self.device)                 # [B,1,H,W] or [B,3,H,W]
            masks  = masks.long().to(self.device)           # [B,H,W]

            logits = self.model(images)                     # [B,C,H,W] (logits)
            inter, denom = _accumulate_inter_union(logits, masks, self.num_classes, self.ignore_index)
            total_inter += inter
            total_denom += denom

        eps = 1e-6
        dice_per_class = (2.0 * total_inter + eps) / (total_denom + eps)   # [C]

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
    def visualize_ranked(self,
                         k: int = 3,
                         save_dir: str | None = None,
                         alpha: float = 0.45,
                         class_colors: dict[int, tuple] | None = None):
        """
        Visualize top-k, mid-k, worst-k predictions using overlay panels (GT and Pred).
        - k: how many samples per group
        - save_dir: optional directory to save PNGs (top/mid/worst). If None, just show.
        - alpha: overlay opacity
        - class_colors: optional color map {class_id: (r,g,b)} in [0..1]
        """
        device = self.device
        C = self.num_classes

        # Default pleasant colors (0=bg)
        class_colors = class_colors or {
            0: (0.00, 0.00, 0.00),   # background - black
            1: (0.90, 0.10, 0.10),   # artery     - red
            2: (0.10, 0.40, 0.90),   # vein       - blue
            3: (0.30, 0.80, 0.30),   # muscle     - green
            4: (0.95, 0.85, 0.20),   # nerve      - yellow
        }

        # Collect per-sample scores and tensors (on CPU for plotting)
        samples = []  # list of dicts: {dice, img, gt, pred}
        for batch in self.loader:
            # batch could be (imgs, masks) or (imgs, masks, meta)
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                images, masks, _ = batch
            else:
                images, masks = batch

            images = images.to(device)
            masks  = masks.long().to(device)

            logits = self.model(images)
            dice_b = _per_sample_mean_dice(logits, masks, C, include_background=self.include_background)  # [B]
            preds  = logits.argmax(dim=1)  # [B,H,W] long

            # move to cpu numpy for plotting
            for i in range(images.size(0)):
                img_np = images[i].detach().cpu().numpy()
                # squeeze channel to [H,W], normalize 0..1 for display
                if img_np.ndim == 3:
                    # [C,H,W]; your images are grayscale → C=1; handle generically
                    if img_np.shape[0] == 1:
                        img_np = img_np[0]
                    else:
                        # take mean if multi-channel
                        img_np = img_np.mean(axis=0)
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

                gt_np   = masks[i].detach().cpu().numpy().astype(np.int32)
                pred_np = preds[i].detach().cpu().numpy().astype(np.int32)
                samples.append({
                    "dice": float(dice_b[i].item()),
                    "image": img_np,
                    "gt": gt_np,
                    "pred": pred_np,
                })

        if len(samples) == 0:
            print("No samples collected for visualization.")
            return

        # Rank by dice
        samples_sorted = sorted(samples, key=lambda x: x["dice"])
        n = len(samples_sorted)
        k = min(k, n)

        worst_idxs = list(range(0, k))
        best_idxs  = list(range(n - k, n))
        mid_center = n // 2
        mid_start  = max(0, mid_center - k // 2)
        mid_idxs   = list(range(mid_start, min(mid_start + k, n)))

        groups = [("Worst", worst_idxs), ("Middle", mid_idxs), ("Best", best_idxs)]

        def _plot_group(title, idxs):
            rows = len(idxs)
            cols = 2  # GT overlay, Pred overlay
            fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 3.5*rows))
            if rows == 1:
                axes = np.expand_dims(axes, 0)

            for r, si in enumerate(idxs):
                s = samples_sorted[si]
                img = s["image"]       # [H,W] float 0..1
                gt  = s["gt"]          # [H,W] int
                pr  = s["pred"]        # [H,W] int
                sc  = s["dice"]

                # Color masks
                gt_rgb  = _colorize_mask(gt, class_colors, C)
                pr_rgb  = _colorize_mask(pr, class_colors, C)
                img_rgb = np.stack([img, img, img], axis=-1)

                # Overlays
                gt_overlay = (1 - alpha) * img_rgb + alpha * gt_rgb
                pr_overlay = (1 - alpha) * img_rgb + alpha * pr_rgb

                # Left: GT overlay
                ax = axes[r, 0]
                ax.imshow(gt_overlay)
                ax.set_title(f"GT Overlay", fontsize=11, weight="bold")
                ax.set_xlabel(f"Classes shown; Dice={sc:.3f}")
                ax.axis("off")

                # Right: Pred overlay
                ax = axes[r, 1]
                ax.imshow(pr_overlay)
                ax.set_title("Prediction Overlay", fontsize=11, weight="bold")
                ax.axis("off")

            plt.suptitle(f"{title} {rows} samples — mean Dice (per-sample) rank", fontsize=14, weight="bold", y=0.995)
            plt.tight_layout()
            if save_dir:
                import os
                os.makedirs(save_dir, exist_ok=True)
                out_path = os.path.join(save_dir, f"{title.lower()}_{rows}.png")
                plt.savefig(out_path, dpi=150, bbox_inches="tight")
                print(f"Saved: {out_path}")
            else:
                plt.show()

        for title, idxs in groups:
            if len(idxs) > 0:
                _plot_group(title, idxs)