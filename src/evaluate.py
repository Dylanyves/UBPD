import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from torch.amp import autocast  # <-- use torch.amp autocast


class Evaluator:
    """Evaluate a trained segmentation model on UBPDataset (clean & modular)."""

    # ----------- Construction ----------- #
    def __init__(
        self,
        model: torch.nn.Module,
        test_dataset,  # UBPDataset
        num_classes: int,
        device: Optional[torch.device] = None,
        batch_size: int = 8,
        num_workers: int = 2,
        pin_memory: bool = True,
        half_precision: bool = True,  # fp16 on CUDA
        ignore_empty_classes: bool = True,  # treatment of empty classes
    ):
        self.model = model
        self.ds = test_dataset
        self.K = int(num_classes)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.bs = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.half_precision = bool(half_precision)
        self.ignore_empty = bool(ignore_empty_classes)

        # dataset → English names & hex palette
        self._en_names: Dict[str, str] = {
            "dongmai": "artery",
            "jingmai": "vein",
            "jirouzuzhi": "muscle",
            "shenjing": "nerve",
        }
        self._colors_hex: Dict[str, str] = {
            "dongmai": "#27ae60",
            "jingmai": "#2980b9",
            "jirouzuzhi": "#f39c12",
            "shenjing": "#e74c3c",
        }

        # id → raw dataset name (fallback to class_i)
        if hasattr(self.ds, "class_map"):
            self.id2name = {v: k for k, v in getattr(self.ds, "class_map", {}).items()}
        else:
            self.id2name = {i: f"class_{i}" for i in range(self.K)}

        # simple fallback color palette (for box colors)
        base_colors = [
            (0, 0, 0),
            (231, 76, 60),
            (41, 128, 185),
            (243, 156, 18),
            (39, 174, 96),
            (155, 89, 182),
            (52, 152, 219),
            (230, 126, 34),
            (46, 204, 113),
        ]
        self.colors = [(r / 255.0, g / 255.0, b / 255.0) for (r, g, b) in base_colors]
        if self.K >= len(self.colors):
            rng = np.random.RandomState(0)
            for _ in range(self.K - len(self.colors) + 1):
                self.colors.append(tuple(rng.rand(3).tolist()))

        self.loader = DataLoader(
            self.ds,
            batch_size=self.bs,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        self.model.to(self.device).eval()

    # ----------- Low-level helpers ----------- #
    def _autocast_enabled(self) -> bool:
        return self.half_precision and (self.device.type == "cuda")

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            with autocast(
                device_type=("cuda" if self.device.type == "cuda" else "cpu"),
                dtype=torch.float16,
                enabled=self._autocast_enabled(),
            ):
                return self.model(x)

    def _postprocess_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Return one-hot predictions [B,K,H,W] (binary → [B,1,H,W])."""
        if self.K == 1:
            probs = torch.sigmoid(logits)
            return (probs > 0.5).float()
        labels = torch.argmax(logits, dim=1)
        return F.one_hot(labels, num_classes=self.K).permute(0, 3, 1, 2).float()

    @staticmethod
    def _to_numpy_img(img_t: torch.Tensor) -> np.ndarray:
        """Convert [C,H,W] (1 or 3 channels) to [H,W] float in [0,1] for display."""
        x = img_t.detach().cpu().numpy()
        if x.ndim == 3:
            x = x[0] if x.shape[0] == 1 else x.mean(axis=0)
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        return x

    @staticmethod
    def _stem_from_path(p: str) -> str:
        base = os.path.basename(p)
        stem, _ = os.path.splitext(base)
        return stem

    def _hex_to_rgb01(self, hx: str) -> Tuple[float, float, float]:
        hx = (
            hx
            if isinstance(hx, str) and hx.startswith("#") and len(hx) == 7
            else "#7f8c8d"
        )
        return tuple(int(hx[i : i + 2], 16) / 255.0 for i in (1, 3, 5))

    # ----------- Class-label utilities ----------- #
    def _class_indices_and_labels(self) -> Tuple[List[int], List[str]]:
        """Return (class_indices, english_labels)."""
        if self.K > 1:
            idxs = list(range(1, self.K))  # skip bg=0
            labels = []
            for c in idxs:
                raw = self.id2name.get(c, f"class_{c}")
                labels.append(self._en_names.get(raw, raw))
            return idxs, labels
        return [1], ["foreground"]

    def _class_colors_rgb(self) -> Dict[int, Tuple[float, float, float]]:
        """Map class id → (r,g,b) in 0..1 (0=bg)."""
        colors = {0: (0.0, 0.0, 0.0)}
        if self.K == 1:
            colors[1] = self._hex_to_rgb01(self._colors_hex.get("dongmai", "#27ae60"))
        else:
            for cid in range(1, self.K):
                raw = self.id2name.get(cid, None)
                hx = (
                    self._colors_hex.get(raw, "#7f8c8d")
                    if raw is not None
                    else "#7f8c8d"
                )
                colors[cid] = self._hex_to_rgb01(hx)
        return colors

    # ----------- Dice helpers ----------- #
    def _dice_per_sample_binary(
        self, logits: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        """Binary per-sample dice [B]."""
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long().squeeze(1)
        tgt = masks.long()
        inter = (preds * tgt).flatten(1).sum(dim=1).float()
        denom = preds.flatten(1).sum(dim=1).float() + tgt.flatten(1).sum(dim=1).float()
        dice = (2 * inter + 1e-6) / (denom + 1e-6)
        if self.ignore_empty:
            dice = torch.where(
                denom > 0, dice, torch.tensor(float("nan"), device=dice.device)
            )
        return dice

    def _dice_per_sample_multiclass(
        self, logits: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        """Multi-class mean dice (exclude bg) per sample [B]."""
        C = self.K
        preds = logits.argmax(dim=1)
        one_p = F.one_hot(preds, num_classes=C).permute(0, 3, 1, 2).float()[:, 1:, ...]
        one_t = F.one_hot(masks, num_classes=C).permute(0, 3, 1, 2).float()[:, 1:, ...]
        inter = (one_p * one_t).sum(dim=(2, 3))  # [B, C-1]
        denom = one_p.sum(dim=(2, 3)) + one_t.sum(dim=(2, 3))
        dice_c = (2 * inter + 1e-6) / (denom + 1e-6)  # [B, C-1]
        if self.ignore_empty:
            dice_c = torch.where(
                denom > 0, dice_c, torch.tensor(float("nan"), device=dice_c.device)
            )
            return torch.nanmean(dice_c, dim=1)  # ignore empty classes in mean
        return dice_c.mean(dim=1)

    # ----------- Public: metrics ----------- #
    def evaluate_dice_score(
        self, show_plot: bool = True
    ) -> Dict[int, Dict[str, float]]:
        """Compute per-class Dice (no background), print mean±std, and show a clean box plot."""
        class_idxs, class_labels = self._class_indices_and_labels()

        per_class_scores: Dict[int, List[float]] = {c: [] for c in class_idxs}
        pooled_scores: List[float] = []

        for images, masks, *rest in self.loader:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            logits = self._forward(images)
            preds_oh = self._postprocess_logits(logits)
            t_oh = (
                masks.float().unsqueeze(1)
                if self.K == 1
                else F.one_hot(masks.long(), num_classes=self.K)
                .permute(0, 3, 1, 2)
                .float()
            )

            for c in class_idxs:
                pc = preds_oh[:, (0 if self.K == 1 else c), ...]
                tc = t_oh[:, (0 if self.K == 1 else c), ...]
                inter = (pc * tc).flatten(1).sum(dim=1)
                denom = pc.flatten(1).sum(dim=1) + tc.flatten(1).sum(dim=1)
                dice_ci = (2 * inter + 1e-6) / (denom + 1e-6)  # [B]
                vals = (
                    (dice_ci[denom > 0] if self.ignore_empty else dice_ci)
                    .detach()
                    .cpu()
                    .tolist()
                )
                per_class_scores[c].extend(vals)
                pooled_scores.extend(vals)

        # stats
        stats: Dict[int, Dict[str, float]] = {}
        for c in class_idxs:
            arr = np.asarray(per_class_scores[c], dtype=np.float32)
            stats[c] = {
                "mean": float(np.nanmean(arr)) if arr.size else float("nan"),
                "std": float(np.nanstd(arr)) if arr.size else float("nan"),
                "median": float(np.nanmedian(arr)) if arr.size else float("nan"),
                "n": int(arr.size),
            }
        pooled_arr = np.asarray(pooled_scores, dtype=np.float32)
        overall = {
            "mean": float(np.nanmean(pooled_arr)) if pooled_arr.size else float("nan"),
            "std": float(np.nanstd(pooled_arr)) if pooled_arr.size else float("nan"),
            "n": int(pooled_arr.size),
        }

        # console summary
        mode_txt = (
            "IGNORING empties" if self.ignore_empty else "INCLUDING empties as Dice=1"
        )
        print(f"\n📊 Dice scores (foreground only; {mode_txt}):")
        print(
            f"  Overall (pooled) mean±std: {overall['mean']:.4f} ± {overall['std']:.4f}  (n={overall['n']})"
        )
        for c, lbl in zip(class_idxs, class_labels):
            s = stats[c]
            print(
                f"  {lbl:<20s} mean±std: {s['mean']:.4f} ± {s['std']:.4f}  (median={s['median']:.4f}, n={s['n']})"
            )

        # plot
        if show_plot and len(class_idxs) > 0:
            self._plot_per_class_box(
                per_class_scores, class_idxs, class_labels, stats, overall
            )

        return {"overall": overall, **stats}

    # ----------- Public: visualization ----------- #
    @torch.no_grad()
    def visualize_ranked(
        self,
        image_name: str = "img",
        save_dir: Optional[str] = None,
        alpha: float = 0.45,
    ):
        """Save clean worst/average/best overlays with Dice and English class colors, using true <p>_<img> ids."""
        class_colors = self._class_colors_rgb()
        samples: List[Dict] = []
        abs_i = 0  # absolute index across loader (not used for naming if meta gives real id)

        for batch in self.loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                images, masks = batch[:2]
                meta = batch[2] if len(batch) >= 3 else None
            else:
                images, masks, meta = batch, None, None  # defensive

            images = images.to(self.device)
            masks = masks.long().to(self.device)

            with autocast(
                device_type=("cuda" if self.device.type == "cuda" else "cpu"),
                dtype=torch.float16,
                enabled=self._autocast_enabled(),
            ):
                logits = self.model(images)

            # per-sample dice
            dice_b = (
                self._dice_per_sample_binary(logits, masks)
                if self.K == 1
                else self._dice_per_sample_multiclass(logits, masks)
            )
            preds = (
                (torch.sigmoid(logits) > 0.5).long().squeeze(1)
                if self.K == 1
                else logits.argmax(dim=1)
            )

            B = images.size(0)
            for i in range(B):
                score_val = (
                    float(dice_b[i].item()) if not np.isnan(dice_b[i].item()) else None
                )
                if (
                    self.ignore_empty and score_val is None
                ):  # skip empty-empty when ignoring empties
                    abs_i += 1
                    continue

                img_np = self._to_numpy_img(images[i])
                gt_np = masks[i].detach().cpu().numpy().astype(np.int32)
                pr_np = preds[i].detach().cpu().numpy().astype(np.int32)

                true_id = self._extract_true_id(abs_i, meta, i)
                samples.append(
                    {
                        "dice": (1.0 if score_val is None else float(score_val)),
                        "image": img_np,
                        "gt": gt_np,
                        "pred": pr_np,
                        "img_id": true_id,
                    }
                )
                abs_i += 1

        if not samples:
            print("No samples collected for visualization.")
            return

        # rank & pick
        samples_sorted = sorted(samples, key=lambda x: x["dice"])
        groups = self._rank_groups(len(samples_sorted))

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # render
        for group_name, idxs in groups:
            for si in idxs:
                s = samples_sorted[si]
                self._plot_overlay_pair(
                    s["image"],
                    s["gt"],
                    s["pred"],
                    s["dice"],
                    s["img_id"],
                    group_name,
                    class_colors,
                    alpha,
                    save_dir,
                    image_name,
                )

    # ----------- Mid-level helpers (plotting & ids) ----------- #
    def _plot_per_class_box(
        self,
        per_class_scores: Dict[int, List[float]],
        class_idxs: List[int],
        class_labels: List[str],
        stats: Dict[int, Dict[str, float]],
        overall: Dict[str, float],
    ) -> None:
        labels = class_labels
        data = [
            per_class_scores[c] if len(per_class_scores[c]) > 0 else [np.nan]
            for c in class_idxs
        ]

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(data)), 5.6))
        bp = ax.boxplot(
            data,
            labels=labels,
            showmeans=True,
            meanline=True,
            patch_artist=True,
            boxprops=dict(linewidth=1.5, facecolor="#cccccc", alpha=0.25),
            medianprops=dict(linewidth=2.0, color="#d62728"),
            whiskerprops=dict(linewidth=1.25),
            capprops=dict(linewidth=1.25),
            meanprops=dict(linewidth=1.5, linestyle="--", color="#1f77b4"),
            flierprops=dict(marker="", markersize=0),
        )

        # subtle box colors
        palette = [self.colors[c] if len(self.colors) > c else None for c in class_idxs]
        for patch, col in zip(bp["boxes"], palette):
            if col is not None:
                patch.set_facecolor(col)
                patch.set_alpha(0.22)

        y_min = 0.0
        ax.set_ylim(y_min, 1.05)
        for i, c in enumerate(class_idxs, start=1):
            s = stats[c]
            ax.text(
                i,
                y_min - 0.05,
                f"n={s['n']}\nμ={s['mean']:.2f}, σ={s['std']:.2f}",
                ha="center",
                va="top",
                fontsize=9,
                transform=ax.get_xaxis_transform(),
            )

        if overall["n"] > 0 and not np.isnan(overall["mean"]):
            ax.axhline(
                overall["mean"],
                linestyle="--",
                linewidth=1.3,
                alpha=0.7,
                color="gray",
                label=f"Overall μ={overall['mean']:.3f}",
            )
            ax.legend(loc="upper right", framealpha=0.9)

        suffix = " • ignore-empty" if self.ignore_empty else " • include-empty"
        ttl = "Per-class Dice (excluding background)" + suffix
        if overall["n"] > 0:
            ttl += f" • Overall μ={overall['mean']:.3f} ± {overall['std']:.3f}"
        ax.set_title(ttl)
        ax.set_ylabel("Dice score")
        ax.grid(True, axis="y", alpha=0.28)
        plt.tight_layout()
        plt.show()

    def _extract_true_id(self, abs_index: int, meta, i_in_batch: int) -> str:
        """Derive `<patient>_<image>` id using meta or dataset filenames."""
        # meta dict with p_id & img_id
        if isinstance(meta, (list, tuple)) and len(meta) > i_in_batch:
            mi = meta[i_in_batch]
            if isinstance(mi, dict) and "p_id" in mi and "img_id" in mi:
                return f"{mi['p_id']}_{mi['img_id']}"
            if isinstance(mi, str):
                return self._stem_from_path(mi)

        # dataset filenames (UBPDataset.json_files or image_files)
        if hasattr(self.ds, "json_files"):
            try:
                return self._stem_from_path(self.ds.json_files[abs_index])
            except Exception:
                pass
        if hasattr(self.ds, "image_files"):
            try:
                return self._stem_from_path(self.ds.image_files[abs_index])
            except Exception:
                pass

        return f"idx_{abs_index}"

    @staticmethod
    def _rank_groups(n: int) -> List[Tuple[str, List[int]]]:
        k = min(2, n)
        worst = list(range(0, k))
        best = list(range(n - k, n))
        mid_center = n // 2
        mid = (
            [max(0, mid_center - 1), min(n - 1, mid_center + 1)]
            if n >= 2
            else list(range(n))
        )
        return [("worst", worst), ("average", mid), ("best", best)]

    def _colorize_mask(
        self, mask: np.ndarray, class_colors: Dict[int, Tuple[float, float, float]]
    ) -> np.ndarray:
        h, w = mask.shape
        rgb = np.zeros((h, w, 3), dtype=np.float32)
        for c, col in class_colors.items():
            m = mask == c
            if m.any():
                rgb[m] = np.array(col, dtype=np.float32)
        return rgb

    def _present_classes_str(self, mask: np.ndarray) -> str:
        ids = sorted(int(x) for x in np.unique(mask).tolist() if int(x) > 0)
        names = []
        for cid in ids:
            raw = self.id2name.get(cid, None)
            if self.K == 1 and raw is None:
                en = "foreground"
            else:
                en = self._en_names.get(raw, raw if raw is not None else f"class_{cid}")
            names.append(en)
        return ", ".join(names)

    def _plot_overlay_pair(
        self,
        img: np.ndarray,
        gt: np.ndarray,
        pred: np.ndarray,
        dice: float,
        img_id: str,
        group_name: str,
        class_colors: Dict[int, Tuple[float, float, float]],
        alpha: float,
        save_dir: Optional[str],
        image_name: str,
    ) -> None:
        gt_rgb = self._colorize_mask(gt, class_colors)
        pr_rgb = self._colorize_mask(pred, class_colors)
        img_rgb = np.stack([img, img, img], axis=-1)

        gt_overlay = (1 - alpha) * img_rgb + alpha * gt_rgb
        pr_overlay = (1 - alpha) * img_rgb + alpha * pr_rgb

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, axes = plt.subplots(
            1, 2, figsize=(10.6, 4.4), gridspec_kw={"wspace": 0.01}
        )
        for ax in axes:
            ax.set_facecolor("white")

        axes[0].imshow(gt_overlay)
        axes[0].axis("off")
        axes[0].set_title("Ground truth", fontsize=12, weight="bold", pad=3)

        axes[1].imshow(pr_overlay)
        axes[1].axis("off")
        axes[1].set_title(
            f"Prediction  •  Dice {dice:.3f}", fontsize=12, weight="bold", pad=3
        )

        cls_str = self._present_classes_str(gt)
        supt = f"{group_name.capitalize()}  •  {img_id}"
        supt += f"  •  [{cls_str}]" if cls_str else ""
        supt += "  •  ignore-empty" if self.ignore_empty else "  •  include-empty"

        fig.suptitle(supt, fontsize=12.5, weight="bold", y=1)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.02, wspace=0.01)

        if save_dir:
            out_path = os.path.join(save_dir, f"{image_name}_{group_name}_{img_id}.png")
            plt.savefig(out_path, dpi=160, bbox_inches="tight")
            print(f"Saved: {out_path}")
            plt.close(fig)
        else:
            plt.show()
