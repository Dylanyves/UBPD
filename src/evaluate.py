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

        self.test_scores = []

        # dataset → English names & hex palette
        self._en_names: Dict[str, str] = {
            "dongmai": "artery",
            "jingmai": "vein",
            "jirouzuzhi": "muscle",
            "shenjing": "nerve",
        }
        self._colors_hex: Dict[str, str] = {
            "dongmai":   "#e74c3c",  # artery -> red
            "jingmai":   "#2980b9",  # vein -> blue
            "jirouzuzhi":"#f39c12",  # muscle -> orange
            "shenjing":  "#27ae60",  # nerve -> green
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
        """
        Compute per-class Dice (no background), print mean±std, show a clean box plot,
        and store per-image Dice scores (float) into self.test_scores.

        For binary (K == 1): store foreground Dice per image.
        For multiclass (K > 1): store only the Dice score of the 'nerve' class per image.
        """
        class_idxs, class_labels = self._class_indices_and_labels()

        # reset per-image scores
        self.test_scores = []

        # storage for per-class stats across ALL images
        per_class_scores: Dict[int, List[float]] = {c: [] for c in class_idxs}
        pooled_scores: List[float] = []  # pool of all foreground-class dice vals

        # figure out which class id is "nerve" (dataset raw name "shenjing")
        nerve_class_id: Optional[int] = None
        if self.K > 1:
            # self.id2name maps class_id -> raw dataset name ("dongmai","shenjing",...)
            for cid, raw_name in self.id2name.items():
                if raw_name == "shenjing":
                    nerve_class_id = cid
                    break
            # sanity: if not found, leave it as None and we'll just skip storing per-image scores

        for images, masks, *rest in self.loader:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            # forward pass
            logits = self._forward(images)

            # postprocess to one-hot preds and gts for per-class Dice aggregation
            preds_oh = self._postprocess_logits(logits)
            t_oh = (
                masks.float().unsqueeze(1)
                if self.K == 1
                else F.one_hot(masks.long(), num_classes=self.K)
                .permute(0, 3, 1, 2)
                .float()
            )

            # ---- 1) per-class Dice distribution (for stats/plots) ----
            # we'll optionally grab the nerve-only dice here too
            nerve_batch_dice = None  # [B] or None

            for c in class_idxs:
                # select the correct channel (binary model always channel 0)
                pc = preds_oh[:, (0 if self.K == 1 else c), ...]  # [B,H,W]
                tc = t_oh[:, (0 if self.K == 1 else c), ...]      # [B,H,W]

                inter = (pc * tc).flatten(1).sum(dim=1)  # [B]
                denom = pc.flatten(1).sum(dim=1) + tc.flatten(1).sum(dim=1)  # [B]
                dice_ci = (2 * inter + 1e-6) / (denom + 1e-6)   # [B]

                # record this class's dice values for global stats
                vals = (
                    (dice_ci[denom > 0] if self.ignore_empty else dice_ci)
                    .detach()
                    .cpu()
                    .tolist()
                )
                per_class_scores[c].extend(vals)
                pooled_scores.extend(vals)

                # capture per-image nerve dice if this is the nerve class
                if self.K > 1 and nerve_class_id is not None and c == nerve_class_id:
                    # respect ignore_empty for nerve specifically
                    if self.ignore_empty:
                        nerve_batch_dice = torch.where(
                            denom > 0,
                            dice_ci,
                            torch.tensor(float("nan"), device=dice_ci.device),
                        )
                    else:
                        nerve_batch_dice = dice_ci

                # capture per-image foreground dice if binary
                if self.K == 1:
                    # binary, only foreground class. We'll store this later.
                    if self.ignore_empty:
                        nerve_batch_dice = torch.where(
                            denom > 0,
                            dice_ci,
                            torch.tensor(float("nan"), device=dice_ci.device),
                        )
                    else:
                        nerve_batch_dice = dice_ci

            # ---- 2) store per-image dice in self.test_scores ----
            # binary: nerve_batch_dice is actually foreground dice
            # multiclass: nerve_batch_dice is nerve dice (or None if nerve wasn't found)
            if nerve_batch_dice is not None:
                self.test_scores.extend(nerve_batch_dice.detach().cpu().tolist())
            else:
                # multiclass but no nerve class found in this dataset:
                # we won't append anything; length may not match dataset size.
                pass

        # ---- 3) summary stats (mean, std, etc.) ----
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

        # ---- 4) console summary ----
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
                f"  {lbl:<20s} mean±std: {s['mean']:.4f} ± {s['std']:.4f}  "
                f"(median={s['median']:.4f}, n={s['n']})"
            )

        # print a short summary of what we stored
        target_label = "nerve" if self.K > 1 else "foreground"
        print(
            f"\n🧪 Stored per-image Dice scores for '{target_label}': {len(self.test_scores)} images"
        )
        if len(self.test_scores) > 0:
            ts_arr = np.asarray(self.test_scores, dtype=np.float32)
            print(
                f"  {target_label} mean±std: "
                f"{float(np.nanmean(ts_arr)):.4f} ± {float(np.nanstd(ts_arr)):.4f}"
            )

        # ---- 5) plot ----
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

    @torch.no_grad()
    def visualize_single(
        self,
        index: int,
        alpha: float = 0.45,
    ) -> None:
        """
        Visualize ground truth vs prediction for a single sample from the dataset
        at the given global index.

        - Shows GT overlay and Pred overlay side by side.
        - suptitle includes <image_id> and overall Dice score.
        - Prints per-class Dice scores (excluding background).
        """
        # ----- 1. get the sample from dataset -----
        sample = self.ds[index]  # UBPDataset __getitem__ expected to return (img, mask, meta?)
        if isinstance(sample, (list, tuple)) and len(sample) >= 2:
            img_t, mask_t = sample[:2]
            meta = sample[2] if len(sample) >= 3 else None
        else:
            img_t, mask_t, meta = sample, None, None  # defensive

        # keep CPU copies for overlay later
        img_cpu = img_t.clone()
        mask_cpu = mask_t.clone()

        # move to device & add batch dim
        img_b = img_t.unsqueeze(0).to(self.device)
        mask_b = mask_t.unsqueeze(0).long().to(self.device)

        # ----- 2. forward pass -----
        with autocast(
            device_type=("cuda" if self.device.type == "cuda" else "cpu"),
            dtype=torch.float16,
            enabled=self._autocast_enabled(),
        ):
            logits_b = self.model(img_b)

        # predicted mask [H,W]
        if self.K == 1:
            preds_b = (torch.sigmoid(logits_b) > 0.5).long().squeeze(1)
        else:
            preds_b = logits_b.argmax(dim=1)

        pred_mask = preds_b[0].detach().cpu().numpy().astype(np.int32)
        gt_mask = mask_cpu.detach().cpu().numpy().astype(np.int32)

        # ----- 3. compute Dice for this sample -----
        # overall dice + per-class dice map
        per_class_dict = {}  # str -> float

        if self.K == 1:
            # binary dice for that one element
            probs = torch.sigmoid(logits_b[0:1])                 # [1,1,H,W]
            pred_bin = (probs > 0.5).long().squeeze(1)           # [1,H,W]
            tgt_bin = mask_b[0:1].long()                         # [1,H,W]

            inter = (pred_bin * tgt_bin).flatten(1).sum(dim=1).float()  # [1]
            denom = pred_bin.flatten(1).sum(dim=1).float() + tgt_bin.flatten(1).sum(dim=1).float()
            dice_val = (2 * inter + 1e-6) / (denom + 1e-6)             # [1]

            # handle ignore_empty for overall
            if self.ignore_empty:
                dice_val = torch.where(
                    denom > 0,
                    dice_val,
                    torch.tensor(float("nan"), device=dice_val.device),
                )

            dice_score = float(dice_val[0].item())

            # per-class dict: only one FG class called "foreground"
            per_class_dict["foreground"] = dice_score

        else:
            # multiclass dice excluding background
            C = self.K
            pred_1h = F.one_hot(preds_b[0], num_classes=C).permute(2, 0, 1).float()  # [C,H,W]
            tgt_1h = F.one_hot(mask_b[0], num_classes=C).permute(2, 0, 1).float()    # [C,H,W]

            pred_fg = pred_1h[1:, ...]  # [C-1,H,W] (skip bg=0)
            tgt_fg = tgt_1h[1:, ...]    # [C-1,H,W]

            inter = (pred_fg * tgt_fg).sum(dim=(1, 2))  # [C-1]
            denom = pred_fg.sum(dim=(1, 2)) + tgt_fg.sum(dim=(1, 2))  # [C-1]

            dice_c = (2 * inter + 1e-6) / (denom + 1e-6)  # [C-1]

            if self.ignore_empty:
                dice_c_masked = torch.where(
                    denom > 0,
                    dice_c,
                    torch.tensor(float("nan"), device=dice_c.device),
                )
                dice_val = torch.nanmean(dice_c_masked)
            else:
                dice_val = dice_c.mean()

            dice_score = float(dice_val.item())

            # build per_class_dict for each foreground class id c=1..C-1
            class_idxs, class_labels = self._class_indices_and_labels()
            # class_idxs aligns with ids 1..C-1, and class_labels are english names in same order
            for local_idx, c in enumerate(class_idxs):
                # dice_c[local_idx] corresponds to class c
                d_val = dice_c[local_idx].item()
                if self.ignore_empty and denom[local_idx].item() <= 0:
                    d_val = float("nan")

                per_class_dict[class_labels[local_idx]] = float(d_val)

        # ----- 4. get human-readable ID for this sample -----
        true_id = self._extract_true_id(
            abs_index=index,
            meta=[meta] if meta is not None else None,
            i_in_batch=0,
        )

        # ----- 5. prep overlays for plotting -----
        class_colors = self._class_colors_rgb()

        img_np = self._to_numpy_img(img_cpu)
        gt_rgb = self._colorize_mask(gt_mask, class_colors)
        pr_rgb = self._colorize_mask(pred_mask, class_colors)
        img_rgb = np.stack([img_np, img_np, img_np], axis=-1)

        gt_overlay = (1 - alpha) * img_rgb + alpha * gt_rgb
        pr_overlay = (1 - alpha) * img_rgb + alpha * pr_rgb

        # ----- 6. print dice scores to console -----
        print(f"\n📌 Sample index {index}  |  ID: {true_id}")
        print(f"Overall Dice: {dice_score:.4f}")
        print("Per-class Dice (no background):")
        for cls_name, d_val in per_class_dict.items():
            if np.isnan(d_val):
                print(f"  {cls_name:<15s}: nan (class absent / ignored)")
            else:
                print(f"  {cls_name:<15s}: {d_val:.4f}")

        # ----- 7. plot -----
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
        axes[1].set_title("Prediction", fontsize=12, weight="bold", pad=3)

        # optional classes present string for context
        cls_str = self._present_classes_str(gt_mask)

        supt = f"{true_id} • Dice {dice_score:.3f}"
        if cls_str:
            supt += f" • [{cls_str}]"
        supt += " • ignore-empty" if self.ignore_empty else " • include-empty"

        fig.suptitle(supt, fontsize=12.5, weight="bold", y=1)
        fig.subplots_adjust(
            left=0.01, right=0.99, top=0.92, bottom=0.02, wspace=0.01
        )

        plt.show()
