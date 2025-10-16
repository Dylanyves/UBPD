import os
import json
import matplotlib.pyplot as plt
import torch
import numpy as np
import random
import torchvision.transforms.functional as TF

from typing import List, Optional, Callable, Tuple
from PIL import Image, ImageDraw
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.lines import Line2D
from torch.utils.data import Dataset

from src.const import Path as P


def _to_rgb_np(t: torch.Tensor) -> np.ndarray:
    """Tensor [C,H,W]/[H,W] → uint8 RGB."""
    x = t.detach().cpu()
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.size(0) == 1:
        x = x.repeat(3, 1, 1)
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    return (x.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()


class UBPDataset(Dataset):
    DEFAULT_CLASS_MAP = {
        "dongmai": 1,  # artery
        "jingmai": 2,  # vein
        "jirouzuzhi": 3,  # muscle
        "shenjing": 4,  # nerve
    }

    def __init__(
        self,
        p_ids: List[int],
        include_classes: Optional[List[int]] = None,
        image_dir: str = P.IMAGE_FOLDER_PATH,
        json_dir: str = P.LABELS_FOLDER_PATH,
        keep_original_indices: bool = True,
        binary: Optional[bool] = None,
        joint_transform: (
            Callable[[Image.Image, Image.Image], Tuple[torch.Tensor, torch.Tensor]]
            | None
        ) = None,
    ):
        """Initialize dataset with mandatory joint_transform for aligned augs."""
        if joint_transform is None:
            raise ValueError(
                "joint_transform must be provided to keep image/mask aligned."
            )
        self.p_ids = set(map(str, p_ids))
        self.include_classes = include_classes
        self.image_dir = image_dir
        self.json_dir = json_dir
        self.keep_original_indices = keep_original_indices
        self.joint_transform = joint_transform

        # class mapping and included ids
        self.class_map = self.DEFAULT_CLASS_MAP.copy()
        self.include_ids = self._resolve_included_classes(include_classes)
        self.binary = (
            bool(binary)
            if binary is not None
            else (self.include_ids is not None and len(self.include_ids) == 1)
        )

        # If binary, we won't use remapping; masks will be {0,1}
        if not self.binary and (
            not self.keep_original_indices and self.include_ids is not None
        ):
            ordered = sorted(self.include_ids)
            self._id_remap = {
                old_id: new_idx for new_idx, old_id in enumerate(ordered, start=1)
            }
        else:
            self._id_remap = None
        self._inv_id_remap = (
            {v: k for k, v in self._id_remap.items()} if self._id_remap else None
        )

        # file list
        self.json_files = sorted(self._collect_json_files())

        # friendly names and colors
        self._en_names = {
            "dongmai": "artery",
            "jingmai": "vein",
            "jirouzuzhi": "muscle",
            "shenjing": "nerve",
        }
        self._colors_hex = {
            "dongmai": "#27ae60",
            "jingmai": "#2980b9",
            "jirouzuzhi": "#f39c12",
            "shenjing": "#e74c3c",
        }

    # ----------------------- helpers -----------------------

    def _resolve_included_classes(self, include_classes):
        """Return a set of class indices to include, or None to include all."""
        if include_classes is None:
            return None
        include_ids = set()
        for c in include_classes:
            if isinstance(c, int) and c in self.class_map.values():
                include_ids.add(c)
            elif isinstance(c, str):
                key = c.strip().lower()
                if key in self.class_map:
                    include_ids.add(self.class_map[key])
                else:
                    aliases = {
                        "artery": "dongmai",
                        "vein": "jingmai",
                        "muscle": "jirouzuzhi",
                        "nerve": "shenjing",
                    }
                    if key in aliases:
                        include_ids.add(self.class_map[aliases[key]])
        return include_ids or None

    def _collect_json_files(self):
        """Return JSON filenames filtered by patient ID."""
        jsons = [f for f in os.listdir(self.json_dir) if f.endswith(".json")]
        if self.p_ids:
            jsons = [f for f in jsons if f.split("_")[0] in self.p_ids]
        return jsons

    def _load_json(self, filename):
        """Load LabelMe JSON robustly."""
        path = os.path.join(self.json_dir, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except UnicodeDecodeError:
            with open(path, "r", encoding="latin1") as f:
                return json.load(f)

    def _label_to_id(self, label_str):
        """Map raw label string → class id."""
        return self.class_map.get((label_str or "").strip().lower(), None)

    def _maybe_remap_id(self, class_id):
        """Optionally remap class IDs to contiguous if requested."""
        if self._id_remap is None:
            return class_id
        return self._id_remap.get(class_id, 0)

    def _create_mask(self, shapes, image_size):
        """Convert LabelMe polygons to integer mask (binary if single class)."""
        mask = Image.new("L", image_size, 0)
        draw = ImageDraw.Draw(mask)
        for shape in shapes:
            class_id = self._label_to_id(shape.get("label", ""))
            if class_id is None:
                continue
            if self.include_ids is not None and class_id not in self.include_ids:
                continue

            if self.binary:
                out_id = 1  # foreground for the single included class
            else:
                out_id = self._maybe_remap_id(class_id)
                if out_id <= 0:
                    continue

            pts = shape.get("points", [])
            # Ensure coordinates are tuples of numbers (no numpy types or strings)
            safe_pts = []
            for p in pts:
                try:
                    x = float(p[0])
                    y = float(p[1])
                    safe_pts.append((x, y))
                except Exception:
                    # skip malformed points
                    continue
            if len(safe_pts) >= 3:
                draw.polygon(safe_pts, outline=out_id, fill=out_id)
        return np.array(mask, dtype=np.int64)

    def _id_to_label_key(self, class_id: int) -> Optional[str]:
        """Map current mask id → canonical key used in class_map."""
        orig_id = (
            class_id
            if self._inv_id_remap is None
            else self._inv_id_remap.get(class_id, None)
        )
        if orig_id is None:
            return None
        for k, v in self.class_map.items():
            if v == orig_id:
                return k
        return None

    # -------------------- visualization --------------------

    def visualize_image(self, idx: int, alpha: float = 0.35, linewidth: float = 2):
        """Visualize raw image with polygons for included classes."""
        json_filename = self.json_files[idx]
        identifier = os.path.splitext(os.path.basename(json_filename))[0]
        image_path = os.path.join(self.image_dir, f"{identifier}.jpg")
        json_path = os.path.join(self.json_dir, f"{identifier}.json")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not os.path.exists(json_path):
            print(f"No JSON found for {identifier}")
            return

        data = self._load_json(json_filename)
        img = Image.open(image_path).convert("RGB")
        W, H = img.size
        fig, ax = plt.subplots(figsize=(5, 5 * H / W))
        ax.imshow(img)
        ax.axis("off")

        if self.include_classes is None:
            allowed_labels = set(self.class_map.keys())
        else:
            inv_map = {v: k for k, v in self.class_map.items()}
            allowed_labels = {inv_map[i] for i in self.include_classes if i in inv_map}

        legend_handles, shapes_drawn = {}, 0
        for shp in data.get("shapes", []):
            raw_label = (shp.get("label", "") or "").strip().lower()
            if raw_label not in allowed_labels:
                continue
            pts = np.array(shp.get("points", []), dtype=float)
            if len(pts) < 3:
                continue
            color = self._colors_hex.get(raw_label, "#7f8c8d")
            poly = MplPolygon(
                pts,
                closed=True,
                fill=True,
                facecolor=color,
                edgecolor=color,
                linewidth=linewidth,
                alpha=alpha,
            )
            ax.add_patch(poly)
            shapes_drawn += 1
            disp = self._en_names.get(raw_label, raw_label)
            if disp not in legend_handles:
                legend_handles[disp] = Line2D([0], [0], color=color, lw=3, label=disp)

        title = f"{identifier} • {W}×{H}px"
        if shapes_drawn == 0:
            title += " • (no polygons for included classes)"
        ax.set_title(title, fontsize=12, weight="bold")
        if legend_handles:
            ax.legend(
                handles=list(legend_handles.values()), loc="lower right", framealpha=0.9
            )
        plt.tight_layout()
        plt.show()

    def visualize_image_transform(self, idx: int, alpha: float = 0.35):
        """Show transformed sample with mask overlays, honoring binary/multiclass settings."""
        image_t, mask_t = self[idx]
        rgb = _to_rgb_np(image_t)
        H, W = mask_t.shape[-2], mask_t.shape[-1]

        # Collect present (non-background) ids
        present_ids = torch.unique(mask_t).detach().cpu().tolist()
        present_ids = [int(c) for c in present_ids if int(c) > 0]

        if not present_ids:
            plt.figure(figsize=(5, 5 * H / max(1, W)))
            plt.imshow(rgb)
            plt.axis("off")
            plt.title(
                f"idx={idx} • transformed • no foreground", fontsize=12, weight="bold"
            )
            plt.tight_layout()
            plt.show()
            return

        overlay = rgb.astype(np.float32) / 255.0
        legend_handles, used_labels = [], set()

        if getattr(self, "binary", False):
            # Binary: mask == 1 corresponds to the single included class
            # Resolve the original class key for legend/color
            orig_cid = None
            if self.include_ids and len(self.include_ids) == 1:
                orig_cid = next(iter(self.include_ids))
            # Find canonical key from class_map
            key = None
            if orig_cid is not None:
                for k, v in self.class_map.items():
                    if v == orig_cid:
                        key = k
                        break
            disp = self._en_names.get(
                key, f"class_{orig_cid if orig_cid is not None else 1}"
            )
            hx = self._colors_hex.get(key, "#7f8c8d")
            color = tuple(int(hx[i : i + 2], 16) / 255.0 for i in (1, 3, 5))

            m = (mask_t.detach().cpu().numpy() == 1).astype(np.float32)
            if m.sum() > 0:
                m3 = np.stack([m, m, m], axis=-1)
                overlay = (
                    overlay * (1 - alpha * m3)
                    + (alpha * m3) * np.array(color)[None, None, :]
                )
                legend_handles.append(Line2D([0], [0], color=color, lw=3, label=disp))
        else:
            # Multiclass: follow (possibly remapped) ids
            for cid in sorted(present_ids):
                key = self._id_to_label_key(cid)
                if key is None:
                    color = (0.5, 0.5, 0.5)
                    disp = f"class_{cid}"
                else:
                    # If include_ids is set, ensure the original id is included
                    if self.include_ids is not None:
                        orig_cid = (
                            cid
                            if self._inv_id_remap is None
                            else self._inv_id_remap.get(cid, None)
                        )
                        if orig_cid is None or orig_cid not in self.include_ids:
                            continue
                    disp = self._en_names.get(key, key)
                    hx = self._colors_hex.get(key, "#7f8c8d")
                    color = tuple(int(hx[i : i + 2], 16) / 255.0 for i in (1, 3, 5))

                m = (mask_t.detach().cpu().numpy() == cid).astype(np.float32)
                if m.sum() == 0:
                    continue
                m3 = np.stack([m, m, m], axis=-1)
                overlay = (
                    overlay * (1 - alpha * m3)
                    + (alpha * m3) * np.array(color)[None, None, :]
                )
                if disp not in used_labels:
                    used_labels.add(disp)
                    legend_handles.append(
                        Line2D([0], [0], color=color, lw=3, label=disp)
                    )

        plt.figure(figsize=(5, 5 * H / max(1, W)))
        plt.imshow((overlay * 255).clip(0, 255).astype(np.uint8))
        plt.axis("off")
        if legend_handles:
            plt.legend(handles=legend_handles, loc="lower right", framealpha=0.9)
        plt.title(f"idx={idx} • transformed • {W}×{H}", fontsize=12, weight="bold")
        plt.tight_layout()
        plt.show()


    def print_stats(self) -> None:
        total = len(self.json_files)
        if total == 0:
            print("No images in dataset.")
            return

        # Which original class IDs (1..4) to report?
        if self.include_ids is None:
            report_ids = sorted(self.class_map.values())  # all
        else:
            report_ids = sorted(self.include_ids)

        # Build helpers: orig_id -> canonical key ('dongmai', ...) and English name ('artery', ...)
        inv_map = {v: k for k, v in self.class_map.items()}
        id_to_key = {cid: inv_map.get(cid, None) for cid in report_ids}
        id_to_en = {cid: self._en_names.get(id_to_key[cid], f"class_{cid}") for cid in report_ids}

        # Counters: count image-level presence per class
        present_counts = {cid: 0 for cid in report_ids}

        for jf in self.json_files:
            data = self._load_json(jf)
            # Collect which *original* class IDs appear in this image
            present_in_image = set()
            for shp in data.get("shapes", []):
                cid = self._label_to_id(shp.get("label", ""))
                if cid is None:
                    continue
                # Respect include filter (if any)
                if self.include_ids is not None and cid not in self.include_ids:
                    continue
                present_in_image.add(cid)

            # Increment once per class if present in this image
            for cid in report_ids:
                if cid in present_in_image:
                    present_counts[cid] += 1

        # Pretty print (aligned)
        label_width = max(len(id_to_en[cid].capitalize()) for cid in report_ids) if report_ids else 0
        count_width = len(str(total))

        for cid in report_ids:
            name = id_to_en[cid].capitalize().ljust(label_width)
            print(f"{name}  present: {present_counts[cid]:>{count_width}}/{total} images")



    # -------------------- torch dataset API --------------------

    def __len__(self):
        """Return dataset length."""
        return len(self.json_files)

    def __getitem__(self, idx):
        """Load image/mask, build mask, and apply joint transform."""
        json_filename = self.json_files[idx]
        data = self._load_json(json_filename)

        base_name = os.path.splitext(json_filename)[0]
        img_path = os.path.join(self.image_dir, f"{base_name}.jpg")
        if not os.path.exists(img_path):
            raise FileNotFoundError(
                f"Image file not found for {json_filename}: {img_path}"
            )

        img = Image.open(img_path).convert("L")
        mask_np = self._create_mask(data.get("shapes", []), img.size)
        mask_img = Image.fromarray(mask_np.astype(np.uint8))

        # enforce paired transform only
        img_t, mask_t = self.joint_transform(img, mask_img)
        # squeeze mask if any stray channel appears
        mask_t = mask_t.squeeze(0) if mask_t.ndim == 3 else mask_t
        return img_t, mask_t
