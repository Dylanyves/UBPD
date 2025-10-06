import os
import json
import torch
import numpy as np
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from PIL import Image, ImageDraw


class BaseUBPDataset(Dataset):
    DEFAULT_CLASS_MAP = {
        "dongmai":    1,  # artery
        "jingmai":    2,  # vein
        "jirouzuzhi": 3,  # muscle
        "shenjing":   4,  # nerve
    }

    def __init__(self,
                 image_dir,
                 json_dir,
                 patient_ids=None,
                 transform=None,
                 target_transform=None,
                 class_map=None,
                 include_classes=None,
                 keep_original_indices=True):
        
        self.image_dir = image_dir
        self.json_dir = json_dir
        self.transform = transform
        self.target_transform = target_transform
        self.class_map = (class_map or self.DEFAULT_CLASS_MAP).copy()
        self.keep_original_indices = keep_original_indices

        # Convert patient IDs to strings for filename matching
        self.patient_ids = {str(pid) for pid in patient_ids} if patient_ids else None

        # Resolve which classes to include
        self.include_ids = self._resolve_included_classes(include_classes)

        # If requested, remap included classes to contiguous indices {1..K}
        if not self.keep_original_indices and self.include_ids is not None:
            # Build a remap dict old_id -> new_id
            ordered = list(self.include_ids) if isinstance(include_classes, (list, tuple)) else sorted(self.include_ids)
            self._id_remap = {old_id: new_idx for new_idx, old_id in enumerate(ordered, start=1)}
        else:
            self._id_remap = None

        # Pre-collect JSON files
        self.json_files = sorted(self._collect_json_files())

    # ------------------------------
    # Utilities
    # ------------------------------
    def _resolve_included_classes(self, include_classes):
        """
        Return a set of class indices to include, or None to include all.
        Supports indices (ints) and/or labels (strings).
        """
        if include_classes is None:
            return None  # include all classes in class_map

        include_ids = set()
        inv_map = {v: k for k, v in self.class_map.items()}

        for c in include_classes:
            if isinstance(c, int):
                if c in inv_map:
                    include_ids.add(c)
            elif isinstance(c, str):
                key = c.strip().lower()
                # accept either raw keys (e.g., 'dongmai') or canonical English if provided by user’s class_map
                if key in self.class_map:
                    include_ids.add(self.class_map[key])
                else:
                    # also accept common English names mapped to defaults, if user passed English
                    aliases = {
                        "artery":    "dongmai",
                        "vein":      "jingmai",
                        "muscle":    "jirouzuzhi",
                        "nerve":     "shenjing",
                    }
                    if key in aliases and aliases[key] in self.class_map:
                        include_ids.add(self.class_map[aliases[key]])
        return include_ids or None

    def _collect_json_files(self):
        """Return JSON filenames filtered by patient ID if provided."""
        jsons = [f for f in os.listdir(self.json_dir) if f.endswith(".json")]
        if self.patient_ids:
            jsons = [f for f in jsons if f.split("_")[0] in self.patient_ids]
        return jsons

    def _load_json(self, filename):
        """Load one LabelMe JSON annotation robustly."""
        path = os.path.join(self.json_dir, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except UnicodeDecodeError:
            with open(path, "r", encoding="latin1") as f:
                return json.load(f)

    def _label_to_id(self, label_str):
        """Map raw label string -> class id using class_map, else None."""
        key = (label_str or "").strip().lower()
        return self.class_map.get(key, None)

    def _maybe_remap_id(self, class_id):
        """Optionally remap class IDs to contiguous if requested."""
        if self._id_remap is None:
            return class_id
        return self._id_remap.get(class_id, 0)  # unmapped -> background

    def _create_mask(self, shapes, image_size):
        """
        Convert LabelMe polygons to integer mask.
        - Includes only classes in self.include_ids (if set)
        - Others become background (0)
        - Optionally remaps to contiguous ids if keep_original_indices=False
        """
        mask = Image.new("L", image_size, 0)
        draw = ImageDraw.Draw(mask)

        for shape in shapes:
            raw_label = shape.get("label", "")
            class_id = self._label_to_id(raw_label)
            if class_id is None:
                continue  # unknown label -> ignore

            # filter by included set (if provided)
            if self.include_ids is not None and class_id not in self.include_ids:
                continue

            out_id = self._maybe_remap_id(class_id)
            if out_id <= 0:
                continue

            draw.polygon(shape["points"], outline=out_id, fill=out_id)

        return np.array(mask, dtype=np.int64)

    # ------------------------------
    # PyTorch Dataset Interface
    # ------------------------------
    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        json_filename = self.json_files[idx]
        data = self._load_json(json_filename)

        # Image path based on JSON stem
        base_name = os.path.splitext(json_filename)[0]
        img_path = os.path.join(self.image_dir, f"{base_name}.jpg")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found for {json_filename}: {img_path}")

        # Load image (grayscale) and mask
        image = Image.open(img_path).convert("L")
        mask = self._create_mask(data.get("shapes", []), image.size)
        mask = Image.fromarray(mask.astype(np.uint8))

        # Transforms
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)  # shape [1,H,W]

        if self.target_transform:
            mask = self.target_transform(mask)
        else:
            mask = torch.from_numpy(np.array(mask)).long()  # shape [H,W], long

        # If any mask transform produced [1,H,W], squeeze channel
        mask = mask.squeeze(0) if mask.ndim == 3 else mask

        return image, mask


# ---- Derived Classes ---------------------------------------------------------
class UBPDatasetTrain(BaseUBPDataset):
    """Training dataset excluding fixed test patients."""
    TEST_PATIENTS = {73, 28, 64, 94, 34, 22, 37, 14, 9, 8, 15, 5, 30, 10, 58}

    def __init__(self, image_dir, json_dir, patient_ids=None, **kwargs):
        if patient_ids is None:
            all_jsons = [f for f in os.listdir(json_dir) if f.endswith(".json")]
            all_patients = {f.split("_")[0] for f in all_jsons}
            patient_ids = sorted(all_patients - {str(pid) for pid in self.TEST_PATIENTS})
        super().__init__(image_dir, json_dir, patient_ids=patient_ids, **kwargs)


class UBPDatasetTest(BaseUBPDataset):
    """Fixed test dataset based on predefined patient IDs."""
    TEST_PATIENTS = {73, 28, 64, 94, 34, 22, 37, 14, 9, 8, 15, 5, 30, 10, 58}

    def __init__(self, image_dir, json_dir, **kwargs):
        super().__init__(image_dir, json_dir, patient_ids=self.TEST_PATIENTS, **kwargs)