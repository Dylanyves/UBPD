import argparse
import os
import torch
import random
import numpy as np

from collections import OrderedDict
from typing import List, Tuple

from src.models.unet import UNet
from src.models.unetpp import UNetPP
from src.const import Path as P


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "yes", "1"):
        return True
    elif v.lower() in ("false", "no", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def load_model(
    model_path: str,
    model_name: str = "unet",
    in_channels: int = 1,
    num_classes: int = 5,
    device: str = "cuda",
):
    """Instantiate model (UNet/UNetPP) using factory and load weights from checkpoint.

    Args:
        model_path: path to checkpoint file
        model_name: 'unet' or 'unetpp'
        in_channels: input channels
        num_classes: number of output classes
        device: map_location for loading and moving model
    Returns:
        model in eval() mode on device
    """
    # Build model via factory so both UNet and UNetPP are supported
    factory = _build_model_factory(model_name)
    model = factory(num_classes=num_classes, in_channels=in_channels)

    ckpt = torch.load(model_path, map_location=device)
    state = ckpt.get("state_dict", ckpt)  # supports raw or wrapped checkpoints

    # Strip 'module.' if saved from DataParallel
    new_state = OrderedDict()
    for k, v in state.items():
        new_k = k.replace("module.", "") if k.startswith("module.") else k
        new_state[new_k] = v

    model.load_state_dict(new_state, strict=True)
    model.to(device).eval()
    return model


def set_seed(seed: int = 42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _build_model_factory(model_name: str, default_in_ch: int = 1, default_bc: int = 32):
    """Return a per-fold factory from a model name."""
    name = (model_name or "unet").strip().lower()
    if name == "unet":

        def factory(
            num_classes: int,
            in_channels: int = default_in_ch,
            vgg_backbone: bool = False,
            base_channels: int = default_bc,
            **_,
        ):
            return UNet(
                in_channels=in_channels,
                num_classes=num_classes,
                base_channels=base_channels,
                vgg_backbone=vgg_backbone,
            )

        return factory
    elif name in ("unetpp", "unet++", "unet_plus_plus"):

        def factory(
            num_classes: int,
            in_channels: int = default_in_ch,
            vgg_backbone: bool = False,
            base_channels: int = default_bc,
            deep_supervision: bool = False,
            **_,
        ):
            return UNetPP(
                in_channels=in_channels,
                num_classes=num_classes,
                base_channels=base_channels,
                deep_supervision=deep_supervision,
                vgg_backbone=vgg_backbone,
            )

        return factory
    else:
        raise ValueError(f"Unknown model name '{model_name}'. Use 'unet' or 'unetpp'.")


def _make_paired_transform(size: int = 400, aug: bool = True):
    """Return a callable joint_transform(image, mask) -> (image_tensor, mask_tensor).
    Uses torchvision transforms for simple resizing and optional flips/rotations.
    """
    import torchvision.transforms.functional as TF
    from PIL import Image

    def joint(img: Image.Image, mask: Image.Image):
        # Resize both (antialias for image; nearest for mask)
        img = img.resize((size, size), resample=Image.BILINEAR)
        mask = mask.resize((size, size), resample=Image.NEAREST)

        if aug:
            # simple random horizontal flip
            if random.random() < 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            # random vertical flip
            if random.random() < 0.2:
                img = TF.vflip(img)
                mask = TF.vflip(mask)

        # to tensor
        img_t = TF.to_tensor(img)
        mask_t = TF.pil_to_tensor(mask).long()
        # mask may be [1,H,W] or [H,W]
        if mask_t.ndim == 3 and mask_t.shape[0] == 1:
            mask_t = mask_t.squeeze(0)
        return img_t, mask_t

    return joint


def get_train_test_pids(
    image_path: str = P.IMAGE_FOLDER_PATH, seed: int = 42
) -> Tuple[List[int], List[int]]:
    p_ids = set()
    for i in os.listdir(image_path):
        if "_" in i:
            p_id = int(i.split("_")[0])
            p_ids.add(p_id)

    p_ids = sorted(list(p_ids))

    # Set seed for reproducibility
    random.seed(seed)
    random.shuffle(p_ids)

    # Split train and test (85:15 ratio)
    split_idx = int(len(p_ids) * 0.85)
    train_pids = p_ids[:split_idx]
    test_pids = p_ids[split_idx:]

    return train_pids, test_pids


def get_cv_pids(
    train_pids: List[int], cv: int = 5, seed: int = 42
) -> List[Tuple[List[int], List[int]]]:
    """Generate n-fold cross-validation splits of patient IDs.

    Accept either a list of patient ids, or the common (train_pids, test_pids)
    tuple returned by `get_train_test_pids`. If a tuple is passed, the first
    element (train ids) will be used.
    """
    random.seed(seed)

    # Accept either a list of ids or a (train_ids, test_ids) tuple
    if (
        isinstance(train_pids, tuple)
        and len(train_pids) >= 1
        and isinstance(train_pids[0], (list, tuple))
    ):
        ids = list(train_pids[0])
    else:
        ids = list(train_pids)

    random.shuffle(ids)

    fold_size = max(1, len(ids) // cv) if len(ids) > 0 else 0
    splits: List[Tuple[List[int], List[int]]] = []

    for i in range(cv):
        if fold_size == 0:
            val_ids = []
        else:
            val_start = i * fold_size
            # make last fold include any remainder
            val_end = (i + 1) * fold_size if i < cv - 1 else len(ids)
            val_ids = ids[val_start:val_end]
        train_ids = [x for j, x in enumerate(ids) if x not in val_ids]
        splits.append((train_ids, val_ids))

    return splits


# Aggregate fold-level results
def aggregate_fold_metrics(fold_results_list):
    """Aggregate a list of evaluator result dicts (one per fold).

    Each fold result is expected to have an "overall" entry with a dict
    containing 'mean', and per-class entries keyed by integer class ids
    mapping to dicts with 'mean'/'std'/... This function returns a tuple
    (overall_mean, overall_std, per_class_stats) where per_class_stats is
    a dict: class_id -> {'mean': m, 'std': s, 'n': count_of_valid_folds}.
    """
    overall_vals = []
    per_class_vals = {}
    for fr in fold_results_list:
        if not isinstance(fr, dict):
            continue
        o = fr.get("overall", {})
        mv = o.get("mean", None)
        overall_vals.append(np.nan if mv is None else float(mv))
        for k, v in fr.items():
            if k == "overall":
                continue
            try:
                cid = int(k)
            except Exception:
                continue
            mean_v = v.get("mean", None) if isinstance(v, dict) else None
            if cid not in per_class_vals:
                per_class_vals[cid] = []
            per_class_vals[cid].append(np.nan if mean_v is None else float(mean_v))

    overall_arr = (
        np.array(overall_vals, dtype=np.float32)
        if overall_vals
        else np.array([], dtype=np.float32)
    )
    overall_mean = float(np.nanmean(overall_arr)) if overall_arr.size else float("nan")
    overall_std = float(np.nanstd(overall_arr)) if overall_arr.size else float("nan")

    per_class_stats = {}
    for cid, vals in per_class_vals.items():
        arr = np.array(vals, dtype=np.float32)
        per_class_stats[cid] = {
            "mean": float(np.nanmean(arr)) if arr.size else float("nan"),
            "std": float(np.nanstd(arr)) if arr.size else float("nan"),
            "n_folds": int(np.sum(~np.isnan(arr))),
        }

    return overall_mean, overall_std, per_class_stats
