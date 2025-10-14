import argparse

import torch
import numpy as np
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader

import glob
import os
import re
from typing import List

from src.helper import str2bool, load_model, _build_model_factory
from src.dataset import UBPDatasetTest
from src.evaluate import Evaluator


def _find_model_paths(
    model_id: str, cv: bool = False, checkpoints_dir: str = "checkpoints"
) -> List[str]:
    """Return sorted list of checkpoint paths matching model_id.

    - If cv=False, look for `{model_id}.pth` in checkpoints_dir.
    - If cv=True, look for `{model_id}_fold_*.pth` in checkpoints_dir.
    """
    if cv:
        pattern = os.path.join(checkpoints_dir, f"{model_id}_fold_*.pth")
    else:
        pattern = os.path.join(checkpoints_dir, f"{model_id}.pth")
    paths = sorted(glob.glob(pattern))
    return paths


def test(variant):
    TARGET_SIZE = (512, 512)
    include_classes = [1, 2, 3, 4]
    out_channels = len(include_classes) + 1
    image_dir = "./data/dataset/images"
    json_dir = "./data/dataset/labels/json_train"

    image_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize(TARGET_SIZE, antialias=True)]
    )
    mask_transform = transforms.Compose(
        [
            transforms.Resize(TARGET_SIZE, interpolation=InterpolationMode.NEAREST),
            transforms.PILToTensor(),
        ]
    )

    test_dataset = UBPDatasetTest(
        image_dir,
        json_dir,
        transform=image_transform,
        target_transform=mask_transform,
        include_classes=include_classes,
    )
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

    model_id = variant["model_id"]
    cv_mode = bool(variant.get("cv", False))

    # Search checkpoints directory according to cv flag
    model_paths = _find_model_paths(model_id, cv=cv_mode, checkpoints_dir="checkpoints")
    if not model_paths:
        print(f"No models found for id {model_id} in ./checkpoints/")
        return

    mean_dice_scores = []
    per_fold_results = []
    for model_path in model_paths:
        # Decide model type by file naming or CLI (assume variant may include model_type)
        model_type = variant.get("model_type", "unet")
        model = load_model(
            model_path,
            model_name=model_type,
            in_channels=1,
            num_classes=out_channels,
            device=("cuda" if torch.cuda.is_available() else "cpu"),
        )
        evaluator = Evaluator(
            model=model,
            test_dataset=test_dataset,
            num_classes=out_channels,
            device=("cuda" if torch.cuda.is_available() else "cpu"),
        )
        res = evaluator.evaluate_dice_score(show_plot=True)

        print(f"\nModel: {os.path.basename(model_path)}")
        # Print overall
        overall = res.get("overall", {})
        print(
            f"Overall mean Dice (foreground): {overall.get('mean', float('nan')):.4f}"
        )

        # Print per-class
        for cid, stats in sorted((k, v) for k, v in res.items() if k != "overall"):
            if isinstance(stats, dict):
                print(
                    f"Class {cid:>2}: mean={stats.get('mean', float('nan')):.4f} std={stats.get('std', float('nan')):.4f} n={stats.get('n', 0)}"
                )

        per_fold_results.append(res)
        mean_dice_scores.append(
            (os.path.basename(model_path), overall.get("mean", float("nan")))
        )

    # Aggregate and print summary when multiple models (folds)
    if per_fold_results:

        def _aggregate_fold_metrics_local(fold_results_list):
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
                    per_class_vals.setdefault(cid, []).append(
                        np.nan if mean_v is None else float(mean_v)
                    )

            overall_arr = (
                np.array(overall_vals, dtype=np.float32)
                if overall_vals
                else np.array([], dtype=np.float32)
            )
            overall_mean = (
                float(np.nanmean(overall_arr)) if overall_arr.size else float("nan")
            )
            overall_std = (
                float(np.nanstd(overall_arr)) if overall_arr.size else float("nan")
            )

            per_class_stats = {}
            for cid, vals in per_class_vals.items():
                arr = np.array(vals, dtype=np.float32)
                per_class_stats[cid] = {
                    "mean": float(np.nanmean(arr)) if arr.size else float("nan"),
                    "std": float(np.nanstd(arr)) if arr.size else float("nan"),
                    "n_folds": int(np.sum(~np.isnan(arr))),
                }
            return overall_mean, overall_std, per_class_stats

        overall_mean, overall_std, per_class_stats = _aggregate_fold_metrics_local(
            per_fold_results
        )
        print("\nAggregated across models/folds:")
        print(f"  Overall mean: {overall_mean:.4f} ± {overall_std:.4f}")
        if per_class_stats:
            print("  Per-class:")
            for cid, s in per_class_stats.items():
                print(
                    f"    {cid}: mean={s['mean']:.4f} ± {s['std']:.4f} (folds={s['n_folds']})"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type", type=str, default="unet"
    )  # only unet is applicable for now
    parser.add_argument("--cv", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--model_id", type=str)

    args = parser.parse_args()

    test(variant=vars(args))
