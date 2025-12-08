"""
Evaluation CLI for UBPD models.

Finds checkpoints in ./checkpoints and evaluates them on the standard test split
that main.py uses (same include_classes/remap policy and transforms).

- CV mode: loads all per-fold checkpoints named <model_id>_fold_<n>.pth
           OR (fallback) any *.pth inside ./checkpoints/<model_id>/.
- Single mode: loads <model_id>.pth
               OR (fallback) a single *.pth inside ./checkpoints/<model_id>/.

Examples:
  uv run evaluate.py --model_id 616483 --model_name unet --cv --include_classes 1 2 3 4
  uv run evaluate.py --model_id 616483 --model_name unetpp --cv --include_classes 2 4
  uv run evaluate.py --model_id 616483 --include_classes 4 --show_plot
"""

from typing import List, Tuple, Dict, Optional
import argparse
import glob
import os

import torch
import numpy as np

from src.const import Path as P
from src.helper import (
    load_model,
    aggregate_fold_metrics,
    _make_paired_transform,
)
from src.dataset import UBPDataset
from src.evaluate import Evaluator


# ------------------------------- Checkpoint discovery -------------------------------


def _top_level_patterns(model_id: str, cv: bool) -> List[str]:
    """Strict patterns in ./checkpoints (top-level files)."""
    ckpt_dir = os.path.abspath("checkpoints")
    if cv:
        pattern = os.path.join(ckpt_dir, f"{model_id}_fold_*.pth")
    else:
        pattern = os.path.join(ckpt_dir, f"{model_id}.pth")
    return sorted(glob.glob(pattern))


def _subdir_patterns(model_id: str, cv: bool) -> List[str]:
    """
    Fallback patterns in ./checkpoints/<model_id> subdirectory.
    - CV mode: any *.pth inside the folder (commonly fold_*.pth or best_fold*.pth)
    - Single mode: expect exactly one *.pth inside that folder
    """
    subdir = os.path.abspath(os.path.join("checkpoints", model_id))
    if not os.path.isdir(subdir):
        return []
    matches = sorted(glob.glob(os.path.join(subdir, "*.pth")))
    if not matches:
        return []
    if cv:
        return matches
    # single: require exactly one file to avoid ambiguity
    return matches if len(matches) == 1 else []


def _find_model_paths(model_id: str, cv: bool) -> List[str]:
    """
    Return list of checkpoint file paths using:
      1) strict top-level patterns, else
      2) subfolder fallback ./checkpoints/<model_id>/...
    """
    matches = _top_level_patterns(model_id, cv=cv)
    if matches:
        return matches
    return _subdir_patterns(model_id, cv=cv)


# --------------------------- Class remap & dataset builders --------------------------


def _resolve_num_classes_and_remap(include_classes: List[int]) -> Tuple[int, bool]:
    """
    Mirror main.py:
      - if only 1 included class => binary channel (num_classes_for_model = 1), keep_original_indices=True
      - else => background + K selected classes (num_classes_for_model = K+1), keep_original_indices=False
    """
    n_inc = len(include_classes)
    if n_inc == 1:
        return 1, True
    else:
        return n_inc + 1, False


def _build_test_dataset(
    include_classes: List[int],
    image_size: int,
    keep_original_indices: bool,
    seed: int,
    test_pids: List[int],
) -> UBPDataset:
    """
    Build the test dataset exactly like in main.py:
      - get_train_test_pids(seed)
      - joint_transform = _make_paired_transform(size=image_size, aug=False)
      - pass include_classes and keep_original_indices
    """
    # _, test_pids = get_train_test_pids(seed=seed)
    paired_val_tf = _make_paired_transform(size=image_size, aug=False)

    test_dataset = UBPDataset(
        p_ids=test_pids,
        include_classes=include_classes,
        joint_transform=paired_val_tf,
        image_dir=P.IMAGE_FOLDER_PATH,
        json_dir=P.LABELS_FOLDER_PATH,
        keep_original_indices=keep_original_indices,
    )
    return test_dataset


def _make_name_mappings(
    include_classes: List[int], keep_original_indices: bool
) -> Tuple[Dict[int, str], Dict[int, str]]:
    """
    Return:
      - id_to_raw: remapped CID (1..K) -> raw dataset key ('dongmai', 'jingmai', ...)
      - id_to_en:  remapped CID (1..K) -> english name ('artery', 'vein', ...)

    These are used to override Evaluator.id2name so labels & plots are correct.
    """
    RAW_NAMES = {1: "dongmai", 2: "jingmai", 3: "jirouzuzhi", 4: "shenjing"}
    EN_NAMES = {1: "artery", 2: "vein", 3: "muscle", 4: "nerve"}

    if keep_original_indices:
        # binary: single foreground channel is cid=1
        cid_to_raw = {1: RAW_NAMES[include_classes[0]]}
        cid_to_en = {1: EN_NAMES[include_classes[0]]}
    else:
        # multi-class remap: new ids 1..K follow the order of include_classes
        cid_to_raw = {i + 1: RAW_NAMES[c] for i, c in enumerate(include_classes)}
        cid_to_en = {i + 1: EN_NAMES[c] for i, c in enumerate(include_classes)}

    return cid_to_raw, cid_to_en


# --------------------------------- Evaluation core ----------------------------------


def evaluate_model_files(
    model_paths: List[str],
    model_name: str,
    device: torch.device,
    include_classes: List[int],
    image_size: int,
    seed: int,
    ignore_empty_classes: bool,
    test_pids: List[int],
    show_plot: bool = False,
) -> Tuple[List[dict], Dict[int, str]]:
    """Evaluate each model in model_paths and return (per-model results, id->english-name)."""
    if not model_paths:
        raise FileNotFoundError(
            "No model checkpoints found for the given model_id and cv setting."
        )

    # num_classes + remap policy same as main.py
    num_classes_for_model, keep_original_indices = _resolve_num_classes_and_remap(
        include_classes
    )

    # Build test dataset
    test_dataset = _build_test_dataset(
        include_classes=include_classes,
        image_size=image_size,
        keep_original_indices=keep_original_indices,
        seed=seed,
        test_pids=test_pids,
    )

    print(f"Number of images in test set: {len(test_dataset)}")

    # Prepare name maps (we'll override evaluator.id2name to fix display labels)
    id2raw, id2en = _make_name_mappings(include_classes, keep_original_indices)

    print("\n=== Evaluation Config ===")
    print(
        f"Included classes      : {include_classes}  (1=artery, 2=vein, 3=muscle, 4=nerve)"
    )
    if keep_original_indices:
        print("Remap policy          : binary (no remap, single foreground channel)")
    else:
        print(
            "Remap policy          : remap to contiguous IDs (0=bg, 1..K=selected in given order)"
        )
    print(f"Model architecture    : {model_name}")
    print(f"Model num_classes     : {num_classes_for_model}")
    print(f"Test images           : {len(test_dataset)}")
    print(f"Ignore empty classes? : {ignore_empty_classes}")
    print("=" * 60)

    evaluator_results = []
    for mp in model_paths:
        print(f"\n🔹 Loading model checkpoint: {mp}")
        model = load_model(
            mp,
            model_name=model_name,
            in_channels=1,
            num_classes=num_classes_for_model,
            device=device,
        )

        evaluator = Evaluator(
            model=model,
            test_dataset=test_dataset,
            num_classes=num_classes_for_model,
            device=device,
            ignore_empty_classes=ignore_empty_classes,
        )

        # --- IMPORTANT: override class mapping so Evaluator prints the correct labels ---
        # Evaluator uses self.id2name (raw keys) → english via its internal _en_names.
        evaluator.id2name = dict(id2raw)  # cid -> raw dataset key (e.g., 'jingmai')

        print(f"▶ Evaluating {os.path.basename(mp)} ...")
        res = evaluator.evaluate_dice_score(show_plot=show_plot)
        overall = res.get("overall", {})
        mean = overall.get("mean", float("nan"))
        std = overall.get("std", float("nan"))
        print(
            f"Result for {os.path.basename(mp)}: overall mean={mean:.4f}, std={std:.4f}"
        )
        evaluator_results.append(res)

    return evaluator_results, id2en


def aggregate_and_print(
    evaluator_results: List[dict], id_to_en: Optional[Dict[int, str]] = None
):
    overall_mean, overall_std, per_class = aggregate_fold_metrics(evaluator_results)

    print("\n=== Aggregated Results ===")
    if not np.isnan(overall_mean):
        print(f"Overall Mean Dice: {overall_mean:.4f}")
        print(f"Overall Std Dice : {overall_std:.4f}")
    else:
        print("Overall Mean Dice: NaN (no valid scores found)")
        print("Overall Std Dice : NaN")

    if per_class:
        print("Per-class Dice:")
        for cls, stats in sorted(per_class.items(), key=lambda x: int(x[0])):
            # pretty label like "1 (vein)"
            label = f"{cls}"
            if id_to_en and int(cls) in id_to_en:
                label = f"{cls} ({id_to_en[int(cls)]})"
            print(
                f"  Class {label}: mean={stats['mean']:.4f} std={stats.get('std', 0):.4f}"
            )
    else:
        print("Per-class Dice: (no classwise stats available)")


# --------------------------------------- CLI ----------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Evaluate UBPD models")
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model identifier. Looks for <id>_fold_*.pth or <id>.pth in ./checkpoints, "
        "or falls back to ./checkpoints/<id>/*.pth.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="unet",
        help="Model architecture name: unet or unetpp",
    )
    parser.add_argument(
        "--cv",
        action="store_true",
        help="If set, evaluate all matching per-fold checkpoints.",
    )
    parser.add_argument(
        "--include_classes",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help="Which landmark classes to include (IDs: 1=artery, 2=vein, 3=muscle, 4=nerve).",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="Image size, must match training/eval pipeline.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used to reproduce the same Train/Test split as in main.py.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda" if torch.cuda.is_available() else "cpu"),
        help="Device: 'cuda' or 'cpu'",
    )
    parser.add_argument(
        "--u1",
        action="store_true",
        help="Ignore empty (all-background) images for class metrics.",
    )
    parser.add_argument(
        "--u2",
        action="store_true",
        help="Ignore empty (all-background) images for class metrics.",
    )
    parser.add_argument(
        "--ignore_empty",
        action="store_true",
        help="Ignore empty (all-background) images for class metrics.",
    )
    parser.add_argument(
        "--show_plot",
        action="store_true",
        help="Show example prediction plots during evaluation.",
    )
    args = parser.parse_args()

    # Locate checkpoints
    model_paths = _find_model_paths(args.model_id, args.cv)
    if not model_paths:
        raise SystemExit(
            f"No checkpoints found for model_id={args.model_id} with cv={args.cv}.\n"
            f"Tried:\n"
            f"  - ./checkpoints/{args.model_id}{'_fold_*' if args.cv else ''}.pth\n"
            f"  - ./checkpoints/{args.model_id}/*.pth (fallback)"
        )

    # In single mode, ensure exactly one checkpoint was found (after all patterns)
    if not args.cv and len(model_paths) != 1:
        raise SystemExit(
            f"Single-mode evaluation expects exactly one checkpoint, but found {len(model_paths)}:\n"
            + "\n".join(model_paths)
        )

    device = torch.device(args.device)

    if args.u1:
        test_pids = [16, 19, 12, 28, 40]
    if args.u2:
        test_pids = [79, 77, 71, 63, 51, 1, 81, 61]

    evaluator_results, id_to_en = evaluate_model_files(
        model_paths=model_paths,
        model_name=args.model_name,
        device=device,
        include_classes=args.include_classes,
        image_size=args.image_size,
        seed=args.seed,
        ignore_empty_classes=args.ignore_empty,
        show_plot=args.show_plot,
        test_pids=test_pids,
    )

    # Aggregate and print CV-style summary (single mode will just print the same)
    aggregate_and_print(evaluator_results, id_to_en=id_to_en)


if __name__ == "__main__":
    main()
