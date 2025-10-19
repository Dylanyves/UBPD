import argparse
import random
import numpy as np
import torch
import wandb
import os

from src.helper import (
    set_seed,
    _build_model_factory,
    str2bool,
    _make_paired_transform,
    get_train_test_pids,
    get_cv_pids,
    aggregate_fold_metrics,
)
from src.dataset import UBPDataset
from src.const import Path as P, Train as T
from src.train import Trainer
from src.evaluate import Evaluator


def experiment(variants):
    """Run CV training with model name (unet|unetpp) and per-fold initialization inside."""

    exp_id = random.randint(int(1e5), int(1e6) - 1)
    seed = variants["seed"]
    model_name = variants["model"]

    set_seed(seed)
    print(f"\n🚀 Beginning experiment #{exp_id}")
    print("=" * 60)

    include_classes = variants.get("include_classes", T.INCLUDE_CLASSES)
    class_names = {
        1: "dongmai (artery)",
        2: "jingmai (vein)",
        3: "jirouzuzhi (muscle)",
        4: "shenjing (nerve)",
    }

    # --- Loss & num_classes (FIXED) ---
    n_inc = len(include_classes)
    if n_inc == 1:
        variants["loss"] = "bce"
        num_classes_for_model = 1  # single foreground channel
        keep_original_indices = True  # irrelevant when binary
    else:
        variants["loss"] = "ce"
        num_classes_for_model = n_inc + 1  # background + selected classes only
        keep_original_indices = False  # remap selected IDs to contiguous {0..K}

    variants["num_classes"] = num_classes_for_model

    # image transforms
    image_size = variants.get("image_size", T.IMG_SIZE)
    paired_train_tf = _make_paired_transform(size=image_size, aug=True)
    paired_val_tf = _make_paired_transform(size=image_size, aug=False)

    print("Included classes:")
    for cid in include_classes:
        print(f"  {cid}: {class_names.get(cid, 'unknown')}")
    print(f"- Model: {model_name}")
    print(
        f"- Using loss='{variants['loss']}' with model num_classes={num_classes_for_model}"
    )
    if not keep_original_indices:
        print(
            "- Remapping labels to contiguous IDs: background=0, selected classes=1..K"
        )
    print(f"- Image size: {image_size}")
    print("-" * 60)

    # Train/Test split
    train_pids_1, test_pids_1 = get_train_test_pids(seed=seed)
    print("Train/Test split:")
    print(f"  Train patient IDs: {train_pids_1}")
    print(f"  Test  patient IDs: {test_pids_1}")
    print("-" * 60)

    # Test dataset (apply same remap policy for consistency)
    test_dataset = UBPDataset(
        p_ids=test_pids_1,
        include_classes=include_classes,
        joint_transform=paired_val_tf,
        image_dir=P.IMAGE_FOLDER_PATH,
        json_dir=P.LABELS_FOLDER_PATH,
        keep_original_indices=keep_original_indices,  # <--- important
    )
    print(f"Test dataset contains {len(test_dataset)} images")
    print("=" * 60)

    # CV
    cv_folds_pids = get_cv_pids(train_pids_1, cv=variants["cv"], seed=seed)
    all_histories = []
    fold_overall_means = []
    fold_results = []  # store per-fold evaluator outputs (dicts)

    for fold, (train_ids, val_ids) in enumerate(cv_folds_pids, start=1):
        if variants["use_wandb"]:
            name = f"{exp_id}_fold_{fold}"
            api_key = os.getenv("WANDB_API_KEY")
            wandb.login(key=api_key)
            wandb.init(
                project="ubpd",
                group=exp_id,
                name=name,
                config=variants,
                reinit=True,
            )

        print(f"\n📂 Fold {fold}/{len(cv_folds_pids)}")
        print(f"  Train patient IDs: {train_ids}")
        print(f"  Val   patient IDs: {val_ids}")

        train_dataset = UBPDataset(
            p_ids=train_ids,
            include_classes=include_classes,
            joint_transform=paired_train_tf,
            image_dir=P.IMAGE_FOLDER_PATH,
            json_dir=P.LABELS_FOLDER_PATH,
            keep_original_indices=keep_original_indices,  # <--- important
        )
        val_dataset = UBPDataset(
            p_ids=val_ids,
            include_classes=include_classes,
            joint_transform=paired_val_tf,
            image_dir=P.IMAGE_FOLDER_PATH,
            json_dir=P.LABELS_FOLDER_PATH,
            keep_original_indices=keep_original_indices,  # <--- important
        )
        print(
            f"  ➜ Train images: {len(train_dataset)} | Val images: {len(val_dataset)}\n"
        )

        # fresh model per fold
        make_model = _build_model_factory(model_name)
        model = make_model(num_classes=num_classes_for_model)

        trainer = Trainer(
            exp_id=exp_id,
            fold_num=fold,
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            arguments=variants,
        )
        history = trainer.train()
        all_histories.append(history)

        if history["val_loss"]:
            best_idx = int(np.argmin(history["val_loss"]))
            print(
                f"  ✅ Best @ epoch {best_idx+1}: val_loss={history['val_loss'][best_idx]:.4f} | val_dice={history['val_dice'][best_idx]:.4f}"
            )
        print("-" * 60)

        # Evaluate one fold (you can move this after the loop to evaluate the final/best model instead)
        evaluator = Evaluator(
            trainer.model,
            test_dataset,
            num_classes=num_classes_for_model,
            ignore_empty_classes=False,
        )
        res = evaluator.evaluate_dice_score(show_plot=True)
        # collect overall mean dice for this fold if available
        try:
            overall_mean = res.get("overall", {}).get("mean", float("nan"))
        except Exception:
            overall_mean = float("nan")
        fold_overall_means.append(
            float(overall_mean) if overall_mean is not None else float("nan")
        )
        fold_results.append(res)

    overall_mean, overall_std, per_class_stats = aggregate_fold_metrics(fold_results)
    if not np.isnan(overall_mean):
        print(
            f"\n🎯 Average overall Dice across folds: {overall_mean:.4f} ± {overall_std:.4f}  (n={len(fold_results)})"
        )
    else:
        print("\n⚠️ No per-fold overall Dice scores collected.")

    # Print per-landmark (per-class) averages
    if per_class_stats:
        print("\n📌 Per-landmark average Dice across folds:")
        for cid in sorted(per_class_stats.keys()):
            stats = per_class_stats[cid]
            name = class_names.get(cid, f"class_{cid}")
            print(
                f"  {cid}: {name:<20s} mean±std: {stats['mean']:.4f} ± {stats['std']:.4f}  (folds={stats['n_folds']})"
            )
    else:
        print("\n⚠️ No per-class stats available to aggregate.")

    print("\n✅ Experiment complete across folds.")

    if variants["use_wandb"]:
        wandb.finish()

    return {
        "histories": all_histories,
        "test_dataset": test_dataset,
        "fold_overall_means": fold_overall_means,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run UBPD cross-validation experiments"
    )
    parser.add_argument(
        "--model", type=str, default="unet", help="Model name: unet or unetpp"
    )
    parser.add_argument("--cv", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--include_classes",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help="List of fold indices, e.g. --folds 0 1 3",
    )

    # Trainer args (common)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument(
        "--half_precision", type=str2bool, nargs="?", const=True, default=True
    )
    parser.add_argument(
        "--ignore_empty", type=str2bool, nargs="?", const=True, default=False
    )
    parser.add_argument("--augment", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument(
        "--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu")
    )
    parser.add_argument(
        "--use_wandb", "-wandb", type=str2bool, nargs="?", const=True, default=False
    )
    args = parser.parse_args()

    print(f"Running experiment with model={args.model} cv={args.cv} seed={args.seed}")
    experiment(variants=vars(args))
