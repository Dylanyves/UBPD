import argparse

from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader

from src.helper import str2bool, load_model
from src.dataset import UBPDatasetTest
from src.evaluate import Evaluate



def test(variant):
    import glob
    import os
    import re

    TARGET_SIZE = (512, 512)
    include_classes = [1, 2, 3, 4]
    out_channels = len(include_classes) + 1
    image_dir = "./data/dataset/images"
    json_dir = "./data/dataset/labels/json_train"

    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(TARGET_SIZE, antialias=True)
    ])
    mask_transform = transforms.Compose([
        transforms.Resize(TARGET_SIZE, interpolation=InterpolationMode.NEAREST),
        transforms.PILToTensor()
    ])

    test_dataset = UBPDatasetTest(
        image_dir,
        json_dir,
        transform=image_transform,
        target_transform=mask_transform,
        include_classes=include_classes,
    )
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

    model_id = variant["model_id"]
    model_pattern = f"./data/models/{model_id}_*.pth"
    model_paths = sorted(glob.glob(model_pattern))

    if not model_paths:
        print(f"No models found for id {model_id} in ./data/models/")
        return

    mean_dice_scores = []
    for model_path in model_paths:
        model = load_model(model_path)
        evaluator = Evaluate(
            model=model, dataloader=test_loader, device="cuda", num_classes=out_channels
        )
        mean_dice, per_class = evaluator.evaluate_dice_score()
        print(f"\nModel: {os.path.basename(model_path)}")
        print(f"Mean Dice (no background): {mean_dice:.4f}")
        for cid, (name, score) in per_class.items():
            print(f"Class {cid:>2} ({name:<10}): Dice = {score:.4f}")

        # Use fold or best as subdir for images
        match = re.search(rf"{model_id}_(.*)_best\\.pth", os.path.basename(model_path))
        fold_name = match.group(1) if match else "best"
        save_dir = f"./data/imgs/{model_id}_{fold_name}"
        evaluator.visualize_ranked(save_dir=save_dir, alpha=0.45, image_name=f"{model_id}_{fold_name}")
        mean_dice_scores.append((os.path.basename(model_path), mean_dice))

    # Print summary if multiple models
    if len(mean_dice_scores) > 1:
        print("\nSummary of mean dice scores:")
        for fname, score in mean_dice_scores:
            print(f"{fname}: {score:.4f}")
        avg = sum(s for _, s in mean_dice_scores) / len(mean_dice_scores)
        print(f"Average mean dice: {avg:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type", type=str, default="unet"
    )  # only unet is applicable for now
    parser.add_argument("--model_id", type=str)  # only unet is applicable for now

    args = parser.parse_args()

    test(variant=vars(args))
