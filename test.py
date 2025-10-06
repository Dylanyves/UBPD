import argparse

from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader

from src.helper import str2bool, load_model
from src.dataset import UBPDatasetTest
from src.evaluate import Evaluate

def test(variant):
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
            image_dir, json_dir,
            transform=image_transform,
            target_transform=mask_transform,
            include_classes=include_classes
        )

        model = load_model(variant["model_path"])
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)
        evaluator = Evaluate(
            model=model,
            dataloader=test_loader,
            device="cuda",
            num_classes=out_channels
        )
        mean_dice, per_class = evaluator.evaluate_dice_score()
        print(f"\nMean Dice (no background): {mean_dice:.4f}")
        for cid, (name, score) in per_class.items():
            print(f"Class {cid:>2} ({name:<10}): Dice = {score:.4f}")

        evaluator.visualize_ranked(k=3, save_dir=None, alpha=0.45)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='unet') # only unet is applicable for now
    parser.add_argument('--model_path', type=str) # only unet is applicable for now
   
    args = parser.parse_args()

    test(variant=vars(args))