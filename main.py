import argparse
import random

from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader

from src.helper import str2bool
from src.dataset import UBPDatasetTrain, UBPDatasetTest
from src.models.unet import UNet
from src.trainer import get_trainer
from src.evaluate import Evaluate


# DatasetManager class to share variables and logic
class DatasetManager:
    def __init__(self):
        self.TARGET_SIZE = (512, 512)
        self.image_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize(self.TARGET_SIZE, antialias=True)]
        )
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize(
                    self.TARGET_SIZE, interpolation=InterpolationMode.NEAREST
                ),
                transforms.PILToTensor(),
            ]
        )
        # TODO: changing the classes will result in CUDA error
        self.include_classes = [1, 2, 3, 4]
        self.image_dir = "./data/dataset/images"
        self.json_dir = "./data/dataset/labels/json_train"

    def get_train_dataset(self):
        train_dataset = UBPDatasetTrain(
            image_dir=self.image_dir,
            json_dir=self.json_dir,
            transform=self.image_transform,
            target_transform=self.mask_transform,
            include_classes=self.include_classes,
        )
        return train_dataset

    def evaluate_on_test(self, model, out_channels):
        test_dataset = UBPDatasetTest(
            self.image_dir,
            self.json_dir,
            transform=self.image_transform,
            target_transform=self.mask_transform,
            include_classes=self.include_classes,
        )

        test_loader = DataLoader(
            test_dataset, batch_size=4, shuffle=False, num_workers=2
        )
        evaluator = Evaluate(
            model=model, dataloader=test_loader, device="cuda", num_classes=out_channels
        )
        mean_dice, per_class = evaluator.evaluate_dice_score()
        print(f"\nMean Dice (no background): {mean_dice:.4f}")
        for cid, (name, score) in per_class.items():
            print(f"Class {cid:>2} ({name:<10}): Dice = {score:.4f}")

        evaluator.visualize_ranked(k=3, save_dir=None, alpha=0.45)


def experiment(variant):
    dm = DatasetManager()
    train_dataset = dm.get_train_dataset()
    out_channels = len(dm.include_classes) + 1
    model = UNet(in_channels=1, out_channels=out_channels)
    exp_id = random.randint(int(1e5), int(1e6) - 1)
    trainer = get_trainer(
        variant.get("cv", False),
        exp_name=exp_id,
        model=model,
        dataset=train_dataset,
        params=variant,
        device="cuda",
        log_to_wandb=variant.get("log_to_wandb", False),
    )
    trainer.run()
    # Optionally, run evaluation after training (for single split)
    if not variant.get("cv", False):
        dm.evaluate_on_test(trainer.model, out_channels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type", type=str, default="unet"
    )  # only unet is applicable for now
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--max_epoch", type=int, default=100
    )  # the maximum number of epoch for training. If model from scratch, may increase the value to 150
    parser.add_argument(
        "--learning_rate", "-lr", type=float, default=1e-3
    )  # initial learning rate, changes over epoch according to the scheduler
    parser.add_argument(
        "--optimizer", type=str, default="adamw"
    )  # values: 'adam', 'adamw', 'rmsprop'
    parser.add_argument(
        "--criterion", type=str, default="ce"
    )  # values: 'bce', 'bcedice' (mixed bce and dice), 'dice'
    parser.add_argument(
        "--scheduler", type=str, default="rrlonp"
    )  # values: 'rrlonp' (reduce rl on plateau), 'ca' (cosine annealing)
    parser.add_argument(
        "--early_stopping", type=str2bool, default=True
    )  # if True, early stopping is applied
    parser.add_argument(
        "--patience", type=int, default=10
    )  # specify the value for early stopping
    parser.add_argument(
        "--half_precision", type=str2bool, default=False
    )  # if False, use single precision
    parser.add_argument(
        "--cv", type=str2bool, default=False
    )  # if True, will do cross-validation with fold=5
    parser.add_argument("--seed", type=int, default=42)  # seed for the experiment
    parser.add_argument("--device", type=str, default="cuda")  # values: 'cuda', 'cpu'
    parser.add_argument(
        "--log_to_wandb", "-wandb", type=str2bool, default=False
    )  # if True, experiment will be logged to weights and biases

    args = parser.parse_args()

    experiment(variant=vars(args))
