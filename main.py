import argparse

from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader

from src.helper import str2bool
from src.dataset import UBPDatasetTrain, UBPDatasetTest
from src.models.unet import UNet
from src.train import Training
from src.evaluate import Evaluate

def experiment(variant):
    print("Hello from ubpd!")


    if variant["cv"]:
        if variant["gkf"]:
            TARGET_SIZE = (512, 512)

            image_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(TARGET_SIZE, antialias=True)
            ])

            mask_transform = transforms.Compose([
                transforms.Resize(TARGET_SIZE, interpolation=InterpolationMode.NEAREST),
                transforms.PILToTensor()
            ])

            # TODO: changing the classes will result in CUDA error
            include_classes = [1,4]
            image_dir = "./data/dataset/images"
            json_dir = "./data/dataset/labels/json_train"

            train_dataset = UBPDatasetTrain(
                                image_dir=image_dir,
                                json_dir=json_dir,
                                transform=image_transform,
                                target_transform=mask_transform,
                                include_classes=include_classes
                            )
            
            out_channels = len(include_classes) + 1
            model = UNet(in_channels=1, out_channels=out_channels)

            trainer = Training(exp_name="exp-test-only-nerve", 
                               model=model, 
                               dataset=train_dataset, 
                               split_strategy="gkf", 
                               params=variant,
                               device="cuda", 
                               log_to_wandb=False
                            )
            trainer.train()


            test_dataset = UBPDatasetTest(
            image_dir, json_dir,
            transform=image_transform,
            target_transform=mask_transform,
            include_classes=include_classes
        )

        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)
        evaluator = Evaluate(
            model=trainer.model,
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
    parser.add_argument('--batch_size', type=int, default=16) 
    parser.add_argument('--max_epoch', type=int, default=100) # the maximum number of epoch for training. If model from scratch, may increase the value to 150 
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3) # initial learning rate, changes over epoch according to the scheduler
    parser.add_argument('--optimizer', type=str, default='adamw') # values: 'adam', 'adamw', 'rmsprop'
    parser.add_argument('--criterion', type=str, default='ce') # values: 'bce', 'bcedice' (mixed bce and dice), 'dice'
    parser.add_argument('--scheduler', type=str, default='rrlonp') # values: 'rrlonp' (reduce rl on plateau), 'ca' (cosine annealing)
    parser.add_argument('--early_stopping', type=str2bool, default=True) # if True, early stopping is applied
    parser.add_argument('--patience', type=int, default=10) # specify the value for early stopping
    parser.add_argument('--half_precision', type=str2bool, default=False) # if False, use single precision
    parser.add_argument('--cv', type=str2bool, default=False) # if True, will do cross-validation with fold=5
    parser.add_argument('--gkf', type=str2bool, default=False) # if True, split based on patients instead of images
    parser.add_argument('--seed', type=int, default=42) # seed for the experiment
    parser.add_argument('--device', type=str, default='cuda') # values: 'cuda', 'cpu'
    parser.add_argument('--log_to_wandb', '-wandb', type=str2bool, default=False) # if True, experiment will be logged to weights and biases
    
    args = parser.parse_args()

    experiment(variant=vars(args))