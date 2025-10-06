import argparse

from helper import str2bool
from dataset import UBPDatasetTrain, UBPDatasetTest

def experiment(variant):
    print("Hello from ubpd!")


    if variant["cv"]:
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='unet') # only unet is applicable for now
    parser.add_argument('--batch_size', type=int, default=32) 
    parser.add_argument('--max_epoch', type=int, default=100) # the maximum number of epoch for training. If model from scratch, may increase the value to 150 
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3) # initial learning rate, changes over epoch according to the scheduler
    parser.add_argument('--optimizer', type=str, default='adamw') # values: 'adam', 'adamw', 'rmsprop'
    parser.add_argument('--criterion', type=str, default='ce') # values: 'bce', 'bcedice' (mixed bce and dice), 'dice'
    parser.add_argument('--scheduler', type=str, default='rrlonp') # values: 'rrlonp' (reduce rl on plateau), 'ca' (cosine annealing)
    parser.add_argument('--early_stopping', type=str2bool, default=True) # if True, early stopping is applied
    parser.add_argument('--patience', type=int, default=10) # specify the value for early stopping
    parser.add_argument('--half_precision', type=str2bool, default=False) # if False, use single precision
    parser.add_argument('--cv', type=str2bool, default=False) # if True, will do cross-validation with fold=5
    parser.add_argument('--group_cv', type=str2bool, default=False) # if True, split based on patients instead of images
    parser.add_argument('--seed', type=int, default=42) # seed for the experiment
    parser.add_argument('--device', type=str, default='cuda') # values: 'cuda', 'cpu'
    parser.add_argument('--log_to_wandb', '-wandb', type=str2bool, default=False) # if True, experiment will be logged to weights and biases
    
    args = parser.parse_args()

    experiment('exp', variant=vars(args))