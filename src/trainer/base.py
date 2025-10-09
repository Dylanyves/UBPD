# Base trainer class and shared utilities


class BaseTrainer:
    def __init__(self, exp_name, model, dataset, params, device, log_to_wandb=False):
        self.exp_name = exp_name
        self.model = model.to(device)
        self.dataset = dataset
        self.params = params
        self.device = device
        self.log_to_wandb = log_to_wandb

    def run(self):
        raise NotImplementedError("Subclasses must implement run()")
