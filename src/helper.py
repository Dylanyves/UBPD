import argparse
import torch

from collections import OrderedDict
from src.models.unet import UNet


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "yes", "1"):
        return True
    elif v.lower() in ("false", "no", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def load_model(model_path, in_channels=1, num_classes=5, device="cuda"):
    """
    Instantiate your UNet and load weights. Adjust the import/args
    to match your actual UNet definition.
    """
    # >>> replace this import/constructor with your actual UNet <<<
    # from your_model_module import UNet
    model = UNet(in_channels=in_channels, out_channels=num_classes)

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
