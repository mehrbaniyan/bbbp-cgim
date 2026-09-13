import random

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_grad_scaler(device: torch.device):
    """Create the AMP gradient scaler for the selected device."""
    try:
        return torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    except TypeError:  # PyTorch versions with the older signature
        return torch.amp.GradScaler(enabled=device.type == "cuda")

