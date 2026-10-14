from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """
    use the same seed for random, numpy, torch and torch.cuda libraries
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
