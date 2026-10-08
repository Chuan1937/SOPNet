from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, deterministic: bool = False) -> int:
    """Seed Python, NumPy and PyTorch (CPU + CUDA) RNGs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed
