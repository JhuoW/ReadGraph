"""Seeding for reproducible runs (CLAUDE.md ground rule 6)."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed python/numpy/torch. `deterministic=True` additionally forces deterministic
    kernels (needed for the exact-repeatability test; slower, and requires
    CUBLAS_WORKSPACE_CONFIG for CUDA matmuls)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
