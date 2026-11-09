"""Runtime environment helpers for CPU/GPU and PyTorch setup.

This replaces the previous src.utils.env module so that top-level scripts
can configure the PyTorch/Sionna runtime before importing heavy modules.
"""

from __future__ import annotations

import os
import tempfile


def configure_env(force_cpu: bool = False, gpu_num: int = 0) -> None:
    """Configure basic environment variables before importing PyTorch.

    - Set TF logging verbosity lower
    - Optionally force CPU
    - Otherwise select a single GPU by index
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    # Keep matplotlib cache in a writable location across runs.
    mpl_cache_dir = os.path.join(tempfile.gettempdir(), "factory6g_mplconfig")
    os.makedirs(mpl_cache_dir, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = mpl_cache_dir
    if force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)


def setup_gpu(gpu_num: int = 0, force_cpu: bool = False) -> None:
    """Point Sionna at the chosen device (PyTorch stack); safe on CPU-only hosts."""
    try:
        import torch
        from sionna.phy import config
    except Exception:
        return
    if force_cpu or not torch.cuda.is_available():
        config.device = "cpu"
        return
    config.device = "cuda:0"  # CUDA_VISIBLE_DEVICES already selects gpu_num
