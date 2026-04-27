"""Heuristics for batch size and DataLoader settings from available hardware.

Used by ``experiments.run_cnn_lstm`` so a single command line can scale to GPU VRAM
and CPU count without hand-tuning. Install a **CUDA** build of PyTorch to use a GPU
(see project README); CPU-only wheels still train but cannot use the discrete GPU.
"""

from __future__ import annotations

import os
import sys

import torch


def gpu_total_memory_gib(device_index: int = 0) -> float | None:
    """Return total GPU memory in GiB, or None if CUDA is not available."""
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(device_index)
    return float(props.total_memory) / (1024.0**3)


def suggest_batch_size(*, use_cuda: bool, gpu_mem_gib: float | None) -> int:
    """Pick a conservative batch size that scales with GPU memory.

    CNN-LSTM on long sequences (e.g. 256 tokens) is memory-heavy; the OOM path in
    :class:`Trainer` can still halve the batch on overflow.
    """
    if use_cuda and gpu_mem_gib is not None:
        if gpu_mem_gib >= 24.0:
            return 160
        if gpu_mem_gib >= 16.0:
            return 128
        if gpu_mem_gib >= 12.0:
            return 96
        if gpu_mem_gib >= 8.0:
            return 64
        if gpu_mem_gib >= 6.0:
            return 48
        return 32
    return 32


def suggest_num_workers() -> int:
    """Background workers for DataLoader (0 on single-core, capped for Windows I/O)."""
    n = os.cpu_count() or 4
    if sys.platform == "win32":
        return max(0, min(4, n - 1))
    return max(0, min(8, n - 1))


def cuda_build_hint() -> str | None:
    """If PyTorch was built without CUDA, return a short install hint, else None."""
    if torch.cuda.is_available():
        return None
    # torch.version.cuda is set when the wheel includes CUDA; None for CPU-only wheels
    if getattr(torch.version, "cuda", None) is None:
        return (
            "This PyTorch build has no CUDA. For GPU training install a CUDA build from "
            "https://pytorch.org/get-started/locally/ (pick your OS and a CUDA version, "
            "then run the given pip install command)."
        )
    return "CUDA is not available (driver/runtime issue? Check nvidia-smi and CUDA toolkit)."
