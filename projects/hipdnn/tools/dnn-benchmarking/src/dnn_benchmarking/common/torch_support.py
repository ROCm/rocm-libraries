# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""PyTorch capability checks.

These helpers only describe PyTorch's availability. They do not answer whether
HIP, hipDNN, or the host has a usable GPU through any non-PyTorch runtime.
"""


def module_available() -> bool:
    """Return True when the torch Python module can be imported."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def gpu_available() -> bool:
    """Return True when PyTorch's CUDA/ROCm facade can use a GPU."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False
