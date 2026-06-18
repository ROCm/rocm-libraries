# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""PyTorch capability checks.

These helpers only describe PyTorch's availability. They do not answer whether
HIP, hipDNN, or the host has a usable GPU through any non-PyTorch runtime.
"""


def module_available() -> bool:
    """Return True when the torch Python module can be imported.

    Broken installs commonly fail import with OSError/RuntimeError from
    missing shared libraries, not ImportError, so any import-time failure
    means "not available".
    """
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def gpu_available() -> bool:
    """Return True when PyTorch's CUDA/ROCm facade can use a GPU."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def is_rocm_build() -> bool:
    """Return True when the installed torch is a ROCm/HIP build."""
    try:
        import torch

        return bool(getattr(torch.version, "hip", None))
    except Exception:
        return False


def is_cuda_build() -> bool:
    """Return True when the installed torch is a CUDA (non-ROCm) build."""
    try:
        import torch

        return bool(getattr(torch.version, "cuda", None)) and not bool(
            getattr(torch.version, "hip", None)
        )
    except Exception:
        return False
