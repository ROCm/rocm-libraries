# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""GPU availability check shared by CLI entry points."""

import subprocess


def gpu_is_available() -> bool:
    """Return True if at least one ROCm/CUDA GPU is visible to the process."""
    try:
        import torch

        if torch.cuda.is_available():
            return True
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["rocm-smi", "--showid"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        pass
    return False
