# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Skip marker for on-GPU numeric tests: run only on given gfx architectures with ROCm torch.

The gate asks rocKE's HIP query for the device's base architecture (``gfx942``, feature flags
stripped), not torch's device name: the name differs within a family and between torch builds
(some gfx942 parts report a generic device name under torch 2.14 + ROCm 7.2), so a name match
skips silently on hardware the test should cover.
"""

from __future__ import annotations

import functools

import pytest


@functools.cache
def gpu_arch() -> str | None:
    """Base gfx id of device 0 when ROCm torch can use it, else ``None``.

    If rocKE's query fails while torch sees the GPU, torch's ``gcnArchName`` decides, so a
    broken rocKE runtime shows up as failing tests instead of a skip.
    """
    try:
        import torch
    except Exception:  # noqa: BLE001 - torch is optional
        return None
    if not torch.cuda.is_available():
        return None
    try:
        from rocke.runtime.hip_module import get_device_arch

        arch = get_device_arch(0)
    except Exception:  # noqa: BLE001 - fall back to torch below
        arch = None
    return arch or torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]


def requires_gpu(*archs: str):
    return pytest.mark.skipif(
        gpu_arch() not in archs,
        reason=f"needs a {' or '.join(archs)} GPU with ROCm torch",
    )
