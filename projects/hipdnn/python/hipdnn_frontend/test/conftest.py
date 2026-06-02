# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Pytest configuration for hipDNN Python binding tests.

Auto-skips tests marked ``gpu`` when no ROCm-capable GPU is usable, so the
suite can run on CPU-only machines without spurious failures.
"""

import functools

import pytest

import hipdnn_frontend as hipdnn


@functools.lru_cache(maxsize=1)
def _gpu_available():
    """Return True if a GPU device can be allocated, False otherwise.

    Probes by attempting a tiny device allocation, which performs a hipMalloc
    and raises when no usable device is present.
    """
    try:
        hipdnn.DeviceBuffer(1)
    except Exception:
        return False
    return True


def pytest_collection_modifyitems(config, items):
    if _gpu_available():
        return
    skip_gpu = pytest.mark.skip(reason="no ROCm-capable GPU available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
