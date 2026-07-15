# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Pytest configuration for hipDNN Python binding tests.

Auto-skips tests marked ``gpu`` when no ROCm-capable GPU is usable, so the
suite can run on CPU-only machines without spurious failures.
"""

import functools
import warnings

import pytest

import hipdnn_frontend as hipdnn


@functools.lru_cache(maxsize=1)
def _gpu_available():
    """Return True when HIP reports at least one visible GPU device."""
    try:
        device_count = hipdnn.hip_get_device_count()
    except Exception as exc:
        warnings.warn(f"HIP device probe failed: {exc!r}", stacklevel=1)
        return False
    if device_count <= 0:
        warnings.warn(
            f"HIP device probe reported {device_count} visible device(s).",
            stacklevel=1,
        )
        return False
    return True


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: test requires a ROCm-capable GPU")


def pytest_collection_modifyitems(config, items):
    if _gpu_available():
        return
    gpu_items = [item for item in items if "gpu" in item.keywords]
    if gpu_items:
        warnings.warn(
            f"No ROCm-capable GPU available; skipping {len(gpu_items)} gpu test(s).",
            stacklevel=1,
        )
    skip_gpu = pytest.mark.skip(reason="no ROCm-capable GPU available")
    for item in gpu_items:
        item.add_marker(skip_gpu)
