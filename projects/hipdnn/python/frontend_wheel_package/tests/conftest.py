# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Pytest configuration for hipDNN Python binding tests.

Auto-skips tests marked ``gpu`` when no ROCm-capable GPU is usable, so the
suite can run on CPU-only machines without spurious failures.
"""

import functools
import os
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


def _load_test_good_plugin():
    """Load the test-only "good" engine plugin, if configured, in ABSOLUTE mode.

    The plugin (``test_good_plugin`` CMake target) is a software-only stub:
    it claims applicability to any graph unconditionally and its execute()
    is a no-op. ABSOLUTE mode replaces default engine discovery (MIOpen/
    hipblaslt/hip-kernel-provider) instead of adding to it, so GoodPlugin is
    the only engine ever loaded. Loading it ADDITIVE alongside a real
    provider makes two engines applicable to the same op, which empirically
    triggers a real intermittent GPU memory fault in MIOpen/ROCr (confirmed
    via AMD_LOG_LEVEL=3 tracing) -- not fixable from this repo.

    Consequence: with this set, tests needing a real provider (conv,
    batchnorm, ...) get zero output from the stub and fail their
    correctness assertions. Only use this for the ops with no real
    coverage yet (build_all_plans_or_skip() tests), never for a full run.

    Must run before any handle is created (pytest_configure, not a
    fixture). No-op if HIPDNN_TEST_GOOD_PLUGIN_PATH is unset/missing.
    """
    plugin_path = os.environ.get("HIPDNN_TEST_GOOD_PLUGIN_PATH")
    if not plugin_path or not os.path.isfile(plugin_path):
        return
    hipdnn.set_engine_plugin_paths([plugin_path], hipdnn.PluginLoadingMode.ABSOLUTE)


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: test requires a ROCm-capable GPU")
    _load_test_good_plugin()


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
