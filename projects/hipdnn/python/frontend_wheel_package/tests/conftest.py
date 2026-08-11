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


_stub_active = False


def _load_test_good_plugin():
    """Load the ``test_good_plugin`` stub in ABSOLUTE mode for the whole session.

    The stub claims every graph and its execute() is a no-op, giving stubbed
    execution for every op; numeric correctness is the C++ tests' job. ABSOLUTE
    mode ensures it is the only engine loaded, so no test silently runs against
    a real provider -- the few that need one check ``stub_engine_active()`` and
    skip themselves.

    CI sets HIPDNN_TEST_GOOD_PLUGIN_PATH to the installed plugin. Must run
    before any handle is created, hence pytest_configure rather than a fixture.
    """
    global _stub_active
    plugin_path = os.environ.get("HIPDNN_TEST_GOOD_PLUGIN_PATH")
    if not plugin_path or not os.path.isfile(plugin_path):
        return
    hipdnn.set_engine_plugin_paths([plugin_path], hipdnn.PluginLoadingMode.ABSOLUTE)
    _stub_active = True


def stub_engine_active() -> bool:
    """Return True once the ABSOLUTE-mode test stub has replaced real engine discovery."""
    return _stub_active


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
