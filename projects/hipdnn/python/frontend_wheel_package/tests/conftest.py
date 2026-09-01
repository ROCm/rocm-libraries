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

from . import helpers


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


_attempted_plugin_path = None


def _default_plugin_path():
    """Where the superbuild installs test_good_plugin under a patched ROCm SDK.

    Mirrors the C++ layout (``<prefix>/<libdir>/test_plugins/custom`` plus the
    platform library naming from ``hipdnn_data_sdk::utilities::getLibraryName``)
    so neither CI nor a developer has to paste the path in by hand.
    """
    root = os.environ.get("ROCM_PATH") or os.environ.get("ROCM_SDK_PATH")
    if not root:
        return None
    libdir, filename = (
        ("bin", "test_good_plugin.dll")
        if os.name == "nt"
        else ("lib", "libtest_good_plugin.so")
    )
    return os.path.join(root, libdir, "test_plugins", "custom", filename)


def _load_test_good_plugin():
    """Load the ``test_good_plugin`` stub in ABSOLUTE mode for the whole session.

    The stub claims every graph and its execute() is a no-op, giving stubbed
    execution for every op; numeric correctness is the C++ tests' job. ABSOLUTE
    mode ensures it is the only engine loaded, so no test silently runs against
    a real provider -- the few that need one check
    ``helpers.stub_engine_active()`` and skip themselves.

    An explicitly configured HIPDNN_TEST_GOOD_PLUGIN_PATH that does not exist is
    a hard error: the caller asked for a specific plugin and did not get it, and
    everything downstream would fail in confusing ways. A missing plugin at the
    derived default is not fatal here, because the no-GPU tier needs no engine
    at all; ``_require_stub_engine`` fails the gpu tier instead.

    Must run before any handle is created, hence pytest_configure not a fixture.
    """
    global _attempted_plugin_path
    override = os.environ.get("HIPDNN_TEST_GOOD_PLUGIN_PATH")
    _attempted_plugin_path = override or _default_plugin_path()
    if override and not os.path.isfile(override):
        raise pytest.UsageError(
            f"HIPDNN_TEST_GOOD_PLUGIN_PATH={override!r} is not a file. Point it "
            "at the superbuild's installed test_good_plugin, or unset it to "
            "resolve the plugin from ROCM_PATH."
        )
    if not _attempted_plugin_path or not os.path.isfile(_attempted_plugin_path):
        return
    hipdnn.set_engine_plugin_paths(
        [_attempted_plugin_path], hipdnn.PluginLoadingMode.ABSOLUTE
    )
    helpers.set_stub_engine_active(True)
    helpers.set_stub_engine_path(_attempted_plugin_path)


@pytest.fixture(autouse=True)
def _require_stub_engine(request):
    """Fail a gpu test outright when the stub it needs was never loaded.

    Scoped to the gpu marker so a CPU-only ``-m 'not gpu'`` run, which needs no
    engine at all, is unaffected. Without this a bad path surfaces as a
    confusing "no compatible engine" deep inside build_all_plans, or silently
    runs the suite against whatever real provider happens to be installed.
    """
    if "gpu" in request.keywords and not helpers.stub_engine_active():
        pytest.fail(
            f"test_good_plugin was not loaded (looked at {_attempted_plugin_path!r}). "
            "The gpu tier needs the superbuild's stub engine: build and "
            "'cmake --install' hipDNN, point ROCM_PATH at that prefix, or set "
            "HIPDNN_TEST_GOOD_PLUGIN_PATH directly. Run 'pytest -m \"not gpu\"' "
            "for the binding-signature tests, which need no engine."
        )


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
