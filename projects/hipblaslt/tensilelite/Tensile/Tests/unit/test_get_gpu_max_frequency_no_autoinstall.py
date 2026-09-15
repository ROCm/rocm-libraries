# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Regression tests for Tensile.Tensile.get_gpu_max_frequency (ROCM-26748 / SEC-00581).

When hip-python is not importable, get_gpu_max_frequency must degrade
gracefully (return None so the caller falls back to amd-smi) and MUST NOT
attempt to auto-install anything via pip/subprocess. Auto-installing from a
package index at build time is a dependency-confusion supply-chain risk and
also hard-crashes on platforms with no hip-python wheel (e.g. Windows).

These run without a real GPU.
"""

import builtins
import subprocess
from unittest.mock import patch

import pytest

# Importing Tensile.Tensile pulls in the full code-generation toolchain
# (rocisa bindings, etc.). If that chain is unavailable in the current
# environment, skip rather than error at collection time.
try:
    from Tensile.Tensile import get_gpu_max_frequency
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment dependent
    get_gpu_max_frequency = None
    _IMPORT_ERROR = exc

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        get_gpu_max_frequency is None,
        reason=f"Tensile.Tensile import unavailable: {_IMPORT_ERROR}",
    ),
]

_real_import = builtins.__import__


def _import_without_hip(name, *args, **kwargs):
    """Force `from hip import hip` to fail, pass everything else through."""
    if name == "hip" or name.startswith("hip."):
        raise ImportError("hip-python not installed (simulated)")
    return _real_import(name, *args, **kwargs)


def test_returns_none_when_hip_missing():
    with patch.object(builtins, "__import__", side_effect=_import_without_hip):
        assert get_gpu_max_frequency(0) is None


def test_does_not_autoinstall_when_hip_missing():
    with patch.object(subprocess, "run") as run_spy, patch.object(
        builtins, "__import__", side_effect=_import_without_hip
    ):
        result = get_gpu_max_frequency(0)

    assert result is None
    # The whole point of the fix: no pip/subprocess bootstrap of hip-python.
    run_spy.assert_not_called()
