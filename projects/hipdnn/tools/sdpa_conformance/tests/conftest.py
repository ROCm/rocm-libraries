# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Pytest configuration for the hipdnn_sdpa tests.

Three tiers:

  * **CPU tests** (Philox, problem layout): need only torch.
  * **Graph tests** (``test_sdpa_graph.py``): need the built ``hipdnn_frontend``
    bindings (SDPA enabled, ragged-offset bindings) but no handle and no device;
    they skip cleanly when the bindings are absent.
  * **GPU tests** (marked ``gpu``): execute on a device. CTest selects ``-m 'not gpu'``.
"""

import sys
from pathlib import Path

import pytest

# `hipdnn_sdpa` lives one directory up; make the tests runnable without PYTHONPATH.
_PACKAGE_ROOT = str(Path(__file__).resolve().parent.parent)
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "gpu: test executes hipDNN on a ROCm device (deselect with -m 'not gpu')",
    )


@pytest.fixture
def cuda_device():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no ROCm device")
    return torch.device("cuda", torch.cuda.current_device())
