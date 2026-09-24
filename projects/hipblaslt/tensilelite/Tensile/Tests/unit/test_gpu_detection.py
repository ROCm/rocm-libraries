# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Unit tests for the visible-device helpers in Tensile.Tests.gpu_detection.

TheRock's GPU runners map every GPU into the container and isolate one with
ROCR_VISIBLE_DEVICES, so amd-smi (detectAvailableGpus) reports the whole node.
Assigning xdist workers from that physical count sent all but gw0 to device
indices HIP could not see ("no ROCm-capable device is detected").
"""

import sys
from types import SimpleNamespace

import pytest

from Tensile.Tests.gpu_detection import visible_device_count, worker_gpu_index

pytestmark = pytest.mark.unit

PHYSICAL_GPUS = 8


@pytest.fixture(autouse=True)
def _eight_physical_gpus(monkeypatch):
    """Pretend amd-smi sees a full 8-GPU node, and start with no isolation."""
    monkeypatch.setitem(
        sys.modules,
        "Tensile.ParallelExecution",
        SimpleNamespace(detectAvailableGpus=lambda: PHYSICAL_GPUS),
    )
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)


@pytest.mark.parametrize(
    "rocr, hip, expected",
    [
        (None, None, PHYSICAL_GPUS),
        ("5", None, 1),
        (None, "0,1", 2),
        ("0,1,2,3", "0,1", 2),
        ("4, 5,", None, 2),
    ],
)
def test_visible_device_count(monkeypatch, rocr, hip, expected):
    if rocr is not None:
        monkeypatch.setenv("ROCR_VISIBLE_DEVICES", rocr)
    if hip is not None:
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", hip)
    assert visible_device_count() == expected


@pytest.mark.parametrize("worker_id", ["gw0", "gw1", "gw2", "gw3"])
def test_isolated_gpu_puts_every_worker_on_index_zero(monkeypatch, worker_id):
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "5")
    assert worker_gpu_index(worker_id) == 0


@pytest.mark.parametrize("worker_id, expected", [("gw3", 3), ("gw9", 1)])
def test_unisolated_workers_wrap_physical_gpus(worker_id, expected):
    assert worker_gpu_index(worker_id) == expected


def test_worker_id_without_number_returns_none():
    assert worker_gpu_index("master") is None
