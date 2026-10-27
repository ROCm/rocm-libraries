# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The GPU test helper checks the arch of the GPU it launches on.

rocke's runtime launches on HIP's current device, so the arch check must read that
device and not a fixed device 0. With no GPU the helper skips, which is its own
contract.
"""

from __future__ import annotations

import pytest

from rocke.tests.gpu import current_device_arch

hip = pytest.importorskip("rocke.runtime.hip_module")


@pytest.fixture(autouse=True)
def _gpu_present() -> None:
    """Skip without a HIP runtime or GPU: the HIP library loads on first use, so the
    import above succeeds even where the device queries below would raise."""
    current_device_arch()


def _select_device(device: int) -> None:
    hip._check(hip._hipSetDevice(device), "hipSetDevice")


def test_reports_the_only_devices_arch():
    if hip.get_device_count() != 1:
        pytest.skip("needs exactly one GPU")
    assert current_device_arch() == hip.get_device_arch(0)


def test_follows_a_change_of_current_device():
    if hip.get_device_count() < 2:
        pytest.skip("needs two GPUs")
    previous = hip.current_device()
    try:
        _select_device(1)
        assert current_device_arch() == hip.get_device_arch(1)
        _select_device(0)
        assert current_device_arch() == hip.get_device_arch(0)
    finally:
        _select_device(previous)
