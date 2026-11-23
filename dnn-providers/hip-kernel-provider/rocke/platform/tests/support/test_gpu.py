# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The GPU test helper checks the arch of the GPU it launches on.

rocke's runtime launches on HIP's current device, so the arch check must read that
device and not a fixed device 0. With no GPU the helper skips, which is its own
contract. A ``ROCKE_REQUIRE_<ARCH>=1`` lane turns the skip for a listed arch into a
failure.
"""

from __future__ import annotations

import pytest
from rocke.core.arch import known_arches
from rocke.tests.gpu import current_device_arch, current_device_arch_in
from rocke.tests.lanes import (
    MULTI_GPU_RUNNER,
    lanes_waited_for,
    runner_lanes_for,
    skip_until,
    skip_until_reason,
)

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
        skip_until(MULTI_GPU_RUNNER, reason="needs two GPUs")
    previous = hip.current_device()
    try:
        _select_device(1)
        assert current_device_arch() == hip.get_device_arch(1)
        _select_device(0)
        assert current_device_arch() == hip.get_device_arch(0)
    finally:
        _select_device(previous)


def _another_arch_with(runner_lane: bool) -> str:
    """A known arch that is not the current GPU's, with a runner lane CI lacks
    (``rocke.tests.lanes.runner_lanes_for``) or without one."""
    arch = current_device_arch()
    return next(
        other
        for other in known_arches()
        if other != arch and bool(runner_lanes_for((other,))) == runner_lane
    )


def test_returns_the_current_arch_when_listed():
    arch = current_device_arch()
    assert current_device_arch_in((arch,)) == arch


def test_skips_when_the_current_arch_is_not_listed(monkeypatch):
    other = _another_arch_with(runner_lane=False)
    monkeypatch.delenv(f"ROCKE_REQUIRE_{other.upper()}", raising=False)
    arch = current_device_arch()
    with pytest.raises(pytest.skip.Exception) as skip:
        current_device_arch_in((other,))
    assert str(skip.value) == f"this test needs one of ['{other}']; found {arch}"


def test_the_skip_names_the_subject(monkeypatch):
    other = _another_arch_with(runner_lane=False)
    monkeypatch.delenv(f"ROCKE_REQUIRE_{other.upper()}", raising=False)
    arch = current_device_arch()
    with pytest.raises(pytest.skip.Exception) as skip:
        current_device_arch_in((other,), subject="kernel_x")
    assert str(skip.value) == f"kernel_x needs one of ['{other}']; found {arch}"


def test_the_skip_declares_the_runner_lane_of_the_listed_arch(monkeypatch):
    other = _another_arch_with(runner_lane=True)
    monkeypatch.delenv(f"ROCKE_REQUIRE_{other.upper()}", raising=False)
    arch = current_device_arch()
    with pytest.raises(pytest.skip.Exception) as skip:
        current_device_arch_in((other,))
    lanes = runner_lanes_for((other,))
    assert str(skip.value) == skip_until_reason(
        *lanes, reason=f"this test needs one of ['{other}']; found {arch}"
    )
    assert lanes_waited_for(str(skip.value)) == lanes


def test_the_skip_declares_no_lane_when_a_listed_arch_has_no_runner_lane(
    monkeypatch,
):
    without_lane = _another_arch_with(runner_lane=False)
    with_lane = _another_arch_with(runner_lane=True)
    for other in (without_lane, with_lane):
        monkeypatch.delenv(f"ROCKE_REQUIRE_{other.upper()}", raising=False)
    with pytest.raises(pytest.skip.Exception) as skip:
        current_device_arch_in((without_lane, with_lane))
    assert lanes_waited_for(str(skip.value)) == ()


def test_fails_when_a_required_arch_is_not_the_current_gpu(monkeypatch):
    arch = current_device_arch()
    other = _another_arch_with(runner_lane=False)
    variable = f"ROCKE_REQUIRE_{other.upper()}"
    monkeypatch.setenv(variable, "1")
    with pytest.raises(pytest.fail.Exception) as failure:
        current_device_arch_in((arch, other))
    assert str(failure.value) == f"{variable}=1 needs a {other} GPU; found {arch}"


def test_a_required_arch_that_is_the_current_gpu_passes(monkeypatch):
    arch = current_device_arch()
    monkeypatch.setenv(f"ROCKE_REQUIRE_{arch.upper()}", "1")
    assert current_device_arch_in((arch,)) == arch
