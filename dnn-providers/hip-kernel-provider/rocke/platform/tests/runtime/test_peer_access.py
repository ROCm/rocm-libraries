# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only contracts for device selection and scoped directed peer mappings."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from rocke.runtime import (
    HipError,
    Runtime,
    can_access_peer,
    device_scope,
    get_device,
    peer_access_scope,
    set_device,
)
from rocke.runtime import hip_module as hip


def test_device_scope_restores_after_exception(fake_hip) -> None:
    fake_hip.device = 2
    with pytest.raises(RuntimeError, match="body failed"), device_scope(1):
        assert get_device() == 1
        raise RuntimeError("body failed")
    assert get_device() == 2


def test_device_scope_restores_body_device_changes_and_nesting(fake_hip) -> None:
    with device_scope(0):
        set_device(1)
        with device_scope(2):
            assert get_device() == 2
        assert get_device() == 1
    assert get_device() == 0


def test_device_scope_failed_selection_preserves_caller(fake_hip) -> None:
    fake_hip.errors[("set", 1)] = 101
    with pytest.raises(HipError, match="hipSetDevice\\(1\\)"), device_scope(1):
        pytest.fail("must not enter after selection failed")
    assert fake_hip.device == 0


def test_initialization_preserves_existing_selection(fake_hip, monkeypatch) -> None:
    fake_hip.device = 2
    monkeypatch.setattr(hip, "_hip_inited", False)
    assert get_device() == 2
    assert ("init", 0) in fake_hip.calls
    hip._hipSetDevice.assert_not_called()


@pytest.mark.parametrize("ordinal", [-1, True, False, 1.5, "1", 2**31, 2**32])
def test_invalid_ordinals_fail_before_hip(fake_hip, ordinal) -> None:
    with pytest.raises((TypeError, ValueError)):
        set_device(ordinal)
    with pytest.raises((TypeError, ValueError)):
        can_access_peer(0, ordinal)
    assert not fake_hip.calls


def test_peer_capability_is_directed_and_does_not_enable(fake_hip) -> None:
    fake_hip.device = 2
    fake_hip.capabilities[(1, 0)] = False
    assert can_access_peer(0, 1)
    assert not can_access_peer(1, 0)
    assert fake_hip.device == 2
    assert not fake_hip.mappings
    hip._hipDeviceEnablePeerAccess.assert_not_called()


def test_peer_query_error_is_not_unsupported(fake_hip) -> None:
    fake_hip.errors[("can", 0, 1)] = 101
    with pytest.raises(HipError, match="hipDeviceCanAccessPeer"):
        can_access_peer(0, 1)


def test_enable_disable_report_owned_vs_existing_mapping(fake_hip) -> None:
    runtime = Runtime()
    assert runtime.enable_peer_access(1)
    assert not runtime.enable_peer_access(1)
    assert runtime.disable_peer_access(1)
    assert not runtime.disable_peer_access(1)


@pytest.mark.parametrize("operation", ["enable", "disable"])
def test_peer_mapping_errors_propagate(fake_hip, operation) -> None:
    fake_hip.errors[(operation, 0, 1)] = 217
    method = getattr(Runtime(), f"{operation}_peer_access")
    with pytest.raises(HipError, match="hipError\\(217\\)"):
        method(1)


def test_scope_preflights_all_paths_before_enabling(fake_hip) -> None:
    fake_hip.capabilities[(1, 0)] = False
    with (
        pytest.raises(HipError, match="unsupported from device 1 to 0"),
        peer_access_scope([(0, 1), (1, 0)]),
    ):
        pytest.fail("must not enter unsupported scope")
    assert not fake_hip.mappings
    hip._hipDeviceEnablePeerAccess.assert_not_called()


@pytest.mark.parametrize("paths", [[(0, 0)], [(0, 1), (0, 1)], [(-1, 0)]])
def test_scope_rejects_invalid_paths_before_hip(fake_hip, paths) -> None:
    with pytest.raises(ValueError), peer_access_scope(paths):
        pytest.fail("invalid path accepted")
    assert not fake_hip.calls


def test_scope_drains_all_sources_before_owned_mapping_teardown(fake_hip) -> None:
    fake_hip.device = 2
    fake_hip.mappings.add((0, 1))
    with peer_access_scope([(0, 1), (1, 0)]):
        assert fake_hip.mappings == {(0, 1), (1, 0)}
        assert get_device() == 2
    assert fake_hip.mappings == {(0, 1)}
    assert fake_hip.device == 2
    syncs = [i for i, call in enumerate(fake_hip.calls) if call[0] == "sync"]
    disables = [i for i, call in enumerate(fake_hip.calls) if call[0] == "disable"]
    assert len(syncs) == 2 and len(disables) == 1
    assert max(syncs) < min(disables)


def test_scope_rolls_back_partial_acquisition(fake_hip) -> None:
    fake_hip.errors[("enable", 1, 0)] = 217
    with (
        pytest.raises(HipError, match="hipDeviceEnablePeerAccess"),
        peer_access_scope([(0, 1), (1, 0)]),
    ):
        pytest.fail("acquisition failure ignored")
    assert not fake_hip.mappings
    assert fake_hip.device == 0
    assert ("sync", 0) in fake_hip.calls
    assert ("disable", 0, 1) in fake_hip.calls


def test_scope_exception_drains_and_cleans_up(fake_hip) -> None:
    with pytest.raises(RuntimeError, match="body failed"), peer_access_scope([(0, 1)]):
        raise RuntimeError("body failed")
    assert not fake_hip.mappings
    assert ("sync", 0) in fake_hip.calls


def test_scope_failed_drain_keeps_mappings_and_attempts_other_drains(fake_hip) -> None:
    fake_hip.errors[("sync", 0)] = 700
    with pytest.raises(HipError, match="drain failed"), peer_access_scope(
        [(0, 1), (1, 0)]
    ):
        pass
    assert fake_hip.mappings == {(0, 1), (1, 0)}
    assert ("sync", 1) in fake_hip.calls
    hip._hipDeviceDisablePeerAccess.assert_not_called()
    assert fake_hip.device == 0


def test_scope_attempts_remaining_cleanup_after_disable_error(fake_hip) -> None:
    fake_hip.errors[("disable", 1, 0)] = 700
    with (
        pytest.raises(HipError, match="mapping cleanup failed"),
        peer_access_scope([(0, 1), (1, 0)]),
    ):
        pass
    assert fake_hip.mappings == {(1, 0)}
    assert ("disable", 0, 1) in fake_hip.calls


def test_scope_supports_several_source_peers(fake_hip) -> None:
    paths = [(0, 1), (0, 2), (1, 0), (2, 0)]
    with peer_access_scope(paths):
        assert fake_hip.mappings == set(paths)
    assert not fake_hip.mappings
    assert {call[1] for call in fake_hip.calls if call[0] == "sync"} == {0, 1, 2}


def test_empty_scope_is_library_free(fake_hip) -> None:
    with peer_access_scope([]):
        pass
    assert not fake_hip.calls


def test_runtime_import_keeps_new_peer_bindings_lazy() -> None:
    script = """
import ctypes

def forbidden_load(*args, **kwargs):
    raise AssertionError("import attempted to load a shared library")

ctypes.CDLL = forbidden_load
import rocke.runtime as runtime
from rocke.runtime import hip_module as hip
assert hip._hip is None
assert hip._hipDeviceCanAccessPeer._fn is None
assert hip._hipDeviceEnablePeerAccess._fn is None
assert hip._hipDeviceDisablePeerAccess._fn is None
assert {"get_device", "set_device", "device_scope", "can_access_peer", "peer_access_scope"} <= set(runtime.__all__)
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(hip.__file__).resolve().parents[2])
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
