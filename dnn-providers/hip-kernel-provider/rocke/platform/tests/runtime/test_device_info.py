# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from unittest import mock

import pytest

from rocke import runtime
from rocke.runtime import device_info, hip_module


def _props(target_id: str | None) -> bytes:
    raw = bytearray(4096)
    raw[:14] = b"Marketing Name"
    if target_id is not None:
        encoded = target_id.encode("ascii")
        raw[256 : 256 + len(encoded)] = encoded
    return bytes(raw)


@pytest.mark.parametrize(
    ("target_id", "base_arch"),
    [
        ("gfx90a", "gfx90a"),
        ("gfx11-generic", "gfx11-generic"),
        ("gfx1250-strict", "gfx1250"),
        ("gfx942:sramecc+:xnack-", "gfx942"),
        (None, None),
    ],
)
def test_device_target_id_and_base_arch_are_separate(
    target_id: str | None, base_arch: str | None
) -> None:
    with mock.patch.object(hip_module, "_device_props", return_value=_props(target_id)):
        assert hip_module.get_device_target_id(3) == target_id
        assert hip_module.get_device_arch(3) == base_arch


def test_get_device_asic_revision_queries_stable_attribute() -> None:
    def set_revision(out, attribute, device):
        assert attribute == 10012
        assert device == 3
        out._obj.value = 0
        return 0

    with mock.patch.object(
        hip_module, "_hipDeviceGetAttribute", side_effect=set_revision
    ):
        assert hip_module._get_device_asic_revision(3) == 0


@pytest.mark.parametrize(
    "failure", [1, OSError("HIP unavailable"), hip_module.HipError("HIP unavailable")]
)
def test_get_device_asic_revision_returns_none_on_error(
    failure: int | OSError | hip_module.HipError,
) -> None:
    replacement = (
        mock.Mock(return_value=failure)
        if isinstance(failure, int)
        else mock.Mock(side_effect=failure)
    )
    with mock.patch.object(hip_module, "_hipDeviceGetAttribute", replacement):
        assert hip_module._get_device_asic_revision() is None


def test_device_properties_retry_after_hip_library_becomes_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hip_module, "_device_props_cache", {})
    device = 3
    unavailable = mock.Mock(side_effect=hip_module.HipError("HIP unavailable"))

    with mock.patch.object(hip_module, "_b", return_value=unavailable):
        assert hip_module._device_props(device) is None

    def available(buffer, queried_device):
        assert queried_device == device
        buffer[0] = ord("x")
        return 0

    resolved = mock.Mock(side_effect=available)
    with mock.patch.object(hip_module, "_b", return_value=resolved):
        props = hip_module._device_props(device)

    assert props is not None
    assert props.startswith(b"x")
    assert resolved.call_count == 1


@pytest.mark.parametrize(
    ("target_id", "base_arch", "compiler_target", "revision"),
    [
        ("gfx1250-strict", "gfx1250", "gfx1250", 0),
        ("gfx942:sramecc+:xnack-", "gfx942", "gfx942:sramecc+:xnack-", 1),
        (None, None, None, None),
    ],
)
def test_get_device_info(
    target_id: str | None,
    base_arch: str | None,
    compiler_target: str | None,
    revision: int | None,
) -> None:
    with (
        mock.patch.object(
            device_info, "get_device_target_id", return_value=target_id
        ) as query_target,
        mock.patch.object(
            device_info, "_get_device_asic_revision", return_value=revision
        ) as query_revision,
    ):
        info = device_info.get_device_info(4)

    query_target.assert_called_once_with(4)
    query_revision.assert_called_once_with(4)
    assert info == device_info.DeviceInfo(target_id=target_id, asic_revision=revision)
    assert info.base_arch == base_arch
    assert info.compiler_target == compiler_target


def test_runtime_exports_device_info_api() -> None:
    assert runtime.DeviceInfo is device_info.DeviceInfo
    assert runtime.get_device_info is device_info.get_device_info
    assert "DeviceInfo" in runtime.__all__
    assert "get_device_info" in runtime.__all__
