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
        ("gfx942", "gfx942"),
        ("gfx950", "gfx950"),
        ("gfx1151", "gfx1151"),
        ("gfx1201", "gfx1201"),
        ("gfx1250", "gfx1250"),
        ("gfx11-generic", "gfx11-generic"),
        ("gfx1250-strict", "gfx1250"),
        ("gfx942:sramecc+:xnack-", "gfx942"),
    ],
)
def test_device_target_id_and_base_arch_are_separate(
    target_id: str, base_arch: str
) -> None:
    with mock.patch.object(hip_module, "_device_props", return_value=_props(target_id)):
        assert hip_module.get_device_target_id(3) == target_id
        assert hip_module.get_device_arch(3) == base_arch


def test_missing_target_id_remains_unknown() -> None:
    with mock.patch.object(hip_module, "_device_props", return_value=_props(None)):
        assert hip_module.get_device_target_id() is None
        assert hip_module.get_device_arch() is None


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


def test_device_properties_retry_after_hip_library_becomes_available() -> None:
    device = 38
    hip_module._device_props_cache.pop(device, None)
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


def test_get_device_info_reports_identity_without_capability_policy() -> None:
    with (
        mock.patch.object(
            device_info, "get_device_target_id", return_value="gfx1250-strict"
        ) as target_id,
        mock.patch.object(
            device_info, "_get_device_asic_revision", return_value=0
        ) as revision,
    ):
        info = device_info.get_device_info(4)

    target_id.assert_called_once_with(4)
    revision.assert_called_once_with(4)
    assert info == device_info.DeviceInfo(
        target_id="gfx1250-strict",
        base_arch="gfx1250",
        compiler_target="gfx1250",
        asic_revision=0,
    )
    assert not hasattr(info, "supported")


def test_get_device_info_preserves_unknown_properties() -> None:
    with (
        mock.patch.object(device_info, "get_device_target_id", return_value=None),
        mock.patch.object(device_info, "_get_device_asic_revision", return_value=None),
    ):
        info = device_info.get_device_info()

    assert info == device_info.DeviceInfo(
        target_id=None,
        base_arch=None,
        compiler_target=None,
        asic_revision=None,
    )


def test_runtime_exports_device_info_api() -> None:
    assert runtime.DeviceInfo is device_info.DeviceInfo
    assert runtime.get_device_info is device_info.get_device_info
    assert "DeviceInfo" in runtime.__all__
    assert "get_device_info" in runtime.__all__
    assert not hasattr(runtime, "DeviceCapability")
    assert not hasattr(runtime, "DeviceCapabilities")
    assert not hasattr(runtime, "get_device_capabilities")
