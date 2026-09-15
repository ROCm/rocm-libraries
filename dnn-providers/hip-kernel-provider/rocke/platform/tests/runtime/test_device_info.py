# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from unittest import mock

import pytest

from rocke import runtime
from rocke.runtime import device_info, hip_module


def _props(target_id: str | None) -> hip_module.HipDevicePropR0600:
    props = hip_module.HipDevicePropR0600()
    props.name = b"Marketing Name"
    if target_id is not None:
        props.gcnArchName = target_id.encode("ascii")
    return props


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


@pytest.mark.parametrize(
    ("target_id", "base_arch", "compiler_target", "revision"),
    [
        ("gfx1250-strict", "gfx1250", "gfx1250", 0),
        ("gfx942:sramecc+:xnack-", "gfx942", "gfx942:sramecc+:xnack-", 1),
        (None, None, None, 0),
    ],
)
def test_get_device_info(
    target_id: str | None,
    base_arch: str | None,
    compiler_target: str | None,
    revision: int,
) -> None:
    props = _props(target_id)
    props.asicRevision = revision
    with mock.patch.object(device_info, "_device_props", return_value=props) as query:
        info = device_info.get_device_info(4)

    query.assert_called_once_with(4)
    assert info == device_info.DeviceInfo(props)
    assert info.target_id == target_id
    assert info.asic_revision == revision
    assert info.base_arch == base_arch
    assert info.compiler_target == compiler_target


def test_runtime_exports_device_info_api() -> None:
    assert runtime.DeviceInfo is device_info.DeviceInfo
    assert runtime.get_device_info is device_info.get_device_info
    assert "DeviceInfo" in runtime.__all__
    assert "get_device_info" in runtime.__all__


def test_device_properties_require_matching_function_version() -> None:
    unavailable = mock.Mock(side_effect=AttributeError("R0600 unavailable"))
    with mock.patch.object(hip_module, "_b", return_value=unavailable) as bind:
        assert hip_module._device_props(0) is None
    assert bind.call_count == 1
    assert bind.call_args.args[0] == "hipGetDevicePropertiesR0600"


def test_target_id_is_read_from_its_field() -> None:
    props = _props("gfx90a:sramecc+:xnack-")
    props.name = b"gfx942"
    with mock.patch.object(hip_module, "_device_props", return_value=props):
        assert hip_module.get_device_target_id() == "gfx90a:sramecc+:xnack-"


@pytest.mark.parametrize(
    "result",
    [1, AttributeError("R0600 unavailable"), hip_module.HipError("HIP unavailable")],
)
def test_device_info_queries_again_after_failure_or_success(
    result: int | Exception,
) -> None:
    unavailable = (
        mock.Mock(return_value=result)
        if isinstance(result, int)
        else mock.Mock(side_effect=result)
    )

    revisions = iter((0, 1))

    def fill_properties(out, device):
        assert device == 4
        out._obj.gcnArchName = b"gfx90a:sramecc+:xnack-"
        out._obj.asicRevision = next(revisions)
        return 0

    available = mock.Mock(side_effect=fill_properties)
    with (
        mock.patch.object(
            hip_module, "_b", side_effect=[unavailable, available, available]
        ),
        mock.patch.object(
            hip_module,
            "_hipDeviceGetAttribute",
            side_effect=AssertionError("DeviceInfo must use only the properties query"),
        ),
    ):
        assert device_info.get_device_info(4) == device_info.DeviceInfo(None)
        expected = device_info.DeviceInfo(_props("gfx90a:sramecc+:xnack-"))
        first = device_info.get_device_info(4)
        second = device_info.get_device_info(4)
        assert first == expected
        assert second.target_id == first.target_id
        assert second.asic_revision == 1
        assert first.asic_revision == 0
    unavailable.assert_called_once()
    assert available.call_count == 2


def test_device_info_owns_snapshot_with_read_only_properties() -> None:
    props = _props("gfx90a:sramecc+:xnack-")
    props.asicRevision = 1
    info = device_info.DeviceInfo(props)
    props.gcnArchName = b"gfx942"
    props.asicRevision = 0
    assert info.target_id == "gfx90a:sramecc+:xnack-"
    assert info.asic_revision == 1
    for name in ("target_id", "asic_revision", "base_arch", "compiler_target"):
        with pytest.raises(AttributeError):
            setattr(info, name, None)
