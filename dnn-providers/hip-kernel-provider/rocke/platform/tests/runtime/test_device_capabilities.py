# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from unittest import mock

import pytest

from rocke import runtime
from rocke.runtime import device_capabilities
from rocke.runtime import hip_module
from rocke.runtime.device_capabilities import DeviceCapability


def test_get_device_asic_revision_queries_stable_attribute() -> None:
    def set_revision(out, attribute, device):
        assert attribute == 10012
        assert device == 3
        out._obj.value = 7
        return 0

    with mock.patch.object(
        hip_module, "_hipDeviceGetAttribute", side_effect=set_revision
    ):
        assert hip_module._get_device_asic_revision(3) == 7


@pytest.mark.parametrize(
    "failure", [1, OSError("HIP unavailable"), hip_module.HipError("HIP unavailable")]
)
def test_get_device_asic_revision_returns_none_on_error(
    failure: int | OSError | hip_module.HipError,
) -> None:
    if isinstance(failure, int):
        replacement = mock.Mock(return_value=failure)
    else:
        replacement = mock.Mock(side_effect=failure)
    with mock.patch.object(hip_module, "_hipDeviceGetAttribute", replacement):
        assert hip_module._get_device_asic_revision() is None


def test_device_properties_return_none_when_hip_library_is_unavailable() -> None:
    device = 37
    hip_module._device_props_cache.pop(device, None)
    unavailable = mock.Mock(side_effect=hip_module.HipError("HIP unavailable"))

    with mock.patch.object(hip_module, "_b", return_value=unavailable):
        assert hip_module._device_props(device) is None


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


def test_revision_zero_preserves_cluster_launch_but_rejects_known_deltas() -> None:
    capabilities = device_capabilities._capabilities_for_properties("gfx1250", 0)

    assert capabilities.supported == frozenset(
        {DeviceCapability.WORKGROUP_CLUSTER_LAUNCH}
    )
    assert capabilities.supports(DeviceCapability.WORKGROUP_CLUSTER_LAUNCH)
    for capability in (
        DeviceCapability.TDM_MULTICAST,
        DeviceCapability.MX_WMMA_FP4_32X16,
        DeviceCapability.MX_BLOCK16_CONVERSION,
    ):
        ok, reason = capabilities.check(capability)
        assert not ok
        assert "revision 0" in reason


def test_later_revision_has_verified_capabilities() -> None:
    capabilities = device_capabilities._capabilities_for_properties("gfx1250", 1)

    assert capabilities.supported == frozenset(DeviceCapability)
    for capability in DeviceCapability:
        assert capabilities.supports(capability)
        assert capabilities.check(capability) == (True, "supported")


def test_unknown_later_revision_keeps_only_target_capabilities() -> None:
    capabilities = device_capabilities._capabilities_for_properties("gfx1250", 2)

    assert capabilities.supported == frozenset(
        {DeviceCapability.WORKGROUP_CLUSTER_LAUNCH}
    )
    assert capabilities.supports(DeviceCapability.WORKGROUP_CLUSTER_LAUNCH)
    ok, reason = capabilities.check(DeviceCapability.TDM_MULTICAST)
    assert not ok
    assert reason == "unknown gfx1250 ASIC revision 2"


@pytest.mark.parametrize(
    ("arch", "revision", "reason"),
    [
        (None, None, "architecture is unavailable"),
        ("gfx1250", None, "ASIC revision is unavailable"),
        ("gfx1250", -1, "invalid gfx1250 ASIC revision"),
        ("gfx950", 1, "no capabilities are listed for gfx950"),
    ],
)
def test_unknown_properties_reject_revision_specific_capabilities(
    arch: str | None, revision: int | None, reason: str
) -> None:
    capabilities = device_capabilities._capabilities_for_properties(arch, revision)

    ok, actual_reason = capabilities.check(DeviceCapability.TDM_MULTICAST)
    assert not ok
    assert reason in actual_reason


@pytest.mark.parametrize("revision", [None, -1])
def test_unavailable_revision_preserves_target_level_capabilities(
    revision: int | None,
) -> None:
    capabilities = device_capabilities._capabilities_for_properties("gfx1250", revision)

    assert capabilities.supported == frozenset(
        {DeviceCapability.WORKGROUP_CLUSTER_LAUNCH}
    )
    assert capabilities.check(DeviceCapability.WORKGROUP_CLUSTER_LAUNCH) == (
        True,
        "supported",
    )


def test_policy_registry_can_add_another_target() -> None:
    policy = device_capabilities._CapabilityPolicy(
        base=frozenset({DeviceCapability.TDM_MULTICAST}),
        by_revision={
            7: frozenset({DeviceCapability.MX_BLOCK16_CONVERSION}),
        },
    )

    with mock.patch.dict(
        device_capabilities._CAPABILITY_POLICIES,
        {"test-arch": policy},
    ):
        capabilities = device_capabilities._capabilities_for_properties("test-arch", 7)

    assert capabilities.supported == frozenset(
        {
            DeviceCapability.TDM_MULTICAST,
            DeviceCapability.MX_BLOCK16_CONVERSION,
        }
    )


def test_get_device_capabilities_queries_both_properties() -> None:
    with (
        mock.patch.object(
            device_capabilities, "get_device_arch", return_value="gfx1250"
        ) as arch,
        mock.patch.object(
            device_capabilities, "_get_device_asic_revision", return_value=1
        ) as revision,
    ):
        capabilities = device_capabilities.get_device_capabilities(4)

    arch.assert_called_once_with(4)
    revision.assert_called_once_with(4)
    assert capabilities.supported == frozenset(DeviceCapability)


def test_runtime_exports_capability_api() -> None:
    assert runtime.DeviceCapability is DeviceCapability
    assert runtime.DeviceCapabilities is device_capabilities.DeviceCapabilities
    assert (
        runtime.get_device_capabilities is device_capabilities.get_device_capabilities
    )
    assert "capabilities_for_properties" not in runtime.__all__
    assert not hasattr(runtime, "capabilities_for_properties")
    assert not hasattr(runtime, "get_device_asic_revision")


def test_check_rejects_untyped_capability() -> None:
    capabilities = device_capabilities._capabilities_for_properties("gfx1250", 1)

    with pytest.raises(TypeError, match="must be DeviceCapability"):
        capabilities.check("tdm_multicast")
