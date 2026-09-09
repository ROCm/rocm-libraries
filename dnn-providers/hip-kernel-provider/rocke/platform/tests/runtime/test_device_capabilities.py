# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest

from rocke.core.arch import known_arches
from rocke.runtime import (
    DeviceCapabilities,
    DeviceCapability,
    DeviceInfo,
    infer_device_capabilities,
)
from rocke.runtime import device_capabilities, device_info
from rocke.runtime._hip_device_properties import HipDevicePropR0600


def _info(target_id: str, revision: int, cluster_launch: int = 0) -> DeviceInfo:
    props = HipDevicePropR0600()
    props.gcnArchName = target_id.encode("ascii")
    props.asicRevision = revision
    props.clusterLaunch = cluster_launch
    return DeviceInfo(props)


@pytest.mark.parametrize("target_id", ["gfx1250", "gfx1250-strict"])
@pytest.mark.parametrize(("revision", "supported"), [(0, False), (1, True)])
def test_gfx1250_revision_features(target_id: str, revision: int, supported: bool):
    caps = infer_device_capabilities(_info(target_id, revision))
    assert caps.support(DeviceCapability.TDM_MULTICAST) is supported
    assert caps.support(DeviceCapability.MX_WMMA_FP4_32X16) is supported
    assert caps.support(DeviceCapability.MX_BLOCK16_CONVERSION) is supported


@pytest.mark.parametrize(
    "info",
    [
        DeviceInfo(None),
        _info("", 0),
        _info("gfx1250", -1),
        _info("gfx1250", 2),
        _info("unrecognized-target", 1),
    ],
)
def test_missing_or_unlisted_identity_has_unknown_support(info: DeviceInfo):
    caps = infer_device_capabilities(info)
    assert caps.support(DeviceCapability.TDM_MULTICAST) is None
    assert caps.support(DeviceCapability.MX_WMMA_FP4_32X16) is None
    assert caps.support(DeviceCapability.MX_BLOCK16_CONVERSION) is None


@pytest.mark.parametrize("arch", [arch for arch in known_arches() if arch != "gfx1250"])
@pytest.mark.parametrize("revision", [0, 1])
def test_revision_rules_do_not_apply_to_other_architectures(arch: str, revision: int):
    caps = infer_device_capabilities(_info(arch, revision))
    assert caps.support(DeviceCapability.TDM_MULTICAST) is None
    assert caps.support(DeviceCapability.MX_WMMA_FP4_32X16) is None
    assert caps.support(DeviceCapability.MX_BLOCK16_CONVERSION) is None


@pytest.mark.parametrize("revision", [-1, 0, 2])
@pytest.mark.parametrize("cluster_launch", [0, 1])
def test_common_facts_revision_overrides_and_hip_authority(
    monkeypatch, revision: int, cluster_launch: int
):
    policy = device_capabilities._ArchitecturePolicy(
        common={
            DeviceCapability.TDM_MULTICAST: True,
            DeviceCapability.MX_WMMA_FP4_32X16: False,
            DeviceCapability.WORKGROUP_CLUSTER_LAUNCH: True,
        },
        by_revision={
            0: {
                DeviceCapability.TDM_MULTICAST: False,
                DeviceCapability.MX_WMMA_FP4_32X16: True,
            }
        },
    )
    monkeypatch.setitem(device_capabilities._CAPABILITY_POLICIES, "gfx950", policy)
    caps = infer_device_capabilities(_info("gfx950", revision, cluster_launch))
    assert caps.support(DeviceCapability.TDM_MULTICAST) is (revision != 0)
    assert caps.support(DeviceCapability.MX_WMMA_FP4_32X16) is (revision == 0)
    assert caps.support(DeviceCapability.WORKGROUP_CLUSTER_LAUNCH) is bool(
        cluster_launch
    )
    assert caps.support(DeviceCapability.MX_BLOCK16_CONVERSION) is None
    # Applying overrides must not change the common policy for the next device.
    assert policy.common[DeviceCapability.TDM_MULTICAST] is True


@pytest.mark.parametrize("target_id", ["gfx1250", "unrecognized-target", ""])
@pytest.mark.parametrize("cluster_launch", [0, 1])
def test_cluster_launch_uses_hip_even_without_known_identity(
    target_id: str, cluster_launch: int
):
    caps = infer_device_capabilities(_info(target_id, -1, cluster_launch))
    assert caps.support(DeviceCapability.WORKGROUP_CLUSTER_LAUNCH) is bool(
        cluster_launch
    )


def test_failed_query_leaves_every_capability_unknown():
    caps = infer_device_capabilities(DeviceInfo(None))
    assert all(caps.support(capability) is None for capability in DeviceCapability)


def test_capabilities_own_immutable_facts():
    facts = {DeviceCapability.TDM_MULTICAST: True}
    caps = DeviceCapabilities(facts)
    facts[DeviceCapability.TDM_MULTICAST] = False
    assert caps.support(DeviceCapability.TDM_MULTICAST) is True
    with pytest.raises(TypeError):
        caps._support[DeviceCapability.TDM_MULTICAST] = False
    with pytest.raises(AttributeError):
        caps._support = facts


@pytest.mark.parametrize(
    "facts", [{"tdm_multicast": True}, {DeviceCapability.TDM_MULTICAST: None}]
)
def test_capability_facts_require_named_features_and_boolean_values(facts):
    with pytest.raises(TypeError):
        DeviceCapabilities(facts)


def test_support_rejects_unnamed_capability():
    with pytest.raises(TypeError):
        DeviceCapabilities().support("tdm_multicast")


def test_inference_reuses_snapshot_without_querying_hip(monkeypatch):
    info = _info("gfx1250-strict", 1, cluster_launch=1)

    def unexpected_query(*args, **kwargs):
        pytest.fail("Capability inference must not query HIP")

    monkeypatch.setattr(device_info, "_device_props", unexpected_query)
    caps = infer_device_capabilities(info)
    assert infer_device_capabilities(info) == caps
    assert info.target_id == "gfx1250-strict"
    assert info.compiler_target == "gfx1250"
    assert info.asic_revision == 1
    assert caps.support(DeviceCapability.WORKGROUP_CLUSTER_LAUNCH) is True
