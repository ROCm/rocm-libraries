# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest

from rocke.core.arch import known_arches
from rocke.runtime import DeviceCapabilities, DeviceInfo, infer_device_capabilities
from rocke.runtime import device_info
from rocke.runtime._hip_device_properties import HipDevicePropR0600


def _info(target_id: str, revision: int) -> DeviceInfo:
    props = HipDevicePropR0600()
    props.gcnArchName = target_id.encode("ascii")
    props.asicRevision = revision
    return DeviceInfo(props)


@pytest.mark.parametrize("target_id", ["gfx1250", "gfx1250-strict"])
@pytest.mark.parametrize(("revision", "supported"), [(0, False), (1, True)])
def test_gfx1250_revision_features(target_id: str, revision: int, supported: bool):
    caps = infer_device_capabilities(_info(target_id, revision))
    assert caps.has_tdm_multicast is supported
    assert caps.has_fp4_wmma_32x16 is supported


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
    assert caps.has_tdm_multicast is None
    assert caps.has_fp4_wmma_32x16 is None


@pytest.mark.parametrize("arch", [arch for arch in known_arches() if arch != "gfx1250"])
@pytest.mark.parametrize("revision", [0, 1])
def test_revision_rules_do_not_apply_to_other_architectures(arch: str, revision: int):
    assert infer_device_capabilities(_info(arch, revision)) == DeviceCapabilities()


def test_inference_reuses_snapshot_without_querying_hip(monkeypatch):
    info = _info("gfx1250-strict", 1)

    def unexpected_query(*args, **kwargs):
        pytest.fail("Capability inference must not query HIP")

    monkeypatch.setattr(device_info, "_device_props", unexpected_query)
    caps = infer_device_capabilities(info)
    assert infer_device_capabilities(info) == caps
    assert info.target_id == "gfx1250-strict"
    assert info.compiler_target == "gfx1250"
    assert info.asic_revision == 1
