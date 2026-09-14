# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Device capability reports derived from a HIP properties snapshot.

HIP supplies cluster-launch support. Architecture and ASIC revision rules
describe instruction features. Runtime callers can use these reports when
choosing features for a device.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType

from .device_info import DeviceInfo


class DeviceCapability(Enum):
    """Individual features recognized by rocKE.

    Cluster launch describes HIP support for launching workgroups in a cluster.
    TDM multicast describes distributing a TDM load across a workgroup cluster.
    FP4 WMMA dimensions identify the physical instruction; operand swapping
    determines the logical matrix dimensions. MX block16 conversion describes
    packed-scale conversion modes that share a scale across 16 elements.
    """

    WORKGROUP_CLUSTER_LAUNCH = "workgroup_cluster_launch"
    TDM_MULTICAST = "tdm_multicast"
    MX_WMMA_FP4_32X16 = "mx_wmma_fp4_32x16"
    MX_BLOCK16_CONVERSION = "mx_block16_conversion"


@dataclass(frozen=True)
class DeviceCapabilities:
    """Support for named device features, stored in an immutable mapping.

    Copies the supplied mapping. Each entry pairs a ``DeviceCapability`` key
    with a boolean value. Omitted entries represent unknown support.
    """

    _support: Mapping[DeviceCapability, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = dict(self._support)
        if any(
            not isinstance(key, DeviceCapability) or not isinstance(value, bool)
            for key, value in values.items()
        ):
            raise TypeError(
                "Capability facts require DeviceCapability keys and bool values"
            )
        object.__setattr__(self, "_support", MappingProxyType(values))

    def support(self, capability: DeviceCapability) -> bool | None:
        """Return support for a ``DeviceCapability`` member.

        ``True`` means supported, ``False`` means unsupported, and ``None`` means
        unknown. Use ``is True`` to select a feature with reported support.
        Invalid argument types raise ``TypeError``.
        """
        if not isinstance(capability, DeviceCapability):
            raise TypeError("capability must be a DeviceCapability")
        return self._support.get(capability)


@dataclass(frozen=True)
class _ArchitecturePolicy:
    """Architecture defaults and overrides for exact ASIC revisions."""

    common: Mapping[DeviceCapability, bool] = field(default_factory=dict)
    by_revision: Mapping[int, Mapping[DeviceCapability, bool]] = field(
        default_factory=dict
    )


# Follows hipBLASLt's multicast/WMMA rules and CK's block16 conversion rules.
# Source links and scope are in dsl_docs/runtime/comgr_and_hipmodule.md.
_CAPABILITY_POLICIES = {
    "gfx1250": _ArchitecturePolicy(
        by_revision={
            0: {
                DeviceCapability.TDM_MULTICAST: False,
                DeviceCapability.MX_WMMA_FP4_32X16: False,
                DeviceCapability.MX_BLOCK16_CONVERSION: False,
            },
            1: {
                DeviceCapability.TDM_MULTICAST: True,
                DeviceCapability.MX_WMMA_FP4_32X16: True,
                DeviceCapability.MX_BLOCK16_CONVERSION: True,
            },
        },
    ),
}


def infer_device_capabilities(info: DeviceInfo) -> DeviceCapabilities:
    """Combine HIP feature reports with architecture and ASIC revision rules.

    Reads the supplied snapshot and applies the architecture's common rules,
    followed by overrides for an exact revision match. Missing or unlisted
    revisions use the common rules. HIP's flag supplies cluster-launch support
    for every architecture. Features with no applicable rule or runtime report
    have unknown support.
    """
    policy = _CAPABILITY_POLICIES.get(info.base_arch)
    values = dict(policy.common) if policy is not None else {}
    if policy is not None and info.asic_revision is not None:
        values.update(policy.by_revision.get(info.asic_revision, {}))

    # HIP's report determines cluster-launch support for this runtime.
    values.pop(DeviceCapability.WORKGROUP_CLUSTER_LAUNCH, None)
    if info.cluster_launch is not None:
        values[DeviceCapability.WORKGROUP_CLUSTER_LAUNCH] = info.cluster_launch
    return DeviceCapabilities(values)


__all__ = ["DeviceCapabilities", "DeviceCapability", "infer_device_capabilities"]
