# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Device feature rules applied to a HIP properties snapshot.

Instruction features come from architecture and revision rules. Launch support
comes from HIP. Compiler support and kernel eligibility need separate checks.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType

from .device_info import DeviceInfo


class DeviceCapability(Enum):
    """Individual features recognized by rocKE.

    TDM multicast is separate from TDM loads, clusters, and cluster barriers.
    FP4 WMMA dimensions refer to the physical instruction, independently of
    logical dimensions after operand swapping. MX block16 conversion refers to
    packed-scale conversion modes that share a scale across 16 elements.
    """

    WORKGROUP_CLUSTER_LAUNCH = "workgroup_cluster_launch"
    TDM_MULTICAST = "tdm_multicast"
    MX_WMMA_FP4_32X16 = "mx_wmma_fp4_32x16"
    MX_BLOCK16_CONVERSION = "mx_block16_conversion"


@dataclass(frozen=True)
class DeviceCapabilities:
    """Immutable feature facts, with absent entries representing unknown support.

    Owns a copy of the supplied mapping. Entries must use ``DeviceCapability``
    keys and boolean values; omit a feature when its support is unknown.
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
        """Return True, False, or None when support is unknown.

        Callers requiring a feature should check ``is True``. Passing a value
        other than a ``DeviceCapability`` raises ``TypeError``.
        """
        if not isinstance(capability, DeviceCapability):
            raise TypeError("capability must be a DeviceCapability")
        return self._support.get(capability)


@dataclass(frozen=True)
class _ArchitecturePolicy:
    """Facts independent of revision, plus overrides for exact revisions."""

    common: Mapping[DeviceCapability, bool] = field(default_factory=dict)
    by_revision: Mapping[int, Mapping[DeviceCapability, bool]] = field(
        default_factory=dict
    )


# Follows hipBLASLt's multicast/WMMA rules and CK's block16 conversion gate.
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

    Makes no HIP calls and does not modify ``info``. Unknown revisions retain
    architecture-common facts but receive no revision overrides. HIP alone
    supplies cluster-launch support, even when the architecture is unlisted.
    Any feature without an applicable fact remains unknown.
    """
    policy = _CAPABILITY_POLICIES.get(info.base_arch)
    values = dict(policy.common) if policy is not None else {}
    if policy is not None and info.asic_revision is not None:
        values.update(policy.by_revision.get(info.asic_revision, {}))

    # An architecture rule must not enable a launch that HIP does not report.
    values.pop(DeviceCapability.WORKGROUP_CLUSTER_LAUNCH, None)
    if info.cluster_launch is not None:
        values[DeviceCapability.WORKGROUP_CLUSTER_LAUNCH] = info.cluster_launch
    return DeviceCapabilities(values)


__all__ = ["DeviceCapabilities", "DeviceCapability", "infer_device_capabilities"]
