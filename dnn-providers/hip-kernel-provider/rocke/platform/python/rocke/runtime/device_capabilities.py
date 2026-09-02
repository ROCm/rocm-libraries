# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Named device capabilities reported from HIP properties.

Keep revision checks in one place so dispatch and launch code do not compare
``asicRevision`` values directly. Features tied to an unknown revision are reported
as unsupported. HIP still decides whether a launch succeeds.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .hip_module import _get_device_asic_revision, get_device_arch


class DeviceCapability(str, Enum):
    """Device features recognized by rocKE."""

    WORKGROUP_CLUSTER_LAUNCH = "workgroup_cluster_launch"
    TDM_MULTICAST = "tdm_multicast"
    MX_WMMA_FP4_32X16 = "mx_wmma_fp4_32x16"
    MX_BLOCK16_CONVERSION = "mx_block16_conversion"


_GFX1250_BASE_CAPABILITIES = frozenset({DeviceCapability.WORKGROUP_CLUSTER_LAUNCH})

_GFX1250_REVISION_CAPABILITIES = {
    0: frozenset(),
    1: frozenset(
        {
            DeviceCapability.TDM_MULTICAST,
            DeviceCapability.MX_WMMA_FP4_32X16,
            DeviceCapability.MX_BLOCK16_CONVERSION,
        }
    ),
}


@dataclass(frozen=True)
class DeviceCapabilities:
    """Capability result for one HIP device.

    ``check`` returns a short explanation when a feature is not supported. Callers
    do not need to compare ASIC revision numbers themselves.
    """

    arch: str | None
    asic_revision: int | None
    supported: frozenset[DeviceCapability]

    def check(self, capability: DeviceCapability) -> tuple[bool, str]:
        """Return whether ``capability`` is supported and explain the result."""

        if not isinstance(capability, DeviceCapability):
            raise TypeError(
                "capability must be DeviceCapability, "
                f"got {type(capability).__name__}"
            )
        if capability in self.supported:
            return True, "supported"
        if self.arch is None:
            return False, "device architecture is unavailable"
        if self.arch != "gfx1250":
            return (
                False,
                f"no revision-specific capabilities are listed for {self.arch}",
            )
        if self.asic_revision is None:
            return False, "gfx1250 ASIC revision is unavailable"
        if self.asic_revision < 0:
            return False, f"invalid gfx1250 ASIC revision {self.asic_revision}"
        if self.asic_revision not in _GFX1250_REVISION_CAPABILITIES:
            return False, f"unknown gfx1250 ASIC revision {self.asic_revision}"
        return False, (
            f"{capability.value} is unavailable on "
            f"gfx1250 ASIC revision {self.asic_revision}"
        )

    def supports(self, capability: DeviceCapability) -> bool:
        """Return whether ``capability`` is supported."""

        return self.check(capability)[0]


def _capabilities_for_properties(
    arch: str | None, asic_revision: int | None
) -> DeviceCapabilities:
    """Build a capability result from device properties already read from HIP."""

    if arch == "gfx1250":
        supported = _GFX1250_BASE_CAPABILITIES | (
            _GFX1250_REVISION_CAPABILITIES.get(asic_revision, frozenset())
        )
    else:
        supported = frozenset()
    return DeviceCapabilities(
        arch=arch,
        asic_revision=asic_revision,
        supported=supported,
    )


def get_device_capabilities(device: int = 0) -> DeviceCapabilities:
    """Query one HIP device and return its named capabilities."""

    return _capabilities_for_properties(
        get_device_arch(device),
        _get_device_asic_revision(device),
    )


__all__ = [
    "DeviceCapabilities",
    "DeviceCapability",
    "get_device_capabilities",
]
