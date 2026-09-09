# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""HIP device properties and the target names rocKE derives from them."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.arch import base_arch_from_target_id, compiler_target_from_target_id
from ._hip_device_properties import HipDevicePropR0600
from .hip_module import _device_props


@dataclass(frozen=True, eq=False)
class DeviceInfo:
    """HIP device properties with derived names for compilation and lowering.

    ``target_id`` is the string read from HIP's ``gcnArchName`` property,
    such as ``gfx1250-strict`` or ``gfx942:sramecc+:xnack-``.

    Stores a private copy of the HIP properties result. ``target_id``,
    ``asic_revision``, ``base_arch``, and ``compiler_target`` are read-only
    properties. Later queries do not change this snapshot.

    ``base_arch`` is derived by
    :func:`~rocke.core.arch.base_arch_from_target_id`. rocKE uses this name
    with :meth:`~rocke.core.arch.ArchTarget.from_gfx` for catalog lookup.

    ``compiler_target`` is derived by
    :func:`~rocke.core.arch.compiler_target_from_target_id`. The compile
    helpers use this name in the COMGR ISA name or hipcc's ``--offload-arch``.

    ``asic_revision`` comes from ``asicRevision`` in the same HIP properties
    result. A successful query may return zero.
    """

    _properties: HipDevicePropR0600 | None = field(repr=False)

    def __post_init__(self) -> None:
        if self._properties is not None:
            object.__setattr__(
                self,
                "_properties",
                HipDevicePropR0600.from_buffer_copy(self._properties),
            )

    @property
    def target_id(self) -> str | None:
        """Target ID from HIP's ``gcnArchName``, or ``None`` if unavailable."""
        if self._properties is None:
            return None
        return self._properties.gcnArchName.decode("ascii", "replace") or None

    @property
    def asic_revision(self) -> int | None:
        """HIP's ``asicRevision`` value, including zero, or ``None``."""
        if self._properties is None:
            return None
        revision = self._properties.asicRevision
        return revision if revision >= 0 else None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DeviceInfo):
            return NotImplemented
        return (self.target_id, self.asic_revision) == (
            other.target_id,
            other.asic_revision,
        )

    def __hash__(self) -> int:
        return hash((self.target_id, self.asic_revision))

    @property
    def base_arch(self) -> str | None:
        """Base architecture, or ``None`` when ``target_id`` is unavailable."""
        return (
            base_arch_from_target_id(self.target_id)
            if self.target_id is not None
            else None
        )

    @property
    def compiler_target(self) -> str | None:
        """Compiler target, or ``None`` when ``target_id`` is unavailable."""
        return (
            compiler_target_from_target_id(self.target_id)
            if self.target_id is not None
            else None
        )


def get_device_info(device: int = 0) -> DeviceInfo:
    """Read target ID and ASIC revision for a HIP device ordinal.

    Each call reads fresh properties with one ``hipGetDevicePropertiesR0600``
    call. Reading properties on the returned object makes no further HIP calls.
    A failed query returns ``None`` for both values.
    An empty target string leaves only the target names unavailable.
    """

    return DeviceInfo(_device_props(device))


__all__ = ["DeviceInfo", "get_device_info"]
