# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""HIP device properties and the target names rocKE derives from them."""

from __future__ import annotations

from dataclasses import dataclass

from ..core.arch import base_arch_from_target_id, compiler_target_from_target_id
from .hip_module import _get_device_asic_revision, get_device_target_id


@dataclass(frozen=True)
class DeviceInfo:
    """HIP device properties with derived names for compilation and lowering.

    ``target_id`` is the string read from HIP's ``gcnArchName`` property,
    such as ``gfx1250-strict`` or ``gfx942:sramecc+:xnack-``.

    Only ``target_id`` and ``asic_revision`` are stored. ``base_arch`` and
    ``compiler_target`` are read-only properties computed from ``target_id``.

    ``base_arch`` is derived by
    :func:`~rocke.core.arch.base_arch_from_target_id`. rocKE uses this name
    with :meth:`~rocke.core.arch.ArchTarget.from_gfx` for catalog lookup.

    ``compiler_target`` is derived by
    :func:`~rocke.core.arch.compiler_target_from_target_id`. The compile
    helpers use this name in the COMGR ISA name or hipcc's ``--offload-arch``.

    ``asic_revision`` comes from HIP's ``hipDeviceAttributeAsicRevision``.
    A successful query may return zero.
    """

    target_id: str | None
    asic_revision: int | None

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

    Uses :func:`get_device_target_id` and :func:`_get_device_asic_revision`.
    If the target ID is unavailable, all three target names are ``None``.
    The revision query is independent and returns ``None`` on failure.
    """

    return DeviceInfo(
        target_id=get_device_target_id(device),
        asic_revision=_get_device_asic_revision(device),
    )


__all__ = ["DeviceInfo", "get_device_info"]
