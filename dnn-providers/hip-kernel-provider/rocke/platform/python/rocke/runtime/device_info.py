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

    ``base_arch`` is derived from ``target_id`` by
    :func:`~rocke.core.arch.base_arch_from_target_id`. rocKE uses this name
    with :meth:`~rocke.core.arch.ArchTarget.from_gfx` for catalog lookup.

    ``compiler_target`` is derived by
    :func:`~rocke.core.arch.compiler_target_from_target_id`. The compile
    helpers use this name in the COMGR ISA name or hipcc's ``--offload-arch``.

    ``asic_revision`` comes from HIP's ``hipDeviceAttributeAsicRevision``.
    A successful query may return zero.
    """

    target_id: str | None
    base_arch: str | None
    compiler_target: str | None
    asic_revision: int | None


def get_device_info(device: int = 0) -> DeviceInfo:
    """Read properties for a HIP device ordinal and derive its target names.

    Uses :func:`get_device_target_id` and :func:`_get_device_asic_revision`.
    If the target ID is unavailable, all three target names are ``None``.
    The revision query is independent and returns ``None`` on failure.
    """

    target_id = get_device_target_id(device)
    return DeviceInfo(
        target_id=target_id,
        base_arch=(
            base_arch_from_target_id(target_id) if target_id is not None else None
        ),
        compiler_target=(
            compiler_target_from_target_id(target_id) if target_id is not None else None
        ),
        asic_revision=_get_device_asic_revision(device),
    )


__all__ = ["DeviceInfo", "get_device_info"]
