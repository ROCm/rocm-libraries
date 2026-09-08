# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Runtime identity reported for a HIP device.

This module reports facts obtained from the destination runtime. It does not
describe the static instruction catalog and does not predict whether an
arbitrary code object can be translated or loaded.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.arch import base_arch_from_target_id
from .hip_module import _get_device_asic_revision, get_device_target_id


@dataclass(frozen=True)
class RuntimeDeviceInfo:
    """Runtime identity for one HIP device.

    ``target_id`` preserves the complete runtime/compiler target string,
    including profiles and feature suffixes such as
    ``gfx1250-strict:sramecc+:xnack-``.

    ``base_arch`` is the normalized rocKE architecture used for static
    :class:`~rocke.core.arch.ArchTarget` lookup. It intentionally omits target
    profiles and feature suffixes.

    ``asic_revision`` is the runtime-reported revision. It carries no
    instruction-support or artifact-compatibility guarantee.
    """

    target_id: str | None
    base_arch: str | None
    asic_revision: int | None


def get_device_info(device: int = 0) -> RuntimeDeviceInfo:
    """Query identity properties for one HIP device.

    Unknown properties remain ``None``. In particular, an unknown property is
    not interpreted as lack of instruction or artifact compatibility.
    """

    target_id = get_device_target_id(device)
    return RuntimeDeviceInfo(
        target_id=target_id,
        base_arch=(
            base_arch_from_target_id(target_id) if target_id is not None else None
        ),
        asic_revision=_get_device_asic_revision(device),
    )


__all__ = ["RuntimeDeviceInfo", "get_device_info"]
