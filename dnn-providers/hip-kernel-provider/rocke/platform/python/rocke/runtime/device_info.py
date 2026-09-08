# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Identity facts reported for a HIP device.

Static instruction facts remain in :class:`rocke.core.arch.ArchTarget`, while
the destination runtime determines code-object loading and compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.arch import base_arch_from_target_id
from .hip_module import _get_device_asic_revision, get_device_target_id


@dataclass(frozen=True)
class DeviceInfo:
    """Identity reported by HIP for one device.

    ``target_id`` preserves the complete runtime/compiler target string,
    including profiles and feature suffixes such as
    ``gfx1250-strict:sramecc+:xnack-``.

    ``base_arch`` is the normalized rocKE architecture used for static
    :class:`~rocke.core.arch.ArchTarget` lookup. It intentionally omits target
    profiles and feature suffixes.

    ``asic_revision`` is the revision value reported by HIP.
    """

    target_id: str | None
    base_arch: str | None
    asic_revision: int | None


def get_device_info(device: int = 0) -> DeviceInfo:
    """Query identity properties for one HIP device.

    Properties that HIP cannot report remain ``None``.
    """

    target_id = get_device_target_id(device)
    return DeviceInfo(
        target_id=target_id,
        base_arch=(
            base_arch_from_target_id(target_id) if target_id is not None else None
        ),
        asic_revision=_get_device_asic_revision(device),
    )


__all__ = ["DeviceInfo", "get_device_info"]
