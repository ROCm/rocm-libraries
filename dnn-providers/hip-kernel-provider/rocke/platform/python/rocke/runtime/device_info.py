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
    """Identity of one HIP device without dispatch or compatibility policy."""

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
