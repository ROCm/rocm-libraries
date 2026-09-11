# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Explicit gfx950 LDS conflict profile."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..model import ProfileIdentity
from ..opcodes import get_opcode_spec
from .gfx90a import Gfx90aProfile

__all__ = ["GFX950_PROFILE", "Gfx950Profile"]


@dataclass(frozen=True)
class Gfx950Profile(Gfx90aProfile):
    """Public wave64 prediction rules for the supported gfx950 LDS opcodes."""

    identity: ProfileIdentity = field(
        default_factory=lambda: ProfileIdentity(target="gfx950", profile_version=1)
    )

    def phase_key(self, opcode: str, lane: int) -> tuple[int, ...]:
        """Return the observed lane phase for one canonical opcode."""

        spec = get_opcode_spec(opcode)
        bit5 = (lane >> 5) & 1
        if spec.direction == "read" and spec.access_width_bytes == 8:
            return (bit5,)
        if spec.direction == "read" and spec.access_width_bytes == 16:
            bit4 = (lane >> 4) & 1
            bit3 = (lane >> 3) & 1
            bit2 = (lane >> 2) & 1
            return (bit5, bit4 ^ bit3 ^ bit2)
        return super().phase_key(opcode, lane)

    def collision_key(self, opcode: str, lds_byte_address: int) -> int:
        """Return the observed repeating byte-address class for one access."""

        spec = get_opcode_spec(opcode)
        if spec.direction == "read" and spec.access_width_bytes in {8, 16}:
            return lds_byte_address % 256
        return super().collision_key(opcode, lds_byte_address)


GFX950_PROFILE = Gfx950Profile()
