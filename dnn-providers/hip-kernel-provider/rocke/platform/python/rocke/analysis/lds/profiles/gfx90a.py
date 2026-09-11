# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Explicit gfx90a LDS conflict profile."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..model import ProfileIdentity
from ..opcodes import get_opcode_spec, supported_opcodes


__all__ = ["GFX90A_PROFILE", "Gfx90aProfile"]


@dataclass(frozen=True)
class Gfx90aProfile:
    """Public wave64 prediction rules for the supported gfx90a LDS opcodes."""

    identity: ProfileIdentity = field(
        default_factory=lambda: ProfileIdentity(target="gfx90a", profile_version=1)
    )
    supported_wave_sizes: frozenset[int] = frozenset({64})
    supported_opcodes: frozenset[str] = frozenset(supported_opcodes())

    def phase_key(self, opcode: str, lane: int) -> tuple[int, ...]:
        """Return the measured lane phase for one canonical opcode."""

        spec = get_opcode_spec(opcode)
        if spec.opcode not in self.supported_opcodes:
            raise ValueError(
                f"opcode is not supported by {self.identity.target}: {opcode}"
            )

        bit5 = (lane >> 5) & 1
        bit4 = (lane >> 4) & 1
        bit3 = (lane >> 3) & 1
        bit2 = (lane >> 2) & 1
        if spec.access_width_bytes == 4:
            return (bit5,)
        if spec.access_width_bytes == 8:
            return (bit5, bit4)
        # The read and write partitions have equal cardinality but different
        # memberships, so direction remains part of the b128 rule.
        if spec.direction == "read":
            return (bit5, bit4 ^ bit2, bit3)
        return (bit5, bit4, bit3)

    def collision_key(self, opcode: str, lds_byte_address: int) -> int:
        """Return the repeating byte-address class for a supported access."""

        spec = get_opcode_spec(opcode)
        if spec.opcode not in self.supported_opcodes:
            raise ValueError(
                f"opcode is not supported by {self.identity.target}: {opcode}"
            )
        return lds_byte_address % 128


GFX90A_PROFILE = Gfx90aProfile()
