# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Explicit gfx1201 wave32 active-pair LDS conflict profile."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ..model import LdsAccess, ProfileIdentity
from ..opcodes import get_opcode_spec

__all__ = ["GFX1201_PROFILE", "Gfx1201Profile"]


@dataclass(frozen=True)
class Gfx1201Profile:
    """Aligned, distinct-address pair rules independent of the gfx9 profiles.

    The address class and lane phases describe empirical conflict equivalence,
    not physical banks. Broadcasts and multi-lane composition are unsupported.
    Dual-address support is limited to the documented fixed operand spacing.
    """

    identity: ProfileIdentity = field(
        default_factory=lambda: ProfileIdentity(target="gfx1201", profile_version=2)
    )
    supported_wave_sizes: frozenset[int] = frozenset({32})
    lds_capacity_bytes: int = 65536
    supported_opcodes: frozenset[str] = frozenset(
        {
            "ds_read_b32",
            "ds_read_b64",
            "ds_read_b128",
            "ds_write_b32",
            "ds_write_b64",
            "ds_write_b128",
            "ds_load_b32",
            "ds_load_b64",
            "ds_load_b96",
            "ds_load_b128",
            "ds_store_b32",
            "ds_store_b64",
            "ds_store_b96",
            "ds_store_b128",
            "ds_load_u8",
            "ds_load_i8",
            "ds_load_u16",
            "ds_load_i16",
            "ds_store_b8",
            "ds_store_b16",
        }
    )

    # Explicit admission keeps global opcode metadata from expanding this profile.
    supported_opcodes = supported_opcodes | frozenset(
        f"ds_{verb}{suffix}_b{bits}"
        for verb, suffix in (
            ("load", "_2addr"),
            ("load", "_2addr_stride64"),
            ("store", "_2addr"),
            ("store", "_2addr_stride64"),
            ("read", "2"),
            ("read", "2st64"),
            ("write", "2"),
            ("write", "2st64"),
        )
        for bits in (32, 64)
    )

    def _check_opcode(self, opcode: str) -> None:
        if get_opcode_spec(opcode).opcode not in self.supported_opcodes:
            raise ValueError(
                f"opcode is not supported by {self.identity.target}: {opcode}"
            )

    def phase_key(self, opcode: str, lane: int) -> tuple[int, ...]:
        """Return the contiguous wave32 phase for one supported opcode."""

        self._check_opcode(opcode)
        width = get_opcode_spec(opcode).access_width_bytes
        if width <= 4:
            return (0,)
        if width == 8:
            return (lane >> 4,)
        return (lane >> 3,)

    def collision_key(self, opcode: str, lds_byte_address: int) -> int:
        """Return the 128-byte repeating address class for an aligned access."""

        self._check_opcode(opcode)
        return lds_byte_address % 128

    def validate_request(
        self, opcode: str, accesses: Sequence[LdsAccess]
    ) -> tuple[bool, str]:
        """Reject requests that would inherit unverified grouping semantics."""

        self._check_opcode(opcode)
        spec = get_opcode_spec(opcode)
        active = [access for access in accesses if access.active]
        lanes = {access.lane for access in active}
        if len(lanes) > 2:
            return False, "gfx1201 supports at most two active lanes"
        if spec.address_count == 2:
            stride64 = "stride64" in opcode or "st64" in opcode
            spacing = spec.access_width_bytes * 64 if stride64 else 64
            for lane in lanes:
                operands = {a.operand_index: a for a in active if a.lane == lane}
                if set(operands) != {0, 1}:
                    return (
                        False,
                        "dual-address lanes require both active address operands",
                    )
                if (
                    operands[1].lds_byte_address - operands[0].lds_byte_address
                    != spacing
                ):
                    return (
                        False,
                        f"gfx1201 requires {spacing}-byte address operand spacing",
                    )
        for operand in range(spec.address_count):
            members = [a for a in active if a.operand_index == operand]
            if len(members) == 2:
                distance = abs(
                    members[0].lds_byte_address - members[1].lds_byte_address
                )
                if distance < spec.access_width_bytes:
                    return (
                        False,
                        "gfx1201 requires distinct, non-overlapping addresses per operand",
                    )

        return True, "ok"


GFX1201_PROFILE = Gfx1201Profile()
