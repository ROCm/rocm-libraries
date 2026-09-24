# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Bit storage shared by matrix elements and scales, independent of an ISA."""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from collections.abc import Sequence

_MAX_BITS = (1 << 64) - 1


def _checked(value: int) -> int:
    if not isinstance(value, int) or not 0 <= value <= _MAX_BITS:
        raise ValueError("storage size or offset exceeds uint64 range")
    return value


@dataclass(frozen=True)
class BitPacking:
    """Low-bit-first patterns in a little-endian bit stream.

    slot_bits=None means dense packing. Wider slots have zero unused high bits.
    No operand role, K grouping, or numeric conversion is implied.
    """

    element_bits: int
    slot_bits: int | None = None

    def __post_init__(self) -> None:
        slot = self.element_bits if self.slot_bits is None else self.slot_bits
        if not (1 <= self.element_bits <= slot <= 64):
            raise ValueError("packing requires 1 <= element_bits <= slot_bits <= 64")
        object.__setattr__(self, "slot_bits", slot)

    def group(self, carrier_bits: int) -> tuple[int, int]:
        """Smallest whole-carrier group: (logical slots, carriers)."""
        if carrier_bits not in (8, 16, 32, 64):
            raise ValueError("carrier_bits must be 8, 16, 32, or 64")
        common = gcd(self.slot_bits, carrier_bits)
        return carrier_bits // common, self.slot_bits // common

    def bit_offset(self, index: int) -> int:
        return _checked(_checked(index) * self.slot_bits)

    def byte_size(self, count: int, bit_offset: int = 0) -> int:
        _checked(bit_offset)
        bits = self.bit_offset(count)
        if count == 0:
            return 0
        return (_checked(bits + bit_offset) + 7) // 8

    def pack(self, patterns: Sequence[int], *, bit_offset: int = 0) -> bytes:
        """Pack encoded unsigned patterns, without interpreting their values."""
        out = bytearray(self.byte_size(len(patterns), bit_offset))
        for i, pattern in enumerate(patterns):
            if not 0 <= pattern < (1 << self.element_bits):
                raise ValueError("pattern does not fit element_bits")
            pos = bit_offset + self.bit_offset(i)
            # Byte-sized pieces avoid an implicit assumption about word boundaries.
            remaining = self.element_bits
            while remaining:
                byte, shift = divmod(pos, 8)
                take = min(remaining, 8 - shift)
                out[byte] |= (pattern & ((1 << take) - 1)) << shift
                pattern >>= take
                remaining -= take
                pos += take
        return bytes(out)

    def unpack(self, data: bytes, count: int, *, bit_offset: int = 0) -> list[int]:
        if len(data) < self.byte_size(count, bit_offset):
            raise ValueError("packed buffer is too small")
        result = []
        for i in range(count):
            pos = bit_offset + self.bit_offset(i)
            value = 0
            for bit in range(self.element_bits):
                at = pos + bit
                value |= ((data[at // 8] >> (at % 8)) & 1) << bit
            result.append(value)
        return result


@dataclass(frozen=True)
class FragmentPacking:
    """A payload and its carrier capacity; all unused bits are zero."""

    packing: BitPacking
    count: int
    carrier_bits: int
    carrier_count: int

    def __post_init__(self) -> None:
        self.packing.group(self.carrier_bits)
        capacity = _checked(_checked(self.carrier_count) * self.carrier_bits)
        if self.packing.bit_offset(self.count) > capacity:
            raise ValueError("fragment payload exceeds carrier capacity")

    @property
    def payload_bits(self) -> int:
        return self.packing.bit_offset(self.count)

    @property
    def live_carriers(self) -> int:
        return (self.payload_bits + self.carrier_bits - 1) // self.carrier_bits

    @property
    def padding_bits(self) -> int:
        return self.carrier_count * self.carrier_bits - self.payload_bits

    def pack(self, patterns: Sequence[int]) -> tuple[int, ...]:
        if len(patterns) != self.count:
            raise ValueError("fragment logical count mismatch")
        width = self.carrier_bits // 8
        data = self.packing.pack(patterns)
        data += bytes(self.carrier_count * width - len(data))
        return tuple(
            int.from_bytes(data[i : i + width], "little")
            for i in range(0, len(data), width)
        )


@dataclass(frozen=True)
class MatrixFragmentLayout:
    """Contiguous K chunks interleaved between lane groups.

    The atom selects this mapping. Storage and carrier widths do not determine it.
    """

    fragment: FragmentPacking
    chunk_elements: int
    lane_groups: int
    lanes_per_group: int

    def __post_init__(self) -> None:
        if (
            self.chunk_elements <= 0
            or self.lane_groups <= 0
            or self.lanes_per_group <= 0
            or self.fragment.count % self.chunk_elements
        ):
            raise ValueError("invalid matrix fragment chunk layout")
        if self.fragment.packing.bit_offset(self.chunk_elements) % 8:
            raise ValueError("matrix fragment chunks must occupy whole bytes")

    @property
    def chunks_per_lane(self) -> int:
        return self.fragment.count // self.chunk_elements

    @property
    def chunk_bytes(self) -> int:
        return self.fragment.packing.byte_size(self.chunk_elements)

    def coord(self, lane: int, slot: int) -> tuple[int, int]:
        if not (
            0 <= lane < self.lane_groups * self.lanes_per_group
            and 0 <= slot < self.fragment.count
        ):
            raise ValueError("matrix fragment coordinate out of bounds")
        chunk, element = divmod(slot, self.chunk_elements)
        return (
            lane % self.lanes_per_group,
            (chunk * self.lane_groups + lane // self.lanes_per_group)
            * self.chunk_elements
            + element,
        )
