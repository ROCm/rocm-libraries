# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Bit storage shared by matrix elements and scales, independent of an ISA."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import gcd

from .dtypes import DTypeCategory, dtype_info

_MAX_BITS = (1 << 64) - 1

# Width of one hardware register: the carrier sub-byte elements are packed into by hand.
REGISTER_BITS = 32


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


def _require_int_at_least(name: str, value: int, minimum: int) -> int:
    """Return ``value`` if it is an int (not a bool) of at least ``minimum``, else raise."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return value


def is_register_packed(dtype: str) -> bool:
    """Whether ``dtype``'s elements are packed into registers by hand: True for a
    sub-byte dtype (fp4, fp6, i4), which has no LLVM element type. A dtype of 8 or
    more bits is False: LLVM packs its vectors into registers itself (a
    ``<4 x half>`` occupies two registers). A predicate raises ``ValueError``.
    """
    info = dtype_info(dtype)
    if info.category is DTypeCategory.PREDICATE:
        raise ValueError(f"{info.name} is a predicate, not register data")
    return info.encoded_bits < 8


# MMA scale encodings: whole bytes, but no arithmetic and no value form in the IR (neither engine
# maps them to an LLVM or HIP type), so they are held and moved as bytes.
SCALE_ENCODINGS = ("e8m0", "e5m3")

# Dtypes with no arithmetic of their own, held in the integer of their width.
_HELD_AS_INTEGER = {"tf32": "i32", **{name: "i8" for name in SCALE_ENCODINGS}}


def storage_dtype(dtype: str) -> str:
    """The dtype whose elements hold ``dtype`` in registers and in LDS -- the one rule::

        dtype                         storage
        fp4e2m1, fp6e2m3, i4, ...     i32     register-packed: whole 32-bit words
        tf32                          i32     no arithmetic of its own
        e8m0, e5m3                    i8      scale encodings (SCALE_ENCODINGS)
        every other dtype             itself

    ``i1`` (a predicate) is not register data and raises ``ValueError``.
    """
    if is_register_packed(dtype):
        return "i32"
    name = dtype_info(dtype).name
    return _HELD_AS_INTEGER.get(name, name)


def storage_group(dtype: str) -> tuple[int, int]:
    """``(elements, storage elements)``: the fewest elements of ``dtype`` that fill
    whole elements of its :func:`storage_dtype`, and how many they fill. A dtype held
    one to one is (1, 1); a register-packed one is a whole-word group, (8, 1) at 4
    bits and (16, 3) at 6. Elements move between registers, and between registers
    and memory, in these groups without repacking bits.
    """
    if not is_register_packed(dtype):
        return (1, 1)
    return BitPacking(dtype_info(dtype).encoded_bits).group(REGISTER_BITS)


def elements_to_storage(dtype: str, count: int) -> int:
    """How many :func:`storage_dtype` elements hold ``count`` elements of ``dtype``
    laid out from element 0: ``count`` for a dtype held one to one, else the 32-bit
    words they occupy low-bit-first, a partly filled last word included. Exact for a
    whole number of :func:`storage_group` groups, as every packed offset and stride
    is; rounded up otherwise, as an allocation needs."""
    _require_int_at_least("element count", count, 0)
    if not is_register_packed(dtype):
        return count
    return -(-count * dtype_info(dtype).encoded_bits // REGISTER_BITS)


def storage_to_elements(dtype: str, storage_count: int) -> int:
    """How many whole elements of ``dtype`` ``storage_count`` :func:`storage_dtype`
    elements hold: the inverse of :func:`elements_to_storage` for a whole number of
    :func:`storage_group` groups."""
    _require_int_at_least("storage element count", storage_count, 0)
    if not is_register_packed(dtype):
        return storage_count
    return storage_count * REGISTER_BITS // dtype_info(dtype).encoded_bits


def element_bytes(dtype: str) -> int:
    """Bytes of one element of ``dtype``. A sub-byte dtype has no byte width of its
    own -- it is addressed in the 32-bit words of its :func:`storage_dtype` -- and
    raises ``ValueError``."""
    info = dtype_info(dtype)
    if info.encoded_bits % 8:
        raise ValueError(
            f"element_bytes failed - dtype={info.name!r} is {info.encoded_bits} bits, "
            "expected a whole number of bytes; a sub-byte dtype is addressed in the "
            "32-bit words of its storage_dtype"
        )
    return info.encoded_bits // 8


def register_packing(dtype: str, count: int) -> FragmentPacking | None:
    """How ``count`` elements of a sub-byte ``dtype`` sit densely in 32-bit registers.

    Elements pack low-bit-first with no padding between them, and must fill
    whole registers. A dtype that is not register-packed (:func:`is_register_packed`)
    returns None, so rocke never packs it by hand. A predicate raises.
    """
    info = dtype_info(dtype)
    _require_int_at_least("element count", count, 1)
    if not is_register_packed(dtype):
        return None
    bits = count * info.encoded_bits
    if bits % REGISTER_BITS:
        raise ValueError(
            f"{count} {info.name} elements occupy {bits} bits, not a whole number of "
            f"{REGISTER_BITS}-bit registers -- packed operands must fill whole "
            "registers"
        )
    return FragmentPacking(
        BitPacking(info.encoded_bits), count, REGISTER_BITS, bits // REGISTER_BITS
    )


def packed_word_range(packing: FragmentPacking, start: int, length: int) -> range:
    """The registers that hold elements ``[start, start + length)`` of ``packing``."""
    end = _require_int_at_least("start", start, 0) + _require_int_at_least(
        "length", length, 1
    )
    if end > packing.count:
        raise ValueError(
            f"elements [{start}, {end}) run past the fragment's {packing.count} "
            "elements"
        )
    first_bit = packing.packing.bit_offset(start)
    end_bit = packing.packing.bit_offset(end)
    word = packing.carrier_bits
    if first_bit % word or end_bit % word:
        step, _ = packing.packing.group(word)
        raise ValueError(
            f"elements [{start}, {end}) of a fragment packed at "
            f"{packing.packing.slot_bits} bits per element do not start and end on "
            f"a register boundary ({word} bits) -- start and length must be "
            f"multiples of {step} elements"
        )
    return range(first_bit // word, end_bit // word)


@dataclass(frozen=True)
class MatrixFragmentLayout:
    """Contiguous K chunks interleaved between lane groups.

    The atom selects this mapping. Storage and carrier widths do not determine it.
    This describes interleaved contiguous K chunks, not arbitrary matrix
    distributions, transposed axes, or swizzled tensor addresses.
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
