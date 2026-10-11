# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Independent bit-stream fixtures for matrices and scales, and register packing."""

import re

import pytest
from rocke.core.arch.wmma_scale import ScalePacking, scaled_matrix_layout
from rocke.core.storage import (
    BitPacking,
    FragmentPacking,
    packed_word_range,
    register_packing,
)


@pytest.mark.parametrize("bits", [4, 6, 8, 16, 32, 64])
@pytest.mark.parametrize("offset", [0, 1, 7, 9])
def test_dense_patterns_against_integer_oracle(bits, offset):
    patterns = list(range(1 << bits)) if bits <= 8 else [0, 1, (1 << bits) - 1, 0x1234]
    packing = BitPacking(bits)
    encoded = packing.pack(patterns, bit_offset=offset)
    expected = sum(v << (offset + i * bits) for i, v in enumerate(patterns))
    assert int.from_bytes(encoded, "little") == expected
    assert packing.unpack(encoded, len(patterns), bit_offset=offset) == patterns
    with pytest.raises(ValueError, match="too small"):
        packing.unpack(encoded[:-1], len(patterns), bit_offset=offset)


def test_fp6_crosses_words_without_per_word_padding():
    packing = BitPacking(6)
    assert packing.pack([], bit_offset=7) == b""
    assert packing.group(8) == (4, 3)
    assert packing.group(32) == (16, 3)
    patterns = [0] * 16
    patterns[5] = patterns[10] = 63
    # slot 5 occupies bits 30..35; slot 10 occupies bits 60..65.
    expected = (0xC0000000, 0xF000000F, 0x00000003)
    assert FragmentPacking(packing, 16, 32, 3).pack(patterns) == expected
    assert BitPacking(6, 8).pack(patterns) == bytes(patterns)
    assert packing.pack([1, 2, 3, 4]) == bytes([0x81, 0x30, 0x10])


@pytest.mark.parametrize(
    "dtype,live", [("fp8", 16), ("fp6", 12), ("bf6", 12), ("fp4", 8)]
)
def test_matrix_payload_and_padding(dtype, live):
    layout = scaled_matrix_layout(dtype, 16)
    assert layout.fragment.count == 64
    assert layout.fragment.live_carriers == live
    assert layout.fragment.padding_bits == (16 - live) * 32
    patterns = [(1 << layout.fragment.packing.element_bits) - 1] * 64
    assert layout.fragment.pack(patterns) == (0xFFFFFFFF,) * live + (0,) * (16 - live)
    for lane in range(32):
        coords = [layout.coord(lane, slot) for slot in range(64)]
        assert all(row == lane % 16 and 0 <= k < 128 for row, k in coords)
        other = {layout.coord(lane ^ 16, slot)[1] for slot in range(64)}
        assert {k for _, k in coords}.isdisjoint(other)
        assert {k for _, k in coords} | other == set(range(128))


@pytest.mark.parametrize(
    "count,block_k,word",
    [(1, 32, 0x01), (2, 32, 0x0201), (4, 32, 0x04030201), (8, 16, 0x0807060504030201)],
)
def test_scales_use_common_bit_packing(count, block_k, word):
    scales = ScalePacking(count, block_k)
    assert scales.block_k == block_k
    assert scales.fragment.pack(list(range(1, count + 1))) == (word,)
    assert scales.packing == BitPacking(8)
    assert scales.word_bits == count * 8
    assert scales.fragment.carrier_bits == count * 8


@pytest.mark.parametrize(
    "count,block_k", [(0, 32), (-1, 32), (3, 32), (16, 32), (4.0, 32), (4, 0)]
)
def test_invalid_scale_packing(count, block_k):
    with pytest.raises(ValueError):
        ScalePacking(count, block_k)


def test_invalid_packing_and_overflow():
    for bits, slots in [(0, None), (65, None), (6, 4)]:
        with pytest.raises(ValueError):
            BitPacking(bits, slots)
    with pytest.raises(ValueError, match="capacity"):
        FragmentPacking(BitPacking(6), 16, 32, 2)
    with pytest.raises(ValueError, match="uint64"):
        BitPacking(6).byte_size(1 << 63)
    with pytest.raises(ValueError, match="fit"):
        BitPacking(6).pack([64])


@pytest.mark.parametrize(
    ("dtype", "count", "registers"),
    [
        ("fp4e2m1", 32, 4),
        ("fp6e2m3", 32, 6),
        ("fp6e3m2", 16, 3),
        ("i4", 16, 2),
        ("u4", 8, 1),
    ],
)
def test_sub_byte_elements_pack_densely_into_whole_registers(dtype, count, registers):
    packing = register_packing(dtype, count)
    assert packing.carrier_bits == 32
    assert packing.carrier_count == registers
    assert packing.padding_bits == 0


@pytest.mark.parametrize("dtype", ["fp8e4m3", "i8", "fp16", "bf16", "fp32", "fp64"])
def test_byte_or_wider_elements_are_not_packed(dtype):
    assert register_packing(dtype, 8) is None


def test_register_packing_lays_out_bits_like_a_hand_built_fragment():
    patterns = [(3 * i + 1) % 64 for i in range(16)]
    by_hand = FragmentPacking(BitPacking(6), 16, 32, 3)
    assert register_packing("fp6e2m3", 16).pack(patterns) == by_hand.pack(patterns)


def test_a_payload_that_leaves_a_register_partly_filled_raises():
    message = (
        "3 fp6e2m3 elements occupy 18 bits, not a whole number of 32-bit registers "
        "-- packed operands must fill whole registers"
    )
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        register_packing("fp6e2m3", 3)


@pytest.mark.parametrize("count", [2.0, True])
def test_register_packing_needs_an_integer_count(count):
    with pytest.raises(TypeError, match="^element count must be an integer"):
        register_packing("fp4e2m1", count)


@pytest.mark.parametrize("count", [0, -4])
def test_register_packing_needs_a_positive_count(count):
    with pytest.raises(ValueError, match="^element count must be at least 1"):
        register_packing("fp4e2m1", count)


def test_a_predicate_is_not_register_data():
    with pytest.raises(ValueError, match="^i1 is a predicate, not register data$"):
        register_packing("i1", 32)


@pytest.mark.parametrize(
    ("dtype", "start", "length", "words"),
    [
        ("fp4e2m1", 32, 32, range(4, 8)),
        ("fp6e2m3", 32, 32, range(6, 12)),
        ("fp6e2m3", 16, 16, range(3, 6)),
        ("i4", 0, 8, range(1)),
    ],
)
def test_an_element_range_maps_to_the_registers_that_hold_it(
    dtype, start, length, words
):
    assert packed_word_range(register_packing(dtype, 64), start, length) == words


def test_an_element_range_off_a_register_boundary_raises():
    message = (
        "elements [3, 35) of a fragment packed at 4 bits per element do not start "
        "and end on a register boundary (32 bits) -- start and length must be "
        "multiples of 8 elements"
    )
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        packed_word_range(register_packing("fp4e2m1", 64), 3, 32)


@pytest.mark.parametrize(
    ("start", "length", "message"),
    [
        (-8, 8, "^start must be at least 0, got -8$"),
        (8, 0, "^length must be at least 1, got 0$"),
        (56, 16, "^elements \\[56, 72\\) run past the fragment's 64 elements$"),
    ],
)
def test_an_element_range_must_lie_inside_the_fragment(start, length, message):
    with pytest.raises(ValueError, match=message):
        packed_word_range(register_packing("fp4e2m1", 64), start, length)


@pytest.mark.parametrize(("start", "length"), [(True, 8), (8.0, 8), (0, 8.0)])
def test_an_element_range_needs_integer_ends(start, length):
    with pytest.raises(TypeError, match="must be an integer"):
        packed_word_range(register_packing("fp4e2m1", 64), start, length)


def test_the_boundary_error_names_the_step_for_fp6():
    with pytest.raises(ValueError, match="multiples of 16 elements$"):
        packed_word_range(register_packing("fp6e2m3", 64), 8, 16)
