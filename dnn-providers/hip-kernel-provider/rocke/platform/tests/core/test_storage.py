# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Independent bit-stream fixtures and storage bounds for matrices and scales."""

import pytest

from rocke.core.arch.wmma_scale import ScalePacking, scaled_matrix_layout
from rocke.core.storage import BitPacking, FragmentPacking, TensorStorage


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
    "count,block_k,word", [(4, 32, 0x04030201), (8, 16, 0x0807060504030201)]
)
def test_scales_use_common_bit_packing(count, block_k, word):
    scales = ScalePacking(count, block_k)
    assert scales.association.block_k == block_k
    assert scales.fragment.pack(list(range(1, count + 1))) == (word,)
    assert scales.packing == BitPacking(8)
    assert scales.word_bits == count * 8


def test_storage_rows_offsets_and_empty_views():
    storage = TensorStorage("bf6", (3, 5), row_stride_bytes=8, base_bit_offset=3)
    assert storage.dtype == "fp6e3m2"
    assert storage.byte_size == 21
    assert storage.address(2, 4) == (19, 3)
    assert TensorStorage("fp4", (0, 5)).byte_size == 0
    assert TensorStorage("fp4", (3, 0), base_bit_offset=7).byte_size == 0
    assert BitPacking(6).pack([], bit_offset=7) == b""
    assert TensorStorage("e8m0", (2, 8)).byte_size == 16
    with pytest.raises(ValueError, match="out of bounds"):
        storage.address(3, 0)
    with pytest.raises(ValueError, match="row stride"):
        TensorStorage("fp6", (3, 5), row_stride_bytes=3)
    with pytest.raises(ValueError, match="power of two"):
        TensorStorage("fp4", (1, 4), alignment_bytes=3)


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_storage_allows_byte_strides(dtype):
    storage = TensorStorage(dtype, (2, 64), row_stride_bytes=129)
    assert storage.address(1, 0) == (129, 0)
    assert storage.byte_size == 257


def test_invalid_packing_and_overflow():
    for bits, slots in [(0, None), (65, None), (6, 4)]:
        with pytest.raises(ValueError):
            BitPacking(bits, slots)
    with pytest.raises(ValueError, match="capacity"):
        FragmentPacking(BitPacking(6), 16, 32, 2)
    with pytest.raises(ValueError, match="uint64"):
        BitPacking(6).byte_size(1 << 63)
    with pytest.raises(ValueError, match="uint64"):
        TensorStorage("fp8", (1 << 63, 3))
    with pytest.raises(ValueError, match="fit"):
        BitPacking(6).pack([64])
