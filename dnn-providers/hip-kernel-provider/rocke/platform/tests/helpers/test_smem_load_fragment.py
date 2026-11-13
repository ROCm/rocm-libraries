# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""``smem_load_fragment`` / ``concat_tree``: piece count, offsets and K order.

The conv kernels that use these helpers pin their emitted IR through the
byte-identity gate; this checks the helpers' own contract on a bare builder.
"""

import pytest
from rocke.core.ir import F16, F32, IRBuilder, VectorType
from rocke.helpers.io import concat_tree, smem_load_fragment


def _ops(b, name):
    return [op for op in b.kernel.body.ops if op.name == name]


def _fixture():
    b = IRBuilder("smem_load_fragment")
    smem = b.smem_alloc(F32, [1, 64])
    row, idx = b.const_i32(0), b.const_i32(8)
    return b, smem, row, idx


def test_narrow_fragment_is_one_load():
    b, smem, row, idx = _fixture()
    frag = smem_load_fragment(b, smem, row, idx, dtype=F32, n=4)
    assert frag.type == VectorType(F32, 4)
    assert len(_ops(b, "tile.smem_load_vN")) == 1
    assert not _ops(b, "vector.concat")


def test_16_bit_fragment_within_max_elems_is_one_load():
    b = IRBuilder("smem_load_fragment_f16")
    smem = b.smem_alloc(F16, [1, 64])
    frag = smem_load_fragment(
        b, smem, b.const_i32(0), b.const_i32(0), dtype=F16, n=8, max_elems=8
    )
    assert frag.type == VectorType(F16, 8)
    assert len(_ops(b, "tile.smem_load_vN")) == 1


@pytest.mark.parametrize("stride", [None, 20])
def test_wide_fragment_is_16_byte_pieces_in_k_order(stride):
    b, smem, row, idx = _fixture()
    frag = smem_load_fragment(b, smem, row, idx, dtype=F32, n=16, stride=stride)
    assert frag.type == VectorType(F32, 16)
    loads = _ops(b, "tile.smem_load_vN")
    assert [op.attrs["vec"] for op in loads] == [4, 4, 4, 4]
    # Piece 0 reads at idx itself; piece h at idx + h * step.
    assert loads[0].operands[2] is idx
    step = 4 if stride is None else stride
    assert [op.attrs.get("value") for op in _ops(b, "arith.constant")][-3:] == [
        step,
        2 * step,
        3 * step,
    ]
    # (p0 ++ p1) ++ (p2 ++ p3): pairwise rounds keep K order.
    concats = _ops(b, "vector.concat")
    assert len(concats) == 3
    pieces = [op.result for op in loads]
    assert concats[0].operands == [pieces[0], pieces[1]]
    assert concats[1].operands == [pieces[2], pieces[3]]
    assert concats[2].operands == [concats[0].result, concats[1].result]


def test_rejects_a_ragged_fragment():
    b, smem, row, idx = _fixture()
    with pytest.raises(ValueError, match="not a multiple of the 4-element load"):
        smem_load_fragment(b, smem, row, idx, dtype=F32, n=6)
    with pytest.raises(ValueError, match="stride must be positive"):
        smem_load_fragment(b, smem, row, idx, dtype=F32, n=8, stride=0)


def test_concat_tree_needs_a_power_of_two():
    b, smem, row, idx = _fixture()
    parts = [b.smem_load_vN(smem, row, idx, dtype=F32, n=4) for _ in range(3)]
    with pytest.raises(ValueError, match="power-of-two"):
        concat_tree(b, parts)
    assert concat_tree(b, parts[:1]) is parts[0]
