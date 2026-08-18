# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for the TLU=1 fp4 column-scatter LDS layout.

col_scatter deals a strip's K-columns across the DTL loads and places each
column's group at the thread position that puts its distinguishing bit on the
bank-pair bit. Everything derives from N = stackM, so the properties below are
stated against N rather than against a table of expected numbers.

Worth testing directly: a wrong derivation here does not fail to build, it
builds a kernel that reads the wrong LDS bytes. The guard in _buildColScatter
exists because at N=32 readStrideBytes silently comes out 0 and the read stops
stepping in K.
"""

from types import SimpleNamespace

import pytest

from Tensile.Components.Subtile.SubtileTLUSwizzle import (
    _buildColScatter,
    selectTLUColScatter,
    selectTLUSwizzle,
)

pytestmark = pytest.mark.unit

# gfx950 fp4 TLU=1: MFMA 16x128, half a byte per element, 64-lane waves.
_INST_M, _INST_K, _BPE, _WAVE = 16, 128, 0.5, 64

_COL_SCATTER_STACKS = (8, 16)


def _cs(n):
    return _buildColScatter(n, _INST_M, _INST_K, _BPE, _WAVE)


def _tile(stack=16, bpe=0.5, perStrip=1, kSplit=1):
    return SimpleNamespace(
        subtileShape=(stack, 1), bpe=bpe, waveSize=_WAVE,
        mmaTileShape=(_INST_M, _INST_K),
        grWavesPerStrip=perStrip, grKSplit=kSplit,
    )


# --- the property the layout exists to provide -----------------------------

@pytest.mark.parametrize("n", _COL_SCATTER_STACKS)
def test_bank_pair_bit_always_carries_the_distinguishing_group_bit(n):
    """Thread bit 3 is the bank-pair bit; col_group[gdBit] must land there.

    This is the whole point of the layout -- if this bit carries anything else
    the two ds_read phases collide on the same bank pair.
    """
    cs = _cs(n)
    assert cs.cgThreadBits[cs.gdBit] == 3


@pytest.mark.parametrize("n", _COL_SCATTER_STACKS)
def test_thread_bits_are_a_permutation_with_no_collisions(n):
    """Every m_chunk and col_group bit gets its own thread bit."""
    cs = _cs(n)
    used = list(cs.mChunkThreadBits) + list(cs.cgThreadBits)
    assert len(used) == cs.cBits + cs.gBits == 6
    assert sorted(used) == sorted(set(used)), "two fields share a thread bit"
    assert set(used) <= {0, 1, 2, 3, 4, 5}


@pytest.mark.parametrize("n", _COL_SCATTER_STACKS)
def test_each_lane_decodes_to_a_distinct_chunk_and_group(n):
    """The lane -> (m_chunk, col_group) map must be bijective over the wave."""
    cs = _cs(n)
    seen = set()
    for lane in range(_WAVE):
        mc = sum(((lane >> b) & 1) << i for i, b in enumerate(cs.mChunkThreadBits))
        cg = sum(((lane >> b) & 1) << i for i, b in enumerate(cs.cgThreadBits))
        seen.add((mc, cg))
    assert len(seen) == _WAVE, "two lanes decode to the same (m_chunk, col_group)"


@pytest.mark.parametrize("n", _COL_SCATTER_STACKS)
def test_the_k_step_is_a_nonzero_per_lane_constant(n):
    """readStrideBytes 0 would stop the LR read stepping in K -- wrong answers."""
    assert _cs(n).readStrideBytes > 0


def test_read_stride_matches_the_emitted_offset_for_the_shipped_stack():
    """stack 16 emits `ds_read_b64_tr_b4 ... offset:256` for read=1 on gfx950."""
    assert _cs(16).readStrideBytes == 256


def test_a_stack_the_derivation_cannot_express_is_rejected():
    """At N=32 the K step degenerates to 0, so the stack must be refused."""
    with pytest.raises(AssertionError, match="col_scatter derives stacks"):
        _cs(32)


# --- field derivations ------------------------------------------------------

@pytest.mark.parametrize("n", _COL_SCATTER_STACKS)
def test_fields_follow_from_the_stack_height(n):
    cs = _cs(n)
    assert cs.N == n
    assert cs.cpc == int(n * _INST_M * _BPE) // 16      # 16B chunks per K column
    assert cs.gGroups == _INST_K // n                    # col_groups per load
    assert cs.mTileBytes == int(_INST_M * _BPE)
    assert cs.blkBytes == _WAVE * 16 + cs.padBytes       # pad separates blocks


# --- selection --------------------------------------------------------------

@pytest.mark.parametrize("stack", _COL_SCATTER_STACKS)
def test_tall_unshared_stacks_select_col_scatter(stack):
    assert selectTLUColScatter(_tile(stack=stack)) is not None
    assert selectTLUSwizzle(_tile(stack=stack)) is None, "the two paths are exclusive"


@pytest.mark.parametrize("stack", (2, 4))
def test_short_unshared_stacks_use_the_xor_swizzle_instead(stack):
    assert selectTLUColScatter(_tile(stack=stack)) is None
    assert selectTLUSwizzle(_tile(stack=stack)) is not None


@pytest.mark.parametrize("stack", (2, 4))
def test_a_shared_strip_routes_short_stacks_to_col_scatter(stack):
    """The XOR acts on the physical chunk index, which a shared strip breaks."""
    shared = _tile(stack=stack, perStrip=2)
    assert selectTLUSwizzle(shared) is None
    assert selectTLUColScatter(shared) is not None


def test_non_fp4_never_selects_col_scatter():
    """Guarded to bpe 0.5; bf16 TLU=1 keeps baseline addressing."""
    assert selectTLUColScatter(_tile(stack=16, bpe=2.0)) is None
