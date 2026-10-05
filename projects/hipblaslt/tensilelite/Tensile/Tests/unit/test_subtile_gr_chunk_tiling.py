# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""chunksPerK comes from the STRIP WIDTH, never from the dtype.

The TLU=1 global read hands each lane a 128-bit chunk covering ``16 / bpe``
contiguous free-dim elements at one K row.  A strip is ``mStripBytes`` wide, so
a lane needs ``chunksPerK = mStripBytes // 16`` chunks to cover one K row, and
the physical chunk index ``P = loadIdx * wavesize + laneId`` splits as::

    K row    = P // chunksPerK
    M block  = P %  chunksPerK      (of elemsPerChunk free-dim elements)

An earlier form of this gated the split on fp4::

    chunksPerK = max(1, mStripBytes // 16) if float(tileInfo.bpe) == 0.5 else 1

The gate is not dead code dressed up as a rule -- fp4 really does exercise the
split, at AB_B4_TLU1_4x1, whose 32 B strip is two chunks.  That is why it
survived review: on every geometry that existed, the gate and the strip-width
rule return the same number.  The fp4 stacks wide enough to disagree (8x1, 16x1)
never arrive here at all, because selectTLUColScatter routes them to the
column-scatter path before the split is reached.

They stop agreeing the moment bf16 TLU=1 exists, because a bf16 element is 4x an
fp4 one, so a bf16 strip of the same stack spans 4x the chunks.  Both reachable
bf16 stacks are wrong under the gate, including AB_B16_TLU1_2x1, which is what
the MT320x320 NN kernel selects -- the dtype-gated rule ramps its K row four
times too fast.  Nothing downstream catches it: the kernel assembles, runs, and
returns wrong A.  These tests exist so that substitution fails here instead.

Nothing below declares a strip width, a chunk count, or a dtype.  Every number
is read from the geometry table, which is the thing that owns it; the table
declared here is the *inventory* of TLU=1 geometries and which of them are
reachable, so that adding or removing one is a visible event rather than a
silent widening of the parametrization.  Both dtypes are asserted in the same
parametrized tests, and the fp4 rows are load-bearing: a resolution that fixes
bf16 by perturbing fp4 fails them.
"""

import pytest

from rocisa.container import sgpr

from gpu_test_helpers import TileConfig, create_writer, init_rocisa

from Tensile.Components.Subtile.Kernel import AB_GEOMETRY_MAP
from Tensile.Components.Subtile.SubtileGREmit import _graTileAssignment_tlu
from Tensile.Components.Subtile.SubtileTLUSwizzle import selectTLUColScatter

WAVE_SIZE = 64
CHUNK_BYTES = 16

# Registered but not emittable, with the reason each one stops.  They are held
# here rather than filtered out silently: "unreachable" is a claim about the
# emit, and test_unreachable_stacks_fail_loudly checks it is still true.
UNREACHABLE = {
    "AB_B16_TLU1":      "need 3 k bits",        # stack 8: swizzle defines 2
    "AB_B16_TLU1_16x1": "kPerGroup",            # stack 16: LR tr read spans too much K
}

# The one end-to-end fact worth pinning by name: Solution.py records this as
# _ABTilePairA for the MT320x320 NN bf16 kernel.
MT320X320_GEOMETRY = "AB_B16_TLU1_2x1"


def _tlu1Geometries():
    """Every TLU=1 entry in the geometry table, by name."""
    return {name: g for name, g in AB_GEOMETRY_MAP.items()
            if g.gr.tlu and g.lr.tlu}


def _emittable():
    return sorted(n for n in _tlu1Geometries() if n not in UNREACHABLE)


EMITTABLE = _emittable()


def _stripBytes(tileInfo):
    """mStripBytes, as the GR emit computes it."""
    return int(tileInfo.subtileShape[0] * int(tileInfo.mmaTileShape[0]) * tileInfo.bpe)


def _chunksPerK(tileInfo):
    """The rule in force, derived from the strip and nothing else."""
    return max(1, _stripBytes(tileInfo) // CHUNK_BYTES)


def _dtypeGatedChunksPerK(tileInfo):
    """The rule these tests exist to reject.  Kept executable, not just quoted,
    so the divergence below is asserted rather than described."""
    strip = _stripBytes(tileInfo)
    return max(1, strip // CHUNK_BYTES) if float(tileInfo.bpe) == 0.5 else 1


def _writer(name, mtA=256, waveGroup=(4, 1)):
    """create_writer for a geometry, taking bpe and MatrixInstK from the
    geometry itself rather than from a value restated here."""
    geometry = AB_GEOMETRY_MAP[name]
    return create_writer(
        TileConfig(mt_a=mtA, mt_b=64, depth_u=128),
        mi_wave_group=list(waveGroup),
        geometry=geometry,
        inst_k=geometry.gr.instK,
        bpe=geometry.gr.bpe,
    )


def _tileInfoA(name, mtA=256, waveGroup=(4, 1)):
    return _writer(name, mtA, waveGroup)[2]


def _grModule(name, mtA=256, waveGroup=(4, 1)):
    """Emit the TLU=1 GR offset module for A and return it as assembly text.

    create_writer's mock kernel stops short of what this path reads -- there is
    no IndexUnroll and no strideRef -- so both are supplied here rather than in
    gpu_test_helpers, which serves the TLU=0 tests that do not need them.
    """
    writer, kernel, tileInfoA, _ = _writer(name, mtA, waveGroup)

    init_rocisa(target="gfx950")
    kernel["ProblemType"]["IndexUnroll"] = 2
    writer.sgprPool.checkOut(12, tag="_grModule_sgprs")
    writer.sgprs["StrideA0I"] = 10
    writer.sgprs["StrideB1J"] = 11
    writer.strideRef = lambda tc, idx: sgpr("StrideA0I" if tc == "A" else "StrideB1J")
    tileInfoA.allocOffsetRegisters(writer, kernel)
    return str(_graTileAssignment_tlu(writer, kernel, tileInfoA)), tileInfoA


# --- inventory ---------------------------------------------------------------


def test_tlu1_geometry_inventory():
    """Pin which geometries exist and which are emittable.

    This is the only declared thing in the file, and it is declared because it
    is not derivable: a new TLU=1 geometry should widen the parametrization
    below deliberately, with someone deciding whether it is reachable, rather
    than by appearing in a map.
    """
    assert set(_tlu1Geometries()) == {
        "AB_B4_TLU1", "AB_B4_TLU1_4x1", "AB_B4_TLU1_8x1", "AB_B4_TLU1_16x1",
        "AB_B16_TLU1_2x1", "AB_B16_TLU1_4x1", "AB_B16_TLU1", "AB_B16_TLU1_16x1",
    }
    assert set(UNREACHABLE) < set(_tlu1Geometries())
    # Both dtypes have to be represented, or the two-sided tests below are
    # quietly one-sided.
    bpes = {float(AB_GEOMETRY_MAP[n].gr.bpe) for n in EMITTABLE}
    assert bpes == {0.5, 2.0}


@pytest.mark.parametrize("name", sorted(UNREACHABLE))
def test_unreachable_stacks_fail_loudly(name):
    """A stack with no swizzle and no transpose read must raise, not emit.

    These are excluded from every test below, so the exclusion has to be
    justified by something.  If one of them becomes emittable the claim here
    fails and it gets added to the parametrization on purpose.
    """
    with pytest.raises((AssertionError, RuntimeError)) as excinfo:
        _grModule(name)
    assert UNREACHABLE[name] in str(excinfo.value)


# --- the rule ----------------------------------------------------------------


@pytest.mark.parametrize("name", EMITTABLE)
def test_chunks_per_k_is_the_strip_in_16b_chunks(name):
    """The arithmetic itself: strip width divided by the 16 B chunk.

    Stated as an invariant rather than a table of answers -- a lane's chunks
    must tile its strip exactly, and a chunk must hold a whole number of
    elements.
    """
    ti = _tileInfoA(name)
    strip = _stripBytes(ti)
    assert strip == int(ti.subtileShape[0]) * int(ti.mmaTileShape[0]) * ti.bpe
    assert _chunksPerK(ti) * CHUNK_BYTES == strip or strip < CHUNK_BYTES
    elemsPerChunk = int(CHUNK_BYTES / ti.bpe)
    assert elemsPerChunk * ti.bpe == CHUNK_BYTES


@pytest.mark.parametrize("name", EMITTABLE)
def test_dtype_gated_rule_diverges_exactly_on_bf16(name):
    """Pin where the rejected rule agrees and where it does not.

    It agrees on every fp4 geometry -- which is why it was not caught, and
    which is the half of this test that a bf16-only fix must not break -- and
    is wrong on every bf16 stack wider than one chunk.
    """
    ti = _tileInfoA(name)
    gated, correct = _dtypeGatedChunksPerK(ti), _chunksPerK(ti)
    if float(ti.bpe) == 0.5:
        assert gated == correct, "fp4 must keep the answer it already had"
    elif correct > 1:
        assert gated == 1, (
            "bf16 strips span several chunks; a dtype gate collapses them to one")
    else:
        assert gated == correct


@pytest.mark.parametrize("name", EMITTABLE)
def test_gr_emit_splits_the_chunk_index(name):
    """The emitted code, not a re-derivation of it.

    This is the assertion that survives the split being refactored into a
    helper: it asks the GR emit what it actually produced.  The split is
    emitted exactly when a lane's row spans more than one chunk AND the stack
    has not been routed to the column-scatter path instead.
    """
    asm, ti = _grModule(name)
    chunksPerK = _chunksPerK(ti)
    expected = chunksPerK > 1 and selectTLUColScatter(ti) is None

    mBlock = [ln for ln in asm.splitlines() if "M block = P %" in ln]
    kRow = [ln for ln in asm.splitlines() if "K row = P //" in ln]
    if not expected:
        assert not mBlock and not kRow
        return
    assert mBlock and kRow
    assert all(f"M block = P % {chunksPerK}" in ln for ln in mBlock)
    assert all(f"K row = P // {chunksPerK}" in ln for ln in kRow)
    assert len(mBlock) == len(kRow)


@pytest.mark.parametrize("name", EMITTABLE)
def test_emitted_split_is_not_the_dtype_gated_one(name):
    """The divergence, asserted against the emit rather than against a table.

    Where the two rules differ, the assembly has to show the strip-width answer.
    Reinstating the gate turns every one of these into a missing split.
    """
    ti = _tileInfoA(name)
    if selectTLUColScatter(ti) is not None:
        pytest.skip("column-scatter path does not reach the split")
    gated, correct = _dtypeGatedChunksPerK(ti), _chunksPerK(ti)
    asm, _ = _grModule(name)
    if gated == correct:
        return
    assert f"M block = P % {correct}" in asm
    assert f"M block = P % {gated}" not in asm


def test_mt320x320_geometry_needs_a_four_way_split():
    """The shape this guard is really for, pinned end to end.

    MT320x320 NN bf16 selects AB_B16_TLU1_2x1.  Its strip is a stack of 2 at
    instM 16 and 2 B, so P splits four ways -- and the dtype-gated rule would
    not split it at all.
    """
    asm, ti = _grModule(MT320X320_GEOMETRY)
    chunksPerK = _chunksPerK(ti)
    assert chunksPerK == 4
    assert f"M block = P % {chunksPerK}" in asm
    assert f"K row = P // {chunksPerK}" in asm
    assert _dtypeGatedChunksPerK(ti) == 1


@pytest.mark.parametrize("name", EMITTABLE)
def test_local_k_span_cannot_exceed_the_subtile_k_extent(name):
    """chunksPerK also divides the swizzle's K span; too small and it overruns.

    b16LocalKSpan = numGRPerSubtile * wavesize // chunksPerK is the K extent one
    wave's own chunk ramp covers, and the swizzle sources k bits at or above it
    from the fetch-group index instead.  It cannot exceed the K rows the subtile
    has.  This is a dtype-independent sanity bound, and the dtype-gated rule
    breaks it by a factor of chunksPerK.
    """
    ti = _tileInfoA(name)
    kRows = int(ti.mmaTileShape[1] * ti.subtileShape[1])
    ramp = int(ti.numGRPerSubtile) * WAVE_SIZE
    assert ramp // _chunksPerK(ti) <= kRows

    gated = _dtypeGatedChunksPerK(ti)
    if gated != _chunksPerK(ti):
        assert ramp // gated > kRows, (
            "the dtype-gated rule should overrun the subtile's K extent")
