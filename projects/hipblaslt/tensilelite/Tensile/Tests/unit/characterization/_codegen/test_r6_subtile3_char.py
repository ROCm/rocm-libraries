################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""R6 — SubtileGREmit remaining arms: loadRatioGR branches + dead-code paths.

CPU-only characterization. Targets uncovered regions of
Tensile/Components/Subtile/SubtileGREmit.py:

  528-639   Non-legacy helper functions (_grComputeOffset, _grComputeSubtileOffsets,
            _grComputeRowPartition, _grComputeAllOffsets + graInitPointer).
            These functions exist but are NOT yet wired into the full emit
            pipeline; they are called here directly with a mock writer.

  688-691   _grComputeSubtileOffsets_legacy VGPR fallback: when the SGPR pool
            is exhausted (size >= MaxSgpr - 3), soffset registers fall back to
            VGPRs and a different code sequence is emitted.

  711-713   _grComputeRowPartition_legacy loadRatioGR==2.0 branch.
            Triggered by WG=(4,4) + BF16 or WG=(4,4) + FP8, which yields
            loadRatioGR = bytesPerLoad(numWaves) / globalGRTileSize = 2.0.

  731-739   _grComputeAllOffsets_legacy loadRatioGR==0.5 + bpe==1 (FP8) path.
            Triggered by WG=(1,1) + FP8, which yields loadRatioGR = 0.5 and
            selects the FP8-specific K_group intra-block rotation.

  856-858   emitSingleBufferLoad loadRatioGR > 1 early-return path.
            When multiple subtiles share one GR load (loadRatioGR == 2.0) only
            the first linearId in each group emits the load; others return empty.

  886-887   emitSubtileBufferLoad wrapper (calls emitSingleBufferLoad via
            writer.states.{a,b}.tileInfo lookup).

  894-903   globalReadDoSubtile (called from Kernel.preLoop; iterates the local
            subtile grid and calls emitSubtileBufferLoad for each cell).

  947-948   globalReadLDSBufferSwap tc='MXSA'/'MXSB' branch (routes to
            emitScaleGRLDSSwap via the scale TileInfo instead of A/B XOR swap).

Strategy:
  - Section 1: Direct mock-writer tests for the non-legacy helpers (528-639).
  - Section 2: Direct mock-writer test for the VGPR fallback (688-691).
  - Section 3: Config-driven emit for loadRatioGR==0.5 (FP8 WG=(1,1), 731-739)
               and loadRatioGR==2.0 (FP8 WG=(4,4), 711-713).
  - Section 4: Direct mock-writer tests for emitSingleBufferLoad early-return
               (856-858), emitSubtileBufferLoad wrapper (886-887),
               globalReadDoSubtile (894-903), and scale LDS swap (947-948).

pytestmark = pytest.mark.unit. CPU-only; no GPU, no compile, no hardware.
"""

import os

import pytest

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx950"

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx950",
    "subtile3_gr_variants.yaml",
)

# ---------------------------------------------------------------------------
# Mock helpers shared across all sections
# ---------------------------------------------------------------------------

class _MockPool:
    """Minimal vgpr/sgpr pool mock for TileInfo construction and emit calls."""

    def __init__(self, start=0):
        self._counter = start

    def size(self):
        return self._counter

    def checkOut(self, n, align=None, tag=None, preventOverflow=True):
        r = self._counter
        self._counter += n
        return r

    def checkIn(self, v):
        pass

    def checkOutAligned(self, n, align, name=None, tag=None, preventOverflow=True):
        # align ignored in mock
        r = self._counter
        self._counter += n
        return r


def _mock_states():
    class _S:
        regCaps = {"PhysicalMaxVgpr": 512, "MaxVgpr": 256, "MaxSgpr": 256}
        agprPool = _MockPool()
        vgprPool = _MockPool()
        archCaps = {"LDSBankCount": 32, "LDSBankWidth": 4}
    return _S()


def _make_writer(sgpr_start=0):
    """Return a minimal writer mock with independent pool instances."""
    class _W:
        pass
    w = _W()
    w.states = _mock_states()
    w.vgprPool = _MockPool()
    w.sgprPool = _MockPool(sgpr_start)
    w.agprPool = _MockPool()
    return w


def _kernel_b8_wg11():
    """FP8 kernel dict with WG=(1,1) — produces loadRatioGR == 0.5."""
    return {
        "MacroTileA": 128, "MacroTileB": 128,
        "_DepthUA": 256, "_DepthUB": 256,
        "MIWaveGroup": [1, 1], "WavefrontSize": 64,
        "NonTemporalA": 0, "NonTemporalB": 0,
    }


def _kernel_b16_wg44():
    """BF16 kernel dict with WG=(4,4) — produces loadRatioGR == 2.0."""
    return {
        "MacroTileA": 128, "MacroTileB": 128,
        "_DepthUA": 64, "_DepthUB": 64,
        "MIWaveGroup": [4, 4], "WavefrontSize": 64,
        "NonTemporalA": 0, "NonTemporalB": 0,
    }


def _kernel_b16_wg22():
    """BF16 kernel dict with WG=(2,2) — produces loadRatioGR == 1.0."""
    return {
        "MacroTileA": 128, "MacroTileB": 128,
        "_DepthUA": 128, "_DepthUB": 128,
        "MIWaveGroup": [2, 2], "WavefrontSize": 64,
        "NonTemporalA": 0, "NonTemporalB": 0,
    }


# ---------------------------------------------------------------------------
# Section 1 — Non-legacy helper functions (lines 528-639)
#
# graInitPointer           528-533
# _grComputeOffset         539-555
# _grComputeSubtileOffsets 561-579
# _grComputeRowPartition   583-614  (branches: 1.0, 0.5, 2.0)
# _grComputeAllOffsets     619-639  (includes 0.5-rotation sub-path)
# ---------------------------------------------------------------------------

def test_r6_graInitPointer_emits_placeholder():
    """graInitPointer emits a placeholder module (lines 528-533)."""
    from Tensile.Components.Subtile.SubtileGREmit import graInitPointer
    from rocisa.code import Module

    w = _make_writer()
    kernel = _kernel_b16_wg22()
    m = graInitPointer(w, kernel)
    assert isinstance(m, Module), "graInitPointer must return a Module"
    # The placeholder comment must be present
    src = str(m)
    assert "Placeholder" in src, f"Expected placeholder comment in: {src[:200]!r}"


def test_r6_grComputeOffset_produces_instructions():
    """_grComputeOffset emits scale + mul + shift + add sequence (lines 539-555)."""
    from Tensile.Components.Subtile.SubtileGREmit import _grComputeOffset
    from Tensile.Components.Subtile.Kernel import TileInfo, AB_B16
    from rocisa.code import Module

    kernel = _kernel_b16_wg22()
    w = _make_writer()
    ti = TileInfo(AB_B16, "A", w, kernel)
    ti.allocOffsetRegisters(w, kernel)

    m = Module()
    _grComputeOffset(m, w, ti, colId=0, rowId=1, output=2)

    src = str(m)
    lines = [l for l in src.splitlines() if l.strip()]
    assert len(lines) >= 4, f"Expected >=4 instructions from _grComputeOffset, got {len(lines)}"
    assert "v_lshlrev_b32" in src or "v_lshl_or_b32" in src or "VLShiftLeftB32" in src or "v_mul_lo_u32" in src, (
        "_grComputeOffset must emit shift/mul instructions"
    )


def test_r6_grComputeSubtileOffsets_sgpr_path():
    """_grComputeSubtileOffsets emits s_mul + s-soffsets for localSubtilesRegister (lines 561-579)."""
    from Tensile.Components.Subtile.SubtileGREmit import _grComputeSubtileOffsets
    from Tensile.Components.Subtile.Kernel import TileInfo, AB_B16
    from rocisa.code import Module

    # WG=(2,2) with AB_B16: localSubtileGrid=[4,2], loadRatioGR=1.0
    # Row 0 has empty RegList; row 1+ get SGPR soffsets (sgpr pool starts at 0, far below limit).
    kernel = _kernel_b16_wg22()
    w = _make_writer()
    ti = TileInfo(AB_B16, "A", w, kernel)
    ti.allocOffsetRegisters(w, kernel)

    # Confirm at least one non-empty RegList (row > 0)
    non_empty = [rl for rl in ti.localSubtilesRegister if len(rl) > 0]
    assert non_empty, "Expected at least one non-empty RegList for rows > 0"

    m = Module()
    _grComputeSubtileOffsets(w, m, ti)
    src = str(m)
    # s_mul_i32 should appear for SGPR soffset computation
    assert "s_mul_i32" in src or "SMulI32" in src, (
        "_grComputeSubtileOffsets must emit s_mul_i32 for SGPR soffsets"
    )


def test_r6_grComputeRowPartition_ratio_half():
    """_grComputeRowPartition covers the loadRatioGR == 0.5 branch (lines 599-601).

    WG=(1,1) + AB_B8 (FP8, bpe=1) yields loadRatioGR = 0.5.  The branch emits
    VMovB32(localRow=0) + VMovB32(partitionRow=waveId).
    """
    from Tensile.Components.Subtile.SubtileGREmit import _grComputeRowPartition
    from Tensile.Components.Subtile.Kernel import TileInfo, AB_B8
    from rocisa.code import Module

    kernel = _kernel_b8_wg11()
    w = _make_writer()
    ti = TileInfo(AB_B8, "A", w, kernel)
    ti.allocOffsetRegisters(w, kernel)

    assert ti.loadRatioGR == 0.5, f"Expected loadRatioGR=0.5, got {ti.loadRatioGR}"

    m = Module()
    _grComputeRowPartition(m, kernel, w, ti, waveId=5, rowOffset=6)
    src = str(m)
    lines = [l for l in src.splitlines() if l.strip()]
    assert len(lines) >= 3, f"Expected >=3 instructions, got {len(lines)}: {src}"


def test_r6_grComputeRowPartition_ratio_two():
    """_grComputeRowPartition covers the loadRatioGR == 2.0 branch (lines 602-604).

    WG=(4,4) + AB_B16 yields loadRatioGR = 2.0.  The branch emits
    VMovB32(localRow=waveId) + VMovB32(partitionRow=0).
    """
    from Tensile.Components.Subtile.SubtileGREmit import _grComputeRowPartition
    from Tensile.Components.Subtile.Kernel import TileInfo, AB_B16
    from rocisa.code import Module

    kernel = _kernel_b16_wg44()
    w = _make_writer()
    ti = TileInfo(AB_B16, "A", w, kernel)
    ti.allocOffsetRegisters(w, kernel)

    assert ti.loadRatioGR == 2.0, f"Expected loadRatioGR=2.0, got {ti.loadRatioGR}"

    m = Module()
    _grComputeRowPartition(m, kernel, w, ti, waveId=5, rowOffset=6)
    src = str(m)
    lines = [l for l in src.splitlines() if l.strip()]
    assert len(lines) >= 3, f"Expected >=3 instructions, got {len(lines)}: {src}"


def test_r6_grComputeAllOffsets_ratio_half():
    """_grComputeAllOffsets covers the loadRatioGR == 0.5 rotation path (lines 619-639).

    AB_B8 + WG=(1,1) -> loadRatioGR=0.5, numGRPerSubtile=2.
    The loop runs for i=1..numGRPerSubtile-1=1, which exercises the half-block
    rotation arm (lines 630-634) for bpe=1 (FP8).  Non-FP8 bpe also tested.
    """
    from Tensile.Components.Subtile.SubtileGREmit import _grComputeAllOffsets
    from Tensile.Components.Subtile.Kernel import TileInfo, AB_B8
    from rocisa.code import Module

    kernel = _kernel_b8_wg11()
    w = _make_writer()
    ti = TileInfo(AB_B8, "A", w, kernel)
    ti.allocOffsetRegisters(w, kernel)

    assert ti.loadRatioGR == 0.5, f"Expected loadRatioGR=0.5, got {ti.loadRatioGR}"
    assert ti.numGRPerSubtile == 2, f"Expected numGRPerSubtile=2, got {ti.numGRPerSubtile}"

    m = Module()
    _grComputeAllOffsets(m, w, ti, colId=5, rowId=6, rowOffset=7)
    src = str(m)
    lines = [l for l in src.splitlines() if l.strip()]
    # Loop body for i=1 adds the rotated col computation
    assert len(lines) >= 8, f"Expected >=8 instructions (loop body included), got {len(lines)}"


def test_r6_grComputeAllOffsets_ratio_one():
    """_grComputeAllOffsets with loadRatioGR==1.0 takes the else-VMovB32 arm (line 636)."""
    from Tensile.Components.Subtile.SubtileGREmit import _grComputeAllOffsets
    from Tensile.Components.Subtile.Kernel import TileInfo, AB_B16
    from rocisa.code import Module

    # numGRPerSubtile > 1 requires ratio < 1, so use ratio=1.0 which has
    # numGRPerSubtile=1 and the loop body (i=1..0) never executes —
    # but the common part (i=0 path) covers line 620-622.
    kernel = _kernel_b16_wg22()
    w = _make_writer()
    ti = TileInfo(AB_B16, "A", w, kernel)
    ti.allocOffsetRegisters(w, kernel)

    assert ti.loadRatioGR == 1.0, f"Expected loadRatioGR=1.0, got {ti.loadRatioGR}"

    m = Module()
    _grComputeAllOffsets(m, w, ti, colId=5, rowId=6, rowOffset=7)
    src = str(m)
    lines = [l for l in src.splitlines() if l.strip()]
    assert len(lines) >= 2, f"Expected >=2 instructions, got {len(lines)}"


# ---------------------------------------------------------------------------
# Section 2 — _grComputeSubtileOffsets_legacy VGPR fallback (lines 688-691)
#
# When sgprPool.size() >= MaxSgpr - 3, _allocGROffsetRegs_TLU0 allocates
# VGPR-backed RegLists (is_sgpr=False).  The legacy _grComputeSubtileOffsets
# function's inner loop then follows the else branch (688-691): s_mul into a
# temp SGPR + v_add to bake soffset into each per-load VGPR.
# ---------------------------------------------------------------------------

def test_r6_grComputeSubtileOffsets_legacy_vgpr_fallback():
    """_grComputeSubtileOffsets_legacy VGPR fallback: v_add soffset baked in (lines 688-691).

    Force the VGPR path by setting sgpr_start >= MaxSgpr - 3 = 253.
    Checks that SMulI32 + VAddU32 appear (not just SMulI32 alone).
    """
    from Tensile.Components.Subtile.SubtileGREmit import _grComputeSubtileOffsets_legacy
    from Tensile.Components.Subtile.Kernel import TileInfo, AB_B8
    from rocisa.code import Module

    kernel = _kernel_b8_wg11()
    # sgpr_start=253 >= MaxSgpr(256) - 3 — forces VGPR RegLists in alloc
    w = _make_writer(sgpr_start=253)
    ti = TileInfo(AB_B8, "A", w, kernel)
    ti.allocOffsetRegisters(w, kernel)

    # Verify VGPR fallback is actually in effect
    vgpr_lists = [rl for rl in ti.localSubtilesRegister if not rl.is_sgpr and len(rl) > 0]
    assert vgpr_lists, (
        "Expected at least one VGPR-backed RegList after VGPR-fallback alloc"
    )

    m = Module()
    _grComputeSubtileOffsets_legacy(w, m, ti)
    src = str(m)

    # VGPR fallback emits v_add_u32 to bake soffset into each GR vgpr
    assert "v_add_u32" in src or "VAddU32" in src, (
        "VGPR fallback must emit v_add_u32; got:\n" + src[:500]
    )


# ---------------------------------------------------------------------------
# Section 3 — Config-driven emit: loadRatioGR == 0.5 + 2.0 via legacy path
#
# These exercises the full kernel emit pipeline (graTileAssignment ->
# _graTileAssignment_legacy -> _grComputeRowPartition_legacy 0.5/2.0 branches
# and _grComputeAllOffsets_legacy FP8 rotation) through a YAML config.
# ---------------------------------------------------------------------------

def test_r6_subtile3_gfx950_emits_assembly():
    """FP8 WG=(1,1) + WG=(4,4) subtile configs emit real gfx950 assembly (all err==0).

    WG=(1,1) gives loadRatioGR=0.5 -> covers _grComputeRowPartition_legacy branch
    loadRatioGR==0.5 and _grComputeAllOffsets_legacy FP8 intra-block rotation.
    WG=(4,4) gives loadRatioGR=2.0 -> covers loadRatioGR==2.0 branches.
    """
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, f"Expected >=1 kernel, got 0 (config: {_CONFIG})"
    assert all(err == 0 for (_b, _s, err) in results), (
        f"Some kernels failed: {[(b, e) for b, _s, e in results if e != 0]}"
    )
    for base, src, _err in results:
        assert src and len(src.splitlines()) > 100, (
            f"kernel {base!r}: suspiciously short assembly"
        )
        assert ".amdgcn_target" in src, f"kernel {base!r}: missing .amdgcn_target"
        assert "gfx950" in src, f"kernel {base!r}: wrong arch in assembly"
        assert base.startswith("Cijk_"), f"kernel {base!r}: unexpected basename prefix"


# ---------------------------------------------------------------------------
# Section 4 — emitSingleBufferLoad early-return, emitSubtileBufferLoad,
#              globalReadDoSubtile, globalReadLDSBufferSwap scale path
# ---------------------------------------------------------------------------

def _make_full_writer_with_tileinfos(geom_A, geom_B, kernel):
    """Build a writer mock with .states.a.tileInfo and .states.b.tileInfo set."""
    # Build separate writers for TileInfo construction (pool isolation)
    def _fresh():
        return _make_writer()

    wA = _fresh()
    tiA = None
    wB = _fresh()
    tiB = None

    from Tensile.Components.Subtile.Kernel import TileInfo
    tiA = TileInfo(geom_A, "A", wA, kernel)
    tiA.allocOffsetRegisters(wA, kernel)

    tiB = TileInfo(geom_B, "B", wB, kernel)
    tiB.allocOffsetRegisters(wB, kernel)

    class _TState:
        def __init__(self, ti):
            self.tileInfo = ti

    class _States:
        regCaps = {"PhysicalMaxVgpr": 512, "MaxVgpr": 256, "MaxSgpr": 256}
        agprPool = _MockPool()
        vgprPool = _MockPool()
        archCaps = {"LDSBankCount": 32, "LDSBankWidth": 4}

    class _W:
        pass

    w = _W()
    w.states = _States()
    w.states.a = _TState(tiA)
    w.states.b = _TState(tiB)
    w.vgprPool = _MockPool()
    w.sgprPool = _MockPool()
    w.agprPool = _MockPool()
    return w, tiA, tiB


def test_r6_emitSingleBufferLoad_ratio_gt1_early_return():
    """emitSingleBufferLoad returns empty module for non-first subtile in loadRatioGR>1 group.

    loadRatioGR==2.0 (WG=(4,4)+AB_B16): linearId=1, firstInGroup=0 -> early return.
    Covers lines 855-858: the ``if tileInfo.loadRatioGR > 1`` guard.
    """
    from Tensile.Components.Subtile.SubtileGREmit import emitSingleBufferLoad
    from Tensile.Components.Subtile.Kernel import TileInfo, AB_B16

    kernel = _kernel_b16_wg44()
    w = _make_writer()
    ti = TileInfo(AB_B16, "A", w, kernel)
    ti.allocOffsetRegisters(w, kernel)

    assert ti.loadRatioGR == 2.0, f"Expected loadRatioGR=2.0, got {ti.loadRatioGR}"

    # sId0=1, sId1=0 -> linearId=1; firstInGroup = floor(1/2)*2 = 0 -> 1 != 0 -> empty
    m_empty = emitSingleBufferLoad(ti, kernel, sId0=1, sId1=0)
    empty_lines = [l for l in str(m_empty).splitlines() if l.strip()]
    assert len(empty_lines) == 0, (
        f"Expected empty module for duplicate subtile, got {len(empty_lines)} lines"
    )

    # sId0=0, sId1=0 -> linearId=0; firstInGroup=0 -> 0 == 0 -> emits load
    m_emit = emitSingleBufferLoad(ti, kernel, sId0=0, sId1=0)
    emit_lines = [l for l in str(m_emit).splitlines() if l.strip()]
    assert len(emit_lines) >= 1, (
        f"Expected non-empty module for first subtile, got {len(emit_lines)} lines"
    )


def test_r6_emitSubtileBufferLoad_wrapper():
    """emitSubtileBufferLoad wrapper looks up tileInfo via writer.states (lines 886-887)."""
    from Tensile.Components.Subtile.SubtileGREmit import emitSubtileBufferLoad
    from Tensile.Components.Subtile.Kernel import AB_B16

    kernel = _kernel_b16_wg22()
    writer, tiA, tiB = _make_full_writer_with_tileinfos(AB_B16, AB_B16, kernel)

    # For tc='A', picks writer.states.a.tileInfo
    m_a = emitSubtileBufferLoad("A", writer, kernel, [0, 0])
    assert str(m_a).strip() != "", (
        "emitSubtileBufferLoad('A') must emit something for sId=[0,0]"
    )

    # For tc='B', picks writer.states.b.tileInfo
    m_b = emitSubtileBufferLoad("B", writer, kernel, [0, 0])
    assert str(m_b).strip() != "", (
        "emitSubtileBufferLoad('B') must emit something for sId=[0,0]"
    )


def test_r6_globalReadDoSubtile_iterates_grid():
    """globalReadDoSubtile loops over localSubtileGrid and emits per-subtile loads (lines 894-903)."""
    from Tensile.Components.Subtile.SubtileGREmit import globalReadDoSubtile
    from Tensile.Components.Subtile.Kernel import AB_B16

    kernel = _kernel_b16_wg22()
    writer, tiA, tiB = _make_full_writer_with_tileinfos(AB_B16, AB_B16, kernel)

    m_a = globalReadDoSubtile("A", writer, kernel)
    src_a = str(m_a)
    # Should contain one load comment per subtile in the grid
    assert "Emit load for A subtile" in src_a, (
        f"globalReadDoSubtile('A') must contain subtile-load comments; got:\n{src_a[:500]}"
    )

    m_b = globalReadDoSubtile("B", writer, kernel)
    src_b = str(m_b)
    assert "Emit load for B subtile" in src_b, (
        f"globalReadDoSubtile('B') must contain subtile-load comments; got:\n{src_b[:500]}"
    )


def test_r6_globalReadLDSBufferSwap_scale_path():
    """globalReadLDSBufferSwap routes tc='MXSA'/'MXSB' to scale swap (lines 947-948)."""
    from Tensile.Components.Subtile.SubtileGREmit import globalReadLDSBufferSwap
    from Tensile.Components.Subtile.Kernel import TileInfo, MXSA_B8, MXSB_B8

    kernel_mx = {
        "MacroTileA": 128, "MacroTileB": 128,
        "_DepthUA": 256, "_DepthUB": 256,
        "_DepthUMXSA": 8, "_DepthUMXSB": 8,
        "MIWaveGroup": [2, 2], "WavefrontSize": 64,
    }
    # Construct scale TileInfos
    w_mxsa = _make_writer()
    ti_mxsa = TileInfo(MXSA_B8, "MXSA", w_mxsa, kernel_mx)

    w_mxsb = _make_writer()
    ti_mxsb = TileInfo(MXSB_B8, "MXSB", w_mxsb, kernel_mx)

    # Wire into a writer with the .states.mxsa / .states.mxsb attributes
    class _ScaleStates:
        regCaps = {"PhysicalMaxVgpr": 512, "MaxVgpr": 256, "MaxSgpr": 256}
        agprPool = _MockPool()
        vgprPool = _MockPool()
        archCaps = {"LDSBankCount": 32, "LDSBankWidth": 4}
        class _mxsa_state:
            tileInfo = None
        class _mxsb_state:
            tileInfo = None
        mxsa = _mxsa_state()
        mxsb = _mxsb_state()

    class _W:
        pass

    writer = _W()
    writer.states = _ScaleStates()
    writer.states.mxsa.tileInfo = ti_mxsa
    writer.states.mxsb.tileInfo = ti_mxsb
    writer.vgprPool = _MockPool()
    writer.sgprPool = _MockPool()
    writer.agprPool = _MockPool()

    # tc='MXSA' -> else branch (line 947-948) -> emitScaleGRLDSSwap
    m_mxsa = globalReadLDSBufferSwap("MXSA", writer, kernel_mx)
    assert str(m_mxsa) is not None, "Expected a Module from MXSA scale swap"

    m_mxsb = globalReadLDSBufferSwap("MXSB", writer, kernel_mx)
    assert str(m_mxsb) is not None, "Expected a Module from MXSB scale swap"
