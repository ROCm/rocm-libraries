#!/usr/bin/env python3
################################################################################
# Step 2 LDS→MFMA validation: LR offset-register ALLOCATION for NN-FP4
# (LRTag_TLU1). Pure-CPU (no GPU / no hip).
#
# Confirms:
#   * LRTag_TLU1 dispatches to _allocLROffsetRegs_TLU1 (not the _stub).
#   * 4 offset VGPRs + 4 swap VGPRs allocated (numLRPerSubtile == 4).
#   * All 8 VGPRs are distinct (no aliasing).
#   * GR alloc is unaffected.
#   * Dealloc cleans up all registers.
#
#   pytest test_nn_f4_lr_alloc.py -v   |   python test_nn_f4_lr_alloc.py
################################################################################
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, TENSILE_ROOT)
from types import SimpleNamespace
from rocisa.register import RegisterPool
from rocisa.enum import RegisterType
from Tensile.Components.Subtile.Kernel import TileInfo, AB_B4_TLU1


def _tr_sample_kernel():
    return {
        "DepthU": 256,
        "_DepthU": 256,
        "_DepthUA": 256,
        "_DepthUB": 256,
        "MacroTileA": 256,
        "MacroTileB": 256,
        "MacroTile0": 256,
        "MacroTile1": 256,
        "MatrixInstM": 16,
        "MatrixInstN": 16,
        "MatrixInstK": 128,
        "MIWaveGroup": [2, 2],
        "WavefrontSize": 64,
        "UseSubtileImpl": True,
        "NonTemporalA": 0,
        "NonTemporalB": 0,
        "ProblemType": {
            "DataTypeA": None,
            "DataTypeB": None,
            "ComputeDataType": None,
        },
    }


def _make_writer():
    w = SimpleNamespace()
    w.vgprPool = RegisterPool(
        0, RegisterType.Vgpr, defaultPreventOverflow=False, printRP=False
    )
    w.sgprPool = RegisterPool(
        0, RegisterType.Sgpr, defaultPreventOverflow=False, printRP=False
    )
    w.states = SimpleNamespace(
        regCaps={"MaxSgpr": 106, "MaxVgpr": 256, "PhysicalMaxVgpr": 512},
        archCaps={"LDSBankCount": 64, "LDSBankWidth": 4},
    )
    return w


# ── Test 1: Dispatch routes to real function, not stub ───────────────────────
def test_tlu1_dispatch_not_stub():
    """LRTag_TLU1 must dispatch to _allocLROffsetRegs_TLU1, not _stub."""
    from Tensile.Components.Subtile.SubtileLREmit import (
        _allocLROffsetRegisters,
        _deallocLROffsetRegisters,
    )
    from Tensile.Components.Subtile.SubtileGeometry import LRTag_TLU1

    alloc_name = _allocLROffsetRegisters.dispatch(LRTag_TLU1).__name__
    dealloc_name = _deallocLROffsetRegisters.dispatch(LRTag_TLU1).__name__
    assert (
        alloc_name == "_allocLROffsetRegs_TLU1"
    ), f"alloc dispatches to {alloc_name}, expected _allocLROffsetRegs_TLU1"
    assert (
        dealloc_name == "_deallocLROffsetRegs_TLU1"
    ), f"dealloc dispatches to {dealloc_name}, expected _deallocLROffsetRegs_TLU1"


# ── Test 2: Alloc reserves 4 offset + 4 swap VGPRs ──────────────────────────
def test_tlu1_alloc_correct_count():
    """4 offset VGPRs + 4 swap VGPRs (numLRPerSubtile == 4)."""
    kernel, writer = _tr_sample_kernel(), _make_writer()
    tiA = TileInfo(AB_B4_TLU1, "A", writer, kernel)
    assert tiA.numLRPerSubtile == 4, "precondition from Step 1"
    tiA.allocOffsetRegisters(writer, kernel)
    assert (
        len(tiA.sharedVgprLROffset) == 4
    ), f"sharedVgprLROffset has {len(tiA.sharedVgprLROffset)} entries, expected 4"
    assert (
        len(tiA.sharedVgprLROffsetSwap) == 4
    ), f"sharedVgprLROffsetSwap has {len(tiA.sharedVgprLROffsetSwap)} entries, expected 4"
    tiA.deallocOffsetRegisters(writer, kernel)


# ── Test 3: All 8 VGPRs are distinct ────────────────────────────────────────
def test_tlu1_alloc_no_aliasing():
    """No two VGPRs alias (offset and swap combined)."""
    kernel, writer = _tr_sample_kernel(), _make_writer()
    tiA = TileInfo(AB_B4_TLU1, "A", writer, kernel)
    tiA.allocOffsetRegisters(writer, kernel)
    all_vgprs = list(tiA.sharedVgprLROffset) + list(tiA.sharedVgprLROffsetSwap)
    assert len(all_vgprs) == 8
    assert len(set(all_vgprs)) == 8, f"VGPR aliasing detected: {all_vgprs}"
    tiA.deallocOffsetRegisters(writer, kernel)


# ── Test 4: GR alloc is not affected ────────────────────────────────────────
def test_gr_alloc_unchanged():
    """GR register allocation must be identical to before Step 2."""
    kernel, writer = _tr_sample_kernel(), _make_writer()
    tiA = TileInfo(AB_B4_TLU1, "A", writer, kernel)
    tiA.allocOffsetRegisters(writer, kernel)
    assert len(tiA.sharedVgprGROffset) == 1, "GR: 1 per-lane offset VGPR"
    assert len(tiA.localSubtilesRegister) == 2, "GR: 2 K-subtile register groups"
    assert len(tiA.localSubtilesRegister[0]) == 0, "GR: K-subtile 0 = no soffset"
    assert len(tiA.localSubtilesRegister[1]) == 1, "GR: K-subtile 1 = 1 SGPR"
    tiA.deallocOffsetRegisters(writer, kernel)


# ── Test 5: Dealloc cleans up everything ────────────────────────────────────
def test_tlu1_dealloc_clears():
    """After dealloc, offset and swap lists are empty."""
    kernel, writer = _tr_sample_kernel(), _make_writer()
    tiA = TileInfo(AB_B4_TLU1, "A", writer, kernel)
    tiA.allocOffsetRegisters(writer, kernel)
    assert len(tiA.sharedVgprLROffset) == 4, "precondition: alloc ran"
    assert len(tiA.sharedVgprLROffsetSwap) == 4
    tiA.deallocOffsetRegisters(writer, kernel)
    assert (
        len(tiA.sharedVgprLROffset) == 0
    ), f"sharedVgprLROffset not empty after dealloc: {tiA.sharedVgprLROffset}"
    assert (
        len(tiA.sharedVgprLROffsetSwap) == 0
    ), f"sharedVgprLROffsetSwap not empty after dealloc: {tiA.sharedVgprLROffsetSwap}"


# ── Test 6: TileInfo convenience accessors work ─────────────────────────────
def test_tileinfo_accessors():
    """numLROffsetVgprs and lrOffsetVgpr(i) work after alloc."""
    kernel, writer = _tr_sample_kernel(), _make_writer()
    tiA = TileInfo(AB_B4_TLU1, "A", writer, kernel)
    tiA.allocOffsetRegisters(writer, kernel)
    assert tiA.numLROffsetVgprs == 4
    for i in range(4):
        v = tiA.lrOffsetVgpr(i)
        assert isinstance(v, int) and v >= 0, f"lrOffsetVgpr({i}) returned {v}"
        s = tiA.lrSwapVgpr(i)
        assert isinstance(s, int) and s >= 0, f"lrSwapVgpr({i}) returned {s}"
        assert v != s, f"offset and swap alias at index {i}"
    tiA.deallocOffsetRegisters(writer, kernel)


if __name__ == "__main__":
    test_tlu1_dispatch_not_stub()
    test_tlu1_alloc_correct_count()
    test_tlu1_alloc_no_aliasing()
    test_gr_alloc_unchanged()
    test_tlu1_dealloc_clears()
    test_tileinfo_accessors()
    print(
        "OK: LRTag_TLU1 alloc/dealloc reserves 4+4 VGPRs; GR unchanged; clean dealloc."
    )
