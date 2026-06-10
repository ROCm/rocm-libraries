#!/usr/bin/env python3
################################################################################
# Step 2 validation: dedicated GR offset-register ALLOCATION for NN-FP4
# (GRTag_TLU1). Pure-CPU (no GPU / no hip).
#
# Confirms:
#   * GRTag_TLU1 dispatches to the DEDICATED _allocGROffsetRegs_TLU1 /
#     _deallocGROffsetRegs_TLU1 (NOT the TLU0-named functions).
#   * It reserves the correct registers for AB_B4_TLU1 and frees them.
#   * The TLU=0 path (AB_B4) is unchanged.
#
#   pytest test_nn_f4_gr_alloc.py -v   |   python test_nn_f4_gr_alloc.py
################################################################################
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, TENSILE_ROOT)
from types import SimpleNamespace
from rocisa.register import RegisterPool
from rocisa.enum import RegisterType
from Tensile.Components.Subtile.Kernel import TileInfo, AB_B4_TLU1, AB_B4


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
        "ProblemType": {"DataTypeA": None, "DataTypeB": None, "ComputeDataType": None},
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


def test_tlu1_uses_dedicated_allocator_not_tlu0():
    """The core of your concern: GRTag_TLU1 must NOT be served by _TLU0 funcs."""
    from Tensile.Components.Subtile.SubtileGREmit import (
        _allocGROffsetRegisters,
        _deallocGROffsetRegisters,
        GRTag_TLU1,
    )

    assert (
        _allocGROffsetRegisters.dispatch(GRTag_TLU1).__name__
        == "_allocGROffsetRegs_TLU1"
    )
    assert (
        _deallocGROffsetRegisters.dispatch(GRTag_TLU1).__name__
        == "_deallocGROffsetRegs_TLU1"
    )


def test_tlu1_alloc_reserves_expected_registers():
    """The dedicated TLU1 allocator reserves real registers (was a no-op stub)."""
    kernel, writer = _tr_sample_kernel(), _make_writer()
    tiA = TileInfo(AB_B4_TLU1, "A", writer, kernel)
    # Geometry facts Step 1 locked down (guards against silent geometry drift).
    assert tiA.numGRPerSubtile == 1
    assert list(tiA.localSubtileGrid) == [4, 2]
    tiA.allocOffsetRegisters(writer, kernel)
    # 1 per-lane vaddr VGPR (numGRPerSubtile == 1); would be 0 if still a stub.
    assert len(tiA.sharedVgprGROffset) == 1

    # AB_B4_TLU1 is COLUMN-MAJOR: the strided (soffset) direction is K, so the
    # dedicated TLU1 allocator is K-indexed -> one RegList per K-subtile.
    #   numKSubtiles = localSubtileGrid[1] = 2   (NOT localSubtileGrid[0] = 4)
    # M-subtiles ride the buffer_load offset12 immediate and need NO register.
    assert len(tiA.localSubtilesRegister) == 2
    assert len(tiA.localSubtilesRegister[0]) == 0    # base K-subtile (k=0): soffset 0
    assert len(tiA.localSubtilesRegister[1]) == 1    # K-subtile 1: one soffset SGPR
    assert tiA.localSubtilesRegister[1].is_sgpr

    tiA.deallocOffsetRegisters(writer, kernel)
    assert len(tiA.sharedVgprGROffset) == 0
    assert len(tiA.localSubtilesRegister) == 0


def test_tlu0_alloc_unchanged():
    """Regression: the existing TLU=0 FP4 path (AB_B4) allocates exactly as before."""
    kernel, writer = _tr_sample_kernel(), _make_writer()
    tiA = TileInfo(AB_B4, "A", writer, kernel)
    assert list(tiA.localSubtileGrid) == [8, 1]
    tiA.allocOffsetRegisters(writer, kernel)
    assert len(tiA.sharedVgprGROffset) == 1
    assert len(tiA.localSubtilesRegister) == 8  # ceil(8/1)
    tiA.deallocOffsetRegisters(writer, kernel)
    assert len(tiA.sharedVgprGROffset) == 0


if __name__ == "__main__":
    test_tlu1_uses_dedicated_allocator_not_tlu0()
    test_tlu1_alloc_reserves_expected_registers()
    test_tlu0_alloc_unchanged()
    print("OK: GRTag_TLU1 uses dedicated TLU1 alloc/dealloc; TLU0 unchanged.")
