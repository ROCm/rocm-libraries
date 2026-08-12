################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""P4 — LRA transposed-MFMA component direct-invocation characterization.

Directly instantiates and invokes ``LraTileAssignmentTransposedMFMA``,
``LraTileAssignmentTransposedMFMAB8``, ``LraTileAssignmentTransposedMFMAF4``
and ``LraTileAssignmentTransposedMFMAF6`` with a minimal mock writer, bypassing
the ``Component.find()`` asmCaps dispatch filter.

Target missing ranges in Tensile/Components/LraTileAssignment.py:
  144-249  : LraTileAssignmentTransposedMFMA.__call__
             (BF16, enableLDSTr=True, isM=False, wave-offset branch num1DWaves>1)
  285-409  : LraTileAssignmentTransposedMFMAB8.__call__
             (I8, enableLDSTr=True, isM=False and isM=True branches)
  473-592  : LraTileAssignmentTransposedMFMAF4.__call__
             (F4, enableLDSTr=True, num1DWaves>1)
  611-693  : LraTileAssignmentTransposedMFMAF6.__call__
             (F6, enableLDSTr=True, num1DWaves>1)

Hardware-cap note: HasLDSTrB128B16 / HasLDSTrB64B8 / HasLDSTrB64B4 / HasLDSTrB96B6
are gfx950-specific caps. On this container gfx950 instance these are 0, so
``Component.find()`` falls back to ``LraTileAssignmentMFMA`` for these types.
Direct invocation is the only CPU-only path to exercise these __call__ bodies.
These components are real production code paths on hardware with the caps set.

The mock writer supplies:
  - ``vgprPool`` (rocisa RegisterPool, Vgpr type, 256 regs)
  - ``sgprPool`` (rocisa RegisterPool, Sgpr type, 256 regs)
  - ``allocTmpSgpr(n, tag=None)`` (context manager yielding ContinuousRegister)
  - ``states.kernel["WavefrontSize"]``  = 64
  - ``states.lraTileProperties``        = {} (component writes tile01 key)
  - ``states.bpr``                      = 4 (bytes per register)

The test asserts each __call__ returns a rocisa.code.Module and that
``tP["gpr"]["lro"]`` is set (the component's single output).
"""

import pytest
from types import SimpleNamespace

pytestmark = pytest.mark.unit

# Module-level import of KernelWriter is required to break the circular import
# that occurs when Tensile.Components.LraTileAssignment is imported before the
# parent Tensile package finishes loading (Component.py line 295:
# ``from .Components import *``).  Importing KernelWriter here ensures the
# Tensile package is fully initialized before any per-test import of a
# specific component module.
import rocisa  # noqa: F401
import Tensile.KernelWriter  # noqa: F401


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_writer():
    """Return a minimal mock writer with rocisa register pools."""

    from rocisa._rocisa.enum import Vgpr, Sgpr
    from rocisa.register import RegisterPool
    from Tensile.Common.RegisterPool import allocTmpGpr

    vgpr_pool = RegisterPool(256, Vgpr, False)
    vgpr_pool.add(0, 256, "init")

    sgpr_pool = RegisterPool(256, Sgpr, False)
    sgpr_pool.add(0, 256, "init")

    def allocTmpSgpr(num, tag=None):
        return allocTmpGpr(sgpr_pool, num, 256, tag=tag)

    states = SimpleNamespace(
        kernel={"WavefrontSize": 64},
        lraTileProperties={},
        bpr=4,
    )

    return SimpleNamespace(
        vgprPool=vgpr_pool,
        sgprPool=sgpr_pool,
        states=states,
        allocTmpSgpr=allocTmpSgpr,
    )


def _make_tP(tile01, tc, is_m=False, is_a=False, is_b=True, block_width=1, bpe_ds=2):
    """Return a minimal tensor-parameters dict for the given tile side."""
    local_read_inst = SimpleNamespace(blockWidth=block_width)
    tile_char = "M" if is_m else ("N" if tile01 == 1 else "M")
    return {
        "enableLDSTr": True,
        "isM": is_m,
        "isA": is_a,
        "isB": is_b,
        "tileChar": tile_char,
        "tile01Idx": tile01,
        "tensorChar": tc,
        "localReadInstruction": local_read_inst,
        "bpeDS": bpe_ds,
        "gpr": {},
    }


def _make_kernel_bf16():
    """Kernel dict for BF16 NN — exercises LraTileAssignmentTransposedMFMA."""
    from Tensile.Common.DataType import DataType
    return {
        "MatrixInstM": 16,
        "MatrixInstN": 16,
        "MatrixInstB": 1,
        "MatrixInstBM": 4,
        "MatrixInstBN": 4,
        "MIWaveGroup": [2, 2],   # num1DWaves=2 for both tiles -> wave-offset branch
        "SourceSwap": True,
        "VectorWidthA": 1,
        "VectorWidthB": 1,
        "MacroTile0": 64,
        "MacroTile1": 64,
        "LdsPadA": 0,
        "LdsPadB": 0,
        "LdsBlockSizePerPadA": 0,
        "LdsBlockSizePerPadB": 0,
        "ProblemType": {
            "Sparse": 0,
            "DataType": DataType("b"),  # BF16
        },
    }


def _make_kernel_i8():
    """Kernel dict for I8 — exercises LraTileAssignmentTransposedMFMAB8."""
    from Tensile.Common.DataType import DataType
    return {
        "MatrixInstM": 16,
        "MatrixInstN": 16,
        "MatrixInstB": 1,
        "MatrixInstBM": 4,
        "MatrixInstBN": 4,
        "MIWaveGroup": [2, 2],
        "SourceSwap": False,
        "VectorWidthA": 1,
        "VectorWidthB": 1,
        "MacroTile0": 64,
        "MacroTile1": 64,
        "LdsPadA": 0,
        "LdsPadB": 0,
        "LdsBlockSizePerPadA": 0,
        "LdsBlockSizePerPadB": 0,
        "MIInputPerThreadA": 4,
        "MIInputPerThreadB": 4,
        "ProblemType": {
            "Sparse": 0,
            "DataType": DataType("I8"),
        },
    }


def _make_kernel_f4():
    """Kernel dict for F4 — exercises LraTileAssignmentTransposedMFMAF4."""
    from Tensile.Common.DataType import DataType
    return {
        "MatrixInstM": 16,
        "MatrixInstN": 16,
        "MatrixInstB": 1,
        "MatrixInstBM": 4,
        "MatrixInstBN": 4,
        "MIWaveGroup": [2, 2],
        "SourceSwap": False,
        "VectorWidthA": 1,
        "VectorWidthB": 1,
        "MacroTile0": 64,
        "MacroTile1": 64,
        "LdsPadA": 0,
        "LdsPadB": 0,
        "LdsBlockSizePerPadA": 0,
        "LdsBlockSizePerPadB": 0,
        "ProblemType": {
            "Sparse": 0,
            "DataType": DataType("F4"),
            "isMixMode": False,
        },
    }


def _make_kernel_f6():
    """Kernel dict for F6 — exercises LraTileAssignmentTransposedMFMAF6."""
    from Tensile.Common.DataType import DataType
    return {
        "MatrixInstM": 16,
        "MatrixInstN": 16,
        "MatrixInstB": 1,
        "MatrixInstBM": 4,
        "MatrixInstBN": 4,
        "MIWaveGroup": [2, 2],
        "SourceSwap": False,
        "VectorWidthA": 1,
        "VectorWidthB": 1,
        "MacroTile0": 64,
        "MacroTile1": 64,
        "LdsPadA": 0,
        "LdsPadB": 0,
        "LdsBlockSizePerPadA": 0,
        "LdsBlockSizePerPadB": 0,
        "ProblemType": {
            "Sparse": 0,
            "DataType": DataType("F6"),
            "isMixMode": False,
        },
    }


# ---------------------------------------------------------------------------
# Tests: LraTileAssignmentTransposedMFMA (lines 144-249)
# ---------------------------------------------------------------------------

class TestLraTileAssignmentTransposedMFMA:
    """Direct-invoke tests for LraTileAssignmentTransposedMFMA (BF16 path).

    Lines 148-249 require enableLDSTr=True AND isM=False (not dispatched to
    LraTileAssignmentTransposedMFMAB8). The two tile sides (tile01=0 and
    tile01=1) exercise different num1DBlocks/num1DWaves branches.
    """

    def test_tile01_0_returns_module(self):
        """tile01=0 (A-side / M-tile) executes BF16 transposed LRA."""
        from Tensile.Components.LraTileAssignment import LraTileAssignmentTransposedMFMA
        writer = _make_writer()
        kernel = _make_kernel_bf16()
        tP = _make_tP(tile01=0, tc="A", is_m=False, is_a=True, is_b=False,
                      block_width=1, bpe_ds=2)
        module = LraTileAssignmentTransposedMFMA()(writer, kernel, tP)
        from rocisa.code import Module
        assert isinstance(module, Module)
        assert "lro" in tP["gpr"], "tP['gpr']['lro'] must be set by the component"

    def test_tile01_1_returns_module(self):
        """tile01=1 (B-side / N-tile) executes BF16 transposed LRA."""
        from Tensile.Components.LraTileAssignment import LraTileAssignmentTransposedMFMA
        writer = _make_writer()
        kernel = _make_kernel_bf16()
        tP = _make_tP(tile01=1, tc="B", is_m=False, is_a=False, is_b=True,
                      block_width=1, bpe_ds=2)
        module = LraTileAssignmentTransposedMFMA()(writer, kernel, tP)
        from rocisa.code import Module
        assert isinstance(module, Module)
        assert "lro" in tP["gpr"]

    def test_num1dwaves_gt1_wave_offset(self):
        """MIWaveGroup[1]=2 -> num1DWaves=2 > 1 triggers wave-offset branch (line 233)."""
        from Tensile.Components.LraTileAssignment import LraTileAssignmentTransposedMFMA
        writer = _make_writer()
        kernel = _make_kernel_bf16()
        # Ensure num1DWaves > 1 for tile01=1
        kernel["MIWaveGroup"] = [1, 4]  # num1DWaves=4 for tile01=1
        tP = _make_tP(tile01=1, tc="B", is_m=False, is_a=False, is_b=True,
                      block_width=1, bpe_ds=2)
        module = LraTileAssignmentTransposedMFMA()(writer, kernel, tP)
        from rocisa.code import Module
        assert isinstance(module, Module)

    def test_source_swap_false(self):
        """SourceSwap=False selects the else-branch of dividedForBlkId (line 183)."""
        from Tensile.Components.LraTileAssignment import LraTileAssignmentTransposedMFMA
        writer = _make_writer()
        kernel = _make_kernel_bf16()
        kernel["SourceSwap"] = False
        tP = _make_tP(tile01=1, tc="B", is_m=False, is_a=False, is_b=True,
                      block_width=1, bpe_ds=2)
        module = LraTileAssignmentTransposedMFMA()(writer, kernel, tP)
        from rocisa.code import Module
        assert isinstance(module, Module)


# ---------------------------------------------------------------------------
# Tests: LraTileAssignmentTransposedMFMAB8 (lines 285-409)
# ---------------------------------------------------------------------------

class TestLraTileAssignmentTransposedMFMAB8:
    """Direct-invoke tests for LraTileAssignmentTransposedMFMAB8 (I8 path).

    Lines 285-409 have two conditional branches:
      - isM=True  -> inner k-stride path (lines 359-371)
      - isM=False -> unroll-offset path  (lines 373-390)
    Wave-offset branch (lines 393-399) fires when num1DWaves > 1.
    """

    def test_is_m_false_returns_module(self):
        """isM=False (N-side) executes unroll-offset branch (lines 373-390)."""
        from Tensile.Components.LraTileAssignment import LraTileAssignmentTransposedMFMAB8
        writer = _make_writer()
        kernel = _make_kernel_i8()
        tP = _make_tP(tile01=1, tc="B", is_m=False, is_a=False, is_b=True,
                      block_width=0.5, bpe_ds=1)
        module = LraTileAssignmentTransposedMFMAB8()(writer, kernel, tP)
        from rocisa.code import Module
        assert isinstance(module, Module)
        assert "lro" in tP["gpr"]

    def test_is_m_true_returns_module(self):
        """isM=True (M-side) executes k-stride inner branch (lines 359-371)."""
        from Tensile.Components.LraTileAssignment import LraTileAssignmentTransposedMFMAB8
        writer = _make_writer()
        kernel = _make_kernel_i8()
        tP = _make_tP(tile01=0, tc="A", is_m=True, is_a=True, is_b=False,
                      block_width=0.5, bpe_ds=1)
        module = LraTileAssignmentTransposedMFMAB8()(writer, kernel, tP)
        from rocisa.code import Module
        assert isinstance(module, Module)
        assert "lro" in tP["gpr"]

    def test_wave_offset_branch(self):
        """num1DWaves > 1 fires wave-offset append (lines 393-399)."""
        from Tensile.Components.LraTileAssignment import LraTileAssignmentTransposedMFMAB8
        writer = _make_writer()
        kernel = _make_kernel_i8()
        kernel["MIWaveGroup"] = [1, 4]  # num1DWaves=4 for tile01=1
        tP = _make_tP(tile01=1, tc="B", is_m=False, is_a=False, is_b=True,
                      block_width=0.5, bpe_ds=1)
        module = LraTileAssignmentTransposedMFMAB8()(writer, kernel, tP)
        from rocisa.code import Module
        assert isinstance(module, Module)


# ---------------------------------------------------------------------------
# Tests: LraTileAssignmentTransposedMFMAF4 (lines 473-592)
# ---------------------------------------------------------------------------

class TestLraTileAssignmentTransposedMFMAF4:
    """Direct-invoke tests for LraTileAssignmentTransposedMFMAF4 (F4 path).

    Lines 506-592 contain the full __call__ body.
    """

    def test_tile01_0_returns_module(self):
        """tile01=0 (A-side) executes F4 transposed LRA."""
        from Tensile.Components.LraTileAssignment import LraTileAssignmentTransposedMFMAF4
        writer = _make_writer()
        kernel = _make_kernel_f4()
        tP = _make_tP(tile01=0, tc="A", is_m=False, is_a=True, is_b=False,
                      block_width=0.25, bpe_ds=0.5)
        module = LraTileAssignmentTransposedMFMAF4()(writer, kernel, tP)
        from rocisa.code import Module
        assert isinstance(module, Module)
        assert "lro" in tP["gpr"]

    def test_tile01_1_returns_module(self):
        """tile01=1 (B-side) executes F4 transposed LRA."""
        from Tensile.Components.LraTileAssignment import LraTileAssignmentTransposedMFMAF4
        writer = _make_writer()
        kernel = _make_kernel_f4()
        tP = _make_tP(tile01=1, tc="B", is_m=False, is_a=False, is_b=True,
                      block_width=0.25, bpe_ds=0.5)
        module = LraTileAssignmentTransposedMFMAF4()(writer, kernel, tP)
        from rocisa.code import Module
        assert isinstance(module, Module)

    def test_wave_offset_branch_f4(self):
        """num1DWaves > 1 fires wave-offset branch (lines 576-582)."""
        from Tensile.Components.LraTileAssignment import LraTileAssignmentTransposedMFMAF4
        writer = _make_writer()
        kernel = _make_kernel_f4()
        kernel["MIWaveGroup"] = [1, 4]
        tP = _make_tP(tile01=1, tc="B", is_m=False, is_a=False, is_b=True,
                      block_width=0.25, bpe_ds=0.5)
        module = LraTileAssignmentTransposedMFMAF4()(writer, kernel, tP)
        from rocisa.code import Module
        assert isinstance(module, Module)


# ---------------------------------------------------------------------------
# Tests: LraTileAssignmentTransposedMFMAF6 (lines 611-693)
# ---------------------------------------------------------------------------

class TestLraTileAssignmentTransposedMFMAF6:
    """Direct-invoke tests for LraTileAssignmentTransposedMFMAF6 (F6 path).

    Lines 610-693 contain the full __call__ body.
    """

    def test_tile01_0_returns_module(self):
        """tile01=0 (A-side) executes F6 transposed LRA."""
        from Tensile.Components.LraTileAssignment import LraTileAssignmentTransposedMFMAF6
        writer = _make_writer()
        kernel = _make_kernel_f6()
        tP = _make_tP(tile01=0, tc="A", is_m=False, is_a=True, is_b=False,
                      block_width=0.375, bpe_ds=0.75)
        module = LraTileAssignmentTransposedMFMAF6()(writer, kernel, tP)
        from rocisa.code import Module
        assert isinstance(module, Module)
        assert "lro" in tP["gpr"]

    def test_tile01_1_returns_module(self):
        """tile01=1 (B-side) executes F6 transposed LRA."""
        from Tensile.Components.LraTileAssignment import LraTileAssignmentTransposedMFMAF6
        writer = _make_writer()
        kernel = _make_kernel_f6()
        tP = _make_tP(tile01=1, tc="B", is_m=False, is_a=False, is_b=True,
                      block_width=0.375, bpe_ds=0.75)
        module = LraTileAssignmentTransposedMFMAF6()(writer, kernel, tP)
        from rocisa.code import Module
        assert isinstance(module, Module)

    def test_wave_offset_branch_f6(self):
        """num1DWaves > 1 fires wave-offset branch (lines 677-683)."""
        from Tensile.Components.LraTileAssignment import LraTileAssignmentTransposedMFMAF6
        writer = _make_writer()
        kernel = _make_kernel_f6()
        kernel["MIWaveGroup"] = [1, 4]
        tP = _make_tP(tile01=1, tc="B", is_m=False, is_a=False, is_b=True,
                      block_width=0.375, bpe_ds=0.75)
        module = LraTileAssignmentTransposedMFMAF6()(writer, kernel, tP)
        from rocisa.code import Module
        assert isinstance(module, Module)
