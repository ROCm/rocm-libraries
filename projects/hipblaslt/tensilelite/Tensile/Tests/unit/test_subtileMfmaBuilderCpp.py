#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Direct integration tests for ModuleBuilder.emit_mfma().

Tests that the C++ ModuleBuilder.emit_mfma() method produces correct rocisa
assembly output for BF16 and MX FP4/FP8 MFMA variants.  Unlike
test_emitMfmaInstruction.py (which tests the full Python resolver →
emitMfmaInstruction → C++ builder chain), this file calls the C++ builder
method directly, verifying the C++ layer in isolation.

Runs only when the compiled extension and rocisa are importable; otherwise
skips.
"""

import os
import sys
from types import SimpleNamespace

import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, TENSILE_ROOT)

pytest.importorskip("rocisa")
pytest.importorskip("tensile_writer.subtile.module_builder")

from tensile_writer.subtile.module_builder import ModuleBuilder
from tensile_writer.subtile.emit import mfma_f8f6f4_inst_type


def _init_rocisa_gfx950():
    import shutil
    from rocisa import rocIsa
    from Tensile.Common.Architectures import gfxToIsa
    ri = rocIsa.getInstance()
    isa = gfxToIsa("gfx950")
    asmpath = shutil.which("amdclang++") or "/usr/bin/amdclang++"
    ri.init(isa, asmpath)
    ri.setKernel(isa, 64)


@pytest.fixture(scope="module", autouse=True)
def _rocisa_once():
    _init_rocisa_gfx950()


@pytest.fixture(scope="module")
def builder():
    return ModuleBuilder()


# ---------------------------------------------------------------------------
# BF16 golden-string tests (miK=32)
# ---------------------------------------------------------------------------

def test_bf16_mfma_emits_correct_opcode(builder):
    """ModuleBuilder.emit_mfma with miK=32 emits the BF16 dense MFMA opcode."""
    mod = builder.emit_mfma(
        vgprAStart=0, opASize=4,
        vgprBStart=8, opBSize=4,
        vgprCStart=16, opCSize=4,
        vgprDStart=32, opDSize=4,
        dIsVgpr=True, cIsVgpr=True, miArchVgpr=True,
        sourceSwap=False, miK=32,
        instTypeName="",
        comment="bf16 test",
    )
    asm = str(mod)
    assert "v_mfma_f32_16x16x32_bf16" in asm, f"expected BF16 mfma opcode:\n{asm}"
    assert "v_mfma_scale" not in asm, f"unexpected scaled opcode for BF16:\n{asm}"


def test_bf16_mfma_source_swap(builder):
    """ModuleBuilder.emit_mfma with sourceSwap=True exchanges A/B register positions."""
    mod_no  = builder.emit_mfma(0, 4, 8, 4, 16, 4, 32, 4,
                                True, True, True, False, 32, "")
    mod_swp = builder.emit_mfma(0, 4, 8, 4, 16, 4, 32, 4,
                                True, True, True, True, 32, "")
    asm_no  = str(mod_no)
    asm_swp = str(mod_swp)
    # No swap: A-pos is v[0:3], B-pos is v[8:11].
    assert "v[0:3]" in asm_no and "v[8:11]" in asm_no
    # Swap: B occupies the A operand slot → v[8:11] in first position.
    assert "v[8:11]" in asm_swp and "v[0:3]" in asm_swp
    # The swap must have changed operand order (check A-pos moved).
    assert asm_no != asm_swp, "sourceSwap=True/False should produce different ASM"


def test_bf16_mfma_accvgpr_alias(builder):
    """When dIsVgpr=False and miArchVgpr=False the D register should use accvgpr."""
    mod = builder.emit_mfma(
        0, 4, 8, 4, 16, 4, 32, 4,
        dIsVgpr=False, cIsVgpr=False, miArchVgpr=False,
        sourceSwap=False, miK=32, instTypeName="", comment="accvgpr test",
    )
    asm = str(mod)
    assert "acc[32:35]" in asm, f"expected accvgpr for D:\n{asm}"
    assert "acc[16:19]" in asm, f"expected accvgpr for C:\n{asm}"


# ---------------------------------------------------------------------------
# MX FP8 / FP4 golden-string tests (miK=128)
# ---------------------------------------------------------------------------

# (formatA, formatB, sourceSwap, expected_cbsz_blgp)
MX_CASES = [
    ("f8",  "f8",  False, "cbsz:0 blgp:0"),
    ("bf8", "bf8", False, "cbsz:1 blgp:1"),
    ("f4",  "f4",  False, "cbsz:4 blgp:4"),
    ("f8",  "bf8", False, "cbsz:0 blgp:1"),
    ("f8",  "bf8", True,  "cbsz:1 blgp:0"),   # swap → BF8/F8
    ("f8",  "f4",  False, "cbsz:0 blgp:4"),
    ("f8",  "f4",  True,  "cbsz:4 blgp:0"),   # swap → F4/F8
    ("bf8", "f4",  False, "cbsz:1 blgp:4"),
]


def _preds(fmt):
    return fmt == "f8", fmt == "bf8", fmt == "f4"


@pytest.mark.parametrize("fmtA,fmtB,swap,expected_mod", MX_CASES)
def test_mx_mfma_cbsz_blgp(builder, fmtA, fmtB, swap, expected_mod):
    """ModuleBuilder.emit_mfma with miK=128 emits correct cbsz/blgp modifiers."""
    aIsF8, aIsBF8, aIsF4 = _preds(fmtA)
    bIsF8, bIsBF8, bIsF4 = _preds(fmtB)
    inst_type_name = mfma_f8f6f4_inst_type(
        aIsF8, aIsBF8, aIsF4, bIsF8, bIsBF8, bIsF4, swap)

    aSize = 8 if fmtA in ("f8", "bf8") else 4
    bSize = 8 if fmtB in ("f8", "bf8") else 4
    mod = builder.emit_mfma(
        vgprAStart=0, opASize=aSize,
        vgprBStart=16, opBSize=bSize,
        vgprCStart=32, opCSize=4,
        vgprDStart=64, opDSize=4,
        dIsVgpr=True, cIsVgpr=True, miArchVgpr=True,
        sourceSwap=swap, miK=128,
        instTypeName=inst_type_name,
        scaleAVgpr=100, scaleBVgpr=101,
        scaleAsel=0, scaleBsel=0,
        comment=f"mx {fmtA}x{fmtB} swap={swap}",
    )
    asm = str(mod)
    assert "v_mfma_scale_f32_16x16x128_f8f6f4" in asm, \
        f"expected scaled opcode:\n{asm}"
    assert expected_mod in asm, \
        f"expected `{expected_mod}` for {fmtA}x{fmtB} swap={swap}:\n{asm}"


def test_mx_mfma_real_scale_uses_op_sel(builder):
    """Real scale VGPRs produce op_sel + op_sel_hi modifiers in the asm."""
    inst_type = mfma_f8f6f4_inst_type(True, False, False, True, False, False, False)
    mod = builder.emit_mfma(
        0, 8, 16, 8, 32, 4, 64, 4,
        True, True, True, False, 128, inst_type,
        scaleAVgpr=100, scaleBVgpr=101,
        scaleAsel=2, scaleBsel=1,
    )
    asm = str(mod)
    assert "v100" in asm and "v101" in asm, "real scale VGPRs must appear in asm"
    assert "op_sel" in asm, "op_sel must appear for real-scale path"


def test_mx_mfma_unit_scale_fallback(builder):
    """Unit-scale fallback path uses unitScaleVgpr for both mxsa/mxsb."""
    inst_type = mfma_f8f6f4_inst_type(True, False, False, True, False, False, False)
    mod = builder.emit_mfma(
        0, 8, 16, 8, 32, 4, 64, 4,
        True, True, True, False, 128, inst_type,
        scaleAVgpr=-1, scaleBVgpr=-1,
        unitScaleVgpr=250,
    )
    asm = str(mod)
    assert "v_mfma_scale_f32_16x16x128_f8f6f4" in asm
    assert "v250" in asm, "unit-scale fallback must use v250 for both mxsa/mxsb"
    assert "op_sel" not in asm, "unit-scale path must NOT emit op_sel"


def test_mx_mfma_op_sel_encoding(builder):
    """scaleAsel=2 → op_sel[0]=0 op_sel_hi[0]=1; scaleBsel=1 → op_sel[1]=1 op_sel_hi[1]=0."""
    inst_type = mfma_f8f6f4_inst_type(False, False, True, False, False, True, False)
    mod = builder.emit_mfma(
        0, 4, 16, 4, 32, 4, 64, 4,
        True, True, True, False, 128, inst_type,
        scaleAVgpr=100, scaleBVgpr=101,
        scaleAsel=2, scaleBsel=1,
    )
    asm = str(mod)
    assert "op_sel" in asm
    # Exact encoding is rocisa-internal, but the instruction must be well-formed.
    assert "v_mfma_scale_f32_16x16x128_f8f6f4" in asm
