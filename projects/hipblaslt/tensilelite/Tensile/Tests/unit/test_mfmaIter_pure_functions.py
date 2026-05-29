################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT
################################################################################
"""Unit tests for the pure static methods extracted from mfmaIter().

These functions have no `self` dependency and can be tested in isolation.
"""

import os
import shutil
import sys
import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, TENSILE_ROOT)

from rocisa import rocIsa
from rocisa.code import Module
from rocisa.container import vgpr, accvgpr
from rocisa.enum import InstType
from rocisa.instruction import MFMAInstruction, MXMFMAInstruction, SMFMAInstruction
from Tensile.Common.Architectures import gfxToIsa

from Tensile.KernelWriterAssembly import KernelWriterAssembly


@pytest.fixture(scope="module", autouse=True)
def _rocisa_once():
    ri = rocIsa.getInstance()
    isa = gfxToIsa("gfx950")
    asmpath = shutil.which('amdclang++') or '/usr/bin/amdclang++'
    ri.init(isa, asmpath)
    ri.setKernel(isa, 64)


# ---------------------------------------------------------------------------
# _accVgprStr
# ---------------------------------------------------------------------------

class TestAccVgprStr:
    def test_miArchVgpr_true_returns_valuC(self):
        result = KernelWriterAssembly._accVgprStr(True, 256, 0, 4)
        assert "ValuC" in str(result)

    def test_miArchVgpr_false_below_limit_returns_accvgpr(self):
        result = KernelWriterAssembly._accVgprStr(False, 256, 0, 4)
        assert "acc" in str(result)

    def test_miArchVgpr_false_at_limit_returns_vgpr(self):
        result = KernelWriterAssembly._accVgprStr(False, 256, 256, 4)
        s = str(result)
        assert "acc" not in s
        assert "v" in s

    def test_miArchVgpr_false_above_limit_returns_vgpr(self):
        result = KernelWriterAssembly._accVgprStr(False, 256, 260, 4)
        s = str(result)
        assert "acc" not in s
        assert "v" in s

    def test_sz_default_is_1(self):
        result = KernelWriterAssembly._accVgprStr(True, 256, 8)
        assert "ValuC+8" in str(result)


# ---------------------------------------------------------------------------
# _emitStandardMfma
# ---------------------------------------------------------------------------

class TestEmitStandardMfma:
    def test_emits_one_mfma(self):
        mod = KernelWriterAssembly._emitStandardMfma(
            miInInstType=InstType.INST_BF16,
            miOutInstType=InstType.INST_F32,
            variant=[16, 16, 32, 1],
            mfma_1k=False,
            acc=vgpr("ValuC+0", 4),
            src0=vgpr(0, 4),
            src1=vgpr(4, 4),
            acc2=vgpr("ValuC+0", 4),
            neg_flag=False,
            comment="test_standard")
        assert isinstance(mod, Module)
        items = mod.items()
        assert len(items) == 1
        assert isinstance(items[0], MFMAInstruction)

    def test_asm_contains_mfma_opcode(self):
        mod = KernelWriterAssembly._emitStandardMfma(
            miInInstType=InstType.INST_BF16,
            miOutInstType=InstType.INST_F32,
            variant=[16, 16, 32, 1],
            mfma_1k=False,
            acc=vgpr("ValuC+0", 4),
            src0=vgpr(0, 4),
            src1=vgpr(4, 4),
            acc2=vgpr("ValuC+0", 4),
            neg_flag=False,
            comment="test_standard")
        asm = str(mod)
        assert "v_mfma" in asm

    def test_comment_appears_in_output(self):
        mod = KernelWriterAssembly._emitStandardMfma(
            miInInstType=InstType.INST_BF16,
            miOutInstType=InstType.INST_F32,
            variant=[16, 16, 32, 1],
            mfma_1k=False,
            acc=vgpr("ValuC+0", 4),
            src0=vgpr(0, 4),
            src1=vgpr(4, 4),
            acc2=vgpr("ValuC+0", 4),
            neg_flag=False,
            comment="my_test_comment")
        asm = str(mod)
        assert "my_test_comment" in asm


# ---------------------------------------------------------------------------
# _emitSparseMfma
# ---------------------------------------------------------------------------

class TestEmitSparseMfma:
    def test_emits_one_smfma(self):
        mod = KernelWriterAssembly._emitSparseMfma(
            miInInstType=InstType.INST_BF16,
            miOutInstType=InstType.INST_F32,
            variant=[16, 16, 32, 1],
            mfma_1k=False,
            acc=vgpr("ValuC+0", 4),
            src0=vgpr(0, 4),
            src1=vgpr(4, 4),
            metadata=vgpr(8, 1),
            neg_flag=False,
            comment="test_sparse")
        assert isinstance(mod, Module)
        items = mod.items()
        assert len(items) == 1
        assert isinstance(items[0], SMFMAInstruction)


# ---------------------------------------------------------------------------
# _emitF32XEmulationMfma
# ---------------------------------------------------------------------------

class TestEmitF32XEmulationMfma:
    def test_emits_three_mfma(self):
        mod = KernelWriterAssembly._emitF32XEmulationMfma(
            miOutInstType=InstType.INST_F32,
            variant=[16, 16, 32, 1],
            mfma_1k=False,
            acc=vgpr("ValuC+0", 4),
            acc2=vgpr("ValuC+0", 4),
            src0_0=vgpr(0, 2),
            src0_1=vgpr(2, 2),
            src1_0=vgpr(4, 2),
            src1_1=vgpr(6, 2),
            neg_flag=False,
            sourceSwap=False,
            comment="test_f32x")
        assert isinstance(mod, Module)
        items = mod.items()
        assert len(items) == 3
        assert all(isinstance(i, MFMAInstruction) for i in items)

    def test_sourceSwap_changes_order(self):
        mod_no_swap = KernelWriterAssembly._emitF32XEmulationMfma(
            miOutInstType=InstType.INST_F32,
            variant=[16, 16, 32, 1],
            mfma_1k=False,
            acc=vgpr("ValuC+0", 4),
            acc2=vgpr("ValuC+0", 4),
            src0_0=vgpr(0, 2),
            src0_1=vgpr(2, 2),
            src1_0=vgpr(4, 2),
            src1_1=vgpr(6, 2),
            neg_flag=False,
            sourceSwap=False,
            comment="test")
        mod_swap = KernelWriterAssembly._emitF32XEmulationMfma(
            miOutInstType=InstType.INST_F32,
            variant=[16, 16, 32, 1],
            mfma_1k=False,
            acc=vgpr("ValuC+0", 4),
            acc2=vgpr("ValuC+0", 4),
            src0_0=vgpr(0, 2),
            src0_1=vgpr(2, 2),
            src1_0=vgpr(4, 2),
            src1_1=vgpr(6, 2),
            neg_flag=False,
            sourceSwap=True,
            comment="test")
        asm_no = str(mod_no_swap)
        asm_sw = str(mod_swap)
        assert asm_no != asm_sw

    def test_comments_include_operand_labels(self):
        mod = KernelWriterAssembly._emitF32XEmulationMfma(
            miOutInstType=InstType.INST_F32,
            variant=[16, 16, 32, 1],
            mfma_1k=False,
            acc=vgpr("ValuC+0", 4),
            acc2=vgpr("ValuC+0", 4),
            src0_0=vgpr(0, 2),
            src0_1=vgpr(2, 2),
            src1_0=vgpr(4, 2),
            src1_1=vgpr(6, 2),
            neg_flag=False,
            sourceSwap=False,
            comment="test")
        asm = str(mod)
        assert "src0_h*src1_h" in asm
        assert "src0_l*src1_h" in asm
        assert "src0_h*src1_l" in asm


# ---------------------------------------------------------------------------
# _emitMXBlockMfma
# ---------------------------------------------------------------------------

class TestEmitMXBlockMfma:
    def test_emits_one_mxmfma(self):
        mod = KernelWriterAssembly._emitMXBlockMfma(
            miInInstType=InstType.INST_F8,
            miOutInstType=InstType.INST_F32,
            miInScale0InstType=InstType.INST_E8,
            miInScale1InstType=InstType.INST_E8,
            variant=[16, 16, 128, 1],
            acc=vgpr("ValuC+0", 4),
            src0=vgpr(0, 8),
            src1=vgpr(8, 8),
            acc2=vgpr("ValuC+0", 4),
            srcMX0=vgpr(16, 1),
            srcMX1=vgpr(17, 1),
            block=32,
            comment="test_mx")
        assert isinstance(mod, Module)
        items = mod.items()
        assert len(items) == 1
        assert isinstance(items[0], MXMFMAInstruction)


# ---------------------------------------------------------------------------
# _emitComplexMfma
# ---------------------------------------------------------------------------

class TestEmitComplexMfma:
    def test_emits_four_mfma(self):
        acc_cr  = vgpr("ValuC+0", 4)
        acc_cr2 = vgpr("ValuC+0", 4)
        acc_ci  = vgpr("ValuC+4", 4)
        acc_ci2 = vgpr("ValuC+4", 4)
        mod = KernelWriterAssembly._emitComplexMfma(
            miInInstType=InstType.INST_F32,
            miOutInstType=InstType.INST_F32,
            variant=[16, 16, 4, 1],
            src0_rr=vgpr(0, 1), src1_rr=vgpr(4, 1),
            acc_cr=acc_cr, acc2_cr=acc_cr,
            src0_ri=vgpr(1, 1), src1_ri=vgpr(5, 1),
            acc_cr2=acc_cr2, acc2_cr2=acc_cr2,
            src0_ir=vgpr(1, 1), src1_ir=vgpr(4, 1),
            acc_ci=acc_ci, acc2_ci=acc_ci,
            src0_ii=vgpr(0, 1), src1_ii=vgpr(5, 1),
            acc_ci2=acc_ci2, acc2_ci2=acc_ci2,
            commentCr1="Cr += Ar*Br",
            commentCr2="Cr += -Ai*Bi",
            commentCi1="Ci += Ai*Br",
            commentCi2="Ci += Ar*Bi")
        assert isinstance(mod, Module)
        items = mod.items()
        assert len(items) == 4
        assert all(isinstance(i, MFMAInstruction) for i in items)

    def test_comments_match(self):
        acc = vgpr("ValuC+0", 4)
        mod = KernelWriterAssembly._emitComplexMfma(
            miInInstType=InstType.INST_F32,
            miOutInstType=InstType.INST_F32,
            variant=[16, 16, 4, 1],
            src0_rr=vgpr(0, 1), src1_rr=vgpr(4, 1),
            acc_cr=acc, acc2_cr=acc,
            src0_ri=vgpr(1, 1), src1_ri=vgpr(5, 1),
            acc_cr2=acc, acc2_cr2=acc,
            src0_ir=vgpr(1, 1), src1_ir=vgpr(4, 1),
            acc_ci=acc, acc2_ci=acc,
            src0_ii=vgpr(0, 1), src1_ii=vgpr(5, 1),
            acc_ci2=acc, acc2_ci2=acc,
            commentCr1="Cr += Ar*Br",
            commentCr2="Cr += -Ai*Bi",
            commentCi1="Ci += Ai*Br",
            commentCi2="Ci += Ar*Bi")
        asm = str(mod)
        assert "Cr += Ar*Br" in asm
        assert "Cr += -Ai*Bi" in asm
        assert "Ci += Ai*Br" in asm
        assert "Ci += Ar*Bi" in asm
