################################################################################
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

from rocisa.asmpass import rocIsaPass, rocIsaPassOption
from rocisa.code import Module, KernelBody
from rocisa.instruction import MacroInstruction
from rocisa.macro import MacroVMagicDiv, PseudoRandomGenerator


def _run_pass(body):
    kb = KernelBody("test")
    kb.addBody(body)
    opt = rocIsaPassOption()
    opt.insertDelayAlu = False
    opt.removeDupFunc = False
    opt.removeDupAssign = False
    opt.getCycles = False
    rocIsaPass(kb, opt)
    return str(kb)


def test_vmagic_div_algo1():
    body = Module("body")
    body.add(MacroVMagicDiv(1))
    body.add(MacroInstruction(name="V_MAGIC_DIV", args=[1, "v2", "s3", "s4", "s5"]))
    result = _run_pass(body)
    assert ".macro" not in result
    assert "V_MAGIC_DIV" not in result
    assert "v_mul_hi_u32 v2, v2, s3" in result
    assert "v_mul_lo_u32 v1, v2, s3" in result
    assert "v_lshrrev_b64 v[1:2], s4, v[1:2]" in result


def test_vmagic_div_algo2():
    body = Module("body")
    body.add(MacroVMagicDiv(2))
    body.add(MacroInstruction(name="V_MAGIC_DIV", args=[3, "v10", "s20", "s21", "s22"]))
    result = _run_pass(body)
    assert ".macro" not in result
    assert "v_mul_hi_u32 v4, v10, s20" in result
    assert "v_mul_lo_u32 v3, v10, s22" in result
    assert "v_add_u32 v3" in result
    assert "v_lshrrev_b32 v3, s21, v3" in result


def test_prnd_generator():
    body = Module("body")
    body.add(PseudoRandomGenerator())
    body.add(MacroInstruction(name="PRND_GENERATOR", args=["v5", "v6", "v7", "v8"]))
    result = _run_pass(body)
    assert ".macro" not in result
    assert "PRND_GENERATOR" not in result
    assert "v_and_b32 v7, 0xFFFF, v6" in result
    assert "v_xor_b32 v5" in result


def test_no_macros_is_noop():
    body = Module("body")
    result = _run_pass(body)
    assert "Begin Kernel" in result


def test_macro_in_submodule():
    body = Module("body")
    sub = body.add(Module("sub"))
    sub.add(MacroVMagicDiv(1))
    sub.add(MacroInstruction(name="V_MAGIC_DIV", args=[0, "v1", "s2", "s3", "s4"]))
    result = _run_pass(body)
    assert ".macro" not in result
    assert "v_mul_hi_u32 v1, v1, s2" in result
