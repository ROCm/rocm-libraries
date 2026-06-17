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

from rocisa import rocIsa
from rocisa.asmpass import rocIsaPass, rocIsaPassOption
from rocisa.code import Module, KernelBody
from rocisa.container import RegisterContainer
from rocisa.instruction import CommonInstruction, MacroInstruction
from rocisa.macro import MacroVMagicDiv, PseudoRandomGenerator


def _run_pass_kb(body):
    kb = KernelBody("test")
    kb.addBody(body)
    opt = rocIsaPassOption()
    opt.insertDelayAlu = False
    opt.removeDupFunc = False
    opt.removeDupAssign = False
    opt.getCycles = False
    rocIsaPass(kb, opt)
    return kb


def _run_pass(body):
    return str(_run_pass_kb(body))


def _all_common_instructions(item):
    """Recursively collect every CommonInstruction under a Module/KernelBody item."""
    result = []
    if isinstance(item, Module):
        for child in item.items():
            result.extend(_all_common_instructions(child))
    elif isinstance(item, CommonInstruction):
        result.append(item)
    return result


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


def test_symbolic_vgpr_src_rematerialized():
    # A macro source operand passed as a symbolic VGPR reference must be turned back
    # into a RegisterContainer during expansion (not left as an opaque string), so the
    # downstream VGPR MSB/bank pass can resolve its index for registers above 255.
    rocIsa.getInstance().setVgprIdx("GlobalReadOffsetA", 512)
    body = Module("body")
    body.add(MacroVMagicDiv(1))
    body.add(MacroInstruction(name="V_MAGIC_DIV",
                              args=[1, "v[vgprGlobalReadOffsetA+0]", "s3", "s4", "s5"]))
    insts = _all_common_instructions(_run_pass_kb(body).body)

    regs = [s for inst in insts for s in inst.srcs
            if isinstance(s, RegisterContainer) and s.regName is not None
            and "GlobalReadOffsetA" in str(s.regName)]
    assert regs, "symbolic VGPR src was not re-materialized as a RegisterContainer"
    reg = regs[0]
    assert reg.regType == "v"
    # Offsets parsed into RegName (not baked into the base name), so the symbol lookup
    # resolves the absolute index above 255.
    assert str(reg.regName) == "GlobalReadOffsetA+0"
    assert reg.regName.getTotalIdx() == 512


def test_symbolic_vgpr_dst_offset_parsing():
    # A macro dst operand whose macro-body offset (e.g. "DstIdx+1") combines with a
    # symbolic argument ("vgprGlobalReadOffsetB+0") must parse every "+N" into
    # RegName.offsets so getTotalIdx() resolves correctly (513 + 0 + 1).
    rocIsa.getInstance().setVgprIdx("GlobalReadOffsetB", 513)
    body = Module("body")
    body.add(MacroVMagicDiv(1))
    body.add(MacroInstruction(name="V_MAGIC_DIV",
                              args=["vgprGlobalReadOffsetB+0", "v2", "s3", "s4", "s5"]))
    insts = _all_common_instructions(_run_pass_kb(body).body)

    dsts = {str(inst.dst.regName): inst.dst for inst in insts
            if isinstance(inst.dst, RegisterContainer) and inst.dst.regName is not None
            and "GlobalReadOffsetB" in str(inst.dst.regName)}
    assert "GlobalReadOffsetB+0+1" in dsts, list(dsts)
    assert dsts["GlobalReadOffsetB+0+1"].regName.getTotalIdx() == 514
    assert "GlobalReadOffsetB+0+0" in dsts, list(dsts)
    assert dsts["GlobalReadOffsetB+0+0"].regName.getTotalIdx() == 513
