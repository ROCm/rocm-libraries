################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

import rocisa
from rocisa.container import sgpr, vgpr
from copy import deepcopy
import math

import pytest

def test_instruction_common():
    from rocisa.instruction import SMovB32

    inst = SMovB32(dst=sgpr(1), src=1.6)
    assert str(inst) == "s_mov_b32 s1, 1.6000000000000001\n"
    assert str(inst.dst) == "s1"
    # Due to conversion between C++ and Python
    assert math.isclose(inst.srcs[0], 1.6, abs_tol=0.000001)
    # You cannot use srcs[0] = 1.3 because it may break the C++ memory layout
    # so nanobind disabled it
    inst.srcs = [1.3]
    assert math.isclose(inst.srcs[0], 1.3, abs_tol=0.000001)
    inst.setSrc(0, 1.4)
    assert math.isclose(inst.srcs[0], 1.4, abs_tol=0.000001)

    inst2 = deepcopy(inst)
    inst.setSrc(0, 2.0)
    assert math.isclose(inst2.srcs[0], 1.4, abs_tol=0.000001)

    from rocisa.instruction import InstructionInputVector
    iiv = InstructionInputVector()
    iiv.append(1.0)
    iiv.append("hello")
    iiv.append(sgpr(1))
    assert str(iiv[2]) == "s1"

    from rocisa.code import Module
    from rocisa.container import vgpr
    from rocisa.instruction import BufferLoadB64
    module = Module("Test")
    module.add(BufferLoadB64(vgpr(1), vgpr(2), vgpr(3), 3))
    assert rocisa.countGlobalRead(module) == 1

def test_instruction_cvt():
    from rocisa.instruction import VCvtF16toF32, VCvtF32toF16, VCvtF32toU32, VCvtU32toF32, \
        VCvtI32toF32, VCvtF32toI32, VCvtFP8toF32, VCvtBF8toF32, VCvtPkFP8toF32, VCvtPkBF8toF32, \
            VCvtPkF32toFP8, VCvtPkF32toBF8, VCvtSRF32toFP8, VCvtSRF32toBF8

    # Test VCvtF16toF32
    inst = VCvtF16toF32(dst=vgpr(1), src=vgpr(2), comment="test comment")
    assert str(inst) == "v_cvt_f32_f16 v1, v2                               // test comment\n"
    assert str(inst.dst) == "v1"
    assert str(inst.srcs[0]) == "v2"
    assert inst.comment == "test comment"

    # Test VCvtF32toF16
    inst = VCvtF32toF16(dst=vgpr(1), src=vgpr(2), comment="test comment")
    assert str(inst) == "v_cvt_f16_f32 v1, v2                               // test comment\n"
    assert str(inst.dst) == "v1"
    assert str(inst.srcs[0]) == "v2"
    assert inst.comment == "test comment"

    # Test VCvtF32toU32
    inst = VCvtF32toU32(dst=vgpr(1), src=vgpr(2), comment="test comment")
    assert str(inst) == "v_cvt_u32_f32 v1, v2                               // test comment\n"
    assert str(inst.dst) == "v1"
    assert str(inst.srcs[0]) == "v2"
    assert inst.comment == "test comment"

    # Test VCvtU32toF32
    inst = VCvtU32toF32(dst=vgpr(1), src=vgpr(2), comment="test comment")
    assert str(inst) == "v_cvt_f32_u32 v1, v2                               // test comment\n"
    assert str(inst.dst) == "v1"
    assert str(inst.srcs[0]) == "v2"
    assert inst.comment == "test comment"

    # Test VCvtI32toF32
    inst = VCvtI32toF32(dst=vgpr(1), src=vgpr(2), comment="test comment")
    assert str(inst) == "v_cvt_f32_i32 v1, v2                               // test comment\n"
    assert str(inst.dst) == "v1"
    assert str(inst.srcs[0]) == "v2"
    assert inst.comment == "test comment"

    # Test VCvtF32toI32
    inst = VCvtF32toI32(dst=vgpr(1), src=vgpr(2), comment="test comment")
    assert str(inst) == "v_cvt_i32_f32 v1, v2                               // test comment\n"
    assert str(inst.dst) == "v1"
    assert str(inst.srcs[0]) == "v2"
    assert inst.comment == "test comment"

    # Test VCvtFP8toF32
    inst = VCvtFP8toF32(dst=vgpr(1), src=vgpr(2), comment="test comment")
    assert str(inst) == "v_cvt_f32_fp8 v1, v2                               // test comment\n"
    assert str(inst.dst) == "v1"
    assert str(inst.srcs[0]) == "v2"
    assert inst.comment == "test comment"

    # Test VCvtBF8toF32
    inst = VCvtBF8toF32(dst=vgpr(1), src=vgpr(2), comment="test comment")
    assert str(inst) == "v_cvt_f32_bf8 v1, v2                               // test comment\n"
    assert str(inst.dst) == "v1"
    assert str(inst.srcs[0]) == "v2"
    assert inst.comment == "test comment"

    # Test VCvtPkFP8toF32
    inst = VCvtPkFP8toF32(dst=vgpr(1), src=vgpr(2), comment="test comment")
    assert str(inst) == "v_cvt_pk_f32_fp8 v1, v2                            // test comment\n"
    assert str(inst.dst) == "v1"
    assert str(inst.srcs[0]) == "v2"
    assert inst.comment == "test comment"

    # Test VCvtPkBF8toF32
    inst = VCvtPkBF8toF32(dst=vgpr(1), src=vgpr(2), comment="test comment")
    assert str(inst) == "v_cvt_pk_f32_bf8 v1, v2                            // test comment\n"
    assert str(inst.dst) == "v1"
    assert str(inst.srcs[0]) == "v2"
    assert inst.comment == "test comment"

    # Test VCvtPkF32toFP8
    inst = VCvtPkF32toFP8(dst=vgpr(1), src0=vgpr(2), src1=vgpr(3), comment="test comment")
    assert str(inst) == "v_cvt_pk_fp8_f32 v1, v2, v3                        // test comment\n"
    assert str(inst.dst) == "v1"
    assert str(inst.srcs[0]) == "v2"
    assert str(inst.srcs[1]) == "v3"
    assert inst.comment == "test comment"

    # Test VCvtPkF32toBF8
    inst = VCvtPkF32toBF8(dst=vgpr(1), src0=vgpr(2), src1=vgpr(3), comment="test comment")
    assert str(inst) == "v_cvt_pk_bf8_f32 v1, v2, v3                        // test comment\n"
    assert str(inst.dst) == "v1"
    assert str(inst.srcs[0]) == "v2"
    assert str(inst.srcs[1]) == "v3"
    assert inst.comment == "test comment"

    # Test VCvtSRF32toFP8
    inst = VCvtSRF32toFP8(dst=vgpr(1), src0=vgpr(2), src1=vgpr(3), comment="test comment")
    assert str(inst) == "v_cvt_sr_fp8_f32 v1, v2, v3                        // test comment\n"
    assert str(inst.dst) == "v1"
    assert str(inst.srcs[0]) == "v2"
    assert str(inst.srcs[1]) == "v3"
    assert inst.comment == "test comment"

    # Test VCvtSRF32toBF8
    inst = VCvtSRF32toBF8(dst=vgpr(1), src0=vgpr(2), src1=vgpr(3), comment="test comment")
    assert str(inst) == "v_cvt_sr_bf8_f32 v1, v2, v3                        // test comment\n"
    assert str(inst.dst) == "v1"
    assert str(inst.srcs[0]) == "v2"
    assert str(inst.srcs[1]) == "v3"
    assert inst.comment == "test comment"

def test_instruction_tdm():
    from rocisa.instruction import TensorLoadToLds
    from rocisa.container import sgpr
    inst = TensorLoadToLds(
        group0=sgpr(0, 4),
        group1=sgpr(0, 8),
        group2=sgpr(0, 4),
        group3=sgpr(0, 4),
        comment=""
    )
    assert str(inst) == "tensor_load_to_lds s[0:3], s[0:7], s[0:3], s[0:3]\n"

    try:
        TensorLoadToLds(
            group0=vgpr(0, 4),
            group1=sgpr(0, 8),
            group2=sgpr(0, 4),
            group3=sgpr(0, 4),
            comment=""
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        pass

def test_instruction_tdm_2_sgprs():
    from rocisa.instruction import TensorLoadToLds
    from rocisa.container import sgpr
    inst = TensorLoadToLds(
        group0=sgpr(0, 4),
        group1=sgpr(0, 8),
        group2=None,
        group3=None,
        comment=""
    )
    assert str(inst) == "tensor_load_to_lds s[0:3], s[0:7]\n"

    try:
        TensorLoadToLds(
            group0=vgpr(0, 4),
            group1=sgpr(0, 8),
            group2=None,
            group3=None,
            comment=""
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        pass

def test_instruction_swait_xcnt():
    from rocisa.instruction import SWaitXCnt

    # Default constructor: xcnt=0, no comment -> "s_wait_xcnt 0\n"
    inst = SWaitXCnt()
    assert str(inst) == "s_wait_xcnt 0\n"
    assert inst.comment == ""

    # Explicit xcnt within range
    inst = SWaitXCnt(xcnt=5)
    assert str(inst) == "s_wait_xcnt 5\n"

    # min(xcnt, 63) clamping: xcnt=100 should clamp to 63
    inst = SWaitXCnt(xcnt=100)
    assert str(inst) == "s_wait_xcnt 63\n"

    # Boundary: xcnt=63 should also produce 63 (not clamped further)
    inst = SWaitXCnt(xcnt=63)
    assert str(inst) == "s_wait_xcnt 63\n"

    # Comment formatting: use relaxed assertions to sidestep alignment math
    inst = SWaitXCnt(xcnt=5, comment="test comment")
    assert inst.comment == "test comment"
    assert str(inst).startswith("s_wait_xcnt 5")
    assert "// test comment" in str(inst)
    assert str(inst).endswith("\n")

    # deepcopy independence: mutate the original AFTER deepcopy and confirm
    # the copy is unaffected (SWaitXCnt does not expose xcnt as a writable
    # attribute, so we mutate the inherited `comment` field).
    inst = SWaitXCnt(xcnt=7, comment="original")
    inst2 = deepcopy(inst)
    inst.comment = "mutated"
    assert inst2.comment == "original"
    assert str(inst2).startswith("s_wait_xcnt 7")
    assert "// original" in str(inst2)
    assert "// mutated" not in str(inst2)

    # Embed in a Module and verify rendered text contains the instruction
    from rocisa.code import Module
    module = Module("Test")
    module.add(SWaitXCnt(xcnt=3))
    assert "s_wait_xcnt 3" in str(module)


def test_instruction_global_wb():
    from rocisa.instruction import GlobalWb
    from rocisa.enum import CacheScope

    # Default constructor: SCOPE_DEV, no comment
    inst = GlobalWb()
    assert str(inst) == "global_wb scope:SCOPE_DEV\n"
    assert inst.scope == CacheScope.SCOPE_DEV
    assert inst.comment == ""

    # SCOPE_NONE: scope modifier omitted
    inst = GlobalWb(scope=CacheScope.SCOPE_NONE)
    assert str(inst) == "global_wb\n"
    assert inst.scope == CacheScope.SCOPE_NONE

    # Alternate scopes: enum -> string conversion
    inst = GlobalWb(scope=CacheScope.SCOPE_CU)
    assert str(inst) == "global_wb scope:SCOPE_CU\n"

    inst = GlobalWb(scope=CacheScope.SCOPE_SE)
    assert str(inst) == "global_wb scope:SCOPE_SE\n"

    inst = GlobalWb(scope=CacheScope.SCOPE_SYS)
    assert str(inst) == "global_wb scope:SCOPE_SYS\n"

    # Comment formatting (relaxed assertions to avoid alignment math)
    inst = GlobalWb(scope=CacheScope.SCOPE_DEV, comment="release fence")
    assert inst.comment == "release fence"
    assert str(inst).startswith("global_wb scope:SCOPE_DEV")
    assert "// release fence" in str(inst)
    assert str(inst).endswith("\n")

    # deepcopy independence: mutate the original `scope` (and comment) after
    # deepcopy and confirm the copy is unaffected.
    inst = GlobalWb(scope=CacheScope.SCOPE_DEV, comment="orig")
    inst2 = deepcopy(inst)
    inst.scope = CacheScope.SCOPE_SYS
    inst.comment = "mutated"
    assert inst2.scope == CacheScope.SCOPE_DEV
    assert inst2.comment == "orig"
    assert str(inst2).startswith("global_wb scope:SCOPE_DEV")
    assert "// orig" in str(inst2)
    assert "// mutated" not in str(inst2)
    assert "SCOPE_SYS" not in str(inst2)

    # Embed in a Module and verify rendered text contains the instruction
    from rocisa.code import Module
    module = Module("Test")
    module.add(GlobalWb(scope=CacheScope.SCOPE_DEV))
    assert "global_wb scope:SCOPE_DEV" in str(module)


def test_instruction_global_inv():
    from rocisa.instruction import GlobalInv
    from rocisa.enum import CacheScope

    # Default constructor: SCOPE_DEV, no comment
    inst = GlobalInv()
    assert str(inst) == "global_inv scope:SCOPE_DEV\n"
    assert inst.scope == CacheScope.SCOPE_DEV
    assert inst.comment == ""

    # SCOPE_NONE: scope modifier omitted
    inst = GlobalInv(scope=CacheScope.SCOPE_NONE)
    assert str(inst) == "global_inv\n"
    assert inst.scope == CacheScope.SCOPE_NONE

    # Alternate scope sanity check
    inst = GlobalInv(scope=CacheScope.SCOPE_CU)
    assert str(inst) == "global_inv scope:SCOPE_CU\n"

    # Comment formatting (relaxed assertions to avoid alignment math)
    inst = GlobalInv(scope=CacheScope.SCOPE_DEV, comment="acquire fence")
    assert inst.comment == "acquire fence"
    assert str(inst).startswith("global_inv scope:SCOPE_DEV")
    assert "// acquire fence" in str(inst)
    assert str(inst).endswith("\n")

    # deepcopy independence: mutate the original `scope` (and comment) after
    # deepcopy and confirm the copy is unaffected.
    inst = GlobalInv(scope=CacheScope.SCOPE_DEV, comment="orig")
    inst2 = deepcopy(inst)
    inst.scope = CacheScope.SCOPE_NONE
    inst.comment = "mutated"
    assert inst2.scope == CacheScope.SCOPE_DEV
    assert inst2.comment == "orig"
    assert str(inst2).startswith("global_inv scope:SCOPE_DEV")
    assert "// orig" in str(inst2)
    assert "// mutated" not in str(inst2)


def test_instruction_scalar_float():
    # gfx12+ scalar-float / scalar-u64 instruction wrappers. Each src is an
    # InstructionInput variant (register | int | double | str); the cases below
    # exercise those arms and assert the emitted mnemonic/operands.
    from rocisa.instruction import SMulF32, SAddF32, SCvtF32U32, SCvtU32F32, \
        VSRcpF32, SMulU64

    # --- SMulF32: s_mul_f32 dst, src0, src1 ---
    inst = SMulF32(sgpr(0), sgpr(1), sgpr(2))
    assert str(inst) == "s_mul_f32 s0, s1, s2\n"
    assert str(inst.dst) == "s0"
    assert str(inst.srcs[0]) == "s1"
    assert str(inst.srcs[1]) == "s2"
    assert str(SMulF32(sgpr(0), sgpr(1), 2)) == "s_mul_f32 s0, s1, 2\n"
    assert str(SMulF32(sgpr(0), "s1", sgpr(2))) == "s_mul_f32 s0, s1, s2\n"

    # --- SAddF32: s_add_f32 dst, src0, src1 ---
    inst = SAddF32(sgpr(0), sgpr(1), sgpr(2))
    assert str(inst) == "s_add_f32 s0, s1, s2\n"
    assert str(inst.dst) == "s0"
    assert str(inst.srcs[0]) == "s1"
    assert str(inst.srcs[1]) == "s2"
    assert str(SAddF32(sgpr(0), sgpr(1), 3)) == "s_add_f32 s0, s1, 3\n"
    # integral doubles get a ".0" suffix
    assert str(SAddF32(sgpr(0), sgpr(1), 2.0)) == "s_add_f32 s0, s1, 2.0\n"

    # --- SCvtF32U32: s_cvt_f32_u32 dst, src ---
    inst = SCvtF32U32(sgpr(0), sgpr(1))
    assert str(inst) == "s_cvt_f32_u32 s0, s1\n"
    assert str(inst.dst) == "s0"
    assert str(inst.srcs[0]) == "s1"
    assert str(SCvtF32U32(sgpr(0), "s2")) == "s_cvt_f32_u32 s0, s2\n"

    # --- SCvtU32F32: s_cvt_u32_f32 dst, src ---
    inst = SCvtU32F32(sgpr(0), sgpr(1))
    assert str(inst) == "s_cvt_u32_f32 s0, s1\n"
    assert str(inst.dst) == "s0"
    assert str(inst.srcs[0]) == "s1"
    assert str(SCvtU32F32(sgpr(0), 1.5)) == "s_cvt_u32_f32 s0, 1.5\n"

    # --- VSRcpF32: v_s_rcp_f32 dst, src (vgpr dst) ---
    inst = VSRcpF32(vgpr(0), sgpr(1))
    assert str(inst) == "v_s_rcp_f32 v0, s1\n"
    assert str(inst.dst) == "v0"
    assert str(inst.srcs[0]) == "s1"
    # vgpr src must stay clean (no s_set_vgpr_msb) under the default ISA
    assert str(VSRcpF32(vgpr(0), vgpr(1))) == "v_s_rcp_f32 v0, v1\n"
    assert str(VSRcpF32(vgpr(0), 0.5)) == "v_s_rcp_f32 v0, 0.5\n"

    # --- SMulU64: s_mul_u64 dst, src0, src1 (64-bit register pairs) ---
    inst = SMulU64(sgpr(0, 2), sgpr(2, 2), sgpr(4, 2))
    assert str(inst) == "s_mul_u64 s[0:1], s[2:3], s[4:5]\n"
    assert str(inst.dst) == "s[0:1]"
    assert str(inst.srcs[0]) == "s[2:3]"
    assert str(inst.srcs[1]) == "s[4:5]"
    assert str(SMulU64(sgpr(0, 2), sgpr(2, 2), 4)) == "s_mul_u64 s[0:1], s[2:3], 4\n"

    # --- Comment formatting (relaxed to sidestep the 50-col alignment pad) ---
    inst = SMulF32(sgpr(0), sgpr(1), sgpr(2), comment="scale")
    assert str(inst).startswith("s_mul_f32 s0, s1, s2")
    assert "// scale" in str(inst)
    assert str(inst).endswith("\n")
    assert inst.comment == "scale"

    # --- deepcopy independence + comment accessor, one per class ---
    for a in [
        SMulF32(sgpr(0), sgpr(1), sgpr(2), comment="orig"),
        SAddF32(sgpr(0), sgpr(1), sgpr(2), comment="orig"),
        SCvtF32U32(sgpr(0), sgpr(1), comment="orig"),
        SCvtU32F32(sgpr(0), sgpr(1), comment="orig"),
        VSRcpF32(vgpr(0), sgpr(1), comment="orig"),
        SMulU64(sgpr(0, 2), sgpr(2, 2), sgpr(4, 2), comment="orig"),
    ]:
        assert a.comment == "orig"
        b = deepcopy(a)
        a.comment = "mutated"
        assert b.comment == "orig"
        assert "// orig" in str(b)
        assert "// mutated" not in str(b)


@pytest.mark.xfail(
    reason="ROCM-3994 (W-KNOWN-BUG two-PR flow): the EMaxF16 true16 helper does not "
           "exist on develop, so this reproducer fails at import. The true16 fix PR "
           "(users/ericwan/true16-patch) must delete this marker; strict=True flips "
           "the resulting XPASS to a failure if it is left behind. Time-box: next "
           "hipBLASLt release.",
    strict=True,
    raises=(ImportError, AttributeError),
)
def test_instruction_vmax_f16_true16():
    # Regression for the true16 (real-true16) activation-clamp defect (ROCM-3994).
    #
    # On a NoSDWA target (isaVersion 11/12) the EMaxF16 helper must emit the
    # v_max_f16 activation clamp in true16 form, binding a half-word selector
    # (.l) to every register operand. The abs() source keeps its suffix *inside*
    # the closing paren: abs(v[..].l) is valid, abs(v[..]).l is not.
    #
    # RED on develop: EMaxF16 (and the VMaxF16 true16 kwarg) do not exist there,
    # so the import / construction below fails outright. GREEN on the fix branch:
    # the emitted text carries the .l selectors.
    #
    # Assertions are substring/regex based to stay agnostic to the ~50-column
    # comment padding, matching test_instruction_swait_xcnt / _global_wb.
    import os
    import re
    import shutil

    import rocisa
    from rocisa.container import vgpr
    from rocisa.instruction import EMaxF16

    # Initialize a NoSDWA gfx11 ISA so EMaxF16 selects the true16 path. Resolve
    # the assembler via ROCM_PATH first (matching test_mubuf.py::_isa_context) so
    # this works when ROCm's bin dir is not on PATH (common in CI / Windows).
    isa = (11, 0, 0)
    rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
    search_path = os.pathsep.join([
        os.path.join(rocm_path, "bin"),
        os.path.join(rocm_path, "lib", "llvm", "bin"),
    ])
    assembler = shutil.which("amdclang++", path=search_path) or "amdclang++"
    ri = rocisa.rocIsa.getInstance()
    ri.init(isa, assembler, False)
    ri.setKernel(isa, 32)
    assert ri.getArchCaps()["NoSDWA"], "expected NoSDWA cap for gfx11"

    # How this test works: rocisa instruction objects render their assembly text
    # through __str__, so we construct the instruction under test, stringify it,
    # and assert on the emitted form. vgpr(name, ...) builds a VGPR operand
    # container (symbolic names like "Output"/"Value+0" mirror what the AMax
    # generator passes; isAbs=True wraps it in abs()). EMaxF16 is the true16-aware
    # max helper under test. Nothing is assembled or run on a device.
    # TODO(#9720): decouple this from the toolchain/env init above; a
    # string-rendering test should not need a resolved assembler path.

    # abs() source (the activation-clamp shape from AMaxGenerator.max_per_data).
    inst = EMaxF16(vgpr("Output"), vgpr("Output"), vgpr("Value+0", isAbs=True))
    s = str(inst)
    assert re.search(
        r"v_max_f16\s+v\[vgprOutput\]\.l,\s+v\[vgprOutput\]\.l,\s+abs\(v\[vgprValue\+0\]\.l\)",
        s,
    ), f"activation clamp not in true16 form: {s!r}"
    # The suffix must be inside the abs() paren, never outside it.
    assert "abs(v[vgprValue+0].l)" in s
    assert "abs(v[vgprValue+0]).l" not in s
    # And it must not be the bare fake16 form.
    assert "abs(v[vgprValue+0])," not in s and not s.rstrip().endswith("abs(v[vgprValue+0])")

    # Plain register sources (the merge_sum shape): every operand gets .l.
    inst2 = EMaxF16(vgpr("Output"), vgpr("Output"), vgpr("OutputB"))
    s2 = str(inst2)
    assert re.search(
        r"v_max_f16\s+v\[vgprOutput\]\.l,\s+v\[vgprOutput\]\.l,\s+v\[vgprOutputB\]\.l",
        s2,
    ), f"merge clamp not in true16 form: {s2!r}"


if __name__ == "__main__":
    test_instruction_common()
    test_instruction_cvt()
    test_instruction_tdm()
    test_instruction_tdm_2_sgprs()
    test_instruction_swait_xcnt()
    test_instruction_global_wb()
    test_instruction_global_inv()
    test_instruction_scalar_float()
