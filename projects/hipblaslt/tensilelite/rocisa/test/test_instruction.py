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
    print(iiv[2])

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

test_instruction_common()
test_instruction_cvt()
