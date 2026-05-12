################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

import os
import shutil
import pytest

import rocisa
from rocisa import rocIsa
from rocisa.asmpass import rocIsaPassOption, rocIsaPass
from rocisa.enum import InstType
from rocisa.code import KernelBody, Module, Label
from rocisa.container import DSModifiers, vgpr, sgpr, accvgpr, ContinuousRegister, VCC
from rocisa.instruction import VLShiftLeftB32, DSLoadB128, DSLoadB64, VMovB32, SBarrier, MFMAInstruction, SNop, SWaitCnt, SBarrier, SMovB32, VMulLOU32, VAddLShiftLeftU32, VAddCOU32
from rocisa.functions import vectorStaticRemainder, vectorStaticMultiply, vectorStaticDivide, vectorStaticMultiplyAdd

def gfx(version=(9,5,0), wavefront_size = 64):
    def decorator(func):
        def wrapper():
            ti = rocIsa.getInstance()
            rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
            search_path = os.pathsep.join([
                os.path.join(rocm_path, "bin"),
                os.path.join(rocm_path, "lib", "llvm", "bin"),
            ])
            assembler = shutil.which("amdclang++", path=search_path) or "amdclang++"
            ti.init(version, assembler)
            ti.setKernel(version, wavefront_size)
            return func()
        return wrapper
    return decorator

@gfx((9,5,0))
def empty_case():
    loop_insts = []
    lra_insts = []
    return loop_insts, lra_insts, 7*4

@gfx((9,5,0))
def simple_lr_case():
    loop_insts = [
        DSLoadB128(dst=vgpr('ValuA_X0_I0+0', 4), src=vgpr('LocalReadAddrA', 1), ds=DSModifiers(na=1, offset=0), comment='L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0'),
        DSLoadB128(dst=vgpr('ValuB_X0_I0+0', 4), src=vgpr('LocalReadAddrB', 1), ds=DSModifiers(na=1, offset=0), comment='L -> Reg lro=0 swapByteOffset=0 ti=256 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0'),
        DSLoadB128(dst=vgpr('ValuA_X0_I0+4', 4), src=vgpr('LocalReadAddrA', 1), ds=DSModifiers(na=1, offset=128), comment='L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0'),
        DSLoadB128(dst=vgpr('ValuA_X0_I0+8', 4), src=vgpr('LocalReadAddrA', 1), ds=DSModifiers(na=1, offset=256), comment='L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0'),
        DSLoadB128(dst=vgpr('ValuA_X0_I0+12', 4), src=vgpr('LocalReadAddrA', 1), ds=DSModifiers(na=1, offset=384), comment='L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0'),
        DSLoadB128(dst=vgpr('ValuA_X0_I0+16', 4), src=vgpr('LocalReadAddrA', 1), ds=DSModifiers(na=1, offset=512), comment='L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=4 rIdx=0 oIdx=0 buffer=0 iui=0'),
        DSLoadB128(dst=vgpr('ValuA_X0_I0+20', 4), src=vgpr('LocalReadAddrA', 1), ds=DSModifiers(na=1, offset=640), comment='L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=5 rIdx=0 oIdx=0 buffer=0 iui=0'),
        DSLoadB128(dst=vgpr('ValuA_X0_I0+24', 4), src=vgpr('LocalReadAddrA', 1), ds=DSModifiers(na=1, offset=768), comment='L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=6 rIdx=0 oIdx=0 buffer=0 iui=0'),
        DSLoadB128(dst=vgpr('ValuA_X0_I0+28', 4), src=vgpr('LocalReadAddrA', 1), ds=DSModifiers(na=1, offset=896), comment='L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=7 rIdx=0 oIdx=0 buffer=0 iui=0'),
        DSLoadB128(dst=vgpr('ValuA_X0_I0+32', 4), src=vgpr('LocalReadAddrA', 1), ds=DSModifiers(na=1, offset=16896), comment='L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0'),
        DSLoadB128(dst=vgpr('ValuA_X0_I0+36', 4), src=vgpr('LocalReadAddrA', 1), ds=DSModifiers(na=1, offset=17024), comment='L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=1 eIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0'),
        DSLoadB128(dst=vgpr('ValuA_X0_I0+40', 4), src=vgpr('LocalReadAddrA', 1), ds=DSModifiers(na=1, offset=17152), comment='L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=1 eIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0'),
        DSLoadB128(dst=vgpr('ValuA_X0_I0+44', 4), src=vgpr('LocalReadAddrA', 1), ds=DSModifiers(na=1, offset=17280), comment='L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=1 eIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0'),
        DSLoadB128(dst=vgpr('ValuA_X0_I0+48', 4), src=vgpr('LocalReadAddrA', 1), ds=DSModifiers(na=1, offset=17408), comment='L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=1 eIdx=4 rIdx=0 oIdx=0 buffer=0 iui=0'),
        DSLoadB128(dst=vgpr('ValuA_X0_I0+52', 4), src=vgpr('LocalReadAddrA', 1), ds=DSModifiers(na=1, offset=17536), comment='L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=1 eIdx=5 rIdx=0 oIdx=0 buffer=0 iui=0'),
        DSLoadB128(dst=vgpr('ValuA_X0_I0+56', 4), src=vgpr('LocalReadAddrA', 1), ds=DSModifiers(na=1, offset=17664), comment='L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=1 eIdx=6 rIdx=0 oIdx=0 buffer=0 iui=0'),
        DSLoadB128(dst=vgpr('ValuA_X0_I0+60', 4), src=vgpr('LocalReadAddrA', 1), ds=DSModifiers(na=1, offset=17792), comment='L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=1 eIdx=7 rIdx=0 oIdx=0 buffer=0 iui=0'),
        DSLoadB128(dst=vgpr('ValuB_X0_I0+4', 4), src=vgpr('LocalReadAddrB', 1), ds=DSModifiers(na=1, offset=128), comment='L -> Reg lro=0 swapByteOffset=0 ti=256 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0'),
        SWaitCnt(dscnt=16, comment='Wait for dependent lr'),
    ]
    lra_insts = [
        vectorStaticRemainder(10, 7, 'Serial', 64, ContinuousRegister(8, 2), ContinuousRegister(48, 1), '0. thread id in wave: wtid = tid % wavelength(64)'),
        vectorStaticRemainder(10, 6, 7, 16, ContinuousRegister(8, 2), ContinuousRegister(48, 1), '1. N offset: nIdx = wtid % MI_N(16)'),
        vectorStaticMultiply(vgpr(6, 1), vgpr(6, 1), 64, ContinuousRegister(48, 1), '1. N offset: nOffset = nIdx * nStride(64)'),
        vectorStaticMultiply(vgpr(6, 1), vgpr(6, 1), 8, ContinuousRegister(48, 1), '4. apply VectorWidth: bnOffset = bnOffset * vw(8)'),
        vectorStaticDivide(7, 7, 16, ContinuousRegister(8, 2), '5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))'),
        vectorStaticMultiplyAdd(vgpr(6, 1), vgpr(7, 1), 8, vgpr(6, 1), ContinuousRegister(48, 1), '5. K offset: lrKOffset = kIdx * mStride(8); 6. offset in wave: lrOffset = bnOffset + lrKOffset'),
        vectorStaticRemainder(9, 8, 'Serial', 64, ContinuousRegister(10, 2), ContinuousRegister(48, 1), '0. thread id in wave: wtid = tid % wavelength(64)'),
        vectorStaticRemainder(9, 7, 8, 16, ContinuousRegister(10, 2), ContinuousRegister(48, 1), '1. N offset: nIdx = wtid % MI_N(16)'),
        vectorStaticMultiply(vgpr(7, 1), vgpr(7, 1), 64, ContinuousRegister(48, 1), '1. N offset: nOffset = nIdx * nStride(64)'),
        vectorStaticMultiply(vgpr(7, 1), vgpr(7, 1), 4, ContinuousRegister(48, 1), '4. apply VectorWidth: bnOffset = bnOffset * vw(4)'),
        vectorStaticDivide(8, 8, 16, ContinuousRegister(10, 2), '5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))'),
        vectorStaticMultiplyAdd(vgpr(7, 1), vgpr(8, 1), 8, vgpr(7, 1), ContinuousRegister(48, 1), '5. K offset: lrKOffset = kIdx * mStride(8); 6. offset in wave: lrOffset = bnOffset + lrKOffset'),
        vectorStaticDivide(9, 'Serial', 64, ContinuousRegister(10, 2), '7. wave offset in N dimen: wtid = tid / dividedForWaveId(64)'),
        vectorStaticRemainder(9, 9, 9, 4, ContinuousRegister(10, 2), ContinuousRegister(48, 1), '7. wave offset in M dimen: wtid0 = wtid / num1DWaves(4)'),
        vectorStaticMultiplyAdd(vgpr(7, 1), vgpr(9, 1), 4096, vgpr(7, 1), ContinuousRegister(48, 1), '7. wave offset in M dimen: wOffset = wtid0 * W0Stride(409...7. final local read offset: flrOffset = lrOffset + WOffset'),
        vectorStaticDivide(8, 'Serial', 64, ContinuousRegister(10, 2)),
        vectorStaticDivide(8, 8, 4, ContinuousRegister(10, 2), comment='LSU offset: Get LSU wave_id'),
        SMovB32(dst=sgpr(48, 1), src=64, comment='LSU offset: stride = lsuStride(64) when umlds==True'),
        VMulLOU32(dst=vgpr(8, 1), src0=sgpr(48, 1), src1=vgpr(8, 1), comment='LSU offset: lsuoffset = wave_id*lsuStride*(MT0+PAD)'),
        VAddLShiftLeftU32(dst=vgpr('LocalReadAddrA', 1), src0=vgpr(8, 1), src1=vgpr(6, 1), shiftHex='0x1', comment='Final Offset: offset = (lro0+lsuoffset)*bpeDS'),
        vectorStaticDivide(9, 'LocalReadAddrA', 1024, ContinuousRegister(10, 2), 'Final Offset: padding 32 per block 1024'),
        vectorStaticMultiplyAdd(vgpr('LocalReadAddrA', 1), vgpr(9, 1), 32, vgpr('LocalReadAddrA', 1), ContinuousRegister(48, 1), 'Final Offset: padding 32 per block 1024'),
        vectorStaticDivide(6, 'Serial', 64, ContinuousRegister(10, 2)),
        vectorStaticDivide(6, 6, 4, ContinuousRegister(10, 2), comment='LSU offset: Get LSU wave_id'),
        SMovB32(dst=sgpr(48, 1), src=64, comment='LSU offset: stride = lsuStride(64) when umlds==True'),
        VMulLOU32(dst=vgpr(6, 1), src0=sgpr(48, 1), src1=vgpr(6, 1), comment='LSU offset: lsuoffset = wave_id*lsuStride*(MT1+PAD)'),
        VAddLShiftLeftU32(dst=vgpr('LocalReadAddrB', 1), src0=vgpr(6, 1), src1=vgpr(7, 1), shiftHex='0x1', comment='Final Offset: offset = (lro1+lsuoffset)*bpeDS'),
        vectorStaticDivide(8, 'LocalReadAddrB', 512, ContinuousRegister(10, 2), 'Final Offset: padding 32 per block 512'),
        vectorStaticMultiplyAdd(vgpr('LocalReadAddrB', 1), vgpr(8, 1), 32, vgpr('LocalReadAddrB', 1), ContinuousRegister(48, 1), 'Final Offset: padding 32 per block 512'),
        VAddCOU32(dst=vgpr('LocalReadAddrB+0', 1), dst1=VCC(), src0='0x8400', src1=vgpr('LocalReadAddrB+0', 1), comment=' += LdsOffsetB (lower)'),
    ]
    return loop_insts, lra_insts, 272

@gfx((9,5,0))
def simple_mfma_case():
    loop_insts = [
        MFMAInstruction(instType=InstType.INST_BF16, accType=InstType.INST_F32, variant=[16, 16, 32, 1], mfma1k=False, acc=accvgpr(64, 4), a=vgpr('ValuB_X0_I0+4+0+0', 4), b=vgpr('ValuA_X0_I0+0+0+0', 4), acc2=accvgpr(64, 4), neg=False, comment='left value = acc[64+0:67+0]'),
        MFMAInstruction(instType=InstType.INST_BF16, accType=InstType.INST_F32, variant=[16, 16, 32, 1], mfma1k=False, acc=accvgpr(68, 4), a=vgpr('ValuB_X0_I0+4+0+0', 4), b=vgpr('ValuA_X0_I0+4+0+0', 4), acc2=accvgpr(68, 4), neg=False, comment='left value = acc[68+0:71+0]'),
        MFMAInstruction(instType=InstType.INST_BF16, accType=InstType.INST_F32, variant=[16, 16, 32, 1], mfma1k=False, acc=accvgpr(72, 4), a=vgpr('ValuB_X0_I0+4+0+0', 4), b=vgpr('ValuA_X0_I0+8+0+0', 4), acc2=accvgpr(72, 4), neg=False, comment='left value = acc[72+0:75+0]'),
        MFMAInstruction(instType=InstType.INST_BF16, accType=InstType.INST_F32, variant=[16, 16, 32, 1], mfma1k=False, acc=accvgpr(76, 4), a=vgpr('ValuB_X0_I0+4+0+0', 4), b=vgpr('ValuA_X0_I0+12+0+0', 4), acc2=accvgpr(76, 4), neg=False, comment='left value = acc[76+0:79+0]'),
        MFMAInstruction(instType=InstType.INST_BF16, accType=InstType.INST_F32, variant=[16, 16, 32, 1], mfma1k=False, acc=accvgpr(80, 4), a=vgpr('ValuB_X0_I0+4+0+0', 4), b=vgpr('ValuA_X0_I0+16+0+0', 4), acc2=accvgpr(80, 4), neg=False, comment='left value = acc[80+0:83+0]'),
        MFMAInstruction(instType=InstType.INST_BF16, accType=InstType.INST_F32, variant=[16, 16, 32, 1], mfma1k=False, acc=accvgpr(84, 4), a=vgpr('ValuB_X0_I0+4+0+0', 4), b=vgpr('ValuA_X0_I0+20+0+0', 4), acc2=accvgpr(84, 4), neg=False, comment='left value = acc[84+0:87+0]'),
        MFMAInstruction(instType=InstType.INST_BF16, accType=InstType.INST_F32, variant=[16, 16, 32, 1], mfma1k=False, acc=accvgpr(88, 4), a=vgpr('ValuB_X0_I0+4+0+0', 4), b=vgpr('ValuA_X0_I0+24+0+0', 4), acc2=accvgpr(88, 4), neg=False, comment='left value = acc[88+0:91+0]'),
        MFMAInstruction(instType=InstType.INST_BF16, accType=InstType.INST_F32, variant=[16, 16, 32, 1], mfma1k=False, acc=accvgpr(92, 4), a=vgpr('ValuB_X0_I0+4+0+0', 4), b=vgpr('ValuA_X0_I0+28+0+0', 4), acc2=accvgpr(92, 4), neg=False, comment='left value = acc[92+0:95+0]'),
        MFMAInstruction(instType=InstType.INST_BF16, accType=InstType.INST_F32, variant=[16, 16, 32, 1], mfma1k=False, acc=accvgpr(96, 4), a=vgpr('ValuB_X0_I0+4+0+0', 4), b=vgpr('ValuA_X0_I0+32+0+0', 4), acc2=accvgpr(96, 4), neg=False, comment='left value = acc[96+0:99+0]'),
        MFMAInstruction(instType=InstType.INST_BF16, accType=InstType.INST_F32, variant=[16, 16, 32, 1], mfma1k=False, acc=accvgpr(100, 4), a=vgpr('ValuB_X0_I0+4+0+0', 4), b=vgpr('ValuA_X0_I0+36+0+0', 4), acc2=accvgpr(100, 4), neg=False, comment='left value = acc[100+0:103+0]'),
        MFMAInstruction(instType=InstType.INST_BF16, accType=InstType.INST_F32, variant=[16, 16, 32, 1], mfma1k=False, acc=accvgpr(104, 4), a=vgpr('ValuB_X0_I0+4+0+0', 4), b=vgpr('ValuA_X0_I0+40+0+0', 4), acc2=accvgpr(104, 4), neg=False, comment='left value = acc[104+0:107+0]'),
        MFMAInstruction(instType=InstType.INST_BF16, accType=InstType.INST_F32, variant=[16, 16, 32, 1], mfma1k=False, acc=accvgpr(108, 4), a=vgpr('ValuB_X0_I0+4+0+0', 4), b=vgpr('ValuA_X0_I0+44+0+0', 4), acc2=accvgpr(108, 4), neg=False, comment='left value = acc[108+0:111+0]'),
        MFMAInstruction(instType=InstType.INST_BF16, accType=InstType.INST_F32, variant=[16, 16, 32, 1], mfma1k=False, acc=accvgpr(112, 4), a=vgpr('ValuB_X0_I0+4+0+0', 4), b=vgpr('ValuA_X0_I0+48+0+0', 4), acc2=accvgpr(112, 4), neg=False, comment='left value = acc[112+0:115+0]'),
        MFMAInstruction(instType=InstType.INST_BF16, accType=InstType.INST_F32, variant=[16, 16, 32, 1], mfma1k=False, acc=accvgpr(116, 4), a=vgpr('ValuB_X0_I0+4+0+0', 4), b=vgpr('ValuA_X0_I0+52+0+0', 4), acc2=accvgpr(116, 4), neg=False, comment='left value = acc[116+0:119+0]'),
        MFMAInstruction(instType=InstType.INST_BF16, accType=InstType.INST_F32, variant=[16, 16, 32, 1], mfma1k=False, acc=accvgpr(120, 4), a=vgpr('ValuB_X0_I0+4+0+0', 4), b=vgpr('ValuA_X0_I0+56+0+0', 4), acc2=accvgpr(120, 4), neg=False, comment='left value = acc[120+0:123+0]'),
        MFMAInstruction(instType=InstType.INST_BF16, accType=InstType.INST_F32, variant=[16, 16, 32, 1], mfma1k=False, acc=accvgpr(124, 4), a=vgpr('ValuB_X0_I0+4+0+0', 4), b=vgpr('ValuA_X0_I0+60+0+0', 4), acc2=accvgpr(124, 4), neg=False, comment='left value = acc[124+0:127+0]'),
        DSLoadB128(dst=vgpr('ValuB_X0_I0+12', 4), src=vgpr('LocalReadAddrB', 1), ds=DSModifiers(na=1, offset=384), comment='L -> Reg lro=0 swapByteOffset=0 ti=256 vIdx=0 eIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0'),
        SWaitCnt(dscnt=1, comment='Wait for dependent lr'),
        MFMAInstruction(instType=InstType.INST_BF16, accType=InstType.INST_F32, variant=[16, 16, 32, 1], mfma1k=False, acc=accvgpr(128, 4), a=vgpr('ValuB_X0_I0+8+0+0', 4), b=vgpr('ValuA_X0_I0+0+0+0', 4), acc2=accvgpr(128, 4), neg=False, comment='left value = acc[128+0:131+0]'),
        DSLoadB128(dst=vgpr('ValuA_X1_I0+0', 4), src=vgpr('LocalReadAddrA', 1), ds=DSModifiers(na=1, offset=64), comment='L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0'),
        MFMAInstruction(instType=InstType.INST_BF16, accType=InstType.INST_F32, variant=[16, 16, 32, 1], mfma1k=False, acc=accvgpr(132, 4), a=vgpr('ValuB_X0_I0+8+0+0', 4), b=vgpr('ValuA_X0_I0+4+0+0', 4), acc2=accvgpr(132, 4), neg=False, comment='left value = acc[132+0:135+0]'),
        DSLoadB128(dst=vgpr('ValuB_X1_I0+0', 4), src=vgpr('LocalReadAddrB', 1), ds=DSModifiers(na=1, offset=64), comment='L -> Reg lro=32 swapByteOffset=0 ti=256 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0'),
        MFMAInstruction(instType=InstType.INST_BF16, accType=InstType.INST_F32, variant=[16, 16, 32, 1], mfma1k=False, acc=accvgpr(136, 4), a=vgpr('ValuB_X0_I0+8+0+0', 4), b=vgpr('ValuA_X0_I0+8+0+0', 4), acc2=accvgpr(136, 4), neg=False, comment='left value = acc[136+0:139+0]'),
        DSLoadB128(dst=vgpr('ValuA_X1_I0+4', 4), src=vgpr('LocalReadAddrA', 1), ds=DSModifiers(na=1, offset=192), comment='L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0'),
        MFMAInstruction(instType=InstType.INST_BF16, accType=InstType.INST_F32, variant=[16, 16, 32, 1], mfma1k=False, acc=accvgpr(140, 4), a=vgpr('ValuB_X0_I0+8+0+0', 4), b=vgpr('ValuA_X0_I0+12+0+0', 4), acc2=accvgpr(140, 4), neg=False, comment='left value = acc[140+0:143+0]'),
        DSLoadB128(dst=vgpr('ValuA_X1_I0+8', 4), src=vgpr('LocalReadAddrA', 1), ds=DSModifiers(na=1, offset=320), comment='L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=2 rIdx=0 oIdx=0 buffer=1 iui=0'),
        MFMAInstruction(instType=InstType.INST_BF16, accType=InstType.INST_F32, variant=[16, 16, 32, 1], mfma1k=False, acc=accvgpr(144, 4), a=vgpr('ValuB_X0_I0+8+0+0', 4), b=vgpr('ValuA_X0_I0+16+0+0', 4), acc2=accvgpr(144, 4), neg=False, comment='left value = acc[144+0:147+0]'),
        DSLoadB128(dst=vgpr('ValuA_X1_I0+12', 4), src=vgpr('LocalReadAddrA', 1), ds=DSModifiers(na=1, offset=448), comment='L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=3 rIdx=0 oIdx=0 buffer=1 iui=0'),
        MFMAInstruction(instType=InstType.INST_BF16, accType=InstType.INST_F32, variant=[16, 16, 32, 1], mfma1k=False, acc=accvgpr(148, 4), a=vgpr('ValuB_X0_I0+8+0+0', 4), b=vgpr('ValuA_X0_I0+20+0+0', 4), acc2=accvgpr(148, 4), neg=False, comment='left value = acc[148+0:151+0]'),
        DSLoadB128(dst=vgpr('ValuA_X1_I0+16', 4), src=vgpr('LocalReadAddrA', 1), ds=DSModifiers(na=1, offset=576), comment='L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=4 rIdx=0 oIdx=0 buffer=1 iui=0'),
        MFMAInstruction(instType=InstType.INST_BF16, accType=InstType.INST_F32, variant=[16, 16, 32, 1], mfma1k=False, acc=accvgpr(152, 4), a=vgpr('ValuB_X0_I0+8+0+0', 4), b=vgpr('ValuA_X0_I0+24+0+0', 4), acc2=accvgpr(152, 4), neg=False, comment='left value = acc[152+0:155+0]'),
        DSLoadB128(dst=vgpr('ValuA_X1_I0+20', 4), src=vgpr('LocalReadAddrA', 1), ds=DSModifiers(na=1, offset=704), comment='L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=5 rIdx=0 oIdx=0 buffer=1 iui=0'),
        MFMAInstruction(instType=InstType.INST_BF16, accType=InstType.INST_F32, variant=[16, 16, 32, 1], mfma1k=False, acc=accvgpr(156, 4), a=vgpr('ValuB_X0_I0+8+0+0', 4), b=vgpr('ValuA_X0_I0+28+0+0', 4), acc2=accvgpr(156, 4), neg=False, comment='left value = acc[156+0:159+0]'),
        DSLoadB128(dst=vgpr('ValuA_X1_I0+24', 4), src=vgpr('LocalReadAddrA', 1), ds=DSModifiers(na=1, offset=832), comment='L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=6 rIdx=0 oIdx=0 buffer=1 iui=0'),
        MFMAInstruction(instType=InstType.INST_BF16, accType=InstType.INST_F32, variant=[16, 16, 32, 1], mfma1k=False, acc=accvgpr(160, 4), a=vgpr('ValuB_X0_I0+8+0+0', 4), b=vgpr('ValuA_X0_I0+32+0+0', 4), acc2=accvgpr(160, 4), neg=False, comment='left value = acc[160+0:163+0]'),
        DSLoadB128(dst=vgpr('ValuA_X1_I0+28', 4), src=vgpr('LocalReadAddrA', 1), ds=DSModifiers(na=1, offset=960), comment='L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=7 rIdx=0 oIdx=0 buffer=1 iui=0'),
        MFMAInstruction(instType=InstType.INST_BF16, accType=InstType.INST_F32, variant=[16, 16, 32, 1], mfma1k=False, acc=accvgpr(164, 4), a=vgpr('ValuB_X0_I0+8+0+0', 4), b=vgpr('ValuA_X0_I0+36+0+0', 4), acc2=accvgpr(164, 4), neg=False, comment='left value = acc[164+0:167+0]'),
        DSLoadB128(dst=vgpr('ValuA_X1_I0+32', 4), src=vgpr('LocalReadAddrA', 1), ds=DSModifiers(na=1, offset=16960), comment='L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0'),
        MFMAInstruction(instType=InstType.INST_BF16, accType=InstType.INST_F32, variant=[16, 16, 32, 1], mfma1k=False, acc=accvgpr(168, 4), a=vgpr('ValuB_X0_I0+8+0+0', 4), b=vgpr('ValuA_X0_I0+40+0+0', 4), acc2=accvgpr(168, 4), neg=False, comment='left value = acc[168+0:171+0]'),
        DSLoadB128(dst=vgpr('ValuA_X1_I0+36', 4), src=vgpr('LocalReadAddrA', 1), ds=DSModifiers(na=1, offset=17088), comment='L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=1 eIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0'),
        MFMAInstruction(instType=InstType.INST_BF16, accType=InstType.INST_F32, variant=[16, 16, 32, 1], mfma1k=False, acc=accvgpr(172, 4), a=vgpr('ValuB_X0_I0+8+0+0', 4), b=vgpr('ValuA_X0_I0+44+0+0', 4), acc2=accvgpr(172, 4), neg=False, comment='left value = acc[172+0:175+0]'),
        DSLoadB128(dst=vgpr('ValuA_X1_I0+40', 4), src=vgpr('LocalReadAddrA', 1), ds=DSModifiers(na=1, offset=17216), comment='L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=1 eIdx=2 rIdx=0 oIdx=0 buffer=1 iui=0'),
        MFMAInstruction(instType=InstType.INST_BF16, accType=InstType.INST_F32, variant=[16, 16, 32, 1], mfma1k=False, acc=accvgpr(176, 4), a=vgpr('ValuB_X0_I0+8+0+0', 4), b=vgpr('ValuA_X0_I0+48+0+0', 4), acc2=accvgpr(176, 4), neg=False, comment='left value = acc[176+0:179+0]'),
        DSLoadB128(dst=vgpr('ValuA_X1_I0+44', 4), src=vgpr('LocalReadAddrA', 1), ds=DSModifiers(na=1, offset=17344), comment='L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=1 eIdx=3 rIdx=0 oIdx=0 buffer=1 iui=0'),
        MFMAInstruction(instType=InstType.INST_BF16, accType=InstType.INST_F32, variant=[16, 16, 32, 1], mfma1k=False, acc=accvgpr(180, 4), a=vgpr('ValuB_X0_I0+8+0+0', 4), b=vgpr('ValuA_X0_I0+52+0+0', 4), acc2=accvgpr(180, 4), neg=False, comment='left value = acc[180+0:183+0]'),
        DSLoadB128(dst=vgpr('ValuA_X1_I0+48', 4), src=vgpr('LocalReadAddrA', 1), ds=DSModifiers(na=1, offset=17472), comment='L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=1 eIdx=4 rIdx=0 oIdx=0 buffer=1 iui=0'),
        DSLoadB128(dst=vgpr('ValuA_X1_I0+52', 4), src=vgpr('LocalReadAddrA', 1), ds=DSModifiers(na=1, offset=17600), comment='L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=1 eIdx=5 rIdx=0 oIdx=0 buffer=1 iui=0'),
        DSLoadB128(dst=vgpr('ValuA_X1_I0+56', 4), src=vgpr('LocalReadAddrA', 1), ds=DSModifiers(na=1, offset=17728), comment='L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=1 eIdx=6 rIdx=0 oIdx=0 buffer=1 iui=0'),
        DSLoadB128(dst=vgpr('ValuA_X1_I0+60', 4), src=vgpr('LocalReadAddrA', 1), ds=DSModifiers(na=1, offset=17856), comment='L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=1 eIdx=7 rIdx=0 oIdx=0 buffer=1 iui=0'),
        DSLoadB128(dst=vgpr('ValuB_X1_I0+4', 4), src=vgpr('LocalReadAddrB', 1), ds=DSModifiers(na=1, offset=192), comment='L -> Reg lro=32 swapByteOffset=0 ti=256 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0'),
        DSLoadB128(dst=vgpr('ValuB_X1_I0+8', 4), src=vgpr('LocalReadAddrB', 1), ds=DSModifiers(na=1, offset=320), comment='L -> Reg lro=32 swapByteOffset=0 ti=256 vIdx=0 eIdx=2 rIdx=0 oIdx=0 buffer=1 iui=0'),
        DSLoadB128(dst=vgpr('ValuB_X1_I0+12', 4), src=vgpr('LocalReadAddrB', 1), ds=DSModifiers(na=1, offset=448), comment='L -> Reg lro=32 swapByteOffset=0 ti=256 vIdx=0 eIdx=3 rIdx=0 oIdx=0 buffer=1 iui=0'),
        SWaitCnt(dscnt=0, comment=''),
        SBarrier(),
    ]
    lra_insts = [
        vectorStaticRemainder(10, 7, 'Serial', 64, ContinuousRegister(8, 2), ContinuousRegister(48, 1), '0. thread id in wave: wtid = tid % wavelength(64)'),
        vectorStaticRemainder(10, 6, 7, 16, ContinuousRegister(8, 2), ContinuousRegister(48, 1), '1. N offset: nIdx = wtid % MI_N(16)'),
        vectorStaticMultiply(vgpr(6, 1), vgpr(6, 1), 64, ContinuousRegister(48, 1), '1. N offset: nOffset = nIdx * nStride(64)'),
        vectorStaticMultiply(vgpr(6, 1), vgpr(6, 1), 8, ContinuousRegister(48, 1), '4. apply VectorWidth: bnOffset = bnOffset * vw(8)'),
        vectorStaticDivide(7, 7, 16, ContinuousRegister(8, 2), '5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))'),
        vectorStaticMultiplyAdd(vgpr(6, 1), vgpr(7, 1), 8, vgpr(6, 1), ContinuousRegister(48, 1), '5. K offset: lrKOffset = kIdx * mStride(8); 6. offset in wave: lrOffset = bnOffset + lrKOffset'),
        vectorStaticRemainder(9, 8, 'Serial', 64, ContinuousRegister(10, 2), ContinuousRegister(48, 1), '0. thread id in wave: wtid = tid % wavelength(64)'),
        vectorStaticRemainder(9, 7, 8, 16, ContinuousRegister(10, 2), ContinuousRegister(48, 1), '1. N offset: nIdx = wtid % MI_N(16)'),
        vectorStaticMultiply(vgpr(7, 1), vgpr(7, 1), 64, ContinuousRegister(48, 1), '1. N offset: nOffset = nIdx * nStride(64)'),
        vectorStaticMultiply(vgpr(7, 1), vgpr(7, 1), 4, ContinuousRegister(48, 1), '4. apply VectorWidth: bnOffset = bnOffset * vw(4)'),
        vectorStaticDivide(8, 8, 16, ContinuousRegister(10, 2), '5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))'),
        vectorStaticMultiplyAdd(vgpr(7, 1), vgpr(8, 1), 8, vgpr(7, 1), ContinuousRegister(48, 1), '5. K offset: lrKOffset = kIdx * mStride(8); 6. offset in wave: lrOffset = bnOffset + lrKOffset'),
        vectorStaticDivide(9, 'Serial', 64, ContinuousRegister(10, 2), '7. wave offset in N dimen: wtid = tid / dividedForWaveId(64)'),
        vectorStaticRemainder(9, 9, 9, 4, ContinuousRegister(10, 2), ContinuousRegister(48, 1), '7. wave offset in M dimen: wtid0 = wtid / num1DWaves(4)'),
        vectorStaticMultiplyAdd(vgpr(7, 1), vgpr(9, 1), 4096, vgpr(7, 1), ContinuousRegister(48, 1), '7. wave offset in M dimen: wOffset = wtid0 * W0Stride(409...7. final local read offset: flrOffset = lrOffset + WOffset'),
        vectorStaticDivide(8, 'Serial', 64, ContinuousRegister(10, 2)),
        vectorStaticDivide(8, 8, 4, ContinuousRegister(10, 2), comment='LSU offset: Get LSU wave_id'),
        SMovB32(dst=sgpr(48, 1), src=64, comment='LSU offset: stride = lsuStride(64) when umlds==True'),
        VMulLOU32(dst=vgpr(8, 1), src0=sgpr(48, 1), src1=vgpr(8, 1), comment='LSU offset: lsuoffset = wave_id*lsuStride*(MT0+PAD)'),
        VAddLShiftLeftU32(dst=vgpr('LocalReadAddrA', 1), src0=vgpr(8, 1), src1=vgpr(6, 1), shiftHex='0x1', comment='Final Offset: offset = (lro0+lsuoffset)*bpeDS'),
        vectorStaticDivide(9, 'LocalReadAddrA', 1024, ContinuousRegister(10, 2), 'Final Offset: padding 32 per block 1024'),
        vectorStaticMultiplyAdd(vgpr('LocalReadAddrA', 1), vgpr(9, 1), 32, vgpr('LocalReadAddrA', 1), ContinuousRegister(48, 1), 'Final Offset: padding 32 per block 1024'),
        vectorStaticDivide(6, 'Serial', 64, ContinuousRegister(10, 2)),
        vectorStaticDivide(6, 6, 4, ContinuousRegister(10, 2), comment='LSU offset: Get LSU wave_id'),
        SMovB32(dst=sgpr(48, 1), src=64, comment='LSU offset: stride = lsuStride(64) when umlds==True'),
        VMulLOU32(dst=vgpr(6, 1), src0=sgpr(48, 1), src1=vgpr(6, 1), comment='LSU offset: lsuoffset = wave_id*lsuStride*(MT1+PAD)'),
        VAddLShiftLeftU32(dst=vgpr('LocalReadAddrB', 1), src0=vgpr(6, 1), src1=vgpr(7, 1), shiftHex='0x1', comment='Final Offset: offset = (lro1+lsuoffset)*bpeDS'),
        vectorStaticDivide(8, 'LocalReadAddrB', 512, ContinuousRegister(10, 2), 'Final Offset: padding 32 per block 512'),
        vectorStaticMultiplyAdd(vgpr('LocalReadAddrB', 1), vgpr(8, 1), 32, vgpr('LocalReadAddrB', 1), ContinuousRegister(48, 1), 'Final Offset: padding 32 per block 512'),
        VAddCOU32(dst=vgpr('LocalReadAddrB+0', 1), dst1=VCC(), src0='0x8400', src1=vgpr('LocalReadAddrB+0', 1), comment=' += LdsOffsetB (lower)'),
    ]
    return loop_insts, lra_insts, 680

@gfx((9,5,0))
def lr_test_case():
    bc2 = 1
    bc = 2 ** bc2
    lra_insts = [
        VLShiftLeftB32(dst=vgpr("LocalReadAddrA", 1), shiftHex=hex(4 + bc2), src=vgpr("Serial")),
    ]
    data_offset = int(128 / 32)
    data_per_vgpr = data_offset * 4 * 256 * bc
    loop_insts = [
        DSLoadB128(
            dst=vgpr("ValuA_X0_I0+%d" % (data_offset * i), data_offset),
            src=vgpr("LocalReadAddrA", 1),
            ds=DSModifiers(offset = i * data_per_vgpr),
        ) for i in range(3)
    ]
    loop_insts.append(SWaitCnt(dscnt=0, comment=''))
    return loop_insts, lra_insts, 144

def get_cycles(module, num_waves=4):
    ripo = rocIsaPassOption()
    ripo.numWaves = num_waves
    pass_result = rocIsaPass(module, ripo)
    return pass_result.cycles

def create_test_module(loop_insts, lra_insts):
    kernel_body = KernelBody("kernelBody")
    ti = rocIsa.getInstance()
    regcaps = ti.getRegCaps()
    kernel_body.setGprs(totalVgprs=regcaps["MaxVgpr"], totalAgprs=256, totalSgprs=regcaps["MaxSgpr"])
    module = Module()
    loop_module = Module("loopBody")
    lra_module = Module("Local Read Addresses")
    for inst in lra_insts:
        lra_module.add(inst)
    for inst in loop_insts:
        loop_module.add(inst)
    module.add(lra_module)
    module.add(loop_module)
    kernel_body.addBody(module)
    return kernel_body

def check_cycles(test_case_f):
    loop_insts, lra_insts, ground_cycles = test_case_f()
    kernel_body = create_test_module(loop_insts, lra_insts)
    cycles = get_cycles(kernel_body)
    return cycles, ground_cycles

@pytest.mark.parametrize("test_case_f", [
    empty_case,
    lr_test_case,
    simple_lr_case,
    simple_mfma_case
])
def test_cycles(test_case_f):
    cycles, ground_cycles = check_cycles(test_case_f)
    assert cycles == ground_cycles
