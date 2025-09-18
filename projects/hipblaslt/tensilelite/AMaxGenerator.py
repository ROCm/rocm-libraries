################################################################################
#
# Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

from rocisa.code import Label, Module, RegSet, TextBlock, ValueSet, SrdUpperValue
from rocisa.container import EXEC, VCC, DSModifiers, MUBUFModifiers, vgpr, sgpr
from rocisa.enum import RegisterType
from rocisa.register import RegisterPool
import rocisa.instruction as ri

from argparse import ArgumentParser
from dataclasses import dataclass
from typing import List, Tuple, Optional
from math import log2
import os
import yaml
import json
import collections
from contextlib import contextmanager
from Tensile.Common.Utilities import _global_ti
from Tensile.Common.Architectures import detectGlobalCurrentISA, isaToGfx, gfxToIsa
from Tensile.Common.DataType import DataType
from Tensile.Common.GlobalParameters import restoreDefaultGlobalParameters, assignGlobalParameters
from Tensile.Common.Types import IsaVersion
from Tensile.Toolchain.Validators import ToolchainDefaults, validateToolchain

def kernel_header(name: str, gfx_arch: str, vgpr: int, sgpr: int, lds: int):
    vgpr = ((vgpr+7)//8)*8
    sgpr = ((sgpr+7)//8)*8
    lds  = ((lds+31)//32)*32

    header = ""
    header += f'.amdgcn_target "amdgcn-amd-amdhsa--{gfx_arch}"\n'
    header += f'.text\n'
    header += f'.protected {name}\n'
    header += f'.globl {name}\n'
    header += f'.p2align 8\n'
    header += f'.type {name},@function\n'
    header += f'.section .rodata,#alloc\n'
    header += f'.p2align 6\n'
    header += f'.amdhsa_kernel {name}\n'
    header += f'  .amdhsa_user_sgpr_kernarg_segment_ptr 1\n'
    if (gfx_arch not in ("gfx900", "gfx908", "gfx1030", "gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1150", "gfx1151", "gfx1200", "gfx1201")):
        header += f'  .amdhsa_accum_offset {vgpr} // accvgpr offset\n'
    header += f'  .amdhsa_next_free_vgpr {vgpr} // vgprs\n'
    header += f'  .amdhsa_next_free_sgpr {sgpr} // sgprs\n'
    header += f'  .amdhsa_group_segment_fixed_size {lds} // lds bytes\n'
    header += f'  .amdhsa_private_segment_fixed_size 0\n'
    header += f'  .amdhsa_system_sgpr_workgroup_id_x 1\n'
    header += f'  .amdhsa_system_sgpr_workgroup_id_y 1\n'
    header += f'  .amdhsa_system_sgpr_workgroup_id_z 1\n'
    header += f'  .amdhsa_system_vgpr_workitem_id 0\n'
    header += f'  .amdhsa_float_denorm_mode_32 3\n'
    header += f'  .amdhsa_float_denorm_mode_16_64 3\n'
    if _global_ti.getArchCaps()["HasWave32"]:
        header += f'  .amdhsa_wavefront_size32 1\n'
    header += f'.end_amdhsa_kernel\n'
    header += f'.text\n'
    return header

@contextmanager
def asm_func(func_name: str, module: Module):
    try:
        module.add(TextBlock(f'{func_name}:\n'))
        yield
    finally:
        end_label_name = f'.L{func_name}_end'
        module.add(ri.SEndpgm())
        module.add(TextBlock(f'{end_label_name}:\n'))
        module.add(TextBlock(f'.size {func_name}, {end_label_name} - {func_name}\n'))

@contextmanager
def asm_loop(mod: Module, name: str, it: str):
    try:
        loop_start_label = Label(name, f'loop {name} starts')
        loop_end_label = Label(f'{name}_end', f'loop {name} ends')
        mod.add(loop_start_label)
        mod.add(ri.SCmpEQU32(sgpr(it), 0))
        mod.add(ri.SCBranchSCC1(loop_end_label.getLabelName()))
        mod.addSpaceLine()
        yield
    finally:
        mod.add(ri.SSubU32(sgpr(it), sgpr(it), 1))
        mod.add(ri.SBranch(loop_start_label.getLabelName()))
        mod.add(loop_end_label)
        mod.addSpaceLine()


class AMaxKernelGenerator:
    srd_num_reg = 4
    srd_alignment = 4

    def __init__(self,
                 i_type: DataType,
                 o_type: DataType,
                 scale_type: DataType,
                 num_workitems: int,
                 num_load_count: int,
                 num_load_size: int,
                 wavefront_size: int,
                 arch: str,
                 isa: IsaVersion,
                 is_scale: bool):
        self.i_type = i_type
        self.o_type = o_type
        self.scale_type = scale_type
        self.bpe = i_type.numBytes()
        self.num_workitems = num_workitems
        self.num_load_count = num_load_count
        self.num_load_size = num_load_size
        self.wavefront_size = wavefront_size
        self.laneSGPRCount = 2 if wavefront_size == 64 else 1
        self.SMovBX = ri.SMovB64 if wavefront_size == 64 else ri.SMovB32
        self.SAndBX = ri.SAndB64 if wavefront_size == 64 else ri.SAndB32
        self.srdUpperValue = SrdUpperValue(isa)
        self.sgpr_pool = RegisterPool(24, RegisterType.Sgpr, True)
        self.vgpr_pool = RegisterPool(40, RegisterType.Vgpr, True)
        self.sgpr_pool.add(0, 23) #TODO: estimate this
        self.vgpr_pool.add(0, 39) #TODO: estimate this
        self.debug_label = True
        self.arch = arch
        self.is_scale = is_scale
        self.op = 'AMax'
        self.sgprs  = collections.OrderedDict()
        self.vgprs  = collections.OrderedDict()

    @property
    def lds_usage_byte(self) -> int:
        # used in reduce inter wave mean and invvar
        # 4 data * half_wave_num * bpe
        return 4 * (self.num_workitems // self.wavefront_size // 2) * self.bpe

    @property
    def func_name(self):
        if self.is_scale:
            return f'AMax_Ti_{self.i_type}_To_{self.o_type}_Ts_{self.scale_type}_W_{self.num_workitems}_C_{self.num_load_count}'
        return f'AMax_Ti_{self.i_type}_To_{self.o_type}_W_{self.num_workitems}_C_{self.num_load_count}'

    def dumps(self, format: str) -> str:
        param_dict = {
            'arch': self.arch,
            'op': self.op,
            'is_scale': self.is_scale,
            'func_name': self.func_name,
            'io_type': self.i_type.toChar(),
            'o_type': self.o_type.toChar(),
            'scale_type': self.scale_type.toChar(),
            'num_workitems': self.num_workitems,
        }

        if format.lower() == 'yaml':
            return yaml.dump(param_dict)
        elif format.lower() == 'json':
            return json.dumps(param_dict)
        else:
            assert False, f'Unsupported format {format}'

    def dump(self, format: str, output_path: str):
        s = self.dumps(format)
        with open(output_path, 'w') as f:
            f.write(s)


    def local_read_inst_type(self, num_elements: int):
        if self.i_type.isSingle():
            insts = {
                1: ri.DSLoadB32,
                2: ri.DSLoadB64,
                4: ri.DSLoadB128
            }
        elif self.i_type.isHalf():
            insts = {
                1: ri.DSLoadU16,
                2: ri.DSLoadB32,
                4: ri.DSLoadB64,
                8: ri.DSLoadB128
            }
        return insts[num_elements]


    def local_write_inst_type(self, num_elements: int):
        if self.i_type.isSingle():
            insts = {
                1: ri.DSStoreB32,
                2: ri.DSStoreB64,
                4: ri.DSStoreB128
            }
        elif self.i_type.isHalf():
            insts = {
                1: ri.DSStoreB16,
                2: ri.DSStoreB32,
                4: ri.DSStoreB64,
                8: ri.DSStoreB128
            }
        return insts[num_elements]


    def global_read_inst_type(self, num_elements: int):
        if self.i_type.isSingle():
            insts = {
                1: ri.BufferLoadB32,
                2: ri.BufferLoadB64,
                4: ri.BufferLoadB128
            }
        elif self.i_type.isHalf():
            insts = {
                1: ri.BufferLoadD16B16,
                2: ri.BufferLoadB32,
                4: ri.BufferLoadB64,
                8: ri.BufferLoadB128
            }
        else:
            raise NotImplementedError
        return insts[num_elements]

    def global_write_inst_type(self, num_elements: int):
        if self.o_type.isSingle():
            insts = {
                1: ri.BufferStoreB32,
                2: ri.BufferStoreB64,
                4: ri.BufferStoreB128
            }
        elif self.o_type.isHalf():
            insts = {
                1: ri.BufferStoreB16,
                2: ri.BufferStoreB32,
                4: ri.BufferStoreB64,
                8: ri.BufferStoreB128
            }
        else:
            raise NotImplementedError
        return insts[num_elements]



    def defineSgpr(self, name, numSgprs, align=1):
        if numSgprs == 0: return
        sgprIdx = self.sgpr_pool.checkOutAligned(numSgprs, align, tag=name, preventOverflow=False)
        self.sgprs[name] = sgprIdx
        return sgprIdx

    def defineVgpr(self, name, numVgprs, align=1):
        if numVgprs == 0: return
        vgprIdx = self.vgpr_pool.checkOutAligned(numVgprs, align, tag=name, preventOverflow=False)
        self.vgprs[name] = vgprIdx
        return vgprIdx


    def kernel_args(self):
        if self.is_scale:
            return (KernelArgument(8, 0,  'global_buffer', 'global'),
                    KernelArgument(8, 8,  'global_buffer', 'global'),
                    KernelArgument(8, 16, 'global_buffer', 'global'),
                    KernelArgument(8, 24, 'global_buffer', 'global'),
                    KernelArgument(4, 32, 'by_value'))
        return (KernelArgument(8,  0, 'global_buffer', 'global'),
                KernelArgument(8,  8, 'global_buffer', 'global'),
                KernelArgument(4, 16, 'by_value'))


    def defineVariables(self):
        self.defineVgpr("Serial",  1, 1)
        self.defineVgpr("Output",  1, 1)
        self.defineVgpr("OutputB", 1, 1)
        self.defineVgpr("Widx",    1, 1)
        self.defineVgpr("Offset",  4, 1)
        self.defineVgpr("Value",   self.num_load_count * self.num_load_size, self.num_load_size)
        self.defineVgpr("Tmp",     4, 1)
        if self.is_scale:
            self.defineVgpr("OffsetD", self.num_load_count * self.num_load_size, 1)
            self.defineVgpr("OutputD", self.num_load_count * self.num_load_size, self.num_load_size)
            self.defineVgpr("TmpD",    4, 1)
            if self.scale_type == DataType("F8") or self.scale_type == DataType("F8N"):
                self.defineVgpr("Fp8NanInf", 1, 1)
                self.defineVgpr("Fp8Max",    1, 1)
                self.defineVgpr("Fp8Min",    1, 1)
                self.defineVgpr("Fp8Tmp",    1, 1)
            elif self.scale_type == DataType("B8") or self.scale_type == DataType("B8N"):
                self.defineVgpr("BF8NanInf", 1, 1)
                self.defineVgpr("BF8Max",    1, 1)
                self.defineVgpr("BF8Min",    1, 1)
                self.defineVgpr("BF8Tmp",    1, 1)

        self.defineSgpr("KernelArg", 2)
        self.defineSgpr("WorkGroup0", 1)
        self.defineSgpr("WorkGroup1", 1)
        self.defineSgpr("WorkGroup2", 1)
        self.defineSgpr("AddressOut", 2, 2)
        self.defineSgpr("AddressOutD", 2, 2)
        self.defineSgpr("AddressIn", 2, 2)
        self.defineSgpr("AddressScale", 2, 2)
        self.defineSgpr("SizeLength", 1)
        self.defineSgpr("MainLoop", 1)
        self.defineSgpr("Offset", 1)
        self.defineSgpr("Src", 4, 4)
        self.defineSgpr("Dst", 4, 4)
        self.defineSgpr("Tmp", 6, 2)
        if self.is_scale:
            self.defineSgpr("DstD", 4, 4)
            self.defineSgpr("TmpD", 6, 2)
            self.defineSgpr("Scale", 1)

        mod = Module("defineVariables")

        for vkey in self.vgprs:
            mod.add(RegSet("v", "vgpr"+vkey, self.vgprs[vkey]))
        mod.addSpaceLine()

        for skey in self.sgprs:
            mod.add(RegSet("s", "sgpr"+skey, self.sgprs[skey]))
        mod.addSpaceLine()

        mod.add(ValueSet("Srd127_96", self.srdUpperValue.getValue(), format=1))
        mod.addSpaceLine()
        mod.addSpaceLine()
        return mod


    def load_kernel_args(self):
        mod = Module('Load kernel args')
        mod.addComment0('Load kernel args')
        if self.is_scale:
            mod.add(ri.SLoadB64(sgpr("AddressOut", 2),    sgpr("KernelArg", 2),  0))
            mod.add(ri.SLoadB64(sgpr("AddressOutD", 2),   sgpr("KernelArg", 2),  8))
            mod.add(ri.SLoadB64(sgpr("AddressIn", 2),     sgpr("KernelArg", 2),  16))
            mod.add(ri.SLoadB64(sgpr("AddressScale", 2),  sgpr("KernelArg", 2),  24))
            mod.add(ri.SLoadB32(sgpr("SizeLength"),       sgpr("KernelArg", 2),  32))
        else:
            mod.add(ri.SLoadB64(sgpr("AddressOut", 2),    sgpr("KernelArg", 2),  0))
            mod.add(ri.SLoadB64(sgpr("AddressIn", 2),     sgpr("KernelArg", 2),  8))
            mod.add(ri.SLoadB32(sgpr("SizeLength"),       sgpr("KernelArg", 2),  16))
        mod.add(ri.SWaitCnt(kmcnt=0))
        mod.addSpaceLine()
        mod.addSpaceLine()
        return mod

    def init_param(self) -> Module:
        mod = Module("init_param")
        mod.addComment0("init_param")
        mod.add(ri.SLShiftLeftB32(sgpr("Tmp"), int(log2(self.bpe)), sgpr("SizeLength")))
        mod.addSpaceLine()

        mod.add(ri.SMovB32(sgpr("Dst+0"), sgpr("AddressOut+0")))
        mod.add(ri.SMovB32(sgpr("Dst+1"), sgpr("AddressOut+1")))
        mod.add(ri.SMovB32(sgpr("Dst+2"), self.o_type.numBytes()))
        mod.add(ri.SMovB32(sgpr("Dst+3"), "Srd127_96"))
        mod.addSpaceLine()

        if self.is_scale: # init inputScale
            mod.add(ri.SLoadB32(sgpr("Scale"), sgpr("AddressScale", 2), 0))
            mod.add(ri.SWaitCnt(kmcnt=0))
            mod.addSpaceLine()

        mod.add(ri.SMovB32(sgpr("Src+0"), sgpr("AddressIn+0")))
        mod.add(ri.SMovB32(sgpr("Src+1"), sgpr("AddressIn+1")))
        mod.add(ri.SMovB32(sgpr("Src+2"), sgpr("Tmp")))
        mod.add(ri.SMovB32(sgpr("Src+3"), "Srd127_96"))
        mod.addSpaceLine()

        mod.add(ri.VMovB32(vgpr("Output"), 0))
        if self.is_scale:
            if self.scale_type == DataType("F8N"):
                mod.add(ri.VMovB32(vgpr("Fp8NanInf"), "0x207", comment="Nan and +/- inf"))
                mod.add(ri.VMovB32(vgpr("Fp8Max"), "0x43700000", comment="Fp8 NANOO Max value 240 as float32"))
                mod.add(ri.VMovB32(vgpr("Fp8Min"), "0xc3700000", comment="Fp8 NANOO Min value -240 as float32"))
            elif self.scale_type == DataType("F8"):
                mod.add(ri.VMovB32(vgpr("Fp8NanInf"), "0x207", comment="Nan and +/- inf"))
                mod.add(ri.VMovB32(vgpr("Fp8Max"), "0x43e00000", comment="Fp8 Max value 448 as float32"))
                mod.add(ri.VMovB32(vgpr("Fp8Min"), "0xc3e00000", comment="Fp8 Min value -448 as float32"))
            elif self.scale_type == DataType("B8") or self.scale_type == DataType("B8N"):
                mod.add(ri.VMovB32(vgpr("BF8NanInf"), "0x207", comment="Nan and +/- inf"))
                mod.add(ri.VMovB32(vgpr("BF8Max"), "0x47600000", comment="BF8 Max value 57344 as float32"))
                mod.add(ri.VMovB32(vgpr("BF8Min"), "0xc7600000", comment="BF8 Min value -57344 as float32"))
        mod.addSpaceLine()

        if self.is_scale:
            mod.add(ri.SMovB32(sgpr("DstD+0"), sgpr("AddressOutD+0")))
            mod.add(ri.SMovB32(sgpr("DstD+1"), sgpr("AddressOutD+1")))
            mod.add(ri.SMovB32(sgpr("DstD+2"), sgpr("SizeLength")))
            mod.add(ri.SMovB32(sgpr("DstD+3"), "Srd127_96"))
            for i in range(self.num_load_count * self.num_load_size):
                mod.add(ri.VMovB32(vgpr(f"OutputD+{i}"), 0))

        mod.addSpaceLine()
        return mod


    def calculate_global_address(self) -> Module:
        mod = Module("calculate_global_address")
        mod.addComment0("calculate_global_address")
        # offset for buffer load
        # total load size = dwordx4 = 16 bytes per PE
        mod.add(ri.VLShiftLeftB32(vgpr("Offset+0"), hex(int(log2(self.num_load_size * 4))), vgpr("Serial")))
        mod.add(ri.SMovB32(sgpr("Tmp"), self.num_workitems * self.num_load_size * 4))
        for i in range(0, self.num_load_count-1):
            mod.add(ri.VAddU32(vgpr(f"Offset+{i+1}"), vgpr(f"Offset+{i}"), sgpr("Tmp")))
        mod.addSpaceLine()

        if self.is_scale: # offset for buffer store
            # total store size = 1byte x 4 = 4 bytes per PE
            mod.add(ri.VLShiftLeftB32(vgpr("OffsetD+0"), hex(int(log2(4))), vgpr("Serial")))
            mod.add(ri.SMovB32(sgpr("TmpD"), 1))
            for i in range(self.num_load_size - 1):
                mod.add(ri.VAddU32(vgpr(f"OffsetD+{i+1}"), vgpr(f"OffsetD+{i}"), sgpr("TmpD")))
            mod.add(ri.SMovB32(sgpr("TmpD"), self.num_workitems * 4))
            for i in range(0, self.num_load_count-1):
                for j in range(0, self.num_load_size):
                    mod.add(ri.VAddU32(vgpr(f"OffsetD+{j+(i+1)*self.num_load_size}"), \
                                       vgpr(f"OffsetD+{j+i*self.num_load_size}"), sgpr("TmpD")))

        mod.addSpaceLine()
        mod.addSpaceLine()
        return mod


    def max_per_data(self, i, onlyOneElement = False) -> Module:
        mod = Module("max_per_data")
        if (self.i_type.isHalf()):
            mod.add(ri.VMaxF16(vgpr("Output"), vgpr("Output"), vgpr(f"Value+{i}", isAbs=True)))
            # On non-Ecc hardware, the top 16 bits are dirty
            if not onlyOneElement:
                mod.add(ri.VLShiftRightB32(vgpr(f"Value+{i}"), 16, vgpr(f"Value+{i}")))
                mod.add(ri.VMaxF16(vgpr("Output"), vgpr("Output"), vgpr(f"Value+{i}", isAbs=True)))
        elif (self.i_type.isSingle()):
            mod.add(ri.VMaxF32(vgpr("Output"), vgpr("Output"), vgpr(f"Value+{i}", isAbs=True)))
        return mod


    def scale_per_data(self, i) -> Module:
        mod = Module("scale_per_data")
        if self.is_scale:
            mod.add(ri.VMulF32(vgpr(f"OutputD+{i}"), sgpr("Scale"), vgpr(f"Value+{i}")))
            if self.scale_type == DataType("F8") or self.scale_type == DataType("F8N"):
                mod.add(ri.VCmpClassF32(dst=VCC(), src0=vgpr(f"OutputD+{i}"), src1=vgpr("Fp8NanInf")))
                mod.add(ri.VMed3F32(dst=vgpr("Fp8Tmp"), src0=vgpr(f"OutputD+{i}"), src1=vgpr("Fp8Min"), src2=vgpr("Fp8Max")))
                mod.add(ri.VCndMaskB32(dst=vgpr(f"OutputD+{i}"), src0=vgpr("Fp8Tmp"), src1=vgpr(f"OutputD+{i}"), src2=VCC()))
                mod.add(ri.VCvtPkF32toFP8(vgpr(f"OutputD+{i}"), vgpr(f"OutputD+{i}"), vgpr(f"OutputD+{i}")))
            elif self.scale_type == DataType("B8") or self.scale_type == DataType("B8N"):
                mod.add(ri.VCmpClassF32(dst=VCC(), src0=vgpr(f"OutputD+{i}"), src1=vgpr("BF8NanInf")))
                mod.add(ri.VMed3F32(dst=vgpr("BF8Tmp"), src0=vgpr(f"OutputD+{i}"), src1=vgpr("BF8Min"), src2=vgpr("BF8Max")))
                mod.add(ri.VCndMaskB32(dst=vgpr(f"OutputD+{i}"), src0=vgpr("BF8Tmp"), src1=vgpr(f"OutputD+{i}"), src2=VCC()))
                mod.add(ri.VCvtPkF32toBF8(vgpr(f"OutputD+{i}"), vgpr(f"OutputD+{i}"), vgpr(f"OutputD+{i}")))
        return mod


    def sum_per_threadxN(self) -> Module:
        mod = Module("sum_per_threadxN")
        mod.addComment0("sum_per_threadxN")
        mod.add(ri.SLShiftRightB32(sgpr("MainLoop"), \
                                   int(log2(self.num_workitems * self.num_load_count * self.num_load_size * (4 // self.bpe))), \
                                   sgpr("SizeLength")))
        with asm_loop(mod, "sum_per_threadxN", "MainLoop"):
            for i in range(0, self.num_load_count): # unroll
                mod.add(ri.BufferLoadB128(vgpr(f"Value+{i*self.num_load_size}",4), \
                                          vgpr(f"Offset+{i}"), sgpr("Src",4), 0, MUBUFModifiers(offen=True)))
            mod.addSpaceLine()
            # max operation
            for i in range(0, self.num_load_count): # unroll
                mod.add(ri.SWaitCnt(vlcnt=(self.num_load_count-i-1)))
                for j in range(0, self.num_load_size): # dwordx4
                    mod.add(self.max_per_data(i * self.num_load_size + j))
            mod.addSpaceLine()
            # scale operation
            for i in range(0, self.num_load_count): # unroll
                for j in range(0, self.num_load_size): # dwordx4
                    mod.add(self.scale_per_data(i * self.num_load_size + j))
            mod.addSpaceLine()
            if self.is_scale: # buffer store fp8
                for i in range(0, self.num_load_count): # unroll
                    for j in range(0, self.num_load_size): # dwordx4
                        mod.add(ri.BufferStoreB8(vgpr(f"OutputD+{i*self.num_load_size+j}"), \
                                                 vgpr(f"OffsetD+{i*self.num_load_size+j}"), \
                                                 sgpr("DstD",4), 0, MUBUFModifiers(offen=True)))
                mod.addSpaceLine()
            # adjust offset of buffer load
            # total bytes = num_workitems * num_unroll * load_size_in_bytes
            # num_unroll = num_load_count
            # load_size_in_bytes = dwordx4 = num_load_size * 4
            mod.add(ri.SMovB32(sgpr("Tmp"), self.num_workitems * self.num_load_count * self.num_load_size * 4))
            for i in range(0, self.num_load_count):
                mod.add(ri.VAddU32(vgpr(f"Offset+{i}"), vgpr(f"Offset+{i}"), sgpr("Tmp")))
            mod.addSpaceLine()
            if self.is_scale: # adjust offset of buffer store fp8
                mod.add(ri.SMovB32(sgpr("TmpD"), self.num_workitems * self.num_load_count * 4))
                for i in range(0, self.num_load_count):
                    for j in range(0, self.num_load_size):
                        mod.add(ri.VAddU32(vgpr(f"OffsetD+{i*self.num_load_size+j}"), \
                                           vgpr(f"OffsetD+{i*self.num_load_size+j}"), sgpr("TmpD")))
                mod.addSpaceLine()
        mod.addSpaceLine()
        return mod


    def sum_per_threadx4(self) -> Module:
        mod = Module("sum_per_threadx4")
        mod.addComment0("sum_per_threadx4")
        mod.add(ri.SLShiftRightB32(sgpr("MainLoop"), \
                                   int(log2(self.num_workitems * self.num_load_size * (4 // self.bpe))), \
                                   sgpr("SizeLength")))
        mod.add(ri.SAndB32(sgpr("MainLoop"), hex(self.num_load_count-1), sgpr("MainLoop")))
        with asm_loop(mod, "sum_per_threadx4", "MainLoop"):
            mod.add(ri.BufferLoadB128(vgpr("Value",4), vgpr("Offset"), sgpr("Src",4), 0, MUBUFModifiers(offen=True)))
            mod.addSpaceLine()
            mod.add(ri.SWaitCnt(vlcnt=0))
            # max operation
            for i in range(0, self.num_load_size): # dwordx4
                mod.add(self.max_per_data(i))
            mod.addSpaceLine()
            # scale operation
            for i in range(0, self.num_load_size): # dwordx4
                mod.add(self.scale_per_data(i))
            mod.addSpaceLine()
            if self.is_scale: # buffer store fp8
                for i in range(0, self.num_load_size): # dwordx4
                    mod.add(ri.BufferStoreB8(vgpr(f"OutputD+{i}"), \
                                             vgpr(f"OffsetD+{i}"), \
                                             sgpr("DstD",4), 0, MUBUFModifiers(offen=True)))
                mod.addSpaceLine()
            # adjust offset of buffer load
            mod.add(ri.SMovB32(sgpr("Tmp"), self.num_workitems * self.num_load_size * 4))
            mod.add(ri.VAddU32(vgpr("Offset"), vgpr("Offset"), sgpr("Tmp")))
            if self.is_scale: # adjust offset of buffer store fp8
                mod.add(ri.SMovB32(sgpr("TmpD"), self.num_workitems * 4))
                for i in range(0, self.num_load_size):
                    mod.add(ri.VAddU32(vgpr(f"OffsetD+{i}"), \
                                       vgpr(f"OffsetD+{i}"), sgpr("TmpD")))
                mod.addSpaceLine()
        mod.addSpaceLine()
        return mod


    def adjust_global_address(self) -> Module:
        mod = Module("adjust_global_address")
        mod.addComment0("adjust_global_address")

        # adjust buffer load offset
        #    buffer_load_dwordx4 = byte * 4 * 4
        # -) buffer_load_dword   = byte * 1 * 4
        # --------------------------------------
        #                          byte * 3 * 4
        mod.add(ri.VMulLOU32(vgpr("Tmp"), 3, vgpr("Serial")))
        mod.add(ri.VLShiftLeftB32(vgpr("Tmp"), 2, vgpr("Tmp")))
        mod.add(ri.VSubU32(vgpr("Offset"), vgpr("Offset"), vgpr("Tmp")))
        if (self.i_type.isHalf()):
            mod.add(ri.VLShiftLeftB32(vgpr("Tmp"), 1, vgpr("Serial")))
            mod.add(ri.VSubU32(vgpr("Offset"), vgpr("Offset"), vgpr("Tmp")))
        mod.addSpaceLine()

        if self.is_scale: # adjust buffer store offset
            mod.add(ri.VMulLOU32(vgpr("TmpD"), 3, vgpr("Serial")))
            mod.add(ri.VSubU32(vgpr("OffsetD"), vgpr("OffsetD"), vgpr("TmpD")))

        mod.addSpaceLine()
        mod.addSpaceLine()
        return mod


    def sum_per_thread(self) -> Module:
        offset = self.num_workitems
        mod = Module("sum_per_thread")
        mod.addComment0("sum_per_thread")
        mod.add(ri.SLShiftRightB32(sgpr("MainLoop"), int(log2(offset)), sgpr("SizeLength")))
        mod.add(ri.SAndB32(sgpr("MainLoop"), sgpr("MainLoop"), self.num_load_size * (4 // self.bpe) - 1))
        with asm_loop(mod, "sum_per_thread", "MainLoop"):
            BufferLoadx1 = self.global_read_inst_type(1)
            mod.add(BufferLoadx1(vgpr("Value"), vgpr("Offset"), sgpr("Src",4), 0, MUBUFModifiers(offen=True)))
            mod.add(ri.SWaitCnt(vlcnt=0))
            mod.addSpaceLine()
            mod.add(self.max_per_data(0, onlyOneElement=True))
            mod.addSpaceLine()
            mod.add(self.scale_per_data(0))
            mod.addSpaceLine()
            if self.is_scale:
                mod.add(ri.BufferStoreB8(vgpr("OutputD"),
                                         vgpr("OffsetD"), sgpr("DstD",4), 0, MUBUFModifiers(offen=True)))
                mod.addSpaceLine()
            mod.add(ri.SMovB32(sgpr("Tmp"), self.num_workitems * self.bpe))
            mod.add(ri.VAddU32(vgpr("Offset"), vgpr("Offset"), sgpr("Tmp")))
            mod.addSpaceLine()
            if self.is_scale:
                mod.add(ri.SMovB32(sgpr("TmpD"), self.num_workitems))
                mod.add(ri.VAddU32(vgpr("OffsetD"), vgpr("OffsetD"), sgpr("TmpD")))
                mod.addSpaceLine()
        mod.addSpaceLine()
        return mod


    def sum_in_some_thread(self)  -> Module:
        label_sum_end = Label("sum", f'loop sum end')
        mod = Module("sum_in_some_thread")
        mod.addComment0("sum_in_some_thread")
        mod.add(ri.SAndB32(sgpr("MainLoop"), sgpr("SizeLength"), self.num_workitems-1))
        mod.add(ri.VCmpLtU32(VCC(), vgpr("Serial"), sgpr("MainLoop")))
        mod.add(ri.SCBranchVCCZ(label_sum_end.getLabelName()))
        mod.add(self.SMovBX(EXEC(), VCC()))
        mod.add(ri.SNop(1))
        BufferLoadx1 = self.global_read_inst_type(1)
        mod.add(BufferLoadx1(vgpr("Value"), vgpr("Offset"), sgpr("Src",4), 0, MUBUFModifiers(offen=True)))
        mod.add(ri.SWaitCnt(vlcnt=0))
        mod.addSpaceLine()
        mod.add(self.max_per_data(0, onlyOneElement=True))
        mod.addSpaceLine()
        mod.add(self.scale_per_data(0))
        mod.addSpaceLine()
        if self.is_scale:
            mod.add(ri.BufferStoreB8(vgpr("OutputD"),
                                     vgpr("OffsetD"), sgpr("DstD",4), 0, MUBUFModifiers(offen=True)))
            mod.addSpaceLine()
        mod.add(self.SMovBX(EXEC(), "-1"))
        mod.add(ri.SNop(1))
        mod.add(label_sum_end)
        mod.addSpaceLine()
        mod.addSpaceLine()
        return mod


    def merge_sum(self) -> Module:
        mod = Module("merge_sum")
        if (self.i_type.isHalf()):
            mod.add(ri.VMaxF16(vgpr("Output"), vgpr("Output"), vgpr("OutputB")))
        elif (self.i_type.isSingle()):
            mod.add(ri.VMaxF32(vgpr("Output"), vgpr("Output"), vgpr("OutputB")))

        return mod


    def intra_wave_reduction(self) -> Module:
        label = Label("permute", "permute")
        mod = Module("intra_wave_reduction")
        mod.addComment0("intra_wave_reduction")
        mod.add(ri.SMovB32(sgpr("Tmp"), 1))
        mod.add(label)
        mod.addSpaceLine()
        mod.add(ri.VAddU32(vgpr("Tmp"), sgpr("Tmp"), vgpr("Serial")))
        mod.add(ri.VAndB32(vgpr("Tmp"), self.wavefront_size-1, vgpr("Tmp")))
        mod.add(ri.VLShiftLeftB32(vgpr("Tmp"), 0x2, vgpr("Tmp")))
        mod.addSpaceLine()
        mod.add(ri.DSBPermuteB32(vgpr("OutputB"), vgpr("Tmp"), vgpr("Output")))
        mod.add(ri.SWaitCnt(dscnt=0))
        mod.addSpaceLine()
        mod.add(self.merge_sum())
        mod.add(ri.SLShiftLeftB32(sgpr("Tmp"), 1, sgpr("Tmp")))
        mod.add(ri.SCmpLtU32(sgpr("Tmp"), self.wavefront_size))
        mod.add(ri.SCBranchSCC1(label.getLabelName()))
        mod.addSpaceLine()
        return mod


    def inter_wave_reduction(self) -> Module:
        label_inter = Label("inter", f'inter')
        label_upper = Label("upper", f'upper')
        label_lower = Label("lower", f'lower')
        label_empty = Label("empty", f'empty')
        label_end   = Label("end", f'end')
        mod = Module("inter_wave_reduction")
        mod.addComment0("inter_wave_reduction")
        mod.add(ri.VLShiftRightB32(vgpr("Widx"), int(log2(self.wavefront_size)), vgpr("Serial")))
        mod.add(ri.SMovB32(sgpr("Offset"), self.num_workitems // self.wavefront_size))
        mod.add(label_inter)
        mod.add(ri.SLShiftRightB32(sgpr("Offset"), 1, sgpr("Offset")))
        mod.add(ri.SCmpEQU32(sgpr("Offset"), 0))
        mod.add(ri.SCBranchSCC1(label_end.getLabelName()))
        mod.add(ri.SLShiftLeftB32(sgpr("Tmp"), 1, sgpr("Offset")))
        mod.add(ri.VCmpLtU32(sgpr("Tmp+2",self.laneSGPRCount), vgpr("Widx"), sgpr("Tmp")))
        mod.add(ri.VCmpGEU32(sgpr("Tmp+4",self.laneSGPRCount), vgpr("Widx"), sgpr("Offset")))
        mod.add(self.SAndBX(VCC(), sgpr("Tmp+2",self.laneSGPRCount), sgpr("Tmp+4",self.laneSGPRCount)))
        mod.add(ri.SCBranchVCCNZ(label_upper.getLabelName()))
        mod.add(ri.VCmpLtU32(VCC(), vgpr("Widx"), sgpr("Offset")))
        mod.add(ri.SCBranchVCCNZ(label_lower.getLabelName()))
        mod.add(ri.SBranch(label_empty.getLabelName()))

        mod.add(label_upper)
        mod.add(ri.VSubU32(vgpr("Tmp"), vgpr("Widx"), sgpr("Offset")))
        mod.add(ri.VLShiftLeftB32(vgpr("Tmp"), int(log2(self.bpe)), vgpr("Tmp")))
        ds = DSModifiers(offset=0)
        DSStorex1 = self.local_write_inst_type(1)
        mod.add(DSStorex1(vgpr("Tmp"), vgpr("Output"), ds))
        mod.add(ri.SWaitCnt(dscnt=0))
        mod.add(ri.SBarrier())
        mod.add(ri.SBranch(label_inter.getLabelName()))
        mod.add(label_lower)
        mod.add(ri.SBarrier())
        mod.add(ri.VLShiftLeftB32(vgpr("Tmp"), int(log2(self.bpe)), vgpr("Widx")))
        ds = DSModifiers(offset=0)
        DSLoadx1 = self.local_read_inst_type(1)
        mod.add(DSLoadx1(vgpr("OutputB"), vgpr("Tmp"), ds))
        mod.add(ri.SWaitCnt(dscnt=0))
        mod.add(self.merge_sum())
        mod.add(ri.SBranch(label_inter.getLabelName()))
        mod.add(label_empty)
        mod.add(ri.SBarrier())
        mod.add(ri.SBranch(label_inter.getLabelName()))
        mod.add(label_end)
        mod.addSpaceLine()
        return mod


    def broadcast(self) -> Module:
        label_lower = Label("broadcast_lower", f'broadcast_lower')
        label_end = Label("broadcast_end", f'broadcast_end')

        mod = Module("broadcast")
        mod.addComment0("broadcast")
        mod.add(ri.VCmpEQU32(VCC(), vgpr("Widx"), 0))
        mod.add(ri.SCBranchVCCZ(label_lower.getLabelName()))
        ds = DSModifiers(offset=0)
        DSStorex1 = self.local_write_inst_type(1)
        mod.add(DSStorex1(vgpr("Widx"), vgpr("Output"), ds))
        mod.add(ri.SWaitCnt(dscnt=0))
        mod.add(ri.SBarrier())
        mod.add(ri.SBranch(label_end.getLabelName()))
        mod.add(label_lower)
        mod.add(ri.SBarrier())
        mod.add(ri.VMovB32(vgpr("Tmp"), 0))
        ds = DSModifiers(offset=0)
        DSLoadx1 = self.local_read_inst_type(1)
        mod.add(DSLoadx1(vgpr("Output"), vgpr("Tmp"), ds))
        mod.add(ri.SWaitCnt(dscnt=0))
        mod.add(label_end)
        mod.addSpaceLine()
        return mod


    def output_result(self) -> Module:
        mod = Module("output_result")
        mod.addComment0("output_result")
        BufferStorex1 = self.global_write_inst_type(1)

        mod.add(ri.VMovB32(vgpr("Offset"), 0))
        if self.i_type.toChar() == 'H' and self.o_type.toChar() == "S":
            mod.add(ri.VCvtF16toF32(vgpr("Output"), vgpr("Output")))
        elif self.i_type.toChar() == 'S' and self.o_type.toChar() == "H":
            mod.add(ri.VCvtF32toF16(vgpr("Output"), vgpr("Output")))
        mod.add(BufferStorex1(vgpr("Output"), vgpr("Offset"), sgpr("Dst",4), 0, MUBUFModifiers(offen=True)))
        mod.addSpaceLine()

        return mod

    def amax_kernel_body(self) -> Module:
        mod = Module(self.func_name)
        mod.add(self.defineVariables())
        with asm_func(self.func_name, mod):
            mod.add(self.load_kernel_args())
            mod.add(self.init_param())
            mod.add(self.calculate_global_address())
            mod.add(self.sum_per_threadxN())
            mod.add(self.sum_per_threadx4())
            mod.add(self.adjust_global_address())
            mod.add(self.sum_per_thread())
            mod.add(self.sum_in_some_thread())
            mod.add(self.intra_wave_reduction())
            mod.add(self.inter_wave_reduction())
            mod.add(self.broadcast())
            mod.add(self.output_result())
        return mod

@dataclass
class KernelArgument:
    size: int
    offset: int
    value_kind: str
    address_space: Optional[str] = None

    def to_dict(self):
        d = {'.size': self.size, '.offset': self.offset,
             '.value_kind': self.value_kind}

        if self.address_space:
            d['.address_space'] = self.address_space

        return d

@dataclass
class KernelMeta:
    name: str
    num_vgpr: int
    num_sgpr: int
    num_agpr: int
    num_lds_bytes: int
    wavefront_size: int
    max_workgroup_size: int
    args_alignment: int
    args: List[KernelArgument]

    def update_args_offsets(self):
        offset = 0
        for arg in args:
            arg.offset = offset
            offset += arg.size

    def _get_args_size(self):
        total_size = sum(arg.size for arg in self.args)
        total_size += (self.args_alignment - (total_size % self.args_alignment))
        return total_size

    def to_dict(self):
        return {
            '.name': self.name,
            '.symbol': f'{self.name}.kd',
            '.kernarg_segment_size': self._get_args_size(),
            '.group_segment_fixed_size': self.num_lds_bytes,
            '.private_segment_fixed_size': 0,
            '.kernarg_segment_align': self.args_alignment,
            '.wavefront_size': self.wavefront_size,
            '.sgpr_count': self.num_sgpr,
            '.vgpr_count': self.num_vgpr,
            '.agpr_count': self.num_agpr,
            '.max_flat_workgroup_size': self.max_workgroup_size,
            '.args': [arg.to_dict() for arg in self.args]
        }

def meta_str(kernels: Tuple[KernelMeta]):
    beg = '.amdgpu_metadata\n---'
    content_str = yaml.dump({'amdhsa.version': [1, 1], 'amdhsa.kernels': [kernel.to_dict() for kernel in kernels]})
    end = '.end_amdgpu_metadata'
    return '\n'.join([beg, content_str, end, ''])


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('-o', '--output', type=str, required=True, help='Output path of compiled binary')
    ap.add_argument('-t', type=str, default="S", help='data type')
    ap.add_argument('-d', type=str, default="None", help='dest data type')
    ap.add_argument('-s', type=str, default="F8N", help='scale data type')
    ap.add_argument('-w', type=int, default=256, help='workitem')
    ap.add_argument('-c', type=int, default=4, help='load conut per iteration')
    ap.add_argument('--toolchain', type=str, default=ToolchainDefaults.CXX_COMPILER, help='Path to ROCm compiler')
    ap.add_argument('--debug-build', action='store_true', dest='debug_build', help='Build with debug information')
    ap.add_argument('--is-scale', action='store_true', dest='is_scale', help='Enable scaled output or not')
    ap.add_argument('--arch', type=str, default='gfx90a', help='Target architecture for assembler, e.g. gfx908. Default is gfx90a')
    ap.set_defaults(debug_build=False)

    args = ap.parse_args()
    output_path: str = args.output
    t: str = args.t
    d: str = t if (args.d =="None") else args.d
    s: str = args.s
    w: int = args.w
    c: int = args.c
    toolchain_path: str = validateToolchain(args.toolchain)
    debug_build: bool = args.debug_build
    arch: str = args.arch
    is_scale: bool = args.is_scale
    isa = gfxToIsa(arch)

    if any([not i for i in (arch, toolchain_path, isa)]):
        restoreDefaultGlobalParameters()
        assignGlobalParameters({})
        enumerator = validateToolchain(ToolchainDefaults.DEVICE_ENUMERATOR)
        isa = detectGlobalCurrentISA(0, enumerator)
        arch = isaToGfx(isa)
        toolchain_path = validateToolchain(ToolchainDefaults.CXX_COMPILER)

    _global_ti.init(isa, toolchain_path, False)
    waveFrontSize = 32 if isa[0] in [11, 12] else 64
    _global_ti.setKernel(isa, waveFrontSize)
    amax = AMaxKernelGenerator(DataType(t), DataType(d), DataType(s), w, c, 4, waveFrontSize, arch, isa, is_scale)
    kernel_body = amax.amax_kernel_body()
    args = amax.kernel_args()
    func_name = amax.func_name
    meta = KernelMeta(func_name, amax.vgpr_pool.size(), amax.sgpr_pool.size(), 0, amax.lds_usage_byte, waveFrontSize, w, 8, args)
    meta.update_args_offsets()
    k_str = '\n'.join([kernel_header(func_name, arch, amax.vgpr_pool.size(), amax.sgpr_pool.size(), amax.lds_usage_byte),
                       meta_str((meta,)),
                       str(kernel_body)])

    with open(output_path, 'w') as f:
        f.write(k_str)

    output_path_basename = os.path.splitext(output_path)[0]
    amax.dump('yaml', f'{output_path_basename}.yaml')
