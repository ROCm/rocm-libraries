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
from Tensile.Common.GlobalParameters import assignGlobalParameters, restoreDefaultGlobalParameters
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
def asm_loop(mod: Module, name: str, it: str, sweep_once: int):
    try:
        if not sweep_once:
            loop_start_label = Label(name, f'loop {name} starts')
            loop_end_label = Label(f'{name}_end', f'loop {name} ends')
            mod.add(loop_start_label)
            mod.add(ri.SCmpEQU32(sgpr(it), 0))
            mod.add(ri.SCBranchSCC1(loop_end_label.getLabelName()))
            mod.addSpaceLine()
        yield
    finally:
        if not sweep_once:
            mod.add(ri.SSubU32(sgpr(it), sgpr(it), 1))
            mod.add(ri.SBranch(loop_start_label.getLabelName()))
            mod.add(loop_end_label)
            mod.addSpaceLine()


class LayerNormKernelGenerator:
    srd_num_reg = 4
    srd_alignment = 4

    def __init__(self,
                 io_type: DataType,
                 num_workitems: int,
                 num_load_count: int,
                 num_load_size: int,
                 sweep_once: int,
                 wavefront_size: int,
                 arch: str,
                 isa: IsaVersion):
        self.io_type = io_type
        self.bpe = io_type.numBytes()
        self.num_workitems = num_workitems
        self.num_load_count = num_load_count
        self.num_load_size = num_load_size
        self.sweep_once = sweep_once
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
        self.op = 'LayerNorm'
        self.sgprs  = collections.OrderedDict()
        self.vgprs  = collections.OrderedDict()

    @property
    def lds_usage_byte(self) -> int:
        # used in reduce inter wave mean and invvar
        # 4 data * half_wave_num * bpe
        return 4 * (self.num_workitems // self.wavefront_size // 2) * self.bpe

    @property
    def func_name(self):
        return f'LayerNorm_DT_{self.io_type}_W_{self.num_workitems}_C_{self.num_load_count}_S_{self.sweep_once}'

    def dumps(self, format: str) -> str:
        limit = self.num_workitems * self.num_load_count * self.num_load_size if sweep_once else 99999999999

        param_dict = {
            'arch': self.arch,
            'op': self.op,
            'func_name': self.func_name,
            'io_type': self.io_type.toChar(),
            'num_workitems': self.num_workitems,
            'limit': limit
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
        return (KernelArgument(8,  0, 'global_buffer', 'global'),
                KernelArgument(8,  8, 'global_buffer', 'global'),
                KernelArgument(8, 16, 'global_buffer', 'global'),
                KernelArgument(8, 24, 'global_buffer', 'global'),
                KernelArgument(8, 32, 'global_buffer', 'global'),
                KernelArgument(8, 40, 'global_buffer', 'global'),
                KernelArgument(4, 48, 'by_value'),
                KernelArgument(4, 52, 'by_value'),
                KernelArgument(4, 56, 'by_value'))


    def defineVariables(self):
        self.defineVgpr("Serial", 1, 1)
        self.defineVgpr("Src",    2, 2)
        self.defineVgpr("Dst",    2, 2)
        self.defineVgpr("Count",  1, 1)
        self.defineVgpr("Mean",   1, 1)
        self.defineVgpr("Invvar",    1, 1)
        self.defineVgpr("CountA", 1, 1)
        self.defineVgpr("MeanA",  1, 1)
        self.defineVgpr("StdA",   1, 1)
        self.defineVgpr("CountB", 1, 1)
        self.defineVgpr("MeanB",  1, 1)
        self.defineVgpr("StdB",   1, 1)
        self.defineVgpr("Widx",   1, 1)
        self.defineVgpr("Index",  1, 1)
        self.defineVgpr("Offset", 4, 1)
        self.defineVgpr("Value",  self.num_load_count * self.num_load_size, self.num_load_size)
        self.defineVgpr("Gamma",  self.num_load_count * self.num_load_size, self.num_load_size)
        self.defineVgpr("Beta",   self.num_load_count * self.num_load_size, self.num_load_size)
        self.defineVgpr("Tmp",    4, 1)

        self.defineSgpr("KernelArg", 2)
        self.defineSgpr("WorkGroup0", 1)
        self.defineSgpr("WorkGroup1", 1)
        self.defineSgpr("WorkGroup2", 1)
        self.defineSgpr("AddressOut", 2, 2)
        self.defineSgpr("AddressMean", 2, 2)
        self.defineSgpr("AddressInvvar", 2, 2)
        self.defineSgpr("AddressIn", 2, 2)
        self.defineSgpr("AddressGamma", 2, 2)
        self.defineSgpr("AddressBeta", 2, 2)
        self.defineSgpr("SizeLength", 1)
        self.defineSgpr("Eps", 1)
        self.defineSgpr("MainLoop", 1)
        self.defineSgpr("Offset", 1)
        self.defineSgpr("Src", 4, 4)
        self.defineSgpr("Dst", 4, 4)
        self.defineSgpr("SrcGamma", 4, 4)
        self.defineSgpr("SrcBeta", 4, 4)
        self.defineSgpr("Tmp", 6, 2)

        mod = Module("defineVariables")

        for vkey in self.vgprs:
            mod.add(RegSet("v", "vgpr"+vkey, self.vgprs[vkey]))
        mod.addSpaceLine()

        for skey in self.sgprs:
            mod.add(RegSet("s", "sgpr"+skey, self.sgprs[skey]))
        mod.addSpaceLine()

        mod.add(ValueSet("Srd127_96", self.srdUpperValue.getValue(), format=1))
        mod.add(ValueSet("BufferLimit", "0xffffffff", format=-1))
        mod.addSpaceLine()

        return mod


    def load_kernel_args(self):
        mod = Module('Load kernel args')
        mod.addComment0('Load kernel args')
        mod.add(ri.SLoadB64(sgpr("AddressOut", 2),    sgpr("KernelArg", 2),  0))
        mod.add(ri.SLoadB64(sgpr("AddressMean", 2),   sgpr("KernelArg", 2),  8))
        mod.add(ri.SLoadB64(sgpr("AddressInvvar", 2), sgpr("KernelArg", 2), 16))
        mod.add(ri.SLoadB64(sgpr("AddressIn", 2),     sgpr("KernelArg", 2), 24))
        mod.add(ri.SLoadB64(sgpr("AddressGamma", 2),  sgpr("KernelArg", 2), 32))
        mod.add(ri.SLoadB64(sgpr("AddressBeta", 2),   sgpr("KernelArg", 2), 40))
        mod.add(ri.SLoadB32(sgpr("SizeLength"),       sgpr("KernelArg", 2), 52))
        mod.add(ri.SLoadB32(sgpr("Eps"),              sgpr("KernelArg", 2), 56))
        mod.add(ri.SWaitCnt(kmcnt=0))
        if _global_ti.getArchCaps()["WorkGroupIdFromTTM"]:
            mod.add(ri.SAndB32(dst=sgpr("WorkGroup1"), src0=hex(0xFFFF), src1="ttmp7"))
        mod.addSpaceLine()
        return mod

    def init_param(self) -> Module:
        mod = Module("defineVariables")
        mod.addComment0("defineVariables")
        mod.add(ri.SMovB32(sgpr("Src+0"), sgpr("AddressIn+0")))
        mod.add(ri.SMovB32(sgpr("Src+1"), sgpr("AddressIn+1")))
        mod.add(ri.SMovB32(sgpr("Src+3"), "Srd127_96"))
        mod.addSpaceLine()

        mod.add(ri.SMovB32(sgpr("SrcGamma+0"), sgpr("AddressGamma+0")))
        mod.add(ri.SMovB32(sgpr("SrcGamma+1"), sgpr("AddressGamma+1")))
        mod.add(ri.SMovB32(sgpr("SrcGamma+3"), "Srd127_96"))
        mod.addSpaceLine()

        mod.add(ri.SMovB32(sgpr("SrcBeta+0"), sgpr("AddressBeta+0")))
        mod.add(ri.SMovB32(sgpr("SrcBeta+1"), sgpr("AddressBeta+1")))
        mod.add(ri.SMovB32(sgpr("SrcBeta+3"), "Srd127_96"))
        mod.addSpaceLine()

        mod.add(ri.SMovB32(sgpr("Dst+0"), sgpr("AddressOut+0")))
        mod.add(ri.SMovB32(sgpr("Dst+1"), sgpr("AddressOut+1")))
        mod.add(ri.SMovB32(sgpr("Dst+3"), "Srd127_96"))
        mod.addSpaceLine()

        mod.add(ri.SMulI32(sgpr("Tmp"), sgpr("WorkGroup1"), sgpr("SizeLength")))
        mod.add(ri.SLShiftLeftB32(sgpr("Tmp"), 2, sgpr("Tmp")))
        mod.add(ri.SAddU32(sgpr("Src+0"), sgpr("Src+0"), sgpr("Tmp")))
        mod.add(ri.SAddCU32(sgpr("Src+1"), sgpr("Src+1"), 0))
        mod.add(ri.SAddU32(sgpr("Dst+0"), sgpr("Dst+0"), sgpr("Tmp")))
        mod.add(ri.SAddCU32(sgpr("Dst+1"), sgpr("Dst+1"), 0))
        mod.addSpaceLine()

        mod.add(ri.SLShiftLeftB32(sgpr("Tmp"), 2, sgpr("SizeLength")))
        mod.add(ri.SMovB32(sgpr("Src+2"), sgpr("Tmp")))
        mod.add(ri.SMovB32(sgpr("SrcGamma+2"), sgpr("Tmp")))
        mod.add(ri.SMovB32(sgpr("SrcBeta+2"), sgpr("Tmp")))
        mod.add(ri.SMovB32(sgpr("Dst+2"), sgpr("Tmp")))
        mod.addSpaceLine()

        mod.add(ri.VMovB32(vgpr("Count"), 0.0))
        mod.add(ri.VMovB32(vgpr("Mean"), 0.0))
        mod.add(ri.VMovB32(vgpr("Invvar"), 0.0))
        mod.addSpaceLine()
        return mod


    def calculate_global_address(self) -> Module:

        mod = Module("calculate_global_address")
        mod.addComment0("calculate_global_address")


        mod.add(ri.VLShiftLeftB32(vgpr("Offset+0"), hex(int(log2(self.num_load_size * self.bpe))), vgpr("Serial")))
        mod.addSpaceLine()

        offset = self.num_workitems * self.num_load_size
        mod.add(ri.SMovB32(sgpr("Tmp"), self.num_workitems * self.num_load_size * self.bpe))
        for i in range(0, self.num_load_count-1):
            mod.add(ri.VAddU32(vgpr(f"Offset+{i+1}"), vgpr(f"Offset+{i}"), sgpr("Tmp")))
        mod.addSpaceLine()
        return mod


    def sum_per_data(self, val) -> Module:
        mod = Module("sum_per_data")
        if self.sweep_once:
            mod.add(ri.VCmpLtU32(VCC(), vgpr("Index"), sgpr("SizeLength")))
            #mod.add(ri.SCBranchddVCCZ(label_sum_end.getLabelName()))
            mod.add(self.SMovBX(EXEC(), VCC()))
            mod.add(ri.SNop(1))
        mod.add(ri.VAddF32(vgpr("Count"), vgpr("Count"), 1.0))
        mod.add(ri.VSubF32(vgpr("Tmp"), val, vgpr("Mean")))  # delta
        mod.add(ri.VRcpF32(vgpr("Tmp+1"), vgpr("Count"))) # 1 / count
        mod.add(ri.SNop(waitState=0, comment="1 wait states"))
        mod.add(ri.VMulF32(vgpr("Tmp+1"), vgpr("Tmp"), vgpr("Tmp+1"))) # delta / count
        mod.add(ri.VAddF32(vgpr("Mean"), vgpr("Mean"), vgpr("Tmp+1"))) # new mean
        mod.add(ri.VSubF32(vgpr("Tmp+1"), val, vgpr("Mean")))
        mod.add(ri.VMulF32(vgpr("Tmp"), vgpr("Tmp"), vgpr("Tmp+1")))
        mod.add(ri.VAddF32(vgpr("Invvar"), vgpr("Invvar"), vgpr("Tmp")))
        if self.sweep_once:
            mod.add(self.SMovBX(EXEC(), "-1"))
            mod.add(ri.SNop(1))
            mod.add(ri.VAddU32(vgpr("Index"), vgpr("Index"), 1))
        mod.addSpaceLine()
        return mod


    def sum_per_threadxN(self) -> Module:
        offset = self.num_workitems * self.num_load_count * self.num_load_size
        mod = Module("sum_per_threadxN")
        mod.addComment0("sum_per_threadxN")
        if not self.sweep_once:
            mod.add(ri.SLShiftRightB32(sgpr("MainLoop"), int(log2(offset)), sgpr("SizeLength")))
        with asm_loop(mod, "sum_per_threadxN", "MainLoop", self.sweep_once):
            for i in range(0, self.num_load_count):
                mod.add(ri.BufferLoadB128(vgpr(f"Value+{i*self.num_load_size}",4), vgpr(f"Offset+{i}"), sgpr("Src",4), 0, MUBUFModifiers(offen=True)))
            mod.addSpaceLine()
            for i in range(0, self.num_load_count):
                mod.add(ri.SWaitCnt(vlcnt=(self.num_load_count-i-1)))
                if self.sweep_once:
                    mod.add(ri.SMovB32(sgpr("Tmp"), i * self.num_workitems * self.num_load_size))
                    mod.add(ri.VLShiftLeftB32(vgpr("Index+0"), hex(int(log2(self.num_load_size))), vgpr("Serial")))
                    mod.add(ri.VAddU32(vgpr("Index+0"), vgpr("Index+0"), sgpr("Tmp")))
                    mod.addSpaceLine()
                mod.add(self.sum_per_data(vgpr(f"Value+{i * self.num_load_size + 0}")))
                mod.add(self.sum_per_data(vgpr(f"Value+{i * self.num_load_size + 1}")))
                mod.add(self.sum_per_data(vgpr(f"Value+{i * self.num_load_size + 2}")))
                mod.add(self.sum_per_data(vgpr(f"Value+{i * self.num_load_size + 3}")))
            if not self.sweep_once:
                mod.add(ri.SMovB32(sgpr("Tmp"), offset * self.bpe))
                for i in range(0, self.num_load_count):
                    mod.add(ri.VAddU32(vgpr(f"Offset+{i}"), vgpr(f"Offset+{i}"), sgpr("Tmp")))
                mod.addSpaceLine()
        mod.addSpaceLine()
        return mod


    def sum_per_threadx4(self) -> Module:
        offset = self.num_workitems * self.num_load_size
        mod = Module("sum_per_threadx4")
        mod.addComment0("sum_per_threadx4")
        mod.add(ri.SLShiftRightB32(sgpr("MainLoop"), int(log2(offset)), sgpr("SizeLength")))
        mod.add(ri.SAndB32(sgpr("MainLoop"), hex(self.num_load_count-1), sgpr("MainLoop")))
        with asm_loop(mod, "sum_per_threadx4", "MainLoop", self.sweep_once):
            mod.add(ri.BufferLoadB128(vgpr("Value",4), vgpr("Offset"), sgpr("Src",4), 0, MUBUFModifiers(offen=True)))
            mod.addSpaceLine()
            mod.add(ri.SWaitCnt(vlcnt=0))
            mod.add(self.sum_per_data(vgpr("Value+0")))
            mod.add(self.sum_per_data(vgpr("Value+1")))
            mod.add(self.sum_per_data(vgpr("Value+2")))
            mod.add(self.sum_per_data(vgpr("Value+3")))
            mod.add(ri.SMovB32(sgpr("Tmp"), self.num_workitems * self.num_load_size * self.bpe))
            mod.add(ri.VAddU32(vgpr("Offset"), vgpr("Offset"), sgpr("Tmp")))
            mod.addSpaceLine()
        return mod


    def adjusst_global_address(self) -> Module:
        mod = Module("adjusst_global_address")
        mod.add(ri.VMulLOU32(vgpr("Tmp"), 3, vgpr("Serial")))
        mod.add(ri.VLShiftLeftB32(vgpr("Tmp"), 2, vgpr("Tmp")))
        mod.add(ri.VSubU32(vgpr("Offset"), vgpr("Offset"), vgpr("Tmp")))
        mod.addSpaceLine()
        return mod


    def sum_per_thread(self) -> Module:
        offset = self.num_workitems
        mod = Module("sum_per_thread")
        mod.addComment0("sum_per_thread")
        mod.add(ri.SLShiftRightB32(sgpr("MainLoop"), int(log2(offset)), sgpr("SizeLength")))
        mod.add(ri.SAndB32(sgpr("MainLoop"), sgpr("MainLoop"), self.num_load_size-1))
        with asm_loop(mod, "sum_per_thread", "MainLoop", self.sweep_once):
            mod.add(ri.BufferLoadB32(vgpr("Value"), vgpr("Offset"), sgpr("Src",4), 0, MUBUFModifiers(offen=True)))
            mod.add(ri.SWaitCnt(vlcnt=0))
            mod.addSpaceLine()
            mod.add(self.sum_per_data(vgpr("Value")))
            mod.add(ri.SMovB32(sgpr("Tmp"), self.num_workitems * self.bpe))
            mod.add(ri.VAddU32(vgpr("Offset"), vgpr("Offset"), sgpr("Tmp")))
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
        mod.add(ri.BufferLoadB32(vgpr("Value"), vgpr("Offset"), sgpr("Src",4), 0, MUBUFModifiers(offen=True)))
        mod.add(ri.SWaitCnt(vlcnt=0))
        mod.addSpaceLine()
        mod.add(self.sum_per_data(vgpr("Value")))
        mod.add(self.SMovBX(EXEC(), "-1"))
        mod.add(ri.SNop(1))
        mod.add(label_sum_end)
        mod.addSpaceLine()
        return mod


    def merge_sum(self) -> Module:
        mod = Module("merge_sum")
        mod.add(ri.VMovB32(vgpr("CountA"), vgpr("Count")))
        mod.add(ri.VMovB32(vgpr("MeanA"), vgpr("Mean")))
        mod.add(ri.VMovB32(vgpr("StdA"), vgpr("Invvar")))

        mod.add(ri.VSubF32(vgpr("Tmp"), vgpr("MeanB"), vgpr("MeanA")))
        mod.add(ri.VAddF32(vgpr("Count"), vgpr("CountA"), vgpr("CountB")))
        mod.add(ri.VCmpGTF32(VCC(), vgpr("Count"), 0))
        mod.add(self.SMovBX(EXEC(), VCC()))
        mod.add(ri.SNop(1))
        mod.add(ri.VRcpF32(vgpr("Tmp+3"), vgpr("Count")))
        mod.add(ri.SNop(waitState=0, comment="1 wait states"))
        mod.add(ri.VMulF32(vgpr("Tmp+1"), vgpr("CountA"), vgpr("Tmp+3")))
        mod.add(ri.VMulF32(vgpr("Tmp+2"), vgpr("CountB"), vgpr("Tmp+3")))
        mod.add(ri.VMulF32(vgpr("MeanA"), vgpr("MeanA"), vgpr("Tmp+1")))
        mod.add(ri.VMulF32(vgpr("MeanB"), vgpr("MeanB"), vgpr("Tmp+2")))
        mod.add(ri.VAddF32(vgpr("Mean"), vgpr("MeanA"), vgpr("MeanB")))

        mod.add(ri.VAddF32(vgpr("Invvar"), vgpr("StdA"), vgpr("StdB")))
        mod.add(ri.VMulF32(vgpr("Tmp"), vgpr("Tmp"), vgpr("Tmp")))
        mod.add(ri.VMulF32(vgpr("Tmp"), vgpr("Tmp"), vgpr("Tmp+1")))
        mod.add(ri.VMulF32(vgpr("Tmp"), vgpr("Tmp"), vgpr("Tmp+2")))
        mod.add(ri.VMulF32(vgpr("Tmp"), vgpr("Tmp"), vgpr("Count")))
        mod.add(ri.VAddF32(vgpr("Invvar"), vgpr("Invvar"), vgpr("Tmp")))
        mod.add(self.SMovBX(EXEC(), "-1"))
        mod.add(ri.SNop(1))
        return mod


    def intra_wave_reduction(self) -> Module:
        label = Label("permute", f'permuge')
        mod = Module("intra_wave_reduction")
        mod.addComment0("intra_wave_reduction")
        mod.add(ri.SMovB32(sgpr("Tmp"), 1))
        mod.add(label)
        mod.addSpaceLine()
        mod.add(ri.VAddU32(vgpr("Tmp"), sgpr("Tmp"), vgpr("Serial")))
        mod.add(ri.VAndB32(vgpr("Tmp"), self.wavefront_size-1, vgpr("Tmp")))
        mod.add(ri.VLShiftLeftB32(vgpr("Tmp"), 0x2, vgpr("Tmp")))
        mod.addSpaceLine()
        mod.add(ri.DSBPermuteB32(vgpr("CountB"), vgpr("Tmp"), vgpr("Count")))
        mod.add(ri.DSBPermuteB32(vgpr("MeanB"), vgpr("Tmp"), vgpr("Mean")))
        mod.add(ri.DSBPermuteB32(vgpr("StdB"), vgpr("Tmp"), vgpr("Invvar")))
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
        mod.add(ri.VCmpLtU32(sgpr("Tmp+2", self.laneSGPRCount), vgpr("Widx"), sgpr("Tmp")))
        mod.add(ri.VCmpGEU32(sgpr("Tmp+4", self.laneSGPRCount), vgpr("Widx"), sgpr("Offset")))
        mod.add(self.SAndBX(VCC(), sgpr("Tmp+2", self.laneSGPRCount), sgpr("Tmp+4", self.laneSGPRCount)))
        mod.add(ri.SCBranchVCCNZ(label_upper.getLabelName()))
        mod.add(ri.VCmpLtU32(VCC(), vgpr("Widx"), sgpr("Offset")))
        mod.add(ri.SCBranchVCCNZ(label_lower.getLabelName()))
        mod.add(ri.SBranch(label_empty.getLabelName()))

        mod.add(label_upper)
        mod.add(ri.VSubU32(vgpr("Tmp"), vgpr("Widx"), sgpr("Offset")))
        mod.add(ri.VMulLOU32(vgpr("Tmp"), vgpr("Tmp"), 4))
        mod.add(ri.VLShiftLeftB32(vgpr("Tmp"), 2, vgpr("Tmp")))
        ds = DSModifiers(offset=0)
        mod.add(ri.DSStoreB32(vgpr("Tmp"), vgpr("Count"), ds))
        ds = DSModifiers(offset=4)
        mod.add(ri.DSStoreB32(vgpr("Tmp"), vgpr("Mean"), ds))
        ds = DSModifiers(offset=8)
        mod.add(ri.DSStoreB32(vgpr("Tmp"), vgpr("Invvar"), ds))
        mod.add(ri.SWaitCnt(dscnt=0))
        mod.add(ri.SBarrier())
        mod.add(ri.SBranch(label_inter.getLabelName()))
        mod.add(label_lower)
        mod.add(ri.SBarrier())
        mod.add(ri.VMulLOU32(vgpr("Tmp"), vgpr("Widx"), 4))
        mod.add(ri.VLShiftLeftB32(vgpr("Tmp"), 2, vgpr("Tmp")))
        ds = DSModifiers(offset=0)
        mod.add(ri.DSLoadB32(vgpr("CountB"), vgpr("Tmp"), ds))
        ds = DSModifiers(offset=4)
        mod.add(ri.DSLoadB32(vgpr("MeanB"), vgpr("Tmp"), ds))
        ds = DSModifiers(offset=8)
        mod.add(ri.DSLoadB32(vgpr("StdB"), vgpr("Tmp"), ds))
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
        mod.add(ri.DSStoreB32(vgpr("Widx"), vgpr("Count"), ds))
        ds = DSModifiers(offset=4)
        mod.add(ri.DSStoreB32(vgpr("Widx"), vgpr("Mean"), ds))
        ds = DSModifiers(offset=8)
        mod.add(ri.DSStoreB32(vgpr("Widx"), vgpr("Invvar"), ds))
        mod.add(ri.SWaitCnt(dscnt=0))
        mod.add(ri.SBarrier())
        mod.add(ri.SBranch(label_end.getLabelName()))
        mod.add(label_lower)
        mod.add(ri.SBarrier())
        mod.add(ri.VMovB32(vgpr("Tmp"), 0))
        ds = DSModifiers(offset=0)
        mod.add(ri.DSLoadB32(vgpr("Count"), vgpr("Tmp"), ds))
        ds = DSModifiers(offset=4)
        mod.add(ri.DSLoadB32(vgpr("Mean"), vgpr("Tmp"), ds))
        ds = DSModifiers(offset=8)
        mod.add(ri.DSLoadB32(vgpr("Invvar"), vgpr("Tmp"), ds))
        mod.add(ri.SWaitCnt(dscnt=0))
        mod.add(label_end)
        mod.addSpaceLine()
        return mod


    def get_average(self) -> Module:
        mod = Module("get_average")
        mod.addComment0("get_average")
        mod.add(ri.VCvtI32toF32(vgpr("Tmp"), sgpr("SizeLength")))
        mod.add(ri.VRcpF32(vgpr("Tmp"),vgpr("Tmp")))
        mod.add(ri.SNop(waitState=0, comment="1 wait states"))
        mod.add(ri.VMulF32(vgpr("Invvar"), vgpr("Tmp"), vgpr("Invvar")))

        mod.add(ri.VAddF32(vgpr("Invvar"), vgpr("Invvar"), sgpr("Eps")))
        mod.add(ri.VRsqF32(vgpr("Invvar"), vgpr("Invvar")))
        mod.add(ri.SNop(waitState=0, comment="1 wait states"))
        mod.addSpaceLine()
        return mod


    def layernorm_cal(self, val) -> Module:
        mod = Module("layernorm_cal")
        mod.add(ri.VSubF32(val, val, vgpr("Mean")))
        mod.add(ri.VMulF32(val, val, vgpr("Invvar")))
        return mod


    def layernorm_threadxN(self) -> Module:
        offset = self.num_workitems * self.num_load_count * self.num_load_size
        mod = Module("layernorm_threadxN")
        mod.addComment0("layernorm_threadxN")
        if not self.sweep_once:
            mod.add(ri.SLShiftRightB32(sgpr("MainLoop"), int(log2(offset)), sgpr("SizeLength")))
        with asm_loop(mod, "layernorm_threadxN", "MainLoop", self.sweep_once):
            if not self.sweep_once:
                for i in range(0, self.num_load_count):
                    mod.add(ri.BufferLoadB128(vgpr(f"Value+{i * self.num_load_size}",4), vgpr(f"Offset+{i}"), sgpr("Src",4), 0, MUBUFModifiers(offen=True)))
            mod.addSpaceLine()
            for i in range(0, self.num_load_count):
                mod.add(ri.SWaitCnt(vlcnt=self.num_load_count-i-1))
                mod.add(self.layernorm_cal(vgpr(f"Value+{i * self.num_load_size + 0}")))
                mod.add(self.layernorm_cal(vgpr(f"Value+{i * self.num_load_size + 1}")))
                mod.add(self.layernorm_cal(vgpr(f"Value+{i * self.num_load_size + 2}")))
                mod.add(self.layernorm_cal(vgpr(f"Value+{i * self.num_load_size + 3}")))
                mod.addSpaceLine()

            label_skip_gamma = Label("skip_gamma_xN", f'skip_gamma')
            mod.add(ri.SCmpEQU64(sgpr("SrcGamma",2), 0))
            mod.add(ri.SCBranchSCC1(label_skip_gamma.getLabelName()))
            for i in range(0, self.num_load_count):
                mod.add(ri.BufferLoadB128(vgpr(f"Gamma+{i * self.num_load_size}",4), vgpr(f"Offset+{i}"), sgpr("SrcGamma",4), 0, MUBUFModifiers(offen=True)))
            for i in range(0, self.num_load_count):
                mod.add(ri.SWaitCnt(vlcnt=self.num_load_count-i-1))
                mod.add(ri.VMulF32(vgpr(f"Value+{i * self.num_load_size + 0}"), vgpr(f"Value+{i * self.num_load_size + 0}"), vgpr(f"Gamma+{i * self.num_load_size + 0}")))
                mod.add(ri.VMulF32(vgpr(f"Value+{i * self.num_load_size + 1}"), vgpr(f"Value+{i * self.num_load_size + 1}"), vgpr(f"Gamma+{i * self.num_load_size + 1}")))
                mod.add(ri.VMulF32(vgpr(f"Value+{i * self.num_load_size + 2}"), vgpr(f"Value+{i * self.num_load_size + 2}"), vgpr(f"Gamma+{i * self.num_load_size + 2}")))
                mod.add(ri.VMulF32(vgpr(f"Value+{i * self.num_load_size + 3}"), vgpr(f"Value+{i * self.num_load_size + 3}"), vgpr(f"Gamma+{i * self.num_load_size + 3}")))
            mod.add(label_skip_gamma)
            mod.addSpaceLine()

            label_skip_beta = Label("skip_beta_xN", f'skip_beta')
            mod.add(ri.SCmpEQU64(sgpr("SrcBeta",2), 0))
            mod.add(ri.SCBranchSCC1(label_skip_beta.getLabelName()))
            for i in range(0, self.num_load_count):
                mod.add(ri.BufferLoadB128(vgpr(f"Beta+{i * self.num_load_size}",4), vgpr(f"Offset+{i}"), sgpr("SrcBeta",4), 0, MUBUFModifiers(offen=True)))
            for i in range(0, self.num_load_count):
                mod.add(ri.SWaitCnt(vlcnt=self.num_load_count-i-1))
                mod.add(ri.VAddF32(vgpr(f"Value+{i * self.num_load_size + 0}"), vgpr(f"Value+{i * self.num_load_size + 0}"), vgpr(f"Beta+{i * self.num_load_size + 0}")))
                mod.add(ri.VAddF32(vgpr(f"Value+{i * self.num_load_size + 1}"), vgpr(f"Value+{i * self.num_load_size + 1}"), vgpr(f"Beta+{i * self.num_load_size + 1}")))
                mod.add(ri.VAddF32(vgpr(f"Value+{i * self.num_load_size + 2}"), vgpr(f"Value+{i * self.num_load_size + 2}"), vgpr(f"Beta+{i * self.num_load_size + 2}")))
                mod.add(ri.VAddF32(vgpr(f"Value+{i * self.num_load_size + 3}"), vgpr(f"Value+{i * self.num_load_size + 3}"), vgpr(f"Beta+{i * self.num_load_size + 3}")))
            mod.add(label_skip_beta)
            mod.addSpaceLine()

            for i in range(0, self.num_load_count):
                mod.add(ri.BufferStoreB128(vgpr(f"Value+{i * self.num_load_size}",4), vgpr(f"Offset+{i}"), sgpr("Dst",4), 0, MUBUFModifiers(offen=True)))
            mod.addSpaceLine()
            if not self.sweep_once:
                mod.add(ri.SMovB32(sgpr("Tmp"), offset * self.bpe))
                for i in range(0, self.num_load_count):
                    mod.add(ri.VAddU32(vgpr(f"Offset+{i}"), vgpr(f"Offset+{i}"), sgpr("Tmp")))
                mod.addSpaceLine()
        mod.addSpaceLine()
        return mod


    def layernorm_threadx4(self) -> Module:
        offset = self.num_workitems * self.num_load_size
        mod = Module("layernorm_threadx4")
        mod.addComment0("layernorm_threadx4")
        mod.add(ri.SLShiftRightB32(sgpr("MainLoop"), int(log2(offset)), sgpr("SizeLength")))
        mod.add(ri.SAndB32(sgpr("MainLoop"), hex(self.num_load_count-1), sgpr("MainLoop")))
        with asm_loop(mod, "layernorm_threadx4", "MainLoop", self.sweep_once):
            mod.add(ri.BufferLoadB128(vgpr("Value",4), vgpr("Offset"), sgpr("Src",4), 0, MUBUFModifiers(offen=True)))
            mod.addSpaceLine()
            mod.add(ri.SWaitCnt(vlcnt=0))
            mod.add(self.layernorm_cal(vgpr("Value+0")))
            mod.add(self.layernorm_cal(vgpr("Value+1")))
            mod.add(self.layernorm_cal(vgpr("Value+2")))
            mod.add(self.layernorm_cal(vgpr("Value+3")))
            mod.addSpaceLine()

            label_skip_gamma = Label("skip_gamma_x4", f'skip_gamma')
            mod.add(ri.SCmpEQU64(sgpr("SrcGamma",2), 0))
            mod.add(ri.SCBranchSCC1(label_skip_gamma.getLabelName()))
            mod.add(ri.BufferLoadB128(vgpr("Gamma",4), vgpr("Offset+0"), sgpr("SrcGamma",4), 0, MUBUFModifiers(offen=True)))
            mod.add(ri.SWaitCnt(vlcnt=0))
            mod.add(ri.VMulF32(vgpr("Value+0"), vgpr("Value+0"), vgpr("Gamma+0")))
            mod.add(ri.VMulF32(vgpr("Value+1"), vgpr("Value+1"), vgpr("Gamma+1")))
            mod.add(ri.VMulF32(vgpr("Value+2"), vgpr("Value+2"), vgpr("Gamma+2")))
            mod.add(ri.VMulF32(vgpr("Value+3"), vgpr("Value+3"), vgpr("Gamma+3")))
            mod.add(label_skip_gamma)
            mod.addSpaceLine()

            label_skip_beta = Label("skip_beta_x4", f'skip_beta')
            mod.add(ri.SCmpEQU64(sgpr("SrcBeta",2), 0))
            mod.add(ri.SCBranchSCC1(label_skip_beta.getLabelName()))
            mod.add(ri.BufferLoadB128(vgpr("Beta",4), vgpr("Offset"), sgpr("SrcBeta",4), 0, MUBUFModifiers(offen=True)))
            mod.add(ri.SWaitCnt(vlcnt=0))
            mod.add(ri.VAddF32(vgpr("Value+0"), vgpr("Value+0"), vgpr("Beta+0")))
            mod.add(ri.VAddF32(vgpr("Value+1"), vgpr("Value+1"), vgpr("Beta+1")))
            mod.add(ri.VAddF32(vgpr("Value+2"), vgpr("Value+2"), vgpr("Beta+2")))
            mod.add(ri.VAddF32(vgpr("Value+3"), vgpr("Value+3"), vgpr("Beta+3")))
            mod.add(label_skip_beta)
            mod.addSpaceLine()

            mod.add(ri.BufferStoreB128(vgpr("Value",4), vgpr("Offset"), sgpr("Dst",4), 0, MUBUFModifiers(offen=True)))
            mod.addSpaceLine()
            mod.add(ri.SMovB32(sgpr("Tmp"), self.num_workitems * self.num_load_size * self.bpe))
            mod.add(ri.VAddU32(vgpr("Offset+0"), vgpr("Offset"), sgpr("Tmp")))
            mod.addSpaceLine()
        return mod


    def layernorm_thread(self) -> Module:
        offset = self.num_workitems
        mod = Module("layernorm_thread")
        mod.addComment0("layernorm_thread")
        mod.add(ri.SLShiftRightB32(sgpr("MainLoop"), int(log2(offset)), sgpr("SizeLength")))
        mod.add(ri.SAndB32(sgpr("MainLoop"), sgpr("MainLoop"), 0x3))
        with asm_loop(mod, "layernorm_thread", "MainLoop", self.sweep_once):
            mod.add(ri.BufferLoadB32(vgpr("Value"), vgpr("Offset"), sgpr("Src",4), 0, MUBUFModifiers(offen=True)))
            mod.add(ri.SWaitCnt(vlcnt=0))
            mod.add(self.layernorm_cal(vgpr("Value")))

            label_skip_gamma = Label("skip_gamma", f'skip_gamma')
            mod.add(ri.SCmpEQU64(sgpr("SrcGamma",2), 0))
            mod.add(ri.SCBranchSCC1(label_skip_gamma.getLabelName()))
            mod.add(ri.BufferLoadB32(vgpr("Gamma"), vgpr("Offset+0"), sgpr("SrcGamma",4), 0, MUBUFModifiers(offen=True)))
            mod.add(ri.SWaitCnt(vlcnt=0))
            mod.add(ri.VMulF32(vgpr("Value+0"), vgpr("Value+0"), vgpr("Gamma+0")))
            mod.add(label_skip_gamma)
            mod.addSpaceLine()

            label_skip_beta = Label("skip_beta", f'skip_beta')
            mod.add(ri.SCmpEQU64(sgpr("SrcBeta",2), 0))
            mod.add(ri.SCBranchSCC1(label_skip_beta.getLabelName()))
            mod.add(ri.BufferLoadB32(vgpr("Beta"), vgpr("Offset"), sgpr("SrcBeta",4), 0, MUBUFModifiers(offen=True)))
            mod.add(ri.SWaitCnt(vlcnt=0))
            mod.add(ri.VAddF32(vgpr("Value+0"), vgpr("Value+0"), vgpr("Beta+0")))
            mod.add(label_skip_beta)
            mod.addSpaceLine()

            mod.add(ri.BufferStoreB32(vgpr("Value"), vgpr("Offset"), sgpr("Dst",4), 0, MUBUFModifiers(offen=True)))
            mod.add(ri.SWaitCnt(vscnt=0))
            mod.add(ri.SMovB32(sgpr("Tmp"), self.num_workitems * self.bpe))
            mod.add(ri.VAddU32(vgpr("Offset"), vgpr("Offset"), sgpr("Tmp")))
            mod.addSpaceLine()
        return mod


    def layernorm_in_some_thread(self)  -> Module:
        label_layernorm_end = Label("layernorm", f'loop layernorm end')
        mod = Module("layernorm_in_some_thread")
        mod.addComment0("layernorm_in_some_thread")
        mod.add(ri.SAndB32(sgpr("MainLoop"), sgpr("SizeLength"), self.num_workitems-1))
        mod.add(ri.VCmpLtU32(VCC(), vgpr("Serial"), sgpr("MainLoop")))
        mod.add(ri.SCBranchVCCZ(label_layernorm_end.getLabelName()))
        mod.add(self.SMovBX(EXEC(), VCC()))
        mod.add(ri.SNop(1))
        mod.add(ri.BufferLoadB32(vgpr("Value"), vgpr("Offset"), sgpr("Src",4), 0, MUBUFModifiers(offen=True)))
        mod.add(ri.SWaitCnt(vlcnt=0))
        mod.add(self.layernorm_cal(vgpr("Value")))

        label_skip_gamma = Label("skip_gamma_partial", f'skip_gamma')
        mod.add(ri.SCmpEQU64(sgpr("SrcGamma",2), 0))
        mod.add(ri.SCBranchSCC1(label_skip_gamma.getLabelName()))
        mod.add(ri.BufferLoadB32(vgpr("Gamma"), vgpr("Offset+0"), sgpr("SrcGamma",4), 0, MUBUFModifiers(offen=True)))
        mod.add(ri.SWaitCnt(vlcnt=0))
        mod.add(ri.VMulF32(vgpr("Value+0"), vgpr("Value+0"), vgpr("Gamma+0")))
        mod.add(label_skip_gamma)
        mod.addSpaceLine()

        label_skip_beta = Label("skip_beta_partial", f'skip_beta')
        mod.add(ri.SCmpEQU64(sgpr("SrcBeta",2), 0))
        mod.add(ri.SCBranchSCC1(label_skip_beta.getLabelName()))
        mod.add(ri.BufferLoadB32(vgpr("Beta"), vgpr("Offset"), sgpr("SrcBeta",4), 0, MUBUFModifiers(offen=True)))
        mod.add(ri.SWaitCnt(vlcnt=0))
        mod.add(ri.VAddF32(vgpr("Value+0"), vgpr("Value+0"), vgpr("Beta+0")))
        mod.add(label_skip_beta)
        mod.addSpaceLine()

        mod.add(ri.BufferStoreB32(vgpr("Value"), vgpr("Offset"), sgpr("Dst",4), 0, MUBUFModifiers(offen=True)))
        mod.add(ri.SWaitCnt(vscnt=0))
        mod.add(self.SMovBX(EXEC(), "-1"))
        mod.add(ri.SNop(1))
        mod.add(label_layernorm_end)
        mod.addSpaceLine()
        return mod


    def output_mean_and_invvar(self) -> Module:
        mod = Module("output_mean_and_invvar")
        mod.addComment0("output_mean_and_invvar")

        mod.add(ri.VLShiftLeftB32(vgpr("Offset"), hex(int(log2(self.bpe))), sgpr("WorkGroup1")))
        mod.addSpaceLine()

        mod.add(ri.SMovB32(sgpr("Dst+0"), sgpr("AddressMean+0")))
        mod.add(ri.SMovB32(sgpr("Dst+1"), sgpr("AddressMean+1")))
        mod.add(ri.SMovB32(sgpr("Dst+2"), "BufferLimit"))
        mod.add(ri.SMovB32(sgpr("Dst+3"), "Srd127_96"))
        mod.add(ri.BufferStoreB32(vgpr("Mean"), vgpr("Offset"), sgpr("Dst",4), 0, MUBUFModifiers(offen=True)))
        mod.addSpaceLine()

        mod.add(ri.SMovB32(sgpr("Dst+0"), sgpr("AddressInvvar+0")))
        mod.add(ri.SMovB32(sgpr("Dst+1"), sgpr("AddressInvvar+1")))
        mod.add(ri.SMovB32(sgpr("Dst+2"), "BufferLimit"))
        mod.add(ri.SMovB32(sgpr("Dst+3"), "Srd127_96"))
        mod.add(ri.BufferStoreB32(vgpr("Invvar"), vgpr("Offset"), sgpr("Dst",4), 0, MUBUFModifiers(offen=True)))
        mod.addSpaceLine()
        return mod

    def layernorm_kernel_body(self) -> Module:
        mod = Module(self.func_name)
        mod.add(self.defineVariables())
        with asm_func(self.func_name, mod):
            mod.add(self.load_kernel_args())
            mod.add(self.init_param())
            mod.add(self.calculate_global_address())
            mod.add(self.sum_per_threadxN())
            if not self.sweep_once:
                mod.add(self.sum_per_threadx4())
                mod.addSpaceLine()
                mod.add(self.adjusst_global_address())
                mod.add(self.sum_per_thread())
                mod.add(self.sum_in_some_thread())
            mod.add(self.intra_wave_reduction())
            mod.add(self.inter_wave_reduction())
            mod.add(self.broadcast())
            mod.add(self.get_average())
            if not self.sweep_once:
                mod.add(self.calculate_global_address())
            mod.add(self.layernorm_threadxN())
            if not self.sweep_once:
                mod.add(self.layernorm_threadx4())
                mod.add(self.adjusst_global_address())
                mod.add(self.layernorm_thread())
                mod.add(self.layernorm_in_some_thread())
            mod.add(self.output_mean_and_invvar())
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
    ap.add_argument('-w', type=int, default=256, help='workitem')
    ap.add_argument('-c', type=int, default=4, help='load conut per iteration')
    ap.add_argument('--sweep-once', type=int, default=0, dest='sweep_once', help='sweep once')
    ap.add_argument('--toolchain', type=str, default=ToolchainDefaults.CXX_COMPILER, help='Path to ROCm compiler')
    ap.add_argument('--debug-build', action='store_true', dest='debug_build', help='Build with debug information')
    ap.set_defaults(debug_build=False)
    ap.add_argument('--arch', type=str, default='gfx90a', help='Target architecture for assembler, e.g. gfx908. Default is gfx90a')
    args = ap.parse_args()
    output_path: str = args.output
    w: int = args.w
    c: int = args.c
    sweep_once: int = args.sweep_once
    toolchain_path: str = validateToolchain(args.toolchain)
    debug_build: bool = args.debug_build
    arch: str = args.arch
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
    layernorm = LayerNormKernelGenerator(DataType('S'), w, c, 4, sweep_once, waveFrontSize, arch, isa)
    kernel_body = layernorm.layernorm_kernel_body()
    args = layernorm.kernel_args()
    func_name = layernorm.func_name
    meta = KernelMeta(func_name, layernorm.vgpr_pool.size(), layernorm.sgpr_pool.size(), 0, layernorm.lds_usage_byte, waveFrontSize, w, 8, args)
    meta.update_args_offsets()
    k_str = '\n'.join([kernel_header(func_name, arch, layernorm.vgpr_pool.size(), layernorm.sgpr_pool.size(), layernorm.lds_usage_byte),
                       meta_str((meta,)),
                       str(kernel_body)])

    with open(output_path, 'w') as f:
        f.write(k_str)

    output_path_basename = os.path.splitext(output_path)[0]
    layernorm.dump('yaml', f'{output_path_basename}.yaml')
