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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from kernels.configs.config_arch import lds_config
from kernels.configs.config_arch import supported_arch
from types import SimpleNamespace as NS
from enum import Enum

class pp_transform_type(Enum):
    # Complex-to-Complex
    C2C = "c2c" 
    # Real-to-Complex
    R2C = "r2c" 
    # Complex-to-Real
    C2R = "c2r" 

# yapf: disable
pp_3d_kernels = [
        #======================================================================
        # Config entries for all architectures (gfx_generic)
        #======================================================================

        NS(type=pp_transform_type.C2C.value, length=[32,32,128], dims=[0,2], factors=[[8,4],[2,4,4,4]], factors_pp=[[4],[8]], threads_per_transform=[8,16], threads_per_transform_pp=[1,1], workgroup_size=[64,128], direct_to_from_reg=[False,False], precision=['dp']),
        NS(type=pp_transform_type.C2C.value, length=[32,32,64], dims=[0,2], factors=[[4,8],[4,4,4]], factors_pp=[[4],[8]], threads_per_transform=[8,16], threads_per_transform_pp=[1,1], workgroup_size=[64,128], direct_to_from_reg=[False,False], precision=['dp']),
        NS(type=pp_transform_type.C2C.value, length=[64,32,128], dims=[0,2], factors=[[8,8],[2,4,4,4]], factors_pp=[[4],[8]], threads_per_transform=[8,16], threads_per_transform_pp=[1,1], workgroup_size=[64,128], direct_to_from_reg=[False,False], precision=['dp']),
        NS(type=pp_transform_type.C2C.value, length=[64,64,128], dims=[0,2], factors=[[8,8],[2,4,4,4]], factors_pp=[[4],[16]], threads_per_transform=[8,16], threads_per_transform_pp=[1,1], workgroup_size=[64,256], direct_to_from_reg=[False,False]),
        NS(type=pp_transform_type.C2C.value, length=[64,64,64], dims=[0,2], factors=[[8,8],[4,4,4]], factors_pp=[[4],[16]], threads_per_transform=[8,16], threads_per_transform_pp=[1,1], workgroup_size=[64,256], direct_to_from_reg=[False,False]),
        NS(type=pp_transform_type.C2C.value, length=[64,64,52], dims=[0,2], factors=[[8,8],[13,4]], factors_pp=[[4],[16]], threads_per_transform=[8,4], threads_per_transform_pp=[1,1], workgroup_size=[64,64], direct_to_from_reg=[False,False]),
        NS(type=pp_transform_type.C2C.value, length=[60,60,60], dims=[0,2], factors=[[6,10],[6,10]], factors_pp=[[6],[10]], threads_per_transform=[10,10], threads_per_transform_pp=[1,1], workgroup_size=[20,100], direct_to_from_reg=[False,False], precision=['dp']),

        #======================================================================
        # Config entries for specific architectures
        #======================================================================

        NS(type=pp_transform_type.C2C.value, length=[32,32,128], dims=[0,2], factors=[[8,4],[2,4,4,4]], factors_pp=[[4],[8]], threads_per_transform=[8,16], threads_per_transform_pp=[1,1], workgroup_size=[64,128], direct_to_from_reg=[False,False], gcn_arch_name=[supported_arch.GFX_1201.value,supported_arch.GFX_90A.value,supported_arch.GFX_942.value,supported_arch.GFX_950.value], precision=['dp']),
        NS(type=pp_transform_type.C2C.value, length=[32,32,128], dims=[0,2], factors=[[8,4],[2,4,4,4]], factors_pp=[[4],[8]], threads_per_transform=[8,16], threads_per_transform_pp=[1,1], workgroup_size=[64,128], direct_to_from_reg=[False,False], gcn_arch_name=[supported_arch.GFX_90A.value,supported_arch.GFX_942.value,supported_arch.GFX_950.value], precision=['sp']),
        NS(type=pp_transform_type.C2C.value, length=[32,32,128], dims=[0,2], factors=[[8,4],[8,16]], factors_pp=[[4],[8]], threads_per_transform=[8,32], threads_per_transform_pp=[1,1], workgroup_size=[64,256], direct_to_from_reg=[False,False], gcn_arch_name=supported_arch.GFX_1201.value, precision=['sp']),

        NS(type=pp_transform_type.C2C.value, length=[32,32,64], dims=[0,2], factors=[[4,8],[4,4,4]], factors_pp=[[4],[8]], threads_per_transform=[8,8], threads_per_transform_pp=[1,1], workgroup_size=[64,64], direct_to_from_reg=[False,False], gcn_arch_name=[supported_arch.GFX_1201.value,supported_arch.GFX_90A.value,supported_arch.GFX_942.value,supported_arch.GFX_950.value], precision=['dp']),
        NS(type=pp_transform_type.C2C.value, length=[32,32,64], dims=[0,2], factors=[[4,8],[4,4,4]], factors_pp=[[4],[8]], threads_per_transform=[8,8], threads_per_transform_pp=[1,1], workgroup_size=[64,64], direct_to_from_reg=[False,False], gcn_arch_name=[supported_arch.GFX_90A.value,supported_arch.GFX_942.value,supported_arch.GFX_950.value], precision=['sp']),
        NS(type=pp_transform_type.C2C.value, length=[32,32,64], dims=[0,2], factors=[[4,8],[16,4]], factors_pp=[[4],[2,2,2]], threads_per_transform=[8,16], threads_per_transform_pp=[1,1], workgroup_size=[32,128], direct_to_from_reg=[False,False], gcn_arch_name=supported_arch.GFX_1201.value, precision=['sp']),

        NS(type=pp_transform_type.C2C.value, length=[64,32,128], dims=[0,2], factors=[[8,8],[2,4,4,4]], factors_pp=[[4],[8]], threads_per_transform=[8,16], threads_per_transform_pp=[1,1], workgroup_size=[64,128], direct_to_from_reg=[False,False], gcn_arch_name=[supported_arch.GFX_1201.value,supported_arch.GFX_90A.value,supported_arch.GFX_942.value,supported_arch.GFX_950.value], precision=['dp']),
        NS(type=pp_transform_type.C2C.value, length=[64,32,128], dims=[0,2], factors=[[8,8],[8,4,4]], factors_pp=[[4],[8]], threads_per_transform=[8,16], threads_per_transform_pp=[1,1], workgroup_size=[32,128], direct_to_from_reg=[False,False], gcn_arch_name=supported_arch.GFX_1201.value, precision=['sp']),
        NS(type=pp_transform_type.C2C.value, length=[64,32,128], dims=[0,2], factors=[[8,8],[8,4,4]], factors_pp=[[4],[8]], threads_per_transform=[8,16], threads_per_transform_pp=[1,1], workgroup_size=[64,128], direct_to_from_reg=[False,False], gcn_arch_name=supported_arch.GFX_90A.value, precision=['sp']),

        NS(type=pp_transform_type.C2C.value, length=[64,64,128], dims=[0,2], factors=[[8,8],[2,4,4,4]], factors_pp=[[4],[16]], threads_per_transform=[8,8], threads_per_transform_pp=[1,1], workgroup_size=[64,128], direct_to_from_reg=[False,False], gcn_arch_name=supported_arch.GFX_1201.value, precision=['dp']),
        NS(type=pp_transform_type.C2C.value, length=[64,64,128], dims=[0,2], factors=[[8,8],[16,8]], factors_pp=[[4],[16]], threads_per_transform=[16,16], threads_per_transform_pp=[1,1], workgroup_size=[128,256], direct_to_from_reg=[False,False], gcn_arch_name=supported_arch.GFX_90A.value, precision=['dp']),
        NS(type=pp_transform_type.C2C.value, length=[64,64,128], dims=[0,2], factors=[[8,8],[4,4,8]], factors_pp=[[4],[16]], threads_per_transform=[8,8], threads_per_transform_pp=[1,1], workgroup_size=[64,128], direct_to_from_reg=[False,False], gcn_arch_name=[supported_arch.GFX_942.value,supported_arch.GFX_950.value], precision=['dp']),
        NS(type=pp_transform_type.C2C.value, length=[64,64,128], dims=[0,2], factors=[[8,8],[4,4,8]], factors_pp=[[4],[16]], threads_per_transform=[8,16], threads_per_transform_pp=[1,1], workgroup_size=[64,256], direct_to_from_reg=[False,False], gcn_arch_name=[supported_arch.GFX_1201.value,supported_arch.GFX_90A.value,supported_arch.GFX_942.value], precision=['sp']),
        NS(type=pp_transform_type.C2C.value, length=[64,64,128], dims=[0,2], factors=[[8,8],[4,4,8]], factors_pp=[[4],[16]], threads_per_transform=[8,8], threads_per_transform_pp=[1,1], workgroup_size=[64,128], direct_to_from_reg=[False,False], gcn_arch_name=supported_arch.GFX_950.value, precision=['sp']),

        NS(type=pp_transform_type.C2C.value, length=[64,64,64], dims=[0,2], factors=[[8,8],[4,4,4]], factors_pp=[[4],[16]], threads_per_transform=[8,4], threads_per_transform_pp=[1,1], workgroup_size=[64,64], direct_to_from_reg=[False,False], gcn_arch_name=supported_arch.GFX_1201.value, precision=['dp']),
        NS(type=pp_transform_type.C2C.value, length=[64,64,64], dims=[0,2], factors=[[8,8],[8,8]], factors_pp=[[4],[16]], threads_per_transform=[16,8], threads_per_transform_pp=[1,1], workgroup_size=[128,128], direct_to_from_reg=[False,False], gcn_arch_name=supported_arch.GFX_90A.value, precision=['dp']),
        NS(type=pp_transform_type.C2C.value, length=[64,64,64], dims=[0,2], factors=[[8,8],[8,8]], factors_pp=[[4],[16]], threads_per_transform=[8,4], threads_per_transform_pp=[1,1], workgroup_size=[64,64], direct_to_from_reg=[False,False], gcn_arch_name=[supported_arch.GFX_942.value,supported_arch.GFX_950.value], precision=['dp']),
        NS(type=pp_transform_type.C2C.value, length=[64,64,64], dims=[0,2], factors=[[8,8],[8,8]], factors_pp=[[4],[4,4]], threads_per_transform=[4,4], threads_per_transform_pp=[1,1], workgroup_size=[32,64], direct_to_from_reg=[False,False], gcn_arch_name=supported_arch.GFX_1201.value, precision=['sp']),
        NS(type=pp_transform_type.C2C.value, length=[64,64,64], dims=[0,2], factors=[[8,8],[8,8]], factors_pp=[[4],[16]], threads_per_transform=[8,8], threads_per_transform_pp=[1,1], workgroup_size=[64,128], direct_to_from_reg=[False,False], gcn_arch_name=[supported_arch.GFX_90A.value,supported_arch.GFX_942.value,supported_arch.GFX_950.value], precision=['sp']),
        NS(type=pp_transform_type.R2C.value, length=[64,64,64], dims=[0,2], factors=[[8,8],[8,4]], factors_pp=[[4],[16]], threads_per_transform=[8,4], threads_per_transform_pp=[1,1], workgroup_size=[64,64], direct_to_from_reg=[False,False], gcn_arch_name=[supported_arch.GFX_90A.value,supported_arch.GFX_942.value,supported_arch.GFX_1201.value], precision=['dp']),
        NS(type=pp_transform_type.R2C.value, length=[64,64,64], dims=[0,2], factors=[[8,8],[8,4]], factors_pp=[[4],[4,4]], threads_per_transform=[8,4], threads_per_transform_pp=[1,1], workgroup_size=[64,64], direct_to_from_reg=[False,False], gcn_arch_name=supported_arch.GFX_950.value, precision=['dp']),

        NS(type=pp_transform_type.C2C.value, length=[64,64,52], dims=[0,2], factors=[[8,8],[13,4]], factors_pp=[[4],[16]], threads_per_transform=[8,4], threads_per_transform_pp=[1,1], workgroup_size=[64,64], direct_to_from_reg=[False,False], gcn_arch_name=[supported_arch.GFX_1201.value,supported_arch.GFX_90A.value,supported_arch.GFX_942.value,supported_arch.GFX_950.value], precision=['dp','sp']),
        NS(type=pp_transform_type.R2C.value, length=[64,64,52], dims=[0,2], factors=[[8,8],[2,13]], factors_pp=[[4],[16]], threads_per_transform=[8,2], threads_per_transform_pp=[1,1], workgroup_size=[128,32], direct_to_from_reg=[False,False], gcn_arch_name=supported_arch.GFX_942.value, precision=['dp']),
        NS(type=pp_transform_type.R2C.value, length=[64,64,52], dims=[0,2], factors=[[8,8],[13,2]], factors_pp=[[4],[16]], threads_per_transform=[8,2], threads_per_transform_pp=[1,1], workgroup_size=[64,32], direct_to_from_reg=[False,False], gcn_arch_name=supported_arch.GFX_1201.value, precision=['dp']),

        NS(type=pp_transform_type.C2C.value, length=[60,60,60], dims=[0,2], factors=[[6,10],[6,10]], factors_pp=[[6],[10]], threads_per_transform=[10,6], threads_per_transform_pp=[1,1], workgroup_size=[20,60], direct_to_from_reg=[False,False], gcn_arch_name=[supported_arch.GFX_1201.value,supported_arch.GFX_90A.value,supported_arch.GFX_942.value,supported_arch.GFX_950.value], precision=['dp']),
        NS(type=pp_transform_type.R2C.value, length=[60,60,60], dims=[0,2], factors=[[6,10],[6,5]], factors_pp=[[6],[10]], threads_per_transform=[10,6], threads_per_transform_pp=[1,1], workgroup_size=[20,60], direct_to_from_reg=[False,False], gcn_arch_name=supported_arch.GFX_1201.value, precision=['dp']),

        NS(type=pp_transform_type.C2C.value, length=[72,72,52], dims=[0,2], factors=[[8,3,3],[13,4]], factors_pp=[[3],[8,3]], threads_per_transform=[3,4], threads_per_transform_pp=[1,1], workgroup_size=[27,96], direct_to_from_reg=[False,False], gcn_arch_name=[supported_arch.GFX_90A.value,supported_arch.GFX_942.value,supported_arch.GFX_950.value], precision=['dp']),
        NS(type=pp_transform_type.R2C.value, length=[72,72,52], dims=[0,2], factors=[[8,3,3],[13,2]], factors_pp=[[3],[8,3]], threads_per_transform=[3,2], threads_per_transform_pp=[1,1], workgroup_size=[27,48], direct_to_from_reg=[False,False], gcn_arch_name=[supported_arch.GFX_950.value,supported_arch.GFX_942.value,supported_arch.GFX_90A.value], precision=['dp']),

        NS(type=pp_transform_type.C2C.value, length=[160,72,72], dims=[0,2], factors=[[4,10,4],[8,3,3]], factors_pp=[[4],[3,3,2]], threads_per_transform=[4,6], threads_per_transform_pp=[1,1], workgroup_size=[32,108], direct_to_from_reg=[False,False], gcn_arch_name=[supported_arch.GFX_1201.value,supported_arch.GFX_950.value], precision=['sp']),
        NS(type=pp_transform_type.C2C.value, length=[160,72,72], dims=[0,2], factors=[[4,10,4],[8,3,3]], factors_pp=[[4],[3,3,2]], threads_per_transform=[4,6], threads_per_transform_pp=[1,1], workgroup_size=[16,108], direct_to_from_reg=[False,False], gcn_arch_name=[supported_arch.GFX_90A.value,supported_arch.GFX_942.value], precision=['sp']),
        NS(type=pp_transform_type.R2C.value, length=[160,72,72], dims=[0,2], factors=[[4,10,4],[4,3,3]], factors_pp=[[4],[3,3,2]], threads_per_transform=[40,6], threads_per_transform_pp=[1,1], workgroup_size=[160,108], direct_to_from_reg=[False,False], gcn_arch_name=supported_arch.GFX_90A.value, precision=['sp']),
        NS(type=pp_transform_type.R2C.value, length=[160,72,72], dims=[0,2], factors=[[4,10,4],[4,3,3]], factors_pp=[[4],[3,3,2]], threads_per_transform=[4,6], threads_per_transform_pp=[1,1], workgroup_size=[32,108], direct_to_from_reg=[False,False], gcn_arch_name=[supported_arch.GFX_942.value,supported_arch.GFX_1201.value], precision=['sp']),
        NS(type=pp_transform_type.R2C.value, length=[160,72,72], dims=[0,2], factors=[[4,10,4],[4,3,3]], factors_pp=[[4],[9,2]], threads_per_transform=[4,3], threads_per_transform_pp=[1,1], workgroup_size=[32,54], direct_to_from_reg=[False,False], gcn_arch_name=[supported_arch.GFX_950.value], precision=['sp']),

        NS(type=pp_transform_type.C2C.value, length=[72,72,72], dims=[0,2], factors=[[8,3,3],[8,3,3]], factors_pp=[[4],[3,3,2]], threads_per_transform=[3,6], threads_per_transform_pp=[1,1], workgroup_size=[27,108], direct_to_from_reg=[False,False], gcn_arch_name=[supported_arch.GFX_1201.value,supported_arch.GFX_90A.value,supported_arch.GFX_942.value], precision=['sp']),
        NS(type=pp_transform_type.C2C.value, length=[72,72,72], dims=[0,2], factors=[[9,8],[9,8]], factors_pp=[[4],[3,3,2]], threads_per_transform=[9,6], threads_per_transform_pp=[1,1], workgroup_size=[72,108], direct_to_from_reg=[False,False], gcn_arch_name=supported_arch.GFX_950.value, precision=['sp']),
        NS(type=pp_transform_type.R2C.value, length=[72,72,72], dims=[0,2], factors=[[9,8],[9,4]], factors_pp=[[4],[3,3,2]], threads_per_transform=[9,6], threads_per_transform_pp=[1,1], workgroup_size=[72,108], direct_to_from_reg=[False,False], gcn_arch_name=[supported_arch.GFX_942.value,supported_arch.GFX_90A.value], precision=['sp']),
        NS(type=pp_transform_type.R2C.value, length=[72,72,72], dims=[0,2], factors=[[8,3,3],[9,4]], factors_pp=[[4],[3,3,2]], threads_per_transform=[3,6], threads_per_transform_pp=[1,1], workgroup_size=[27,108], direct_to_from_reg=[False,False], gcn_arch_name=supported_arch.GFX_1201.value, precision=['sp']),
        NS(type=pp_transform_type.R2C.value, length=[72,72,72], dims=[0,2], factors=[[9,8],[9,4]], factors_pp=[[4],[9,2]], threads_per_transform=[9,3], threads_per_transform_pp=[1,1], workgroup_size=[72,54], direct_to_from_reg=[False,False], gcn_arch_name=[supported_arch.GFX_950.value], precision=['sp']),
        
        NS(type=pp_transform_type.C2C.value, length=[160,80,72], dims=[0,2], factors=[[4,10,4],[8,3,3]], factors_pp=[[4],[5,4]], threads_per_transform=[4,8], threads_per_transform_pp=[1,1], workgroup_size=[32,160], direct_to_from_reg=[False,False], gcn_arch_name=[supported_arch.GFX_1201.value,supported_arch.GFX_950.value], precision=['sp']),
        NS(type=pp_transform_type.C2C.value, length=[160,80,72], dims=[0,2], factors=[[4,10,4],[8,3,3]], factors_pp=[[4],[5,4]], threads_per_transform=[40,8], threads_per_transform_pp=[1,1], workgroup_size=[160,160], direct_to_from_reg=[False,False], gcn_arch_name=[supported_arch.GFX_90A.value,supported_arch.GFX_942.value], precision=['sp']),
        NS(type=pp_transform_type.R2C.value, length=[160,80,72], dims=[0,2], factors=[[4,10,4],[4,3,3]], factors_pp=[[4],[5,4]], threads_per_transform=[4,4], threads_per_transform_pp=[1,1], workgroup_size=[32,80], direct_to_from_reg=[False,False], gcn_arch_name=supported_arch.GFX_1201.value, precision=['sp']),
        NS(type=pp_transform_type.R2C.value, length=[160,80,72], dims=[0,2], factors=[[10,4,4],[4,3,3]], factors_pp=[[4],[5,4]], threads_per_transform=[4,2], threads_per_transform_pp=[1,1], workgroup_size=[32,40], direct_to_from_reg=[False,False], gcn_arch_name=supported_arch.GFX_942.value, precision=['sp']),
        NS(type=pp_transform_type.R2C.value, length=[160,80,72], dims=[0,2], factors=[[4,10,4],[4,3,3]], factors_pp=[[4],[5,4]], threads_per_transform=[40,4], threads_per_transform_pp=[1,1], workgroup_size=[160,80], direct_to_from_reg=[False,False], gcn_arch_name=supported_arch.GFX_90A.value, precision=['sp']),

        NS(type=pp_transform_type.C2C.value, length=[96,96,96], dims=[0,2], factors=[[6,4,4],[6,4,4]], factors_pp=[[4],[3,8]], threads_per_transform=[4,4], threads_per_transform_pp=[1,1], workgroup_size=[24,96], direct_to_from_reg=[False,False], gcn_arch_name=supported_arch.GFX_1201.value, precision=['sp']),
        NS(type=pp_transform_type.C2C.value, length=[96,96,96], dims=[0,2], factors=[[6,4,4],[6,4,4]], factors_pp=[[4],[3,8]], threads_per_transform=[8,4], threads_per_transform_pp=[1,1], workgroup_size=[64,96], direct_to_from_reg=[False,False], gcn_arch_name=[supported_arch.GFX_90A.value,supported_arch.GFX_942.value,supported_arch.GFX_950.value], precision=['sp']),
        NS(type=pp_transform_type.R2C.value, length=[96,96,96], dims=[0,2], factors=[[6,4,4],[6,8]], factors_pp=[[4],[3,8]], threads_per_transform=[8,4], threads_per_transform_pp=[1,1], workgroup_size=[64,96], direct_to_from_reg=[False,False], gcn_arch_name=[supported_arch.GFX_942.value,supported_arch.GFX_90A.value], precision=['sp']),
        NS(type=pp_transform_type.R2C.value, length=[96,96,96], dims=[0,2], factors=[[6,4,4],[6,8]], factors_pp=[[4],[3,8]], threads_per_transform=[4,4], threads_per_transform_pp=[1,1], workgroup_size=[24,96], direct_to_from_reg=[False,False], gcn_arch_name=supported_arch.GFX_1201.value, precision=['sp']),

        NS(type=pp_transform_type.C2C.value, length=[80,80,80], dims=[0,2], factors=[[5,2,8],[10,8]], factors_pp=[[4],[10,2]], threads_per_transform=[2,5], threads_per_transform_pp=[1,1], workgroup_size=[20,100], direct_to_from_reg=[False,False], gcn_arch_name=supported_arch.GFX_1201.value, precision=['sp']),
        NS(type=pp_transform_type.C2C.value, length=[80,80,80], dims=[0,2], factors=[[10,8],[10,8]], factors_pp=[[4],[10,2]], threads_per_transform=[10,5], threads_per_transform_pp=[1,1], workgroup_size=[80,100], direct_to_from_reg=[False,False], gcn_arch_name=supported_arch.GFX_90A.value, precision=['sp']),
        NS(type=pp_transform_type.R2C.value, length=[80,80,80], dims=[0,2], factors=[[10,8],[10,4]], factors_pp=[[4],[10,2]], threads_per_transform=[10,5], threads_per_transform_pp=[1,1], workgroup_size=[80,100], direct_to_from_reg=[False,False], gcn_arch_name=supported_arch.GFX_90A.value, precision=['sp']),
        NS(type=pp_transform_type.R2C.value, length=[80,80,80], dims=[0,2], factors=[[5,2,8],[10,4]], factors_pp=[[4],[10,2]], threads_per_transform=[2,5], threads_per_transform_pp=[1,1], workgroup_size=[20,100], direct_to_from_reg=[False,False], gcn_arch_name=supported_arch.GFX_1201.value, precision=['sp']),

        NS(type=pp_transform_type.C2C.value, length=[84,84,72], dims=[0,2], factors=[[7,2,6],[8,3,3]], factors_pp=[[4],[7,3]], threads_per_transform=[21,6], threads_per_transform_pp=[1,1], workgroup_size=[42,126], direct_to_from_reg=[False,False], gcn_arch_name=supported_arch.GFX_1201.value, precision=['sp']),
        NS(type=pp_transform_type.C2C.value, length=[84,84,72], dims=[0,2], factors=[[7,2,6],[8,3,3]], factors_pp=[[4],[7,3]], threads_per_transform=[21,6], threads_per_transform_pp=[1,1], workgroup_size=[168,126], direct_to_from_reg=[False,False], gcn_arch_name=[supported_arch.GFX_90A.value,supported_arch.GFX_942.value], precision=['sp']),
        NS(type=pp_transform_type.R2C.value, length=[84,84,72], dims=[0,2], factors=[[7,2,6],[4,3,3]], factors_pp=[[4],[7,3]], threads_per_transform=[21,6], threads_per_transform_pp=[1,1], workgroup_size=[42,126], direct_to_from_reg=[False,False], gcn_arch_name=supported_arch.GFX_1201.value, precision=['sp']),
        NS(type=pp_transform_type.R2C.value, length=[84,84,72], dims=[0,2], factors=[[7,2,6],[6,6]], factors_pp=[[4],[7,3]], threads_per_transform=[21,3], threads_per_transform_pp=[1,1], workgroup_size=[189,63], direct_to_from_reg=[False,False], gcn_arch_name=supported_arch.GFX_942.value, precision=['sp']),

        NS(type=pp_transform_type.C2C.value, length=[160,80,80], dims=[0,2], factors=[[4,10,4],[10,8]], factors_pp=[[4],[5,4]], threads_per_transform=[4,5], threads_per_transform_pp=[1,1], workgroup_size=[32,100], direct_to_from_reg=[False,False], gcn_arch_name=[supported_arch.GFX_1201.value,supported_arch.GFX_950.value], precision=['sp']),
        NS(type=pp_transform_type.C2C.value, length=[160,80,80], dims=[0,2], factors=[[16,10],[10,8]], factors_pp=[[4],[5,4]], threads_per_transform=[8,5], threads_per_transform_pp=[1,1], workgroup_size=[64,100], direct_to_from_reg=[False,False], gcn_arch_name=[supported_arch.GFX_90A.value,supported_arch.GFX_942.value], precision=['sp']),
        NS(type=pp_transform_type.R2C.value, length=[160,80,80], dims=[0,2], factors=[[16,10],[10,4]], factors_pp=[[4],[5,4]], threads_per_transform=[8,5], threads_per_transform_pp=[1,1], workgroup_size=[64,100], direct_to_from_reg=[False,False], gcn_arch_name=supported_arch.GFX_90A.value, precision=['sp']),
        NS(type=pp_transform_type.R2C.value, length=[160,80,80], dims=[0,2], factors=[[4,10,4],[10,4]], factors_pp=[[4],[5,4]], threads_per_transform=[4,5], threads_per_transform_pp=[1,1], workgroup_size=[32,100], direct_to_from_reg=[False,False], gcn_arch_name=supported_arch.GFX_1201.value, precision=['sp']),

        NS(type=pp_transform_type.C2C.value, length=[108,108,80], dims=[0,2], factors=[[6,6,3],[10,8]], factors_pp=[[4],[3,3,3]], threads_per_transform=[9,5], threads_per_transform_pp=[1,1], workgroup_size=[36,135], direct_to_from_reg=[False,False], gcn_arch_name=supported_arch.GFX_1201.value, precision=['sp']),
        NS(type=pp_transform_type.C2C.value, length=[108,108,80], dims=[0,2], factors=[[6,6,3],[10,8]], factors_pp=[[4],[9,3]], threads_per_transform=[9,8], threads_per_transform_pp=[1,1], workgroup_size=[72,216], direct_to_from_reg=[False,False], gcn_arch_name=[supported_arch.GFX_90A.value,supported_arch.GFX_942.value], precision=['sp']),
        NS(type=pp_transform_type.C2C.value, length=[108,108,80], dims=[0,2], factors=[[6,6,3],[10,8]], factors_pp=[[4],[9,3]], threads_per_transform=[9,5], threads_per_transform_pp=[1,1], workgroup_size=[72,135], direct_to_from_reg=[False,False], gcn_arch_name=supported_arch.GFX_950.value, precision=['sp']),
        NS(type=pp_transform_type.R2C.value, length=[108,108,80], dims=[0,2], factors=[[6,6,3],[4,10]], factors_pp=[[4],[9,3]], threads_per_transform=[9,4], threads_per_transform_pp=[1,1], workgroup_size=[72,108], direct_to_from_reg=[False,False], gcn_arch_name=supported_arch.GFX_90A.value, precision=['sp']),
        NS(type=pp_transform_type.R2C.value, length=[108,108,80], dims=[0,2], factors=[[6,6,3],[10,4]], factors_pp=[[4],[9,3]], threads_per_transform=[9,5], threads_per_transform_pp=[1,1], workgroup_size=[72,135], direct_to_from_reg=[False,False], gcn_arch_name=supported_arch.GFX_1201.value, precision=['sp']),
        NS(type=pp_transform_type.R2C.value, length=[108,108,80], dims=[0,2], factors=[[6,6,3],[4,10]], factors_pp=[[4],[9,3]], threads_per_transform=[9,2], threads_per_transform_pp=[1,1], workgroup_size=[72,54], direct_to_from_reg=[False,False], gcn_arch_name=supported_arch.GFX_950.value, precision=['sp']),
]
