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

from kernels.configs import config_lds
from types import SimpleNamespace as NS

# yapf: disable
pp_3d_kernels = [
        NS(length=[32,32,128], dims=[0,2], factors=[[8,4],[2,4,4,4]], factors_pp=[[4],[8]], threads_per_transform=[8,16], workgroup_size=[64,128], direct_to_from_reg=[False,False], precision=['dp']),
        NS(length=[32,32,64], dims=[0,2], factors=[[4,8],[4,4,4]], factors_pp=[[4],[8]], threads_per_transform=[8,16], workgroup_size=[64,128], direct_to_from_reg=[False,False], precision=['dp']),
        NS(length=[64,32,128], dims=[0,2], factors=[[8,8],[2,4,4,4]], factors_pp=[[4],[8]], threads_per_transform=[8,16], workgroup_size=[64,128], direct_to_from_reg=[False,False], precision=['dp']),
        NS(length=[64,64,128], dims=[0,2], factors=[[8,8],[2,4,4,4]], factors_pp=[[4],[16]], threads_per_transform=[8,16], workgroup_size=[64,256], direct_to_from_reg=[False,False]),
        NS(length=[64,64,64], dims=[0,2], factors=[[8,8],[4,4,4]], factors_pp=[[4],[16]], threads_per_transform=[8,16], workgroup_size=[64,256], direct_to_from_reg=[False,False]),
        NS(length=[64,64,52], dims=[0,2], factors=[[8,8],[13,4]], factors_pp=[[4],[16]], threads_per_transform=[8,4], workgroup_size=[64,64], direct_to_from_reg=[False,False]),
        NS(length=[60,60,60], dims=[0,2], factors=[[6,10],[6,10]], factors_pp=[[6],[10]], threads_per_transform=[10,10], workgroup_size=[20,100], direct_to_from_reg=[False,False], precision=['dp']),
]
