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

# for SBRC, if direct_to_from_reg is True, we do store-from-reg, but will not do load-to-reg
#           And since SBRC is is dir-from-lds but NOT dir-to-reg, the global load part requires full LDS
#           So, SBRC is able to use half-lds.

# yapf: disable
sbrc_kernels = [
    NS(length=17,  factors=[17], scheme='CS_KERNEL_STOCKHAM_BLOCK_RC', workgroup_size=256, threads_per_transform=1, runtime_compile=True),
    NS(length=49,  factors=[7, 7], scheme='CS_KERNEL_STOCKHAM_BLOCK_RC', workgroup_size=196, threads_per_transform=7), # block_width=28
    NS(length=50,  factors=[10, 5], scheme='CS_KERNEL_STOCKHAM_BLOCK_RC', workgroup_size=50, threads_per_transform=5, direct_to_from_reg=False), # block_width=10
    # SBRC64: wgs=256 poor in MI50
    NS(length=64,  factors=[4, 4, 4], scheme='CS_KERNEL_STOCKHAM_BLOCK_RC', workgroup_size=128, threads_per_transform=16), # block_width=8
    # 9,9 not good by experiments
    NS(length=81,  factors=[3, 3, 3, 3], scheme='CS_KERNEL_STOCKHAM_BLOCK_RC', workgroup_size=243, threads_per_transform=27), # block_width=9
    NS(length=100, factors=[5, 5, 4], scheme='CS_KERNEL_STOCKHAM_BLOCK_RC', workgroup_size=100, threads_per_transform=25), # block_width=4
    NS(length=112, factors=[4, 7, 4], scheme='CS_KERNEL_STOCKHAM_BLOCK_RC', workgroup_size=448, threads_per_transform=28), # block_width=16
    NS(length=121, factors=[11, 11], scheme='CS_KERNEL_STOCKHAM_BLOCK_RC', workgroup_size=128, threads_per_transform=11, runtime_compile=True),
    NS(length=125, factors=[5, 5, 5], scheme='CS_KERNEL_STOCKHAM_BLOCK_RC', workgroup_size=250, threads_per_transform=25), # block_width=10
    NS(length=128, factors=[8, 4, 4], scheme='CS_KERNEL_STOCKHAM_BLOCK_RC', workgroup_size=128, threads_per_transform=16), # block_width=8
    NS(length=169, factors=[13, 13], scheme='CS_KERNEL_STOCKHAM_BLOCK_RC', workgroup_size=256, threads_per_transform=13, runtime_compile=True),
    NS(length=192, factors=[6, 4, 4, 2], scheme='CS_KERNEL_STOCKHAM_BLOCK_RC', workgroup_size=256, threads_per_transform=32), # block_width=8
    NS(length=200, factors=[8, 5, 5], scheme='CS_KERNEL_STOCKHAM_BLOCK_RC', workgroup_size=400, threads_per_transform=40), # block_width=10
    NS(length=243, factors=[3, 3, 3, 3, 3], scheme='CS_KERNEL_STOCKHAM_BLOCK_RC', workgroup_size=256, threads_per_transform=27, runtime_compile=True), # block_width=10
    NS(length=256, factors=[4, 4, 4, 4], scheme='CS_KERNEL_STOCKHAM_BLOCK_RC', workgroup_size=256, threads_per_transform=32), # block_width=8
    NS(length=289, factors=[17, 17], scheme='CS_KERNEL_STOCKHAM_BLOCK_RC', workgroup_size=128, threads_per_transform=17, runtime_compile=True),
    NS(length=343, factors=[7, 7, 7], scheme='CS_KERNEL_STOCKHAM_BLOCK_RC', workgroup_size=256, threads_per_transform=49, runtime_compile=True),
    NS(length=512, factors=[8, 8, 8], scheme='CS_KERNEL_STOCKHAM_BLOCK_RC', workgroup_size=512, threads_per_transform=128),
    NS(length=625, factors=[5, 5, 5, 5], scheme='CS_KERNEL_STOCKHAM_BLOCK_RC', workgroup_size=128, threads_per_transform=125, runtime_compile=True),
    NS(length=1331, factors=[11, 11, 11], scheme='CS_KERNEL_STOCKHAM_BLOCK_RC', workgroup_size=256, threads_per_transform=121, runtime_compile=True),
]
