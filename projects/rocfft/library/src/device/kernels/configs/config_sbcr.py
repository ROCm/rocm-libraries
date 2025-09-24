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

# NB:
# Technically, we could have SBCR kernels the same amount as SBCC.
#
# sbcr_kernels = copy.deepcopy(sbcc_kernels)
# for k in sbcr_kernels:
#     k.scheme = 'CS_KERNEL_STOCKHAM_BLOCK_CR'
#

# for SBCR, if direct_to_from_reg is True, we do load-to-reg, but will not do store-from-reg
#           And since sbcr is dir-to-reg BUT NOT dir-from-reg, the global store part requires full LDS
#           So, we can't satifly half_lds in SBCR !

# yapf: disable
sbcr_kernels = [
    NS(length=56,  factors=[7, 8], direct_to_from_reg=False),
    NS(length=100, factors=[10, 10], workgroup_size=100),
    NS(length=200, factors=[8, 5, 5]),
    NS(length=336, factors=[6, 7, 8])
]
