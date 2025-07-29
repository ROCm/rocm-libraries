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

# Note: Default direct_to_from_reg is True

# yapf: disable
sbcc_kernels = [
    NS(length=50,  factors=[10, 5],      use_3steps_large_twd={
        'sp': 'true',  'dp': 'true'}, workgroup_size=256),
    NS(length=52,  factors=[13, 4],      use_3steps_large_twd={
        'sp': 'true',  'dp': 'true'}),
    NS(length=60,  factors=[6, 10],      use_3steps_large_twd={
        'sp': 'false',  'dp': 'false'}),
    NS(length=64,  factors=[8, 8],       use_3steps_large_twd={
        'sp': 'true',  'dp': 'false'}, workgroup_size=256),
    NS(length=72,  factors=[8, 3, 3],    use_3steps_large_twd={
        'sp': 'true',  'dp': 'false'}),
    NS(length=80,  factors=[10, 8],      use_3steps_large_twd={
        'sp': 'false',  'dp': 'false'}),
    # 9,9 is good when direct-to-reg, but bad for Navi, so still uses radix-3
    NS(length=81,  factors=[3, 3, 3, 3], use_3steps_large_twd={
        'sp': 'true',  'dp': 'true'}),
    NS(length=84,  factors=[7, 2, 6],    use_3steps_large_twd={
        'sp': 'true',  'dp': 'true'}, threads_per_transform=14),
    NS(length=96,  factors=[8, 3, 4],    use_3steps_large_twd={
        'sp': 'false',  'dp': 'false'}, workgroup_size=256),
    NS(length=100, factors=[5, 5, 4],    use_3steps_large_twd={
        'sp': 'true',  'dp': 'false'}, workgroup_size=100, half_lds=True),
    NS(length=104, factors=[13, 8],      use_3steps_large_twd={
        'sp': 'true',  'dp': 'false'}),
    NS(length=108, factors=[6, 6, 3],    use_3steps_large_twd={
        'sp': 'true',  'dp': 'false'}),
    NS(length=112, factors=[4, 7, 4],    use_3steps_large_twd={
        'sp': 'false',  'dp': 'false'}),
    NS(length=121, factors=[11, 11],    use_3steps_large_twd={
        'sp': 'true',  'dp': 'true'}, workgroup_size=128, runtime_compile=True),
    NS(length=125, factors=[5, 5, 5],    use_3steps_large_twd={
        'sp': 'true',  'dp': 'false'}),
    NS(length=128, factors=[16, 8],    use_3steps_large_twd={
        'sp': 'true',  'dp': 'true'}, workgroup_size=256, threads_per_transform= 16),
    NS(length=160, factors=[4, 10, 4],   use_3steps_large_twd={
        'sp': 'false', 'dp': 'false'}, flavour='wide'),
    NS(length=168, factors=[7, 6, 4],    use_3steps_large_twd={
        'sp': 'true', 'dp': 'false'}, workgroup_size=128, half_lds=True),
    NS(length=169, factors=[13, 13],    use_3steps_large_twd={
        'sp': 'true', 'dp': 'false'}, workgroup_size=256, runtime_compile=True),
    NS(length=192, factors=[8, 6, 4],    use_3steps_large_twd={
        'sp': 'true', 'dp': 'true'}),
    NS(length=200, factors=[5, 8, 5],    use_3steps_large_twd={
        'sp': 'false', 'dp': 'false'}),
    NS(length=208, factors=[13, 16],     use_3steps_large_twd={
        'sp': 'false', 'dp': 'false'}),
    NS(length=216, factors=(6, 6, 6), use_3steps_large_twd={
        'sp': 'false', 'dp': 'false'}, threads_per_transform=36),
    NS(length=224, factors=[8, 7, 4],    use_3steps_large_twd={
        'sp': 'true', 'dp': 'false'}),
    NS(length=240, factors=[8, 5, 6],    use_3steps_large_twd={
        'sp': 'false', 'dp': 'false'}),
    # 9,9,3 isn't better on all archs, some are much better, some get regressions
    NS(length=243, factors=[3, 3, 3, 3, 3],    use_3steps_large_twd={
        'sp': 'true', 'dp': 'false'}, workgroup_size=243),
    NS(length=256, factors=[8, 4, 8], use_3steps_large_twd={
        'sp': 'true',  'dp': 'false'}, flavour='wide'),
    NS(length=280, factors=[8, 5, 7], use_3steps_large_twd={
        'sp': 'false',  'dp': 'false'}, runtime_compile=True),
    NS(length=289, factors=[17, 17],    use_3steps_large_twd={
        'sp': 'true', 'dp': 'true'}, runtime_compile=True),
    NS(length=336, factors=[6, 7, 8],    use_3steps_large_twd={
        'sp': 'false', 'dp': 'false'}),
    NS(length=343, factors=[7, 7, 7],    use_3steps_large_twd={
        'sp': 'true', 'dp': 'true'}),
    NS(length=512, factors=[8, 8, 8],    use_3steps_large_twd={
        'sp': 'true', 'dp': 'false'}),
]
