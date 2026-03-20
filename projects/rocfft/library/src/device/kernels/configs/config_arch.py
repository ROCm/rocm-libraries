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

from enum import Enum


class lds_config(Enum):
    SIZE_64KiB = 64 * 1024
    SIZE_160KiB = 160 * 1024


class supported_arch(Enum):
    GFX_GENERIC = "gfx_generic"
    GFX_803 = "gfx803"
    GFX_900 = "gfx900"
    GFX_906 = "gfx906"
    GFX_908 = "gfx908"
    GFX_90A = "gfx90a"
    GFX_942 = "gfx942"
    GFX_950 = "gfx950"
    GFX_1030 = "gfx1030"
    GFX_1100 = "gfx1100"
    GFX_1101 = "gfx1101"
    GFX_1102 = "gfx1102"
    GFX_1150 = "gfx1150"
    GFX_1151 = "gfx1151"
    GFX_1152 = "gfx1152"
    GFX_1153 = "gfx1153"
    GFX_1200 = "gfx1200"
    GFX_1201 = "gfx1201"
