#!/usr/bin/cmake -P
# ########################################################################
# Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ########################################################################

set(TEST_MATRICES
    Cylshell/s1rmt3m1
    Cylshell/s3rmt3m1
    Cylshell/s2rmt3m1
    Puri/ABACUS_shell_ud
    Puri/ABACUS_shell_hd
    Puri/ABACUS_shell_ld
    Puri/ABACUS_shell_md
    HB/bcsstk28
    VDOL/lowThrust_9
    Meszaros/scfxm1-2r
    VDOL/lowThrust_10
    ATandT/onetone2
    VDOL/lowThrust_11
    VDOL/lowThrust_12
    VDOL/lowThrust_13
    HB/man_5976
    FIDAP/ex35
    Mallya/lhr10
    Hollinger/jan99jac120
    Hollinger/jan99jac120sc
)

include(MatrixDownloadAndConvert.cmake)
