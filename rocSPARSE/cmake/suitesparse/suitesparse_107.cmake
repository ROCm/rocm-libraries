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
    SNAP/soc-sign-Slashdot090221
    Hohn/sinc15
    GHS_indef/c-58
    Shen/e40r0100
    Oberwolfach/rail_79841
    JGD_Trefethen/Trefethen_20000b
    JGD_Trefethen/Trefethen_20000
    Rajat/rajat23
    GHS_indef/stokes128
    GHS_indef/c-62ghs
    Schenk_IBMNA/c-62
    vanHeukelum/cage11
    GHS_indef/k1_san
    Pajek/patents_main
    Goodwin/Goodwin_040
    Hollinger/g7jac160
    Hollinger/g7jac160sc
    GHS_psdef/wathen120
    GHS_psdef/c-68
    Bydder/mri2
)

include(MatrixDownloadAndConvert.cmake)
