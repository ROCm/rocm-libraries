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
    TSOPF/TSOPF_RS_b162_c1
    CPM/cz20468
    Schenk_IBMSDS/2D_27628_bjtcai
    Meszaros/route
    Cylshell/s3rmt3m3
    Pajek/internet
    Brethour/coater2
    VDOL/lowThrust_6
    TKK/g3rmt3m3
    JGD_Homology/mk12-b3
    GHS_psdef/copter1
    Schenk_IBMNA/c-47
    VDOL/lowThrust_7
    JGD_Homology/ch7-7-b5
    Meszaros/kl02
    Hollinger/mark3jac080sc
    Hollinger/mark3jac080
    JGD_Forest/TF16
    VDOL/lowThrust_8
    DIMACS10/ak2010
)

include(MatrixDownloadAndConvert.cmake)
