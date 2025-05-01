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
    JGD_Homology/n4c6-b9
    Rajat/rajat21
    JGD_Homology/ch8-8-b4
    VLSI/power9
    SNAP/loc-Gowalla
    VanVelzen/Zd_Jac3
    QCD/conf5_4-8x8-15
    QCD/conf5_4-8x8-10
    QCD/conf5_4-8x8-05
    QCD/conf5_4-8x8-20
    QCD/conf6_0-8x8-20
    QCD/conf6_0-8x8-30
    QCD/conf6_0-8x8-80
    GHS_psdef/opt1
    Sandia/ASIC_320k
    DNVS/trdheim
    Rajat/rajat24
    DIMACS10/coAuthorsDBLP
    TSOPF/TSOPF_FS_b39_c19
    POLYFLOW/mixtank_new
)

include(MatrixDownloadAndConvert.cmake)
