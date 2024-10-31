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
    GHS_indef/copter2
    GHS_indef/copter2
    Andrews/Andrews
    IPSO/HTC_336_9129
    Mallya/lhr34c
    IBM_EDA/dc1
    IBM_EDA/dc2
    IBM_EDA/dc3
    Lourakis/bundle1
    DIMACS10/ma2010
    DIMACS10/vsp_mod2_pgp2_slptsk
    Dattorro/EternityII_A
    IPSO/HTC_336_4438
    DIMACS10/delaunay_n17
    DIMACS10/ky2010
    JGD_Homology/m133-b3
    JGD_Homology/shar_te2-b3
    Chevron/Chevron2
    Grueninger/windtunnel_evap3d
    PowerSystem/power197k
)

include(MatrixDownloadAndConvert.cmake)
