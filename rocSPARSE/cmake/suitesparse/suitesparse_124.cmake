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
    Precima/analytics
    IPSO/TSC_OPF_1047
    HB/bcsstk32
    vanHeukelum/cage12
    Andrianov/net100
    Rudnyi/water_tank
    HB/bcsstk30
    Mittelmann/neos3
    DIMACS10/pa2010
    Boeing/bcsstk39
    FEMLAB/sme3Db
    Schenk_ISEI/para-10
    Schenk_ISEI/para-5
    Schenk_ISEI/para-7
    Schenk_ISEI/para-9
    Schenk_ISEI/para-6
    Schenk_ISEI/para-8
    GHS_indef/darcy003
    GHS_indef/mario002
    SNAP/com-DBLP
)

include(MatrixDownloadAndConvert.cmake)
