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
    Williams/mc2depi
    AG-Monien/wave
    Schenk_ISEI/barrier2-1
    Schenk_ISEI/barrier2-4
    Schenk_ISEI/barrier2-3
    Schenk_ISEI/barrier2-2
    GHS_psdef/oilpan
    Schenk_ISEI/barrier2-9
    Schenk_ISEI/barrier2-12
    Schenk_ISEI/barrier2-11
    Schenk_ISEI/barrier2-10
    Gupta/gupta1
    DIMACS10/il2010
    JGD_Margulies/wheel_601
    Chen/pkustk05
    PARSEC/H2O
    GHS_indef/exdata_1
    Kamvar/Stanford
    SNAP/web-Stanford
)

include(MatrixDownloadAndConvert.cmake)
