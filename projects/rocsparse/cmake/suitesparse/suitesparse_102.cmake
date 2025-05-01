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
    DIMACS10/vsp_south31_slptsk
    DIMACS10/vsp_model1_crew1_cr42_south31
    Schenk_IBMNA/c-56
    Sanghavi/ecl32
    Quaglino/viscoplastic2
    Mallya/lhr17c
    Schenk_IBMNA/c-54
    Meszaros/fxm3_16
    DIMACS10/delaunay_n16
    Nemeth/nemeth02
    Nemeth/nemeth05
    Nemeth/nemeth04
    Nemeth/nemeth03
    Nemeth/nemeth06
    Nemeth/nemeth07
    Nemeth/nemeth08
    Nemeth/nemeth09
    SNAP/ca-AstroPh
    Andrianov/net25
    Nemeth/nemeth10
)

include(MatrixDownloadAndConvert.cmake)
