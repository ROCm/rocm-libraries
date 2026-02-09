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
    Schenk_IBMNA/c-39
    HB/bcsstk15
    Sumner/graphics
    Meszaros/p010
    QCD/conf6_0-4x4-30
    QCD/conf5_0-4x4-22
    QCD/conf5_0-4x4-26
    QCD/conf5_0-4x4-18
    QCD/conf5_0-4x4-10
    QCD/conf6_0-4x4-20
    QCD/conf5_0-4x4-14
    Meszaros/scagr7-2r
    Pajek/foldoc
    Pothen/bodyy4
    Meszaros/sc205-2r
    Schenk_IBMNA/c-43
    DIMACS10/hi2010
    DIMACS10/vsp_c-30_data_data
    ML_Graph/indianpines_10NN
    Okunbor/aft01
)

include(MatrixDownloadAndConvert.cmake)
