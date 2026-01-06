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
    Pajek/dictionary28
    DIMACS10/t60k
    GHS_indef/brainpc2
    Schenk_IBMNA/c-50
    Meszaros/nemsemm2
    Goodwin/Goodwin_023
    Hollinger/g7jac060
    Hollinger/g7jac060sc
    Meszaros/sctap1-2r
    SNAP/ca-CondMat
    SNAP/wiki-RfA
    Meszaros/scsd8-2r
    Hollinger/jan99jac100
    Hollinger/jan99jac100sc
    Szczerba/Ill_Stokes
    Meszaros/nemswrld
    Sandia/mult_dcop_03
    Sandia/mult_dcop_02
    Sandia/mult_dcop_01
    Rajat/rajat22
)

include(MatrixDownloadAndConvert.cmake)
