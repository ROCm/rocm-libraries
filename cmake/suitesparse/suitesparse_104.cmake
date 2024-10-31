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
    DIMACS10/wy2010
    SNAP/loc-Brightkite
    GHS_indef/helm3d01
    HB/bcsstk17
    Botonakis/FEM_3D_thermal1
    Fluorem/GT01R
    Meszaros/t0331-4l
    Meszaros/stormg2-125
    GHS_indef/c-63
    GHS_indef/cont-201
    Guettel/TEM27623
    Rajat/rajat15
    Schenk_IBMNA/c-66b
    Schenk_IBMNA/c-66
    Nemeth/nemeth12
    FIDAP/ex40
    Pothen/tandem_dual
    Bai/af23560
    Bai/qc2534
    Averous/epb3
)

include(MatrixDownloadAndConvert.cmake)
