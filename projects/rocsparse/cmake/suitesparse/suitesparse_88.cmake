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
    LPnetlib/lp_maros_r7
    Rommes/juba40k
    Rommes/bauru5727
    Shen/shermanACb
    Hollinger/g7jac050sc
    ML_Graph/mnist_test_norm_10NN
    JGD_Homology/D6-6
    AG-Monien/ccc
    SNAP/p2p-Gnutella31
    QY/case9
    TSOPF/TSOPF_FS_b9_c6
    Mittelmann/nug08-3rd
    HB/bcsstk18
    VDOL/hangGlider_4
    GHS_psdef/pds10
    GHS_psdef/bloweya
    Meszaros/model10
    vanHeukelum/cage10
    DIMACS10/wing_nodal
    Hollinger/jan99jac080
)

include(MatrixDownloadAndConvert.cmake)
