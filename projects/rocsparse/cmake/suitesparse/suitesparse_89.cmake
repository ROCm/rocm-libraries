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
    Hollinger/jan99jac080sc
    JGD_Kocay/Trec12
    ML_Graph/har_10NN
    Meszaros/cari
    Priebel/192bit
    Mallya/lhr07
    Meszaros/nsir
    VDOL/hangGlider_5
    DIMACS10/vt2010
    Meszaros/e18
    Mallya/lhr07c
    ML_Graph/kmnist_norm_10NN
    Schenk_IBMNA/c-49
    ML_Graph/Fashion_MNIST_norm_10NN
    HVDC/hvdc1
    HB/bcsstk24
    HB/lock3491
    Hollinger/mark3jac060sc
    Hollinger/mark3jac060
    VDOL/lowThrust_4
)

include(MatrixDownloadAndConvert.cmake)
