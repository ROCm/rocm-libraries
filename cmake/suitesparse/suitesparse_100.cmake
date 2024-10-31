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
    JGD_Franz/Franz11
    Shyy/shyy161
    Chevron/Chevron1
    Gleich/usroads
    IBM_EDA/ckt11752_tr_0
    JGD_Homology/n4c6-b12
    IBM_EDA/ckt11752_dc_1
    Graham/graham1
    DIMACS10/me2010
    ATandT/onetone1
    Hollinger/g7jac100
    Hollinger/g7jac100sc
    DIMACS10/ct2010
    MathWorks/Kuu
    Oberwolfach/gyro_m
    Mittelmann/pds-30
    Pajek/HEP-th
    GHS_indef/a2nnsnsl
    GHS_indef/cvxbqp1
    GHS_indef/ncvxbqp1
)

include(MatrixDownloadAndConvert.cmake)
