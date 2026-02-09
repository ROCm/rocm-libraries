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
    LPnetlib/lp_cre_d
    Toledo/deltaX
    Rajat/rajat26
    HB/bcsstk25
    TSOPF/TSOPF_RS_b39_c7
    UTEP/Dubcova1
    Pothen/tandem_vtx
    GHS_indef/a5esindl
    JGD_BIBD/bibd_81_3
    FIDAP/ex19
    Hollinger/g7jac080
    Hollinger/g7jac080sc
    Engwirda/airfoil_2d
    LPnetlib/lp_cre_b
    Cylshell/s1rmq4m1
    Cylshell/s3rmq4m1
    Cylshell/s2rmq4m1
    Meszaros/rlfddd
    Meszaros/fxm4_6
    Hollinger/mark3jac100sc
)

include(MatrixDownloadAndConvert.cmake)
