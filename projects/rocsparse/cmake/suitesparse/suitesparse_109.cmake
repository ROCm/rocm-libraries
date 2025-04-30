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
    DIMACS10/vsp_bump2_e18_aa01_model1_crew1
    Rajat/rajat20
    LPnetlib/lp_osa_30
    Rajat/rajat25
    Rajat/rajat28
    TSOPF/TSOPF_FS_b162_c1
    TSOPF/TSOPF_RS_b162_c3
    ML_Graph/k49_norm_10NN
    HB/bcsstk29
    GHS_indef/c-69
    DIMACS10/nd2010
    Nemeth/nemeth17
    DIMACS10/mt2010
    GHS_indef/blockqp1
    Hollinger/g7jac180sc
    Hollinger/g7jac180
    Clark/tomographic1
    ANSYS/Delor64K
    JGD_Kocay/Trec13
    GHS_indef/c-70
)

include(MatrixDownloadAndConvert.cmake)
