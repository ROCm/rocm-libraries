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
    Chen/pkustk02
    JGD_Homology/mk13-b5
    TSOPF/TSOPF_RS_b162_c4
    Nemeth/nemeth19
    DIMACS10/fe_ocean
    IPSO/TSC_OPF_300
    JGD_Relat/rel8
    DIMACS10/nj2010
    JGD_Homology/n4c6-b11
    DIMACS10/nm2010
    Mittelmann/sgpf5y6
    Mittelmann/pds-70
    MKS/fp
    DIMACS10/ms2010
    SNAP/soc-sign-epinions
    VLSI/ss1
    JGD_Homology/ch7-8-b5
    Yoshiyasu/mesh_deform
    GHS_indef/c-71
    DIMACS10/vsp_vibrobox_scagr7-2c_rlfddd
)

include(MatrixDownloadAndConvert.cmake)
