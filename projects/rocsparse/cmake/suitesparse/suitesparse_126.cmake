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
    DIMACS10/citationCiteseer
    GHS_psdef/vanbody
    Bova/rma10
    Schenk_IBMNA/c-big
    DIMACS10/fl2010
    FEMLAB/poisson3Db
    TSOPF/TSOPF_FS_b162_c4
    ANSYS/Delor295K
    Chen/pkustk07
    Andrianov/net4-1
    DIMACS10/vsp_msc10848_300sep_100in_1Kout
    DNVS/tsyl201
    JGD_Groebner/c8_mat11_I
    JGD_Groebner/c8_mat11
    Bodendiek/CurlCurl_1
    JGD_Homology/ch7-9-b5
    Chen/pkustk06
    Andrianov/net125
    Boeing/ct20stif
    Williams/cop20k_A
)

include(MatrixDownloadAndConvert.cmake)
