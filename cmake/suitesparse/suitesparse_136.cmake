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
    DNVS/troll
    Hardesty/Hardesty1
    DNVS/halfb
    DIMACS10/delaunay_n21
    Zaoui/kkt_power
    Sinclair/3Dspectralwave2
    TSOPF/TSOPF_FS_b300_c3
    BenElechi/BenElechi1
    Freescale/memchip
    Bodendiek/CurlCurl_3
    Harvard_Seismology/JP
    DIMACS10/hugetrace-00000
    DIMACS10/rgg_n_2_20_s0
    DIMACS10/italy_osm
    GHS_psdef/crankseg_2
    ND/nd12k
    Freescale/Freescale2
    JGD_GL7d/GL7d16
    Chen/pkustk14
    SNAP/higgs-twitter
)

include(MatrixDownloadAndConvert.cmake)
