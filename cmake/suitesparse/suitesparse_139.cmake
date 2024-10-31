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
    DIMACS10/hugetric-00020
    Dziekonski/gsm_106857
    SNAP/as-Skitter
    DIMACS10/333SP
    Buss/12month1
    DIMACS10/AS365
    Janna/Transport
    JGD_Relat/rel9
    Belcastro/human_gene1
    DIMACS10/germany_osm
    DIMACS10/NLR
    DIMACS10/delaunay_n22
    Dziekonski/dielFilterV2clx
    DIMACS10/asia_osm
    JGD_GL7d/GL7d17
    Bodendiek/CurlCurl_4
    VLSI/dgreen
    Freescale/FullChip
    Koutsovasilis/F1
    vanHeukelum/cage14
)

include(MatrixDownloadAndConvert.cmake)
