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
    Freescale/circuit5M_dc
    Pajek/patents
    PARSEC/Si41Ge41H72
    DIMACS10/venturiLevel3
    TSOPF/TSOPF_RS_b2383_c1
    TSOPF/TSOPF_RS_b2383
    DIMACS10/great-britain_osm
    SNAP/cit-Patents
    LAW/in-2004
    Freescale/Freescale1
    Janna/CoupCons3D
    DIMACS10/hugetric-00000
    Schenk_AFE/af_3_k101
    Schenk_AFE/af_2_k101
    Schenk_AFE/af_1_k101
    Schenk_AFE/af_0_k101
    Schenk_AFE/af_5_k101
    Schenk_AFE/af_4_k101
    Schenk_AFE/af_shell2
    Schenk_AFE/af_shell1
)

include(MatrixDownloadAndConvert.cmake)
