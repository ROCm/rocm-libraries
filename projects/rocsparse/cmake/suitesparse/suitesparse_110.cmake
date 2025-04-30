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
    DIMACS10/wv2010
    VanVelzen/Zd_Jac6_db
    VanVelzen/Zd_Jac2_db
    TKK/cbuckle
    Andrianov/ex3sta1
    Norris/heart2
    Norris/heart3
    JGD_BIBD/bibd_17_8
    TSOPF/TSOPF_RS_b39_c19
    DIMACS10/rgg_n_2_16_s0
    Mittelmann/neos2
    Nemeth/nemeth18
    Meszaros/nsct
    DIMACS10/md2010
    JGD_Homology/ch7-8-b4
    GHS_indef/c-72
    Schenk_IBMNA/c-64b
    Schenk_IBMNA/c-64
    Botonakis/thermomech_TC
    Botonakis/thermomech_TK
)

include(MatrixDownloadAndConvert.cmake)
