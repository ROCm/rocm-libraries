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
    GHS_indef/spmsrtls
    Mallya/lhr11
    Mallya/lhr10c
    Mittelmann/fome20
    LPnetlib/lp_pds_20
    Mallya/lhr11c
    Meszaros/stat96v5
    DIMACS10/nh2010
    JGD_Homology/ch7-8-b3
    SNAP/ca-HepPh
    Schenk_ISEI/nmos3
    Rothberg/struct4
    TAMU_SmartGridCenter/ACTIVSg70K
    DIMACS10/luxembourg_osm
    SNAP/sx-mathoverflow
    Newman/cond-mat-2003
    ML_Graph/worms20_10NN
    Newman/astro-ph
    PARSEC/benzene
    DIMACS10/wing
)

include(MatrixDownloadAndConvert.cmake)
