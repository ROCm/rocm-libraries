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
    Williams/webbase-1M
    TSOPF/TSOPF_FS_b39_c30
    Andrianov/net150
    Chen/pkustk03
    DIMACS10/delaunay_n19
    FEMLAB/sme3Dc
    SNAP/amazon0312
    Rothberg/3dtube
    LAW/cnr-2000
    Goodwin/Goodwin_095
    Chen/pkustk08
    ND/nd3k
    DNVS/shipsec8
    SNAP/wiki-talk-temporal
    Meszaros/stat96v3
    JGD_BIBD/bibd_19_9
    SNAP/amazon0505
    DIMACS10/m14b
    PARSEC/GaAsH6
    JGD_Homology/ch8-8-b5
)

include(MatrixDownloadAndConvert.cmake)
