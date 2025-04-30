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
    Dziekonski/dielFilterV2real
    DIMACS10/delaunay_n23
    Schenk_AFE/af_shell10
    DIMACS10/hugebubbles-00000
    Gleich/wb-edu
    GAP/GAP-road
    DIMACS10/road_usa
    DIMACS10/hugebubbles-00010
    Janna/Hook_1498
    Freescale/circuit5M
    Janna/Geo_1438
    DIMACS10/rgg_n_2_22_s0
    DIMACS10/hugebubbles-00020
    Janna/Serena
    VLSI/vas_stokes_2M
    SNAP/soc-LiveJournal1
    SNAP/com-LiveJournal
    MAWI/mawi_201512020000
    GHS_psdef/audikw_1
    LAW/ljournal-2008
)

include(MatrixDownloadAndConvert.cmake)
