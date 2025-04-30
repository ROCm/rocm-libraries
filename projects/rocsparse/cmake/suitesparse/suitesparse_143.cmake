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
    Janna/Long_Coup_dt0
    Janna/Long_Coup_dt6
    DIMACS10/channel-500x100x100-b050
    DIMACS10/kron_g500-logn20
    Dziekonski/dielFilterV3real
    Schenk/nlpkkt120
    vanHeukelum/cage15
    Mycielski/mycielskian17
    DIMACS10/delaunay_n24
    DIMACS10/europe_osm
    Janna/ML_Geer
    LAW/hollywood-2009
    Janna/Flan_1565
    GenBank/kmer_V2a
    Janna/Cube_Coup_dt6
    Janna/Cube_Coup_dt0
    DIMACS10/rgg_n_2_23_s0
    Janna/Bump_2911
    VLSI/vas_stokes_4M
    GenBank/kmer_U1a
)

include(MatrixDownloadAndConvert.cmake)
