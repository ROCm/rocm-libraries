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
    Janna/Fault_639
    DIMACS10/adaptive
    Janna/ML_Laplace
    Schenk/nlpkkt80
    SNAP/wiki-topcats
    ND/nd24k
    Belcastro/mouse_gene
    DIMACS10/rgg_n_2_21_s0
    JGD_GL7d/GL7d20
    Sinclair/3Dspectralwave
    DIMACS10/coPapersDBLP
    SNAP/soc-Pokec
    DIMACS10/coPapersCiteseer
    Dziekonski/dielFilterV3clx
    Mycielski/mycielskian16
    DIMACS10/road_central
    VLSI/ss
    VLSI/vas_stokes_1M
    DIMACS10/packing-500x100x100-b050
    JGD_GL7d/GL7d18
)

include(MatrixDownloadAndConvert.cmake)
