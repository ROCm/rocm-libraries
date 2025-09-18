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
    Chevron/Chevron4
    Guettel/TEM152078
    DIMACS10/rgg_n_2_19_s0
    Chen/pkustk13
    DIMACS10/auto
    TSOPF/TSOPF_RS_b2052_c1
    Bates/sls
    Schenk_ISEI/ohne2
    ND/nd6k
    Mittelmann/cont1_l
    GHS_psdef/bmw7st_1
    vanHeukelum/cage13
    Chen/pkustk12
    Kamvar/Stanford_Berkeley
    SNAP/web-BerkStan
    Brogan/specular
    AMD/G3_circuit
    PARSEC/CO
    Rucci/Rucci1
    Guettel/TEM181302
)

include(MatrixDownloadAndConvert.cmake)
