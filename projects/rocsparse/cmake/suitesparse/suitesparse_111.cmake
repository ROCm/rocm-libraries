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
    Yoshiyasu/image_interp
    VanVelzen/Zd_Jac3_db
    TKK/cyl6
    ACUSIM/Pres_Poisson
    Hollinger/g7jac200
    Hollinger/g7jac200sc
    Mittelmann/pds-60
    Nemeth/nemeth01
    Andrianov/pf2177
    AMD/G2_circuit
    DIMACS10/id2010
    JGD_Homology/n4c6-b6
    TSOPF/TSOPF_FS_b39_c7
    AG-Monien/brack2
    PARSEC/Si5H12
    GHS_indef/olesnik0
    Mallya/lhr34
    Watson/Baumann
    IBM_EDA/trans5
    IBM_EDA/trans4
)

include(MatrixDownloadAndConvert.cmake)
