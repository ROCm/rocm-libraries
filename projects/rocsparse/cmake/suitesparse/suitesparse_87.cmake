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
    TAMU_SmartGridCenter/ACTIVSg10K
    JGD_Taha/abtaha2
    DRIVCAV/cavity16
    DRIVCAV/cavity26
    DRIVCAV/cavity24
    DRIVCAV/cavity22
    DRIVCAV/cavity20
    DRIVCAV/cavity18
    Oberwolfach/rail_20209
    AG-Monien/cca
    GHS_indef/stokes64s
    GHS_indef/stokes64
    IPSO/OPF_3754
    Cote/mplate
    Mittelmann/fome12
    AG-Monien/shock-9
    Meszaros/pltexpa
    Bindel/ted_B_unscaled
    Bindel/ted_B
    LPnetlib/lp_osa_07
)

include(MatrixDownloadAndConvert.cmake)
