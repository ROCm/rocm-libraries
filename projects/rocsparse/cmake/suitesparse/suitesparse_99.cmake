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
    DIMACS10/rgg_n_2_15_s0
    Meszaros/rlfprim
    Boeing/bcsstm36
    HB/cegb2919
    Hollinger/mark3jac120
    Hollinger/mark3jac120sc
    Boeing/crystm02
    JGD_G5/IG5-15
    Gleich/usroads-48
    Goodwin/goodwin
    Pajek/EAT_SR
    Pajek/EAT_RS
    Pothen/pwt
    Nasa/pwt
    GHS_psdef/pwt
    DIMACS10/fe_body
    MaxPlanck/shallow_water2
    MaxPlanck/shallow_water1
    Andrianov/lpl1
    Oberwolfach/inlet
)

include(MatrixDownloadAndConvert.cmake)
