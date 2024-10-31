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
    Hollinger/mark3jac100
    Meszaros/rat
    IPSO/OPF_6000
    Grund/bayer01
    LAW/enron
    HB/cegb2802
    Oberwolfach/chipcool1
    Oberwolfach/chipcool0
    DIMACS10/vsp_sctap1-2b_and_seymourl
    Meszaros/rlfdual
    Hohn/sinc12
    Mittelmann/fome13
    Schenk_IBMSDS/3D_28984_Tetra
    TKK/t520
    HB/bcsstk16
    Simon/raefsky1
    Simon/raefsky2
    JGD_SPG/EX6
    JGD_SPG/EX5
    Pajek/Reuters911
)

include(MatrixDownloadAndConvert.cmake)
