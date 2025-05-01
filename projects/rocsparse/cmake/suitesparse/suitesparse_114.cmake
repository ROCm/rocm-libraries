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
    FEMLAB/sme3Da
    PARSEC/Si10H16
    DIMACS10/sc2010
    Schenk_IBMSDS/matrix-new_3
    TKK/tube2
    TKK/tube1
    DIMACS10/ar2010
    Meszaros/nw14
    DIMACS10/fe_tooth
    SNAP/soc-Slashdot0811
    DIMACS10/ne2010
    SNAP/sx-superuser
    Mittelmann/pds-80
    Barabasi/NotreDame_www
    Kim/kim1
    JGD_Groebner/HFE18_96_in
    Sandia/ASIC_100k
    Andrianov/net50
    DIMACS10/wa2010
    SNAP/soc-Slashdot0902
)

include(MatrixDownloadAndConvert.cmake)
