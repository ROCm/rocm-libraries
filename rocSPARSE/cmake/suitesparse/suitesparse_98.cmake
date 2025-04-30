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
    Meszaros/us04
    GHS_psdef/finance256
    Schenk_IBMNA/c-60
    Priebel/208bit
    GHS_indef/dixmaanl
    Cote/vibrobox
    Davis/FX_March2010
    FEMLAB/waveguide3D
    Qaplib/lp_nug20
    PARSEC/Na5
    Mallya/lhr14
    Bomhof/circuit_4
    Mallya/lhr14c
    Schenk_IBMNA/c-61
    JGD_Homology/n4c6-b5
    JGD_Homology/mk12-b4
    Goodwin/Goodwin_030
    Mittelmann/rail516
    Boeing/crystk01
    LPnetlib/lp_osa_14
)

include(MatrixDownloadAndConvert.cmake)
