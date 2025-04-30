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
    JGD_CAG/CAG_mat1916
    DIMACS10/vsp_befref_fxm_2_4_air02
    Nasa/skirt
    Pothen/skirt
    DIMACS10/delaunay_n15
    AG-Monien/bfly
    GHS_psdef/torsion1
    GHS_psdef/obstclae
    VDOL/lowThrust_5
    PARSEC/SiNa
    Meszaros/world
    GHS_psdef/jnlbrng1
    Meszaros/mod2
    Watson/chem_master1
    JGD_Margulies/flower_7_4
    Schenk_IBMNA/c-52
    Schenk_IBMNA/c-51
    DIMACS10/vsp_barth5_1Ksep_50in_5Kout
    GHS_psdef/minsurfo
    GHS_psdef/mario001
)

include(MatrixDownloadAndConvert.cmake)
