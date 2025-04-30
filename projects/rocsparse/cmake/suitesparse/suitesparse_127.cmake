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
    Sandia/ASIC_680k
    Freescale/nxp1
    Nasa/nasasrb
    JGD_GL7d/GL7d23
    Boeing/pct20stif
    Oberwolfach/filter3D
    GHS_indef/helm2d03
    Andrianov/ins2
    Botonakis/thermomech_dK
    Meszaros/stat96v2
    GHS_psdef/ramage02
    JGD_Kocay/Trec14
    Schenk_ISEI/para-4
    TSOPF/TSOPF_RS_b300_c2
    GHS_psdef/srb1
    Norris/stomach
    SNAP/roadNet-PA
    Rothberg/cfd2
    DIMACS10/rgg_n_2_18_s0
    DIMACS10/belgium_osm
)

include(MatrixDownloadAndConvert.cmake)
