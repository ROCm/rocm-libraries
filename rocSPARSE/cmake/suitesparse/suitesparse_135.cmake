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
    VLSI/test1
    DNVS/m_t1
    Gleich/flickr
    GHS_psdef/hood
    DIMACS10/kron_g500-logn17
    Bourchtein/atmosmodm
    Bourchtein/atmosmodl
    Andrianov/mip1
    GHS_psdef/crankseg_1
    GHS_psdef/bmwcra_1
    PARSEC/Si87H76
    Mycielski/mycielskian15
    PARSEC/SiO2
    Mittelmann/rail4284
    GHS_indef/bmw3_2
    DNVS/fcondp2
    Kim/kim2
    Boeing/pwtk
    Meszaros/tp-6
    DNVS/fullb
)

include(MatrixDownloadAndConvert.cmake)
