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
    SNAP/amazon0601
    Chevron/Chevron3
    Mittelmann/stormG2_1000
    Botonakis/FEM_3D_thermal2
    DIMACS10/ca2010
    DNVS/shipsec1
    Embree/ifiss_mat
    UTEP/Dubcova3
    Wissgott/parabolic_fem
    GHS_psdef/s3dkt3m2
    Mycielski/mycielskian14
    TKK/smt
    TKK/s4dkt3m2
    Rajat/rajat29
    DNVS/ship_003
    Pajek/IMDB
    Raju/laminar_duct3D
    SNAP/roadNet-TX
    Ronis/xenon2
    DNVS/ship_001
)

include(MatrixDownloadAndConvert.cmake)
