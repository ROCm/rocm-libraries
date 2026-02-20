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
    Mancktelow/viscorocks
    Boeing/bcsstk37
    Boeing/msc23052
    Boeing/bcsstk36
    Pereyra/landmark
    Castrillon/denormal
    Meszaros/dbir2
    Dattorro/EternityII_Etilde
    Rothberg/struct3
    Nemeth/nemeth21
    Ronis/xenon1
    HB/bcsstk31
    DIMACS10/tn2010
    DIMACS10/az2010
    NYPA/Maragal_7
    Schenk_IBMSDS/matrix_9
    ATandT/twotone
    Sorensen/Linux_call_graph
    DIMACS10/wi2010
    GHS_indef/boyd1
)

include(MatrixDownloadAndConvert.cmake)
