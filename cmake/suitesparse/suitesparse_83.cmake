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
    Meszaros/lpl3
    Kemelmacher/Kemelmacher
    GHS_psdef/ford1
    Schenk_IBMNA/c-41
    Bai/mhd4800a
    YCheng/psse0
    CPM/cz10228
    Nasa/shuttle_eddy
    Pothen/shuttle_eddy
    SNAP/wiki-Vote
    Meszaros/r05
    Nasa/nasa4704
    Boeing/nasa4704
    Boeing/crystm01
    Rajat/rajat09
    SNAP/as-caida
    Hollinger/mark3jac040sc
    Hollinger/mark3jac040
    Meszaros/gen4
    Pothen/barth5
)

include(MatrixDownloadAndConvert.cmake)
