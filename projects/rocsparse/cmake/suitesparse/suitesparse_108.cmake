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
    DIMACS10/ut2010
    Schmid/thermal1
    GHS_indef/ncvxqp7
    Sandia/ASIC_100ks
    Boeing/crystm03
    JGD_Forest/TF17
    Nemeth/nemeth16
    Meszaros/bas1lp
    JGD_G5/IG5-16
    Meszaros/stat96v1
    Bydder/mri1
    Mittelmann/pds-50
    HB/bcsstk33
    LeGresley/LeGresley_87936
    JGD_GL7d/GL7d24
    SNAP/sx-askubuntu
    Mulvey/finan512
    Mulvey/pfinan512
    Mittelmann/neos1
    JGD_Homology/shar_te2-b2
)

include(MatrixDownloadAndConvert.cmake)
