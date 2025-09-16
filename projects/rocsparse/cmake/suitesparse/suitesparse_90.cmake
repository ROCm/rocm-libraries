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
    Meszaros/ulevimin
    Schenk_IBMNA/c-48
    Zhao/Zhao2
    Zhao/Zhao1
    Simon/raefsky5
    MathWorks/Muu
    JGD_Groebner/f855_mat9
    JGD_Groebner/f855_mat9_I
    PARSEC/SiH4
    JGD_G5/IG5-14
    GHS_indef/bratu3d
    Boeing/nasa2910
    Nasa/nasa2910
    Schenk_IBMNA/c-45
    Averous/epb2
    Oberwolfach/t2dah_e
    Oberwolfach/t2dah_a
    Oberwolfach/t2dah
    Wang/wang3
    Wang/wang4
)

include(MatrixDownloadAndConvert.cmake)
