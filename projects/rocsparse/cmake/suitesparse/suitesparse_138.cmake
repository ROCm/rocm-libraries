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
    Schenk_AFE/af_shell3
    Schenk_AFE/af_shell4
    Schenk_AFE/af_shell6
    Schenk_AFE/af_shell8
    Schenk_AFE/af_shell5
    Schenk_AFE/af_shell7
    Schenk_AFE/af_shell9
    Belcastro/human_gene2
    JGD_GL7d/GL7d21
    PARSEC/Ga41As41H72
    INPRO/msdoor
    LAW/eu-2005
    Gleich/wikipedia-20051105
    DIMACS10/hugetric-00010
    Mazaheri/bundle_adj
    Lee/fem_hifreq_circuit
    Rajat/rajat31
    DIMACS10/M6
    Janna/StocF-1465
    DIMACS10/kron_g500-logn18
)

include(MatrixDownloadAndConvert.cmake)
