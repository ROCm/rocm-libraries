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
    Mittelmann/rail582
    Schenk_IBMNA/c-57
    GHS_indef/c-55
    Dehghani/light_in_tissue
    Mycielski/mycielskian12
    Nemeth/nemeth11
    Mittelmann/rail507
    DIMACS10/sd2010
    CPM/cz40948
    Hollinger/g7jac120sc
    Hollinger/g7jac120
    DIMACS10/nv2010
    Pothen/onera_dual
    SNAP/email-EuAll
    SNAP/cit-HepPh
    JGD_Homology/ch7-9-b3
    Bindel/ted_A
    Bindel/ted_A_unscaled
    GHS_indef/ncvxqp5
    IPSO/OPF_10000
)

include(MatrixDownloadAndConvert.cmake)
