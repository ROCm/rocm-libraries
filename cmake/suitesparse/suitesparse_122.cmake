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
    Lee/fem_filter
    JGD_BIBD/bibd_18_9
    QLi/majorbasis
    QLi/crashbasis
    Boeing/pcrystk03
    Boeing/crystk03
    Lin/Lin
    DIMACS10/oh2010
    Meszaros/karted
    Simon/bbmat
    JGD_Homology/n4c6-b8
    JGD_G5/IG5-18
    POLYFLOW/invextr1_new
    Goodwin/Goodwin_071
    TSOPF/TSOPF_FS_b162_c3
    Rothberg/cfd1
    JGD_GL7d/GL7d14
    Mittelmann/watson_2
    SNAP/com-Amazon
    Simon/appu
)

include(MatrixDownloadAndConvert.cmake)
