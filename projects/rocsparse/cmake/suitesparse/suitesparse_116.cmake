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
    Simon/olafu
    Oberwolfach/gyro
    Oberwolfach/gyro_k
    DIMACS10/ia2010
    UTEP/Dubcova2
    Goodwin/Goodwin_054
    Norris/torso2
    JGD_G5/IG5-17
    QY/case39
    Meszaros/nemsemm1
    Mittelmann/watson_1
    Meszaros/ts-palko
    Meszaros/dbir1
    TSOPF/TSOPF_RS_b39_c30
    Meszaros/dbic1
    Mittelmann/pds-100
    FIDAP/ex11
    DIMACS10/vsp_finan512_scagr7-2c_rlfddd
    DIMACS10/ks2010
    Buss/connectus
)

include(MatrixDownloadAndConvert.cmake)
