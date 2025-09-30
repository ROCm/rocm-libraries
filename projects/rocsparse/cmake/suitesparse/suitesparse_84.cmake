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
    Hollinger/g7jac040
    Hollinger/g7jac040sc
    LPnetlib/lp_pds_10
    DIMACS10/vsp_p0291_seymourl_iiasa
    Grueninger/windtunnel_evap2d
    Meszaros/co9
    GHS_indef/bloweybl
    Schenk_IBMNA/c-42
    Meszaros/scfxm1-2b
    Meszaros/baxter
    DIMACS10/vsp_data_and_seymourl
    Hollinger/jan99jac060sc
    Hollinger/jan99jac060
    Meszaros/south31
    Meszaros/scsd8-2c
    Meszaros/scsd8-2b
    Bodendiek/CurlCurl_0
    GHS_indef/cvxqp3
    YCheng/psse2
    DIMACS10/de2010
)

include(MatrixDownloadAndConvert.cmake)
