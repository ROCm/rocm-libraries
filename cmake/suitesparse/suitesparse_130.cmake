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
    Williams/cant
    Hardesty/Hardesty2
    DIMACS10/vsp_bcsstk30_500sep_10in_1Kout
    Meng/iChem_Jacobian
    AG-Monien/debr
    ANSYS/Delor338K
    Chen/pkustk04
    Um/offshore
    Gupta/gupta2
    Chen/pkustk10
    Williams/pdb1HYS
    Oberwolfach/t3dh
    Oberwolfach/t3dh_a
    Oberwolfach/t3dh_e
    JGD_Forest/TF19
    TSOPF/TSOPF_RS_b678_c1
    TSOPF/TSOPF_FS_b300_c1
    TSOPF/TSOPF_FS_b300
    TSOPF/TSOPF_RS_b300_c3
    GHS_psdef/s3dkq4m2
)

include(MatrixDownloadAndConvert.cmake)
