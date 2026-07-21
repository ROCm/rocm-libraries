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
    Oberwolfach/t3dl_a
    Oberwolfach/t3dl
    GHS_psdef/gridgena
    Hamm/hcircuit
    SNAP/soc-sign-Slashdot081106
    Bindel/ted_AB_unscaled
    Bindel/ted_AB
    Schenk_IBMNA/c-67
    Schenk_IBMNA/c-67b
    VanVelzen/std1_Jac3_db
    Schenk_IBMSDS/ibm_matrix_2
    Schenk_IBMSDS/3D_51448_3D
    NYPA/Maragal_6
    Nemeth/nemeth15
    HB/psmigr_2
    GHS_psdef/apache1
    HB/psmigr_3
    HB/psmigr_1
    GHS_psdef/ford2
    SNAP/soc-sign-Slashdot090216
)

include(MatrixDownloadAndConvert.cmake)
