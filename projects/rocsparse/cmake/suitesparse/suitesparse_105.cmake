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
    Mittelmann/fome21
    Mittelmann/pds-40
    JGD_Homology/ch8-8-b3
    GHS_psdef/wathen100
    Nemeth/nemeth13
    Rajat/rajat16
    Rajat/rajat18
    Rajat/rajat17
    GHS_indef/c-59
    DIMACS10/vsp_c-60_data_cti_cs4
    Schenk_IBMSDS/2D_54019_highK
    Hollinger/g7jac140
    Hollinger/g7jac140sc
    Meszaros/stat96v4
    Norris/lung2
    Nemeth/nemeth14
    Andrianov/fxm4_6
    VanVelzen/std1_Jac2_db
    GHS_indef/ncvxqp3
    SNAP/soc-Epinions1
)

include(MatrixDownloadAndConvert.cmake)
