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
    Newman/cond-mat-2005
    FEMLAB/poisson3Da
    SNAP/cit-HepTh
    Pajek/HEP-th-new
    GHS_indef/a0nsdsil
    Schenk_IBMNA/c-53
    Boeing/bcsstk38
    JGD_GL7d/GL7d13
    LPnetlib/lp_ken_18
    JGD_BIBD/bibd_16_8
    Schenk_IBMNA/c-65
    Rost/RFdevice
    SNAP/email-Enron
    Garon/garon2
    JGD_Margulies/flower_8_4
    Hamm/bcircuit
    Hollinger/mark3jac140sc
    Hollinger/mark3jac140
    Cunningham/k3plates
    Mallya/lhr17
)

include(MatrixDownloadAndConvert.cmake)
