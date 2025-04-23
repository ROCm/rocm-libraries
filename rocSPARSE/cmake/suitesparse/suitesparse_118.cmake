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
    Li/li
    Tromble/language
    DIMACS10/caidaRouterLevel
    DIMACS10/mn2010
    Mycielski/mycielskian13
    Boeing/msc10848
    DIMACS10/al2010
    SNAP/amazon0302
    VanVelzen/std1_Jac2
    Williams/mac_econ_fwd500
    DIMACS10/ok2010
    Schenk_IBMNA/c-73b
    Schenk_IBMNA/c-73
    DIMACS10/in2010
    Rajat/Raj1
    JGD_Homology/n4c6-b7
    NYPA/Maragal_8
    Sandia/ASIC_320ks
    Simon/raefsky4
    PARSEC/SiO
)

include(MatrixDownloadAndConvert.cmake)
