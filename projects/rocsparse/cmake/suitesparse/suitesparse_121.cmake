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
    LAW/dblp-2010
    DIMACS10/coAuthorsCiteseer
    VLSI/nv1
    GHS_indef/d_pretok
    VanVelzen/Zd_Jac2
    Andrianov/lp1
    Um/2cubes_sphere
    DIMACS10/mo2010
    Cunningham/qa8fm
    Cunningham/qa8fk
    FEMLAB/ns3Da
    Vavasis/av41092
    GHS_indef/turon_m
    Sandia/ASIC_680ks
    Oberwolfach/gas_sensor
    DIMACS10/ny2010
    VanVelzen/Zd_Jac6
    Simon/venkat25
    Simon/venkat50
    Simon/venkat01
)

include(MatrixDownloadAndConvert.cmake)
