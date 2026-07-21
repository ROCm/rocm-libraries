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
    DIMACS10/fe_rotor
    JGD_Relat/relat8
    HVDC/hvdc2
    Li/pli
    Nemeth/nemeth22
    Norris/heart1
    DIMACS10/va2010
    LPnetlib/lp_osa_60
    DIMACS10/nc2010
    DIMACS10/ga2010
    Botonakis/thermomech_dM
    CEMW/vfem
    VLSI/imagesensor
    Boeing/bcsstk35
    VanVelzen/std1_Jac3
    JGD_Homology/n4c6-b10
    DIMACS10/rgg_n_2_17_s0
    Barabasi/NotreDame_actors
    TSOPF/TSOPF_RS_b300_c1
    Oberwolfach/windscreen
)

include(MatrixDownloadAndConvert.cmake)
