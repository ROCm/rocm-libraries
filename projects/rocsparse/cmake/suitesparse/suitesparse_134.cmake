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
    PARSEC/Ge87H76
    Mittelmann/rail2586
    Meszaros/degme
    Fluorem/PR02R
    JGD_GL7d/GL7d22
    JGD_BIBD/bibd_20_10
    PARSEC/Ge99H100
    Norris/torso1
    Schmid/thermal2
    DNVS/x104
    TSOPF/TSOPF_FS_b300_c2
    TSOPF/TSOPF_RS_b678_c2
    Bourchtein/atmosmodd
    Bourchtein/atmosmodj
    PARSEC/Ga19As19H42
    Bodendiek/CurlCurl_2
    JGD_BIBD/bibd_22_8
    Rothberg/gearbox
    Gupta/gupta3
    Andrianov/pattern1
)

include(MatrixDownloadAndConvert.cmake)
