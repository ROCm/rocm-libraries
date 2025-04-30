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
    Norris/torso3
    DNVS/thread
    DIMACS10/tx2010
    CEMW/tmt_unsym
    CEMW/t2em
    DNVS/shipsec5
    TKK/engine
    GHS_psdef/apache2
    DIMACS10/netherlands_osm
    DIMACS10/kron_g500-logn16
    Stevenson/LargeRegFile
    McRae/ecology2
    McRae/ecology1
    SNAP/wiki-Talk
    FreeFieldTechnologies/mono_500Hz
    CEMW/tmt_sym
    SNAP/web-Google
    PARSEC/Si34H36
    LAW/amazon-2008
    Chen/pkustk11
)

include(MatrixDownloadAndConvert.cmake)
