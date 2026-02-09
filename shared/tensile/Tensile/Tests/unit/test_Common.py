################################################################################
#
# Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

from __future__ import print_function

import Tensile.Common as Common

import os

def test_gfxArch():
    assert Common.gfxArch('gfx9') is None

    assert Common.gfxArch('gfx803') == (8,0,3)
    assert Common.gfxArch('gfx900') == (9,0,0)
    assert Common.gfxArch('gfx906') == (9,0,6)

    assert Common.gfxArch('gfx1010') == (10,1,0)

    assert Common.gfxArch('gfx90015') == (900,1,5)

    assert Common.gfxArch('blah gfx900 stuff') == (9,0,0)

def test_paths():
    workingPathName = os.path.join("working", "path")
    Common.globalParameters["WorkingPath"] = workingPathName
    expectedWorkingPath = os.path.join("working", "path")
    assert Common.globalParameters["WorkingPath"] == expectedWorkingPath

    recursiveWorkingPath = "next1"
    expectedRecurrsiveWorkingPath = os.path.join("working", "path", "next1")
    Common.pushWorkingPath (recursiveWorkingPath)
    assert Common.globalParameters["WorkingPath"] == expectedRecurrsiveWorkingPath
    Common.popWorkingPath()
    assert Common.globalParameters["WorkingPath"] == expectedWorkingPath

    set1WorkingPath = os.path.join("working", "path", "set1")
    expectedSet1WorkingPath = os.path.join("working", "path", "set1")
    Common.setWorkingPath (set1WorkingPath)
    assert Common.globalParameters["WorkingPath"] == expectedSet1WorkingPath
    Common.popWorkingPath()
    assert Common.globalParameters["WorkingPath"] == expectedWorkingPath
