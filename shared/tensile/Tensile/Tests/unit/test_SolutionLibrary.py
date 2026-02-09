################################################################################
#
# Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

import Tensile.SolutionLibrary as SolutionLibrary

from pytest import raises

def test_ArchitectureMap():
    msl = SolutionLibrary.MasterSolutionLibrary(None, None)
    assert msl.ArchitectureIndexMap("gfx90a") == (int("90a", 16) << 18), "Incorrect index value."
    assert msl.ArchitectureIndexMap("gfx900") == (int("900", 16) << 18), "Incorrect index value."
    assert msl.ArchitectureIndexMap("gfx908") == (int("908", 16) << 18), "Incorrect index value."
    assert msl.ArchitectureIndexMap("fallback") == (0), "Incorrect index value."

    # Should throw when computing index for previously computed architecture.
    with raises(RuntimeError, match="ERROR in architecture solution index mapping."):
        msl.ArchitectureIndexMap("gfx90a")

    # Should throw when computing index for non-gfx architecture accept for fallback.
    with raises(RuntimeError, match="ERROR in architecture solution index mapping."):
        msl.ArchitectureIndexMap("test90a")

class MockSolution:
    index = 0
    def __init__(self, name):
        self.name = name

class MockLibrary:
    def remapSolutionIndices(self, indexMap):
        pass

def test_remapSolutionIndexStartingFrom():
    mockSolutions = { 0 : MockSolution("foo"), 1 : MockSolution("bar")}
    SolutionLibrary.MasterSolutionLibrary(mockSolutions, MockLibrary()).remapSolutionIndicesStartingFrom(10)
    assert mockSolutions[0].index == 10, "Zeroith entry should have a remapped index of 10"
    assert mockSolutions[1].index == 11, "First entry should have a remapped index of 11"
