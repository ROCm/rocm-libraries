################################################################################
#
# Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
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

from Tensile.Utilities.merge import cmpHelper, fixSizeInconsistencies, removeUnusedKernels, mergeLogic
from Tensile.Utilities.ConditionalImports import yamlLoader

import yaml, pytest

# the merge scripts does not differentiate solutions based on index or name
# so "DummyParam" is used to mark if solutions should be equal or not

logicPrefix = r"""
- DummyVersionRequirement
- DummyLanguage
- DummyScheduleName
- DummyDevice
- DummyProblemType
"""

baseLogic=logicPrefix + r"""
-
  - SolutionIndex: 2
    SolutionNameMin: InUseForSize256
    DummyParam: InUseForSize256
  - SolutionIndex: 1
    SolutionNameMin: UnusedSolution
    DummyParam: UnusedSolution
  - SolutionIndex: 0
    SolutionNameMin: InUseForSize128or64
    DummyParam: InUseForSize128or64
- DummyIndexAssignment
-
  - - [256, 256, 1, 256]
    - [2, 111.1]
  - - [128, 128, 1, 128]
    - [0, 99.9]
  - - [64, 64, 1, 64]
    - [0, 88.8]
"""

incLogic=logicPrefix + r"""
-
  - SolutionIndex: 39
    SolutionNameMin: InUseForSize256or1024xxx
    DummyParam: InUseForSize256or1024xxx
  - SolutionIndex: 1
    SolutionNameMin: InUseForSize128xxx
    DummyParam: InUseForSize128xxx
- DummyIndexAssignment
-
  - - [128, 128, 1, 128]
    - [1, 999.9]
  - - [256, 256, 1, 256]
    - [39, 999.9]
  - - [1024, 1024, 1, 1024]
    - [39, 999.9]
"""

notUniqueSolution=logicPrefix+r"""
-
  - SolutionIndex: 0
    SolutionNameMin: Kernel0
    DummyParam: Kernel0
  - SolutionIndex: 1
    SolutionNameMin: Kernel0
    DummyParam: Kernel0
"""

uniqueSolution=logicPrefix+r"""
-
  - SolutionIndex: 0
    SolutionNameMin: Kernel0
    DummyParam: Kernel0
  - SolutionIndex: 1
    SolutionNameMin: Kernel1
    DummyParam: Kernel1
"""

notTrimmedSize=r"""
-
  - - [128, 128, 1, 128, 128, 128, 128, 128]
    - [1, 999.9]
  - - [1024, 1024, 1, 1024]
    - [42, 0.0]
  - - [128, 128, 1, 128]
    - [39, 0.0]
  - - [1024, 1024, 1, 1024, 1024, 1024, 1024, 1024]
    - [39, 999.9]
  - - [1024, 1024, 1, 1024, 1044, 1044, 1044, 1044]
    - [39, 999.9]
"""

trimmedSize=r"""
-
  - - [128, 128, 1, 128]
    - [1, 999.9]
  - - [512, 512, 1, 512]
    - [39, 999.9]
  - - [1024, 1024, 1, 1024]
    - [42, 999.9]
"""

mfmaMergeBaseLogic=logicPrefix+r"""
-
  - SolutionIndex: 0
    SolutionNameMin: MFMA_base
    DummyParam: MFMA_base
    EnableMatrixInstruction: True
    MatrixInstruction: [16, 16, 4, 1]
  - SolutionIndex: 1
    SolutionNameMin: VALU_base
    DummyParam: VALU_base
    EnableMatrixInstruction: False
    MatrixInstruction: []
- DummyIndexAssignment
"""

mfmaMergeIncLogic=logicPrefix+r"""
-
  - SolutionIndex: 0
    SolutionNameMin: MFMA_inc
    DummyParam: MFMA_inc
    EnableMatrixInstruction: True
    MatrixInstruction: [16, 16, 4, 1]
  - SolutionIndex: 1
    SolutionNameMin: VALU_inc
    DummyParam: VALU_inc
    EnableMatrixInstruction: False
    MatrixInstruction: []
- DummyIndexAssignment
"""

mfmaMergeBaseSizes=r"""
-
  - - [128, 128, 1, 128]
    - [0, 3.0]
  - - [128, 128, 1, 128]
    - [1, 6.0]
  - - [130, 128, 1, 128]
    - [1, 9.0]
  - - [131, 128, 1, 128]
    - [0, 12.0]
"""

mfmaMergeIncFasterSizes=r"""
-
  - - [128, 128, 1, 128]
    - [0, 4.0]
  - - [128, 128, 1, 128]
    - [1, 7.0]
  - - [131, 128, 1, 128]
    - [0, 13.0]
  - - [130, 128, 1, 128]
    - [1, 10.0]
"""

mfmaMergeIncSlowerSizes=r"""
-
  - - [128, 128, 1, 128]
    - [0, 2.0]
  - - [128, 128, 1, 128]
    - [1, 5.0]
  - - [131, 128, 1, 128]
    - [0, 11.0]
  - - [130, 128, 1, 128]
    - [1, 8.0]
"""

mfmaMergeIncNotMatchingMFMA=r"""
-
  - - [130, 128, 1, 128]
    - [0, 7.0]
  - - [131, 128, 1, 128]
    - [1, 12.0]
"""

mfmaMergeResNotMatchingMFMA=r"""
-
  - - [128, 128, 1, 128]
    - [0, 3.0]
  - - [128, 128, 1, 128]
    - [1, 6.0]
  - - [130, 128, 1, 128]
    - [1, 9.0]
  - - [131, 128, 1, 128]
    - [0, 12.0]
  - - [130, 128, 1, 128]
    - [2, 7.0]
  - - [131, 128, 1, 128]
    - [3, 12.0]
"""

def checkUniqueSolution(solutionPool):
  uniq = set()
  # note: any([False or None, True or None]) -> True
  return not any(frozenset(cmpHelper(i).items()) in uniq or uniq.add(frozenset(cmpHelper(i).items()))
                 for i in solutionPool)

@pytest.mark.parametrize("sizes, expected", [
  (   trimmedSize, [[128,128,1,128],[512,512,1,512],[1024,1024,1,1024]]),
  (notTrimmedSize, [[1024,1024,1,1024],[128,128,1,128]])
  ])
def test_fixSizeInconsistencies(sizes, expected):
  data = yaml.load(sizes, yamlLoader)
  data_ = fixSizeInconsistencies(data[0], "dummy")

  for [size, [_,_]], expected_ in zip(data_[0], expected):
    assert size == expected_

@pytest.mark.parametrize("input,expectedNumKernelRemoved", [(baseLogic, 1),
                                                            (incLogic, 0)])
def test_removeUnusedKernels(input, expectedNumKernelRemoved):
  data = yaml.load(input, yamlLoader)
  dataFiltered, numKernelRemoved = removeUnusedKernels(data)

  # test if number of solution removed is correct
  assert numKernelRemoved == expectedNumKernelRemoved

  # test if solution is re-indexed
  for index, s in enumerate(dataFiltered[5]):
    assert index == s["SolutionIndex"]

  # check if size-kernel mapping is not compromised
  solutionMapBefore = {}
  solutionMapAfter = {}
  for [size, [index, _]] in data[7]:
    solutionMapBefore[tuple(size)] = [s["SolutionNameMin"] for s in data[5] if s["SolutionIndex"]==index][0]
  for [size, [index, _]] in dataFiltered[7]:
    solutionMapAfter[tuple(size)] = [s["SolutionNameMin"] for s in dataFiltered[5] if s["SolutionIndex"]==index][0]

  assert solutionMapBefore == solutionMapAfter

  # print final yaml for visual inspection
  # stream = io.StringIO("")
  # yaml.safe_dump(dataFiltered, stream, default_flow_style=None)
  # print(stream.getvalue())

@pytest.mark.parametrize("baseLogic, incLogic, expectedStats, expectedSizes, expectedSolutions", [
# test case #1: merge incLogic into baseLogic
  (baseLogic, incLogic, [1,2,2], # 1 sizes added, 2 kernels added, 2 kernels removed/replaced
  [(1024,1024,1,1024), (256,256,1,256), (128,128,1,128), (64,64,1,64)],
  ["InUseForSize256or1024xxx", "InUseForSize256or1024xxx", "InUseForSize128xxx", "InUseForSize128or64"]),
# test case #2: merge baseLogic into itself
  (baseLogic, baseLogic, [0,0,1], # 0 sizes added, 0 kernels added, 1 kernel removed because it's unused
  [(256,256,1,256), (128,128,1,128), (64,64,1,64)],
  ["InUseForSize256", "InUseForSize128or64", "InUseForSize128or64"]),
])
def test_mergeLogic(baseLogic, incLogic, expectedStats, expectedSizes, expectedSolutions):
  baseData = yaml.load(baseLogic, yamlLoader)
  incData = yaml.load(incLogic, yamlLoader)

  mergedData, *stats = mergeLogic(baseData, incData, False)

  # check if stats are as expected
  assert stats == expectedStats

  # check if solution matches expected. assumes SolutionNameMin is uniqueSolution
  # (which is satisfied by the test data given here)
  solutionMap = {} # size -> solutionName
  for size, [index, _] in mergedData[7]:
    solutionMap[tuple(size)] = [s["SolutionNameMin"] for s in mergedData[5] if s["SolutionIndex"]==index][0]

  for size, expected in zip(expectedSizes, expectedSolutions):
    assert solutionMap[size] == expected

  # check if each solution in merged data is uniqueSolution
  assert checkUniqueSolution(mergedData[5])

@pytest.mark.parametrize("input,expected", [(uniqueSolution, True), (notUniqueSolution, False)])
def test_checkUniqueSolution(input, expected):
  data = yaml.load(input, yamlLoader)
  assert checkUniqueSolution(data[5]) == expected

@pytest.mark.parametrize("baseLogic, incLogic, expectedSizesYaml, expectedSolutions", [
# test case #1: Slower sizes in incremental logic file
  (mfmaMergeBaseLogic+mfmaMergeBaseSizes, mfmaMergeIncLogic+mfmaMergeIncSlowerSizes,
   mfmaMergeBaseSizes, ["MFMA_base", "VALU_base"]),
# test case #2: Faster sizes in incremental logic file
  (mfmaMergeBaseLogic+mfmaMergeBaseSizes, mfmaMergeIncLogic+mfmaMergeIncFasterSizes,
   mfmaMergeIncFasterSizes, ["MFMA_inc", "VALU_inc"]),
# test case #3: Test that VALU size is included alongside MFMA size, and vice versa (regardless of efficiency)
  (mfmaMergeBaseLogic+mfmaMergeBaseSizes, mfmaMergeIncLogic+mfmaMergeIncNotMatchingMFMA,
   mfmaMergeResNotMatchingMFMA, ["MFMA_base", "VALU_base", "MFMA_inc", "VALU_inc"])
])
def test_mfmaMergeLogic(baseLogic, incLogic, expectedSizesYaml, expectedSolutions):
  baseData      = yaml.load(baseLogic, yamlLoader)
  incData       = yaml.load(incLogic,  yamlLoader)
  expectedSizes = yaml.load(expectedSizesYaml, yamlLoader)[0]

  mergedData, _, _, _ = mergeLogic(baseData, incData, False, True, True)

  solutionIndices = {s['SolutionNameMin']: s['SolutionIndex'] for s in mergedData[5]} # size -> solutionName

  #Ensure all correct solutions are present in merged data
  for solution in expectedSolutions:
    assert solution in solutionIndices.keys()

  assert len(expectedSolutions) == len(mergedData[5])

  #Convert expected sizes to use mergedData's solution indices
  expectedSizes = [ [size, [solutionIndices[expectedSolutions[solIndex]], eff]] for size, [solIndex, eff] in expectedSizes ]

  #Ensure all expected sizes are present in merged data
  for item in expectedSizes:
    assert item in mergedData[7]

  assert len(expectedSizes) == len(mergedData[7])

if __name__ == "__main__":
    # test_mergeLogic(baseLogic, incLogic, [1,2,2], [(1024, 1024, 1, 1024), (256, 256, 1, 256), (128, 128, 1, 128), (64, 64, 1, 64)], [ "InUseForSize256or1024xxx", "InUseForSize256or1024xxx", "InUseForSize128xxx", "InUseForSize128or64"])
    # test_checkUniqueSolution(uniqueSolution, True)
    # test_fixSizeInconsistencies(trimmedSize, [[128,128,1,128],[512,512,1,512],[1024,1024,1,1024]])
    test_fixSizeInconsistencies(notTrimmedSize, [[1024,1024,1,1024],[128,128,1,128]])
