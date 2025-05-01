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

import os
from copy import deepcopy

import yaml

from Tensile.__init__ import __version__
import Tensile.LibraryIO as LibraryIO
from Tensile.Utilities.ConditionalImports import yamlLoader


# string literals for testing
# list format
version_l = "- {MinimumRequiredVersion: " + __version__ + "}\n"

vega20_l = r"""
- vega20
- gfx906
- [Device 66a0, Device 66a1, Device 66a7, Device 66af, Vega 20]
"""

aldebaran_l = r"""
- aldebaran
- {Architecture: gfx90a, CUCount: 104}
- [Device 0050, Device 0051, Device 0052, Device 0054, Device 0062, Device 7400, Device 740c]
"""

sizes = r"""
- DummyIndexAssignment
- - - [128, 128, 1, 128]
    - [0, 80.0]
  - - [512, 512, 1, 512]
    - [1, 85.0]
"""

dvEff_l = r"""
- null
- null
- DeviceEfficiency
"""

cuEff_l = r"""
- null
- null
- CUEfficiency
"""

fp16AltImpl_l = "- Fp16AltImpl\n"
fp16AltImplRound_l = "- Fp16AltImplRound\n"
legacySuffix_l = "- null\n"

# dict format general
version_d = "MinimumRequiredVersion: " + __version__ + "\n"
dvEff_d = "PerfMetric: DeviceEfficiency\n"
cuEff_d = "PerfMetric: CUEfficiency\n"
fp16AltImpl_d = "Fp16AltImpl: true\n"
fp16AltImplFalse_d = "Fp16AltImpl: false\n"
fp16AltImplRound_d = "Fp16AltImplRound: true\n"
fp16AltImplRoundFalse_d = "Fp16AltImplRound: false\n"

vega20_d = r"""
ScheduleName: vega20
ArchitectureName: gfx906
DeviceNames: [Device 66a0, Device 66a1, Device 66a7, Device 66af, Vega 20]
"""

aldebaran_d = r"""
ScheduleName: aldebaran
ArchitectureName: gfx90a
CUCount: 104
DeviceNames: [Device 0050, Device 0051, Device 0052, Device 0054, Device 0062]
"""

# dict format matching
matchingLibrary = r"""
LibraryType: Matching
Library:
  indexOrder: DummyIndexAssignment
  table:
  - - [128, 128, 1, 128]
    - [0, 80.0]
  - - [512, 512, 1, 512]
    - [1, 85.0]
  distance: Euclidean
"""

# dict format decision tree
treeLibrary = r"""
LibraryType: DecisionTree
Library:
- features:
  - {index: 0, type: FreeSizeA}
  - {index: 0, type: BoundSize}
  trees:
  - tree:
    - {featureIdx: 0, threshold: 7000, nextIdxLTE: -2, nextIdxGT: -1}
    solution: 0
  - tree:
    - {featureIdx: 1, threshold: 7000, nextIdxLTE: -1, nextIdxGT: -2}
    solution: 1
  region:
  - {type: SizeInRange, index: 0, value: {min: 6000, max: 8000}}
  - {type: SizeInRange, index: 1, value: {min: 6000, max: 7000}}

- features:
  - {index: 0, type: FreeSizeB}
  trees:
  - tree:
    - {featureIdx: -1, threshold: 1, nextIdxLTE: 0, nextIdxGT: 0}
    solution: 0
  region:
  - {type: SizeInRange, index: 0, value: {max: 128}}
  - {type: SizeInRange, index: 1, value: {min: 1, max: 256}}
  - {type: SizeInRange, index: 3, value: {min: 1}}
"""


def createLibraryLogicList(arch_str, suffix_str, fp16AltImpl, fp16AltImplRound):
    # paths to test data
    scriptDir = os.path.dirname(os.path.realpath(__file__))
    dataDir = os.path.realpath(os.path.join(scriptDir, "..", "test_data", "unit"))
    problemTypePath = os.path.realpath( \
            os.path.join(dataDir, "library_data", "problemType.yaml"))
    solutionParametersPath = os.path.realpath( \
            os.path.join(dataDir, "library_data", "initialSolutionParameters.yaml"))

    # read test data
    problemType = LibraryIO.readYAML(problemTypePath)["ProblemType"]
    solutionParameters = LibraryIO.readYAML(solutionParametersPath)

    # solutions
    sol0 = deepcopy(solutionParameters)
    sol0["SolutionIndex"] = 0
    sol0["SolutionNameMin"] = "foo"
    sol0["ProblemType"] = problemType

    sol1 = deepcopy(solutionParameters)
    sol1["SolutionIndex"] = 1
    sol1["SolutionNameMin"] = "bar"
    sol0["ProblemType"] = problemType

    # other components
    prefixData = yaml.load(version_l + arch_str, yamlLoader)
    sizeData = yaml.load(sizes, yamlLoader)
    suffixData = yaml.load(suffix_str, yamlLoader)

    # handle fp16AltImpl and combine
    rv = prefixData + [problemType] + [[sol0, sol1]] + sizeData + suffixData
    if fp16AltImpl:
        fp16AltData = yaml.load(fp16AltImpl_l, yamlLoader)
        rv += fp16AltData

    if fp16AltImplRound:
        fp16AltRoundData = yaml.load(fp16AltImplRound_l, yamlLoader)
        rv += fp16AltRoundData

    return rv


def createLibraryLogicDict(arch_str, suffix_str, lib_str, fp16AltImpl_str, fp16AltImplRound_str):
    # paths to test data
    scriptDir = os.path.dirname(os.path.realpath(__file__))
    dataDir = os.path.realpath(os.path.join(scriptDir, "..", "test_data", "unit"))
    problemTypePath = os.path.realpath( \
            os.path.join(dataDir, "library_data", "problemType.yaml"))
    solutionParametersPath = os.path.realpath( \
            os.path.join(dataDir, "library_data", "initialSolutionParameters.yaml"))

    # read test data
    problemType = LibraryIO.readYAML(problemTypePath)["ProblemType"]
    solutionParameters = LibraryIO.readYAML(solutionParametersPath)

    # solutions
    sol0 = deepcopy(solutionParameters)
    sol0["SolutionIndex"] = 0
    sol0["SolutionNameMin"] = "foo"
    sol0["ProblemType"] = problemType

    sol1 = deepcopy(solutionParameters)
    sol1["SolutionIndex"] = 1
    sol1["SolutionNameMin"] = "bar"
    sol0["ProblemType"] = problemType

    # other components
    prefixData = yaml.load(version_d + arch_str, yamlLoader)
    libData = yaml.load(lib_str, yamlLoader)
    suffixData = yaml.load(suffix_str, yamlLoader)

    # handle fp16AltImpl and combine
    fp16Data = {}
    if fp16AltImpl_str is not None:
        fp16Data = yaml.load(fp16AltImpl_str, yamlLoader)

    fp16RoundData = {}
    if fp16AltImplRound_str is not None:
        fp16RoundData = yaml.load(fp16AltImplRound_str, yamlLoader)

    data = {**prefixData, **libData, **suffixData, **fp16Data, **fp16RoundData}
    data["ProblemType"] = problemType
    data["Solutions"] = [sol0, sol1]
    return data


def test_parseSolutionsData(useGlobalParameters):
    with useGlobalParameters():
        # paths to test data
        scriptDir = os.path.dirname(os.path.realpath(__file__))
        dataDir = os.path.realpath(os.path.join(scriptDir, "..", "test_data", "unit"))
        solutionsPath = os.path.realpath(os.path.join( \
                dataDir, "solutions", "solutions_nn_3.yaml"))

        solutions = LibraryIO.readYAML(solutionsPath)

        LibraryIO.parseSolutionsData(solutions, "test_parseSolutionsData")
        assert True


def test_parseLibraryLogicList(useGlobalParameters):
    with useGlobalParameters():
        LibraryIO.parseLibraryLogicData(createLibraryLogicList(vega20_l, dvEff_l, False, False),
                                        "test_parseLibraryLogicList")

        LibraryIO.parseLibraryLogicData(createLibraryLogicList(aldebaran_l, cuEff_l, False, False),
                                        "test_parseLibraryLogicList")

        LibraryIO.parseLibraryLogicData(createLibraryLogicList(vega20_l, legacySuffix_l, False, False),
                                        "test_parseLibraryLogicList")

        LibraryIO.parseLibraryLogicData(createLibraryLogicList(aldebaran_l, dvEff_l, True, False),
                                        "test_parseLibraryLogicList")

        LibraryIO.parseLibraryLogicData(createLibraryLogicList(aldebaran_l, dvEff_l, True, True),
                                        "test_parseLibraryLogicList")
        assert True


def test_parseLibraryLogicMatching(useGlobalParameters):
    with useGlobalParameters():
        LibraryIO.parseLibraryLogicData(
            createLibraryLogicDict(vega20_d, matchingLibrary, dvEff_d, fp16AltImplFalse_d, None),
            "test_parseLibraryLogicMatching")

        LibraryIO.parseLibraryLogicData(
            createLibraryLogicDict(aldebaran_d, matchingLibrary, cuEff_d, None, None),
            "test_parseLibraryLogicMatching")

        LibraryIO.parseLibraryLogicData(
            createLibraryLogicDict(aldebaran_d, matchingLibrary, dvEff_d, fp16AltImpl_d, None),
            "test_parseLibraryLogicMatching")

        LibraryIO.parseLibraryLogicData(
            createLibraryLogicDict(aldebaran_d, matchingLibrary, dvEff_d, fp16AltImpl_d, fp16AltImplRound_d),
            "test_parseLibraryLogicMatching")
        assert True


def test_parseLibraryLogicDecisionTree(useGlobalParameters):
    with useGlobalParameters():
        LibraryIO.parseLibraryLogicData(
            createLibraryLogicDict(vega20_d, treeLibrary, dvEff_d, fp16AltImplFalse_d, None),
            "test_parseLibraryLogicDecisionTree")

        LibraryIO.parseLibraryLogicData(
            createLibraryLogicDict(aldebaran_d, treeLibrary, cuEff_d, None, None),
            "test_parseLibraryLogicDecisionTree")

        LibraryIO.parseLibraryLogicData(
            createLibraryLogicDict(aldebaran_d, treeLibrary, dvEff_d, fp16AltImpl_d, None),
            "test_parseLibraryLogicDecisionTree")

        LibraryIO.parseLibraryLogicData(
            createLibraryLogicDict(aldebaran_d, treeLibrary, dvEff_d, fp16AltImpl_d, fp16AltImplRound_d),
            "test_parseLibraryLogicDecisionTree")
        assert True
