# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""HelperKernelObjectsMemo: keyed on every solution field initHelperKernelObjects reads, and
the deduplicated helper list it yields is the one built without the memo."""

import copy
from collections.abc import Mapping
from pathlib import Path

import pytest

import Tensile.KernelHelperNaming as KHN
from Tensile.Common.Architectures import gfxToIsa
from Tensile.Common.Capabilities import makeIsaInfoMap
from Tensile.Common.DataType import DataType
from Tensile.Common.GlobalParameters import internalParameters
from Tensile.Toolchain.Assembly import makeAssemblyToolchain
from Tensile.Toolchain.Validators import validateToolchain, ToolchainDefaults
import Tensile.LibraryIO as LibraryIO

pytestmark = pytest.mark.unit

# gfx942 HSS solution: UseBias (BiasDataTypeList [S, H]), ActivationType hipblaslt_all,
# GlobalSplitU 1, MultipleBuffer, TileProcessingStrategy None.
_LOGIC = (Path(__file__).parent / "characterization" / "LibraryIO" / "data"
          / "logic_gfx942_HSS_BH.yaml")


@pytest.fixture(scope="module")
def toolchain():
    cxx = validateToolchain("amdclang++")
    isaInfoMap = makeIsaInfoMap([gfxToIsa("gfx942")], cxx)
    bundler = validateToolchain(ToolchainDefaults.OFFLOAD_BUNDLER)
    assembler = makeAssemblyToolchain(cxx, bundler, "default").assembler
    return cxx, isaInfoMap, assembler


@pytest.fixture(scope="module")
def baseSolution(toolchain):
    _, isaInfoMap, assembler = toolchain
    logic = LibraryIO.parseLibraryLogicFile(str(_LOGIC), assembler, False, False, False,
                                            isaInfoMap, False)
    return logic.solutions[0]


def _variant(base, solutionFields, problemTypeFields):
    solution = copy.deepcopy(base)
    for key, value in solutionFields.items():
        solution[key] = value
    for key, value in problemTypeFields.items():
        solution["ProblemType"][key] = value
    return solution


_STREAM_K = {"TileProcessingStrategy": "StreamK", "WorkAssignment": "StaticGrid",
             "_PersistentLoop": True}

# Each takes a different path through the init*KernelObjects functions.
_VARIANTS = {
    "base": ({}, {}),
    "gsu4": ({"GlobalSplitU": 4}, {}),
    "gsuAuto": ({"GlobalSplitU": -1}, {}),
    "streamKAtomic": ({**_STREAM_K, "StreamKAtomic": 1}, {}),
    "streamKNoAtomic": ({**_STREAM_K, "StreamKAtomic": 0}, {}),
    "singleBuffer": ({"GlobalSplitU": 4, "GlobalSplitUAlgorithm": "SingleBuffer",
                      "_GlobalAccumulation": "SingleBuffer"}, {}),
    "mbskAdaptive": ({"GlobalSplitUAlgorithm": "MultipleBufferSingleKernel",
                      "AdaptiveGemmGSUA": 1}, {}),
    "mbsk": ({"GlobalSplitUAlgorithm": "MultipleBufferSingleKernel", "AdaptiveGemmGSUA": 0}, {}),
    "atomicDest": ({"GlobalSplitUAlgorithm": "AtomicDest"}, {}),
    "gradient": ({"GlobalSplitU": 4}, {"Gradient": True}),
    "noBias": ({"GlobalSplitU": 4}, {"UseBias": 0, "BetaOnlyUseBias": False, "BiasDataTypeList": []}),
    "noActivation": ({"ActivationFused": False}, {"ActivationType": "none"}),
    "gateResidual": ({}, {"UseGateResidual": True,
                          "GateResidualDataTypeList": [DataType("s"), DataType("h")]}),
}


@pytest.fixture(scope="module")
def variants(baseSolution):
    return {name: _variant(baseSolution, *fields) for name, fields in _VARIANTS.items()}


class _RecordingSolution(Mapping):
    """Read-only view of a solution that records the keys looked up (including `in` and
    get(), which Mapping implements through __getitem__)."""

    def __init__(self, solution):
        self._solution = solution
        self.read = set()

    def __getitem__(self, key):
        self.read.add(key)
        return self._solution[key]

    def __iter__(self):
        raise AssertionError("iterating the solution reads every field")

    def __len__(self):
        return len(self._solution)


def _helperSignature(objects):
    return [(o.getKernelName(), o.getHeaderFileString(), o.getSourceFileString())
            for o in objects]


def _uniqueHelpers(solutions, helperKernelObjects):
    """The name-based dedupe of BenchmarkProblems.writeBenchmarkFiles."""
    kept, names = [], set()
    for solution in solutions:
        for ko in helperKernelObjects(solution):
            if ko.getKernelName() not in names:
                kept.append(ko)
                names.add(ko.getKernelName())
    return kept


def test_fields_read_are_in_memo_key(toolchain, variants):
    cxx, isaInfoMap, _ = toolchain
    readByAll = set()
    for name, solution in variants.items():
        recording = _RecordingSolution(solution)
        objects = KHN.initHelperKernelObjects(recording, KHN.KernelHelperEnum.All, cxx,
                                              isaInfoMap)
        assert objects or name in ("mbsk", "atomicDest", "noActivation")
        assert recording.read <= set(KHN.HELPER_KERNEL_SOLUTION_FIELDS), name
        readByAll |= recording.read
    # The variants reach every branch that reads a solution field (the legacy StreamK
    # selectors are only read when TileProcessingStrategy is absent).
    assert readByAll == set(KHN.HELPER_KERNEL_SOLUTION_FIELDS) - {"StreamK", "StreamKForceDPOnly"}


def test_memo_keeps_helper_list(toolchain, variants, monkeypatch):
    cxx, isaInfoMap, _ = toolchain
    # Repeats (same object and an equal deep copy) interleaved with distinct inputs.
    v = variants
    solutions = [v["base"], v["gsu4"], copy.deepcopy(v["base"]), v["base"], v["gradient"],
                 v["gsu4"], v["noBias"], v["streamKAtomic"], v["gradient"], v["gateResidual"],
                 v["singleBuffer"], v["mbskAdaptive"], v["noActivation"], v["streamKAtomic"]]
    distinct = 9  # the deep copy of base is equal to base

    expected = _uniqueHelpers(
        solutions,
        lambda s: KHN.initHelperKernelObjects(s, KHN.KernelHelperEnum.All, cxx, isaInfoMap))

    calls = []
    initHelperKernelObjects = KHN.initHelperKernelObjects
    def countingInit(solution, *args):
        calls.append(solution)
        return initHelperKernelObjects(solution, *args)
    monkeypatch.setattr(KHN, "initHelperKernelObjects", countingInit)

    memo = KHN.HelperKernelObjectsMemo(KHN.KernelHelperEnum.All, cxx, isaInfoMap)
    actual = _uniqueHelpers(solutions, memo)

    assert len(calls) == distinct
    assert _helperSignature(actual) == _helperSignature(expected)


def test_memo_key_includes_global_split_u_pgr(toolchain, variants, monkeypatch):
    cxx, isaInfoMap, _ = toolchain
    memo = KHN.HelperKernelObjectsMemo(KHN.KernelHelperEnum.All, cxx, isaInfoMap)
    names = lambda objects: [o.getKernelName() for o in objects]

    first = memo(variants["base"])
    assert memo(variants["base"]) is first
    monkeypatch.setitem(internalParameters, "GlobalSplitUPGR",
                        internalParameters["GlobalSplitUPGR"] * 2)
    second = memo(variants["base"])
    assert second is not first
    assert names(second) == names(KHN.initHelperKernelObjects(
        variants["base"], KHN.KernelHelperEnum.All, cxx, isaInfoMap))
    assert names(second) != names(first)
