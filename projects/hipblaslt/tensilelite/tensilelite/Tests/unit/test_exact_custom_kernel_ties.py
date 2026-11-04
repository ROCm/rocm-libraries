# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Cpu-only custom kernels tie at 1000 GFLOPS. Every tied kernel stays, and the
exact size that rocRoller would launch maps to that kernel."""

import array
from types import SimpleNamespace

import pytest

from tensilelite.LibraryLogic import (
    LogicAnalyzer,
    customKernelTile,
    isCustomKernelSolution,
    libraryDeviceNames,
    preferredExactCustomKernel,
)


class _Kernel:
    """Solution-shaped stub: fields live behind __getitem__, not attributes.

    LibraryLogic merges Tensile Solution objects, which store CustomKernelName
    in state and do not expose it as an attribute.
    """

    def __init__(self, name):
        self._state = {
            "CustomKernelName": name,
            "CustomKernel": {"name": name},
        }

    def __getitem__(self, key):
        return self._state[key]

    def __setitem__(self, key, value):
        self._state[key] = value

    def __eq__(self, other):
        return isinstance(other, _Kernel) and self._state == other._state

    def __hash__(self):
        return hash(self._state["CustomKernelName"])


def _kernel(name):
    return _Kernel(name)


def _names(*tiles):
    prefix = "RR_GEMM_TN_{pair}_SA_BE8M0_32_SB_BE8M0_32_WGT_{tile}"
    out = []
    for pair, tile in tiles:
        suffix = "_WGM_" if "WGM" in tile or tile.endswith("wgm") else ""
        tile = tile.replace("_wgm", "")
        if tile.endswith("WGM"):
            suffix = "_WGM_"
            tile = tile[: -len("WGM")].rstrip("_")
        out.append(_kernel(prefix.format(pair=pair, tile=tile) + suffix))
    return out


def _analyzer(solutions, exactWinners=None, exactScores=None):
    analyzer = LogicAnalyzer.__new__(LogicAnalyzer)
    analyzer.solutions = solutions
    analyzer.exactWinners = dict(exactWinners or {})
    analyzer.exactScores = dict(exactScores or {})
    analyzer.exactProblemSizes = set(analyzer.exactWinners)
    analyzer.mustKeepSolutions = set()
    return analyzer


def test_small_f6_prefers_32x32_and_256_prefers_16x16():
    pair = "FP6_FP6_Float_Float_Float"
    solutions = _names(
        (pair, "16x16x128"),
        (pair, "16x32x128"),
        (pair, "32x16x128"),
        (pair, "32x32x64"),
    )
    assert preferredExactCustomKernel(64, 64, 128, solutions) is solutions[3]
    assert preferredExactCustomKernel(96, 128, 128, solutions) is solutions[3]
    assert preferredExactCustomKernel(256, 256, 256, solutions) is solutions[0]


def test_large_f6_and_f8_tiles():
    f6 = "FP6_FP6_Half_Half_Float"
    f6_solutions = _names(
        (f6, "32x32x64"),
        (f6, "128x128x128_WGM"),
        (f6, "256x256x128_WGM"),
    )
    assert preferredExactCustomKernel(64, 64, 128, f6_solutions) is f6_solutions[0]
    assert preferredExactCustomKernel(3072, 3072, 16384, f6_solutions) is f6_solutions[2]
    assert preferredExactCustomKernel(4096, 4096, 16384, f6_solutions) is f6_solutions[2]

    f8 = "FP8_FP4_Half_Half_Float"
    f8_solutions = _names(
        (f8, "32x32x64"),
        (f8, "128x128x128_WGM"),
        (f8, "256x256x128_WGM"),
    )
    assert preferredExactCustomKernel(64, 64, 128, f8_solutions) is f8_solutions[0]
    assert preferredExactCustomKernel(3072, 3072, 16384, f8_solutions) is f8_solutions[1]
    assert preferredExactCustomKernel(4096, 4096, 16384, f8_solutions) is f8_solutions[2]


def test_tied_custom_kernels_are_kept_and_exact_sizes_remap():
    f6 = "FP6_FP6_Half_Half_Float"
    solutions = _names(
        (f6, "128x128x128_WGM"),
        (f6, "256x256x128_WGM"),
        (f6, "32x32x64"),
    )
    size64 = (64, 64, 1, 128, 64, 64, 128, 128)
    analyzer = LogicAnalyzer.__new__(LogicAnalyzer)
    analyzer.solutions = solutions
    analyzer.exactWinners = {size64: [0, 1000.0]}
    analyzer.exactScores = {size64: {0: 1000.0, 1: 1000.0, 2: 1000.0}}
    analyzer.exactProblemSizes = {size64}
    analyzer.mustKeepSolutions = set()
    analyzer._retainTiedExactCustomKernels()

    assert analyzer.mustKeepSolutions == {0, 1, 2}
    assert analyzer.exactWinners[size64][0] == 2
    large = (3072, 3072, 1, 16384, 3072, 3072, 16384, 16384)
    huge = (4096, 4096, 1, 16384, 4096, 4096, 16384, 16384)
    assert analyzer.exactWinners[large][0] == 1
    assert analyzer.exactWinners[huge][0] == 1
    small = solutions[2]["CustomKernelSizePredicate"]
    big = solutions[1]["CustomKernelSizePredicate"]
    assert {"tag": "SizeLessThan", "index": 0, "value": 129} in small
    assert {"tag": "SizeGreaterThan", "index": 0, "value": 3071} in big


def test_gfx950_without_a_supported_id_uses_mi355_ids():
    assert libraryDeviceNames("gfx950", ["Device 0049", "Device 0050"]) == [
        "Device 75a3",
        "Device 75a2",
    ]
    assert libraryDeviceNames("gfx950", "fallback") == ["Device 75a3", "Device 75a2"]
    assert libraryDeviceNames("gfx950", ["Device 75a3", "Device 75a2"]) == [
        "Device 75a3",
        "Device 75a2",
    ]
    assert libraryDeviceNames("gfx942", ["Device 0049", "Device 0050"]) == [
        "Device 0049",
        "Device 0050",
    ]
    assert libraryDeviceNames("gfx950", "Device 75a3") == "Device 75a3"


def test_custom_kernel_tile_sources():
    wgm = "RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM_"
    assert customKernelTile({"CustomKernel": {"name": wgm}}) == (128, 128, 128, True)
    assert customKernelTile({"CustomKernel": {"macrotile": [64, 32, 128, 1]}}) == (
        64,
        32,
        128,
        False,
    )
    assert customKernelTile({"CustomKernel": {}}) is None
    assert customKernelTile({}) is None

    # Not subscriptable: fields fall back to attributes.
    plain = SimpleNamespace(
        CustomKernelName="RR_GEMM_TN_FP6_FP6_Float_Float_Float_WGT_32x32x64"
    )
    assert customKernelTile(plain) == (32, 32, 64, False)
    assert isCustomKernelSolution(plain)
    assert isCustomKernelSolution({"CustomKernel": {"macrotile": [16, 16, 128]}})
    assert not isCustomKernelSolution({"CustomKernel": {}})
    assert not isCustomKernelSolution({})


def test_preferred_kernel_without_a_parsable_tile():
    untiled = {"CustomKernel": {}}
    tiled = _names(("FP6_FP6_Float_Float_Float", "32x32x64"))[0]
    assert preferredExactCustomKernel(64, 64, 128, []) is None
    assert preferredExactCustomKernel(64, 64, 128, [untiled]) is untiled
    assert preferredExactCustomKernel(64, 64, 128, [untiled, tiled]) is tiled


def test_wgm_fallbacks_when_the_preferred_tile_is_missing():
    f8 = "FP8_FP4_Half_Half_Float"
    only128 = _names((f8, "128x128x128_WGM"))
    only256 = _names((f8, "256x256x128_WGM"))
    # 4096 wants 256x256 and does not launch 128x128, so the largest tile left wins.
    assert preferredExactCustomKernel(4096, 4096, 16384, only128) is only128[0]
    # 1024 wants 128x128; 256x256 is next.
    assert preferredExactCustomKernel(1024, 1024, 16384, only256) is only256[0]


def test_non_wgm_fallbacks():
    f6 = "FP6_FP6_Float_Float_Float"
    rect = _names((f6, "16x32x128"), (f6, "32x16x128"))
    # Neither 32x32x64 nor 16x16x128: largest tile, then larger M.
    assert preferredExactCustomKernel(64, 64, 128, rect) is rect[1]
    mixed = _names((f6, "16x16x128"), (f6, "32x32x64"))
    # Small M with a large N is neither the small case nor the 256 case.
    assert preferredExactCustomKernel(64, 512, 128, mixed) is mixed[1]


def test_untied_exact_rows_are_left_alone():
    f6 = "FP6_FP6_Half_Half_Float"
    solutions = _names((f6, "32x32x64"), (f6, "128x128x128_WGM"))
    single = (64, 64, 1, 128, 64, 64, 128, 128)
    zero = (96, 96, 1, 128, 96, 96, 128, 128)
    clear = (128, 128, 1, 128, 128, 128, 128, 128)
    winners = {single: [0, 1000.0], zero: [0, 0.0], clear: [1, 1000.0]}
    analyzer = _analyzer(
        solutions,
        exactWinners=winners,
        exactScores={
            single: {0: 1000.0},
            zero: {0: 0.0, 1: 0.0},
            clear: {0: 900.0, 1: 1000.0},
        },
    )
    analyzer._retainTiedExactCustomKernels()

    assert analyzer.mustKeepSolutions == set()
    assert analyzer.exactWinners == winners
    assert all("CustomKernelSizePredicate" not in s._state for s in solutions)


def test_ties_without_wgm_kernels_add_no_large_sizes():
    f6 = "FP6_FP6_Float_Float_Float"
    solutions = _names((f6, "16x16x128"), (f6, "32x32x64"))
    size64 = (64, 64, 1, 128, 64, 64, 128, 128)
    size96 = (96, 96, 1, 128, 96, 96, 128, 128)
    short = (256, 256, 256)
    analyzer = _analyzer(
        solutions,
        exactWinners={size64: [0, 1000.0], size96: [0, 1000.0], short: [0, 1000.0]},
        exactScores={size64: {0: 1000.0, 1: 1000.0}, size96: {0: 1000.0}},
    )
    analyzer._retainTiedExactCustomKernels()

    assert analyzer.mustKeepSolutions == {0, 1}
    assert analyzer.exactWinners[size64][0] == 1
    # Only one kernel measured this row, so its measured winner stays.
    assert analyzer.exactWinners[size96][0] == 0
    assert analyzer.exactWinners[(64, 96, 1, 128, 64, 96, 128, 128)][0] == 1
    assert (3072, 3072, 1, 16384, 3072, 3072, 16384, 16384) not in analyzer.exactWinners
    assert (4096, 4096, 1, 16384, 4096, 4096, 16384, 16384) not in analyzer.exactWinners
    assert {"tag": "SizeLessThan", "index": 0, "value": 97} in solutions[0][
        "CustomKernelSizePredicate"
    ]


@pytest.mark.parametrize("problemType, kIndex", [({"NumIndicesC": 4}, 4), ({}, 3)])
def test_size_predicates_follow_the_k_index(problemType, kIndex):
    solutions = _names(("FP6_FP6_Float_Float_Float", "32x32x64"))
    zeroM = (0, 64, 1, 128, 0, 64, 128, 128)
    wideM = (128, 64, 1, 256, 128, 64, 256, 256)
    analyzer = _analyzer(
        solutions, exactWinners={zeroM: [0, 1000.0], wideM: [0, 1000.0]}
    )
    analyzer.problemType = problemType
    analyzer._assignCustomKernelSizePredicates()

    # M starts at 0, so it has no lower bound.
    assert solutions[0]["CustomKernelSizePredicate"] == [
        {"tag": "SizeLessThan", "index": 0, "value": 129},
        {"tag": "SizeGreaterThan", "index": 1, "value": 63},
        {"tag": "SizeLessThan", "index": 1, "value": 65},
        {"tag": "SizeGreaterThan", "index": kIndex, "value": 127},
        {"tag": "SizeLessThan", "index": kIndex, "value": 257},
    ]


def test_least_important_solution_skips_must_keep_kernels():
    f6 = "FP6_FP6_Float_Float_Float"
    analyzer = _analyzer(_names((f6, "16x16x128"), (f6, "32x32x64")))
    analyzer.numSolutions = 2
    analyzer.numIndices = 4
    analyzer.problemIndicesForGlobalRange = []
    analyzer.solutionNames = ["16x16", "32x32"]

    assert analyzer.leastImportantSolution()[0] == 0
    analyzer.mustKeepSolutions = {0}
    assert analyzer.leastImportantSolution()[0] == 1


def test_must_keep_indices_follow_removed_solutions(monkeypatch):
    monkeypatch.setattr(
        "Tensile.LibraryLogic.getSolutionNameMin",
        lambda solution, splitGSU: solution["CustomKernelName"],
    )
    f6 = "FP6_FP6_Half_Half_Float"
    solutions = _names((f6, "16x16x128"), (f6, "32x32x64"), (f6, "128x128x128_WGM"))
    for solution in solutions:
        solution["MacroTile0"] = 32
        solution["MacroTile1"] = 32
    size = (64, 64, 1, 128, 64, 64, 128, 128)
    analyzer = _analyzer(solutions, exactWinners={size: [2, 1000.0]})
    analyzer.splitGSU = False
    analyzer.numSolutions = 3
    analyzer.totalProblems = 1
    analyzer.data = array.array("f", [1.0, 2.0, 3.0])
    analyzer.mustKeepSolutions = {0, 2}

    analyzer.removeSolution(1)
    assert analyzer.mustKeepSolutions == {0, 1}
    assert analyzer.exactWinners[size][0] == 1
    assert list(analyzer.data) == [1.0, 3.0]

    analyzer.pruneSolutions({1})
    assert analyzer.mustKeepSolutions == {0}
    assert analyzer.exactWinners[size][0] == 0
    assert analyzer.solutions == [solutions[2]]
    assert list(analyzer.data) == [3.0]
