# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Cpu-only custom kernels tie at 1000 GFLOPS. Every tied kernel stays, and the
exact size that rocRoller would launch maps to that kernel."""

from Tensile.LibraryLogic import (
    LogicAnalyzer,
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
