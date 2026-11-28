# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Surplus grouped GEMM workgroups end right after the group search.

The grouped grid is sized on the host for the worst case, because the group sizes
live on the device, so workgroups past the last tile of every gemm are common. When
the search finds no gemm for WorkGroup0 the workgroup ends there, instead of loading
the last gemm's arguments and exiting later in remapWgSerial. Kernels whose
workgroups wait on each other keep every workgroup: persistent kernels, cluster
launches, and GSU MultipleBufferSingleKernel or AdaptiveGemmGSUA.

The policy is checked on kernel dictionaries; the emitted code on the kernels of the
common GroupedGemm config (Tests/common/groupedgemm/grouped_gemm.yaml), whose forks
cover both the wave-parallel and the linear search, with and without the exit.
"""

import os
import re
import sys
from types import SimpleNamespace

import pytest

from tensilelite.KernelWriterAssembly import KernelWriterAssembly

pytestmark = pytest.mark.unit

_CODEGEN = os.path.join(os.path.dirname(__file__), "characterization", "_codegen")
if _CODEGEN not in sys.path:
    sys.path.insert(0, _CODEGEN)

_CONFIG = os.path.join(os.path.dirname(os.path.dirname(__file__)), "common", "groupedgemm",
                       "grouped_gemm.yaml")
_ARCH = "gfx942"

_EXIT_AFTER_WAVE_SEARCH = re.compile(r"^(label_GG_WaveSearch_Last\d+):[^\n]*\n\s*(\S+)", re.M)
_EXIT_AFTER_LINEAR_SEARCH = re.compile(
    r"s_cbranch_scc1 (label_GG_HasTile\w*)[^\n]*\n\s*s_endpgm\b[^\n]*\n\s*\1:", re.M)


def _kernel(**overrides):
    kernel = {
        "TileProcessingStrategy": "None",
        "ClusterDim": [1, 1],
        "GlobalSplitU": 1,
        "GlobalSplitUAlgorithm": "MultipleBuffer",
        "AdaptiveGemmGSUA": 0,
    }
    kernel.update(overrides)
    return kernel


@pytest.mark.parametrize("overrides, expected", [
    ({}, True),
    ({"GlobalSplitU": 0, "GlobalSplitUAlgorithm": "MultipleBufferSingleKernel"}, True),
    ({"GlobalSplitU": 2}, True),
    ({"GlobalSplitU": -1}, True),
    ({"GlobalSplitU": 2, "GlobalSplitUAlgorithm": "MultipleBufferSingleKernel"}, False),
    ({"GlobalSplitU": -1, "GlobalSplitUAlgorithm": "MultipleBufferSingleKernel"}, False),
    ({"GlobalSplitU": -1, "AdaptiveGemmGSUA": 1}, False),
    ({"TileProcessingStrategy": "DataParallel"}, False),
    ({"TileProcessingStrategy": "StreamK"}, False),
    ({"ClusterDim": [2, 1]}, False),
    ({"ClusterDim": [1, 4]}, False),
])
def test_surplus_exit_policy(overrides, expected):
    assert KernelWriterAssembly.groupedSurplusEarlyExit(None, _kernel(**overrides)) is expected


def test_surplus_exit_sequence():
    """WorkGroup0 at or past the total tiles ends the workgroup; below, it branches past."""
    stub = SimpleNamespace(labels=SimpleNamespace(getNameInc=lambda name: name + "_7"))
    src = str(KernelWriterAssembly.groupedSurplusExit(stub, 42, "no work"))
    lines = [l.split("//")[0].strip() for l in src.splitlines()]
    lines = [l for l in lines if l and not l.startswith("/*")]
    assert lines == [
        "s_cmp_lt_u32 s[sgprWorkGroup0], s42",
        "s_cbranch_scc1 label_GG_HasTile_7",
        "s_endpgm",
        "label_GG_HasTile_7:",
    ]


def _emit_grouped_kernels(limit=8):
    """[(kernel, source, err)] for the first kernels of the common GroupedGemm config,
    keeping each kernel's parameters (the emitted names are hashes)."""
    import config_harness as ch
    from tensilelite.Common.Types import DebugConfig
    from tensilelite.SolutionStructs.Naming import getKernelFileBase
    from tensilelite.create_library.run import generateKernelObjectsFromSolutions

    assembler, iim = ch._toolchain_for(_ARCH)
    results = []
    with ch._isolated_globals_with_isa(iim):
        sols = ch._solutions_from_config_unguarded(_CONFIG, assembler, iim, limit_solutions=limit,
                                                   problem_index=0, problem_fingerprint=None,
                                                   expected_fork_count=None)
        kernels = sorted(generateKernelObjectsFromSolutions(sols),
                         key=lambda k: getKernelFileBase(False, k))[:limit]
        kwa = KernelWriterAssembly(assembler, DebugConfig())
        for kernel in kernels:
            _base, src, err = ch._emit_one(kwa, kernel, False, True)
            results.append((kernel, src, err))
    return results


def test_wave_search_ends_surplus_workgroups():
    """Every 'past the last tile' branch of the wave-parallel search is an s_endpgm, the
    scalar path these kernels take for a single gemm ends surplus workgroups too, and
    kernels outside the wave search keep the linear search's old path when they must."""
    results = _emit_grouped_kernels()
    assert results and all(err == 0 for _k, _s, err in results)
    wave = 0
    for kernel, src, _err in results:
        exits = _EXIT_AFTER_WAVE_SEARCH.findall(src)
        if exits:
            wave += 1
            assert KernelWriterAssembly.groupedSurplusEarlyExit(None, kernel)
            assert all(first == "s_endpgm" for _label, first in exits), exits
            assert _EXIT_AFTER_LINEAR_SEARCH.search(src)
        elif not KernelWriterAssembly.groupedSurplusEarlyExit(None, kernel):
            assert not _EXIT_AFTER_LINEAR_SEARCH.search(src)
    assert wave, "no kernel of the config uses the wave-parallel search"


def test_linear_search_ends_surplus_workgroups(monkeypatch):
    """With the wave search off, the linear search ends surplus workgroups exactly where
    the policy allows it."""
    monkeypatch.setattr(KernelWriterAssembly, "groupedGemmWaveSearchSupported",
                        lambda self, kernel: False)
    results = _emit_grouped_kernels()
    assert results and all(err == 0 for _k, _s, err in results)
    seen = set()
    for kernel, src, _err in results:
        expected = KernelWriterAssembly.groupedSurplusEarlyExit(None, kernel)
        assert "label_GG_WaveSearch_Loop" not in src
        assert bool(_EXIT_AFTER_LINEAR_SEARCH.search(src)) is expected, kernel["GlobalSplitUAlgorithm"]
        seen.add(expected)
    assert seen == {True, False}, "the config no longer covers kernels with and without the exit"
