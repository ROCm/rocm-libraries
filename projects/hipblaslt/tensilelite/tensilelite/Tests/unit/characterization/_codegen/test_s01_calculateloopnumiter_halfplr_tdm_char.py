################################################################################
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
"""Characterize calculateLoopNumIter's gfx1250 HalfPLR TDM paths.

The designed config requests the single-wave HalfPLR and TDMSplit combination.
``Solution.assignDerivedParameters`` rejects every solution because TDMSplit is
currently disabled, so the config cannot reach the corresponding
``KernelWriterAssembly.calculateLoopNumIter`` code. These tests record that
boundary without claiming coverage of unreachable emitter lines.

The emitted cells cover HalfPLR's K tail with MX scales on one tensor only, on
TDMFuse 0-3 at two and four waves and TDMFuse 0 on one wave, and without MX
scales on one wave. After the main loop the HalfPLR adjust sets every group0
enable word the last iterations zeroed back to 1, and removeStagger rebases each
tensor whose own GlobalReadIncs the mask zeroes from an intact copy primed
before the loop. CPU-only: Python and rocisa codegen, nothing is assembled.
"""

import os
import re

import pytest
import yaml

from config_harness import derive_states, emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx1250"

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx1250",
    "s01_calculateloopnumiter_halfplr_tdm.yaml",
)


def test_s01_calculateloopnumiter_halfplr_tdm_is_rejected():
    """The disabled combination produces no kernel."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert results == []


def test_s01_calculateloopnumiter_halfplr_tdm_reports_reason(capsys, monkeypatch):
    """The rejection names TDMSplit instead of silently dropping the config."""
    import Tensile.BenchmarkProblems as benchmark_problems

    def serial_map(function, objects, *_args, **_kwargs):
        return [function(*args) for args in objects]

    monkeypatch.setattr(benchmark_problems, "ParallelMap2", serial_map)
    emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert "reject: TDMSplit is currently disabled" in capsys.readouterr().out


_HALFPLR_PROBLEM = {
    "OperationType": "GEMM",
    "DestDataType": "s",
    "ComputeDataType": "s",
    "HighPrecisionAccumulate": True,
    "TransposeA": True,
    "TransposeB": False,
    "UseBeta": True,
    "Batched": True,
    "Activation": False,
    "SupportUserArgs": True,
    "UseBias": 0,
}
_HALFPLR_ONE_SCALE = dict(_HALFPLR_PROBLEM, MacDataTypeA="F8", MacDataTypeB="F8", DataType="F8",
                          DataTypeMXSA="E8", DataTypeMXSB="E8")
_HALFPLR_PROBLEMS = {
    "mx_a": dict(_HALFPLR_ONE_SCALE, MXBlockA=32, MXBlockB=0),
    "mx_b": dict(_HALFPLR_ONE_SCALE, MXBlockA=0, MXBlockB=32),
    "no_mx": dict(_HALFPLR_PROBLEM, DataType="b", DestDataType="b"),
}
_HALFPLR_TENSORS = {"mx_a": ("A", "MXSA", "B"), "mx_b": ("A", "MXSB", "B"), "no_mx": ("A", "B")}
# Even wave tiles, as HalfPLR requires, by (DataType, NumWaves).
_HALFPLR_TILES = {
    ("F8", 1): [16, 16, 128, 1, 1, 4, 4, 1, 1],
    ("F8", 2): [16, 16, 128, 1, 1, 2, 4, 2, 1],
    ("F8", 4): [16, 16, 128, 1, 1, 2, 4, 2, 2],
    ("b", 1): [16, 16, 32, 1, 1, 2, 2, 1, 1],
}
_HALFPLR_DEPTHU = {"F8": 256, "b": 128}
_HALFPLR_FORK = {
    "TDMInst": 3,
    "TDMSplit": False,
    "PrefetchGlobalRead": 2,
    "ScheduleIterAlg": 4,
    "TransposeLDS": -1,
    "LdsPadA": -1,
    "LdsPadB": -1,
    "LdsBlockSizePerPadA": -1,
    "LdsBlockSizePerPadB": -1,
    "LdsPadMetadata": 0,
    "LDSTrInst": False,
    "PrefetchLocalRead": 1,
    "StoreRemapVectorWidth": 0,
    "GlobalSplitU": 1,
    "GlobalSplitUAlgorithm": "MultipleBuffer",
    "GlobalReadVectorWidthA": -1,
    "GlobalReadVectorWidthB": -1,
    "VectorWidthA": -1,
    "VectorWidthB": -1,
    "LocalReadVectorWidth": -1,
    "1LDSBuffer": 0,
    "SourceSwap": False,
    "WorkGroupMapping": 1,
    "StoreVectorWidth": -1,
    "WavefrontSize": 32,
    "ExpandPointerSwap": False,
}

# (problem, TDMFuse, NumWaves, HalfPLR, StaggerU) -> tensors whose removeStagger reads a copy:
# every live tensor on one wave; on more waves the lone scale of TDMFuse=0 and the data tensor
# beside the scale's set on TDMFuse 1-3.
_HALFPLR_CELLS = {
    ("mx_a", 0, 1, 1, 0): ("A", "B", "MXSA"),
    ("mx_b", 0, 1, 2, 16): ("A", "B", "MXSB"),
    ("no_mx", 0, 1, 1, 0): ("A", "B"),
    ("no_mx", 0, 1, 3, 32): ("A", "B"),
    ("mx_a", 0, 2, 1, 0): ("MXSA",),
    ("mx_b", 0, 2, 1, 32): ("MXSB",),
    ("mx_a", 0, 4, 3, 32): ("MXSA",),
    ("mx_b", 0, 4, 2, 0): ("MXSB",),
    ("mx_a", 1, 2, 2, 0): ("B",),
    ("mx_b", 1, 2, 1, 0): ("A",),
    ("mx_a", 1, 4, 1, 32): ("B",),
    ("mx_b", 1, 4, 3, 16): ("A",),
    ("mx_a", 2, 2, 3, 0): ("B",),
    ("mx_b", 2, 2, 2, 32): ("B",),
    ("mx_a", 2, 4, 1, 0): ("B",),
    ("mx_b", 2, 4, 1, 0): ("B",),
    ("mx_a", 3, 2, 1, 32): ("A",),
    ("mx_b", 3, 2, 3, 0): ("A",),
    ("mx_a", 3, 4, 2, 0): ("A",),
    ("mx_b", 3, 4, 2, 32): ("A",),
}


def _halfPlrConfig(problem, fuse, numWaves, halfPLR, staggerU):
    dataType = _HALFPLR_PROBLEMS[problem]["DataType"]
    fork = dict(_HALFPLR_FORK, MatrixInstruction=_HALFPLR_TILES[(dataType, numWaves)],
                DepthU=_HALFPLR_DEPTHU[dataType], TDMFuse=fuse, HalfPLR=halfPLR,
                StaggerU=staggerU, ClusterDim=[1, 1])
    return {
        "GlobalParameters": {"PrintSolutionRejectionReason": True},
        "BenchmarkProblems": [[_HALFPLR_PROBLEMS[problem], {
            "BenchmarkCommonParameters": [{"KernelLanguage": ["Assembly"]}],
            "ForkParameters": [{key: [value]} for key, value in fork.items()],
        }]],
    }


_SGPR_SET = re.compile(r"^\s*\.set\s+(sgpr\w+)\s*,\s*(\w+)\s*(?:\+\s*(\d+))?\s*$", re.M)
_SGPR_NAME = re.compile(r"s\[sgpr(\w+?)(?:\+\d+)?\]")
_ENABLE_WORD = re.compile(r"^\s*(s_mov_b32|s_cmov_b32)\s+s\[(sgprtdm\w+Group0)\+0\],\s*([01])\b")
_TAIL_ENABLE = "HalfPLR: re-enable TDM"


class _HalfPlrKernel:
    """One emitted kernel, read at the register level."""

    def __init__(self, source):
        self.source = source
        # The first definition is the one the loop uses; the kernel releases SGPRs as UNDEF later.
        self.sets = {}
        for name, value, offset in _SGPR_SET.findall(source):
            if value != "UNDEF":
                self.sets.setdefault(name, (value, int(offset or 0)))
        self.prologue, sep, rest = source.partition("label_LoopBeginL:")
        assert sep, "no main loop"
        self.loop = rest.partition("label_LoopEndL:")[0]
        self.staggerReads = []
        for line in source.splitlines():
            code, _, comment = line.partition("//")
            if "start offset S in bytes" in comment and code.strip().startswith("s_mul"):
                self.staggerReads.extend(_SGPR_NAME.findall(code))

    def index(self, name):
        """The SGPR a .set name lands on, through aliases."""
        value, offset = self.sets[name]
        return (int(value) if value.isdigit() else self.index(value)) + offset

    def enableWords(self):
        """Group0 enable words, as SGPR indices, the main loop zeroes and the HalfPLR tail sets."""
        zeroed = set()
        for line in self.loop.splitlines():
            match = _ENABLE_WORD.match(line)
            if match and match.group(1) == "s_cmov_b32" and match.group(3) == "0":
                zeroed.add(self.index(match.group(2)))
        _, sep, tail = self.source.partition(_TAIL_ENABLE)
        assert sep, "no HalfPLR tail"
        enabled = set()
        for line in tail.splitlines()[1:16]:
            match = _ENABLE_WORD.match(line)
            if match and match.group(1) == "s_mov_b32" and match.group(3) == "1":
                enabled.add(self.index(match.group(2)))
        return zeroed, enabled


@pytest.fixture(scope="module")
def halfPlrKernel(tmp_path_factory):
    """Derive and emit one HalfPLR kernel per cell, once."""
    root = tmp_path_factory.mktemp("s01_halfplr_k_tail")
    kernels = {}

    def emit(problem, fuse, numWaves, halfPLR, staggerU):
        cell = (problem, fuse, numWaves, halfPLR, staggerU)
        if cell not in kernels:
            path = root / ("%s_tdmfuse%d_w%d_halfplr%d_staggeru%d.yaml" % cell)
            path.write_text(yaml.safe_dump(_halfPlrConfig(*cell)))
            states = derive_states(path, arch=_ARCH, limit_solutions=1)
            assert len(states) == 1, "%s derives no solution" % (cell,)
            assert (states[0]["TDMFuse"], states[0]["NumWaves"], states[0]["HalfPLR"]) \
                == (fuse, numWaves, halfPLR), cell
            results = emit_kernels_from_config(path, limit=1, arch=_ARCH)
            assert len(results) == 1, cell
            name, source, err = results[0]
            assert err == 0, (cell, name)
            kernels[cell] = _HalfPlrKernel(source)
        return kernels[cell]

    return emit


@pytest.mark.parametrize("problem, fuse, numWaves, halfPLR, staggerU", sorted(_HALFPLR_CELLS))
def test_s01_halfplr_tail_reenables_every_disabled_set(
        halfPlrKernel, problem, fuse, numWaves, halfPLR, staggerU):
    """The K tail loads through every descriptor set, so each enable word the loop zeroed is set."""
    kernel = halfPlrKernel(problem, fuse, numWaves, halfPLR, staggerU)
    zeroed, enabled = kernel.enableWords()
    scales = {kernel.index("sgprtdm%sGroup0" % tc) for tc in ("MXSA", "MXSB")
              if _HALFPLR_PROBLEMS[problem].get("MXBlock" + tc[-1])}
    assert scales <= zeroed, (sorted(scales), sorted(zeroed))
    assert zeroed and zeroed <= enabled, sorted(zeroed - enabled)


@pytest.mark.parametrize("problem, fuse, numWaves, halfPLR, staggerU", sorted(_HALFPLR_CELLS))
def test_s01_halfplr_remove_stagger_reads_the_intact_copy(
        halfPlrKernel, problem, fuse, numWaves, halfPLR, staggerU):
    """Each tensor whose own increment the mask zeroes rebases the K tail from its copy."""
    cell = (problem, fuse, numWaves, halfPLR, staggerU)
    kernel = halfPlrKernel(*cell)
    masked = _HALFPLR_CELLS[cell]
    prefix = "sgprtdmRemoveStaggerIncs"
    assert {name[len(prefix):] for name in kernel.sets if name.startswith(prefix)} == set(masked)
    reads = sorted(set(kernel.staggerReads))
    for tc in _HALFPLR_TENSORS[problem]:
        copy, own = "tdmRemoveStaggerIncs" + tc, "GlobalReadIncs" + tc
        if tc in masked:
            assert "s_mov_b32 s[sgpr%s], s[sgpr%s+0]" % (copy, own) in kernel.prologue, tc
            assert "s_cmov_b32 s[sgpr%s], 0" % own in kernel.source, tc
            assert copy in reads and own not in reads, (tc, reads)
        else:
            assert own in reads and copy not in reads, (tc, reads)
