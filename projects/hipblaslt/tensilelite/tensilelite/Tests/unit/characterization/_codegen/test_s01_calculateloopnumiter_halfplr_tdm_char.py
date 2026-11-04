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

The emitted cells cover HalfPLR's K tail. HalfPLR's last-iteration mask zeroes
the increment each TDM set advances by, and after the main loop removeStagger
rebases every descriptor for the K tail. A parity or paired set advances through
a shadow increment, which the mask zeroes instead, so its GlobalReadIncs
survives. A tensor that advances by its own GlobalReadIncs loses it: every live
tensor on one wave, both tensors of TDMFuse 1-3 without MX scales, the data
tensor beside the scale set of TDMFuse 1-3, and a lone scale on TDMFuse=0. On
more than one wave such a tensor keeps a copy for removeStagger; one wave keeps
none. A workgroup cluster turns the writer's
stagger code off, so a clustered kernel has no removeStagger and keeps no copy.
Each kernel is checked at the register level:

* the copy is defined, and primed from GlobalReadIncs before the main loop,
  exactly for the tensors whose GlobalReadIncs the mask zeroes in a kernel with
  removeStagger, and the mask still zeroes them;
* removeStagger multiplies by the copy for those tensors and by GlobalReadIncs
  for the others; a clustered kernel has no removeStagger multiply at all;
* after the main loop the HalfPLR adjust sets every group0 enable word the last
  iterations zeroed back to 1.

CPU-only: Python and rocisa codegen, nothing is assembled or launched.
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
    import tensilelite.BenchmarkProblems as benchmark_problems

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
_HALFPLR_MX = dict(_HALFPLR_PROBLEM, MacDataTypeA="F8", MacDataTypeB="F4", DataType="F8",
                   MXBlockA=32, MXBlockB=32, DataTypeMXSA="E8", DataTypeMXSB="E8")
_HALFPLR_PROBLEMS = {
    # The two problem types of decouple_pgr_tdm_fuse_gfx1250.yaml, and MX scales on one tensor.
    "mx": _HALFPLR_MX,
    "no_mx": dict(_HALFPLR_PROBLEM, DataType="b", DestDataType="b"),
    "mx_a": dict(_HALFPLR_MX, MacDataTypeB="F8", MXBlockB=0),
    "mx_b": dict(_HALFPLR_MX, MacDataTypeB="F8", MXBlockA=0),
}
_HALFPLR_TENSORS = {"mx": ("A", "MXSA", "MXSB", "B"), "no_mx": ("A", "B"),
                    "mx_a": ("A", "MXSA", "B"), "mx_b": ("A", "MXSB", "B")}
# Even wave tiles, as HalfPLR requires, by (DataType, NumWaves).
_HALFPLR_TILES = {
    ("F8", 2): [16, 16, 128, 1, 1, 2, 4, 2, 1],
    ("F8", 4): [16, 16, 128, 1, 1, 2, 4, 2, 2],
    ("b", 1): [16, 16, 32, 1, 1, 2, 2, 1, 1],
    ("b", 2): [16, 16, 32, 1, 1, 2, 2, 2, 1],
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

_ONE = (1, 1)
# (problem, TDMFuse, NumWaves, HalfPLR, StaggerU, ClusterDim) -> tensors whose removeStagger
# reads a copy.
_HALFPLR_CELLS = {
    # MX on both tensors (64x128 on 4 waves) and none (64x32 on 2 waves).
    ("mx", 2, 4, 1, 0, _ONE): ("B",),
    ("mx", 2, 4, 2, 0, _ONE): ("B",),
    ("mx", 3, 4, 2, 0, _ONE): ("A",),
    ("mx", 2, 4, 3, 32, _ONE): ("B",),
    ("mx", 3, 4, 1, 32, _ONE): ("A",),
    ("mx", 0, 4, 1, 0, _ONE): (),
    ("mx", 1, 4, 1, 0, _ONE): (),
    ("mx", 2, 4, 0, 0, _ONE): (),
    ("mx", 2, 4, 1, 0, (2, 4)): (),
    ("mx", 3, 4, 2, 0, (4, 2)): (),
    ("no_mx", 1, 2, 1, 0, _ONE): ("A", "B"),
    ("no_mx", 3, 2, 3, 0, _ONE): ("A", "B"),
    ("no_mx", 2, 2, 2, 32, _ONE): ("A", "B"),
    ("no_mx", 3, 2, 2, 32, _ONE): ("A", "B"),
    ("no_mx", 0, 2, 1, 0, _ONE): (),
    ("no_mx", 2, 2, 0, 0, _ONE): (),
    ("no_mx", 1, 2, 1, 0, (2, 4)): (),
    # One wave keeps no copy.
    ("no_mx", 0, 1, 1, 0, _ONE): (),
    ("no_mx", 0, 1, 3, 32, _ONE): (),
    # MX on one tensor: the lone scale on TDMFuse=0, the data tensor beside the scale's set on 1-3.
    ("mx_a", 0, 2, 1, 0, _ONE): ("MXSA",),
    ("mx_b", 0, 2, 1, 32, _ONE): ("MXSB",),
    ("mx_a", 0, 4, 3, 32, _ONE): ("MXSA",),
    ("mx_b", 0, 4, 2, 0, _ONE): ("MXSB",),
    ("mx_b", 0, 2, 1, 0, (2, 4)): (),
    ("mx_a", 1, 2, 2, 0, _ONE): ("B",),
    ("mx_b", 1, 2, 1, 0, _ONE): ("A",),
    ("mx_a", 1, 4, 1, 32, _ONE): ("B",),
    ("mx_b", 1, 4, 3, 16, _ONE): ("A",),
    ("mx_a", 2, 2, 3, 0, _ONE): ("B",),
    ("mx_b", 2, 2, 2, 32, _ONE): ("B",),
    ("mx_a", 2, 4, 1, 0, _ONE): ("B",),
    ("mx_b", 2, 4, 1, 0, _ONE): ("B",),
    ("mx_a", 3, 2, 1, 32, _ONE): ("A",),
    ("mx_b", 3, 2, 3, 0, _ONE): ("A",),
    ("mx_a", 3, 4, 2, 0, _ONE): ("A",),
    ("mx_b", 3, 4, 2, 32, _ONE): ("A",),
}
_HALFPLR_IDS = ["%s-tdmfuse%d-w%d-halfplr%d-staggeru%d-cluster%dx%d" % (cell[:5] + cell[5])
                for cell in _HALFPLR_CELLS]
_HALFPLR_TAIL_CELLS = [cell for cell in _HALFPLR_CELLS if cell[3]]
_HALFPLR_TAIL_IDS = [i for cell, i in zip(_HALFPLR_CELLS, _HALFPLR_IDS) if cell[3]]


def _halfPlrConfig(problem, fuse, numWaves, halfPLR, staggerU, cluster):
    dataType = _HALFPLR_PROBLEMS[problem]["DataType"]
    fork = dict(_HALFPLR_FORK, MatrixInstruction=_HALFPLR_TILES[(dataType, numWaves)],
                DepthU=_HALFPLR_DEPTHU[dataType], TDMFuse=fuse, HalfPLR=halfPLR,
                StaggerU=staggerU, ClusterDim=list(cluster))
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

    def emit(problem, fuse, numWaves, halfPLR, staggerU, cluster):
        cell = (problem, fuse, numWaves, halfPLR, staggerU, cluster)
        if cell not in kernels:
            path = root / ("%s.yaml" % _HALFPLR_IDS[list(_HALFPLR_CELLS).index(cell)])
            path.write_text(yaml.safe_dump(_halfPlrConfig(*cell)))
            states = derive_states(path, arch=_ARCH, limit_solutions=1)
            assert len(states) == 1, "%s derives no solution" % (cell,)
            assert (states[0]["TDMFuse"], states[0]["NumWaves"], states[0]["HalfPLR"],
                    tuple(states[0]["ClusterDim"])) == (fuse, numWaves, halfPLR, cluster), cell
            results = emit_kernels_from_config(path, limit=1, arch=_ARCH)
            assert len(results) == 1, cell
            name, source, err = results[0]
            assert err == 0, (cell, name)
            kernels[cell] = _HalfPlrKernel(source)
        return kernels[cell]

    return emit


@pytest.mark.parametrize("cell", _HALFPLR_TAIL_CELLS, ids=_HALFPLR_TAIL_IDS)
def test_s01_halfplr_tail_reenables_every_disabled_set(halfPlrKernel, cell):
    """The K tail loads through every descriptor set, so each enable word the loop zeroed is set."""
    kernel = halfPlrKernel(*cell)
    zeroed, enabled = kernel.enableWords()
    scales = {kernel.index("sgprtdm%sGroup0" % tc) for tc in ("MXSA", "MXSB")
              if _HALFPLR_PROBLEMS[cell[0]].get("MXBlock" + tc[-1])}
    assert scales <= zeroed, (sorted(scales), sorted(zeroed))
    assert zeroed and zeroed <= enabled, sorted(zeroed - enabled)


@pytest.mark.parametrize("cell", _HALFPLR_CELLS, ids=_HALFPLR_IDS)
def test_s01_halfplr_copy_is_primed_before_the_main_loop_and_the_mask_is_unchanged(
        halfPlrKernel, cell):
    """Each tensor whose own increment the mask zeroes keeps a copy filled before the loop."""
    kernel = halfPlrKernel(*cell)
    masked = _HALFPLR_CELLS[cell]
    prefix = "sgprtdmRemoveStaggerIncs"
    assert {name[len(prefix):] for name in kernel.sets if name.startswith(prefix)} == set(masked)
    for tc in masked:
        assert "s_mov_b32 s[%s%s], s[sgprGlobalReadIncs%s+0]" % (prefix, tc, tc) \
            in kernel.prologue, tc
        assert "s_cmov_b32 s[sgprGlobalReadIncs%s], 0" % tc in kernel.source, tc


@pytest.mark.parametrize("cell", _HALFPLR_CELLS, ids=_HALFPLR_IDS)
def test_s01_halfplr_remove_stagger_multiplies_by_an_increment_the_mask_left_intact(
        halfPlrKernel, cell):
    """removeStagger reads the copy for each masked tensor and GlobalReadIncs for the others."""
    reads = sorted(set(halfPlrKernel(*cell).staggerReads))
    if cell[5] != _ONE:
        assert not reads, reads
        return
    masked = _HALFPLR_CELLS[cell]
    for tc in _HALFPLR_TENSORS[cell[0]]:
        copy, own = "tdmRemoveStaggerIncs" + tc, "GlobalReadIncs" + tc
        if tc in masked:
            assert copy in reads and own not in reads, (tc, reads)
        else:
            assert own in reads and copy not in reads, (tc, reads)
