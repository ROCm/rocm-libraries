################################################################################
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
"""gfx1250 HalfPLR TDM kernels rebase the K tail by an intact increment (CPU-only characterization).

HalfPLR's last-iteration mask zeroes the increment each TDM set advances by, and after the main
loop removeStagger rebases every descriptor for the K tail by its tensor's GlobalReadIncs. A
parity or paired set advances through a shadow increment, which the mask zeroes instead, so its
GlobalReadIncs survives. A tensor that advances by its own GlobalReadIncs loses it: both tensors
of TDMFuse 1-3 without MX scales, and the data tensor beside the scale set of TDMFuse 2/3 with MX
on both tensors. Such a tensor keeps a copy for removeStagger. A workgroup cluster turns the
writer's stagger code off, so a clustered kernel has no removeStagger and keeps no copy.

Emits one kernel per cell from the two problem types of decouple_pgr_tdm_fuse_gfx1250.yaml and
checks it at the register level:

* the copy is defined, and primed from GlobalReadIncs before the main loop, exactly for the
  tensors whose GlobalReadIncs the mask zeroes in a kernel with removeStagger, and the mask
  still zeroes them;
* removeStagger multiplies by the copy for those tensors and by GlobalReadIncs for the others;
  a clustered kernel has no removeStagger multiply at all.

CPU-only: the emit harness runs Python+rocisa codegen; nothing is assembled or launched.
"""

import re

import pytest
import yaml

from config_harness import derive_states, emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx1250"

_PROBLEM = {
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
_PROBLEMS = {
    "mx": dict(_PROBLEM, MacDataTypeA="F8", MacDataTypeB="F4", DataType="F8", MXBlockA=32,
               MXBlockB=32, DataTypeMXSA="E8", DataTypeMXSB="E8"),
    "bf16": dict(_PROBLEM, DataType="b", DestDataType="b"),
}
_TENSORS = {"mx": ("A", "MXSA", "MXSB", "B"), "bf16": ("A", "B")}

# 64x128 on waves [2,2] for MX, 64x32 on waves [2,1] for bf16.
_TILE = {
    "mx": {"MatrixInstruction": [16, 16, 128, 1, 1, 2, 4, 2, 2], "DepthU": 256},
    "bf16": {"MatrixInstruction": [16, 16, 32, 1, 1, 2, 2, 2, 1], "DepthU": 128},
}
_NUM_WAVES = {"mx": 4, "bf16": 2}

# (problem, TDMFuse, HalfPLR, StaggerU, ClusterDim) -> tensors whose removeStagger reads the copy.
_CELLS = {
    ("mx", 2, 1, 0, (1, 1)): ("B",),
    ("mx", 3, 2, 0, (1, 1)): ("A",),
    ("mx", 2, 3, 32, (1, 1)): ("B",),
    ("mx", 0, 1, 0, (1, 1)): (),
    ("mx", 1, 1, 0, (1, 1)): (),
    ("mx", 2, 0, 0, (1, 1)): (),
    ("mx", 2, 1, 0, (2, 4)): (),
    ("mx", 3, 2, 0, (4, 2)): (),
    ("bf16", 1, 1, 0, (1, 1)): ("A", "B"),
    ("bf16", 3, 3, 0, (1, 1)): ("A", "B"),
    ("bf16", 2, 2, 32, (1, 1)): ("A", "B"),
    ("bf16", 0, 1, 0, (1, 1)): (),
    ("bf16", 2, 0, 0, (1, 1)): (),
}
_CELL_IDS = ["%s-tdmfuse%d-halfplr%d-staggeru%d-cluster%dx%d" % (cell[:4] + cell[4]) for cell in _CELLS]

_FORK = {
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


def _config(problem, fuse, halfPLR, staggerU, cluster):
    fork = dict(_FORK, **_TILE[problem], TDMFuse=fuse, HalfPLR=halfPLR, StaggerU=staggerU,
                ClusterDim=list(cluster))
    return {
        "GlobalParameters": {"PrintSolutionRejectionReason": True},
        "BenchmarkProblems": [[_PROBLEMS[problem], {
            "BenchmarkCommonParameters": [{"KernelLanguage": ["Assembly"]}],
            "ForkParameters": [{key: [value]} for key, value in fork.items()],
        }]],
    }


_SET = re.compile(r"^\s*\.set\s+sgpr(\w+)\s*,", re.M)
_LOOP_BEGIN = "label_LoopBeginL:"
_SGPR_NAME = re.compile(r"s\[sgpr(\w+?)(?:\+\d+)?\]")


class _Kernel:
    def __init__(self, source):
        self.source = source
        self.sgprs = set(_SET.findall(source))
        self.prologue = source.split(_LOOP_BEGIN, 1)[0]
        assert self.prologue != source, "no main loop"
        self.staggerMultiplies = []
        for line in source.splitlines():
            code, _, comment = line.partition("//")
            if "start offset S in bytes" in comment and code.strip().startswith("s_mul"):
                self.staggerMultiplies.extend(_SGPR_NAME.findall(code))


@pytest.fixture(scope="module")
def emitted(tmp_path_factory):
    """Derive and emit each cell's kernel once, on first use."""
    root = tmp_path_factory.mktemp("tdm_halfplr_remove_stagger")
    kernels = {}

    def emit(problem, fuse, halfPLR, staggerU, cluster):
        cell = (problem, fuse, halfPLR, staggerU, cluster)
        if cell not in kernels:
            path = root / ("%s_tdmfuse%d_halfplr%d_staggeru%d_cluster%dx%d.yaml" % (cell[:4] + cluster))
            path.write_text(yaml.safe_dump(_config(*cell)))
            states = derive_states(path, arch=_ARCH, limit_solutions=1)
            assert len(states) == 1, "%s derives no solution" % (cell,)
            assert (states[0]["TDMFuse"], states[0]["HalfPLR"], states[0]["NumWaves"],
                    tuple(states[0]["ClusterDim"])) == (fuse, halfPLR, _NUM_WAVES[problem], cluster), cell
            results = emit_kernels_from_config(path, limit=1, arch=_ARCH)
            assert len(results) == 1, cell
            base, source, err = results[0]
            assert err == 0, (cell, base)
            assert "gfx1250" in source, (cell, base)
            kernels[cell] = _Kernel(source)
        return kernels[cell]

    return emit


@pytest.mark.parametrize("cell", _CELLS, ids=_CELL_IDS)
def test_the_copy_is_primed_before_the_main_loop_and_the_mask_is_unchanged(emitted, cell):
    kernel = emitted(*cell)
    copies = {tc for tc in _TENSORS[cell[0]] if "tdmRemoveStaggerIncs" + tc in kernel.sgprs}
    assert copies == set(_CELLS[cell])
    for tc in _CELLS[cell]:
        assert "s_mov_b32 s[sgprtdmRemoveStaggerIncs%s], s[sgprGlobalReadIncs%s+0]" % (tc, tc) \
            in kernel.prologue, tc
        assert "s_cmov_b32 s[sgprGlobalReadIncs%s], 0" % tc in kernel.source, tc


@pytest.mark.parametrize("cell", _CELLS, ids=_CELL_IDS)
def test_remove_stagger_multiplies_by_an_increment_the_mask_left_intact(emitted, cell):
    reads = emitted(*cell).staggerMultiplies
    if cell[4] != (1, 1):
        assert not reads, sorted(set(reads))
        return
    for tc in _TENSORS[cell[0]]:
        copy, own = "tdmRemoveStaggerIncs" + tc, "GlobalReadIncs" + tc
        if tc in _CELLS[cell]:
            assert copy in reads and own not in reads, (tc, sorted(set(reads)))
        else:
            assert own in reads and copy not in reads, (tc, sorted(set(reads)))
