# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Codegen checks for single-wave TDM kernels (HalfPLR tail, StreamK, clusters)."""
from pathlib import Path
import re

import pytest

from config_harness import assert_assembles, emit_kernels_from_config

pytestmark = pytest.mark.unit
_CONFIG = Path(__file__).resolve().parents[1] / "common/gemm/gfx12/tdm_single_wave_gfx1250.yaml"


def _instructions(source):
    lines = [line.split("//", 1)[0].strip() for line in source.splitlines()]
    return [line for line in lines if line]


@pytest.fixture(scope="module")
def kernels():
    # HalfPLR x DepthU x ClusterDim x StreamK; HalfPLR needs DepthU 64 here
    # (LoopIters > 1), so 12 of the 16 forks survive.
    results = emit_kernels_from_config(_CONFIG, limit=16, arch="gfx1250", canonical=False,
                                       expected_fork_count=16)
    assert len(results) == 12
    for name, _, error in results:
        assert error == 0, (name, error)
    return results


def test_kernels_assemble(kernels):
    for name, source, _ in kernels:
        assert_assembles(source, name)


def test_halfplr_keeps_global_read_increments(kernels):
    # The tail loop, the stagger removal and the StreamK tile offset read the
    # increments after the unrolled loop, so they must not be cleared in place.
    halfPLR = [(name, source) for name, source, _ in kernels if "_HPLR1_" in source]
    assert halfPLR
    for name, source in halfPLR:
        for line in _instructions(source):
            assert not re.match(r"s_c?mov_b32 s\[sgprGlobalReadIncs[AB]\]", line), (name, line)
        assert "no increment on the last iters" in source, name


def test_streamk_unrolled_loop_waits_for_tensor_loads(kernels):
    # Without a cross-iteration local-read prefetch the loop body reads the
    # buffer the previous trip's tensor load filled; the drained StreamK
    # prologue must not hide that wait.
    streamK = [(name, source) for name, source, _ in kernels
               if "_GSU0_" in source and "_MT32x16x32_" in source]
    assert streamK
    for name, source in streamK:
        assert "For stream-k / persistent loop" not in source, name
        insts = _instructions(source)
        loopBegin = insts.index("label_LoopBeginL:")
        firstRead = next(i for i in range(loopBegin, len(insts))
                         if insts[i].startswith("ds_load"))
        assert any(line.startswith("s_wait_tensorcnt")
                   for line in insts[loopBegin:firstRead]), name


_STORE_SWAP_ADDR_CONFIG = """
GlobalParameters:
  PrintSolutionRejectionReason: True
BenchmarkProblems:
  -
    - OperationType: GEMM
      DataType: b
      DestDataType: b
      ComputeDataType: s
      HighPrecisionAccumulate: True
      TransposeA: False
      TransposeB: False
      UseBeta: True
      Batched: True
    - InitialSolutionParameters:
      BenchmarkCommonParameters:
        - KernelLanguage: ["Assembly"]
      ForkParameters:
        - MatrixInstruction:
          - [16, 16, 32, 1, 1, 2, 17, 1, 1]
        - DepthU: [256]
        - WavefrontSize: [32]
        - ClusterLocalRead: [0]
        - GlobalSplitU: [0]
        - PrefetchGlobalRead: [2]
        - PrefetchLocalRead: [1]
        - ScheduleIterAlg: [4]
        - StreamK: [3]
        - TDMInst: [3]
        - WorkGroupMapping: [8]
        - LDSTrInst: [True]
      BenchmarkJoinParameters:
      BenchmarkFinalParameters:
        - ProblemSizes:
          - Exact: [241, 2176, 1, 256]
"""


def test_store_swap_addr_swaps_each_tensor_descriptor(tmp_path):
    # MT 32x272 at DepthU 256: two A + B buffers fit MaxLDS only without a
    # power-of-two stride, so StoreSwapAddr is set and the single wave moves
    # each of its two descriptors by a compare and add instead of an xor.
    config = tmp_path / "tdm_single_wave_store_swap_addr.yaml"
    config.write_text(_STORE_SWAP_ADDR_CONFIG)
    results = emit_kernels_from_config(config, limit=1, arch="gfx1250", canonical=False,
                                       expected_fork_count=1)
    assert len(results) == 1
    name, source, error = results[0]
    assert error == 0, (name, error)
    assert_assembles(source, name)
    assert re.search(r"\.max_flat_workgroup_size:\s+32\b", source), name
    insts = _instructions(source)
    swaps = {}
    for tc in "AB":
        lds = re.escape("s[sgprtdm%sGroup0+1]" % tc)
        cmps = [m for m in (re.fullmatch(r"s_cmp_lt_u32 %s, (\d+)" % lds, line) for line in insts) if m]
        blks = {int(m.group(1)) for m in cmps}
        assert len(blks) == 1, (name, tc, blks)
        blk = blks.pop()
        assert blk & (blk - 1), (name, tc, blk)
        assert any(re.fullmatch(r"s_add_i32 %s, %s, s\d+" % (lds, lds), line) for line in insts), (name, tc)
        assert not any(re.match(r"s_xor_b32 %s" % lds, line) for line in insts), (name, tc)
        swaps[tc] = len(cmps)
    # Every buffer switch swaps both descriptors.
    assert swaps["A"] == swaps["B"] >= 1, (name, swaps)
