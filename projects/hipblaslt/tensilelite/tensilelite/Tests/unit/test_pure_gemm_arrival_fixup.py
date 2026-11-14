# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# Codegen test: the pure-gemm store pass flag must outlive StreamK's writePartials.
#
# Under the StreamK arrival fixup the last arriving part runs writePartials, jumps
# back to SK_ArrivalFixup and falls through SK_Store into the pure-gemm branch, so
# the flag sgpr computed before the prelude must still hold its value there. The
# generator asserts that registers live across writePartials stay checked out.
#
# CompactLoopStore needs v_movrelsd_2_b32 and the arrival fixup needs scalar
# atomics; no shipping architecture has both (gfx950 has the atomics, gfx1250 the
# instruction). The test gives gfx950 the missing capability and checks codegen
# only: gfx950 cannot assemble the CLS store, so run.main ends with "No valid
# solutions found" after the assembly is written.
################################################################################

import re
import shutil
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

from tensilelite import run

_CONFIG = """
GlobalParameters:
  KeepBuildTmp: True
  NumElementsToValidate: -1
  BoundsCheck: 0
  KernelTime: False
  DataInitTypeAlpha: 1
  DataInitTypeBeta: 1
  MaxWorkspaceSize: 134217728
  NumWarmups: 0
  EnqueuesPerSync: 1
  Architecture: gfx950

BenchmarkProblems:
  - # SGEMM NT, StreamK DynamicWorkQueue + bias + scaleAlphaVec + runtime activation, CompactLoopStore on
    - # ProblemType
      OperationType: GEMM
      DataType: s
      TransposeA: False
      TransposeB: True
      UseBeta: True
      Batched: True
      Activation: True
      ActivationType: all
      UseBias: 1
      BiasDataTypeList: ['s']
      UseScaleAlphaVec: 1
      ActivationFused: true
      SupportUserArgs: True

    - InitialSolutionParameters:
      BenchmarkCommonParameters:
        - KernelLanguage: ["Assembly"]
      ForkParameters:
        - TileProcessingStrategy: [StreamK]
        - WorkAssignment: [DynamicWorkQueue]
        - 1LDSBuffer: [1]
        - DepthU: [16]
        - ExpandPointerSwap: [False]
        - GlobalReadVectorWidthA: [1]
        - GlobalReadVectorWidthB: [1]
        - GlobalSplitU: [0]
        - LocalReadVectorWidth: [1]
        - MatrixInstruction:
          - [32, 32, 2, 1, 1, 2,2, 2,2]
        - MIArchVgpr: [False]
        - PrefetchGlobalRead: [2]
        - PrefetchLocalRead: [1]
        - ScheduleIterAlg: [3]
        - SourceSwap: [True]
        - StoreRemapVectorWidth: [0]
        - TransposeLDS: [0]
        - VectorWidthA: [1]
        - VectorWidthB: [1]
        - WorkGroupMapping: [1]
        - CompactLoopStore: [True]
      BenchmarkJoinParameters:
      BenchmarkFinalParameters:
        - ProblemSizes:
          - Exact: [512, 512, 1, 256]
        - BiasTypeArgs: ['s']
        - ActivationArgs:
          - [Enum: none]
"""

_ASSEMBLER = shutil.which("amdclang++") or "/opt/rocm/bin/amdclang++"

_SCALAR_WRITE = re.compile(r"^(s_[a-z0-9_]+|v_readfirstlane_b32|v_readlane_b32)\s+s\[?(\d+)(?::(\d+))?\]?[, ]")
_NOT_A_WRITE = ("s_cmp", "s_bitcmp", "s_cbranch", "s_wait", "s_nop", "s_barrier", "s_branch", "s_setpc",
                "s_prefetch", "s_set_vgpr", "s_store", "s_atomic")


@pytest.mark.skipif(not Path(_ASSEMBLER).exists(), reason="no assembler to probe the gfx950 capabilities")
def test_pure_flag_outlives_the_streamk_arrival_fixup(tensile_args, tmp_path, monkeypatch):
    orig = run.applyArchCapOverrides

    def withMovrelsD2B32(isaInfoMap, archNames):
        orig(isaInfoMap, archNames)
        for info in isaInfoMap.values():
            info.asmCaps["HasMovRelsD2B32"] = True

    monkeypatch.setattr(run, "applyArchCapOverrides", withMovrelsD2B32)

    config = tmp_path / "sk_dyn_cls_bias.yaml"
    config.write_text(_CONFIG)
    output_dir = tmp_path / "output"
    try:
        run.main([str(config), str(output_dir), "--build-only", "--global-parameters", "KeepBuildTmp=True",
                  *tensile_args, "--gpu-targets", "gfx950"])
    except RuntimeError as e:
        # gfx950 cannot assemble the CLS store; an AssertionError from codegen is not caught here.
        assert "No valid solutions found" in str(e)

    asm_files = list(output_dir.rglob("*.s"))
    assert asm_files, "no assembly generated"
    lines = asm_files[0].read_text().split("\n")

    assert any(l.startswith("label_SK_ArrivalFixup:") for l in lines), "kernel does not use the arrival fixup"
    labels = {m.group(1): i for i, l in enumerate(lines) if (m := re.match(r"^(label_[A-Za-z0-9_]+):", l))}
    assert "label_GW_Pure" in labels, "no pure-gemm store pass generated"

    flag = next(i for i, l in enumerate(lines) if l.startswith("s_mov_b32") and "pure-gemm flag" in l)
    reg = int(re.match(r"s_mov_b32 s(\d+), 1", lines[flag]).group(1))

    # Everything before the pure pass runs ahead of the pure-gemm branch, including writePartials and
    # the arrival code (they sit after SK_Store in the text but run before it): none may rewrite the flag.
    rewrites = []
    for i in range(flag + 1, labels["label_GW_Pure"]):
        m = _SCALAR_WRITE.match(lines[i])
        if not m or m.group(1).startswith(_NOT_A_WRITE):
            continue
        lo = int(m.group(2))
        hi = int(m.group(3)) if m.group(3) else lo
        if lo <= reg <= hi and "not pure" not in lines[i]:
            rewrites.append((i + 1, lines[i]))
    assert not rewrites, "the pure-gemm flag s%d is rewritten before the pure-gemm branch: %s" % (reg, rewrites[:3])
