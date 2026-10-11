################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""R7 — ShiftVectorComponents full characterization test.

Targets: tensilelite/Components/ShiftVectorComponents.py
  Primary ranges:   47-200  (ShiftVectorComponentsVALU.__call__)
  Secondary ranges: ShiftVectorComponentsMFMA.__call__ + PartialThread

The gfx1101 int8 case exercises AllThread: four-byte reads exceed the
two-element partial-thread threshold. It covers NT/TT and int8/int32 outputs,
matching the Windows edge failures in ROCm/rocm-libraries#12455.

Strategy
--------
Two config sweeps are combined in one isolated test run:

  A) VALU sweep (existing shiftvector2 config, lines 47-200):
     WorkGroup+ThreadTile kernels with VectorWidthA=4 and
     AssertFree0ElementMultiple=1 so the VALU edge-shift path (lines 47-200)
     runs for all three ThreadTile shapes.  Covers ~105 lines in the
     47-200 target range.

  B) MFMA sweep (new shiftvec_full config, lines 208-546):
     Two gfx942 MFMA shapes ([16,16,16] and [32,32,8]) with VectorWidthA=2
     and AssertFree0ElementMultiple=1 so the MFMA edge-shift path routes to
     ShiftVectorComponentsMFMAPartialThread.  SourceSwap=[0,1] varies the
     thread-coal orientation to cover both conThInProcDim branches at lines
     290-297.

Line 122 (ShiftVectorComponentsVALU, glvw < vectorWidth branch):
  Requires GlobalReadVectorWidthA < VectorWidthA.  Solution.py (lines
  5029-5032) rejects GRVWA > 1 unless GRVWA == VWA; GRVWA=1 gives
  GuaranteeNoPartialA = (1 % 1 == 0) = True, suppressing shiftVectorComponents
  (KernelWriter.py:5894). P5-ceiling: dead code.

pytestmark = pytest.mark.unit per project convention.
CPU-only; no GPU device required.
"""

import os
import re

import pytest
import yaml

from config_harness import derive_states, emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx942"

# VALU config (covers lines 47-200 of ShiftVectorComponentsVALU)
_CFG_VALU = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx942",
    "shiftvector2.yaml",
)

# MFMA config (covers lines 208-546 of ShiftVectorComponentsMFMA + PartialThread)
_CFG_MFMA = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx942",
    "shiftvec_full.yaml",
)


def test_r7_shiftvec_full_valu_emits():
    """ShiftVectorComponentsVALU path (lines 47-200) emits real gfx942 assembly.

    Three VALU kernels (different ThreadTile shapes) with VectorWidthA=4 and
    AssertFree0ElementMultiple=1 so the edge-shift path runs for all shapes.
    Covers ~105 executable lines in the 47-200 target range.
    """
    results = emit_kernels_from_config(_CFG_VALU, limit=8, arch=_ARCH)
    assert len(results) >= 1, (
        f"Expected >=1 VALU kernel from shiftvector2 sweep, got {len(results)}"
    )
    assert all(err == 0 for (_b, _s, err) in results), (
        "All VALU kernels must emit err==0; "
        + str([(b, e) for (b, _s, e) in results if e != 0])
    )
    for base, src, _err in results:
        assert src and len(src.splitlines()) > 50, (
            f"Expected non-trivial asm for {base!r}"
        )
        assert ".amdgcn_target" in src, f"Expected AMDGCN target in {base!r}"
        assert "gfx942" in src, f"Expected gfx942 arch in {base!r}"
        # VALU kernels emit ShiftVectorComponents label sequences
        assert "ShiftVectorComponents" in src, (
            f"Expected ShiftVectorComponents labels in VALU asm {base!r}"
        )


def test_r7_shiftvec_full_mfma_emits():
    """ShiftVectorComponentsMFMAPartialThread (lines 208-546) emits gfx942 asm.

    Two MFMA shapes x SourceSwap=[0,1] with VectorWidthA=2 and
    AssertFree0ElementMultiple=1 drive the MFMA edge-shift partial-thread path.
    At least one kernel must emit successfully; all valid permutations must be
    err==0.

    """
    results = emit_kernels_from_config(_CFG_MFMA, limit=8, arch=_ARCH)
    assert len(results) >= 1, (
        f"Expected >=1 MFMA kernel from shiftvec_full sweep, got {len(results)}"
    )
    assert all(err == 0 for (_b, _s, err) in results), (
        "All MFMA kernels must emit err==0; "
        + str([(b, e) for (b, _s, e) in results if e != 0])
    )
    for base, src, _err in results:
        assert src and len(src.splitlines()) > 50, (
            f"Expected non-trivial asm for {base!r}"
        )
        assert ".amdgcn_target" in src, f"Expected AMDGCN target in {base!r}"
        assert "gfx942" in src, f"Expected gfx942 arch in {base!r}"
        # MFMA kernels also emit ShiftVectorComponents label sequences
        assert "ShiftVectorComponents" in src, (
            f"Expected ShiftVectorComponents labels in MFMA asm {base!r}"
        )


@pytest.mark.parametrize("transpose_a", [False, True], ids=["NT", "TT"])
@pytest.mark.parametrize("dest_type", [8, 6], ids=["int8-output", "int32-output"])
def test_gfx1101_int8_edge_shift(tmp_path, transpose_a, dest_type):
    """Preserve cross-thread B edge shifts for four-element int8 reads.

    Shapes come from navi32's GridBased Alik_Bjlk_I8II solution 1 (TT)
    and Ailk_Bjlk_I8II solution 0 (NT).
    Int8's minimum four-element read width exceeds 1 * (32 / 16) = 2,
    so partial B edges need the all-thread algorithm. Sizes 131 and 1031
    expose this in the Windows numerical tests; both have remainder 3.
    """
    parameters = {
        "MatrixInstruction": [[16, 16, 16, 1, 1] + ([6, 1, 1, 4] if transpose_a else [1, 1, 2, 2])],
        "WavefrontSize": [32],
        "SourceSwap": [True],
        "VectorWidthA": [1],
        "VectorWidthB": [1],
        "GlobalReadVectorWidthA": [4],
        "GlobalReadVectorWidthB": [4],
        "DepthU": [32],
        "PrefetchGlobalRead": [0],
        "PrefetchLocalRead": [1],
        "ScheduleIterAlg": [3],
        "GlobalSplitU": [1],
        "AssertFree0ElementMultiple": [1],
        "AssertFree1ElementMultiple": [1],
    }
    config = {
        "BenchmarkProblems": [[{
            "OperationType": "GEMM",
            "DataType": 8,
            "DestDataType": dest_type,
            "ComputeDataType": 6,
            "HighPrecisionAccumulate": True,
            "TransposeA": transpose_a,
            "TransposeB": True,
            "Batched": True,
        }, {
            "ForkParameters": [{key: values} for key, values in parameters.items()],
        }]],
    }
    config_path = tmp_path / "int8_edge_shift.yaml"
    config_path.write_text(yaml.safe_dump(config))
    states = derive_states(config_path, arch="gfx1101")
    assert len(states) == 1
    assert states[0]["GlobalReadVectorWidthB"] == 4
    assert not states[0]["GuaranteeNoPartialB"]

    results = emit_kernels_from_config(config_path, arch="gfx1101", expected_fork_count=1)
    assert len(results) == 1
    _base, source, error = results[0]
    assert error == 0
    for remainder in (1, 2, 3):
        block = re.search(
            rf"^label_ShiftVectorComponents1_shift{remainder}_glvwblk0[^:\n]*:\n"
            r"(.*?)(?=^label_)",
            source,
            re.MULTILINE | re.DOTALL,
        )
        assert block is not None, f"missing B edge shift for remainder {remainder}"
        assert "v_mov_b32" in block[1], "edge shift must move accumulator values"
        if remainder != 2:
            assert "ds_bpermute_b32" in block[1], "odd shifts must exchange lanes"
