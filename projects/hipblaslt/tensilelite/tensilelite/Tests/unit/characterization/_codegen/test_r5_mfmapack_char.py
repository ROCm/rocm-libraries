################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""R5 — gfx942 MFMA pack scheduling for sparse kernels (KW 2155-2377).

Exercises the schedulePackConsiderMetadata=True arm of KernelWriter.scheduleMacro
(else branch at line 2153).  This block runs when:

    schedulePackConsiderMetadata = (
        kernel["ProblemType"]["Sparse"]
        and not kernel["DirectToVgprSparseMetadata"]
    )

With Sparse=1 and DirectToVgprSparseMetadata=0, the scheduler separates pack
items by kind (packItemsA / packItemsB / packItemsM) and interleaves them with
MFMA iterations via the instPerPack / packItems logic at lines 2155-2377:

  - Step 1 (2158-2259): insert required packs for A, M, B with instPerPack
    boundary tracking
  - Step 2 (2261-2300): fill remaining latency with desired packs
  - Step 3 (2302-2377): compute remainLatency, drain trailing packs or add
    s_nop as needed

Config design:
- gfx942 SMFMA [16,16,32,1] HH (half-half/fp32 accumulate)
- NT orientation (TransposeA=0, TransposeB=1): TLUA=True, TLUB=True
- TransposeLDS=0: UnrollMajorLDSA=0 AND UnrollMajorLDSB=0
  Both A and B generate pack items (bpeDS=2 < 4, no unroll-major LDS)
  → packItemsA non-empty, packItemsB non-empty
- ScheduleIterAlg=3: required for the schedulePackConsiderMetadata scheduler
- MIWaveTile [1,1] and [2,2]: different MFMA tile sizes exercise different
  pack scheduling patterns in Steps 1/2/3

This covers the B-side pack scheduling body (lines 2229-2241) and Step 3
instPackLast tracking (2315-2344) that were missed by NN-orientation configs.

No GPU required; all emit paths are CPU-only code generation.
"""

import os

import pytest

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx942"

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx942",
    "sparse_pack.yaml",
)


def test_r5_mfmapack_sparse_emits_assembly():
    """Sparse pack-scheduling seed config emits real gfx942 assembly, all err==0."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, f"Expected >=1 kernel, got {len(results)}"
    assert all(err == 0 for (_b, _s, err) in results), (
        "All kernels must emit with err==0; "
        + str([(b, e) for (b, _s, e) in results if e != 0])
    )
    for base, src, _err in results:
        assert src and len(src.splitlines()) > 100, f"Kernel {base!r} source too short"
        assert ".amdgcn_target" in src, f"Kernel {base!r} missing .amdgcn_target"
        assert "gfx942" in src, f"Kernel {base!r} missing gfx942 arch marker"
        assert base.startswith("Cijk_"), f"Unexpected basename: {base!r}"


def test_r5_mfmapack_sparse_pack_scheduling_comment():
    """Verify pack scheduling comment is emitted (confirms the target arm ran)."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1
    # The schedulePackConsiderMetadata=True path emits this comment at line 2306
    # in the target block: "pack scheduling: curPackIdx:..."
    found = any(
        "pack scheduling: curPackIdx" in (src or "")
        for (_b, src, _err) in results
    )
    assert found, (
        "Expected 'pack scheduling: curPackIdx' comment from the MFMA pack "
        "scheduler target block (KW 2306); none found in any emitted kernel. "
        "The schedulePackConsiderMetadata=True arm may not have been reached."
    )
