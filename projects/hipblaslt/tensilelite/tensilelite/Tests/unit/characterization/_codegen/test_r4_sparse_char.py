################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""R4 — gfx942 structured-sparsity (Sparse=1 + DirectToVgprSparseMetadata) codegen
characterization (no GPU required).

Drives the designed sparse_dtvsm.yaml config through the config-driven emit
harness and asserts every emitted kernel is real gfx942 AMDGCN assembly with
err==0.

Target missing ranges in Tensile/KernelWriterAssembly.py:
  - 3048-3105 : graMetadataTileAssignment, gated by
                  ``kernel["DirectToVgprSparseMetadata"]``
  - 4102-4192 : computeMetaDataSrd, gated by
                  ``kernel["ProblemType"]["Sparse"] and
                    kernel["DirectToVgprSparseMetadata"]``

Config uses:
  - Sparse: 1  (2:4 structured-sparse A matrix)
  - MatrixInstruction [16,16,32,1] (valid SMFMA HH shape for gfx942)
  - DirectToVgprSparseMetadata: 1  (routes metadata through VGPRs, not LDS)
  - MIWaveTile [2,2]  (exercises the inner loop in graMetadataTileAssignment)
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
    "sparse_dtvsm.yaml",
)


def test_r4_sparse_gfx942_emits_assembly():
    """Sparse DTVSM seed config emits real gfx942 assembly, all err==0."""
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
