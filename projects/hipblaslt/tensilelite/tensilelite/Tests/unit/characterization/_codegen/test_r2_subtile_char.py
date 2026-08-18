################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""R2 — gfx950 Subtile/SubtileGREmit characterization.

Drives the P2 designed BenchmarkProblems config
(``data/_designed/gfx950/subtile.yaml``) through the config-driven emit harness
and asserts every emitted kernel is real gfx950 AMDGCN assembly with err==0.

Target: Tensile/Components/Subtile/SubtileGREmit.py, uncovered ranges:
  359-411  (_emitGR_TLU0 — subtile GR buffer load emit)
  528-639  (_grComputeSubtileOffsets_legacy / _grComputeRowPartition_legacy)
  658-834  (_grComputeAllOffsets_legacy / _grSwizzleColIds_legacy /
            _graTileAssignment_legacy)
  856-903  (emitSingleBufferLoad / globalReadDoSubtile)

The UseSubtileImpl=1 + DirectToLds=1 parameter combination enables the
SubtileBasedKernel path that routes through all of the above functions.

CPU-only. No GPU, no compile, no hardware access.
"""

import os

import pytest

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx950"

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx950",
    "subtile.yaml",
)


def test_r2_subtile_gfx950_emits_assembly():
    """UseSubtileImpl=1 subtile config emits real gfx950 assembly, all err==0."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, f"expected >=1 kernel, got 0 (config: {_CONFIG})"
    assert all(err == 0 for (_b, _s, err) in results), (
        f"some kernels failed: {[(b, e) for b, _s, e in results if e != 0]}"
    )
    for base, src, _err in results:
        assert src and len(src.splitlines()) > 100, f"kernel {base!r}: suspiciously short assembly"
        assert ".amdgcn_target" in src, f"kernel {base!r}: missing .amdgcn_target"
        assert "gfx950" in src, f"kernel {base!r}: wrong arch in assembly"
        assert base.startswith("Cijk_"), f"kernel {base!r}: unexpected basename prefix"


def test_r2_subtile_gfx950_golden(snapshot):
    """Order-invariant golden: pin {basename, err} for every emitted subtile kernel."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
