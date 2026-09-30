################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""P2-WIDEN — gfx950 DTL (DirectToLds) family seed (emit smoke).

The dominant F8/StreamK seed does not exercise the DirectToLds emitter paths
the DTL attribution entry (marginal 74 lines) covers. This widens the seed set
with the shipped ``Tests/common/gemm/gfx950/dtl.yaml`` (S/S/S TN with a
DirectToLds ForkParameters sweep) and asserts it emits real gfx950 assembly
with err==0.
"""

import os

import pytest

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx950"
_SEED = "Tensile/Tests/common/gemm/gfx950/dtl.yaml"


def test_seed_gfx950_dtl_emits_assembly():
    """DirectToLds (DTL) config emits real gfx950 assembly, all err==0."""
    results = emit_kernels_from_config(_SEED, limit=8, arch=_ARCH)
    assert len(results) >= 1
    assert all(err == 0 for (_b, _s, err) in results)


def test_gfx950_dtl_golden(snapshot):
    """Order-invariant golden: pin {basename, err} for every emitted DTL kernel."""
    results = emit_kernels_from_config(_SEED, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
