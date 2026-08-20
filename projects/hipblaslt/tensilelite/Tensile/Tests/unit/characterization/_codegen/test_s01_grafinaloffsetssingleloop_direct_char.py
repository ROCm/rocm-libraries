# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
"""S01 - KernelWriterAssembly graFinalOffsetsSingleLoop scalar-GRO characterization.

Drives the designed DirectToLds + UseInstOffsetForGRO config
(``data/test_data/_designed/gfx942/s01_grafinaloffsetssingleloop_direct.yaml``)
through the config-driven emit harness. Targets the scalar arm of
``graFinalOffsetsSingleLoop`` (``computeScalarGroImpl``) in
``Tensile/KernelWriterAssembly.py``:

  - 4444 : ``if DirectToLds%tc and UseInstOffsetForGRO`` guard, and
  - 4457 : ``ldsInc = (ldsInc*graIdx) % buff_load_inst_offset_max`` pad math.

``DirectToLdsA=1`` auto-derives ``UseInstOffsetForGRO=1`` (Solution derivation);
``UseSgprForGRO=1`` forces the SCALAR arm, so both target lines fire during the
emit call (``assignDerivedParameters`` + emission run in ``emit``).

CPU-only; no GPU, no compile, no hardware. pytestmark = pytest.mark.unit.
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
    "s01_grafinaloffsetssingleloop_direct.yaml",
)


def test_s01_grafinaloffsetssingleloop_direct_emits():
    """DTL + scalar-GRO config emits kernels and all have err==0."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, f"expected >=1 kernel, got {len(results)}"
    for base, src, err in results:
        assert err == 0, f"kernel {base!r} emitted with err={err}"
        assert base.startswith("Cijk_")
        assert ".amdgcn_target" in src, f"kernel {base!r}: missing .amdgcn_target"
        assert "gfx942" in src, f"kernel {base!r}: wrong arch in assembly"


def test_s01_grafinaloffsetssingleloop_direct_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest of the emit."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
