################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""S02b - KernelWriterAssembly globalReadGuardK Body tail-loop characterization.

Drives the designed non-buffer flat-addressing config
(``data/test_data/_designed/gfx950/s02b_globalreadguardk_body_tail_loop.yaml``)
through the config-driven emit harness. Targets the ``globalReadGuardK`` Body
guarded-read emission in ``Tensile/KernelWriterAssembly.py``:

  - 11029 : the non-BufferLoad else-arm VCmpXLtU64 addr<maxAddr masking, and
  - 11227 : the BufferLoad=0 checkIn of the maxAddr/bpe/zero VGPRs.

``BufferLoad=False`` forces the flat else-arm, and choosing K (=130) not a
multiple of DepthU (=16) makes the tail loop fire ``globalReadDo(mode=2)`` ->
``globalReadGuardK`` Body, arming both target lines during emission.

CPU-only; no GPU, no compile, no hardware. pytestmark = pytest.mark.unit.
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
    "s02b_globalreadguardk_body_tail_loop.yaml",
)


def test_s02b_globalreadguardk_body_tail_loop_emits():
    """Non-buffer tail-loop config emits kernels and all have err==0."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, f"expected >=1 kernel, got {len(results)}"
    for base, src, err in results:
        assert err == 0, f"kernel {base!r} emitted with err={err}"
        assert base.startswith("Cijk_")
        assert ".amdgcn_target" in src, f"kernel {base!r}: missing .amdgcn_target"
        assert "gfx950" in src, f"kernel {base!r}: wrong arch in assembly"


def test_s02b_globalreadguardk_body_tail_loop_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest of the tail-loop emit."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
