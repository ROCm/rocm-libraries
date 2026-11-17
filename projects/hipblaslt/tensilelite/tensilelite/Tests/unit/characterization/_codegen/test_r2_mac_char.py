################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""Non-MFMA FP16 HPA dot2 kernel generation for gfx942 and gfx1030."""

import os

import pytest

from config_harness import assert_assembles, emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx942"

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx90a",
    "mac.yaml",
)


@pytest.mark.parametrize("arch", ["gfx942", "gfx1030"])
def test_r2_mac_hhs_dot2_emits_assembly(arch):
    """Non-MFMA HHS DOT2 config emits and assembles for each target."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=arch, canonical=False)
    assert len(results) >= 1, f"expected >=1 kernel, got {len(results)}"
    assert all(err == 0 for (_b, _s, err) in results), (
        f"expected all err==0, got {[(b, e) for b, _s, e in results if e != 0]}"
    )
    for base, src, _err in results:
        assert src and len(src.splitlines()) > 500, (
            f"suspiciously short source for {base}"
        )
        assert ".amdgcn_target" in src, f"missing .amdgcn_target in {base}"
        assert arch in src, f"missing {arch} target marker in {base}"
        assert base.startswith("Cijk_"), f"unexpected basename pattern: {base}"
        # Confirm DOT2 MAC instructions are present in the emitted assembly.
        assert "v_dot2_f32_f16" in src, (
            f"expected v_dot2_f32_f16 MAC instructions in {base}"
        )
        assert_assembles(src, base)


def test_r2_mac_hhs_dot2_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest of the mac config emit."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
