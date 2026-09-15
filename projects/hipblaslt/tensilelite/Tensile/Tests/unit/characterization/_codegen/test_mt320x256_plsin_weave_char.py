################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""MT320x256 last-K PLSIN Weave: fused arm uses permlane16, plain keeps ds_bpermute.

CPU-only emit of the gfx950 MXFP4 320x256 Weave kernel. No GPU.
"""

import os

import pytest

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx950"
_CONFIG = os.path.normpath(os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "common", "gemm", "gfx950",
    "subtile_mxfp4_mt320x256_plsin_weave.yaml",
))


def _split_fused_vs_rest(src: str):
    """Fused store sits before the 'FUSED done, skip PLAIN NLL' branch."""
    marker = "PostLoopStoreInNll: FUSED done"
    idx = src.find(marker)
    if idx < 0:
        return src, ""
    return src[:idx], src[idx:]


def test_mt320x256_plsin_weave_emits_dual_shuffle():
    from config_harness import solutions_from_config

    sols = solutions_from_config(_CONFIG, arch=_ARCH, limit_solutions=2)
    assert sols, f"no solutions from {_CONFIG}"
    for sol in sols:
        st = sol._state if hasattr(sol, "_state") else sol
        assert st["PostLoopStoreInNll"] is True, st.get("PostLoopStoreInNll")
        assert st["PLSINStoreMode"] == "Weave"
        assert st["MacroTile0"] == 320 and st["MacroTile1"] == 256

    results = emit_kernels_from_config(_CONFIG, limit=2, arch=_ARCH, canonical=False)
    assert results, f"no kernels from {_CONFIG}"
    assert all(err == 0 for (_b, _s, err) in results), (
        f"emit failed: {[(b, e) for b, _s, e in results if e != 0]}"
    )
    for base, src, _err in results:
        fused, rest = _split_fused_vs_rest(src)
        assert fused, f"{base}: missing fused NLL (PostLoopStoreInNll Weave)"
        assert "v_permlane16_swap_b32" in fused, (
            f"{base}: fused Weave store should use v_permlane16_swap"
        )
        assert rest, f"{base}: missing plain-NLL / post-loop tail"
        assert "ds_bpermute_b32" in rest, (
            f"{base}: plain NLL/post-loop should keep ds_bpermute"
        )
