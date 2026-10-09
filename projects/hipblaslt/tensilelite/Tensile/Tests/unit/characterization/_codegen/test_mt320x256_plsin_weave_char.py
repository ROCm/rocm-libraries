################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""MT320x256 last-K PLSIN Weave: fused arm uses permlane16, plain keeps ds_bpermute.

MIWT [10,8] has a tile product of 80, past the 64 ``spillFree`` admits, so this
is the one emit in the unit suite that resolves PLSIN True through
``largeTileLend``. The generated kernel carries both store arms -- the fused
one for full tiles with K a multiple of DepthU, the plain one for the K tail
and the M/N edges -- and the point of this test is that each uses its own
shuffle.

CPU-only emit of the gfx950 MXFP4 320x256 Weave kernel. No GPU.
"""

import os

import pytest

from config_harness import emit_kernels_from_config, solutions_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx950"

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx950",
    "mt320x256_plsin_weave.yaml",
)

# Emitted where the fused store hands back to the ordinary path. Everything
# before it is the fused arm, everything after is the plain NLL and the
# post-loop tail.
_FUSED_DONE = "PostLoopStoreInNll: FUSED done"


def _split_fused_vs_rest(base: str, src: str):
    """Split the kernel at the fused-store boundary.

    The marker not being there means the fused store was never generated, which
    is the failure this test exists to catch. It has to raise rather than fall
    back to ``(src, "")``: that fallback hands the whole kernel back as the
    "fused" half, where the ambient MX-scale ``v_permlane16_swap`` satisfies the
    shuffle assertion and the test passes having checked nothing.
    """
    idx = src.find(_FUSED_DONE)
    assert idx >= 0, (
        f"{base}: no {_FUSED_DONE!r} marker -- the fused PLSIN Weave store was "
        f"not generated for this kernel"
    )
    return src[:idx], src[idx:]


def test_mt320x256_plsin_weave_solutions_are_in_scope():
    """The config resolves to the tile and the gate state the emit test assumes."""
    sols = solutions_from_config(_CONFIG, arch=_ARCH, limit_solutions=2)
    assert sols, f"no solutions from {_CONFIG}"
    for sol in sols:
        st = sol._state if hasattr(sol, "_state") else sol
        assert st["PostLoopStoreInNll"] is True, st.get("PostLoopStoreInNll")
        assert st["MacroTile0"] == 320 and st["MacroTile1"] == 256
        # Past spillFree's limit, so PLSIN here depends on largeTileLend.
        assert st["MIWaveTile"][0] * st["MIWaveTile"][1] > 64


def test_mt320x256_plsin_weave_emits_dual_shuffle():
    results = emit_kernels_from_config(_CONFIG, limit=2, arch=_ARCH, canonical=True)
    assert results, f"no kernels from {_CONFIG}"
    assert all(err == 0 for (_b, _s, err) in results), (
        f"emit failed: {[(b, e) for b, _s, e in results if e != 0]}"
    )
    for base, src, _err in results:
        fused, rest = _split_fused_vs_rest(base, src)
        assert "v_permlane16_swap_b32" in fused, (
            f"{base}: fused Weave store should use v_permlane16_swap"
        )
        assert rest, f"{base}: missing plain-NLL / post-loop tail"
        assert "ds_bpermute_b32" in rest, (
            f"{base}: plain NLL/post-loop should keep ds_bpermute"
        )
