# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
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
import re

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


# ── PGR=2: the NGLL arms ──────────────────────────────────────────────────
#
# Problem group 1 is the same tile at PrefetchGlobalRead 2. It matters because
# the NGLL only exists at PGR=2 -- the scheduler returns an empty one at PGR 0
# and 1 -- so group 0 above never reaches the NGLL emit path at all. That is why
# a wrong claim about which tiles reach it survived the whole unit suite despite
# this file already emitting a 320x256 PLSIN kernel: nothing emitted the group
# that would have shown it.

_PGR2 = 1

_PLAIN_ARM = re.compile(r"^label_NGLL\w*_PlainNGLL:")
_DONE_ARM = re.compile(r"^label_NGLL\w*_PostInitNGLL:")


def _ngll_arms(src):
    """``(init arm, plain arm)`` instruction lists for each dual NGLL in ``src``.

    The NGLL splits into an arm for the workgroup that owns the fused store and
    a lean one for every other workgroup. They are emitted in that order and
    closed by the rejoin label.
    """
    lines = src.splitlines()
    arms = []
    for i, line in enumerate(lines):
        if "_PostLoopInitInNGLL" not in line:
            continue
        plain = next(j for j in range(i, len(lines)) if _PLAIN_ARM.match(lines[j]))
        done = next(j for j in range(plain, len(lines)) if _DONE_ARM.match(lines[j]))
        arms.append((_instructions(lines[i + 1:plain]), _instructions(lines[plain + 1:done])))
    return arms


def _instructions(segment):
    body = [l.split("//")[0].strip() for l in segment]
    return [l for l in body if l and not l.startswith("/*") and not l.endswith(":")]


def _valu(instructions):
    return [i for i in instructions if i.startswith("v_") and not i.startswith("v_mfma")]


def test_mt320x256_pgr2_solutions_are_in_scope():
    sols = solutions_from_config(_CONFIG, arch=_ARCH, limit_solutions=2, problem_index=_PGR2)
    assert sols, f"no solutions from {_CONFIG} group {_PGR2}"
    for sol in sols:
        st = sol._state if hasattr(sol, "_state") else sol
        assert st["PrefetchGlobalRead"] == 2, "the NGLL is only built at PGR=2"
        assert st["PostLoopStoreInNll"] is True
        assert st["MacroTile0"] == 320 and st["MacroTile1"] == 256


def test_ngll_init_arm_is_unwoven_above_256x256():
    """Above 256x256 the owner's NGLL arm is the plain one, with nothing hoisted.

    At 256x256 and below, the fused store's coordinate math is lifted into the
    owner's NGLL arm and interleaved into the MFMA gaps, so that arm comes out
    with more VALU than the lean arm beside it. A tile this size cannot afford
    the extra live coordinate registers -- it already sits at the arch-VGPR
    occupancy ceiling inside the loop -- so it takes the unwoven arm instead,
    and the two arms carry the same work.

    Asserted on instruction counts rather than on the emitted comments: this is
    the branch a review called unreachable and a source comment called a
    defensive guard, and both were wrong. ``test_ngll_init_arm_hoists_the_store_coords``
    in ``test_mxf4_bf16_subtile_char.py`` is the 256x256 half of the pair.
    """
    results = emit_kernels_from_config(_CONFIG, limit=2, arch=_ARCH, canonical=True,
                                       problem_index=_PGR2)
    assert results, f"no kernels from {_CONFIG} group {_PGR2}"
    assert all(err == 0 for (_b, _s, err) in results), (
        f"emit failed: {[(b, e) for b, _s, e in results if e != 0]}"
    )
    for base, src, _err in results:
        arms = _ngll_arms(src)
        assert arms, (
            f"{base}: PGR=2 emitted no dual NGLL, so the large-tile arm is untested again"
        )
        for init, plain in arms:
            assert len(_valu(init)) == len(_valu(plain)), (
                f"{base}: the owner's NGLL arm carries {len(_valu(init))} VALU against the "
                f"plain arm's {len(_valu(plain))}; above 256x256 nothing should be woven in"
            )
            mfma = [i for i in init if i.startswith("v_mfma")]
            assert len(mfma) == len([i for i in plain if i.startswith("v_mfma")]), (
                f"{base}: the two NGLL arms disagree about how much K they reduce"
            )
            assert mfma, f"{base}: the NGLL arm has no MFMAs"
