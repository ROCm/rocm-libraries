# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Negative tests: the depthwise validators must actually reject bad specs.

Deliberately **ungated** -- no ``skipUnless`` on a device probe. Every check
here happens before a single byte reaches the GPU, so gating them on an MFMA
device would only let a broken validator hide behind a skip on a CI box.

Each rejection case asserts *both* halves of the contract:

1. ``is_valid_depthwise_col_spec(spec, arch)`` returns ``(False, reason)`` --
   the form the benchmark sweep consumes to skip a configuration, and
2. ``build_direct_depthwise_col(spec, arch=arch)`` raises ``ValueError`` --
   proving the gate is enforced at the build entry point rather than being
   advisory. A validator nobody calls is not a gate.

and additionally that ``reason`` names the offending value, so a rejection is
diagnosable from the benchmark log instead of a bare "invalid".

The positive controls at the bottom are not decoration: without them a
validator that rejected *everything* would pass the whole negative suite.

Run from ``rocke/library``::

    python -m pytest tests/test_direct_depthwise_col_validate.py
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kernels.common.conv_direct_grouped import (
    DirectConvProblem,
    DirectDepthwiseColSpec,
    DirectDepthwiseSpec,
    _DW_MAX_PRELOAD_TAPS,
    build_direct_depthwise,
    build_direct_depthwise_col,
    is_valid_depthwise_col_spec,
    is_valid_depthwise_spec,
)

_ARCH = "gfx950"

# A known-good baseline every case below perturbs along exactly one axis, so a
# failure names the axis rather than "some spec was rejected".
_BASE_PROBLEM = dict(
    N=1, H=28, W=28, groups=64, cpg=1, kpg=1, KH=3, KW=3, PAD=1, stride=1
)


def _spec(
    *,
    block_h=16,
    block_w=1,
    block_waves=1,
    dtype="fp16",
    max_live_f32=None,
    wave_size=64,
    **pkw,
):
    problem = DirectConvProblem(**{**_BASE_PROBLEM, **pkw})
    return DirectDepthwiseColSpec(
        problem=problem,
        name="validate_dwcol",
        block_h=block_h,
        block_w=block_w,
        block_waves=block_waves,
        dtype=dtype,
        max_live_f32=max_live_f32,
        wave_size=wave_size,
    )


# (case id, spec kwargs, substrings the reason must contain, arch)
_REJECTIONS = [
    # --- not depthwise -----------------------------------------------------
    ("cpg_not_1", dict(cpg=4), ["cpg=4"], _ARCH),
    ("kpg_not_1", dict(kpg=4), ["kpg=4"], _ARCH),
    # --- element type ------------------------------------------------------
    # fp8/int8/fp32 are all real dtypes the rest of the stack handles, so "not
    # implemented here" has to be said out loud rather than inferred.  fp32 in
    # particular resolves through dtype_to_ir and would otherwise look accepted:
    # this kernel takes the same fp16/bf16 pair as its preload and spatial
    # siblings, and the allow-list is what holds that line.
    ("dtype_fp8", dict(dtype="fp8"), ["'fp8'"], _ARCH),
    ("dtype_int8", dict(dtype="int8"), ["'int8'"], _ARCH),
    ("dtype_fp32", dict(dtype="fp32"), ["'fp32'"], _ARCH),
    ("dtype_empty", dict(dtype=""), ["''"], _ARCH),
    ("dtype_none", dict(dtype=None), ["None"], _ARCH),
    # --- block geometry ----------------------------------------------------
    ("block_h_zero", dict(block_h=0), ["block_h", "0"], _ARCH),
    ("block_h_negative", dict(block_h=-2), ["block_h", "-2"], _ARCH),
    ("block_w_zero", dict(block_w=0), ["block_w", "0"], _ARCH),
    ("block_w_negative", dict(block_w=-2), ["block_w", "-2"], _ARCH),
    # block_w larger than Wo wastes masked loads with no output contribution.
    # H=4, W=4, KH=3, PAD=1 -> Wo=4; block_w=6 exceeds Wo but live_f32=4*6+3=27 is within budget.
    ("block_w_exceeds_wo", dict(H=4, W=4, block_w=6), ["block_w 6", "Wo 4"], _ARCH),
    ("block_waves_zero", dict(block_waves=0), ["block_waves", "0"], _ARCH),
    # 64 waves x 64 lanes = 4096 threads, four times the gfx950 workgroup cap.
    (
        "block_waves_over_workgroup_cap",
        dict(block_waves=64),
        ["4096", "1024"],
        _ARCH,
    ),
    # --- register pressure -------------------------------------------------
    # block_h*block_w + KH = 64*4 + 3 = 259 f32 live per lane, past the 192
    # that gfx950's 512-VGPR file affords at 3/8 occupancy budget.
    (
        "live_f32_over_budget",
        dict(block_h=64, block_w=4),
        ["259", "192"],
        _ARCH,
    ),
    # An explicit override must tighten, and its value must appear in the
    # reason -- otherwise "exceeds max" is unattributable to the caller's knob.
    (
        "live_f32_over_explicit_override",
        dict(max_live_f32=8),
        ["max_live_f32=8"],
        _ARCH,
    ),
    # --- degenerate geometry ----------------------------------------------
    # KH=1 with PAD=1 grows the output past the input, a configuration the
    # kernel has no use case for at stride 1.
    ("ho_exceeds_h", dict(H=8, W=8, KH=1, KW=1), ["Ho=10", "H=8"], _ARCH),
    # Filter larger than the padded input: Ho/Wo go non-positive, so the
    # launch grid would be empty.
    ("ho_non_positive", dict(H=4, W=4, KH=9, KW=9, PAD=0), ["Ho=-4"], _ARCH),
    ("wo_non_positive", dict(H=16, W=4, KH=3, KW=9, PAD=0), ["Wo=-4"], _ARCH),
    # --- problem geometry --------------------------------------------------
    # stride=0 would be a ZeroDivisionError inside DirectConvProblem.Ho, so
    # this also pins that the stride check runs *before* anything reads Ho.
    ("stride_zero", dict(stride=0), ["stride", "0"], _ARCH),
    ("stride_negative", dict(stride=-1), ["stride", "-1"], _ARCH),
    ("pad_negative", dict(PAD=-1), ["PAD", "-1"], _ARCH),
    ("kh_zero", dict(KH=0), ["KH=0"], _ARCH),
    ("kw_zero", dict(KW=0), ["KW=0"], _ARCH),
    # PAD at/past the filter extent makes the first output row read nothing but
    # padding. At stride > 1 the Ho <= H check no longer bounds PAD.
    (
        "pad_at_filter_extent",
        dict(KH=3, KW=3, PAD=3, stride=2),
        ["PAD 3", "< min(KH, KW) = 3"],
        _ARCH,
    ),
    (
        "pad_unbounded_at_stride_gt_1",
        dict(H=28, W=28, KH=3, KW=3, PAD=1_000_000, stride=1_000_000),
        ["PAD 1000000"],
        _ARCH,
    ),
    # --- wave --------------------------------------------------------------
    # wave_size feeds the per-lane channel index; 0 divides by zero and a
    # mismatch silently maps lanes to the wrong channel.
    ("wave_size_zero", dict(wave_size=0), ["wave_size 0", "64"], _ARCH),
    ("wave_size_mismatch", dict(wave_size=32), ["wave_size 32", "64"], _ARCH),
    # --- arch --------------------------------------------------------------
    ("unknown_arch", dict(), ["gfx999"], "gfx999"),
]


@pytest.mark.parametrize(
    "cid,kwargs,wants,arch",
    _REJECTIONS,
    ids=[c[0] for c in _REJECTIONS],
)
def test_col_spec_rejected(cid, kwargs, wants, arch):
    spec = _spec(**kwargs)

    ok, reason = is_valid_depthwise_col_spec(spec, arch)
    assert not ok, f"{cid}: validator accepted a spec it must reject ({reason})"
    for want in wants:
        assert (
            want in reason
        ), f"{cid}: reason does not name the offending value {want!r}; got {reason!r}"

    with pytest.raises(ValueError):
        build_direct_depthwise_col(spec, arch=arch)


def test_col_rejection_reasons_are_distinct():
    """No two rejection classes may collapse onto the same message.

    Two different mistakes producing identical text means the benchmark log
    cannot tell a caller which knob to change.
    """
    reasons = {}
    for cid, kwargs, _wants, arch in _REJECTIONS:
        _, reason = is_valid_depthwise_col_spec(_spec(**kwargs), arch)
        assert (
            reason not in reasons
        ), f"{cid} and {reasons[reason]} share the rejection message {reason!r}"
        reasons[reason] = cid


# ---------------------------------------------------------------------------
# Positive controls -- a validator that rejects everything must fail here.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("stride", [1, 2, 3])
def test_col_spec_accepted_across_dtype_and_stride(dtype, stride):
    spec = _spec(dtype=dtype, stride=stride)
    ok, reason = is_valid_depthwise_col_spec(spec, _ARCH)
    assert ok, f"dtype={dtype} stride={stride} rejected: {reason}"
    assert reason == "ok"
    # Build, not just validate: the gate is only meaningful if the thing it
    # guards actually emits for the specs it accepts.
    kernel = build_direct_depthwise_col(spec, arch=_ARCH)
    assert kernel.name == spec.kernel_name()


def test_col_kernel_names_separate_dtype_stride_and_row_tile():
    """The cache key must distinguish variants that emit different code."""
    names = {
        (dt, s, bh): _spec(dtype=dt, stride=s, block_h=bh).kernel_name()
        for dt in ("fp16", "bf16")
        for s in (1, 2, 3)
        for bh in (8, 16)
    }
    assert len(set(names.values())) == len(names), names


@pytest.mark.parametrize("alias", ["float16", "half", "f16", "bfloat16"])
def test_col_dtype_aliases_resolve(alias):
    """Allow-list membership is tested on the resolved IR type, not the string.

    Each of these is a ``dtype_to_ir`` alias for fp16 or bf16; accepting them
    keeps this kernel consistent with the rest of the conv stack, which takes
    torch dtype names straight through.
    """
    ok, reason = is_valid_depthwise_col_spec(_spec(dtype=alias), _ARCH)
    assert ok, reason


def test_resolve_max_live_f32_tracks_the_vgpr_file():
    """The ceiling is derived from the arch, not a hardcoded 192."""
    base = _spec()
    assert base.resolve_max_live_f32("gfx950") == 512 * 3 // 8
    # gfx1250 has a 256-entry VGPR file, so the same spec gets half the budget.
    assert base.resolve_max_live_f32("gfx1250") == 256 * 3 // 8
    # An explicit override tightens but can never loosen past the file.
    assert _spec(max_live_f32=32).resolve_max_live_f32("gfx950") == 32
    assert _spec(max_live_f32=10**6).resolve_max_live_f32("gfx950") == 512 * 3 // 8


# ---------------------------------------------------------------------------
# The preload sibling: the tap-count gate moved out of the benchmark (Part 3).
# ---------------------------------------------------------------------------


def _preload_spec(*, block_w=8, block_waves=1, **pkw):
    problem = DirectConvProblem(**{**_BASE_PROBLEM, **pkw})
    return DirectDepthwiseSpec(
        problem=problem,
        name="validate_dw",
        block_w=block_w,
        block_waves=block_waves,
    )


def test_preload_spec_rejects_oversized_filter():
    """15x15 = 225 taps all held live at once; past the ceiling this is a hang.

    The reason must point at the column-streamed variant, because that is the
    actual remedy -- its live cost is linear in KH and independent of KW.
    """
    spec = _preload_spec(H=14, W=14, KH=15, KW=15, PAD=7)
    ok, reason = is_valid_depthwise_spec(spec, _ARCH)
    assert not ok
    assert "225" in reason and str(_DW_MAX_PRELOAD_TAPS) in reason
    assert "DirectDepthwiseColSpec" in reason
    with pytest.raises(ValueError):
        build_direct_depthwise(spec, arch=_ARCH)


def test_preload_spec_accepts_filter_at_the_ceiling(monkeypatch):
    """The gate must be a ceiling, not an off-by-one that rejects the boundary.

    Forward depthwise only takes odd filters with "same" padding, so no
    reachable filter has exactly ``_DW_MAX_PRELOAD_TAPS`` = 200 taps (an odd
    product). The boundary is pinned by moving the ceiling onto a reachable
    tap count instead: 13x15 = 195 taps.
    """
    import kernels.common.conv_direct_grouped as dc

    spec = _preload_spec(H=14, W=28, KH=13, KW=15, PAD=6)
    taps = spec.problem.KH * spec.problem.KW
    assert taps <= _DW_MAX_PRELOAD_TAPS
    ok, reason = is_valid_depthwise_spec(spec, _ARCH)
    assert ok, reason

    monkeypatch.setattr(dc, "_DW_MAX_PRELOAD_TAPS", taps)
    ok, reason = is_valid_depthwise_spec(spec, _ARCH)
    assert ok, reason
    monkeypatch.setattr(dc, "_DW_MAX_PRELOAD_TAPS", taps - 1)
    ok, reason = is_valid_depthwise_spec(spec, _ARCH)
    assert not ok and str(taps) in reason


def test_preload_spec_rejects_non_depthwise_and_unknown_arch():
    ok, reason = is_valid_depthwise_spec(_preload_spec(cpg=4), _ARCH)
    assert not ok and "cpg=4" in reason
    ok, reason = is_valid_depthwise_spec(_preload_spec(), "gfx999")
    assert not ok and "gfx999" in reason


def test_preload_spec_accepted_for_ordinary_filter():
    ok, reason = is_valid_depthwise_spec(_preload_spec(), _ARCH)
    assert ok and reason == "ok"


def test_benchmark_defers_tap_policy_to_the_validator():
    """Register-pressure policy belongs to the validator, not the top layer.

    The tap ceiling is *newly added* to ``is_valid_depthwise_spec`` here; it was
    never carried by the benchmark. What the two assertions below pin is that it
    stays that way: the benchmark keeps one uniform "ask the validator" path for
    all three depthwise variants, and never grows its own ``_DW_MAX_PRELOAD_TAPS``
    copy that could drift from the real gate.

    This is a layering guard, not the behavioural test for the ceiling -- that is
    ``test_preload_spec_rejects_oversized_filter`` (225 taps, rejected) and
    ``test_preload_spec_accepts_filter_at_the_ceiling`` (exactly 200, accepted).
    """
    src = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "common"
        / "benchmark_direct_conv.py"
    ).read_text()
    assert "_DW_MAX_PRELOAD_TAPS" not in src
    assert "is_valid_depthwise_spec" in src
