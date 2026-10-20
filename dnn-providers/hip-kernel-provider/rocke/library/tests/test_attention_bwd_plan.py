# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-only tests of the general backward plan, predicate and declaration.

* every reason code is reachable, with accept rows for every declared axis;
* every backward table case maps to a request that yields a decision (and a
  plan when accepted) without exceptions; every decline carries a code;
* workspace formula: hand table, alignment, THD with and without the bound,
  plan layout equal to the formula;
* the declaration agrees with the predicate on a sampled grid;
* no device query and no device-module import;
* functional feature groups agree with the case-level feature table;
* routing: exactly one family per accepted request, dense only when eligible.
"""

from __future__ import annotations

import ast
import builtins
import collections
import dataclasses
import itertools
import json
import math
import random
from pathlib import Path

import pytest
from tests.sdpa.bwd_cases import backward_cases, select_bwd
from tests.sdpa.bwd_kernel_cases import (
    SUPPLEMENTARY_CASES,
    request_from_case,
    stage_of,
)

from kernels.common import attention_bwd_plan as plan_mod
from kernels.common.attention_bwd import AttnBwdSpec, attn_bwd_params
from kernels.common.attention_bwd_plan import (
    ATTN_BWD_DECLARATION_VERSION,
    DECLARED_ARCHS,
    DEFERRAL_REGISTER,
    FEATURE_GROUPS,
    REASON_CODES,
    ROUTES,
    AttnBwdPolicy,
    AttnBwdRequest,
    attn_bwd_declaration,
    attn_bwd_feature_groups,
    attn_bwd_perf_class,
    attn_bwd_plan,
    attn_bwd_route,
    attn_bwd_stage_vec,
    attn_bwd_support,
    attn_bwd_tuning_key,
    attn_bwd_workspace_bytes,
    attn_bwd_workspace_layout,
    dense_bwd_eligibility,
    shipped_policy,
)
from kernels.common.attention_bwd_plan import (
    ANY_BUCKET,
    DECLINE_KIND_NAMES,
    DECLINE_KINDS,
    FEATURE_DECLINES,
    MIRROR_PAIRING_BUILT,
    REQ_SCOPE_D256_BWD,
    SHIPPED_DECLARED_GROUPS,
)

_KERNEL_DIR = Path(__file__).resolve().parents[1] / "kernels" / "common"
DECLARED = ("gfx942", "gfx950", "gfx1151")

# Every functional group declared and unverified performance waived: the
# functional matrix as it will stand once the correctness gates pass.
FULL = AttnBwdPolicy(declared_groups=frozenset(FEATURE_GROUPS), waive_unverified=True)


def _req(**kw):
    base = dict(b=2, h_q=8, h_k=8, h_v=8, s_q=128, s_kv=128, d_qk=128, d_v=128)
    base.update(kw)
    return AttnBwdRequest(**base)


def _bhsd(b, h, s, d):
    return (h * s * d, s * d, d, 1)


# ---------------------------------------------------------------------------
# every reason code reachable, plus accept rows
# ---------------------------------------------------------------------------

_WAIVED_UNDECLARED = AttnBwdPolicy(
    declared_groups=frozenset(FEATURE_GROUPS) - {"narrow_access"},
    waive_unverified=True,
)

_TRIGGERS = {
    "ARCH_UNSUPPORTED": (_req(), "gfx90a", FULL),
    # no shipped target lacks a hardware run; a policy may still list one
    "ARCH_NOT_VALIDATED": (
        _req(),
        "gfx1201",
        dataclasses.replace(FULL, not_validated_archs=frozenset({"gfx1201"})),
    ),
    "DROPOUT": (_req(dropout=0.1), "gfx942", FULL),
    "BIAS_IN_BACKWARD": (_req(has_bias=True), "gfx942", FULL),
    "DBIAS": (_req(want_dbias=True), "gfx942", FULL),
    "ALIBI": (_req(alibi=True), "gfx942", FULL),
    "PAGED": (_req(paged=True), "gfx942", FULL),
    "FP8": (_req(fp8=True), "gfx942", FULL),
    "DETERMINISTIC": (_req(deterministic=True), "gfx942", FULL),
    "UNSUPPORTED_FEATURE": (_req(other_features=("sink_token",)), "gfx942", FULL),
    "SCALE_DEVICE_TENSOR": (_req(scale_is_device_tensor=True), "gfx942", FULL),
    "DTYPE": (_req(k_dtype="bf16"), "gfx942", FULL),
    "OUT_DTYPE": (_req(dq_dtype="fp32"), "gfx942", FULL),
    "LSE_FORMAT": (_req(lse_dtype="fp16"), "gfx942", FULL),
    "DQK_NE_DV": (_req(d_v=64), "gfx942", FULL),
    "HEAD_DIM_256_BWD": (_req(d_qk=256, d_v=256), "gfx942", FULL),
    "HEAD_DIM": (_req(d_qk=96, d_v=96), "gfx942", FULL),
    "GQA_RATIO": (_req(h_k=3, h_v=3), "gfx942", FULL),
    "BAND_BOUND": (_req(causal=True, causal_bottom_right=True), "gfx942", FULL),
    "STRIDE_FORMAT": (_req(strides={"q": (128 * 128, 128, 1)}), "gfx942", FULL),
    "STRIDE_D": (
        _req(strides={"q": (128 * 128 * 8 * 2, 128 * 128 * 2, 128 * 2, 2)}),
        "gfx942",
        FULL,
    ),
    "STRIDE_NEGATIVE": (
        _req(strides={"k": (-128 * 128 * 8, 128 * 128, 128, 1)}),
        "gfx942",
        FULL,
    ),
    "OUTPUT_OVERLAP": (_req(strides={"dq": (0, 128 * 128, 128, 1)}), "gfx942", FULL),
    "STRIDE_ALIGN": (_req(tensor_alignment=1), "gfx942", FULL),
    "INDEX_RANGE": (
        _req(b=4096, s_q=2**19, s_kv=2**19, d_qk=64, d_v=64, h_q=1, h_k=1, h_v=1),
        "gfx942",
        FULL,
    ),
    "GRID_LIMIT": (_req(b=70000, s_q=16, s_kv=16, h_q=1, h_k=1, h_v=1), "gfx942", FULL),
    "SEQ_LEN_FORMAT": (_req(has_seq_len_q=True, has_seq_len_kv=True), "gfx942", FULL),
    "RAGGED_FORMAT": (_req(layout="thd", ragged_offset_dtype="int16"), "gfx942", FULL),
    "EMPTY": (_req(b=0), "gfx942", FULL),
    "INSTANCE_UNAVAILABLE": (
        _req(),
        "gfx942",
        dataclasses.replace(
            FULL,
            instance_unavailable=frozenset(
                {AttnBwdSpec(head_size=128, mask_class="none").kernel_name("main")}
            ),
        ),
    ),
    "PERF_BELOW_BASELINE": (
        _req(),
        "gfx942",
        dataclasses.replace(
            FULL,
            perf_gate=True,
            perf_status={attn_bwd_perf_class(_req(), "gfx942"): "below_baseline"},
        ),
    ),
    # the performance codes need the opt-in performance gate
    "PERF_UNVERIFIED": (
        _req(),
        "gfx942",
        AttnBwdPolicy(declared_groups=frozenset(FEATURE_GROUPS), perf_gate=True),
    ),
    "NOT_YET_DECLARED": (
        # stage_vec = 1 under a policy that has not declared narrow_access
        _req(tensor_alignment=8),
        "gfx942",
        _WAIVED_UNDECLARED,
    ),
}


def test_reason_code_order_is_the_documented_one():
    assert REASON_CODES[:2] == ("ARCH_UNSUPPORTED", "ARCH_NOT_VALIDATED")
    assert REASON_CODES[-4:] == (
        "INSTANCE_UNAVAILABLE",
        "PERF_BELOW_BASELINE",
        "PERF_UNVERIFIED",
        "NOT_YET_DECLARED",
    )
    assert REASON_CODES.index("STRIDE_ALIGN") < REASON_CODES.index("INDEX_RANGE")
    assert REASON_CODES.index("RAGGED_FORMAT") < REASON_CODES.index("EMPTY")
    assert len(set(REASON_CODES)) == len(REASON_CODES) == 33
    assert (
        REASON_CODES.index("BAND_BOUND") + 1
        == REASON_CODES.index("STRIDE_FORMAT")
        == REASON_CODES.index("STRIDE_D") - 1
    )


@pytest.mark.parametrize("code", REASON_CODES)
def test_every_reason_code_reachable(code):
    req, arch, policy = _TRIGGERS[code]
    v = attn_bwd_support(req, arch, policy=policy)
    assert (v.ok, v.code) == (False, code), v
    assert v.detail
    with pytest.raises(ValueError, match=code):
        attn_bwd_plan(req, arch, policy=policy)


def _accept_rows():
    thd = dict(
        layout="thd",
        strides={
            n: (0, 128, 8 * 128, 1)
            for n in ("q", "k", "v", "o", "do", "dq", "dk", "dv")
        },
    )
    rows = {
        "thd_without_max_total": _req(**thd),
        "thd_with_max_total": _req(**thd, max_total_q=200, max_total_kv=200),
        "thd_int64_offsets": _req(**thd, ragged_offset_dtype="int64"),
        "thd_multiplier": _req(**thd, ragged_offset_multiplier=1024),
        "s_q_1": _req(s_q=1, s_kv=777, h_k=2, h_v=2),
        "s_q_1_mha": _req(s_q=1, s_kv=33),
        "hk_ne_hv": _req(h_k=4, h_v=2),
        "mqa": _req(h_k=1, h_v=1),
        "d32": _req(d_qk=32, d_v=32),
        "d64_bf16": _req(d_qk=64, d_v=64, dtype="bf16"),
        "causal_tl_bool": _req(causal=True),
        "causal_br_bool": _req(causal_bottom_right=True, s_q=33, s_kv=100),
        "window_br": _req(left_bound=15, right_bound=0, bottom_right=True),
        "padded": _req(padding=True, has_seq_len_q=True, has_seq_len_kv=True),
        "explicit_scale": _req(scale=0.3),
        "bshd": _req(
            strides={
                n: (128 * 8 * 128, 128, 8 * 128, 1) for n in ("q", "o", "do", "dq")
            }
        ),
        # index-range rows served by the i64 per-head workspace rebase
        "i64_rebase_h16": _req(b=1, h_q=16, h_k=16, h_v=16, s_q=2**21, s_kv=2**21),
        "i64_rebase_b64": _req(b=64, s_q=2**19, s_kv=2**19, h_q=1, h_k=1, h_v=1),
        "zero_b_stride_input": _req(
            strides={"k": (0, 128 * 128, 128, 1), "v": (0, 128 * 128, 128, 1)}
        ),
    }
    return rows


@pytest.mark.parametrize("arch", DECLARED)
@pytest.mark.parametrize("name", sorted(_accept_rows()))
def test_accept_rows(name, arch):
    req = _accept_rows()[name]
    v = attn_bwd_support(req, arch, policy=FULL)
    assert v.ok, (name, v)
    plan = attn_bwd_plan(req, arch, policy=FULL)
    assert plan.workspace_bytes == attn_bwd_workspace_bytes(
        req, g_split=plan.runtime["g_split"], dq_mode=plan.specs["main"].dq_mode
    )


_THD_ROLES = ("q", "k", "v", "o", "do", "dq", "dk", "dv")
# Stride rows: every B/H/S stride and every element-aligned base is
# served; strides that are not multiples of 8 elements or a base below 16 B
# select the narrow instances (stage_vec = 1).
_NARROW_ROWS = {
    "odd_s_stride": dict(strides={"q": (128 * 128 * 8, 128 * 128, 129, 1)}),
    "h_stride_not_8": dict(strides={"k": (128 * 128 * 8, 128 * 128 + 4, 128, 1)}),
    "odd_b_stride": dict(strides={"dv": (128 * 128 * 8 + 1, 128 * 128, 128, 1)}),
    "bshd_odd_h_and_s": dict(
        strides={
            n: (128 * (8 * 133 + 3), 133, 8 * 133 + 3, 1)
            for n in ("q", "o", "do", "dq")
        }
    ),
    "thd_odd_token_stride": dict(
        layout="thd",
        strides={n: (0, 128, 8 * 128 + 3, 1) for n in _THD_ROLES},
    ),
    "base_2": dict(tensor_alignment=2),
    "base_4": dict(tensor_alignment=4),
    "base_8": dict(tensor_alignment=8),
    "base_2_gqa_band": dict(
        tensor_alignment=2, h_k=2, h_v=2, causal_bottom_right=True, s_q=33
    ),
    "base_4_hk_ne_hv": dict(tensor_alignment=4, h_k=4, h_v=2),
    "base_8_padded": dict(
        tensor_alignment=8, padding=True, has_seq_len_q=True, has_seq_len_kv=True
    ),
}


@pytest.mark.parametrize("name", sorted(_NARROW_ROWS))
def test_narrow_access_rows_accept_with_stage_vec_1(name):
    req = _req(**_NARROW_ROWS[name])
    assert attn_bwd_stage_vec(req) == 1
    assert "narrow_access" in attn_bwd_feature_groups(req)
    for arch in DECLARED:
        assert attn_bwd_support(req, arch, policy=FULL).ok
        plan = attn_bwd_plan(req, arch, policy=FULL)
        main = plan.specs["main"]
        assert main.stage_vec == 1
        assert {plan.specs[s].stage_vec for s in ("prep", "convert")} == {1}
        assert all("_sv1" in step.kernel for step in plan.launches)
    # the shipped policy declares narrow access and serves every accepted
    # request whatever its performance status: never a stride decline
    for arch in ("gfx1151", "gfx942", "gfx950"):
        assert attn_bwd_support(req, arch).ok


def test_wide_access_needs_every_used_stride_a_multiple_of_8():
    # strides of extents of 1 are never used: an odd batch stride with b = 1
    # and an odd head stride with one head keep stage_vec = 8
    assert attn_bwd_stage_vec(_req()) == 8
    assert attn_bwd_stage_vec(_req(tensor_alignment=16)) == 8
    one = _req(b=1, strides={"q": (3, 128 * 128, 128, 1)})
    assert attn_bwd_stage_vec(one) == 8
    mqa = _req(h_k=1, h_v=1, strides={"k": (128 * 128, 7, 128, 1)})
    assert attn_bwd_stage_vec(mqa) == 8
    # only a base that is not element aligned is declined
    assert attn_bwd_support(_req(tensor_alignment=1), "gfx942", policy=FULL).code == (
        "STRIDE_ALIGN"
    )
    for align in (2, 4, 8, 16):
        assert attn_bwd_support(_req(tensor_alignment=align), "gfx942", policy=FULL).ok


def test_index_range_decline_rows():
    for req in (
        _req(b=4096, s_q=2**19, s_kv=2**19, d_qk=64, d_v=64, h_q=1, h_k=1, h_v=1),
        _req(b=1, s_q=2**24, s_kv=16, h_q=1, h_k=1, h_v=1),
    ):
        assert attn_bwd_support(req, "gfx942", policy=FULL).code == "INDEX_RANGE"


def test_grid_limit_rows():
    many_heads = _req(h_q=70000, h_k=70000, h_v=70000, s_q=16, s_kv=16, b=1)
    assert attn_bwd_support(many_heads, "gfx942", policy=FULL).code == "GRID_LIMIT"
    prep_y = _req(h_q=40000, h_k=20000, h_v=10000, s_q=16, s_kv=16, b=1)
    assert attn_bwd_support(prep_y, "gfx942", policy=FULL).code == "GRID_LIMIT"


def test_seq_len_and_ragged_format_rows():
    bad = [
        (_req(padding=True, has_seq_len_q=True), "SEQ_LEN_FORMAT"),
        (
            _req(
                padding=True,
                has_seq_len_q=True,
                has_seq_len_kv=True,
                seq_len_dtype="int64",
            ),
            "SEQ_LEN_FORMAT",
        ),
        (_req(layout="thd", ragged_same_table=False), "RAGGED_FORMAT"),
        (_req(layout="thd", ragged_offset_multiplier=0), "RAGGED_FORMAT"),
        (_req(layout="thd", ragged_offset_dims=(5,)), "RAGGED_FORMAT"),
    ]
    for req, code in bad:
        assert attn_bwd_support(req, "gfx950", policy=FULL).code == code


def test_format_decline_rows_per_rule():
    """One decline row per predicate rule that no other row reaches first."""
    thd = dict(layout="thd")
    rows = [
        # h_k divides h_q but h_v does not
        (_req(h_k=2, h_v=3), "GQA_RATIO"),
        (
            _req(
                padding=True,
                has_seq_len_q=True,
                has_seq_len_kv=True,
                seq_len_dims=(2, 1, 1, 2),
            ),
            "SEQ_LEN_FORMAT",
        ),
        (_req(lse_strides=(-1, 1, 1, 1)), "LSE_FORMAT"),
        (_req(lse_dims=(2, 8, 128, 1), lse_strides=(1024, 128, 1)), "LSE_FORMAT"),
        (_req(**thd, lse_dims=(256, 8, 2)), "LSE_FORMAT"),
        (_req(**thd, max_total_q=-5), "RAGGED_FORMAT"),
        (_req(**thd, max_total_kv=-1), "RAGGED_FORMAT"),
        # a zero token bound for a non-empty problem leaves no workspace row
        (_req(**thd, h_k=8, h_v=4, max_total_q=0), "RAGGED_FORMAT"),
        (_req(**thd, h_k=8, h_v=4, max_total_kv=0), "RAGGED_FORMAT"),
        (_req(**thd, max_total_kv=0), "RAGGED_FORMAT"),
    ]
    for req, code in rows:
        v = attn_bwd_support(req, "gfx942", policy=FULL)
        assert v.code == code, (req, v)
    # the same accepted requests without the broken field
    assert attn_bwd_support(_req(h_k=2, h_v=2), "gfx942", policy=FULL).ok
    assert attn_bwd_support(
        _req(**thd, max_total_q=1, max_total_kv=1), "gfx942", policy=FULL
    ).ok


def test_thd_lse_head_count_and_rank_rows():
    thd = dict(layout="thd")
    # [T_q, H, 1] whose H is not h_q (8)
    assert (
        attn_bwd_support(_req(**thd, lse_dims=(256, 4, 1)), "gfx942", policy=FULL).code
        == "LSE_FORMAT"
    )
    # dims and strides of different rank
    bad_rank = _req(**thd, lse_dims=(256, 8, 1), lse_strides=(8, 1, 1, 1))
    assert attn_bwd_support(bad_rank, "gfx942", policy=FULL).code == "LSE_FORMAT"
    for ok in (
        _req(**thd, lse_dims=(256, 8, 1)),
        _req(**thd, lse_dims=(256, 8, 1), lse_strides=(8, 1, 1)),
    ):
        assert attn_bwd_support(ok, "gfx942", policy=FULL).ok


@pytest.mark.parametrize(
    "stride, ok",
    [(-1, False), (2**31, False), (2**31 - 1, True), (0, True), (1, True)],
)
def test_seq_len_stride_range(stride, ok):
    req = _req(
        padding=True, has_seq_len_q=True, has_seq_len_kv=True, seq_len_stride=stride
    )
    v = attn_bwd_support(req, "gfx942", policy=FULL)
    assert v.ok if ok else v.code == "SEQ_LEN_FORMAT", v


_D, _S, _H = 128, 128, 8


@pytest.mark.parametrize(
    "name, strides, overlap",
    [
        # S stride one short of D: rows overlap by one element
        ("dq", (_H * _S * _D, _S * _D, _D - 1, 1), True),
        ("dq", (_H * _S * _D, _S * _D, _D, 1), False),
        # H stride one short of the S * D extent already covered
        ("dq", (_H * _S * _D, _S * _D - 1, _D, 1), True),
        ("dq", (_H * _S * _D, _S * _D, _D, 1), False),
        # B stride one short of H * S * D
        ("dk", (_H * _S * _D - 1, _S * _D, _D, 1), True),
        # interleaved packed [B, S, 2, H, D] views: dK and dV do not self-overlap
        ("dk", (_S * 2 * _H * _D, _D, 2 * _H * _D, 1), False),
        ("dv", (_S * 2 * _H * _D, _D, 2 * _H * _D, 1), False),
        # same packed view with an S stride one short of 2 * H * D still fits
        # (the H extent covers only H * D), one short of H * D does not
        ("dv", (_S * 2 * _H * _D, _D, _H * _D - 1, 1), True),
        ("dv", (_S * _H * _D, _D, _H * _D, 1), False),
    ],
)
def test_output_overlap_at_the_boundary(name, strides, overlap):
    req = _req(strides={name: strides})
    v = attn_bwd_support(req, "gfx942", policy=FULL)
    if overlap:
        assert v.code == "OUTPUT_OVERLAP", v
    else:
        assert v.ok, v


def test_thd_output_overlap_at_the_boundary():
    thd = dict(layout="thd", max_total_q=256, max_total_kv=256)
    # (ignored, H, token, D): token stride one short of H * D overlaps
    bad = _req(**thd, strides={"dq": (0, _D, _H * _D - 1, 1)})
    assert attn_bwd_support(bad, "gfx942", policy=FULL).code == "OUTPUT_OVERLAP"
    good = _req(**thd, strides={"dq": (0, _D, _H * _D, 1)})
    assert attn_bwd_support(good, "gfx942", policy=FULL).ok


def test_interleaved_disjoint_outputs_are_admitted():
    """Rows of different heads interleaved between the rows of one head (head
    stride 96, token stride 64, D = 32: rows at 0, 64, 128 and 96, 160, 224)
    share no element; the sorted test alone would flag them."""
    strides = {r: (512, 32, 128, 1) for r in ("q", "k", "v", "o", "do", "dk", "dv")}
    strides["dq"] = (512, 96, 64, 1)
    req = _req(
        b=1, h_q=2, h_k=2, h_v=2, s_q=3, s_kv=3, d_qk=32, d_v=32, strides=strides
    )
    assert plan_mod._overlaps_sorted((1, 2, 3, 32), strides["dq"])
    assert attn_bwd_support(req, "gfx942", policy=FULL).ok
    # one more token row lands on the next head's first row: overlap
    over = dict(strides, dq=(512, 96, 48, 1))
    req = _req(b=1, h_q=2, h_k=2, h_v=2, s_q=3, s_kv=3, d_qk=32, d_v=32, strides=over)
    assert attn_bwd_support(req, "gfx942", policy=FULL).code == "OUTPUT_OVERLAP"
    # THD: tokens of the two heads interleaved, (ignored, H, token, D)
    thd = dict(layout="thd", b=1, h_q=2, h_k=2, h_v=2, s_q=3, s_kv=3, d_qk=32, d_v=32)
    thd_strides = {r: (0, 32, 64, 1) for r in ("q", "k", "v", "o", "do", "dk", "dv")}
    good = _req(**thd, strides={**thd_strides, "dq": (0, 96, 64, 1)})
    assert attn_bwd_support(good, "gfx942", policy=FULL).ok


def _brute_overlap(dims, strides):
    seen = set()
    for idx in itertools.product(*(range(n) for n in dims)):
        off = sum(i * s for i, s in zip(idx, strides))
        if off in seen:
            return True
        seen.add(off)
    return False


def test_overlap_test_is_exact_on_random_small_layouts():
    """The overlap test agrees with an enumeration of every element offset
    on random unit-D layouts (the backward's output class)."""
    rng = random.Random(11)
    checked = interleaved = 0
    for _ in range(4000):
        dims = [
            rng.randint(1, 4),
            rng.randint(1, 4),
            rng.randint(1, 5),
            rng.randint(1, 9),
        ]
        strides = [rng.randint(0, 60), rng.randint(0, 60), rng.randint(0, 60), 1]
        want = _brute_overlap(dims, strides)
        assert plan_mod._overlaps(dims, strides) == want, (dims, strides)
        checked += 1
        interleaved += plan_mod._overlaps_sorted(dims, strides) and not want
    # the sample really exercises layouts the sorted test cannot prove
    assert checked == 4000 and interleaved > 50, interleaved


def test_empty_extents_decline_empty_not_overlap():
    for kw in (
        dict(h_q=0),
        dict(h_k=0),
        dict(h_v=0),
        dict(s_q=0),
        dict(s_kv=0),
        dict(b=0),
        dict(layout="thd", h_q=0),
        dict(layout="thd", h_k=0),
        dict(layout="thd", s_q=0),
        dict(layout="thd", b=0, max_total_q=0, max_total_kv=0),
        # an empty output never overlaps, whatever its (degenerate) strides
        dict(h_q=0, strides={"dq": (0, 0, 128, 1)}),
        dict(s_kv=0, strides={"dk": (0, 0, 0, 1), "dv": (0, 0, 0, 1)}),
    ):
        v = attn_bwd_support(_req(**kw), "gfx942", policy=FULL)
        assert v.code == "EMPTY", (kw, v)


def test_kv_workspace_rows_index_range_only_in_atomic_mode():
    # B * S_kv = 2^31 rows of dK / dV workspace; every per-tensor index is in range
    big = dict(b=2**15, s_q=1, s_kv=2**16, d_qk=32, d_v=32, h_q=2, h_k=2)
    v = attn_bwd_support(_req(**big, h_v=1), "gfx942", policy=FULL)
    assert v.code == "INDEX_RANGE" and "rows_kv" in v.detail, v
    assert attn_bwd_support(_req(**big, h_v=2), "gfx942", policy=FULL).ok


def test_index_and_grid_limits_at_the_exact_boundary():
    one = dict(b=1, h_q=1, h_k=1, h_v=1, s_q=1, s_kv=16, d_qk=128, d_v=128)

    def with_q_stride(ss):
        return _req(**one, strides={"q": (ss, ss, ss, 1)})

    assert attn_bwd_support(with_q_stride(2**31 - 1 - 128), "gfx942", policy=FULL).ok
    v = attn_bwd_support(with_q_stride(2**31 - 128), "gfx942", policy=FULL)
    assert v.code == "INDEX_RANGE" and "q:" in v.detail, v
    grid = dict(h_q=1, h_k=1, h_v=1, s_q=16, s_kv=16, d_qk=32, d_v=32)
    assert attn_bwd_support(_req(b=65535, **grid), "gfx942", policy=FULL).ok
    v = attn_bwd_support(_req(b=65536, **grid), "gfx942", policy=FULL)
    assert v.code == "GRID_LIMIT" and "B=65536" in v.detail, v


def test_thd_lse_strides_in_both_forms_reach_prep():
    thd = dict(
        layout="thd",
        strides={
            n: (0, 128, 8 * 128, 1)
            for n in ("q", "k", "v", "o", "do", "dq", "dk", "dv")
        },
    )

    def prep_lse(req):
        assert attn_bwd_support(req, "gfx942", policy=FULL).ok
        s = attn_bwd_plan(req, "gfx942", policy=FULL).launches[0].scalars
        return s["l_b"], s["l_h"], s["l_t"]

    # [T_q, H_q, 1] with (t, h, 1) strides
    assert prep_lse(_req(**thd, lse_dims=(256, 8, 1), lse_strides=(8, 1, 1))) == (
        0,
        1,
        8,
    )
    assert prep_lse(_req(**thd, lse_dims=(256, 8, 1), lse_strides=(1, 256, 1))) == (
        0,
        256,
        1,
    )
    # [B, H_q, S_q, 1] with (b, h, t, 1) strides; the B stride is unused
    assert prep_lse(
        _req(**thd, lse_dims=(2, 8, 128, 1), lse_strides=(1024, 1, 8, 1))
    ) == (0, 1, 8)
    assert prep_lse(
        _req(**thd, lse_dims=(2, 8, 128, 1), lse_strides=(1024, 128, 1, 1))
    ) == (0, 128, 1)
    # packed default: token-major [T_q, H_q]
    assert prep_lse(_req(**thd)) == (0, 1, 8)
    # dense keeps (b, h, t)
    assert prep_lse(_req(lse_strides=(1024, 128, 1, 1))) == (1024, 128, 1)


# ---------------------------------------------------------------------------
# LSE stride forms: every rank either honoured exactly or declined
# ---------------------------------------------------------------------------

# Distinct per-position values: a stride read from the wrong position, or a
# default used in place of a given tuple, changes the kernargs.
_LSE_ST = (4099, 257, 3, 9, 11, 13)
_THD8 = {n: (0, 128, 8 * 128, 1) for n in ("q", "k", "v", "o", "do", "dq", "dk", "dv")}
# (layout, dims form) -> (dims, {accepted stride rank: (l_b, l_h, l_t)}); the
# _req() shape is b=2, h_q=8, s_q=128. Hand-written, not derived from the
# implementation.
_LSE_FORMS = {
    ("dense", "default_dims"): ((), {0: (1024, 128, 1), 4: (4099, 257, 3)}),
    ("dense", "bhs1"): ((2, 8, 128, 1), {0: (1024, 128, 1), 4: (4099, 257, 3)}),
    ("thd", "default_dims"): ((), {0: (0, 1, 8), 3: (0, 257, 4099)}),
    ("thd", "th1"): ((256, 8, 1), {0: (0, 1, 8), 3: (0, 257, 4099)}),
    ("thd", "bhs1"): ((2, 8, 128, 1), {4: (0, 257, 3)}),
}


def _lse_req(layout, dims, strides):
    kw = dict(layout="thd", strides=_THD8) if layout == "thd" else {}
    return _req(lse_dims=dims, lse_strides=strides, **kw)


@pytest.mark.parametrize("rank", range(7))
@pytest.mark.parametrize("form", sorted(_LSE_FORMS), ids="-".join)
def test_lse_stride_tuple_of_every_rank(form, rank):
    """A stride tuple of every length 0..6 under every dims form: the accepted
    ranks reach the prep kernargs exactly (``()`` gives the packed strides of
    the dims); every other rank is declined ``LSE_FORMAT`` naming the rank,
    never replaced by the packed default."""
    dims, accepted = _LSE_FORMS[form]
    req = _lse_req(form[0], dims, _LSE_ST[:rank])
    v = attn_bwd_support(req, "gfx942", policy=FULL)
    if rank in accepted:
        assert v.ok, v
        s = attn_bwd_plan(req, "gfx942", policy=FULL).launches[0].scalars
        assert (s["l_b"], s["l_h"], s["l_t"]) == accepted[rank]
    else:
        assert v.code == "LSE_FORMAT", v
        assert str(_LSE_ST[:rank]) in v.detail, v
        with pytest.raises(ValueError, match="LSE_FORMAT"):
            plan_mod._lse_kernarg_strides(req)
        with pytest.raises(ValueError):
            attn_bwd_plan(req, "gfx942", policy=FULL)


@pytest.mark.parametrize(
    "layout, dims, strides, phrase",
    [
        # rank of the canonical dims: the short forms that used to be dropped
        ("dense", (), (4099,), "rank 1"),
        ("dense", (), (4099, 257), "rank 2"),
        ("dense", (), (4099, 257, 3), "rank 3"),
        ("dense", (2, 8, 128, 1), (4099, 257), "rank 2"),
        ("thd", (), (4099,), "rank 1"),
        ("thd", (), (4099, 257), "rank 2"),
        ("thd", (), (4099, 257, 3, 9), "rank 4"),
        ("thd", (256, 8, 1), (8, 1), "rank 2"),
        ("thd", (2, 8, 128, 1), (4099, 257, 3), "rank-4"),
        # the batched THD shape has no packed default
        ("thd", (2, 8, 128, 1), (), "batched shape"),
        # entries
        ("dense", (), (4099, -1, 3, 1), "negative"),
        ("thd", (), (-8, 1, 1), "negative"),
        ("dense", (), (4099.0, 257, 3, 1), "non-integer"),
        ("dense", (), (4099, True, 3, 1), "non-integer"),
        ("thd", (256.0, 8, 1), (8, 1, 1), "non-integer"),
        ("thd", (-1, 8, 1), (8, 1, 1), "negative token count"),
        # dims
        ("dense", (2, 8, 128), (), "dense LSE dims"),
        ("dense", (2, 4, 128, 1), (), "dense LSE dims"),
        ("dense", (2, 8, 64, 1), (), "dense LSE dims"),
        ("dense", (2, 8, 128, 2), (), "dense LSE dims"),
        ("thd", (256, 4, 1), (), "THD LSE dims"),
        ("thd", (256, 8, 2), (), "THD LSE dims"),
        ("thd", (256, 8), (), "THD LSE dims"),
        ("thd", (3, 8, 128, 1), (4099, 257, 3, 1), "THD LSE dims"),
    ],
)
def test_lse_malformed_forms_are_declined_with_a_reason(layout, dims, strides, phrase):
    v = attn_bwd_support(_lse_req(layout, dims, strides), "gfx942", policy=FULL)
    assert v.code == "LSE_FORMAT" and phrase in v.detail, v


def test_lse_dtype_decline_names_the_dtype():
    v = attn_bwd_support(_req(lse_dtype="fp16"), "gfx942", policy=FULL)
    assert v.code == "LSE_FORMAT" and "fp16" in v.detail, v


def test_lse_accepted_forms_honour_numpy_ints_and_unit_dim_strides():
    """numpy integer entries are integers; the stride of the trailing extent-1
    dim never moves an address, so any non-negative value is accepted."""
    np = pytest.importorskip("numpy")
    st = tuple(np.int64(x) for x in (4099, 257, 3, 0))
    for unit in (0, 1, 2**40):
        req = _req(lse_strides=st[:3] + (unit,))
        assert attn_bwd_support(req, "gfx942", policy=FULL).ok
        s = attn_bwd_plan(req, "gfx942", policy=FULL).launches[0].scalars
        assert (s["l_b"], s["l_h"], s["l_t"]) == (4099, 257, 3)


def test_band_bound_rows():
    for kw in (
        dict(left_bound=-2),
        dict(right_bound=2**30),
        dict(causal=True, left_bound=3),
    ):
        assert attn_bwd_support(_req(**kw), "gfx942", policy=FULL).code == "BAND_BOUND"


def test_request_device_hints_default_to_none():
    req = _req()
    assert req.num_cus is None and req.num_xcds is None
    assert req.max_total_q is None and req.block_table_injective is None
    # the reserved paged-table assertion never changes the general verdict
    assert attn_bwd_support(
        dataclasses.replace(req, block_table_injective=False), "gfx942", policy=FULL
    ).ok


# ---------------------------------------------------------------------------
# case table coverage
# ---------------------------------------------------------------------------

_CASES = backward_cases()


def test_case_table_size():
    assert len(_CASES) == 343


@pytest.mark.parametrize("arch", ("gfx942", "gfx950", "gfx1151", "gfx1201", "gfx90a"))
def test_every_case_yields_a_decision(arch):
    shipped = shipped_policy()
    for case in _CASES:
        req = request_from_case(case)
        for policy in (shipped, FULL, _WAIVED_UNDECLARED):
            v = attn_bwd_support(req, arch, policy=policy)
            assert v.ok or (v.code in REASON_CODES and v.detail), (case.id, v)
            if v.ok:
                attn_bwd_plan(req, arch, policy=policy)


@pytest.mark.parametrize("arch", DECLARED)
def test_case_table_cells_accepted(arch):
    for case in select_bwd("exhaustive"):
        req = request_from_case(case)
        v = attn_bwd_support(req, arch, policy=FULL)
        assert v.ok, (case.id, v)
        plan = attn_bwd_plan(req, arch, policy=FULL)
        main = plan.specs["main"]
        assert main.seq_mode == ("thd" if case.layout == "thd" else "batched")
        assert main.dkv_mode == ("atomic" if case.h_k != case.h_v else "direct")
        assert len(plan.launches) == (5 if main.dkv_mode == "atomic" else 3)


def test_case_table_gfx1201_is_declared_like_gfx1151():
    """gfx1201 is declared on its hardware run: no request is declined
    ``ARCH_NOT_VALIDATED``, and every case gets the gfx1151 verdict (the same
    RDNA functional matrix and performance status)."""
    assert "gfx1201" in DECLARED_ARCHS
    assert not shipped_policy().not_validated_archs
    for case in select_bwd("exhaustive"):
        req = request_from_case(case)
        v1201 = attn_bwd_support(req, "gfx1201", policy=FULL)
        assert v1201.code != "ARCH_NOT_VALIDATED"
        assert v1201.code == attn_bwd_support(req, "gfx1151", policy=FULL).code
        shipped = attn_bwd_support(req, "gfx1201")
        assert shipped.code == attn_bwd_support(req, "gfx1151").code


def test_shipped_policy_declares_the_gated_groups():
    # The base path, the masks-and-stats group, the grouped-heads group (GQA /
    # MQA with h_k == h_v), the split-kv-heads group (h_k != h_v, atomic
    # dK / dV), the padding group (per-batch SEQ_LEN counts) and the ragged
    # group (THD offsets) and the narrow-access group (stage_vec = 1: any
    # B/H/S stride, any element-aligned base) have passed their correctness
    # gates. The performance status is information only, so every target
    # serves exactly the cells of the declared groups.
    gated = frozenset(
        {"base", "masks_and_stats", "grouped_heads", "split_kv_heads", "padding"}
        | {"ragged", "narrow_access"}
    )
    assert gated == frozenset(FEATURE_GROUPS)
    assert SHIPPED_DECLARED_GROUPS == gated
    for case in _CASES:
        req = request_from_case(case)
        for arch in ("gfx942", "gfx950", "gfx1151"):
            v = attn_bwd_support(req, arch)
            if attn_bwd_feature_groups(req) <= gated:
                assert v.ok, (case.id, arch, v)
            else:
                assert v.code == "NOT_YET_DECLARED", (case.id, arch, v)


# ---------------------------------------------------------------------------
# workspace
# ---------------------------------------------------------------------------


def test_workspace_hand_table():
    req = _req()
    assert attn_bwd_workspace_bytes(req) == 1_048_576 + 8_192 + 8_192 == 1_064_960
    assert attn_bwd_workspace_bytes(req, g_split=2) == 1_064_960 + 2_097_152
    assert attn_bwd_workspace_bytes(req, dq_mode="split") == 16_384


def test_workspace_layout_packing_and_alignment():
    req = _req(b=3, s_q=33, s_kv=17, h_q=6, h_k=3, h_v=2, d_qk=64, d_v=64)
    lay = attn_bwd_workspace_layout(req)
    assert list(lay) == ["WS_DQ", "WS_LSE2", "WS_DSUM", "WS_DK", "WS_DV"]
    off = 0
    for name, (o, n) in lay.items():
        assert o == off and o % 256 == 0
        off += -(-n // 256) * 256
    assert attn_bwd_workspace_bytes(req) == off
    rows_q, rows_kv = 3 * 33, 3 * 17
    assert lay["WS_DQ"][1] == 4 * rows_q * 6 * 64
    assert lay["WS_LSE2"][1] == lay["WS_DSUM"][1] == 4 * rows_q * 6
    assert lay["WS_DK"][1] == 4 * rows_kv * 3 * 64
    assert lay["WS_DV"][1] == 4 * rows_kv * 2 * 64
    wl = attn_bwd_workspace_layout(req, worklist=True, block_n=128)
    # header (count, 0) plus two words per grid item: B * ceil(S_kv / kN0)
    assert wl["WORKLIST"][1] == 4 * (2 + 2 * 3 * 1)
    assert list(wl) == list(lay) + ["WORKLIST"]
    wl16 = attn_bwd_workspace_layout(req, worklist=True, block_n=16)
    assert wl16["WORKLIST"][1] == 4 * (2 + 2 * 3 * 2)


def test_workspace_thd_with_and_without_the_bound():
    thd = _req(
        layout="thd",
        b=4,
        s_q=100,
        s_kv=50,
        h_k=4,
        h_v=2,
        strides={n: (0, 128, 8 * 128, 1) for n in ("q", "o", "do", "dq")},
    )
    no_bound = attn_bwd_workspace_layout(thd)
    assert no_bound["WS_LSE2"][1] == 4 * (4 * 100) * 8
    assert no_bound["WS_DK"][1] == 4 * (4 * 50) * 4 * 128
    bound = attn_bwd_workspace_layout(
        dataclasses.replace(thd, max_total_q=123, max_total_kv=77)
    )
    assert bound["WS_LSE2"][1] == 4 * 123 * 8
    assert bound["WS_DQ"][1] == 4 * 123 * 8 * 128
    assert bound["WS_DK"][1] == 4 * 77 * 4 * 128
    # one bound alone is not used: rows stay per-sequence slots (B * S_max)
    for one in (dict(max_total_q=123), dict(max_total_kv=77)):
        assert attn_bwd_workspace_layout(dataclasses.replace(thd, **one)) == no_bound
    # padded rows use the tensor dims, exactly like fixed lengths
    padded = _req(padding=True, has_seq_len_q=True, has_seq_len_kv=True)
    assert attn_bwd_workspace_bytes(padded) == attn_bwd_workspace_bytes(_req())


def test_plan_layout_equals_the_formula():
    for req in (
        _req(),
        _req(h_k=4, h_v=2),
        _req(layout="thd", max_total_q=300),
        _req(layout="thd", max_total_q=300, max_total_kv=200),
        _req(layout="thd", h_k=4, h_v=2),
    ):
        plan = attn_bwd_plan(req, "gfx950", policy=FULL)
        assert dict(plan.workspace_layout) == attn_bwd_workspace_layout(req)
        assert plan.workspace_bytes == attn_bwd_workspace_bytes(req)
        main = next(s for s in plan.launches if s.stage == "main")
        by_token = req.max_total_q is not None and req.max_total_kv is not None
        rows_q = req.max_total_q if by_token else req.b * req.s_q
        assert main.scalars["ws_rows_q"] == rows_q


def test_thd_rows_by_sequence_slot_without_both_bounds():
    """``ws_seg`` is 1 in every launch of a THD plan that lacks either
    caller bound (rows ``b * S_max + i``, ``B * S_max`` rows), 0 with both
    (rows by storage token) and 0 for batched plans (the flag is not read)."""
    thd = dict(layout="thd", b=3, s_q=40, s_kv=24, h_k=4, h_v=2)
    cases = (
        (_req(**thd), 1, (3 * 40, 3 * 24)),
        (_req(**thd, max_total_q=500), 1, (3 * 40, 3 * 24)),
        (_req(**thd, max_total_kv=500), 1, (3 * 40, 3 * 24)),
        (_req(**thd, max_total_q=500, max_total_kv=90), 0, (500, 90)),
        (_req(h_k=4, h_v=2), 0, None),
    )
    for req, seg, rows in cases:
        plan = attn_bwd_plan(req, "gfx942", policy=FULL)
        for step in plan.launches:
            assert step.scalars["ws_seg"] == seg, (req, step.stage)
        main = plan.launches[1].scalars
        if rows is not None:
            assert (main["ws_rows_q"], main["ws_rows_kv"]) == rows


# ---------------------------------------------------------------------------
# plan details
# ---------------------------------------------------------------------------


def test_plan_kernargs_match_the_abi():
    for req in (
        _req(),
        _req(h_k=4, h_v=2, layout="thd"),
        _req(padding=True, has_seq_len_q=True, has_seq_len_kv=True),
    ):
        plan = attn_bwd_plan(req, "gfx942", policy=FULL)
        for step in plan.launches:
            names = [n for n, _ in attn_bwd_params(step.stage)]
            assert sorted(list(step.pointers) + list(step.scalars)) == sorted(names)
            assert all(isinstance(v, (int, float)) for v in step.scalars.values())


def test_plan_launch_sequence_and_grids():
    req = _req(s_kv=300, h_k=2, h_v=2)
    plan = attn_bwd_plan(req, "gfx942", policy=FULL)
    assert [s.stage for s in plan.launches] == ["prep", "main", "convert"]
    main = plan.launches[1]
    spec = plan.specs["main"]
    assert main.grid == (math.ceil(300 / spec.block_n), 2, 2)
    assert main.block == spec.waves * 64
    assert main.scalars["G"] == 4 and main.pointers["dK"] == "dk"
    atomic = attn_bwd_plan(_req(h_k=4, h_v=2), "gfx942", policy=FULL)
    assert [s.stage for s in atomic.launches] == [
        "prep",
        "main",
        "convert",
        "convert",
        "convert",
    ]
    m = atomic.launches[1]
    assert m.grid[1] == 8 // math.gcd(2, 4)
    assert m.pointers["dK"] is None and m.pointers["WS_DK"] == "WS_DK"
    assert atomic.launches[0].grid[1] == 8 + 4 + 2


def test_scale_placements():
    scale = 0.25

    def hook(name):
        return lambda req, arch, route, cls: {"scale_placement": name}

    for name, want in (
        ("fold_ds", (scale, 1.0, 1.0, 1.0)),
        ("convert", (1.0, scale, 1.0, scale)),
        ("dq_before_atomic", (1.0, scale, scale, 1.0)),
    ):
        policy = dataclasses.replace(FULL, tuning=hook(name))
        plan = attn_bwd_plan(_req(scale=scale), "gfx950", policy=policy)
        main, conv = plan.launches[1].scalars, plan.launches[2].scalars
        assert (main["ds_mult"], main["dk_mult"], main["dq_mult"], conv["mult"]) == want
        assert main["scale_log2"] == pytest.approx(scale * math.log2(math.e))


def test_main_is_compiled_for_the_scale_placement():
    """The plan compiles main for its scale placement (``static_scale``), so
    the kernel skips exactly the multipliers the plan passes as 1.0; a hook
    may pin ``"runtime"`` (every multiplier applied) but no other placement."""
    from kernels.common.attention_bwd import STATIC_SCALES, static_unit_mults

    assert set(STATIC_SCALES) == set(plan_mod.SCALE_PLACEMENTS) | {"runtime"}
    assert AttnBwdSpec(head_size=64).static_scale == "fold_ds"  # shipped default
    scale = 0.25
    for name, g_split in itertools.product(plan_mod.SCALE_PLACEMENTS, (1, 2)):
        knobs = {"scale_placement": name, "g_split": g_split}
        policy = dataclasses.replace(FULL, tuning=lambda *a, k=knobs: dict(k))
        plan = attn_bwd_plan(_req(scale=scale, h_k=2, h_v=2), "gfx942", policy=policy)
        spec = plan.specs["main"]
        assert spec.static_scale == name
        assert spec.dkv_mode == ("atomic" if g_split > 1 else "direct")
        main = plan.launches[1].scalars
        unit = static_unit_mults(name, spec.dkv_mode)
        for m in ("ds_mult", "dk_mult", "dq_mult"):
            assert (main[m] == 1.0) is (m in unit), (name, g_split, m)
        # the default placement keeps the unsalted (default) kernel name
        salted = (
            spec.kernel_name()
            != dataclasses.replace(spec, static_scale="fold_ds").kernel_name()
        )
        assert salted is (name != "fold_ds")
        pinned = {**knobs, "static_scale": "runtime"}
        policy = dataclasses.replace(FULL, tuning=lambda *a, k=pinned: dict(k))
        plan = attn_bwd_plan(_req(scale=scale, h_k=2, h_v=2), "gfx942", policy=policy)
        assert plan.specs["main"].static_scale == "runtime"
        assert static_unit_mults("runtime", spec.dkv_mode) == frozenset()
    bad = {"scale_placement": "convert", "static_scale": "fold_ds"}
    policy = dataclasses.replace(FULL, tuning=lambda *a: dict(bad))
    with pytest.raises(ValueError, match="static_scale"):
        attn_bwd_plan(_req(scale=scale), "gfx942", policy=policy)
    with pytest.raises(ValueError, match="static_scale"):
        AttnBwdSpec(head_size=64, static_scale="fold")


def test_g_split_from_the_tuning_hook_adds_kv_workspace():
    policy = dataclasses.replace(FULL, tuning=lambda *a: {"g_split": 2})
    req = _req(h_k=2, h_v=2)
    plan = attn_bwd_plan(req, "gfx942", policy=policy)
    assert plan.specs["main"].dkv_mode == "atomic"
    assert "WS_DK" in plan.workspace_layout
    assert plan.launches[1].grid[1] == 2 * 2
    bad = dataclasses.replace(FULL, tuning=lambda *a: {"g_split": 3})
    with pytest.raises(ValueError, match="g_split"):
        attn_bwd_plan(req, "gfx942", policy=bad)


def test_tuning_hook_rejects_unknown_knobs_and_applies_spec_knobs():
    with pytest.raises(ValueError, match="unknown knobs"):
        attn_bwd_plan(
            _req(),
            "gfx942",
            policy=dataclasses.replace(FULL, tuning=lambda *a: {"bogus": 1}),
        )
    policy = dataclasses.replace(
        FULL,
        tuning=lambda *a: {
            "edge_tiles": True,
            "block_n": 64,
            "waves": 4,
            "warp_grid_g4": (1, 4),
        },
    )
    spec = attn_bwd_plan(_req(), "gfx942", policy=policy).specs["main"]
    assert spec.edge_tiles is True and spec.block_n == 64


def test_mirror_pairing_and_xcd_remap_need_device_hints():
    mirror = dataclasses.replace(FULL, tuning=lambda *a: {"lb_order": "mirror"})
    with pytest.raises(ValueError, match="mask"):
        attn_bwd_plan(_req(), "gfx942", policy=mirror)
    with pytest.raises(ValueError, match="num_cus"):
        attn_bwd_plan(_req(causal=True), "gfx942", policy=mirror)
    filled = _req(causal=True, s_q=4096, s_kv=4096, num_cus=8)
    if not MIRROR_PAIRING_BUILT:
        # the main kernel runs one kv tile per CTA: the plan refuses pairing
        with pytest.raises(ValueError, match="not built"):
            attn_bwd_plan(filled, "gfx942", policy=mirror)
    else:  # the two-tile CTA (lb_mirror) is built
        plan = attn_bwd_plan(filled, "gfx942", policy=mirror)
        assert plan.specs["main"].lb_mirror is True
        assert plan.runtime["pair"] == 1 and plan.launches[1].scalars["pair"] == 1
        assert plan.launches[1].grid[0] == math.ceil(
            math.ceil(4096 / plan.specs["main"].block_n) / 2
        )
    reverse = dataclasses.replace(FULL, tuning=lambda *a: {"lb_order": "reverse"})
    plan = attn_bwd_plan(_req(causal=True), "gfx942", policy=reverse)
    assert plan.launches[1].scalars["lb_order"] == 1
    assert plan.launches[1].scalars["pair"] == 0
    xcd = dataclasses.replace(FULL, tuning=lambda *a: {"xcd_chunk": 64})
    with pytest.raises(ValueError, match="num_xcds"):
        attn_bwd_plan(_req(), "gfx942", policy=xcd)
    plan = attn_bwd_plan(_req(num_xcds=8), "gfx942", policy=xcd)
    assert (plan.runtime["xcd_n"], plan.runtime["xcd_chunk"]) == (8, 64)
    plain = attn_bwd_plan(_req(num_xcds=8), "gfx942", policy=FULL)
    assert plain.runtime["xcd_n"] == 0
    assert attn_bwd_plan(_req(), "gfx942", policy=FULL).fill_bucket == "unknown"
    assert attn_bwd_plan(_req(num_cus=4), "gfx942", policy=FULL).fill_bucket == "over"


def test_xcd_remap_selects_the_remap_kernel():
    xcd = dataclasses.replace(FULL, tuning=lambda *a: {"xcd_chunk": 64})
    plan = attn_bwd_plan(_req(num_xcds=8), "gfx942", policy=xcd)
    plain = attn_bwd_plan(_req(num_xcds=8), "gfx942", policy=FULL)
    assert plan.specs["main"].xcd_remap is True
    assert plain.specs["main"].xcd_remap is False
    assert plan.launches[1].kernel != plain.launches[1].kernel
    assert (
        plan.launches[1].scalars["xcd_n"],
        plan.launches[1].scalars["xcd_chunk"],
    ) == (
        8,
        64,
    )
    # the hint alone changes nothing; chunk = grid x is accepted
    assert plain.launches == attn_bwd_plan(_req(), "gfx942", policy=FULL).launches
    gx = plain.launches[1].grid[0]
    for chunk in (gx, 2 * gx, 4 * gx):
        pol = dataclasses.replace(FULL, tuning=lambda *a, c=chunk: {"xcd_chunk": c})
        p = attn_bwd_plan(_req(num_xcds=8), "gfx942", policy=pol)
        assert p.runtime["xcd_chunk"] == chunk and p.specs["main"].xcd_remap
    bad = dataclasses.replace(FULL, tuning=lambda *a: {"xcd_chunk": 5 * gx + 1})
    with pytest.raises(ValueError, match="xcd_chunk"):
        attn_bwd_plan(_req(num_xcds=8), "gfx942", policy=bad)
    # a remap kernel without the runtime remap stays legal (identity decode)
    forced = dataclasses.replace(FULL, tuning=lambda *a: {"xcd_remap": True})
    p = attn_bwd_plan(_req(), "gfx942", policy=forced)
    assert p.specs["main"].xcd_remap is True and p.runtime["xcd_n"] == 0


# ---------------------------------------------------------------------------
# THD work list
# ---------------------------------------------------------------------------


def _wl_policy(**extra):
    return dataclasses.replace(FULL, tuning=lambda *a: {"use_worklist": True, **extra})


def _thd_req(**kw):
    base = dict(layout="thd", b=3, s_q=100, s_kv=300, h_k=2, h_v=2, causal=True)
    base.update(kw)
    return _req(**base)


@pytest.mark.parametrize("arch", ["gfx942", "gfx950", "gfx1151", "gfx1201"])
def test_worklist_plan(arch):
    """``use_worklist``: main and prep built with ``worklist`` (the convert
    unchanged), the flag, batch count, kN0 and the list pointer in the prep
    and main kernargs, the same grids, and the ``WORKLIST`` term appended to
    an otherwise unchanged workspace."""
    req = _thd_req()
    plain = attn_bwd_plan(req, arch, policy=FULL)
    plan = attn_bwd_plan(req, arch, policy=_wl_policy())
    kn = plan.specs["main"].block_n
    assert (
        plan.runtime["use_worklist"] is True and plain.runtime["use_worklist"] is False
    )
    assert plan.specs["main"].worklist and plan.specs["prep"].worklist
    assert not plan.specs["convert"].worklist
    assert (
        dataclasses.replace(plan.specs["main"], worklist=False) == plain.specs["main"]
    )
    assert (
        dataclasses.replace(plan.specs["prep"], worklist=False) == plain.specs["prep"]
    )
    prep, main = plan.launches[0], plan.launches[1]
    assert main.scalars["use_worklist"] == prep.scalars["use_worklist"] == 1
    assert (prep.scalars["n_batch"], prep.scalars["wl_kn0"]) == (3, kn)
    assert main.pointers["WORKLIST"] == prep.pointers["WORKLIST"] == "WORKLIST"
    assert plain.launches[0].scalars["wl_kn0"] == 0
    assert plain.launches[1].pointers["WORKLIST"] is None
    assert [s.grid for s in plan.launches] == [s.grid for s in plain.launches]
    assert [s.block for s in plan.launches] == [s.block for s in plain.launches]
    assert prep.kernel == plain.launches[0].kernel + "_wl"
    assert main.kernel != plain.launches[1].kernel
    assert [s.kernel for s in plan.launches[2:]] == [
        s.kernel for s in plain.launches[2:]
    ]
    for name, place in plain.workspace_layout.items():
        assert plan.workspace_layout[name] == place
    assert list(plan.workspace_layout)[-1] == "WORKLIST"
    off, n = plan.workspace_layout["WORKLIST"]
    assert n == 4 * (2 + 2 * 3 * math.ceil(300 / kn))
    assert off % 256 == 0
    assert plan.workspace_bytes == plain.workspace_bytes + -(-n // 256) * 256
    # the other launches' kernargs are the plain plan's
    for a, b in zip(plan.launches[2:], plain.launches[2:]):
        assert a == b


def test_worklist_composes_with_the_head_split_atomic_mode_and_the_remap():
    for extra, req in (
        ({"g_split": 2}, _thd_req()),
        ({}, _thd_req(h_k=4, h_v=2)),
        ({"xcd_chunk": 64}, _thd_req(num_xcds=8)),
        ({"scale_placement": "convert"}, _thd_req()),
    ):
        plan = attn_bwd_plan(req, "gfx942", policy=_wl_policy(**extra))
        assert plan.specs["main"].worklist and plan.runtime["use_worklist"]
        for k, v in extra.items():
            assert plan.runtime[k] == v
        if "xcd_chunk" in extra:
            assert plan.specs["main"].xcd_remap and plan.runtime["xcd_n"] == 8
        if extra.get("g_split") == 2 or req.h_k != req.h_v:
            assert plan.specs["main"].dkv_mode == "atomic"
            assert {"WS_DK", "WS_DV"} <= set(plan.workspace_layout)
        assert list(plan.workspace_layout)[-1] == "WORKLIST"


def test_worklist_refusals():
    with pytest.raises(ValueError, match="THD"):
        attn_bwd_plan(_req(), "gfx942", policy=_wl_policy())
    for order in ("reverse", "interleave", "mirror"):
        with pytest.raises(ValueError, match="lb_order|mirror"):
            attn_bwd_plan(_thd_req(), "gfx942", policy=_wl_policy(lb_order=order))
    bad = dataclasses.replace(FULL, tuning=lambda *a: {"use_worklist": "yes"})
    with pytest.raises(ValueError, match="bool"):
        attn_bwd_plan(_thd_req(), "gfx942", policy=bad)
    # B * ceil(S_kv / kN0) items past i32 word indexing (kN0 = 16 on gfx1151;
    # rows by storage token keep the workspace rows in range)
    huge = _thd_req(
        b=65535, s_q=1, s_kv=2**20, h_q=1, h_k=1, h_v=1, d_qk=32, d_v=32,
        causal=False, max_total_q=65535, max_total_kv=2**20,
    )  # fmt: skip
    assert attn_bwd_support(huge, "gfx1151", policy=FULL).ok
    assert attn_bwd_plan(huge, "gfx1151", policy=FULL).runtime["use_worklist"] is False
    with pytest.raises(ValueError, match="i32"):
        attn_bwd_plan(huge, "gfx1151", policy=_wl_policy())
    # an explicit False is the default plan
    off = dataclasses.replace(FULL, tuning=lambda *a: {"use_worklist": False})
    assert attn_bwd_plan(_thd_req(), "gfx942", policy=off) == attn_bwd_plan(
        _thd_req(), "gfx942", policy=FULL
    )


def test_worklist_kernel_without_the_runtime_list():
    """A hook that pins ``worklist = True`` without ``use_worklist`` gets the
    work-list kernels with the flag 0 and no list (the grid decode)."""
    pinned = dataclasses.replace(FULL, tuning=lambda *a: {"worklist": True})
    plan = attn_bwd_plan(_thd_req(), "gfx942", policy=pinned)
    assert plan.specs["main"].worklist and plan.specs["prep"].worklist
    assert plan.launches[1].scalars["use_worklist"] == 0
    assert plan.launches[1].pointers["WORKLIST"] is None
    assert "WORKLIST" not in plan.workspace_layout
    with pytest.raises(ValueError, match="thd"):
        attn_bwd_plan(_req(), "gfx942", policy=pinned)


# ---------------------------------------------------------------------------
# tuning key: performance class plus kv-tile and grid-fill buckets
# ---------------------------------------------------------------------------


def test_tuning_key_appends_kv_tile_and_fill_buckets():
    req = _req()  # gfx942 d128: default kN0 = 128, so one kv tile
    key = attn_bwd_tuning_key(req, "gfx942")
    assert key[:-2] == attn_bwd_perf_class(req, "gfx942")
    assert key[-2:] == ("t1", "unknown")
    expect = {
        128: "t1",
        129: "t4",
        512: "t4",
        513: "t16",
        2048: "t16",
        8192: "t64",
        8193: "t>64",
    }
    for s_kv, bucket in expect.items():
        assert attn_bwd_tuning_key(_req(s_kv=s_kv), "gfx942")[-2] == bucket
    # default grid B * units * tiles = 2 * 8 * 1 = 16 against num_cus
    assert attn_bwd_tuning_key(_req(num_cus=17), "gfx942")[-1] == "under"
    assert attn_bwd_tuning_key(_req(num_cus=16), "gfx942")[-1] == "full"
    assert attn_bwd_tuning_key(_req(num_cus=9), "gfx942")[-1] == "full"
    assert attn_bwd_tuning_key(_req(num_cus=8), "gfx942")[-1] == "over"
    # atomic mode counts the atomic units (h_q / gcd group)
    hk = _req(h_k=4, h_v=2, num_cus=1000)
    assert attn_bwd_tuning_key(hk, "gfx942")[-1] == "under"
    # the buckets never change the class
    assert attn_bwd_perf_class(_req(s_kv=9000, num_cus=8), "gfx942") == (
        attn_bwd_perf_class(_req(s_kv=9000), "gfx942")
    )
    plan = attn_bwd_plan(_req(num_cus=8), "gfx942", policy=FULL)
    assert plan.tuning_key == attn_bwd_tuning_key(_req(num_cus=8), "gfx942")


def test_tuned_table_lookup_most_specific_first_then_the_hook():
    req = _req(num_cus=8)
    key = attn_bwd_tuning_key(req, "gfx942")
    base = key[:-2]
    kv, fill = key[-2:]
    table = {
        base + (ANY_BUCKET, ANY_BUCKET): {"lb_order": "reverse", "edge_tiles": True},
        base + (ANY_BUCKET, fill): {"scale_placement": "convert"},
        base + (kv, ANY_BUCKET): {"scale_placement": "dq_before_atomic"},
    }
    pol = dataclasses.replace(FULL, tuned_table=table)

    def got(r, policy=pol):
        p = attn_bwd_plan(r, "gfx942", policy=policy)
        return (
            p.runtime["scale_placement"],
            p.runtime["lb_order"],
            bool(p.specs["main"].edge_tiles),
        )

    # (kv, *) wins over (*, fill) and (*, *); entries do not merge
    assert got(req) == ("dq_before_atomic", "natural", False)
    exact = dict(table)
    exact[key] = {"scale_placement": "convert", "edge_tiles": True}
    assert got(req, dataclasses.replace(FULL, tuned_table=exact)) == (
        "convert",
        "natural",
        True,
    )
    # another kv-tile bucket: (*, fill), then (*, *) without a fill entry
    assert got(_req(s_q=1024, s_kv=1024, num_cus=8)) == ("convert", "natural", False)
    assert got(_req(s_q=1024, s_kv=1024)) == ("fold_ds", "reverse", True)
    # the hook overrides the table entry it lands on
    hooked = dataclasses.replace(pol, tuning=lambda *a: {"lb_order": "reverse"})
    assert got(req, hooked) == ("dq_before_atomic", "reverse", False)
    # another class never matches
    assert got(_req(causal=True, num_cus=8)) == ("fold_ds", "natural", False)
    # an empty table (the default) is the identity on the plan
    assert AttnBwdPolicy().tuned_table == {}
    assert attn_bwd_plan(req, "gfx942", policy=FULL) == attn_bwd_plan(
        req, "gfx942", policy=dataclasses.replace(FULL, tuned_table={})
    )


def test_head_packing_for_short_query_spans():
    plan = attn_bwd_plan(_req(s_q=1, s_kv=500, h_k=2, h_v=2), "gfx942", policy=FULL)
    spec = plan.specs["main"]
    assert spec.head_pack is True
    assert plan.launches[1].scalars["pack_heads"] == 4
    assert (
        plan.specs["main"].kernel_name()
        != attn_bwd_plan(_req(), "gfx942", policy=FULL).specs["main"].kernel_name()
    )
    assert (
        attn_bwd_plan(_req(), "gfx942", policy=FULL).launches[1].scalars["pack_heads"]
        == 0
    )


def test_thd_offsets_and_band_kernargs():
    req = _req(
        layout="thd",
        ragged_offset_dtype="int64",
        ragged_offset_multiplier=3,
        causal_bottom_right=True,
        strides={
            n: (0, 128, 8 * 128, 1)
            for n in ("q", "k", "v", "o", "do", "dq", "dk", "dv")
        },
    )
    m = attn_bwd_plan(req, "gfx942", policy=FULL).launches[1]
    s = m.scalars
    assert (s["off64"], s["q_mult"], s["q_div"], s["kv_div"]) == (
        1,
        3,
        8 * 128,
        8 * 128,
    )
    assert (s["left"], s["right"], s["bottom_right"]) == (-1, 0, 1)
    assert s["q_b"] == 0 and m.pointers["OFF_Q"] == "offsets_q"


# ---------------------------------------------------------------------------
# declaration agrees with the predicate
# ---------------------------------------------------------------------------


def test_declaration_shape_and_serialisable():
    for arch in ("gfx942", "gfx950", "gfx1151", "gfx1201", "gfx90a"):
        d = attn_bwd_declaration(arch, policy=FULL)
        json.dumps(d)
        assert d["version"] == ATTN_BWD_DECLARATION_VERSION == 4
        assert d["abi"] == "rocke.attn_bwd.v3"
        assert d["reason_codes"] == list(REASON_CODES)
        assert d["reason_kinds"] == {c: DECLINE_KINDS[c] for c in REASON_CODES}
        assert d["not_yet_implemented_prefix"] == "NOT_YET_IMPLEMENTED:"
        assert d["lse"]["base"] == "e" and d["lse"]["dtype"] == "fp32"
        assert "dq" in d["determinism"] and "dk_dv" in d["determinism"]
    assert attn_bwd_declaration("gfx1201")["status"] == "declared"
    not_validated = dataclasses.replace(
        FULL, not_validated_archs=frozenset({"gfx1201"})
    )
    assert attn_bwd_declaration("gfx1201", policy=not_validated)["status"] == (
        "not_validated"
    )
    assert attn_bwd_declaration("gfx90a")["status"] == "unsupported"
    d = attn_bwd_declaration("gfx942")
    assert d["perf_policy"] == {
        "gate": "information_only",
        "unverified": "served",
        "below_baseline": "served",
    }
    gated = dataclasses.replace(FULL, perf_gate=True)
    assert attn_bwd_declaration("gfx942", policy=gated)["perf_policy"] == {
        "gate": "declines",
        "unverified": "waived",
        "below_baseline": "declined",
    }
    assert {e["perf"] for e in d["perf"]} == {"unverified"}
    assert {e["perf"] for e in attn_bwd_declaration("gfx1151")["perf"]} == {"degraded"}
    assert {e["perf"] for e in attn_bwd_declaration("gfx1201")["perf"]} == {"degraded"}


def test_d256_decline_cites_the_scope_and_the_register():
    """d = 256 backward is declined as not implemented yet (the
    ``NOT_YET_IMPLEMENTED:`` detail prefix of the forward families), and the
    declaration publishes the register row with its consequence for hipDNN."""
    v = attn_bwd_support(_req(d_qk=256, d_v=256), "gfx942", policy=FULL)
    assert v.code == "HEAD_DIM_256_BWD" and v.kind == "not_yet_implemented"
    assert v.detail == f"NOT_YET_IMPLEMENTED: {REQ_SCOPE_D256_BWD}"
    assert "32, 64 and 128" in v.detail
    for arch in ("gfx942", "gfx950", "gfx1151", "gfx1201"):
        d = attn_bwd_declaration(arch)
        declined = d["axes"]["head_dim_declined"]["256"]
        assert declined["code"] == "HEAD_DIM_256_BWD"
        assert declined["kind"] == "not_yet_implemented"
        rows = {r["reason_code"]: r for r in d["deferral_register"]}
        row = rows["HEAD_DIM_256_BWD"]
        assert "not trainable" in row["consequence"]
        assert "CK / AOTriton" in row["consequence"]
        assert row["rationale"] == REQ_SCOPE_D256_BWD
        # every register row names a code the predicate really emits
        assert set(rows) <= set(REASON_CODES)
        assert all(r["consequence"] for r in d["deferral_register"])
        assert any("gfx1201" in n for n in d["notes"])
    assert [r["reason_code"] for r in DEFERRAL_REGISTER].count("HEAD_DIM_256_BWD") == 1


def test_every_reason_code_has_a_kind():
    """Every decline code of both entries has exactly one published kind, and
    the register covers every feature decline plus d = 256 with the kind of
    its code."""
    from kernels.common.attention_bwd_unified_plan import (
        REASON_CODES as UNIFIED_CODES,
    )

    assert set(REASON_CODES) | set(UNIFIED_CODES) == set(DECLINE_KINDS)
    assert set(DECLINE_KINDS.values()) == set(DECLINE_KIND_NAMES)
    rows = {r["reason_code"]: r for r in DEFERRAL_REGISTER}
    assert set(rows) == set(FEATURE_DECLINES) | {"HEAD_DIM_256_BWD"}
    for code, row in rows.items():
        assert row["kind"] == DECLINE_KINDS[code], code
        assert row["kind"] in ("out_of_scope", "not_yet_implemented"), code
        assert row["feature"] and row["rationale"] and row["consequence"], code
    # the permanent declines of this entry, and the features not built yet
    assert {c for c, r in rows.items() if r["kind"] == "out_of_scope"} == {
        "BIAS_IN_BACKWARD",
        "DBIAS",
        "PAGED",
    }
    assert {"DROPOUT", "ALIBI", "FP8", "DETERMINISTIC", "HEAD_DIM_256_BWD"} <= {
        c for c, r in rows.items() if r["kind"] == "not_yet_implemented"
    }
    # no plan label or requirement-section reference in a published string
    blob = json.dumps([dict(r) for r in DEFERRAL_REGISTER]) + REQ_SCOPE_D256_BWD
    for word in ("Tier", "section", "item "):
        assert word not in blob, word


@pytest.mark.parametrize(
    "kw,code,prefixed",
    [
        (dict(dropout=0.1), "DROPOUT", True),
        (dict(alibi=True), "ALIBI", True),
        (dict(fp8=True), "FP8", True),
        (dict(deterministic=True), "DETERMINISTIC", True),
        (dict(other_features=("block_mask",)), "UNSUPPORTED_FEATURE", True),
        (dict(has_bias=True), "BIAS_IN_BACKWARD", False),
        (dict(want_dbias=True), "DBIAS", False),
        (dict(paged=True), "PAGED", False),
        (dict(d_qk=96, d_v=96), "HEAD_DIM", True),
        (dict(left_bound=-5), "BAND_BOUND", False),
    ],
)
def test_decline_detail_carries_the_kind(kw, code, prefixed):
    v = attn_bwd_support(_req(**kw), "gfx950")
    assert v.code == code and v.kind == DECLINE_KINDS[code]
    assert v.detail.startswith("NOT_YET_IMPLEMENTED: ") == prefixed, v.detail
    assert attn_bwd_support(_req(), "gfx950").kind == ""


def _sample_requests(n, seed=7):
    rng = random.Random(seed)
    for _ in range(n):
        d = rng.choice((32, 64, 128))
        h_k = rng.choice((1, 2, 4, 8))
        h_v = rng.choice((h_k, h_k, 2))
        if 8 % h_v:
            h_v = h_k
        s_q = rng.choice((1, 17, 64, 128))
        s_kv = rng.choice((s_q, 33, 200))
        mask = rng.choice(("none", "tl", "br", "win"))
        kw = dict(
            b=rng.choice((1, 3)),
            h_q=8,
            h_k=h_k,
            h_v=h_v,
            s_q=s_q,
            s_kv=s_kv,
            d_qk=d,
            d_v=d,
            dtype=rng.choice(("fp16", "bf16")),
        )
        if mask == "tl":
            kw.update(right_bound=0)
        elif mask == "br":
            kw.update(right_bound=0, bottom_right=True)
        elif mask == "win":
            kw.update(left_bound=7, right_bound=0)
        lm = rng.choice(("fixed", "padded", "thd"))
        if lm == "padded":
            kw.update(padding=True, has_seq_len_q=True, has_seq_len_kv=True)
        elif lm == "thd":
            kw.update(layout="thd")
        if rng.random() < 0.2:
            kw.update(tensor_alignment=4)
        yield AttnBwdRequest(**kw)


def test_declaration_agrees_with_predicate():
    reqs = list(_sample_requests(300))
    rng = random.Random(11)
    statuses = {}
    for r in reqs:
        cls = attn_bwd_perf_class(r, "gfx942")
        statuses[cls] = rng.choice(("verified", "below_baseline", "unverified"))
    for gate, waive in ((False, False), (True, False), (True, True)):
        policy = AttnBwdPolicy(
            declared_groups=frozenset(FEATURE_GROUPS),
            waive_unverified=waive,
            perf_status=statuses,
            perf_gate=gate,
        )
        decl = attn_bwd_declaration("gfx942", policy=policy)
        table = {
            tuple(
                e[k]
                for k in (
                    "arch",
                    "route",
                    "d",
                    "dtype",
                    "mask",
                    "heads",
                    "length_relation",
                    "length_mode",
                    "stage_vec",
                )
            ): e["perf"]
            for e in decl["perf"]
        }
        for r in reqs:
            cls = attn_bwd_perf_class(r, "gfx942")
            assert cls in table
            status = table[cls]
            v = attn_bwd_support(r, "gfx942", policy=policy)
            if gate and status == "below_baseline":
                assert v.code == "PERF_BELOW_BASELINE"
            elif gate and status == "unverified" and not waive:
                assert v.code == "PERF_UNVERIFIED"
            else:
                assert v.ok, (r, v)
                plan = attn_bwd_plan(r, "gfx942", policy=policy)
                assert plan.perf_status == status


def test_declared_feature_groups_track_the_policy():
    partial = AttnBwdPolicy(
        declared_groups=frozenset({"base", "masks_and_stats"}), waive_unverified=True
    )
    decl = attn_bwd_declaration("gfx950", policy=partial)
    assert decl["declared_feature_groups"] == ["base", "masks_and_stats"]
    assert attn_bwd_support(_req(causal=True), "gfx950", policy=partial).ok
    v = attn_bwd_support(_req(h_k=2, h_v=2), "gfx950", policy=partial)
    assert v.code == "NOT_YET_DECLARED" and "grouped_heads" in v.detail


# ---------------------------------------------------------------------------
# no device query
# ---------------------------------------------------------------------------

_DEVICE_MODULES = (
    "torch",
    "hip",
    "ctypes",
    "rocke.runtime",
    "rocke.core.comgr",
    "subprocess",
)


def test_plan_modules_import_no_device_module():
    for fname in ("attention_bwd.py", "attention_bwd_plan.py"):
        tree = ast.parse((_KERNEL_DIR / fname).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for n in names:
                assert not any(
                    n == m or n.startswith(m + ".") for m in _DEVICE_MODULES
                ), (fname, n)
                assert n.split(".")[0] not in ("dispatch", "builders", "benchmarks"), (
                    fname,
                    n,
                )


def test_no_device_query(monkeypatch):
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if any(name == m or name.startswith(m + ".") for m in _DEVICE_MODULES):
            raise AssertionError(f"device module {name} imported during planning")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    for case in select_bwd("smoke"):
        req = request_from_case(case)
        for arch in ("gfx942", "gfx950", "gfx1151", "gfx1201"):
            attn_bwd_support(req, arch)
            if attn_bwd_support(req, arch, policy=FULL).ok:
                attn_bwd_plan(req, arch, policy=FULL)
        attn_bwd_declaration("gfx942")


# ---------------------------------------------------------------------------
# feature groups agree with the case-level feature table
# ---------------------------------------------------------------------------


_case_feature_rank = stage_of


def test_feature_group_counts_per_tier():
    counts = collections.Counter()
    for c in _CASES:
        rank = _case_feature_rank(c)
        for tier in ("smoke", "full", "exhaustive"):
            if tier in c.tags:
                counts[(rank, tier)] += 1
    expected = {
        1: (2, 16, 27),
        2: (16, 83, 121),
        3: (5, 26, 57),
        4: (0, 0, 2),
        5: (5, 32, 67),
        6: (3, 29, 69),
    }
    for rank, (smoke, full, exh) in expected.items():
        assert (
            counts[(rank, "smoke")],
            counts[(rank, "full")],
            counts[(rank, "exhaustive")],
        ) == (smoke, full, exh)


def test_stage_gate_selection_covers_every_base_layout_and_supplement():
    smoke1 = {c.layout for c in select_bwd("smoke") if stage_of(c) == 1}
    smoke1 |= {c.layout for c in SUPPLEMENTARY_CASES if stage_of(c) == 1}
    assert {"bhsd", "bshd", "strided_bhsd", "strided_bshd"} <= smoke1
    for c in SUPPLEMENTARY_CASES:
        groups = attn_bwd_feature_groups(request_from_case(c))
        assert stage_of(c) == 1 + max(FEATURE_GROUPS.index(g) for g in groups), c.id
        assert attn_bwd_support(request_from_case(c), "gfx942", policy=FULL).ok


def test_request_groups_match_case_features():
    for c in _CASES:
        groups = attn_bwd_feature_groups(request_from_case(c))
        assert "base" in groups and groups <= set(FEATURE_GROUPS)
        rank = 1 + max(FEATURE_GROUPS.index(g) for g in groups)
        want = _case_feature_rank(c)
        # packed QKV and an explicit scale equal to the default are not visible
        # in a request; everything else agrees exactly
        if c.layout == "packed_qkv" or (
            c.scale is not None and c.scale == 1 / math.sqrt(c.d)
        ):
            assert rank <= want, c.id
        else:
            assert rank == want, c.id


# ---------------------------------------------------------------------------
# routing
# ---------------------------------------------------------------------------


def test_routing_exactly_one_family_and_dense_only_when_eligible():
    eligible_keys = set()
    for c in _CASES:
        req = request_from_case(c)
        for arch in ("gfx942", "gfx950"):
            ok, _ = dense_bwd_eligibility(req, arch)
            if ok:
                eligible_keys.add((arch, plan_mod._dense_key(req)))
    assert eligible_keys
    policy = dataclasses.replace(FULL, dense_pack=frozenset(eligible_keys))
    routes = collections.Counter()
    for c in _CASES:
        req = request_from_case(c)
        for arch in DECLARED:
            if not attn_bwd_support(req, arch, policy=policy).ok:
                continue
            route, tag = attn_bwd_route(req, arch, policy=policy)
            assert route in ROUTES
            routes[route] += 1
            ok, _ = dense_bwd_eligibility(req, arch)
            if route == "dense":
                assert ok and tag == "dense_eligible"
            plan = attn_bwd_plan(req, arch, policy=policy)
            assert plan.route == route
    assert routes["dense"] and routes["general"]
    # an ineligible request sharing a packed dense key still takes the general route
    fixed = _req(b=1, s_q=128, s_kv=128)
    padded = dataclasses.replace(
        fixed, padding=True, has_seq_len_q=True, has_seq_len_kv=True
    )
    pack = frozenset({("gfx942", plan_mod._dense_key(fixed))})
    policy = dataclasses.replace(FULL, dense_pack=pack)
    assert attn_bwd_route(fixed, "gfx942", policy=policy)[0] == "dense"
    assert attn_bwd_route(padded, "gfx942", policy=policy) == (
        "general",
        "dense_ineligible_lengths",
    )
    assert attn_bwd_route(fixed, "gfx1151", policy=policy)[0] == "general"
    assert (
        attn_bwd_route(fixed, "gfx950", policy=policy)[1]
        == "dense_ineligible_no_tuned_config"
    )


def test_mixed_dense_layouts_never_take_the_dense_route():
    h, s, d = 8, 128, 128
    bhsd = (h * s * d, s * d, d, 1)
    bshd = (s * h * d, d, h * d, 1)
    all_bhsd = _req(
        strides={n: bhsd for n in ("q", "k", "v", "o", "do", "dq", "dk", "dv")}
    )
    all_bshd = _req(
        strides={n: bshd for n in ("q", "k", "v", "o", "do", "dq", "dk", "dv")}
    )
    mixed = _req(
        strides={
            **{n: bhsd for n in ("q", "o", "do", "dq")},
            **{n: bshd for n in ("k", "v", "dk", "dv")},
        }
    )
    mixed_rev = _req(
        strides={
            **{n: bshd for n in ("q", "o", "do", "dq")},
            **{n: bhsd for n in ("k", "v", "dk", "dv")},
        }
    )
    pack = frozenset(
        {
            ("gfx942", plan_mod._dense_key(all_bhsd)),
            ("gfx942", plan_mod._dense_key(all_bshd)),
        }
    )
    policy = dataclasses.replace(FULL, dense_pack=pack)
    assert attn_bwd_route(all_bhsd, "gfx942", policy=policy) == (
        "dense",
        "dense_eligible",
    )
    assert attn_bwd_route(all_bshd, "gfx942", policy=policy) == (
        "dense",
        "dense_eligible",
    )
    for req in (mixed, mixed_rev):
        assert dense_bwd_eligibility(req, "gfx942") == (
            False,
            "dense_ineligible_layout",
        )
        assert attn_bwd_route(req, "gfx942", policy=policy) == (
            "general",
            "dense_ineligible_layout",
        )
        assert attn_bwd_support(req, "gfx942", policy=policy).ok
        assert attn_bwd_plan(req, "gfx942", policy=policy).route == "general"
        assert attn_bwd_perf_class(req, "gfx942", "general")[:2] == (
            "gfx942",
            "general",
        )


# ---------------------------------------------------------------------------
# request normal form
# ---------------------------------------------------------------------------

_NAMES = ("q", "k", "v", "o", "do", "dq", "dk", "dv")


def test_request_normalized_is_plain_json_and_order_independent():
    from rocke.dispatch.core import stable_json_hash

    strides = {n: (128 * 128 * 8, 128 * 128, 128, 1) for n in _NAMES}
    a = _req(strides=strides, other_features=["x"], lse_dims=[2, 8, 128, 1], scale=1)
    shuffled = list(strides.items())
    random.Random(7).shuffle(shuffled)
    b = _req(
        strides={k: list(v) for k, v in shuffled},
        other_features=("x",),
        lse_dims=(2, 8, 128, 1),
        scale=1.0,
    )
    na = a.normalized()
    assert json.loads(json.dumps(na, sort_keys=True)) == na
    assert list(na["strides"]) == sorted(_NAMES)
    assert na["strides"]["q"] == [128 * 128 * 8, 128 * 128, 128, 1]
    assert na["other_features"] == ["x"] and isinstance(na["scale"], float)
    assert set(na) == {f.name for f in dataclasses.fields(AttnBwdRequest)}
    assert a == b and hash(a) == hash(b)
    assert stable_json_hash(na) == stable_json_hash(b.normalized())
    assert _req().normalized()["strides"] is None
    assert stable_json_hash(_req(b=3).normalized()) != stable_json_hash(
        _req().normalized()
    )
    # asdict / copy work, and the stride map is read-only
    assert dataclasses.asdict(a)["strides"] == dict(a.strides)
    import copy
    import pickle

    assert copy.deepcopy(a) == a and pickle.loads(pickle.dumps(a)) == a
    with pytest.raises(TypeError):
        a.strides["q"] = (0, 0, 0, 1)


def test_request_hash_is_stable_across_processes():
    import os
    import subprocess
    import sys

    from rocke.dispatch.core import stable_json_hash

    code = (
        "from kernels.common.attention_bwd_plan import AttnBwdRequest\n"
        "from rocke.dispatch.core import stable_json_hash\n"
        "r = AttnBwdRequest(b=2, h_q=8, h_k=8, h_v=8, s_q=128, s_kv=128, d_qk=128, d_v=128,\n"
        "    strides={'v': (1, 2, 3, 1), 'q': (4, 5, 6, 1)}, scale=0.25)\n"
        "print(stable_json_hash(r.normalized()))\n"
    )
    env = dict(os.environ, PYTHONHASHSEED="12345")
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    local = _req(strides={"q": (4, 5, 6, 1), "v": (1, 2, 3, 1)}, scale=0.25)
    assert out.stdout.strip() == stable_json_hash(local.normalized())


def test_perf_class_keys():
    cls = attn_bwd_perf_class(_req(causal=True, h_k=2, h_v=2, s_q=1, s_kv=40), "gfx950")
    assert cls == (
        "gfx950",
        "general",
        128,
        "fp16",
        "causal_tl",
        "gqa",
        "s_q_1",
        "fixed",
        8,
    )
    cls = attn_bwd_perf_class(
        _req(
            left_bound=3, right_bound=0, h_k=4, h_v=2, layout="thd", tensor_alignment=4
        ),
        "gfx942",
    )
    assert cls[4:] == ("window", "hk_ne_hv", "square", "thd", 1)
    # device-fill hints never enter the class
    assert attn_bwd_perf_class(
        _req(num_cus=304, num_xcds=8), "gfx942"
    ) == attn_bwd_perf_class(_req(), "gfx942")


@pytest.mark.parametrize(
    "strides, phrase",
    [
        ((128 * 128, 128, 1), "rank 3"),
        ((8 * 128 * 128, 128 * 128, 128, 1, 1), "rank 5"),
        ((), "rank 0"),
        ((8 * 128 * 128, 128 * 128, 128.0, 1), "non-integer"),
        ((8 * 128 * 128, 128 * 128, True, 1), "non-integer"),
        (7, "not a sequence"),
    ],
)
@pytest.mark.parametrize("name", ["q", "dv"])
def test_malformed_tensor_strides_decline_instead_of_raising(strides, phrase, name):
    """Every tensor takes exactly four integer strides; any other form is
    declined STRIDE_FORMAT with the cause named (it used to raise IndexError
    for rank 3 and ValueError for rank 5)."""
    v = attn_bwd_support(_req(strides={name: strides}), "gfx942", policy=FULL)
    assert v.code == "STRIDE_FORMAT" and phrase in v.detail and name in v.detail, v
    with pytest.raises(ValueError, match="STRIDE_FORMAT"):
        attn_bwd_plan(_req(strides={name: strides}), "gfx942", policy=FULL)


def test_numpy_integer_strides_are_accepted():
    np = pytest.importorskip("numpy")
    st = tuple(np.int64(x) for x in (8 * 128 * 128, 128 * 128, 128, 1))
    assert attn_bwd_support(_req(strides={"q": st}), "gfx942", policy=FULL).ok


# ---------------------------------------------------------------------------
# aux tuning values (prep / convert rows per CTA, convert vector width)
# ---------------------------------------------------------------------------


def _aux_plan(knobs, arch="gfx942", **req):
    policy = dataclasses.replace(FULL, tuning=lambda *a: dict(knobs))
    return attn_bwd_plan(_req(**req), arch, policy=policy)


def test_aux_knobs_are_plan_values_not_spec_fields():
    from kernels.common.attention_bwd_plan import AUX_KNOBS, RUNTIME_KNOBS

    assert AUX_KNOBS == ("prep_rows_per_cta", "convert_rows_per_cta", "convert_vec")
    fields = set(AttnBwdSpec.__dataclass_fields__)
    assert not fields & set(AUX_KNOBS) and not set(RUNTIME_KNOBS) & set(AUX_KNOBS)


@pytest.mark.parametrize("arch", ["gfx942", "gfx950", "gfx1151"])
@pytest.mark.parametrize("d", [32, 64, 128])
def test_aux_rows_per_cta_reaches_prep_and_convert(arch, d):
    """``"1x"`` (and an explicit count equal to the default) keeps the shipped
    aux kernels; ``"2x"`` / ``"4x"`` and explicit counts recompile prep or
    convert only: block, grid x and name follow, the main kernel does not."""
    plain = _aux_plan({}, arch, d_qk=d, d_v=d, s_q=300, s_kv=200, h_k=4, h_v=2)
    rows0 = 64 // (d // 8)
    assert plain.specs["prep"].rows_per_cta is None
    assert plain.specs["convert"].rows_per_cta is None
    assert plain.runtime["prep_rows_per_cta"] == "1x"
    assert plain.runtime["convert_rows_per_cta"] == "1x"
    assert plain.runtime["convert_vec"] == 8
    same = _aux_plan(
        {"prep_rows_per_cta": rows0, "convert_rows_per_cta": "1x"}, arch,
        d_qk=d, d_v=d, s_q=300, s_kv=200, h_k=4, h_v=2,
    )  # fmt: skip
    assert same.launches == plain.launches and same.specs == plain.specs
    for prep_x, conv_x in (("2x", "1x"), ("1x", "4x"), ("4x", "2x")):
        p = _aux_plan(
            {"prep_rows_per_cta": prep_x, "convert_rows_per_cta": conv_x}, arch,
            d_qk=d, d_v=d, s_q=300, s_kv=200, h_k=4, h_v=2,
        )  # fmt: skip
        assert p.specs["main"] == plain.specs["main"]
        assert p.launches[1] == plain.launches[1]
        for stage, x in (("prep", prep_x), ("convert", conv_x)):
            mult = int(x[0])
            rows = rows0 * mult
            spec = p.specs[stage]
            assert spec.rows_per_cta == (None if mult == 1 else rows)
            steps = [s for s in p.launches if s.stage == stage]
            for s in steps:
                assert s.block == 64 * mult
                assert s.kernel == spec.kernel_name(stage)
                assert (s.kernel == plain.specs[stage].kernel_name(stage)) == (
                    mult == 1
                )
            assert steps[0].grid[0] == -(-(300 if stage == "prep" else 300) // rows)
        assert p.runtime["prep_rows_per_cta"] == prep_x
        assert p.runtime["convert_rows_per_cta"] == conv_x


def test_aux_rows_per_cta_rejects_illegal_values():
    for bad in ("3x", "x2", 0, -16, 1.5, None):
        with pytest.raises(ValueError, match="prep_rows_per_cta"):
            _aux_plan({"prep_rows_per_cta": bad})
    # not a whole number of waves (d128: 16 lanes per row on a 64-wide wave)
    with pytest.raises(ValueError, match="multiple of the wave size"):
        _aux_plan({"convert_rows_per_cta": 3})
    # more than 1024 threads
    with pytest.raises(ValueError, match="at most 1024"):
        _aux_plan({"prep_rows_per_cta": 128})


@pytest.mark.parametrize("vec", [4, 16])
def test_convert_vec_reaches_the_convert_only(vec):
    plain = _aux_plan({}, h_k=4, h_v=2)
    p = _aux_plan({"convert_vec": vec}, h_k=4, h_v=2)
    assert p.specs["main"] == plain.specs["main"]
    assert p.specs["prep"] == plain.specs["prep"]
    assert p.launches[:2] == plain.launches[:2]
    conv = p.specs["convert"]
    assert conv.convert_vec == vec and conv.rows_per_cta is None
    lanes = 128 // vec
    steps = [s for s in p.launches if s.stage == "convert"]
    assert len(steps) == 3
    for s in steps:
        assert s.kernel == conv.kernel_name("convert") and s.kernel.endswith(f"_v{vec}")
        assert s.block == 64
        rows = 64 // lanes
        assert s.grid[0] == -(-128 // rows)
    assert p.runtime["convert_vec"] == vec
    # rows multiples scale the convert's own default rows
    p2 = _aux_plan({"convert_vec": vec, "convert_rows_per_cta": "2x"}, h_k=4, h_v=2)
    assert p2.specs["convert"].rows_per_cta == 2 * (64 // lanes)
    assert all(s.block == 128 for s in p2.launches if s.stage == "convert")
    with pytest.raises(ValueError, match="convert_vec"):
        _aux_plan({"convert_vec": 2})


def test_aux_knobs_pass_the_custom_configuration():
    from tests.sdpa.bwd_kernel_cases import custom_rules

    rules = custom_rules(
        json.dumps([{"knobs": {"convert_vec": 16, "prep_rows_per_cta": "2x",
                               "qdo_stage": "joint", "stats_load": "step_top"}}])
    )  # fmt: skip
    assert rules[0][1]["convert_vec"] == 16


# ---------------------------------------------------------------------------
# shipped policy, predicate-accept implies plannable, request types
# ---------------------------------------------------------------------------


def test_shipped_policy_is_the_default_policy_and_reads_no_table():
    """The kernel-level shipped policy is the dataclass default: no
    performance gate, no per-class status, no hidden tuning-module hook."""
    assert shipped_policy() == AttnBwdPolicy()
    p = shipped_policy()
    assert not p.perf_gate and not p.perf_status and not p.waive_unverified
    assert not p.tuned_table and not p.dense_pack and not p.instance_unavailable
    src = (_KERNEL_DIR / "attention_bwd_plan.py").read_text(encoding="utf-8")
    assert "attention_bwd_tuning" not in src
    assert not hasattr(plan_mod, "_tuning_module")


@pytest.mark.parametrize("arch", ("gfx942", "gfx950", "gfx1151", "gfx1201"))
def test_shipped_policy_serves_every_legal_request_without_override(arch):
    """No override: every exhaustive-tier case whose feature groups are
    declared is accepted by the shipped policy on every declared arch (gfx942
    and gfx950 included) and plans; the plan's status is information only."""
    served = 0
    for case in select_bwd("exhaustive"):
        req = request_from_case(case)
        v = attn_bwd_support(req, arch)
        if not attn_bwd_feature_groups(req) <= SHIPPED_DECLARED_GROUPS:
            assert v.code == "NOT_YET_DECLARED", (case.id, v)
            continue
        assert v.ok, (case.id, arch, v)
        plan = attn_bwd_plan(req, arch)
        assert plan.perf_status in ("unverified", "degraded"), plan.perf_status
        served += 1
    assert served


@pytest.mark.parametrize("arch", ("gfx942", "gfx950", "gfx1151", "gfx1201"))
def test_predicate_accept_implies_the_plan_builds(arch):
    """Every request of the case table the predicate accepts (shipped and
    full policies) plans, and the plan's main spec passes the family
    validator with the resource-gate list (RESOURCE_REFUSED)."""
    from kernels.common.attention_bwd import validate_attn_bwd_spec

    for case in _CASES:
        req = request_from_case(case)
        for policy in (None, FULL):
            if not attn_bwd_support(req, arch, policy=policy).ok:
                continue
            plan = attn_bwd_plan(req, arch, policy=policy)
            assert validate_attn_bwd_spec(plan.specs["main"], arch) == (
                plan.specs["main"]
            )


def test_resource_refused_instance_is_declined_not_raised(monkeypatch):
    """A resolved main spec on the resource-gate list is declined
    INSTANCE_UNAVAILABLE by the predicate (it never reaches the plan)."""
    import kernels.common.attention_bwd as bwd

    req = _req()
    assert attn_bwd_support(req, "gfx950").ok
    monkeypatch.setattr(bwd, "resource_refusal", lambda spec, arch: "scratch (test)")
    v = attn_bwd_support(req, "gfx950")
    assert v.code == "INSTANCE_UNAVAILABLE" and v.kind == "policy"
    assert "RESOURCE_REFUSED" in v.detail
    with pytest.raises(ValueError, match="INSTANCE_UNAVAILABLE.*RESOURCE_REFUSED"):
        attn_bwd_plan(req, "gfx950")


def test_untakeable_tuned_knobs_are_declined_not_raised():
    """A tuned value the request cannot take declines INSTANCE_UNAVAILABLE
    (first failure in the predicate's order) instead of raising in the plan
    of an accepted request."""
    bad = AttnBwdPolicy(tuning=lambda req, arch, route, pclass: {"g_split": 3})
    v = attn_bwd_support(_req(h_k=2, h_v=2), "gfx942", policy=bad)
    assert v.code == "INSTANCE_UNAVAILABLE" and "g_split=3" in v.detail
    with pytest.raises(ValueError, match="INSTANCE_UNAVAILABLE.*g_split=3"):
        attn_bwd_plan(_req(h_k=2, h_v=2), "gfx942", policy=bad)
    # a request check that fails first still wins
    v = attn_bwd_support(_req(dropout=0.5), "gfx942", policy=bad)
    assert v.code == "DROPOUT"


@pytest.mark.parametrize(
    "field_name", ["causal", "padding", "has_seq_len_kv", "deterministic", "fp8"]
)
@pytest.mark.parametrize("bad", [1, 0, "yes", None])
def test_request_flags_must_be_bools(field_name, bad):
    with pytest.raises(TypeError, match=f"{field_name} must be a bool"):
        _req(**{field_name: bad})


def test_request_extents_must_be_ints():
    import numpy as np

    for name, bad in (("s_q", 128.0), ("b", True), ("h_k", "8"), ("num_cus", 1.5)):
        with pytest.raises(TypeError, match=f"{name} must be"):
            _req(**{name: bad})
    with pytest.raises(TypeError, match="block_table_injective must be a bool"):
        _req(block_table_injective=1)
    # numpy integers are integers; None stays legal for the optional hints
    assert attn_bwd_support(_req(s_q=np.int64(128), num_cus=None), "gfx942").ok


def test_dense_route_configuration_is_answered_by_the_dense_family():
    """A policy whose tuning configures the dense family (dense-only knobs) on
    a dense-eligible request outside its dense pack: the general plan fails
    only through those knobs, so the dense family's plan decides (the dense
    planner's forced route). The same knobs on a request the dense family
    cannot take decline INSTANCE_UNAVAILABLE."""
    dense_knobs = AttnBwdPolicy(
        tuning=lambda req, arch, route, pclass: {"epilogue": "direct"}
    )
    req = _req(causal=True)
    assert dense_bwd_eligibility(req, "gfx950")[0]
    assert attn_bwd_support(req, "gfx950", policy=dense_knobs).ok
    with pytest.raises(ValueError, match="INSTANCE_UNAVAILABLE.*unknown knobs"):
        attn_bwd_plan(req, "gfx950", policy=dense_knobs)  # the general planner
    thd = _req(layout="thd")
    v = attn_bwd_support(thd, "gfx950", policy=dense_knobs)
    assert v.code == "INSTANCE_UNAVAILABLE" and "unknown knobs" in v.detail
    # RDNA has no dense family
    v = attn_bwd_support(req, "gfx1151", policy=dense_knobs)
    assert v.code == "INSTANCE_UNAVAILABLE"
    # the forced-route check: request and policy checks, no plan resolution
    from kernels.common.attention_bwd_plan import attn_bwd_request_verdict

    assert attn_bwd_request_verdict(thd, "gfx950", policy=dense_knobs).ok
    assert attn_bwd_request_verdict(_req(dropout=0.1), "gfx950").code == "DROPOUT"
    assert attn_bwd_request_verdict(_req(d_qk=256, d_v=256), "gfx950").kind == (
        "not_yet_implemented"
    )
    undeclared = AttnBwdPolicy(declared_groups=frozenset({"base"}))
    v = attn_bwd_request_verdict(_req(h_k=2, h_v=2), "gfx950", policy=undeclared)
    assert v.code == "NOT_YET_DECLARED"
