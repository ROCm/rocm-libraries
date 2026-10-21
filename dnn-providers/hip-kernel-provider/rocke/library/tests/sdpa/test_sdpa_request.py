# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``SdpaRequest``: features, normalization, validation, and the case table."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from dispatch.sdpa import (
    SDPA_DIM_VOCABULARY,
    SDPA_FEATURES,
    SdpaBias,
    SdpaRequest,
    SdpaTensor,
)
from rocke.dispatch.core import Capability, stable_json_hash
from tests.sdpa.request_cases import CASES, bhsd, bshd, make_request, packed

BASE = CASES["dense_bshd"]


# --- case table ---------------------------------------------------------------


def test_case_table_covers_every_feature():
    reported = frozenset().union(*(req.features() for req in CASES.values()))
    assert reported == SDPA_FEATURES


@pytest.mark.parametrize("name", sorted(CASES))
def test_case_reports_only_declared_features(name):
    assert CASES[name].features() <= SDPA_FEATURES


@pytest.mark.parametrize("name", sorted(CASES))
def test_normalized_round_trips(name):
    req = CASES[name]
    payload = json.loads(json.dumps(req.normalized()))
    rebuilt = SdpaRequest.from_normalized(payload)
    assert rebuilt == req
    assert stable_json_hash(rebuilt.normalized()) == stable_json_hash(payload)


@pytest.mark.parametrize(
    "name, expected",
    [
        ("dense_bshd", {"layout_bshd"}),
        ("dense_bhsd", {"layout_bhsd"}),
        ("gqa", {"layout_bshd", "gqa"}),
        ("mqa", {"layout_bshd", "gqa"}),
        ("kv_heads_differ", {"layout_bshd", "gqa", "kv_heads_differ"}),
        ("causal_tl", {"layout_bshd", "causal_tl"}),
        ("causal_br_square", {"layout_bshd", "causal_tl"}),
        ("causal_br", {"layout_bshd", "causal_br"}),
        ("window_causal", {"layout_bshd", "window"}),
        ("window_br", {"layout_bshd", "window", "window_br"}),
        ("window_right", {"layout_bshd", "window", "window_right"}),
        ("window_left_only", {"layout_bshd", "window", "window_right"}),
        ("bias_full", {"layout_bshd", "bias"}),
        ("bias_broadcast", {"layout_bshd", "bias", "bias_broadcast"}),
        ("padding_mask_only", {"layout_bshd", "padded"}),
        ("fp8", {"layout_bshd", "fp8", "dtype_mixed"}),
        ("fp8_s_o", {"layout_bshd", "fp8", "fp8_s_o"}),
        (
            "strided",
            {"layout_strided", "kv_layout_differs", "o_layout_differs", "batch_stride"},
        ),
        ("batch_stride", {"layout_bshd", "batch_stride"}),
        ("d_stride", {"layout_bshd", "kv_layout_differs", "d_stride", "batch_stride"}),
        ("scale_default", {"layout_bshd"}),
        ("causal_br_lse", {"layout_bshd", "causal_br", "lse"}),
    ],
)
def test_case_features(name, expected):
    assert CASES[name].features() == frozenset(expected)


# --- every row of the coverage table changes features() or normalized() --------

# (row, case, decline feature). A row with no feature is supported by some
# kernel today; the test then shows the field still reaches normalized().
COVERAGE_ROWS = [
    ("Q/K/V/O fp16, bf16", "bf16", None),
    ("fp32 compute", "compute_fp16", "compute_dtype"),
    ("General strides", "strided", "layout_strided"),
    ("Free batch stride", "batch_stride", "batch_stride"),
    ("d_stride != 1", "d_stride", "d_stride"),
    ("GQA/MQA (h_k == h_v)", "gqa", "gqa"),
    ("h_k != h_v", "kv_heads_differ", "kv_heads_differ"),
    ("d_qk != d_v", "d_v_differs", "d_v_differs"),
    ("Scale value", "scale_default", None),
    ("Scale tensor", "scale_tensor", "scale_tensor"),
    ("Causal top-left", "causal_tl", "causal_tl"),
    ("Causal bottom-right", "causal_br", "causal_br"),
    ("Window band", "window_causal", "window"),
    ("Window with right > 0", "window_right", "window_right"),
    ("Bias tensor", "bias_full", "bias"),
    ("Bias broadcast", "bias_broadcast", "bias_broadcast"),
    ("ALiBi", "alibi", "alibi"),
    ("Padding (SEQ_LEN_*)", "padded", "padded"),
    ("Ragged", "ragged", "ragged"),
    ("Stats (LSE)", "lse", "lse"),
    ("Max, Sum_exp", "max_sum_exp", "max_sum_exp"),
    ("Dropout", "dropout", "dropout"),
    ("Paged KV", "paged", "paged"),
    ("Block mask", "block_mask", "block_mask"),
    ("Sink token", "sinks", "sinks"),
    ("FP8 q/k/v descale", "fp8", "fp8"),
    ("FP8 S/O scale, amax", "fp8_s_o", "fp8_s_o"),
    ("mma_core_mode", "mma_core_mode", "mma_core_mode"),
    ("implementation unified", "implementation_unified", "implementation_hint"),
    ("implementation composite", "implementation_composite", "implementation_hint"),
]


@pytest.mark.parametrize(
    "case, feature", [r[1:] for r in COVERAGE_ROWS], ids=[r[0] for r in COVERAGE_ROWS]
)
def test_coverage_row_is_represented(case, feature):
    req = CASES[case]
    assert req.normalized() != BASE.normalized()
    if feature is not None:
        assert feature in req.features()
        assert feature not in BASE.features()


# --- normalization --------------------------------------------------------------


def test_minus_one_is_unbounded_and_is_kept():
    req = replace(BASE, left_bound=-1, right_bound=-1)
    assert (req.left_bound, req.right_bound) == (-1, -1)
    assert req.features() == BASE.features()


@pytest.mark.parametrize("left, right", [(-2, -1), (-1, -2), (-7, -2), (-4, 0)])
def test_bounds_below_minus_one_are_rejected(left, right):
    """hipDNN's resolvers throw for a bound < -1; it must not read as unbounded."""
    with pytest.raises(ValueError, match=r"must be >= -1"):
        replace(BASE, left_bound=left, right_bound=right)


@pytest.mark.parametrize("arch", ["GFX950", " gfx950 ", "gfx950:sramecc+:xnack-"])
def test_arch_is_canonical(arch):
    """A target-ID spelling is the same arch, so Capability's exact match admits it."""
    req = replace(BASE, arch=arch)
    assert req.arch == "gfx950"
    cap = Capability(arches=("gfx950",), supports_features=BASE.features())
    assert cap.check(req) == (True, "ok")


@pytest.mark.parametrize("name", ["padded", "ragged", "paged"])
def test_bottom_right_on_per_sequence_lengths_moves(name):
    """Equal max extents do not make a per-sequence bottom-right diagonal top-left.

    The offset is ``s_kv[b] - s_q[b]``, so a top-left-only kernel would mask
    batches whose lengths differ wrongly.
    """
    base = CASES[name]
    assert base.s_q == base.s_kv
    causal = replace(base, diagonal="bottom_right", left_bound=-1, right_bound=0)
    assert "causal_br" in causal.features()
    assert "causal_tl" not in causal.features()
    window = replace(base, diagonal="bottom_right", left_bound=2, right_bound=0)
    assert "window_br" in window.features()


@pytest.mark.parametrize("source", ["default", "tensor"])
def test_scale_is_one_unless_baked(source):
    extra = {"scale_tensor": packed((1,))} if source == "tensor" else {}
    req = replace(BASE, scale_source=source, scale=0.125, **extra)
    assert req.scale == 1.0


def test_bias_rank_and_broadcast_strides_have_one_spelling():
    rank2 = SdpaBias(packed((8, 8)))
    rank4 = SdpaBias(packed((1, 1, 8, 8)))
    assert rank2 == rank4
    assert rank2.tensor.dims == (1, 1, 8, 8)
    assert rank2.tensor.strides == (0, 0, 8, 1)


def test_bias_size_one_target_is_not_broadcast():
    req = make_request(b=1, bias=SdpaBias(packed((1, 4, 8, 8))))
    assert "bias" in req.features()
    assert "bias_broadcast" not in req.features()


def test_size_one_axes_ignore_strides():
    # B == 1 and H == 1: neither the batch nor the head stride means anything.
    q = SdpaTensor("fp16", (1, 1, 8, 16), (999, 7, 16, 1))
    req = make_request(b=1, h_q=1, h_k=1, q=q)
    assert req.features() == {"layout_bshd"}


def test_ragged_batch_stride_is_ignored():
    req = CASES["ragged"]
    q = replace(req.q, strides=(12345,) + req.q.strides[1:])
    assert "batch_stride" not in replace(req, q=q).features()


def test_dims_vocabulary():
    req = make_request(h_q=8, h_k=2, h_v=4, s_q=4, s_kv=8, d=16, d_v=8)
    assert tuple(req.dims()) == SDPA_DIM_VOCABULARY
    assert req.dims() == {
        "batch": 2,
        "h_q": 8,
        "h_k": 2,
        "h_v": 4,
        "s_q": 4,
        "s_kv": 8,
        "d_qk": 16,
        "d_v": 8,
        "gqa_ratio": 4,
    }


# --- validation: only what the frontend (or a Python caller) could break ------


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"q": packed((2, 4, 8))}, "q must be rank 4"),
        ({"k": bshd(3, 4, 8, 16)}, "batch mismatch"),
        ({"k": bshd(2, 4, 8, 32)}, "head dim mismatch"),
        ({"v": bshd(2, 4, 9, 16)}, "seq_kv mismatch"),
        ({"k": bshd(2, 3, 8, 16), "v": bshd(2, 3, 8, 16)}, "divisible"),
        ({"o": bshd(2, 4, 8, 8)}, "O dims"),
        ({"scale_source": "tensor"}, "needs scale_tensor"),
        ({"scale_tensor": packed((1,))}, "mutually exclusive"),
        (
            {"scale_source": "tensor", "scale_tensor": packed((2,))},
            "volume 1",
        ),
        ({"bias": SdpaBias(packed((8, 4)))}, "s_kv dim"),
        ({"bias": SdpaBias(packed((4, 8)))}, "s_q dim"),
        ({"bias": SdpaBias(packed((2, 2, 8, 8)))}, "h_q dim"),
        ({"diagonal": "center"}, "diagonal"),
        ({"implementation": "fast"}, "implementation"),
        ({"compute_dtype": "fp12"}, "compute dtype"),
        ({"mma_core_mode": "fp12"}, "mma_core_mode"),
    ],
)
def test_malformed_requests_are_rejected(overrides, match):
    with pytest.raises(ValueError, match=match):
        replace(BASE, **overrides)


def test_bias_rank_above_four_is_rejected():
    with pytest.raises(ValueError, match="bias rank"):
        SdpaBias(packed((1, 2, 4, 8, 8)))


def test_tensor_rejects_mismatched_strides():
    with pytest.raises(ValueError, match="differ in rank"):
        SdpaTensor("fp16", (2, 4), (4,))


def test_valid_but_unimplemented_is_not_an_error():
    # Every decline-only case constructs; the decline is a feature, not a raise.
    for name in ("alibi", "dropout", "paged", "block_mask", "sinks", "fp8_s_o"):
        assert CASES[name].features() - BASE.features()


# --- Capability.check gates on the request --------------------------------------


def test_capability_admits_and_declines_with_reason():
    cap = Capability(
        arches=("gfx950",),
        dtypes=("fp16", "bf16"),
        supports_features=frozenset({"layout_bshd", "gqa", "causal_tl", "causal_br"}),
    )
    assert cap.check(CASES["gqa"]) == (True, "ok")
    ok, why = cap.check(CASES["bias_broadcast"])
    assert not ok and "bias" in why
    ok, why = cap.check(CASES["fp8"])
    assert not ok and "fp8_e4m3" in why
    ok, why = cap.check(make_request(layout=bhsd))
    assert not ok and "layout_bhsd" in why
