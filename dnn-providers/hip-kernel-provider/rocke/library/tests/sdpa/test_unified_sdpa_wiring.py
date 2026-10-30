# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``dispatch.sdpa.bind_unified``: routing onto the unified attention candidates.

Every case uses a real ``SdpaRequest`` built from ``request_cases.py``.
Decline-reason assertions check the feature name the caller can act on.
"""

from __future__ import annotations

import itertools
import tracemalloc
from dataclasses import replace

import numpy as np
import pytest

from dispatch.sdpa.bind_unified import (
    _DECLINED,
    SDPA_UNIFIED_FEATURES,
    _kept_keys,
    sdpa_unified_admits,
    sdpa_unified_candidates,
)
from dispatch.sdpa.common import (
    SDPA_FEATURES,
    SdpaBias,
    SdpaDropout,
    SdpaFp8,
    SdpaPaged,
)
from tests.sdpa.request_cases import bhsd, bshd, make_request, packed, ragged_offset

UNIFIED_HEAD_SIZES = (64, 128, 256)
ARCHES = ("gfx942", "gfx950")


def _unified_request(arch="gfx950", d=128, dtype="fp16", **overrides):
    return make_request(arch=arch, d=d, dtype=dtype, **overrides)


def _ragged(t):
    return replace(
        t, ragged_offset=ragged_offset(t.dims[0]), ragged_offset_multiplier=t.strides[2]
    )


def _declined(req):
    ok, reason = sdpa_unified_admits(req)
    assert not ok
    assert sdpa_unified_candidates(req) == ()
    return reason


def _admitted(req):
    ok, reason = sdpa_unified_admits(req)
    assert ok, reason
    assert sdpa_unified_candidates(req)


# --- every feature is carried or declined ------------------------------------


def test_every_sdpa_feature_is_classified_exactly_once():
    declined = frozenset(_DECLINED)
    assert SDPA_UNIFIED_FEATURES | declined == SDPA_FEATURES
    assert not SDPA_UNIFIED_FEATURES & declined


def test_decline_reasons_start_with_their_feature():
    for feature, reason in _DECLINED.items():
        assert reason.startswith(f"{feature}: "), feature


_BASE = _unified_request()
_DECODE = _unified_request(s_q=1, s_kv=64)
_DECLINED_CASES = {
    "kv_heads_differ": _unified_request(h_q=8, h_k=2, h_v=4),
    "dtype_mixed": replace(_BASE, o=bshd(2, 4, 8, 128, "fp32")),
    "compute_dtype": replace(_BASE, compute_dtype="fp16"),
    "scale_tensor": replace(_BASE, scale_source="tensor", scale_tensor=packed((1,))),
    "bias": replace(_BASE, bias=SdpaBias(packed((2, 4, 8, 8)))),
    "bias_broadcast": replace(_BASE, bias=SdpaBias(packed((8, 8)))),
    "alibi": replace(_BASE, alibi=True),
    "sinks": replace(_BASE, sink_token=packed((1, 4, 1, 1))),
    "dropout": replace(_BASE, dropout=SdpaDropout(0.1)),
    "paged": replace(_BASE, paged=SdpaPaged(max_seq_len_kv=8)),
    "block_mask": replace(_BASE, block_mask=packed((2, 4, 1, 1), "uint8")),
    "fp8": _unified_request(dtype="fp8_e4m3", fp8=SdpaFp8(descale_q=packed((1,)))),
    "fp8_s_o": _unified_request(
        dtype="fp8_e4m3", fp8=SdpaFp8(descale_q=packed((1,)), scale_o=packed((1,)))
    ),
    "max_sum_exp": replace(
        _DECODE, max=packed((2, 4, 1, 1)), sum_exp=packed((2, 4, 1, 1))
    ),
}


def test_every_declined_feature_has_a_case():
    assert set(_DECLINED_CASES) == set(_DECLINED)


@pytest.mark.parametrize("feature", sorted(_DECLINED_CASES))
def test_declined_feature_is_declined_with_a_declined_features_reason(feature):
    req = _DECLINED_CASES[feature]
    assert feature in req.features()
    reason = _declined(req)
    assert reason.split(":")[0] in req.features() & set(_DECLINED)


# --- head_dim / arch validity ------------------------------------------------


def test_unsupported_head_dim_reports_head_dim_before_a_feature_gap():
    req = replace(_unified_request(d=32), paged=SdpaPaged(max_seq_len_kv=8))
    assert "head_dim" in _declined(req)


def test_unknown_arch_declines():
    assert "arch" in _declined(_unified_request(arch="gfx000"))


# --- value checks on carried hints -------------------------------------------


def test_mma_core_mode_matching_the_dtype_is_admitted():
    _admitted(replace(_BASE, mma_core_mode="fp16"))


def test_mma_core_mode_other_than_the_dtype_is_declined():
    assert _declined(replace(_BASE, mma_core_mode="fp32")).startswith("mma_core_mode:")


def test_implementation_unified_is_admitted():
    _admitted(replace(_BASE, implementation="unified"))


def test_implementation_composite_is_declined():
    assert _declined(replace(_BASE, implementation="composite")).startswith(
        "implementation_hint:"
    )


def test_padding_mask_without_seq_len_tensors_is_declined():
    assert _declined(replace(_BASE, padding_mask=True)).startswith("padded:")


@pytest.mark.parametrize(
    "pin",
    [
        {"algorithm": "unified_2d"},
        {"algorithm": "no_such_algorithm"},
        {"spec_id": "no_such_spec"},
        {"tuning_id": "t0"},
        {"tuning_knobs": (("block_m", 64),)},
    ],
    ids=["algorithm", "unknown_algorithm", "spec_id", "tuning_id", "tuning_knobs"],
)
def test_dispatch_pin_is_declined(pin):
    """The unified launch takes no pin, so a pinned request would run unpinned."""
    assert _declined(replace(_BASE, **pin)).startswith("dispatch_pin:")


# --- carried features --------------------------------------------------------


@pytest.mark.parametrize("d", UNIFIED_HEAD_SIZES)
@pytest.mark.parametrize("dtype", ("fp16", "bf16"))
def test_full_attention_is_admitted_on_gfx950(d, dtype):
    _admitted(_unified_request(arch="gfx950", d=d, dtype=dtype))


@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize("d", UNIFIED_HEAD_SIZES)
@pytest.mark.parametrize("dtype", ("fp16", "bf16"))
def test_causal_is_admitted(arch, d, dtype):
    _admitted(_unified_request(arch=arch, d=d, dtype=dtype, right_bound=0))


@pytest.mark.parametrize(
    "req",
    [
        _unified_request(layout=bhsd),
        replace(_BASE, k=bhsd(2, 4, 8, 128)),
        replace(_BASE, o=bhsd(2, 4, 8, 128)),
        replace(
            _BASE,
            q=replace(
                _BASE.q, strides=(_BASE.q.strides[0] + 64,) + _BASE.q.strides[1:]
            ),
        ),
        replace(_BASE, q=replace(_BASE.q, alignment=4)),
    ],
    ids=["bhsd", "kv_layout_differs", "o_layout_differs", "batch_stride", "unaligned"],
)
def test_layouts_are_admitted(req):
    _admitted(req)


def test_padded_is_admitted():
    lens = packed((2, 1, 1, 1), "int32")
    _admitted(replace(_BASE, padding_mask=True, seq_len_q=lens, seq_len_kv=lens))


def test_ragged_is_admitted():
    _admitted(
        replace(
            _BASE,
            q=_ragged(_BASE.q),
            k=_ragged(_BASE.k),
            v=_ragged(_BASE.v),
            o=_ragged(_BASE.o),
        )
    )


def test_causal_bottom_right_is_admitted():
    _admitted(_unified_request(s_q=4, s_kv=8, right_bound=0, diagonal="bottom_right"))


def test_sliding_window_is_admitted():
    _admitted(replace(_BASE, left_bound=3, right_bound=0))


def test_lse_is_admitted_on_a_decode_shape():
    """Only the split-KV (3D) path writes LSE."""
    _admitted(replace(_unified_request(s_q=1, s_kv=4096), stats=packed((2, 4, 1, 1))))


def test_lse_is_declined_on_a_prefill_shape():
    reason = _declined(replace(_BASE, stats=packed((2, 4, 8, 1))))
    assert "split-KV" in reason


# --- band faithfulness -------------------------------------------------------


def test_gfx942_full_attention_is_declined():
    """gfx942's unified kernels serve NO_MASK with the bottom-right causal body
    (see ``normalize_band``), so full attention would come back causal."""
    reason = _declined(_unified_request(arch="gfx942"))
    assert reason.startswith("band:")
    assert "full attention" in reason


def test_gfx942_full_attention_decode_is_admitted():
    """With one query row, bottom-right causal keeps every key."""
    _admitted(_unified_request(arch="gfx942", s_q=1, s_kv=4096))


def test_gfx942_ragged_full_attention_decode_is_admitted():
    base = _unified_request(arch="gfx942", s_q=1, s_kv=64)
    _admitted(
        replace(
            base,
            q=_ragged(base.q),
            k=_ragged(base.k),
            v=_ragged(base.v),
            o=_ragged(base.o),
        )
    )


def test_gfx942_noncausal_band_is_declined():
    req = replace(_unified_request(arch="gfx942"), left_bound=2, right_bound=1)
    assert _declined(req).startswith("band:")


def test_gfx950_noncausal_band_is_admitted():
    _admitted(replace(_BASE, left_bound=2, right_bound=1))


# Arches outside the attention layer's band dispatch: the unified problem is
# always built bottom-right causal there, whatever band the request names.
NON_BAND_ARCHES = ("gfx1151", "gfx1201", "gfx90a", "gfx1100")


@pytest.mark.parametrize("arch", NON_BAND_ARCHES)
@pytest.mark.parametrize(
    "bounds", [{"right_bound": 0}, {"left_bound": 2, "right_bound": 0}]
)
def test_non_band_arch_top_left_with_unequal_lengths_is_declined(arch, bounds):
    req = _unified_request(arch=arch, s_q=4, s_kv=8, **bounds)
    assert _declined(req).startswith("band:")


@pytest.mark.parametrize("arch", NON_BAND_ARCHES)
def test_non_band_arch_bottom_right_causal_is_admitted(arch):
    _admitted(
        _unified_request(
            arch=arch, s_q=4, s_kv=8, right_bound=0, diagonal="bottom_right"
        )
    )


@pytest.mark.parametrize("arch", NON_BAND_ARCHES)
def test_non_band_arch_square_top_left_causal_is_admitted(arch):
    """At ``s_q == s_kv`` with fixed lengths the two diagonals coincide."""
    _admitted(_unified_request(arch=arch, right_bound=0))


@pytest.mark.parametrize("arch", NON_BAND_ARCHES)
def test_non_band_arch_full_attention_is_declined(arch):
    assert "full attention" in _declined(_unified_request(arch=arch))


def _dense_keep(s_q, s_kv, top_left, left, right):
    q = np.arange(s_q)[:, None]
    k = np.arange(s_kv)[None, :]
    d = q + (0 if top_left else s_kv - s_q)
    keep = np.ones((s_q, s_kv), dtype=bool)
    if right >= 0:
        keep &= k <= d + right
    if left >= 0:
        keep &= k >= d - left
    return keep


def test_kept_keys_matches_the_dense_mask():
    for s_q, s_kv, top_left, left, right in itertools.product(
        (1, 3, 8), (1, 5, 8), (True, False), (-1, 0, 2, 9), (-1, 0, 1, 9)
    ):
        first, last = _kept_keys(s_q, s_kv, top_left, left, right)
        keep = _dense_keep(s_q, s_kv, top_left, left, right)
        k = np.arange(s_kv)[None, :]
        assert np.array_equal(keep, (k >= first[:, None]) & (k <= last[:, None])), (
            s_q,
            s_kv,
            top_left,
            left,
            right,
        )


def test_long_context_band_check_is_not_quadratic():
    """gfx942 resolves square top-left causal to bottom-right, so the gate
    compares kept keys; at 128k a dense mask would need tens of GiB."""
    req = _unified_request(arch="gfx942", s_q=131072, s_kv=131072, right_bound=0)
    tracemalloc.start()
    try:
        _admitted(req)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 64 << 20, peak


# --- d=256 -------------------------------------------------------------------


@pytest.mark.parametrize("arch", ARCHES)
def test_d256_decode_reaches_the_dedicated_decode_candidate(arch):
    """``attention_d256_decode`` serves gfx942 decode as well as gfx950."""
    req = _unified_request(arch=arch, d=256, dtype="bf16", s_q=1, s_kv=4096)
    _admitted(req)
    assert "attention_d256_decode" in sdpa_unified_candidates(req)


def _gfx942_d256_fast(req) -> bool:
    from dispatch.attention.common import _problem
    from dispatch.sdpa.bind_unified import _to_attention_request
    from kernels.common.attention_unified import _d256_gfx942_fast

    return _d256_gfx942_fast(_problem(_to_attention_request(req)), "gfx942")


def test_gfx942_bf16_d256_prefill_lands_in_the_fast_cohort():
    """gfx942 has no dedicated d=256 prefill candidate; its fast path is the
    ``_d256_gfx942_fast`` cohort inside ``attention_unified_2d``, so assert
    the cohort itself rather than the generic candidate's name."""
    req = _unified_request(arch="gfx942", d=256, dtype="bf16", right_bound=0)
    _admitted(req)
    assert _gfx942_d256_fast(req)


def test_gfx942_d256_outside_the_fast_cohort_is_still_admitted():
    """A sliding window drops the request out of the fast cohort onto the
    default builder. The binding defers to the registry, which still serves it."""
    req = _unified_request(
        arch="gfx942", d=256, dtype="bf16", left_bound=3, right_bound=0
    )
    assert not _gfx942_d256_fast(req)
    _admitted(req)


# --- shared-gate consistency -------------------------------------------------


def test_candidates_is_empty_exactly_when_admits_is_false():
    cases = [
        _unified_request(d=32),
        replace(_BASE, paged=SdpaPaged(max_seq_len_kv=8)),
        replace(_BASE, stats=packed((2, 4, 8, 1))),
        _unified_request(arch="gfx942"),
        _unified_request(arch="gfx1151", s_q=4, s_kv=8, right_bound=0),
        replace(_BASE, algorithm="unified_2d"),
        _BASE,
    ]
    for req in cases:
        ok, _ = sdpa_unified_admits(req)
        assert bool(sdpa_unified_candidates(req)) == ok


def test_candidates_are_unified_route_candidates_in_priority_order():
    from dispatch.attention import ATTENTION_ROUTE_REGISTRY

    req = _unified_request(arch="gfx942", d=256, dtype="bf16", s_q=1, s_kv=4096)
    names = sdpa_unified_candidates(req)
    by_name = {c.name: c for c in ATTENTION_ROUTE_REGISTRY.candidates()}
    priorities = [by_name[n].priority for n in names]
    assert priorities == sorted(priorities)
    assert all(by_name[n].algorithm != "unified_tuning" for n in names)
