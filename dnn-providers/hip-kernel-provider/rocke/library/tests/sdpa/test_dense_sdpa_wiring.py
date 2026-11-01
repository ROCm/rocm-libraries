# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``dispatch.sdpa.bind_dense``: routing onto the dense attention candidates.

Every case uses a real ``SdpaRequest`` built from ``request_cases.py``, at
lengths the dense kernels take without an on-chip ragged path (S a multiple
of 256). Decline-reason assertions check the feature name the caller can act
on.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from dispatch.sdpa.bind_dense import (
    _DECLINED,
    SDPA_DENSE_FEATURES,
    sdpa_dense_admits,
    sdpa_dense_candidates,
)
from dispatch.sdpa.common import (
    SDPA_FEATURES,
    SdpaBias,
    SdpaDropout,
    SdpaFp8,
    SdpaPaged,
)
from tests.sdpa.request_cases import bhsd, bshd, make_request, packed, ragged_offset

ARCHES = ("gfx942", "gfx950")
S = 256


def _dense_request(arch="gfx950", d=128, dtype="fp16", s_q=S, s_kv=S, **overrides):
    return make_request(arch=arch, d=d, dtype=dtype, s_q=s_q, s_kv=s_kv, **overrides)


def _ragged(t):
    return replace(
        t, ragged_offset=ragged_offset(t.dims[0]), ragged_offset_multiplier=t.strides[2]
    )


def _declined(req):
    ok, reason = sdpa_dense_admits(req)
    assert not ok
    assert sdpa_dense_candidates(req) == ()
    return reason


def _admitted(req):
    ok, reason = sdpa_dense_admits(req)
    assert ok, reason
    names = sdpa_dense_candidates(req)
    assert names
    return names


# --- every feature is carried or declined ------------------------------------


def test_every_sdpa_feature_is_classified_exactly_once():
    declined = frozenset(_DECLINED)
    assert SDPA_DENSE_FEATURES | declined == SDPA_FEATURES
    assert not SDPA_DENSE_FEATURES & declined


def test_decline_reasons_start_with_their_feature():
    for feature, reason in _DECLINED.items():
        assert reason.startswith(f"{feature}: "), feature


_BASE = _dense_request(right_bound=0)
_LENS = packed((2, 1, 1, 1), "int32")
_DECLINED_CASES = {
    "kv_heads_differ": _dense_request(h_q=8, h_k=2, h_v=4),
    "d_v_differs": _dense_request(d_v=64),
    "dtype_mixed": replace(_BASE, o=bshd(2, 4, S, 128, "fp32")),
    "compute_dtype": replace(_BASE, compute_dtype="fp16"),
    "scale_tensor": replace(_BASE, scale_source="tensor", scale_tensor=packed((1,))),
    "bias": replace(_BASE, bias=SdpaBias(packed((2, 4, S, S)))),
    "bias_broadcast": replace(_BASE, bias=SdpaBias(packed((S, S)))),
    "alibi": replace(_BASE, alibi=True),
    "sinks": replace(_BASE, sink_token=packed((1, 4, 1, 1))),
    "padded": replace(_BASE, padding_mask=True, seq_len_q=_LENS, seq_len_kv=_LENS),
    "ragged": replace(
        _BASE,
        q=_ragged(_BASE.q),
        k=_ragged(_BASE.k),
        v=_ragged(_BASE.v),
        o=_ragged(_BASE.o),
    ),
    "dropout": replace(_BASE, dropout=SdpaDropout(0.1)),
    "paged": replace(_BASE, paged=SdpaPaged(max_seq_len_kv=8)),
    "block_mask": replace(_BASE, block_mask=packed((2, 4, 1, 1), "uint8")),
    "fp8": _dense_request(dtype="fp8_e4m3", fp8=SdpaFp8(descale_q=packed((1,)))),
    "fp8_s_o": _dense_request(
        dtype="fp8_e4m3", fp8=SdpaFp8(descale_q=packed((1,)), scale_o=packed((1,)))
    ),
    "max_sum_exp": replace(
        _BASE, max=packed((2, 4, S, 1)), sum_exp=packed((2, 4, S, 1))
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


# --- arch ---------------------------------------------------------------------


def test_unknown_arch_declines():
    assert "arch" in _declined(_dense_request(arch="gfx000"))


def test_arch_without_a_dense_candidate_declines():
    assert "no dense candidate for arch" in _declined(_dense_request(arch="gfx1151"))


# --- value checks on carried hints -------------------------------------------


def test_mma_core_mode_matching_the_dtype_is_admitted():
    _admitted(replace(_BASE, mma_core_mode="fp16"))


def test_mma_core_mode_other_than_the_dtype_is_declined():
    assert _declined(replace(_BASE, mma_core_mode="fp32")).startswith("mma_core_mode:")


def test_implementation_composite_is_declined():
    assert _declined(replace(_BASE, implementation="composite")).startswith(
        "implementation_hint:"
    )


@pytest.mark.parametrize(
    "pin",
    [
        {"algorithm": "attention_dense_grid"},
        {"spec_id": "gfx950_dense_grid"},
        {"tuning_id": "t0"},
        {"tuning_knobs": (("block_m", 128),)},
    ],
    ids=["algorithm", "spec_id", "tuning_id", "tuning_knobs"],
)
def test_dispatch_pin_is_declined(pin):
    """The binding picks the dense variant itself, so a pin would be ignored."""
    assert _declined(replace(_BASE, **pin)).startswith("dispatch_pin:")


@pytest.mark.parametrize("scale", [2.0**5, 2.0**-70])
def test_gfx950_scale_outside_the_kernel_range_is_declined(scale):
    """The gfx950 dense kernel takes a scale in [2^-64, 2^4]; an
    ``AttentionRequest`` cannot carry the scale, so the gate checks it."""
    assert _declined(replace(_BASE, scale=scale)).startswith("scale:")


def test_non_finite_scale_is_declined():
    assert _declined(replace(_BASE, scale=float("nan"))).startswith("scale:")


# --- carried features --------------------------------------------------------


@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize("d", (64, 128))
@pytest.mark.parametrize("dtype", ("fp16", "bf16"))
def test_causal_top_left_is_admitted(arch, d, dtype):
    _admitted(_dense_request(arch=arch, d=d, dtype=dtype, right_bound=0))


@pytest.mark.parametrize("arch", ARCHES)
def test_full_attention_is_admitted_at_equal_lengths(arch):
    _admitted(_dense_request(arch=arch))


@pytest.mark.parametrize("arch", ARCHES)
def test_top_left_window_is_admitted(arch):
    _admitted(_dense_request(arch=arch, left_bound=63, right_bound=0))


@pytest.mark.parametrize("arch", ARCHES)
def test_lse_is_admitted(arch):
    _admitted(replace(_BASE, arch=arch, stats=packed((2, 4, S, 1))))


@pytest.mark.parametrize(
    "req",
    [
        _dense_request(right_bound=0, layout=bhsd),
        replace(_BASE, k=bhsd(2, 4, S, 128)),
        replace(_BASE, o=bhsd(2, 4, S, 128)),
        replace(
            _BASE,
            q=replace(
                _BASE.q, strides=(_BASE.q.strides[0] + 64,) + _BASE.q.strides[1:]
            ),
        ),
        replace(_BASE, q=replace(_BASE.q, alignment=4)),
        _dense_request(h_q=8, h_k=2, right_bound=0),
    ],
    ids=[
        "bhsd",
        "kv_layout_differs",
        "o_layout_differs",
        "batch_stride",
        "unaligned",
        "gqa",
    ],
)
def test_layouts_and_gqa_are_admitted(req):
    _admitted(req)


# --- band ----------------------------------------------------------------------


def test_gfx950_bottom_right_causal_is_admitted_on_the_grid_body():
    """Only the grid body runs a moving bottom-right diagonal."""
    req = _dense_request(s_q=S, s_kv=2 * S, right_bound=0, diagonal="bottom_right")
    names = _admitted(req)
    assert all("persist" not in n for n in names)


def test_gfx942_bottom_right_causal_is_admitted():
    """gfx942's dense candidate now serves a moving bottom-right diagonal (#14367)."""
    req = _dense_request(
        arch="gfx942", s_q=S, s_kv=2 * S, right_bound=0, diagonal="bottom_right"
    )
    assert _admitted(req) == ("attention_gfx942_dense",)


def test_gfx950_bottom_right_window_with_unequal_lengths_is_declined():
    """The dense body anchors windows top-left."""
    req = _dense_request(
        s_q=S, s_kv=2 * S, left_bound=63, right_bound=0, diagonal="bottom_right"
    )
    _declined(req)


def test_gfx950_lookahead_band_is_admitted():
    _admitted(_dense_request(left_bound=63, right_bound=2))


# --- the candidates' own rules decide ----------------------------------------


def test_gfx942_non_tile_multiple_length_is_declined_with_the_kernels_reason():
    """No length rule here: the reason is the gfx942 kernel's own."""
    reason = _declined(_dense_request(arch="gfx942", s_q=200, s_kv=200, right_bound=0))
    assert "attention_gfx942_dense" in reason


def test_unsupported_head_size_is_declined_with_the_kernels_reason():
    reason = _declined(_dense_request(d=32, right_bound=0))
    assert "head_size" in reason


# --- choice and consistency --------------------------------------------------


def test_gfx950_launches_the_first_variant_in_registry_order():
    names = _admitted(_BASE)
    assert names[0] == "attention_gfx950_dense_grid"


def test_gfx942_launches_its_dense_candidate():
    assert _admitted(replace(_BASE, arch="gfx942")) == ("attention_gfx942_dense",)


def test_candidates_is_empty_exactly_when_admits_is_false():
    cases = [
        _dense_request(d=32),
        replace(_BASE, paged=SdpaPaged(max_seq_len_kv=8)),
        _dense_request(arch="gfx942", s_q=200, s_kv=200),
        replace(_BASE, algorithm="attention_dense_grid"),
        _BASE,
        replace(_BASE, arch="gfx942"),
    ]
    for req in cases:
        ok, _ = sdpa_dense_admits(req)
        assert bool(sdpa_dense_candidates(req)) == ok
