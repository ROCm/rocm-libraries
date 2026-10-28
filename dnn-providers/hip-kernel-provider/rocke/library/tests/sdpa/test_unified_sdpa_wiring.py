# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``dispatch.sdpa.bind_unified``: routing onto the unified attention candidates.

Every case uses a real ``SdpaRequest`` built from ``request_cases.py``.
Decline-reason assertions check the feature name the caller can act on.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from dispatch.sdpa.bind_unified import (
    sdpa_unified_admits,
    sdpa_unified_candidates,
)
from dispatch.sdpa.common import SdpaBias, SdpaDropout, SdpaFp8, SdpaPaged
from tests.sdpa.request_cases import make_request, packed

UNIFIED_HEAD_SIZES = (64, 128, 256)
ARCHES = ("gfx942", "gfx950")


def _unified_request(arch="gfx950", d=128, dtype="fp16", **overrides):
    return make_request(arch=arch, d=d, dtype=dtype, **overrides)


# --- head_dim / arch validity --------------------------------------------------


def test_unsupported_head_dim_reports_head_dim_before_a_feature_gap():
    req = replace(_unified_request(d=32), paged=SdpaPaged(max_seq_len_kv=8))
    ok, reason = sdpa_unified_admits(req)
    assert not ok
    assert "head_dim" in reason
    assert sdpa_unified_candidates(req) == ()


def test_unknown_arch_declines():
    req = _unified_request(arch="gfx000")
    ok, reason = sdpa_unified_admits(req)
    assert not ok
    assert "arch" in reason


# --- d=256 -------------------------------------------------------------------


@pytest.mark.parametrize("arch", ARCHES)
def test_d256_decode_reaches_the_dedicated_decode_candidate(arch):
    """``attention_d256_decode`` serves gfx942 decode as well as gfx950."""
    req = _unified_request(arch=arch, d=256, dtype="bf16", s_q=1, s_kv=4096)
    ok, _ = sdpa_unified_admits(req)
    assert ok
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
    req = _unified_request(arch="gfx942", d=256, dtype="bf16")
    assert sdpa_unified_admits(req)[0]
    assert _gfx942_d256_fast(req)


def test_gfx942_d256_outside_the_fast_cohort_is_still_admitted():
    """Sinks drop the request out of the fast cohort onto the default builder.
    The binding defers to the registry, which still serves it."""
    req = replace(
        _unified_request(arch="gfx942", d=256, dtype="bf16"),
        sink_token=packed((1, 4, 1, 1)),
    )
    assert not _gfx942_d256_fast(req)
    ok, reason = sdpa_unified_admits(req)
    assert ok, reason


# --- admitted features ---------------------------------------------------------


@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize("d", UNIFIED_HEAD_SIZES)
@pytest.mark.parametrize("dtype", ("fp16", "bf16"))
def test_plain_dense_request_is_admitted(arch, d, dtype):
    ok, reason = sdpa_unified_admits(_unified_request(arch=arch, d=d, dtype=dtype))
    assert ok, reason


def test_sinks_is_admitted():
    req = replace(_unified_request(), sink_token=packed((1, 4, 1, 1)))
    ok, reason = sdpa_unified_admits(req)
    assert ok, reason


def test_causal_top_left_is_admitted():
    req = replace(_unified_request(), right_bound=0)
    ok, reason = sdpa_unified_admits(req)
    assert ok, reason


def test_causal_bottom_right_is_admitted():
    req = _unified_request(s_q=4, s_kv=8, right_bound=0, diagonal="bottom_right")
    ok, reason = sdpa_unified_admits(req)
    assert ok, reason


def test_sliding_window_is_admitted():
    req = replace(_unified_request(), left_bound=3, right_bound=0)
    ok, reason = sdpa_unified_admits(req)
    assert ok, reason


def test_lse_is_admitted_on_a_decode_shape():
    """AttentionRequest.emit_lse docstring: only the split-KV (3D/decode)
    path admits lse; the 2D prefill path refuses it."""
    req = replace(_unified_request(s_q=1, s_kv=4096), stats=packed((2, 4, 1, 1)))
    ok, reason = sdpa_unified_admits(req)
    assert ok, reason


def test_lse_is_declined_on_a_prefill_shape():
    req = replace(_unified_request(), stats=packed((2, 4, 8, 1)))
    ok, _ = sdpa_unified_admits(req)
    assert not ok


# --- declines for SdpaRequest features the unified Torch binding cannot carry -


def test_kv_heads_differ_declines():
    req = _unified_request(h_q=8, h_k=2, h_v=4)
    ok, reason = sdpa_unified_admits(req)
    assert not ok
    assert "kv_heads_differ" in reason


def test_paged_declines():
    req = replace(_unified_request(), paged=SdpaPaged(max_seq_len_kv=8))
    ok, reason = sdpa_unified_admits(req)
    assert not ok
    assert "paged" in reason


def test_dropout_declines():
    req = replace(_unified_request(), dropout=SdpaDropout(0.1))
    ok, reason = sdpa_unified_admits(req)
    assert not ok
    assert "dropout" in reason


def test_fp8_declines():
    from tests.sdpa.request_cases import bshd

    req = _unified_request(
        dtype="fp8_e4m3",
        o=bshd(2, 4, 8, 128, "bf16"),
        fp8=SdpaFp8(descale_q=packed((1,))),
    )
    ok, reason = sdpa_unified_admits(req)
    assert not ok
    assert "fp8" in reason


def test_block_mask_declines():
    req = replace(_unified_request(), block_mask=packed((2, 4, 1, 1), "uint8"))
    ok, reason = sdpa_unified_admits(req)
    assert not ok
    assert "block_mask" in reason


def test_scale_tensor_declines():
    req = replace(_unified_request(), scale_source="tensor", scale_tensor=packed((1,)))
    ok, reason = sdpa_unified_admits(req)
    assert not ok
    assert "scale_tensor" in reason


def test_bias_declines():
    req = replace(_unified_request(), bias=SdpaBias(packed((2, 4, 8, 8))))
    ok, reason = sdpa_unified_admits(req)
    assert not ok
    assert "bias" in reason


def test_alibi_declines():
    """No unified candidate declares the "alibi" Capability feature today
    (grepped every dispatch/attention/*.py Capability declaration) -- this is
    a real, currently-unimplemented gap, not a translation bug."""
    req = replace(_unified_request(), alibi=True)
    ok, reason = sdpa_unified_admits(req)
    assert not ok
    assert "alibi" in reason


# --- shared-gate consistency: admits() and candidates() cannot drift ----------


def test_candidates_is_empty_exactly_when_admits_is_false():
    cases = [
        _unified_request(d=32),  # invalid head_dim
        replace(_unified_request(), paged=SdpaPaged(max_seq_len_kv=8)),  # feature gap
        replace(_unified_request(), stats=packed((2, 4, 8, 1))),  # registry refuses
        _unified_request(),  # admitted
    ]
    for req in cases:
        ok, _ = sdpa_unified_admits(req)
        names = sdpa_unified_candidates(req)
        assert bool(names) == ok


def test_candidates_are_in_registry_priority_order():
    from dispatch.attention import ATTENTION_ROUTE_REGISTRY

    req = _unified_request(arch="gfx942", d=256, dtype="bf16", s_q=1, s_kv=4096)
    names = sdpa_unified_candidates(req)
    by_priority = {c.name: c.priority for c in ATTENTION_ROUTE_REGISTRY.candidates()}
    priorities = [by_priority[n] for n in names]
    assert priorities == sorted(priorities)
