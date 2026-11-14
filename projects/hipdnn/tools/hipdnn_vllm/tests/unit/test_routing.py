# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

import dataclasses

import pytest

from hipdnn_vllm.attention.metadata import BatchClass, classify, encoder_groups
from hipdnn_vllm.attention.routing import P1E, P2A, P2B, LayerStatic, select
from hipdnn_vllm.declines import Stage

BASE = LayerStatic(
    num_heads=32,
    num_kv_heads=8,
    head_size=128,
    scale=128**-0.5,
    sliding_window=(-1, -1),
    logits_soft_cap=0.0,
    has_alibi=False,
    has_sinks=False,
    kv_cache_dtype="auto",
    attn_type="decoder",
)


@pytest.mark.parametrize(
    "args,expected",
    [
        ((8, 8, 1, 300, 1), BatchClass.UNIFORM_DECODE),
        ((8, 32, 4, 300, 4), BatchClass.UNIFORM_DECODE),  # spec decode k=3
        ((8, 30, 4, 300, 4), BatchClass.MIXED),  # non-uniform q_len
        ((1, 128, 128, 128, 1), BatchClass.PREFILL_NO_CONTEXT),
        ((1, 64, 64, 200, 1), BatchClass.MIXED),  # chunked prefill with context
        ((4, 512, 128, 128, 1), BatchClass.MIXED),  # several prompts: not provable from host scalars
        ((3, 130, 128, 900, 1), BatchClass.MIXED),
        ((0, 0, 0, 0, 1), BatchClass.MIXED),
    ],
)
def test_classify(args, expected):
    assert classify(*args) is expected


@pytest.mark.parametrize(
    "bc,path,capturing",
    [
        (BatchClass.UNIFORM_DECODE, P2A, False),
        (BatchClass.UNIFORM_DECODE, P2A, True),
        (BatchClass.MIXED, P2B, False),
        (BatchClass.MIXED, P2B, True),
        (BatchClass.PREFILL_NO_CONTEXT, P2B, False),
    ],
)
def test_decoder_routes_to_paged_paths(bc, path, capturing):
    r = select(BASE, bc, attention_enabled=True, capturing=capturing)
    assert r.path == path and r.decline is None


@pytest.mark.parametrize(
    "bc,reason",
    [(None, "UNSUPPORTED_FEATURE:metadata"), (BatchClass.ENCODER_ONLY, "UNSUPPORTED_FEATURE:non_causal")],
)
def test_decoder_without_paged_metadata_falls_back(bc, reason):
    r = select(BASE, bc, attention_enabled=True)
    assert r.path is None and r.reason == reason


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"logits_soft_cap": 30.0}, "UNSUPPORTED_FEATURE:softcap"),
        ({"has_alibi": True}, "UNSUPPORTED_FEATURE:alibi"),
        ({"kv_cache_dtype": "fp8"}, "UNSUPPORTED_FEATURE:fp8_kv"),
        ({"kv_cache_dtype": "fp8_e4m3"}, "UNSUPPORTED_FEATURE:fp8_kv"),
        ({"kv_cache_dtype": "int8_per_token_head"}, "UNSUPPORTED_FEATURE:quant_kv"),
        ({"attn_type": "encoder_decoder"}, "UNSUPPORTED_FEATURE:encoder_decoder"),
        ({"attn_type": "encoder_only"}, "UNSUPPORTED_FEATURE:encoder_only_metadata"),
        ({"chunk_lookback": 2}, "UNSUPPORTED_FEATURE:chunk_lookback"),
    ],
)
def test_prefilter_reasons(change, reason):
    r = select(dataclasses.replace(BASE, **change), BatchClass.MIXED, attention_enabled=True)
    assert r.path is None and r.reason == reason


@pytest.mark.parametrize(
    "bc,path", [(BatchClass.UNIFORM_DECODE, P2A), (BatchClass.MIXED, P2B), (BatchClass.PREFILL_NO_CONTEXT, P2B)]
)
def test_sinks_route_to_paged_paths(bc, path):
    """Sinks are not pre-filtered: the graph carries the sink_token and the
    per-layer probe decides."""
    r = select(dataclasses.replace(BASE, has_sinks=True), bc, attention_enabled=True)
    assert r.path == path and r.decline is None


def test_disabled_wins():
    r = select(dataclasses.replace(BASE, has_sinks=True), BatchClass.MIXED, attention_enabled=False)
    assert r.decline.stage is Stage.DISABLED


@pytest.mark.parametrize(
    "bc", [BatchClass.UNIFORM_DECODE, BatchClass.MIXED, BatchClass.PREFILL_NO_CONTEXT, BatchClass.ENCODER_ONLY, None]
)
@pytest.mark.parametrize(
    "change",
    [{}, {"has_sinks": True}, {"logits_soft_cap": 30.0}, {"kv_cache_dtype": "fp8"}, {"chunk_lookback": 2}],
)
def test_decoder_route_is_capture_invariant(bc, change):
    """A decoder layer gets the same route (path and reason) eager and under
    capture, so a captured graph replays the route its eager warm-up took."""
    layer = dataclasses.replace(BASE, **change)
    eager = select(layer, bc, attention_enabled=True, capturing=False)
    captured = select(layer, bc, attention_enabled=True, capturing=True)
    assert (captured.path, captured.reason) == (eager.path, eager.reason)


ENCODER = dataclasses.replace(BASE, num_kv_heads=32, attn_type="encoder_only")


def test_classify_encoder_only_wins():
    assert classify(4, 512, 128, 128, 1, encoder_only=True) is BatchClass.ENCODER_ONLY
    assert classify(4, 4, 1, 1, 1, encoder_only=True) is BatchClass.ENCODER_ONLY


def test_encoder_only_routes_to_p1e_when_eager():
    r = select(ENCODER, BatchClass.ENCODER_ONLY, attention_enabled=True)
    assert r.path == P1E and r.decline is None


def test_encoder_only_falls_back_under_capture():
    r = select(ENCODER, BatchClass.ENCODER_ONLY, attention_enabled=True, capturing=True)
    assert r.path is None
    assert r.reason == "CAPTURE_INVARIANT:p1e_eager_only"


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"logits_soft_cap": 30.0}, "UNSUPPORTED_FEATURE:softcap"),
        ({"has_alibi": True}, "UNSUPPORTED_FEATURE:alibi"),
    ],
)
def test_encoder_only_prefilter(change, reason):
    r = select(dataclasses.replace(ENCODER, **change), BatchClass.ENCODER_ONLY, True)
    assert r.path is None and r.reason == reason


def test_encoder_only_sinks_route_to_p1e():
    r = select(dataclasses.replace(ENCODER, has_sinks=True), BatchClass.ENCODER_ONLY, attention_enabled=True)
    assert r.path == P1E and r.decline is None


def test_encoder_only_disabled():
    r = select(ENCODER, BatchClass.ENCODER_ONLY, attention_enabled=False)
    assert r.path is None and r.decline.stage is Stage.DISABLED


@pytest.mark.parametrize(
    "starts,groups",
    [
        ([0], ()),
        ([0, 7], ((0, 1, 7),)),
        ([0, 4, 8, 12], ((0, 3, 4),)),
        ([0, 4, 8, 11, 14, 20], ((0, 2, 4), (8, 2, 3), (14, 1, 6))),
        ([0, 5, 5, 10], ((0, 2, 5),)),  # an empty request between equal runs
        ([0, 2, 5, 7], ((0, 1, 2), (2, 1, 3), (5, 1, 2))),
    ],
)
def test_encoder_groups(starts, groups):
    assert encoder_groups(starts) == groups
