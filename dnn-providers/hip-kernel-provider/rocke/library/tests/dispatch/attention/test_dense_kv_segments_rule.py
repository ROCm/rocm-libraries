# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The dense short-chunk rule: KV-segment count for an underfilled default grid."""

from dataclasses import replace

import pytest

import dispatch.attention as attention
from dispatch.attention.dense_rules import dense_kv_segments, dense_wave_segments


@pytest.mark.parametrize(
    "work,cus,expected",
    [
        (128, 256, 4),
        (64, 256, 8),
        (160, 256, 4),
        (112, 256, 4),
        (256, 256, 1),
        (640, 256, 1),
        (1, 256, 8),
        (0, 256, 1),
    ],
)
def test_dense_kv_segments(work, cus, expected):
    assert dense_kv_segments(work, cus) == expected


def _gfx950_grid_spec(sq, sk, hq, hkv=8, num_cus=256, **over):
    req = attention.AttentionRequest(
        batch=1,
        nhead_q=hq,
        nhead_k=hkv,
        seqlen_q=sq,
        seqlen_k=sk,
        hdim_q=128,
        hdim_v=128,
        arch="gfx950",
        mask_type=2,
        kv_block_size=64,
        dtype="bf16",
        num_cus=num_cus,
        **over,
    )
    cand = next(
        c
        for c in attention.ATTENTION_EXECUTION_REGISTRY.candidates()
        if c.spec_id == "gfx950_dense_grid"
    )
    res = attention.dispatch_attention(
        replace(req, algorithm=cand.algorithm, spec_id=cand.spec_id)
    )
    return getattr(res.spec, "kernel_spec", res.spec)


@pytest.mark.parametrize(
    "sq,sk,hq,block_m,segments",
    [
        (512, 8192, 32, 128, 4),
        (256, 4096, 32, 128, 8),
        (2048, 8192, 32, 256, 1),
        (2048, 2048, 32, 256, 1),
        (2048, 32768, 40, 256, 2),
        (2048, 16384, 64, 256, 1),
    ],
)
def test_gfx950_short_bottom_right_chunk_splits_kv(sq, sk, hq, block_m, segments):
    spec = _gfx950_grid_spec(sq, sk, hq)
    assert (spec.block_m, spec.kv_segments) == (block_m, segments)


def test_gfx950_split_is_off_with_sinks():
    assert _gfx950_grid_spec(512, 8192, 32, use_sinks=True).kv_segments == 1


def _gfx942_spec(sq, sk, hq, hkv=8, mask_type=2, num_cus=304, **over):
    req = attention.AttentionRequest(
        batch=1,
        nhead_q=hq,
        nhead_k=hkv,
        seqlen_q=sq,
        seqlen_k=sk,
        hdim_q=128,
        hdim_v=128,
        arch="gfx942",
        mask_type=mask_type,
        kv_block_size=64,
        dtype="bf16",
        num_cus=num_cus,
        **over,
    )
    return attention.attention_tuning_spec(req, "gfx942_dense").kernel_spec


@pytest.mark.parametrize(
    "sq,sk,hq,block_m,persistent,segments",
    [
        (512, 8192, 32, 256, False, 4),
        (256, 4096, 32, 256, False, 8),
        (2048, 8192, 32, 256, True, 1),
        (2048, 32768, 40, 256, False, 4),
    ],
)
def test_gfx942_bottom_right_geometry(sq, sk, hq, block_m, persistent, segments):
    spec = _gfx942_spec(sq, sk, hq)
    assert spec.causal_bottom_right and spec.causal_diag_split
    assert (spec.block_m, spec.persistent, spec.kv_segments) == (
        block_m,
        persistent,
        segments,
    )


def test_gfx942_equal_length_keeps_default_geometry():
    spec = _gfx942_spec(2048, 2048, 32, mask_type=1)
    assert not spec.causal_bottom_right and not spec.causal_diag_split
    assert spec.kv_segments == 1 and spec.persistent


@pytest.mark.parametrize(
    "work,cus,expected",
    [
        (64, 304, 4),
        (56, 304, 4),
        (32, 304, 8),
        (80, 304, 8),
        (256, 304, 1),
        (320, 304, 4),
        (512, 304, 1),
        (304, 304, 1),
        (0, 304, 1),
    ],
)
def test_dense_wave_segments(work, cus, expected):
    assert dense_wave_segments(work, cus) == expected


# segm_output is [B*Sq, Hq, S, D] FP32 indexed in i32: at B=1, Sq=32768, D=128 the
# extent reaches 2**31 at Hq*S = 512. A CU count large enough to ask for 8 segments
# shows the rule falling back to the largest split that still fits.
@pytest.mark.parametrize("hq,segments", [(32, 8), (64, 4)])
def test_gfx950_kv_split_stays_within_i32_workspace(hq, segments):
    spec = _gfx950_grid_spec(32768, 65536, hq, num_cus=100000)
    assert spec.kv_segments == segments


@pytest.mark.parametrize("hq,segments", [(32, 8), (64, 4)])
def test_gfx942_kv_split_stays_within_i32_workspace(hq, segments):
    spec = _gfx942_spec(32768, 65536, hq, num_cus=100000)
    assert spec.kv_segments == segments


@pytest.mark.parametrize(
    "arch,make", [("gfx950", _gfx950_grid_spec), ("gfx942", _gfx942_spec)]
)
def test_supports_rejects_kv_segments_past_i32_workspace(arch, make):
    import importlib

    supports = importlib.import_module(
        f"kernels.{arch}.attention_dense"
    ).supports_attention_dense
    spec = make(32768, 65536, 64, num_cus=100000)
    assert supports(replace(spec, kv_segments=4))[0]
    ok, why = supports(replace(spec, kv_segments=8))
    assert not ok and "32-bit" in why


def test_max_kv_segments_for_extent():
    from kernels.common.attention_dense_kv_split import max_kv_segments_for_extent

    assert max_kv_segments_for_extent(1, 32768, 64, 128, max_segments=8) == 4
    assert max_kv_segments_for_extent(1, 32768, 32, 128, max_segments=8) == 8
    assert max_kv_segments_for_extent(1, 512, 32, 128, max_segments=8) == 8
    assert max_kv_segments_for_extent(1, 32768, 512, 128) == 1


def _bottom_right_req(arch, sq, sk, mask_type=2):
    return attention.AttentionRequest(
        batch=1,
        nhead_q=32,
        nhead_k=8,
        seqlen_q=sq,
        seqlen_k=sk,
        hdim_q=128,
        hdim_v=128,
        arch=arch,
        mask_type=mask_type,
        kv_block_size=64,
        dtype="bf16",
        num_cus=256 if arch == "gfx950" else 304,
    )


@pytest.mark.parametrize(
    "arch,spec_id,block_m",
    [("gfx950", "gfx950_dense_grid", 128), ("gfx942", "gfx942_dense", 256)],
)
def test_bottom_right_tuning_id_replays_across_chunk_shapes(arch, spec_id, block_m):
    # (512, 8192) and (256, 4096) default to different KV-segment counts; the
    # same recorded knobs must name the same configuration on both.
    knobs = {"block_m": block_m, "kv_segments": 2}
    first = attention.attention_tuning_spec(
        _bottom_right_req(arch, 512, 8192), spec_id, knobs=knobs
    )
    second_req = _bottom_right_req(arch, 256, 4096)
    second = attention.attention_tuning_spec(second_req, spec_id, knobs=knobs)
    assert dict(first.knobs) == dict(second.knobs)
    assert first.tuning_id == second.tuning_id
    replay = attention.attention_tuning_spec(
        second_req, spec_id, tuning_id=first.tuning_id
    )
    assert replay.tuning_id == first.tuning_id
    assert dict(replay.knobs) == dict(first.knobs)


@pytest.mark.parametrize(
    "arch,spec_id",
    [("gfx950", "gfx950_dense_grid"), ("gfx942", "gfx942_dense")],
)
@pytest.mark.parametrize("sq,sk,mask_type", [(2048, 2048, 0), (2048, 2048, 2)])
def test_defaults_fingerprint_unchanged_off_the_moving_diagonal(
    arch, spec_id, sq, sk, mask_type
):
    # Only a moving bottom-right diagonal records extra fields; every other base
    # keeps the fingerprint (and so the stored keys) it had before.
    from dispatch.attention.dense_rules import DenseSpace, _dense_defaults

    base = attention.attention_tuning_spec(
        _bottom_right_req(arch, sq, sk, mask_type), spec_id
    ).kernel_spec
    space = DenseSpace(
        abi="t",
        arch=arch,
        path="dense",
        variant_id="t",
        candidate_name="t",
        recorded_fields=frozenset({"waves_per_eu"}),
    )
    assert space.recorded(base) == space.recorded_fields
    assert space.defaults(base, base) == dict(
        _dense_defaults(base, space.recorded_fields)
    )
