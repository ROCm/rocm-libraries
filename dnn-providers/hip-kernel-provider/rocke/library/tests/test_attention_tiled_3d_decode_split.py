# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx950 and gfx942 3D decode split rules: segment count and KV tile."""

from __future__ import annotations

import pytest

from builders.common import attention_spec_builder as bld
from kernels.common import attention_unified as au
from kernels.common.attention_unified import UnifiedAttentionProblem


def _problem(**kw) -> UnifiedAttentionProblem:
    base: dict[str, object] = {
        "total_q": 1,
        "num_seqs": 1,
        "num_query_heads": 64,
        "num_kv_heads": 8,
        "head_size": 128,
        "block_size": 16,
        "max_seqlen_q": 1,
        "max_seqlen_k": 32768,
        "dtype": "bf16",
        "num_cus": 256,
    }
    base.update(kw)
    if "num_seqs" in kw and "total_q" not in kw:
        base["total_q"] = base["num_seqs"]
    return UnifiedAttentionProblem(**base)  # type: ignore[arg-type]


@pytest.fixture
def gfx950(monkeypatch):
    monkeypatch.setattr(au, "_RESOLVED_ATTENTION_ARCH", "gfx950")


@pytest.mark.parametrize(
    "kw, arch, in_scope",
    [
        ({}, "gfx950", True),
        ({"max_seqlen_k": 4096}, "gfx950", True),
        ({"dtype": "fp16", "num_kv_heads": 4}, "gfx950", True),
        ({"num_seqs": 64}, "gfx950", True),
        ({"max_seqlen_k": 1024}, "gfx950", True),
        ({"head_size": 256, "num_seqs": 4}, "gfx950", True),
        ({}, "gfx942", False),
        ({"total_q": 4, "max_seqlen_q": 4}, "gfx950", False),
        ({"head_size": 256}, "gfx950", False),
        ({"head_size": 64}, "gfx950", False),
        ({"block_size": 32}, "gfx950", False),
        ({"sliding_window": 128}, "gfx950", False),
        ({"target_ctas": 8}, "gfx950", False),
        ({"use_fp8": True}, "gfx950", False),
    ],
)
def test_scope(kw, arch, in_scope):
    p = _problem(**kw)
    ctas = au._gfx950_3d_decode_ctas_per_segment(p, arch=arch)
    assert ctas == (p.num_seqs * p.num_kv_heads if in_scope else None)


@pytest.mark.parametrize(
    "num_seqs, num_kv_heads, max_seqlen_k, num_cus, tile, segments",
    [
        # Underfilled (CTAs per segment <= num_cus / 32): 64-token tiles, at
        # most one tile per segment.
        (1, 8, 32768, 256, 64, 64),
        (1, 4, 65536, 256, 64, 128),
        (2, 4, 65536, 256, 64, 64),
        (8, 1, 16384, 256, 64, 64),
        (1, 8, 2048, 256, 64, 32),
        (1, 1, 1024, 256, 64, 16),
        # About 4 CTAs per CU, at most 64 segments below 8K KV tokens and 128
        # from there, at least 2.
        (2, 8, 16384, 256, None, 64),
        (4, 8, 16384, 256, None, 32),
        (16, 8, 8192, 256, None, 8),
        (16, 8, 1024, 256, None, 8),
        (32, 4, 8192, 256, None, 8),
        (64, 4, 4096, 256, None, 4),
        (128, 4, 8192, 256, None, 2),
        (256, 16, 8192, 256, None, 2),
        (3, 4, 16384, 256, None, 85),
        (3, 4, 4096, 256, None, 64),
        # The boundary follows the CU count.
        (1, 8, 16384, 128, None, 64),
        (1, 4, 16384, 128, 64, 64),
        (1, 8, 16384, 120, None, 60),
    ],
)
def test_split(gfx950, num_seqs, num_kv_heads, max_seqlen_k, num_cus, tile, segments):
    p = _problem(
        num_seqs=num_seqs,
        num_kv_heads=num_kv_heads,
        num_query_heads=8 * num_kv_heads,
        max_seqlen_k=max_seqlen_k,
        num_cus=num_cus,
    )
    assert au._gfx950_3d_tile_size_override(p, "gfx950") == tile
    assert au._kv_lds_swizzle_3d(p, arch="gfx950") is (tile is not None)
    assert au._num_segments(p, "gfx950") == segments
    spec = bld._tiled_3d_spec_from_problem(p, "gfx950")
    assert spec.tile_size_override == tile
    assert spec.num_segments == segments
    assert spec.use_kv_lds_swizzle is (tile is not None)
    assert spec.use_wave_page_id_load


def test_out_of_scope_keeps_the_previous_spec(gfx950):
    spec = bld._tiled_3d_spec_from_problem(_problem(sliding_window=128), "gfx950")
    assert spec.tile_size_override is None
    assert not spec.use_kv_lds_swizzle
    assert not spec.use_wave_page_id_load


@pytest.mark.parametrize("num_seqs, tile, segments", [(1, 64, 64), (16, None, 8)])
def test_strided_decode_follows_the_rule(gfx950, num_seqs, tile, segments):
    problem = _problem(num_seqs=num_seqs)
    segment, reduce = au._strided_3d_specs_from_problem(problem, arch="gfx950")
    assert segment.tile_size_override == tile
    assert segment.num_segments == reduce.num_segments == segments
    # The swizzle measured slower on strided K/V; page ids need a page table.
    assert not segment.use_kv_lds_swizzle
    assert not segment.use_wave_page_id_load


def test_cache_key_follows_the_rule(gfx950):
    assert au._tiled_3d_cache_key(_problem(), "gfx950") != au._tiled_3d_cache_key(
        _problem(sliding_window=128), "gfx950"
    )
    assert au._tiled_3d_cache_key(_problem(), "gfx950")[-3:] == (64, True, True)


@pytest.fixture
def gfx942(monkeypatch):
    monkeypatch.setattr(au, "_RESOLVED_ATTENTION_ARCH", "gfx942")


@pytest.mark.parametrize(
    "kw, in_scope",
    [
        ({"num_seqs": 4}, True),
        ({"num_seqs": 4, "head_size": 64}, True),
        ({"max_seqlen_k": 4096}, True),
        ({}, False),
        ({"num_seqs": 4, "head_size": 256}, False),
        ({"num_seqs": 4, "block_size": 32}, False),
        ({"num_seqs": 4, "sliding_window": 128}, False),
        ({"num_seqs": 4, "target_ctas": 8}, False),
        ({"num_seqs": 4, "use_fp8": True}, False),
        ({"num_seqs": 4, "total_q": 8, "max_seqlen_q": 2}, False),
    ],
)
def test_gfx942_scope(kw, in_scope):
    p = _problem(num_cus=304, **kw)
    ctas = au._gfx942_3d_decode_ctas_per_segment(p, arch="gfx942")
    assert ctas == (p.num_seqs * p.num_kv_heads if in_scope else None)


@pytest.mark.parametrize(
    "num_seqs, num_kv_heads, head_size, max_seqlen_k, segments",
    [
        # Exactly up to 4 CTAs per CU (6 at head size 64) on 304 CUs.
        (4, 8, 128, 16384, 38),
        (8, 8, 128, 16384, 19),
        (16, 8, 128, 16384, 9),
        (32, 8, 128, 4096, 4),
        (64, 8, 128, 4096, 2),
        (16, 4, 64, 16384, 28),
        # At most 64 segments below 8K KV tokens, 128 from there.
        (1, 8, 128, 2048, 64),
        (2, 8, 128, 4096, 64),
        (5, 4, 64, 16384, 91),
    ],
)
def test_gfx942_split(
    gfx942, num_seqs, num_kv_heads, head_size, max_seqlen_k, segments
):
    p = _problem(
        num_seqs=num_seqs,
        num_kv_heads=num_kv_heads,
        num_query_heads=8 * num_kv_heads,
        head_size=head_size,
        max_seqlen_k=max_seqlen_k,
        num_cus=304,
    )
    assert au._num_segments(p, "gfx942") == segments
    spec = bld._tiled_3d_spec_from_problem(p, "gfx942")
    assert spec.num_segments == segments
    assert spec.use_kv_lds_swizzle and spec.use_wave_page_id_load
    segment, reduce = au._strided_3d_specs_from_problem(p, arch="gfx942")
    assert segment.num_segments == reduce.num_segments == segments


def test_gfx942_few_ctas_at_long_kv_keep_the_previous_count(gfx942, monkeypatch):
    p = _problem(num_seqs=1, num_kv_heads=8, max_seqlen_k=32768, num_cus=304)
    assert au._gfx942_3d_decode_ctas_per_segment(p, "gfx942") is None
    monkeypatch.setattr(au, "_gfx942_3d_decode_ctas_per_segment", lambda *a, **k: None)
    assert au._num_segments(p, "gfx942") == 128
