# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx950 3D single-sequence decode carve-out (``_gfx950_3d_tile_size_override``).

Long single-sequence paged decode gets 64-token KV tiles, 64 segments and
swizzled K/V LDS; every other problem keeps the previous dispatch.
CPU only: the GPU numerics of these specs are covered by
``test_tiled_3d_kv_lds_swizzle.py``.
"""

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
    }
    base.update(kw)
    return UnifiedAttentionProblem(**base)  # type: ignore[arg-type]


@pytest.fixture
def gfx950(monkeypatch):
    monkeypatch.setattr(au, "_RESOLVED_ATTENTION_ARCH", "gfx950")


@pytest.mark.parametrize(
    "kw, arch, expected",
    [
        ({}, "gfx950", True),
        ({"max_seqlen_k": 4096}, "gfx950", True),
        ({"dtype": "fp16", "num_kv_heads": 4}, "gfx950", True),
        ({}, "gfx942", False),
        ({"max_seqlen_k": 2048}, "gfx950", False),
        ({"num_seqs": 2, "total_q": 2}, "gfx950", False),
        ({"total_q": 4, "max_seqlen_q": 4}, "gfx950", False),
        ({"head_size": 256}, "gfx950", False),
        ({"block_size": 32}, "gfx950", False),
        ({"sliding_window": 128}, "gfx950", False),
        ({"target_ctas": 8}, "gfx950", False),
        ({"use_fp8": True}, "gfx950", False),
    ],
)
def test_scope(kw, arch, expected):
    tile = au._gfx950_3d_tile_size_override(_problem(**kw), arch=arch)
    assert tile == (64 if expected else None)
    assert au._gfx950_3d_kv_lds_swizzle(_problem(**kw), arch=arch) is expected


def test_spec_and_segments(gfx950):
    problem = _problem()
    spec = bld._tiled_3d_spec_from_problem(problem)
    assert spec.tile_size_override == 64
    assert spec.num_segments == 64
    assert spec.use_kv_lds_swizzle
    assert au._num_segments(problem) == 64


def test_other_problems_keep_the_previous_spec(gfx950):
    spec = bld._tiled_3d_spec_from_problem(_problem(max_seqlen_k=2048))
    assert spec.tile_size_override is None
    assert not spec.use_kv_lds_swizzle


def test_strided_decode_is_not_covered(gfx950, monkeypatch):
    problem = _problem()
    segment, reduce = au._strided_3d_specs_from_problem(problem, arch="gfx950")
    assert not segment.use_kv_lds_swizzle
    assert segment.tile_size_override is None
    monkeypatch.setattr(au, "_gfx950_3d_tile_size_override", lambda *a, **k: None)
    assert segment.num_segments == reduce.num_segments == au._num_segments(problem)


def test_cache_key_follows_the_policy(gfx950):
    assert au._tiled_3d_cache_key(_problem()) != au._tiled_3d_cache_key(
        _problem(max_seqlen_k=2048)
    )
    assert au._tiled_3d_cache_key(_problem())[-2:] == (64, True)
