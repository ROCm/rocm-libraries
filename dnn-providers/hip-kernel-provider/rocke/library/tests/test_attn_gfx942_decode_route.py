"""gfx942 all-decode routes to the split-KV 3D kernel past the 2D grid target,
while other arches, windowed decode, short KV and i64-sized caches keep the
shared 2D/3D selector."""

from __future__ import annotations

import pytest
from kernels.common.attention_unified import UnifiedAttentionProblem


def _decode(batch, hq, hkv, skv, **kw):
    return UnifiedAttentionProblem(
        total_q=batch,
        num_seqs=batch,
        num_query_heads=hq,
        num_kv_heads=hkv,
        head_size=kw.pop("head_size", 128),
        block_size=16,
        max_seqlen_q=1,
        max_seqlen_k=skv,
        dtype="bf16",
        num_cus=304,
        **kw,
    )


@pytest.mark.parametrize("batch,skv", [(128, 4096), (1, 512)])
@pytest.mark.parametrize("hq", [32, 64, 128])
def test_gfx942_decode_routes_3d(hq, batch, skv):
    p = _decode(batch, hq, 8, skv)
    assert p.select_path() == "2d"
    assert p.select_path("gfx942") == "3d"
    assert p.select_path("gfx950") == "2d"


@pytest.mark.parametrize(
    "problem",
    [
        _decode(128, 64, 8, 4096, sliding_window=128),
        _decode(512, 64, 8, 4096),
        _decode(128, 64, 8, 4096, head_size=256),
    ],
    ids=["window", "over_3d_kv_limit", "d256"],
)
def test_gfx942_decode_route_scope(problem):
    assert problem.select_path("gfx942") == problem.select_path()


def test_gfx942_prefill_unchanged():
    p = UnifiedAttentionProblem(
        total_q=128 * 4,
        num_seqs=128,
        num_query_heads=64,
        num_kv_heads=8,
        head_size=128,
        block_size=16,
        max_seqlen_q=4,
        max_seqlen_k=4096,
        dtype="bf16",
        num_cus=304,
    )
    assert p.select_path("gfx942") == p.select_path()
