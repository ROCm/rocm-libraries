"""gfx942/gfx950 all-decode routes to the split-KV 3D kernel where the shared
2D/3D selector picks 2D (large-batch grids, short KV); windowed decode, other
head sizes, prefill, other arches and caches past the gfx942 3D K/V loader keep
the shared selector."""

from __future__ import annotations

import pytest
from kernels.common.attention_unified import UnifiedAttentionProblem

_CUS = {"gfx942": 304, "gfx950": 256}


def _decode(arch, batch, hq, hkv, skv, **kw):
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
        num_cus=_CUS[arch],
        **kw,
    )


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("batch,skv", [(128, 4096), (1, 512)])
@pytest.mark.parametrize("hq", [32, 64, 128])
def test_decode_routes_3d(arch, hq, batch, skv):
    p = _decode(arch, batch, hq, 8, skv)
    assert p.select_path() == "2d"
    assert p.select_path(arch) == "3d"


def test_gfx950_d256_short_kv_routes_3d():
    p = _decode("gfx950", 1, 16, 2, 512, head_size=256)
    assert p.select_path() == "2d"
    assert p.select_path("gfx950") == "3d"


@pytest.mark.parametrize(
    "arch,problem",
    [
        ("gfx942", _decode("gfx942", 128, 64, 8, 4096, sliding_window=128)),
        ("gfx950", _decode("gfx950", 128, 64, 8, 4096, sliding_window=128)),
        ("gfx942", _decode("gfx942", 512, 64, 8, 4096)),
        ("gfx942", _decode("gfx942", 1, 16, 2, 512, head_size=256)),
        ("gfx950", _decode("gfx950", 1, 16, 2, 512, head_size=64)),
        ("gfx1250", _decode("gfx950", 128, 64, 8, 4096)),
    ],
    ids=[
        "gfx942_window",
        "gfx950_window",
        "gfx942_over_3d_kv_limit",
        "gfx942_d256",
        "gfx950_d64",
        "other_arch",
    ],
)
def test_decode_route_scope(arch, problem):
    assert problem.select_path(arch) == problem.select_path()


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_prefill_unchanged(arch):
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
        num_cus=_CUS[arch],
    )
    assert p.select_path(arch) == p.select_path()
