# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""One gfx950 tiled binary serves every head pair, window size and batch.

The gfx950 tiled 2D and 3D kernels read the head counts and the sliding window
as kernargs and derive the cu_q search trip count from ``num_seqs``, so shapes
that differ only in those share one cache key and one compiled binary. Each
case below runs a pair of such shapes back to back through
``run_unified_attention_torch``: the second launch must reuse the first binary
(no new compile) and both must match an FP32 paged reference.

The second shape of each pair changes the head counts at the same GQA ratio,
the window size, or the batch by several log2 search buckets, so a value that
is still baked into the binary corrupts it. Block-table rows are wider than the
batch needs and unused slots point at a poison block, so a wrong stride cannot
pass by luck.

The fast paged-KV descriptor bakes the 64/8 head geometry, so its pair keeps
64/8 and changes the batch; its 32/4 and 16/2 neighbours at the same ratio take
the plain combo binary.

The routing and key checks run on any host; the launches need a gfx950 GPU.
"""

from __future__ import annotations

import itertools
import math

import pytest

from kernels.common import attention_unified as au

_ARCH = "gfx950"
_CUS = 256


def _problem(*, q_lens, kv_lens, hq, hkv, d, bs, dtype, window, num_blocks):
    return au.UnifiedAttentionProblem(
        total_q=sum(q_lens),
        num_seqs=len(q_lens),
        num_query_heads=hq,
        num_kv_heads=hkv,
        head_size=d,
        block_size=bs,
        max_seqlen_q=max(q_lens),
        max_seqlen_k=max(kv_lens),
        dtype=dtype,
        sliding_window=window,
        num_cus=_CUS,
        num_kv_blocks=num_blocks,
    )


def _shape(q_lens, kv_lens, hq, hkv, window):
    return dict(q_lens=q_lens, kv_lens=kv_lens, hq=hq, hkv=hkv, window=window)


# (id, path, route predicate or None, (d, bs, dtype), first shape, second shape).
# The route predicate is an attention_unified selector both shapes must satisfy.
# Both shapes of a pair sit on one side of every seqlen threshold the selectors
# read (combo above 256 query tokens, hipcc above 512), so only the head counts,
# the window and the batch differ.
_CASES = (
    (
        "2d-bf16-d128",
        "2d",
        None,
        (128, 16, "bf16"),
        _shape([300, 200, 250], [300, 450, 250], 32, 8, 0),
        _shape(
            [90, 130, 70, 110, 40, 300, 60, 80, 120],
            [90, 300, 70, 210, 40, 300, 60, 380, 120],
            16,
            4,
            0,
        ),
    ),
    # Head counts alone: the token and sequence counts match, so only the head
    # counts tell the two launches apart.
    (
        "2d-bf16-d128-single-batch-heads-only",
        "2d",
        "_enable_single_batch_combo",
        (128, 16, "bf16"),
        _shape([1025], [1025], 32, 8, 0),
        _shape([1025], [1025], 64, 16, 0),
    ),
    (
        "2d-fp16-d64-window",
        "2d",
        None,
        (64, 32, "fp16"),
        _shape([260, 180], [400, 180], 8, 8, 128),
        _shape(
            [70, 90, 50, 100, 60, 80, 40, 30, 110],
            [200, 90, 300, 100, 60, 250, 40, 30, 110],
            16,
            16,
            512,
        ),
    ),
    # The single-batch fp16 d128 window cohort: heads and window change.
    (
        "2d-fp16-d128-single-batch-window",
        "2d",
        "_enable_single_batch_combo",
        (128, 16, "fp16"),
        _shape([2048], [2048], 32, 8, 256),
        _shape([2048], [2048], 64, 16, 1024),
    ),
    (
        "2d-bf16-d64-fast-desc-batch",
        "2d",
        "_enable_fast_paged_kv_desc",
        (64, 32, "bf16"),
        _shape([500, 400], [700, 900], 64, 8, 0),
        _shape(
            [300, 280, 400, 260, 350, 310, 270, 290, 330],
            [300, 600, 400, 800, 350, 310, 500, 290, 330],
            64,
            8,
            0,
        ),
    ),
    (
        "2d-bf16-d64-combo-heads",
        "2d",
        "_enable_combo_2d",
        (64, 32, "bf16"),
        _shape([500, 400], [700, 900], 32, 4, 0),
        _shape(
            [300, 280, 400, 260, 350, 310, 270, 290, 330],
            [300, 600, 400, 800, 350, 310, 500, 290, 330],
            16,
            2,
            0,
        ),
    ),
    (
        "2d-bf16-d256",
        "2d",
        "_d256_gfx950_fast",
        (256, 16, "bf16"),
        _shape([512, 512], [1024, 1024], 16, 2, 0),
        _shape(
            [300, 500, 200, 400, 250, 350, 150, 450, 280],
            [300, 700, 200, 400, 250, 600, 150, 450, 280],
            32,
            4,
            0,
        ),
    ),
    # The decode split sizes segments by num_seqs * num_kv_heads, so the 3D pairs
    # hold that product fixed; windowed decode routes to 2D.
    (
        "3d-decode",
        "3d",
        None,
        (128, 16, "bf16"),
        _shape([1, 1], [4096, 2500], 64, 16, 0),
        _shape([1] * 8, [3000, 900, 4096, 1700, 2200, 3500, 1200, 2800], 16, 4, 0),
    ),
    (
        "3d-two-token",
        "3d",
        None,
        (128, 16, "bf16"),
        _shape([2, 2], [4096, 3100], 64, 16, 0),
        _shape([2] * 8, [2500, 4096, 1900, 2600, 3100, 2200, 2400, 2700], 16, 4, 0),
    ),
)


def _problems(geometry, shape):
    d, bs, dtype = geometry
    needed = sum(math.ceil(n / bs) for n in shape["kv_lens"])
    return _problem(
        q_lens=shape["q_lens"],
        kv_lens=shape["kv_lens"],
        hq=shape["hq"],
        hkv=shape["hkv"],
        d=d,
        bs=bs,
        dtype=dtype,
        window=shape["window"],
        num_blocks=needed + 1,
    )


def _key(problem, path):
    if path == "2d":
        return au._tiled_cache_key(problem, _ARCH)
    return au._tiled_3d_cache_key(problem, _ARCH)


@pytest.mark.parametrize(
    "path, route, geometry, first, second",
    [pytest.param(*c[1:], id=c[0]) for c in _CASES],
)
def test_pair_shares_one_binary(path, route, geometry, first, second):
    """Both shapes take the intended path and route and share one key, so the
    GPU test below exercises one binary on two shapes."""
    a, b = _problems(geometry, first), _problems(geometry, second)
    for p in (a, b):
        assert p.select_path() == path
        if route is not None:
            assert getattr(au, route)(p, _ARCH)
    assert a.num_queries_per_kv == b.num_queries_per_kv
    batch_changes = a.num_seqs != b.num_seqs
    if batch_changes and not (path == "3d" and au._use_decode_grid(a, _ARCH)):
        # The decode grid indexes sequences directly and emits no search.
        assert au._binary_search_iters(a) != au._binary_search_iters(b)
    assert _key(a, path) == _key(b, path)


@pytest.fixture(scope="module")
def torch():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires a gfx950 GPU")
    name = torch.cuda.get_device_properties(0).gcnArchName
    if not name.startswith(_ARCH):
        pytest.skip(f"requires a gfx950 GPU, found {name}")
    return torch


def _case(torch, geometry, shape, seed):
    d, bs, dtype = geometry
    tdt = torch.bfloat16 if dtype == "bf16" else torch.float16
    q_lens, kv_lens, hq, hkv = (
        shape["q_lens"],
        shape["kv_lens"],
        shape["hq"],
        shape["hkv"],
    )
    gen = torch.Generator().manual_seed(seed)
    needed = [math.ceil(n / bs) for n in kv_lens]
    num_blocks = sum(needed) + 1
    poison = num_blocks - 1
    order = torch.randperm(num_blocks - 1, generator=gen).tolist()
    table = torch.full((len(kv_lens), max(needed) + 3), poison, dtype=torch.int32)
    for s, n in enumerate(needed):
        table[s, :n] = torch.tensor(order[:n], dtype=torch.int32)
        order = order[n:]
    q = torch.randn(sum(q_lens), hq, d, generator=gen).to(tdt)
    k = torch.randn(num_blocks, bs, hkv, d, generator=gen).to(tdt)
    v = torch.randn(num_blocks, bs, hkv, d, generator=gen).to(tdt)
    k[poison] = 64.0
    v[poison] = 64.0
    cu = torch.tensor([0, *itertools.accumulate(q_lens)], dtype=torch.int32)
    return q.cuda(), k.cuda(), v.cuda(), cu.cuda(), table.cuda()


def _reference(torch, q, k, v, cu, table, shape, d, bs):
    q_lens, kv_lens, hq, hkv = (
        shape["q_lens"],
        shape["kv_lens"],
        shape["hq"],
        shape["hkv"],
    )
    window = shape["window"]
    out = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
    for s, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens)):
        blocks = table[s, : math.ceil(kv_len / bs)].long()
        keys = k[blocks].reshape(-1, hkv, d)[:kv_len].float()
        values = v[blocks].reshape(-1, hkv, d)[:kv_len].float()
        keys = keys.repeat_interleave(hq // hkv, dim=1)
        values = values.repeat_interleave(hq // hkv, dim=1)
        start = int(cu[s])
        queries = q[start : start + q_len].float()
        scores = torch.einsum("qhd,khd->hqk", queries, keys) / math.sqrt(d)
        q_pos = (kv_len - q_len) + torch.arange(q_len, device=q.device)[:, None]
        k_pos = torch.arange(kv_len, device=q.device)[None, :]
        visible = k_pos <= q_pos
        if window > 0:
            visible = visible & (q_pos - k_pos < window)
        scores = scores.masked_fill(~visible, float("-inf"))
        out[start : start + q_len] = torch.einsum(
            "hqk,khd->qhd", scores.softmax(-1), values
        )
    return out


def _compiled():
    return len(au._ATTN_TILED_CACHE) + len(au._ATTN_3D_TILED_CACHE)


@pytest.mark.parametrize(
    "path, geometry, first, second",
    [pytest.param(c[1], *c[3:], id=c[0]) for c in _CASES],
)
def test_pair_numeric_on_one_binary(torch, path, geometry, first, second):
    d, bs, _ = geometry
    compiled = None
    for i, shape in enumerate((first, second)):
        problem = _problems(geometry, shape)
        q, k, v, cu, table = _case(torch, geometry, shape, seed=i)
        out = torch.empty_like(q)
        au.run_unified_attention_torch(
            problem=problem,
            q=q,
            k=k,
            v=v,
            out=out,
            cu_seqlens_q=cu,
            seqused_k=torch.tensor(shape["kv_lens"], dtype=torch.int32, device="cuda"),
            softmax_scale=1.0 / math.sqrt(d),
            block_table=table,
            softcap=0.0,
            backend="tiled",
            use_graph=False,
        )
        torch.cuda.synchronize()
        if compiled is None:
            compiled = _compiled()
        else:
            assert _compiled() == compiled, "the second shape compiled a new binary"
        want = _reference(torch, q, k, v, cu, table, shape, d, bs)
        max_abs = (out.float() - want).abs().max().item()
        assert (
            math.isfinite(max_abs) and max_abs <= 4e-2
        ), f"shape {i}: max_abs {max_abs:.3e} vs fp32 reference"
