# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""One gfx942 or gfx950 tiled binary serves every head pair, window size and batch.

The tiled 2D (gfx942 also 4-warp GQA) and 3D kernels read the head counts and
the sliding window as kernargs and derive the cu_q search trip count from
``num_seqs``, so shapes that differ only in those share one cache key and one
compiled binary. Each case below runs a pair of such shapes back to back: the
second launch must reuse the first binary (no new compile) and both must match
an FP32 paged reference.

The second shape of each pair changes at least one of the head counts (at the
same GQA ratio), the window size, or the batch (by several log2 search
buckets), so a value that is still baked into the binary corrupts it. Block-table
rows are wider than the batch needs and unused slots point at a poison block,
so a wrong stride cannot pass by luck.

gfx942 routes windowed decode to 2D, so its 3D pairs keep the window at 0. The
gfx950 fast paged-KV descriptor bakes the 64/8 head geometry, so its pair keeps
64/8 and changes the batch; its 32/4 and 16/2 neighbours at the same ratio take
the plain combo binary.

The kernels run on ``rocke.runtime`` device buffers filled from numpy, and the
reference is numpy over the fp16/bf16-rounded inputs, so the launches run in GPU
lanes without torch. 2D launches go through ``run_unified_attention_torch``;
3D launches build and launch the segment + reduce pair as its 3D path does.
The routing and key checks run on any host; the launches need the case's GPU.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from kernels.common import attention_unified as au
from rocke.numeric.bf16 import bf16_to_f32, f32_to_bf16
from rocke.runtime.hip_module import get_device_arch

from ._attention_device_array import DeviceArray

_CUS = {"gfx942": 304, "gfx950": 256}


def _shape(q_lens, kv_lens, hq, hkv, window):
    return {
        "q_lens": q_lens,
        "kv_lens": kv_lens,
        "hq": hq,
        "hkv": hkv,
        "window": window,
    }


# Pairs both arches run unchanged.
_HEADS_ONLY = (_shape([1025], [1025], 32, 8, 0), _shape([1025], [1025], 64, 16, 0))
_D64_WINDOW = (
    _shape([260, 180], [400, 180], 8, 8, 128),
    _shape(
        [70, 90, 50, 100, 60, 80, 40, 30, 110],
        [200, 90, 300, 100, 60, 250, 40, 30, 110],
        16,
        16,
        512,
    ),
)
# The decode split sizes segments by num_seqs * num_kv_heads, so the 3D pairs
# hold that product fixed; windowed decode routes to 2D.
_DECODE = (
    _shape([1, 1], [4096, 2500], 64, 16, 0),
    _shape([1] * 8, [3000, 900, 4096, 1700, 2200, 3500, 1200, 2800], 16, 4, 0),
)
_TWO_TOKEN = (
    _shape([2, 2], [4096, 3100], 64, 16, 0),
    _shape([2] * 8, [2500, 4096, 1900, 2600, 3100, 2200, 2400, 2700], 16, 4, 0),
)
_D64_BATCH_KV = [300, 600, 400, 800, 350, 310, 500, 290, 330]
_D64_BATCH_Q = [300, 280, 400, 260, 350, 310, 270, 290, 330]

# (arch, id, path, route, (d, bs, dtype), first shape, second shape).
# gfx942: ``route`` is the 4-warp GQA cohort both 2D shapes take, or None for
# the plain tiled kernel. gfx950: ``route`` names an attention_unified selector
# both shapes must satisfy, or None for no check. gfx950 pairs sit on one side
# of every seqlen threshold the selectors read (combo above 256 query tokens,
# hipcc above 512), so only the head counts, the window and the batch differ.
_CASES = (
    (
        "gfx942",
        "2d-bf16-d128",
        "2d",
        None,
        (128, 16, "bf16"),
        _shape([300, 200, 250], [300, 450, 250], 32, 8, 0),
        _shape(
            [90, 130, 70, 110, 40, 150, 60, 80, 120],
            [90, 300, 70, 210, 40, 150, 60, 380, 120],
            16,
            4,
            0,
        ),
    ),
    # Head counts alone: the token and sequence counts match, so only the head
    # counts tell the two launches apart.
    ("gfx942", "2d-bf16-d128-heads-only", "2d", None, (128, 16, "bf16"), *_HEADS_ONLY),
    ("gfx942", "2d-fp16-d64-window", "2d", None, (64, 32, "fp16"), *_D64_WINDOW),
    (
        "gfx942",
        "4warp-d256",
        "2d",
        "gfx942_d256_causal",
        (256, 16, "bf16"),
        _shape([1024, 1024], [1024, 1024], 16, 2, 0),
        _shape(
            [300, 500, 200, 400, 250, 350, 150, 450, 280],
            [300, 700, 200, 400, 250, 600, 150, 450, 280],
            32,
            4,
            0,
        ),
    ),
    (
        "gfx942",
        "4warp-fold-window",
        "2d",
        "gfx942_d128_swa",
        (128, 16, "bf16"),
        _shape([1024], [1024], 32, 8, 512),
        _shape(
            [200, 300, 150, 250, 180, 220, 260, 140, 310],
            [400, 300, 350, 250, 180, 520, 260, 140, 310],
            16,
            4,
            256,
        ),
    ),
    ("gfx942", "3d-decode", "3d", None, (128, 16, "bf16"), *_DECODE),
    ("gfx942", "3d-two-token", "3d", None, (128, 16, "bf16"), *_TWO_TOKEN),
    (
        "gfx950",
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
    (
        "gfx950",
        "2d-bf16-d128-single-batch-heads-only",
        "2d",
        "_enable_single_batch_combo",
        (128, 16, "bf16"),
        *_HEADS_ONLY,
    ),
    ("gfx950", "2d-fp16-d64-window", "2d", None, (64, 32, "fp16"), *_D64_WINDOW),
    # The single-batch fp16 d128 window cohort: heads and window change.
    (
        "gfx950",
        "2d-fp16-d128-single-batch-window",
        "2d",
        "_enable_single_batch_combo",
        (128, 16, "fp16"),
        _shape([2048], [2048], 32, 8, 256),
        _shape([2048], [2048], 64, 16, 1024),
    ),
    (
        "gfx950",
        "2d-bf16-d64-fast-desc-batch",
        "2d",
        "_enable_fast_paged_kv_desc",
        (64, 32, "bf16"),
        _shape([500, 400], [700, 900], 64, 8, 0),
        _shape(_D64_BATCH_Q, _D64_BATCH_KV, 64, 8, 0),
    ),
    (
        "gfx950",
        "2d-bf16-d64-combo-heads",
        "2d",
        "_enable_combo_2d",
        (64, 32, "bf16"),
        _shape([500, 400], [700, 900], 32, 4, 0),
        _shape(_D64_BATCH_Q, _D64_BATCH_KV, 16, 2, 0),
    ),
    (
        "gfx950",
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
    # D192 short prefill takes the one-warp 2D kernel, whose 512-half K/V copy
    # ends mid-row: the loaders must keep the paged-KV descriptor.
    (
        "gfx950",
        "2d-bf16-d192-one-warp",
        "2d",
        None,
        (192, 16, "bf16"),
        _shape([64, 48], [256, 192], 16, 2, 0),
        _shape([64, 60, 50, 40, 56], [256, 192, 320, 200, 280], 32, 4, 0),
    ),
    ("gfx950", "3d-decode", "3d", None, (128, 16, "bf16"), *_DECODE),
    ("gfx950", "3d-two-token", "3d", None, (128, 16, "bf16"), *_TWO_TOKEN),
)


def _params(*fields):
    return [pytest.param(*(c[i] for i in fields), id=f"{c[0]}-{c[1]}") for c in _CASES]


def _problem(arch, geometry, shape):
    d, bs, dtype = geometry
    q_lens, kv_lens = shape["q_lens"], shape["kv_lens"]
    return au.UnifiedAttentionProblem(
        total_q=sum(q_lens),
        num_seqs=len(q_lens),
        num_query_heads=shape["hq"],
        num_kv_heads=shape["hkv"],
        head_size=d,
        block_size=bs,
        max_seqlen_q=max(q_lens),
        max_seqlen_k=max(kv_lens),
        dtype=dtype,
        sliding_window=shape["window"],
        num_cus=_CUS[arch],
        num_kv_blocks=sum(math.ceil(n / bs) for n in kv_lens) + 1,
    )


def _key(problem, path, arch):
    if path == "2d":
        return au._tiled_cache_key(problem, arch)
    return au._tiled_3d_cache_key(problem, arch)


def _check_route(arch, path, route, problem):
    if arch == "gfx942":
        found = au._gfx942_4warp_route(problem, arch) if path == "2d" else None
        assert (found.cohort if found else None) == route
    elif route is not None:
        assert getattr(au, route)(problem, arch)


@pytest.mark.parametrize(
    "arch, path, route, geometry, first, second", _params(0, 2, 3, 4, 5, 6)
)
def test_pair_shares_one_binary(arch, path, route, geometry, first, second):
    """Both shapes take the intended path and route and share one key, so the
    GPU test below exercises one binary on two shapes."""
    a, b = _problem(arch, geometry, first), _problem(arch, geometry, second)
    for p in (a, b):
        assert p.select_path() == path
        _check_route(arch, path, route, p)
    assert a.num_queries_per_kv == b.num_queries_per_kv
    batch_changes = a.num_seqs != b.num_seqs
    if batch_changes and not (path == "3d" and au._use_decode_grid(a, arch)):
        # The decode grid indexes sequences directly and emits no search.
        assert au._binary_search_iters(a) != au._binary_search_iters(b)
    assert _key(a, path, arch) == _key(b, path, arch)


def test_d192_pair_takes_the_one_warp_kernel():
    """The D192 pair must stay on the one-warp spec: that is the geometry whose
    K/V copy ends mid-row, which the numeric pair is there to cover."""
    case = next(c for c in _CASES if c[1] == "2d-bf16-d192-one-warp")
    for shape in case[5:]:
        spec = au._tiled_spec_from_problem(_problem("gfx950", case[4], shape), "gfx950")
        assert spec.head_size == 192 and spec.num_warps == 1, spec


def _encode(x, dtype):
    """fp32 values as the kernel's fp16 or bf16 (raw uint16) host array."""
    return x.astype(np.float16) if dtype == "fp16" else f32_to_bf16(x)


def _decode(x, dtype):
    return x.astype(np.float32) if dtype == "fp16" else bf16_to_f32(x)


def _case(geometry, shape, seed):
    """fp32 q, k, v exact in the kernel dtype, cu_seqlens_q and a block table
    wider than needed whose unused slots point at a poison block."""
    d, bs, dtype = geometry
    q_lens, kv_lens, hq, hkv = (shape[f] for f in ("q_lens", "kv_lens", "hq", "hkv"))
    rng = np.random.default_rng(seed)
    needed = [math.ceil(n / bs) for n in kv_lens]
    num_blocks = sum(needed) + 1
    poison = num_blocks - 1
    order = iter(rng.permutation(poison))
    table = np.full((len(kv_lens), max(needed) + 3), poison, dtype=np.int32)
    for s, n in enumerate(needed):
        table[s, :n] = [next(order) for _ in range(n)]

    def randn(*dims):
        return _decode(_encode(rng.standard_normal(dims, np.float32), dtype), dtype)

    q = randn(sum(q_lens), hq, d)
    k = randn(num_blocks, bs, hkv, d)
    v = randn(num_blocks, bs, hkv, d)
    k[poison] = 64.0
    v[poison] = 64.0
    cu = np.array([0, *itertools.accumulate(q_lens)], dtype=np.int32)
    return q, k, v, cu, table


def _reference(q, k, v, cu, table, shape, bs):
    """FP32 causal (optionally windowed) paged attention, one KV head group at
    a time."""
    q_lens, kv_lens, hkv, window = (
        shape[f] for f in ("q_lens", "kv_lens", "hkv", "window")
    )
    d = q.shape[-1]
    ratio = shape["hq"] // hkv
    out = np.zeros(q.shape, dtype=np.float32)
    for s, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens)):
        blocks = table[s, : math.ceil(kv_len / bs)]
        keys = k[blocks].reshape(-1, hkv, d)[:kv_len]
        values = v[blocks].reshape(-1, hkv, d)[:kv_len]
        start = int(cu[s])
        q_pos = (kv_len - q_len) + np.arange(q_len)[:, None]
        k_pos = np.arange(kv_len)[None, :]
        visible = k_pos <= q_pos
        if window > 0:
            visible &= q_pos - k_pos < window
        for g in range(hkv):
            heads = slice(g * ratio, (g + 1) * ratio)
            queries = q[start : start + q_len, heads].transpose(1, 0, 2)
            scores = (queries @ keys[:, g].T) / math.sqrt(d)
            scores = np.where(visible, scores, -np.inf)
            probs = np.exp(scores - scores.max(-1, keepdims=True))
            probs /= probs.sum(-1, keepdims=True)
            out[start : start + q_len, heads] = (probs @ values[:, g]).transpose(
                1, 0, 2
            )
    return out


def _launch_3d(rt, arch, problem, q, k, v, out, cu, seq_lens, table, scale):
    """The 3D segment + reduce pair, launched as ``run_unified_attention_torch``'s
    3D path does (same support check, cache key, segment count and grids)."""
    ok, why = au._explicit_path_supported(problem, None, "3d", arch)
    assert ok, why
    num_segments = au._num_segments(problem, arch)
    prepared = au._get_3d_pipeline(
        problem, au._tiled_3d_cache_key(problem, arch), num_segments, arch=arch
    )
    # NaN until the kernels write it, so a skipped workspace slot fails.
    segm_output, segm_max, segm_expsum = (
        DeviceArray(rt, np.full(spec.shape, np.nan, np.float32))
        for spec in au._attention_3d_workspace_specs(problem, num_segments, None)
    )
    seg_vals = {
        "segm_output_ptr": segm_output,
        "segm_max_ptr": segm_max,
        "segm_expsum_ptr": segm_expsum,
        "query_ptr": q,
        "key_cache_ptr": k,
        "value_cache_ptr": v,
        "sink_ptr": None,
        "block_tables_ptr": table,
        "seq_lens_ptr": seq_lens,
        "alibi_slopes_ptr": 0,
        "qq_bias_ptr": 0,
        "query_start_len_ptr": cu,
        "scale": scale,
        "k_scale": 1.0,
        "v_scale": 1.0,
        "softcap": 0.0,
        "num_seqs": problem.num_seqs,
        "block_table_stride": table.stride(0),
        "qq_bias_stride0": 0,
        **au._runtime_kernarg_values(problem),
    }
    red_vals = {
        "output_ptr": out,
        "segm_output_ptr": segm_output,
        "segm_max_ptr": segm_max,
        "segm_expsum_ptr": segm_expsum,
        "seq_lens_ptr": seq_lens,
        **au._runtime_kernarg_values(problem),
    }
    au._launch_3d_pipeline(prepared, seg_vals, red_vals, stream=0, capturing=False)


def _launch(rt, arch, path, problem, geometry, q, k, v, cu, table, kv_lens):
    """Run one shape on ``path``; returns O as float32."""
    d, _, dtype = geometry
    host_q = _encode(q, dtype)
    # Every byte 0xFF is NaN in fp16 and bf16: an element the kernel skips
    # fails the check instead of passing on stale memory.
    out_host = np.empty_like(host_q)
    out_host.view(np.uint8)[...] = 0xFF
    out = DeviceArray(rt, out_host)
    args = {
        "q": DeviceArray(rt, host_q),
        "k": DeviceArray(rt, _encode(k, dtype)),
        "v": DeviceArray(rt, _encode(v, dtype)),
        "out": out,
        "cu": DeviceArray(rt, cu),
        "seq_lens": DeviceArray(rt, np.array(kv_lens, dtype=np.int32)),
        "table": DeviceArray(rt, table),
        "scale": 1.0 / math.sqrt(d),
    }
    if path == "3d":
        _launch_3d(rt, arch, problem, **args)
    else:
        au.run_unified_attention_torch(
            problem=problem,
            q=args["q"],
            k=args["k"],
            v=args["v"],
            out=out,
            cu_seqlens_q=args["cu"],
            seqused_k=args["seq_lens"],
            softmax_scale=args["scale"],
            block_table=args["table"],
            softcap=0.0,
            backend="tiled",
            use_graph=False,
        )
    # The read is a blocking copy, so the launch has finished.
    return _decode(out.read(), dtype)


def _cache(path):
    return au._ATTN_3D_TILED_CACHE if path == "3d" else au._ATTN_TILED_CACHE


def _compiled():
    return len(au._ATTN_TILED_CACHE) + len(au._ATTN_3D_TILED_CACHE)


@pytest.fixture(scope="module")
def device_arch():
    return get_device_arch()


@pytest.mark.gpu
@pytest.mark.parametrize("arch, path, geometry, first, second", _params(0, 2, 4, 5, 6))
def test_pair_numeric_on_one_binary(
    device_arch, monkeypatch, arch, path, geometry, first, second
):
    if device_arch != arch:
        pytest.skip(f"requires a {arch} GPU, found {device_arch or 'no GPU'}")
    from rocke.runtime.hip_module import Runtime

    # The gfx942 2D graph replay captures through torch; launch eagerly.
    monkeypatch.setenv("HIPDNN_GFX942_2D_GRAPH", "0")
    rt = Runtime()
    _, bs, _ = geometry
    compiled = None
    for i, shape in enumerate((first, second)):
        problem = _problem(arch, geometry, shape)
        q, k, v, cu, table = _case(geometry, shape, seed=i)
        got = _launch(
            rt, arch, path, problem, geometry, q, k, v, cu, table, shape["kv_lens"]
        )
        missed = f"shape {i} missed the {path} cache"
        assert _key(problem, path, arch) in _cache(path), missed
        if compiled is None:
            compiled = _compiled()
        else:
            assert _compiled() == compiled, "the second shape compiled a new binary"
        want = _reference(q, k, v, cu, table, shape, bs)
        max_abs = float(np.abs(got - want).max())
        error = f"shape {i}: max_abs {max_abs:.3e} vs fp32 reference"
        assert math.isfinite(max_abs) and max_abs <= 4e-2, error
