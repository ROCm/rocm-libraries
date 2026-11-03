# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx942 attention band numerics without torch.

The band is the cuDNN ``(left = sliding_window, right = right_bound)`` mask
around the diagonal ``d = p + context``, with ``context = 0`` for top-left and
``kv_len - q_len`` for bottom-right. Key ``k`` is visible to query row ``p`` iff
``k < kv_len``, ``window == 0 or d - k < window`` and ``right < 0 or
k <= d + right``. A row with no visible key must be exactly 0.

``test_attention_unified_mask_gfx942_numeric.py`` covers the same kernels in
more depth, but it needs torch, which the CI GPU lanes do not install. This file
launches the gfx942 tiled 2D, split-KV 3D and dense kernels on ``rocke.runtime``
device buffers (the shared ``DeviceArray``) and checks them against the
torch-free float64 oracle ``rocke.numeric.sdpa_reference``, after rounding the
inputs to fp16 / bf16 with the shared ``rocke.numeric.bf16`` codec. It never
imports torch, and skips only off a gfx942 device.

Run standalone:

    HIP_VISIBLE_DEVICES=0 python -m pytest tests/test_attention_band_gfx942_numpy_numeric.py
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from kernels.common import attention_unified as au
from rocke.numeric.bf16 import bf16_to_f32, f32_to_bf16
from rocke.numeric.sdpa_reference import hipdnn_left_to_window, sdpa_reference
from rocke.runtime.hip_module import get_device_arch

from ._attention_device_array import DeviceArray

_ARCH = get_device_arch()
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(_ARCH != "gfx942", reason=f"needs a gfx942 GPU, found {_ARCH}"),
]

_TOL = {"fp16": 2e-2, "bf16": 4e-2}
# Wild block index after the last block-table row: a read past the row faults.
_WILD_BLOCK = 100_000_000


@pytest.fixture(scope="module")
def rt():
    from rocke.runtime.hip_module import Runtime

    return Runtime()


# ---------------------------------------------------------------------------
# Host encoding and the reference
# ---------------------------------------------------------------------------


def _encode(x, dtype):
    """fp32 values as the kernel's fp16 or bf16 (raw uint16) host array."""
    x = np.asarray(x, dtype=np.float32)
    return x.astype(np.float16) if dtype == "fp16" else f32_to_bf16(x)


def _decode(bits, dtype):
    return bits.astype(np.float32) if dtype == "fp16" else bf16_to_f32(bits)


def _rounded(x, dtype):
    """``x`` as ``dtype``: (device bits, the fp32 values they hold)."""
    bits = _encode(x, dtype)
    return bits, _decode(bits, dtype)


def _reference(q, k, v, *, top_left, window, right, **layout):
    """``sdpa_reference`` for the band; returns (O, rows that see no key).

    ``window`` is the cuDNN left bound (keys ``d - window < k``), which the
    oracle takes as ``window - 1``. ``layout`` is the oracle's ragged offsets."""
    out, lse = sdpa_reference(
        q,
        k,
        v,
        scale=1.0 / math.sqrt(q.shape[-1]),
        left_bound=hipdnn_left_to_window(window if window > 0 else -1),
        right_bound=right,
        top_left=top_left,
        **layout,
    )
    return out, np.isneginf(lse)


def _check(got, want, empty, dtype, label):
    """``empty`` flags the rows (first axis of ``got``) that see no key."""
    assert not np.isnan(got).any(), f"{label}: unwritten (NaN) output"
    err = float(np.abs(got - want).max())
    assert err < _TOL[dtype], f"{label}: max_abs={err:.3e} >= {_TOL[dtype]}"
    if empty.any():
        worst = float(np.abs(got[empty]).max())
        assert worst == 0.0, f"{label}: {int(empty.sum())} empty rows not 0: {worst}"


# ---------------------------------------------------------------------------
# Unified paged attention: tiled 2D and split-KV 3D
# ---------------------------------------------------------------------------

_HQ, _HKV, _D, _BS = 16, 2, 128, 16

# (id, [(q_len, kv_len)], top_left, window, right, bf16 too). right: -1 unbounded,
# 0 causal, R > 0 lookahead.
_UNIFIED_CASES = [
    ("tl_q_lt_kv", [(96, 300), (64, 200)], True, 0, 0, True),
    ("tl_q_gt_kv", [(300, 96), (200, 64)], True, 0, 0, False),
    ("no_mask", [(96, 300), (200, 64)], False, 0, -1, False),
    ("left_only_w64", [(128, 300), (300, 128)], False, 64, -1, False),
    ("tl_two_sided_w64_r16", [(128, 300), (300, 128)], True, 64, 16, True),
    # Bottom-right with q > kv: the first q - kv rows of sequence 0 see no key.
    ("br_empty_rows", [(200, 64), (40, 100)], False, 0, 0, False),
    # Top-left decode: the single query sees key 0 only, so its output is V[0].
    ("tl_decode", [(1, 300), (1, 77), (1, 16)], True, 0, 0, False),
    ("tl_varlen", [(1, 300), (64, 64), (200, 120), (17, 513)], True, 0, 0, True),
]

_UNIFIED_ROWS = [
    pytest.param(seqs, tl, w, r, dtype, path, id=f"{name}-{dtype}-{path}")
    for name, seqs, tl, w, r, bf16 in _UNIFIED_CASES
    for dtype in (("fp16", "bf16") if bf16 else ("fp16",))
    for path in ("2d", "3d")
]


def _problem(seqs, dtype, **kw):
    q_lens = [q for q, _ in seqs]
    kv_lens = [k for _, k in seqs]
    return au.UnifiedAttentionProblem(
        total_q=sum(q_lens),
        num_seqs=len(seqs),
        num_query_heads=kw.pop("heads", _HQ),
        num_kv_heads=kw.pop("kv_heads", _HKV),
        head_size=_D,
        block_size=_BS,
        max_seqlen_q=max(q_lens),
        max_seqlen_k=max(kv_lens),
        dtype=dtype,
        **kw,
    )


def _run_unified(rt, problem, seqs, path, *, seed=0, wild_tail=False):
    """Launch ``problem`` on random paged inputs; returns (got, want, empty).

    Blocks are shuffled; slots past a sequence's blocks hold a poison block of
    large values, so reading past a sequence corrupts the output. With
    ``wild_tail`` the table is followed by wild block indices, so a read past
    the last row faults instead."""
    rng = np.random.default_rng(seed)
    hq, hkv, dtype = problem.num_query_heads, problem.num_kv_heads, problem.dtype
    q_lens = [q for q, _ in seqs]
    kv_lens = [k for _, k in seqs]
    needed = [math.ceil(n / _BS) for n in kv_lens]
    width = max(needed)
    num_blocks = sum(needed) + 1
    poison = num_blocks - 1
    order = iter(rng.permutation(num_blocks - 1))
    table = np.full((len(seqs) + 1, width), poison, dtype=np.int32)
    for s, n in enumerate(needed):
        table[s, :n] = [next(order) for _ in range(n)]
    table[-1] = _WILD_BLOCK if wild_tail else poison

    q_bits, q = _rounded(rng.standard_normal((sum(q_lens), hq, _D)), dtype)
    kv_shape = (num_blocks, _BS, hkv, _D)
    k_raw = rng.standard_normal(kv_shape)
    v_raw = rng.standard_normal(kv_shape)
    k_raw[poison] = v_raw[poison] = 64.0
    k_bits, k = _rounded(k_raw, dtype)
    v_bits, v = _rounded(v_raw, dtype)
    cu = np.array([0, *itertools.accumulate(q_lens)], dtype=np.int32)

    out = DeviceArray(rt, _encode(np.full(q.shape, np.nan), dtype))
    au.run_unified_attention_torch(
        problem=problem,
        q=DeviceArray(rt, q_bits),
        k=DeviceArray(rt, k_bits),
        v=DeviceArray(rt, v_bits),
        out=out,
        cu_seqlens_q=DeviceArray(rt, cu),
        seqused_k=DeviceArray(rt, np.array(kv_lens, dtype=np.int32)),
        softmax_scale=1.0 / math.sqrt(_D),
        block_table=DeviceArray(rt, table, shape=(len(seqs), width)),
        softcap=0.0,
        backend="tiled" if path == "2d" else "3d",
    )
    got = _decode(out.read(), dtype)

    # Each sequence's keys in order, packed for the oracle's ragged layout.
    def packed(cache):
        return np.concatenate(
            [
                cache[table[s, : needed[s]]].reshape(-1, hkv, _D)[:kv_len]
                for s, kv_len in enumerate(kv_lens)
            ]
        )

    want, empty = _reference(
        q,
        packed(k),
        packed(v),
        top_left=problem.causal_top_left,
        window=problem.sliding_window,
        right=problem.right_bound,
        q_offsets=cu,
        kv_offsets=[0, *itertools.accumulate(kv_lens)],
    )
    return got, want, empty.all(axis=1)


@pytest.mark.parametrize("seqs,top_left,window,right,dtype,path", _UNIFIED_ROWS)
def test_unified_band_matches_numpy(rt, seqs, top_left, window, right, dtype, path):
    problem = _problem(
        seqs,
        dtype,
        sliding_window=window,
        causal_top_left=top_left,
        right_bound=right,
    )
    supports = (
        au.supports_native_unified_attention_tiled
        if path == "2d"
        else au.supports_native_unified_attention_3d_tiled
    )
    ok, why = supports(problem, "gfx942")
    assert ok, f"{path} rejected the band: {why}"
    got, want, empty = _run_unified(rt, problem, seqs, path)
    _check(got, want, empty, dtype, f"{path}/{dtype}")


def test_d128_swa_4warp_top_left_reads_only_its_blocks(rt):
    """The 4-warp D128 sliding-window body, top-left. ``kv_len = 100`` is not a
    multiple of its 32-key tile, and ``q_len > kv_len`` rows reach the last tile,
    whose tail keys map to block-table entry 7 of a 7-entry row unless the page
    index is clamped. The row after the table holds wild indices, so an
    unclamped read faults."""
    seqs = [(40, 100), (128, 100)]
    problem = _problem(
        seqs,
        "fp16",
        heads=2,
        kv_heads=1,
        sliding_window=16,
        causal_top_left=True,
        right_bound=0,
    )
    assert au._d128_gfx942_swa_fast(problem, "gfx942"), "left the 4-warp SWA body"
    got, want, empty = _run_unified(rt, problem, seqs, "2d", wild_tail=True)
    _check(got, want, empty, "fp16", "d128_swa_4warp")


# ---------------------------------------------------------------------------
# Dense attention: the top-left band, and the plain bottom-right causal mask
# ---------------------------------------------------------------------------

_DENSE_HQ, _DENSE_HKV, _DENSE_BATCH = 4, 1, 2

# (id, sq, skv, window, right, top_left, persistent, ragged)
_DENSE_CASES = [
    ("tl_causal_256x512", 256, 512, 0, 0, True, False, False),
    ("tl_causal_512x256", 512, 256, 0, 0, True, False, False),
    ("full_512x1024", 512, 1024, 0, -1, True, False, False),
    # Sq != Sk and neither a tile multiple: the ragged q- and k-tile tails.
    ("full_ragged_197x400", 197, 400, 0, -1, True, False, True),
    # q > skv + window: the last rows see no key, in both batches.
    ("left_only_w128_1024x512_empty", 1024, 512, 128, -1, True, False, False),
    ("persist_left_only_w128_1024x512_empty", 1024, 512, 128, -1, True, True, False),
    ("lookahead_r7_512x1024", 512, 1024, 0, 7, True, False, False),
    # Ragged bottom-right causal (chunked prefill), the route unpinned dense
    # dispatch takes by default, on both grids. Batch 0's padded key tail reads
    # batch 1's K/V unless the kernel masks it.
    ("br_ragged_200x1000", 200, 1000, 0, 0, False, False, True),
    ("persist_br_ragged_200x1000", 200, 1000, 0, 0, False, True, True),
    ("br_ragged_197x400", 197, 400, 0, 0, False, False, True),
    ("persist_br_ragged_197x400", 197, 400, 0, 0, False, True, True),
]


@pytest.mark.parametrize(
    "sq,skv,window,right,top_left,persistent,ragged",
    [pytest.param(*c[1:], id=c[0]) for c in _DENSE_CASES],
)
def test_dense_band_matches_numpy(
    rt, sq, skv, window, right, top_left, persistent, ragged
):
    from kernels.gfx942.attention_dense import (
        Gfx942AttentionDenseSpec,
        run_attention_dense_torch,
    )

    dtype = "bf16"
    spec = Gfx942AttentionDenseSpec(
        batch=_DENSE_BATCH,
        seqlen_q=sq,
        seqlen_kv=skv,
        num_query_heads=_DENSE_HQ,
        num_kv_heads=_DENSE_HKV,
        head_size=_D,
        causal=right >= 0,
        right_bound=max(right, 0),
        causal_bottom_right=not top_left,
        dtype=dtype,
        sliding_window=window,
        persistent=persistent,
        num_persistent=304,
        ragged=ragged,
    )
    rng = np.random.default_rng(1)
    b = _DENSE_BATCH
    q_bits, q = _rounded(rng.standard_normal((b, sq, _DENSE_HQ, _D)), dtype)
    k_bits, k = _rounded(rng.standard_normal((b, skv, _DENSE_HKV, _D)), dtype)
    v_bits, v = _rounded(rng.standard_normal((b, skv, _DENSE_HKV, _D)), dtype)
    out = DeviceArray(rt, _encode(np.full(q.shape, np.nan), dtype))
    run_attention_dense_torch(
        spec=spec,
        q=DeviceArray(rt, q_bits),
        k=DeviceArray(rt, k_bits),
        v=DeviceArray(rt, v_bits),
        out=out,
        scale=1.0 / math.sqrt(_D),
    )
    got = _decode(out.read(), dtype)
    # The oracle's dense layout is [B, H, S, D]; the kernel's is [B, S, H, D].
    bhsd = (0, 2, 1, 3)
    want, empty = _reference(
        q.transpose(bhsd),
        k.transpose(bhsd),
        v.transpose(bhsd),
        top_left=top_left,
        window=window,
        right=right,
    )
    want, empty = want.transpose(bhsd), empty.all(axis=1)
    if window and sq > skv + window:
        assert empty.any(), "case meant to have empty rows has none"
    for i in range(b):
        _check(got[i], want[i], empty[i], dtype, f"{spec.kernel_name()} batch {i}")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
