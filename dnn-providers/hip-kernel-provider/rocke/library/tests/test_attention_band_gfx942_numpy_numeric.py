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
device buffers and checks them against an FP32 numpy reference computed from
the band definition above, after rounding the inputs to fp16 / bf16. It never
imports torch, and skips only off a gfx942 device.

Run standalone:

    HIP_VISIBLE_DEVICES=0 python -m pytest tests/test_attention_band_gfx942_numpy_numeric.py
"""

from __future__ import annotations

import ctypes
import itertools
import math

import numpy as np
import pytest

from kernels.common import attention_unified as au
from rocke.runtime.hip_module import get_device_arch

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
# Host buffers: fp16 natively, bf16 as uint16 bit patterns (numpy has no bf16)
# ---------------------------------------------------------------------------


def _to_device_dtype(x, dtype):
    """Round fp32 ``x`` to ``dtype``; returns (device bits, rounded fp32)."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    if dtype == "fp16":
        bits = x.astype(np.float16)
        return bits, bits.astype(np.float32)
    u = x.view(np.uint32)
    bits = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    return bits, _from_device_dtype(bits, dtype)


def _from_device_dtype(bits, dtype):
    if dtype == "fp16":
        return bits.astype(np.float32)
    return (bits.astype(np.uint32) << 16).view(np.float32)


def _nan_bits(shape, dtype):
    if dtype == "fp16":
        return np.full(shape, np.nan, dtype=np.float16)
    return np.full(shape, 0x7FC0, dtype=np.uint16)


def _u8(array):
    return (ctypes.c_uint8 * array.nbytes).from_buffer(array)


class _DeviceArray:
    """A device copy of a numpy array, exposing the tensor attributes the
    launchers read. ``shape`` may be smaller than the host array along the first
    axis, which models a view of the leading rows."""

    device = None

    def __init__(self, rt, host, *, shape=None):
        from rocke.runtime.launcher import DeviceMem

        self._rt = rt
        self._host = np.ascontiguousarray(host)
        self._mem = DeviceMem(self._host.nbytes)
        rt.memcpy_h2d(self._mem.ptr(), _u8(self._host), self._host.nbytes)
        self.shape = self._host.shape if shape is None else tuple(shape)

    def stride(self, dim):
        return self._host.strides[dim] // self._host.itemsize

    def data_ptr(self):
        return self._mem.ptr()

    def read(self):
        out = np.empty_like(self._host)
        self._rt.memcpy_d2h(_u8(out), self._mem.ptr(), out.nbytes)
        return out


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------


def _visible(q_len, kv_len, *, top_left, window, right):
    p = np.arange(q_len)[:, None]
    k = np.arange(kv_len)[None, :]
    d = p + (0 if top_left else kv_len - q_len)
    vis = np.ones((q_len, kv_len), dtype=bool)
    if window > 0:
        vis &= d - k < window
    if right >= 0:
        vis &= k <= d + right
    return vis


def _attend(q, k, v, vis, scale):
    """``q [Sq,Hq,D]``, ``k``/``v`` ``[Skv,Hkv,D]`` fp32 -> ``[Sq,Hq,D]``; rows
    with no visible key are 0."""
    ratio = q.shape[1] // k.shape[1]
    k = np.repeat(k, ratio, axis=1)
    v = np.repeat(v, ratio, axis=1)
    scores = np.einsum("qhd,khd->hqk", q, k) * scale
    scores = np.where(vis[None], scores, -np.inf)
    row_max = scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores - np.where(np.isfinite(row_max), row_max, 0.0))
    denom = probs.sum(axis=-1, keepdims=True)
    probs = np.divide(probs, denom, out=np.zeros_like(probs), where=denom > 0)
    return np.einsum("hqk,khd->qhd", probs, v)


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

    q_bits, q = _to_device_dtype(rng.standard_normal((sum(q_lens), hq, _D)), dtype)
    kv_shape = (num_blocks, _BS, hkv, _D)
    k_raw = rng.standard_normal(kv_shape)
    v_raw = rng.standard_normal(kv_shape)
    k_raw[poison] = v_raw[poison] = 64.0
    k_bits, k = _to_device_dtype(k_raw, dtype)
    v_bits, v = _to_device_dtype(v_raw, dtype)
    cu = np.array([0, *itertools.accumulate(q_lens)], dtype=np.int32)

    out = _DeviceArray(rt, _nan_bits(q.shape, dtype))
    au.run_unified_attention_torch(
        problem=problem,
        q=_DeviceArray(rt, q_bits),
        k=_DeviceArray(rt, k_bits),
        v=_DeviceArray(rt, v_bits),
        out=out,
        cu_seqlens_q=_DeviceArray(rt, cu),
        seqused_k=_DeviceArray(rt, np.array(kv_lens, dtype=np.int32)),
        softmax_scale=1.0 / math.sqrt(_D),
        block_table=_DeviceArray(rt, table, shape=(len(seqs), width)),
        softcap=0.0,
        backend="tiled" if path == "2d" else "3d",
    )
    got = _from_device_dtype(out.read(), dtype)

    want = np.zeros_like(q)
    empty = np.zeros(q.shape[0], dtype=bool)
    for s, (q_len, kv_len) in enumerate(seqs):
        rows = slice(int(cu[s]), int(cu[s + 1]))
        keys = k[table[s, : needed[s]]].reshape(-1, hkv, _D)[:kv_len]
        values = v[table[s, : needed[s]]].reshape(-1, hkv, _D)[:kv_len]
        vis = _visible(
            q_len,
            kv_len,
            top_left=problem.causal_top_left,
            window=problem.sliding_window,
            right=problem.right_bound,
        )
        want[rows] = _attend(q[rows], keys, values, vis, 1.0 / math.sqrt(_D))
        empty[rows] = ~vis.any(axis=1)
    return got, want, empty


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
# Dense attention (top-left band; gfx942 dense has no bottom-right body)
# ---------------------------------------------------------------------------

_DENSE_HQ, _DENSE_HKV, _DENSE_BATCH = 4, 1, 2

# (id, sq, skv, window, right, persistent)
_DENSE_CASES = [
    ("tl_causal_256x512", 256, 512, 0, 0, False),
    ("tl_causal_512x256", 512, 256, 0, 0, False),
    ("full_512x1024", 512, 1024, 0, -1, False),
    # q > skv + window: the last rows see no key, in both batches.
    ("left_only_w128_1024x512_empty", 1024, 512, 128, -1, False),
    ("persist_left_only_w128_1024x512_empty", 1024, 512, 128, -1, True),
    ("lookahead_r7_512x1024", 512, 1024, 0, 7, False),
]


@pytest.mark.parametrize(
    "sq,skv,window,right,persistent",
    [pytest.param(*c[1:], id=c[0]) for c in _DENSE_CASES],
)
def test_dense_band_matches_numpy(rt, sq, skv, window, right, persistent):
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
        dtype=dtype,
        sliding_window=window,
        persistent=persistent,
        num_persistent=304,
    )
    rng = np.random.default_rng(1)
    b = _DENSE_BATCH
    q_bits, q = _to_device_dtype(rng.standard_normal((b, sq, _DENSE_HQ, _D)), dtype)
    k_bits, k = _to_device_dtype(rng.standard_normal((b, skv, _DENSE_HKV, _D)), dtype)
    v_bits, v = _to_device_dtype(rng.standard_normal((b, skv, _DENSE_HKV, _D)), dtype)
    out = _DeviceArray(rt, _nan_bits(q.shape, dtype))
    scale = 1.0 / math.sqrt(_D)
    run_attention_dense_torch(
        spec=spec,
        q=_DeviceArray(rt, q_bits),
        k=_DeviceArray(rt, k_bits),
        v=_DeviceArray(rt, v_bits),
        out=out,
        scale=scale,
    )
    got = _from_device_dtype(out.read(), dtype)
    vis = _visible(sq, skv, top_left=True, window=window, right=right)
    empty = ~vis.any(axis=1)
    if window and sq > skv + window:
        assert empty.any(), "case meant to have empty rows has none"
    for i in range(b):
        want = _attend(q[i], k[i], v[i], vis, scale)
        _check(got[i], want, empty, dtype, f"{spec.kernel_name()} batch {i}")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
