# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""On-GPU numeric lane for additive attention bias on gfx950.

Drives ``run_unified_attention_torch`` with an fp32 additive bias on both the 2D
tiled (``backend="tiled"``) and 3D split-KV (``backend="3d"``) paths and checks
max_abs against a float64 NumPy reference of
``softmax(softcap(Q @ K^T * scale) + bias, causal/window) @ V`` -- the bias is
added after the softcap, as for ALiBi/QQ bias. The 2D lane covers both score
paths: short prefill (16x16) and long prefill (max_seqlen_q > 256, multi- and
single-batch), which selects the transposed 32x32 path, plus a multi-batch
group past max_seqlen_q=512 that compiles through hipcc. Softcap and
sliding-window groups run on 2D and 3D, and a d64 / block 32 GQA-8 bf16 group
covers the transposed 32x32 combo stack. Every 2D group asserts its score path
and compile backend so the coverage cannot silently drift.

Bias addressing (element strides; Sk stride is always 1):
``bias[seq_idx*batch_stride + q_head*head_stride + q_local*sq_stride + k_abs]``.
A stride of 0 broadcasts that dimension. ``q_head`` is the *query* head, so
the GQA cases check that a per-query-head bias is not collapsed onto KV heads.

The numeric tests are marked ``gpu``, need no torch, and skip off a gfx950 device.
The host-side argument-validation tests at the bottom need torch for real CPU
tensors (``_validate_additive_bias`` checks a torch-specific dtype string), so they
are gated with ``requires_torch`` and skip, rather than erroring the whole module,
on an image with no torch (e.g. TheRock's gfx94X CI). Run standalone:

    HIP_VISIBLE_DEVICES=0 python -m pytest tests/test_attention_additive_bias_gfx950_numeric.py
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from kernels import UnifiedAttentionProblem, run_unified_attention_torch
from rocke.runtime.hip_module import get_device_arch

from . import _attention_numpy as an
from . import _unified_paged_numpy as U
from ._attention_device_array import DeviceArray


class _LazyTorch:
    """``torch``, imported on first use; the test skips when it is absent."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        torch = pytest.importorskip(
            "torch", reason="ROCm torch required", exc_type=ImportError
        )
        return getattr(torch, name)


torch = _LazyTorch()

requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="ROCm torch required"
)

_ARCH = get_device_arch()

requires_gfx950_gpu = pytest.mark.skipif(
    _ARCH != "gfx950", reason=f"needs a gfx950 GPU, found {_ARCH}"
)


@pytest.fixture(scope="module")
def rt():
    return an.runtime()


_TOL = {"fp16": 2e-2, "bf16": 4e-2}

_HEAD_SIZE = 128
_BLOCK_SIZE = 16

# case -> (backend, [(query_len, kv_len) per sequence]). ``tiled`` is short
# mixed prefill with a context prefix (16x16 score path). ``tiled_long``
# (multi-batch) and ``tiled_long_1seq`` (single-batch) have max_seqlen_q > 256,
# which routes the 2D kernel onto the transposed 32x32 score path; the test
# asserts that routing. ``3d`` covers decode-style split-KV with a tail block.
_CASES = {
    "tiled": ("tiled", [(37, 101), (64, 64), (5, 130)]),
    "tiled_long": ("tiled", [(300, 420), (512, 512)]),
    "tiled_long_1seq": ("tiled", [(384, 640)]),
    "3d": ("3d", [(1, 300), (3, 517), (1, 64)]),
}

# Extra paths: case -> (backend, seq_lens, features). ``tiled_long_hipcc`` is
# multi-batch prefill past max_seqlen_q=512, which compiles through hipcc
# instead of LLVM-direct. The softcap and sliding-window groups pin the score
# order and the masked bias load under a window, on 2D and 3D; the single-seq
# fp16 window case lands on the transposed 32x32 path. The softcap is small
# enough that ``softcap(QK*scale) + bias`` and ``softcap(QK*scale + bias)``
# differ. Each group runs on two bias layouts (a padded full layout and a
# broadcast one) to bound the lane's runtime; _CASES covers all seven layouts.
_EXTRA_CASES = {
    "tiled_long_hipcc": ("tiled", [(640, 640), (768, 900)], {}),
    "tiled_softcap": ("tiled", [(300, 420), (512, 512)], {"softcap": 2.0}),
    "tiled_window": ("tiled", [(300, 420), (512, 512)], {"sliding_window": 64}),
    "tiled_window_1seq": ("tiled", [(384, 640)], {"sliding_window": 64}),
    "3d_softcap": ("3d", [(1, 300), (3, 517), (1, 64)], {"softcap": 2.0}),
    "3d_window": ("3d", [(1, 300), (3, 517), (1, 64)], {"sliding_window": 64}),
}
_EXTRA_BIAS_MODES = ("full_padded_sq", "bcast_hsq")

# The d64 / block 32 / GQA-8 bf16 prefill cohort takes the transposed 32x32 combo
# stack (scalar softmax state, skipped legacy Q gather; 64/8 heads also use the
# fast paged-KV descriptor). With bias its mask-once / mask-limit opts stay off.
_COMBO_SEQ_LENS = [(512, 512), (300, 420)]
_COMBO_HEADS = {"gqa8_h32": (32, 4), "gqa8_h64": (64, 8)}
_COMBO_KNOBS = {
    "use_transposed_scalar_state": True,
    "use_mfma32_skip_legacy_qreg": True,
    "use_transposed_mask_once": False,
}

# Pinned 2D routing so coverage cannot silently drift:
# case -> {dtype: (score path, compile backend)}.
_ROUTES = {
    "tiled": {"fp16": ("16x16", "llvm"), "bf16": ("16x16", "llvm")},
    "tiled_long": {"fp16": ("transposed32", "llvm"), "bf16": ("transposed32", "llvm")},
    "tiled_long_1seq": {
        "fp16": ("transposed32", "llvm"),
        "bf16": ("transposed32", "llvm"),
    },
    "tiled_long_hipcc": {
        "fp16": ("transposed32", "hipcc"),
        "bf16": ("transposed32", "hipcc"),
    },
    "tiled_softcap": {"fp16": ("16x16", "llvm"), "bf16": ("16x16", "llvm")},
    "tiled_window": {"fp16": ("16x16", "llvm"), "bf16": ("16x16", "llvm")},
    "tiled_window_1seq": {
        "fp16": ("transposed32", "llvm"),
        "bf16": ("transposed32", "llvm"),
    },
    "tiled_combo": {"bf16": ("transposed32", "llvm")},
}

# bias layouts: name -> (broadcast batch, broadcast head, broadcast Sq,
# extra Sk padding). ``bcast_hsq`` is the [B, 1, 1, Sk] key-padding shape.
_BIAS_MODES = {
    "full": (False, False, False, 0),
    "full_padded_sq": (False, False, False, 7),
    "bcast_b": (True, False, False, 0),
    "bcast_h": (False, True, False, 0),
    "bcast_bh": (True, True, False, 0),
    "bcast_sq": (False, False, True, 0),
    "bcast_hsq": (False, True, True, 0),
}

_HEADS = {"mha": (8, 8), "gqa": (8, 2)}


def _make_inputs(
    seq_lens, hq, hkv, dtype, seed=0, head_size=_HEAD_SIZE, block_size=_BLOCK_SIZE
):
    rng = np.random.default_rng(seed)
    query_lens = [s[0] for s in seq_lens]
    kv_lens = [s[1] for s in seq_lens]
    max_blocks = (max(kv_lens) + block_size - 1) // block_size
    num_blocks = len(seq_lens) * max_blocks
    q = an.randn(rng, (sum(query_lens), hq, head_size), dtype)
    k = an.randn(rng, (num_blocks, block_size, hkv, head_size), dtype)
    v = an.randn(rng, (num_blocks, block_size, hkv, head_size), dtype)
    # Disjoint shuffled pages so every sequence reads distinct KV.
    block_table = (
        rng.permutation(num_blocks).astype(np.int32).reshape(len(seq_lens), max_blocks)
    )
    cu_q = np.concatenate([[0], np.cumsum(query_lens)]).astype(np.int32)
    seqused_k = np.array(kv_lens, dtype=np.int32)
    return q, k, v, block_table, cu_q, seqused_k


def _make_bias(mode, num_seqs, hq, max_q, max_k, seed=1):
    """Return (storage, batch_stride, head_stride, sq_stride, full_view).

    ``full_view`` is the logical ``[num_seqs, hq, max_q, max_k]`` bias the
    reference adds; ``storage`` is the compact array the kernel reads.
    """
    bcast_b, bcast_h, bcast_sq, pad = _BIAS_MODES[mode]
    rows = 1 if bcast_sq else max_q
    row_pitch = max_k + pad
    shape = [1 if bcast_b else num_seqs, 1 if bcast_h else hq, rows, row_pitch]
    storage = np.random.default_rng(seed).standard_normal(shape, dtype=np.float32)
    sq_stride = 0 if bcast_sq else row_pitch
    head_stride = 0 if bcast_h else rows * row_pitch
    batch_stride = 0 if bcast_b else (1 if bcast_h else hq) * rows * row_pitch
    full_view = np.broadcast_to(storage[..., :max_k], (num_seqs, hq, max_q, max_k))
    return storage, batch_stride, head_stride, sq_stride, full_view


def _reference(
    q,
    k,
    v,
    block_table,
    seq_lens,
    scale,
    bias_full,
    softcap=0.0,
    sliding_window=0,
    bias_inside_softcap=False,
):
    """float64 paged attention (bottom-right causal) in the kernels' order:
    ``softcap(QK*scale) + bias``, then the causal and sliding-window masks.

    ``bias_inside_softcap`` computes the other order, ``softcap(QK*scale +
    bias)``, only to show a softcap case can tell the two apart.
    """
    hq, hkv = q.shape[1], k.shape[2]
    block_size, head_size = k.shape[1], k.shape[3]
    outs = []
    start = 0
    for i, (q_len, kv_len) in enumerate(seq_lens):
        n_blk = (kv_len + block_size - 1) // block_size
        blocks = block_table[i, :n_blk]
        ks = k[blocks].reshape(-1, hkv, head_size)[:kv_len].astype(np.float64)
        vs = v[blocks].reshape(-1, hkv, head_size)[:kv_len].astype(np.float64)
        ks = np.repeat(ks, hq // hkv, axis=1)
        vs = np.repeat(vs, hq // hkv, axis=1)
        qs = q[start : start + q_len].astype(np.float64)
        s = np.einsum("qhd,khd->hqk", qs, ks) * scale
        if bias_full is not None and bias_inside_softcap:
            s = s + bias_full[i, :, :q_len, :kv_len]
        if softcap > 0:
            s = softcap * np.tanh(s / softcap)
        if bias_full is not None and not bias_inside_softcap:
            s = s + bias_full[i, :, :q_len, :kv_len]
        q_abs = np.arange(q_len)[:, None] + (kv_len - q_len)
        k_pos = np.arange(kv_len)[None, :]
        mask = k_pos > q_abs
        if sliding_window > 0:
            # The kernels keep key k for query position q_abs iff q_abs - k < window.
            mask |= (q_abs - k_pos) >= sliding_window
        s = np.where(mask, -np.inf, s)
        p = np.exp(s - s.max(axis=-1, keepdims=True))
        p /= p.sum(axis=-1, keepdims=True)
        outs.append(np.einsum("hqk,khd->qhd", p, vs))
        start += q_len
    return np.concatenate(outs, axis=0)


def _2d_route(problem):
    from kernels.common.attention_unified import (
        _select_2d_compile_backend,
        _tiled_spec_from_problem,
    )

    spec = _tiled_spec_from_problem(problem, "gfx950")
    assert spec.use_additive_bias
    if spec.use_mfma_32x32 and spec.use_transposed_qk_32x32:
        assert spec.block_m_per_warp == 32
        path = "transposed32"
    elif spec.use_mfma_32x32:
        path = "32x32"
    else:
        path = "16x16"
    return path, _select_2d_compile_backend(problem, "gfx950"), spec


def _check_case(
    rt,
    case,
    backend,
    seq_lens,
    features,
    dtype,
    bias_mode,
    heads,
    head_size=_HEAD_SIZE,
    block_size=_BLOCK_SIZE,
    spec_knobs=None,
):
    hq, hkv = {**_HEADS, **_COMBO_HEADS}[heads]
    softcap = float(features.get("softcap", 0.0))
    sliding_window = int(features.get("sliding_window", 0))
    q, k, v, block_table, cu_q, seqused_k = _make_inputs(
        seq_lens, hq, hkv, dtype, head_size=head_size, block_size=block_size
    )
    max_q = max(s[0] for s in seq_lens)
    max_k = max(s[1] for s in seq_lens)
    scale = head_size**-0.5
    storage, b_stride, h_stride, sq_stride, bias_full = _make_bias(
        bias_mode, len(seq_lens), hq, max_q, max_k
    )

    problem = UnifiedAttentionProblem(
        total_q=q.shape[0],
        num_seqs=len(seq_lens),
        num_query_heads=hq,
        num_kv_heads=hkv,
        head_size=head_size,
        block_size=block_size,
        max_seqlen_q=max_q,
        max_seqlen_k=max_k,
        dtype=dtype,
        sliding_window=sliding_window,
        softcap=softcap,
        use_additive_bias=True,
        num_kv_blocks=int(k.shape[0]),
    )
    if backend == "tiled":
        *route, spec = _2d_route(problem)
        route = tuple(route)
        for knob, want in (spec_knobs or {}).items():
            assert getattr(spec, knob) == want, f"{case}/{dtype}/{heads}: {knob}"
        assert route == _ROUTES[case][dtype], (
            f"{case}/{dtype}/{heads}: expected (path, compile backend) "
            f"{_ROUTES[case][dtype]}, got {route}"
        )
    out = an.empty_device(rt, q.shape, dtype)
    with U.workspace_allocator(rt):
        run_unified_attention_torch(
            problem=problem,
            q=an.to_device(rt, q, dtype),
            k=an.to_device(rt, k, dtype),
            v=an.to_device(rt, v, dtype),
            out=out,
            cu_seqlens_q=DeviceArray(rt, cu_q),
            seqused_k=DeviceArray(rt, seqused_k),
            softmax_scale=scale,
            block_table=DeviceArray(rt, block_table),
            softcap=softcap,
            additive_bias=U.StorageArray(rt, storage),
            additive_bias_batch_stride=b_stride,
            additive_bias_head_stride=h_stride,
            additive_bias_sq_stride=sq_stride,
            backend=backend,
            use_graph=False,
        )
    got = an.read(out, dtype)

    ref_args = (q, k, v, block_table, seq_lens, scale)
    feats = dict(softcap=softcap, sliding_window=sliding_window)
    ref = _reference(*ref_args, bias_full, **feats)
    ref_nobias = _reference(*ref_args, None, **feats)
    tol = _TOL[dtype]
    an.assert_close(got, ref, dtype, f"{case}/{dtype}/{bias_mode}/{heads}", atol=tol)
    # Guard against a silently ignored bias: the bias must move the output.
    shift = np.abs(ref - ref_nobias).max()
    assert shift > 10 * tol, f"bias too weak to be observable (shift={shift:.3e})"
    if softcap > 0:
        # Guard the order: a kernel that put the bias inside the softcap would be
        # off by about ``gap``, so the gap must clear the tolerance with margin
        # for this case to catch the other order.
        wrong = _reference(*ref_args, bias_full, bias_inside_softcap=True, **feats)
        gap = np.abs(ref - wrong).max()
        assert gap > 3 * tol, f"softcap too weak to pin the order (gap={gap:.3e})"


@requires_gfx950_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("heads", sorted(_HEADS))
@pytest.mark.parametrize("bias_mode", list(_BIAS_MODES))
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("case", list(_CASES))
def test_additive_bias_matches_reference(rt, case, dtype, bias_mode, heads):
    backend, seq_lens = _CASES[case]
    _check_case(rt, case, backend, seq_lens, {}, dtype, bias_mode, heads)


@requires_gfx950_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("heads", sorted(_HEADS))
@pytest.mark.parametrize("bias_mode", _EXTRA_BIAS_MODES)
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("case", list(_EXTRA_CASES))
def test_additive_bias_with_features_matches_reference(
    rt, case, dtype, bias_mode, heads
):
    backend, seq_lens, features = _EXTRA_CASES[case]
    _check_case(rt, case, backend, seq_lens, features, dtype, bias_mode, heads)


@requires_gfx950_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("bias_mode", _EXTRA_BIAS_MODES)
@pytest.mark.parametrize("heads", sorted(_COMBO_HEADS))
def test_additive_bias_combo_cohort_matches_reference(rt, heads, bias_mode):
    _check_case(
        rt,
        "tiled_combo",
        "tiled",
        _COMBO_SEQ_LENS,
        {},
        "bf16",
        bias_mode,
        heads,
        head_size=64,
        block_size=32,
        spec_knobs=_COMBO_KNOBS,
    )


# Host-side argument validation. These raise before any compile or launch, so
# they run on CPU tensors and need no GPU.


def _bias_kw():
    return dict(dtype=torch.float32)


def _bias_problem(use_additive_bias=True):
    return UnifiedAttentionProblem(
        total_q=4,
        num_seqs=1,
        num_query_heads=8,
        num_kv_heads=8,
        head_size=_HEAD_SIZE,
        block_size=_BLOCK_SIZE,
        max_seqlen_q=4,
        max_seqlen_k=64,
        dtype="bf16",
        use_additive_bias=use_additive_bias,
    )


def _launch_with_bias(problem, additive_bias, **kwargs):
    """Call the public entry with placeholder tensors; validation fires first."""
    run_unified_attention_torch(
        problem=problem,
        q=None,
        k=None,
        v=None,
        out=None,
        cu_seqlens_q=None,
        seqused_k=None,
        softmax_scale=1.0,
        block_table=None,
        softcap=0.0,
        additive_bias=additive_bias,
        **kwargs,
    )


@pytest.mark.parametrize(
    "use_bias, make_bias, sq_stride, match",
    [
        (True, lambda: None, 64, "requires an additive_bias tensor"),
        (
            False,
            lambda: torch.zeros(4, 64, **_bias_kw()),
            64,
            "use_additive_bias is False",
        ),
        (True, lambda: torch.zeros(4, 64, dtype=torch.float16), 64, "float32"),
        (True, lambda: torch.zeros(64, 4, **_bias_kw()).t(), 64, "contiguous along Sk"),
        (
            True,
            lambda: torch.zeros(4, 64, **_bias_kw()),
            None,
            "requires additive_bias_sq",
        ),
        (True, lambda: torch.zeros(4, 64, **_bias_kw()), 32, "or >= max_seqlen_k=64"),
    ],
    ids=["missing", "flag_off", "fp16", "sk_strided", "no_sq_stride", "short_sq"],
)
@requires_torch
def test_additive_bias_bad_args_rejected(use_bias, make_bias, sq_stride, match):
    with pytest.raises(ValueError, match=match):
        _launch_with_bias(
            _bias_problem(use_bias),
            make_bias(),
            additive_bias_sq_stride=sq_stride,
        )


@requires_torch
@pytest.mark.parametrize("sq_stride", [0, 64, 71])
def test_additive_bias_sq_stride_accepted(sq_stride):
    from kernels.common.attention_unified import _validate_additive_bias

    bias = torch.zeros(4, 71, **_bias_kw())
    assert (
        _validate_additive_bias(_bias_problem(), bias, 0, 0, sq_stride, bias.device)
        == sq_stride
    )


def test_additive_bias_off_needs_no_sq_stride():
    from kernels.common.attention_unified import _validate_additive_bias

    assert _validate_additive_bias(_bias_problem(False), None, 0, 0, None, None) == 0


@requires_torch
def test_additive_bias_device_mismatch_rejected():
    from kernels.common.attention_unified import _validate_additive_bias

    bias = torch.zeros(4, 64, **_bias_kw())  # cpu
    with pytest.raises(ValueError, match="same device as q"):
        _validate_additive_bias(_bias_problem(), bias, 0, 0, 64, torch.device("meta"))


# i32 boundary shape: H=16, Sq=Sk=8192 -> a full [B,H,Sq,Sk] bias is 2**30
# elements per sequence, so B=2 tops out at index 2**31 - 1 and B=3 wraps i32.
_LONG = 8192
_LONG_HEAD = _LONG * _LONG
_LONG_BATCH = 16 * _LONG_HEAD


def _long_bias_problem(num_seqs):
    return UnifiedAttentionProblem(
        total_q=num_seqs * _LONG,
        num_seqs=num_seqs,
        num_query_heads=16,
        num_kv_heads=16,
        head_size=_HEAD_SIZE,
        block_size=_BLOCK_SIZE,
        max_seqlen_q=_LONG,
        max_seqlen_k=_LONG,
        dtype="bf16",
        use_additive_bias=True,
    )


def _validate_long(num_seqs, batch_stride, numel=1):
    from kernels.common.attention_unified import _validate_additive_bias

    # A meta tensor reports its storage size without allocating it.
    return _validate_additive_bias(
        _long_bias_problem(num_seqs),
        torch.empty(numel, device="meta", **_bias_kw()),
        batch_stride,
        _LONG_HEAD,
        _LONG,
        torch.device("meta"),
    )


@requires_torch
def test_additive_bias_i32_index_overflow_rejected():
    with pytest.raises(NotImplementedError, match="i32 bias addressing"):
        _validate_long(3, _LONG_BATCH)


@pytest.mark.parametrize(
    "num_seqs, batch_stride, numel",
    [(2, _LONG_BATCH, 2 * _LONG_BATCH), (3, 0, _LONG_BATCH)],
    ids=["max_index_fits", "batch_broadcast"],
)
@requires_torch
def test_additive_bias_i32_index_limit_accepted(num_seqs, batch_stride, numel):
    assert _validate_long(num_seqs, batch_stride, numel) == _LONG


@requires_torch
@pytest.mark.parametrize("batch_stride", [-1, 2**31], ids=["negative", "over_i32"])
def test_additive_bias_stride_range_rejected(batch_stride):
    from kernels.common.attention_unified import _validate_additive_bias

    bias = torch.zeros(4, 64, **_bias_kw())
    with pytest.raises(ValueError, match="additive_bias_batch_stride"):
        _validate_additive_bias(_bias_problem(), bias, batch_stride, 0, 64, bias.device)


# _bias_problem reads index 3*sq_stride + 63 at most, so sq_stride=64 needs 256
# backed elements past the storage offset.
@pytest.mark.parametrize(
    "make_bias, ok",
    [
        (lambda: torch.zeros(256, **_bias_kw()), True),
        (lambda: torch.zeros(255, **_bias_kw()), False),
        (lambda: torch.zeros(320, **_bias_kw())[64:], True),
        (lambda: torch.zeros(320, **_bias_kw())[65:], False),
    ],
    ids=["exact", "short", "offset_exact", "offset_short"],
)
@requires_torch
def test_additive_bias_storage_bounds(make_bias, ok):
    from kernels.common.attention_unified import _validate_additive_bias

    bias = make_bias()
    args = (_bias_problem(), bias, 0, 0, 64, bias.device)
    if ok:
        assert _validate_additive_bias(*args) == 64
    else:
        with pytest.raises(ValueError, match="only backs"):
            _validate_additive_bias(*args)


def test_additive_bias_attn_values_rejects_null_ptr():
    """Direct ``_attn_values`` callers (benchmarks, parity builders) bypass the
    entry-point check; the packer must still refuse a null bias pointer."""
    from kernels.common.attention_unified import _attn_values

    with pytest.raises(ValueError, match="requires an additive_bias tensor"):
        _attn_values(
            problem=_bias_problem(),
            q=None,
            k=None,
            v=None,
            out=None,
            cu_seqlens_q=None,
            seqused_k=None,
            softmax_scale=1.0,
            block_table=None,
            softcap=0.0,
            sinks=None,
            bt_stride=0,
            include_bt_stride=True,
            additive_bias=None,
            include_additive_bias_strides=True,
            kv_addr_limit=None,
        )
