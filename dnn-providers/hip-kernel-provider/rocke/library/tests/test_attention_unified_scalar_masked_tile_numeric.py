# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Scalar unified attention: a row whose leading keys are all masked must still
get the right O.

A sliding window masks every key before the window start. The online softmax
replaces a running max of -inf by 0 only as the exponent reference; carrying
that 0 to the next key pins m at 0, and with every visible score at about -110
(natural log) exp2(s - 0) underflows the fp32 P to 0, so O collapses (NaN or 0)
although the row sees keys.

Covered:
* scalar 2D (``backend="scalar"``), bottom-right and top-left, fp16 and bf16;
* scalar split-3D segment + reduce (launched directly: no dispatcher route
  reaches them), where a whole segment can be masked for a row and must not
  set the merge's overall max;
* rows that see no key at all (top-left, ``q_len > kv_len``, and the default
  bottom-right mask with ``q_len > kv_len`` and no window): O must be 0.
* a NaN per-segment sum must make the reduce give NaN for that row, not be
  skipped as an empty segment; a NaN query row must give NaN O in scalar 2D.
* the default-mask scalar 3D body (bottom-right causal, no window) on a
  ``q_len == kv_len`` prefill, where early rows see only masked keys in their
  trailing segments: those segments must count as empty, not give NaN.

Not covered: sinks, softcap and multi-sequence batches.

The kernels run on ``rocke.runtime`` device buffers filled from numpy, and the
FP32 reference is numpy over the fp16/bf16-rounded inputs, so the tests run in
GPU lanes without torch. The scalar kernels are arch-neutral: the tests run on
any GPU whose target id the attention builders know, and skip elsewhere.
``test_attention_unified_masked_tile_ir.py`` pins the same properties on the
emitted IR without a GPU.
"""

import math

import numpy as np
import pytest

from kernels.common.attention_unified import (
    UnifiedAttention3DSpec,
    UnifiedAttentionProblem,
    UnifiedAttentionReduceSpec,
    build_unified_attention_3d,
    build_unified_attention_reduce,
    run_unified_attention_torch,
)
from rocke.core.arch.target import known_arches
from rocke.numeric.bf16 import bf16_to_f32, f32_to_bf16
from rocke.runtime.hip_module import get_device_arch

from ._attention_device_array import DeviceArray

# The launches compile for the device's own target id, which the builders must
# know (gfx11 devices are only known through their generic target).
_ARCH = get_device_arch()
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        _ARCH not in known_arches(),
        reason=f"needs a GPU with a known attention target, found {_ARCH}",
    ),
]

# Natural-log score of every visible key: exp2(-110 * log2(e)) = 2^-158.7 is
# below the smallest fp32 denormal, so P underflows to 0 when m is pinned at 0.
_SCORE = -110.0
_WINDOW = 32
_HQ, _HKV, _D, _BS = 8, 2, 64, 16
_NUM_SEGMENTS = 8
_DTYPES = ["fp16", "bf16"]


@pytest.fixture(scope="module")
def rt():
    from rocke.runtime.hip_module import Runtime

    return Runtime()


def _encode(x, dtype):
    """fp32 values as the kernel's fp16 or bf16 (raw uint16) host array."""
    x = np.asarray(x, dtype=np.float32)
    return x.astype(np.float16) if dtype == "fp16" else f32_to_bf16(x)


def _decode(x, dtype):
    return x.astype(np.float32) if dtype == "fp16" else bf16_to_f32(x)


def _nan_filled(like):
    """An array like ``like`` with every byte 0xFF: NaN as fp16, bf16 and f32,
    so an element a kernel skips fails the checks instead of passing on stale
    memory."""
    out = np.empty_like(like)
    out.view(np.uint8)[...] = 0xFF
    return out


def _problem(dtype, q_len, kv_len, num_blocks, top_left, window=_WINDOW):
    return UnifiedAttentionProblem(
        total_q=q_len,
        num_seqs=1,
        num_query_heads=_HQ,
        num_kv_heads=_HKV,
        head_size=_D,
        block_size=_BS,
        max_seqlen_q=q_len,
        max_seqlen_k=kv_len,
        dtype=dtype,
        sliding_window=window,
        num_kv_blocks=num_blocks,
        causal_top_left=top_left,
    )


def _inputs(dtype, q_len, kv_len):
    """Q, paged K/V and metadata where every visible score equals ``_SCORE``,
    as host arrays."""
    rng = np.random.default_rng(0)
    max_blocks = (kv_len + _BS - 1) // _BS
    num_blocks = max_blocks + 3
    scale = 1.0 / math.sqrt(_D)
    # q = 1, k = b with D * b * scale = _SCORE: b = -13.75, exact in fp16 and bf16.
    b = _SCORE / (_D * scale)
    shape = (num_blocks, _BS, _HKV, _D)
    perm = rng.permutation(num_blocks)[:max_blocks]
    return dict(
        dtype=dtype,
        q=_encode(np.ones((q_len, _HQ, _D)), dtype),
        k=_encode(np.full(shape, b), dtype),
        v=_encode(rng.standard_normal(shape), dtype),
        block_table=perm.reshape(1, max_blocks).astype(np.int32),
        cu=np.array([0, q_len], dtype=np.int32),
        seqused=np.array([kv_len], dtype=np.int32),
        scale=scale,
        num_blocks=num_blocks,
    )


def _reference(data, kv_len, top_left, window=_WINDOW):
    """FP32 O for one sequence, causal + sliding window (0: none); keyless rows give 0."""
    dtype = data["dtype"]
    q = _decode(data["q"], dtype)
    q_len = q.shape[0]
    nblk = (kv_len + _BS - 1) // _BS
    blocks = data["block_table"][0, :nblk]
    k = _decode(data["k"], dtype)[blocks].reshape(-1, _HKV, _D)[:kv_len]
    v = _decode(data["v"], dtype)[blocks].reshape(-1, _HKV, _D)[:kv_len]
    k = np.repeat(k.transpose(1, 0, 2), _HQ // _HKV, axis=0)
    v = np.repeat(v.transpose(1, 0, 2), _HQ // _HKV, axis=0)
    s = np.matmul(q.transpose(1, 0, 2), k.transpose(0, 2, 1)) * np.float32(
        data["scale"]
    )
    p = np.arange(q_len).reshape(-1, 1)
    kk = np.arange(kv_len).reshape(1, -1)
    diag = p + (0 if top_left else kv_len - q_len)
    visible = kk <= diag
    if window:
        visible &= diag - kk < window
    has_key = visible.any(axis=-1)
    s = np.where(visible, s, -np.inf)
    m = np.where(has_key[:, None], s.max(axis=-1, keepdims=True), 0.0)
    e = np.exp(s - m)
    total = e.sum(axis=-1, keepdims=True)
    probs = np.divide(e, total, out=np.zeros_like(e), where=total > 0)
    return np.matmul(probs, v).transpose(1, 0, 2), has_key


def _run_scalar_2d(rt, problem, data):
    out = DeviceArray(rt, _nan_filled(data["q"]))
    run_unified_attention_torch(
        problem=problem,
        q=DeviceArray(rt, data["q"]),
        k=DeviceArray(rt, data["k"]),
        v=DeviceArray(rt, data["v"]),
        out=out,
        cu_seqlens_q=DeviceArray(rt, data["cu"]),
        seqused_k=DeviceArray(rt, data["seqused"]),
        softmax_scale=data["scale"],
        block_table=DeviceArray(rt, data["block_table"]),
        softcap=0.0,
        backend="scalar",
    )
    return _decode(out.read(), data["dtype"])


def _run_scalar_3d(rt, problem, data, between=None):
    """Scalar split-3D segment kernel then the scalar reduce, launched directly.

    ``between(segm_sum)``, if given, runs after the segment launch and before
    the reduce launch on a host copy of the per-segment sum workspace; the copy
    is written back before the reduce, so a test can edit the workspace.
    """
    from rocke.helpers.compile import compile_kernel
    from rocke.helpers.spec import SignatureBuilder
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    arch = _ARCH
    io = "f16" if problem.dtype == "fp16" else "bf16"
    q_len = problem.total_q

    def launcher(kernel, signature):
        art = compile_kernel(kernel, arch=arch, capture_ir_text=False)
        return KernelLauncher(
            hsaco=art.hsaco, kernel_name=art.kernel_name, signature=signature
        )

    seg_sig = (
        SignatureBuilder()
        .ptr("segm_output_ptr", "f32")
        .ptr("segm_max_ptr", "f32")
        .ptr("segm_expsum_ptr", "f32")
        .ptr("query_ptr", io)
        .ptr("key_cache_ptr", io)
        .ptr("value_cache_ptr", io)
        .ptr("sink_ptr", io)
        .ptr("block_tables_ptr", "i32")
        .ptr("seq_lens_ptr", "i32")
        .ptr("alibi_slopes_ptr", "f32")
        .ptr("qq_bias_ptr", "f32")
        .ptr("query_start_len_ptr", "i32")
        .scalar("scale", "f32")
        .scalar("k_scale", "f32")
        .scalar("v_scale", "f32")
        .scalar("softcap", "f32")
        .scalar("num_seqs", "i32")
        .build()
    )
    red_sig = (
        SignatureBuilder()
        .ptr("output_ptr", io)
        .ptr("segm_output_ptr", "f32")
        .ptr("segm_max_ptr", "f32")
        .ptr("segm_expsum_ptr", "f32")
        .ptr("seq_lens_ptr", "i32")
        .ptr("query_start_len_ptr", "i32")
        .build()
    )
    seg = launcher(
        build_unified_attention_3d(
            UnifiedAttention3DSpec(problem, num_segments=_NUM_SEGMENTS), arch=arch
        ),
        seg_sig,
    )
    red = launcher(
        build_unified_attention_reduce(
            UnifiedAttentionReduceSpec(problem, num_segments=_NUM_SEGMENTS), arch=arch
        ),
        red_sig,
    )

    # The segm_ml descriptor spans max_seqlen_q + 1 tokens.
    rows = problem.max_seqlen_q + 1
    segm_out = DeviceArray(
        rt, np.full((rows, _HQ, _NUM_SEGMENTS, _D), np.nan, dtype=np.float32)
    )
    segm_max = DeviceArray(rt, np.full((rows, _HQ, _NUM_SEGMENTS), np.nan, np.float32))
    segm_sum = DeviceArray(rt, np.full((rows, _HQ, _NUM_SEGMENTS), np.nan, np.float32))
    out = DeviceArray(rt, _nan_filled(data["q"]))
    seqused = DeviceArray(rt, data["seqused"])
    cu = DeviceArray(rt, data["cu"])
    seg(
        {
            "segm_output_ptr": segm_out,
            "segm_max_ptr": segm_max,
            "segm_expsum_ptr": segm_sum,
            "query_ptr": DeviceArray(rt, data["q"]),
            "key_cache_ptr": DeviceArray(rt, data["k"]),
            "value_cache_ptr": DeviceArray(rt, data["v"]),
            "sink_ptr": 0,
            "block_tables_ptr": DeviceArray(rt, data["block_table"]),
            "seq_lens_ptr": seqused,
            "alibi_slopes_ptr": 0,
            "qq_bias_ptr": 0,
            "query_start_len_ptr": cu,
            "scale": float(data["scale"]),
            "k_scale": 1.0,
            "v_scale": 1.0,
            "softcap": 0.0,
            "num_seqs": 1,
        },
        config=LaunchConfig(grid=(q_len, _HQ, _NUM_SEGMENTS * _D), block=(64, 1, 1)),
    )
    if between is not None:
        sums = segm_sum.read()
        between(sums)
        segm_sum.write(sums)
    red(
        {
            "output_ptr": out,
            "segm_output_ptr": segm_out,
            "segm_max_ptr": segm_max,
            "segm_expsum_ptr": segm_sum,
            "seq_lens_ptr": seqused,
            "query_start_len_ptr": cu,
        },
        config=LaunchConfig(grid=(q_len, _HQ, _D), block=(64, 1, 1)),
    )
    return _decode(out.read(), data["dtype"])


def _check(out, ref, has_key, dtype):
    finite = np.isfinite(out)
    assert (
        finite.all()
    ), f"{int((~finite).any(axis=(1, 2)).sum())} rows with non-finite O"
    err = np.abs(out - ref).max(axis=(1, 2))
    # A row that sees keys has |O| ~ mean of V over its window (~0.2 for randn
    # V), so a collapsed row (O = 0) fails; a keyless row must be exactly 0.
    tol = 2e-2 if dtype == "fp16" else 4e-2
    bad = err >= tol
    assert not bad.any(), (
        f"{int(bad.sum())} of {out.shape[0]} query rows wrong; first bad row "
        f"{int(np.flatnonzero(bad)[0])} (sees keys: {bool(has_key[bad][0])})"
    )
    if (~has_key).any():
        assert (out[~has_key] == 0).all(), "keyless rows must give O = 0"


def _check_nan_slot(out, ref, row, head, dtype, message):
    """O is NaN at (row, head) only, and matches ``ref`` everywhere else."""
    assert np.isnan(out[row, head]).all(), message
    others = np.ones(out.shape[:2], dtype=bool)
    others[row, head] = False
    assert np.isfinite(out[others]).all(), "NaN leaked into another (row, head)"
    tol = 2e-2 if dtype == "fp16" else 4e-2
    err = np.abs(out - ref).max(axis=-1)[others]
    assert err.max() < tol, f"{int((err >= tol).sum())} (row, head) wrong"


# (q_len, kv_len, top_left). Rows past the first window have masked leading
# keys; the top-left q_len > kv_len case also has rows that see no key.
_CASES = {
    "bottom_right": (128, 128, False),
    "top_left": (128, 128, True),
    "top_left_keyless": (160, 96, True),
}


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("case", list(_CASES))
def test_scalar_2d_masked_leading_keys(rt, case, dtype):
    q_len, kv_len, top_left = _CASES[case]
    data = _inputs(dtype, q_len, kv_len)
    problem = _problem(dtype, q_len, kv_len, data["num_blocks"], top_left)
    out = _run_scalar_2d(rt, problem, data)
    ref, has_key = _reference(data, kv_len, top_left)
    _check(out, ref, has_key, dtype)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_scalar_2d_default_mask_keyless_rows(rt, dtype):
    """Default mask (bottom-right causal, no window) with q_len > kv_len.

    Query row i sees keys 0..i - (q_len - kv_len), so the first 16 of 32 rows
    see no key, l = 0 and acc * rcp(l) is NaN unless the epilogue stores 0.
    """
    q_len, kv_len = 32, 16
    data = _inputs(dtype, q_len, kv_len)
    problem = _problem(dtype, q_len, kv_len, data["num_blocks"], False, window=0)
    assert problem.default_mask
    out = _run_scalar_2d(rt, problem, data)
    ref, has_key = _reference(data, kv_len, False, window=0)
    assert int((~has_key).sum()) == 16
    _check(out, ref, has_key, dtype)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("case", ["bottom_right", "top_left"])
def test_scalar_2d_propagates_nan_query_row(rt, case, dtype):
    """A NaN query row must give NaN O, not be stored as a keyless 0.

    The NaN scores make the row's sum l NaN. The epilogue stores 0 only for an
    exact-zero sum (``fcmp oeq``); a guard of the form ``l > 0`` (``fcmp ogt``)
    is false for NaN and would turn the corrupt row into a finite 0.

    What each case pins:

    * ``top_left`` (non-default mask): fails under the former ``l > 0`` guard,
      which the scalar 2D epilogue applied to non-default masks only.
    * ``bottom_right`` (the default mask): the keyless guard now runs under
      every mask, and this case checks it does not hide NaN there. It also
      passes against the kernel before that change, which applied no guard
      under the default mask (NaN * rcp(NaN) is NaN anyway).
    """
    q_len, kv_len, top_left = _CASES[case]
    row, head = 100, 3
    data = _inputs(dtype, q_len, kv_len)
    problem = _problem(dtype, q_len, kv_len, data["num_blocks"], top_left)
    ref, has_key = _reference(data, kv_len, top_left)
    assert bool(has_key[row])
    data["q"][row, head] = _encode(np.nan, dtype)
    out = _run_scalar_2d(rt, problem, data)
    _check_nan_slot(
        out, ref, row, head, dtype, f"row {row} head {head}: NaN query did not reach O"
    )


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("case", ["top_left", "top_left_keyless"])
def test_scalar_3d_masked_leading_keys_and_segments(rt, case, dtype):
    # The scalar 3D body applies the window only for non-default masks.
    q_len, kv_len, top_left = _CASES[case]
    data = _inputs(dtype, q_len, kv_len)
    problem = _problem(dtype, q_len, kv_len, data["num_blocks"], top_left)
    out = _run_scalar_3d(rt, problem, data)
    ref, has_key = _reference(data, kv_len, top_left)
    _check(out, ref, has_key, dtype)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_scalar_3d_default_mask_masked_trailing_segments(rt, dtype):
    """Default mask with q_len == kv_len = 128, keys split over 8 segments.

    Row 0 sees key 0 only, so every later segment sees only masked keys.
    Without a safe exponent reference alpha = exp2(-inf - -inf) is NaN, l is
    NaN and the reduce turns the whole row into NaN.
    """
    q_len = kv_len = 128
    data = _inputs(dtype, q_len, kv_len)
    problem = _problem(dtype, q_len, kv_len, data["num_blocks"], False, window=0)
    assert problem.default_mask
    out = _run_scalar_3d(rt, problem, data)
    ref, has_key = _reference(data, kv_len, False, window=0)
    assert bool(has_key.all())
    _check(out, ref, has_key, dtype)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_scalar_3d_reduce_propagates_nan_segment_sum(rt, dtype):
    """A NaN per-segment sum must reach O, not be skipped as an empty segment.

    The reduce treats only an exact-zero sum as an empty segment (``fcmp oeq``).
    A guard of the form ``sum > 0`` (``fcmp ogt``) is false for NaN, so it would
    drop the segment and leave the row finite, hiding the corruption.
    """
    q_len, kv_len, top_left = _CASES["top_left"]
    # Row 100 sees keys 69..100 (top-left causal, window 32). The workspace
    # segm_ml is [max_seqlen_q + 1, Hq, num_segments], indexed by the global
    # token (one sequence starting at 0), so [row, head, seg] is the slot the
    # segment kernel wrote and the reduce reads for (row, head).
    row, head = 100, 3
    picked = {}

    def poison(segm_sum):
        with_keys = np.flatnonzero(segm_sum[row, head] > 0)
        assert with_keys.size > 0, f"row {row} head {head}: no segment has keys"
        seg = int(with_keys[0])
        picked["seg"] = seg
        segm_sum[row, head, seg] = np.nan

    data = _inputs(dtype, q_len, kv_len)
    problem = _problem(dtype, q_len, kv_len, data["num_blocks"], top_left)
    out = _run_scalar_3d(rt, problem, data, between=poison)
    ref, has_key = _reference(data, kv_len, top_left)
    assert bool(has_key[row])

    _check_nan_slot(
        out,
        ref,
        row,
        head,
        dtype,
        f"row {row} head {head}: NaN sum in segment {picked['seg']} did not "
        "reach O (reduce skipped it as empty)",
    )
