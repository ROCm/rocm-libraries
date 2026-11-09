# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tiled 2D attention: a row whose first KV tile is fully masked (sliding
window) must still get the right O.

The CTA's tile range starts at the window start of its first row, so later rows
can see a fully masked first tile. The running max must not be pinned at 0 by
that tile: with every visible score negative, exp2(s - 0) rounds P to 0 and O
collapses although the row sees keys. P rounds to 0 below about -17 natural for
fp16 and -93 for bf16 (P is cast to bf16 before PV); the scalar kernels keep P
in fp32, which rounds to 0 below about -104 (about -87 with denormals flushed). Each
epilogue branch is pinned so all three clamp sites are covered, at KV page
sizes 16 and 64 and head sizes 64, 128 and 256 (256 only where the variant fits
the LDS budget). One auto-route case per arch (no ``tuning_spec``, D64 bf16)
covers a multi-warp production kernel of the default builder.

Blind spots: gfx942 D128 sliding-window shapes normally route to the 4-warp
GQA kernel (finite -1e30 sentinel, not affected), so these cases pin tuning
variants of the default builder via ``tuning_spec``. Not covered here: the
scalar kernels (see test_attention_unified_scalar_masked_tile_numeric.py) and
multi-sequence batches.

The epilogue stores 0 only for a row whose softmax sum is exactly 0 (no visible
key); ``test_nan_query_row_reaches_output`` pins that a NaN sum still reaches O
in every branch.

The kernels run on ``rocke.runtime`` device buffers filled from numpy, and the
FP32 reference is numpy over the fp16/bf16-rounded inputs, so the tests run in
GPU lanes without torch. ``test_attention_unified_masked_tile_ir.py`` pins the
running-max carry and the epilogue guard of every branch on the emitted IR
without a GPU.
"""

import math

import numpy as np
import pytest

from kernels.common.attention_unified import (
    UnifiedAttentionProblem,
    _gfx942_4warp_route,
    _tiled_spec_from_problem,
    run_unified_attention_torch,
)
from rocke.numeric.bf16 import bf16_to_f32, f32_to_bf16
from rocke.runtime.hip_module import get_device_arch

from ._attention_device_array import DeviceArray

_ARCH = get_device_arch()
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        _ARCH not in ("gfx942", "gfx950"),
        reason=f"needs a gfx942 or gfx950 GPU, found {_ARCH}",
    ),
]

# One dispatcher geometry per epilogue branch of build_unified_attention_2d_tiled,
# keyed by (arch, KV page size). On gfx942 the 2x-block tiles of the 32-row
# variants exceed the 64 KiB LDS budget at page size 64, so the 1x-block
# variants of the same branches stand in there.
_BRANCH_VARIANT = {
    ("gfx942", 16): {
        "transposed32": "transposed_x8_nw1_mw32_t2xb_llvm",
        "mfma32": "wide32x8_nw1_mw32_t2xb_llvm",
        "striped16": "narrow_nw1_mw16_t1xb_llvm",
    },
    ("gfx942", 64): {
        "transposed32": "transposed_x8_nw1_mw32_t1xb_llvm",
        "mfma32": "wide32x8_nw1_mw32_t1xb_llvm",
        "striped16": "narrow_nw1_mw16_t1xb_llvm",
    },
    ("gfx950", 16): {
        "transposed32": "transposed32_nw1_mw32_t2xb_llvm",
        "mfma32": "wide32_nw1_mw32_t2xb_llvm",
        "striped16": "narrow_nw1_mw16_t1xb_llvm",
    },
    ("gfx950", 64): {
        "transposed32": "transposed32_nw1_mw32_t2xb_llvm",
        "mfma32": "wide32_nw1_mw32_t2xb_llvm",
        "striped16": "narrow_nw1_mw16_t1xb_llvm",
    },
}
_BRANCHES = ["transposed32", "transposed32_s1", "mfma32", "striped16"]

# (dtype, natural-log score of every visible key), below the P underflow point
# when m is pinned at 0: about -17 for fp16 and -93 for bf16.
_SCORES = [("fp16", -20.0), ("bf16", -200.0)]
_WINDOW = 64
_HQ, _HKV = 32, 8
_HEAD_SIZES = [64, 128, 256]
_PAGE_SIZES = [16, 64]


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


def _problem(dtype, q_len, kv_len, num_blocks, bs, head_size):
    return UnifiedAttentionProblem(
        total_q=q_len,
        num_seqs=1,
        num_query_heads=_HQ,
        num_kv_heads=_HKV,
        head_size=head_size,
        block_size=bs,
        max_seqlen_q=q_len,
        max_seqlen_k=kv_len,
        dtype=dtype,
        sliding_window=_WINDOW,
        num_kv_blocks=num_blocks,
    )


def _branch_spec(problem, branch, bs, arch):
    """The tuning spec that pins ``branch`` of the epilogue on ``arch``."""
    from dispatch.attention.unified_rules import canonicalize_tuning_spec

    if arch == "gfx942":
        from dispatch.attention.gfx942_unified import (
            GFX942_TUNING_VARIANTS as variants,
        )
    else:
        from dispatch.attention.gfx950_unified import (
            GFX950_TUNING_VARIANTS as variants,
        )
    base = branch.removesuffix("_s1")
    spec_id = f"{arch}_u2d_{_BRANCH_VARIANT[(arch, bs)][base]}"
    variant = next(v for v in variants if v.spec_id == spec_id)
    knobs = {"use_transposed_scalar_state": True} if branch.endswith("_s1") else {}
    spec, why = canonicalize_tuning_spec(problem, variant, knobs)
    if spec is None and problem.head_size == 256:
        pytest.skip(f"{spec_id} {knobs} does not take D256 here: {why}")
    assert spec is not None, f"{spec_id} {knobs}: {why}"
    return spec


def _rows_with_masked_leading_tile(kernel_spec, q_len):
    """Query rows whose CTA's first KV tile lies wholly before their window.

    This recomputes the kernel's tile-start rule (the CTA's first tile is the one
    holding its first row's window start) instead of reading it from the kernel,
    so it would miss a change to that rule; the GPU failure on unfixed kernels
    is what confirms the two agree.
    """
    tile = kernel_spec.tile_size
    tokens = kernel_spec.block_m_per_warp * kernel_spec.num_warps // (_HQ // _HKV)
    rows = []
    for q0 in range(0, q_len, tokens):
        if q0 < _WINDOW - 1:
            continue
        first_tile_end = ((q0 - _WINDOW + 1) // tile + 1) * tile
        rows += [
            q0 + k
            for k in range(tokens)
            if q0 + k < q_len and q0 + k - _WINDOW + 1 >= first_tile_end
        ]
    return rows


def _reference(q, k_cache, v_cache, kv_len, block_table, scale, bs):
    """FP32 O for one sequence, bottom-right causal + sliding window, from
    float32 arrays of the uploaded values."""
    q_len, _, d = q.shape
    nblk = (kv_len + bs - 1) // bs
    blocks = block_table[0, :nblk]
    k = k_cache[blocks].reshape(-1, _HKV, d)[:kv_len]
    v = v_cache[blocks].reshape(-1, _HKV, d)[:kv_len]
    k = np.repeat(k.transpose(1, 0, 2), _HQ // _HKV, axis=0)
    v = np.repeat(v.transpose(1, 0, 2), _HQ // _HKV, axis=0)
    s = np.matmul(q.transpose(1, 0, 2), k.transpose(0, 2, 1)) * np.float32(scale)
    p = np.arange(q_len).reshape(-1, 1)
    kk = np.arange(kv_len).reshape(1, -1)
    diag = p + (kv_len - q_len)
    visible = (kk <= diag) & (diag - kk < _WINDOW)
    # Every row sees at least its diagonal key, so no row is all -inf.
    s = np.where(visible, s, -np.inf)
    e = np.exp(s - s.max(axis=-1, keepdims=True))
    probs = e / e.sum(axis=-1, keepdims=True)
    return np.matmul(probs, v).transpose(1, 0, 2)


def _launch(rt, problem, spec, q, k_cache, v_cache, block_table, scale, dtype):
    """Run the tiled 2D path on one sequence; ``q``/``k_cache``/``v_cache`` are
    float32 arrays of values exact in ``dtype``. Returns O as float32."""
    q_len, kv_len = problem.total_q, problem.max_seqlen_k
    host_q = _encode(q, dtype)
    # Every byte 0xFF is NaN in fp16 and bf16: an element the kernel skips
    # fails the checks instead of passing on stale memory.
    out_host = np.empty_like(host_q)
    out_host.view(np.uint8)[...] = 0xFF
    out = DeviceArray(rt, out_host)
    run_unified_attention_torch(
        problem=problem,
        q=DeviceArray(rt, host_q),
        k=DeviceArray(rt, _encode(k_cache, dtype)),
        v=DeviceArray(rt, _encode(v_cache, dtype)),
        out=out,
        cu_seqlens_q=DeviceArray(rt, np.array([0, q_len], dtype=np.int32)),
        seqused_k=DeviceArray(rt, np.array([kv_len], dtype=np.int32)),
        softmax_scale=scale,
        block_table=DeviceArray(rt, block_table),
        softcap=0.0,
        backend="tiled",
        tuning_spec=spec,
    )
    return _decode(out.read(), dtype)


def _paged_cache(rng, kv_len, bs, head_size, dtype, k_value=None):
    """K/V caches (float32 values exact in ``dtype``) with three spare pages and
    a shuffled block table. K is ``k_value`` everywhere, or randn if None."""
    max_blocks = (kv_len + bs - 1) // bs
    num_blocks = max_blocks + 3
    shape = (num_blocks, bs, _HKV, head_size)
    if k_value is None:
        k_cache = _decode(_encode(rng.standard_normal(shape), dtype), dtype)
    else:
        k_cache = np.full(shape, k_value, dtype=np.float32)
    v_cache = _decode(_encode(rng.standard_normal(shape), dtype), dtype)
    perm = rng.permutation(num_blocks)[:max_blocks]
    block_table = perm.reshape(1, max_blocks).astype(np.int32)
    return k_cache, v_cache, block_table, num_blocks


def _run_and_check(rt, dtype, score, bs, head_size, spec_for):
    """Run one sequence where every visible score is ``score`` and check O.

    ``spec_for(problem)`` returns the tuning spec to pin, or ``None`` for the
    production auto route.
    """
    q_len = kv_len = 256
    rng = np.random.default_rng(0)
    scale = 1.0 / math.sqrt(head_size)
    # Every visible score equals `score`: q = a, k = b with a * b * D * scale = score.
    a = 1.0 if dtype == "fp16" else 4.0
    # Rounded to ``dtype``, the value the kernel reads.
    b = _decode(_encode([score / (a * head_size * scale)], dtype), dtype)[0]
    q = np.full((q_len, _HQ, head_size), a, dtype=np.float32)
    k_cache, v_cache, block_table, num_blocks = _paged_cache(
        rng, kv_len, bs, head_size, dtype, k_value=b
    )

    problem = _problem(dtype, q_len, kv_len, num_blocks, bs, head_size)
    spec = spec_for(problem)
    kernel_spec = (
        spec.kernel_spec
        if spec is not None
        else _tiled_spec_from_problem(problem, _ARCH)
    )
    assert _rows_with_masked_leading_tile(
        kernel_spec, q_len
    ), "geometry no longer gives any row a fully masked leading tile"
    out = _launch(rt, problem, spec, q, k_cache, v_cache, block_table, scale, dtype)
    o_ref = _reference(q, k_cache, v_cache, kv_len, block_table, scale, bs)

    assert np.isfinite(out).all(), "non-finite O"
    # Every row sees at least one key, so a collapsed row has |O| ~ 0 while the
    # reference is the mean of V over its window (|mean| ~ 0.1 for randn V).
    worst = np.abs(out - o_ref).max(axis=(1, 2))
    tol = 2e-2 if dtype == "fp16" else 4e-2
    assert worst.max() < tol, (
        f"{int((worst >= tol).sum())} of {q_len} query rows wrong; first bad "
        f"row {int(np.flatnonzero(worst >= tol)[0])}"
    )


@pytest.mark.parametrize("head_size", _HEAD_SIZES)
@pytest.mark.parametrize("bs", _PAGE_SIZES)
@pytest.mark.parametrize("dtype,score", _SCORES)
@pytest.mark.parametrize("branch", _BRANCHES)
def test_masked_leading_tile_keeps_rows_with_keys(
    rt, branch, dtype, score, bs, head_size
):
    _run_and_check(
        rt,
        dtype,
        score,
        bs,
        head_size,
        lambda p: _branch_spec(p, branch, bs, _ARCH),
    )


def test_masked_leading_tile_auto_route_d64_bf16(rt):
    # No tuning spec: the production route. At D64 bf16 with a sliding window it
    # is the default builder with several warps per CTA (gfx942: 4 warps, tile
    # 16; gfx950: 2 warps, tile 64), not the gfx942 4-warp GQA kernel.
    problem = _problem("bf16", 256, 256, 256 // 16 + 3, 16, 64)
    assert _gfx942_4warp_route(problem, _ARCH) is None
    assert _tiled_spec_from_problem(problem, _ARCH).num_warps > 1
    _run_and_check(rt, "bf16", -200.0, 16, 64, lambda p: None)


_NAN_ROW, _NAN_HEAD = 100, 5


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("branch", _BRANCHES)
def test_nan_query_row_reaches_output(rt, branch, dtype):
    """A NaN in one query row must give NaN O for that (row, head) only.

    The NaN scores make the row's softmax sum ``l`` NaN. The epilogue's keyless
    guard must treat only ``l == 0`` as "no key": the former ``l > 0`` guard
    stored 0 for this row in every branch below (``NaN > 0`` is false).
    """
    bs, head_size = 16, 128
    q_len = kv_len = 256
    rng = np.random.default_rng(0)
    scale = 1.0 / math.sqrt(head_size)
    q = _decode(_encode(rng.standard_normal((q_len, _HQ, head_size)), dtype), dtype)
    q[_NAN_ROW, _NAN_HEAD] = np.nan
    k_cache, v_cache, block_table, num_blocks = _paged_cache(
        rng, kv_len, bs, head_size, dtype
    )

    problem = _problem(dtype, q_len, kv_len, num_blocks, bs, head_size)
    spec = _branch_spec(problem, branch, bs, _ARCH)
    out = _launch(rt, problem, spec, q, k_cache, v_cache, block_table, scale, dtype)
    o_ref = _reference(q, k_cache, v_cache, kv_len, block_table, scale, bs)

    assert np.isnan(out[_NAN_ROW, _NAN_HEAD]).all(), "NaN row was hidden"
    nan_slot = np.zeros((q_len, _HQ), dtype=bool)
    nan_slot[_NAN_ROW, _NAN_HEAD] = True
    rest, rest_ref = out[~nan_slot], o_ref[~nan_slot]
    assert np.isfinite(rest).all(), "NaN leaked into another (row, head)"
    tol = 2e-2 if dtype == "fp16" else 4e-2
    assert np.abs(rest - rest_ref).max() < tol
