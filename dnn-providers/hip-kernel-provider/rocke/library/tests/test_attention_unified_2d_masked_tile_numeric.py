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
)
from rocke.runtime.hip_module import get_device_arch
from rocke.tests.lanes import skip_needing_arch

from ._attention_paged_reference import paged_attention_reference, round_to_half
from ._attention_tiled_launch import launch_tiled_one_sequence

_ARCHES = ("gfx942", "gfx950")
_ARCH = get_device_arch()
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        _ARCH not in _ARCHES,
        reason=f"needs a {' or '.join(_ARCHES)} GPU, found {_ARCH}",
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


_SEQUENCE_LEN = 256
# The KV pages _paged_cache allocates beyond those the sequence needs.
_SPARE_PAGES = 3


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


def _canonical_branch_spec(problem, branch, bs, arch):
    """``(spec, why, spec_id, knobs)``: the tuning spec that pins ``branch`` of
    the epilogue on ``arch``, or ``None`` and the reason the rules refuse it."""
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
    return spec, why, spec_id, knobs


def _admitting_arches(problem, branch, bs):
    """The arches whose tuning rules take ``branch`` for ``problem``."""
    return tuple(
        arch
        for arch in _ARCHES
        if _canonical_branch_spec(problem, branch, bs, arch)[0] is not None
    )


def _branch_spec(problem, branch, bs, arch):
    """The tuning spec that pins ``branch`` of the epilogue on ``arch``."""
    spec, why, spec_id, knobs = _canonical_branch_spec(problem, branch, bs, arch)
    if spec is None and problem.head_size == 256:
        # Only gfx950's LDS holds D256 tiles, so elsewhere a D256 case waits
        # for gfx950. _masked_tile_cases leaves out the D256 cases no arch takes.
        skip_needing_arch(
            _admitting_arches(problem, branch, bs),
            reason=f"{spec_id} {knobs} does not take D256 here: {why}",
        )
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


def _reference(q, k_cache, v_cache, kv_len, block_table, scale):
    """FP32 O for one sequence, bottom-right causal + sliding window, from
    float32 arrays of the uploaded values."""
    return paged_attention_reference(
        q, k_cache, v_cache, kv_len, block_table[0], scale, sliding_window=_WINDOW
    )


def _kv_pages(kv_len, bs):
    """The KV pages a sequence of ``kv_len`` keys fills at page size ``bs``."""
    return (kv_len + bs - 1) // bs


def _cache_pages(kv_len, bs):
    """The pages of the cache ``_paged_cache`` allocates: the sequence's and
    ``_SPARE_PAGES`` more."""
    return _kv_pages(kv_len, bs) + _SPARE_PAGES


def _paged_cache(rng, kv_len, bs, head_size, dtype, k_value=None):
    """K/V caches (float32 values exact in ``dtype``) with ``_SPARE_PAGES``
    spare pages and a shuffled block table. K is ``k_value`` everywhere, or
    randn if None."""
    max_blocks = _kv_pages(kv_len, bs)
    num_blocks = _cache_pages(kv_len, bs)
    shape = (num_blocks, bs, _HKV, head_size)
    if k_value is None:
        k_cache = round_to_half(rng.standard_normal(shape), dtype)
    else:
        k_cache = np.full(shape, k_value, dtype=np.float32)
    v_cache = round_to_half(rng.standard_normal(shape), dtype)
    perm = rng.permutation(num_blocks)[:max_blocks]
    block_table = perm.reshape(1, max_blocks).astype(np.int32)
    return k_cache, v_cache, block_table, num_blocks


def _run_and_check(dtype, score, bs, head_size, spec_for):
    """Run one sequence where every visible score is ``score`` and check O.

    ``spec_for(problem)`` returns the tuning spec to pin, or ``None`` for the
    production auto route.
    """
    q_len = kv_len = _SEQUENCE_LEN
    rng = np.random.default_rng(0)
    scale = 1.0 / math.sqrt(head_size)
    # Every visible score equals `score`: q = a, k = b with a * b * D * scale = score.
    a = 1.0 if dtype == "fp16" else 4.0
    # Rounded to ``dtype``, the value the kernel reads.
    b = round_to_half([score / (a * head_size * scale)], dtype)[0]
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
    out = launch_tiled_one_sequence(
        problem, spec, q, k_cache, v_cache, block_table, scale, dtype
    )
    o_ref = _reference(q, k_cache, v_cache, kv_len, block_table, scale)

    assert np.isfinite(out).all(), "non-finite O"
    # Every row sees at least one key, so a collapsed row has |O| ~ 0 while the
    # reference is the mean of V over its window (|mean| ~ 0.1 for randn V).
    worst = np.abs(out - o_ref).max(axis=(1, 2))
    tol = 2e-2 if dtype == "fp16" else 4e-2
    assert worst.max() < tol, (
        f"{int((worst >= tol).sum())} of {q_len} query rows wrong; first bad "
        f"row {int(np.flatnonzero(worst >= tol)[0])}"
    )


def _masked_tile_cases():
    """``(branch, dtype, score, bs, head_size)`` of each case. A D256 case
    that no arch's tuning rules take is left out rather than skipped, since no
    lane could run it."""
    cases = []
    for branch in _BRANCHES:
        for dtype, score in _SCORES:
            for bs in _PAGE_SIZES:
                for head_size in _HEAD_SIZES:
                    problem = _problem(
                        dtype,
                        _SEQUENCE_LEN,
                        _SEQUENCE_LEN,
                        _cache_pages(_SEQUENCE_LEN, bs),
                        bs,
                        head_size,
                    )
                    if head_size != 256 or _admitting_arches(problem, branch, bs):
                        cases.append((branch, dtype, score, bs, head_size))
    return cases


@pytest.mark.parametrize("branch,dtype,score,bs,head_size", _masked_tile_cases())
def test_masked_leading_tile_keeps_rows_with_keys(branch, dtype, score, bs, head_size):
    _run_and_check(
        dtype,
        score,
        bs,
        head_size,
        lambda p: _branch_spec(p, branch, bs, _ARCH),
    )


def test_masked_leading_tile_auto_route_d64_bf16():
    # No tuning spec: the production route. At D64 bf16 with a sliding window it
    # is the default builder with several warps per CTA (gfx942: 4 warps, tile
    # 16; gfx950: 2 warps, tile 64), not the gfx942 4-warp GQA kernel.
    problem = _problem("bf16", 256, 256, 256 // 16 + 3, 16, 64)
    assert _gfx942_4warp_route(problem, _ARCH) is None
    assert _tiled_spec_from_problem(problem, _ARCH).num_warps > 1
    _run_and_check("bf16", -200.0, 16, 64, lambda p: None)


_NAN_ROW, _NAN_HEAD = 100, 5


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("branch", _BRANCHES)
def test_nan_query_row_reaches_output(branch, dtype):
    """A NaN in one query row must give NaN O for that (row, head) only.

    The NaN scores make the row's softmax sum ``l`` NaN. The epilogue's keyless
    guard must treat only ``l == 0`` as "no key": the former ``l > 0`` guard
    stored 0 for this row in every branch below (``NaN > 0`` is false).
    """
    bs, head_size = 16, 128
    q_len = kv_len = 256
    rng = np.random.default_rng(0)
    scale = 1.0 / math.sqrt(head_size)
    q = round_to_half(rng.standard_normal((q_len, _HQ, head_size)), dtype)
    q[_NAN_ROW, _NAN_HEAD] = np.nan
    k_cache, v_cache, block_table, num_blocks = _paged_cache(
        rng, kv_len, bs, head_size, dtype
    )

    problem = _problem(dtype, q_len, kv_len, num_blocks, bs, head_size)
    spec = _branch_spec(problem, branch, bs, _ARCH)
    out = launch_tiled_one_sequence(
        problem, spec, q, k_cache, v_cache, block_table, scale, dtype
    )
    o_ref = _reference(q, k_cache, v_cache, kv_len, block_table, scale)

    assert np.isnan(out[_NAN_ROW, _NAN_HEAD]).all(), "NaN row was hidden"
    nan_slot = np.zeros((q_len, _HQ), dtype=bool)
    nan_slot[_NAN_ROW, _NAN_HEAD] = True
    rest, rest_ref = out[~nan_slot], o_ref[~nan_slot]
    assert np.isfinite(rest).all(), "NaN leaked into another (row, head)"
    tol = 2e-2 if dtype == "fp16" else 4e-2
    assert np.abs(rest - rest_ref).max() < tol
