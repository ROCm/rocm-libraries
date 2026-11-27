# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""GPU checks of dropout the per-arch unified tiled 2D numeric tests share, on a
paged cache with ragged sequences.

Each case names the body it covers and fails if dispatch moves it to another.

* The mask bit for bit (:func:`check_mask`): with V = identity (every
  ``kv_len <= head_size``) the output element ``O[t, h, j]`` is non-zero iff
  key ``j`` of that row is visible and kept.
* O against the FP64 ``sdpa_reference`` with the oracle mask and
  ``1 / (1 - p)``, and LSE bit for bit against the dropout-off launch of the
  same kernel: LSE stays pre-dropout (:func:`check_output_and_lse`).
"""

from __future__ import annotations

import dataclasses
import itertools
from typing import Callable, NamedTuple

import numpy as np
from kernels.common import attention_unified as au
from rocke.numeric.philox_dropout import dropout_scale
from rocke.numeric.sdpa_reference import sdpa_reference
from rocke.runtime.device_array import DeviceArray, host_values

from ._attention_paged_reference import encode_half
from ._attention_unified_support import paged_cache, random_half
from ._dense_dropout_gpu import (
    assert_mask_bit_exact,
    dropout_args,
    keep_mask,
)
from ._dense_numeric_gpu import filled, values

HQ, HKV, BLOCK = 8, 2, 16


class Case(NamedTuple):
    name: str
    q_lens: tuple
    kv_lens: tuple
    causal: bool
    dtype: str
    head_size: int
    body: str
    p: float
    form: str
    compile_backend: str | None = None


class _Inputs(NamedTuple):
    problem: au.UnifiedAttentionProblem
    q: np.ndarray  # [T, Hq, D], rounded to the dtype
    k: np.ndarray  # packed [Tk, Hkv, D]
    v: np.ndarray
    arrays: dict


def _inputs(case: Case, *, identity: bool, seed: int = 0) -> _Inputs:
    rng = np.random.default_rng(seed)
    d, cap = case.head_size, max(case.kv_lens)
    # Half-scale Q and K keep every kept probability well above zero.
    q = random_half(rng, (sum(case.q_lens), HQ, d), case.dtype) * 0.5
    k = random_half(rng, (len(case.kv_lens), HKV, cap, d), case.dtype) * 0.5
    if identity:
        v = np.zeros_like(k)
        v[..., :cap, :] = np.eye(cap, d)[None, None]
    else:
        v = random_half(rng, k.shape, case.dtype)
    # No NaN pages past a sequence's end: the last 32-key tile still reads the
    # block-table entry past the last 16-key page (unbounded page id).
    k_pages, table = paged_cache(k, BLOCK, case.dtype)
    v_pages, _ = paged_cache(v, BLOCK, case.dtype)
    problem = au.UnifiedAttentionProblem(
        total_q=q.shape[0],
        num_seqs=len(case.q_lens),
        num_query_heads=HQ,
        num_kv_heads=HKV,
        head_size=d,
        block_size=BLOCK,
        max_seqlen_q=max(case.q_lens),
        max_seqlen_k=cap,
        dtype=case.dtype,
        num_kv_blocks=k_pages.shape[0],
        right_bound=0 if case.causal else -1,
        emit_lse=True,
        dropout=True,
        compile_backend=case.compile_backend,
    )
    arrays = {
        "q": DeviceArray(encode_half(q, case.dtype), case.dtype),
        "k": k_pages,
        "v": v_pages,
        "cu_seqlens_q": DeviceArray(
            np.array([0, *itertools.accumulate(case.q_lens)], np.int32)
        ),
        "seqused_k": DeviceArray(np.array(case.kv_lens, np.int32)),
        "block_table": DeviceArray(table),
    }
    packed = [
        np.concatenate(
            [x[s, :, :n].transpose(1, 0, 2) for s, n in enumerate(case.kv_lens)]
        )
        for x in (k, v)
    ]
    return _Inputs(problem, q, *packed, arrays)


def _run(inputs: _Inputs, problem, case: Case, **dropout) -> tuple:
    out = filled(inputs.q.shape, case.dtype)
    lse = filled((problem.total_q, HQ, 1), "fp32")
    au.run_unified_attention_torch(
        problem=problem,
        **inputs.arrays,
        out=out,
        softmax_scale=case.head_size**-0.5,
        softcap=0.0,
        lse=lse,
        **dropout,
    )
    return out, lse


def _keep(case: Case, problem):
    return keep_mask(
        problem.num_seqs, HQ, problem.max_seqlen_q, problem.max_seqlen_k, case.p
    )


def _check_body(case: Case, problem, arch: str, body_of: Callable) -> None:
    got = body_of(au._tiled_spec_from_problem(problem, arch))
    assert got == case.body, f"{case.name} dispatches to {got}"


def check_mask(case: Case, arch: str, body_of: Callable) -> None:
    inputs = _inputs(case, identity=True)
    _check_body(case, inputs.problem, arch, body_of)
    out, _ = _run(inputs, inputs.problem, case, **dropout_args(case.form, case.p))
    keep = _keep(case, inputs.problem)
    expected = np.zeros(inputs.q.shape, bool)
    starts = itertools.accumulate((0, *case.q_lens))
    for s, (start, ql, kl) in enumerate(zip(starts, case.q_lens, case.kv_lens)):
        # keep[s, h, i, j] -> O[start + i, h, j]; keys past kv_len give 0, and
        # bottom-right causal hides key j from query i when j > i + kl - ql.
        rows = keep[s, :, :ql, :kl].copy()
        if case.causal:
            i, j = np.ogrid[:ql, :kl]
            rows &= (j <= i + (kl - ql))[None]
        expected[start : start + ql, :, :kl] = rows.transpose(1, 0, 2)
    assert case.head_size >= max(case.kv_lens)
    assert_mask_bit_exact(out, expected, case.name)


def check_output_and_lse(case: Case, arch: str, body_of: Callable) -> None:
    inputs = _inputs(case, identity=False, seed=1)
    _check_body(case, inputs.problem, arch, body_of)
    out, lse = _run(inputs, inputs.problem, case, **dropout_args(case.form, case.p))
    # Auto routing sends the long dropout-off rows to split-KV, whose LSE differs
    # by a few ulp; on the same tiled 2D kernel LSE must not change at all.
    off = dataclasses.replace(inputs.problem, dropout=False)
    _, lse_off = _run(inputs, off, case, backend="tiled")
    offsets = [list(itertools.accumulate((0, *n))) for n in (case.q_lens, case.kv_lens)]
    ref, ref_lse = sdpa_reference(
        inputs.q,
        inputs.k,
        inputs.v,
        q_offsets=offsets[0],
        kv_offsets=offsets[1],
        causal_bottom_right=case.causal,
        dropout_mask=_keep(case, inputs.problem),
        dropout_scale=dropout_scale(case.p),
    )
    tol = (2e-2 if case.dtype == "fp16" else 4e-2) / (1.0 - case.p)
    err = float(np.abs(values(out) - ref).max())
    assert err < tol, f"{case.name}: max_abs={err:.3e} (tol {tol:.1e})"
    got, off_lse = (host_values(x).reshape(-1, HQ) for x in (lse, lse_off))
    np.testing.assert_allclose(got, ref_lse, rtol=0, atol=2e-2)
    np.testing.assert_array_equal(got, off_lse)
