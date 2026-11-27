# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""GPU checks of dropout the per-arch dense numeric tests share.

* The mask bit for bit: with V = identity (Skv == D) the output element
  ``O[b, i, h, j]`` is non-zero iff key ``j`` of row ``(b, h, i)`` is visible and
  kept (:func:`assert_mask_bit_exact`).
* O against ``dense_reference`` with the oracle mask and ``1 / (1 - p)``, and LSE
  equal bit for bit to the dropout-off launch: LSE stays pre-dropout
  (:func:`check_output_and_lse`).
* Paged against contiguous, bit for bit (:func:`check_paged`).

Each takes the arch's ``run_attention_dense_torch`` and a shipped spec with
``dropout=True``.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from rocke.numeric.compare import max_abs_error
from rocke.numeric.philox_dropout import dropout_scale, philox_keep_mask
from rocke.runtime.device_array import DeviceArray, host_values
from rocke.tests.ieee_compare import same_bits

from ._dense_numeric_gpu import (
    dense_reference,
    filled,
    guarded,
    operand,
    upload,
    values,
)

# High seed bits set, and an offset whose low word wraps inside every problem
# below: both halves of the 64-bit seed and counter reach the kernel.
SEED = 0xDEADBEEF_F00DCAFE
OFFSET = (1 << 32) - 37

#: How seed and offset reach the kernel: by value, from device memory, or one each.
FORMS = ("value", "device", "mixed")


def _device_i64(value: int) -> DeviceArray:
    signed = value - (1 << 64) if value >= 1 << 63 else value
    return DeviceArray(np.array([signed], dtype=np.int64), "i64")


def dropout_args(form: str, p: float, *, seed=SEED, offset=OFFSET) -> dict:
    """The ``run_attention_dense_torch`` dropout keywords in ``form``."""
    by_device = {
        "value": (False, False),
        "device": (True, True),
        "mixed": (True, False),
    }
    seed_dev, offset_dev = by_device[form]
    return {
        "dropout_seed": _device_i64(seed) if seed_dev else seed,
        "dropout_offset": _device_i64(offset) if offset_dev else offset,
        "dropout_p": p,
    }


def keep_mask(batch, num_heads, max_sq, max_sk, p, *, seed=SEED, offset=OFFSET):
    """The oracle's keep mask ``[B, Hq, max_sq, max_sk]`` (bool)."""
    return philox_keep_mask(batch, num_heads, max_sq, max_sk, p, seed, offset)


def scores_inputs(rng, q_shape, k_shape, dtype):
    """Q and K at half scale: every ``exp(S - m)`` of a visible key stays well
    inside the dtype's range, so a kept probability never rounds to zero."""
    return operand(rng, q_shape, dtype, scale=0.5), operand(
        rng, k_shape, dtype, scale=0.5
    )


def identity_v(lead: tuple, num_kv_heads: int, d: int, dtype: str, kv_lens=None):
    """V whose key ``j`` of each sequence is the unit row ``e_j``.

    ``lead`` is ``(B,)`` for ``[B, Skv = d, Hkv, d]``; for packed
    ``[T, Hkv, d]`` pass ``kv_lens`` (each at most ``d``) instead.
    """
    eye = np.eye(d)
    if kv_lens is None:
        host = np.broadcast_to(eye[None, :, None, :], (lead[0], d, num_kv_heads, d))
    else:
        host = np.concatenate(
            [np.broadcast_to(eye[:n, None, :], (n, num_kv_heads, d)) for n in kv_lens]
        )
    return upload(np.ascontiguousarray(host), dtype)


def assert_mask_bit_exact(out, expected, what: str) -> None:
    """``out != 0`` equals ``expected`` (same layout as ``out``)."""
    got = values(out) != 0
    bad = np.argwhere(got != expected)
    assert bad.size == 0, (
        f"{what}: {len(bad)} of {expected.size} mask bits differ, first at {tuple(bad[0])}"
        f" (kernel {'keeps' if got[tuple(bad[0])] else 'drops'})"
    )


def dense_identity_expectation(keep, causal: bool):
    """Expected ``O != 0`` ``[B, Sq, Hq, D]`` of a V = identity launch (Skv == D)."""
    sq, sk = keep.shape[2:]
    visible = np.tril(np.ones((sq, sk), bool)) if causal else np.ones((sq, sk), bool)
    return (keep & visible).transpose(0, 2, 1, 3)


def check_mask(run_dense, spec, *, form: str, p: float = 0.5) -> None:
    """Mask bit for bit on a V = identity launch of ``spec`` (``seqlen_kv == head_size``)."""
    b, sq, hq, hkv, d = (
        spec.batch,
        spec.seqlen_q,
        spec.num_query_heads,
        spec.num_kv_heads,
        spec.head_size,
    )
    assert spec.seqlen_kv == d, "V = identity needs seqlen_kv == head_size"
    rng = np.random.default_rng(0)
    q, k = scores_inputs(rng, (b, sq, hq, d), (b, d, hkv, d), spec.dtype)
    v = identity_v((b,), hkv, d, spec.dtype)
    out = filled(q.shape, spec.dtype)
    run_dense(spec=spec, q=q, k=k, v=v, out=out, scale=d**-0.5, **dropout_args(form, p))
    expected = dense_identity_expectation(keep_mask(b, hq, sq, d, p), spec.causal)
    assert_mask_bit_exact(out, expected, spec.kernel_name())


def check_output_and_lse(run_dense, spec, *, form: str, p: float, tol: float) -> None:
    """O against the reference with the oracle mask; LSE equal to dropout off.

    ``spec`` has ``emit_lse``; ``tol`` is the dropout-off tolerance, scaled here by
    ``1 / (1 - p)`` as the kept probabilities are.
    """
    b, sq, sk, hq, hkv, d = (
        spec.batch,
        spec.seqlen_q,
        spec.seqlen_kv,
        spec.num_query_heads,
        spec.num_kv_heads,
        spec.head_size,
    )
    rng = np.random.default_rng(1)
    q, k = scores_inputs(rng, (b, sq, hq, d), (b, sk, hkv, d), spec.dtype)
    v = operand(rng, (b, sk, hkv, d), spec.dtype)
    scale = d**-0.5
    lse, lse_off = (guarded((b, hq, sq, 1), "fp32", 12345.0) for _ in range(2))
    out = filled(q.shape, spec.dtype)
    common = dict(q=q, k=k, v=v, scale=scale)
    run_dense(spec=spec, out=out, lse=lse, **common, **dropout_args(form, p))
    off = dataclasses.replace(spec, dropout=False)
    run_dense(spec=off, out=filled(q.shape, spec.dtype), lse=lse_off, **common)
    keep = keep_mask(b, hq, sq, sk, p)
    ref, _ = dense_reference(
        q, k, v, scale, causal=spec.causal, dropout=(keep, dropout_scale(p))
    )
    err = max_abs_error(values(out), values(ref))
    assert err < tol / (1.0 - p), f"{spec.kernel_name()}: max_abs={err:.3e}"
    assert same_bits(
        host_values(lse), host_values(lse_off)
    ), f"{spec.kernel_name()}: LSE differs from the dropout-off launch"


def check_paged(
    run_dense, paged_spec, contiguous_spec, *, form: str, p: float, tol: float
):
    """A paged launch equals its contiguous twin bit for bit (logical key positions
    feed the counter), and the twin matches the reference."""
    from ._dense_numeric_gpu import paged_cache, same_values

    sq, sk, hq, hkv, d = (
        paged_spec.seqlen_q,
        paged_spec.seqlen_kv,
        paged_spec.num_query_heads,
        paged_spec.num_kv_heads,
        paged_spec.head_size,
    )
    rng = np.random.default_rng(2)
    q, k = scores_inputs(rng, (1, sq, hq, d), (1, sk, hkv, d), paged_spec.dtype)
    v = operand(rng, (1, sk, hkv, d), paged_spec.dtype)
    k_cache, v_cache, block_tables, kv_lens = paged_cache(
        rng, k, v, paged_spec.block_size
    )
    paged_spec = dataclasses.replace(paged_spec, num_kv_blocks=k_cache.shape[0])
    scale = d**-0.5
    drop = dropout_args(form, p)
    out = filled(q.shape, q.dtype)
    run_dense(spec=contiguous_spec, q=q, k=k, v=v, out=out, scale=scale, **drop)
    paged_out = filled(q.shape, q.dtype)
    run_dense(
        spec=paged_spec,
        q=q,
        k=k_cache,
        v=v_cache,
        out=paged_out,
        scale=scale,
        block_tables=block_tables,
        kv_lens=kv_lens,
        **drop,
    )
    assert same_values(
        paged_out, out
    ), f"{paged_spec.kernel_name()}: paged != contiguous"
    keep = keep_mask(1, hq, sq, sk, p)
    ref, _ = dense_reference(
        q, k, v, scale, causal=True, dropout=(keep, dropout_scale(p))
    )
    err = max_abs_error(values(out), values(ref))
    assert err < tol / (1.0 - p), f"{contiguous_spec.kernel_name()}: max_abs={err:.3e}"
