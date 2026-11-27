# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Stricter checks of the shipped dense attention kernels on gfx942 and gfx950.

The per-arch dense numeric suites compare one seed of one input distribution.
This file adds what they leave open, against the float64 oracle
``rocke.numeric.sdpa_reference`` on exactly the stored values:

* seeds and input distributions (uniform, large magnitude, near-constant K),
  reporting how many rows miss the tolerance and the worst one;
* an output between two guard regions and pre-filled with NaN, so a write
  outside the output, or an element left unwritten, fails;
* rows that see no key (O = 0, LSE = -inf) and extreme but legal magnitudes;
* requests through default dispatch, with no spec id pinned;
* mutation checks: the comparison rejects a wrong GQA head mapping, a top-left
  mask where the kernel applies a bottom-right one, and a wrong softmax scale,
  first on oracle output (host only) and then on kernel output.

Every launch goes through ``dispatch.attention``: request, dispatch, bind,
launch, on ``rocke.runtime`` device arrays.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from kernels.common.attention_mask_type import AttentionMaskType
from rocke.numeric import storage
from rocke.numeric.compare import max_abs_error
from rocke.tests.gpu import current_device_arch_in

from ._dense_numeric_gpu import (
    DENSE_GRID_SPEC_ID,
    attention_tolerance,
    dense_reference,
    filled,
    guard_intact,
    guarded,
    upload,
    values,
)
from ._dense_paged_stand_ins import pinned

# A deliberate bug must move the output by at least this many tolerances.
_MUTATION_MARGIN = 2.0

# The guard regions' value: far outside any attention output of these inputs.
_GUARD_FILL = 12345.0


@pytest.fixture
def arch():
    """The current GPU's arch; skips unless it is gfx942 or gfx950."""
    return current_device_arch_in(
        tuple(DENSE_GRID_SPEC_ID), subject="the dense attention kernels"
    )


def _request(
    arch,
    dtype,
    d,
    hq,
    hkv,
    *,
    batch=1,
    sq=512,
    sk=None,
    mask_type=AttentionMaskType.TOP_LEFT_CAUSAL,
    **options,
):
    """A dense ``[B, S, H, D]`` request with ``Hq = hq`` and ``Hkv = hkv``."""
    from dispatch.attention import AttentionRequest

    return AttentionRequest(
        batch=batch,
        nhead_q=hq,
        nhead_k=hkv,
        seqlen_q=sq,
        seqlen_k=sq if sk is None else sk,
        hdim_q=d,
        hdim_v=d,
        arch=arch,
        dtype=dtype,
        mask_type=mask_type,
        layout="dense",
        **options,
    )


def _host_operands(rng, shapes, kind):
    """Q, K, V of ``shapes`` drawn from the distribution ``kind``, as float64."""
    q_shape, k_shape, v_shape = shapes
    if kind == "normal":
        return tuple(rng.standard_normal(shape) for shape in shapes)
    if kind == "uniform":
        return tuple(rng.uniform(-1.0, 1.0, shape) for shape in shapes)
    if kind == "large":
        return tuple(4.0 * rng.standard_normal(shape) for shape in shapes)
    if kind == "near_constant_k":
        return (
            rng.standard_normal(q_shape),
            1.0 + 1e-3 * rng.standard_normal(k_shape),
            rng.standard_normal(v_shape),
        )
    raise ValueError(f"unknown input distribution {kind!r}")


def _operands(rng, dtype, batch, sq, sk, hq, hkv, d, *, kind="normal"):
    """Device Q ``[B, Sq, Hq, D]`` and K, V ``[B, Sk, Hkv, D]`` of ``dtype``."""
    shapes = ((batch, sq, hq, d), (batch, sk, hkv, d), (batch, sk, hkv, d))
    return tuple(upload(host, dtype) for host in _host_operands(rng, shapes, kind))


def _launch(request, q, k, v, *, pin=True):
    """Dispatch ``request`` (pinned to the arch's default-grid dense kernel unless
    ``pin`` is False) and launch it into a guarded NaN-filled output; returns the
    output, and the LSE when the request emits it."""
    from dispatch.attention import dispatch_attention

    if pin:
        request = pinned(request, DENSE_GRID_SPEC_ID[request.arch])
    result = dispatch_attention(request)
    out = guarded(q.shape, request.dtype, float("nan"), guard_fill=_GUARD_FILL)
    tensors = {"q": q, "k": k, "v": v, "out": out}
    lse = None
    if request.emit_lse:
        lse = filled((request.batch, request.nhead_q, request.seqlen_q, 1), "fp32")
        tensors["lse"] = lse
    result.bind_torch(tensors).launch()
    _assert_written_inside(out)
    return out, lse


def _assert_written_inside(out):
    """The guard regions around ``out`` are intact and none of its elements is
    still NaN."""
    assert guard_intact(out, _GUARD_FILL), "the kernel wrote outside the output"
    unwritten = np.isnan(values(out))
    assert not unwritten.any(), (
        f"{int(unwritten.sum())} output elements are NaN: never written, or "
        "computed as NaN from finite inputs"
    )


def _assert_rows_close(got, want, dtype, what):
    """Every output row within the max-abs tolerance; a failure names how many rows
    miss it and the worst one."""
    tolerance = attention_tolerance(dtype)
    row_error = np.abs(values(got) - values(want)).max(axis=-1)
    row_error = np.nan_to_num(row_error, nan=np.inf)
    over = row_error >= tolerance
    assert not over.any(), (
        f"{what}: {int(over.sum())} of {over.size} rows over {tolerance:.1e}; worst "
        f"row {tuple(np.unravel_index(int(np.argmax(row_error)), row_error.shape))} "
        f"max_abs={row_error.max():.3e}"
    )


# ------------------------------------------------ seeds and distributions ----

_SWEEP_KINDS = [
    ("normal", 0),
    ("normal", 1),
    ("normal", 2),
    ("normal", 3),
    ("uniform", 4),
    ("large", 5),
    ("near_constant_k", 6),
]


@pytest.mark.gpu
@pytest.mark.parametrize("kind,seed", _SWEEP_KINDS, ids=str)
def test_seed_and_distribution_sweep(arch, kind, seed):
    """One binary, inputs the per-arch suites' single seed does not reach,
    compared row by row with the oracle."""
    dtype, d, hq, hkv, s = "fp16", 128, 16, 4, 512
    rng = np.random.default_rng(seed)
    q, k, v = _operands(rng, dtype, 1, s, s, hq, hkv, d, kind=kind)
    out, _ = _launch(_request(arch, dtype, d, hq, hkv, sq=s), q, k, v)
    want, _ = dense_reference(q, k, v, 1.0 / math.sqrt(d))
    _assert_rows_close(out, want, dtype, f"{arch} fp16 D128 GQA16/4 {kind} seed {seed}")


# ------------------------------------------------------- keyless rows, LSE ----

# (dtype, d, hq, hkv, sq, sk, window), top-left causal: row p sees keys
# p - window + 1 .. p, so rows from sk + window - 1 on see none.
_KEYLESS = ("bf16", 128, 16, 4, 512, 256, 64)


@pytest.mark.gpu
def test_keyless_rows_write_zero_output_and_neg_inf_lse(arch):
    dtype, d, hq, hkv, sq, sk, window = _KEYLESS
    request = _request(
        arch, dtype, d, hq, hkv, sq=sq, sk=sk, sliding_window=window, emit_lse=True
    )
    rng = np.random.default_rng(7)
    q, k, v = _operands(rng, dtype, 1, sq, sk, hq, hkv, d)
    out, lse = _launch(request, q, k, v)
    want, want_lse = dense_reference(q, k, v, 1.0 / math.sqrt(d), window=window)
    first_keyless = sk + window - 1
    assert first_keyless < sq, "setup error: no row is keyless"
    assert np.isneginf(want_lse[:, :, first_keyless:]).all()
    got = values(out)
    assert not np.any(
        got[:, first_keyless:]
    ), f"rows from {first_keyless} on see no key but are not all 0"
    _assert_rows_close(out, want, dtype, f"{arch} keyless O")
    got_lse = values(lse)
    np.testing.assert_array_equal(
        np.isneginf(got_lse), np.isneginf(want_lse), err_msg=f"{arch} keyless LSE"
    )
    finite = np.isfinite(want_lse)
    assert max_abs_error(got_lse[finite], want_lse[finite]) < attention_tolerance(
        dtype
    ), f"{arch} LSE of the rows that see a key"


@pytest.mark.gpu
@pytest.mark.parametrize(
    "q_scale,k_scale,v_scale",
    [(8.0, 8.0, 1.0), (1.0, 1.0, 4.0), (32.0, 1.0, 1.0)],
    ids=["peaky-scores", "large-values", "huge-q"],
)
def test_extreme_legal_magnitudes_stay_finite_and_correct(
    arch, q_scale, k_scale, v_scale
):
    """Large scores (a near one-hot softmax) and large V must not overflow or lose
    the row maximum: the output stays finite and inside the tolerance."""
    dtype, d, hq, hkv, s = "fp16", 128, 16, 4, 512
    rng = np.random.default_rng(11)
    q = upload(q_scale * rng.standard_normal((1, s, hq, d)), dtype)
    k = upload(k_scale * rng.standard_normal((1, s, hkv, d)), dtype)
    v = upload(v_scale * rng.standard_normal((1, s, hkv, d)), dtype)
    out, _ = _launch(_request(arch, dtype, d, hq, hkv, sq=s), q, k, v)
    # _launch already refuses NaN; this also refuses an overflow to infinity.
    assert np.isfinite(values(out)).all(), "non-finite output for finite inputs"
    want, _ = dense_reference(q, k, v, 1.0 / math.sqrt(d))
    _assert_rows_close(
        out, want, dtype, f"{arch} magnitudes q{q_scale} k{k_scale} v{v_scale}"
    )


# --------------------------------------------------------- default dispatch ----

# (dtype, d, hq, hkv, batch, seq, mask_type, window)
_CAUSAL, _NO_MASK = AttentionMaskType.TOP_LEFT_CAUSAL, AttentionMaskType.NO_MASK
_DEFAULT_DISPATCH = [
    pytest.param("fp16", 128, 16, 4, 1, 512, _CAUSAL, 0, id="fp16-d128-gqa-causal"),
    pytest.param(
        "bf16", 128, 16, 4, 1, 512, _CAUSAL, 128, id="bf16-d128-gqa-window128"
    ),
    pytest.param("fp16", 64, 16, 16, 2, 256, _NO_MASK, 0, id="fp16-d64-mha-b2-full"),
]


@pytest.mark.gpu
@pytest.mark.parametrize("dtype,d,hq,hkv,batch,s,mask_type,window", _DEFAULT_DISPATCH)
def test_default_dispatch_matches_oracle(
    arch, dtype, d, hq, hkv, batch, s, mask_type, window
):
    """No spec id is pinned: dispatch picks the kernel, binds and launches it."""
    request = _request(
        arch,
        dtype,
        d,
        hq,
        hkv,
        batch=batch,
        sq=s,
        mask_type=mask_type,
        sliding_window=window,
    )
    assert request.spec_id == "auto" and request.algorithm == "auto"
    rng = np.random.default_rng(21)
    q, k, v = _operands(rng, dtype, batch, s, s, hq, hkv, d)
    out, _ = _launch(request, q, k, v, pin=False)
    want, _ = dense_reference(
        q,
        k,
        v,
        1.0 / math.sqrt(d),
        causal=mask_type != AttentionMaskType.NO_MASK,
        window=window,
    )
    _assert_rows_close(out, want, dtype, f"{arch} default dispatch")


# ------------------------------------------------------------ mutation checks ----


def _gqa_interleaved(kv, hq):
    """``kv`` ``[B, S, Hkv, D]`` expanded to ``hq`` heads with the wrong grouping:
    query head ``h`` reads KV head ``h % Hkv`` instead of ``h // (Hq // Hkv)``."""
    hkv = kv.shape[2]
    return values(kv)[:, :, [head % hkv for head in range(hq)]]


def _mutations(q, k, v, scale, hq):
    """The oracle's O for the bottom-right causal launch under one deliberate bug
    each, on the same inputs."""
    return {
        "gqa-interleaved": dense_reference(
            q,
            _gqa_interleaved(k, hq),
            _gqa_interleaved(v, hq),
            scale,
            bottom_right=True,
        )[0],
        "mask-offset": dense_reference(q, k, v, scale, bottom_right=False)[0],
        "scale": dense_reference(q, k, v, scale * 1.05, bottom_right=True)[0],
    }


def _assert_rejected(good, mutated, dtype, label):
    """``mutated`` misses ``good`` by at least ``_MUTATION_MARGIN`` tolerances."""
    tolerance = attention_tolerance(dtype)
    error = max_abs_error(values(good), values(mutated))
    assert error >= _MUTATION_MARGIN * tolerance, (
        f"{label}: the bug moves the output by only {error:.3e}, under "
        f"{_MUTATION_MARGIN:g} x tolerance {tolerance:.1e}"
    )


# (hq, hkv, d, sq, sk): sq < sk moves the bottom-right diagonal off the top-left one.
_MUTATION_SHAPE = (8, 2, 64, 128, 256)


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_comparison_rejects_known_bugs_on_oracle_output(dtype):
    """Host only. The oracle rounded to ``dtype`` stands in for a correct kernel:
    the comparison accepts it and rejects each deliberate bug."""
    hq, hkv, d, sq, sk = _MUTATION_SHAPE
    rng = np.random.default_rng(3)
    q, k, v = (
        storage.round_to(rng.standard_normal(shape), dtype).astype(np.float64)
        for shape in ((1, sq, hq, d), (1, sk, hkv, d), (1, sk, hkv, d))
    )
    scale = 1.0 / math.sqrt(d)
    good, _ = dense_reference(q, k, v, scale, bottom_right=True)
    kernel_like = storage.round_to(good, dtype).astype(np.float64)
    _assert_rows_close(kernel_like, good, dtype, "unmutated")
    for label, mutated in _mutations(q, k, v, scale, hq).items():
        _assert_rejected(kernel_like, mutated, dtype, f"{dtype} {label}")


@pytest.mark.gpu
def test_comparison_rejects_known_bugs_on_kernel_output(arch):
    """The kernel matches the right oracle; the same kernel output misses the
    oracle with a wrong GQA mapping, a top-left mask in place of the kernel's
    bottom-right one, or a wrong scale."""
    dtype, d, hq, hkv, sq, sk = "fp16", 128, 16, 4, 256, 512
    request = _request(
        arch,
        dtype,
        d,
        hq,
        hkv,
        sq=sq,
        sk=sk,
        mask_type=AttentionMaskType.BOTTOM_RIGHT_CAUSAL,
    )
    rng = np.random.default_rng(13)
    q, k, v = _operands(rng, dtype, 1, sq, sk, hq, hkv, d)
    out, _ = _launch(request, q, k, v)
    scale = 1.0 / math.sqrt(d)
    want, _ = dense_reference(q, k, v, scale, bottom_right=True)
    _assert_rows_close(out, want, dtype, f"{arch} bottom-right {sq}x{sk}")
    for label, mutated in _mutations(q, k, v, scale, hq).items():
        _assert_rejected(out, mutated, dtype, f"{arch} kernel vs {label}")


@pytest.mark.gpu
@pytest.mark.parametrize(
    "row,message",
    [(0, "outside the output"), (-1, "outside the output"), (3, "never written")],
    ids=["before", "after", "unwritten"],
)
def test_output_check_sees_a_stray_or_missing_write(row, message):
    """The harness's own check, no kernel: an output written in full passes, and
    one element changed in the guard before it, in the guard after it, or left NaN
    in it fails."""
    out = guarded((4, 8), "fp16", float("nan"), guard_fill=_GUARD_FILL)
    written = out.read()
    # Buffer rows 0-1 and 6-7 are the guards (16 elements a side), 2-5 the output.
    written[2:6] = storage.encode(np.ones((4, 8)), "fp16")
    out.write(written)
    _assert_written_inside(out)
    broken = written.copy()
    broken[row, 0] = storage.encode(
        0.0 if message != "never written" else np.nan, "fp16"
    )
    out.write(broken)
    with pytest.raises(AssertionError, match=message):
        _assert_written_inside(out)


@pytest.mark.gpu
def test_guard_check_refuses_an_array_without_guards():
    with pytest.raises(ValueError, match="guard_intact failed"):
        guard_intact(filled((4, 8), "fp16"), _GUARD_FILL)
