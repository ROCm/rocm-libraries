# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Stricter Torch-free checks of the shipped dense attention kernels (gfx942, gfx950).

The pinned reference suite compares one seed of one distribution on the O tensor
only. This file adds what that leaves open, against the float64 NumPy oracle
``rocke.numeric.sdpa_reference``:

* seeds and input distributions (uniform, large magnitude, near-constant K) with
  the worst row reported;
* an output buffer inside canary guards, so a write outside the tensor, or an
  element left unwritten (NaN prefill), fails;
* rows that see no key (O = 0, LSE = -inf) and extreme-but-legal magnitudes;
* cases through default dispatch, with no spec id pinned;
* a mutation check: the comparison rejects a wrong GQA mapping, a top-left mask
  where the kernel uses bottom-right, and a wrong softmax scale.

The mutation checks first run on the oracle alone (no GPU) and then on kernel
output.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from rocke.numeric.sdpa_reference import sdpa_reference
from rocke.runtime.hip_module import get_device_arch

from . import _attention_numpy as an
from ._attention_device_array import DeviceArray

_ARCH = get_device_arch()
_ARCHS = ("gfx942", "gfx950")

_gpu_marks = (
    pytest.mark.gpu,
    pytest.mark.skipif(
        _ARCH not in _ARCHS, reason=f"needs a gfx942 or gfx950 GPU, found {_ARCH}"
    ),
)


def requires_cdna_gpu(fn):
    for mark in _gpu_marks:
        fn = mark(fn)
    return fn


# A mutated comparison must miss the tolerance by at least this factor.
_MUTATION_MARGIN = 2.0

_GUARD = 4096
_GUARD_BITS = 0xA5A5
_UNWRITTEN_BITS = 0xFFFF


@pytest.fixture(scope="module")
def rt():
    return an.runtime()


# ---------------------------------------------------------------- oracle ----


def _oracle(q, k, v, scale, *, causal=True, bottom_right=False, window=0):
    """O ``[B, Sq, Hq, D]`` and LSE ``[B, Hq, Sq]`` (float64) for ``[B, S, H, D]``
    inputs. ``window`` keeps the ``window`` most recent keys of the causal band."""
    kwargs = {}
    if window:
        kwargs = {
            "left_bound": window - 1,
            "right_bound": 0,
            "top_left": not bottom_right,
        }
    elif causal:
        kwargs = {"causal_bottom_right": bottom_right, "causal": not bottom_right}
    o, lse = sdpa_reference(
        q.transpose(0, 2, 1, 3),
        k.transpose(0, 2, 1, 3),
        v.transpose(0, 2, 1, 3),
        scale=scale,
        **kwargs,
    )
    return o.transpose(0, 2, 1, 3), lse


def _max_err(got, want):
    return float(np.abs(np.asarray(got, np.float64) - want).max())


def _assert_rows_close(got, want, dtype, what):
    """``assert_close`` plus the number of rows over tolerance and the worst row."""
    tol = an.tolerance(dtype)
    row_err = np.abs(np.asarray(got, np.float64) - want).max(axis=(-1,))
    row_err = np.nan_to_num(row_err, nan=np.inf)
    bad = row_err >= tol
    assert not bad.any(), (
        f"{what}: {int(bad.sum())} of {bad.size} rows over {tol:.1e}; worst row "
        f"{tuple(np.unravel_index(int(np.argmax(row_err)), row_err.shape))} "
        f"max_abs={row_err.max():.3e}"
    )
    an.assert_close(got, want, dtype, what)


# ------------------------------------------------------- guarded output ----


class _GuardedOutput:
    """An output tensor in the middle of a larger device buffer.

    The element range before and after the tensor holds a canary, and the tensor
    itself starts as NaN bits, so a launch that writes outside the tensor, or skips
    an element, is caught by :meth:`read`.
    """

    is_cuda = True
    device = None

    def __init__(self, rt, shape, dtype):
        self.shape = tuple(shape)
        self._n = int(np.prod(shape))
        self._dtype = dtype
        flat = np.full(_GUARD + self._n + _GUARD, _GUARD_BITS, dtype=np.uint16)
        flat[_GUARD : _GUARD + self._n] = _UNWRITTEN_BITS
        self._buf = DeviceArray(
            rt, flat, torch_dtype="float16" if dtype == "fp16" else "bfloat16"
        )
        self.dtype = self._buf.dtype

    def stride(self, dim=None):
        strides = tuple(
            int(np.prod(self.shape[i + 1 :])) for i in range(len(self.shape))
        )
        return strides if dim is None else strides[dim]

    def data_ptr(self):
        return self._buf.data_ptr() + _GUARD * 2

    def is_contiguous(self):
        return True

    def numel(self):
        return self._n

    def element_size(self):
        return 2

    def dim(self):
        return len(self.shape)

    def read(self):
        return check_guards(self._buf.read(), self._n, self._dtype).reshape(self.shape)


def check_guards(flat, n, dtype):
    """The tensor part of ``flat`` as fp32, after checking both guards are intact
    and no element of the tensor is still the unwritten NaN pattern."""
    before, body, after = flat[:_GUARD], flat[_GUARD : _GUARD + n], flat[_GUARD + n :]
    assert (before == _GUARD_BITS).all(), (
        f"{int((before != _GUARD_BITS).sum())} elements before the output were "
        "overwritten"
    )
    assert (after == _GUARD_BITS).all(), (
        f"{int((after != _GUARD_BITS).sum())} elements after the output were "
        "overwritten"
    )
    unwritten = body == _UNWRITTEN_BITS
    assert not unwritten.any(), f"{int(unwritten.sum())} output elements never written"
    return an.decode(body.view(np.float16) if dtype == "fp16" else body, dtype)


# ------------------------------------------------------------ launching ----


def _request(dtype, d, hq, hkv, *, batch, sq, sk, mask_type, window=0):
    from dispatch.attention import AttentionRequest

    return AttentionRequest(
        batch=batch,
        nhead_q=hq,
        nhead_k=hkv,
        seqlen_q=sq,
        seqlen_k=sk,
        hdim_q=d,
        hdim_v=d,
        arch=_ARCH,
        mask_type=mask_type,
        dtype=dtype,
        sliding_window=window,
        layout="dense",
    )


def _shipped_spec(
    dtype, d, hq, hkv, *, batch=1, sq=512, sk=None, bottom_right=False, window=0
):
    """The shipped default-grid dense spec for the row, built through its dispatch
    factory (the spec the pinned suites run), not hand-rolled."""
    sk = sq if sk is None else sk
    mask_type = 2 if bottom_right else 1
    request = _request(
        dtype, d, hq, hkv, batch=batch, sq=sq, sk=sk, mask_type=mask_type, window=window
    )
    if _ARCH == "gfx942":
        from dispatch.attention import tuning_spec_with_knobs

        # One KV segment: the split path allocates its workspace with torch.
        return tuning_spec_with_knobs(
            request, "gfx942_dense", {"persistent": False, "kv_segments": 1}
        ).kernel_spec
    from dispatch.attention import attention_tuning_spec

    spec = attention_tuning_spec(request, "gfx950_dense_grid").kernel_spec
    return dataclasses.replace(spec, kv_segments=1)


def _run_dense(spec, rt, q, k, v, dtype, *, scale, with_lse=False):
    """Launch the dense kernel of ``spec``; returns fp32 O (and LSE)."""
    if _ARCH == "gfx942":
        from kernels.gfx942.attention_dense import run_attention_dense_torch
    else:
        from kernels.gfx950.attention_dense import run_attention_dense_torch

    out = _GuardedOutput(rt, q.shape, dtype)
    extra = {}
    if with_lse:
        extra["lse"] = an.lse_device(rt, (q.shape[0], q.shape[2], q.shape[1], 1))
    run_attention_dense_torch(
        spec=spec,
        q=an.to_device(rt, q, dtype),
        k=an.to_device(rt, k, dtype),
        v=an.to_device(rt, v, dtype),
        out=out,
        scale=scale,
        **extra,
    )
    o = out.read()
    if with_lse:
        return o, extra["lse"].read()[..., 0]
    return o


def _inputs(rng, dtype, batch, sq, sk, hq, hkv, d, *, kind="normal"):
    """fp32 ``[B, S, H, D]`` Q, K, V, already rounded to ``dtype``."""
    shapes = ((batch, sq, hq, d), (batch, sk, hkv, d), (batch, sk, hkv, d))
    if kind == "normal":
        q, k, v = (an.randn(rng, s, dtype) for s in shapes)
    elif kind == "uniform":
        q, k, v = (an.round_to(rng.uniform(-1.0, 1.0, s), dtype) for s in shapes)
    elif kind == "large":
        q, k = (an.randn(rng, s, dtype, scale=4.0) for s in shapes[:2])
        v = an.randn(rng, shapes[2], dtype, scale=4.0)
    elif kind == "near_constant_k":
        q = an.randn(rng, shapes[0], dtype)
        k = an.round_to(1.0 + 1e-3 * rng.standard_normal(shapes[1]), dtype)
        v = an.randn(rng, shapes[2], dtype)
    else:
        raise ValueError(kind)
    return q, k, v


# ------------------------------------------------- (a) seeds, distributions --

_SWEEP_KINDS = [
    ("normal", 0),
    ("normal", 1),
    ("normal", 2),
    ("normal", 3),
    ("uniform", 4),
    ("large", 5),
    ("near_constant_k", 6),
]


@pytest.mark.parametrize("kind,seed", _SWEEP_KINDS, ids=lambda x: str(x))
@requires_cdna_gpu
def test_seed_and_distribution_sweep(rt, kind, seed):
    """One binary, many inputs: seeds and distributions the single pinned seed
    does not reach, compared row by row with the float64 oracle."""
    dtype, d, hq, hkv, s = "fp16", 128, 16, 4, 512
    spec = _shipped_spec(dtype, d, hq, hkv, sq=s)
    rng = np.random.default_rng(seed)
    q, k, v = _inputs(rng, dtype, 1, s, s, hq, hkv, d, kind=kind)
    scale = 1.0 / math.sqrt(d)
    got = _run_dense(spec, rt, q, k, v, dtype, scale=scale)
    want, _ = _oracle(q, k, v, scale)
    _assert_rows_close(
        got, want, dtype, f"{_ARCH} fp16 D128 GQA16/4 {kind} seed {seed}"
    )


# ------------------------------------------------- (c) keyless rows, LSE ----

# (dtype, d, hq, hkv, sq, sk, window): rows from sk + window - 1 on see no key.
_KEYLESS = {
    "gfx942": ("bf16", 128, 16, 4, 512, 256, 64),
    "gfx950": ("bf16", 128, 4, 1, 256, 64, 64),
}


@requires_cdna_gpu
def test_keyless_rows_write_zero_output_and_neg_inf_lse(rt):
    dtype, d, hq, hkv, sq, sk, window = _KEYLESS[_ARCH]
    spec = dataclasses.replace(
        _shipped_spec(dtype, d, hq, hkv, sq=sq, sk=sk, window=window), emit_lse=True
    )
    rng = np.random.default_rng(7)
    q, k, v = _inputs(rng, dtype, 1, sq, sk, hq, hkv, d)
    scale = 1.0 / math.sqrt(d)
    got, got_lse = _run_dense(spec, rt, q, k, v, dtype, scale=scale, with_lse=True)
    want, want_lse = _oracle(q, k, v, scale, window=window)
    first_keyless = sk + window - 1
    assert first_keyless < sq, "setup error: no keyless rows"
    assert np.isneginf(want_lse[:, :, first_keyless:]).all()
    assert (
        np.count_nonzero(got[:, first_keyless:]) == 0
    ), f"rows >= {first_keyless} see no key but are not all 0"
    _assert_rows_close(got, want, dtype, f"{_ARCH} keyless O")
    an.assert_close(got_lse, want_lse, dtype, f"{_ARCH} keyless LSE")


@pytest.mark.parametrize(
    "q_scale,k_scale,v_scale",
    [(8.0, 8.0, 1.0), (1.0, 1.0, 4.0), (32.0, 1.0, 1.0)],
    ids=["peaky-scores", "large-values", "huge-q"],
)
@requires_cdna_gpu
def test_extreme_legal_magnitudes_stay_finite_and_correct(
    rt, q_scale, k_scale, v_scale
):
    """Large scores (near one-hot softmax) and large V must not overflow or lose the
    row maximum: the output stays finite and inside tolerance."""
    dtype, d, hq, hkv, s = "fp16", 128, 16, 4, 512
    spec = _shipped_spec(dtype, d, hq, hkv, sq=s)
    rng = np.random.default_rng(11)
    q = an.randn(rng, (1, s, hq, d), dtype, scale=q_scale)
    k = an.randn(rng, (1, s, hkv, d), dtype, scale=k_scale)
    v = an.randn(rng, (1, s, hkv, d), dtype, scale=v_scale)
    scale = 1.0 / math.sqrt(d)
    got = _run_dense(spec, rt, q, k, v, dtype, scale=scale)
    assert np.isfinite(got).all(), "non-finite output for finite inputs"
    want, _ = _oracle(q, k, v, scale)
    _assert_rows_close(
        got, want, dtype, f"{_ARCH} magnitudes q{q_scale} k{k_scale} v{v_scale}"
    )


# ------------------------------------------------- (d) default dispatch ----

# (dtype, d, hq, hkv, batch, seq, mask_type, window)
_DEFAULT_DISPATCH = [
    pytest.param("fp16", 128, 16, 4, 1, 512, 1, 0, id="fp16-d128-gqa-causal"),
    pytest.param("bf16", 128, 16, 4, 1, 512, 1, 128, id="bf16-d128-gqa-window128"),
    pytest.param("fp16", 64, 16, 16, 2, 256, 0, 0, id="fp16-d64-mha-b2-full"),
]


@pytest.mark.parametrize("dtype,d,hq,hkv,batch,s,mask_type,window", _DEFAULT_DISPATCH)
@requires_cdna_gpu
def test_default_dispatch_matches_oracle(
    rt, dtype, d, hq, hkv, batch, s, mask_type, window
):
    """No spec id is pinned: dispatch picks the kernel, binds and launches it."""
    from dispatch.attention import dispatch_attention

    request = _request(
        dtype, d, hq, hkv, batch=batch, sq=s, sk=s, mask_type=mask_type, window=window
    )
    assert request.spec_id == "auto" and request.algorithm == "auto"
    result = dispatch_attention(request)
    rng = np.random.default_rng(21)
    q, k, v = _inputs(rng, dtype, batch, s, s, hq, hkv, d)
    out = _GuardedOutput(rt, q.shape, dtype)
    result.bind_torch(
        {
            "q": an.to_device(rt, q, dtype),
            "k": an.to_device(rt, k, dtype),
            "v": an.to_device(rt, v, dtype),
            "out": out,
        }
    ).launch()
    want, _ = _oracle(
        q, k, v, 1.0 / math.sqrt(d), causal=bool(mask_type), window=window
    )
    _assert_rows_close(
        out.read(),
        want,
        dtype,
        f"{_ARCH} default dispatch -> {result.candidate.spec_id}",
    )


# ------------------------------------------------------ (e) mutation checks --


def _gqa_interleaved(k, hq):
    """``k`` expanded to ``hq`` heads with the wrong grouping: head ``h`` reads KV
    head ``h % hkv`` instead of ``h // (hq // hkv)``."""
    hkv = k.shape[2]
    return k[:, :, [h % hkv for h in range(hq)]]


def _mutations(q, k, v, scale, hq, *, bottom_right):
    """Oracle outputs under one deliberate bug each, for the same inputs."""
    return {
        "gqa-interleaved": _oracle(
            q,
            _gqa_interleaved(k, hq),
            _gqa_interleaved(v, hq),
            scale,
            bottom_right=bottom_right,
        )[0],
        "mask-offset": _oracle(q, k, v, scale, bottom_right=not bottom_right)[0],
        "scale": _oracle(q, k, v, scale * 1.05, bottom_right=bottom_right)[0],
    }


def _assert_rejected(good, mutated, dtype, label):
    tol = an.tolerance(dtype)
    err = _max_err(good, mutated)
    print(f"mutation margin {label}: max_abs={err:.3e} = {err / tol:.1f} x tolerance")
    assert err >= _MUTATION_MARGIN * tol, (
        f"{label}: the bug moves the output by only {err:.3e}, under "
        f"{_MUTATION_MARGIN:g} x tolerance {tol:.1e}"
    )
    with pytest.raises(AssertionError):
        an.assert_close(good, mutated, dtype, label)


@pytest.mark.parametrize("dtype", an.DTYPES)
def test_comparison_rejects_known_bugs_on_oracle_output(dtype):
    """Host only. Stand in for a correct kernel with the oracle rounded to ``dtype``;
    the comparison accepts it and rejects each deliberate bug."""
    rng = np.random.default_rng(3)
    hq, hkv, d, sq, sk = 8, 2, 64, 128, 256
    q, k, v = _inputs(rng, dtype, 1, sq, sk, hq, hkv, d)
    scale = 1.0 / math.sqrt(d)
    good, _ = _oracle(q, k, v, scale, bottom_right=True)
    kernel_like = an.round_to(good, dtype)
    an.assert_close(kernel_like, good, dtype, "unmutated")
    for label, mutated in _mutations(q, k, v, scale, hq, bottom_right=True).items():
        _assert_rejected(kernel_like, mutated, dtype, f"{dtype} {label}")


def test_output_guard_rejects_overwrite_and_unwritten_elements():
    """Host only. The guard check passes a clean buffer and fails each violation."""
    n = 64
    clean = np.full(_GUARD + n + _GUARD, _GUARD_BITS, dtype=np.uint16)
    clean[_GUARD : _GUARD + n] = an.encode(np.ones(n), "fp16").view(np.uint16)
    assert (check_guards(clean, n, "fp16") == 1.0).all()
    for index, message in (
        (_GUARD - 1, "before the output"),
        (_GUARD + n, "after the output"),
        (_GUARD + n // 2, "never written"),
    ):
        broken = clean.copy()
        broken[index] = 0 if message != "never written" else _UNWRITTEN_BITS
        with pytest.raises(AssertionError, match=message):
            check_guards(broken, n, "fp16")


# (dtype, d, hq, hkv, sq, sk): moving diagonal, so top-left and bottom-right differ.
_MUTATION_GPU = ("fp16", 128, 16, 4, 256, 512)


@requires_cdna_gpu
def test_comparison_rejects_known_bugs_on_kernel_output(rt):
    """GPU. The kernel is right against the right oracle; the same kernel output is
    rejected by the oracle with a wrong GQA mapping, a top-left mask in place of the
    kernel's bottom-right one, or a wrong scale."""
    dtype, d, hq, hkv, sq, sk = _MUTATION_GPU
    spec = _shipped_spec(dtype, d, hq, hkv, sq=sq, sk=sk, bottom_right=True)
    rng = np.random.default_rng(13)
    q, k, v = _inputs(rng, dtype, 1, sq, sk, hq, hkv, d)
    scale = 1.0 / math.sqrt(d)
    got = _run_dense(spec, rt, q, k, v, dtype, scale=scale)
    want, _ = _oracle(q, k, v, scale, bottom_right=True)
    _assert_rows_close(got, want, dtype, f"{_ARCH} bottom-right {sq}x{sk}")
    for label, mutated in _mutations(q, k, v, scale, hq, bottom_right=True).items():
        _assert_rejected(got, mutated, dtype, f"{_ARCH} kernel vs {label}")
