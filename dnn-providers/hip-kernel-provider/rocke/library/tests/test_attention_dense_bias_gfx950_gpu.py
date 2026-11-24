# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Torch-free on-GPU checks of the gfx950 dense kernel's additive bias.

The CI GPU lane has no torch, so the full matrix in
``test_attention_dense_bias_gfx950_numeric.py`` (driven through
``run_attention_dense_torch``) skips there. This file launches the same kernels
through ``rocke.runtime`` from numpy arrays and checks a representative subset
against the float64 oracle in ``_dense_bias_common`` (inputs, 16-bit storage,
tolerances and device buffers are shared with the gfx942 lane): one bias shape per mask, ragged lengths,
packed varlen with and without LSE, ``-inf`` keys, ``finfo(float32).min`` rows (the
uniform softmax) and NaN keys (a NaN row), rows with no live key with and without
LSE, and window q-blocks past the keys (dead rows store zeros).
That case checks the output only: a read past the bias there is overwritten by the
window mask and cannot change it, so the start-tile clamp itself is pinned by the
representative IR golden (attention_dense/gfx950/bias_*window*), not by this file.

Specs come from the dispatch factory (block_m 256); the hipDNN catalog ships the
bias at 128/64, so the per-mask test also runs block_m 128 (at the factory's block
order; the torch matrix runs the catalog's exact configuration). The launch packs
the kernargs the way ``run_attention_dense_torch`` does, minus its host checks.

    HIP_VISIBLE_DEVICES=0 python -m pytest tests/test_attention_dense_bias_gfx950_gpu.py

Set ``ROCKE_REQUIRE_GFX950=1`` to fail instead of skip without a gfx950.
"""

from __future__ import annotations

import math
import os
from dataclasses import replace

import numpy as np
import pytest

from _dense_bias_common import (
    WINDOW,
    band_kwargs,
    bias_strides,
    device,
    from_bits,
    make_inputs,
    max_abs,
    nan_filled,
    reference,
    round_to,
    tolerance,
    to_bits,
)
from dispatch.attention import AttentionRequest, attention_tuning_spec
from kernels.common._attention_additive_bias import additive_bias_launch_values
from kernels.common.attention_dense_spec import attention_dense_cache_key
from kernels.gfx950.attention_dense import (
    attention_dense_block,
    attention_dense_grid,
    attention_dense_runtime_args,
    attention_dense_signature,
    build_attention_dense,
)
from rocke.helpers.compile import compile_kernel
from rocke.numeric.sdpa_reference import sdpa_reference
from rocke.runtime import KernelLauncher, LaunchConfig
from rocke.runtime.hip_module import Runtime, get_device_arch
from rocke.runtime.host_buffers import as_u8_buffer

pytestmark = pytest.mark.gpu

_MASK_TYPES = {"none": 0, "top_left": 1, "bottom_right": 2, "window": 3}
_LAUNCHERS: dict = {}


@pytest.fixture(scope="module", autouse=True)
def gfx950():
    arch = get_device_arch()
    if arch == "gfx950":
        return
    if os.environ.get("ROCKE_REQUIRE_GFX950") == "1":
        pytest.fail(f"ROCKE_REQUIRE_GFX950=1 but the device is {arch or 'absent'}")
    pytest.skip("needs a gfx950 GPU")


def _spec(dtype, d, hq, hkv, *, batch, sq, skv, mask, lse=False):
    """The shipped grid-body spec for a bias request, from the dispatch factory."""
    extra = {"sliding_window": WINDOW, "right_bound": 0} if mask == "window" else {}
    req = AttentionRequest(
        batch=batch,
        nhead_q=hq,
        nhead_k=hkv,
        seqlen_q=sq,
        seqlen_k=skv,
        hdim_q=d,
        hdim_v=d,
        arch="gfx950",
        mask_type=_MASK_TYPES[mask],
        dtype=dtype,
        emit_lse=lse,
        use_attn_bias=True,
        **extra,
    )
    spec = attention_tuning_spec(req, "gfx950_dense_grid").kernel_spec
    assert spec.use_additive_bias and not spec.persistent
    return spec


def _launch(spec, q, k, v, bias, scale, *, cu_q=None, cu_kv=None, bias_view=None):
    """Launch ``spec`` on float32 arrays of ``spec.dtype`` values: ``q``/``k``/``v``
    dense [B, S, H, D] or packed [T, H, D] (varlen), ``bias`` a contiguous rank-4
    fp32 array. ``bias_view = (storage, offset, strides)`` instead passes a strided
    view: a 1-D fp32 ``storage``, the element ``offset`` of the view in it and the
    view's ``(batch, head, sq)`` element strides (``bias`` is then ignored). Returns
    ``(O, LSE or None)`` as float32 arrays in the kernel's layouts (LSE [B, Hq, Sq]
    or [T, Hq])."""
    key = attention_dense_cache_key(spec, arch="gfx950")
    launcher = _LAUNCHERS.get(key)
    if launcher is None:
        art = compile_kernel(
            build_attention_dense(spec, arch="gfx950"),
            arch="gfx950",
            backend="python",
            capture_ir_text=False,
        )
        launcher = KernelLauncher(
            hsaco=art.hsaco,
            kernel_name=art.kernel_name,
            signature=attention_dense_signature(spec),
        )
        _LAUNCHERS[key] = launcher
    out_bits = np.empty(q.shape, np.uint16)
    out = nan_filled(out_bits.nbytes)
    vals = {
        "q_ptr": device(to_bits(q, spec.dtype)),
        "k_ptr": device(to_bits(k, spec.dtype)),
        "v_ptr": device(to_bits(v, spec.dtype)),
        "o_ptr": out,
        "scale": float(scale),
    }
    vals.update(attention_dense_runtime_args(spec))
    if spec.varlen:
        vals["cu_seqlens_q"] = device(np.asarray(cu_q, np.int32))
        vals["cu_seqlens_kv"] = device(np.asarray(cu_kv, np.int32))
    lse = None
    if spec.emit_lse:
        lse_shape = (
            (q.shape[0], spec.num_query_heads)
            if spec.varlen
            else (spec.batch, spec.num_query_heads, spec.seqlen_q)
        )
        lse = np.empty(lse_shape, np.float32)
        lse_mem = nan_filled(lse.nbytes)
        vals["lse_ptr"] = lse_mem
    if bias_view is None:
        bias = np.ascontiguousarray(bias, np.float32)
        vals.update(additive_bias_launch_values(device(bias), bias_strides(bias)))
    else:
        storage, offset, strides = bias_view
        bias_mem = device(np.asarray(storage, np.float32))
        # A raw pointer into bias_mem, which this frame keeps alive until the
        # synchronous launch returns.
        vals.update(
            additive_bias_launch_values(bias_mem.ptr() + 4 * int(offset), strides)
        )
    # A synchronous launch: the kernel has finished when this returns.
    launcher(
        vals,
        config=LaunchConfig(
            grid=attention_dense_grid(spec), block=attention_dense_block(spec)
        ),
    )
    Runtime().memcpy_d2h(as_u8_buffer(out_bits), out.ptr(), out_bits.nbytes)
    if lse is not None:
        Runtime().memcpy_d2h(as_u8_buffer(lse), lse_mem.ptr(), lse.nbytes)
    return from_bits(out_bits, spec.dtype), lse


@pytest.mark.parametrize("block_m", [256, 128], ids=["factory-tile", "catalog-tile"])
@pytest.mark.parametrize(
    "mask, bias_shape",
    [
        ("none", "full"),
        ("top_left", "head_bcast"),
        ("bottom_right", "batch_bcast"),
        ("window", "key_row"),
    ],
)
def test_one_bias_shape_per_mask(mask, bias_shape, block_m):
    """bf16 D128 GQA-4, B=2; bottom-right runs Sq != Skv so its diagonal moves. The
    bias must also move the output far past the tolerance, or a kernel that ignored
    it would pass."""
    dtype, d, hq, hkv, batch = "bf16", 128, 8, 2, 2
    sq, skv = (256, 512) if mask == "bottom_right" else (512, 512)
    shape = {
        "full": (batch, hq, sq, skv),
        "head_bcast": (batch, 1, sq, skv),
        "batch_bcast": (1, hq, sq, skv),
        "key_row": (1, 1, 1, skv),
    }[bias_shape]
    spec = replace(
        _spec(dtype, d, hq, hkv, batch=batch, sq=sq, skv=skv, mask=mask),
        block_m=block_m,
    )
    q, k, v, bias = make_inputs(dtype, d, hq, hkv, batch, sq, skv, shape)
    scale = 1.0 / math.sqrt(d)
    out, _ = _launch(spec, q, k, v, bias, scale)
    ref, _ = reference(q, k, v, scale, bias, mask)
    assert max_abs(out, ref) < tolerance(dtype)
    plain, _ = reference(q, k, v, scale, None, mask)
    assert max_abs(out, plain) > 5 * tolerance(dtype)


@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("mask", ["none", "top_left"])
def test_misaligned_bias_view(mask, d):
    """The aligned body loads four consecutive keys with one 16-byte load that is
    only element-aligned: a view one element into its storage, with an odd row
    pitch, puts most of those loads off a 16-byte boundary. ``-inf`` entries ride
    along so a lane mix-up inside a load shows."""
    dtype, hq, hkv, batch, sq, skv = "bf16", 4, 2, 2, 256, 512
    spec = _spec(dtype, d, hq, hkv, batch=batch, sq=sq, skv=skv, mask=mask)
    assert not spec.ragged
    q, k, v, _ = make_inputs(dtype, d, hq, hkv, batch, sq, skv, (1,))
    pitch = skv + 5
    strides = (hq * sq * pitch, sq * pitch, pitch)
    rng = np.random.default_rng(1)
    storage = (2 * rng.standard_normal(1 + batch * hq * sq * pitch)).astype(np.float32)
    view = np.lib.stride_tricks.as_strided(
        storage[1:],
        shape=(batch, hq, sq, skv),
        strides=tuple(4 * s for s in (*strides, 1)),
        writeable=True,
    )
    view[..., 3::7] = -np.inf
    # Group loads start at element 1 + s*pitch + 4m of a 16-byte-aligned device
    # allocation (the batch and head terms are multiples of 4): every 4-byte residue.
    assert {(1 + r * pitch) % 4 for r in range(4)} == {0, 1, 2, 3}
    scale = 1.0 / math.sqrt(d)
    out, _ = _launch(spec, q, k, v, None, scale, bias_view=(storage, 1, strides))
    ref, _ = reference(q, k, v, scale, np.array(view), mask)
    assert max_abs(out, ref) < tolerance(dtype)


@pytest.mark.parametrize("lse", [False, True], ids=["no-lse", "lse"])
@pytest.mark.parametrize("d", [64, 128])
def test_vector_bias_loads_match_the_scalar_path_bitwise(d, lse):
    """At tile-multiple lengths the ragged body (masked scalar bias loads) reads the
    same keys as the aligned body (16-byte loads) with the same clamp, scale and add,
    so the outputs (and the LSE) are bitwise equal: on the misaligned view of the test
    above. D128 with LSE keeps scalar loads in both bodies (a cheap control)."""
    dtype, hq, hkv, batch, sq, skv = "bf16", 4, 2, 2, 256, 512
    spec = _spec(
        dtype, d, hq, hkv, batch=batch, sq=sq, skv=skv, mask="top_left", lse=lse
    )
    assert not spec.ragged
    q, k, v, _ = make_inputs(dtype, d, hq, hkv, batch, sq, skv, (1,))
    pitch = skv + 5
    strides = (hq * sq * pitch, sq * pitch, pitch)
    rng = np.random.default_rng(1)
    storage = (2 * rng.standard_normal(1 + batch * hq * sq * pitch)).astype(np.float32)
    storage[1 + 3 :: 7] = -np.inf
    scale = 1.0 / math.sqrt(d)
    view = (storage, 1, strides)
    vec, vec_lse = _launch(spec, q, k, v, None, scale, bias_view=view)
    scalar, scalar_lse = _launch(
        replace(spec, ragged=True), q, k, v, None, scale, bias_view=view
    )
    np.testing.assert_array_equal(vec, scalar)
    if lse:
        np.testing.assert_array_equal(vec_lse, scalar_lse)


@pytest.mark.parametrize(
    "mask, bias_shape", [("none", "full"), ("bottom_right", "key_row")]
)
def test_ragged_lengths(mask, bias_shape):
    """Non-tile-multiple lengths take the ragged body, whose padded rows and keys
    load the bias masked."""
    dtype, d, hq, hkv, batch, sq, skv = "fp16", 128, 8, 8, 2, 300, 700
    shape = (batch, hq, sq, skv) if bias_shape == "full" else (1, 1, 1, skv)
    spec = _spec(dtype, d, hq, hkv, batch=batch, sq=sq, skv=skv, mask=mask)
    assert spec.ragged
    q, k, v, bias = make_inputs(dtype, d, hq, hkv, batch, sq, skv, shape)
    scale = 1.0 / math.sqrt(d)
    out, _ = _launch(spec, q, k, v, bias, scale)
    ref, _ = reference(q, k, v, scale, bias, mask)
    assert max_abs(out, ref) < tolerance(dtype)


@pytest.mark.parametrize("lse", [False, True], ids=["no-lse", "lse"])
@pytest.mark.parametrize("mask", ["none", "top_left"])
@pytest.mark.parametrize("d", [64, 128])
def test_rows_are_independent_of_the_other_rows_of_their_wave(d, mask, lse):
    """The lazy rescale keeps or moves each row's running max on that row's own scores;
    the wave skips the O / l multiply only when every row keeps its max. A bias bumped
    by +400 at the last key of the last query row alone moves that row's max and no
    other row's, so every other row's output (and LSE) must be bit-identical to the run
    without the bump, and the last row must change. A wave-wide decision would move the
    max of the 31 rows that share the last row's wave and change their output bits."""
    dtype, hq, hkv, batch, sq, skv = "bf16", 4, 2, 1, 512, 512
    spec = _spec(dtype, d, hq, hkv, batch=batch, sq=sq, skv=skv, mask=mask, lse=lse)
    assert spec.lazy_rescale and not spec.ragged
    q, k, v, bias = make_inputs(dtype, d, hq, hkv, batch, sq, skv, (batch, hq, sq, skv))
    bumped = bias.copy()
    bumped[:, :, sq - 1, skv - 1] += 400.0
    scale = 1.0 / math.sqrt(d)
    base, base_lse = _launch(spec, q, k, v, bias, scale)
    out, out_lse = _launch(spec, q, k, v, bumped, scale)
    np.testing.assert_array_equal(out[:, : sq - 1], base[:, : sq - 1])
    if lse:
        np.testing.assert_array_equal(out_lse[..., : sq - 1], base_lse[..., : sq - 1])
    assert max_abs(out[:, sq - 1], base[:, sq - 1]) > 5 * tolerance(dtype)


@pytest.mark.parametrize("lse", [False, True], ids=["no-lse", "lse"])
def test_rows_with_no_live_key_store_zero(lse):
    """Rows 0..63 see only keys 0..q under the causal mask, all -inf: they store
    O = 0 and LSE = -inf, the hipDNN reference's answer for a row of -inf scores."""
    dtype, d, hq, hkv, batch, sq, skv = "fp16", 128, 4, 4, 1, 256, 256
    spec = _spec(
        dtype, d, hq, hkv, batch=batch, sq=sq, skv=skv, mask="top_left", lse=lse
    )
    q, k, v, bias = make_inputs(dtype, d, hq, hkv, batch, sq, skv, (batch, hq, sq, skv))
    bias[:, :, :64, :64] = -np.inf
    scale = 1.0 / math.sqrt(d)
    out, got_lse = _launch(spec, q, k, v, bias, scale)
    ref, ref_lse = reference(q, k, v, scale, bias, "top_left")
    assert np.isfinite(out).all()
    assert np.abs(out[:, :64]).max() == 0.0
    assert max_abs(out, ref) < tolerance(dtype)
    if lse:
        assert np.isneginf(got_lse[:, :, :64]).all()
        assert max_abs(got_lse[:, :, 64:], ref_lse[:, :, 64:]) < tolerance(dtype)


@pytest.mark.parametrize("lse", [False, True], ids=["no-lse", "lse"])
@pytest.mark.parametrize("mask", ["none", "top_left"])
def test_finfo_min_rows_are_the_uniform_softmax(mask, lse):
    """A row whose every visible key has bias finfo(float32).min (HF's padding value)
    has no key above the others: O is the mean of V over the visible keys, as in the
    reference and PyTorch (here head 1, every row; V has mean 1, so zeros fail), and
    the LSE is finite, about the kernel's -2**97 clamp. Head 2 mixes finfo.min on
    even keys with -inf on odd ones (their V shifted by 10, for that head only): uniform
    over the even keys only, -inf and finfo.min being different keys."""
    dtype, d, hq, hkv, batch, sq, skv = "bf16", 128, 4, 4, 1, 256, 256
    spec = _spec(dtype, d, hq, hkv, batch=batch, sq=sq, skv=skv, mask=mask, lse=lse)
    q, k, v, bias = make_inputs(dtype, d, hq, hkv, batch, sq, skv, (batch, hq, sq, skv))
    v = round_to(v + 1.0, dtype)
    v[:, 1::2, 2] = round_to(v[:, 1::2, 2] + 10.0, dtype)
    lowest = np.finfo(np.float32).min
    bias[:, 1] = lowest
    bias[:, 2, :, 0::2] = lowest
    bias[:, 2, :, 1::2] = -np.inf
    scale = 1.0 / math.sqrt(d)
    out, got_lse = _launch(spec, q, k, v, bias, scale)
    ref, ref_lse = reference(q, k, v, scale, bias, mask)
    assert np.isfinite(out).all()
    assert np.abs(ref[:, :, 1:3]).mean() > 0.5  # the rows under test are not zero
    assert max_abs(out, ref) < tolerance(dtype)
    if lse:
        assert np.isfinite(got_lse[:, 1:3]).all()
        assert (got_lse[:, 1] < -1e28).all()
        assert max_abs(got_lse[:, [0, 3]], ref_lse[:, [0, 3]]) < tolerance(dtype)


@pytest.mark.parametrize("lse", [False, True], ids=["no-lse", "lse"])
@pytest.mark.parametrize("mask", ["none", "top_left"])
def test_a_nan_bias_makes_its_row_nan(mask, lse):
    """A NaN bias on a visible key makes its whole row NaN (O and LSE), as in the
    reference. Head 0 row 10 has one NaN key (key 7, visible under every mask), head 2
    row 20 has NaN on every key (the max ignores a NaN score, so only the NaN
    denominator keeps the row from reading as dead), and head 1 row 5 has a NaN at
    key 200: visible only without the mask, so under the causal mask the key is
    skipped, whatever its bias, and the row stays finite. Every other row matches the
    reference."""
    dtype, d, hq, hkv, batch, sq, skv = "fp16", 128, 4, 4, 1, 256, 256
    spec = _spec(dtype, d, hq, hkv, batch=batch, sq=sq, skv=skv, mask=mask, lse=lse)
    q, k, v, bias = make_inputs(dtype, d, hq, hkv, batch, sq, skv, (batch, hq, sq, skv))
    bias[0, 0, 10, 7] = np.nan
    bias[0, 2, 20, :] = np.nan
    bias[0, 1, 5, 200] = np.nan
    expect_nan = np.zeros((batch, sq, hq), bool)  # [B, Sq, Hq]
    expect_nan[0, 10, 0] = expect_nan[0, 20, 2] = True
    expect_nan[0, 5, 1] = mask == "none"
    scale = 1.0 / math.sqrt(d)
    out, got_lse = _launch(spec, q, k, v, bias, scale)
    ref, ref_lse = reference(
        q, k, v, scale, np.where(np.isnan(bias), -np.inf, bias), mask
    )
    nan_out = np.isnan(out)
    np.testing.assert_array_equal(nan_out.all(-1), expect_nan)
    assert not (nan_out.any(-1) & ~expect_nan).any()
    live = ~expect_nan
    assert np.abs(out - ref).max(-1)[live].max() < tolerance(dtype)
    if lse:
        got = got_lse.transpose(0, 2, 1)  # [B, Sq, Hq]
        np.testing.assert_array_equal(np.isnan(got), expect_nan)
        diff = np.abs(got - ref_lse.transpose(0, 2, 1))
        assert diff[live].max() < tolerance(dtype)


@pytest.mark.parametrize("lse", [False, True], ids=["no-lse", "lse"])
def test_window_rows_past_the_keys(lse):
    """Sq > Skv + W - 1: the last q-blocks have no visible key. Rows q >= Skv + W - 1
    store O = 0 (and LSE = -inf); the rest match the reference. no-lse runs the
    EMPTY_ROWS start-tile clamp, lse the bias-only one (EMPTY_ROWS is off with LSE).

    This pins the outputs of the clamped path, not the clamp: an unclamped start
    tile reads about 3 KB past the bias, which does not fault, and those values are
    masked and the rows zeroed, so the output would not change. The representative
    IR golden (bias_lse_window_past_keys_sq512, bias_varlen_window_sq512) is what
    fails if the clamp is removed."""
    dtype, d, hq, hkv, batch, sq, skv = "bf16", 128, 4, 4, 1, 1024, 256
    spec = _spec(dtype, d, hq, hkv, batch=batch, sq=sq, skv=skv, mask="window", lse=lse)
    q, k, v, bias = make_inputs(dtype, d, hq, hkv, batch, sq, skv, (1, 1, 1, skv))
    scale = 1.0 / math.sqrt(d)
    out, got_lse = _launch(spec, q, k, v, bias, scale)
    ref, ref_lse = reference(q, k, v, scale, bias, "window")
    dead = skv + WINDOW - 1
    assert np.abs(out[:, dead:]).max() == 0.0
    assert max_abs(out, ref) < tolerance(dtype)
    if lse:
        assert np.isneginf(got_lse[:, :, dead:]).all()
        assert max_abs(got_lse[:, :, :dead], ref_lse[:, :, :dead]) < tolerance(dtype)


@pytest.mark.parametrize(
    "mask, lse, q_lens, kv_lens",
    [
        ("top_left", False, (256, 512), (128, 256)),
        ("top_left", True, (256, 512), (128, 256)),
        ("window", False, (1024, 512), (256, 128)),
        # A first and a last slot with no queries and no keys: the kernel returns
        # on the query-block bound before it loads K/V, so neither is read.
        ("top_left", False, (0, 256, 512, 0), (0, 128, 256, 0)),
        ("top_left", True, (0, 256, 512, 0), (0, 128, 256, 0)),
    ],
    ids=["causal", "causal-lse", "window", "idle-slots", "idle-slots-lse"],
)
def test_varlen(mask, lse, q_lens, kv_lens):
    """Packed varlen reads bias rows ``[s, h, 0:len_q, 0:len_kv]`` by in-sequence
    position; with the window, each sequence's rows past its own keys + W - 1 store
    exact zeros."""
    dtype, d, hq, hkv = "bf16", 128, 4, 4
    q_off = [0, *np.cumsum(q_lens).tolist()]
    kv_off = [0, *np.cumsum(kv_lens).tolist()]
    nseq, max_q, max_kv = len(q_lens), max(q_lens), max(kv_lens)
    base = _spec(
        dtype, d, hq, hkv, batch=nseq, sq=max_q, skv=max_kv, mask=mask, lse=lse
    )
    spec = replace(base, varlen=True, ragged=False)
    rng = np.random.default_rng(5)
    q = round_to(rng.standard_normal((q_off[-1], hq, d)), dtype)
    k = round_to(rng.standard_normal((kv_off[-1], hkv, d)), dtype)
    v = round_to(rng.standard_normal((kv_off[-1], hkv, d)), dtype)
    bias = (2 * rng.standard_normal((nseq, hq, max_q, max_kv))).astype(np.float32)
    scale = 1.0 / math.sqrt(d)
    out, got_lse = _launch(spec, q, k, v, bias, scale, cu_q=q_off, cu_kv=kv_off)
    ref, ref_lse = sdpa_reference(
        q,
        k,
        v,
        scale=scale,
        bias=bias.astype(np.float64),
        q_offsets=q_off,
        kv_offsets=kv_off,
        **band_kwargs(mask),
    )
    assert max_abs(out, ref) < tolerance(dtype)
    if mask == "window":
        # Each sequence's rows past its own keys + W - 1 store exact zeros.
        for s, kv_len in enumerate(kv_lens):
            dead = q_off[s] + kv_len + WINDOW - 1
            assert np.abs(out[dead : q_off[s + 1]]).max() == 0.0, s
    if lse:
        assert max_abs(got_lse, ref_lse) < tolerance(dtype)


@pytest.mark.parametrize("lse", [False, True], ids=["no-lse", "lse"])
@pytest.mark.parametrize("mask", ["top_left", "window"])
def test_varlen_finfo_min_rows(mask, lse):
    """Packed varlen, every bias finfo(float32).min: each row is the uniform softmax
    over its own sequence's visible keys (the causal triangle, the window), never a
    neighbour's keys. V has mean 1, so zeros fail. Under the window, rows past a
    sequence's keys + W - 1 still store exact zeros: no key is visible there."""
    dtype, d, hq, hkv = "bf16", 128, 4, 4
    q_lens, kv_lens = (1024, 512), (256, 128)
    if mask == "top_left":
        q_lens, kv_lens = (256, 512), (128, 256)
    q_off = [0, *np.cumsum(q_lens).tolist()]
    kv_off = [0, *np.cumsum(kv_lens).tolist()]
    nseq, max_q, max_kv = len(q_lens), max(q_lens), max(kv_lens)
    base = _spec(
        dtype, d, hq, hkv, batch=nseq, sq=max_q, skv=max_kv, mask=mask, lse=lse
    )
    spec = replace(base, varlen=True, ragged=False)
    rng = np.random.default_rng(6)
    q = round_to(rng.standard_normal((q_off[-1], hq, d)), dtype)
    k = round_to(rng.standard_normal((kv_off[-1], hkv, d)), dtype)
    v = round_to(rng.standard_normal((kv_off[-1], hkv, d)) + 1.0, dtype)
    bias = np.full((nseq, hq, max_q, max_kv), np.finfo(np.float32).min, np.float32)
    scale = 1.0 / math.sqrt(d)
    out, got_lse = _launch(spec, q, k, v, bias, scale, cu_q=q_off, cu_kv=kv_off)
    ref, ref_lse = sdpa_reference(
        q,
        k,
        v,
        scale=scale,
        bias=bias.astype(np.float64),
        q_offsets=q_off,
        kv_offsets=kv_off,
        **band_kwargs(mask),
    )
    assert np.isfinite(out).all()
    assert max_abs(out, ref) < tolerance(dtype)
    if mask == "window":
        for s, kv_len in enumerate(kv_lens):
            dead = q_off[s] + kv_len + WINDOW - 1
            assert np.abs(out[dead : q_off[s + 1]]).max() == 0.0, s
            assert np.abs(ref[q_off[s] : dead]).mean() > 0.5, s
    else:
        assert np.abs(ref).mean() > 0.5
    if lse:
        live = np.isfinite(ref_lse)
        assert np.isneginf(got_lse[~live]).all()
        assert (got_lse[live] < -1e28).all() and np.isfinite(got_lse[live]).all()
