# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""On-GPU numeric lane for the gfx942 dense kernel's additive bias.

Drives ``run_attention_dense_torch`` (and once the dispatcher's own binding) with an
fp32 ``[B|1, Hq|1, Sq|1, Skv]`` bias and checks the output against the float64
``sdpa_reference`` oracle (``_dense_bias_common.reference``), which adds the bias after
the scale as hipDNN does. The matrix crosses every bias shape with every mask the
kernel serves (none, top-left, bottom-right with Sq != Skv, sliding window), then dtype
x head size, then the sub-modes the bias composes with (sinks, LSE, a padded bias row
pitch, ``-inf`` / ``finfo(float32).min`` / NaN / below-the-floor entries, and the row
classes those make under a window that leaves keyless rows). gfx942 has no ragged or
varlen body, so every length is a tile multiple.

Every case runs on both grids: ``grid`` (the default grid) and ``persistent`` (the
persistent grid-stride body; ``gfx942_dense`` serves both). Specs come from the
dispatch factory, never hand-rolled; the 4-wide and scalar bias load forms
(``use_bias_x4``) are both run where the load address matters.

Every test is marked ``gpu`` and skips without a gfx942 and ROCm torch:

    HIP_VISIBLE_DEVICES=0 python -m pytest tests/test_attention_dense_bias_gfx942_numeric.py

The torch-free subset the CI GPU lane runs is ``test_attention_dense_bias_gfx942_gpu.py``.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from _dense_bias_common import (
    BIAS_SHAPES,
    MASKS,
    ROW_CLASS_DEAD_FROM,
    ROW_CLASS_HQ,
    ROW_CLASS_SKV,
    ROW_CLASS_SQ,
    SHAPE_FOR_MASK,
    WINDOW,
    check_window_row_classes,
    extents,
    max_abs,
    reference,
    tolerance,
    window_row_classes_inputs,
)
from kernels.common.attention_dense_spec import attention_dense_cache_key
from kernels.gfx942.attention_dense import run_attention_dense_torch

torch = pytest.importorskip("torch", reason="ROCm torch required")


def _gpu_ready():
    """True only on a gfx942 box with ROCm torch. Gate on the ISA target (gcnArchName),
    which every part of the family shares, not on the marketing name."""
    if not torch.cuda.is_available():
        return False
    return "gfx942" in torch.cuda.get_device_properties(0).gcnArchName.lower()


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not _gpu_ready(), reason="needs a gfx942 GPU with ROCm torch"),
]

_TORCH_DT = {"fp16": torch.float16, "bf16": torch.bfloat16}
# The request's mask kind; the window is a causal band with a left bound.
_MASK_TYPES = {"none": 0, "top_left": 1, "bottom_right": 2, "window": 1}


@pytest.fixture(params=[False, True], ids=["grid", "persistent"])
def persistent(request):
    return request.param


def _spec(
    dtype,
    d,
    hq,
    hkv,
    *,
    batch,
    sq,
    skv,
    mask,
    persistent,
    sinks=False,
    lse=False,
    tile=None,
):
    """The bias spec the dispatch factory builds for the request, on the grid asked
    for and, with ``tile``, at that ``(block_m, block_n)`` (knobs on its default spec,
    as the sweeps and the pins set them)."""
    from dispatch.attention import AttentionRequest, tuning_spec_with_knobs

    req = AttentionRequest(
        batch=batch,
        nhead_q=hq,
        nhead_k=hkv,
        seqlen_q=sq,
        seqlen_k=skv,
        hdim_q=d,
        hdim_v=d,
        arch="gfx942",
        mask_type=_MASK_TYPES[mask],
        sliding_window=WINDOW if mask == "window" else 0,
        dtype=dtype,
        use_sinks=sinks,
        emit_lse=lse,
        use_attn_bias=True,
    )
    knobs = {"persistent": persistent}
    if tile is not None:
        knobs.update(block_m=tile[0], block_n=tile[1])
    spec = tuning_spec_with_knobs(req, "gfx942_dense", knobs).kernel_spec
    assert spec.use_additive_bias and spec.persistent is persistent
    assert tile is None or (spec.block_m, spec.block_n) == tile
    assert spec.kv_segments == 1 and not spec.ragged
    return spec


def _inputs(dtype, d, hq, hkv, batch, sq, skv, bias_shape, *, seed=0):
    torch.manual_seed(seed)
    tdt = _TORCH_DT[dtype]
    q = torch.randn(batch, sq, hq, d, device="cuda", dtype=tdt)
    k = torch.randn(batch, skv, hkv, d, device="cuda", dtype=tdt)
    v = torch.randn(batch, skv, hkv, d, device="cuda", dtype=tdt)
    bias = 2 * torch.randn(bias_shape, device="cuda", dtype=torch.float32)
    return q, k, v, bias


def _host(t):
    """``t`` as a float32 numpy array."""
    return t.detach().float().cpu().numpy()


def _reference(q, k, v, scale, bias, mask, sinks=None):
    """``(O [B, Sq, Hq, D], LSE [B, Hq, Sq])``: the float64 oracle on the inputs the
    kernel reads (NaN bias entries read as ``-inf``; the kernel's NaN rows are checked
    apart)."""
    return reference(
        _host(q),
        _host(k),
        _host(v),
        scale,
        None if bias is None else _host(bias),
        mask,
        sinks=None if sinks is None else _host(sinks),
    )


def _run(spec, q, k, v, bias, scale, **kw):
    out = torch.full_like(q, float("nan"))
    returned = run_attention_dense_torch(
        spec=spec, q=q, k=k, v=v, out=out, scale=scale, additive_bias=bias, **kw
    )
    torch.cuda.synchronize()
    assert returned is out
    return _host(out)


def _lse_buffer(batch, hq, sq):
    """The caller-owned LSE buffer the kernel fills ([B, Hq, Sq, 1], NaN-filled so a
    row it skips fails the comparison)."""
    return torch.full((batch, hq, sq, 1), float("nan"), device="cuda")


@pytest.mark.parametrize("bias_name", list(BIAS_SHAPES))
@pytest.mark.parametrize("mask", MASKS)
@pytest.mark.parametrize(
    "d, x4", [(64, True), (128, False)], ids=["d64-x4", "d128-scalar"]
)
def test_bias_shape_x_mask(bias_name, mask, d, x4, persistent):
    """Every bias shape under every mask, bf16 GQA-4, B=2, with a different load form
    per head size (4-wide at D64, scalar at D128; forced, so a policy change does not
    move the coverage). Bottom-right runs Sq != Skv so its diagonal moves; the window
    runs past one KV tile."""
    dtype, hq, hkv, batch = "bf16", 8, 2, 2
    sq, skv = extents(mask)
    spec = _spec(
        dtype, d, hq, hkv, batch=batch, sq=sq, skv=skv, mask=mask, persistent=persistent
    )
    spec = dataclasses.replace(spec, use_bias_x4=x4)
    shape = BIAS_SHAPES[bias_name](batch, hq, sq, skv)
    q, k, v, bias = _inputs(dtype, d, hq, hkv, batch, sq, skv, shape)
    scale = 1.0 / math.sqrt(d)
    out = _run(spec, q, k, v, bias, scale)
    ref, _ = _reference(q, k, v, scale, bias, mask)
    assert max_abs(out, ref) < tolerance(dtype), (bias_name, mask)
    # The bias must move the output far past the tolerance, or a kernel that
    # ignored it (or read a constant row) would pass the check above.
    plain, _ = _reference(q, k, v, scale, None, mask)
    assert max_abs(out, plain) > 5 * tolerance(dtype), (bias_name, mask)


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize(
    "d,hq,hkv", [(64, 16, 16), (128, 32, 4)], ids=["d64-mha", "d128-gqa"]
)
@pytest.mark.parametrize("mask", MASKS)
def test_dtype_x_head_size(dtype, d, hq, hkv, mask, persistent):
    """Both dtypes and both head sizes (64, 128: all the dense kernel serves) under
    every mask, each with a different bias shape; B=2 so a batch broadcast is
    distinguishable from a full bias."""
    batch = 2
    sq, skv = extents(mask)
    spec = _spec(
        dtype, d, hq, hkv, batch=batch, sq=sq, skv=skv, mask=mask, persistent=persistent
    )
    shape = BIAS_SHAPES[SHAPE_FOR_MASK[mask]](batch, hq, sq, skv)
    q, k, v, bias = _inputs(dtype, d, hq, hkv, batch, sq, skv, shape)
    scale = 1.0 / math.sqrt(d)
    out = _run(spec, q, k, v, bias, scale)
    ref, _ = _reference(q, k, v, scale, bias, mask)
    assert max_abs(out, ref) < tolerance(dtype), (dtype, d, mask)


@pytest.mark.parametrize("mask", ["none", "top_left"])
@pytest.mark.parametrize(
    "dtype, d", [("bf16", 128), ("fp16", 64)], ids=["bf16-d128", "fp16-d64"]
)
def test_the_default_geometry_tile_serves_the_bias(dtype, d, mask, persistent):
    """The 256-row tile of the bias-off default geometry on both grids. The factory
    builds the 128-row tile for these requests, but a pin or a sweep can still ask for
    this one, and the persistent grid at D128 is where it used to be the default."""
    hq, hkv, batch = 8, 2, 2
    sq, skv = extents(mask)
    spec = _spec(
        dtype,
        d,
        hq,
        hkv,
        batch=batch,
        sq=sq,
        skv=skv,
        mask=mask,
        persistent=persistent,
        tile=(256, 64),
    )
    q, k, v, bias = _inputs(dtype, d, hq, hkv, batch, sq, skv, (batch, hq, sq, skv))
    scale = 1.0 / math.sqrt(d)
    out = _run(spec, q, k, v, bias, scale)
    ref, _ = _reference(q, k, v, scale, bias, mask)
    assert max_abs(out, ref) < tolerance(dtype), (dtype, d, mask)


@pytest.mark.parametrize("x4", [False, True], ids=["scalar", "x4"])
def test_padded_bias_row_pitch(x4, persistent):
    """A bias view with a padded Skv row pitch and a non-packed batch stride: the
    kernel reads the strides it is given, not the packed ones."""
    dtype, d, hq, hkv, batch, sq, skv = "bf16", 64, 4, 4, 2, 256, 256
    spec = _spec(
        dtype,
        d,
        hq,
        hkv,
        batch=batch,
        sq=sq,
        skv=skv,
        mask="top_left",
        persistent=persistent,
    )
    spec = dataclasses.replace(spec, use_bias_x4=x4)
    q, k, v, _ = _inputs(dtype, d, hq, hkv, batch, sq, skv, (1,))
    storage = torch.randn(
        batch, hq + 1, sq, skv + 40, device="cuda", dtype=torch.float32
    )
    bias = storage[:, 1:, :, 8 : 8 + skv]
    assert bias.stride(-1) == 1 and not bias.is_contiguous()
    scale = 1.0 / math.sqrt(d)
    out = _run(spec, q, k, v, bias, scale)
    ref, _ = _reference(q, k, v, scale, bias.contiguous(), "top_left")
    assert max_abs(out, ref) < tolerance(dtype)


@pytest.mark.parametrize("x4", [False, True], ids=["scalar", "x4"])
def test_expanded_bias_has_stride_zero_batch_and_head_axes(x4, persistent):
    """A bias expanded from one ``[Sq, Skv]`` plane to every batch and head: full-extent
    axes of stride 0, which is not the size-1 broadcast the shape tests cover."""
    dtype, d, hq, hkv, batch, sq, skv = "bf16", 64, 4, 2, 2, 512, 512
    spec = _spec(
        dtype,
        d,
        hq,
        hkv,
        batch=batch,
        sq=sq,
        skv=skv,
        mask="top_left",
        persistent=persistent,
    )
    spec = dataclasses.replace(spec, use_bias_x4=x4)
    q, k, v, _ = _inputs(dtype, d, hq, hkv, batch, sq, skv, (1,))
    plane = 2 * torch.randn(sq, skv, device="cuda", dtype=torch.float32)
    bias = plane.expand(batch, hq, sq, skv)
    assert bias.shape[:2] == (batch, hq) and bias.stride()[:2] == (0, 0)
    scale = 1.0 / math.sqrt(d)
    out = _run(spec, q, k, v, bias, scale)
    ref, _ = _reference(q, k, v, scale, bias.contiguous(), "top_left")
    assert max_abs(out, ref) < tolerance(dtype)


@pytest.mark.parametrize("x4", [False, True], ids=["scalar", "x4"])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("mask", ["none", "top_left"])
def test_misaligned_bias_view(mask, d, x4, persistent):
    """The 4-wide form loads four consecutive keys with one 16-byte load that is only
    element-aligned: a view one element into its storage, with an odd row pitch, puts
    most of those loads off a 16-byte boundary. ``-inf`` entries ride along so a lane
    mix-up inside a load shows."""
    dtype, hq, hkv, batch, sq, skv = "bf16", 4, 2, 2, 512, 512
    spec = _spec(
        dtype,
        d,
        hq,
        hkv,
        batch=batch,
        sq=sq,
        skv=skv,
        mask=mask,
        persistent=persistent,
    )
    spec = dataclasses.replace(spec, use_bias_x4=x4)
    q, k, v, _ = _inputs(dtype, d, hq, hkv, batch, sq, skv, (1,))
    pitch = skv + 5
    shape = (batch, hq, sq, skv)
    strides = (hq * sq * pitch, sq * pitch, pitch, 1)
    storage = 2 * torch.randn(
        1 + batch * hq * sq * pitch, device="cuda", dtype=torch.float32
    )
    bias = storage.as_strided(shape, strides, storage_offset=1)
    bias[..., 3::7] = float("-inf")
    assert storage.data_ptr() % 16 == 0
    # Group loads start at element 1 + s*pitch + 4m (the batch and head terms are
    # multiples of 4): every 4-byte residue of a 16-byte line is hit.
    assert {(bias.storage_offset() + r * pitch) % 4 for r in range(4)} == {0, 1, 2, 3}
    scale = 1.0 / math.sqrt(d)
    out = _run(spec, q, k, v, bias, scale)
    ref, _ = _reference(q, k, v, scale, bias.contiguous(), mask)
    assert max_abs(out, ref) < tolerance(dtype)


def test_minus_inf_entries_mask_keys(persistent):
    """A float mask spelled with ``-inf`` (what PyTorch SDPA makes of a boolean mask)
    hides those keys: here every odd key of every row, plus whole key blocks for one
    head."""
    dtype, d, hq, hkv, batch, sq, skv = "fp16", 128, 4, 4, 1, 512, 512
    spec = _spec(
        dtype,
        d,
        hq,
        hkv,
        batch=batch,
        sq=sq,
        skv=skv,
        mask="none",
        persistent=persistent,
    )
    q, k, v, bias = _inputs(dtype, d, hq, hkv, batch, sq, skv, (batch, hq, sq, skv))
    bias[..., 1::2] = float("-inf")
    bias[:, 2, :, 128:384] = float("-inf")
    scale = 1.0 / math.sqrt(d)
    out = _run(spec, q, k, v, bias, scale)
    ref, _ = _reference(q, k, v, scale, bias, "none")
    assert max_abs(out, ref) < tolerance(dtype)


@pytest.mark.parametrize("mask", ["none", "top_left"])
def test_bias_below_the_mask_sentinel(mask, persistent):
    """A finite bias far below the kernel's mask sentinel is still an ordinary (very
    negative) score: a row whose every visible key has one gets the uniform softmax over
    them, as the reference and PyTorch give it, and not the mean of the masked future
    keys (which would win the row max if the sentinel outranked the bias) or NaN. Causal
    rows 0..3 see only such keys and, with no mask, head 1 has nothing else. V has mean
    1, so a row of zeros fails."""
    dtype, d, hq, hkv, batch, sq, skv = "bf16", 128, 4, 4, 1, 256, 256
    spec = _spec(
        dtype, d, hq, hkv, batch=batch, sq=sq, skv=skv, mask=mask, persistent=persistent
    )
    q, k, v, bias = _inputs(dtype, d, hq, hkv, batch, sq, skv, (batch, hq, sq, skv))
    v = v + 1.0
    bias[..., :4] = -1e30
    if mask == "none":
        bias[:, 1] = -1e30
    scale = 1.0 / math.sqrt(d)
    out = _run(spec, q, k, v, bias, scale)
    ref, _ = _reference(q, k, v, scale, bias, mask)
    assert np.isfinite(out).all()
    assert np.abs(ref[:, :4]).mean() > 0.5  # the rows under test are not zero
    assert max_abs(out, ref) < tolerance(dtype)


def test_a_finfo_min_mask_matches_the_reference(persistent):
    """HF-style float masks use finfo(float32).min. Here every odd key and the whole
    first KV tile are masked for every row; each row keeps live keys later on, so the
    masked keys get weight 0."""
    dtype, d, hq, hkv, batch, sq, skv = "fp16", 128, 4, 4, 1, 512, 512
    spec = _spec(
        dtype,
        d,
        hq,
        hkv,
        batch=batch,
        sq=sq,
        skv=skv,
        mask="none",
        persistent=persistent,
    )
    q, k, v, bias = _inputs(dtype, d, hq, hkv, batch, sq, skv, (batch, hq, sq, skv))
    lowest = torch.finfo(torch.float32).min
    bias[..., 1::2] = lowest
    bias[..., :128] = lowest
    scale = 1.0 / math.sqrt(d)
    out = _run(spec, q, k, v, bias, scale)
    ref, _ = _reference(q, k, v, scale, bias, "none")
    assert not np.isnan(out).any()
    assert max_abs(out, ref) < tolerance(dtype)


@pytest.mark.parametrize("x4", [False, True], ids=["scalar", "x4"])
@pytest.mark.parametrize("lse", [False, True], ids=["no-lse", "lse"])
@pytest.mark.parametrize("mask", ["none", "top_left", "window"])
def test_a_row_of_finfo_min_biases_is_the_uniform_softmax(mask, lse, x4, persistent):
    """Head 1 has finfo(float32).min for every key, so each of its rows has no key
    above the others: the softmax is uniform over the keys the mask leaves visible
    (the causal triangle, the window), O = mean(V) there, as in the reference and
    PyTorch. V has mean 1, so a row of zeros fails. The LSE the reference gives is
    about finfo.min; the kernel raises the bias to -2**97 and reports about that
    (finite, below -1e28), and the other heads' LSE match the reference."""
    dtype, d, hq, hkv, batch, sq, skv = "bf16", 128, 4, 4, 1, 512, 512
    spec = _spec(
        dtype,
        d,
        hq,
        hkv,
        batch=batch,
        sq=sq,
        skv=skv,
        mask=mask,
        persistent=persistent,
        lse=lse,
    )
    spec = dataclasses.replace(spec, use_bias_x4=x4)
    q, k, v, bias = _inputs(dtype, d, hq, hkv, batch, sq, skv, (batch, hq, sq, skv))
    v = v + 1.0
    bias[:, 1] = torch.finfo(torch.float32).min
    scale = 1.0 / math.sqrt(d)
    kw = {"lse": _lse_buffer(batch, hq, sq)} if lse else {}
    out = _run(spec, q, k, v, bias, scale, **kw)
    ref, lse_ref = _reference(q, k, v, scale, bias, mask)
    assert np.isfinite(out).all()
    assert np.abs(ref[:, :, 1]).mean() > 0.5
    assert max_abs(out, ref) < tolerance(dtype)
    if lse:
        got = _host(kw["lse"])[..., 0]
        assert np.isfinite(got).all() and (got[:, 1] < -1e28).all()
        other = [0, 2, 3]
        assert max_abs(got[:, other], lse_ref[:, other]) < tolerance(dtype)


@pytest.mark.parametrize("lse", [False, True], ids=["no-lse", "lse"])
@pytest.mark.parametrize("mask", ["none", "top_left"])
def test_minus_inf_and_finfo_min_in_one_row(mask, lse, persistent):
    """Head 1 has finfo.min on even keys and -inf on odd ones, with the odd keys' V
    (that head's) shifted by 10: a row is uniform over its visible even keys only, as in
    the reference. Neither value may stand in for the other: -inf is a key out of the
    softmax, finfo.min one in it."""
    dtype, d, hq, hkv, batch, sq, skv = "bf16", 128, 4, 4, 1, 256, 256
    spec = _spec(
        dtype,
        d,
        hq,
        hkv,
        batch=batch,
        sq=sq,
        skv=skv,
        mask=mask,
        persistent=persistent,
        lse=lse,
    )
    q, k, v, bias = _inputs(dtype, d, hq, hkv, batch, sq, skv, (batch, hq, sq, skv))
    v = v + 1.0
    v[:, 1::2, 1] += 10.0
    bias[:, 1, :, 0::2] = torch.finfo(torch.float32).min
    bias[:, 1, :, 1::2] = float("-inf")
    scale = 1.0 / math.sqrt(d)
    kw = {"lse": _lse_buffer(batch, hq, sq)} if lse else {}
    out = _run(spec, q, k, v, bias, scale, **kw)
    ref, _ = _reference(q, k, v, scale, bias, mask)
    assert np.isfinite(out).all()
    assert np.abs(ref[:, :, 1]).mean() > 0.5
    assert max_abs(out, ref) < tolerance(dtype)
    if lse:
        assert np.isfinite(_host(kw["lse"])[:, 1]).all()


@pytest.mark.parametrize("lse", [False, True], ids=["no-lse", "lse"])
def test_a_row_with_no_finite_score_outputs_zero(lse, persistent):
    """sdpa_reference (hipDNN semantics): a row whose every visible key has a -inf
    bias outputs O = 0 and LSE = -inf. Rows 0..63 see only keys 0..q under the causal
    mask and have no live key (-inf); rows 64.. keep live keys."""
    dtype, d, hq, hkv, batch, sq, skv = "fp16", 128, 4, 4, 1, 256, 256
    spec = _spec(
        dtype,
        d,
        hq,
        hkv,
        batch=batch,
        sq=sq,
        skv=skv,
        mask="top_left",
        persistent=persistent,
        lse=lse,
    )
    q, k, v, bias = _inputs(dtype, d, hq, hkv, batch, sq, skv, (batch, hq, sq, skv))
    bias[:, :, :64, :64] = float("-inf")
    scale = 1.0 / math.sqrt(d)
    kw = {"lse": _lse_buffer(batch, hq, sq)} if lse else {}
    out = _run(spec, q, k, v, bias, scale, **kw)
    ref, lse_ref = _reference(q, k, v, scale, bias, "top_left")
    assert not np.isnan(out).any()
    assert np.abs(out[:, :64]).max() == 0.0
    assert max_abs(out[:, 64:], ref[:, 64:]) < tolerance(dtype)
    if lse:
        got = _host(kw["lse"])[..., 0]
        assert np.isneginf(got[:, :, :64]).all()
        assert max_abs(got[:, :, 64:], lse_ref[:, :, 64:]) < tolerance(dtype)


@pytest.mark.parametrize("lse", [False, True], ids=["no-lse", "lse"])
def test_a_head_of_minus_inf_biases_outputs_zero(lse, persistent):
    """With no mask, head 1 has -inf for every key: each of its rows has no finite
    score, so O = 0 and LSE = -inf (the hipDNN reference); the other heads are
    ordinary."""
    dtype, d, hq, hkv, batch, sq, skv = "bf16", 128, 4, 4, 1, 256, 256
    spec = _spec(
        dtype,
        d,
        hq,
        hkv,
        batch=batch,
        sq=sq,
        skv=skv,
        mask="none",
        persistent=persistent,
        lse=lse,
    )
    q, k, v, bias = _inputs(dtype, d, hq, hkv, batch, sq, skv, (batch, hq, sq, skv))
    bias[:, 1] = float("-inf")
    scale = 1.0 / math.sqrt(d)
    kw = {"lse": _lse_buffer(batch, hq, sq)} if lse else {}
    out = _run(spec, q, k, v, bias, scale, **kw)
    ref, lse_ref = _reference(q, k, v, scale, bias, "none")
    assert np.abs(out[:, :, 1]).max() == 0.0
    assert max_abs(out, ref) < tolerance(dtype)
    if lse:
        got = _host(kw["lse"])[..., 0]
        assert np.isneginf(got[:, 1]).all()
        other = [0, 2, 3]
        assert max_abs(got[:, other], lse_ref[:, other]) < tolerance(dtype)


@pytest.mark.parametrize("value", [float("-inf"), -1e30], ids=["-inf", "below-floor"])
@pytest.mark.parametrize("lse", [False, True], ids=["no-lse", "lse"])
def test_a_keyless_row_with_sinks_keeps_the_sink(lse, value, persistent):
    """With sinks, a row whose keys carry no weight still has its sink in the softmax:
    O = 0 (the sink has no value vector) and LSE = the sink logit. Rows 0..63 of a
    causal mask see only keys 0..q, all at ``value``: -inf, or a finite bias far below
    the floor (raised to it: its keys get P = 0 next to the sink, which is the
    reference's answer too)."""
    dtype, d, hq, hkv, batch, sq, skv = "bf16", 128, 4, 4, 1, 256, 256
    spec = _spec(
        dtype,
        d,
        hq,
        hkv,
        batch=batch,
        sq=sq,
        skv=skv,
        mask="top_left",
        persistent=persistent,
        sinks=True,
        lse=lse,
    )
    q, k, v, bias = _inputs(dtype, d, hq, hkv, batch, sq, skv, (batch, hq, sq, skv))
    bias[:, :, :64, :64] = value
    sinks = torch.randn(hq, device="cuda", dtype=_TORCH_DT[dtype])
    scale = 1.0 / math.sqrt(d)
    kw = {"sinks": sinks}
    if lse:
        kw["lse"] = _lse_buffer(batch, hq, sq)
    out = _run(spec, q, k, v, bias, scale, **kw)
    ref, lse_ref = _reference(q, k, v, scale, bias, "top_left", sinks)
    assert np.abs(out[:, :64]).max() == 0.0
    assert max_abs(out, ref) < tolerance(dtype)
    if lse:
        got = _host(kw["lse"])[..., 0]
        assert max_abs(got, lse_ref) < tolerance(dtype)
        assert max_abs(got[:, :, :64], _host(sinks)[None, :, None]) < 1e-6


@pytest.mark.parametrize("lse", [False, True], ids=["no-lse", "lse"])
def test_window_rows_past_the_keys(lse, persistent):
    """Sq > Skv + W - 1: the last q-blocks have no visible key. Rows q >= Skv + W - 1
    store O = 0 (and LSE = -inf, with LSE on), without sinks too; the rest match the
    reference."""
    dtype, d, hq, hkv, batch, sq, skv = "bf16", 128, 4, 4, 1, 1024, 256
    spec = _spec(
        dtype,
        d,
        hq,
        hkv,
        batch=batch,
        sq=sq,
        skv=skv,
        mask="window",
        persistent=persistent,
        lse=lse,
    )
    q, k, v, bias = _inputs(dtype, d, hq, hkv, batch, sq, skv, (1, 1, 1, skv))
    scale = 1.0 / math.sqrt(d)
    kw = {"lse": _lse_buffer(batch, hq, sq)} if lse else {}
    out = _run(spec, q, k, v, bias, scale, **kw)
    ref, lse_ref = _reference(q, k, v, scale, bias, "window")
    dead = skv + WINDOW - 1
    assert np.abs(out[:, dead:]).max() == 0.0
    assert max_abs(out, ref) < tolerance(dtype)
    if lse:
        got = _host(kw["lse"])[..., 0]
        assert np.isneginf(got[:, :, dead:]).all()
        assert max_abs(got[:, :, :dead], lse_ref[:, :, :dead]) < tolerance(dtype)


@pytest.mark.parametrize("x4", [False, True], ids=["scalar", "x4"])
@pytest.mark.parametrize("sinks", [False, True], ids=["no-sinks", "sinks"])
@pytest.mark.parametrize("lse", [False, True], ids=["no-lse", "lse"])
def test_the_row_classes_of_a_window_with_a_bias(lse, sinks, x4, persistent):
    """One launch, four bias classes (one per head; see ``window_row_classes_inputs``)
    under a window that leaves the last q-blocks without a key: keyless-by-window rows
    store zeros in every head, whatever their bias; an all -inf head is zeros; a
    finfo.min head is the uniform softmax over its window; a NaN on a visible key makes
    only that row NaN, while a NaN on a key the window removes is ignored. With sinks
    the sink logit is the LSE of every row with no weight and outweighs a finfo.min
    head."""
    dtype, d = "bf16", 128
    hq, sq, skv = ROW_CLASS_HQ, ROW_CLASS_SQ, ROW_CLASS_SKV
    spec = _spec(
        dtype,
        d,
        hq,
        hq,
        batch=1,
        sq=sq,
        skv=skv,
        mask="window",
        persistent=persistent,
        sinks=sinks,
        lse=lse,
    )
    spec = dataclasses.replace(spec, use_bias_x4=x4)
    qn, kn, vn, bn = window_row_classes_inputs(dtype, d=d)
    tdt = _TORCH_DT[dtype]
    q, k, v = (torch.from_numpy(a).to("cuda", tdt) for a in (qn, kn, vn))
    bias = torch.from_numpy(bn).to("cuda")
    values = (1.0 + torch.randn(hq, device="cuda")).to(tdt) if sinks else None
    scale = 1.0 / math.sqrt(d)
    kw = {"sinks": values} if sinks else {}
    if lse:
        kw["lse"] = _lse_buffer(1, hq, sq)
    out = _run(spec, q, k, v, bias, scale, **kw)
    ref, lse_ref = _reference(q, k, v, scale, bias, "window", values)
    assert 0 < ROW_CLASS_DEAD_FROM < sq
    got_lse = _host(kw["lse"])[..., 0] if lse else None
    check_window_row_classes(
        out,
        got_lse,
        ref,
        lse_ref,
        sinks=None if values is None else _host(values),
        dtype=dtype,
    )


@pytest.mark.parametrize("lse", [False, True], ids=["no-lse", "lse"])
def test_window_rows_past_the_keys_with_sinks(lse, persistent):
    """Sq > Skv + W - 1: the last q-blocks have no visible key. With sinks their rows
    are softmax([sink]) over no values: O = 0, LSE = the sink logit; the rest match the
    reference."""
    dtype, d, hq, hkv, batch, sq, skv = "bf16", 128, 4, 4, 1, 1024, 256
    spec = _spec(
        dtype,
        d,
        hq,
        hkv,
        batch=batch,
        sq=sq,
        skv=skv,
        mask="window",
        persistent=persistent,
        sinks=True,
        lse=lse,
    )
    q, k, v, bias = _inputs(dtype, d, hq, hkv, batch, sq, skv, (1, 1, 1, skv))
    sinks = torch.randn(hq, device="cuda", dtype=_TORCH_DT[dtype])
    scale = 1.0 / math.sqrt(d)
    kw = {"sinks": sinks}
    if lse:
        kw["lse"] = _lse_buffer(batch, hq, sq)
    out = _run(spec, q, k, v, bias, scale, **kw)
    ref, lse_ref = _reference(q, k, v, scale, bias, "window", sinks)
    dead = skv + WINDOW - 1
    assert np.abs(out[:, dead:]).max() == 0.0
    assert max_abs(out, ref) < tolerance(dtype)
    if lse:
        got = _host(kw["lse"])[..., 0]
        assert max_abs(got, lse_ref) < tolerance(dtype)
        assert max_abs(got[:, :, dead:], _host(sinks)[None, :, None]) < 1e-6


@pytest.mark.parametrize("x4", [False, True], ids=["scalar", "x4"])
@pytest.mark.parametrize("lse", [False, True], ids=["no-lse", "lse"])
@pytest.mark.parametrize("mask", ["none", "top_left"])
def test_a_nan_bias_makes_its_row_nan(mask, lse, x4, persistent):
    """A NaN bias on a visible key makes its whole row NaN (O and LSE), as in the
    reference; every other row matches it. Head 0 row 10 has one NaN key (key 7,
    visible under every mask), head 2 row 20 has NaN on every key (the max ignores a
    NaN score, so only the NaN denominator keeps the row from reading as dead), and
    head 1 row 5 has a NaN at key 200: visible only without the mask, so under the
    causal mask the key is skipped, whatever its bias, and the row stays finite."""
    dtype, d, hq, hkv, batch, sq, skv = "fp16", 128, 4, 4, 1, 256, 256
    spec = _spec(
        dtype,
        d,
        hq,
        hkv,
        batch=batch,
        sq=sq,
        skv=skv,
        mask=mask,
        persistent=persistent,
        lse=lse,
    )
    spec = dataclasses.replace(spec, use_bias_x4=x4)
    q, k, v, bias = _inputs(dtype, d, hq, hkv, batch, sq, skv, (batch, hq, sq, skv))
    nan = float("nan")
    bias[0, 0, 10, 7] = nan
    bias[0, 2, 20, :] = nan
    bias[0, 1, 5, 200] = nan
    expect_nan = np.zeros((batch, sq, hq), bool)  # [B, Sq, Hq]
    expect_nan[0, 10, 0] = expect_nan[0, 20, 2] = True
    expect_nan[0, 5, 1] = mask == "none"
    scale = 1.0 / math.sqrt(d)
    kw = {"lse": _lse_buffer(batch, hq, sq)} if lse else {}
    out = _run(spec, q, k, v, bias, scale, **kw)
    ref, lse_ref = _reference(q, k, v, scale, bias, mask)  # reads NaN as -inf
    nan_out = np.isnan(out)
    np.testing.assert_array_equal(nan_out.all(-1), expect_nan)
    assert not (nan_out.any(-1) & ~expect_nan).any()
    live = ~expect_nan
    assert np.abs(out - ref).max(-1)[live].max() < tolerance(dtype)
    if lse:
        got = _host(kw["lse"])[..., 0].transpose(0, 2, 1)  # [B, Sq, Hq]
        np.testing.assert_array_equal(np.isnan(got), expect_nan)
        diff = np.abs(got - lse_ref.transpose(0, 2, 1))
        assert diff[live].max() < tolerance(dtype)


@pytest.mark.parametrize("mask", ["top_left", "bottom_right", "window"])
def test_lse_includes_the_bias(mask, persistent):
    """The natural-log LSE the kernel writes is that of the biased scores."""
    dtype, d, hq, hkv, batch = "bf16", 128, 8, 2, 2
    sq, skv = extents(mask)
    spec = _spec(
        dtype,
        d,
        hq,
        hkv,
        batch=batch,
        sq=sq,
        skv=skv,
        mask=mask,
        persistent=persistent,
        lse=True,
    )
    q, k, v, bias = _inputs(dtype, d, hq, hkv, batch, sq, skv, (batch, 1, sq, skv))
    lse = _lse_buffer(batch, hq, sq)
    scale = 1.0 / math.sqrt(d)
    out = _run(spec, q, k, v, bias, scale, lse=lse)
    ref, lse_ref = _reference(q, k, v, scale, bias, mask)
    assert max_abs(out, ref) < tolerance(dtype), mask
    assert max_abs(_host(lse)[..., 0], lse_ref) < tolerance(dtype), mask


@pytest.mark.parametrize("lse", [False, True], ids=["no-lse", "lse"])
@pytest.mark.parametrize("mask", ["top_left", "bottom_right", "window"])
def test_sinks_with_bias(mask, lse, persistent):
    """Sinks join the softmax denominator next to the biased scores. They are shifted up
    (about 8) so that they move the output of every mask, bottom-right included, far
    past the tolerance: a kernel that dropped them would otherwise pass."""
    dtype, d, hq, hkv, batch = "bf16", 128, 8, 2, 1
    sq, skv = extents(mask)
    spec = _spec(
        dtype,
        d,
        hq,
        hkv,
        batch=batch,
        sq=sq,
        skv=skv,
        mask=mask,
        persistent=persistent,
        sinks=True,
        lse=lse,
    )
    q, k, v, bias = _inputs(dtype, d, hq, hkv, batch, sq, skv, (batch, hq, sq, skv))
    sinks = (8.0 + torch.randn(hq, device="cuda")).to(_TORCH_DT[dtype])
    scale = 1.0 / math.sqrt(d)
    kw = {"sinks": sinks}
    if lse:
        kw["lse"] = _lse_buffer(batch, hq, sq)
    out = _run(spec, q, k, v, bias, scale, **kw)
    ref, lse_ref = _reference(q, k, v, scale, bias, mask, sinks)
    assert max_abs(out, ref) < tolerance(dtype)
    no_sinks, _ = _reference(q, k, v, scale, bias, mask)
    assert max_abs(out, no_sinks) > 5 * tolerance(dtype)
    if lse:
        assert max_abs(_host(kw["lse"])[..., 0], lse_ref) < tolerance(dtype)


def test_an_lse_buffer_overlapping_the_bias_is_refused(persistent):
    """The kernel writes LSE while other workgroups still read the bias, so an LSE
    buffer sharing storage with it is refused before launch. The alias is otherwise
    a valid key-row bias (fp32, contiguous along Skv), so only the overlap check
    can refuse it."""
    dtype, d, hq, hkv, sq, skv = "fp16", 128, 4, 1, 256, 256
    spec = _spec(
        dtype,
        d,
        hq,
        hkv,
        batch=1,
        sq=sq,
        skv=skv,
        mask="top_left",
        persistent=persistent,
        lse=True,
    )
    q, k, v, _ = _inputs(dtype, d, hq, hkv, 1, sq, skv, (1, 1, 1, skv))
    lse_elems = hq * sq
    storage = torch.zeros(lse_elems + skv, device="cuda", dtype=torch.float32)
    lse = storage[:lse_elems].view(1, hq, sq, 1)
    # The bias starts on the last LSE element.
    bias = storage[lse_elems - 1 : lse_elems - 1 + skv].view(1, 1, 1, skv)
    with pytest.raises(ValueError, match="lse overlaps additive_bias"):
        _run(spec, q, k, v, bias, 1.0 / math.sqrt(d), lse=lse)


def test_dispatcher_binding_passes_the_bias(persistent):
    """``dispatch_attention(...).bind_torch`` hands ``tensors['attn_bias']`` to the
    runner; leaving it out is refused rather than launched with a null pointer."""
    from dispatch.attention import AttentionRequest, dispatch_attention

    dtype, d, hq, hkv, batch, sq, skv = "bf16", 128, 8, 2, 2, 512, 512
    result = dispatch_attention(
        AttentionRequest(
            batch=batch,
            nhead_q=hq,
            nhead_k=hkv,
            seqlen_q=sq,
            seqlen_k=skv,
            hdim_q=d,
            hdim_v=d,
            arch="gfx942",
            mask_type=1,
            dtype=dtype,
            algorithm="attention_dense",
            spec_id="gfx942_dense",
            tuning_knobs={"persistent": persistent},
            use_attn_bias=True,
        )
    )
    assert result.spec.kernel_spec == _spec(
        dtype,
        d,
        hq,
        hkv,
        batch=batch,
        sq=sq,
        skv=skv,
        mask="top_left",
        persistent=persistent,
    )
    q, k, v, bias = _inputs(dtype, d, hq, hkv, batch, sq, skv, (batch, hq, sq, skv))
    out = torch.full_like(q, float("nan"))
    scale = 1.0 / math.sqrt(d)
    tensors = {"q": q, "k": k, "v": v, "out": out}
    with pytest.raises(
        ValueError, match=r"spec\.use_additive_bias=True requires additive_bias"
    ):
        result.bind_torch(tensors, scale=scale).launch()
    result.bind_torch({**tensors, "attn_bias": bias}, scale=scale).launch()
    torch.cuda.synchronize()
    ref, _ = _reference(q, k, v, scale, bias, "top_left")
    assert max_abs(_host(out), ref) < tolerance(dtype)


def test_runtime_shape_bias_kernel_serves_another_shape(persistent):
    """The bias kernel takes its shape and bias strides at run time, so one compiled
    binary serves a second batch, length and head configuration."""
    dtype, d = "fp16", 64
    specs = []
    for batch, seq, hq, hkv in ((1, 256, 8, 8), (3, 768, 12, 4)):
        spec = _spec(
            dtype,
            d,
            hq,
            hkv,
            batch=batch,
            sq=seq,
            skv=seq,
            mask="top_left",
            persistent=persistent,
        )
        specs.append(spec)
        q, k, v, bias = _inputs(dtype, d, hq, hkv, batch, seq, seq, (batch, hq, 1, seq))
        scale = 1.0 / math.sqrt(d)
        out = _run(spec, q, k, v, bias, scale)
        ref, _ = _reference(q, k, v, scale, bias, "top_left")
        assert max_abs(out, ref) < tolerance(dtype), (batch, seq, hq, hkv)
    assert len({attention_dense_cache_key(s, arch="gfx942") for s in specs}) == 1
