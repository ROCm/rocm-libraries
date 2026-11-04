# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""On-GPU numeric lane for the gfx950 dense kernel's additive bias.

Drives ``run_attention_dense_torch`` (and once the dispatcher's own binding) with an
fp32 ``[B|1, Hq|1, Sq|1, Skv]`` bias and checks the output against the float64
``sdpa_reference`` oracle, which adds the bias after the scale as hipDNN does. The
matrix crosses every bias shape with every mask the grid body serves (none, top-left,
bottom-right with Sq != Skv, sliding window), then dtype x head size, then the
sub-modes the bias composes with (ragged lengths, packed varlen, sinks, LSE, a padded
bias row pitch, ``-inf`` entries).

Every case runs at two tiles, both built by the dispatch factory: ``factory-tile``,
the ``gfx950_dense_grid`` candidate's default spec (block_m 256 except under a
window), and ``catalog-tile``, the configuration the hipDNN catalog ships its bias
kernels at (the ``gfx950_dense_qbminor`` candidate at block_m 128 / block_n 64).

Every test is marked ``gpu`` and skips without a gfx950 and ROCm torch:

    HIP_VISIBLE_DEVICES=0 python -m pytest tests/test_attention_dense_bias_gfx950_numeric.py
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from kernels.gfx950.attention_dense import run_attention_dense_torch

torch = pytest.importorskip("torch", reason="ROCm torch required")


def _gpu_ready():
    """True only on a gfx950 box with ROCm torch (gate on the ISA target)."""
    if not torch.cuda.is_available():
        return False
    return "gfx950" in torch.cuda.get_device_properties(0).gcnArchName.lower()


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not _gpu_ready(), reason="needs a gfx950 GPU with ROCm torch"),
]

_TORCH_DT = {"fp16": torch.float16, "bf16": torch.bfloat16}
_MASK_TYPES = {"none": 0, "top_left": 1, "bottom_right": 2, "window": 3}
_WINDOW = 128

# Bias extents, each axis but the key one full or broadcast.
_BIAS_SHAPES = {
    "full": lambda b, h, sq, skv: (b, h, sq, skv),
    "batch_bcast": lambda b, h, sq, skv: (1, h, sq, skv),
    "head_bcast": lambda b, h, sq, skv: (b, 1, sq, skv),
    "both_bcast": lambda b, h, sq, skv: (1, 1, sq, skv),
    "key_row": lambda b, h, sq, skv: (1, 1, 1, skv),
}


def _tolerance(dtype):
    """The dense numeric lane's max_abs tolerance (fp16 2e-2, bf16 4e-2)."""
    return 2e-2 if dtype == "fp16" else 4e-2


# The tiles every case runs at: (candidate, knobs on top of its default spec).
_TILES = {
    "factory-tile": ("gfx950_dense_grid", {}),
    "catalog-tile": ("gfx950_dense_qbminor", {"block_m": 128, "block_n": 64}),
}


@pytest.fixture(params=list(_TILES))
def tile(request):
    return request.param


def _spec(dtype, d, hq, hkv, *, batch, sq, skv, mask, tile, sinks=False, lse=False):
    """The grid-body spec for a bias request at ``tile``, from the dispatch factory."""
    from dispatch.attention import AttentionRequest, attention_tuning_spec

    extra = {"sliding_window": _WINDOW, "right_bound": 0} if mask == "window" else {}
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
        use_sinks=sinks,
        emit_lse=lse,
        use_attn_bias=True,
        **extra,
    )
    spec_id, knobs = _TILES[tile]
    spec = attention_tuning_spec(req, spec_id, knobs=knobs).kernel_spec
    assert spec.use_additive_bias and not spec.persistent
    if tile == "catalog-tile":
        assert (spec.block_m, spec.block_n) == (128, 64)
        assert spec.nonpersist_decode == "qb_minor"
    return spec


def _inputs(dtype, d, hq, hkv, batch, sq, skv, bias_shape, *, seed=0):
    torch.manual_seed(seed)
    tdt = _TORCH_DT[dtype]
    q = torch.randn(batch, sq, hq, d, device="cuda", dtype=tdt)
    k = torch.randn(batch, skv, hkv, d, device="cuda", dtype=tdt)
    v = torch.randn(batch, skv, hkv, d, device="cuda", dtype=tdt)
    bias = 2 * torch.randn(bias_shape, device="cuda", dtype=torch.float32)
    return q, k, v, bias


def _band(mask):
    """sdpa_reference band arguments for ``mask``. The dense window keeps keys with
    ``q - k < W``, i.e. a left bound of W - 1."""
    if mask == "top_left":
        return {"causal": True}
    if mask == "bottom_right":
        return {"causal_bottom_right": True}
    if mask == "window":
        return {"left_bound": _WINDOW - 1, "right_bound": 0, "top_left": True}
    return {}


def _bhsd(t):
    return t.float().permute(0, 2, 1, 3).cpu().numpy()


def _reference(q, k, v, scale, bias, mask):
    """``(O [B, Sq, Hq, D], LSE [B, Hq, Sq])`` from the float64 oracle."""
    from rocke.numeric.sdpa_reference import sdpa_reference

    o, lse = sdpa_reference(
        _bhsd(q),
        _bhsd(k),
        _bhsd(v),
        scale=scale,
        bias=None if bias is None else bias.cpu().numpy().astype(np.float64),
        **_band(mask),
    )
    return (
        torch.from_numpy(o).permute(0, 2, 1, 3).float(),
        torch.from_numpy(lse).float(),
    )


def _max_abs(out, ref):
    return (out.float().cpu() - ref).abs().max().item()


def _run(spec, q, k, v, bias, scale, **kw):
    out = torch.full_like(q, float("nan"))
    returned = run_attention_dense_torch(
        spec=spec, q=q, k=k, v=v, out=out, scale=scale, additive_bias=bias, **kw
    )
    torch.cuda.synchronize()
    assert returned is out
    return out


@pytest.mark.parametrize("bias_name", list(_BIAS_SHAPES))
@pytest.mark.parametrize("mask", list(_MASK_TYPES))
def test_bias_shape_x_mask(bias_name, mask, tile):
    """Every bias shape under every mask, bf16 D128 GQA-4, B=2. Bottom-right runs
    Sq != Skv so its diagonal moves; the window runs past one KV tile."""
    dtype, d, hq, hkv, batch = "bf16", 128, 8, 2, 2
    sq, skv = (256, 512) if mask == "bottom_right" else (512, 512)
    spec = _spec(dtype, d, hq, hkv, batch=batch, sq=sq, skv=skv, tile=tile, mask=mask)
    shape = _BIAS_SHAPES[bias_name](batch, hq, sq, skv)
    q, k, v, bias = _inputs(dtype, d, hq, hkv, batch, sq, skv, shape)
    scale = 1.0 / math.sqrt(d)
    out = _run(spec, q, k, v, bias, scale)
    ref, _ = _reference(q, k, v, scale, bias, mask)
    assert _max_abs(out, ref) < _tolerance(dtype), (bias_name, mask)
    # The bias must move the output far past the tolerance, or a kernel that
    # ignored it (or read a constant row) would pass the check above.
    plain, _ = _reference(q, k, v, scale, None, mask)
    assert _max_abs(out, plain) > 5 * _tolerance(dtype), (bias_name, mask)


# One bias shape per mask, so dtype x head x mask also walks every shape.
_SHAPE_FOR_MASK = {
    "none": "full",
    "top_left": "head_bcast",
    "bottom_right": "batch_bcast",
    "window": "both_bcast",
}


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize(
    "d,hq,hkv", [(64, 16, 16), (128, 32, 4)], ids=["d64-mha", "d128-gqa"]
)
@pytest.mark.parametrize("mask", list(_MASK_TYPES))
def test_dtype_x_head_size(dtype, d, hq, hkv, mask, tile):
    """Both dtypes and both head sizes (64, 128: all the dense kernel serves) under
    every mask, each with a different bias shape; B=2 so a batch broadcast is
    distinguishable from a full bias."""
    batch = 2
    sq, skv = (256, 512) if mask == "bottom_right" else (512, 512)
    spec = _spec(dtype, d, hq, hkv, batch=batch, sq=sq, skv=skv, tile=tile, mask=mask)
    shape = _BIAS_SHAPES[_SHAPE_FOR_MASK[mask]](batch, hq, sq, skv)
    q, k, v, bias = _inputs(dtype, d, hq, hkv, batch, sq, skv, shape)
    scale = 1.0 / math.sqrt(d)
    out = _run(spec, q, k, v, bias, scale)
    ref, _ = _reference(q, k, v, scale, bias, mask)
    assert _max_abs(out, ref) < _tolerance(dtype), (dtype, d, mask)


@pytest.mark.parametrize(
    "mask, bias_name",
    [
        ("none", "full"),
        ("top_left", "head_bcast"),
        ("bottom_right", "full"),
        ("bottom_right", "key_row"),
    ],
)
def test_ragged_lengths(mask, bias_name, tile):
    """Non-tile-multiple lengths take the ragged body, whose padded query rows and keys
    must not read past the bias (its loads are masked there), with full and
    broadcast (stride-0) biases. The factory refuses ragged + window."""
    dtype, d, hq, hkv, batch, sq, skv = "fp16", 128, 8, 8, 2, 300, 700
    spec = _spec(dtype, d, hq, hkv, batch=batch, sq=sq, skv=skv, tile=tile, mask=mask)
    assert spec.ragged
    shape = _BIAS_SHAPES[bias_name](batch, hq, sq, skv)
    q, k, v, bias = _inputs(dtype, d, hq, hkv, batch, sq, skv, shape)
    scale = 1.0 / math.sqrt(d)
    out = _run(spec, q, k, v, bias, scale)
    ref, _ = _reference(q, k, v, scale, bias, mask)
    assert _max_abs(out, ref) < _tolerance(dtype)


def test_padded_bias_row_pitch(tile):
    """A bias view with a padded Skv row pitch and a non-packed batch stride: the
    kernel reads the strides it is given, not the packed ones."""
    dtype, d, hq, hkv, batch, sq, skv = "bf16", 64, 4, 4, 2, 256, 256
    spec = _spec(
        dtype, d, hq, hkv, batch=batch, sq=sq, skv=skv, tile=tile, mask="top_left"
    )
    q, k, v, _ = _inputs(dtype, d, hq, hkv, batch, sq, skv, (1,))
    storage = torch.randn(
        batch, hq + 1, sq, skv + 40, device="cuda", dtype=torch.float32
    )
    bias = storage[:, 1:, :, 8 : 8 + skv]
    assert bias.stride(-1) == 1 and not bias.is_contiguous()
    scale = 1.0 / math.sqrt(d)
    out = _run(spec, q, k, v, bias, scale)
    ref, _ = _reference(q, k, v, scale, bias.contiguous(), "top_left")
    assert _max_abs(out, ref) < _tolerance(dtype)


@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("mask", ["none", "top_left"])
def test_misaligned_bias_view(mask, d, tile):
    """The aligned body loads four consecutive keys with one 16-byte load that is
    only element-aligned: a view one element into its storage, with an odd row
    pitch, puts most of those loads off a 16-byte boundary. ``-inf`` entries ride
    along so a lane mix-up inside a load shows."""
    dtype, hq, hkv, batch, sq, skv = "bf16", 4, 2, 2, 256, 512
    spec = _spec(dtype, d, hq, hkv, batch=batch, sq=sq, skv=skv, tile=tile, mask=mask)
    assert not spec.ragged
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
    assert _max_abs(out, ref) < _tolerance(dtype)


def test_minus_inf_entries_mask_keys(tile):
    """A float mask spelled with ``-inf`` (what PyTorch SDPA makes of a boolean mask)
    hides those keys: here every odd key of every row, plus whole key blocks for one
    head."""
    dtype, d, hq, hkv, batch, sq, skv = "fp16", 128, 4, 4, 1, 256, 512
    spec = _spec(dtype, d, hq, hkv, batch=batch, sq=sq, skv=skv, tile=tile, mask="none")
    q, k, v, bias = _inputs(dtype, d, hq, hkv, batch, sq, skv, (batch, hq, sq, skv))
    bias[..., 1::2] = float("-inf")
    bias[:, 2, :, 128:384] = float("-inf")
    scale = 1.0 / math.sqrt(d)
    out = _run(spec, q, k, v, bias, scale)
    ref, _ = _reference(q, k, v, scale, bias, "none")
    assert _max_abs(out, ref) < _tolerance(dtype)


@pytest.mark.parametrize("mask", ["none", "top_left"])
def test_bias_below_the_mask_sentinel(mask, tile):
    """A bias below the kernel's -2**99 mask sentinel must not hand the row max to
    masked keys. Every key at -1e30 counts as masked (below about -1.1e29 the kernel
    treats a bias as -inf): causal rows 0..3 see only such keys and, with no mask,
    head 1 has nothing else, so those rows store 0 -- not the mean of the future keys
    and not NaN. Every other row matches the reference.

    This is a deliberate divergence, documented on run_attention_dense_torch: the
    reference (like PyTorch) gives a row of finite scores the softmax over them, here
    the uniform one, so the expected rows are set to 0 by hand below."""
    dtype, d, hq, hkv, batch, sq, skv = "bf16", 128, 4, 4, 1, 256, 256
    spec = _spec(dtype, d, hq, hkv, batch=batch, sq=sq, skv=skv, tile=tile, mask=mask)
    q, k, v, bias = _inputs(dtype, d, hq, hkv, batch, sq, skv, (batch, hq, sq, skv))
    bias[..., :4] = -1e30
    if mask == "none":
        bias[:, 1] = -1e30
    scale = 1.0 / math.sqrt(d)
    out = _run(spec, q, k, v, bias, scale)
    ref, _ = _reference(q, k, v, scale, bias, mask)
    if mask == "top_left":
        ref[:, :4] = 0.0  # [B, Sq, Hq, D]: rows 0..3 of every head
    else:
        ref[:, :, 1] = 0.0  # head 1, every row
    assert torch.isfinite(out).all()
    assert _max_abs(out, ref) < _tolerance(dtype)


def test_a_finfo_min_mask_matches_the_reference(tile):
    """HF-style float masks use finfo(float32).min, which the kernel treats as -inf.
    Here every odd key and the whole first KV tile are masked for every row; each
    row keeps live keys later on."""
    dtype, d, hq, hkv, batch, sq, skv = "fp16", 128, 4, 4, 1, 256, 512
    spec = _spec(dtype, d, hq, hkv, batch=batch, sq=sq, skv=skv, tile=tile, mask="none")
    q, k, v, bias = _inputs(dtype, d, hq, hkv, batch, sq, skv, (batch, hq, sq, skv))
    lowest = torch.finfo(torch.float32).min
    bias[..., 1::2] = lowest
    bias[..., :128] = lowest
    scale = 1.0 / math.sqrt(d)
    out = _run(spec, q, k, v, bias, scale)
    ref, _ = _reference(q, k, v, scale, bias, "none")
    assert not torch.isnan(out).any()
    assert _max_abs(out, ref) < _tolerance(dtype)


@pytest.mark.parametrize("lse", [False, True], ids=["no-lse", "lse"])
def test_a_row_with_no_finite_score_outputs_zero(lse, tile):
    """sdpa_reference (hipDNN semantics): a row whose every visible key has a -inf
    bias outputs O = 0 and LSE = -inf. Rows 0..63 see only keys 0..q under the
    causal mask and all of them are -inf; rows 64.. keep live keys."""
    dtype, d, hq, hkv, batch, sq, skv = "fp16", 128, 4, 4, 1, 256, 256
    spec = _spec(
        dtype,
        d,
        hq,
        hkv,
        batch=batch,
        sq=sq,
        skv=skv,
        tile=tile,
        mask="top_left",
        lse=lse,
    )
    q, k, v, bias = _inputs(dtype, d, hq, hkv, batch, sq, skv, (batch, hq, sq, skv))
    bias[:, :, :64, :64] = float("-inf")
    scale = 1.0 / math.sqrt(d)
    kw = {}
    if lse:
        kw["lse"] = torch.full(
            (batch, hq, sq, 1), float("nan"), device="cuda", dtype=torch.float32
        )
    out = _run(spec, q, k, v, bias, scale, **kw)
    ref, lse_ref = _reference(q, k, v, scale, bias, "top_left")
    assert not torch.isnan(out).any()
    assert out[:, :64].abs().max().item() == 0.0
    assert _max_abs(out[:, 64:], ref[:, 64:]) < _tolerance(dtype)
    if lse:
        got = kw["lse"][..., 0].cpu()
        assert torch.isneginf(got[:, :, :64]).all()
        assert (got[:, :, 64:] - lse_ref[:, :, 64:]).abs().max().item() < _tolerance(
            dtype
        )


@pytest.mark.parametrize("lse", [False, True], ids=["no-lse", "lse"])
def test_window_rows_past_the_keys(lse, tile):
    """Sq > Skv + W - 1: q-blocks past the keys have no visible key, so their first
    window tile would lie past the last KV tile, and the start tile is clamped to the
    last real one. Rows q >= Skv + W - 1 store O = 0 (and LSE = -inf), the rest match
    the reference. This pins the clamped path's outputs; an out-of-range bias read
    here neither faults (the caching allocator's block is larger) nor changes the
    output (the values are masked), so the IR golden is the clamp's regression
    gate."""
    dtype, d, hq, hkv, batch, sq, skv = "bf16", 128, 4, 4, 1, 1024, 256
    spec = _spec(
        dtype,
        d,
        hq,
        hkv,
        batch=batch,
        sq=sq,
        skv=skv,
        tile=tile,
        mask="window",
        lse=lse,
    )
    q, k, v, bias = _inputs(dtype, d, hq, hkv, batch, sq, skv, (1, 1, 1, skv))
    scale = 1.0 / math.sqrt(d)
    kw = {}
    if lse:
        kw["lse"] = torch.full(
            (batch, hq, sq, 1), float("nan"), device="cuda", dtype=torch.float32
        )
    out = _run(spec, q, k, v, bias, scale, **kw)
    ref, lse_ref = _reference(q, k, v, scale, bias, "window")
    dead = skv + _WINDOW - 1
    assert out[:, dead:].abs().max().item() == 0.0
    assert _max_abs(out, ref) < _tolerance(dtype)
    if lse:
        got = kw["lse"][..., 0].cpu()
        assert torch.isneginf(got[:, :, dead:]).all()
        live = (got[:, :, :dead] - lse_ref[:, :, :dead]).abs().max().item()
        assert live < _tolerance(dtype)


def test_varlen_window_sequences_past_their_keys(tile):
    """Varlen with a window: each sequence's q-blocks past its own keys clamp to its
    last KV tile, not the spec maxima's; its rows past its keys + W - 1 store exact
    zeros."""
    from rocke.numeric.sdpa_reference import sdpa_reference

    dtype, d, hq, hkv = "bf16", 128, 4, 4
    q_lens, kv_lens = (1024, 512), (256, 128)
    q_offsets = [0, *np.cumsum(q_lens).tolist()]
    k_offsets = [0, *np.cumsum(kv_lens).tolist()]
    base = _spec(
        dtype,
        d,
        hq,
        hkv,
        batch=2,
        sq=max(q_lens),
        skv=max(kv_lens),
        tile=tile,
        mask="window",
    )
    spec = replace(base, varlen=True, ragged=False)
    torch.manual_seed(7)
    tdt = _TORCH_DT[dtype]
    q = torch.randn(q_offsets[-1], hq, d, device="cuda", dtype=tdt)
    k = torch.randn(k_offsets[-1], hkv, d, device="cuda", dtype=tdt)
    v = torch.randn_like(k)
    bias = 2 * torch.randn(1, 1, 1, max(kv_lens), device="cuda", dtype=torch.float32)
    cu_q = torch.tensor(q_offsets, device="cuda", dtype=torch.int32)
    cu_k = torch.tensor(k_offsets, device="cuda", dtype=torch.int32)
    scale = 1.0 / math.sqrt(d)
    out = _run(spec, q, k, v, bias, scale, cu_seqlens_q=cu_q, cu_seqlens_kv=cu_k)
    ref, _ = sdpa_reference(
        q.float().cpu().numpy(),
        k.float().cpu().numpy(),
        v.float().cpu().numpy(),
        scale=scale,
        bias=bias.cpu().numpy().astype(np.float64),
        q_offsets=q_offsets,
        kv_offsets=k_offsets,
        **_band("window"),
    )
    assert _max_abs(out, torch.from_numpy(ref).float()) < _tolerance(dtype)
    for s, kv_len in enumerate(kv_lens):
        dead = q_offsets[s] + kv_len + _WINDOW - 1
        assert out[dead : q_offsets[s + 1]].abs().max().item() == 0.0, s


def test_nan_entries_mask_keys(tile):
    """A NaN bias entry masks its key (the floor clamp returns the non-NaN operand),
    like -inf: every odd key here. The output stays finite and matches the reference
    with NaN read as -inf."""
    dtype, d, hq, hkv, batch, sq, skv = "fp16", 128, 4, 4, 1, 256, 512
    spec = _spec(dtype, d, hq, hkv, batch=batch, sq=sq, skv=skv, tile=tile, mask="none")
    q, k, v, bias = _inputs(dtype, d, hq, hkv, batch, sq, skv, (batch, hq, sq, skv))
    bias[..., 1::2] = float("nan")
    scale = 1.0 / math.sqrt(d)
    out = _run(spec, q, k, v, bias, scale)
    ref, _ = _reference(
        q, k, v, scale, torch.nan_to_num(bias, nan=float("-inf")), "none"
    )
    assert torch.isfinite(out).all()
    assert _max_abs(out, ref) < _tolerance(dtype)


@pytest.mark.parametrize("mask", ["top_left", "window"])
def test_lse_includes_the_bias(mask, tile):
    """The natural-log LSE the kernel writes is that of the biased scores."""
    dtype, d, hq, hkv, batch, sq, skv = "bf16", 128, 8, 2, 2, 512, 512
    spec = _spec(
        dtype, d, hq, hkv, batch=batch, sq=sq, skv=skv, tile=tile, mask=mask, lse=True
    )
    q, k, v, bias = _inputs(dtype, d, hq, hkv, batch, sq, skv, (batch, 1, sq, skv))
    lse = torch.full(
        (batch, hq, sq, 1), float("nan"), device="cuda", dtype=torch.float32
    )
    scale = 1.0 / math.sqrt(d)
    out = _run(spec, q, k, v, bias, scale, lse=lse)
    ref, lse_ref = _reference(q, k, v, scale, bias, mask)
    assert _max_abs(out, ref) < _tolerance(dtype), mask
    assert (lse[..., 0].cpu() - lse_ref).abs().max().item() < _tolerance(dtype), mask


def test_sinks_with_bias(tile):
    """Sinks join the softmax denominator next to the biased scores."""
    dtype, d, hq, hkv, batch, sq, skv = "bf16", 128, 8, 2, 1, 512, 512
    spec = _spec(
        dtype,
        d,
        hq,
        hkv,
        batch=batch,
        sq=sq,
        skv=skv,
        tile=tile,
        mask="top_left",
        sinks=True,
    )
    q, k, v, bias = _inputs(dtype, d, hq, hkv, batch, sq, skv, (batch, hq, sq, skv))
    sinks = torch.randn(hq, device="cuda", dtype=_TORCH_DT[dtype])
    scale = 1.0 / math.sqrt(d)
    out = _run(spec, q, k, v, bias, scale, sinks=sinks)

    rep = hq // hkv
    qf = q.float().transpose(1, 2)
    kf = k.float().transpose(1, 2).repeat_interleave(rep, dim=1)
    vf = v.float().transpose(1, 2).repeat_interleave(rep, dim=1)
    s = qf @ kf.transpose(-1, -2) * scale + bias
    causal = torch.ones(sq, skv, device="cuda", dtype=torch.bool).tril()
    s = s.masked_fill(~causal, float("-inf"))
    sink = sinks.float().view(1, hq, 1, 1).expand(batch, hq, sq, 1)
    p = torch.softmax(torch.cat([s, sink], dim=-1), dim=-1)[..., :skv]
    ref = (p @ vf).transpose(1, 2).cpu()
    assert _max_abs(out, ref) < _tolerance(dtype)


@pytest.mark.parametrize(
    "q_lens, kv_lens, bias_name",
    [((256, 512), (128, 256), "full"), ((0, 256, 512), (64, 128, 256), "batch_bcast")],
    ids=["two-seqs-full", "empty-first-seq-batch-bcast"],
)
@pytest.mark.parametrize("lse", [False, True], ids=["no-lse", "lse"])
def test_varlen_indexes_the_bias_by_in_sequence_position(
    q_lens, kv_lens, bias_name, lse, tile
):
    """Packed varlen: sequence ``s`` reads bias rows ``[s|0, h, 0:len_q, 0:len_kv]``;
    a zero-length query sequence writes nothing. Checked against the oracle's ragged
    layout, which indexes the bias the same way; with LSE on, the [tokens, Hq, 1]
    LSE too."""
    from rocke.numeric.sdpa_reference import sdpa_reference

    dtype, d, hq, hkv = "fp16", 128, 4, 1
    nseq = len(q_lens)
    q_offsets = [0, *np.cumsum(q_lens).tolist()]
    k_offsets = [0, *np.cumsum(kv_lens).tolist()]
    max_q, max_kv = max(q_lens), max(kv_lens)
    base = _spec(
        dtype,
        d,
        hq,
        hkv,
        batch=nseq,
        sq=max_q,
        skv=max_kv,
        tile=tile,
        mask="top_left",
        lse=lse,
    )
    spec = replace(base, varlen=True, ragged=False)
    torch.manual_seed(5)
    tdt = _TORCH_DT[dtype]
    q = torch.randn(q_offsets[-1], hq, d, device="cuda", dtype=tdt)
    k = torch.randn(k_offsets[-1], hkv, d, device="cuda", dtype=tdt)
    v = torch.randn_like(k)
    shape = _BIAS_SHAPES[bias_name](nseq, hq, max_q, max_kv)
    bias = 2 * torch.randn(shape, device="cuda", dtype=torch.float32)
    cu_q = torch.tensor(q_offsets, device="cuda", dtype=torch.int32)
    cu_k = torch.tensor(k_offsets, device="cuda", dtype=torch.int32)
    scale = 1.0 / math.sqrt(d)
    kw = {}
    if lse:
        kw["lse"] = torch.full(
            (q_offsets[-1], hq, 1), float("nan"), device="cuda", dtype=torch.float32
        )
    out = _run(spec, q, k, v, bias, scale, cu_seqlens_q=cu_q, cu_seqlens_kv=cu_k, **kw)
    ref, lse_ref = sdpa_reference(
        q.float().cpu().numpy(),
        k.float().cpu().numpy(),
        v.float().cpu().numpy(),
        scale=scale,
        bias=bias.cpu().numpy().astype(np.float64),
        causal=True,
        q_offsets=q_offsets,
        kv_offsets=k_offsets,
    )
    assert _max_abs(out, torch.from_numpy(ref).float()) < _tolerance(dtype)
    if lse:
        got = kw["lse"][..., 0].cpu().numpy()
        assert np.abs(got - lse_ref).max() < _tolerance(dtype)


def test_an_lse_buffer_overlapping_the_bias_is_refused(tile):
    """The kernel writes LSE while other workgroups still read the bias, so an LSE
    buffer sharing storage with it is refused before launch. The alias is otherwise
    a valid key-row bias (fp32, contiguous along Skv), so only the overlap check
    can refuse it."""
    dtype, d, hq, hkv, sq, skv = "fp16", 128, 4, 1, 256, 256
    spec = _spec(
        dtype, d, hq, hkv, batch=1, sq=sq, skv=skv, mask="top_left", tile=tile, lse=True
    )
    q, k, v, _ = _inputs(dtype, d, hq, hkv, 1, sq, skv, (1, 1, 1, skv))
    lse_elems = hq * sq
    storage = torch.zeros(lse_elems + skv, device="cuda", dtype=torch.float32)
    lse = storage[:lse_elems].view(1, hq, sq, 1)
    # The bias starts on the last LSE element.
    bias = storage[lse_elems - 1 : lse_elems - 1 + skv].view(1, 1, 1, skv)
    with pytest.raises(ValueError, match="lse overlaps additive_bias"):
        _run(spec, q, k, v, bias, 1.0 / math.sqrt(d), lse=lse)


@pytest.mark.parametrize("name", ["cu_seqlens_q", "cu_seqlens_kv"])
def test_host_varlen_offsets_are_refused_without_lse(name, tile):
    """With LSE off nothing else checks the offsets' device; the kernel would
    dereference a host pointer."""
    dtype, d, hq, hkv = "fp16", 128, 4, 1
    q_offsets, k_offsets = [0, 256, 768], [0, 128, 384]
    base = _spec(
        dtype, d, hq, hkv, batch=2, sq=512, skv=256, mask="top_left", tile=tile
    )
    spec = replace(base, varlen=True, ragged=False)
    tdt = _TORCH_DT[dtype]
    q = torch.randn(q_offsets[-1], hq, d, device="cuda", dtype=tdt)
    k = torch.randn(k_offsets[-1], hkv, d, device="cuda", dtype=tdt)
    v = torch.randn_like(k)
    bias = torch.zeros(1, 1, 1, 256, device="cuda", dtype=torch.float32)
    offsets = {
        "cu_seqlens_q": torch.tensor(q_offsets, device="cuda", dtype=torch.int32),
        "cu_seqlens_kv": torch.tensor(k_offsets, device="cuda", dtype=torch.int32),
    }
    offsets[name] = offsets[name].cpu()
    with pytest.raises(ValueError, match=rf"{name} must be on the same device as q"):
        _run(spec, q, k, v, bias, 1.0 / math.sqrt(d), **offsets)


def test_dispatcher_binding_passes_the_bias(tile):
    """``dispatch_attention(...).bind_torch`` hands ``tensors['attn_bias']`` to the
    runner; leaving it out is refused rather than launched with a null pointer."""
    from dispatch.attention import AttentionRequest, dispatch_attention

    dtype, d, hq, hkv, batch, sq, skv = "bf16", 128, 8, 2, 2, 512, 512
    spec_id, knobs = _TILES[tile]
    result = dispatch_attention(
        AttentionRequest(
            batch=batch,
            nhead_q=hq,
            nhead_k=hkv,
            seqlen_q=sq,
            seqlen_k=skv,
            hdim_q=d,
            hdim_v=d,
            arch="gfx950",
            mask_type=1,
            dtype=dtype,
            algorithm="attention_dense_grid",
            spec_id=spec_id,
            tuning_knobs=knobs,
            use_attn_bias=True,
        )
    )
    assert result.spec.kernel_spec == _spec(
        dtype, d, hq, hkv, batch=batch, sq=sq, skv=skv, tile=tile, mask="top_left"
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
    assert _max_abs(out, ref) < _tolerance(dtype)


def test_runtime_shape_bias_kernel_serves_another_shape(tile):
    """The aligned bias kernel takes its shape and bias strides at run time, so one
    compiled binary serves a second batch and length."""
    dtype, d, hq, hkv = "fp16", 64, 8, 8
    for batch, seq in ((1, 256), (3, 768)):
        spec = _spec(
            dtype, d, hq, hkv, batch=batch, sq=seq, skv=seq, tile=tile, mask="top_left"
        )
        assert spec.runtime_shape
        q, k, v, bias = _inputs(dtype, d, hq, hkv, batch, seq, seq, (batch, 1, 1, seq))
        scale = 1.0 / math.sqrt(d)
        out = _run(spec, q, k, v, bias, scale)
        ref, _ = _reference(q, k, v, scale, bias, "top_left")
        assert _max_abs(out, ref) < _tolerance(dtype), (batch, seq)
