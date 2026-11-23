# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Torch-free on-GPU checks of the gfx942 dense kernel's additive bias.

The CI GPU lane has no torch, so the full matrix in
``test_attention_dense_bias_gfx942_numeric.py`` (driven through
``run_attention_dense_torch``) skips there. This file launches the same kernels
through ``rocke.runtime`` from numpy arrays and checks a representative subset against
the float64 oracle in ``_dense_bias_common``: one bias shape per mask on both grids,
both head sizes and dtypes, 4-wide and scalar bias loads on misaligned views, rows with
no live key (``-inf`` entries) with and without LSE and sinks, ``finfo(float32).min``
rows (the uniform softmax), NaN entries (a NaN row), and window q-blocks past the keys,
alone and together with those bias classes. It also pins the two load forms against each
other: 4-wide and scalar loads add the same scores, so their outputs are bitwise equal,
special values included (gfx942 has no ragged body to act as a scalar control, so the
knob is the control).

Specs come from the dispatch factory. The launch packs the kernargs the way
``run_attention_dense_torch`` does, minus its host checks (those are the host twin's:
``test_attention_dense_bias_gfx942.py``).

    HIP_VISIBLE_DEVICES=0 python -m pytest tests/test_attention_dense_bias_gfx942_gpu.py

Set ``ROCKE_REQUIRE_GFX942=1`` to fail instead of skip without a gfx942.
"""

from __future__ import annotations

import math
from dataclasses import replace

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
    bias_strides,
    check_window_row_classes,
    device,
    extents,
    from_bits,
    make_inputs,
    max_abs,
    nan_filled,
    reference,
    round_to,
    tolerance,
    to_bits,
    window_row_classes_inputs,
)
from dispatch.attention import AttentionRequest, tuning_spec_with_knobs
from kernels.common._attention_additive_bias import additive_bias_launch_values
from kernels.common.attention_dense_spec import attention_dense_cache_key
from kernels.gfx942.attention_dense import (
    attention_dense_block,
    attention_dense_grid,
    attention_dense_runtime_args,
    attention_dense_signature,
    build_attention_dense,
)
from rocke.helpers.compile import compile_kernel
from rocke.runtime import KernelLauncher, LaunchConfig
from rocke.runtime.hip_module import Runtime
from rocke.runtime.host_buffers import as_u8_buffer
from rocke.tests.gpu import current_device_arch_in

pytestmark = pytest.mark.gpu

# The request's mask kind; the window is a causal band with a left bound.
_MASK_TYPES = {"none": 0, "top_left": 1, "bottom_right": 2, "window": 1}
_LAUNCHERS: dict = {}


@pytest.fixture(scope="module", autouse=True)
def gfx942():
    # Skips off gfx942; ROCKE_REQUIRE_GFX942=1 fails instead (a lane that lost its GPU).
    current_device_arch_in(("gfx942",), subject="the gfx942 dense bias lane")


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
    persistent=None,
    sinks=False,
    lse=False,
    x4=None,
    tile=None,
):
    """The shipped bias spec for the request, from the dispatch factory. ``persistent``
    picks the grid and ``tile`` the ``(block_m, block_n)`` tile as knobs on that spec
    (None: the factory's own choice); ``x4`` forces the bias load form (None: the
    kernel's policy)."""
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
    knobs = {} if persistent is None else {"persistent": persistent}
    if tile is not None:
        knobs.update(block_m=tile[0], block_n=tile[1])
    spec = tuning_spec_with_knobs(req, "gfx942_dense", knobs).kernel_spec
    assert spec.use_additive_bias
    assert spec.kv_segments == 1 and not spec.ragged
    if persistent is not None:
        assert spec.persistent is persistent
    if tile is not None:
        assert (spec.block_m, spec.block_n) == tile
    return spec if x4 is None else replace(spec, use_bias_x4=x4)


def _launch(spec, q, k, v, bias, scale, *, sinks=None, bias_view=None):
    """Launch ``spec`` on float32 arrays of ``spec.dtype`` values: ``q``/``k``/``v``
    dense [B, S, H, D], ``bias`` a contiguous rank-4 fp32 array, ``sinks`` [Hq] values
    of ``spec.dtype``. ``bias_view = (storage, offset, strides)`` instead passes a
    strided view: a 1-D fp32 ``storage``, the element ``offset`` of the view in it and
    the view's ``(batch, head, sq)`` element strides (``bias`` is then ignored).
    Returns ``(O, LSE or None)`` as float32 arrays in the kernel's layouts (O
    [B, Sq, Hq, D], LSE [B, Hq, Sq])."""
    key = attention_dense_cache_key(spec, arch="gfx942")
    launcher = _LAUNCHERS.get(key)
    if launcher is None:
        art = compile_kernel(
            build_attention_dense(spec, arch="gfx942"),
            arch="gfx942",
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
    if spec.use_sinks:
        vals["sink_ptr"] = device(to_bits(sinks, spec.dtype))
    lse, lse_mem = None, None
    if spec.emit_lse:
        lse = np.empty((spec.batch, spec.num_query_heads, spec.seqlen_q), np.float32)
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


def _sinks(dtype, hq, *, seed=11, shift=0.0):
    """Per-head sink logits ``shift + N(0, 1)``, rounded to ``dtype`` (what the kernel
    reads). A shift of 0 keeps them O(1), small enough for a keyless row's LSE to equal
    its sink within the tolerance; a bottom-right row sees so many keys that such sinks
    barely move its output, so the test that must observe them shifts them up."""
    values = shift + np.random.default_rng(seed).standard_normal(hq)
    return round_to(values, dtype)


def _bias_view(batch, hq, sq, skv, *, offset=0, pitch=None, expand=False, seed=1):
    """``(storage, offset, strides)`` of a bias view, for the layouts a host tensor can
    have: ``offset`` elements into its storage (the view's base address is that many
    dwords past a 16-byte-aligned allocation), rows ``pitch`` elements apart (default
    ``skv``), and with ``expand`` full-extent batch and head axes of stride 0 (one
    ``[Sq, Skv]`` plane read by every batch and head, as torch's ``expand`` makes it,
    not a size-1 broadcast). ``strides`` is the kernels' ``(batch, head, sq)``. Every
    7th element is ``-inf`` so a lane mix-up inside a load shows."""
    pitch = skv if pitch is None else pitch
    if expand:
        strides, planes = (0, 0, pitch), 1
    else:
        strides, planes = (hq * sq * pitch, sq * pitch, pitch), batch * hq
    rng = np.random.default_rng(seed)
    storage = (2 * rng.standard_normal(offset + planes * sq * pitch)).astype(np.float32)
    storage[offset + 3 :: 7] = -np.inf
    return storage, offset, strides


def _misaligned_view(batch, hq, sq, skv, *, seed=1):
    """A view one element into its storage with an odd row pitch (``skv + 5``): the
    4-wide form's 16-byte loads start at every 4-byte residue of a 16-byte line."""
    # Group loads start at element 1 + s*pitch + 4m of a 16-byte-aligned device
    # allocation (the batch and head terms are multiples of 4): every 4-byte residue.
    assert {(1 + r * (skv + 5)) % 4 for r in range(4)} == {0, 1, 2, 3}
    return _bias_view(batch, hq, sq, skv, offset=1, pitch=skv + 5, seed=seed)


def _view_array(view, shape):
    """The ``[B, Hq, Sq, Skv]`` array a ``_bias_view`` describes."""
    storage, offset, strides = view
    return np.lib.stride_tricks.as_strided(
        storage[offset:],
        shape=shape,
        strides=tuple(4 * s for s in (*strides, 1)),
        writeable=False,
    )


@pytest.mark.parametrize("mask", MASKS)
def test_one_bias_shape_per_mask(mask, persistent):
    """bf16 D128 GQA-4, B=2, one bias shape per mask (full, head-broadcast,
    batch-broadcast, both-broadcast); bottom-right runs Sq != Skv so its diagonal moves.
    The bias must also move the output far past the tolerance, or a kernel that ignored
    it would pass."""
    dtype, d, hq, hkv, batch = "bf16", 128, 8, 2, 2
    sq, skv = extents(mask)
    shape = BIAS_SHAPES[SHAPE_FOR_MASK[mask]](batch, hq, sq, skv)
    spec = _spec(
        dtype, d, hq, hkv, batch=batch, sq=sq, skv=skv, mask=mask, persistent=persistent
    )
    q, k, v, bias = make_inputs(dtype, d, hq, hkv, batch, sq, skv, shape)
    scale = 1.0 / math.sqrt(d)
    out, _ = _launch(spec, q, k, v, bias, scale)
    ref, _ = reference(q, k, v, scale, bias, mask)
    assert max_abs(out, ref) < tolerance(dtype)
    plain, _ = reference(q, k, v, scale, None, mask)
    assert max_abs(out, plain) > 5 * tolerance(dtype)


@pytest.mark.parametrize("mask", MASKS)
@pytest.mark.parametrize(
    "dtype, d, hq, hkv",
    [("fp16", 64, 16, 16), ("bf16", 64, 16, 4), ("fp16", 128, 16, 4)],
    ids=["fp16-d64-mha", "bf16-d64-gqa", "fp16-d128-gqa"],
)
def test_dtype_x_head_size(dtype, d, hq, hkv, mask, persistent):
    """Both dtypes and both head sizes under every mask, on both grids, with a
    query-broadcast ``[B, Hq, 1, Skv]`` bias (a head index taken from the KV head would
    show: every query head has its own row)."""
    batch = 2
    sq, skv = extents(mask)
    spec = _spec(
        dtype, d, hq, hkv, batch=batch, sq=sq, skv=skv, mask=mask, persistent=persistent
    )
    q, k, v, bias = make_inputs(dtype, d, hq, hkv, batch, sq, skv, (batch, hq, 1, skv))
    scale = 1.0 / math.sqrt(d)
    out, _ = _launch(spec, q, k, v, bias, scale)
    ref, _ = reference(q, k, v, scale, bias, mask)
    assert max_abs(out, ref) < tolerance(dtype)


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
    q, k, v, bias = make_inputs(dtype, d, hq, hkv, batch, sq, skv, (batch, hq, sq, skv))
    scale = 1.0 / math.sqrt(d)
    out, _ = _launch(spec, q, k, v, bias, scale)
    ref, _ = reference(q, k, v, scale, bias, mask)
    assert max_abs(out, ref) < tolerance(dtype)


@pytest.mark.parametrize("x4", [False, True], ids=["scalar", "x4"])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("mask", ["none", "top_left"])
def test_misaligned_bias_view(mask, d, x4):
    """The 4-wide form loads four consecutive keys with one 16-byte load that is only
    element-aligned: a view one element into its storage, with an odd row pitch, puts
    most of those loads off a 16-byte boundary. ``-inf`` entries ride along so a lane
    mix-up inside a load shows. (Each head size runs the grid the factory ships for
    it: persistent at D128, the default grid at D64.)"""
    dtype, hq, hkv, batch, sq, skv = "bf16", 4, 2, 2, 512, 512
    spec = _spec(dtype, d, hq, hkv, batch=batch, sq=sq, skv=skv, mask=mask, x4=x4)
    q, k, v, _ = make_inputs(dtype, d, hq, hkv, batch, sq, skv, (1,))
    view = _misaligned_view(batch, hq, sq, skv)
    scale = 1.0 / math.sqrt(d)
    out, _ = _launch(spec, q, k, v, None, scale, bias_view=view)
    ref, _ = reference(q, k, v, scale, _view_array(view, (batch, hq, sq, skv)), mask)
    assert max_abs(out, ref) < tolerance(dtype)


# The memory layouts of a host bias tensor the 16-byte loads must read right at dword
# alignment, as ``_bias_view`` keyword arguments (rows are Skv = 512 long).
_LAYOUTS = {
    "offset-1": {"offset": 1},
    "padded-rows": {"pitch": 513},
    "expanded": {"expand": True},
    "expanded-offset-padded": {"expand": True, "offset": 1, "pitch": 513},
}


@pytest.mark.parametrize("x4", [False, True], ids=["scalar", "x4"])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("layout", list(_LAYOUTS))
def test_bias_memory_layouts(layout, d, x4, persistent):
    """A base one dword into the allocation, rows padded by one element (every row then
    starts at the next dword residue) and full-extent batch and head axes with stride 0
    (an ``expand``), alone and together, on both grids and with both load forms forced:
    the 16-byte loads are element-aligned only, and a stride-0 axis must not be taken
    for a size-1 broadcast or the other way round."""
    dtype, hq, hkv, batch, sq, skv = "bf16", 4, 2, 2, 512, 512
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
        x4=x4,
    )
    q, k, v, _ = make_inputs(dtype, d, hq, hkv, batch, sq, skv, (1,))
    view = _bias_view(batch, hq, sq, skv, **_LAYOUTS[layout])
    bias = _view_array(view, (batch, hq, sq, skv))
    scale = 1.0 / math.sqrt(d)
    out, _ = _launch(spec, q, k, v, None, scale, bias_view=view)
    ref, _ = reference(q, k, v, scale, bias, "top_left")
    assert max_abs(out, ref) < tolerance(dtype)
    plain, _ = reference(q, k, v, scale, None, "top_left")
    assert max_abs(out, plain) > 5 * tolerance(dtype)


# (mask, LSE, sinks): the plain causal body, the LSE epilogue, and the window body
# with sinks and LSE (the largest register footprint of the bias modes).
_WIDTH_MODES = [
    ("top_left", False, False),
    ("top_left", True, False),
    ("window", True, True),
]


@pytest.mark.parametrize("mask, lse, sinks", _WIDTH_MODES, ids=["plain", "lse", "all"])
@pytest.mark.parametrize("d", [64, 128])
def test_wide_bias_loads_match_the_scalar_loads_bitwise(
    d, mask, lse, sinks, persistent
):
    """The two load forms read the same keys and add the same floored bias to the same
    score, so a kernel built with each gives the same output (and LSE) bit for bit,
    on the misaligned view of the test above. A form that shifted a lane's keys or
    floored differently would differ; the reference check cannot see a 1-ulp
    difference, this can."""
    dtype, hq, hkv, batch, sq, skv = "bf16", 4, 2, 2, 256, 512
    kw = dict(batch=batch, sq=sq, skv=skv, mask=mask, lse=lse, sinks=sinks)
    wide = _spec(dtype, d, hq, hkv, persistent=persistent, x4=True, **kw)
    scalar = _spec(dtype, d, hq, hkv, persistent=persistent, x4=False, **kw)
    assert wide.resolved_use_bias_x4() and not scalar.resolved_use_bias_x4()
    q, k, v, _ = make_inputs(dtype, d, hq, hkv, batch, sq, skv, (1,))
    view = _misaligned_view(batch, hq, sq, skv)
    scale = 1.0 / math.sqrt(d)
    values = _sinks(dtype, hq) if sinks else None
    got = _launch(wide, q, k, v, None, scale, sinks=values, bias_view=view)
    want = _launch(scalar, q, k, v, None, scale, sinks=values, bias_view=view)
    # One side is pinned to the oracle, so the equality below cannot be two copies of
    # the same wrong answer; the bits are compared, so -0.0 vs 0.0 or two NaN
    # payloads would not pass as equal.
    ref, ref_lse = reference(
        q, k, v, scale, _view_array(view, (batch, hq, sq, skv)), mask, values
    )
    assert max_abs(want[0], ref) < tolerance(dtype)
    np.testing.assert_array_equal(got[0].view(np.uint32), want[0].view(np.uint32))
    if lse:
        np.testing.assert_allclose(want[1], ref_lse, rtol=0, atol=tolerance(dtype))
        np.testing.assert_array_equal(got[1].view(np.uint32), want[1].view(np.uint32))


@pytest.mark.parametrize("lse", [False, True], ids=["no-lse", "lse"])
def test_wide_and_scalar_loads_agree_on_special_values(lse, persistent):
    """Both load forms apply the same floor and NaN guard to every element they read, so
    a bias with ``finfo(float32).min``, ``-inf`` and NaN entries in every lane position
    of a 4-key group gives the same output (and LSE) bit for bit in both, and the rows
    the entries leave live match the reference. Head 0 has finfo.min on every 4th key
    from key 1, head 1 -inf on rows 3..5, head 2 a NaN on one visible key of row 100
    (that row is NaN in both forms, and no other), head 3 -inf on keys 2, 6, 10, ..."""
    dtype, d, hq, hkv, batch, sq, skv = "bf16", 64, 4, 4, 1, 256, 512
    kw = dict(batch=batch, sq=sq, skv=skv, mask="top_left", lse=lse)
    wide = _spec(dtype, d, hq, hkv, persistent=persistent, x4=True, **kw)
    scalar = _spec(dtype, d, hq, hkv, persistent=persistent, x4=False, **kw)
    assert wide.resolved_use_bias_x4() and not scalar.resolved_use_bias_x4()
    q, k, v, bias = make_inputs(dtype, d, hq, hkv, batch, sq, skv, (batch, hq, sq, skv))
    bias[:, 0, :, 1::4] = np.finfo(np.float32).min
    # Rows 40..43 of head 0 are finfo.min on EVERY key: only the floor keeps them live
    # (the uniform softmax over their visible keys), so both forms must floor alike.
    floored = slice(40, 44)
    bias[:, 0, floored, :] = np.finfo(np.float32).min
    bias[:, 1, 3:6, :] = -np.inf
    bias[0, 2, 100, 13] = np.nan
    bias[:, 3, :, 2::4] = -np.inf
    scale = 1.0 / math.sqrt(d)
    got = _launch(wide, q, k, v, bias, scale)
    want = _launch(scalar, q, k, v, bias, scale)
    np.testing.assert_array_equal(got[0].view(np.uint32), want[0].view(np.uint32))
    expect_nan = np.zeros((batch, sq, hq), bool)
    expect_nan[0, 100, 2] = True
    np.testing.assert_array_equal(np.isnan(want[0]).all(-1), expect_nan)
    assert not (np.isnan(want[0]).any(-1) & ~expect_nan).any()
    ref, ref_lse = reference(q, k, v, scale, bias, "top_left")
    live = ~expect_nan
    assert np.abs(want[0] - ref).max(-1)[live].max() < tolerance(dtype)
    assert np.abs(want[0][:, 3:6, 1]).max() == 0.0
    if lse:
        np.testing.assert_array_equal(got[1].view(np.uint32), want[1].view(np.uint32))
        lse_got = want[1].transpose(0, 2, 1)
        np.testing.assert_array_equal(np.isnan(lse_got), expect_nan)
        assert np.isneginf(lse_got[:, 3:6, 1]).all()
        want_lse = ref_lse.transpose(0, 2, 1)
        finite = live & np.isfinite(want_lse)
        assert np.isfinite(lse_got[:, floored, 0]).all()
        assert (lse_got[:, floored, 0] < -1e28).all()  # about -2**97
        finite[:, floored, 0] = False  # the reference's LSE there is about finfo.min
        with np.errstate(invalid="ignore"):  # -inf - -inf, in rows `finite` leaves out
            diff = np.abs(lse_got - want_lse)
        assert diff[finite].max() < tolerance(dtype)


@pytest.mark.parametrize("x4", [False, True], ids=["scalar", "x4"])
@pytest.mark.parametrize("lse", [False, True], ids=["no-lse", "lse"])
def test_rows_with_no_live_key_store_zero(lse, x4, persistent):
    """Rows 0..63 see only keys 0..q under the causal mask, all -inf: they store
    O = 0 and LSE = -inf, the hipDNN reference's answer for a row of -inf scores. The
    -inf is not a NaN: both load forms keep it as -inf (a NaN would make its row NaN,
    see ``test_a_nan_bias_makes_its_row_nan``)."""
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
        x4=x4,
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


@pytest.mark.parametrize("x4", [False, True], ids=["scalar", "x4"])
@pytest.mark.parametrize("lse", [False, True], ids=["no-lse", "lse"])
@pytest.mark.parametrize("mask", ["none", "top_left"])
def test_finfo_min_rows_are_the_uniform_softmax(mask, lse, x4, persistent):
    """A row whose every visible key has bias finfo(float32).min (HF's padding value)
    has no key above the others: O is the mean of V over the visible keys, as in the
    reference and PyTorch (here head 1, every row; V has mean 1, so zeros fail), and
    the LSE is finite, about the kernel's -2**97 floor. Head 2 mixes finfo.min on even
    keys with -inf on odd ones (their V shifted by 10, for that head only): uniform
    over the even keys only, -inf and finfo.min being different keys."""
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
        x4=x4,
    )
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
    q, k, v, bias = make_inputs(dtype, d, hq, hkv, batch, sq, skv, (batch, hq, sq, skv))
    bias[:, 1] = -np.inf
    scale = 1.0 / math.sqrt(d)
    out, got_lse = _launch(spec, q, k, v, bias, scale)
    ref, ref_lse = reference(q, k, v, scale, bias, "none")
    assert np.abs(out[:, :, 1]).max() == 0.0
    assert max_abs(out, ref) < tolerance(dtype)
    if lse:
        assert np.isneginf(got_lse[:, 1]).all()
        assert max_abs(got_lse[:, [0, 2, 3]], ref_lse[:, [0, 2, 3]]) < tolerance(dtype)


@pytest.mark.parametrize("x4", [False, True], ids=["scalar", "x4"])
@pytest.mark.parametrize("lse", [False, True], ids=["no-lse", "lse"])
@pytest.mark.parametrize("mask", ["none", "top_left"])
def test_a_nan_bias_makes_its_row_nan(mask, lse, x4, persistent):
    """A NaN bias on a visible key makes its whole row NaN (O and LSE), as in the
    reference. Head 0 row 10 has one NaN key (key 7, visible under every mask), head 2
    row 20 has NaN on every key (the max ignores a NaN score, so only the NaN
    denominator keeps the row from reading as dead), and head 1 row 5 has a NaN at
    key 200: visible only without the mask, so under the causal mask the key is
    skipped, whatever its bias, and the row stays finite. Every other row matches the
    reference."""
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
        x4=x4,
    )
    q, k, v, bias = make_inputs(dtype, d, hq, hkv, batch, sq, skv, (batch, hq, sq, skv))
    bias[0, 0, 10, 7] = np.nan
    bias[0, 2, 20, :] = np.nan
    bias[0, 1, 5, 200] = np.nan
    expect_nan = np.zeros((batch, sq, hq), bool)  # [B, Sq, Hq]
    expect_nan[0, 10, 0] = expect_nan[0, 20, 2] = True
    expect_nan[0, 5, 1] = mask == "none"
    scale = 1.0 / math.sqrt(d)
    out, got_lse = _launch(spec, q, k, v, bias, scale)
    ref, ref_lse = reference(q, k, v, scale, bias, mask)  # reads NaN as -inf
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


@pytest.mark.parametrize("value", [-np.inf, -1e30], ids=["-inf", "below-floor"])
@pytest.mark.parametrize("lse", [False, True], ids=["no-lse", "lse"])
def test_a_keyless_row_with_sinks_keeps_the_sink(lse, value, persistent):
    """With sinks, a row whose keys carry no weight still has its sink in the softmax:
    O = 0 and LSE = the sink logit. Rows 0..63 of a causal mask see only keys 0..q, all
    at ``value``: -inf, or a finite bias far below the floor (raised to it: its keys
    get P = 0 next to the sink, which is the reference's answer too)."""
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
    q, k, v, bias = make_inputs(dtype, d, hq, hkv, batch, sq, skv, (batch, hq, sq, skv))
    bias[:, :, :64, :64] = value
    sinks = _sinks(dtype, hq)
    scale = 1.0 / math.sqrt(d)
    out, got_lse = _launch(spec, q, k, v, bias, scale, sinks=sinks)
    ref, ref_lse = reference(q, k, v, scale, bias, "top_left", sinks)
    assert np.abs(out[:, :64]).max() == 0.0
    assert max_abs(out, ref) < tolerance(dtype)
    if lse:
        assert max_abs(got_lse, ref_lse) < tolerance(dtype)
        assert max_abs(got_lse[:, :, :64], sinks[None, :, None]) < 1e-6


@pytest.mark.parametrize("lse", [False, True], ids=["no-lse", "lse"])
def test_window_rows_past_the_keys(lse, persistent):
    """Sq > Skv + W - 1: the last q-blocks have no visible key. Rows q >= Skv + W - 1
    store O = 0 (and LSE = -inf, with LSE on), without sinks too: the O store zeroes
    rows the window leaves without a key, and the bias's live-row test joins that
    predicate. The rest match the reference. The bias is a key row, so a read past it
    for those q-blocks would be a read past its allocation (the default grid runs
    their KV loop zero times, the persistent grid clamps it to one tile)."""
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


# (lse, sinks, x4): the four corners that matter, each on both grids. With LSE the
# epilogue folds the bias's live-row test into ``row_has_key``; without it that happens
# only under a window without sinks, and with sinks the zeros come from P = 0 alone.
_ROW_CLASS_MODES = [
    (False, False, False),
    (True, False, True),
    (False, True, True),
    (True, True, False),
]


@pytest.mark.parametrize(
    "lse, sinks, x4",
    _ROW_CLASS_MODES,
    ids=["plain", "lse", "sinks", "lse-sinks"],
)
def test_the_row_classes_of_a_window_with_a_bias(lse, sinks, x4, persistent):
    """One launch, four bias classes (one per head; see ``window_row_classes_inputs``)
    under a window that leaves the last q-blocks without a key: keyless-by-window rows
    store zeros in every head, whatever their bias (a finfo.min row is the uniform
    softmax only where it has keys); an all -inf head is zeros; a finfo.min head is the
    uniform softmax over its window; a NaN on a visible key makes only that row NaN,
    while a NaN on a key the window removes (in a live row and in a keyless one) is
    ignored. With sinks the sink logit is the LSE of every row with no weight, and it
    outweighs a finfo.min head. Without LSE the live-row test zeroes through rcp(l);
    with it, through the O store's mask."""
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
        x4=x4,
    )
    q, k, v, bias = window_row_classes_inputs(dtype, d=d)
    values = _sinks(dtype, hq, shift=1.0) if sinks else None
    scale = 1.0 / math.sqrt(d)
    out, got_lse = _launch(spec, q, k, v, bias, scale, sinks=values)
    ref, ref_lse = reference(q, k, v, scale, bias, "window", values)
    assert 0 < ROW_CLASS_DEAD_FROM < sq
    check_window_row_classes(out, got_lse, ref, ref_lse, sinks=values, dtype=dtype)


@pytest.mark.parametrize("lse", [False, True], ids=["no-lse", "lse"])
def test_window_rows_past_the_keys_with_sinks(lse, persistent):
    """Sq > Skv + W - 1: the last q-blocks have no visible key. With sinks their rows
    are softmax([sink]) over no values: O = 0, LSE = the sink logit; the rest match the
    reference. The bias is a key row, so a read past it for those q-blocks would be a
    read past its allocation."""
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
    q, k, v, bias = make_inputs(dtype, d, hq, hkv, batch, sq, skv, (1, 1, 1, skv))
    sinks = _sinks(dtype, hq)
    scale = 1.0 / math.sqrt(d)
    out, got_lse = _launch(spec, q, k, v, bias, scale, sinks=sinks)
    ref, ref_lse = reference(q, k, v, scale, bias, "window", sinks)
    dead = skv + WINDOW - 1
    assert np.abs(out[:, dead:]).max() == 0.0
    assert max_abs(out, ref) < tolerance(dtype)
    if lse:
        assert max_abs(got_lse, ref_lse) < tolerance(dtype)
        assert max_abs(got_lse[:, :, dead:], sinks[None, :, None]) < 1e-6


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
    q, k, v, bias = make_inputs(dtype, d, hq, hkv, batch, sq, skv, (batch, 1, sq, skv))
    scale = 1.0 / math.sqrt(d)
    out, got_lse = _launch(spec, q, k, v, bias, scale)
    ref, ref_lse = reference(q, k, v, scale, bias, mask)
    assert max_abs(out, ref) < tolerance(dtype), mask
    assert max_abs(got_lse, ref_lse) < tolerance(dtype), mask


@pytest.mark.parametrize("mask", ["top_left", "bottom_right", "window"])
def test_sinks_with_bias(mask, persistent):
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
    )
    q, k, v, bias = make_inputs(dtype, d, hq, hkv, batch, sq, skv, (batch, hq, sq, skv))
    sinks = _sinks(dtype, hq, shift=8.0)
    scale = 1.0 / math.sqrt(d)
    out, _ = _launch(spec, q, k, v, bias, scale, sinks=sinks)
    ref, _ = reference(q, k, v, scale, bias, mask, sinks)
    assert max_abs(out, ref) < tolerance(dtype)
    no_sinks, _ = reference(q, k, v, scale, bias, mask)
    assert max_abs(out, no_sinks) > 5 * tolerance(dtype)


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
        q, k, v, bias = make_inputs(
            dtype, d, hq, hkv, batch, seq, seq, (batch, hq, 1, seq)
        )
        scale = 1.0 / math.sqrt(d)
        out, _ = _launch(spec, q, k, v, bias, scale)
        ref, _ = reference(q, k, v, scale, bias, "top_left")
        assert max_abs(out, ref) < tolerance(dtype), (batch, seq, hq, hkv)
    assert len({attention_dense_cache_key(s, arch="gfx942") for s in specs}) == 1
