# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Arch-neutral pieces of the dense-attention additive-bias tests.

The numeric lanes (torch: ``test_attention_dense_bias_<arch>_numeric.py``; torch-free:
``..._gpu.py``) check one kernel family against the same float64 oracle, with the same
inputs, bias shapes, band names and tolerances; this module holds them so a second arch
adds only its launch code. numpy only: no torch, no GPU, no kernel module at import, so
the CPU lane can import it too (``test_attention_dense_bias_gfx942.py`` checks the
oracle's sink handling here, on the host, before a GPU test relies on it). The two
device-buffer helpers import ``rocke.runtime`` when called, not here; the
kernel-constants check at the end imports ``pytest`` and the shared bias helper when
called, and reads the source of the kernel module it is given.

Layout: dense inputs are ``q [B, Sq, Hq, D]`` and ``k``/``v [B, Skv, Hkv, D]``, the
kernels' own; the oracle returns ``O [B, Sq, Hq, D]`` and ``LSE [B, Hq, Sq]``.

The gfx950 and gfx942 lanes both import this module; what stays in a lane's own file is
its spec factory, its launch code and, for the torch lanes, the torch-side inputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rocke.numeric.bf16 import bf16_to_f32, f32_to_bf16
from rocke.numeric.sdpa_reference import sdpa_reference

# Sliding-window width: a multiple of every block_n a dense bias kernel runs.
WINDOW = 128

# The band each mask name stands for in the oracle. The dense window is a causal
# band: it keeps keys with ``q - k < WINDOW``, i.e. a left bound of WINDOW - 1.
MASKS = ("none", "top_left", "bottom_right", "window")

# Bias extents: every axis but the key one full or broadcast (extent 1).
BIAS_SHAPES = {
    "full": lambda b, h, sq, skv: (b, h, sq, skv),
    "batch_bcast": lambda b, h, sq, skv: (1, h, sq, skv),
    "head_bcast": lambda b, h, sq, skv: (b, 1, sq, skv),
    "both_bcast": lambda b, h, sq, skv: (1, 1, sq, skv),
    "key_row": lambda b, h, sq, skv: (1, 1, 1, skv),
}

# One bias shape per mask, so a dtype x head-size x mask sweep also walks every shape.
SHAPE_FOR_MASK = {
    "none": "full",
    "top_left": "head_bcast",
    "bottom_right": "batch_bcast",
    "window": "both_bcast",
}


# The kernels raise a finite bias below -2**97 to that floor (so ``finfo(float32).min``
# is an ordinary, very negative score: a row of them is the uniform softmax, as in the
# reference and PyTorch). Only ``-inf`` masks a key. A NaN bias on a visible key makes
# its row NaN (O and LSE), as in the reference.


def tolerance(dtype: str) -> float:
    """The dense numeric lane's ``max_abs`` tolerance: fp16 2e-2, bf16 4e-2. Outputs
    are O(1) (unit-normal V averaged under a softmax), the scores' error is the
    16-bit rounding of Q/K/V's products and the PV accumulation over at most a few
    thousand keys."""
    return {"fp16": 2e-2, "bf16": 4e-2}[dtype]


def extents(mask: str, base: int = 512) -> tuple[int, int]:
    """``(Sq, Skv)`` for ``mask``: bottom-right runs Sq < Skv so its diagonal moves
    (a bottom-right band with Sq == Skv is plain causal)."""
    return (base // 2, base) if mask == "bottom_right" else (base, base)


def band_kwargs(mask: str) -> dict:
    """``sdpa_reference`` band arguments for ``mask`` (one of :data:`MASKS`)."""
    if mask == "top_left":
        return {"causal": True}
    if mask == "bottom_right":
        return {"causal_bottom_right": True}
    if mask == "window":
        return {"left_bound": WINDOW - 1, "right_bound": 0, "top_left": True}
    if mask == "none":
        return {}
    raise ValueError(f"unknown mask {mask!r}")


# --------------------------------------------------------------------------- #
# 16-bit storage
# --------------------------------------------------------------------------- #


def to_bits(x, dtype: str) -> np.ndarray:
    """16-bit storage of ``x`` in ``dtype`` (``"fp16"`` or ``"bf16"``), as uint16."""
    if dtype == "bf16":
        return f32_to_bf16(np.asarray(x, np.float32))
    if dtype == "fp16":
        return np.asarray(x, np.float16).view(np.uint16)
    raise ValueError(f"unknown dtype {dtype!r}")


def from_bits(bits: np.ndarray, dtype: str) -> np.ndarray:
    """float32 values of 16-bit storage ``bits``."""
    if dtype == "bf16":
        return bf16_to_f32(bits)
    if dtype == "fp16":
        return bits.view(np.float16).astype(np.float32)
    raise ValueError(f"unknown dtype {dtype!r}")


def round_to(x, dtype: str) -> np.ndarray:
    """``x`` rounded to ``dtype``, as float32: what the kernel reads."""
    return from_bits(to_bits(x, dtype), dtype)


# --------------------------------------------------------------------------- #
# Device buffers (the torch-free GPU lanes)
# --------------------------------------------------------------------------- #


def device(array):
    """A device copy of ``array`` (a ``DeviceMem``; the caller keeps it alive)."""
    from rocke.runtime.host_buffers import as_u8_buffer
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.launcher import DeviceMem

    array = np.ascontiguousarray(array)
    mem = DeviceMem(array.nbytes)
    Runtime().memcpy_h2d(mem.ptr(), as_u8_buffer(array), array.nbytes)
    return mem


def nan_filled(nbytes: int):
    """``nbytes`` of device memory, every byte 0xFF: NaN as f16, bf16 and f32, so
    an element the kernel skips fails the comparison."""
    return device(np.full(nbytes, 0xFF, np.uint8))


# --------------------------------------------------------------------------- #
# Inputs and oracle
# --------------------------------------------------------------------------- #


def make_inputs(dtype, d, hq, hkv, batch, sq, skv, bias_shape, *, seed=0):
    """Seeded ``(q, k, v, bias)``: q/k/v unit-normal rounded to ``dtype`` (float32
    arrays), the fp32 bias ``2 * N(0, 1)`` of ``bias_shape`` (large enough that
    leaving it out moves the output well past :func:`tolerance`)."""
    rng = np.random.default_rng(seed)
    q = round_to(rng.standard_normal((batch, sq, hq, d)), dtype)
    k = round_to(rng.standard_normal((batch, skv, hkv, d)), dtype)
    v = round_to(rng.standard_normal((batch, skv, hkv, d)), dtype)
    bias = (2 * rng.standard_normal(bias_shape)).astype(np.float32)
    return q, k, v, bias


def bias_for_reference(bias):
    """``bias`` as the oracle reads it: a NaN entry masks its key like ``-inf`` does.
    The oracle has no NaN rows, so a test with a NaN bias checks the kernel's NaN rows
    apart (see :func:`window_row_classes`) and compares every other row with this."""
    if bias is None:
        return None
    bias = np.asarray(bias, np.float64)
    return np.where(np.isnan(bias), -np.inf, bias)


def reference(q, k, v, scale, bias, mask, sinks=None):
    """``(O [B, Sq, Hq, D], LSE [B, Hq, Sq])`` in float64 for dense inputs.

    ``softmax(q k^T * scale + bias)`` under ``mask``, through ``sdpa_reference`` (the
    hipDNN reference's semantics: ``-inf`` masks, a row with no live key gives
    ``O = 0`` and ``LSE = -inf``). NaN bias entries read as ``-inf`` (the kernels'
    rows with one are NaN: callers compare those apart).

    ``sinks`` ([Hq], optional) are per-head logits that join the softmax denominator
    and have no value vector. The oracle has no such input, so they are folded in
    from its sink-free ``(O, LSE)``: with ``L = LSE`` over the keys, adding one logit
    ``g`` makes ``LSE' = logaddexp(L, g)`` and scales ``O`` by ``exp(L - LSE')``. A row
    with no live key has ``L = -inf``: ``O' = 0`` and ``LSE' = g``, the sink logit.
    """

    def bhsd(x):
        return np.asarray(x).transpose(0, 2, 1, 3)

    o, lse = sdpa_reference(
        bhsd(q),
        bhsd(k),
        bhsd(v),
        scale=scale,
        bias=bias_for_reference(bias),
        **band_kwargs(mask),
    )
    if sinks is not None:
        sink = np.asarray(sinks, np.float64).reshape(1, -1, 1)
        total = np.logaddexp(lse, sink)
        with np.errstate(invalid="ignore"):
            weight = np.where(np.isneginf(lse), 0.0, np.exp(lse - total))
        o = o * weight[..., None]
        lse = total
    return o.transpose(0, 2, 1, 3), lse


def max_abs(out, ref) -> float:
    """Largest absolute difference, in float64."""
    diff = np.asarray(out, np.float64) - np.asarray(ref, np.float64)
    return float(np.abs(diff).max())


def bias_strides(bias: np.ndarray) -> tuple[int, int, int]:
    """``(batch, head, sq)`` element strides of a C-contiguous rank-4 fp32 array,
    0 on each size-1 (broadcast) axis: what the kernels take as bias strides."""
    return tuple(
        0 if bias.shape[a] == 1 else bias.strides[a] // bias.itemsize for a in range(3)
    )


# --------------------------------------------------------------------------- #
# The row classes of a sliding window x bias launch
# --------------------------------------------------------------------------- #

# Four heads, 1024 queries over 256 keys under a window of WINDOW: query rows from
# ROW_CLASS_DEAD_FROM on see no key at all (the window starts past the keys), the others
# see up to WINDOW of them. Each head holds one bias class, so one launch tells the
# classes' answers apart:
#   head 0  ordinary scores (the control);
#   head 1  finfo(float32).min everywhere: a live row is the uniform softmax (O = mean
#           of V over its visible keys, LSE about -2**97), a keyless row is zeros;
#   head 2  -inf everywhere: every row is zeros (LSE -inf, or the sink with sinks);
#   head 3  ordinary scores, and a NaN on key 7 of three rows: row 10 sees key 7 (its
#           O and LSE are NaN), row 300 does not (the window removes it) and neither
#           does keyless row ROW_CLASS_DEAD_FROM + 20 (both stay as the others).
ROW_CLASS_SQ, ROW_CLASS_SKV, ROW_CLASS_HQ = 1024, 256, 4
ROW_CLASS_DEAD_FROM = ROW_CLASS_SKV + WINDOW - 1
_NAN_ROWS = (10, 300, ROW_CLASS_DEAD_FROM + 20)


def window_row_classes_inputs(dtype: str, *, d: int = 128, seed: int = 5):
    """``(q, k, v, bias)`` of the row-class launch (see above): batch 1, MHA, a full
    ``[1, Hq, Sq, Skv]`` bias. V has mean 1, so a row of zeros (the wrong answer for a
    finfo.min row) fails by far more than the tolerance."""
    hq, sq, skv = ROW_CLASS_HQ, ROW_CLASS_SQ, ROW_CLASS_SKV
    q, k, v, bias = make_inputs(
        dtype, d, hq, hq, 1, sq, skv, (1, hq, sq, skv), seed=seed
    )
    v = round_to(v + 1.0, dtype)
    bias[:, 1] = np.finfo(np.float32).min
    bias[:, 2] = -np.inf
    for row in _NAN_ROWS:
        bias[0, 3, row, 7] = np.nan
    return q, k, v, bias


def check_window_row_classes(out, lse, ref, ref_lse, *, sinks, dtype):
    """Assert the kernel's ``out`` ([B, Sq, Hq, D]) and ``lse`` ([B, Hq, Sq] or None)
    answer every class of :func:`window_row_classes_inputs`; ``ref``/``ref_lse`` are
    :func:`reference` of the same inputs (``sinks``: the logits it was given, or None).

    * keyless rows (``>= ROW_CLASS_DEAD_FROM``): exact zeros in every head, whatever the
      bias, and LSE -inf (the sink logit with sinks);
    * the -inf head: zeros everywhere, the same LSE;
    * the finfo.min head, live rows: the reference's uniform softmax (not zeros, unless
      sinks outweigh it), LSE finite and below -1e28 without sinks;
    * the NaN head: row 10 is NaN in O and LSE, no other row is (a NaN on a key the
      window removes is ignored, in a live row and in a keyless one);
    * everything else matches the reference."""
    tol = tolerance(dtype)
    dead = ROW_CLASS_DEAD_FROM
    batch, sq, hq, _ = out.shape
    expect_nan = np.zeros((batch, sq, hq), bool)
    expect_nan[0, _NAN_ROWS[0], 3] = True
    nan = np.isnan(out)
    np.testing.assert_array_equal(nan.all(-1), expect_nan)
    assert not (nan.any(-1) & ~expect_nan).any()
    assert np.abs(out[:, dead:]).max() == 0.0
    assert np.abs(out[:, :, 2]).max() == 0.0
    assert np.abs(np.where(expect_nan[..., None], 0.0, out - ref)).max() < tol
    if sinks is None:
        # The finfo.min head's live rows are not the zeros a masked row would give.
        assert np.abs(ref[:, :dead, 1]).mean() > 0.5
        assert np.abs(out[:, :dead, 1]).mean() > 0.5
    if lse is None:
        return
    got, want = lse.transpose(0, 2, 1), ref_lse.transpose(0, 2, 1)  # [B, Sq, Hq]
    np.testing.assert_array_equal(np.isnan(got), expect_nan)
    skip = expect_nan.copy()
    if sinks is None:
        skip[:, :dead, 1] = True
        assert np.isfinite(got[:, :dead, 1]).all()
        assert (got[:, :dead, 1] < -1e28).all()
    np.testing.assert_array_equal(np.isneginf(got)[~skip], np.isneginf(want)[~skip])
    finite = ~skip & np.isfinite(want)
    with np.errstate(invalid="ignore"):  # -inf - -inf, in the rows `finite` leaves out
        diff = np.abs(got - want)
    assert diff[finite].max() < tol


# --------------------------------------------------------------------------- #
# The bias constants of a dense kernel
# --------------------------------------------------------------------------- #

# The floor and the live-row threshold live once, in the shared bias helper; each
# kernel keeps its own mask sentinel, aliases the shared values and checks their order
# against that sentinel when it is imported. A kernel's test file calls
# ``assert_kernel_bias_constants`` with its module; the shared check itself is tested
# once, in ``test_attention_dense_bias_gfx950.py``.


def _bias_constants_block(module, class_name):
    """``module``'s mask-sentinel / bias-constant block as source text: from its
    ``_MASK_SENTINEL`` to the spec class ``class_name`` that follows it."""
    src = Path(module.__file__).read_text(encoding="utf-8")
    start = src.index("_MASK_SENTINEL = ")
    marker = f"@dataclass(frozen=True)\nclass {class_name}"
    end = src.find(marker, start)
    assert end != -1, f"{module.__name__}: {marker!r} must follow _MASK_SENTINEL"
    return src[start:end]


def _exec_bias_constants_block(block, log2e):
    """Run ``block`` as the module runs it at import, with ``LOG2E`` and the shared bias
    constants and check in scope; returns its namespace, and raises what the block's own
    ``check_bias_constants`` call raises."""
    import kernels.common._attention_additive_bias as shared

    ns = {
        "LOG2E": log2e,
        "BIAS_FLOOR": shared.BIAS_FLOOR,
        "BIAS_LIVE_ROW_MIN": shared.BIAS_LIVE_ROW_MIN,
        "check_bias_constants": shared.check_bias_constants,
    }
    exec(compile(block, "<constants>", "exec"), ns)
    return ns


def assert_kernel_bias_constants(module, class_name):
    """Assert that the dense kernel module ``module`` keeps the order
    ``_MASK_SENTINEL < threshold < floor * log2e``: a masked score cannot read as live,
    and a floored bias (finfo.min padding) cannot read as dead. Fails when its
    ``_BIAS_FLOOR`` / ``_BIAS_LIVE_ROW_MIN`` are not the shared values, when its
    constants block (the source from ``_MASK_SENTINEL`` to the spec class
    ``class_name``) does not reproduce the module's values or stops calling
    ``check_bias_constants`` with its own sentinel, or when that block does not raise
    once its sentinel is moved above the threshold. The module raises, not asserts:
    ``python -O`` would strip an assert."""
    import re

    import pytest

    import kernels.common._attention_additive_bias as shared

    # This module is not assert-rewritten (it is not a test file), so each assert
    # carries its values.
    got = (module._BIAS_FLOOR, module._BIAS_LIVE_ROW_MIN)
    want = (shared.BIAS_FLOOR, shared.BIAS_LIVE_ROW_MIN)
    assert got == want, f"{module.__name__} aliases {got}, the shared values are {want}"
    sentinel, floor_score = module._MASK_SENTINEL, module._BIAS_FLOOR * module.LOG2E
    assert (
        sentinel < module._BIAS_LIVE_ROW_MIN < floor_score
    ), f"{module.__name__}: {sentinel} < {module._BIAS_LIVE_ROW_MIN} < {floor_score}"
    block = _bias_constants_block(module, class_name)
    assert (
        "check_bias_constants(_MASK_SENTINEL, LOG2E)" in block
    ), "the block does not call check_bias_constants(_MASK_SENTINEL, LOG2E)"
    ns = _exec_bias_constants_block(block, module.LOG2E)
    block_values = (ns["_MASK_SENTINEL"], ns["_BIAS_FLOOR"], ns["_BIAS_LIVE_ROW_MIN"])
    module_values = (sentinel, module._BIAS_FLOOR, module._BIAS_LIVE_ROW_MIN)
    assert block_values == module_values, (
        f"{module.__name__}: the constants block gives {block_values}, "
        f"the module has {module_values}"
    )
    # The same block with the sentinel moved above the threshold: the call must raise.
    line = re.compile(r"^_MASK_SENTINEL = .*$", re.M)
    assert line.search(block), "the block has no `_MASK_SENTINEL = ...` line"
    broken = line.sub(f"_MASK_SENTINEL = {module._BIAS_LIVE_ROW_MIN / 2!r}", block)
    with pytest.raises(ValueError, match="bias constants must satisfy"):
        _exec_bias_constants_block(broken, module.LOG2E)
