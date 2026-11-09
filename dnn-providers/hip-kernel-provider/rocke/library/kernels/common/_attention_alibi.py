# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""ALiBi score bias for the attention forward kernels.

The term follows hipDNN (``SdpaSemanticsRegister`` SEM-021): the scaled
score of query ``i`` and key ``j`` gets ``slope(h) * (j - i)``, added before the
mask. Slopes come from a caller tensor (one fp32 value per query head, the vLLM
convention) when the kernel's ``alibi_slopes_ptr`` argument is not null, or are
built in, derived from the query head count as hipDNN does (Press et al.)::

    n = 2 ** floor(log2(H))
    slope(h) = 2 ** (-8 * (h + 1) / n)          for h < n
    slope(h) = 2 ** (-8 * (h - n + 0.5) / n)    for h >= n

Kernels evaluate ``slope * (j - a_i)`` against a per-row anchor ``a_i`` instead
of ``j - i``: the query's diagonal under a causal mask, the last key otherwise.
No visible key lies past the anchor, so the term is never positive and the
scores the row max sees keep their fp32 precision however far apart ``i`` and
``j`` are. The causal anchor is not clamped to the last key: for rows past it
(``Sq > Skv``, top-left) the nearest visible key's term grows with the row, so
precision drops slowly there. The difference ``slope * (a_i - i)`` is constant per
row, so softmax and O are unchanged; only the log-sum-exp moves, and
:func:`alibi_lse_shift` gives the correction to add to the log2-domain row max
before the LSE store. A sink logit is not shifted, so kernels refuse sinks with
ALiBi.

hipDNN defines the term on the top-left diagonal only. Off it (bottom-right
alignment with ``Sq != Skv``) the kernels anchor at the shifted diagonal, which
fixes O but leaves the LSE undefined, so they refuse LSE output there.

One kernel serves both slope sources: the pointer is a kernel argument, so the
choice is uniform and costs a few prologue instructions, and hipDNN (which has
no slopes tensor) binds a null pointer to the same binary.

The dense kernels have no hand-written C++ twin (the C++ engine lowers their IR
generically); the unified tiled kernels do, so :func:`alibi_slope` with a runtime
head count is mirrored in ``cpp/instances/common/_attention_alibi.cpp``.
"""

from __future__ import annotations

from rocke.core.ir import F32, I32, IRBuilder, Value

__all__ = [
    "alibi_lse_shift",
    "alibi_slope",
    "alibi_slope_log2",
    "alibi_tile_bias",
    "check_alibi_slopes",
]

_LOG2E = 1.4426950408889634


def check_alibi_slopes(spec, alibi_slopes) -> None:
    """Host check of the ``alibi_slopes`` argument of a runner: ``None`` (the
    built-in slopes) or an fp32 contiguous GPU tensor ``[num_query_heads]``, and
    only when ``spec`` (a dense spec or a unified problem) has ALiBi on."""
    if alibi_slopes is None:
        return
    if not spec.use_alibi:
        raise ValueError("alibi_slopes provided but use_alibi is off")
    if tuple(alibi_slopes.shape) != (spec.num_query_heads,):
        raise ValueError(
            f"alibi_slopes must have shape ({spec.num_query_heads},), "
            f"got {tuple(alibi_slopes.shape)}"
        )
    if str(alibi_slopes.dtype) != "torch.float32":
        raise ValueError(f"alibi_slopes must be float32, got {alibi_slopes.dtype}")
    if not alibi_slopes.is_contiguous():
        raise ValueError("alibi_slopes must be contiguous")
    if not alibi_slopes.is_cuda:
        raise ValueError("alibi_slopes must be a CUDA tensor")


def _alibi_builtin_slope_exponent(b: IRBuilder, head: Value, num_heads) -> Value:
    """``log2(slope(head))`` as f32, from a baked or runtime head count."""
    if isinstance(num_heads, int):
        n = 1 << (num_heads.bit_length() - 1)
        step = -8.0 / n  # a power of two, exact in f32
        n_v = b.const_i32(n)
        step_v = b.const_f32(step)
        tail_off = b.const_f32(step * (0.5 - n))
    else:
        # floor(log2(H)) is the exponent field of float(H) (exact below 2**24);
        # -8 / n = -(2 ** (3 - e)) is rebuilt from the same field.
        e = b.sub(
            b.lshr(b.bitcast(b.sitofp_f32(num_heads), I32), b.const_i32(23)),
            b.const_i32(127),
        )
        n_v = b.shl(b.const_i32(1), e)
        inv8 = b.bitcast(b.shl(b.sub(b.const_i32(127 + 3), e), b.const_i32(23)), F32)
        step_v = b.fneg(inv8)
        # step * (0.5 - n) = step / 2 + 8: n * step = -8 exactly.
        tail_off = b.fma(step_v, b.const_f32(0.5), b.const_f32(8.0))
    # h < n: step * (h + 1); otherwise step * (h - n + 0.5).
    off = b.select(b.cmp_lt(head, n_v), step_v, tail_off)
    return b.fma(b.sitofp_f32(head), step_v, off)


def alibi_slope(b: IRBuilder, head: Value, *, num_heads, slopes: Value) -> Value:
    """``slope(head)`` as f32 for ``head`` among ``num_heads`` query heads (a
    Python int when baked, an i32 value when a kernel param).

    ``slopes`` is the ``alibi_slopes_ptr`` kernel argument: not null, the slope
    is ``slopes[head]``; null, the built-in slope. The pointer is a kernel
    argument, so the branch is uniform. The read is a buffer load bounded by
    ``num_heads``: a padded query row whose head index is out of range reads 0
    instead of faulting, and its built-in slope is finite, so padded rows stay
    deterministic either way."""
    has_slopes = b.cmp_ne(b.global_ptr_to_i64(slopes), b.const_i64(0))
    branch = b.scf_if_else(has_slopes, result_types=[F32])
    with branch as (then_ctx, else_ctx):
        with then_ctx:
            if isinstance(num_heads, int):
                num_bytes = b.const_i32(4 * num_heads)
            else:
                num_bytes = b.mul(num_heads, b.const_i32(4))
            rsrc = b.buffer_rsrc(slopes, num_bytes)
            offset = b.mul(head, b.const_i32(4))
            b.scf_yield(b.buffer_load(rsrc, offset, b.const_i32(0), F32))
        with else_ctx:
            b.scf_yield(b.exp2(_alibi_builtin_slope_exponent(b, head, num_heads)))
    (slope,) = branch.results
    return slope


def alibi_slope_log2(b: IRBuilder, head: Value, *, num_heads, slopes: Value) -> Value:
    """``slope(head) * log2(e)`` as f32, the slope in log2-domain score units
    (see :func:`alibi_slope`)."""
    slope = alibi_slope(b, head, num_heads=num_heads, slopes=slopes)
    return b.fmul(slope, b.const_f32(_LOG2E))


def alibi_tile_bias(
    b: IRBuilder,
    slope: Value,
    tile_key0: Value,
    lane_key_off: Value,
    anchor: Value,
    n_sub: int,
) -> list:
    """Bias ``slope * (key - anchor)`` for one KV tile in the transposed 32x32
    MFMA C layout (key on the accumulator element, query on ``lane % 32``).

    The key of element ``i`` of 32-key block ``nsub`` is ``tile_key0 +
    lane_key_off + nsub * 32 + (i // 4) * 8 + i % 4``, with ``lane_key_off =
    (lane // 32) * 4``. The per-lane part is folded once per tile, so each
    element costs one FMA against a compile-time offset. Returns ``[n_sub][16]``
    f32 values in the units of ``slope``."""
    base = b.fmul(slope, b.sitofp_f32(b.sub(b.add(tile_key0, lane_key_off), anchor)))
    out = []
    for nsub in range(n_sub):
        row = []
        for i in range(16):
            c = nsub * 32 + (i // 4) * 8 + i % 4
            row.append(b.fma(slope, b.const_f32(float(c)), base) if c else base)
        out.append(row)
    return out


def alibi_lse_shift(b: IRBuilder, slope_log2: Value, anchor: Value, query: Value):
    """``slope * (anchor - query)`` in log2 units: add it to the row max before
    the LSE store to undo the anchor (see the module docstring)."""
    return b.fmul(slope_log2, b.sitofp_f32(b.sub(anchor, query)))
