# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Softmax-gradient VALU levers of the attention backward main kernel.

Family glue (Python-only, like the rest of the backward body). Each lever is a
defaulted :class:`~kernels.common.attention_bwd.AttnBwdSpec` field whose
default emits the shipped code; this module holds their emitters so the body
keeps one small hook per lever.

``bf16_cast`` (P / dS rounded to bf16 for G1 / G3 / G4)
    ``"generic"``: ``fptrunc``. On targets without a bf16 conversion
    instruction (gfx942) the backend expands it per element to round to
    nearest even plus a NaN guard (``v_bfe_u32``, ``v_add3_u32``, ``v_or_b32``,
    ``v_cmp_u_f32``, ``v_cndmask_b32``), then packs pairs with ``v_perm_b32``.
    ``"rne_noguard"``: the same rounding written as integer IR without the
    guard: ``u + 0x7fff + ((u >> 16) & 1)`` per element (``v_bfe_u32`` +
    ``v_add3_u32``) and one ``v_perm_b32`` (selector ``0x07060302``) per pair.
    Bit for bit equal to ``"generic"`` for every non-NaN input, +-Inf and
    subnormals included. A NaN input is not guaranteed to stay NaN: the
    carry of the rounding add can turn its payload into ``-0`` (for example
    ``0x7fffffff``), ``+Inf`` (``0x7f800001``) or ``-Inf`` (``0xff800001``);
    the result of a NaN input is unspecified under this value. The adds are
    vector ``add`` (no ``nsw``: the wrap of a NaN pattern is defined).
    Legal for ``dtype = "bf16"`` only.

``bwd_exp2`` (P = exp2(S * scale_log2 - lse2))
    ``"exact"``: ``llvm.exp2.f32``, whose expansion keeps subnormal results
    (for an argument below -126 it adds 64 before ``v_exp_f32`` and scales the
    result by ``2^-64``: five extra VALU per element). ``"fast"``:
    ``llvm.amdgcn.exp2.f32``, a bare ``v_exp_f32`` (what the forward uses).
    Both execute the same ``v_exp_f32`` for every argument at or above -126,
    so P is bit for bit the same there; below -126 (P < 2^-126) the fast form
    flushes P to 0. ``-inf`` (dead rows: ``lse2 = +inf``) gives 0 and NaN
    gives NaN either way. The argument is <= 0 up to the rounding of the
    recomputed S against the forward's LSE, so the overflow side never
    matters.

``stats_fma`` (the P argument)
    False: ``fsub(fmul(S, scale_log2), lse2)`` (two roundings; the backend
    does not contract them). True: ``fma(S, scale_log2, -lse2)`` (one
    rounding, one VALU fewer per element). Within the backward tolerance, not
    bit for bit. A dead row (``lse2 = +inf``) gives ``-inf`` either way.

``stats_vec`` (lse2 / Dsum loads)
    1: one scalar ``global_load_dword`` of each array per C row of the lane,
    each at a clamped row (the shipped loads). 2 / 4: one ``<n x f32>`` load of
    each array per run of ``n`` consecutive C rows of the lane (the MFMA C map
    gives each lane runs of 4 consecutive rows). The window of a run is moved
    inside the arrays when the run would leave them, and each row takes its
    value by a select on the shift, so every row reads the same element as the
    scalar load and the values are bit for bit those of ``stats_vec = 1``.
    Head-major workspace only (token-major rows are ``H`` floats apart).
    A window lies inside the whole ``[h][rows]`` array (it may start in the
    previous head or end in the next one); only when the array holds fewer
    than ``n`` floats in total does it read up to ``n - 1`` floats past its
    end, which stay inside the plan's workspace sub-buffer (every sub-buffer
    is rounded up to 256 bytes).
"""

from __future__ import annotations

from collections.abc import Sequence

from rocke.core.arch.target import MmaOp
from rocke.core.ir import BF16, F32, I32, IRBuilder, Value, VectorType
from rocke.helpers.attention_band import I32_FAR

from ._attention_bwd_addr import (
    _clamp64_to_i32,
    _max64,
    _min64,
    WorkspaceAddr,
    as_i64,
)
from ._attention_bwd_frag import eval_layout

__all__ = [
    "BF16_CASTS",
    "BWD_EXP2S",
    "STATS_VECS",
    "NATIVE_BF16_CVT_ARCHS",
    "batched_stats_rows",
    "cast_c_frag",
    "check_stats_vec_layout",
    "rne_noguard_bf16",
    "softmax_p",
    "stats_rows_neg",
]

# P / dS rounding to bf16: the generic fptrunc (default) or round to nearest
# even without the NaN guard.
BF16_CASTS = ("generic", "rne_noguard")
# P = exp2(arg): the subnormal-keeping expansion (default) or the bare v_exp_f32.
BWD_EXP2S = ("exact", "fast")
# Width of the lse2 / Dsum loads, in consecutive C rows of a lane.
STATS_VECS = (1, 2, 4)
# Targets whose generic f32 -> bf16 cast is one packed conversion instruction
# (v_cvt_pk_bf16_f32): ``rne_noguard`` can only add VALU there, so the sweep
# does not enumerate it (it stays legal and correct).
NATIVE_BF16_CVT_ARCHS = ("gfx950",)

# v_perm_b32 selector taking bytes 2, 3 of the low source and bytes 2, 3 of
# the high source: the upper halves of two words, low element first.
_HI_HALVES = 0x07060302


def rne_noguard_bf16(b: IRBuilder, frag: Value) -> Value:
    """``<n x f32>`` -> ``<n x bf16>`` (``n`` even), round to nearest even, no
    NaN guard (see the module docstring for the NaN results)."""
    n = frag.type.count
    if n % 2:
        raise ValueError("rne_noguard_bf16 packs element pairs: n must be even")
    vi = VectorType(I32, n)
    u = b.vec_bitcast(frag, vi)

    def splat(v: int) -> Value:
        return b.vector_splat(b.const_i32(v), n)

    lsb = b.vector_and(b.vector_lshr(u, splat(16)), splat(1))
    r = b.vector_add(b.vector_add(u, lsb), splat(0x7FFF))
    words = [
        b.byte_perm(b.vec_extract(r, 2 * j + 1), b.vec_extract(r, 2 * j), _HI_HALVES)
        for j in range(n // 2)
    ]
    return b.vec_bitcast(b.vec_pack(words, I32), VectorType(BF16, n))


def cast_c_frag(b: IRBuilder, frag: Value, elem, bf16_cast: str = "generic") -> Value:
    """Round an fp32 C fragment to the I/O dtype ``elem`` under ``bf16_cast``."""
    if bf16_cast == "generic":
        return b.vec_cast_f32_to(frag, elem)
    if bf16_cast != "rne_noguard":
        raise ValueError(f"bf16_cast={bf16_cast!r} not in {BF16_CASTS}")
    if elem != BF16:
        raise ValueError("bf16_cast='rne_noguard' applies to bf16 operands only")
    return rne_noguard_bf16(b, frag)


def softmax_p(
    b: IRBuilder,
    sv: Value,
    scale_log2: Value,
    lse2: Value,
    *,
    neg_lse2: Value | None = None,
    bwd_exp2: str = "exact",
) -> Value:
    """``P = exp2(S * scale_log2 - lse2)`` of one element.

    ``neg_lse2`` (``-lse2``, formed once per row by the caller) selects the
    fused form ``fma(S, scale_log2, -lse2)`` (``stats_fma``); ``None`` keeps
    ``fsub(fmul(S, scale_log2), lse2)``.
    """
    if neg_lse2 is None:
        arg = b.fsub(b.fmul(sv, scale_log2), lse2)
    else:
        arg = b.fma(sv, scale_log2, neg_lse2)
    if bwd_exp2 == "exact":
        return b.exp2(arg)
    if bwd_exp2 == "fast":
        return b.exp2_fast(arg)
    raise ValueError(f"bwd_exp2={bwd_exp2!r} not in {BWD_EXP2S}")


def check_stats_vec_layout(op: MmaOp, width: int) -> None:
    """Raise unless every lane's C slots form runs of ``width`` consecutive rows.

    Run ``g`` is slots ``g * width .. g * width + width - 1``: their rows must
    be ``row(g * width) + j`` and their column the same, on every lane.
    """
    if width not in STATS_VECS:
        raise ValueError(f"stats_vec={width!r} not in {STATS_VECS}")
    lanes = eval_layout(op.c_layout())
    n = len(lanes[0])
    if n % width:
        raise ValueError(
            f"stats_vec={width}: the C fragment of {op.op_id} has {n} slots"
        )
    for coords in lanes:
        for g in range(0, n, width):
            r0, c0 = coords[g]
            if any(coords[g + j] != (r0 + j, c0) for j in range(width)):
                raise ValueError(
                    f"stats_vec={width}: the C rows of {op.op_id} are not runs of "
                    f"{width} consecutive rows (a lane holds {coords})"
                )


def batched_stats_rows(
    b: IRBuilder,
    lse2: WorkspaceAddr,
    dsum: WorkspaceAddr,
    h: Value,
    q0: Value,
    rows: Sequence[Value],
    *,
    width: int,
    n_heads: Value,
) -> list[tuple[Value, Value]]:
    """``(lse2, Dsum)`` of q-step rows ``rows``, one vector load per run.

    ``rows`` are the lane's C rows (offsets from ``q0``) in key order; every
    run of ``width`` of them must be consecutive (:func:`check_stats_vec_layout`
    proves it for the atom). Returns the values of
    :meth:`._attention_bwd_body.StatsView.rows` bit for bit: an in-range row
    reads its own element, a row at or past ``ws_rows`` the neutral ``+inf`` /
    ``0``.

    Address model (head-major, one fp32 per row): the scalar path reads row
    ``r`` at element ``elem + clamp(r, 0, n - 1)`` of the array, with ``elem =
    h * ws_rows + clamp(q_base, 0, ws_rows - 1)`` and ``n`` the rows left in
    the head. Here a run ``r0 .. r0 + width - 1`` loads the window ``[elem +
    A, elem + A + width)`` with ``A = clamp(r0, lo, hi - width + 1)``, where
    ``lo = -min(elem, width - 1)`` and ``hi = H * ws_rows - 1 - elem`` bound
    the window inside the array. ``A = r0`` whenever the run fits (shift
    ``d = r0 - A = 0``); otherwise the in-range rows of the run (``r0 + i <=
    n - 1 <= hi``) sit at window slot ``i + d <= width - 1``, picked by a
    select on ``d``. Rows past ``n`` are replaced by the neutral value.
    """
    for wa in (lse2, dsum):
        if wa.layout != "head_major" or wa.width != 1:
            raise ValueError("stats_vec > 1 needs head-major one-float stats rows")
    if lse2.ws_rows is not dsum.ws_rows or lse2.row0 is not dsum.row0:
        raise ValueError("lse2 / Dsum workspaces must share their row model")
    if len(rows) % width:
        raise ValueError(f"{len(rows)} rows do not split into runs of {width}")
    # Uniform part (once per step): the slab base of the scalar path, the
    # rows left in the head and the window bounds.
    base = lse2.row0
    if not (isinstance(q0, int) and q0 == 0):
        base = b.add(base, as_i64(b, q0, signed=True))
    rows64 = as_i64(b, lse2.ws_rows)
    one64 = b.const_i64(1)
    base_ws = _max64(b, _min64(b, base, b.sub(rows64, one64)), b.const_i64(0))
    rows_left = _clamp64_to_i32(b, b.sub(rows64, base), b.const_i32(I32_FAR))
    elem = b.add(b.mul(as_i64(b, h), rows64), base_ws)
    total = b.mul(as_i64(b, n_heads), rows64)
    back = _min64(b, elem, b.const_i64(width - 1))  # in [0, width - 1]
    lo = b.sub(b.const_i32(0), b.trunc(back, I32))
    hi = _clamp64_to_i32(b, b.sub(b.sub(total, one64), elem), b.const_i32(I32_FAR))
    top = b.sub(hi, b.const_i32(width - 1))
    # pointers at the lowest window start, so every in-slab index is >= 0
    off = b.mul(b.sub(elem, back), b.const_i64(4))
    p_l = b.global_ptr_add(lse2.ptr, off)
    p_d = b.global_ptr_add(dsum.ptr, off)
    inf, zero = b.const_f32(float("inf")), b.const_f32(0.0)
    out: list[tuple[Value, Value]] = []
    for g in range(0, len(rows), width):
        r0 = rows[g]
        a = b.smax(b.smin(r0, top), lo)
        idx = b.sub(a, lo)
        vl = b.global_load_vN(p_l, idx, F32, width, align=4)
        vd = b.global_load_vN(p_d, idx, F32, width, align=4)
        d = b.sub(r0, a)
        shift = [None] + [b.cmp_eq(d, b.const_i32(k)) for k in range(1, width)]
        for i in range(width):
            r = rows[g + i]
            ok = b.cmp_lt(r, rows_left)
            pair = []
            for v, neutral in ((vl, inf), (vd, zero)):
                val = b.vec_extract(v, i)
                for k in range(1, width - i):
                    val = b.select(shift[k], b.vec_extract(v, i + k), val)
                pair.append(b.select(ok, val, neutral))
            out.append((pair[0], pair[1]))
    return out


def stats_rows_neg(b: IRBuilder, stats: dict) -> dict:
    """``{key: -lse2}`` of a stats dict ``{key: (lse2, Dsum)}`` (``stats_fma``)."""
    return {key: b.fneg(lse2) for key, (lse2, _dsum) in stats.items()}
