# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The attention band's right bound, shared by the gfx942 / gfx950 attention
specs and bodies.

Key ``k`` is visible to the query at diagonal position ``d`` iff
``k <= d + right_bound`` (and ``d - k < sliding_window`` when a window is set).
``right_bound`` 0 is the causal mask, ``R > 0`` adds ``R`` keys of lookahead and
``-1`` (tiled specs only) leaves the right side unbounded. ``causal_top_left``
anchors the diagonal at the first key (``context_len = 0``) instead of the last.

Consumers:

* the tiled 2D / 3D specs: ``check_right_bound`` (``__post_init__``) and
  ``tiled_band_name_parts`` (kernel name); the unified problem and scalar kernel
  name use ``check_right_bound`` / ``right_bound_tag``;
* the tiled 2D / 3D bodies (the 4-warp bodies included): ``diagonal_offset`` and
  :class:`BandMask`;
* the gfx942 / gfx950 dense specs: ``check_dense_right_bound`` and
  ``dense_band_name_parts``; the dense bodies: ``upper_lim``.

Every helper emits nothing for the causal band (``right_bound == 0``), so a
default spec builds exactly the IR it built before the band existed. The IR
helpers create their ops at the call site, in the order the inlined code did.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rocke.core.ir import IRBuilder, Value


def check_right_bound(
    right_bound: int, *, use_alibi: bool = False, use_qq_bias: bool = False
) -> None:
    """Raise ``ValueError`` for a tiled ``right_bound`` out of range, or one combined
    with ALiBi / qq_bias (whose offsets are defined on the causal diagonal)."""
    if right_bound < -1:
        raise ValueError(
            f"right_bound must be >= -1 (-1 = unbounded), got {right_bound}"
        )
    if right_bound != 0 and (use_alibi or use_qq_bias):
        raise ValueError(
            "right_bound != 0 is incompatible with use_alibi / use_qq_bias "
            "(their offsets are defined on the causal diagonal)"
        )


def right_bound_tag(right_bound: int) -> str:
    """Kernel-name token: ``""`` (causal), ``rbu`` (unbounded) or ``rb<R>``."""
    if right_bound == 0:
        return ""
    return "rbu" if right_bound < 0 else f"rb{right_bound}"


def tiled_band_name_parts(causal_top_left: bool, right_bound: int) -> tuple[str, str]:
    """The tiled kernel-name parts ``(tl, rb)``; empty strings when unset."""
    return ("tl" if causal_top_left else "", right_bound_tag(right_bound))


def dense_band_name_parts(right_bound: int) -> tuple[str, ...]:
    """The dense kernel-name part: ``rb<R>`` for a lookahead, none otherwise."""
    return (right_bound_tag(right_bound),) if right_bound > 0 else ()


def check_dense_right_bound(spec) -> None:
    """Raise ``ValueError`` for a dense spec's ``right_bound`` the dense bodies do
    not compute. Dense ``right_bound`` is never negative: ``causal=False`` is the
    unbounded right side."""
    if spec.right_bound < 0:
        raise ValueError(
            f"right_bound must be >= 0 (causal=False is the unbounded side), "
            f"got {spec.right_bound}"
        )
    if spec.right_bound > 0:
        if not spec.causal:
            raise ValueError("right_bound > 0 requires causal=True")
        if spec.causal_bottom_right:
            raise ValueError(
                "right_bound > 0 is not supported with causal_bottom_right"
            )
        if spec.paged or spec.varlen:
            raise ValueError("right_bound > 0 is not supported with paged / varlen")


def diagonal_offset(
    b: IRBuilder, causal_top_left: bool, kv_len: Value, q_len: Value
) -> Value:
    """``context_len``: query ``q`` sits on diagonal position ``context_len + q``.
    Bottom-right ``kv_len - q_len``; top-left 0."""
    if causal_top_left:
        return b.const_i32(0)
    return b.sub(kv_len, q_len)


def upper_lim(b: IRBuilder, causal_lim: Value, right_bound: int) -> Value:
    """``causal_lim + right_bound``: the last visible key. Returns ``causal_lim``
    itself (no op) unless there is a lookahead."""
    if right_bound <= 0:
        return causal_lim
    return b.add(causal_lim, b.const_i32(right_bound))


@dataclass(frozen=True)
class BandMask:
    """The per-score right-bound compares of a tiled body.

    ``upper_ok`` is ``col <= causal_lim + right_bound``, or ``None`` when the right
    side is unbounded; ``row_mask`` folds it into a score's row / prefix mask.
    Every site that derives a causal limit from ``context_len`` adds the right bound
    through ``upper_lim``. All of them emit the causal IR when ``right_bound == 0``.
    """

    b: IRBuilder
    right_bound: int

    def upper_lim(self, causal_lim: Value) -> Value:
        return upper_lim(self.b, causal_lim, self.right_bound)

    def upper_ok(self, col_abs: Value, causal_lim: Value) -> Optional[Value]:
        if self.right_bound < 0:
            return None
        return self.b.cmp_le(col_abs, self.upper_lim(causal_lim))

    def row_mask(
        self, row_ok: Value, upper_ok: Optional[Value], in_prefix: Value
    ) -> Value:
        if upper_ok is None:
            return self.b.land(row_ok, in_prefix)
        return self.b.land(self.b.land(row_ok, upper_ok), in_prefix)

    def max_prefix_len(
        self, context_len: Value, qb_start_pos: Value, last_row: int, seq_len: Value
    ) -> Value:
        """Keys a query block reaches: ``min(context_len + qb_start_pos + last_row +
        1 + right_bound, seq_len)``, or ``seq_len`` when the right side is
        unbounded. ``last_row`` is the block's last query-position offset."""
        if self.right_bound < 0:
            return seq_len
        b = self.b
        msp_raw = b.add(
            b.add(context_len, qb_start_pos),
            b.const_i32(last_row + 1 + self.right_bound),
        )
        return b.select(b.cmp_lt(msp_raw, seq_len), msp_raw, seq_len)

    def last_key(self, context_len: Value, qpos: Value, seq_len: Value) -> Value:
        """The last key visible to query position ``qpos``: ``seq_len - 1`` when
        the right side is unbounded."""
        if self.right_bound < 0:
            return self.b.sub(seq_len, self.b.const_i32(1))
        return self.upper_lim(self.b.add(context_len, qpos))

    def valid_tail(self, causal_lim: Value, prefix_tail: Value) -> Value:
        """``min(causal_lim + right_bound, prefix_tail)``: the last key a row may
        keep, or ``prefix_tail`` when the right side is unbounded."""
        if self.right_bound < 0:
            return prefix_tail
        lim = self.upper_lim(causal_lim)
        return self.b.select(self.b.cmp_lt(lim, prefix_tail), lim, prefix_tail)
