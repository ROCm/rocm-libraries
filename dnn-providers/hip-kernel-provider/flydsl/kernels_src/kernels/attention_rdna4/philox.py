# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Philox-4x{32,64}-N counter-based PRNG for FlyDSL kernels.

A counter-based PRNG is a pure function of ``(key, counter)`` with no state and
no sequence: every element derives its own random from its own coordinates,
with no ordering between them. That is what a GPU needs, and it is why the
backward pass can regenerate a dropout mask from ``(seed, offset)`` alone
rather than storing it -- a 4K x 4K mask is 268 MB per head.

**This module is deliberately arch-agnostic and layout-agnostic.** It maps
``(seed, offset) -> N randoms`` and nothing else. How a caller assigns offsets
to elements is the caller's business; there are no wave-size assumptions, no
target intrinsics, and no attention-specific code. It is written to the
standard a FlyDSL library needs because that is where it is going.

Using it
--------
Prefer the configured object over the free functions -- it carries the width
and round count together, which is what forward, backward and the mask kernel
must agree on::

    from philox import Philox

    rng = Philox.for_arch()               # measured default for this device
    vals = rng.u32(seed, offset)          # len == rng.randoms_per_offset
    keep = [v > threshold for v in vals]

``randoms_per_offset`` is 4 at width 32 and 8 at width 64, and it belongs in
the caller's offset arithmetic (``n // rng.randoms_per_offset``). The free
functions ``philox_u32`` / ``philox_4x`` remain for callers that already have
a width in hand.

Matches Triton's ``triton.language.random`` bit for bit -- see
``test_philox.py``, which checks against both a CPU reference and Triton
itself. Callers compare dropout against seeded ``torch`` runs, so the stream is
part of the contract, not an implementation detail.

Two widths
----------
``PHILOX_WIDTH`` selects 32- or 64-bit lanes. They are *different PRNGs* and
produce different streams -- as any two PRNGs do -- so the width is part of
whatever contract a caller builds on top:

    width  lanes    randoms per call   state
    32     u32      4 x u32            6 registers
    64     u64      4 x u64 = 8 x u32  12 registers

Which is faster is a property of the target's integer ALU, so it is a
parameter here and a per-arch default in the caller. On gfx1201 the 32-bit
variant needs one ``v_mul_hi_u32`` per high product while the 64-bit one needs
a multi-instruction sequence; see ``kernels/microbench/philox_bench.py``.

Both widths take a **64-bit seed and a 64-bit offset**. That does not require
64-bit arithmetic: at width 32 the seed fills the two key words and the offset
the low two counter words. Splitting a 64-bit value costs nothing -- it lives
in two consecutive 32-bit registers, so the halves are addressable directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import flydsl.expr as fx
from flydsl._mlir.dialects import arith

__all__ = [
    "PHILOX_WIDTHS",
    "DEFAULT_ROUNDS",
    "Philox",
    "PhiloxWidth",
    "Word",
    "U64",
    "default_width",
    "randoms_per_offset",
    "philox_4x",
    "philox_u32",
    "dropout_threshold",
    "keep_mask",
    "to_uniform_f32",
]

# --- types -----------------------------------------------------------------

PhiloxWidth: TypeAlias = Literal[32, 64]
"""Lane width. Selects *which PRNG*, not just how it is computed: the two
widths have different round constants and produce different streams."""

Word: TypeAlias
"""One PRNG lane, at whatever width is in play: `fx.Int32` at 32, `fx.Int64`
at 64. Counter and key words are `Word`; the *outputs* are always `fx.Int32`
however wide the lanes were, because callers consume u32."""
Word = fx.Int32 | fx.Int64

U64: TypeAlias = fx.Int64
"""A 64-bit seed or offset. Both are 64-bit at either width -- see the module
docstring; the 32-bit variant splits them across two lanes, for free."""

PHILOX_WIDTHS = (32, 64)
DEFAULT_ROUNDS = 10

# Per-arch lane width. **Changing an entry changes every stream derived from
# it**, and therefore every dropout mask -- unlike a tuning table, where a
# re-sweep is invisible. Version it and record why, rather than re-measuring
# and overwriting.
#
# gfx1201, measured by `kernels/microbench/philox_bench.py`:
#
#   width  u32/call   G randoms/s   VGPRs (one call)
#   32     4          284.5         8
#   64     8           65.7         26
#
# 32-bit is **4.3x** the throughput per random and costs 18 fewer registers.
# Both margins are wider than the instruction counts predicted (~2x): gfx1201
# has `v_mul_hi_u32` as a single instruction, while the 64-bit variant needs
# the *high* half of a 64x64 product, whose expansion is worse than the low
# half that was counted. The register gap is the six-vs-twelve state words
# plus the eight-vs-four live outputs.
#
# So the 64-bit variant's one advantage -- eight randoms per call, matching an
# eight-column accumulator group exactly -- does not come close to paying for
# itself here. It is kept because that trade will look different on a target
# with a native 64-bit multiplier.
_WIDTH_BY_ARCH = {
    "gfx1201": 32,
}
_FALLBACK_WIDTH = 32


def default_width(arch: str | None = None) -> PhiloxWidth:
    """Lane width for `arch`, defaulting to 32 where nothing was measured.

    32 is the conservative default: it is the cheaper variant on every target
    without a native 64-bit integer multiplier, which is most of them.
    """
    if arch is None:
        from flydsl.runtime.device import get_rocm_arch

        arch = get_rocm_arch()
    base = str(arch).split(":")[0]
    return _WIDTH_BY_ARCH.get(base, _FALLBACK_WIDTH)


# Weyl-sequence key increments and round multipliers. The 32-bit values are the
# original Philox-4x32 constants (golden ratio / sqrt(3) fractions); the 64-bit
# ones are their 64-bit counterparts. Both match Triton's table.
_CONSTS = {
    32: dict(KEY_A=0x9E3779B9, KEY_B=0xBB67AE85, MUL_A=0xD2511F53, MUL_B=0xCD9E8D57),
    64: dict(
        KEY_A=0x9E3779B97F4A7C15,
        KEY_B=0xBB67AE8584CAA73B,
        MUL_A=0xD2E7470EE14C6C93,
        MUL_B=0xCA5A826395121157,
    ),
}


def randoms_per_offset(width: PhiloxWidth) -> int:
    """u32 values one ``philox_4x`` call yields. Part of a caller's contract.

    It appears in offset arithmetic (``n // RN``), so changing it changes every
    derived stream -- which is why callers pin it rather than infer it.
    """
    _check_width(width)
    return 4 if width == 32 else 8


def _check_width(width: PhiloxWidth) -> None:
    if width not in PHILOX_WIDTHS:
        raise ValueError(f"PHILOX_WIDTH must be one of {PHILOX_WIDTHS}, got {width}")


def _ty(width: PhiloxWidth) -> type[fx.Int32] | type[fx.Int64]:
    return fx.Int32 if width == 32 else fx.Int64


def _const(width: PhiloxWidth, name: str) -> Word:
    return _ty(width)(_CONSTS[width][name])


# The round arithmetic stays on raw `arith` ops rather than the `fx` operator
# overloads. Philox is modular arithmetic over u32/u64, and the overloads
# constant-fold through unbounded Python ints: with a compile-time seed and
# offset every round is foldable, and the key schedule's repeated `k += KEY`
# runs off the end instead of wrapping. At width 64 that raises outright
# ("Python int too large to convert to C long"); at width 32 it only survives
# because `fx.Int32` truncates what it is handed, which is the wrap by luck.
# `arith.addi`/`muli`/`xori` wrap at the declared width, which is the semantics
# this PRNG is defined in.
def _mul_lo(a: Word, b: Word, width: PhiloxWidth) -> Word:
    return _ty(width)(arith.muli(fx.as_ir_value(a), fx.as_ir_value(b)))


def _xor(a: Word, b: Word, width: PhiloxWidth) -> Word:
    return _ty(width)(arith.xori(fx.as_ir_value(a), fx.as_ir_value(b)))


def _add(a: Word, b: Word, width: PhiloxWidth) -> Word:
    return _ty(width)(arith.addi(fx.as_ir_value(a), fx.as_ir_value(b)))


def _mul_hi(a: Word, b: Word, width: PhiloxWidth) -> Word:
    """High half of an unsigned product.

    `arith.mului_extended` yields both halves in one op, which lowers to
    `v_mul_hi_u32` on gfx1201 rather than a shift of a widened product. At width
    64 the backend expands it; that expansion is exactly what makes the 64-bit
    variant expensive, and is the thing the microbenchmark prices.
    """
    op = arith.MulUIExtendedOp(fx.as_ir_value(a), fx.as_ir_value(b))
    return _ty(width)(op.high)


def philox_4x(
    c0: Word,
    c1: Word,
    c2: Word,
    c3: Word,
    k0: Word,
    k1: Word,
    width: PhiloxWidth = 32,
    n_rounds: int = DEFAULT_ROUNDS,
) -> tuple[Word, Word, Word, Word]:
    """`n_rounds` Philox rounds over counter `(c0..c3)` and key `(k0, k1)`.

    Returns the four counter words, which are the random output. Operands must
    already be `fx.Int32` at width 32 or `fx.Int64` at width 64.

    The round is Triton's verbatim, and the ordering matters: `c1` and `c3` are
    computed from the *pre-round* `_c0`/`_c2`, so the temporaries are not an
    optimisation to remove.
    """
    _check_width(width)
    if n_rounds < 1:
        raise ValueError(f"n_rounds must be >= 1, got {n_rounds}")
    A, B = _const(width, "MUL_A"), _const(width, "MUL_B")
    KA, KB = _const(width, "KEY_A"), _const(width, "KEY_B")
    for _ in range(n_rounds):
        _c0, _c2 = c0, c2
        c0 = _xor(_xor(_mul_hi(B, _c2, width), c1, width), k0, width)
        c2 = _xor(_xor(_mul_hi(A, _c0, width), c3, width), k1, width)
        c1 = _mul_lo(B, _c2, width)
        c3 = _mul_lo(A, _c0, width)
        k0 = _add(k0, KA, width)
        k1 = _add(k1, KB, width)
    return c0, c1, c2, c3


def philox_u32(
    seed: U64,
    offset: U64,
    width: PhiloxWidth = 32,
    n_rounds: int = DEFAULT_ROUNDS,
) -> list[fx.Int32]:
    """`randoms_per_offset(width)` uniform u32 values from a 64-bit `(seed, offset)`.

    `seed` and `offset` are `fx.Int64`. This is the entry point callers want:
    it handles the width-dependent packing so they do not have to.

    At width 32 the seed fills the key and the offset the low counter words,
    which is how a 64-bit seed and offset survive 32-bit arithmetic. The
    splitting is free -- a 64-bit value occupies two consecutive 32-bit
    registers, so `trunc` and `shr 32; trunc` lower to register naming.

    At width 64 the whole seed and offset go in one word each, and the four
    u64 outputs are unpacked into eight u32 **low half first**, matching
    Triton's `join(hi, lo)` ordering.
    """
    _check_width(width)
    seed64, off64 = fx.Int64(seed), fx.Int64(offset)
    if width == 32:
        lo, hi = _split64(off64)
        klo, khi = _split64(seed64)
        zero = fx.Int32(0)
        return list(philox_4x(lo, hi, zero, zero, klo, khi, 32, n_rounds))
    words = philox_4x(
        off64, fx.Int64(0), fx.Int64(0), fx.Int64(0), seed64, fx.Int64(0), 64, n_rounds
    )
    out = []
    for w in words:
        w_lo, w_hi = _split64(w)
        out.extend((w_lo, w_hi))
    return out


# --- dropout helpers -------------------------------------------------------
#
# These are not attention: they are "a block of consecutive randoms" and "an
# f32 probability compared as an i32", which is what AOTriton's dropout.py and
# dropout_rng.py are made of. They live here so the attention kernel does not
# grow a PRNG, and so the debug mask kernel can reach them without importing
# the attention builder.
#
# What stays with the caller is the *offset scheme* -- which element gets which
# offset. That is layout-specific and belongs where the layout is.

_U32_SCALE = 2.3283064365386963e-10  # 2**-32, exact


def dropout_threshold(p: float) -> int:
    """The i32 threshold for a drop probability `p`.

    A uniform u32 reinterpreted as i32 is uniform on `[-2**31, 2**31)`, so
    comparing against `(p - 0.5) * 2**32` keeps a `1 - p` fraction and needs no
    float conversion per element -- one compare instead of a convert and a
    float compare, on a path that runs once per score.

    `p = 0` gives `-2**31` (keep everything) and `p -> 1` gives `+2**31`
    (keep nothing). AOTriton computes the same value as
    `((dropout_p - 0.5) * 0xFFFFFFFF).to(tl.int32)`.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"dropout p must be in [0, 1], got {p}")
    t = int((p - 0.5) * 0xFFFFFFFF)
    return max(-(2**31), min(2**31 - 1, t))


def keep_mask(vals: list[fx.Int32], threshold: int | fx.Int32) -> list[fx.Boolean]:
    """`vals > threshold` as **signed** compares -- one predicate per value.

    Signed is not an implementation detail: the randoms span the full u32
    range reinterpreted as `i32`, and `dropout_threshold` is negative for
    every `p < 0.5`. Comparing them unsigned keeps everything, which looks
    like working dropout at a glance because the output is still finite and
    still attention-shaped. AOTriton makes the same choice for the same
    reason, having bitcast to `int32` to get a comparison at all.

    `fx.Int32` is `signed=True`, so its `>` overload emits the signed compare.
    The re-wrap of each `v` is not redundant: the values arrive as raw `u32`
    bit patterns whose Python-side type is incidental, and this module rather
    than each caller should own which way they are read.
    """
    thr = fx.Int32(threshold)
    return [fx.Int32(v) > thr for v in vals]


def to_uniform_f32(v: fx.Int32) -> fx.Float32:
    """A u32 random as a float in `[0, 1)`, for inspection rather than dropout.

    The dropout path never calls this -- that is the point of
    `dropout_threshold`. The debug mask kernel does, because a float mask is
    what a human reads.
    """
    f = fx.Int32(v).to(fx.Float32)
    return f * fx.Float32(_U32_SCALE) + fx.Float32(0.5)


@dataclass(frozen=True, slots=True)
class Philox:
    """A *configured* PRNG: the width and round count that define a stream.

    The free functions below take `width` and `n_rounds` as arguments, which
    means a caller can pass different values in two places and get two
    different streams with nothing to notice. That is not hypothetical for the
    intended use: a dropout mask is generated in the forward pass and
    *regenerated* in the backward pass and in the debug mask kernel, and all
    three must agree bit for bit or gradients are silently wrong.

    So prefer passing one of these around rather than a pair of integers:

        rng = Philox.for_arch()          # or Philox(width=32)
        vals = rng.u32(seed, offset)     # len(vals) == rng.randoms_per_offset

    It is a host-side object read at trace time; nothing survives into the
    kernel but the constants it selects.
    """

    width: PhiloxWidth = 32
    n_rounds: int = DEFAULT_ROUNDS

    def __post_init__(self) -> None:
        _check_width(self.width)
        if self.n_rounds < 1:
            raise ValueError(f"n_rounds must be >= 1, got {self.n_rounds}")

    @classmethod
    def for_arch(
        cls, arch: str | None = None, n_rounds: int = DEFAULT_ROUNDS
    ) -> "Philox":
        """The measured default for `arch` (current device if omitted)."""
        return cls(width=default_width(arch), n_rounds=n_rounds)

    @property
    def randoms_per_offset(self) -> int:
        """u32 values one `u32()` call yields: 4 at width 32, 8 at width 64.

        Part of a caller's own contract, because it appears in offset
        arithmetic (`n // randoms_per_offset`).
        """
        return randoms_per_offset(self.width)

    def u32(self, seed: U64, offset: U64) -> list[fx.Int32]:
        """`randoms_per_offset` uniform u32 values from a 64-bit seed/offset."""
        return philox_u32(seed, offset, self.width, self.n_rounds)

    def words(
        self, c0: Word, c1: Word, c2: Word, c3: Word, k0: Word, k1: Word
    ) -> tuple[Word, Word, Word, Word]:
        """The raw round function, for callers packing the counter themselves."""
        return philox_4x(c0, c1, c2, c3, k0, k1, self.width, self.n_rounds)

    def span_u32(self, seed: U64, first_offset: U64, count: int) -> list[fx.Int32]:
        """`count` consecutive randoms, starting at `first_offset`'s slot 0.

        The blocked form: a run of adjacent stream elements, which is what a
        tile of a dropout mask is. `count` must be a multiple of
        `randoms_per_offset`, because a partial call would still cost a full
        Philox and callers who cannot arrange that should say so explicitly.

        `first_offset` is an **absolute** offset. This function never advances
        a counter across calls -- see `sdpa-dropout-plan.md` §4.2 for why an
        incremental version agrees with the blocked one right up until it
        wraps.
        """
        rn = self.randoms_per_offset
        if count % rn:
            raise ValueError(
                f"count must be a multiple of randoms_per_offset ({rn}), got {count}"
            )
        out: list[fx.Int32] = []
        base = fx.Int64(first_offset)
        for k in range(count // rn):
            out.extend(self.u32(seed, base + fx.Int64(k)))
        return out

    def keep_span(
        self, seed: U64, first_offset: U64, count: int, threshold: int | fx.Int32
    ) -> list[fx.Boolean]:
        """`span_u32` and `keep_mask` together: the dropout mask for a run."""
        return keep_mask(self.span_u32(seed, first_offset, count), threshold)

    # -- the 2D grid scheme -------------------------------------------------
    #
    # These two exist so that a producer and a checker of the same mask cannot
    # hold different opinions about which offset an element gets. That is the
    # whole content of the reproducibility contract: the attention kernel and
    # the debug mask kernel agree because they call these, not because two
    # transcriptions of the same formula happened to match. AOTriton's
    # `fast_dropout_mask` computes the identical thing.
    #
    # Randoms are a row-major `(n_rows, n_cols)` grid per plane, packed
    # `randoms_per_offset` to an offset along the row.

    def grid_plane(
        self,
        offset_base: U64,
        plane: fx.Int32 | int,
        n_rows: fx.Int32 | int,
        n_cols: fx.Int32 | int,
    ) -> tuple[U64, U64]:
        """`(first offset of this plane, offsets per row)`.

        A plane is one `(batch, head)` pair. Built in 64 bits throughout:
        `plane * n_rows * row_stride` reaches 2**32 at B*H = 256 with 8K
        sequences, and a 32-bit wrap there does not fault -- it aliases two
        heads onto one stream, which every statistical test still passes
        (`sdpa-dropout-plan.md` §3.2).
        """
        rn = self.randoms_per_offset
        row_stride = fx.Int64((fx.Int32(n_cols) + fx.Int32(rn - 1)) // fx.Int32(rn))
        base = fx.Int64(offset_base) + fx.Int64(plane) * fx.Int64(n_rows) * row_stride
        return base, row_stride

    def grid_offset(
        self,
        plane_base: U64,
        row_stride: U64,
        row: fx.Int32 | fx.Int64 | int,
        col: fx.Int32 | fx.Int64 | int,
    ) -> U64:
        """The offset holding element `(row, col)` of a plane.

        `col` is a column index, not an offset index -- the division by
        `randoms_per_offset` is here so that callers cannot forget it. The
        element sits at slot `col % randoms_per_offset` of the result.
        """
        return (
            fx.Int64(plane_base)
            + fx.Int64(row) * fx.Int64(row_stride)
            + fx.Int64(col) // fx.Int64(self.randoms_per_offset)
        )


def _split64(v: U64) -> tuple[fx.Int32, fx.Int32]:
    """(low 32, high 32) of a 64-bit value, as `fx.Int32`.

    Free on GPU targets: a 64-bit value is a pair of 32-bit registers, so this
    lowers to register naming rather than a shift and a mask.

    `fx.Int64` is signed, so `>>` is an arithmetic shift where the value is a
    `u64`. That is still correct here because the truncation keeps only bits
    0..31 of the result, which are bits 32..63 of `v` for either shift; the two
    differ only in the top half, which is discarded.
    """
    v = fx.Int64(v)
    return fx.Int32(v), fx.Int32(v >> fx.Int64(32))
