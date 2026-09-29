# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Work-item decodes for the dense attention kernel.

A decode maps a CTA's work index to the (query block, query head, batch) it
computes. Every decode is a bijection onto the work space, so the choice only
moves which items run together (L2 locality, causal load balance), never the
result. One class per ``persist_decode`` / ``nonpersist_decode`` value keeps
each decode's IR, grid, legality check and kernel-name tag together; both arch
builders call the same class.

The shape reaches a decode through :class:`DecodeShape`: each extent is either
baked (emitted as a constant at its use) or a runtime value from the kernel's
params, so one decode body serves both.

The classes read spec attributes and the shape, and emit through the IR builder
``b``.
"""

from __future__ import annotations

from typing import Any, ClassVar, NamedTuple


class Decoded(NamedTuple):
    """A decoded work item. ``hkv`` is set only where the decode has the kv
    head as a digit; otherwise callers derive it as ``hq // gqa``."""

    qb: Any
    hq: Any
    bt: Any
    hkv: Any = None


def _spec_nqb(spec) -> int:
    """Query blocks per head for the spec's shape (a ragged last block counts)."""
    return (spec.seqlen_q + spec.block_m - 1) // spec.block_m


def _registry(*classes) -> dict:
    return {c.name: c() for c in classes}


class DecodeShape:
    """The problem shape as the decodes see it.

    ``values`` maps an extent (``batch``, ``seqlen_q``, ``seqlen_kv``,
    ``num_query_heads``, ``num_kv_heads``, ``gqa``, ``nqb``, ``w``) to the runtime
    IR value the kernel holds; any other extent is baked and emitted as a constant
    at its use. The decodes are written once on these values, and LLVM folds the
    baked ones. ``nqb`` and ``w``, when not passed, are derived from ``seqlen_q``
    (ceil) and the heads/batch at each use.

    A runtime divide goes through ``fastdiv(n, name)`` when the builder passes one
    (the persistent loop: host-computed magic/shift pairs, no division in the
    loop), else it is a plain division (once per CTA on the default grid).
    """

    def __init__(self, b, spec, values=None, fastdiv=None):
        self.b, self.spec = b, spec
        self.values = dict(values or {})
        self.fastdiv = fastdiv

    def _baked(self, name: str) -> int:
        s = self.spec
        nqb = _spec_nqb(s)
        return {
            "batch": s.batch,
            "seqlen_q": s.seqlen_q,
            "seqlen_kv": s.seqlen_kv,
            "num_query_heads": s.num_query_heads,
            "num_kv_heads": s.num_kv_heads,
            "gqa": s.num_queries_per_kv,
            "nqb": nqb,
            "w": nqb * s.num_query_heads * s.batch,
        }[name]

    def runtime(self, name: str) -> bool:
        if name in self.values:
            return True
        if name == "nqb":
            return self.runtime("seqlen_q")
        if name == "w":
            return any(self.runtime(n) for n in ("nqb", "num_query_heads", "batch"))
        return False

    def static(self, name: str) -> int | None:
        """The baked value, or None when ``name`` is runtime."""
        return None if self.runtime(name) else self._baked(name)

    def val(self, name: str):
        """``name`` as an IR value."""
        b = self.b
        if not self.runtime(name):
            return b.const_i32(self._baked(name))
        if name in self.values:
            return self.values[name]
        if name == "nqb":
            bm = self.spec.block_m
            q = self.values["seqlen_q"]
            return b.div(b.add(q, b.const_i32(bm - 1)), b.const_i32(bm))
        # w
        return b.mul(
            b.mul(self.val("nqb"), self.val("num_query_heads")), self.val("batch")
        )

    def div(self, n, name: str):
        """``n // name``."""
        b = self.b
        if not self.runtime(name):
            return b.div(n, b.const_i32(self._baked(name)))
        if self.fastdiv is None:
            return b.div(n, self.val(name))
        return self.fastdiv(n, name)

    def mod(self, n, name: str):
        """``n % name``."""
        b = self.b
        if not self.runtime(name):
            return b.mod(n, b.const_i32(self._baked(name)))
        if self.fastdiv is None:
            return b.mod(n, self.val(name))
        return b.sub(n, b.mul(self.div(n, name), self.val(name)))

    def balance(self):
        """Whether the decodes balance causal work (longest-first, fold, pair
        fold): under causal masking. A bool, fixed at compile time."""
        return bool(self.spec.causal)

    def branch(self, flag, then, other):
        """``then()`` if ``flag`` else ``other()``: only the taken side is
        emitted when ``flag`` is a bool; with an i1 both are emitted and the
        results (a value or a tuple of values) selected."""
        if isinstance(flag, bool):
            return then() if flag else other()
        t, o = then(), other()
        if isinstance(t, tuple):
            return tuple(self.b.select(flag, x, y) for x, y in zip(t, o))
        return self.b.select(flag, t, o)


# --------------------------------------------------------------------------- #
# Non-persistent grid: one CTA per work item.
# --------------------------------------------------------------------------- #


def emit_reverse(b, blk, shape: DecodeShape):
    """Longest-first query-block order for causal cost: ``NQB-1-blk``.

    Under a runtime shape one binary serves every seqlen, so NQB must come from
    the param; a baked NQB would reverse against the wrong extent.
    """
    return b.sub(b.sub(shape.val("nqb"), b.const_i32(1)), blk)


class NonpersistDecode:
    """Block ids -> work item, for the one-CTA-per-item grid.

    The hardware dispatches workgroups x-fastest and round-robins them over the
    XCDs, so the grid shape and the decode together decide which items share an
    XCD's L2 and which run last.
    """

    name: ClassVar[str]
    tag: ClassVar[str] = ""  # kernel-name part; "" = none

    def check(self, spec) -> str | None:
        """Why ``spec`` cannot use this decode, or None."""
        return None

    def grid(self, spec) -> tuple[int, int, int]:
        """Launch grid, from the spec's actual (host-side) shape."""
        raise NotImplementedError

    def emit_decode(self, b, spec, bx, by, bz, shape: DecodeShape) -> Decoded:
        raise NotImplementedError


class NonpersistQbMinor(NonpersistDecode):
    """Grid ``(nqb, Hq, B)``: the query block is the fastest digit."""

    name = "qb_minor"

    def grid(self, spec):
        return (_spec_nqb(spec), spec.num_query_heads, spec.batch)

    def emit_decode(self, b, spec, bx, by, bz, shape):
        return Decoded(bx, by, bz)


class NonpersistBtHkvMinor(NonpersistDecode):
    """Grid ``(B, Hq, nqb)`` with the kv head as the low digit of the head axis:
    batch, then kv head, are the fastest digits, so the XCD round-robin spreads
    distinct K/V streams over the XCDs. Query blocks run longest-first under
    causal masking."""

    name = "bt_hkv_minor"
    tag = "npbthkvmin"

    def grid(self, spec):
        return (spec.batch, spec.num_query_heads, _spec_nqb(spec))

    def emit_decode(self, b, spec, bx, by, bz, shape):
        hq = b.add(
            b.mul(shape.mod(by, "num_kv_heads"), shape.val("gqa")),
            shape.div(by, "num_kv_heads"),
        )
        qb = shape.branch(
            shape.balance(), lambda: emit_reverse(b, bz, shape), lambda: bz
        )
        return Decoded(qb, hq, bx)


NONPERSIST_DECODES = _registry(NonpersistQbMinor, NonpersistBtHkvMinor)


# --------------------------------------------------------------------------- #
# Persistent grid: NP CTAs grid-stride over the work index ``wi``.
# --------------------------------------------------------------------------- #


def _baked_only(shape: DecodeShape, name: str) -> int:
    """NQB for the decodes that exist only on baked-shape bodies (gfx950)."""
    nqb = shape.static("nqb")
    if nqb is None:
        raise NotImplementedError(f"persist_decode={name!r} needs a baked shape")
    return nqb


def emit_fold(b, blk, shape: DecodeShape):
    """Causal fold: ``blk < half -> blk``, else ``NQB-1-(blk-half)``, so a CTA
    striding both halves pairs a cheap block with an expensive one."""
    nqb = shape.val("nqb")
    half = b.lshr(nqb, b.const_i32(1))
    qb_hi = b.sub(b.add(b.sub(nqb, b.const_i32(1)), half), blk)  # NQB-1-(blk-half)
    return b.select(b.cmp_lt(blk, half), blk, qb_hi)


class PersistDecode:
    """Grid-stride work index ``wi`` -> work item, for the persistent grid.

    ``fastdiv_divisors``: the extents a runtime-shape body divides by in the
    grid-stride loop, each needing a magic/shift kernarg pair; empty for the
    decodes that only baked-shape bodies emit."""

    name: ClassVar[str]
    tag: ClassVar[str] = ""  # kernel-name part; "" = none
    fastdiv_divisors: ClassVar[tuple[str, ...]] = ()

    def check(self, spec) -> str | None:
        """Why ``spec`` cannot use this decode, or None."""
        return None

    def fastdiv_dividend_bound(self, spec) -> int:
        """An upper bound on every runtime fastdiv dividend: the work count."""
        return _spec_nqb(spec) * spec.num_query_heads * spec.batch

    def emit_decode(self, b, spec, wi, shape: DecodeShape) -> Decoded:
        raise NotImplementedError


class PersistQbMajor(PersistDecode):
    """``wi = qb*(Hq*B) + hq*B + bt``: the query block is the slowest digit, so
    grid-stride spreads cheap and expensive blocks over each CTA. ``interleave``
    reverses the block for odd ``hq``. Keying the reversal on ``hq`` keeps it a
    bijection: each head's blocks are either in order or reversed."""

    name = "qb_major"
    fastdiv_divisors = ("batch", "num_query_heads", "gqa")

    def emit_decode(self, b, spec, wi, shape):
        bt = shape.mod(wi, "batch")
        rem = shape.div(wi, "batch")
        hq = shape.mod(rem, "num_query_heads")
        qb0 = shape.div(rem, "num_query_heads")
        if spec.interleave and spec.causal:
            # With NQB == 1, qb0 is 0 and the flip is too.
            odd = b.cmp_eq(b.mod(hq, b.const_i32(2)), b.const_i32(1))
            last = b.sub(shape.val("nqb"), b.const_i32(1))
            qb = b.select(odd, b.sub(last, qb0), qb0)
        else:
            qb = qb0
        return Decoded(qb, hq, bt)


class PersistHkvMajor(PersistDecode):
    """``wi = hkv*(NQB*gqa*B) + blk*(gqa*B) + hql*B + bt``: each grid-stride
    phase stays within about one kv head, so its K/V stays in L2 across the GQA
    group. The block is folded."""

    name = "hkv_major"
    tag = "hkvmaj"
    fastdiv_divisors = ("batch", "gqa", "nqb")

    def emit_decode(self, b, spec, wi, shape):
        bt = shape.mod(wi, "batch")
        rem = shape.div(wi, "batch")  # hkv*(NQB*gqa) + blk*gqa + hql
        hql = shape.mod(rem, "gqa")
        r2 = shape.div(rem, "gqa")  # hkv*NQB + blk
        blk = shape.mod(r2, "nqb")
        hkv = shape.div(r2, "nqb")
        hq = b.add(b.mul(hkv, shape.val("gqa")), hql)
        return Decoded(emit_fold(b, blk, shape), hq, bt, hkv)


def _aligned_causal_error(spec, name: str):
    if not spec.persistent or not spec.causal:
        return f"{name} requires persistent causal attention"
    if spec.ragged or spec.varlen or spec.paged:
        return f"{name} is validated only for aligned dense attention"
    return None


class PersistGqaPair(PersistDecode):
    """NP = NQB*Hkv*B CTAs; two neighbouring CTAs cover one (qb pair, hkv, bt)
    group, each half the local query heads at both complementary blocks, so the
    two costs sum to a constant."""

    name = "gqa_pair"
    tag = "gqapair"

    def check(self, spec):
        nqb = _spec_nqb(spec)
        expected_np = nqb * spec.num_kv_heads * spec.batch
        why = _aligned_causal_error(spec, self.name)
        if why:
            return why
        if nqb % 2 != 0 or spec.num_queries_per_kv % 2 != 0:
            return f"{self.name} requires even NQB and even GQA ratio"
        if spec.num_persistent != expected_np:
            return (
                f"{self.name} requires num_persistent == NQB*Hkv*B "
                f"({expected_np}), got {spec.num_persistent}"
            )
        return None

    def emit_decode(self, b, spec, wi, shape):
        nqb = _baked_only(shape, self.name)
        NP, B, Hkv = spec.num_persistent, spec.batch, spec.num_kv_heads
        gqa = spec.num_queries_per_kv
        cta = b.mod(wi, b.const_i32(NP))
        phase = b.div(wi, b.const_i32(NP))
        pair_lane = b.mod(cta, b.const_i32(2))
        rem = b.div(cta, b.const_i32(2))
        bt = b.mod(rem, b.const_i32(B))
        rem = b.div(rem, b.const_i32(B))
        hkv = b.mod(rem, b.const_i32(Hkv))
        qb_pair = b.div(rem, b.const_i32(Hkv))
        half_gqa = gqa // 2
        high = b.cmp_ge(phase, b.const_i32(half_gqa))
        phase_half = b.mod(phase, b.const_i32(half_gqa))
        hql = b.add(b.mul(pair_lane, b.const_i32(half_gqa)), phase_half)
        hq = b.add(b.mul(hkv, b.const_i32(gqa)), hql)
        qb = b.select(high, b.sub(b.const_i32(nqb - 1), qb_pair), qb_pair)
        return Decoded(qb, hq, bt, hkv)


class PersistGqaPair2Phase(PersistDecode):
    """NP = W/2 CTAs; gqa neighbouring CTAs cover all local query heads of one
    (qb pair, hkv, bt), and phase 0/1 selects the complementary blocks."""

    name = "gqa_pair_2phase"
    tag = "gqapair2"

    def check(self, spec):
        gqa = spec.num_queries_per_kv
        nqb = _spec_nqb(spec)
        expected_np = nqb * spec.num_kv_heads * spec.batch * gqa // 2
        why = _aligned_causal_error(spec, self.name)
        if why:
            return why
        if nqb % 2 != 0 or gqa < 2:
            return f"{self.name} requires even NQB and GQA ratio >= 2"
        if spec.num_persistent != expected_np:
            return (
                f"{self.name} requires num_persistent == W/2 "
                f"({expected_np}), got {spec.num_persistent}"
            )
        return None

    def emit_decode(self, b, spec, wi, shape):
        nqb = _baked_only(shape, self.name)
        NP, B, Hkv = spec.num_persistent, spec.batch, spec.num_kv_heads
        gqa = spec.num_queries_per_kv
        cta = b.mod(wi, b.const_i32(NP))
        phase = b.div(wi, b.const_i32(NP))
        hql = b.mod(cta, b.const_i32(gqa))
        rem = b.div(cta, b.const_i32(gqa))
        bt = b.mod(rem, b.const_i32(B))
        rem = b.div(rem, b.const_i32(B))
        hkv = b.mod(rem, b.const_i32(Hkv))
        qb_pair = b.div(rem, b.const_i32(Hkv))
        hq = b.add(b.mul(hkv, b.const_i32(gqa)), hql)
        qb = b.select(
            b.cmp_ne(phase, b.const_i32(0)),
            b.sub(b.const_i32(nqb - 1), qb_pair),
            qb_pair,
        )
        return Decoded(qb, hq, bt, hkv)


PERSIST_DECODES = _registry(
    PersistQbMajor,
    PersistHkvMajor,
    PersistGqaPair,
    PersistGqaPair2Phase,
)
