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
params, so one decode body serves both, and a shape-dependent choice (causal
balancing under a window, the pair fold) is a compile-time branch when baked and
a ``select`` on a CTA-uniform flag when runtime, so one binary serves every shape
of a runtime-shape kernel.

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
    ``scale`` divides an extent by a baked factor that divides it: ``n // (d/k)``
    is computed as ``(n*k) // d``, which needs only ``d``'s fastdiv pair.
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

    def val(self, name: str, scale: int = 1):
        """``name // scale`` as an IR value."""
        b = self.b
        if not self.runtime(name):
            return b.const_i32(self._baked(name) // scale)
        if name in self.values:
            v = self.values[name]
        elif name == "nqb":
            bm = self.spec.block_m
            q = self.values["seqlen_q"]
            v = b.div(b.add(q, b.const_i32(bm - 1)), b.const_i32(bm))
        else:  # w
            v = b.mul(
                b.mul(self.val("nqb"), self.val("num_query_heads")), self.val("batch")
            )
        return v if scale == 1 else b.div(v, b.const_i32(scale))

    def div(self, n, name: str, scale: int = 1):
        """``n // (name // scale)``."""
        b = self.b
        if not self.runtime(name):
            return b.div(n, b.const_i32(self._baked(name) // scale))
        if self.fastdiv is None:
            return b.div(n, self.val(name, scale))
        if scale != 1:
            n = b.mul(n, b.const_i32(scale))
        return self.fastdiv(n, name)

    def mod(self, n, name: str, scale: int = 1):
        """``n % (name // scale)``."""
        b = self.b
        if not self.runtime(name):
            return b.mod(n, b.const_i32(self._baked(name) // scale))
        if self.fastdiv is None:
            return b.mod(n, self.val(name, scale))
        return b.sub(n, b.mul(self.div(n, name, scale), self.val(name, scale)))

    def balance(self):
        """Whether the decodes balance causal work (longest-first, fold, pair
        fold): under causal masking. A bool, fixed at compile time."""
        return bool(self.spec.causal)

    def pair_fold(self, NP: int):
        """Whether the pair fold applies: balancing, even NQB and more than one
        round of work (``W > NP``). A bool or an i1, as :meth:`balance`."""
        bal = self.balance()
        if bal is False:
            return False
        nqb = self.static("nqb")
        if nqb is not None and not self.runtime("w"):
            return bal is True and nqb % 2 == 0 and self._baked("w") > NP
        b = self.b
        flag = b.land(
            b.cmp_eq(b.land(self.val("nqb"), b.const_i32(1)), b.const_i32(0)),
            b.cmp_lt(b.const_i32(NP), self.val("w")),
        )
        return flag if bal is True else b.land(flag, bal)

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


class NonpersistHqMinorSwz(NonpersistDecode):
    """Swizzled Head-first (arXiv 2511.02132, Fig. 11): grid
    ``(M*nqb, Hq/M, B)`` with M = ``chiplet_num_xcds``, so XCD ``a`` owns the
    contiguous query-head band ``[a*Hq/M, (a+1)*Hq/M)`` and runs all its query
    blocks before the next head. Batch is the slowest digit. Query blocks run
    longest-first under causal masking."""

    name = "hq_minor_swz"
    tag = "nphqminswz"

    def check(self, spec):
        if spec.num_query_heads % spec.chiplet_num_xcds != 0:
            return (
                f"{self.name} needs num_query_heads ({spec.num_query_heads}) "
                f"divisible by chiplet_num_xcds ({spec.chiplet_num_xcds})"
            )
        return None

    def grid(self, spec):
        M = spec.chiplet_num_xcds
        return (M * _spec_nqb(spec), spec.num_query_heads // M, spec.batch)

    def emit_decode(self, b, spec, bx, by, bz, shape):
        M = spec.chiplet_num_xcds
        # Head band of Hq/M heads per XCD.
        hq = b.add(
            b.mul(b.mod(bx, b.const_i32(M)), shape.val("num_query_heads", M)), by
        )
        blk = b.div(bx, b.const_i32(M))
        qb = shape.branch(
            shape.balance(), lambda: emit_reverse(b, blk, shape), lambda: blk
        )
        return Decoded(qb, hq, bz)


NONPERSIST_DECODES = _registry(
    NonpersistQbMinor, NonpersistBtHkvMinor, NonpersistHqMinorSwz
)


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


def emit_pair_unit(b, wi, shape: DecodeShape, NP: int):
    """``(unit, half)`` for ``wi < W`` (``W`` even): unit ``u`` holds items
    ``2u, 2u+1``, and one CTA runs both halves on consecutive grid-stride steps.
    In the last round, CTAs that still get two items get both halves of a unit.

    Branch-free: every case is emitted and selected, each correction a no-op where
    it does not apply (no whole double rounds, no tail, no paired tail); the only
    divisors are NP, 2*NP and 2."""
    c = b.const_i32
    W = shape.val("w")
    # Items before the last round: whole double rounds of 2*NP items.
    rounds_end = b.mul(b.div(W, c(2 * NP)), c(2 * NP))
    re_half = b.lshr(rounds_end, c(1))
    paired_tail = b.sub(b.sub(W, rounds_end), c(NP))  # may be <= 0
    pt = b.smax(paired_tail, c(0))
    j = b.div(wi, c(NP))
    unit = b.add(b.sub(wi, b.mul(j, c(NP))), b.mul(b.div(j, c(2)), c(NP)))
    half = b.mod(j, c(2))
    # Last round: CTAs below `paired_tail` still get both halves of a unit.
    r = b.sub(wi, rounds_end)
    s = b.sub(r, pt)
    unit_t = b.add(b.add(re_half, pt), b.div(s, c(2)))
    half_t = b.mod(s, c(2))
    lo = b.cmp_lt(r, paired_tail)  # never when paired_tail <= 0
    hi = b.cmp_lt(c(NP - 1), r)  # never when paired_tail <= 0 (tail <= NP)
    unit_t = b.select(
        lo,
        b.add(re_half, r),
        b.select(hi, b.add(b.sub(re_half, c(NP)), r), unit_t),
    )
    half_t = b.select(lo, c(0), b.select(hi, c(1), half_t))
    in_rounds = b.cmp_lt(wi, rounds_end)
    return b.select(in_rounds, unit, unit_t), b.select(in_rounds, half, half_t)


def emit_pair_block(b, p, half, shape: DecodeShape):
    """Query block of fold pair ``p``: ``p`` for the first half, ``NQB-1-p`` for
    the second, so every unit costs the same under causal masking."""
    first = b.cmp_lt(half, b.const_i32(1))
    last = b.sub(shape.val("nqb"), b.const_i32(1))
    return b.select(first, p, b.sub(last, p))


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
    grid-stride spreads cheap and expensive blocks over each CTA. Under causal
    masking the blocks are folded; ``interleave``
    instead reverses the block for odd ``hq`` (measured behind the fold, so auto
    never sets it). Keying the reversal on ``hq`` keeps it a bijection: each head's
    blocks are either in order or reversed."""

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
            qb = shape.branch(
                shape.balance(), lambda: emit_fold(b, qb0, shape), lambda: qb0
            )
        return Decoded(qb, hq, bt)


class PersistHkvMajor(PersistDecode):
    """``wi = hkv*(NQB*gqa*B) + blk*(gqa*B) + hql*B + bt``: each grid-stride
    phase stays within about one kv head, so its K/V stays in L2 across the GQA
    group. Under causal masking, with even NQB and
    ``W > NP``, one CTA runs both blocks of a fold pair ``{p, NQB-1-p}`` on
    consecutive steps (as ``hq_minor_swz``); otherwise the block is folded."""

    name = "hkv_major"
    tag = "hkvmaj"
    fastdiv_divisors = ("batch", "gqa", "nqb")

    @staticmethod
    def _digits(b, idx, shape: DecodeShape, nqb_scale: int):
        """``idx = hkv*(nblk*gqa*B) + blk*(gqa*B) + hql*B + bt`` ->
        ``(bt, hq, blk, hkv)``, with ``nblk = NQB // nqb_scale``."""
        bt = shape.mod(idx, "batch")
        rem = shape.div(idx, "batch")  # hkv*(nblk*gqa) + blk*gqa + hql
        hql = shape.mod(rem, "gqa")
        r2 = shape.div(rem, "gqa")  # hkv*nblk + blk
        blk = shape.mod(r2, "nqb", nqb_scale)
        hkv = shape.div(r2, "nqb", nqb_scale)
        hq = b.add(b.mul(hkv, shape.val("gqa")), hql)
        return bt, hq, blk, hkv

    def emit_decode(self, b, spec, wi, shape):
        NP = spec.num_persistent

        def paired():
            unit, half = emit_pair_unit(b, wi, shape, NP)
            bt, hq, p, hkv = self._digits(b, unit, shape, 2)
            return emit_pair_block(b, p, half, shape), hq, bt, hkv

        def folded():
            bt, hq, blk, hkv = self._digits(b, wi, shape, 1)
            return emit_fold(b, blk, shape), hq, bt, hkv

        return Decoded(*shape.branch(shape.pair_fold(NP), paired, folded))


class PersistBtHkvMinor(PersistDecode):
    """``wi = ((blk*gqa + hql)*Hkv + hkv)*B + bt``: batch, then kv head, are the
    fastest digits, so each XCD's CTAs stride over few K/V streams; query
    blocks are folded under causal masking."""

    name = "bt_hkv_minor"
    tag = "bthkvmin"
    fastdiv_divisors = ("batch", "num_query_heads")

    def fastdiv_dividend_bound(self, spec) -> int:
        # gqa * t with t < Hq, besides the work index itself.
        return max(
            super().fastdiv_dividend_bound(spec),
            spec.num_queries_per_kv * spec.num_query_heads,
        )

    def emit_decode(self, b, spec, wi, shape):
        # rem = blk*Hq + t with t = hql*Hkv + hkv < Hq, so the head digits split
        # on Hq: hql = (gqa*t) // Hq exactly (gqa*hkv < gqa*Hkv = Hq), and no
        # divide by Hkv is needed.
        bt = shape.mod(wi, "batch")
        rem = shape.div(wi, "batch")
        t = shape.mod(rem, "num_query_heads")
        blk = shape.div(rem, "num_query_heads")
        hql = shape.div(b.mul(t, shape.val("gqa")), "num_query_heads")
        hkv = b.sub(t, b.mul(hql, shape.val("num_kv_heads")))
        hq = b.add(b.mul(hkv, shape.val("gqa")), hql)
        qb = shape.branch(
            shape.balance(), lambda: emit_fold(b, blk, shape), lambda: blk
        )
        return Decoded(qb, hq, bt, hkv)


class PersistHqMinorSwz(PersistDecode):
    """Swizzled Head-first on the persistent grid: XCD ``a`` (``wi % M``, needs
    ``M | NP``) owns head band ``a``. Under causal masking one CTA runs both blocks of a fold pair ``{p, NQB-1-p}`` on consecutive steps, so per-CTA cost
    is balanced however NP and NQB divide; the unit's XCD is ``wi % M``. Per XCD,
    fastest first: pair, head in band, batch. With ``W <= NP`` the plain decode
    is emitted.

    Odd NQB is not supported. It worked when tried: also pair the middle blocks
    of two consecutive (head, batch) of the band into one unit (with
    ``NQB = 2P+1`` it costs ``2·(P+1) = NQB+1``, as a fold pair), and run the
    per-XCD leftover middle (odd heads x batch per band) as single items at the
    end of the work index (``wi >= W - M``, XCD ``wi % M``)."""

    name = "hq_minor_swz"
    tag = "hqminswz"
    fastdiv_divisors = ("nqb", "num_query_heads", "gqa")

    def check(self, spec):
        M = spec.chiplet_num_xcds
        if spec.num_query_heads % M != 0 or spec.num_persistent % M != 0:
            return (
                f"{self.name} needs num_query_heads and num_persistent divisible by {M}"
            )
        if spec.causal and _spec_nqb(spec) % 2 != 0:
            return (
                f"{self.name} needs an even number of query blocks under causal "
                "masking"
            )
        return None

    def emit_decode(self, b, spec, wi, shape):
        M, NP = spec.chiplet_num_xcds, spec.num_persistent

        def paired():
            unit, half = emit_pair_unit(b, wi, shape, NP)
            a = b.mod(unit, b.const_i32(M))
            rest = b.div(unit, b.const_i32(M))
            p = shape.mod(rest, "nqb", 2)
            hb = shape.div(rest, "nqb", 2)
            return a, hb, emit_pair_block(b, p, half, shape)

        def plain():
            a = b.mod(wi, b.const_i32(M))
            rest = b.div(wi, b.const_i32(M))
            qb = shape.mod(rest, "nqb")  # if balancing: W <= NP, one item per CTA
            hb = shape.div(rest, "nqb")
            return a, hb, qb

        a, hb, qb = shape.branch(shape.pair_fold(NP), paired, plain)
        # Head band of Hq/M heads per XCD: hb // (Hq/M) == (M*hb) // Hq.
        hq = b.add(
            b.mul(a, shape.val("num_query_heads", M)),
            shape.mod(hb, "num_query_heads", M),
        )
        bt = shape.div(hb, "num_query_heads", M)
        return Decoded(qb, hq, bt)


def _aligned_causal_error(spec, name: str):
    if not spec.persistent or not spec.causal:
        return f"{name} requires persistent causal attention"
    if spec.ragged or spec.varlen or spec.paged:
        return f"{name} is validated only for aligned dense attention"
    return None


class PersistGqaPair(PersistDecode):
    """NP = NQB*Hkv*B CTAs; two neighbouring CTAs cover one (qb pair, hkv, bt)
    group, each half the local query heads at both complementary blocks, so the
    two costs sum to a constant. Not auto-selected: behind the best order on
    every shape auto used to pick it for."""

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
    (qb pair, hkv, bt), and phase 0/1 selects the complementary blocks. Not
    auto-selected: behind the best order on every shape auto used to pick it
    for."""

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
    PersistBtHkvMinor,
    PersistHqMinorSwz,
    PersistGqaPair,
    PersistGqaPair2Phase,
)
