# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Work-item decodes for the dense attention kernel.

A decode maps a CTA's work index to the (query block, query head, batch) it
computes. Every decode is a bijection onto the work space, so the choice only
moves which items run together (L2 locality, causal load balance), never the
result. One class per ``persist_decode`` / ``nonpersist_decode`` value keeps
each decode's IR, grid, legality check and kernel-name tag together; both arch
builders call the same class.

``seqlen_q`` is the kernel's runtime param, or None when the shape is baked;
decodes read NQB from the spec in the latter case.

The classes read only spec attributes and emit through the IR builder ``b``.
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


# --------------------------------------------------------------------------- #
# Non-persistent grid: one CTA per work item.
# --------------------------------------------------------------------------- #


def emit_reverse(b, blk, spec, seqlen_q):
    """Longest-first query-block order for causal cost: ``NQB-1-blk``.

    Under a runtime shape one binary serves every seqlen, so NQB must come from
    the param; a baked NQB would reverse against the wrong extent.
    """
    if seqlen_q is None:
        nqb = _spec_nqb(spec)
        return b.sub(b.const_i32(nqb - 1), blk) if nqb > 1 else blk
    bm = spec.block_m
    nqb = b.div(b.add(seqlen_q, b.const_i32(bm - 1)), b.const_i32(bm))
    return b.sub(b.sub(nqb, b.const_i32(1)), blk)


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

    def emit_decode(self, b, spec, bx, by, bz, seqlen_q) -> Decoded:
        raise NotImplementedError


class NonpersistQbMinor(NonpersistDecode):
    """Grid ``(nqb, Hq, B)``: the query block is the fastest digit."""

    name = "qb_minor"

    def grid(self, spec):
        return (_spec_nqb(spec), spec.num_query_heads, spec.batch)

    def emit_decode(self, b, spec, bx, by, bz, seqlen_q):
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

    def emit_decode(self, b, spec, bx, by, bz, seqlen_q):
        Hkv, gqa = spec.num_kv_heads, spec.num_queries_per_kv
        hq = b.add(
            b.mul(b.mod(by, b.const_i32(Hkv)), b.const_i32(gqa)),
            b.div(by, b.const_i32(Hkv)),
        )
        qb = emit_reverse(b, bz, spec, seqlen_q) if spec.causal else bz
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
        if spec.num_query_heads % spec.chiplet_num_xcds:
            return (
                f"{self.name} needs num_query_heads ({spec.num_query_heads}) "
                f"divisible by chiplet_num_xcds ({spec.chiplet_num_xcds})"
            )
        return None

    def grid(self, spec):
        M = spec.chiplet_num_xcds
        return (M * _spec_nqb(spec), spec.num_query_heads // M, spec.batch)

    def emit_decode(self, b, spec, bx, by, bz, seqlen_q):
        M = spec.chiplet_num_xcds
        band = spec.num_query_heads // M
        hq = b.add(b.mul(b.mod(bx, b.const_i32(M)), b.const_i32(band)), by)
        blk = b.div(bx, b.const_i32(M))
        qb = emit_reverse(b, blk, spec, seqlen_q) if spec.causal else blk
        return Decoded(qb, hq, bz)


NONPERSIST_DECODES = _registry(
    NonpersistQbMinor, NonpersistBtHkvMinor, NonpersistHqMinorSwz
)


# --------------------------------------------------------------------------- #
# Persistent grid: NP CTAs grid-stride over the work index ``wi``.
# --------------------------------------------------------------------------- #


def _baked_nqb(spec, seqlen_q) -> int:
    """NQB for the persistent decodes, which emit it as constants."""
    if seqlen_q is not None:
        raise NotImplementedError(
            "persistent decodes need a baked shape; a runtime seqlen_q needs "
            "NQB emitted as a value in each decode"
        )
    return _spec_nqb(spec)


def emit_fold(b, blk, nqb: int):
    """Causal fold: ``blk < half -> blk``, else ``NQB-1-(blk-half)``, so a CTA
    striding both halves pairs a cheap block with an expensive one."""
    half = nqb // 2
    qb_hi = b.sub(b.const_i32(nqb - 1 + half), blk)  # NQB-1-(blk-half)
    return b.select(b.cmp_lt(blk, b.const_i32(half)), blk, qb_hi)


def emit_pair_unit(b, wi, W: int, NP: int):
    """``(unit, half)`` for ``wi < W`` (``W`` even): unit ``u`` holds items
    ``2u, 2u+1``, and one CTA runs both halves on consecutive grid-stride steps.
    In the last round, CTAs that still get two items get both halves of a unit."""
    # Items before the last round: whole double rounds of 2*NP items.
    rounds_end = (W // (2 * NP)) * (2 * NP)
    unit = half = None
    if rounds_end > 0:
        j = b.div(wi, b.const_i32(NP))
        unit = b.add(
            b.sub(wi, b.mul(j, b.const_i32(NP))),
            b.mul(b.div(j, b.const_i32(2)), b.const_i32(NP)),
        )
        half = b.mod(j, b.const_i32(2))
    if rounds_end < W:
        # Last round: CTAs below `paired_tail` still get both halves of a unit.
        paired_tail = W - rounds_end - NP
        r = b.sub(wi, b.const_i32(rounds_end))
        s = b.sub(r, b.const_i32(max(paired_tail, 0)))
        unit_t = b.add(
            b.const_i32(rounds_end // 2 + max(paired_tail, 0)), b.div(s, b.const_i32(2))
        )
        half_t = b.mod(s, b.const_i32(2))
        if paired_tail > 0:
            lo = b.cmp_lt(r, b.const_i32(paired_tail))
            hi = b.cmp_lt(b.const_i32(NP - 1), r)
            unit_t = b.select(
                lo,
                b.add(b.const_i32(rounds_end // 2), r),
                b.select(hi, b.add(b.const_i32(rounds_end // 2 - NP), r), unit_t),
            )
            half_t = b.select(lo, b.const_i32(0), b.select(hi, b.const_i32(1), half_t))
        if rounds_end > 0:
            in_rounds = b.cmp_lt(wi, b.const_i32(rounds_end))
            unit = b.select(in_rounds, unit, unit_t)
            half = b.select(in_rounds, half, half_t)
        else:
            unit, half = unit_t, half_t
    return unit, half


def emit_pair_block(b, p, half, nqb: int):
    """Query block of fold pair ``p``: ``p`` for the first half, ``NQB-1-p`` for
    the second, so every unit costs the same under causal masking."""
    return b.select(b.cmp_lt(half, b.const_i32(1)), p, b.sub(b.const_i32(nqb - 1), p))


class PersistDecode:
    """Grid-stride work index ``wi`` -> work item, for the persistent grid.

    ``seqlen_q`` is the kernel's runtime param, or None when the shape is baked
    (the only case the persistent builders emit today)."""

    name: ClassVar[str]
    tag: ClassVar[str] = ""  # kernel-name part; "" = none

    def check(self, spec) -> str | None:
        """Why ``spec`` cannot use this decode, or None."""
        return None

    def emit_decode(self, b, spec, wi, seqlen_q) -> Decoded:
        raise NotImplementedError


class PersistQbMajor(PersistDecode):
    """``wi = qb*(Hq*B) + hq*B + bt``: the query block is the slowest digit, so
    grid-stride spreads cheap and expensive blocks over each CTA. Under causal
    masking the blocks are folded; ``interleave`` instead reverses the block on
    odd ``qb*Hq + hq`` (measured behind the fold, so auto never sets it)."""

    name = "qb_major"

    def emit_decode(self, b, spec, wi, seqlen_q):
        nqb = _baked_nqb(spec, seqlen_q)
        B, Hq = spec.batch, spec.num_query_heads
        bt = b.mod(wi, b.const_i32(B))
        rem = b.div(wi, b.const_i32(B))
        hq = b.mod(rem, b.const_i32(Hq))
        qb0 = b.div(rem, b.const_i32(Hq))
        if spec.interleave and spec.causal and nqb > 1:
            odd = b.cmp_eq(b.mod(rem, b.const_i32(2)), b.const_i32(1))
            qb = b.select(odd, b.sub(b.const_i32(nqb - 1), qb0), qb0)
        elif spec.causal:
            qb = emit_fold(b, qb0, nqb)
        else:
            qb = qb0
        return Decoded(qb, hq, bt)


class PersistHkvMajor(PersistDecode):
    """``wi = hkv*(NQB*gqa*B) + blk*(gqa*B) + hql*B + bt``: each grid-stride
    phase stays within about one kv head, so its K/V stays in L2 across the GQA
    group. Under causal masking with even NQB and ``W > NP`` one CTA runs both
    blocks of a fold pair ``{p, NQB-1-p}`` on consecutive steps (as
    ``hq_minor_swz``); otherwise the block is folded."""

    name = "hkv_major"
    tag = "hkvmaj"

    @staticmethod
    def _digits(b, idx, B: int, gqa: int, nblk: int):
        """``idx = hkv*(nblk*gqa*B) + blk*(gqa*B) + hql*B + bt`` ->
        ``(bt, hq, blk, hkv)``."""
        bt = b.mod(idx, b.const_i32(B))
        rem = b.div(idx, b.const_i32(B))  # hkv*(nblk*gqa) + blk*gqa + hql
        hql = b.mod(rem, b.const_i32(gqa))
        r2 = b.div(rem, b.const_i32(gqa))  # hkv*nblk + blk
        blk = b.mod(r2, b.const_i32(nblk))
        hkv = b.div(r2, b.const_i32(nblk))
        hq = b.add(b.mul(hkv, b.const_i32(gqa)), hql)
        return bt, hq, blk, hkv

    def emit_decode(self, b, spec, wi, seqlen_q):
        nqb = _baked_nqb(spec, seqlen_q)
        B, gqa, NP = spec.batch, spec.num_queries_per_kv, spec.num_persistent
        W = nqb * spec.num_query_heads * B
        if spec.causal and nqb % 2 == 0 and W > NP:
            unit, half = emit_pair_unit(b, wi, W, NP)
            bt, hq, p, hkv = self._digits(b, unit, B, gqa, nqb // 2)
            qb = emit_pair_block(b, p, half, nqb)
        else:
            bt, hq, blk, hkv = self._digits(b, wi, B, gqa, nqb)
            qb = emit_fold(b, blk, nqb)
        return Decoded(qb, hq, bt, hkv)


class PersistBtHkvMinor(PersistDecode):
    """``wi = ((blk*gqa + hql)*Hkv + hkv)*B + bt``: batch, then kv head, are the
    fastest digits, so each XCD's CTAs stride over few K/V streams; query
    blocks are folded under causal masking."""

    name = "bt_hkv_minor"
    tag = "bthkvmin"

    def emit_decode(self, b, spec, wi, seqlen_q):
        nqb = _baked_nqb(spec, seqlen_q)
        B, Hkv, gqa = spec.batch, spec.num_kv_heads, spec.num_queries_per_kv
        bt = b.mod(wi, b.const_i32(B))
        rem = b.div(wi, b.const_i32(B))
        hkv = b.mod(rem, b.const_i32(Hkv))
        r2 = b.div(rem, b.const_i32(Hkv))
        hql = b.mod(r2, b.const_i32(gqa))
        blk = b.div(r2, b.const_i32(gqa))
        hq = b.add(b.mul(hkv, b.const_i32(gqa)), hql)
        qb = emit_fold(b, blk, nqb) if spec.causal else blk
        return Decoded(qb, hq, bt, hkv)


class PersistHqMinorSwz(PersistDecode):
    """Swizzled Head-first on the persistent grid: XCD ``a`` (``wi % M``, needs
    ``M | NP``) owns head band ``a``. Under causal masking one CTA runs both
    blocks of a fold pair ``{p, NQB-1-p}`` on consecutive steps, so per-CTA cost
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

    def check(self, spec):
        M = spec.chiplet_num_xcds
        if spec.num_query_heads % M or spec.num_persistent % M:
            return (
                f"{self.name} needs num_query_heads and num_persistent divisible by {M}"
            )
        if spec.causal and _spec_nqb(spec) % 2:
            return (
                f"{self.name} needs an even number of query blocks under causal masking"
            )
        return None

    def emit_decode(self, b, spec, wi, seqlen_q):
        nqb = _baked_nqb(spec, seqlen_q)
        M, NP = spec.chiplet_num_xcds, spec.num_persistent
        hpm = spec.num_query_heads // M
        W = nqb * spec.num_query_heads * spec.batch
        if not spec.causal or W <= NP:
            a = b.mod(wi, b.const_i32(M))
            rest = b.div(wi, b.const_i32(M))
            qb = b.mod(rest, b.const_i32(nqb))  # at most one item per CTA when causal
            hb = b.div(rest, b.const_i32(nqb))
        else:
            unit, half = emit_pair_unit(b, wi, W, NP)
            a = b.mod(unit, b.const_i32(M))
            rest = b.div(unit, b.const_i32(M))
            p = b.mod(rest, b.const_i32(nqb // 2))
            hb = b.div(rest, b.const_i32(nqb // 2))
            qb = emit_pair_block(b, p, half, nqb)
        hq = b.add(b.mul(a, b.const_i32(hpm)), b.mod(hb, b.const_i32(hpm)))
        bt = b.div(hb, b.const_i32(hpm))
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
        if nqb % 2 or spec.num_queries_per_kv % 2:
            return f"{self.name} requires even NQB and even GQA ratio"
        if spec.num_persistent != expected_np:
            return (
                f"{self.name} requires num_persistent == NQB*Hkv*B "
                f"({expected_np}), got {spec.num_persistent}"
            )
        return None

    def emit_decode(self, b, spec, wi, seqlen_q):
        nqb = _baked_nqb(spec, seqlen_q)
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
        if nqb % 2 or gqa < 2:
            return f"{self.name} requires even NQB and GQA ratio >= 2"
        if spec.num_persistent != expected_np:
            return (
                f"{self.name} requires num_persistent == W/2 "
                f"({expected_np}), got {spec.num_persistent}"
            )
        return None

    def emit_decode(self, b, spec, wi, seqlen_q):
        nqb = _baked_nqb(spec, seqlen_q)
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
