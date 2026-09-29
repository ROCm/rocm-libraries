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
        hq = b.add(b.mul(b.mod(by, b.const_i32(Hkv)), b.const_i32(gqa)),
                   b.div(by, b.const_i32(Hkv)))
        qb = emit_reverse(b, bz, spec, seqlen_q) if spec.causal else bz
        return Decoded(qb, hq, bx)


NONPERSIST_DECODES = _registry(NonpersistQbMinor, NonpersistBtHkvMinor)


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
    grid-stride spreads cheap and expensive blocks over each CTA. ``interleave``
    reverses the block on odd ``qb*Hq + hq``."""

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
        else:
            qb = qb0
        return Decoded(qb, hq, bt)


class PersistHkvMajor(PersistDecode):
    """``wi = hkv*(NQB*gqa*B) + blk*(gqa*B) + hql*B + bt``, folded: each
    grid-stride phase stays within about one kv head, so its K/V stays in L2
    across the GQA group."""

    name = "hkv_major"
    tag = "hkvmaj"

    def emit_decode(self, b, spec, wi, seqlen_q):
        nqb = _baked_nqb(spec, seqlen_q)
        B, gqa = spec.batch, spec.num_queries_per_kv
        bt = b.mod(wi, b.const_i32(B))
        rem = b.div(wi, b.const_i32(B))  # hkv*(NQB*gqa) + blk*gqa + hql
        hql = b.mod(rem, b.const_i32(gqa))
        r2 = b.div(rem, b.const_i32(gqa))  # hkv*NQB + blk
        blk = b.mod(r2, b.const_i32(nqb))
        hkv = b.div(r2, b.const_i32(nqb))
        hq = b.add(b.mul(hkv, b.const_i32(gqa)), hql)
        return Decoded(emit_fold(b, blk, nqb), hq, bt, hkv)


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
        why = _aligned_causal_error(spec, "gqa_pair")
        if why:
            return why
        if nqb % 2 or spec.num_queries_per_kv % 2:
            return "gqa_pair requires even NQB and even GQA ratio"
        if spec.num_persistent != expected_np:
            return ("gqa_pair requires num_persistent == NQB*Hkv*B "
                    f"({expected_np}), got {spec.num_persistent}")
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
    (qb pair, hkv, bt), and phase 0/1 selects the complementary blocks."""

    name = "gqa_pair_2phase"
    tag = "gqapair2"

    def check(self, spec):
        gqa = spec.num_queries_per_kv
        nqb = _spec_nqb(spec)
        expected_np = nqb * spec.num_kv_heads * spec.batch * gqa // 2
        why = _aligned_causal_error(spec, "gqa_pair_2phase")
        if why:
            return why
        if nqb % 2 or gqa < 2:
            return "gqa_pair_2phase requires even NQB and GQA ratio >= 2"
        if spec.num_persistent != expected_np:
            return ("gqa_pair_2phase requires num_persistent == W/2 "
                    f"({expected_np}), got {spec.num_persistent}")
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
    PersistQbMajor, PersistHkvMajor, PersistGqaPair, PersistGqaPair2Phase
)
