# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CTA work decode of the KV-parallel attention backward main kernels.

Family glue for the backward kernels (Python-only, no C++ twin). Everything
here emits CTA-uniform integer IR through the builder it is handed and only
calls integer methods of ``IRBuilder``, so the duck-typed host evaluator of the
tests can execute it.

Grid ``(gx, units * g_split, B)`` of the main kernel; one CTA decodes, in this
fixed order:

1. the linear id ``L = x + gx * (y + gy * z)``; without an XCD remap (a
   kernel built with ``xcd_remap = False``, or runtime ``xcd_n = 0``) the
   decode below reads ``(x, y, z)`` directly. With the remap
   (:class:`XcdRemap`) the plane id ``P_id = x + gx * y`` of batch ``z`` goes
   through the chunked chiplet transform of ``helpers/grid.py`` with runtime
   ``num_xcds = xcd_n`` and ``chunk_size = xcd_chunk`` over ``num_wgs = gx *
   gy`` (:func:`xcd_remap_id`), and ``(x, y)`` are re-read from the result
   (``x = P' % gx``, ``y = P' / gx``). The transform runs per batch plane
   because the main kernarg ABI carries no batch count; it equals the
   whole-grid transform whenever ``xcd_n * xcd_chunk`` divides ``gx * gy``,
   and is a bijection of each plane for every grid;
2. the kv-tile index ``t = x``, the head unit and split of ``y`` (``unit = y /
   g_split``, ``split = y % g_split``; ``unit = y`` when the kernel has no
   split) and the batch ``z``;
3. the load-balance order on ``t`` (runtime ``lb_order`` / ``pair``):

   * natural (0): one kv tile ``kt = t``;
   * reverse (1): one kv tile ``kt = n_kv_tiles - 1 - t``;
   * mirror pairing (``pair = 1``, ``gx = ceil(n_kv_tiles / 2)``): the tiles
     ``t`` and ``n_kv_tiles - 1 - t``, once when they are equal. ``pair = 1``
     takes precedence over ``lb_order``;
   * interleave (3; only in a decode emitted with ``interleave=True``): one kv
     tile ``kt = t / 2`` for even ``t`` and ``n_kv_tiles - 1 - t / 2`` for odd
     ``t``, so the tiles run ``0, n-1, 1, n-2, ...`` (a heavy causal tile next
     to a light one). A decode emitted without it reads 3 as natural, so the
     host plan sets the compile-time switch whenever it asks for interleave.

   Every order visits each kv tile of a (unit, batch) exactly once. The main
   kernel decodes with ``pairing=False`` (one kv tile per CTA: natural or
   reverse); the two-tile CTA of mirror pairing is not built (a CTA-uniform
   loop over two kv tiles keeps the staging and epilogue values live across
   the q loops and spills registers), so the host plan refuses it.

:func:`head_range` maps a (unit, split) to its query heads and kv heads:
direct mode (``h_k == h_v``, no split) ``h0 = unit * G``, ``G`` heads,
``hk = hv = unit``; atomic mode (``h_k != h_v`` or ``g_split > 1``, ``G =
gcd(h_q / h_k, h_q / h_v)``) ``G / g_split`` heads from ``unit * G + split *
G / g_split``, ``hk = unit * G / gk`` and ``hv = unit * G / gv`` (every query
head of a unit shares one K head and one V head, because ``G`` divides ``gk``
and ``gv``).
"""

from __future__ import annotations

from dataclasses import dataclass

from rocke.core.ir import IRBuilder, Value

__all__ = [
    "LB_INTERLEAVE",
    "LB_MIRROR",
    "LB_NATURAL",
    "LB_REVERSE",
    "CtaWork",
    "HeadRange",
    "XcdRemap",
    "decode_cta",
    "head_range",
    "main_grid_x",
    "xcd_remap_id",
]

LB_NATURAL = 0
LB_REVERSE = 1
LB_MIRROR = 2
LB_INTERLEAVE = 3


def main_grid_x(n_kv_tiles: int, pair: int) -> int:
    """Grid x of the main kernel: ``ceil(n / 2)`` under mirror pairing, else ``n``."""
    if n_kv_tiles < 0:
        raise ValueError("n_kv_tiles must be >= 0")
    return -(-n_kv_tiles // 2) if pair else n_kv_tiles


@dataclass(frozen=True, eq=False)
class CtaWork:
    """Decoded work of one CTA (every value CTA-uniform, i32).

    ``first_kt`` / ``second_kt`` are the kv tiles of pass 0 / pass 1;
    ``n_pass`` is 1, or 2 under mirror pairing when the two tiles differ.
    """

    t: Value
    unit: Value
    split: Value
    batch: Value
    first_kt: Value
    second_kt: Value
    n_pass: Value

    def kv_tile(self, b: IRBuilder, p: Value) -> Value:
        """kv tile of pass ``p`` (0 or 1)."""
        return b.select(b.cmp_eq(p, b.const_i32(0)), self.first_kt, self.second_kt)


@dataclass(frozen=True, eq=False)
class XcdRemap:
    """Runtime XCD remap of the main grid: ``n`` XCDs (``0`` = identity),
    ``chunk`` consecutive plane ids per XCD, ``gy`` the grid y extent."""

    n: Value
    chunk: Value
    gy: Value


def xcd_remap_id(
    b: IRBuilder, wgid: Value, *, num_wgs: Value, n: Value, chunk: Value
) -> Value:
    """``chiplet_transform_chunked_dynamic`` of ``helpers/grid.py`` with
    runtime ``num_xcds = n`` and ``chunk_size = chunk``; the identity when
    ``n <= 0`` or ``chunk <= 0``.

    Ids below the last full ``n * chunk`` block boundary are remapped so the
    i-th CTA dispatched to XCD ``wgid % n`` gets the i-th id of that XCD's
    chunk; the tail keeps its id. A bijection of ``[0, num_wgs)``.
    """
    zero, one = b.const_i32(0), b.const_i32(1)
    on = b.land(b.cmp_gt(n, zero), b.cmp_gt(chunk, zero))
    n1 = b.select(on, n, one)
    c1 = b.select(on, chunk, one)
    block = b.mul(n1, c1)
    limit = b.mul(b.div(num_wgs, block), block)
    xcd = b.mod(wgid, n1)
    local = b.div(wgid, n1)
    new = b.add(b.add(b.mul(b.div(local, c1), block), b.mul(xcd, c1)), b.mod(local, c1))
    return b.select(b.land(on, b.cmp_lt(wgid, limit)), new, wgid)


def decode_cta(
    b: IRBuilder,
    *,
    x: Value,
    y: Value,
    z: Value,
    n_kv_tiles: Value,
    lb_order: Value,
    pair: Value | None = None,
    g_split: Value | None = None,
    interleave: bool = False,
    xcd: XcdRemap | None = None,
) -> CtaWork:
    """CTA decode of the main grid (see the module docstring).

    ``pair = None``: no mirror pairing (one kv tile per CTA; ``second_kt`` is
    the first and ``n_pass`` is 1). ``g_split = None``: the kernel has no head
    split (``unit = y``, split 0). ``interleave``: also serve ``lb_order =
    LB_INTERLEAVE`` (without it the emitted decode is unchanged). ``xcd =
    None``: no XCD remap (no IR is emitted for it); with a remap the
    interleave applies to the remapped kv-tile index.
    """
    if xcd is not None:
        gx = n_kv_tiles
        if pair is not None:
            two_ = b.const_i32(2)
            half = b.div(b.add(n_kv_tiles, b.const_i32(1)), two_)
            gx = b.select(b.cmp_ne(pair, b.const_i32(0)), half, n_kv_tiles)
        gx1 = b.select(b.cmp_gt(gx, b.const_i32(0)), gx, b.const_i32(1))
        plane = xcd_remap_id(
            b, b.add(x, b.mul(gx1, y)), num_wgs=b.mul(gx1, xcd.gy), n=xcd.n,
            chunk=xcd.chunk,
        )  # fmt: skip
        x, y = b.mod(plane, gx1), b.div(plane, gx1)
    zero, one = b.const_i32(0), b.const_i32(1)
    if g_split is None:
        unit, split = y, zero
    else:
        unit, split = b.div(y, g_split), b.mod(y, g_split)
    mirror_t = b.sub(b.sub(n_kv_tiles, one), x)
    reverse = b.cmp_eq(lb_order, b.const_i32(LB_REVERSE))
    if interleave:
        c2 = b.const_i32(2)
        half = b.div(x, c2)
        odd = b.cmp_ne(b.mod(x, c2), zero)
        woven = b.select(odd, b.sub(b.sub(n_kv_tiles, one), half), half)
        inter = b.cmp_eq(lb_order, b.const_i32(LB_INTERLEAVE))
        # reverse and interleave are exclusive values of one kernarg
        x_or_woven = b.select(inter, woven, x)
    else:
        x_or_woven = x
    if pair is None:
        first = b.select(reverse, mirror_t, x_or_woven)
        return CtaWork(
            t=x, unit=unit, split=split, batch=z, first_kt=first, second_kt=first,
            n_pass=one,
        )  # fmt: skip
    paired = b.cmp_ne(pair, zero)
    first = b.select(b.land(reverse, b.lnot(paired)), mirror_t, x)
    if interleave:
        first = b.select(b.land(inter, b.lnot(paired)), woven, first)
    two = b.land(paired, b.cmp_ne(mirror_t, first))
    n_pass = b.select(two, b.const_i32(2), one)
    return CtaWork(
        t=x,
        unit=unit,
        split=split,
        batch=z,
        first_kt=first,
        second_kt=mirror_t,
        n_pass=n_pass,
    )


@dataclass(frozen=True, eq=False)
class HeadRange:
    """Query heads ``[h0, h0 + n_heads)`` of a CTA and its K / V heads."""

    h0: Value
    n_heads: Value
    hk: Value
    hv: Value


def head_range(
    b: IRBuilder,
    *,
    unit: Value,
    split: Value,
    group: Value,
    gk: Value,
    gv: Value,
    g_split: Value | None = None,
) -> HeadRange:
    """Query and kv heads of a (unit, split) (see the module docstring).

    ``g_split = None`` is direct mode: the unit is the kv head and owns its
    whole group (``group = h_q / h_k``). Otherwise atomic mode: ``group`` is
    ``gcd(h_q / h_k, h_q / h_v)`` and ``g_split`` divides it.
    """
    base = b.mul(unit, group)
    if g_split is None:
        return HeadRange(h0=base, n_heads=group, hk=unit, hv=unit)
    per = b.div(group, g_split)
    return HeadRange(
        h0=b.add(base, b.mul(split, per)),
        n_heads=per,
        hk=b.div(base, gk),
        hv=b.div(base, gv),
    )
