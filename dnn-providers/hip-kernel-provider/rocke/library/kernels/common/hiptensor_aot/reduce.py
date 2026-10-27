# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Shape-generic strided reduction kernels (hipTensor reduction).

``D = alpha * reduce_opR(opA(A)) + beta * opC(C)`` with ``opR`` in ADD / MUL / MAX / MIN (MAX and
MIN propagate NaN), every tensor read through its own strides, accumulation in f32 (f64 for f64
data) and one rounding to D's dtype. With ``alpha == 0`` A is not read; with ``beta == 0`` C is
not read (either may then be null).

The host canonicalizes the problem: kept modes (those of D) and reduced modes, extent-1 modes
dropped, each set ordered by A's strides (innermost first) and contiguous modes merged, at most
8 of each (rank bucket R8). Two kernels:

* ``row`` (A's innermost mode is reduced): one wave per output, ``block / 64`` outputs per block
  (grid x). The lanes read 64 consecutive elements of the innermost reduced mode per trip; the
  trips (``nchunk`` chunks of that mode for each index of the other reduced modes) are split
  into ``split`` contiguous ranges of ``jchunk`` trips (grid y). The wave combines its lanes
  with XOR shuffles.
* ``col`` (A's innermost mode is kept): one thread per output of the innermost kept mode
  (grid x), outer kept index on grid y (rows ``y, y + ystep, ...``), the flat reduced index
  space split into ``split`` ranges of ``rchunk`` (grid z); every thread walks the same reduced
  indices (wave-uniform coordinates), four independent loads per loop trip.

``split == 1``: the kernel applies alpha / beta / C and writes D. ``split > 1``: it writes its
unrounded partial to ``W[s * n_out + o]`` (``o`` = flat output index over the canonical kept
modes) and :func:`build_reduce_finalize` combines the partials and writes D.

Kernel ABI ``hiptensor-aot-reduce/v1``: ``A, C, D, W``, ``alpha, beta``, ``opA, opC, opR``,
``n_out, split``, kept ``ke0..ke7``, ``km0..km6, ks0..ks6`` (magic pairs), strides
``kA0..kA7, kC0..kC7, kD0..kD7``; reduced ``re0..re7``, ``rm0..rm6, rs0..rs6``,
``rA0..rA7``; row: ``nchunk, jchunk, jm, js`` (magic of nchunk); col: ``n_red, rchunk,
ystep, n_outer``. Finalize: ``W, C, D, alpha, beta, opC, opR, n_out, split`` and the kept
geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from rocke.core.ir import I32, IRBuilder, KernelDef, PtrType, Value

from .common import MAX_RANK, OP_ADD, OP_MAX, OP_MIN, OP_MUL, Emit, ptr

REDUCE_ABI = "hiptensor-aot-reduce/v1"
WAVE = 64


@dataclass(frozen=True)
class ReduceSpec:
    mode: str  # "row" | "col" | "final"
    a: str  # A dtype (unused by "final")
    d: str  # C / D dtype
    idx64: bool = False
    block: int = 256
    math: bool = False  # the operators of common.MATH_OPS too

    @property
    def cdt(self) -> str:
        return "f64" if "f64" in (self.a, self.d) else "f32"

    def kernel_name(self) -> str:
        return (
            f"hiptensor_aot_reduce_{self.mode}_{self.a}_{self.d}"
            f"_{'i64' if self.idx64 else 'i32'}_b{self.block}"
            + ("_m" if self.math else "")
        )


def _identity(e: Emit, op: Value) -> Value:
    b = e.b
    inf = float("inf")
    r = e.cf(0.0)
    r = b.select(b.cmp_eq(op, e.c32(OP_MUL)), e.cf(1.0), r)
    r = b.select(b.cmp_eq(op, e.c32(OP_MAX)), e.cf(-inf), r)
    r = b.select(b.cmp_eq(op, e.c32(OP_MIN)), e.cf(inf), r)
    return r


class _Geo:
    """Kernel arguments of the kept / reduced geometry."""

    def __init__(self, b: IRBuilder, e: Emit, with_red: bool):
        self.ke = [b.param(f"ke{i}", I32) for i in range(MAX_RANK)]
        self.km = [
            (b.param(f"km{i}", I32), b.param(f"ks{i}", I32))
            for i in range(MAX_RANK - 1)
        ]
        self.kA = (
            [e.stride_param(f"kA{i}") for i in range(MAX_RANK)] if with_red else None
        )
        self.kC = [e.stride_param(f"kC{i}") for i in range(MAX_RANK)]
        self.kD = [e.stride_param(f"kD{i}") for i in range(MAX_RANK)]
        if with_red:
            self.re = [b.param(f"re{i}", I32) for i in range(MAX_RANK)]
            self.rm = [
                (b.param(f"rm{i}", I32), b.param(f"rs{i}", I32))
                for i in range(MAX_RANK - 1)
            ]
            self.rA = [e.stride_param(f"rA{i}") for i in range(MAX_RANK)]

    def decode(self, e: Emit, x: Value, ext, mg, first: int = 0) -> List[Value]:
        """Coordinates of flat index ``x`` over dims ``first..7`` (innermost first)."""
        out = []
        r = x
        for i in range(first, MAX_RANK - 1):
            r, c = e.divmod(r, ext[i], *mg[i])
            out.append(c)
        out.append(r)
        return out

    @staticmethod
    def dot(
        e: Emit, coords: Sequence[Value], strides: Sequence[Value], first: int = 0
    ) -> Value:
        b = e.b
        o = None
        for j, c in enumerate(coords):
            t = e.off(c, strides[first + j])
            o = t if o is None else b.add(o, t)
        return o


def _epilogue(e: Emit, acc: Value, alpha, beta, opC, C, coff, d: str) -> Value:
    """``alpha * acc + beta * opC(C)``; C is read only when beta != 0, acc only counts when
    alpha != 0."""
    b = e.b
    zero = e.cf(0.0)
    v = b.select(b.fcmp("one", alpha, zero), b.fmul(alpha, acc), zero)
    cv = e.when(
        b.fcmp("one", beta, zero),
        [zero],
        lambda _c: [e.unary(e.load(C, coff, d)[0], opC)],
    )
    return b.select(b.fcmp("one", beta, zero), b.fadd(v, b.fmul(beta, cv[0])), v)


def _common_params(b: IRBuilder, e: Emit, spec: ReduceSpec, with_a: bool):
    P = {}
    if with_a:
        P["A"] = ptr(b, "A", spec.a, readonly=True)
    P["C"] = ptr(b, "C", spec.d, readonly=True)
    P["D"] = ptr(b, "D", spec.d, readonly=False)
    P["W"] = b.param("W", PtrType(e.cty, "global"), align=8)
    P["alpha"] = e.scalar_param("alpha")
    P["beta"] = e.scalar_param("beta")
    if with_a:
        P["opA"] = b.param("opA", I32)
    P["opC"] = b.param("opC", I32)
    P["opR"] = b.param("opR", I32)
    P["n_out"] = b.param("n_out", I32)
    P["split"] = b.param("split", I32)
    return P


def _finish(
    e: Emit, P, geo: _Geo, acc: Value, o: Value, kcoords, sidx: Value, spec
) -> None:
    """Write the epilogue result (split == 1) or the partial (split > 1) of output ``o``."""
    b = e.b
    one = b.cmp_eq(P["split"], e.c32(1))
    with b.scf_if(one):
        doff = _Geo.dot(e, kcoords, geo.kD)
        coff = _Geo.dot(e, kcoords, geo.kC)
        v = _epilogue(e, acc, P["alpha"], P["beta"], P["opC"], P["C"], coff, spec.d)
        e.store(P["D"], doff, [v], spec.d)
    with b.scf_if(b.lnot(one)):
        widx = b.add(b.mul(sidx, P["n_out"]), o)
        b.global_store(P["W"], widx, acc, align=8 if e.cdt == "f64" else 4)


def build_reduce(spec: ReduceSpec) -> KernelDef:
    if spec.mode == "final":
        return build_reduce_finalize(spec)
    if spec.block % WAVE:
        raise ValueError("block must be a multiple of 64")
    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.block
    e = Emit(b, spec.idx64, spec.cdt, spec.math)
    P = _common_params(b, e, spec, True)
    geo = _Geo(b, e, True)
    if spec.mode == "row":
        nchunk = b.param("nchunk", I32)
        jchunk = b.param("jchunk", I32)
        jm = b.param("jm", I32)
        js = b.param("js", I32)
        _row(b, e, spec, P, geo, nchunk, jchunk, jm, js)
    elif spec.mode == "col":
        n_red = b.param("n_red", I32)
        rchunk = b.param("rchunk", I32)
        ystep = b.param("ystep", I32)
        n_outer = b.param("n_outer", I32)
        _col(b, e, spec, P, geo, n_red, rchunk, ystep, n_outer)
    else:
        raise ValueError(spec.mode)
    return b.kernel


def _row(b, e: Emit, spec, P, geo: _Geo, nchunk, jchunk, jm, js) -> None:
    tid = b.thread_id_x()
    lane = b.land(tid, e.c32(WAVE - 1))
    wave = b.readfirstlane(b.lshr(tid, e.c32(6)))
    o = b.add(b.mul(b.block_id_x(), e.c32(spec.block // WAVE)), wave)
    sidx = b.block_id_y()
    zero = e.cf(0.0)
    alpha_nz = b.fcmp("one", P["alpha"], zero)
    with b.scf_if(b.cmp_lt(o, P["n_out"])):
        kc = geo.decode(e, o, geo.ke, geo.km)
        abase = _Geo.dot(e, kc, geo.kA)
        ident = _identity(e, P["opR"])
        # trips [j0, j1) of this split: trip j = ro * nchunk + ch
        n_ro = _red_outer_count(b, e, geo)
        jtot = b.mul(nchunk, n_ro)
        j0 = b.mul(sidx, jchunk)
        j1 = b.smin(b.add(j0, jchunk), jtot)
        ro0, ch0 = e.divmod(b.smin(j0, jtot), nchunk, jm, js)

        def red(_c):
            rbase0 = _Geo.dot(
                e, geo.decode(e, ro0, geo.re, geo.rm, first=1), geo.rA, first=1
            )
            loop = b.scf_for_iter(
                j0,
                j1,
                e.c32(1),
                [("racc", ident), ("rch", ch0), ("rro", ro0), ("rbase", rbase0)],
                iv_name="rj",
            )
            with loop as (_j, cur):
                acc, ch, ro, rbase = cur
                i = b.add(b.mul(ch, e.c32(WAVE)), lane)
                ok = b.cmp_lt(i, geo.re[0])
                isafe = b.select(ok, i, e.c32(0))
                off = b.add(b.add(abase, rbase), e.off(isafe, geo.rA[0]))
                x = e.unary(e.load(P["A"], off, spec.a)[0], P["opA"])
                x = b.select(ok, x, ident)
                acc = e.binary(acc, x, P["opR"])
                ch = b.add(ch, e.c32(1))
                wrap = b.cmp_eq(ch, nchunk)
                ro2 = b.select(wrap, b.add(ro, e.c32(1)), ro)
                ch = b.select(wrap, e.c32(0), ch)

                def rebase(cc):
                    co = geo.decode(e, cc[0], geo.re, geo.rm, first=1)
                    return [_Geo.dot(e, co, geo.rA, first=1)]

                rbase = e.when(
                    b.land(wrap, b.cmp_lt(ro2, n_ro)), [rbase], lambda c: rebase([ro2])
                )[0]
                b.scf_yield(acc, ch, ro2, rbase)
            acc = loop.results[0]
            for m in (32, 16, 8, 4, 2, 1):
                acc = e.binary(acc, b.warp_shuffle_xor(acc, m), P["opR"])
            return [acc]

        acc = e.when(alpha_nz, [ident], red)[0]
        with b.scf_if(b.cmp_eq(lane, e.c32(0))):
            _finish(e, P, geo, acc, o, kc, sidx, spec)


def _red_outer_count(b, e: Emit, geo: _Geo) -> Value:
    n = geo.re[1]
    for i in range(2, MAX_RANK):
        n = b.mul(n, geo.re[i])
    return n


def _col(b, e: Emit, spec, P, geo: _Geo, n_red, rchunk, ystep, n_outer) -> None:
    tid = b.thread_id_x()
    k0 = b.add(b.mul(b.block_id_x(), e.c32(spec.block)), tid)
    k_ok = b.cmp_lt(k0, geo.ke[0])
    k0s = b.select(k_ok, k0, e.c32(0))
    sidx = b.block_id_z()
    zero = e.cf(0.0)
    alpha_nz = b.fcmp("one", P["alpha"], zero)
    ident = _identity(e, P["opR"])
    r0 = b.mul(sidx, rchunk)
    r1 = b.smin(b.add(r0, rchunk), n_red)
    U = 4
    rows = b.scf_for(b.block_id_y(), n_outer, ystep, iv_name="krow")
    with rows as ko:
        kc = [k0s] + geo.decode(e, ko, geo.ke, geo.km, first=1)
        abase = _Geo.dot(e, kc, geo.kA)

        def red(_c):
            n_it = b.div(b.add(b.sub(r1, r0), e.c32(U - 1)), e.c32(U))
            n_it = b.smax(n_it, e.c32(0))
            loop = b.scf_for_iter(
                e.c32(0),
                n_it,
                e.c32(1),
                [(f"cacc{u}", ident) for u in range(U)],
                iv_name="cj",
            )
            with loop as (j, cur):
                base = b.add(r0, b.mul(j, e.c32(U)))
                nxt = []
                if spec.math:
                    # the U loads first, then the operator once for all of them (one copy of
                    # the math programs in the loop)
                    oks, xs = [], []
                    for u in range(U):
                        r = b.add(base, e.c32(u))
                        ok = b.cmp_lt(r, r1)
                        rs = b.select(ok, r, e.c32(0))
                        co = geo.decode(e, rs, geo.re, geo.rm)
                        off = b.add(abase, _Geo.dot(e, co, geo.rA))
                        oks.append(ok)
                        xs.append(e.load(P["A"], off, spec.a)[0])
                    xs = e.unary_list(xs, P["opA"])
                    for u in range(U):
                        x = b.select(oks[u], xs[u], ident)
                        nxt.append(e.binary(cur[u], x, P["opR"]))
                    b.scf_yield(*nxt)
                else:
                    for u in range(U):
                        r = b.add(base, e.c32(u))
                        ok = b.cmp_lt(r, r1)
                        rs = b.select(ok, r, e.c32(0))
                        co = geo.decode(e, rs, geo.re, geo.rm)
                        off = b.add(abase, _Geo.dot(e, co, geo.rA))
                        x = e.unary(e.load(P["A"], off, spec.a)[0], P["opA"])
                        x = b.select(ok, x, ident)
                        nxt.append(e.binary(cur[u], x, P["opR"]))
                    b.scf_yield(*nxt)
            r = loop.results
            acc = e.binary(
                e.binary(r[0], r[1], P["opR"]), e.binary(r[2], r[3], P["opR"]), P["opR"]
            )
            return [acc]

        acc = e.when(alpha_nz, [ident], red)[0]
        with b.scf_if(k_ok):
            o = b.add(k0, b.mul(ko, geo.ke[0]))
            _finish(e, P, geo, acc, o, kc, sidx, spec)


def build_reduce_finalize(spec: ReduceSpec) -> KernelDef:
    """Combine ``split`` partials ``W[s * n_out + o]`` per output and write D."""
    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.block
    e = Emit(b, spec.idx64, spec.cdt, spec.math)
    P = _common_params(b, e, spec, False)
    geo = _Geo(b, e, False)
    o = b.add(b.mul(b.block_id_x(), e.c32(spec.block)), b.thread_id_x())
    zero = e.cf(0.0)
    with b.scf_if(b.cmp_lt(o, P["n_out"])):
        ident = _identity(e, P["opR"])
        loop = b.scf_for_iter(
            e.c32(0), P["split"], e.c32(1), [("facc", ident)], iv_name="fs"
        )
        with loop as (s, cur):
            x = b.global_load(
                P["W"],
                b.add(b.mul(s, P["n_out"]), o),
                e.cty,
                align=8 if e.cdt == "f64" else 4,
            )
            b.scf_yield(e.binary(cur[0], x, P["opR"]))
        acc = loop.results[0]
        kc = geo.decode(e, o, geo.ke, geo.km)
        doff = _Geo.dot(e, kc, geo.kD)
        coff = _Geo.dot(e, kc, geo.kC)
        v = _epilogue(e, acc, P["alpha"], P["beta"], P["opC"], P["C"], coff, spec.d)
        _ = zero
        e.store(P["D"], doff, [v], spec.d)
    return b.kernel
