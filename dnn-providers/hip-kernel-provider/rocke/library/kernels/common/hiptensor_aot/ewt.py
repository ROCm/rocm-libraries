# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Shape-generic tiled elementwise kernel for transposing layouts (hipTensor permute and
elementwise binary / trinary whose inputs are fastest along another mode than the output).

``D = opABC(opAB(alpha * opA(A), beta * opB(B)), gamma * opC(C))`` with the operators, scales,
conversions and zero-scale rules of :mod:`.ew`, over D's index space split into two tiled modes
and a batch space:

* ``o``: D's stride-1 mode, ``t``: the stride-1 mode of the inputs that are transposed; each CTA
  computes one ``TO x TT`` tile of the ``(o, t)`` plane (``tile_o`` x ``tile_t``) for every batch
  index of its grid row;
* the batch space: every other mode, at most :data:`SLOTS_B` digits (extent, magic pair, a stride
  per tensor), one index per loop trip (wave-uniform offsets).

Each input has a kind (``kA``, ``kB``, ``kC``; wave-uniform branches): 2 (``T``) its stride along
``t`` is 1: the tile is read with 16-byte vector loads along ``t`` into an LDS tile ``[o][t]`` and
read back along ``o``; 1 (``V``) its stride along ``o`` is 1: vector loads along ``o``; 0 (``G``)
one load per element. D is written with 16-byte vector stores along ``o``. The host guarantees:
D's stride along ``o`` is 1 and its other strides are multiples of the vector width ``vo`` (16
bytes of D's type), ``eo`` is a multiple of ``vo``; a ``T`` input's other strides and ``et`` are
multiples of its vector width ``vt`` (16 bytes of its type), a ``V`` input's other strides of
``vo``; buffers aligned to the vectors (checked at execution: an unaligned buffer takes the
generic kernel).

Folded classes (``fold``): when the whole ``(o, t)`` plane is smaller than half a tile, one CTA
stacks ``fo x ft`` planes of consecutive batch indices in its tile (``fo = TO // eo`` along ``o``,
``ft = TT // et`` along ``t``): the batch offsets of its planes go to an LDS table first, then
every lane finds its plane and position with magic-number divisions by ``eo`` and ``et``. A
16 x 16 plane otherwise fills a sixteenth of the tile.

Kernel ABI ``hiptensor-aot-ewt/v2`` (v1: 6 batch digits, no folded classes): ``A [B] [C] D``, ``alpha [beta] [gamma]``, ``opA [opB]
[opC] [opAB] [opABC]``, ``kA [kB] [kC]``, ``eo, et, tiles_o, to_m, to_s, nb, ystep`` (i32),
``s<T>o, s<T>t`` and the batch digits ``be<i>, bm<i>, bs<i>, s<T><i>`` (i32 element offsets);
folded classes also ``fo, ft, eo_m, eo_s, et_m, et_s, nbt`` (``nb``: groups of ``fo * ft`` batch
indices, ``nbt``: batch indices). Launch: block 256, grid ``(tiles_o * tiles_t, min(nb, 65535),
1)`` (folded: ``(1, min(nb, 65535), 1)``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from rocke.core.ir import I32, IRBuilder, KernelDef, Value

from .common import DTYPES, Emit, max_vec, ptr, require_bool, vector_bytes

EWT_ABI = "hiptensor-aot-ewt/v2"
#: batch digits: every mode of a contraction's operand pack or epilogue but the tile plane (at most
#: 6 free and 6 contracted modes, or 6 + 6 free modes) in one launch
SLOTS_B = 10
THREADS = 256
TILE_O = 64
KIND_G, KIND_V, KIND_T = 0, 1, 2


@dataclass(frozen=True)
class EwtSpec:
    arity: int  # 1 (permute), 2 (binary), 3 (trinary)
    a: str
    d: str
    b: Optional[str] = None
    c: Optional[str] = None
    fold: bool = False  # several small planes per tile (see the module docstring)

    @property
    def cdt(self) -> str:
        dts = [x for x in (self.a, self.b, self.c, self.d) if x]
        return "f64" if "f64" in dts else "f32"

    def tensors(self) -> List[str]:
        return {1: ["A", "D"], 2: ["A", "C", "D"], 3: ["A", "B", "C", "D"]}[self.arity]

    def dtypes(self) -> Dict[str, str]:
        return {"A": self.a, "B": self.b, "C": self.c, "D": self.d}

    def tile_t(self) -> int:
        """The tile's extent along ``t`` (elements): 64, 32 with an 8-byte input."""
        ins = [self.dtypes()[n] for n in self.tensors()[:-1]]
        return 32 if any(DTYPES[x][1] == 8 for x in ins) else 64

    def vo(self) -> int:
        return max_vec(DTYPES[self.d][1])

    def vt(self, n: str) -> int:
        return max_vec(DTYPES[self.dtypes()[n]][1])

    def check(self) -> None:
        require_bool(self, "fold")
        if self.arity not in (1, 2, 3):
            raise ValueError("arity must be 1, 2 or 3")
        if self.arity >= 2 and not self.c:
            raise ValueError("arity >= 2 needs a C dtype")
        if self.arity == 3 and not self.b:
            raise ValueError("arity 3 needs a B dtype")
        for n in self.tensors():
            if self.dtypes()[n] not in DTYPES:
                raise ValueError(f"{n}: dtype {self.dtypes()[n]!r}")

    def kernel_name(self) -> str:
        dts = "_".join(x for x in (self.a, self.b, self.c, self.d) if x)
        return f"hiptensor_aot_ewt{self.arity}_{dts}" + ("_f" if self.fold else "")


def build_ewt(spec: EwtSpec, *, arch: str = "gfx950") -> KernelDef:
    """The tiled kernel of a class. ``arch`` is not used: the kernel is target-neutral IR (the pack builds a target's
    classes for it)."""
    spec.check()
    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = THREADS
    e = Emit(b, False, spec.cdt, False)
    names = spec.tensors()
    ins = names[:-1]
    dts = spec.dtypes()
    TT, TO = spec.tile_t(), TILE_O
    vo = spec.vo()
    P = {n: ptr(b, n, dts[n], readonly=n != "D") for n in names}
    scal_of = {"A": "alpha", "B": "beta", "C": "gamma"}
    S = {n: e.scalar_param(scal_of[n]) for n in ins}
    OPS = {n: b.param("op" + n, I32) for n in ins}
    if spec.arity >= 2:
        OPS["AB"] = b.param("opAB", I32)
    if spec.arity == 3:
        OPS["ABC"] = b.param("opABC", I32)
    KIND = {n: b.param("k" + n, I32) for n in ins}
    eo = b.param("eo", I32)
    et = b.param("et", I32)
    tiles_o = b.param("tiles_o", I32)
    to_m = b.param("to_m", I32)
    to_s = b.param("to_s", I32)
    nb = b.param("nb", I32)
    ystep = b.param("ystep", I32)
    so = {n: b.param(f"s{n}o", I32) for n in names}
    st = {n: b.param(f"s{n}t", I32) for n in names}
    bext, bmg = [], []
    bst: Dict[str, List[Value]] = {n: [] for n in names}
    for i in range(SLOTS_B):
        bext.append(b.param(f"be{i}", I32))
        if i < SLOTS_B - 1:
            bmg.append((b.param(f"bm{i}", I32), b.param(f"bs{i}", I32)))
        for n in names:
            bst[n].append(b.param(f"s{n}{i}", I32))
    if spec.fold:
        fo = b.param("fo", I32)
        ft = b.param("ft", I32)
        eo_ms = (b.param("eo_m", I32), b.param("eo_s", I32))
        et_ms = (b.param("et_m", I32), b.param("et_s", I32))
        nbt = b.param("nbt", I32)

    c0 = b.const_i32(0)
    tid = b.thread_id_x()
    bid = b.block_id_x()
    tile_t_idx = e.udiv(bid, to_m, to_s)
    tile_o_idx = b.sub(bid, b.mul(tile_t_idx, tiles_o))
    o0 = b.mul(tile_o_idx, e.c32(TO))
    t0 = b.mul(tile_t_idx, e.c32(TT))
    zero = e.cf(0.0)
    nz = {n: b.fcmp("one", S[n], zero) for n in ins}

    # LDS tiles of the inputs read along t: [o][t] in the input's type, rows padded by a vector
    lds = {}
    for n in ins:
        ty, nb_ = DTYPES[dts[n]]
        lds[n] = b.smem_alloc(ty, [TO, TT + max_vec(nb_)], name_hint=f"t{n}")

    # phase 1 lanes (per T input): vt consecutive t, rows of o
    def p1_lanes(n):
        vt = spec.vt(n)
        tpr = TT // vt
        rpp = THREADS // tpr
        tl = b.mul(b.mod(tid, e.c32(tpr)), e.c32(vt))
        r0 = b.div(tid, e.c32(tpr))
        return (
            vt,
            tl,
            [b.add(r0, e.c32(p * rpp)) if p else r0 for p in range(TO // rpp)],
        )

    # phase 2 lanes: vo consecutive o, rows of t
    cpr = TO // vo
    tpp = THREADS // cpr
    p2_o = b.mul(b.mod(tid, e.c32(cpr)), e.c32(vo))
    p2_t0 = b.div(tid, e.c32(cpr))
    p2_ts = [b.add(p2_t0, e.c32(q * tpp)) if q else p2_t0 for q in range(TT // tpp)]
    go = b.add(o0, p2_o)
    o_ok = b.cmp_lt(go, eo)  # whole vectors: eo is a multiple of vo
    if spec.fold:
        # the batch offsets of the tile's planes (column per tensor; D's -1: past the batches)
        btab = b.smem_alloc(I32, [THREADS, len(names)], name_hint="btab")
        nplanes = b.mul(fo, ft)

        def plane_of(o_l, t_l):
            """(plane index, in-plane o, in-plane t, in a plane) of tile position (o_l, t_l)."""
            jo, oo = e.divmod(o_l, eo, *eo_ms)
            jt, tt = e.divmod(t_l, et, *et_ms)
            inside = b.land(b.cmp_lt(jo, fo), b.cmp_lt(jt, ft))
            # a position outside every plane (eo or et does not divide the tile) reads table row
            # 0 instead of a row past the table (its value is dropped by ``inside``)
            pl = b.select(inside, b.add(jo, b.mul(jt, fo)), e.c32(0))
            return pl, oo, tt, inside

        def tab(plane, k):
            v = b.smem_load_vN(btab, plane, e.c32(k), dtype=I32, n=1)
            return b.vec_extract(v, 0)

    rows = b.scf_for(b.block_id_y(), nb, ystep, iv_name="bt")
    with rows as bi:
        # batch offsets (wave-uniform; folded: of the thread's plane, into the table)
        boff = {n: None for n in names}
        r = b.add(b.mul(bi, nplanes), tid) if spec.fold else bi
        for i in range(SLOTS_B):
            if i < SLOTS_B - 1:
                r, d = e.divmod(r, bext[i], *bmg[i])
            else:
                d = r
            for n in names:
                t_ = b.mul(d, bst[n][i])
                boff[n] = t_ if boff[n] is None else b.add(boff[n], t_)
        # every wave done with the previous batch's LDS tiles
        b.sync()
        if spec.fold:
            with b.scf_if(b.cmp_lt(tid, nplanes)):
                live_b = b.cmp_lt(b.add(b.mul(bi, nplanes), tid), nbt)
                for k, n in enumerate(names):
                    v = boff[n]
                    if n == "D":
                        v = b.select(live_b, v, e.c32(-1))
                    b.smem_store_vN(btab, [tid, e.c32(k)], v, 1)
            b.sync()
        # phase 1: the transposed inputs into LDS
        for n in ins:
            cond = b.land(b.cmp_eq(KIND[n], e.c32(KIND_T)), nz[n])
            with b.scf_if(cond):
                vt, tl, ors = p1_lanes(n)
                ty, nbytes = DTYPES[dts[n]]
                gt = b.add(t0, tl)
                t_ok = b.cmp_lt(gt, et)  # whole vectors: et is a multiple of vt
                for orow in ors:
                    if spec.fold:
                        pl, oo, tt, inside = plane_of(orow, tl)
                        k_ = names.index(n)
                        ok = b.land(inside, b.cmp_ge(tab(pl, len(names) - 1), c0))
                        off = b.add(
                            b.add(tab(pl, k_), b.mul(oo, so[n])), b.mul(tt, st[n])
                        )
                        off = b.select(ok, off, c0)
                    else:
                        g_o = b.add(o0, orow)
                        ok = b.land(t_ok, b.cmp_lt(g_o, eo))
                        off = b.add(b.add(boff[n], b.mul(g_o, so[n])), b.mul(gt, st[n]))
                        off = b.select(ok, off, c0)
                    v = b.global_load_vN(P[n], off, ty, vt, align=vector_bytes())
                    v = b.select(ok, v, b.vec_pack([_zero(b, ty)] * vt, ty))
                    b.smem_store_vN(lds[n], [orow, tl], v, vt)
        b.sync()
        # phase 2: per lane vo consecutive o of each of its t rows
        for t_local in p2_ts:
            if spec.fold:
                pl, oo2, tt2, inside = plane_of(p2_o, t_local)
                dbase = tab(pl, len(names) - 1)
                ok = b.land(inside, b.cmp_ge(dbase, c0))
            else:
                gt = b.add(t0, t_local)
                ok = b.land(o_ok, b.cmp_lt(gt, et))
            vals = {}
            for n in ins:
                ty, nbytes = DTYPES[dts[n]]
                if spec.fold:
                    base = b.add(
                        b.add(tab(pl, names.index(n)), b.mul(oo2, so[n])),
                        b.mul(tt2, st[n]),
                    )
                else:
                    base = b.add(b.add(boff[n], b.mul(go, so[n])), b.mul(gt, st[n]))
                base = b.select(ok, base, c0)

                def from_lds(_c, n=n):
                    out = []
                    for j in range(vo):
                        x = b.smem_load_vN(
                            lds[n],
                            b.add(p2_o, e.c32(j)) if j else p2_o,
                            t_local,
                            dtype=DTYPES[dts[n]][0],
                            n=1,
                        )
                        out.append(e.to_c(b.vec_extract(x, 0), dts[n]))
                    return out

                def vec_load(_c, n=n, base=base):
                    ty, nb_ = DTYPES[dts[n]]
                    out = []
                    done = 0
                    w = min(vo, max_vec(nb_))
                    while done < vo:
                        o_ = base if done == 0 else b.add(base, e.c32(done))
                        v = b.global_load_vN(P[n], o_, ty, w, align=w * nb_)
                        out += [e.to_c(b.vec_extract(v, k), dts[n]) for k in range(w)]
                        done += w
                    return out

                def gather(_c, n=n, base=base):
                    out = []
                    for j in range(vo):
                        o_ = base if j == 0 else b.add(base, b.mul(e.c32(j), so[n]))
                        out += e.load(P[n], o_, dts[n])
                    return out

                live = b.land(nz[n], ok)
                v = [zero] * vo
                v = e.when(b.land(live, b.cmp_eq(KIND[n], e.c32(KIND_T))), v, from_lds)
                v = e.when(b.land(live, b.cmp_eq(KIND[n], e.c32(KIND_V))), v, vec_load)
                v = e.when(b.land(live, b.cmp_eq(KIND[n], e.c32(KIND_G))), v, gather)
                # an input that is not read contributes exactly zero (its operator dropped)
                xs = e.unary_list(v, OPS[n])
                vals[n] = [b.select(nz[n], b.fmul(S[n], x), zero) for x in xs]
            out = []
            for k in range(vo):
                if spec.arity == 1:
                    out.append(vals["A"][k])
                elif spec.arity == 2:
                    out.append(e.binary(vals["A"][k], vals["C"][k], OPS["AB"]))
                else:
                    ab = e.binary(vals["A"][k], vals["B"][k], OPS["AB"])
                    out.append(e.binary(ab, vals["C"][k], OPS["ABC"]))
            with b.scf_if(ok):
                dty = DTYPES[spec.d][0]
                if spec.fold:
                    doff = b.add(b.add(dbase, b.mul(oo2, so["D"])), b.mul(tt2, st["D"]))
                else:
                    doff = b.add(
                        b.add(boff["D"], b.mul(go, so["D"])), b.mul(gt, st["D"])
                    )
                outs = [e.from_c(x, spec.d) for x in out]
                b.global_store_vN(
                    P["D"], doff, b.vec_pack(outs, dty), vo, align=vector_bytes()
                )
    return b.kernel


def _zero(b: IRBuilder, ty):
    from rocke.core.ir import F32, F64

    if ty is F64:
        return b.const_f64(0.0)
    z = b.const_f32(0.0)
    return z if ty is F32 else b.cast_f32_to(z, ty)


def ewt_fold(spec: EwtSpec, eo: int, et: int):
    """``(fo, ft)`` of a folded launch: the planes of an ``eo x et`` plane per tile, or None
    when the plane is not smaller than half a tile or the tile would hold more planes than the
    batch-offset table has rows (:data:`THREADS`, the pack attribute ``max_planes`` hipTensor's
    planner checks)."""
    if eo > TILE_O or et > spec.tile_t():
        return None
    fo, ft = TILE_O // eo, spec.tile_t() // et
    return (fo, ft) if 2 <= fo * ft <= THREADS else None


def ewt_plan(spec: EwtSpec, dims, limits=(1 << 31) - 1, transposed=True):
    """Arguments and grid of an :mod:`.ewt` launch (the rules hipTensor's C++ planner applies),
    or None when the layout is not one the kernel serves (no input fastest along another mode
    than D, D not fastest along a mode, extents or strides that do not keep the vectors aligned,
    more than :data:`SLOTS_B` batch modes).

    ``dims``: canonical modes ``(extent, [stride per tensor in spec.tensors() order])`` (extent-1
    modes dropped, contiguous modes merged). ``transposed`` False: also layouts without an input
    fastest along another mode (the tile's ``t`` is then D's next fastest mode; hipTensor uses
    them for contraction packs and epilogues of more modes than the generic kernel's slots).
    """
    from .common import as_i32_bits, magic_u31

    names = spec.tensors()
    nd = len(names) - 1
    o = next((i for i, (_e, s) in enumerate(dims) if s[nd] == 1), None)
    if o is None:
        return None
    t = None
    for k in range(nd):
        cand = next((i for i, (_e, s) in enumerate(dims) if s[k] == 1 and i != o), None)
        if cand is not None:
            t = cand
            break
    if t is None and not transposed:
        rest = [i for i in range(len(dims)) if i != o]
        t = min(rest, key=lambda i: dims[i][1][nd]) if rest else None
    if t is None:
        return None
    vo = spec.vo()
    eo, et = dims[o][0], dims[t][0]
    others = [i for i in range(len(dims)) if i not in (o, t)]
    if len(others) > SLOTS_B or eo % vo:
        return None
    if any(dims[i][1][nd] % vo for i in [t] + others):
        return None
    kinds = []
    for k, n in enumerate(names[:-1]):
        vt = spec.vt(n)
        st = lambda i: dims[i][1][k]  # noqa: E731
        if st(t) == 1 and et % vt == 0 and all(st(i) % vt == 0 for i in [o] + others):
            kinds.append(KIND_T)
            continue
        w = min(vo, vt)
        if st(o) == 1 and all(st(i) % w == 0 for i in [t] + others):
            kinds.append(KIND_V)
            continue
        kinds.append(KIND_G)
    if transposed and KIND_T not in kinds:
        return None
    for k in range(len(names)):
        span = sum((e - 1) * abs(s[k]) for e, s in dims)
        if span > limits:
            return None
    vals = {}
    for k, n in enumerate(names):
        vals[f"s{n}o"] = dims[o][1][k]
        vals[f"s{n}t"] = dims[t][1][k]
    for i in range(SLOTS_B):
        e, s = dims[others[i]] if i < len(others) else (1, [0] * len(names))
        vals[f"be{i}"] = e
        if i < SLOTS_B - 1:
            m, sh = magic_u31(e)
            vals[f"bm{i}"], vals[f"bs{i}"] = as_i32_bits(m), sh
        for k, n in enumerate(names):
            vals[f"s{n}{i}"] = s[k] if e != 1 else 0
    for k, n in enumerate(names[:-1]):
        vals["k" + n] = kinds[k]
    TT = spec.tile_t()
    tiles_o, tiles_t = -(-eo // TILE_O), -(-et // TT)
    nbatch = 1
    for i in others:
        nbatch *= dims[i][0]
    if spec.fold:
        f = ewt_fold(spec, eo, et)
        if f is None:
            return None
        fo, ft = f
        groups = -(-nbatch // (fo * ft))
        mo, so_ = magic_u31(eo)
        mt, st_ = magic_u31(et)
        m1, s1 = magic_u31(1)
        vals.update(
            eo=eo,
            et=et,
            tiles_o=1,
            to_m=as_i32_bits(m1),
            to_s=s1,
            nb=groups,
            ystep=min(groups, 65535),
            fo=fo,
            ft=ft,
            eo_m=as_i32_bits(mo),
            eo_s=so_,
            et_m=as_i32_bits(mt),
            et_s=st_,
            nbt=nbatch,
        )
        return vals, (1, min(groups, 65535), 1), kinds
    m, sh = magic_u31(tiles_o)
    vals.update(
        eo=eo,
        et=et,
        tiles_o=tiles_o,
        to_m=as_i32_bits(m),
        to_s=sh,
        nb=nbatch,
        ystep=min(nbatch, 65535),
    )
    return vals, (tiles_o * tiles_t, min(nbatch, 65535), 1), kinds


__all__ = [
    "EWT_ABI",
    "EwtSpec",
    "KIND_G",
    "KIND_T",
    "KIND_V",
    "SLOTS_B",
    "THREADS",
    "TILE_O",
    "build_ewt",
    "ewt_fold",
    "ewt_plan",
]
