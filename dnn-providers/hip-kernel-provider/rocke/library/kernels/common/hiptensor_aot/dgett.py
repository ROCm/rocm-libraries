# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Shape-generic direct GETT: a tensor contraction in one launch, with no workspace.

``D = alpha * sum_K opA(A) opB(B) + beta * opC(C)`` where A, B, C and D are read and written
through their own strides: no packed copy of an operand, no intermediate result. hipTensor's
rocKE backend uses it for every contraction its other routes cannot serve within the plan's
workspace limit (a limit of zero included), on every target with MFMA (gfx90a, gfx942, gfx950).

Index spaces (the host's plan, :func:`.host.dgett_plan`): the M modes (A and D), the N modes
(B and D), the K modes (A and B) and the batch modes (all four) of the contraction, each a list
of mixed-radix digits ``(extent, stride per tensor)``, innermost first, at most
:data:`SLOTS_MN` / :data:`SLOTS_KT` / :data:`SLOTS_L` digits (unused slots: extent 1). The K
space is split at the K tile ``tk``: ``k = kt * tk + kl`` with the in-tile digits ``kl`` (at most
:data:`SLOTS_KL`, their product ``tk``) and the tile digits ``kt``, so the K offset of an element
is a per-lane, loop-invariant part plus one scalar per K tile. When the K extents do not split
at ``tk`` the host pads one in-tile digit (``kps`` / ``kpl``: digit ``kps`` must stay below
``kpl``) or the innermost K mode (``krl``: ``kl + tk * kt0 < krl`` with ``kt0`` the first tile
digit); padded K indices read zeros.

Per CTA (4 waves, 2 x 2): one ``tm x tn`` output tile. Each K tile of A and B is read with one
load per element (loads of consecutive lanes run along the operand's fastest dimension:
``a_fast`` / ``b_fast`` = ``"k"``, or ``"r"`` for the M / N rows), converted to the compute type
and written to a double-buffered LDS stage (``[rows][tk]``, K fastest), one barrier per K tile;
the MFMAs of the tile read their fragments from it while the next tile's loads are in flight.

Data types are kernel arguments (``dtA, dtB, dtC, dtD``: 0 f16, 1 bf16, 2 f32, 3 f64; complex
kernels: 2 complex f32, 3 complex f64), selected by wave-uniform branches. ``comp`` is the MFMA
operand type:

* ``f16`` / ``bf16``: operands f16, bf16 or f32, the operator applied in f32 and the result
  rounded once (to nearest even) to ``comp``; products exact, accumulated in f32 (hipTensor's
  16-bit compute types, and 32F contractions of f16 x f16 or bf16 x bf16 without operators);
* ``f32``: operands f16, bf16 or f32 as f32 (32F; gfx942, gfx950);
* ``f64``: operands of any type as f64, the operator applied in f64; ``rnd`` = 1 / 2 rounds the
  result once to f16 / bf16 (a 16-bit compute type with an f64 operand), 0 keeps it. Products
  of f64 values accumulated in f64.

Complex kernels (``cplx``): ``comp`` f32 (complex f32 data, 4M on the f32 atom) or f64 (complex
f32 or f64 data); operators IDENTITY and CONJ.

Epilogue in f64 (complex: in complex f64), one rounding to D's type: ``alpha * acc`` (not
formed when ``alpha == 0``: a NaN or Inf in A or B then does not reach D), plus ``beta *
opC(C)`` when ``beta != 0`` (C is not read otherwise; C may be D itself). A complex scale with a
zero imaginary part multiplies both components by its real part. Rows and columns past the
tensor are not stored.

Kernel ABI ``hiptensor-aot-dgett/v1`` (argument names): ``A, B, C, D`` (pointers), ``alpha,
beta`` (f64; complex ``alpha_re, alpha_im, beta_re, beta_im``), ``dtA, dtB, dtC, dtD, opA, opB,
opC, rnd`` (i32), ``M, N, nk, tiles_n, tn_m, tn_s`` (i32), the byte spans ``a_bytes, b_bytes,
c_bytes, d_bytes`` (i32; 32-bit-offset kernels), the digits ``me<i>, mm<i>, ms<i>, mA<i>,
mC<i>, mD<i>`` (M), ``ne, nm, ns, nB, nC, nD`` (N), ``kle, klm, kls, klA, klB`` (in-tile K),
``kte, ktm, kts, ktA, ktB`` (K tiles), ``le, lm, ls, lA, lB, lC, lD`` (batch), ``kps, kpl,
krl`` (i32). ``m``/``s`` are magic pairs (:func:`.common.magic_u31`) of every digit but the
last of its space; strides are elements (complex elements), i32 or i64 (``idx64``). Launch:
block 256, grid ``(tiles_m * tiles_n, 1, L)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from rocke.core.ir import (
    BF16,
    F16,
    F32,
    F64,
    I8,
    I32,
    I64,
    IRBuilder,
    KernelDef,
    PtrType,
    Value,
    VectorType,
)

from .common import DTYPES, OP_CONJ, OP_IDENTITY, Emit, require_bool

DGETT_ABI = "hiptensor-aot-dgett/v1"

#: Digit slots of the M and N spaces, the K tiles, the in-tile K digits and the batch space.
SLOTS_MN = 8
SLOTS_KT = 8
SLOTS_KL = 6
SLOTS_L = 4

#: Data type codes of the ``dt*`` arguments (complex kernels: 2 and 3 are complex f32 / f64).
DT_CODES = {"f16": 0, "bf16": 1, "f32": 2, "f64": 3}
DT_NAMES = {v: k for k, v in DT_CODES.items()}

#: ``rnd`` codes of the f64 kernels.
RND_NONE = 0
RND_F16 = 1
RND_BF16 = 2

COMPS = ("f16", "bf16", "f32", "f64")
THREADS = 256
#: byte offset of a lane that loads or stores nothing (past every buffer range)
_SENTINEL = -(1 << 31)
_COMP_TYPE = {"f16": F16, "bf16": BF16, "f32": F32, "f64": F64}


@dataclass(frozen=True)
class DgettSpec:
    """One kernel of the family (see the module docstring)."""

    comp: str  # MFMA operand type: f16, bf16, f32 or f64
    cplx: bool = False
    a_fast: str = (
        "k"  # loads of A: consecutive lanes along "k" or along the M rows ("r")
    )
    b_fast: str = "k"  # loads of B: along "k" or the N rows ("r")
    math: bool = False  # the operators of common.MATH_OPS too
    idx64: bool = False  # 64-bit offsets (tensors past 2^31 bytes)

    def check(self, arch: str) -> None:
        require_bool(self, "cplx", "math", "idx64")
        if self.comp not in COMPS:
            raise ValueError(f"comp {self.comp!r} not in {COMPS}")
        if self.a_fast not in ("k", "r") or self.b_fast not in ("k", "r"):
            raise ValueError("a_fast / b_fast must be 'k' or 'r'")
        if self.cplx and self.comp not in ("f32", "f64"):
            raise ValueError("complex kernels compute in f32 or f64")
        if self.cplx and self.math:
            raise ValueError("complex kernels have no math operators")
        if dgett_atom(self, arch) is None:
            raise ValueError(f"{arch} has no MFMA atom for comp {self.comp}")

    def kernel_name(self) -> str:
        return (
            f"hiptensor_aot_dgett_{'c' if self.cplx else 'r'}_{self.comp}"
            f"_a{self.a_fast}_b{self.b_fast}_{'i64' if self.idx64 else 'i32'}"
            + ("_m" if self.math else "")
        )

    def load_types(self) -> Tuple[str, ...]:
        """The operand data types the kernel reads (``dtA`` / ``dtB`` values it accepts)."""
        if self.cplx:
            return ("f32",) if self.comp == "f32" else ("f32", "f64")
        if self.comp == "f64":
            return ("f16", "bf16", "f32", "f64")
        return ("f16", "bf16", "f32")


def dgett_atom(spec: DgettSpec, arch: str):
    """The 16x16 MFMA atom of the kernel on ``arch`` (None: none)."""
    from rocke.core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except (KeyError, ValueError):
        return None
    dt = {"f16": "fp16", "bf16": "bf16", "f32": "fp32", "f64": "fp64"}[spec.comp]
    cdt = "fp64" if spec.comp == "f64" else "fp32"
    ks = (32, 16) if spec.comp in ("f16", "bf16") else (4,)
    for k in ks:
        try:
            op = target.mma.op_for_shape(
                family="mma", a_dtype=dt, b_dtype=dt, c_dtype=cdt, m=16, n=16, k=k
            )
        except (KeyError, ValueError):
            op = None
        if op is not None and op.wave_size == 64:
            return op
    return None


@dataclass(frozen=True)
class DgettConfig:
    tm: int
    tn: int
    tk: int
    wave_m: int
    wave_n: int
    vk: int  # consecutive K elements per lane of an "k" load
    pad: int  # LDS row padding, elements


def dgett_config(spec: DgettSpec) -> DgettConfig:
    """Tile shape of a kernel (the host plans the K split with ``tk``)."""
    if spec.cplx:
        if spec.comp == "f32":
            return DgettConfig(64, 64, 16, 32, 32, 1, 1)
        return DgettConfig(64, 64, 8, 32, 32, 1, 1)
    if spec.comp in ("f16", "bf16"):
        return DgettConfig(128, 128, 32, 64, 64, 4, 8)
    if spec.comp == "f32":
        return DgettConfig(128, 128, 16, 64, 64, 2, 2)
    return DgettConfig(64, 64, 16, 32, 32, 1, 1)


def dgett_arg_names(spec: DgettSpec) -> List[Tuple[str, str]]:
    """``(name, type)`` of every argument after the pointers and scalars, in order."""
    st = "i64" if spec.idx64 else "i32"
    out = [(n, "i32") for n in ("dtA", "dtB", "dtC", "dtD", "opA", "opB", "opC", "rnd")]
    out += [(n, "i32") for n in ("M", "N", "nk", "tiles_n", "tn_m", "tn_s")]
    out += [(n, "i32") for n in ("a_bytes", "b_bytes", "c_bytes", "d_bytes")]

    def space(p, n, tensors):
        for i in range(n):
            out.append((f"{p}e{i}", "i32"))
            if i < n - 1:
                out.append((f"{p}m{i}", "i32"))
                out.append((f"{p}s{i}", "i32"))
            for t in tensors:
                out.append((f"{p}{t}{i}", st))

    space("m", SLOTS_MN, "ACD")
    space("n", SLOTS_MN, "BCD")
    space("kl", SLOTS_KL, "AB")
    space("kt", SLOTS_KT, "AB")
    space("l", SLOTS_L, "ABCD")
    out += [("kps", "i32"), ("kpl", "i32"), ("krl", "i32")]
    return out


class _E(Emit):
    """:class:`.common.Emit` whose compute type can change (operators of f32 and f64 values in
    one kernel share one Emit: loop names stay unique)."""

    def compute(self, cdt: str) -> None:
        self.cdt = cdt
        self.cty = F64 if cdt == "f64" else F32


def _space(b: IRBuilder, prefix: str, n: int, tensors: str, st_type):
    """Declares the arguments of a digit space; returns (extents, magic pairs, strides)."""
    ext, mg = [], []
    st: Dict[str, List[Value]] = {t: [] for t in tensors}
    for i in range(n):
        ext.append(b.param(f"{prefix}e{i}", I32))
        if i < n - 1:
            mg.append((b.param(f"{prefix}m{i}", I32), b.param(f"{prefix}s{i}", I32)))
        for t in tensors:
            st[t].append(b.param(f"{prefix}{t}{i}", st_type))
    return ext, mg, st


def _chain(e: Emit, x: Value, ext, mg, strides: Sequence[Sequence[Value]]):
    """Offsets of coordinate ``x`` of a digit space, one per stride list (and the digit values):
    ``sum_i digit_i(x) * stride_i``; the last digit takes the rest of ``x``."""
    b = e.b
    offs = [None] * len(strides)
    digits = []
    rest = x
    n = len(ext)
    for i in range(n):
        if i < n - 1:
            q, d = e.divmod(rest, ext[i], *mg[i])
        else:
            q, d = None, rest
        digits.append(d)
        for j, st in enumerate(strides):
            t = e.off(d, st[i])
            offs[j] = t if offs[j] is None else b.add(offs[j], t)
        rest = q
    return offs, digits


def build_dgett(spec: DgettSpec, *, arch: str) -> KernelDef:
    """Builds the kernel of ``spec`` for ``arch``."""
    spec.check(arch)
    cfg = dgett_config(spec)
    atom = dgett_atom(spec, arch)
    cx = spec.cplx
    ncomp = 2 if cx else 1
    tk, tm, tn = cfg.tk, cfg.tm, cfg.tn
    am_n, an_n = cfg.wave_m // atom.m, cfg.wave_n // atom.n
    ctype = _COMP_TYPE[spec.comp]
    acc_f64 = spec.comp == "f64"
    st_type = I64 if spec.idx64 else I32

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = THREADS
    e = _E(b, spec.idx64, "f64", spec.math)

    gptr = PtrType(I8, "global")
    P = {
        "A": b.param("A", gptr, readonly=True, align=2),
        "B": b.param("B", gptr, readonly=True, align=2),
        # C may be D itself (in-place update): no noalias / readonly on either
        "C": b.param("C", gptr, align=2),
        "D": b.param("D", gptr, align=2),
    }
    if cx:
        alpha = (b.param("alpha_re", F64), b.param("alpha_im", F64))
        beta = (b.param("beta_re", F64), b.param("beta_im", F64))
    else:
        alpha = (b.param("alpha", F64),)
        beta = (b.param("beta", F64),)
    arg = {}
    for name in ("dtA", "dtB", "dtC", "dtD", "opA", "opB", "opC", "rnd"):
        arg[name] = b.param(name, I32)
    for name in ("M", "N", "nk", "tiles_n", "tn_m", "tn_s"):
        arg[name] = b.param(name, I32)
    for name in ("a_bytes", "b_bytes", "c_bytes", "d_bytes"):
        arg[name] = b.param(name, I32)
    m_sp = _space(b, "m", SLOTS_MN, "ACD", st_type)
    n_sp = _space(b, "n", SLOTS_MN, "BCD", st_type)
    kl_sp = _space(b, "kl", SLOTS_KL, "AB", st_type)
    kt_sp = _space(b, "kt", SLOTS_KT, "AB", st_type)
    l_sp = _space(b, "l", SLOTS_L, "ABCD", st_type)
    kps = b.param("kps", I32)
    kpl = b.param("kpl", I32)
    krl = b.param("krl", I32)

    c0 = b.const_i32(0)
    tid = b.thread_id_x()
    lane = b.mod(tid, b.const_i32(64))
    wave = b.readfirstlane(b.div(tid, b.const_i32(64)))
    wm = b.mod(wave, b.const_i32(2))
    wn = b.div(wave, b.const_i32(2))
    bid = b.block_id_x()
    tile_m = e.udiv(bid, arg["tn_m"], arg["tn_s"])
    tile_n = b.sub(bid, b.mul(tile_m, arg["tiles_n"]))
    m0 = b.mul(tile_m, b.const_i32(tm))
    n0 = b.mul(tile_n, b.const_i32(tn))

    # batch: every tensor's base moved by its batch offset (64-bit)
    loffs, _ = _chain(
        e,
        b.block_id_z(),
        l_sp[0],
        l_sp[1],
        [l_sp[2][t] for t in "ABCD"],
    )
    base = {}
    for t, off in zip("ABCD", loffs):
        off64 = off if spec.idx64 else b.sext(off, I64)
        base[t] = off64  # element offset; scaled by the element size where it is used

    # ------------------------------------------------------------------ operand loaders
    lds = {}
    for nm, rows in (("A", tm), ("B", tn)):
        lds[nm] = b.smem_alloc(
            ctype, [2 * ncomp * rows, tk + cfg.pad], name_hint=f"{nm.lower()}_lds"
        )

    def lane_map(fast: str, rows: int):
        """Per element of a lane's share of a K tile: (pass, row, k) with python ints for the
        pass / vector index; returns (rows list, ks list, elements [(ri, ki)])."""
        if fast == "k":
            tpr = tk // cfg.vk
            rpp = THREADS // tpr
            passes = rows // rpp
            r0 = b.div(tid, b.const_i32(tpr))
            k0 = b.mul(b.mod(tid, b.const_i32(tpr)), b.const_i32(cfg.vk))
            rs = [b.add(r0, b.const_i32(p * rpp)) if p else r0 for p in range(passes)]
            ks = [b.add(k0, b.const_i32(i)) if i else k0 for i in range(cfg.vk)]
            return rs, ks, [(p, i) for p in range(passes) for i in range(cfg.vk)]
        kpp = THREADS // rows
        passes = tk // kpp
        r0 = b.mod(tid, b.const_i32(rows))
        k0 = b.div(tid, b.const_i32(rows))
        ks = [b.add(k0, b.const_i32(p * kpp)) if p else k0 for p in range(passes)]
        return [r0], ks, [(0, p) for p in range(passes)]

    class Operand:
        def __init__(self, nm, fast, rows, row0, lim, space, strides):
            self.nm = nm
            self.rows = rows
            self.dt = arg["dt" + nm]
            self.op = arg["op" + nm]
            self.bytes = arg["a_bytes" if nm == "A" else "b_bytes"]
            rs, ks, self.elems = lane_map(fast, rows)
            self.ks = ks
            self.fast = fast
            # rows of this output tile: offsets and validity (loop invariant)
            self.row_ok, self.row_off = [], []
            ext, mg = space[0], space[1]
            for r in rs:
                g = b.add(row0, r)
                ok = b.cmp_lt(g, lim)
                gs = b.select(ok, g, c0)
                (off,), _ = _chain(e, gs, ext, mg, [strides])
                self.row_ok.append(ok)
                self.row_off.append(off)
            self.rloc = rs
            # in-tile K offsets and the padded-digit test (loop invariant)
            self.k_off, self.k_ok = [], []
            kext, kmg, kst = kl_sp
            for k in ks:
                (off,), digits = _chain(e, k, kext, kmg, [kst[nm]])
                ok = None
                for i, d in enumerate(digits):
                    t = b.lor(b.cmp_ne(kps, b.const_i32(i)), b.cmp_lt(d, kpl))
                    ok = t if ok is None else b.land(ok, t)
                self.k_off.append(off)
                self.k_ok.append(ok)
            self.ptr = P[nm]

        def load(self, koff: Value, kmax: Value):
            """The lane's elements of the K tile whose K offset is ``koff`` (elements) and
            whose in-tile limit is ``kmax``: per element its components in the operator type
            (zero where the element is padding or past the tensor), and the validity of each.
            """
            ok = []
            off = []
            for ri, ki in self.elems:
                v = b.land(
                    b.land(self.row_ok[ri], self.k_ok[ki]), b.cmp_lt(self.ks[ki], kmax)
                )
                ok.append(v)
                off.append(b.add(self.row_off[ri], self.k_off[ki]))
            n = len(self.elems)
            optype = "f64" if (spec.comp == "f64") else "f32"
            e.compute(optype)
            zero = e.cf(0.0)
            vals = [zero] * (n * ncomp)
            for dt in spec.load_types():
                code = DT_CODES[dt]
                cond = b.cmp_eq(self.dt, b.const_i32(code))
                vals = e.when(
                    cond,
                    vals,
                    lambda _cur, dt=dt: self._raw(dt, koff, off, ok, optype),
                )
            return [vals[i * ncomp : (i + 1) * ncomp] for i in range(n)], ok

        def _raw(self, dt, koff, off, ok, optype) -> List[Value]:
            ty, nb = DTYPES[dt]
            es = nb * ncomp
            out = []
            if not spec.idx64:
                rsrc = b.buffer_rsrc(
                    b.global_ptr_add(self.ptr, b.mul(base[self.nm], b.const_i64(es))),
                    self.bytes,
                )
                soff = b.mul(koff, b.const_i32(es))
            for o, v in zip(off, ok):
                if spec.idx64:
                    tot = b.add(b.add(o, koff), base[self.nm])
                    tot = b.select(v, tot, b.const_i64(0))
                    p = b.global_ptr_add(self.ptr, b.mul(tot, b.const_i64(es)))
                    if cx:
                        raw = b.global_load_vN(p, c0, ty, 2, align=nb)
                        comps = [b.vec_extract(raw, 0), b.vec_extract(raw, 1)]
                    else:
                        comps = [b.global_load(p, c0, ty, align=nb)]
                    comps = [b.select(v, x, _zero_of(b, ty)) for x in comps]
                else:
                    vo = b.select(v, b.mul(o, b.const_i32(es)), b.const_i32(_SENTINEL))
                    comps = _buffer_elem(b, rsrc, vo, soff, dt, ncomp)
                for x in comps:
                    out.append(_to(e, x, dt, optype))
            return out

        def finish(self, loaded) -> List[List[Value]]:
            """Operator, zero padding and the rounding to the MFMA type."""
            vals, valid = loaded
            optype = "f64" if spec.comp == "f64" else "f32"
            e.compute(optype)
            zero = e.cf(0.0)
            if cx:
                conj = b.cmp_eq(self.op, b.const_i32(OP_CONJ))
                out = []
                for (re, im), v in zip(vals, valid):
                    im = b.select(conj, b.fneg(im), im)
                    out.append([b.select(v, re, zero), b.select(v, im, zero)])
            else:
                flat = [x[0] for x in vals]
                flat = e.unary_list(flat, self.op)
                out = [[b.select(v, x, zero)] for x, v in zip(flat, valid)]
            return [[_to_comp(e, x, spec, arg["rnd"]) for x in c] for c in out]

        def store(self, vals: List[List[Value]], stage: Value) -> None:
            rows = self.rows
            row_base = b.mul(stage, b.const_i32(rows))
            for c in range(ncomp):
                plane = b.add(row_base, b.const_i32(c * 2 * rows)) if c else row_base
                if self.fast == "k" and cfg.vk > 1:
                    for ri in range(len(self.rloc)):
                        chunk = [vals[ri * cfg.vk + i][c] for i in range(cfg.vk)]
                        b.smem_store_vN(
                            lds[self.nm],
                            [b.add(plane, self.rloc[ri]), self.ks[0]],
                            b.vec_pack(chunk, ctype),
                            cfg.vk,
                        )
                else:
                    for j, (ri, ki) in enumerate(self.elems):
                        b.smem_store_vN(
                            lds[self.nm],
                            [b.add(plane, self.rloc[ri]), self.ks[ki]],
                            vals[j][c],
                            1,
                        )

    opa = Operand("A", spec.a_fast, tm, m0, arg["M"], m_sp, m_sp[2]["A"])
    opb = Operand("B", spec.b_fast, tn, n0, arg["N"], n_sp, n_sp[2]["B"])

    def ktile(kt: Value):
        """(K offset of A, of B, in-tile limit) of K tile ``kt`` (wave-uniform)."""
        (oa, ob), digits = _chain(
            e, kt, kt_sp[0], kt_sp[1], [kt_sp[2]["A"], kt_sp[2]["B"]]
        )
        kmax = b.sub(krl, b.mul(digits[0], b.const_i32(tk)))
        return oa, ob, kmax

    def load_tile(kt: Value, live: Value):
        """The loads of K tile ``kt``; every element zero unless ``live``."""
        oa, ob, kmax = ktile(kt)
        kmax = b.select(live, kmax, c0)
        return opa.load(oa, kmax), opb.load(ob, kmax)

    def store_tile(regs, stage: Value):
        ra, rb = regs
        opa.store(opa.finish(ra), stage)
        opb.store(opb.finish(rb), stage)

    # fragment coordinates (atom lane maps)
    a_row, a_k = atom.a_layout().coord(b, lane, 0)
    b_k, b_col = atom.b_layout().coord(b, lane, 0)
    a_row0 = b.add(b.mul(wm, b.const_i32(cfg.wave_m)), a_row)
    b_row0 = b.add(b.mul(wn, b.const_i32(cfg.wave_n)), b_col)
    kpl_a = atom.a_frag_len
    kpl_b = atom.b_frag_len

    def frag(nm, stage, row, k, comp_i, n):
        rows = tm if nm == "A" else tn
        r = b.add(
            b.add(b.mul(stage, b.const_i32(rows)), row), b.const_i32(comp_i * 2 * rows)
        )
        v = b.smem_load_vN(lds[nm], r, k, dtype=ctype, n=n)
        return v if n > 1 else b.vec_extract(v, 0)

    acc_ty = F64 if acc_f64 else F32
    zero_acc = b.vec_pack(
        [b.const_f64(0.0) if acc_f64 else b.const_f32(0.0)] * atom.c_frag_len, acc_ty
    )
    n_acc = am_n * an_n * ncomp

    def mfma_tile(accs: List[Value], stage: Value) -> List[Value]:
        accs = list(accs)
        for ks in range(tk // atom.k):
            kk_a = b.add(a_k, b.const_i32(ks * atom.k)) if ks else a_k
            kk_b = b.add(b_k, b.const_i32(ks * atom.k)) if ks else b_k
            fa = []
            for am in range(am_n):
                row = b.add(a_row0, b.const_i32(am * atom.m)) if am else a_row0
                fa.append([frag("A", stage, row, kk_a, c, kpl_a) for c in range(ncomp)])
            fb = []
            for an in range(an_n):
                row = b.add(b_row0, b.const_i32(an * atom.n)) if an else b_row0
                fb.append([frag("B", stage, row, kk_b, c, kpl_b) for c in range(ncomp)])
            neg_ai = [b.fneg(x[1]) for x in fa] if cx else None
            for am in range(am_n):
                for an in range(an_n):
                    i = am * an_n + an
                    if not cx:
                        accs[i] = b.mma(atom, fa[am][0], fb[an][0], accs[i])
                        continue
                    ar, ai = fa[am]
                    br, bi = fb[an]
                    r_, i_ = accs[2 * i], accs[2 * i + 1]
                    r_ = b.mma(atom, ar, br, r_)
                    r_ = b.mma(atom, neg_ai[am], bi, r_)
                    i_ = b.mma(atom, ar, bi, i_)
                    i_ = b.mma(atom, ai, br, i_)
                    accs[2 * i], accs[2 * i + 1] = r_, i_
        return accs

    # ------------------------------------------------------------------ K loop
    # Trip kt loads K tile kt into registers (none past the last), runs the MFMAs of tile kt - 1
    # from LDS stage (kt - 1) & 1 (stage 1 starts zero: trip 0 adds nothing) and stores tile kt
    # into stage kt & 1: one copy of the load, MFMA and store code, one barrier per trip.
    nk = arg["nk"]
    zeros = [[_zero_of(b, ctype)] * ncomp for _ in opa.elems]
    opa.store(zeros, b.const_i32(1))
    opb.store([[_zero_of(b, ctype)] * ncomp for _ in opb.elems], b.const_i32(1))
    b.sync()
    loop = b.scf_for_iter(
        c0,
        b.add(nk, b.const_i32(1)),
        b.const_i32(1),
        [(f"dg_acc{i}", zero_acc) for i in range(n_acc)],
        iv_name="dg_kt",
        elide_trailing_barrier=False,
    )
    with loop as (kt, carried):
        accs = list(carried)
        stage = b.land(kt, b.const_i32(1))
        nxt = load_tile(kt, b.cmp_lt(kt, nk))
        accs = mfma_tile(accs, b.xor(stage, b.const_i32(1)))
        store_tile(nxt, stage)
        b.sync()
        b.scf_yield(*accs)
    accs = list(loop.results)

    # ------------------------------------------------------------------ epilogue (f64)
    e.compute("f64")
    zero = b.const_f64(0.0)
    c_rows, col_l = [], None
    for slot in range(atom.c_frag_len):
        r, c = atom.c_layout().coord(b, lane, slot)
        c_rows.append(r)
        col_l = c if col_l is None else col_l
    mw = b.add(m0, b.mul(wm, b.const_i32(cfg.wave_m)))
    nw = b.add(n0, b.mul(wn, b.const_i32(cfg.wave_n)))
    cols = []
    for an in range(an_n):
        n = b.add(nw, b.add(col_l, b.const_i32(an * atom.n)) if an else col_l)
        ok = b.cmp_lt(n, arg["N"])
        (dn, cn), _ = _chain(
            e, b.select(ok, n, c0), n_sp[0], n_sp[1], [n_sp[2]["D"], n_sp[2]["C"]]
        )
        cols.append((ok, dn, cn))
    # alpha * acc is not formed when alpha == 0; beta * opC(C) only when beta != 0 (C is read
    # only then)
    a_zero = b.fcmp("oeq", alpha[0], zero)
    ai_zero = bi_zero = None
    if cx:
        ai_zero = b.fcmp("oeq", alpha[1], zero)
        a_zero = b.land(a_zero, ai_zero)
    use_c = b.lnot(b.fcmp("oeq", beta[0], zero))
    if cx:
        bi_zero = b.fcmp("oeq", beta[1], zero)
        use_c = b.lor(use_c, b.lnot(bi_zero))
    out_types = spec.load_types() if cx else ("f16", "bf16", "f32", "f64")

    def scaled(s, vals, s_im_zero):
        """``s * vals``; a complex scale with a zero imaginary part multiplies each component
        by its real part (no 0 * Inf from the other component)."""
        if not cx:
            return [b.fmul(s[0], vals[0])]
        xr, xi = vals
        pr, pi = e.cmul(s[0], s[1], xr, xi)
        return [
            b.select(s_im_zero, b.fmul(s[0], xr), pr),
            b.select(s_im_zero, b.fmul(s[0], xi), pi),
        ]

    def out_rsrc(nm, span, es):
        return b.buffer_rsrc(
            b.global_ptr_add(P[nm], b.mul(base[nm], b.const_i64(es))), arg[span]
        )

    def load_c(elems):
        def one(dt):
            ty, nb = DTYPES[dt]
            es = nb * ncomp
            rsrc = None if spec.idx64 else out_rsrc("C", "c_bytes", es)
            out = []
            for ok, _do, coff, _v in elems:
                if spec.idx64:
                    tot = b.select(ok, b.add(coff, base["C"]), b.const_i64(0))
                    p = b.global_ptr_add(P["C"], b.mul(tot, b.const_i64(es)))
                    if cx:
                        raw = b.global_load_vN(p, c0, ty, 2, align=nb)
                        comps = [b.vec_extract(raw, 0), b.vec_extract(raw, 1)]
                    else:
                        comps = [b.global_load(p, c0, ty, align=nb)]
                    comps = [b.select(ok, x, _zero_of(b, ty)) for x in comps]
                else:
                    vo = b.select(
                        ok, b.mul(coff, b.const_i32(es)), b.const_i32(_SENTINEL)
                    )
                    comps = _buffer_elem(b, rsrc, vo, c0, dt, ncomp)
                out += [_to(e, x, dt, "f64") for x in comps]
            return out

        def branches(_cur):
            vals = [zero] * (len(elems) * ncomp)
            for dt in out_types:
                cond = b.cmp_eq(arg["dtC"], b.const_i32(DT_CODES[dt]))
                vals = e.when(cond, vals, lambda _c, dt=dt: one(dt))
            return vals

        return e.when(use_c, [zero] * (len(elems) * ncomp), branches)

    def store_d(elems, outs):
        for dt in out_types:
            with b.scf_if(b.cmp_eq(arg["dtD"], b.const_i32(DT_CODES[dt]))):
                ty, nb = DTYPES[dt]
                es = nb * ncomp
                rsrc = None if spec.idx64 else out_rsrc("D", "d_bytes", es)
                for (ok, doff, _co, _v), o in zip(elems, outs):
                    comps = [_from64(e, x, dt) for x in o]
                    if spec.idx64:
                        with b.scf_if(ok):
                            p = b.global_ptr_add(
                                P["D"], b.mul(b.add(doff, base["D"]), b.const_i64(es))
                            )
                            if cx:
                                b.global_store_vN(
                                    p, c0, b.vec_pack(comps, ty), 2, align=nb
                                )
                            else:
                                b.global_store(p, c0, comps[0], align=nb)
                    else:
                        vo = b.select(
                            ok, b.mul(doff, b.const_i32(es)), b.const_i32(_SENTINEL)
                        )
                        _buffer_store(b, rsrc, vo, dt, comps)

    # one row of atoms at a time: its C values are read before its D values are written (C may
    # be D itself), and the registers of one row only are live
    for am in range(am_n):
        elems = []  # (ok, d offset, c offset, accumulator values)
        for slot in range(atom.c_frag_len):
            r = c_rows[slot]
            m = b.add(mw, b.add(r, b.const_i32(am * atom.m)) if am else r)
            mok = b.cmp_lt(m, arg["M"])
            (dm, cm), _ = _chain(
                e, b.select(mok, m, c0), m_sp[0], m_sp[1], [m_sp[2]["D"], m_sp[2]["C"]]
            )
            for an in range(an_n):
                nok, dn, cn = cols[an]
                i = am * an_n + an
                vals = [b.vec_extract(accs[ncomp * i + c], slot) for c in range(ncomp)]
                if not acc_f64:
                    vals = [b.cvt_f32_to_f64(v) for v in vals]
                elems.append((b.land(mok, nok), b.add(dm, dn), b.add(cm, cn), vals))
        cvals = load_c(elems)
        if cx:
            conj_c = b.cmp_eq(arg["opC"], b.const_i32(OP_CONJ))
            cs = []
            for i in range(len(elems)):
                re, im = cvals[2 * i], cvals[2 * i + 1]
                cs.append([re, b.select(conj_c, b.fneg(im), im)])
        else:
            cs = [[v] for v in _unary_looped(e, cvals, arg["opC"])]
        outs = []
        for (_ok, _do, _co, vals), c in zip(elems, cs):
            t = [b.select(a_zero, zero, x) for x in scaled(alpha, vals, ai_zero)]
            q = scaled(beta, c, bi_zero)
            outs.append([b.select(use_c, b.fadd(x, y), x) for x, y in zip(t, q)])
        store_d(elems, outs)
    return b.kernel


# ---------------------------------------------------------------------------- helpers


def _unary_looped(e: Emit, vs: Sequence[Value], code: Value) -> List[Value]:
    """:meth:`.common.Emit.unary_list` with one copy of the operators for all of ``vs``: a loop
    over the values (each picked and put back with selects) whose trip count is opaque, entered
    only when the operator is not the identity. Smaller code than one copy per value, for the
    values of the epilogue."""
    b = e.b
    n = len(vs)
    if n == 1:
        return e.unary_list(vs, code)
    not_id = b.land(b.cmp_ne(code, e.c32(OP_IDENTITY)), b.cmp_ne(code, e.c32(OP_CONJ)))

    def body(cur):
        k = next(e._uid)
        opaque = b.inline_asm(
            "s_mov_b32 $0, 0", "=s", [], I32, sideeffect=True, result_name_hint="ozero"
        )
        loop = b.scf_for_iter(
            e.c32(0),
            b.add(e.c32(n), opaque),
            e.c32(1),
            [(f"u{k}_{i}", v) for i, v in enumerate(cur)],
            iv_name=f"ui{k}",
        )
        with loop as (it, vals):
            x = vals[0]
            for j in range(1, n):
                x = b.select(b.cmp_eq(it, e.c32(j)), vals[j], x)
            y = e.unary_list([x], code)[0]
            b.scf_yield(
                *[b.select(b.cmp_eq(it, e.c32(j)), y, vals[j]) for j in range(n)]
            )
        return list(loop.results)

    return e.when(not_id, list(vs), body)


def _zero_of(b: IRBuilder, ty) -> Value:
    if ty is F64:
        return b.const_f64(0.0)
    z = b.const_f32(0.0)
    return z if ty is F32 else b.cast_f32_to(z, ty)


def _buffer_elem(b, rsrc, voff, soff, dt, ncomp) -> List[Value]:
    """One element (complex: both components) of type ``dt`` at byte offset ``voff + soff``;
    out of range: zeros."""
    if dt == "f64":
        raw = b.buffer_load_vN(rsrc, voff, soff, I32, 2 * ncomp)
        v = b.vec_bitcast(raw, VectorType(F64, ncomp))
        return [b.vec_extract(v, i) for i in range(ncomp)]
    ty = DTYPES[dt][0]
    if ncomp == 2:
        v = b.buffer_load_vN(rsrc, voff, soff, ty, 2)
        return [b.vec_extract(v, 0), b.vec_extract(v, 1)]
    return [b.buffer_load(rsrc, voff, soff, ty)]


def _buffer_store(b, rsrc, voff, dt, comps) -> None:
    c0 = b.const_i32(0)
    if dt == "f64":
        v = b.vec_bitcast(b.vec_pack(comps, F64), VectorType(F32, 2 * len(comps)))
        b.buffer_store_vN_f32(rsrc, voff, c0, v, 2 * len(comps))
        return
    if dt == "f32":
        if len(comps) == 1:
            b.buffer_store_f32(rsrc, voff, c0, comps[0])
        else:
            b.buffer_store_vN_f32(rsrc, voff, c0, b.vec_pack(comps, F32), 2)
        return
    if dt == "f16":
        b.buffer_store_f16(rsrc, voff, c0, comps[0])
    else:
        b.buffer_store_bf16(rsrc, voff, c0, comps[0])


def _to(e: Emit, x: Value, dt: str, optype: str) -> Value:
    """A loaded value of type ``dt`` in the operator type (exact)."""
    b = e.b
    if optype == "f64":
        if dt == "f64":
            return x
        x32 = x if dt == "f32" else b.cast_to_f32(x)
        return b.cvt_f32_to_f64(x32)
    if dt == "f32":
        return x
    return b.cast_to_f32(x)


def _to_comp(e: Emit, x: Value, spec: DgettSpec, rnd: Value) -> Value:
    """An operator-type value as the MFMA operand type: one rounding to nearest even."""
    b = e.b
    if spec.comp == "f64":
        if spec.cplx:
            return x
        e.compute("f64")
        # rnd: one rounding to f16 / bf16 (round to odd to f32, then to nearest even)
        for code, ty in ((RND_F16, F16), (RND_BF16, BF16)):
            x = e.when(
                b.cmp_eq(rnd, b.const_i32(code)),
                [x],
                lambda c, ty=ty: [
                    b.cvt_f32_to_f64(
                        b.cast_to_f32(e.narrow(e.f64_to_f32_odd(c[0]), ty))
                    )
                ],
            )[0]
        return x
    if spec.comp == "f32":
        return x
    return e.narrow(x, _COMP_TYPE[spec.comp])


def _from64(e: Emit, x: Value, dt: str) -> Value:
    """An f64 value rounded once to ``dt``."""
    e.compute("f64")
    return e.from_c(x, dt)


def dgett_grid(tiles_m: int, tiles_n: int, batches: int) -> Tuple[int, int, int]:
    return (tiles_m * tiles_n, 1, batches)
