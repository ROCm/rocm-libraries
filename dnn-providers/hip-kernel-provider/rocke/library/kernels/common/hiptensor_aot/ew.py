# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Shape-generic strided elementwise kernel (hipTensor permute, elementwise binary / trinary,
and the pack / epilogue steps of a contraction).

``D = opABC(opAB(alpha * opA(A), beta * opB(B)), gamma * opC(C))`` over D's index space, every
tensor read through its own strides (a stride of 0 broadcasts). Arity 1 is the permute /
pack form ``D = alpha * opA(A)``; arity 2 is ``D = opAC(alpha * opA(A), gamma * opC(C))``
(the operator argument ``opAB`` carries opAC). A tensor whose scaling factor is zero is not
read (and may be null).

Index space (rank bucket R8, after the host canonicalizes the problem: extent-1 modes dropped,
modes ordered by D's strides, contiguous modes merged): dims ``0..2`` form the thread space
(``n_in = e0 * e1 * e2`` threads along grid x, coordinates by magic division), dims ``3..7`` the
row space (``n_out`` rows; grid y rows, the block loops over rows ``y, y + ystep, ...``, the
coordinates of a row are wave-uniform). Arity 1 takes ``lim0..lim7``: a coordinate at or past
its limit reads zero instead of A (zero padding of a packed contraction operand).

Complex kernels (``cplx``, arity 1 or 2): elements are (re, im) pairs, unary operators
IDENTITY / CONJ, the scalars are complex (``alpha_re``, ``alpha_im``, ...) and arity 2 adds.

Kernel ABI ``hiptensor-aot-ew/v1`` (argument names; the order follows from the spec):
``A [B] [C] D``, scalars ``alpha [beta] [gamma]`` (``_re`` / ``_im`` for complex), operator
codes ``opA [opB] [opC] [opAB] [opABC]`` (i32), ``n_in, n_out, ystep`` (i32),
``e0..e7`` (i32), magic pairs ``m0, s0, m1, s1`` (thread space) and ``m3..m6, s3..s6`` (row
space), strides ``sA0..sA7``, ... ``sD0..sD7`` (i32 or i64), and for arity 1 ``lim0..lim7``.
Launch: block ``(block, 1, 1)``, grid ``(ceil(n_in / block), min(n_out, 65535), 1)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from rocke.core.ir import I32, IRBuilder, KernelDef

from .common import MAX_RANK, Emit, ptr

EW_ABI = "hiptensor-aot-ew/v1"


@dataclass(frozen=True)
class EwSpec:
    arity: int  # 1 (permute / pack), 2 (binary), 3 (trinary)
    a: str  # A dtype
    d: str  # D dtype
    b: Optional[str] = None  # B dtype (arity 3)
    c: Optional[str] = None  # C dtype (arity >= 2)
    cplx: bool = False
    idx64: bool = False
    block: int = 256

    @property
    def cdt(self) -> str:
        dts = [x for x in (self.a, self.b, self.c, self.d) if x]
        return "f64" if "f64" in dts else "f32"

    def tensors(self) -> List[str]:
        return {1: ["A", "D"], 2: ["A", "C", "D"], 3: ["A", "B", "C", "D"]}[self.arity]

    def check(self) -> None:
        if self.arity not in (1, 2, 3):
            raise ValueError("arity must be 1, 2 or 3")
        if self.arity >= 2 and not self.c:
            raise ValueError("arity >= 2 needs a C dtype")
        if self.arity == 3 and not self.b:
            raise ValueError("arity 3 needs a B dtype")
        if self.cplx and self.arity == 3:
            raise ValueError("complex kernels have arity 1 or 2")
        if self.cplx and any(
            x not in (None, "f32", "f64") for x in (self.a, self.c, self.d)
        ):
            raise ValueError("complex kernels move f32 or f64 component pairs")
        if self.block % 64:
            raise ValueError("block must be a multiple of 64")

    def kernel_name(self) -> str:
        dts = "_".join(x for x in (self.a, self.b, self.c, self.d) if x)
        return (
            f"hiptensor_aot_ew{self.arity}_{'c' if self.cplx else 'r'}_{dts}"
            f"_{'i64' if self.idx64 else 'i32'}_b{self.block}"
        )


def build_ew(spec: EwSpec) -> KernelDef:
    spec.check()
    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.block
    e = Emit(b, spec.idx64, spec.cdt)
    names = spec.tensors()
    dts = {"A": spec.a, "B": spec.b, "C": spec.c, "D": spec.d}
    P = {n: ptr(b, n, dts[n], readonly=n != "D") for n in names}
    scal_of = {"A": "alpha", "B": "beta", "C": "gamma"}
    S = {}
    for n in names[:-1]:
        if spec.cplx:
            S[n] = (
                e.scalar_param(scal_of[n] + "_re"),
                e.scalar_param(scal_of[n] + "_im"),
            )
        else:
            S[n] = (e.scalar_param(scal_of[n]),)
    OPS = {}
    for n in names[:-1]:
        OPS[n] = b.param("op" + n, I32)
    if spec.arity >= 2:
        OPS["AB"] = b.param("opAB", I32)
    if spec.arity == 3:
        OPS["ABC"] = b.param("opABC", I32)
    n_in = b.param("n_in", I32)
    n_out = b.param("n_out", I32)
    ystep = b.param("ystep", I32)
    ext = [b.param(f"e{i}", I32) for i in range(MAX_RANK)]
    mg = {}
    for i in (0, 1, 3, 4, 5, 6):
        mg[i] = (b.param(f"m{i}", I32), b.param(f"s{i}", I32))
    ST = {n: [e.stride_param(f"s{n}{i}") for i in range(MAX_RANK)] for n in names}
    lim = (
        [b.param(f"lim{i}", I32) for i in range(MAX_RANK)] if spec.arity == 1 else None
    )

    tid = b.thread_id_x()
    t = b.add(b.mul(b.block_id_x(), e.c32(spec.block)), tid)
    t_ok = b.cmp_lt(t, n_in)
    ts = b.select(t_ok, t, e.c32(0))
    t1, c0 = e.divmod(ts, ext[0], *mg[0])
    c2, c1 = e.divmod(t1, ext[1], *mg[1])
    inner = (c0, c1, c2)
    ioff = {}
    for n in names:
        o = e.off(c0, ST[n][0])
        o = b.add(o, e.off(c1, ST[n][1]))
        ioff[n] = b.add(o, e.off(c2, ST[n][2]))
    in_lim = None
    if lim is not None:
        in_lim = b.land(
            b.land(b.cmp_lt(c0, lim[0]), b.cmp_lt(c1, lim[1])), b.cmp_lt(c2, lim[2])
        )

    zero = e.cf(0.0)
    nz = {
        n: b.lor(b.fcmp("one", S[n][0], zero), b.fcmp("one", S[n][-1], zero)) for n in S
    }

    rows = b.scf_for(b.block_id_y(), n_out, ystep, iv_name="row")
    with rows as row:
        # row coordinates (wave-uniform)
        r = row
        coords = []
        for i in (3, 4, 5, 6):
            r, ci = e.divmod(r, ext[i], *mg[i])
            coords.append(ci)
        coords.append(r)  # dim 7
        roff = {}
        for n in names:
            o = ioff[n]
            for j, ci in enumerate(coords):
                o = b.add(o, e.off(ci, ST[n][3 + j]))
            roff[n] = o
        ok = t_ok
        if lim is not None:
            rl = in_lim
            for j, ci in enumerate(coords):
                rl = b.land(rl, b.cmp_lt(ci, lim[3 + j]))
            ok_load = rl
        with b.scf_if(ok):
            vals = {}
            ncomp = 2 if spec.cplx else 1
            for n in names[:-1]:
                zeros = [zero] * ncomp
                cond = nz[n]
                if lim is not None:
                    cond = b.land(cond, ok_load)
                # NB: the load is skipped (zeros) when the scale is zero or the coordinate is
                # past the limit; the branch is uniform only in the first case, which is fine
                v = e.when(
                    cond,
                    zeros,
                    lambda _c, n=n: e.load(P[n], roff[n], dts[n], spec.cplx),
                )
                if spec.cplx:
                    re_, im_ = v
                    im_ = b.select(b.cmp_eq(OPS[n], e.c32(9)), b.fneg(im_), im_)  # CONJ
                    v = list(e.cmul(S[n][0], S[n][1], re_, im_))
                    vals[n] = [b.select(cond, c, zero) for c in v]
                else:
                    # an input that is not read (zero scale, padding) contributes exactly
                    # zero: its operator is not applied (log(0) would give -inf)
                    x = e.unary(v[0], OPS[n])
                    vals[n] = [b.select(cond, b.fmul(S[n][0], x), zero)]
            if spec.cplx:
                out = vals["A"]
                if spec.arity == 2:
                    out = [b.fadd(out[0], vals["C"][0]), b.fadd(out[1], vals["C"][1])]
            elif spec.arity == 1:
                out = vals["A"]
            elif spec.arity == 2:
                out = [e.binary(vals["A"][0], vals["C"][0], OPS["AB"])]
            else:
                ab = e.binary(vals["A"][0], vals["B"][0], OPS["AB"])
                out = [e.binary(ab, vals["C"][0], OPS["ABC"])]
            e.store(P["D"], roff["D"], out, spec.d)
    return b.kernel
