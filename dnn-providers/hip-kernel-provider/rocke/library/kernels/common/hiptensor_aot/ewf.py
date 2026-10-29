# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Flat elementwise kernel: the elementwise operation of :mod:`.ew` on tensors that are all
contiguous with the same layout (hipTensor permutations, elementwise binary and trinary
operations whose inputs and output are one contiguous run of elements in the same order).

``D = opABC(opAB(alpha * opA(A), beta * opB(B)), gamma * opC(C))`` element by element, with the
operator order, the compute type, the rounding and the zero-scale rule of :mod:`.ew` (a tensor
whose scale is zero is not read and contributes zero). No index arithmetic: element ``i`` of
every tensor is at offset ``i``. Each thread handles ``EWF_UNROLL`` vectors of ``vec`` elements
(16 bytes of D), ``block`` vectors apart, with vector loads and stores; with ``nt`` (gfx942 /
gfx950) they are streaming accesses (the data is read and written once). When every scale is
non-zero (a wave-uniform test), the loads of all vectors are issued without branches; otherwise
the loads of the inputs whose scale is zero are skipped.

hipTensor's rocKE backend runs this kernel (the pack's ``ewf`` classes: the rows whose tensors are
all f16 or bf16, where it measured faster than the vector class of :mod:`.jit_ew`, on gfx942 and
gfx950) when its canonical elementwise problem is one mode with stride 1 in every tensor; it builds no kernel in Python: the
C++ twin ``rocke_build_htaot_ewf`` (``rocke/instance_hiptensor_aot.h``) writes the pack's kernels,
kept byte-identical by the ``hiptensor_aot_ewf`` parity family.

Kernel ABI ``hiptensor-aot-ewf/v1``: ``A [B] [C] D``, scalars ``alpha [beta] [gamma]``, operator
codes ``opA [opB] [opC] [opAB] [opABC]`` (i32), ``n`` (i32): the number of vectors (elements /
``vec``). Launch: block ``(block, 1, 1)``, grid ``(ceil(n / (block * EWF_UNROLL)), 1, 1)``. The
caller guarantees that the element count is a multiple of ``vec``, below 2^31, and that every
pointer is aligned to ``vec`` elements of its type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from rocke.core.ir import I32, IRBuilder, KernelDef, TemporalHint

from .common import DTYPES, Emit, ptr
from .ew import EwSpec

EWF_ABI = "hiptensor-aot-ewf/v1"
#: Vectors per thread (``block`` apart).
EWF_UNROLL = 2
#: Targets whose streaming (``nt``) accesses the pack's classes use.
EWF_STREAMING_ARCHS = ("gfx942", "gfx950")

#: Widest vector access per element size (bytes -> elements) of the IR builder's vector ops.
_MAX_VEC = {2: 8, 4: 4, 8: 2}


@dataclass(frozen=True)
class EwfSpec:
    """A flat class: the real row of the generic kernel (data types of A, B, C, D), the block and
    the streaming flag."""

    arity: int
    a: str
    d: str
    b: Optional[str] = None
    c: Optional[str] = None
    block: int = 256
    nt: bool = False

    def base(self) -> EwSpec:
        return EwSpec(
            arity=self.arity, a=self.a, d=self.d, b=self.b, c=self.c, block=self.block
        )

    @property
    def vec(self) -> int:
        """Elements per vector: 16 bytes of D."""
        return 16 // DTYPES[self.d][1]

    def check(self) -> None:
        self.base().check()
        if self.block not in (64, 128, 256, 512, 1024):
            raise ValueError("block must be 64, 128, 256, 512 or 1024")

    def kernel_name(self) -> str:
        return (
            self.base()
            .kernel_name()
            .replace("hiptensor_aot_ew", "hiptensor_aot_ewf", 1)
            .replace("_i32_", "_", 1)
            + f"_v{self.vec}_x{EWF_UNROLL}"
            + ("_nt" if self.nt else "")
        )


def _runs(n: int, nbytes: int) -> List[int]:
    w = min(n, _MAX_VEC[nbytes])
    return [w] * (n // w)


def build_ewf(spec: EwfSpec) -> KernelDef:
    spec.check()
    base = spec.base()
    vec = spec.vec
    U = EWF_UNROLL
    hint = TemporalHint.STREAMING if spec.nt else TemporalHint.DEFAULT
    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.block
    e = Emit(b, False, base.cdt, False)
    names = base.tensors()
    ins = names[:-1]
    dts: Dict[str, Optional[str]] = {"A": spec.a, "B": spec.b, "C": spec.c, "D": spec.d}
    P = {n: ptr(b, n, dts[n], readonly=n != "D") for n in names}
    scal_of = {"A": "alpha", "B": "beta", "C": "gamma"}
    S = {n: e.scalar_param(scal_of[n]) for n in ins}
    OPS = {n: b.param("op" + n, I32) for n in ins}
    if spec.arity >= 2:
        OPS["AB"] = b.param("opAB", I32)
    if spec.arity == 3:
        OPS["ABC"] = b.param("opABC", I32)
    n_vec = b.param("n", I32)

    tid = b.thread_id_x()
    t0 = b.add(b.mul(b.block_id_x(), e.c32(spec.block * U)), tid)
    lanes = []
    for u in range(U):
        t = t0 if u == 0 else b.add(t0, e.c32(u * spec.block))
        ok = b.cmp_lt(t, n_vec)
        # past the end: read vector 0 (in bounds), store nothing
        ts = b.select(ok, t, e.c32(0))
        lanes.append((ok, b.mul(ts, e.c32(vec))))

    zero = e.cf(0.0)
    # a scale is non-zero when it compares unequal to 0 (a NaN scale counts as zero, as in .ew)
    nz = {n: b.fcmp("one", S[n], zero) for n in ins}
    allnz = nz[ins[0]]
    for n in ins[1:]:
        allnz = b.land(allnz, nz[n])

    def load_vec(n: str, off) -> List:
        dt = dts[n]
        ty, nb = DTYPES[dt]
        out = []
        done = 0
        for w in _runs(vec, nb):
            o = off if done == 0 else b.add(off, e.c32(done))
            p, idx = e._addr(P[n], o, nb)
            v = b.global_load_vN(p, idx, ty, w, align=w * nb, temporal_hint=hint)
            out += [e.to_c(b.vec_extract(v, j), dt) for j in range(w)]
            done += w
        return out

    def store_vec(off, vals: Sequence) -> None:
        dt = spec.d
        ty, nb = DTYPES[dt]
        outs = [e.from_c(v, dt) for v in vals]
        done = 0
        for w in _runs(vec, nb):
            o = off if done == 0 else b.add(off, e.c32(done))
            p, idx = e._addr(P["D"], o, nb)
            b.global_store_vN(
                p,
                idx,
                b.vec_pack(outs[done : done + w], ty),
                w,
                align=w * nb,
                temporal_hint=hint,
            )
            done += w

    def body(every_scale: bool) -> None:
        results = []
        for ok, off in lanes:
            vals = {}
            for n in ins:
                if every_scale:
                    v = load_vec(n, off)
                    vals[n] = [b.fmul(S[n], x) for x in e.unary_list(v, OPS[n])]
                else:
                    v = e.when(
                        nz[n], [zero] * vec, lambda _c, n=n, off=off: load_vec(n, off)
                    )
                    vals[n] = [
                        b.select(nz[n], b.fmul(S[n], x), zero)
                        for x in e.unary_list(v, OPS[n])
                    ]
            out = []
            for k in range(vec):
                if spec.arity == 1:
                    out.append(vals["A"][k])
                elif spec.arity == 2:
                    out.append(e.binary(vals["A"][k], vals["C"][k], OPS["AB"]))
                else:
                    ab = e.binary(vals["A"][k], vals["B"][k], OPS["AB"])
                    out.append(e.binary(ab, vals["C"][k], OPS["ABC"]))
            results.append((ok, off, out))
        for ok, off, out in results:
            with b.scf_if(ok):
                store_vec(off, out)

    # every scale non-zero: straight-line loads; otherwise skip the inputs of zero scale
    with b.scf_if(allnz):
        body(True)
    with b.scf_if(b.lnot(allnz)):
        body(False)
    return b.kernel


#: The data types of the rows that have a flat class: every tensor 16-bit (the rows where the
#: flat kernel measured faster than the vector class of the generic kernel).
EWF_DTYPES = ("f16", "bf16")


def ewf_classes(rows: Sequence[Dict], arch: str) -> List[EwfSpec]:
    """The flat classes of the pack's real 32-bit elementwise rows (``make_rocke_pack.ew_params``
    rows without math operators) whose tensors are all of :data:`EWF_DTYPES`: one per row,
    streaming, on :data:`EWF_STREAMING_ARCHS`. Empty on other targets (no measured benefit
    there)."""
    if arch not in EWF_STREAMING_ARCHS:
        return []
    out = []
    for p in rows:
        if p.get("cplx") or p.get("math") or p.get("idx64"):
            continue
        dts = [p.get(k) for k in ("a", "b", "c", "d") if p.get(k)]
        if any(x not in EWF_DTYPES for x in dts):
            continue
        out.append(
            EwfSpec(
                arity=p["arity"],
                a=p["a"],
                d=p["d"],
                b=p.get("b"),
                c=p.get("c"),
                nt=True,
            )
        )
    return out
