# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Problem-specialized strided elementwise kernel: the JIT form of :mod:`.ew`.

hipTensor's rocKE backend serves elementwise problems (permute, elementwise binary / trinary, and
the pack / epilogue steps of its contractions) from the shape-generic kernel of :mod:`.ew`, which
reads the problem (operator codes, sizes, magic numbers, strides, padding limits) from kernel
arguments. With JIT enabled (``HIPTENSOR_JIT_MODE_DEFAULT``) it replaces such a launch by this
kernel, in which every one of those integers is a compile-time constant, so the compiler folds the
operator dispatch, the unused index slots and the stride arithmetic away; and, when the output's
fastest mode allows it, each thread handles ``vec`` consecutive elements of that mode with vector
loads and stores.

The kernel is the generic one partially evaluated: the integer geometry arguments of :mod:`.ew`
become ``arith.constant`` ops at the places the generic kernel reads them, and nothing else
changes for ``vec == 1``. So the kernel computes exactly what the generic kernel computes for the
same argument values (the numeric tests check both against float64 references).

hipTensor builds no kernel in Python: its C++ runtime builds the kernel of a problem with this
builder's C++ twin (``rocke_build_htaot_ew_jit``, ``rocke/instance_hiptensor_aot.h``; the
``hiptensor_aot_jit_ew`` parity family keeps the two byte-identical), lowers it and compiles it
with COMGR. :func:`roll_jit_ew` rolls a class (:class:`EwJitSpec`) over every geometry constant
into one parametric recipe of rocKE's recipe VM (``rocke.portable_ir``); hipTensor does not use
the recipes.

Geometry (the class's axes) has the argument names of :mod:`.ew`: ``opA [opB] [opC]
[opAB] [opABC]``, ``n_in, n_out, ystep``, ``e0..e7``, ``m0, s0, m1, s1, m3..m6, s3..s6``,
``s<T>0..s<T>7`` for every tensor ``T``, and for arity 1 ``lim0..lim7``. With ``vec > 1`` the
thread space counts vectors: ``n_in``, ``e0`` and its magic pair ``m0, s0`` are those of
``e0 / vec``, the other values are unchanged.

``vec > 1`` (real kernels; the ``math`` rows only with every input a vector, each input's math
programs applied in a loop over the ``vec`` values; their other layouts take the ``vec`` 1
class): D's stride of mode 0 is 1; a tensor marked in ``unit`` has stride 1 in
mode 0 and is accessed with vector loads, the others with ``vec`` scalar loads ``s<T>0`` apart.
The caller guarantees ``e0 % vec == 0``, ``lim0 % vec == 0`` (arity 1), that the strides of modes
``1..7`` of the vector tensors (and D) are multiples of ``vec`` and that their pointers are aligned
to ``vec`` elements: vector accesses are emitted with that alignment.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from rocke.core.ir import I32, I64, IRBuilder, KernelDef

from .common import (
    DTYPES,
    MAX_RANK,
    Emit,
    ptr,
)
from .ew import EwSpec

EW_JIT_ABI = "hiptensor-jit-ew/v1"
#: The ABI of the vector classes of the elementwise kernel (:func:`ewv_kernel_name`, the class
#: kernels of hipTensor's kernel pack).
EWV_ABI = "hiptensor-aot-ewv/v1"

#: Widest vector access per element size (bytes -> elements) of the IR builder's vector ops.
_MAX_VEC = {2: 8, 4: 4, 8: 2}


@dataclass(frozen=True)
class EwJitSpec:
    """A kernel class: the generic kernel's row (:class:`EwSpec`), the vector width and which
    inputs are read with vector loads (``unit``, in ``base.tensors()`` order without D).
    """

    base: EwSpec
    vec: int = 1
    unit: Tuple[bool, ...] = field(default_factory=tuple)

    def check(self) -> None:
        self.base.check()
        if self.vec not in (1, 2, 4, 8):
            raise ValueError("vec must be 1, 2, 4 or 8")
        if self.vec > 1 and self.base.cplx:
            raise ValueError("complex kernels have vec 1")
        if self.vec > 1 and self.base.math and not all(self.unit):
            raise ValueError(
                "math kernels have vector classes only with every input a vector"
            )
        n_in = len(self.base.tensors()) - 1
        if self.vec == 1 and self.unit:
            raise ValueError("unit flags need vec > 1")
        if self.vec > 1 and len(self.unit) != n_in:
            raise ValueError(f"unit needs {n_in} flags")

    def kernel_name(self) -> str:
        u = "".join("u" if x else "g" for x in self.unit)
        return (
            self.base.kernel_name().replace("hiptensor_aot_ew", "hiptensor_jit_ew", 1)
            + f"_v{self.vec}"
            + (f"_{u}" if u else "")
        )

    def key(self) -> str:
        """The key of the class (hipTensor's JIT names its kernel class by it)."""
        return "ew:" + self.kernel_name()

    def axes(self) -> List[Tuple[str, bool]]:
        """The geometry constants: ``(name, is_i64)`` in emission order."""
        names = self.base.tensors()
        out = [("op" + n, False) for n in names[:-1]]
        if self.base.arity >= 2:
            out.append(("opAB", False))
        if self.base.arity == 3:
            out.append(("opABC", False))
        out += [("n_in", False), ("n_out", False), ("ystep", False)]
        out += [(f"e{i}", False) for i in range(MAX_RANK)]
        for i in (0, 1, 3, 4, 5, 6):
            out += [(f"m{i}", False), (f"s{i}", False)]
        for n in names:
            out += [(f"s{n}{i}", self.base.idx64) for i in range(MAX_RANK)]
        if self.base.arity == 1:
            out += [(f"lim{i}", False) for i in range(MAX_RANK)]
        return out


def _runs(n: int, nbytes: int) -> List[int]:
    w = min(n, _MAX_VEC[nbytes])
    return [w] * (n // w)


def build_ew_jit(spec: EwJitSpec, geom: Optional[Dict[str, int]]) -> KernelDef:
    """The kernel of class ``spec`` for the geometry values ``geom`` (every name of
    :meth:`EwJitSpec.axes`). With ``geom`` None, the class kernel of hipTensor's kernel pack
    instead (:func:`ewv_kernel_name`): every geometry value is a kernel argument of that name,
    declared after the scalars in the order of :meth:`EwJitSpec.axes`."""
    spec.check()
    base = spec.base
    vec = spec.vec
    b = IRBuilder(spec.kernel_name() if geom is not None else ewv_kernel_name(spec))
    b.kernel.attrs["max_workgroup_size"] = base.block
    e = Emit(b, base.idx64, base.cdt, base.math)
    names = base.tensors()
    dts = {"A": base.a, "B": base.b, "C": base.c, "D": base.d}
    unit = dict(zip(names[:-1], spec.unit)) if vec > 1 else {}
    P = {n: ptr(b, n, dts[n], readonly=n != "D") for n in names}
    scal_of = {"A": "alpha", "B": "beta", "C": "gamma"}
    S = {}
    for n in names[:-1]:
        if base.cplx:
            S[n] = (
                e.scalar_param(scal_of[n] + "_re"),
                e.scalar_param(scal_of[n] + "_im"),
            )
        else:
            S[n] = (e.scalar_param(scal_of[n]),)

    pv = {}
    if geom is None:
        pv = {n: b.param(n, I64 if is64 else I32) for n, is64 in spec.axes()}

    def k32(name: str):
        if geom is None:
            return pv[name]
        return b.const_i32(int(geom[name]))

    def koff(v: int):
        return b.const_i64(v) if base.idx64 else b.const_i32(v)

    def kst(name: str):
        if geom is None:
            return pv[name]
        return (
            b.const_i64(int(geom[name])) if base.idx64 else b.const_i32(int(geom[name]))
        )

    OPS = {n: k32("op" + n) for n in names[:-1]}
    if base.arity >= 2:
        OPS["AB"] = k32("opAB")
    if base.arity == 3:
        OPS["ABC"] = k32("opABC")
    n_in = k32("n_in")
    n_out = k32("n_out")
    ystep = k32("ystep")
    ext = [k32(f"e{i}") for i in range(MAX_RANK)]
    mg = {i: (k32(f"m{i}"), k32(f"s{i}")) for i in (0, 1, 3, 4, 5, 6)}
    ST = {n: [kst(f"s{n}{i}") for i in range(MAX_RANK)] for n in names}
    lim = [k32(f"lim{i}") for i in range(MAX_RANK)] if base.arity == 1 else None

    tid = b.thread_id_x()
    t = b.add(b.mul(b.block_id_x(), e.c32(base.block)), tid)
    t_ok = b.cmp_lt(t, n_in)
    ts = b.select(t_ok, t, e.c32(0))
    t1, c0 = e.divmod(ts, ext[0], *mg[0])
    if vec > 1:
        c0 = b.mul(c0, e.c32(vec))
    c2, c1 = e.divmod(t1, ext[1], *mg[1])
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

    def load_vec(n: str, off) -> List:
        """``vec`` elements of tensor ``n`` along mode 0 from element offset ``off``, in the
        compute type."""
        dt = dts[n]
        ty, nb = DTYPES[dt]
        out = []
        if unit.get(n):
            done = 0
            for w in _runs(vec, nb):
                o = off if done == 0 else b.add(off, koff(done))
                p, idx = e._addr(P[n], o, nb)
                v = b.global_load_vN(p, idx, ty, w, align=w * nb)
                out += [e.to_c(b.vec_extract(v, j), dt) for j in range(w)]
                done += w
            return out
        for k in range(vec):
            o = off if k == 0 else b.add(off, e.off(e.c32(k), ST[n][0]))
            out += e.load(P[n], o, dt)
        return out

    def store_vec(off, vals: Sequence) -> None:
        dt = base.d
        ty, nb = DTYPES[dt]
        outs = [e.from_c(v, dt) for v in vals]
        done = 0
        for w in _runs(vec, nb):
            o = off if done == 0 else b.add(off, koff(done))
            p, idx = e._addr(P["D"], o, nb)
            b.global_store_vN(
                p, idx, b.vec_pack(outs[done : done + w], ty), w, align=w * nb
            )
            done += w

    rows = b.scf_for(b.block_id_y(), n_out, ystep, iv_name="row")
    with rows as row:
        r = row
        coords = []
        for i in (3, 4, 5, 6):
            r, ci = e.divmod(r, ext[i], *mg[i])
            coords.append(ci)
        coords.append(r)
        roff = {}
        for n in names:
            o = ioff[n]
            for j, ci in enumerate(coords):
                o = b.add(o, e.off(ci, ST[n][3 + j]))
            roff[n] = o
        ok = t_ok
        ok_load = None
        if lim is not None:
            rl = in_lim
            for j, ci in enumerate(coords):
                rl = b.land(rl, b.cmp_lt(ci, lim[3 + j]))
            ok_load = rl
        with b.scf_if(ok):
            vals = {}
            ncomp = 2 if base.cplx else vec
            if base.math and vec == 1:
                # every input loaded, then each input's operators (one copy per input)
                conds, vs = [], []
                for n in names[:-1]:
                    cond = nz[n]
                    if ok_load is not None:
                        cond = b.land(cond, ok_load)
                    v = e.when(
                        cond, [zero], lambda _c, n=n: e.load(P[n], roff[n], dts[n])
                    )
                    conds.append(cond)
                    vs.append(v[0])
                xs = e.unary_inputs(vs, [OPS[n] for n in names[:-1]])
                for n, cond, x in zip(names[:-1], conds, xs):
                    vals[n] = [b.select(cond, b.fmul(S[n][0], x), zero)]
            for n in names[:-1] if not (base.math and vec == 1) else []:
                zeros = [zero] * ncomp
                cond = nz[n]
                if ok_load is not None:
                    cond = b.land(cond, ok_load)
                if vec > 1:
                    v = e.when(cond, zeros, lambda _c, n=n: load_vec(n, roff[n]))
                else:
                    v = e.when(
                        cond,
                        zeros,
                        lambda _c, n=n: e.load(P[n], roff[n], dts[n], base.cplx),
                    )
                if base.cplx:
                    re_, im_ = v
                    im_ = b.select(b.cmp_eq(OPS[n], e.c32(9)), b.fneg(im_), im_)
                    v = list(e.cmul(S[n][0], S[n][1], re_, im_))
                    vals[n] = [b.select(cond, c, zero) for c in v]
                elif vec == 1:
                    vals[n] = [
                        b.select(cond, b.fmul(S[n][0], e.unary(v[0], OPS[n])), zero)
                    ]
                else:
                    vals[n] = [
                        b.select(cond, b.fmul(S[n][0], x), zero)
                        for x in e.unary_list(v, OPS[n])
                    ]
            if base.cplx:
                out = vals["A"]
                if base.arity == 2:
                    out = [b.fadd(out[0], vals["C"][0]), b.fadd(out[1], vals["C"][1])]
                e.store(P["D"], roff["D"], out, base.d)
            else:
                out = []
                for k in range(vec):
                    if base.arity == 1:
                        out.append(vals["A"][k])
                    elif base.arity == 2:
                        out.append(e.binary(vals["A"][k], vals["C"][k], OPS["AB"]))
                    else:
                        ab = e.binary(vals["A"][k], vals["B"][k], OPS["AB"])
                        out.append(e.binary(ab, vals["C"][k], OPS["ABC"]))
                if vec > 1:
                    store_vec(roff["D"], out)
                else:
                    e.store(P["D"], roff["D"], out, base.d)
    return b.kernel


# ------------------------------------------------------------------------------------------------
# Kernel classes and recipes
# ------------------------------------------------------------------------------------------------


def ewv_kernel_name(spec: EwJitSpec) -> str:
    """The name of a class's kernel in hipTensor's kernel pack (its geometry as arguments)."""
    return spec.kernel_name().replace("hiptensor_jit_ew", "hiptensor_aot_ewv", 1)


def jit_vec(d: str) -> int:
    """The vector width of a class whose output type is ``d``: 16-byte output accesses."""
    return 16 // DTYPES[d][1]


def jit_classes(bases: Sequence[EwSpec]) -> List[EwJitSpec]:
    """The kernel classes of the generic rows ``bases``: each row with ``vec`` 1 and, for real
    rows, with the output's vector width and every combination of vector / gathered inputs (rows
    with 64-bit offsets and math rows: only every input a vector; for 64-bit offsets the layout
    of tensors that large in practice). Every launch of a row has a class (:func:`jit_class_of`
    falls back to ``vec`` 1).
    """
    out = []
    for base in bases:
        out.append(EwJitSpec(base))
        if base.math:
            # the math rows: vec 1 and the class of every input a vector (their kernels carry
            # the math programs; a vector class with gathered inputs would add a math program
            # per gathered load)
            n_in = len(base.tensors()) - 1
            out.append(EwJitSpec(base, jit_vec(base.d), (True,) * n_in))
            continue
        if base.cplx:
            continue
        n_in = len(base.tensors()) - 1
        masks = [(1 << n_in) - 1] if base.idx64 else range(1 << n_in)
        for mask in masks:
            unit = tuple(bool(mask >> i & 1) for i in range(n_in))
            out.append(EwJitSpec(base, jit_vec(base.d), unit))
    return out


def jit_class_of(
    base: EwSpec, vals: Dict[str, int]
) -> Tuple[EwJitSpec, Dict[str, int], int]:
    """The class, geometry and thread count of a launch of the generic kernel ``base`` with
    argument values ``vals`` (the rule hipTensor's C++ runtime applies, ``rocke_jit.cpp``
    ``classify``): the output's vector width when D has stride 1 in mode 0, ``e0`` (and ``lim0``)
    are multiples of it and so are the other strides of D and of every input read with vector
    loads (stride 1 in mode 0); else ``vec`` 1. Buffer alignment is checked at execution.
    """
    from .common import magic_u31, as_i32_bits

    names = base.tensors()
    vec = 1
    unit: Tuple[bool, ...] = ()
    if not base.cplx:
        v = jit_vec(base.d)
        ok = vals["sD0"] == 1 and vals["e0"] % v == 0 and vals["e0"] >= v
        if ok and base.arity == 1:
            ok = vals["lim0"] % v == 0
        flags = []
        for n in names:
            vector = n == "D" or vals[f"s{n}0"] == 1
            if vector:
                ok = ok and all(vals[f"s{n}{i}"] % v == 0 for i in range(1, MAX_RANK))
            if n != "D":
                flags.append(vector)
        # 64-bit offset rows and math rows have vector classes only with every input a vector
        if ok and not ((base.idx64 or base.math) and not all(flags)):
            vec, unit = v, tuple(flags)
    spec = EwJitSpec(base, vec, unit)
    geom = {name: int(vals[name]) for name, _ in spec.axes()}
    if vec > 1:
        e0 = vals["e0"] // vec
        m, s = magic_u31(e0)
        geom.update(n_in=vals["n_in"] // vec, e0=e0, m0=as_i32_bits(m), s0=s)
    return spec, geom, geom["n_in"]


def _sentinels(spec: EwJitSpec, seed: int) -> Dict[str, int]:
    """Distinct values for every axis (the recorder sees each constant once)."""
    rng = random.Random(seed)
    used = set()
    vals = {}
    for name, _ in spec.axes():
        v = rng.randrange(1000, 1 << 30)
        while v in used:
            v = rng.randrange(1000, 1 << 30)
        used.add(v)
        vals[name] = v
    return vals


def roll_jit_ew(spec: EwJitSpec, verify: int = 2):
    """One parametric recipe of class ``spec`` over all geometry constants (``rocke.recipe/v1``),
    verified at ``verify`` fresh random points against new recordings of the builder. Raises
    ``RuntimeError`` when the class does not roll."""
    from rocke.portable_ir.src.roll_nd import roll_nd

    base = _sentinels(spec, 1)
    probe = _sentinels(spec, 2)
    axes = {name: [base[name], probe[name]] for name, _ in spec.axes()}
    points = [_sentinels(spec, 100 + i) for i in range(verify)]
    res = roll_nd(
        lambda **g: build_ew_jit(spec, g),
        axes=axes,
        verify_points=points,
        fit_cross_terms=False,
    )
    if not res.ok:
        raise RuntimeError(f"{spec.kernel_name()}: recipe does not roll: {res.reason}")
    return res.recipe


def recipe_entry(spec: EwJitSpec, arch: str) -> Dict:
    """The bundle entry of class ``spec`` (``rocke.bundle/v1``)."""
    return {
        "key": spec.key(),
        "arch": arch,
        "family": EW_JIT_ABI,
        "recipe": roll_jit_ew(spec),
    }
