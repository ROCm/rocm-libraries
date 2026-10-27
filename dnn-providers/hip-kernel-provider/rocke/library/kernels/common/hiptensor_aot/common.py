# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Shared pieces of the shape-generic hipTensor kernels (:mod:`.ew`, :mod:`.reduce`).

Every kernel of this package takes its problem (extents, strides, scalars and operator codes)
as kernel arguments, so one compiled kernel serves every problem of its dtype row and rank
bucket: the kernels are built once, ahead of time, into hipTensor's kernel pack and launched by
hipTensor's C++ runtime without Python.

Index arithmetic:

* Coordinates are i32. A flat index is split into coordinates with host-computed magic numbers
  (:func:`magic_u31`): ``q = x / d`` is ``umulhi(x, m) >> s`` for every ``0 <= x < 2^31``; the
  host passes ``m = 0`` for ``d = 1`` (then ``q = x``).
* Offsets are element offsets, i32 (``idx64 = False``, every offset of every tensor below 2^31)
  or i64 (the pointer is advanced by a 64-bit byte offset before each access).

Operator codes are hipTensor's ``hiptensorOperator_t`` values (:data:`UNARY_OPS`,
:data:`BINARY_OPS`); a kernel branches on them with wave-uniform branches, so an operator a
problem does not use costs a scalar compare per element.
"""

from __future__ import annotations

import itertools
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from rocke.core.ir import BF16, F16, F32, F64, I32, I64, IRBuilder, PtrType, Type, Value
from rocke.helpers import elementary

#: I/O dtypes and their element size in bytes.
DTYPES: Dict[str, Tuple[Type, int]] = {
    "f16": (F16, 2),
    "bf16": (BF16, 2),
    "f32": (F32, 4),
    "f64": (F64, 8),
}

#: hipTensor unary operators the kernels implement (hiptensorOperator_t values).
OP_IDENTITY = 1
OP_SQRT = 2
OP_RELU = 8
OP_CONJ = 9
OP_RCP = 10
OP_SIGMOID = 11
OP_TANH = 12
OP_EXP = 22
OP_LOG = 23
OP_ABS = 24
OP_NEG = 25
OP_SIN = 26
OP_COS = 27
OP_TAN = 28
OP_SINH = 29
OP_COSH = 30
OP_ASIN = 31
OP_ACOS = 32
OP_ATAN = 33
OP_ASINH = 34
OP_ACOSH = 35
OP_ATANH = 36
OP_CEIL = 37
OP_FLOOR = 38

#: The operators every kernel implements (single instructions or short sequences).
BASIC_UNARY_OPS = (
    OP_IDENTITY,
    OP_SQRT,
    OP_RELU,
    OP_CONJ,
    OP_RCP,
    OP_SIGMOID,
    OP_TANH,
    OP_EXP,
    OP_LOG,
    OP_ABS,
    OP_NEG,
)

#: The operators only the ``math`` kernels implement (:mod:`rocke.helpers.elementary`
#: programs, evaluated in f64), by the elementary function name.
MATH_OPS = {
    OP_SIN: "sin",
    OP_COS: "cos",
    OP_TAN: "tan",
    OP_SINH: "sinh",
    OP_COSH: "cosh",
    OP_ASIN: "asin",
    OP_ACOS: "acos",
    OP_ATAN: "atan",
    OP_ASINH: "asinh",
    OP_ACOSH: "acosh",
    OP_ATANH: "atanh",
    OP_CEIL: "ceil",
    OP_FLOOR: "floor",
}

#: Every hipTensor unary operator (``HIPTENSOR_OP_IDENTITY`` to ``HIPTENSOR_OP_FLOOR``).
UNARY_OPS = BASIC_UNARY_OPS + tuple(MATH_OPS)

#: hipTensor binary operators (elementwise op_AB / op_ABC, reduction op).
OP_ADD = 3
OP_MUL = 5
OP_MAX = 6
OP_MIN = 7
BINARY_OPS = (OP_ADD, OP_MUL, OP_MAX, OP_MIN)

#: Largest rank of a tensor after canonicalization (rank bucket R8).
MAX_RANK = 8

INT32_LIMIT = 1 << 31


def magic_u31(d: int) -> Tuple[int, int]:
    """``(m, s)`` with ``x // d == umulhi(x, m) >> s`` for every ``0 <= x < 2^31``.

    ``(0, 0)`` for ``d == 1`` (the kernels then use ``x`` itself). hipTensor's C++ runtime
    computes the same numbers (``rocke_runtime.cpp``, ``magicU31``)."""
    if d <= 0:
        raise ValueError("divisor must be positive")
    if d == 1:
        return 0, 0
    for s in range(32, 64):
        m = -(-(1 << s) // d)
        if m >= (1 << 32):
            break
        e = m * d - (1 << s)
        if e * (INT32_LIMIT - 1) < (1 << s):
            return m, s - 32
    raise ValueError(f"no 32-bit magic for divisor {d}")


def as_i32_bits(v: int) -> int:
    """An unsigned 32-bit value as the signed i32 kernel argument with the same bits."""
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v >= (1 << 31) else v


class Emit:
    """Small helpers over an :class:`IRBuilder` for the generic kernels."""

    def __init__(self, b: IRBuilder, idx64: bool, cdt: str, math: bool = False):
        self.b = b
        # ``math``: the unary operators include :data:`MATH_OPS` (the kernels' ``_m`` rows)
        self.math = math
        # Loop-variable names are numbered per kernel (one Emit per IRBuilder), so a kernel's IR
        # and its rolled recipe do not depend on what the process built before.
        self._uid = itertools.count()
        self.idx64 = idx64
        self.cdt = cdt
        self.cty = F64 if cdt == "f64" else F32

    # ---------------------------------------------------------------- constants / params
    def c32(self, v: int) -> Value:
        return self.b.const_i32(int(v))

    def cf(self, v: float) -> Value:
        return self.b.const_f64(v) if self.cdt == "f64" else self.b.const_f32(v)

    def zero_off(self) -> Value:
        return self.b.const_i64(0) if self.idx64 else self.b.const_i32(0)

    def stride_param(self, name: str) -> Value:
        return self.b.param(name, I64 if self.idx64 else I32)

    def scalar_param(self, name: str) -> Value:
        return self.b.param(name, self.cty)

    # ---------------------------------------------------------------- integer helpers
    def udiv(self, x: Value, m: Value, s: Value) -> Value:
        """``x // d`` for ``0 <= x < 2^31`` with the magic pair ``(m, s)`` of ``d``."""
        b = self.b
        q = b.lshr(b.umul_hi_i32(x, m), s)
        return b.select(b.cmp_eq(m, self.c32(0)), x, q)

    def divmod(self, x: Value, d: Value, m: Value, s: Value) -> Tuple[Value, Value]:
        b = self.b
        q = self.udiv(x, m, s)
        return q, b.sub(x, b.mul(q, d))

    def off(self, coord: Value, stride: Value) -> Value:
        """``coord * stride`` as an offset (``stride`` already has the offset type)."""
        b = self.b
        if self.idx64:
            return b.mul(b.sext(coord, I64), stride)
        return b.mul(coord, stride)

    # ---------------------------------------------------------------- memory
    def _addr(self, ptr: Value, off: Value, nbytes: int) -> Tuple[Value, Value]:
        b = self.b
        if self.idx64:
            return b.global_ptr_add(ptr, b.mul(off, b.const_i64(nbytes))), b.const_i32(
                0
            )
        return ptr, off

    def load(self, ptr: Value, off: Value, dt: str, cplx: bool = False) -> List[Value]:
        """The element at element offset ``off`` (complex: both components), in the
        compute type."""
        b = self.b
        ty, nb = DTYPES[dt]
        if cplx:
            p, idx = self._addr(ptr, off, 2 * nb)
            # complex elements: index in units of the component pair
            if not self.idx64:
                idx = b.add(idx, idx)
            v = b.global_load_vN(p, idx, ty, 2, align=2 * nb)
            return [
                self.to_c(b.vec_extract(v, 0), dt),
                self.to_c(b.vec_extract(v, 1), dt),
            ]
        p, idx = self._addr(ptr, off, nb)
        return [self.to_c(b.global_load(p, idx, ty, align=nb), dt)]

    def store(self, ptr: Value, off: Value, vals: Sequence[Value], dt: str) -> None:
        b = self.b
        ty, nb = DTYPES[dt]
        outs = [self.from_c(v, dt) for v in vals]
        if len(outs) == 2:
            p, idx = self._addr(ptr, off, 2 * nb)
            if not self.idx64:
                idx = b.add(idx, idx)
            b.global_store_vN(p, idx, b.vec_pack(outs, ty), 2, align=2 * nb)
            return
        p, idx = self._addr(ptr, off, nb)
        b.global_store(p, idx, outs[0], align=nb)

    # ---------------------------------------------------------------- conversions
    def to_c(self, v: Value, dt: str) -> Value:
        b = self.b
        if self.cdt == "f64":
            if dt == "f64":
                return v
            v32 = v if dt == "f32" else b.cast_to_f32(v)
            return b.cvt_f32_to_f64(v32)
        if dt == "f64":
            return b.cvt_f64_to_f32(v)
        return v if dt == "f32" else b.cast_to_f32(v)

    def from_c(self, v: Value, dt: str) -> Value:
        b = self.b
        ty = DTYPES[dt][0]
        if self.cdt == "f64":
            if dt == "f64":
                return v
            if dt == "f32":
                return b.cvt_f64_to_f32(v)
            return self.narrow(self.f64_to_f32_odd(v), ty)
        if dt == "f64":
            return b.cvt_f32_to_f64(v)
        return v if dt == "f32" else self.narrow(v, ty)

    def narrow(self, v: Value, ty) -> Value:
        """f32 ``v`` rounded to nearest even to f16 / bf16, a zero keeping its sign.

        The AMDGPU backend folds an f32 multiply whose only use is the f16 conversion into one
        mixed-precision FMA with a +0 addend (``v_fma_mixlo_f16 a, b, 0``), which turns a -0
        product into +0. A zero therefore takes the conversion of a zero built from ``v``'s
        sign bit (no multiply behind it), every other value the plain conversion."""
        b = self.b
        if ty is not F16:
            return b.cast_f32_to(v, ty)
        signed_zero = b.bitcast(b.land(b.bitcast(v, I32), self.c32(-(1 << 31))), F32)
        is_zero = b.fcmp("oeq", v, self.cf_f32(0.0))
        return b.cast_f32_to(b.select(is_zero, signed_zero, v), ty)

    def cf_f32(self, v: float) -> Value:
        return self.b.const_f32(v)

    def f64_to_f32_odd(self, v: Value) -> Value:
        """``v`` rounded to f32 with round-to-odd: the f32 value toward zero, its last bit set
        when ``v`` is not exactly an f32. Rounding that value to nearest even once more to a
        format of at most 22 significand bits (f16: 11, bf16: 8) gives ``v`` rounded to nearest
        even directly, so f64 -> f16 / bf16 rounds once. NaN stays NaN, infinities stay
        infinite, and a finite value past the f32 range becomes the largest f32 (it still
        rounds to infinity in the narrower format)."""
        b = self.b
        r = b.cvt_f64_to_f32(v)  # nearest even
        back = b.cvt_f32_to_f64(r)
        inexact = b.fcmp("one", back, v)  # false for NaN, which stays NaN
        away = b.fcmp("ogt", b.fabs(back), b.fabs(v))
        bits = b.bitcast(r, I32)
        # one step toward zero when the nearest even rounding went away from zero (the bits
        # of a nonzero f32 are its magnitude in sign-magnitude order)
        bits = b.select(b.land(inexact, away), b.sub(bits, self.c32(1)), bits)
        bits = b.select(inexact, b.lor(bits, self.c32(1)), bits)
        return b.bitcast(bits, F32)

    # ---------------------------------------------------------------- uniform branches
    def when(
        self,
        cond: Value,
        vals: Sequence[Value],
        fn: Callable[[List[Value]], List[Value]],
    ):
        """``fn(vals)`` when the wave-uniform ``cond`` holds, else ``vals``: a loop of
        ``cond ? 1 : 0`` trips that carries the values (an ``scf.if`` yields nothing).
        """
        b = self.b
        trips = b.select(cond, self.c32(1), self.c32(0))
        k = next(self._uid)
        loop = b.scf_for_iter(
            self.c32(0),
            trips,
            self.c32(1),
            [(f"w{k}_{i}", v) for i, v in enumerate(vals)],
            iv_name=f"wi{k}",
        )
        with loop as (_it, cur):
            b.scf_yield(*fn(list(cur)))
        return list(loop.results)

    # ---------------------------------------------------------------- operators
    def unary(self, v: Value, code: Value) -> Value:
        """hipTensor unary operator ``code`` applied to a real value (CONJ is the identity)."""
        return self.unary_list([v], code)[0]

    def unary_list(
        self, vs: Sequence[Value], code: Value, math: Optional[bool] = None
    ) -> List[Value]:
        """:meth:`unary` of each of ``vs`` (one wave-uniform branch per operator for all of
        them). With ``math`` (default: the Emit's) the operators of :data:`MATH_OPS` too: one
        branch per :data:`rocke.helpers.elementary.FAMILIES` group, evaluated in f64 (f32
        compute values are rounded once back to f32), one copy of each program for all of
        ``vs``."""
        b = self.b
        one = self.cf(1.0)
        math = self.math if math is None else math

        def table(xs: List[Value]) -> List[Value]:
            r = list(xs)
            for op, fn in (
                (OP_SQRT, lambda c: [b.sqrt(y) for y in c]),
                (OP_RELU, lambda c: [b.fmax(y, self.cf(0.0)) for y in c]),
                (OP_RCP, lambda c: [b.fdiv(one, y) for y in c]),
                (OP_SIGMOID, lambda c: [self.sigmoid(y) for y in c]),
                (OP_TANH, lambda c: [b.tanh(y) for y in c]),
                (OP_EXP, lambda c: [b.exp(y) for y in c]),
                (OP_LOG, lambda c: [b.log(y) for y in c]),
                (OP_ABS, lambda c: [b.fabs(y) for y in c]),
                (OP_NEG, lambda c: [b.fneg(y) for y in c]),
            ):
                r = self.when(b.cmp_eq(code, self.c32(op)), r, fn)
            if math:
                r = self.math_ops(r, code)
            return r

        not_id = b.land(
            b.cmp_ne(code, self.c32(OP_IDENTITY)), b.cmp_ne(code, self.c32(OP_CONJ))
        )
        return self.when(not_id, list(vs), table)

    def unary_inputs(self, vs: Sequence[Value], codes: Sequence[Value]) -> List[Value]:
        """:meth:`unary` of value ``vs[i]`` with operator ``codes[i]``: one copy of the
        operators per input (so that a kernel whose operators are constants keeps only the
        programs it uses, and the inputs' programs are not scheduled together)."""
        return [self.unary_list([v], c)[0] for v, c in zip(vs, codes)]

    def sigmoid(self, y: Value) -> Value:
        """``1 / (1 + e^-y)`` as ``1 / (1 + t)`` (y >= 0) or ``t / (1 + t)`` (y < 0) with
        ``t = e^-|y|``: ``t`` never overflows, so large negative arguments keep their tiny (or
        subnormal) results instead of ``1 / inf = 0``."""
        b = self.b
        one = self.cf(1.0)
        t = b.exp(b.fneg(b.fabs(y)))
        den = b.fadd(one, t)
        pos = b.fcmp("oge", y, self.cf(0.0))
        return b.fdiv(b.select(pos, one, t), den)

    def math_ops(self, xs: Sequence[Value], code: Value) -> List[Value]:
        """The :data:`MATH_OPS` operator ``code`` applied to each of ``xs`` (other codes leave
        the values), per :data:`rocke.helpers.elementary.FAMILIES` group: a wave-uniform branch
        when ``code`` is of the group, then the group's program once, in a loop over the values
        (each picked and put back with selects) when there are several."""
        b = self.b
        r = list(xs)
        for fam, names in elementary.FAMILIES.items():
            preds = {
                n: b.cmp_eq(code, self.c32(op))
                for op, n in MATH_OPS.items()
                if n in names
            }
            hit = None
            for p in preds.values():
                hit = p if hit is None else b.lor(hit, p)
            r = self.when(
                hit, r, lambda c, fam=fam, preds=preds: self._math_loop(fam, c, preds)
            )
        return r

    def _math_loop(self, fam: str, xs: Sequence[Value], preds) -> List[Value]:
        b = self.b
        if len(xs) == 1:
            return [self.math_family(fam, xs[0], preds)]
        k = next(self._uid)
        # the trip count on an opaque zero: the loop stays a loop (unrolled, its copies of the
        # program would be scheduled together and need their registers at once)
        opaque = b.inline_asm(
            "s_mov_b32 $0, 0", "=s", [], I32, sideeffect=True, result_name_hint="ozero"
        )
        loop = b.scf_for_iter(
            self.c32(0),
            b.add(self.c32(len(xs)), opaque),
            self.c32(1),
            [(f"m{k}_{i}", v) for i, v in enumerate(xs)],
            iv_name=f"mi{k}",
        )
        with loop as (it, cur):
            y = cur[0]
            for j in range(1, len(cur)):
                y = b.select(b.cmp_eq(it, self.c32(j)), cur[j], y)
            r = self.math_family(fam, y, preds)
            b.scf_yield(
                *[
                    b.select(b.cmp_eq(it, self.c32(j)), r, cur[j])
                    for j in range(len(cur))
                ]
            )
        return list(loop.results)

    def math_family(self, fam: str, y: Value, preds) -> Value:
        """The function of :data:`rocke.helpers.elementary.FAMILIES` group ``fam`` selected by
        ``preds`` (name -> i1) applied to the compute-type value ``y``, in f64."""
        b = self.b
        # constants anchored in the branch: not hoisted out of the kernel's loops into VGPRs
        m = elementary.IRMath(b, elementary.opaque_zero(b))
        m.materialize(elementary.family_constants(fam, preds))
        if self.cdt == "f64":
            return elementary.family(m, fam, y, preds)
        return b.cvt_f64_to_f32(elementary.family(m, fam, b.cvt_f32_to_f64(y), preds))

    def binary(self, a: Value, c: Value, code: Value, nan_max: bool = True) -> Value:
        """hipTensor binary operator ``code`` (ADD / MUL / MAX / MIN). MAX and MIN propagate
        NaN (``nan_max``)."""
        b = self.b
        mx = b.fmax(a, c)
        mn = b.fmin(a, c)
        if nan_max:
            uno = b.fcmp("uno", a, c)
            nan = b.fadd(a, c)
            mx = b.select(uno, nan, mx)
            mn = b.select(uno, nan, mn)
        r = b.fadd(a, c)
        r = b.select(b.cmp_eq(code, self.c32(OP_MUL)), b.fmul(a, c), r)
        r = b.select(b.cmp_eq(code, self.c32(OP_MAX)), mx, r)
        r = b.select(b.cmp_eq(code, self.c32(OP_MIN)), mn, r)
        return r

    def cmul(self, xr: Value, xi: Value, yr: Value, yi: Value) -> Tuple[Value, Value]:
        b = self.b
        return (
            b.fsub(b.fmul(xr, yr), b.fmul(xi, yi)),
            b.fadd(b.fmul(xr, yi), b.fmul(xi, yr)),
        )


def ptr(b: IRBuilder, name: str, dt: str, *, readonly: bool) -> Value:
    ty, nb = DTYPES[dt]
    attrs = {"align": nb}
    if readonly:
        attrs["readonly"] = True
    return b.param(name, PtrType(ty, "global"), **attrs)
