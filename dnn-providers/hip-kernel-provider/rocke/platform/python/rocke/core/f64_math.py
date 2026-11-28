# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Straight-line f64 math expansions shared by both LLVM lowerers.

The AMDGPU backend selects f64 ``sqrt`` and ``fma`` natively but has no f64
``exp``/``log``/``tanh``: a bare ``llvm.exp.f64`` becomes a libcall that never
links. Each such op is therefore expanded at lowering time into a fixed program
of LLVM instructions. The program is data (:class:`Program`), so:

* the Python lowerer and the C++ lowerer print the same instruction text in the
  same order -- the C++ engine reads ``cpp/core/lower_llvm/f64_math_programs.inc``,
  generated from this module by ``tools/gen_f64_math.py`` (a test keeps the two
  in sync);
* :func:`evaluate` runs the program on the host in IEEE f64 (exact ``fma``), so
  tests measure the accuracy of exactly the instructions that are emitted.

Algorithms (no lookup tables, branch-free, every special value by ``select``):

* ``exp``/``exp2``: ``x = n*ln2 + r`` (Cody-Waite with ``fma``), degree-13
  Taylor polynomial of ``e^r - 1`` in Horner form, ``ldexp(1 + s, n)``.
* ``log``/``log2``: ``frexp`` with subnormal pre-scaling, ``m`` in
  ``[sqrt(1/2), sqrt(2))``, ``log(1 + f) = f - (f^2/2 - s*(f^2/2 + R))`` with
  ``s = f/(2 + f)`` and ``R`` the ``atanh`` series in ``s^2`` (the classic fdlibm
  formulation), ``ln2`` split so ``e*ln2_hi`` is exact; ``log2`` carries the
  result as a double-word sum before the final rounding.
* ``tanh``: ``tanh|x| = u/(u + 2)`` with ``u = expm1(2|x|)``, carried as a
  double-word pair through the reduction (an exact ``n*ln2_hi`` step), the
  ``e^r - 1`` sum, the ``2^n`` reconstruction and a Newton-corrected quotient;
  ``|x| < 2^-27`` returns ``x``, ``|x| > 22`` returns ``1``; sign restored with
  the sign bit.
* ``sqrt``: ``llvm.sqrt.f64`` (the backend expansion is IEEE correctly rounded);
  ``rsqrt``: ``1 / sqrt(x)`` with one correction from the exact residuals of the
  square root and the reciprocal (tiny arguments scaled by ``2^256`` first).

The polynomial coefficients are ``1/k!``, ``ln2^k/k!`` and ``2/(2k+1)`` rounded
to the nearest double; :func:`coefficient_table` recomputes them exactly and the
unit tests compare.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------- literals


def f64_literal(value: float) -> str:
    """LLVM hexadecimal spelling of an f64 constant (exact bits)."""
    (bits,) = struct.unpack("<Q", struct.pack("<d", float(value)))
    return f"0x{bits:016X}"


def _hexf(s: str) -> float:
    return float.fromhex(s)


# ln2 = LN2 + LN2_LO (LN2 the nearest double), for the exp reductions.
LN2 = _hexf("0x1.62e42fefa39efp-1")
LN2_LO = _hexf("0x1.abc9e3b39803fp-56")
# ln2 = LN2_HI33 + LN2_LO33 with a 33-bit LN2_HI33, so k*LN2_HI33 is exact for
# |k| < 2^20 (log reconstruction).
LN2_HI33 = _hexf("0x1.62e42feep-1")
LN2_LO33 = _hexf("0x1.a39ef35793c76p-33")
LOG2E = _hexf("0x1.71547652b82fep+0")
LOG2E_LO = _hexf("0x1.777d0ffda0d24p-56")
SQRT_HALF = _hexf("0x1.6a09e667f3bcdp-1")


def coefficient_table() -> Dict[str, Tuple[float, ...]]:
    """The polynomial coefficients, recomputed from their exact definitions.

    ``exp``: ``1/k!`` for k = 2..13. ``exp2``: ``ln2^k/k!`` for k = 1..13
    (ln2 from 120-bit ``math`` series). ``log``: ``2/(2k+1)`` for k = 1..11.
    """
    exp_c = tuple(float(Fraction(1, math.factorial(k))) for k in range(2, 14))
    ln2 = _ln2_fraction()
    exp2_c = tuple(float(ln2**k / math.factorial(k)) for k in range(1, 14))
    log_c = tuple(float(Fraction(2, 2 * k + 1)) for k in range(1, 12))
    return {"exp": exp_c, "exp2": exp2_c, "log": log_c}


def _ln2_fraction() -> Fraction:
    """ln 2 to ~200 bits: ln2 = sum_{k>=1} 1 / (k * 2^k)."""
    total = Fraction(0)
    for k in range(1, 220):
        total += Fraction(1, k * (1 << k))
    return total


def _log2e_fraction() -> Fraction:
    return 1 / _ln2_fraction()


_COEF = coefficient_table()
EXP_C = _COEF["exp"]  # 1/2!, 1/3!, ..., 1/13!
EXP2_C = _COEF["exp2"]  # ln2/1!, ..., ln2^13/13!
LOG_C = _COEF["log"]  # 2/3, 2/5, ..., 2/23

# Guard thresholds (exp(x) overflows above ln(DBL_MAX); underflows to +0 below
# ln(2^-1075)).
EXP_OVERFLOW = _hexf("0x1.62e42fefa39efp+9")
EXP_UNDERFLOW = _hexf("-0x1.74910d52d3051p+9")

# ---------------------------------------------------------------- programs

# Result types of the steps (LLVM spelling).
_DOUBLE = "double"
_I1 = "i1"
_I32 = "i32"
_I64 = "i64"
_FREXP = "{ double, i32 }"


@dataclass(frozen=True)
class Step:
    """One instruction: ``%<fresh hint> = <text>``.

    ``text`` holds ``$0``/``$1``/``$2`` where the operands go. An operand is
    the op input (``-1``) or the result of an earlier step (its index).
    """

    hint: str
    text: str
    args: Tuple[int, ...]
    kind: str
    rtype: str
    imm: Tuple[object, ...] = ()


@dataclass(frozen=True)
class Program:
    name: str
    keys: Tuple[str, ...]
    steps: Tuple[Step, ...]


class _Ref(int):
    pass


_X = _Ref(-1)


class _Builder:
    def __init__(self, name: str) -> None:
        self.name = name
        self.steps: List[Step] = []
        self.keys: List[str] = []

    def _need(self, key: str) -> None:
        if key not in self.keys:
            self.keys.append(key)

    def _add(self, hint, text, args, kind, rtype, imm=()) -> _Ref:
        self.steps.append(
            Step(
                f"{self.name}.{hint}",
                text,
                tuple(int(a) for a in args),
                kind,
                rtype,
                tuple(imm),
            )
        )
        return _Ref(len(self.steps) - 1)

    # Binary arithmetic. A float `b` is an inline constant.
    def _bin(self, opcode: str, hint: str, a, b) -> _Ref:
        if isinstance(b, float):
            return self._add(
                hint,
                f"{opcode} double $0, {f64_literal(b)}",
                [a],
                opcode + "_k",
                _DOUBLE,
                [b],
            )
        if isinstance(a, float):
            return self._add(
                hint,
                f"{opcode} double {f64_literal(a)}, $0",
                [b],
                opcode + "_kl",
                _DOUBLE,
                [a],
            )
        return self._add(hint, f"{opcode} double $0, $1", [a, b], opcode, _DOUBLE)

    def fmul(self, hint, a, b):
        return self._bin("fmul", hint, a, b)

    def fadd(self, hint, a, b):
        return self._bin("fadd", hint, a, b)

    def fsub(self, hint, a, b):
        return self._bin("fsub", hint, a, b)

    def fdiv(self, hint, a, b):
        return self._bin("fdiv", hint, a, b)

    def fma(self, hint, a, b, c):
        """``a*b + c`` rounded once; float `b`/`c` are inline constants."""
        self._need("fma.f64")
        args, imm, parts = [], [], []
        for v in (a, b, c):
            if isinstance(v, float):
                parts.append(f"double {f64_literal(v)}")
                imm.append(v)
            else:
                parts.append(f"double ${len(args)}")
                args.append(v)
                imm.append(None)
        return self._add(
            hint,
            f"call double @llvm.fma.f64({', '.join(parts)})",
            args,
            "fma",
            _DOUBLE,
            imm,
        )

    def _call1(self, hint, intrin, key, a):
        self._need(key)
        return self._add(
            hint, f"call double @llvm.{intrin}(double $0)", [a], intrin, _DOUBLE
        )

    def fabs(self, hint, a):
        return self._call1(hint, "fabs.f64", "fabs.f64", a)

    def rint(self, hint, a):
        return self._call1(hint, "rint.f64", "rint.f64", a)

    def sqrt(self, hint, a):
        return self._call1(hint, "sqrt.f64", "sqrt.f64", a)

    def minnum_k(self, hint, a, k: float):
        self._need("minnum.f64")
        return self._add(
            hint,
            f"call double @llvm.minnum.f64(double $0, double {f64_literal(k)})",
            [a],
            "minnum_k",
            _DOUBLE,
            [k],
        )

    def maxnum_k(self, hint, a, k: float):
        self._need("maxnum.f64")
        return self._add(
            hint,
            f"call double @llvm.maxnum.f64(double $0, double {f64_literal(k)})",
            [a],
            "maxnum_k",
            _DOUBLE,
            [k],
        )

    def ldexp(self, hint, a, e):
        self._need("ldexp.f64")
        if isinstance(a, float):
            return self._add(
                hint,
                f"call double @llvm.ldexp.f64.i32(double {f64_literal(a)}, i32 $0)",
                [e],
                "ldexp_k",
                _DOUBLE,
                [a],
            )
        return self._add(
            hint,
            "call double @llvm.ldexp.f64.i32(double $0, i32 $1)",
            [a, e],
            "ldexp",
            _DOUBLE,
        )

    def frexp(self, hint, a):
        self._need("frexp.f64")
        return self._add(
            hint, f"call {_FREXP} @llvm.frexp.f64.i32(double $0)", [a], "frexp", _FREXP
        )

    def extract(self, hint, a, idx: int):
        rt = _DOUBLE if idx == 0 else _I32
        return self._add(
            hint, f"extractvalue {_FREXP} $0, {idx}", [a], "extract", rt, [idx]
        )

    def fcmp_k(self, hint, pred: str, a, k: float):
        return self._add(
            hint,
            f"fcmp {pred} double $0, {f64_literal(k)}",
            [a],
            "fcmp_k",
            _I1,
            [pred, k],
        )

    def fcmp(self, hint, pred: str, a, b):
        return self._add(
            hint, f"fcmp {pred} double $0, $1", [a, b], "fcmp", _I1, [pred]
        )

    def select(self, hint, c, a, b, ty: str = _DOUBLE):
        """``c ? a : b``; a float/int `a` or `b` is an inline constant."""
        args, imm, parts = [c], [], []
        for v in (a, b):
            if isinstance(v, float):
                parts.append(f"{ty} {f64_literal(v)}")
                imm.append(v)
            elif isinstance(v, int) and not isinstance(v, _Ref):
                parts.append(f"{ty} {v}")
                imm.append(v)
            else:
                parts.append(f"{ty} ${len(args)}")
                args.append(v)
                imm.append(None)
        return self._add(
            hint, f"select i1 $0, {', '.join(parts)}", args, "select", ty, imm
        )

    def fptosi(self, hint, a):
        return self._add(hint, "fptosi double $0 to i32", [a], "fptosi", _I32)

    def sitofp(self, hint, a):
        return self._add(hint, "sitofp i32 $0 to double", [a], "sitofp", _DOUBLE)

    def add_i32(self, hint, a, k: int):
        return self._add(hint, f"add i32 $0, {k}", [a], "add_i32", _I32, [k])

    def add2_i32(self, hint, a, b):
        return self._add(hint, "add i32 $0, $1", [a, b], "add2_i32", _I32)

    def to_bits(self, hint, a):
        return self._add(hint, "bitcast double $0 to i64", [a], "to_bits", _I64)

    def from_bits(self, hint, a):
        return self._add(hint, "bitcast i64 $0 to double", [a], "from_bits", _DOUBLE)

    def and_i64(self, hint, a, k: int):
        return self._add(hint, f"and i64 $0, {k}", [a], "and_i64", _I64, [k])

    def or_i64(self, hint, a, b):
        return self._add(hint, "or i64 $0, $1", [a, b], "or_i64", _I64)

    def done(self) -> Program:
        return Program(self.name, tuple(self.keys), tuple(self.steps))


def _horner(p: _Builder, hint: str, z, coefs: Sequence[float]):
    """``coefs[0] + z*(coefs[1] + z*(...))`` with fma, highest degree first."""
    acc = None
    for i in range(len(coefs) - 1, 0, -1):
        if acc is None:
            acc = p.fma(f"{hint}{i}", z, coefs[i], coefs[i - 1])
        else:
            acc = p.fma(f"{hint}{i}", z, acc, coefs[i - 1])
    return acc


def _expm1_reduced(p: _Builder, r):
    """``e^r - 1`` for ``|r| <= ln2/2``: ``r + r^2 * P(r)``, P the 1/k! series."""
    q = _horner(p, "q", r, EXP_C)
    rr = p.fmul("rr", r, r)
    return p.fma("em1", rr, q, r)


def _build_exp() -> Program:
    p = _Builder("exp")
    t = p.fmul("t", _X, LOG2E)
    n = p.rint("n", t)
    n = p.maxnum_k("nlo", p.minnum_k("nhi", n, 2000.0), -2000.0)
    ni = p.fptosi("ni", n)
    r = p.fma("r", n, -LN2, _X)
    r = p.fma("r", n, -LN2_LO, r)
    s = _expm1_reduced(p, r)
    y = p.fadd("y", s, 1.0)
    z = p.ldexp("z", y, ni)
    z = p.select("ovf", p.fcmp_k("isovf", "ogt", _X, EXP_OVERFLOW), math.inf, z)
    p.select("res", p.fcmp_k("isunf", "olt", _X, EXP_UNDERFLOW), 0.0, z)
    return p.done()


def _build_exp2() -> Program:
    p = _Builder("exp2")
    n = p.rint("n", _X)
    n = p.maxnum_k("nlo", p.minnum_k("nhi", n, 2000.0), -2000.0)
    ni = p.fptosi("ni", n)
    r = p.fsub("r", _X, n)
    # 2^r - 1 = r*ln2 + r^2*Q(r): the leading term with ln2 as LN2 + LN2_LO.
    q = _horner(p, "q", r, EXP2_C[1:])
    tail = p.fma("tail", r, LN2_LO, p.fmul("rq", p.fmul("rr", r, r), q))
    s = p.fma("s", r, LN2, tail)
    y = p.fadd("y", s, 1.0)
    z = p.ldexp("z", y, ni)
    z = p.select("ovf", p.fcmp_k("isovf", "oge", _X, 1024.0), math.inf, z)
    p.select("res", p.fcmp_k("isunf", "olt", _X, -1080.0), 0.0, z)
    return p.done()


def _log_front(p: _Builder):
    """(f, s, hfsq, t, dk) with x = 2^dk * (1 + f), log(1 + f) = f - (hfsq - t)."""
    sub = p.fcmp_k("issub", "olt", _X, _hexf("0x1p-1022"))
    xs = p.select("xs", sub, p.fmul("xsc", _X, _hexf("0x1p54")), _X)
    eadj = p.select("eadj", sub, -54, 0, ty=_I32)
    fr = p.frexp("fr", xs)
    m0 = p.extract("m0", fr, 0)
    e0 = p.extract("e0", fr, 1)
    small = p.fcmp_k("small", "olt", m0, SQRT_HALF)
    m = p.select("m", small, p.fadd("m2", m0, m0), m0)
    e1 = p.add2_i32("e1", e0, eadj)
    e = p.select("e", small, p.add_i32("em1", e1, -1), e1, ty=_I32)
    f = p.fsub("f", m, 1.0)
    s = p.fdiv("s", f, p.fadd("den", f, 2.0))
    z = p.fmul("z", s, s)
    q = _horner(p, "q", z, LOG_C)
    big_r = p.fmul("R", z, q)
    hfsq = p.fmul("hfsq", p.fmul("ff", f, f), 0.5)
    t = p.fmul("t", s, p.fadd("hr", hfsq, big_r))
    dk = p.sitofp("dk", e)
    return f, hfsq, t, dk


def _log_specials(p: _Builder, val):
    val = p.select("zero", p.fcmp_k("iszero", "oeq", _X, 0.0), -math.inf, val)
    val = p.select("neg", p.fcmp_k("isneg", "olt", _X, 0.0), math.nan, val)
    val = p.select("inf", p.fcmp_k("isinf", "oeq", _X, math.inf), math.inf, val)
    return p.select("res", p.fcmp("isnan", "uno", _X, _X), _X, val)


def _build_log() -> Program:
    p = _Builder("log")
    f, hfsq, t, dk = _log_front(p)
    lo = p.fma("lo", dk, LN2_LO33, t)
    inner = p.fsub("in", p.fsub("in0", hfsq, lo), f)
    val = p.fsub("v", p.fmul("hi", dk, LN2_HI33), inner)
    _log_specials(p, val)
    return p.done()


def _build_log2() -> Program:
    p = _Builder("log2")
    f, hfsq, t, dk = _log_front(p)
    # L = f - hfsq + t as the pair (lhi, llo): Fast2Sum(f, -hfsq), |f| >= |hfsq|.
    lhi = p.fsub("lhi", f, hfsq)
    err = p.fsub("lerr", p.fsub("lres", f, lhi), hfsq)
    llo = p.fadd("llo", err, t)
    # L*log2(e) as (phi, plo).
    phi = p.fmul("phi", lhi, LOG2E)
    perr = p.fma("perr", lhi, LOG2E, p.fmul("nphi", phi, -1.0))
    plo = p.fma("plo", llo, LOG2E, p.fma("pl2", lhi, LOG2E_LO, perr))
    # dk + phi: Fast2Sum (|dk| >= 1 > |phi| or dk == 0).
    sm = p.fadd("sm", dk, phi)
    serr = p.fadd("serr", p.fsub("sd", dk, sm), phi)
    val = p.fadd("v", sm, p.fadd("lo", serr, plo))
    _log_specials(p, val)
    return p.done()


def _build_tanh() -> Program:
    p = _Builder("tanh")
    ax = p.fabs("ax", _X)
    y = p.fadd("y", ax, ax)
    t = p.fmul("t", y, LOG2E)
    n = p.minnum_k("nhi", p.rint("n", t), 64.0)
    ni = p.fptosi("ni", n)
    # y = n*ln2 + (r + c): n*LN2_HI33 and y - n*LN2_HI33 are exact (|n| <= 64,
    # Sterbenz), r the rounded difference with the low part, c its error.
    hi = p.fma("hi", n, -LN2_HI33, y)
    lo = p.fmul("lo", n, LN2_LO33)
    r = p.fsub("r", hi, lo)
    c = p.fsub("c", p.fsub("hr", hi, r), lo)
    # e^(r + c) - 1 as the pair (eh, el): eh = r + r^2*P(r) by Fast2Sum
    # (|r| > |r^2 P|), el its error plus c*e^r.
    q = _horner(p, "q", r, EXP_C)
    rq = p.fmul("rq", p.fmul("rr", r, r), q)
    eh = p.fadd("eh", r, rq)
    el0 = p.fsub("el0", rq, p.fsub("ehr", eh, r))
    el = p.fadd("el", el0, p.fma("ce", c, eh, c))
    # u = 2^n * (1 + e) - 1 = (2^n - 1) + 2^n*eh + 2^n*el as (uh, ul): 2^n*eh is
    # exact, (2^n - 1) + 2^n*eh by TwoSum.
    sc = p.ldexp("sc", 1.0, ni)
    a = p.fsub("scm1", sc, 1.0)
    bb = p.fmul("sce", sc, eh)
    uh = p.fadd("uh", a, bb)
    bv = p.fsub("bv", uh, a)
    av = p.fsub("av", uh, bv)
    ts = p.fadd("ts", p.fsub("da", a, av), p.fsub("db", bb, bv))
    ul = p.fma("ul", sc, el, ts)
    # u / (u + 2) with the denominator as a pair (dhi, dlo) and one Newton
    # correction of the quotient that includes the low parts.
    dhi = p.fadd("dhi", uh, 2.0)
    big_u = p.fcmp_k("bigu", "oge", uh, 2.0)
    dlo0 = p.select(
        "dlo0",
        big_u,
        p.fadd("dlo1", p.fsub("dd1", uh, dhi), 2.0),
        p.fadd("dlo2", p.fsub("dd2", 2.0, dhi), uh),
    )
    dlo = p.fadd("dlo", dlo0, ul)
    q0 = p.fdiv("q0", uh, dhi)
    res = p.fsub(
        "res0",
        p.fadd("rn", p.fma("rn0", q0, p.fmul("ndhi", dhi, -1.0), uh), ul),
        p.fmul("qdl", q0, dlo),
    )
    mag = p.fma("q", res, p.fdiv("rd", 1.0, dhi), q0)
    mag = p.select("big", p.fcmp_k("isbig", "ogt", ax, 22.0), 1.0, mag)
    mag = p.select("tiny", p.fcmp_k("istiny", "olt", ax, _hexf("0x1p-27")), ax, mag)
    sign = p.and_i64("sign", p.to_bits("xb", _X), -(1 << 63))
    bits = p.or_i64("sb", p.to_bits("mb", mag), sign)
    p.from_bits("res", bits)
    return p.done()


def _build_sqrt() -> Program:
    p = _Builder("sqrt")
    p.sqrt("res", _X)
    return p.done()


def _build_rsqrt() -> Program:
    p = _Builder("rsqrt")
    # Below 2^-900 the residual x - s^2 is not exact in f64: scale by 2^256
    # first (the result by 2^128 after).
    tiny = p.fcmp_k("istiny", "olt", _X, _hexf("0x1p-900"))
    xs = p.select("xs", tiny, p.fmul("xsc", _X, _hexf("0x1p256")), _X)
    s = p.sqrt("s", xs)
    r0 = p.fdiv("r0", 1.0, s)
    # 1/sqrt(x) = r0 * (1 + e - sres / (2 x)) with e = 1 - r0*s and
    # sres = x - s^2 (both exact by fma): one correction of the two roundings.
    e = p.fma("e", p.fmul("nr0", r0, -1.0), s, 1.0)
    sres = p.fma("sres", p.fmul("ns", s, -1.0), s, xs)
    half = p.fmul("h", p.fmul("hs", sres, 0.5), r0)
    corr = p.fsub("corr", e, p.fmul("hr", half, r0))
    val = p.fma("v", r0, corr, r0)
    val = p.fmul("vs", val, p.select("sc", tiny, _hexf("0x1p128"), 1.0))
    # x = +-0: +-inf (r0); x = +inf: 0 (the correction would be NaN).
    val = p.select("inf", p.fcmp_k("isinf", "oeq", _X, math.inf), 0.0, val)
    p.select("res", p.fcmp_k("iszero", "oeq", _X, 0.0), r0, val)
    return p.done()


#: Ops expanded for f64, keyed by IR op name.
PROGRAMS: Dict[str, Program] = {
    "math.exp": _build_exp(),
    "math.exp2": _build_exp2(),
    "math.log": _build_log(),
    "math.log2": _build_log2(),
    "math.tanh": _build_tanh(),
    "math.sqrt": _build_sqrt(),
    "math.rsqrt": _build_rsqrt(),
}


def render(prog: Program, x: str, names: Sequence[str]) -> List[str]:
    """Instruction lines (without indentation) for input text `x` and step names."""
    out = []
    for step, name in zip(prog.steps, names):
        text = step.text
        for i in range(len(step.args) - 1, -1, -1):
            a = step.args[i]
            text = text.replace(f"${i}", x if a < 0 else names[a])
        out.append(f"{name} = {text}")
    return out


# ---------------------------------------------------------------- host model


def _fma_exact(a: float, b: float, c: float) -> float:
    if any(math.isnan(v) for v in (a, b, c)):
        return math.nan
    if any(math.isinf(v) for v in (a, b, c)):
        return a * b + c
    exact = Fraction(a) * Fraction(b) + Fraction(c)
    if exact == 0:
        # IEEE: exact zero sum takes +0 unless both terms are -0.
        prod_neg = (math.copysign(1.0, a) * math.copysign(1.0, b)) < 0
        if (a * b == 0) and c == 0 and prod_neg and math.copysign(1.0, c) < 0:
            return -0.0
        return 0.0
    try:
        return float(exact)
    except OverflowError:
        return math.inf if exact > 0 else -math.inf


def _rint(x: float) -> float:
    if math.isnan(x) or math.isinf(x):
        return x
    r = float(round(x))
    return math.copysign(r, x) if r == 0 else r


def _minnum(a: float, b: float) -> float:
    if math.isnan(a):
        return b
    if math.isnan(b):
        return a
    return min(a, b)


def _maxnum(a: float, b: float) -> float:
    if math.isnan(a):
        return b
    if math.isnan(b):
        return a
    return max(a, b)


def _fcmp(pred: str, a: float, b: float) -> bool:
    unordered = math.isnan(a) or math.isnan(b)
    if pred == "uno":
        return unordered
    if unordered:
        return False
    return {
        "olt": a < b,
        "ole": a <= b,
        "ogt": a > b,
        "oge": a >= b,
        "oeq": a == b,
        "one": a != b,
    }[pred]


def _ldexp(a: float, e: int) -> float:
    try:
        return math.ldexp(a, e)
    except OverflowError:
        return math.copysign(math.inf, a)


def _to_bits(x: float) -> int:
    return struct.unpack("<q", struct.pack("<d", x))[0]


def _from_bits(b: int) -> float:
    return struct.unpack(
        "<d", struct.pack("<q", ((b + (1 << 63)) % (1 << 64)) - (1 << 63))
    )[0]


def _wrap32(v: int) -> int:
    return ((v + (1 << 31)) % (1 << 32)) - (1 << 31)


def _div(a: float, b: float) -> float:
    try:
        return a / b
    except ZeroDivisionError:
        if math.isnan(a) or a == 0:
            return math.nan
        neg = (math.copysign(1.0, a) * math.copysign(1.0, b)) < 0
        return -math.inf if neg else math.inf


def _mul(a: float, b: float) -> float:
    return a * b


def _sqrt(a: float) -> float:
    if math.isnan(a) or a < 0:
        return math.nan
    return math.sqrt(a)


def evaluate(prog: Program, x: float) -> float:
    """Run ``prog`` on the host with IEEE f64 semantics (exact fma, RNE)."""
    vals: List[object] = []

    def arg(i: int):
        return x if i < 0 else vals[i]

    for st in prog.steps:
        a = [arg(i) for i in st.args]
        k = st.kind
        if k in ("fmul", "fadd", "fsub", "fdiv"):
            v = _binop(k, a[0], a[1])
        elif k.endswith("_k") and k[:-2] in ("fmul", "fadd", "fsub", "fdiv"):
            v = _binop(k[:-2], a[0], st.imm[0])
        elif k.endswith("_kl"):
            v = _binop(k[:-3], st.imm[0], a[0])
        elif k == "fma":
            it = iter(a)
            ops = [next(it) if im is None else im for im in st.imm]
            v = _fma_exact(*ops)
        elif k == "fabs.f64":
            v = abs(a[0])
        elif k == "rint.f64":
            v = _rint(a[0])
        elif k == "sqrt.f64":
            v = _sqrt(a[0])
        elif k == "minnum_k":
            v = _minnum(a[0], st.imm[0])
        elif k == "maxnum_k":
            v = _maxnum(a[0], st.imm[0])
        elif k == "ldexp":
            v = _ldexp(a[0], a[1])
        elif k == "ldexp_k":
            v = _ldexp(st.imm[0], a[0])
        elif k == "frexp":
            m, e = math.frexp(a[0])
            if math.isinf(a[0]) or math.isnan(a[0]):
                e = 0
            v = (m, e)
        elif k == "extract":
            v = a[0][st.imm[0]]
        elif k == "fcmp_k":
            v = _fcmp(st.imm[0], a[0], st.imm[1])
        elif k == "fcmp":
            v = _fcmp(st.imm[0], a[0], a[1])
        elif k == "select":
            it = iter(a[1:])
            ta, fa = [next(it) if im is None else im for im in st.imm]
            v = ta if a[0] else fa
        elif k == "fptosi":
            if math.isnan(a[0]) or abs(a[0]) >= 2**31:
                raise ValueError("fptosi out of range: poison")
            v = int(a[0])
        elif k == "sitofp":
            v = float(a[0])
        elif k == "add_i32":
            v = _wrap32(a[0] + st.imm[0])
        elif k == "add2_i32":
            v = _wrap32(a[0] + a[1])
        elif k == "to_bits":
            v = _to_bits(a[0])
        elif k == "from_bits":
            v = _from_bits(a[0])
        elif k == "and_i64":
            v = a[0] & st.imm[0]
        elif k == "or_i64":
            v = a[0] | a[1]
        else:
            raise NotImplementedError(k)
        vals.append(v)
    return vals[-1]


def _binop(k: str, a: float, b: float) -> float:
    if k == "fmul":
        return _mul(a, b)
    if k == "fadd":
        return a + b
    if k == "fsub":
        return a - b
    return _div(a, b)


__all__ = [
    "PROGRAMS",
    "Program",
    "Step",
    "coefficient_table",
    "evaluate",
    "f64_literal",
    "render",
]
