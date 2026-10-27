# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Fused operand prologues for the universal GEMM body (a spec side channel).

A :class:`Prologue` is attached to a :class:`UniversalGemmSpec` with
:func:`attach_prologue` (``spec._prologue``, the same frozen-dataclass side
channel as ``_fused_epilogue``). Without it the GEMM emission is unchanged:
:func:`build_universal_gemm` only reads the attribute and every hook below is
skipped, so default kernels stay byte-identical (no TraitSpec field). The C++
engine mirrors it (``cpp/instances/common/gemm_prologue.cpp``,
``rocke_gemm_universal_spec_t._prologue``) except the operand ops
(``unary_a`` / ``unary_b``), ``stage="vgpr"`` and ``epi``, which it refuses;
the ``gemm_cf32`` parity family keeps the two engines byte-identical.

What a prologue does, per operand fragment, between the LDS ``ds_read`` and the
MFMA:

* ``src="f32"`` -- **f32 operands in global memory, 16-bit compute.** The f32
  ``[rows, K]`` matrix is the bit-identical 16-bit ``[rows, 2K]`` matrix (each
  f32 = its low and high halves, little-endian), so the unmodified
  direct-to-LDS ring copies it with the 16-bit addressing (the kernel's ``K``
  argument is ``2 K``; ``tile_k`` counts 16-bit halves, i.e. ``tile_k / 2`` f32
  per K tile). Each lane reads ``2 x frag_len`` halves (= ``frag_len`` f32),
  bit-casts them to f32 and rounds them on the fly (RNE, ``v_cvt_pk_bf16_f32``)
  to the compute type. ``recipe`` selects:

  * ``cvt``  one part, one MFMA: hipTensor ``f32/f32/bf16`` (or ``/f16``);
  * ``x3``   2-part split (hi + lo), 3 products (bf16s2p3, the TF32 class);
  * ``x4``   2 parts, 4 products;
  * ``x6``   3-part split, 6 products i + j <= 2 (bf16s3p6, 32F class);
  * ``x9``   3 parts, 9 products (exact products).

  The split is Dekker-style and exact in f32: ``p0 = RN(x)``,
  ``r1 = x - p0`` (exact), ``p1 = RN(r1)``, ``p2 = RN(r1 - p1)``.
* ``unary_a`` / ``unary_b`` -- an elementwise operator applied to the operand
  before the MFMA (hipTensor opA / opB): ``relu``, ``abs``, ``neg``, ``exp``,
  ``sq`` (x*x), ``sigmoid``. For 16-bit operands the op runs in f32 and is rounded
  once to the compute type (``unary_impl="f32"``) or, for ``abs`` / ``neg`` /
  ``relu``, on the packed 16-bit lanes (``unary_impl="bits"``: ``v_and`` /
  ``v_xor`` / ``v_pk_max_i16``; exact, the same result). For f32 operands the op
  runs in f32 before the split.
* ``out_f32`` -- the direct epilogue writes the f32 accumulators (hipTensor
  ``To = f32``) instead of rounding them to the storage type.

Knobs (all kept; see :class:`Prologue`): ``split_at`` (convert in the loader or
at the first MFMA use), ``order`` (small terms first or big first), ``cap``
(saturate the hi part at the largest finite bf16 so ``|x| >= 0x7F7F8000``
does not become ``inf - inf = NaN``).

Roles on gfx950: the ``x6`` split (f32 contractions at f32-class accuracy),
the ``x3`` split (an opt-in TF32-class compute type) and ``cplx_b`` with
``x6`` (interleaved complex-f32 operands, 4M in one kernel) are the intended
paths. The ``cvt`` and unary prologues and ``stage="vgpr"`` are retained
alternatives (the separate conversion / unary pass they replace was not
slower). Bodies: the direct-to-LDS universal GEMM bodies without
``sia_sched`` / ``asm_loop`` (``build_universal_gemm`` rejects those).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ...core.ir import BF16, F32, I32, IRBuilder, Type, Value, VectorType

RECIPES: Dict[str, Tuple[int, Tuple[Tuple[int, int], ...]]] = {
    # name: (parts, products (i, j) = part i of A times part j of B), listed BIG FIRST
    "cvt": (1, ((0, 0),)),
    "x3": (2, ((0, 0), (0, 1), (1, 0))),
    "x4": (2, ((0, 0), (0, 1), (1, 0), (1, 1))),
    "x6": (3, ((0, 0), (0, 1), (1, 0), (0, 2), (1, 1), (2, 0))),
    "x9": (
        3,
        ((0, 0), (0, 1), (1, 0), (0, 2), (1, 1), (2, 0), (1, 2), (2, 1), (2, 2)),
    ),
}
UNARY = ("", "relu", "abs", "neg", "exp", "sq", "sigmoid")
# largest f32 whose RNE bf16 rounding is finite (0x7F7F7FFF)
_BF16_SAT = 3.3895313892515355e38


@dataclass(frozen=True)
class Prologue:
    src: str = "same"  # "same" (storage dtype) or "f32" (f32 in global, pair view)
    recipe: str = "cvt"
    unary_a: str = ""
    unary_b: str = ""
    unary_impl: str = "f32"  # "f32" | "bits" (abs / neg / relu on packed lanes)
    out_f32: bool = False
    cap: bool = False
    order: str = "small_first"  # "small_first" | "big_first"
    split_at: str = "load"  # "load" | "use"
    # "lds": f32 pairs staged by the DTL ring, converted after the ds_read (per wave);
    # "vgpr": f32 loaded to VGPRs, converted once per workgroup and written to LDS as the
    # compute type (the LDS tile and every fragment read are those of the 16-bit body)
    stage: str = "lds"
    pin_v: bool = (
        False  # pin every converted fragment to the VGPR file (asm "=v,0"): no AGPR detours
    )
    fast_exp: bool = (
        True  # unary exp via one v_exp_f32 (exp2_fast) instead of the guarded exp2
    )
    psub: bool = (
        False  # residual subtract as packed v_pk_add_f32 (inline asm) instead of v_sub_f32
    )
    # complex 4M in one kernel (interleaved cf32 operands through the f32 pair view):
    # B (complex [Nc, Kc], K contiguous) is read as the real embedding B' [2 Nc, 2 Kc] with
    # row 2n = (Bre, -s_a s_b Bim) and row 2n+1 = (s_b Bim, s_a Bre) (s = -1 under conj),
    # so D [M, 2 Nc] f32 = A_int [M, 2 Kc] . B'^T is the interleaved complex product.
    # The ring DMA loads complex row n into tile rows 2n and 2n+1 (addressing only); the
    # swap / sign is applied to the B fragment after the ds_read, before the split.
    cplx_b: bool = False
    conj_a: bool = False
    conj_b: bool = False
    # output scaling of an ``out_f32`` kernel (row-interleave vector epilogue):
    # "" (D = acc), "alpha" (D = alpha acc) or "bilinear" (D = alpha acc + beta Cin,
    # Cin with D's layout). Kernel params alpha_re / alpha_im / beta_re / beta_im (f32)
    # and Cin (f32, read-only); with ``cplx_b`` the scalars are complex and every
    # (even, odd) column pair of D / Cin is one complex value, else only the real parts
    # are used.
    epi: str = ""
    # ``agpr_out`` (out_f32, acc_agpr bodies): the f32 stores read the accumulators
    # straight from the AGPRs (``"=a,0"`` hand-out). Default off: they leave the
    # AGPRs through the compiler's own copy, whose MFMA hazard wait the compiler
    # inserts (the direct form gave wrong stream-K tiles on the device).
    agpr_out: bool = False

    def check(self) -> None:
        if self.src not in ("same", "f32"):
            raise ValueError(f"prologue src {self.src!r}")
        if self.recipe not in RECIPES:
            raise ValueError(f"prologue recipe {self.recipe!r}")
        if self.src != "f32" and self.recipe != "cvt":
            raise ValueError("split recipes need src='f32'")
        for u in (self.unary_a, self.unary_b):
            if u not in UNARY:
                raise ValueError(f"prologue unary {u!r}")
        if self.unary_impl not in ("f32", "bits"):
            raise ValueError(f"prologue unary_impl {self.unary_impl!r}")
        if self.order not in ("small_first", "big_first"):
            raise ValueError(f"prologue order {self.order!r}")
        if self.split_at not in ("load", "use"):
            raise ValueError(f"prologue split_at {self.split_at!r}")
        if self.stage not in ("lds", "vgpr"):
            raise ValueError(f"prologue stage {self.stage!r}")
        if self.stage == "vgpr" and (self.src != "f32" or self.recipe != "cvt"):
            raise ValueError("stage='vgpr' is built for src='f32', recipe='cvt'")
        if (self.conj_a or self.conj_b) and not self.cplx_b:
            raise ValueError("conj_a / conj_b need cplx_b")
        if self.cplx_b and (
            self.src != "f32" or self.stage != "lds" or self.split_at != "use"
        ):
            raise ValueError("cplx_b needs src='f32', stage='lds', split_at='use'")
        if self.epi not in ("", "alpha", "bilinear"):
            raise ValueError(f"prologue epi {self.epi!r}")
        if self.epi and not self.out_f32:
            raise ValueError("prologue epi (alpha / bilinear) needs out_f32")

    @property
    def kmul(self) -> int:
        return 2 if (self.src == "f32" and self.stage == "lds") else 1

    @property
    def parts(self) -> int:
        return RECIPES[self.recipe][0]

    def products(self) -> Tuple[Tuple[int, int], ...]:
        prods = RECIPES[self.recipe][1]
        if self.order == "small_first":
            # stable on level i + j, larger level first
            return tuple(sorted(prods, key=lambda ij: -(ij[0] + ij[1])))
        return prods

    def suffix(self) -> str:
        s = "_pro"
        if self.src == "f32":
            s += f"_f32{self.recipe}"
        if self.unary_a:
            s += f"_ua{self.unary_a}"
        if self.unary_b:
            s += f"_ub{self.unary_b}"
        if (self.unary_a or self.unary_b) and self.unary_impl != "f32":
            s += f"_ui{self.unary_impl}"
        if self.out_f32:
            s += "_of32"
        if self.cap:
            s += "_cap"
        if self.order != "small_first":
            s += "_bf"
        if self.split_at != "load":
            s += "_su"
        if self.psub:
            s += "_pk"
        if self.stage != "lds":
            s += "_vg"
        if not self.fast_exp and "exp" in (self.unary_a, self.unary_b):
            s += "_gexp"
        if self.pin_v:
            s += "_pv"
        if self.cplx_b:
            s += "_cx" + ("a" if self.conj_a else "") + ("b" if self.conj_b else "")
        if self.epi:
            s += "_e" + ("a" if self.epi == "alpha" else "ab")
        if self.agpr_out:
            s += "_ao"
        return s


def attach_prologue(spec, pro: Prologue):
    """Return ``spec`` with ``pro`` attached (``spec._prologue``)."""
    pro.check()
    object.__setattr__(spec, "_prologue", pro)
    return spec


def get_prologue(spec) -> Optional[Prologue]:
    return getattr(spec, "_prologue", None)


class PrologueEmitter:
    """Per-kernel emitter of the prologue fragment transforms and MFMA expansion."""

    def __init__(self, b: IRBuilder, pro: Prologue, compute: Type, frag_len: int):
        self.b = b
        self.pro = pro
        self.compute = compute
        self.n = frag_len
        self._memo: Dict[int, List[Value]] = {}
        # cplx_b: parity of the B' row a B fragment comes from. "ni": ni & 1 (row-interleaved
        # bodies: a lane's mf rows are consecutive); else the lane-parity i1 below.
        self.cplx_mode = "lane"
        self.cplx_lane_odd: Optional[Value] = None

    # ---------------------------------------------------------------- scalar helpers
    def _unary_f32(self, x: Value, u: str) -> Value:
        b = self.b
        if u == "relu":
            return b.fmax(x, b.const_f32(0.0))
        if u == "abs":
            return b.fabs(x)
        if u == "neg":
            return b.fneg(x)
        if u == "exp":
            # one v_exp_f32 (no range-reduction guard: overflow gives inf, as exp does)
            e = b.fmul(x, b.const_f32(1.0 / math.log(2.0)))
            return b.exp2_fast(e) if self.pro.fast_exp else b.exp2(e)
        if u == "sq":
            return b.fmul(x, x)
        if u == "sigmoid":
            e = b.exp2(b.fmul(x, b.const_f32(-1.0 / math.log(2.0))))
            return b.fdiv(b.const_f32(1.0), b.fadd(b.const_f32(1.0), e))
        raise ValueError(u)

    def _to_f32_vec(self, v: Value) -> Value:
        """Exact widening of a packed ``<n x bf16>`` part to ``<n x f32>``.

        bf16 -> f32 is a 16-bit shift, done on the packed dwords (element 2i is
        the low half): ``w << 16`` and ``w & 0xFFFF0000``, one VALU per element
        and no re-rounding of the f32 source (LLVM would otherwise rebuild
        ``fpext(fptrunc(x))`` per element from the f32 source). For f16 parts
        (unscaled fp16 splits) the per-element ``fpext`` is used.
        """
        b = self.b
        n = v.type.count
        if v.type.elem != BF16 or n % 2:
            return b.vec_pack(
                [b.cast_to_f32(b.vec_extract(v, i)) for i in range(n)], F32
            )
        w = b.vec_bitcast(v, VectorType(I32, n // 2))
        c16 = b.const_i32(16)
        chi = b.const_i32(-65536)  # 0xFFFF0000
        outs = []
        for i in range(n // 2):
            d = b.vec_extract(w, i)
            outs.append(b.bitcast(b.shl(d, c16), F32))
            outs.append(b.bitcast(b.land(d, chi), F32))
        return b.vec_pack(outs, F32)

    def _slice(self, v: Value, lo: int, n: int) -> Value:
        b = self.b
        return b.vec_pack([b.vec_extract(v, lo + i) for i in range(n)], v.type.elem)

    # ---------------------------------------------------------------- 16-bit unary
    def _unary16(self, v: Value, u: str) -> Value:
        """Elementwise ``u`` on a ``<n x bf16/f16>`` fragment, rounded once."""
        b = self.b
        n = v.type.count
        if self.pro.unary_impl == "bits" and u in ("abs", "neg", "relu"):
            w = b.vec_bitcast(v, VectorType(I32, n // 2))
            outs = []
            for i in range(n // 2):
                d = b.vec_extract(w, i)
                if u == "abs":
                    d = b.land(d, b.const_i32(0x7FFF7FFF))
                elif u == "neg":
                    d = b.xor(d, b.const_i32(-0x7FFF8000))  # 0x80008000
                else:
                    d = b.inline_asm(
                        "v_pk_max_i16 $0, $1, 0", "=v,v", [d], I32, sideeffect=False
                    )
                outs.append(d)
            return b.vec_bitcast(b.vec_pack(outs, I32), v.type)
        xs = [self._unary_f32(b.cast_to_f32(b.vec_extract(v, i)), u) for i in range(n)]
        return b.vec_cast_f32_to(b.vec_pack(xs, F32), v.type.elem)

    # ---------------------------------------------------------------- f32 split
    def _split(self, x: Value) -> List[Value]:
        """``<n x f32>`` -> ``parts`` fragments of the compute type (RNE)."""
        b = self.b
        P = self.pro.parts
        hi_src = x
        if self.pro.cap and self.compute == BF16:
            n = x.type.count
            sat = b.const_f32(_BF16_SAT)
            nsat = b.const_f32(-_BF16_SAT)
            hi_src = b.vec_pack(
                [b.fmax(b.fmin(b.vec_extract(x, i), sat), nsat) for i in range(n)],
                F32,
            )
        parts = [b.vec_cast_f32_to(hi_src, self.compute)]
        r = x
        for _p in range(1, P):
            r = self._sub(r, self._to_f32_vec(parts[-1]))
            parts.append(b.vec_cast_f32_to(r, self.compute))
        return parts

    def _sub(self, x: Value, y: Value) -> Value:
        """``x - y`` on ``<n x f32>`` (exact here: y is the RNE head of x)."""
        b = self.b
        if not self.pro.psub or x.type.count % 2:
            return b.fsub(x, y)
        v2 = VectorType(F32, 2)
        outs = []
        for i in range(0, x.type.count, 2):
            xa = b.vec_pack([b.vec_extract(x, i), b.vec_extract(x, i + 1)], F32)
            ya = b.vec_pack([b.vec_extract(y, i), b.vec_extract(y, i + 1)], F32)
            d = b.inline_asm(
                "v_pk_add_f32 $0, $1, $2 neg_lo:[0,1] neg_hi:[0,1]",
                "=v,v,v",
                [xa, ya],
                v2,
                sideeffect=False,
            )
            outs += [b.vec_extract(d, 0), b.vec_extract(d, 1)]
        return b.vec_pack(outs, F32)

    def _pin(self, v: Value) -> Value:
        if not self.pro.pin_v:
            return v
        return self.b.inline_asm("", "=v,0", [v], v.type, sideeffect=False)

    def _convert_raw(self, raw: Value, unary: str, cpar=None) -> List[Value]:
        """Parts of one raw fragment (the ds_read result)."""
        return [self._pin(p_) for p_ in self._convert_raw0(raw, unary, cpar)]

    def _cplx_fix(self, x: Value, par) -> Value:
        """B' row of a complex B fragment: ``par`` 0 -> (re, -sa sb im), 1 -> (sb im, sa re).

        ``par`` is a Python int (static per fragment) or an i1 Value (per lane)."""
        b = self.b
        sa = -1 if self.pro.conj_a else 1
        sb = -1 if self.pro.conj_b else 1

        def sg(v: Value, s: int) -> Value:
            return b.fneg(v) if s < 0 else v

        outs = []
        for i in range(0, self.n, 2):
            x0, x1 = b.vec_extract(x, i), b.vec_extract(x, i + 1)
            if isinstance(par, int):
                if par:
                    outs += [sg(x1, sb), sg(x0, sa)]
                else:
                    outs += [x0, sg(x1, -sa * sb)]
                continue
            outs += [
                b.select(par, sg(x1, sb), x0),
                b.select(par, sg(x0, sa), sg(x1, -sa * sb)),
            ]
        return b.vec_pack(outs, F32)

    def _convert_raw0(self, raw: Value, unary: str, cpar=None) -> List[Value]:
        b = self.b
        if self.pro.src == "f32":
            x = b.vec_bitcast(raw, VectorType(F32, self.n))
            if unary:
                x = b.vec_pack(
                    [
                        self._unary_f32(b.vec_extract(x, i), unary)
                        for i in range(self.n)
                    ],
                    F32,
                )
            if cpar is not None:
                x = self._cplx_fix(x, cpar)
            return self._split(x)
        if unary:
            return [self._unary16(raw, unary)]
        return [raw]

    # ---------------------------------------------------------------- public hooks
    def raw_len(self) -> int:
        """16-bit elements one lane reads per fragment."""
        return self.n * self.pro.kmul

    @property
    def vgpr_stage(self) -> bool:
        return self.pro.stage == "vgpr"

    def stage_convert(self, x: Value, which: str) -> Value:
        """``stage='vgpr'``: one lane's ``<n x f32>`` global chunk -> the LDS chunk."""
        b = self.b
        unary = self.pro.unary_a if which == "A" else self.pro.unary_b
        if unary:
            n = x.type.count
            x = b.vec_pack(
                [self._unary_f32(b.vec_extract(x, i), unary) for i in range(n)], F32
            )
        return b.vec_cast_f32_to(x, self.compute)

    def after_load(self, raw: Value, which: str) -> Value:
        """Loader hook: with ``split_at='load'`` the fragment Value carried to the
        MFMA is the concatenation of its parts (loop-carry safe)."""
        unary = self.pro.unary_a if which == "A" else self.pro.unary_b
        if self.vgpr_stage:
            return raw  # converted (and op applied) when it was staged
        if self.pro.split_at == "use":
            if self.pro.src != "f32" and not unary:
                return raw
            # tag the raw value with its operand so the use site applies the right op
            self._memo[id(raw)] = None  # type: ignore[assignment]
            self._which = getattr(self, "_which", {})
            self._which[id(raw)] = which
            return raw
        parts = self._convert_raw(raw, unary)
        v = parts[0]
        for p in parts[1:]:
            v = self.b.vec_concat(v, p)
        return v

    def _parts_of(self, frag: Value, which: str, ni=None) -> List[Value]:
        key = id(frag)
        hit = self._memo.get(key)
        if hit:
            return hit
        P = self.pro.parts
        if self.pro.split_at == "use" and (
            self.pro.src == "f32"
            or (self.pro.unary_a if which == "A" else self.pro.unary_b)
        ):
            unary = self.pro.unary_a if which == "A" else self.pro.unary_b
            cpar = None
            if self.pro.cplx_b and which == "B":
                if self.cplx_mode == "ni":
                    if ni is None:
                        raise ValueError(
                            "cplx_b (row-interleaved body): MFMA site without ni"
                        )
                    cpar = int(ni) & 1
                else:
                    if self.cplx_lane_odd is None:
                        raise ValueError("cplx_b: lane parity not set")
                    cpar = self.cplx_lane_odd
            parts = self._convert_raw(frag, unary, cpar)
        elif P == 1:
            parts = [frag]
        else:
            parts = [self._slice(frag, p * self.n, self.n) for p in range(P)]
        self._memo[key] = parts
        return parts

    def mma(
        self,
        emit_mma,
        a_frag: Value,
        b_frag: Value,
        acc: Value,
        swap: bool,
        pin,
        ni=None,
    ):
        """All products of the recipe into ``acc`` (one accumulator)."""
        pa = self._parts_of(a_frag, "A")
        pb = self._parts_of(b_frag, "B", ni)
        for i, j in self.pro.products():
            if swap:
                acc = emit_mma(pb[j], pa[i], acc)
            else:
                acc = emit_mma(pa[i], pb[j], acc)
            if pin is not None:
                acc = pin(acc)
        return acc
