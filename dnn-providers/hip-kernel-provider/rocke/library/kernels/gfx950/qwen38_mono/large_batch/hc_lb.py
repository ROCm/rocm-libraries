# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""WP-L2b: the large-batch HC mixers (DESIGN-LB 4.2 S1 / S2 / S6 / S7, emitter E2), gfx950.

One HC mixer (attention or MLP; the same code with other weights) at rows 5-128 in ONE persistent launch per step
(256 CTAs, runtime layer loop on W5's ``persist`` platform, grid edges, residency ticket, launch-fatal word), built on
WP-L2a's E1 GEMM core (:mod:`.e1`, imported, never edited) and on WP-L2c's composition helpers (:mod:`.gdn_lb`'s
``_Scope`` / ``_FreshScope``, imported). Stages per HC instance (ATOM's numerics: MXFP4 checkpoint weights, MXFP8
activations; ``kernels/stages/hc.py`` ``a1`` / ``a2`` and the combine of ``out_proj.py`` / ``moe.py``):

* **C** (combine + norm + pack + injection; FlyTape ``mix_combine`` plus the TwoStage source normalisation of its
  ``mix_down`` and the gate rows of its ``mix_reduce``): per (token b, stream i), ``cw`` waves:
  ``r'_k = bf16(r_k + bf16(y_k inj_i))`` (TP2: ``y = bf16(y_0 + y_1)``, rank order), the stream's square sum,
  ``inv_i = f32(1 / sqrt_f64(ss fl(1/2560) + eps))`` (ATOM's rule), the down GEMM's activations
  ``MXFP8(n)`` in E1's op_sel layout with ``n_k = bf16((r'_k inv_i) w'_k)`` (``norm="faithful"``, ATOM / FlyTape order)
  or ``n_k = bf16(r'_k w'_k)`` (``norm="reassoc"``: the per-stream re-association, ``inv_i`` applied after the GEMM by
  E1's ``kseg``), and the next combine's gates ``g_j = bf16(2 bf16(sigmoid(bf16(bf16(t_j) / 4))))``,
  ``t_j = sum_i sum_k w_inj[j][i H + k] n_k`` (bf16 injection weights, ATOM's precision; FlyTape folds them into its
  324-row down GEMM). Outputs ``r'`` (the next HC's residual), ``inv``, the gates, the MXFP8 activations.
* **D** (the down projection, FlyTape ``mix_down`` + ``mix_reduce``): an E1 build (``k = 10240``, ``M = 320``,
  MXFP4 x MXFP8, any WP-L2a plan: split-K with the last-arriver fold or ``kw``) whose ``yq`` producer epilogue applies
  ATOM's activation ``bf16(s / (1 + exp(-s)))``, ``s = bf16(bf16(y) / 4)``, before the MXFP8 pack: the up GEMM's
  low-rank operand in the op_sel layout (no separate reduce phase; the fold is the last arriver's).
* **U** (the up projection + mix, FlyTape ``mix_up``): an E1 build (``k = 320``, ``M = 10240``) over the up weights with
  their rows permuted host-side to ``4 col + stream`` (``pack_up_perm``), so the MFMA lane that holds rows
  ``4 g .. 4 g + 3`` holds the 4 streams of one column: the mix is in registers (FlyTape sums its 4 stream waves
  through LDS). Epilogue (ATOM ``a2``): ``x = bf16((((p_0 + p_1) + p_2) + p_3) / 4)``,
  ``p_i = bf16(bf16(sigmoid(bf16(acc_i))) bf16((r'_i inv_i) w'_i))``; ``xq_pass`` writes bf16 ``x`` and its MXFP8 in the
  next GEMM's op_sel layout (blocks of 32 columns: units of ``R = 16 wr mt`` rows with ``R % 128 == 0``).

Edges C -> D -> U -> next layer are W5's striped grid sync (WP-L1's measured idle winner; per-edge flags are WP-L3's).
The layer chain is real: layer ``l + 1``'s C reads layer ``l``'s ``r'`` and gates.

Knobs (:class:`HcKnobs`; each default-off lever in the kernel name): ``norm``, ``waves`` (4 | 8: every stage's waves),
the C map ``CKnobs.cw`` / ``inv``, the D and U E1 plans (any E1 lever), ``epre`` (U's epilogue operands issued at the
unit's start, FlyTape's ``mix_up`` order), ``pre`` (D / U weight chunks issued between an edge's arrival and wait),
``acq`` / ``ticket`` / ``lpad`` (WP-L2c's), ``stages`` (ablation), ``dbg`` (U also stores its f32 accumulators).

Mutants (must fail): C ``ygate`` (gate not applied), ``eps`` (eps dropped), ``xblk`` (stream 1's MXFP8 scales one binade
high), ``inj`` (gates from streams 0-2 only), ``rskip`` (``r'`` not stored); D ``silu`` (activation skipped) and E1's own
(``kskip``, ``fold``, ``seg``, ``kwred``) through the D knobs; U ``mix`` (stream 3 dropped), ``sig`` (sigmoid skipped),
``xs`` (``x`` scales one binade high) and E1's through the U knobs; chain ``early`` (every edge released one arrival
early while CTA 0 starts every stage late: the deterministic stale-read twin; ``delay`` alone is its positive
control), ``noacq`` (informational).

Python engine only (``buffer_*_cp``, ``int_to_ptr``, ``math.exp``, f64 sqrt); names ``q38lbh_*``; nothing served imports
this module. Measurements: ``experiments/qwen38-true-mono/large-batch/WP-L2b.md``.
"""

from __future__ import annotations

import contextlib
import hashlib
from dataclasses import dataclass, fields, replace

from rocke.core.ir import CPOL_SC1, F32, I8, I32, I64, IRBuilder, KernelDef, PtrType, Value, VectorType
from rocke.helpers import device_ops as D
from rocke.helpers import mx_quant as MQ

from ..persist import sync as PS
from . import e1 as E1M
from . import e1_plans as EP
from . import gdn_lb as G

__all__ = [
    "HID",
    "NS",
    "HCW",
    "LR",
    "CKnobs",
    "HcKnobs",
    "build_hc_kernel",
    "hc_kernel_name",
    "hc_params",
    "hc_lds_bytes",
    "default_d",
    "default_u",
    "pack_up_perm",
    "up_perm",
    "GS_BYTES",
    "CNT_LAYER",
]

HID = 2560
NS = 4
HCW = NS * HID  # 10240
LR = 320
TILE = 16
CHUNK = 128
GRID = 256
OOB = 0x7FFFFFF0
NV = HID // 8  # 320 vectors of 8 bf16 per (token, stream)
INV_HIDDEN = 0x39CCCCCD  # fl(1/2560)
EPS_BITS = 0x358637BD  # fl(1e-6)
BAR_LINE = G.BAR_LINE
BAR_WORDS = G.BAR_WORDS
GS_BYTES = G.GS_BYTES
CNT_LAYER = G.CNT_LAYER
CTL_WORDS = 16
NQ4_X = -(-(HCW // CHUNK) // 4)  # 20: op_sel scale groups of the down GEMM's K
NQ4_A = 1  # the up GEMM's K (320 -> 3 chunks)
NQ4_O = -(-(HID // CHUNK) // 4)  # 5: the next GEMM's K (2560)

C_MUTANTS = ("none", "ygate", "eps", "xblk", "inj", "rskip")
D_MUTANTS = ("none", "silu")
U_MUTANTS = ("none", "mix", "sig", "xs")
CHAIN_MUTANTS = ("none", "early", "noacq")
CP_VALUES = (1, 2, 4, 5, 10)
SP_MAX = 10 * NS * 5  # per token: square sums [cp][NS] + injection dots [cp][NS][NS] (f32), cp <= 10
SPF_MAX = 10 * 80 * 5  # cfuse, per token: [row group <= 10][split x K group <= 80][ss + 4 dots] (f32)
LDS_MAX = 160 * 1024 - 256

_c, _f, _ptr, _rsrc, _i64, _addl = G._c, G._f, G._ptr, G._rsrc, G._i64, G._addl


def _bf16r(b, x: Value) -> Value:
    return b.cast_to_f32(b.trunc_f32_to_bf16(x))


def _lo(b, w: Value) -> Value:
    return b.bitcast(b.shl(w, _c(b, 16)), F32)


def _hi(b, w: Value) -> Value:
    return b.bitcast(b.land(w, _c(b, -65536)), F32)


def _half(b, w: Value, odd: Value) -> Value:
    """The bf16 of dword ``w`` at element parity ``odd`` (1 = the high half), as f32."""
    return b.select(b.cmp_ne(odd, _c(b, 0)), _hi(b, w), _lo(b, w))


def _sigmoid(b, x: Value, sig: str = "acc") -> Value:
    """``sig="acc"``: ATOM ``sigmoid_acc``, ``1 / (1 + expf(0 - x))`` (correctly rounded divide, accurate exp);
    ``"fast"``: ATOM ``sigmoid_fast`` (``v_rcp(1 + v_exp((0 - x) log2 e))``, tolerance class)."""
    if sig == "fast":
        return G._sigmoid(b, x)
    return b.fdiv(_f(b, 1.0), b.fadd(_f(b, 1.0), b.exp(b.fsub(_f(b, 0.0), x))))


def _silu(b, s: Value, sig: str = "acc") -> Value:
    """``acc``: ATOM's ``s / (1 + expf(0 - s))``; ``fast``: ``s sigmoid_fast(s)``."""
    if sig == "fast":
        return b.fmul(s, G._sigmoid(b, s))
    return b.fdiv(s, b.fadd(_f(b, 1.0), b.exp(b.fsub(_f(b, 0.0), s))))


def _pack8(b, vals) -> Value:
    """8 f32 (bf16-exact) -> <4 x i32> of 8 bf16 (element 0 in the low half of dword 0)."""
    return b.vec_bitcast(b.vec_trunc_f32_to_bf16(b.vec_pack(list(vals), F32)), VectorType(I32, 4))


def _pkw(b, v2: Value) -> Value:
    """<2 x f32> -> one i32 of 2 bf16 (RNE; element 0 in the low half): v_cvt_pk_bf16_f32."""
    return b.vec_extract(b.vec_bitcast(b.vec_trunc_f32_to_bf16(v2), VectorType(I32, 1)), 0)


def _unpk(b, w: Value) -> Value:
    return b.vec_pack([_lo(b, w), _hi(b, w)], F32)


def _dot2(b, a: Value, c: Value, acc: Value) -> Value:
    """``acc + a.lo c.lo + a.hi c.hi`` over bf16 pairs (v_dot2_f32_bf16)."""
    return b.inline_asm("v_dot2_f32_bf16 $0, $1, $2, $3", "=v,v,v,v", [a, c, acc], F32)


def _wsum(b, v: Value) -> Value:
    """Wave sum, butterfly (xor 1, 2, .., 32): every lane gets the same deterministic total."""
    for off in (1, 2, 4, 8, 16, 32):
        v = b.fadd(v, b.warp_shuffle_xor(v, off))
    return v


def _suffix(obj, d, skip=("tag", "c", "d", "u")) -> str:
    return G._suffix(obj, d, skip=skip)


# ------------------------------------------------------------------ knobs


@dataclass(frozen=True)
class CKnobs:
    """The C stage (combine + norm + pack + injection)."""

    cw: int = 1  # waves per (token, stream): a CTA unit holds waves / (4 cw) tokens
    inv: str = "f64"  # inverse RMS: "f64" (ATOM: 1 / sqrt in f64) | "rsq" (v_rsq_f32, tolerance class)
    mutant: str = "none"

    def check(self, waves: int) -> None:
        if self.cw not in (1, 2) or waves % (4 * self.cw):
            raise ValueError("C: cw 1 | 2 with 4 cw dividing the waves")
        if self.inv not in ("f64", "rsq"):
            raise ValueError("C: inv f64 | rsq")
        if self.mutant not in C_MUTANTS:
            raise ValueError(f"C mutant in {C_MUTANTS}")


def _adapt_kw(k: E1M.E1Knobs, mult: int) -> E1M.E1Knobs:
    """The nearest build with ``splits * kw`` a multiple of ``mult`` (kseg): same K parallelism first, then the kw /
    split change that keeps the rows per unit a multiple of 32 (``yq``), raising ``mt`` when ``kw`` thins the row waves."""
    import math

    if (k.splits * k.kw) % mult == 0:
        return k
    nq = k.nq
    cands = []
    for kw in (k.kw, 2 * k.kw, 4 * k.kw):
        if kw > k.waves or k.waves % kw:
            continue
        for s in range(1, nq + 1):
            if nq % (s * kw) or (s * kw) % mult:
                continue
            for mt in (k.mt, 2 * k.mt, 4 * k.mt):
                if mt > 4:
                    continue
                c = replace(k, splits=s, kw=kw, mt=mt)
                try:
                    c.check()
                except ValueError:
                    continue
                cost = (abs(math.log((s * kw) / (k.splits * k.kw))), kw != k.kw, mt != k.mt, s)
                cands.append((cost, c))
                break
    if not cands:
        raise ValueError("no kseg-compatible split")
    return min(cands, key=lambda t: t[0])[1]


def default_d(rows: int, bucket: int, waves: int = 4, norm: str = "faithful") -> E1M.E1Knobs:
    """The down GEMM's E1 build: WP-L2a's ``hc_down`` pick for the row count, as a chain stage (the chain's barrier:
    ``bar="probe"``, no occupancy pad of its own), with the ``yq`` producer epilogue (the up GEMM's MXFP8 operand) and,
    for ``norm="reassoc"``, the per-stream segment scales (``kseg=4``)."""
    k = EP.plan("hc_down", 0, rows)
    k = replace(k, k=HCW, bucket=bucket, bar="probe", occ1=False, yq=True, tag="")
    if k.waves != waves:
        k = replace(k, waves=waves, kw=min(k.kw, waves))
    if norm == "reassoc":
        k = _adapt_kw(replace(k, kseg=4), 4)
    k.check()
    return k


def default_dfuse(rows: int, bucket: int, waves: int = 4) -> E1M.E1Knobs:
    """cfuse's down GEMM: the row count's WP-L2a ``hc_down`` pick with LDS staging (the fused C writes it), no cross-unit
    prefetch, no B read-ahead, and the stream-aligned split nearest the pick's K parallelism."""
    import math

    k = EP.plan("hc_down", 0, rows)
    k = replace(k, k=HCW, bucket=bucket, bar="probe", occ1=False, yq=True, kseg=0, xsrc="lds", xscale="glb",
                xslay="chunk", xu=False, bpre=0, fold="last", tag="")  # fmt: skip
    if k.waves != waves:
        k = replace(k, waves=waves, kw=min(k.kw, waves))
    best = None
    for kw in (k.kw, 1, 2, 4):
        if kw > k.waves or k.waves % kw:
            continue
        for sp in (4, 8, 20, 40):
            if 80 % (sp * kw):
                continue
            c = replace(k, splits=sp, kw=kw)
            while c.rows_per_unit % 32 and c.mt < 4:
                c = replace(c, mt=c.mt * 2)
            try:
                c.check()
            except ValueError:
                continue
            cost = (abs(math.log((sp * kw) / (k.splits * k.kw))), kw != k.kw, sp)
            if best is None or cost < best[0]:
                best = (cost, c)
    if best is None:
        raise ValueError("cfuse: no stream-aligned down plan")
    return best[1]


def default_dstat(rows: int, bucket: int, waves: int = 4) -> E1M.E1Knobs:
    """cstat="dfold"'s down GEMM: the row count's WP-L2a ``hc_down`` pick (its own X path) with kseg off and the
    stream-aligned split (a multiple of 4) nearest the pick's K parallelism."""
    import math

    k = EP.plan("hc_down", 0, rows)
    k = replace(k, k=HCW, bucket=bucket, bar="probe", occ1=False, yq=True, kseg=0, fold="last", tag="")
    if k.waves != waves:
        k = replace(k, waves=waves, kw=min(k.kw, waves))
    best = None
    for kw in (k.kw, 1, 2, 4):
        if kw > k.waves or k.waves % kw:
            continue
        for sp in (4, 8, 20, 40):
            if 80 % (sp * kw):
                continue
            c = replace(k, splits=sp, kw=kw)
            while c.rows_per_unit % 32 and c.mt < 4:
                c = replace(c, mt=c.mt * 2)
            try:
                c.check()
            except ValueError:
                continue
            cost = (abs(math.log((sp * kw) / (k.splits * k.kw))), kw != k.kw, sp)
            if best is None or cost < best[0]:
                best = (cost, c)
    if best is None:
        raise ValueError("cstat=dfold: no stream-aligned down plan")
    return best[1]


def default_u(rows: int, bucket: int, waves: int = 4) -> E1M.E1Knobs:
    """The up GEMM's E1 build: WP-L2a's ``hc_up`` pick for the row count as a chain stage, one split (the mix needs the
    complete accumulators), rows per unit raised to a multiple of 128 (32 columns x 4 streams: the x MXFP8 blocks)."""
    k = EP.plan("hc_up", 0, rows)
    k = replace(k, k=LR, bucket=bucket, bar="probe", occ1=False, splits=1, yq=False, kseg=0, fold="last", tag="")
    if k.waves != waves:
        k = replace(k, waves=waves, kw=min(k.kw, waves))
    while k.rows_per_unit % 128:
        if k.mt < 4:
            k = replace(k, mt=k.mt * 2)
        elif k.kw > 1:
            k = replace(k, kw=k.kw // 2)
        else:
            raise ValueError("up: no 128-row unit")
    k.check()
    return k


@dataclass(frozen=True)
class HcKnobs:
    tp: int = 1
    bucket: int = 128
    prow: int = 0  # row count whose WP-L2a picks seed the default D / U plans (0 = the bucket)
    waves: int = 4  # waves of every stage (the composed kernel's 64 * waves threads)
    norm: str = "faithful"  # "faithful" (normalise before the MXFP8 pack) | "reassoc" (inv after the GEMM, E1 kseg)
    c: CKnobs = CKnobs()
    d: E1M.E1Knobs | None = None
    u: E1M.E1Knobs | None = None
    epre: bool = True  # U: the epilogue operands (r', inv, w') issued at the unit's start (they land under the GEMM)
    stages: str = "CDU"
    ticket: bool = True
    acq: str = "load"  # as gdn_lb.ChainKnobs: "load" (sc1 loads of in-launch data) | "wave" | "cta"
    # weight chunks issued between an edge's arrival and its wait (L2 warming, values discarded): "d" the down GEMM's
    # at the C -> D edge, "u" the up GEMM's at the D -> U edge, "n" the NEXT layer's down GEMM at the layer's last edge;
    # any subset of "dun" (or "none")
    pre: str = "none"
    lpad: str = "entry"
    delay: bool = False
    dmut: str = "none"
    umut: str = "none"
    mutant: str = "none"
    dbg: bool = False  # U stores its f32 accumulators (the dbg twin)
    sig: str = "acc"  # sigmoid / silu of the gates, the down activation and the mix: "acc" (ATOM) | "fast" (tolerance)
    # where the norm statistics are finished: "cta" (the C unit holds a whole token: inv + gates in C) | "fold" (C's
    # wave units are (token, stream, part) with no cross-wave step; they write per-part square sums and injection dots,
    # every D unit finishes its stream's inv from them and the first D unit of a token block writes inv + gates; needs
    # norm="reassoc": C quantizes before inv is known)
    cstat: str = "cta"
    cp: int = 5  # cstat="fold": parts per (token, stream) (a wave unit = 320 / cp vectors of 8)
    # the C stage fused into the down GEMM's activation staging (DESIGN-LB S1): no C pass, no C -> D edge; needs
    # norm="reassoc" and a D plan with stream-aligned units (splits a multiple of 4), LDS staging, no cross-unit prefetch
    cfuse: bool = False
    # C's arithmetic (cstat="cta"): "scalar" | "pk" (combine, normalisation, square sums and dots on packed f32 pairs,
    # v_pk_*; the combine / normalisation bits equal, the square-sum and dot sums in pair order: tolerance class) |
    # "dot2" (square sums and dots by v_dot2_f32_bf16: measured 0.13 relative inverse-RMS error on hardware, refused
    # by the checker: kept as a recorded loser)
    cvec: str = "scalar"
    uvec: str = "scalar"  # U's mix epilogue: "scalar" | "pk" (packed f32 pairs: the same bits)
    # where the next combine's gates are computed (cstat="cta"): "c" (in C) | "d" / "u" (on the top CTAs during the D / U
    # stage, after their own units: off C's critical path; the same bits)
    gplace: str = "c"
    eload: str = "mi"  # U's epilogue operand loads: "mi" (inv per weight x token tile) | "t" (inv once per token tile)
    # diagnosis: thread 0 of every CTA stores s_memrealtime (low 32 bits, 100 MHz) at the stage / edge boundaries of every
    # layer into the ``du`` buffer (``[layer][CTA][16]`` i32: 0 layer start, 1 C end, 2 edge 1 out, 3 D end, 4 edge 2
    # out, 5 U end, 6 edge 3 out; inside the CTA's last D / U unit: 7 / 10 unit start, 8 / 11 GEMM done, 9 / 12
    # epilogue (fold, packs) done; D's split-K last arriver: 13 fold start (after the arrival), 14 fold done (before the
    # activation pack)); not with ``dbg``
    stamp: bool = False
    hz: bool = True
    tag: str = ""

    @property
    def tbk(self) -> int:
        return -(-self.bucket // TILE) * TILE

    def parts(self):
        rows = self.prow or self.bucket
        if self.cfuse and self.d is None:
            d = default_dfuse(rows, self.bucket, self.waves)
            return d, self.u or default_u(rows, self.bucket, self.waves)
        if self.cstat == "dfold" and self.d is None:
            return default_dstat(rows, self.bucket, self.waves), self.u or default_u(rows, self.bucket, self.waves)
        d = self.d or default_d(rows, self.bucket, self.waves, self.norm)
        u = self.u or default_u(rows, self.bucket, self.waves)
        return d, u

    def check(self) -> None:
        if self.tp not in (1, 2):
            raise ValueError("tp 1 | 2")
        if not 1 <= self.bucket <= 128:
            raise ValueError("bucket 1..128")
        if self.waves not in (4, 8):
            raise ValueError("waves 4 | 8")
        if self.norm not in ("faithful", "reassoc"):
            raise ValueError("norm faithful | reassoc")
        self.c.check(self.waves)
        d, u = self.parts()
        for name, k in (("D", d), ("U", u)):
            k.check()
            if k.waves != self.waves:
                raise ValueError(f"{name}: waves must be the chain's")
            if k.bar != "probe" or k.occ1:
                raise ValueError(f"{name}: in the chain the stage barrier is the chain's (bar=probe, occ1=False)")
            if k.bucket != self.bucket or k.grid != GRID:
                raise ValueError(f"{name}: bucket / grid must be the chain's")
            if k.w != "mx" or not k.sop or k.act != "fp8":
                raise ValueError(f"{name}: mx, fp8 activations, sop")
            if k.chain or k.stamp or k.tstamp or k.diag != "none" or k.mfma != 16:
                raise ValueError(f"{name}: no chain / stamps / diag / mfma=32 builds")
        if d.k != HCW or not d.yq or d.fold == "spread":
            raise ValueError("D: k = 10240 with the yq producer epilogue (fold last | none)")
        if self.cfuse:
            if self.norm != "reassoc" or d.kseg or not d.sop or d.xsrc != "lds" or d.xu or d.splits % 4 or d.bpre:
                raise ValueError("cfuse: norm=reassoc; D sop, xsrc=lds, no xu / kseg / bpre, splits a multiple of 4")
            if -(-LR // d.rows_per_unit) * d.splits * d.kw > SPF_MAX // 5:
                raise ValueError("cfuse: row groups x splits x K groups > 800")
        elif self.cstat == "dfold":
            if d.kseg or d.splits % 4 or d.fold != "last":
                raise ValueError("cstat=dfold: D kseg off, splits a multiple of 4 (stream-aligned), fold=last")
        elif d.kseg != (4 if self.norm == "reassoc" else 0):
            raise ValueError("D: kseg 4 exactly with norm=reassoc")
        if u.k != LR or u.splits != 1 or u.yq or u.kseg or u.rows_per_unit % 128:
            raise ValueError("U: k = 320, one split, no yq / kseg, rows per unit a multiple of 128")
        if set(self.stages) - set("CDU"):
            raise ValueError("stages: a subset of 'CDU'")
        if self.acq not in ("wave", "cta", "load"):
            raise ValueError("acq wave | cta | load")
        if self.cstat not in ("cta", "fold", "dfold", "cfold"):
            raise ValueError("cstat cta | fold | dfold | cfold")
        if self.cstat in ("fold", "dfold", "cfold") and (self.norm != "reassoc" or self.cp not in CP_VALUES):
            raise ValueError(f"cstat=fold / dfold: norm=reassoc, cp in {CP_VALUES}")
        if self.cstat == "dfold" and self.cfuse:
            raise ValueError("cstat=dfold: not with cfuse")
        if self.stamp and self.dbg:
            raise ValueError("stamp: not with dbg (both use the du buffer)")
        if self.cvec not in ("scalar", "pk", "dot2") or (self.cvec != "scalar" and self.cstat != "cta"):
            raise ValueError("cvec scalar | pk | dot2 (cstat=cta)")
        if d.rows_per_unit > 128 or u.rows_per_unit > 256:
            raise ValueError("D rows per unit <= 128 (256-row down units corrupt memory in the chain: open), U <= 256")
        if self.gplace not in ("c", "d", "u") or (self.gplace != "c" and (self.cstat != "cta" or self.cvec != "scalar")):
            raise ValueError("gplace c | d | u (d / u: cstat=cta, cvec=scalar)")
        if self.uvec not in ("scalar", "pk"):
            raise ValueError("uvec scalar | pk")
        if self.eload not in ("mi", "t"):
            raise ValueError("eload mi | t")
        if self.sig not in ("acc", "fast"):
            raise ValueError("sig acc | fast")
        if self.pre != "none" and (not self.pre or set(self.pre) - set("dunDU")):
            raise ValueError("pre none | a subset of 'dunDU'")
        if ("D" in self.pre and not d.xu) or ("U" in self.pre and not u.xu):
            raise ValueError("pre D / U (register prefetch): the stage's plan must have xu (its prologue takes them)")
        if "D" in self.pre and ("d" in self.pre or self.cfuse) or "U" in self.pre and "u" in self.pre:
            raise ValueError("pre: D excludes d (and cfuse), U excludes u")
        if self.lpad not in ("entry", "fit"):
            raise ValueError("lpad entry | fit")
        if self.dmut not in D_MUTANTS or self.umut not in U_MUTANTS or self.mutant not in CHAIN_MUTANTS:
            raise ValueError("mutants: dmut / umut / mutant")
        lds = hc_lds_bytes(self)
        if lds > LDS_MAX:
            raise ValueError(f"LDS {lds} bytes > {LDS_MAX}")


def hc_lds_bytes(hk: HcKnobs) -> int:
    """Estimate (the check's pre-filter): the stages' own buffers, padded past half a CU (one CTA per CU)."""
    from rocke.helpers.lds_arena import ONE_CTA_PER_CU_WORDS

    d, u = hk.parts()
    own = d.lds_bytes() + u.lds_bytes() + hk.tbk * (u.rows_per_unit // 4) * 4 + 4 * 64 + 4 * CTL_WORDS + 256
    return max(ONE_CTA_PER_CU_WORDS * 4, own + 1024)


def hc_kernel_name(hk: HcKnobs) -> str:
    d, u = hk.parts()
    base = HcKnobs(tp=hk.tp, bucket=hk.bucket)
    own = _suffix(hk, base)
    cs = G._suffix(hk.c, CKnobs())
    h = hashlib.sha1("|".join(map(repr, (d, u))).encode()).hexdigest()[:10]
    name = f"q38lbh_tp{hk.tp}_b{hk.tbk}_{h}"
    if own:
        name += f"_{own}"
    if cs:
        name += f"_c{cs}"
    return name


HC_I64 = ["wd", "wds", "wu", "wus", "wn", "winj", "r", "y", "g", "xq", "xqs", "inv", "dy", "part", "cnt",
          "aq", "aqs", "xo", "xo8", "xo8s", "du", "gs", "err", "ctl", "info", "ydump", "sp"]  # fmt: skip
HC_I32 = ["rows", "l0", "nl", "maxit", "gridn", "am", "wm"]


def hc_params() -> list[str]:
    """Kernel arguments: i64 ``HC_I64`` then i32 ``HC_I32`` (host packs ``<26Q7i``). Activation buffers hold ``am``
    slots (layer ``l`` uses slot ``l mod am``; ``r`` and ``g``: in ``l mod am``, out ``(l + 1) mod am``), weights ``wm``
    sets (``l mod wm``); ``cnt`` holds ``nl`` per-layer counter blocks of ``CNT_LAYER`` bytes (never reused in a launch)."""
    return HC_I64 + HC_I32


def strides(hk: HcKnobs) -> dict:
    """Per-slot / per-set byte strides of every per-layer buffer (host and kernel agree on these)."""
    d, u = hk.parts()
    TP = hk.tbk
    wl_d, sl_d = G._wstride_mx(LR, HCW)
    wl_u, sl_u = G._wstride_mx(HCW, LR)
    return dict(wd=wl_d, wds=sl_d, wu=wl_u, wus=sl_u, wn=HCW * 2, winj=NS * HCW * 2, r=TP * HCW * 2,
                y=hk.tp * TP * HID * 2, g=TP * NS * 4, xq=TP * HCW, xqs=NQ4_X * TP * 16, inv=NS * TP * 4,
                dy=TP * E1M.mpad(LR, d) * 2, aq=TP * LR, aqs=NQ4_A * TP * 16, xo=TP * HID * 2, xo8=TP * HID,
                xo8s=NQ4_O * TP * 16, du=TP * HCW * 4, sp=max(SP_MAX, SPF_MAX) * TP * 4)  # fmt: skip


# ------------------------------------------------------------------ host layouts


def up_perm():
    """Row permutation of the up weights: new row ``4 col + i`` = checkpoint row ``i H + col``."""
    import torch

    col = torch.arange(HID).repeat_interleave(NS)
    i = torch.arange(NS).repeat(HID)
    return i * HID + col


def pack_up_perm(codes, scales):
    """One-time repack of the up weights (MXFP4 codes ``[10240, 160]``, E8M0 scales ``[10240, >= 10]``): rows permuted
    by :func:`up_perm`, then E1's tile + op_sel layout (:func:`.e1.pack_weights_tile_sop`)."""
    p = up_perm().to(codes.device)
    return E1M.pack_weights_tile_sop(codes[p], scales[p], HCW, LR)


# ------------------------------------------------------------------ stage C


class _Comb:
    """The C stage on scope ``b`` (fresh thread ids). ``P``: this layer's base addresses (i64) and ``rows``."""

    def __init__(self, hk: HcKnobs, b, P: dict, TP: int):
        self.hk, self.k, self.b, self.P, self.TP = hk, hk.c, b, P, TP

    def emit(self) -> None:
        if self.hk.cstat in ("fold", "dfold", "cfold"):
            return self.emit_fold()
        b, hk, k, P, TP = self.b, self.hk, self.k, self.P, self.TP
        W, cw = hk.waves, k.cw
        cr = W // (4 * cw)
        nthr = 64 * cw
        passes = -(-NV // nthr)
        tid = b.thread_id_x()
        lane = b.land(tid, _c(b, 63))
        wave = b.lshr(tid, _c(b, 6))
        rloc = b.div(wave, _c(b, 4 * cw)) if cr > 1 else _c(b, 0)
        stream = b.land(b.div(wave, _c(b, cw)), _c(b, 3)) if cw > 1 else b.land(wave, _c(b, 3))
        sub = b.land(wave, _c(b, cw - 1)) if cw > 1 else _c(b, 0)
        t = b.add(b.mul(sub, _c(b, 64)), lane)
        rows = P["rows"]
        units = b.readfirstlane(b.div(b.add(rows, _c(b, cr - 1)), _c(b, cr)))
        bid = b.readfirstlane(b.block_id_x())
        red = b.smem_alloc(F32, [cr * NS * cw * 5 + 8], name_hint="hccred")
        with b.scf_if(b.cmp_lt(bid, units)):
            brow = b.add(b.mul(bid, _c(b, cr)), rloc)
            valid = b.cmp_lt(brow, rows)
            br = b.select(valid, brow, _c(b, 0))
            rin = _rsrc(b, P["rin"], _c(b, TP * HCW * 2))
            rout = _rsrc(b, P["rout"], _c(b, TP * HCW * 2))
            yrs = _rsrc(b, P["y"], _c(b, hk.tp * TP * HID * 2))
            wn = _rsrc(b, P["wn"], _c(b, HCW * 2))
            wj = _rsrc(b, P["winj"], _c(b, NS * HCW * 2))
            gin = _rsrc(b, P["gin"], _c(b, TP * NS * 4))
            gout = _rsrc(b, P["gout"], _c(b, TP * NS * 4))
            xq = _rsrc(b, P["xq"], _c(b, TP * HCW))
            inv_rs = _rsrc(b, P["inv"], _c(b, NS * TP * 4))
            xqs_a = b.readfirstlane(P["xqs"])
            gate = b.bitcast(b.buffer_load_cp(gin, b.mul(b.add(b.mul(br, _c(b, NS)), stream), _c(b, 4)), _c(b, 0),
                                              dwords=1), F32)  # fmt: skip
            # ---- loads of every pass first (all in flight), then the combine
            vs, oks, L = [], [], []
            for p in range(passes):
                v = b.add(t, _c(b, p * nthr))
                ok = valid if NV % nthr == 0 or p < passes - 1 else b.land(valid, b.cmp_lt(v, _c(b, NV)))
                kk = b.add(b.mul(stream, _c(b, HID)), b.mul(v, _c(b, 8)))  # K index of the first of the 8
                ro = b.mul(b.add(b.mul(br, _c(b, HCW)), kk), _c(b, 2))
                rec = dict(r=b.buffer_load_cp(rin, b.select(ok, ro, _c(b, OOB)), _c(b, 0), dwords=4))
                yo = b.mul(b.add(b.mul(br, _c(b, HID)), b.mul(v, _c(b, 8))), _c(b, 2))
                rec["y"] = [b.buffer_load_cp(yrs, b.select(ok, b.add(yo, _c(b, rk * TP * HID * 2)), _c(b, OOB)), _c(b, 0),
                                             dwords=4) for rk in range(hk.tp)]  # fmt: skip
                rec["w"] = b.buffer_load_cp(wn, b.select(ok, b.mul(kk, _c(b, 2)), _c(b, OOB)), _c(b, 0), dwords=4)
                rec["j"] = [b.buffer_load_cp(wj, b.select(ok, b.mul(b.add(kk, _c(b, j * HCW)), _c(b, 2)), _c(b, OOB)),
                                             _c(b, 0), dwords=4) for j in range(NS)] if hk.gplace == "c" else []  # fmt: skip
                vs.append((v, kk, ro))
                oks.append(ok)
                L.append(rec)
            rp_all = []
            ss = None
            pk = hk.cvec in ("pk", "dot2")
            d2 = hk.cvec == "dot2"
            if pk:
                gate2 = b.vec_pack([gate, gate], F32)
            for p in range(passes):
                rec, (v, kk, ro), ok = L[p], vs[p], oks[p]
                rp = []
                if pk:
                    rpw = []
                    for d in range(4):
                        rv2 = _unpk(b, b.vec_extract(rec["r"], d))
                        yws = [b.vec_extract(yy, d) for yy in rec["y"]]
                        yv2 = _unpk(b, yws[0])
                        for yx in yws[1:]:
                            yv2 = _unpk(b, _pkw(b, b.fadd(yv2, _unpk(b, yx))))
                        tt2 = yv2 if k.mutant == "ygate" else _unpk(b, _pkw(b, b.fmul(yv2, gate2)))
                        w32 = _pkw(b, b.fadd(rv2, tt2))
                        rpw.append(w32)
                        rp += [_lo(b, w32), _hi(b, w32)]
                    if k.mutant != "rskip":
                        b.buffer_store_cp(b.vec_pack(rpw, I32), rout, b.select(ok, ro, _c(b, OOB)), _c(b, 0), aux=CPOL_SC1)
                    if d2:
                        s8 = _f(b, 0.0)
                        for w32 in rpw:
                            s8 = _dot2(b, w32, w32, s8)
                    else:  # pairs: (q0 q1) + (q2 q3), (q4 q5) + (q6 q7), then the two lanes of the pair
                        q2 = [b.fmul(_unpk(b, w32), _unpk(b, w32)) for w32 in rpw]
                        t2 = b.fadd(b.fadd(q2[0], q2[1]), b.fadd(q2[2], q2[3]))
                        s8 = b.fadd(b.vec_extract(t2, 0), b.vec_extract(t2, 1))
                    s8 = b.select(ok, s8, _f(b, 0.0))
                    ss = s8 if ss is None else b.fadd(ss, s8)
                    rp_all.append((rp, rpw))
                    continue
                for e in range(8):
                    rw = b.vec_extract(rec["r"], e // 2)
                    rv = _hi(b, rw) if e % 2 else _lo(b, rw)
                    yw = [b.vec_extract(yy, e // 2) for yy in rec["y"]]
                    yv = _hi(b, yw[0]) if e % 2 else _lo(b, yw[0])
                    for yx in yw[1:]:  # TP2: the ranks' bf16 partials in rank order, rounded to bf16
                        yv = _bf16r(b, b.fadd(yv, _hi(b, yx) if e % 2 else _lo(b, yx)))
                    tt = yv if k.mutant == "ygate" else _bf16r(b, b.fmul(yv, gate))
                    rp.append(_bf16r(b, b.fadd(rv, tt)))
                if k.mutant != "rskip":
                    b.buffer_store_cp(_pack8(b, rp), rout, b.select(ok, ro, _c(b, OOB)), _c(b, 0), aux=CPOL_SC1)
                q = [b.fmul(x, x) for x in rp]
                s8 = b.fadd(b.fadd(b.fadd(q[0], q[1]), b.fadd(q[2], q[3])), b.fadd(b.fadd(q[4], q[5]), b.fadd(q[6], q[7])))
                s8 = b.select(ok, s8, _f(b, 0.0))
                ss = s8 if ss is None else b.fadd(ss, s8)
                rp_all.append(rp)
            ss = _wsum(b, ss)
            sidx = b.add(b.mul(b.add(b.mul(rloc, _c(b, NS)), stream), _c(b, cw)), sub)  # [cr][NS][cw]
            if cw > 1:
                with b.scf_if(b.cmp_eq(lane, _c(b, 0))):
                    b.smem_store_vN(red, [sidx], ss, 1)
                b.sync_lds_only()
                s0 = b.vec_extract(b.smem_load_vN_f32(red, b.sub(sidx, sub), n=1), 0)
                s1 = b.vec_extract(b.smem_load_vN_f32(red, b.add(b.sub(sidx, sub), _c(b, 1)), n=1), 0)
                ss = b.fadd(s0, s1)
            mean = b.fmul(ss, b.bitcast(_c(b, INV_HIDDEN), F32))
            stab = mean if k.mutant == "eps" else b.fadd(mean, b.bitcast(_c(b, EPS_BITS), F32))
            if k.inv == "f64":
                inv = b.cvt_f64_to_f32(b.fdiv(b.const_f64(1.0), b.sqrt(b.cvt_f32_to_f64(stab))))
            else:
                inv = b.rsqrt(stab)
            with b.scf_if(b.land(valid, b.land(b.cmp_eq(lane, _c(b, 0)), b.cmp_eq(sub, _c(b, 0))))):
                b.buffer_store_cp(b.bitcast(inv, I32), inv_rs, b.mul(b.add(b.mul(stream, _c(b, TP)), br), _c(b, 4)),
                                  _c(b, 0), aux=CPOL_SC1)  # fmt: skip
            # ---- the down GEMM's activations (MXFP8, op_sel scales) and the injection dots
            dots = [None] * NS
            dots2 = [None] * NS
            p4 = b.land(t, _c(b, 3))
            if pk:
                inv2 = b.vec_pack([inv, inv], F32)
            for p in range(passes):
                rec, (v, kk, ro), ok, rp = L[p], vs[p], oks[p], rp_all[p]
                n = []
                if pk:
                    rp, _rpw = rp
                    nw = []
                    for d in range(4):
                        w2 = _unpk(b, b.vec_extract(rec["w"], d))
                        r2 = b.vec_pack([rp[2 * d], rp[2 * d + 1]], F32)
                        x2 = b.fmul(b.fmul(r2, inv2), w2) if hk.norm == "faithful" else b.fmul(r2, w2)
                        word = _pkw(b, x2)
                        nw.append(word)
                        n += [_lo(b, word), _hi(b, word)]
                for e in range(0 if pk else 8):
                    ww = b.vec_extract(rec["w"], e // 2)
                    wv = _hi(b, ww) if e % 2 else _lo(b, ww)
                    if hk.norm == "faithful":
                        n.append(_bf16r(b, b.fmul(b.fmul(rp[e], inv), wv)))
                    else:
                        n.append(_bf16r(b, b.fmul(rp[e], wv)))
                words, E = MQ.quantize_mxfp8(b, n, 4)
                if k.mutant == "xblk":
                    E = b.select(b.cmp_eq(stream, _c(b, 1)), b.add(E, _c(b, 1)), E)
                xo = b.add(b.mul(br, _c(b, HCW)), kk)
                b.buffer_store_cp(b.vec_pack(words, I32), xq, b.select(ok, xo, _c(b, OOB)), _c(b, 0), aux=CPOL_SC1)
                qc = b.lshr(kk, _c(b, 7))
                gq = b.land(b.lshr(kk, _c(b, 5)), _c(b, 3))
                soff = b.add(b.mul(b.add(b.mul(b.add(b.mul(b.lshr(qc, _c(b, 2)), _c(b, TP)), br), _c(b, 4)), gq), _c(b, 4)),
                             b.land(qc, _c(b, 3)))  # fmt: skip
                with b.scf_if(b.land(ok, b.cmp_eq(p4, _c(b, 0)))):
                    b.inline_asm("global_store_byte $0, $1, off sc1", "v,v,~{memory}", [b.add(xqs_a, _i64(b, soff)), E], None)
                for j in range(NS if hk.gplace == "c" else 0):
                    jw = rec["j"][j]
                    acc = dots[j] if dots[j] is not None else _f(b, 0.0)
                    if pk and d2:
                        for d in range(4):
                            acc = _dot2(b, b.vec_extract(jw, d), nw[d], acc)
                        dots[j] = acc
                        continue
                    if pk:  # packed fma on (even, odd) element pairs; the pair's two sums added at the end of C
                        acc2 = dots2[j] if dots2[j] is not None else b.vec_pack([_f(b, 0.0), _f(b, 0.0)], F32)
                        for d in range(4):
                            acc2 = b.vector_fma(_unpk(b, b.vec_extract(jw, d)), _unpk(b, nw[d]), acc2)
                        dots2[j] = acc2
                        continue
                    for e in range(8):
                        x = b.vec_extract(jw, e // 2)
                        acc = b.fma(_hi(b, x) if e % 2 else _lo(b, x), n[e], acc)
                    dots[j] = acc
            # ---- gates: per-stream dot partials -> the row's 4 gates (stream order)
            if hk.gplace != "c":
                return
            if pk and not d2:
                dots = [b.fadd(b.vec_extract(x, 0), b.vec_extract(x, 1)) for x in dots2]
            base = cr * NS * cw
            for j in range(NS):
                dj = _wsum(b, dots[j])
                if hk.norm == "reassoc":
                    dj = b.fmul(dj, inv)
                with b.scf_if(b.cmp_eq(lane, _c(b, 0))):
                    b.smem_store_vN(red, [b.add(_c(b, base), b.add(b.mul(sidx, _c(b, NS)), _c(b, j)))], dj, 1)
            b.sync_lds_only()
            with b.scf_if(b.cmp_lt(tid, _c(b, cr * NS))):
                rl = b.lshr(tid, _c(b, 2))
                j = b.land(tid, _c(b, 3))
                grow = b.add(b.mul(bid, _c(b, cr)), rl)
                tot = None
                nst = NS - 1 if k.mutant == "inj" else NS
                for i in range(nst):
                    for s_ in range(cw):
                        idx = b.add(_c(b, base), b.add(b.mul(b.add(b.mul(b.add(b.mul(rl, _c(b, NS)), _c(b, i)), _c(b, cw)),
                                                                   _c(b, s_)), _c(b, NS)), j))  # fmt: skip
                        x = b.vec_extract(b.smem_load_vN_f32(red, idx, n=1), 0)
                        tot = x if tot is None else b.fadd(tot, x)
                val = _bf16r(b, b.fmul(_bf16r(b, tot), _f(b, 0.25)))
                val = _bf16r(b, _sigmoid(b, val, hk.sig))
                val = _bf16r(b, b.fmul(val, _f(b, 2.0)))
                gok = b.cmp_lt(grow, rows)
                goff = b.mul(b.add(b.mul(b.select(gok, grow, _c(b, 0)), _c(b, NS)), j), _c(b, 4))
                b.buffer_store_cp(b.bitcast(val, I32), gout, b.select(gok, goff, _c(b, OOB)), _c(b, 0), aux=CPOL_SC1)


    def emit_fold(self) -> None:
        """cstat="fold": wave unit (token b, stream i, part p) over vectors ``[p n, (p + 1) n)``, ``n = 320 / cp``: the
        combine, ``r'``, the MXFP8 of ``bf16(r' w')`` (reassoc), the part's square sum and its 4 injection dots
        (``sp[p NS + i][b]``, ``sp[cp NS + (p NS + i) NS + j][b]``). No LDS, no barrier, no cross-wave step."""
        b, hk, k, P, TP = self.b, self.hk, self.k, self.P, self.TP
        W, cp = hk.waves, hk.cp
        nvp = NV // cp
        passes = -(-nvp // 64)
        tid = b.thread_id_x()
        lane = b.land(tid, _c(b, 63))
        wave = b.lshr(tid, _c(b, 6))
        rows = P["rows"]
        bid = b.readfirstlane(b.block_id_x())
        nwu = b.mul(rows, _c(b, NS * cp))
        nk = b.readfirstlane(b.div(b.add(nwu, _c(b, W * GRID - 1)), _c(b, W * GRID)))
        rin = _rsrc(b, P["rin"], _c(b, TP * HCW * 2))
        rout = _rsrc(b, P["rout"], _c(b, TP * HCW * 2))
        yrs = _rsrc(b, P["y"], _c(b, hk.tp * TP * HID * 2))
        wn = _rsrc(b, P["wn"], _c(b, HCW * 2))
        wj = _rsrc(b, P["winj"], _c(b, NS * HCW * 2))
        gin = _rsrc(b, P["gin"], _c(b, TP * NS * 4))
        xq = _rsrc(b, P["xq"], _c(b, TP * HCW))
        sp = _rsrc(b, P["sp"], _c(b, SP_MAX * TP * 4))
        xqs_a = b.readfirstlane(P["xqs"])
        self.flg = b.smem_alloc(F32, [W], name_hint="hccflg") if hk.cstat == "cfold" else None
        with b.scf_for(_c(b, 0), nk, _c(b, 1), iv_name="hcck") as kk_:
            w = b.add(b.mul(b.add(b.mul(kk_, _c(b, GRID)), bid), _c(b, W)), wave)
            brow = b.div(w, _c(b, NS * cp))
            stream = b.land(b.div(w, _c(b, cp)), _c(b, 3)) if cp > 1 else b.land(w, _c(b, 3))
            part = b.mod(w, _c(b, cp)) if cp > 1 else _c(b, 0)
            valid = b.cmp_lt(brow, rows)
            br = b.select(valid, brow, _c(b, 0))
            goff = b.select(valid, b.mul(b.add(b.mul(br, _c(b, NS)), stream), _c(b, 4)), _c(b, OOB))
            gate = b.bitcast(b.buffer_load_cp(gin, goff, _c(b, 0), dwords=1), F32)
            L = []
            for ps in range(passes):
                lv = b.add(lane, _c(b, 64 * ps))
                ok = valid if nvp % 64 == 0 or ps < passes - 1 else b.land(valid, b.cmp_lt(lv, _c(b, nvp)))
                v = b.add(b.mul(part, _c(b, nvp)), lv)
                kk = b.add(b.mul(stream, _c(b, HID)), b.mul(v, _c(b, 8)))
                ro = b.mul(b.add(b.mul(br, _c(b, HCW)), kk), _c(b, 2))
                yo = b.mul(b.add(b.mul(br, _c(b, HID)), b.mul(v, _c(b, 8))), _c(b, 2))
                rec = dict(ok=ok, kk=kk, ro=ro, r=b.buffer_load_cp(rin, b.select(ok, ro, _c(b, OOB)), _c(b, 0), dwords=4))
                rec["y"] = [b.buffer_load_cp(yrs, b.select(ok, b.add(yo, _c(b, rk * TP * HID * 2)), _c(b, OOB)), _c(b, 0),
                                             dwords=4) for rk in range(hk.tp)]  # fmt: skip
                rec["w"] = b.buffer_load_cp(wn, b.select(ok, b.mul(kk, _c(b, 2)), _c(b, OOB)), _c(b, 0), dwords=4)
                rec["j"] = [b.buffer_load_cp(wj, b.select(ok, b.mul(b.add(kk, _c(b, j * HCW)), _c(b, 2)), _c(b, OOB)),
                                             _c(b, 0), dwords=4) for j in range(NS)]  # fmt: skip
                L.append(rec)
            ss = None
            dots = [None] * NS
            p4 = b.land(lane, _c(b, 3))
            for rec in L:
                ok = rec["ok"]
                rp, n = [], []
                for e in range(8):
                    rw = b.vec_extract(rec["r"], e // 2)
                    rv = _hi(b, rw) if e % 2 else _lo(b, rw)
                    yw = [b.vec_extract(yy, e // 2) for yy in rec["y"]]
                    yv = _hi(b, yw[0]) if e % 2 else _lo(b, yw[0])
                    for yx in yw[1:]:
                        yv = _bf16r(b, b.fadd(yv, _hi(b, yx) if e % 2 else _lo(b, yx)))
                    tt = yv if k.mutant == "ygate" else _bf16r(b, b.fmul(yv, gate))
                    rp.append(_bf16r(b, b.fadd(rv, tt)))
                    ww = b.vec_extract(rec["w"], e // 2)
                    n.append(_bf16r(b, b.fmul(rp[e], _hi(b, ww) if e % 2 else _lo(b, ww))))
                if k.mutant != "rskip":
                    b.buffer_store_cp(_pack8(b, rp), rout, b.select(ok, rec["ro"], _c(b, OOB)), _c(b, 0), aux=CPOL_SC1)
                q = [b.fmul(x, x) for x in rp]
                s8 = b.fadd(b.fadd(b.fadd(q[0], q[1]), b.fadd(q[2], q[3])), b.fadd(b.fadd(q[4], q[5]), b.fadd(q[6], q[7])))
                s8 = b.select(ok, s8, _f(b, 0.0))
                ss = s8 if ss is None else b.fadd(ss, s8)
                words, E = MQ.quantize_mxfp8(b, n, 4)
                if k.mutant == "xblk":
                    E = b.select(b.cmp_eq(stream, _c(b, 1)), b.add(E, _c(b, 1)), E)
                kk = rec["kk"]
                xo = b.add(b.mul(br, _c(b, HCW)), kk)
                b.buffer_store_cp(b.vec_pack(words, I32), xq, b.select(ok, xo, _c(b, OOB)), _c(b, 0), aux=CPOL_SC1)
                qc = b.lshr(kk, _c(b, 7))
                gq = b.land(b.lshr(kk, _c(b, 5)), _c(b, 3))
                soff = b.add(b.mul(b.add(b.mul(b.add(b.mul(b.lshr(qc, _c(b, 2)), _c(b, TP)), br), _c(b, 4)), gq), _c(b, 4)),
                             b.land(qc, _c(b, 3)))  # fmt: skip
                with b.scf_if(b.land(ok, b.cmp_eq(p4, _c(b, 0)))):
                    b.inline_asm("global_store_byte $0, $1, off sc1", "v,v,~{memory}", [b.add(xqs_a, _i64(b, soff)), E], None)
                for j in range(NS):
                    acc = dots[j] if dots[j] is not None else _f(b, 0.0)
                    jw = rec["j"][j]
                    for e in range(8):
                        x = b.vec_extract(jw, e // 2)
                        acc = b.fma(_hi(b, x) if e % 2 else _lo(b, x), n[e], acc)
                    dots[j] = acc
            ss = _wsum(b, ss)
            dd = [_wsum(b, d) for d in dots]
            with b.scf_if(b.land(valid, b.cmp_eq(lane, _c(b, 0)))):
                pi = b.add(b.mul(part, _c(b, NS)), stream)
                b.buffer_store_cp(b.bitcast(ss, I32), sp, b.mul(b.add(b.mul(pi, _c(b, TP)), br), _c(b, 4)), _c(b, 0),
                                  aux=CPOL_SC1)  # fmt: skip
                for j in range(NS):
                    di = b.add(_c(b, cp * NS), b.add(b.mul(pi, _c(b, NS)), _c(b, j)))
                    b.buffer_store_cp(b.bitcast(dd[j], I32), sp, b.mul(b.add(b.mul(di, _c(b, TP)), br), _c(b, 4)), _c(b, 0),
                                      aux=CPOL_SC1)  # fmt: skip
            if hk.cstat == "cfold":
                self._cfold_tail(valid, br, lane, sp)

    def _cfold_tail(self, valid, br, lane, sp) -> None:
        """cstat="cfold": the wave unit arrives on its token's counter (after its stores: vmcnt(0), one returning
        agent-scope add); the token's last arriver (4 cp units) finishes the 4 inverse RMS and the 4 gates from the parts
        (lane j < 4: stream / gate j), stores them and zeroes the counter."""
        b, hk, P, TP = self.b, self.hk, self.P, self.TP
        cp, mut = hk.cp, hk.c.mutant
        b.s_waitcnt(vmcnt=0)
        cnt_a = b.readfirstlane(P["ccnt"])
        wave = b.lshr(b.thread_id_x(), _c(b, 6))
        # the wave's returned count through LDS (every wave of the CTA runs the unit loop in step: one CTA barrier)
        with b.scf_if(b.cmp_eq(lane, _c(b, 0))):
            b.smem_store_vN(self.flg, [wave], b.bitcast(_c(b, -1), F32), 1)
            with b.scf_if(valid):
                old = D.atomic_add_scoped(b, cnt_a, b.mul(br, _c(b, 4)), _c(b, 1), scope="agent", returns=True)
                b.smem_store_vN(self.flg, [wave], b.bitcast(old, F32), 1)
        b.sync_lds_only()
        last = b.readfirstlane(b.bitcast(b.vec_extract(b.smem_load_vN_f32(self.flg, wave, n=1), 0), I32))
        b.sync_lds_only()  # the next unit may overwrite the slot only after every wave read it
        with b.scf_if(b.land(valid, b.cmp_eq(last, _c(b, NS * cp - 1)))):
            D.acquire_l2(b, "agent")
            inv_rs = _rsrc(b, P["inv"], _c(b, NS * TP * 4))
            gout = _rsrc(b, P["gout"], _c(b, TP * NS * 4))
            j = b.land(lane, _c(b, 3))

            def ld(idx):
                off = b.mul(b.add(b.mul(idx, _c(b, TP)), br), _c(b, 4))
                return b.bitcast(b.buffer_load_cp(sp, off, _c(b, 0), dwords=1, aux=CPOL_SC1), F32)

            invs = []
            for i in range(NS):
                tot = None
                for p_ in range(cp):
                    x = ld(_c(b, p_ * NS + i))
                    tot = x if tot is None else b.fadd(tot, x)
                invs.append(_inv_rule(b, tot, mut, hk.c.inv))
            gt = None
            for i in range(NS - 1 if mut == "inj" else NS):
                di = None
                for p_ in range(cp):
                    x = ld(b.add(_c(b, cp * NS + (p_ * NS + i) * NS), j))
                    di = x if di is None else b.fadd(di, x)
                v = b.fmul(di, invs[i])
                gt = v if gt is None else b.fadd(gt, v)
            gv = _gate_rule(b, gt, hk.sig)
            myinv = b.select(b.cmp_eq(j, _c(b, 0)), invs[0], b.select(b.cmp_eq(j, _c(b, 1)), invs[1],
                             b.select(b.cmp_eq(j, _c(b, 2)), invs[2], invs[3])))  # fmt: skip
            with b.scf_if(b.cmp_lt(lane, _c(b, NS))):
                b.buffer_store_cp(b.bitcast(gv, I32), gout, b.mul(b.add(b.mul(br, _c(b, NS)), j), _c(b, 4)), _c(b, 0),
                                  aux=CPOL_SC1)  # fmt: skip
                b.buffer_store_cp(b.bitcast(myinv, I32), inv_rs, b.mul(b.add(b.mul(j, _c(b, TP)), br), _c(b, 4)), _c(b, 0),
                                  aux=CPOL_SC1)  # fmt: skip
            with b.scf_if(b.cmp_eq(lane, _c(b, 0))):
                b.store(_c(b, 0), _ptr(b, cnt_a), br)


def _inv_rule(b, ss: Value, mut: str = "none", kind: str = "f64") -> Value:
    """ATOM's inverse RMS of a stream from its square sum: ``f32(1 / sqrt_f64(ss fl(1/2560) + eps))`` (``rsq``: the
    hardware reciprocal square root, tolerance class); mutant ``eps`` drops eps."""
    mean = b.fmul(ss, b.bitcast(_c(b, INV_HIDDEN), F32))
    stab = mean if mut == "eps" else b.fadd(mean, b.bitcast(_c(b, EPS_BITS), F32))
    if kind == "f64":
        return b.cvt_f64_to_f32(b.fdiv(b.const_f64(1.0), b.sqrt(b.cvt_f32_to_f64(stab))))
    return b.rsqrt(stab)


def _gate_rule(b, tot: Value, sig: str) -> Value:
    """ATOM's injection weight: ``bf16(2 bf16(sigmoid(bf16(bf16(t) / 4))))``."""
    val = _bf16r(b, b.fmul(_bf16r(b, tot), _f(b, 0.25)))
    val = _bf16r(b, _sigmoid(b, val, sig))
    return _bf16r(b, b.fmul(val, _f(b, 2.0)))




class _RegPre:
    """Mixin for an E1 stage: ``pre_issue`` loads the weight part (codes + weight scales) of the prologue's first
    ``depth`` chunks now (before an edge's wait); E1's prologue (``xu`` plans: ``wpos(unit 0, p)`` for p < depth, in
    order) then receives them, with the X part of each chunk loaded at that point (after the edge)."""

    _pre = None

    def _wpart_len(self, c: int) -> int:
        k = self.k
        if k.sop:
            return k.mt + (k.mt * (2 if k.sop_funnel else 1) if c % 4 == 0 else 0)
        return 2 * k.mt if k.w == "mx" else 4 * k.mt

    def pre_issue(self) -> list:
        b, k = self.b, self.k
        no = b.cmp_ne(_c(b, 0), _c(b, 0))
        out = []
        for p_ in range(k.depth):
            d, c = divmod(p_, k.kc)
            u = E1M._Unit(self, _c(b, d))
            full = E1M._E1.wissue(self, u, c, xonly_gate=no)
            out.append((c, full[: self._wpart_len(c)]))
        self._pre = out
        return []

    def wissue(self, u, c: int, xonly_gate=None, xonly: bool = False) -> list:
        if self._pre and not xonly and xonly_gate is None:
            pc, wpart = self._pre.pop(0)
            assert pc == c, (pc, c)
            return wpart + super().wissue(u, c, None, True)
        return super().wissue(u, c, xonly_gate, xonly)




class _Gates:
    """``gplace="d" / "u"``: the next combine's gates, off C's critical path. Gate unit = one token (``waves / 4``
    tokens per CTA, wave ``w`` = stream ``w mod 4``) on CTA ``255 - unit`` (the top CTAs: idle in the D / U stage at
    small rows), after that CTA's own D / U units. The arithmetic is C's (cstat="cta") bit for bit: ``n`` from r' / inv /
    w' in the same pass order, the 4 fma chains per stream, the wave sums, the stream-ordered sum, the gate rule; the
    inputs (r', inv) are C's outputs, read after the C -> D edge."""

    def __init__(self, hk: HcKnobs, b, P: dict, TP: int):
        self.hk, self.k, self.b, self.P, self.TP = hk, hk.c, b, P, TP

    def emit(self) -> None:
        b, hk, k, P, TP = self.b, self.hk, self.k, self.P, self.TP
        W = hk.waves
        cr = W // 4
        passes = -(-NV // 64)
        tid = b.thread_id_x()
        lane = b.land(tid, _c(b, 63))
        wave = b.lshr(tid, _c(b, 6))
        rloc = b.lshr(wave, _c(b, 2)) if cr > 1 else _c(b, 0)
        stream = b.land(wave, _c(b, 3))
        rows = P["rows"]
        units = b.readfirstlane(b.div(b.add(rows, _c(b, cr - 1)), _c(b, cr)))
        bid = b.readfirstlane(b.block_id_x())
        gu = b.sub(_c(b, GRID - 1), bid)
        red = b.smem_alloc(F32, [cr * NS * NS + 8], name_hint="hcgred")
        with b.scf_if(b.cmp_lt(gu, units)):
            brow = b.add(b.mul(gu, _c(b, cr)), rloc)
            valid = b.cmp_lt(brow, rows)
            br = b.select(valid, brow, _c(b, 0))
            rp_rs = _rsrc(b, P["rout"], _c(b, TP * HCW * 2))
            wn = _rsrc(b, P["wn"], _c(b, HCW * 2))
            wj = _rsrc(b, P["winj"], _c(b, NS * HCW * 2))
            inv_rs = _rsrc(b, P["inv"], _c(b, NS * TP * 4))
            gout = _rsrc(b, P["gout"], _c(b, TP * NS * 4))
            inv = b.bitcast(b.buffer_load_cp(inv_rs, b.mul(b.add(b.mul(stream, _c(b, TP)), br), _c(b, 4)), _c(b, 0),
                                             dwords=1), F32)  # fmt: skip
            L = []
            for p_ in range(passes):
                v = b.add(lane, _c(b, p_ * 64))
                ok = valid if NV % 64 == 0 or p_ < passes - 1 else b.land(valid, b.cmp_lt(v, _c(b, NV)))
                kk = b.add(b.mul(stream, _c(b, HID)), b.mul(v, _c(b, 8)))
                ro = b.mul(b.add(b.mul(br, _c(b, HCW)), kk), _c(b, 2))
                rec = dict(r=b.buffer_load_cp(rp_rs, b.select(ok, ro, _c(b, OOB)), _c(b, 0), dwords=4))
                rec["w"] = b.buffer_load_cp(wn, b.select(ok, b.mul(kk, _c(b, 2)), _c(b, OOB)), _c(b, 0), dwords=4)
                rec["j"] = [b.buffer_load_cp(wj, b.select(ok, b.mul(b.add(kk, _c(b, j * HCW)), _c(b, 2)), _c(b, OOB)),
                                             _c(b, 0), dwords=4) for j in range(NS)]  # fmt: skip
                L.append(rec)
            dots = [None] * NS
            for rec in L:
                n = []
                for e in range(8):
                    rw = b.vec_extract(rec["r"], e // 2)
                    rv = _hi(b, rw) if e % 2 else _lo(b, rw)
                    ww = b.vec_extract(rec["w"], e // 2)
                    wv = _hi(b, ww) if e % 2 else _lo(b, ww)
                    if hk.norm == "faithful":
                        n.append(_bf16r(b, b.fmul(b.fmul(rv, inv), wv)))
                    else:
                        n.append(_bf16r(b, b.fmul(rv, wv)))
                for j in range(NS):
                    jw = rec["j"][j]
                    acc = dots[j] if dots[j] is not None else _f(b, 0.0)
                    for e in range(8):
                        x = b.vec_extract(jw, e // 2)
                        acc = b.fma(_hi(b, x) if e % 2 else _lo(b, x), n[e], acc)
                    dots[j] = acc
            sidx = b.add(b.mul(rloc, _c(b, NS)), stream)
            for j in range(NS):
                dj = _wsum(b, dots[j])
                if hk.norm == "reassoc":
                    dj = b.fmul(dj, inv)
                with b.scf_if(b.cmp_eq(lane, _c(b, 0))):
                    b.smem_store_vN(red, [b.add(b.mul(sidx, _c(b, NS)), _c(b, j))], dj, 1)
            b.sync_lds_only()
            with b.scf_if(b.cmp_lt(tid, _c(b, cr * NS))):
                rl = b.lshr(tid, _c(b, 2))
                j = b.land(tid, _c(b, 3))
                grow = b.add(b.mul(gu, _c(b, cr)), rl)
                tot = None
                for i in range(NS - 1 if k.mutant == "inj" else NS):
                    x = b.vec_extract(b.smem_load_vN_f32(red, b.add(b.mul(b.add(b.mul(rl, _c(b, NS)), _c(b, i)), _c(b, NS)), j),
                                                         n=1), 0)  # fmt: skip
                    tot = x if tot is None else b.fadd(tot, x)
                gv = _gate_rule(b, tot, hk.sig)
                gok = b.cmp_lt(grow, rows)
                goff = b.mul(b.add(b.mul(b.select(gok, grow, _c(b, 0)), _c(b, NS)), j), _c(b, 4))
                b.buffer_store_cp(b.bitcast(gv, I32), gout, b.select(gok, goff, _c(b, OOB)), _c(b, 0), aux=CPOL_SC1)
            b.sync_lds_only()


# ------------------------------------------------------------------ stage D (E1 + activation pack)


class _Dn(_RegPre, E1M._E1):
    """E1 on a scope; ``yq``'s LDS values are the activation ``bf16(s / (1 + exp(-s)))``, ``s = bf16(bf16(y) / 4)``."""

    def __init__(self, k: E1M.E1Knobs, scope, dmut: str = "none", sig: str = "acc", hk: HcKnobs | None = None,
                 TP: int = 0):  # fmt: skip
        with G._swap(E1M, "IRBuilder", lambda _n: scope):
            super().__init__(k)
        self.dmut, self.sig, self.hk, self.TP = dmut, sig, hk, TP
        if hk is not None and hk.cstat == "fold":
            b = self.b
            self.sp_rs = _rsrc(b, scope.param("sp", I64), _c(b, SP_MAX * TP * 4))
            self.hinv_rs = _rsrc(b, scope.param("hinv", I64), _c(b, NS * TP * 4))
            self.g_rs = _rsrc(b, scope.param("gout", I64), _c(b, TP * NS * 4))

    def _ssum(self, si: Value, tok: Value) -> Value:
        """The stream's square sum of token ``tok``: the cp parts in part order."""
        b, cp, TP = self.b, self.hk.cp, self.TP
        tot = None
        for p in range(cp):
            off = b.mul(b.add(b.mul(b.add(_c(b, p * NS), si), _c(b, TP)), tok), _c(b, 4))
            x = b.bitcast(b.buffer_load_cp(self.sp_rs, off, _c(b, 0), dwords=1), F32)
            tot = x if tot is None else b.fadd(tot, x)
        return tot

    _stampf = None
    _stbase = 7

    def epilogue(self, u, acc, segv=None) -> None:
        if self._stampf is not None:
            self._stampf(self._stbase + 1)
        super().epilogue(u, acc, segv)
        if self._stampf is not None:
            self._stampf(self._stbase + 2)

    def seg_loads(self, u) -> list:
        if self._stampf is not None:
            self._stampf(self._stbase)
        return self._seg_loads(u)

    def fold(self, u, p_rs, part=None) -> None:
        if self._stampf is not None:
            self._stampf(13)
        super().fold(u, p_rs, part)
        if self._stampf is not None:
            self._stampf(14)

    def _seg_loads(self, u) -> list:
        """cstat="fold": this wave's segment scale per token tile = its stream's inv, finished here from the C parts;
        the first unit of a token block (row group 0, split 0, K group 0) also writes inv (for U) and the gates."""
        hk = self.hk
        if hk is None or hk.cstat != "fold":
            return super().seg_loads(u)
        b, k = self.b, self.k
        mut = hk.c.mutant
        segs = k.nq // k.kseg
        si = _c(b, 0) if k.mutant == "seg" else b.div(u.q0, _c(b, segs))
        out = []
        for t in range(k.tbt):
            tok = b.smin(b.add(u.tok0, b.add(_c(b, t * TILE), self.l16)), self.rlast)
            out.append(b.bitcast(_inv_rule(b, self._ssum(si, tok), mut, hk.c.inv), I32))
        first = b.land(b.cmp_eq(u.rg, _c(b, 0)), b.cmp_eq(u.s, _c(b, 0)))
        if k.kw > 1:
            first = b.land(first, b.cmp_eq(self.kg, _c(b, 0)))
        with b.scf_if(b.land(u.valid, first)):
            for t in range(k.tbt):
                tok = b.add(u.tok0, b.add(_c(b, t * TILE), self.l16))
                okt = b.cmp_lt(tok, self.rows)
                tk = b.select(okt, tok, _c(b, 0))
                invs = [_inv_rule(b, self._ssum(_c(b, i), tk), mut, hk.c.inv) for i in range(NS)]
                tot = None
                nst = NS - 1 if mut == "inj" else NS
                for i in range(nst):  # gate j = this lane's group g: sum over streams of inv_i x the parts' dots
                    di = None
                    for p in range(hk.cp):
                        idx = b.add(_c(b, hk.cp * NS + (p * NS + i) * NS), self.g)
                        off = b.mul(b.add(b.mul(idx, _c(b, self.TP)), tk), _c(b, 4))
                        x = b.bitcast(b.buffer_load_cp(self.sp_rs, off, _c(b, 0), dwords=1), F32)
                        di = x if di is None else b.fadd(di, x)
                    v = b.fmul(di, invs[i])
                    tot = v if tot is None else b.fadd(tot, v)
                gv = _gate_rule(b, tot, hk.sig)
                g = self.g
                myinv = b.select(b.cmp_eq(g, _c(b, 0)), invs[0], b.select(b.cmp_eq(g, _c(b, 1)), invs[1],
                                 b.select(b.cmp_eq(g, _c(b, 2)), invs[2], invs[3])))  # fmt: skip
                goff = b.select(okt, b.mul(b.add(b.mul(tk, _c(b, NS)), g), _c(b, 4)), _c(b, OOB))
                b.buffer_store_cp(b.bitcast(gv, I32), self.g_rs, goff, _c(b, 0), aux=CPOL_SC1)
                ioff = b.select(okt, b.mul(b.add(b.mul(g, _c(b, self.TP)), tk), _c(b, 4)), _c(b, OOB))
                b.buffer_store_cp(b.bitcast(myinv, I32), self.hinv_rs, ioff, _c(b, 0), aux=CPOL_SC1)
        return out

    def _ybuf_put(self, tl: Value, lrow: Value, ybf: Value) -> None:
        b = self.b
        vals = []
        for d in range(2):
            w = b.vec_extract(ybf, d)
            for x in (_lo(b, w), _hi(b, w)):
                s = _bf16r(b, b.fmul(x, _f(b, 0.25)))
                if self.dmut == "silu":
                    vals.append(s)
                else:
                    vals.append(_bf16r(b, _silu(b, s, self.sig)))
        b.smem_store_vN(self.ybuf, [tl, lrow], b.vec_pack(vals, F32), 4)


class _Regs(list):
    """xissue's loads of one chunk plus their (Python-side) metadata for the fused staging."""


class _DnF(_Dn):
    """``cfuse``: the C stage inside the down GEMM's activation staging (DESIGN-LB S1, the E2 re-association). Each unit
    stages its own K range (one stream: ``splits`` a multiple of 4) from the raw inputs: per 16-byte piece (8 K of one
    token) ``r' = bf16(r + bf16(y inj))`` (row group 0 stores it), ``n = bf16(r' w')``, the MXFP8 block of 4 pieces
    (codes to the LDS staging rows, the E8M0 byte to an LDS scale array read per chunk by op_sel), and the piece's
    square sum and 4 injection dots, summed over the unit's chunks in registers. At the epilogue the unit stores its
    token sums (per row group, split and K group: no cross-row-group dependency); the split-K last arriver finishes
    ``inv`` per stream from them and folds ``sum_s inv_s sum_(splits of s) P`` (row group 0 also writes inv and the
    gates), then the activation pack (``yq``). No C stage and no C -> D edge."""

    def __init__(self, k: E1M.E1Knobs, scope, hk: "HcKnobs", TP: int):
        super().__init__(k, scope, hk.dmut, hk.sig, None, TP)
        self.hkf = hk
        b = self.b
        P = scope
        self.rin_rs = _rsrc(b, P.param("rin", I64), _c(b, TP * HCW * 2))
        self.rout_rs = _rsrc(b, P.param("rout", I64), _c(b, TP * HCW * 2))
        self.yin_rs = _rsrc(b, P.param("yin", I64), _c(b, hk.tp * TP * HID * 2))
        self.wn_rs = _rsrc(b, P.param("wn", I64), _c(b, HCW * 2))
        self.wj_rs = _rsrc(b, P.param("winj", I64), _c(b, NS * HCW * 2))
        self.gin_rs = _rsrc(b, P.param("gin", I64), _c(b, TP * NS * 4))
        self.gout_rs = _rsrc(b, P.param("gout", I64), _c(b, TP * NS * 4))
        self.hinv_rs = _rsrc(b, P.param("hinv", I64), _c(b, NS * TP * 4))
        self.spf_rs = _rsrc(b, P.param("sp", I64), _c(b, SPF_MAX * TP * 4))
        rows = k.tbt * TILE
        self.xsc = b.smem_alloc(F32, [2 * k.kw, rows + k.xjunk, 4], name_hint="hcxsc")
        # the chunk's shared operands (w' and the 4 injection rows: 5 x 128 bf16 = 80 pieces of 16 bytes), staged
        # cooperatively once per chunk instead of per token piece (VGPR footprint)
        self.wsh = b.smem_alloc(F32, [2 * k.kw, 80, 4], name_hint="hcwsh")
        self.ns = k.splits * k.kw  # stat slots per row group
        self._acc = None

    # ---- staging
    def xissue(self, u, c: int, gate=None) -> list:
        b, k, hk, TP = self.b, self.k, self.hkf, self.TP
        q = b.add(u.q0, _c(b, c))
        rows = k.tbt * TILE
        total = rows * 16
        tid, nthr = self._xthreads()
        out = _Regs()
        out.c, out.u, out.meta = c, u, []
        for it in range(-(-total // nthr)):
            v = b.add(tid, _c(b, it * nthr))
            live = b.cmp_lt(v, _c(b, total))
            vv = b.select(live, v, _c(b, total - 1))
            tl = E1M._udiv(b, vv, 16)
            pc = E1M._umod(b, vv, 16)
            tok = b.smin(b.add(u.tok0, tl), self.rlast)
            kk = b.add(b.mul(q, _c(b, CHUNK)), b.mul(pc, _c(b, 8)))
            stream = E1M._udiv(b, kk, HID)
            col = b.sub(kk, b.mul(stream, _c(b, HID)))
            ok = b.land(live, u.valid)
            ro = b.mul(b.add(b.mul(tok, _c(b, HCW)), kk), _c(b, 2))
            out.append(b.buffer_load_cp(self.rin_rs, b.select(ok, ro, _c(b, OOB)), _c(b, 0), dwords=4))
            yo = b.mul(b.add(b.mul(tok, _c(b, HID)), col), _c(b, 2))
            for rk in range(hk.tp):
                out.append(b.buffer_load_cp(self.yin_rs, b.select(ok, b.add(yo, _c(b, rk * TP * HID * 2)), _c(b, OOB)),
                                            _c(b, 0), dwords=4))  # fmt: skip
            goff = b.select(ok, b.mul(b.add(b.mul(tok, _c(b, NS)), stream), _c(b, 4)), _c(b, OOB))
            out.append(b.buffer_load_cp(self.gin_rs, goff, _c(b, 0), dwords=1))
            out.meta.append(dict(tl=tl, pc=pc, tok=tok, kk=kk, ro=ro, live=live, ok=ok, stream=stream))
        out.shared = []
        for it2 in range(-(-80 // nthr)):
            v2 = b.add(tid, _c(b, it2 * nthr))
            ok2 = b.land(b.cmp_lt(v2, _c(b, 80)), u.valid)
            row = E1M._udiv(b, v2, 16)
            kq = b.add(b.mul(q, _c(b, CHUNK)), b.mul(E1M._umod(b, v2, 16), _c(b, 8)))
            jrow = b.smax(b.sub(row, _c(b, 1)), _c(b, 0))
            wo = b.mul(kq, _c(b, 2))
            jo = b.mul(b.add(kq, b.mul(jrow, _c(b, HCW))), _c(b, 2))
            lw = b.buffer_load_cp(self.wn_rs, b.select(b.land(ok2, b.cmp_eq(row, _c(b, 0))), wo, _c(b, OOB)), _c(b, 0), dwords=4)
            lj = b.buffer_load_cp(self.wj_rs, b.select(b.land(ok2, b.cmp_gt(row, _c(b, 0))), jo, _c(b, OOB)), _c(b, 0), dwords=4)
            out.shared.append((v2, b.cmp_lt(v2, _c(b, 80)), b.select(b.cmp_eq(row, _c(b, 0)), lw, lj)))
        return out

    def xput(self, regs, buf) -> None:
        b, k, hk = self.b, self.k, self.hkf
        mut = hk.c.mutant
        rows = k.tbt * TILE
        bufg = self.xbuf(buf)
        per = 2 + hk.tp
        for v2, in80, val in regs.shared:
            with b.scf_if(in80):
                b.smem_store_vN(self.wsh, [bufg, v2, _c(b, 0)], b.vec_bitcast(val, VectorType(F32, 4)), 4)
        b.sync_lds_only()
        if regs.c == 0:
            self._acc = [dict(ss=_f(b, 0.0), d=[_f(b, 0.0)] * NS) for _ in regs.meta]
            self._meta = regs.meta
            self._u = regs.u
        u = regs.u
        st_ok = b.cmp_eq(u.rg, _c(b, 0))  # row group 0 stores r' (every K group owns its own K range)
        for it, m in enumerate(regs.meta):
            vals = regs[it * per : (it + 1) * per]
            r, ys, gw = vals[0], vals[1 : 1 + hk.tp], vals[1 + hk.tp]
            gate = b.bitcast(gw, F32)
            w = b.vec_bitcast(b.smem_load_vN_f32(self.wsh, bufg, m["pc"], _c(b, 0), n=4), VectorType(I32, 4))
            js = [b.vec_bitcast(b.smem_load_vN_f32(self.wsh, bufg, b.add(m["pc"], _c(b, 16 * (j + 1))), _c(b, 0), n=4),
                                VectorType(I32, 4)) for j in range(NS)]  # fmt: skip
            rp, n = [], []
            for e in range(8):
                rw = b.vec_extract(r, e // 2)
                rv = _hi(b, rw) if e % 2 else _lo(b, rw)
                yw = [b.vec_extract(yy, e // 2) for yy in ys]
                yv = _hi(b, yw[0]) if e % 2 else _lo(b, yw[0])
                for yx in yw[1:]:
                    yv = _bf16r(b, b.fadd(yv, _hi(b, yx) if e % 2 else _lo(b, yx)))
                tt = yv if mut == "ygate" else _bf16r(b, b.fmul(yv, gate))
                rp.append(_bf16r(b, b.fadd(rv, tt)))
                ww = b.vec_extract(w, e // 2)
                n.append(_bf16r(b, b.fmul(rp[e], _hi(b, ww) if e % 2 else _lo(b, ww))))
            if mut != "rskip":
                b.buffer_store_cp(_pack8(b, rp), self.rout_rs, b.select(b.land(m["ok"], st_ok), m["ro"], _c(b, OOB)),
                                  _c(b, 0), aux=CPOL_SC1)  # fmt: skip
            words, E = MQ.quantize_mxfp8(b, n, 4)
            if mut == "xblk":
                E = b.select(b.cmp_eq(m["stream"], _c(b, 1)), b.add(E, _c(b, 1)), E)
            tl = m["tl"]
            if k.xjunk:
                tl = b.select(m["live"], tl, _c(b, rows))
            cv = b.vec_bitcast(b.vec_pack(words, I32), VectorType(I8, 8))
            pred = None if k.xjunk else m["live"]
            with contextlib.nullcontext() if pred is None else b.scf_if(pred):
                b.smem_store_vN(self.xs, [bufg, tl, b.mul(m["pc"], _c(b, 8))], cv, 8)
            lead = b.cmp_eq(b.land(m["pc"], _c(b, 3)), _c(b, 0))
            with b.scf_if(b.land(lead, m["live"]) if not k.xjunk else lead):
                b.smem_store_vN(self.xsc, [bufg, tl, b.lshr(m["pc"], _c(b, 2))], b.bitcast(E, F32), 1)
            q = [b.fmul(x, x) for x in rp]
            s8 = b.fadd(b.fadd(b.fadd(q[0], q[1]), b.fadd(q[2], q[3])), b.fadd(b.fadd(q[4], q[5]), b.fadd(q[6], q[7])))
            a = self._acc[it]
            a["ss"] = b.fadd(a["ss"], b.select(m["ok"], s8, _f(b, 0.0)))
            dn = []
            for j in range(NS):
                x = js[j]
                acc = a["d"][j]
                for e in range(8):
                    xw = b.vec_extract(x, e // 2)
                    acc = b.fma(_hi(b, xw) if e % 2 else _lo(b, xw), n[e], acc)
                dn.append(acc)
            a["d"] = dn
        # pin the chunk's running sums here (opaque copies): otherwise LLVM sinks every chunk's sums to the epilogue,
        # keeping all chunks' n values and injection rows alive (512 VGPRs + scratch at 10 chunks a unit)
        pin = lambda x: b.bitcast(D.fresh(b, b.bitcast(x, I32)), F32)  # noqa: E731
        for a in self._acc:
            a["ss"] = pin(a["ss"])
            a["d"] = [pin(x) for x in a["d"]]

    # ---- MFMA with the X scales from LDS (byte j of the op_sel scale operand, per chunk)
    def _sop_math(self, acc, wl, buf, skip, j, u):
        b, k = self.b, self.k
        acc = [list(r) for r in acc]
        if j == 0:
            nws = k.mt * (2 if k.sop_funnel else 1)
            ws = wl[k.mt : k.mt + nws]
            if k.sop_funnel:
                sh = b.mul(b.land(u.q0, _c(b, 3)), _c(b, 8))
                sc = [E1M._funnel(b, ws[i], ws[k.mt + i], sh) for i in range(k.mt)]
            else:
                sc = list(ws)
            if k.sop_short:
                sh = b.mul(b.land(u.q0, _c(b, 3)), _c(b, 8))
                sc = [b.lshr(v, sh) for v in sc]
            self.sop_sc = sc
        SA = self.sop_sc[: k.mt]
        zw = b.vec_pack([_c(b, 0)] * 4, I32) if skip else None
        js = (j + 1) % 4 if k.mutant == "sopsel" else j
        bufg = self.xbuf(buf)
        for t in range(k.tbt):
            row = b.add(_c(b, t * TILE), self.l16)
            e = b.bitcast(b.vec_extract(b.smem_load_vN_f32(self.xsc, bufg, row, self.g, n=1), 0), I32)
            sb = b.shl(e, _c(b, 8 * js))
            bv = self.bop(t, buf, None)[0]
            for mi in range(k.mt):
                acc[mi][t] = b.mfma_scale_mx(zw if skip else wl[mi], bv, acc[mi][t], SA[mi], sb, shape="16x16x128",
                                             fmt_a="fp4", fmt_b="fp8", opsel_a=js, opsel_b=js)  # fmt: skip
        return acc

    def wissue(self, u, c: int, xonly_gate=None, xonly: bool = False) -> list:
        """E1's sop weight loads without the X scale dwords (the X scales are staged in LDS)."""
        out = super().wissue(u, c, xonly_gate, xonly)
        if c % 4 == 0:
            k = self.k
            ng = 2 if k.sop_funnel else 1
            out = out[: len(out) - ng * k.tbt]
        return out

    # ---- the unit's statistics, then E1's split-K protocol with the per-stream fold
    def _sidx(self, u, f: int, tok: Value, split_slot: Value) -> Value:
        b = self.b
        return b.mul(b.add(b.mul(b.add(b.mul(b.add(b.mul(u.rg, _c(b, self.ns)), split_slot), _c(b, 5)), _c(b, f)),
                                 _c(b, self.TP)), tok), _c(b, 4))  # fmt: skip

    def epilogue(self, u, acc, segv=None) -> None:
        b, k = self.b, self.k
        slot = b.add(b.mul(u.s, _c(b, k.kw)), self.kg) if k.kw > 1 else u.s
        for it, m in enumerate(self._meta):
            a = self._acc[it]
            vals = [a["ss"]] + list(a["d"])
            red = []
            for v in vals:
                for off in (1, 2, 4, 8):
                    v = b.fadd(v, b.warp_shuffle_xor(v, off))
                red.append(v)
            okw = b.land(m["ok"], b.cmp_eq(m["pc"], _c(b, 0)))
            tokc = b.add(u.tok0, m["tl"])
            for f, v in enumerate(red):
                off = self._sidx(u, f, tokc, slot)
                b.buffer_store_cp(b.bitcast(v, I32), self.spf_rs, b.select(okw, off, _c(b, OOB)), _c(b, 0), aux=CPOL_SC1)
        super().epilogue(u, acc, segv)

    def _stream_sum(self, u, f: int, s: int, tok: Value, rg: Value | None = None) -> Value:
        """Sum over the stat slots (split, K group) of stream ``s`` of field ``f`` for token ``tok`` (slot order)."""
        b, k = self.b, self.k
        per = self.ns // NS
        tot = None
        for i in range(per):
            off = self._sidx(u, f, tok, _c(b, s * per + i))
            x = b.bitcast(b.buffer_load_cp(self.spf_rs, off, _c(b, 0), dwords=1, aux=CPOL_SC1), F32)
            tot = x if tot is None else b.fadd(tot, x)
        return tot

    def _ss(self, u, s: int, tok: Value) -> Value:
        return self._stream_sum(u, 0, s, tok)

    def _dot(self, u, s: int, j: Value, tok: Value) -> Value:
        """Injection dot ``j`` (a runtime lane index) of stream ``s``: the slots of the stream in slot order."""
        b = self.b
        per_s = self.ns // NS
        dsum = None
        for i in range(per_s):
            off = b.mul(b.add(b.mul(b.add(b.mul(b.add(b.mul(u.rg, _c(b, self.ns)), _c(b, s * per_s + i)), _c(b, 5)),
                                          b.add(_c(b, 1), j)), _c(b, self.TP)), tok), _c(b, 4))  # fmt: skip
            x = b.bitcast(b.buffer_load_cp(self.spf_rs, off, _c(b, 0), dwords=1, aux=CPOL_SC1), F32)
            dsum = x if dsum is None else b.fadd(dsum, x)
        return dsum

    def fold(self, u, p_rs, part=None) -> None:
        """The per-stream split-K fold: ``Y = sum_s inv_s sum_(splits of s) P`` (split order inside a stream, streams in
        order), inv finished here from the statistics; row group 0 also writes inv (U) and the gates."""
        b, k, hk = self.b, self.k, self.hkf
        mut = hk.c.mutant
        TP = k.tiles * TILE
        R = k.rows_per_unit
        q4 = R // 4
        pieces = k.tbt * TILE * q4
        y_rs = _rsrc(b, u.y_a, self.yl32)
        S = k.splits
        per = S // NS  # splits of one stream
        drop = k.mutant == "fold"
        for it in range(-(-pieces // k.threads)):
            v = b.add(self.tid, _c(b, it * k.threads))
            tl = E1M._udiv(b, v, q4)
            row = b.add(b.mul(u.rg, _c(b, R)), b.mul(E1M._umod(b, v, q4), _c(b, 4)))
            tok = b.add(u.tok0, tl)
            ok = b.land(b.cmp_lt(v, _c(b, pieces)), b.cmp_lt(tok, self.rows))
            tk = b.select(ok, tok, _c(b, 0))
            invs = [_inv_rule(b, self._ss(u, s, tk), mut, hk.c.inv) for s in range(NS)]
            tot = None
            for s in range(NS):
                part_s = None
                for i in range(per):
                    sp_ = s * per + i
                    if drop and sp_ == S - 1:
                        continue
                    off = b.mul(b.add(b.mul(b.add(_c(b, sp_ * TP), tk), self.Mpad), row), _c(b, 4))
                    f4 = b.vec_bitcast(b.buffer_load_cp(p_rs, b.select(ok, off, _c(b, OOB)), _c(b, 0), dwords=4, aux=CPOL_SC1),
                                       VectorType(F32, 4))  # fmt: skip
                    x = [b.vec_extract(f4, e) for e in range(4)]
                    part_s = x if part_s is None else [b.fadd(part_s[e], x[e]) for e in range(4)]
                if part_s is None:  # the fold mutant dropped this stream's only split
                    part_s = [_f(b, 0.0)] * 4
                sc = [b.fmul(part_s[e], invs[s]) for e in range(4)]
                tot = sc if tot is None else [b.fadd(tot[e], sc[e]) for e in range(4)]
            yv = b.vec_bitcast(b.vec_trunc_f32_to_bf16(b.vec_pack(tot, F32)), VectorType(I32, 2))
            yoff = b.mul(b.add(b.mul(tok, self.Mpad), row), _c(b, 2))
            b.buffer_store_cp(yv, y_rs, b.select(ok, yoff, _c(b, OOB)), _c(b, 0))
            if k.yq:
                with b.scf_if(b.cmp_lt(v, _c(b, pieces))):
                    self._ybuf_put(tl, b.mul(E1M._umod(b, v, q4), _c(b, 4)), yv)
        # row group 0: inv (for U) and the gates of the token block
        nt = k.tbt * TILE * NS
        with b.scf_if(b.cmp_eq(u.rg, _c(b, 0))):
            for it in range(-(-nt // k.threads)):
                v = b.add(self.tid, _c(b, it * k.threads))
                tl = E1M._udiv(b, v, NS)
                j = E1M._umod(b, v, NS)
                tok = b.add(u.tok0, tl)
                ok = b.land(b.cmp_lt(v, _c(b, nt)), b.cmp_lt(tok, self.rows))
                tk = b.select(ok, tok, _c(b, 0))
                invs = [_inv_rule(b, self._ss(u, s, tk), mut, hk.c.inv) for s in range(NS)]
                tot = None
                for s in range(NS - 1 if mut == "inj" else NS):
                    vv = b.fmul(self._dot(u, s, j, tk), invs[s])
                    tot = vv if tot is None else b.fadd(tot, vv)
                gv = _gate_rule(b, tot, hk.sig)
                myinv = b.select(b.cmp_eq(j, _c(b, 0)), invs[0], b.select(b.cmp_eq(j, _c(b, 1)), invs[1],
                                 b.select(b.cmp_eq(j, _c(b, 2)), invs[2], invs[3])))  # fmt: skip
                goff = b.select(ok, b.mul(b.add(b.mul(tk, _c(b, NS)), j), _c(b, 4)), _c(b, OOB))
                b.buffer_store_cp(b.bitcast(gv, I32), self.gout_rs, goff, _c(b, 0), aux=CPOL_SC1)
                ioff = b.select(ok, b.mul(b.add(b.mul(j, _c(b, self.TP)), tk), _c(b, 4)), _c(b, OOB))
                b.buffer_store_cp(b.bitcast(myinv, I32), self.hinv_rs, ioff, _c(b, 0), aux=CPOL_SC1)


class _DnS(_Dn):
    """``cstat="dfold"``: the down GEMM over C's MXFP8 activations (E1's own staging / loads, ``kseg`` off) with the
    per-stream fold of :class:`_DnF`: the split-K last arriver finishes inv from C's per-part square sums, scales each
    stream's split sum, and (row group 0) writes inv and the gates from C's per-part injection dots. No statistics on
    any unit's critical path before its GEMM; C has no cross-wave step."""

    fold = _DnF.fold

    def __init__(self, k: E1M.E1Knobs, scope, hk: "HcKnobs", TP: int):
        super().__init__(k, scope, hk.dmut, hk.sig, None, TP)
        self.hkf = hk
        b = self.b
        self.sp_rs = _rsrc(b, scope.param("sp", I64), _c(b, SP_MAX * TP * 4))
        self.gout_rs = _rsrc(b, scope.param("gout", I64), _c(b, TP * NS * 4))
        self.hinv_rs = _rsrc(b, scope.param("hinv", I64), _c(b, NS * TP * 4))

    def _ss(self, u, s: int, tok: Value) -> Value:
        b, cp = self.b, self.hkf.cp
        tot = None
        for p in range(cp):
            off = b.mul(b.add(b.mul(_c(b, p * NS + s), _c(b, self.TP)), tok), _c(b, 4))
            x = b.bitcast(b.buffer_load_cp(self.sp_rs, off, _c(b, 0), dwords=1, aux=CPOL_SC1), F32)
            tot = x if tot is None else b.fadd(tot, x)
        return tot

    def _dot(self, u, s: int, j: Value, tok: Value) -> Value:
        b, cp = self.b, self.hkf.cp
        tot = None
        for p in range(cp):
            idx = b.add(_c(b, cp * NS + (p * NS + s) * NS), j)
            off = b.mul(b.add(b.mul(idx, _c(b, self.TP)), tok), _c(b, 4))
            x = b.bitcast(b.buffer_load_cp(self.sp_rs, off, _c(b, 0), dwords=1, aux=CPOL_SC1), F32)
            tot = x if tot is None else b.fadd(tot, x)
        return tot


# ------------------------------------------------------------------ stage U (E1 + mix epilogue + x pack)


class _Up(_RegPre, E1M._E1):
    """E1 over the row-permuted up weights; epilogue: the 4 streams of a column (a lane's 4 accumulator rows) mixed in
    registers, then ``xq_pass`` (bf16 x + MXFP8 x in the op_sel layout)."""

    def __init__(self, k: E1M.E1Knobs, scope, hk: HcKnobs, TP: int):
        with G._swap(E1M, "IRBuilder", lambda _n: scope):
            super().__init__(k)
        b = self.b
        self.hk, self.TP = hk, TP
        self.NC = k.rows_per_unit // 4
        self.xmix = b.smem_alloc(F32, [k.tbt * TILE, self.NC], name_hint="hcxb")
        P = scope
        self.rp_rs = _rsrc(b, P.param("rp", I64), _c(b, TP * HCW * 2))
        self.inv_rs = _rsrc(b, P.param("hinv", I64), _c(b, NS * TP * 4))
        self.wn_rs = _rsrc(b, P.param("wn", I64), _c(b, HCW * 2))
        self.xo_rs = _rsrc(b, P.param("xo", I64), _c(b, TP * HID * 2))
        self.xo8_rs = _rsrc(b, P.param("xo8", I64), _c(b, TP * HID))
        self.xo8s_a = b.readfirstlane(P.param("xo8s", I64))
        if hk.dbg:
            self.du_rs = _rsrc(b, P.param("du", I64), _c(b, TP * HCW * 4))

    def _lcol(self, mi: int) -> Value:
        b = self.b
        return b.add(b.lshr(b.sub(self.wrow[mi], self.l16), _c(b, 2)), self.g)

    def _eops(self, u):
        """The epilogue's operands per weight tile: w' (4 dwords), then per token tile r' (4 dwords) and inv (4);
        ``eload="t"``: the inv words once per token tile (they do not depend on the weight tile), then per weight tile
        w' and per token tile r' (a dict)."""
        b, k = self.b, self.k
        if self.hk.eload == "t":
            res = dict(i={}, w={}, r={})
            for t in range(k.tbt):
                tok = b.smin(b.add(u.tok0, b.add(_c(b, t * TILE), self.l16)), self.rlast)
                res["i"][t] = [b.buffer_load_cp(self.inv_rs, b.mul(b.add(_c(b, e * self.TP), tok), _c(b, 4)), _c(b, 0),
                                                dwords=1) for e in range(NS)]  # fmt: skip
            for mi in range(k.mt):
                col = b.add(b.mul(u.rg, _c(b, self.NC)), self._lcol(mi))
                cev = b.land(col, _c(b, -2))
                res["w"][mi] = [b.buffer_load_cp(self.wn_rs, b.mul(b.add(cev, _c(b, e * HID)), _c(b, 2)), _c(b, 0), dwords=1)
                                for e in range(NS)]  # fmt: skip
                for t in range(k.tbt):
                    tok = b.smin(b.add(u.tok0, b.add(_c(b, t * TILE), self.l16)), self.rlast)
                    res["r"][(mi, t)] = [b.buffer_load_cp(self.rp_rs, b.mul(b.add(b.mul(tok, _c(b, HCW)), b.add(cev, _c(b, e * HID))),
                                                                            _c(b, 2)), _c(b, 0), dwords=1) for e in range(NS)]  # fmt: skip
            return res
        out = []
        for mi in range(k.mt):
            col = b.add(b.mul(u.rg, _c(b, self.NC)), self._lcol(mi))
            cev = b.land(col, _c(b, -2))
            for e in range(NS):
                out.append(b.buffer_load_cp(self.wn_rs, b.mul(b.add(cev, _c(b, e * HID)), _c(b, 2)), _c(b, 0), dwords=1))
            for t in range(k.tbt):
                tok = b.smin(b.add(u.tok0, b.add(_c(b, t * TILE), self.l16)), self.rlast)
                for e in range(NS):
                    off = b.mul(b.add(b.mul(tok, _c(b, HCW)), b.add(cev, _c(b, e * HID))), _c(b, 2))
                    out.append(b.buffer_load_cp(self.rp_rs, off, _c(b, 0), dwords=1))
                for e in range(NS):
                    out.append(b.buffer_load_cp(self.inv_rs, b.mul(b.add(_c(b, e * self.TP), tok), _c(b, 4)), _c(b, 0),
                                                dwords=1))  # fmt: skip
        return out

    _stampf = None

    def seg_loads(self, u) -> list:
        if self._stampf is not None:
            self._stampf(10)
        return self._eops(u) if self.hk.epre else []

    def epilogue(self, u, acc, segv=None) -> None:
        if self._stampf is not None:
            self._stampf(11)
        self._epilogue_mix(u, acc, segv)
        if self._stampf is not None:
            self._stampf(12)

    def _epilogue_mix(self, u, acc, segv=None) -> None:
        b, k, hk = self.b, self.k, self.hk
        ops = segv if hk.epre else self._eops(u)
        acc = self.kw_reduce(acc)
        g0 = b.cmp_eq(self.kg, _c(b, 0)) if k.kw > 1 else None
        i = 0
        for mi in range(k.mt):
            lc = self._lcol(mi)
            col = b.add(b.mul(u.rg, _c(b, self.NC)), lc)
            odd = b.land(col, _c(b, 1))
            if isinstance(ops, dict):
                wv = [_half(b, x, odd) for x in ops["w"][mi]]
            else:
                wv = [_half(b, ops[i + e], odd) for e in range(NS)]
                i += NS
            for t in range(k.tbt):
                if isinstance(ops, dict):
                    rv = [_half(b, x, odd) for x in ops["r"][(mi, t)]]
                    iv = [b.bitcast(x, F32) for x in ops["i"][t]]
                else:
                    rv = [_half(b, ops[i + e], odd) for e in range(NS)]
                    iv = [b.bitcast(ops[i + NS + e], F32) for e in range(NS)]
                    i += 2 * NS
                a4 = acc[mi][t]
                ps = []
                if hk.uvec == "pk":  # the same arithmetic on packed f32 pairs (v_pk_*, v_cvt_pk_bf16_f32): the same bits
                    for e0 in (0, 2):
                        a2 = _unpk(b, _pkw(b, b.vec_pack([b.vec_extract(a4, e0), b.vec_extract(a4, e0 + 1)], F32)))
                        if hk.umut == "sig":
                            rw2 = a2
                        else:
                            sg = [_sigmoid(b, b.vec_extract(a2, i), hk.sig) for i in range(2)]
                            rw2 = _unpk(b, _pkw(b, b.vec_pack(sg, F32)))
                        r2 = b.vec_pack([rv[e0], rv[e0 + 1]], F32)
                        i2 = b.vec_pack([iv[e0], iv[e0 + 1]], F32)
                        w2 = b.vec_pack([wv[e0], wv[e0 + 1]], F32)
                        yn2 = _unpk(b, _pkw(b, b.fmul(b.fmul(r2, i2), w2)))
                        p2 = _unpk(b, _pkw(b, b.fmul(rw2, yn2)))
                        ps += [b.vec_extract(p2, 0), b.vec_extract(p2, 1)]
                for e in range(0 if hk.uvec == "pk" else NS):
                    a = _bf16r(b, b.vec_extract(a4, e))
                    rw = a if hk.umut == "sig" else _bf16r(b, _sigmoid(b, a, hk.sig))
                    yn = _bf16r(b, b.fmul(b.fmul(rv[e], iv[e]), wv[e]))
                    ps.append(_bf16r(b, b.fmul(rw, yn)))
                s = b.fadd(b.fadd(ps[0], ps[1]), ps[2])
                if hk.umut != "mix":
                    s = b.fadd(s, ps[3])
                x = _bf16r(b, b.fmul(s, _f(b, 0.25)))
                tl = b.add(_c(b, t * TILE), self.l16)
                cond = g0
                with contextlib.nullcontext() if cond is None else b.scf_if(cond):
                    b.smem_store_vN(self.xmix, [tl, lc], x, 1)
                    if hk.dbg:
                        tok = b.add(u.tok0, tl)
                        ok = b.land(u.valid, b.cmp_lt(tok, self.rows))
                        row = b.mul(col, _c(b, NS))
                        off = b.mul(b.add(b.mul(tok, _c(b, HCW)), row), _c(b, 4))
                        b.buffer_store_cp(b.vec_bitcast(a4, VectorType(I32, 4)), self.du_rs, b.select(ok, off, _c(b, OOB)),
                                          _c(b, 0))  # fmt: skip
        self.xq_pass(u)

    def xq_pass(self, u) -> None:
        """bf16 x and MXFP8 x per (token, 32-column block): 4 lanes x 8 columns; codes ``xo8[tok][col]``, the E8M0 byte
        at ``xo8s[(G bucket16 + tok) 16 + 4 g + j]`` for chunk ``q = col / 128`` (``G = q / 4``, ``j = q mod 4``, block
        ``g``), bf16 ``xo[tok][col]``. Write-through stores."""
        b, k, hk = self.b, self.k, self.hk
        NC, TP = self.NC, self.TP
        nb = NC // 32
        TPT = k.tbt * TILE
        total = TPT * nb * 4
        b.sync_lds_only()
        for it in range(-(-total // k.threads)):
            v = b.add(self.tid, _c(b, it * k.threads))
            vin = b.cmp_lt(v, _c(b, total))
            vv = b.select(vin, v, _c(b, 0))
            p = E1M._umod(b, vv, 4)
            blk = E1M._umod(b, E1M._udiv(b, vv, 4), nb)
            tl = E1M._udiv(b, vv, 4 * nb)
            lcol = b.add(b.mul(blk, _c(b, 32)), b.mul(p, _c(b, 8)))
            lo = b.smem_load_vN_f32(self.xmix, tl, lcol, n=4)
            hi = b.smem_load_vN_f32(self.xmix, tl, b.add(lcol, _c(b, 4)), n=4)
            vals = [b.vec_extract(lo, e) for e in range(4)] + [b.vec_extract(hi, e) for e in range(4)]
            words, E = MQ.quantize_mxfp8(b, vals, 4)
            if hk.umut == "xs":
                E = b.add(E, _c(b, 1))
            tok = b.add(u.tok0, tl)
            col = b.add(b.mul(u.rg, _c(b, NC)), lcol)
            ok = b.land(b.land(vin, u.valid), b.land(b.cmp_lt(tok, self.rows), b.cmp_lt(col, _c(b, HID))))
            coff = b.add(b.mul(tok, _c(b, HID)), col)
            b.buffer_store_cp(b.vec_pack(words, I32), self.xo8_rs, b.select(ok, coff, _c(b, OOB)), _c(b, 0), aux=CPOL_SC1)
            b.buffer_store_cp(_pack8(b, vals), self.xo_rs, b.select(ok, b.mul(coff, _c(b, 2)), _c(b, OOB)), _c(b, 0),
                              aux=CPOL_SC1)  # fmt: skip
            q = b.lshr(col, _c(b, 7))
            gq = b.land(b.lshr(col, _c(b, 5)), _c(b, 3))
            soff = b.add(b.mul(b.add(b.mul(b.add(b.mul(b.lshr(q, _c(b, 2)), _c(b, TP)), tok), _c(b, 4)), gq), _c(b, 4)),
                         b.land(q, _c(b, 3)))  # fmt: skip
            with b.scf_if(b.land(ok, b.cmp_eq(p, _c(b, 0)))):
                b.inline_asm("global_store_byte $0, $1, off sc1", "v,v,~{memory}", [b.add(self.xo8s_a, _i64(b, soff)), E],
                             None)  # fmt: skip
        b.sync_lds_only()


# ------------------------------------------------------------------ the chain


class _HcChain:
    def __init__(self, hk: HcKnobs, padw: int | None = None):
        hk.check()
        self.hk = hk
        self.padw = padw
        self.dk, self.uk = hk.parts()
        b = self.b = IRBuilder(hc_kernel_name(hk))
        b.kernel.attrs["max_workgroup_size"] = 64 * hk.waves
        P = {n: b.param(n, I64) for n in HC_I64}
        for n in HC_I32:
            P[n] = b.param(n, I32)
        self.A = {n: b.readfirstlane(v) for n, v in P.items()}
        self.bid = b.readfirstlane(b.block_id_x())
        self.TP = hk.tbk
        self.S = strides(hk)
        self.gsync = PS.GridSync(grid=GRID, stripes=8, pause="s_nop")

    def scope(self, sfx: str, params: dict, inlaunch=()) -> G._Scope:
        load = self.hk.acq == "load" and self.hk.mutant != "noacq"
        return G._FreshScope(self.b, sfx, params, None, inlaunch if load else ())

    def stages(self, li: Value, lay: Value) -> None:
        b, A, hk, S, TP = self.b, self.A, self.hk, self.S, self.TP
        c0, c1 = _c(b, 0), _c(b, 1)
        rows = A["rows"]
        la = b.readfirstlane(b.mod(lay, A["am"]))
        la1 = b.readfirstlane(b.mod(b.add(lay, c1), A["am"]))
        lw = b.readfirstlane(b.mod(lay, A["wm"]))

        def at(n, slot):
            return _addl(b, A[n], slot, S[n])

        rin, rout, gin, gout = at("r", la), at("r", la1), at("g", la), at("g", la1)
        xq, xqs, inv = at("xq", la), at("xqs", la), at("inv", la)
        aq, aqs = at("aq", la), at("aqs", la)
        wn = at("wn", lw)
        if hk.cfuse:
            return self.stages_fused(li, lay, rin, rout, gin, gout, inv, aq, aqs, wn, la, lw, at)
        e0 = b.mul(li, _c(b, 3))
        pgate = dict(rout=rout, inv=inv, gout=gout, wn=wn, winj=at("winj", lw), rows=rows) if hk.gplace != "c" else None
        self._stamp(li, 0)
        if "C" in hk.stages:
            self.late()
            pc = dict(rin=rin, rout=rout, gin=gin, gout=gout, y=at("y", la), wn=wn, winj=at("winj", lw), xq=xq, xqs=xqs,
                      inv=inv, rows=rows, sp=at("sp", la))
            if hk.cstat == "cfold":  # the C tokens' arrival counters: a quarter of this layer's counter block
                pc["ccnt"] = b.add(_addl(b, A["cnt"], li, CNT_LAYER), b.const_i64(3 * CNT_LAYER // 4))
            _Comb(hk, self.scope("_c", pc, inlaunch=(rin, gin)), pc, TP).emit()

        def e1p(w, ws, x, xs, y, cnt, m):
            return dict(w=w, ws=ws, x=x, xs=xs, y=y, part=A["part"], cnt=cnt, gs=A["gs"], m=_c(b, m), rows=rows, layers=c1,
                        gsync=c0)

        cnt_l = _addl(b, A["cnt"], li, CNT_LAYER)
        dn = None
        if "D" in hk.stages:
            pd = e1p(at("wd", lw), at("wds", lw), xq, xqs, at("dy", la), cnt_l, LR)
            pd.update(yqc=aq, yqs=aqs)
            if self.dk.kseg:
                pd["seg"] = inv
            spl = at("sp", la)
            pd.update(sp=spl, hinv=inv, gout=gout)
            if hk.cstat == "dfold":
                dn = _DnS(self.dk, self.scope("_d", pd, inlaunch=(xq, xqs, inv, spl)), hk, TP)
            else:
                dn = _Dn(self.dk, self.scope("_d", pd, inlaunch=(xq, xqs, inv, spl)), hk.dmut, hk.sig, hk, TP)
            if hk.stamp:
                dn._stampf = lambda ev: self._stamp(li, ev)
        self._stamp(li, 1)
        pre1 = None
        if dn is not None and "d" in hk.pre:
            pre1 = lambda: self.prefetch(dn)  # noqa: E731
        elif dn is not None and "D" in hk.pre:
            pre1 = dn.pre_issue
        self.edge(b.add(e0, c1), pre=pre1)
        self._stamp(li, 2)
        if dn is not None:
            self.late()
            dn.build()
        if hk.gplace == "d" and "C" in hk.stages:
            _Gates(hk, self.scope("_g", pgate, inlaunch=(rout, inv)), pgate, TP).emit()
        self._stamp(li, 3)
        up = None
        if "U" in hk.stages:
            pu = e1p(at("wu", lw), at("wus", lw), aq, aqs, A["ydump"], b.add(cnt_l, b.const_i64(CNT_LAYER // 2)), HCW)
            pu.update(rp=rout, hinv=inv, wn=wn, xo=at("xo", la), xo8=at("xo8", la), xo8s=at("xo8s", la), du=at("du", la))
            up = _Up(self.uk, self.scope("_u", pu, inlaunch=(aq, aqs, rout, inv)), hk, TP)
            if hk.stamp:
                up._stampf = lambda ev: self._stamp(li, ev)
        pre2 = None
        if up is not None and "u" in hk.pre:
            pre2 = lambda: self.prefetch(up)  # noqa: E731
        elif up is not None and "U" in hk.pre:
            pre2 = up.pre_issue
        self.edge(b.add(e0, _c(b, 2)), pre=pre2)
        self._stamp(li, 4)
        if up is not None:
            self.late()
            up.build()
        if hk.gplace == "u" and "C" in hk.stages:
            _Gates(hk, self.scope("_g", pgate, inlaunch=(rout, inv)), pgate, TP).emit()
        self._stamp(li, 5)
        self.edge(b.add(e0, _c(b, 3)), pre=self._pre_next(lay) if hk.pre != "none" and "n" in hk.pre and "D" in hk.stages else None)
        self._stamp(li, 6)

    def _stamp(self, li: Value, ev: int) -> None:
        if not self.hk.stamp:
            return
        b, A = self.b, self.A
        with b.scf_if(D.is_thread0(b)):
            rs = b.buffer_rsrc(_ptr(b, A["du"]), _c(b, 0x7FFFFFF0))
            off = b.mul(b.add(b.mul(b.add(b.mul(li, _c(b, GRID)), self.bid), _c(b, 16)), _c(b, ev)), _c(b, 4))
            b.buffer_store_cp(b.trunc(D.memrealtime(b), I32), rs, off, _c(b, 0))

    def _pre_next(self, lay):
        """The next layer's first down-GEMM weight chunks (a weight-only E1 instance on the next weight set)."""
        b, A = self.b, self.A
        lwn = b.readfirstlane(b.mod(b.add(lay, _c(b, 1)), A["wm"]))
        c0, c1 = _c(b, 0), _c(b, 1)
        p = dict(w=_addl(b, A["wd"], lwn, self.S["wd"]), ws=_addl(b, A["wds"], lwn, self.S["wds"]), x=A["ydump"],
                 xs=A["ydump"], y=A["ydump"], part=A["part"], cnt=A["cnt"], gs=A["gs"], m=_c(b, LR), rows=A["rows"],
                 layers=c1, gsync=c0, seg=A["ydump"], yqc=A["ydump"], yqs=A["ydump"])  # fmt: skip
        g = _Dn(self.dk, self.scope("_pn", p))
        return lambda: self.prefetch(g)

    def stages_fused(self, li, lay, rin, rout, gin, gout, inv, aq, aqs, wn, la, lw, at) -> None:
        """cfuse: D (with the combine in its staging) -> edge -> U -> edge."""
        b, A, hk, TP = self.b, self.A, self.hk, self.TP
        c0, c1 = _c(b, 0), _c(b, 1)
        rows = A["rows"]
        e0 = b.mul(li, _c(b, 2))
        cnt_l = _addl(b, A["cnt"], li, CNT_LAYER)

        def e1p(w, ws, x, xs, y, cnt, m):
            return dict(w=w, ws=ws, x=x, xs=xs, y=y, part=A["part"], cnt=cnt, gs=A["gs"], m=_c(b, m), rows=rows, layers=c1,
                        gsync=c0)

        if "D" in hk.stages:
            spl = at("sp", la)
            pd = e1p(at("wd", lw), at("wds", lw), at("xq", la), at("xqs", la), at("dy", la), cnt_l, LR)
            pd.update(yqc=aq, yqs=aqs, rin=rin, rout=rout, yin=at("y", la), wn=wn, winj=at("winj", lw), gin=gin, gout=gout,
                      hinv=inv, sp=spl)  # fmt: skip
            self.late()
            _DnF(self.dk, self.scope("_d", pd, inlaunch=(rin, gin, spl)), hk, TP).build()
        up = None
        if "U" in hk.stages:
            pu = e1p(at("wu", lw), at("wus", lw), aq, aqs, A["ydump"], b.add(cnt_l, b.const_i64(CNT_LAYER // 2)), HCW)
            pu.update(rp=rout, hinv=inv, wn=wn, xo=at("xo", la), xo8=at("xo8", la), xo8s=at("xo8s", la), du=at("du", la))
            up = _Up(self.uk, self.scope("_u", pu, inlaunch=(aq, aqs, rout, inv)), hk, TP)
        self.edge(b.add(e0, c1), pre=(lambda: self.prefetch(up)) if (up is not None and "u" in hk.pre) else None)
        if up is not None:
            self.late()
            up.build()
        self.edge(b.add(e0, _c(b, 2)), pre=self._pre_next(lay) if hk.pre != "none" and "n" in hk.pre and "D" in hk.stages else None)

    def prefetch(self, g) -> list:
        """Weight loads of this CTA's first unit of E1 stage ``g`` (its first ``depth`` chunks; X loads gated off)."""
        b = self.b
        u = E1M._Unit(g, _c(b, 0))
        no = b.cmp_ne(_c(b, 0), _c(b, 0))
        out = []
        for c in range(min(g.k.depth, g.k.kc)):
            out += [v for v in g.wissue(u, c, xonly_gate=no)]
        return out

    # the edge, the late-start twin and the launch frame: WP-L2c's chain (gdn_lb._Chain), unchanged in behaviour
    edge = G._Chain.edge
    late = G._Chain.late

    @property
    def kc(self):  # gdn_lb._Chain.edge / late read ``self.kc.mutant`` / ``.acq`` / ``.delay``
        return self.hk

    def build(self) -> KernelDef:
        b, A, hk = self.b, self.A, self.hk
        from rocke.helpers.lds_arena import ONE_CTA_PER_CU_WORDS

        zero = _c(b, 0)
        self.ctl = b.smem_alloc(I32, [CTL_WORDS], name_hint="hcctl")
        self.lds0 = b.trunc(b.smem_addr_of(self.ctl), I32)
        if hk.lpad == "fit":
            padw = self.padw
            pad = b.smem_alloc(I32, [padw], name_hint="hcpad", exclusive=True)
        else:
            padw = ONE_CTA_PER_CU_WORDS
            pad = b.smem_alloc(I32, [padw], name_hint="hcpad")
        with b.scf_if(D.is_thread0(b)):
            for w in range(4):
                PS.lds_write(b, b.add(self.lds0, _c(b, 4 * w)), zero)
            PS.lds_write(b, b.add(b.trunc(b.smem_addr_of(pad), I32), _c(b, 4 * (padw - 1))), zero)
        b.s_barrier_bare(fence="compiler")
        tk_a = b.add(A["ctl"], b.const_i64(1024))
        if hk.ticket:
            row = b.mul(self.bid, _c(b, 8))
            info = b.int_to_ptr(A["info"], PtrType(I64, "global"))

            def st(col, v):
                b.store(v if v.type == I64 else b.zext(v, I64), info, b.add(row, _c(b, col)))

            PS.ticket_xcd(b, G._Tk(), tk_a, row, st, A["gridn"], A["maxit"], _c(b, 1), fatal_err=A["err"])
            b.s_barrier_bare(fence="compiler")
        with b.scf_for(zero, A["nl"], _c(b, 1), iv_name="hcl") as li:
            self.stages(li, b.add(A["l0"], li))
        b.s_waitcnt(vmcnt=0)
        b.s_barrier_bare(fence="compiler")
        gsa = A["gs"]
        endo = BAR_LINE * BAR_WORDS
        with b.scf_if(D.is_thread0(b)):
            old = D.atomic_add_scoped(b, gsa, _c(b, endo), _c(b, 1), scope="agent", returns=True)
            with b.scf_if(b.cmp_eq(old, _c(b, GRID - 1))):
                rs = b.buffer_rsrc(_ptr(b, gsa), _c(b, GS_BYTES))
                for w_ in range(BAR_WORDS):
                    b.buffer_store_cp(zero, rs, _c(b, BAR_LINE * w_), zero, aux=CPOL_SC1)
                b.buffer_store_cp(zero, rs, _c(b, endo), zero, aux=CPOL_SC1)
                if hk.ticket:
                    for w in range(9):
                        PS.st_sc1(b, tk_a, _c(b, w * PS.LINE), zero)
        b.ret()
        return b.kernel


def build_hc_kernel(hk: HcKnobs) -> KernelDef:
    if hk.lpad == "fit":
        from rocke.helpers.lds_arena import ONE_CTA_PER_CU_WORDS

        own = G._pool_bytes(_HcChain(hk, padw=1).build()) - 4
        need = max(1, -(-(ONE_CTA_PER_CU_WORDS * 4 + 1024 - own) // 4))
        kd = _HcChain(hk, padw=need).build()
    else:
        kd = _HcChain(hk).build()
    if hk.hz:
        from .lbf_hazard import pad_asm_vmem

        pad_asm_vmem(kd)
    return kd
