# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Rich GEMM epilogues fused into one rocKE kernel (gfx950).

The problem is ``D[M, N] = epi(A[M, K] B[N, K]^T)``: an RCR GEMM, or a tensor
contraction with K-contiguous operands. The GEMM body is a persistent
row-interleave body (:data:`BODIES`: DTL ring, AGPR accumulators,
``lds_row_interleave``, chiplet raster, 256 persistent workgroups). Its
``rilv_vec_epilogue`` hands this hook the f32 accumulators of an 8 x 8
block per lane and per ``v`` (rows ``row0 + v*8 + mi``, columns ``col0 + ni``),
and the hook computes, in f32 with ONE rounding at the output::

    x  = alpha * acc                       (runtime alpha)
    x += bias[n] | bias[m]                  (bias="col" | "row")
    x += beta * C[m, n]                     (beta="same": row-major C, ldc;
                                             beta="t": column-major C, C[m + n*ldc])
    x  = act(x)                             (relu | gelu (tanh form) | silu)
    x *= G[m, n]                            (gate, multi-D)
    x += R[m, n]                            (resid, multi-D)
    swiglu: GEMM columns interleaved in groups of 4 (4 gate | 4 up):
        out[m, 4j + i] = silu(x[8j + i]) * x[8j + 4 + i]
    D[m, n] = x  (storage dtype) | Q[m, n] = fp8e4m3(x * scale) | nothing

plus reductions of x (before rounding), all fence-free device atomics:

    amax="all"  : Amax[0] = max |x|      (u32 atomic max on the f32 bits)
    amax="row"  : Amax[m] = max_n |x|
    rowsum      : RowSum[m] += sum_n x   (f32 atomic add)
    rowsq       : RowSq[m]  += sum_n x^2
    colsum      : ColSum[n] += sum_m x

Row reductions use a reduce-SCATTER butterfly across the 16 lanes that share
a row (30 swizzles per lane per stat instead of 128 for a per-row
all-reduce): after it lane ``cn`` owns rows ``row0 + 2cn + {0, 1}``.

Every lever is a knob (:class:`EpiKnobs`). The hook rides the ``spec._mono``
side channel of ``gemm_universal`` (absent -> byte-identical default kernels;
Python engine only, no C++ mirror). M and N must be multiples of 256 and K of
64 (no masks are emitted).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

from rocke.core.ir import BF16, F16, F32, FP8E4M3, I32, IRBuilder, PtrType, Value

LOG2E = 1.4426950408889634


@dataclass(frozen=True)
class EpiKnobs:
    beta: str = ""  # "" | "same" | "t"
    bias: str = ""  # "" | "col" | "row"
    act: str = ""  # "" | "relu" | "gelu" | "silu"
    gate: bool = False
    resid: bool = False
    swiglu: bool = False
    out: str = (
        "d"  # "d" | "fp8" (per-tensor scale) | "fp8b" (per 1 x qblock block: dynamic scale) | "none"
    )
    qblock: int = 128  # fp8b block along N (32 = MX block, 128 = 1x128 block)
    amax: str = ""  # "" | "all" | "row"
    rowsum: bool = False
    rowsq: bool = False
    colsum: bool = False
    store_nt: bool = True  # streaming hint on 16-byte D stores
    red_scatter: bool = (
        True  # reduce-scatter row butterfly (else all-reduce + lane cn==0)
    )
    ct_vec: bool = (
        True  # beta="t": 8-row vector loads + register transpose (else scalar loads)
    )
    fast_math: bool = (
        True  # v_exp_f32 / v_rcp_f32 for gelu / silu (else exp2 / IEEE rcp)
    )
    aux_nt: bool = False  # streaming hint on the C / G / R loads (read once)
    aux_pipe: int = (
        0  # aux (C / C^T / G / R / row-bias) loads issued this many 8-row blocks ahead (0 = inline)
    )
    zero_next: bool = (
        False  # reduction outputs double-buffered: the kernel zeroes the other slot (no memset launch)
    )

    def __post_init__(self) -> None:
        # reject, never coerce: every knob changes the kernel (params, math or
        # emission), so a wrong type or an unknown value is an error, not a
        # silent default ("false" is truthy, a negative pipe depth is not 0)
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if f.name in _EPI_CHOICES:
                if type(v) is not str or v not in _EPI_CHOICES[f.name]:
                    raise ValueError(
                        f"EpiKnobs.{f.name} must be one of "
                        f"{_EPI_CHOICES[f.name]}, got {v!r}"
                    )
            elif f.name in ("qblock", "aux_pipe"):
                if type(v) is not int:
                    raise ValueError(f"EpiKnobs.{f.name} must be an int, got {v!r}")
            elif type(v) is not bool:
                raise ValueError(f"EpiKnobs.{f.name} must be a bool, got {v!r}")
        if self.aux_pipe < 0:
            raise ValueError(f"EpiKnobs.aux_pipe must be >= 0, got {self.aux_pipe}")
        if self.qblock not in (32, 64, 128):
            raise ValueError(
                "EpiKnobs.qblock must be 32, 64 or 128 (8 columns per lane, 16 "
                f"lanes per row group), got {self.qblock}"
            )

    def tag(self) -> str:
        d = dataclasses.asdict(self)
        base = EpiKnobs()
        parts = [f"{k}={v}" for k, v in d.items() if getattr(base, k) != v]
        return ",".join(parts) or "plain"


_EPI_CHOICES = {
    "beta": ("", "same", "t"),
    "bias": ("", "col", "row"),
    "act": ("", "relu", "gelu", "silu"),
    "out": ("d", "fp8", "fp8b", "none"),
    "amax": ("", "all", "row"),
}


def _ty(dtype: str):
    return {"f16": F16, "fp16": F16, "bf16": BF16}[dtype]


class EpiHooks:
    """``spec._mono`` side channel: rich epilogue, no producer tasks."""

    writes_operands = False

    def __init__(self, knobs: EpiKnobs, dtype: str = "f16"):
        self.k = knobs
        self.dt = _ty(dtype)
        if knobs.swiglu and (
            knobs.act or knobs.gate or knobs.colsum or knobs.beta or knobs.bias == "col"
        ):
            raise ValueError(
                "swiglu: no act / gate / colsum / beta / col bias (out columns differ)"
            )
        if knobs.out == "fp8b" and knobs.qblock not in (32, 64, 128):
            raise ValueError(
                "qblock must be 32, 64 or 128 (8 columns per lane, 16 lanes per row group)"
            )
        if knobs.out in ("fp8", "fp8b") and knobs.swiglu:
            raise ValueError("fp8 output with swiglu not built")

    # ------------------------------------------------------------ params
    def add_params(self, b: IRBuilder, *, A, B, C, M, N, K) -> None:
        dt, k = self.dt, self.k
        self.D = C
        p: Dict[str, Value] = {}
        p["alpha"] = b.param("alpha", F32)
        p["ldd"] = b.param("ldd", I32)
        ro = dict(readonly=True, noalias=True)
        if k.beta:
            p["Cin"] = b.param("Cin", PtrType(dt, "global"), align=16, **ro)
            p["beta"] = b.param("beta", F32)
            p["ldc"] = b.param("ldc", I32)
        if k.bias:
            p["Bias"] = b.param("Bias", PtrType(dt, "global"), align=16, **ro)
        if k.gate:
            p["G"] = b.param("G", PtrType(dt, "global"), align=16, **ro)
        if k.resid:
            p["R"] = b.param("R", PtrType(dt, "global"), align=16, **ro)
        if k.out == "fp8b":
            p["Q"] = b.param("Q", PtrType(FP8E4M3, "global"), align=8, noalias=True)
            p["S"] = b.param("S", PtrType(F32, "global"), align=4, noalias=True)
        if k.out == "fp8":
            p["Q"] = b.param("Q", PtrType(FP8E4M3, "global"), align=8, noalias=True)
            p["scale"] = b.param("scale", F32)
        if k.amax:
            p["Amax"] = b.param("Amax", PtrType(I32, "global"), align=4)
        if k.rowsum:
            p["RowSum"] = b.param("RowSum", PtrType(F32, "global"), align=4)
        if k.rowsq:
            p["RowSq"] = b.param("RowSq", PtrType(F32, "global"), align=4)
        if k.colsum:
            p["ColSum"] = b.param("ColSum", PtrType(F32, "global"), align=4)
        if k.zero_next and self._has_red():
            # slot s in {0, 1}: reductions go to buffer[s * len + i]; this launch zeroes buffer[(1 - s) * len + i]
            p["slot"] = b.param("slot", I32)
        self.M_, self.N_ = M, N
        self.p = p

    def _has_red(self) -> bool:
        k = self.k
        return bool(k.amax or k.rowsum or k.rowsq or k.colsum)

    def _red_len(self, b, nm):
        """runtime length of one slot of a reduction buffer"""
        k = self.k
        if nm == "Amax":
            return b.const_i32(1) if k.amax == "all" else self.M_
        if nm == "ColSum":
            return b.div(self.N_, b.const_i32(2)) if k.swiglu else self.N_
        return self.M_

    def _slot_base(self, b, nm):
        if not (self.k.zero_next and self._has_red()):
            return None
        return b.mul(self.p["slot"], self._red_len(b, nm))

    def pre_tasks(self, b, *, wg, nwg, tid) -> None:
        if not (self.k.zero_next and self._has_red()):
            return
        # zero the OTHER slot of every reduction buffer (consumed by the next launch, stream-ordered)
        other = b.sub(b.const_i32(1), self.p["slot"])
        g = b.add(b.mul(wg, b.const_i32(256)), tid)
        stride = b.mul(nwg, b.const_i32(256))
        for nm in ("Amax", "RowSum", "RowSq", "ColSum"):
            if nm not in self.p:
                continue
            ln = self._red_len(b, nm)
            base = b.mul(other, ln)
            with b.scf_for(g, ln, stride, iv_name=f"z{nm}") as i:
                if nm == "Amax":
                    b.global_store(self.p[nm], b.add(base, i), b.const_i32(0), align=4)
                else:
                    b.global_store(
                        self.p[nm], b.add(base, i), b.const_f32(0.0), align=4
                    )

    def wait_tile(self, b, bm, bn, valid, *, tid) -> None:
        pass

    def post_tile(self, b, bm, bn, *, tid) -> None:
        pass

    def post_tasks(self, b, *, wg, nwg, tid) -> None:
        pass

    # ------------------------------------------------------------ math helpers
    def _load8(self, b, ptr, off, *, nt=False) -> List[Value]:
        return self._cvt8(b, self._raw8(b, ptr, off, nt=nt))

    def _raw8(self, b, ptr, off, *, nt=False):
        from rocke.core.ir import TemporalHint as _TH

        return b.global_load_vN(
            ptr,
            off,
            self.dt,
            8,
            align=16,
            temporal_hint=_TH.STREAMING if nt else _TH.DEFAULT,
        )

    def _cvt8(self, b, v) -> List[Value]:
        if isinstance(v, list):
            return v
        return [b.cast_to_f32(b.vec_extract(v, e)) for e in range(8)]

    def _aux_loads(self, b, v, row0, col0, mfmas_m):
        """raw (un-converted) aux loads of the 8-row block v"""
        k, p, dt = self.k, self.p, self.dt
        rbase = b.add(row0, b.const_i32(v * mfmas_m))
        out = {"rbase": rbase}
        if k.beta == "t":
            if k.ct_vec:
                out["ct"] = [
                    self._raw8(
                        b,
                        p["Cin"],
                        b.add(b.mul(b.add(col0, b.const_i32(ni)), p["ldc"]), rbase),
                        nt=k.aux_nt,
                    )
                    for ni in range(8)
                ]
            else:
                out["ct"] = [
                    [
                        b.cast_to_f32(
                            b.global_load(
                                p["Cin"],
                                b.add(
                                    b.mul(b.add(col0, b.const_i32(ni)), p["ldc"]),
                                    b.add(rbase, b.const_i32(mi)),
                                ),
                                dt,
                                align=2,
                            )
                        )
                        for mi in range(8)
                    ]
                    for ni in range(8)
                ]
        if k.bias == "row":
            out["bias_r"] = self._raw8(b, p["Bias"], rbase)
        for nm, on, ptr in (
            ("c", k.beta == "same", "Cin"),
            ("g", k.gate, "G"),
            ("r", k.resid, "R"),
        ):
            if not on:
                continue
            rows = []
            for mi in range(mfmas_m):
                row = b.add(rbase, b.const_i32(mi))
                ld = p["ldc"] if nm == "c" else p["ldd"]
                rows.append(
                    self._raw8(b, p[ptr], b.add(b.mul(row, ld), col0), nt=k.aux_nt)
                )
            out[nm] = rows
        return out

    def _sigmoid_mul(self, b, x, z):
        """x * sigmoid(z) = x / (1 + 2^(-z log2 e))."""
        t = b.fmul(z, b.const_f32(-LOG2E))
        if self.k.fast_math:
            e = b.exp2_fast(t)
            return b.fmul(x, b.rcp_fast(b.fadd(b.const_f32(1.0), e)))
        e = b.exp2(t)
        return b.fmul(x, b.rcp(b.fadd(b.const_f32(1.0), e)))

    def _act(self, b, x):
        a = self.k.act
        if a == "relu":
            return b.fmax(x, b.const_f32(0.0))
        if a == "silu":
            return self._sigmoid_mul(b, x, x)
        if a == "gelu":
            # tanh form: 0.5 x (1 + tanh(s (x + c x^3)))
            #   = x * sigmoid(2 s (x + c x^3))
            x2 = b.fmul(x, x)
            u = b.fmul(x, b.fma(x2, b.const_f32(0.044715), b.const_f32(1.0)))
            return self._sigmoid_mul(
                b, x, b.fmul(u, b.const_f32(2.0 * 0.7978845608028654))
            )
        assert not a, a
        return x

    def _shfl(self, b, x, h):
        return b.warp_shuffle_xor(x, h)

    def _rs_rows(self, b, vals, op, cn):
        """Reduce-scatter over the 16 lanes of a row group: 32 values -> 2
        (rows 2cn, 2cn + 1 of the list)."""
        L = list(vals)
        for h in (8, 4, 2, 1):
            half = len(L) // 2
            lo, hi = L[:half], L[half:]
            bit = b.cmp_ne(
                b.mod(b.div(cn, b.const_i32(h)), b.const_i32(2)), b.const_i32(0)
            )
            keep = [b.select(bit, hi[i], lo[i]) for i in range(half)]
            send = [b.select(bit, lo[i], hi[i]) for i in range(half)]
            L = [op(keep[i], self._shfl(b, send[i], h)) for i in range(half)]
        return L

    def _atomic_umax(self, b, ptr, idx, val_f32):
        off = b.mul(idx, b.const_i32(4))
        # the leading s_nop 4 covers the gfx9 hazard "VALU writes an SGPR ->
        # VMEM reads it" (5 wait states), which LLVM pads for its own VMEM
        # instructions but not inside inline asm: the SGPR base may be
        # restored from a spill by v_readlane right before this asm
        b.inline_asm(
            "s_nop 4\n\tglobal_atomic_umax $0, $1, $2 sc1",
            "v,v,s",
            [off, b.bitcast(val_f32, I32), ptr],
        )

    # ------------------------------------------------------------ epilogue
    def epilogue(
        self,
        b: IRBuilder,
        *,
        accf,
        row0,
        col0,
        mfmas_m,
        mfmas_n,
        c_per_lane,
        C,
        M,
        N,
        block_m_off,
        block_n_off,
        warp_n_idx,
        lane,
    ) -> None:
        from rocke.core.ir import TemporalHint as _TH

        k, dt, p = self.k, self.dt, self.p
        assert mfmas_n == 8 and mfmas_m == 8, (mfmas_m, mfmas_n)
        alpha, ldd = p["alpha"], p["ldd"]
        c16 = b.const_i32(16)
        cn = b.mod(lane, c16)
        qg = b.div(lane, c16)
        fadd = b.fadd
        fmax = b.fmax
        bias_c = self._load8(b, p["Bias"], col0) if k.bias == "col" else None
        dcol = b.div(col0, b.const_i32(2)) if k.swiglu else col0
        nt = _TH.STREAMING if k.store_nt else _TH.DEFAULT
        row_stats = {
            nm: []
            for nm, on in (
                ("amax", k.amax == "row"),
                ("rowsum", k.rowsum),
                ("rowsq", k.rowsq),
            )
            if on
        }
        col_acc = [None] * 8 if k.colsum else None
        gmax = None
        # aux loads: issued aux_pipe blocks ahead of their use (0 = right before)
        pend = {}
        ahead = k.aux_pipe  # validated >= 0 by EpiKnobs
        for v0 in range(min(ahead, c_per_lane)):
            pend[v0] = self._aux_loads(b, v0, row0, col0, mfmas_m)
        for v in range(c_per_lane):
            if v + ahead < c_per_lane and (v + ahead) not in pend:
                pend[v + ahead] = self._aux_loads(b, v + ahead, row0, col0, mfmas_m)
            ax_ = pend.pop(v)
            rbase = ax_["rbase"]
            ct = [self._cvt8(b, c_) for c_ in ax_["ct"]] if "ct" in ax_ else None
            bias_r = self._cvt8(b, ax_["bias_r"]) if "bias_r" in ax_ else None
            for mi in range(mfmas_m):
                row = b.add(rbase, b.const_i32(mi))
                x = [
                    b.fmul(alpha, b.vec_extract(accf[mi * mfmas_n + ni], v))
                    for ni in range(8)
                ]
                if bias_c is not None:
                    x = [fadd(x[i], bias_c[i]) for i in range(8)]
                if bias_r is not None:
                    x = [fadd(x[i], bias_r[mi]) for i in range(8)]
                roff = b.mul(row, ldd)
                if k.beta == "same":
                    cv = self._cvt8(b, ax_["c"][mi])
                    x = [b.fma(p["beta"], cv[i], x[i]) for i in range(8)]
                elif k.beta == "t":
                    x = [b.fma(p["beta"], ct[i][mi], x[i]) for i in range(8)]
                if k.act:
                    x = [self._act(b, xi) for xi in x]
                if k.gate:
                    g = self._cvt8(b, ax_["g"][mi])
                    x = [b.fmul(x[i], g[i]) for i in range(8)]
                if k.resid:
                    r = self._cvt8(b, ax_["r"][mi])
                    x = [fadd(x[i], r[i]) for i in range(8)]
                if k.swiglu:
                    x = [
                        b.fmul(self._sigmoid_mul(b, x[i], x[i]), x[4 + i])
                        for i in range(4)
                    ]
                # ---- stores
                if k.out == "d":
                    nv = len(x)
                    vec = b.vec_pack([b.cast_f32_to(xi, dt) for xi in x], dt)
                    th = nt if nv == 8 else _TH.DEFAULT
                    b.global_store_vN(
                        C, b.add(roff, dcol), vec, nv, align=2 * nv, temporal_hint=th
                    )
                elif k.out == "fp8":
                    sc = p["scale"]
                    q0 = b.cvt_pk_fp8_f32x4(
                        b.vec_pack([b.fmul(x[i], sc) for i in range(4)], F32)
                    )
                    q1 = b.cvt_pk_fp8_f32x4(
                        b.vec_pack([b.fmul(x[i], sc) for i in range(4, 8)], F32)
                    )
                    q = b.vec_pack(
                        [b.vec_extract(q0, i) for i in range(4)]
                        + [b.vec_extract(q1, i) for i in range(4)],
                        FP8E4M3,
                    )
                    b.global_store_vN(p["Q"], b.add(roff, col0), q, 8, align=8)
                elif k.out == "fp8b":
                    # dynamic block scale: s = amax(block) / 448, q = fp8(x / s); one pass, no delayed scaling
                    m8 = b.fabs(x[0])
                    for xi in x[1:]:
                        m8 = fmax(m8, b.fabs(xi))
                    for h in (1, 2, 4, 8)[: {32: 2, 64: 3, 128: 4}[k.qblock]]:
                        m8 = fmax(m8, self._shfl(b, m8, h))
                    m8 = fmax(m8, b.const_f32(1e-30))
                    inv = b.fmul(b.const_f32(448.0), b.rcp(m8))
                    q0 = b.cvt_pk_fp8_f32x4(
                        b.vec_pack([b.fmul(x[i], inv) for i in range(4)], F32)
                    )
                    q1 = b.cvt_pk_fp8_f32x4(
                        b.vec_pack([b.fmul(x[i], inv) for i in range(4, 8)], F32)
                    )
                    q = b.vec_pack(
                        [b.vec_extract(q0, i) for i in range(4)]
                        + [b.vec_extract(q1, i) for i in range(4)],
                        FP8E4M3,
                    )
                    b.global_store_vN(p["Q"], b.add(roff, col0), q, 8, align=8)
                    lpb = k.qblock // 8  # lanes per block
                    with b.scf_if(
                        b.cmp_eq(b.mod(cn, b.const_i32(lpb)), b.const_i32(0))
                    ):
                        nb = b.div(ldd, b.const_i32(k.qblock))
                        b.global_store(
                            p["S"],
                            b.add(b.mul(row, nb), b.div(col0, b.const_i32(k.qblock))),
                            b.fmul(m8, b.const_f32(1.0 / 448.0)),
                            align=4,
                        )
                # ---- reductions (on x before rounding)
                if k.amax:
                    ax = [b.fabs(xi) for xi in x]
                    m8 = ax[0]
                    for xi in ax[1:]:
                        m8 = fmax(m8, xi)
                    if k.amax == "all":
                        gmax = m8 if gmax is None else fmax(gmax, m8)
                    else:
                        row_stats["amax"].append(m8)
                if k.rowsum:
                    s = x[0]
                    for xi in x[1:]:
                        s = fadd(s, xi)
                    row_stats["rowsum"].append(s)
                if k.rowsq:
                    s = b.fmul(x[0], x[0])
                    for xi in x[1:]:
                        s = b.fma(xi, xi, s)
                    row_stats["rowsq"].append(s)
                if col_acc is not None:
                    col_acc = [
                        x[i] if col_acc[i] is None else fadd(col_acc[i], x[i])
                        for i in range(8)
                    ]
        # ---- row statistics: one atomic per (row, stat) per 16-lane group
        sb = {
            nm: self._slot_base(b, nm)
            for nm in ("Amax", "RowSum", "RowSq", "ColSum")
            if nm in p
        }

        def _ix(nm, i):
            return i if sb.get(nm) is None else b.add(sb[nm], i)

        for nm, vals in row_stats.items():
            op = fmax if nm == "amax" else fadd
            ptr = {
                "amax": p.get("Amax"),
                "rowsum": p.get("RowSum"),
                "rowsq": p.get("RowSq"),
            }[nm]
            pn = {"amax": "Amax", "rowsum": "RowSum", "rowsq": "RowSq"}[nm]
            if k.red_scatter:
                two = self._rs_rows(b, vals, op, cn)
                for j in range(2):
                    r = _ix(
                        pn,
                        b.add(row0, b.add(b.mul(cn, b.const_i32(2)), b.const_i32(j))),
                    )
                    if nm == "amax":
                        self._atomic_umax(b, ptr, r, two[j])
                    else:
                        b.global_atomic_add(ptr, r, two[j])
            else:
                red = []
                for x in vals:
                    for h in (1, 2, 4, 8):
                        x = op(x, self._shfl(b, x, h))
                    red.append(x)
                with b.scf_if(b.cmp_eq(cn, b.const_i32(0))):
                    for j, x in enumerate(red):
                        r = _ix(pn, b.add(row0, b.const_i32(j)))
                        if nm == "amax":
                            self._atomic_umax(b, ptr, r, x)
                        else:
                            b.global_atomic_add(ptr, r, x)
        # ---- column sums: reduce-scatter over the 4 row groups (xor 16, 32)
        if col_acc is not None:
            L = col_acc
            idx_parts = []
            for h, step in ((16, 4), (32, 2)):
                half = len(L) // 2
                lo, hi = L[:half], L[half:]
                bit = b.cmp_ne(
                    b.mod(b.div(lane, b.const_i32(h)), b.const_i32(2)), b.const_i32(0)
                )
                keep = [b.select(bit, hi[i], lo[i]) for i in range(half)]
                send = [b.select(bit, lo[i], hi[i]) for i in range(half)]
                L = [fadd(keep[i], self._shfl(b, send[i], h)) for i in range(half)]
                idx_parts.append(
                    b.mul(
                        b.mod(b.div(lane, b.const_i32(h)), b.const_i32(2)),
                        b.const_i32(step),
                    )
                )
            base = b.add(col0, b.add(idx_parts[0], idx_parts[1]))
            for j in range(2):
                b.global_atomic_add(
                    p["ColSum"], _ix("ColSum", b.add(base, b.const_i32(j))), L[j]
                )
        if gmax is not None:
            for h in (1, 2, 4, 8, 16, 32):
                gmax = fmax(gmax, self._shfl(b, gmax, h))
            with b.scf_if(b.cmp_eq(lane, b.const_i32(0))):
                self._atomic_umax(b, p["Amax"], _ix("Amax", b.const_i32(0)), gmax)
        del qg


# ---------------------------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------------------------

# GEMM bodies: "HBR" = the hb_sched row-interleave schedule with the wgm 16 / chunk 8 raster (the default),
# "RI" = the merged-interleave schedule with the wgm 8 / chunk 32 raster (a retained alternative, slower on
# gfx950). Both are persistent over 256 workgroups with the non-temporal vector epilogue.
BODIES = {
    "HBR": dict(
        hb_sched=True,
        hb_read_every=2,
        ilv_merge=False,
        chiplet_chunk_size=8,
        chiplet_wgm=16,
    ),
    "RI": dict(),
}


def body_spec(dtype: str = "f16", body: str = "HBR", **trait_over):
    """A persistent row-interleave body (:data:`BODIES`, default "HBR") + trait overrides."""
    from rocke.instances.common.gemm_universal import (
        DataSpec,
        TileSpec,
        TraitSpec,
        UniversalGemmSpec,
    )

    tile = dict(
        tile_m=256,
        tile_n=256,
        tile_k=64,
        warp_m=2,
        warp_n=2,
        warp_k=1,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=32,
    )
    trait = dict(
        pipeline="compv4",
        scheduler="intrawave",
        epilogue="default",
        chiplet_swizzle=True,
        chiplet_wgm=8,
        chiplet_num_xcds=8,
        chiplet_chunk_size=32,
        direct_to_lds=True,
        dtl_prefetch=True,
        lds_swizzle=False,
        acc_agpr=True,
        dtl_frag_prefetch=True,
        mfma_interleave=True,
        ilv_merge=True,
        dtl_hoist=True,
        dtl_stages=2,
        lds_row_interleave=True,
        rilv_vec_epilogue=True,
        rve_store_nt=True,
        persistent_ctas=256,
    )
    trait.update(BODIES[body])
    trait.update(trait_over)
    dt = "fp16" if dtype in ("f16", "fp16") else dtype
    return UniversalGemmSpec(
        name="gemm_rich_epi",
        tile=TileSpec(**tile),
        trait=TraitSpec(**trait),
        data=DataSpec(dtype_a=dt, dtype_b=dt, dtype_c=dt),
    )


@dataclass(frozen=True)
class RichEpilogueSpec:
    """One fused GEMM + rich epilogue kernel: the epilogue ``knobs``, the
    storage dtype, the GEMM body (:data:`BODIES`) and extra body trait
    overrides as ``(field, value)`` pairs."""

    knobs: EpiKnobs = EpiKnobs()
    dtype: str = "f16"
    body: str = "HBR"
    trait_over: Tuple[Tuple[str, object], ...] = ()

    def kernel_name(self) -> str:
        over = dict(self.trait_over)
        if self.body != "HBR":
            over["body"] = self.body
        key = json.dumps(
            [dataclasses.asdict(self.knobs), self.dtype, over], sort_keys=True
        )
        return "gemm_rich_epi_" + hashlib.sha1(key.encode()).hexdigest()[:10]


RICH_EPILOGUE_ARCHS = ("gfx950",)


def rich_epilogue_gemm_spec(spec: RichEpilogueSpec):
    """The universal GEMM spec of ``spec`` with its :class:`EpiHooks`
    attached (``gemm_spec._mono``)."""
    gs = body_spec(spec.dtype, body=spec.body, **dict(spec.trait_over))
    gs = dataclasses.replace(gs, name=spec.kernel_name())
    object.__setattr__(gs, "_mono", EpiHooks(spec.knobs, dtype=spec.dtype))
    return gs


def build_rich_epilogue(spec: RichEpilogueSpec, *, arch: str = "gfx950"):
    """KernelDef of the fused GEMM + rich epilogue.
    Params (in order): A, B, C(=D), M, N, K, then the hook params
    (alpha, ldd, [Cin, beta, ldc], [Bias], [G], [R], [Q, scale | Q, S], [Amax],
    [RowSum], [RowSq], [ColSum], [slot]). N is the GEMM column count (2x the
    output columns for swiglu). Persistent grid of 256 workgroups, block 256."""
    from rocke.instances.common.gemm_universal import build_universal_gemm

    if arch not in RICH_EPILOGUE_ARCHS:
        raise ValueError(f"rich epilogue: arch {arch!r} not in {RICH_EPILOGUE_ARCHS}")
    return build_universal_gemm(rich_epilogue_gemm_spec(spec), arch=arch)
