# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""WP6 fold: DeepSeek-V4.1 router GEMM + sqrtsoftplus top-k in ONE launch (gfx950), for wide decode rows.

ATOM's native MoE routing at wide rows is two launches: the gate's bf16 GEMM (``x [rows, 5120] @ W_gate[384, 5120]^T``,
hipBLASLt, bf16 logits) and aiter's ``topk_gating`` (``csrc/include/topk_gating_kernels.cuh``, SCORE_SQRTSOFTPLUS:
``score = sqrt(x > 20 ? x : log2(1 + exp2(x log2e)) ln2)``, selection on ``score + bias``, weights = the unbiased score
of the winners, renormalised and times ``routed_scaling_factor``). This kernel computes both:

* CTA (token tile ``t``, K split ``ks``): the 384 x 16 logits partial of its K slice (eight waves x three 16-expert
  tiles, ``v_mfma_f32_16x16x32_bf16``), f32 partials (``sc1``), an agent-scope arrival counter per token tile;
* the last arriver of a token tile sums the ``KS`` partials in split order (deterministic), rounds each logit to bf16
  (as the gate GEMM's bf16 output), scores it, stages (biased, unbiased) scores of its 16 tokens in LDS, and then each
  wave takes two tokens: six rounds of a wave arg-max (ties: the lowest expert index; aiter breaks ties by the lowest
  lane of its own layout), renormalisation, ``routed_scaling_factor``. Writes ``ids [rows, 6]`` int32 and
  ``weights [rows, 6]`` f32 (``FusedMoE.select_experts``' outputs) and resets the counter.

Tolerance class (another logit summation order than hipBLASLt's; ties near-equal scores can pick another expert).
Mutants: ``tie`` (ties to the HIGHEST index), ``bias`` (selection without the correction bias), ``drop`` (the fold
drops the last K split), ``renorm`` (no renormalisation). Params: i64 ``x, w, bias, part, cnt, ids, wts``, i32 ``rows``,
f32-bits ``scale``. Grid ``tiles x splits`` CTAs of 512 threads.
"""

from __future__ import annotations

from dataclasses import dataclass

from rocke.core.ir import BF16, CPOL_SC1, F32, I32, I64, IRBuilder, KernelDef, PtrType, Value, VectorType
from rocke.helpers import device_ops as D

EXPERTS = 384
HIDDEN = 5120
TOPK = 6
TILE = 16
WAVES = 8
THREADS = 512
OOB = 0x7FFFFFF0
LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453
MUTANTS = ("none", "tie", "bias", "drop", "renorm")
PARAMS = ("x", "w", "bias", "part", "cnt", "ids", "wts", "rows", "scale")


@dataclass(frozen=True)
class RouterKnobs:
    tiles: int = 6  # token tiles (bucket = 16 tiles rows)
    splits: int = 16  # K splits (divides 5120 / 32)
    fast: bool = True  # top-k reductions on DPP / permlane32_swap (VALU) instead of ds_swizzle / bpermute (LDS)
    vbias: bool = True  # the fold reads its 4 experts' biases with one 16-byte load
    stamp: bool = False  # bench only: thread 0 writes s_memrealtime at 6 points to the extra i64 argument ``st``
    mutant: str = "none"
    tag: str = ""

    def check(self) -> None:
        if not 1 <= self.tiles <= 24:
            raise ValueError("tiles 1..24")
        if (HIDDEN // 32) % self.splits:
            raise ValueError("splits must divide 160")
        if self.mutant not in MUTANTS:
            raise ValueError(f"mutant in {MUTANTS}")
        if self.mutant == "drop" and self.splits < 2:
            raise ValueError("drop needs splits > 1")


def kernel_name(k: RouterKnobs) -> str:
    lev = ("" if k.fast else "_slow") + ("" if k.vbias else "_sbias") + ("_stamp" if k.stamp else "")
    return f"rk41r_t{k.tiles}_s{k.splits}{lev}" + ("" if k.mutant == "none" else f"_{k.mutant}") + (f"_{k.tag}" if k.tag else "")


def grid_of(k: RouterKnobs) -> int:
    return k.tiles * k.splits


def _c(b, v):
    return b.const_i32(int(v))


def _rsrc(b, addr, nbytes):
    return b.buffer_rsrc(b.int_to_ptr(b.readfirstlane(addr), PtrType(I32, "global")), nbytes)


def _xchg(b, v, x, fast):
    """Lane l gets v of lane l ^ x (i32). fast: DPP in-row (1..8), ds_swizzle (16), permlane32_swap (32)."""
    if not fast or x in (16, 32):
        return b.warp_shuffle_xor(v, x)
    return b.warp_shuffle_xor_dpp(v, x)


# reduction stages: fast = in-row DPP (quad xor 1, 2, half-row mirror 7, row mirror 15: after the quad stages every
# quad is uniform, so the mirrors pair whole quads), then ds_swizzle 16 and ds_bpermute 32 (a permlane32_swap form
# gave wrong top-k on gfx950, job r3: not used)
STAGES_FAST = (1, 2, 7, 15, 16, 32)
STAGES_LDS = (32, 16, 8, 4, 2, 1)


def _wave_max_f32(b, v, fast=False):
    """Butterfly max over the wave (the wave.reduce intrinsics do not link with the ROCm 7.2 COMGR)."""
    for x in (STAGES_FAST if fast else STAGES_LDS):
        o = b.bitcast(_xchg(b, b.bitcast(v, I32), x, fast), F32)
        v = b.fmax(v, o)
    return v


def _wave_min_i32(b, v, fast=False):
    for x in (STAGES_FAST if fast else STAGES_LDS):
        v = b.smin(v, _xchg(b, v, x, fast))
    return v


def _wave_max_i32(b, v, fast=False):
    for x in (STAGES_FAST if fast else STAGES_LDS):
        v = b.smax(v, _xchg(b, v, x, fast))
    return v


def build_router_kernel(k: RouterKnobs) -> KernelDef:
    k.check()
    b = IRBuilder(kernel_name(k))
    b.kernel.attrs["max_workgroup_size"] = THREADS
    a = {n: b.param(n, I64) for n in PARAMS[:7]}
    if k.stamp:
        a["st"] = b.param("st", I64)
    rows = b.readfirstlane(b.param("rows", I32))
    scale = b.bitcast(b.readfirstlane(b.param("scale", I32)), F32)
    cta = b.readfirstlane(b.block_id_x())

    def stamp(pt):
        if not k.stamp:
            return
        with b.scf_if(D.is_thread0(b)):
            sp = b.int_to_ptr(b.readfirstlane(a["st"]), PtrType(I64, "global"))
            b.store(D.memrealtime(b), sp, b.add(b.mul(cta, _c(b, 8)), _c(b, pt)))

    stamp(0)
    S, T = k.splits, k.tiles
    t = b.readfirstlane(b.div(cta, _c(b, S)))
    ks = b.readfirstlane(b.sub(cta, b.mul(t, _c(b, S))))
    tid = b.thread_id_x()
    lane = b.lane_id()
    wave = D.wave_id(b)
    g = b.lshr(lane, _c(b, 4))
    l16 = b.land(lane, _c(b, 15))
    rl = b.smax(rows, _c(b, 1))
    x_rs = _rsrc(b, a["x"], b.mul(rl, _c(b, 2 * HIDDEN)))
    w_rs = _rsrc(b, a["w"], _c(b, EXPERTS * HIDDEN * 2))
    TP = T * TILE
    p_rs = _rsrc(b, a["part"], _c(b, S * TP * EXPERTS * 4))
    steps = (HIDDEN // 32) // S  # 32-K steps of this split
    k0 = b.mul(ks, _c(b, steps * 32))
    tok = b.add(b.mul(t, _c(b, TILE)), l16)
    tok_ok = b.cmp_lt(tok, rows)
    # this wave's expert tiles: wave, wave + 8, wave + 16
    etiles = [b.add(wave, _c(b, WAVES * i)) for i in range(EXPERTS // TILE // WAVES)]
    acc = [b.zero_vec_f32(4) for _ in etiles]
    # all loads first (weights then activations), then the MFMA chains
    wl, xl = [], []
    for st in range(steps):
        kk = b.add(k0, _c(b, st * 32 + 0))
        koff = b.add(kk, b.mul(g, _c(b, 8)))
        for et in etiles:
            row = b.add(b.mul(et, _c(b, TILE)), l16)
            wl.append(b.buffer_load_cp(w_rs, b.mul(b.add(b.mul(row, _c(b, HIDDEN)), koff), _c(b, 2)), _c(b, 0), dwords=4))
        xo = b.select(tok_ok, b.mul(b.add(b.mul(tok, _c(b, HIDDEN)), koff), _c(b, 2)), _c(b, OOB))
        xl.append(b.buffer_load_cp(x_rs, xo, _c(b, 0), dwords=4))
    ne = len(etiles)
    for st in range(steps):
        bv = b.vec_bitcast(xl[st], VectorType(BF16, 8))
        for i in range(ne):
            av = b.vec_bitcast(wl[st * ne + i], VectorType(BF16, 8))
            acc[i] = b.mfma_f32_16x16x32_bf16(av, bv, acc[i])
    # partials [ks][TP][384]: lane holds experts 16 et + 4 g .. + 3 of token l16
    for i, et in enumerate(etiles):
        e0 = b.add(b.mul(et, _c(b, TILE)), b.mul(g, _c(b, 4)))
        off = b.mul(b.add(b.mul(b.add(b.mul(ks, _c(b, TP)), tok), _c(b, EXPERTS)), e0), _c(b, 4))
        b.buffer_store_cp(b.vec_bitcast(acc[i], VectorType(I32, 4)), p_rs, b.select(tok_ok, off, _c(b, OOB)), _c(b, 0), aux=CPOL_SC1)
    stamp(1)
    b.s_waitcnt(vmcnt=0)
    b.sync_lds_only()
    stamp(2)
    flag = b.smem_alloc(F32, [1], name_hint="rflag")
    cbase = b.readfirstlane(a["cnt"])
    with b.scf_if(D.is_thread0(b)):
        old = b.inline_asm(
            "s_nop 4\nglobal_atomic_add $0, $1, $2, $3 sc0\ns_waitcnt vmcnt(0)",
            "=&v,v,v,s,~{memory}",
            [b.mul(t, _c(b, 4)), _c(b, 1), cbase],
            I32,
            result_name_hint="aold",
        )
        b.smem_store_vN(flag, [_c(b, 0)], b.bitcast(old, F32), 1)
    b.sync_lds_only()
    stamp(3)
    seen = b.readfirstlane(b.bitcast(b.vec_extract(b.smem_load_vN(flag, _c(b, 0), dtype=F32, n=1), 0), I32))
    sc_b = b.smem_alloc(F32, [TILE, EXPERTS], name_hint="rbiased")
    sc_u = b.smem_alloc(F32, [TILE, EXPERTS], name_hint="runbiased")
    with b.scf_if(b.cmp_eq(seen, _c(b, S - 1))):
        D.acquire_l2(b, "agent")
        bias_ptr = b.int_to_ptr(b.readfirstlane(a["bias"]), PtrType(F32, "global"))
        pieces = TILE * EXPERTS // 4  # (token, 4 experts)
        S_ = S - (1 if k.mutant == "drop" else 0)
        for it in range(-(-pieces // THREADS)):
            v = b.add(tid, _c(b, it * THREADS))
            vt = b.div(v, _c(b, EXPERTS // 4))
            e0 = b.mul(b.sub(v, b.mul(vt, _c(b, EXPERTS // 4))), _c(b, 4))
            tk = b.add(b.mul(t, _c(b, TILE)), vt)
            ok = b.cmp_lt(v, _c(b, pieces))
            tot = None
            loads = []
            for s_ in range(S_):
                off = b.mul(b.add(b.mul(b.add(_c(b, s_ * TP), tk), _c(b, EXPERTS)), e0), _c(b, 4))
                loads.append(b.buffer_load_cp(p_rs, b.select(ok, off, _c(b, OOB)), _c(b, 0), dwords=4, aux=CPOL_SC1))
            for ld in loads:
                f = b.vec_bitcast(ld, VectorType(F32, 4))
                tot = [b.vec_extract(f, e) for e in range(4)] if tot is None else [b.fadd(tot[e], b.vec_extract(f, e)) for e in range(4)]
            if k.vbias and k.mutant != "bias":
                b_rs = _rsrc(b, a["bias"], _c(b, EXPERTS * 4))
                bvec = b.vec_bitcast(b.buffer_load_cp(b_rs, b.mul(e0, _c(b, 4)), _c(b, 0), dwords=4), VectorType(F32, 4))
            with b.scf_if(ok):
                for e in range(4):
                    lg = b.cast_to_f32(b.trunc_f32_to_bf16(tot[e]))  # the gate GEMM's bf16 logit
                    sp = b.select(b.fcmp("ogt", lg, b.const_f32(20.0)), lg,
                                  b.fmul(b.log2(b.fadd(b.const_f32(1.0), b.exp2(b.fmul(lg, b.const_f32(LOG2E))))), b.const_f32(LN2)))
                    sc = b.sqrt(sp)
                    ee = b.add(e0, _c(b, e))
                    if k.mutant == "bias":
                        bs = b.const_f32(0.0)
                    elif k.vbias:
                        bs = b.vec_extract(bvec, e)
                    else:
                        bs = b.global_load_f32(bias_ptr, ee)
                    b.smem_store_vN(sc_b, [vt, ee], b.fadd(sc, bs), 1)
                    b.smem_store_vN(sc_u, [vt, ee], sc, 1)
        b.sync_lds_only()
        stamp(4)
        # top-6 per token: wave w takes tokens 2 w, 2 w + 1; lane l holds experts l + 64 j (j < 6)
        ids_ptr = b.int_to_ptr(b.readfirstlane(a["ids"]), PtrType(I32, "global"))
        wts_ptr = b.int_to_ptr(b.readfirstlane(a["wts"]), PtrType(F32, "global"))
        ninf = b.const_f32(float("-inf"))
        for r in range(TILE // WAVES):
            vt = b.add(b.mul(wave, _c(b, TILE // WAVES)), _c(b, r))
            tk = b.add(b.mul(t, _c(b, TILE)), vt)
            vals = [b.vec_extract(b.smem_load_vN(sc_b, vt, b.add(lane, _c(b, 64 * j)), dtype=F32, n=1), 0) for j in range(6)]
            origs = [b.vec_extract(b.smem_load_vN(sc_u, vt, b.add(lane, _c(b, 64 * j)), dtype=F32, n=1), 0) for j in range(6)]
            sel_ids, sel_w = [], []
            for _round in range(TOPK):
                # this lane's best (lowest j on ties = lowest expert index; mutant tie: highest)
                best_v, best_j = vals[0], _c(b, 0)
                for j in range(1, 6):
                    better = b.fcmp("oge", vals[j], best_v) if k.mutant == "tie" else b.fcmp("ogt", vals[j], best_v)
                    best_v = b.select(better, vals[j], best_v)
                    best_j = b.select(better, _c(b, j), best_j)
                best_e = b.add(lane, b.mul(best_j, _c(b, 64)))
                m = _wave_max_f32(b, best_v, k.fast)
                cand = b.select(b.fcmp("oeq", best_v, m), best_e, _c(b, 1 << 20 if k.mutant != "tie" else -1))
                if k.mutant == "tie":
                    win = _wave_max_i32(b, cand, k.fast)
                else:
                    win = _wave_min_i32(b, cand, k.fast)
                win = b.readfirstlane(win)
                wl_lane = b.land(win, _c(b, 63))
                wj = b.lshr(win, _c(b, 6))
                # the winner's unbiased score
                mine = b.cmp_eq(lane, wl_lane)
                ov = origs[0]
                for j in range(1, 6):
                    ov = b.select(b.cmp_eq(wj, _c(b, j)), origs[j], ov)
                wscore = b.readlane(ov, wl_lane)
                sel_ids.append(win)
                sel_w.append(wscore)
                for j in range(6):
                    vals[j] = b.select(b.land(mine, b.cmp_eq(wj, _c(b, j))), ninf, vals[j])
            tot = sel_w[0]
            for wv in sel_w[1:]:
                tot = b.fadd(tot, wv)
            tot = b.fmax(tot, b.const_f32(1e-20))
            with b.scf_if(b.land(b.cmp_lt(lane, _c(b, TOPK)), b.cmp_lt(tk, rows))):
                wid = sel_ids[0]
                wsc = sel_w[0]
                for kk in range(1, TOPK):
                    pick = b.cmp_eq(lane, _c(b, kk))
                    wid = b.select(pick, sel_ids[kk], wid)
                    wsc = b.select(pick, sel_w[kk], wsc)
                out_w = wsc if k.mutant == "renorm" else b.fdiv(wsc, tot)
                out_w = b.fmul(out_w, scale)
                idx = b.add(b.mul(tk, _c(b, TOPK)), lane)
                b.store(wid, ids_ptr, idx)
                b.store(out_w, wts_ptr, idx)
        with b.scf_if(D.is_thread0(b)):
            b.store(_c(b, 0), b.int_to_ptr(cbase, PtrType(I32, "global")), t)
        b.sync_lds_only()
        stamp(5)
    return b.kernel
