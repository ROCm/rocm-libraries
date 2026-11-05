# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The mHC seam slice (``stage_slice``, with K2a's ``slice_prefetch`` + ``stage_reduce``) of ATOM's DeepSeek-V4.1
mono as a rocKE stage function (W15 PORT K2-attn-gemv).

Bit-identical re-implementation of ATOM ``attn_pre.py`` (origin/main df86c482) ``stage_slice(c, task, pre)`` for the
hidden columns ``32 task .. +32`` of every token and all four HC streams, in two forms:

* ``k1`` (K1 ``k1.slice``): the mix weights (``slice_fn_loads``, 6 a thread) issued first; the owed post ``pend`` read
  from global memory; ``fold`` (the settled residual is folded with post / comb, ATOM's ``mhc_fused_post_pre`` order:
  ``((post x + c0 r0) + c1 r1) + c2 r2`` products rounded, then ``fma(c3, r3, ..)``, bf16 -> ``res_out``) or not (the
  residual read as is); ``aux`` (the DSpark tap: mean over the bf16 streams -> ``aux``);
* ``k2`` (K2a ``k2a.slice``): ``slice_prefetch`` (the first batch of residual / gate loads and, up to 24 rows, the
  weights) before ``stage_reduce`` (the attention output at the 32 columns: both TP ranks' bf16 pair partials polled from
  this rank's ``attn`` peer region at system scope, summed in rank order in f32, bf16 -> the ``pl`` LDS staging), then the
  slice with ``pend`` read from ``pl``.

Then (both): the weights as bf16 hi + bf16(fn - hi) into ``fl``; barrier; the layer input (``slice_input``: bf16 R_new
of the 4 streams times the incoming pre-mix, products rounded, summed in stream order; two columns a thread, one bf16
pair a ``lin`` put); the sum of squares (token t on wave t % 8, a 64-lane butterfly 32, 16, 8, 4, 2, 1 -> ``pmix``);
the 24 mix-projection partials with ``v_mfma_f32_16x16x4_f32`` (A = fn rows, B = R_new^T, a 16-token tile at a time),
summed over the 8 waves in order -> ``pmix``.

ABI (``lds="arg"``; ``rl`` / ``fl`` / ``red`` / ``pl`` LDS word offsets from ``lds_base``)::

    k1: rocke_dsv41_slice_k1f<F>a<A>n<S>  void (i32 tag, i32 task, i64 res_in, i64 pend, i64 post_in, i64 comb_in,
        i64 pre_in, i64 hc_fn, i64 res_out, i64 aux, i64 lin, i64 pmix, i32 rl, i32 fl, i32 red, i32 lds_base)
    k2: rocke_dsv41_slice_k2n<S>  void (i32 tag, i32 task, i64 res_in, i64 post_in, i64 comb_in, i64 pre_in, i64 hc_fn,
        i64 res_out, i64 lin, i64 pmix, i64 attn, i32 rl, i32 fl, i32 red, i32 pl, i32 lds_base)

Knobs (:class:`SliceKnobs`): :class:`seam_common.SeamKnobs` plus ``batch`` (rounds a load batch: ATOM's LOAD_BATCH 4)
and ``fn_early_max`` (k2: rows up to which the weights go out before the reduce; ATOM 24). Mutants: ``put_bit`` (the lin
pair's low bit), ``res_bit`` (res_out), ``mix_drop`` (one MFMA K step left out), ``pend0`` (the owed post read as 0),
``sq_order`` (sum of squares butterfly order 1 .. 32: an f32 reassociation, report-only).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass

from rocke.core.ir import BF16, F32, I16, I32, I64, IRBuilder

from . import seam_common as S

STAGE = "slice"
MUTANTS = ("put_bit", "res_bit", "mix_drop", "pend0", "sq_order")
REPORT_ONLY = ("sq_order",)
ABI_K1 = (
    ("tag", "i32"), ("task", "i32"), ("res_in", "i64"), ("pend", "i64"), ("post_in", "i64"), ("comb_in", "i64"),
    ("pre_in", "i64"), ("hc_fn", "i64"), ("res_out", "i64"), ("aux", "i64"), ("lin", "i64"), ("pmix", "i64"),
    ("rl", "i32"), ("fl", "i32"), ("red", "i32"),
)  # fmt: skip
ABI_K2 = (
    ("tag", "i32"), ("task", "i32"), ("res_in", "i64"), ("post_in", "i64"), ("comb_in", "i64"), ("pre_in", "i64"),
    ("hc_fn", "i64"), ("res_out", "i64"), ("lin", "i64"), ("pmix", "i64"), ("attn", "i64"), ("rl", "i32"),
    ("fl", "i32"), ("red", "i32"), ("pl", "i32"),
)  # fmt: skip
TILE = 16
CPOL_SYS = 17


@dataclass(frozen=True)
class SliceKnobs(S.SeamKnobs):
    batch: int = 4
    fn_early_max: int = 24

    def __post_init__(self) -> None:
        self.check(MUTANTS)
        if not 1 <= self.batch <= 8:
            raise ValueError("batch must be in 1..8")


def parse_variant(v: str):
    """``k1f<F>a<A>n<S>`` -> ("k1", S, F, A); ``k2n<S>`` -> ("k2", S, 1, 0)."""
    if v.startswith("k2n"):
        return "k2", int(v[3:]), 1, 0
    if v.startswith("k1f"):
        f = int(v[3])
        a = int(v[5])
        return "k1", int(v[7:]), f, a
    raise ValueError(f"slice variant {v!r}")


def variant(form, rows, fold=1, aux=0):
    return f"k2n{rows}" if form == "k2" else f"k1f{fold}a{aux}n{rows}"


def build_slice_kernel(form: str, rows: int, fold: int = 1, aux: int = 0, knobs: SliceKnobs = SliceKnobs()):
    if form not in ("k1", "k2"):
        raise ValueError("form must be k1 or k2")
    if form == "k2" and (not fold or aux):
        raise ValueError("k2 slices fold and have no tap")
    abi = ABI_K1 if form == "k1" else ABI_K2
    b = IRBuilder(f"rocke_dsv41_{STAGE}_{variant(form, rows, fold, aux)}")
    b.kernel.attrs["max_workgroup_size"] = S.THREADS
    b.fence_barriers = "compiler"
    p = {n: b.param(n, I32 if ty == "i32" else I64) for n, ty in abi}
    e = S.SeamEnv(b, knobs)
    c = e.c
    tag = b.readfirstlane(p["tag"])
    task = b.readfirstlane(p["task"])
    rl, fl, red = (b.readfirstlane(p[n]) for n in ("rl", "fl", "red"))
    pl = b.readfirstlane(p["pl"]) if form == "k2" else None
    tid, lane, wave = e.tid, e.lane, e.wave
    s = rows
    c0 = b.mul(task, c(S.COLS))
    n_el = s * S.KT
    rounds = -(-n_el // S.THREADS)

    def fn_loads():
        out = []
        for i in range(S.MIX * S.KT // S.THREADS):
            ee = b.add(tid, c(S.THREADS * i))
            j = b.lshr(ee, c(7))  # // KT
            o = b.lshr(b.land(ee, c(S.KT - 1)), c(5))  # (e % KT) // COLS
            at = b.add(b.add(b.mul(j, c(S.K_MIX)), b.mul(o, c(S.HIDDEN))), b.add(c0, b.land(ee, c(S.COLS - 1))))
            out.append(e.ld_f32(p["hc_fn"], at))
        return out

    def at_of(ee):
        t = b.lshr(ee, c(7))
        o = b.lshr(b.land(ee, c(S.KT - 1)), c(5))
        cc = b.land(ee, c(S.COLS - 1))
        return t, o, cc, b.add(c0, cc)

    def row_loads(i):
        ee = b.smin(b.add(tid, c(S.THREADS * i)), c(n_el - 1))
        t, o, _cc, col = at_of(ee)
        if not fold:
            return e.ld_bf(p["res_in"], b.add(b.mul(b.add(b.mul(t, c(S.HC)), o), c(S.HIDDEN)), col))
        r = [e.ld_bf(p["res_in"], b.add(b.mul(b.add(b.mul(t, c(S.HC)), c(h)), c(S.HIDDEN)), col)) for h in range(S.HC)]
        cm = [e.ld_f32(p["comb_in"], b.add(b.mul(b.add(b.mul(t, c(S.HC)), c(h)), c(S.HC)), o)) for h in range(S.HC)]
        return r, cm, e.ld_f32(p["post_in"], b.add(b.mul(t, c(S.HC)), o))

    def pend_value(t, col, cc):
        if knobs.mutate == "pend0":
            return e.f(0.0)
        if form == "k1":
            return e.ld_bf(p["pend"], b.add(b.mul(t, c(S.HIDDEN)), col))
        return e.lds_f32(b.add(pl, b.add(b.mul(t, c(S.COLS)), cc)))

    def row_use(i, got):
        ee = b.add(tid, c(S.THREADS * i))
        t, o, cc, col = at_of(ee)
        if fold:
            r, cm, post = got
            v = b.fadd(b.fmul(post, pend_value(t, col, cc)), b.fmul(r[0], cm[0]))
            for h in range(1, S.HC - 1):
                v = b.fadd(v, b.fmul(cm[h], r[h]))
            v = e.bf16r(e.fma(cm[S.HC - 1], r[S.HC - 1], v))
            bits = b.trunc_f32_to_bf16(v)
            if knobs.mutate == "res_bit":
                bits = b.bitcast(b.trunc(b.xor(e.bf16_bits(v), c(1)), I16), BF16)
            at = b.add(b.mul(b.add(b.mul(t, c(S.HC)), o), c(S.HIDDEN)), col)
            b.buffer_store_bf16(e.rsrc(p["res_out"]), b.mul(at, c(2)), c(0), bits)
        else:
            v = got
        e.lds_st_f32(b.add(rl, ee), v)

    def batched(first):
        for b0 in range(0, rounds, knobs.batch):
            idx = list(range(b0, min(b0 + knobs.batch, rounds)))
            got = first if (b0 == 0 and first is not None) else [row_loads(i) for i in idx]
            for i, g in zip(idx, got):
                if (i + 1) * S.THREADS <= n_el:
                    row_use(i, g)
                else:
                    with b.scf_if(b.cmp_lt(b.add(tid, c(S.THREADS * i)), c(n_el))):
                        row_use(i, g)

    # ---- issue order
    if form == "k1":
        fns = fn_loads()
        first = None
    else:
        first = [row_loads(i) for i in range(min(knobs.batch, rounds))]
        fns = fn_loads() if s <= knobs.fn_early_max else None
        # stage_reduce: the attention output at this slice's columns, rank-ordered f32 sum -> bf16 -> pl
        r_at = e.rsrc(p["attn"])
        for pas in range(0, s * S.COLS // 2, S.THREADS):
            idx = b.add(c(pas), tid)

            def reduce_one(idx=idx):
                t = b.lshr(idx, c(4))
                col = b.add(c0, b.mul(b.land(idx, c(15)), c(2)))
                specs = [(r_at, None, b.add(b.mul(b.add(c(src * s), t), c(S.HIDDEN // 2)), b.lshr(col, c(1))))
                         for src in range(2)]  # fmt: skip
                ws = e.poll_aux(tag, specs, CPOL_SYS)
                acc0, acc1 = e.bf_lo(ws[0]), e.bf_hi(ws[0])
                acc0 = b.fadd(acc0, e.bf_lo(ws[1]))
                acc1 = b.fadd(acc1, e.bf_hi(ws[1]))
                lat = b.add(pl, b.add(b.mul(t, c(S.COLS)), b.sub(col, c0)))
                e.lds_st_f32(lat, e.bf16r(acc0))
                e.lds_st_f32(b.add(lat, c(1)), e.bf16r(acc1))

            if pas + S.THREADS <= s * S.COLS // 2:
                reduce_one()
            else:
                with b.scf_if(b.cmp_lt(idx, c(s * S.COLS // 2))):
                    reduce_one()
        e.barrier()
    batched(first)
    if fns is None:
        fns = fn_loads()
    for i in range(S.MIX * S.KT // S.THREADS):
        hi = e.bf16r(fns[i])
        e.lds_st_f32(b.add(fl, b.add(tid, c(S.THREADS * i))), b.fadd(hi, e.bf16r(b.fsub(fns[i], hi))))
    e.barrier()
    # ---- the layer input (slice_input) and the DSpark tap
    for pas in range(0, s * S.COLS // 2, S.THREADS):
        idx = b.add(c(pas), tid)

        def inp(idx=idx):
            t = b.lshr(idx, c(4))
            cc = b.mul(b.land(idx, c(15)), c(2))
            outs = []
            for d in range(2):
                acc = e.f(0.0)
                for h in range(S.HC):
                    rb = e.lds_f32(b.add(rl, b.add(b.add(b.mul(t, c(S.KT)), c(h * S.COLS)), b.add(cc, c(d)))))
                    acc = b.fadd(acc, b.fmul(e.ld_f32(p["pre_in"], b.add(b.mul(t, c(S.HC)), c(h))), rb))
                outs.append(acc)
            word = b.lor(e.bf16_bits(outs[0]), b.shl(e.bf16_bits(outs[1]), c(16)))
            if knobs.mutate == "put_bit":
                word = b.xor(word, c(1))
            i_el = b.add(b.mul(t, c(S.HIDDEN)), b.add(c0, cc))
            b.buffer_store_cp(b.vec_pack([word, tag], I32), e.rsrc(p["lin"]), b.mul(i_el, c(4)), c(0), aux=S.CPOL_SC1)
            if aux:
                for d in range(2):
                    acc = e.f(0.0)
                    for h in range(S.HC):
                        acc = b.fadd(acc, e.lds_f32(b.add(rl, b.add(b.add(b.mul(t, c(S.KT)), c(h * S.COLS)), b.add(cc, c(d))))))
                    v = b.fmul(acc, e.f(1.0 / S.HC))
                    b.buffer_store_bf16(e.rsrc(p["aux"]), b.mul(b.add(i_el, c(d)), c(2)), c(0), b.trunc_f32_to_bf16(v))

        if pas + S.THREADS <= s * S.COLS // 2:
            inp()
        else:
            with b.scf_if(b.cmp_lt(idx, c(s * S.COLS // 2))):
                inp()
    # ---- sum of squares: token t on wave t % 8
    r_pm = e.rsrc(p["pmix"])
    order = (1, 2, 4, 8, 16, 32) if knobs.mutate == "sq_order" else (32, 16, 8, 4, 2, 1)
    for t in range(s):
        with b.scf_if(b.cmp_eq(wave, c(t % S.WAVES))):
            x0 = e.lds_f32(b.add(rl, b.add(c(t * S.KT), b.mul(lane, c(2)))))
            x1 = e.lds_f32(b.add(rl, b.add(c(t * S.KT + 1), b.mul(lane, c(2)))))
            sq = e.butterfly(b.fadd(b.fmul(x0, x0), b.fmul(x1, x1)), order)
            with b.scf_if(b.cmp_eq(lane, c(0))):
                pi = b.add(b.mul(b.add(b.mul(task, c(s)), c(t)), c(S.PART)), c(S.MIX))
                b.buffer_store_cp(b.vec_pack([b.bitcast(sq, I32), tag], I32), r_pm, b.mul(pi, c(8)), c(0), aux=S.CPOL_SC1)
    # ---- the mix projection partials (slice_mix), a token tile at a time
    kw = S.KT // S.WAVES  # 16
    for t0 in range(0, s, TILE):
        n = min(TILE, s - t0)
        tok = b.add(c(t0), b.smin(b.land(lane, c(15)), c(n - 1)))
        cs = [b.vec_pack([e.f(0.0)] * 4, F32) for _ in range(2)]
        for kk in range(kw // 4):
            if knobs.mutate == "mix_drop" and kk == 1:
                continue
            colk = b.add(b.add(b.mul(wave, c(kw)), b.lshr(lane, c(4))), c(4 * kk))
            bv = e.lds_f32(b.add(rl, b.add(b.mul(tok, c(S.KT)), colk)))
            for h in range(2):
                row = b.add(c(h * 16), b.land(lane, c(15)))
                av = e.lds_f32(b.add(fl, b.add(b.mul(b.smin(row, c(S.MIX - 1)), c(S.KT)), colk)))
                av = b.select(b.cmp_lt(row, c(S.MIX)), av, e.f(0.0))
                cs[h] = b.mma("mfma_f32_16x16x4_f32", av, bv, cs[h])
        for h in range(2):
            at = b.add(red, b.mul(b.add(b.mul(b.add(b.mul(wave, c(2)), c(h)), c(64)), lane), c(4)))
            b.smem_store_vN(e.pool, [at], b.vec_pack([b.bitcast(b.vec_extract(cs[h], q), I32) for q in range(4)], I32), 4)
        e.barrier()
        with b.scf_if(b.cmp_lt(tid, c(S.MIX * n))):
            tl = b.div(tid, c(S.MIX))
            j = b.mod(tid, c(S.MIX))
            h = b.div(j, c(16))
            ln = b.add(b.mul(b.div(b.mod(j, c(16)), c(4)), c(16)), tl)
            tot = e.f(0.0)
            for w in range(S.WAVES):
                at = b.add(red, b.add(b.mul(b.add(b.mul(b.add(c(2 * w), h), c(64)), ln), c(4)), b.mod(j, c(4))))
                tot = b.fadd(tot, e.lds_f32(at))
            pi = b.add(b.mul(b.add(b.mul(task, c(s)), b.add(c(t0), tl)), c(S.PART)), j)
            b.buffer_store_cp(b.vec_pack([b.bitcast(tot, I32), tag], I32), r_pm, b.mul(pi, c(8)), c(0), aux=S.CPOL_SC1)
        e.barrier()
    b.ret()
    return b.kernel


def write_slice_functions(out: str, variants, knobs: SliceKnobs = SliceKnobs(), **kw) -> list:
    res = []
    for v in variants:
        form, rows, fold, aux = parse_variant(v)
        res += S.write_functions(lambda _v, f=form, r=rows, fo=fold, a=aux: build_slice_kernel(f, r, fo, a, knobs), out, [v],
                                 STAGE, ABI_K1 if form == "k1" else ABI_K2, knobs, **kw)  # fmt: skip
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="emit the rocKE V4.1 seam slice stage functions")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default=",".join(
        [f"k1f1a0n{r}" for r in range(1, 25)] + [f"k2n{r}" for r in range(1, 25)]))  # fmt: skip
    ap.add_argument("--knobs", default="")
    ap.add_argument("--flavor", default=None)
    ap.add_argument("--arch", default="gfx950")
    ap.add_argument("--bitcode", choices=("auto", "yes", "no"), default="no")
    a = ap.parse_args(argv)
    kn = S.knobs_from(SliceKnobs, a.knobs)
    vs = [v for v in a.variants.split(",") if v]
    bc = {"auto": "auto", "yes": True, "no": False}[a.bitcode]
    out = write_slice_functions(a.out, vs, kn, arch=a.arch, flavor=a.flavor, bitcode=bc)
    print(json.dumps({"knobs": S.spec_of(kn), "written": out}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
