# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""K2b shared expert gate / up of ATOM's DeepSeek-V4.1 mono as a rocKE stage function (W13 WP4 item 4.2; hybrid).

Bit-identical re-implementation of ATOM ``moe.py`` (origin/main df86c482) ``run_moe``'s one-token-tile shared unit,
TP2: the gate rows' weights in flight (``gemv_fp8_loads`` of ``sgu`` rows 16 task ..), the tokens' MXFP8 rows
(``load_x8_rows``: lanes < rows wait on their token's ``x8rdy`` pair, then plain 16-byte device-scope loads of the
``x8`` / ``x8s`` words into LDS), then ``stage_shared``: the gate GEMV (``stage_wqkv``'s core, K = 5120), the gate's
row sums; the up rows' weights (``shared_tasks + task``) and their GEMV; bf16 (gate, up) -> ATOM's
``_silu_mul`` (gate min 10 through bf16, up med3 +-10, g rcp(1 + exp(-g)) x up) -> bf16 -> the ``smid`` pair and the
LDS ``pair`` word; an odd task then ``smid_quant``: the even task's 16 values polled, the group's amax (floor 1e-4),
the ceil code, x rcp_pow2(code) clamped, RNE E4M3 -> ``smq`` words and the ``smc`` code.

ABI (``rocke_dsv41_shared_n<rows>``, ``lds="arg"``)::

    void (i32 tag, i32 task, i64 w, i64 ws, i64 x8, i64 x8s, i64 x8rdy, i64 smid, i64 smq, i64 smc,
          i32 xl, i32 xsl, i32 red, i32 pair, i32 lds_base)

Knobs / mutants: ``stage_wqkv.WqkvKnobs`` (``sop`` / ``ra``; ``put_bit`` on the smid payload).
"""

from __future__ import annotations

import argparse
import json
import sys

from rocke.core.ir import F32, I32, I64, IRBuilder

from . import stage_wqkv as W

HIDDEN = 5120
SH_INTER = 1152  # Dims(2).sh_inter
SHARED_TASKS = SH_INTER // 16  # 72
X8_WORDS = HIDDEN // 4
X8_GROUPS = HIDDEN // 32
ROWS = 16
TILE = 16
LIMIT = 10.0
FP8_MAX = 448.0
STAGE = "shared"
ABI = (
    ("tag", "i32"), ("task", "i32"), ("w", "i64"), ("ws", "i64"), ("x8", "i64"), ("x8s", "i64"), ("x8rdy", "i64"),
    ("smid", "i64"), ("smq", "i64"), ("smc", "i64"), ("xl", "i32"), ("xsl", "i32"), ("red", "i32"), ("pair", "i32"),
)  # fmt: skip


def symbol(rows: int) -> str:
    return f"{W.SYMBOL_PREFIX}{STAGE}_n{rows}"


def build_shared_kernel(rows: int, knobs: W.WqkvKnobs = W.WqkvKnobs()):
    if not 1 <= rows <= TILE:
        raise ValueError(f"rows {rows}: the one-tile build has 1..{TILE}")
    if knobs.x_first or knobs.poll != "ir":
        raise ValueError("shared: x_first / poll=asm are wqkv-only knobs")
    b = IRBuilder(symbol(rows))
    b.kernel.attrs["max_workgroup_size"] = W.THREADS
    b.fence_barriers = "compiler"
    p = {n: b.param(n, I32 if t == "i32" else I64) for n, t in ABI}
    e = W._E(b, knobs, rows)
    c = e.c
    tag = b.readfirstlane(p["tag"])
    task = b.readfirstlane(p["task"])
    xl, xsl, red, pair = (b.readfirstlane(p[n]) for n in ("xl", "xsl", "red", "pair"))
    tid = b.thread_id_x()
    lane = b.land(tid, c(63))
    wave = b.lshr(tid, c(6))
    j = b.lshr(lane, c(4))
    k = HIDDEN
    kg = k // 32
    per = k // 128 // W.WAVES  # 5

    def weights(rg):
        r_w, r_ws = e.rsrc(p["w"]), e.rsrc(p["ws"])
        ops = []
        for i in range(per):
            st = b.add(wave, c(W.WAVES * i))
            rgk = b.mul(rg, c(k // 64))
            lo_at = b.add(b.mul(b.add(rgk, b.mul(st, c(2))), c(256)), b.mul(lane, c(4)))
            hi_at = b.add(b.mul(b.add(rgk, b.add(b.mul(st, c(2)), c(1))), c(256)), b.mul(lane, c(4)))
            lo = b.buffer_load_cp(r_w, b.mul(lo_at, c(4)), c(0), dwords=4, aux=knobs.w_cpol)
            hi = b.buffer_load_cp(r_w, b.mul(hi_at, c(4)), c(0), dwords=4, aux=knobs.w_cpol)
            wsi = b.add(b.mul(b.lshr(rg, c(1)), c(kg)), b.add(b.mul(st, c(4)), j))
            word = b.buffer_load_cp(r_ws, b.mul(b.lshr(wsi, c(2)), c(4)), c(0), dwords=1, aux=0)
            byte = b.land(wsi, c(3))
            if knobs.mutate == "scale_byte":
                byte = b.land(b.add(byte, c(1)), c(3))
            ops.append((st, lo, hi, b.land(b.lshr(word, b.mul(byte, c(8))), c(0xFF))))
        return ops

    gops = weights(task)
    # load_x8_rows(c, "x8", "x8s", x8l, x8sl, lambda j: j, rows)
    with b.scf_if(b.cmp_lt(tid, c(TILE))):
        e.poll(tag, [(e.rsrc(p["x8rdy"]), None, b.smin(tid, c(rows - 1)))])
    e.barrier()
    for rs_addr, width, dst in ((p["x8"], X8_WORDS, xl), (p["x8s"], X8_GROUPS, xsl)):
        quads = width // 4
        rsrc = e.rsrc(rs_addr)
        for i in range(-(-TILE * quads // W.THREADS)):
            q = b.add(tid, c(W.THREADS * i))
            row = b.div(q, c(quads))
            ok = b.cmp_lt(q, c(TILE * quads))
            with b.scf_if(ok):
                with b.scf_if(b.cmp_lt(row, c(rows))):
                    qq = b.sub(q, b.mul(row, c(quads)))
                    off = b.add(b.mul(row, c(width)), b.mul(qq, c(4)))
                    v = b.buffer_load_cp(rsrc, b.mul(off, c(4)), c(0), dwords=4, aux=W.CPOL_SC1)
                    b.smem_store_vN(e.pool, [b.add(dst, b.add(b.mul(row, c(W.lds_row(width))), b.mul(qq, c(4))))], v, 4)
    e.barrier()

    x_row, s_row = W.lds_row(k // 4), W.lds_row(kg)
    col = b.smin(b.land(lane, c(15)), c(rows - 1))

    def x_operands(st):
        xb = b.add(xl, b.add(b.mul(col, c(x_row)), b.add(b.mul(st, c(32)), b.mul(j, c(4)))))
        xlo = b.smem_load_vN(e.pool, xb, dtype=I32, n=4)
        xhi = b.smem_load_vN(e.pool, b.add(xb, c(16)), dtype=I32, n=4)
        grp = b.add(b.mul(st, c(4)), j)
        sb = b.vec_extract(b.smem_load_vN(e.pool, b.add(xsl, b.add(b.mul(col, c(s_row)), grp)), dtype=I32, n=1), 0)
        return b.vec_concat(xlo, xhi), sb

    def pack(vals):
        out = vals[0]
        for q in range(1, len(vals)):
            out = b.lor(out, b.shl(vals[q], c(8 * q)))
        return out

    def gemv(ops):
        acc = b.vec_pack([b.const_f32(0.0)] * 4, F32)
        xs = [x_operands(st) for st, *_ in ops] if knobs.ra or knobs.sop == 2 else None
        if knobs.sop:
            pa = [pack([o[3] for o in ops][q0 : q0 + 4]) for q0 in range(0, per, 4)]
        if knobs.sop == 2:
            pb = [pack([x[1] for x in xs][q0 : q0 + 4]) for q0 in range(0, per, 4)]
        for i, (st, lo, hi, sa) in enumerate(ops):
            if knobs.mutate == "chain_drop" and i == 1:
                continue
            xv, sb = xs[i] if xs is not None else x_operands(st)
            osa = osb = 0
            if knobs.sop:
                sa, osa = pa[i // 4], i % 4
                if knobs.mutate == "opsel" and i == 1:
                    osa = (osa + 1) % 4
            if knobs.sop == 2:
                sb, osb = pb[i // 4], i % 4
            acc = b.mfma_scale_mx(b.vec_concat(lo, hi), xv, acc, sa, sb, shape="16x16x128", fmt_a="fp8", fmt_b="fp8",
                                  opsel_a=osa, opsel_b=osb)  # fmt: skip
        red_at = b.add(red, b.mul(b.add(b.mul(wave, c(64)), lane), c(4)))
        b.smem_store_vN(e.pool, [red_at], b.vec_pack([b.bitcast(b.vec_extract(acc, q), I32) for q in range(4)], I32), 4)

    mine = b.cmp_lt(tid, c(ROWS * rows))
    r = b.land(tid, c(ROWS - 1))
    tl = b.lshr(tid, c(4))
    trow = c(0) if knobs.mutate == "row" else tl

    def row_sum():
        base = b.add(b.mul(b.add(b.mul(b.lshr(r, c(2)), c(16)), trow), c(4)), b.land(r, c(3)))
        tot = b.const_f32(0.0)
        for w in range(W.WAVES):
            v = b.vec_extract(b.smem_load_vN(e.pool, b.add(red, b.add(base, c(w * 256))), dtype=F32, n=1), 0)
            tot = b.fadd(tot, v)
        return tot

    def bfr(v):
        return b.cast_to_f32(b.trunc_f32_to_bf16(v))

    gemv(gops)
    e.barrier()
    gate = b.select(mine, row_sum(), b.const_f32(0.0))
    e.barrier()
    uops = weights(b.add(task, c(SHARED_TASKS)))
    gemv(uops)
    e.barrier()
    with b.scf_if(mine):
        up = row_sum()
        g = bfr(b.fmin(bfr(gate), b.const_f32(LIMIT)))
        u = b.fmax(b.fmin(bfr(up), b.const_f32(LIMIT)), b.const_f32(-LIMIT))
        act = b.fmul(g, b.rcp_fast(b.fadd(b.const_f32(1.0), b.exp(b.fneg(g)) if hasattr(b, "fneg") else b.exp(b.fsub(b.const_f32(0.0), g)))))
        v = bfr(b.fmul(act, u))
        bits = b.bitcast(v, I32)
        if knobs.mutate == "put_bit":
            bits = b.xor(bits, c(1))
        i_pair = b.add(b.mul(tl, c(SH_INTER)), b.add(b.mul(task, c(ROWS)), r))
        b.buffer_store_cp(b.vec_pack([bits, tag], I32), e.rsrc(p["smid"]), b.mul(i_pair, c(8)), c(0), aux=W.CPOL_SC1)
        b.smem_store_vN(e.pool, [b.add(pair, tid)], b.bitcast(v, I32), 1)
    e.barrier()
    # smid_quant: an odd task, thread t < rows
    with b.scf_if(b.cmp_eq(b.land(task, c(1)), c(1))):
        with b.scf_if(b.cmp_lt(tid, c(rows))):
            r_sm = e.rsrc(p["smid"])
            base_pair = b.add(b.mul(tid, c(SH_INTER)), b.mul(b.sub(task, c(1)), c(ROWS)))
            got = e.poll(tag, [(r_sm, None, b.add(base_pair, c(rr))) for rr in range(ROWS)])
            vals = [b.bitcast(v, F32) for v in got]
            vals += [b.bitcast(b.vec_extract(b.smem_load_vN(e.pool, b.add(pair, b.add(b.mul(tid, c(ROWS)), c(rr))),
                                                            dtype=I32, n=1), 0), F32) for rr in range(ROWS)]  # fmt: skip
            amax = b.const_f32(1e-4)
            for y in vals:
                amax = b.fmax(amax, b.fabs(y))
            bits = b.bitcast(b.fmul(amax, b.const_f32(1.0 / FP8_MAX)), I32)
            ex = b.add(b.land(b.lshr(bits, c(23)), c(0xFF)),
                       b.select(b.cmp_ne(b.land(bits, c(0x7FFFFF)), c(0)), c(1), c(0)))  # fmt: skip
            code = b.smax(b.smin(ex, c(254)), c(1))
            inv = b.bitcast(b.shl(b.sub(c(254), code), c(23)), F32)
            gq = b.lshr(task, c(1))
            r_q, r_c = e.rsrc(p["smq"]), e.rsrc(p["smc"])
            for w in range(8):
                q4 = [b.fmax(b.fmin(b.fmul(vals[4 * w + i], inv), b.const_f32(FP8_MAX)), b.const_f32(-FP8_MAX))
                      for i in range(4)]  # fmt: skip
                word = b.bitcast(b.cvt_pk_fp8_f32x4(b.vec_pack(q4, F32)), I32)
                at = b.add(b.mul(tid, c(SH_INTER // 4)), b.add(b.mul(gq, c(8)), c(w)))
                b.buffer_store_cp(b.vec_pack([word, tag], I32), r_q, b.mul(at, c(8)), c(0), aux=W.CPOL_SC1)
            atc = b.add(b.mul(tid, c(SH_INTER // 32)), gq)
            b.buffer_store_cp(b.vec_pack([code, tag], I32), r_c, b.mul(atc, c(8)), c(0), aux=W.CPOL_SC1)
    e.barrier()
    b.ret()
    return b.kernel


def lower_shared_function(rows: int, knobs: W.WqkvKnobs = W.WqkvKnobs(), *, arch="gfx950", flavor=None):
    from ..qwen38_mono.device_function import DeviceFunctionKnobs, lower_device_function

    return lower_device_function(
        build_shared_kernel(rows, knobs), arch=arch, symbol=symbol(rows), llvm_flavor=flavor,
        knobs=DeviceFunctionKnobs(lds="arg", inline="always"),
    )  # fmt: skip


def write_shared_functions(out: str, rows_list, knobs: W.WqkvKnobs = W.WqkvKnobs(), *, arch="gfx950", flavor=None,
                           bitcode="auto") -> list:  # fmt: skip
    from ..qwen38_mono.device_function import write_device_function

    written = []
    for rows in rows_list:
        fn = lower_shared_function(rows, knobs, arch=arch, flavor=flavor)
        paths = write_device_function(fn, out, stem=f"{STAGE}_n{rows}", bitcode=bitcode)
        with open(paths["json"]) as f:
            meta = json.load(f)
        meta.update(stage=STAGE, rows=rows, stage_knobs=knobs.spec(), abi_v41=[list(a) for a in ABI])
        with open(paths["json"], "w", newline="\n") as f:
            json.dump(meta, f, indent=1)
            f.write("\n")
        written.append({"rows": rows, **paths})
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="emit the rocKE V4.1 shared-expert stage functions")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", default=",".join(str(r) for r in range(1, 17)))
    ap.add_argument("--knobs", default="")
    ap.add_argument("--flavor", default=None)
    ap.add_argument("--arch", default="gfx950")
    ap.add_argument("--bitcode", choices=("auto", "yes", "no"), default="no")
    a = ap.parse_args(argv)
    kn = W.knobs_from_spec(a.knobs)
    rows = [int(r) for r in a.rows.split(",") if r]
    if kn.mutate == "row":
        rows = [r for r in rows if r > 1]
    bc = {"auto": "auto", "yes": True, "no": False}[a.bitcode]
    out = write_shared_functions(a.out, rows, kn, arch=a.arch, flavor=a.flavor, bitcode=bc)
    print(json.dumps({"knobs": kn.spec(), "written": out}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
