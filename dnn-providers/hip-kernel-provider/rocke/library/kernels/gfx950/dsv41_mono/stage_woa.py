# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""K2a ``wo_a`` of ATOM's DeepSeek-V4.1 mono as a rocKE stage function (W13 WP4 item 4.2; hybrid path).

Bit-identical re-implementation of ATOM ``attn_post.py`` (origin/main df86c482) ``run_attn_post``'s one-token-tile wo_a
task (``woa_loads`` + ``woa_tile`` + ``stage_woa``), TP2: rows 32 task .. +32 (two 16-row tiles) of the grouped FP8
``wo_a`` [o_rows 4096, group_k 4096] times the task's group's attention output (``xo`` / ``xos`` mailboxes, MXFP8)
-- split K over 4 waves a tile (``WOA_SPLIT_K``; wave w: tile w / 4, part w % 4, K steps 8 part .. 8 part + 7 on one
accumulator), ``row_sum`` over the tile's 4 waves in order, bf16, then ATOM's ``quantize_fp8`` of a token's 32 rows
(amax floored 1e-4, the ceil code, x / scale clamped, RNE E4M3) -> ``x8b`` words (``(word, tag)``) and the code
-> ``x8bs``.

ABI (``rocke_dsv41_woa_n<rows>``, ``lds="arg"``)::

    void (i32 tag, i32 task, i64 w, i64 ws, i64 xo, i64 xos, i64 x8b, i64 x8bs, i32 xl, i32 xsl, i32 red, i32 lds_base)

Knobs / mutants: ``stage_wqkv.WqkvKnobs`` (``sop`` / ``ra`` on the chain; ``put_bit`` on the x8b word).
"""

from __future__ import annotations

import argparse
import json
import sys

from rocke.core.ir import F32, I32, I64, IRBuilder, VectorType

from . import stage_wqkv as W

HEADS = 32  # Dims(2).heads
HEAD_DIM = 512
GROUPS = 4  # Dims(2).groups
GROUP_K = HEADS // GROUPS * HEAD_DIM  # 4096
O_ROWS = GROUPS * 1024  # 4096
WOA_ROWS = 32
WOA_TASKS = O_ROWS // WOA_ROWS  # 128
SPLIT = W.WAVES * 16 // WOA_ROWS  # 4
XO_WORDS = HEADS * HEAD_DIM // 4  # 4096
XO_GROUPS = HEADS * HEAD_DIM // 32  # 512
YB_WORDS = O_ROWS // 4
FP8_MAX = 448.0
QUANT_FLOOR = 1e-4
TILE = 16
STAGE = "woa"
ABI = (
    ("tag", "i32"), ("task", "i32"), ("w", "i64"), ("ws", "i64"), ("xo", "i64"), ("xos", "i64"),
    ("x8b", "i64"), ("x8bs", "i64"), ("xl", "i32"), ("xsl", "i32"), ("red", "i32"),
)  # fmt: skip


def symbol(rows: int) -> str:
    return f"{W.SYMBOL_PREFIX}{STAGE}_n{rows}"


def build_woa_kernel(rows: int, knobs: W.WqkvKnobs = W.WqkvKnobs()):
    if not 1 <= rows <= TILE:
        raise ValueError(f"rows {rows}: the one-tile build has 1..{TILE}")
    if knobs.x_first or knobs.poll != "ir":
        raise ValueError("woa: x_first / poll=asm are wqkv-only knobs")
    b = IRBuilder(symbol(rows))
    b.kernel.attrs["max_workgroup_size"] = W.THREADS
    b.fence_barriers = "compiler"
    p = {n: b.param(n, I32 if t == "i32" else I64) for n, t in ABI}
    e = W._E(b, knobs, rows)
    c = e.c
    tag = b.readfirstlane(p["tag"])
    task = b.readfirstlane(p["task"])
    xl, xsl, red = (b.readfirstlane(p[n]) for n in ("xl", "xsl", "red"))
    tid = b.thread_id_x()
    lane = b.land(tid, c(63))
    wave = b.lshr(tid, c(6))
    j = b.lshr(lane, c(4))
    k = GROUP_K
    kg = k // 32
    per = k // 128 // SPLIT  # 8
    part = b.land(wave, c(SPLIT - 1))
    rg = b.add(b.mul(task, c(WOA_ROWS // 16)), b.lshr(wave, c(2)))  # tiled: rg + wave / split

    # weights in flight first (woa_loads before woa_tile's load_rows)
    r_w, r_ws = e.rsrc(p["w"]), e.rsrc(p["ws"])
    ops = []
    for i in range(per):
        st = b.add(b.mul(part, c(per)), c(i))
        rgk = b.mul(rg, c(k // 64))
        lo_at = b.add(b.mul(b.add(rgk, b.mul(st, c(2))), c(256)), b.mul(lane, c(4)))
        hi_at = b.add(b.mul(b.add(rgk, b.smin(b.add(b.mul(st, c(2)), c(1)), c(k // 64 - 1))), c(256)), b.mul(lane, c(4)))
        lo = b.buffer_load_cp(r_w, b.mul(lo_at, c(4)), c(0), dwords=4, aux=knobs.w_cpol)
        hi = b.buffer_load_cp(r_w, b.mul(hi_at, c(4)), c(0), dwords=4, aux=knobs.w_cpol)
        wsi = b.add(b.mul(b.lshr(rg, c(1)), c(kg)), b.smin(b.add(b.mul(st, c(4)), j), c(kg - 1)))
        word = b.buffer_load_cp(r_ws, b.mul(b.lshr(wsi, c(2)), c(4)), c(0), dwords=1, aux=0)
        byte = b.land(wsi, c(3))
        if knobs.mutate == "scale_byte":
            byte = b.land(b.add(byte, c(1)), c(3))
        sa = b.land(b.lshr(word, b.mul(byte, c(8))), c(0xFF))
        ops.append((st, lo, hi, sa))

    # load_rows: the task's group's attention output rows (words, then codes) into LDS
    g = b.lshr(task, c(5))  # task // (WOA_TASKS // GROUPS)
    words, groups = k // 4, k // 32
    idx = e.fresh(tid)
    rx, rxs = e.rsrc(p["xo"]), e.rsrc(p["xos"])
    spans = [(rx, rows * words, xl, words, XO_WORDS, b.mul(g, c(words))),
             (rxs, rows * groups, xsl, groups, XO_GROUPS, b.mul(g, c(groups)))]  # fmt: skip
    chunks = [
        (rs, n, dst, row, stride, off, b.add(idx, c(W.THREADS * i)))
        for rs, n, dst, row, stride, off in spans
        for i in range((n + W.THREADS - 1) // W.THREADS)
    ]
    for b0 in range(0, len(chunks), knobs.poll_max):
        batch = chunks[b0 : b0 + knobs.poll_max]
        specs = []
        for rs, n, dst, row, stride, off, u in batch:
            uu = b.smin(u, c(n - 1))
            # pair (u / row) stride + off + u % row
            specs.append((rs, None, b.add(b.add(b.mul(b.div(uu, c(row)), c(stride)), off), b.sub(uu, b.mul(b.div(uu, c(row)), c(row))))))
        got = e.poll(tag, specs)
        for (_, n, dst, row, _s, _o, u), v in zip(batch, got):
            at = b.add(u, b.mul(b.div(u, c(row)), c(W.LDS_PAD)))
            with b.scf_if(b.cmp_lt(u, c(n))):
                b.smem_store_vN(e.pool, [b.add(dst, at)], v, 1)
    e.barrier()

    # the chain (stage_woa's gemv_fp8_mfmas, split 4 tiled)
    x_row, s_row = W.lds_row(words), W.lds_row(groups)
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

    acc = b.vec_pack([b.const_f32(0.0)] * 4, F32)
    xs = [x_operands(st) for st, *_ in ops] if knobs.ra or knobs.sop == 2 else None
    if knobs.sop:
        packed_a = [pack([o[3] for o in ops][q0 : q0 + 4]) for q0 in range(0, per, 4)]
    if knobs.sop == 2:
        packed_b = [pack([x[1] for x in xs][q0 : q0 + 4]) for q0 in range(0, per, 4)]
    for i, (st, lo, hi, sa) in enumerate(ops):
        if knobs.mutate == "chain_drop" and i == 1:
            continue
        xv, sb = xs[i] if xs is not None else x_operands(st)
        osa = osb = 0
        if knobs.sop:
            sa, osa = packed_a[i // 4], i % 4
            if knobs.mutate == "opsel" and i == 1:
                osa = (osa + 1) % 4
        if knobs.sop == 2:
            sb, osb = packed_b[i // 4], i % 4
        acc = b.mfma_scale_mx(b.vec_concat(lo, hi), xv, acc, sa, sb, shape="16x16x128", fmt_a="fp8", fmt_b="fp8",
                              opsel_a=osa, opsel_b=osb)  # fmt: skip
    red_at = b.add(red, b.mul(b.add(b.mul(wave, c(64)), lane), c(4)))
    b.smem_store_vN(e.pool, [red_at], b.vec_pack([b.bitcast(b.vec_extract(acc, q), I32) for q in range(4)], I32), 4)
    e.barrier()

    # the epilogue: thread (t, row), a token's 32 rows on 32 lanes of one wave
    mine = b.cmp_lt(tid, c(WOA_ROWS * rows))
    tl = b.smin(b.lshr(tid, c(5)), c(rows - 1))
    trow = c(0) if knobs.mutate == "row" else tl
    row = b.land(tid, c(WOA_ROWS - 1))
    tile = b.lshr(row, c(4))
    r = b.land(row, c(15))
    base = b.add(b.mul(b.add(b.mul(b.lshr(r, c(2)), c(16)), trow), c(4)), b.land(r, c(3)))
    tot = b.const_f32(0.0)
    for w in range(SPLIT):  # row_sum over waves [tile * split + w]
        wv = b.add(b.mul(tile, c(SPLIT)), c(w))
        v = b.vec_extract(b.smem_load_vN(e.pool, b.add(red, b.add(base, b.mul(wv, c(256)))), dtype=F32, n=1), 0)
        tot = b.fadd(tot, v)
    y = b.cast_to_f32(b.trunc_f32_to_bf16(tot))
    ay = b.fabs(y)
    for off in (1, 2, 4, 8, 16):
        ay = b.fmax(ay, b.bitcast(b.warp_shuffle_xor(b.bitcast(ay, I32), off), F32))
    amax = b.fmax(ay, b.const_f32(QUANT_FLOOR))
    # code_ceil(amax / 448): biased exponent, + 1 if a mantissa bit, clamped [1, 254]
    bits = b.bitcast(b.fmul(amax, b.const_f32(1.0 / FP8_MAX)), I32)
    ex = b.add(b.land(b.lshr(bits, c(23)), c(0xFF)), b.select(b.cmp_ne(b.land(bits, c(0x7FFFFF)), c(0)), c(1), c(0)))
    code = b.smax(b.smin(ex, c(254)), c(1))
    scale = b.bitcast(b.shl(code, c(23)), F32)
    q = b.fmax(b.fmin(b.fdiv(y, scale), b.const_f32(FP8_MAX)), b.const_f32(-FP8_MAX))
    qi = b.bitcast(q, I32)
    nb = [b.bitcast(b.ds_bpermute(b.mul(b.smin(b.add(lane, c(kk)), c(63)), c(4)), qi), F32) for kk in (1, 2, 3)]
    q8 = b.cvt_pk_fp8_f32x4(b.vec_pack([q, *nb], F32))
    word = b.bitcast(q8, I32)
    if knobs.mutate == "put_bit":
        word = b.xor(word, c(1))
    with b.scf_if(mine):
        with b.scf_if(b.cmp_eq(b.land(row, c(3)), c(0))):
            i_pair = b.add(b.mul(tl, c(YB_WORDS)), b.lshr(b.add(b.mul(task, c(WOA_ROWS)), row), c(2)))
            b.buffer_store_cp(b.vec_pack([word, tag], I32), e.rsrc(p["x8b"]), b.mul(i_pair, c(8)), c(0),
                              aux=W.CPOL_SC1)  # fmt: skip
        with b.scf_if(b.cmp_eq(row, c(0))):
            i_pair = b.add(b.mul(tl, c(O_ROWS // 32)), task)
            b.buffer_store_cp(b.vec_pack([code, tag], I32), e.rsrc(p["x8bs"]), b.mul(i_pair, c(8)), c(0),
                              aux=W.CPOL_SC1)  # fmt: skip
    e.barrier()
    b.ret()
    return b.kernel


def lower_woa_function(rows: int, knobs: W.WqkvKnobs = W.WqkvKnobs(), *, arch="gfx950", flavor=None):
    from ..qwen38_mono.device_function import DeviceFunctionKnobs, lower_device_function

    return lower_device_function(
        build_woa_kernel(rows, knobs), arch=arch, symbol=symbol(rows), llvm_flavor=flavor,
        knobs=DeviceFunctionKnobs(lds="arg", inline="always"),
    )  # fmt: skip


def write_woa_functions(out: str, rows_list, knobs: W.WqkvKnobs = W.WqkvKnobs(), *, arch="gfx950", flavor=None,
                        bitcode="auto") -> list:  # fmt: skip
    from ..qwen38_mono.device_function import write_device_function

    written = []
    for rows in rows_list:
        fn = lower_woa_function(rows, knobs, arch=arch, flavor=flavor)
        paths = write_device_function(fn, out, stem=f"{STAGE}_n{rows}", bitcode=bitcode)
        with open(paths["json"]) as f:
            meta = json.load(f)
        meta.update(stage=STAGE, rows=rows, stage_knobs=knobs.spec(), abi_v41=[list(a) for a in ABI])
        with open(paths["json"], "w", newline="\n") as f:
            json.dump(meta, f, indent=1)
            f.write("\n")
        written.append({"rows": rows, **paths})
    return written


def check_atom(atom_root: str | None = None) -> None:
    if atom_root:
        sys.path.insert(0, atom_root)
    from atom.models.deepseek_v41.mono.kernels import attn_post as k2a
    from atom.models.deepseek_v41.mono.kernels import dims as D

    d = D.Dims(2)
    got = {"GROUP_K": d.group_k, "O_ROWS": d.o_rows, "WOA_TASKS": d.woa_tasks, "SPLIT": k2a.WOA_SPLIT_K,
           "XO_WORDS": d.xo_words, "QUANT_FLOOR": k2a.QUANT_FLOOR}  # fmt: skip
    want = {"GROUP_K": GROUP_K, "O_ROWS": O_ROWS, "WOA_TASKS": WOA_TASKS, "SPLIT": SPLIT, "XO_WORDS": XO_WORDS,
            "QUANT_FLOOR": QUANT_FLOOR}  # fmt: skip
    bad = {k: (got[k], want[k]) for k in want if got[k] != want[k]}
    if bad:
        raise ValueError(f"ATOM constants differ (atom, rocke): {bad}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="emit the rocKE V4.1 wo_a stage functions")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", default=",".join(str(r) for r in range(1, 17)))
    ap.add_argument("--knobs", default="")
    ap.add_argument("--flavor", default=None)
    ap.add_argument("--arch", default="gfx950")
    ap.add_argument("--bitcode", choices=("auto", "yes", "no"), default="no")
    ap.add_argument("--atom-root", default=None)
    a = ap.parse_args(argv)
    if a.atom_root:
        check_atom(a.atom_root)
    kn = W.knobs_from_spec(a.knobs)
    rows = [int(r) for r in a.rows.split(",") if r]
    if kn.mutate == "row":
        rows = [r for r in rows if r > 1]
    bc = {"auto": "auto", "yes": True, "no": False}[a.bitcode]
    out = write_woa_functions(a.out, rows, kn, arch=a.arch, flavor=a.flavor, bitcode=bc)
    print(json.dumps({"knobs": kn.spec(), "written": out}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
