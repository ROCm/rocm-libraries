# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""K2a ``wo_b`` of ATOM's DeepSeek-V4.1 mono as a rocKE stage function (W13 WP4 item 4.2; hybrid path).

Bit-identical re-implementation of ATOM ``attn_post.py`` (origin/main df86c482) ``run_attn_post``'s one-token-tile
wo_b block: ``wob_loads`` x 2 (``gemv_fp8_loads`` of the CTA's task and, past BLOCKS, its second task) +
``load_rows`` of the wo_a output (``x8b`` words / ``x8bs`` codes) + ``stage_wob`` per task (the scaled-MFMA chain of
``stage_wqkv``'s core, K = ``o_rows``; ``row_sum`` in wave order; the bf16 pair of rows (2 rp, 2 rp + 1) pushed to
every rank's ``attn`` peer region at ``attn_region_pair`` with ``put_bf``: ``(bf16 pair, tag)`` system scope). TP2
(``o_rows`` 4096, 32 K steps, 4 a wave), rows 1-16 (one tile).

The stage's core is ``stage_wqkv``'s (``WqkvKnobs`` levers and mutants): see that module. ABI
(``rocke_dsv41_wob_n<rows>``, ``lds="arg"``)::

    void (i32 tag, i32 task, i32 task2, i32 has2, i64 w, i64 ws, i64 x8b, i64 x8bs, i64 dst0, i64 dst1, i32 rank,
          i32 xl, i32 xsl, i32 red, i32 lds_base)

``task2``: the second task's index (its weights are loaded whatever ``has2``; ATOM loads the first task's again when
there is none: L2 hits), ``has2`` != 0: run it. ``dst<p>``: rank p's ``attn`` peer region (base + offset).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields

from rocke.core.ir import F32, I16, I32, I64, IRBuilder

from . import stage_wqkv as W

# ATOM ``dims`` (TP2) / ``attn_post`` constants (``check_atom`` compares them)
HIDDEN = 5120
O_ROWS = 4096  # Dims(2).o_rows
YB_WORDS = O_ROWS // 4
YB_GROUPS = O_ROWS // 32
ROWS = 16
TILE = 16
WOB_TASKS = HIDDEN // ROWS  # 320
TP = 2
CPOL_SYS = 17  # CM_SYS: sc0 | sc1
STAGE = "wob"
ABI = (
    ("tag", "i32"), ("task", "i32"), ("task2", "i32"), ("has2", "i32"), ("w", "i64"), ("ws", "i64"),
    ("x8b", "i64"), ("x8bs", "i64"), ("dst0", "i64"), ("dst1", "i64"), ("rank", "i32"),
    ("xl", "i32"), ("xsl", "i32"), ("red", "i32"),
)  # fmt: skip


def symbol(rows: int) -> str:
    return f"{W.SYMBOL_PREFIX}{STAGE}_n{rows}"


def build_wob_kernel(rows: int, knobs: W.WqkvKnobs = W.WqkvKnobs()):
    if not 1 <= rows <= TILE:
        raise ValueError(f"rows {rows}: the one-tile build has 1..{TILE}")
    if knobs.x_first or knobs.poll != "ir":
        raise ValueError("wob: x_first / poll=asm are wqkv-only knobs")
    b = IRBuilder(symbol(rows))
    b.kernel.attrs["max_workgroup_size"] = W.THREADS
    b.fence_barriers = "compiler"
    p = {n: b.param(n, I32 if t == "i32" else I64) for n, t in ABI}
    e = W._E(b, knobs, rows)
    c = e.c
    tag = b.readfirstlane(p["tag"])
    tasks = [b.readfirstlane(p["task"]), b.readfirstlane(p["task2"])]
    has2 = b.readfirstlane(p["has2"])
    rank = b.readfirstlane(p["rank"])
    xl, xsl, red = (b.readfirstlane(p[n]) for n in ("xl", "xsl", "red"))
    tid = b.thread_id_x()
    lane = b.land(tid, c(63))
    wave = b.lshr(tid, c(6))
    j = b.lshr(lane, c(4))
    k = O_ROWS
    kg = k // 32
    steps = k // 128
    per = steps // W.WAVES  # 4

    def weights(task):
        r_w, r_ws = e.rsrc(p["w"]), e.rsrc(p["ws"])
        ops = []
        for i in range(per):
            st = b.add(wave, c(W.WAVES * i))
            rgk = b.mul(task, c(k // 64))
            lo_at = b.add(b.mul(b.add(rgk, b.mul(st, c(2))), c(256)), b.mul(lane, c(4)))
            hi_at = b.add(b.mul(b.add(rgk, b.add(b.mul(st, c(2)), c(1))), c(256)), b.mul(lane, c(4)))
            lo = b.buffer_load_cp(r_w, b.mul(lo_at, c(4)), c(0), dwords=4, aux=knobs.w_cpol)
            hi = b.buffer_load_cp(r_w, b.mul(hi_at, c(4)), c(0), dwords=4, aux=knobs.w_cpol)
            wsi = b.add(b.mul(b.lshr(task, c(1)), c(kg)), b.add(b.mul(st, c(4)), j))
            word = b.buffer_load_cp(r_ws, b.mul(b.lshr(wsi, c(2)), c(4)), c(0), dwords=1, aux=0)
            byte = b.land(wsi, c(3))
            if knobs.mutate == "scale_byte":
                byte = b.land(b.add(byte, c(1)), c(3))
            sa = b.land(b.lshr(word, b.mul(byte, c(8))), c(0xFF))
            ops.append((st, lo, hi, sa))
        return ops

    def load_rows():
        idx = e.fresh(tid)
        r8, r8s = e.rsrc(p["x8b"]), e.rsrc(p["x8bs"])
        spans = [(r8, rows * YB_WORDS, xl, YB_WORDS), (r8s, rows * YB_GROUPS, xsl, YB_GROUPS)]
        chunks = [
            (rs, n, dst, row, b.add(idx, c(W.THREADS * i)))
            for rs, n, dst, row in spans
            for i in range((n + W.THREADS - 1) // W.THREADS)
        ]
        for b0 in range(0, len(chunks), knobs.poll_max):
            batch = chunks[b0 : b0 + knobs.poll_max]
            got = e.poll(tag, [(rs, None, b.smin(u, c(n - 1))) for rs, n, _, _, u in batch])
            for (_, n, dst, row, u), v in zip(batch, got):
                at = b.add(u, b.mul(b.div(u, c(row)), c(W.LDS_PAD)))
                with b.scf_if(b.cmp_lt(u, c(n))):
                    b.smem_store_vN(e.pool, [b.add(dst, at)], v, 1)
        e.barrier()

    ops_all = [weights(tasks[0]), weights(tasks[1])]
    load_rows()
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

    dsts = [p["dst0"], p["dst1"]]

    def stage(task, ops):
        acc = b.vec_pack([b.const_f32(0.0)] * 4, F32)
        xs = [x_operands(st) for st, *_ in ops] if knobs.ra or knobs.sop == 2 else None
        if knobs.sop:
            packed_a = [pack([sa for *_, sa in ops][q0 : q0 + 4]) for q0 in range(0, len(ops), 4)]
        if knobs.sop == 2:
            packed_b = [pack([sb for _xv, sb in xs][q0 : q0 + 4]) for q0 in range(0, len(xs), 4)]
        for i, (st, lo, hi, sa) in enumerate(ops):
            if knobs.mutate == "chain_drop" and i == 1:
                continue
            xv, sb = xs[i] if xs is not None else x_operands(st)
            osa = osb = 0
            if knobs.sop:
                sa, osa = packed_a[i // 4], i % 4
            if knobs.sop == 2:
                sb, osb = packed_b[i // 4], i % 4
            if knobs.mutate == "opsel" and knobs.sop and i == 1:
                osa = (osa + 1) % 4
            acc = b.mfma_scale_mx(b.vec_concat(lo, hi), xv, acc, sa, sb, shape="16x16x128", fmt_a="fp8",
                                  fmt_b="fp8", opsel_a=osa, opsel_b=osb)  # fmt: skip
        red_at = b.add(red, b.mul(b.add(b.mul(wave, c(64)), lane), c(4)))
        b.smem_store_vN(e.pool, [red_at], b.vec_pack([b.bitcast(b.vec_extract(acc, q), I32) for q in range(4)], I32), 4)
        e.barrier()  # stage_wob's gpu.barrier
        with b.scf_if(b.cmp_lt(tid, c(ROWS * rows // 2))):
            rp = b.land(tid, c(ROWS // 2 - 1))
            tl = b.lshr(tid, c(3))
            trow = c(0) if knobs.mutate == "row" else tl

            def row_sum(r):
                # red[(w * 64 + 16 (r / 4) + t) * 4 + r % 4], from +0.0 in wave order
                base = b.add(b.mul(b.add(b.mul(b.lshr(r, c(2)), c(16)), trow), c(4)), b.land(r, c(3)))
                tot = b.const_f32(0.0)
                order = [1, 0, 2, 3, 4, 5, 6, 7] if knobs.mutate == "wave_order" else list(range(W.WAVES))
                for w in order:
                    v = b.vec_extract(b.smem_load_vN(e.pool, b.add(red, b.add(base, c(w * 256))), dtype=F32, n=1), 0)
                    tot = b.fadd(tot, v)
                return tot

            v0 = row_sum(b.mul(rp, c(2)))
            v1 = row_sum(b.add(b.mul(rp, c(2)), c(1)))
            lo16 = b.zext(b.bitcast(b.trunc_f32_to_bf16(v0), I16), I32)
            hi16 = b.zext(b.bitcast(b.trunc_f32_to_bf16(v1), I16), I32)
            word = b.lor(lo16, b.shl(hi16, c(16)))
            if knobs.mutate == "put_bit":
                word = b.xor(word, c(1))
            colx = b.add(b.mul(task, c(ROWS)), b.mul(rp, c(2)))
            # attn_region_pair(rank, t, s, col) = (rank s + t) HIDDEN / 2 + col / 2
            pair = b.add(b.mul(b.add(b.mul(rank, c(rows)), tl), c(HIDDEN // 2)), b.lshr(colx, c(1)))
            for q in range(TP):
                b.buffer_store_cp(b.vec_pack([word, tag], I32), e.rsrc(dsts[q]), b.mul(pair, c(8)), c(0), aux=CPOL_SYS)
        e.barrier()  # stage_wob's closing gpu.barrier

    stage(tasks[0], ops_all[0])
    with b.scf_if(b.cmp_ne(has2, c(0))):
        stage(tasks[1], ops_all[1])
    b.ret()
    return b.kernel


def lower_wob_function(rows: int, knobs: W.WqkvKnobs = W.WqkvKnobs(), *, arch="gfx950", flavor=None):
    from ..qwen38_mono.device_function import DeviceFunctionKnobs, lower_device_function

    return lower_device_function(
        build_wob_kernel(rows, knobs), arch=arch, symbol=symbol(rows), llvm_flavor=flavor,
        knobs=DeviceFunctionKnobs(lds="arg", inline="always"),
    )  # fmt: skip


def write_wob_functions(out: str, rows_list, knobs: W.WqkvKnobs = W.WqkvKnobs(), *, arch="gfx950", flavor=None,
                        bitcode="auto") -> list:  # fmt: skip
    from ..qwen38_mono.device_function import write_device_function

    written = []
    for rows in rows_list:
        fn = lower_wob_function(rows, knobs, arch=arch, flavor=flavor)
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

    d = D.Dims(TP)
    got = {"HIDDEN": D.HIDDEN, "O_ROWS": d.o_rows, "YB_WORDS": d.yb_words, "WOB_TASKS": k2a.WOB_TASKS, "ROWS": D.ROWS}
    want = {"HIDDEN": HIDDEN, "O_ROWS": O_ROWS, "YB_WORDS": YB_WORDS, "WOB_TASKS": WOB_TASKS, "ROWS": ROWS}
    bad = {k: (got[k], want[k]) for k in want if got[k] != want[k]}
    if bad:
        raise ValueError(f"ATOM constants differ (atom, rocke): {bad}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="emit the rocKE V4.1 wo_b stage functions")
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
    out = write_wob_functions(a.out, rows, kn, arch=a.arch, flavor=a.flavor, bitcode=bc)
    print(json.dumps({"knobs": kn.spec(), "written": out}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
