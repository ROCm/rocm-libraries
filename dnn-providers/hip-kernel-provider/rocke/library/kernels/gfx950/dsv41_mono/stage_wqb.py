# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""K1 ``wq_b`` of ATOM's DeepSeek-V4.1 mono as a rocKE stage function (W13 WP4 item 4.2; hybrid path).

Bit-identical re-implementation of ATOM ``attn_pre.py`` (origin/main df86c482) ``build_attn_pre``'s wq_b block for a
one-token-tile, non-indexer, non-token-parallel TP2 build (rows 1-16): the CTA's four tasks (``nth_task``: first,
+ BLOCKS, ..), the first two tasks' weights in flight before the q latent's wait, every task's RoPE tables
(``wqb_rope``) loaded before it, ``load_x8`` of ``qx8`` / ``qx8s`` into LDS, then per pair of tasks ``stage_wqb``
(the scaled-MFMA core of ``stage_wqkv``, K = Q_RANK = 1280: 10 K steps, waves 0 / 1 take a second live step, the
others a dead one -- zero weights at unit scale, as ATOM's ``live`` steps; ``row_sum`` in wave order, bf16; the
GPT-J RoPE of a head's last 64 dims; bf16 stores to ``q_out``), the next pair's weights loaded after the first pair.

ABI (``rocke_dsv41_wqb_n<rows>``, ``lds="arg"``)::

    void (i32 tag, i32 task, i64 w, i64 ws, i64 qx8, i64 qx8s, i64 q_out, i64 pos, i64 cos, i64 sin,
          i32 xl, i32 xsl, i32 red, i32 lds_base)

Knobs and mutants: ``stage_wqkv.WqkvKnobs`` (``sop`` / ``ra`` levers on the chain; ``put_bit`` flips the stored
q's low bit).
"""

from __future__ import annotations

import argparse
import json
import sys

from rocke.core.ir import BF16, F32, I16, I32, I64, IRBuilder

from . import stage_wqkv as W

Q_RANK = 1280
QX8_WORDS = Q_RANK // 4
QX8_GROUPS = Q_RANK // 32
HEAD_DIM = 512
ROPE = 64
HALF = ROPE // 2
HEADS = 32  # Dims(2).heads
ROWS = 16
TILE = 16
BLOCKS = 256
PER_CTA = HEADS * HEAD_DIM // ROWS // BLOCKS  # 4
UNIT_SCALE = 127  # E8M0 1.0 (atom.mono.device.mx.UNIT_SCALE)
STAGE = "wqb"
ABI = (
    ("tag", "i32"), ("task", "i32"), ("w", "i64"), ("ws", "i64"), ("qx8", "i64"), ("qx8s", "i64"),
    ("q_out", "i64"), ("pos", "i64"), ("cos", "i64"), ("sin", "i64"), ("xl", "i32"), ("xsl", "i32"), ("red", "i32"),
)  # fmt: skip


def symbol(rows: int) -> str:
    return f"{W.SYMBOL_PREFIX}{STAGE}_n{rows}"


def build_wqb_kernel(rows: int, knobs: W.WqkvKnobs = W.WqkvKnobs()):
    if not 1 <= rows <= TILE:
        raise ValueError(f"rows {rows}: the one-tile build has 1..{TILE}")
    if knobs.x_first or knobs.poll != "ir":
        raise ValueError("wqb: x_first / poll=asm are wqkv-only knobs")
    b = IRBuilder(symbol(rows))
    b.kernel.attrs["max_workgroup_size"] = W.THREADS
    b.fence_barriers = "compiler"
    p = {n: b.param(n, I32 if t == "i32" else I64) for n, t in ABI}
    e = W._E(b, knobs, rows)
    c = e.c
    tag = b.readfirstlane(p["tag"])
    task0 = b.readfirstlane(p["task"])
    tasks = [task0 if i == 0 else b.add(task0, c(BLOCKS * i)) for i in range(PER_CTA)]
    xl, xsl, red = (b.readfirstlane(p[n]) for n in ("xl", "xsl", "red"))
    tid = b.thread_id_x()
    lane = b.land(tid, c(63))
    wave = b.lshr(tid, c(6))
    j = b.lshr(lane, c(4))
    k = Q_RANK
    kg = k // 32
    steps = k // 128  # 10
    per = -(-steps // W.WAVES)  # 2, ragged

    def step_list():
        out = []
        for i in range(per):
            st = b.add(wave, c(W.WAVES * i))
            live = None if (W.WAVES * (i + 1) <= steps) else b.cmp_lt(st, c(steps))
            out.append((st, live))
        return out

    def weights(task):
        r_w, r_ws = e.rsrc(p["w"]), e.rsrc(p["ws"])
        ops = []
        for st, live in step_list():
            stl = st if live is None else b.smin(st, c(steps - 1))
            rgk = b.mul(task, c(k // 64))
            lo_at = b.add(b.mul(b.add(rgk, b.mul(stl, c(2))), c(256)), b.mul(lane, c(4)))
            # hi: fx.min(2 st + 1, k / 64 - 1)
            hi_st = b.smin(b.add(b.mul(stl, c(2)), c(1)), c(k // 64 - 1))
            hi_at = b.add(b.mul(b.add(rgk, hi_st), c(256)), b.mul(lane, c(4)))
            lo = b.buffer_load_cp(r_w, b.mul(lo_at, c(4)), c(0), dwords=4, aux=knobs.w_cpol)
            hi = b.buffer_load_cp(r_w, b.mul(hi_at, c(4)), c(0), dwords=4, aux=knobs.w_cpol)
            wsi = b.add(b.mul(b.lshr(task, c(1)), c(kg)), b.smin(b.add(b.mul(stl, c(4)), j), c(kg - 1)))
            word = b.buffer_load_cp(r_ws, b.mul(b.lshr(wsi, c(2)), c(4)), c(0), dwords=1, aux=0)
            byte = b.land(wsi, c(3))
            if knobs.mutate == "scale_byte":
                byte = b.land(b.add(byte, c(1)), c(3))
            sa = b.land(b.lshr(word, b.mul(byte, c(8))), c(0xFF))
            ops.append((st, live, lo, hi, sa))
        return ops

    def ld_bf(ptr, i):
        v = b.buffer_load_cp(e.rsrc(ptr), b.mul(b.lshr(i, c(1)), c(4)), c(0), dwords=1, aux=0)
        half = b.land(b.lshr(v, b.mul(b.land(i, c(1)), c(16))), c(0xFFFF))
        return b.bitcast(b.shl(half, c(16)), F32)  # bf16 -> f32 is the bits in the top half

    def rope(task):
        """``wqb_rope`` of thread tid for ``task``: (cos, sin) of its token's position at its pair."""
        d = b.land(b.add(b.mul(task, c(ROWS)), b.mul(b.land(tid, c(7)), c(2))), c(HEAD_DIM - 1))
        dim0 = b.sub(d, c(HEAD_DIM - ROPE))
        tl = b.smin(b.lshr(tid, c(3)), c(rows - 1))
        pos = b.buffer_load_cp(e.rsrc(p["pos"]), b.mul(b.mul(tl, c(2)), c(4)), c(0), dwords=1, aux=0)
        jj = b.lshr(b.smax(dim0, c(0)), c(1))
        at = b.add(b.mul(pos, c(HALF)), jj)
        return ld_bf(p["cos"], at), ld_bf(p["sin"], at)

    def load_x8():
        idx = e.fresh(tid)
        r8, r8s = e.rsrc(p["qx8"]), e.rsrc(p["qx8s"])
        spans = [(r8, rows * QX8_WORDS, xl, QX8_WORDS), (r8s, rows * QX8_GROUPS, xsl, QX8_GROUPS)]
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

    x_row, s_row = W.lds_row(k // 4), W.lds_row(kg)
    col = b.smin(b.land(lane, c(15)), c(rows - 1))

    def operands(st, live, lo, hi, sa):
        """``_fp8_step_mfma``'s operands of a (possibly dead) step."""
        if live is not None:
            st = b.smin(st, c(steps - 1))
            lo = b.vec_pack([b.select(live, b.vec_extract(lo, q), c(0)) for q in range(4)], I32)
        past = b.cmp_ge(b.mul(b.add(b.mul(st, c(2)), c(1)), c(64)), c(k))
        if live is not None:
            past = b.select(live, past, b.cmp_eq(c(0), c(0)))  # past | ~live
        hi = b.vec_pack([b.select(past, c(0), b.vec_extract(hi, q)) for q in range(4)], I32)
        av = b.vec_concat(lo, hi)
        xb = b.add(xl, b.add(b.mul(col, c(x_row)), b.add(b.mul(st, c(32)), b.mul(j, c(4)))))
        xlo = b.smem_load_vN(e.pool, xb, dtype=I32, n=4)
        xhi = b.smem_load_vN(e.pool, b.add(xb, c(16)), dtype=I32, n=4)
        xhi = b.vec_pack([b.select(past, c(0), b.vec_extract(xhi, q)) for q in range(4)], I32)
        grp = b.add(b.mul(st, c(4)), j)
        sb = b.vec_extract(
            b.smem_load_vN(e.pool, b.add(xsl, b.add(b.mul(col, c(s_row)), b.smin(grp, c(kg - 1)))), dtype=I32, n=1), 0
        )
        unit = b.cmp_ge(grp, c(kg))
        if live is not None:
            unit = b.select(live, unit, b.cmp_eq(c(0), c(0)))  # unit | ~live
        sa = b.select(unit, c(UNIT_SCALE), sa)
        sb = b.select(unit, c(UNIT_SCALE), sb)
        return av, b.vec_concat(xlo, xhi), sa, sb

    def stage(task, ops, rp_rope):
        acc = b.vec_pack([b.const_f32(0.0)] * 4, F32)
        early = knobs.ra or knobs.sop
        opnd = [operands(*o) for o in ops] if early else None

        def pack(vals):
            out = vals[0]
            for q in range(1, len(vals)):
                out = b.lor(out, b.shl(vals[q], c(8 * q)))
            return out

        if knobs.sop:
            pa = pack([o[2] for o in opnd])
        if knobs.sop == 2:
            pb = pack([o[3] for o in opnd])
        for i in range(len(ops)):
            if knobs.mutate == "chain_drop" and i == 1:
                continue
            av, xv, sa, sb = opnd[i] if early else operands(*ops[i])
            osa = osb = 0
            if knobs.sop:
                sa, osa = pa, i
                if knobs.mutate == "opsel" and i == 1:
                    osa = (osa + 1) % 4
            if knobs.sop == 2:
                sb, osb = pb, i
            acc = b.mfma_scale_mx(av, xv, acc, sa, sb, shape="16x16x128", fmt_a="fp8", fmt_b="fp8",
                                  opsel_a=osa, opsel_b=osb)  # fmt: skip
        red_at = b.add(red, b.mul(b.add(b.mul(wave, c(64)), lane), c(4)))
        b.smem_store_vN(e.pool, [red_at], b.vec_pack([b.bitcast(b.vec_extract(acc, q), I32) for q in range(4)], I32), 4)
        e.barrier()
        cs, sn = rp_rope
        with b.scf_if(b.cmp_lt(tid, c(ROWS * rows // 2))):
            rp = b.land(tid, c(ROWS // 2 - 1))
            tl = b.lshr(tid, c(3))
            trow = c(0) if knobs.mutate == "row" else tl

            def row_sum(r):
                base = b.add(b.mul(b.add(b.mul(b.lshr(r, c(2)), c(16)), trow), c(4)), b.land(r, c(3)))
                tot = b.const_f32(0.0)
                for w in range(W.WAVES):
                    v = b.vec_extract(b.smem_load_vN(e.pool, b.add(red, b.add(base, c(w * 256))), dtype=F32, n=1), 0)
                    tot = b.fadd(tot, v)
                return b.cast_to_f32(b.trunc_f32_to_bf16(tot))

            r0 = b.add(b.mul(task, c(ROWS)), b.mul(rp, c(2)))
            v0 = row_sum(b.mul(rp, c(2)))
            v1 = row_sum(b.add(b.mul(rp, c(2)), c(1)))
            d = b.land(r0, c(HEAD_DIM - 1))
            live = b.cmp_ge(d, c(HEAD_DIM - ROPE))
            e0 = b.select(live, b.fsub(b.fmul(v0, cs), b.fmul(v1, sn)), v0)
            e1 = b.select(live, b.fadd(b.fmul(v0, sn), b.fmul(v1, cs)), v1)
            out = b.add(b.mul(tl, c(HEADS * HEAD_DIM)), r0)
            q0 = b.zext(b.bitcast(b.trunc_f32_to_bf16(e0), I16), I32)
            q1 = b.zext(b.bitcast(b.trunc_f32_to_bf16(e1), I16), I32)
            if knobs.mutate == "put_bit":
                q0 = b.xor(q0, c(1))
            word = b.lor(q0, b.shl(q1, c(16)))
            # the two adjacent bf16 stores of ATOM as one dword (out is even: r0 is)
            b.buffer_store_cp(word, e.rsrc(p["q_out"]), b.mul(out, c(2)), c(0), aux=0)
        e.barrier()

    ops = {0: weights(tasks[0]), 1: weights(tasks[1])}
    ropes = [rope(t) for t in tasks]
    load_x8()
    for bb in range(0, PER_CTA, 2):
        if bb > 0:
            for i in range(2):
                ops[bb + i] = weights(tasks[bb + i])
        for i in range(2):
            stage(tasks[bb + i], ops[bb + i], ropes[bb + i])
    b.ret()
    return b.kernel


def lower_wqb_function(rows: int, knobs: W.WqkvKnobs = W.WqkvKnobs(), *, arch="gfx950", flavor=None):
    from ..qwen38_mono.device_function import DeviceFunctionKnobs, lower_device_function

    return lower_device_function(
        build_wqb_kernel(rows, knobs), arch=arch, symbol=symbol(rows), llvm_flavor=flavor,
        knobs=DeviceFunctionKnobs(lds="arg", inline="always"),
    )  # fmt: skip


def write_wqb_functions(out: str, rows_list, knobs: W.WqkvKnobs = W.WqkvKnobs(), *, arch="gfx950", flavor=None,
                        bitcode="auto") -> list:  # fmt: skip
    from ..qwen38_mono.device_function import write_device_function

    written = []
    for rows in rows_list:
        fn = lower_wqb_function(rows, knobs, arch=arch, flavor=flavor)
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
    from atom.models.deepseek_v41.mono.kernels import attn_pre as k1
    from atom.models.deepseek_v41.mono.kernels import dims as D

    d = D.Dims(2)
    got = {"Q_RANK": D.Q_RANK, "HEADS": d.heads, "ROPE": k1.ROPE, "QX8_WORDS": k1.QX8_WORDS,
           "PER_CTA": d.wqb_tasks // k1.BLOCKS, "HEAD_DIM": D.HEAD_DIM}  # fmt: skip
    want = {"Q_RANK": Q_RANK, "HEADS": HEADS, "ROPE": ROPE, "QX8_WORDS": QX8_WORDS, "PER_CTA": PER_CTA,
            "HEAD_DIM": HEAD_DIM}  # fmt: skip
    bad = {k: (got[k], want[k]) for k in want if got[k] != want[k]}
    if bad:
        raise ValueError(f"ATOM constants differ (atom, rocke): {bad}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="emit the rocKE V4.1 wq_b stage functions")
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
    out = write_wqb_functions(a.out, rows, kn, arch=a.arch, flavor=a.flavor, bitcode=bc)
    print(json.dumps({"knobs": kn.spec(), "written": out}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
