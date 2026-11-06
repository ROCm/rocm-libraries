# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""K2b routed-expert ``ug`` (MXFP4 gate / up + its epilogue) of ATOM's DeepSeek-V4.1 mono as a rocKE stage function
(W13 WP4 item 4.3, the K2b MXFP4 MoE port; hybrid path).

Bit-identical re-implementation of ATOM ``moe.py`` (origin/main df86c482) ``ug_pass`` as one-token-tile ``stage_ug``
calls it, TP2 (a routed expert's 1152 intermediate columns = 36 FP4 groups, w13 = 2304 rows a expert, K = 5120):

* ``ug_wave_pair``: wave w takes group p = w / (2 slices) of the task's ``groups`` (ATOM ``ug_groups``: 1 below 7
  rows, else 2), its 16-column half bb = w % 2 -- w13's gate tile 2 b and up tile 2 b + 1 (b = 2 (part0 + p) + bb),
  one 32-row e8m0 scale block -- over K slice ks = w / 2 % slices (slices = 8 / (2 groups): quarters or halves).
* ``gemv_fp4_pair``: per round of ``ug_rounds`` the block's scale words (aiter ``shuffle_scale`` layout) to the wave's
  LDS scale area, both tiles' (16, 16)-preshuffled FP4 weight operands (``fp4_tile_load``, nontemporal), the rounds
  ``ahead`` = slices / 2 of loads in flight past the computing one; each step's MXFP8 B operand (the pick's token row
  in LDS, column ``col``) read once for both tiles' ``v_mfma_scale_f32_16x16x128_f8f6f4`` (A fp4, B fp8).
* the partial C of (group, half, slice, gate / up) to ``red``; barrier; ``ug_epilogue`` a thread a (group, pick,
  column): the slices summed pairwise, gate min 10, up med3 +-10, ``g rcp(1 + exp2(-g log2 e)) u`` (v_exp_f32 /
  v_rcp_f32), bf16, the 32 columns' amax (floor 1e-10), the ceil e8m0 code, x rcp_pow2(code) clamped, RNE E4M3,
  four lanes' bytes a word -> MID (plain words, device scope) and the code -> MIDS.

For TP2 every task's groups lie in the real width (36 % (32 groups) parts), so ATOM's ``real`` branch is always
taken and is not emitted (``build_ug_kernel`` asserts it).

ABI (``rocke_dsv41_ug_n<rows>g<groups>``, ``lds="arg"``)::

    void (i32 e, i32 part0, i32 first, i32 count, i32 col, i64 w, i64 ws, i64 mid, i64 mids,
          i32 xl, i32 xsl, i32 red, i32 wsl, i32 lds_base)

``e`` the slot's expert, ``part0`` the task's first group, ``first`` / ``count`` the slot's picks (pick order),
``col`` (per lane) the LDS row of the pick of column lane % 16 (its token). ``xl`` / ``xsl`` / ``red`` / ``wsl``:
LDS word offsets from ``lds_base`` (ATOM passes ``lds_base`` = 0 and absolute word addresses). The caller publishes
the task's UGF flag (ATOM's ``stage_ug``).

Knobs (:class:`stage_wqkv.WqkvKnobs`): ``sop`` 1 (the scale word as read from LDS, its byte picked with a static
``op_sel`` = 2 (step % 2) + tile: ATOM's ``usop`` lever), ``ra`` 1 (a round's B operands read before its first MFMA:
ATOM's ``ura``), ``w_cpol`` (weight cache bits). Mutants: ``put_bit`` (MID word low bit), ``chain_drop`` (step 1 of
every round of 2+ steps left out), ``scale_byte`` (scale byte index + 1), ``row`` (every pick's MID row = the
slot's first), ``opsel`` (with sop 1: step 1's op_sel + 1).
"""

from __future__ import annotations

import argparse
import json
import sys

from rocke.core.ir import F32, I32, I64, IRBuilder

from . import stage_wqkv as W

HIDDEN = 5120
MOE_INTER_REAL = 1152  # Dims(2).inter_real
MOE_INTER = 1152  # Dims(2).inter (padded to 128)
N_ROWS = 2 * MOE_INTER  # w13 rows a expert
GROUPS_REAL = MOE_INTER_REAL // 32  # 36
SCALE_COLS = HIDDEN // 32  # 160
MID_WORDS = MOE_INTER // 4  # 288
MID_CODES = MOE_INTER // 32  # 36
DOWN_SCALE_COLS = 40  # padded(MOE_INTER // 32, 8)
WSL_WORDS = max(256, 2 * (-(-(DOWN_SCALE_COLS * 8) // 256) * 256))  # moe.wsl_words: 1024
UG_PART = 32
LIMIT = 10.0
LOG2E = 1.4426950408889634
FP8_MAX = 448.0
STAGE = "ug"
MUTANTS = ("put_bit", "chain_drop", "scale_byte", "row", "opsel")
ABI = (
    ("e", "i32"), ("part0", "i32"), ("first", "i32"), ("count", "i32"), ("col", "i32"), ("w", "i64"), ("ws", "i64"),
    ("mid", "i64"), ("mids", "i64"), ("xl", "i32"), ("xsl", "i32"), ("red", "i32"), ("wsl", "i32"),
)  # fmt: skip


def variant(rows: int, groups: int) -> str:
    return f"n{rows}g{groups}"


def symbol(rows: int, groups: int) -> str:
    return f"{W.SYMBOL_PREFIX}{STAGE}_{variant(rows, groups)}"


def _rounds(b, c, ks, slices):
    """ATOM ``ug_rounds``: (sc0, st0, n, sc_steps), sc0 / st0 traced."""
    blocks = HIDDEN // 128 // 8  # 5
    if slices == 4:
        return [(b.add(c(8 * i), b.mul(ks, c(2))),) * 2 + (2, 2) for i in range(blocks)]
    assert slices == 2
    out = [(b.add(c(16 * r), b.mul(ks, c(8))),) * 2 + (8, 8) for r in range(blocks // 2)]
    if blocks % 2:
        last = (blocks - 1) * 8
        out.append((c(last), b.add(c(last), b.mul(ks, c(4))), 4, 8))
    return out


def build_ug_kernel(rows: int, groups: int, knobs: W.WqkvKnobs = W.WqkvKnobs(), la: int = 0):
    """``la`` (W15 climb r0 fix; 0 = parity): B-operand LOOKAHEAD of depth ``la`` (1 or 2) over the flattened K steps
    of every round: step g + la's LDS reads issued right before step g's MFMAs (across round boundaries), so only
    ``la`` steps' operands are live ahead, where ``ra`` 1 reads a whole round's (8 steps at groups 2) before its first
    MFMA. Same MFMA operands, order and accumulators: bitwise. Not with ``ra``."""
    if la not in (0, 1, 2) or (la and knobs.ra):
        raise ValueError("ug la: 0 | 1 | 2, and not with ra")
    if not 1 <= rows <= W.TILE:
        raise ValueError(f"rows {rows}: the one-tile build has 1..{W.TILE}")
    if groups not in (1, 2):
        raise ValueError("groups (ATOM ug_groups) must be 1 or 2")
    if knobs.x_first or knobs.poll != "ir" or knobs.sop == 2 or knobs.mutate in ("wave_order",):
        raise ValueError("ug: x_first / poll=asm / sop=2 / wave_order do not apply")
    if knobs.mutate and knobs.mutate not in MUTANTS:
        raise ValueError(f"ug mutants: {MUTANTS}")
    if knobs.mutate == "opsel" and not knobs.sop:
        raise ValueError("mutant opsel needs sop=1")
    if knobs.mutate == "row" and rows < 2:
        raise ValueError("mutant row needs rows > 1")
    assert GROUPS_REAL % groups == 0  # every part real at TP2: ATOM's ``if real`` always taken
    slices = W.WAVES // (2 * groups)
    ahead = slices // 2
    b = IRBuilder(symbol(rows, groups))
    b.kernel.attrs["max_workgroup_size"] = W.THREADS
    b.fence_barriers = "compiler"
    p = {n: b.param(n, I32 if t == "i32" else I64) for n, t in ABI}
    e_ = W._E(b, knobs, rows)
    c = e_.c

    def rsrc_full(addr):  # aiter's buffer resource: num_records 0xFFFFFFFF (w13 is > 2 GiB a rank)
        from rocke.core.ir import PtrType

        ptr = b.int_to_ptr(b.readfirstlane(addr), PtrType(I32, "global"))
        return b.buffer_rsrc(ptr, c(-1))

    ex, part0, first, count = (b.readfirstlane(p[n]) for n in ("e", "part0", "first", "count"))
    xl, xsl, red, wsl = (b.readfirstlane(p[n]) for n in ("xl", "xsl", "red", "wsl"))
    col = p["col"]
    tid = b.thread_id_x()
    lane = b.land(tid, c(63))
    wave = b.lshr(tid, c(6))
    j = b.lshr(lane, c(4))
    pw = b.div(wave, c(2 * slices))  # group p of the task
    bb = b.land(wave, c(1))
    ks = b.land(b.lshr(wave, c(1)), c(slices - 1))
    rg0 = b.mul(b.add(b.mul(b.add(part0, pw), c(2)), bb), c(2))
    sl = b.add(wsl, b.mul(wave, c(WSL_WORDS)))
    r_w, r_ws = rsrc_full(p["w"]), rsrc_full(p["ws"])
    k = HIDDEN
    x_row, s_row = W.lds_row(k // 4), W.lds_row(k // 32)
    wbase = b.mul(ex, c(N_ROWS * (k // 2) // 4))  # dwords: expert e
    sblock = b.mul(b.add(b.mul(ex, c(N_ROWS // 32)), b.lshr(rg0, c(1))), c(SCALE_COLS * 8))
    rounds = _rounds(b, c, ks, slices)

    def loads(rnd):
        sc0, st0, n, sc_steps = rnd
        words = sc_steps // 2
        at = b.add(sblock, b.add(b.mul(sc0, c(32)), b.mul(lane, c(words))))
        sv = b.buffer_load_cp(r_ws, b.mul(at, c(4)), c(0), dwords=words, aux=W.CPOL_NT)
        avs = []
        for t in range(2):
            rg = b.add(rg0, c(t))
            tl = []
            for i in range(n):
                dw = b.add(wbase, b.add(b.mul(rg, c(2 * k)), b.add(b.mul(b.add(st0, c(i)), c(256)), b.mul(lane, c(4)))))
                tl.append(b.buffer_load_cp(r_w, b.mul(dw, c(4)), c(0), dwords=4, aux=knobs.w_cpol))
            avs.append(tl)
        return sv, avs

    def b_operand(st):
        xb = b.add(xl, b.add(b.mul(col, c(x_row)), b.add(b.mul(st, c(32)), b.mul(j, c(4)))))
        xlo = b.smem_load_vN(e_.pool, xb, dtype=I32, n=4)
        xhi = b.smem_load_vN(e_.pool, b.add(xb, c(16)), dtype=I32, n=4)
        sb = b.vec_extract(
            b.smem_load_vN(e_.pool, b.add(xsl, b.add(b.mul(col, c(s_row)), b.add(b.mul(st, c(4)), j))), dtype=I32, n=1), 0
        )
        return b.vec_concat(xlo, xhi), sb

    def lds_word(at):
        return b.vec_extract(b.smem_load_vN(e_.pool, at, dtype=I32, n=1), 0)

    accs = [b.vec_pack([b.const_f32(0.0)] * 4, F32) for _ in range(2)]
    # ``la``: the flattened K steps (round, step) and their pending B operands (W15 r0 fix; la=0: unused)
    flat = [(rr, ii) for rr, rd in enumerate(rounds) for ii in range(rd[2])]
    pend = {}

    def ensure(g):
        if la and g < len(flat) and g not in pend:
            rr, ii = flat[g]
            pend[g] = b_operand(b.add(rounds[rr][1], c(ii)))

    for g0 in range(la):
        ensure(g0)
    ops = [loads(rounds[q]) for q in range(min(ahead, len(rounds)))]
    for r, rnd in enumerate(rounds):
        sc0, st0, n, sc_steps = rnd
        words = sc_steps // 2
        sv, avs = ops[r]
        b.smem_store_vN(e_.pool, [b.add(sl, b.mul(lane, c(words)))], sv, words)
        sas = []
        for t in range(2):
            row = []
            for i in range(n):
                st = b.add(st0, c(i))
                # scale_index(t 16 + lane % 16, st 4 + j, 160) - 128 sc0 (r32 = 0, a = t, e = j, d = st % 2)
                idx = b.add(
                    b.mul(b.add(b.mul(b.add(b.mul(b.add(b.mul(b.lshr(st, c(1)), c(4)), j), c(16)), b.land(lane, c(15))),
                                      c(2)), b.land(st, c(1))), c(2)),
                    c(t),
                )  # fmt: skip
                idx = b.sub(idx, b.mul(sc0, c(128)))
                word = lds_word(b.add(sl, b.lshr(idx, c(2))))
                if knobs.sop:
                    osa = 2 * (i % 2) + t
                    if knobs.mutate == "opsel" and i == 1:
                        osa = (osa + 1) % 4
                    row.append((word, osa))
                else:
                    byte = b.land(idx, c(3))
                    if knobs.mutate == "scale_byte":
                        byte = b.land(b.add(byte, c(1)), c(3))
                    row.append((b.land(b.lshr(word, b.mul(byte, c(8))), c(0xFF)), 0))
            sas.append(row)
        bs = [b_operand(b.add(st0, c(i))) for i in range(n)] if knobs.ra else None
        for i in range(n):
            if la:
                g = flat.index((r, i))
                ensure(g + la)
                ensure(g)
                bv_la = pend.pop(g)
            if knobs.mutate == "chain_drop" and i == 1 and n > 1:
                continue
            bv, sb = bv_la if la else (bs[i] if bs is not None else b_operand(b.add(st0, c(i))))
            for t in range(2):
                sa, osa = sas[t][i]
                av = avs[t][i]
                accs[t] = b.mfma_scale_mx(av, bv, accs[t], sa, sb, shape="16x16x128", fmt_a="fp4", fmt_b="fp8",
                                          opsel_a=osa, opsel_b=0)  # fmt: skip
        if r + ahead < len(rounds):
            ops.append(loads(rounds[r + ahead]))

    # the partial C: red slot ((p 2 + bb) slices + ks) 2 + t, a 64-lane vec4 each
    for t in range(2):
        slot = b.add(b.mul(b.add(b.mul(b.add(b.mul(pw, c(2)), bb), c(slices)), ks), c(2)), c(t))
        at = b.add(red, b.mul(b.add(b.mul(slot, c(64)), lane), c(4)))
        b.smem_store_vN(e_.pool, [at], b.vec_pack([b.bitcast(b.vec_extract(accs[t], q), I32) for q in range(4)], I32), 4)
    e_.barrier()

    # ug_epilogue: thread idx = pass + tid of groups x rows x 32
    total = groups * rows * UG_PART
    r_mid, r_mids = rsrc_full(p["mid"]), rsrc_full(p["mids"])

    def pairwise(vs):
        while len(vs) > 1:
            vs = [b.fadd(vs[i], vs[i + 1]) for i in range(0, len(vs), 2)]
        return vs[0]

    def bfr(v):
        return b.cast_to_f32(b.trunc_f32_to_bf16(v))

    for pas in range(0, total, W.THREADS):
        idx = b.add(tid, c(pas))
        with b.scf_if(b.cmp_lt(idx, c(total))):
            pg = b.div(idx, c(rows * UG_PART))
            part = b.add(part0, pg)
            t = b.div(b.sub(idx, b.mul(pg, c(rows * UG_PART))), c(UG_PART))
            i = b.land(idx, c(UG_PART - 1))
            tile = b.lshr(i, c(4))
            rr = b.land(i, c(15))
            at = b.add(b.mul(b.add(b.mul(b.lshr(rr, c(2)), c(16)), t), c(4)), b.land(rr, c(3)))
            gu = []
            for tt in range(2):
                vals = []
                for kk in range(slices):
                    slot = b.add(b.mul(b.add(b.mul(b.add(b.mul(pg, c(2)), tile), c(slices)), c(kk)), c(2)), c(tt))
                    vals.append(b.bitcast(lds_word(b.add(red, b.add(b.mul(slot, c(256)), at))), F32))
                gu.append(pairwise(vals))
            g, up = gu
            gc = b.fmin(g, b.const_f32(LIMIT))
            uc = b.fmax(b.fmin(up, b.const_f32(LIMIT)), b.const_f32(-LIMIT))
            sig = b.rcp_fast(b.fadd(b.const_f32(1.0), b.exp2_fast(b.fmul(gc, b.const_f32(-LOG2E)))))
            v = bfr(b.fmul(b.fmul(gc, sig), uc))
            a = b.fabs(v)
            for off in (1, 2, 4, 8, 16):
                a = b.fmax(a, b.bitcast(b.ds_swizzle_xor(b.bitcast(a, I32), off), F32))
            amax = b.fmax(a, b.const_f32(1e-10))
            bits = b.bitcast(b.fmul(amax, b.const_f32(1.0 / FP8_MAX)), I32)
            ex8 = b.add(b.land(b.lshr(bits, c(23)), c(0xFF)),
                        b.select(b.cmp_ne(b.land(bits, c(0x7FFFFF)), c(0)), c(1), c(0)))  # fmt: skip
            code = b.smax(b.smin(ex8, c(254)), c(1))
            inv = b.bitcast(b.shl(b.sub(c(254), code), c(23)), F32)
            q8 = b.fmax(b.fmin(b.fmul(v, inv), b.const_f32(FP8_MAX)), b.const_f32(-FP8_MAX))
            q8i = b.bitcast(q8, I32)
            nb = [b.bitcast(b.ds_bpermute(b.mul(b.smin(b.add(lane, c(q)), c(63)), c(4)), q8i), F32) for q in (1, 2, 3)]
            word = b.bitcast(b.cvt_pk_fp8_f32x4(b.vec_pack([q8, *nb], F32)), I32)
            if knobs.mutate == "put_bit":
                word = b.xor(word, c(1))
            row = first if knobs.mutate == "row" else b.add(first, t)
            live = b.cmp_lt(t, count)
            with b.scf_if(live):
                with b.scf_if(b.cmp_eq(b.land(i, c(3)), c(0))):
                    dw = b.add(b.mul(row, c(MID_WORDS)), b.add(b.mul(part, c(8)), b.lshr(i, c(2))))
                    b.buffer_store_cp(word, r_mid, b.mul(dw, c(4)), c(0), aux=W.CPOL_SC1)
                with b.scf_if(b.cmp_eq(i, c(0))):
                    ci = b.add(b.mul(row, c(MID_CODES)),
                               b.add(b.mul(b.land(part, c(3)), c(MOE_INTER // 128)), b.lshr(part, c(2))))  # fmt: skip
                    b.buffer_store_cp(code, r_mids, b.mul(ci, c(4)), c(0), aux=W.CPOL_SC1)
    b.ret()
    return b.kernel


def lower_ug_function(rows: int, groups: int, knobs: W.WqkvKnobs = W.WqkvKnobs(), *, arch="gfx950", flavor=None):
    from ..qwen38_mono.device_function import DeviceFunctionKnobs, lower_device_function

    return lower_device_function(
        build_ug_kernel(rows, groups, knobs), arch=arch, symbol=symbol(rows, groups), llvm_flavor=flavor,
        knobs=DeviceFunctionKnobs(lds="arg", inline="always"),
    )  # fmt: skip


def write_ug_functions(out: str, rows_list, groups_list=(1, 2), knobs: W.WqkvKnobs = W.WqkvKnobs(), *,
                       arch="gfx950", flavor=None, bitcode="auto") -> list:  # fmt: skip
    from ..qwen38_mono.device_function import write_device_function

    written = []
    for rows in rows_list:
        for groups in groups_list:
            fn = lower_ug_function(rows, groups, knobs, arch=arch, flavor=flavor)
            paths = write_device_function(fn, out, stem=f"{STAGE}_{variant(rows, groups)}", bitcode=bitcode)
            with open(paths["json"]) as f:
                meta = json.load(f)
            meta.update(stage=STAGE, rows=rows, groups=groups, stage_knobs=knobs.spec(), abi_v41=[list(a) for a in ABI])
            with open(paths["json"], "w", newline="\n") as f:
                json.dump(meta, f, indent=1)
                f.write("\n")
            written.append({"rows": rows, "groups": groups, **paths})
    return written


def check_atom(atom_root: str | None = None) -> None:
    """Compare the copied constants with an ATOM tree (``moe.py`` / ``dims.py`` / ``moe_shape.py``)."""
    if atom_root:
        sys.path.insert(0, atom_root)
    from atom.models.deepseek_v41.mono.kernels import dims, moe

    d = dims.Dims(2)
    got = {
        "HIDDEN": moe.HIDDEN, "MOE_INTER_REAL": d.inter_real, "MOE_INTER": d.inter, "MID_WORDS": d.mid_words,
        "DOWN_SCALE_COLS": d.down_scale_cols, "WSL_WORDS": moe.wsl_words(d), "UG_PART": dims.UG_PART,
        "LIMIT": moe.LIMIT, "LOG2E": moe.LOG2E,
    }  # fmt: skip
    for name, v in got.items():
        if globals()[name] != v:
            raise AssertionError(f"stage_ug.{name} = {globals()[name]!r}, ATOM {v!r}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="emit the rocKE V4.1 routed ug stage functions")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", default=",".join(str(r) for r in range(1, 17)))
    ap.add_argument("--groups", default="1,2")
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
    groups = [int(g) for g in a.groups.split(",") if g]
    bc = {"auto": "auto", "yes": True, "no": False}[a.bitcode]
    out = write_ug_functions(a.out, rows, groups, kn, arch=a.arch, flavor=a.flavor, bitcode=bc)
    print(json.dumps({"knobs": kn.spec(), "written": len(out)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
