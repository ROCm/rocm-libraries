# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""K1 ``wqkv_a`` of ATOM's DeepSeek-V4.1 mono as a rocKE stage function (hybrid path).

Bit-identical re-implementation of ATOM ``atom/models/deepseek_v41/mono/kernels/attn_pre.py`` (origin/main
df86c482): ``gemv_fp8_loads`` + ``load_x8`` + ``stage_wqkv`` for one CTA's task of the one-token-tile build
(``own_n`` = this build's rows <= 16), i.e. rows ``16 task ..`` of the (16, 16)-preshuffled FP8 ``wqkv_a`` (E8M0
per 32x32 block) times every token's MXFP8 normed row (mailboxes ``x8`` / ``x8s``), summed over the eight waves'
16x16 accumulators in wave order, rounded to bf16 and published as ``(f32 bits, tag)`` pairs (mailbox ``qkv``).

Structure (the FlyDSL stage's, in its order):

1. weights in flight first: wave ``w`` takes K steps ``w, w + 8, ..`` (5 of the 40 128-k steps), two ``dwordx4``
   nontemporal loads and one scale dword a step (``_fp8_step_load``);
2. the token rows' MXFP8 words and E8M0 codes polled from the mailbox into LDS (``poll_copy``: thread ``tid`` takes
   elements ``tid, tid + 512, ..``; ``POLL_MAX`` pairs a round trip, each batch stored before the next is polled; rows
   padded by 4 words, ``lds_row``), then a CTA barrier;
3. the five-step scaled-MFMA chain (``v_mfma_scale_f32_16x16x128_f8f6f4``, fp8 x fp8, weights as A, the token rows
   as B) on one accumulator, stored to ``red[wave][lane]``; barrier;
4. thread ``tid < 16 rows``: row ``tid % 16``, token ``tid / 16``: ``0.0 + red[w0] + .. + red[w7]`` (wave order),
   bf16 round, ``put`` at pair ``t * 1792 + 16 task + r``; barrier.

Every floating-point operation is the mono's, in its order, without fast-math flags (the function is linked into
the mono's module and compiled by the same LLVM). The MFMA operands are the same integers.

ABI (``rocke_dsv41_wqkv_n<rows>``; ``lds="arg"``, ``lds_base`` the byte address the three LDS word offsets are
relative to)::

    void (i32 tag, i32 task, i64 w, i64 ws, i64 x8, i64 x8s, i64 qkv,
          i32 xl, i32 xsl, i32 red, i32 lds_base)

``tag`` = the launch's ``layer + 1``; ``task`` = this CTA's wqkv task (< 112); ``x8`` / ``x8s`` / ``qkv`` the mailbox
regions' addresses (scratch + offset); ``xl`` / ``xsl`` / ``red`` the LDS word offsets of the stage's buffers.

Knobs (:class:`WqkvKnobs`; defaults = the FlyDSL stage; every knob keeps the numerics bit for bit unless it is a
test mutant): ``poll`` (``ir``: the compiler-visible ``tile.tagged_poll`` with the reload fix, buffer loads ``sc1``,
the FlyDSL poll's shape; ``asm``: rocKE's asm poll, ``global_load_dwordx2 sc1``), ``pause`` (retry: -1 none, 0
``s_nop 0``, N ``s_sleep N``; FlyDSL: ``s_sleep 1``), ``poll_max`` (pairs a poll batch; FlyDSL 12), ``w_cpol``
(weight load cache bits; FlyDSL 2 = nt), ``x_first`` (poll the token rows BEFORE issuing the weights: a must-lose
order control), ``red_bar`` (``wg``: FlyDSL's fenced workgroup barrier; ``lds``: ``lgkmcnt(0)`` + barrier for the
two barriers that only order LDS), ``sop`` (0: a scale VGPR made right before each scaled MFMA, the FlyDSL form,
which P-L3 measured as a slow MFMA pattern on gfx950; 1: the weight scales of four K steps packed into one dword
before the chain, picked with ``op_sel``; 2: the token scales too), ``ra`` (1: every step's LDS operands read before
the first MFMA), ``mutate`` (test only, :data:`MUTANTS`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field, fields

from rocke.core.ir import F32, I32, I64, IRBuilder, PtrType, VectorType

# ATOM ``dims`` / ``attn_pre`` constants (read-only copies; ``check_atom`` compares them with an ATOM tree).
HIDDEN = 5120
Q_RANK, KV_DIM = 1280, 512
QKV_ROWS = Q_RANK + KV_DIM  # 1792
ROWS = 16
TILE = 16
WQKV_TASKS = QKV_ROWS // ROWS  # 112
X8_WORDS = HIDDEN // 4  # 1280
X8_GROUPS = HIDDEN // 32  # 160
LDS_PAD = 4
WAVES, THREADS = 8, 512
POLL_MAX = 12
CPOL_NT, CPOL_SC1 = 2, 16
SYMBOL_PREFIX = "rocke_dsv41_"
STAGE = "wqkv"
POOL_WORDS = 160 * 1024 // 4  # the whole LDS (lds="arg": nothing is allocated)

# Test-only mutants: each must make the in-situ bitwise A/B differ.
#   put_bit:    every published payload's lowest bit flipped
#   chain_drop: each wave's chain leaves out its step 1
#   row:        tokens > 0 publish token 0's row (rows > 1)
#   scale_byte: the weight scale's byte index off by one (wsi % 4 + 1)
#   wave_order: the wave sum in order 1, 0, 2, .. 7 (an fp32 reassociation: rare bf16 effect; report-only)
#   opsel:      with sop >= 1, step 1 picks the next packed weight-scale byte
MUTANTS = ("put_bit", "chain_drop", "row", "scale_byte", "wave_order", "opsel")
REPORT_ONLY = ("wave_order",)


@dataclass(frozen=True)
class WqkvKnobs:
    poll: str = "ir"
    pause: int = 1
    poll_max: int = POLL_MAX
    w_cpol: int = CPOL_NT
    x_first: bool = False
    red_bar: str = "wg"
    sop: int = 0
    ra: int = 0
    mutate: str = ""

    def __post_init__(self) -> None:
        if self.poll not in ("ir", "asm"):
            raise ValueError("poll must be ir or asm")
        if not 1 <= self.poll_max <= 16:
            raise ValueError("poll_max must be in 1..16")
        if self.red_bar not in ("wg", "lds"):
            raise ValueError("red_bar must be wg or lds")
        if self.sop not in (0, 1, 2):
            raise ValueError("sop must be 0 (per-step scales), 1 (A packed) or 2 (A and B packed)")
        if self.ra not in (0, 1):
            raise ValueError("ra must be 0 or 1")
        if self.mutate and self.mutate not in MUTANTS:
            raise ValueError(f"mutate must be one of {MUTANTS}")

    def spec(self) -> str:
        """Non-default knobs as ``k=v,..`` (empty = the FlyDSL stage)."""
        d = WqkvKnobs()
        return ",".join(f"{f.name}={getattr(self, f.name)}" for f in fields(self) if getattr(self, f.name) != getattr(d, f.name))


def knobs_from_spec(spec: str) -> WqkvKnobs:
    kw = {}
    types = {f.name: f.type for f in fields(WqkvKnobs)}
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        k, _, v = part.partition("=")
        if k not in types:
            raise ValueError(f"unknown wqkv knob {k!r}")
        t = types[k]
        kw[k] = (v in ("1", "true", "True")) if t in ("bool", bool) else (int(v) if t in ("int", int) else v)
    return WqkvKnobs(**kw)


def lds_row(words: int) -> int:
    return words + LDS_PAD


def symbol(rows: int) -> str:
    return f"{SYMBOL_PREFIX}{STAGE}_n{rows}"


ABI = (
    ("tag", "i32"), ("task", "i32"), ("w", "i64"), ("ws", "i64"), ("x8", "i64"), ("x8s", "i64"),
    ("qkv", "i64"), ("xl", "i32"), ("xsl", "i32"), ("red", "i32"),
)  # fmt: skip


class _E:
    def __init__(self, b: IRBuilder, k: WqkvKnobs, rows: int):
        self.b, self.k, self.rows = b, k, rows
        self.pool = b.smem_alloc(I32, [POOL_WORDS], name_hint="mono_lds")

    def c(self, v):
        return self.b.const_i32(int(v))

    def rsrc(self, addr):
        b = self.b
        ptr = b.int_to_ptr(b.readfirstlane(addr), PtrType(I32, "global"))
        return b.buffer_rsrc(ptr, self.c(0x7FFFFFF0))

    def barrier(self, lds_only=False):
        if lds_only and self.k.red_bar == "lds":
            self.b.sync_lds_only(fence="compiler")
        else:
            self.b.s_barrier_bare(fence="workgroup")  # FlyDSL gpu.barrier: release fence, s_barrier, acquire fence

    def fresh(self, v):
        """ATOM ``ops.fresh``: an opaque ``v_mov_b32`` (indices not hoisted / CSEd)."""
        return self.b.inline_asm("v_mov_b32 $0, $1", "=v,v", [v], I32, sideeffect=True, result_name_hint="fresh")

    def pause_text(self) -> str:
        p = self.k.pause
        return "" if p < 0 else ("s_nop 0" if p == 0 else f"s_sleep {p}")

    def poll(self, tag, specs):
        """``specs`` = [(rsrc or i64 addr, pair index)] -> payload words (one pair each), one round trip."""
        b = self.b
        if self.k.poll == "ir":
            tspecs = [(rs, b.mul(i, self.c(8)), 2) for rs, _, i in specs]
            vecs = b.tagged_poll(tag, tspecs, aux=CPOL_SC1, pause=self.pause_text(), reload="launder")
            return [b.vec_extract(v, 0) for v in vecs]
        from rocke.helpers import mailbox as M

        pause = {-1: "none", 0: "s_nop", 1: "s_sleep1"}.get(self.k.pause)
        if pause is None:
            raise ValueError("poll=asm takes pause -1, 0 or 1")
        kn = M.MailboxKnobs(pause=pause, poll_max=self.k.poll_max)
        mspecs = [(M.Mailbox(addr=addr), i, 1) for _, addr, i in specs]
        return [w[0] for w in M.poll(b, mspecs, tag, kn)]


def build_wqkv_kernel(rows: int, knobs: WqkvKnobs = WqkvKnobs()):
    """The stage function's ``KernelDef`` for a build of ``rows`` tokens (one tile)."""
    if not 1 <= rows <= TILE:
        raise ValueError(f"rows {rows}: the one-tile build has 1..{TILE}")
    if knobs.mutate == "row" and rows < 2:
        raise ValueError("mutant row needs rows > 1")
    b = IRBuilder(symbol(rows))
    b.kernel.attrs["max_workgroup_size"] = THREADS
    b.fence_barriers = "compiler"
    p = {n: b.param(n, I32 if t == "i32" else I64) for n, t in ABI}
    e = _E(b, knobs, rows)
    c = e.c
    tag = b.readfirstlane(p["tag"])
    task = b.readfirstlane(p["task"])
    xl, xsl, red = (b.readfirstlane(p[n]) for n in ("xl", "xsl", "red"))
    tid = b.thread_id_x()
    lane = b.land(tid, c(63))
    wave = b.lshr(tid, c(6))
    j = b.lshr(lane, c(4))
    k = HIDDEN
    kg = k // 32
    steps = k // 128  # 40
    per = steps // WAVES  # 5

    def weights():
        r_w, r_ws = e.rsrc(p["w"]), e.rsrc(p["ws"])
        ops = []
        for i in range(per):
            st = b.add(wave, c(WAVES * i))
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

    def load_x8():
        idx = e.fresh(tid)
        x8 = b.readfirstlane(p["x8"])
        x8s = b.readfirstlane(p["x8s"])
        r8, r8s = e.rsrc(x8), e.rsrc(x8s)
        spans = [(r8, x8, rows * X8_WORDS, xl, X8_WORDS), (r8s, x8s, rows * X8_GROUPS, xsl, X8_GROUPS)]
        chunks = [
            (rs, addr, n, dst, row, b.add(idx, c(THREADS * i)))
            for rs, addr, n, dst, row in spans
            for i in range((n + THREADS - 1) // THREADS)
        ]
        for b0 in range(0, len(chunks), knobs.poll_max):
            batch = chunks[b0 : b0 + knobs.poll_max]
            got = e.poll(tag, [(rs, addr, b.smin(u, c(n - 1))) for rs, addr, n, _, _, u in batch])
            for (_, _, n, dst, row, u), v in zip(batch, got):
                # lds_at(u, row) = u + u // row * LDS_PAD (u >= 0)
                at = b.add(u, b.mul(b.div(u, c(row)), c(LDS_PAD)))
                with b.scf_if(b.cmp_lt(u, c(n))):
                    b.smem_store_vN(e.pool, [b.add(dst, at)], v, 1)
        e.barrier()

    if knobs.x_first:
        load_x8()
        ops = weights()
    else:
        ops = weights()
        load_x8()

    # the chain: wave w, steps w + 8 i, one accumulator
    x_row, s_row = lds_row(k // 4), lds_row(kg)
    col = b.smin(b.land(lane, c(15)), c(rows - 1))
    acc = b.vec_pack([b.const_f32(0.0)] * 4, F32)

    def x_operands(st):
        xb = b.add(xl, b.add(b.mul(col, c(x_row)), b.add(b.mul(st, c(32)), b.mul(j, c(4)))))
        xlo = b.smem_load_vN(e.pool, xb, dtype=I32, n=4)
        xhi = b.smem_load_vN(e.pool, b.add(xb, c(16)), dtype=I32, n=4)
        xv = b.vec_concat(xlo, xhi)
        grp = b.add(b.mul(st, c(4)), j)
        sb = b.vec_extract(b.smem_load_vN(e.pool, b.add(xsl, b.add(b.mul(col, c(s_row)), grp)), dtype=I32, n=1), 0)
        return xv, sb

    def pack(vals):
        # sop: the codes (bytes 0..255) of up to four steps, byte q = step q; op_sel q picks it
        out = vals[0]
        for q in range(1, len(vals)):
            out = b.lor(out, b.shl(vals[q], c(8 * q)))
        return out

    xs = [x_operands(st) for st, *_ in ops] if knobs.ra else None
    if knobs.sop:
        sas = [sa for _st, _lo, _hi, sa in ops]
        packed_a = [pack(sas[q0 : q0 + 4]) for q0 in range(0, len(sas), 4)]
    if knobs.sop == 2:
        if xs is None:
            xs = [x_operands(st) for st, *_ in ops]
        sbs = [sb for _xv, sb in xs]
        packed_b = [pack(sbs[q0 : q0 + 4]) for q0 in range(0, len(sbs), 4)]
    for i, (st, lo, hi, sa) in enumerate(ops):
        if knobs.mutate == "chain_drop" and i == 1:
            continue
        av = b.vec_concat(lo, hi)
        xv, sb = xs[i] if xs is not None else x_operands(st)
        osa = osb = 0
        if knobs.sop:
            sa, osa = packed_a[i // 4], i % 4
        if knobs.sop == 2:
            sb, osb = packed_b[i // 4], i % 4
        if knobs.mutate == "opsel" and knobs.sop and i == 1:
            osa = (osa + 1) % 4
        acc = b.mfma_scale_mx(av, xv, acc, sa, sb, shape="16x16x128", fmt_a="fp8", fmt_b="fp8", opsel_a=osa, opsel_b=osb)
    red_at = b.add(red, b.mul(b.add(b.mul(wave, c(64)), lane), c(4)))
    accw = b.vec_pack([b.bitcast(b.vec_extract(acc, q), I32) for q in range(4)], I32)
    b.smem_store_vN(e.pool, [red_at], accw, 4)
    e.barrier(lds_only=True)
    with b.scf_if(b.cmp_lt(tid, c(ROWS * rows))):
        r = b.land(tid, c(ROWS - 1))
        t = b.lshr(tid, c(4))
        tr = c(0) if knobs.mutate == "row" else t
        # row_sum(red, r, t): red[(w * 64 + 16 (r / 4) + t) * 4 + r % 4], f32 sum from +0.0 in wave order
        base = b.add(b.mul(b.add(b.mul(b.lshr(r, c(2)), c(16)), tr), c(4)), b.land(r, c(3)))
        tot = b.const_f32(0.0)
        order = [1, 0, 2, 3, 4, 5, 6, 7] if knobs.mutate == "wave_order" else list(range(WAVES))
        for w in order:
            v = b.vec_extract(b.smem_load_vN(e.pool, b.add(red, b.add(base, c(w * 64 * 4))), dtype=F32, n=1), 0)
            tot = b.fadd(tot, v)
        out = b.cast_to_f32(b.trunc_f32_to_bf16(tot))
        bits = b.bitcast(out, I32)
        if knobs.mutate == "put_bit":
            bits = b.xor(bits, c(1))
        i_pair = b.add(b.mul(t, c(QKV_ROWS)), b.add(b.mul(task, c(ROWS)), r))
        r_q = e.rsrc(p["qkv"])
        b.buffer_store_cp(b.vec_pack([bits, tag], I32), r_q, b.mul(i_pair, c(8)), c(0), aux=CPOL_SC1)
    e.barrier(lds_only=True)
    b.ret()
    return b.kernel


def lower_wqkv_function(rows: int, knobs: WqkvKnobs = WqkvKnobs(), *, arch="gfx950", flavor=None):
    from ..qwen38_mono.device_function import DeviceFunctionKnobs, lower_device_function

    return lower_device_function(
        build_wqkv_kernel(rows, knobs), arch=arch, symbol=symbol(rows), llvm_flavor=flavor,
        knobs=DeviceFunctionKnobs(lds="arg", inline="always"),
    )  # fmt: skip


def write_wqkv_functions(out: str, rows_list, knobs: WqkvKnobs = WqkvKnobs(), *, arch="gfx950", flavor=None,
                         bitcode="auto") -> list:  # fmt: skip
    from ..qwen38_mono.device_function import write_device_function

    written = []
    for rows in rows_list:
        fn = lower_wqkv_function(rows, knobs, arch=arch, flavor=flavor)
        paths = write_device_function(fn, out, stem=f"{STAGE}_n{rows}", bitcode=bitcode)
        js = paths["json"]
        with open(js) as f:
            meta = json.load(f)
        meta["stage"] = STAGE
        meta["rows"] = rows
        meta["stage_knobs"] = knobs.spec()
        meta["abi_v41"] = [list(a) for a in ABI]
        with open(js, "w", newline="\n") as f:
            json.dump(meta, f, indent=1)
            f.write("\n")
        written.append({"rows": rows, **paths})
    return written


def check_atom(atom_root: str | None = None) -> None:
    """The constants above against an ATOM tree's ``dims`` / ``attn_pre`` (raises on a difference)."""
    if atom_root:
        sys.path.insert(0, atom_root)
    from atom.models.deepseek_v41.mono.kernels import attn_pre as k1
    from atom.models.deepseek_v41.mono.kernels import dims as D

    got = {
        "HIDDEN": D.HIDDEN, "QKV_ROWS": k1.QKV_ROWS, "ROWS": k1.ROWS, "TILE": k1.TILE,
        "WQKV_TASKS": k1.WQKV_TASKS, "X8_WORDS": k1.X8_WORDS, "LDS_PAD": k1.LDS_PAD, "POLL_MAX": k1.POLL_MAX,
    }  # fmt: skip
    want = {
        "HIDDEN": HIDDEN, "QKV_ROWS": QKV_ROWS, "ROWS": ROWS, "TILE": TILE, "WQKV_TASKS": WQKV_TASKS,
        "X8_WORDS": X8_WORDS, "LDS_PAD": LDS_PAD, "POLL_MAX": POLL_MAX,
    }  # fmt: skip
    bad = {k: (got[k], want[k]) for k in want if got[k] != want[k]}
    if bad:
        raise ValueError(f"ATOM constants differ (atom, rocke): {bad}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="emit the rocKE V4.1 wqkv_a stage functions")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16")
    ap.add_argument("--knobs", default="")
    ap.add_argument("--flavor", default=None)
    ap.add_argument("--arch", default="gfx950")
    ap.add_argument("--bitcode", choices=("auto", "yes", "no"), default="no")
    ap.add_argument("--atom-root", default=None, help="check the constants against this ATOM tree")
    a = ap.parse_args(argv)
    if a.atom_root:
        check_atom(a.atom_root)
    kn = knobs_from_spec(a.knobs)
    rows = [int(r) for r in a.rows.split(",") if r]
    if kn.mutate == "row":
        rows = [r for r in rows if r > 1]
    bc = {"auto": "auto", "yes": True, "no": False}[a.bitcode]
    out = write_wqkv_functions(a.out, rows, kn, arch=a.arch, flavor=a.flavor, bitcode=bc)
    print(json.dumps({"knobs": kn.spec(), "written": out}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
