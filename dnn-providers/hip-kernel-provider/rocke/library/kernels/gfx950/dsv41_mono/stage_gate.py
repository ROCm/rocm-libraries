# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The mHC seam gate (``stage_gate``) of ATOM's DeepSeek-V4.1 mono as a rocKE stage function (W15 PORT K2-attn-gemv).

Bit-identical re-implementation of ATOM ``attn_pre.py`` (origin/main df86c482) ``stage_gate(c, t)``, which K1 (the
attention seam's gate, ``k1.gate``) and K2a (``run_gates``, the FFN seam's gate) both run: one CTA a token t.

1. threads ``tid < 20 x 25``: group g = tid / 25, mix j = tid % 25 (24 mixes + the sum of squares) polls the 8 slices'
   ``pmix`` pairs ``((8 g + k) S + t) 25 + j`` (one batch, ``sc1``, ``s_sleep 1``), sums them in slice order
   (``acc = v0; acc += v1 .. v7``) into ``red[g 25 + j]``; barrier;
2. wave 0: lane -> its mix (pre / post on lanes 0..7: j = lane % 8; comb from lane 16: j = min(lane - 8, 23)); the 20
   group sums in order, then ``v = mix rsq(fma(sq, 1/K, eps))``; lanes 0..3 ``pre = sigmoid(fma(v, s0, base)) + 1e-6``,
   lanes 4..7 ``post = 2 sigmoid(fma(v, s1, base))``; ``comb`` = Sinkhorn on lanes 16..31 (row sums lanes xor 1, 2,
   column sums xor 4, 8): ``exp2((cv - max) log2 e)``, ``fma(cv, rcp(row), eps)``, ``cv rcp(col + eps)``, then 19 x
   (row, col) normalizations; f32 stores; barrier.

ABI (``rocke_dsv41_gate_n<S>``, ``lds="arg"``; S = the rows the stage sees, a token-parallel build's own rows)::

    void (i32 tag, i32 t, i64 pmix, i64 hc_scale, i64 hc_base, i64 pre_out, i64 post_out, i64 comb_out,
          i32 red, i32 lds_base)

Knobs (:class:`GateKnobs`): the :class:`seam_common.SeamKnobs` set plus ``early`` (1: wave 0 issues the ``hc_scale``
/ ``hc_base`` loads at entry, before the poll; FlyDSL 0: after the barrier). Mutants: ``put_bit`` (pre's low bit),
``sink`` (19 Sinkhorn rounds), ``slice_drop`` (the group sum leaves slice 7 out), ``row`` (comb of token t -> t's
pairs read from token 0), ``order`` (slice sum reversed: an f32 reassociation, report-only).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass

from rocke.core.ir import F32, I32, I64, IRBuilder

from . import seam_common as S

STAGE = "gate"
MUTANTS = ("put_bit", "sink", "slice_drop", "row", "order")
REPORT_ONLY = ("order",)
ABI = (
    ("tag", "i32"), ("t", "i32"), ("pmix", "i64"), ("hc_scale", "i64"), ("hc_base", "i64"), ("pre_out", "i64"),
    ("post_out", "i64"), ("comb_out", "i64"), ("red", "i32"),
)  # fmt: skip
GROUPS = S.SLICES // S.GATE_BLOCK  # 20


@dataclass(frozen=True)
class GateKnobs(S.SeamKnobs):
    early: int = 0

    def __post_init__(self) -> None:
        self.check(MUTANTS)
        if self.early not in (0, 1):
            raise ValueError("early must be 0 or 1")


def symbol(rows: int) -> str:
    return f"rocke_dsv41_{STAGE}_n{rows}"


def build_gate_kernel(rows: int, knobs: GateKnobs = GateKnobs()):
    if not 1 <= rows <= 48:
        raise ValueError(f"rows {rows}: 1..48")
    b = IRBuilder(symbol(rows))
    b.kernel.attrs["max_workgroup_size"] = S.THREADS
    b.fence_barriers = "compiler"
    p = {n: b.param(n, I32 if t == "i32" else I64) for n, t in ABI}
    e = S.SeamEnv(b, knobs)
    c = e.c
    tag = b.readfirstlane(p["tag"])
    t = b.readfirstlane(p["t"])
    red = b.readfirstlane(p["red"])
    tid, lane, wave = e.tid, e.lane, e.wave
    w0 = b.cmp_eq(wave, c(0))
    # lane -> its mix (pre / post on lanes 0 .. 7, comb from lane COMB0)
    j_lane = b.select(b.cmp_lt(lane, c(S.COMB0)), b.land(lane, c(2 * S.HC - 1)),
                      b.smin(b.add(b.sub(lane, c(S.COMB0)), c(2 * S.HC)), c(S.MIX - 1)))  # fmt: skip

    def scales():
        sc = [e.ld_f32(p["hc_scale"], c(i)) for i in range(3)]
        return sc, e.ld_f32(p["hc_base"], j_lane)

    early = scales() if knobs.early else None
    with b.scf_if(b.cmp_lt(tid, c(GROUPS * S.PART))):
        g = b.div(tid, c(S.PART))
        j = b.mod(tid, c(S.PART))
        r_pm = e.rsrc(p["pmix"])
        tt = c(0) if knobs.mutate == "row" else t
        specs = []
        for k in range(S.GATE_BLOCK):
            i = b.add(b.mul(b.add(b.mul(b.add(b.mul(g, c(S.GATE_BLOCK)), c(k)), c(rows)), tt), c(S.PART)), j)
            specs.append((r_pm, b.readfirstlane(p["pmix"]), i))
        got = e.poll(tag, specs)
        vals = [b.bitcast(v, F32) for v in got]
        ks = list(range(S.GATE_BLOCK))
        if knobs.mutate == "order":
            ks = ks[::-1]
        if knobs.mutate == "slice_drop":
            ks = ks[:-1]
        acc = vals[ks[0]]
        for k in ks[1:]:
            acc = b.fadd(acc, vals[k])
        e.lds_st_f32(b.add(red, b.add(b.mul(g, c(S.PART)), j)), acc)
    e.barrier(lds_only=True)
    with b.scf_if(w0):
        mix = e.lds_f32(b.add(red, j_lane))
        sq = e.lds_f32(b.add(red, c(S.MIX)))
        for g in range(1, GROUPS):
            mix = b.fadd(mix, e.lds_f32(b.add(red, b.add(c(g * S.PART), j_lane))))
            sq = b.fadd(sq, e.lds_f32(b.add(red, c(g * S.PART + S.MIX))))
        sc, base = early if early is not None else scales()
        v = b.fmul(mix, b.rsqrt(e.fma(sq, e.f(1.0 / S.K_MIX), e.f(S.EPS))))
        with b.scf_if(b.cmp_lt(lane, c(S.HC))):
            pre = b.fadd(e.sigmoid(e.fma(v, sc[0], base)), e.f(S.HC_EPS))
            if knobs.mutate == "put_bit":
                pre = b.bitcast(b.xor(b.bitcast(pre, I32), c(1)), F32)
            e.st_f32(p["pre_out"], b.add(b.mul(t, c(S.HC)), lane), pre)
        with b.scf_if(b.land(b.cmp_ge(lane, c(S.HC)), b.cmp_lt(lane, c(2 * S.HC)))):
            post = b.fmul(e.sigmoid(e.fma(v, sc[1], base)), e.f(S.POST_MULT))
            e.st_f32(p["post_out"], b.add(b.mul(t, c(S.HC)), b.sub(lane, c(S.HC))), post)
        cv = e.fma(v, sc[2], base)
        m = e.butterfly(cv, (1, 2), b.fmax)
        cv = b.exp2_fast(b.fmul(b.fsub(cv, m), e.f(S.LOG2E)))

        def rsum(x):
            return e.butterfly(x, (1, 2))

        def csum(x):
            return e.butterfly(x, (4, 8))

        cv = e.fma(cv, b.rcp_fast(rsum(cv)), e.f(S.HC_EPS))
        cv = b.fmul(cv, b.rcp_fast(b.fadd(csum(cv), e.f(S.HC_EPS))))
        rounds = S.SINKHORN - 1 - (1 if knobs.mutate == "sink" else 0)
        for _ in range(rounds):
            cv = b.fmul(cv, b.rcp_fast(b.fadd(rsum(cv), e.f(S.HC_EPS))))
            cv = b.fmul(cv, b.rcp_fast(b.fadd(csum(cv), e.f(S.HC_EPS))))
        with b.scf_if(b.land(b.cmp_ge(lane, c(S.COMB0)), b.cmp_lt(lane, c(S.COMB0 + S.HC * S.HC)))):
            e.st_f32(p["comb_out"], b.add(b.mul(t, c(S.HC * S.HC)), b.sub(lane, c(S.COMB0))), cv)
    e.barrier()
    b.ret()
    return b.kernel


def write_gate_functions(out: str, rows_list, knobs: GateKnobs = GateKnobs(), **kw) -> list:
    return S.write_functions(lambda v: build_gate_kernel(int(v[1:]), knobs), out, [f"n{r}" for r in rows_list],
                             STAGE, ABI, knobs, **kw)  # fmt: skip


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="emit the rocKE V4.1 seam gate stage functions")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", default=",".join(str(r) for r in range(1, 25)))
    ap.add_argument("--knobs", default="")
    ap.add_argument("--flavor", default=None)
    ap.add_argument("--arch", default="gfx950")
    ap.add_argument("--bitcode", choices=("auto", "yes", "no"), default="no")
    ap.add_argument("--atom-root", default=None)
    a = ap.parse_args(argv)
    if a.atom_root:
        S.check_atom(a.atom_root)
    kn = S.knobs_from(GateKnobs, a.knobs)
    rows = [int(r) for r in a.rows.split(",") if r]
    if kn.mutate == "row":
        rows = [r for r in rows if r > 1]
    bc = {"auto": "auto", "yes": True, "no": False}[a.bitcode]
    out = write_gate_functions(a.out, rows, kn, arch=a.arch, flavor=a.flavor, bitcode=bc)
    print(json.dumps({"knobs": S.spec_of(kn), "written": out}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
