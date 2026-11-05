# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""K2b's router GEMV (``stage_router``) of ATOM's DeepSeek-V4.1 mono as a rocKE stage function (W15 PORT
K2-attn-gemv).

Bit-identical re-implementation of ATOM ``moe.py`` (origin/main df86c482) ``run_moe``'s router block for one task:
``await_x`` (every token's ``xrdy`` pair, in a launch that writes x) and ``stage_router`` with the shared tile of
``atom.model_ops.deepseek_v41.router`` (``router_weight_loads`` / ``router_x_loads`` / ``router_tile_mfma`` /
``row_sum``): task = (token group, 16-row tile, K part of 4); wave w takes the part's K steps w, w + 8, .. (5 steps of
32), the 16 gate rows (bf16, ``nt``) times the group's tokens (bf16 x, device scope) with ``v_mfma_f32_16x16x32_bf16``
on one accumulator, the eight wave partials summed in wave order -> the part's f32 logit put to ``logit`` (``route``
sums the 4 parts in order and rounds to bf16). From 12 rows the weights are issued before x's wait (ATOM's
``ROUTER_EARLY_FROM``); below, after it. Past one group (17-48 rows: 2 groups of ceil(S/2) tokens) the group's token
tiles are traced.

ABI (``rocke_dsv41_router_n<S>e<E>``, ``lds="arg"``)::

    void (i32 tag, i32 task, i64 x, i64 gate_w, i64 xrdy, i64 logit, i32 red, i32 lds_base)

Knobs (:class:`RouterKnobs`): :class:`seam_common.SeamKnobs` plus ``early_from`` (rows from which the weights go out
before x's wait; ATOM 12), ``w_cpol`` (weight cache bits; ATOM 2 = nt), ``x_cpol`` (x load bits; ATOM 16 = sc1),
``await_`` (1: poll xrdy as ATOM; 0: skip, only for a launch that does not write x -- never in layer_post). Mutants:
``put_bit``, ``chain_drop`` (K step 1 left out), ``part`` (the logit put to the next part's slot).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass

from rocke.core.ir import BF16, F32, I32, I64, IRBuilder, VectorType

from . import seam_common as S

STAGE = "router"
MUTANTS = ("put_bit", "chain_drop", "part")
ABI = (("tag", "i32"), ("task", "i32"), ("x", "i64"), ("gate_w", "i64"), ("xrdy", "i64"), ("logit", "i64"), ("red", "i32"))
ROWS = 16
TILE = 16
PARTS = 4
PER_WAVE = S.HIDDEN // 32 // S.WAVES // PARTS  # 5


@dataclass(frozen=True)
class RouterKnobs(S.SeamKnobs):
    early_from: int = 12
    w_cpol: int = 2
    x_cpol: int = 16
    await_: int = 1

    def __post_init__(self) -> None:
        self.check(MUTANTS)


def parse_variant(v: str):
    n, _, e = v[1:].partition("e")
    return int(n), int(e)


def groups_of(s: int, experts: int):
    """``RouteShape.router_groups``: (groups, tokens a group)."""
    per_group = experts // ROWS * PARTS
    ntiles = -(-s // TILE)
    groups = max(1, min(ntiles, S.BLOCKS // per_group))
    return groups, -(-s // groups)


def build_router_kernel(rows: int, experts: int, knobs: RouterKnobs = RouterKnobs()):
    s = rows
    b = IRBuilder(f"rocke_dsv41_{STAGE}_n{s}e{experts}")
    b.kernel.attrs["max_workgroup_size"] = S.THREADS
    b.fence_barriers = "compiler"
    p = {n: b.param(n, I32 if ty == "i32" else I64) for n, ty in ABI}
    e = S.SeamEnv(b, knobs)
    c = e.c
    tag = b.readfirstlane(p["tag"])
    task = b.readfirstlane(p["task"])
    red = b.readfirstlane(p["red"])
    tid, lane, wave = e.tid, e.lane, e.wave
    groups, per = groups_of(s, experts)
    per_group = experts // ROWS * PARTS
    tile = b.div(b.mod(task, c(per_group)), c(PARTS))
    part = b.mod(task, c(PARTS))
    g0 = c(0) if groups == 1 else b.mul(b.div(task, c(per_group)), c(per))
    l16 = b.land(lane, c(15))
    k0s = [b.add(b.mul(b.add(b.add(b.mul(part, c(PER_WAVE * S.WAVES)), wave), c(S.WAVES * i)), c(32)),
                 b.mul(b.lshr(lane, c(4)), c(8))) for i in range(PER_WAVE)]  # fmt: skip

    def weights():
        r_w = e.rsrc(p["gate_w"])
        row = b.add(b.mul(tile, c(ROWS)), l16)
        return [b.buffer_load_cp(r_w, b.mul(b.add(b.mul(row, c(S.HIDDEN)), k0), c(2)), c(0), dwords=4, aux=knobs.w_cpol)
                for k0 in k0s]  # fmt: skip

    def await_x():
        if knobs.await_:
            r_x = e.rsrc(p["xrdy"])
            specs = [(r_x, b.readfirstlane(p["xrdy"]), c(t)) for t in range(s)]
            for b0 in range(0, s, knobs.poll_max):  # Mailbox.poll's batches of POLL_MAX pairs
                e.poll(tag, specs[b0 : b0 + knobs.poll_max])

    if s >= knobs.early_from:
        wvs = weights()
        await_x()
    else:
        await_x()
        wvs = weights()
    tiles = []
    for j0 in range(0, per, TILE):
        if groups == 1:
            tiles.append((c(j0), c(min(TILE, s - j0))))
        else:
            t0 = b.add(g0, c(j0))
            tiles.append((t0, b.smin(c(min(TILE, per - j0)), b.sub(c(s), t0))))
    r_x = e.rsrc(p["x"])
    xs = []
    for t0, n in tiles:
        tok = b.add(b.smin(l16, b.sub(n, c(1))), t0)
        xs.append([b.buffer_load_cp(r_x, b.mul(b.add(b.mul(tok, c(S.HIDDEN)), k0), c(2)), c(0), dwords=4,
                                    aux=knobs.x_cpol) for k0 in k0s])  # fmt: skip
    r_lg = e.rsrc(p["logit"])
    for (t0, n), xvs in zip(tiles, xs):
        acc = b.vec_pack([e.f(0.0)] * 4, F32)
        for i in range(PER_WAVE):
            if knobs.mutate == "chain_drop" and i == 1:
                continue
            av = b.vec_bitcast(wvs[i], VectorType(BF16, 8))
            bv = b.vec_bitcast(xvs[i], VectorType(BF16, 8))
            acc = b.mfma_f32_16x16x32_bf16(av, bv, acc)
        at = b.add(red, b.mul(b.add(b.mul(wave, c(64)), lane), c(4)))
        b.smem_store_vN(e.pool, [at], b.vec_pack([b.bitcast(b.vec_extract(acc, q), I32) for q in range(4)], I32), 4)
        e.barrier(lds_only=True)
        with b.scf_if(b.cmp_lt(tid, b.mul(n, c(ROWS)))):
            r = b.land(tid, c(ROWS - 1))
            tl = b.lshr(tid, c(4))
            base = b.add(b.mul(b.add(b.mul(b.lshr(r, c(2)), c(16)), tl), c(4)), b.land(r, c(3)))
            tot = e.f(0.0)
            for w in range(S.WAVES):
                tot = b.fadd(tot, e.lds_f32(b.add(red, b.add(base, c(w * 256)))))
            bits = b.bitcast(tot, I32)
            if knobs.mutate == "put_bit":
                bits = b.xor(bits, c(1))
            pt = b.add(part, c(1)) if knobs.mutate == "part" else part
            pi = b.add(b.mul(b.add(b.mul(b.add(t0, tl), c(experts)), b.add(b.mul(tile, c(ROWS)), r)), c(PARTS)), pt)
            b.buffer_store_cp(b.vec_pack([bits, tag], I32), r_lg, b.mul(pi, c(8)), c(0), aux=S.CPOL_SC1)
        e.barrier(lds_only=True)
    b.ret()
    return b.kernel


def write_router_functions(out: str, variants, knobs: RouterKnobs = RouterKnobs(), **kw) -> list:
    def build(v):
        n, ex = parse_variant(v)
        return build_router_kernel(n, ex, knobs)

    return S.write_functions(build, out, list(variants), STAGE, ABI, knobs, **kw)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="emit the rocKE V4.1 router stage functions")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default=",".join([f"n{r}e384" for r in range(1, 49)] + [f"n{r}e128" for r in range(1, 49)]))
    ap.add_argument("--knobs", default="")
    ap.add_argument("--flavor", default=None)
    ap.add_argument("--arch", default="gfx950")
    ap.add_argument("--bitcode", choices=("auto", "yes", "no"), default="no")
    a = ap.parse_args(argv)
    kn = S.knobs_from(RouterKnobs, a.knobs)
    vs = [v for v in a.variants.split(",") if v]
    bc = {"auto": "auto", "yes": True, "no": False}[a.bitcode]
    out = write_router_functions(a.out, vs, kn, arch=a.arch, flavor=a.flavor, bitcode=bc)
    print(json.dumps({"knobs": S.spec_of(kn), "written": out}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
