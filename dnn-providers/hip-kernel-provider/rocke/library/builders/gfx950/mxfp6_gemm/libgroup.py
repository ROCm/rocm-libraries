#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Grouped (MoE) GEMM driver for the gfx950 MXFP6 tile-config library: ``group_table`` kernels on ragged routing.

Problem: E groups (experts), group e has ``m_e`` tokens (0 allowed); C_e (m_e x N, bf16) = A_e (m_e x K) B_e^T
(N x K). MXFP6 operands (E2M3 / E3M2), E8M0 scales per 32 along K, fp32 accumulation.

Arms (one GPU, one process, interleaved):

* ``grp`` -- the ``group_table`` kernel: one packed A / C of ``Mpad`` rows (group e's rows padded to a multiple of
  tile_m, sorted by group), B / its scales per group at the batch strides, and the int32 table ``gtab[r]`` = group of
  tile row r. Groups without tokens own no tile rows: their weights are never read.
* ``twin`` -- the same code object loaded a second time (A-vs-A gate).
* ``pad`` -- what a library without a group table runs: the strided-batched kernel of the same config over every
  group, each padded to ``Mp`` = max_e m_e rounded up to tile_m (the same B buffer).

Every arm is verified before timing: ``n_src`` distinct data sources (group e uses source e mod n_src, at a
per-group row offset into the source's A), every valid row against its source's float64 reference (integer set
bit-exact, Gaussian / uniform-code sets within the scaled-MFMA tolerance), every padding row of the grouped C exactly
zero, two launches bitwise equal, the twin bitwise equal; then every must-fail mutant that applies (``gemm.MUTANTS``,
``LIB_MUTANTS``, ``GROUP_MUTANTS``: ``gt_zero`` ignores the table, ``gt_swap`` multiplies the neighbouring expert)
must fail. Throughput counts the useful FLOP 2 sum(m_e) N K for every arm (padding is not work).

Routing (``--route``): ``uniform:E:m`` | ``topk:E:T:k:alpha:seed`` (T tokens each pick k distinct experts with
probability proportional to (1 + rank)^-alpha over a random expert permutation; alpha 0 = uniform) |
``list:m0,m1,...``. Several routes per process: ``--route r1 --route r2`` (one compile, one GPU session).

``python -m builders.gfx950.mxfp6_gemm.libgroup run --tile 64x64x256 --family ak --fmt e2m3 --n 4096 --k 8192
--route uniform:64:64 --out DIR --cache DIR [--over JSON] [--base family|config]``. Absolute figures are for the
operator's internal record only; repository documents carry ratios.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import random
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

MODULE = "builders.gfx950.mxfp6_gemm.libgroup"
ROW_JITTER = 64  # group e reads its source's A rows from (13 e) mod ROW_JITTER: equal-source groups differ


def say(*a) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ===================================================================== routing / layout (pure host, tested)


def route(spec: str) -> List[int]:
    """Tokens per group from a route string (see the module docstring)."""
    kind, _, rest = spec.partition(":")
    if kind == "uniform":
        e, m = (int(x) for x in rest.split(":"))
        return [m] * e
    if kind == "list":
        return [int(x) for x in rest.split(",") if x != ""]
    if kind == "topk":
        e, t, k, alpha, seed = rest.split(":")
        e, t, k, alpha, seed = int(e), int(t), int(k), float(alpha), int(seed)
        if not 1 <= k <= e:
            raise ValueError("topk: 1 <= k <= E")
        rng = np.random.default_rng(seed)
        p = (1.0 + np.arange(e)) ** -alpha
        p = p[rng.permutation(e)]
        p /= p.sum()
        cnt = np.zeros(e, dtype=np.int64)
        for _ in range(t):  # each token: k distinct experts (no replacement), popularity-weighted
            cnt[rng.choice(e, size=k, replace=False, p=p)] += 1
        return [int(x) for x in cnt]
    raise ValueError(f"unknown route {spec!r}")


def group_layout(ms: Sequence[int], tile_m: int) -> Tuple[List[int], int, np.ndarray]:
    """(row offset of each group in the packed A / C, Mpad, int32 table: group of each tile row). Group e occupies
    rows [off_e, off_e + m_e) of its ceil(m_e / tile_m) tile rows; groups with m_e = 0 own none."""
    offs, tab, r = [], [], 0
    for e, m in enumerate(ms):
        if m < 0:
            raise ValueError("tokens per group must be >= 0")
        offs.append(r)
        nt = -(-m // tile_m)
        tab += [e] * nt
        r += nt * tile_m
    return offs, r, np.asarray(tab, dtype=np.int32)


def padded_rows(ms: Sequence[int], tile_m: int) -> int:
    """Rows per entry of the strided-batched alternative: the largest group, rounded up to the tile."""
    return max(tile_m, -(-max(ms) // tile_m) * tile_m)


def route_stats(ms: Sequence[int], tile_m: int) -> dict:
    _, mpad, tab = group_layout(ms, tile_m)
    mp = padded_rows(ms, tile_m)
    tot = int(sum(ms))
    return {"groups": len(ms), "tokens": tot, "max": int(max(ms)), "nonzero": int(sum(1 for m in ms if m)),
            "mean": tot / len(ms), "Mpad": mpad, "tile_rows": int(tab.size), "Mp": mp,
            "grp_row_eff": tot / max(1, mpad), "pad_row_eff": tot / max(1, mp * len(ms))}


# ===================================================================== specs


def base_spec(tile: str, family: str, fmt: str, k: int, base: str, over: Mapping[str, object], n: int = 4096,
              max_tokens: int = 0):
    """The config's kernel (``family`` = library.family_spec native tile, ``config`` = library.config_spec) at
    k_target = K, with ``over``; returns (grouped spec, padded strided-batched spec)."""
    from kernels.gfx950.mxfp6_gemm import library as LIB

    mt0, mt1, du = (int(x) for x in tile.lower().split("x"))
    over = dict(over)
    if base == "grouped":  # library.grouped_spec: the G work group for decode-like routing, group_m 1
        grp = LIB.grouped_spec(mt0, mt1, du, family, fmt, n, max_tokens, k_target=k, **over)
        return grp, replace(grp, group_table=False)
    fn = LIB.config_spec if base == "config" else LIB.family_spec
    pad = fn(mt0, mt1, du, family, fmt, k_target=k, **over)
    grp = replace(pad, group_table=True)
    return grp, pad


def mutants_of(g, spec, n_groups: int) -> List[str]:
    out = []
    for mu in [m for m in g.MUTANTS if m != "none"] + list(g.LIB_MUTANTS) + list(getattr(g, "GROUP_MUTANTS", ())):
        if mu == "batch_swap":
            continue
        if mu == "gt_swap" and n_groups % 2:
            continue
        if g.validate_spec(replace(spec, mutant=mu))[0]:
            out.append(mu)
    return out


# ===================================================================== host data


class GroupData:
    """Host images of one route: packed grouped A (+ scales), padded batched A (+ scales), per-group B (+ scales),
    the table; per-group references (rows of the source's float64 reference)."""

    def __init__(self, data, L, g, spec_g, spec_p, ms: Sequence[int], n: int, k: int, dist: str, n_src: int):
        self.ms, self.n, self.k, self.dist, self.n_src = list(ms), n, k, dist, n_src
        tm = spec_g.tile_m
        self.offs, self.mpad, self.tab = group_layout(ms, tm)
        self.mp = padded_rows(ms, tm)
        E = len(ms)
        self.rows0 = [(13 * e) % ROW_JITTER for e in range(E)]
        self.mbig = max(r + m for r, m in zip(self.rows0, ms)) + 1
        self.mbig = -(-self.mbig // 64) * 64
        fmt = spec_g.elem_fmt
        self.src = [data.operands(dist, fmt, self.mbig, n, k, s) for s in range(n_src)]
        rb, nsc = k * 6 // 8, k // 32

        def blank(rows):
            return np.zeros((rows, rb), np.uint8), np.full((rows, nsc), 127, np.uint8)

        ca, sa = blank(self.mpad)
        cp, sp = blank(self.mp * E)
        for e, m in enumerate(ms):
            if not m:
                continue
            a = self.src[e % n_src][0]
            r0 = self.rows0[e]
            ca[self.offs[e]:self.offs[e] + m] = a.codes[r0:r0 + m]
            sa[self.offs[e]:self.offs[e] + m] = a.scales[r0:r0 + m]
            cp[e * self.mp:e * self.mp + m] = a.codes[r0:r0 + m]
            sp[e * self.mp:e * self.mp + m] = a.scales[r0:r0 + m]
        A = type(self.src[0][0])
        self.a_grp = A(rows=self.mpad, k=k, codes=ca, scales=sa, fmt=fmt)
        self.a_pad = [A(rows=self.mp, k=k, codes=cp[e * self.mp:(e + 1) * self.mp],
                        scales=sp[e * self.mp:(e + 1) * self.mp], fmt=fmt) for e in range(E)]
        self.refs = [data.reference(dist, fmt, self.mbig, n, k, s) for s in range(n_src)]

    def images(self, L, g, spec, grouped: bool) -> dict:
        """Byte images + strides for ``spec`` (grouped: the packed A; else the padded entries)."""
        la, lb = g.operand_layout_of(spec, "A"), g.operand_layout_of(spec, "B")
        sla, slb = g.scale_layout_of(spec, "A"), g.scale_layout_of(spec, "B")
        bimg = [(L.pack_operand(b, lb, spec.tile_n), L.pack_scales_for(b.scales, spec.atom, slb))
                for _, b in self.src]
        E = len(self.ms)
        out = {"b": np.concatenate([bimg[e % self.n_src][0].view(np.uint8).reshape(-1) for e in range(E)]),
               "sb": np.concatenate([bimg[e % self.n_src][1].view(np.uint8).reshape(-1) for e in range(E)]),
               "stride_b": bimg[0][0].nbytes, "stride_sb": bimg[0][1].nbytes}
        if grouped and la == "mcontig":  # one [K][tile_m] slab per tile row (gemm group_table + mcontig)
            tm, A = spec.tile_m, type(self.a_grp)
            out["a"] = np.concatenate([L.pack_operand(A(rows=tm, k=self.k, codes=self.a_grp.codes[r:r + tm],
                                                         scales=self.a_grp.scales[r:r + tm], fmt=self.a_grp.fmt),
                                                       la, tm).view(np.uint8).reshape(-1)
                                       for r in range(0, self.mpad, tm)])
            out["sa"] = L.pack_scales_for(self.a_grp.scales, spec.atom, sla).view(np.uint8).reshape(-1)
            out["stride_a"] = out["stride_sa"] = out["stride_c"] = 0
        elif grouped:
            out["a"] = L.pack_operand(self.a_grp, la, spec.tile_m).view(np.uint8).reshape(-1)
            out["sa"] = L.pack_scales_for(self.a_grp.scales, spec.atom, sla).view(np.uint8).reshape(-1)
            out["stride_a"] = out["stride_sa"] = out["stride_c"] = 0
        else:
            ia = [(L.pack_operand(a, la, spec.tile_m), L.pack_scales_for(a.scales, spec.atom, sla)) for a in self.a_pad]
            out["a"] = np.concatenate([x[0].view(np.uint8).reshape(-1) for x in ia])
            out["sa"] = np.concatenate([x[1].view(np.uint8).reshape(-1) for x in ia])
            out["stride_a"], out["stride_sa"] = ia[0][0].nbytes, ia[0][1].nbytes
            out["stride_c"] = self.mp * self.n * 2
        return out

    def check(self, L, out: np.ndarray, grouped: bool) -> dict:
        """Every valid row against its source's reference; grouped: every padding row exactly zero."""
        exact = self.dist == "int"
        bad, n_tot, worst, finite, bad_groups = 0, 0, 0.0, True, []
        mask = np.ones(out.shape[0], dtype=bool)
        for e, m in enumerate(self.ms):
            if not m:
                continue
            o0 = self.offs[e] if grouped else e * self.mp
            mask[o0:o0 + m] = False
            rc, rs = self.refs[e % self.n_src]
            r0 = self.rows0[e]
            r = L.check_against_reference(out[o0:o0 + m], rc[r0:r0 + m], rs[r0:r0 + m], exact=exact)
            bad += r["bad"]
            n_tot += r["n"]
            finite &= r["finite"]
            worst = max(worst, r.get("worst_err_over_tol", 0.0))
            if r["bad"] or not r["finite"]:
                bad_groups.append(e)
        pad_nonzero = int(np.count_nonzero(out[mask] & 0x7FFF)) if grouped else 0  # +-0 only
        return {"pass": bad == 0 and finite and pad_nonzero == 0, "bad": bad, "n": n_tot, "finite": finite,
                "worst_err_over_tol": worst, "bad_groups": bad_groups[:16], "pad_nonzero": pad_nonzero,
                "exact": exact}


# ===================================================================== GPU


class Arm:
    def __init__(self, ses, meta: dict, spec, flop: float):
        self.ses, self.spec, self.meta, self.flop = ses, spec, meta, flop
        self.mod = ses.rt.load_module(Path(meta["hsaco"]).read_bytes())
        self.fn = self.mod.get_function(meta["name"])


class Bufs:
    """Device buffers of one route x dist for the grouped and padded arms (B shared)."""

    def __init__(self, ses, gd: GroupData, g, L, spec_g, spec_p):
        self.rt = ses.rt
        self.allocs = []
        ig = gd.images(L, g, spec_g, True)
        ip = gd.images(L, g, spec_p, False)
        put = self._put
        self.b, self.sb = put(ig["b"]), put(ig["sb"])
        self.ga, self.gsa = put(ig["a"]), put(ig["sa"])
        self.pa, self.psa = put(ip["a"]), put(ip["sa"])
        self.tab = put(gd.tab)
        self.gc = self.rt.alloc(gd.mpad * gd.n * 2)
        self.pc = self.rt.alloc(len(gd.ms) * gd.mp * gd.n * 2)
        self.allocs += [self.gc, self.pc]
        self.ig = {k: v for k, v in ig.items() if k.startswith("stride")}
        self.ip = {k: v for k, v in ip.items() if k.startswith("stride")}
        self.gd = gd

    def _put(self, arr: np.ndarray) -> int:
        arr = np.ascontiguousarray(arr)
        p = self.rt.alloc(max(arr.nbytes, 4))
        self.rt.memcpy_h2d(p, (ctypes.c_ubyte * int(arr.nbytes)).from_buffer(arr), arr.nbytes)
        self.allocs.append(p)
        return p

    def kargs(self, g, spec, grouped: bool) -> bytes:
        gd = self.gd
        if grouped:
            return g.kernargs(spec, self.ga, self.b, self.gsa, self.sb, self.gc, gd.mpad, gd.n, gd.k, 1,
                              stride_b=self.ig["stride_b"], stride_sb=self.ig["stride_sb"], gtab=self.tab)
        return g.kernargs(spec, self.pa, self.b, self.psa, self.sb, self.pc, gd.mp, gd.n, gd.k, len(gd.ms),
                          **self.ip)

    def grid(self, g, spec, grouped: bool):
        gd = self.gd
        return g.grid_for(spec, gd.mpad, gd.n, 1) if grouped else g.grid_for(spec, gd.mp, gd.n, len(gd.ms))

    def run(self, g, arm: Arm, grouped: bool) -> np.ndarray:
        gd = self.gd
        rows = gd.mpad if grouped else len(gd.ms) * gd.mp
        c = self.gc if grouped else self.pc
        host = np.empty((rows, gd.n), dtype=np.uint16)
        self.rt.memset(c, 0xFF, host.nbytes)  # NaN fill: unwritten valid rows fail, unwritten padding rows too
        grid, block = self.grid(g, arm.spec, grouped)
        self.rt.launch(arm.fn, grid, block, self.kargs(g, arm.spec, grouped))
        self.rt.wait_stream(0)
        self.rt.memcpy_d2h((ctypes.c_ubyte * int(host.nbytes)).from_buffer(host), c, host.nbytes)
        return host

    def free(self) -> None:
        for p in self.allocs:
            try:
                self.rt.free(p)
            except Exception:  # noqa: BLE001
                pass


def time_arms(ses, g, bufs: Bufs, arms: Dict[str, Tuple[Arm, bool]], cold_rounds: int, warm_windows: int,
              warm_s: float, rng: random.Random) -> dict:
    """Cold-start (post-idle) rounds and warm (steady-state) windows, arms shuffled per round; SCLK / power."""
    rt, clk, TU = ses.rt, ses.clk, ses.TU
    seq = {a: (arm.fn, *bufs.grid(g, arm.spec, gr), bufs.kargs(g, arm.spec, gr)) for a, (arm, gr) in arms.items()}
    flop = {a: arm.flop for a, (arm, _) in arms.items()}
    res = {a: {"cold": [], "warm": []} for a in arms}
    for _ in range(cold_rounds):
        order = list(arms)
        rng.shuffle(order)
        for a in order:
            rt.wait_stream(0)
            time.sleep(0.025)
            rt.launch(*seq[a])
            e0, e1 = rt.event(), rt.event()
            e0.record()
            for _i in range(3):
                rt.launch(*seq[a])
            e1.record()
            e1.synchronize()
            res[a]["cold"].append(3 * flop[a] / (e0.elapsed_to(e1) * 1e9))
            e0.destroy()
            e1.destroy()
    for _ in range(warm_windows):
        order = list(arms)
        rng.shuffle(order)
        for a in order:
            rt.wait_stream(0)
            t0 = time.time()
            for _i in range(5):
                rt.launch(*seq[a])
            rt.wait_stream(0)
            est = max(0.005, (time.time() - t0) * 1000.0 / 5)
            iters = max(50, int(warm_s * 1000.0 / est))
            for _i in range(20):
                rt.launch(*seq[a])
            clk.start()
            e0, e1 = rt.event(), rt.event()
            e0.record()
            for _i in range(iters):
                rt.launch(*seq[a])
            e1.record()
            e1.synchronize()
            ms = e0.elapsed_to(e1) / iters
            e0.destroy()
            e1.destroy()
            cs = clk.stop()
            w = cs.get("watts_med")
            tf = flop[a] / (ms * 1e9)
            res[a]["warm"].append({"tflops": tf, "ms": ms, "iters": iters, "mj": w * ms if w else None, **cs})
    med = lambda v: float(np.median(v)) if len(v) else None  # noqa: E731
    out = {"arms": res}
    for a in arms:
        out[f"{a}_warm_tflops"] = med([x["tflops"] for x in res[a]["warm"]])
        out[f"{a}_cold_tflops"] = med(res[a]["cold"])
        out[f"{a}_warm_ms"] = med([x["ms"] for x in res[a]["warm"]])
        out[f"{a}_warm_mj"] = med([x["mj"] for x in res[a]["warm"] if x["mj"]])
        out[f"{a}_warm_sclk"] = med([x["sclk_med"] for x in res[a]["warm"] if x.get("sclk_med")])
        out[f"{a}_warm_watts"] = med([x["watts_med"] for x in res[a]["warm"] if x.get("watts_med")])
    W = lambda a: [x["tflops"] for x in res[a]["warm"]]  # noqa: E731
    if "twin" in arms:
        out["twin_ratio_warm"] = TU.bootstrap_ci(W("grp"), W("twin"))
        out["twin_ratio_cold"] = TU.bootstrap_ci(res["grp"]["cold"], res["twin"]["cold"])
    if "pad" in arms:
        out["grp_over_pad_warm"] = TU.bootstrap_ci(W("grp"), W("pad"))
    for a in arms:
        if a.startswith("ref"):
            out[f"grp_over_{a}_warm"] = TU.bootstrap_ci(W("grp"), W(a))
    return out


# ===================================================================== run


def cmd_run(args) -> int:
    from builders.gfx950.mxfp6_gemm import libbench as LB

    over = json.loads(Path(args.over).read_text()) if args.over and os.path.exists(args.over) else (
        json.loads(args.over) if args.over else {})
    ref_over = json.loads(args.ref_over) if args.ref_over else {}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    g = LB._kmod()
    routes = [(r, route(r)) for r in args.route]
    spec_g, spec_p = base_spec(args.tile, args.family, args.fmt, args.k, args.base, over, n=args.n,
                               max_tokens=max(max(ms) for _, ms in routes))
    tool = args.tool or os.environ.get("MXFP6_RUST_COMPILER") or os.environ.get("ROCKE_RUST_COMPILE")
    levers = args.levers or LB.default_levers()
    cs = LB.CompileServer(tool, Path(args.cache) / "objs", jobs=args.jobs)
    E_all = {len(ms) for _, ms in routes}
    items = {"grp": spec_g, "pad": spec_p}
    for mu in mutants_of(g, spec_g, min(E_all)):
        items[f"mut_{mu}"] = replace(spec_g, mutant=mu)
    for nm, ro in ref_over.items():  # extra grouped arms (e.g. another schedule), timed against grp
        items[f"ref_{nm}"] = replace(spec_g, **ro)
    t0 = time.time()
    metas = cs.compile(items, levers)
    say(f"compiled {len(items)} builds in {time.time() - t0:.1f} s:",
        {k: (v.get("ok"), (v.get("notes") or {}).get("vgpr_spill_count")) for k, v in metas.items()})
    bad = [k for k, v in metas.items() if not v.get("ok") and not k.startswith("mut_")]
    if bad:
        (out / "error.json").write_text(json.dumps({k: metas[k] for k in bad}, default=str, indent=1))
        say("build failed:", bad)
        return 2
    if args.compile_only:
        (out / "builds.json").write_text(json.dumps(metas, default=str, indent=1))
        return 0
    ses = LB.GpuSession(Path(args.cache) / "data", seed=args.seed, workers=args.workers)
    data = ses.data_for(Path(args.cache) / "data", args.seed, args.workers)
    L = ses.L
    rng = random.Random(args.seed)
    for rs, ms in routes:
        tag = rs.replace(":", "_").replace(",", "-")[:80]
        res_p = out / f"{tag}.json"
        if res_p.exists() and not args.redo:
            say("done already:", rs)
            continue
        st = route_stats(ms, spec_g.tile_m)
        flop = 2.0 * st["tokens"] * args.n * args.k
        say(f"route {rs}: {st}")
        arms = {"grp": (Arm(ses, metas["grp"], spec_g, flop), True), "twin": (Arm(ses, metas["grp"], spec_g, flop), True)}
        if not args.no_pad:
            arms["pad"] = (Arm(ses, metas["pad"], spec_p, flop), False)
        for nm in ref_over:
            arms[f"ref_{nm}"] = (Arm(ses, metas[f"ref_{nm}"], items[f"ref_{nm}"], flop), True)
        muts = {k: Arm(ses, v, items[k], flop) for k, v in metas.items() if k.startswith("mut_") and v.get("ok")}
        record = {"route": rs, "ms": ms, "stats": st, "tile": args.tile, "family": args.family, "fmt": args.fmt,
                  "n": args.n, "k": args.k, "base": args.base, "over": over, "ref_over": ref_over,
                  "spec": asdict(spec_g), "levers": levers, "node": ses.node, "bdf": ses.bdf,
                  "job": os.environ.get("SLURM_JOB_ID"),
                  "builds": {k: {x: metas[k].get(x) for x in ("name", "notes", "loop", "ir_sha256", "rust_tool_sha256")}
                             for k in metas if metas[k].get("ok")},
                  "mx4_problem": f"{len(ms)}x{st['Mp']}x{args.n}x{args.k}"}
        ver, ok = {}, True
        for dist in args.verify_dists.split(","):
            gd = GroupData(data, L, g, spec_g, spec_p, ms, args.n, args.k, dist, args.n_src)
            bufs = Bufs(ses, gd, g, L, spec_g, spec_p)
            try:
                v = {}
                for a, (arm, gr) in arms.items():
                    if a == "twin":
                        continue
                    o1, o2 = bufs.run(g, arm, gr), bufs.run(g, arm, gr)
                    c = gd.check(L, o1, gr)
                    c["deterministic"] = bool(np.array_equal(o1, o2))
                    if a == "grp":
                        c["twin_bitwise"] = bool(np.array_equal(o1, bufs.run(g, arms["twin"][0], True)))
                    v[a] = c
                    ok &= c["pass"] and c["deterministic"] and c.get("twin_bitwise", True)
                mv = {}
                for nm, arm in muts.items():
                    if dist == "int" and "scale_sel" in nm:
                        continue  # all scales 2^0 on the integer set
                    o = bufs.run(g, arm, True)
                    mv[nm] = {"failed_as_required": not gd.check(L, o, True)["pass"]}
                v["mutants"] = mv
                ver[dist] = v
            finally:
                bufs.free()
        mres = {}
        for dist, v in ver.items():
            for nm, r in v.get("mutants", {}).items():
                mres[nm] = mres.get(nm, False) or r["failed_as_required"]
        ok &= all(mres.values()) and len(mres) == len(muts)
        record["verify"] = {"ok": bool(ok), "dists": ver, "mutants_failed": mres}
        say(f"verify {rs}: ok={ok} mutants={mres}")
        if ok and not args.verify_only:
            gd = GroupData(data, L, g, spec_g, spec_p, ms, args.n, args.k, args.time_dist, args.n_src)
            bufs = Bufs(ses, gd, g, L, spec_g, spec_p)
            try:
                tm = time_arms(ses, g, bufs, arms, args.cold_rounds, args.warm_windows, args.warm_s, rng)
            finally:
                bufs.free()
            record["timing"] = tm
            say(f"timing {rs}: " + " ".join(f"{a} {tm.get(a + '_warm_tflops') or 0:.0f}" for a in arms),
                "twin", tm.get("twin_ratio_warm"), "grp/pad", tm.get("grp_over_pad_warm"))
        for a in list(arms.values()) + [(m, True) for m in muts.values()]:
            try:
                a[0].mod.unload()
            except Exception:  # noqa: BLE001
                pass
        res_p.write_text(json.dumps(record, default=str, indent=1))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog=MODULE)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tile", required=True)
    r.add_argument("--family", choices=("ak", "am"), default="ak")
    r.add_argument("--fmt", choices=("e2m3", "e3m2"), default="e2m3")
    r.add_argument("--n", type=int, required=True)
    r.add_argument("--k", type=int, required=True)
    r.add_argument("--route", action="append", required=True)
    r.add_argument("--base", choices=("family", "config", "grouped"), default="family")
    r.add_argument("--over", default="")
    r.add_argument("--ref-over", default="", help="JSON {name: overrides}: extra grouped arms timed against grp")
    r.add_argument("--levers", default="")
    r.add_argument("--tool", default="")
    r.add_argument("--out", required=True)
    r.add_argument("--cache", required=True)
    r.add_argument("--jobs", type=int, default=8)
    r.add_argument("--workers", type=int, default=16)
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--n-src", type=int, default=3)
    r.add_argument("--verify-dists", default="int,gauss,uniform")
    r.add_argument("--time-dist", default="gauss")
    r.add_argument("--cold-rounds", type=int, default=9)
    r.add_argument("--warm-windows", type=int, default=6)
    r.add_argument("--warm-s", type=float, default=1.5)
    r.add_argument("--no-pad", action="store_true")
    r.add_argument("--redo", action="store_true")
    r.add_argument("--compile-only", action="store_true")
    r.add_argument("--verify-only", action="store_true", help="verify + mutants, no timing (correctness nodes)")
    a = ap.parse_args(argv)
    return {"run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
