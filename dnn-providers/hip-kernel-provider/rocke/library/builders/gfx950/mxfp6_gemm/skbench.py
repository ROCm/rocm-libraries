# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Split-K / stream-K lane harness (``splitk="fused"``): verify-then-time on one GPU, one long-lived process.

A *case* is one problem (batch x M x N x K, format, family) with named *arms* (spec overrides on a library base:
``config`` = ``library.config_spec``, ``family`` = ``library.family_spec`` (native tile), ``dispatch``). Per case:

1. build every arm, its A-vs-A twin (the first arm's code object loaded twice) and, for split arms, the must-fail
   mutants (``splitk.SK_MUTANTS`` + ``drop_kstep``) through one in-process ``libbench.CompileServer``;
2. verify each arm on the integer set (bit-exact), Gaussian and uniform-code data (scaled-MFMA tolerance): three
   launches each, C NaN-filled before every launch, all three bitwise identical (the counters reset by the kernel
   itself between launches); split arms with s >= 3 also on the **order set** (``order_operands``: per column, two
   +-2^25-scale partials and a 1 whose sum depends on the fix-up order; the expected result is the float32 fold in
   split order, compared bit for bit);
3. every mutant must fail one of those checks;
4. time the verified arms: cold-start (post-idle: 25 ms gap, 1 warmup + 3 timed) and warm (steady-state) windows,
   arms shuffled per window, inputs rotated over replicated buffer sets (footprint >= 2 x the 256 MiB MALL), sysfs
   SCLK / power sampled (mJ per launch); the twin gives the A-vs-A ratio with a paired bootstrap CI.

Usage (node side): ``python -m builders.gfx950.mxfp6_gemm.skbench run --cases cases.json --out out.jsonl --cache DIR``
(``--tool`` / ``ROCKE_RUST_COMPILE``: the direct compiler). Output: one JSON line per case (no numbers in the repo).
A HIP fault exits 86 after recording the case; the caller restarts and the case is skipped.
"""

from __future__ import annotations

import argparse
import ctypes
import functools
import json
import os
import random
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import numpy as np

SK_EXTRA_MUTANTS = ("drop_kstep",)
MALL = 256 * 1024 * 1024


def _mods():
    from benchmarks.gfx950.mxfp6_gemm import bench as B
    from builders.gfx950.mxfp6_gemm import libbench as LB
    from builders.gfx950.mxfp6_gemm import tune as TU
    from kernels.gfx950.mxfp6_gemm import gemm as G
    from kernels.gfx950.mxfp6_gemm import layout as L
    from kernels.gfx950.mxfp6_gemm import library as LIB
    from kernels.gfx950.mxfp6_gemm import splitk as SK

    return B, LB, TU, G, L, LIB, SK


# ----------------------------------------------------------------------------------------------------- specs
def arm_spec(case: Mapping, over: Mapping):
    """The spec of an arm: the library base of the case's tile with ``over`` applied; a pure split arm takes its
    reference K from the split length (the build's K residue class is the unit's, not the problem's)."""
    _, _, _, G, _, LIB, SK = _mods()
    mt0, mt1, du = (int(x) for x in case["tile"].split("x"))
    fam, fmt = case["fam"], case.get("fmt", "e2m3")
    over = dict(over)
    base = over.pop("base", case.get("base", "config"))
    m, n, k, batch = int(case["m"]), int(case["n"]), int(case["k"]), int(case.get("batch", 1))

    if base == "select":  # the lane's per-problem choice (splitk.select_config); over: sk_release / sk_static / sk_fix / sk_kargs only
        sp = SK.select_config(mt0, mt1, du, fam, fmt, m, n, k, batch)
        if sp.splitk != "none":
            sp = replace(sp, **{kk: v for kk, v in over.items() if kk in ("sk_release", "sk_static", "sk_fix", "sk_kargs")})
            if sp.sk_fix == "owner" and (sp.sk_split < 2 or not sp.sk_static
                                         or not G.runtime_problem_ok(sp, m, n, k, batch)[0]):
                sp = replace(sp, sk_fix="last")  # (owner mode needs co-resident units)
            if sp.sk_static and sp.sk_split < 2:
                sp = replace(sp, sk_static=False)
        return sp

    def mk(k_target):
        if base == "family":
            return LIB.family_spec(mt0, mt1, du, fam, fmt, k_target=k_target, **over)
        if base == "dispatch":
            return LIB.dispatch(mt0, mt1, du, fam, fmt, m, n, batch, k_target=k_target, **over)
        return LIB.config_spec(mt0, mt1, du, fam, fmt, k_target=k_target, **over)

    short = bool(over.pop("short", False))  # exact-K split build (units of exactly K / s, any length >= stages + 2)
    if base == "dispatch":  # the shipped choice as is (library.dispatch routes short grids through splitk.select)
        return mk(k)
    try:
        sp = mk(k)
    except ValueError:
        sp = mk(None)
    if getattr(sp, "splitk", "none") != "none":
        p = SK.plan(sp, m, n, k, batch)
        if short and p["D"] == p["T"]:  # data-parallel units of an exact-K build (any K >= stages + 2 tiles)
            return SK.short_spec(mk(None), k // 128)
        if p["D"] == 0 and p["s"] > 1 and k % (128 * p["s"]) == 0:
            c = k // 128 // p["s"]
            if not short:
                try:
                    return mk(k // p["s"])
                except ValueError:
                    pass
            return SK.short_spec(mk(None), c)
    return sp


# ----------------------------------------------------------------------------------------------------- data
def order_operands(m: int, n: int, k: int, s: int, fmt: str = "e2m3"):
    """The order set of an s-way split (s >= 3): A = 1.0 everywhere; B row n: split ``(n + 0) % s`` holds +2^E per
    element (its partial is +P, P in [2^25, 2^26)), split ``(n + 1) % s`` the same negated (-P), split ``(n + 2) % s``
    a single 1.0 (partial 1), the rest 0. Every partial is exact; adding them in a different order changes whether
    the 1 survives. Returns (A, B, expected f32 [n] per column = the float32 fold in split order)."""
    _, _, _, _, L, _, _ = _mods()
    if s < 3:
        raise ValueError("order set: s >= 3")
    kc = k // s
    e = 0
    while kc * 2 ** e < 2 ** 25:
        e += 1
    one = int(np.where(L.FMT_VALUES[fmt] == 1.0)[0][0])
    neg = one | 32
    a_codes = np.full((m, k), one, dtype=np.uint8)
    b_codes = np.zeros((n, k), dtype=np.uint8)
    b_sc = np.full((n, k // 32), 127, dtype=np.uint8)
    parts = np.zeros((n, s), dtype=np.float64)
    for r in range(n):
        jp, jn, j1 = r % s, (r + 1) % s, (r + 2) % s
        b_codes[r, jp * kc:(jp + 1) * kc] = one
        b_sc[r, jp * kc // 32:(jp + 1) * kc // 32] = 127 + e
        b_codes[r, jn * kc:(jn + 1) * kc] = neg
        b_sc[r, jn * kc // 32:(jn + 1) * kc // 32] = 127 + e
        b_codes[r, j1 * kc] = one
        parts[r, jp], parts[r, jn], parts[r, j1] = kc * 2.0 ** e, -kc * 2.0 ** e, 1.0
    A = L.Operand(m, k, L.pack_codes(a_codes), np.full((m, k // 32), 127, dtype=np.uint8), fmt)
    B = L.Operand(n, k, L.pack_codes(b_codes), b_sc, fmt)
    acc = parts[:, 0].astype(np.float32)
    for j in range(1, s):
        acc = (acc + parts[:, j].astype(np.float32)).astype(np.float32)
    return A, B, acc


def pack_inputs(spec, a, b):
    """Device images of one entry: A codes (canonical / mcontig), B c0c1 (edge blocks), wtile scales (``m_mask``: A's
    scale rows padded to whole tile rows instead of the edge shift)."""
    _, _, _, G, L, _, _ = _mods()
    la, lb = G.operand_layout_of(spec, "A"), G.operand_layout_of(spec, "B")
    sla, slb = G.scale_layout_of(spec, "A"), G.scale_layout_of(spec, "B")
    if getattr(spec, "k_tail", False):  # K % 128 != 0: B and both scale images padded to whole 128-K tiles
        b = L.pad_k(b)
        sa = L.pad_k(a).scales
    else:
        sa = a.scales
    if getattr(spec, "m_mask", False) and a.rows % spec.tile_m:
        pad = spec.tile_m - a.rows % spec.tile_m
        sa = np.concatenate([sa, np.zeros((pad, sa.shape[1]), dtype=np.uint8)])
    if la == "mcontig" and a.rows % 16:
        # the M-padded producer layout (gemm.mcontig_rows): the padding rows are don't-care -- random codes here (the
        # device image's own padding is zeros), so a kernel that let them reach a stored C row fails verification
        r16 = G.mcontig_rows(a.rows)
        junk = np.random.default_rng(a.rows).integers(0, 256, (r16 - a.rows, a.codes.shape[1]), dtype=np.uint8)
        a = L.Operand(r16, a.k, np.concatenate([a.codes, junk]), np.concatenate(
            [a.scales, np.full((r16 - a.rows, a.scales.shape[1]), 127, np.uint8)]), a.fmt)
    return (L.pack_operand(a, la, spec.tile_m), L.pack_operand(b, lb, spec.tile_n),
            L.pack_scales_for(sa, spec.atom, sla), L.pack_scales_for(b.scales, spec.atom, slb))


class Dev:
    """Device buffers of one case."""

    def __init__(self, rt):
        self.rt = rt
        self.allocs: List[int] = []

    def put(self, arr: np.ndarray) -> int:
        arr = np.ascontiguousarray(arr)
        ptr = self.rt.alloc(max(16, arr.nbytes))
        self.rt.memcpy_h2d(ptr, (ctypes.c_ubyte * int(arr.nbytes)).from_buffer(arr), arr.nbytes)
        self.allocs.append(ptr)
        return ptr

    def zeros(self, nbytes: int) -> int:
        ptr = self.rt.alloc(max(16, nbytes))
        self.rt.memset(ptr, 0, max(16, nbytes))
        self.allocs.append(ptr)
        return ptr

    def free(self) -> None:
        for p in self.allocs:
            try:
                self.rt.free(p)
            except Exception:  # noqa: BLE001
                pass
        self.allocs = []


# ----------------------------------------------------------------------------------------------------- one case
def run_case(rt, cs, levers: str, data, clk, case: Mapping, rng: random.Random) -> dict:
    B, LB, TU, G, L, LIB, SK = _mods()
    m, n, k, batch = int(case["m"]), int(case["n"]), int(case["k"]), int(case.get("batch", 1))
    fmt = case.get("fmt", "e2m3")
    flop = 2.0 * batch * m * n * k
    out: Dict[str, object] = {"case": dict(case), "arms": {}, "skipped": {}}
    specs: Dict[str, object] = {}
    for name, over in case["arms"].items():
        try:
            sp = arm_spec(case, over)
        except ValueError as e:
            out["skipped"][name] = f"spec: {e}"[:300]
            continue
        ok, why = G.validate_spec(sp)
        if ok:
            ok, why = G.runtime_problem_ok(sp, m, n, k, batch)
        if not ok:
            out["skipped"][name] = why[:300]
            continue
        specs[name] = sp
    if not specs:
        return out
    builds = dict(specs)
    want_mut = case.get("mutants", True)  # True: every split arm; a list: those arms only
    for name, sp in specs.items():
        if isinstance(want_mut, list) and name not in want_mut:
            continue
        if sp.splitk != "none" and want_mut and SK.plan(sp, m, n, k, batch)["D"] < SK.plan(sp, m, n, k, batch)["T"]:
            for mu in SK.SK_MUTANTS + SK_EXTRA_MUTANTS:
                mo = replace(sp, mutant=mu)
                if G.validate_spec(mo)[0]:
                    builds[f"{name}~{mu}"] = mo
        if getattr(sp, "m_mask", False) and want_mut:  # the row mask: one row short / one row over (into the guard)
            for mu in SK.MASK_MUTANTS:
                if mu == "mask_over" and m % sp.tile_m == 0:
                    continue  # no tile holds row M: the mutant cannot store it
                mo = replace(sp, mutant=mu)
                if G.validate_spec(mo)[0]:
                    builds[f"{name}~{mu}"] = mo
    t0 = time.time()
    metas = cs.compile(builds, levers)
    out["compile_s"] = round(time.time() - t0, 1)
    funcs, mods = {}, []
    for bid, meta in metas.items():
        if not meta.get("ok"):
            out["skipped"][bid] = ("compile: " + str(meta.get("error")))[:600]
            continue
        md = rt.load_module(Path(meta["hsaco"]).read_bytes())
        mods.append(md)
        funcs[bid] = (md.get_function(meta["name"]), meta)
        if bid in specs:
            n_ = meta.get("notes", {})
            out["arms"].setdefault(bid, {})["build"] = {
                "name": meta["name"], "vgpr": n_.get("vgpr_count"), "agpr": n_.get("agpr_count"),
                "sgpr": n_.get("sgpr_count"), "spill": n_.get("vgpr_spill_count"),
                "scratch": n_.get("private_segment_fixed_size"), "lds": n_.get("group_segment_fixed_size"),
                "loop": meta.get("loop")}
    first = next((a for a in specs if a in funcs), None)
    if first is not None:  # A-vs-A twin: the first arm's code object loaded a second time
        md = rt.load_module(Path(metas[first]["hsaco"]).read_bytes())
        mods.append(md)
        funcs["~twin"] = (md.get_function(metas[first]["name"]), metas[first])
        specs_twin = specs[first]
    dev = Dev(rt)
    c_bytes = batch * m * n * 2
    # a guard after C (one tile_m-row block of the widest arm, at least 64 KiB): stores past the last entry's M
    # rows land there; every check verifies the guard is untouched
    g_bytes = max(65536, max(sp_.tile_m for sp_ in builds.values()) * n * 2)
    pc = dev.zeros(c_bytes + g_bytes)
    g_pat = np.full(g_bytes, 0x5A, np.uint8)
    g_host = np.empty(g_bytes, np.uint8)
    c_host = np.empty((batch, m, n), dtype=np.uint16)

    def spec_of(bid):
        return specs_twin if bid == "~twin" else builds[bid]

    ws_of: Dict[str, tuple] = {}

    def wsbuf(bid):  # per build: its own workspace + counters (a mutant cannot poison another arm's counters)
        if bid not in ws_of:
            sp = spec_of(bid)
            if sp.splitk == "none":
                ws_of[bid] = (0, 0)
            else:
                wsb, cb = SK.buffers(sp, m, n, k, batch)
                ws_of[bid] = (dev.zeros(wsb), dev.zeros(cb))
        return ws_of[bid]

    inp: Dict[tuple, dict] = {}

    def inputs(sp, dist: str, setix: int = 0, nsrc: int = 3):
        key = (dist, setix, nsrc, G.operand_layout_of(sp, "A"), G.operand_layout_of(sp, "B"), G.scale_layout_of(sp, "A"),
               G.scale_layout_of(sp, "B"), sp.tile_m, sp.tile_n, bool(getattr(sp, "m_mask", False)),
               case.get("_order_s") if dist == "order" else None)
        if key in inp:
            return inp[key]
        imgs = []
        nsrc = min(nsrc, batch)
        for s_ in range(nsrc):
            if dist == "order":
                a, b_, _ = order_operands(m, n, k, int(case["_order_s"]), fmt)
            else:
                a, b_ = data.operands(dist, fmt, m, n, k, s_)
            imgs.append(pack_inputs(sp, a, b_))
        cat = [np.concatenate([imgs[e % nsrc][i].view(np.uint8).reshape(-1) for e in range(batch)]) for i in range(4)]
        d = {"ptrs": [dev.put(x) for x in cat], "strides": [imgs[0][i].nbytes for i in range(4)] + [m * n * 2]}
        inp[key] = d
        return d

    def kargs(bid, dist, setix=0, nsrc=3, c_ptr=None):
        sp = spec_of(bid)
        d = inputs(sp, dist, setix, nsrc)
        ws, cnt = wsbuf(bid)
        a_, b_, sa_, sb_ = d["ptrs"]
        st = d["strides"]
        return G.kernargs(sp, a_, b_, sa_, sb_, c_ptr or pc, m, n, k, batch, st[0], st[1], st[2], st[3], st[4],
                          ws=ws, cnt=cnt)

    def launch_out(bid, ka):
        sp = spec_of(bid)
        grid, block = G.grid_for(sp, m, n, batch)
        rt.memset(pc, 0xFF, c_bytes)  # NaN fill: every element must be written
        rt.memset(pc + c_bytes, 0x5A, g_bytes)
        rt.launch(funcs[bid][0], grid, block, ka)
        rt.wait_stream(0)
        rt.memcpy_d2h((ctypes.c_ubyte * int(c_host.nbytes)).from_buffer(c_host), pc, c_host.nbytes)
        rt.memcpy_d2h((ctypes.c_ubyte * int(g_bytes)).from_buffer(g_host), pc + c_bytes, g_bytes)
        guard_hit[0] += int(np.count_nonzero(g_host != g_pat))
        return c_host.copy()

    guard_hit = [0]

    def check(bid, dist, nl=int(case.get("launches", 3))) -> dict:
        ka = kargs(bid, dist)
        guard_hit[0] = 0
        outs = [launch_out(bid, ka) for _ in range(nl)]
        det = all(np.array_equal(outs[0], o) for o in outs[1:])
        if guard_hit[0]:  # a store past C
            return {"pass": False, "bad": guard_hit[0], "deterministic": det, "guard": guard_hit[0]}
        if dist == "order":
            _, _, want = order_operands(1, n, k, int(case["_order_s"]), fmt)
            wb = L.f32_to_bf16_bits(want)
            bad = sum(int(np.count_nonzero(o[e] != wb[None, :])) for o in outs for e in range(batch))
            return {"pass": bad == 0, "bad": bad, "deterministic": det}
        res = {"pass": True, "bad": 0, "deterministic": det, "worst": 0.0}
        for o in outs:
            for e in range(batch):
                rc, rs_ = data.reference(dist, fmt, m, n, k, e % min(3, batch))
                r = L.check_against_reference(o[e], rc, rs_, exact=(dist == "int"))
                res["bad"] += r["bad"]
                res["pass"] = res["pass"] and r["pass"]
                res["worst"] = max(res["worst"], r.get("worst_err_over_tol", 0.0))
        return res

    dists = case.get("verify", ["int", "gauss", "uniform"])
    verified = []
    for bid in list(specs) + ["~twin"]:
        if bid not in funcs:
            continue
        sp = spec_of(bid)
        v: Dict[str, object] = {}
        okall = True
        for dist in dists:
            r = check(bid, dist)
            v[dist] = r
            okall &= r["pass"] and r["deterministic"]
        p = SK.plan(sp, m, n, k, batch) if sp.splitk != "none" else None
        if p and p["D"] < p["T"] and p["s"] >= 3:
            case = {**case, "_order_s": p["s"]}
            r = check(bid, "order")
            v["order"] = r
            okall &= r["pass"] and r["deterministic"]
        name = first if bid == "~twin" else bid
        tgt = out["arms"].setdefault(name, {})
        if bid == "~twin":
            tgt["twin_verify"] = {"ok": okall}
        else:
            tgt.update({"verify": v, "ok": okall, "plan": p, "spec": {kk: vv for kk, vv in asdict(sp).items()
                                                                       if vv != getattr(G.Mxfp6GemmSpec(), kk)}})
        if not okall:
            continue
        muts = {}
        for mb in [x for x in funcs if x.startswith(bid + "~")]:
            mu = mb.split("~", 1)[1]
            fails, why = False, ""
            for dist in dists + (["order"] if (p and p["D"] < p["T"] and p["s"] >= 3) else []):
                try:
                    r = check(mb, dist)
                except Exception as e:  # noqa: BLE001 -- a failing mutant may fault
                    fails, why = True, f"error {e!r}"[:200]
                    raise
                if not (r["pass"] and r["deterministic"]):
                    fails, why = True, f"{dist}: pass={r['pass']} det={r['deterministic']}"
                    break
            muts[mu] = {"failed_as_required": fails, "how": why, "required": mu not in SK.SK_SHOULD_FAIL}
        if muts:
            tgt["mutants"] = muts
            okall &= all(x["failed_as_required"] for x in muts.values() if x["required"])
            tgt["ok"] = okall
        if okall and bid != "~twin":
            verified.append(bid)
    stress = int(case.get("stress", 0))  # repeat launches of every verified split arm: all bitwise equal to the first
    for bid in list(verified):
        if stress and spec_of(bid).splitk != "none":
            ka = kargs(bid, "gauss")
            ref = launch_out(bid, ka)
            diff = sum(int(not np.array_equal(ref, launch_out(bid, ka))) for _ in range(stress))
            out["arms"][bid]["stress"] = {"launches": stress, "mismatches": diff}
            if diff:
                out["arms"][bid]["ok"] = False
                verified.remove(bid)
    twin_ok = first in verified and out["arms"][first].get("twin_verify", {}).get("ok")
    if case.get("time", True) and verified:
        sp0 = specs[verified[0]]
        a0 = inputs(sp0, "gauss", 0, 1)
        in_bytes = sum(int(x) for x in a0["strides"][:4]) * batch
        rot = int(min(int(case.get("max_rotate", 12)), max(1, -(-2 * MALL // max(1, in_bytes)))))
        timed = list(verified) + (["~twin"] if twin_ok else [])
        seqs = {}
        # launcher "prepared" (default): every launch's ctypes arguments built once (rt.prepare_launch), the timed
        # loop only enqueues -- the host cost per launch is hipModuleLaunchKernel's, as the MXFP4 harness's C++ loop.
        # "plain": rt.launch rebuilds them per call (the Python host cost; A/B of the launcher only).
        prepared = case.get("launcher", "prepared") == "prepared"
        for bid in timed:
            sp = spec_of(bid)
            grid, block = G.grid_for(sp, m, n, batch)
            if prepared:
                seqs[bid] = [rt.prepare_launch(funcs[bid][0], grid, block, kargs(bid, "gauss", s_, 1))
                             for s_ in range(rot)]
            else:
                seqs[bid] = [functools.partial(rt.launch, funcs[bid][0], grid, block, kargs(bid, "gauss", s_, 1))
                             for s_ in range(rot)]
        res = {bid: {"cold": [], "warm": []} for bid in timed}
        for _ in range(int(case.get("cold_rounds", 8))):
            order = list(timed)
            rng.shuffle(order)
            for bid in order:
                s_ = seqs[bid]
                rt.wait_stream(0)
                time.sleep(0.025)
                s_[0]()
                e0, e1 = rt.event(), rt.event()
                e0.record()
                for i in range(3):
                    s_[(i + 1) % rot]()
                e1.record()
                e1.synchronize()
                res[bid]["cold"].append(3 * flop / (e0.elapsed_to(e1) * 1e9))
                e0.destroy()
                e1.destroy()
        warm_s = float(case.get("warm_s", 1.0))
        for _ in range(int(case.get("windows", 4))):
            order = list(timed)
            rng.shuffle(order)
            for bid in order:
                s_ = seqs[bid]
                rt.wait_stream(0)
                t0 = time.time()
                for i in range(5):
                    s_[i % rot]()
                rt.wait_stream(0)
                est = max(0.002, (time.time() - t0) * 1000.0 / 5)
                iters = max(100, int(warm_s * 1000.0 / est))
                for i in range(20):
                    s_[i % rot]()
                clk.start()
                e0, e1 = rt.event(), rt.event()
                e0.record()
                for i in range(iters):
                    s_[i % rot]()
                e1.record()
                e1.synchronize()
                ms = e0.elapsed_to(e1) / iters
                e0.destroy()
                e1.destroy()
                cs_ = clk.stop()
                w = cs_.get("watts_med")
                res[bid]["warm"].append({"tflops": flop / (ms * 1e9), "ms": ms, "iters": iters, "sclk": cs_.get("sclk_med"),
                                         "watts": w, "mj": (w * ms) if w else None})
        gpu = {bid: [] for bid in timed}
        n_pq = int(case.get("prequeue", 0))
        if n_pq:
            # GPU-bound contract ("prequeued"): a queued fill of a scratch buffer keeps the GPU busy while the host
            # enqueues n_pq launches behind it; the events then bound back-to-back GPU execution only (each
            # dispatch's command-processor gap included, the host's launch cost excluded) -- the same contract as
            # the MXFP4 harness's prequeue mode
            scratch_b = int(case.get("prequeue_bytes", 4 << 30))
            scratch = dev.zeros(scratch_b)
            for _ in range(int(case.get("windows", 4))):
                order = list(timed)
                rng.shuffle(order)
                for bid in order:
                    s_ = seqs[bid]
                    for i in range(10):
                        s_[i % rot]()
                    rt.wait_stream(0)
                    for _f in range(int(case.get("prequeue_fills", 3))):
                        rt.memset_async(scratch, _f & 0xFF, scratch_b)
                    e0, e1 = rt.event(), rt.event()
                    e0.record()
                    t0 = time.time()
                    for i in range(n_pq):
                        s_[i % rot]()
                    t_enq = time.time() - t0
                    e1.record()
                    e1.synchronize()
                    ms = e0.elapsed_to(e1) / n_pq
                    e0.destroy()
                    e1.destroy()
                    gpu[bid].append({"ms": ms, "host_enqueue_ms": t_enq * 1000.0 / n_pq})
        tim = {"rotate": rot, "in_bytes": in_bytes, "launcher": "prepared" if prepared else "plain", "prequeue": n_pq}
        for bid in timed:
            ws_ = [x["tflops"] for x in res[bid]["warm"]]
            name = first + "~twin" if bid == "~twin" else bid
            tim[name] = {"warm_tflops": float(np.median(ws_)), "warm": res[bid]["warm"],
                         "cold_tflops": float(np.median(res[bid]["cold"])), "cold": res[bid]["cold"]}
            if gpu[bid]:
                gms = float(np.median([x["ms"] for x in gpu[bid]]))
                tim[name].update(gpu_ms=gms, gpu_tflops=flop / (gms * 1e9), gpu=gpu[bid])
        if "~twin" in res:
            tim["twin_ratio_warm"] = TU.bootstrap_ci([x["tflops"] for x in res[first]["warm"]],
                                                     [x["tflops"] for x in res["~twin"]["warm"]])
        out["timing"] = tim
    for md in mods:
        try:
            md.unload()
        except Exception:  # noqa: BLE001
            pass
    dev.free()
    return out


# ----------------------------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--cases", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--cache", required=True)
    r.add_argument("--tool", default=os.environ.get("ROCKE_RUST_COMPILE", ""))
    r.add_argument("--objcache", default="")
    r.add_argument("--levers", default="")
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--workers", type=int, default=16)
    r.add_argument("--only", default="", help="comma list of case ids")
    a = ap.parse_args(argv)
    B, LB, TU, G, L, LIB, SK = _mods()
    from rocke.runtime.hip_module import Runtime

    cases = json.loads(Path(a.cases).read_text())
    if a.only:
        keep = set(a.only.split(","))
        cases = [c for c in cases if c["id"] in keep]
    outp = Path(a.out)
    done = set()
    if outp.exists():
        for line in outp.read_text().splitlines():
            try:
                done.add(json.loads(line)["case"]["id"])
            except Exception:  # noqa: BLE001
                pass
    inflight = outp.with_suffix(".inflight")
    if inflight.exists():  # the previous process died in this case (HIP fault): record and skip it
        cid = inflight.read_text().strip()
        if cid and cid not in done:
            with outp.open("a") as f:
                f.write(json.dumps({"case": {"id": cid}, "fault": "process died in this case"}) + "\n")
            done.add(cid)
        inflight.unlink()
    rt = Runtime()
    bdf = B.device_bdf()
    clk = TU.Clock(bdf)
    data = LB.Data(Path(a.cache), a.seed, a.workers)
    cs = LB.CompileServer(a.tool, Path(a.objcache or (Path(a.cache) / "objs")), jobs=8, batch=16)
    levers = a.levers or LIB.LEVERS
    rng = random.Random(a.seed)
    node = os.environ.get("SLURMD_NODENAME", "")
    for c in cases:
        if c["id"] in done:
            continue
        inflight.write_text(c["id"])
        t0 = time.time()
        try:
            res = run_case(rt, cs, c.get("levers", levers), data, clk, c, rng)
        except Exception as e:  # noqa: BLE001
            res = {"case": c, "error": repr(e)[:2000]}
            msg = repr(e)
            with outp.open("a") as f:
                f.write(json.dumps({**res, "node": node, "bdf": bdf, "job": os.environ.get("SLURM_JOB_ID")},
                                   default=str) + "\n")
            inflight.unlink()
            if "hip" in msg.lower() or "HIP" in msg:
                return 86
            continue
        res.update({"node": node, "bdf": bdf, "job": os.environ.get("SLURM_JOB_ID"), "s": round(time.time() - t0, 1)})
        with outp.open("a") as f:
            f.write(json.dumps(res, default=str) + "\n")
        inflight.unlink()
        print(f"[sk] {c['id']} done in {res['s']} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
