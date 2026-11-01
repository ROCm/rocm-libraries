"""RT: the rocKE fused MoE suffix (M3 router + M4 + M5), direct code objects vs COMGR (q38 container, ATOM rig).

    python rt_fmoe.py --rows 4 --arms D,Ddpp=dppmirror+dppcombine --rounds 20 --out DIR

Built on ``moe-fused/fused_ab.py`` / ``moe/moe_ab.py`` (``Cell``: a mono layer of ``--kind`` cut after M2 by a
FlyDSL prefix; every timed launch is the prefix then the fused suffix, ``moe = pair - pre`` per round).
The suffix is ``moe_fused.build_fused_suffix(rows, stage_moe_fused.best_knobs(rows, "best_plain"))`` (the
router inside: ``rt_builds.fmoe``), compiled as:

  C    COMGR (what ships)   C2  C's bytes again (byte-copy null)   R  COMGR recompiled in a fresh process
  D    the direct compiler (lifted IR)                             NAME=LEVERS  direct with ROCKE_DIRECT_LEVERS

Numerics first: per case (``--check-cases`` inputs, a fresh step each) every arm runs from one snapshot;
``h_out`` and the ``lg`` / ``sg`` / ``dn`` payloads are compared bitwise with the FlyDSL layer (``full``)
and with C, and twice for determinism. A non-bitwise arm passes only within ``--tol`` of ``full``'s
``h_out`` (max abs diff / max |h_out|). Timing as ``fused_ab.bench`` (shuffled rounds, ``--reps`` replays per
arm per round, median), ratios per round of the MoE part ``(arm - pre)`` vs C's, controls C2 and R.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="delta", choices=("delta", "delta_ple", "qsa"))
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--arms", default="D")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--launches", type=int, default=96)
    ap.add_argument("--copies", type=int, default=8)
    ap.add_argument("--inputs", type=int, default=4)
    ap.add_argument("--check-cases", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--placements", type=int, default=3)
    ap.add_argument("--tol", type=float, default=2e-2)
    ap.add_argument("--no-time", action="store_true")
    ap.add_argument("--stamps", type=int, default=0,
                    help="N > 0: build every arm with the phase stamps (label fmoe_r<rows>_st) and, instead of the "
                         "graph bench, run N eager stamped launches per arm (shuffled): per-phase time vs C")  # fmt: skip
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    fdir = os.path.join(os.path.dirname(HERE), "moe-fused")
    sys.path.insert(0, fdir)
    import fused_ab as FA
    import moe_ab as MA
    import torch
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig, no_fence

    import rt_builds
    from kernels.gfx950.qwen38_mono import abi as A
    from kernels.gfx950.qwen38_mono import moe_fused as MF
    from rt_b3 import arm_specs, compile_arms, dedupe, drop_failing, fresh_compile, win_rule

    a = SimpleNamespace(
        kind=args.kind, rows=args.rows, seq=args.rows > 1, experts=512, stages="m3,m4,m5", routing="random",
        copies=args.copies, inputs=args.inputs, check_cases=args.check_cases, context=2048, share_experts=False,
        launches=args.launches, reps=args.reps, rounds=args.rounds, warmup=3, row_corr=0.0, seed=args.seed,
        stamps=False, hyb_stages="m3,m4,m5",
    )  # fmt: skip
    t0 = time.time()
    cell = MA.Cell(a)
    label = f"fmoe_r{args.rows}" + ("_st" if args.stamps else "")
    kern = rt_builds.build(label)
    arts = {}
    c_art = compile_kernel(kern, arch="gfx950", backend="python", code_object_backend="comgr")
    arts["C"] = c_art.hsaco
    arts["C2"] = bytes(c_art.hsaco)
    arts["R"] = fresh_compile(label, "comgr", "", os.path.join(args.out, "R.hsaco"))
    d_art = compile_kernel(rt_builds.build(label), arch="gfx950", backend="python", code_object_backend="direct")
    if getattr(d_art, "code_object_backend", "") != "direct":
        print("DIRECT REFUSED", getattr(d_art, "direct_fallback", ""), flush=True)
        return 2
    specs = arm_specs(args.arms)
    built = compile_arms(label, [(n, lv) for n, lv in specs if n != "D"], args.out)
    for n, _ in specs:
        arts[n] = d_art.hsaco if n == "D" else built[n]
    for n, h in arts.items():
        with open(os.path.join(args.out, f"{label}.{n}.hsaco"), "wb") as fh:
            fh.write(h)
    alias = dedupe(arts)
    for n, t in alias.items():
        del arts[n]
        print("ALIAS", label, n, "=", t, flush=True)

    def launcher(hsaco, name):
        lau = KernelLauncher(hsaco=hsaco, kernel_name=kern.name, signature=MF.signature())

        def launch(step, layer, scratch, *xs, stream):
            vals = {"step": step, "layer": layer, "scratch": scratch}
            vals.update(zip(A.LAYER_ARGS[1:], xs))
            with no_fence():
                lau(vals, config=LaunchConfig(grid=(A.GRID, 1, 1), block=(A.THREADS, 1, 1), stream=stream.cuda_stream))

        launch.name = name
        launch.compile_s = 0.0
        return launch

    for n, h in arts.items():
        cell.launch[n] = launcher(h, n)
    names = list(arts)
    # Timing placements: every arm loaded ``--placements`` times (separate modules, separate graphs in bench);
    # an arm's round time is the mean over its placements (byte-identical B3 arms differed by up to 1.5% with one).
    placed = {n: [n] + [f"{n}@{k}" for k in range(1, args.placements)] for n in names}
    for n in names:
        for pn in placed[n][1:]:
            cell.launch[pn] = launcher(arts[n], pn)
    # ---- numerics
    stream = torch.cuda.current_stream()
    r = cell.r
    regions = ["lg", "sg", "dn"]
    live = MA.state_tensors(cell.rig)
    ok, cases = True, []
    failed = set()
    for j in range(min(a.inputs, a.check_cases)):
        slot = j % FA.SLOTS
        c, _ = cell.args(j)
        r.counter.add_(1)
        r.scratch.zero_()
        torch.cuda.synchronize()
        snap = {k: t.clone() for k, t in live.items()}
        got = {}
        for arm in ["full"] + names + [n for n in names if n not in ("C2", "R")]:
            for k, t in live.items():
                t.copy_(snap[k])
            cell.outs[c].zero_()
            torch.cuda.synchronize()
            FA.pair(cell, arm, j, slot, stream)
            torch.cuda.synchronize()
            res = (cell.outs[c].clone(), {n: MA.payload(MA.words_of(r.scratch), n, a.rows) for n in regions})
            if arm in got:
                got[arm + "#2"] = res
            else:
                got[arm] = res
        ref_h, ref_m = got["full"]
        scale = float(ref_h.float().abs().max().clamp(min=1e-30))
        case = {"case": j, "U": MA.lg_unique(r.scratch, a.rows, a.experts), "arms": {}}
        for arm in names:
            h, m = got[arm]
            h2, m2 = got.get(arm + "#2", got[arm])
            ch, cm = got["C"]
            e = {
                "h_bitwise_full": bool(torch.equal(h, ref_h)),
                "h_bitwise_C": bool(torch.equal(h, ch)),
                "h_rel_full": float((h.float() - ref_h.float()).abs().max()) / scale,
                "regions_bitwise_C": all(bool(torch.equal(m[n], cm[n])) for n in regions),
                "deterministic": bool(torch.equal(h, h2)) and all(bool(torch.equal(m[n], m2[n])) for n in regions),
                "nonzero": bool((h != 0).any()),
            }
            e["pass"] = e["deterministic"] and e["nonzero"] and (e["h_bitwise_full"] or e["h_bitwise_C"] or e["h_rel_full"] <= args.tol)
            ok &= e["pass"]
            if not e["pass"]:
                failed.add(arm)
            case["arms"][arm] = e
        cases.append(case)
        print("CHECK", label, json.dumps(case), flush=True)
    out = {"label": label, "kernel": kern.name, "cases": cases, "node": os.uname().nodename, "pass": bool(ok), "alias": alias}
    go, dropped = drop_failing(failed)
    for n in dropped:
        print("DROPPED", label, n, "(numerics failed)", flush=True)
        for t in [n] + [x for x, y in alias.items() if y == n]:
            alias.pop(t, None)
        names.remove(n)
    out["dropped"] = dropped
    if args.no_time or not go:
        print("NUMERICS", "PASS" if ok else "FAIL", flush=True)
        json.dump(out, open(os.path.join(args.out, "ratios.json"), "w"), indent=1)
        return 0 if ok else 1
    if args.stamps:
        out.update(phase_stamps(cell, names, args, FA, MF, A, torch))
        json.dump(out, open(os.path.join(args.out, "ratios.json"), "w"), indent=1)
        return 0
    # ---- timing (fused_ab.bench: full, full2, pre + our arms; we read its per-round medians)
    cell.launch["full2"] = cell.launch["full"]
    timed = [pn for n in names for pn in placed[n]]
    random.Random(5 + args.seed).shuffle(timed)
    b = FA.bench(cell, [(pn, None) for pn in timed])
    t = b["us_per_launch"]
    for n in names:
        t[n] = [statistics.mean(v) for v in zip(*(t[pn] for pn in placed[n]))]
    moe = {n: [x - p for x, p in zip(t[n], t["pre"])] for n in names}
    ratios = {n: [x / cc for x, cc in zip(moe[n], moe["C"])] for n in names if n != "C"}
    controls = [ratios["C2"], ratios["R"]]
    summ = {}
    for n, rs in ratios.items():
        summ[n] = {"median": round(statistics.median(rs), 4), "min": round(min(rs), 4), "max": round(max(rs), 4),
                   "below1": f"{sum(x < 1 for x in rs)}/{len(rs)}",
                   "verdict": "control" if n in ("C2", "R") else win_rule(rs, controls)}  # fmt: skip
        print("TIME", label, n, json.dumps(summ[n]), flush=True)
    for n, t in alias.items():
        summ[n] = dict(summ[t], same_as=t)
        print("TIME", label, n, json.dumps(summ[n]), flush=True)
    out["timing"] = summ
    out["alias"] = alias
    out["U_mean"] = b["U_mean"]
    out["wall_s"] = round(time.time() - t0, 1)
    json.dump(out, open(os.path.join(args.out, "ratios.json"), "w"), indent=1)
    json.dump({"us_per_launch": t}, open(os.path.join(args.out, "raw.json"), "w"))
    return 0


def phase_stamps(cell, names, args, FA, MF, A, torch) -> dict:
    """Eager stamped launches (``--stamps`` per arm, arms shuffled per iteration): thread 0 of every CTA stamps
    ``MF.PHASES`` with ``s_memrealtime``. Per launch and phase: the median over CTAs of the stamp minus the
    launch's earliest entry stamp (``at``) and of the CTA's own step from the previous phase (``step``); per
    arm the median over launches. Ratios vs C only (the absolute ticks go to raw.json)."""
    r = cell.r
    stream = torch.cuda.current_stream()
    base = (A.SC["ss"] + MF.STAMP_OFF) // 8
    npts = MF.STAMP_PTS
    region = r.scratch.view(torch.int64)[base: base + A.GRID * npts]
    P = list(MF.PHASES)
    per = {n: {"at": {p: [] for p in P}, "step": {p: [] for p in P}, "last": {p: [] for p in P}} for n in names}
    # per CTA and phase, every launch's stamp minus the launch's first entry (the critical path's CTAs; private)
    cta = {n: {p: [[] for _ in range(A.GRID)] for p in P} for n in names}
    rng = random.Random(77 + args.seed)
    order = list(names)
    for it in range(args.stamps):
        rng.shuffle(order)
        for n in order:
            j = it % args.copies
            r.counter.add_(1)
            region.zero_()
            torch.cuda.synchronize()
            FA.pair(cell, n, j, 0, stream)
            torch.cuda.synchronize()
            st = region.view(A.GRID, npts)[:, : len(P)].cpu().double()
            seen = st != 0
            if not seen[:, 0].any():
                continue
            t0 = st[:, 0][seen[:, 0]].min()
            for i, p in enumerate(P):
                ok = seen[:, i]
                if ok.any():
                    per[n]["at"][p].append(float((st[:, i][ok] - t0).median()))
                    per[n]["last"][p].append(float((st[:, i][ok] - t0).max()))
                    for c in ok.nonzero().flatten().tolist():
                        cta[n][p][c].append(float(st[c, i] - t0))
                if i:
                    ok2 = ok & seen[:, i - 1]
                    if ok2.any():
                        per[n]["step"][p].append(float((st[:, i][ok2] - st[:, i - 1][ok2]).median()))
    med = {n: {k: {p: (statistics.median(v) if v else None) for p, v in d.items()} for k, d in per[n].items()} for n in names}
    ratios = {}
    for n in names:
        if n == "C":
            continue
        ratios[n] = {k: {p: (round(med[n][k][p] / med["C"][k][p], 4) if med[n][k][p] is not None and med["C"][k][p] else None)
                         for p in P} for k in ("at", "step", "last")}  # fmt: skip
        print("STAMPS", n, json.dumps(ratios[n]), flush=True)
    per_cta = {n: {p: [statistics.median(v) if v else None for v in cta[n][p]] for p in P} for n in names}
    json.dump({"stamp_ticks": med, "per_cta": per_cta}, open(os.path.join(args.out, "raw.json"), "w"), indent=1)
    return {"stamps": ratios, "stamp_iters": args.stamps}


if __name__ == "__main__":
    sys.exit(main())
