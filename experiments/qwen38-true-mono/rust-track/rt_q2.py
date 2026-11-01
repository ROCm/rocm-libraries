"""RT: the rocKE Q2 stage in the true-mono framework layer kernel, direct code objects vs COMGR, gfx950 (container).

    python rt_q2.py --variant r1 --arms D,Dl=litreuse+litshare --rounds 20 --out DIR

The kernel: ``rt_builds`` label ``fw_q2_<variant>``: ``framework.build_layer("qsa", stages=("q2",))`` with the Q2 stage
function through ``stage_adapt`` on ATOM's own layout (``mono_layout.from_atom``) and schedule, the mono layer
kernel's ABI (``step_ptr, layer, *LAYER_ARGS``). The standalone ``stage_fn.build_stage_kernel`` (labels ``q2_*``)
takes one task per launch on a test layout, so it cannot run on a ``MonoRig``'s scratch; the framework layer
kernel runs Q2's whole task schedule on the mono's scratch, as the true mono does.

Rig (``harness/rig.py`` ``MonoRig``, as ``hybrid/ab_extern.py``): the four base layers plus ``--copies`` QSA
layers (cold weights and caches per copy), random HC input rows, distinct cache lengths per batched row. One
launch of an arm = the FlyDSL prefix ``a1 .. q1`` of the QSA layer (``pre``) then the arm's Q2 (same tag), as
``moe_ab``'s split arms; ``fly`` = ATOM's own Q2 (``_build_layer(("q2",))``).

Arms (all the SAME ``KernelDef``): ``C`` COMGR, ``C2`` C's bytes loaded again, ``R`` a fresh-process COMGR compile,
``D`` the direct compiler, ``NAME=LEVERS[@SCOPES]`` direct with ``ROCKE_DIRECT_LEVERS`` (``rt_b3.fresh_compile``).

Numerics first, every arm, ``--check-cases`` cases: from one snapshot after the prefix (scratch, the copy's QSA
caches and its output row), the arm's Q2 must give the same bytes as ``C`` everywhere, twice (determinism), and
write the ``qn`` region with nonzero payloads; ``bitwise_fly`` (equal to ATOM's own Q2) is reported. A failing
arm stops the run, or with ``RT_DROP_FAILING=1`` leaves the timing (``rt_tune.py``).

Timing: per (arm, placement) a CUDA graph of ``--launches`` launch pairs (prefix + Q2) over the copies, plus the
prefix alone (``pre``); every round replays every graph once in a shuffled order; an arm's round time is the
mean over its placements minus ``pre``; per-round ratios vs ``C``; win rule against both controls. Absolute
times only in ``DIR/raw.json`` (private; ``raw_us`` = Q2 time per launch); ratios in ``DIR/ratios.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for d in (os.path.join(os.path.dirname(HERE), "harness"), os.path.join(HERE, "..", "harness")):
    if os.path.isdir(d):
        sys.path.insert(0, d)

BASE_KINDS = ["delta", "delta_ple", "delta", "qsa"]
ROW_OFFSETS = (0, 7, 30, 61)


def fw_signature(layer_args):
    return [{"name": "step_ptr", "type": "i64"}, {"name": "layer", "type": "i32"}] + [
        {"name": n, "type": "i64"} for n in layer_args
    ]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="r1", help="r1 r2 r3 r4 r2s r3s r4s")
    ap.add_argument("--arms", default="D")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--launches", type=int, default=32)
    ap.add_argument("--copies", type=int, default=8)
    ap.add_argument("--check-cases", type=int, default=2)
    ap.add_argument("--placements", type=int, default=3)
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--experts", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-time", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    import rt_builds
    from rocke.helpers.compile import compile_kernel
    from rt_b3 import arm_specs, compile_arms, dedupe, drop_failing, fresh_compile, win_rule

    from kernels.gfx950.qwen38_mono import abi as A
    from kernels.gfx950.qwen38_mono import stage_fn as SF
    from kernels.gfx950.qwen38_mono.framework import ATOM_LAYER_ARGS

    rows, seq, _group = SF.parse_variant(a.variant)
    label = f"fw_q2_{a.variant}"
    t0 = time.time()
    # ---- every code object before HIP is touched (fresh-process compiles)
    kern = rt_builds.build(label)
    arts = {}
    c_art = compile_kernel(kern, arch="gfx950", backend="python", code_object_backend="comgr")
    arts["C"] = c_art.hsaco
    arts["C2"] = bytes(c_art.hsaco)
    arts["R"] = fresh_compile(label, "comgr", "", os.path.join(a.out, "R.hsaco"))
    d_art = compile_kernel(rt_builds.build(label), arch="gfx950", backend="python", code_object_backend="direct")
    if getattr(d_art, "code_object_backend", "") != "direct":
        print("DIRECT REFUSED", getattr(d_art, "direct_fallback", ""), flush=True)
        return 2
    specs = arm_specs(a.arms)
    built = compile_arms(label, [(n, lv) for n, lv in specs if n != "D"], a.out)
    for n, _ in specs:
        arts[n] = d_art.hsaco if n == "D" else built[n]
    for n, h in arts.items():
        with open(os.path.join(a.out, f"{label}.{n}.hsaco"), "wb") as fh:
            fh.write(h)
    alias = dedupe(arts)
    for n, t in alias.items():
        del arts[n]
        print("ALIAS", label, n, "=", t, flush=True)

    import torch
    from rig import MonoRig
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig, no_fence

    from atom.models.qwen4_exp_mono.kernels import kernel as K
    from atom.models.qwen4_exp_mono.kernels import layout as LY

    assert ATOM_LAYER_ARGS[0] == "scratch" and tuple(ATOM_LAYER_ARGS) == tuple(K.LAYER_ARGS), "layer ABI drifted"
    sig = fw_signature(ATOM_LAYER_ARGS)

    def launcher(hsaco):
        lau = KernelLauncher(hsaco=hsaco, kernel_name=kern.name, signature=sig)

        def launch(step, layer, scratch, *xs, stream):
            vals = {"step_ptr": step, "layer": layer, "scratch": scratch}
            vals.update(zip(ATOM_LAYER_ARGS[1:], xs))
            with no_fence():
                lau(vals, config=LaunchConfig(grid=(A.GRID, 1, 1), block=(A.THREADS, 1, 1), stream=stream.cuda_stream))

        return launch

    # ---- rig
    kinds = BASE_KINDS + ["qsa"] * a.copies
    idx = list(range(len(BASE_KINDS), len(kinds)))
    rig = MonoRig(kinds, experts=a.experts, context=a.context, spec_tokens=rows - 1 if seq else 0, seed=a.seed)
    s = rig.step(rows)
    if not seq and rows > 1:
        offs = torch.tensor(ROW_OFFSETS[:rows], dtype=s.positions.dtype, device=s.positions.device)
        s.positions = s.positions + offs
    rig.meta(s)
    gen = torch.Generator(device=rig.dev)
    gen.manual_seed(1234 + a.seed)
    for L in idx:
        hin = rig.h_in(s, L)
        hin.copy_(torch.randn(hin.shape, generator=gen, device=hin.device).to(hin.dtype))
    r = rig.runner
    order = K.STAGES["qsa"]
    prefix = tuple(order[: order.index("q2")])
    pre = K._build_layer(prefix, a.experts, rows, seq, 0, False)
    fly = K._build_layer(("q2",), a.experts, rows, seq, 0, False)
    names = list(arts)
    placed = {n: [launcher(arts[n]) for _ in range(a.placements)] for n in names}
    launch = {"fly": fly, **{n: placed[n][0] for n in names}}

    def pair(arm_fn, L, stream):
        cp, sp = r.counter.data_ptr(), r.scratch.data_ptr()
        pre(cp, L, sp, *s.w.args[L], stream=stream)
        if arm_fn is not None:
            arm_fn(cp, L, sp, *s.w.args[L], stream=stream)

    def live_of(L):
        out = {"scratch": r.scratch, "h_out": s.w.chain[L]}
        for k, t in rig.state.qsa[L].items():
            if isinstance(t, torch.Tensor):
                out["qsa." + k] = t
        return out

    # ---- numerics
    stream = torch.cuda.current_stream()
    ok, cases, failed = True, [], set()
    qn = LY.SC["qn"]
    for j in range(a.check_cases):
        L = idx[j % len(idx)]
        r.counter.add_(1)
        r.scratch.zero_()
        torch.cuda.synchronize()
        pair(None, L, stream)  # the prefix: Q2's inputs under this tag
        torch.cuda.synchronize()
        live = live_of(L)
        snap = {k: t.clone() for k, t in live.items()}
        got = {}
        for arm in ["fly"] + names + names:
            for k, t in live.items():
                t.copy_(snap[k])
            torch.cuda.synchronize()
            launch[arm](r.counter.data_ptr(), L, r.scratch.data_ptr(), *s.w.args[L], stream=stream)
            torch.cuda.synchronize()
            res = {k: t.clone() for k, t in live.items()}
            got[arm + "#2" if arm in got else arm] = res
        case = {"case": j, "layer": L, "arms": {}}
        eq = lambda x, y: all(torch.equal(x[k], y[k]) for k in x)  # noqa: E731
        for arm in names:
            g, g2 = got[arm], got[arm + "#2"]
            w = g["scratch"].view(torch.int32)[qn // 4: qn // 4 + 4096]
            e = {
                "bitwise_C": eq(g, got["C"]),
                "bitwise_fly": eq(g, got["fly"]),
                "deterministic": eq(g, g2),
                "qn_written": not torch.equal(g["scratch"].view(torch.uint8)[qn: qn + 4096],
                                              snap["scratch"].view(torch.uint8)[qn: qn + 4096]),
                "qn_payload_nonzero": int((w[0::2] != 0).sum()),
            }  # fmt: skip
            if not e["bitwise_C"]:
                e["differ"] = sorted(k for k in g if not torch.equal(g[k], got["C"][k]))[:8]
            e["pass"] = e["bitwise_C"] and e["deterministic"] and e["qn_written"] and e["qn_payload_nonzero"] > 0
            ok &= e["pass"]
            if not e["pass"]:
                failed.add(arm)
            case["arms"][arm] = e
        cases.append(case)
        print("CHECK", label, json.dumps(case), flush=True)
    out = {"label": label, "kernel": kern.name, "cases": cases, "node": os.uname().nodename, "pass": bool(ok),
           "alias": alias}  # fmt: skip
    go, dropped = drop_failing(failed)
    for n in dropped:
        print("DROPPED", label, n, "(numerics failed)", flush=True)
        for t in [n] + [x for x, y in alias.items() if y == n]:
            alias.pop(t, None)
        names.remove(n)
    out["dropped"] = dropped
    if a.no_time or not go:
        print("NUMERICS", "PASS" if ok else "FAIL", flush=True)
        json.dump(out, open(os.path.join(a.out, "ratios.json"), "w"), indent=1)
        return 0 if ok else 1
    if dropped:
        print("NUMERICS FAIL (dropped arms, timing the rest)", flush=True)

    # ---- timing
    def run_all(fn, st):
        for i in range(a.launches):
            pair(fn, idx[i % len(idx)], st)
            if i % len(idx) == len(idx) - 1:
                r.counter.add_(1)

    cells = [("pre", 0, None)] + [(n, k, placed[n][k]) for n in names for k in range(a.placements)]
    random.Random(7 + a.seed).shuffle(cells)
    graphs = {}
    cs = torch.cuda.Stream()
    for n, k, fn in cells:
        g = torch.cuda.CUDAGraph()
        cs.wait_stream(stream)
        with torch.cuda.stream(cs):
            run_all(fn, cs)
        torch.cuda.synchronize()
        with torch.cuda.graph(g, stream=cs):
            run_all(fn, cs)
        torch.cuda.synchronize()
        graphs[(n, k)] = g
    for g in graphs.values():
        g.replay()
    torch.cuda.synchronize()
    raw = {n: [] for n in ["pre"] + names}
    rng = random.Random(1234 + a.seed)
    slots = list(graphs)
    for _ in range(a.rounds):
        rng.shuffle(slots)
        cur = {n: [] for n in raw}
        for key in slots:
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            graphs[key].replay()
            e1.record()
            torch.cuda.synchronize()
            cur[key[0]].append(e0.elapsed_time(e1) * 1e3 / a.launches)
        for n in raw:
            raw[n].append(statistics.mean(cur[n]))
    q2 = {n: [x - p for x, p in zip(raw[n], raw["pre"])] for n in names}
    ratios = {n: [x / c for x, c in zip(q2[n], q2["C"])] for n in names if n != "C"}
    controls = [ratios["C2"], ratios["R"]]
    pooled = [statistics.mean(v) for v in zip(q2["C"], q2["C2"], q2["R"])]
    summary = {}
    for n, rs in ratios.items():
        verdict = win_rule(rs, controls) if n not in ("C2", "R") else "control"
        vp = [x / p for x, p in zip(q2[n], pooled)]
        summary[n] = {"median": round(statistics.median(rs), 4), "min": round(min(rs), 4), "max": round(max(rs), 4),
                      "below1": f"{sum(x < 1 for x in rs)}/{len(rs)}", "verdict": verdict,
                      "vs_pooled": round(statistics.median(vp), 4)}  # fmt: skip
        print("TIME", label, n, json.dumps(summary[n]), flush=True)
    for n, t in alias.items():
        summary[n] = dict(summary[t], same_as=t)
        print("TIME", label, n, json.dumps(summary[n]), flush=True)
    out["timing"] = summary
    out["alias"] = alias
    out["rounds"] = a.rounds
    out["pre_share"] = round(statistics.median(raw["pre"]) / statistics.median([p + q for p, q in zip(raw["pre"], q2["C"])]), 3)
    out["wall_s"] = round(time.time() - t0, 1)
    json.dump(out, open(os.path.join(a.out, "ratios.json"), "w"), indent=1, default=str)
    json.dump({"raw_us": q2, "pre_us": raw["pre"]}, open(os.path.join(a.out, "raw.json"), "w"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
