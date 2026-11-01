"""RT: the rocKE B3 stage kernel, direct code objects vs COMGR, on gfx950 (q38 container, ATOM rig).

    python rt_b3.py --variant r4s --which tuned --arms D,Dpl=dppmirror+dppcombine --rounds 24 --out DIR

Built on ``gdn-b3/b3_rig.py`` (``B3Rig``: the mono DeltaNet layer fills B3's inputs and the reference
outputs and state; an isolated B3 launch is compared with them). Arms, all the SAME ``KernelDef``
(``rt_builds.b3``):

  C    COMGR (the hybrid's B3: what ships)          C2  C's bytes loaded again (byte-copy null)
  R    COMGR recompiled in a fresh process            D   the direct compiler (lifted IR)
  <name>=<levers>   the direct compiler in a fresh process with ROCKE_DIRECT_LEVERS=<levers> (+-joined)

Numerics first (every arm): B3's outputs (``core`` / ``d8`` / ``de``) and the recurrent state against the
mono layer's, bitwise; a non-bitwise arm reports its state drift (max abs / rel) and payload diffs and is
judged against the tolerance gate (``--tol-state``) instead. Determinism: each direct arm is launched
twice from the same inputs and must give the same bits. Timing (only when every arm passed): per arm a
HIP graph of ``--launches`` launches over ``--copies`` cold weight / state copies, one replay per arm per
round in a shuffled order, ``--rounds`` rounds; per-round ratio vs C; the DESIGN 9.1 win rule against both
controls (C2, R). Absolute times only in ``DIR/raw.json`` (private); ratios in ``DIR/ratios.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def fresh_compile(label: str, backend: str, levers: str, out: str) -> bytes:
    """``levers``: ``ROCKE_DIRECT_LEVERS`` (``+``-joined), optionally ``@`` and the build's lever scopes
    (``RT_PHASE_LEVERS``: the fused MoE's ``phase_levers``, e.g. ``dn=nofence+loadorder;combine=-nofence``)."""
    levers, _, scopes = levers.partition("@")
    env = dict(os.environ, ROCKE_DIRECT_LEVERS=levers.replace("+", ","), RT_PHASE_LEVERS=scopes)
    subprocess.run([sys.executable, os.path.join(HERE, "rt_compile.py"), label, backend, out], check=True, env=env)
    return open(out, "rb").read()


def compile_arms(label: str, specs: list[tuple[str, str]], out_dir: str, prefix: str = "") -> dict[str, bytes]:
    """Direct lever arms (``(name, levers)``), each compiled in its own fresh process, ``RT_COMPILE_JOBS`` at once."""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(int(os.environ.get("RT_COMPILE_JOBS", "8"))) as ex:
        futs = [(n, ex.submit(fresh_compile, label, "direct", lv, os.path.join(out_dir, f"{prefix}{n}.hsaco"))) for n, lv in specs]
        return {n: f.result() for n, f in futs}


def dedupe(arts: dict, keys: dict | None = None, keep=("C", "C2", "R")) -> dict[str, str]:
    """Byte-identical direct arms are measured once: ``{alias: the first arm with the same bytes}``.

    A lever that finds nothing to rewrite in a kernel gives the same code object; timing it again would only
    spend rounds (and a screening design's runs). The alias inherits the measured arm's numerics and timing.
    """
    seen, alias = {}, {}
    for n in arts:
        if n in keep:
            continue
        k = keys[n] if keys else arts[n]
        if k in seen:
            alias[n] = seen[k]
        else:
            seen[k] = n
    return alias


def arm_specs(arms: str) -> list[tuple[str, str]]:
    """``D,NAME=LEVERS,...`` -> [(name, levers)] (D: no levers, the plain direct object)."""
    out = []
    for s in [x.strip() for x in arms.split(",") if x.strip()]:
        name, _, levers = s.partition("=")
        out.append((name, levers if s != "D" else ""))
    return out


def drop_failing(failed: set[str], keep=("C", "C2", "R")) -> tuple[bool, list[str]]:
    """``RT_DROP_FAILING=1`` (screening): arms that fail numerics leave the timing instead of stopping it.

    Returns (timing may go on, dropped arms). A failing control (or with the env unset, any failure) stops it.
    """
    if not failed:
        return True, []
    if os.environ.get("RT_DROP_FAILING", "0") != "1" or failed & set(keep):
        return False, []
    return True, sorted(failed)


def win_rule(ratios: list[float], controls: list[list[float]]) -> str:
    med = statistics.median(ratios)
    below = sum(r < 1 for r in ratios) / len(ratios)
    lo = min(min(c) for c in controls)
    hi = max(max(c) for c in controls)
    if med < lo - 0.002 and below >= 0.75:
        return "WIN"
    if med > hi + 0.002 and below <= 0.25:
        return "LOSS"
    return "neutral"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="r1")
    ap.add_argument("--which", default="tuned")
    ap.add_argument("--arms", default="D", help="comma list: D and NAME=LEVERS (direct with levers)")
    ap.add_argument("--rounds", type=int, default=24)
    ap.add_argument("--launches", type=int, default=96)
    ap.add_argument("--copies", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--placements", type=int, default=4, help="module + graph copies per arm, rotated per round")
    ap.add_argument("--no-time", action="store_true")
    ap.add_argument("--tol-state", type=float, default=1e-4, help="max relative state drift of a non-bitwise arm")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    import rt_builds
    from b3_rig import B3Rig
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime.hip_module import Runtime

    from kernels.gfx950.qwen38_mono import stage_b3 as SB
    from kernels.gfx950.qwen38_mono import stage_fn as SF

    rows, seq, group = SF.parse_variant(a.variant)
    label = f"b3_{a.variant}_{a.which}"
    t0 = time.time()
    br = B3Rig(rows, seq, copies=a.copies, seed=a.seed, group=group)
    br.fill()
    kern = rt_builds.build(label)
    knobs = SB.knobs_from_spec(SB.TUNED_SPECS[a.variant]) if a.which == "tuned" else SB.B3Knobs()
    internal = knobs.split == 1 or knobs.xchg != "core"
    rt = Runtime()
    keep = []

    def loader(hsaco: bytes, name: str):
        mod = rt.load_module(hsaco)
        keep.append(mod)
        return mod.get_function(name)

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
    fns = {n: loader(h, kern.name) for n, h in arts.items()}
    # ``--placements`` separately loaded modules per arm, each with its own graph (timing): byte-identical
    # arms differed by up to 1.5% with one graph each (rt-g1), so every round replays every placement of
    # every arm (shuffled) and an arm's round time is the mean over its placements.
    order = list(arts)
    random.Random(99 + a.seed).shuffle(order)
    placed = {n: [] for n in arts}
    for k in range(a.placements):
        for n in order:
            placed[n].append(fns[n] if k == 0 else loader(arts[n], kern.name))

    def launcher(fn):
        def go(i, slot, stream=None):
            r = br.rig.runner
            import torch

            from b3_rig import pack_args

            stream = stream or torch.cuda.current_stream()
            args = pack_args(r.counter.data_ptr(), slot, r.scratch.data_ptr(), br.args_of(i))
            keep.append(args)
            rt.launch(fn, (256, 1, 1), (512, 1, 1), args, stream=int(stream.cuda_stream))

        return go

    launches = {n: launcher(f) for n, f in fns.items()}
    # ---- numerics
    checks, ok = {}, True
    for n, go in launches.items():
        res = br.check(go, n, internal_core=internal)
        res2 = br.check(go, n, internal_core=internal)
        same = all(res[k]["words_diff"] == res2[k]["words_diff"] for k in ("core", "d8", "de"))
        res["deterministic"] = same and res["state"] == res2["state"]
        tol_ok = all(s["max_rel"] <= a.tol_state for s in res["state"])
        res["pass"] = bool(res["deterministic"] and (res["bitwise"] or tol_ok))
        checks[n] = res
        ok &= res["pass"]
        print("CHECK", label, n, "bitwise" if res["bitwise"] else "NOT-bitwise", "det" if res["deterministic"] else "NONDET",
              "PASS" if res["pass"] else "FAIL",
              json.dumps({k: res[k] for k in ("core", "d8", "de", "d8_payload_diff", "de_payload_diff")}),
              json.dumps(res["state"]), flush=True)  # fmt: skip
    for n, t in alias.items():
        checks[n] = {"same_as": t, "pass": checks[t]["pass"]}
    out = {"label": label, "kernel": kern.name, "checks": checks, "node": os.uname().nodename, "alias": alias}
    go, dropped = drop_failing({n for n, c in checks.items() if not c["pass"]})
    for n in dropped:
        print("DROPPED", label, n, "(numerics failed)", flush=True)
        for t in [n] + [x for x, y in alias.items() if y == n]:
            arts.pop(t, None)
            alias.pop(t, None)
    out["dropped"] = dropped
    if a.no_time or not go:
        out["timing"] = "skipped" if a.no_time else "numerics failed"
        json.dump(out, open(os.path.join(a.out, "ratios.json"), "w"), indent=1, default=str)
        print("NUMERICS", "PASS" if ok else "FAIL", flush=True)
        return 0 if ok else 1
    if dropped:
        print("NUMERICS FAIL (dropped arms, timing the rest)", flush=True)
    # ---- timing: shuffled interleaved rounds
    import torch

    graphs = {n: [] for n in arts}
    cap = [(n, k) for k in range(a.placements) for n in arts]
    random.Random(7 + a.seed).shuffle(cap)
    for n, k in cap:
        graphs[n].append(br.graph(launcher(placed[n][k]), a.launches))
    for gs in graphs.values():
        for g in gs:
            for _ in range(2):
                g.replay()
    torch.cuda.synchronize()
    raw = {n: [] for n in graphs}
    rng = random.Random(1234 + a.seed)
    slots = [(n, k) for n in graphs for k in range(a.placements)]
    for rnd in range(a.rounds):
        rng.shuffle(slots)
        cur = {n: [] for n in graphs}
        for n, k in slots:
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            graphs[n][k].replay()
            e1.record()
            torch.cuda.synchronize()
            cur[n].append(e0.elapsed_time(e1) * 1e3 / a.launches)
        for n in graphs:
            raw[n].append(statistics.mean(cur[n]))  # the arm's round time: mean over its placements
    ratios = {n: [x / c for x, c in zip(raw[n], raw["C"])] for n in raw if n != "C"}
    controls = [ratios["C2"], ratios["R"]]
    # vs the pooled COMGR arms (mean of C, C2, R per round): placement-robust view
    pooled = [statistics.mean(v) for v in zip(raw["C"], raw["C2"], raw["R"])]
    summary = {}
    for n, rs in ratios.items():
        verdict = win_rule(rs, controls) if n not in ("C2", "R") else "control"
        vp = [x / p for x, p in zip(raw[n], pooled)]
        summary[n] = {"median": round(statistics.median(rs), 4), "min": round(min(rs), 4), "max": round(max(rs), 4),
                      "below1": f"{sum(r < 1 for r in rs)}/{len(rs)}", "verdict": verdict,
                      "vs_pooled": round(statistics.median(vp), 4)}  # fmt: skip
        print("TIME", label, n, json.dumps(summary[n]), flush=True)
    for n, t in alias.items():
        summary[n] = dict(summary[t], same_as=t)
        print("TIME", label, n, json.dumps(summary[n]), flush=True)
    out["timing"] = summary
    out["alias"] = alias
    out["rounds"] = a.rounds
    out["wall_s"] = round(time.time() - t0, 1)
    json.dump(out, open(os.path.join(a.out, "ratios.json"), "w"), indent=1, default=str)
    json.dump({"raw_us": raw}, open(os.path.join(a.out, "raw.json"), "w"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
