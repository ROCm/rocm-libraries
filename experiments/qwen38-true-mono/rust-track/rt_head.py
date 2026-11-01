"""RT: the MXFP4 lm_head GEMV (``mx_decode_gemv``), direct code objects vs COMGR, gfx950 (venv or container).

    python rt_head.py --epi fused --rows 4 --arms D,Ddpp=dppmirror+dppcombine --rounds 24 --out DIR

Arms of ONE ``KernelDef`` (``rt_builds.head``: n 248320, k 2560, ``rows_max`` 4 or 16, shipped device-op form):
``C`` COMGR, ``C2`` its bytes again, ``R`` a fresh-process COMGR compile, ``D`` the direct compiler (lifted IR),
``NAME=LEVERS`` direct with ``ROCKE_DIRECT_LEVERS``. Numerics first, every arm (R2's checks, generalized; the R2
verifier's caveat 3 closed: lever arms are checked on the logits too):

- logits builds: bf16 logits of m = 1, rows_max / 2, rows_max bit for bit against C, and against an f64 reference
  on a column sample (half a bf16 ulp plus 2^-14 sum |x w|);
- fused (greedy argmax) builds: the logits build of the same arm's backend/levers gives the argmax reference;
  out and the workspace counter bit for bit against C over a launch sequence with both ``rev`` epochs, and out ==
  the argmax of C's logits. Determinism: two independent runs per arm.

Timing: per (arm, m, placement) a HIP graph of ``--launches`` launches cycling 3 cold weight copies; every round
replays every graph once (shuffled), an arm's round time is the mean over its placements; ratios vs C per round,
win rule vs both controls (C2, R), plus a pooled-COMGR view.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "rust-pilot"))


def head_data(G, dev, copies: int, rows: int, seed: int = 1234):
    """``r2_gpu.HeadData`` with ``rows`` activation rows (R2's has 4)."""
    import numpy as np

    d = G.HeadData.__new__(G.HeadData)
    rng = np.random.default_rng(seed)
    d.codes = rng.integers(0, 256, size=(G.HEAD_N, G.HEAD_K // 2), dtype=np.uint8)
    d.scales = rng.integers(120, 132, size=(G.HEAD_N, G.HEAD_K // 32), dtype=np.uint8)
    xf = rng.standard_normal((rows, G.HEAD_K)).astype(np.float32)
    xb = (xf.view(np.uint32) + 0x7FFF + ((xf.view(np.uint32) >> 16) & 1)) >> 16
    d.xbits = xb.astype(np.uint16)
    d.x = dev.put(d.xbits)
    d.w = [dev.put(d.codes) for _ in range(copies)]
    d.s = [dev.put(d.scales) for _ in range(copies)]
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epi", default="fused", choices=("fused", "logits"))
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--arms", default="D")
    ap.add_argument("--rounds", type=int, default=24)
    ap.add_argument("--launches", type=int, default=24)
    ap.add_argument("--placements", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20261030)
    ap.add_argument("--no-time", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    import numpy as np
    import r2_gpu as G
    from rocke.examples.gfx950.persistent_mailbox import harness as H
    from rocke.helpers.compile import compile_kernel

    import kernels.gfx950.mx_decode_gemv as M
    import rt_builds
    from rt_b3 import arm_specs, compile_arms, dedupe, drop_failing, fresh_compile, win_rule

    label = f"head_{a.epi}_r{a.rows}"
    llabel = f"head_logits_r{a.rows}"
    kern = rt_builds.build(label)
    spec = M.MxDecodeGemvSpec(**rt_builds.HEAD, rows_max=a.rows, epilogue=a.epi, **({"waves_per_eu": 1} if a.epi == "fused" else {}))
    lspec = M.MxDecodeGemvSpec(**rt_builds.HEAD, rows_max=a.rows, epilogue="logits")

    def arms_of(lab):
        k = rt_builds.build(lab)
        c = compile_kernel(k, arch="gfx950", backend="python", code_object_backend="comgr")
        out = {"C": c.hsaco, "C2": bytes(c.hsaco), "R": fresh_compile(lab, "comgr", "", os.path.join(a.out, f"{lab}.R.hsaco"))}
        d = compile_kernel(rt_builds.build(lab), arch="gfx950", backend="python", code_object_backend="direct")
        if getattr(d, "code_object_backend", "") != "direct":
            raise SystemExit(f"DIRECT REFUSED {lab}: {getattr(d, 'direct_fallback', '')}")
        specs = arm_specs(a.arms)
        built = compile_arms(lab, [(n, lv) for n, lv in specs if n != "D"], a.out, prefix=f"{lab}.")
        for n, _ in specs:
            out[n] = d.hsaco if n == "D" else built[n]
        return k.name, out

    name, arts = arms_of(label)
    lname, larts = arms_of(llabel) if a.epi == "fused" else (name, arts)
    for n, h in arts.items():
        open(os.path.join(a.out, f"{label}.{n}.hsaco"), "wb").write(h)
    # an arm is a duplicate only when both its objects (fused and logits) match another arm's
    alias = dedupe(arts, keys={n: (arts[n], larts[n]) for n in arts})
    for n, t in alias.items():
        del arts[n]
        if larts is not arts:
            del larts[n]
        print("ALIAS", label, n, "=", t, flush=True)
    dev = H.Dev(watchdog_s=180)
    ok = True
    failed = set()
    res = {"label": label, "node": os.uname().nodename}
    try:
        d = head_data(G, dev, 3, a.rows)
        rng = np.random.default_rng(99)
        ms = sorted({1, max(1, a.rows // 2), a.rows})
        logits = {}
        for n, h in larts.items():
            fn = dev.function(G.raw_art(lname, h))
            for m in ms:
                for rep in range(2):
                    y = dev.alloc(4 * G.HEAD_N * a.rows)
                    dev.launch(fn, lspec.ctas, lspec.threads, G.head_args(lspec, d, y, None, m, 0))
                    dev.sync()
                    logits[(n, m, rep)] = dev.get(y, 4 * G.HEAD_N * a.rows, np.uint16).reshape(2 * a.rows, G.HEAD_N)[:a.rows]
        res["logits"] = {}
        for n in larts:
            for m in ms:
                same = bool(np.array_equal(logits[(n, m, 0)][:m], logits[("C", m, 0)][:m]))
                det = bool(np.array_equal(logits[(n, m, 0)][:m], logits[(n, m, 1)][:m]))
                chk = G.check_logits(n, logits[(n, m, 0)], d, m, rng)
                r = {"bitwise_vs_C": same, "deterministic": det, "ref_bad": chk["bad"], "max_err_over_tol": round(chk["max_err_over_tol"], 4)}
                r["pass"] = det and chk["bad"] == 0
                ok &= r["pass"]
                if not r["pass"]:
                    failed.add(n)
                res["logits"][f"{n}|m{m}"] = r
                print("CHECK", llabel, n, f"m{m}", json.dumps(r), flush=True)
        if a.epi == "fused":
            seq = [(m, rev) for rev in (0, 1) for m in ms] + [(ms[0], 0), (ms[-1], 1)]
            want = [M.ref_argmax(logits[("C", m, 0)][:m]).tolist() for m, _ in seq]
            got = {}
            for n, h in arts.items():
                runs = []
                for rep in range(2):
                    fn = dev.function(G.raw_art(name, h))
                    ws = dev.alloc(M.fused_workspace_bytes(spec))
                    outs = []
                    for m, rev in seq:
                        y = dev.alloc(64 * a.rows)
                        dev.launch(fn, spec.ctas, spec.threads, G.head_args(spec, d, y, ws, m, rev))
                        dev.sync()
                        outs.append(dev.get(y, 4 * a.rows, np.int32)[:m].tolist())
                    runs.append((outs, dev.get(ws, M.fused_workspace_bytes(spec), np.int32).tolist()))
                got[n] = runs
            res["fused"] = {}
            for n in arts:
                (o0, w0), (o1, w1) = got[n]
                r = {"out_eq_C": o0 == got["C"][0][0], "ws_eq_C": w0 == got["C"][0][1], "out_eq_argmax": o0 == want,
                     "deterministic": o0 == o1 and w0 == w1}  # fmt: skip
                r["pass"] = r["out_eq_argmax"] and r["deterministic"] and r["ws_eq_C"]
                ok &= r["pass"]
                if not r["pass"]:
                    failed.add(n)
                res["fused"][n] = r
                print("CHECK", label, n, json.dumps(r), flush=True)
        res["pass"] = bool(ok)
        go, dropped = drop_failing(failed)
        for n in dropped:
            print("DROPPED", label, n, "(numerics failed)", flush=True)
            for t in [n] + [x for x, y in alias.items() if y == n]:
                alias.pop(t, None)
            arts.pop(n, None)
        res["dropped"] = dropped
        if a.no_time or not go:
            print("NUMERICS", "PASS" if ok else "FAIL", flush=True)
            json.dump(res, open(os.path.join(a.out, "ratios.json"), "w"), indent=1)
            return 0 if ok else 1
        # ---- timing
        ws = {n: dev.alloc(M.fused_workspace_bytes(spec)) for n in arts} if a.epi == "fused" else {}
        ys = dev.alloc(64 * a.rows) if a.epi == "fused" else dev.alloc(4 * G.HEAD_N * a.rows)
        graphs = {}
        order = [(n, m, p) for n in arts for m in ms for p in range(a.placements)]
        random.Random(a.seed + 1).shuffle(order)
        for n, m, p in order:
            fn = dev.function(G.raw_art(name, arts[n]))
            g = H._Graph(dev)

            def issue(stream, fn=fn, n=n, m=m):
                for i in range(a.launches):
                    dev.rt.launch(fn, (spec.ctas, 1, 1), (spec.threads, 1, 1),
                                  G.head_args(spec, d, ys, ws.get(n), m, 0, copy=i % 3), stream=stream)  # fmt: skip

            g.capture(issue)
            g.replay()
            graphs[(n, m, p)] = g
        rnd = random.Random(a.seed)
        raw = {f"{n}|m{m}": [] for n in arts for m in ms}
        # SCLK / power per arm (sysfs of the job's card, sampled at every replay's start and every 0.25 s):
        # a code object that draws more power can run at a lower clock under the cap (a cross-CU effect the
        # per-CU traces cannot show)
        try:
            from kernels.gfx950.qwen38_mono.tune.gate import ClockSampler
            from kernels.gfx950.qwen38_mono.tune.live_toy import _bdf

            sampler = ClockSampler(bdf=_bdf(), period=0.25)
        except Exception:  # noqa: BLE001 - clocks are best effort
            sampler = None
        if sampler is not None:
            sampler.__enter__()
        try:
            for _ in range(a.rounds):
                rnd.shuffle(order)
                cur = {k: [] for k in raw}
                for n, m, p in order:
                    if sampler is not None:
                        sampler.mark(n)
                    g = graphs[(n, m, p)]
                    e0, e1 = dev.rt.event(), dev.rt.event()
                    e0.record(g.stream.value)
                    g.lib.hipGraphLaunch(g.exec, g.stream)
                    e1.record(g.stream.value)
                    e1.synchronize()
                    cur[f"{n}|m{m}"].append(e0.elapsed_to(e1) / a.launches)
                    e0.destroy()
                    e1.destroy()
                for k, v in cur.items():
                    raw[k].append(statistics.mean(v))
        finally:
            if sampler is not None:
                sampler.__exit__(None, None, None)
            for g in graphs.values():
                g.close()
        if sampler is not None:
            clk = sampler.per_label()
            base_clk = clk.get("C", {})
            res["clocks"] = {n: {"sclk_vs_C": round(v["sclk"] / base_clk["sclk"], 4) if v.get("sclk") and base_clk.get("sclk") else None,
                                 "power_vs_C": round(v["power"] / base_clk["power"], 4) if v.get("power") and base_clk.get("power") else None,
                                 "n": v["n"]} for n, v in clk.items()}  # fmt: skip
            res["clock_source"] = sampler.source
            for n, v in res["clocks"].items():
                print("CLOCK", label, n, json.dumps(v), flush=True)
        res["timing"] = {}
        for m in ms:
            base = raw[f"C|m{m}"]
            pooled = [statistics.mean(v) for v in zip(base, raw[f"C2|m{m}"], raw[f"R|m{m}"])]
            ratios = {n: [x / c for x, c in zip(raw[f"{n}|m{m}"], base)] for n in arts if n != "C"}
            controls = [ratios["C2"], ratios["R"]]
            for n, rs in ratios.items():
                s = {"median": round(statistics.median(rs), 4), "min": round(min(rs), 4), "max": round(max(rs), 4),
                     "below1": f"{sum(x < 1 for x in rs)}/{len(rs)}",
                     "verdict": "control" if n in ("C2", "R") else win_rule(rs, controls),
                     "vs_pooled": round(statistics.median([x / p for x, p in zip(raw[f"{n}|m{m}"], pooled)]), 4)}  # fmt: skip
                res["timing"][f"{n}|m{m}"] = s
                print("TIME", label, n, f"m{m}", json.dumps(s), flush=True)
            for n, t in alias.items():
                s = dict(res["timing"][f"{t}|m{m}"], same_as=t)
                res["timing"][f"{n}|m{m}"] = s
                print("TIME", label, n, f"m{m}", json.dumps(s), flush=True)
        res["alias"] = alias
        json.dump(res, open(os.path.join(a.out, "ratios.json"), "w"), indent=1)
        json.dump({"raw_us": raw}, open(os.path.join(a.out, "raw.json"), "w"))
        return 0
    finally:
        dev.close()


if __name__ == "__main__":
    sys.exit(main())
