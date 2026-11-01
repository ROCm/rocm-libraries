"""R2 on gfx950: numerics, then timing, of the direct code-object compiler's
pilots against COMGR (Qwen3.8 true monokernel, Rust pilot track).

    python r2_gpu.py numerics --out DIR [--spill-dir DIR]
    python r2_gpu.py time --out DIR [--rounds N] [--what head toy]
    python r2_gpu.py compile LABEL BACKEND PATH     (one compile in this process)

Runs with PYTHONPATH on the R2 rust tree (platform/python + library) and
``ROCKE_CODEOBJ_DIRECT_LIB`` set. Every arm of a pilot compiles from ONE
``KernelDef`` (or the same serialized IR): the COMGR build of the shipped
(asm) form, the COMGR build of the IR-only (``irops``) form, and the direct
build of the IR-only form (refused = failure).

numerics (gates, before any timing):
  * head logits (the hybrid's logits head: no device asm, direct vs COMGR of
    the same kernel): bf16 logits bit for bit, m = 1 and 4, and against an
    f64 reference on a column sample (tolerance: half a bf16 ulp plus
    2^-14 sum |x w|);
  * head fused argmax (the hybrid's argmax head, ``waves_per_eu=1``): per arm
    its own zeroed workspace, 6 launches over m = 1, 4 and rev = 0, 1 (the
    counter64 epochs advance); out and the workspace counter bit for bit
    across arms, and out equal to the argmax of the validated logits;
  * forced spills (``--spill-dir``: direct objects compiled by
    ``target_compile --max-vgprs N``, ``<label>.direct_v<N>.hsaco``) run the
    same checks;
  * toy chain at ``best_spec(4, 8, layers=2)`` and ``best_spec(8, 0)``: per arm
    graph replays (step counter straddling the tag wrap) checked against the
    host model (``toy_chain_bench.check_state`` + running checksums), buffers
    shared by the arms.
time (interleaved, arm order shuffled per round, one process):
  * head: per arm a hipGraph of 24 launches cycling 3 weight copies (cold:
    ~1 GB > MALL), event time per launch; arms ``A`` COMGR shipped head (the
    hybrid's), ``A2`` the same bytes loaded again (byte-copy null), ``R`` a
    fresh-process COMGR recompile (recompiled control), ``I`` COMGR irops,
    ``D`` direct irops (+ lever arms given with ``--arm``: ``WHAT:NAME=LABEL/BACKEND[/LEVERS]``,
    compiled in a fresh process with ``ROCKE_DIRECT_LEVERS=LEVERS`` (``+``-joined), ``WHAT`` ``head`` or
    ``toy_<shape>``, ``LABEL`` a :func:`build` label);
  * toy: ``toy_chain_bench.time_cell`` (12 chained launches, cache flush
    between reps) per arm, the same arm set, buffers shared by the arms and
    rotated over 3 calibrated allocations per round.
  Raw times go to ``DIR/time_raw.json`` (private), per-round ratios to
  ``DIR/time_ratios.json`` with the DESIGN 9.1 win rule evaluated.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import random
import socket
import statistics
import struct
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

HEAD_N, HEAD_K = 248320, 2560


# ---------------------------------------------------------------- builds


def head_spec(epilogue: str, irops: bool = False, **kw):
    from kernels.gfx950 import mx_decode_gemv as M
    from kernels.gfx950.qwen38_mono import direct_pilot as DP

    base = dict(DP.HYBRID_HEAD)
    base["epilogue"] = epilogue
    if epilogue == "logits":
        base.pop("waves_per_eu", None)  # the hybrid's logits head: DEFAULT_SPEC["logits"] = {}
    base.update(kw)
    return M.MxDecodeGemvSpec(irops=irops, **base)


def toy_spec(shape: str, irops: bool):
    from kernels.gfx950.qwen38_mono import toy_chain as T

    spec = T.best_spec(4, 8, layers=2) if shape == "d4w8l2" else T.best_spec(8, 0)
    if irops:
        spec = dataclasses.replace(spec, knobs=dataclasses.replace(spec.knobs, irops=True))
    return spec


def build(label: str):
    """KernelDef of a pilot arm label: head_{logits,fused}[_irops], toy_{d4w8l2,d8w0}[_irops];
    ``head_...@k=v,k=v`` overrides head spec fields (ints, else strings)."""
    from kernels.gfx950 import mx_decode_gemv as M
    from kernels.gfx950.qwen38_mono import toy_chain as T

    label, _, over = label.partition("@")
    kw = {}
    for item in filter(None, over.split(",")):
        k, v = item.split("=", 1)
        kw[k] = int(v) if v.lstrip("-").isdigit() else v
    irops = label.endswith("_irops")
    base = label[: -len("_irops")] if irops else label
    if base.startswith("head_"):
        return M.build_mx_decode_gemv(head_spec(base[len("head_"):], irops, **kw))
    if kw:
        raise ValueError("toy labels take no overrides")
    return T.build_toy_chain(toy_spec(base[len("toy_"):], irops))


def compile_art(kernel, backend: str):
    from rocke.helpers.compile import compile_kernel

    art = compile_kernel(kernel, arch="gfx950", backend="python", code_object_backend=backend)
    got = getattr(art, "code_object_backend", backend)
    if got != backend:
        raise RuntimeError(f"{kernel.name}: asked for {backend}, got {got} ({getattr(art, 'direct_fallback', '')})")
    return art


def raw_art(name: str, hsaco: bytes):
    """A loadable artifact from raw code-object bytes (a byte copy, a
    fresh-process compile, a forced-spill object)."""
    return SimpleNamespace(kernel_name=name, hsaco=bytes(hsaco))


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def fresh_compile(label: str, backend: str, tmp: str, levers: str = "") -> bytes:
    """``label`` compiled with ``backend`` in a fresh process (direct: with
    ``ROCKE_DIRECT_LEVERS=levers``)."""
    path = os.path.join(tmp, f"fresh_{abs(hash((label, backend, levers)))}.hsaco")
    env = dict(os.environ, ROCKE_DIRECT_LEVERS=levers.replace("+", ","))
    subprocess.run([sys.executable, os.path.abspath(__file__), "compile", label, backend, path], check=True, env=env)
    return open(path, "rb").read()


def lever_arms(specs: list[str], what: str, tmp: str) -> dict:
    """``--arm`` entries for ``what``: {name: art}."""
    out = {}
    for s in specs:
        w, _, rest = s.partition(":")
        if w != what:
            continue
        name, _, how = rest.partition("=")
        label, backend, *lv = how.split("/")
        k = build(label)
        out[name] = raw_art(k.name, fresh_compile(label, backend, tmp, lv[0] if lv else ""))
    return out


# ---------------------------------------------------------------- head


class HeadData:
    """MXFP4 head operands: ``copies`` weight copies, x for 4 rows."""

    def __init__(self, dev, copies: int, seed: int = 1234):
        rng = np.random.default_rng(seed)
        self.codes = rng.integers(0, 256, size=(HEAD_N, HEAD_K // 2), dtype=np.uint8)
        self.scales = rng.integers(120, 132, size=(HEAD_N, HEAD_K // 32), dtype=np.uint8)
        xf = rng.standard_normal((4, HEAD_K)).astype(np.float32)
        self.xbits = (xf.view(np.uint32) + 0x7FFF + ((xf.view(np.uint32) >> 16) & 1)) >> 16
        self.xbits = self.xbits.astype(np.uint16)
        self.x = dev.put(self.xbits)
        self.w = [dev.put(self.codes) for _ in range(copies)]
        self.s = [dev.put(self.scales) for _ in range(copies)]

    def x_f64(self, m: int) -> np.ndarray:
        return (self.xbits[:m].astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def head_args(spec, d: HeadData, y: int, ws: int | None, m: int, rev: int, copy: int = 0) -> bytes:
    from kernels.gfx950 import mx_decode_gemv as M

    vals = dict(x=d.x, w=d.w[copy], s=d.s[copy], y=y, m=m, sxm=HEAD_K, sym=HEAD_N, rev=rev)
    if ws is not None:
        vals["ws"] = ws
    return M.launch_args(spec, **vals)


def bf16_ulp(v: np.ndarray) -> np.ndarray:
    e = np.floor(np.log2(np.maximum(np.abs(v), 1e-30)))
    return np.exp2(e - 7)


def head_logits_ref(d: HeadData, m: int, cols: np.ndarray):
    """f64 reference logits of ``cols`` and sum |x w| per entry."""
    from kernels.gfx950 import mx_decode_gemv as M

    x = d.x_f64(m)
    ref = M.ref_logits_f32(x, d.codes[cols], d.scales[cols])
    values = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])
    c = np.stack([d.codes[cols] & 15, d.codes[cols] >> 4], axis=-1).reshape(len(cols), -1)
    w = values[c] * np.exp2(d.scales[cols].astype(np.float64) - 127.0).repeat(32, axis=1)
    mag = np.abs(x) @ np.abs(w).T
    return ref, mag


def check_logits(name: str, bits: np.ndarray, d: HeadData, m: int, rng) -> dict:
    cols = np.unique(np.concatenate([rng.choice(HEAD_N, 4096, replace=False), np.argmax(bits[:m].astype(np.int64), axis=1)]))
    ref, mag = head_logits_ref(d, m, cols)
    got = (bits[:m, cols].astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    tol = 0.5 * bf16_ulp(ref) + mag * 2.0**-14
    err = np.abs(got - ref)
    bad = int((err > tol).sum())
    return {"arm": name, "m": m, "cols": int(len(cols)), "bad": bad, "max_err_over_tol": float((err / tol).max())}


def run_head_numerics(dev, out: dict, spill_dir: str | None, arm_specs=(), tmp: str = ".") -> bool:
    import kernels.gfx950.mx_decode_gemv as M
    from rocke.examples.gfx950.persistent_mailbox import harness as H

    ok = True
    d = HeadData(dev, copies=1)
    rng = np.random.default_rng(99)
    # ---- logits: direct vs COMGR of one kernel
    spec_l = head_spec("logits")
    k_l = M.build_mx_decode_gemv(spec_l)
    arms_l = {"comgr": compile_art(k_l, "comgr"), "direct": compile_art(k_l, "direct")}
    logits = {}
    for name, art in arms_l.items():
        fn = dev.function(art)
        for m in (1, 4):
            y = dev.alloc(4 * HEAD_N * 2)
            dev.launch(fn, spec_l.ctas, spec_l.threads, head_args(spec_l, d, y, None, m, 0))
            dev.sync()
            logits[(name, m)] = dev.get(y, 4 * HEAD_N * 2, np.uint16).reshape(4, HEAD_N)
    res = {"logits": []}
    for m in (1, 4):
        same = bool(np.array_equal(logits[("comgr", m)][:m], logits[("direct", m)][:m]))
        chk = check_logits("comgr", logits[("comgr", m)], d, m, rng)
        res["logits"].append({"m": m, "bitwise_direct_vs_comgr": same, "ref": chk})
        ok &= same and chk["bad"] == 0
    # ---- fused argmax: COMGR asm (hybrid), COMGR irops, direct irops (+ forced spills)
    spec_a, spec_i = head_spec("fused"), head_spec("fused", irops=True)
    k_a, k_i = M.build_mx_decode_gemv(spec_a), M.build_mx_decode_gemv(spec_i)
    arms = {"A": compile_art(k_a, "comgr"), "I": compile_art(k_i, "comgr"), "D": compile_art(k_i, "direct")}
    if spill_dir:
        for f in sorted(os.listdir(spill_dir)):
            if f.startswith("head_fused_irops.direct_v") and f.endswith(".hsaco"):
                arms["D" + f.split(".direct_")[1][: -len(".hsaco")]] = raw_art(k_i.name, open(os.path.join(spill_dir, f), "rb").read())
    arms.update(lever_arms(list(arm_specs), "head", tmp))
    res["arms"] = {n: {"bytes": len(a.hsaco), "sha": sha(a.hsaco), "name": a.kernel_name} for n, a in arms.items()}
    res["text_A_eq_I"] = None
    seq = [(1, 0), (4, 0), (1, 1), (4, 1), (1, 0), (4, 1)]
    got = {}
    for name, art in arms.items():
        fn = dev.function(art)
        ws = dev.alloc(M.fused_workspace_bytes(spec_a))
        outs = []
        for m, rev in seq:
            y = dev.alloc(64)
            dev.launch(fn, spec_a.ctas, spec_a.threads, head_args(spec_a, d, y, ws, m, rev))
            dev.sync()
            outs.append(dev.get(y, 16, np.int32)[:m].tolist())
        got[name] = {"out": outs, "ws": dev.get(ws, M.fused_workspace_bytes(spec_a), np.int32).tolist()}
    want = [M.ref_argmax(logits[("comgr", m)][:m]).tolist() for m, _ in seq]
    res["fused"] = {}
    for name in arms:
        r = {
            "out_eq_A": got[name]["out"] == got["A"]["out"],
            "ws_eq_A": got[name]["ws"] == got["A"]["ws"],
            "out_eq_argmax_of_logits": got[name]["out"] == want,
        }
        res["fused"][name] = r
        ok &= all(r.values())
    out["head"] = res
    return ok


# ---------------------------------------------------------------- toy


def toy_bench():
    from benchmarks.gfx950.qwen38_mono import toy_chain_bench as B

    return B


def toy_replay(dev, spec, art, step_art, bufs, replays: int) -> dict:
    """``toy_chain_bench.replay_check`` with a given code object and shared buffers."""
    from rocke.examples.gfx950.persistent_mailbox import harness as H

    B = toy_bench()
    chain = len(bufs.h) - 1
    step0 = (1 << 26) - replays // 2
    dev.rt.memset(bufs.scratch, 0, bufs.scratch_bytes)
    dev.write(bufs.step, np.array([step0], np.int32))
    for k in range(1, len(bufs.h)):
        dev.write(bufs.h[k], np.zeros(B.H_WORDS, np.int32))
    g = H._Graph(dev)
    try:
        g.capture(B.graph_launches(dev, bufs, art, step_art, launches=chain, copies=1))
        for _ in range(replays):
            g.lib.hipGraphLaunch(g.exec, g.stream)
        g._chk(g.lib.hipStreamSynchronize(g.stream), "sync")
    finally:
        g.close()
    layers0 = [k * spec.layers for k in range(chain)]
    errs = B.check_state(dev, bufs, step0 + replays - 1, layers0)
    want = B.checksum_model(bufs, step0, replays, layers0)
    for (k, li), v in want.items():
        g2 = int(B.read_i32(dev, bufs.h[k + 1], B.H_WORDS)[4 * li + 1])
        if g2 != v:
            errs.append(f"checksum launch {k} layer {li}: {g2} != {v}")
    return {"replays": replays, "ok": not errs, "errors": errs[:5]}


def toy_arms(shape: str, tmp: str | None):
    """{arm: art} for the toy pilot ``shape``."""
    k_a, k_i = build(f"toy_{shape}"), build(f"toy_{shape}_irops")
    arms = {"A": compile_art(k_a, "comgr"), "I": compile_art(k_i, "comgr"), "D": compile_art(k_i, "direct")}
    return arms, k_a, k_i


def run_toy_numerics(dev, out: dict, spill_dir: str | None, replays: int = 2000, arm_specs=(), tmp: str = ".") -> bool:
    B = toy_bench()
    ok = True
    out["toy"] = {}
    step_art = compile_art(B.T.build_step_kernel(), "comgr")
    for shape in ("d4w8l2", "d8w0"):
        spec = toy_spec(shape, False)
        arms, _, k_i = toy_arms(shape, None)
        if spill_dir:
            for f in sorted(os.listdir(spill_dir)):
                if f.startswith(f"toy_{shape}_irops.direct_v") and f.endswith(".hsaco"):
                    arms["D" + f.split(".direct_")[1][: -len(".hsaco")]] = raw_art(k_i.name, open(os.path.join(spill_dir, f), "rb").read())
        arms.update(lever_arms(list(arm_specs), f"toy_{shape}", tmp))
        bufs = B.alloc(dev, spec, chain=2)
        r = {"arms": {n: {"bytes": len(a.hsaco), "sha": sha(a.hsaco)} for n, a in arms.items()}}
        for name, art in arms.items():
            rep = toy_replay(dev, spec, art, step_art, bufs, replays)
            single = []
            for step in (5, 1000):
                dev.rt.memset(bufs.scratch, 0, bufs.scratch_bytes)
                dev.write(bufs.step, np.array([step], np.int32))
                B.run_chain_once(dev, bufs, art, step=step, layers0=[0, spec.layers])
                single += B.check_state(dev, bufs, step, [0, spec.layers])
            r[name] = {"replay": rep, "single_ok": not single, "single_errors": single[:3]}
            ok &= rep["ok"] and not single
        out["toy"][shape] = r
    return ok


# ---------------------------------------------------------------- timing


def win_rule(ratios: dict, arm: str, controls=("A2", "R")) -> dict:
    """DESIGN 9.1: the arm's median ratio below both controls' minima by
    0.002 AND at least 3/4 of rounds below 1 (a loss mirrored)."""
    x = ratios[arm]
    med = statistics.median(x)
    floor = min(min(ratios[c]) for c in controls if c in ratios)
    ceil = max(max(ratios[c]) for c in controls if c in ratios)
    below = sum(v < 1 for v in x) / len(x)
    above = sum(v > 1 for v in x) / len(x)
    if med < floor - 0.002 and below >= 0.75:
        verdict = "WIN"
    elif med > ceil + 0.002 and above >= 0.75:
        verdict = "LOSS"
    else:
        verdict = "NEUTRAL"
    return {"median": round(med, 4), "min": round(min(x), 4), "max": round(max(x), 4), "frac_below_1": round(below, 3),
            "controls_min": round(floor, 4), "controls_max": round(ceil, 4), "verdict": verdict}


def time_head(dev, arms: dict, rounds: int, launches: int, reps: int, rows_list, seed: int) -> dict:
    import kernels.gfx950.mx_decode_gemv as M
    from rocke.examples.gfx950.persistent_mailbox import harness as H

    spec = head_spec("fused")
    d = HeadData(dev, copies=3)
    ws = {n: dev.alloc(M.fused_workspace_bytes(spec)) for n in arms}
    ys = dev.alloc(64)
    graphs = {}
    for name, art in arms.items():
        fn = dev.function(art)
        for m in rows_list:
            g = H._Graph(dev)

            def issue(stream, fn=fn, name=name, m=m):
                for i in range(launches):
                    dev.rt.launch(fn, (spec.ctas, 1, 1), (spec.threads, 1, 1),
                                  head_args(spec, d, ys, ws[name], m, 0, copy=i % 3), stream=stream)  # fmt: skip

            g.capture(issue)
            g.replay()
            graphs[(name, m)] = g
    rnd = random.Random(seed)
    raw = {f"{n}|m{m}": [] for n in arms for m in rows_list}
    try:
        for _ in range(rounds):
            order = [(n, m) for n in arms for m in rows_list]
            rnd.shuffle(order)
            for name, m in order:
                g = graphs[(name, m)]
                per = []
                for _ in range(reps):
                    e0, e1 = dev.rt.event(), dev.rt.event()
                    e0.record(g.stream.value)
                    g.lib.hipGraphLaunch(g.exec, g.stream)
                    e1.record(g.stream.value)
                    e1.synchronize()
                    per.append(e0.elapsed_to(e1) / launches)
                    e0.destroy()
                    e1.destroy()
                raw[f"{name}|m{m}"].append(statistics.median(per))
    finally:
        for g in graphs.values():
            g.close()
    return raw


def time_toy(dev, shape: str, arms: dict, rounds: int, reps: int, seed: int, allocs: int = 3) -> dict:
    B = toy_bench()
    spec = toy_spec(shape, False)
    step_art = compile_art(B.T.build_step_kernel(), "comgr")
    sets = [B.alloc(dev, spec, chain=12, rng_seed=7 + i) for i in range(allocs)]
    flush = dev.alloc(B.FLUSH_BYTES, zero=False)
    rnd = random.Random(seed)
    raw = {n: [] for n in arms}
    errors = {n: [] for n in arms}
    for r in range(rounds):
        bufs = sets[r % allocs]
        order = list(arms)
        rnd.shuffle(order)
        for name in order:
            dev.rt.memset(bufs.scratch, 0, bufs.scratch_bytes)
            rec = B.time_cell(dev, spec, launches=12, reps=reps, art=arms[name], step_art=step_art, bufs=bufs, flush=flush)
            raw[name].append(statistics.median(rec["per_launch_ms"]))
            if not rec["ok"]:
                errors[name] += rec["errors"]
    return {"raw": raw, "errors": {n: e[:5] for n, e in errors.items() if e}}


def ratios_of(raw: dict, base: str) -> dict:
    return {n: [x / b for x, b in zip(v, raw[base])] for n, v in raw.items() if n != base}


# ---------------------------------------------------------------- main


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("numerics", "time", "compile"))
    ap.add_argument("rest", nargs="*")
    ap.add_argument("--out", default=".")
    ap.add_argument("--spill-dir", default="")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--launches", type=int, default=24)
    ap.add_argument("--what", nargs="+", default=["head", "toy"])
    ap.add_argument("--seed", type=int, default=20261030)
    ap.add_argument("--arm", nargs="*", default=[], help="lever arms: WHAT:NAME=LABEL/BACKEND[/LEVERS]")
    a = ap.parse_args(argv)
    if a.cmd == "compile":
        label, backend, path = a.rest
        art = compile_art(build(label), backend)
        with open(path, "wb") as fh:
            fh.write(art.hsaco)
        return 0
    from rocke.examples.gfx950.persistent_mailbox import harness as H

    os.makedirs(a.out, exist_ok=True)
    prob = H.env_problem()
    if prob:
        print("SKIP:", prob)
        return 2
    node = socket.gethostname()
    dev = H.Dev(watchdog_s=180)
    try:
        if a.cmd == "numerics":
            res = {"node": node}
            ok_h = run_head_numerics(dev, res, a.spill_dir or None, a.arm, a.out) if "head" in a.what else True
            ok_t = run_toy_numerics(dev, res, a.spill_dir or None, 2000, a.arm, a.out) if "toy" in a.what else True
            res["ok"] = bool(ok_h and ok_t)
            with open(os.path.join(a.out, "numerics.json"), "w") as fh:
                json.dump(res, fh, indent=1)
            print(json.dumps({k: v for k, v in res.items() if k != "toy"}, indent=1)[:6000])
            print("toy:", json.dumps({s: {n: (r.get("replay", {}).get("ok"), r.get("single_ok")) for n, r in v.items() if n != "arms"} for s, v in res.get("toy", {}).items()}))
            print("NUMERICS", "OK" if res["ok"] else "FAIL")
            return 0 if res["ok"] else 1
        raw_all, rat_all = {"node": node}, {"node": node}
        if "head" in a.what:
            k_a, k_i = build("head_fused"), build("head_fused_irops")
            A = compile_art(k_a, "comgr")
            arms = {"A": A, "A2": raw_art(A.kernel_name, bytes(A.hsaco)),
                    "R": raw_art(A.kernel_name, fresh_compile("head_fused", "comgr", a.out)),
                    "I": compile_art(k_i, "comgr"), "D": compile_art(k_i, "direct")}  # fmt: skip
            arms.update(lever_arms(a.arm, "head", a.out))
            print("head arms:", {n: sha(v.hsaco) for n, v in arms.items()}, flush=True)
            raw = time_head(dev, arms, a.rounds, a.launches, a.reps, (1, 4), a.seed)
            raw_all["head"] = raw
            rat = {}
            for m in (1, 4):
                sub = {n: raw[f"{n}|m{m}"] for n in arms}
                r = ratios_of(sub, "A")
                rat[f"m{m}"] = {"ratios": {n: [round(x, 5) for x in v] for n, v in r.items()},
                                "rule": {n: win_rule(r, n) for n in r if n not in ("A2", "R")},
                                "controls": {n: win_rule(r, n) for n in ("A2", "R")}}  # fmt: skip
            rat_all["head"] = rat
            print("head:", json.dumps({m: v["rule"] for m, v in rat.items()}), flush=True)
        if "toy" in a.what:
            for shape in ("d4w8l2", "d8w0"):
                arms, k_a, k_i = toy_arms(shape, a.out)
                A = arms["A"]
                arms = {"A": A, "A2": raw_art(A.kernel_name, bytes(A.hsaco)),
                        "R": raw_art(A.kernel_name, fresh_compile(f"toy_{shape}", "comgr", a.out)),
                        "I": arms["I"], "D": arms["D"]}  # fmt: skip
                arms.update(lever_arms(a.arm, f"toy_{shape}", a.out))
                print(f"toy {shape} arms:", {n: sha(v.hsaco) for n, v in arms.items()}, flush=True)
                res = time_toy(dev, shape, arms, a.rounds, 3, a.seed)
                raw_all[f"toy_{shape}"] = res
                r = ratios_of(res["raw"], "A")
                rat_all[f"toy_{shape}"] = {"ratios": {n: [round(x, 5) for x in v] for n, v in r.items()},
                                           "rule": {n: win_rule(r, n) for n in r if n not in ("A2", "R")},
                                           "controls": {n: win_rule(r, n) for n in ("A2", "R")},
                                           "numerics_errors": res["errors"]}  # fmt: skip
                print(f"toy {shape}:", json.dumps(rat_all[f"toy_{shape}"]["rule"]), "errors:", res["errors"], flush=True)
        with open(os.path.join(a.out, "time_raw.json"), "w") as fh:
            json.dump(raw_all, fh, indent=1)
        with open(os.path.join(a.out, "time_ratios.json"), "w") as fh:
            json.dump(rat_all, fh, indent=1)
        bad = any(v.get("numerics_errors") for k, v in rat_all.items() if k.startswith("toy_"))
        return 1 if bad else 0
    finally:
        dev.close()


if __name__ == "__main__":
    sys.exit(main())
