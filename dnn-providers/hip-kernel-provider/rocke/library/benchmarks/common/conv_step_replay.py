# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Replay a training step's conv calls back to back with pinned cached kernels.

Torch-free (``import torch`` is blocked in this process). Every call is bound
exactly as ``benchmark_implicit_gemm_conv --run-from-cache`` binds it
(:func:`_cached_bind_batch`: the same kernargs, grid, output / scratch zeroing
and two-stage Stage 2), from a *pins* file naming one kernel (and launch
parameters) per case.

Pins file (JSON)::

    {"cases": {"fwd_s1": {"kernel": "<identity label> [@split_k=...]",
                          "cmd": "MIOpenDriver ... -F 1 ...", "calls": 2}, ...}}

``derive_conv_pins.py`` writes one from benchmark results (one or several runs).

Modes (``--mode``, repeatable; default ``staged step``):

* ``staged``  each case's call timed as the sweep's staged protocol times it
              (stalled single-call samples, median) -- must reproduce the
              per-cell record numbers.
* ``parts``   the same per case, split: the whole call, the call with its
              zeroing memsets skipped, and the memsets alone (``rt.memset_async``
              is intercepted). The difference is what the launcher adds around
              the kernel.
* ``step``    ``--steps`` training steps back to back (after ``--warmup-steps``),
              in model order (forward encoder -> decoder, then backward
              decoder -> encoder with dgrad then wgrad per layer); one event pair
              per step, optionally one event between consecutive calls
              (``--per-call``). GFX clock / socket power are sampled with
              ``amd-smi`` meanwhile (``--no-clock`` turns that off).

Every (case, direction) gets its own operand buffers (``--share`` reuses one
set per case and direction across its calls, which is the default: the calls of
one case in a step differ only in their data, and a fresh buffer per call costs
memory, not fidelity, since no buffer of one case fits the 256 MB MALL beside
the others).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time


class _NoTorch:
    """``import torch`` fails in this process (torch-free replay)."""

    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch is blocked in conv_step_replay")
        return None


# Forward calls of one ScaFFold UNet step in model order (shape ids of
# DESCRIPTION.md; 15 calls): encoder L0..L4, then the decoder L3..L0.
FWD_ORDER = (1, 9, 4, 6, 7, 10, 10, 8, 7, 5, 6, 3, 4, 2, 1)

SPLIT_RE = re.compile(r"@split_k=(\d+)(?:/full(\d+))?")


def step_sequence(order=FWD_ORDER):
    """``[(direction, shape)]`` of one training step: fwd, then bwd reversed."""
    seq = [("fwd", s) for s in order]
    for s in reversed(order):
        seq += [("dgrad", s), ("wgrad", s)]
    return seq


def _bf16_random(n, seed):
    import numpy as np

    rng = np.random.default_rng(seed)
    u = rng.uniform(-1.0, 1.0, size=n).astype(np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


class ClockSampler:
    """GFX clock (mean of the XCDs) and socket power from amd-smi, in a thread."""

    def __init__(self, gpu, period=0.5):
        self.gpu, self.period = gpu, period
        self.samples = []
        self._stop = threading.Event()
        self._t = None

    def _one(self):
        try:
            out = subprocess.run(
                ["/opt/rocm/bin/amd-smi", "metric", "-g", str(self.gpu), "--clock", "--power"],
                capture_output=True, text=True, timeout=5,
            ).stdout
        except Exception:  # noqa: BLE001
            return None
        clks, pw, in_gfx = [], None, False
        for line in out.splitlines():
            s = line.strip()
            if s.startswith("GFX_") and s.endswith(":"):
                in_gfx = True
                continue
            if s.startswith("MEM_") or s.startswith("VCLK") or s.startswith("DCLK"):
                in_gfx = False
            if in_gfx and s.startswith("CLK:"):
                try:
                    clks.append(float(s.split()[1]))
                except ValueError:
                    pass
            if s.startswith("SOCKET_POWER:"):
                try:
                    pw = float(s.split()[1])
                except ValueError:
                    pass
        if not clks:
            return None
        return (time.monotonic(), sum(clks) / len(clks), pw)

    def _run(self):
        while not self._stop.is_set():
            r = self._one()
            if r:
                self.samples.append(r)
            self._stop.wait(self.period)

    def start(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def stop(self):
        self._stop.set()
        if self._t:
            self._t.join()

    def summary(self, t0=None, t1=None):
        xs = [s for s in self.samples if (t0 is None or s[0] >= t0) and (t1 is None or s[0] <= t1)]
        if not xs:
            return None
        clk = [s[1] for s in xs]
        pw = [s[2] for s in xs if s[2] is not None]
        return {
            "n": len(xs),
            "clk_mean_mhz": round(statistics.mean(clk), 1),
            "clk_min_mhz": round(min(clk), 1),
            "clk_max_mhz": round(max(clk), 1),
            "power_mean_w": round(statistics.mean(pw), 1) if pw else None,
        }


class Case:
    """One (direction, shape) with its pinned kernel bound on its own buffers."""

    def __init__(self, key, pin, cache, arch, B, rt, host_bits, stage2_kernels, args, share_from=None, stream=0):
        from benchmarks.common.kernel_cache import KernelIdentity
        from kernels.conv.kernel_label import strip_launch_suffix
        from rocke.runtime.launcher import KernelLauncher, LaunchConfig

        self.key = key
        self.label = pin["kernel"]
        self.calls = int(pin.get("calls", 1))
        self.ref_ms = pin.get("ms")
        problem, dtype, forw = B.parse_miopen_cmd(pin["cmd"])
        direction = {1: "fwd", 2: "dgrad", 4: "wgrad"}[int(forw)]
        self.direction, self.problem, self.dtype = direction, problem, dtype
        name = strip_launch_suffix(self.label)
        ident = KernelIdentity.from_label(name)
        m = SPLIT_RE.search(self.label)
        split_k = None
        if m:
            split_k = int(m.group(1))
            if m.group(2) is not None:
                split_k = B.TailSplit(split_k, int(m.group(2)))
        entry = cache.get(ident)
        if entry is None:
            raise SystemExit(f"{key}: {name} is not in the cache")
        blob, meta = entry
        launcher = KernelLauncher(
            hsaco=blob,
            kernel_name=meta["kernel_name"],
            signature=B._signature_for_identity(ident, dtype),
        )
        p = problem
        nhwc = (p.N, p.Di, p.Hi, p.Wi, p.C)
        nhwk = (p.N, p.Do, p.Ho, p.Wo, p.K)
        krsc = (p.K, p.Z, p.Y, p.X, p.cpg)
        shapes = {"fwd": (nhwc, krsc, nhwk), "dgrad": (nhwk, krsc, nhwc), "wgrad": (nhwk, nhwc, krsc)}[direction]
        elem = 2
        sizes = []
        for s in shapes:
            n = 1
            for d in s:
                n *= d
            sizes.append(n * elem)
        self.sizes = tuple(sizes)
        self.bufs = []
        n_sets = self.calls if not args.share else 1
        self.ptr_sets = list(share_from.ptr_sets) if share_from is not None else []
        for _ in range(0 if share_from is not None else n_sets):
            ptrs = tuple(rt.alloc(b) for b in sizes)
            self.bufs += list(ptrs)
            for ptr, nb in zip(ptrs[:2], sizes[:2]):
                n = nb // 2
                rt.memcpy_h2d(ptr, host_bits.ctypes.data_as(__import__("ctypes").POINTER(__import__("ctypes").c_uint8)).contents if False else _u8(host_bits, n * 2), n * 2)
            rt.memset(ptrs[2], 0, sizes[2])
            self.ptr_sets.append(ptrs)
        run_problem = B._launch_problem(direction, problem)
        chunks = B._batch_chunks(direction, run_problem, dtype)
        self.live_extras = []
        self.runs = []
        cfg_cls = LaunchConfig
        if stream:
            # Bind every launch of this call (the conv kernel, its zeroing, a
            # two-stage Stage 2) to ``stream``.
            import rocke.runtime.launcher as _L

            def cfg_cls(**kw):
                kw["stream"] = stream
                return LaunchConfig(**kw)

        self.stream = stream
        for ptrs in self.ptr_sets:
            if stream:
                import rocke.runtime.launcher as _L

                _saved = _L.LaunchConfig
                _L.LaunchConfig = cfg_cls
            try:
                bound = B._cached_bind_batch(
                    chunks, direction, run_problem, ident, meta, rt, launcher, dtype, split_k,
                    ptrs, self.sizes, arch, stage2_kernels, KernelLauncher, cfg_cls, self.live_extras,
                )
            finally:
                if stream:
                    _L.LaunchConfig = _saved
            if bound is None:
                raise SystemExit(f"{key}: {self.label} cannot launch this problem")
            self.runs.append(bound[0])
        self.flops = float(problem.flops)

    def run(self, i=0):
        self.runs[i % len(self.runs)]()


def _u8(host_bits, nbytes):
    import ctypes

    return (ctypes.c_uint8 * nbytes).from_address(host_bits.ctypes.data)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pins", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--arch", default="gfx942")
    ap.add_argument("--mode", action="append", choices=["staged", "parts", "b2b", "step", "ab", "stepsel", "energy"])
    ap.add_argument("--cases", default=None, help="comma list of case keys (default: all pinned)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=1, help="staged/parts: interleaved rounds over the cases")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup-steps", type=int, default=3)
    ap.add_argument("--step-reps", type=int, default=1, help="step mode: repetitions (each its own drain)")
    ap.add_argument("--per-call", action="store_true", help="step mode: an event between consecutive calls")
    ap.add_argument("--share", action="store_true", help="one buffer set per case (default: one per call)")
    ap.add_argument("--no-clock", action="store_true")
    ap.add_argument("--variant", action="append", default=[], metavar="NAME:K=V[,K=V]",
                    help="also bind every case with these env vars set (e.g. HIP_FORCE_DEV_KERNARG=1); "
                         "'ab' and 'step' modes then time base and variants interleaved")
    ap.add_argument("--pins-variant", action="append", default=[], metavar="NAME:PINS",
                    help="a variant that runs other pins (selection judging); buffers are shared")
    ap.add_argument("--sched", type=int, default=-1,
                    help="hipSetDeviceFlags(value) before the HIP context: 1 spin, 2 yield, 4 blocking sync "
                         "(default: leave HIP's default, a spin-wait that keeps a CPU core of the socket busy)")
    ap.add_argument("--energy-s", type=float, default=0.6, help="energy mode: seconds of back-to-back calls per visit")
    ap.add_argument("--energy-out", default=None, metavar="PINS",
                    help="energy mode: write pins choosing each case's lowest-energy candidate here")
    ap.add_argument("--streams-variant", action="store_true",
                    help="step mode: add variant '2stream': fwd and the dgrad chain on one non-blocking stream, every "
                         "wgrad on a second one (waiting only for its dY), joined at the end of the step")
    ap.add_argument("--streams-shapes", default=None,
                    help="with --streams-variant: only these shapes' wgrads go to the second stream (e.g. 7,8,10)")
    ap.add_argument("--stepsel-out", default=None, metavar="PINS",
                    help="stepsel mode: write the pins it selects here")
    ap.add_argument("--stepsel-margin", type=float, default=0.0,
                    help="stepsel: switch from the pin only if the in-step median is this fraction lower")
    ap.add_argument("--zero-variants", action="store_true",
                    help="step mode: add variants 'arena' (every zeroing memset of the step replaced by ONE memset of "
                         "their total bytes into one arena at the start of the step -- what a caller that hands "
                         "pre-zeroed outputs / workspace from one zeroed arena would pay, cf. zero_grad) and "
                         "'nomem' (the zeroing skipped: the lower bound). Timing only: outputs are not zeroed.")
    ap.add_argument("--tied-variant", action="store_true",
                    help="add variant 'tied': each layer's calls share their tensors as training does -- dgrad reads "
                         "the forward's W, wgrad reads dgrad's dY and the forward's X (default: every case has its own "
                         "buffers, so the backward pair never finds dY in MALL / L2)")
    ap.add_argument("--graph-variant", action="store_true",
                    help="step mode: add 'stream' (the step on one non-blocking stream) and 'graph' (that step "
                         "captured once into a hipGraph and replayed with one hipGraphLaunch per step: no host "
                         "launch cost at all)")
    ap.add_argument("--json", default=None, help="append a result row here")
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)
    modes = args.mode or ["staged", "step"]

    sys.meta_path.insert(0, _NoTorch())
    from pathlib import Path

    from benchmarks.common import benchmark_implicit_gemm_conv as B
    from benchmarks.common.kernel_cache import KernelCache
    from rocke.runtime import synchronize_and_release
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.launcher import no_fence
    from rocke.runtime.stall_gate import time_launches_staged

    if args.sched < 0:
        from benchmarks.common.batched_timing import apply_hip_device_flags

        apply_hip_device_flags()  # ROCKE_HIP_DEVICE_FLAGS, as the sweeps honour it
    if args.sched >= 0:
        import ctypes

        from rocke.runtime import hip_module as _hm

        rc = _hm._resolve_hip().hipSetDeviceFlags(ctypes.c_uint(args.sched))
        print(f"[replay] hipSetDeviceFlags({args.sched}) -> {rc}", flush=True)
    pins = json.load(open(args.pins))["cases"]
    keys = args.cases.split(",") if args.cases else list(pins)
    need = set(keys)
    if "step" in modes:
        need |= {f"{d}_s{s}" for d, s in step_sequence()}
        missing = sorted(k for k in need if k not in pins)
        if missing:
            raise SystemExit(f"step mode needs pins for {missing}")
    rt = Runtime()
    cache = KernelCache(Path(args.cache), args.arch)
    max_bytes = 0
    for k in need:
        problem, _, _ = B.parse_miopen_cmd(pins[k]["cmd"])
        p = problem
        max_bytes = max(max_bytes, 2 * p.N * max(p.Di * p.Hi * p.Wi * p.C, p.Do * p.Ho * p.Wo * p.K))
    host_bits = _bf16_random(max_bytes // 2, 42)
    stage2 = {}
    t0 = time.time()
    cases = {}
    for k in sorted(need):
        cases[k] = Case(k, pins[k], cache, args.arch, B, rt, host_bits, stage2, args)
    variants = {"base": cases}
    for spec in args.variant:
        name, _, kv = spec.partition(":")
        env = dict(x.split("=", 1) for x in kv.split(",") if x)
        saved = {k_: os.environ.get(k_) for k_ in env}
        os.environ.update(env)
        try:
            variants[name] = {
                k: Case(k, pins[k], cache, args.arch, B, rt, host_bits, stage2, args, share_from=cases[k])
                for k in sorted(need)
            }
        finally:
            for k_, v_ in saved.items():
                if v_ is None:
                    os.environ.pop(k_, None)
                else:
                    os.environ[k_] = v_
    for spec in args.pins_variant:
        name, _, path = spec.partition(":")
        vp = json.load(open(path))["cases"]
        variants[name] = {
            k: Case(k, dict(pins[k], kernel=vp[k]["kernel"]) if k in vp else pins[k], cache, args.arch, B, rt,
                    host_bits, stage2, args, share_from=cases[k])
            for k in sorted(need)
        }
    if args.tied_variant:
        from types import SimpleNamespace

        tied = {}
        for k in sorted(need):
            d, sh = k.split("_s")
            f, g = cases.get(f"fwd_s{sh}"), cases.get(f"dgrad_s{sh}")
            sets = []
            for i, own in enumerate(cases[k].ptr_sets):
                # fwd (X, W, Y); dgrad (dY, W, dX); wgrad (dY, X, dW)
                if d == "dgrad" and f is not None:
                    own = (own[0], f.ptr_sets[i % len(f.ptr_sets)][1], own[2])
                elif d == "wgrad":
                    dy = g.ptr_sets[i % len(g.ptr_sets)][0] if g is not None else own[0]
                    x = f.ptr_sets[i % len(f.ptr_sets)][0] if f is not None else own[1]
                    own = (dy, x, own[2])
                sets.append(own)
            tied[k] = Case(k, pins[k], cache, args.arch, B, rt, None, stage2, args,
                           share_from=SimpleNamespace(ptr_sets=sets))
        variants["tied"] = tied
    streams = None
    if args.streams_variant:
        import ctypes

        from rocke.runtime import hip_module as _hm

        _hip = _hm._resolve_hip()
        streams = []
        for _ in range(2):
            h = ctypes.c_void_p(0)
            rc = _hip.hipStreamCreateWithFlags(ctypes.byref(h), ctypes.c_uint(1))  # hipStreamNonBlocking
            if rc:
                raise SystemExit(f"hipStreamCreateWithFlags -> {rc}")
            streams.append(int(h.value))
        _s1 = {int(x) for x in args.streams_shapes.split(",")} if args.streams_shapes else None

        def _on_s1(k):
            return _s1 is None or int(k.split("_s")[1]) in _s1

        _wait = _hip.hipStreamWaitEvent
        _wait.argtypes = [ctypes.c_void_p, _hm._HipEventHandle, ctypes.c_uint]
        variants["2stream"] = {
            k: Case(k, pins[k], cache, args.arch, B, rt, None, stage2, args, share_from=cases[k],
                    stream=streams[1] if k.startswith("wgrad") and _on_s1(k) else streams[0])
            for k in sorted(need)
        }
    gstream = None
    if args.graph_variant:
        import ctypes

        from rocke.runtime import hip_module as _hm

        _hipg = _hm._resolve_hip()
        h = ctypes.c_void_p(0)
        rc = _hipg.hipStreamCreateWithFlags(ctypes.byref(h), ctypes.c_uint(1))  # hipStreamNonBlocking
        if rc:
            raise SystemExit(f"hipStreamCreateWithFlags -> {rc}")
        gstream = int(h.value)
        gcases = {k: Case(k, pins[k], cache, args.arch, B, rt, None, stage2, args, share_from=cases[k], stream=gstream)
                  for k in sorted(need)}
        variants["stream"] = gcases
        variants["graph"] = gcases
    del host_bits
    rt.sync()
    print(f"[replay] bound {len(cases)} cases x {len(variants)} variant(s) in {time.time() - t0:.1f} s", flush=True)
    gpu = os.environ.get("HIP_VISIBLE_DEVICES", "0").split(",")[0]
    sampler = None if args.no_clock else ClockSampler(gpu)
    if sampler:
        sampler.start()
    out = {"sched": args.sched, "tag": args.tag, "pins": args.pins, "cache": args.cache, "gpu": gpu, "time": time.strftime("%H:%M:%S")}

    if "b2b" in modes:
        from rocke.runtime.launcher import time_launches

        b2b = {k: [] for k in keys}
        for r in range(args.rounds):
            for k in keys:
                c = cases[k]
                b2b[k].append(time_launches(c.run, warmup=args.warmup, iters=args.iters * 3))
        print(f"\n[replay] back-to-back same call (mean of {args.iters * 3}), median over rounds")
        out["b2b"] = {}
        for k in keys:
            ms = statistics.median(b2b[k])
            out["b2b"][k] = round(ms, 4)
            ref = cases[k].ref_ms
            print(f"  {k:10s} {ms:8.4f} ms" + (f"  ref {ref:.4f} ({(ms / ref - 1) * 100:+.1f}%)" if ref else ""), flush=True)

    if "ab" in modes:
        from benchmarks.common.batched_timing import time_interleaved_staged

        names = list(variants)
        print(f"\n[replay] ab: staged medians, variants interleaved per case ({args.rounds} x {args.iters})")
        print("  " + " " * 11 + "".join(f"{n:>12s}" for n in names))
        out["ab"] = {}
        tot = {n: 0.0 for n in names}
        for k in keys:
            labels = [variants[n][k].label for n in names]
            # one timing per distinct (label, env) callable; identical pins of a
            # pins-variant reuse the base timing
            t = time_interleaved_staged(
                [variants[n][k].run for n in names], rounds=args.rounds, iters=args.iters, warmup=args.warmup
            )
            out["ab"][k] = {n: (round(x, 4) if x else None) for n, x in zip(names, t)}
            out["ab"][k]["labels"] = labels
            for n, x in zip(names, t):
                tot[n] += (x or 0.0) * cases[k].calls
            line = f"  {k:10s}" + "".join(f"{x:12.4f}" for x in t)
            line += "  " + " ".join(f"{(x / t[0] - 1) * 100:+.1f}%" for x in t[1:])
            if len(set(labels)) > 1:
                line += "  [" + " ".join("=" if lb == labels[0] else "*" for lb in labels[1:]) + "]"
            print(line, flush=True)
        print("  calls-weighted sum: " + "  ".join(f"{n} {v:.3f}" for n, v in tot.items()))
        out["ab_sum"] = {n: round(v, 4) for n, v in tot.items()}

    if "energy" in modes:
        # Energy per call of each case's candidates (pins[k]["alts"], pin first):
        # back-to-back calls for --energy-s seconds per visit, socket energy from
        # the amdsmi accumulator, candidates interleaved over --rounds. At the
        # power cap a sustained step's time is its energy over the cap, so the
        # candidate that does a call with the fewest joules is the one that
        # leaves the most clock to every other call.
        import amdsmi

        amdsmi.amdsmi_init()
        hnd = amdsmi.amdsmi_get_processor_handles()[int(gpu)]

        def _joules():
            e = amdsmi.amdsmi_get_energy_count(hnd)
            return e["energy_accumulator"] * e["counter_resolution"] * 1e-6

        res_e = {}
        print(f"\n[replay] energy: J/call and ms/call back to back ({args.rounds} rounds x {args.energy_s} s per candidate)")
        for k in keys:
            labs = pins[k].get("alts") or [pins[k]["kernel"]]
            if labs[0] != pins[k]["kernel"]:
                labs = [pins[k]["kernel"]] + [x for x in labs if x != pins[k]["kernel"]]
            cs = [cases[k]] + [
                Case(k, dict(pins[k], kernel=lab), cache, args.arch, B, rt, None, stage2, args, share_from=cases[k])
                for lab in labs[1:]
            ]
            jj = [[] for _ in cs]
            tt = [[] for _ in cs]
            for r in range(args.rounds):
                for c_i in [(j + r) % len(cs) for j in range(len(cs))]:
                    c = cs[c_i]
                    with no_fence():
                        for _ in range(3):
                            c.run()
                    rt.sync()
                    n_calls = 0
                    e0 = _joules()
                    t0_ = time.perf_counter()
                    with no_fence():
                        while time.perf_counter() - t0_ < args.energy_s:
                            for _ in range(8):
                                c.run()
                            n_calls += 8
                            if n_calls % 64 == 0:
                                rt.sync()
                    rt.sync()
                    t1_ = time.perf_counter()
                    e1 = _joules()
                    jj[c_i].append((e1 - e0) / n_calls)
                    tt[c_i].append((t1_ - t0_) * 1e3 / n_calls)
                    synchronize_and_release()
            je = [statistics.median(x) for x in jj]
            te = [statistics.median(x) for x in tt]
            best = min(range(len(cs)), key=lambda i: je[i])
            res_e[k] = {"labels": [c.label for c in cs], "j": [round(x, 5) for x in je], "ms": [round(x, 4) for x in te], "pick": cs[best].label}
            print(
                f"  {k:10s} " + "  ".join(
                    f"{te[i]:.4f}ms/{je[i] * 1e3:.1f}mJ{'*' if i == best else ''}" for i in range(len(cs))
                ) + (f"  -> alt {best} ({(je[best] / je[0] - 1) * 100:+.1f}% J)" if best else ""),
                flush=True,
            )
            for c in cs[1:]:
                B._free_extras(rt, c.live_extras)
        out["energy"] = res_e
        if args.energy_out:
            newp = {"source": f"energy {args.pins}", "cases": {k: dict(v, kernel=res_e[k]["pick"]) if k in res_e else v for k, v in pins.items()}}
            json.dump(newp, open(args.energy_out, "w"), indent=1)

    if "stepsel" in modes:
        # In-step selection: every case's candidates (pins[k]["alts"], the pin
        # first) take turns inside back-to-back training steps -- candidate
        # (step + offset_k) % n_k runs all of case k's calls in that step, with
        # an event between consecutive calls -- so each candidate is timed in
        # the cache / clock / power context the real step gives it.
        seq = step_sequence()
        calls = []
        seen = {}
        for d, s_ in seq:
            k = f"{d}_s{s_}"
            calls.append((k, seen.get(k, 0)))
            seen[k] = seen.get(k, 0) + 1
        alt = {}
        for k in sorted({k for k, _ in calls}):
            labs = pins[k].get("alts") or [pins[k]["kernel"]]
            if labs[0] != pins[k]["kernel"]:
                labs = [pins[k]["kernel"]] + [x for x in labs if x != pins[k]["kernel"]]
            alt[k] = [cases[k]] + [
                Case(k, dict(pins[k], kernel=lab), cache, args.arch, B, rt, None, stage2, args, share_from=cases[k])
                for lab in labs[1:]
            ]
        off = {k: j for j, k in enumerate(sorted(alt))}
        times = {k: [[] for _ in alt[k]] for k in alt}
        n_steps = args.steps
        evs = [rt.event() for _ in range(len(calls) + 1)]
        with no_fence():
            for st in range(args.warmup_steps + n_steps):
                pick = {k: (st + off[k]) % len(alt[k]) for k in alt}
                evs[0].record()
                for j, (k, i) in enumerate(calls):
                    alt[k][pick[k]].run(i)
                    evs[j + 1].record()
                evs[-1].synchronize()
                if st >= args.warmup_steps:
                    for j, (k, i) in enumerate(calls):
                        times[k][pick[k]].append(evs[j].elapsed_to(evs[j + 1]))
        for e in evs:
            e.destroy()
        synchronize_and_release()
        sel = {}
        print(f"\n[replay] stepsel: in-step medians per candidate ({n_steps} steps, candidates rotate per step)")
        for k in sorted(alt):
            med = [statistics.median(t) if t else float("inf") for t in times[k]]
            best = min(range(len(med)), key=lambda c: med[c])
            if med[best] > med[0] * (1 - args.stepsel_margin):
                best = 0
            sel[k] = alt[k][best].label
            print(
                f"  {k:10s} pin {med[0]:.4f}  "
                + " ".join(f"{m:.4f}{'*' if c == best else ''}" for c, m in enumerate(med[1:], 1))
                + (f"  -> alt {best} ({(med[best] / med[0] - 1) * 100:+.1f}%)" if best else ""),
                flush=True,
            )
        out["stepsel"] = {k: {"labels": [c.label for c in alt[k]],
                              "med": [round(statistics.median(t), 4) if t else None for t in times[k]],
                              "pick": sel[k]} for k in alt}
        if args.stepsel_out:
            newp = {"source": f"stepsel {args.pins}", "cases": {k: dict(v, kernel=sel.get(k, v["kernel"])) for k, v in pins.items()}}
            json.dump(newp, open(args.stepsel_out, "w"), indent=1)
            print(f"[replay] selected pins -> {args.stepsel_out}", flush=True)

    if "staged" in modes or "parts" in modes:
        from benchmarks.common.batched_timing import time_interleaved_staged

        # rt.memset_async, switchable: "parts" times each call whole, with its
        # zeroing memsets skipped, and the memsets alone -- interleaved.
        import types

        _real_memset = rt.memset_async
        gate = {"on": True, "log": None}

        def _memset(self, ptr, value, nbytes, stream=0):
            if gate["log"] is not None:
                gate["log"].append((ptr, value, nbytes, stream))
            if gate["on"]:
                _real_memset(ptr, value, nbytes, stream)

        rt.memset_async = types.MethodType(_memset, rt)
        res = {}
        if "staged" in modes:
            ts = time_interleaved_staged(
                [cases[k].run for k in keys], rounds=args.rounds, iters=args.iters, warmup=args.warmup
            )
            print(f"\n[replay] staged per call: median of {args.rounds} interleaved rounds x {args.iters} samples")
            for k, ms in zip(keys, ts):
                ref = cases[k].ref_ms
                res[k] = {"ms": round(ms, 4), "ref_ms": ref}
                print(f"  {k:10s} {ms:8.4f} ms" + (f"  ref {ref:.4f} ({(ms / ref - 1) * 100:+.1f}%)" if ref else ""), flush=True)
        if "parts" in modes:
            print(f"\n[replay] parts: whole call / memsets skipped / memsets alone (interleaved, {args.rounds} x {args.iters})")
            for k in keys:
                c = cases[k]
                log = []
                gate["log"], gate["on"] = log, False
                c.run()
                gate["log"], gate["on"] = None, True
                synchronize_and_release()

                def nomem(c=c):
                    gate["on"] = False
                    try:
                        c.run()
                    finally:
                        gate["on"] = True

                def memonly(log=tuple(log)):
                    for a_ in log:
                        _real_memset(*a_)

                fns = [c.run, nomem] + ([memonly] if log else [])
                t = time_interleaved_staged(fns, rounds=args.rounds, iters=args.iters, warmup=args.warmup)
                me = t[2] if log else 0.0
                row = res.setdefault(k, {})
                row.update(whole_ms=round(t[0], 4), nomemset_ms=round(t[1], 4), memset_ms=round(me, 4), n_memsets=len(log))
                print(
                    f"  {k:10s} whole {t[0]:8.4f}  no-memset {t[1]:8.4f} ({(t[1] - t[0]) * 1e3:+6.1f} us)"
                    f"  memsets alone {me * 1e3:5.1f} us ({len(log)})",
                    flush=True,
                )
        rt.memset_async = _real_memset
        out["staged"] = res

    if "step" in modes:
        seq = step_sequence()
        calls = []
        seen = {}
        for d, s in seq:
            k = f"{d}_s{s}"
            i = seen.get(k, 0)
            seen[k] = i + 1
            calls.append((k, i))
        flops_step = sum(cases[k].flops for k, _ in calls)
        zero = None
        if args.zero_variants:
            import types

            _real_ms = rt.memset_async
            zero = {"on": True, "log": []}

            def _zms(self, ptr, value, nbytes, stream=0):
                if zero["log"] is not None:
                    zero["log"].append(nbytes)
                if zero["on"]:
                    _real_ms(ptr, value, nbytes, stream)

            rt.memset_async = types.MethodType(_zms, rt)
            for k, i in calls:
                cases[k].run(i)
            rt.sync()
            zero["arena_bytes"] = sum(zero["log"])
            zero["n"] = len(zero["log"])
            zero["log"] = None
            zero["arena"] = rt.alloc(max(zero["arena_bytes"], 256))
            print(f"[replay] zero variants: {zero['n']} memsets per step, {zero['arena_bytes'] / 2**20:.1f} MiB "
                  f"-> arena: one memset per step", flush=True)
            variants["arena"] = cases
            variants["nomem"] = cases
        gexec = None
        if gstream is not None:
            from rocke.runtime.launcher import no_fence as _nf

            _hipg.hipStreamBeginCapture.argtypes = [ctypes.c_void_p, ctypes.c_int]
            _hipg.hipStreamEndCapture.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
            _hipg.hipGraphInstantiate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
                                                  ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
            _hipg.hipGraphLaunch.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            rt.sync()
            with _nf():
                rc = _hipg.hipStreamBeginCapture(ctypes.c_void_p(gstream), 2)  # relaxed
                if rc:
                    raise SystemExit(f"hipStreamBeginCapture -> {rc}")
                for k, i in calls:
                    variants["graph"][k].run(i)
                g = ctypes.c_void_p(0)
                rc = _hipg.hipStreamEndCapture(ctypes.c_void_p(gstream), ctypes.byref(g))
                if rc:
                    raise SystemExit(f"hipStreamEndCapture -> {rc}")
            ge = ctypes.c_void_p(0)
            rc = _hipg.hipGraphInstantiate(ctypes.byref(ge), g, None, None, 0)
            if rc:
                raise SystemExit(f"hipGraphInstantiate -> {rc}")
            gexec = ge
            n_nodes = ctypes.c_size_t(0)
            _hipg.hipGraphGetNodes(g, None, ctypes.byref(n_nodes))
            print(f"[replay] graph variant: one step captured, {n_nodes.value} nodes", flush=True)
            # Same outputs as the eager step: run each, read every output buffer, compare (bf16 -> f32).
            import numpy as _np

            def _outs():
                res = {}
                for k in sorted({k for k, _ in calls}):
                    for ps in cases[k].ptr_sets:
                        nb = cases[k].sizes[2]
                        buf = (ctypes.c_uint8 * nb)()
                        rt.memcpy_d2h(buf, ps[2], nb)
                        a = _np.frombuffer(bytes(buf), dtype=_np.uint16).astype(_np.uint32) << 16
                        res[(k, ps[2])] = a.view(_np.float32).copy()
                return res

            for k, i in calls:
                cases[k].run(i)
            rt.sync()
            eager = _outs()
            # poison outputs so a graph that writes nothing cannot pass
            for (k, ptr) in eager:
                rt.memset(ptr, 0x7F, cases[k].sizes[2])
            _hipg.hipGraphLaunch(gexec, ctypes.c_void_p(gstream))
            rt.sync()
            gout = _outs()
            worst = 0.0
            for key_, a in eager.items():
                b = gout[key_]
                den = float(_np.max(_np.abs(a))) or 1.0
                worst = max(worst, float(_np.max(_np.abs(a - b))) / den)
            print(f"[replay] graph check: {len(eager)} outputs, max |eager - graph| / max|eager| = {worst:.3e} "
                  f"({'PASS' if worst < 1e-2 else 'FAIL'})", flush=True)
        reps = []
        for rep in range(args.step_reps):
            _vl = list(variants.items())
            _sh = rep % len(_vl)
            for vname, vcases in _vl[_sh:] + _vl[:_sh]:  # rotated per rep: no fixed position in a rep
                rt.sync()
                n_ev = args.steps + 1
                step_ev = [rt.event() for _ in range(n_ev)]
                call_ev = (
                    [[rt.event() for _ in range(len(calls) + 1)] for _ in range(args.steps)]
                    if args.per_call
                    else None
                )
                two = vname == "2stream"
                s0 = streams[0] if two else (gstream if vname in ("stream", "graph") else 0)
                if two:
                    dep = [rt.event() for _ in range(len(calls))]
                    join = rt.event()

                zmode = vname if vname in ("arena", "nomem") and zero is not None else None
                if zero is not None:
                    zero["on"] = zmode is None

                def one_step(st=None):
                    if vname == "graph":
                        rc_ = _hipg.hipGraphLaunch(gexec, ctypes.c_void_p(gstream))
                        if rc_:
                            raise SystemExit(f"hipGraphLaunch -> {rc_}")
                        return
                    if not two:
                        if call_ev and st is not None:
                            call_ev[st][0].record(stream=s0)
                        if zmode == "arena":
                            _real_ms(zero["arena"], 0, zero["arena_bytes"], 0)
                        for j, (k, i) in enumerate(calls):
                            vcases[k].run(i)
                            if call_ev and st is not None:
                                call_ev[st][j + 1].record(stream=s0)
                        return
                    last = None
                    for j, (k, i) in enumerate(calls):
                        if k.startswith("dgrad"):
                            # dY of this layer is ready now (before its dgrad).
                            dep[j].record(stream=streams[0])
                            last = dep[j]
                        elif k.startswith("wgrad") and _on_s1(k):
                            _wait(ctypes.c_void_p(streams[1]), last.handle, 0)
                        vcases[k].run(i)
                    join.record(stream=streams[1])
                    _wait(ctypes.c_void_p(streams[0]), join.handle, 0)

                with no_fence():
                    for _ in range(args.warmup_steps):
                        one_step()
                    ta = time.monotonic()
                    step_ev[0].record(stream=s0)
                    for st in range(args.steps):
                        one_step(st)
                        step_ev[st + 1].record(stream=s0)
                    step_ev[-1].synchronize()
                    tb = time.monotonic()
                if zero is not None:
                    zero["on"] = True
                if two:
                    rt.sync()
                    for e in dep + [join]:
                        e.destroy()
                synchronize_and_release()
                step_ms = [step_ev[i].elapsed_to(step_ev[i + 1]) for i in range(args.steps)]
                row = {
                    "step_ms_median": round(statistics.median(step_ms), 4),
                    "step_ms_mean": round(statistics.mean(step_ms), 4),
                    "step_ms_min": round(min(step_ms), 4),
                    "steps": [round(x, 4) for x in step_ms],
                    "tflops_eff": round(flops_step / (statistics.median(step_ms) * 1e9), 1),
                }
                if call_ev and not two and vname != "graph":
                    per_key = {}
                    for st in range(args.steps):
                        for j, (k, i) in enumerate(calls):
                            per_key.setdefault(k, []).append(call_ev[st][j].elapsed_to(call_ev[st][j + 1]))
                    row["per_case_ms"] = {
                        k: round(statistics.median(v), 4) for k, v in sorted(per_key.items())
                    }
                    dirs = {}
                    for k, v in per_key.items():
                        d = k.split("_")[0]
                        dirs[d] = dirs.get(d, 0.0) + statistics.median(v) * cases[k].calls
                    row["per_dir_ms"] = {d: round(v, 3) for d, v in dirs.items()}
                    for e in sum(call_ev, []):
                        e.destroy()
                if sampler:
                    row["clock"] = sampler.summary(ta, tb)
                for e in step_ev:
                    e.destroy()
                row["variant"] = vname
                reps.append(row)
                print(
                    f"\n[replay] step rep {rep} [{vname}]: median {row['step_ms_median']:.3f} ms  mean {row['step_ms_mean']:.3f}"
                    f"  min {row['step_ms_min']:.3f}  ({row['tflops_eff']} TF)  clock {row.get('clock')}",
                    flush=True,
                )
                if call_ev and "per_dir_ms" in row:
                    print("  per dir (sum of per-case medians x calls): " + json.dumps(row["per_dir_ms"]))
                    for k, v in row["per_case_ms"].items():
                        ref = cases[k].ref_ms
                        print(f"    {k:10s} {v:8.4f}" + (f"  (staged ref {ref:.4f}, {(v / ref - 1) * 100:+.1f}%)" if ref else ""))
        out["step"] = reps
        if zero is not None:
            rt.memset_async = _real_ms
            rt.free(zero["arena"])
            out["zero"] = {"n_memsets": zero["n"], "arena_bytes": zero["arena_bytes"]}
        if len(variants) > 1:
            print("\n[replay] step summary (median over reps of the per-rep medians):")
            for vname in variants:
                xs = [r["step_ms_median"] for r in reps if r["variant"] == vname]
                print(f"  {vname:12s} {statistics.median(xs):8.3f} ms  reps {' '.join(f'{x:.3f}' for x in xs)}")

    if sampler:
        sampler.stop()
        out["clock_all"] = sampler.summary()
    if args.json:
        with open(args.json, "a") as f:
            f.write(json.dumps(out) + "\n")
    for c in {id(c): c for v in variants.values() for c in v.values()}.values():
        B._free_extras(rt, c.live_extras)
        for b in c.bufs:
            rt.free(b)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
