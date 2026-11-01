# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A live engine backend for the layer framework's toy chain (gfx950).

The toy chain (``toy_chain.py``) is the framework's lever testbed: exact int32
math (a host model checks every mailbox and output word), the framework's
hand-offs, entry ramp and prefetch levers, and a recorded optimum per shape
(``toy_chain.best_spec``, found by hill climbing with lookback). This backend
lets the search engine tune it on hardware under the tuner's rules:

- ``compile``: build + compile (Python engine where a knob needs it), and the
  code object's resources (scratch / spills refuse the config);
- ``check``: the numerics gate BEFORE any timing: two chained launches at
  two steps (one straddling the tag wrap), every mailbox, tag, output word
  and counter against the host model. The first check of a session runs the
  gate's own must-fail mutant (a corrupted weight word on the device): if
  the gate does not see it, the session stops (a gate that cannot fail
  proves nothing);
- ``measure_batch``: every item of the batch plus the baseline arm, timed in
  ``fidelity.reps`` rounds with a seeded shuffled order per round; each arm
  is one captured graph of ``launches`` chained launches (+ the step
  increment), replayed after a READ cache flush (a write flush leaves dirty
  MALL lines whose write-back lands in the timed call, alola rule 50);
  every arm's state is re-checked against the host model after timing;
- :meth:`confirm`: the DESIGN 9.1 decision round: baseline, its A-vs-A
  control (a second instance on its own buffers), its recompiled control (a
  separately loaded code object), the finalists; shuffled rounds; per-round
  ratios; :func:`~.gate.win_rule`; SCLK / power sampled per round and
  throttled rounds dropped.

Times stay in the backend's records (absolute: private store only); the
search sees samples, the reports carry ratios.
"""

from __future__ import annotations

import os
import random
import statistics
import struct
import time
from collections.abc import Mapping, Sequence

import numpy as np

from rocke.dispatch.tuning.search.measure import CompileOutcome
from rocke.dispatch.tuning.search.trial import Fidelity

from . import gate as G
from .knobs import KnobAdapter

__all__ = ["ToyChainBackend", "build_read_flush"]

ARCH = "gfx950"
FLUSH_THREADS = 256


def build_read_flush():
    """A read-only cache flush: every thread loads a strided dword stream of
    ``src`` and stores one xor per thread (a few KiB), so no dirty line of
    the flushed footprint is left to write back inside a timed call."""
    from rocke.core.ir import I32, IRBuilder, PtrType

    b = IRBuilder("rk_tune_read_flush")
    b.kernel.attrs["max_workgroup_size"] = FLUSH_THREADS
    src = b.param("src", PtrType(I32, "global"))
    out = b.param("out", PtrType(I32, "global"))
    n = b.param("n", I32)
    stride = b.param("stride", I32)
    gid = b.add(b.mul(b.block_id_x(), b.const_i32(FLUSH_THREADS)), b.thread_id_x())
    loop = b.scf_for_iter(b.const_i32(0), n, b.const_i32(1), [("fl_acc", b.const_i32(0))], iv_name="fl_it")
    with loop as (it, (acc,)):
        v = b.global_load_i32(src, b.add(b.mul(it, stride), gid))
        b.scf_yield(b.xor(acc, v))
    b.global_store(out, gid, loop.results[0], align=4)
    b.ret()
    return b.kernel


class _Arm:
    """One timed instance: buffers, a function handle, a captured graph."""

    def __init__(self, name, spec, bufs, fn, graph, nbufs):
        self.name = name
        self.spec = spec
        self.bufs = bufs
        self.fn = fn
        self.graph = graph
        self.nbufs = nbufs


class ToyChainBackend:
    """Engine backend over :func:`.catalog.toy_chain` configs.

    ``launches``: chained launches per timed graph (each with its own layer
    weights, layer ids ``k * layers`` < 63). ``replays``: graph replays per
    round (the round's sample is their median). ``flush_bytes``: read-flush
    footprint before each replay (0: warm). ``seed``: round shuffles."""

    def __init__(
        self,
        adapter: KnobAdapter,
        *,
        launches: int = 12,
        replays: int = 3,
        flush_bytes: int = 1 << 30,
        seed: int = 0,
        watchdog_s: float = 120.0,
        sample_clocks: bool = True,
    ):
        from benchmarks.gfx950.qwen38_mono import toy_chain_bench as TB
        from rocke.examples.gfx950.persistent_mailbox import harness as H

        from .. import mailbox_ir  # noqa: F401  (registers impl="ir")
        from .. import toy_chain as T

        self.T, self.TB, self.H = T, TB, H
        prob = H.env_problem()
        if prob:
            raise RuntimeError(f"no usable GPU runtime: {prob}")
        self.adapter = adapter
        self.launches = int(launches)
        self.replays = int(replays)
        self.flush_bytes = int(flush_bytes)
        self.seed = int(seed)
        self.sample_clocks = bool(sample_clocks)
        self.dev = H.Dev(watchdog_s=watchdog_s)
        self.step_art = TB.compile_art(T.build_step_kernel())[0]
        self.gate_selftest: dict | None = None
        self.records: list[dict] = []  # absolute per-round times (private)
        self._batch = 0
        self.compiles = 0
        self.bdf = _bdf()
        if self.flush_bytes:
            fl, _ = H.compile_for(build_read_flush(), ARCH)
            self._flush_fn = self.dev.function(fl)
            self._flush_src = self.dev.alloc(self.flush_bytes, zero=True)
            self._flush_out = self.dev.alloc(4 << 20, zero=True)
            threads = (4 << 20) // 4
            self._flush_grid = threads // FLUSH_THREADS
            self._flush_args = struct.pack(
                "<QQii", self._flush_src, self._flush_out, self.flush_bytes // 4 // threads, threads
            )

    # ----------------------------------------------------------- identity
    def environment(self) -> dict:
        env = {k: os.environ.get(k) for k in ("ROCKE_CODEOBJ_BACKEND", "ROCKE_DIRECT_LEVERS", "ROCKE_BACKEND")}
        try:
            from rocke.runtime import hip_module as hm

            dev = hm.current_device()
            env["target_id"] = hm.get_device_target_id(dev)
            env["num_cus"] = hm.get_device_num_cus(dev)
        except Exception:  # noqa: BLE001
            pass
        return {"toolchain": {"codeobj": env.pop("ROCKE_CODEOBJ_BACKEND"),
                              "direct_levers": env.pop("ROCKE_DIRECT_LEVERS")}, **env}  # fmt: skip

    def spec_of(self, config: Mapping[str, object]):
        return self.adapter.build(self.adapter.canonical_knobs(config))

    # ------------------------------------------------------------- compile
    def compile(self, config, key):
        spec = self.spec_of(config)
        t0 = time.perf_counter()
        art, used = self.TB.compile_art(spec)
        self.compiles += 1
        res = _resources(art.hsaco)
        return CompileOutcome(True, artifact={"spec": spec, "art": art, "backend": used},
                              seconds=time.perf_counter() - t0, resources=res)  # fmt: skip

    # --------------------------------------------------------------- check
    def _alloc(self, spec, chain):
        n0 = len(self.dev._bufs)
        bufs = self.TB.alloc(self.dev, spec, chain=chain)
        return bufs, n0

    def _free_from(self, n0: int) -> None:
        extra = self.dev._bufs[n0:]
        del self.dev._bufs[n0:]
        self.dev.sync()
        for p in extra:
            self.dev.rt.free(p)

    def _mutate(self, spec, bufs, mutant: str):
        """Host arrays of a planted fault in the LAST launch of the chain
        (``weight``: one level-0 weight word of producer 0; ``x``: one
        activation word), or ``None`` when the chain has no weights. The
        fault sits in the last launch on purpose: the toy's int32 chain
        multiplies a level's ``r_prev`` by ``P (P + 1) / 2`` (12880 = 2^4 x
        805 at P = 160), so after 8 levels any fault confined to an EARLIER
        launch vanishes mod 2^32 from that launch's output, and the check
        reads only the last launch's mailboxes plus every launch's output."""
        if spec.work == 0:
            return None
        last = 1 * spec.layers  # launch 1 of the 2-launch chain, its first layer
        w, x = bufs.w_host.copy(), bufs.x_host.copy()
        if mutant == "weight":
            j = int(np.flatnonzero(x)[0])
            w[last, 0, 0, j] = np.int32(int(w[last, 0, 0, j]) ^ 0x10)
        elif mutant == "x":
            j = int(np.flatnonzero(w[last, 0, 0])[0])
            x[j] = np.int32(int(x[j]) ^ 0x10)
        else:
            raise ValueError(mutant)
        return w, x

    def _check_once(self, spec, art, *, mutant: str = "") -> list[str]:
        """The numerics gate on fresh buffers: two chained launches at two
        steps (one near the tag wrap; fresh step values per call, so no tag of
        an earlier run can satisfy a poll) against the host model: every
        mailbox of the last launch, every launch's output word.
        ``mutant`` (:meth:`_mutate`): the fault is planted on the device only;
        the host model is first shown to change (an unobservable mutant would
        prove nothing), then the gate must report errors."""
        import dataclasses

        self._nchecks = getattr(self, "_nchecks", 0) + 1
        steps = (5 + 2 * self._nchecks, (1 << 26) - 1 - (self._nchecks % 7))
        bufs, n0 = self._alloc(spec, 2)
        try:
            if mutant:
                mut = self._mutate(spec, bufs, mutant)
                if mut is None:
                    return [f"(no {mutant} mutant on a hop-only chain)"]
                w, x = mut
                fake = dataclasses.replace(bufs, w_host=w, x_host=x)
                layers0 = [0, spec.layers]
                observable = False
                for st in steps:
                    a, b_ = self.TB.model_chain(bufs, st, layers0), self.TB.model_chain(fake, st, layers0)
                    observable |= any(
                        int(a[k]["out"][li]) != int(b_[k]["out"][li]) for k in range(2) for li in range(spec.layers)
                    ) or any(
                        not np.array_equal(a[-1]["y"][li][i], b_[-1]["y"][li][i])
                        for li in range(spec.layers) for i in range(spec.depth)
                    )  # fmt: skip
                if not observable:
                    raise ValueError(f"{mutant} mutant does not change the host model: an invalid self-test")
                self.dev.write(bufs.w[0], w)
                self.dev.write(bufs.x, x)
            errs = []
            for step in steps:
                self.TB.run_chain_once(self.dev, bufs, art, step=step, layers0=[0, spec.layers])
                errs += self.TB.check_state(self.dev, bufs, step, [0, spec.layers])
            return errs
        finally:
            self._free_from(n0)

    def selftest(self) -> dict:
        """The gate's must-fail mutants on the baseline build, BEFORE any
        timing: both must be caught and a clean run must pass, else the
        session is refused (every later check fails too)."""
        if self.gate_selftest is not None:
            return self.gate_selftest
        base = self._artifact_of_base()
        spec, art = base["spec"], base["art"]
        out = {}
        for m in ("weight", "x"):
            errs = self._check_once(spec, art, mutant=m)
            out[m] = {"caught": bool(errs), "first": errs[:1]}
        clean = self._check_once(spec, art)
        out["clean_passes"] = not clean
        out["clean_first"] = clean[:1]
        out["ok"] = all(v["caught"] for v in out.values() if isinstance(v, dict)) and out["clean_passes"]
        self.gate_selftest = out
        return out

    def check(self, config, key, artifact):
        spec, art = artifact["spec"], artifact["art"]
        if not self.selftest()["ok"]:
            raise RuntimeError(f"numerics gate self-test failed {self.gate_selftest}: nothing may be timed")
        errs = self._check_once(spec, art)
        return (not errs), ("; ".join(errs[:3]) if errs else "ok")

    # --------------------------------------------------------------- timing
    def _make_arm(self, name, spec, art, *, fresh_module: bool = False) -> _Arm:
        bufs, n0 = self._alloc(spec, self.launches)
        if fresh_module:
            mod = self.dev.rt.load_module(art.hsaco)
            self.dev._mods.append(mod)
            fn = mod.get_function(art.kernel_name)
        else:
            fn = self.dev.function(art)
        sfn = self.dev.function(self.step_art)
        TB, s = self.TB, spec

        def issue(stream: int) -> None:
            for k in range(self.launches):
                self.dev.launch(fn, s.grid, s.threads, TB.launch_args(bufs, k, k * s.layers, 0), stream=stream)
            self.dev.launch(sfn, 1, 64, struct.pack("<Q", bufs.step), stream=stream)

        self.dev.write(bufs.step, np.array([1000], np.int32))
        g = self.H._Graph(self.dev)
        g.capture(issue)
        g.replay()  # warm: code object, TLB
        return _Arm(name, spec, bufs, fn, g, n0)

    def _time_arm(self, arm: _Arm) -> float:
        g = arm.graph
        times = []
        for _ in range(self.replays):
            if self.flush_bytes:
                self.dev.launch(self._flush_fn, self._flush_grid, FLUSH_THREADS, self._flush_args, stream=g.stream.value)
            e0, e1 = self.dev.rt.event(), self.dev.rt.event()
            e0.record(g.stream.value)
            g.lib.hipGraphLaunch(g.exec, g.stream)
            e1.record(g.stream.value)
            e1.synchronize()
            times.append(e0.elapsed_to(e1) / self.launches)
            e0.destroy()
            e1.destroy()
        return statistics.median(times)

    def _check_arm(self, arm: _Arm) -> list[str]:
        step = int(self.TB.read_i32(self.dev, arm.bufs.step, 1)[0]) - 1
        return self.TB.check_state(self.dev, arm.bufs, step, [k * arm.spec.layers for k in range(self.launches)])

    def _close(self, arms: Sequence[_Arm]) -> None:
        for a in sorted(arms, key=lambda a: -a.nbufs):
            a.graph.close()
        n0 = min((a.nbufs for a in arms), default=len(self.dev._bufs))
        self._free_from(n0)

    def _rounds(self, arms: dict[str, _Arm], rounds: int, seed: int, sampler=None) -> list[dict[str, float]]:
        def run(name, r):
            if sampler is not None:
                sampler.mark(f"{r}:{name}")
            return self._time_arm(arms[name])

        return G.shuffled_rounds(list(arms), rounds, run, seed=seed)

    def measure(self, config, key, artifact, fidelity: Fidelity):
        return self.measure_batch([(config, key, artifact)], fidelity)[0]

    def measure_batch(self, items, fidelity: Fidelity):
        """Raw per-round times (one sample per round) of every item; the
        baseline arm runs in every round too (its times go to the record)."""
        self._batch += 1
        base_art = self._artifact_of_base()
        arms: dict[str, _Arm] = {}
        try:
            arms[G.BASE] = self._make_arm(G.BASE, base_art["spec"], base_art["art"])
            for i, (_cfg, key, art) in enumerate(items):
                arms[f"i{i}:{key}"] = self._make_arm(key, art["spec"], art["art"])
            rows = self._rounds(arms, max(1, fidelity.reps), self.seed * 7919 + self._batch)
            bad = {n: self._check_arm(a) for n, a in arms.items()}
        finally:
            self._close(list(arms.values()))
        self.records.append({"batch": self._batch, "rounds": rows, "post_check": {n: e[:2] for n, e in bad.items()}})
        out = []
        for i, (_cfg, key, _art) in enumerate(items):
            name = f"i{i}:{key}"
            if bad[name]:
                raise RuntimeError(f"{key}: state wrong after timing: {bad[name][:2]}")
            out.append([row[name] for row in rows])
        return out

    _base_cache: dict | None = None

    def _artifact_of_base(self):
        if self._base_cache is None:
            self._base_cache = self.compile({}, "base").artifact
        return self._base_cache

    # -------------------------------------------------------- confirmation
    def confirm(self, finalists: Mapping[str, Mapping], *, rounds: int = 12, seed: int = 1,
                margin: float = 0.002) -> dict:  # fmt: skip
        """The DESIGN 9.1 decision: baseline, A-vs-A control, recompiled
        control and ``finalists`` (``{name: config}``), ``rounds`` shuffled
        rounds, per-round ratios, the win rule per finalist."""
        if not self.selftest()["ok"]:  # the gate's must-fail mutants first, whoever timed the search
            raise RuntimeError(f"numerics gate self-test failed {self.gate_selftest}")
        base_spec = self.spec_of({})
        base_art = self.TB.compile_art(base_spec)[0]
        rc_art = self.TB.compile_art(self.T.build_toy_chain(base_spec))[0]  # compiled again
        arms: dict[str, _Arm] = {}
        sampler = G.ClockSampler(bdf=self.bdf) if self.sample_clocks else None
        try:
            arms[G.BASE] = self._make_arm(G.BASE, base_spec, base_art)
            arms[G.CTL_AVA] = self._make_arm(G.CTL_AVA, base_spec, base_art)
            arms[G.CTL_RC] = self._make_arm(G.CTL_RC, base_spec, rc_art, fresh_module=True)
            for name, cfg in finalists.items():
                spec = self.spec_of(cfg)
                art = self.TB.compile_art(spec)[0]
                errs = self._check_once(spec, art)
                if errs:
                    raise RuntimeError(f"finalist {name} fails numerics: {errs[:2]}")
                arms[name] = self._make_arm(name, spec, art)
            if sampler is not None:
                with sampler:
                    rows = self._rounds(arms, rounds, seed, sampler)
            else:
                rows = self._rounds(arms, rounds, seed)
            bad = {n: self._check_arm(a) for n, a in arms.items()}
        finally:
            self._close(list(arms.values()))
        ratios = G.per_round_ratios(rows)
        drop: list[int] = []
        clocks = {}
        if sampler is not None and sampler.samples:
            per = sampler.per_label()
            by_round: dict[int, list[float]] = {}
            for lab, v in per.items():
                if ":" in lab and v["sclk"]:
                    by_round.setdefault(int(lab.split(":", 1)[0]), []).append(v["sclk"])
            rsclk = {r: statistics.median(v) for r, v in by_round.items()}
            drop = G.throttled_rounds(rsclk)
            clocks = {"source": sampler.source, "samples": len(sampler.samples), "dropped_rounds": drop,
                      "sclk_rel_spread": (max(rsclk.values()) / min(rsclk.values()) - 1.0) if rsclk else None}  # fmt: skip
        controls = [ratios[G.CTL_AVA], ratios[G.CTL_RC]]
        verdicts = {}
        for name in finalists:
            verdicts[name] = G.win_rule(ratios[name], controls, margin=margin, drop=drop).to_json()
        self.records.append({"confirm": True, "rounds": rows, "post_check": {n: e[:2] for n, e in bad.items()}})
        return {
            "ratios": ratios,
            "verdicts": verdicts,
            "controls": {G.CTL_AVA: ratios[G.CTL_AVA], G.CTL_RC: ratios[G.CTL_RC]},
            "post_check_ok": not any(bad.values()),
            "clocks": clocks,
            "gate_selftest": self.gate_selftest,
        }

    def close(self) -> None:
        self.dev.close()


def _bdf() -> str:
    """PCI address of the current HIP device (``0000:65:00.0``), or ``""``."""
    try:
        from rocke.runtime import hip_module as hm

        p = hm._device_props(hm.current_device())
        return f"{int(p.pciDomainID):04x}:{int(p.pciBusID):02x}:{int(p.pciDeviceID):02x}.0"
    except Exception:  # noqa: BLE001 - clocks are best effort
        return ""


def _resources(hsaco: bytes) -> dict:
    """Scratch / spill counts of a code object from its AMDHSA kernel
    descriptor notes via ``llvm-readelf --notes`` (best effort; ``{}`` when
    the tool is missing)."""
    import re
    import shutil
    import subprocess
    import tempfile

    exe = shutil.which("llvm-readelf") or (
        "/opt/rocm/llvm/bin/llvm-readelf" if os.path.exists("/opt/rocm/llvm/bin/llvm-readelf") else None
    )
    if exe is None:
        return {}
    with tempfile.NamedTemporaryFile(suffix=".hsaco", delete=False) as f:
        f.write(hsaco)
        path = f.name
    try:
        r = subprocess.run([exe, "--notes", path], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return {}
    finally:
        os.unlink(path)
    out = {}
    for key, name in ((".private_segment_fixed_size", "scratch_bytes"), (".vgpr_spill_count", "spill_vgpr"),
                      (".sgpr_spill_count", "spill_sgpr"), (".vgpr_count", "vgpr"), (".sgpr_count", "sgpr")):  # fmt: skip
        m = re.search(re.escape(key) + r":\s+(\d+)", r.stdout)
        if m:
            out[name] = float(m.group(1))
    return out


def shuffled(seq, seed):
    s = list(seq)
    random.Random(seed).shuffle(s)
    return s
