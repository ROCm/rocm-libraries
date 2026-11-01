# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Evaluating candidates in cluster jobs.

A search can run in two places:

- **in the job** (the search driver itself runs on the GPU node): the
  engine's :class:`~rocke.dispatch.tuning.search.measure.Evaluator` with an
  in-process backend (:class:`.live_toy.ToyChainBackend`, ...). Cheapest per
  evaluation; one job per search.
- **from the login side** (the driver runs anywhere): :class:`JobEvaluator`
  prefetches every round's fresh candidates as ONE batch job (``runner``:
  compile with a CPU pool, numerics gate with its must-fail mutant, then the
  interleaved shuffled timing rounds, all on the node), stores the results in
  a :class:`PrefetchedBackend` and lets the engine's evaluation logic read
  them: statuses map one to one (compile error, spill, numeric failure,
  timeout), so strategies, the noise model, caches and checkpoints behave as
  in-process. The batch is the unit of pairing: every item of a job is timed
  in the same shuffled rounds against the same baseline.

Transports: :class:`LocalTransport` runs the runner in a subprocess (tests,
or a driver already inside a GPU job); :class:`Q38Transport` submits a q38
job through the cluster wrapper named by ``Q38_SUBMIT`` (e.g. the
``truemono-lb/submit.sh`` pattern: submit, detach the poller) and polls the
job's log / results over ssh (``Q38_LOGIN``, ``Q38_AUS``: the Austin q38 root);
all three come from the environment, never from this tree.
"""

from __future__ import annotations

import base64
import json
import math
import os
import shlex
import subprocess
import sys
import tempfile
import time
import zlib
from collections.abc import Mapping, Sequence
from pathlib import Path

from rocke.dispatch.tuning.search.measure import CompileOutcome, Evaluator
from rocke.dispatch.tuning.search.trial import Fidelity, Status, Trial

__all__ = [
    "JobEvaluator",
    "LocalTransport",
    "PrefetchedBackend",
    "Q38Transport",
    "decode_batch",
    "encode_batch",
]


def encode_batch(batch: Mapping) -> str:
    """A batch as a compact shell-safe token (zlib + urlsafe base64)."""
    return base64.urlsafe_b64encode(zlib.compress(json.dumps(batch, sort_keys=True).encode(), 9)).decode()


def decode_batch(token: str) -> dict:
    return json.loads(zlib.decompress(base64.urlsafe_b64decode(token.encode())).decode())


class PrefetchedBackend:
    """Results of batch jobs as an engine backend: ``compile`` / ``check`` /
    ``measure`` replay what the job recorded for a key (a key never sent to a
    job fails to compile: the evaluator only asks for prefetched keys)."""

    def __init__(self, environment: Mapping | None = None):
        self.results: dict[str, dict] = {}
        self._env = dict(environment or {})
        self.samples_batch: dict[str, int] = {}

    def environment(self) -> dict:
        return dict(self._env)

    def compile(self, config, key):
        r = self.results.get(key)
        if r is None:
            return CompileOutcome(False, detail="not measured by any batch job")
        if r["status"] == Status.COMPILE_ERROR:
            return CompileOutcome(False, detail=r.get("detail", "compile error"))
        return CompileOutcome(True, artifact=key, seconds=float(r.get("compile_s", 0.0)),
                              resources=r.get("resources", {}))  # fmt: skip

    def check(self, config, key, artifact):
        r = self.results[key]
        if r["status"] == Status.NUMERIC_FAIL:
            return False, r.get("detail", "numeric failure")
        return True, "checked in the job"

    def measure(self, config, key, artifact, fidelity: Fidelity):
        r = self.results[key]
        if r["status"] != Status.OK:
            raise RuntimeError(f"{r['status']}: {r.get('detail', '')}")
        return list(r["samples"])

    def measure_batch(self, items, fidelity: Fidelity):
        return [self.measure(c, k, a, fidelity) for c, k, a in items]


class LocalTransport:
    """Runs the batch runner in a subprocess of this machine."""

    def __init__(self, python: str = sys.executable, env: Mapping | None = None, timeout: float = 3600.0):
        self.python = python
        self.env = dict(os.environ, **dict(env or {}))
        self.timeout = float(timeout)

    def run(self, batch: Mapping) -> dict:
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "results.json"
            cmd = [self.python, "-m", "kernels.gfx950.qwen38_mono.tune", "runner", "--batch", encode_batch(batch),
                   "--out", str(out)]  # fmt: skip
            r = subprocess.run(cmd, env=self.env, capture_output=True, text=True, timeout=self.timeout)
            if not out.exists():
                raise RuntimeError(f"runner failed (rc {r.returncode}): {r.stderr[-2000:]}")
            return json.loads(out.read_text())


class Q38Transport:
    """One q38 job per batch. Environment: ``Q38_SUBMIT`` (the submit
    wrapper: ``<log> <q38_job.sh args>``), ``Q38_LOGIN`` (ssh host),
    ``Q38_AUS`` (the Austin q38 root holding ``logs/`` and ``results/``).
    ``stage``: the rocKE stage name (q38_sync.sh)."""

    def __init__(self, stage: str, *, tag_prefix: str = "se-batch", gpus: int = 1, time_limit: str = "00:40:00",
                 poll_s: float = 30.0, timeout_s: float = 3 * 3600.0, extra_env: Sequence[str] = ()):  # fmt: skip
        for var in ("Q38_SUBMIT", "Q38_LOGIN", "Q38_AUS"):
            if not os.environ.get(var):
                raise RuntimeError(f"Q38Transport needs {var} in the environment")
        self.submit = os.environ["Q38_SUBMIT"]
        self.login = os.environ["Q38_LOGIN"]
        self.aus = os.environ["Q38_AUS"]
        self.stage = stage
        self.tag_prefix = tag_prefix
        self.gpus = int(gpus)
        self.time_limit = time_limit
        self.poll_s = float(poll_s)
        self.timeout_s = float(timeout_s)
        self.extra_env = list(extra_env)
        self.n = 0
        self.jobs: list[dict] = []

    def _ssh(self, cmd: str) -> str:
        r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", self.login, cmd],
                           capture_output=True, text=True, timeout=120)  # fmt: skip
        return r.stdout

    def run(self, batch: Mapping) -> dict:
        self.n += 1
        tag = f"{self.tag_prefix}-{int(time.time())}-{self.n}"
        token = encode_batch(batch)
        # the venv job pins ROCKE_COMGR_LIB to torch's comgr; the HIP runtime
        # must be torch's too, or the runner's GPU check fails (two COMGRs
        # mapped; seen in the first cluster run of this transport, WP1 M12)
        hip = ("export ROCKE_HIP_LIB=${ROCKE_HIP_LIB:-$(python -c 'import os, torch; "
               "print(os.path.join(os.path.dirname(torch.__file__), \"lib\", \"libamdhip64.so\"))')}; ")
        inner = (f"{hip}python -m kernels.gfx950.qwen38_mono.tune runner --batch {token} "
                 f"--out $Q38_OUT/results.json")  # fmt: skip
        args = ["--tag", tag, "--gpus", str(self.gpus), "--time", self.time_limit, "--stage", self.stage,
                "--mode", "venv"]  # fmt: skip
        for e in self.extra_env:
            args += ["--env", e]
        log = Path(tempfile.gettempdir()) / f"{tag}.submit.log"
        cmd = " ".join(shlex.quote(x) for x in ["bash", self.submit, str(log), *args, "--", inner])
        subprocess.run(cmd, shell=True, timeout=7200)
        text = log.read_text(errors="replace") if log.exists() else ""
        jid = None
        for line in text.splitlines():
            if line.startswith("[q38_job] submitted"):
                jid = line.split()[2]
        if jid is None:
            raise RuntimeError(f"batch job {tag} was not submitted: {text[-1000:]}")
        self.jobs.append({"tag": tag, "job": jid, "items": len(batch.get("items", []))})
        t0 = time.time()
        while time.time() - t0 < self.timeout_s:
            out = self._ssh(f"grep -a '^== rc=' {self.aus}/logs/{tag}-{jid}.out 2>/dev/null | tail -1")
            if out.strip():
                break
            time.sleep(self.poll_s)
        else:
            raise TimeoutError(f"batch job {tag} ({jid}) did not finish")
        res = self._ssh(f"cat {self.aus}/results/{tag}/results.json 2>/dev/null")
        if not res.strip():
            log_tail = self._ssh(f"tail -40 {self.aus}/logs/{tag}-{jid}.out")
            raise RuntimeError(f"batch job {tag} wrote no results: {log_tail}")
        return json.loads(res)


class JobEvaluator(Evaluator):
    """An :class:`Evaluator` whose fresh candidates are measured by batch
    jobs: before each engine evaluation, the round's not-yet-known keys go
    to ``transport.run`` as one batch (``target``: what the runner builds,
    e.g. ``{"kind": "toy_chain", "depth": 8, "work": 8, "layers": 1}``),
    and the results are served by the :class:`PrefetchedBackend`."""

    def __init__(self, space, target: Mapping, transport, *, ladder: Sequence[Fidelity] = (Fidelity(),),
                 environment: Mapping | None = None, **kw):  # fmt: skip
        self.prefetched = PrefetchedBackend({"target": dict(target)})
        super().__init__(space, self.prefetched, ladder=ladder, environment=environment, **kw)
        self.target = dict(target)
        self.transport = transport
        self.batches: list[dict] = []

    def _fresh(self, trials: Sequence[Trial]) -> dict[str, dict]:
        out = {}
        for t in trials:
            key = self.space.key(t.genome)
            if key is None or self.known(key, t.fidelity) is not None or key in self.prefetched.results:
                continue
            out[key] = self.space.decode(t.genome)
        return out

    def evaluate(self, trials, *, max_fresh=None, deadline=None):
        fresh = self._fresh(trials)
        if max_fresh is not None:
            fresh = dict(list(fresh.items())[: max(0, int(max_fresh))])
        if fresh:
            fid = self.ladder[-1]
            batch = {"target": self.target, "fidelity": fid.to_json(),
                     "items": [{"key": k, "config": c} for k, c in fresh.items()]}  # fmt: skip
            res = self.transport.run(batch)
            self.batches.append({"items": len(fresh), "meta": res.get("meta", {})})
            for r in res["items"]:
                self.prefetched.results[r["key"]] = r
            for k in fresh:
                self.prefetched.results.setdefault(k, {"status": Status.RUN_ERROR, "detail": "missing from the job"})
        return super().evaluate(trials, max_fresh=max_fresh, deadline=deadline)


def run_batch(batch: Mapping) -> dict:
    """Job side: compile, gate and time one batch (``target.kind``
    ``toy_chain``); returns ``{"items": [{key, status, samples, ...}],
    "meta": {...}}``. Absolute samples go back to the driver (its record is
    private); the console gets ratios only."""
    target = batch["target"]
    fid = Fidelity.from_json(batch["fidelity"])
    if target.get("kind") != "toy_chain":
        raise ValueError(f"unknown batch target {target!r}")
    from . import catalog as C
    from .live_toy import ToyChainBackend

    ad = C.toy_chain(int(target["depth"]), int(target["work"]), layers=int(target.get("layers", 1)))
    be = ToyChainBackend(ad, launches=int(target.get("launches", 12)), replays=int(target.get("replays", 3)),
                         seed=int(target.get("seed", 0)))  # fmt: skip
    items, ok_items = [], []
    gate = be.selftest()  # the gate's must-fail mutants before anything of this batch is timed
    if not gate["ok"]:
        be.close()
        return {"items": [{"key": it["key"], "status": Status.RUN_ERROR, "detail": f"gate self-test failed: {gate}"}
                          for it in batch["items"]], "meta": {"gate_selftest": gate}}  # fmt: skip
    for it in batch["items"]:
        key, cfg = it["key"], it["config"]
        rec = {"key": key}
        try:
            comp = be.compile(cfg, key)
        except Exception as e:  # noqa: BLE001 - a build failure is an outcome
            items.append({**rec, "status": Status.COMPILE_ERROR, "detail": f"{type(e).__name__}: {e}"})
            continue
        rec["compile_s"] = comp.seconds
        rec["resources"] = dict(comp.resources)
        spilled = {k: v for k, v in comp.resources.items() if k in ("scratch_bytes", "spill_vgpr", "spill_sgpr") and v}
        if spilled:
            items.append({**rec, "status": Status.SPILL, "detail": f"spills {spilled}"})
            continue
        try:
            ok, why = be.check(cfg, key, comp.artifact)
        except Exception as e:  # noqa: BLE001
            items.append({**rec, "status": Status.RUN_ERROR, "detail": f"check {type(e).__name__}: {e}"})
            continue
        if not ok:
            items.append({**rec, "status": Status.NUMERIC_FAIL, "detail": why})
            continue
        ok_items.append((cfg, key, comp.artifact, rec))
    if ok_items:
        try:
            samples = be.measure_batch([(c, k, a) for c, k, a, _r in ok_items], fid)
        except Exception as e:  # noqa: BLE001 - the batch's timing failed: every item is a run error
            samples = None
            for _c, _k, _a, rec in ok_items:
                items.append({**rec, "status": Status.RUN_ERROR, "detail": f"timing {type(e).__name__}: {e}"})
        rows = be.records[-1]["rounds"] if samples is not None and be.records else []
        for j, ((_c, k, _a, rec), s) in enumerate(zip(ok_items, samples or [])):
            name = f"i{j}:{k}"
            ratio = None
            if rows and all(name in r and r.get("base") for r in rows):
                ratio = sorted(r[name] / r["base"] for r in rows)[len(rows) // 2]
            items.append({**rec, "status": Status.OK, "samples": [float(x) for x in s], "ratio_vs_base": ratio})
    meta = {"gate_selftest": be.gate_selftest, "compiles": be.compiles, "host": os.uname().nodename
            if hasattr(os, "uname") else ""}  # fmt: skip
    be.close()
    return {"items": items, "meta": meta}


def finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)
