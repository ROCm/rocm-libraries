# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Adaptive kernel search for the GEMM sweep driver (``--search-config``).

``GemmSearchAdapter`` is the GEMM ``OperationAdapter`` of the op-agnostic
coordinator in ``tile_engine/ops/common/adaptive_search.py``. It reuses the
driver's own build and measure functions, which the driver passes in as
``DriverHooks`` (this module never imports the driver):

- candidates: per problem, the kernels a full run would measure with it, one
  per kernel name (with ``--pairs``: only its problems, and the listed kernels
  among those);
- builds: ``setup_multiple_gemm_dispatchers`` on the round's missing kernels;
  each built .so is copied to ``<out>/artifacts/<name>.so`` with its sha256 and
  only those copies are ever loaded (a file the run did not copy is ignored,
  a copy that changed on disk is refused);
- measurements: ``benchmark_on_devices`` with --verify on, one kernel per worker
  and one problem entry per request, so every row and failure record belongs to
  exactly one request; the worker also checks that the library reports the
  requested kernel (``check_identity``). Each call writes its own CSV under
  ``<out>/rounds/``, next to a JSON file mapping its ``problem_idx`` to the
  request (call id, trial) and holding its failure records;
- confirmation: the same path with fresh calls.

``reconcile`` turns a request's rows and failure records into its one outcome;
``read_ranked_csv`` turns ``select_kernels.py --out_ranked`` predictions into
the latency prior of ``warm_start.ranked_csv``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

_COMMON_DIR = str(Path(__file__).resolve().parents[1] / "common")
if _COMMON_DIR not in sys.path:
    sys.path.insert(0, _COMMON_DIR)

from adaptive_search import (  # noqa: E402
    EXIT_USAGE,
    BuildResult,
    Candidate,
    PairOutcome,
    PairRequest,
    RunWriter,
    SearchCoordinator,
    SearchSettings,
    SettingsError,
    valid_latency,
)

SEARCH_VARIANT = "gemm_universal"
# Driver options a search run does not take: it writes its own outputs and
# confirms its finalists itself.
CONFLICTS = (
    ("--list-pairs", "list_pairs"),
    ("--repeat-top", "repeat_top"),
    ("--failures-csv", "failures_csv"),
)
# Driver failure kind (--failures-csv) -> outcome status. The first kind in
# this order that a request has decides its outcome.
FAILURE_STATUS = {
    "timeout": "timeout",
    "batch-error": "run_error",
    "error": "run_error",
    "missing": "run_error",
    "worker-exit": "run_error",
    "unsupported": "run_error",
    "bad-output": "invalid_measurement",
    "mismatch": "numeric_fail",
}
# dispatcher_run_gemm codes in "kernel returned status N" (gemm_ctypes_lib.cpp).
# The pairing should have excluded both: they point at an eligibility gap.
RUN_STATUS = {
    "-2": "no kernel selected for the problem",
    "-3": "the kernel does not support the problem",
}


@dataclass(frozen=True)
class DriverHooks:
    """The driver functions the search reuses."""

    build: Callable  # setup_multiple_gemm_dispatchers
    measure: Callable  # benchmark_on_devices
    record: Callable  # _kernel_record: config -> its scalar fields
    problem_key: Callable  # problem dict -> "MxNxK"
    load_pairs: Callable  # load_pair_list, reads warm_start.pairs_file
    build_reason: Callable  # build_failure_summary: build error -> one line


def check_search_options(args) -> SearchSettings:
    """The validated ``--search-config``. Raises SettingsError for a conflicting
    option, invalid settings or an output directory holding an earlier run."""
    if not (args.search_config and args.search_output_dir):
        raise SettingsError(
            "--search-config and --search-output-dir go together, each with a path")
    for opt, attr in CONFLICTS:
        if getattr(args, attr):
            raise SettingsError(f"--search-config does not take {opt}")
    if args.variant != SEARCH_VARIANT:
        raise SettingsError(
            f"--search-config supports --variant {SEARCH_VARIANT} only, "
            f"got {args.variant}"
        )
    settings = SearchSettings.from_json(args.search_config)
    RunWriter(args.search_output_dir).check_fresh()
    return settings


def parse_verified(value) -> Optional[bool]:
    """A row's ``verified``: True only for True or "True" (rows in memory carry
    a bool, the CSV the string), False for False or "False", else None."""
    if value is True or (isinstance(value, str) and value == "True"):
        return True
    if value is False or (isinstance(value, str) and value == "False"):
        return False
    return None


def _failure_diag(fail) -> str:
    kind, err = fail.get("kind", ""), str(fail.get("error", ""))
    m = re.search(r"status (-\d+)", err)
    why = RUN_STATUS.get(m.group(1)) if m else None
    return f"{kind}: {err}" + (f" ({why})" if why else "")


def reconcile(req: PairRequest, problem: dict, rows: Sequence[dict],
              fails: Sequence[dict]):
    """(status, latency_ms, diag) of one request from its measured rows and
    failure records. Only a verified row of the requested kernel and problem
    with a finite positive latency, and no failure record, is ``ok``."""
    if fails:
        kinds = [f.get("kind") for f in fails]
        kind = next((k for k in FAILURE_STATUS if k in kinds), kinds[0])
        status = FAILURE_STATUS.get(kind, "run_error")
        return status, None, _failure_diag(fails[kinds.index(kind)])
    if not rows:
        return "unattempted", None, "no row and no failure: never run"
    if len(rows) > 1:
        return "invalid_measurement", None, f"{len(rows)} rows for one request"
    row = rows[0]
    got = tuple(str(row.get(f)) for f in ("kernel", "M", "N", "K"))
    want = (req.kernel_name, *(str(problem[d]) for d in "MNK"))
    if got != want:
        return "run_error", None, f"row {got} is not the requested pair {want}"
    verified = parse_verified(row.get("verified"))
    if verified is not True:
        status = "numeric_fail" if verified is False else "invalid_measurement"
        diag = f"verified={row.get('verified')!r} max_rel={row.get('max_rel')!r}"
        return status, None, diag
    lat = row.get("latency_ms")
    if not valid_latency(lat):
        return "invalid_measurement", None, f"latency_ms={lat!r}"
    return "ok", float(lat), ""


def read_ranked_csv(path) -> Dict[str, Dict[str, float]]:
    """The search prior from ``select_kernels.py --out_ranked``: {problem:
    {kernel: predicted latency ms}} with latency = 2*M*N*K / (predicted_tflops
    * 1e9); rows without a finite positive prediction are skipped. Raises
    ValueError (OSError when unreadable) for a file it cannot use."""
    out: Dict[str, Dict[str, float]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        need = {"problem", "kernel", "predicted_tflops"}
        if not need <= set(reader.fieldnames or ()):
            raise ValueError(f"{path}: expected columns {sorted(need)}")
        for i, row in enumerate(reader, 2):
            try:  # a short row holds None
                m, n, k = (int(d) for d in row["problem"].split("x"))
                tflops = float(row["predicted_tflops"])
            except (AttributeError, TypeError, ValueError):
                raise ValueError(
                    f"{path} line {i}: expected an MxNxK problem and a number, "
                    f"got {row['problem']!r}, {row['predicted_tflops']!r}"
                ) from None
            if math.isfinite(tflops) and tflops > 0:
                out.setdefault(row["problem"], {})[row["kernel"]] = (
                    2 * m * n * k / (tflops * 1e9))
    return out


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_artifact(src, dst: Path) -> str:
    """Copy ``src`` over ``dst`` (temp file + rename); the copy's sha256."""
    tmp = dst.with_name(f".{dst.name}.tmp")
    shutil.copyfile(src, tmp)
    digest = _sha256(tmp)
    tmp.replace(dst)
    return digest


def _git_commit() -> Optional[str]:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(Path(__file__).resolve().parent),
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return (r.stdout.strip() or None) if r.returncode == 0 else None


class GemmSearchAdapter:
    """``OperationAdapter`` for gemm_universal (see the module docstring).

    ``eligible[i]`` lists the kernel names ``problems[i]`` may run, in order;
    ``worker`` is the driver's ``(worker_path, base_env)``; ``context`` adds
    the driver's build context (arch, layout, variant, configs) to run.json.
    """

    def __init__(self, args, hooks: DriverHooks, problems, configs, eligible,
                 devices, worker, out_dir, context=None):
        if len(problems) != len(eligible):
            raise ValueError(f"{len(problems)} problems, {len(eligible)} pair lists")
        self.args = args
        self.hooks = hooks
        self.devices = list(devices)
        self.worker_path, self.base_env = worker
        self.out_dir = Path(out_dir)
        self._context = dict(context or {})
        # Verified, one kernel per worker: a batch failure names one request.
        self.margs = argparse.Namespace(**{**vars(args), "verify": True, "batch_size": 1})
        by_name = {}
        for cfg in configs:
            by_name.setdefault(cfg.name, cfg)
        self.cands: Dict[str, Candidate] = {}
        self.problems: Dict[str, dict] = {}
        self.eligible: Dict[str, List[Candidate]] = {}
        self.repeated = []  # problem keys listed more than once, searched once
        for prob, names in zip(problems, eligible):
            key = hooks.problem_key(prob)
            if key in self.problems:
                self.repeated.append(key)
                continue
            self.problems[key] = {d: int(prob[d]) for d in "MNK"}
            self.eligible[key] = [self._cand(by_name[n]) for n in dict.fromkeys(names)]
        self.artifacts: Dict[str, tuple] = {}  # key -> (path, sha256), this run's copies
        self._files = 0

    def _cand(self, cfg) -> Candidate:
        c = self.cands.get(cfg.name)
        if c is None:
            c = Candidate(cfg.name, cfg.name, self.hooks.record(cfg), cfg)
            self.cands[cfg.name] = c
        return c

    # -------------------------------------------------- OperationAdapter

    def context(self) -> dict:
        a = self.args
        files = [*self._context.get("configs", ()), a.problems, a.pairs]
        domain = {k: [c.kernel_name for c in v] for k, v in self.eligible.items()}
        notes = ["candidates are the pairs a full run measures when every kernel builds"]
        if not a.no_vector_fallback:
            notes.append("the fixed-vector-width kernels a full run falls back to "
                         "when a native kernel fails to build are not candidates")
        if a.pairs:
            notes.append(f"--pairs {a.pairs} restricts the problems to its keys and "
                         "the candidates to its pairs")
        if a.max_kernels:
            notes.append(f"--max-kernels {a.max_kernels} limits the expansion")
        if self.repeated:
            notes.append(f"repeated problems searched once: {sorted(set(self.repeated))}")
        return {
            **self._context,
            "dtype": a.dtype,
            "problems_file": a.problems,
            "pairs_file": a.pairs,
            "max_kernels": a.max_kernels,
            "vector_fallback": not a.no_vector_fallback,
            "tune_c_vector_width": a.tune_c_vector_width,
            "devices": self.devices,
            "measurement": {"verify": True, "verify_tol": a.verify_tol,
                            "batch_size": 1, "kernel_timeout": a.kernel_timeout,
                            "check_identity": True},
            "build": {"workers": a.workers, "build_timeout": a.build_timeout},
            "sources": {"ck_commit": _git_commit(),
                        "sha256": {str(p): _sha256(p) for p in files if p}},
            "domain_sha256": hashlib.sha256(
                json.dumps(domain, sort_keys=True).encode()).hexdigest(),
            "domain_notes": notes,
        }

    def enumerate_eligible(self) -> Dict[str, List[Candidate]]:
        return {k: list(v) for k, v in self.eligible.items()}

    def build_missing(self, cands: Sequence[Candidate]) -> Dict[str, BuildResult]:
        errors: Dict[str, str] = {}
        paths = self.hooks.build(
            [c.original_config for c in cands], verbose=False,
            max_workers=self.args.workers, errors=errors,
            build_timeout=self.args.build_timeout,
        )
        if len(paths) != len(cands):
            raise RuntimeError(f"build returned {len(paths)} paths for {len(cands)} kernels")
        art_dir = self.out_dir / "artifacts"
        art_dir.mkdir(parents=True, exist_ok=True)
        out = {}
        for c, lib in zip(cands, paths):
            if lib is None:
                why = self.hooks.build_reason(errors.get(c.kernel_name))
                out[c.canonical_key] = BuildResult(None, why or "build failed")
                continue
            dst = art_dir / f"{c.kernel_name}.so"
            digest = _copy_artifact(lib, dst)
            self.artifacts[c.canonical_key] = (dst, digest)
            out[c.canonical_key] = BuildResult(str(dst), digest=digest)
        return out

    def measure_pairs(self, requests: Sequence[PairRequest]) -> List[PairOutcome]:
        return self._measure(requests, "measure")

    def confirm(self, requests: Sequence[PairRequest]) -> List[PairOutcome]:
        return self._measure(requests, "confirm")

    # ----------------------------------------------------------- measure

    def _measure(self, requests, kind) -> List[PairOutcome]:
        out, run, usable = {}, [], {}
        for r in requests:
            key = r.candidate_key
            if key not in usable:
                usable[key] = self._unusable(key)
            if usable[key]:
                out[r.call_id] = PairOutcome(r, usable[key][0], None, usable[key][1])
            else:
                run.append(r)
        if run:
            rows, fails = self._run(run, kind)
            for i, r in enumerate(run):
                out[r.call_id] = PairOutcome(r, *reconcile(
                    r, self.problems[r.problem_key], rows[i], fails[i]))
        return [out[r.call_id] for r in requests]

    def _unusable(self, key):
        """(status, diag) when ``key`` has no usable artifact of this run."""
        art = self.artifacts.get(key)
        if art is None:
            return "compile_error", "not built in this run"
        try:
            same = _sha256(art[0]) == art[1]
        except OSError:
            same = False
        return None if same else ("run_error", f"artifact {art[0]} changed since its build")

    def _run(self, reqs, kind):
        """One measure call over ``reqs``; their rows and failures by index."""
        built, index = [], {}
        for r in reqs:
            if r.candidate_key not in index:
                index[r.candidate_key] = len(built)
                cfg = self.cands[r.candidate_key].original_config
                built.append((cfg, self.artifacts[r.candidate_key][0]))
        rounds = self.out_dir / "rounds"
        rounds.mkdir(parents=True, exist_ok=True)
        while True:  # one new file per call, never an existing one
            self._files += 1
            csv_path = rounds / f"{self._files:04d}-{kind}.csv"
            if not csv_path.exists():
                break
        rows, fails = [], []
        self.hooks.measure(
            [self.problems[r.problem_key] for r in reqs],
            [[index[r.candidate_key]] for r in reqs],
            built, self.devices, self.margs, self.worker_path, self.base_env,
            csv_path, extra={"check_identity": True},
            record_failures=fails.extend, measured=rows,
        )
        RunWriter(rounds).write_json(csv_path.with_suffix(".json").name, {
            "requests": [
                {"problem_idx": i, "call_id": r.call_id, "trial_uid": r.trial_uid,
                 "problem": r.problem_key, "kernel": r.kernel_name}
                for i, r in enumerate(reqs)
            ],
            "failures": fails,
        })
        by_rows, by_fails = defaultdict(list), defaultdict(list)
        for row in rows:
            by_rows[row.get("problem_idx")].append(row)
        for f in fails:
            by_fails[f.get("problem_idx")].append(f)
        return by_rows, by_fails


def run_gemm_search(args, settings: SearchSettings, hooks: DriverHooks, *,
                    problems, configs, eligible, devices, worker,
                    context=None) -> int:
    """The driver's search mode. Returns the exit code: 0 when every problem
    has a confirmed winner, 1 otherwise, 2 for invalid options."""
    out_dir = Path(args.search_output_dir)
    print(f"\n{'=' * 80}")
    print(f"Adaptive search: {settings.strategy}, {settings.trials_per_problem} "
          f"trials per problem, at most {settings.new_builds_per_run} builds")
    print(f"{'=' * 80}")
    if not devices:
        print("  ERROR: no GPU to measure on (--devices)")
        return EXIT_USAGE
    adapter = GemmSearchAdapter(args, hooks, problems, configs, eligible, devices,
                                worker, out_dir, context)
    try:
        coordinator = SearchCoordinator(adapter, settings, out_dir,
                                        pair_loader=hooks.load_pairs,
                                        prior_loader=read_ranked_csv)
    except SettingsError as e:
        print(f"  ERROR: {e}")
        return EXIT_USAGE
    s = coordinator.run()
    n = len(s.winners) + len(s.unresolved)
    print(f"\n{'=' * 80}")
    print(f"SEARCH {s.status.upper()}: {len(s.winners)}/{n} problems with a "
          f"confirmed winner (stopped: {s.stop_reason})")
    print(f"{'=' * 80}")
    c = s.counts
    print(f"  Builds: {c['built']} ok, {c['build_failed']} failed; measurements: "
          f"{c['measured']}; confirmation calls: {c['confirm_calls']}; "
          f"{s.wall_s:.0f}s")
    for key, why in sorted(s.unresolved.items()):
        print(f"  Unresolved {key}: {why}")
    if s.error:
        print(f"  ERROR: {s.error}")
    print(f"  Output: {out_dir} (winners.json is a --pairs file)")
    return s.exit_code
