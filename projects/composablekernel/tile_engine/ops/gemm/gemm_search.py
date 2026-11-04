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
- builds: ``setup_multiple_gemm_dispatchers`` on the round's missing kernels,
  verbose like a full run's; each built .so goes into the run's
  ``ArtifactStore`` as ``<out>/artifacts/<name>.so``, and each call's full
  build errors go to ``<out>/rounds/NNNN-build.json``;
- measurements: ``benchmark_on_devices`` with --verify on, one kernel per worker
  and one problem entry per request, so every row and failure record belongs to
  exactly one request; the worker also checks that the library reports the
  requested kernel (``check_identity``). Each call writes its own CSV under
  ``<out>/rounds/``, next to a JSON file mapping its ``problem_idx`` to the
  request (call id, trial) and holding its failure records;
- confirmation: the same path with fresh calls, each problem's on one device
  (problem i on device i mod n), so its finalists run on the same GPU.

``reconcile`` turns a request's rows and failure records into its one outcome;
``read_ranked_csv`` turns ``select_kernels.py --out_ranked`` predictions into
the latency prior of ``warm_start.ranked_csv``. The op-agnostic plumbing
(artifact copies, per-call files, provenance, option checks, the entry point)
is in ``adaptive_search``.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

_COMMON_DIR = str(Path(__file__).resolve().parents[1] / "common")
if _COMMON_DIR not in sys.path:
    sys.path.insert(0, _COMMON_DIR)

import adaptive_search  # noqa: E402
from adaptive_search import (  # noqa: E402
    ArtifactStore,
    BuildResult,
    CallFiles,
    Candidate,
    PairOutcome,
    PairRequest,
    SearchSettings,
    SettingsError,  # the driver imports it from here
    measure_built,
    source_info,
    valid_latency,
)

# Driver options a search run does not take: it writes its own outputs and
# confirms its finalists itself.
CONFLICTS = (
    ("--list-pairs", "list_pairs"),
    ("--repeat-top", "repeat_top"),
    ("--failures-csv", "failures_csv"),
)
REQUIRED = (("--variant", "variant", "gemm_universal"),)  # the one searched
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
# A verified one-kernel worker also pays the host side of verification (random
# inputs, the fp32 numpy reference GEMM, the comparison), which a batched
# --verify run pays once per batch. Each call's timeout is --kernel-timeout
# plus that time, budgeted at 50 GFLOP/s for the reference: a rate the host
# keeps with several device workers sharing its cores.
VERIFY_HOST_FLOPS = 5e10
BUILD_ERROR_CHARS = 8000  # per failed kernel in rounds/NNNN-build.json


def verify_allowance_s(problem) -> int:
    """Seconds added to ``--kernel-timeout`` for a verified call on ``problem``."""
    return math.ceil(2 * problem["M"] * problem["N"] * problem["K"] / VERIFY_HOST_FLOPS)


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
    """The validated ``--search-config`` of a GEMM driver run (SettingsError
    otherwise, see ``adaptive_search.check_search_options``)."""
    return adaptive_search.check_search_options(args, CONFLICTS, REQUIRED)


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
        self.store = ArtifactStore(out_dir)
        self.calls = CallFiles(out_dir)

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
                            "timeout_s": "kernel_timeout + ceil(2*M*N*K / "
                                         f"{VERIFY_HOST_FLOPS:g}) per call",
                            "check_identity": True},
            "build": {"workers": a.workers, "build_timeout": a.build_timeout},
            **source_info(files, domain),
            "domain_notes": notes,
        }

    def enumerate_eligible(self) -> Dict[str, List[Candidate]]:
        return {k: list(v) for k, v in self.eligible.items()}

    def build_missing(self, cands: Sequence[Candidate]) -> Dict[str, BuildResult]:
        errors: Dict[str, str] = {}
        paths = self.hooks.build(
            [c.original_config for c in cands], verbose=True,
            max_workers=self.args.workers, errors=errors,
            build_timeout=self.args.build_timeout,
        )
        if len(paths) != len(cands):
            raise RuntimeError(f"build returned {len(paths)} paths for {len(cands)} kernels")
        log = self.calls.new("build", ".json")
        self.calls.write_json(log, {
            "kernels": [c.kernel_name for c in cands],
            "errors": {c.kernel_name: str(errors.get(c.kernel_name, ""))[:BUILD_ERROR_CHARS]
                       for c, lib in zip(cands, paths) if lib is None},
        })
        out = {}
        for c, lib in zip(cands, paths):
            if lib is None:
                why = self.hooks.build_reason(errors.get(c.kernel_name))
                out[c.canonical_key] = BuildResult(None, why or "build failed")
            else:
                out[c.canonical_key] = self.store.add(c.canonical_key, lib,
                                                      f"{c.kernel_name}.so")
        return out

    def measure_pairs(self, requests: Sequence[PairRequest]) -> List[PairOutcome]:
        return measure_built(self.store, requests, lambda rs: self._run(rs, "measure"))

    def confirm(self, requests: Sequence[PairRequest]) -> List[PairOutcome]:
        return measure_built(self.store, requests, lambda rs: self._run(rs, "confirm"))

    # ----------------------------------------------------------- measure

    def _run(self, reqs, kind) -> List[PairOutcome]:
        """One measure call over ``reqs`` (all with a usable artifact)."""
        built, index = [], {}
        for r in reqs:
            if r.candidate_key not in index:
                index[r.candidate_key] = len(built)
                cfg = self.cands[r.candidate_key].original_config
                built.append((cfg, self.store.path(r.candidate_key)))
        csv_path = self.calls.new(kind, ".csv")
        keys = list(dict.fromkeys(r.problem_key for r in reqs))
        title = (f"[search] {csv_path.stem}: {len(reqs)} request(s) on "
                 + ", ".join(keys[:6]) + (", ..." if len(keys) > 6 else ""))
        rows, fails, pin = [], [], {}
        if kind == "confirm":  # a problem's finalists are compared on one GPU
            order = {key: i for i, key in enumerate(self.problems)}
            pin["device_of"] = [self.devices[order[r.problem_key] % len(self.devices)]
                                for r in reqs]
        problems = [self.problems[r.problem_key] for r in reqs]
        margs = argparse.Namespace(**{**vars(self.margs), "kernel_timeout": (
            self.margs.kernel_timeout + max(map(verify_allowance_s, problems)))})
        self.hooks.measure(
            problems, [[index[r.candidate_key]] for r in reqs],
            built, self.devices, margs, self.worker_path, self.base_env,
            csv_path, extra={"check_identity": True},
            record_failures=fails.extend, measured=rows, title=title, **pin,
        )
        self.calls.write_json(csv_path, {
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
        out = []
        for i, r in enumerate(reqs):
            ran = by_rows[i] + by_fails[i]
            out.append(PairOutcome(
                r, *reconcile(r, self.problems[r.problem_key], by_rows[i], by_fails[i]),
                device=str(ran[0].get("device", "")) if ran else ""))
        return out


def run_gemm_search(args, settings: SearchSettings, hooks: DriverHooks, *,
                    problems, configs, eligible, devices, worker,
                    context=None) -> int:
    """The driver's search mode (``adaptive_search.run_search``): its exit code."""
    adapter = GemmSearchAdapter(args, hooks, problems, configs, eligible, devices,
                                worker, args.search_output_dir, context)
    return adaptive_search.run_search(adapter, settings, args.search_output_dir,
                                      devices, pair_loader=hooks.load_pairs,
                                      prior_loader=read_ranked_csv)
