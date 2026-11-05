#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Run AOTriton's SDPA suite against hipDNN and record one CSV row per case.

    run_aotriton.py [--suite default|forward] [--release N] [--shard I/N] [--fake always-decline]
                    [--layout-mode MODE] [--fwd-no-stats] [--packed-gpu] [--kernel-ids]
                    [--all-engines [--pairs]] [--exhaustive]
                    [--out DIR | --arch ARCH --run NAME]
                    [--workers N] [--resume] [--nodeids-file PATH] [--hang-timeout S] [--lock-dir DIR]
                    [--hipdnn-prefix DIR] [-- PYTEST...]

Per pass the runner (1) collects with the same options and writes the node ids to
``<out>/collected.txt``, then (2) runs under ``-n`` with ``--max-worker-restart`` equal to the
collected count, so a crashing case costs one row and not the run. ``<out>/results.csv`` is
flushed per row; ``reconcile.py`` adds ``not-run`` rows for whatever never finished.

``--suite default`` is ``test_backward.py`` plus ``test_varlen.py``; ``--suite forward`` is
``test_forward.py``, named explicitly because the vendored conftest excludes it from directory
collection. ``--packed-gpu`` adds a second pass over the ``gpu``-marked cases of
``sdpa_conformance/tests``. The default ``--out`` is
``./sdpa_conformance_results/<arch>/<run>/shard-<I>of<N>/`` under the working directory.
``--nodeids-file`` limits the first pass to the listed node ids (the pinned passes keep their own
listings).

``--layout-mode`` picks the memory layout every SDPA tensor is allocated in (default ``pytorch``):
``pytorch`` stores O, dO and the gradients like Q (and dQ/dK/dV/dB like their inputs), ``efficient``
stores O, dO, dQ, dK and dV BSHD (dB like the bias), ``aotriton`` keeps the vendored suite's own
layouts. It is exported as ``HIPDNN_SDPA_CONFORMANCE_LAYOUT_MODE`` to every pytest process,
``--all-engines`` pass 2 included.

``--fwd-no-stats`` runs the forward without Stats (``HIPDNN_SDPA_CONFORMANCE_FWD_NO_STATS=1`` in
the pytest workers), for engines that decline when Stats are requested. The backward tests
consume the forward's Stats, so the runner refuses the flag unless every collected case is in
``test_forward.py``.

``--all-engines`` adds more passes after the normal run (which still decides ``<out>/results.csv``).
Pass 1 records, per case, the applicable engines in heuristic rank order (``fwd_engines`` /
``bwd_engines``; for a case whose worker died, the lists its in-flight marker held). Under
``<out>/all-engines/`` every pass has its own directory with ``results.csv``, ``nodeids.txt`` (the
cases it must run) and ``pass.json`` (kind, direction, pins):

* ``fwd/<engine>/`` and ``bwd/<engine>/``: one pytest run per applicable engine that did not
  serve a case, pinning ``HIPDNN_FWD_ENGINE`` (``test_forward.py`` cases) or ``HIPDNN_BWD_ENGINE``
  (every other case, forward left to the heuristic);
* ``pair/<fwd>__<bwd>/`` (only with ``--pairs``): backward cases with both pins, every
  stats-capable forward that did not serve the case against every applicable backward;
* ``refwd/discovery/`` and ``refwd/<bwd>/``: for backward cases whose forward failed or crashed in
  any pass, the backward unpinned and then pinned on the reference forward
  (``HIPDNN_FWD_ENGINE=reference``); with ``--pairs`` the pair passes then cover those cases with
  the backward engines the discovery pass recorded.

``--pairs`` needs ``--all-engines`` (refused otherwise) and is off by default: each pass pins one
direction, so a failure names the engine of that direction. ``reconcile.py --engines <out>`` folds
the passes into ``engines.csv``. ``--all-engines`` and ``--exhaustive`` compose with ``--suite``,
``--shard``, ``--release`` and ``--workers``; ``--fwd-no-stats`` composes with them for the forward
sweep only. Both refuse ``--fake`` and pre-set ``HIPDNN_FWD_ENGINE`` / ``HIPDNN_BWD_ENGINE``.

``--kernel-ids`` fills the ``fwd_kernel`` / ``bwd_kernel`` columns by exporting
``HIPDNN_SDPA_KERNEL_LOG=1`` to every pass, which makes the plugin install ``hipdnn_sdpa``'s backend
log callback. That callback raises the backend log level to INFO and forces ingestor applicability
inline, so it changes how the product runs; it is off by default and the kernel columns stay empty.
``--exhaustive`` implies it, because candidate honoring compares kernel ids.

``--exhaustive`` pins kernels inside an engine (``--all-engines`` pins engines; the flags compose, and
both together are the EXTREME run). It needs an executable ``<prefix>/bin/hipdnn_bench`` and refuses
to start without one. Pass 1 records, per case and engine, the candidate count
``hipdnn_bench enumerate`` lists (``fwd_candidate`` / ``bwd_candidate``: a JSON object ``{engine:
count | reason the engine cannot enumerate}``). Under ``<out>/cand/``:

* ``<fwd|bwd>/<engine>/<k>/``: one pass per candidate index up to the largest count any case
  reported for the engine, pinning the engine and ``HIPDNN_FWD_CANDIDATE`` / ``HIPDNN_BWD_CANDIDATE``
  ``=k`` over the cases whose count exceeds ``k``;
* ``<fwd|bwd>/<engine>/not-enumerable/``: no pytest run; a ``not-enumerable`` row carrying the
  message for each case where the engine could not enumerate;
* with ``--all-engines --pairs`` also ``pair/<fwd>__<bwd>/<i>x<j>/`` (and ``.../not-enumerable/``) for
  backward cases: every forward engine against every backward engine of the case, the winners
  included, over every pair of candidate indices. Without ``--pairs`` each candidate pass pins one
  direction, so EXTREME costs the sum of the candidate counts, not their product.

A case whose worker died in pass 1 has no counts and gets no candidate pass. The runner exports
``HIPDNN_SDPA_PREFIX`` (the ``--hipdnn-prefix``, where ``bin/hipdnn_bench`` lives) to every pass and
``HIPDNN_SDPA_EXHAUSTIVE`` to pass 1 only; it refuses ``--fake`` and pre-set candidate variables.

Every pass exports ``HIPDNN_SDPA_PASS_SLUG`` and a per-pass in-flight directory
(``HIPDNN_SDPA_INFLIGHT_DIR``, node-local: default a fresh temporary directory) where
``hipdnn_sdpa`` leaves a marker per worker; the plugin turns a worker death into a ``crash`` row
naming the engine, direction and stage from it. After each pass the runner re-runs every new
``crash`` case once with ``HIPDNN_SDPA_REPRO_CAPTURE=<out>/repro`` (within the repro caps) into
``<pass dir>/capture/``, which no reader of the pass rows looks at; the crash repeats and leaves a
full repro directory. ``--hang-timeout S`` (default 1800; 0 disables) starts
``python -m pytest_gpu_lease.watchdog`` beside every pytest run on a node-local lock file under
``--lock-dir`` (default ``$HIPDNN_SDPA_LOCK_DIR`` or a temporary directory) with its stderr in
``<pass dir>/watchdog.log``; a worker it kills becomes a ``crash`` row with reason ``hang``.

``--resume`` accepts a non-empty ``--out``: per pass directory it runs only the node ids of the
pass (pass 1: the shard's collection) that have no row, from a node-id file and without
``--shard``. A ``crash`` row is a row, so a crashed case is never run again.
A fresh start records its options in ``<out>/run-options.json``: ``schema`` (1), ``hipdnn_prefix``,
``all_engines``, ``pairs``, ``exhaustive``, ``kernel_ids``, ``layout_mode``, ``fwd_no_stats``,
``suite``, ``release``, ``shard`` and ``packed_gpu``; ``completion.py`` reads it to know which
passes the shard calls for. A resumed link must repeat them: ``--resume`` exits 2 and prints each
differing key as recorded vs requested when any key but ``schema`` and ``hipdnn_prefix`` differs
(the file keeps the first link's values). A shard without the file (written before the runner
recorded its options; read so such shards still resume) is compared on ``all_engines``, ``pairs``
and ``exhaustive`` against ``--all-engines`` without the others, and the full file is then written.

Every pytest process checks that the ``libhipdnn_backend`` mapped into it is the one under
``--hipdnn-prefix`` (default ``$ROCM_PATH``); the runner refuses to start without a prefix. A ROCm
torch next to a hipDNN install of another prefix maps the wrong library: see README.md,
"Mixed ROCm torch and hipDNN prefixes".

Exit status: 0 when every pytest run exited 0, else the status of the first pytest run that did
not (run order; pytest's 1 means failed cases); 2 when collection fails, ``--fwd-no-stats`` meets a
case outside ``test_forward.py``, ``--resume`` is refused, or the arguments are invalid (``--pairs``
without ``--all-engines``); 1 for the other refusals (no prefix, no ``hipdnn_bench``, existing rows
without ``--resume``, ``--fake`` or a pre-set pin).
"""

import argparse
import contextlib
import copy
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple

import aotriton_layout as layout
import conformance_contract as contract
import reconcile

DEFAULT_RESULTS = Path.cwd() / "sdpa_conformance_results"
WATCHDOG_MODULE = "pytest_gpu_lease.watchdog"
LOCK_NAME = "gpu.lock"
CAPTURE_DIR = "capture"
PIN_ENV = (*contract.ENGINE_ENV.values(), *contract.CANDIDATE_ENV.values())
RUN_OPTIONS_NAME = "run-options.json"
RUN_OPTIONS_SCHEMA = 1
# Keys of run-options.json that ``--resume`` does not compare: the schema, and the prefix, which
# may be mounted elsewhere on another node.
UNCHECKED_OPTIONS = ("schema", "hipdnn_prefix")
# A shard written before the runner recorded its options (no run-options.json) ran --all-engines
# without --pairs or --exhaustive; read so that such a shard still resumes and reaches completion.py.
LEGACY_RUN_OPTIONS = {"all_engines": True, "pairs": False, "exhaustive": False}


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--suite", choices=("default", "forward"), default="default")
    parser.add_argument(
        "--release", type=int, choices=(0, 1, 2, 3), help="exports FOR_RELEASE"
    )
    parser.add_argument("--shard", metavar="I/N")
    parser.add_argument("--fake", choices=("always-decline",))
    parser.add_argument("--packed-gpu", action="store_true")
    parser.add_argument(
        "--layout-mode",
        choices=layout.LAYOUT_MODES,
        default=layout.DEFAULT_LAYOUT_MODE,
        help="memory layout of every SDPA tensor: pytorch (Q's layout), efficient (BSHD) "
        "or aotriton (the vendored suite's own)",
    )
    parser.add_argument(
        "--fwd-no-stats",
        action="store_true",
        help=f"run the forward without Stats; only valid for {contract.FORWARD_TESTS}",
    )
    parser.add_argument(
        "--all-engines",
        action="store_true",
        help="after the normal run, re-run every case with each other applicable engine pinned",
    )
    parser.add_argument(
        "--pairs",
        action="store_true",
        help="with --all-engines, also run the forward x backward pair passes (off: one direction "
        "per pass); refused without --all-engines",
    )
    parser.add_argument(
        "--kernel-ids",
        action="store_true",
        help="record the kernel id of each case (installs a backend INFO-log callback, which "
        "changes how the product runs); --exhaustive implies it",
    )
    parser.add_argument(
        "--exhaustive",
        action="store_true",
        help="after the normal run, run one pass per candidate (kernel) of every enumerable engine; "
        "with --all-engines (EXTREME) and --pairs, per pair of candidates too",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue into a non-empty --out: run only the node ids of each pass that have no row",
    )
    parser.add_argument(
        "--nodeids-file",
        type=Path,
        help="limit the first pass to the node ids listed in this file (one per line)",
    )
    parser.add_argument(
        "--hang-timeout",
        type=float,
        default=1800.0,
        help="seconds a case may run before the lease watchdog kills its worker; 0 disables",
    )
    parser.add_argument(
        "--lock-dir",
        type=Path,
        default=os.environ.get("HIPDNN_SDPA_LOCK_DIR"),
        help="node-local directory of the lease lock file (default: a temporary directory)",
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--arch", help="gfx arch for the default --out (default: detected)"
    )
    parser.add_argument("--run", help="run name for the default --out")
    parser.add_argument(
        "--workers", type=int, help="xdist workers (default: one per GPU)"
    )
    parser.add_argument(
        "--hipdnn-prefix",
        default=os.environ.get("ROCM_PATH"),
        help="hipDNN install prefix whose libhipdnn_backend must be loaded (default: $ROCM_PATH)",
    )
    parser.add_argument(
        "pytest_args", nargs="*", help="passed to every pytest invocation"
    )
    args = parser.parse_args(argv)
    if args.pairs and not args.all_engines:
        parser.error("--pairs needs --all-engines")
    return args


def detect_gpus():
    import torch

    return torch.cuda.device_count(), (
        torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
        if torch.cuda.is_available()
        else None
    )


def resolve_out(args, detected_arch):
    if args.out:
        return args.out
    arch = args.arch or detected_arch
    if not (arch and args.run):
        raise SystemExit("pass --out, or --run (and --arch where no GPU is visible)")
    shard = args.shard.replace("/", "of") if args.shard else "1of1"
    return DEFAULT_RESULTS / arch / args.run / f"shard-{shard}"


def passes(args):
    target = (
        layout.FLASH_TESTS
        if args.suite == "default"
        else layout.FLASH_TESTS / contract.FORWARD_TESTS
    )
    result = [("suite", [str(target)])]
    if args.packed_gpu:
        result.append(("packed-gpu", ["-m", "gpu", str(layout.PACKED_TESTS)]))
    return result


def engine_slug(name):
    """A directory name for an engine name."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def _engines(text):
    return [e for e in text.split(";") if e]


def engine_work(rows):
    """Pass-2 work from pass-1 rows: ``{(direction, engine): [nodeid, ...]}``.

    A ``test_forward.py`` case is a forward case and reads ``fwd_engines`` / ``fwd``; any other
    case is a backward case and reads ``bwd_engines`` / ``bwd``. Per case, every listed engine
    except the one(s) that served it (a ``crash`` row names the engine in flight in the same
    state column). A ``crash`` in the other direction schedules nothing: the case's forward
    would die again before its backward runs.
    """
    work = {}
    for row in rows:
        direction = reconcile.direction_of(row["nodeid"])
        if row["outcome"] == "crash" and row["fail_side"] != direction:
            continue
        winners = reconcile.served_engines(row[direction])
        for engine in _engines(row[f"{direction}_engines"]):
            if engine not in winners:
                work.setdefault((direction, engine), []).append(row["nodeid"])
    return work


def pair_work(rows, bwd_lists=None):
    """``{(fwd engine, bwd engine): [nodeid, ...]}`` for backward cases: every forward in the
    case's forward list except the one that served (or crashed in) it, against every backward in
    the case's row, or in ``bwd_lists[nodeid]`` (the reference-forward discovery) where the row has none.
    """
    work = {}
    for row in rows:
        if reconcile.direction_of(row["nodeid"]) != "bwd":
            continue
        served = reconcile.served_engines(row["fwd"])
        backwards = _engines(
            row["bwd_engines"] or (bwd_lists or {}).get(row["nodeid"], "")
        )
        for fwd in _engines(row["fwd_engines"]):
            if fwd not in served:
                for bwd in backwards:
                    work.setdefault((fwd, bwd), []).append(row["nodeid"])
    return work


def _counts(text):
    """The ``{engine: count | message}`` JSON of a pass-1 candidate column; ``{}`` when empty."""
    try:
        found = json.loads(text) if text else {}
    except ValueError:
        return {}
    return found if isinstance(found, dict) else {}


def candidate_work(rows):
    """``{(direction, engine): ({nodeid: count}, {nodeid: message})}`` from pass-1 rows: a
    ``test_forward.py`` case reads ``fwd_candidate``, any other case ``bwd_candidate``.
    """
    work = {}
    for row in rows:
        direction = reconcile.direction_of(row["nodeid"])
        for engine, found in _counts(row[f"{direction}_candidate"]).items():
            counts, messages = work.setdefault((direction, engine), ({}, {}))
            if isinstance(found, int):
                counts[row["nodeid"]] = found
            else:
                messages[row["nodeid"]] = str(found)
    return work


def candidate_pair_work(rows):
    """``{(fwd engine, bwd engine): ({nodeid: (fwd count, bwd count)}, {nodeid: message})}`` for
    backward cases: every forward and backward engine pass 1 counted, winners included.
    """
    work = {}
    for row in rows:
        if reconcile.direction_of(row["nodeid"]) != "bwd":
            continue
        forwards, backwards = _counts(row["fwd_candidate"]), _counts(
            row["bwd_candidate"]
        )
        for fwd, fwd_found in forwards.items():
            for bwd, bwd_found in backwards.items():
                counts, messages = work.setdefault((fwd, bwd), ({}, {}))
                if isinstance(fwd_found, int) and isinstance(bwd_found, int):
                    counts[row["nodeid"]] = (fwd_found, bwd_found)
                else:
                    why = [
                        f"{side} {engine}: {found}"
                        for side, engine, found in (
                            ("fwd", fwd, fwd_found),
                            ("bwd", bwd, bwd_found),
                        )
                        if not isinstance(found, int)
                    ]
                    messages[row["nodeid"]] = "; ".join(why)
    return work


def pair_rel(fwd, bwd):
    """Directory of the (forward, backward) pair passes, under ``all-engines/`` or ``cand/``."""
    return Path("pair") / f"{engine_slug(fwd)}__{engine_slug(bwd)}"


class PlannedPass(NamedTuple):
    root: str  # under <out>: reconcile.ENGINES_DIR or contract.CANDIDATES_DIR
    rel: Path  # under root
    kind: str  # contract.PASS_*
    direction: str
    pins: dict
    cases: object  # node ids; {nodeid: message} for a not-enumerable pass
    note: str
    not_enumerable: bool = False  # no pytest run: one ``not-enumerable`` row per case


def candidate_passes(rows, pairs):
    """Every pass ``--exhaustive`` runs under ``cand/`` from pass-1 rows, in run order; with
    ``pairs`` (``--all-engines --pairs``) the EXTREME pair passes follow. A ``not-enumerable`` pass runs no
    pytest and carries ``{nodeid: message}`` as its cases.
    """
    root = contract.CANDIDATES_DIR
    for (direction, engine), (counts, messages) in sorted(candidate_work(rows).items()):
        rel = Path(direction) / engine_slug(engine)
        pin = {contract.ENGINE_ENV[direction]: engine}
        note = f"candidates {direction} {engine}"
        for k in range(max(counts.values(), default=0)):
            yield PlannedPass(
                root,
                rel / str(k),
                contract.PASS_CANDIDATE,
                direction,
                {**pin, contract.CANDIDATE_ENV[direction]: str(k)},
                [n for n, count in counts.items() if count > k],
                f"{note} #{k}",
            )
        if messages:
            yield PlannedPass(
                root,
                rel / contract.NOT_ENUMERABLE,
                contract.PASS_CANDIDATE,
                direction,
                pin,
                messages,
                note,
                not_enumerable=True,
            )
    if not pairs:
        return
    for (fwd, bwd), (counts, messages) in sorted(candidate_pair_work(rows).items()):
        rel = pair_rel(fwd, bwd)
        pin = {
            contract.ENGINE_ENV["fwd"]: fwd,
            contract.ENGINE_ENV["bwd"]: bwd,
        }
        note = f"candidates pair {fwd} x {bwd}"
        for i in range(max((c[0] for c in counts.values()), default=0)):
            for j in range(max((c[1] for c in counts.values()), default=0)):
                chosen = [n for n, c in counts.items() if c[0] > i and c[1] > j]
                if chosen:
                    yield PlannedPass(
                        root,
                        rel / f"{i}x{j}",
                        contract.PASS_CANDIDATE_PAIR,
                        "bwd",
                        {
                            **pin,
                            contract.CANDIDATE_ENV["fwd"]: str(i),
                            contract.CANDIDATE_ENV["bwd"]: str(j),
                        },
                        chosen,
                        f"{note} #{i}x{j}",
                    )
        if messages:
            yield PlannedPass(
                root,
                rel / contract.NOT_ENUMERABLE,
                contract.PASS_CANDIDATE,
                "bwd",
                pin,
                messages,
                note,
                not_enumerable=True,
            )


def _sweep_passes(work):
    for (direction, engine), nodeids in sorted(work.items()):
        yield PlannedPass(
            reconcile.ENGINES_DIR,
            Path(direction) / engine_slug(engine),
            contract.PASS_SWEEP,
            direction,
            {contract.ENGINE_ENV[direction]: engine},
            nodeids,
            f"{direction} {engine}",
        )


def _pair_passes(work):
    for (fwd, bwd), nodeids in sorted(work.items()):
        yield PlannedPass(
            reconcile.ENGINES_DIR,
            pair_rel(fwd, bwd),
            contract.PASS_PAIR,
            "bwd",
            {contract.ENGINE_ENV["fwd"]: fwd, contract.ENGINE_ENV["bwd"]: bwd},
            nodeids,
            f"pair {fwd} x {bwd}",
        )


def planned_passes(out, rows, options):
    """Every pass beyond pass 1 that ``options`` (``run_options``) call for, in run order, from the
    pass-1 ``rows``: with ``all_engines`` the pinned sweeps, the pair passes (``pairs``) and the
    reference-forward passes of ``forward_failed``; with ``exhaustive`` the ``candidate_passes``.

    A generator that reads the results of earlier passes only when it reaches the passes that
    depend on them (``forward_failed`` after the sweeps and pairs, the discovery rows after the
    discovery pass). The runner runs each pass before asking for the next; ``completion.py``
    iterates it over what is on disk.
    """
    if options["all_engines"]:
        yield from _sweep_passes(engine_work(rows))
        if options["pairs"]:
            yield from _pair_passes(pair_work(rows))
        failed = forward_failed(out)
        if failed:
            engines = Path(out) / reconcile.ENGINES_DIR / "refwd"
            reference = {contract.ENGINE_ENV["fwd"]: contract.REFERENCE_PIN}
            yield PlannedPass(
                reconcile.ENGINES_DIR,
                Path("refwd") / "discovery",
                contract.PASS_DISCOVERY,
                "bwd",
                reference,
                failed,
                "refwd discovery",
            )
            discovery = reconcile.read_rows(engines / "discovery" / reconcile.CSV_NAME)
            for (_, engine), nodeids in sorted(engine_work(discovery).items()):
                yield PlannedPass(
                    reconcile.ENGINES_DIR,
                    Path("refwd") / engine_slug(engine),
                    contract.PASS_REFWD,
                    "bwd",
                    {**reference, contract.ENGINE_ENV["bwd"]: engine},
                    nodeids,
                    f"refwd bwd {engine}",
                )
            if options["pairs"]:
                lists = {r["nodeid"]: r["bwd_engines"] for r in discovery}
                yield from _pair_passes(pair_work(rows, lists))
    if options["exhaustive"]:
        yield from candidate_passes(rows, options["pairs"])


def run_options(out):
    """The pass-planning flags of the shard's links as ``main`` records them in
    ``<out>/run-options.json`` (``all_engines``, ``pairs``, ``exhaustive`` and the other keys of
    ``requested_options``); a shard without the file reads as ``LEGACY_RUN_OPTIONS``.
    """
    path = Path(out) / RUN_OPTIONS_NAME
    return json.loads(path.read_text()) if path.exists() else dict(LEGACY_RUN_OPTIONS)


def requested_options(args):
    """What ``main`` writes to ``run-options.json`` for ``args``."""
    return {
        "schema": RUN_OPTIONS_SCHEMA,
        "hipdnn_prefix": str(args.hipdnn_prefix),
        "all_engines": args.all_engines,
        "pairs": args.pairs,
        "exhaustive": args.exhaustive,
        "kernel_ids": args.kernel_ids,
        "layout_mode": args.layout_mode,
        "fwd_no_stats": args.fwd_no_stats,
        "suite": args.suite,
        "release": args.release,
        "shard": args.shard,
        "packed_gpu": args.packed_gpu,
    }


def option_differences(recorded, requested):
    """``"<key>: recorded <v>, requested <v>"`` per compared key of ``recorded`` whose value
    differs in ``requested``; keys the record lacks (a legacy shard, an older file) are not compared.
    """
    return [
        f"{key}: recorded {recorded[key]!r}, requested {requested[key]!r}"
        for key in requested
        if key in recorded
        and key not in UNCHECKED_OPTIONS
        and recorded[key] != requested[key]
    ]


def forward_failed(out):
    """Node ids of backward cases whose forward failed or crashed in pass 1, a pinned or a pair pass."""
    out = Path(out)
    dirs = [out] + [
        d
        for d, meta in reconcile.pass_dirs(out)
        if meta["kind"] in (contract.PASS_SWEEP, contract.PASS_PAIR)
    ]
    failed = {
        row["nodeid"]
        for d in dirs
        for row in reconcile.read_rows(d / reconcile.CSV_NAME)
        if row["outcome"] in ("fail", "crash")
        and row["fail_side"] == "fwd"
        and reconcile.direction_of(row["nodeid"]) == "bwd"
    }
    return sorted(failed)


def start_watchdog(cmd, env, log):
    """Start the lease watchdog with its stderr in ``log`` (an open file)."""
    return subprocess.Popen(cmd, env=env, stderr=log)


@contextlib.contextmanager
def watchdog(args, env, pass_dir, workers):
    """The lease watchdog beside one pytest run; its stderr is ``<pass_dir>/watchdog.log``."""
    if not args.hang_timeout:
        yield
        return
    cmd = [
        sys.executable,
        "-m",
        WATCHDOG_MODULE,
        "--lockfile",
        env["GPU_LEASE_LOCKFILE"],
        "--workers",
        str(workers),
        "--threshold",
        str(args.hang_timeout),
    ]
    pass_dir.mkdir(parents=True, exist_ok=True)
    with open(pass_dir / "watchdog.log", "a") as log:
        dog = start_watchdog(cmd, env, log)
        try:
            yield
        finally:
            dog.terminate()
            dog.wait(timeout=60)


def pytest_cmd(args, selection, out=None, nodeids=None, repro=None):
    """The pytest command; ``out`` (the results directory) is omitted for the collection pass."""
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        layout.PLUGIN,
        "-p",
        layout.GPU_LEASE_PLUGIN,
        "-p",
        "no:cacheprovider",
        f"--rootdir={layout.SDPA_ROOT}",
    ]
    if out:
        cmd.append(f"--results-dir={out}")
    if repro:
        cmd.append(f"--repro-dir={repro}")
    if args.shard:
        cmd.append(f"--shard={args.shard}")
    if args.fake:
        cmd.append(f"--fake={args.fake}")
    if nodeids:
        cmd.append(f"--hipdnn-nodeids-file={nodeids}")
    cmd.append(f"--hipdnn-prefix={args.hipdnn_prefix}")
    return cmd + args.pytest_args + selection


def recorded_ids(results_dir):
    rows = reconcile.read_rows(Path(results_dir) / reconcile.CSV_NAME)
    return {r["nodeid"] for r in rows}


def listing_file(path, nodeids):
    path = path.resolve()
    path.write_text("".join(f"{n}\n" for n in nodeids))
    return path


class Runner:
    """Runs the passes of one shard into ``out``; ``status`` is the first non-zero pytest exit."""

    def __init__(self, args, env, out, workers):
        self.args, self.env, self.out, self.workers = args, env, out, workers
        self.pass_ids = {}  # selection label -> node ids that selection collected
        self.status = 0

    def lock_file(self):
        """Node-local lease lock file: the caller's ``GPU_LEASE_LOCKFILE``, else under ``--lock-dir``."""
        if "GPU_LEASE_LOCKFILE" in self.env:
            return self.env["GPU_LEASE_LOCKFILE"]
        if not self.args.lock_dir:
            self.args.lock_dir = Path(tempfile.mkdtemp(prefix="sdpa-lease-"))
        self.args.lock_dir.mkdir(parents=True, exist_ok=True)
        return str(self.args.lock_dir / LOCK_NAME)

    def pytest(
        self,
        selection,
        pass_dir,
        slug,
        pins,
        nodeids,
        count,
        capture=False,
        shard=False,
    ):
        """One pytest run of ``selection`` into ``pass_dir`` under the lease watchdog; ``pins``
        (``None``: the caller's environment) are the engine pins. Hangs are folded into the rows
        afterwards."""
        args = copy.copy(self.args)
        if not shard:
            args.shard = None  # the listed node ids are already this shard's cases
        env = dict(self.env)
        if pins is not None:
            env = {k: v for k, v in env.items() if k not in PIN_ENV}
            env.update(pins)
        if slug != reconcile.MAIN_PASS:
            env.pop(contract.EXHAUSTIVE_ENV, None)  # only pass 1 counts candidates
        env[contract.PASS_SLUG_ENV] = slug
        if capture:
            env[contract.REPRO_CAPTURE_ENV] = str(self.out / contract.REPRO_DIR)
        else:
            env.pop(contract.REPRO_CAPTURE_ENV, None)
        inflight = Path(env[contract.INFLIGHT_DIR_ENV]) / slug
        shutil.rmtree(
            inflight, ignore_errors=True
        )  # no marker of an earlier run names a crash
        inflight.mkdir(parents=True)
        env["GPU_LEASE_LOCKFILE"] = self.lock_file()
        workers = self.workers
        cmd = pytest_cmd(
            args, selection, pass_dir, nodeids, self.out / contract.REPRO_DIR
        )
        cmd += ["-n", str(workers), f"--max-worker-restart={count}"]
        with watchdog(args, env, pass_dir, workers):
            run = subprocess.run(cmd, env=env)
        reconcile.mark_hangs(pass_dir, self.out, slug)
        self.status = self.status or run.returncode

    def run_pass(self, pass_dir, slug, pins, nodeids, note):
        """Run the ``nodeids`` of ``pass_dir`` that have no row, per selection that collected
        them, then re-run each new crash with repro capture."""
        todo = [n for n in nodeids if n not in recorded_ids(pass_dir)]
        for label, selection in passes(self.args):
            chosen = [n for n in todo if n in self.pass_ids[label]]
            if not chosen:
                continue
            listing = listing_file(pass_dir / f"todo-{label}.txt", chosen)
            print(f"{note} (pass '{label}'): {len(chosen)} cases", flush=True)
            self.pytest(selection, pass_dir, slug, pins, listing, len(chosen))
        self.capture_crashes(pass_dir, slug, pins)

    def capture_crashes(self, pass_dir, slug, pins):
        """Re-run each crash of ``pass_dir`` without a capture row once, with the same pins and
        ``HIPDNN_SDPA_REPRO_CAPTURE``, into ``<pass_dir>/capture/``; within the repro caps.
        """
        capture_dir = pass_dir / CAPTURE_DIR
        done = recorded_ids(capture_dir)
        todo = []
        for row in reconcile.read_rows(pass_dir / reconcile.CSV_NAME):
            if row["outcome"] != "crash" or row["nodeid"] in done:
                continue
            side = row["fail_side"]
            served = reconcile.served_engines(row[side]) if side else set()
            if reconcile.claim_repro(
                self.out / contract.REPRO_DIR, "+".join(sorted(served)), row["nodeid"]
            ):
                todo.append(row["nodeid"])
        for label, selection in passes(self.args):
            chosen = [n for n in todo if n in self.pass_ids[label]]
            if not chosen:
                continue
            capture_dir.mkdir(parents=True, exist_ok=True)
            listing = listing_file(capture_dir / f"nodeids-{label}.txt", chosen)
            print(f"repro capture (pass '{label}'): {len(chosen)} crashes", flush=True)
            self.pytest(
                selection, capture_dir, slug, pins, listing, len(chosen), capture=True
            )

    def run_planned(self, p):
        """One ``PlannedPass``: its metadata and full listing, then its missing cases; a
        not-enumerable pass runs no pytest and gets a message row per case."""
        pass_dir = self.out / p.root / p.rel
        pass_dir.mkdir(parents=True, exist_ok=True)
        meta = {"kind": p.kind, "direction": p.direction}
        for d, name in contract.ENGINE_ENV.items():
            meta[f"pin_{d}"] = p.pins.get(name, "")
            meta[f"{d}_candidate"] = p.pins.get(contract.CANDIDATE_ENV[d], "")
        (pass_dir / reconcile.PASS_META_NAME).write_text(json.dumps(meta) + "\n")
        listing_file(pass_dir / reconcile.NODEIDS_NAME, list(p.cases))
        if p.not_enumerable:
            _not_enumerable_rows(pass_dir, p.cases)
            print(f"{p.note}: {len(p.cases)} not enumerable", flush=True)
            return
        slug = reconcile.pass_slug(self.out, pass_dir)
        self.run_pass(pass_dir, slug, p.pins, p.cases, f"all-engines {p.note}")

    def run_planned_passes(self, options):
        """Every pass beyond pass 1 that ``options`` call for (see ``planned_passes``)."""
        rows = reconcile.read_rows(self.out / reconcile.CSV_NAME)
        for p in planned_passes(self.out, rows, options):
            self.run_planned(p)


def _not_enumerable_rows(pass_dir, messages):
    """Write ``pass_dir``'s ``results.csv``: a ``not-enumerable`` row per case of ``messages`` that
    has none yet. No pytest run is involved."""
    done = recorded_ids(pass_dir)
    with reconcile.open_rows_for_append(pass_dir / reconcile.CSV_NAME) as f:
        for nodeid, message in messages.items():
            if nodeid not in done:
                f.write(
                    reconcile.format_row(
                        {
                            "nodeid": nodeid,
                            "outcome": contract.NOT_ENUMERABLE,
                            "reason": message,
                        }
                    )
                )


def main(argv=None):
    args = parse_args(argv)
    if args.all_engines or args.exhaustive:
        pinned = [k for k in PIN_ENV if os.environ.get(k)]
        if args.fake or pinned:
            raise SystemExit(
                f"{'--exhaustive' if args.exhaustive else '--all-engines'} needs the real "
                "heuristic run: refused with "
                + ("--fake" if args.fake else ", ".join(pinned))
            )
    if not args.hipdnn_prefix:
        raise SystemExit(
            "no hipDNN prefix: activate the setup_env.py venv (it sets ROCM_PATH) or pass "
            "--hipdnn-prefix; every run checks that this prefix's libhipdnn_backend is loaded"
        )
    if args.exhaustive:
        bench = Path(args.hipdnn_prefix) / contract.BENCH_RELATIVE
        if not (bench.is_file() and os.access(bench, os.X_OK)):
            raise SystemExit(
                f"--exhaustive needs {bench} to be an executable file: it lists the candidates "
                "of each engine"
            )
    if args.fwd_no_stats and (args.suite != "forward" or args.packed_gpu):
        raise SystemExit(
            f"--fwd-no-stats needs --suite forward and no --packed-gpu: the backward tests "
            f"consume the forward's Stats, so only {contract.FORWARD_TESTS} is valid"
        )
    gpus, arch = (0, None)
    if args.workers is None or not (args.out or (args.arch and args.run)):
        gpus, arch = detect_gpus()
    workers = args.workers or gpus
    if not workers:
        raise SystemExit("no GPU visible: pass --workers")
    out = resolve_out(args, arch)
    out.mkdir(parents=True, exist_ok=True)
    results = out / reconcile.CSV_NAME
    if results.exists() and results.stat().st_size and not args.resume:
        raise SystemExit(
            f"{results} already has rows; use a fresh --out per run and shard, or --resume"
        )
    options = requested_options(args)
    options_path = out / RUN_OPTIONS_NAME
    if args.resume and (
        options_path.exists() or results.exists() and results.stat().st_size
    ):
        differing = option_differences(run_options(out), options)
        if differing:
            print(
                f"--resume refused: {out} was started with other options "
                f"({options_path if options_path.exists() else 'no ' + RUN_OPTIONS_NAME}):\n  "
                + "\n  ".join(differing)
                + "\nresume with the options of the first link, or use a fresh --out",
                file=sys.stderr,
            )
            return 2
    if not (args.resume and options_path.exists()):
        options_path.write_text(json.dumps(options) + "\n")
    env = layout.clean_env(release=args.release)
    env.pop(contract.FWD_NO_STATS_ENV, None)
    env[contract.LAYOUT_MODE_ENV] = args.layout_mode
    print(f"layout mode: {args.layout_mode}", flush=True)
    if args.fwd_no_stats:
        env[contract.FWD_NO_STATS_ENV] = "1"
        print("forward mode: no Stats", flush=True)
    env[contract.PREFIX_ENV] = str(args.hipdnn_prefix)
    env.pop(contract.EXHAUSTIVE_ENV, None)
    if args.exhaustive:
        env[contract.EXHAUSTIVE_ENV] = (
            "1"  # pass 1 counts candidates; later passes drop it
        )
    env.pop(contract.KERNEL_LOG_ENV, None)
    if args.kernel_ids or args.exhaustive:
        env[contract.KERNEL_LOG_ENV] = (
            "1"  # every pass: candidate honoring compares ids
        )
    env.setdefault(contract.INFLIGHT_DIR_ENV, tempfile.mkdtemp(prefix="sdpa-inflight-"))
    env.setdefault(contract.DECLINES_DIR_ENV, str(out / reconcile.DECLINES_DIR))
    runner = Runner(args, env, out, workers)
    collected_path = out / reconcile.COLLECTED_NAME
    collected_path.write_text("")
    for label, selection in passes(args):
        collect = subprocess.run(
            pytest_cmd(args, selection, nodeids=args.nodeids_file)
            + ["--collect-only", "-q"],
            env=env,
            capture_output=True,
            text=True,
        )
        ids = [line for line in collect.stdout.splitlines() if "::" in line]
        if collect.returncode not in (0, 5):  # 5: nothing collected
            print(
                f"collection failed in pass '{label}':\n"
                + "\n".join(collect.stdout.splitlines()[-40:])
                + collect.stderr
            )
            return 2
        foreign = sorted(
            {
                i.split("::")[0]
                for i in ids
                if Path(i.split("::")[0]).name != contract.FORWARD_TESTS
            }
        )
        if args.fwd_no_stats and foreign:
            print(
                f"--fwd-no-stats is only valid for {contract.FORWARD_TESTS}; pass '{label}' "
                f"also collected: {', '.join(foreign)}"
            )
            return 2
        runner.pass_ids[label] = set(ids)
        with open(collected_path, "a") as f:
            f.write("".join(f"{nodeid}\n" for nodeid in ids))
        print(
            f"pass '{label}': {len(ids)} cases collected -> {collected_path}",
            flush=True,
        )
        todo = [i for i in ids if i not in recorded_ids(out)] if args.resume else ids
        if not todo:
            continue
        if args.resume:  # the missing ids as a node-id file, not --shard
            listing = listing_file(out / f"todo-{label}.txt", todo)
            runner.pytest(selection, out, reconcile.MAIN_PASS, None, listing, len(todo))
        else:
            runner.pytest(
                selection,
                out,
                reconcile.MAIN_PASS,
                None,
                args.nodeids_file,
                len(todo),
                shard=True,
            )
    runner.capture_crashes(out, reconcile.MAIN_PASS, None)
    runner.run_planned_passes(options)
    return runner.status


if __name__ == "__main__":
    sys.exit(main())
