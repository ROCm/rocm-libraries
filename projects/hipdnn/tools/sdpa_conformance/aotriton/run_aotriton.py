#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Run AOTriton's SDPA suite against hipDNN and record one CSV row per case.

    run_aotriton.py [--suite default|forward] [--release N] [--shard I/N] [--fake always-decline]
                    [--fwd-no-stats] [--all-engines] [--packed-gpu] [--out DIR | --arch ARCH --run NAME]
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
* ``pair/<fwd>__<bwd>/``: backward cases with both pins, every stats-capable forward that did not
  serve the case against every applicable backward;
* ``refwd/discovery/`` and ``refwd/<bwd>/``: for backward cases whose forward failed or crashed in
  any pass, the backward unpinned and then pinned on the reference forward
  (``HIPDNN_FWD_ENGINE=reference``); the pair passes then cover those cases with the backward
  engines the discovery pass recorded.

``reconcile.py --engines <out>`` folds the passes into ``engines.csv``. The option composes with
``--suite``, ``--shard``, ``--release``, ``--workers`` and ``--fwd-no-stats`` (forward sweep
only); it refuses ``--fake`` and pre-set ``HIPDNN_FWD_ENGINE`` / ``HIPDNN_BWD_ENGINE``.

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

Every pytest process checks that the ``libhipdnn_backend`` mapped into it is the one under
``--hipdnn-prefix`` (default ``$ROCM_PATH``, which dnn-benchmarking's ``setup_env.py``
activation sets); the runner refuses to start without a prefix.
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

import aotriton_layout as layout
import conformance_contract as contract
import reconcile

DEFAULT_RESULTS = Path.cwd() / "sdpa_conformance_results"
FWD_NO_STATS_ENV = "HIPDNN_SDPA_CONFORMANCE_FWD_NO_STATS"
FWD_NO_STATS_TESTS = "test_forward.py"
ENGINE_ENV = {"fwd": "HIPDNN_FWD_ENGINE", "bwd": "HIPDNN_BWD_ENGINE"}
WATCHDOG_MODULE = "pytest_gpu_lease.watchdog"
LOCK_NAME = "gpu.lock"
CAPTURE_DIR = "capture"


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
        "--fwd-no-stats",
        action="store_true",
        help=f"run the forward without Stats; only valid for {FWD_NO_STATS_TESTS}",
    )
    parser.add_argument(
        "--all-engines",
        action="store_true",
        help="after the normal run, re-run every case with each other applicable engine pinned",
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
    return parser.parse_args(argv)


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
        else layout.FLASH_TESTS / "test_forward.py"
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


def forward_failed(out):
    """Node ids of backward cases whose forward failed or crashed in pass 1, a pinned or a pair pass."""
    out = Path(out)
    dirs = [out] + [
        d for d, meta in reconcile.pass_dirs(out) if meta["kind"] in ("sweep", "pair")
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
            env = {k: v for k, v in env.items() if k not in ENGINE_ENV.values()}
            env.update(pins)
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

    def extra_pass(self, rel, kind, direction, pins, nodeids, note):
        """A pass under ``all-engines/<rel>``: its metadata and full listing, then its missing cases."""
        pass_dir = self.out / reconcile.ENGINES_DIR / rel
        pass_dir.mkdir(parents=True, exist_ok=True)
        meta = {"kind": kind, "direction": direction}
        for d, name in ENGINE_ENV.items():
            meta[f"pin_{d}"] = pins.get(name, "")
        (pass_dir / reconcile.PASS_META_NAME).write_text(json.dumps(meta) + "\n")
        listing_file(pass_dir / reconcile.NODEIDS_NAME, nodeids)
        slug = reconcile.pass_slug(self.out, pass_dir)
        self.run_pass(pass_dir, slug, pins, nodeids, f"all-engines {note}")

    def sweeps(self, work):
        for (direction, engine), nodeids in sorted(work.items()):
            self.extra_pass(
                Path(direction) / engine_slug(engine),
                "sweep",
                direction,
                {ENGINE_ENV[direction]: engine},
                nodeids,
                f"{direction} {engine}",
            )

    def pairs(self, work):
        for (fwd, bwd), nodeids in sorted(work.items()):
            self.extra_pass(
                Path("pair") / f"{engine_slug(fwd)}__{engine_slug(bwd)}",
                "pair",
                "bwd",
                {ENGINE_ENV["fwd"]: fwd, ENGINE_ENV["bwd"]: bwd},
                nodeids,
                f"pair {fwd} x {bwd}",
            )

    def all_engines(self):
        """Pinned sweeps, pair passes, then the reference-forward passes for failed forwards."""
        rows = reconcile.read_rows(self.out / reconcile.CSV_NAME)
        self.sweeps(engine_work(rows))
        self.pairs(pair_work(rows))
        failed = forward_failed(self.out)
        if not failed:
            return
        reference = {ENGINE_ENV["fwd"]: contract.REFERENCE_PIN}
        self.extra_pass(
            Path("refwd") / "discovery",
            "discovery",
            "bwd",
            reference,
            failed,
            "refwd discovery",
        )
        discovery_dir = self.out / reconcile.ENGINES_DIR / "refwd" / "discovery"
        discovery = reconcile.read_rows(discovery_dir / reconcile.CSV_NAME)
        for (_, engine), nodeids in sorted(engine_work(discovery).items()):
            self.extra_pass(
                Path("refwd") / engine_slug(engine),
                "refwd",
                "bwd",
                {**reference, ENGINE_ENV["bwd"]: engine},
                nodeids,
                f"refwd bwd {engine}",
            )
        self.pairs(pair_work(rows, {r["nodeid"]: r["bwd_engines"] for r in discovery}))


def main(argv=None):
    args = parse_args(argv)
    if args.all_engines:
        pinned = [k for k in ENGINE_ENV.values() if os.environ.get(k)]
        if args.fake or pinned:
            raise SystemExit(
                "--all-engines needs the real heuristic run: refused with "
                + ("--fake" if args.fake else ", ".join(pinned))
            )
    if not args.hipdnn_prefix:
        raise SystemExit(
            "no hipDNN prefix: activate the setup_env.py venv (it sets ROCM_PATH) or pass "
            "--hipdnn-prefix; every run checks that this prefix's libhipdnn_backend is loaded"
        )
    if args.fwd_no_stats and (args.suite != "forward" or args.packed_gpu):
        raise SystemExit(
            f"--fwd-no-stats needs --suite forward and no --packed-gpu: the backward tests "
            f"consume the forward's Stats, so only {FWD_NO_STATS_TESTS} is valid"
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
    env = layout.clean_env(release=args.release)
    env.pop(FWD_NO_STATS_ENV, None)
    if args.fwd_no_stats:
        env[FWD_NO_STATS_ENV] = "1"
        print("forward mode: no Stats", flush=True)
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
                if Path(i.split("::")[0]).name != FWD_NO_STATS_TESTS
            }
        )
        if args.fwd_no_stats and foreign:
            print(
                f"--fwd-no-stats is only valid for {FWD_NO_STATS_TESTS}; pass '{label}' "
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
    if args.all_engines:
        runner.all_engines()
    return runner.status


if __name__ == "__main__":
    sys.exit(main())
