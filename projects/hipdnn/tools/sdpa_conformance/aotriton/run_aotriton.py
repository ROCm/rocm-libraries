#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Run AOTriton's SDPA suite against hipDNN and record one CSV row per case.

    run_aotriton.py [--suite default|forward] [--release N] [--shard I/N] [--fake always-decline]
                    [--layout-mode MODE] [--fwd-no-stats] [--all-engines] [--packed-gpu] [--out DIR | --arch ARCH --run NAME] [--workers N]
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

``--layout-mode`` picks the memory layout every SDPA tensor is allocated in (default ``pytorch``):
``pytorch`` stores O, dO and the gradients like Q (and dQ/dK/dV/dB like their inputs), ``efficient``
stores O, dO, dQ, dK and dV BSHD (dB like the bias), ``aotriton`` keeps the vendored suite's own
layouts. It is exported as ``HIPDNN_SDPA_CONFORMANCE_LAYOUT_MODE`` to every pytest process,
``--all-engines`` pass 2 included.

``--fwd-no-stats`` runs the forward without Stats (``HIPDNN_SDPA_CONFORMANCE_FWD_NO_STATS=1`` in
the pytest workers), for engines that decline when Stats are requested. The backward tests
consume the forward's Stats, so the runner refuses the flag unless every collected case is in
``test_forward.py``.

``--all-engines`` adds a second stage after the normal run (which still decides ``<out>/results.csv``).
Pass 1 records, per case, the applicable engines in heuristic rank order (``fwd_engines`` /
``bwd_engines``). For every engine that is applicable to a case but did not serve it, the runner
pins that engine and re-runs exactly those cases (``--hipdnn-nodeids-file``): ``test_forward.py``
cases with ``HIPDNN_FWD_ENGINE``, every other case with ``HIPDNN_BWD_ENGINE`` (forward left to the
heuristic). Results go to ``<out>/all-engines/<fwd|bwd>/<engine-slug>/`` in the same CSV format,
with ``engine.txt`` naming the engine; ``reconcile.py --engines <out>`` folds them into
``engines.csv``. It composes with ``--suite``, ``--shard``, ``--release``, ``--workers`` and
``--fwd-no-stats`` (forward sweep only); it refuses ``--fake`` and pre-set
``HIPDNN_FWD_ENGINE`` / ``HIPDNN_BWD_ENGINE``.

Every pytest process checks that the ``libhipdnn_backend`` mapped into it is the one under
``--hipdnn-prefix`` (default ``$ROCM_PATH``, which dnn-benchmarking's ``setup_env.py``
activation sets); the runner refuses to start without a prefix.
"""

import argparse
import copy
import os
import re
import subprocess
import sys
from pathlib import Path

import aotriton_layout as layout
import reconcile

DEFAULT_RESULTS = Path.cwd() / "sdpa_conformance_results"
FWD_NO_STATS_ENV = "HIPDNN_SDPA_CONFORMANCE_FWD_NO_STATS"
FWD_NO_STATS_TESTS = "test_forward.py"
ENGINE_ENV = {"fwd": "HIPDNN_FWD_ENGINE", "bwd": "HIPDNN_BWD_ENGINE"}


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
        help=f"run the forward without Stats; only valid for {FWD_NO_STATS_TESTS}",
    )
    parser.add_argument(
        "--all-engines",
        action="store_true",
        help="after the normal run, re-run every case with each other applicable engine pinned",
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


def engine_work(rows):
    """Pass-2 work from pass-1 rows: ``{(direction, engine): [nodeid, ...]}``.

    A ``test_forward.py`` case is a forward case and reads ``fwd_engines`` / ``fwd``; any other
    case is a backward case and reads ``bwd_engines`` / ``bwd``. Per case, every listed engine
    except the one(s) that served it.
    """
    work = {}
    for row in rows:
        direction = reconcile.direction_of(row["nodeid"])
        winners = reconcile.served_engines(row[direction])
        for engine in row[f"{direction}_engines"].split(";"):
            if engine and engine not in winners:
                work.setdefault((direction, engine), []).append(row["nodeid"])
    return work


def pytest_cmd(args, selection, out=None, nodeids=None):
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
    if args.shard:
        cmd.append(f"--shard={args.shard}")
    if args.fake:
        cmd.append(f"--fake={args.fake}")
    if nodeids:
        cmd.append(f"--hipdnn-nodeids-file={nodeids}")
    cmd.append(f"--hipdnn-prefix={args.hipdnn_prefix}")
    return cmd + args.pytest_args + selection


def sweep_engines(args, env, out, pass_ids, workers):
    """Pass 2: one pytest run per (direction, engine) over the cases that engine did not win."""
    status = 0
    args = copy.copy(args)
    args.shard = None  # the listed node ids are already this shard's cases
    work = engine_work(reconcile.read_rows(out / reconcile.CSV_NAME))
    for (direction, engine), nodeids in sorted(work.items()):
        sweep = out / reconcile.ENGINES_DIR / direction / engine_slug(engine)
        sweep.mkdir(parents=True, exist_ok=True)
        (sweep / reconcile.ENGINE_NAME_FILE).write_text(engine + "\n")
        listing = (sweep / "nodeids.txt").resolve()
        listing.write_text("".join(f"{n}\n" for n in nodeids))
        sweep_env = {k: v for k, v in env.items() if k not in ENGINE_ENV.values()}
        sweep_env[ENGINE_ENV[direction]] = engine
        wanted = set(nodeids)
        for label, selection in passes(args):
            count = len(wanted & pass_ids[label])
            if not count:
                continue
            print(
                f"all-engines {direction} {engine} (pass '{label}'): {count} cases",
                flush=True,
            )
            run = subprocess.run(
                pytest_cmd(args, selection, sweep, listing)
                + ["-n", str(workers), f"--max-worker-restart={count}"],
                env=sweep_env,
            )
            status = status or run.returncode
    return status


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
    if results.exists() and results.stat().st_size:
        raise SystemExit(
            f"{results} already has rows; use a fresh --out per run and shard"
        )
    env = layout.clean_env(release=args.release)
    env.pop(FWD_NO_STATS_ENV, None)
    env[layout.LAYOUT_MODE_ENV] = args.layout_mode
    print(f"layout mode: {args.layout_mode}", flush=True)
    if args.fwd_no_stats:
        env[FWD_NO_STATS_ENV] = "1"
        print("forward mode: no Stats", flush=True)
    collected_path = out / reconcile.COLLECTED_NAME
    collected_path.write_text("")
    status = 0
    pass_ids = {}
    for label, selection in passes(args):
        collect = subprocess.run(
            pytest_cmd(args, selection) + ["--collect-only", "-q"],
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
        pass_ids[label] = set(ids)
        with open(collected_path, "a") as f:
            f.write("".join(f"{nodeid}\n" for nodeid in ids))
        print(
            f"pass '{label}': {len(ids)} cases collected -> {collected_path}",
            flush=True,
        )
        if not ids:
            continue
        run = subprocess.run(
            pytest_cmd(args, selection, out)
            + ["-n", str(workers), f"--max-worker-restart={len(ids)}"],
            env=env,
        )
        status = status or run.returncode
    if args.all_engines:
        sweep_status = sweep_engines(args, env, out, pass_ids, workers)
        status = status or sweep_status
    return status


if __name__ == "__main__":
    sys.exit(main())
