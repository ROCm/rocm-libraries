#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Run AOTriton's SDPA suite against hipDNN and record one CSV row per case.

    run_aotriton.py [--suite default|forward] [--release N] [--shard I/N] [--fake always-decline]
                    [--packed-gpu] [--out DIR | --arch ARCH --run NAME] [--workers N]
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

Every pytest process checks that the ``libhipdnn_backend`` mapped into it is the one under
``--hipdnn-prefix`` (default ``$ROCM_PATH``, which dnn-benchmarking's ``setup_env.py``
activation sets); the runner refuses to start without a prefix.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import aotriton_layout as layout
import reconcile

DEFAULT_RESULTS = Path.cwd() / "sdpa_conformance_results"


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--suite", choices=("default", "forward"), default="default")
    parser.add_argument(
        "--release", type=int, choices=(0, 1, 2, 3), help="exports FOR_RELEASE"
    )
    parser.add_argument("--shard", metavar="I/N")
    parser.add_argument("--fake", choices=("always-decline",))
    parser.add_argument("--packed-gpu", action="store_true")
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


def pytest_cmd(args, selection, out=None):
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
    cmd.append(f"--hipdnn-prefix={args.hipdnn_prefix}")
    return cmd + args.pytest_args + selection


def main(argv=None):
    args = parse_args(argv)
    if not args.hipdnn_prefix:
        raise SystemExit(
            "no hipDNN prefix: activate the setup_env.py venv (it sets ROCM_PATH) or pass "
            "--hipdnn-prefix; every run checks that this prefix's libhipdnn_backend is loaded"
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
    collected_path = out / reconcile.COLLECTED_NAME
    collected_path.write_text("")
    status = 0
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
    return status


if __name__ == "__main__":
    sys.exit(main())
