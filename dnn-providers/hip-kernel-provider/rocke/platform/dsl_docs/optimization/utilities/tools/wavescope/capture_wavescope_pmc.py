# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Capture PMC artifacts for WaveScope using rocKE's perf primitives.

Always saves original profiler CSVs AND versioned measurement JSON. Upload a
CSV in WaveScope's Bottlenecks tab today; the manifest and JSON are the contract
for a future JSON reader. No WaveScope installation or ATT decoder is required.

Run from any directory; the adjacent platform/python package is selected for
both the perf subprocess and its launcher. The launcher's working directory and
other environment settings are preserved. Set PYTHONPATH for additional library
imports your launcher needs.

Usage:
    python3 capture_wavescope_pmc.py --output-dir ./pmc-before \\
        --arch gfx950 --op gemm --shape '{"M":512,"N":512,"K":512}' \\
        --kernel-name my_gemm --match-kernel my_gemm --repeats 3 --warmup 5 \\
        -- python3 bench.py

The destination must not exist. This is a separate PMC execution, not an ATT
capture: matching a kernel name does not establish the same workload or binary.
History storage is off unless --store-history is given. Remaining options are
forwarded to `rocke.benchmark.perf.tool profile` (e.g. --per-dispatch, --cache,
--threshold, --noise-k, --json). Regression exit status 1 is preserved.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PYTHON_ROOT = Path(__file__).resolve().parents[5] / "python"


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="new CSV + JSON bundle directory; never overwritten",
    )
    parser.add_argument(
        "--store-history",
        action="store_true",
        help="also append measurements to the perf history cache",
    )
    if "--" not in raw:
        parser.parse_known_args(raw)
        parser.error("put the kernel launch command after --")
    split = raw.index("--")
    own, command = raw[:split], raw[split + 1 :]
    args, forward = parser.parse_known_args(own)
    if not command:
        parser.error("put the kernel launch command after --")
    if any(
        flag.split("=", 1)[0] in {"--artifacts-dir", "--no-store"} for flag in forward
    ):
        parser.error("use --output-dir and --store-history to control utility output")
    if not (PYTHON_ROOT / "rocke/benchmark/perf/tool/cli.py").is_file():
        parser.error(f"adjacent rocKE perf tool not found under {PYTHON_ROOT}")
    output = args.output_dir.expanduser().absolute()
    if output.exists() or output.is_symlink():
        parser.error(f"output directory already exists: {output}")
    env = os.environ.copy()
    previous = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(PYTHON_ROOT) + (os.pathsep + previous if previous else "")
    invocation = [
        sys.executable,
        "-m",
        "rocke.benchmark.perf.tool",
        "profile",
        *forward,
        "--artifacts-dir",
        str(output),
    ]
    if not args.store_history:
        invocation.append("--no-store")
    invocation.extend(["--", *command])
    result = subprocess.run(invocation, env=env)
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        return result.returncode or 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"\nWaveScope PMC bundle: {output}", file=sys.stderr)
    print(f"Status: {manifest['status']}", file=sys.stderr)
    if manifest["status"] != "complete":
        print(
            "Incomplete capture: retained files are for diagnosis, not a finalized baseline.",
            file=sys.stderr,
        )
        return result.returncode or 1
    print(f"JSON entry point: {manifest_path}", file=sys.stderr)
    print(f"Measurement JSON: {output / manifest['measurement']}", file=sys.stderr)
    for sample in manifest["samples"]:
        capture = sample.get("profile_capture") or {}
        print(
            f"Sample {sample['sample_index']}: profiler {capture.get('status', 'unknown')}",
            file=sys.stderr,
        )
    csvs = [item for item in manifest["files"] if item["kind"] == "pmc_csv"]
    if csvs:
        print(
            "WaveScope: open the ATT trace, then upload a CSV in Bottlenecks:",
            file=sys.stderr,
        )
        for item in csvs:
            print(f"  {output / item['path']}", file=sys.stderr)
        print(
            "Use one repeat at a time; replay passes may contain different counters. Raw CSV includes warmup and other kernels, unlike filtered JSON medians.",
            file=sys.stderr,
        )
    else:
        print(
            "No raw PMC CSVs were captured; JSON timing may still be available. This is not a PMC capture success.",
            file=sys.stderr,
        )
    print(
        "ATT association: UNBOUND. Verify the workload, GPU and binary. JSON is always exported; current WaveScope CSV import does not read it.",
        file=sys.stderr,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
