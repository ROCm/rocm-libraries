# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Capture PMC artifacts for WaveScope using rocKE's perf primitives.

Capture retains original profiler CSVs and writes versioned measurement JSON.
WaveScope's Bottlenecks tab imports the recommended CSVs from successful profiler
samples; `manifest.json` indexes the artifacts and records each sample's status.
Wall-only measurements retain their timing JSON and any profiler troubleshooting
artifacts. Capture runs through the adjacent rocKE perf package.

With `--trace-dir`, keep `--output-dir` as a sibling of the ATT capture tree.
The complete bundle remains separate, while one successful repeat's replay CSVs
are copied beside `code.json`; WaveScope discovers and merges those top-level
sidecars when it opens the trace folder.

Run from any directory; the adjacent platform/python package is selected for
both the perf subprocess and its launcher. The launcher's working directory and
other environment settings are preserved. Set PYTHONPATH for additional library
imports your launcher needs.

Usage:
    python3 capture_wavescope_pmc.py --output-dir ./pmc-before \\
        --arch gfx950 --op gemm --shape '{"M":512,"N":512,"K":512}' \\
        --kernel-name my_gemm --match-kernel my_gemm --repeats 3 --warmup 5 \\
        -- python3 bench.py

    python3 capture_wavescope_pmc.py --output-dir "$RUN_DIR/pmc_bundle" \\
        --trace-dir "$TRACE_DIR" --arch gfx950 --op gemm \\
        --kernel-name my_gemm --match-kernel my_gemm -- python3 bench.py

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
import shutil
import subprocess
import sys
from pathlib import Path

PYTHON_ROOT = Path(__file__).resolve().parents[5] / "python"

TRACE_REQUIRED = ("code.json", "filenames.json", "occupancy.json")


def _is_wave_pmc_sidecar(path: Path) -> bool:
    name = path.name.lower()
    return name == "pmc_perf.csv" or name.endswith("counter_collection.csv")


def _recommended_csvs(manifest: dict) -> list[dict]:
    """PMC CSV inventory entries from the first successful sample that has any."""
    for sample in manifest.get("samples", []):
        capture = sample.get("profile_capture") or {}
        if capture.get("status") != "complete":
            continue
        raw_dir = Path(sample["raw_dir"])
        selected = sorted(
            (
                item
                for item in manifest.get("files", [])
                if item.get("kind") == "pmc_csv"
                and raw_dir in Path(item["path"]).parents
            ),
            key=lambda item: item["path"],
        )
        if selected:
            return selected
    return []


def _publish_to_trace(output: Path, trace_dir: Path, csvs: list[dict]) -> list[Path]:
    """Copy one sample's replay CSVs beside code.json for WaveScope auto-discovery."""
    plans: list[tuple[Path, Path, Path]] = []
    used: dict[str, int] = {}
    for item in csvs:
        relative = Path(item["path"])
        source = output / relative
        replay = next(
            (part for part in relative.parts if part.startswith("pmc_")), "pmc_0"
        )
        used[replay] = used.get(replay, 0) + 1
        ordinal = "" if used[replay] == 1 else f"_{used[replay]}"
        destination = trace_dir / f"rocke_{replay}{ordinal}_counter_collection.csv"
        temporary = destination.with_name(destination.name + ".tmp")
        if not source.is_file():
            raise RuntimeError(f"recommended PMC CSV is missing: {source}")
        if destination.exists() or destination.is_symlink() or temporary.exists():
            raise RuntimeError(f"WaveScope sidecar already exists: {destination}")
        plans.append((source, temporary, destination))

    published: list[Path] = []
    try:
        for source, temporary, _ in plans:
            shutil.copyfile(source, temporary)
        for _, temporary, destination in plans:
            temporary.replace(destination)
            published.append(destination)
    except OSError:
        for _, temporary, destination in plans:
            temporary.unlink(missing_ok=True)
            if destination in published:
                destination.unlink(missing_ok=True)
        raise
    return published


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
    parser.add_argument(
        "--trace-dir",
        type=Path,
        help=(
            "existing WaveScope ATT dispatch directory; copy one successful "
            "sample's replay CSVs beside code.json for automatic discovery"
        ),
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
    trace_dir = args.trace_dir.expanduser().absolute() if args.trace_dir else None
    if trace_dir is not None:
        missing = [name for name in TRACE_REQUIRED if not (trace_dir / name).is_file()]
        if not trace_dir.is_dir() or missing:
            detail = ", ".join(missing) if missing else "not a directory"
            parser.error(f"--trace-dir is not a WaveScope dispatch folder ({detail})")
        existing = sorted(
            path.name
            for path in trace_dir.iterdir()
            if path.is_file() and _is_wave_pmc_sidecar(path)
        )
        if (
            trace_dir == output
            or trace_dir in output.parents
            or output in trace_dir.parents
        ):
            parser.error(
                "--output-dir and --trace-dir must be separate sibling trees; "
                "a nested bundle makes browser folder import count retained CSVs twice"
            )
        if existing:
            parser.error(
                "--trace-dir already contains PMC sidecars; refusing ambiguous "
                f"replacement: {', '.join(existing)}"
            )
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
    print(f"Export status: {manifest['status']}", file=sys.stderr)
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
    csvs = _recommended_csvs(manifest)
    published: list[Path] = []
    if trace_dir is not None and csvs:
        try:
            published = _publish_to_trace(output, trace_dir, csvs)
        except (OSError, RuntimeError) as exc:
            print(f"Could not publish WaveScope CSV sidecars: {exc}", file=sys.stderr)
            return result.returncode or 1
    if csvs:
        if published:
            print(f"WaveScope-ready trace folder: {trace_dir}", file=sys.stderr)
            print("Published counter sidecars:", file=sys.stderr)
            for path in published:
                print(f"  {path}", file=sys.stderr)
        else:
            print(
                "WaveScope: open the ATT trace, then upload these CSVs in Bottlenecks:",
                file=sys.stderr,
            )
            for item in csvs:
                print(f"  {output / item['path']}", file=sys.stderr)
        print(
            "One successful repeat was selected; replay passes may contain different counters. Raw CSV includes warmup and other kernels, unlike filtered JSON medians.",
            file=sys.stderr,
        )
    else:
        print(
            "CSV upload requires a successful profiler capture with raw PMC CSVs. Retained profiler output is available for diagnosis; measurement JSON records timing.",
            file=sys.stderr,
        )
    print(
        "ATT association: UNBOUND. Verify the workload, GPU and binary.",
        file=sys.stderr,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
