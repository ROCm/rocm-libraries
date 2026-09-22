#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Run the installed gfx950 convolution tests and direct rocKE comparisons.

Use the Python environment containing the matching hipDNN frontend wheel, NumPy,
and ROCm-enabled PyTorch. The installed provider and packaged kernels must have
been built first. Run GPU benchmarks one at a time for interpretable timings.

The smoke phase runs the 11 installed correctness cases and eight comparisons
covering FP16/BF16, both tuning choices, automatic selection, and cache restart.
The headline phase compares the 51 shipped headline requests. Each invocation
creates a new batch directory inside --output-dir, preserving prior results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from show_conv_results import (
    ENGINE_NAME,
    REPORT_ERRORS,
    read_report,
    require,
    restart_check,
    show_results,
    validate_report,
)

GENERATOR = Path("projects/hipdnn/tools/IngestorGenerator")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("phase", choices=("smoke", "headline"))
    parser.add_argument("--install-prefix", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Results directory outside Git."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        help="Repository root; otherwise infer from this tool.",
    )
    parser.add_argument(
        "--comgr-library",
        type=Path,
        default=os.environ.get("ROCKE_COMGR_LIB", "/opt/rocm/lib/libamd_comgr.so"),
        help="Same COMGR library used to package the kernels (or ROCKE_COMGR_LIB).",
    )
    parser.add_argument(
        "--llvm-flavor",
        choices=("llvm20", "llvm22"),
        default=os.environ.get("ROCKE_LLVM_FLAVOR", "llvm22"),
        help="Packaging LLVM flavor; defaults to ROCKE_LLVM_FLAVOR or llvm22.",
    )
    args = parser.parse_args()
    if args.llvm_flavor not in ("llvm20", "llvm22"):
        parser.error("ROCKE_LLVM_FLAVOR must be llvm20 or llvm22")
    if args.source_root is None:
        for parent in Path(__file__).resolve().parents:
            if (parent / GENERATOR / "tools/benchmark_conv_integration.py").is_file():
                args.source_root = parent
                break
        if args.source_root is None:
            parser.error("Cannot locate the checkout; pass --source-root")
    for name in ("source_root", "install_prefix", "output_dir", "comgr_library"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


def make_jobs(phase: str, catalog: Path, batch: Path) -> list[tuple]:
    jobs = []
    if phase == "smoke":
        for dtype in ("fp16", "bf16"):
            for tile in (64, 128):
                label = f"{dtype}-forced-{tile}"
                jobs.append((label, ["--dtype", dtype, "--tile-k", str(tile)], label))
        for dtype in ("fp16", "bf16"):
            for mode in ("auto", "reuse"):
                jobs.append(
                    (
                        f"{dtype}-{mode}",
                        ["--dtype", dtype, "--mode", mode],
                        f"{dtype}-ranking",
                    )
                )
        return jobs
    requests = json.loads(catalog.read_text(encoding="utf-8"))
    require(isinstance(requests, list), "Catalog must be a JSON array")
    headline = [row for row in requests if row["_catalog_tile_k"] == [64]]
    require(
        len(headline) == 51, f"Expected 51 headline requests, found {len(headline)}"
    )
    for index, row in enumerate(headline):
        label = f"headline-{index:02d}"
        request = batch / f"{label}.request.json"
        request.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
        jobs.append(
            (
                label,
                [
                    "--request-file",
                    str(request),
                    "--tile-k",
                    "64",
                    "--reference-device",
                    "gpu",
                    "--iterations",
                    "10",
                    "--samples",
                    "5",
                    "--warmup",
                    "3",
                ],
                label,
            )
        )
    return jobs


def precheck(command: list[str], log_path: Path, cwd: Path, env: dict) -> int:
    with (
        log_path.open("w", encoding="utf-8") as log,
        subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        ) as process,
    ):
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return process.wait()


def main() -> int:
    args = arguments()
    tool = args.source_root / GENERATOR / "tools/benchmark_conv_integration.py"
    catalog = args.source_root / GENERATOR / "configs/gfx950_conv_fwd.requests.json"
    plugin_dir = args.install_prefix / "lib/hipdnn_plugins/engines"
    plugin = plugin_dir / "libhip_kernel_provider.so"
    backend = args.install_prefix / "lib/libhipdnn_backend.so"
    descriptor_root = plugin_dir / "arch_content/hip-kernel-provider"
    for path in (tool, catalog, plugin, backend, args.comgr_library):
        require(path.is_file(), f"Required file not found: {path}")
    require(descriptor_root.is_dir(), f"Missing descriptor tree: {descriptor_root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    batch = Path(tempfile.mkdtemp(prefix=f"probe-{args.phase}-", dir=args.output_dir))
    print(f"Results directory: {batch}", flush=True)
    env = dict(os.environ)
    env.pop("HIPDNN_DESCRIPTOR_DIR", None)
    library_dirs = [str(args.install_prefix / "lib"), str(args.comgr_library.parent)]
    if env.get("LD_LIBRARY_PATH"):
        library_dirs.append(env["LD_LIBRARY_PATH"])
    env.update(
        LD_LIBRARY_PATH=os.pathsep.join(library_dirs),
        AMD_COMGR_CACHE_DIR=str(args.output_dir / "comgr-cache"),
        ROCKE_COMGR_LIB=str(args.comgr_library),
        ROCKE_LLVM_FLAVOR=args.llvm_flavor,
        HIPDNN_CACHE_DIR=str(batch / "cache/prechecks"),
        HIPDNN_LOG_FILE=str(batch / "prechecks.hipdnn.log"),
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS=env.get("OMP_NUM_THREADS") or "4",
    )
    common = [
        sys.executable,
        str(tool),
        "--plugin-path",
        str(plugin),
        "--descriptor-root",
        str(descriptor_root),
        "--backend-library",
        str(backend),
        "--comgr-library",
        str(args.comgr_library),
        "--llvm-flavor",
        args.llvm_flavor,
    ]
    summary = {
        "phase": args.phase,
        "batch_directory": str(batch),
        "status": "running",
        "tool_sha256": hashlib.sha256(tool.read_bytes()).hexdigest(),
        "catalog_sha256": hashlib.sha256(catalog.read_bytes()).hexdigest(),
        "planned_runs": 8 if args.phase == "smoke" else 51,
        "runs": [],
        "environment": {
            key: env[key]
            for key in ("LD_LIBRARY_PATH", "AMD_COMGR_CACHE_DIR", "OMP_NUM_THREADS")
        },
    }
    summary_path = batch / "summary.json"

    def save() -> None:
        summary_path.write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

    active = None
    save()
    try:
        jobs = make_jobs(args.phase, catalog, batch)
        if args.phase == "smoke":
            xml_path = batch / "integration-tests.xml"
            commands = [
                (
                    "list-engines",
                    [
                        str(args.install_prefix / "bin/hipdnn_list_engines"),
                        "--plugin-dir",
                        str(plugin_dir),
                    ],
                ),
                (
                    "integration-tests",
                    [
                        str(args.install_prefix / "bin/hipdnn_integration_tests"),
                        "--test-article",
                        str(plugin),
                        "--test-engine",
                        ENGINE_NAME,
                        "--reference-executor",
                        "gpu",
                        "--fail-on-unsupported",
                        "--gtest_filter=quick_Gfx950ConvFwd*:standard_Gfx950ConvFwd*",
                        f"--gtest_output=xml:{xml_path}",
                    ],
                ),
            ]
            summary["prechecks"] = []
            for label, command in commands:
                print(f"\nChecking {label}:", flush=True)
                log_path = batch / f"{label}.log"
                row = {
                    "label": label,
                    "command": command,
                    "log": str(log_path),
                    "status": "running",
                }
                active = row
                summary["prechecks"].append(row)
                save()
                row["exit_code"] = precheck(command, log_path, args.source_root, env)
                require(row["exit_code"] == 0, f"{label} failed; read {log_path}")
                if label == "list-engines":
                    require(
                        ENGINE_NAME in log_path.read_text(encoding="utf-8"),
                        f"{ENGINE_NAME} was not enumerated; read {log_path}",
                    )
                else:
                    cases = ET.parse(xml_path).getroot().findall(".//testcase")
                    require(
                        len(cases) == 11
                        and all(
                            case.get("status") == "run"
                            and case.find("failure") is None
                            and case.find("error") is None
                            and case.find("skipped") is None
                            for case in cases
                        ),
                        f"Expected 11 passing, unskipped tests; read {xml_path}",
                    )
                    row["passed_tests"] = len(cases)
                row["status"] = "passed"
                active = None
                save()
        for index, (label, flags, cache) in enumerate(jobs, 1):
            output = batch / f"{label}.json"
            log_path = batch / f"{label}.console.log"
            command = (
                common
                + flags
                + [
                    "--cache-dir",
                    str(batch / "cache" / cache),
                    "--log-file",
                    str(batch / f"{label}.hipdnn.log"),
                    "--output",
                    str(output),
                ]
            )
            row = {
                "label": label,
                "command": command,
                "status": "running",
                "report": str(output),
                "console_log": str(log_path),
            }
            active = row
            summary["runs"].append(row)
            save()
            print(f"START {index}/{len(jobs)} {label}; log: {log_path}", flush=True)
            start = time.monotonic()
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    command,
                    cwd=args.source_root,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            row.update(
                exit_code=result.returncode, wall_seconds=time.monotonic() - start
            )
            require(result.returncode == 0, f"{label} failed; read {log_path}")
            report = read_report(output)
            row["status"] = report.get("status", "missing")
            validate_report(row, report, args.phase, require_logging=True)
            row["timing_logging_verified"] = True
            active = None
            save()
            print(f"END {index}/{len(jobs)} {label}: PASS", flush=True)
        if args.phase == "smoke":
            summary["restart_cache_checks"] = [
                restart_check(
                    dtype,
                    read_report(batch / f"{dtype}-auto.json"),
                    read_report(batch / f"{dtype}-reuse.json"),
                )
                for dtype in ("fp16", "bf16")
            ]
        summary["status"] = "passed"
        save()
        require(show_results(summary_path) == 0, "Final batch verification failed")
        return 0
    except (*REPORT_ERRORS, ET.ParseError, KeyboardInterrupt) as exc:
        summary["status"] = "failed"
        summary["error"] = str(exc) or "Interrupted"
        if active is not None:
            active.update(status="failed", error=summary["error"])
        save()
        print(f"ERROR: {summary['error']}", file=sys.stderr)
        print(f"Results and logs: {batch}\nSummary: {summary_path}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except REPORT_ERRORS as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
