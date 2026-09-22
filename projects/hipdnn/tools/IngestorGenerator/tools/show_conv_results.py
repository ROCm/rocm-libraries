#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Display and verify a convolution demo batch from its directory or summary.json.

Only the Python standard library is needed. Reports copied from another machine
are resolved beside the summary before considering their original paths.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ENGINE_NAME = "hipkernel:Gfx950ConvFwd"
PHASES = ("smoke", "headline", "events")
REPORT_ERRORS = (OSError, ValueError, KeyError, TypeError, AttributeError)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_report(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(result, dict), f"Expected a JSON object in {path}")
    return result


def positive_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def expected_labels(phase: str) -> set[str]:
    if phase == "headline":
        return {f"headline-{index:02d}" for index in range(51)}
    labels = {
        f"{dtype}-forced-{tile}" for dtype in ("fp16", "bf16") for tile in (64, 128)
    }
    if phase == "smoke":
        labels.update(
            f"{dtype}-{mode}"
            for dtype in ("fp16", "bf16")
            for mode in ("auto", "reuse")
        )
    return labels


def validate_report(
    row: dict, report: dict, phase: str, *, require_logging: bool = False
) -> tuple[int, float, float, float]:
    require(
        row.get("status") == report.get("status") == "passed"
        and type(row.get("exit_code")) is int
        and row["exit_code"] == 0,
        "Run failed or did not complete",
    )
    for path in ("integrated_correctness", "direct_correctness"):
        require(
            isinstance(report.get(path), dict) and report[path].get("passed") is True,
            f"Independent correctness failed or is missing: {path}",
        )
    require(report.get("engine_name") == ENGINE_NAME, "Unexpected engine")
    require(
        report["comparison"].get("timing_mode")
        == ("events" if phase == "events" else "graph"),
        "Unexpected timing method",
    )
    tile = report["selected_tile_k"]
    require(type(tile) is int and tile in (64, 128), "Invalid selected tile")
    if phase in ("smoke", "events"):
        dtype, mode = row["label"].split("-", 1)
        require(report.get("dtype") == dtype, "Reported dtype differs from the job")
        if mode.startswith("forced-"):
            requested = int(mode.removeprefix("forced-"))
            require(
                report.get("mode") == "forced"
                and report.get("requested_tile_k") == tile == requested,
                "Forced kernel differs from the job",
            )
        else:
            require(report.get("mode") == mode, "Reported mode differs from the job")
    else:
        require(
            report.get("mode") == "forced"
            and report.get("requested_tile_k") == tile == 64,
            "Headline job did not force tile_k=64",
        )
    integrated = report["integrated_timing"]["median_ms"]
    direct = report["direct_timing"]["median_ms"]
    ratio = report["comparison"]["integrated_over_direct"]
    require(
        all(positive_number(value) for value in (integrated, direct, ratio)),
        "Invalid timing or ratio",
    )
    require(
        math.isclose(ratio, integrated / direct, rel_tol=1e-9),
        "Ratio differs from recorded times",
    )
    logging = report.get("timing_logging")
    if require_logging or logging is not None:
        require(
            isinstance(logging, dict)
            and logging.get("backend_global_level_during") == "OFF"
            and logging.get("sync_callback_registered_during") is False
            and logging.get("callbacks_during_timing") == 0
            and logging.get("restored") is True
            and logging.get("backend_global_level_after") == "INFO"
            and logging.get("sync_callback_registered_after") is True,
            "Timing logging controls failed or are missing",
        )
    return tile, integrated, direct, ratio


def restart_check(dtype: str, auto: dict, reuse: dict) -> dict:
    require(
        auto.get("dtype") == reuse.get("dtype") == dtype
        and auto.get("mode") == "auto"
        and reuse.get("mode") == "reuse",
        f"{dtype} search/restart modes are invalid",
    )
    require(
        type(auto.get("pid")) is int
        and type(reuse.get("pid")) is int
        and auto["pid"] > 0
        and reuse["pid"] > 0
        and auto["pid"] != reuse["pid"]
        and auto.get("cache_before", {}) is None,
        f"{dtype} search/restart process evidence is invalid",
    )
    ranking = auto["cache_after"]["record_sha256"]
    require(
        isinstance(ranking, str)
        and ranking
        and ranking
        == reuse["cache_before"]["record_sha256"]
        == reuse["cache_after"]["record_sha256"],
        f"{dtype} cached ranking changed",
    )
    kernel = auto["selected_kernel_id"]
    require(
        isinstance(kernel, str)
        and kernel
        and kernel == reuse["selected_kernel_id"]
        and auto["selected_tile_k"] == reuse["selected_tile_k"],
        f"{dtype} selected kernel changed",
    )
    return {
        "dtype": dtype,
        "different_processes": True,
        "auto_pid": auto["pid"],
        "reuse_pid": reuse["pid"],
        "kernel_id": kernel,
        "ranking_sha256": ranking,
        "ranking_preserved": True,
    }


def show_results(batch: Path) -> int:
    path = batch.expanduser().resolve()
    if path.is_dir():
        path /= "summary.json"
    summary = read_report(path)
    phase, runs = summary.get("phase"), summary.get("runs")
    print(f"Batch: {path.parent}\nPhase: {phase}; status: {summary.get('status')}")
    require(phase in PHASES, "Missing or unknown batch phase")
    require(isinstance(runs, list) and runs, "Batch has no results")
    problems, reports, ratios = [], {}, []
    planned = summary.get("planned_runs")
    labels = expected_labels(phase)
    if summary.get("status") != "passed":
        problems.append(f"Batch has not passed: {summary.get('error', 'incomplete')}")
    if type(planned) is not int or planned != len(labels) or len(runs) != planned:
        problems.append(f"Incomplete batch: {len(runs)} results, planned {planned}")
    if "prechecks" in summary:
        for check in summary["prechecks"]:
            if check.get("exit_code") != 0 or check.get("status") != "passed":
                problems.append(f"Precheck failed: {check.get('label')}")
    print("Both paths are checked independently against the reference.")
    print(
        f"{'Run':<22} {'hipDNN':<7} {'rocKE':<7} {'tile':>4} "
        f"{'hipDNN us':>11} {'rocKE us':>11} {'ratio':>10}"
    )
    for index, row in enumerate(runs):
        label = row.get("label", f"row-{index}") if isinstance(row, dict) else index
        try:
            require(isinstance(row, dict), "Invalid run record")
            require(
                isinstance(label, str) and label in labels and label not in reports,
                "Missing, unexpected, or duplicate run label",
            )
            declared = Path(row["report"])
            local = path.parent / declared.name
            report_path = local if local.is_file() else declared
            if not report_path.is_absolute():
                report_path = path.parent / report_path
            report = read_report(report_path)
            reports[label] = report
            tile, integrated, direct, ratio = validate_report(row, report, phase)
            print(
                f"{label:<22} {'PASS':<7} {'PASS':<7} {tile:>4} "
                f"{integrated * 1000:>11.3f} {direct * 1000:>11.3f} {ratio:>10.6f}"
            )
            ratios.append(ratio)
        except REPORT_ERRORS as exc:
            problems.append(f"{label}: {exc}")
            print(f"{label}: ERROR: {exc}")
    if set(reports) != labels:
        problems.append("Reports do not cover every planned job")
    if phase == "smoke":
        try:
            checks = summary.get("restart_cache_checks", [])
            require(
                len(checks) == 2
                and {check["dtype"] for check in checks} == {"fp16", "bf16"},
                "Missing restart checks",
            )
            for check in checks:
                dtype = check["dtype"]
                actual = restart_check(
                    dtype, reports[f"{dtype}-auto"], reports[f"{dtype}-reuse"]
                )
                require(check == actual, f"{dtype} summary restart evidence differs")
                print(f"{dtype} cache restart: PASS (same winner in a new process)")
        except REPORT_ERRORS as exc:
            problems.append(f"Cache restart: {exc}")
    if problems:
        print("\nBatch verification: FAILED", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        return 1
    if phase == "headline":
        geomean = math.exp(sum(math.log(value) for value in ratios) / len(ratios))
        print(f"Geometric mean hipDNN/direct ratio: {geomean:.6f}")
    print(f"\nVerified {len(ratios)}/{planned} runs: PASS")
    print("Ratio = hipDNN / direct rocKE; 1.0 means equal measured time.")
    if phase == "events":
        print("Ordinary submission timing includes Python enqueue gaps.")
    else:
        print("HIP graph timing excludes compilation and initial execution/search.")
    print(f"Summary: {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path, help="Batch directory or summary.json.")
    args = parser.parse_args()
    return show_results(args.batch)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except REPORT_ERRORS as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
