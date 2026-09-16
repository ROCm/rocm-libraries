# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""Read one engine's row out of a dnn-benchmark report and flatten it.

A flow cannot read this report directly. The extractor understands dotted keys and
numeric indices and nothing else (runner/outputs.py), so there is no way to say "the
row whose engine_name is X" in YAML -- a step can only reach `graphs[0].results[0]`
and hope. Hope is how a run reports the timings of whichever engine happened to sort
first. This script does the selection by name, fails loudly when that name is absent,
and writes flat keys a step can asserts on.

It also collapses the two things a benchmark can mean. `gpu_kernel_stats` is the
kernel; `host_stats` is submit-plus-drain, which has a floor of roughly 0.03 ms on
this class of device and cannot distinguish a good kernel from no kernel on a small
graph. Both are reported, the kernel number is the headline, and `measurable` says
whether the graph was large enough for the number to mean anything.

`--oracle-required` exists because an oracle pass is easy to request and easy to
silently not get: an engine with a single compiled plan reports `tuning_available:
false` and no delta at all, which reads as "no speedup available" when it actually
means "nothing was searched". A flow that is optimising against the oracle should
refuse to proceed on a row that was never tuned.

Exit codes: 0 whenever a verdict was written, including a negative one. 1 only when
this script could not do its job -- an unreadable report, or a requested engine that
does not appear in it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _rows(report: dict) -> list[tuple[str, dict, dict]]:
    """(graph_name, graph, row) for every engine row, reference rows excluded."""
    out = []
    for graph in report.get("graphs", []) or []:
        for row in graph.get("results", []) or []:
            # `role` is emitted only when it is not "engine"; the timed PyTorch
            # comparison row carries role="reference" and is not a candidate.
            if row.get("role") not in (None, "engine"):
                continue
            out.append(
                (graph.get("graph_name") or graph.get("graph_path") or "?", graph, row)
            )
    return out


def _stat(row: dict, block: str, key: str) -> float:
    stats = row.get(block) or {}
    value = stats.get(key)
    return float(value) if isinstance(value, (int, float)) else -1.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", required=True, help="dnn-benchmark results.json")
    ap.add_argument(
        "--engine", required=True, help="engine name the row must belong to"
    )
    ap.add_argument("--out", required=True, help="where to write the flattened JSON")
    ap.add_argument(
        "--floor-ms",
        type=float,
        default=0.05,
        help="kernel median below this is reported as not measurable (default: 0.05)",
    )
    ap.add_argument(
        "--oracle-required",
        action="store_true",
        help="fail the verdict when the row carries no oracle delta",
    )
    args = ap.parse_args()

    out_path = Path(args.out)
    try:
        report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        print(f"error: could not read {args.report}: {error}", file=sys.stderr)
        return 1

    rows = _rows(report)
    match = [(g, gr, r) for (g, gr, r) in rows if r.get("engine_name") == args.engine]
    if not match:
        seen = sorted({str(r.get("engine_name")) for _, _, r in rows}) or ["(none)"]
        print(
            f"error: no row for engine {args.engine!r} in {args.report}; "
            f"the report contains: {', '.join(seen)}",
            file=sys.stderr,
        )
        return 1

    graph_name, _graph, row = match[0]
    kernel_median = _stat(row, "gpu_kernel_stats", "median_ms")
    kernel_mean = _stat(row, "gpu_kernel_stats", "mean_ms")
    host_median = _stat(row, "host_stats", "median_ms")

    correctness = row.get("correctness") or {}
    # `passed` is False for "not checked" as well as for a real mismatch, so the
    # explicit-failure reading is the one a gate can act on.
    validated = 1 if correctness.get("tolerance_match") is not None else 0
    correctness_failed = (
        1
        if (
            correctness.get("execution_success") is False
            or correctness.get("tolerance_match") is False
        )
        else 0
    )

    oracle = row.get("oracle") or {}
    delta = row.get("oracle_delta") or {}
    tuning_available = 1 if oracle.get("tuning_available") else 0
    speedup = (
        float(delta["speedup"])
        if isinstance(delta.get("speedup"), (int, float))
        else -1.0
    )

    result: dict[str, Any] = {
        "engine": args.engine,
        "graph": graph_name,
        "status": row.get("status", "unknown"),
        "ran": 1 if row.get("status") == "success" else 0,
        "kernel_median_ms": kernel_median,
        "kernel_mean_ms": kernel_mean,
        "host_median_ms": host_median,
        "measurable": 1 if kernel_median >= args.floor_ms else 0,
        "floor_ms": args.floor_ms,
        "validated": validated,
        "correctness_failed": correctness_failed,
        "tuning_available": tuning_available,
        "oracle_speedup": speedup,
        "oracle_kernel_mean_ms": (
            _stat(oracle, "gpu_kernel_stats", "mean_ms") if oracle else -1.0
        ),
        "oracle_plans_benchmarked": oracle.get("compiled_plans_benchmarked", -1),
        "oracle_plans_total": oracle.get("compiled_plans_total", -1),
        "oracle_error": row.get("oracle_error", ""),
        "report": str(Path(args.report)),
        "ok": 0,
        "feedback": "",
    }

    problems: list[str] = []
    if result["ran"] != 1:
        problems.append(
            f"the engine did not run this graph: status={result['status']}"
            + (
                f", {row.get('skip_reason') or row.get('error_message')}"
                if row.get("skip_reason") or row.get("error_message")
                else ""
            )
        )
    if result["correctness_failed"]:
        problems.append(
            "the run was validated against a reference and did not match: "
            f"max_rel_diff={correctness.get('max_rel_diff')} rtol={correctness.get('rtol')} "
            f"atol={correctness.get('atol')}. A faster wrong kernel is not an optimisation."
        )
    if result["measurable"] != 1 and result["ran"] == 1:
        problems.append(
            f"kernel median {kernel_median:g} ms is at or below the {args.floor_ms:g} ms "
            "measurement floor, so this graph cannot distinguish a good kernel from a bad "
            "one. Optimise against a larger graph, or the numbers are noise."
        )
    if args.oracle_required and not tuning_available:
        problems.append(
            "no oracle was measured for this row: tuning_available is false, meaning the "
            "engine offered a single compiled plan and nothing was searched. That is not "
            "'already optimal' -- it is 'not measured'. "
            + (
                f"oracle_error: {result['oracle_error']}"
                if result["oracle_error"]
                else ""
            )
        )

    result["ok"] = 0 if problems else 1
    if problems:
        result["feedback"] = (
            "The benchmark for this round is not usable as evidence:\n- "
            + "\n- ".join(problems)
        )
    else:
        head = f"{args.engine} on {graph_name}: kernel median {kernel_median:g} ms"
        if tuning_available and speedup > 0:
            head += (
                f"; the oracle's best plan of {result['oracle_plans_benchmarked']} benchmarked "
                f"is {speedup:.3f}x the heuristic pick"
            )
            if speedup > 1.02:
                head += " -- that gap is what tuning is leaving on the table."
            elif speedup < 0.98:
                head += " -- the heuristic is already beating the tuned pick; the gap is not where the win is."
            else:
                head += " -- the heuristic is already at the oracle, so look elsewhere for the win."
        result["feedback"] = head

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
