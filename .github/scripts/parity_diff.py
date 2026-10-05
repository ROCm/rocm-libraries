#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Join rocJITsu gtest results against a hardware baseline.

Classifications:
  MATCH              same verdict on hardware and under rocJITsu
  EMU_FAIL_HW_PASS   emulator failed a case hardware passed
  EMU_PASS_HW_FAIL   emulator passed a case hardware failed
  UNSUPPORTED        emulator could not execute the case
  MISSING            emulator reported a case absent from the baseline
  NOT_RUN            baseline case was not executed (excluded from parity %)
"""

import argparse
import json
import sys
from pathlib import Path


def classify(hw_status: str, emu_status: str) -> str:
    if emu_status == "unsupported":
        return "UNSUPPORTED"
    if hw_status == emu_status:
        return "MATCH"
    if emu_status == "failed" and hw_status == "passed":
        return "EMU_FAIL_HW_PASS"
    if emu_status == "passed" and hw_status == "failed":
        return "EMU_PASS_HW_FAIL"
    return "UNSUPPORTED"


def unique_cases(cases: list[dict]) -> list[dict]:
    """One row per gtest id. A failure replaces an earlier pass for that id."""
    rows = []
    index_by_id = {}
    for case in cases:
        case_id = case["id"]
        previous = index_by_id.get(case_id)
        if previous is None:
            index_by_id[case_id] = len(rows)
            rows.append(case)
            continue
        if rows[previous].get("status") != "failed" and case.get("status") == "failed":
            rows[previous] = case
    return rows


def diff_results(hw: dict, emu: dict) -> dict:
    hw_cases = {case["id"]: case for case in hw.get("gtest", [])}
    rows = []
    seen = set()
    for case in unique_cases(emu.get("cases", [])):
        case_id = case["id"]
        seen.add(case_id)
        hw_case = hw_cases.get(case_id)
        if hw_case is None:
            verdict = "MISSING"
            hw_status = ""
            hw_seconds = None
        else:
            hw_status = hw_case.get("status", "")
            hw_seconds = hw_case.get("seconds")
            verdict = classify(hw_status, case.get("status", ""))
        rows.append(
            {
                "id": case_id,
                "class": verdict,
                "hw_status": hw_status,
                "emu_status": case.get("status", ""),
                "hw_seconds": hw_seconds,
                "emu_seconds": case.get("seconds"),
                "binary": case.get("binary", ""),
            }
        )
    not_run = [case_id for case_id in hw_cases if case_id not in seen]
    comparable = [row for row in rows if row["class"] not in ("MISSING",)]
    matches = sum(1 for row in comparable if row["class"] == "MATCH")
    denominator = len(comparable)
    parity_pct = (100.0 * matches / denominator) if denominator else 0.0
    baseline_count = len(hw_cases)
    coverage_pct = (
        (100.0 * denominator / baseline_count) if baseline_count else 0.0
    )
    counts = {}
    for row in rows:
        counts[row["class"]] = counts.get(row["class"], 0) + 1
    counts["NOT_RUN"] = len(not_run)
    binaries = emu.get("binaries", [])
    timed_out = sum(1 for binary in binaries if binary.get("status") == "timeout")
    failed_binaries = sum(1 for binary in binaries if binary.get("status") == "failed")
    return {
        "rows": rows,
        "not_run": not_run,
        "counts": counts,
        "matches": matches,
        "denominator": denominator,
        "parity_pct": parity_pct,
        "baseline_count": baseline_count,
        "coverage_pct": coverage_pct,
        "binaries": binaries,
        "timed_out_binaries": timed_out,
        "failed_binaries": failed_binaries,
        "slowdowns": largest_slowdowns(rows),
    }


def largest_slowdowns(rows: list[dict], limit: int = 10) -> list[dict]:
    """Cases where both sides recorded a positive hardware time, slowest first."""
    ranked = []
    for row in rows:
        hw_seconds = row.get("hw_seconds")
        emu_seconds = row.get("emu_seconds")
        if not hw_seconds or emu_seconds is None or hw_seconds <= 0:
            continue
        ranked.append(
            {
                "id": row["id"],
                "hw_seconds": hw_seconds,
                "emu_seconds": emu_seconds,
                "slowdown": emu_seconds / hw_seconds,
            }
        )
    ranked.sort(key=lambda item: item["slowdown"], reverse=True)
    return ranked[:limit]


def render_report(hw: dict, emu: dict, result: dict) -> str:
    lines = [
        "# hipRAND rocJITsu parity (gfx942)",
        "",
        f"- Hardware baseline run: {hw.get('run_id', '')} job {hw.get('job_id', '')}",
        f"- Hardware runner: {hw.get('runner_label', '')}",
        f"- Hardware SHA: {hw.get('head_sha', '')}",
        f"- Emulator runner: {emu.get('runner_label', '')}",
        f"- Emulator ROCm version: {emu.get('rocm_version', '')}",
        f"- ctest label: `{emu.get('ctest_label', '')}`",
        f"- gtest filter: `{emu.get('gtest_filter', '')}`",
        "",
        (
            f"Parity among completed gtest cases: "
            f"{result['matches']}/{result['denominator']} "
            f"({result['parity_pct']:.1f}% MATCH)"
        ),
        (
            f"Coverage: {result['denominator']}/{result['baseline_count']} "
            f"unique baseline cases executed ({result['coverage_pct']:.1f}%)"
        ),
        "",
        "Each gtest id is counted once. NOT_RUN cases were not reached, "
        "including cases still running when a binary hit the CTest timeout, "
        "and they stay out of the parity percentage.",
        "",
    ]
    binaries = result.get("binaries") or []
    if binaries:
        lines.extend(
            [
                "## CTest binaries",
                "",
                "| binary | status | seconds |",
                "| --- | --- | --- |",
            ]
        )
        for binary in binaries:
            lines.append(
                f"| {binary['name']} | {binary['status']} | {binary['seconds']:.2f} |"
            )
        lines.append("")
        if result["timed_out_binaries"] or result["failed_binaries"]:
            lines.append(
                f"Timed-out binaries: {result['timed_out_binaries']}. "
                f"Failed binaries: {result['failed_binaries']}. "
                "The MATCH percentage above is only the cases that finished."
            )
            lines.append("")
    slowdowns = result.get("slowdowns") or []
    if slowdowns:
        lines.extend(
            [
                "## Largest slowdowns",
                "",
                "Slowdown is emulator seconds divided by hardware seconds. "
                "Cases whose hardware time is 0 are omitted.",
                "",
                "| id | hw_s | emu_s | slowdown |",
                "| --- | --- | --- | --- |",
            ]
        )
        for item in slowdowns:
            lines.append(
                f"| {item['id']} | {item['hw_seconds']} | "
                f"{item['emu_seconds']} | {item['slowdown']:.0f}x |"
            )
        lines.append("")
    lines.extend(
        [
            "| class | count |",
            "| --- | --- |",
        ]
    )
    for name in (
        "MATCH",
        "EMU_FAIL_HW_PASS",
        "EMU_PASS_HW_FAIL",
        "UNSUPPORTED",
        "MISSING",
        "NOT_RUN",
    ):
        lines.append(f"| {name} | {result['counts'].get(name, 0)} |")
    lines.extend(
        [
            "",
            "| id | class | hw | emu | hw_s | emu_s |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in result["rows"]:
        lines.append(
            "| {id} | {klass} | {hw} | {emu} | {hw_s} | {emu_s} |".format(
                id=row["id"],
                klass=row["class"],
                hw=row["hw_status"],
                emu=row["emu_status"],
                hw_s="" if row["hw_seconds"] is None else row["hw_seconds"],
                emu_s="" if row["emu_seconds"] is None else row["emu_seconds"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hw", required=True)
    parser.add_argument("--emu", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--summary-json", default="")
    args = parser.parse_args()

    hw = json.loads(Path(args.hw).read_text(encoding="utf-8"))
    emu = json.loads(Path(args.emu).read_text(encoding="utf-8"))
    result = diff_results(hw, emu)
    report = render_report(hw, emu, result)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    if args.summary_json:
        summary = {
            "parity_pct": result["parity_pct"],
            "matches": result["matches"],
            "denominator": result["denominator"],
            "baseline_count": result["baseline_count"],
            "coverage_pct": result["coverage_pct"],
            "counts": result["counts"],
            "timed_out_binaries": result["timed_out_binaries"],
            "failed_binaries": result["failed_binaries"],
            "slowdowns": result["slowdowns"],
        }
        Path(args.summary_json).write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
