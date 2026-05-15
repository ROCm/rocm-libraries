#!/usr/bin/env python3
"""Run CK Tile direct convolution profiler test cases and summarize results."""

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Test case definitions
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    category: str
    binary: str   # e.g. "grouped_conv_fwd_tile"
    args: str     # space-separated argument string


# Mapping from (section_kind, data_type) -> (category_label, binary)
# section_kind: "fwd" or "bwd_data"
# data_type: "1" (FP16) or "2" (BF16)
_CASE_MAP: dict[tuple[str, str], tuple[str, str]] = {
    ("fwd",      "1"): ("FWD conv FP16",      "ckProfiler grouped_conv_fwd_tile"),
    ("fwd",      "2"): ("FWD conv BF16",      "ckProfiler grouped_conv_fwd_tile"),
    ("bwd_data", "1"): ("BWD data conv FP16", "ckProfiler grouped_conv_bwd_data_tile"),
    ("bwd_data", "2"): ("BWD data conv BF16", "ckProfiler grouped_conv_bwd_data_tile"),
}

# Parse test cases from a text file.
# Format:
#   <Section header line>   e.g. "FWD cases" or "BWD data cases"
#   <column header line>    (first token is not a digit – skipped)
#   <arg rows>              first token is data_type (1 or 2)
#   <blank lines>           ignored
#
def parse_test_cases(path: Path) -> list[TestCase]:
    cases: list[TestCase] = []
    current_section: str | None = None  # "fwd" or "bwd_data"

    with open(path) as f:
        for raw_line in f:
            line = raw_line.rstrip()
            stripped = line.strip()

            if not stripped:
                continue

            # Detect section headers (non-digit first token, no column header tokens)
            lower = stripped.lower()
            if not stripped[0].isdigit():
                if "bwd" in lower and "data" in lower:
                    current_section = "bwd_data"
                elif "fwd" in lower:
                    current_section = "fwd"
                # else: column-header line inside a section – skip
                continue

            # Data row
            tokens = stripped.split()
            data_type = tokens[0]  # "1" or "2"
            args = " ".join(tokens)

            if current_section is None:
                continue

            key = (current_section, data_type)
            if key not in _CASE_MAP:
                continue

            category, binary = _CASE_MAP[key]
            cases.append(TestCase(category, binary, args))

    return cases


# ---------------------------------------------------------------------------
# Running a single test case
# ---------------------------------------------------------------------------

@dataclass
class Result:
    case: TestCase
    passed: bool
    best_instance: str = ""
    avg_time_ms: float = 0.0
    tflops: float = 0.0
    gb_s: float = 0.0
    stderr_output: str = ""
    stdout_output: str = ""
    error_msg: str = ""


# Matches the "Best configuration parameters:" block.
_BEST_NAME_RE = re.compile(r"^\s*name:\s*(.+)$")
_BEST_TIME_RE = re.compile(r"^\s*avg_time:\s*([\d.]+)ms$")
_BEST_TFLOPS_RE = re.compile(r"^\s*tflops:\s*([\d.]+)$")
_BEST_GBS_RE = re.compile(r"^\s*GB/s:\s*([\d.]+)$")


def parse_best_perf(stdout: str) -> tuple[str, float, float, float]:
    """Return (name, avg_time_ms, tflops, gb_s) from profiler stdout."""
    in_best = False
    name = ""
    avg_time = 0.0
    tflops = 0.0
    gb_s = 0.0

    for line in stdout.splitlines():
        if "Best configuration parameters:" in line:
            in_best = True
            continue
        if not in_best:
            continue
        m = _BEST_NAME_RE.match(line)
        if m:
            name = m.group(1).strip()
            continue
        m = _BEST_TIME_RE.match(line)
        if m:
            avg_time = float(m.group(1))
            continue
        m = _BEST_TFLOPS_RE.match(line)
        if m:
            tflops = float(m.group(1))
            continue
        m = _BEST_GBS_RE.match(line)
        if m:
            gb_s = float(m.group(1))
            continue

    return name, avg_time, tflops, gb_s


def run_case(bin_path: Path, case: TestCase, verbose: bool) -> Result:
    binary_tokens = case.binary.split()
    exe = bin_path / binary_tokens[0]
    cmd = [str(exe)] + binary_tokens[1:] + case.args.split()

    if verbose:
        print(f"  $ {' '.join(cmd)}")

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError:
        return Result(
            case=case,
            passed=False,
            error_msg=f"Executable not found: {' '.join(cmd)}",
        )

    stdout = proc.stdout
    stderr = proc.stderr.strip()

    passed = len(stderr) == 0
    name, avg_time, tflops, gb_s = parse_best_perf(stdout)

    return Result(
        case=case,
        passed=passed,
        best_instance=name,
        avg_time_ms=avg_time,
        tflops=tflops,
        gb_s=gb_s,
        stderr_output=stderr,
        stdout_output=stdout,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(results: list[Result]) -> None:
    # Group by category
    categories: dict[str, list[Result]] = {}
    for r in results:
        categories.setdefault(r.case.category, []).append(r)

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for cat, cat_results in categories.items():
        print(f"\n[{cat}]")
        print(f"  {'#':<4} {'Status':<6}  {'Time(ms)':>10}  {'TFlops':>8}  {'GB/s':>8}  Args")
        print(f"  {'-'*4}  {'-'*6}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*30}")
        for i, r in enumerate(cat_results, 1):
            status = "PASS" if r.passed else "FAIL"
            if r.passed and r.best_instance:
                print(
                    f"  {i:<4} {status:<6}  {r.avg_time_ms:>10.3f}  {r.tflops:>8.2f}"
                    f"  {r.gb_s:>8.1f}  {r.case.args}"
                )
                print(f"       {'':6}  best: {r.best_instance}")
            else:
                print(f"  {i:<4} {status:<6}  {'N/A':>10}  {'N/A':>8}  {'N/A':>8}  {r.case.args}")
                if r.error_msg:
                    print(f"       error: {r.error_msg}")
                if r.stderr_output:
                    for line in r.stderr_output.splitlines():
                        print(f"       stderr: {line}")

    print()
    print(f"Result: {passed}/{total} passed", end="")
    if failed:
        print(f", {failed} FAILED")
    else:
        print(" (all passed)")
    print("=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run CK Tile direct convolution profiler test cases."
    )
    parser.add_argument(
        "--bin-path",
        required=True,
        help="Directory containing the ckProfiler executables.",
    )
    parser.add_argument(
        "--test-cases",
        default=Path(__file__).parent / "direct_conv_test_cases.txt",
        type=Path,
        help="Path to test-cases file (default: direct_conv_test_cases.txt next to this script).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print commands and full stdout/stderr for each case.",
    )
    parser.add_argument(
        "--category", "-c",
        help="Only run cases whose category label contains this substring (case-insensitive).",
    )
    args = parser.parse_args()

    bin_path = Path(args.bin_path)
    if not bin_path.is_dir():
        print(f"ERROR: --bin-path '{bin_path}' is not a directory.", file=sys.stderr)
        return 1

    cases = parse_test_cases(args.test_cases)
    if not cases:
        print(f"ERROR: No test cases found in '{args.test_cases}'.", file=sys.stderr)
        return 1

    if args.category:
        filter_str = args.category.lower()
        cases = [c for c in cases if filter_str in c.category.lower()]
        if not cases:
            print(f"ERROR: No cases matched category filter '{args.category}'.", file=sys.stderr)
            return 1

    print(f"Running {len(cases)} test case(s) from '{args.test_cases}'")
    print(f"Binary path: {bin_path}")
    print()

    results: list[Result] = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case.category}  args: {case.args}")
        result = run_case(bin_path, case, args.verbose)
        results.append(result)

        if result.passed:
            if result.best_instance:
                print(
                    f"  PASS  {result.avg_time_ms:.3f} ms  {result.tflops:.2f} TFlops"
                    f"  {result.gb_s:.1f} GB/s"
                )
                print(f"        best: {result.best_instance}")
            else:
                print("  PASS  (no best-instance line found in output)")
        else:
            print("  FAIL")
            if result.error_msg:
                print(f"  error: {result.error_msg}")
            if result.stderr_output:
                for line in result.stderr_output.splitlines():
                    print(f"  stderr: {line}")

        if args.verbose and result.stdout_output:
            print("  --- stdout ---")
            for line in result.stdout_output.splitlines():
                print(f"  {line}")
            print("  --- end stdout ---")

    print_summary(results)

    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
