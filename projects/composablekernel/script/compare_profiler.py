#!/usr/bin/env python3
"""
Compare CK Profiler outputs between baseline (CK Builder) and dispatcher builds.

Runs the CK profiler test cases using two builds (baseline
and dispatcher), parses the output, and generates a markdown comparison report.
"""

import argparse
import subprocess
import re
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# ─── Constants ────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

DEFAULT_BASELINE_DIR = str(PROJECT_DIR / "build-gfx942-baseline" / "bin")
DEFAULT_DISPATCHER_DIR = str(PROJECT_DIR / "build-gfx942" / "bin")
DEFAULT_REPORT_PATH = str(PROJECT_DIR / "profiler_comparison_report.md")
DEFAULT_TOLERANCE_PCT = 5.0

# ─── Regex patterns for parsing profiler output ──────────────────────────────

# Backward data/weight format: [Valid] Perf: ... , KernelName<...> (instance N), SplitK N
# Forward format:              Perf: ... , KernelName<...>
VALID_RE = re.compile(
    r"(?:\[Valid\]\s+)?Perf:\s+([\d.]+)\s+ms,\s+(.+>)"
    r"(?:\s+\(instance \d+\))?"
    r"(?:,\s+SplitK\s+(\d+))?"
)
# Backward data/weight format: [Not supported] KernelName<...>, SplitK N
NOT_SUPPORTED_RE = re.compile(
    r"\[Not supported\]\s+(.+>)(?:,\s+SplitK\s+(\d+))?"
)
# Forward format: bare kernel name on a line (leading whitespace, no Perf: prefix)
BARE_KERNEL_RE = re.compile(
    r"^\s+(Grouped\w+Kernel<.+>)\s*$"
)
BEST_NAME_RE = re.compile(r"name:\s+(.+)")
BEST_TIME_RE = re.compile(r"avg_time:\s+([\d.]+)(?:,\s+SplitK\s+(\d+))?")

# ─── Test case definitions  ─────────────────────────────────────

TEST_CASES = [
    {
        "name": "Forward Case 1",
        "command": "grouped_conv_fwd_tile",
        "args": "1 1 0 2 1 0 1 2 32 32 4 4 3 3 200 200 1 1 1 1 1 1 1 1",
        "shape": "G=32 N=32 K=4 C=4 F=3x3 I=200x200 S=1x1 layout=NHWGC",
    },
    {
        "name": "Forward Case 2",
        "command": "grouped_conv_fwd_tile",
        "args": "1 1 0 2 1 0 1 2 1 32 256 256 3 3 200 200 2 2 1 1 1 1 1 1",
        "shape": "G=1 N=32 K=256 C=256 F=3x3 I=200x200 S=2x2 layout=NHWGC",
    },
    {
        "name": "Forward Depthwise Case 1",
        "command": "grouped_conv_fwd_tile",
        "args": "1 2 0 2 1 0 1 2 32 32 1 1 3 3 200 200 1 1 1 1 1 1 1 1",
        "shape": "G=32 N=32 K=1 C=1 F=3x3 I=200x200 S=1x1 layout=NGCHW (depthwise)",
    },
    {
        "name": "Forward Depthwise Case 2",
        "command": "grouped_conv_fwd_tile",
        "args": "1 2 0 2 1 0 1 2 64 16 1 1 3 3 100 100 2 2 1 1 1 1 1 1",
        "shape": "G=64 N=16 K=1 C=1 F=3x3 I=100x100 S=2x2 layout=NGCHW (depthwise)",
    },
    {
        "name": "Forward Depthwise Case 3",
        "command": "grouped_conv_fwd_tile",
        "args": "1 2 0 2 1 0 1 2 32 32 1 1 5 5 100 100 1 1 1 1 2 2 2 2",
        "shape": "G=32 N=32 K=1 C=1 F=5x5 I=100x100 S=1x1 P=2x2 layout=NGCHW (depthwise)",
    },
    {
        "name": "Backward Data Case 1",
        "command": "grouped_conv_bwd_data_tile",
        "args": "1 1 2 1 0 1 2 32 32 4 4 3 3 200 200 1 1 1 1 1 1 1 1 1",
        "shape": "G=32 N=32 K=4 C=4 F=3x3 I=200x200 S=1x1 split-K=1 layout=NHWGC",
    },
    {
        "name": "Backward Data Case 2",
        "command": "grouped_conv_bwd_data_tile",
        "args": "1 1 2 1 0 1 2 1 32 256 256 3 3 100 100 1 1 1 1 1 1 1 1 1",
        "shape": "G=1 N=32 K=256 C=256 F=3x3 I=100x100 S=1x1 split-K=1 layout=NHWGC",
    },
    {
        "name": "Backward Weight Case 1",
        "command": "grouped_conv_bwd_weight_tile",
        "args": "5 2 2 1 0 1 2 32 32 4 4 3 3 200 200 1 1 1 1 1 1 1 1 4",
        "shape": "G=32 N=32 K=4 C=4 F=3x3 I=200x200 S=1x1 split-K=4 layout=NHWGC (bf16)",
    },
    {
        "name": "Backward Weight Case 2",
        "command": "grouped_conv_bwd_weight_tile",
        "args": "5 2 2 1 0 1 2 1 32 256 256 3 3 100 100 1 1 1 1 1 1 1 1 4",
        "shape": "G=1 N=32 K=256 C=256 F=3x3 I=100x100 S=1x1 split-K=4 layout=NHWGC (bf16)",
    },
]


# ─── Data structures ─────────────────────────────────────────────────────────


@dataclass
class Instance:
    kernel_name: str
    status: str  # "valid" or "not_supported"
    perf_ms: Optional[float] = None
    split_k: Optional[int] = None


@dataclass
class ProfilerResult:
    instances: list = field(default_factory=list)
    best_name: str = ""
    best_time: Optional[float] = None
    best_split_k: Optional[int] = None
    raw_output: str = ""
    return_code: int = 0
    error: str = ""


@dataclass
class ComparisonResult:
    test_name: str
    shape: str
    baseline: ProfilerResult = field(default_factory=ProfilerResult)
    dispatcher: ProfilerResult = field(default_factory=ProfilerResult)
    instances_match: bool = False
    missing_in_dispatcher: list = field(default_factory=list)
    extra_in_dispatcher: list = field(default_factory=list)
    best_match: bool = False
    perf_comparisons: list = field(default_factory=list)
    perf_match: bool = False
    overall_pass: bool = False
    error: str = ""


# ─── Parsing ──────────────────────────────────────────────────────────────────


def parse_profiler_output(output: str) -> ProfilerResult:
    """Parse the raw output of a CK profiler run.

    Handles two output formats:
    - Backward data/weight: [Valid] Perf: ... and [Not supported] ...
    - Forward: bare Perf: ... for valid, bare kernel name for not-supported
    """
    result = ProfilerResult(raw_output=output)

    for raw_line in output.splitlines():
        stripped = raw_line.strip()

        m = VALID_RE.match(stripped)
        if m:
            perf_ms = float(m.group(1))
            kernel_name = m.group(2).strip()
            split_k = int(m.group(3)) if m.group(3) else None
            result.instances.append(
                Instance(kernel_name, "valid", perf_ms, split_k)
            )
            continue

        m = NOT_SUPPORTED_RE.match(stripped)
        if m:
            kernel_name = m.group(1).strip()
            split_k = int(m.group(2)) if m.group(2) else None
            result.instances.append(
                Instance(kernel_name, "not_supported", None, split_k)
            )
            continue

        # Forward format: bare kernel name with leading whitespace (not-supported)
        m = BARE_KERNEL_RE.match(raw_line)
        if m:
            kernel_name = m.group(1).strip()
            result.instances.append(
                Instance(kernel_name, "not_supported", None, None)
            )
            continue

        m = BEST_NAME_RE.match(stripped)
        if m:
            result.best_name = m.group(1).strip()
            continue

        m = BEST_TIME_RE.match(stripped)
        if m:
            result.best_time = float(m.group(1))
            result.best_split_k = int(m.group(2)) if m.group(2) else None
            continue

    return result


def get_unique_instances(instances: list) -> dict:
    """Deduplicate instances by kernel name, keeping the best (lowest) perf."""
    unique = {}
    for inst in instances:
        if inst.kernel_name not in unique:
            unique[inst.kernel_name] = inst
        elif inst.status == "valid" and inst.perf_ms is not None:
            existing = unique[inst.kernel_name]
            if existing.perf_ms is None or inst.perf_ms < existing.perf_ms:
                unique[inst.kernel_name] = inst
    return unique


# ─── Running profiler ─────────────────────────────────────────────────────────


def run_profiler(bin_dir: str, command: str, args: str, dry_run: bool = False) -> ProfilerResult:
    """Run the CK profiler and return parsed results."""
    profiler_path = os.path.join(bin_dir, "ckProfiler")

    if not os.path.isfile(profiler_path):
        result = ProfilerResult()
        result.error = f"Profiler not found: {profiler_path}"
        result.return_code = -1
        return result

    full_cmd = f"{profiler_path} {command} {args}"

    if dry_run:
        print(f"  [dry-run] {full_cmd}")
        result = ProfilerResult()
        result.error = "dry-run mode"
        return result

    print(f"  Running: {full_cmd}")
    try:
        proc = subprocess.run(
            full_cmd,
            shell=True,
            capture_output=True,
            text=True,
            cwd=bin_dir,
            timeout=600,
        )
        output = proc.stdout
        if proc.stderr:
            output += "\n" + proc.stderr
        result = parse_profiler_output(output)
        result.return_code = proc.returncode
        return result
    except subprocess.TimeoutExpired:
        result = ProfilerResult()
        result.error = "Profiler timed out after 600s"
        result.return_code = -1
        return result
    except Exception as e:
        result = ProfilerResult()
        result.error = str(e)
        result.return_code = -1
        return result


# ─── Comparison ───────────────────────────────────────────────────────────────


def compare_results(
    test_case: dict,
    baseline: ProfilerResult,
    dispatcher: ProfilerResult,
    tolerance_pct: float,
) -> ComparisonResult:
    """Compare baseline and dispatcher profiler results."""
    comp = ComparisonResult(
        test_name=test_case["name"],
        shape=test_case["shape"],
        baseline=baseline,
        dispatcher=dispatcher,
    )

    # Handle errors
    if baseline.error or dispatcher.error:
        comp.error = f"Baseline: {baseline.error}" if baseline.error else ""
        if dispatcher.error:
            comp.error += ("; " if comp.error else "") + f"Dispatcher: {dispatcher.error}"
        return comp

    # Deduplicate instances
    baseline_unique = get_unique_instances(baseline.instances)
    dispatcher_unique = get_unique_instances(dispatcher.instances)

    baseline_names = set(baseline_unique.keys())
    dispatcher_names = set(dispatcher_unique.keys())

    # All baseline instances must be present in the dispatcher.
    # The dispatcher may have extra instances (that is acceptable).
    comp.missing_in_dispatcher = sorted(baseline_names - dispatcher_names)
    comp.extra_in_dispatcher = sorted(dispatcher_names - baseline_names)
    comp.instances_match = len(comp.missing_in_dispatcher) == 0

    # Compare performance for shared valid instances (informational)
    shared_names = baseline_names & dispatcher_names
    for name in sorted(shared_names):
        b_inst = baseline_unique[name]
        d_inst = dispatcher_unique[name]

        if b_inst.status == "valid" and d_inst.status == "valid" and b_inst.perf_ms and d_inst.perf_ms:
            diff_pct = abs(d_inst.perf_ms - b_inst.perf_ms) / b_inst.perf_ms * 100
            comp.perf_comparisons.append({
                "kernel": name,
                "baseline_ms": b_inst.perf_ms,
                "dispatcher_ms": d_inst.perf_ms,
                "diff_pct": diff_pct,
                "pass": diff_pct <= tolerance_pct,
            })

    # Compare best instance: accept different names if performance is within tolerance
    comp.best_match = baseline.best_name == dispatcher.best_name

    if baseline.best_time is not None and dispatcher.best_time is not None:
        best_diff_pct = abs(dispatcher.best_time - baseline.best_time) / baseline.best_time * 100
        comp.perf_match = best_diff_pct <= tolerance_pct
    else:
        comp.perf_match = baseline.best_time is None and dispatcher.best_time is None

    # Overall: instances covered, and best performance within tolerance.
    # Different best kernel name is acceptable when performance matches.
    comp.overall_pass = comp.instances_match and comp.perf_match

    return comp


# ─── Report generation ────────────────────────────────────────────────────────


def truncate_kernel_name(name: str, max_len: int = 80) -> str:
    """Shorten kernel name for table display."""
    if len(name) <= max_len:
        return name
    return name[: max_len - 3] + "..."


def generate_report(
    comparisons: list,
    baseline_dir: str,
    dispatcher_dir: str,
    tolerance_pct: float,
) -> str:
    """Generate a markdown comparison report."""
    lines = []
    lines.append("# CK Profiler Comparison Report\n")
    lines.append(f"- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- **Baseline**: `{baseline_dir}`")
    lines.append(f"- **Dispatcher**: `{dispatcher_dir}`")
    lines.append(f"- **Performance tolerance**: {tolerance_pct}%")
    lines.append("")

    for comp in comparisons:
        lines.append(f"## {comp.test_name}\n")
        lines.append(f"**Shape**: {comp.shape}\n")

        if comp.error:
            lines.append(f"**ERROR**: {comp.error}\n")
            lines.append("---\n")
            continue

        # Instance counts
        b_total = len(get_unique_instances(comp.baseline.instances))
        d_total = len(get_unique_instances(comp.dispatcher.instances))
        b_dupes = len(comp.baseline.instances) - b_total
        d_dupes = len(comp.dispatcher.instances) - d_total

        lines.append("### Instance Summary\n")
        lines.append(f"| | Baseline | Dispatcher |")
        lines.append(f"|---|---|---|")
        lines.append(f"| Total instances | {len(comp.baseline.instances)} | {len(comp.dispatcher.instances)} |")
        lines.append(f"| Unique instances | {b_total} | {d_total} |")
        if b_dupes > 0 or d_dupes > 0:
            lines.append(f"| Duplicates removed | {b_dupes} | {d_dupes} |")
        valid_b = sum(1 for i in get_unique_instances(comp.baseline.instances).values() if i.status == "valid")
        valid_d = sum(1 for i in get_unique_instances(comp.dispatcher.instances).values() if i.status == "valid")
        lines.append(f"| Valid instances | {valid_b} | {valid_d} |")
        lines.append(f"| All baseline instances in dispatcher | {'PASS' if comp.instances_match else '**FAIL**'} ||")
        lines.append("")

        if comp.missing_in_dispatcher:
            lines.append(f"#### Baseline Instances Missing in Dispatcher ({len(comp.missing_in_dispatcher)})\n")
            for name in comp.missing_in_dispatcher:
                lines.append(f"- `{name}`")
            lines.append("")

        if comp.extra_in_dispatcher:
            lines.append(f"#### Extra Instances in Dispatcher ({len(comp.extra_in_dispatcher)})\n")
            for name in comp.extra_in_dispatcher:
                lines.append(f"- `{name}`")
            lines.append("")

        # Best instance comparison
        lines.append("### Best Instance\n")
        lines.append(f"| | Name | Time (ms) |")
        lines.append(f"|---|---|---|")
        b_time_str = f"{comp.baseline.best_time:.5f}" if comp.baseline.best_time is not None else "N/A"
        d_time_str = f"{comp.dispatcher.best_time:.5f}" if comp.dispatcher.best_time is not None else "N/A"
        lines.append(f"| Baseline | `{truncate_kernel_name(comp.baseline.best_name, 100)}` | {b_time_str} |")
        lines.append(f"| Dispatcher | `{truncate_kernel_name(comp.dispatcher.best_name, 100)}` | {d_time_str} |")
        if comp.baseline.best_time and comp.dispatcher.best_time:
            best_diff = abs(comp.dispatcher.best_time - comp.baseline.best_time) / comp.baseline.best_time * 100
            lines.append(f"| Perf diff | {best_diff:.2f}% | {'PASS' if comp.perf_match else '**FAIL**'} |")
        lines.append(f"| Same kernel | {'Yes' if comp.best_match else 'No (accepted — perf within tolerance)' if comp.perf_match else 'No'} ||")
        lines.append("")

        # Per-instance performance comparison (informational)
        if comp.perf_comparisons:
            lines.append("### Per-Instance Performance (informational)\n")
            lines.append("| Kernel | Baseline (ms) | Dispatcher (ms) | Diff (%) |")
            lines.append("|---|---|---|---|")
            for p in comp.perf_comparisons:
                kname = truncate_kernel_name(p["kernel"])
                diff_str = f"{p['diff_pct']:.2f}" if p["diff_pct"] is not None else "N/A"
                lines.append(
                    f"| `{kname}` | {p['baseline_ms']:.5f} | {p['dispatcher_ms']:.5f} | {diff_str} |"
                )
            lines.append("")

        # Overall result
        overall = "PASS" if comp.overall_pass else "**FAIL**"
        lines.append(f"### Result: {overall}\n")
        lines.append("---\n")

    # Summary table
    lines.append("## Summary\n")
    lines.append("| Test Case | Instances | Best Perf | Same Best Kernel | Overall |")
    lines.append("|---|---|---|---|---|")
    total_pass = 0
    total_tests = 0
    for comp in comparisons:
        if comp.error:
            lines.append(f"| {comp.test_name} | ERROR | ERROR | ERROR | **ERROR** |")
            total_tests += 1
            continue
        inst_str = "PASS" if comp.instances_match else "**FAIL**"
        perf_str = "PASS" if comp.perf_match else "**FAIL**"
        best_str = "Yes" if comp.best_match else "No"
        overall = "PASS" if comp.overall_pass else "**FAIL**"
        lines.append(f"| {comp.test_name} | {inst_str} | {perf_str} | {best_str} | {overall} |")
        total_tests += 1
        if comp.overall_pass:
            total_pass += 1

    lines.append("")
    lines.append(f"**Total: {total_pass}/{total_tests} tests passed**\n")

    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Compare CK Profiler outputs between baseline and dispatcher builds."
    )
    parser.add_argument(
        "--baseline-dir",
        default=DEFAULT_BASELINE_DIR,
        help=f"Path to baseline profiler bin directory (default: {DEFAULT_BASELINE_DIR})",
    )
    parser.add_argument(
        "--dispatcher-dir",
        default=DEFAULT_DISPATCHER_DIR,
        help=f"Path to dispatcher profiler bin directory (default: {DEFAULT_DISPATCHER_DIR})",
    )
    parser.add_argument(
        "--report",
        default=DEFAULT_REPORT_PATH,
        help=f"Output report path (default: {DEFAULT_REPORT_PATH})",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE_PCT,
        help=f"Performance tolerance percentage (default: {DEFAULT_TOLERANCE_PCT}%%)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )
    parser.add_argument(
        "--tests",
        nargs="+",
        type=int,
        help="Run only specific test case numbers (1-based index)",
    )
    args = parser.parse_args()

    # Validate directories
    for label, d in [("Baseline", args.baseline_dir), ("Dispatcher", args.dispatcher_dir)]:
        if not args.dry_run and not os.path.isdir(d):
            print(f"Error: {label} directory not found: {d}", file=sys.stderr)
            sys.exit(1)

    # Select test cases
    selected_cases = TEST_CASES
    if args.tests:
        selected_cases = []
        for idx in args.tests:
            if 1 <= idx <= len(TEST_CASES):
                selected_cases.append(TEST_CASES[idx - 1])
            else:
                print(f"Warning: test index {idx} out of range (1-{len(TEST_CASES)})", file=sys.stderr)

    if not selected_cases:
        print("No test cases selected.", file=sys.stderr)
        sys.exit(1)

    print(f"Running {len(selected_cases)} test case(s)...")
    print(f"Baseline:   {args.baseline_dir}")
    print(f"Dispatcher: {args.dispatcher_dir}")
    print(f"Tolerance:  {args.tolerance}%")
    print()

    comparisons = []
    for i, tc in enumerate(selected_cases, 1):
        print(f"[{i}/{len(selected_cases)}] {tc['name']} ({tc['shape']})")

        print(f"  Running baseline...")
        baseline_result = run_profiler(args.baseline_dir, tc["command"], tc["args"], args.dry_run)
        if baseline_result.error:
            print(f"  Baseline error: {baseline_result.error}")

        print(f"  Running dispatcher...")
        dispatcher_result = run_profiler(args.dispatcher_dir, tc["command"], tc["args"], args.dry_run)
        if dispatcher_result.error:
            print(f"  Dispatcher error: {dispatcher_result.error}")

        comp = compare_results(tc, baseline_result, dispatcher_result, args.tolerance)
        comparisons.append(comp)

        if comp.error:
            print(f"  Result: ERROR - {comp.error}")
        else:
            status = "PASS" if comp.overall_pass else "FAIL"
            print(f"  Result: {status}")
            if not comp.instances_match:
                print(f"    Instances: FAIL ({len(comp.missing_in_dispatcher)} baseline instances missing in dispatcher)")
            if not comp.perf_match:
                print(f"    Best perf: FAIL")
            if not comp.best_match:
                print(f"    Note: different best kernel (accepted if perf within tolerance)")
        print()

    # Generate report
    report = generate_report(comparisons, args.baseline_dir, args.dispatcher_dir, args.tolerance)

    if not args.dry_run:
        with open(args.report, "w") as f:
            f.write(report)
        print(f"Report written to: {args.report}")

    # Print summary
    total_pass = sum(1 for c in comparisons if c.overall_pass)
    total = len(comparisons)
    print(f"\nSummary: {total_pass}/{total} tests passed")

    return 0 if total_pass == total else 1


if __name__ == "__main__":
    sys.exit(main())
