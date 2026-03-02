#!/usr/bin/env python3
"""
Analyze timing instrumentation output from tensilelite-client.

This script parses the TIMING: lines from stderr and generates a summary report
of where time is being spent during test execution.

Usage:
    # Run tests with timing instrumentation and capture stderr:
    pytest -s ... --global-parameters "TimingInstrumentation=True" 2>&1 | tee test_output.log

    # Analyze the output:
    python scripts/analyze_timing.py test_output.log

The timing output format is:
    TIMING:<category>:<duration_ms>
    TIMING_CONTEXT:M=...,N=...,K=...,batch=...,typeA=...,typeD=...

Timing category nesting hierarchy:

    The timing categories have a hierarchical nesting structure. Wall clock time
    should be calculated from the top-level Python phases only to avoid double-counting.

    SEQUENTIAL TOP-LEVEL PHASES (no overlap):
    ├── python_benchmark_problems
    │   │
    │   └── NESTED DETAIL PHASES (siblings, not nested in each other):
    │       ├── python_solution_generation
    │       ├── python_kernel_compilation
    │       └── python_client_execution
    │           │
    │           └── ALL C++ CLIENT PHASES (run as subprocess):
    │               ├── library_loading
    │               ├── code_object_loading
    │               ├── lazy_loading_init
    │               ├── data_init_setup
    │               ├── listener_setup
    │               ├── reporter_setup
    │               ├── pre_problem
    │               ├── cpu_data_init
    │               ├── cpu_reference_gemm
    │               ├── gpu_input_preparation
    │               ├── rotating_buffer_preparation
    │               ├── solution_selection
    │               ├── pre_solution
    │               ├── kernel_solving
    │               ├── warmup_runs
    │               ├── benchmark_runs
    │               ├── gpu_kernel_execution
    │               ├── post_solution
    │               ├── post_problem
    │               └── finalize_report
    │
    ├── python_library_logic (no nested timings)
    │
    └── python_client_writer (no nested timings)

    Key nesting relationships:
    - python_solution_generation, python_kernel_compilation, and python_client_execution
      are all nested inside python_benchmark_problems
    - These three detail phases are siblings (sequential, not nested in each other)
    - All C++ phases are nested inside python_client_execution
    - The C++ phases themselves are sequential/siblings, not nested within each other
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

# --- Hierarchy definition ---

TOP_LEVEL_PHASES = [
    "python_benchmark_problems",
    "python_library_logic",
    "python_client_writer",
]

DETAIL_PHASES = [
    "python_solution_generation",
    "python_kernel_compilation",
    "python_client_execution",
]
DETAIL_PARENT = "python_benchmark_problems"

CPP_PHASE_GROUPS = {
    "C++ Client Setup": [
        "library_loading",
        "code_object_loading",
        "lazy_loading_init",
        "data_init_setup",
        "listener_setup",
        "reporter_setup",
    ],
    "Data Preparation": [
        "pre_problem",
        "cpu_data_init",
        "gpu_input_preparation",
        "rotating_buffer_preparation",
    ],
    "Reference Computation": [
        "cpu_reference_gemm",
    ],
    "Kernel Execution": [
        "solution_selection",
        "pre_solution",
        "kernel_solving",
        "warmup_runs",
        "benchmark_runs",
        "gpu_kernel_execution",
        "post_solution",
        "post_problem",
        "finalize_report",
    ],
}
CPP_PARENT = "python_client_execution"

ALL_CPP_PHASES = [c for cats in CPP_PHASE_GROUPS.values() for c in cats]

ALL_KNOWN_CATEGORIES = set(TOP_LEVEL_PHASES + DETAIL_PHASES + ALL_CPP_PHASES)

TABLE_WIDTH = 120
BAR_CHART_WIDTH = 50


def as_percentage(ms: float, total: float) -> float:
    """Return ms as a percentage of total, or 0 if total is non-positive."""
    return (ms / total * 100) if total > 0 else 0


@dataclass
class TimingRecord:
    category: str
    duration_ms: float
    context: Optional[Dict[str, str]] = None


@dataclass
class ProblemTiming:
    context: Dict[str, str]
    cpu_data_init_ms: float = 0.0
    cpu_reference_gemm_ms: float = 0.0
    gpu_kernel_execution_ms: float = 0.0


@dataclass
class PhaseNode:
    """A node in the timing hierarchy tree."""
    name: str
    count: Optional[int]
    total_ms: float
    mean_ms: Optional[float]
    children: List['PhaseNode']


def build_hierarchy(timings: Dict[str, List[float]]) -> List[PhaseNode]:
    """Build the phase hierarchy from raw timings, sorted by total time descending."""
    top_nodes = []
    for phase in TOP_LEVEL_PHASES:
        if phase not in timings:
            continue
        values = timings[phase]
        count = len(values)
        total_ms = sum(values)
        mean_ms = total_ms / count if count > 0 else 0
        children = []

        if phase != DETAIL_PARENT:
            top_nodes.append(PhaseNode(phase, count, total_ms, mean_ms, children))
            continue

        detail_sum = 0.0
        for detail_phase in DETAIL_PHASES:
            if detail_phase not in timings:
                continue
            d_values = timings[detail_phase]
            d_count = len(d_values)
            d_total = sum(d_values)
            d_mean = d_total / d_count if d_count > 0 else 0
            detail_sum += d_total

            cpp_children: List[PhaseNode] = []
            if detail_phase == CPP_PARENT and d_total > 0:
                cpp_sum = 0.0
                for group_name, group_cats in CPP_PHASE_GROUPS.items():
                    g_total = 0.0
                    g_items: List[PhaseNode] = []
                    for cat in group_cats:
                        if cat in timings:
                            c_values = timings[cat]
                            c_count = len(c_values)
                            c_total = sum(c_values)
                            c_mean = c_total / c_count if c_count > 0 else 0
                            g_total += c_total
                            g_items.append(PhaseNode(cat, c_count, c_total, c_mean, []))
                    if g_items:
                        cpp_sum += g_total
                        g_items.sort(key=lambda n: n.total_ms, reverse=True)
                        cpp_children.append(PhaseNode(group_name, None, g_total, None, g_items))

                cpp_untracked = d_total - cpp_sum
                if cpp_untracked > 0.01:
                    cpp_children.append(PhaseNode("(untracked C++ overhead)", None, cpp_untracked, None, []))
                cpp_children.sort(key=lambda n: n.total_ms, reverse=True)

            children.append(PhaseNode(detail_phase, d_count, d_total, d_mean, cpp_children))

        detail_untracked = total_ms - detail_sum
        if detail_untracked > 0.01:
            children.append(PhaseNode("(untracked overhead)", None, detail_untracked, None, []))
        children.sort(key=lambda n: n.total_ms, reverse=True)

        top_nodes.append(PhaseNode(phase, count, total_ms, mean_ms, children))

    top_nodes.sort(key=lambda n: n.total_ms, reverse=True)
    return top_nodes


def parse_timing_line(line: str) -> Optional[TimingRecord]:
    """Parse a TIMING: line and return a TimingRecord."""
    start_sentinel = 'TIMING:'
    if not line.startswith(start_sentinel):
        return None
    parts = line.split(':', 2)
    if len(parts) != 3:
        return None
    try:
        return TimingRecord(category=parts[1], duration_ms=float(parts[2]))
    except ValueError:
        return None


def parse_context_line(line: str) -> Optional[Dict[str, str]]:
    """Parse a TIMING_CONTEXT: line and return context dict."""
    start_sentinel = 'TIMING_CONTEXT:'
    if not line.startswith(start_sentinel):
        return None
    payload = line[len(start_sentinel):]
    context = {}
    for part in payload.split(','):
        if '=' in part:
            key, value = part.split('=', 1)
            context[key] = value
    return context


def analyze_timing_file(filepath: str) -> Tuple[Dict[str, List[float]], List[ProblemTiming]]:
    """Analyze a file containing timing output and return categorized timings."""
    timings = defaultdict(list)
    problem_timings = []
    current_problem = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()

            context = parse_context_line(line)
            if context:
                if current_problem:
                    problem_timings.append(current_problem)
                current_problem = ProblemTiming(context=context)
                continue

            record = parse_timing_line(line)
            if record:
                timings[record.category].append(record.duration_ms)

                if current_problem:
                    if record.category == 'cpu_data_init':
                        current_problem.cpu_data_init_ms = record.duration_ms
                    elif record.category == 'cpu_reference_gemm':
                        current_problem.cpu_reference_gemm_ms = record.duration_ms
                    elif record.category == 'gpu_kernel_execution':
                        current_problem.gpu_kernel_execution_ms = record.duration_ms

    if current_problem:
        problem_timings.append(current_problem)

    return timings, problem_timings


# ---------------------------------------------------------------------------
# Hierarchical visual bar chart
# ---------------------------------------------------------------------------

def print_visual_breakdown(nodes: List[PhaseNode], wall_clock_ms: float):
    """Print a hierarchical bar chart reflecting the timing hierarchy."""
    if wall_clock_ms <= 0:
        return

    print("TIME BREAKDOWN (visual, % of wall clock)")
    print("-" * TABLE_WIDTH)

    label_width = 42

    def bar_line(indent: int, name: str, ms: float):
        pct = ms / wall_clock_ms * 100
        prefix = "  " * indent
        label = f"  {prefix}{name}"
        bar_len = int(pct / 100 * BAR_CHART_WIDTH)
        bar = "#" * bar_len
        print(f"{label:<{label_width}} {pct:>6.1f}% {bar}")

    for node in nodes:
        bar_line(0, node.name, node.total_ms)
        for child in node.children:
            bar_line(1, child.name, child.total_ms)
            for group in child.children:
                bar_line(2, group.name, group.total_ms)

    print()


# ---------------------------------------------------------------------------
# Hierarchy validation
# ---------------------------------------------------------------------------

def print_hierarchy_validation(totals: Dict[str, float]):
    """Check and print hierarchy invariant violations."""
    warnings = []

    pbp_ms = totals.get(DETAIL_PARENT, 0)
    if pbp_ms > 0:
        detail_sum = sum(totals.get(p, 0) for p in DETAIL_PHASES)
        if detail_sum > pbp_ms:
            excess = detail_sum - pbp_ms
            warnings.append(
                f"Detail phases ({detail_sum:,.2f} ms) exceed "
                f"{DETAIL_PARENT} ({pbp_ms:,.2f} ms) "
                f"by {excess:,.2f} ms ({excess / pbp_ms * 100:.1f}%)"
            )

    ce_ms = totals.get(CPP_PARENT, 0)
    if ce_ms > 0:
        cpp_sum = sum(totals.get(c, 0) for c in ALL_CPP_PHASES)
        if cpp_sum > ce_ms:
            excess = cpp_sum - ce_ms
            warnings.append(
                f"C++ phases ({cpp_sum:,.2f} ms) exceed "
                f"{CPP_PARENT} ({ce_ms:,.2f} ms) "
                f"by {excess:,.2f} ms ({excess / ce_ms * 100:.1f}%)"
            )

    if warnings:
        print("HIERARCHY VALIDATION WARNINGS")
        print("-" * TABLE_WIDTH)
        for w in warnings:
            print(f"  WARNING: {w}")
    else:
        print("HIERARCHY VALIDATION: OK (all child phases fit within their parents)")


# ---------------------------------------------------------------------------
# Summary output
# ---------------------------------------------------------------------------

def print_summary(timings: Dict[str, List[float]], problem_timings: List[ProblemTiming]):
    """Print a summary of timing data."""
    print("=" * TABLE_WIDTH)
    print("TIMING ANALYSIS SUMMARY")
    print("=" * TABLE_WIDTH)
    print()

    if not timings:
        print("No timing data found!")
        print()
        print('Make sure you ran with --global-parameters "TimingInstrumentation=True"')
        print("and captured stderr output (2>&1).")
        return

    total_time_by_category = {cat: sum(vals) for cat, vals in timings.items()}

    wall_clock_ms = sum(total_time_by_category.get(c, 0) for c in TOP_LEVEL_PHASES)
    if wall_clock_ms == 0:
        wall_clock_ms = sum(total_time_by_category.values())
    wall_clock_s = wall_clock_ms / 1000.0

    # -- Hierarchical table --------------------------------------------------

    COL_CAT = 44
    COL_CNT = 8
    COL_TOT = 14
    COL_MEAN = 14
    COL_WALL = 11
    COL_PAR = 11

    def fmt_row(indent, name, count, total_ms, mean_ms, pct_wall, pct_parent):
        prefix = "  " * indent
        cat = f"  {prefix}{name}"
        c = f"{count:>{COL_CNT},}" if count is not None else " " * COL_CNT
        t = f"{total_ms:>{COL_TOT},.2f}" if total_ms is not None else " " * COL_TOT
        m = f"{mean_ms:>{COL_MEAN},.2f}" if mean_ms is not None else " " * COL_MEAN
        w = f"{pct_wall:>{COL_WALL - 1}.1f}%" if pct_wall is not None else " " * COL_WALL
        p = f"{pct_parent:>{COL_PAR - 1}.1f}%" if pct_parent is not None else " " * COL_PAR
        print(f"{cat:<{COL_CAT}} {c} {t} {m} {w}   {p}")

    print("TIMING BY PHASE")
    print("-" * TABLE_WIDTH)
    hdr_cat = "Category"
    print(
        f"  {hdr_cat:<{COL_CAT - 2}}"
        f" {'Count':>{COL_CNT}}"
        f" {'Total (ms)':>{COL_TOT}}"
        f" {'Mean (ms)':>{COL_MEAN}}"
        f" {'% of Wall':>{COL_WALL}}"
        f"   {'% of Parent':>{COL_PAR}}"
    )
    print("-" * TABLE_WIDTH)

    def sep_top():
        indent = "  "  # level 0
        print(indent + "=" * (TABLE_WIDTH - len(indent)))

    def sep_detail():
        indent = "    "  # level 1
        print(indent + "-" * (TABLE_WIDTH - len(indent)))

    def sep_group():
        indent = "      "  # level 2
        w = TABLE_WIDTH - len(indent)
        print(indent + "- " * (w // 2))

    def sep_minor():
        indent = "        "  # level 3
        w = TABLE_WIDTH - len(indent)
        print(indent + ". " * (w // 2))

    nodes = build_hierarchy(timings)

    for top_idx, node in enumerate(nodes):
        if top_idx > 0:
            sep_top()
        fmt_row(0, node.name, node.count, node.total_ms, node.mean_ms,
                as_percentage(node.total_ms, wall_clock_ms), None)

        for child in node.children:
            sep_detail()
            fmt_row(1, child.name, child.count, child.total_ms, child.mean_ms,
                    as_percentage(child.total_ms, wall_clock_ms),
                    as_percentage(child.total_ms, node.total_ms))

            for group in child.children:
                sep_group()
                fmt_row(2, group.name, None, group.total_ms, None,
                        as_percentage(group.total_ms, wall_clock_ms),
                        as_percentage(group.total_ms, child.total_ms))
                if not group.children:
                    continue
                sep_minor()
                for leaf in group.children:
                    fmt_row(3, leaf.name, leaf.count, leaf.total_ms, leaf.mean_ms,
                            as_percentage(leaf.total_ms, wall_clock_ms),
                            as_percentage(leaf.total_ms, group.total_ms))

    # Unknown categories
    unknown_cats = set(timings.keys()) - ALL_KNOWN_CATEGORIES
    if unknown_cats:
        print()
        print("  Unknown Categories")
        print(f"  {'-' * (TABLE_WIDTH - 2)}")
        for cat in sorted(unknown_cats):
            values = timings[cat]
            count = len(values)
            total_ms = sum(values)
            mean_ms = total_ms / count if count > 0 else 0
            fmt_row(0, cat, count, total_ms, mean_ms, as_percentage(total_ms, wall_clock_ms), None)

    print()
    print("-" * TABLE_WIDTH)
    print(f"  ESTIMATED WALL CLOCK TIME: {wall_clock_s:.2f}s")
    print()

    # -- Hierarchy validation -------------------------------------------------
    print_hierarchy_validation(total_time_by_category)
    print()

    # -- Visual breakdown -----------------------------------------------------
    print_visual_breakdown(nodes, wall_clock_ms)

    # -- Key insights ---------------------------------------------------------
    python_codegen_time = (
        total_time_by_category.get("python_solution_generation", 0)
        + total_time_by_category.get("python_kernel_compilation", 0)
    )
    client_setup_time = sum(
        total_time_by_category.get(c, 0)
        for c in CPP_PHASE_GROUPS.get("C++ Client Setup", [])
    )
    data_prep_time = sum(
        total_time_by_category.get(c, 0)
        for c in CPP_PHASE_GROUPS.get("Data Preparation", [])
    )
    ref_time = sum(
        total_time_by_category.get(c, 0)
        for c in CPP_PHASE_GROUPS.get("Reference Computation", [])
    )
    kernel_time = sum(
        total_time_by_category.get(c, 0)
        for c in CPP_PHASE_GROUPS.get("Kernel Execution", [])
    )

    print("KEY INSIGHTS (% of wall clock)")
    print("-" * TABLE_WIDTH)
    print(f"  Python phases (code gen, compilation):  {python_codegen_time / 1000:.2f}s ({as_percentage(python_codegen_time, wall_clock_ms):.1f}%)")
    print(f"  C++ client setup (library loading):     {client_setup_time / 1000:.2f}s ({as_percentage(client_setup_time, wall_clock_ms):.1f}%)")
    print(f"  Data preparation (CPU init, GPU prep):  {data_prep_time / 1000:.2f}s ({as_percentage(data_prep_time, wall_clock_ms):.1f}%)")
    print(f"  Reference computation (CPU GEMM):       {ref_time / 1000:.2f}s ({as_percentage(ref_time, wall_clock_ms):.1f}%)")
    print(f"  Kernel execution (warmup, benchmark):   {kernel_time / 1000:.2f}s ({as_percentage(kernel_time, wall_clock_ms):.1f}%)")
    print()

    cpu_ref_time = total_time_by_category.get('cpu_reference_gemm', 0)
    gpu_exec_time = total_time_by_category.get('gpu_kernel_execution', 0)
    if cpu_ref_time > 0 and gpu_exec_time > 0:
        ratio = cpu_ref_time / gpu_exec_time
        print(f"  CPU reference / GPU execution ratio: {ratio:.1f}x")
        if ratio > 10:
            print("  >>> CPU reference > 10x GPU execution time!")
    print()

    # -- Top 10 slowest problems ----------------------------------------------
    if problem_timings:
        print("TOP 10 SLOWEST PROBLEMS (by CPU reference time)")
        print("-" * TABLE_WIDTH)
        sorted_problems = sorted(
            problem_timings,
            key=lambda p: p.cpu_reference_gemm_ms,
            reverse=True,
        )[:10]

        print(
            f"  {'M':>8} {'N':>8} {'K':>8} {'Batch':>8}"
            f" {'TypeA':>10} {'TypeD':>10}"
            f" {'CPU Ref (ms)':>12} {'GPU (ms)':>10}"
        )
        print(f"  {'-' * (TABLE_WIDTH - 2)}")
        for p in sorted_problems:
            ctx = p.context
            print(
                f"  {ctx.get('M', '?'):>8} {ctx.get('N', '?'):>8}"
                f" {ctx.get('K', '?'):>8} {ctx.get('batch', '?'):>8}"
                f" {ctx.get('typeA', '?'):>10} {ctx.get('typeD', '?'):>10}"
                f" {p.cpu_reference_gemm_ms:>12.2f}"
                f" {p.gpu_kernel_execution_ms:>10.2f}"
            )
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze timing instrumentation output from tensilelite-client',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('logfile', help='Log file containing timing output')
    parser.add_argument('--csv', help='Output CSV file for detailed data')

    args = parser.parse_args()

    timings, problem_timings = analyze_timing_file(args.logfile)
    print_summary(timings, problem_timings)

    if args.csv:
        with open(args.csv, 'w') as f:
            f.write("M,N,K,batch,typeA,typeD,cpu_data_init_ms,cpu_reference_gemm_ms,gpu_kernel_execution_ms\n")
            for p in problem_timings:
                ctx = p.context
                f.write(
                    f"{ctx.get('M', '')},{ctx.get('N', '')},{ctx.get('K', '')},"
                    f"{ctx.get('batch', '')},{ctx.get('typeA', '')},{ctx.get('typeD', '')},"
                    f"{p.cpu_data_init_ms},{p.cpu_reference_gemm_ms},{p.gpu_kernel_execution_ms}\n"
                )
        print(f"Detailed data written to: {args.csv}")


if __name__ == '__main__':
    main()
