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

Timing categories form a hierarchy: parents contain their children's wall-clock
    time, so only top-level phases should be summed to get the total.  The
    TIMING_HIERARCHY dict (below) is the single source of truth for the nesting
    structure.  Display grouping (e.g. grouping C++ phases into "C++ Client
    Setup", "Data Preparation", etc.) is a rendering concern handled by
    DISPLAY_GROUPS / CPP_PHASE_GROUPS and applied at print time only.
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

# --- Hierarchy definition (single source of truth) ---
#
# Each key is a timing category; its value is a dict of child categories.
# An empty dict {} means the category has no known sub-phases.
# The tree structure mirrors the docstring hierarchy diagram above.

TIMING_HIERARCHY = {
    "python_benchmark_problems": {
        "python_cache_check": {},
        "python_solution_generation": {
            "python_solgen_fork_permutations": {},
            "python_solgen_forked_solutions": {},
            "python_solgen_custom_kernels": {},
        },
        "python_kernel_compilation": {
            "python_kernel_bench_setup": {},
            "python_kernel_setup": {},
            "python_kernel_codegen": {},
            "python_kernel_validate": {},
            "python_kernel_write_assemble": {},
            "python_kernel_write_helpers": {},
            "python_kernel_build_co": {},
            "python_kernel_build_src_co": {
                "python_kernel_build_src_co.setup": {},
                "python_kernel_build_src_co.cache_check": {},
                "python_kernel_build_src_co.cache_hit": {},
                "python_kernel_build_src_co.compile": {},
                "python_kernel_build_src_co.unbundle": {},
                "python_kernel_build_src_co.move": {},
                "python_kernel_build_src_co.cache_populate": {},
            },
            "python_kernel_bench_postprocess": {
                "python_benchpost_naming": {},
                "python_benchpost_lib_construction": {},
                "python_benchpost_library_write": {},
                "python_benchpost_client_config": {},
            },
        },
        "python_write_cache": {},
        "python_solution_indexing": {},
        "python_write_solutions": {
            "python_wsol_prepare": {
                "python_wsol_prepare_cache": {},
                "python_wsol_prepare_nocache": {},
            },
        },
        "python_client_execution": {
            "hip_initialization": {},
            "library_loading": {},
            "code_object_loading": {},
            "lazy_loading_init": {},
            "data_init_setup": {},
            "solution_iterator_setup": {},
            "listener_setup": {},
            "reporter_setup": {},
            "pre_problem": {
                "cpu_data_init": {},
                "cpu_reference_gemm": {
                    "solve_cpu_fast": {},
                    "solve_cpu_slow": {},
                },
            },
            "validate_warmups": {
                "validate_gpu_sync": {},
                "validate_gpu_readback": {},
                "validate_element_comparison": {},
            },
            "gpu_input_preparation": {},
            "gpu_input_reset": {},
            "rotating_buffer_preparation": {},
            "solution_selection": {},
            "pre_solution": {},
            "kernel_solving": {},
            "warmup_runs": {},
            "benchmark_runs": {},
            "gpu_kernel_execution": {},
            "post_solution": {
                "post_solution_perf_calc": {},
                "post_solution_reporting": {},
                "post_solution_validation": {},
                "post_solution_log": {},
                "post_solution_result_file": {},
                "post_solution_lib_update": {},
                "post_solution_perf_reset": {},
                "post_solution_sol_advance": {},
            },
            "post_problem": {},
            "finalize_report": {},
        },
    },
    "python_library_logic": {
        "python_logic_parse_solutions": {},
        "python_logic_analyze": {},
        "python_logic_write": {},
    },
    "python_client_writer": {},
}

# Display grouping for C++ phases (rendering concern only).
# Under python_client_execution, children are grouped into these display
# categories instead of being shown individually.
CPP_PHASE_GROUPS = {
    "C++ Client Setup": [
        "hip_initialization",
        "library_loading",
        "code_object_loading",
        "lazy_loading_init",
        "data_init_setup",
        "solution_iterator_setup",
        "listener_setup",
        "reporter_setup",
    ],
    "Data Preparation": [
        "pre_problem",
        "gpu_input_preparation",
        "gpu_input_reset",
        "rotating_buffer_preparation",
    ],
    "Reference Computation": [
        "validate_warmups",
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

# Nodes whose children should be grouped for display
DISPLAY_GROUPS = {
    "python_client_execution": CPP_PHASE_GROUPS,
}

# --- Derived constants ---

def _walk_hierarchy(d):
    """Yield all category names from a nested hierarchy dict."""
    for k, v in d.items():
        yield k
        yield from _walk_hierarchy(v)

ALL_KNOWN_CATEGORIES = set(_walk_hierarchy(TIMING_HIERARCHY))

# Named constants for commonly-referenced hierarchy points (used in tests)
SOLUTION_GENERATION_PARENT = "python_solution_generation"
BENCH_POSTPROCESS_PARENT = "python_kernel_bench_postprocess"
SOLUTION_GENERATION_PHASES = list(
    TIMING_HIERARCHY["python_benchmark_problems"]["python_solution_generation"].keys()
)
BENCH_POSTPROCESS_PHASES = list(
    TIMING_HIERARCHY["python_benchmark_problems"]["python_kernel_compilation"]
    ["python_kernel_bench_postprocess"].keys()
)

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


def _build_node(name: str, children_dict: Dict, timings: Dict[str, List[float]]) -> Optional['PhaseNode']:
    """Build a PhaseNode recursively from the hierarchy dict.

    Args:
        name: The timing category name.
        children_dict: Dict of {child_name: grandchildren_dict} from TIMING_HIERARCHY.
        timings: Raw timing data from analyze_timing_file.
    """
    if name not in timings:
        return None
    values = timings[name]
    count = len(values)
    total_ms = sum(values)
    mean_ms = total_ms / count if count > 0 else 0

    children = []
    if children_dict:
        child_sum = 0.0
        for child_name, grandchildren in children_dict.items():
            child_node = _build_node(child_name, grandchildren, timings)
            if child_node:
                children.append(child_node)
                child_sum += child_node.total_ms
        untracked = total_ms - child_sum
        if untracked > 0.01:
            children.append(PhaseNode("(untracked)", None, untracked, None, []))
        children.sort(key=lambda n: n.total_ms, reverse=True)

    return PhaseNode(name, count, total_ms, mean_ms, children)


def _group_children_for_display(
    children: List[PhaseNode], parent_total: float,
    groups: Dict[str, List[str]]
) -> List[PhaseNode]:
    """Group PhaseNode children into display categories for rendering."""
    grouped = []
    grouped_sum = 0.0
    cat_to_node = {c.name: c for c in children if c.name != "(untracked)"}

    for group_name, group_cats in groups.items():
        g_children = [cat_to_node[c] for c in group_cats if c in cat_to_node]
        if g_children:
            g_total = sum(c.total_ms for c in g_children)
            grouped_sum += g_total
            g_children.sort(key=lambda n: n.total_ms, reverse=True)
            grouped.append(PhaseNode(group_name, None, g_total, None, g_children))

    untracked = parent_total - grouped_sum
    if untracked > 0.01:
        grouped.append(PhaseNode("(untracked C++ overhead)", None, untracked, None, []))
    grouped.sort(key=lambda n: n.total_ms, reverse=True)

    return grouped


def _get_display_children(node: PhaseNode) -> List[PhaseNode]:
    """Return display-grouped children for a node, if applicable."""
    if node.name in DISPLAY_GROUPS:
        return _group_children_for_display(node.children, node.total_ms, DISPLAY_GROUPS[node.name])
    return node.children


def build_hierarchy(timings: Dict[str, List[float]]) -> List[PhaseNode]:
    """Build the phase hierarchy from raw timings, sorted by total time descending."""
    top_nodes = []
    for phase, children_dict in TIMING_HIERARCHY.items():
        node = _build_node(phase, children_dict, timings)
        if node:
            top_nodes.append(node)
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

    def render_visual(node: PhaseNode, depth: int):
        bar_line(depth, node.name, node.total_ms)
        for child in _get_display_children(node):
            render_visual(child, depth + 1)

    for node in nodes:
        render_visual(node, 0)

    print()


# ---------------------------------------------------------------------------
# Hierarchy validation
# ---------------------------------------------------------------------------

def print_hierarchy_validation(totals: Dict[str, float]):
    """Check and print hierarchy invariant violations.

    Recursively walks TIMING_HIERARCHY and checks that for every parent-child
    relationship, the sum of child durations does not exceed the parent.
    """
    warnings = []

    def _check(parent_name: str, children_dict: Dict):
        parent_ms = totals.get(parent_name, 0)
        if parent_ms <= 0 or not children_dict:
            return
        child_sum = sum(totals.get(c, 0) for c in children_dict)
        if child_sum > parent_ms:
            excess = child_sum - parent_ms
            warnings.append(
                f"Child phases of {parent_name} ({child_sum:,.2f} ms) exceed "
                f"parent ({parent_ms:,.2f} ms) "
                f"by {excess:,.2f} ms ({excess / parent_ms * 100:.1f}%)"
            )
        for child_name, grandchildren in children_dict.items():
            if grandchildren:
                _check(child_name, grandchildren)

    for parent_name, children_dict in TIMING_HIERARCHY.items():
        _check(parent_name, children_dict)

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

    wall_clock_ms = sum(total_time_by_category.get(c, 0) for c in TIMING_HIERARCHY)
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

    def fmt_row(indent, name, count, total_ms, mean_ms, pct_parent, pct_wall):
        prefix = "  " * indent
        cat = f"  {prefix}{name}"
        c = f"{count:>{COL_CNT},}" if count is not None else " " * COL_CNT
        t = f"{total_ms:>{COL_TOT},.2f}" if total_ms is not None else " " * COL_TOT
        m = f"{mean_ms:>{COL_MEAN},.2f}" if mean_ms is not None else " " * COL_MEAN
        par = f"{pct_parent:>{COL_WALL - 1}.1f}%" if pct_parent is not None else " " * COL_WALL
        wall = f"{pct_wall:>{COL_PAR - 1}.1f}%" if pct_wall is not None else " " * COL_PAR
        print(f"{cat:<{COL_CAT}} {c} {t} {m} {par}   {wall}")

    print("TIMING BY PHASE")
    print("-" * TABLE_WIDTH)
    hdr_cat = "Category"
    print(
        f"  {hdr_cat:<{COL_CAT - 2}}"
        f" {'Count':>{COL_CNT}}"
        f" {'Total (ms)':>{COL_TOT}}"
        f" {'Mean (ms)':>{COL_MEAN}}"
        f" {'% of Parent':>{COL_WALL}}"
        f"   {'% of Wall':>{COL_PAR}}"
    )
    print("-" * TABLE_WIDTH)

    SEPARATORS = {
        0: lambda: None,  # no separator before top-level (handled specially)
        1: lambda: print("    " + "-" * (TABLE_WIDTH - 4)),
        2: lambda: print("      " + "- " * ((TABLE_WIDTH - 6) // 2)),
        3: lambda: print("        " + ". " * ((TABLE_WIDTH - 8) // 2)),
    }

    def sep_top():
        print("  " + "=" * (TABLE_WIDTH - 2))

    def render_node(node: PhaseNode, depth: int, parent_ms: Optional[float],
                    wall_clock_ms: float, is_first: bool):
        """Recursively render a PhaseNode at a given depth."""
        # Separators between siblings
        if depth == 0:
            pass  # handled by caller
        elif depth == 1:
            SEPARATORS[1]()
        elif depth == 2:
            SEPARATORS[2]()
        elif depth == 3 and is_first:
            SEPARATORS[3]()
        # depth >= 4 or depth == 3 non-first: no separator

        pct_parent = as_percentage(node.total_ms, parent_ms) if parent_ms else None
        pct_wall = as_percentage(node.total_ms, wall_clock_ms)

        fmt_row(depth, node.name, node.count, node.total_ms, node.mean_ms,
                pct_parent, pct_wall)

        for i, child in enumerate(_get_display_children(node)):
            render_node(child, depth + 1, node.total_ms, wall_clock_ms, is_first=(i == 0))

    nodes = build_hierarchy(timings)

    for top_idx, node in enumerate(nodes):
        if top_idx > 0:
            sep_top()
        pct_wall = as_percentage(node.total_ms, wall_clock_ms)
        fmt_row(0, node.name, node.count, node.total_ms, node.mean_ms, None, pct_wall)

        for i, child in enumerate(node.children):
            render_node(child, 1, node.total_ms, wall_clock_ms, is_first=(i == 0))

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
    python_io_time = (
        total_time_by_category.get("python_write_cache", 0)
        + total_time_by_category.get("python_write_solutions", 0)
        + total_time_by_category.get("python_cache_check", 0)
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
    print(f"  Python I/O overhead (cache, YAML):      {python_io_time / 1000:.2f}s ({as_percentage(python_io_time, wall_clock_ms):.1f}%)")
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
