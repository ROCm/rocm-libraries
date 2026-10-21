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

By default the report regroups the same data by purpose (PURPOSE_OF / build_purpose_hierarchy):
for example all of the CPU reference GEMM is one group, wherever it is called from, and the groups
add up to the wall clock time. ``--legacy`` prints the tree as the timers are nested instead.
"""

import argparse
import gzip
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
        # DuctileBackend (GA): evaluation pieces outside benchmark_runner, plus the GA's own time.
        "python_ductile_setup": {},
        "python_ductile_generate_solutions": {},
        "python_ductile_clear_artifacts": {},
        "python_ductile_read_scores": {},
        "python_ductile_initial_sampling": {},
        "python_ductile_ga_between_evaluations": {},
        "python_ductile_verification": {},
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
            "python_wsol_header": {},
            "python_wsol_dump": {},
        },
        "python_client_execution": {
            "python_client_write_run_script": {},
            "process_startup": {},
            "hip_initialization": {},
            "library_loading": {},
            "code_object_loading": {},
            "lazy_loading_init": {},
            "data_init_setup": {},
            "solution_iterator_setup": {},
            "listener_setup": {},
            "reporter_setup": {},
            "main_parse_args": {},
            "main_timing_calibration": {},
            "main_problem_factory": {
                "factory_hip_get_device_properties": {},
            },
            "pre_problem": {
                "cpu_data_init": {},
                "cpu_reference_gemm": {
                    "solve_cpu_fast": {},
                    "solve_cpu_slow": {},
                },
            },
            "validate_warmups": {
                # validate_warmups runs every listener's validateWarmups(); one row per listener.
                "validate_warmups_listener.ReferenceValidator": {
                    "validate_inputs_copy": {},
                    "validate_setup": {},
                    "validate_result_buffer_alloc": {},
                    "validate_gpu_readback": {},
                    "validate_element_comparison": {},
                    "validate_mismatch_printing": {},
                    "validate_report": {},
                },
                "validate_warmups_listener.SynchronizerValidator": {
                    # Per validation: read the Synchronizer buffer back and scan it for residue.
                    # Only solutions that may use the Synchronizer (StreamK, MBSK, GSU partial
                    # reduction, amaxD) do the readback and scan; the rest are "inactive".
                    "validate_synchronizer_check": {
                        "validate_synchronizer_readback": {},
                        "validate_synchronizer_scan": {},
                        "validate_synchronizer_inactive": {},
                        "validate_synchronizer_no_buffer": {},
                    },
                },
                "validate_warmups_listener.BenchmarkTimer": {
                    "validate_gpu_sync": {},
                },
                "validate_warmups_listener.DataInitialization": {},
                "validate_warmups_listener.MetaResultReporter": {},
                "validate_warmups_listener.AllSolutionsIterator": {},
                "validate_warmups_listener.ProgressListener": {},
                "validate_warmups_listener.HardwareMonitorListener": {},
            },
            "gpu_input_preparation": {
                "gpu_input_prep_fast_reset_output": {},
                "gpu_input_prep_slow_path": {},
                "gpu_input_prep_init_cpu_inputs": {},
                "gpu_input_prep_copy_valid_to_gpu": {},
                "gpu_input_prep_copy_swizzled": {},
                "gpu_input_prep_copy_inputs_d2d": {},
                "gpu_input_prep_rotating_init": {},
                "gpu_input_prep_batched_inputs": {},
                "gpu_input_prep_constant_inputs": {},
                "gpu_input_prep_convert_to_inputs": {},
            },
            "gpu_input_reset": {
                "gpu_input_reset_fast_reset_output": {},
                "gpu_input_reset_slow_path": {},
                "gpu_input_reset_init_cpu_inputs": {},
                "gpu_input_reset_copy_valid_to_gpu": {},
                "gpu_input_reset_copy_swizzled": {},
                "gpu_input_reset_copy_inputs_d2d": {},
                "gpu_input_reset_rotating_init": {},
                "gpu_input_reset_batched_inputs": {},
                "gpu_input_reset_constant_inputs": {},
                "gpu_input_reset_convert_to_inputs": {},
            },
            "rotating_buffer_preparation": {},
            "solution_selection": {},
            "pre_solution": {
                "pre_solution_listener.ReferenceValidator": {
                    # DataInitialization's input epoch did not move, so the reference is reused
                    # (count-only marker).
                    "pre_solution_ref_skipped": {},
                    # Per-solution recompute because the input epoch moved (inputs were really
                    # regenerated). "inputs_unchanged" only appears with
                    # --verify-reference-reuse / TENSILELITE_VERIFY_REFERENCE_REUSE=1, which
                    # recomputes anyway to check the reuse is exact; it is the old behaviour's
                    # redundant work and also what older client builds always did.
                    "cpu_reference_gemm_per_solution_inputs_changed": {
                        "solve_cpu_fast_per_solution_inputs_changed": {},
                        "solve_cpu_slow_per_solution_inputs_changed": {},
                    },
                    "cpu_reference_gemm_per_solution_inputs_unchanged": {
                        "solve_cpu_fast_per_solution_inputs_unchanged": {},
                        "solve_cpu_slow_per_solution_inputs_unchanged": {},
                    },
                    # Older logs (before the changed/unchanged split).
                    "cpu_reference_gemm_per_solution": {
                        "solve_cpu_fast_per_solution": {},
                        "solve_cpu_slow_per_solution": {},
                    },
                },
                "pre_solution_listener.DataInitialization": {
                    "pre_solution_mx_check": {},
                    # initOneMXSide runs once per side (A, B) per call: it either reuses its
                    # cache or really regenerates data + scales.
                    "pre_solution_mx_init_data": {
                        "pre_solution_mx_side_cache_hit": {},
                        "pre_solution_mx_side_regenerate": {},
                    },
                    # These run for every gated solution; the suffix says whether the MX data
                    # was actually regenerated in that same call.
                    "pre_solution_mx_copy_valid_to_gpu_after_regen": {},
                    "pre_solution_mx_copy_valid_to_gpu_no_regen": {},
                    "pre_solution_mx_copy_inputs_d2d_after_regen": {},
                    "pre_solution_mx_copy_inputs_d2d_no_regen": {},
                    "pre_solution_mx_cpu_sync_after_regen": {},
                    "pre_solution_mx_cpu_sync_no_regen": {},
                    # Older logs (before the regen split).
                    "pre_solution_mx_copy_valid_to_gpu": {},
                    "pre_solution_mx_copy_inputs_d2d": {},
                    "pre_solution_mx_cpu_sync": {},
                },
                "pre_solution_listener.BenchmarkTimer": {},
                "pre_solution_listener.MetaResultReporter": {},
                "pre_solution_listener.AllSolutionsIterator": {},
                "pre_solution_listener.ProgressListener": {},
                "pre_solution_listener.SynchronizerValidator": {},
                "pre_solution_listener.HardwareMonitorListener": {},
            },
            "run_current_solution_check": {},
            "kernel_solving": {},
            "warmup_events_setup": {},
            "warmup_runs": {},
            "benchmark_runs": {
                "benchmark_pre_syncs": {},
                "benchmark_events_setup": {},
                "gpu_kernel_execution": {},
                # One row per listener's validateEnqueues(). BenchmarkTimer is where the GPU timing
                # is read (event sync / elapsed time) and where --sleep-percent idles on purpose.
                "benchmark_validate_enqueues": {
                    "validate_enqueues_listener.BenchmarkTimer": {
                        "benchmark_event_sync": {},
                        "benchmark_event_elapsed_time": {},
                        "benchmark_event_destroy": {},
                        "benchmark_sleep": {},
                    },
                    "validate_enqueues_listener.ProgressListener": {},
                    "validate_enqueues_listener.SynchronizerValidator": {},
                    "validate_enqueues_listener.ReferenceValidator": {},
                    "validate_enqueues_listener.MetaResultReporter": {},
                    "validate_enqueues_listener.HardwareMonitorListener": {},
                    "validate_enqueues_listener.DataInitialization": {},
                    "validate_enqueues_listener.AllSolutionsIterator": {},
                },
                "benchmark_events_teardown": {},
                "benchmark_post_syncs": {},
            },
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
            "problem_report_index": {},
            "problem_teardown": {},
            "solution_teardown": {},
            "run_iteration_teardown": {},
            "finalize_report": {},
            "timing_overhead": {},
            "flush_timing_buffer": {},
            "main_teardown_locals": {},
            "process_exit": {},
        },
    },
    "python_library_logic": {
        "python_logic_parse_solutions": {},
        "python_logic_analyze": {},
        "python_logic_write": {},
    },
    "python_client_writer": {
        "python_library_client_setup": {},
        # TensileCreateLibrary subprocess that builds the library client's library/code objects.
        "python_library_client_create_library": {},
        "python_library_client_write_configs": {},
        # The library-client launch has its own timer names (it runs without C++ timing).
        "python_library_client_execution": {
            "python_library_client_write_run_script": {},
            "python_library_client_subprocess": {},
        },
    },
}

# Display grouping for C++ phases (rendering concern only).
# Under python_client_execution, children are grouped into these display
# categories instead of being shown individually.
CPP_PHASE_GROUPS = {
    # Everything paid once per client launch, independent of how many problems/solutions run:
    # process startup, in-process setup, and teardown/exit. process_startup, main_teardown_locals
    # and process_exit come from wall-clock stamps (they fall outside the client's own timers).
    "C++ Client Setup & Teardown": [
        "python_client_write_run_script",
        "process_startup",
        "main_parse_args",
        "main_timing_calibration",
        "main_problem_factory",
        "hip_initialization",
        "library_loading",
        "code_object_loading",
        "lazy_loading_init",
        "data_init_setup",
        "solution_iterator_setup",
        "listener_setup",
        "reporter_setup",
        "main_teardown_locals",
        "process_exit",
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
        "run_current_solution_check",
        "kernel_solving",
        "warmup_events_setup",
        "warmup_runs",
        "benchmark_runs",
        "post_solution",
        "post_problem",
        "problem_report_index",
        "problem_teardown",
        "solution_teardown",
        "run_iteration_teardown",
        "finalize_report",
    ],
    "Timing Overhead": [
        "timing_overhead",
        "flush_timing_buffer",
    ],
}

# Nodes whose children should be grouped for display
DISPLAY_GROUPS = {
    "python_client_execution": CPP_PHASE_GROUPS,
}

# --- Purpose view ---------------------------------------------------------------------------
#
# TIMING_HIERARCHY mirrors how the timers are nested in the code, so the same kind of work shows up
# in several places (the CPU reference GEMM is called from pre_problem *and* from pre_solution).
# The default report regroups the very same data by what the time is spent on. Each category is
# assigned to a purpose group below; categories that are not listed inherit the group of the
# category that contains them. The time of every category is shown exactly once, so the groups add
# up to the wall clock time.

G_REFERENCE = "Reference GEMM (CPU)"
G_VALIDATION = "Validation"
G_INPUT = "Input data preparation"
G_KERNEL = "Kernel launch & execution"
G_BOOKKEEPING = "Per-solution bookkeeping & reporting"
G_LAUNCH = "Client launch (startup, setup & teardown)"
G_INSTRUMENTATION = "MEASUREMENT OVERHEAD (instrumentation)"
G_BUILD = "Solution generation & build (Python)"
G_PY_BOOKKEEPING = "SHARED: cache & solution files"
G_DUCTILE = "Ductile GA search (Python)"
G_LIBRARY_CLIENT = "Library client (Python)"
G_LIBRARY_LOGIC = "Library logic (Python)"
G_UNATTRIBUTED = "UNATTRIBUTED (untracked time & timer overhead)"

PURPOSE_OF = {
    # Python side
    "python_benchmark_problems": G_BUILD,
    "python_cache_check": G_PY_BOOKKEEPING,
    "python_write_cache": G_PY_BOOKKEEPING,
    "python_solution_indexing": G_PY_BOOKKEEPING,
    "python_write_solutions": G_PY_BOOKKEEPING,
    "python_ductile_setup": G_DUCTILE,
    "python_ductile_generate_solutions": G_DUCTILE,
    "python_ductile_clear_artifacts": G_DUCTILE,
    "python_ductile_read_scores": G_DUCTILE,
    "python_ductile_initial_sampling": G_DUCTILE,
    "python_ductile_ga_between_evaluations": G_DUCTILE,
    "python_ductile_verification": G_DUCTILE,
    "python_library_logic": G_LIBRARY_LOGIC,
    "python_client_writer": G_LIBRARY_CLIENT,
    # C++ client: one launch's setup and teardown
    "python_client_execution": G_BOOKKEEPING,
    "python_client_write_run_script": G_LAUNCH,
    "process_startup": G_LAUNCH,
    "hip_initialization": G_LAUNCH,
    "library_loading": G_LAUNCH,
    "code_object_loading": G_LAUNCH,
    "lazy_loading_init": G_LAUNCH,
    "data_init_setup": G_LAUNCH,
    "solution_iterator_setup": G_LAUNCH,
    "listener_setup": G_LAUNCH,
    "reporter_setup": G_LAUNCH,
    "main_parse_args": G_LAUNCH,
    "main_problem_factory": G_LAUNCH,
    "main_teardown_locals": G_LAUNCH,
    "process_exit": G_LAUNCH,
    # Cost of the measurement itself
    "main_timing_calibration": G_INSTRUMENTATION,
    "timing_overhead": G_INSTRUMENTATION,
    "flush_timing_buffer": G_INSTRUMENTATION,
    # Reference GEMM, wherever it is called from
    "cpu_reference_gemm": G_REFERENCE,
    "pre_solution_listener.ReferenceValidator": G_REFERENCE,
    # Preparing the inputs
    "pre_problem": G_INPUT,
    "gpu_input_preparation": G_INPUT,
    "gpu_input_reset": G_INPUT,
    "rotating_buffer_preparation": G_INPUT,
    "pre_solution_listener.DataInitialization": G_INPUT,
    # Checking the results
    "validate_warmups": G_VALIDATION,
    # Running the kernels
    "kernel_solving": G_KERNEL,
    "warmup_events_setup": G_KERNEL,
    "warmup_runs": G_KERNEL,
    "benchmark_runs": G_KERNEL,
    "validate_warmups_listener.BenchmarkTimer": G_KERNEL,  # waits for the warmup kernel to finish
    # Everything else that happens around each solution / problem
    "solution_selection": G_BOOKKEEPING,
    "pre_solution": G_BOOKKEEPING,
    "run_current_solution_check": G_BOOKKEEPING,
    "post_solution": G_BOOKKEEPING,
    "post_problem": G_BOOKKEEPING,
    "problem_report_index": G_BOOKKEEPING,
    "problem_teardown": G_BOOKKEEPING,
    "solution_teardown": G_BOOKKEEPING,
    "run_iteration_teardown": G_BOOKKEEPING,
    "finalize_report": G_BOOKKEEPING,
}

# Containers that only hold other timers: they get no row of their own in the purpose view. Their
# children are shown under their own groups, and whatever time is left over (their untracked time)
# goes to the "Unattributed" group.
PURPOSE_TRANSPARENT = {
    "python_benchmark_problems",
    "python_client_execution",
    "python_client_writer",
    "python_library_logic",
    "pre_problem",
    "pre_solution",
    "post_solution",
    "validate_warmups",
}

# The groups are gathered into stages. A stage is only shown if it has time, so the same report
# works for a build-only log (BUILD), a run-only log (RUN, with --use-cache) and for both together.
# Groups that are not part of a stage (cache and solution-file bookkeeping, which happens in both
# stages, the measurement overhead and the unattributed time) are shown next to the stages.
STAGES = [
    ("RUN: benchmark client",
     [G_REFERENCE, G_INPUT, G_VALIDATION, G_KERNEL, G_BOOKKEEPING, G_LAUNCH,
      G_LIBRARY_CLIENT, G_LIBRARY_LOGIC]),
    ("BUILD: solution generation & compile", [G_BUILD, G_DUCTILE]),
]
ROOT_NAME = "TOTAL RUNTIME"

# --- Derived constants ---

def _walk_hierarchy(d):
    """Yield all category names from a nested hierarchy dict."""
    for k, v in d.items():
        yield k
        yield from _walk_hierarchy(v)

ALL_KNOWN_CATEGORIES = set(_walk_hierarchy(TIMING_HIERARCHY))

def _categories_with_children(d, out=None):
    """Names of the categories that have known sub-timers (only those can have untracked time)."""
    out = set() if out is None else out
    for name, children in d.items():
        if children:
            out.add(name)
            _categories_with_children(children, out)
    return out

CATEGORIES_WITH_CHILDREN = _categories_with_children(TIMING_HIERARCHY)

# Timers that wrap other timers (diagnostics added to localize untracked time).
# python_client_subprocess = process_startup + main_total + flush + main_teardown_locals + process_exit
# main_total              = parse_args + calibration + factory + setup + problem_total + finalize
# problem_total           = problem children + solution_total
# solution_total          = solution children + run_iteration_total
OVERLAPPING_TOTALS_ORDER = [
    "python_client_subprocess",
    "main_total",
    "problem_total",
    "solution_total",
    "run_iteration_total",
    # ReferenceValidator's validateSolution(): wraps the per-tensor timers below it.
    "validate_total",
    "validate_check_results_typed",
]
OVERLAPPING_TOTALS = set(OVERLAPPING_TOTALS_ORDER)

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

# Minimum table width; print_summary widens TABLE_WIDTH when category labels need more room.
TABLE_WIDTH_MIN = 120
TABLE_WIDTH = TABLE_WIDTH_MIN
BAR_CHART_WIDTH = 50


def as_percentage(ms: float, total: float) -> float:
    """Return ms as a percentage of total, or 0 if total is non-positive."""
    return (ms / total * 100) if total > 0 else 0


# Fixed numeric columns of the phase table (see print_summary).
COL_CAT_MIN = 44
COL_CNT = 8
COL_TOT = 14
COL_MEAN = 14
COL_PCT_PARENT = 11
COL_PCT_WALL = 11


def _label_len(depth: int, name: str) -> int:
    """Width of a row label as rendered: 2 leading spaces + 2 per depth level + name."""
    return 2 + 2 * depth + len(name)


def _max_label_len(nodes: List['PhaseNode'], extra_names: List[str]) -> int:
    """Longest row label over every tree row (plain and display-grouped) and extra_names."""
    longest = max((_label_len(0, n) for n in extra_names), default=0)

    def walk(node: 'PhaseNode', depth: int):
        nonlocal longest
        longest = max(longest, _label_len(depth, node.name))
        for child in list(node.children) + list(_get_display_children(node)):
            walk(child, depth + 1)

    for node in nodes:
        walk(node, 0)
    return longest


def _table_layout(nodes: List['PhaseNode'], extra_rows: List[Tuple[str, int, float, float, float]],
                  wall_clock_ms: float) -> Dict[str, int]:
    """Return the column widths of the phase table, sized from the data.

    The category column fits the longest label and each numeric column fits its widest value
    (formatted exactly as ``fmt_row`` does), so large counts or totals, e.g. when several runs
    are summed, cannot push the columns out of line. ``extra_rows`` are the unknown /
    overlapping-total rows as ``(name, count, total_ms, mean_ms, pct_of_wall)``.
    Keys: cat, cnt, tot, mean, parent, wall, table.
    """
    width = {
        "cnt": max(COL_CNT, len("Count")),
        "tot": max(COL_TOT, len("Total (ms)")),
        "mean": max(COL_MEAN, len("Mean (ms)")),
        "parent": max(COL_PCT_PARENT, len("% of Parent")),
        "wall": max(COL_PCT_WALL, len("% of Wall")),
    }

    def note(count, total_ms, mean_ms, pct_parent, pct_wall):
        if count is not None:
            width["cnt"] = max(width["cnt"], len(f"{count:,}"))
        if total_ms is not None:
            width["tot"] = max(width["tot"], len(f"{total_ms:,.2f}"))
        if mean_ms is not None:
            width["mean"] = max(width["mean"], len(f"{mean_ms:,.2f}"))
        # A percentage is printed as "<value>%", so it needs one more column than its digits.
        if pct_parent is not None:
            width["parent"] = max(width["parent"], len(f"{pct_parent:.1f}") + 1)
        if pct_wall is not None:
            width["wall"] = max(width["wall"], len(f"{pct_wall:.1f}") + 1)

    def walk(node: 'PhaseNode', parent_ms: Optional[float]):
        pct_parent = as_percentage(node.total_ms, parent_ms) if parent_ms else None
        note(node.count, node.total_ms, node.mean_ms, pct_parent,
             as_percentage(node.total_ms, wall_clock_ms))
        # Same two views the renderer walks: plain children and display-grouped children.
        for child in list(node.children) + list(_get_display_children(node)):
            walk(child, node.total_ms)

    for node in nodes:
        walk(node, None)
    for _name, count, total_ms, mean_ms, pct in extra_rows:
        note(count, total_ms, mean_ms, pct, None)

    cat = max(COL_CAT_MIN, _max_label_len(nodes, [r[0] for r in extra_rows]
                                          + ["(untracked C++ overhead)", "Overlapping Totals"]) + 1)
    numeric = 1 + width["cnt"] + 1 + width["tot"] + 1 + width["mean"] + 1 + width["parent"] + 3 \
        + width["wall"]
    width["cat"] = cat
    width["table"] = max(TABLE_WIDTH_MIN, cat + numeric)
    return width


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


# Categories that represent instrumentation overhead measurements, not actual work.
OVERHEAD_CATEGORIES = {'timing_overhead'}


def _count_descendant_invocations(children_dict: Dict, timings: Dict[str, List[float]]) -> int:
    """Count total timer invocations across all descendants in the hierarchy."""
    count = 0
    for child_name, grandchildren in children_dict.items():
        count += len(timings.get(child_name, []))
        count += _count_descendant_invocations(grandchildren, timings)
    return count


def _compute_per_call_overhead(timings: Dict[str, List[float]]) -> float:
    """Compute per-call instrumentation overhead from the C++ timing_overhead record.

    The C++ client emits a timing_overhead record containing the total estimated
    overhead across all ScopedTimer invocations (calibrated at startup).  Dividing
    by total invocations gives the average overhead each timer call adds to its
    parent's measurement.

    Note: Python-side timing_context overhead is not tracked or adjusted here.
    Python has only ~a dozen calls per run, so the overhead is negligible.
    """
    total_overhead_ms = sum(timings.get('timing_overhead', []))
    if total_overhead_ms <= 0:
        return 0.0
    total_invocations = sum(
        len(vals) for cat, vals in timings.items()
        if cat not in OVERHEAD_CATEGORIES
    )
    if total_invocations == 0:
        return 0.0
    return total_overhead_ms / total_invocations


def _build_node(name: str, children_dict: Dict, timings: Dict[str, List[float]],
                per_call_overhead_ms: float = 0) -> Optional['PhaseNode']:
    """Build a PhaseNode recursively from the hierarchy dict.

    Args:
        name: The timing category name.
        children_dict: Dict of {child_name: grandchildren_dict} from TIMING_HIERARCHY.
        timings: Raw timing data from analyze_timing_file.
        per_call_overhead_ms: Per-invocation overhead to subtract from parent totals
            based on descendant invocation count.
    """
    if name not in timings:
        return None
    values = timings[name]
    count = len(values)
    raw_total_ms = sum(values)

    # Subtract estimated overhead from descendant timer invocations.
    # Each descendant invocation adds ~per_call_overhead_ms to this node's
    # raw measurement via context-manager / RAII bookkeeping.
    desc_invocations = _count_descendant_invocations(children_dict, timings)
    total_ms = max(0, raw_total_ms - desc_invocations * per_call_overhead_ms)
    mean_ms = total_ms / count if count > 0 else 0

    children = []
    if children_dict:
        child_sum = 0.0
        for child_name, grandchildren in children_dict.items():
            child_node = _build_node(child_name, grandchildren, timings, per_call_overhead_ms)
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
    """Build the phase hierarchy from raw timings, sorted by total time descending.

    Automatically adjusts parent totals by subtracting estimated per-call
    instrumentation overhead based on descendant invocation counts.
    """
    per_call_overhead_ms = _compute_per_call_overhead(timings)
    top_nodes = []
    for phase, children_dict in TIMING_HIERARCHY.items():
        node = _build_node(phase, children_dict, timings, per_call_overhead_ms)
        if node:
            top_nodes.append(node)
    top_nodes.sort(key=lambda n: n.total_ms, reverse=True)
    return top_nodes


def build_purpose_hierarchy(timings: Dict[str, List[float]]) -> List[PhaseNode]:
    """Regroup the timer hierarchy by purpose (see PURPOSE_OF) and return the top-level groups.

    Starts from ``build_hierarchy`` (so totals are overhead-adjusted exactly as in the legacy
    report) and moves every category under its purpose group. A row nested under a category of
    another group is taken out of that category, whose row then only shows its own remaining time.
    Each category's own time is counted once, so the groups sum to the wall clock time (to within
    the 0.01 ms tolerance used for untracked rows).
    """
    groups: Dict[str, List[PhaseNode]] = {}
    unattributed: List[PhaseNode] = []

    def add_unattributed(label: str, ms: float):
        if abs(ms) > 0.01:
            unattributed.append(PhaseNode(label, None, ms, None, []))

    def build(node: PhaseNode, parent_group: str, nest_into: Optional[List[PhaseNode]]) -> float:
        """Place ``node`` and return the time it contributes to the row it is nested in."""
        group = PURPOSE_OF.get(node.name, parent_group)
        real = [c for c in node.children if c.name != "(untracked)"]
        transparent = node.name in PURPOSE_TRANSPARENT
        nested_rows: List[PhaseNode] = []
        nested_ms = 0.0
        for child in real:
            same_group = PURPOSE_OF.get(child.name, group) == group
            nested_ms += build(child, group, nested_rows if (same_group and not transparent) else None)

        if node.name not in CATEGORIES_WITH_CHILDREN:
            # A timer with no known sub-timers: all of its time is its own.
            row = PhaseNode(node.name, node.count, node.total_ms, node.mean_ms, [])
            if nest_into is not None:
                nest_into.append(row)
                return row.total_ms
            groups.setdefault(group, []).append(row)
            return 0.0

        remainder = node.total_ms - sum(c.total_ms for c in real)
        if transparent:
            add_unattributed(f"(untracked in {node.name})" if remainder >= 0
                             else f"(children exceed {node.name})", remainder)
            return 0.0

        children = sorted(nested_rows, key=lambda n: n.total_ms, reverse=True)
        if remainder > 0.01:
            children.append(PhaseNode("(untracked)", None, remainder, None, []))
            shown = remainder + nested_ms
        else:
            add_unattributed(f"(children exceed {node.name})", remainder)
            shown = nested_ms
        mean = shown / node.count if node.count else None
        row = PhaseNode(node.name, node.count, shown, mean, children)
        if nest_into is not None:
            nest_into.append(row)
            return shown
        groups.setdefault(group, []).append(row)
        return 0.0

    for top in build_hierarchy(timings):
        build(top, PURPOSE_OF.get(top.name, G_UNATTRIBUTED), None)

    if unattributed:
        groups.setdefault(G_UNATTRIBUTED, []).extend(unattributed)

    nodes = []
    for name, rows in groups.items():
        rows.sort(key=lambda n: n.total_ms, reverse=True)
        nodes.append(PhaseNode(name, None, sum(r.total_ms for r in rows), None, rows))
    # Largest first, with the unattributed leftovers always last.
    nodes.sort(key=lambda n: (n.name == G_UNATTRIBUTED, -n.total_ms))
    return nodes


def _wall_clock_ms(timings: Dict[str, List[float]]) -> float:
    """Total runtime: the raw sum of the top-level phases (all categories if none was logged)."""
    totals = {cat: sum(vals) for cat, vals in timings.items()}
    wall = sum(totals.get(c, 0) for c in TIMING_HIERARCHY)
    return wall if wall else sum(totals.values())


def build_staged_hierarchy(timings: Dict[str, List[float]]) -> List[PhaseNode]:
    """The purpose tree under one root, with the groups gathered into stages (see STAGES).

    Returns ``[root]``: TOTAL RUNTIME -> stage rows (RUN, BUILD) and stand-alone groups -> purpose
    groups -> rows. The root is the raw wall clock total. The parent timers' totals are adjusted
    for the overhead of the timers inside them, so the groups add up to slightly less than that;
    the difference is shown as its own row in the measurement overhead group.
    """
    groups = build_purpose_hierarchy(timings)
    if not groups:
        return []
    by_name = {g.name: g for g in groups}

    diff = _wall_clock_ms(timings) - sum(g.total_ms for g in groups)
    if abs(diff) > 0.01:
        name, label = ((G_INSTRUMENTATION, "(timer overhead subtracted from parent timers)")
                       if diff > 0 else (G_UNATTRIBUTED, "(children exceed parents)"))
        target = by_name.setdefault(name, PhaseNode(name, None, 0.0, None, []))
        target.children.append(PhaseNode(label, None, diff, None, []))
        target.children.sort(key=lambda n: n.total_ms, reverse=True)
        target.total_ms += diff

    top: List[PhaseNode] = []
    staged = set()
    for stage_name, stage_groups in STAGES:
        members = sorted((by_name[g] for g in stage_groups if g in by_name),
                         key=lambda n: n.total_ms, reverse=True)
        if members:
            top.append(PhaseNode(stage_name, None, sum(m.total_ms for m in members), None, members))
            staged.update(m.name for m in members)
    top.extend(g for g in by_name.values() if g.name not in staged)
    top.sort(key=lambda n: (n.name == G_UNATTRIBUTED, -n.total_ms))
    return [PhaseNode(ROOT_NAME, None, sum(n.total_ms for n in top), None, top)]


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


def _open_log(filepath: str):
    """Open a timing log as text; ``.gz`` files are decompressed on the fly."""
    if str(filepath).endswith('.gz'):
        return gzip.open(filepath, 'rt', errors='replace')
    return open(filepath, 'r', errors='replace')


def analyze_timing_file(filepath: str) -> Tuple[Dict[str, List[float]], List[ProblemTiming]]:
    """Analyze a file (plain text or ``.gz``) containing timing output and return categorized timings."""
    timings = defaultdict(list)
    problem_timings = []
    current_problem = None
    ts = {}  # wall-clock stamps (ns) of the current client invocation

    with _open_log(filepath) as f:
        for line in f:
            line = line.strip()

            stamp = parse_timestamp_line(line)
            if stamp:
                _handle_timestamp(stamp[0], stamp[1], ts, timings)
                continue

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
# Tiered (dict) representation of a run's timing results
# ---------------------------------------------------------------------------

class TimingSeries(list):
    """A category's timings reduced to (count, total): behaves like the list of durations.

    The report code only ever takes ``len()`` and ``sum()`` of a category's durations, so a
    series that yields its total once and reports its count as its length is
    indistinguishable from the full list there. This lets results that were summed across
    runs go through the unchanged report/hierarchy code.
    """

    def __init__(self, count: int, total_ms: float):
        super().__init__([total_ms])
        self._count = count

    def __len__(self) -> int:
        return self._count


def timings_to_tree(timings: Dict[str, List[float]]) -> dict:
    """Return a run's timing results as a tiered dict.

    Layout::

        {"phases": {<top-level category>: {"count": int, "total_ms": float,
                                            "children": {<child category>: {...}, ...}}},
         "overlapping_totals": {<category>: {"count": int, "total_ms": float}},
         "unknown_categories": {<category>: {"count": int, "total_ms": float}}}

    ``phases`` mirrors TIMING_HIERARCHY. Totals are the raw sums of the logged durations, so trees
    from different runs can be added node by node (count and total_ms) and then turned back into
    timings with ``tree_to_timings``. Categories that were not logged are left out.
    """
    def node(name: str, children_dict: Dict) -> Optional[dict]:
        values = timings.get(name)
        children = {}
        for child_name, grandchildren in children_dict.items():
            child = node(child_name, grandchildren)
            if child is not None:
                children[child_name] = child
        count = len(values) if values else 0
        if count == 0 and not children:
            return None
        return {"count": count, "total_ms": sum(values) if values else 0.0, "children": children}

    phases = {}
    for phase, children_dict in TIMING_HIERARCHY.items():
        n = node(phase, children_dict)
        if n is not None:
            phases[phase] = n

    def flat(name: str) -> dict:
        return {"count": len(timings[name]), "total_ms": sum(timings[name])}

    overlapping = {c: flat(c) for c in OVERLAPPING_TOTALS_ORDER if c in timings}
    unknown = {c: flat(c) for c in sorted(timings)
               if c not in ALL_KNOWN_CATEGORIES and c not in OVERLAPPING_TOTALS}
    return {"phases": phases, "overlapping_totals": overlapping, "unknown_categories": unknown}


def tree_to_timings(tree: dict) -> Dict[str, TimingSeries]:
    """Inverse of ``timings_to_tree``: flatten a (possibly summed) tree back into timings."""
    timings: Dict[str, TimingSeries] = {}

    def walk(name: str, n: dict):
        if n["count"] > 0:
            timings[name] = TimingSeries(n["count"], n["total_ms"])
        for child_name, child in n["children"].items():
            walk(child_name, child)

    for name, n in tree["phases"].items():
        walk(name, n)
    for section in ("overlapping_totals", "unknown_categories"):
        for name, n in tree[section].items():
            timings[name] = TimingSeries(n["count"], n["total_ms"])
    return timings


def parse_timestamp_line(line: str) -> Optional[Tuple[str, int]]:
    """Parse a TIMING_TS:<name>:<epoch_ns> line."""
    sentinel = 'TIMING_TS:'
    if not line.startswith(sentinel):
        return None
    parts = line.split(':', 2)
    if len(parts) != 3:
        return None
    try:
        return parts[1], int(parts[2])
    except ValueError:
        return None


def _handle_timestamp(name: str, ns: int, ts: Dict[str, int], timings: Dict[str, List[float]]):
    """Turn the per-invocation wall-clock stamps into process-level durations.

    Sequence per client invocation:
      py_popen_start -> main_start -> ... -> main_end -> main_locals_destroyed -> py_communicate_end
    """
    ts[name] = ns
    ms = lambda a, b: (ts[b] - ts[a]) / 1e6
    if name == 'main_start' and 'py_popen_start' in ts:
        timings['process_startup'].append(ms('py_popen_start', 'main_start'))
    elif name == 'main_locals_destroyed' and 'main_end' in ts:
        timings['main_teardown_locals'].append(ms('main_end', 'main_locals_destroyed'))
    elif name == 'py_communicate_end' and 'main_locals_destroyed' in ts:
        timings['process_exit'].append(ms('main_locals_destroyed', 'py_communicate_end'))
        ts.clear()


# ---------------------------------------------------------------------------
# Hierarchical visual bar chart
# ---------------------------------------------------------------------------

def print_visual_breakdown(nodes: List[PhaseNode], wall_clock_ms: float):
    """Print a hierarchical bar chart reflecting the timing hierarchy."""
    if wall_clock_ms <= 0:
        return

    print("TIME BREAKDOWN (visual, % of wall clock)")
    print("-" * TABLE_WIDTH)

    label_width = max(42, _max_label_len(nodes, []) + 1)

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

def print_summary(timings: Dict[str, List[float]], problem_timings: List[ProblemTiming],
                  legacy: bool = False):
    """Print a summary of timing data.

    By default the tree is grouped by purpose (``build_purpose_hierarchy``). With ``legacy=True``
    it is the tree as the timers are nested in the code, with the original display groups.
    """
    global TABLE_WIDTH
    legacy_nodes = build_hierarchy(timings) if timings else []
    purpose_nodes = [] if (legacy or not timings) else build_staged_hierarchy(timings)
    # The tables printed, in order: (title, tree, separator lines between rows). The default report
    # shows the purpose tree first and the legacy tree (nested as the timers are) below it.
    if legacy:
        tables = [("TIMING BY PHASE", legacy_nodes, True)]
        nodes = legacy_nodes
    else:
        tables = [("TIMING BY PURPOSE", purpose_nodes, False),
                  ("TIMING BY TIMER NESTING (legacy view)", legacy_nodes, True)]
        nodes = purpose_nodes

    total_time_by_category = {cat: sum(vals) for cat, vals in timings.items()}

    wall_clock_ms = sum(total_time_by_category.get(c, 0) for c in TIMING_HIERARCHY)
    if wall_clock_ms == 0:
        wall_clock_ms = sum(total_time_by_category.values())
    wall_clock_s = wall_clock_ms / 1000.0

    def extra_row(cat):
        values = timings[cat]
        count = len(values)
        total_ms = sum(values)
        mean_ms = total_ms / count if count > 0 else 0
        return (cat, count, total_ms, mean_ms, as_percentage(total_ms, wall_clock_ms))

    # Rows shown below the trees: categories the hierarchy does not know, and the overlapping totals.
    unknown_rows = [extra_row(c) for c in sorted(
        set(timings.keys()) - ALL_KNOWN_CATEGORIES - OVERLAPPING_TOTALS)]
    overlapping_rows = [extra_row(c) for c in OVERLAPPING_TOTALS_ORDER if c in timings]

    # One layout for every table so their columns line up.
    layout = _table_layout([n for _, tree, _ in tables for n in tree],
                           unknown_rows + overlapping_rows, wall_clock_ms)
    COL_CAT = layout["cat"]
    TABLE_WIDTH = layout["table"]

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

    # -- Hierarchical table --------------------------------------------------

    w_cnt, w_tot, w_mean = layout["cnt"], layout["tot"], layout["mean"]
    w_parent, w_wall = layout["parent"], layout["wall"]

    def fmt_row(indent, name, count, total_ms, mean_ms, pct_parent, pct_wall):
        prefix = "  " * indent
        cat = f"  {prefix}{name}"
        c = f"{count:>{w_cnt},}" if count is not None else " " * w_cnt
        t = f"{total_ms:>{w_tot},.2f}" if total_ms is not None else " " * w_tot
        m = f"{mean_ms:>{w_mean},.2f}" if mean_ms is not None else " " * w_mean
        par = f"{pct_parent:>{w_parent - 1}.1f}%" if pct_parent is not None else " " * w_parent
        wall = f"{pct_wall:>{w_wall - 1}.1f}%" if pct_wall is not None else " " * w_wall
        print(f"{cat:<{COL_CAT}} {c} {t} {m} {par}   {wall}")

    SEPARATORS = {
        0: lambda: None,  # no separator before top-level (handled specially)
        1: lambda: print("    " + "-" * (TABLE_WIDTH - 4)),
        2: lambda: print("      " + "- " * ((TABLE_WIDTH - 6) // 2)),
        3: lambda: print("        " + ". " * ((TABLE_WIDTH - 8) // 2)),
    }

    def sep_top():
        print("  " + "=" * (TABLE_WIDTH - 2))

    def render_table(title: str, tree: List[PhaseNode], row_separators: bool):
        print(title)
        print("-" * TABLE_WIDTH)
        hdr_cat = "Category"
        print(
            f"  {hdr_cat:<{COL_CAT - 2}}"
            f" {'Count':>{w_cnt}}"
            f" {'Total (ms)':>{w_tot}}"
            f" {'Mean (ms)':>{w_mean}}"
            f" {'% of Parent':>{w_parent}}"
            f"   {'% of Wall':>{w_wall}}"
        )
        print("-" * TABLE_WIDTH)

        def render_node(node: PhaseNode, depth: int, parent_ms: Optional[float],
                        wall_clock_ms: float, is_first: bool):
            """Recursively render a PhaseNode at a given depth."""
            # Separators between siblings (not in the purpose tree, which has many sibling rows)
            if depth == 0 or not row_separators:
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

        for top_idx, node in enumerate(tree):
            if top_idx > 0:
                sep_top()
            pct_wall = as_percentage(node.total_ms, wall_clock_ms)
            fmt_row(0, node.name, node.count, node.total_ms, node.mean_ms, None, pct_wall)

            for i, child in enumerate(node.children):
                render_node(child, 1, node.total_ms, wall_clock_ms, is_first=(i == 0))

    for table_idx, (title, tree, row_separators) in enumerate(tables):
        if table_idx > 0:
            print()
        render_table(title, tree, row_separators)

    # Unknown categories
    if unknown_rows:
        print()
        print("  Unknown Categories")
        print(f"  {'-' * (TABLE_WIDTH - 2)}")
        for cat, count, total_ms, mean_ms, pct in unknown_rows:
            fmt_row(0, cat, count, total_ms, mean_ms, pct, None)

    # Overlapping totals: wrap other timers, so they are not part of the tree (they would be
    # double counted) but are useful to check that the tree accounts for all the time.
    if overlapping_rows:
        print()
        print("  Overlapping Totals (wrap other timers; not summed into the tree)")
        print(f"  {'-' * (TABLE_WIDTH - 2)}")
        for cat, count, total_ms, mean_ms, pct in overlapping_rows:
            fmt_row(0, cat, count, total_ms, mean_ms, pct, None)

    print()
    print("-" * TABLE_WIDTH)
    if legacy:
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
        for c in CPP_PHASE_GROUPS.get("C++ Client Setup & Teardown", [])
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
    if not legacy:
        insight_groups = build_purpose_hierarchy(timings)
        label_width = max([43] + [len(group.name) for group in insight_groups])
        for group in insight_groups:
            print(f"  {group.name:<{label_width}} {group.total_ms / 1000:.2f}s "
                  f"({as_percentage(group.total_ms, wall_clock_ms):.1f}%)")
    else:
        print(f"  {'Python phases (code gen, compilation):':<43} {python_codegen_time / 1000:.2f}s ({as_percentage(python_codegen_time, wall_clock_ms):.1f}%)")
        print(f"  {'Python I/O overhead (cache, YAML):':<43} {python_io_time / 1000:.2f}s ({as_percentage(python_io_time, wall_clock_ms):.1f}%)")
        print(f"  {'C++ client setup & teardown (per launch):':<43} {client_setup_time / 1000:.2f}s ({as_percentage(client_setup_time, wall_clock_ms):.1f}%)")
        print(f"  {'Data preparation (CPU init, GPU prep):':<43} {data_prep_time / 1000:.2f}s ({as_percentage(data_prep_time, wall_clock_ms):.1f}%)")
        print(f"  {'Reference computation (CPU GEMM):':<43} {ref_time / 1000:.2f}s ({as_percentage(ref_time, wall_clock_ms):.1f}%)")
        print(f"  {'Kernel execution (warmup, benchmark):':<43} {kernel_time / 1000:.2f}s ({as_percentage(kernel_time, wall_clock_ms):.1f}%)")
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
    parser.add_argument('--legacy', action='store_true',
                        help='Print the tree as the timers are nested in the code, with the original '
                             'display groups, instead of grouping by purpose')

    args = parser.parse_args()

    timings, problem_timings = analyze_timing_file(args.logfile)
    print_summary(timings, problem_timings, legacy=args.legacy)

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
