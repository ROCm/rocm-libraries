---
phase: 01-suite-execution-engine
plan: 02
subsystem: cli, reporting
tags: [cli-glob, suite-mode, provider-filter, progress-output, exit-codes]
dependency_graph:
  requires: [suite_runner.py, suite_results.py, SuiteConfig, benchmark_config.py, reporter.py, parser.py, main.py]
  provides: [run_suite, --provider flag, --engine flag, glob-based suite mode, suite reporter methods]
  affects: [cli/main.py (routing logic), cli/parser.py (--graph type change)]
tech_stack:
  added: []
  patterns: [glob-based-file-resolution, per-graph-error-isolation, exit-code-convention]
key_files:
  created:
    - tests/unit/cli/__init__.py
    - tests/unit/cli/test_suite_cli.py
    - tests/unit/reporting/test_suite_reporter.py
  modified:
    - src/dnn_benchmarking/cli/parser.py
    - src/dnn_benchmarking/cli/main.py
    - src/dnn_benchmarking/reporting/reporter.py
decisions:
  - Reporter suite methods added during Task 1 (Rule 3 blocking dependency) since run_suite() calls them directly
  - Glob resolution uses stdlib glob.glob() with sorted() for deterministic file order
  - Single file backward compatibility preserved via Path(resolved_files[0]) in BenchmarkConfig creation
  - hipdnn_frontend import remains lazy (inside run_suite) consistent with existing run_benchmark pattern
metrics:
  duration: 12m
  completed: "2026-04-09"
  tasks_completed: 2
  tasks_total: 2
  tests_added: 26
  files_created: 3
  files_modified: 3
---

# Phase 01 Plan 02: Suite CLI Integration Summary

Extended CLI to accept glob patterns for multi-graph suite execution with per-graph error isolation, provider/engine filter flags, structured JSON output, and D-09 exit code convention (0=pass, 1=error, 2=correctness failure).

## What Was Built

### CLI Parser Extensions (`parser.py`)
- Changed `--graph` argument from `type=Path` to `type=str` so glob patterns like `'graphs/*.json'` are preserved as strings rather than being converted to Path objects.
- Added `Suite Options` argument group with `--provider` (str, default None) and `--engine` (int, default None) filter flags per D-03.
- Full backward compatibility: single file paths still work unchanged.

### Suite Workflow (`run_suite()` in `main.py`)
- `run_suite(graph_paths, config, output_path)`: Iterates all graph files sequentially with a single shared `hipdnn.Handle`. Per D-06, warmup and benchmark iterations apply per graph independently.
- Per-graph error isolation (D-08): `GraphLoadError` and generic exceptions are caught per graph, recorded as error `GraphResult`, and execution continues to the next graph.
- Exit code convention (D-09): returns 2 for correctness failures, 1 for errors, 0 for all pass.
- JSON output (D-16): writes `SuiteResult.save_json()` only when `--output` is specified.
- Console progress (D-17): prints `[i/total] graph_name...` before each graph and `-> N passed, N failed, N skipped, N errored` after each.

### Main Routing Logic (`main()` in `main.py`)
- Resolves `--graph` value via `glob.glob()` with `sorted()` for deterministic order.
- Routes to `run_suite()` when multiple files match (automatic suite mode per D-05).
- Routes to existing `run_benchmark()` when exactly one file matches (backward compatible).
- Returns exit code 1 with error message when zero files match.
- A/B testing mode still works via `--AId`/`--BId` flags (checked before suite routing).

### Reporter Suite Methods (`reporter.py`)
- `print_suite_header(total_graphs)`: Banner with graph count.
- `print_suite_graph_start(index, total, graph_name)`: Per-graph progress line.
- `print_suite_graph_result(passed, failed, skipped, errored)`: Per-graph result summary.
- `print_suite_graph_error(graph_name, error)`: Inline error display (D-08).
- `print_suite_summary(...)`: Final counts (graphs, combinations, pass/fail/skip/error).
- `print_suite_footer()`: Closing banner.
- All methods use `self._print()` and `self._print_line()` per existing Reporter pattern.

## Decisions Made

1. **Reporter methods in Task 1**: Added suite reporter methods during Task 1 implementation (Rule 3 deviation) because `run_suite()` calls them directly. Task 2 then added comprehensive tests for these methods.

2. **Sorted glob resolution**: `glob.glob()` results are sorted to ensure deterministic file ordering across runs. This matters for reproducibility of suite results.

3. **Lazy hipdnn import**: Kept `import hipdnn_frontend` inside `run_suite()` (not at module level) consistent with the existing `run_benchmark()` pattern. This allows CLI help and argument parsing to work without hipDNN installed.

4. **SuiteConfig wiring**: Suite config maps CLI args directly: `--warmup` to `warmup_iters`, `--iters` to `benchmark_iters`, `--provider` to `provider_filter`, `--engine` to `engine_filter`. No additional validation needed beyond what SuiteConfig.__post_init__ provides.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Reporter methods added during Task 1**
- **Found during:** Task 1 implementation
- **Issue:** `run_suite()` calls `reporter.print_suite_header()`, `print_suite_graph_start()`, etc., which were planned for Task 2. Without them, Task 1 tests cannot pass.
- **Fix:** Added all 6 suite reporter methods to `reporter.py` during Task 1's GREEN phase. Task 2 then focused on adding comprehensive tests.
- **Files modified:** src/dnn_benchmarking/reporting/reporter.py
- **Commit:** 0e19588b48

## Threat Flags

No new threat surfaces detected. Error messages show graph file path and exception message only (T-01-05 mitigated). No stack traces in console output.

## Known Stubs

None. All functions are fully implemented with no placeholder data or TODO markers.

## Commits

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 (RED) | Failing tests for suite CLI and run_suite | 1bb6919834 | tests/unit/cli/test_suite_cli.py |
| 1 (GREEN) | CLI parser, run_suite(), reporter methods | 0e19588b48 | parser.py, main.py, reporter.py, test_suite_cli.py |
| 2 | Suite reporter tests | 75711e755c | tests/unit/reporting/test_suite_reporter.py |

## Self-Check: PASSED

All 6 created/modified files verified present. All 3 commits verified in git log. 225 unit tests pass (including 26 new).
