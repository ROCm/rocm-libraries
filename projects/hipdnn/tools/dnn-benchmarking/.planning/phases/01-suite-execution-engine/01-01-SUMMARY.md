---
phase: 01-suite-execution-engine
plan: 01
subsystem: execution, reporting, config
tags: [suite-runner, timing, correctness, data-model, json-serialization]
dependency_graph:
  requires: [executor.py, buffer_manager.py, timing.py, comparison.py, reference_provider.py, statistics.py, benchmark_config.py]
  provides: [suite_runner.py, suite_results.py, SuiteConfig]
  affects: [execution/__init__.py (not modified - downstream integration)]
tech_stack:
  added: []
  patterns: [dataclass-based-result-model, provider-engine-iteration, correctness-comparison-pipeline]
key_files:
  created:
    - src/dnn_benchmarking/execution/suite_runner.py
    - src/dnn_benchmarking/reporting/suite_results.py
    - tests/unit/execution/test_suite_runner.py
    - tests/unit/reporting/test_suite_results.py
  modified:
    - src/dnn_benchmarking/config/benchmark_config.py
decisions:
  - SuiteConfig placed in existing benchmark_config.py to keep config dataclasses co-located
  - suite_results.py implements its own TimingStats rather than reusing BenchmarkStats to keep suite result model self-contained and JSON-serialization-aware
  - Provider/engine discovery falls back to default list when hipDNN API enumeration is unavailable
  - _is_support_error heuristic checks error message keywords to distinguish skipped vs error status
metrics:
  duration: 8m
  completed: "2026-04-09"
  tasks_completed: 2
  tasks_total: 2
  tests_added: 35
  files_created: 4
  files_modified: 1
---

# Phase 01 Plan 01: Suite Execution Core Summary

Suite runner with per-graph provider/engine iteration, granular timing (CPU build, GPU kernel, E2E), correctness checking via ArrayComparator against reference providers, and a dataclass-based result model with graph-first JSON serialization.

## What Was Built

### Suite Runner (`suite_runner.py`)
- `run_graph_all_providers()`: Iterates all discovered providers and engines for a single graph, capturing separated CPU build time (via `Executor.init_time_ms`), GPU kernel timing, and E2E wall-clock timing per combination.
- `discover_providers()` / `discover_engines()`: Dynamic runtime discovery of hipDNN providers and engines (D-01). Falls back gracefully when API enumeration is unavailable.
- `_get_reference_provider()`: Validates reference provider availability and graph support before correctness checking.
- `_check_correctness()`: Compares GPU output against reference provider output using `ArrayComparator` (CORR-02). Collects input data from BufferManager, computes reference via the provider, and aggregates per-output-tensor comparison results.
- Provider/engine filter support via `config.provider_filter` and `config.engine_filter` (D-03).
- Single attempt per combination with no retry (D-10).
- Error entries record status + message only with no partial timing (D-07).

### Suite Config (`SuiteConfig` in `benchmark_config.py`)
- Dataclass with: `warmup_iters`, `benchmark_iters`, `seed`, `provider_filter`, `engine_filter`, `rtol`, `atol`, `gpu_backend`, `reference_provider`.
- Validates warmup/benchmark iteration counts on construction.

### Suite Result Data Model (`suite_results.py`)
- `TimingStats`: mean, std, min, max, p95, p99 (D-13) with `from_timings()` and `to_dict()`.
- `CorrectnessResult`: `execution_success` (CORR-01), `tolerance_match` (CORR-02), `passed` property (CORR-03), rtol/atol (D-15), max diffs, error message.
- `ProviderEngineResult`: status (success/error/skipped), separated timing fields, correctness. `to_dict()` omits timing on error (D-07).
- `GraphResult`: graph_name, graph_path, results list.
- `SuiteMetadata`: timestamp, hostname, counts, ROCm/GPU/Python/hipDNN version (D-12).
- `SuiteResult`: graph-first nesting with `to_dict()`, `to_json()`, `save_json()` (D-11).
- `collect_environment_info()`: Detects ROCm, GPU, Python, hipDNN versions.

## Decisions Made

1. **SuiteConfig in existing file**: Placed `SuiteConfig` in `benchmark_config.py` alongside `BenchmarkConfig`, `ABTestConfig`, and `ValidationConfig` to keep configuration dataclasses co-located.

2. **Separate TimingStats**: Created a new `TimingStats` in `suite_results.py` rather than reusing `BenchmarkStats` from `statistics.py`. This keeps the suite result model self-contained with its own `to_dict()` serialization, avoiding coupling to the single-benchmark reporting structure.

3. **Support error heuristic**: `_is_support_error()` inspects error message text for keywords like "support check failed", "not supported". This distinguishes skipped (unsupported combination) from error (hard failure) without requiring structured exception types from hipDNN.

4. **Fallback provider list**: When hipDNN API doesn't expose provider enumeration, falls back to `["miopen"]`. This is pragmatic since MIOpen is the primary provider, and the runner records errors/skips for any that don't work.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Created suite_results.py during Task 1 instead of Task 2**
- **Found during:** Task 1 implementation
- **Issue:** Task 1's suite_runner.py imports from suite_results.py (CorrectnessResult, ProviderEngineResult, GraphResult, TimingStats), but suite_results.py was planned for Task 2.
- **Fix:** Implemented suite_results.py fully during Task 1's GREEN phase since it was a blocking import dependency. Task 2 then focused on adding comprehensive tests.
- **Files modified:** src/dnn_benchmarking/reporting/suite_results.py
- **Commit:** 6ac9df6d4f

## Threat Flags

No new threat surfaces detected beyond what was identified in the plan's threat model.

## Known Stubs

None. All functions are fully implemented with no placeholder data or TODO markers.

## Commits

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 (RED) | Failing tests for suite runner | 22e36230ef | tests/unit/execution/test_suite_runner.py |
| 1 (GREEN) | Suite runner + suite results + SuiteConfig | 6ac9df6d4f | suite_runner.py, suite_results.py, benchmark_config.py |
| 2 | Suite results tests | 210369b6ed | tests/unit/reporting/test_suite_results.py |

## Self-Check: PASSED

All 5 created/modified files verified present. All 3 commits verified in git log.
