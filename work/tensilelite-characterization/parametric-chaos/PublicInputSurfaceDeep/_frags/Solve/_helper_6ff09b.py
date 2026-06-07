def benchmark_rerun_guard(results_csv_exists: bool, force_redo: bool) -> bool:
    """Mirror of BenchmarkProblems.py:657 guard.
    Returns True iff the benchmark client should (re)run."""
    return (not results_csv_exists) or force_redo
