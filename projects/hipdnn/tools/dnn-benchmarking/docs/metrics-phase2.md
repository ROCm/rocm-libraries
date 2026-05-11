# Metrics expansion — Phase 2 design notes

Phase 1 of the metrics expansion (PR landing in `users/sareeder/dnn-benchmarking-metrics-expansion`) added always-on probes that cost ~zero extra wallclock: analytical FLOPs/IO, workspace size, host CPU rusage + RAM, amdsmi GPU snapshot, and machine metadata. Phase 2 introduces the **opt-in profiling pass** built on top of `rocprofv3` to capture per-kernel PMC counters and trace data.

This document captures the design at the point of Phase 1's landing so the Phase 2 PR can pick it up without re-investigation.

## Scope

Two opt-in flags, both reserved in the Phase 1 CLI but rejected at config-build time today:

* `--pmc {basic,memory,flops,all}` — re-runs the benchmark under `rocprofv3 --pmc <set>`, parses the resulting rocpd SQLite database, and folds per-kernel counter values into `ProviderEngineResult.extra_metrics["pmc"]`.
* `--emit-trace {pftrace,kineto}` — re-runs the benchmark under `rocprofv3 --kernel-trace --memory-copy-trace --output-format <fmt>`. Stores the output path in `extra_metrics["trace"]["path"]` and exposes a Perfetto-importable artifact for offline inspection.

Both make a *separate* benchmarking pass (timed pass first, profiling pass second) because PMC sampling inflates per-kernel time substantially — see *Measured overhead* below.

## Measured overhead (from Phase 1 investigation)

`2048×2048 fp16 GEMM`, 500 iters + 50 warmup, repeated 3× each, on a gfx942-class GPU. Baseline wallclock ≈ 2.05 s, baseline mean kernel time ≈ 0.180 ms.

| Mode | Wallclock vs baseline | Mean kernel-time inflation |
|---|---|---|
| `rocprofv3 --kernel-trace` | +17 % | **+5 %** |
| `rocprofv3 --pmc` × 4 single-pass counters | +30 % | **+50 %** |
| `rocprofv3 --pmc` × 8 single-pass counters | +22 % | **+55 %** |
| `rocprofv3 --pmc` × 10 (incl. derived `MeanOccupancyPerCU`) — multi-pass | hung, killed at 4 min for what was a 0.4 s baseline | — |

Tool init/finalise was ~30–50 ms in all cases (rocpd SQLite generation), independent of iteration count.

**Implications for the design:**

* `kernel-trace` is cheap enough that we can do it in the same run as the timed pass if we want, but a separate pass keeps timing accuracy clean and matches the architecture used for `--pmc`.
* `--pmc` *must* be a separate pass — kernel timing recorded under PMC is ~1.5× the real value.
* Multi-pass replay is a footgun. The CLI must hand-curate single-pass-safe counter sets per arch and either drop offending counters or warn loudly.

## Architecture: the re-exec sub-mode

Adding three hidden CLI flags to `cli/parser.py` (suppressed in `--help`):

```
--internal-profiling-run          # this process is the profiling pass
--internal-profiling-engine ID    # the single engine to run
--internal-profiling-graph PATH   # the single graph to load
```

When `args.internal_profiling_run` is true, `cli/main.py:main()` short-circuits to a minimal one-engine, one-graph path that:

1. Skips `gpu_check.gpu_is_available()` (parent already did it).
2. Skips engine discovery — uses the engine ID passed in directly.
3. Skips `Reporter` console output entirely (the profiler writes to stderr and a results dir).
4. Calls `executor.prepare → bm.allocate → bm.fill_inputs_random(seed=parent_seed) → executor.warmup → executor.benchmark`.
5. Exits 0 on success, 1 on any error.

The orchestrator (parent) builds the inner argv with all `--pmc/--emit-trace/--perf/--roofline` flags **stripped** to prevent recursion. The orchestrator lives in a new `metrics/profiling_orchestrator.py` and exports:

```python
def run_profiling_passes(
    graph_path: Path,
    engine_id: int,
    seed: Optional[int],
    warmup_iters: int,
    benchmark_iters: int,
    metrics_config: MetricsConfig,
    plugin_path: Optional[Path],
    out_dir: Path,
) -> Dict[str, Any]:
    """Run one profiling pass per requested source. Returns extra_metrics dict."""
```

Called once per `(graph, engine)` pair from `_run_single_provider_engine` after the timed pass succeeds. Failure of any single profiling pass adds a diagnostic to the result but does not fail the engine.

### Profiling output directory

New CLI flag: `--profiling-output-dir DIR` (default `./profiling-output/<utc-timestamp>/`). Each profiling pass writes into a unique subdir keyed by `<graph-stem>_<engine_id>_<source>/`:

```
profiling-output/2026-05-11T12-00-00Z/
  sample_conv_fwd_1_pmc_basic/
    <hostname>/<pid>_results.db        # rocpd SQLite
  sample_conv_fwd_1_trace_pftrace/
    <hostname>/<pid>_results.pftrace   # Perfetto
```

Path strings are recorded in `extra_metrics` so JSON consumers can locate the raw artifacts.

## PMC counter sets

Hardcoded per-arch single-pass-safe sets in `metrics/rocprof_pmc.py`. Counter names verified at runtime against `rocprofv3-avail counters --device <id>` output; missing counters are dropped with a warning, and an empty set after pruning skips the pass entirely.

```python
PMC_SETS: Dict[str, Dict[str, List[str]]] = {
    "gfx942": {
        "basic":   ["GRBM_GUI_ACTIVE", "SQ_WAVES", "SQ_INSTS_VALU", "SQ_BUSY_CYCLES"],
        "memory":  ["TCC_HIT_sum", "TCC_MISS_sum", "TCP_TCC_READ_REQ_sum", "TCC_EA_RDREQ_sum"],
        "flops":   ["SQ_INSTS_VALU_MFMA_F16", "SQ_INSTS_VALU_MFMA_BF16", "SQ_INSTS_VALU_MFMA_F32"],
    },
    "gfx90a":  { ... MFMA names differ — needs verification ... },
    "fallback": { "basic": ["GRBM_GUI_ACTIVE", "SQ_WAVES"] },
}
```

`--pmc all` is the union of all sets for the detected arch and emits a warning that multi-pass replay will trigger; we may decide to skip it by default and require `--pmc-allow-multipass` to enable.

`detect_arch()` order of preference:

1. `torch.cuda.get_device_properties(0).gcnArchName` (cheap, already imported).
2. `rocminfo` parse (subprocess fallback).
3. Returns `"fallback"` on failure → sparse counter set.

## rocpd SQLite parsing

The rocpd db schema has uuid-suffixed table names (`rocpd_pmc_event_<uuid>`, `rocpd_kernel_dispatch_<uuid>`). Parser walks `sqlite_master` to discover the actual table names, then joins:

```sql
SELECT k.kernel_name,
       p.pmc_id,
       AVG(p.value) AS mean_value,
       SUM(p.value) AS total_value
FROM rocpd_pmc_event_<uuid> p
JOIN rocpd_kernel_dispatch_<uuid> k ON p.event_id = k.id
GROUP BY k.kernel_name, p.pmc_id;
```

PMC names live in `rocpd_info_pmc_<uuid>` (id → name mapping), joined separately. Per-kernel rollups are aggregated up to per-engine totals; per-iter values are not stored in `extra_metrics["pmc"]` to keep JSON sane (pointer to the raw db is in `extra_metrics["pmc"]["db_path"]`).

## Trace export

Maps cleanly to rocprofv3 native flags:

* `--emit-trace pftrace` → `rocprofv3 --kernel-trace --memory-copy-trace --output-format pftrace`. Single `.pftrace` file → opens directly in [Perfetto UI](https://ui.perfetto.dev).
* `--emit-trace kineto` → emits the rocpd db plus a post-conversion to PyTorch Kineto JSON via `python3 -m rocpd convert -i <db> --output-format chrome`. Requires the rocpd Python module to be importable; if not, fall back to pftrace and log a notice.

## Test strategy

* New pytest marker in `pyproject.toml`: `rocprofv3: requires rocprofv3 binary on PATH`. Fixture `requires_rocprofv3` skips when binary absent.
* Unit tests for `rocprof_pmc.parse_pmc_db` using a checked-in fixture rocpd db (recorded once and committed under `tests/fixtures/`).
* Unit tests for `detect_arch` mocking `torch.cuda` and `subprocess.run`.
* Integration test: tiny conv graph, run with `--pmc basic`, assert `extra_metrics["pmc"]` is non-empty and the recorded `db_path` exists.

## Decisions to make before implementation

1. **Profiling output directory default** — `./profiling-output/<timestamp>/` vs. require user to pass `--profiling-output-dir`?
2. **`--pmc all` policy** — default-skip multi-pass with a warning, or default-allow with the warning?
3. **Per-iter PMC values** — keep aggregated only (recommended), or expose a `--pmc-per-iter` flag for users who want raw rows?
4. **Counter name verification cadence** — every run, or once at install time and cached?
5. **gfx942 MFMA counter names** — listed names are educated guesses; Phase 2 spike must verify against `rocprofv3-avail counters` on a real gfx942 host before locking the sets.
6. **Trace format default** — if both PyTorch and rocpd are available, prefer pftrace or kineto?
