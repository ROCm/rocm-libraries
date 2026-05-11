# Metrics expansion — Phase 3 design notes

Phase 3 adds two more opt-in profiling sources to the same re-exec architecture introduced by Phase 2 (see [metrics-phase2.md](./metrics-phase2.md)): `perf` for CPU-side hardware counters and `rocprof-compute --roof-only` for an automated roofline plot.

## Scope

Two CLI flags, both reserved in Phase 1:

* `--perf` — wraps the re-exec sub-mode in `perf stat -x, -e <events>`, parses the CSV from stderr, and folds CPU cycles/instructions into `extra_metrics["perf"]`.
* `--roofline` — re-runs under `rocprof-compute profile --roof-only` and records the resulting PDF + SQLite paths in `extra_metrics["roofline"]`.

## `--perf` design

### Events

Two tiers of events depending on host configuration:

```python
PERF_EVENTS_USER = [
    "cycles:u",
    "instructions:u",
    "task-clock",
    "context-switches",
    "page-faults",
]

PERF_EVENTS_KERNEL = [
    "cycles:k",
    "instructions:k",
]
```

User-space events are always available to the running user. Kernel-space events require `kernel.perf_event_paranoid <= 1` *or* `CAP_PERFMON` on the perf binary. The orchestrator probes `/proc/sys/kernel/perf_event_paranoid` once and concatenates `PERF_EVENTS_USER + PERF_EVENTS_KERNEL` only when allowed.

### Host preconditions

Documented in `docs/troubleshooting.md` (already exists):

* `perf` binary must be on PATH (Ubuntu: `apt install linux-tools-common linux-tools-generic`).
* For full coverage: `sudo sysctl kernel.perf_event_paranoid=1` (or set `CAP_PERFMON` on the perf binary).
* Inside enroot containers: `kernel.perf_event_paranoid` is host-wide, so the host setting applies. The Ubuntu `paranoid=4` quirk means even `--cap-add PERFMON` may not be enough on older kernels (see Phase 1 investigation notes).

If `perf` is missing, `--perf` skips with a single warning and the engine still runs. If only user-space events succeed, kernel fields stay None.

### Parsing

`perf stat -x,` writes one CSV row per event to stderr:

```
<value>,<unit>,<event-name>,<run-time-ns>,<percent-time-running>,<metric>,<metric-unit>
```

Parser splits on `,`, indexes by event name, normalises units. Result dict shape:

```json
{
  "perf": {
    "cycles_user": 1234567890,
    "instructions_user": 987654321,
    "ipc_user": 0.80,
    "cycles_kernel": null,
    "instructions_kernel": null,
    "task_clock_ms": 123.4,
    "context_switches": 12,
    "page_faults": 3,
    "kernel_perf_paranoid": 4,
    "kernel_events_skipped_reason": "perf_event_paranoid > 1"
  }
}
```

`ipc_user` is computed client-side (`instructions_user / cycles_user`) so JSON consumers don't have to derive it.

## `--roofline` design

### Mechanics

`rocprof-compute profile --roof-only -- python -m dnn_benchmarking --internal-profiling-run …` runs the workload twice (once for compute peak, once for memory peak), generates a `roofline.pdf` and a SQLite db. Outputs land in the standard `--profiling-output-dir/<graph>_<engine>_roofline/` subdirectory.

We do not parse the SQLite content — the value of `--roofline` is the visual artifact for human inspection. `extra_metrics["roofline"]` records pointers only:

```json
{
  "roofline": {
    "pdf_path": ".../roofline.pdf",
    "db_path":  ".../workload.db",
    "data_type": "FP32"
  }
}
```

### CLI sub-flag

`--roofline-data-type {FP32,FP16,BF16,FP64,INT8}` — passed through verbatim to `rocprof-compute --roofline-data-type`. Default `FP32`. Stack-style (multiple types in one PDF) is *not* exposed in Phase 3; users who want that can run `rocprof-compute` directly against the `extra_metrics["roofline"]["db_path"]`.

### Cost

Roofline collection runs the workload 3 separate times (timed pass + compute peak + memory peak), so wallclock for `--roofline` alone is ~3× baseline. Combined with `--pmc` it's ~5×. Documented prominently in the CLI help text.

## Re-using Phase 2 infrastructure

Both Phase 3 sources use the same re-exec architecture:

* The hidden `--internal-profiling-run` flag and the orchestrator in `metrics/profiling_orchestrator.py` are unchanged.
* New modules `metrics/perf.py` and `metrics/roofline.py` each export a single `run_<source>(...) -> Dict[str, Any]` function called by the orchestrator.
* Each respects `--profiling-output-dir`.
* Each uses `warn_once` for graceful degrade.

## Test strategy

* New pytest markers: `perf` (requires `/usr/bin/perf` and paranoid <= 1), `rocprof_compute` (requires `rocprof-compute` binary on PATH).
* Unit tests for `perf` CSV parser using checked-in `perf stat -x,` output snippets.
* Integration tests gated on the new markers.

## Decisions to make before implementation

1. **Perf paranoid behaviour** — auto-skip kernel events with a warning (current plan), or refuse to run `--perf` at all when paranoid > 1?
2. **Multiple roofline data types per run** — Phase 3 (single type) or defer to a future enhancement?
3. **Roofline output**: should we copy/inline the PDF as base64 in JSON, or always reference by path?
4. **Combined cost** — should `--pmc` + `--roofline` + `--perf` together be allowed in one CLI invocation, or require separate runs to keep wallclock manageable?
