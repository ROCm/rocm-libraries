# Troubleshooting

## Using ROCm libraries from the venv

Prefer the venv ROCm SDK libraries first to avoid LLVM symbol mismatches:

```bash
export LD_LIBRARY_PATH=$PWD/.venv/lib/python3.12/site-packages/_rocm_sdk_core/lib:\
$PWD/.venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx90X_dcgpu/lib:\
$PWD/.venv/lib/python3.12/site-packages/triton/backends/amd/lib:\
$LD_LIBRARY_PATH
```

You can make this venv-agnostic by resolving `site-packages` at runtime:

```bash
VENV_SITE=$(python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)
export LD_LIBRARY_PATH=$VENV_SITE/_rocm_sdk_core/lib:\
$VENV_SITE/_rocm_sdk_libraries_gfx90X_dcgpu/lib:\
$VENV_SITE/triton/backends/amd/lib:\
$LD_LIBRARY_PATH
```

## Profiling integration tests

The four tests in `tests/integration/test_profiling.py` (`--pmc`,
`--emit-trace`, `--perf`, `--roofline`) plus the combined
`test_combined_pmc_perf_roofline_merge_into_one_extra_metrics` are
**double-gated**: each carries a pytest marker (`rocprofv3`, `perf`, or
`rocprof_compute`) AND an inline binary/host probe that calls
`pytest.skip` when the precondition isn't met. Default `pytest`
invocations therefore skip them silently, even on a GPU host.

### Running them locally

On a host with `/opt/rocm/bin/rocprofv3`, `perf` and
`rocprof-compute` installed:

```bash
# Single sources
pytest -m rocprofv3 tests/integration/test_profiling.py
pytest -m perf tests/integration/test_profiling.py
pytest -m rocprof_compute tests/integration/test_profiling.py

# Combined-source smoke (requires all three binaries + paranoid<=1)
pytest -m "rocprofv3 and perf and rocprof_compute" tests/integration/test_profiling.py
```

`perf` needs `apt install linux-tools-generic` plus
`sudo sysctl kernel.perf_event_paranoid=1` for user-event collection.

### CI

CI does **not** currently run these — they require a GPU runner. Until
one is wired up, the gate is: anyone touching
`metrics/profiling_orchestrator.py`, `metrics/rocprof_pmc.py`,
`metrics/rocprof_trace.py`, `metrics/perf.py`, or `metrics/roofline.py`
should run the relevant marker-gated tests manually on a gfx90a or
gfx942 host before merging. Tracking a nightly GPU runner job is
follow-up work, not a blocker.

### Tuning the profiling subprocess timeout

Every external profiler invocation (rocprofv3 PMC, rocprofv3 trace,
rocpd convert, perf stat, rocprof-compute) is capped at a per-process
wall-clock budget. A wedged child surfaces as
`extra_metrics["<source>"]["skipped"] == "timed out after Ns"`
instead of blocking the entire suite.

Default is **600 s (10 min)** per subprocess. Override via env var:

```bash
# Bump to 30 min for genuinely-long workloads (large convs under
# multi-pass PMC replay on a slow host).
DNN_BENCH_PROFILING_TIMEOUT_S=1800 python -m dnn_benchmarking ...

# Disable the timeout entirely (not recommended — a wedged
# subprocess will hang the suite indefinitely).
DNN_BENCH_PROFILING_TIMEOUT_S=0 python -m dnn_benchmarking ...
```

The budget applies *per subprocess*, not per suite — a four-source
`--pmc basic --emit-trace pftrace --perf --roofline` invocation can
spend up to 4 × the budget under the worst case.

### Profiling VRAM headroom

The opt-in profiling pass spawns a fresh dnn-benchmarking subprocess
that re-runs the same workload under the external profiler. The parent
process tears down its `BufferManager` and `Executor` (releasing
workspace + I/O buffers) *before* spawning, so the subprocess gets the
full VRAM headroom the parent had — there is no double-allocation
peak. If you still see OOMs only under `--pmc` / `--roofline` and not
on the headline timed run, the cause is the profiler's own overhead
(rocprof-compute's roofline replay in particular allocates extra
device buffers); reduce `--iters` or run sources one at a time.
