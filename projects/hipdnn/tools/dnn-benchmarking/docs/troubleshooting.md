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

The profiling smoke tests in `tests/integration/test_profiling.py`
(`--pmc`, `--emit-trace`, `--perf`, `--roofline`) plus the combined
`test_combined_pmc_perf_roofline_merge_into_one_extra_metrics` are
**double-gated**: each carries a pytest marker (`rocprofv3`, `perf`, or
`rocprof_compute`) AND an inline binary/host probe that calls
`pytest.skip` when the precondition isn't met. Strict payload tests also
carry `profiling_strict` and are skipped unless pytest is run with
`--profiling-strict`.

### Running them locally

On a host with `/opt/rocm/bin/rocprofv3`, `perf` and
`rocprof-compute` installed:

```bash
# Single sources — the `not perf and not rocprof_compute` exclusions
# matter: the combined-source smoke also carries the `rocprofv3`
# marker, so the bare `-m rocprofv3` form would collect it and demand
# perf + rocprof-compute too.
pytest -m "rocprofv3 and not perf and not rocprof_compute" tests/integration/test_profiling.py
pytest -m "perf and not rocprofv3 and not rocprof_compute" tests/integration/test_profiling.py
pytest -m "rocprof_compute and not rocprofv3 and not perf" tests/integration/test_profiling.py

# Combined-source smoke (requires all three binaries + paranoid<=1)
pytest -m "rocprofv3 and perf and rocprof_compute" tests/integration/test_profiling.py
# Strict payload tests — require real profiler artifacts, not just
# error/skip diagnostics. Run only on a known-good profiling host.
pytest --profiling-strict -m profiling_strict tests/integration/test_profiling.py
```
`perf` needs `apt install linux-tools-generic`. User-space counters
(``cycles_user``, ``instructions_user``, ``ipc_user``) collect on a
default host without privileges. Kernel-space counters
(``cycles_kernel``, ``instructions_kernel``) additionally require
``sudo sysctl kernel.perf_event_paranoid=1``; without it they record
as ``None`` and the slice carries ``kernel_events_skipped_reason``.

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

Default is **600 s (10 min)** per subprocess. Override via
`--profiling-timeout SECONDS`:

```bash
# Bump to 30 min for genuinely-long workloads (large convs under
# multi-pass PMC replay on a slow host).
python -m dnn_benchmarking --profiling-timeout 1800 ...

# Disable the timeout entirely (not recommended — a wedged
# subprocess will hang the suite indefinitely).
python -m dnn_benchmarking --profiling-timeout 0 ...
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

## Viewing profiling artefacts

The opt-in profiling sources don't render their own visualisations —
they capture raw artefacts (CSVs, sqlite dbs, pftrace files) and
record paths in `extra_metrics["<source>"]` of the result JSON. The
console reporter prints the open-it-with hint next to each path. The
recipes below cover everything beyond the one-liner hint.

### Roofline (`--roofline`)

The recorded artefacts:

| Key | What |
|---|---|
| `roofline_csv` | Empirical HBM/compute ceilings for the GPU |
| `sysinfo_csv` | Host + GPU topology rocprof-compute captured |
| `workload_path` | Directory to feed to `rocprof-compute analyze` |

#### Set up the analyze venv (once per host)

`rocprof-compute analyze` has its own Python dependencies that are
**not** in the dnn-benchmarking venv. Installing them into the
dnn-benchmarking venv would downgrade numpy 2.x → 1.26.4 and break
torch — use a separate venv:

```bash
python3 -m venv /tmp/rcompute-venv
/tmp/rcompute-venv/bin/pip install -r /opt/rocm/libexec/rocprofiler-compute/requirements.txt
```

Activate it before any `analyze` invocation:

```bash
source /tmp/rcompute-venv/bin/activate
```

#### ASCII roofline (terminal, no setup)

Use `--block 4` to isolate the roofline panel — bare `analyze --path`
runs every speed-of-light block and floods stdout with "Failed to
evaluate expression" warnings for the ~200 counters that `--roof-only`
didn't capture.

```bash
rocprof-compute analyze --path <workload_path> --block 4
```

Output: empirical ceilings table + AI plot points table + the ASCII
roofline plot with `Peak MFMA-FP32 / Peak VALU-FP32 / HBM / L2 / L1 /
LDS` bandwidth lines and per-kernel AI points (`AI_HBM_K0`, etc.).

#### Interactive GUI (web)

```bash
rocprof-compute analyze --path <workload_path> --gui
```

Starts a Dash app on `http://0.0.0.0:8050/`. On a remote host
(Alola, etc.), open an SSH tunnel from your local machine:

```bash
ssh -L 8050:localhost:8050 sareeder@ctr2-alola-login-04.adc.amd.com
```

Then browse to `http://localhost:8050`. The GUI offers an
interactive roofline (zoom, hover for per-kernel AI), datatype
selector (FP16/BF16/FP32/INT8 ceilings), and the full speed-of-light
dashboard.

#### Textual UI (same shell, no port forwarding)

```bash
rocprof-compute analyze --path <workload_path> --tui
```

#### Re-rendering with a non-default datatype

`--roofline` profiles ceilings at rocprof-compute's default (FP32);
the post-hoc render can switch:

```bash
rocprof-compute analyze --path <workload_path> --roofline-data-type FP16
```

### Kernel + memcpy trace (`--emit-trace pftrace` / `kineto`)

The recorded artefacts:

| Key | What |
|---|---|
| `path` | The viewable trace file |
| `db_path` | (kineto only) The rocpd source sqlite db |
| `format` | What you asked for |
| `recorded_format` | What was actually recorded (differs only on the kineto-downgraded-to-pftrace path) |
| `kineto_unavailable` | Set when rocpd isn't importable and we recorded pftrace instead |

#### Open in Perfetto

Both `.pftrace` and the chrome JSON drop into the Perfetto UI:

```text
https://ui.perfetto.dev/
```

Drag the file onto the page (or `Open Trace File`). Tracks: GPU
kernels grouped by stream, memcpy operations, host-side HIP API
calls. Zoom with `W`/`S`, pan with `A`/`D`. Click a kernel slice for
launch params, duration, queue depth.

#### Open the kineto chrome JSON in Chrome

```text
chrome://tracing/
```

Click `Load`, pick the `.chrome.json` file. Perfetto is the better
viewer; this is the legacy path if you need Chrome's older trace
processor.

#### If conversion failed (rocpd db only)

`kineto_unavailable` means we have the rocpd `.db` but no chrome
JSON. Convert manually:

```bash
python -m rocpd convert -i <db_path> --output-format chrome -o trace.chrome.json
```

Then open `trace.chrome.json` in Perfetto.

### PMC counters (`--pmc basic|memory|flops|all`)

The recorded artefacts:

| Key | What |
|---|---|
| `counters` | Per-counter aggregates (sum + mean-per-kernel) |
| `per_kernel` | Per-kernel × counter values |
| `db_path` | Raw rocpd sqlite db with every event |
| `set` / `arch` | Which counter set was collected and on which arch |
| `arch_narrowed_to_fallback` | Set when `--pmc all` couldn't find your arch in `PMC_SETS` and fell back to the 2-counter fallback group |

For aggregates, just read `extra_metrics["pmc"]["counters"]` directly
from the result JSON — they're already summarised.

#### Full speed-of-light dashboard via rocprof-compute

The rocpd db can be fed to `rocprof-compute analyze`, which provides
the same per-block panels (compute pipeline, LDS, L1/L2, fabric,
etc.). Point `--path` at the db's parent directory:

```bash
source /tmp/rcompute-venv/bin/activate  # same venv as for roofline
rocprof-compute analyze --path "$(dirname <db_path>)"
```

`--list-blocks gfx90a` (or your arch) shows the available panels;
filter with `-b <id>`.

#### Ad-hoc sqlite query

For one-off questions ("which kernel had the most cache misses"):

```bash
sqlite3 <db_path>
sqlite> .tables
sqlite> SELECT sym.kernel_name, p.pmc_id, AVG(p.value), COUNT(*)
   ...> FROM rocpd_pmc_event_<uuid> p
   ...> JOIN rocpd_kernel_dispatch_<uuid> k ON p.event_id = k.dispatch_id
   ...> JOIN rocpd_info_kernel_symbol_<uuid> sym ON k.kernel_id = sym.id
   ...> GROUP BY sym.kernel_name, p.pmc_id;
```

(Table suffixes vary per run — `.tables` first.)

### CPU counters (`--perf`)

The recorded artefacts:

| Key | What |
|---|---|
| `cycles_user` / `instructions_user` / `ipc_user` | Always present |
| `cycles_kernel` / `instructions_kernel` | Only present when `/proc/sys/kernel/perf_event_paranoid <= 1` |
| `task_clock_ms` / `context_switches` / `page_faults` | Process-wide host counters |
| `csv_path` | Raw `perf stat -x,` output |
| `kernel_perf_paranoid` | The paranoid value at run time (so you know why kernel counters might be None) |

The summary fields cover the common asks. The raw CSV exists for
people who want the per-event breakdown without our aggregation —
it's the standard `perf stat -x,` seven-column format
(`<value>,<unit>,<event>,<run-time-ns>,<percent>,<metric>,<metric-unit>`).

If `cycles_kernel` is None, check `kernel_events_skipped_reason` in
the slice — it spells out the paranoid threshold:

```bash
sudo sysctl kernel.perf_event_paranoid=1
```

…then rerun for kernel events. `perf_event_paranoid=2` is the
default on most distros and blocks kernel events; `>= 3` blocks
unprivileged tracing entirely.
