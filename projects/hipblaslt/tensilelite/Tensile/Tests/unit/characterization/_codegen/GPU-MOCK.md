# GPU-less seam — the `--cpu-only` switch

This documents the `--cpu-only` switch that lets the TensileLite benchmark flow
run **without a GPU**, so the client/perf-run and device-probe paths are
reachable in CPU-only CI and in the coverage/characterization suite. It records
what the seam touches and — importantly — the **synthetic-perf caveat** that
comes with it.

## Why the seam is small

The campaign metric is whole-project line coverage, and almost the entire
pipeline is already CPU-reachable. Only a few rows actually need a device:

| Component | Needs GPU? | Notes |
| --- | --- | --- |
| Codegen emit (`KernelWriterAssembly`, `KernelWriter`, `Components/*`, `Asm*`) | No | Emits assembly *text*; CPU-only. |
| Solution derivation (`parseLibraryLogicFile`, forked-solution build) | No | Pure `Solution` construction from params. |
| `TensileCreateLibrary` | No | A cross-compiler — host `amdclang++` for the target ISA; no device. |
| **Client perf-run** (`ClientWriter.runClient`) | **Yes** | Launches the compiled client on a device to collect GEMM perf. ← stubbed |
| **System probes** (`amd-smi`, `rocm_agent_enumerator`, clock-frequency) | **Yes** | Shell out to detect/describe the device. ← skipped |
| **ISA detection** (`detectGlobalCurrentISA` → `_detectGlobalCurrentISA`) | **Yes** | Runs `amdgpu-arch` / `rocm_agent_enumerator`; raises GPU-less. ← spoofed |

So the switch only has to cover the last three rows. Everything above them runs
CPU-only with no mock.

## What the switch does (implemented surface)

The flag is `--cpu-only` and **requires `--gpu-targets`** (you must name the
target arch to spoof). It is plumbed through an internal global, not the
documented `--global-parameters` surface.

- **CLI flag** — `Tensile/Tensile.py` (`--cpu-only`, `dest="cpuOnly"`). It is
  stashed into internal plumbing: `globalParameters["CpuOnly"]` and the target
  arch into `globalParameters["CpuOnlyArch"]`.
- **Plumbing keys** — `Tensile/Common/GlobalParameters.py` defines
  `globalParameters["CpuOnly"]` (default `False`) and
  `globalParameters["CpuOnlyArch"]` (default `"gfx942"`); both reset via
  `restoreDefaultGlobalParameters()`. The flag is intentionally **not** exposed
  on the `--global-parameters` surface.
- **ISA spoof** — `Tensile/Common/Architectures.py::_detectGlobalCurrentISA`:
  when `CpuOnly` is set it returns a spoofed `IsaVersion` derived from
  `gfxToIsa(CpuOnlyArch)` instead of shelling out to `amdgpu-arch` /
  `rocm_agent_enumerator`, so `detectGlobalCurrentISA` no longer raises on a
  GPU-less host and `Tensile.Tensile()` runs CPU-only.
- **Device-launch stub** — `Tensile/ClientWriter.py::runClient`: when `CpuOnly`
  is set it writes the client config / run-script as usual but skips the
  device-bound client launch and returns returncode `0`.
- **Synthetic results CSV** — `Tensile/BenchmarkProblems.py::_writeSyntheticResultsCSV`
  writes a deterministic results CSV in the schema `LibraryLogic.addFromCSV`
  consumes. Every solution cell holds the fixed constant
  `_CPU_ONLY_SYNTHETIC_GFLOPS = 1000.0` (never random, never timestamped) so the
  file is byte-identical across runs and winner selection has a well-defined
  result.

When the switch is **off**, behavior is byte-identical to today — every spoof is
gated on `globalParameters["CpuOnly"]`.

The unit tests for the switch live in `Tensile/Tests/unit/test_cpu_only_switch.py`
(the T1–T12 rigor-gate suite referenced from this doc).

## Caveat — synthetic perf is not real perf

The synthetic CSV makes **measured performance** a stub. Any code that *branches
on measured performance* — winner selection in `addFromCSV`, retuning decisions,
efficiency thresholds — follows the constant `1000.0`, not real measurements.
Consequences:

- **Coverage of those branches is real**, but the **decisions** they produce are
  synthetic. Do not mistake a `LibraryLogic` / tuning result produced under
  `--cpu-only` for a meaningful one.
- Because every cell is identical, winner selection is determined by tie-breaking
  order, not performance. The output `*.yaml` is structurally valid but
  perf-meaningless.

For this reason `--cpu-only` must never *silently* drive a real `LibraryLogic`
generation step. The seam itself enforces this structurally: `Tensile.py`
declines the efficiency-based (`UseEffLike`) frequency path — and therefore the
real `LibraryLogic` winner-selection path — under `CpuOnly`, so synthetic perf
is never consumed for tuning in the first place.

Always treat `--cpu-only` runs as CI/coverage artifacts, not tuning results.

## Golden / determinism note

The synthetic CSV is fixed precisely so CPU-only runs are reproducible (a golden
recorded under `--cpu-only` stays byte-identical across runs). The ISA spoof is
keyed by `CpuOnlyArch`, so a CPU-only golden is implicitly keyed by the target
arch you pass — record and compare goldens per arch, as with the rest of the
codegen suite (see `../README.md`).
