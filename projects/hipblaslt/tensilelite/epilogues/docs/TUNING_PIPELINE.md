# TensileLite Tuning Pipeline

The tuning process is **fully automated end-to-end**. A human writes a YAML
file that defines the search space (which parameters to explore, which problem
sizes to benchmark). The pipeline generates every candidate kernel, compiles it
to GPU assembly, benchmarks it on real hardware, and selects the best mapping
from problem size to solution. The checked-in LibraryLogic YAMLs in
`hipBLASLt/library/` are the *output* of this process, not the input.

---

## The Three-Phase Pipeline

Entry point: `Tensile/bin/Tensile <config.yaml> <output_dir> [flags]`
→ `Tensile.Tensile()` → `executeStepsInConfig()` in `Tensile/Tensile.py`.

### Phase 1 — BenchmarkProblems (`Tensile/BenchmarkProblems.py`)

1. Reads the benchmark config YAML (`BenchmarkProblems` section).
2. Expands `ForkParameters` into a cartesian product of candidate solutions
   via `constructForkPermutations` (`itertools.product` over all value lists).
3. Constructs a `Solution` object per permutation — invalid combos (wrong tile
   shape for the ISA, LDS overflow, bad MI parameters, etc.) are silently
   dropped here.
4. Compiles all surviving solutions to `.co` code objects via rocisa + amdclang++
   (parallelised across CPU cores via `ParallelMap2`). Results are SHA-256
   cached so incremental re-runs skip recompilation.
5. Runs the `tensilelite-client` C++ binary on the GPU: for each (problem size,
   solution) pair it dispatches the kernel, measures wall-clock GFlops, and
   writes a CSV row.

**Output:** `2_BenchmarkData/*.csv` — raw GFlops matrix (rows = problem sizes,
columns = solutions).

### Phase 2 — LibraryLogic (`Tensile/LibraryLogic.py`)

1. Reads the CSV. Prunes solutions that never win on any benchmarked size
   (`removeLeastImportantSolutions` — iteratively removes the least marginal
   contributor while protecting any solution that is the sole winner for some
   size).
2. Builds a nested range-logic decision tree (`enRule`) over M/N/K that maps
   each region of the size space to the best surviving solution index.
3. Records exact winners for exact problem sizes.
4. Writes one LibraryLogic YAML per `(ScheduleName, ProblemType)`.

**Output:** `3_LibraryLogic/*.yaml` — these files are what gets shipped in
`projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/`.

### Phase 3 — LibraryClient (`Tensile/ClientWriter.py`)

Runs `TensileCreateLibrary` on the winning solutions and produces the final
deployable `.co`/`.dat` device library. Optionally runs a correctness
verification pass.

**Output:** `4_LibraryClient/` — compiled library ready for deployment.

---

## Invocation

```bash
# Standard full run (compile + benchmark + analyze)
Tensile/bin/Tensile my_sweep.yaml /path/to/output \
    --cxx-compiler $(which amdclang++) \
    --gpu-targets gfx950

# Split: compile on a build node, benchmark on a GPU node
Tensile/bin/Tensile my_sweep.yaml /output --build-only --gpu-targets gfx950
Tensile/bin/Tensile my_sweep.yaml /output --use-cache  --gpu-targets gfx950

# Re-benchmark known solutions from existing logic files (fastest iteration)
Tensile/bin/Tensile my_sweep.yaml /output \
    --solution-pool "existing_3_LibraryLogic/*.yaml" \
    --gpu-targets gfx950

# Resume a crashed benchmark run
Tensile/bin/Tensile my_sweep.yaml /output \
    --restore-from-log /output/tuning.log \
    --use-cache --gpu-targets gfx950
```

---

## Scale of a Typical Sweep

| Phase | Duration |
|-------|---------|
| Compilation | ~1–5 min per solution (rocisa codegen + amdclang++) |
| Benchmarking | ~10–30 min for 400 solutions × 9 sizes |
| Analysis | seconds |

A real BF16 GEMM production sweep:
- ~130 MI variants × DepthU(2) × TransposeLDS(2) × GSU(2) × SourceSwap(2)
  ≈ **4,160 candidates** before validation
- ~2,500 survive after the `Solution` constructor rejects invalid combos
- Small targeted sweeps (e.g. StreamK variants only): 20–80 candidates

---

## Note: `tox -e py3` is correctness testing, not tuning

The `tox -e py3` (common-tests) environment runs `invoke build-client` then
pytest over `Tensile/Tests/`. The YAML configs in `Tensile/Tests/common/` are
**correctness tests** — they expand to a small set of solutions (20–200) with
`NumElementsToValidate: 128` and assert numerical accuracy. They are not tuning
runs and do not produce LibraryLogic output.

---

## Tuning the PartialRMS K1 Kernel

`epilogues/yaml/gemm_partial_rms_k1.yaml` is a **single pinned solution** — a smoke
config for correctness verification, not a tuning sweep. To properly tune the
PartialRMS K1 kernel, create a benchmark config YAML that sweeps the relevant
parameters.

### Recommended parameter sweep

```yaml
GlobalParameters:
  MinimumRequiredVersion: 5.0.0

BenchmarkProblems:
  -
    - # ProblemType — bf16 TN GEMM with PartialRMS
      OperationType: GEMM
      DataType: b
      DestDataType: b
      ComputeDataType: s
      HighPrecisionAccumulate: True
      TransposeA: True
      TransposeB: False
      UseBeta: True
      Batched: True

    - # BenchmarkProblemSizeGroup
      BenchmarkCommonParameters:
        - KernelLanguage: ["Assembly"]
        - PartialRMS: [True]
        - StreamK: [3]
        - StreamKForceDPOnly: [1]
        - StreamKAtomic: [0]
        - DirectToLdsA: [1]
        - DirectToLdsB: [1]
        - UseSubtileImpl: [True]
        - GlobalSplitU: [1]          # StreamK DP-only requires GSU=1
        - _1LDSBuffer: [0]
        - PrefetchAcrossPersistent: [0]

      ForkParameters:
        - MatrixInstruction:          # tile size sweep via MI repetition counts
            - [16, 16, 32, 1, 1, 2, 8, 4, 2]   # MT128x256
            - [16, 16, 32, 1, 1, 4, 8, 4, 2]   # MT256x256 (current default)
            - [16, 16, 32, 1, 1, 4, 4, 4, 2]   # MT256x128
            - [16, 16, 32, 1, 1, 2, 4, 4, 2]   # MT128x128
            - [16, 16, 32, 1, 1, 4, 8, 2, 2]   # MT256x256, 2 waves
            - [16, 16, 32, 1, 1, 4, 2, 4, 2]   # MT256x64
            - [16, 16, 32, 1, 1, 8, 4, 2, 2]   # MT512x128
            - [16, 16, 32, 1, 1, 4, 4, 2, 2]   # MT256x128, 2 waves
        - DepthU: [32, 64, 128]
        - PrefetchLocalRead: [1, 3]
        - PrefetchGlobalRead: [2]
        - StaggerU: [0, 32]
        - SourceSwap: [False, True]
        - WorkGroupMapping: [1, 4, 8]
        - NonTemporalD: [0, 7]
        - StreamKFixupTreeReduction: [0, 1]
        - PartialRMSResidualAdd: [False]  # set [False, True] to tune both paths

      BenchmarkFinalParameters:
        - ProblemSizes:
            # Square shapes covering LLM attention/FFN use cases
            - Exact: [1024, 1024, 1, 4096]
            - Exact: [2048, 2048, 1, 4096]
            - Exact: [4096, 4096, 1, 4096]
            - Exact: [4096, 4096, 1, 8192]
            - Exact: [8192, 8192, 1, 4096]
            - Exact: [8192, 8192, 1, 8192]
            # Rectangular shapes
            - Exact: [2048, 4096, 1, 4096]
            - Exact: [4096, 2048, 1, 4096]
            # Odd sizes (stress partial tiles)
            - Exact: [4099, 4096, 1, 4096]
            - Exact: [4096, 4099, 1, 4099]
```

This sweep produces approximately **576 candidates** before validation filtering
(8 MI × 3 DepthU × 2 PLR × 1 PGR × 2 StaggerU × 2 SourceSwap × 3 WGM × 2 NTD
× 2 SKFixup), of which ~300–400 typically survive and are compiled and
benchmarked.

### Shortcut: seed from the known-good solution

If you only want to tune a few parameters around the existing working solution
(e.g. only DepthU and PrefetchLocalRead), use `--solution-pool` to start from
previously generated LibraryLogic YAMLs instead of sweeping from scratch.
First generate the logic YAMLs from the known-good YAML, then pass the
resulting directory as the pool:

```bash
# Step 1: generate LibraryLogic YAMLs from the known-good pinned solution.
python -m Tensile.Tensile epilogues/yaml/gemm_partial_rms_k1_rowmajor.yaml \
    /tmp/partialrms_out

# Step 2: seed the sweep from those YAMLs.
Tensile/bin/Tensile my_partialrms_sweep.yaml /output \
    --solution-pool "/tmp/partialrms_out/3_LibraryLogic/*.yaml" \
    --cxx-compiler $(which amdclang++) \
    --gpu-targets gfx950
```

### Producing and deploying new logic YAMLs

After tuning completes, the winning logic YAML from `3_LibraryLogic/` can be
committed to `projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx950/gfx950/Equality/`
and will be picked up automatically by the next `invoke build` run.

---

## Epilogue-Specific Tooling

The `epilogues/` subdirectory provides scripts and utilities for the PartialRMS
fused-epilogue kernels that sit outside the standard three-phase Tensile pipeline.

### Epilogue scripts (`epilogues/gen/` and `epilogues/scripts/`)

PartialRMS libraries are built through the native three-phase Tensile pipeline
(`Tensile.Tensile` → `3_LibraryLogic/` → `TensileCreateLibrary`). The one
remaining generator in `epilogues/gen/` is `gen_asm_logic.py`, which handles
hand-written `.s` kernels only.

| Script | Purpose |
|--------|---------|
| `build_library.sh` | Run Tensile + TensileCreateLibrary for any benchmark YAML; flags: `--yaml`, `--chip`, `--out-dir`, `--library-format`, `--client`. |
| `tune_all_epilogues.sh` | One-shot tuning for all three epilogue variants from `tune_all_epilogues.yaml`; flags: `--fast`, `--chip`, `--client`. |
| `test_client_partialrms.sh` | Run Tensile pipeline → compile → `tensilelite-client` for no-residual and residual-add modes; reports PASS/FAIL. |
| `test_client_partialrms_residual.sh` | Same library, sweeps M ∈ {256,255,1024,4096,100,1001} with `--partial-rms-residual-add`. |
| `run_client_partialrms.sh` | End-to-end run against a pre-built or freshly generated PartialRMS library; supports `--build-client`, `--lib-yaml`, `--problem-size`, `--num-elements`, `--extra-args`. |
| `build_asm_library.sh` | Wrap `gen_asm_logic.py` + `TensileCreateLibrary` for hand-written `.s` kernels. |
| `gen_asm_logic.py` | Register a hand-written `.s` kernel as a custom kernel and emit a LibraryLogic YAML referencing it via `CustomKernelName`. |

Flags shared by the test scripts: `--chip CHIP`, `--client PATH`,
`--out-dir DIR`. The scripts discover `tensilelite-client` on `PATH`
automatically; pass `--client` to override.

```bash
epilogues/scripts/test_client_partialrms.sh --chip gfx950 \
    --client build_tmp/tensilelite/client/tensilelite-client

epilogues/scripts/run_client_partialrms.sh  --chip gfx950 --build-client

epilogues/scripts/build_asm_library.sh my_kernel.s --chip gfx950
```

### Benchmark scripts (`epilogues/bench/`)

| Script | Benchmark | Key CLI flags |
|--------|-----------|---------------|
| `bench_gemm_rms.py` | Col-major K1 (GEMM+PartialRMS) → `partial_rms_epilogue` pipeline | `--M`, `--N-hidden`, `--K`, `--wg-n`, `--config`, `--eps`, `--warmup`, `--iters`, `--no-verify`, `--chip` |
| `bench_gemm_rmsnorm.py` | Row-major K1 (GEMM+PartialRMS, `PartialRMSAxis=0`) → `row_div` pipeline | `--M`, `--N-hidden`, `--K`, `--wg-n`, `--eps`, `--warmup`, `--num-iters`, `--no-verify`, `--chip`, `--config` |
| `bench_row_div.py` | Standalone `row_div` kernel | `--m`, `--n`, `--n-d`, `--num-iters`, `--warmup`, `--block-size`, `--eps`, `--inv-d`, `--chip`, `--no-verify` |

```bash
python epilogues/bench/bench_gemm_rms.py --M 4096 --N-hidden 4096 --K 4096
python epilogues/bench/bench_gemm_rmsnorm.py --M 4096 --N-hidden 4096 --K 4096 --no-verify
python epilogues/bench/bench_row_div.py --m 4096 --n 4096 --n-d 32
```

### Unit tests (`epilogues/unittests/`)

The unit tests live under `epilogues/unittests/` and are discovered by pytest
via `epilogues/pyproject.toml` (`testpaths = ["unittests"]`). Run them from
the `epilogues/` directory:

```bash
cd epilogues
pytest unittests/ -v
```

Or from the TensileLite root (pytest resolves paths via `pyproject.toml`):

```bash
pytest epilogues/unittests/ -v
```

The `conftest.py` inserts the TensileLite root into `sys.path` so that
`Tensile`, `rocisa`, and `epilogues` are importable regardless of the
working directory.

---

## Key Flags Reference

| Flag | Purpose |
|------|---------|
| `--gpu-targets gfx950` | Target GPU architecture (semicolon-separated for multiple) |
| `--cxx-compiler amdclang++` | Compiler for assembly |
| `--global-parameters CpuThreads:N` | Parallel compilation workers (0=no threading, -1=nproc) |
| `--build-only` | Compile only, skip benchmarking |
| `--use-cache` | Skip recompilation of cached solutions |
| `--solution-pool <glob>` | Seed from existing LibraryLogic YAMLs (glob pattern, e.g. `3_LibraryLogic/*.yaml`) |
| `--restore-from-log <log>` | Resume from a crashed benchmark run |
| `--library-format yaml\|msgpack` | Output format (default: msgpack) |
