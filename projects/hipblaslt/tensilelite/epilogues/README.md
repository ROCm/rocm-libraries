<!-- Copyright Advanced Micro Devices, Inc., or its affiliates. -->
<!-- SPDX-License-Identifier: MIT -->

# epilogues/

This directory holds the non-C++ assets for the fused PartialRMS GEMM
epilogues (gfx950, bf16) added to TensileLite. It contains the Python
codegen helpers, unit tests, benchmarks, driver scripts, tuning YAML,
PyTorch reference, and design docs.

The C++ runtime, Tensile-core codegen, client, and rocisa live in their normal
locations (`src/`, `include/`, `client/`, `rocisa/`, `Tensile/`) and are not
under this directory. The codegen entry point outside this directory is
`Tensile/Components/Subtile/SubtilePartialRMSEmit.py`.

## The three epilogue variants

| Index | Name | ProblemType flags |
|-------|------|-------------------|
| 0 | Plain GEMM | (none) |
| 1 | PartialRMS (no residual) | `UsePartialRMS: True`, `PartialRMSResidualAdd: False` |
| 2 | PartialRMS + residual add | `UsePartialRMS: True`, `PartialRMSResidualAdd: True` |

All three variants use bf16 TN layout (`TransposeA: True`, `TransposeB: False`),
`DataType: b`, `ComputeDataType: s`, `HighPrecisionAccumulate: True`,
`StreamK: 3`, and `StreamKForceDPOnly: 1`.

### Plain GEMM

```
D = α · op(A) · op(B) + β · C
```

### PartialRMS (no residual)

K1 writes two outputs in a single pass over the accumulator:

```
D[t, i]          = h1[t, i] · gamma[i]           (bf16)
partialBuf[t, τ] = Σ_{i ∈ tile τ} h1[t, i]²     (f32)
```

where `h1 = op(A) · op(B)` and `n_d = ceil(N_hidden / MT0)`.

A second-stage kernel (`row_div` or `col_div`) then normalises in-place:

```
D /= sqrt(invD · Σ_τ partialBuf[t, τ] + ε)      (invD = 1 / N_hidden)
```

### PartialRMS + residual add

Same as above but with `h_eff = h1 + residual` replacing `h1` throughout.
**MatrixInstruction wg0 (index 7) must be ≤ 2** for this variant.

## Prerequisites

```bash
source ~/.tensile/bin/activate                              # activate venv
export LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH:-}  # prefix GPU commands
invoke build-client                                        # build tensilelite-client once
```

A gfx950 GPU is required for GPU steps (tuning, integration tests, benchmarks).
Unit tests skip automatically on other chips. Python dependencies `amdgpu_exec`,
`ml_dtypes`, and `numpy` are installed in the venv.

## Directory layout

| Path | Contents |
|------|----------|
| `tensilelite/` | Importable package `epilogues.tensilelite`; four modules described below. |
| `unittests/` | pytest suites plus `conftest.py` and `epilogue_test_common.py`. |
| `bench/` | Three standalone microbenchmark scripts. |
| `gen/` | `gen_asm_logic.py` — registers a hand-written `.s` file as a custom kernel. |
| `scripts/` | Bash drivers: `build_library.sh`, `build_asm_library.sh`, `tune_all_epilogues.sh`, `test_client_partialrms.sh`, `test_client_partialrms_residual.sh`, `run_client_partialrms.sh`. |
| `yaml/` | Benchmark and solution YAMLs: `gemm_partial_rms_k1.yaml`, `gemm_partial_rms_k1_rowmajor.yaml`, `tune_all_epilogues.yaml`, `tune_all_epilogues_fast.yaml`, `tune_gemm_variants.yaml`, `tune_subtile_bf16_prms_8192.yaml`, `tune_subtile_bf16_prmsq_8192.yaml`, `debug_prmsq_single.yaml`. |
| `kernels/` | `row_div.s` — hand-written AMDGPU assembly for the second-stage row normalisation. |
| `torch/` | PyTorch GEMM+RMSNorm reference (`gemm_rmsnorm.py`). |
| `docs/` | Design docs: `TUNING_PIPELINE.md`, `GEMM_PARTIALRMS_LIBRARY.md`, `LIBRARY_CREATION_GUIDE.md`, `KERNEL_SELECTION.md`, `PARTIALRMS_EXTENSION.md`, `HIPBLASLT_INTEGRATION_STATUS.md`, `gemm_rmsnorm_analysis.md`. |

### `epilogues.tensilelite` package

| Module | Key public symbols |
|--------|--------------------|
| `partialrms_helpers` | `setup_tensile`, `compute_sk3_dp_args`, `generate_asm`, `compileSolution`, `buildSubtileArgs`, `buildRowDivArgs` |
| `numpy_helpers` | `randBf16`, `randGamma`, `partialSumSq`, `rmsDenom`, `rmsNormReference` |
| `yaml_solution_builder` | `buildSolutionsFromYaml`, `solutionsFromYaml`, `solutionId`, `problemSizesFromYaml`, `readTestAxes` |
| `partial_rms_epilogue_generator` | `build_partial_rms_epilogue` — rocisa-based second-stage RMSNorm kernel generator |

## How the build pipeline works

The three-phase Tensile flow is:

```
Tensile.Tensile  →  3_LibraryLogic/  →  TensileCreateLibrary  →  device library
```

**Phase 1+2 — benchmark and logic generation:**

```bash
python -m Tensile.Tensile <yaml> <out-dir>
# Produces: <out-dir>/1_BenchmarkProblems/, 2_BenchmarkData/, 3_LibraryLogic/
```

**Phase 3 — compile:**

```bash
python -m Tensile.TensileCreateLibrary \
    --library-format msgpack --architecture gfx950 \
    <out-dir>/3_LibraryLogic <out-dir>/library HIP
```

### Why `NumElementsToValidate: 0`

Both tune YAMLs set `NumElementsToValidate: 0`. Tensile's internal benchmark
client compares D against a plain GEMM reference and cannot validate the fused
epilogue side outputs (`partialBuf`). Correctness is validated separately by
the `unittests/` pytest suite and the `test_client_*.sh` integration tests,
which use epilogue-aware reference implementations.

### Distinct LibraryLogic filenames

`ProblemType.__str__` now includes `PRMS` and `PRMS_RA` suffixes, so all three
variants produce distinct filenames in `3_LibraryLogic/` and coexist in one
merged device library. At runtime, dispatch predicates select the correct kernel
based on the problem flags.

## Quick start: build a library

```bash
# Build a merged library covering all three variants (fast, ~5 min on gfx950).
epilogues/scripts/build_library.sh \
    --yaml epilogues/yaml/tune_all_epilogues_fast.yaml \
    --chip gfx950 \
    --library-format yaml

# Optionally run a client smoke test (requires tensilelite-client).
epilogues/scripts/build_library.sh \
    --yaml epilogues/yaml/tune_all_epilogues_fast.yaml \
    --chip gfx950 --library-format yaml \
    --client build_tmp/tensilelite/client/tensilelite-client
```

### `build_library.sh` flags

| Flag | Default | Description |
|------|---------|-------------|
| `--yaml PATH` | (required) | Benchmark YAML to run through the Tensile pipeline. |
| `--chip CHIP` | `gfx950` | GPU architecture string passed to TensileCreateLibrary. |
| `--out-dir DIR` | `/tmp/lib_<yaml-stem>` | Output root for all generated files. |
| `--library-format yaml\|msgpack` | `msgpack` | Format for the compiled device library. |
| `--client PATH` | (none) | tensilelite-client binary; enables a post-build smoke test. |

## Tuning

```bash
# Fast round: one tile + one problem size per variant (<5 min on gfx950).
epilogues/scripts/tune_all_epilogues.sh --fast --chip gfx950

# Comprehensive round: multiple tiles and problem sizes.
epilogues/scripts/tune_all_epilogues.sh --chip gfx950

# With GPU smoke tests after the build.
epilogues/scripts/tune_all_epilogues.sh --fast --chip gfx950 \
    --client build_tmp/tensilelite/client/tensilelite-client
```

### `tune_all_epilogues.sh` flags

| Flag | Default | Description |
|------|---------|-------------|
| `--yaml PATH` | `epilogues/yaml/tune_all_epilogues.yaml` | Benchmark YAML (overrides `--fast`). |
| `--fast` | (off) | Use `tune_all_epilogues_fast.yaml` instead of the full YAML. |
| `--chip CHIP` | `gfx950` | GPU architecture. |
| `--out-dir DIR` | `/tmp/tune_all_epilogues_<chip>` | Output root. |
| `--library-format yaml\|msgpack` | `msgpack` | Device library format. |
| `--client PATH` | (none) | tensilelite-client binary; enables smoke tests for all three variants. |

### Expected output

- At least 3 LibraryLogic YAMLs in `<out-dir>/3_LibraryLogic/`.
- One merged `.co` plus `.dat` or `.dat.zlib` in `<out-dir>/library/library/<chip>/`.
- With `--client`: three `PASS:` lines followed by `Results: 3 passed, 0 failed`.

### Adding a new tile

Edit `ForkParameters.MatrixInstruction` in the YAML. Keep wg0 (index 7) ≤ 2 for
variant 2 (PartialRMS+residual). `DepthU: 64` is pinned for all subtile epilogue
kernels and must not change.

## Subtile bf16 PartialRMS tuning (8192×8192×8192)

`tune_subtile_bf16_prms_8192.yaml` and `tune_subtile_bf16_prmsq_8192.yaml` sweep
the full tile/DepthU/StreamK/PGR space for PartialRMS and PartialRMSQuant
respectively at the fixed shape 8192×8192×1×8192 on gfx950.

```bash
# Run both sweeps (PartialRMS then PartialRMSQuant).
epilogues/scripts/run_subtile_bf16_tune.sh

# PartialRMS only.
epilogues/scripts/run_subtile_bf16_tune.sh --variant prms

# PartialRMSQuant only.
epilogues/scripts/run_subtile_bf16_tune.sh --variant prmsq

# Custom venv, client, and output directory.
epilogues/scripts/run_subtile_bf16_tune.sh \
    --variant both \
    --venv ~/.tensile-epilogues \
    --client build_tmp/tensilelite/client/tensilelite-client \
    --out-dir /tmp/my_tune_out
```

### `run_subtile_bf16_tune.sh` flags

| Flag | Default | Description |
|------|---------|-------------|
| `--variant prms\|prmsq\|both` | `both` | Which YAML(s) to run. |
| `--chip CHIP` | `gfx950` | GPU architecture (informational; gfx950 required). |
| `--venv PATH` | `~/.tensile-epilogues` | Venv root; `$PATH/bin/python` is used. |
| `--client PATH` | `build_tmp/tensilelite/client/tensilelite-client` | tensilelite-client binary. |
| `--out-dir DIR` | `epilogues/out/` | Output root; one subdirectory per YAML. |

### Key YAML constraints

- Both YAMLs use `DepthU: [64, 128]` (DU=192 is excluded for MT128×128 — it
  exceeds the gfx950 LDS limit of 163840 bytes at that tile size).
- `PartialRMSQuant: True` must appear in **both** the `OperationType` block
  (sets the host dispatch predicate) and `ForkParameters` (drives the emitter).
  The anchor `&prms_pt` in `tune_subtile_bf16_prmsq_8192.yaml` carries both.
- `StreamKForceDPOnly: [1]` is required for StreamK=3 kernels and is harmless
  (cleared automatically) for StreamK=0 kernels.
- `NumElementsToValidate: 256` in `debug_prmsq_single.yaml` — use this for
  quick single-kernel validation; the full tuning YAMLs use `0` to skip the
  Tensile generic GEMM reference (which cannot validate fused epilogue outputs).

### Reference results on gfx950 (2026-07-29)

Shape: 8192×8192×1×8192, BF16, TN, PartialRMS / PartialRMSQuant.

| Variant | Best GFlops | Best µs | Winner tile | DU | SK | PGR |
|---------|------------:|--------:|-------------|----|----|-----|
| PartialRMS | 1,324,060 | 829 | MT=256×384, MIWT=8×12 | 128 | 0 | 2 |
| PartialRMSQuant | 1,300,080 | 846 | MT=256×384, MIWT=8×12 | 128 | 0 | 2 |

PartialRMSQuant adds ~1.8% overhead over PartialRMS for the amax epilogue.
Both sweeps benchmarked 102 kernels, all PASSED.

## Integration tests

```bash
epilogues/scripts/test_client_partialrms.sh \
    --chip gfx950 --client build_tmp/tensilelite/client/tensilelite-client

epilogues/scripts/test_client_partialrms_residual.sh \
    --chip gfx950 --client build_tmp/tensilelite/client/tensilelite-client
```

Both scripts accept `--chip CHIP`, `--client PATH`, and `--out-dir DIR`.

### `test_client_partialrms.sh`

Builds a library from `gemm_partial_rms_k1_rowmajor.yaml`, then runs two
client invocations at shape `4096,4096,1,4096`:

1. `--use-partial-rms` (no residual add).
2. `--use-partial-rms --partial-rms-residual-add`.

### `test_client_partialrms_residual.sh`

Builds the same library from `gemm_partial_rms_k1_rowmajor.yaml`, then sweeps
M ∈ {256, 255, 1024, 4096, 100, 1001} at N=4096, K=4096 with
`--use-partial-rms --partial-rms-residual-add`. The shapes cover full M tiles
(M = MT1, 4×MT1, 16×MT1), a partial M tile (255 = MT1 − 1), and
non-power-of-2 values.

### Expected output

Each shape produces `PASS: <label>`. The script exits with a
`Results: N passed, 0 failed` summary and returns non-zero on any failure.

## Unit tests

```bash
source ~/.tensile/bin/activate
python -m pytest epilogues/unittests -v
```

`epilogues/unittests/conftest.py` inserts the tensilelite root into `sys.path`
automatically, so no `PYTHONPATH` override is needed.

### Coverage

Two test suites parametrised directly from the benchmark YAMLs:

| File | What it tests |
|------|--------------|
| `test_gemm_partial_rms.py` | K1 output accuracy: D (bf16) and partialBuf (f32). |
| `test_gemm_partial_rms_epilogue.py` | End-to-end K1 → row_div pipeline (full RMSNorm). |

A gfx950 GPU plus `amdgpu_exec` and `ml_dtypes` are required; tests skip
automatically on other chips.

## Benchmarks

```bash
# Col-major K1 + partial_rms_epilogue pipeline.
python epilogues/bench/bench_gemm_rms.py --M 4096 --N-hidden 4096 --K 4096

# Row-major K1 + row_div pipeline.
python epilogues/bench/bench_gemm_rmsnorm.py --M 4096 --N-hidden 4096 --K 4096

# Standalone row_div kernel.
python epilogues/bench/bench_row_div.py --m 4096 --n 4096 --n-d 32
```

### `bench_gemm_rms.py` flags

| Flag | Default | Description |
|------|---------|-------------|
| `--M` | 4096 | Number of token rows. |
| `--N-hidden` | 4096 | Hidden dimension (free dimension of the GEMM). |
| `--K` | 4096 | Contraction dimension. |
| `--wg-n` | 2 | MIWaveGroup[1]; selects MT1. |
| `--config` | (yaml default) | Path to the col-major K1 YAML config. |
| `--eps` | 1e-5 | RMSNorm epsilon. |
| `--warmup` | 3 | Warm-up iterations before timing. |
| `--iters` | 10 | Timed iterations. |
| `--no-verify` | (off) | Skip end-to-end correctness check. |
| `--chip` | (auto-detect) | Override the GPU architecture string. |

### `bench_gemm_rmsnorm.py` flags

| Flag | Default | Description |
|------|---------|-------------|
| `--M` | 4096 | Number of token rows. |
| `--N-hidden` | 4096 | Hidden dimension. |
| `--K` | 4096 | Contraction dimension. |
| `--wg-n` | 2 | MIWaveGroup[1]; selects MT1. |
| `--eps` | 1e-5 | RMSNorm epsilon. |
| `--warmup` | 3 | Warm-up iterations. |
| `--num-iters` | 10 | Timed iterations. |
| `--no-verify` | (off) | Skip pipeline correctness check. |
| `--chip` | (auto-detect) | Override the GPU architecture string. |
| `--config` | (yaml default) | Path to the row-major K1 YAML config. |

### `bench_row_div.py` flags

| Flag | Default | Description |
|------|---------|-------------|
| `--m` | 4096 | Row count. |
| `--n` | 4096 | Column count of C (bf16 matrix). |
| `--n-d` | 4096 | Number of partial-sum tiles (columns of D). |
| `--block-size` | 128 | Columns processed per thread block (grid_dim_y = n/block_size). |
| `--inv-d` | 1/n_d | Reciprocal of hidden dimension. |
| `--eps` | 1e-5 | RMSNorm epsilon. |
| `--warmup` | 3 | Warm-up iterations. |
| `--num-iters` | 10 | Timed iterations. |
| `--no-verify` | (off) | Skip output correctness check. |
| `--chip` | (auto-detect) | Override the GPU architecture string. |

## Benchmark YAML structure

Annotated skeleton for a four-variant YAML:

```yaml
GlobalParameters:
  MinimumRequiredVersion: 5.0.0
  NumElementsToValidate: 0   # fused epilogue side outputs are not validated by Tensile's generic GEMM reference

BenchmarkProblems:
  # One [ProblemType, BenchmarkProblemSizeGroup] pair per variant.

  # --- Variant 0: plain GEMM ---
  -
    - # ProblemType
      OperationType: GEMM
      DataType: b               # bf16
      DestDataType: b
      ComputeDataType: s        # fp32 accumulation
      HighPrecisionAccumulate: True
      TransposeA: True          # TN layout
      TransposeB: False
      UseBeta: True
      Batched: True
      StridedBatched: True
      GroupedGemm: False
      UseBias: 0
      UseScaleAB: ""
      UseScaleCD: False
      UseScaleAlphaVec: 0
      Sparse: 0

    - # BenchmarkProblemSizeGroup
      BenchmarkCommonParameters:
        - KernelLanguage: ["Assembly"]
      ForkParameters:
        - MatrixInstruction:
            - [16, 16, 32, 1, 1, 4, 4, wg0, wg1]   # indices 7,8 set wg0,wg1
        - StreamK: [3]
        - StreamKForceDPOnly: [1]
        - StreamKAtomic: [0]
        - GlobalSplitU: [1]
        - DepthU: [64]              # pinned for all subtile epilogue kernels
        # ... other fork params
      BenchmarkFinalParameters:
        - ProblemSizes:
            - Exact: [4096, 4096, 1, 4096]   # [free0, free1, batch, bound]

  # --- Variant 1: PartialRMS, no residual add ---
  -
    - # ProblemType — add epilogue flags here
      # (same base fields as above, plus:)
      UsePartialRMS: True
      PartialRMSResidualAdd: False

    - # BenchmarkProblemSizeGroup
      ForkParameters:
        - MatrixInstruction:
            - [16, 16, 32, 1, 1, 4, 8, wg0, wg1]
        - UseSubtileImpl: [True]
        - PartialRMS: [True]              # solution-level flag (mirrors UsePartialRMS)
        - PartialRMSResidualAdd: [False]  # solution-level flag
        - DepthU: [64]
        # ...

  # --- Variant 2: PartialRMS + residual add ---
  -
    - # ProblemType
      UsePartialRMS: True
      PartialRMSResidualAdd: True       # wg0 <= 2 required

    - ForkParameters:
        - MatrixInstruction:
            - [16, 16, 32, 1, 1, 4, 8, wg0, wg1]   # wg0 must be <= 2
        - UseSubtileImpl: [True]
        - PartialRMS: [True]
        - PartialRMSResidualAdd: [True]
        - DepthU: [64]
        # ...

LibraryLogic:
  ScheduleName: "my_epilogue"
  DeviceNames: ["Device 74a1", "Device 75a0", "Device 75b0", "Device 75a2",
                "Device 75b2", "Device 75a3", "Device 75b3", "Device 75a8", "Device 75b8"]
  ArchitectureName: "gfx950"
```

### Key constraints

- `DepthU: 64` is pinned for all subtile epilogue kernels.
- `ProblemType` flags (`UsePartialRMS`) determine the runtime dispatch predicate.
- Fork-level `PartialRMS` and `PartialRMSResidualAdd` are solution-level
  parameters that must match the `ProblemType` flags.
- MatrixInstruction index 7 (wg0) must be ≤ 2 for PartialRMS+residual.

## Adding a new epilogue

1. **ProblemType flag** — add to `Tensile/SolutionStructs/Problem.py`
   (`_defaultProblemType`, `__str__`, `__eq__`) and
   `Tensile/SolutionStructs/Solution.py` (solution parameters); validate in
   `Tensile/Common/ValidParameters.py`.

2. **Kernel codegen** — create a new
   `Tensile/Components/Subtile/Subtile<Name>Emit.py`; model it on
   `SubtilePartialRMSEmit.py`.

3. **C++ runtime** — thread the flag through
   `include/Tensile/ContractionProblem.hpp` (tensor enum, getter/setter),
   `include/Tensile/ContractionSolution.hpp` (SizeMapping field),
   `include/Tensile/ContractionProblemPredicates.hpp` (`UseXxxEqual` predicate),
   the serialisation headers, and `src/ContractionSolution.cpp` (kernel arg
   append).

4. **Tests and YAML** — add a `BenchmarkProblems` entry with the new flag to
   `tune_all_epilogues.yaml` and `tune_all_epilogues_fast.yaml`; add a
   `scripts/test_client_<name>.sh` integration test; add a
   `unittests/test_gemm_<name>.py` unit test.

## Design docs

See `epilogues/docs/` for deeper notes. `TUNING_PIPELINE.md` is the most
detailed and covers the full benchmark-to-library workflow end-to-end.
Other docs cover hipBLASLt integration status, kernel selection logic, library
creation steps, and the PartialRMS C++ extension points.
