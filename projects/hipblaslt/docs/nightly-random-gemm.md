# Nightly random GEMMs

This prototype [runner](../../../.github/scripts/run_random_hipblaslt_gemms.py)
tests GEMMs through rocJITsu on CPUs, checking numerics and detecting races.
No physical GPU is required. In this initial prototype, bench chooses its
kernel through the ordinary heuristic.

The [sampler](../../../.github/scripts/sample_hipblaslt_gemms.py) has separate
functions for numerics, shapes, layouts, scalars and epilogues. M, N and K are
independent uniform draws from `{32, 64, ..., 480}`. Six equally likely families
cover FP32, FP16, BF16, FP8, MXFP8 and MXFP4, with FP32 computation and one batch.
Ordinary families sample all four transposes; MX uses TN and preswizzled E8M0
block scales. Outputs are chosen from each family's supported types.

| Family | A and B | C and D | Block scales |
| --- | --- | --- | --- |
| FP32 | FP32 | FP32 | None |
| FP16 | FP16 | FP16 or FP32 | None |
| BF16 | BF16 | BF16 | None |
| FP8 | E4M3 FP8 | BF16 | None |
| MXFP8 | E4M3 FP8 | FP32, FP16 or BF16 | E8M0 per 32 elements |
| MXFP4 | E2M1 FP4 | FP32, FP16 or BF16 | E8M0 per 32 elements |

The MX mode is `1001` (`HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_32_8_EXT`),
using gfx950's preswizzled scale layout. All dimensions are aligned to 32.
These profiles follow the project's [type tables](reference/api-reference.rst)
and gfx950 MX tests; they cover a useful subset of supported combinations.

Alpha is sampled from `{-2, -1, 0.5, 1, 2}` and beta from
`{-2, -1, 0, 0.5, 1, 2}`. Epilogues sample none, bias, ReLU, bias+ReLU, GELU
and bias+GELU, with FP32 bias. The fixed seed reproduces the sequence across
thread counts. Alpha stays nonzero so every case exercises multiplication.

The fixed distribution passed 5,000 local samples with no missing solutions.
That evidence applies to the tested library build: normal heuristic selection
can still return `error: NO solution found!` for another configuration or build.
The runner records that outcome as `unsupported`, retains its input and logs,
and fails the run. The missing-solution rate measures how many samples could
execute; it has no percentage cutoff. Cases are never replaced.
[Exploratory observations](random-gemm-observations.md) document individual
inputs and failures outside the default distribution.

## Running

Supply the matching ROCm library, bench client and rocJITsu CPU-emulation config.
The config must emulate gfx950. The Python environment needs PyYAML for bench
input conversion, and the bench build must enable MX data generation.

```sh
HIPBLASLT_TENSILE_LIBPATH=/path/to/library/gfx950 python .github/scripts/run_random_hipblaslt_gemms.py \
    --bench /path/to/hipblaslt-bench --rocjitsu /path/to/rocjitsu \
    --config /path/to/gfx950_mi355x_kmd.json --count 100 --threads 8 \
    --output /path/to/results
```

`--threads` controls the number of concurrent GEMMs; each rocJITsu process uses
one emulator worker. Each GEMM has one YAML record and its own logs, so a failure
has a separate reproduction. `--stop-on-failure` stops workers from taking new
cases after a failure, including a missing solution; cases already running
finish. The CI callers attempt the full loop without early stopping.

A pass requires successful benchmark numerical validation and evidence that
the selected GEMM ran with race detection. Races, numerical failures, missing
solutions and emulator diagnostics fail the run. There are no exclusions or
expected-failure exemptions; `known_failures` is reserved and stays zero.

`--explicit-cases` adds a JSON list with `id`, `description` and `problem` fields.
These cases do not consume random draws and use the same checks. Reports contain
separate random and explicit totals, plus sampled types, transposes, epilogues
and elapsed time. Each case retains its input, commands, logs and result;
`cases.md` and `cases.json` collect the inputs and outcomes for the whole run.

The checked-in [required cases](../../../.github/scripts/hipblaslt_explicit_gemms.json)
contain one BF16 TN input, 256×160×192 with one batch, matching the reduced shape from
[the reported prefetch race](https://github.com/ROCm/rocm-libraries/pull/14147).
The heuristic may select a different kernel, so this checks the shape without
guaranteeing coverage of that particular custom schedule. Results record
the selected kernel names. Use `--count 0 --explicit-cases ...` to run the list
alone, or `--explicit-case ID --explicit-cases ...` to replay one named input.
`--case N` continues to replay only the Nth random sample. Early stopping also
applies to explicit tests; omit `--stop-on-failure` to attempt the full list.

The separate [diagnostic cases](../../../.github/scripts/hipblaslt_diagnostic_gemms.json)
retain five observed numerical-validation failures near zero or FP16 overflow.
They are available through `--count 0 --explicit-cases` using that file, with
the same strict checks and no exemptions. They are outside the initial CI
workload because they exercise validation edge cases documented in the
[observation table](random-gemm-observations.md).

## CI and pre-merge validation

The existing Multi-Arch workflows call the CPU
[race-check workflow](../../../.github/workflows/therock-rocjitsu-race-check-linux.yml)
after the Linux build publishes its artifacts. The per-PR run samples 100 GEMMs;
nightly samples 1,000. Both use gfx950, seed 0 and eight concurrent GEMMs, so
the per-PR inputs are the first 100 nightly inputs. The job fetches the matching
BLAS runtime, device library and bench test payload from that workflow run.
The emulator is pinned independently to a revision validated with these types.
Both callers also run the same required BF16 case, giving 101 per-PR inputs
and 1,001 nightly inputs.

These GEMM jobs are advisory: a failed test step and its reports remain visible,
but they do not change the required Multi-Arch summary. No known-failure entries
are enabled. Every sample is attempted. Missing solutions, race reports and
numerical failures make the test step fail. Runner unit tests execute on a
small GitHub-hosted runner when the runner or its workflows change.

For local validation of the CI driver, fetch the BLAS and test artifacts with
TheRock's `install_rocm_from_artifacts.py --blas --tests`. Use the workflow's
pinned rocm-systems revision and set `ROCM_PATH`, `ROCJITSU_SOURCE_DIR`,
`ROCJITSU_BUILD_DIR`, `RACE_REPORT_DIR`, `AMDGPU_FAMILIES=gfx950-dcgpu`,
`RANDOM_GEMM_COUNT`, `RANDOM_GEMM_SEED` and `RANDOM_GEMM_THREADS`. Run
`.github/scripts/run_rocjitsu_hipblaslt_race_check.sh`; it builds the emulator and
plugins, then invokes the same Python runner. Build and report paths belong
outside the source checkout.

The reusable workflow also accepts manual dispatch with `artifact_run_id` and
the sampling settings. It tests the selected scripts against an existing
gfx950 artifact run without rebuilding hipBLASLt. GitHub exposes manual
dispatch only after a workflow exists on the default branch; the per-PR caller
provides the hosted check before merge.
