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

These are initial profiles, not an enumeration of every supported combination.
The MX client mode is `1001`, corresponding to
`HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_32_8_EXT`. Its scales use the gfx950
preswizzled layout; the documented K-multiple-of-128 restriction for legacy
`VEC32_UE8M0` mode does not describe this extension. Local CPU probes passed
MXFP8 and MXFP4 at K=32, 64, 96 and 128 with M=32 and N=64.

Alpha is sampled from `{-2, -1, 0.5, 1, 2}` and beta from
`{-2, -1, 0, 0.5, 1, 2}`. Epilogues sample none, bias, ReLU, bias+ReLU, GELU
and bias+GELU, with FP32 bias. The fixed seed reproduces the sequence across
thread counts. Alpha stays nonzero so every case exercises multiplication.

The numerical configurations follow the project's
[type documentation](reference/data-type-support.rst),
[combination tables](reference/api-reference.rst) and gfx950 MX tests.
Bench's normal heuristic checks the complete configuration against the loaded
library. A reported lack of a heuristic solution is counted as `unsupported`
and retains its input and logs; cases are never resampled. The distribution
aims to produce executable GEMMs. Its minimum dimension is 32, so random
samples cannot have M, N or K equal to 1. Missing solutions remain failures
to investigate when assessing or extending a sampling profile.
The public `hipblasLtMatmulAlgoGetHeuristic` query checks one set of descriptors;
there is no public query that enumerates all legal numerical combinations.
Alignment and a documented type pairing do not guarantee that the loaded
library contains a kernel for every sampled configuration.
The [individual observation table](random-gemm-observations.md) records the
inputs and failures from local exploration, with complete machine-readable
benchmark inputs alongside it.

## Running

Supply the matching ROCm library, bench client and rocJITsu CPU-emulation config.
The config must emulate gfx950. The Python environment needs PyYAML for bench
input conversion, and the bench build must enable MX data generation.

```sh
HIPBLASLT_TENSILE_LIBPATH=/path/to/library/gfx950 python .github/scripts/run_random_hipblaslt_gemms.py \
    --bench /path/to/hipblaslt-bench --rocjitsu /path/to/rocjitsu \
    --config /path/to/gfx950_cdna4_kmd.json --count 100 --threads 8 \
    --output /path/to/results
```

`--threads` controls the number of concurrent GEMMs; each rocJITsu process uses
one emulator worker. Each GEMM has one YAML record and its own logs, so a failure
has a separate reproduction. `--stop-on-failure` stops workers from taking new
cases after a failure, including a missing solution; cases already running
finish. The CI callers attempt the full loop without early stopping.

The summary counts passes, failures, unsupported cases and known failures, and
records elapsed time and sampled families, transposes and epilogues.
It also records the random missing-solution rate as a coverage metric, using
completed random attempts as the denominator. A list-only run has no random
rate. Explicit regression inputs can deliberately include size-1 dimensions.
A pass requires successful numerics and race detection. Known-failure exclusions
start empty, and every race report fails. Inputs and results are saved under the
output directory. Use `--help` for all options.

`--explicit-cases` loads a JSON list of required inputs in addition to the random
samples. Each entry has a stable `id`, a description and a complete `problem`.
Required cases never consume random draws. The summary
reports `random` and `explicit` counts separately; `sampled` counts only random
inputs, while `requested` includes both. Each run writes `cases.md` and
`cases.json`, listing every input and observed result. Explicit tests retain the
same numerical assertions and race checks as random tests; there are no expected
failure exemptions.

The checked-in [required cases](../../../.github/scripts/hipblaslt_explicit_gemms.json)
include five numerical-validation reproducers and one BF16 TN input,
256×160×192 with one batch, matching the reduced shape from
[the reported prefetch race](https://github.com/ROCm/rocm-libraries/pull/14147).
The heuristic may select a different kernel, so these check the shapes rather
than guaranteeing coverage of that particular custom schedule. Results record
the selected kernel names. Use `--count 0 --explicit-cases ...` to run the list
alone, or `--explicit-case ID --explicit-cases ...` to replay one named input.
`--case N` continues to replay only the Nth random sample. Early stopping also
applies to explicit tests; omit `--stop-on-failure` to attempt the full list.

## CI and pre-merge validation

The existing Multi-Arch workflows call the CPU
[race-check workflow](../../../.github/workflows/therock-rocjitsu-race-check-linux.yml)
after the Linux build publishes its artifacts. The per-PR run samples 100 GEMMs;
nightly samples 1,000. Both use gfx950, seed 0 and eight concurrent GEMMs, so
the per-PR inputs are the first 100 nightly inputs. The job fetches the matching
BLAS runtime, device library and bench test payload from that workflow run.
The emulator is pinned independently to a revision validated with these types.
Both callers also run the same six required cases, giving 106 per-PR inputs and
1,006 nightly inputs. The known numerical reproducers currently fail strict
validation and remain visible in the advisory job; no blacklist hides them.

These GEMM jobs are advisory: a failed test step and its reports remain visible,
but they do not change the required Multi-Arch summary. No known-failure entries
are enabled. Every sample is attempted. Missing solutions, race reports and
numerical failures make the test step fail. Runner unit tests execute on a
small GitHub-hosted runner when the runner or its workflows change.

The race-check workflow also supports manual dispatch with `artifact_run_id`,
`random_gemm_count`, `random_gemm_seed`, `random_gemm_threads` and `rocjitsu_ref`.
Choose a completed rocm-libraries Multi-Arch run that published gfx950 artifacts.
Manual replay tests the scripts from the selected workflow ref against that
artifact run; it does not rebuild hipBLASLt. For a pre-merge end-to-end check,
push the draft branch so its per-PR caller builds and tests the merged tree.
GitHub requires a workflow to exist on the default branch before its manual
dispatch is available.

For local validation of the same driver, fetch the artifact payload with
TheRock's `install_rocm_from_artifacts.py --blas --tests`, and check out the
workflow's pinned rocm-systems revision. Set `ROCM_PATH`, `ROCJITSU_SOURCE_DIR`,
`ROCJITSU_BUILD_DIR`, `RACE_REPORT_DIR`, `AMDGPU_FAMILIES=gfx950-dcgpu`,
`RANDOM_GEMM_COUNT`, `RANDOM_GEMM_SEED` and `RANDOM_GEMM_THREADS`, then run
`.github/scripts/run_rocjitsu_hipblaslt_race_check.sh`. Build and report paths
must be outside the source checkout. The driver builds the CLI and race/logging
plugins, then invokes the same Python runner used by CI.
