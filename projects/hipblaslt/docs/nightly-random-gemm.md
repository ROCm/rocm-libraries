# Nightly random GEMMs

This prototype [runner](../../../.github/scripts/run_random_hipblaslt_gemms.py)
tests GEMMs through rocJITsu on CPUs, checking numerics and detecting races.
No physical GPU is required. In this initial prototype, bench chooses its
kernel through the ordinary heuristic. Nightly workflow integration is pending.

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
library. A reported lack of a solution is counted as `unsupported`, retains
its input and logs, and makes the run unsuccessful; cases are never resampled.
The public `hipblasLtMatmulAlgoGetHeuristic` query checks one set of descriptors;
there is no public query that enumerates all legal numerical combinations.
Alignment and a documented type pairing do not guarantee that the loaded
library contains a kernel for every sampled configuration.

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
cases after a failure; cases already running finish.

The summary counts passes, failures, unsupported cases and known failures, and
records elapsed time and sampled families, transposes and epilogues.
A pass requires successful numerics and race detection. Known-failure exclusions
start empty, and every race report fails. Inputs and results are saved under the
output directory. Use `--help` for all options.
