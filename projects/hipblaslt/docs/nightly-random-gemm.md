# Nightly random GEMMs

This prototype [runner](../../../.github/scripts/run_random_hipblaslt_gemms.py)
tests GEMMs through rocJITsu on CPUs, checking numerics and detecting races.
No physical GPU is required. In this initial prototype, bench chooses its
kernel through the ordinary heuristic. Nightly workflow integration is pending.

To keep sampling simple, iteration `i` starts at 1 and uses `M=N=K=10+i`,
alternating FP32 and BF16 storage with FP32 computation. There is one batch,
no transpose, alpha 1 and beta 0. The final sampling distribution will replace
this sequence.

## Running

Supply the matching ROCm library, bench client and rocJITsu CPU-emulation config.
The Python environment needs PyYAML for bench input conversion.

```sh
HIPBLASLT_TENSILE_LIBPATH=/path/to/library/gfx942 python .github/scripts/run_random_hipblaslt_gemms.py \
    --bench /path/to/hipblaslt-bench --rocjitsu /path/to/rocjitsu \
    --config /path/to/gfx942_cdna3_kmd.json --count 100 --threads 8 \
    --output /path/to/results
```

`--threads` controls the number of concurrent GEMMs; each rocJITsu process uses
one emulator worker. Each GEMM has one YAML record and its own logs, so a failure
has a separate reproduction. `--stop-on-failure` stops workers from taking new
cases after a failure; cases already running finish.

The summary counts passes, failures and known failures, and records elapsed time.
A pass requires successful numerics and race detection. Known-failure exclusions
start empty, and every race report fails. Inputs and results are saved under the
output directory. Use `--help` for all options.
