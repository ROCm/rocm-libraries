# Nightly random GEMMs

The runner in [run_random_hipblaslt_gemms.py](../../../.github/scripts/run_random_hipblaslt_gemms.py)
executes GEMMs through rocJITsu entirely on CPUs. No physical GPU is required.
Bench chooses its kernel through the ordinary heuristic; the runner does not
read solution metadata or select solution indices. Nightly workflow integration
is pending.

## Provisional sampling

Iterations start at 1. Iteration `i` uses `M=N=K=i`, alternating FP32 and BF16
storage, starting with FP32. Computation is FP32, with no transposes, one batch,
alpha 1 and beta 0. This sequence is a placeholder for the eventual distribution.
The separate `sample_gemm(iteration, rng)` function is the place to change it.

The script accepts a seed, but this provisional sampler does not use it.
Execution and logging do not draw random values. Increasing the case count
preserves the existing prefix; `--case` reproduces one 1-based iteration.

## Running

Supply the matching ROCm runtime/device library, bench client with its converter
and template, and a rocJITsu executable with race and logging plugins. The Python
interpreter needs PyYAML for the bench converter. The config must describe CPU
emulation, such as rocJITsu's `gfx942_cdna3_kmd.json`.

```sh
HIPBLASLT_TENSILE_LIBPATH=/path/to/library/gfx942 python .github/scripts/run_random_hipblaslt_gemms.py \
    --bench /path/to/hipblaslt-bench --rocjitsu /path/to/rocjitsu \
    --config /path/to/gfx942_cdna3_kmd.json --count 100 --seed 0 \
    --output /path/to/results
```

`--count` defaults to 100 and `--timeout` defaults to 180 seconds per command.
`--dry-run` writes the inputs without requiring bench or rocJITsu. The output
directory must be new so stale detector logs cannot affect results. Reproduce
one case with the same count/seed and `--case 42` in another output directory.

## The loop

For each sampled GEMM, the script writes one bench YAML record, converts it with
the matching bench template, then runs bench under rocJITsu. The active Python
interpreter performs conversion, avoiding bench's build-time Python dependency.
Each case has its own input, command, rocJITsu config, conversion/bench/detector
logs and result. Execution is sequential, with a separate process per case.

The input enables numerical assertions and all-close checking. A pass requires
a successful exit, one numerical result matching the input, and the selected
GEMM kernel in the race-detector log, with no race reports or rocJITsu warnings.
The runner uses zero warmup iterations and one timed iteration; validation also
executes the GEMM. The test does not require exactly one dispatch.

Individual case failures and timeouts are counted, and later cases continue.
The final `summary.json` reports requested/sampled counts, passed, failed and
blacklisted totals. Any failure makes the script exit unsuccessfully. An
incomplete run is marked `complete: false`.

Initially there are no known kernel races and the rocJITsu blacklist is empty.
Every case executes, every race report is a failure, and `blacklisted` is zero.
Confirmed emulator or detector bugs can later be tracked by complete GEMM
parameters and a tracker. The runner never replaces rejected inputs or retries
failures. Saved inputs reproduce earlier runs even after the sampler changes;
the chosen kernel can change when the library changes.

Focused runner tests cover sequence/reproduction, full-loop failure counting,
missing/invalid numerical results, missing race instrumentation and timeouts:

```sh
python -B -m unittest discover -s .github/scripts/tests -p test_random_hipblaslt_gemms.py
```
