# Nightly GEMM race detection with rocjitsu

The nightly job runs 100 sampled hipBLASLt GEMMs through rocjitsu with race
detection enabled. It executes gfx950 GPU instructions on CPUs and checks the
numerical output with `hipblaslt-bench`.

See [rocjitsu's race detector documentation](https://github.com/ROCm/rocm-systems/blob/6f5d9658a4b13656e0c1a3719d86a320c9206dbe/emulation/rocjitsu/docs/race-detector.md)
for how detection works and its coverage limits.

## Design

The test repeats three steps for each sample:

1. The [sampler](../clients/scripts/rocjitsu/rocjitsu_gemm_sampler.py) chooses the
   matrix dimensions, number formats and operation options.
2. The [runner](../clients/scripts/rocjitsu/rocjitsu_random_gemms.py) sends
   that input to `hipblaslt-bench` under rocjitsu. The library chooses the GPU
   function, or kernel, as it normally would. No explicit solution index is used
   to force a particular kernel.
3. The runner checks the numerical result and race report, then saves the input,
   selected kernel, commands, logs and outcome.

Every sample must complete numerical validation and execute with race detection.
A race, numerical failure, emulator warning or error, timeout or missing
solution fails the run. A missing solution is reported as `unsupported`: the
library did not find a kernel for that input. Samples are never replaced, so the
reports show how much of the workload actually ran.

The seed fixes the sequence of GEMM configurations. A given sample has the same input
regardless of how many workers run concurrently; worker assignment and completion
order can vary. Each GEMM runs in its own process and has its own report directory.
Sampling and execution are separate, so the distribution can be extended without
changing the runner.

## Current sampling

For a product of an M×K matrix and a K×N matrix, the output is M×N. M, N and K
are independent uniform draws from `{32, 64, ..., 480}`. These dimensions are
aligned to 32 to accommodate the low-precision formats used by this test.
Each sample contains one matrix multiplication, with no batching.

The six equally likely number-format families are FP32, FP16, BF16, FP8, MXFP8
and MXFP4. The MX formats store values in 8 or 4 bits and share a scale across
blocks of 32 elements. All families add intermediate products in FP32. The
sampler chooses input and output pairings based on the project's
[supported types](reference/api-reference.rst) and gfx950 tests; it covers a
subset of supported combinations.

Transposing swaps a matrix's rows and columns. The ordinary families sample all
four combinations of transposing the two input matrices. MX samples transpose
the first input only. Alpha, the product's scale, is chosen from
`{-2, -1, 0.5, 1, 2}`. Beta, the added matrix's scale, is chosen
from `{-2, -1, 0, 0.5, 1, 2}`. Alpha stays nonzero to exercise multiplication.
The optional operations after multiplication are chosen equally from none, bias,
ReLU, bias+ReLU, GELU and bias+GELU. A bias adds a vector to the result; ReLU and
GELU apply a function to each output value.

## Automated runs and reproduction

The nightly Multi-Arch workflow runs 100 samples with seed 0 and eight
concurrent GEMMs. It emulates gfx950, the GPU instruction set used by the MI350
family. With a fixed seed, every night checks the same inputs against the
latest library, so it works as a smoke test rather than a growing search.

The [CPU test job](../../../.github/workflows/therock-rocjitsu-race-check-linux.yml)
runs this as its `random-gemms` workload. It uses the library and benchmark
built by the same workflow, and
[its driver](../../../.github/scripts/run_rocjitsu_hipblaslt_gemm_race_checks.sh)
builds a fixed version of rocjitsu. It attempts every sample and uploads reports
even after a failure. The job is advisory: its failure is visible, but does not
fail the required Multi-Arch summary.

The same job can be dispatched manually with the ID of a completed Multi-Arch
run. It then tests that run's artifacts with the requested sample count, seed
and number of concurrent GEMMs.

To run locally, provide a matching library and benchmark, rocjitsu, and its
gfx950 configuration. The Python environment needs PyYAML, and the benchmark
build must enable MX data generation.

```sh
python3 projects/hipblaslt/clients/scripts/rocjitsu/rocjitsu_random_gemms.py \
    --client /path/to/hipblaslt-bench --rocjitsu /path/to/rocjitsu \
    --library-dir /path/to/library/gfx950 \
    --config /path/to/gfx950_mi355x_kmd.json --count 100 --workers 8 \
    --continue-on-failure --reports /path/to/new-results
```

Use `--case N` with the same seed to replay one sample. `--dry-run` lets you
inspect the sampled inputs before running them: it writes inputs and reports
without starting the benchmark or rocjitsu. These samples are marked `dry-run`,
so they do not count as passes. Without `--continue-on-failure`, the runner
stops starting new samples after a failure; already running samples finish.
Each run creates a new reports directory, with `summary.json` and `summary.md`
for totals and timings, `cases.md` for individual results and `kernels.json`
for the kernels each sample ran.
The runner saves partial totals during execution. If the job is interrupted,
`summary.json` retains those totals with `complete` set to `false`; completed
cases also retain their individual results and logs.

The runner's unit tests use a stand-in benchmark and emulator, so they need no
build or GPU. The
[component CI lane](../../../.github/workflows/component-ci-hipblaslt-rocjitsu-scripts.yml)
runs the tests in `projects/hipblaslt/clients/scripts/rocjitsu` on pull requests
that change that directory, and its result gates the required
`Component CI Summary`. Run them locally with
`python3 -m unittest discover -s projects/hipblaslt/clients/scripts/rocjitsu -p 'test_*.py'`.
