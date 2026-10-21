# Nightly random GEMMs: script design

Status: initial design. The sampling distribution and executable runner are pending.

The nightly test samples GEMM inputs from a fixed distribution and runs them with
`hipblaslt-bench`. A fixed seed gives the same ordered inputs on each night.
The bench client chooses a kernel through its ordinary heuristic. The script
does not read solution metadata or select solution indices.

## Script interface

Proposed location: `.github/scripts/run_random_hipblaslt_gemms.py`.

The script accepts the bench executable, seed, case count, per-case timeout and
output directory. Proposed defaults are seed 0 and 100 cases; the timeout will
be chosen once the workload sizes are known. A dry-run option writes inputs
without executing them. A case option reproduces one position in the sequence.

The caller supplies the ROCm environment, matching device library and
rocJITsu executable/configuration. rocJITsu executes GPU code under emulation
for race detection. Each case runs bench through rocJITsu with race reporting
enabled and a separate directory for its detector output. The runner does not
build clients or download libraries.

## Sampling and execution

Keep one sampling function separate from the runner:

```python
def sample_gemm(rng):
    # Return one complete bench input using the agreed distribution.
    ...

def main():
    rng = random.Random(args.seed)
    for case_number in range(args.count):
        problem = sample_gemm(rng)
        save_input(case_number, problem)
        if not args.dry_run:
            run_and_check(case_number, problem)
```

The distribution will define dimensions, storage and compute types,
transposes, scalars, batching, strides and any optional output operations.
There is no provisional distribution in this draft. Execution and logging
never draw from the sampling generator. Increasing the case count preserves
the existing prefix. Changing the distribution can change that sequence;
saved inputs remain the authoritative reproduction of an earlier run.

Run cases sequentially, with one bench process per case. Each input is a YAML
list containing one record with ordinary heuristic selection and one requested
algorithm.
Enable numerical checks and `norm_check_assert` in that record: the direct
command-line `--verify` path computes diagnostics but disables that assertion.
Use the matching bench client's `--yaml` conversion environment.

Use zero warmup iterations, fixed timing and one timed iteration initially.
Numerical validation also executes the GEMM. This avoids depending on a client
change to make zero timed iterations suppress execution. This test measures
correctness rather than performance, and does not require exactly one dispatch.

## Results and reproduction

Write the input before launching each case. Save its bench command and combined
stdout/stderr alongside it. Print progress with the case number and GEMM
parameters, so a stalled or failing case is identifiable in the job log.
Record the seed, sampling-code version, Python version and bench version once
for the run. The selected kernel may change when the library changes, even
when the sampled input stays fixed.

A pass requires a completed numerical result for the requested input as well
as a successful process exit. No-solution responses, missing results, numerical
failures and timeouts fail the run. Race reports also fail the run, even when
bench exits successfully. Do not silently resample rejected inputs or retry
failures. Stop at the first failure and preserve its files for reproduction.

## Next steps

1. Specify the distribution.
2. Implement the sampling function and sequential runner on this branch.
3. Check deterministic prefixes, reproduction and failure propagation with
   small inputs, then run a focused test through rocJITsu.
4. Add a nightly caller using the existing artifact setup and upload the input
   and result directory even when the job fails.
