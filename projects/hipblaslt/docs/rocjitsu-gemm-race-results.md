# rocJITsu GEMM race-test results

The [race-test runner](rocjitsu-gemm-race-testing.md) samples 100 GEMMs per PR
and 1,000 nightly with a fixed seed. It checks numerical validation and race
detection entirely on CPUs using gfx950 emulation.

The current 100-case per-PR workload passed locally through the shared CI
driver in 45.5 seconds, with no races, numerical failures, missing solutions
or emulator diagnostics. It used eight concurrent GEMMs, the
[matched gfx950 artifacts](https://github.com/ROCm/rocm-libraries/actions/runs/39535282903)
and rocJITsu `29656a8dcc05`. All inputs match the nightly prefix exactly.
These results come from local CPU execution; full hosted CI remains pending.

Failure-handling checks use injected faults in copies of a kernel selected
for an MXFP4 sample. The race control reports a race and the numerical control
fails validation; both produce failed case results and an unsuccessful runner
exit. The runner tests also check deterministic inputs, concurrent execution,
single-sample replay, early stopping, missing solutions, subprocess launch
failures and timeouts.
