# MIOpen gtest results — `build-flagon`

Date completed: 2026-05-27
Build under test: `build-flagon` (`MIOPEN_ENABLE_HIPDNN_WRAPPER=ON`)
Hardware: AMD Instinct MI300X (gfx942), ROCm 7.13
Wall clock: 15:31:21 → 20:25:25 UTC (≈4h 54m, single-threaded sequential)

The build uses `MIOPEN_TEST_DISCRETE=ON`, so there is no single
`miopen_gtest` binary — gtest content is split across **265 discrete
`test_*` executables** in `build-flagon/bin/`. Source for each lives under
`test/gtest/*.cpp`. All 265 were executed (7 in the first pass with the
v1 runner, 258 in the resume pass with the v2 runner).

## Top-line numbers

| Metric | Value |
| --- | ---: |
| Binaries executed                       | 265 / 265 |
| Binaries clean (exit 0, no failures)    | 261 |
| Binaries with failures                  | 2 |
| Binaries hit the 1200 s timeout         | 2 |
| Individual gtest cases run              | 58,039 |
| Cases passed                            | 53,516 |
| Cases skipped                           | 4,520 |
| Cases failed                            | **4** |

The 4,520 skips are expected — they're guards in the test harness for
unsupported architectures / dtypes / configs (e.g. `test_rnn_seq_api`
skips 2,560 of 4,608 cases on gfx942, `test_bad_fusion_plan` skips all 8).

## Failures (4 cases across 2 binaries)

### `test_smoke_tuning_policy` — 2 failures (likely wrapper-induced)

Both failing tests capture stderr from a `miopen{Get,Set}TuningPolicy()`
call and assert the captured text contains the public function name. The
build under test routes through the wrapper, so the impl's auto-generated
function-entry log emits `_impl`-suffixed names — the substring check
misses by exactly that suffix:

```
test/gtest/smoke_tuning_policy.cpp:105: Failure
  Expected: has substring " miopenGetTuningPolicy("
    Actual: "MIOpen(HIP): miopenStatus_t miopenGetTuningPolicy_impl(miopenHandle_t, miopenTuningPolicy_t *){\n…"
```

Same shape for `TestSetApiLogged`. This is a real wrapper-related test
breakage, not noise — the test was written against the un-wrapped API
name and the wrapper's pass-through changes the logged symbol. Worth
fixing as part of the shim work (either rename in the wrapper's
`_impl` symbols' `MIOPEN_LOG_FUNCTION` output, or relax the assertion to
accept the `_impl` form).

The other 4 tests in this binary pass.

### `test_db_sync` — 2 failures (environmental, unrelated to wrapper)

```
db_sync.cpp:547: Failure
  C++ exception: "filesystem error: cannot get file size: No such file or
  directory [.../build-flagon/share/miopen/db/gfx942.kdb]"
```

The pre-built `gfx942.kdb` kernel database isn't installed in this build
tree. Affects `CPU_DBSync_NONE.KDBTargetID` and one parameterized
`StaticFDBSync/3` case. The remaining 4 cases in the binary skip (also
file-presence guards).

## Timeouts (2 binaries, runner killed at 1200 s)

Both tests were actively making progress when the timeout fired — they're
not hung, they're just enormous suites:

| Binary | Tests started | Tests passed before kill |
| --- | ---: | ---: |
| `test_lrn`      | 123 | 122 (still running #123) |
| `test_soft_max` | 267 | 266 (still running #267) |

Neither produced a `[ FAILED ]` line. To get clean numbers either raise
the per-binary timeout (e.g. `timeout 3600`) or filter to a subset via
`--gtest_filter`.

## Largest suites (top 10 by test count)

| Binary | Ran | Pass | Fail | Skip |
| --- | ---: | ---: | ---: | ---: |
| `test_tensor_reorder`        | 7,452 | 7,452 | 0 | 0 |
| `test_ternary_tensor_ops`    | 7,200 | 7,200 | 0 | 0 |
| `test_binary_tensor_ops`     | 5,143 | 5,143 | 0 | 0 |
| `test_rnn_seq_api`           | 4,608 | 2,048 | 0 | 2,560 |
| `test_tensor_transform`      | 3,270 | 3,270 | 0 | 0 |
| `test_tensor_api`            | 3,042 | 3,042 | 0 | 0 |
| `test_w_supertensor`         | 1,944 | 1,944 | 0 | 0 |
| `test_gpu_reference_kernel`  | 1,870 | 1,870 | 0 | 0 |
| `test_reduce`                | 1,625 | 701   | 0 | 924 |
| `test_bn_infer`              | 1,308 | 1,308 | 0 | 0 |

## Slowest binaries (top 10 by wall time)

| Binary | Duration |
| --- | ---: |
| `test_soft_max`             | 1200 s (timeout) |
| `test_lrn`                  | 1200 s (timeout) |
| `test_reduce`               | 1171 s |
| `test_bn_activ_infer`       | 901 s |
| `test_bn_infer`             | 856 s |
| `test_na_train_find2`       | 815 s |
| `test_bn_fwd_train`         | 744 s |
| `test_na_train`             | 734 s |
| `test_layernorm`            | 602 s |
| `test_bn_bwd`               | 572 s |

## Artifacts

All under `perf-results/gtest-flagon/`:

- `_all_gtests.txt` — canonical list of all 265 binaries.
- `_remaining_at_resume.txt` — the 258 binaries fed to the v2 runner.
- `_run_gtests.sh` / `_progress.log` / `_summary.tsv` — v1 runner (first
  7 binaries). The v1 summary's pass/fail/skip columns are bogus (the
  `--gtest_brief=1` flag suppressed the per-test lines its grep relied on).
- `_run_gtests_v2.sh` / `_progress_v2.log` / `_summary_v2.tsv` — v2 runner
  (remaining 258). Drops `--gtest_brief=1` and parses the `[==========]` /
  `[ PASSED ]` / `[ SKIPPED ]` / `[ FAILED ]` summary lines for accurate
  counts. **This is the source of truth for runner-side counts.**
- `test_<name>.log` — raw stdout+stderr for each of the 265 binaries.

The numbers in this document were re-aggregated directly from the 265
`test_*.log` files (not from `_summary*.tsv`), so they're consistent
regardless of which runner produced a given log.

## Re-run hints

- Just the failing binaries: `build-flagon/bin/test_smoke_tuning_policy`
  and `build-flagon/bin/test_db_sync` (each <2 s).
- Just the timeout victims with a larger budget:
  `timeout 3600 build-flagon/bin/test_lrn` and the same for `test_soft_max`.
- Same suite against `build-flagoff` (to confirm `test_smoke_tuning_policy`
  passes there and conclusively pin it on the wrapper): swap the BINDIR in
  `_run_gtests_v2.sh` to `build-flagoff/bin` and re-run.
