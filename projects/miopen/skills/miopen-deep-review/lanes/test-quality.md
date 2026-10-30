# Lane: test quality (`TESTQ`)

Judge whether each added or changed test, and each change to test configuration, would fail on a plausible bug. Supply the `M1:` and
`M3:` lines ([../policy.md](../policy.md) *M-rules*; format in [../findings.md](../findings.md) *Lane report*).

## Checks

- **Defends behavior.** Apply [../policy.md](../policy.md) *Test smells* to every added or changed test: it asserts what a
  consumer observes (results against a reference, statuses, the solver chosen, emitted artifacts), not plumbing, source text or
  incidental defaults.
- **Mutation-sensitive.** For each changed product branch, name the mutation (flip a comparison, drop a check, swap operands,
  widen a tolerance, return `true` from `IsApplicable`) and the test that should kill it. A surviving candidate is a `test-gap` finding.
- **Reaches the solver.** A convolution or fusion test that goes through find or immediate mode passes with whichever applicable
  solver wins. A test meant to cover one solver pins it (`MIOPEN_DEBUG_FIND_ONLY_SOLVER`, a solver-specific `GPU_UnitTestConv*`
  fixture, or calling the solver directly) and asserts it was applicable; otherwise a broken solver that falls back passes.
- **Regression test for fixes.** A defect fix has a test that fails without the fix (M1). Write `M1: UNVERIFIED <reason>` until an
  executed run shows the test failing at `$BASE` (or at `$HEAD` with the fix reverted); reading what the test exercises against
  what the fix changed does not confirm M1. That run comes from `RUN` ([build-validate.md](build-validate.md) step 7).
- **No weakened tests.** Every disabled, skipped, deleted or loosened test needs a stated reason (M3 allows no waiver for
  greening CI). In MIOpen much of the weakening happens outside test source, so diff these too:
  - `projects/miopen/test/gtest/test_categories.yaml`: a pattern added to a tier's `exclude`, to an `exclude_gpu_*` block, or moved
    from `quick`/`standard` to only `comprehensive`/`full`.
  - `add_gtest_negative_filter(...)` lines in `projects/miopen/test/gtest/CMakeLists.txt`, and `MIOPEN_TEST_*` option defaults in
    `projects/miopen/test/CMakeLists.txt` that drop a suite from the default selection.
  - A case moved from the `Smoke` instantiation to `Full` (the default build runs only `Smoke*` and `*NONE*` cases unless
    `MIOPEN_TEST_ALL=ON`).
  - `DISABLED_`, `GTEST_SKIP`, early `return` on an arch or data-type check, a raised tolerance or error threshold, or a reduced
    shape list.
- **Skips and architectures** (green is not coverage, [../policy.md](../policy.md) *Review rules*):
  - Arch guards (`IsTestSupportedByDevice(Gpu::…)`, `miopen::StartsWith(dev, ...)`, `MIOPEN_TEST_GFX*`) and `exclude_gpu_*` blocks are
    correct, and the test still executes on an arch CI runs. A test that can only skip in CI is a finding.
  - A case only the `comprehensive` or `full` patterns of `test_categories.yaml` select never runs in PR CI; it counts only for
    the nightly exposure `ROCK` reports.
- **Determinism and isolation.** Fixed seeds, no timing dependence, no shared global state, no reliance on test order, on a warm
  user database or kernel cache (`MIOPEN_USER_DB_PATH`, `MIOPEN_CUSTOM_CACHE_DIR`), or on the developer's machine.
- **Tolerances and references.** Each tolerance is justified by data type and reduction size; the reference is independent of the
  code under test (the CPU reference under `projects/miopen/test/cpu_*.hpp` or the GPU reference kernel, not the solver itself).
- **Conventions.** Names pass `projects/miopen/test/gtest/check_names.py` (`<Smoke|Standard|Full|Perf|Unit>/<CPU|GPU>_<Suite>_<datatype>`, no `GPU` with `NONE`, unique across `miopen_gtest`)
  and follow `projects/miopen/test/gtest/README.md`; tiers follow `test_categories.yaml`.

## Reconcile with `RUN` (`TESTQ-f1`)

Read the `TESTQ` and `RUN` reports and rewrite the `M1:` line from `RUN`'s regression-test rows under **Checked, no finding**: `PASS` when
each regression test fails at `$BASE` (or with the fix reverted) and passes at `$HEAD`; `FAIL` when one passes without the fix, so it
is not a regression test (file the finding); `UNVERIFIED` when `RUN` could not run it. Write only this line and any new findings.
