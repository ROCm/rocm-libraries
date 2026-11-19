# Lane: test quality (`TESTQ`)

Judge whether each added or changed test, and each change to test configuration, would fail on a plausible bug. Supply the `M1:` and
`M3:` lines ([../policy.md](../policy.md) *M-rules*; format in [../findings.md](../findings.md) *Lane report*).

## Checks

- **Defends behavior.** Apply [../policy.md](../policy.md) *Test smells* to every added or changed test: it asserts what a
  consumer observes (results, statuses, errors, emitted artifacts), not plumbing, source text or incidental defaults.
- **Mutation-sensitive.** For each changed product branch, name the mutation (flip a comparison, drop a check, swap operands,
  widen a tolerance) and the test that should kill it. A surviving candidate is a `test-gap` finding.
- **Regression test for fixes.** A defect fix has a test that fails without the fix (M1). Write `M1: UNVERIFIED <reason>` until an
  executed run shows the test failing at `$BASE` (or at `$HEAD` with the fix reverted); reading what the test exercises against
  what the fix changed does not confirm M1. That run comes from `RUN` ([build-validate.md](build-validate.md) step 7).
- **No weakened tests.** Every disabled, skipped, deleted or loosened test needs a stated reason (M3 allows no waiver for
  greening CI). In hipDNN most weakening happens outside test source, so diff these too:
  - Provider `--test-config` TOMLs (`dnn-providers/**/config/*.toml`): raised `[[tolerance_overrides]]`, new
    `[[validator_overrides]]` (allclose is the default; an `rms` override that buys slack is a loosening), new or wider
    `[[test_skips]]` (`filters`, `archs`, `platforms`). See `dnn-providers/integration-tests/docs/file-formats.md` § "Per-engine test config".
  - Support-claim sidecars removed or narrowed (`<bundle>.support.json` beside a single-graph bundle, `support.json` beside a
    sweep, under `dnn-providers/integration-tests/integration-test-bundles/`), and any `--enforce-support-claims=false`. `dnn-providers/integration-tests/docs/file-formats.md` § "Support-claim sidecars", `dnn-providers/integration-tests/docs/adding-tests.md` § "Updating support claims",
    `dnn-providers/integration-tests/docs/support-claim-enforcement.md`; the `verify-support-claims` pre-commit hook checks the sidecars.
  - `DISABLED_`, `GTEST_SKIP`, `SKIP_IF_ASAN()`; CMake that skips a test registration under an ASAN build
    (`if(BUILD_ADDRESS_SANITIZER OR THEROCK_SANITIZER ...)`); new patterns in `__asan_default_suppressions()`
    (`projects/hipdnn/test_sdk/src/AsanDefaultSuppressions.cpp`).
- **Skips and engines** (green is not coverage, [../policy.md](../policy.md) *Review rules*):
  - `SKIP_IF_NO_DEVICES()`, arch guards and TOML `[[test_skips]]` are correct, and the test still executes on an arch CI runs.
    A test that can only skip in CI is a finding.
  - Integration tests run with `--test-engine <ENGINE>`, which turns unsupported ops into SKIP instead of falling through to
    another engine (`dnn-providers/integration-tests/docs/adding-tests.md` § "Add an engine lane"; `dnn-providers/integration-tests/docs/running-tests.md`). A registration with no engine pinned is a finding. With
    `--enforce-support-claims` (default on), a skipped case its sidecar claims fails; a case with no claim stays a silent skip.
- **Determinism and isolation.** Fixed seeds, no timing dependence, no shared global state, no reliance on test order or the
  developer's machine.
- **Tolerances and references.** Each tolerance is justified by data type and reduction size; the reference is independent of the
  code under test.
- **Conventions.** Names and tiers follow `projects/hipdnn/docs/TESTING.md`, `projects/hipdnn/test_categories.yaml` and the
  provider `test_categories*.yaml`. Graph correctness uses JSON bundles per `dnn-providers/integration-tests/docs/file-formats.md` unless
  there is a reason.

## Reconcile with `RUN` (`TESTQ-f1`)

Read the `TESTQ` and `RUN` reports and rewrite the `M1:` line from `RUN`'s regression-test rows under **Checked, no finding**: `PASS` when
each regression test fails at `$BASE` (or with the fix reverted) and passes at `$HEAD`; `FAIL` when one passes without the fix, so it
is not a regression test (file the finding); `UNVERIFIED` when `RUN` could not run it. Write only this line and any new findings.
