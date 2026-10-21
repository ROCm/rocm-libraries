# Lane: test breadth (`TESTB`)

Map the behavior the change adds or alters to the tests that exercise it, and report the gaps.

## Method

1. List each behavior the diff introduces or changes: new op or attribute, branch, data type, layout, error, config knob, arch path.
2. For each, find the tests at `$HEAD` that reach it (`git -C "$CLONE" grep` the symbol, follow the fixture, read the bundle and
   sweep JSON). Record test name and tier.
3. Report every behavior with no reaching test, and every dimension below that the change claims but tests do not cover.
4. For each behavior with no reaching test, record whether the PR provides a safe-default flag (default OFF; cite the option) or a
   written waiver (cite where). A flag satisfies M2 only as `M2: WAIVED(W-FLAG)` with a tracker ([../policy.md](../policy.md)
   *M-rules*), never `PASS`. Give the `M2:` line ([../findings.md](../findings.md) *Lane report*). New functionality with none of the three is `BLOCKING` ([../policy.md](../policy.md) *M-rules*).

## Dimensions

- Data types, including compute type and mixed precision.
- Layouts and strides: packed, NHWC/NCHW, non-contiguous, broadcast, zero-size, size one.
- Shapes: boundaries of every tiling, alignment and workspace threshold the code checks.
- Attribute combinations the code branches on, and their defaults.
- Negative cases: invalid inputs rejected with the right status; unsupported cases declined by applicability, not mis-executed.
- Architectures: each arch the change touches has a test that executes there ([../policy.md](../policy.md) *Review rules*, green
  is not coverage).
- Support claims: a new bundle for an engine and arch the change claims has a claim in its sidecar (`<bundle>.support.json`, or
  `support.json` for a sweep), so a regression fails instead of skipping (`dnn-providers/integration-tests/docs/file-formats.md` §
  "Support-claim sidecars", `dnn-providers/integration-tests/docs/adding-tests.md` § "Updating support claims", `dnn-providers/integration-tests/docs/support-claim-enforcement.md`).
- Tiers: new cases land in the tier CI runs for the PR, not only `full` or `comprehensive`.
- Python and C++ entry points both, when the change is visible through both (Python tests:
  `projects/hipdnn/python/frontend_wheel_package/tests`).

"Covered" is a claim that needs a test name. A gap is a `test-gap` finding. Give the mutation that would demonstrate it, the file
it applies to, and the suite or selection that must pass unmutated (derive the selection yourself: the test target and the gtest filter or ctest regex where the
behavior belongs; verifiers may reuse the mapping `RUN` records from changed files to tests).
