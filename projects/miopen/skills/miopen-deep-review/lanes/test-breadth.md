# Lane: test breadth (`TESTB`)

Map the behavior the change adds or alters to the tests that exercise it, and report the gaps.

## Method

1. List each behavior the diff introduces or changes: new solver, primitive or API entry point, `IsApplicable` branch, data type,
   layout, error, environment variable, find or database path, arch path.
2. For each, find the tests at `$HEAD` that reach it (`git -C "$CLONE" grep` the symbol and the solver's `SolverDbId` under
   `projects/miopen/test/`, follow the fixture and its parameter lists). Record test name and tier (`Smoke`, `Full`, and the
   `test_categories.yaml` tier that selects it).
3. Report every behavior with no reaching test, and every dimension below that the change claims but tests do not cover.
4. For each behavior with no reaching test, record whether the PR provides a safe-default flag (default OFF; cite the option or
   environment variable) or a written waiver (cite where). A flag satisfies M2 only as `M2: WAIVED(W-FLAG)` with a tracker
   ([../policy.md](../policy.md) *M-rules*), never `PASS`. Give the `M2:` line ([../findings.md](../findings.md) *Lane report*).
   New functionality with none of the three is `BLOCKING` ([../policy.md](../policy.md) *M-rules*).

## Dimensions

- Data types, including compute type, mixed precision, FP8 and BF16, and int8 where the primitive supports them.
- Layouts and strides: NCHW, NHWC, NCDHW, NDHWC, packed and non-packed, groups and depthwise, zero-size, size one.
- Shapes: boundaries of every tiling, alignment, 32-bit index and workspace threshold the solver checks in `IsApplicable`.
- Directions: forward, backward data and backward weights, each a separate solver path in convolution.
- Attribute combinations the code branches on (stride, dilation, padding, alpha/beta, activation mode), and their defaults.
- Negative cases: invalid inputs rejected with the right `miopenStatus_t`; unsupported problems declined by `IsApplicable`, not
  mis-executed. A test that a solver is *not* applicable is as important as one that it is.
- Tuning: a tunable solver's search space has a test that a non-default perf config runs and verifies, not only the default.
- Find modes: normal find, immediate mode and the find-db fast path when the change touches selection.
- Architectures: each arch the change touches has a test that executes there ([../policy.md](../policy.md) *Review rules*, green
  is not coverage).
- Tiers: new cases land in an instantiation and pattern CI runs for the PR (`Smoke` and `quick`/`standard`), not only `Full` or
  `comprehensive`.
- Entry points: the C API and MIOpenDriver both, when the change is visible through both (`GPU_MIOpenDriver*` tests), and the
  hipDNN MIOpen provider when the change alters an entry point it calls.

"Covered" is a claim that needs a test name. A gap is a `test-gap` finding. Give the mutation that would demonstrate it, the file
it applies to, and the suite or selection that must pass unmutated (derive the selection yourself: `miopen_gtest` with a
`--gtest_filter` where the behavior belongs, or a `ctest -R` regex; verifiers may reuse the mapping `RUN` records from changed
files to tests).
