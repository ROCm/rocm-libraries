# gtest conventions

Full test-naming rules live on the wiki: https://github.com/ROCm/MIOpen/wiki/GTest-development#naming
(`Smoke`/`Standard`/`Full`/`Perf`/`Unit` prefixes, `CPU`/`GPU` hardware token, datatype
suffix). `check_names.py` enforces that schema against `miopen_gtest --gtest_list_tests`.

## hipDNN shim surface

A test belongs to the hipDNN backend-swap surface if its full gtest name contains the
token `HipdnnShim`. Parameterized tests get it from the instantiation prefix, e.g.
`INSTANTIATE_TEST_SUITE_P(HipdnnShim, GPU_ConvFwdApi_FP32, ...)`. Non-parameterized tests
carry it directly in the suite name instead, e.g. `TEST(GPU_HipdnnShimConvFwdApi_FP32, ...)`,
since `check_names.py`'s prefix check only applies to the `Smoke`/`Standard`/`Full`/`Perf`/
`Unit` token used by parameterized instantiations.

The token is deliberately specific: it is a filter, so a generic word like `Forwarding`
would sooner or later pull in an unrelated test and silently double its runtime.

Select the surface with `--gtest_filter='*HipdnnShim*'`. That filter is what the
`forwarding_parity` ctest entries replay under `MIOPEN_HIPDNN_FORWARDING=disabled` and
`=enabled`; a test outside the surface is never replayed, and a test inside it that is not
reachable through public entry points makes the comparison meaningless.

### Configuring a build that runs the parity entries

The parity entries exist in any build configured with `-DMIOPEN_ENABLE_HIPDNN_WRAPPER=ON` on a
GPU node. `MIOPEN_TEST_DISCRETE` does not matter: CMake scans the test sources for the
`HipdnnShim` token and registers a parity entry for each executable that holds one, so a
discrete build replays `test_hipdnn_shim_conv` and a single-binary build replays
`miopen_gtest`. Configure prints which binaries it registered.

The two things that do suppress the entries are no GPU (`MIOPEN_NO_GPU`, since the shim
surface is all `*GPU*` tests) and no source carrying the token. Both are reported at configure
time, because the failure is otherwise silent: with nothing registered, `ctest -L
forwarding_parity` selects nothing and reports success. A CI job that is supposed to enforce
parity should still assert that `ctest -N -L forwarding_parity` lists a non-zero number of
tests, rather than trusting a green run that selected none.

One asymmetry remains in packaged artifacts. The mirrored entry written into the installed
`CTestTestfile.cmake` is single-binary only, because packaging installs one test executable and
it is `miopen_gtest`. That costs nothing in practice — the builds that produce artifacts
already configure with `-DMIOPEN_TEST_DISCRETE=OFF`, which packaged dbsync and the categorized
test list need anyway — but a discrete build tests parity in its build tree only.

Two consequences of the parity entries being the intended path, both easy to "fix" in the wrong
direction:

- **Shim-surface tests stay out of `test_categories.yaml` on purpose.** The categorized
  smoke/standard/full entries turn those patterns into a `--gtest_filter`, so a matching
  pattern would register a third run of tests the parity entries already run twice. Absence
  from that file is the design, not a gap.
- **Excluding a shim-surface test means narrowing the parity filter,** not calling
  `add_gtest_negative_filter`. That function feeds the shard filter, which the parity entries
  do not use, so disabling a shim test through it looks like it worked and changes nothing.

To qualify, a shim-surface test must:
1. Reach compute only through a public `miopen.h` entry point — never through
   `miopen::solver::` or `ProblemDescription` directly.
2. Validate against an independent CPU or analytically-known reference, not a
   self-comparison.
3. Use a tolerance appropriate for cross-implementation comparison, not
   bit-reproducibility.

See `hipdnn_shim_conv.cpp` for worked examples.
