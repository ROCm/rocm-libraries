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

To qualify, a shim-surface test must:
1. Reach compute only through a public `miopen.h` entry point — never through
   `miopen::solver::` or `ProblemDescription` directly.
2. Validate against an independent CPU or analytically-known reference, not a
   self-comparison.
3. Use a tolerance appropriate for cross-implementation comparison, not
   bit-reproducibility.

See `hipdnn_shim_conv.cpp` for worked examples.
