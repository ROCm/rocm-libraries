# gtest conventions

Full test-naming rules live on the wiki: https://github.com/ROCm/MIOpen/wiki/GTest-development#naming
(`Smoke`/`Standard`/`Full`/`Perf`/`Unit` prefixes, `CPU`/`GPU` hardware token, datatype
suffix). `check_names.py` enforces that schema against `miopen_gtest --gtest_list_tests`.

## Forwarding surface

A test belongs to the hipDNN-forwarding backend-swap surface if its full gtest name
contains the token `Forwarding`. Parameterized tests get it from the instantiation
prefix, e.g. `INSTANTIATE_TEST_SUITE_P(Forwarding, GPU_ConvFwdApi_FP32, ...)`.
Non-parameterized tests carry it directly in the suite name instead, e.g.
`TEST(GPU_ForwardingConvFwdApi_FP32, ...)`, since `check_names.py`'s prefix check only
applies to the `Smoke`/`Standard`/`Full`/`Perf`/`Unit` token used by parameterized
instantiations.

Select the surface with `--gtest_filter='*Forwarding*'`.

To qualify, a forwarding test must:
1. Reach compute only through a public `miopen.h` entry point — never through
   `miopen::solver::` or `ProblemDescription` directly.
2. Validate against an independent CPU or analytically-known reference, not a
   self-comparison.
3. Use a tolerance appropriate for cross-implementation comparison, not
   bit-reproducibility.

See `forwarding_conv_seed.cpp` for a minimal example.
