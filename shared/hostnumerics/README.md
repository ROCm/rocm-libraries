# HostNumerics

`hostnumerics` is a standalone, CPU-only shared component for generating
tensors, computing reference results, and deciding whether observed results are
numerically acceptable. At its center is a NumPy-like tensor model in which a
scalar type, shape, layout, and storage stay together.

The long-term goal is to serve the host-side numerical needs of projects across
`rocm-libraries` through one clear API. The component is designed to be useful
without HIP or a GPU, independently buildable, and fast enough for large test
problems. Product-specific concepts remain outside the component: hipBLASLt,
TensileLite, rocRoller, and future users translate their own descriptors at a
small adapter boundary.

## Incremental landing

The prototype in #10553 serves as an end-to-end integration reference while
the implementation lands in `develop` through small, independently reviewable
PRs. The table tracks the main stages; this prototype also contains the later
stages.

| Status | Stage | Scope |
| --- | --- | --- |
| ✅ [#12208](https://github.com/ROCm/rocm-libraries/pull/12208) | Build and package foundations | CMake targets and installed package, Python module, C++/Python and installed-package smoke tests, dedicated CPU CI, and ownership. |
| In progress | Tensor and datatype core | Tensor model, storage and ownership rules, scalar formats, and conversions, with C++ tests and independent Python coverage. |
| Planned | Numerical operations | Deterministic input generation, reference arithmetic, and comparison, with independent expected results and sanitizer coverage added alongside the implementation. |
| Planned | Consumer migrations and removal | Migrate hipBLASLt, TensileLite, and rocRoller's GEMM paths incrementally. Remove duplicate implementations as their callers migrate, and retire `mxDataGenerator` once its remaining responsibilities and build dependencies have moved. |

Keep each PR to one concept and as small as possible. The standalone stages
develop and test HostNumerics independently of the consumers. They keep it
outside the default monorepo build and provide fast feedback through the
standalone CPU CI jobs. The target is for the component
build and tests to take on the order of two minutes, excluding runner queue and
setup time.

For HostNumerics PRs that have no effect on other components, add the `ci:skip`
and `project: none` labels.

Consumer migrations must preserve existing product tests, supported numerical
behavior, and tolerances. Independent numerical tests should establish the
expected results before the shared implementation becomes a product's sole
reference. Compare representative input-generation and reference-computation
timings, as well as their effect on complete test and benchmark runs, before
removing an old accelerated path.

## Documentation

See [concepts and usage](docs/usage.md),
[build and package instructions](docs/build_and_test.md), and
[error handling](docs/error_handling.md).

The C++ and Python walkthroughs are [tutorial.cpp](examples/tutorial.cpp) and
[tutorial.py](examples/tutorial.py). Scalar metadata and dispatch are documented
in [scalar_type.hpp](include/roc/hostnumerics/scalar_type.hpp); the
[core tests](tests/core_test.cpp) include a compiled dispatch example.
