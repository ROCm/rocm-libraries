# HostNumerics

HostNumerics is a shared CPU-only component for generating test inputs,
computing numerical reference results, and comparing them with observed results.
It builds and tests independently of HIP, GPU toolchains, and GPU hardware.

## Current scope

The component is being introduced incrementally. The endpoint is demonstrated in
[prototype PR #10553](https://github.com/ROCm/rocm-libraries/pull/10553).

The goal is to share CPU numerical work across hipBLASLt, TensileLite,
rocRoller's GEMM paths, and other components. Products own GPU execution,
tolerance selection, and reporting. See the
[migration plan](docs/migration.md) for the intended architecture and landing
stages.

## Documentation

- [Build, test, and package integration](docs/build_and_test.md).
- [Error handling](docs/error_handling.md).
- [Scalar metadata and dispatch](include/roc/hostnumerics/scalar_type.hpp).
- [Core tests and compiled usage example](tests/core_test.cpp).
- [Logical tensor shapes](include/roc/hostnumerics/shape.hpp) and the Python
  `Shape` docstrings.
- [Tensor layouts](include/roc/hostnumerics/layout.hpp), which map coordinates
  to element offsets, and the Python `Layout` docstrings.

## Contributing

For HostNumerics PRs that have no effect on other components, add the `ci:skip`
and `project: none` labels. Standalone HostNumerics CI still runs. See the
[migration plan](docs/migration.md) for guidance on implementation and consumer
changes.
