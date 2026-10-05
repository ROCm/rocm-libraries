# ROCm Host Numerics

ROCm Host Numerics is a CPU-only C++ library for shared numerical reference
operations and test-data utilities. It is being introduced incrementally and
is not yet used by another ROCm Libraries project.

The initial package establishes stable CMake and Python integration. Numerical
types and operations will be added in focused follow-up changes.

## Build and test

```shell
cmake -S shared/host-numerics -B build -G Ninja \
  -DHOST_NUMERICS_BUILD_TESTING=ON \
  -DHOST_NUMERICS_BUILD_PYTHON=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

The installed CMake package exports `roc::host-numerics-core` and
`roc::host-numerics`. The Python package is named `roc_host_numerics`.
