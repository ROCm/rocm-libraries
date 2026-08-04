# ROCm host validation

`host-validation` owns GPU-independent generation, reference arithmetic, and
comparison used by ROCm library clients and tests.

## Targets

- `roc::host-validation`
  - GPU-independent C++20 headers.
  - Builds with an ordinary host compiler and the C++ standard library.
- `roc::host-validation-adapters`
  - Build-tree include surface for product-specific compatibility adapters.
  - Does not add GPU code to the core target.
- `roc::host-validation-tensilelite`
  - Optional compiled TensileLite reference implementation, created by the
    hipBLASLt/TensileLite client build.

## Layout

```text
include/roc/host_validation/
  tensor.hpp
  data_generation.hpp
  reference_gemm.hpp
  comparison.hpp

adapters/include/roc/host_validation/adapters/
  hipblaslt/
  tensilelite/

adapters/hipblaslt/
adapters/tensilelite/
```

The core public headers must not include HIP, hipBLASLt, TensileLite, rocisa,
GTest, BLAS, or GPU-runtime headers. Product enums, packed types, compatibility
entry points, and accelerated product backends belong in `adapters/`.

Existing consumer include names may temporarily forward to adapter headers.
New code should include the stable adapter path directly, for example:

```cpp
#include <roc/host_validation/adapters/tensilelite/Reference.hpp>
```

## Standalone tests

```bash
cmake -S shared/host-validation -B build/host-validation \
  -DHOST_VALIDATION_BUILD_TESTING=ON
cmake --build build/host-validation
ctest --test-dir build/host-validation --output-on-failure
```

The next architectural step is replacing the typed POC surface with a runtime
scalar type and general tensor view, then adding matching Python bindings and
NumPy differential tests.
