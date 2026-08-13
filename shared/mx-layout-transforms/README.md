# ROCMxLayoutTransforms

ROCMxLayoutTransforms is a small, CPU-only component for constructing the
physical MX layouts consumed by AMD GPU kernels. It owns layout permutation
only: numerical formats, value generation, reference arithmetic, device
allocation, and upload policy belong to their respective host-validation and
product components.

The installed header is:

```cpp
#include <roc/mx_layout_transforms/pre_swizzle.hpp>
```

The public namespace `roc::mx_layout_transforms` provides:

- `preSwizzle`
- `preSwizzleScalesGFX950`
- `preSwizzleScalesGFX950PaddedSize`
- `preSwizzleScalesGFX1250`
- `preSwizzleScalesGFX1250PaddedSize`

The component requires only a C++17 compiler. It has no HIP, BLAS, or
host-validation dependency. OpenMP is used when available and can be disabled
with `ROCMX_LAYOUT_TRANSFORMS_ENABLE_OPENMP=OFF`. Large transforms use at most
eight threads by default; an explicit `OMP_NUM_THREADS` setting takes
precedence.

## Standalone build and test

```bash
cmake -S . -B build -G Ninja
cmake --build build --parallel 8
ctest --test-dir build --output-on-failure
```

## Installed package

```cmake
find_package(ROCMxLayoutTransforms CONFIG REQUIRED)
target_link_libraries(my-target PRIVATE roc::mx-layout-transforms)
```
