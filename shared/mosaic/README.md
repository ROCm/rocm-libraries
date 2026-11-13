# Mosaic: Layout Abstractions on AMD Instinct GPUs

## Overview

Mosaic is a header-only C++20 layout-algebra library for spatially aware
index transformations on AMD Instinct GPUs.

The algebra is inspired by published layout systems:

- **CuTe** — Cecka, *"CuTe Layout Representation and Algebra"*,
  [arXiv:2603.02298](https://arxiv.org/abs/2603.02298).
- **Linear Layouts** — Zhou, Lezcano-Casado et al., *"Linear Layouts: Robust Code Generation of Efficient
  Tensor Computation Using F2"*, ASPLOS '26,
  [arXiv:2505.23819](https://arxiv.org/abs/2505.23819).

Host-side Python bindings (`pymosaic`) JIT the same C++ headers.

See [docs/programming_model.md](docs/programming_model.md) and
[docs/layout_tutorial.md](docs/layout_tutorial.md).

## License

MIT. See [LICENSE.md](LICENSE.md).

## Requirements

- CMake 3.25+
- A C++20 compiler
- A HIP install that provides the `hip::host` CMake target. `include/mosaic/utils.hpp`
  includes `<hip/hip_runtime.h>`, and the mosaic interface library links `hip::host`
  so that include path is available to dependents.
- [GoogleTest](https://github.com/google/googletest) (`find_package(GTest)`) for the C++ tests
- For `pymosaic`: Python 3.10–3.13, `nanobind`, `ninja`

Device code is opt-in. `MOSAIC_BUILD_HIP_EXAMPLES=ON` builds
`examples/cpp/basic_tiling` and `examples/cpp/mfma/mfma_example` as HIP
and links `hip::host`.

## Build and test (C++)

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Host examples (no GPU):

```text
./build/examples/cpp/inverse_layout/inverse_demo
./build/examples/cpp/slice/slice
./build/examples/cpp/mfma/mfma_layout
```

## pymosaic

```bash
pip install -e python/pymosaic
python -m pytest python/tests
```

`setup.py` copies `include/mosaic` into the wheel at build time; keep
`python/pymosaic` two directories below the mosaic root so that path still
works.

## API at a glance

```cpp
#include <mosaic/mosaic.hpp>

auto L = mosaic::make_layout(mosaic::make_shape(4, 8));   // (4,8):(1,4) col-major
L(2, 3);                                                  // = 14

float buf[32];
auto T = mosaic::make_tensor(buf, mosaic::make_shape(4, 8));
T(2, 3) = 7.0f;

using mosaic::_;
auto col3 = T(_, 3);
col3(0) = 99.0f;

auto tiled = mosaic::blocked_product(mosaic::row_major(2, 2),
                                     mosaic::column_major(4, 4));
```

## C++ examples

- [basic_tiling/](examples/cpp/basic_tiling/) — 2-level workgroup tiling via `blocked_product` (HIP; opt-in)
- [inverse_layout/](examples/cpp/inverse_layout/) — `right_inverse` round-trip
- [slice/](examples/cpp/slice/) — `_`-slicing and `dice`
- [mfma/](examples/cpp/mfma/) — MFMA register layouts as mosaic `Layout`s (host `mfma_layout`; HIP `mfma_example` opt-in)

## Header layout

Everything is in `include/mosaic/`. The umbrella `<mosaic/mosaic.hpp>` pulls in:

```text
mosaic/int_const.hpp     C<v>, Int<v>
mosaic/int_tuple.hpp     rank, depth, size, product, get, flatten, transform, ...
mosaic/stride.hpp        crd2idx, idx2crd, compact_col_major, compact_row_major
mosaic/layout.hpp        Layout<Shape, Stride>, make_layout
mosaic/slice.hpp         Underscore (_), slice, dice
mosaic/tensor.hpp        Tensor<Engine, Layout>, make_tensor
mosaic/convenience.hpp   row_major, column_major, strided<>, make_hierarchical
mosaic/algebra.hpp       blocked_product, right_inverse, ...
mosaic/print.hpp         print_layout helpers
```

`include/mosaic/utils.hpp` is HIP error-checking (`MOSAIC_CHECK`) and is **not**
pulled in by the umbrella header.
