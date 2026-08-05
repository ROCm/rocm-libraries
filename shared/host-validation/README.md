# ROCm host validation

`host-validation` owns GPU-independent generation, reference arithmetic, and
comparison used by ROCm library clients and tests.

## Targets

- `roc::host-validation-core`
  - The stable, GPU-independent tensor layer.
  - Exports only `roc/host_validation/tensor.hpp`.
  - Builds with an ordinary host compiler and the C++ standard library.
- `roc::host-validation`
  - Transitional validation operations layered on the tensor core.
  - Exports `roc/host_validation/validation.hpp`.
  - Exposes runtime-typed generation, reference GEMM, and comparison.
  - Implementation headers live under `roc/host_validation/detail/`; consumers
    include only `validation.hpp`.
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
  validation.hpp
  detail/

adapters/include/roc/host_validation/adapters/
  hipblaslt/
  tensilelite/

adapters/hipblaslt/
adapters/tensilelite/
```

The core layer is GEMM- and AMDGPU-agnostic. It contains only:

- `ScalarType` and `ScalarTypeInfo`;
- `Shape`;
- `Layout`;
- owning, runtime-typed `Tensor`;
- non-owning, runtime-typed `TensorView`; and
- non-owning, runtime-typed `MutableTensorView`.

The core public header must not include or name GEMM, HIP, AMDGPU, hipBLASLt,
TensileLite, rocisa, GTest, BLAS, GPU runtimes, generation policies, comparison
policies, or validation operations. Product enums, packed types, compatibility
entry points, and accelerated product backends belong in `adapters/`. Scalar
formats and their host codecs belong in the core because they are properties
of tensor storage, not of a validation operation or GPU product.

## Core API contract

The intended stable C++ surface is:

```cpp
scalarTypeInfo(ScalarType);
scalarTypeName(ScalarType);
visitScalarType(ScalarType, visitor);

Shape::Shape(std::vector<size_t>);
Shape::rank();
Shape::dimensions();
Shape::elementCount();

Layout::Layout(Shape, std::vector<ptrdiff_t>, ptrdiff_t offset = 0);
Layout::contiguous(const Shape&);
Layout::shape();
Layout::strides();
Layout::offset();
Layout::elementOffset(indices);

Tensor::Tensor(ScalarType, Shape);
Tensor::Tensor(ScalarType, Layout);
Tensor::fromStorage(ScalarType, Layout, std::vector<std::byte>);
Tensor::fromValues(ScalarType, Shape, values);
Tensor::fromNativeValues(Shape, nativeValues);
Tensor::type();
Tensor::shape();
Tensor::layout();
Tensor::storage();
Tensor::view();
Tensor::mutableView();

TensorView::fromNative(Layout, nativeValues);
TensorView::type();
TensorView::shape();
TensorView::layout();
TensorView::storage();
TensorView::loadAs<T>(indices);

MutableTensorView::fromNative(Layout, nativeValues);
MutableTensorView::loadAs<T>(indices);
MutableTensorView::storeFrom(indices, value);
```

`Tensor` owns its bytes and has value semantics: copying it performs a deep
copy. Views never own storage. Layout strides and offsets are measured in
logical scalar elements, including for sub-byte formats; the tensor layer
performs the element-to-bit addressing internally.

`visitScalarType` dispatches a runtime scalar type once to a unique semantic
tag. Operations should dispatch at their boundary and run typed inner loops;
they should not switch on `ScalarType` for every element.

Everything else currently exposed through `validation.hpp` or
`adapters/` is transitional and may be renamed or replaced as the operation
and runtime scalar-type APIs mature.

## Runtime reference GEMM

The canonical reference-GEMM API is tensor-centric and runtime-typed:

```cpp
GemmOperand a(TensorView);
GemmOperand b(TensorView);
GemmProblem problem(a, b, cView, dView, ScalarType::Float32);

problem.a.computeType = ScalarType::Float8E4M3;  // optional MAC-input quantization
problem.mathMode = MathMode::XFloat32;           // optional operand math
problem.epilogue.alpha = {1.0, 0.0};
problem.epilogue.beta = {0.0, 0.0};

GemmRunInfo run = referenceGemm(problem);
```

The normalized shapes are A `[M,K]`, B `[K,N]`, and C/D `[M,N]`.
Transpose, leading dimensions, padding, and adjusted base locations are
represented by `Layout`; no product transpose or matrix-layout enum crosses
the API.

`GemmProblem` currently supports:

- F32, F64, complex-F32, and complex-F64 accumulation;
- arbitrary runtime storage types supported by the tensor codecs;
- distinct compute-input types for A and B;
- default and XFloat32 operand math;
- alpha/beta;
- explicit row- or column-axis bias and scale-alpha bindings;
- row scale-A and column scale-B;
- tensor-backed block scales with independent A/B block sizes;
- ReLU, GELU, SiLU, and clamp; and
- canonical execution, backend support queries, fallback reporting, and
  grouped invocation.

All hipBLASLt/TensileLite call sites that used the former typed
`GemmInvocation<...>` now construct this runtime API. The typed reference GEMM
and its function-pointer quantization bridge have been removed.

New non-adapter consumers should need one of only two includes:

```cpp
#include <roc/host_validation/tensor.hpp>      // stable tensor core
#include <roc/host_validation/validation.hpp>  // transitional validation operations
```

Installed consumers can use:

```cmake
find_package(ROCHostValidation CONFIG REQUIRED)
target_link_libraries(app PRIVATE roc::host-validation-core)
```

Existing consumer include names may temporarily forward to adapter headers.
New code should include the stable adapter path directly, for example:

```cpp
#include <roc/host_validation/adapters/hipblaslt/Types.hpp>
#include <roc/host_validation/adapters/tensilelite/Reference.hpp>
```

## Python and NumPy oracle

An optional nanobind module mirrors the runtime scalar, shape, layout, tensor,
and reference-GEMM concepts:

```bash
source .venv/bin/activate
cmake -S shared/host-validation \
  -B build/host-validation-python \
  -DHOST_VALIDATION_BUILD_TESTING=ON \
  -DHOST_VALIDATION_BUILD_PYTHON=ON
cmake --build build/host-validation-python
ctest --test-dir build/host-validation-python --output-on-failure
```

The `roc_host_validation` package currently provides:

- `ScalarType`, `ScalarTypeInfo`, `Shape`, `Layout`, and owning `Tensor`;
- tensor construction from logical values or exact storage bytes;
- `from_numpy` and `to_numpy` copying conversions;
- deterministic tensor generation and structured comparison; and
- `reference_gemm` with runtime storage/output/accumulator types, alpha/beta,
  compute-input quantization, math mode, and activation.

The NumPy suite independently checks:

- every raw FP4, FP6, OCP/FNUZ FP8, and E5M3 encoding;
- all 65,536 FP16 and BF16 decodings;
- finite low-precision round trips;
- the OCP E8M0 no-zero contract;
- affine layout decoding;
- deterministic generation and structured comparison;
- F32, F64, and complex GEMM against NumPy; and
- mixed FP8-storage/FP4-compute-input quantization.

The first binding deliberately copies between NumPy and `Tensor`. A follow-up
should expose lifetime-safe non-owning NumPy-backed `TensorView` objects.

## Standalone tests

```bash
cmake -S shared/host-validation -B build/host-validation \
  -DHOST_VALIDATION_BUILD_TESTING=ON
cmake --build build/host-validation
ctest --test-dir build/host-validation --output-on-failure
```
