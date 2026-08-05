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
  - GEMM, generation, and comparison template implementations live under
    `roc/host_validation/detail/` and are not the stable API surface.
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
#include <roc/host_validation/adapters/tensilelite/Reference.hpp>
```

## Standalone tests

```bash
cmake -S shared/host-validation -B build/host-validation \
  -DHOST_VALIDATION_BUILD_TESTING=ON
cmake --build build/host-validation
ctest --test-dir build/host-validation --output-on-failure
```
