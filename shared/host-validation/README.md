# ROCm host validation

`host-validation` owns product-independent CPU generation, reference
arithmetic, and comparison used by ROCm library clients and tests. It also
contains a separate CPU-only module for constructing physical AMD GPU layouts.

## Targets

- `roc::host-validation-core`
  - The stable, GPU-independent tensor layer.
  - Exports only `roc/host_validation/tensor.hpp`.
  - Builds with an ordinary host compiler and the C++ standard library.
- `roc::host-validation-amd-gpu-layout`
  - Compiled physical MX layout permutations requested by product adapters.
  - Exports `roc/host_validation/amd_gpu_layout/mx.hpp`.
  - Does not depend on the tensor or numerical targets.
  - Keeps type-generic copy templates in the public header and compiles
    validation, layout planning, and scheduling into a private source file.
  - Uses private optional OpenMP parallelism, disabled with
    `HOST_VALIDATION_AMD_GPU_LAYOUT_ENABLE_OPENMP=OFF`. Large transforms use at
    most eight threads by default; `OMP_NUM_THREADS` overrides the cap.
    Consumers do not inherit OpenMP compile definitions or include `omp.h`.
- `roc::host-validation`
  - Transitional validation operations layered on the tensor core.
  - Exports `axpby.hpp`, `comparison.hpp`, `epilogue.hpp`, `generation.hpp`,
    `gemm.hpp`, `layer_norm.hpp`, `reduction.hpp`, `softmax.hpp`,
    `structured_sparsity.hpp`, and the convenience umbrella `validation.hpp`.
  - Exposes runtime-typed generation, tensor AXPBY, reference GEMM, reference
    epilogues, LayerNorm, reductions, softmax, structured sparsity, and comparison.
  - Numerical implementations are compiled into the library. Consumers include
    the operation header they need or `validation.hpp`.
  - Ordinary tensor generation uses optional private OpenMP parallelism with a
    work-aware default cap of eight threads. `OMP_NUM_THREADS` overrides the
    cap; small, nested, packed, or potentially aliased generation remains
    serial.
- `roc::host-validation-blas`
  - Optional compiled CBLAS implementation of `GemmBackend::Blas`.
  - Built with `HOST_VALIDATION_BUILD_BLAS_BACKEND=ON`.
  - Requires `cblas.h`, all four real/complex CBLAS GEMM entry points, and the
    LP64 ABI with 32-bit dimensions. Configuration rejects ILP64 providers
    rather than compiling against a mismatched integer interface.
  - Does not choose BLIS, OpenBLAS, MKL, or another implementation.
    `BLA_VENDOR`, `BLA_STATIC`, `BLA_PREFER_PKGCONFIG`,
    `BLA_PKGCONFIG_BLAS`, `CMAKE_PREFIX_PATH`, and the surrounding build
    environment select the provider.
  - Leaves provider-global threading to the selected BLAS implementation.
    Reproducible runs should set the provider's process-start environment,
    such as `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `GOTO_NUM_THREADS`,
    `BLIS_NUM_THREADS`, `MKL_NUM_THREADS`, or
    `VECLIB_MAXIMUM_THREADS`.
- `roc::host-validation-blocked`
  - GPU-independent blocked implementation of `GemmBackend::Blocked`.
  - Supports dense F32/F64 accumulation, runtime input/output types,
    compute-input quantization, XFloat32, vector/scalar epilogue operands,
    bias, and activation.
- `roc::host-validation-mx`
  - Compiled block-scaled tensor data-generation implementation.
  - Built consistently with the component rather than hidden behind an
    optional configuration.
  - Uses native coordinate-based generation with optional OpenMP parallelism
    and exports only component-owned scalar, tensor, block-axis, and recipe
    types.

## Component dependency contract

Every file and target owned by `shared/host-validation` is product-independent.
The component must not include, compile against, or use hipBLASLt, TensileLite,
rocisa, HIP, GPU architecture types, or product enums. Product-private adapters
translate their descriptors into component-owned tensor and operation types:

```text
private product adapter -> roc::host-validation -> roc::host-validation-core
private product adapter -> roc::host-validation-amd-gpu-layout
```

The dependency may never point in the opposite direction. The
`host-validation-component-boundary` test scans the complete component source
tree for forbidden product dependencies. Generic tensor and numerical code
must not include or link the AMD GPU layout module.

## Layout

```text
include/roc/host_validation/
  amd_gpu_layout/mx.hpp
  index_order.hpp
  scalar.hpp
  detail/scalar_codec.hpp
  tensor.hpp
  comparison.hpp
  validation.hpp

src/
python/
tests/
```

The core layer is GEMM- and AMDGPU-agnostic. It contains only:

- `ScalarType` and `ScalarTypeInfo`;
- `Shape`;
- `Layout`;
- shared-storage, runtime-typed `Tensor`; and
- allocator-independent `TensorStorage`.

The core public header must not include or name GEMM, GPU runtimes, GPU
architectures, product enums, test frameworks, BLAS, generation policies,
comparison policies, or validation operations. Scalar formats and their host
codecs belong in the core because they are properties of tensor storage, not
of a validation operation or consuming product.

## Core API contract

The intended stable C++ surface is:

```cpp
scalarTypeInfo(ScalarType);
scalarTypeName(ScalarType);
visitScalarType(ScalarType, visitor);

IndexOrder::{FirstDimensionFastest, LastDimensionFastest};

Shape::Shape(std::vector<size_t>);
Shape::rank();
Shape::extent(dimension);
Shape::dimensions();
Shape::elementCount();
Shape::elementCount(firstDimension, lastDimension);
Shape::elementCountExcluding(dimension);
Shape::linearIndex(coordinates, IndexOrder);
Shape::coordinates(linearIndex, IndexOrder);

Layout::Layout(Shape, std::vector<ptrdiff_t>, ptrdiff_t offset = 0);
Layout::contiguous(const Shape&);
Layout::contiguousLastDimensionFastest(const Shape&);
Layout::contiguousFirstDimensionFastest(const Shape&);
Layout::shape();
Layout::rank();
Layout::extent(dimension);
Layout::elementCount();
Layout::strides();
Layout::stride(dimension);
Layout::offset();
Layout::elementOffset(indices);

Scalar::from(nativeValue);
Scalar::fromStorage(ScalarType, encodedBytes);
Scalar::zero(ScalarType);
Scalar::one(ScalarType);
Scalar::type();
Scalar::storage();
Scalar::as<T>();

TensorStorage::allocate(bytes);
TensorStorage::wrap(owner, mutableBytes);

Tensor::Tensor(ScalarType, Shape);
Tensor::Tensor(ScalarType, Layout);
Tensor::Tensor(ScalarType, Shape, TensorStorageAllocator);
Tensor::Tensor(ScalarType, Layout, TensorStorageAllocator);
Tensor::wrapStorage(ScalarType, Layout, TensorStorage);
Tensor::fromStorage(ScalarType, Layout, std::vector<std::byte>);
Tensor::fromValues(ScalarType, Shape, values);
Tensor::fromNativeValues(Shape, nativeValues);
Tensor::fromNative(Layout, nativeValues);
Tensor::type();
Tensor::shape();
Tensor::layout();
Tensor::storage();
Tensor::loadAs<T>(indices);
Tensor::storeFrom(indices, value);
Tensor::alias(Layout);
Tensor::clone();
Tensor::clone(TensorStorageAllocator);
Tensor::copyFrom(Tensor);
Tensor::copyFrom(Tensor, linearIndices);
Tensor::to(ScalarType);
Tensor::to(ScalarType, ScalarConversionOptions);
```

`Scalar` owns one encoded runtime-typed value inline. It has no shape, layout, or
shared storage. Copying a Scalar copies its bytes. `fromStorage` copies exactly
one encoded value; it does not retain the supplied span.

`Tensor` is a reference-counted handle. Copy construction and assignment copy
type/layout metadata and share the same storage; mutations are visible through
every alias. `clone()` is the explicit deep-copy operation. `alias(Layout)`
creates another layout over the same owned storage. Shallow constness is
intentional: a const Tensor handle may still mutate its shared data.

Every Tensor participates in owning its storage lifetime. The default storage
uses ordinary host bytes; an allocator callback may return owned storage backed
by a product-specific allocator such as pooled HIP-pinned memory. Native-array
factories copy into owned storage. `TensorStorage::wrap` requires a lifetime
owner and anchors an existing mutable allocation; `Tensor::wrapStorage` retains
that owner without copying. There is no public unowned tensor/view type.

Layout strides and offsets are measured in logical scalar elements, including
for sub-byte formats; the tensor layer performs the element-to-bit addressing
internally.

`to(ScalarType)` performs explicit runtime storage conversion while preserving
shape, strides, and offset. Same-type conversion copies the layout's required
raw storage without decoding; cross-type conversion decodes and re-encodes
logical values through the core scalar codecs. Option-bearing overloads make
integer rounding (`TowardZero` or deterministic `NearestEven`) and overflow
handling (`Reject`, `Saturate`, or `ModuloWrap`) explicit. NaN-to-integer and
lossy complex-to-real conversions are rejected. No-options native conversions
truncate and modulo-wrap integers deterministically. No-options storage
conversions use the same policy for byte-addressable integers and saturate the
synthetic packed Int4 and Int12 encodings.

`visitScalarType` dispatches a runtime scalar type once to a unique semantic
tag. Operations should dispatch at their boundary and run typed inner loops;
they should not switch on `ScalarType` for every element.

Everything else currently exposed through `validation.hpp` is transitional
and may be renamed or replaced as the operation and runtime scalar-type APIs
mature.

## Deterministic tensor generation

`GenerationRecipe` is the typed C++ generation API. Parameter objects use
distribution terms rather than numbered fields, and the public recipe exposes
one seed rather than RNG stream identifiers:

```cpp
GenerationRecipe recipe = GenerationRecipe::cartesian(
    GenerationRecipe::sine(),
    GenerationRecipe::cosine(),
    {.seed = 17,
     .indexOrder = IndexOrder::FirstDimensionFastest});

Tensor output = generate(ScalarType::ComplexFloat32, layout, recipe);
GenerationRunInfo patch = generateAt(output, logicalIndex, recipe);
```

Named factories cover constants, candidate sets, uniform integer/real and
normal distributions, sine/cosine, logical or dimension indices, identity,
affine-index remainders, type-derived values, encoded-exponent sampling, and
raw storage. Modifiers return new components and apply an absolute/sine/cosine
transform, affine scale/offset, or coordinate-parity sign alternation.

Complex construction is explicit:

- `realOnly(component)` writes zero to the imaginary component;
- `replicated(component)` evaluates once and writes the same value to both
  components; and
- `cartesian(real, imaginary)` evaluates each component in a distinct internal
  random domain.

Random values are counter-based, so a tensor element depends only on the seed,
internal domain, and logical index. Loop order and thread count do not affect
the result.

`generateAt` applies the same recipe to one logical element. Its coordinate is
decoded with the recipe's index order, so callers do not reproduce
layout-independent linear-index arithmetic.

## Tensor linear combination

`axpby.hpp` evaluates an elementwise linear combination with explicit
accumulator and output storage types:

```cpp
AxpbyProblem problem(
    xTensor, yTensor, ScalarType::Float32, ScalarType::Float32);
problem.alpha = 2.0;
problem.beta = -0.5;
AxpbyResult result = referenceAxpby(problem);
```

Either input may be absent, but at least one is required. The Tensors must share
one shape. The owning overload creates a contiguous result and accepts an
optional `TensorStorageAllocator`. `AxpbyRequest` binds the same problem to a
caller-owned destination when a specific output layout is required.

## Softmax

`softmax.hpp` computes a numerically stabilized softmax over one explicit
tensor axis:

```cpp
SoftmaxProblem problem(
    inputTensor, ScalarType::Float32, axis, ScalarType::Float32);
SoftmaxResult result = referenceSoftmax(problem);
```

The owning overload returns a contiguous output and accepts an optional
`TensorStorageAllocator`. `SoftmaxRequest` writes a caller-owned destination
when a specific output layout is required. Input and output storage types may
differ. The accumulator is explicitly Float32 or Float64, and the
implementation subtracts each slice's maximum before exponentiation.

## LayerNorm

`layer_norm.hpp` applies Welford population-variance normalization over one
explicit tensor axis:

```cpp
LayerNormProblem problem(
    inputTensor, ScalarType::Float32, axis, ScalarType::Float32);
problem.meanType = ScalarType::Float32;
problem.inverseVarianceType = ScalarType::Float32;
problem.gamma = gammaView;
problem.beta = betaView;
problem.epsilon = 1e-5;
LayerNormResult result = referenceLayerNorm(problem);
```

`meanType` and `inverseVarianceType` independently request statistics whose
shape removes the normalized axis. The owning overload returns contiguous
tensors and accepts an optional `TensorStorageAllocator`. `LayerNormRequest`
writes caller-owned destinations. Gamma and beta are optional vectors over the
normalized axis.

## Structured tensor comparison

`comparison.hpp` owns the complete host-side numerical decision after a
product has copied an observed tensor into host memory:

```cpp
ComparisonOptions options
    = defaultComparisonOptions(ScalarType::Float32);
options.selection.indexOrder
    = IndexOrder::FirstDimensionFastest;
options.computeUlp = true;
options.ulpType = ScalarType::Float32;

ComparisonResult report = compare(observedView, expectedView, options);
```

One plan can select logical elements and collect:

- exact, absolute, relative, and symmetric-relative pointwise decisions;
- complex component evidence;
- explicit NaN, infinity, and signed-zero policy;
- maximum absolute/relative differences and mismatch samples;
- observed, expected, difference, and relative Frobenius evidence;
- maximum, sum, and average ULP evidence;
- candidate-grid allclose tolerance search; and
- unwritten-sentinel checks before, inside, and after logical tensor storage.

Comparison policy is intentionally entered through Tensors. The former
scalar `valuesClose` helper and typed allclose-tolerance search were removed;
one-element values use ordinary tensors, while tolerance search remains on the
runtime-typed API.

The exhaustive numerical policy matrix is tested through the Python API
against NumPy and raw-bit Python oracles. C++ comparison tests retain layout,
selection, exact-integer, packaging, sanitizer, and performance
contracts rather than serving as a second independent source of numerical
truth.

This ownership deliberately excludes hipBLASLt's runtime device
`check_numerics_matrix` facility. That HIP kernel scans device memory for
NaN/Inf while the library is running; it is not expected-versus-reference
validation and remains a separate product/GPU concern.

## Runtime reference GEMM

The reference-GEMM API is tensor-centric and runtime-typed:

```cpp
GemmOperand a(aTensor);
GemmOperand b(bTensor);
GemmRequest request(a, b, cTensor, dTensor, ScalarType::Float32);

request.a.computeType = ScalarType::Float8E4M3;  // optional MAC-input quantization
request.accumulationRounding = AccumulationRounding::FullPrecision;
request.mathMode = MathMode::XFloat32;           // optional operand math
request.epilogue.alpha = {1.0, 0.0};
request.epilogue.beta = {0.0, 0.0};

GemmExecution execution;
execution.backend = GemmBackend::Automatic;

GemmSupportInfo support = queryGemmSupport(request, execution);
GemmResult result = referenceGemm(request, execution);
```

An owning call uses the same numerical descriptor:

```cpp
GemmProblem problem(a, b, cTensor, ScalarType::Float32, ScalarType::Float32);
GemmOutputOptions output;
output.selection = OutputSelection::explicitIndices({0, 3});
GemmResult owned = referenceGemm(problem, output, execution);
```

The owning overload accepts an optional `TensorStorageAllocator`, validates the
output layout and selection before allocation, and zero-initializes D so
unselected logical coordinates and affine-layout gaps are deterministic.

The normalized shapes are A `[M,K]`, B `[K,N]`, and C/D `[M,N]`.
Transpose, leading dimensions, padding, and adjusted base locations are
represented by `Layout`; no product transpose or matrix-layout enum crosses
the API. Support queries and execution consume the same `GemmRequest`.
`GemmExecution` contains backend policy, while an optional backend
implementation object is supplied at call time rather than becoming part of
the numerical request.

`GemmRequest` currently supports:

- F16, BF16, F32, F64, I32, complex-F32, and complex-F64 accumulation;
- arbitrary runtime storage types supported by the tensor codecs;
- distinct compute-input types for A and B;
- explicit type-default, full-precision, or per-product/per-sum accumulator
  rounding;
- ordered scalar/vector operand factors before compute-input quantization;
- default and XFloat32 operand math;
- alpha/beta;
- explicit row- or column-axis bias and scale-alpha bindings;
- row scale-A and column scale-B;
- tensor-backed block scales with independent A/B block sizes;
- all, explicit-index, and prime-stride output selection;
- output scaling plus explicit rounded/saturating Int8 conversion;
- absolute, clipped/leaky ReLU, ReLU, GELU, GELU scaling and derivative,
  sigmoid, tanh, SiLU, Swish, clamp, and explicit ReLU derivative; and
- Pointwise execution, pluggable object-oriented backend implementations,
  backend support queries, and fallback reporting.

Int32 GEMM arithmetic never uses floating-point values as an accumulator
proxy. Products, partial sums, alpha/beta combination, integer scales and
integer epilogue arithmetic use explicitly defined two's-complement wrapping
modulo `2^32`. Integer scalar parameters must be finite, integral and within
the Int32 range before entering the operation. Python oracle tests use Python
integers plus explicit wrapping rather than relying on NumPy's implicit
integer-matmul accumulation behavior.

Consumers construct this runtime API without passing product-specific types.
The former typed reference GEMM and its function-pointer quantization bridge
have been removed.

Python exposes the native caller-owned request model:

```python
import roc_host_validation as hv

a = hv.GemmOperand(hv.from_numpy(array_a))
b = hv.GemmOperand(hv.from_numpy(array_b))
d = hv.Tensor(hv.ScalarType.Float32, hv.Shape([m, n]))
request = hv.GemmRequest(
    a,
    b,
    hv.from_numpy(array_c),
    d,
    accumulator_type=hv.ScalarType.Float32,
)
request.epilogue.alpha = 2.0
result = hv.reference_gemm_result(request)
```

Python `VectorBinding`, `BlockScaleBinding`, `GemmOperand`, `GemmEpilogue`,
`GemmProblem`, `GemmRequest`, and `GemmResult` are direct bindings of the C++
types. C and D are explicit tensors; the result aliases D. Requests retain
shallow Tensor handles, so inputs, outputs, aliases, and allocator-backed
storage remain alive for the synchronous call.
`reference_gemm_flat_result(a, b, c, ...)` is the explicitly named convenience
that builds native descriptors and allocates a contiguous output.
`GemmProblem` plus `GemmOutputOptions` exposes the same owning path with an
affine output layout or partial logical selection.

The optional `BlasGemmBackend` implements the same interface for dense
F32/F64/complex GEMM and is selected through `GemmExecution`.
`TransformingBlasGemmBackend` additionally materializes runtime-typed,
scaled, and compute-input-quantized operands into component-owned scratch,
invokes the ordinary BLAS backend, and performs component-owned output
scaling/conversion. This preserves accelerated large mixed-type references
without placing conversion loops in product adapters.
Configuration links through the semantic `CBLAS::CBLAS` target and prints the
resolved header, link interface, requested `BLA_VENDOR`, and validated LP64
ABI. The installed package exports
`ROCHostValidation_BLAS_BUILD_PROVIDER` and
`ROCHostValidation_BLAS_INTEGER_SIZE`; because the backend is a static
archive, an installed consumer may legitimately resolve a different
conforming provider.

The BLAS conformance executable is registered in separate one-thread and
multi-thread CTest processes. It uses only the host-validation API and common
provider environment controls, including a moderately large exact GEMM.
There is no standard CBLAS API for proving how many worker threads an arbitrary
provider created, so these tests establish numerical conformance under both
requested thread configurations rather than introspecting provider internals.
The Pointwise strategy covers and writes exactly the selected outputs.
Accelerated strategies report selected writes separately from all output
coordinates covered by their execution granularity.

`BlockedGemmBackend` implements the same object-oriented interface without BLAS
or product dependencies. It reuses decoded A/B tiles across output elements
and is the migration target for TensileLite's product-local fast CPU path.
It also supports block-scaled MX operands when both block sizes and K align to
the backend's eight-element reduction tile, and it preserves partial-output
selection by committing only selected epilogues.

F16 and BF16 accumulator modes execute with a float host register but quantize
the product and accumulated sum after each arithmetic step. They therefore
model low-precision accumulation rather than silently substituting F32
accumulation.

## Block-scaled tensor generation

The MX target generates packed data, natural-layout scales, explicit
per-element scale indices, and the decoded F32 reference tensor:

```cpp
MxGenerationProblem problem;
problem.dataType = ScalarType::Float4E2M1;
problem.scaleType = ScalarType::E8M0;
problem.shape = Shape{64, 128};
problem.leadingDimension = 64;
problem.blockAxis = 0;
problem.blockSize = 32;
problem.data = MxDataRecipe::bounded({.lower = -1.0, .upper = 1.0});
problem.scale = MxScaleGenerationMode::Derived;

MxGenerationResult result = generateMx(problem);
```

`MxDataRecipe` is an immutable, factory-built description of source data.
Bounded intervals, alternating magnitudes, normal distributions, and integer
distributions have typed parameter objects; fixed patterns use named factories.
`MxScaleGenerationMode` independently selects derived per-block scales or a
constant minimum, one, two, maximum, or NaN scale.

The allocator overload applies one `TensorStorageAllocator` to data, scales,
scale indices, and the decoded reference tensor.

The operation is GEMM- and architecture-agnostic. A product adapter identifies
which tensor axis is block-scaled and may subsequently transform the natural
scale bytes for an architecture-specific upload layout. GFX950/GFX1250
selection and swizzling are intentionally outside this target. The
`scaleIndices` tensor makes the blocked-axis mapping explicit and lets NumPy
independently verify
`reference == data * scales[scaleIndices]`.

When OpenMP is available, MX generation uses at most eight threads by default
and scales down for small tensors. An explicit `OMP_NUM_THREADS` overrides that
default cap. Calls made from an existing OpenMP parallel region execute
serially rather than creating a nested pool.

## Runtime reference epilogue

The component also owns the elementwise host epilogue used after reference
GEMM:

```cpp
EpilogueProblem problem(
    inputTensor, ScalarType::Float32, ScalarType::Float32);
problem.bias = VectorBinding{biasView, MatrixAxis::Row};
problem.activation = Activation::Gelu;
problem.auxiliaryOutputType = ScalarType::Float32;
problem.rawOutputType = ScalarType::Float32;
problem.amaxType = ScalarType::Float32;
problem.outputScale = 2.0;
problem.auxiliaryScale = 3.0;

EpilogueResult result = referenceEpilogue(problem);
```

The current operation supports runtime input/output/bias/auxiliary types, F32,
F64, and I32 compute, explicit bias axes, forward and gradient activation,
auxiliary E input/output, scale-D, scale-E, raw output, and AMax. The
hipBLASLt pointer and `hipDataType` translation lives in its private client
adapter and is not compiled by this component.

The owning overload returns contiguous outputs, accepts an optional
`TensorStorageAllocator`, and initializes every unselected logical output to
numeric zero independently of allocator contents. `EpilogueRequest` binds
existing destinations, preserves unselected coordinates, and optionally
accumulates into an existing AMax value.

An optional gate-residual tensor applies `gate * value + gate` after output
scaling. `rawOutput`, when requested, captures the scaled value before the
gate; AMax is measured before output scaling and the gate. `OutputSelection`
can restrict the elementwise program to the same explicit or prime-stride
logical subset used by reference GEMM.

## Runtime tensor reduction

`referenceSum` reduces arbitrary tensor axes while preserving all remaining
dimensions in order:

```cpp
ReductionProblem problem(inputView,
                         ScalarType::Float32,
                         ScalarType::Float32,
                         {0, 2});
ReductionResult result = referenceSum(problem);
```

The current implementation supports F32, F64, I32, complex-F32, and complex-F64
accumulation, runtime input/output storage types, validated signed-strided layouts
used by current consumers, rank-zero outputs, and multiple reduction axes. The
owning overload derives a contiguous output shape and accepts an optional
`TensorStorageAllocator`. `ReductionRequest` writes a caller-owned destination.
hipBLASLt's bias-gradient adapter represents its matrix as a strided tensor and
reduces the K axis; no product type enters the component.

`referenceMaximumAbsolute` reduces every logical input dimension into a
rank-zero output. It supports F16, BF16, F32, and F64 accumulator policies,
ignores NaN inputs consistently with the existing GEMM-epilogue AMax program,
and converts the final value through the requested output tensor codec.

## Structured sparsity

`applyStructuredSparsity` applies a logical N:M pattern along one tensor axis
without naming a GPU instruction or product descriptor:

```cpp
StructuredSparsityPattern pattern;
pattern.axis = 1;
pattern.groupSize = 4;
pattern.retainedElements = 2;
pattern.fixedPositions = {0, 2};

StructuredSparsityProblem problem(
    inputTensor,
    pattern,
    {.retainedIndices = true, .twoOfFourMetadata = true});
StructuredSparsityResult result = applyStructuredSparsity(problem);
```

The operation supports:

- fixed or counter-based deterministic random retained positions;
- validated signed-strided input and output layouts used by current consumers;
- in-place pruning;
- byte-preserving copies for ordinary and packed scalar formats;
- optional retained-position indices;
- optional fused 2:4 metadata encoding; and
- independent slice ranges for caller-selected parallel scheduling.

A problem's owning overload derives contiguous output shapes and accepts an
optional `TensorStorageAllocator`. `StructuredSparsityRequest` binds existing
destinations, permits exact input/pruned in-place operation, and exposes slice
ranges. Every other input/output storage overlap is rejected.

A slice is one logical combination of all coordinates except the sparsity
axis. Product adapters may partition slices with OpenMP or another host
executor without moving pruning, compression, random selection, or metadata
arithmetic back into the product. The sparsity operation itself has no OpenMP,
HIP, or AMDGPU dependency; MX generation uses the bounded OpenMP
policy described above.

`encodeTwoOfFourMetadata` remains available when retained indices already
exist and metadata must be produced as a separate operation.

Consumers should need one of only these focused includes:

```cpp
#include <roc/host_validation/tensor.hpp>      // stable tensor core
#include <roc/host_validation/comparison.hpp>  // host comparison plan/report
#include <roc/host_validation/validation.hpp>  // transitional validation operations
```

Installed consumers can use:

```cmake
find_package(ROCHostValidation CONFIG REQUIRED COMPONENTS Core)
target_link_libraries(app PRIVATE roc::host-validation-core)
```

The installed package exposes `Core`, `Operations`, `Blocked`, `BLAS`, `MX`, and
`AMDGPULayout` components. Component dependencies are loaded transitively:
`Blocked` and `BLAS` require `Operations`, while `Operations` and `MX` require
`Core`. A component lookup loads only the requested closure, so `Core` does not
search for CBLAS or OpenMP. The `BLAS` component locates a conforming
`CBLAS::CBLAS` target when requested; consumers do not manually repeat a
provider-specific link line.

## Python and NumPy oracle

The standalone component build includes a required nanobind module that
mirrors the runtime scalar, shape, layout, tensor, and reference-GEMM
concepts. Embedding projects may disable the artifact explicitly so that a
C++ consumer does not acquire a Python build dependency:

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
- deterministic tensor generation and structured comparison;
- `ComparisonOptions`/`ComparisonResult`, logical selection, complex and
  non-finite policy, Frobenius/ULP evidence, allclose search, and sentinel
  diagnostics;
- typed `GenerationRecipe` factories, `generate_tensor`, and `generate_at`;
- `reference_axpby` with optional X/Y tensors, explicit alpha/beta,
  accumulator type, and output type;
- `reference_softmax` with an explicit axis and runtime
  input/output/accumulator types;
- `reference_layer_norm` with optional gamma/beta, mean/inverse-variance
  outputs, epsilon, and an explicit normalized axis;
- `reference_gemm` with runtime storage/output/accumulator types, alpha/beta,
  ordered pre-quantization factors, compute-input quantization, math mode,
  output scaling/conversion, and activation;
- `reference_gemm_result` with the same operation plus backend, fallback, and
  output-element execution evidence;
- `reference_epilogue` with bias, forward/gradient activation, E, scale-D/E,
  gate residual, raw output, and AMax results; and
- `reference_sum` with runtime input/output/accumulator types and explicit
  tensor axes;
- `reference_maximum_absolute` with runtime input/output/accumulator types; and
- `apply_structured_sparsity` and `encode_two_of_four_metadata`, including the
  fused metadata result used by product adapters.

The NumPy suite independently checks:

- every raw FP4, FP6, OCP/FNUZ FP8, E4M3-scale, and E5M3-scale encoding;
- all 65,536 FP16 and BF16 decodings;
- differential raw decoding against matching `ml_dtypes` formats;
- finite low-precision round trips;
- the OCP E8M0 no-zero contract;
- affine layout decoding;
- owned NumPy conversions with gapped and negative affine strides, source-copy
  independence, and explicit Tensor clone behavior;
- deterministic generation, logical index ordering, complex component
  recipes, and structured comparison;
- pointwise, selected, complex, non-finite, Frobenius, ULP, allclose-search,
  and unwritten-sentinel comparison behavior against NumPy;
- F16 stepwise, F32, F64, I32, complex, and blocked GEMM against NumPy;
- mixed FP8-storage/FP4-compute-input quantization;
- selected-output GEMM and prime-stride selection;
- full/selected forward and gradient epilogues for the configured activation
  family against NumPy; and
- multi-axis tensor reduction against `numpy.sum` and max-absolute reduction
  against an explicit NumPy NaN policy; and
- fixed/random N:M pruning, logical compression, packed-value preservation,
  and 2:4 metadata encoding against NumPy.

`Tensor.from_numpy` copies exact native NumPy bool, integer, FP16/FP32/FP64,
and complex64/complex128 storage into an owned Tensor while preserving the
normalized affine shape, strides, and offset. The Tensor does not retain the
ndarray owner and does not observe later ndarray mutations. Packed and
decoded-only formats such as BF16, FP8, FP6, FP4, E8M0, E4M3, and E5M3
continue through the explicit owning conversion path.

`to_numpy` decodes logical tensor values into an owning NumPy array. Its dtype
selects the decoded output container and never reinterprets encoded
`tensor.storage`; E5M3, for example, remains one encoded byte per tensor value
and decodes to float32 by default.

## Standalone tests

```bash
cmake -S shared/host-validation -B build/host-validation \
  -DHOST_VALIDATION_BUILD_TESTING=ON
cmake --build build/host-validation
ctest --test-dir build/host-validation --output-on-failure
```

An optional development benchmark compares the fused component path with a
legacy-shaped two-pass 2:4 implementation using a Tensile-like column-major
layout:

```bash
cmake -S shared/host-validation -B build/host-validation-benchmark \
  -DHOST_VALIDATION_BUILD_TESTING=ON \
  -DHOST_VALIDATION_BUILD_PYTHON=OFF \
  -DHOST_VALIDATION_BUILD_BENCHMARKS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/host-validation-benchmark
OMP_NUM_THREADS=24 \
  build/host-validation-benchmark/tests/host-validation-structured-sparsity-benchmark \
  2048 4096 7
```
