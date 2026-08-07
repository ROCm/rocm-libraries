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
  - Exposes runtime-typed generation, reference GEMM, reference epilogues,
    reductions, structured sparsity, and comparison.
  - Implementation headers live under `roc/host_validation/detail/`; consumers
    include only `validation.hpp`.
- `roc::host-validation-blas`
  - Optional compiled CBLAS implementation of `GemmBackend::Blas`.
  - Built with `HOST_VALIDATION_BUILD_BLAS_BACKEND=ON`.
- `roc::host-validation-tiled`
  - GPU-independent tiled implementation of `GemmBackend::Tiled`.
  - Supports dense F32/F64 accumulation, runtime input/output types,
    compute-input quantization, XFloat32, vector/scalar epilogue operands,
    bias, and activation.
- `roc::host-validation-mx`
  - Optional compiled block-scaled tensor data-generation implementation.
  - Built with `HOST_VALIDATION_BUILD_MX_BACKEND=ON`.
  - Uses the product-independent `mxDataGenerator` backend and exports only
    component-owned scalar, tensor, block-axis, and recipe types.

## Component dependency contract

Every file and target owned by `shared/host-validation` is product-independent.
The component must not include, compile against, or use hipBLASLt, TensileLite,
rocisa, HIP, GPU architecture types, or product enums. Product-private adapters
translate their descriptors into component-owned tensor and operation types:

```text
private product adapter -> roc::host-validation -> roc::host-validation-core
```

The dependency may never point in the opposite direction. The
`host-validation-component-boundary` test scans the complete component source
tree for forbidden product dependencies.

## Layout

```text
include/roc/host_validation/
  tensor.hpp
  validation.hpp
  detail/

src/
python/
tests/
```

The core layer is GEMM- and AMDGPU-agnostic. It contains only:

- `ScalarType` and `ScalarTypeInfo`;
- `Shape`;
- `Layout`;
- owning, runtime-typed `Tensor`;
- non-owning, runtime-typed `TensorView`; and
- non-owning, runtime-typed `MutableTensorView`.

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

Everything else currently exposed through `validation.hpp` is transitional
and may be renamed or replaced as the operation and runtime scalar-type APIs
mature.

## Deterministic tensor generation

`GenerationOptions` describes an index-to-value recipe independently of
storage type, product enums, and traversal parallelism:

```cpp
GenerationOptions options;
options.seed = 17;
options.indexOrder = LogicalIndexOrder::FirstDimensionFastest;
options.real.pattern = GenerationPattern::Sine;
options.imaginary.pattern = GenerationPattern::Cosine;

GenerationRunInfo run = generate(outputView, options);
```

The current patterns cover constants, uniform integer/real values, normal
values, sine/cosine and absolute variants, serial logical indices, one
selected dimension, identity tensors, checkerboard integers, type-derived
extrema/non-finite values, encoded-exponent sampling, and explicit raw-storage
recipes. Raw recipes are used only when compatibility depends on exact scalar
encodings rather than numerical conversion. Real and imaginary components
have independent recipes and random streams. Random values are counter-based,
so a tensor element depends only on the seed, stream, and logical index—not
loop order or thread count.

Each numerical component recipe can also apply one unary transform
(`absolute`, `sine`, or `cosine`), an affine value scale/offset, and sign
alternation over explicitly selected tensor dimensions. These modifiers let
product adapters describe small-value, checkerboard, positive-only, and probe
inputs without owning element loops.

hipBLASLt and TensileLite keep private enum/type adapters. Common host
initialization modes now translate to this API; unsupported legacy raw-bit,
sentinel, and packed-format recipes remain bounded fallbacks while they are
modeled explicitly.

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

- F16, BF16, F32, F64, I32, complex-F32, and complex-F64 accumulation;
- arbitrary runtime storage types supported by the tensor codecs;
- distinct compute-input types for A and B;
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
- canonical execution, pluggable object-oriented backend implementations,
  backend support queries, fallback reporting, and grouped invocation.

Consumers construct this runtime API without passing product-specific types.
The former typed reference GEMM and its function-pointer quantization bridge
have been removed.

The optional `BlasGemmBackend` implements the same interface for dense
F32/F64/complex GEMM and is selected through `GemmRunOptions`.
`TransformingBlasGemmBackend` additionally materializes runtime-typed,
scaled, and compute-input-quantized operands into component-owned scratch,
invokes the ordinary BLAS backend, and performs component-owned output
scaling/conversion. This preserves accelerated large mixed-type references
without placing conversion loops in product adapters.
The canonical backend computes only selected outputs. Accelerated backends may
compute all outputs and report the actual count through `GemmRunInfo`.

`TiledGemmBackend` implements the same object-oriented interface without BLAS
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

The optional MX target generates packed data, natural-layout scales, explicit
per-element scale indices, and the decoded F32 reference tensor:

```cpp
MxGenerationProblem problem;
problem.dataType = ScalarType::Float4E2M1;
problem.scaleType = ScalarType::E8M0;
problem.shape = Shape{64, 128};
problem.leadingDimension = 64;
problem.blockAxis = 0;
problem.blockSize = 32;
problem.data.mode = MxGenerationMode::Bounded;

MxGenerationResult result = generateMx(problem);
```

The operation is GEMM- and architecture-agnostic. A product adapter identifies
which tensor axis is block-scaled and may subsequently transform the natural
scale bytes for an architecture-specific upload layout. GFX950/GFX1250
selection and swizzling are intentionally outside this target. The
`scaleIndices` tensor makes the storage contract explicit, including legacy
flat-buffer tail behavior, and lets NumPy independently verify
`reference == data * scales[scaleIndices]`.

## Runtime reference epilogue

The component also owns the elementwise host epilogue used after reference
GEMM:

```cpp
EpilogueProblem problem(inputView, outputView, ScalarType::Float32);
problem.bias = VectorBinding{biasView, MatrixAxis::Row};
problem.activation = Activation::Gelu;
problem.auxiliaryOutput = auxiliaryView;
problem.rawOutput = rawOutputView;
problem.amax = amaxView;
problem.outputScale = 2.0;
problem.auxiliaryScale = 3.0;

EpilogueRunInfo run = referenceEpilogue(problem);
```

The current operation supports runtime input/output/bias/auxiliary types, F32,
F64, and I32 compute, explicit bias axes, forward and gradient activation,
auxiliary E input/output, scale-D, scale-E, raw output, and AMax. The
hipBLASLt pointer and `hipDataType` translation lives in its private client
adapter and is not compiled by this component.

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
                         outputView,
                         ScalarType::Float32,
                         {0, 2});
ReductionRunInfo run = referenceSum(problem);
```

The current implementation supports F32, F64, I32, complex-F32, and complex-F64
accumulation, runtime input/output storage types, validated signed-strided layouts
used by current consumers,
rank-zero outputs, and multiple reduction axes. hipBLASLt's bias-gradient
adapter represents its matrix as a strided tensor and reduces the K axis; no
product type enters the component.

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
    inputView, prunedView, compressedView, pattern);
problem.twoOfFourMetadata = metadataView; // optional fused output

StructuredSparsityRunInfo run = applyStructuredSparsity(problem);
```

The operation supports:

- fixed or counter-based deterministic random retained positions;
- validated signed-strided input and output layouts used by current consumers;
- in-place pruning;
- byte-preserving copies for ordinary and packed scalar formats;
- optional retained-position indices;
- optional fused 2:4 metadata encoding; and
- independent slice ranges for caller-selected parallel scheduling.

A slice is one logical combination of all coordinates except the sparsity
axis. Product adapters may partition slices with OpenMP or another host
executor without moving pruning, compression, random selection, or metadata
arithmetic back into the product. The component itself has no OpenMP, HIP, or
AMDGPU dependency.

`encodeTwoOfFourMetadata` remains available when retained indices already
exist and metadata must be produced as a separate operation.

Consumers should need one of only two includes:

```cpp
#include <roc/host_validation/tensor.hpp>      // stable tensor core
#include <roc/host_validation/validation.hpp>  // transitional validation operations
```

Installed consumers can use:

```cmake
find_package(ROCHostValidation CONFIG REQUIRED)
target_link_libraries(app PRIVATE roc::host-validation-core)
```

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
- `GenerationOptions`, `GenerationPatternSpec`, and `generate_tensor`;
- `reference_gemm` with runtime storage/output/accumulator types, alpha/beta,
  ordered pre-quantization factors, compute-input quantization, math mode,
  output scaling/conversion, and activation;
- `reference_epilogue` with bias, forward/gradient activation, E, scale-D/E,
  gate residual, raw output, and AMax results; and
- `reference_sum` with runtime input/output/accumulator types and explicit
  tensor axes; and
- `apply_structured_sparsity` and `encode_two_of_four_metadata`, including the
  fused metadata result used by product adapters.

The NumPy suite independently checks:

- every raw FP4, FP6, OCP/FNUZ FP8, and E5M3 encoding;
- all 65,536 FP16 and BF16 decodings;
- finite low-precision round trips;
- the OCP E8M0 no-zero contract;
- affine layout decoding;
- deterministic generation, logical index ordering, complex component
  recipes, and structured comparison;
- F16 stepwise, F32, F64, I32, complex, and tiled GEMM against NumPy;
- mixed FP8-storage/FP4-compute-input quantization;
- selected-output GEMM and prime-stride selection;
- full/selected forward and gradient epilogues for the configured activation
  family against NumPy; and
- multi-axis tensor reduction against `numpy.sum`; and
- fixed/random N:M pruning, logical compression, packed-value preservation,
  and 2:4 metadata encoding against NumPy.

The first binding deliberately copies between NumPy and `Tensor`. A follow-up
should expose lifetime-safe non-owning NumPy-backed `TensorView` objects.

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
