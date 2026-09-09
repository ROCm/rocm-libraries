# ROCm host numerics

`host-numerics` is a standalone, CPU-only shared component for generating
tensors, computing reference results, and deciding whether observed results are
numerically acceptable. At its center is a NumPy-like tensor model in which a
scalar type, shape, layout, and storage stay together.

The long-term goal is to serve the host-side numerical needs of projects across
`rocm-libraries` through one clear API. The component is designed to be useful
without HIP or a GPU, independently buildable, and fast enough for large test
problems. Product-specific concepts remain outside the component: hipBLASLt,
TensileLite, rocRoller, and future users translate their own descriptors at a
small adapter boundary.

## Mental model

```text
 product descriptors and buffers
              |
       product-owned adapter
              |
              v
  ScalarType + Shape + Layout + Tensor
              |
       +------+------+----------+
       |             |          |
       v             v          v
  generation   reference math  comparison
       |
       +---- block-scaled (MX) generation
                         |
                         v
            optional AMD GPU layout transform
```

The arrows point from product code into reusable code. `host-numerics` never
depends on a consuming product, HIP, or a GPU runtime. The AMD GPU layout
module is a separate CPU implementation that rearranges already-generated
bytes into an architecture's physical format; generic tensors and numerical
operations do not depend on it.

## Tensors and layouts

A `Tensor` combines four things that should not travel as unrelated arguments:
the `ScalarType` of each element, a logical `Shape`, an affine `Layout`, and the
encoded backing storage. The layout maps logical coordinates to storage
offsets, so the same tensor model represents row-major and column-major data,
padding, batches, transposed views, offsets, and negative strides.

Tensor storage is owned or lifetime-anchored. Copying a `Tensor` creates a
shallow handle to the same bytes; `deepCopy()` makes an independent value.
Factories also support copying native values, preserving exact encoded bytes,
or sharing externally owned mutable storage. This makes ownership explicit at
the boundary instead of passing an untracked pointer beside separate type and
stride metadata.

`broadcastTo()` creates a shallow zero-stride view using NumPy's trailing-axis
broadcasting rules. Elementwise `add` and `multiply` therefore consume ordinary
tensors without separate axis or replication descriptors.

`ScalarType` includes ordinary integer, floating-point, and complex types as
well as the packed FP4, FP6, and Int4 encodings and the scale formats used by
MX. Strides are measured in logical elements even when several encoded values
share one byte.

Rank-zero tensors represent runtime-typed numerical values without a parallel
scalar container. Operations also accept ordinary native C++ numbers where that
is convenient; they are converted to a rank-zero tensor of the other operand's
type. `Tensor::item<T>()` returns a rank-zero value as a chosen native C++ type.

## Deterministic generation

Generation uses an immutable `GenerationRecipe`. A recipe says how each
logical value is produced, which logical index order to use, and which seed to
use for randomized components. For example:

```cpp
using namespace roc::host_numerics;

// Fill a 2-by-3 F32 tensor with reproducible values sampled uniformly from
// the range -1 to 1.
GenerationRecipe recipe = GenerationRecipe::realOnly(
    GenerationRecipe::uniformReal({.lower = -1.0, .upper = 1.0}),
    {.seed = 17});
Tensor values = generate(ScalarType::Float32, Shape{2, 3}, recipe);
```

The caller owns seed selection. Each generation call sees exactly one explicit
seed; the component neither advances caller state nor derives named streams.
Callers that want stable values for several operands can assign separate seeds
directly:

```cpp
generate(a, recipe.withSeed(seed + 0));
generate(b, recipe.withSeed(seed + 10));
if (useBias)
    generate(bias, recipe.withSeed(seed + 20));
generate(c, recipe.withSeed(seed + 30));
```

The complete seed is mixed by the counter-based generator, so adjacent seeds
are valid. A generated element depends on the seed and its logical index, not
on loop order or thread count. Complex Cartesian generation uses an internal
separation between real and imaginary components, but that implementation
detail is not part of the caller's seed contract.

Recipes cover constants, uniform and normal distributions, indices,
trigonometric patterns, type limits, encoded exponents, and raw storage.
`choice({.values = {...}})` chooses one supplied value for each element;
it is the equivalent of sampling from a finite list, like NumPy's
`random.choice`. Component modifiers can then apply a transform, affine
mapping, or coordinate-based sign pattern without introducing mutable global
generator state.

## Tutorials

The complete C++ and Python walkthroughs are
[`examples/tutorial.cpp`](examples/tutorial.cpp) and
[`examples/tutorial.py`](examples/tutorial.py). They progress from basic
tensors through deterministic generation, broadcast arithmetic,
ordinary/integer/complex matrix multiplication, MXFP4 generation, higher-level
operations, zero extents, and comparison diagnostics.

### C++

Create tensors from native values, multiply them, and compose the result with
ordinary broadcast operations:

```cpp
#include <array>
#include <roc/host_numerics/gemm.hpp>
#include <roc/host_numerics/tensor_operations.hpp>

using namespace roc::host_numerics;

const std::array<float, 6> aValues{1, 2, 3, 4, 5, 6};
const std::array<float, 6> bValues{7, 8, 9, 10, 11, 12};
Tensor a = Tensor::copyNativeValues<float>(Shape{2, 3}, aValues);
Tensor b = Tensor::copyNativeValues<float>(Shape{3, 2}, bValues);

Tensor product = matmul(a, b, ScalarType::Float32);
Tensor bias = Tensor::copyNativeValues<float>(
    Shape{2}, std::array<float, 2>{-100.0f, 1.0f});
Tensor result = relu(product * 0.5f - bias);
```

The last dimension of `bias` broadcasts over the columns. Native `0.5f` is
converted to the other operand's element type. A rank-zero tensor can be
supplied instead when its encoded type matters.

Generation is similarly tensor-first:

```cpp
GenerationRecipe recipe = GenerationRecipe::realOnly(
    GenerationRecipe::uniformReal({.lower = -1.0, .upper = 1.0}),
    {.seed = 17});
Tensor random = generate(ScalarType::Float32, Shape{2, 2}, recipe);
```

Products needing a particular destination layout or a sparse validation
selection use the corresponding `...Into` operation. The ordinary forms own
and return their outputs.

### Python

The same basic expression uses Python operators and ordinary scalars:

```python
import numpy as np
import roc_host_numerics as hn

a = hn.from_numpy(np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
b = hn.from_numpy(np.asarray([[7, 8], [9, 10], [11, 12]], dtype=np.float32))
bias = hn.from_numpy(np.asarray([-100, 1], dtype=np.float32))

result = hn.relu(hn.matmul(a, b) * 0.5 + bias)
```

## Reference operations

The component provides CPU references for matrix multiplication, elementwise
tensor arithmetic and activations, product epilogues, softmax, LayerNorm,
reductions, and structured sparsity. Storage, compute, accumulator, and result
types stay explicit because the purpose is to model low-precision behavior
rather than silently promote every calculation to the host's preferred type.

Operations have two forms. The ordinary form accepts tensors and options, then
allocates and returns its output tensors. An `...Into` form accepts
caller-owned destinations when a product needs a particular layout, wants
in-place operation where it is valid, or needs only selected outputs. Product
adapters translate raw pointers and enums before calling either form.

Zero-length dimensions are valid. A matrix multiplication with zero M or N
does no work, while zero K produces the additive-identity product. Product
adapters compose any C or epilogue terms afterward and preserve a zero batch
count as empty work.

`add`, `multiply`, and named activations such as `relu`, `gelu`, and `clip`
follow NumPy-style trailing-dimension broadcasting. Product adapters express
`alpha * product + beta * c` by composing those operations; it is not encoded
as a special GEMM request.

`matmul` supports ordinary and complex arithmetic, explicit low-precision input
quantization and accumulation behavior, block scales, and selected outputs. A
built-in blocked CPU implementation accelerates common cases, and an optional
CBLAS backend can accelerate compatible dense problems. Backend choice changes
execution, not the numerical request. Scales, bias, activation, and output
conversion are separate tensor or epilogue operations.

## Numerical comparison

Comparison consumes two tensors and a policy, then returns structured evidence
rather than printing or depending on a test framework. Policies cover exact,
absolute, relative, and ULP comparisons; NaN and infinity behavior; norms;
selected logical elements; and unwritten sentinel regions. Product code decides
how to render the result and attach its own problem context.

Default relative and absolute tolerances follow the component's documented
type policy, while explicit tolerances use NumPy's `allclose` relationship:

```text
absolute_difference <= absolute_tolerance
                     + relative_tolerance * abs(expected)
```

## Block-scaled MX data

MX formats store low-precision data together with one scale shared by a block
of elements. `generateMx` produces the packed data tensor, a natural-layout
scale tensor, a per-element map to those scales, and a decoded F32 reference.
This keeps the generated encoding and the mathematical value available from
one result.

Data generation and scale generation are orthogonal. `MxDataGeneration`
controls source values and how they are quantized into the data format.
`MxScaleGenerationMode` controls only how block scales are selected—for
example, deriving each scale from its block or using a fixed diagnostic value.
Changing the scale mode does not select a different random data stream.

Natural scale layout is architecture-independent. Products that need a
GFX950- or GFX1250-specific physical scale layout pass the natural bytes to the
separate AMD GPU layout target. That boundary keeps GPU storage conventions out
of the tensor and reference-operation layers.

## Python use

The `roc_host_numerics` module mirrors the tensor, generation, operation, and
comparison APIs shown in the Python tutorial above. Python operations accept
tensors and ordinary numeric operands directly; there is no public request,
operand, scalar, or result wrapper to construct.

`from_numpy` creates an owning tensor and `to_numpy` returns an owning decoded
array. Packed and custom encodings remain packed in `Tensor.storage`; their
default NumPy representation is a wider decoded type such as `float32`.

## CMake integration

Installed consumers normally request the operations component:

```cmake
find_package(ROCHostNumerics CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE roc::host-numerics)
```

`roc::host-numerics-core` contains the tensor and scalar-type model.
`roc::host-numerics` adds generation, reference operations, and comparison.
`roc::host-numerics-blas` adds the optional CBLAS GEMM backend, and
`roc::host-numerics-amd-gpu-layout` provides the independent CPU transforms for
physical MX scale layouts. Build options and their defaults are documented next
to their declarations in CMake.
