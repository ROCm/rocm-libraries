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
broadcasting rules. Operations that support broadcasting, such as
`linearCombination`, therefore consume ordinary tensors without separate
axis or replication descriptors.

`ScalarType` includes ordinary integer, floating-point, and complex types as
well as the packed FP4, FP6, and Int4 encodings and the scale formats used by
MX. Strides are measured in logical elements even when several encoded values
share one byte.

`Scalar` represents one runtime-typed numerical value, primarily an operation
coefficient such as alpha or beta. It is deliberately distinct from a rank-zero
`Tensor`: it has ordinary value semantics, stores its encoding inline, and has
no shape, layout, aliasing, or shared-lifetime behavior. A sub-byte scalar still
occupies a private whole-byte slot; only the format's defined low bits belong to
the value. This distinction is about semantics and API clarity rather than a
measured performance requirement, and mirrors the distinction between a NumPy
scalar and a zero-dimensional `ndarray`. Operation coefficients accept native
numbers directly; `Tensor::item()` snapshots a zero-dimensional tensor when a
tensor-backed scalar is more convenient, and `Tensor::item<T>()` returns a
chosen native C++ type directly.

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

## Reference operations

The component provides CPU references for GEMM, GEMM epilogues, linear tensor
combinations, softmax, LayerNorm, reductions, and structured sparsity. Storage,
compute, accumulator, and result types stay explicit because the purpose is to
model low-precision behavior rather than silently promote every calculation to
the host's preferred type.

Operations have two forms. The ordinary form accepts tensors and options, then
allocates and returns its output tensors. An `...Into` form accepts
caller-owned destinations when a product needs a particular layout, wants
in-place operation where it is valid, or needs only selected outputs. Product
adapters translate raw pointers and enums before calling either form.

`linearCombination` implements `alpha * x + beta * y` with NumPy-style input
broadcasting for the hipBLASLt matrix-transform reference while sharing the
component's conversion, layout, ownership, and aliasing rules. It is a small
operation rather than a parallel tensor-algebra framework.

Reference GEMM supports ordinary and complex arithmetic, explicit
low-precision input quantization and accumulation behavior, scaling, bias,
activation, block scales, and selected-output validation. A built-in blocked
CPU implementation accelerates common cases, and an optional CBLAS backend can
accelerate compatible dense problems. Backend choice changes execution, not the
numerical request.

## Numerical comparison

Comparison consumes two tensors and a policy, then returns structured evidence
rather than printing or depending on a test framework. Policies cover exact,
absolute, relative, symmetric-relative, and ULP comparisons; NaN, infinity,
and signed-zero behavior; norms; selected logical elements; and unwritten
sentinel regions. Product code decides how to render the result and attach its
own problem context.

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

The `roc_host_numerics` module exposes the same scalar types, tensors,
generation recipes, reference operations, and comparison results. NumPy
conversion functions copy values so ownership is unambiguous:

```python
import numpy as np
import roc_host_numerics as hv

a_np = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
b_np = np.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
c_np = np.zeros((2, 2), dtype=np.float32)

a = hv.from_numpy(a_np)
b = hv.from_numpy(b_np)
c = hv.from_numpy(c_np)
d = hv.reference_gemm(
    a,
    b,
    c,
    output_type=hv.ScalarType.Float32,
    accumulator_type=hv.ScalarType.Float32,
)
d_np = hv.to_numpy(d)
```

`from_numpy` creates an owning tensor and `to_numpy` returns an owning decoded
array. Packed and custom encodings remain packed in `Tensor.storage`; their
default NumPy representation is a wider decoded type such as `float32`.

## CMake integration

Installed consumers normally request the operations component:

```cmake
find_package(ROCHostNumerics CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE roc::host-numerics)
```

`roc::host-numerics-core` contains only the tensor and scalar model.
`roc::host-numerics` adds generation, reference operations, and comparison.
`roc::host-numerics-blas` adds the optional CBLAS GEMM backend, and
`roc::host-numerics-amd-gpu-layout` provides the independent CPU transforms for
physical MX scale layouts. Build options and their defaults are documented next
to their declarations in CMake.
