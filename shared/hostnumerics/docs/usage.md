# HostNumerics concepts and usage

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

The arrows point from product code into reusable code. `hostnumerics` never
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

`PeriodicInitialization` repeats an encoded seed tensor across a logical shape
of the same rank. It owns a compact snapshot: changing the original seed cannot
change later fills. `fillInto(destination, logicalOrigin)` fills a region using
global coordinates, preserving the destination's layout gaps and packed bits.
For example, a seed of shape `{2, 3}` can fill shape `{5, 7}` in any sequence of
chunks; each value comes from seed coordinate `{row % 2, column % 3}`. The
operation applies to every scalar encoding and does not require a GEMM or
integer-valued data. Convert the seed explicitly before constructing the
initialization if another stored type is required.

`preparePeriodicMatmul` reuses periodic definitions to compute small reference
products and check bounded output regions. It offers exact-integer and
floating-point contracts, described in
[periodic verification](periodic_verification.md). `prepareFreivalds`
instead accepts ordinary eligible integer-valued inputs and prepares modular
projections, with bounded recomputation for rounded output values. Its
[numerical and observation contracts](freivalds_verification.md) are
independent of device execution. Both verifiers retain immutable preparation
state for reuse across output checks.

`verifyExactActivation`, `verifyExactSum` and `recoverExactMaximumAbsolute`
cover supporting checks on auxiliary outputs and reductions. They accept
owned Tensor handles, retain strict exactness guards, and reuse the ordinary
activation, reduction, conversion and comparison operations. See the
[exact-check contracts](freivalds_verification.md#supporting-exact-checks).

## Deterministic generation

Generation uses an immutable `GenerationRecipe`. A recipe says how each
logical value is produced, which logical index order to use, and which seed to
use for randomized components. For example:

```cpp
using namespace roc::hostnumerics;

// Fill a 2-by-3 F32 tensor with reproducible values sampled uniformly from
// the range -1 to 1.
GenerationRecipe recipe = GenerationRecipe::realOnlyLastDimensionFastest(
    GenerationRecipe::uniformReal({.lower = -1.0, .upper = 1.0}), 17);
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

`binomialHalf({.trials = n})` counts successes in `n` independent fair trials,
with `n` from 0 through 64. `withExp2Transform()` maps its result `k` to `2^k`;
an affine value mapping can then shift the exponent by multiplying by a power
of two. `withIndexOffset(offset)` adds to the sampling index while preserving
coordinates and storage addresses. Generation rejects an overflowing sampling
index before writing the destination.

The `generate` and `generateAt` overloads accepting `ScalarConversionOptions`
select source precision and overflow behavior for encoded floating storage.
`FloatingSourcePrecision::Direct` retains the double significand until FP16,
BF16 or narrower destination rounding. `Float32` preserves the intermediate
FP32 conversion. `FloatingOverflow::SaturateFinite` maps infinity to the largest
finite destination value; the default retains each format's existing policy.
Native FP32/FP64 and complex storage use native conversion independently of
these encoding options. `E8M0Underflow::FlushBelowFloat32Normal` maps values
below `2^-126` to raw zero, which decodes as `2^-127` in E8M0. Its default
`Nearest` uses arithmetic midpoint rounding with an even biased exponent.

Recipes cover constants, uniform and normal distributions, indices,
trigonometric patterns, type limits, encoded exponents, and raw storage.
`choice({.values = {...}})` chooses one supplied value for each element;
it is the equivalent of sampling from a finite list, like NumPy's
`random.choice`. Component modifiers can then apply a transform, affine
mapping, or coordinate-based sign pattern without introducing mutable global
generator state.

## Tutorials

The complete C++ and Python walkthroughs are
[`examples/tutorial.cpp`](../examples/tutorial.cpp) and
[`examples/tutorial.py`](../examples/tutorial.py). They progress from basic
tensors through deterministic generation, broadcast arithmetic,
ordinary/integer/complex matrix multiplication, MXFP4 generation, higher-level
operations, zero extents, and comparison diagnostics.

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

An explicitly passed `ComparisonOptions{}` uses zero absolute and relative
tolerances. Caller-supplied tolerances use NumPy's `allclose` relationship:

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

Numerical operations require explicit arithmetic types, traversal axes,
comparison tolerances, generation seeds, execution policies, and selection or
reporting limits. C++ callers pass option objects; Python callers supply the
corresponding arguments. Optional input tensors can be omitted, and NumPy
imports infer the storage type from the array dtype.
`to_numpy` requires a dtype argument; passing `None` selects the decoded mapping
returned by `default_decoded_dtype`.

Shape defines an ordered set of coordinates to visit from either an IndexRange or a `std::set<size_t>` of flat indices, with an axis order passed separately. IndexRange requires a start, exclusive end and positive stride, in that order; `IndexRange::all(end)` supplies the contiguous range `[0, end)`. Explicit sets contain ascending unique indices. Shape checks the complete axis permutation and explicit bounds before traversal, even for empty work. Range traversal streams coordinates without materializing an index list. Layout converts coordinates to storage offsets.

IndexRange::count() counts entries without enumerating them, and IndexRange::indices() materializes those entries only when requested. Both use the end stored in the range; a start at or beyond that end yields no entries. Shape traversal clips ranges to its element count. A set supplies its own size. Numerical invocation boundaries validate and bind the count, order and index lookup without expanding ranges. Shape::forEachCoordinate supplies coordinates to tensor-copy and batch-projection consumers; its callback borrows a reused coordinate span. Python Shape.selected_coordinates returns owning lists. primeIndexRange retains the allocated-element quotient, approximate sampling count and zero-request full coverage policy.

The `hostnumerics` module mirrors the tensor, generation, operation, and
comparison APIs shown in the Python tutorial above. Python operations accept
tensors and ordinary numeric operands directly; there is no public request,
operand, scalar, or result wrapper to construct.

`from_numpy` creates an owning tensor and `to_numpy` returns an owning decoded
array. Packed and custom encodings remain packed in `Tensor.storage`; their
default NumPy representation is a wider decoded type such as `float32`.

Python tensors support integer positions and slices, including negative indices
and steps: `tensor[:, ::2, -1, :]` selects every other position on the second axis
and the last position on the third. Integer selections remove an axis; omitted
trailing axes receive full slices. One ellipsis fills the remaining axes, and
`None` inserts a singleton axis with zero stride. The result shares the original
storage, and writes through either handle are visible through the other. Selecting
all axes returns a rank-zero Tensor; use `.item()` to decode its scalar value.
`len(tensor)` returns the first-axis extent, and iteration lazily selects shared
first-axis views. Rank-zero tensors have neither length nor iteration.
Truth testing uses the numerical value when the tensor has exactly one element;
otherwise it raises an error because the truth value is ambiguous.
Boolean and array indices and indexed assignment are not supported.
The indexing contract preserves selected values and shared storage, rather than
every ndarray metadata convention. Shape and stride properties return lists,
and strides count encoded elements rather than bytes. Empty selections retain
the source offset; empty and singleton axes can have different strides from
NumPy without changing any addressed element. NumPy imports and exports copy,
and broadcast views remain mutable shared Tensor handles. `.item()` requires
a rank-zero Tensor, even when a higher-rank Tensor has only one element.
If a slice stride exceeds the representable range on an empty or singleton axis,
it retains the original stride because that axis never advances through storage.
