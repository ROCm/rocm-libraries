# HostNumerics

`hostnumerics` is a standalone, CPU-only shared component for generating
tensors, computing reference results, and deciding whether observed results are
numerically acceptable. At its center is a NumPy-like tensor model in which a
scalar type, shape, layout, and storage stay together.

The long-term goal is to serve the host-side numerical needs of projects across
`rocm-libraries` through one clear API. The component is designed to be useful
without HIP or a GPU, independently buildable, and fast enough for large test
problems. Product-specific concepts remain outside the component: hipBLASLt,
TensileLite, rocRoller, and future users translate their own descriptors at a
small adapter boundary.

## Error handling

Expected compatibility, support and descriptor-translation failures use explicit
results or checks. Callers inspect those results to select a backend, materialize
an operand, or reject a descriptor. They must not infer recovery decisions from
an exception type or its message.

Internal fallible helpers return `detail::Expected<T, E>`, a C++20 subset of the
`std::expected` model. Values construct directly; errors use `detail::Unexpected`.
The result provides `[[nodiscard]]`, boolean/`has_value()` checks, value access,
and `error()` for propagation. It retains C++20 consumer compatibility; assignment
and unused standard APIs are omitted. Layout, storage, scalar conversion and
descriptor checks use a closed failure vocabulary; numerical conversion also carries the
destination width and signedness. Diagnostics are obtained from the failure at
reporting boundaries, without error-message output parameters. Checks needing
dynamic diagnostic text can carry an owned string in the same result interface.

The integer-conversion leaf is independent of throwing tensor/metadata headers
and is tested with exception handling disabled and allocation attempts fatal.
Its success and failure paths allocate no storage. Result payload construction
retains its ordinary exception behaviour; the wrapper does not make allocating
operations or arbitrary user conversions nonthrowing.

HostNumerics-owned terminal errors throw `std::runtime_error` with a useful
message. Boundary handlers may report these failures, cleanup may catch and
rethrow, and worker code may transport the original exception to the joining
caller. Allocation, system, dependency and user-callback exceptions retain their
original types. Python sequence/index protocols retain their native exceptions;
HostNumerics-owned operation errors become Python `RuntimeError`.

TensileLite's benchmark client may catch a terminal error to report the current
solution as `INVALID` and continue to the next solution. This is a failed-operation
reporting boundary; it does not select a numerical backend or retry the operation.

This policy does not promise exceptions-disabled builds or transactional output:
a failed operation can have written part of a caller-owned output.

## Incremental landing

The prototype in #10553 serves as an end-to-end integration reference while
the implementation lands in `develop` through small, independently reviewable
PRs. The table tracks the main stages; this prototype also contains the later
stages.

| Status | Stage | Scope |
| --- | --- | --- |
| ✅ [#12208](https://github.com/ROCm/rocm-libraries/pull/12208) | Build and package foundations | CMake targets and installed package, Python module, C++/Python and installed-package smoke tests, dedicated CPU CI, and ownership. |
| In progress | Tensor and datatype core | Tensor model, storage and ownership rules, scalar formats, and conversions, with C++ tests and independent Python coverage. |
| Planned | Numerical operations | Deterministic input generation, reference arithmetic, and comparison, with independent expected results and sanitizer coverage added alongside the implementation. |
| Planned | Consumer migrations and removal | Migrate hipBLASLt, TensileLite, and rocRoller's GEMM paths incrementally. Remove duplicate implementations as their callers migrate, and retire `mxDataGenerator` once its remaining responsibilities and build dependencies have moved. |

Keep each PR to one concept and as small as possible. The standalone stages
develop and test HostNumerics independently of the consumers. They keep it
outside the default monorepo build and provide fast feedback through the
standalone CPU CI jobs. The target is for the component
build and tests to take on the order of two minutes, excluding runner queue and
setup time.

For HostNumerics PRs that have no effect on other components, add the `ci:skip`
and `project: none` labels.

Consumer migrations must preserve existing product tests, supported numerical
behavior, and tolerances. Independent numerical tests should establish the
expected results before the shared implementation becomes a product's sole
reference. Compare representative input-generation and reference-computation
timings, as well as their effect on complete test and benchmark runs, before
removing an old accelerated path.

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

C++ code that handles elements of a runtime `ScalarType` uses
`visitScalarType(type, visitor)` to select code specialized for that encoding
once per operation. The visitor's template parameter supplies `Tag::type`,
`Tag::storageBits`, and `Tag::Storage`. The storage type describes the encoded
representation: `Float16`, `BFloat16`, and `UInt16` all use `uint16_t`, even though
the same bits can represent different numbers. It is not generally a type to
perform arithmetic on, or the inverse of `nativeScalarType`.

For example, this operation counts zeros in a buffer of `Float16`, `BFloat16`,
or `UInt16` elements. Both floating-point formats have positive and negative
zero (`0x0000` and `0x8000`); for `UInt16`, only `0x0000` is zero. The dispatch
selects the storage type and zero mask once, outside the element loop:

```cpp
#include <cstddef>
#include <cstring>
#include <roc/hostnumerics/scalar_type.hpp>
#include <span>
#include <stdexcept>

using namespace roc::hostnumerics;

size_t countZeros16(ScalarType type, std::span<const std::byte> data) {
    return visitScalarType(type, [&]<typename Tag>() -> size_t {
        if constexpr (Tag::type == ScalarType::Float16 ||
                      Tag::type == ScalarType::BFloat16 || Tag::type == ScalarType::UInt16) {
            using Storage = typename Tag::Storage;
            if (data.size() % sizeof(Storage) != 0)
                throw std::runtime_error("Incomplete encoded element.");
            constexpr Storage mask = Tag::type == ScalarType::UInt16 ? 0xffff : 0x7fff;
            size_t zeros = 0;
            for (size_t offset = 0; offset < data.size(); offset += sizeof(Storage)) {
                Storage encoded;
                std::memcpy(&encoded, data.data() + offset, sizeof(encoded));
                zeros += (encoded & mask) == 0;
            }
            return zeros;
        } else {
            throw std::runtime_error("Expected Float16, BFloat16, or UInt16.");
        }
    });
}
```

The input contains encoded elements in host byte order. `memcpy` loads them
without requiring the byte buffer to contain aligned `Storage` objects.
The visitor is instantiated for every supported `ScalarType`, even when a
caller passes a constant type. The `if constexpr` keeps the typed loop out of
unsupported specializations, including sub-byte formats whose `Storage` is
`void`; the explicit `size_t` return type also covers the throwing branch.
An empty buffer returns zero for the three accepted formats.

`scalar_type.hpp` documents the full visitor contract, including forwarding,
reference returns, and rejecting invalid enum values. Use `scalarTypeInfo`
directly when only runtime metadata is needed. A visitor can access metadata
at compile time through `scalarTypeInfo(Tag::type)`.

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
[periodic verification](docs/periodic_verification.md). `prepareFreivalds`
instead accepts ordinary eligible integer-valued inputs and prepares modular
projections, with bounded recomputation for rounded output values. Its
[numerical and observation contracts](docs/freivalds_verification.md) are
independent of device execution. Both verifiers retain immutable preparation
state for reuse across output checks.

`verifyExactActivation`, `verifyExactSum` and `recoverExactMaximumAbsolute`
cover supporting checks on auxiliary outputs and reductions. They accept
owned Tensor handles, retain strict exactness guards, and reuse the ordinary
activation, reduction, conversion and comparison operations. See the
[exact-check contracts](docs/freivalds_verification.md#supporting-exact-checks).

## Deterministic generation

Generation uses an immutable `GenerationRecipe`. A recipe says how each
logical value is produced, which logical index order to use, and which seed to
use for randomized components. For example:

```cpp
using namespace roc::hostnumerics;

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
#include <roc/hostnumerics/gemm.hpp>
#include <roc/hostnumerics/tensor_operations.hpp>

using namespace roc::hostnumerics;

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
import hostnumerics as hn

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

C++ tests use GoogleTest 1.12.1 or newer, retaining the repository's older compatibility baseline because these assertions need no newer API. Top-level test builds default to fetching 1.17.0, matching the shared superbuild, when no compatible installed package is available; set `HOSTNUMERICS_ENABLE_FETCH=OFF` to require an existing package. When a parent enables HostNumerics tests through `add_subdirectory`, fetching defaults to `OFF`: provide a compatible GoogleTest package or target, or explicitly set `HOSTNUMERICS_ENABLE_FETCH=ON`. Existing `GTest::gtest_main` targets must expose their version through the target's `VERSION` property or `GTest_VERSION`. GoogleTest is not installed or exported with HostNumerics. When configuring a package or embedded test project directly, pass `-DHOSTNUMERICS_SOURCE_DIR=/path/to/hostnumerics` so it can locate the shared test sources and CMake helper.

Test numerical behavior primarily through the Python API, using independent
expected results. Keep native core tests in `tests/core_test.cpp` for contracts
that Python cannot exercise, such as C++ construction and ownership rules.

## Python Shape

`Shape` behaves like an immutable Python tuple of nonnegative extents. Integer
indexing returns an extent, including negative positions; slicing selects axes
and returns a new `Shape`. Length, iteration, membership, `count`, `index`, and
hashing follow tuple semantics. Shape values compare equal to other Shape values
with the same extents. They do not compare equal to tuples or lists. Mutation,
concatenation, repetition, and ordering comparisons are not supported.

```python
shape = Shape([2, 3, 5])
shape[-1]       # 5
shape[1:]       # Shape((3, 5))
shape[::-1]     # Shape((5, 3, 2))
tuple(shape)    # (2, 3, 5)
len(shape)      # 3
Shape(range(20))[0:16:5]  # Shape((0, 5, 10, 15))
```

Slicing a Shape selects dimensions, rather than positions within each dimension.
An empty Shape has length zero and represents a rank-zero scalar with one
element. Shape-specific operations such as `element_count`, `project`, and
coordinate conversion retain their logical shape semantics.

Named methods use nonnegative dimension numbers, coordinates, and range bounds:
`extent`, `project`, `element_count_in_range`, and the `order` arguments do not
normalize negative axes. Construction stores the extents without checking their
product. Counts and coordinate conversions check the arithmetic they need, so
`Shape([SIZE_MAX, 2]).linear_index([0, 1])` can return `1` even though its total
count is too large for `size_t` and `coordinates(1)` cannot compute that count.
Here `SIZE_MAX` is the maximum value of `size_t`. A zero extent makes the full
count zero, including when other extents would produce an overflowing product.

## Internal C++ results

`detail::Expected<T, E>` in `result.hpp` represents either a value or an error.
Use `detail::Unexpected(error)` to construct a failure, check `has_value()` or
its boolean conversion, and use `*`/`->` or `error()` only in the matching state.
The wrapper stores its payload inline and adds no allocation or exception
translation. An owning payload can still allocate or throw.

This follows the core `std::expected<T, E>` model while retaining C++20 support;
HostNumerics cannot yet assume C++23 and compatible standard libraries across
its consumers. It deliberately omits assignment, void values, converting result
constructors, throwing `value()` access, fallback access, comparisons, and
monadic operations. Assignment is disabled to keep the stored variant from
becoming valueless after a throwing payload assignment. The utility remains an
implementation helper in `detail`, rather than a new supported public API.

## Build and test


The standalone build requires CMake 3.25.2 or newer, a C++20 compiler, and Ninja.
The Python module also needs Python 3.10 or newer with development headers and
nanobind. The Python test suite also requires NumPy and `ml_dtypes`. From the
repository root, the following creates a virtual environment and build directory
outside the source tree:

```shell
hostnumerics_venv="$PWD/../venvs/hostnumerics"
hostnumerics_build="$PWD/../builds/hostnumerics"
python3 -m venv "$hostnumerics_venv"
"$hostnumerics_venv/bin/python" -m pip install nanobind==3.0.1 numpy ml_dtypes

cmake -S shared/hostnumerics -B "$hostnumerics_build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$hostnumerics_venv/bin/python" \
  -DHOSTNUMERICS_BUILD_TESTING=ON \
  -DHOSTNUMERICS_BUILD_PYTHON=ON
cmake --build "$hostnumerics_build" --parallel 4
ctest --test-dir "$hostnumerics_build" --output-on-failure
```

For a C++-only build, set `HOSTNUMERICS_BUILD_PYTHON=OFF`; Python and nanobind
are then unnecessary.

`HOSTNUMERICS_BUILD_TESTING` and `HOSTNUMERICS_BUILD_PYTHON` both default to `ON`
when building HostNumerics standalone and `OFF` when another project includes it
with `add_subdirectory`. CI covers ON/ON with Ninja Multi-Config and OFF/OFF with
Ninja, including installation and an external C++ consumer for both. It also
checks the embedded defaults.

Install into a local prefix with:

```shell
cmake --install "$hostnumerics_build" --prefix "$hostnumerics_build/install"
```

## CMake integration

Installed consumers normally request the operations component:

```cmake
find_package(HostNumerics CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE roc::hostnumerics)
```

`roc::hostnumerics-core` is a static library containing scalar types and
conversions, plus the tensor model (shape, layout, and storage). It can be used
independently of numerical operations. `roc::hostnumerics` is a static library
that adds generation, reference operations, and comparison, and links core
transitively. The single Python `hostnumerics` package exposes both layers.
`roc::hostnumerics-blas` adds the optional CBLAS GEMM backend, and
`roc::hostnumerics-amd-gpu-layout` provides the independent CPU transforms for
physical MX scale layouts. Build options and their defaults are documented next
to their declarations in CMake.
