# HostNumerics

HostNumerics is a shared CPU-only component for generating test inputs,
computing numerical reference results, and comparing them with observed results.
It is being introduced incrementally. Build, packaging, test infrastructure,
and scalar type metadata are available. The full numerical implementation and
consumer integrations are demonstrated in the
[prototype PR #10553](https://github.com/ROCm/rocm-libraries/pull/10553).

## Intended end state

hipBLASLt, TensileLite, and rocRoller's GEMM paths currently contain overlapping
input generation, datatype conversion, reference computation, and comparison
code. The goal is for all three to use HostNumerics for this shared CPU work,
with the same APIs available to other rocm-libraries components as they adopt
it. New datatype support and numerical fixes can then be implemented and tested
in one place.

The shared API will use a NumPy-like tensor model that keeps datatype, shape,
layout, and storage together. It will cover ordinary and packed low-precision
types, deterministic generation, reference operations such as GEMM and its
epilogues, and numerical comparison. Consumers will use the C++ API; Python
bindings will expose the same numerical behavior for independent tests using
NumPy, `ml_dtypes`, encoded values, and Python integer arithmetic. The component
will build and test without HIP, a GPU toolchain, or GPU hardware.

HostNumerics will subsume `mxDataGenerator`. Its CPU generation and datatype
support will become part of the shared numerical APIs. CPU transforms that
construct physical AMD GPU layouts, such as scale swizzling and pre-tiling,
will live behind a separate API within HostNumerics, keeping architecture
details out of generic tensors and numerical operations. Once all callers and
build dependencies have migrated, the old `mxDataGenerator` component will be
removed.

Each consuming product will translate its descriptors and buffers through a
small private adapter. Products will continue to own GPU allocation, execution,
transfers, tolerance selection, and reporting. The rocRoller migration covers
its GEMM validation paths; its general kernel-generation functionality stays
with rocRoller.

## Incremental landing

The prototype in #10553 serves as an end-to-end integration reference while
the implementation lands in `develop` through small, independently reviewable
PRs. The main stages are:

| Status | Stage | Scope |
| --- | --- | --- |
| ✅ [#12208](https://github.com/ROCm/rocm-libraries/pull/12208) | Build and package foundations | CMake targets and installed package, Python module, C++/Python and installed-package smoke tests, dedicated CPU CI, and ownership. |
| In progress | Tensor and datatype core | Tensor model, storage and ownership rules, scalar formats, and conversions, with C++ tests and independent Python coverage. |
| Planned | Numerical operations | Deterministic input generation, reference arithmetic, and comparison, with independent expected results and sanitizer coverage added alongside the implementation. |
| Planned | Consumer migrations and removal | Migrate hipBLASLt, TensileLite, and rocRoller's GEMM paths incrementally. Remove duplicate implementations as their callers migrate, and retire `mxDataGenerator` once its remaining responsibilities and build dependencies have moved. |

Keep each PR to one concept and as small as possible. The standalone stages
develop and test HostNumerics independently of the consumers. They keep it
outside the default monorepo build and provide fast
feedback through the standalone CPU CI jobs. The target is for the component
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

C++ tests use GoogleTest 1.12.1 or newer, retaining the repository's older compatibility baseline because these assertions need no newer API. Top-level test builds default to fetching 1.17.0, matching the shared superbuild, when no compatible installed package is available; set `HOSTNUMERICS_ENABLE_FETCH=OFF` to require an existing package. When a parent enables HostNumerics tests through `add_subdirectory`, fetching defaults to `OFF`: provide a compatible GoogleTest package or target, or explicitly set `HOSTNUMERICS_ENABLE_FETCH=ON`. Existing `GTest::gtest_main` targets must expose their version through the target's `VERSION` property or `GTest_VERSION`. GoogleTest is not installed or exported with HostNumerics.

Test numerical behavior primarily through the Python API, using independent
expected results. Keep native core tests in `tests/core_test.cpp` for contracts
that Python cannot exercise, such as C++ construction and ownership rules.

## Error handling

HostNumerics-owned terminal errors throw `std::runtime_error` with a useful
message. Expected compatibility and support failures should use explicit checks
or results; callers must not select recovery from exception types or messages.
Allocation, system, dependency and user-callback errors retain their original
types. Python tuple operations retain their native errors, while named
HostNumerics operations report `RuntimeError`. This policy does not promise an
exceptions-disabled build.

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
normalize negative axes. Construction requires the product of all nonzero extents
to fit in `size_t`, so both `Shape([SIZE_MAX, 2])` and
`Shape([SIZE_MAX, 2, 0])` fail immediately. Here `SIZE_MAX` is the maximum value
of `size_t`. Zero extents still describe empty shapes: `Shape([SIZE_MAX, 1, 0])`
has zero elements. This invariant guarantees that every dimension subrange,
projection, and slice has a representable element count. Counts and index
conversions need no further overflow checks; their arguments are validated
before computing the result.

Axes are numbered by their positions in `dimensions`. For `[2, 3, 5]`, axis 0
has extent 2, axis 1 has extent 3, and axis 2 has extent 5. `order` lists axes
from outermost to innermost loop, so its last entry changes fastest. Empty order
selects `[0, 1, 2]`, the row-major convention with the last axis fastest.

| Order | Fastest axis | Logical strides for axes 0, 1, 2 | Index of `[1, 2, 3]` |
| --- | --- | --- | --- |
| `[0, 1, 2]` (default) | 2 | `[15, 5, 1]` | 28 |
| `[2, 1, 0]` | 0 | `[1, 2, 6]` | 23 |
| `[1, 2, 0]` | 0 | `[1, 10, 2]` | 27 |

Thus `[1, 2, 0]` means a loop over axis 1 containing a loop over axis 2
containing a loop over axis 0. Both `linear_index` and `coordinates` use this
same order; it is independent of physical storage strides.

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
nanobind. The Python test suite also requires NumPy. From the repository root,
the following creates a virtual environment and build directory outside the
source tree:

```shell
hostnumerics_venv="$PWD/../venvs/hostnumerics"
hostnumerics_build="$PWD/../builds/hostnumerics"
python3 -m venv "$hostnumerics_venv"
"$hostnumerics_venv/bin/python" -m pip install nanobind==3.0.1 numpy

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
checks the embedded defaults and an installed GoogleTest package with fetching disabled. When configuring the package or embedded test project directly, pass `-DHOSTNUMERICS_SOURCE_DIR=/path/to/hostnumerics` so it can locate the shared test source and CMake helper.

Install into a local prefix with:

```shell
cmake --install "$hostnumerics_build" --prefix "$hostnumerics_build/install"
```

The installed `HostNumerics` CMake package exports two C++ targets with these
intended responsibilities:

- `roc::hostnumerics-core` will provide scalar types and conversions, plus the
  tensor model (shape, layout, and storage). It can be used independently of
  numerical operations.
- `roc::hostnumerics` will add input generation, reference operations such as
  GEMM, and numerical comparison. It links core transitively, so consumers
  needing these operations only need to link `roc::hostnumerics`.

Both targets currently use `INTERFACE` libraries exposing version and scalar
type metadata. The prototype implements them as static libraries.

Consumers can add the install prefix to `CMAKE_PREFIX_PATH` and use:

```cmake
find_package(HostNumerics CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE roc::hostnumerics)
```

The Python bindings expose scalar metadata and `__version__` through the single
`hostnumerics` package; both layers will use that package as they land.

`ScalarType` identifies an encoding, and `scalarTypeInfo(type)` describes its
category, storage width, exponent, fraction, and special-value support. Python
exposes the same metadata through `scalar_type_info(type)`.

`Layout` owns a `Shape`, signed strides, and a starting offset. Strides and
offsets count elements rather than bytes: `elementOffset(indices)` computes
`offset + sum(indices[d] * strides[d])`. For shape `[2, 3]`, C order has strides
`[3, 1]` and Fortran order has `[1, 2]`. Negative strides describe reversal;
zero strides describe repeated addresses. Layout does not own storage or check
allocation bounds. Coordinate conversion checks rank, bounds, and intermediate
arithmetic overflow; production loops should reuse a prepared mapping after
validation. Python exposes the same interface as `Layout`, with
`element_offset` and copied `shape`/`strides` properties.

Internal fallible layout helpers return `detail::Expected<Layout, LayoutError>`.
Callers can inspect the returned error before choosing a recovery path; the
throwing contiguous factories translate it into a terminal C++ error.
