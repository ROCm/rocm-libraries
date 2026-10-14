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

## Build and test

The standalone build requires CMake 3.25.2 or newer, a C++20 compiler, and Ninja.
The Python module also needs Python 3.10 or newer with development headers and
nanobind. From the repository root, the following creates a virtual environment
and build directory outside the source tree:

```shell
hostnumerics_venv="$PWD/../venvs/hostnumerics"
hostnumerics_build="$PWD/../builds/hostnumerics"
python3 -m venv "$hostnumerics_venv"
"$hostnumerics_venv/bin/python" -m pip install nanobind==3.0.1

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
