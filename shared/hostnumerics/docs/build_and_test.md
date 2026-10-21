# Building and using HostNumerics

C++ tests use GoogleTest 1.12.1 or newer, retaining the repository's older compatibility baseline because these assertions need no newer API. Top-level test builds default to fetching 1.17.0, matching the shared superbuild, when no compatible installed package is available; set `HOSTNUMERICS_ENABLE_FETCH=OFF` to require an existing package. When a parent enables HostNumerics tests through `add_subdirectory`, fetching defaults to `OFF`: provide a compatible GoogleTest package or target, or explicitly set `HOSTNUMERICS_ENABLE_FETCH=ON`. Existing `GTest::gtest_main` targets must expose their version through the target's `VERSION` property or `GTest_VERSION`. GoogleTest is not installed or exported with HostNumerics.

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
