# Building and testing HostNumerics

## Requirements

The standalone build requires CMake 3.25.2 or newer and a C++20 compiler. The
commands below use Ninja. The Python module requires Python 3.10 or newer with
development headers and nanobind; CI uses nanobind 3.0.1. The Python test suite
also requires NumPy. C++ tests require
GoogleTest 1.12.1 or newer. See [test dependencies](#test-dependencies) for
installed and fetched dependency options.

Test numerical behavior primarily through the Python API, using independent
expected results. Keep native core tests in `tests/core_test.cpp` for contracts
that Python cannot exercise, such as C++ construction and ownership rules.

## Standalone build

From the repository root, create a virtual environment and build directory
outside the source tree:

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
ctest --test-dir "$hostnumerics_build" --no-tests=error --output-on-failure
```

For a C++-only build, set `HOSTNUMERICS_BUILD_PYTHON=OFF`; Python and nanobind
are then unnecessary. `HOSTNUMERICS_BUILD_TESTING` and
`HOSTNUMERICS_BUILD_PYTHON` default to `ON` for a standalone build and `OFF`
when a parent includes HostNumerics with `add_subdirectory`. For a multi-config
generator such as Ninja Multi-Config, select the same configuration with
`--config Release` when building or installing and `-C Release` with CTest.

## Test dependencies

When no compatible installed GoogleTest package or existing target is
available, standalone test builds fetch GoogleTest 1.17.0. Set
`HOSTNUMERICS_ENABLE_FETCH=OFF` to require an existing dependency instead.
Fetching defaults to `OFF` when HostNumerics is embedded in another project;
provide a compatible package or target, or explicitly enable fetching.

An existing `GTest::gtest_main` target must expose its version through its
`VERSION` property or `GTest_VERSION`. GoogleTest is a test dependency and is
not installed or exported with HostNumerics. The dependency lookup and version
checks are defined in
[HostNumericsTestDependencies.cmake](../cmake/HostNumericsTestDependencies.cmake).

## Installation and CMake integration

Install into a local prefix:

```shell
cmake --install "$hostnumerics_build" --prefix "$hostnumerics_build/install"
```

Add the install prefix to the consuming project's `CMAKE_PREFIX_PATH`, then
load and link the package:

```cmake
find_package(HostNumerics CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE roc::hostnumerics)
```

The installed package exports `roc::hostnumerics-core` and
`roc::hostnumerics`. Both are currently `INTERFACE` libraries exposing version
information and scalar encoding metadata. The operations target
links core transitively. Their intended responsibilities as implementation
lands are:

| Target | Responsibility |
| --- | --- |
| `roc::hostnumerics-core` | Scalar types and conversions, plus shape, layout, and tensor storage. |
| `roc::hostnumerics` | Input generation, reference operations, and numerical comparison. |

The Python package installs under the prefix's library directory in
`python<major>.<minor>/site-packages`. Add that directory to `PYTHONPATH` to
import the installed module with the Python interpreter used for the build.

## CI and package validation

The [standalone workflow](../../../.github/workflows/hostnumerics-ci.yml) covers
Python/tests ON with Ninja Multi-Config and OFF with Ninja, installation,
external C++ consumers, installed Python imports, embedded defaults,
ASan/UBSan, and clang-tidy. These jobs run on Ubuntu.

CI also builds and tests native C++ on Windows with MSVC and Ninja, installs
HostNumerics, and builds and tests an external consumer of the installed package.
Python bindings are disabled in this Windows job; Linux covers binding builds
and installed imports, with Windows Python coverage left for a follow-up.

The [package test project](../tests/package/CMakeLists.txt) and
[embedded test project](../tests/embedded/CMakeLists.txt) can also be configured
directly. Pass `-DHOSTNUMERICS_SOURCE_DIR=/path/to/hostnumerics` so they can find
the shared test source and dependency helper. The package test project also
needs the install prefix in `CMAKE_PREFIX_PATH`.
