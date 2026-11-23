# Installed-package regression tests

The `no_openmp` consumer loads the installed rocWMMA package with both OpenMP package variants disabled,
compiles a HIP kernel, and links and runs host FP16 conversions. It does not launch
a kernel or require an available GPU. It must fail if OpenMP dependency discovery
or an unresolved OpenMP target is restored to the installed package.

The HIP consumer defaults to the `gfx1100` compilation target before enabling
the HIP language, so configuration does not need GPU detection. Override it
with `CMAKE_HIP_ARCHITECTURES`; if only `GPU_TARGETS` is supplied, that value is
used instead. `GPU_TARGETS` otherwise defaults to `CMAKE_HIP_ARCHITECTURES` so
HIP's package targets use the same architecture. No GPU hardware is required,
but the SDK must support the selected compilation target.

From the repository root, build and install rocWMMA without clients. For example
on Linux with ROCm under `/opt/rocm`:

```sh
cmake -S projects/rocwmma -B build/rocwmma-package -G Ninja \
  -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_INSTALL_PREFIX="$PWD/build/rocwmma-install" \
  -DGPU_TARGETS=gfx1100 \
  -DROCWMMA_BUILD_TESTS=OFF -DROCWMMA_BUILD_SAMPLES=OFF \
  -DROCWMMA_BUILD_COMMUNITY_SAMPLES=OFF \
  -DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=ON \
  -DCMAKE_DISABLE_FIND_PACKAGE_openmp=ON
cmake --build build/rocwmma-package --target install

cmake -S projects/rocwmma/test/cmake/package_consumer/no_openmp \
  -B build/rocwmma-consumer -G Ninja \
  -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -Drocwmma_DIR="$PWD/build/rocwmma-install/lib/cmake/rocwmma" \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DGPU_TARGETS=gfx1100
cmake --build build/rocwmma-consumer
ctest --test-dir build/rocwmma-consumer --output-on-failure
```

Use compiler and SDK paths appropriate to the installation on Windows, and a GPU
architecture supported by both the compiler and rocWMMA. The explicit
`rocwmma_DIR` ensures the test consumes the newly installed package. Run the
consumer in a fresh build directory when testing another package installation.
These standalone projects are not registered in the GPU test suite.
When `ROCWMMA_BUILD_TESTS=ON`, their sources are installed in the `tests`
component under `share/rocwmma/tests/package_consumer`. TheRock's
`rocwmma_install` artifact-consumption job configures these installed sources
in separate build directories, builds them, and runs CTest. It runs the FP16
case on Linux and Windows and the FindOpenMP case on Linux, where TheRock
ships LLVM OpenMP.

Each consumer has its own source directory:

```text
package_consumer/
├── README.md
├── tests.json
├── no_openmp/
│   ├── CMakeLists.txt
│   └── main.hip
└── openmp/
    ├── CMakeLists.txt
    └── test_openmp_consumer.cpp
```

## Registering another consumer

Add a standalone CMake project under this directory and register it in
`tests.json`. TheRock reads the installed manifest, so adding a consumer does
not require changing TheRock's runner or CI selection rules. Each entry is
configured in a fresh build directory; do not combine tests of different
package-discovery settings into one CMake configuration, where existing
imported targets could hide a missing dependency.

The manifest contains `"version": 1` and a nonempty `"tests"` array. Each entry
has these fields:

| Field | Meaning | Default |
| --- | --- | --- |
| `name` | Unique lowercase case name, beginning with a letter and containing only letters, digits, `_`, or `-`; Windows device names are reserved | Required |
| `source` | Source directory relative to the manifest, using forward slashes; must remain within this test tree and contain `CMakeLists.txt` | Required |
| `platforms` | Nonempty list containing `linux`, `windows`, or both | Both |
| `hip` | Boolean selecting explicit HIP compiler and architecture configuration | `false` |
| `phase` | Last phase to run: `configure`, `build`, or `test` | `test` |
| `cmake_options` | Object mapping bare CMake cache variable names to string values | `{}` |

For example, append this entry to register a link-only consumer:

```json
{
  "name": "link_check",
  "source": "link_check",
  "hip": true,
  "phase": "build",
  "cmake_options": {
    "CMAKE_DISABLE_FIND_PACKAGE_OpenMP": "ON"
  }
}
```

All cases configure and generate; `build` additionally builds the default
targets, and `test` also runs all registered CTest tests. Each requested phase
must succeed. A full case with no CTest tests fails. Configure-only and
build-only cases do not need dummy CTest tests. Cases must not require GPU
execution, although HIP compilation is supported.

The runner selects the SDK, compilers, Ninja generator, and compilation
architecture. These cache variables cannot be overridden through `cmake_options`:
`CMAKE_PREFIX_PATH`, `rocwmma_DIR`, `hip_DIR`, `GPU_TARGETS`,
`CMAKE_C_COMPILER`, `CMAKE_CXX_COMPILER`, `CMAKE_HIP_COMPILER`,
`CMAKE_RC_COMPILER`, `CMAKE_HIP_COMPILER_ROCM_ROOT`, `CMAKE_HIP_ARCHITECTURES`,
`CMAKE_TOOLCHAIN_FILE`, `ROCM_PATH`, `HIP_PATH`, `HIP_ROOT_DIR`, and
`CMAKE_GENERATOR`. Use bare cache names without `-D` or `:TYPE` suffixes.
Values are passed as individual arguments, without shell expansion; use strings
such as `"ON"` rather than JSON booleans for CMake options.

The test installation includes `CMakeLists.txt`, `*.cpp`, `*.hip`, `*.json`,
`*.cmake`, `*.h`, `*.hpp`, `*.in`, and `README.md`. If a case needs another
file type, extend the installation patterns in rocWMMA's main `CMakeLists.txt`.
TheRock already includes this entire installed directory in its test artifact.

To run the whole registered suite locally against a TheRock SDK, use the same
runner as CI (from any working directory):

```sh
python /path/to/TheRock/build_tools/github_actions/test_executable_scripts/test_rocwmma_install.py \
  --sdk /path/to/installed/sdk --gpu-targets gfx1100 \
  --build-dir /path/to/fresh/consumer-builds
```

The runner prints commands and preserves `configure.log`, `build.log`, and
`test.log` for executed phases under each case's build directory. Platform
exclusions are logged. Malformed manifests, missing sources, no applicable
cases, and previously used case build directories fail before execution.

Use this layout with TheRock's manifest-aware runner. The older hardcoded
runner expects a consumer at the root of this directory and cannot run this
layout. Coordinate the layout change with the runner update and CI pin.
The manifest-aware runner requires `tests.json` and fails explicitly for older
artifacts rather than silently omitting registered tests.

## Consumer that already uses FindOpenMP

The `openmp` project covers the original exported-target regression: a C++
consumer calls `find_package(hip)`, `find_package(OpenMP MODULE)`, and then
`find_package(rocwmma)`. FindOpenMP defines `OpenMP::OpenMP_CXX`; a rocWMMA
package that exports `OpenMP::omp` without loading its defining package fails
at CMake generate time, even though the consumer has already found OpenMP.

Run this case on a system where FindOpenMP can discover a working C++ OpenMP
installation. Use a separate build directory so package targets from one test
cannot mask missing dependencies in the other. Do not preload LLVM's `openmp`
config as a workaround; rocWMMA must resolve its exported dependencies itself.

After the installation step above:

```sh
cmake -S projects/rocwmma/test/cmake/package_consumer/openmp \
  -B build/rocwmma-openmp-consumer -G Ninja \
  -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH="/opt/rocm;/opt/rocm/lib/llvm" \
  -Drocwmma_DIR="$PWD/build/rocwmma-install/lib/cmake/rocwmma" \
  -DGPU_TARGETS=gfx1100
cmake --build build/rocwmma-openmp-consumer
ctest --test-dir build/rocwmma-openmp-consumer --output-on-failure
```

This case requires neither the CMake HIP language nor a GPU. Unlike the first
test, it permits rocWMMA to load LLVM's OpenMP config if its exported interface
requires that target, so both a self-contained OpenMP-dependent package and a
package with no OpenMP dependency can pass.
It defaults `GPU_TARGETS` to `gfx1100` for HIP's exported device target; an
explicit `GPU_TARGETS` overrides that default. Its source is
`openmp/test_openmp_consumer.cpp`.
