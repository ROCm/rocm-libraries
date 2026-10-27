# Installed-package regression tests

This test loads the installed rocWMMA package with both OpenMP package variants disabled,
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

cmake -S projects/rocwmma/test/cmake/package_consumer \
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
