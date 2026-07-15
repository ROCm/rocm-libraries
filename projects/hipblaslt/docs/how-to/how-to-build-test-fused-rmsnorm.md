# Build and Test Fused RMSNorm E2E

This guide summarizes the end-to-end workflow for building hipBLASLt, generating
the gfx950 fused RMSNorm device libraries, and running the focused GoogleTest
coverage:

```bash
./clients/hipblaslt-test --gtest_filter='FusedEpilogue*.*'
```

The flow is useful when validating the composable fused-epilogue RMSNorm path:
full RMSNorm, residual-add plus RMSNorm, the decomposed producer
(`PARTIAL_RMSNORM_STATS`), and the decomposed consumer (`RMSNORM_SCALE_APPLY`,
Kernel 3 / RstdScale).

## Prerequisites

Use a ROCm SDK that contains `amdclang++`, HIP, amd-smi, hipBLAS common, and the
gfx950 runtime libraries. In TheRock-style environments, set:

```bash
export ROCM_PATH=/home/ossci/therock-tarball/install
export ROCM_HOME="$ROCM_PATH"
export PATH="$ROCM_PATH/bin:$ROCM_PATH/lib/llvm/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib/llvm/lib:${LD_LIBRARY_PATH:-}"
```

Confirm that the target GPU is visible:

```bash
rocm_agent_enumerator
rocminfo
```

The expected target for these tests is `gfx950`.

## Python Environment

TensileLite uses Python tools during device-library generation. Create a local
environment and install the TensileLite requirements:

```bash
cd /path/to/rocm-libraries/projects/hipblaslt
python3 -m venv build/python-venv
build/python-venv/bin/python -m pip install --upgrade pip
build/python-venv/bin/python -m pip install -r tensilelite/requirements.txt
```

Configure CMake with this interpreter:

```bash
-DPython_EXECUTABLE=$PWD/build/python-venv/bin/python
-DPython3_EXECUTABLE=$PWD/build/python-venv/bin/python
```

The examples below use one build directory throughout:

```bash
export BUILD_DIR=build/release
```

If `$BUILD_DIR` was already configured with a different CMake generator, either
remove its `CMakeCache.txt`/`CMakeFiles` or choose a fresh directory, for example
`export BUILD_DIR=build/fused-rmsnorm-release`.

## Compiler Wrapper

Some ROCm clang builds need an explicit GCC install directory to find the host
C++ standard library. The rocisa assembler capability probes also fail if the
compiler prints warnings to stdout or stderr. Use a quiet wrapper:

```bash
mkdir -p build/toolchain
cat > build/toolchain/amdclang++ <<'SH'
#!/usr/bin/env bash
has_assembler=0
prev=
for arg in "$@"; do
  if [ "$prev" = "-x" ] && [ "$arg" = "assembler" ]; then
    has_assembler=1
    break
  fi
  prev="$arg"
done
if [ "$has_assembler" = 1 ]; then
  exec /home/ossci/therock-tarball/install/bin/amdclang++ -Wno-gcc-install-dir-libstdcxx "$@"
fi
exec ccache /home/ossci/therock-tarball/install/bin/amdclang++ \
  -Wno-gcc-install-dir-libstdcxx \
  --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13 "$@"
SH
chmod +x build/toolchain/amdclang++
```

After configuring the build and building the rocisa target below, verify that
rocisa recognizes gfx950 as supported:

```bash
PYTHONPATH="$PWD/$BUILD_DIR/tensilelite/rocisa:$PWD/$BUILD_DIR/tensilelite:$PWD/tensilelite" \
build/python-venv/bin/python - <<'PY'
from Tensile.Common.Types import IsaVersion
from Tensile.Common.Capabilities import makeIsaInfoMap
compiler = "build/toolchain/amdclang++"
info = makeIsaInfoMap([IsaVersion(9, 5, 0)], compiler)[IsaVersion(9, 5, 0)]
print(info.asmCaps["SupportedISA"])
PY
```

The output should be `1`.

## Local Build Dependencies

If system GTest, BLAS, and LAPACK are unavailable, build the local dependency
bundle:

```bash
mkdir -p build/deps
cmake -S deps -B build/deps \
  -D CMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -D CMAKE_INSTALL_PREFIX=$PWD/build/deps/install \
  -D CMAKE_INSTALL_LIBDIR=lib \
  -D BUILD_LAPACK=ON \
  -D BUILD_GTEST=ON
cmake --build build/deps --target googletest lapack --parallel 16
cmake --build build/deps --target install --parallel 16
```

Build msgpack-c into the same prefix for msgpack-mode libraries:

```bash
git clone -b cpp-3.1.0 https://github.com/msgpack/msgpack-c.git build/deps/msgpack-c --depth 1
cmake -S build/deps/msgpack-c -B build/deps/msgpack-c-build \
  -D CMAKE_INSTALL_PREFIX=$PWD/build/deps/install \
  -D CMAKE_INSTALL_LIBDIR=lib \
  -D MSGPACK_BUILD_TESTS=OFF \
  -D MSGPACK_BUILD_EXAMPLES=OFF \
  -D MSGPACK_CXX17=ON
cmake --build build/deps/msgpack-c-build --target install --parallel 16
```

## Configure hipBLASLt

Configure a focused gfx950 build:

```bash
cmake -S . -B "$BUILD_DIR" -G Ninja \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_CXX_COMPILER=$PWD/build/toolchain/amdclang++ \
  -D CMAKE_C_COMPILER=$ROCM_PATH/bin/amdclang \
  -D CMAKE_C_COMPILER_LAUNCHER=ccache \
  -D CMAKE_CXX_FLAGS="--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13 -I$PWD/build/deps/install/include" \
  -D CMAKE_C_FLAGS="--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13 -I$PWD/build/deps/install/include" \
  -D CMAKE_PREFIX_PATH="$PWD/build/deps/install;$ROCM_PATH" \
  -D CMAKE_INSTALL_PREFIX=$PWD/hipblaslt-install \
  -D CMAKE_INSTALL_LIBDIR=lib \
  -D ROCM_PATH=$ROCM_PATH \
  -D GPU_TARGETS=gfx950 \
  -D HIPBLASLT_ENABLE_FETCH=ON \
  -D HIPBLASLT_ENABLE_ROCROLLER=OFF \
  -D HIPBLASLT_ENABLE_YAML=OFF \
  -D HIPBLASLT_ENABLE_CLIENT=ON \
  -D HIPBLASLT_BUILD_TESTING=ON \
  -D HIPBLASLT_ENABLE_SAMPLES=OFF \
  -D HIPBLASLT_ENABLE_BLIS=OFF \
  -D HIPBLASLT_ENABLE_MXDATAGENERATOR=ON \
  -D TENSILELITE_ENABLE_CLIENT=ON \
  -D BLAS_LIBRARIES=$PWD/build/deps/install/lib/libblas.a \
  "-D LAPACK_LIBRARIES=$PWD/build/deps/install/lib/liblapack.a;$PWD/build/deps/install/lib/libcblas.a;$PWD/build/deps/install/lib/libblas.a;/usr/lib/gcc/x86_64-linux-gnu/13/libgfortran.so;/usr/lib/gcc/x86_64-linux-gnu/13/libquadmath.so" \
  -D Python_EXECUTABLE=$PWD/build/python-venv/bin/python \
  -D Python3_EXECUTABLE=$PWD/build/python-venv/bin/python \
  -D CLANG_TIDY_EXE=/bin/false \
  -D RUN_CLANG_TIDY_EXE=/bin/false
```

The explicit dependency include path makes `cblas.h` visible to the client
targets. Including `libblas.a` in `LAPACK_LIBRARIES` keeps static `libcblas.a`
from leaving unresolved Fortran BLAS symbols at link time.

## Build the Test Binary and Client

Build the GoogleTest binary:

```bash
cmake --build "$BUILD_DIR" --target hipblaslt-test --parallel 16
```

Build the TensileLite benchmark client. The client is needed to generate
benchmark-derived PartialRMS `3_LibraryLogic` files:

```bash
cmake --build "$BUILD_DIR" --target tensilelite-client --parallel 16
```

Build the rocisa Python extension before running the Tensile logic generator:

```bash
cmake --build "$BUILD_DIR" --target tensilelite/rocisa/all --parallel 16
```

## Generate PartialRMS and RstdScale Library Logic

The full RMSNorm and decomposed producer tests need PartialRMS K1 logic that is
not part of the normal generic gfx950 logic path. Generate it from the row-major
PartialRMS benchmark YAML:

```bash
rm -rf /tmp/hipblaslt_partialrms_out
PYTHONPATH="$PWD/$BUILD_DIR/tensilelite/rocisa:$PWD/$BUILD_DIR/tensilelite:$PWD/tensilelite" \
LD_LIBRARY_PATH="$PWD/$BUILD_DIR/tensilelite:$PWD/$BUILD_DIR/clients/common:$PWD/$BUILD_DIR/library:$ROCM_PATH/lib:$ROCM_PATH/lib/llvm/lib:${LD_LIBRARY_PATH:-}" \
build/python-venv/bin/python tensilelite/Tensile/bin/Tensile \
  tensilelite/epilogues/yaml/gemm_partial_rms_k1_rowmajor.yaml \
  /tmp/hipblaslt_partialrms_out \
  --cxx-compiler "$PWD/build/toolchain/amdclang++" \
  --gpu-targets gfx950 \
  --prebuilt-client "$PWD/$BUILD_DIR/tensilelite/client/tensilelite-client" \
  --global-parameters LibraryFormat='"msgpack"'
```

This should produce:

```text
/tmp/hipblaslt_partialrms_out/3_LibraryLogic/
  partialrms_k1_Cijk_Alik_Bljk_BBS_BH_PRMS_UserArgs.yaml
  partialrms_k1_Cijk_Alik_Bljk_BBS_BH_PRMS_RA_UserArgs.yaml
```

The decomposed consumer E2E test also needs Kernel 3 RstdScale logic:

```bash
rm -rf /tmp/hipblaslt_rstdscale_out
PYTHONPATH="$PWD/$BUILD_DIR/tensilelite/rocisa:$PWD/$BUILD_DIR/tensilelite:$PWD/tensilelite" \
LD_LIBRARY_PATH="$PWD/$BUILD_DIR/tensilelite:$PWD/$BUILD_DIR/clients/common:$PWD/$BUILD_DIR/library:$ROCM_PATH/lib:$ROCM_PATH/lib/llvm/lib:${LD_LIBRARY_PATH:-}" \
build/python-venv/bin/python tensilelite/Tensile/bin/Tensile \
  tensilelite/epilogues/yaml/gemm_rstdscale_k3.yaml \
  /tmp/hipblaslt_rstdscale_out \
  --cxx-compiler "$PWD/build/toolchain/amdclang++" \
  --gpu-targets gfx950 \
  --prebuilt-client "$PWD/$BUILD_DIR/tensilelite/client/tensilelite-client" \
  --global-parameters LibraryFormat='"msgpack"'
```

This should produce:

```text
/tmp/hipblaslt_rstdscale_out/3_LibraryLogic/
  rstdscale_k3_Cijk_Alik_Bljk_BBS_BH_Rstd_UserArgs.yaml
```

Merge the generated logic files into one directory for the device-library build:

```bash
rm -rf /tmp/hipblaslt_fused_epilogue_logic
mkdir -p /tmp/hipblaslt_fused_epilogue_logic
cp /tmp/hipblaslt_partialrms_out/3_LibraryLogic/*.yaml /tmp/hipblaslt_fused_epilogue_logic/
cp /tmp/hipblaslt_rstdscale_out/3_LibraryLogic/*.yaml /tmp/hipblaslt_fused_epilogue_logic/
```

If the generated files contain `Device 74a1` in the device list, remove it. The
gfx950 chip-ID validator rejects `74a1` because it is not a gfx950 device ID.
One way to strip it from the merged logic directory is:

```bash
python3 - <<'PY'
from pathlib import Path
for p in Path('/tmp/hipblaslt_fused_epilogue_logic').glob('*.yaml'):
    text = p.read_text()
    text = text.replace('Device 74a1, ', '').replace('Device 74a1', '')
    p.write_text(text)
PY
```

## Build Device Libraries

Build the PartialRMS K1 and RstdScale K3 libraries from the merged generated
logic:

```bash
rm -f "$BUILD_DIR/device-library/tensilelite-device-libraries.stamp"
rm -f "$BUILD_DIR/device-library/tensilelite-device-libraries-TensileLogic.stamp"

cmake -S . -B "$BUILD_DIR" \
  -D HIPBLASLT_LIBLOGIC_PATH=/tmp/hipblaslt_fused_epilogue_logic \
  -D CMAKE_CXX_FLAGS="--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13 -I$PWD/build/deps/install/include" \
  -D CMAKE_C_FLAGS="--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13 -I$PWD/build/deps/install/include" \
  -D BLAS_LIBRARIES=$PWD/build/deps/install/lib/libblas.a \
  "-D LAPACK_LIBRARIES=$PWD/build/deps/install/lib/liblapack.a;$PWD/build/deps/install/lib/libcblas.a;$PWD/build/deps/install/lib/libblas.a;/usr/lib/gcc/x86_64-linux-gnu/13/libgfortran.so;/usr/lib/gcc/x86_64-linux-gnu/13/libquadmath.so" \
  -D HIPBLASLT_ENABLE_MXDATAGENERATOR=ON \
  -D HIPBLASLT_ENABLE_ROCROLLER=OFF \
  -D HIPBLASLT_ENABLE_YAML=OFF \
  -D Python_EXECUTABLE=$PWD/build/python-venv/bin/python \
  -D Python3_EXECUTABLE=$PWD/build/python-venv/bin/python \
  -D CLANG_TIDY_EXE=/bin/false \
  -D RUN_CLANG_TIDY_EXE=/bin/false

cmake --build "$BUILD_DIR" --target tensilelite-device-libraries --parallel 16
```

Build and install the row-major Kernel 2 code objects:

```bash
cmake --build "$BUILD_DIR" --target row_div-library-gfx950 row_rstd-library-gfx950 --parallel 16
```

`row_div_gfx950.co` is the full-flow reduce-and-apply Kernel 2.
`row_rstd_gfx950.co` is the decomposed producer reduce-and-return Kernel 2 that
writes the per-row rstd handoff consumed by Kernel 3.

Kernel 1 (PartialRMS) and Kernel 3 (RstdScale) are not separate `row_*` code
objects. They are TensileLite GEMM solutions generated from
`gemm_partial_rms_k1_rowmajor.yaml` and `gemm_rstdscale_k3.yaml`, then packaged
into the generated `TensileLibrary_*_gfx950.co` /
`TensileLibrary_lazy_gfx950.dat.zlib` library artifacts by the
`tensilelite-device-libraries` target above.

The runtime should now have these artifacts:

```text
${BUILD_DIR}/Tensile/library/gfx950/
  TensileLibrary_lazy_gfx950.dat.zlib
  TensileLibrary_*_gfx950.co      # contains K1 PartialRMS and K3 RstdScale GEMM kernels
  TensileLiteLibrary_lazy_gfx950_Mapping.dat.zlib
  row_div_gfx950.co
  row_rstd_gfx950.co
  ...
```

## Run the Focused Tests

Run the filtered gtest from the build directory:

```bash
cd "$BUILD_DIR"
LD_LIBRARY_PATH="$PWD/tensilelite:$PWD/clients/common:$PWD/library:$ROCM_PATH/lib:$ROCM_PATH/lib/llvm/lib:${LD_LIBRARY_PATH:-}" \
./clients/hipblaslt-test --gtest_filter='FusedEpilogue*.*'
```

Expected result:

```text
[==========] 46 tests from 4 test suites ran.
[  PASSED  ] 46 tests.
```

## Troubleshooting

- **`Could not find standard C++ header 'cmath'`**: use the compiler wrapper
  above so HIP/C++ compilation receives `--gcc-install-dir`.
- **`cblas.h` not found**: add `-I$PWD/build/deps/install/include` to
  `CMAKE_C_FLAGS` and `CMAKE_CXX_FLAGS`, or make sure the local dependency
  prefix is visible to the client targets.
- **Undefined `sgemm_`/`dgemm_`/`cgemm_`/`zgemm_` while linking
  `hipblaslt-test`**: include `$PWD/build/deps/install/lib/libblas.a` in the
  `LAPACK_LIBRARIES` list after `libcblas.a`.
- **`SupportedISA == 0` for gfx950**: ensure the wrapper suppresses
  `-Wgcc-install-dir-libstdcxx` for rocisa assembler probes.
- **`no PartialRMS solution selected`**: the gfx950 library is missing the
  `PRMS`/`PRMS_RA` logic. Regenerate `3_LibraryLogic` and rebuild
  `tensilelite-device-libraries` with `HIPBLASLT_LIBLOGIC_PATH` pointing to it.
- **`no RstdScale (K3) solution selected`**: the gfx950 library is missing the
  `RstdScale` logic from `gemm_rstdscale_k3.yaml`. Regenerate the K3 logic,
  merge it with the PartialRMS logic, and rebuild `tensilelite-device-libraries`.
- **`getKernel failed: row_div`**: build `row_div-library-gfx950` and make sure
  `row_div_gfx950.co` is present under `$BUILD_DIR/Tensile/library/gfx950`.
- **`getKernel failed: row_rstd`**: build `row_rstd-library-gfx950` and make
  sure `row_rstd_gfx950.co` is present under `$BUILD_DIR/Tensile/library/gfx950`.
- **Residual and non-residual tests interfere with each other**: ensure
  `ContractionProblemGemm` comparison and hashing include the PartialRMS
  discriminator fields (`usePartialRMS`, `partialRMSResidualAdd`,
  `partialRMSMT0`, `partialRMSMT1`, and `useRstdScale`) so solution-cache keys
  do not alias.
