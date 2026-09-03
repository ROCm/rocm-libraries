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
Kernel 3 / RstdScale via ScaleAlphaVec).

## Prerequisites

Use a ROCm SDK that contains `amdclang++`, HIP, amd-smi, hipBLAS common, and the
gfx950 runtime libraries. Set:

```bash
export ROCM_PATH=/opt/rocm
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
  exec /opt/rocm/bin/amdclang++ -Wno-gcc-install-dir-libstdcxx "$@"
fi
exec ccache /opt/rocm/bin/amdclang++ \
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

## Build the Test Binary

Build the GoogleTest binary. This also builds the `hipblaslt` library and the
`hipblaslt-test-data` target that produces `clients/hipblaslt_gtest.data`:

```bash
cmake --build "$BUILD_DIR" --target hipblaslt-test --parallel 16
```

The rocisa Python extension is needed by the device-library step below, and the
`tensilelite-device-libraries` target builds it as a dependency. To build it on
its own:

```bash
cmake --build "$BUILD_DIR" --target tensilelite/rocisa/all --parallel 16
```

The TensileLite benchmark client (`tensilelite-client`) is only required if you
are re-tuning solutions and regenerating `3_LibraryLogic` from benchmark YAML.
The workflow below uses the committed logic, so it is not needed.

## Select the Library Logic

The PartialRMS (K1) and RstdScale (K3) `3_LibraryLogic` YAML files are committed
to the source tree, so no benchmark run is needed to produce them:

```text
library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx950/gfx950/
  Equality/partialrms_k1_*.yaml               # bf16 K1 producer (PRMS, PRMS_RA)
  Equality/partialrms_mxfp8_quant_k1_*.yaml   # K1 producer with MX fp8 quant
  Equality/mxfp8_quant_k1_*.yaml              # standalone MX fp8 quant
  Equality/mxfp8_rstdscale_k3_*.yaml          # K3 consumer via ScaleAlphaVec
  Origami/partialrms_*.yaml
  partialrms_residualout_*.yaml
```

Earlier revisions of this guide generated these from
`tensilelite/epilogues/yaml/gemm_partial_rms_k1_rowmajor.yaml` and
`gemm_rstdscale_k3.yaml`. Those files no longer exist and those steps are obsolete.

Building every gfx950 logic file compiles roughly 71,000 assembly kernels and
takes hours. For fused-epilogue work, restrict the build with
`HIPBLASLT_LIBLOGIC_PATH` to the logic the tests actually need, which cuts it to
about 8,500 kernels (~7 minutes on 128 cores). Copy that logic into a scratch
directory, preserving the original relative path:

```bash
SRC=library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx950/gfx950
OUT=/tmp/hipblaslt_fused_epilogue_logic
REL=src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx950/gfx950

rm -rf $OUT
mkdir -p $OUT/$REL/Equality $OUT/$REL/Origami
cp $SRC/Equality/*.yaml          $OUT/$REL/Equality/
cp $SRC/*.yaml                   $OUT/$REL/
cp $SRC/Origami/partialrms*.yaml $OUT/$REL/Origami/
```

Two details matter here:

- **Keep the relative path.** `TensileLogic --check-all` validates the logic
  against `tensilelite/Tensile/TensileLogic/known_bugs.yaml`, whose skip entries
  are keyed on paths relative to the logic root. A flattened directory makes those
  entries stop matching, and the build fails on four pre-existing bad solutions
  with `Validation failed: MatrixInstruction.py:329 MIInputPerThread ...`.
- **Copy all of `Equality/`, not just `partialrms_*`.** The decomposed consumer
  (K3) and the reference GEMMs need the standard ScaleAlphaVec solutions.

As an alternative to copying, `TENSILELITE_LOGIC_FILTER` passes a glob straight
through to `TensileCreateLibrary --logic-filter`, which is matched as
`<logic-path>/**/<filter>.yaml`, for example `gfx950/Equality/*`.

## Build Device Libraries

Build the PartialRMS K1 and RstdScale K3 libraries from the selected logic. If
`$BUILD_DIR` is already configured, pointing it at the scratch logic directory is
the only change needed:

```bash
rm -f "$BUILD_DIR/device-library/tensilelite-device-libraries.stamp"
rm -f "$BUILD_DIR/device-library/tensilelite-device-libraries-TensileLogic.stamp"

cmake -S . -B "$BUILD_DIR" -D HIPBLASLT_LIBLOGIC_PATH=/tmp/hipblaslt_fused_epilogue_logic
cmake --build "$BUILD_DIR" --target tensilelite-device-libraries --parallel 16
```

For a build directory configured from scratch, pass the full option set instead:

```bash
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

Build and install the row-major Kernel 2 code objects. There are three of them:

```bash
cmake --build "$BUILD_DIR" \
  --target row_div-library-gfx950 row_rstd-library-gfx950 row_div_quant-library-gfx950 \
  --parallel 16
```

`row_div_gfx950.co` is the full-flow reduce-and-apply Kernel 2.
`row_rstd_gfx950.co` is the decomposed producer reduce-and-return Kernel 2 that
writes the per-row rstd handoff consumed by Kernel 3.
`row_div_quant_gfx950.co` is the full-flow Kernel 2 that reduces, applies rstd,
and quantizes to FP8; `fullRmsNormResidualAddRequantMatchesReference` fails with
`getKernel failed: row_div_quant` without it.

Kernel 1 (PartialRMS) and Kernel 3 (RstdScale via ScaleAlphaVec) are not
separate `row_*` code objects. They are TensileLite GEMM solutions defined by the
committed `partialrms_*` and `mxfp8_rstdscale_k3_*` logic, packaged into the
generated `TensileLibrary_*_gfx950.co` / `TensileLibrary_lazy_gfx950.dat.zlib`
artifacts by the `tensilelite-device-libraries` target above.

The runtime should now have these artifacts:

```text
${BUILD_DIR}/Tensile/library/gfx950/
  TensileLibrary_lazy_gfx950.dat.zlib
  TensileLibrary_*_gfx950.co      # contains K1 PartialRMS and K3 RstdScale GEMM kernels
  TensileLiteLibrary_lazy_gfx950_Mapping.dat.zlib
  row_div_gfx950.co
  row_rstd_gfx950.co
  row_div_quant_gfx950.co
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
[==========] 67 tests from 4 test suites ran.
[  PASSED  ] 67 tests.
```

Every RMSNorm flow — full, residual-add, requant, and the decomposed
producer/consumer — passes, as does the **standalone** MX fp8 quant path (a
REQUANT-only chain with no RMSNorm stage). If the four `mxfp8Quant*` tests fail on
the UE8M0 scale buffer while `D` still matches, see the scale-grid orientation
entry under troubleshooting.

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
  ScaleAlphaVec logic from `gemm_rstdscale_k3.yaml`. Regenerate the K3 logic,
  merge it with the PartialRMS logic, and rebuild `tensilelite-device-libraries`.
- **`getKernel failed: row_div`**: build `row_div-library-gfx950` and make sure
  `row_div_gfx950.co` is present under `$BUILD_DIR/Tensile/library/gfx950`.
- **`getKernel failed: row_rstd`**: build `row_rstd-library-gfx950` and make
  sure `row_rstd_gfx950.co` is present under `$BUILD_DIR/Tensile/library/gfx950`.
- **`getKernel failed: row_div_quant`**: build `row_div_quant-library-gfx950`.
  This target is easy to miss because the full RMSNorm flow only needs it once
  the chain also requests static per-tensor FP8 requant.
- **Every E2E test fails instantly with `algoCount == 0`**: the device libraries
  in `$BUILD_DIR/Tensile/library/gfx950` are stale or were built from a logic set
  that predates the fused-epilogue kernels. Check their timestamps and rebuild
  `tensilelite-device-libraries`. A whole-suite runtime of well under a second is
  the giveaway that nothing reached the GPU.
- **`Validation failed: MatrixInstruction.py:329 MIInputPerThread ...` when using
  `HIPBLASLT_LIBLOGIC_PATH`**: the scratch logic directory does not reproduce the
  original relative paths, so the `known_bugs.yaml` skip entries no longer match.
  Recreate it under
  `src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx950/gfx950/...`.
- **`Cannot open .../hipblaslt_gtest.data`**: reconfiguring the build directory
  can drop the generated gtest data file. Rebuild the `hipblaslt-test` target
  (which depends on `hipblaslt-test-data`) to regenerate it.
- **`no standalone MX fp8 quant solution selected`**: this was a host-side defect
  in `tensile_host.cpp`, now fixed. For a REQUANT-only chain,
  `ConstructTensileProblem` and `updateTensileProblem` unconditionally set
  `DQuantSize0 = requantMxBlockSize` and `DQuantSize1 = 1`. That pairing is only
  correct for the PartialRMS flow, where the problem has already been transposed
  so free0 is the hidden dimension. The standalone `mxfp8_quant_k1_*` solutions
  are tuned with `DQuantSize0 = 1` and `DQuantSize1 = blockSize`, so the
  `DQuantSize0Equal` / `DQuantSize1Equal` predicates rejected every candidate.
  Both sites now pick the pair based on whether the PartialRMS transpose is active.
  The `setMxScale` arguments are deliberately left in terms of `q0`/`q1`, since that
  expression already matches Tensile's convention for either orientation.
- **`MX UE8M0 scale buffer mismatch` (`FusedEpilogueE2E.mxfp8Quant*`)**: this was a
  scale-grid orientation disagreement, now fixed on the test side. The numerics were
  never wrong — the fp8 `D` output matched exactly, so the blocking and every
  block's scale value were right.

  Tensile places the scale grid's **rows on free1 and its columns on free0**,
  regardless of which axis carries the block. Both halves of Tensile encode this:
  `SubtileMXFP8QuantEmitter._tileByteOffset`
  (`tensilelite/Tensile/Components/Subtile/SubtileDynamicQuant.py`) takes the
  swizzle row from `WorkGroup1` and the column from `WorkGroup0`, and the client's
  CPU model (`tensilelite/client/src/Reference.cpp`, "Scale grid: rows = free dim
  (M_tokens, nTiles), cols = kblock dim (mTiles)") pads rows from the free1 tile
  count and writes at `mxScaleSwizzleOffset(tj, ti, colBlocks)`. `tensilelite-client`
  validates the MXScale bytes exactly and passes for the standalone `Q=[1,32]`
  kernels at `(128,512,1,64)` — the same shape the gtest uses — so the emitter's
  layout is the validated one. `quantizeMxfp8Standard` and the four `mxfp8Quant*`
  bodies now follow it.

  Be aware of the consequence: the two paths emit transposed grids for the same
  logical operation. After the PartialRMS transpose tokens sit on free1, giving
  `[M_tokens × N_hidden/blockSize]`; a REQUANT-only chain leaves tokens on free0,
  giving `[N_hidden/blockSize × M_tokens]`. Both are self-consistent and validated,
  but a caller that feeds these scales downstream must orient them per path. Making
  the two agree would require re-tuning non-PartialRMS solutions for the transposed
  orientation.
- **Residual and non-residual tests interfere with each other**: ensure
  `ContractionProblemGemm` comparison and hashing include the PartialRMS
  discriminator fields (`usePartialRMS`, `partialRMSResidualAdd`,
  `partialRMSMT0`, `partialRMSMT1`) so solution-cache keys do not alias.
- **Benchmark aborts with `Failed to load solution library`**: the prebuilt
  TensileLite client cannot load msgpack libraries if it was compiled with
  `HIPBLASLT_ENABLE_YAML=ON`. Drop `--global-parameters LibraryFormat='"msgpack"'`
  from the Tensile invocation or rebuild the client with `HIPBLASLT_ENABLE_YAML=OFF`.
