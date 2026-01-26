# AOCL Performance Work Log

Branch: `users/todavis/aocl-build-settings`

## Objective
Investigate and resolve performance issues with AOCL BLAS library in rocBLAS clients (rocblas-bench, rocblas-test).

## System Configuration
- **GPU:** 8x AMD Instinct MI300X (Aqua Vanjaram)
- **CPU:** 24 cores, AMD Zen architecture
- **ROCm:** 7.3.0
- **OS:** Linux 6.8.0-31-generic (Ubuntu 24.04)

## Issues Identified

### Issue #1: AOCL Default Build Type is Debug
**Date:** 2026-01-26  
**Status:** Identified

**Problem:**
- AOCL 5.2 CMakeLists.txt defaults to Debug build type (lines 14-23)
- This causes severe performance degradation in CPU-based reference BLAS
- Current `install.sh` passes `-DCMAKE_BUILD_TYPE=Release` but AOCL uses aggressive `FORCE` cache logic that may override it

**Evidence:**
```cmake
# AOCL CMakeLists.txt lines 20-23
if(NOT DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type" FORCE)
elseif(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type" FORCE)
endif()
```

**Current install.sh AOCL build (line 316):**
```bash
${cmake_executable} -S . -B build -DCMAKE_BUILD_TYPE=Release ...
```

**Impact:** Suspected major performance degradation in benchmarks and tests

**Proposed Fix:**
1. Verify current build is actually using Debug
2. Add explicit `-DCMAKE_CONFIGURATION_TYPES=Release` to AOCL build ✅ **APPLIED**
3. Fix case mismatch: `--config release` → `--config Release` ✅ **APPLIED**

**Changes Made (install.sh line 316-317):**
```bash
# Before:
${cmake_executable} -S . -B build -DCMAKE_BUILD_TYPE=Release ... -DCMAKE_INSTALL_PREFIX=$PWD/install_package
elevate_if_not_root ${cmake_executable} --build build --config release -j --target install

# After:
${cmake_executable} -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CONFIGURATION_TYPES=Release ... -DCMAKE_INSTALL_PREFIX=$PWD/install_package
elevate_if_not_root ${cmake_executable} --build build --config Release -j --target install
```

---

### Issue #2: Thread Oversubscription (Known Issue)
**Date:** 2026-01-26  
**Status:** Documented in Programmer's Guide

**Problem:**
- AOCL-BLAS has known thread oversubscription issue
- When OMP_NUM_THREADS equals or exceeds available CPU cores, severe performance degradation or hangs can occur
- Documented in `docs/how-to/Programmers_Guide.rst` lines 1002-1008

**Recommended Setting:**
```bash
export OMP_NUM_THREADS=20  # For 24-core system, use ~4 fewer than total
```

**Current Status:** Not automatically set by install.sh or client executables

---

### Issue #3: Missing Architecture-Specific Optimizations?
**Date:** 2026-01-26  
**Status:** Under investigation

**Questions:**
- Is AOCL building with `amdzen` architecture target? (Default should be "amdzen")
- Is OpenMP properly linked? (Currently using `-DOpenMP_libomp_LIBRARY=""`)
- Should we use standalone BLIS instead of AOCL unified library?

**Standalone BLIS Recommendation (from authors):**
```bash
git clone https://github.com/amd/blis
cd blis
./configure --enable-cblas -b 64 -t openmp amdzen
make -j
make install prefix=<path>
```

**Differences from current AOCL build:**
1. Explicit `amdzen` target
2. Explicit `-t openmp` threading
3. Simpler build system (configure script vs CMake)

---

## Work Plan

### Phase 1: Verify Current Build Configuration
- [ ] Build rocBLAS clients with current install.sh
- [ ] Check actual CMAKE_BUILD_TYPE used in AOCL build
- [ ] Run baseline benchmark with current configuration
- [ ] Document performance metrics

### Phase 2: Fix Debug Build Issue
- [x] Modify `install.sh` to ensure Release build for AOCL
- [ ] Add verification that Release is actually used
- [ ] Rebuild and benchmark
- [ ] Compare performance vs baseline

### Phase 3: Optimize Threading
- [ ] Add OMP_NUM_THREADS configuration to clients or install.sh
- [ ] Test various thread counts (16, 20, 22)
- [ ] Document optimal settings

### Phase 4: Evaluate Architecture Optimizations
- [ ] Verify AOCL is using `amdzen` target
- [ ] Test standalone BLIS build
- [ ] Compare performance: AOCL vs standalone BLIS
- [ ] Decision: Keep AOCL or switch to BLIS

---

## Build Commands

### Current Standard Build
```bash
./install.sh -dc
```

### Recommended Build for Testing
```bash
./install.sh -d -c --cmake_install --architecture auto
```
- `-d` = Build dependencies (AOCL, googletest)
- `-c` = Build clients (rocblas-bench, rocblas-test)
- `--cmake_install` = Force CMake 3.26.0 installation
- `--architecture auto` = Build only for detected GPU (MI300X/gfx942)

### Debug Existing AOCL Build
```bash
# Check if AOCL was built in Debug mode
grep CMAKE_BUILD_TYPE build/release/deps/aocl/build/CMakeCache.txt

# Check optimization flags used
grep "CMAKE_C_FLAGS\|CMAKE_CXX_FLAGS" build/release/deps/aocl/build/CMakeCache.txt
```

---

## Benchmarking Plan

### Test Cases
1. Small GEMM: M=N=K=128
2. Medium GEMM: M=N=K=1024
3. Large GEMM: M=N=K=4096

### Metrics to Capture
- Execution time
- GFLOPS
- Thread utilization
- Memory usage

---

## Session Log

### 2026-01-26 - Session 1: Baseline Testing

**Action:** Building baseline configuration (WITHOUT fixes) to establish performance baseline
- Reverted install.sh to original (Debug build issue still present)
- Building with: `./install.sh -d -c --cmake_install --architecture auto`

**Test Command:**
```bash
./rocblas-bench -f trsm -r f32_r --side L --uplo U --transposeA N --diag N \
  -m 1280 -n 1280 --alpha 1 --lda 1280 --ldb 1280 -v
```

**Test Details:**
- Function: TRSM (Triangular Solve with Multiple right-hand sides)
- Precision: Single precision (f32)
- Matrix size: 1280 x 1280
- Side: Left
- Triangle: Upper
- Transpose: None
- Diagonal: Non-unit

---

## Next Steps
1. ~~Build clients with explicit Release mode for AOCL~~ → Build baseline FIRST
2. Run baseline benchmarks
3. Fix issues and re-benchmark
4. Compare performance: baseline vs fixed
5. Document findings and recommendations
