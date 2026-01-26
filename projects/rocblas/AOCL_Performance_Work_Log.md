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
**Status:** ⚠️ CONFIRMED - Partially affects current build

**Problem:**
- AOCL 5.2 CMakeLists.txt defaults **both** `CMAKE_BUILD_TYPE` and `CMAKE_CONFIGURATION_TYPES` to Debug
- This causes severe performance degradation in CPU-based reference BLAS
- Current `install.sh` passes `-DCMAKE_BUILD_TYPE=Release` ✅ but is missing `-DCMAKE_CONFIGURATION_TYPES=Release` ❌
- GitHub Issue: [amd/aocl#6](https://github.com/amd/aocl/issues/6) reports 100-450x slowdowns on trsm/trmm

**Evidence from AOCL CMakeLists.txt (lines 13-23):**
```cmake
# Sets both variables to Debug by default!
if(NOT DEFINED CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_CONFIGURATION_TYPES "Debug" CACHE STRING "Build configurations" FORCE)
endif()

if(NOT DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type" FORCE)
elseif(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type" FORCE)
endif()
```

**Current Build Status (Verified from CMakeCache.txt):**
```bash
CMAKE_BUILD_TYPE:STRING=Release                   ✅ Working correctly
CMAKE_CONFIGURATION_TYPES:STRING=Debug            ❌ Still Debug!
CMAKE_C_FLAGS_RELEASE:STRING=-O3 -DNDEBUG        ✅ Correct flags
```

**Current install.sh AOCL build (line 316-317):**
```bash
${cmake_executable} -S . -B build -DCMAKE_BUILD_TYPE=Release ... -DCMAKE_INSTALL_PREFIX=$PWD/install_package
elevate_if_not_root ${cmake_executable} --build build --config release -j --target install
```

**Impact:** 
- Unknown performance impact from `CMAKE_CONFIGURATION_TYPES=Debug`
- May be limiting performance even with threading fix applied
- GitHub issue reports 100-450x degradation on triangular operations

**Required Fix:**
```bash
# Line 316: Add -DCMAKE_CONFIGURATION_TYPES=Release
${cmake_executable} -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CONFIGURATION_TYPES=Release ... -DCMAKE_INSTALL_PREFIX=$PWD/install_package

# Line 317: Fix case mismatch (release → Release)
elevate_if_not_root ${cmake_executable} --build build --config Release -j --target install
```

**Status:** NOT YET APPLIED (waiting to establish baseline with threading fix first)

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

### Phase 1: Verify Current Build Configuration ✅ COMPLETE
- [x] Build rocBLAS clients with current install.sh
- [x] Check actual CMAKE_BUILD_TYPE used in AOCL build
- [x] Run baseline benchmark with current configuration
- [x] Document performance metrics

**Findings:**
- `CMAKE_BUILD_TYPE=Release` ✅ working
- `CMAKE_CONFIGURATION_TYPES=Debug` ❌ problematic
- Baseline CPU performance: 11.37 GFLOPS (terrible)
- Thread oversubscription identified as primary issue

### Phase 2: Fix Threading Issue ✅ COMPLETE
- [x] Identified OMP_NUM_THREADS not set (defaults to 24 cores)
- [x] Tested with `OMP_NUM_THREADS=20`
- [x] Documented 62x speedup (11.37 → 706.11 GFLOPS)
- [ ] Apply permanent fix to install.sh or runtime environment

### Phase 3: Fix Debug Build Issue 🔄 IN PROGRESS
- [ ] Modify `install.sh` line 316 to add `-DCMAKE_CONFIGURATION_TYPES=Release`
- [ ] Fix case mismatch on line 317: `--config release` → `--config Release`
- [ ] Clean AOCL build directory
- [ ] Rebuild AOCL with correct configuration
- [ ] Re-benchmark with full fix (threading + Release)
- [ ] Compare: baseline (11.37) → threading fix (706) → full fix (??)

### Phase 4: Evaluate Architecture Optimizations (DEFERRED)
- [ ] Verify AOCL is using `amdzen` target
- [ ] Test standalone BLIS build
- [ ] Compare performance: AOCL vs standalone BLIS
- [ ] Decision: Keep AOCL or switch to BLIS

**Note:** Deferring Phase 4 until Phase 3 complete to isolate variables

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

### 2026-01-26 - Session 1: Baseline Testing & Threading Fix

**Build Configuration:**
- Command: `./install.sh -d -c --cmake_install --architecture auto`
- Build time: 16.9 minutes
- System: Single AMD Instinct MI300X (gfx942), 24 CPU cores

**AOCL Build Verification:**
```bash
# Verified from build/deps/aocl/build/CMakeCache.txt
CMAKE_BUILD_TYPE:STRING=Release                    ✅ Correct
CMAKE_C_FLAGS_RELEASE:STRING=-O3 -DNDEBUG         ✅ Correct
CMAKE_CXX_FLAGS_RELEASE:STRING=-O3 -DNDEBUG       ✅ Correct
CMAKE_CONFIGURATION_TYPES:STRING=Debug            ❌ PROBLEM!

# BLIS sub-component
CMAKE_BUILD_TYPE:STRING=Release                    ✅
CMAKE_CONFIGURATION_TYPES:UNINITIALIZED=Debug     ❌ PROBLEM!
```

**Key Finding:** Issue #1 (Debug build) IS present via `CMAKE_CONFIGURATION_TYPES=Debug`, despite `CMAKE_BUILD_TYPE=Release`. This matches GitHub issue [amd/aocl#6](https://github.com/amd/aocl/issues/6).

---

#### Baseline Test Results (No Threading Optimization)

**Test Command:**
```bash
./build/release/clients/staging/rocblas-bench -f trsm -r f32_r \
  --side L --uplo U --transposeA N --diag N \
  -m 1280 -n 1280 --alpha 1 --lda 1280 --ldb 1280 -v 1
```

**Results:**
| Metric | Value | Notes |
|--------|-------|-------|
| GPU Performance | 8,677 GFLOPS @ 241 μs | Normal |
| **CPU Performance** | **11.37 GFLOPS @ 184,461 μs** | **❌ TERRIBLE** |
| Initialization Time | 7,628 ms | Slow (>5s warning) |
| OMP_NUM_THREADS | Not set (defaults to 24) | All cores used |

**Problem Identified:** CPU reference BLAS (AOCL) is **762x slower** than GPU (184ms vs 0.24ms).

---

#### Test with OMP_NUM_THREADS=20

**Test Command:**
```bash
export OMP_NUM_THREADS=20
./build/release/clients/staging/rocblas-bench -f trsm -r f32_r \
  --side L --uplo U --transposeA N --diag N \
  -m 1280 -n 1280 --alpha 1 --lda 1280 --ldb 1280 -v 1
```

**Results:**
| Metric | Baseline (no OMP) | With OMP_NUM_THREADS=20 | Improvement |
|--------|-------------------|-------------------------|-------------|
| **CPU GFLOPS** | 11.37 | **706.11** | **62x faster** |
| **CPU Time (μs)** | 184,461 | **2,970** | **62x faster** |
| Initialization (ms) | 7,628 | 5,272 | 31% faster |

**Root Cause Confirmed:** Issue #2 (Thread Oversubscription) - Using all 24 cores causes severe AOCL performance degradation. Setting `OMP_NUM_THREADS=20` (4 cores reserved) gives **62x speedup**.

---

#### Analysis

**Issue #2 (Thread Oversubscription) - CONFIRMED & PARTIALLY FIXED**
- ✅ Documented in Programmer's Guide (lines 1002-1008)
- ✅ Root cause verified: OMP_NUM_THREADS unset → uses all 24 cores → severe degradation
- ✅ Workaround effective: `OMP_NUM_THREADS=20` → 62x speedup (11.37 → 706.11 GFLOPS)
- ⚠️ Still needs permanent solution (set in install.sh or runtime)

**Issue #1 (Debug Build) - CONFIRMED BUT NOT YET FIXED**
- ❌ `CMAKE_CONFIGURATION_TYPES=Debug` found in AOCL build
- ⚠️ According to [GitHub issue #6](https://github.com/amd/aocl/issues/6): causes 100-450x slowdowns on trsm/trmm
- ❓ Question: Is 706 GFLOPS still limited by Debug build? Should be much higher with true Release build
- 🔧 Fix needed: Add `-DCMAKE_CONFIGURATION_TYPES=Release` to install.sh line 316

**Combined Impact Hypothesis:**
- Baseline (Debug + no threading fix): 11.37 GFLOPS
- With threading fix only: 706.11 GFLOPS
- With both fixes (threading + Release): Expected >>706 GFLOPS?

---

---

### 2026-01-26 - Session 2: Threading Fix Implementation

**Problem Diagnosed:**
- Thread oversubscription is the PRIMARY issue, not Debug builds
- AOCL was built with `-O3 -DNDEBUG` (verified via objdump and flags.make) ✅
- `CMAKE_CONFIGURATION_TYPES=Debug` was set but IGNORED (Unix Makefiles only uses `CMAKE_BUILD_TYPE`)
- Real culprit: `OMP_NUM_THREADS` defaulting to 24 cores triggers AOCL bug

**Solution Implemented:**
Modified `clients/common/client_omp.cpp` to automatically protect against thread oversubscription:

```cpp
void client_omp_manager::limit_by_processor_count()
{
    const char* env_omp_threads = std::getenv("OMP_NUM_THREADS");
    
    if(env_omp_threads == nullptr)
    {
        // User did not set OMP_NUM_THREADS - apply AOCL workaround
        const int omp_default_threads = omp_get_max_threads();
        int safe_thread_count = omp_default_threads - 4;  // Leave 4 cores free
        omp_set_num_threads(safe_thread_count);
    }
    // else: User explicitly set OMP_NUM_THREADS - respect their choice
}
```

**Changes:**
- `clients/common/client_omp.cpp`: Changed `c_thread_reducer` from 2 to 4
- Added logic to check if OMP_NUM_THREADS is explicitly set
- If not set: automatically reduce threads by 4 (e.g., 24 → 20)
- If set: respect user's choice (gives them full control)

**Test Results:**

| Scenario | OMP_NUM_THREADS | Final Threads | CPU GFLOPS | Notes |
|----------|-----------------|---------------|------------|-------|
| Baseline (broken) | Not set (defaulted to 24) | 24 | 11.37 | ❌ AOCL bug |
| **After Fix** | **Not set** | **20** | **729.19** | ✅ **64x improvement!** |
| User override | Set to 24 | 24 | 11.18 | User's choice (still hits bug) |
| User override | Set to 16 | 16 | 619.36 | User's choice (works) |

**Key Insight:** 
The fancy approach of detecting system allocation even when OMP_NUM_THREADS is set cannot work because OpenMP initializes before we can query it. The simple approach (protect when not set, respect when set) is the right balance between safety and user control.

---

## Next Steps
1. ✅ ~~Build baseline and run benchmarks~~
2. ✅ ~~Identify root causes (threading, debug build)~~
3. ✅ **Implement threading fix in client code**
4. ✅ **Verify 64x performance improvement**
5. ❌ **CMAKE_CONFIGURATION_TYPES fix NOT needed** (was not affecting compilation)
6. 📝 **Document final recommendations for users**
7. 🔄 **Consider adding warning when user sets dangerous OMP_NUM_THREADS**

---

## Summary & Recommendations

### Root Cause
AOCL 5.2 BLAS has a known thread oversubscription bug where using all or nearly all CPU cores causes severe performance degradation (64x slower). This is documented in `docs/how-to/Programmers_Guide.rst` lines 1002-1008.

### Solution
**Automatic Protection in Client Code:** Modified `clients/common/client_omp.cpp` to automatically reduce OpenMP threads by 4 when `OMP_NUM_THREADS` is not explicitly set. This protects rocblas-bench and rocblas-test users from the AOCL bug by default.

### Performance Impact
- **Before:** 11.37 GFLOPS (AOCL using all 24 cores)  
- **After:** 729.19 GFLOPS (AOCL using 20 cores) 
- **Improvement:** **64x faster**

### For End Users (Production)
If you're using AOCL 5.2 in production with rocBLAS:
```bash
# Recommended: Leave 4 cores free for best AOCL performance
export OMP_NUM_THREADS=20  # On 24-core system
export OMP_NUM_THREADS=28  # On 32-core system
# General formula: (total_cores - 4)
```

### For rocBLAS CI/Testing
No action needed - the fix is now in the client code and will automatically apply when `OMP_NUM_THREADS` is not set.

### GitHub Issue Status
- [amd/aocl#6](https://github.com/amd/aocl/issues/6): `CMAKE_CONFIGURATION_TYPES=Debug` issue is real but did NOT affect this build
- The issue correctly identifies the CMake dual-variable problem for multi-config generators
- On Linux with Unix Makefiles, only `CMAKE_BUILD_TYPE` matters (which was correctly set to Release)
- AOCL was verified to be built with `-O3 -DNDEBUG -mavx2 -mfma -march=znver4` optimization flags
