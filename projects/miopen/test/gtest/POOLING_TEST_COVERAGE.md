# MIOpen Pooling Test Coverage

Complete overview of pooling tests including forward, backward, 2D, 3D, NCHW, and NHWC layouts.

---

## Test File Organization

### Standalone Driver Tests (`test/` directory)
| File | Executable | Purpose | Usage |
|------|------------|---------|-------|
| `pooling2d.cpp` + `pooling2d.hpp` | `MIOpenDriver pool` | Standalone 2D pooling driver | Manual command-line testing |
| `pooling3d.cpp` + `pooling3d.hpp` | `MIOpenDriver pool` | Standalone 3D pooling driver | Manual command-line testing |
| `pooling_common.hpp` | N/A | Shared pooling test infrastructure | CPU reference, verification |

**Run:**
```bash
# Direct driver invocation
./bin/MIOpenDriver pool --input 1,3,32,32 --pool_stride 2,2 --pool_size 2,2

# Test executable (uses pooling2d.hpp)
./bin/test_pooling2d --float --all
./bin/test_pooling3d --float --all
```

### Unit/Applicability Tests (`test/gtest/` - Fast)
| File | Purpose | Test Type | Runtime |
|------|---------|-----------|---------|
| `pooling_backward.cpp` | Backward solver applicability, type support, edge cases | Unit | < 1s |
| `layout_transpose.cpp` | Batched transpose type support, solver selection | Unit | < 1s |

### Driver/Execution Tests (`test/gtest/` - Slow - Full GPU Execution)
| File | Direction | Layout | Dataset | Precision | Purpose |
|------|-----------|--------|---------|-----------|---------|
| `pooling2d_asymmetric.cpp` | Forward | NCHW | Minimal (1) | FP32, FP16 | Asymmetric configs |
| `pooling2d_asymmetric_nhwc.cpp` | Forward | **NHWC** | Minimal (1) | FP32, FP16 | Asymmetric + transpose |
| `pooling2d_wide.cpp` | Forward | NCHW | Wide (2) | FP32, FP16 | Large window sizes |
| `pooling2d_wide_nhwc.cpp` | Forward | **NHWC** | Wide (2) | FP32, FP16 | Wide + transpose |
| `pooling3d_ndhwc.cpp` | Forward | **NDHWC** | Standard | FP32, FP16 | 3D pooling + transpose |
| `pooling_backward_nhwc.cpp` | **Backward** | **NHWC** | Standard | FP32, FP16, **BF16** | **NEW: Backward + transpose** |

---

## Test Coverage Matrix

### 2D Pooling Forward (Existing)

| Test File | Layout | Transpose Used | Data Types | Test Configs |
|-----------|--------|----------------|------------|--------------|
| `pooling2d_asymmetric.cpp` | NCHW | None | FP32, FP16 | Asymmetric pads/strides |
| `pooling2d_asymmetric_nhwc.cpp` | NHWC | ✅ **Batched** | FP32, FP16 | Asymmetric + NHWC |
| `pooling2d_wide.cpp` | NCHW | None | FP32, FP16 | Wide windows (35x35, 100x100, 255x255, 410x400) |
| `pooling2d_wide_nhwc.cpp` | NHWC | ✅ **Batched** | FP32, FP16 | Wide + NHWC |

**Transpose Selection:** Forward pooling supports FP32/FP16, batched transpose supports FP32/FP16, so **batched transpose IS used** ✅

### 3D Pooling Forward (Existing)

| Test File | Layout | Transpose Used | Data Types | Notes |
|-----------|--------|----------------|------------|-------|
| `pooling3d_ndhwc.cpp` | NDHWC | ✅ Universal | FP32, FP16 | 5D tensors, batched only for 2D NCHW↔NHWC |

**Transpose Selection:** Batched transpose only handles 2D NCHW↔NHWC, so 3D uses universal transpose

### 2D Pooling Backward (NEW)

| Test File | Layout | Transpose Used | Data Types | Test Type |
|-----------|--------|----------------|------------|-----------|
| `pooling_backward.cpp` | NCHW | None | FP32, FP16, **BF16** | Unit (IsApplicable) |
| `pooling_backward_nhwc.cpp` | NHWC | ✅ **Batched** | FP32, FP16, **BF16** | **Driver (Output validation)** |

**Transpose Selection:** Backward supports FP32/FP16/BF16, batched supports all three, so **batched transpose IS used** ✅

---

## Data Type Support by Component

### Pooling Forward Solvers
| Solver | FP32 | FP16 | BF16 | Int8 | Int32 | Double |
|--------|------|------|------|------|-------|--------|
| `PoolingForward2d` | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| `PoolingForwardNd` | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |

### Pooling Backward Solvers (NEW CHANGES)
| Solver | FP32 | FP16 | BF16 | Int8 | Int32 | Double |
|--------|------|------|------|------|-------|--------|
| `PoolingBackward2d` | ✅ | ✅ | ✅ **NEW** | ❌ | ❌ | ❌ |
| `PoolingBackwardNd` | ✅ | ✅ | ✅ **NEW** | ❌ | ❌ | ❌ |

**NEW:** Added BF16 support + type validation to prevent compile errors

### Transpose Solvers
| Solver | FP32 | FP16 | BF16 | Int8 | Int32 | Double | Layout Support |
|--------|------|------|------|------|-------|--------|----------------|
| `BatchedNchw2NhwcTranspose` | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | NCHW→NHWC only |
| `BatchedNhwc2NchwTranspose` | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | NHWC→NCHW only |
| `UniversalTranspose` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Any→Any (fallback) |

### Transposing Wrapper Solvers
| Solver | FP32 | FP16 | BF16 | Direction | Notes |
|--------|------|------|------|-----------|-------|
| `TransposedPoolingFwd2d` | ✅ | ✅ | ❌ | Forward | Uses batched transpose for FP32/FP16 |
| `TransposedPoolingFwdNd` | ✅ | ✅ | ❌ | Forward | Uses batched transpose for FP32/FP16 |
| `TransposedPoolingBwd2d` | ✅ | ✅ | ✅ **NEW** | Backward | Uses batched transpose for FP32/FP16/BF16 |
| `TransposedPoolingBwdNd` | ✅ | ✅ | ✅ **NEW** | Backward | Uses batched transpose for FP32/FP16/BF16 |

---

## Transpose Solver Selection Logic

### How It Works

All transposing pooling solvers (forward and backward) use the same `GetTransposeSolvers()` mechanism:

**src/include/miopen/utility/transposing_solver.hpp:528-535**
```cpp
static std::vector<AnyTransposePseudoSolver> GetTransposeSolvers()
{
    return {
        BatchedNchw2NhwcTransposeSolver{}, // Tried first for NCHW→NHWC
        BatchedNhwc2NchwTransposeSolver{}, // Tried first for NHWC→NCHW
        UniversalTransposeSolver{}         // Fallback for everything else
    };
}
```

**Selection Algorithm (lines 686-717):**
1. Try exact layout pair: `"NHWC-NCHW"` → `BatchedNhwc2NchwTransposeSolver` ✅
2. Try wildcards: `"NHWC-*"`, `"*-NCHW"`
3. Fallback: `"*-*"` → `UniversalTransposeSolver`

**Each transpose solver checks IsApplicable() based on data type.**

### When Batched Transpose Is Used

| Scenario | Inner Solver Types | Batched Types | Batched Used? |
|----------|-------------------|---------------|---------------|
| **Forward NHWC (FP32)** | FP32, FP16 | FP32, FP16, BF16, Int8, Int32 | ✅ YES |
| **Forward NHWC (FP16)** | FP32, FP16 | FP32, FP16, BF16, Int8, Int32 | ✅ YES |
| **Backward NHWC (FP32)** | FP32, FP16, BF16 | FP32, FP16, BF16, Int8, Int32 | ✅ YES |
| **Backward NHWC (FP16)** | FP32, FP16, BF16 | FP32, FP16, BF16, Int8, Int32 | ✅ YES |
| **Backward NHWC (BF16)** | FP32, FP16, BF16 | FP32, FP16, BF16, Int8, Int32 | ✅ YES (NEW) |
| **3D NDHWC (any type)** | FP32, FP16 | N/A (3D not supported) | ❌ NO (universal) |
| **Any with Double** | N/A | ❌ Not supported | ❌ NO (universal) |

**Key Point:** Batched transpose is used whenever:
1. Layout conversion is NCHW↔NHWC (2D only)
2. Data type is FP32, FP16, BF16, Int8, or Int32
3. Inner pooling solver supports that data type

---

## Test Details

### 1. `pooling_backward.cpp` (Unit Tests)

**Purpose:** Fast smoke tests for solver applicability

**Test Classes:**
- `PoolingBackward2dDataTypeTest` - Parameterized over data types
- `PoolingBackwardNdDataTypeTest` - Parameterized over data types

**Test Coverage:**
```
Smoke Tests (6 tests):
  ✅ PoolingBackward2d: FP32, FP16, BF16
  ✅ PoolingBackwardNd: FP32, FP16, BF16

Full Tests (12 tests):
  ✅ PoolingBackward2d: FP32, FP16, BF16 (supported)
  ❌ PoolingBackward2d: Int8, Int32, Double (rejected)
  ✅ PoolingBackwardNd: FP32, FP16, BF16 (supported)
  ❌ PoolingBackwardNd: Int8, Int32, Double (rejected)

Edge Cases (1 test):
  - RejectsForwardDirection: Verify backward solvers reject forward problems

Integration Tests (2 tests):
  - BatchedTransposeSelectedForNHWC: Batched transpose applicable for NHWC
  - BFloat16EndToEndSupport: 3-layer BF16 validation
    1. Pooling backward supports BF16
    2. Batched transpose supports BF16
    3. Transposing wrappers work with BF16 + NHWC
```

**Run:**
```bash
./bin/test_pooling_backward --gtest_filter="Smoke*"  # < 1 second
./bin/test_pooling_backward --gtest_filter="Full*"   # < 1 second
```

**What It Tests:**
- Type checking prevents unsupported types from reaching kernel compilation
- BF16 added to backward solvers (was missing)
- Batched transpose selected for NHWC layouts

---

### 2. `pooling_backward_nhwc.cpp` (Driver Tests) **NEW**

**Purpose:** End-to-end execution tests with output validation

**Test Classes:**
- `GPU_PoolingBackward_NHWC_FP32`
- `GPU_PoolingBackward_NHWC_FP16`
- `GPU_PoolingBackward_NHWC_BF16` ⭐

**Test Coverage:**
```
Smoke Tests (3 tests):
  ✅ FP32 backward pooling with NHWC layout
  ✅ FP16 backward pooling with NHWC layout
  ✅ BF16 backward pooling with NHWC layout (CRITICAL)
```

**Command:** `test_pooling2d --forw 0 --in_layout NHWC --out_layout NHWC`
- `--forw 0`: Backward direction
- `--in_layout NHWC`: Input gradient layout
- `--out_layout NHWC`: Output gradient layout

**Run:**
```bash
./bin/test_pooling_backward_nhwc --gtest_filter="Smoke*"  # ~30-60s first run
```

**What It Actually Does:**
1. Creates NHWC input tensors (e.g., [N, H, W, C] = [1, 8, 8, 3])
2. Invokes: `test_drive<pooling2d_driver>` with backward + NHWC flags
3. Driver detects NHWC → selects `TransposedPoolingBwd2d/Nd`
4. Solver execution flow:
   ```
   Input (NHWC) → Batched Transpose (NHWC→NCHW)
                → PoolingBackward2d/Nd (NCHW)
                → Batched Transpose (NCHW→NHWC)
                → Output (NHWC)
   ```
5. Compares GPU output vs CPU reference
6. Test **PASSES** only if numerical accuracy is correct

**Critical Validation:**
- ✅ Batched transpose kernels compile and execute
- ✅ Pooling backward kernels compile and execute with BF16
- ✅ Transposing wrapper correctly chains operations
- ✅ Output matches CPU reference within tolerance

---

### 3. `layout_transpose.cpp` (Batched Transpose Tests)

**Purpose:** Pure transpose solver tests (no pooling)

**Test Classes:**
- `BatchedTransposeDataTypeTest` - Parameterized over data types
- `BatchedTransposeSolverSelection` - Solver priority verification

**Test Coverage:**
```
Smoke Tests (5 tests):
  ✅ Batched transpose: FP32, FP16, BF16, Int8, Int32

Full Tests (6 tests):
  ✅ Batched transpose: FP32, FP16, BF16, Int8, Int32 (supported)
  ❌ Batched transpose: Double (falls back to universal)

Solver Selection (1 test):
  - Verifies batched transpose preferred over universal for supported types
```

**Run:**
```bash
./bin/test_layout_transpose --gtest_filter="*BatchedTranspose*"
```

---

## Forward Pooling Tests (Existing)

### `pooling2d_asymmetric.cpp` & `pooling2d_asymmetric_nhwc.cpp`

**Dataset 1:** Minimal asymmetric configs
- Input: `{1, 4, 4, 4}`
- Tests asymmetric padding/strides: `{0, 1}`, `{1, 0}`, `{1, 1}`

**NHWC Version:** Uses `TransposedPoolingFwd2d/Nd` with **batched transpose** for FP32/FP16

**Command:**
```bash
./bin/test_pooling2d_asymmetric       # NCHW
./bin/test_pooling2d_asymmetric_nhwc  # NHWC (uses batched transpose)
```

### `pooling2d_wide.cpp` & `pooling2d_wide_nhwc.cpp`

**Dataset 2:** Wide pooling windows
- Inputs: `{1, 3, 255, 255}`, `{2, 3, 227, 227}`, `{1, 7, 127, 127}`, `{1, 1, 410, 400}`
- Windows: `{35, 35}`, `{100, 100}`, `{255, 255}`, `{410, 400}`

**NHWC Version:** Uses `TransposedPoolingFwd2d/Nd` with **batched transpose** for FP32/FP16

**Command:**
```bash
./bin/test_pooling2d_wide       # NCHW
./bin/test_pooling2d_wide_nhwc  # NHWC (uses batched transpose)
```

### `pooling3d_ndhwc.cpp`

**3D Pooling:** Tests NDHWC layout (5D tensors)
- Uses **universal transpose** (batched only available for 2D NCHW↔NHWC)

**Command:**
```bash
./bin/test_pooling3d_ndhwc
```

---

## Standalone Driver Tests (`test/` directory)

### `pooling2d.cpp` / `pooling3d.cpp`

**Purpose:** Standalone test executables that invoke `test_drive<pooling2d_driver>` or `test_drive<pooling3d_driver>`

**Files:**
- `test/pooling2d.cpp` - Main executable entry point
- `test/pooling2d.hpp` - Driver template and test cases
- `test/pooling3d.cpp` - Main executable entry point
- `test/pooling3d.hpp` - Driver template and test cases
- `test/pooling_common.hpp` - Shared verification infrastructure

**Usage:**
```bash
# Standalone driver invocation (manual testing)
./bin/MIOpenDriver pool --input 1,3,32,32 --pool_stride 2,2 --pool_size 2,2 --forw 1

# Test executables (use pooling2d.hpp configurations)
./bin/test_pooling2d --float --all
./bin/test_pooling2d --half --all --in_layout NHWC --out_layout NHWC

./bin/test_pooling3d --float --all
```

**How They Work:**
1. `pooling2d.cpp` contains only: `int main(int argc, const char* argv[]) { test_drive<pooling2d_driver>(argc, argv); }`
2. `pooling2d.hpp` defines `pooling2d_driver` template with test case configurations
3. `pooling_common.hpp` provides CPU reference implementations for verification

**Test Cases in pooling2d.hpp:**
- Dataset 0: Standard configs (various input sizes)
- Dataset 1: Minimal asymmetric configs
- Dataset 2: Wide window configs

**Registered in CMakeLists.txt:**
- Line 461-462: Listed as `LONG_TESTS` (Cost: 800)
- Line 468: `set_tests_properties(${NAME} PROPERTIES COST 800)`

---

## Code Changes Summary

### Solvers Modified

**1. src/solver/pooling/backward2d.cpp:176-179**
```cpp
// ADDED: Type validation to prevent compile errors
bool PoolingBackward2d::IsApplicable(...) const
{
    return problem.GetDirection() == miopen::pooling::Direction::Backward &&
           problem.GetXDesc().GetType() == problem.GetYDesc().GetType() &&
           (problem.GetXDesc().GetType() == miopenFloat ||
            problem.GetXDesc().GetType() == miopenHalf ||
            problem.GetXDesc().GetType() == miopenBFloat16) &&  // NEW: Added BF16
           // ... other checks
}
```

**2. src/solver/pooling/backwardNd.cpp:50-52**
```cpp
// ADDED: BF16 support (was missing)
return problem.GetDirection() == miopen::pooling::Direction::Backward &&
       problem.GetXDesc().GetType() == problem.GetYDesc().GetType() &&
       (problem.GetXDesc().GetType() == miopenFloat ||
        problem.GetXDesc().GetType() == miopenHalf ||
        problem.GetXDesc().GetType() == miopenBFloat16) &&  // NEW
```

---

## Why Batched Transpose Matters

### Performance Comparison
| Transpose Type | FP32 | FP16 | BF16 | Algorithm | Use Case |
|----------------|------|------|------|-----------|----------|
| **Batched** (NCHW↔NHWC) | ✅ Fast | ✅ Fast | ✅ Fast | LDS-tiled, optimized for 4D | Forward/Backward 2D NHWC |
| **Universal** (Any→Any) | ✅ Slow | ✅ Slow | ✅ Slow | General-purpose, grid-stride | 3D layouts, fallback |

**Speed Difference:** Batched transpose is ~2-10× faster for NCHW↔NHWC conversions

### Actual Usage in Tests

| Test | Direction | Layout | Data Type | Transpose Used | Reason |
|------|-----------|--------|-----------|----------------|--------|
| `pooling2d_asymmetric_nhwc.cpp` | Forward | NHWC | FP32 | **Batched** | Inner FP32 + batched FP32 ✅ |
| `pooling2d_asymmetric_nhwc.cpp` | Forward | NHWC | FP16 | **Batched** | Inner FP16 + batched FP16 ✅ |
| `pooling2d_wide_nhwc.cpp` | Forward | NHWC | FP32 | **Batched** | Inner FP32 + batched FP32 ✅ |
| `pooling2d_wide_nhwc.cpp` | Forward | NHWC | FP16 | **Batched** | Inner FP16 + batched FP16 ✅ |
| `pooling_backward_nhwc.cpp` (NEW) | Backward | NHWC | FP32 | **Batched** | Inner FP32 + batched FP32 ✅ |
| `pooling_backward_nhwc.cpp` (NEW) | Backward | NHWC | FP16 | **Batched** | Inner FP16 + batched FP16 ✅ |
| `pooling_backward_nhwc.cpp` (NEW) | Backward | NHWC | BF16 | **Batched** | Inner BF16 + batched BF16 ✅ **NEW** |
| `pooling3d_ndhwc.cpp` | Forward | NDHWC | Any | **Universal** | 3D not supported by batched |

---

## Build & Run All Tests

### Build
```bash
cd build
cmake .. -DMIOPEN_TEST_ALL=ON
make -j$(nproc)
```

### Run Unit Tests (Fast)
```bash
# Backward applicability tests
./bin/test_pooling_backward --gtest_filter="Smoke*"           # < 1s

# Transpose tests
./bin/test_layout_transpose --gtest_filter="*BatchedTranspose*"  # < 1s
```

### Run Driver Tests (Slow - First Run)
```bash
# NEW: Backward with NHWC
./bin/test_pooling_backward_nhwc --gtest_filter="Smoke*"      # ~30-60s (1st run)

# Existing: Forward with NHWC (batched transpose)
./bin/test_pooling2d_asymmetric_nhwc
./bin/test_pooling2d_wide_nhwc

# Existing: 3D (universal transpose)
./bin/test_pooling3d_ndhwc
```

### Run Standalone Tests
```bash
# Standalone executables
./bin/test_pooling2d --float --all
./bin/test_pooling3d --float --all

# Or via ctest
ctest -R test_pooling2d -V
ctest -R test_pooling3d -V
```

---

## Success Criteria

### Unit Tests ✅
- [x] FP32, FP16, BF16 return `IsApplicable() == true` for backward solvers
- [x] Int8, Int32, Double return `IsApplicable() == false` (rejected at applicability check)
- [x] Batched transpose applicable for NHWC with supported types
- [x] Edge cases handled (forward direction rejected, type mismatch rejected)

### Driver Tests ✅
- [x] Kernels compile successfully for FP32, FP16, BF16
- [x] GPU output matches CPU reference within tolerance
- [x] Batched transpose selected (not universal) for forward and backward NHWC
- [x] BF16 precision maintained through transpose→pool→transpose pipeline

### No Regressions ✅
- [x] Existing forward pooling tests still pass (NCHW and NHWC)
- [x] 3D pooling tests still pass (NDHWC with universal transpose)
- [x] Universal transpose fallback works for unsupported types

---

## Notes

1. **Test Organization:**
   - **Standalone drivers:** `test/pooling2d.cpp`, `test/pooling3d.cpp` (manual testing)
   - **Applicability tests:** Fast unit tests (pooling_backward.cpp, layout_transpose.cpp)
   - **Execution tests:** Full driver tests (pooling_backward_nhwc.cpp, pooling2d_*_nhwc.cpp)

2. **Batched Transpose Usage - CORRECTED:**
   - ✅ **Forward pooling NHWC:** DOES use batched transpose for FP32/FP16
   - ✅ **Backward pooling NHWC:** DOES use batched transpose for FP32/FP16/BF16 (NEW)
   - ❌ **3D pooling NDHWC:** Does NOT use batched (only supports 2D), falls back to universal
   - **Selection is automatic** based on layout + data type compatibility

3. **Why Forward Didn't Have BF16 Before:**
   - Forward pooling solvers (`PoolingForward2d/Nd`) only support FP32, FP16
   - Batched transpose supports FP32, FP16, BF16, Int8, Int32
   - Forward with FP32/FP16 → batched transpose works
   - Forward with BF16 → pooling solver rejects (not batched transpose issue)

4. **BF16 Importance:**
   - MI300X/gfx940+ prefer BF16 for training
   - Wider dynamic range than FP16 (prevents gradient overflow)
   - Same memory bandwidth as FP16

5. **Automatic Test Discovery:**
   - gtest/CMakeLists.txt line 197: `file(GLOB TESTS *.cpp)` - All `*.cpp` files auto-included
   - test/CMakeLists.txt: Manually lists `test_pooling2d` and `test_pooling3d` as LONG_TESTS

---

## Test File Summary

| File | Location | Lines | Purpose | Direction | Layout | Types | Speed |
|------|----------|-------|---------|-----------|--------|-------|-------|
| `pooling_backward.cpp` | gtest/ | 320 | Unit tests | Backward | NCHW | FP32, FP16, BF16 | < 1s |
| `pooling_backward_nhwc.cpp` | gtest/ | 193 | **Driver tests (NEW)** | **Backward** | **NHWC** | FP32, FP16, **BF16** | ~30-60s |
| `layout_transpose.cpp` | gtest/ | 521 | Transpose tests | N/A | NCHW↔NHWC | All types | < 1s |
| `pooling2d_asymmetric.cpp` | gtest/ | ~150 | Driver tests | Forward | NCHW | FP32, FP16 | Variable |
| `pooling2d_asymmetric_nhwc.cpp` | gtest/ | ~150 | Driver tests | Forward | NHWC | FP32, FP16 | Variable |
| `pooling2d_wide.cpp` | gtest/ | ~150 | Driver tests | Forward | NCHW | FP32, FP16 | Variable |
| `pooling2d_wide_nhwc.cpp` | gtest/ | ~150 | Driver tests | Forward | NHWC | FP32, FP16 | Variable |
| `pooling3d_ndhwc.cpp` | gtest/ | ~150 | Driver tests | Forward | NDHWC | FP32, FP16 | Variable |
| `pooling2d.cpp` + `.hpp` | test/ | ~100 | Standalone driver | Both | Both | All | Manual |
| `pooling3d.cpp` + `.hpp` | test/ | ~100 | Standalone driver | Both | Both | All | Manual |
| `pooling_common.hpp` | test/ | ~500 | Shared infra | N/A | N/A | N/A | N/A |

**Total Test Coverage:**
- 11 test files (8 gtest + 3 standalone)
- ~2,400 lines of test code
- Forward + Backward directions
- NCHW + NHWC + NDHWC layouts
- FP32, FP16, BF16 precision types
- 2D + 3D pooling operations
- Batched (fast) + Universal (fallback) transpose strategies
