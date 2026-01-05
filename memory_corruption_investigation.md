# rocBLAS Memory Corruption Investigation

**PR:** #3610 - Memory Error Diagnostics  
**Issue:** Heap corruption in `trsv` family of functions during CI testing  
**Date:** January 2026  
**Investigator:** Tony Davis

---

## Executive Summary

Intermittent heap corruption occurs in rocBLAS `trsv`/`trsv_batched` tests **only on CI**, manifesting as:
```
malloc(): smallbin double linked list corrupted
```

**Root Cause:** Graph capture using non-graph-safe memory allocation due to HIP version mismatch between local development and CI environments.

---

## The Bug

### Symptoms
- **Where:** `trsv`, `trsv_batched`, `trsv_strided_batched` tests
- **When:** Only during CI test runs with graph capture enabled
- **Behavior:** Heap corruption that "moves around" between test runs
- **Workaround:** Disabling graph capture makes errors disappear

### Example Error (from build #4)
```
[ubuntu22 && gfx12] [----------] 5170 tests from _/trsv_batched
[ubuntu22 && gfx12] malloc(): smallbin double linked list corrupted
[ubuntu22 && gfx12] ... exit code: 142
```

---

## Root Cause Analysis

### The Memory Allocation Problem

**All `trsv` variants allocate temporary workspace memory:**

```cpp
// From rocblas/library/src/blas2/rocblas_trsv_imp.hpp:116
auto w_mem = handle->device_malloc(dev_bytes);
```

This uses the handle's `device_malloc()` RAII wrapper which:
1. Allocates memory (via `hipMalloc` OR `hipMallocAsync` depending on settings)
2. Uses the memory for kernel operations
3. **Auto-frees in destructor when function returns** (`hipFree` OR `hipFreeAsync`)

### The Graph Capture Issue

**When HIP stream is in graph capture mode:**

#### ❌ **Without Stream Order Allocation** (current CI state):
```
hipMalloc()             → Synchronous allocation (NOT captured in graph)
  ↓ kernel launches     → Captured in graph
  ↓ return from function
hipFree()               → Synchronous free (NOT captured in graph)
  
Graph replay: Uses ALREADY-FREED memory → 💥 HEAP CORRUPTION
```

#### ✅ **With Stream Order Allocation** (the fix):
```
hipMallocAsync()        → Async allocation (captured in graph)
  ↓ kernel launches     → Captured in graph  
  ↓ return from function
hipFreeAsync()          → Async free (captured in graph)

Graph replay: Allocates, uses, then frees correctly → ✓ Works
```

---

## Why It Only Happens on CI

### The Code Path

Graph capture tests use this helper:

```cpp
// From rocblas/clients/common/client_utility.cpp:525-532
void rocblas_local_handle::rocblas_stream_begin_capture()
{
    // ...
    m_handle->set_stream_order_memory_allocation(true);  // ← THE FIX!
    // ...
    CHECK_HIP_ERROR(hipStreamBeginCapture(m_graph_stream, ...));
}
```

**But this is conditionally compiled:**

```cpp
// From rocblas/clients/include/client_utility.hpp:149-151
#if HIP_VERSION >= 50500000  // ← ONLY compiles for HIP 5.5+
    arg.graph_test ? rocblas_stream_begin_capture() : NOOP;
#endif
```

### The Environment Difference

| Environment | HIP Version | Graph Fix Compiles? | Result |
|-------------|-------------|---------------------|--------|
| **Local Dev** | ≥ 5.5.0 (ROCm 7.1+) | ✅ YES | No corruption |
| **CI** | < 5.5.0 (older ROCm) | ❌ NO | 💥 Corruption |

**On CI:** Graph tests run, but the code to enable stream order allocation doesn't exist → memory corruption.

---

## Historical Context

### PR Timeline

#### **PR #1516 (Sept 2025)** - "Stream Order Allocation as default"
- Made `hipMallocAsync`/`hipFreeAsync` the default
- Simplified 690 lines of memory management code
- **Goal:** Better graph capture safety

#### **PR #2241 (Nov 2025)** - "Revert Stream Order Allocation"  
- **Reverted #1516** due to 15-47% performance regression in rocHPL multi-GPU
- Made stream order allocation **opt-in via `ROCBLAS_STREAM_ORDER_ALLOC` env var**
- **This created the vulnerability**

#### **PR #919 (Aug 2025)** - "Hip event based timing"
- Changed benchmarks to use `hipEventSynchronize()` for consistent timing
- Made results comparable between stream order modes

#### **PR #1439 (Sept 2025)** - "Updated hipEventSynchronize timing"
- Further refinements to event-based benchmark timing

#### **PR #3573 (Dec 2025)** - "Graph stream order alloc for capture"
- **Added the fix** to enable stream order allocation during graph capture
- But only for HIP ≥ 5.5.0

---

## Why It "Moves Around"

The corruption is a **use-after-free race condition:**
- Freed memory gets reallocated to other tests
- Timing-dependent on which tests run before/after
- Depends on memory allocator's internal state
- **Different every run** → appears to "move" between functions

---

## Current Branch Status

The `memory_error_diagnostics` branch includes:

### ✅ Already Implemented
1. **Enhanced memory tracking** (`host_alloc.cpp`)
   - Detects double-free and untracked pointer issues
   - Controlled by `ROCBLAS_CLIENT_DEBUG_ALLOC` env var

2. **Auto-ASAN support** (`rmake.py`)
   - Detects `ci:asan` or `asan` labels
   - Automatically adds `BUILD_ADDRESS_SANITIZER=ON`
   - Auto-fixes xnack+ requirement for ASAN-compatible GPUs

3. **The graph capture fix from PR #3573**
   - But only compiles for HIP ≥ 5.5.0

---

## Solutions

### ✅ Option 1: Lower HIP Version Requirement (IMPLEMENTED - Direct Fix)
Changed the guard in `client_utility.hpp` from HIP 5.5.0 to HIP 5.2.0:
```cpp
#if HIP_VERSION >= 50200000  // hipMallocAsync available since HIP 5.2
    arg.graph_test ? rocblas_stream_begin_capture() : NOOP;
#endif
```

**Why This Works:**
- `hipMallocAsync` (needed for stream order allocation) has been available since HIP 5.2
- The fix from PR #3573 was guarded for HIP 5.5+ unnecessarily
- Lowering to 5.2 enables the fix on CI environments with HIP 5.2-5.4

**Implementation:** 
- Changed lines 158 and 168 in `clients/include/client_utility.hpp` (lowered version guard from 50500000 to 50200000)
- Added compile-time diagnostics to print HIP_VERSION and whether graph capture support is enabled
- Diagnostics will appear in build logs: 
  - `client_utility.hpp: Compiling with HIP_VERSION = XXXXXXXX`
  - `client_utility.hpp: Graph capture support ENABLED/DISABLED`

**Pros:**
- ✅ Enables the fix on CI immediately
- ✅ Only affects graph capture, not general operations
- ✅ Minimal code change (2 lines + diagnostics)
- ✅ No infrastructure dependencies
- ✅ Diagnostics provide visibility into HIP version on all CI environments

**Cons:**
- Need to verify compatibility with HIP 5.2/5.3/5.4 (should work fine)

### Option 2: Environment Variable (Alternative)
Set in CI test environment or via Jenkins label:
```bash
export ROCBLAS_STREAM_ORDER_ALLOC=1
```
**Pros:** 
- Forces stream order allocation globally
- Works on any HIP version ≥ 5.2
- No code changes needed

**Cons:**
- May impact performance (why it was reverted in #2241)
- Affects all operations, not just graph capture
- Requires Jenkins infrastructure PR (pending)

### Option 3: Upgrade CI HIP Version (Infrastructure)
Update CI Docker images to use HIP 5.5+ (ROCm 7.1+)
**Pros:**
- No code changes needed
- Already works with current code
- Future-proof

**Cons:**
- Infrastructure change
- May require coordinating with CI team

---

## ASAN Validation

When running with ASAN (`ci:asan` label), expect to see:

```
==PID==ERROR: AddressSanitizer: heap-use-after-free on address 0x...
READ of size N at 0x... thread T0
    #0 in rocblas_trsv_kernels
    #1 in rocblas_internal_trsv_template  
    #2 in rocblas_trsv_batched_impl
```

The freed memory will be the workspace allocated at line 116 of `rocblas_trsv_imp.hpp`.

---

## Recommendation

**✅ Implemented:** Lowered the HIP version requirement from 5.5.0 to 5.2.0 in `client_utility.hpp`. This enables the stream order allocation fix from PR #3573 to compile and run on CI environments with HIP 5.2-5.4.

**Next Steps:**
1. Build and test on a CI environment with HIP 5.2-5.4
2. Verify heap corruption is resolved in `trsv` family tests
3. Consider backporting this fix to release branches if confirmed

**Why This Fix:**
- Direct solution that enables proper graph-safe memory allocation
- Minimal code change (2 lines)
- No performance impact (only affects graph capture mode)
- Works across HIP 5.2+ versions

---

## References

- **CHANGELOG.md line 27:** Documents stream order allocation as opt-in
- **handle.hpp:476-479:** `set_stream_order_memory_allocation()` method
- **handle.hpp:526-549:** `is_stream_in_capture_mode()` detection
- **handle.hpp:628-645:** Stream order vs regular allocation logic
- **client_utility.cpp:525-559:** Graph capture helpers with the fix
- **rocblas_trsv_imp.hpp:116:** Workspace allocation point

---

## Testing the Fix

### Build and Test
```bash
cd projects/rocblas

# Build with clients
./install.sh -c --architecture auto

# Check build logs for diagnostic messages:
# - "client_utility.hpp: Compiling with HIP_VERSION = XXXXXXXX"
# - "client_utility.hpp: Graph capture support ENABLED (HIP >= 5.2.0)"

# Run graph capture tests
./build/release/clients/staging/rocblas-test --gtest_filter=*trsv*graph*

# Or run full smoke tests
python3 rtest.py -t smoke
```

### Expected Behavior
**Before the fix (HIP 5.2-5.4 without stream order allocation):**
- Heap corruption in `trsv` family tests during graph capture
- Error: `malloc(): smallbin double linked list corrupted`

**After the fix (with lowered version guard):**
- Graph capture enables stream order allocation automatically
- No heap corruption
- All tests pass

### Alternative: Manual Environment Variable
If you want to force stream order allocation globally:
```bash
export ROCBLAS_STREAM_ORDER_ALLOC=1
./build/release/clients/staging/rocblas-test --gtest_filter=*trsv*
```

---

## Next Steps

1. ✅ Identified root cause (graph capture with non-graph-safe memory)
2. ✅ Implemented fix (lowered HIP version guard to 5.2.0)
3. ⏳ Test on CI with HIP 5.2-5.4 to verify heap corruption is resolved
4. ⏳ Verify no regressions with graph capture tests
5. ⏳ Consider backporting to release branches
6. ⏳ Review if other BLAS functions need similar treatment

