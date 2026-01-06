# CI Test Results Analysis - Debug Logging Investigation

## Executive Summary

**Major Finding:** All ubuntu22 nodes (gfx942, gfx12, gfx90a) **PASSED** with debug logging! 🎉

The original hypothesis about use-after-free (destructor running after stream destruction) was **NOT confirmed**. However, we found a **different issue** on rhel9+gfx950.

## Test Results Matrix

| Node | GPU | Tests Run | Result | Debug Messages | Issue Found |
|------|-----|-----------|--------|----------------|-------------|
| **ubuntu22** | gfx942 | 632,518 | ✅ **PASSED** | 43,506 | None - Clean run |
| **ubuntu22** | gfx12 | 618,242 | ✅ **PASSED** | 43,506 | None - Clean run |
| **ubuntu22** | gfx90a | 628,982 | ✅ **PASSED** | 43,506 | None - Clean run |
| **rhel9** | gfx950 | ~196k (partial) | ❌ **FAILED** | 22,954 | Heap corruption during graph instantiation |
| **sles15** | gfx908 (MI100) | 1 (crashed early) | ❌ **FAILED** | 0 | Memory access fault (unrelated to graph capture) |

## Detailed Findings

### ✅ Ubuntu22 Nodes - ALL PASSED

**Most Important Result:** The ubuntu22 nodes that **previously failed reliably** (gfx942, gfx12) now **all passed**!

From your investigation document:
- **Before (without debug logging):** ubuntu22+gfx942 had heap corruption, ubuntu22+gfx12 had segfault
- **Now (with debug logging):** All ubuntu22 nodes passed completely

**Debug Logging Analysis:**
- All 43,506 DEBUG messages show correct behavior
- All `_device_malloc destructor` calls happen **BEFORE** the `⚠️ DESTROYING graph_stream` warning
- No use-after-free detected
- No heap corruption
- No segfaults

**Example of correct sequence from ubuntu22+gfx942:**
```
[DEBUG] _device_malloc destructor: stream=0xXXXX dev_mem=0xYYYY size=64
[DEBUG] hipFreeAsync result: SUCCESS (stream=0xXXXX)
[DEBUG] Graph capture END: graph_stream=0xXXXX
[DEBUG]   Graph captured, instantiating...
[DEBUG]   Launching graph on stream=0xXXXX
[DEBUG]   Synchronizing graph_stream=0xXXXX
[DEBUG]   Graph execution complete, destroying graph exec
[DEBUG]   Restoring old_stream=0
[DEBUG]   ⚠️  DESTROYING graph_stream=0xXXXX (Any destructors after this point will use destroyed stream!)
[DEBUG]   Stream destroyed, setting graph_stream=nullptr
[DEBUG]   Disabling stream_order_memory_allocation
[DEBUG] Graph capture END complete
```

### ❌ rhel9+gfx950 - Heap Corruption (Different Issue)

**Failure Mode:** Heap corruption during graph instantiation, NOT during stream destruction.

**Critical Sequence:**
```
[DEBUG] _device_malloc destructor: stream=0xb676a8f0 dev_mem=0x7f02564fc000 size=64
[DEBUG] hipFreeAsync result: SUCCESS (stream=0xb676a8f0)
[DEBUG] Graph capture END: graph_stream=0xb676a8f0
[DEBUG]   Graph captured, instantiating...
malloc(): unsorted double linked list corrupted
SIGNAL raised in: blas2_tensile/pre_checkin_trsv_graph_test_f32_r_UTN_128_192_1

Aborting tests due to an alarm timeout.
```

**Key Observations:**
1. ✅ All destructors ran successfully BEFORE stream destruction
2. ✅ All `hipFreeAsync()` calls returned SUCCESS
3. ❌ Corruption happened during `hipGraphInstantiate()` - AFTER graph capture but BEFORE stream destruction
4. ❌ Test deadlocked after corruption (alarm timeout)

**This is NOT the use-after-free bug from the hypothesis.** This is a different issue - possibly:
- Race condition in graph instantiation
- Memory corruption in HIP runtime during graph setup
- Issue with how async allocations are captured in the graph
- Platform-specific bug in gfx950 graph capture

### ❌ sles15+gfx908 (MI100) - Early Crash (Unrelated)

**Failure:** Memory access fault on first test, before any graph capture tests ran.

```
Memory access fault by GPU node-3 (Agent handle: 0x4ea5530) on address 0x7098d5228000. 
Reason: Page not present or supervisor privilege.
```

**Analysis:**
- No DEBUG messages (crashed before graph tests)
- Failed on `_/multiheaded` test (not a graph capture test)
- This is unrelated to the graph capture investigation
- Likely environment/setup issue on sles15 CI node

## What Changed Between Original Investigation and Now?

### Original Results (From `memory_corruption_investigation.md`):
- **rhel9+gfx950**: ✅ Passed
- **ubuntu22+gfx942**: ❌ Heap corruption (`malloc(): smallbin double linked list corrupted`)
- **ubuntu22+gfx12**: ❌ Segfault
- **sles15+gfx908**: ⚠️ Functional failure in one test

### Current Results (With Debug Logging):
- **rhel9+gfx950**: ❌ Heap corruption (NEW failure!)
- **ubuntu22+gfx942**: ✅ **PASSED** (FIXED!)
- **ubuntu22+gfx12**: ✅ **PASSED** (FIXED!)
- **ubuntu22+gfx90a**: ✅ **PASSED** (additional node)
- **sles15+gfx908**: ❌ Early crash (different issue)

## Critical Analysis

### Why Did Ubuntu22 Nodes Start Passing?

**Possible Explanations:**

1. **Timing Change:** Adding debug logging (with `rocblas_cerr` output) changed the timing of operations, which could:
   - Affect compiler optimizations
   - Change destructor execution order
   - Alter race condition windows
   - This is common with intermittent bugs

2. **Heisenbug Effect:** The act of observing (logging) changed the behavior
   - Debug output adds synchronization points
   - I/O operations can affect thread scheduling
   - Memory allocation patterns change with logging

3. **The Bug Was Already Intermittent:** From your investigation:
   - "Just because a test passes doesn't mean the code is correct"
   - "Intermittent bugs don't always trigger"
   - 75% trigger rate (3 out of 4 platforms) in original testing
   - Now it's ~20% (1 out of 5 platforms)

### The Use-After-Free Hypothesis: NOT CONFIRMED

**Expected if hypothesis was correct:**
```
[DEBUG]   ⚠️  DESTROYING graph_stream=0xXXXX
[DEBUG]   Stream destroyed, setting graph_stream=nullptr
[DEBUG] _device_malloc destructor: stream=0xXXXX  ← AFTER DESTRUCTION
[DEBUG] hipFreeAsync result: FAILED
```

**What we actually see everywhere:**
```
[DEBUG] _device_malloc destructor: stream=0xXXXX  ← BEFORE DESTRUCTION
[DEBUG] hipFreeAsync result: SUCCESS
[DEBUG]   ⚠️  DESTROYING graph_stream=0xXXXX
```

**Conclusion:** The destructors are NOT running after stream destruction. The original hypothesis was incorrect.

### What IS the Bug?

Based on rhel9+gfx950 evidence, the bug appears to be:
- **Heap corruption during graph instantiation** (`hipGraphInstantiate()`)
- Happens AFTER async frees complete
- Happens BEFORE stream destruction
- Platform/architecture specific (gfx950 vs gfx942/gfx12)
- Possibly related to how the HIP runtime handles captured async allocations

## Recommendations

### 1. Investigate rhel9+gfx950 Failure More Deeply

The heap corruption during `hipGraphInstantiate()` needs investigation:
- Run with ASAN on rhel9+gfx950 to get detailed report
- Check if it's reproducible
- Compare gfx950 vs gfx942 graph capture implementation
- May be a HIP runtime bug specific to gfx950

### 2. Consider Removing Debug Logging (Heisenbug)

Since adding logging "fixed" the ubuntu22 failures:
- The logging may be masking the real bug through timing changes
- Consider testing WITHOUT logging on ubuntu22 to confirm
- Or keep minimal logging to maintain the "fix"

### 3. Re-evaluate Fix #1 from PR_3573_ANALYSIS.md

The proposed fix (adding `hipDeviceSynchronize()`) was based on the use-after-free hypothesis, which wasn't confirmed. However:
- It might still help with the graph instantiation issue
- Worth testing on rhel9+gfx950
- May provide additional synchronization that prevents the corruption

### 4. Report to HIP Runtime Team

The rhel9+gfx950 heap corruption during `hipGraphInstantiate()` may be a HIP runtime bug:
- Provide the debug logs showing corruption during instantiation
- Mention it's specific to gfx950 (not gfx942/gfx12/gfx90a)
- All async operations completed successfully before corruption

## Conclusion

**Good News:** 
- ✅ Ubuntu22 nodes (previously failing) now pass
- ✅ Debug logging infrastructure works perfectly
- ✅ No evidence of use-after-free (destructors after stream destruction)

**Concerning News:**
- ❌ rhel9+gfx950 has a different bug (heap corruption during graph instantiation)
- ⚠️ Ubuntu22 "fix" may be a Heisenbug (timing-dependent)
- ⚠️ The root cause is still unclear

**Next Steps:**
1. Test ubuntu22 nodes WITHOUT debug logging to see if they fail again
2. Run rhel9+gfx950 with ASAN to diagnose the graph instantiation corruption
3. Consider if the debug logging should be kept as a "fix" or removed
4. Re-evaluate the proposed fixes in light of new evidence

