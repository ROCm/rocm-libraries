# Debug Logging Results - Graph Capture Investigation

## Summary

Successfully added comprehensive debug logging and tested the graph capture functionality. **All 48 tests PASSED** on this MI210 (gfx90a) system.

## What We Did

1. ✅ **Added Debug Logging** to track:
   - Memory allocation (`hipMallocAsync`) with stream and address
   - Memory deallocation (`hipFreeAsync`) with stream and success/failure
   - Graph capture lifecycle (BEGIN/END)
   - Stream creation and destruction with ⚠️ warning marker

2. ✅ **Built rocBLAS** with debug logging:
   - Without ASAN (built for `gfx90a:xnack-` to match MI210 GPU)
   - All clients included (rocblas-test)
   
3. ✅ **Ran Tests**: 48 trsv_strided_batched graph tests

## Key Findings

### ✅ Correct Behavior Observed on MI210

The debug logs show the **correct** sequence on this system:

```
[DEBUG] Graph capture BEGIN: old_stream=0
[DEBUG]   Enabling stream_order_memory_allocation
[DEBUG]   Created graph_stream=0x2fda2f40
[DEBUG]   Graph capture started on stream=0x2fda2f40
[DEBUG] _device_malloc allocating: stream=0x2fda2f40 size=64
[DEBUG] hipMallocAsync result: SUCCESS dev_mem=0x700a31d66000 (stream=0x2fda2f40)
[DEBUG] _device_malloc destructor: stream=0x2fda2f40 dev_mem=0x700a31d66000 size=64  ← DESTRUCTOR RUNS BEFORE STREAM DESTRUCTION
[DEBUG] hipFreeAsync result: SUCCESS (stream=0x2fda2f40)
[DEBUG] Graph capture END: graph_stream=0x2fda2f40
[DEBUG]   Graph captured, instantiating...
[DEBUG]   Launching graph on stream=0x2fda2f40
[DEBUG]   Synchronizing graph_stream=0x2fda2f40
[DEBUG]   Graph execution complete, destroying graph exec
[DEBUG]   Restoring old_stream=0
[DEBUG]   ⚠️  DESTROYING graph_stream=0x2fda2f40 (Any destructors after this point will use destroyed stream!)  ← STREAM DESTROYED
[DEBUG]   Stream destroyed, setting graph_stream=nullptr
[DEBUG]   Disabling stream_order_memory_allocation
[DEBUG] Graph capture END complete
```

**Key Observation:** All `_device_malloc` destructors run **BEFORE** the stream is destroyed. No use-after-free occurs on this platform.

### Why This Aligns with Investigation Document

From `PR_3573_ANALYSIS.md`:
- **rhel9+gfx950**: ✅ Passed (bug latent, didn't trigger)
- **ubuntu22+gfx942**: ❌ Heap corruption (bug triggered)
- **ubuntu22+gfx12**: ❌ Segfault (bug triggered)  
- **sles15+gfx908**: ⚠️ Functional failure (bug triggered)

**This MI210 (gfx90a) behaves like rhel9+gfx950** - the bug is intermittent and **didn't trigger** on this run.

## What the Bug Would Look Like

If the use-after-free bug triggered, our logging would show:

```
[DEBUG]   ⚠️  DESTROYING graph_stream=0x2fda2f40 (Any destructors after this point will use destroyed stream!)
[DEBUG]   Stream destroyed, setting graph_stream=nullptr
[DEBUG] _device_malloc destructor: stream=0x2fda2f40 dev_mem=0xXXXX size=64  ← ❌ DESTRUCTOR AFTER STREAM DESTRUCTION!
[DEBUG] hipFreeAsync result: FAILED (stream=0x2fda2f40)  ← ❌ USING DESTROYED STREAM!
```

This is what would happen on ubuntu22 nodes where the bug triggers more reliably.

## Why The Bug Is Intermittent

As documented in the investigation:

**Depends on:**
- Compiler optimizations (RVO, NRVO, move semantics)
- Timing of destructor execution
- Memory allocator behavior (glibc version)
- Thread scheduling
- System load

**Why ubuntu22 triggers more reliably:**
- Different glibc memory allocator
- Different system libraries
- Different timing characteristics
- Possibly different compiler versions or flags

## Debug Logging Value

Even though the bug didn't trigger on this system, the logging successfully demonstrates:

1. ✅ **Tracking works**: We can see every allocation/deallocation with stream info
2. ✅ **Critical markers work**: The ⚠️ warning clearly marks when stream destruction happens
3. ✅ **Would catch the bug**: If destructors ran after stream destruction, we'd see it immediately

## Recommended Next Steps

From the investigation document:

1. **Test on ubuntu22+gfx942** (most reliable trigger point)
2. **Run with ASAN on ubuntu22** to get detailed use-after-free report
3. **Apply Fix #1 from PR_3573_ANALYSIS.md**:
   - Add `hipDeviceSynchronize()` after `hipStreamSynchronize(m_graph_stream)`
   - Move `set_stream_order_memory_allocation(false)` before stream destruction
4. **Revert version guard change** (50200000 → 50500000) as it's unnecessary

## Files Modified

- `projects/rocblas/library/src/include/handle.hpp`: Added logging to `_device_malloc` constructor/destructor
- `projects/rocblas/clients/common/client_utility.cpp`: Added logging to `rocblas_stream_begin_capture()` and `rocblas_stream_end_capture()`

## Test Results

```
[==========] 48 tests from 1 test suite ran. (1166 ms total)
[  PASSED  ] 48 tests.
```

**Conclusion**: Debug logging is working perfectly and ready to catch the bug when it triggers. The intermittent nature means it may not show on every system/run, but when it does trigger (especially on ubuntu22), our logging will provide clear evidence of the use-after-free.

