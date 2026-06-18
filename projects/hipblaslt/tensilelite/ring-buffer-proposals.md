# Ring-Buffer Implementation — Proposals

**Branch:** `users/alvasile/ring_buffer`
**Scope:** `projects/hipblaslt/tensilelite/client/`

---

## Context

The ring-buffer mechanism maintains up to `m_numActiveBuffers` pre-filled GPU buffer sets (slots) for tensor data. One slot is active (used by the running kernel); the others are alt slots pre-filled by `initializeAltBufferSets`. When the fast path fires in `prepareGPUInputs`, `advanceBuffer` rotates to the next pre-filled slot and returns `m_cachedGPUInputs` without re-running tensor copy. The ring is gated by `ringEligible()`, a five-term predicate:

```cpp
bool ringEligible() const {
    return m_hasAltBuffers
        && m_copyStream
        && m_gpuInit
        && m_curBoundsCheck == BoundsCheckMode::Disable
        && !m_problemDependentData;
}
```

`beginAsyncReset(problem)` replenishes a consumed alt slot asynchronously on `m_copyStream`. It has two internal paths:
- **Warm path** (`m_altSlotsFilled == true`): the slot was filled by `initializeAltBufferSets` at problem start and has never been written by the compute kernel; currently records a no-op event and increments `m_availableSlots` without issuing any DMA.
- **Cold path**: calls `fillSlot(targetIdx, problem, m_copyStream)` which copies all tensors into the target slot, then records a copy-done event.

**Baseline (develop branch).** On develop, `prepareGPUInputsInternal` takes a fast path when `m_gpuInit && BoundsCheck::Disable && !problemDependentData`: if `m_elementsToValidate > 0` it calls `resetOutput(...)` — a synchronous `hipMemcpy` of D only (from `gpuInput.valid` to `gpuInput.current`) — then returns `m_cachedGPUInputs`. A/B/C are not re-copied between solutions for the same problem because their content is invariant across solutions. D is reset synchronously once per solution, before the benchmark run, to ensure the reference comparison for validation sees a defined initial value.

---

## Proposal 1: D-only async reset on the warm path

### Problem

The warm path in `beginAsyncReset` (lines ~347–353 of `DataInitialization.hpp`) records a no-op event and returns without resetting D in the target slot:

```cpp
if(m_altSlotsFilled)
{
    HIP_CHECK_EXC(hipEventRecord(m_copyDoneEvents[targetIdx], m_copyStream));
    m_availableSlots++;
    return;
}
```

The comment justifying this reads: "D's initialized value is irrelevant (the next kernel fully overwrites it)." That is true for pure benchmarking — the kernel unconditionally writes every element of D. However, it does not hold when `m_elementsToValidate > 0`: the validation listener compares the kernel's output D against a reference computed from the same initial inputs. If D in the alt slot was not reset from `gpuInput.valid` before the kernel ran, the comparison is still valid (the kernel fully overwrites D regardless), but the divergence from develop's explicit reset contract is a latent risk as the code evolves.

More concretely: develop always calls `resetOutput` before using a slot, whether via its own fast path (`prepareGPUInputsInternal`) or elsewhere. The ring's warm path bypasses this entirely. Any future change that conditions validation on D's pre-kernel value (e.g. for partial-write kernels or masked output modes) would silently break the ring path without any assertion or guard.

Separately, the warm path was designed for the benchmark case where no validation occurs. For no-benchmark runs (validation-only / warmup-only), the warm path fires on every slot advance, meaning D is never reset between solutions within a problem for those runs.

### Proposed change

Replace the no-op event record in the warm path with an async D-only reset:

```cpp
if(m_altSlotsFilled)
{
    // Reset D in the target slot asynchronously on m_copyStream.
    // A/B/C are invariant within a problem (see invariant below) and are not re-copied.
    SlotGuard guard(m_vdata, targetIdx);
    resetOutput(m_gpuPtrsRing[targetIdx],
                m_gpuBatchPtrsRing[targetIdx],
                m_maxElements,
                m_groupedOffsets,
                problem,
                hipMemcpyDeviceToDevice,
                m_copyStream);
    HIP_CHECK_EXC(hipEventRecord(m_copyDoneEvents[targetIdx], m_copyStream));
    m_availableSlots++;
    return;
}
```

The `SlotGuard` RAII type already exists in `fillSlot` to redirect `gpuInput.current`/`batch` to the target slot for the duration of the call; the same guard is required here so that `resetOutput` writes into `targetIdx`'s buffers, not the active slot's.

`resetOutput` is an async-capable function: it takes a `targetStream` parameter and uses `hipMemcpyDeviceToDevice` with that stream when `kind == hipMemcpyDeviceToDevice`. The D-only reset is genuinely asynchronous and will overlap with the previous solution's post-benchmark work.

The existing `waitCopyDone` call in `main.cpp` (before kernel launch) ensures the reset DMA is complete before the slot is consumed by the compute stream.

### Why A/B/C are safe to skip

A/B/C in an alt slot are safe to skip on the warm path because:

1. `initializeAltBufferSets` (called at first problem entry, after `prepareGPUInputsInternal` returns slot 0's inputs) calls `fillSlot` for each alt slot, copying A/B/C from the same `gpuInput.valid` source used for slot 0.
2. A/B/C do not change between solutions within the same problem — they are input tensors whose values are fixed by the problem's initialization.
3. Alt slots are never written by the compute kernel — the benchmark loop dispatches only to `m_activeIdx` (the active slot). `beginAsyncReset` targets alt slots, never the active slot.
4. When the problem changes, `cancelAsyncReset` is called (main.cpp line ~1240, at `preProblem`), which clears `m_altSlotsFilled` and clears `m_gpuPtrsRing[1..N]`. The subsequent first call to `beginAsyncReset` for the new problem will have `m_altSlotsFilled == false` and will take the cold path (`fillSlot`), re-copying all tensors including A/B/C.

Therefore, on the warm path, A/B/C in the target slot are guaranteed to match the current problem's data.

### Interaction with correctness validation

`m_elementsToValidate` controls whether the validation listener compares results. On develop, `resetOutput` is called in `prepareGPUInputsInternal`'s fast path conditionally on `m_elementsToValidate`:

```cpp
if(m_elementsToValidate)
{
    resetOutput(m_gpuPtrs, m_gpuBatchPtrs, m_maxElements, m_groupedOffsets,
                problem, kind, targetStream);
}
return m_cachedGPUInputs;
```

The warm-path fix should mirror this condition: only issue the async D reset when `m_elementsToValidate > 0`. When `m_elementsToValidate == 0` (pure benchmarking), the no-op event path is correct and avoids unnecessary DMA traffic.

### Files and functions to change

| File | Function | Change |
|------|----------|--------|
| `client/include/DataInitialization.hpp` | `beginAsyncReset` | Replace no-op event record in the `m_altSlotsFilled` branch with `SlotGuard` + `resetOutput` (D2D, `m_copyStream`), guarded on `m_elementsToValidate` |
| `client/include/DataInitialization.hpp` | `beginAsyncReset` | `problem` parameter must be typed (`ContractionProblemGemm const&`) or the call must dynamic-cast, as `resetOutput` requires the concrete type |

No changes to `main.cpp`, `DataInitialization.cpp`, or the cold path are required.

---

## Proposal 2: Async copy only for no-benchmark runs

### Problem

The constructor assigns `m_numActiveBuffers` based on whether benchmark runs will execute:

```cpp
bool noBenchmarkRuns = (numBenchmarks == 0 || numEnqPerSync == 0 || numSyncsPerBench == 0);
m_numActiveBuffers = noBenchmarkRuns ? 3 : 2;
```

This is inverted. `beginAsyncReset` is called exclusively inside the `while(listeners.needMoreRunsInSolution())` loop body (main.cpp lines ~1521–1522), which never executes when `noBenchmarkRuns == true` because `m_numEnqueuesPerSolution = numEnqPerSync * numSyncsPerBench = 0` causes `needMoreRunsInSolution()` to return false immediately. The consequences:

- **No-benchmark runs** (validation-only, warmup-only): triple-buffering is allocated (`m_numActiveBuffers = 3`), but `beginAsyncReset` is never called, so the extra GPU buffer set for every tensor is permanently idle. This wastes device memory.
- **Benchmark runs**: double-buffering is used (`m_numActiveBuffers = 2`). `beginAsyncReset` is called, so the ring fires, but with only one alt slot the overlap depth is minimal. More critically, async DMA running during warmup can perturb cache state and DRAM bandwidth measurements even when not concurrent with the measured kernel.

### Design rationale: benchmarks must be clean

During benchmark runs, async DMA should be disabled entirely. `waitCopyDone` prevents the DMA from overlapping with the measured kernel launch, but the DMA traffic during warmup — which occurs in the same temporal window as kernel pre-scheduling — can:
- Evict A/B/C data from the GPU's L2 cache before the kernel reads it, biasing bandwidth measurements.
- Introduce DRAM bandwidth contention that is not reproducible in production workloads.

The benchmark path should match develop exactly: synchronous D-only `resetOutput` per solution (already handled by `prepareGPUInputsInternal`'s fast path), no background DMA. The ring is unnecessary for benchmark runs because the per-solution setup cost is dominated by kernel launch overhead and post-sync, not by the D reset.

### Proposed change

**For benchmark runs (`noBenchmarkRuns == false`):**
- Disable the ring. `ringEligible()` should add `!m_hasBenchmarkRuns` as a sixth term, or the `beginAsyncReset` call site in main.cpp should be gated on `noBenchmarkRuns`.
- `prepareGPUInputs` always takes the slow path through `prepareGPUInputsInternal`, which already handles the synchronous D reset via `resetOutput` when `m_elementsToValidate > 0`.
- `m_numActiveBuffers = 2`: double-buffer is allocated (consistent with current code) but the ring fast path never fires.

**For no-benchmark runs (`noBenchmarkRuns == true`):**
- Enable the ring with triple-buffering: `m_numActiveBuffers = 3` (one active, one DMA in-flight, one spare).
- Add a `beginAsyncReset(problem)` call in `main.cpp` after each solution's warmup block (after `postWarmup`, before `postSolution`) so the DMA for the next slot overlaps with correctness checking. With triple-buffering, two calls to `beginAsyncReset` can be made to fill both alt slots ahead.
- The existing `waitCopyDone` call (before kernel launch) already ensures the DMA is complete before the slot is consumed.

**The assignment becomes correct under this design:**

```cpp
m_numActiveBuffers = noBenchmarkRuns ? 3 : 2;
```

Triple-buffering is used when and only when the ring's async path fires (no-benchmark runs). Benchmark runs use double-buffering with the ring disabled.

### Interaction with `ringEligible()`

The cleanest implementation adds `m_hasBenchmarkRuns` as a member set in the constructor alongside `m_numActiveBuffers`:

```cpp
m_hasBenchmarkRuns = !noBenchmarkRuns;
m_numActiveBuffers = noBenchmarkRuns ? 3 : 2;
```

Then `ringEligible()` adds one term:

```cpp
bool ringEligible() const {
    return m_hasAltBuffers
        && m_copyStream
        && m_gpuInit
        && m_curBoundsCheck == BoundsCheckMode::Disable
        && !m_problemDependentData
        && !m_hasBenchmarkRuns;   // ring is for no-benchmark paths only
}
```

This gates `beginAsyncReset` at its entry point (the existing `if(!ringEligible()) return;` guard), so the main.cpp call sites at lines 1521–1522 do not need to be conditioned — they become no-ops when benchmark runs are active.

### New `beginAsyncReset` call site for no-benchmark runs

Currently `beginAsyncReset` is called only after `postSyncs` inside the benchmark loop. For no-benchmark runs, there is no benchmark loop. The DMA must be triggered from somewhere in the solution loop. The correct location is after the warmup block completes (after `postWarmup`, before `postSolution`):

```cpp
// After postWarmup, before postSolution:
if(noBenchmarkRuns) {
    ScopedTimer timer("async_reset_submit");
    dataInit->beginAsyncReset(problem);
    dataInit->beginAsyncReset(problem);  // fill both alt slots for triple-buffer
}
```

The `waitCopyDone` call already present before the first kernel launch ensures correctness on the next solution iteration.

### Files and functions to change

| File | Function / Location | Change |
|------|---------------------|--------|
| `client/src/DataInitialization.cpp` | Constructor (~line 991) | Add `m_hasBenchmarkRuns = !noBenchmarkRuns;` alongside existing `m_numActiveBuffers` assignment |
| `client/include/DataInitialization.hpp` | `ringEligible()` | Add `&& !m_hasBenchmarkRuns` |
| `client/include/DataInitialization.hpp` | Member declarations | Add `bool m_hasBenchmarkRuns = false;` |
| `client/main.cpp` | After `postWarmup` block | Add `beginAsyncReset` call pair, guarded on `noBenchmarkRuns` |

The two existing `beginAsyncReset` calls inside the benchmark loop (main.cpp lines 1521–1522) become permanently dead for benchmark runs once `ringEligible()` returns false for that path. They can be removed or left with a comment; removing them is cleaner.

---

## Combined effect

### Before (current branch state)

| Mode | `m_numActiveBuffers` | Ring fires? | D reset per solution |
|------|---------------------|-------------|---------------------|
| No-benchmark runs | 3 (triple) | No (`beginAsyncReset` never called) | Via `prepareGPUInputsInternal` fast path (synchronous) |
| Benchmark runs | 2 (double) | Yes (async DMA during warmup) | Via ring warm path: no-op (D not reset); or cold path: full `fillSlot` |

### After (both proposals applied)

| Mode | `m_numActiveBuffers` | Ring fires? | D reset per solution |
|------|---------------------|-------------|---------------------|
| No-benchmark runs | 3 (triple) | Yes — `beginAsyncReset` called after warmup, overlaps correctness checking | Async `hipMemcpyDeviceToDevice` on `m_copyStream` in warm path, complete before next kernel via `waitCopyDone` |
| Benchmark runs | 2 (double) | No — `ringEligible()` returns false; ring disabled | Synchronous `resetOutput` in `prepareGPUInputsInternal` fast path (develop parity) |

### Key invariants preserved after both proposals

- D in every slot consumed by the compute kernel has been reset from `gpuInput.valid` before the kernel runs, whether via sync or async DMA.
- A/B/C in alt slots are always valid for the current problem (filled by `initializeAltBufferSets`; invalidated by `cancelAsyncReset` on problem change).
- Benchmark measurements are free of DMA-induced cache and bandwidth contention.
- `m_numActiveBuffers = noBenchmarkRuns ? 3 : 2` is semantically correct: triple-buffering is allocated when and only when the async pipeline is active.
