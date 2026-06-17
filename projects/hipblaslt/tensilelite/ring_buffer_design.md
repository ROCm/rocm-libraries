# Ring Buffer Design for GPU Input Reset Overlap

This document describes the design introduced by the `users/alvasile/ring_buffer` branch.
It covers the motivation, the multi-buffer scheme, the data structures, the key operations,
the synchronization protocol, and naming observations to guide a later renaming pass.

---

## 1. Motivation

### The bottleneck before this change

The benchmark client loops over problems and solutions. For each iteration of the inner
`benchmark_runs` loop, the kernel receives a GPU buffer that was prepared by
`prepareGPUInputs`. After the kernel runs, output tensor D is overwritten (that is the
point of the benchmark). Before the next run, output D must be reset to its pristine
state so that the kernel result is valid and comparable.

Before this branch, that reset was synchronous and blocking:

1. Kernel launches and runs on the compute stream.
2. CPU blocks waiting for the kernel to finish (`hipStreamSynchronize`).
3. CPU calls `prepareGPUInputs` which calls `resetOutput` using a synchronous
   `hipMemcpy` (device-to-device).
4. Once the copy finishes, the next kernel can launch.

Steps 2–3 serialize the GPU. The copy engine is idle during step 2, and the compute
engine is idle during step 3. For large output tensors the copy can be a significant
fraction of total wall time.

### The secondary bug fixed by this branch

The test `BatchPointerReset_test.cpp` documents a separate bug: `initializeGPUBatchedInputs`
was called only inside the slow-path branch of `prepareGPUInputs` (the branch guarded by
`!m_gpuInit`). On subsequent calls for a different problem, the fast path (`m_gpuInit &&
BoundsCheckMode::Disable && !m_problemDependentData`) returned the cached inputs without
re-uploading the batch pointer arrays to the GPU. Because batch pointer arrays encode
per-problem strides, the GPU kernel received pointers computed from the previous problem,
causing silent incorrect results for batched GEMMs. The fix is to track a separate
`m_batchInit` flag (reset to `false` by `preProblem`) and always call
`initializeGPUBatchedInputs` when `!m_batchInit`, regardless of `m_gpuInit`.

---

## 2. High-Level Approach

### Double-buffer (benchmark runs present)

Two GPU buffer sets (slots 0 and 1) hold identical pristine data for the current problem.
While the compute stream is executing a kernel using slot 0, the copy engine is resetting
slot 1 on the dedicated `m_copyStream`. Before the next kernel launch, a stream-wait event
makes the compute stream wait for slot 1's copy to finish. The GPU never idles between the
copy and the kernel.

### Triple-buffer (no benchmark runs, validation only)

When there are no effective benchmark runs (i.e., `num-benchmarks=0`,
`num-enqueues-per-sync=0`, or `num-syncs-per-benchmark=0`), the CPU has a longer gap
between kernel launches, so an extra buffer slot (slot 2) is added. After each solution, `beginAsyncReset` is called twice to fill
both non-active slots ahead of time, giving two full solution-execution cycles of overlap.

### Ring advancement

`m_activeIdx` indexes which buffer slot is currently in use. After each benchmark
completes, `beginAsyncReset` submits a DMA to the next slot modulo `m_numActiveBuffers`.
`m_pendingResets` counts how many slots have been filled but not yet consumed.
`prepareGPUInputs` detects `m_pendingResets > 0` and calls `advanceBuffer()` to rotate
`m_activeIdx` forward, updating all working-state pointers to point at the new slot.

### Warm ring optimization

After `initializeAltBufferSets` fills all non-zero slots synchronously on the first call
for a new problem, `m_ringBufferWarm` is set to `true`. While warm, `beginAsyncReset`
skips the full DMA (since input tensors A, B, C are never modified by kernels and D is
completely overwritten by the next kernel run anyway). It simply records an event on
`m_copyStream` and increments `m_pendingResets`. This eliminates even the async DMA
overhead in the steady state.

---

## 3. Data Structures

### `MemoryInput` (extended)

Previously contained only `current`, `valid`, `bad`, `batch`. This branch adds:

```
std::shared_ptr<void>  buffers[MAX_BUFFER_SETS];   // one GPU allocation per ring slot
std::shared_ptr<void*> batchBufs[MAX_BUFFER_SETS]; // batch pointer array per ring slot
```

`current` and `batch` remain the "active window" aliases: they always point to
`buffers[m_activeIdx]` and `batchBufs[m_activeIdx]`. Existing code that reads
`current`/`batch` sees the currently-active ring slot without modification.

### Ring-level pointer caches in `DataInitialization`

```
std::vector<void*>             m_gpuPtrsRing[MAX_BUFFER_SETS];
std::vector<void**>            m_gpuBatchPtrsRing[MAX_BUFFER_SETS];
std::shared_ptr<ProblemInputs> m_cachedInputsRing[MAX_BUFFER_SETS];
```

These are the flat per-slot snapshots of `m_gpuPtrs`, `m_gpuBatchPtrs`, and
`m_cachedGPUInputs`. Slot 0 is populated at the end of `prepareGPUInputsInternal` (only
when `asyncStream == nullptr`). Slots 1 and 2 are populated by `initializeAltBufferSets`
calling `fillSlot`. `advanceBuffer` copies slot `m_activeIdx` into the working-state
fields `m_gpuPtrs`, `m_gpuBatchPtrs`, and `m_cachedGPUInputs`.

### Ring control state

| Field | Type | Meaning |
|---|---|---|
| `m_hasAltBuffers` | `bool` | `true` when alternate GPU allocations were successfully made |
| `m_numActiveBuffers` | `size_t` | 2 (double) or 3 (triple) depending on benchmark mode |
| `m_activeIdx` | `size_t` | Index of the slot the CPU and current kernel use |
| `m_pendingResets` | `size_t` | Count of slots that have been DMA'd but not yet consumed |
| `m_activeNeedsSync` | `bool` | `true` when `advanceBuffer` rotated to a new slot that has not yet been waited on by the compute stream |
| `m_ringBufferWarm` | `bool` | `true` when all non-zero slots have been filled for the current problem |

### Synchronization resources

```
hipStream_t m_copyStream;                          // dedicated DMA stream
hipEvent_t  m_copyDoneEvents[MAX_BUFFER_SETS];    // one event per slot
uint8_t**   m_pinnedBatchStaging;                  // reusable pinned host staging for batch pointer upload
```

`m_copyStream` is separate from the compute stream so DMA and compute can run
concurrently. `m_copyDoneEvents[i]` is recorded on `m_copyStream` after slot `i` is
filled. `m_pinnedBatchStaging` is a permanent allocation sized `m_maxBatch * sizeof(void*)`,
reused across all calls to `initializeGPUBatchedInputs`. Pinned memory is required for
async H2D copies; previously the code `malloc`'d a temporary buffer and used synchronous
`hipMemcpy`, which prevented overlap.

### `m_batchInit` flag

Added to `DataInitialization`. Reset to `false` by `preProblem` (which runs at problem
boundary). Checked by `prepareGPUInputsInternal`: when `!m_batchInit`, batch pointer
arrays are unconditionally re-uploaded, then `m_batchInit` is set to `true`. This fixes
the bug where the fast path skipped batch pointer re-upload on the second problem.

### `SlotGuard` RAII class

`SlotGuard(vdata, slotIdx)` swaps `gpuInput.current` and `gpuInput.batch` with
`buffers[slotIdx]` and `batchBufs[slotIdx]` for every `PristineUnit` in `m_vdata`. The
destructor swaps back. This allows `fillSlot` to invoke the existing
`copyInputs`/`resetOutput`/`initializeGPUBatchedInputs` code paths (which all write to
`gpuInput.current`) while targeting a non-active slot, without modifying those code
paths.

---

## 4. Key Operations

### `allocNewGPUInputs()`

**Role.** Extended to allocate `MAX_BUFFER_SETS` sets of GPU data buffers and batch
pointer arrays, and a pinned staging buffer for async batch pointer upload. If any
alternate allocation fails, `m_hasAltBuffers` is set to `false` and the ring buffer path
is disabled gracefully.

**Postcondition.** `gpuInput.buffers[0]` == `gpuInput.current`, all slots allocated or
`m_hasAltBuffers = false`.

### `prepareGPUInputsInternal(problem, asyncStream)`

**Role.** The unified slow-path initializer. Called from both the synchronous front-end
(`asyncStream = nullptr`) and from `fillSlot` (`asyncStream = m_copyStream`).

**Key behavior changes from pre-branch.**
- Batch init is now separated: `if(!m_batchInit) { initializeGPUBatchedInputs(...); m_batchInit = true; }` runs unconditionally before the fast/slow fork.
- After full initialization (slow path), if `asyncStream == nullptr`, slot 0 ring state
  is snapshotted and `initializeAltBufferSets` is called to fill the remaining slots.
- The `asyncStream` guard prevents double-snapshotting when called from `fillSlot`.

**Precondition.** `m_batchInit` reflects whether batch pointers are current.
**Postcondition.** `m_gpuPtrs`, `m_gpuBatchPtrs`, `m_cachedGPUInputs` reflect the fully
initialized slot 0. `m_gpuPtrsRing[0..m_numActiveBuffers-1]` are all populated.

### `fillSlot(slotIdx, problem, asyncStream)`

**Role.** Fills a single ring slot with pristine data, issuing all copies on `asyncStream`.
Called by `initializeAltBufferSets` (synchronously) and by `beginAsyncReset` (on
`m_copyStream`).

**Mechanism.** Constructs a `SlotGuard` to redirect `gpuInput.current/batch` to slot
`slotIdx`. Then either takes the fast path (slot already initialized, output-only reset via
`resetOutput`) or the full path (full `copyInputs` + `initializeGPUBatchedInputs`).
Builds and stores `m_cachedInputsRing[slotIdx]` from the result. On `SlotGuard`
destruction, `current/batch` are restored to slot 0.

**Precondition.** `m_gpuPtrsRing[slotIdx]` may be empty (first fill) or populated (re-fill).
**Postcondition.** `m_gpuPtrsRing[slotIdx]`, `m_gpuBatchPtrsRing[slotIdx]`, and
`m_cachedInputsRing[slotIdx]` are up to date.

### `initializeAltBufferSets(problem)`

**Role.** One-time synchronous fill of slots 1 (and 2 for triple) on the first call for a
given problem. Sets `m_ringBufferWarm = true`.

**Guard.** Returns immediately if `!m_hasAltBuffers` or if `m_gpuPtrsRing[1]` is already
populated (idempotent).

**Postcondition.** All non-zero slots populated, `m_ringBufferWarm = true`.

### `beginAsyncReset(problem)`

**Role.** Submits an async fill of the next unfilled slot on `m_copyStream`. Called twice
by `main.cpp` after each benchmark completes to pipeline the reset for the next two
iterations.

**Target slot calculation.** `targetIdx = (m_activeIdx + m_pendingResets + 1) % m_numActiveBuffers`.
This is the next slot after all currently pending ones.

**Warm path.** When `m_ringBufferWarm`, skips the DMA entirely: records a no-op event
and increments `m_pendingResets`. The event serves only as a synchronization marker for
`waitCopyDone`.

**Cold path.** Calls `fillSlot(targetIdx, problem, m_copyStream)`. Records
`m_copyDoneEvents[targetIdx]` on `m_copyStream` after the fill. Increments `m_pendingResets`.

**Guard.** Returns early if `m_pendingResets >= m_numActiveBuffers - 1` (all non-active
slots already have pending DMAs).

### `advanceBuffer()`

**Role.** Rotates `m_activeIdx` forward and updates all working-state pointers to the new
slot. Decrements `m_pendingResets` and sets `m_activeNeedsSync = true` to mark that the
compute stream must wait for the copy-done event before using this slot.

**Called by.** `prepareGPUInputs(ContractionProblem*)` when it detects `m_pendingResets > 0`.

### `waitCopyDone(computeStream)`

**Role.** Inserts a GPU-side dependency: makes `computeStream` wait for
`m_copyDoneEvents[m_activeIdx]` to be recorded, without blocking the CPU. Clears
`m_activeNeedsSync`. This is called in `main.cpp` before `warmup` and `benchmark_runs`
for each solution, ensuring the slot the kernel is about to use has been fully populated.

**Precondition.** `m_activeNeedsSync` is `true` (i.e., `advanceBuffer` was called since
the last `waitCopyDone`).
**Postcondition.** The compute stream will not start reading from the active slot until
the `m_copyStream` DMA that preceded the event record is complete.

### `cancelAsyncReset()`

**Role.** Called at problem boundaries (from `main.cpp` after `preProblem`). Synchronizes
`m_copyStream` to drain any in-flight DMA, resets the ring to slot 0, clears all alt slot
snapshots so `initializeAltBufferSets` will re-run for the new problem's data, and clears
`m_ringBufferWarm`.

**Postcondition.** `m_activeIdx = 0`, `m_pendingResets = 0`, `m_activeNeedsSync = false`,
`m_ringBufferWarm = false`, `m_gpuPtrsRing[1..2]` empty.

### `syncCopyStream()`

**Role.** CPU-blocking synchronization of `m_copyStream`. Called by `cancelAsyncReset`
when there are pending resets or an unsynced active buffer, to guarantee the copy stream
is quiescent before the problem changes.

---

## 5. Synchronization Protocol

### Stream topology

```
m_copyStream  ──[DMA slot N]──[event record copyDoneEvents[N]]────────────────────────────
                                                                          \
computeStream ──────────────────────────────────[hipStreamWaitEvent]──[kernel slot N]───
```

The CPU never blocks between `beginAsyncReset` and `waitCopyDone`. The synchronization
point is entirely on the GPU via `hipStreamWaitEvent`.

### Per-problem sequence (benchmark mode)

```
main.cpp:
  preProblem()                          // m_batchInit = false
  cancelAsyncReset()                    // drain copy stream, reset to slot 0
  prepareGPUInputs()                    // slow path: fills slot 0 synchronously,
                                        // calls initializeAltBufferSets (fills slot 1)
                                        // m_ringBufferWarm = true
  for each solution:
    prepareGPUInputs()                  // m_pendingResets > 0? -> advanceBuffer()
                                        // m_activeNeedsSync = true
    waitCopyDone(computeStream)         // GPU: computeStream waits for copyDoneEvents[activeIdx]
                                        // CPU: does not block
    warmup_runs (uses active slot)
    benchmark_runs (uses active slot)
    beginAsyncReset(problem)            // enqueue DMA to next slot on copyStream
    beginAsyncReset(problem)            // (double-call for double-buffering; second
                                        // call is a no-op if pendingResets >= numActiveBuffers-1)
```

### What each guarantee provides

| Step | Guarantee |
|---|---|
| `initializeAltBufferSets` | All non-zero slots contain valid pristine data before any async use |
| `hipEventRecord(copyDoneEvents[N], m_copyStream)` | The event captures the tail of all `hipMemcpyAsync` calls that filled slot N |
| `hipStreamWaitEvent(computeStream, copyDoneEvents[N], 0)` | The compute stream will not issue any command past this point until the event is recorded |
| `cancelAsyncReset` / `syncCopyStream` | CPU-visible guarantee: copy stream is idle before problem state changes |

### When the CPU blocks vs. runs freely

The CPU blocks only at `cancelAsyncReset` (at problem boundaries) and in the destructor.
During steady-state iteration over solutions, the CPU never blocks on the copy stream.
The GPU ensures ordering via stream-wait events.

---

## 6. Naming Observations

The following names are ambiguous or misleading and should be renamed.

### `asyncStream` parameter on `copyInputs`, `resetOutput`, and `initializeGPUBatchedInputs`

**Current name.** `hipStream_t asyncStream = nullptr`

**Actual meaning.** A stream selector. When non-null, DMA is submitted on this stream and
the caller is responsible for synchronization. When null, DMA is submitted on `m_copyStream`
and the function synchronizes it before returning. The word "async" belongs to the
operation, not the parameter — the parameter's presence/absence is what determines
synchronous vs. asynchronous behavior from the caller's perspective.

In `resetOutput` this is explicit:
```cpp
hipStream_t copyStream = asyncStream ? asyncStream : m_copyStream;
bool useAsync = (kind == hipMemcpyDeviceToDevice) && copyStream;
```

**Suggested name.** `targetStream`.

### `asyncStream` parameter on `copyValidToGPUBuffer`

**Current name.** `hipStream_t asyncStream = nullptr`

**Actual meaning.** Despite having the same signature as the above, this parameter does
**not** select the DMA stream. `copyValidToGPUBuffer` always submits copies on
`m_copyStream` regardless of what is passed. The parameter's only role is to suppress
the trailing `hipStreamSynchronize(m_copyStream)`:
```cpp
if(m_copyStream && !asyncStream)
    HIP_CHECK_EXC(hipStreamSynchronize(m_copyStream));
```
A non-null value signals "caller owns the sync for `m_copyStream`; don't sync here."
This is a different contract from the other three functions — passing a non-null stream
does not change where copies are submitted.

**Suggested name.** `callerOwnsCopySync` (bool would be more honest than a stream
pointer), or fix the function to actually dispatch on the supplied stream (making it
consistent with `copyInputs`/`resetOutput`).

### `m_activeNeedsSync`

**Current name.** `m_activeNeedsSync`

**Actual meaning.** "The active buffer was just advanced (`advanceBuffer` was called) and
the compute stream has not yet been told to wait for the copy-done event." This is reset
by `waitCopyDone` after inserting the stream-wait event. The name sounds like a CPU sync
is needed; it is actually a pending GPU-side dependency.

**Suggested name.** `m_computeNeedsCopyBarrier`.

### `m_pendingResets`

**Current name.** `m_pendingResets`

**Actual meaning.** The count of ring slots that have been filled (or had their copy-done
event recorded) but have not yet been consumed by `advanceBuffer`. "Reset" is a legacy
term from when this operation was always a full D-reset. In the warm path it does not
reset anything — it just records a no-op event. The field really counts how many
pre-filled slots are available for the next `advanceBuffer` call.

**Suggested name.** `m_availableSlots`.

### `m_ringBufferWarm`

**Current name.** `m_ringBufferWarm`

**Actual meaning.** All non-zero ring slots have been filled for the current problem and
are ready for the fast-path in `beginAsyncReset`. "Warm" is a reasonable informal term
but its exact invariant ("all alt slots populated for the current problem") is not
obvious.

**Suggested name.** `m_altSlotsReady`. The flag should be cleared on problem change
(which `cancelAsyncReset` does).

### `fillSlot` / `initializeAltBufferSets`

These names are clear. No rename needed.

### `SlotGuard`

This name accurately describes the RAII pattern (scoped redirect of active pointers to a
slot). No rename needed.

### `beginAsyncReset` double-call in `main.cpp`

In `main.cpp`, `beginAsyncReset` is called twice in sequence. The comment explains this
is to fill two slots (for triple-buffering), but `beginAsyncReset` already guards against
overfilling (`m_pendingResets >= m_numActiveBuffers - 1`). The second call is a no-op in
double-buffer mode. This is correct behavior but not obvious from reading the call site.
A comment at the call site should explain that the second call is the triple-buffer case
no-op.

---

## Appendix: MAX_BUFFER_SETS vs. m_numActiveBuffers

`MAX_BUFFER_SETS = 3` is the compile-time upper bound on the ring size. Arrays of that
size are always allocated. `m_numActiveBuffers` is the runtime ring size (2 or 3), set
in the constructor based on whether benchmark runs are present. Only slots `0 ..
m_numActiveBuffers - 1` are ever filled and used; slot 2 goes unused in double-buffer
mode but its memory is still allocated.
