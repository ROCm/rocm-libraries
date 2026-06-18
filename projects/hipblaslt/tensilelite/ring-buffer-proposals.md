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

## Proposal 1: Async output reset on the warm path

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

The comment justifying this says the slot is kernel-write-free and that D's initialized value is irrelevant because the next kernel fully overwrites it. That is too strong. It is only true before an alt slot's first use. Once `prepareGPUInputs` calls `advanceBuffer`, that alt slot becomes the active slot, the kernel writes its output tensors, and a later `beginAsyncReset` may target that previously active slot while `m_altSlotsFilled` is still true.

The safer invariant is narrower:

- Under `ringEligible()`, non-output tensors are stable across solutions (`!m_problemDependentData`) and kernels should not write A/B/C or other input-only tensors.
- Output tensors may be dirty once their slot has been used by a kernel.
- When validation is enabled, the ring path should preserve the same "reset outputs before reuse" contract as `prepareGPUInputsInternal`.

This is not strictly a current correctness failure for ordinary GEMM benchmarking, because the standard GEMM kernel overwrites D. It is still a real parity and maintenance problem: develop's fast path resets output tensors when `m_elementsToValidate > 0`, while the ring fast path bypasses that reset entirely. Any output mode that depends on the pre-kernel value, any partial-write/masked-output evolution, or any future validation of an output that is not unconditionally overwritten would silently diverge on the warm ring path.

Also, this should not be called "D-only." `resetOutput` resets every tensor whose descriptor has `isOutput()`: D is always an output, but E, gradient bias, and AMAXD can also be outputs. The warm-path fix should therefore reset output tensors, not just D.

### Proposed change

When the ring warm path is used and validation is enabled, reset output tensors in the target slot on `m_copyStream` before recording the copy-done event. When validation is disabled, keep the no-op event path to avoid benchmark-only DMA traffic.

```cpp
if(m_altSlotsFilled)
{
    if(m_elementsToValidate)
    {
        auto resetWarmOutputs = [&](ContractionProblemGemm const& gemm) {
            // Redirect gpuInput.current/batch to targetIdx for resetOutput.
            SlotGuard guard(m_vdata, targetIdx);
            resetOutput(m_gpuPtrsRing[targetIdx],
                        m_gpuBatchPtrsRing[targetIdx],
                        m_maxElements,
                        m_groupedOffsets,
                        gemm,
                        hipMemcpyDeviceToDevice,
                        m_copyStream);
        };

        if(auto gemmProblem = dynamic_cast<ContractionProblemGemm const*>(problem))
        {
            resetWarmOutputs(*gemmProblem);
        }
        else if(auto groupedProblem
                = dynamic_cast<ContractionProblemGroupedGemm const*>(problem))
        {
            assert(std::all_of(
                groupedProblem->gemms.begin(),
                groupedProblem->gemms.end(),
                [&](ContractionProblemGemm const& g) {
                    return g.a().dataType() == groupedProblem->gemms[0].a().dataType()
                        && g.b().dataType() == groupedProblem->gemms[0].b().dataType()
                        && g.c().dataType() == groupedProblem->gemms[0].c().dataType()
                        && g.d().dataType() == groupedProblem->gemms[0].d().dataType();
                }));
            resetWarmOutputs(groupedProblem->gemms[0]);
        }
        else
        {
            throw std::runtime_error("Failed to cast to any ContractionProblem.");
        }
    }

    HIP_CHECK_EXC(hipEventRecord(m_copyDoneEvents[targetIdx], m_copyStream));
    m_availableSlots++;
    return;
}
```

The `SlotGuard` RAII type already exists in `fillSlot` to redirect `gpuInput.current`/`batch` to the target slot for the duration of the call. The same guard is required here because `resetOutput` writes through `p.gpuInput.current`; the `ptrs` vector is updated after the copy, but it is not the source of the destination pointer. Without `SlotGuard`, the reset would write into the active slot, not `targetIdx`.

`resetOutput` is async-capable for D2D output resets: it takes a `targetStream` parameter and uses `hipMemcpyAsync` when `kind == hipMemcpyDeviceToDevice` and a stream is supplied. Recording `m_copyDoneEvents[targetIdx]` after `resetOutput` on the same stream makes the existing `waitCopyDone` barrier sufficient before the slot is consumed by the compute stream.

### Required eligibility guard

The warm-path implementation above uses `hipMemcpyDeviceToDevice`, so it is only correct when the pristine GPU copy is populated. Today `ringEligible()` does not require this:

```cpp
bool ringEligible() const {
    return m_hasAltBuffers
        && m_copyStream
        && m_gpuInit
        && m_curBoundsCheck == BoundsCheckMode::Disable
        && !m_problemDependentData;
}
```

If `--pristine-on-gpu=false`, `prepareGPUInputsInternal` uses host-to-device copies for output reset. In that mode, a D2D warm reset would read from `gpuInput.valid`, which is allocated but not guaranteed to contain the pristine output data. Proposal 1 must therefore also add `m_keepPristineCopyOnGPU` to `ringEligible()`:

```cpp
bool ringEligible() const {
    return m_hasAltBuffers
        && m_copyStream
        && m_gpuInit
        && m_curBoundsCheck == BoundsCheckMode::Disable
        && !m_problemDependentData
        && m_keepPristineCopyOnGPU;
}
```

An alternative is to teach `resetOutput` to perform stream-aware H2D resets and mirror `prepareGPUInputsInternal`'s `kind` selection in the warm path. That is a larger change. The smaller and safer Proposal 1 fix is to keep the ring warm reset D2D-only and disable the ring when no pristine GPU copy exists.

### Why non-output tensors are safe to skip

Non-output tensors in an alt slot are safe to skip on the warm path because:

1. `initializeAltBufferSets` fills the alt slots for the current problem by calling `fillSlot`.
2. `ringEligible()` excludes problem-dependent data, so tensor values do not change between solutions for the same problem.
3. Kernels consume A/B/C and other input-only tensors but should not write them.
4. `cancelAsyncReset` is called at the problem boundary, synchronizes any pending copy work, clears alt ring metadata, and clears `m_altSlotsFilled`, so stale alt slots are not reused for a different problem.

Output tensors do not share that invariant. A slot that has been active may contain previous kernel output and must be reset when validation is enabled.

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

The warm-path fix should mirror this condition. When `m_elementsToValidate == 0`, the no-op event path is correct and avoids unnecessary DMA traffic in pure benchmark runs.

One caveat: `m_altSlotsFilled` is set only by `initializeAltBufferSets`, not by the cold path in `beginAsyncReset`. Proposal 1 should not assume that a cold fill transitions the ring into the warm state. It only changes the behavior once the existing warm state is already true.

### Files and functions to change

| File | Function | Change |
|------|----------|--------|
| `client/include/DataInitialization.hpp` | `beginAsyncReset` | In the `m_altSlotsFilled` branch, when `m_elementsToValidate > 0`, use `SlotGuard` + `resetOutput` to reset output tensors in `targetIdx` on `m_copyStream`, then record the copy-done event |
| `client/include/DataInitialization.hpp` | `beginAsyncReset` | Preserve the existing `ContractionProblem const*` API and mirror the cold path's GEMM/grouped-GEMM casts; do not make `beginAsyncReset` GEMM-only |
| `client/include/DataInitialization.hpp` | `ringEligible()` | Add `&& m_keepPristineCopyOnGPU` so the warm D2D reset never reads from an unpopulated GPU pristine buffer |

No changes to `main.cpp`, `DataInitialization.cpp`, or the cold path are required for this smaller fix.

### Tests to add

- A ring warm-path validation test that advances into a slot, lets a kernel or test hook dirty its output buffer, calls `beginAsyncReset`, advances into that same slot again, and verifies the output tensor was reset before reuse.
- A `--pristine-on-gpu=false` coverage case proving the ring does not become eligible, so validation falls back to the synchronous `prepareGPUInputsInternal` reset path.

---

## Proposal 2: Make ring usage explicit and allocation-aware

### Problem

The constructor currently assigns ring depth from whether the benchmark timer has timed enqueue work:

```cpp
bool noBenchmarkRuns = (numBenchmarks == 0 || numEnqPerSync == 0 || numSyncsPerBench == 0);
m_numActiveBuffers = noBenchmarkRuns ? 3 : 2;
```

The old proposal treated `noBenchmarkRuns` as equivalent to "the solution loop never executes." That is not true. The outer and inner loops are driven by the union of all listeners, not only by `BenchmarkTimer`:

- `BenchmarkTimer::needMoreRunsInSolution()` returns false when `numEnqPerSync * numSyncsPerBench == 0`.
- `ReferenceValidator::needMoreRunsInSolution()` returns true until validation has run, even when `num-benchmarks=0`.
- `MetaRunListener::needMoreRunsInSolution()` ORs those listener results.

Therefore validation-only runs can already enter the solution loop and can already reach the existing `beginAsyncReset` calls after the `benchmark_runs` block. "Warmup-only" with validation disabled and zero benchmark enqueues is not a real current execution mode: warmups live inside the same solution loop, so no listener drives the loop and warmups do not run.

There is a separate allocation bug: `m_numActiveBuffers` controls ring indexing, but it does not control physical allocation. `allocNewGPUInputs()` allocates alternate buffers for `slot < MAX_BUFFER_SETS`, not `slot < m_numActiveBuffers`, so benchmark mode still physically allocates slot 2 even when `m_numActiveBuffers == 2`.

Finally, disabling only `ringEligible()` is not enough to match develop. `initializeAltBufferSets()` is called after the initial GPU input preparation and fills alt slots according to `m_numActiveBuffers`; that path is not gated by `ringEligible()`.

### Design rationale

Separate three concepts that are currently conflated:

1. **Timed benchmark enqueues**: `numBenchmarks > 0 && numEnqPerSync > 0 && numSyncsPerBench > 0`.
2. **Listener-driven untimed solution runs**: validation/printing can drive the solution loop even with no timed benchmark enqueues.
3. **Physical ring allocation**: whether extra GPU buffer sets should be allocated and filled at all.

This proposal chooses benchmark cleanliness and develop parity for timed benchmark enqueues: no ring consumption, no background reset submission, and no physical alt-buffer allocation for that path. Validation-driven untimed runs may use the ring, because they are not producing timed performance numbers and can benefit from preparing the next slot while the CPU moves through reporting and next-solution setup.

### Proposed change

Introduce explicit policy state based on timed benchmark enqueues, not the broader phrase "benchmark runs":

```cpp
bool hasTimedBenchmarkEnqueues = numBenchmarks > 0
                              && numEnqPerSync > 0
                              && numSyncsPerBench > 0;
bool hasValidationOrPrintWork = referenceValidatorWouldRun(args);

m_ringAllowed = !hasTimedBenchmarkEnqueues && hasValidationOrPrintWork;
m_numActiveBuffers = m_ringAllowed ? 3 : 1;
```

`referenceValidatorWouldRun(args)` should be a shared helper or an exact duplicate of `ReferenceValidator`'s enable predicate (`m_elementsToValidate != 0 || m_printAny`). Do not hand-maintain a similar-but-different list of print flags in `DataInitialization`; otherwise the ring can be enabled for a mode that does not actually drive the solution loop. `m_ringAllowed` should be a member because the same policy is needed by allocation, initialization, and `ringEligible()`. `m_numActiveBuffers = 1` means "no ring"; slot 0 is still used normally and no alt slot is addressed.

Proposal 1's `m_keepPristineCopyOnGPU` guard remains required. The ring warm path performs D2D output resets and must not read from an unpopulated GPU pristine buffer.

### Physical allocation and fill policy

Change alt-buffer allocation to respect `m_numActiveBuffers`:

```cpp
m_hasAltBuffers = (m_numActiveBuffers > 1);

for(size_t slot = 1; m_hasAltBuffers && slot < m_numActiveBuffers; slot++)
{
    ...
}
```

This fixes the current mismatch where benchmark mode indexes only slots 0 and 1 but still allocates slot 2 because the loop uses `MAX_BUFFER_SETS`.

`initializeAltBufferSets()` already loops to `m_numActiveBuffers`, and its existing `!m_hasAltBuffers` guard becomes meaningful once allocation initializes `m_hasAltBuffers = false` for `m_numActiveBuffers == 1`.

### Interaction with `ringEligible()`

`ringEligible()` should use the explicit ring policy, not infer policy from buffer presence alone:

```cpp
bool ringEligible() const {
    return m_hasAltBuffers
        && m_copyStream
        && m_gpuInit
        && m_curBoundsCheck == BoundsCheckMode::Disable
        && !m_problemDependentData
        && m_keepPristineCopyOnGPU
        && m_ringAllowed;
}
```

This makes the existing `beginAsyncReset` call pair harmless in timed benchmark mode: it becomes a no-op. If the goal is to avoid even the call overhead and timing label, the call site can also be guarded in `main.cpp`, but correctness should not depend on that.

### Call-site policy

Do not add the old proposed call after `postWarmup`. In the current code, validation happens in `listeners.validateWarmups()` before `postWarmup`, so a reset submitted after `postWarmup` does not overlap correctness checking.

The minimal implementation should keep the existing call pair after the `benchmark_runs` block. Validation-only runs can already reach that location when `ReferenceValidator` drives the solution loop. With `m_ringAllowed == true` and `m_numActiveBuffers == 3`, both calls can be useful: the first prepares one non-active slot and the second prepares the other.

If a later change wants to overlap reset DMA with validation readback or CPU comparison, the candidate location is between the first warmup launch and `listeners.validateWarmups()`, guarded by `m_ringAllowed`. That is a more aggressive scheduling change and should be tested separately for copy-engine contention with validation readback.

### Files and functions to change

| File | Function / Location | Change |
|------|---------------------|--------|
| `client/src/DataInitialization.cpp` | Constructor (~line 991) | Compute `hasTimedBenchmarkEnqueues`, use a shared/helper predicate matching `ReferenceValidator` enablement to determine whether validation/printing can drive untimed solution runs, set `m_ringAllowed`, and set `m_numActiveBuffers = m_ringAllowed ? 3 : 1` |
| `client/src/DataInitialization.cpp` | `allocNewGPUInputs` | Initialize `m_hasAltBuffers = (m_numActiveBuffers > 1)` and allocate alternate buffers only for `slot < m_numActiveBuffers` |
| `client/include/DataInitialization.hpp` | Member declarations | Add `bool m_ringAllowed = false;` |
| `client/include/DataInitialization.hpp` | `ringEligible()` | Add `&& m_ringAllowed`; keep Proposal 1's `&& m_keepPristineCopyOnGPU` |
| `client/main.cpp` | Existing `beginAsyncReset` call pair | Leave in place or guard for timing cleanliness; do not add the old post-`postWarmup` call |

### Tests to add

- Timed benchmark configuration (`num-benchmarks > 0`, `num-enqueues-per-sync > 0`, `num-syncs-per-benchmark > 0`) should not allocate alt slots and should not advance the ring.
- Validation-only configuration (`num-benchmarks=0`, `num-elements-to-validate > 0`) should use three active ring slots and the existing `beginAsyncReset` call pair should make subsequent `prepareGPUInputs` calls advance slots.
- Zero-enqueue / no-validation configuration should not claim warmup-only ring behavior; the solution loop should not run unless another listener drives it.
- Allocation test: slot 2 should not be allocated when `m_numActiveBuffers == 1`.

---

## Combined effect

### Before (current branch state)

| Mode | `m_numActiveBuffers` | Physical alt allocation | Ring fires? | Output reset per solution |
|------|---------------------|-------------------------|-------------|---------------------------|
| Timed benchmark enqueues | 2 | Slots 1 and 2 are allocated (`MAX_BUFFER_SETS`) even though only slot 1 is indexed | Yes, via existing `beginAsyncReset` calls | Warm path records no-op event; cold path full-fills |
| Validation-only / print-only, no timed enqueues | 3 | Slots 1 and 2 allocated | Yes when validation/printing drives the solution loop | Warm path records no-op event today; Proposal 1 fixes this |
| Zero enqueues and no validation/printing | 3 | Slots 1 and 2 allocated | No solution loop, so no useful ring activity | Initial preparation only |

### After (both proposals applied)

| Mode | `m_numActiveBuffers` | Physical alt allocation | Ring fires? | Output reset per solution |
|------|---------------------|-------------------------|-------------|---------------------------|
| Timed benchmark enqueues | 1 | No alt slots | No — `ringEligible()` returns false | Synchronous `resetOutput` in `prepareGPUInputsInternal` fast path when validation is enabled |
| Validation-only / print-only, no timed enqueues | 3 | Slots 1 and 2 allocated | Yes, via the existing post-`benchmark_runs` call pair | Proposal 1 async output reset on `m_copyStream`, fenced by `waitCopyDone` |
| Zero enqueues and no validation/printing | 1 | No alt slots | No solution loop and no ring | Initial preparation only |

### Key invariants preserved after both proposals

- Output tensors in every consumed slot are reset before validation-sensitive reuse, whether by the synchronous fast path or Proposal 1's async warm path.
- Non-output tensors in ring slots are valid because either `initializeAltBufferSets` filled the slots or the cold `beginAsyncReset` / `fillSlot` path filled them after a problem-boundary invalidation.
- Timed benchmark measurements do not consume ring slots or submit background reset work.
- Physical allocation matches policy: no alt slots are allocated when the ring is disabled.
