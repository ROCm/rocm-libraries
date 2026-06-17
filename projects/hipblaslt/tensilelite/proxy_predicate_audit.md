# Proxy Predicate Audit

## Candidate 1: `m_batchInit`

### Evidence

**Where it is set `true`:**

`DataInitialization.cpp:3501-3502` — inside `prepareGPUInputsInternal`:
```cpp
if(!m_batchInit)
{
    initializeGPUBatchedInputs(problem, targetStream);
    m_batchInit = true;
}
```
The flag is set after uploading batch pointer arrays to the GPU for the active problem.

**Where it is read (the decision site):**

Same function (`prepareGPUInputsInternal`), line `3497`:
```cpp
if(!m_batchInit)
{
    initializeGPUBatchedInputs(problem, targetStream);
    m_batchInit = true;
}
```
When `m_batchInit == true`, `initializeGPUBatchedInputs` is skipped entirely.

**Where it is reset `false`:**

`DataInitialization.hpp:895-901` — inside `preProblem`:
```cpp
virtual void preProblem(ContractionProblem* const problem) override
{
    m_currentGemmProblem = dynamic_cast<ContractionProblemGemm const*>(problem);
    m_currentSolution = nullptr;
    m_batchInit       = false;
}
```
Reset unconditionally when the problem changes.

**The ring path — is there a second consumer?**

`DataInitialization.hpp:275-288` — `prepareGPUInputs(ContractionProblem*)` fast path:
```cpp
if(m_availableSlots > 0)
{
    assert(ringEligible());
    advanceBuffer();
    return m_cachedGPUInputs;
}
```
When the ring fast path fires, `prepareGPUInputsInternal` is bypassed entirely. The `m_batchInit` check inside `prepareGPUInputsInternal` is therefore never reached. The ring slots themselves are filled by `fillSlot`, which calls `initializeGPUBatchedInputs` directly at `DataInitialization.cpp:3658`:
```cpp
initializeGPUBatchedInputs(problem, targetStream);
```
`fillSlot` does not set `m_batchInit`, nor does it read it. The two paths (ring, non-ring) are thus independent on the batch-init question.

**Failure scenario analysis:**

Within a single problem, the sequence is:
1. `preProblem` clears `m_batchInit = false`.
2. `prepareGPUInputs` is called; either the ring path or the non-ring path runs.
3. If non-ring: `prepareGPUInputsInternal` runs `initializeGPUBatchedInputs` on first call, sets `m_batchInit = true`, skips on subsequent calls within the same problem (correct).
4. If ring: `fillSlot` runs `initializeGPUBatchedInputs` for alt slots during `initializeAltBufferSets`. Slot 0 is filled by the initial `prepareGPUInputsInternal` call where `m_batchInit` is also set. So by the time the ring fast path fires, batch pointers are already current for the active problem.

The `preProblem` reset does cover problem changes on the non-ring path. On the ring path, `cancelAsyncReset` is called before `preProblem` fires (main.cpp:1240), and the ring is invalidated before any new problem's data is uploaded — so the stale-problem risk is neutralised by construction.

The `BatchPointerReset_test` regression was about `m_batchInit` surviving a problem change (it was reset only in `cancelAsyncReset`, not in `preProblem`). That regression is now fixed: `preProblem` at line 900 unconditionally sets `m_batchInit = false`.

**Is there a path where `m_batchInit == true` but batch pointers are not current for the active problem?**

With the current code: No. `preProblem` is called before every new problem (main.cpp:1234-1236), which resets the flag. The non-ring path then re-runs `initializeGPUBatchedInputs` on the first `prepareGPUInputs` call after the reset. The ring path re-fills all slots via `initializeAltBufferSets` at the start of each problem (called from `prepareGPUInputsInternal` when `targetStream == nullptr`, which is only the initial non-ring call — same first call).

### Is it a genuine Proxy Predicate flaw?

**Partial — the historical flaw is fixed; a residual naming fragility remains.**

The bool still means "batch pointers uploaded for _some_ problem" at its write site, but the `preProblem` reset now ensures that the predicate cannot be `true` when it should be `false` for the active problem. The flaw is structurally present (the flag encodes a narrow fact — "the function ran" — rather than the broader fact "batch pointers are current for the active problem"), but the missing writer (the `preProblem` reset) was added as the fix to the regression.

The residual fragility: if a future caller invokes `prepareGPUInputsInternal` (e.g. via `fillSlot` for a different problem without going through `preProblem`), `m_batchInit` would be stale. `fillSlot` itself does not read `m_batchInit` at all — it calls `initializeGPUBatchedInputs` unconditionally on the full path — so this is not a current bug, but the flag's meaning is still not encoded in a way the compiler can verify.

### Severity (as-is)

Unreachable today but fragile. The regression was real; the fix is correct but relies on call-site discipline (`preProblem` must always be called before a problem change). No current execution path reaches the stale-flag state.

---

## Candidate 2: `m_altSlotsReady` warm-path in `beginAsyncReset`

### Evidence

**The warm-path decision site:**

`DataInitialization.hpp:374-380`:
```cpp
// Warm path: A/B/C are read-only and D is fully overwritten by the next
// kernel (D = alpha*A*B + beta*C), so the existing slot data is still
// valid.  Skip the DMA; just record a no-op event as a sync marker.
if(m_altSlotsReady)
{
    HIP_CHECK_EXC(hipEventRecord(m_copyDoneEvents[targetIdx], m_copyStream));
    m_availableSlots++;
    return;
}
```
The claim is "D is fully overwritten." When this is true, the previous run's D contents do not matter, so the existing slot can be reused without re-DMA-ing D's initial value.

**Who sets `m_altSlotsReady = true`:**

`DataInitialization.cpp:3681`:
```cpp
void DataInitialization::initializeAltBufferSets(ContractionProblemGemm const& problem)
{
    if(!m_hasAltBuffers || !m_gpuPtrsRing[1].empty())
        return;
    for(size_t slot = 1; slot < m_numActiveBuffers; slot++)
        fillSlot(slot, problem, /*targetStream=*/nullptr);
    m_altSlotsReady = true;
}
```
`m_altSlotsReady` is set after filling alt buffer slots, unconditionally — no check for beta, in-place, or epilogue mode.

**Who sets `m_altSlotsReady = false`:**

`DataInitialization.hpp:338`:
```cpp
m_altSlotsReady = false;
```
Only inside `cancelAsyncReset`, which is called on problem change.

**The premise check — where is "D fully overwritten" verified?**

`ringEligible()` at `DataInitialization.hpp:1177-1184`:
```cpp
bool ringEligible() const
{
    return m_hasAltBuffers
        && m_copyStream
        && m_gpuInit
        && m_curBoundsCheck == BoundsCheckMode::Disable
        && !m_problemDependentData;
}
```
`beginAsyncReset` checks `ringEligible()` before checking `m_altSlotsReady`:
```cpp
void beginAsyncReset(ContractionProblem const* problem)
{
    if(!ringEligible())
        return;
    // ...
    if(m_altSlotsReady)
    { ... }
}
```

**Does `ringEligible` gate out beta != 0 / in-place D=C?**

No. `ringEligible()` does not check:
- `beta` value (0 vs nonzero)
- `m_cEqualsD` (in-place D=C mode)
- any epilogue mode or activation type

The premise "D is fully overwritten" is an assertion about the GEMM computation, not about initialization. A standard GEMM with beta=1 computes `D = alpha*A*B + beta*C`, which reads from both A, B, and C and writes to D. In that case D _is_ fully overwritten (the write covers every output element regardless of beta), so the warm-path claim is actually correct for standard GEMM — the concern would only arise if D=C (in-place), where the "old D" is also read as C.

**The in-place D=C case:**

When `m_cEqualsD == true`, the C pointer equals the D pointer. After a kernel run, D has been modified. If the warm-path skips the DMA and reuses the slot, the "C" input for the next kernel run is the _output_ of the previous run, not the initialized C value.

Does `ringEligible` exclude `m_cEqualsD`? No. `m_cEqualsD` is not part of `ringEligible`.

However: look at what the warm path actually skips. `beginAsyncReset` warm path does nothing GPU-side — it only records an event. The _next_ kernel run then reads from the slot's `current` buffer, which still holds the initialized A/B/C/D values from `initializeAltBufferSets` — because the _previous_ kernel ran on the _active_ slot (slot 0 or whichever `m_activeIdx` points to), not on the alt slot. The alt slot's C buffer was never overwritten by the kernel.

But wait: is D in the alt slot ever overwritten by a kernel run? Only if the kernel is launched against that slot. The ring works as: after a kernel run on slot `m_activeIdx`, `beginAsyncReset` prepares the _next_ alt slot. The kernel does not run on the alt slot — it only runs on the active slot. Therefore the alt slot's C/D buffers remain as initialized. The warm-path claim "D is fully overwritten by the _next_ kernel" refers to the upcoming launch that will _consume_ the slot, not a prior launch that wrote to it.

The warm path does not skip D's initialization — it asserts that D's prior-initialized value doesn't matter because the next kernel fully overwrites it regardless of what's there. This is true for GEMM (D is an output, every element written). The beta=1 case reads C and writes D, but does not read D before writing, so D's prior value is irrelevant.

**Is there any mode where D is _read_ before being written by the kernel?**

Standard GEMM: No. `D = alpha*A*B + beta*C` — C is read, D is written. Even with in-place (C==D), the compute reads C, writes D. If C==D in memory, the kernel reads a D element and writes D — this is an _in-place_ operation where each output element depends on the input at the same location. After the kernel, D holds the result. The _next_ kernel run on this slot reads C (the slot's C pointer). If C==D in memory for this slot, it would read the _output_ of the previous run that used this slot — but that previous run used a _different_ slot (the slot that was active at that time), not this alt slot. The alt slot's C buffer was only written by `initializeAltBufferSets` (or `fillSlot`), never by a kernel run.

**Conclusion on the beta/in-place concern:**

The warm path is safe for the current code because:
1. Kernels only run on the active slot, not the alt slots.
2. Alt slots are only written by `fillSlot`, which properly initializes all tensors.
3. The warm path simply re-issues the already-filled alt slot without re-DMA-ing. Since no kernel has written to it, A/B/C/D contents are exactly what was put there during `initializeAltBufferSets`.

The comment at the warm-path site ("D is fully overwritten by the next kernel") is technically correct but potentially misleading: D's contents in the slot don't matter because the slot's D was never touched since initialization, not because the kernel "will overwrite it." This is a documentation/reasoning flaw, not a code flaw.

**The real latent risk:**

If `initializeAltBufferSets` (or a future path) ever runs a kernel against an alt slot _before_ `beginAsyncReset` consumes it, or if a future feature passes an alt slot to a kernel as an output _and_ passes it again as a C input on a subsequent run, the warm-path assumption would break. The flag encodes "alt slots have been initially filled" but is read as "safe to skip re-DMA for output tensor." Those are not the same thing, and nothing enforces the gap.

### Is it a genuine Proxy Predicate flaw?

**Partial (safe by construction today, fragile by design).**

The warm-path skip is safe today because alt slots are never used as kernel output destinations — they are only filled by `fillSlot` and then consumed (read-only from the kernel's perspective via `m_cachedInputsRing`). But `m_altSlotsReady = true` encodes only "alt slots are filled"; it does not encode "no kernel has ever written to an alt slot" or "D contents are irrelevant." The safety is maintained by an invariant that lives in the ring architecture, not in the flag.

`ringEligible()` does not check beta or m_cEqualsD, and neither does the warm-path guard. If a future mode caused the kernel to write into an alt slot (e.g. rotating output across all ring slots), the warm path would silently reuse stale D contents.

### Specific failure scenario

1. A hypothetical future change makes benchmark runs rotate through all ring slots (instead of always using the active slot), so a kernel writes its output into slot `targetIdx` before `beginAsyncReset` claims it is "ready."
2. `m_altSlotsReady` is still true (not cleared by any of this).
3. `beginAsyncReset` fires the warm path: no re-DMA, event recorded.
4. The next kernel run reads D from that slot as its C input (in-place or beta!=0 mode), getting the _previous run's output_ instead of the initialized C.
5. Results are silently wrong.

### Severity

Unreachable today but fragile. The invariant "alt slots are never kernel output targets" is architectural and undocumented in the flag. A locally-reasonable extension (ring-wide rotating output) would silently break it.

---

## Candidate 3: `m_cpuInit && Disable && !problemDependent` in `prepareCPUInputs` overloads

### Evidence

**Both overloads:**

`DataInitialization.hpp:208-219` — `prepareCPUInputs(ContractionProblemGroupedGemm const&)`:
```cpp
if(m_cpuInit && m_curBoundsCheck == BoundsCheckMode::Disable
   && !m_problemDependentData)
{
    std::vector<void**> bPtr;
    if(m_elementsToValidate)
        resetOutput(m_cpuPtrs, bPtr, m_maxElements, m_groupedOffsets,
                    problem.gemms[0], hipMemcpyHostToHost);
}
else
{
    if(m_problemDependentData)
        initializeCPUInputs(problem);
    std::vector<void**> bPtr;
    copyInputs(m_cpuPtrs, bPtr, m_maxElements, m_groupedOffsets,
               problem.gemms[0], hipMemcpyHostToHost);
    m_cpuInit = false;
}
```

`DataInitialization.hpp:239-268` — `prepareCPUInputs(ContractionProblemGemm const&)`:
```cpp
if(m_cpuInit && m_curBoundsCheck == BoundsCheckMode::Disable
   && !m_problemDependentData)
{
    std::vector<void**> bPtr;
    if(m_elementsToValidate)
        resetOutput(m_cpuPtrs, bPtr, m_maxElements, m_groupedOffsets,
                    problem, hipMemcpyHostToHost);
}
else
{
    if(m_problemDependentData)
        initializeCPUInputs(problem);
    std::vector<void**> bPtr;
    copyInputs(m_cpuPtrs, bPtr, m_maxElements, m_groupedOffsets,
               problem, hipMemcpyHostToHost);
    m_cpuInit = false;
}
```

**Are the conditions byte-for-byte identical?**

Yes. The three-term condition `m_cpuInit && m_curBoundsCheck == BoundsCheckMode::Disable && !m_problemDependentData` appears verbatim in both overloads.

**Is there a named predicate?**

No. There is no named method or variable that captures this conjunction. The condition is spelled out inline in both places.

**Note on `m_cpuInit`:**

`m_cpuInit` is declared at `DataInitialization.hpp:1218`:
```cpp
bool m_cpuInit   = false;
```
It is set to `false` in the else branches of both overloads (`m_cpuInit = false`). However, there is no site in the current code that sets `m_cpuInit = true`. The fast path (`m_cpuInit && ...`) can therefore never fire — `m_cpuInit` is always `false`. Both overloads always take the else branch.

This means:
- The fast path code is dead code.
- The three-term condition, while duplicated, is unreachable.

**Are the overloads subtly different?**

The only difference between the two overloads is the `problem` argument passed to `resetOutput` and `copyInputs`:
- Grouped: `problem.gemms[0]` (accesses the first sub-problem of a grouped GEMM)
- Plain: `problem` directly

The predicate itself is identical. The fast-path body also differs in the same way (same structural pattern, different argument). No drift has occurred _yet_, but there is no named predicate to prevent drift from occurring independently.

### Is it a genuine Proxy Predicate flaw?

**Partial — genuine duplication risk, but currently the fast path is dead code.**

The condition is spelled out identically in two places with no named owner. If `m_cpuInit` were ever set to `true` (it currently is never set to `true`), the two copies of the condition would be the single mechanism preventing drift. A change to the condition in one overload (e.g. adding a check for `!m_sparse` in one but not the other) would silently diverge.

The deeper issue: `m_cpuInit` has no writer, making the fast path permanently dead. This suggests the condition was designed for a feature that was either removed or never completed. The duplicated condition is a maintenance liability regardless.

### Specific failure scenario (hypothetical, requires enabling `m_cpuInit`)

1. A future change adds `m_cpuInit = true` (e.g. after a successful `initializeCPUInputs` call).
2. Developer adds a new condition check to the plain-GEMM overload (e.g. `&& !m_sparse`) but forgets to add it to the grouped-GEMM overload.
3. Sparse grouped-GEMM problems take the fast path incorrectly, skipping `initializeCPUInputs`.
4. The reference validator computes against stale or zero-initialized CPU buffers.
5. Validation silently passes because both reference and GPU see the same stale data.

### Severity

Unreachable today (`m_cpuInit` is never set `true`). The duplication is a latent drift risk if `m_cpuInit` is ever enabled. The dead-code status of `m_cpuInit`'s writer is itself a maintenance hazard (the flag name implies it should be set somewhere).

---

## Candidate 4: `!m_gpuPtrs.empty()` as proxy for "GPU inputs current for this problem"

### Evidence

**The decision site:**

`DataInitialization.hpp:910-912` — inside `preSolution`:
```cpp
if(m_currentSolution != nullptr
   && m_mxScaleFormat > 0
   && m_currentGemmProblem != nullptr
   && !m_gpuPtrs.empty())
{
    bool isMX = isMXProblemExceptF6(*m_currentGemmProblem);
    if(isMX)
    {
        initializeMXData(*m_currentGemmProblem);
        copyValidToGPUBuffer(*m_currentGemmProblem);
        copyInputs(m_gpuPtrs, m_gpuBatchPtrs, ...);
        // ... sync cpuInput.current from valid
    }
}
```
`!m_gpuPtrs.empty()` is used as a proxy for "GPU inputs have been prepared for the active problem."

**What does non-empty actually guarantee?**

`m_gpuPtrs` is populated by `copyInputs` (which push_backs to it) during `prepareGPUInputsInternal`. It is cleared by `copyInputs` at its start (`ptrs.clear()`). It is _also_ updated by `advanceBuffer`:

`DataInitialization.hpp:1196`:
```cpp
m_gpuPtrs = m_gpuPtrsRing[m_activeIdx];
```
And by `cancelAsyncReset`:
`DataInitialization.hpp:323`:
```cpp
m_gpuPtrs = m_gpuPtrsRing[0];
```

So after `cancelAsyncReset`, `m_gpuPtrs` is set to `m_gpuPtrsRing[0]` — the pointers for ring slot 0. If slot 0 was filled for the _previous_ problem and the ring was not cleared (but it is cleared by `cancelAsyncReset` at lines 332-337), this would be stale. However, `cancelAsyncReset` does clear the alt ring slots (indices 1..N) but explicitly sets `m_gpuPtrs = m_gpuPtrsRing[0]`.

After `cancelAsyncReset`, `m_gpuPtrsRing[0]` still holds the previous problem's pointers (the `for` loop only clears indices 1..MAX_BUFFER_SETS, not index 0). So immediately after a problem change:
- `m_gpuPtrs` = `m_gpuPtrsRing[0]` = previous problem's GPU pointers
- `!m_gpuPtrs.empty()` is **true**

Then `prepareGPUInputs` is called (main.cpp:1245), which calls `prepareGPUInputsInternal`, which calls `copyInputs` (clearing and re-populating `m_gpuPtrs`) and ultimately calls `copyInputs(m_gpuPtrs, ...)` on the new problem. Only after this does `m_gpuPtrs` hold current-problem data.

**The `preSolution` call ordering:**

In main.cpp, the per-solution loop is:
1. `listeners.preProblem(problem)` (line 1234) — sets `m_currentGemmProblem`, clears `m_batchInit`
2. `dataInit->cancelAsyncReset()` (line 1240) — resets ring, sets `m_gpuPtrs = m_gpuPtrsRing[0]` (previous problem)
3. `inputs = dataInit->prepareGPUInputs(problem)` (line 1245) — re-initializes GPU buffers for new problem, repopulates `m_gpuPtrs`
4. (solution iteration starts)
5. `listeners.preSolution(solution.get())` (line 1367) — calls `preSolution` which checks `!m_gpuPtrs.empty()`
6. `inputs = dataInit->prepareGPUInputs(problem)` (line 1378) — `gpu_input_reset` before benchmark_runs

By step 5, `prepareGPUInputs` has already been called (step 3), so `m_gpuPtrs` has been repopulated for the current problem. The stale-window (between steps 2 and 3) does not include a `preSolution` call.

**Can `preSolution` be called before `prepareGPUInputs` after a problem change?**

Not in the current call order. The main.cpp loop strictly calls `prepareGPUInputs` before entering the solution loop. Within the solution loop, `preSolution` is called, then `prepareGPUInputs` again (as `gpu_input_reset`). So by the time `preSolution` fires, `m_gpuPtrs` is always populated for the current problem.

**However — the proxy meaning:**

`!m_gpuPtrs.empty()` means "at least one `copyInputs` call has completed." It does not mean:
- The content is current for the active problem (could be stale if `cancelAsyncReset` ran after the last `prepareGPUInputs` — but this doesn't happen in the current ordering).
- The content is consistent with `m_currentGemmProblem` (they are set by different paths: `m_currentGemmProblem` by `preProblem`, `m_gpuPtrs` by `prepareGPUInputs`).

In practice, by the time `preSolution` checks the flag, the call order guarantees they are in sync. But the guarantee comes from call-site discipline in main.cpp, not from any relationship between the flag and `m_currentGemmProblem`.

**After `advanceBuffer` — are pointers still valid?**

`advanceBuffer` at line 1196:
```cpp
m_gpuPtrs = m_gpuPtrsRing[m_activeIdx];
```
After advancing, `m_gpuPtrs` points to the new active ring slot's pointer set. `!m_gpuPtrs.empty()` remains true (ring slots are pre-filled). The MX re-init in `preSolution` would then write into the current active slot's GPU buffers and also call `copyInputs(m_gpuPtrs, ...)` which clears and repopulates `m_gpuPtrs`. This is correct — the MX re-init modifies the active slot's data.

**Does `advanceBuffer` correctly maintain the invariant?**

Yes — `m_gpuPtrsRing[slotIdx]` is only ever non-empty if that slot was filled by `fillSlot`. `cancelAsyncReset` clears alt ring slots but not slot 0. Slot 0 is only non-empty after the first `prepareGPUInputsInternal` call for the current problem. So the invariant "non-empty implies filled for current problem" holds for slot 0 by the time `preSolution` fires, because `prepareGPUInputs` was called first.

### Is it a genuine Proxy Predicate flaw?

**Partial (safe by construction today, fragile naming).**

The flag `!m_gpuPtrs.empty()` gates an MX-specific re-initialization in `preSolution`. The intent is "has `prepareGPUInputs` been called for this problem?" The current call ordering in main.cpp guarantees this is true by the time `preSolution` fires. The proxy is correct today.

The fragility: `m_gpuPtrs` can be non-empty with data from a _previous_ problem (between `cancelAsyncReset` and the subsequent `prepareGPUInputs` call). If `preSolution` were ever called in that window — e.g. if a caller restructured the problem/solution loops — the MX re-init would run against `m_currentGemmProblem` (new problem) but use the old problem's GPU pointers until `copyInputs` inside the re-init overwrites them. The `copyInputs` call within `preSolution`'s MX block does clear and repopulate `m_gpuPtrs`, so it would self-correct — but only after uploading MX data computed from `m_currentGemmProblem` into buffers whose _prior_ content reflects the previous problem.

### Specific failure scenario

1. A refactoring moves `preSolution` to be called before `prepareGPUInputs` on the first solution of a new problem (e.g. to allow solution-selection to influence initialization).
2. After `cancelAsyncReset`, `m_gpuPtrs = m_gpuPtrsRing[0]` — non-empty, holds previous problem's pointers.
3. `preSolution` fires: `!m_gpuPtrs.empty()` is true; `m_currentGemmProblem` is the new problem.
4. `initializeMXData(*m_currentGemmProblem)` computes scale data for the new problem's shape.
5. `copyValidToGPUBuffer(*m_currentGemmProblem)` uploads data sized for the new problem into `gpuInput.valid`.
6. `copyInputs(m_gpuPtrs, ...)` copies from `gpuInput.valid` into `m_gpuPtrs[0]` — the previous problem's slot 0 buffer — which may be the wrong size if problem shapes differ.
7. Results: possible buffer overrun or data corruption; silently wrong output.

### Severity

Unreachable today, fragile. The invariant "preSolution always fires after prepareGPUInputs within the same problem" is a call-site convention in main.cpp, not enforced by the DataInitialization interface.

---

## Summary Table

| Candidate | Current Status | Proxy Encodes | Reader Assumes | Gap Enforced By |
|-----------|---------------|---------------|----------------|-----------------|
| `m_batchInit` | Fixed (regression was real) | "batch upload ran at least once" | "batch pointers current for active problem" | `preProblem` reset (call-site discipline) |
| `m_altSlotsReady` | Safe today, fragile | "alt slots were initially filled" | "D contents in alt slot are irrelevant (overwritten by next kernel)" | Architectural invariant: kernels never write to alt slots |
| `m_cpuInit && Disable && !problemDependent` | Dead code (m_cpuInit never set true) | N/A | N/A | N/A (unreachable) |
| `!m_gpuPtrs.empty()` | Safe today, fragile | "copyInputs ran at least once" | "GPU inputs are current for the active problem" | Call-site ordering in main.cpp |

---

## Additional Candidates from Branch Diff

### Candidate 5: `slotInitialized` in `fillSlot` fast path

#### Evidence

**The decision site (`DataInitialization.cpp:3614-3637`):**

```cpp
bool slotInitialized = !m_gpuPtrsRing[slotIdx].empty();

if(slotInitialized && m_gpuInit
   && m_curBoundsCheck == BoundsCheckMode::Disable
   && !m_problemDependentData && !needSwizzle && !needMXSwizzle)
{
    // Fast path: slot already has data, only reset output D
    if(m_elementsToValidate)
    {
        resetOutput(m_gpuPtrsRing[slotIdx], ...);
        m_cachedInputsRing[slotIdx] = buildGPUProblemInputs(...);
    }
    // else: pointers unchanged, reuse existing m_cachedInputsRing[slotIdx]
}
else
{
    // Full path: initialize all tensors into target slot
    ...
    copyInputs(m_gpuPtrsRing[slotIdx], ...);
    initializeGPUBatchedInputs(problem, targetStream);
    m_cachedInputsRing[slotIdx] = buildGPUProblemInputs(...);
}
```

`slotInitialized` encodes "this slot's pointer list was previously populated" and is read as "this slot was filled with the current problem's data, so only the output tensor (D) needs reset."

**Who calls `fillSlot` and is the fast path actually reachable:**

There are two call sites:

1. `initializeAltBufferSets` (`DataInitialization.cpp:3678-3681`): guarded by `!m_gpuPtrsRing[1].empty()` early return, so `fillSlot` is only ever called here on empty slots. `slotInitialized == false` here always.

2. `beginAsyncReset` (`DataInitialization.hpp:387-392`): calls `fillSlot` only when `!m_altSlotsReady`. A staff-engineer review caught a nuance the initial analysis missed: **`m_gpuInit` is set `true` once and never reset** (`.cpp:3570`). From the second problem onward, `prepareGPUInputsInternal` takes the fast-path return at line 3518 *before* the `if(!targetStream)` block that calls `initializeAltBufferSets`. So on problems 2..N, `m_altSlotsReady` stays `false` while `ringEligible()` is already `true` — meaning `beginAsyncReset` does call `fillSlot` on every problem after the first.

The fast path in `fillSlot` still doesn't fire, however, because `cancelAsyncReset` clears `m_gpuPtrsRing[1..N]` before each new problem, so every `fillSlot` call from `beginAsyncReset` targets a just-cleared slot (`slotInitialized == false`). The conclusion (fast path is dead code) is correct, but for this reason — not because `fillSlot` is unreachable.

**What the proxy would mean if the fast path became reachable:**

`!m_gpuPtrsRing[slotIdx].empty()` encodes "at least one `copyInputs` wrote to this slot." The reader assumes "slot was filled for the active problem." A future caller that invoked `fillSlot` on a slot that was already populated for the current problem (e.g., a "refresh" feature) would trigger the fast path, skip `initializeGPUBatchedInputs`, and leave stale batch pointer arrays — the exact regression `m_batchInit` was introduced to fix on the non-ring path.

#### Is it a genuine Proxy Predicate flaw?

**Partial — dead-code fast path with a latent regression trap.** The fast path is unreachable today because every `fillSlot` caller targets a just-cleared slot. If a future caller bypassed that invariant, silently wrong results for batched problems would follow.

#### Specific failure scenario (hypothetical)

1. A future change calls `fillSlot` directly on an already-populated slot to refresh D's initial value mid-problem.
2. `slotInitialized == true`; all other fast-path conditions hold.
3. Fast path fires: only `resetOutput` runs; `initializeGPUBatchedInputs` is skipped.
4. Batch pointer array in the slot is stale.
5. Kernel runs with wrong batch pointers — silently wrong results for batched GEMMs.

#### Severity

Unreachable today (dead code). Latent risk mirrors the original `BatchPointerReset` regression.

---

### Candidate 9: `m_availableSlots > 0` in `prepareGPUInputs` fast path — proxy for ring validity

#### Evidence

**The fast-path decision (`DataInitialization.hpp:277-288`):**

```cpp
if(m_availableSlots > 0)
{
    // The ring is only ever filled under ringEligible() (enforced in
    // beginAsyncReset), i.e. BoundsCheck::Disable and !problemDependent.
    // Under those conditions the typed overloads' GuardPage flip and
    // conditional CPU-init are both no-ops, so bypassing the dynamic_cast
    // dispatch is exact, not approximate.  Caller must waitCopyDone()
    // before use (main.cpp before benchmark_runs).
    assert(ringEligible());
    advanceBuffer();
    return m_cachedGPUInputs;
}
```

`m_availableSlots > 0` is the gate for bypassing the full typed `prepareGPUInputs` dispatch. The comment explains the reasoning, and an `assert(ringEligible())` has been added as the debug verifier.

**What the proxy encodes vs. what the reader assumes:**

`m_availableSlots > 0` encodes: "at least one ring slot has been filled and its DMA event recorded, but not yet consumed by `advanceBuffer`." The reader assumes: "the GuardPage bounce, the CPU-init call, and the full `dynamic_cast` dispatch are all safe to skip."

The implication `m_availableSlots > 0 → safe-to-skip` is guaranteed because:
1. `m_availableSlots` is only incremented by `beginAsyncReset`.
2. `beginAsyncReset` only runs when `ringEligible()` is true.
3. `ringEligible()` includes `m_curBoundsCheck == BoundsCheckMode::Disable` (no GuardPage) and `!m_problemDependentData` (no CPU-init needed).
4. Therefore every slot counted by `m_availableSlots` was prepared under the conditions that make the dispatch bypass exact.

A staff-engineer review closed one further gap the initial analysis left open: it verified that no term in `ringEligible()` can flip between fill-time and consume-time while slots are outstanding. All five terms (`m_hasAltBuffers`, `m_copyStream`, `m_gpuInit`, `m_curBoundsCheck`, `m_problemDependentData`) are effectively immutable for the lifetime of the `DataInitialization` object — set from constructor arguments or one-time initialization, never mutated. `ringEligible()` therefore returns the same value from construction to destruction, making the assert a genuine structural guard rather than a placebo.

The `assert(ringEligible())` at the reader converts the reasoning above from a comment into a debug-mode check, catching any future drift where `m_availableSlots` could become positive under conditions that invalidate the bypass.

**Is this still a proxy predicate after the `assert` was added?**

Structurally, yes: `m_availableSlots > 0` is still a narrower predicate than "safe-to-bypass." But the `assert(ringEligible())` is precisely the structural fix recommended in `architectural_principle.md` (section 5): "Add a debug `assert(predicate)` at the reader to convert any future drift into a loud test failure instead of a silent wrong answer." The implication is now compiler-verifiable in debug builds.

#### Is it a genuine Proxy Predicate flaw?

**No — the assert converts the flaw from silent to loud.** Prior to the assert, this was a textbook proxy predicate: `m_availableSlots > 0` read for the broader meaning "ring conditions were met when the slot was filled." After the assert, the reader explicitly checks the full predicate at debug time. The structural gap (the implication lives in the writer, not in the type) still exists, but it is enforced by a runtime check that would produce an assertion failure rather than silently wrong results if a future change broke the invariant.

The remaining fragility: `assert` is a no-op in release builds. If the invariant were broken, a release build would silently take the wrong fast path. The principled fix would be to make `ringEligible()` invariant-carrying at the type level — but the assert is a substantial improvement over the pre-branch state.

#### Severity

As-is: reduced risk (assert in debug mode). In release builds: fragile in the same way as Candidates 2 and 4 — safe today, breaks silently if a future change increments `m_availableSlots` outside `beginAsyncReset`, or if `ringEligible()` conditions change without updating the filling logic.

---

## Candidate 10: `gemms[0]`-sufficiency assumption in grouped GEMM fast path

### Evidence

**The fill-site call (`DataInitialization.hpp:389-392`):**

```cpp
else if(auto groupedProblem
        = dynamic_cast<ContractionProblemGroupedGemm const*>(problem))
    fillSlot(targetIdx, groupedProblem->gemms[0], m_copyStream);
```

`fillSlot` accepts only `ContractionProblemGemm const&`. When the problem is a grouped GEMM, `gemms[0]` is the argument. The fast path in `prepareGPUInputs(ContractionProblem const*)` (`DataInitialization.hpp:277-288`) then returns `m_cachedInputsRing[m_activeIdx]` — built from this `gemms[0]`-based fill — to the caller without re-entering the grouped overload.

**What `fillSlot` does with `gemms[0]`:**

In the full path (`DataInitialization.cpp:3639-3666`), `fillSlot`:

1. Takes `localOffsets = m_groupedOffsets` at line 3612. (`m_groupedOffsets` was populated by the initial slot-0 `prepareGPUInputsInternal` call, which in turn got its offsets from `copyInputs`, which pushed `p.groupedGemmOffsets` for each `VectorDataInitProperties` pristine entry.)
2. Clears `localOffsets` at line 3650, then repopulates it via `copyInputs(m_gpuPtrsRing[slotIdx], ..., localOffsets, problem, ...)` where `problem` is `gemms[0]`. `copyInputs` pushes `p.groupedGemmOffsets` for each tensor — the same per-gemm offset vector that was set during construction from ALL gemms.
3. Calls `buildGPUProblemInputs(m_gpuPtrsRing[slotIdx], ..., localOffsets, problem)` at line 3660.

**How `buildGPUProblemInputs` uses `localOffsets` (`DataInitialization.cpp:3099-3140`):**

```cpp
if(offsets.empty() || offsets[0].empty())
{
    // Plain GEMM path → ContractionInputs
}
else
{
    auto inputs = new ContractionGroupedInputs();
    setContractionGroupedInputs(ptrs, dummyBatchPtrs, ws, cdata,
                                /*isGPU=*/true, problem, offsets, inputs);
    // ...
}
```

Because `localOffsets[0]` is non-empty for a grouped problem (it contains one element-count entry per gemm in the group), the else branch fires and produces a `ContractionGroupedInputs`.

**How `setContractionGroupedInputs` uses `problem` (`DataInitialization.cpp:3019-3095`):**

```cpp
for(int idx = 0; idx < offsets[0].size(); idx++)
{
    // ... build unit per gemm
    u8Ptr[A] += multiplyElementSize(offsets[A][idx], problem.a().elementBytes());
    u8Ptr[B] += multiplyElementSize(offsets[B][idx], problem.b().elementBytes());
    u8Ptr[C] += multiplyElementSize(offsets[C][idx], problem.c().elementBytes());
    u8Ptr[D] += multiplyElementSize(offsets[D][idx], problem.d().elementBytes());
    // ... same for E, BIAS, SCALEA, SCALEB, SCALEC, SCALED
}
```

`problem.a().elementBytes()` etc. are taken from the single `ContractionProblemGemm` passed in — here `gemms[0]`. These element byte sizes are used to advance raw pointers across the contiguous buffer for ALL N gemms in the group, including `gemms[1]` through `gemms[N-1]`.

**Where `groupedGemmOffsets` is set (constructor, `DataInitialization.cpp:1136-1215`):**

The constructor iterates over ALL `problems.gemms` for grouped problems, accumulating per-gemm tensor sizes into `pristine.groupedGemmOffsets`. This means `groupedGemmOffsets` encodes the actual element count for each gemm in the group. The constructor throws if grouped offset sizes are inconsistent (`"Unable to update groupedGemmOffsets."`), enforcing that all problems in the benchmark set agree on the grouped structure.

**Is the call to `prepareGPUInputs(ContractionProblemGroupedGemm const&)` on the slow path structurally identical?**

`prepareGPUInputs(ContractionProblemGroupedGemm const& problem)` (`DataInitialization.hpp:401-416`) ultimately calls `prepareGPUInputsInternal(problem.gemms[0], nullptr)` at line 415. `prepareGPUInputsInternal` calls `copyInputs(m_gpuPtrs, ..., m_groupedOffsets, problem, ...)` at line 3543 (also with `gemms[0]` as the problem) and then `ConvertToProblemInputs(problem, true)` which calls `buildGPUProblemInputs(m_gpuPtrs, ..., m_groupedOffsets, problem)`. The slow path also uses `gemms[0]` everywhere `problem` is named.

So `fillSlot(targetIdx, gemms[0], ...)` exactly mirrors `prepareGPUInputsInternal(gemms[0], nullptr)`. The fast path is not introducing a new asymmetry; it faithfully reproduces the same `gemms[0]`-only pattern that the slow path uses.

**Is `gemms[0]`-sufficiency anywhere asserted or commented?**

No assert at the fill site (`DataInitialization.hpp:389-392`). The only documentation is in `grouped_gemm_fast_path_analysis.md` (Invariant B), which is external to the source. The `beginAsyncReset` code has no comment explaining why `gemms[0]` is sufficient.

### Is it a genuine Proxy Predicate flaw?

**No** — but with a structural dependency that is unstated in the code.

The `gemms[0]`-sufficiency is not a flaw in the fast path specifically. It is a shared assumption between the slow path and the fast path: both paths call through `gemms[0]`. The fast path does not introduce a weaker contract than the slow path. They are exactly equivalent in how they use `gemms[0]`.

The assumption `gemms[0]`-sufficiency for grouped GEMM pointer arithmetic rests on one fact: all gemms in the group must share the same element byte sizes for each tensor. This is structurally enforced by the data model in two ways:

1. **`m_vdata` is keyed by tensor-index × `DataType`** (`VectorDataInitProperties::pristine` is a `std::map<rocisa::DataType, PristineUnit>`). There is one memory allocation per unique `(tensor_index, DataType)` pair shared across all gemms. If `gemms[j].a().dataType()` differed from `gemms[0].a().dataType()`, the allocation loop at constructor line 1140 would fail to find a matching pristine entry and either create a separate one or throw — but critically, `setContractionGroupedInputs` only sees one starting pointer per tensor index and advances it using `gemms[0]`'s element size. If element sizes differed across gemms, the pointer arithmetic would be wrong.

2. **The constructor throws on mismatched grouped offset sizes** (line 1205-1208): `"Unable to update groupedGemmOffsets."` This checks that the number of per-gemm entries is consistent, but does not explicitly verify that all gemms' `elementBytes()` match.

**What could go wrong:** If a future extension allowed a grouped GEMM where `gemms[0].a().elementBytes() != gemms[1].a().elementBytes()` (e.g., mixing FP16 and FP32 gemms in one group — not currently supported but the data structure `std::vector<ContractionProblemGemm> gemms` does not prohibit it), then `setContractionGroupedInputs` called with `gemms[0]` would advance byte pointers for all subsequent gemms using the wrong element size. Both the slow path and the fast path would produce wrong `ContractionGroupedInputs`; this would not be a fast-path-specific bug.

### Specific failure scenario

**This is not a fast-path-specific bug.** The same issue exists on the slow path. If it triggers, both paths fail identically:

1. A grouped GEMM benchmark is configured with `gemms[0]` being FP16 (2 bytes/element) and `gemms[1]` being FP32 (4 bytes/element) for tensor A.
2. `setContractionGroupedInputs` uses `gemms[0].a().elementBytes() = 2` to advance the A pointer after gemm 0's chunk, arriving at `base + gemm0_A_elements * 2` bytes.
3. The intended start of gemm 1's A data is at `base + gemm0_A_elements * 2` bytes only if gemm 0 uses FP16. If gemm 1 actually stores FP32 data starting at the same offset, the pointer arithmetic is internally consistent — but the `setContractionInputs` call for gemm 1 would set the A pointer from this advanced position using gemm 1's descriptor, which expects FP32. The mismatch in types across gemms would be caught much earlier by the `m_vdata` lookup (the pristine map for tensor A would have two entries keyed by FP16 and FP32 respectively, and the construction would need to handle sub-buffer allocation differently).

In practice: the current client does not support mixed-type grouped GEMMs, and the `m_vdata` data model does not accommodate them cleanly. The homogeneous-type invariant for grouped GEMMs is implicit in the allocation model, not asserted, but it is architecturally enforced by the constructor's allocation loop structure.

### Severity

**Not a current flaw — structural and slow-path-shared assumption.**

The `gemms[0]`-sufficiency in `fillSlot` (and identically in `prepareGPUInputsInternal`) is safe because:

1. The `groupedGemmOffsets` data (number of elements per gemm per tensor) is populated from ALL gemms at constructor time, not just `gemms[0]`. The slot built by `fillSlot` correctly encodes N sub-inputs for an N-gemm group.
2. The element byte sizes used for pointer arithmetic come from `gemms[0]`, which is sufficient because the current data model structurally enforces homogeneous element types across gemms in a group (one buffer per `(tensor_index, DataType)` pair shared by all gemms).
3. The fast path is strictly equivalent to the slow path: both call through `gemms[0]` for the same purpose.

The one missing guard: there is no assertion at the `fillSlot` call site (`DataInitialization.hpp:389-392`) that `gemms[0].a().elementBytes() == gemms[i].a().elementBytes()` for all i, or that `gemms[0].a().dataType() == gemms[i].a().dataType()`. Adding such a check would convert the implicit homogeneous-type invariant into a structural guarantee at the fill site. Without it, a future extension that relaxes the type homogeneity restriction would silently produce wrong pointer arithmetic in `setContractionGroupedInputs` on both the slow and the fast path.

---

## Summary Table

| Candidate | Current Status | Proxy Encodes | Reader Assumes | Gap Enforced By |
|-----------|---------------|---------------|----------------|-----------------|
| `m_batchInit` | Fixed (regression was real) | "batch upload ran at least once" | "batch pointers current for active problem" | `preProblem` reset (call-site discipline) |
| `m_altSlotsReady` warm path | Safe today, fragile | "alt slots were initially filled" | "alt slot contents are untouched since init (no kernel ever wrote to them)" | Architectural invariant: kernels never write to alt slots |
| `m_cpuInit && Disable && !problemDependent` | Dead code (`m_cpuInit` never set `true`) | N/A | N/A | N/A (unreachable) |
| `!m_gpuPtrs.empty()` in `preSolution` | Safe today, fragile | "`copyInputs` ran at least once" | "GPU inputs are current for the active problem" | Call-site ordering in main.cpp |
| `slotInitialized` in `fillSlot` | Dead code (fast path unreachable) | "slot pointer list was populated" | "slot filled for current problem; only D needs reset" | Every caller targets a just-cleared slot — latent regression trap if that changes |
| `m_availableSlots > 0` fast path | Mitigated by `assert(ringEligible())` | "ring slot filled and ready" | "dispatch bypass is safe (no GuardPage, no problem-dep data)" | `beginAsyncReset` gate (writer) + `assert(ringEligible())` at reader; all `ringEligible()` terms are immutable for object lifetime |
| `gemms[0]`-sufficiency in grouped GEMM fill | Not a flaw — shared slow/fast-path assumption | N/A | "all gemms share element types; grouped offsets come from full group construction" | Implicit: `m_vdata` allocation model (one buffer per DataType across all gemms); no per-gemm element-type assertion at fill site |
