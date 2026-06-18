# Ring-Buffer Alt Slot Usage — Test Investigation

**Question:** How do we write tests that verify ring-buffer alt slots (slots 1 and 2) are actually
being consumed — i.e., that `advanceBuffer()` fires and the ring fast path is exercised — in the
no-benchmark case?

---

## Context

The ring fast path in `prepareGPUInputs(ContractionProblem*)` (header line ~245) fires when
`m_availableSlots > 0`, calling `advanceBuffer()` to rotate the active slot and returning
`m_cachedGPUInputs` without re-running tensor copy.

The bug being tested: when `noBenchmarkRuns=true`, `beginAsyncReset()` is never called (it only
exists inside the benchmark while-loop body in `main.cpp`), so `m_availableSlots` stays 0 and the
fast path never fires. Slots 1 and 2 are allocated and filled by `initializeAltBufferSets` but
permanently idle.

The proposed fix adds `beginAsyncReset` calls after the warmup block in the no-benchmark solution
path. The tests here verify that fix works and guard against regressions.

---

## Testability Assessment

### What is accessible today (no production changes)

Ring-buffer state fields are **`protected`**, not `private`:

```
m_availableSlots      — count of filled-but-not-consumed slots
m_activeIdx           — which ring slot is currently active (0, 1, 2)
m_numActiveBuffers     — 2 (benchmark) or 3 (no-benchmark)
m_hasAltBuffers       — whether alt allocations succeeded
m_altSlotsFilled      — whether initializeAltBufferSets has run
m_cachedGPUInputs     — active slot's cached shared_ptr<ProblemInputs>
m_cachedInputsRing[]  — per-slot cached inputs (distinct allocations)
m_gpuPtrsRing[]       — per-slot GPU pointer vectors
```

A test subclass can expose all of these via public accessors without touching production code.

### Public methods tests can call directly

```
prepareGPUInputs(ContractionProblem*)   — ring fast path fires inside this
beginAsyncReset(ContractionProblem const*)  — increments m_availableSlots
cancelAsyncReset()                      — resets m_availableSlots and m_activeIdx to 0
waitCopyDone(hipStream_t)               — insert GPU-side barrier before kernel
syncCopyStream()                        — CPU-block until copy stream is idle
preProblem(ContractionProblem*)         — resets m_batchInitProblem, m_currentSolution
```

### The pointer-identity shortcut (no subclassing)

`m_cachedInputsRing[0]`, `[1]`, and `[2]` are distinct `shared_ptr` objects built by separate
`fillSlot` calls. After `advanceBuffer()`, `m_cachedGPUInputs` is set to
`m_cachedInputsRing[m_activeIdx]`. So the GPU pointer embedded in the returned `ProblemInputs`
(e.g. `ContractionInputs::a`) differs between slot 0 and slot 1. A test can detect ring
advancement purely via the public API by comparing these pointers.

---

## Setup: `buildRingArgs` helper

The existing `buildArgs` in `BatchPointerReset_test.cpp` does not set `num-benchmarks`,
`num-enqueues-per-sync`, or `num-syncs-per-benchmark`. These must be added (the DataInitialization
constructor reads them at line ~988 of `DataInitialization.cpp`). Setting all three to 0 triggers
`noBenchmarkRuns = true`, which sets `m_numActiveBuffers = 3`.

```cpp
// In the test file's anonymous namespace alongside buildArgs / makeBatchedProblem:
po::variables_map buildRingArgs(std::vector<std::vector<size_t>> problemSizes)
{
    auto args = buildArgs(std::move(problemSizes));

    // Trigger noBenchmarkRuns=true → m_numActiveBuffers=3 (triple-buffering)
    args["num-benchmarks"]          = vv(std::any(int(0)));
    args["num-enqueues-per-sync"]   = vv(std::any(int(0)));
    args["num-syncs-per-benchmark"] = vv(std::any(int(0)));

    return args;
}
```

---

## Approach A — Subclass accessor (recommended primary test)

Create a thin subclass in the test file that exposes the protected fields. No production code
changes required.

```cpp
class TestableDataInit : public DataInitialization
{
public:
    using DataInitialization::DataInitialization;
    size_t activeIdx()      const { return m_activeIdx; }
    size_t availableSlots() const { return m_availableSlots; }
    bool   altSlotsFilled() const { return m_altSlotsFilled; }
    size_t numActiveBuffers()const { return m_numActiveBuffers; }
};
```

### Test: slot index advances on fast path

```cpp
// ---------------------------------------------------------------------------
// Verifies that calling beginAsyncReset() primes m_availableSlots and that
// subsequent prepareGPUInputs() calls consume those slots via advanceBuffer(),
// incrementing m_activeIdx from 0 → 1 → 2.
//
// This is the direct test of the ring-buffer fast path.  It fails if:
//   - beginAsyncReset() does not increment m_availableSlots (ring not primed)
//   - prepareGPUInputs() does not call advanceBuffer() when m_availableSlots>0
//   - advanceBuffer() does not update m_activeIdx
// ---------------------------------------------------------------------------
TEST(RingBuffer, SlotIndexAdvancesOnFastPath)
{
    constexpr size_t M = 64, N = 64, K = 64;
    auto args = buildRingArgs({{M, N, 1, K}});
    ClientProblemFactory factory(args);
    TestableDataInit     dataInit(args, factory);

    ASSERT_EQ(dataInit.numActiveBuffers(), 3u)
        << "noBenchmarkRuns=true must select triple-buffering";

    auto p = makePlainProblem(M, N, K); // non-batched ContractionProblemGemm

    // Slow path: fills slot 0, initializeAltBufferSets fills slots 1 and 2
    dataInit.prepareGPUInputs(&p);
    EXPECT_EQ(dataInit.activeIdx(), 0u);
    EXPECT_TRUE(dataInit.altSlotsFilled());
    EXPECT_EQ(dataInit.availableSlots(), 0u);

    // Prime ring: warm path records no-op events, increments m_availableSlots
    dataInit.beginAsyncReset(&p);
    EXPECT_EQ(dataInit.availableSlots(), 1u);
    dataInit.beginAsyncReset(&p);
    EXPECT_EQ(dataInit.availableSlots(), 2u);

    hipStream_t stream;
    HIP_CHECK_EXC(hipStreamCreate(&stream));

    // Fast-path call 1: advanceBuffer → m_activeIdx=1, m_availableSlots=1
    dataInit.waitCopyDone(stream);
    dataInit.prepareGPUInputs(&p);
    EXPECT_EQ(dataInit.activeIdx(), 1u);
    EXPECT_EQ(dataInit.availableSlots(), 1u);

    // Fast-path call 2: advanceBuffer → m_activeIdx=2, m_availableSlots=0
    dataInit.beginAsyncReset(&p);  // refill slot that was just vacated
    dataInit.waitCopyDone(stream);
    dataInit.prepareGPUInputs(&p);
    EXPECT_EQ(dataInit.activeIdx(), 2u);

    HIP_CHECK_EXC(hipStreamDestroy(stream));
}
```

**Strength:** Unambiguous — directly observes `m_activeIdx` changing. No inference required.

---

## Approach B — Black-box pointer identity (no subclassing)

Each slot is filled by a separate `fillSlot` call, which allocates separate GPU buffers. After
`advanceBuffer()`, the `ContractionInputs::a` pointer returned by `prepareGPUInputs` differs from
slot 0's pointer. This is detectable via the public API alone.

```cpp
// ---------------------------------------------------------------------------
// Verifies that the ring fast path returns a different GPU allocation than
// the initial slow-path call, proving advanceBuffer() rotated to an alt slot.
//
// Does not require subclassing.  Fails if prepareGPUInputs() returns slot 0's
// pointer when the ring should have advanced to slot 1.
// ---------------------------------------------------------------------------
TEST(RingBuffer, FastPathReturnsDifferentGPUPointer)
{
    constexpr size_t M = 64, N = 64, K = 64;
    auto args = buildRingArgs({{M, N, 1, K}});
    ClientProblemFactory factory(args);
    DataInitialization   dataInit(args, factory);

    auto p = makePlainProblem(M, N, K);

    // Slow path: get slot 0's GPU pointer for tensor A
    auto inputs0 = dataInit.prepareGPUInputs(&p);
    auto* ci0    = dynamic_cast<ContractionInputs*>(inputs0.get());
    ASSERT_NE(ci0, nullptr);
    void* ptr_a_slot0 = ci0->a;

    // Prime ring with 1 available slot
    dataInit.beginAsyncReset(&p);

    hipStream_t stream;
    HIP_CHECK_EXC(hipStreamCreate(&stream));
    dataInit.waitCopyDone(stream);

    // Fast path: should return slot 1's inputs
    auto inputs1 = dataInit.prepareGPUInputs(&p);
    auto* ci1    = dynamic_cast<ContractionInputs*>(inputs1.get());
    ASSERT_NE(ci1, nullptr);
    void* ptr_a_slot1 = ci1->a;

    EXPECT_NE(ptr_a_slot0, ptr_a_slot1)
        << "Fast path must return alt-slot GPU pointer for tensor A.  "
           "Identical pointers mean the ring did not advance — "
           "beginAsyncReset was not called or the fast-path guard failed.";

    HIP_CHECK_EXC(hipStreamDestroy(stream));
}
```

**Strength:** Pure public API, survives field renames. Catches the regression where the fix to
`main.cpp` is reverted (no `beginAsyncReset` in the no-benchmark path → same pointer returned).

---

## Approach C — GPU data readback (strongest correctness guarantee)

Reads tensor A back from the alt slot and verifies it contains the correct initialized values.
This proves the alt slot holds valid data — not just that a pointer changed.

```cpp
// ---------------------------------------------------------------------------
// Verifies that the alt slot (slot 1) served by the ring fast path contains
// valid tensor data initialized to InitMode::Two (value 2.0f for tensor A).
//
// Proves: initializeAltBufferSets correctly filled the alt slot, AND
//         advanceBuffer rotated to it, AND the data survived intact.
// ---------------------------------------------------------------------------
TEST(RingBuffer, AltSlotContainsValidInitializedData)
{
    constexpr size_t M = 32, N = 32, K = 32;
    constexpr size_t CHECK_ELEMS = 8;

    // init-a = InitMode::Two → every element of tensor A = 2.0f
    auto args = buildRingArgs({{M, N, 1, K}}, /*initA=*/InitMode::Two);
    ClientProblemFactory factory(args);
    DataInitialization   dataInit(args, factory);

    auto p = makePlainProblem(M, N, K);

    // Slow path: fills slot 0 and alt slots via initializeAltBufferSets
    dataInit.prepareGPUInputs(&p);

    // Prime ring: warm path records event, m_availableSlots=1
    dataInit.beginAsyncReset(&p);
    dataInit.syncCopyStream(); // ensure event is fully recorded

    hipStream_t stream;
    HIP_CHECK_EXC(hipStreamCreate(&stream));
    dataInit.waitCopyDone(stream);
    HIP_CHECK_EXC(hipStreamSynchronize(stream));

    // Fast path: advance to slot 1
    auto inputs1 = dataInit.prepareGPUInputs(&p);
    auto* ci1    = dynamic_cast<ContractionInputs*>(inputs1.get());
    ASSERT_NE(ci1,      nullptr);
    ASSERT_NE(ci1->a,   nullptr);

    // Read back CHECK_ELEMS floats from tensor A in slot 1
    float host_a[CHECK_ELEMS];
    HIP_CHECK_EXC(hipMemcpy(host_a, ci1->a,
                            CHECK_ELEMS * sizeof(float), hipMemcpyDeviceToHost));

    for(size_t i = 0; i < CHECK_ELEMS; i++)
        EXPECT_EQ(host_a[i], 2.0f)
            << "Alt-slot tensor A[" << i << "] must equal 2.0f (InitMode::Two). "
               "Got " << host_a[i] << ". "
               "This means either initializeAltBufferSets did not fill the slot "
               "or advanceBuffer did not rotate to it.";

    HIP_CHECK_EXC(hipStreamDestroy(stream));
}
```

**Strength:** End-to-end correctness proof. Fails if alt slot data is corrupt, uninitialized, or
if the wrong slot was served.

---

## Approach D — cancelAsyncReset regression (ring reset correctness)

Verifies that `cancelAsyncReset` correctly returns the ring to slot 0 and clears `m_altSlotsFilled`,
so the next problem gets fresh slots.

```cpp
// ---------------------------------------------------------------------------
// Verifies that cancelAsyncReset resets m_activeIdx to 0, clears
// m_availableSlots, and forces initializeAltBufferSets to re-run on the
// next prepareGPUInputs call (m_altSlotsFilled cleared).
//
// Regression guard for the problem-boundary ring reset.
// ---------------------------------------------------------------------------
TEST(RingBuffer, CancelAsyncResetRestoresSlot0)
{
    auto args = buildRingArgs({{64, 64, 1, 64}});
    ClientProblemFactory factory(args);
    TestableDataInit     dataInit(args, factory);

    auto p = makePlainProblem(64, 64, 64);

    // Prime and advance to slot 1
    dataInit.prepareGPUInputs(&p);
    dataInit.beginAsyncReset(&p);
    hipStream_t stream; HIP_CHECK_EXC(hipStreamCreate(&stream));
    dataInit.waitCopyDone(stream);
    dataInit.prepareGPUInputs(&p);
    EXPECT_EQ(dataInit.activeIdx(), 1u);

    // Simulate problem change
    dataInit.preProblem(nullptr);
    dataInit.cancelAsyncReset();

    EXPECT_EQ(dataInit.activeIdx(),      0u);
    EXPECT_EQ(dataInit.availableSlots(), 0u);
    EXPECT_FALSE(dataInit.altSlotsFilled());

    HIP_CHECK_EXC(hipStreamDestroy(stream));
}
```

---

## Additional thoughts

### What the tests collectively catch

| Regression | Caught by |
|-----------|-----------|
| `beginAsyncReset` not called in no-benchmark `main.cpp` path | A, B, C (fast path never fires) |
| `advanceBuffer()` broken (m_activeIdx not updated) | A (direct assertion) |
| `initializeAltBufferSets` skipped / early-return broken | C (data not 2.0f), A (altSlotsFilled false) |
| `m_altSlotsFilled` warm path records wrong event | A (availableSlots count wrong) |
| `cancelAsyncReset` fails to clear ring state | D |
| Alt slot holds corrupt or stale data | C |

### Optional production change: 2-line public accessors

If subclassing is undesirable, add to `DataInitialization.hpp`'s public section:

```cpp
size_t getActiveSlotIndex()   const { return m_activeIdx; }
size_t getAvailableSlots()    const { return m_availableSlots; }
```

This documents that these values are part of the observable ring contract. All Approach A tests
would then use these directly on `DataInitialization` without a subclass.

### File locations

| File | Action |
|------|--------|
| `tests/RingBuffer_test.cpp` | New file — contains all tests above |
| `tests/CMakeLists.txt` | Add `ring-buffer-test` target mirroring `batch-pointer-test` (same source list) |
| `client/include/DataInitialization.hpp` | No changes required; optional 2-line accessors |

### Recommended test ordering

1. Implement Approach A first — it gives the most direct signal and is easiest to debug.
2. Add Approach B as a public-API regression guard (survives future refactors).
3. Add Approach C after the fix is confirmed working — it is the behavioral correctness proof.
4. Add Approach D to round out the problem-boundary reset coverage.

### Note on `makePlainProblem`

The existing `makeBatchedProblem` helper creates a batched GEMM with `setStridedBatched(false)`.
For ring-buffer tests that focus on slot rotation rather than batch pointer correctness, a simpler
non-batched helper avoids the `initializeGPUBatchedInputs` complexity:

```cpp
ContractionProblemGemm makePlainProblem(size_t m, size_t n, size_t k)
{
    auto f32 = rocisa::DataType::Float;
    return ContractionProblemGemm::GEMM_Strides(
        false, false, f32, f32, f32, f32,
        m, n, k, /*batch=*/1,
        m, m*k, k, k*n, m, m*n, m, m*n, 0.0);
}
```
