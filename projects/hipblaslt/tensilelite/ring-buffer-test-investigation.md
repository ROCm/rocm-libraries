# Ring-Buffer Alt Slot Usage - Revised Test Plan

## Question

How do we test that ring-buffer alt slots are actually consumed in the no-benchmark path?

The behavior we need to prove has two distinct parts:

1. `DataInitialization` can rotate from slot 0 to an initialized alt slot when the ring is primed.
2. The no-benchmark production path actually primes the ring by calling `beginAsyncReset()` at the
   right point in the solution flow.

Tests that call `beginAsyncReset()` directly only prove the first point. They do not prove the
`main.cpp` no-benchmark regression is fixed.

---

## Context

The ring fast path in `DataInitialization::prepareGPUInputs(ContractionProblem const*)` fires when
`m_availableSlots > 0`. It calls `advanceBuffer()`, rotates `m_activeIdx`, updates the active GPU
pointer vectors, and returns `m_cachedGPUInputs`.

The bug under investigation is a production scheduling bug:

- In no-benchmark mode, `num-benchmarks == 0` or `num-enqueues-per-sync == 0` or
  `num-syncs-per-benchmark == 0`.
- That mode selects triple buffering with `m_numActiveBuffers = 3`.
- The existing `beginAsyncReset()` calls live inside the benchmark-loop body.
- In the no-benchmark path, that loop does not execute, so `m_availableSlots` stays 0.
- Slots 1 and 2 may be allocated and initialized, but they are never consumed by the ring fast path.

The intended fix is to submit `beginAsyncReset()` after warmup in the no-benchmark solution path, so
the next solution can consume an alt slot.

The test strategy must therefore include one test that observes the production scheduling decision,
not only tests that manually prime `DataInitialization`.

---

## Critique Of The Earlier Test Plan

The earlier A/B/C/D proposal had useful pieces, but it overclaimed what those tests proved.

### Manual Priming Does Not Test The Production Fix

Approaches A, B, and C called `dataInit.beginAsyncReset(&p)` directly. Those tests would still pass
if the `main.cpp` no-benchmark call-site fix were deleted.

They are valid `DataInitialization` integration tests, but they are not regression tests for
"`beginAsyncReset()` was not called in the no-benchmark production path."

### The Synchronization Order Was Wrong

The proposed snippets called `waitCopyDone(stream)` before the fast-path `prepareGPUInputs()` call.
That is not the production order.

Correct order:

```cpp
dataInit.beginAsyncReset(problem);
auto inputs = dataInit.prepareGPUInputs(problem); // calls advanceBuffer()
dataInit.waitCopyDone(stream);                    // waits for the newly active slot
// now the compute stream may use inputs
```

`advanceBuffer()` sets `m_computeNeedsCopyBarrier = true`. Calling `waitCopyDone()` before the
advance is a no-op and does not test the important event/barrier transition.

### Readback Alone Does Not Prove Slot Rotation

Reading tensor A from the returned inputs and expecting `2.0f` is not enough. Slot 0 and every alt
slot are initialized from the same source data. If the ring failed to advance and returned slot 0,
the readback would still pass.

Data validation must be paired with slot identity:

- Capture slot-0 pointer.
- Prime and advance the ring.
- Assert the returned pointer differs from slot 0.
- Then verify the returned alt-slot data is valid.

### Reset-Boundary Coverage Stopped Too Early

Checking that `cancelAsyncReset()` clears `m_availableSlots`, `m_activeIdx`, and
`m_altSlotsFilled` is not enough. The important behavior is that the next problem does not consume
stale alt-slot data from the previous problem.

A useful boundary test must switch from p1 to p2, prime an alt slot for p2, advance into it, and
verify the alt slot reflects p2.

### White-Box State Is Diagnostic, Not The Best Public Contract

A subclass exposing `m_activeIdx` and `m_availableSlots` is useful for debugging the state machine,
but it couples the test to implementation details. Public accessors for these fields would make
internal slot indices part of the API surface without a product-facing need.

If slot arithmetic deserves direct unit testing, extract a small HIP-free state machine and test it
directly.

---

## Recommended Test Shape

Use three layers.

### Layer 1: HIP-Free Slot State Machine Unit Tests

Best architecture seam: extract the slot accounting from `DataInitialization` into a small class,
for example `RingSlotController` or `AsyncSlotRing`.

This class should not know about HIP streams, events, tensors, or `ProblemInputs`. It should own
only the slot state:

```cpp
class RingSlotController
{
public:
    explicit RingSlotController(size_t activeBufferCount);

    size_t activeIndex() const;
    size_t availableSlots() const;
    bool   needsCopyBarrier() const;

    bool canPrime() const;
    size_t nextPrimeIndex() const;
    void markPrimed();

    size_t advance();
    void markBarrierWaited();
    void cancel();
};
```

The exact API can differ, but the state transitions should be isolated enough to test without a GPU.

Recommended tests:

| Test | What It Proves |
|------|----------------|
| `PrimeDoesNotOverfill` | Priming stops at `numActiveBuffers - 1`. |
| `AdvanceConsumesAvailableSlot` | Advancing increments active index and decrements available count. |
| `AdvanceWrapsModuloActiveBufferCount` | Slot rotation wraps from slot 2 to slot 0 in triple-buffer mode. |
| `AdvanceMarksCopyBarrierRequired` | A consumed slot requires a later copy-done wait. |
| `MarkBarrierWaitedClearsBarrier` | The compute-side wait clears the barrier flag. |
| `CancelResetsToSlotZero` | Cancel returns to slot 0, clears pending availability, and clears barrier state. |

These tests give deterministic coverage of the ring arithmetic and avoid GPU runtime cost.

If extracting this seam is too large for the current patch, keep one temporary white-box
`DataInitialization` test for `m_activeIdx` and `m_availableSlots`, but do not add public production
accessors solely for tests.

### Layer 2: `DataInitialization` GPU Integration Tests

These tests prove that the state machine is wired correctly to real GPU buffers and
`ProblemInputs`.

They may call `beginAsyncReset()` directly because their scope is `DataInitialization`, not
`main.cpp` scheduling. The test names and comments should say that explicitly.

#### Test 2.1: Fast Path Returns Distinct Valid Alt Slot

Purpose: prove that, once primed, `prepareGPUInputs()` returns a different GPU allocation and that
the alt allocation contains valid initialized data.

Sketch:

```cpp
TEST(RingBufferDataInit, FastPathReturnsDistinctValidAltSlot)
{
    auto args = buildRingArgs({{32, 32, 1, 32}}, InitMode::Two);

    ClientProblemFactory factory(args);
    DataInitialization   dataInit(args, factory);

    auto p = makePlainProblem(32, 32, 32);

    auto inputs0 = dataInit.prepareGPUInputs(&p);
    auto* ci0 = dynamic_cast<ContractionInputs*>(inputs0.get());
    ASSERT_NE(ci0, nullptr);
    ASSERT_NE(ci0->a, nullptr);
    void* slot0A = ci0->a;

    dataInit.beginAsyncReset(&p);

    auto inputs1 = dataInit.prepareGPUInputs(&p); // advances to slot 1

    hipStream_t stream;
    HIP_CHECK_EXC(hipStreamCreate(&stream));
    dataInit.waitCopyDone(stream);
    HIP_CHECK_EXC(hipStreamSynchronize(stream));

    auto* ci1 = dynamic_cast<ContractionInputs*>(inputs1.get());
    ASSERT_NE(ci1, nullptr);
    ASSERT_NE(ci1->a, nullptr);

    EXPECT_NE(slot0A, ci1->a)
        << "The ring fast path must return an alt-slot allocation.";

    std::array<float, 8> hostA{};
    HIP_CHECK_EXC(hipMemcpy(
        hostA.data(), ci1->a, hostA.size() * sizeof(float), hipMemcpyDeviceToHost));

    for(size_t i = 0; i < hostA.size(); ++i)
        EXPECT_EQ(hostA[i], 2.0f);

    HIP_CHECK_EXC(hipStreamDestroy(stream));
}
```

Notes:

- The important assertions are both pointer inequality and data validity.
- The wait happens after `prepareGPUInputs()` advances the ring.
- Use an RAII stream wrapper in real code so assertion failures do not leak the HIP stream.
- If `vv` is not in scope, define `using vv = po::variable_value;` in the helper or call site.

#### Test 2.2: Problem Change Refreshes Alt Slot Data

Purpose: prove `cancelAsyncReset()` prevents stale alt slots from a previous problem.

Use batched p1 and p2 with different strides. Slot 1 must reflect p2 after the problem switch.

Sketch:

```cpp
TEST(RingBufferDataInit, ProblemChangeRefreshesAltSlotData)
{
    constexpr size_t Batch = 4;

    auto p1 = makeBatchedProblem(32, 32, 32, Batch); // aStride = 1024
    auto p2 = makeBatchedProblem(64, 64, 64, Batch); // aStride = 4096

    auto args = buildRingArgs({{64, 64, Batch, 64}});
    ClientProblemFactory factory(args);
    DataInitialization   dataInit(args, factory);

    dataInit.prepareGPUInputs(&p1); // fills slot 0 and initializes p1 alt slots

    dataInit.preProblem(&p2);
    dataInit.cancelAsyncReset();

    dataInit.prepareGPUInputs(&p2); // prepares active slot for p2
    dataInit.beginAsyncReset(&p2);  // cold-fills an alt slot for p2

    auto inputs2Alt = dataInit.prepareGPUInputs(&p2); // advances into p2 alt slot

    hipStream_t stream;
    HIP_CHECK_EXC(hipStreamCreate(&stream));
    dataInit.waitCopyDone(stream);
    HIP_CHECK_EXC(hipStreamSynchronize(stream));

    auto* ci = dynamic_cast<ContractionInputs*>(inputs2Alt.get());
    ASSERT_NE(ci, nullptr);
    ASSERT_NE(ci->batchA, nullptr);

    void* batchA[Batch]{};
    HIP_CHECK_EXC(hipMemcpy(
        batchA, ci->batchA, Batch * sizeof(void*), hipMemcpyDeviceToHost));

    ptrdiff_t observedStride = static_cast<uint8_t*>(batchA[1])
                             - static_cast<uint8_t*>(batchA[0]);
    EXPECT_EQ(observedStride, ptrdiff_t(64 * 64))
        << "Alt slot must contain p2 batch pointers, not stale p1 pointers.";

    HIP_CHECK_EXC(hipStreamDestroy(stream));
}
```

This test is stronger than checking `m_altSlotsFilled == false` because it validates the observable
postcondition after the next alt-slot consumption.

#### Test 2.3: Targeted Ineligibility Guards

Add only the negative cases that protect likely regressions. Do not build a full combinatorial
matrix.

Recommended guards:

| Test | Setup | Expected Behavior |
|------|-------|-------------------|
| `BoundsCheckDoesNotPrimeRing` | `bounds-check != Disable` | `beginAsyncReset()` does not make the next `prepareGPUInputs()` return an alt pointer. |
| `ProblemDependentDataDoesNotFastPath` | Use `InitMode::SerialIdx`, `SerialDim0`, `Identity`, or another `IsProblemDependent()` mode | The ring does not bypass typed input preparation incorrectly. |
| `BenchmarkModeDoesNotUseNoBenchmarkPath` | Nonzero benchmark counts | The no-benchmark scheduling hook does not submit extra resets. |

Use black-box pointer behavior where possible. Use white-box availability counters only if there is
no stable observable behavior.

### Layer 3: Production Flow / Scheduler Test

This is the missing test in the earlier proposal.

Purpose: prove the no-benchmark solution path calls `beginAsyncReset()` after warmup. This cannot be
proven by directly calling `DataInitialization::beginAsyncReset()` in a unit test.

Recommended architecture seam:

```cpp
class RingResetSink
{
public:
    virtual ~RingResetSink() = default;
    virtual void beginAsyncReset(ContractionProblem const* problem) = 0;
};

void submitNoBenchmarkRingResets(RingResetSink& sink,
                                 ContractionProblem const* problem,
                                 bool noBenchmarkRuns)
{
    if(!noBenchmarkRuns)
        return;

    sink.beginAsyncReset(problem);
    sink.beginAsyncReset(problem);
}
```

The real implementation can be a smaller helper, a policy object, or part of a larger extracted
solution runner. The key is that the no-benchmark scheduling decision must be testable with a fake
or spy.

Recommended tests:

| Test | What It Proves |
|------|----------------|
| `NoBenchmarkPathSubmitsTwoRingResetsAfterWarmup` | The no-benchmark path primes both alt slots. |
| `BenchmarkPathDoesNotUseNoBenchmarkResetHook` | The hook is gated and does not perturb benchmark runs. |
| `ResetSubmissionHappensBeforePostSolutionOrNextPrepare` | Ordering matches the intended pipeline. |

Spy example:

```cpp
class SpyRingResetSink : public RingResetSink
{
public:
    void beginAsyncReset(ContractionProblem const* problem) override
    {
        calls.push_back(problem);
    }

    std::vector<ContractionProblem const*> calls;
};

TEST(RingBufferScheduling, NoBenchmarkPathSubmitsTwoRingResets)
{
    SpyRingResetSink sink;
    auto problem = makePlainProblem(1, 1, 1);

    submitNoBenchmarkRingResets(sink, &problem, true);

    ASSERT_EQ(sink.calls.size(), 2u);
    EXPECT_EQ(sink.calls[0], &problem);
    EXPECT_EQ(sink.calls[1], &problem);
}
```

The helper should not inspect the problem; it only forwards the pointer to the reset sink.

This layer is the actual regression guard for the no-benchmark call-site bug.

---

## Test Helpers

The existing `BatchPointerReset_test.cpp` has useful setup code, but it should not be copied into
every new GPU regression test.

Recommended helper extraction:

| Helper | Location | Notes |
|--------|----------|-------|
| `buildBaseDataInitArgs` | `tests/include/DataInitTestUtils.hpp` or similar | Shared complete `po::variables_map` defaults. |
| `buildRingArgs` | same helper header | Sets no-benchmark counts to zero and allows init-mode overrides. |
| `makePlainProblem` | same helper header | Non-batched GEMM for slot-rotation tests. |
| `makeBatchedProblem` | same helper header | Batched GEMM for stale batch-pointer tests. |
| `HipStreamGuard` | same helper header | RAII wrapper around `hipStreamCreate` / `hipStreamDestroy`. |

`buildRingArgs` should be compile-ready and support the init-mode override used by the data
readback test:

```cpp
po::variables_map buildRingArgs(std::vector<std::vector<size_t>> problemSizes,
                                InitMode initA = InitMode::Random)
{
    using vv = po::variable_value;

    auto args = buildBaseDataInitArgs(std::move(problemSizes));

    args["num-benchmarks"]          = vv(std::any(int(0)));
    args["num-enqueues-per-sync"]   = vv(std::any(int(0)));
    args["num-syncs-per-benchmark"] = vv(std::any(int(0)));
    args["num-elements-to-validate"] = vv(std::any(int(1)));
    args["init-a"]                  = vv(std::any(initA));

    return args;
}
```

Set `num-elements-to-validate` nonzero in tests that are meant to model a real no-benchmark
validation flow. Direct `DataInitialization` integration tests may not strictly need validation,
but using a realistic no-benchmark configuration reduces drift from production behavior.

---

## CMake Organization

Avoid adding a separate copied CMake target for every `DataInitialization` regression.

Preferred options:

1. Add a single client data-initialization GPU test target, for example
   `client-data-init-test`, containing `BatchPointerReset_test.cpp`,
   `RingBufferDataInit_test.cpp`, and shared helpers.
2. Or add `RingBufferDataInit_test.cpp` to the existing `batch-pointer-test` target and rename the
   target later when convenient.

Do not duplicate the long source list from `tests/CMakeLists.txt` unless there is a concrete reason
to split binaries.

---

## What The Final Tests Should Catch

| Regression | Test Layer |
|------------|------------|
| No-benchmark production path stops calling `beginAsyncReset()` | Layer 3 scheduler test |
| Ring slot arithmetic breaks | Layer 1 state-machine test |
| `advanceBuffer()` does not update returned `ProblemInputs` | Layer 2 distinct-pointer integration test |
| Alt slot contains uninitialized or corrupt tensor data | Layer 2 data-readback integration test |
| `waitCopyDone()` is ordered incorrectly relative to advancement | Layer 1 barrier state test plus Layer 2 production-order integration test |
| Problem switch serves stale p1 alt-slot data for p2 | Layer 2 two-problem batched stale-slot test |
| Bounds-check or problem-dependent data incorrectly uses ring fast path | Layer 2 targeted negative tests |
| Benchmark path is perturbed by no-benchmark reset hook | Layer 3 benchmark-mode negative test |

---

## Recommended Implementation Order

1. Add or extract shared `DataInitialization` test helpers.
2. Add the production scheduling seam and spy test. This is the required regression guard for the
   no-benchmark call-site fix.
3. Add the `DataInitialization` distinct-valid-alt-slot integration test using the correct
   `prepareGPUInputs()` then `waitCopyDone()` order.
4. Add the two-problem stale-alt-slot test.
5. Extract the HIP-free ring slot controller if the implementation is still changing or if more
   state-machine cases are needed.
6. Add targeted negative tests for the highest-risk `ringEligible()` guards.

This keeps the GPU-heavy tests small and makes the highest-value regression test the one that
actually observes the production behavior that failed.
