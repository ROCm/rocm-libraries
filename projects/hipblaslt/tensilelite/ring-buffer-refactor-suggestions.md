# Ring-Buffer Maintainability And Testability Refactors

## Goal

This note captures refactors that complement the planned `RingSlotController` extraction.

`RingSlotController` isolates slot arithmetic, but the broader maintainability issue is that
`DataInitialization` and `main.cpp` still encode several lifecycles implicitly through mutable
state, direct HIP calls, and call-site ordering discipline.

The recommendations below are ordered by impact. The intent is to create seams that let most
behavior be tested with focused unit or fake-based tests, keeping GPU-heavy integration tests small.

---

## 1. Extract A Testable Run/Lifecycle Scheduler From `main.cpp`

### Problem

`main.cpp` owns too many sequencing responsibilities in one nested loop:

- Problem lifecycle.
- Solution lifecycle.
- Input preparation.
- Async reset cancellation and submission.
- Copy-barrier waiting.
- Kernel solving.
- Warmup execution and validation.
- Benchmark enqueueing.
- Profiler hooks.
- User-args cleanup.

The ring-buffer production behavior is therefore hard to test without running the whole client flow
on GPU hardware.

Relevant code:

- `client/main.cpp`: problem setup and `listeners.preProblem(problem)`.
- `client/main.cpp`: `dataInit->cancelAsyncReset()`.
- `client/main.cpp`: initial `dataInit->prepareGPUInputs(problem)`.
- `client/main.cpp`: solution loop and `listeners.needMoreRunsInSolution()`.
- `client/main.cpp`: `dataInit->waitCopyDone(stream)`.
- `client/main.cpp`: `dataInit->beginAsyncReset(problem)` calls.

### Refactor

Extract a `SolutionRunExecutor` or `ClientRunScheduler`.

Keep option parsing and object construction in `main.cpp`. Move sequencing into the executor.

Suggested collaborators:

```cpp
class InputPipeline;
class SolutionProvider;
class KernelLauncher;
class RunEvents;
class Reporter;
```

The executor should make ordering explicit:

```cpp
executor.runProblem(problem);
executor.runSolution(solution);
executor.runWarmups();
executor.runBenchmarks();
executor.finishSolution();
```

### Tests Enabled

Use fakes/spies to verify:

- Problem change calls cancel before input preparation.
- `prepareGPUInputs()` happens before `waitCopyDone()`.
- `waitCopyDone()` happens before the first launch.
- No-benchmark mode submits the ring reset hook at the intended lifecycle point.
- Benchmark mode does not submit the no-benchmark reset hook.

This is the highest-value seam for proving the no-benchmark fix.

---

## 2. Make `DataInitialization` Problem/Solution Lifecycle Explicit

### Problem

`DataInitialization` is both a `RunListener` and a directly called input service.

Some required state changes happen through listener callbacks:

- `preProblem()` resets `m_batchInitProblem`.
- `preSolution()` stores solution-dependent MX state and may refresh MX inputs.

Other required state changes happen through direct calls:

- `cancelAsyncReset()`.
- `prepareGPUInputs()`.
- `waitCopyDone()`.
- `beginAsyncReset()`.

Correctness depends on call-site discipline across both APIs.

### Refactor

Introduce explicit lifecycle methods:

```cpp
void beginProblem(ContractionProblem* problem);
void beginSolution(ContractionSolution* solution);
std::shared_ptr<ProblemInputs> prepareGpuInputs();
void primeNextInputSlot();
void waitForPreparedSlot(hipStream_t stream);
void endProblem();
```

Initially these can delegate to existing callback/direct-call methods. The important change is that
the caller should not need to know that `preProblem()` resets batch state while
`cancelAsyncReset()` clears ring slots.

### Tests Enabled

- Problem transition resets both batch-pointer and ring-slot state.
- Solution transition refreshes solution-dependent MX state before inputs are consumed.
- Calling lifecycle methods in the intended order is enough; tests do not need to mix listener and
  service APIs manually.

---

## 3. Separate Slot Storage/Activation From Slot Arithmetic

### Problem

`RingSlotController` will own index/count/barrier arithmetic, but slot storage is still represented
in several places:

- `gpuInput.current`.
- `gpuInput.batch`.
- `m_gpuPtrs`.
- `m_gpuBatchPtrs`.
- `m_cachedGPUInputs`.
- `m_gpuPtrsRing[]`.
- `m_gpuBatchPtrsRing[]`.
- `m_cachedInputsRing[]`.

`advanceBuffer()` updates several aliases at once. `SlotGuard` temporarily mutates
`gpuInput.current` and `gpuInput.batch` so existing copy paths can be reused.

### Refactor

Add `GpuInputSlotSet` or `InputSlotRepository`.

Suggested model:

```cpp
struct GpuInputSlot
{
    std::vector<void*> ptrs;
    std::vector<void**> batchPtrs;
    std::shared_ptr<ProblemInputs> cachedInputs;
};

class GpuInputSlotSet
{
public:
    GpuInputSlot& slot(size_t index);
    GpuInputSlot& active();
    void activate(size_t index);
    void clearAltSlots();
};
```

`DataInitialization` should ask this repository for the active slot instead of keeping parallel
mutable aliases.

### Tests Enabled

- Activation updates the active pointer view exactly once.
- Clearing alt slots invalidates only nonzero slots.
- Slot identity can be tested with fake pointer tokens instead of HIP allocations.

---

## 4. Extract The Output-Reset Contract From Copy Mechanics

### Problem

Output reset behavior is spread across:

- `prepareGPUInputsInternal()`.
- `beginAsyncReset()`.
- `fillSlot()`.
- `resetOutput()`.

This hides policy decisions inside copy mechanics. The risky contract is not the copy itself, but
when output tensors must be reset before a slot is consumed.

### Refactor

Introduce `OutputResetPlanner` or `OutputInitializationPolicy`.

It should answer:

```cpp
enum class SlotPreparationAction
{
    NoOutputResetRequired,
    ResetOutputsFromValid,
    FullSlotFillRequired,
};
```

Inputs should include:

- Validation enabled.
- Bounds-check mode.
- Problem-dependent data.
- Whether the slot is known pristine.
- Relevant beta/output-write assumptions.

Use the same planner from normal fast path and ring-buffer paths.

### Tests Enabled

- Validation-on mode requests output reset where required.
- Pristine alt slot can skip reset only under documented assumptions.
- Bounds-check and problem-dependent modes do not accidentally take the wrong path.

Keep one GPU integration test proving the planned operation reaches the right buffer.

---

## 5. Introduce A HIP Copy/Stream/Event Abstraction With RAII

### Problem

`DataInitialization` directly creates and manages HIP streams/events, records events, waits on
events, synchronizes streams, allocates memory, and performs copies.

That forces behavior tests to require HIP even when the behavior under test is only:

- Which copy was submitted.
- Which event was recorded.
- Whether a stream waited on the right event.
- Whether cancellation synchronized before clearing state.

### Refactor

Define a narrow runtime abstraction:

```cpp
class GpuCopyEngine
{
public:
    virtual ~GpuCopyEngine() = default;

    virtual void memcpy(void* dst, void const* src, size_t bytes, hipMemcpyKind kind) = 0;
    virtual void memcpyAsync(void* dst,
                             void const* src,
                             size_t bytes,
                             hipMemcpyKind kind,
                             hipStream_t stream) = 0;
    virtual void recordEvent(hipEvent_t event, hipStream_t stream) = 0;
    virtual void waitEvent(hipStream_t stream, hipEvent_t event) = 0;
    virtual void synchronize(hipStream_t stream) = 0;
};
```

Provide:

- `HipCopyEngine` for production.
- `RecordingCopyEngine` for tests.
- RAII wrappers for streams, events, and pinned staging memory.

### Tests Enabled

- Async reset records the expected event.
- `waitCopyDone()` waits only after advancement.
- `cancelAsyncReset()` synchronizes before clearing pending state.
- Tests do not leak HIP resources on assertion failure.

---

## 6. Extract Batch Pointer Layout Calculation From GPU Upload

### Problem

`initializeGPUBatchedInputs()` mixes several responsibilities:

- Batch-index selection.
- Bounds-check padding.
- Bias-source special cases.
- Sparse metadata/compressed tensor handling.
- Pinned staging buffer mutation.
- HIP upload.

The existing stale batch-pointer tests are GPU-heavy because offset calculation and upload are not
separable.

### Refactor

Extract a pure `BatchPointerLayout`.

Suggested API:

```cpp
std::vector<ptrdiff_t> computeBatchPointerOffsets(BatchPointerLayoutRequest const& request);
```

The request should include:

- Tensor descriptor.
- Base pointer token or base offset.
- Batch indices.
- Bounds-check mode.
- Padding/swizzle metadata.
- Tensor role: A, B, C, D, bias, metadata, compressed.

Keep `BatchPointerUploader` responsible for writing computed pointers to device memory.

### Tests Enabled

- Normal batched pointer offsets.
- Bounds-check front/back/NaN padding offsets.
- Bias-source handling.
- Sparse metadata/compressed offsets.
- Problem-switch p1 to p2 stale-pointer prevention without GPU.

Keep one GPU smoke test for upload/readback wiring.

---

## 7. Split Copy Planning From Copy Execution

### Problem

`copyInputs()` and `resetOutput()` both:

- Iterate tensors.
- Decide source and destination buffers.
- Handle bounds-check behavior.
- Perform copies.
- Mutate pointer/max-element/offset vectors.

`fillSlot()` has to defensively copy and clear offsets because `copyInputs()` mutates caller-owned
vectors in non-obvious ways.

### Refactor

Create a pure planner:

```cpp
struct TensorCopyOp
{
    size_t tensorIndex;
    TensorBufferRole source;
    TensorBufferRole destination;
    size_t bytes;
    BoundsCheckMode boundsMode;
    bool outputOnly;
    hipMemcpyKind copyKind;
};

std::vector<TensorCopyOp> planTensorCopies(TensorCopyRequest const& request);
```

Then execute through `GpuCopyEngine`.

If the current methods remain during migration, rename them around intent:

- `copyInputs()` -> `copyAllTensorInputsAndPopulateViews()`.
- `resetOutput()` -> `resetOutputViewsFromValid()`.

### Tests Enabled

- Branch-heavy copy behavior becomes pure operation-planning tests.
- Device integration coverage can shrink to a small set of executor smoke tests.
- Slot fill code no longer needs defensive local-vector workarounds.

---

## 8. Introduce `DataInitConfig` And Shared Test Builders

### Problem

`DataInitialization` is constructed directly from `po::variables_map`. Tests must recreate a large
option map to exercise one focused behavior.

This makes tests brittle: adding an unrelated option can break focused data-init tests.

### Refactor

Add a typed config:

```cpp
struct DataInitConfig
{
    bool stridedBatched = false;
    int sparse = 0;
    bool cEqualsD = false;
    int elementsToValidate = 0;
    bool keepPristineCopyOnGPU = true;
    BoundsCheckMode boundsCheck = BoundsCheckMode::Disable;
    int numBenchmarks = 0;
    int numEnqueuesPerSync = 0;
    int numSyncsPerBenchmark = 0;
    // Continue with fields currently read from po::variables_map.

    static DataInitConfig fromArgs(po::variables_map const& args);
};
```

Give `DataInitialization` an overload:

```cpp
DataInitialization(DataInitConfig const& config,
                   ClientProblemFactory const& problemFactory);
```

Move shared test setup into `tests/include/DataInitTestUtils.hpp`:

- `buildBaseDataInitConfig()`.
- `makePlainProblem()`.
- `makeBatchedProblem()`.
- `HipStreamGuard`.

### Tests Enabled

- Focused tests can set one field without constructing a full CLI option map.
- Defaults live in one place.
- The ring-buffer tests and batch-pointer tests can share setup.

---

## 9. Move MX/Swizzle Solution-Specific Logic Behind A Policy Object

### Problem

MX scale initialization depends on:

- Architecture.
- Problem shape.
- User options.
- Selected solution.

The selected solution is currently ambient state set by `preSolution()`. Copy/swizzle logic later
reads that state indirectly.

### Refactor

Extract `MxInputSpecializer` or `MxScaleLayoutPolicy`.

Suggested API:

```cpp
MxInputPlan specializeMxInputs(ContractionProblemGemm const& problem,
                               ContractionSolution const* solution,
                               ArchitectureInfo const& arch,
                               DataInitConfig const& config);
```

The policy should produce a data/copy plan instead of mutating `DataInitialization` based on
`m_currentSolution`.

### Tests Enabled

- `mxScaleFormat` decisions are unit-testable.
- Architecture-dependent preswizzle decisions are explicit.
- Ring slots cannot accidentally consume a pre-solution MX layout without a testable plan.

---

## 10. Separate Rotating-Output And I-Cache Rotation Policy From The Core Run Loop

### Problem

The client has several adjacent but distinct rotation concepts:

- Ring-buffer input slots.
- Rotating output/input buffers.
- I-cache code-object rotation.

These are currently adjacent in `main.cpp`, which makes ring-buffer reasoning harder.

### Refactor

Extract:

- `RotatingInputPlanner` for `prepareRotatingGPUOutput()` sizing and selection.
- `ICacheRotationPolicy` for `extras = max(dataRotationExtras, cacheOverflowExtras)`.

Keep `DataInitialization` responsible for actual buffers only. Keep `main.cpp` from mixing
ring-slot scheduling with code-object rotation policy.

### Tests Enabled

- Rotating buffer size/selection tests do not need ring-buffer setup.
- I-cache extra-copy decisions are pure unit tests.
- Ring-buffer scheduling tests are not polluted by unrelated rotation behavior.

---

## Suggested Implementation Order

1. Add `DataInitConfig` and shared data-init test builders. This lowers friction for every later
   test.
2. Extract `RingSlotController` and `GpuInputSlotSet`. This addresses current ring-buffer fragility
   directly.
3. Extract `SolutionRunExecutor` or `ClientRunScheduler`. This gives a real production-flow test for
   the no-benchmark fix.
4. Add `GpuCopyEngine` and RAII HIP wrappers. This removes HIP from event/copy ordering tests.
5. Extract `BatchPointerLayout` and `OutputResetPlanner`. These target known correctness risks.
6. Extract MX/swizzle and rotating/I-cache policies. These are valuable, but less central to the
   immediate ring-buffer testability issue.
