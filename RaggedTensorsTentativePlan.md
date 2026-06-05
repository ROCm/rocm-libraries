Here's the current state of the plan for the `Tensors.hpp` portion of the Ragged Tensors RFC.

The two configurations targeted in this iteration are the ones AITER kernels actually use:

- **Packed with `ragged_offset` only.**
- **Packed with `ragged_offset` + `seq_lens`.**

A "padded, seq-lens-only" mode (no `ragged_offset`) is intentionally deferred. It is reachable later by adding a constructor (or a tag-typed ctor) without breaking the API designed here.

This document has been updated to fold in the findings of `RaggedTensorsStageByStageAnalysis.md` (a per-stage walk through the user-facing sample flow and the integration-test harness flow). Companion documents:

- `RaggedTensorsTensorFlows.md` — neutral description of the two flow paths in the codebase, with no A vs C analysis.
- `RaggedTensorsStageByStageAnalysis.md` — per-stage analysis of where A and C diverge, with the constraints and trade-offs that this plan now reflects.

---

## Important constraint: `seq_lens` is NOT on `TensorAttributes`

For cuDNN frontend-API compatibility, **`TensorAttributes` will gain `set_ragged_offset(...)` only — not `set_seq_len(...)`**. The `seq_lens` tensor is referenced from the *node-level* op attributes (e.g. `SdpaAttributes::set_seq_len_q(...)` and `set_seq_len_kv(...)`), not from the primary tensor's `TensorAttributes`. In the SDPA case, a single `seq_lens_q` tensor is shared across the Q primary and a single `seq_lens_kv` is shared across the K and V primaries — so there is no 1:1 binding between a primary `TensorAttributes` and a `seq_lens` tensor.

This constraint has consequences that ripple through several of the design decisions below:

- Code that walks the graph one `TensorAttributes` at a time (notably the test-harness's `createTensorFromAttribute` / `generateBundles` / `tryAddTensorToBundles` path, and the CPU executor's virtual-tensor allocation) can discover whether a tensor is a ragged primary and which tensor is its `ragged_offset`, but **cannot** discover which `seq_lens` it is paired with. That requires walking the node-level op attributes.
- The plan layer for each op (e.g. `SdpaFwdPlanBuilder`) already has access to the node attributes and *can* resolve `seq_lens`. That is the natural place to attach `seq_lens` to a ragged view.
- The harness's bundle layer therefore should not try to attach `seq_lens` to ragged primaries at all. Ragged primaries in the bundle are constructed without a `seq_lens`; the plan layer attaches it when the view is built for the CPU reference call.

The design below reflects this constraint consistently.

---

## Shared elements (apply to both options below)

1. **Aux tensors are `std::shared_ptr<TensorBase<IndexT>>`**, with `IndexT` templated and defaulting to `int32_t` (and `int64_t` permitted).
2. **Auxiliaries are constructor-only / immutable** on the ragged object. The ragged object never mutates which aux tensors it points at.
3. **Single packed-mode shape**: `ragged_offset` is **required**; `seq_lens` is **optional** (`nullptr` permitted, and is in fact null in the harness's bundle layer per the constraint above). No tag types — there is currently only one ragged mode to construct.
4. **`dims()[1] == S_max`** by construction. `elementSpace()` is decoupled from `prod(dims)` and equals the physical buffer size of the underlying storage.
5. **Polymorphic index-strategy hook on `ITensor`**: introduce `virtual std::unique_ptr<ITensorIndex> makeIndex(bool isEnd) const` so ragged iteration can supply a `RaggedCompositeIndex` that walks only valid elements (skipping padding and, when `seq_lens` is present, the per-batch invalid tail). The existing `LinearIndex` / `CompositeIndex` strategies move behind this interface; non-ragged behaviour is unchanged. `ITensorIterator`'s `std::variant` does not grow.
6. **Optional `virtual int64_t validSeqLen(int64_t b) const` on `TensorBase`** (default `dims()[1]`) so CPU references can bound inner loops without `dynamic_cast`.
7. **Iteration semantics for `elementCount()`**: if `seq_lens != nullptr`, `elementCount()` returns the sum of valid elements implied by `seq_lens`. If `seq_lens == nullptr`, `elementCount()` returns the underlying physical element count.

---

## Option A — `RaggedTensor<T, IndexT>` as a self-owning `TensorBase<T>` subclass (retained for analysis, **not recommended**)

**Class shape (packed-only, after this iteration's scope reduction)**

```cpp
template <typename IndexT>
using RaggedIndexTensor = std::shared_ptr<TensorBase<IndexT>>;

template <typename T,
          typename IndexT      = int32_t,
          typename HostAlloc   = HostAllocator<T>,
          typename DeviceAlloc = DeviceAllocator<T>>
class RaggedTensor : public TensorBase<T>
{
public:
    // Packed mode: physical buffer size is given explicitly. raggedOffset is required;
    // seqLens is optional (covers both "ragged_offset only" and "ragged_offset + seq_lens").
    RaggedTensor(std::vector<int64_t>      paddedDims,
                 std::vector<int64_t>      strides,
                 size_t                    physicalElementCount,
                 RaggedIndexTensor<IndexT> raggedOffset,
                 RaggedIndexTensor<IndexT> seqLens = nullptr);

    // ITensor / TensorBase<T> overrides:
    //   dims()         -> padded dims (dims()[1] == S_max)
    //   elementSpace() -> physicalElementCount
    //   elementCount() -> sum-of-seqLens if seqLens, else physicalElementCount
    //   isPacked()     -> false
    //   begin/end      -> RaggedCompositeIndex strategy (walks `[0, ragged_offset[B])`
    //                     when seqLens == nullptr; walks valid-only when seqLens is set)

    bool hasRaggedOffset() const;            // always true in this iteration
    bool hasSeqLens()      const;
    const TensorBase<IndexT>* raggedOffset() const;
    const TensorBase<IndexT>* seqLens()      const;
    int64_t validSeqLen(int64_t b) const;    // S_max if no seqLens

private:
    MigratableMemory<T, HostAlloc, DeviceAlloc> _memory;
    std::vector<int64_t>                        _paddedDims;
    std::vector<int64_t>                        _strides;
    size_t                                      _physicalElementCount;
    RaggedIndexTensor<IndexT>                   _raggedOffset;   // required, non-null
    RaggedIndexTensor<IndexT>                   _seqLens;        // nullable; null in the harness's bundle layer
};
```

**Key properties**
- Memory is self-owned through `MigratableMemory<T, HostAlloc, DeviceAlloc>`, mirroring `Tensor<T>`. `PinnedTensor<T>` equivalence is reached via `RaggedTensor<T, IndexT, PinnedHostAllocator<T>>`.
- `dims()[1] == S_max`; `elementSpace()` is decoupled from `prod(dims)`.
- No UID is stored on the tensor.
- Ctor-time structural validation: ranks, sizes, and `aux->elementCount() == B+1` / `B` are checked at construction so a `RaggedTensor` cannot exist in an invalid state.

**Costs of A under the seq_lens constraint**

Because `seq_lens` is not on `TensorAttributes`, the test-harness's bundle-allocation pass (`generateBundles` → `createTensorFromAttribute`) cannot attach a `seq_lens` to the `RaggedTensor` it constructs. The bundle therefore holds `RaggedTensor<T>` with `seqLens == nullptr`. This is correct for bundle-level operations (randomize, dense iteration), but it means A loses the "free valid-element iteration" property at the bundle layer — its iterator walks `[0, ragged_offset[B])`, which is functionally equivalent to "iterate the physical buffer" in the harness's randomize/compare uses.

It also introduces two harness-side mechanical costs:

- **Two-pass bundle allocation.** Because A's ctor requires non-null `raggedOffset`, the bundle must allocate the aux `ragged_offset` tensor *before* allocating the primary `RaggedTensor`. The current one-pass walk in `tryAddTensorToBundles` has no deterministic visit order between primaries and their aux tensors, so the harness has to be restructured into a two-pass loop.
- **`unique_ptr → shared_ptr` shift in `GraphTensorBundle`.** Today the bundle holds `unordered_map<int64_t, unique_ptr<ITensor>>`. A's `RaggedTensor` ctor takes the aux as a `shared_ptr<TensorBase<IndexT>>`, which requires either widening the bundle's storage to `shared_ptr` or introducing a `share()` accessor that aliases into it.

**No `resolveTensor` reconciler in the test-harness flow**

An earlier draft of this plan introduced a `resolveTensor<T, IndexT>(primaryUid, raggedOffsetUid, seqLensUid, tensorMap, role)` helper in the plan layer to verify the runtime `RaggedTensor`'s aux pointers against the graph's declared aux UIDs. **That helper is degenerate in the test-harness flow** for two reasons:

1. The variant pack at the executor / plan boundary is `unordered_map<int64_t, void*>` — there is no `RaggedTensor` to read aux pointers from.
2. With the seq_lens constraint, the "graph's declared seq_lens UID" lives on the *node*, not the tensor, so the reconciler's data shape doesn't match either.

The reconciler made sense only in a hypothetical user-facing flow where the user constructs a `RaggedTensor` with explicit aux refs and the framework wants to check those against the graph's declaration. No such code path currently exists. **Drop the reconciler from the current scope.** If A is ever revisited as a user-facing convenience type, the reconciler can be added at that time.

**Output-validation infrastructure A needs**

The test harness's `verifyGraph` validation loop compares ragged output tensors element-by-element between the CPU and GPU bundles. Because GPU kernels may write to output padding while the CPU reference does not, the validator must use seq_lens-bounded iteration (or the sentinel-skip approach described under *Shared output-validation strategy* below). If the seq_lens-bounded route is taken, A's bundle holds `RaggedTensor<T>` with `seqLens == nullptr` — and attaching a `seqLens` for the compare requires **additional `RaggedTensor` infrastructure** that the plan as written does not provide:

- A second `RaggedTensor` ctor that takes existing memory + new aux refs (aliasing-share with the bundle's storage), **or**
- A mutator `setSeqLens(...)` (violating the immutability rule), **or**
- A wider `allClose` validator signature that takes an explicit `seqLens` parameter.

The sentinel-skip approach (recommended below) avoids this cost entirely in both A and C.

**Summary of A's case**

A satisfies every functional requirement in the RFC. The original framing ("A's central cost is the triplicated wiring / reconciler") is overweighted now that the reconciler is degenerate and the seq_lens constraint has neutralized A's "free valid-element iteration" advantage. A's remaining real costs at the test-harness layer are:

- Two-pass bundle allocation in `generateBundles`.
- `unique_ptr → shared_ptr` shift in `GraphTensorBundle`.
- Additional `RaggedTensor` infrastructure if seq_lens-bounded output validation is chosen (avoided by sentinel-skip).

A's remaining advantage over C is the **user-facing single-object ergonomic**: a user (in a sample like `BnInference.cpp`) holds one owning `RaggedTensor` value rather than C's split `(IrregularTensor storage, optional RaggedView)` pair.

---

## Option C — `RaggedView<T, IndexT>` over `IrregularTensor<T>`, constructed in the plan layer (**recommended**)

This option introduces two narrow types and pushes all ragged-aux wiring down to a single source of truth at each layer: the graph carries the `ragged_offset` wiring, the node-level op attributes carry the `seq_lens` wiring, and the plan layer reads both to assemble a `RaggedView` at the moment it's needed. The runtime tensorMap carries only dense storage and dense aux buffers.

### Type 1 — `IrregularTensor<T>`

A memory-owning buffer whose physical element count is independent of `prod(dims)`, used as the underlying storage for ragged views and as the natural type for graph intermediates that don't need iteration.

```cpp
template <typename T,
          typename HostAlloc   = HostAllocator<T>,
          typename DeviceAlloc = DeviceAllocator<T>>
class IrregularTensor : public TensorBase<T>
{
public:
    IrregularTensor(std::vector<int64_t> paddedDims,
                    std::vector<int64_t> strides,
                    size_t               physicalElementCount);

    // ITensor / TensorBase<T> overrides:
    //   dims()         -> paddedDims
    //   strides()      -> strides
    //   elementSpace() -> physicalElementCount
    //   elementCount() -> physicalElementCount  (NOT prod(dims))
    //   isPacked()     -> false
    //
    // Direct addressing (rawHostData / rawDeviceData / hostDataOffsetFromIndex(linearOffset))
    //   is fully supported. This is how RaggedView and any executor address the buffer.
    //
    // Iteration walks the physical buffer linearly:
    //   begin()/end()/cbegin()/cend() iterate `physicalElementCount` contiguous elements,
    //   treating the buffer as a flat `prod(dims)`-less dense buffer. This is the natural
    //   semantics for the harness's `bundle.randomizeTensor(uid)` and `validator.allClose(*cpu, *gpu)`
    //   call sites, neither of which has access to the `seq_lens` needed for ragged-aware
    //   iteration (per the constraint at the top of this document). Ragged-aware iteration
    //   that respects per-batch valid lengths is the job of `RaggedView<T>`, not this type.

private:
    MigratableMemory<T, HostAlloc, DeviceAlloc> _memory;
    std::vector<int64_t>                        _paddedDims;
    std::vector<int64_t>                        _strides;
    size_t                                      _physicalElementCount;
};
```

**Pinned host memory** is supported via `IrregularTensor<T, PinnedHostAllocator<T>, …>`, exactly the same trick `PinnedTensor<T>` uses for `Tensor<T>`.

> **Note on the iteration policy.** An earlier draft of this plan had `IrregularTensor`'s iterators throw `std::logic_error` to rule out accidental walks over a buffer that has no well-defined dense iteration order. Under the seq_lens constraint, this turned out to force the test harness into a wrapping step (look up `seq_lens` from node attributes, build a `RaggedView`, iterate ragged-aware) at randomize/compare time that is unnecessary — the harness can't do the lookup at all without major restructuring, and the natural fallback (iterate the full physical buffer) is correct for the randomize use case as long as both bundles are seeded identically. The current policy is therefore linear iteration over the physical buffer.

### Type 2 — `RaggedView<T, IndexT>`

A non-owning wrapper that turns any `TensorBase<T>`-derived underlying storage into a fully iterable ragged tensor by attaching the required aux buffers.

```cpp
template <typename T, typename IndexT = int32_t>
class RaggedView : public TensorBase<T>
{
public:
    // Shape, strides, and physicalElementCount are NOT passed in: they come
    // authoritatively from the underlying tensor's dims()/strides()/elementSpace().
    // This removes a whole class of "view disagrees with its underlying" bugs by
    // construction.
    //
    // The underlying is `shared_ptr<TensorBase<T>>` rather than `shared_ptr<IrregularTensor<T>>`
    // so the view can wrap either:
    //   - An owning `IrregularTensor<T>` (the bundle / executor intermediate case), OR
    //   - A non-owning `ShallowTensor<T>` (the plan layer's per-execute case, where the
    //     underlying is a wrapper over a `void*` from the variant pack).
    //
    // (An earlier draft of this plan typed the underlying as
    // `shared_ptr<IrregularTensor<T>>`, but the plan layer at Stage 2.6 of the
    // integration-test harness does not have an `IrregularTensor` at the point where it
    // needs to build the view — it has a `void*` from `variantPack` and the cached
    // `(dims, strides, physicalCount)` from its plan params. The wider underlying type
    // accommodates both call sites without introducing a third tensor type.)
    RaggedView(std::shared_ptr<TensorBase<T>>       underlying,
               std::shared_ptr<TensorBase<IndexT>>  raggedOffset,                  // required
               std::shared_ptr<TensorBase<IndexT>>  seqLens     = nullptr);        // optional

    // ITensor / TensorBase<T> overrides:
    //   dims()         -> _underlying->dims()         (forwarded)
    //   strides()      -> _underlying->strides()      (forwarded)
    //   elementSpace() -> _underlying->elementSpace() (forwarded)
    //   memory()       -> _underlying->memory()       (forwarded)
    //   elementCount() -> sum-of-seqLens if _seqLens, else _underlying->elementSpace()
    //   isPacked()     -> false
    //   begin/end      -> RaggedCompositeIndex strategy via the polymorphic index hook

    // Ragged accessors used by tests / verification:
    bool hasRaggedOffset() const;              // always true
    bool hasSeqLens()      const;
    const TensorBase<IndexT>* raggedOffset() const;
    const TensorBase<IndexT>* seqLens()      const;
    int64_t validSeqLen(int64_t b) const;      // dims()[1] if no seqLens

private:
    std::shared_ptr<TensorBase<T>>       _underlying;
    std::shared_ptr<TensorBase<IndexT>>  _raggedOffset;   // required, non-null
    std::shared_ptr<TensorBase<IndexT>>  _seqLens;        // nullable
};
```

**Ctor-time structural validation**
- `_underlying != nullptr`, `_raggedOffset != nullptr`.
- `_raggedOffset->elementCount() == B + 1`, where `B = _underlying->dims()[0]`.
- If `_seqLens != nullptr`: `_seqLens->elementCount() == B`.
- `_raggedOffset` and `_seqLens` (if present) have rank 1.

No cross-checks against redundant shape arguments are needed, because there are no redundant shape arguments.

### Tiered API: what each role uses

| Role | Type | Iteration |
|---|---|---|
| Owning storage for a ragged primary in the bundle / executor intermediates | `IrregularTensor<T>` | Linear over physical buffer |
| Ragged inputs / outputs of an op at the plan layer | `RaggedView<T, IndexT>` over a `ShallowTensor<T>` of the variant-pack pointer | Full ragged iteration |
| Ragged tensor a user iterates directly (sample flow) | `RaggedView<T, IndexT>` over the owning `IrregularTensor<T>` | Full ragged iteration |
| Aux tensors (`ragged_offset`, `seq_lens`) | Plain `Tensor<IndexT>` via `shared_ptr` | Dense iteration |
| Graph intermediates that no one walks | `IrregularTensor<T>` directly | Linear over physical buffer (incidental) |
| Non-ragged tensors | Plain `Tensor<T>` | Dense iteration |

### Wiring: source-of-truth allocation

- **Graph (`TensorAttributes`)**: gains `set_ragged_offset(...)` only (per the seq_lens constraint). The flatbuffer gains `ragged_offset_tensor_uid: long = null` (per the original RFC).
- **Node-level op attributes** (e.g. `SdpaAttributes`): retain their existing `set_seq_len_q(...)` / `set_seq_len_kv(...)` accessors. The flatbuffer's per-op attribute tables already carry the corresponding `seq_len_*_tensor_uid` fields.
- **Runtime tensorMap (`UID → ITensor`)**: carries only the dense buffers — the `IrregularTensor<T>` that holds each packed ragged primary, ordinary `Tensor<IndexT>` for each aux, ordinary `Tensor<T>` for non-ragged tensors. **No ragged-view objects at this layer.**
- **Plan layer (executor)**: when executing an op whose graph declares a given input/output as ragged, the plan constructs the view at the call site, reading `ragged_offset` from the primary's `TensorAttributes::ragged_offset_tensor_uid()` and `seq_lens` from the node's op-specific attribute (e.g. `SdpaAttributes::seq_len_q_tensor_uid()`):

    ```cpp
    auto qUnderlying = std::make_shared<ShallowTensor<QType>>(
        variantPack.at(_params.qTensor.uid),
        _params.qTensor.dims,
        _params.qTensor.strides);
    auto qRaggedOffset = std::make_shared<ShallowTensor<IndexT>>(
        variantPack.at(_params.qRaggedOffsetTensor.uid),
        _params.qRaggedOffsetTensor.dims,
        _params.qRaggedOffsetTensor.strides);
    auto qSeqLens = _params.seqLenQTensor.has_value()
        ? std::make_shared<ShallowTensor<IndexT>>(
              variantPack.at(_params.seqLenQTensor->uid),
              _params.seqLenQTensor->dims,
              _params.seqLenQTensor->strides)
        : nullptr;

    auto qView = std::make_shared<RaggedView<QType, IndexT>>(
        qUnderlying, qRaggedOffset, qSeqLens);

    // ... pass *qView (as TensorBase<QType>&) into CpuFpReferenceSdpa::forward ...
    ```

  The view is built once per execute, from the graph + node attributes, and immediately consumed.

### Impact on `CpuFpReferenceSdpa::forward` and `SdpaFwdPlan::execute`

- **`CpuFpReferenceSdpa::forward` signature is unchanged** (`TensorBase<T>&`). The view is constructed in the plan layer and passed in. No `shared_ptr` migration through CPU-ref signatures.
- Body changes inside `forward` are limited to per-batch sequence-length bounding using `q.validSeqLen(b)` / `k.validSeqLen(b)` (Shared element #6), with an early-out at the top of the parallel lambda to skip `(b, sq)` pairs where `sq >= q.validSeqLen(b)`.
- `SdpaFwdPlan::execute` constructs the view(s) and forwards them. All ragged-specific code stays at the plan layer.

### Unique benefits over Option A

1. **One-pass bundle allocation.** The harness's `generateBundles` / `tryAddTensorToBundles` can stay a single loop over `TensorAttributes`; ragged primaries become `IrregularTensor<T>` without needing the aux ref at allocation time.
2. **`GraphTensorBundle` storage stays `unique_ptr<ITensor>`.** No need to widen to `shared_ptr` or introduce a `share()` accessor.
3. **Intermediates that don't need iteration get a type that admits it.** `IrregularTensor` lets the executor allocate ragged storage for graph intermediates without paying for view semantics. When iteration is later required, the same buffer is wrapped in a `RaggedView` at the appropriate boundary.
4. **Output-validation wrap is constructible from existing primitives.** When the harness needs a seq_lens-bounded view of a ragged output (for the per-output wrap strategy at Stage 2.9), it can build a `RaggedView` over the bundle's `IrregularTensor` with the looked-up `seq_lens` attached — no new tensor-type infrastructure required. (This advantage is moot if the sentinel-skip strategy is adopted; see *Shared output-validation strategy* below.)
5. **One physical buffer can be re-clothed.** A single `IrregularTensor<T>` can have several `RaggedView`s built over it during the lifetime of an executor run (e.g. for cross-checks against alternative aux interpretations) without copies or re-allocation.

### Costs relative to Option A

- **Two new types** instead of one (`IrregularTensor<T>` and `RaggedView<T, IndexT>`).
- **User-facing two-object pattern at Stage 1.2.** A user constructing a ragged tensor in a sample holds the `IrregularTensor` (whose `rawDeviceData()` goes into the variant pack) plus optionally a `RaggedView` (if they want to iterate it for CPU validation). A's user holds a single `RaggedTensor` value. This ergonomic gap can be hidden behind a runtime-side `TensorBundle` helper (see *Follow-up #6* below).
- A `dynamic_pointer_cast` (or `static_pointer_cast` after a type check) at view-construction sites in the plan layer, because the tensorMap holds `ITensor`s polymorphically. Confined to the plan layer.
- One extra level of indirection on `memory()` / `rawHostData()` accesses. Negligible for CPU references.
- Iterator-strategy refactor on `ITensor` is required (same as Option A — listed in *Shared elements* §5).

---

## Shared output-validation strategy

The test harness's `verifyGraph` validation loop compares ragged output tensors element-by-element between the CPU and GPU bundles. The CPU reference (e.g. `CpuFpReferenceSdpa::forward`) writes only the valid sequence positions of the output (its loop is bounded by `validSeqLen_q(b)`); padding positions retain whatever value `randomizeTensor` left them with. The GPU kernel, however, has no contractual guarantee about what it does to output padding — AITER FMHA kernels in particular may leave padding untouched, zero it, or write garbage from registers. A naive full-buffer compare therefore fails for reasons unrelated to algorithmic correctness.

Two viable approaches handle this. The seq_lens-aware wrap requires the harness to walk node-level op attributes to discover each ragged output's `seq_lens` and to attach it for the compare. The sentinel-skip approach uses the CPU side's own data as the padding detector and requires no node walk.

**Approach A — seq_lens-aware wrap (fallback).** The harness inverts the output-UID → producing-node relationship; for each output UID it finds the node, reads the seq_lens UID from the node's op attribute (e.g. `SdpaAttributes::seq_len_q_tensor_uid()`), looks up the seq_lens runtime tensor in the bundle, and constructs a seq_lens-bounded view of the output for the compare. In Option C this view is a `RaggedView` over the bundle's `IrregularTensor`; in Option A this requires *additional* `RaggedTensor` infrastructure (second ctor accepting existing memory, `setSeqLens` mutator, or wider `allClose` signature). C wins at this approach.

**Approach B — sentinel-skip (recommended).** At init time the harness fills each ragged output's physical buffer (both bundles) with a known sentinel value the CPU reference will not legitimately produce — NaN for floating-point types, a magic number for integer types. After both executions, the validator iterates the full physical buffer of each ragged output and skips any element where the CPU side is bit-equal to the sentinel (those are positions the CPU reference never wrote, i.e. padding). For non-padding positions it applies the existing tolerance-based equality check between CPU and GPU. This approach:

- Requires **no node-attribute walk**.
- Requires **no per-output wrap**.
- Requires **no new tensor-type infrastructure** in either A or C.
- Adds one validator overload (`allCloseSkipSentinel(cpu, gpu, sentinel, tolerance)`).
- Adds a small init-time branch (`if attr.has_ragged_offset() && is_output(uid)` fill with sentinel instead of randomize).
- Works identically in A and C.

**Recommendation: adopt sentinel-skip.** The caveats are that the sentinel must be a value the relevant CPU references cannot produce even pathologically (NaN is generally safe, but watch out for SDPA's `seq_lens_q[b] == 0` edge case where `sumExp == 0` and `log(0)` shows up — a non-NaN magic-number sentinel sidesteps this) and that the GPU's padding values are computed-but-ignored on the host side. If a future op turns out to have a reference that *can* produce the sentinel, fall back to seq_lens-aware wrap for that op.

The recommendation to adopt sentinel-skip neutralizes Stage 2.9 as a place where A and C differ in implementation cost — both options run the same harness code at that stage.

---

## Where A and C diverge (after the constraints above)

| Axis | Option A (`RaggedTensor`, owning subclass) | Option C (`RaggedView` + `IrregularTensor`, plan-layer construction) |
|---|---|---|
| Source of truth for `ragged_offset` wiring | Graph (`TensorAttributes`) — same as C | Graph (`TensorAttributes`) — same as A |
| Source of truth for `seq_lens` wiring | Node-level op attributes — same as C (constraint) | Node-level op attributes — same as A (constraint) |
| Bundle allocation in `generateBundles` | **Two-pass** (aux before primary), `unique_ptr → shared_ptr` shift required | **One-pass**, bundle storage unchanged |
| Executor intermediate allocation | Same two-pass requirement | One-pass; intermediates get `IrregularTensor` naturally |
| Plan-layer view construction (Stage 2.6) | Requires a new non-owning ragged type (`ShallowRaggedTensor`) or a wider `forward` signature | Construct `RaggedView` over a `ShallowTensor` — no additional types needed (with the widened `RaggedView` underlying) |
| Output validation (Stage 2.9) with sentinel-skip | One shared validator overload — tied with C | One shared validator overload — tied with A |
| Output validation (Stage 2.9) with seq_lens-wrap fallback | Requires additional `RaggedTensor` infrastructure | Construct a `RaggedView` with `seq_lens` attached — no additional infrastructure |
| User-facing single-object ergonomic (Stage 1.2) | Yes — one `RaggedTensor` value | No — `IrregularTensor` for `variantPack`, optional `RaggedView` for iteration |
| `resolveTensor` reconciler | Was planned; **dropped from current scope** (degenerate in the harness flow, not needed in the user flow) | Not needed |
| `shared_ptr` migration through `forward` / CPU-ref signatures | None | None |
| Memory ownership | `RaggedTensor` owns its own `MigratableMemory` | `IrregularTensor` owns memory; `RaggedView` borrows |
| Iteration in graph intermediates that are never iterated | Awkward (a `RaggedTensor` with aux refs no one reads) | Natural (`IrregularTensor` has no aux refs) |
| Independent re-views of one buffer | Not natively | Yes (multiple `RaggedView`s over one `IrregularTensor`) |
| Pinned host memory support | Via `RaggedTensor`'s `HostAlloc` template | Via `IrregularTensor`'s `HostAlloc` template |
| Iterator-strategy refactor on `ITensor` | Required | Required |
| Ctor-time structural validation | At `RaggedTensor` ctor | At `RaggedView` ctor (and at `IrregularTensor` ctor for storage) |
| Risk of "view disagrees with its underlying" | N/A | Impossible by construction (no redundant shape args) |
| New types | 1 | 2 |
| Wide call-site changes outside the plan layer | None | None |

---

## Recommendation

**Option C**, with the sentinel-skip strategy for output validation.

The rationale, with the original framing updated:

- The original "single source of truth + no reconciler" argument for C is partly invalidated. The `ragged_offset` source-of-truth point is symmetric across A and C (both put it on `TensorAttributes`); the `seq_lens` source-of-truth point is also symmetric across both (the seq_lens constraint forces both options to discover `seq_lens` from node-level attributes, not from the primary's `TensorAttributes`). The `resolveTensor` reconciler is degenerate in the harness flow and has been dropped.
- C's surviving structural advantages are at the **bundle allocation** layer (one-pass loop, no `shared_ptr` shift, `IrregularTensor` as the natural type for graph intermediates) and at the **plan layer view construction** (no new type required — `RaggedView` over `ShallowTensor` with the widened underlying type).
- C's surviving cost is the **user-facing two-object pattern** (`IrregularTensor` + optional `RaggedView`). A multi-input factory or a runtime-side `TensorBundle` helper can hide this; the help is out of scope for the immediate RFC but is the natural follow-up.
- The choice of output-validation strategy (sentinel-skip vs seq_lens-aware wrap) is independent of the A-vs-C decision but interacts with it. With sentinel-skip, A and C are tied at the output-validation stage. With seq_lens-aware wrap, C is meaningfully simpler than A at that stage too (because `RaggedView` already takes `seq_lens` at construction, whereas `RaggedTensor` needs new infrastructure). Sentinel-skip is recommended; the wrap is a fallback.

A satisfies every functional requirement in the RFC and would also work. The choice is closer than the earlier framing suggested — the deciding factor is "easier bundle / intermediate allocation that survives well into future ragged-aware intermediates" (favours C) vs "easier user-facing single-object API" (favours A). Bundle simplicity has more leverage in the in-scope RFC work, so C wins.

---

## Follow-ups for implementation

These are verification / decision items to handle when implementing.

1. **Add `physical_element_count` to the flatbuffer and to `TensorAttributes`.** Neither option can construct a ragged primary (`IrregularTensor` in C, `RaggedTensor` in A) from a `TensorAttributes` alone because `physicalElementCount` is independent of `prod(dims)` and not currently stored on the attributes. Add `physical_element_count: long = null` to `tensor_attributes.fbs` and a matching `set_physical_element_count(...)` / `get_physical_element_count(...)` pair on the frontend `TensorAttributes`. The plan layer (and the harness's `createTensorFromAttribute`) reads this field directly.

2. **Flatbuffer schema additions.** Add `ragged_offset_tensor_uid: long = null` to `tensor_attributes.fbs`. The `seq_lens` UIDs already exist (or will be added) on the per-op attribute tables — there is **no** `seq_lens_tensor_uid` on `tensor_attributes.fbs`, per the seq_lens constraint at the top of this document.

3. **Verify the `IrregularTensor`-based allocation pattern fits cleanly into `CpuReferenceGraphExecutor`.** The executor's virtual-tensor allocation loop already only calls `rawHostData()` on the intermediates (it never iterates them), so `IrregularTensor` is a drop-in replacement for `Tensor<T>` at that site for ragged virtual intermediates. Confirmed by the per-stage analysis in the companion document.

4. **Confirm no existing code iterates a tensor whose physical size doesn't match `prod(dims)`.** Based on `Tensor.hpp`, this invariant holds today (`Tensor<T>` asserts `_packed = (elementCount == elementSpace)`), so `IrregularTensor<T>` with linear physical-buffer iteration is genuinely new ground. A quick `search_files` over `data_sdk` and `test_sdk` before implementation will confirm nothing relies on the current invariant in a way that would break.

5. **`elementCount()` contract when `seq_lens` is absent.** Documented above as `_underlying->elementSpace()` for `RaggedView`, and as `physicalElementCount` for `IrregularTensor`. Worth a short comment on each public API explaining the "valid elements" vs "physical buffer" distinction so callers don't conflate them.

6. **Runtime-side `TensorBundle` helper for the user-facing flow (post-RFC).** A `UID → owning_runtime_tensor` helper on the user side, with a `bundle.add(attr, …)` factory that for non-ragged attrs builds a `Tensor<T>` and for ragged attrs builds an `IrregularTensor<T>` (using `physical_element_count` from the attr) plus, on demand, a `RaggedView` resolved against the node attributes. Collapses the user's API surface to a one-call-per-`TensorAttributes` pattern regardless of ragged-ness, hiding the two-object cost of Option C. Out of scope for this RFC but the natural direction for the user-facing API.

7. **Sentinel value selection per CPU reference.** Document the sentinel choice and its "the reference cannot produce this value" guarantee in each CPU reference header. NaN is the default safe choice for floats; explicitly call out the SDPA `seq_lens_q[b] == 0` edge case and use a non-NaN magic number (e.g. some specific finite value) if any input regime can reach NaN naturally.

8. **Future extension point for padded mode.** A "seq-lens only, dense underlying" mode (no `ragged_offset`) is straightforward to add to `RaggedView` later since its underlying is already `shared_ptr<TensorBase<T>>` — pass a `Tensor<T>` underlying and a null `ragged_offset`, with the structural validation relaxed accordingly. Non-breaking against the API above.
