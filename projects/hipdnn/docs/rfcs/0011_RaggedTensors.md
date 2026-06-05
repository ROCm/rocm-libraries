Ragged tensors differ from traditional tensors by allowing variable dimensions for the second largest dimension in physical memory. Ie: if the physical memory is laid out [B, X, H, D], X can differ for every batch. Despite this, the memory is laid out contiguously between the batches – potentially with padding.

This will have components in hipDNN frontend, backend and flatbuffers as well as for the data_objects::ITensor class

RFC should cover:
- Adding support for frontend
    - Add ragged_offset getters and setters
    - Gather ragged tensors from tensor bundle (backend?)
    - Checks for ragged tensor (ie: size equal batch dim +1)
    - What is the expectation for what the reported size of the
- Adding flatbuffer support
    - Add field for ragged offset
- Create design for data_objects::ITensor subclass for ragged tensors (or a more general class) with the following qualities:
    - Indexing either works, or is not available
    - Needs to support both ragged tensors, as well as an additional tensor that specifies sequence lengths
    - Size of memory is not derived from dimensions, but specified through other means

Notes:

TensorAttributes needs get_ragged_offset and set_ragged_offset added with definitions similar to:
```c++
    auto
    set_ragged_offset(std::shared_ptr<Tensor_attributes> const& value) -> Tensor_attributes& {
        ragged_offset = value;
        return *this;
    }
```

tensor_attributes.fbs table needs optional field added:

```
   ragged_offset_tensor_uid: long = null
```

Note: Below I will be referring to the Tensor class and the TensorAttributes. The Tensor object refers to the classes in Tensor.hpp in hipdnn_data_sdk, such as ITensor, TensorBase and PinnedTensor ie: The object that contains the tensor data. The TensorAttributes object refers to the graph object that in hipdnn_frontend, and potentially the flatbuffer, ie: the object that describes the tensor for the purpose of the graph

Tensors.hpp needs to have functionality added for connecting a ragged tensor to a tensor object. This requires:
- The ability to specify whether the ragged tensor is packed, or if it uses an additional length tensor (ie like seqlen_q_ptr for aiter kernels)
    - Potentially provide a means for connecting a sequence length tensor to the Tensor
- There needs to be a means to verify that the ragged tensor pointed to by a tensor is the same one mapped to by TensorAttributes (for the purpose of the cpu references at the very least)
- There needs to be a way for an ITensor derived object to have a memory allocation that doesn't exactly match its dimensions
- The maximum dimension for the Tensor and the TensorAttributes with ragged tensors needs to be equal to the maximum padded sequence length
- Whatever object represents the ragged tensor data, its indexing and iteration functions should either work properly, not exist, or throw an error (that is the order of preference for those options)
- Working with the Cpu reference is a high priority, and one of the main reasons why this class is useful to exist. However this is designed, it should allow the Cpu fp reference for Sdpa to use ragged tensors
- Some potential options for how to support this:
    - Wrapper class that wraps a tensor object and overrides its indexing and iterators with the information in the ragged tensor (and potentially sequence length tensor)
    - Add functions to the Tensor class (and/or potentially an additional constructor) to support connecting ragged tensors and potentially sequence length tensors which would allow for the iteration to work out of the box
    - Create a new specialization of ITensor which is designed for ragged tensors
    - Some other option that hasn't been considered

---------Do not edit content of this file above this point, have claude write the plan below-------

# Plan: `Tensors.hpp` design for ragged tensors

The two configurations targeted in this iteration are the ones AITER kernels actually use:

- **Packed with `ragged_offset` only.**
- **Packed with `ragged_offset` + `seq_lens`.**

A "padded, seq-lens-only" mode (no `ragged_offset`) is intentionally deferred. It is reachable later by adding a constructor (or a tag-typed ctor) without breaking the API designed here.

---

## Shared elements (apply to both options below)

1. **Aux tensors are `std::shared_ptr<TensorBase<IndexT>>`**, with `IndexT` templated and defaulting to `int32_t` (and `int64_t` permitted).
2. **Auxiliaries are constructor-only / immutable** on the ragged object. The ragged object never mutates which aux tensors it points at.
3. **Single packed-mode shape**: `ragged_offset` is **required**; `seq_lens` is **optional** (`nullptr` permitted). No tag types — there is currently only one ragged mode to construct.
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
    //   begin/end      -> RaggedCompositeIndex strategy

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
    RaggedIndexTensor<IndexT>                   _seqLens;        // nullable
};
```

**Key properties**
- Memory is self-owned through `MigratableMemory<T, HostAlloc, DeviceAlloc>`, mirroring `Tensor<T>`. `PinnedTensor<T>` equivalence is reached via `RaggedTensor<T, IndexT, PinnedHostAllocator<T>>`.
- `dims()[1] == S_max`; `elementSpace()` is decoupled from `prod(dims)`.
- No UID is stored on the tensor. Identity verification against the graph happens at the plan layer by `shared_ptr` / `TensorBase<IndexT>*` pointer equality.
- Ctor-time structural validation: ranks, sizes, and `aux->elementCount() == B+1` / `B` are checked at construction so a `RaggedTensor` cannot exist in an invalid state.

**Plan-layer verification (required by A)**
- A helper `resolveTensor<T, IndexT>(primaryUid, raggedOffsetUid, seqLensUid, tensorMap, role)` in the test-SDK plan layer performs a three-way consistency check:
  - graph says dense AND runtime is dense → OK
  - graph says ragged AND runtime is `RaggedTensor<T, IndexT>` with matching `raggedOffset()` / `seqLens()` pointers → OK
  - any other combination → throw
- `SdpaFwdPlan::execute` (and any analogous plan) calls `resolveTensor` for each ragged tensor *before* invoking the CPU reference.

**Why A is retained but not recommended**
- A's central cost is **triplicated wiring**: the aux relationship is declared on the graph (`TensorAttributes::set_ragged_offset(...)`), declared again on the runtime `RaggedTensor` ctor, and a third piece of code (`resolveTensor`) exists only to reconcile the two. Every new ragged op has to remember both declarations, and every test author has to either trust the reconciler or hit a runtime exception.
- A satisfies every functional requirement in the RFC. It does not, however, give us a single source of truth for the aux wiring, and the reconciler is dead weight if a structure exists that makes the reconciliation impossible to need.

---

## Option C — `RaggedView<T, IndexT>` over `IrregularTensor<T>`, constructed in the plan layer (**recommended**)

This option introduces two narrow types and places **all** ragged-aux wiring on the graph as the single source of truth. The plan layer reads that wiring and assembles a `RaggedView` from `graph.uid → tensorMap[uid]` lookups. The runtime tensorMap carries only dense storage.

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
    // Iteration is intentionally NOT supported:
    //   begin()/end()/cbegin()/cend() throw std::logic_error with a message directing
    //   the caller at RaggedView<T> if iteration is required. This is consistent with
    //   ITensor's runtime-polymorphic style and rules out accidental walks over a buffer
    //   that has no well-defined dense iteration order.

private:
    MigratableMemory<T, HostAlloc, DeviceAlloc> _memory;
    std::vector<int64_t>                        _paddedDims;
    std::vector<int64_t>                        _strides;
    size_t                                      _physicalElementCount;
};
```

**Pinned host memory** is supported via `IrregularTensor<T, PinnedHostAllocator<T>, …>`, exactly the same trick `PinnedTensor<T>` uses for `Tensor<T>`.

### Type 2 — `RaggedView<T, IndexT>`

A non-owning wrapper that turns an `IrregularTensor<T>` into a fully iterable ragged tensor by attaching the required aux buffers.

```cpp
template <typename T, typename IndexT = int32_t>
class RaggedView : public TensorBase<T>
{
public:
    // Shape, strides, and physicalElementCount are NOT passed in: they come
    // authoritatively from the underlying IrregularTensor. This removes a whole
    // class of "view disagrees with its underlying" bugs by construction.
    RaggedView(std::shared_ptr<IrregularTensor<T>>  underlying,
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
    std::shared_ptr<IrregularTensor<T>>  _underlying;
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
| Underlying storage for a ragged view | `IrregularTensor<T>` | Intentionally none (throws) |
| Ragged inputs / outputs of an op | `RaggedView<T, IndexT>` | Full ragged iteration |
| Aux tensors (`ragged_offset`, `seq_lens`) | Plain `Tensor<IndexT>` via `shared_ptr` | Dense iteration |
| Graph intermediates that no one walks | `IrregularTensor<T>` directly | Intentionally none |
| Non-ragged tensors | Plain `Tensor<T>` | Dense iteration |

### Wiring: single source of truth on the graph

- **Graph (`TensorAttributes`)**: gains `set_ragged_offset(...)` / `set_seq_len(...)` as the RFC requires, plus the flatbuffer field (`ragged_offset_tensor_uid`, and analogously for `seq_lens`).
- **Runtime tensorMap (`UID → ITensor`)**: carries only the dense buffers — the `IrregularTensor<T>` that holds the packed primary, and the `Tensor<IndexT>` that holds each aux. **No ragged metadata at this layer.**
- **Plan layer (executor)**: when executing an op whose graph declares a given input/output as ragged, the plan constructs the view at the call site:

    ```cpp
    auto qView = std::make_shared<RaggedView<QType>>(
        std::static_pointer_cast<IrregularTensor<QType>>(tensorMap.at(graph.qUid)),
        std::static_pointer_cast<TensorBase<IndexT>>(tensorMap.at(graph.qRaggedOffsetUid)),
        graph.qSeqLensUid ? std::static_pointer_cast<TensorBase<IndexT>>(
                                tensorMap.at(*graph.qSeqLensUid))
                          : nullptr);
    // ... pass *qView (as TensorBase<QType>&) into CpuFpReferenceSdpa::forward ...
    ```

  The view is built once, from the graph, and immediately consumed. Because nothing outside the graph ever declares the aux relationship, there is nothing to disagree with — and therefore nothing to reconcile. The `resolveTensor` helper that Option A required does not exist in Option C.

### Impact on `CpuFpReferenceSdpa::forward` and `SdpaFwdPlan::execute`

- **`CpuFpReferenceSdpa::forward` signature is unchanged** (`TensorBase<T>&`). The view is constructed in the plan layer and passed in. No `shared_ptr` migration through CPU-ref signatures.
- Body changes inside `forward` are limited to per-batch sequence-length bounding (~2 extra `dynamic_cast`s, or zero if we use the `validSeqLen` virtual on `TensorBase`).
- `SdpaFwdPlan::execute` constructs the view(s) and forwards them. All ragged-specific code stays at the plan layer.

### Unique benefits over Option A

1. **Single source of truth for aux wiring.** The graph is authoritative; the runtime cannot redundantly (and possibly inconsistently) re-declare the relationship.
2. **No reconciler.** `resolveTensor`'s entire reason for existing is gone.
3. **No ctor redundancy inside the view itself.** Shape, strides, and physical element count come from the underlying — there is no opportunity to construct a `RaggedView` that disagrees with its storage.
4. **Intermediates that don't need iteration get a type that admits it.** `IrregularTensor` lets the executor allocate ragged storage without paying for view semantics or pretending the buffer has a dense iteration order. When iteration is later required, the same buffer is wrapped in a `RaggedView` at the appropriate boundary.
5. **One physical buffer can be re-clothed.** A single `IrregularTensor<T>` can have several `RaggedView`s built over it during the lifetime of an executor run (e.g. for cross-checks against alternative aux interpretations) without copies or re-allocation.

### Costs relative to Option A

- **Two new types** instead of one (`IrregularTensor<T>` and `RaggedView<T, IndexT>`), plus a `shared_ptr<IrregularTensor<T>>` lifetime contract on the view's underlying.
- A `dynamic_pointer_cast` (or `static_pointer_cast` after a type check) at view-construction sites in the plan layer, because the tensorMap holds `ITensor`s polymorphically. Confined to the plan layer.
- One extra level of indirection on `memory()` / `rawHostData()` accesses. Negligible for CPU references.
- Iterator-strategy refactor on `ITensor` is required (same as Option A — listed in *Shared elements* §5).

---

## Where A and C diverge

| Axis | Option A (`RaggedTensor`, owning subclass) | Option C (`RaggedView` + `IrregularTensor`, plan-layer construction) |
|---|---|---|
| Source of truth for aux wiring | Graph **and** runtime tensor; a reconciler bridges them | **Graph only**; runtime tensorMap carries dense storage and aux buffers, nothing else |
| Plan-layer reconciliation helper | Required (`resolveTensor` pointer-equality on aux) | Not required (no second declaration to reconcile against) |
| `shared_ptr` migration through `forward` / CPU-ref signatures | None | None (view is built in the plan layer, passed as `TensorBase<T>&`) |
| Memory ownership | `RaggedTensor` owns its own `MigratableMemory` | `IrregularTensor` owns memory; `RaggedView` borrows via `shared_ptr<IrregularTensor<T>>` |
| Intermediate tensors that no one iterates | Either build a "no-aux" `RaggedTensor` (semantically odd) or introduce a separate type ad hoc | `IrregularTensor<T>` exists for exactly this role |
| Output tensors easily iterable for verification | Yes (`RaggedTensor`'s ragged iteration) | Yes (`RaggedView`'s ragged iteration) |
| Independent re-views of one buffer | Not natively | Yes (multiple views over one `IrregularTensor`) |
| Pinned host memory support | Via `RaggedTensor`'s `HostAlloc` template | Via `IrregularTensor`'s `HostAlloc` template |
| Iterator-strategy refactor on `ITensor` | Required | Required |
| Ctor-time structural validation | At `RaggedTensor` ctor | At `RaggedView` ctor (and at `IrregularTensor` ctor for storage) |
| Risk of "view disagrees with its underlying" | N/A | Impossible by construction (no redundant shape args) |
| New types | 1 | 2 |
| Wide call-site changes outside the plan layer | None | None |

---

## Recommendation

**Option C.** It is the only option that gives us a single source of truth for the aux wiring (eliminating the duplicate-spec / reconciler cost that is Option A's structural weakness), provides a natural type for graph intermediates that don't need iteration (`IrregularTensor`), keeps output tensors and verification inputs fully iterable (`RaggedView`), and does so without forcing `shared_ptr` through the CPU-reference API surface.

Option A remains viable and satisfies the RFC's functional requirements, but it carries the duplicate-spec / reconciler cost as long as it lives, and that cost compounds with every new ragged op.

---

## Follow-ups for implementation

These do not change the architecture above; they're verification / decision items to handle when implementing.

1. **Verify the `IrregularTensor`-based allocation pattern fits cleanly into `CpuReferenceGraphExecutor`.** The executor needs to know that a UID flagged ragged in the graph should be allocated as an `IrregularTensor<T>` with a graph-supplied `physicalElementCount`, rather than the default `Tensor<T>` allocation. This is a small change but should be confirmed against the current executor before writing the patch.

2. **Confirm no existing code iterates a tensor whose physical size doesn't match `prod(dims)`.** Based on `Tensor.hpp`, this invariant holds today (`Tensor<T>` asserts `_packed = (elementCount == elementSpace)`), so `IrregularTensor<T>` is genuinely new ground. A quick `search_files` over `data_sdk` and `test_sdk` before implementation will confirm nothing relies on the current invariant in a way that would break.

3. **Decide whether `IrregularTensor`'s iterators throw or are `=delete`.** Recommend **throw `std::logic_error`** with a message directing the caller at `RaggedView<T>`, for consistency with the rest of `ITensor`'s runtime-polymorphic design. Cost is one runtime check per `begin()`/`end()`.

4. **`elementCount()` contract when `seq_lens` is absent.** Documented above as `_underlying->elementSpace()`. Worth a short comment on the public API explaining the "valid elements" vs "physical buffer" distinction so callers don't conflate them.

5. **Flatbuffer schema additions.** Add `ragged_offset_tensor_uid: long = null` and (in the same change) `seq_lens_tensor_uid: long = null` to `tensor_attributes.fbs`. The `seq_lens` field is added now even though the seqlens-only-padded mode is deferred, because the packed+seqlens configuration is in scope from day one.

6. **Future extension point for padded mode.** If a "seq-lens only, dense underlying" use case appears later, it is added either as a second `RaggedView` ctor that accepts a `shared_ptr<TensorBase<T>>` underlying (with no `ragged_offset`) or as a separate small type. Either path is non-breaking against the API designed here.
