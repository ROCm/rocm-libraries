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

## Scope

The two configurations targeted in this iteration are the ones AITER kernels actually use:

- **Packed with `ragged_offset` only.**
- **Packed with `ragged_offset` + `seq_lens`.**

A "padded, seq-lens-only" mode (no `ragged_offset`) is intentionally deferred. It is reachable later by adding a constructor or relaxing the structural validation without breaking the API designed here.

---

## Key constraint: `seq_lens` is not on `TensorAttributes`

For cuDNN frontend-API compatibility, **`TensorAttributes` gains `set_ragged_offset(...)` only — not `set_seq_len(...)`**. The `seq_lens` tensor is referenced from the *node-level* op attributes (e.g. `SdpaAttributes::set_seq_len_q(...)` / `set_seq_len_kv(...)`), not from the primary tensor's `TensorAttributes`. In the SDPA case a single `seq_lens_q` tensor is shared across the Q primary and a single `seq_lens_kv` is shared across K and V — so there is no 1:1 binding between a primary `TensorAttributes` and a `seq_lens` tensor in the first place.

This constraint shapes the design at several layers:

- Code that walks the graph one `TensorAttributes` at a time can discover whether a tensor is a ragged primary and which tensor is its `ragged_offset`, but **cannot** discover which `seq_lens` it is paired with. Discovering `seq_lens` requires walking the node-level op attributes.
- The plan layer for each op (e.g. `SdpaFwdPlanBuilder`) already has access to the node attributes and *is* the natural place to attach `seq_lens` to a ragged view.
- The test harness's bundle layer therefore should not try to attach `seq_lens` to ragged primaries at all. The bundle holds dense storage; ragged-aware views are assembled at the plan layer when needed.

---

## Shared elements

These apply regardless of the type-system shape chosen below.

1. **Aux tensors are `std::shared_ptr<TensorBase<IndexT>>`**, with `IndexT` templated (defaulting to `int32_t`, with `int64_t` also permitted).
2. **Auxiliaries are constructor-only / immutable** on the ragged object.
3. **`dims()[1] == S_max`** by construction. `elementSpace()` is decoupled from `prod(dims)` and equals the physical buffer size of the underlying storage.
4. **Polymorphic index-strategy hook on `ITensor`**: introduce `virtual std::unique_ptr<ITensorIndex> makeIndex(bool isEnd) const` so ragged iteration can supply a `RaggedCompositeIndex` that walks only valid elements. The existing `LinearIndex` / `CompositeIndex` strategies move behind this interface; non-ragged behaviour is unchanged. `ITensorIterator`'s `std::variant` does not grow.
5. **Optional `virtual int64_t validSeqLen(int64_t b) const` on `TensorBase`** (default `dims()[1]`) so CPU references can bound inner loops without `dynamic_cast`.
6. **Iteration semantics for `elementCount()`**: if `seq_lens` is attached, `elementCount()` returns the sum of valid elements implied by `seq_lens`. Otherwise it returns the underlying physical element count.

---

## Design: `RaggedView<T, IndexT>` over `IrregularTensor<T>`, assembled at the plan layer

Two narrow types are introduced. Storage is decoupled from ragged semantics: the storage type carries the physical buffer and the padded dims; the view type attaches `ragged_offset` and (optionally) `seq_lens` and provides ragged iteration. The view is assembled at the moment ragged-aware access is needed, reading `ragged_offset` from the graph's `TensorAttributes` and `seq_lens` from the relevant node-level op attributes.

### Type 1 — `IrregularTensor<T>`

A memory-owning buffer whose physical element count is independent of `prod(dims)`. Used as the underlying storage for ragged views and as the natural type for graph intermediates that don't need iteration.

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
    //   dims()         -> paddedDims                    (dims()[1] == S_max)
    //   strides()      -> strides
    //   elementSpace() -> physicalElementCount
    //   elementCount() -> physicalElementCount          (NOT prod(dims))
    //   isPacked()     -> false
    //
    // Direct addressing (rawHostData / rawDeviceData) is fully supported.
    //
    // Iteration walks the physical buffer linearly:
    //   begin()/end()/cbegin()/cend() iterate `physicalElementCount` contiguous
    //   elements. See the iteration-policy note below for the rationale.

private:
    MigratableMemory<T, HostAlloc, DeviceAlloc> _memory;
    std::vector<int64_t>                        _paddedDims;
    std::vector<int64_t>                        _strides;
    size_t                                      _physicalElementCount;
};
```

**Pinned host memory** is supported via `IrregularTensor<T, PinnedHostAllocator<T>, …>`, the same pattern `PinnedTensor<T>` uses for `Tensor<T>`.

**Iteration policy.** An earlier formulation had `IrregularTensor`'s iterators throw `std::logic_error` to rule out walks over a buffer with no well-defined dense iteration order. The `seq_lens` constraint forced a re-evaluation: the test harness's bundle layer has call sites (`randomizeTensor`, output validation) that need to walk the buffer but cannot look up `seq_lens` from a `TensorAttributes` alone. Throwing would force the harness into a wrapping step (node-attribute walk to identify if there is an associated `seq_lens` tensor + view construction + ragged-aware iteration) that is both complicated and unnecessary, because for the harness's purposes a linear walk over the physical buffer produces correct results (see *Special-handling stages* below). The current policy is therefore linear iteration; ragged-aware iteration that respects per-batch valid lengths is the job of `RaggedView<T>`.

### Type 2 — `RaggedView<T, IndexT>`

A non-owning wrapper that turns any `TensorBase<T>`-derived underlying storage into a fully iterable ragged tensor by attaching the required aux buffers.

```cpp
template <typename T, typename IndexT = int32_t>
class RaggedView : public TensorBase<T>
{
public:
    // Shape, strides, and physicalElementCount come authoritatively from
    // the underlying tensor's dims()/strides()/elementSpace(). They are
    // NOT passed in — this removes "view disagrees with its underlying"
    // bugs by construction.
    //
    // The underlying is `shared_ptr<TensorBase<T>>` so the view can wrap
    // either an owning `IrregularTensor<T>` (user / bundle case) OR a
    // non-owning `ShallowTensor<T>` over a variant-pack `void*` (plan-layer
    // case).
    RaggedView(std::shared_ptr<TensorBase<T>>       underlying,
               std::shared_ptr<TensorBase<IndexT>>  raggedOffset,        // required
               std::shared_ptr<TensorBase<IndexT>>  seqLens = nullptr);  // optional

    // ITensor / TensorBase<T> overrides:
    //   dims()         -> _underlying->dims()         (forwarded)
    //   strides()      -> _underlying->strides()      (forwarded)
    //   elementSpace() -> _underlying->elementSpace() (forwarded)
    //   memory()       -> _underlying->memory()       (forwarded)
    //   elementCount() -> sum-of-seqLens if _seqLens, else _underlying->elementSpace()
    //   isPacked()     -> false
    //   begin/end      -> RaggedCompositeIndex strategy via the polymorphic
    //                     index hook (walks only valid elements)

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

**Ctor-time structural validation:**
- `_underlying != nullptr`, `_raggedOffset != nullptr`.
- `_raggedOffset->elementCount() == B + 1`, where `B = _underlying->dims()[0]`.
- If `_seqLens != nullptr`: `_seqLens->elementCount() == B`.
- `_raggedOffset` and `_seqLens` (if present) have rank 1.

**Why the underlying is typed as `TensorBase<T>` rather than `IrregularTensor<T>`.** The intuitive choice for the underlying would be `IrregularTensor<T>`, since that is the owning storage type for ragged primaries in the bundle and in executor intermediates. But the plan layer at execute time only has a `void*` from the variant pack — there is no `IrregularTensor` to point at. Widening the underlying to `TensorBase<T>` accommodates both the owning case and a non-owning `ShallowTensor<T>` wrap of the variant-pack pointer without introducing a third tensor type.

### Tiered API

| Role | Type | Iteration |
|---|---|---|
| Owning storage for a ragged primary (bundle / executor intermediate) | `IrregularTensor<T>` | Linear over physical buffer |
| Ragged inputs/outputs at the Cpu reference plan layer | `RaggedView<T, IndexT>` over a `ShallowTensor<T>` of the variant-pack pointer | Full ragged iteration |
| Ragged tensor a user iterates directly (sample flow) | `RaggedView<T, IndexT>` over the owning `IrregularTensor<T>` | Full ragged iteration |
| Aux tensors (`ragged_offset`, `seq_lens`) | Plain `Tensor<IndexT>` via `shared_ptr` | Dense iteration |
| Graph intermediates that no one walks | `IrregularTensor<T>` directly | Linear (incidental) |
| Non-ragged tensors | Plain `Tensor<T>` | Dense iteration |

### Wiring sources of truth

- **Graph (`TensorAttributes`)** gains two new accessors:
  - `set_ragged_offset(...)` / `get_ragged_offset()` — points at the `ragged_offset` aux tensor's attributes.
  - `set_alignment(int64_t)` / `get_alignment()` — per-batch alignment requirement of the physical buffer, in elements. Defaults to `1` (no alignment), which is what AITER FMHA currently needs.
  The flatbuffer gains `ragged_offset_tensor_uid: long = null` and `alignment: long = 1`.
- **Node-level op attributes** (e.g. `SdpaAttributes`) retain their `set_seq_len_q(...)` / `set_seq_len_kv(...)` accessors. The per-op attribute flatbuffer tables carry the corresponding `seq_len_*_tensor_uid` fields.
- **Runtime tensorMap (`UID → ITensor`)** carries only the dense buffers: an `IrregularTensor<T>` for each ragged primary, ordinary `Tensor<IndexT>` for each aux, ordinary `Tensor<T>` for non-ragged tensors. **No ragged-view objects at this layer.**
- **Plan layer (executor)**: when executing an op whose graph declares a given input/output as ragged, the plan constructs the view at the call site, reading `ragged_offset` from `TensorAttributes::ragged_offset_tensor_uid()` and `seq_lens` from the node's op-specific attribute. The physical element count needed to construct the underlying `ShallowTensor<T>` is cached on the plan's `params` struct at plan-build time (recovered from the `ragged_offset` aux's last element — see *Special-handling stages* below):

    ```cpp
    auto qUnderlying = std::make_shared<ShallowTensor<QType>>(
        variantPack.at(_params.qTensor.uid),
        _params.qTensor.dims,
        _params.qTensor.strides,
        _params.qTensor.physicalElementCount);   // cached at plan build
    auto qRaggedOffset = std::make_shared<ShallowTensor<IndexT>>(/* ... */);
    auto qSeqLens      = _params.seqLenQTensor.has_value()
        ? std::make_shared<ShallowTensor<IndexT>>(/* ... */)
        : nullptr;

    auto qView = std::make_shared<RaggedView<QType, IndexT>>(
        qUnderlying, qRaggedOffset, qSeqLens);

    // pass *qView as TensorBase<QType>& into CpuFpReferenceSdpa::forward
    ```

### Impact on CPU references

`CpuFpReferenceSdpa::forward`'s signature is **unchanged** (`TensorBase<T>&`). The view is built in the plan layer and passed in. Body changes are limited to per-batch sequence-length bounding using `q.validSeqLen(b)` / `k.validSeqLen(b)`, with an early-out at the top of the parallel lambda to skip `(b, sq)` pairs where `sq >= q.validSeqLen(b)`. The parallel decomposition continues to use `S_max` so the work scheduler doesn't have to know about ragged-ness.

---

## Special-handling stages in the test-harness flow

Most stages of the integration-test harness flow (`IntegrationGraphVerificationHarness` → `GraphTensorBundle` → `CpuReferenceGraphExecutor` → per-op plan → CPU reference) are not affected by this design beyond the type substitutions described above. Three stages warrant explicit discussion.

### Bundle allocation in `createTensorFromAttribute`

The physical element count of a ragged primary is not present on its `TensorAttributes` — it is recovered from the `ragged_offset` aux's host values and the `TensorAttributes::get_alignment()` value. Because `ragged_offset` isn't populated until the aux's host buffer has been filled, this forces a **two-pass bundle allocation**:

- **Pass 1**: walk the graph and allocate every non-ragged tensor (non-ragged primaries as `Tensor<T>`, auxes including each `ragged_offset` as `Tensor<int32_t>`). Then run the existing per-bundle initialization step so the `ragged_offset` host values are populated.
- **Pass 2**: walk the graph again; for each `TensorAttributes` with `has_ragged_offset()`, look up its already-populated `ragged_offset` aux in the bundle, compute the physical element count, and allocate `make_unique<IrregularTensor<T>>(dims, strides, physicalElementCount)`.

The physical element count is computed as:

```
physicalElementCount = sum_b align_up(ragged_offset[b+1] - ragged_offset[b], attr.get_alignment())
```

For the default `alignment = 1`, this reduces to `ragged_offset[B]` (the packed valid size) — the natural answer for AITER FMHA and any other kernel that has no per-batch alignment requirement. For larger alignment values, each batch's contribution is rounded up to a multiple of the alignment, producing a physical buffer that's strictly larger than `ragged_offset[B]` and matches the kernel's per-batch padding requirement.

The two passes are over the same `unordered_map<int64_t, unique_ptr<ITensor>>`; no structural change to `GraphTensorBundle` is required (storage stays `unique_ptr<ITensor>`, no `shared_ptr` shift, no aliasing ctor needed). The plan layer is given the same value (cached in its `params`) at plan-build time using the same lookup, so neither the plan nor the executor re-derives it at execute time.

Because the view is not built at this layer, the bundle does not need to know `seq_lens`. The user-facing flow naturally mirrors this two-step pattern: the user allocates and populates `ragged_offset`, then allocates an `IrregularTensor` sized from the same `(ragged_offset, alignment)` pair.

### Input randomization at bundle init

`bundle.randomizeTensor(uid)` walks the underlying tensor's iterator. For a ragged primary stored as an `IrregularTensor<T>` this walks the entire physical buffer linearly, which fills more elements than strictly valid (the per-batch padding tails are also randomized). This is harmless: neither the GPU kernel nor the CPU reference reads padding positions, so the values placed there don't affect correctness. Since both bundles are seeded identically, padding values stay byte-equal between the two paths.

This is the principal reason `IrregularTensor`'s iterators walk the physical buffer rather than throwing. Throwing would have forced the harness to perform a node-attribute walk to discover `seq_lens` and wrap the bundle entry in a `RaggedView` purely so randomization could be ragged-aware — none of which is necessary for the actual correctness requirement.

### Output validation in `verifyGraph`

This stage requires real care. The CPU reference (e.g. `CpuFpReferenceSdpa::forward`) writes only the valid sequence positions of an output, leaving padding at its randomize-time value. The GPU kernel, however, has **no contractual guarantee** about what it does to output padding — AITER FMHA forward kernels in particular may leave padding alone, zero it out, or write garbage from registers. A naive full-buffer element-by-element comparison therefore fails for reasons unrelated to algorithmic correctness.

**Recommended approach: sentinel-skip.** At harness init time, fill each ragged output's physical buffer (both bundles) with a known sentinel value the CPU reference cannot legitimately produce — typically NaN for floating-point types, or a magic number for integer types — instead of randomizing it. After execution the validator iterates the full physical buffer; for any position where the CPU side is bit-equal to the sentinel, the CPU reference did not write there (i.e. it is padding), so the compare is skipped. For non-sentinel positions the existing tolerance-based equality check is applied.

This approach:
- Requires no node-attribute walk to discover `seq_lens` at the validation site.
- Requires no per-output `RaggedView` construction.
- Adds one validator overload (`allCloseSkipSentinel(cpu, gpu, sentinel, tolerance)`).
- Adds a small init-time branch to fill ragged outputs with the sentinel.

**Fallback: seq_lens-aware wrap.** If a future op turns out to have a CPU reference that can naturally produce the sentinel value, the fallback is for the harness to walk node-level op attributes to find each output's `seq_lens` UID, look up the runtime aux in the bundle, and construct a `RaggedView` over the bundle's `IrregularTensor` with `seq_lens` attached for the compare. Constructing this wrap from existing primitives is straightforward — `RaggedView` already accepts `seq_lens` at construction.

### Why the plan-layer view construction site is straightforward

`SdpaFwdPlan::execute` already has access to the node-level op attributes via its cached `params`. It already constructs `ShallowTensor<T>` per call to give `CpuFpReferenceSdpa` something typed to walk. The only new work is wrapping that `ShallowTensor` plus the aux `ShallowTensor`s in a `RaggedView`. No `shared_ptr` propagates further than the plan layer.

---

## Alternate design considered: self-owning `RaggedTensor<T, IndexT>` subclass

An alternative shape was considered in which a `RaggedTensor<T, IndexT>` class derived from `TensorBase<T>` owns both its physical buffer (`MigratableMemory`) and its aux refs (`shared_ptr<TensorBase<IndexT>>` for `ragged_offset` and optionally `seq_lens`), constructed with all of these passed to its ctor:

```cpp
RaggedTensor(std::vector<int64_t> paddedDims,
             std::vector<int64_t> strides,
             size_t               physicalElementCount,
             std::shared_ptr<TensorBase<IndexT>> raggedOffset,        // required
             std::shared_ptr<TensorBase<IndexT>> seqLens = nullptr);  // optional
```

The user / bundle would hold a single owning `RaggedTensor` value rather than a `(IrregularTensor, RaggedView)` pair, and the ragged iteration / `validSeqLen` machinery would live on this class.

This alternate is not actually a one-type design. The plan layer at execute time has only a `void*` from the variant pack — there is no owning `RaggedTensor` to hand to the CPU reference. The existing `ShallowTensor<T>` is dense-only and walks `prod(dims)` elements, which is wrong for ragged primaries. So a non-owning peer is also required: `ShallowRaggedTensor<T, IndexT>`, a ragged-aware analogue of `ShallowTensor<T>` that takes a borrowed `void*` plus dims/strides/physicalElementCount plus aux refs and provides the same ragged iteration / `validSeqLen` semantics as `RaggedTensor` without owning memory. It would be constructed per-execute at the plan layer (and per-output in the seq_lens-wrap output-validation fallback). The two types share their index/iteration machinery; only memory ownership differs.

The alternative — keeping the alternate to one type by widening `CpuFpReferenceSdpa::forward` (and every other op's CPU reference) to take aux pointers + dims as extra arguments alongside the `TensorBase<T>&` primary — was rejected because it forces every reference signature to take ragged-specific parameters whether or not the op is ragged. The two-type design is cleaner.

### Pros over the recommended design

- **Single owning object at the user-facing layer (samples).** A user holds one `RaggedTensor` whose `rawDeviceData()` goes into the variant pack and which is passed as `TensorBase<T>&` into CPU references. The recommended design splits this into an `IrregularTensor` (whose `rawDeviceData()` goes into the variant pack) plus optionally a `RaggedView` (for iteration). This single-object ergonomic is the strongest argument for this alternate.
- **Pinned host memory** can be expressed via `RaggedTensor`'s template parameters (`RaggedTensor<T, IndexT, PinnedHostAllocator<T>, …>`), mirroring `PinnedTensor<T>`.

### Cons relative to the recommended design

- **Same new-type count as the recommended design.** The need for `ShallowRaggedTensor` brings the total to two new types (`RaggedTensor` + `ShallowRaggedTensor`), matching the recommended design's `IrregularTensor` + `RaggedView`. The "smaller surface" intuition does not survive contact with the plan-layer construction site.
- **Stricter two-pass bundle allocation.** The recommended design is also two-pass (because the physical element count is recovered from `ragged_offset[B]` after the aux's host values are populated), but the two passes there are "allocate-and-init auxes, then allocate primaries." In this alternate the two passes are "allocate auxes, then allocate primaries with the aux `shared_ptr` threaded into the primary's ctor" — the ctor-time dependency between sibling tensors is harder to manage than the recommended design's pure ordering-of-allocations dependency.
- **`GraphTensorBundle` must shift `unique_ptr → shared_ptr`** (or expose an aliasing `share()`) so the aux ref the bundle holds can be installed as the primary's aux at construction. The recommended design avoids this.
- **Awkward fit for graph intermediates that no one iterates.** An intermediate ragged buffer would either be allocated as a "no-aux" `RaggedTensor` (semantically odd: the class's whole point is the aux refs), or a separate `IrregularTensor`-style type would have to be introduced ad hoc — pushing the type count above two.
- **Output validation requires new infrastructure on the class itself.** The bundle holds the `RaggedTensor` with `seqLens == nullptr` (per the `seq_lens` constraint at the bundle layer); attaching a `seq_lens` for the seq_lens-aware compare requires either a second ctor accepting existing memory, a `setSeqLens` mutator (violating the immutability rule), or a wider `allClose` validator signature. None of this is needed in the recommended design — `RaggedView` already accepts `seq_lens` at construction. (This cost disappears in both designs if the sentinel-skip approach is adopted for output validation.)
- **No single source of truth on the runtime side.** The aux relationship is declared once on the graph (`TensorAttributes`) and again on the runtime ctor. In a hypothetical user-facing flow that took both declarations as inputs, a reconciler would be needed to verify they agree. (In the actual test-harness flow this reconciler is degenerate, because the variant pack waist erases the runtime declaration before it reaches the plan layer — but the asymmetry remains a smell.)

The deciding factor in favour of the recommended design is the bundle / intermediate allocation simplicity, which holds unconditionally and survives well into future ragged-aware intermediates. The single-object user-facing ergonomic the alternate offers is a real but smaller advantage that can be partly recovered by a runtime-side `TensorBundle` factory (see *Open issues and follow-ups* below).

---

## Open issues and follow-ups

1. **Flatbuffer schema additions.** Add `ragged_offset_tensor_uid: long = null` and `alignment: long = 1` to `tensor_attributes.fbs`. The `seq_lens` UIDs already exist (or will be added) on the per-op attribute tables — there is no `seq_lens_tensor_uid` on `tensor_attributes.fbs`, per the `seq_lens` constraint.

2. **Sentinel value selection per CPU reference.** Document the sentinel choice and the "the reference cannot produce this value" guarantee in each CPU reference header. NaN is the default safe choice for floats, but watch for edge cases — e.g. SDPA forward with `seq_lens_q[b] == 0` can produce NaN naturally (`log(0)` from a fully masked row), in which case a non-NaN magic-number sentinel is required. For low-precision floating-point types (bf16, fp16, fp8) the sentinel space is smaller and "unproducible by reference" is harder to guarantee.

3. **`elementCount()` contract when `seq_lens` is absent.** For `RaggedView` without `seq_lens`, `elementCount()` is `_underlying->elementSpace()` (the physical buffer size); for `IrregularTensor` it is `physicalElementCount`. Worth a short comment on each public API explaining the "valid elements" vs "physical buffer" distinction so callers don't conflate them.

4. **Runtime-side `TensorBundle` helper for the user-facing flow (post-RFC).** A `UID → owning_runtime_tensor` helper on the user side, with a factory that for non-ragged attrs builds a `Tensor<T>` and for ragged attrs handles the two-step (allocate `ragged_offset`, populate it from a user-supplied source, then allocate the `IrregularTensor<T>` sized from `(ragged_offset, alignment)`) plus, on demand, a `RaggedView` resolved against the node attributes. This collapses the user's API surface to a one-call-per-`TensorAttributes` pattern regardless of ragged-ness and hides the two-object cost of this design. Out of scope for this RFC but the natural direction.

5. **Future "padded, seq-lens-only" mode.** Straightforward to add later because `RaggedView`'s underlying is already `shared_ptr<TensorBase<T>>` — pass a `Tensor<T>` underlying and a null `ragged_offset`, with the structural validation relaxed accordingly. Non-breaking against the API above.

---

## Future work

These are larger items that the recommended design accommodates but does not require, and that would be appropriate to tackle as separate efforts.

### `TypedTensor<T>` intermediate in the tensor hierarchy

`IrregularTensor<T>` currently derives from `TensorBase<T>`, which carries the `getHostValue(indices)` / `setHostValue(value, indices)` multi-dim addressing API. Those methods have no meaningful semantics on a ragged primary's physical buffer (the strides describe padded geometry, but per-batch addressing requires the `ragged_offset` math that lives in `RaggedView`). The recommended design accepts this mismatch — callers who hold an `IrregularTensor` are expected to wrap it in a `RaggedView` before addressing it — but the type system does not enforce it.

A cleaner end state would introduce a `TypedTensor<T>` interface between `ITensor` and `TensorBase<T>`:

```
ITensor                              // type-erased: dims, strides, rawHostData (void*), …
   ↑
TypedTensor<T>                       // adds T-awareness: typed rawHostData() -> T*,
   ↑                                 //                  memory() -> MigratableMemory<T>&,
   |                                 //                  typed fill / sentinel ops
   |
   +-- IrregularTensor<T>            // owning, no multi-dim indexing
   |
   +-- TensorBase<T>                 // adds getHostValue / setHostValue + everything below
          ↑
          +-- Tensor<T>, RaggedView<T>, ShallowTensor<T>, …
```

`RaggedView`'s underlying would re-type as `shared_ptr<TypedTensor<T>>` instead of `shared_ptr<TensorBase<T>>`, which accepts both the owning `IrregularTensor<T>` (bundle / user / output-validation cases) and a non-owning `ShallowTensor<T>` (plan-layer case) — the same one-type-fits-both property the design relies on today, just at a narrower interface that doesn't drag along the meaningless multi-dim indexing API. `memory()` plumbing continues to work (`MigratableMemory<T>&` is exposed on `TypedTensor<T>`, so `RaggedView::memory()` remains a clean forwarder).

This is a refactor of `Tensor.hpp` that introduces one new public class and shifts a couple of method declarations between bases. It is not invasive — nothing changes for callers of `TensorBase<T>` — but it touches a foundational header and is worth taking on its own merits rather than as part of the ragged-tensors patch. Until it lands, `IrregularTensor<T>` derives from `TensorBase<T>` and overrides `getHostValue` / `setHostValue` to throw `std::logic_error` with a message directing the caller at `RaggedView<T>`.
