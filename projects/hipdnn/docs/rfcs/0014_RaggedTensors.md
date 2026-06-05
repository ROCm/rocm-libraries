# hipDNN: Ragged Tensors Design Document

- Contributors: hipDNN maintainers
- **Status**: Draft

## Table of Contents
1. [Summary](#summary)
2. [Problem Statement](#problem-statement)
   - 2.1 [What is a ragged tensor](#21-what-is-a-ragged-tensor)
   - 2.2 [hipDNN gap](#22-hipdnn-gap)
   - 2.3 [Configurations targeted in this iteration](#23-configurations-targeted-in-this-iteration)
   - 2.4 [Constraint: `seq_lens` is not on `TensorAttributes`](#24-constraint-seq_lens-is-not-on-tensorattributes)
3. [Existing Infrastructure](#existing-infrastructure)
   - 3.1 [Frontend: `TensorAttributes`](#31-frontend-tensorattributes)
   - 3.2 [Flatbuffer: `tensor_attributes.fbs`](#32-flatbuffer-tensor_attributesfbs)
   - 3.3 [Backend: tensor descriptor and variant pack](#33-backend-tensor-descriptor-and-variant-pack)
   - 3.4 [Data SDK: `ITensor` / `TensorBase<T>` / `Tensor<T>` / `ShallowTensor<T>`](#34-data-sdk-itensor--tensorbaset--tensort--shallowtensort)
   - 3.5 [Test harness: `GraphTensorBundle` and `CpuReferenceGraphExecutor`](#35-test-harness-graphtensorbundle-and-cpureferencegraphexecutor)
4. [Design](#design)
   - 4.1 [Overview](#41-overview)
   - 4.2 [Frontend: `TensorAttributes` additions](#42-frontend-tensorattributes-additions)
   - 4.3 [Flatbuffer schema additions](#43-flatbuffer-schema-additions)
   - 4.4 [Backend wiring](#44-backend-wiring)
   - 4.5 [Data SDK: shared elements across the design](#45-data-sdk-shared-elements-across-the-design)
   - 4.6 [Data SDK: `IrregularTensor<T>` (owning storage)](#46-data-sdk-irregulartensort-owning-storage)
   - 4.7 [Data SDK: `RaggedView<T, IndexT>` (non-owning ragged view)](#47-data-sdk-raggedviewt-indext-non-owning-ragged-view)
   - 4.8 [Tiered API: which type each role uses](#48-tiered-api-which-type-each-role-uses)
   - 4.9 [Wiring sources of truth](#49-wiring-sources-of-truth)
   - 4.10 [Plan-layer view construction and CPU reference impact](#410-plan-layer-view-construction-and-cpu-reference-impact)
   - 4.11 [Test-harness integration](#411-test-harness-integration)
     - 4.11.1 [Pre-supplied input bundle](#4111-pre-supplied-input-bundle)
     - 4.11.2 [Bundle allocation in `createTensorFromAttribute`](#4112-bundle-allocation-in-createtensorfromattribute)
     - 4.11.3 [Input randomization at bundle init](#4113-input-randomization-at-bundle-init)
     - 4.11.4 [Output validation in `verifyGraph`](#4114-output-validation-in-verifygraph)
5. [Known limitations](#known-limitations)
6. [Alternatives considered](#alternatives-considered)
   - 6.1 [Self-owning `RaggedTensor<T, IndexT>` subclass](#61-self-owning-raggedtensort-indext-subclass)
   - 6.2 [Wrapper class around an existing dense `Tensor<T>`](#62-wrapper-class-around-an-existing-dense-tensort)
   - 6.3 [Throwing iterators on `IrregularTensor<T>`](#63-throwing-iterators-on-irregulartensort)
   - 6.4 [Adding `seq_lens` to `TensorAttributes`](#64-adding-seq_lens-to-tensorattributes)
   - 6.5 [Random initialization of aux tensors with structural fixup](#65-random-initialization-of-aux-tensors-with-structural-fixup)
7. [Future work](#future-work)
   - 7.1 [`TypedTensor<T>` intermediate in the tensor hierarchy](#71-typedtensort-intermediate-in-the-tensor-hierarchy)
   - 7.2 [Runtime-side `TensorBundle` helper](#72-runtime-side-tensorbundle-helper)
   - 7.3 [Padded mode (seq-lens only, no `ragged_offset`)](#73-padded-mode-seq-lens-only-no-ragged_offset)
   - 7.4 [Sentinel selection per CPU reference](#74-sentinel-selection-per-cpu-reference)

---

## Summary

Ragged tensors differ from traditional tensors by allowing variable
dimensions for the second-largest dimension in physical memory. If a
tensor's logical layout is `[B, X, H, D]`, `X` may differ for every
batch element. Despite the variable per-batch extent, the memory is
laid out contiguously across batches — potentially with per-batch
alignment padding.

This RFC adds end-to-end ragged-tensor support across hipDNN:

1. **Frontend (`TensorAttributes`)** gains `set_ragged_offset` /
   `get_ragged_offset` accessors and a per-batch `alignment` field.
2. **Flatbuffer schema** (`tensor_attributes.fbs`) gains a defaulted
   `ragged_offset_tensor_uid` field and a defaulted `alignment` field
   so ragged metadata round-trips through serialization.
3. **Backend** propagates the new tensor-attribute fields end-to-end;
   no new backend C-API entry points are introduced.
4. **Data SDK (`Tensor.hpp`)** gains two new tensor types,
   `IrregularTensor<T>` (owning storage whose physical element count
   is decoupled from `prod(dims)`) and `RaggedView<T, IndexT>` (a
   non-owning wrapper that turns any `TensorBase<T>` into a fully
   iterable ragged tensor by attaching the required aux buffers).
5. **CPU references** are updated to use `validSeqLen(b)` bounding;
   the per-op plan layer builds a `RaggedView` per execute and passes
   it in as `TensorBase<T>&`. Reference signatures are unchanged.
6. **Integration-test harness** gains a *pre-supplied input bundle*:
   the test author hands the harness a single
   `unordered_map<int64_t, unique_ptr<ITensor>>` of pre-populated
   tensors keyed by UID. The harness consults this map alongside
   the per-execution-path `GraphTensorBundle`s when constructing
   each variant pack, so aux tensors (`ragged_offset`, `seq_lens`)
   — which cannot be meaningfully randomized — carry deliberate
   values shared between the GPU and CPU paths via the
   `MigratableMemory` host/device synchronization built into each
   `ITensor`. This also collapses the ragged-primary allocation to
   a single pass.

The immediate consumer is scaled dot-product attention (SDPA) using
AITER FMHA kernels, whose ragged contract is "packed batches addressed
through `ragged_offset`, with optional per-batch valid length tensors".

The design deliberately decouples storage from ragged semantics: the
storage type carries the physical buffer and the padded dims; the view
type attaches `ragged_offset` and (optionally) `seq_lens` and provides
ragged iteration. The view is assembled at the moment ragged-aware
access is needed, reading `ragged_offset` from the graph's
`TensorAttributes` and `seq_lens` from the relevant node-level op
attributes.

---

## Problem Statement

### 2.1 What is a ragged tensor

A ragged tensor is a logically `[B, X, …]` tensor where the second
dimension (`X`, typically the sequence dimension) varies per batch
element. Physical memory is laid out as a single contiguous buffer
indexed by a `ragged_offset[B+1]` aux tensor: batch `b` occupies the
contiguous range `[ragged_offset[b], ragged_offset[b+1])` of the
underlying buffer. Optionally, a `seq_lens[B]` aux tensor carries the
**valid** sequence length per batch when the physical layout is
packed-with-trailing-pad (i.e. each batch's reserved region may exceed
its real valid extent so kernels can meet alignment requirements).

### 2.2 hipDNN gap

hipDNN's tensor model in both the frontend (`TensorAttributes`) and
the data SDK (`Tensor.hpp`) currently assumes a tensor's physical
element count equals `prod(dims)`. There is no way to express:

- A primary tensor whose physical buffer is not `prod(dims)` elements
  long.
- A reference from one tensor (the primary) to another tensor (the
  aux `ragged_offset` or `seq_lens`).
- Iteration that walks only valid positions of a ragged primary.

Without these primitives, the CPU reference path cannot validate
ragged-tensor kernels, and the frontend cannot describe ragged inputs
to the backend at all.

Separately, the integration-test harness today initializes input
tensors by random fill. Random values are meaningless for ragged-aux
tensors: a random `int32_t` `ragged_offset` is almost never
monotonically non-decreasing and almost never has
`ragged_offset[0] == 0` or `ragged_offset[B]` equal to any sensible
packed size; a random `seq_lens` routinely exceeds the per-batch
reserved extent. The harness needs a way to consume *deliberate*
values for these tensors.

### 2.3 Configurations targeted in this iteration

Two configurations are in scope — both used by AITER FMHA kernels:

- **Packed with `ragged_offset` only.** The physical buffer is
  exactly `ragged_offset[B]` elements; each batch contributes
  `ragged_offset[b+1] - ragged_offset[b]` rows; there is no padding
  inside the buffer.
- **Packed with `ragged_offset` + `seq_lens`.** The physical buffer is
  the sum of per-batch reserved regions (possibly aligned up); each
  batch's first `seq_lens[b]` rows are valid; trailing rows in each
  region are padding.

A "padded, seq-lens-only" mode (no `ragged_offset`) is intentionally
deferred. It is reachable later by relaxing the structural validation
on the view type without breaking the API designed here; see
[§7.3](#73-padded-mode-seq-lens-only-no-ragged_offset).

### 2.4 Constraint: `seq_lens` is not on `TensorAttributes`

For cuDNN frontend-API compatibility, **`TensorAttributes` gains
`set_ragged_offset(...)` only — not `set_seq_len(...)`**. The
`seq_lens` tensor is referenced from the *node-level* op attributes
(e.g. `SdpaAttributes::set_seq_len_q(...)` /
`set_seq_len_kv(...)`), not from the primary tensor's
`TensorAttributes`. In the SDPA case a single `seq_lens_q` tensor is
shared across the Q primary and a single `seq_lens_kv` is shared
across the K and V primaries, so there is no 1:1 binding between a
primary `TensorAttributes` and a `seq_lens` tensor in the first
place.

This constraint shapes the design at several layers:

- Code that walks the graph one `TensorAttributes` at a time can
  discover whether a tensor is a ragged primary and which tensor is
  its `ragged_offset`, but **cannot** discover which `seq_lens` it
  is paired with. Discovering `seq_lens` requires walking the
  node-level op attributes.
- The plan layer for each op (e.g. `SdpaFwdPlanBuilder`) already has
  access to the node attributes and *is* the natural place to attach
  `seq_lens` to a ragged view.
- The test harness's bundle layer therefore should not try to attach
  `seq_lens` to ragged primaries at all. The bundle holds dense
  storage; ragged-aware views are assembled at the plan layer when
  needed.

---

## Existing Infrastructure

### 3.1 Frontend: `TensorAttributes`

`hipdnn_frontend::graph::Tensor_attributes` declares per-tensor
metadata used to build the operation graph: dims, strides, dtype,
UID, virtual flag, alignment-related fields, etc. It exposes
chainable setters (`set_dim`, `set_stride`, `set_data_type`, …) and
corresponding getters. There is currently no concept of a per-tensor
reference to another tensor in the same graph.

### 3.2 Flatbuffer: `tensor_attributes.fbs`

`tensor_attributes.fbs` carries the persistent representation of
`TensorAttributes` used for graph serialization (and, in the
Compiled-Plan Serialization path, for plan import/export). Schema
evolution rules (RFC 0005) require appending optional defaulted
fields rather than reordering or repurposing existing fields.

### 3.3 Backend: tensor descriptor and variant pack

The backend tensor descriptor mirrors `TensorAttributes` and is
constructed from the frontend representation during graph lowering.
The variant pack at execute time carries `UID → void*` device-buffer
bindings. Neither layer currently carries any ragged-specific
information.

### 3.4 Data SDK: `ITensor` / `TensorBase<T>` / `Tensor<T>` / `ShallowTensor<T>`

`hipdnn_data_sdk::utilities` defines the runtime tensor hierarchy
used by the test harness and CPU references:

- **`ITensor`** — type-erased base: `dims()`, `strides()`,
  `elementSpace()`, `elementCount()`, `isPacked()`, `rawHostData()`,
  iteration via `LinearIndex` / `CompositeIndex` strategies.
- **`TensorBase<T>`** — adds typed addressing
  (`getHostValue(indices)`, `setHostValue(value, indices)`) and a
  typed view of memory.
- **`Tensor<T>`** — owning dense tensor backed by
  `MigratableMemory<T, HostAlloc, DeviceAlloc>`. Asserts
  `_packed = (elementCount == elementSpace)`.
- **`PinnedTensor<T>`** — alias for `Tensor<T>` with
  `PinnedHostAllocator<T>`.
- **`ShallowTensor<T>`** — non-owning wrapper over a borrowed
  `void*` plus dims/strides; provides dense iteration over
  `prod(dims)` elements.

`ITensorIterator` is implemented via a `std::variant` over the
known index strategies.

### 3.5 Test harness: `GraphTensorBundle` and `CpuReferenceGraphExecutor`

`GraphTensorBundle` holds `unordered_map<int64_t, unique_ptr<ITensor>>`
keyed by tensor UID. The integration-test harness walks
`TensorAttributes` and calls `createTensorFromAttribute(attr)` to
allocate each entry. Two bundles are created (one for the GPU plan,
one for the CPU reference) with identical random seeds; the same
inputs are placed into both, the two paths are executed, and
`verifyGraph` element-by-element compares outputs.

`CpuReferenceGraphExecutor` walks the graph in topological order,
allocates virtual intermediates as ordinary `Tensor<T>` instances,
and dispatches each node through its `<Op>Plan::execute(variantPack)`
which in turn calls into the corresponding CPU reference (e.g.
`CpuFpReferenceSdpa::forward(q, k, v, …)` typed as
`TensorBase<T>&`).

---

## Design

### 4.1 Overview

The design has four layers:

1. **Frontend / flatbuffer / backend** propagate two new fields per
   tensor: `ragged_offset_tensor_uid` and `alignment`. These are
   purely declarative; no new C-API entry points are needed.
2. **Data SDK** introduces two new tensor types: `IrregularTensor<T>`
   for owning storage with `physicalElementCount ≠ prod(dims)`, and
   `RaggedView<T, IndexT>` as a non-owning ragged-aware wrapper.
3. **`ITensor`** gains a small polymorphic index-strategy hook so
   ragged iteration can supply a `RaggedCompositeIndex`. `TensorBase`
   gains an optional `validSeqLen(b)` so CPU references can bound
   inner loops without `dynamic_cast`.
4. **Plan layer** (per op) becomes the assembly point that, at execute
   time, reads `ragged_offset` from each ragged input's
   `TensorAttributes` and `seq_lens` from the node-level op
   attributes, constructs a `RaggedView` over the variant-pack
   pointer, and hands it to the CPU reference as `TensorBase<T>&`.
5. **Test harness** gains a pre-supplied input bundle that lets the
   test author hand deterministic values for aux tensors (and, if
   desired, any other input tensor) directly to the harness, avoiding
   meaningless random initialization for tensors whose values must
   satisfy structural invariants.

### 4.2 Frontend: `TensorAttributes` additions

`Tensor_attributes` gains chainable setters / getters for the two
new fields, following the existing `set_*` precedent:

```cpp
auto
set_ragged_offset(std::shared_ptr<Tensor_attributes> const& value)
    -> Tensor_attributes&
{
    ragged_offset = value;
    return *this;
}

auto
get_ragged_offset() const -> std::shared_ptr<Tensor_attributes>;

auto
set_alignment(int64_t alignmentInElements) -> Tensor_attributes&;

auto
get_alignment() const -> int64_t;   // default 1
```

`set_alignment` declares a per-batch alignment requirement of the
physical buffer, in elements. The default value of `1` (no
alignment) matches what AITER FMHA currently needs and means the
physical buffer is exactly the packed valid size; larger alignment
values force each batch's contribution to be rounded up to a multiple
of the alignment.

**Frontend validation.** When `validate()` is called on the graph,
ragged primaries are checked structurally:

- The `ragged_offset` aux tensor exists in the graph (by UID).
- The aux tensor has rank 1.
- The aux tensor's first dimension equals `B + 1`, where `B` is the
  primary's first dimension.
- `get_alignment() >= 1`.

Per-batch valid-length validation (matching `seq_lens` to the
primary) does not belong on `TensorAttributes` per the constraint in
[§2.4](#24-constraint-seq_lens-is-not-on-tensorattributes); per-op
validation in (e.g.) `SdpaAttributes::validate()` handles it.

**Reported `dims()`.** The primary tensor's reported `dims()[1]`
remains the max padded sequence length (`S_max`). This matches the
semantics needed by overridable-shape consumers and keeps the graph
representation deterministic without re-deriving `S_max` from
`ragged_offset` at runtime.

### 4.3 Flatbuffer schema additions

`tensor_attributes.fbs` gains two appended optional defaulted fields:

```
ragged_offset_tensor_uid: long = null;
alignment:                long = 1;
```

Both are wire-compatible additions per RFC 0005: graphs serialized by
older binaries deserialize cleanly in newer ones with `null` and
`1` as defaults. No `seq_lens_tensor_uid` is added to
`tensor_attributes.fbs` (the existing per-op attribute tables — e.g.
`SdpaAttributes` — already carry `seq_len_q_tensor_uid` /
`seq_len_kv_tensor_uid` for ops that consume them, per the constraint
in [§2.4](#24-constraint-seq_lens-is-not-on-tensorattributes)).

### 4.4 Backend wiring

The backend tensor descriptor mirrors the frontend additions: it
gains `ragged_offset_tensor_uid` (optional) and `alignment` (default
1). These are exposed through existing get/set attribute paths under
new tensor-descriptor enum values in the reserved hipDNN extension
range. No new `hipdnnBackend*` entry points are introduced, and the
variant-pack representation is unchanged: at execute time the variant
pack still carries `UID → void*` for every tensor in the graph,
including ragged primaries and their aux buffers.

### 4.5 Data SDK: shared elements across the design

These apply to both new tensor types:

1. **Aux tensors are `std::shared_ptr<TensorBase<IndexT>>`**, with
   `IndexT` templated (defaulting to `int32_t`, with `int64_t` also
   permitted).
2. **Auxiliaries are constructor-only / immutable** on the ragged
   view object — the view never mutates which aux tensors it points
   at.
3. **`dims()[1] == S_max`** by construction. `elementSpace()` is
   decoupled from `prod(dims)` and equals the physical buffer size of
   the underlying storage.
4. **Polymorphic index-strategy hook on `ITensor`**: introduce
   `virtual std::unique_ptr<ITensorIndex> makeIndex(bool isEnd) const`
   so ragged iteration can supply a `RaggedCompositeIndex` that walks
   only valid elements. The existing `LinearIndex` / `CompositeIndex`
   strategies move behind this interface; non-ragged behaviour is
   unchanged. `ITensorIterator`'s `std::variant` does not grow.
5. **Optional `virtual int64_t validSeqLen(int64_t b) const` on
   `TensorBase`** (default `dims()[1]`) so CPU references can bound
   inner loops without `dynamic_cast`.
6. **Iteration semantics for `elementCount()`**: if `seq_lens` is
   attached, `elementCount()` returns the sum of valid elements
   implied by `seq_lens`. Otherwise it returns the underlying
   physical element count.

### 4.6 Data SDK: `IrregularTensor<T>` (owning storage)

A memory-owning buffer whose physical element count is independent
of `prod(dims)`. Used as the underlying storage for ragged views and
as the natural type for graph intermediates that don't need
iteration.

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
    //   begin()/end()/cbegin()/cend() iterate `physicalElementCount`
    //   contiguous elements. See the iteration-policy note below.

private:
    MigratableMemory<T, HostAlloc, DeviceAlloc> _memory;
    std::vector<int64_t>                        _paddedDims;
    std::vector<int64_t>                        _strides;
    size_t                                      _physicalElementCount;
};
```

**Pinned host memory** is supported via
`IrregularTensor<T, PinnedHostAllocator<T>, …>`, the same pattern
`PinnedTensor<T>` uses for `Tensor<T>`.

**Iteration policy.** An earlier formulation had `IrregularTensor`'s
iterators throw `std::logic_error` to rule out walks over a buffer
with no well-defined dense iteration order. The `seq_lens` constraint
forced a re-evaluation: the test harness's bundle layer has call
sites (`randomizeTensor`, output validation) that need to walk the
buffer but cannot look up `seq_lens` from a `TensorAttributes` alone.
Throwing would force the harness into a wrapping step
(node-attribute walk to identify if there is an associated `seq_lens`
tensor + view construction + ragged-aware iteration) that is both
complicated and unnecessary, because for the harness's purposes a
linear walk over the physical buffer produces correct results (see
[§4.11](#411-test-harness-integration)). The current policy is
therefore linear iteration; ragged-aware iteration that respects
per-batch valid lengths is the job of `RaggedView<T>`. The
alternative is recorded in
[§6.3](#63-throwing-iterators-on-irregulartensort).

### 4.7 Data SDK: `RaggedView<T, IndexT>` (non-owning ragged view)

A non-owning wrapper that turns any `TensorBase<T>`-derived
underlying storage into a fully iterable ragged tensor by attaching
the required aux buffers.

```cpp
template <typename T, typename IndexT = int32_t>
class RaggedView : public TensorBase<T>
{
public:
    // Shape, strides, and physicalElementCount come authoritatively
    // from the underlying tensor's dims()/strides()/elementSpace().
    // They are NOT passed in — this removes "view disagrees with its
    // underlying" bugs by construction.
    //
    // The underlying is `shared_ptr<TensorBase<T>>` so the view can
    // wrap either an owning `IrregularTensor<T>` (user / bundle case)
    // OR a non-owning `ShallowTensor<T>` over a variant-pack `void*`
    // (plan-layer case).
    RaggedView(std::shared_ptr<TensorBase<T>>       underlying,
               std::shared_ptr<TensorBase<IndexT>>  raggedOffset,        // required
               std::shared_ptr<TensorBase<IndexT>>  seqLens = nullptr);  // optional

    // ITensor / TensorBase<T> overrides:
    //   dims()         -> _underlying->dims()         (forwarded)
    //   strides()      -> _underlying->strides()      (forwarded)
    //   elementSpace() -> _underlying->elementSpace() (forwarded)
    //   memory()       -> _underlying->memory()       (forwarded)
    //   elementCount() -> sum-of-seqLens if _seqLens,
    //                     else _underlying->elementSpace()
    //   isPacked()     -> false
    //   begin/end      -> RaggedCompositeIndex strategy via the
    //                     polymorphic index hook (walks only valid
    //                     elements)

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
- `_raggedOffset->elementCount() == B + 1`, where
  `B = _underlying->dims()[0]`.
- If `_seqLens != nullptr`: `_seqLens->elementCount() == B`.
- `_raggedOffset` and `_seqLens` (if present) have rank 1.

**Why the underlying is typed as `TensorBase<T>` rather than
`IrregularTensor<T>`.** The intuitive choice for the underlying
would be `IrregularTensor<T>`, since that is the owning storage type
for ragged primaries in the bundle and in executor intermediates.
But the plan layer at execute time only has a `void*` from the
variant pack — there is no `IrregularTensor` to point at. Widening
the underlying to `TensorBase<T>` accommodates both the owning case
and a non-owning `ShallowTensor<T>` wrap of the variant-pack pointer
without introducing a third tensor type.

### 4.8 Tiered API: which type each role uses

| Role | Type | Iteration |
|---|---|---|
| Owning storage for a ragged primary (bundle / executor intermediate) | `IrregularTensor<T>` | Linear over physical buffer |
| Ragged inputs/outputs at the CPU reference plan layer | `RaggedView<T, IndexT>` over a `ShallowTensor<T>` of the variant-pack pointer | Full ragged iteration |
| Ragged tensor a user iterates directly (sample flow) | `RaggedView<T, IndexT>` over the owning `IrregularTensor<T>` | Full ragged iteration |
| Aux tensors (`ragged_offset`, `seq_lens`) | Plain `Tensor<IndexT>` via `shared_ptr` | Dense iteration |
| Graph intermediates that no one walks | `IrregularTensor<T>` directly | Linear (incidental) |
| Non-ragged tensors | Plain `Tensor<T>` | Dense iteration |

### 4.9 Wiring sources of truth

- **Graph (`TensorAttributes`)** carries `ragged_offset` (pointer to
  the aux tensor's attributes) and `alignment` (per-batch alignment
  in elements). See [§4.2](#42-frontend-tensorattributes-additions).
- **Node-level op attributes** (e.g. `SdpaAttributes`) carry the
  `seq_lens` UIDs through their existing `set_seq_len_q(...)` /
  `set_seq_len_kv(...)` accessors. The per-op attribute flatbuffer
  tables already carry (or will carry) the corresponding
  `seq_len_*_tensor_uid` fields.
- **Runtime tensorMap (`UID → ITensor`)** carries only the dense
  buffers: an `IrregularTensor<T>` for each ragged primary, ordinary
  `Tensor<IndexT>` for each aux, ordinary `Tensor<T>` for non-ragged
  tensors. **No ragged-view objects at this layer.**
- **Plan layer (executor)**: when executing an op whose graph
  declares a given input/output as ragged, the plan constructs the
  view at the call site, reading `ragged_offset` from
  `TensorAttributes::ragged_offset_tensor_uid()` and `seq_lens` from
  the node's op-specific attribute.

### 4.10 Plan-layer view construction and CPU reference impact

The physical element count needed to construct the underlying
`ShallowTensor<T>` is cached on the plan's `params` struct at
plan-build time (recovered from the `ragged_offset` aux's last
element together with the alignment — see
[§4.11.2](#4112-bundle-allocation-in-createtensorfromattribute)):

```cpp
auto qUnderlying = std::make_shared<ShallowTensor<QType>>(
    variantPack.at(_params.qTensor.uid),
    _params.qTensor.dims,
    _params.qTensor.strides,
    _params.qTensor.physicalElementCount);   // cached at plan build

auto qRaggedOffset = std::make_shared<ShallowTensor<IndexT>>(/* ... */);

auto qSeqLens = _params.seqLenQTensor.has_value()
    ? std::make_shared<ShallowTensor<IndexT>>(/* ... */)
    : nullptr;

auto qView = std::make_shared<RaggedView<QType, IndexT>>(
    qUnderlying, qRaggedOffset, qSeqLens);

// pass *qView as TensorBase<QType>& into CpuFpReferenceSdpa::forward
```

`CpuFpReferenceSdpa::forward`'s signature is **unchanged**
(`TensorBase<T>&`). The view is built in the plan layer and passed
in. Body changes are limited to per-batch sequence-length bounding
using `q.validSeqLen(b)` / `k.validSeqLen(b)`, with an early-out at
the top of the parallel lambda to skip `(b, sq)` pairs where
`sq >= q.validSeqLen(b)`. The parallel decomposition continues to
use `S_max` so the work scheduler doesn't have to know about
ragged-ness.

### 4.11 Test-harness integration

Most stages of the integration-test harness flow
(`IntegrationGraphVerificationHarness` → `GraphTensorBundle` →
`CpuReferenceGraphExecutor` → per-op plan → CPU reference) are not
affected by this design beyond the type substitutions described
above. Four stages warrant explicit discussion: the pre-supplied
input bundle that aux tensors require, bundle allocation, input
randomization, and output validation.

#### 4.11.1 Pre-supplied input bundle

Ragged-aux tensors (`ragged_offset`, `seq_lens`) cannot be
meaningfully initialized by random fill: their values must be
structurally consistent (e.g. `ragged_offset` monotonic with
`ragged_offset[0] == 0`, `seq_lens[b]` within the per-batch reserved
extent) and consistent with the rest of the test scenario. The
harness therefore accepts a **pre-supplied input bundle** at setup
time:

```cpp
using PreSuppliedInputs =
    std::unordered_map<int64_t, std::unique_ptr<ITensor>>;

class IntegrationGraphVerificationHarness {
public:
    IntegrationGraphVerificationHarness(/* graph, etc. */,
                                        PreSuppliedInputs preSuppliedInputs);
    // ...
};
```

The test author constructs each pre-supplied entry as an owning
`Tensor<T>` (or `Tensor<IndexT>` for aux tensors) populated with
deliberately chosen host-side values, and hands the single map to
the harness. Only one map is needed: each `ITensor` owns both a
host buffer and a device buffer via `MigratableMemory`, so the same
pre-supplied tensor serves both execution paths — the GPU path
reads `rawDeviceData()` (with `MigratableMemory` migrating the
host-side values to the device before kernel launch), and the CPU
reference path reads `rawHostData()` directly. The pre-supplied
bundle is owned by the harness, not by either `GraphTensorBundle`,
so cross-bundle ownership concerns do not arise.

The harness handles the map as follows:

1. **Hold.** The harness retains ownership of the pre-supplied
   bundle for its lifetime. The per-execution-path
   `GraphTensorBundle`s continue to exist and continue to own the
   *non*-pre-supplied tensors (auto-allocated, random-filled).
   The harness's logical "UID → tensor" lookup is now a
   pre-supplied-first fallthrough: pre-supplied bundle, then the
   relevant `GraphTensorBundle`.
2. **Walk.** When the per-`TensorAttributes` allocation pass visits
   each UID, it first checks the pre-supplied bundle: if the UID
   is present there, the walk skips both allocation and the
   subsequent randomize call for that UID in either
   `GraphTensorBundle`. Otherwise it allocates and randomizes as
   today, separately in each bundle.
3. **Strict requirement check.** Before the walk, the harness
   verifies that every aux UID actually required by the graph is
   present in the pre-supplied bundle:
   - For every `TensorAttributes` with `has_ragged_offset()`, the
     referenced `ragged_offset` UID must be in the pre-supplied
     bundle.
   - For every node-level op attribute that references a `seq_lens`
     UID (e.g. `SdpaAttributes::seq_len_q_tensor_uid()` and
     `seq_len_kv_tensor_uid()`), the referenced UID must be in the
     pre-supplied bundle.

   Missing aux UIDs cause the harness to fail setup with a clear
   error rather than silently random-initialize a structurally
   invalid aux tensor.

**Variant-pack construction.** When the harness builds the GPU or
CPU variant pack for an execute call, it iterates the graph's
tensor UIDs and looks each one up through the
pre-supplied-first fallthrough. UIDs resolved from the pre-supplied
bundle contribute the *same* `void*` to both bundles' variant packs
(via the same `ITensor`'s `rawDeviceData()` / `rawHostData()`);
UIDs resolved from each `GraphTensorBundle` contribute that bundle's
own buffer. This automatically guarantees that the GPU and CPU
paths see byte-equal input values for any pre-supplied tensor —
there is only one host buffer feeding both paths.

**General override semantics.** The pre-supplied bundle is not
restricted to aux UIDs: a test author may also supply non-aux inputs
to pin them to fixed values (useful for reproducing a specific
failing case, golden-value testing, or any scenario where the
random-seeded path is the wrong choice). The strict check only
rejects *missing* aux entries; *extra* non-aux entries are accepted
and override the would-be random fill. Because both paths read the
same `ITensor`, GPU and CPU bundles automatically agree on the
pre-supplied input values regardless of what they are.

The pre-supplied bundle is purely a harness-layer mechanism. The
data SDK types (`IrregularTensor<T>`, `RaggedView<T, IndexT>`),
the plan layer, the CPU references, and the variant pack are all
unaffected by it.

#### 4.11.2 Bundle allocation in `createTensorFromAttribute`

With the pre-supplied bundle held alongside each
`GraphTensorBundle`, ragged-primary allocation collapses to a
single pass. For each `TensorAttributes`:

- If the UID is in the pre-supplied bundle: skip — already
  resolved.
- Else if `attr.has_ragged_offset() == false`: allocate into the
  `GraphTensorBundle` as today (`Tensor<T>` sized by `prod(dims)`).
- Else: look up the `ragged_offset` UID in the pre-supplied bundle
  (the strict check in [§4.11.1](#4111-pre-supplied-input-bundle)
  guarantees it is present), read its host values, compute the
  physical element count, and allocate into the `GraphTensorBundle`
  via
  `make_unique<IrregularTensor<T>>(dims, strides, physicalElementCount)`.

The physical element count is computed as:

```
physicalElementCount =
    sum_b align_up(ragged_offset[b+1] - ragged_offset[b],
                   attr.get_alignment())
```

For the default `alignment = 1` this reduces to `ragged_offset[B]`
(the packed valid size) — the natural answer for AITER FMHA and any
other kernel that has no per-batch alignment requirement. For larger
alignment values, each batch's contribution is rounded up to a
multiple of the alignment, producing a physical buffer that is
strictly larger than `ragged_offset[B]` and matches the kernel's
per-batch padding requirement.

The plan layer is given the same `physicalElementCount` value
(cached in its `params`) at plan-build time using the same lookup,
so neither the plan nor the executor re-derives it at execute time.

No structural change to `GraphTensorBundle` is required: storage
stays `unordered_map<int64_t, unique_ptr<ITensor>>`; no `shared_ptr`
shift, no aliasing ctor, no two passes.

Because the view is not built at this layer, the bundle does not
need to know `seq_lens` to allocate the primary. The user-facing
flow naturally mirrors the same idea: the user constructs and
populates `ragged_offset` directly, then allocates an
`IrregularTensor` sized from the same `(ragged_offset, alignment)`
pair.

#### 4.11.3 Input randomization at bundle init

Per [§4.11.1](#4111-pre-supplied-input-bundle), any UID in the
pre-supplied bundle is skipped by randomization. The remaining
inputs are randomized as today.

For a *non-pre-supplied* ragged primary, `bundle.randomizeTensor(uid)`
walks the underlying `IrregularTensor<T>`'s iterator. This walks the
entire physical buffer linearly, which fills more elements than
strictly valid (the per-batch padding tails are also randomized).
This is harmless: neither the GPU kernel nor the CPU reference reads
padding positions, so the values placed there don't affect
correctness. Since both bundles are seeded identically, padding
values stay byte-equal between the two paths.

This is the principal reason `IrregularTensor`'s iterators walk the
physical buffer rather than throwing. Throwing would have forced the
harness to perform a node-attribute walk to discover `seq_lens` and
wrap the bundle entry in a `RaggedView` purely so randomization
could be ragged-aware — none of which is necessary for the actual
correctness requirement.

#### 4.11.4 Output validation in `verifyGraph`

This stage requires real care **even with deliberate input aux
values**, because the issue here is on the output side. The CPU
reference (e.g. `CpuFpReferenceSdpa::forward`) writes only the valid
sequence positions of an output, leaving padding at its
randomize-time value. The GPU kernel, however, has **no contractual
guarantee** about what it does to output padding — AITER FMHA
forward kernels in particular may leave padding alone, zero it out,
or write garbage from registers. A naive full-buffer element-by-
element comparison therefore fails for reasons unrelated to
algorithmic correctness.

**Recommended approach: sentinel-skip.** At harness init time, fill
each ragged output's physical buffer (both bundles) with a known
sentinel value the CPU reference cannot legitimately produce —
typically NaN for floating-point types, or a magic number for
integer types — instead of randomizing it. After execution the
validator iterates the full physical buffer; for any position where
the CPU side is bit-equal to the sentinel, the CPU reference did
not write there (i.e. it is padding), so the compare is skipped.
For non-sentinel positions the existing tolerance-based equality
check is applied.

This approach:

- Requires no node-attribute walk to discover `seq_lens` at the
  validation site.
- Requires no per-output `RaggedView` construction.
- Adds one validator overload
  (`allCloseSkipSentinel(cpu, gpu, sentinel, tolerance)`).
- Adds a small init-time branch to fill ragged outputs with the
  sentinel.

**Fallback: seq_lens-aware wrap.** If a future op turns out to have
a CPU reference that can naturally produce the sentinel value, the
fallback is for the harness to walk node-level op attributes to find
each output's `seq_lens` UID, look up the runtime aux in the bundle,
and construct a `RaggedView` over the bundle's `IrregularTensor`
with `seq_lens` attached for the compare. Constructing this wrap
from existing primitives is straightforward — `RaggedView` already
accepts `seq_lens` at construction. (Note: ragged outputs' aux UIDs
are typically the same UIDs as their inputs' aux — `ragged_offset`
and `seq_lens` are reused across the SDPA tensors — so the lookup
naturally finds them in the pre-supplied bundle. For graphs in which
an output has its own aux UIDs not shared with any input, the lookup
falls through to the per-bundle `GraphTensorBundle` as usual.)

---

## Known limitations

1. **Packed-only.** Only the two AITER-driven configurations
   (`ragged_offset` only, `ragged_offset + seq_lens`) are in scope.
   A "padded, seq-lens-only" mode (no `ragged_offset`) is not
   supported by `RaggedView`'s current structural validation; see
   [§7.3](#73-padded-mode-seq-lens-only-no-ragged_offset).
2. **`seq_lens` is not per-tensor.** Per
   [§2.4](#24-constraint-seq_lens-is-not-on-tensorattributes), the
   `seq_lens` reference lives on node-level op attributes only. Any
   code that walks the graph one `TensorAttributes` at a time can
   identify ragged primaries and their `ragged_offset` but not their
   `seq_lens`. Discovery of `seq_lens` requires walking the
   node-level op attributes.
3. **`IrregularTensor<T>`'s `getHostValue` / `setHostValue` are
   semantically inert.** The methods are inherited from
   `TensorBase<T>` but have no meaningful per-batch addressing
   semantics on a ragged primary's physical buffer (the strides
   describe padded geometry; per-batch addressing requires the
   `ragged_offset` math that lives in `RaggedView`). They throw
   `std::logic_error` directing the caller at `RaggedView<T>` until
   [§7.1](#71-typedtensort-intermediate-in-the-tensor-hierarchy)
   lands.
4. **Sentinel-skip validation requires per-CPU-reference
   guarantees.** The sentinel value chosen for output validation
   (see [§4.11.4](#4114-output-validation-in-verifygraph)) must be a
   value the CPU reference for the op being tested cannot produce
   even pathologically. NaN is the safe default for floats, but
   edge cases (e.g. SDPA with `seq_lens_q[b] == 0` producing NaN
   from a fully masked row) may force a non-NaN magic-number
   sentinel for some ops. Low-precision floating-point types (bf16,
   fp16, fp8) have a smaller sentinel space and harder
   "unproducible by reference" guarantees.
5. **`elementCount()` contract differs between the two new types.**
   For `RaggedView` without `seq_lens`, `elementCount()` is
   `_underlying->elementSpace()` (the physical buffer size); for
   `IrregularTensor` it is `physicalElementCount`. Callers must be
   aware of the "valid elements" vs "physical buffer" distinction;
   each public API documents which semantic it uses.
6. **Test authors must construct aux tensors by hand.** Every
   integration test of a ragged-consuming op must build deliberate
   `ragged_offset` (and optionally `seq_lens`) tensors and supply
   them via the pre-supplied input bundle (see
   [§4.11.1](#4111-pre-supplied-input-bundle)). There is no
   harness-level helper for generating "a plausible random ragged
   layout" in this RFC; that would require introducing a
   ragged-layout generator with its own structural-validity rules
   and is left to follow-up work.

---

## Alternatives considered

### 6.1 Self-owning `RaggedTensor<T, IndexT>` subclass

An alternative shape was considered in which a
`RaggedTensor<T, IndexT>` class derived from `TensorBase<T>` owns
both its physical buffer (`MigratableMemory`) and its aux refs
(`shared_ptr<TensorBase<IndexT>>` for `ragged_offset` and optionally
`seq_lens`), constructed with all of these passed to its ctor:

```cpp
RaggedTensor(std::vector<int64_t> paddedDims,
             std::vector<int64_t> strides,
             size_t               physicalElementCount,
             std::shared_ptr<TensorBase<IndexT>> raggedOffset,        // required
             std::shared_ptr<TensorBase<IndexT>> seqLens = nullptr);  // optional
```

The user / bundle would hold a single owning `RaggedTensor` value
rather than a `(IrregularTensor, RaggedView)` pair, and the ragged
iteration / `validSeqLen` machinery would live on this class.

This alternate is not actually a one-type design. The plan layer at
execute time has only a `void*` from the variant pack — there is no
owning `RaggedTensor` to hand to the CPU reference. The existing
`ShallowTensor<T>` is dense-only and walks `prod(dims)` elements,
which is wrong for ragged primaries. So a non-owning peer is also
required: `ShallowRaggedTensor<T, IndexT>`, a ragged-aware analogue
of `ShallowTensor<T>` that takes a borrowed `void*` plus
dims/strides/physicalElementCount plus aux refs and provides the
same ragged iteration / `validSeqLen` semantics as `RaggedTensor`
without owning memory. The two types share their index/iteration
machinery; only memory ownership differs.

The alternative — keeping the alternate to one type by widening
`CpuFpReferenceSdpa::forward` (and every other op's CPU reference)
to take aux pointers + dims as extra arguments alongside the
`TensorBase<T>&` primary — was rejected because it forces every
reference signature to take ragged-specific parameters whether or
not the op is ragged. The two-type design (recommended vs alternate)
is the better choice along that axis.

**Pros relative to the recommended design**

- **Single owning object at the user-facing layer (samples).** A
  user holds one `RaggedTensor` whose `rawDeviceData()` goes into
  the variant pack and which is passed as `TensorBase<T>&` into CPU
  references. The recommended design splits this into an
  `IrregularTensor` plus optionally a `RaggedView`. This
  single-object ergonomic is the strongest argument for this
  alternate.
- **Pinned host memory** can be expressed via `RaggedTensor`'s
  template parameters
  (`RaggedTensor<T, IndexT, PinnedHostAllocator<T>, …>`), mirroring
  `PinnedTensor<T>`.

**Cons relative to the recommended design**

- **Same new-type count as the recommended design.** The need for
  `ShallowRaggedTensor` brings the total to two new types
  (`RaggedTensor` + `ShallowRaggedTensor`), matching the recommended
  design's `IrregularTensor` + `RaggedView`. The "smaller surface"
  intuition does not survive contact with the plan-layer
  construction site.
- **Bundle-side ctor-time aux dependency.** The pre-supplied input
  bundle (see [§4.11.1](#4111-pre-supplied-input-bundle)) owns aux
  tensors and is consulted alongside the per-path
  `GraphTensorBundle`s, so in the recommended design the
  allocation walk simply reads `ragged_offset`'s host values from
  the pre-supplied bundle to size each `IrregularTensor<T>`. This
  alternate additionally requires that the aux `shared_ptr` be
  threaded into the `RaggedTensor` ctor at primary-allocation time.
  That moves the bundle from "size the primary from values it can
  read" to "wire two sibling tensors together at construction" —
  which forces `GraphTensorBundle` to shift
  `unique_ptr → shared_ptr` (or expose an aliasing `share()`) and
  constrains the walk's visit order. The recommended design avoids
  both costs.
- **Awkward fit for graph intermediates that no one iterates.** An
  intermediate ragged buffer would either be allocated as a "no-aux"
  `RaggedTensor` (semantically odd: the class's whole point is the
  aux refs), or a separate `IrregularTensor`-style type would have
  to be introduced ad hoc — pushing the type count above two.
- **Output validation requires new infrastructure on the class
  itself.** The bundle holds the `RaggedTensor` with
  `seqLens == nullptr` (per the `seq_lens` constraint at the bundle
  layer); attaching a `seq_lens` for the seq_lens-aware compare
  requires either a second ctor accepting existing memory, a
  `setSeqLens` mutator (violating the immutability rule), or a
  wider `allClose` validator signature. None of this is needed in
  the recommended design — `RaggedView` already accepts `seq_lens`
  at construction. (This cost disappears in both designs if the
  sentinel-skip approach is adopted for output validation.)
- **No single source of truth on the runtime side.** The aux
  relationship is declared once on the graph (`TensorAttributes`)
  and again on the runtime ctor. In a hypothetical user-facing flow
  that took both declarations as inputs, a reconciler would be
  needed to verify they agree. (In the actual test-harness flow
  this reconciler is degenerate, because the variant pack waist
  erases the runtime declaration before it reaches the plan layer
  — but the asymmetry remains a smell.)

The deciding factor in favour of the recommended design is the
bundle / intermediate allocation simplicity, which holds
unconditionally and survives well into future ragged-aware
intermediates. The single-object user-facing ergonomic the alternate
offers is a real but smaller advantage that can be partly recovered
by a runtime-side `TensorBundle` factory (see
[§7.2](#72-runtime-side-tensorbundle-helper)).

### 6.2 Wrapper class around an existing dense `Tensor<T>`

An earlier sketch from the original problem statement proposed a
wrapper class that takes an existing dense `Tensor<T>` and overrides
its indexing and iterators with information from the ragged-aux
tensors. This was rejected because `Tensor<T>` asserts
`_packed = (elementCount == elementSpace)` and is sized at
construction by `prod(dims)`. There is no way to wrap a
`prod(dims)`-sized buffer to fake a `physicalElementCount`-sized
ragged primary without either:

- Allocating a `prod(dims) = B * S_max * …` buffer (much larger than
  the packed buffer kernels actually want), or
- Bypassing the dense allocation entirely (in which case the wrapper
  is constructing the storage anyway and the dense `Tensor<T>` adds
  nothing).

The recommended design's `IrregularTensor<T>` is the minimal
storage type that supports `physicalElementCount ≠ prod(dims)`; the
wrapper-over-`Tensor<T>` approach effectively re-derives that type
under a different name.

### 6.3 Throwing iterators on `IrregularTensor<T>`

The discarded design had `IrregularTensor<T>`'s iterators throw
`std::logic_error` to make it impossible to accidentally walk a
buffer whose dense iteration order has no well-defined meaning. The
intent was to force every iteration call site to first wrap the
storage in a `RaggedView<T>` that knows how to iterate ragged-aware.

This was rejected when the `seq_lens`-on-node-attributes constraint
became firm: the test harness's `randomizeTensor` and validation
call sites cannot look up `seq_lens` from a `TensorAttributes`
alone, so they cannot construct the required `RaggedView`. Forcing
them to walk node-level op attributes purely to enable iteration is
unnecessary complexity — a linear walk over the physical buffer
produces correct results for both call sites (see
[§4.11.3](#4113-input-randomization-at-bundle-init) and
[§4.11.4](#4114-output-validation-in-verifygraph)).

The recommended design's linear iteration on `IrregularTensor` is
the smaller, simpler choice. Callers who need ragged-aware
iteration use `RaggedView`; callers who only need to fill the
buffer or to compare it elementwise to another identically-shaped
`IrregularTensor` walk it directly.

### 6.4 Adding `seq_lens` to `TensorAttributes`

A symmetry-oriented design would put `set_seq_len(...)` on
`TensorAttributes` alongside `set_ragged_offset(...)`, so every
ragged primary carries both aux references. This was rejected for
two reasons:

- **cuDNN frontend-API compatibility.** Existing cuDNN frontend
  consumers configure `seq_lens` on op-level attributes (e.g.
  `SdpaAttributes::set_seq_len_q`). Mirroring `seq_lens` onto
  `TensorAttributes` would create two ways to express the same
  thing and require reconciliation between them.
- **Many-to-one binding.** In SDPA a single `seq_lens_q` tensor is
  shared by the Q primary and a single `seq_lens_kv` is shared by
  the K and V primaries. Putting `seq_lens` on `TensorAttributes`
  would either store the same `shared_ptr` redundantly on multiple
  tensor attributes, or invent a per-op binding rule that the
  node-level location already expresses naturally.

The recommended design keeps `seq_lens` on node-level op attributes
and accepts the consequences for code that walks the graph one
`TensorAttributes` at a time (see
[§4.11](#411-test-harness-integration) for how those consequences
are managed).

### 6.5 Random initialization of aux tensors with structural fixup

An earlier formulation of the test-harness flow had aux tensors
(`ragged_offset`, `seq_lens`) initialized by the existing random
fill, then "fixed up" before use — e.g. sort `ragged_offset` to
restore monotonicity, clamp `seq_lens[b]` to the per-batch reserved
extent, force `ragged_offset[0] = 0`. The harness would derive a
ragged layout from whatever values came out of the random seed,
making setup require no extra arguments from the test author.

This was rejected for three reasons:

- **Loss of test determinism in a non-obvious way.** A small change
  to seed handling or to the fixup heuristic silently changes the
  per-batch sequence lengths every test sees, which can mask or
  surface bugs unrelated to the change. Tests that pass today and
  fail tomorrow because of an upstream random-seed refactor are
  much harder to debug than tests with explicit aux values.
- **Fixup heuristics encode an implicit policy.** "How much padding
  to leave per batch?", "what fraction of `S_max` should the average
  `seq_lens[b]` be?", and "are degenerate cases like
  `seq_lens[b] == 0` allowed?" are real decisions that affect what
  the test is actually testing. Burying them in a fixup function
  in the harness makes them invisible to the test author.
- **No actual saving of work for the test author.** The author still
  has to reason about which sequence-length distribution they want
  to exercise; an explicit aux bundle just makes that reasoning
  visible at the test site. The pre-supplied bundle mechanism
  ([§4.11.1](#4111-pre-supplied-input-bundle)) costs no more
  cognitive effort than understanding what the fixup function would
  have produced — and is much easier to audit.

A future helper that generates "a plausible random ragged layout"
on top of the pre-supplied bundle interface is reasonable
(Known limitation #6 in [§5](#known-limitations)); the harness
should not silently apply such a generator without the test author
having asked for it.

---

## Future work

These are larger items that the recommended design accommodates but
does not require, and that would be appropriate to tackle as
separate efforts.

### 7.1 `TypedTensor<T>` intermediate in the tensor hierarchy

`IrregularTensor<T>` currently derives from `TensorBase<T>`, which
carries the `getHostValue(indices)` / `setHostValue(value, indices)`
multi-dim addressing API. Those methods have no meaningful semantics
on a ragged primary's physical buffer (the strides describe padded
geometry, but per-batch addressing requires the `ragged_offset` math
that lives in `RaggedView`). The recommended design accepts this
mismatch — callers who hold an `IrregularTensor` are expected to
wrap it in a `RaggedView` before addressing it — but the type system
does not enforce it.

A cleaner end state would introduce a `TypedTensor<T>` interface
between `ITensor` and `TensorBase<T>`:

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

`RaggedView`'s underlying would re-type as
`shared_ptr<TypedTensor<T>>` instead of `shared_ptr<TensorBase<T>>`,
which accepts both the owning `IrregularTensor<T>` (bundle / user /
output-validation cases) and a non-owning `ShallowTensor<T>`
(plan-layer case) — the same one-type-fits-both property the design
relies on today, just at a narrower interface that doesn't drag
along the meaningless multi-dim indexing API. `memory()` plumbing
continues to work (`MigratableMemory<T>&` is exposed on
`TypedTensor<T>`, so `RaggedView::memory()` remains a clean
forwarder).

This is a refactor of `Tensor.hpp` that introduces one new public
class and shifts a couple of method declarations between bases. It
is not invasive — nothing changes for callers of `TensorBase<T>` —
but it touches a foundational header and is worth taking on its own
merits rather than as part of the ragged-tensors patch. Until it
lands, `IrregularTensor<T>` derives from `TensorBase<T>` and
overrides `getHostValue` / `setHostValue` to throw
`std::logic_error` with a message directing the caller at
`RaggedView<T>`.

### 7.2 Runtime-side `TensorBundle` helper

A `UID → owning_runtime_tensor` helper on the user side, with a
factory that for non-ragged attrs builds a `Tensor<T>` and for
ragged attrs handles the pattern of constructing `ragged_offset`
from user-supplied values and then allocating an `IrregularTensor<T>`
sized from `(ragged_offset, alignment)` plus, on demand, a
`RaggedView` resolved against the node attributes. This collapses
the user's API surface to a one-call-per-`TensorAttributes` pattern
regardless of ragged-ness and hides the two-object cost of this
design. Out of scope for this RFC but the natural direction for the
user-facing API.

A companion helper on the harness side could generate plausible
random ragged layouts and emit them as pre-supplied input bundles
(see [§5](#known-limitations) item 6), reducing boilerplate for
authors of conformance tests where the specific ragged layout
doesn't matter as long as it is well-formed.

### 7.3 Padded mode (seq-lens only, no `ragged_offset`)

A "padded, seq-lens-only" mode (in which the physical buffer is
exactly `prod(dims) = B * S_max * …` elements and only `seq_lens`
distinguishes valid from padded entries) is straightforward to add
later because `RaggedView`'s underlying is already
`shared_ptr<TensorBase<T>>` — pass a `Tensor<T>` underlying and a
null `ragged_offset`, with the structural validation relaxed
accordingly. Non-breaking against the API above.

### 7.4 Sentinel selection per CPU reference

Document the sentinel choice and the "the reference cannot produce
this value" guarantee in each CPU reference header. NaN is the
default safe choice for floats, but watch for edge cases — e.g.
SDPA forward with `seq_lens_q[b] == 0` can produce NaN naturally
(`log(0)` from a fully masked row), in which case a non-NaN
magic-number sentinel is required. For low-precision floating-point
types (bf16, fp16, fp8) the sentinel space is smaller and
"unproducible by reference" is harder to guarantee. A future RFC
could codify the sentinel-per-dtype-per-op table.
