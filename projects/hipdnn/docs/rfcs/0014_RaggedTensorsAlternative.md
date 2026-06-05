# hipDNN: Ragged Tensors Design Document

- Contributors: hipDNN maintainers
- **Status**: Draft

## Table of Contents
1. [Summary](#summary)
2. [Problem Statement](#problem-statement)
   - 2.1 [What is a ragged tensor](#21-what-is-a-ragged-tensor)
   - 2.2 [hipDNN gap](#22-hipdnn-gap)
   - 2.3 [Configurations targeted in this iteration](#23-configurations-targeted-in-this-iteration)
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
   - 4.5 [Data SDK: shared elements](#45-data-sdk-shared-elements)
   - 4.6 [Data SDK: `RaggedTensor<T, IndexT>` (owning, ragged-aware)](#46-data-sdk-raggedtensort-indext-owning-ragged-aware)
   - 4.7 [Data SDK: `ShallowRaggedTensor<T, IndexT>` (non-owning peer)](#47-data-sdk-shallowraggedtensort-indext-non-owning-peer)
   - 4.8 [Tiered API: which type each role uses](#48-tiered-api-which-type-each-role-uses)
   - 4.9 [Wiring sources of truth](#49-wiring-sources-of-truth)
   - 4.10 [Plan-layer construction and CPU reference impact](#410-plan-layer-construction-and-cpu-reference-impact)
   - 4.11 [Test-harness integration](#411-test-harness-integration)
5. [Known limitations](#known-limitations)
6. [Alternatives considered](#alternatives-considered)
7. [Future work](#future-work)

---

## Summary

A ragged tensor is logically `[B, X, …]` where the per-batch extent
`X` varies. Physical memory is a single contiguous buffer indexed
by a `ragged_offset[B+1]` aux tensor: batch `b` occupies the
contiguous range `[ragged_offset[b], ragged_offset[b+1])`.

This RFC adds end-to-end ragged-tensor support:

1. **Frontend (`TensorAttributes`)** gains `set_ragged_offset` /
   `get_ragged_offset` and `set_alignment` / `get_alignment`.
2. **Flatbuffer schema** gains defaulted `ragged_offset_tensor_uid`
   and `alignment` fields (wire-compatible per RFC 0005).
3. **Backend** propagates the new fields through existing
   get/set-attribute paths; no new C-API entry points; variant pack
   unchanged.
4. **Data SDK** adds a single owning type `RaggedTensor<T, IndexT>`
   (memory + `ragged_offset` aux ref) and its non-owning peer
   `ShallowRaggedTensor<T, IndexT>`. Both expose ragged-aware
   iteration over `ragged_offset` ranges.
5. **Plan layer** wraps the variant-pack pointer in a
   `ShallowRaggedTensor` per execute and passes it to the CPU
   reference as `TensorBase<T>&` — reference signatures gain only
   `seq_lens` as a separate input where applicable.
6. **Integration-test harness** accepts a *pre-supplied input
   bundle* — `unordered_map<int64_t, shared_ptr<ITensor>>` keyed
   by UID — so the structurally-constrained `ragged_offset` aux
   can carry deliberate values shared between the GPU and CPU
   paths via each `ITensor`'s built-in host/device
   `MigratableMemory`.

`seq_lens` is intentionally **not** part of the ragged-tensor
abstraction. It is a tensor in the graph that ops reference
through their own node-level attribute APIs (e.g.
`SdpaAttributes::set_seq_len_q`); CPU references and kernels look
it up from the variant pack like any other input. The case for
this decoupling is given in
[§6.2](#62-keeping-seq_lens-in-the-ragged-tensor-sdk-abstraction).

The immediate consumer is SDPA on AITER FMHA kernels.

---

## Problem Statement

### 2.1 What is a ragged tensor

A logically `[B, X, …]` tensor where `X` (typically the sequence
dimension) varies per batch. Batch `b` occupies the contiguous
range `[ragged_offset[b], ragged_offset[b+1])` of the physical
buffer.

Some ops additionally need to distinguish *valid* sequence
positions within each batch's range from *trailing padding*
positions kept for alignment. That information is carried by a
separate `seq_lens[B]` tensor referenced through the op's
node-level attributes (e.g. `SdpaAttributes::set_seq_len_q`) —
matching cuDNN frontend-API conventions. `seq_lens` is independent
of the ragged-tensor abstraction: a ragged primary always reports
the full per-batch range from `ragged_offset` and never consults
`seq_lens` itself. Ops that care about per-batch valid lengths
read `seq_lens` from the variant pack directly.

Both `ragged_offset` and `seq_lens` aux tensors may be referenced
by more than one primary in the same graph — SDPA, for example,
typically shares a single `seq_lens_kv` across the K and V
primaries and may share a `ragged_offset_kv` between them as well.
The design supports this naturally: each primary's
`TensorAttributes` references the aux by UID; the runtime holds
one `ITensor` per UID; multiple `RaggedTensor`s share the same
`ragged_offset` aux via `shared_ptr`.

### 2.2 hipDNN gap

hipDNN's tensor model in both the frontend (`TensorAttributes`)
and the data SDK (`Tensor.hpp`) currently assumes a tensor's
physical element count equals `prod(dims)`. There is no way to
express:

- A primary tensor whose physical buffer is not `prod(dims)`
  elements long.
- A reference from one tensor (primary) to another (the aux
  `ragged_offset`).
- Iteration that walks per-batch ranges rather than a single dense
  `prod(dims)` walk.

Separately, the integration-test harness initializes inputs by
random fill. Random values are meaningless for `ragged_offset`: it
must be monotonically non-decreasing with `ragged_offset[0] == 0`
and its last element must agree with the primary's physical
buffer size. The harness needs a way to consume deliberate values
for this aux tensor.

### 2.3 Configurations targeted in this iteration

Both AITER FMHA configurations are uniformly representable. The
physical buffer is always exactly `ragged_offset[B]` elements
(modulo per-batch alignment, see
[§4.2](#42-frontend-tensorattributes-additions)):

- **Packed only.** Each batch contributes
  `ragged_offset[b+1] - ragged_offset[b]` rows of valid data; no
  internal padding.
- **Packed with per-batch trailing padding.** Each batch's range
  still goes from `ragged_offset[b]` to `ragged_offset[b+1]`, but
  only the first `seq_lens[b]` rows within that range are valid;
  the remaining
  `ragged_offset[b+1] - ragged_offset[b] - seq_lens[b]` rows are
  padding kept for alignment.

  Whether the trailing rows are padding is an op-level concern.
  The ragged tensor itself iterates the full per-batch range; the
  op's CPU reference or kernel queries `seq_lens` from the variant
  pack to decide what to do with positions `>= seq_lens[b]`.

A "padded, seq-lens-only" mode — in which the physical buffer is
exactly `prod(dims) = B * S_max * …` elements, every batch has
the same padded extent, and only `seq_lens` distinguishes valid
from padded entries — needs no new SDK type at all. The primary
in that mode has consistent strides and a non-ragged tensor
shape, so an ordinary `Tensor<T>` / `TensorBase<T>` represents it
correctly; the op's CPU reference looks up `seq_lens` from the
variant pack the same way the ragged case does. The mode is
therefore in scope by construction; this RFC's new SDK types only
become necessary when `physicalElementCount ≠ prod(dims)`.

---

## Existing Infrastructure

### 3.1 Frontend: `TensorAttributes`

`hipdnn_frontend::graph::Tensor_attributes` declares per-tensor
metadata used to build the operation graph (dims, strides, dtype,
UID, virtual flag, alignment, …) with chainable setters/getters.
There is currently no concept of a per-tensor reference to another
tensor in the same graph.

### 3.2 Flatbuffer: `tensor_attributes.fbs`

`tensor_attributes.fbs` carries the persistent representation of
`TensorAttributes` used for graph serialization (and Compiled-Plan
Serialization). Schema evolution rules (RFC 0005) require
appending optional defaulted fields rather than reordering or
repurposing existing fields.

### 3.3 Backend: tensor descriptor and variant pack

The backend tensor descriptor mirrors `TensorAttributes` and is
constructed from the frontend representation during graph
lowering. The variant pack at execute time carries
`UID → void*` device-buffer bindings. Neither layer currently
carries any ragged-specific information.

### 3.4 Data SDK: `ITensor` / `TensorBase<T>` / `Tensor<T>` / `ShallowTensor<T>`

`hipdnn_data_sdk::utilities` defines the runtime tensor hierarchy:

- **`ITensor`** — type-erased base: `dims()`, `strides()`,
  `elementSpace()`, `elementCount()`, `isPacked()`,
  `rawHostData()`, iteration via `LinearIndex` / `CompositeIndex`
  strategies.
- **`TensorBase<T>`** — adds typed addressing
  (`getHostValue` / `setHostValue`).
- **`Tensor<T>`** — owning dense tensor backed by
  `MigratableMemory<T, HostAlloc, DeviceAlloc>`. Asserts
  `_packed = (elementCount == elementSpace)`.
- **`PinnedTensor<T>`** — `Tensor<T>` with
  `PinnedHostAllocator<T>`.
- **`ShallowTensor<T>`** — non-owning wrapper over a borrowed
  `void*` plus dims/strides; dense iteration over `prod(dims)`.

`ITensorIterator` is implemented via a `std::variant` over the
known index strategies.

### 3.5 Test harness: `GraphTensorBundle` and `CpuReferenceGraphExecutor`

`GraphTensorBundle` holds the runtime tensors keyed by UID. The
integration-test harness walks `TensorAttributes` and calls
`createTensorFromAttribute(attr)` to allocate each entry. Two
bundles (GPU plan, CPU reference) are created with identical
random seeds; outputs are compared element-by-element by
`verifyGraph`.

`CpuReferenceGraphExecutor` walks the graph in topological order,
allocates virtual intermediates, and dispatches each node through
its `<Op>Plan::execute(variantPack)` which in turn calls into the
corresponding CPU reference (e.g.
`CpuFpReferenceSdpa::forward(q, k, v, …)` typed as
`TensorBase<T>&`).

---

## Design

### 4.1 Overview

1. **Frontend / flatbuffer / backend** propagate
   `ragged_offset_tensor_uid` and `alignment` per tensor.
   Declarative only; no new C-API entry points.
2. **Data SDK** introduces a single new owning type
   `RaggedTensor<T, IndexT>` and its non-owning peer
   `ShallowRaggedTensor<T, IndexT>`. Both expose ragged-aware
   iteration over `ragged_offset` ranges. `seq_lens` is not part
   of either type.
3. **`ITensor`** gains a polymorphic index-strategy hook so ragged
   iteration can supply a `RaggedCompositeIndex`.
4. **Plan layer** wraps the variant-pack pointer in a
   `ShallowRaggedTensor` per execute and passes it as
   `TensorBase<T>&`. CPU references and kernels that need
   `seq_lens` look it up from the variant pack themselves.
5. **Test harness** accepts a pre-supplied input bundle so the
   structurally-constrained `ragged_offset` aux carries
   deliberate values rather than meaningless random fill. The
   bundle stores tensors as `shared_ptr<ITensor>` to allow the
   same aux to be threaded into a ragged primary's constructor.

### 4.2 Frontend: `TensorAttributes` additions

Chainable setters/getters for the two new fields:

```cpp
auto set_ragged_offset(std::shared_ptr<Tensor_attributes> const& value)
    -> Tensor_attributes&;
auto get_ragged_offset() const -> std::shared_ptr<Tensor_attributes>;

auto set_alignment(int64_t alignmentInElements) -> Tensor_attributes&;
auto get_alignment() const -> int64_t;   // default 1
```

`set_alignment` declares a trailing-alignment requirement of the
physical buffer, in elements. The default `1` (no alignment)
matches AITER FMHA's needs. `alignment` affects only how much
storage is allocated — not how many elements the tensor reports or
iterates over (see [§4.5](#45-data-sdk-shared-elements)).

**Frontend validation** (in `validate()`):

- The `ragged_offset` aux exists in the graph (by UID), has rank
  1, and its first dim equals `B + 1` where `B` is the primary's
  first dim.
- `get_alignment() >= 1`.

`seq_lens` validation, where an op cares, lives on that op's
node-level `validate()`.

**Reported `dims()`.** The primary's `dims()[1]` remains the max
padded sequence length (`S_max`), matching overridable-shape
semantics without re-deriving `S_max` from `ragged_offset` at
runtime.

### 4.3 Flatbuffer schema additions

`tensor_attributes.fbs` gains two appended optional defaulted
fields:

```
ragged_offset_tensor_uid: long = null;
alignment:                long = 1;
```

Wire-compatible per RFC 0005. No `seq_lens_tensor_uid` is added
here: ops that consume `seq_lens` reference it through their own
per-op attribute tables (e.g. `SdpaAttributes` already carries
`seq_len_q_tensor_uid` / `seq_len_kv_tensor_uid`).

### 4.4 Backend wiring

Backend tensor descriptor mirrors the frontend additions: optional
`ragged_offset_tensor_uid` and `alignment` (default 1), exposed
through existing get/set-attribute paths under new enum values in
the hipDNN extension range. No new `hipdnnBackend*` entry points,
and the variant pack representation is unchanged: at execute time
the variant pack carries `UID → void*` for every tensor in the
graph, including ragged primaries, their `ragged_offset`, and any
`seq_lens` an op references.

### 4.5 Data SDK: shared elements

These apply to both `RaggedTensor<T, IndexT>` and
`ShallowRaggedTensor<T, IndexT>`:

1. **The `ragged_offset` aux is
   `std::shared_ptr<TensorBase<IndexT>>`**, with `IndexT`
   templated (defaulting to `int32_t`, with `int64_t` also
   permitted).
2. **`dims()[1]` is required to be `S_max`** (the max padded
   sequence length). This is a convention the user / graph must
   uphold so the reported geometry matches kernel and
   overridable-shape expectations; the SDK types do not derive or
   verify it.
3. **Polymorphic index-strategy hook on `ITensor`**: introduce
   `virtual std::unique_ptr<ITensorIndex> makeIndex(bool isEnd) const`.
   The `isEnd` parameter selects the begin vs end iterator
   position, mirroring the existing `LinearIndex` /
   `CompositeIndex` constructor pattern; the new types supply a
   `RaggedCompositeIndex` that walks each batch's full
   `[ragged_offset[b], ragged_offset[b+1])` range in turn.
4. **Iteration walks `ragged_offset` ranges, not `seq_lens`-bounded
   ranges.** Each batch's full per-batch range is iterated as
   part of that batch. Padding never leaks into the wrong batch's
   range, so indexing is always semantically correct (every
   visited element belongs to the right batch and to the tensor's
   owned memory); it only over-reports by visiting padded
   positions for ops that could in principle skip them. Ops that
   must skip padding query `seq_lens` directly from the variant
   pack.
5. **Constructor-time structural validation** (enforced by both
   types):
   - `raggedOffset != nullptr`.
   - `raggedOffset->elementCount() == paddedDims[0] + 1`
     (i.e. `B + 1`).
   - `raggedOffset` has rank 1.
6. **Two element-count concepts.** The types distinguish:
   - **`elementSpace()` = `physicalElementCount`**, the size of
     the allocated buffer including any trailing pad introduced by
     `alignment`. This is what the bundle / allocator asks for.
   - **`elementCount()` = `ragged_offset[B]`**, the number of
     addressable elements across all batches' per-batch ranges
     (which is what the iterator visits). The trailing
     alignment-pad region between `ragged_offset[B]` and
     `align_up(ragged_offset[B], alignment)` belongs to no batch
     and is never iterated. **`elementCount()` does not account
     for `seq_lens` limits** — it reports every position the
     iterator visits, including per-batch padding tails.
7. **`isPacked()` returns `false`** for both new types. This is
   consistent with the existing `Tensor<T>` convention
   `_packed = (elementCount == elementSpace)` — for these types,
   `elementCount()` (sum of per-batch ranges) generally differs
   from `elementSpace()` (`physicalElementCount` including any
   trailing alignment pad). However, a `RaggedTensor` with
   `alignment = 1` is "packed" in the colloquial sense (no
   internal gaps within each batch's range). The current
   `isPacked()` predicate conflates two distinct properties —
   "elementCount equals elementSpace" vs "buffer is `prod(dims)`
   elements with regular strides". A follow-up could split these
   apart, e.g. by introducing a separate predicate like
   `hasRegularDims()` (see [§7](#future-work)).

### 4.6 Data SDK: `RaggedTensor<T, IndexT>` (owning, ragged-aware)

A memory-owning ragged tensor whose physical element count is
independent of `prod(dims)` and which holds a `shared_ptr` to its
`ragged_offset` aux. Used as the runtime type for ragged primaries
in the bundle, as the type for ragged graph intermediates, and as
the user-facing type for samples.

```cpp
template <typename T,
          typename IndexT      = int32_t,
          typename HostAlloc   = HostAllocator<T>,
          typename DeviceAlloc = DeviceAllocator<T>>
class RaggedTensor : public TensorBase<T>
{
public:
    RaggedTensor(std::vector<int64_t>                paddedDims,
                 std::vector<int64_t>                strides,
                 size_t                              physicalElementCount,
                 std::shared_ptr<TensorBase<IndexT>> raggedOffset);

    // ITensor / TensorBase<T> overrides:
    //   dims()         -> paddedDims                    (dims()[1] == S_max)
    //   strides()      -> strides
    //   elementSpace() -> physicalElementCount           (allocation size)
    //   elementCount() -> ragged_offset[B]               (iterated elements)
    //   isPacked()     -> false
    //   begin/end      -> RaggedCompositeIndex via makeIndex()
    //                      (walks each batch's ragged_offset range)
    //
    // Direct addressing (rawHostData / rawDeviceData) is supported.

    const TensorBase<IndexT>* raggedOffset() const;

private:
    MigratableMemory<T, HostAlloc, DeviceAlloc> _memory;
    std::vector<int64_t>                        _paddedDims;
    std::vector<int64_t>                        _strides;
    size_t                                      _physicalElementCount;
    std::shared_ptr<TensorBase<IndexT>>         _raggedOffset;   // non-null
};
```

Structural validation at construction is shared with
`ShallowRaggedTensor` and listed in
[§4.5](#45-data-sdk-shared-elements).

Pinned host memory is available via
`RaggedTensor<T, IndexT, PinnedHostAllocator<T>, …>` — same
pattern `PinnedTensor<T>` uses for `Tensor<T>`.

**Immutability.** The `_raggedOffset` `shared_ptr` member is fixed
at construction and never reseated; the type exposes no setter for
it. The primary's T-typed buffer remains mutable as usual via
`rawHostData` / `rawDeviceData` / `MigratableMemory`.

**Ragged-aware multi-dim addressing.** `RaggedTensor<T, IndexT>`
overrides `getHostValue` / `setHostValue` from `TensorBase<T>` so
that a multi-dim index `{b, sq, …}` translates to a physical
offset using `ragged_offset[b]` as the per-batch base — i.e.
`physical_offset = ragged_offset[b] + sq * stride_1 + …`. (The
inherited implementation uses only the padded strides, which
would index into `b * stride_0 + …` regardless of where batch
`b`'s range actually starts in the physical buffer.) Callers may
index into batch `b` with `sq` ranging up to that batch's per-batch
extent (`ragged_offset[b+1] - ragged_offset[b]`); indices outside
that range are out-of-bounds for that batch and behavior is
unspecified.

### 4.7 Data SDK: `ShallowRaggedTensor<T, IndexT>` (non-owning peer)

A non-owning peer to `RaggedTensor<T, IndexT>`, used by the plan
layer at execute time when only a `void*` from the variant pack
is available.

```cpp
template <typename T, typename IndexT = int32_t>
class ShallowRaggedTensor : public TensorBase<T>
{
public:
    ShallowRaggedTensor(
        void*                               data,
        std::vector<int64_t>                paddedDims,
        std::vector<int64_t>                strides,
        size_t                              physicalElementCount,
        std::shared_ptr<TensorBase<IndexT>> raggedOffset);

    // Same overrides as RaggedTensor:
    //   dims()/strides() as provided
    //   elementSpace() -> physicalElementCount  (allocation size)
    //   elementCount() -> ragged_offset[B]      (iterated elements)
    //   isPacked()     -> false
    //   begin/end      -> RaggedCompositeIndex via makeIndex()
    //
    // rawHostData() / rawDeviceData() return the borrowed pointer.

    const TensorBase<IndexT>* raggedOffset() const;
};
```

`ShallowRaggedTensor` performs the same constructor-time
structural validation listed in
[§4.5](#45-data-sdk-shared-elements). It shares its
`RaggedCompositeIndex` implementation with `RaggedTensor`; only
memory ownership differs.

Unlike `RaggedTensor`, `ShallowRaggedTensor` does not carry an
allocator template parameter — pinned-vs-pageable is determined
by the caller-supplied buffer being wrapped, not by the wrapper.

### 4.8 Tiered API: which type each role uses

| Role | Type | Iteration |
|---|---|---|
| Owning storage for a ragged primary (bundle / executor intermediate) | `RaggedTensor<T, IndexT>` | Per-batch over `ragged_offset` ranges |
| Ragged I/O at the CPU reference plan layer | `ShallowRaggedTensor<T, IndexT>` over variant-pack `void*` | Per-batch over `ragged_offset` ranges |
| Ragged tensor a user iterates directly (samples) | `RaggedTensor<T, IndexT>` | Per-batch over `ragged_offset` ranges |
| Aux tensor (`ragged_offset`) | Plain `Tensor<IndexT>` via `shared_ptr` | Dense |
| `seq_lens` (op input, not part of the ragged abstraction) | Plain `Tensor<IndexT>` | Dense |
| Non-ragged tensors | Plain `Tensor<T>` | Dense |

### 4.9 Wiring sources of truth

- **Graph (`TensorAttributes`)** carries `ragged_offset` (pointer
  to the aux tensor's attributes) and `alignment`.
- **Node-level op attributes** (e.g. `SdpaAttributes`) carry
  `seq_lens` UIDs via existing accessors; these are referenced
  like any other tensor input — the SDK ragged types know
  nothing about them.
- **Runtime tensorMap (`UID → ITensor`)** carries:
  `RaggedTensor<T, IndexT>` per ragged primary, ordinary
  `Tensor<IndexT>` per `ragged_offset` aux and per `seq_lens`,
  ordinary `Tensor<T>` per non-ragged tensor.
- **Plan layer (executor)**: when executing an op whose graph
  declares a given input/output as ragged, the plan wraps the
  variant-pack pointer in a `ShallowRaggedTensor` whose
  `ragged_offset` ref is obtained from the variant pack by the
  UID stored in `TensorAttributes::ragged_offset_tensor_uid()`.
  `seq_lens`, if the op consumes it, is fetched as a separate
  `TensorBase<IndexT>&` from the variant pack the same way any
  other input would be.

**Ragged intermediates and aux connectivity.** Every
`ragged_offset` aux — whether referenced by a graph input,
output, or virtual intermediate — must be present in the
pre-supplied input bundle (see
[§4.11.1](#4111-pre-supplied-input-bundle)). The graph layer is
responsible for declaring this connectivity by giving each
ragged tensor's `TensorAttributes` an explicit
`ragged_offset_tensor_uid`. Ragged intermediates therefore reuse
the same `ragged_offset` aux as some other tensor in the graph;
the executor never synthesizes a new aux for an intermediate. A
hypothetical op that *computes* a new `ragged_offset` is out of
scope for this RFC.

### 4.10 Plan-layer construction and CPU reference impact

The plan's `params` struct caches each ragged tensor's UID and
that of its `ragged_offset` aux UID. At execute time the plan
resolves both via the variant pack:

```cpp
// Resolve ragged_offset aux from the variant pack first — its
// host values are needed to size the underlying view.
auto qRaggedOffset = std::make_shared<ShallowTensor<IndexT>>(
    variantPack.at(_params.qTensor.raggedOffsetUid),
    /* aux dims/strides: [B + 1], packed */);

// computePhysicalElementCount performs a host-side read of
// qRaggedOffset to obtain ragged_offset[B], then applies the
// trailing-alignment requirement.
const auto qPhysicalElementCount =
    computePhysicalElementCount(*qRaggedOffset,
                                _params.qTensor.alignment);

auto qView = std::make_shared<ShallowRaggedTensor<QType, IndexT>>(
    variantPack.at(_params.qTensor.uid),
    _params.qTensor.dims,
    _params.qTensor.strides,
    qPhysicalElementCount,
    qRaggedOffset);

// If the op consumes seq_lens, fetch it as an ordinary input:
auto seqLenQ = _params.seqLenQTensor.has_value()
    ? std::make_shared<ShallowTensor<IndexT>>(/* ... */)
    : nullptr;

// CPU reference signature changes only by accepting seq_lens as
// an ordinary extra parameter (where it isn't already), since
// it's no longer attached to the ragged primary:
cpuFpReferenceSdpa.forward(*qView, /* ..., */
                           seqLenQ ? &*seqLenQ : nullptr,
                           /* ... */);
```

Computing `physicalElementCount` per execute (rather than caching
it at plan-build time) avoids pinning `ragged_offset` values to
the lifetime of the compiled plan: each execute is free to use
different `ragged_offset` contents.

**CPU reference body.** The ragged primary's iterator walks each
batch's full per-batch range. When the reference also takes a
`seq_lens`, the body queries `seq_lens[b]` directly to decide
whether to skip indices `>= seq_lens[b]` within batch `b` — the
same pattern as any other op input that bounds the work shape.
The reference does **not** need any new `validSeqLen(b)`-style API
on the ragged primary itself.

### 4.11 Test-harness integration

#### 4.11.1 Pre-supplied input bundle

`ragged_offset` cannot be meaningfully randomized: its values
must be structurally consistent (monotonic,
`ragged_offset[0] == 0`, last element equal to the sized physical
buffer). The harness therefore accepts a pre-supplied input
bundle at setup:

```cpp
using PreSuppliedInputs =
    std::unordered_map<int64_t, std::shared_ptr<ITensor>>;

class IntegrationGraphVerificationHarness {
public:
    IntegrationGraphVerificationHarness(/* graph, etc. */,
                                        PreSuppliedInputs preSuppliedInputs);
    // ...
};
```

The value type is `shared_ptr<ITensor>` (not `unique_ptr`) so the
same aux entry can be threaded into a ragged primary's
constructor (see [§4.11.2](#4112-bundle-allocation)). Sharing
also lets the same `ITensor` serve both execution paths via its
built-in host/device `MigratableMemory`.

The harness retains ownership of the pre-supplied bundle and the
per-path `GraphTensorBundle`s continue to own the
non-pre-supplied tensors. The harness's UID lookup is a
pre-supplied-first fallthrough: if a UID is present in the
pre-supplied bundle, both `GraphTensorBundle`s see the same
`ITensor` via shared ownership, and randomization is skipped for
that UID.

**Strict requirement check.** Before allocation, the harness
verifies that every `ragged_offset` UID referenced by a
`TensorAttributes` is present in the pre-supplied bundle.
Missing `ragged_offset` UIDs fail setup with a clear error
rather than silently random-initializing a structurally invalid
aux.

A "random-then-fix-up" alternative — random-fill `ragged_offset`
and then sort / clamp the values into a structurally valid layout
— was considered and rejected. It loses test determinism in a
non-obvious way (small seed-handling or fixup-heuristic changes
silently shift every test's per-batch lengths), encodes implicit
policy choices ("how much padding per batch?", "is
`seq_lens[b] == 0` allowed?") invisible to the test author, and
saves no real work: the author still has to reason about which
distribution they want, so an explicit pre-supplied value is
strictly clearer at the same cognitive cost.

`seq_lens` is **not** strict-checked: it's an ordinary input
tensor from the harness's point of view. Test authors who want
deterministic `seq_lens` values for ops that consume them can put
them in the pre-supplied bundle as ordinary entries; the harness
treats them like any other override.

**Variant-pack construction.** UIDs resolved from the
pre-supplied bundle contribute the same `void*` to both bundles'
variant packs (via the same `ITensor`), guaranteeing byte-equal
input values across the GPU and CPU paths.

#### 4.11.2 Bundle allocation

For each `TensorAttributes`:

- If the UID is in the pre-supplied bundle: skip (already
  resolved).
- Else if `attr.has_ragged_offset() == false`: allocate
  `Tensor<T>` sized by `prod(dims)` as today.
- Else: look up the `ragged_offset` UID in the pre-supplied
  bundle to obtain its `shared_ptr<TensorBase<IndexT>>` — the
  strict check in
  [§4.11.1](#4111-pre-supplied-input-bundle) guarantees its
  presence — read its host values, compute the physical element
  count, and allocate
  `make_shared<RaggedTensor<T>>(dims, strides,
  physicalElementCount, raggedOffsetSharedPtr)`.

A single-pass walk suffices because every `ragged_offset` aux a
primary could reference is, by the strict check, already held by
the pre-supplied bundle before allocation begins. No auxiliary
pre-pass over `TensorAttributes` is needed.

The physical element count is:

```
physicalElementCount = align_up(ragged_offset[B], attr.get_alignment())
```

For `alignment = 1` this reduces to `ragged_offset[B]` (packed
valid size). Larger alignment values round the total buffer size
up to satisfy a trailing-alignment requirement. The plan layer
re-evaluates this expression at execute time against the current
variant-pack `ragged_offset` (per
[§4.10](#410-plan-layer-construction-and-cpu-reference-impact)).
Note that `elementCount()` on the resulting `RaggedTensor` is
still `ragged_offset[B]` — alignment expands the allocation, not
the element space (see
[§4.5](#45-data-sdk-shared-elements) item 6).

#### 4.11.3 Input randomization at bundle init

UIDs in the pre-supplied bundle are skipped. For non-pre-supplied
ragged primaries, `bundle.randomizeTensor(uid)` walks the
`RaggedTensor<T>` per-batch over its `ragged_offset` ranges,
filling every position in every batch's range (including any
trailing per-batch padding tails). This is harmless: neither GPU
kernels nor CPU references read padding positions, and identical
seeds keep padding values byte-equal between the two paths.

#### 4.11.4 Output validation in `verifyGraph`

The CPU reference writes only valid sequence positions of an
output (those `< seq_lens[b]`); the GPU kernel makes no
contractual guarantees about output padding. A naive full-buffer
compare therefore fails on padding for reasons unrelated to
algorithmic correctness.

**Recommended approach: sentinel-skip.** At harness init, fill
each ragged output's physical buffer (both bundles) with a
sentinel the CPU reference cannot legitimately produce (NaN for
floats; a magic number for ints) instead of randomizing. After
execution the validator iterates the full physical buffer;
positions where the CPU side is bit-equal to the sentinel are
skipped (the reference didn't write there); other positions use
the existing tolerance check. This needs one validator overload
(`allCloseSkipSentinel(cpu, gpu, sentinel, tolerance)`) and a
small init-time branch.

Sentinel-skip is the only validation mechanism in this design.
Because `seq_lens` is not attached to the ragged primary, there
is no SDK-level "wrap with `seq_lens` for ragged-aware compare"
fallback to fall back to.

---

## Known limitations

1. **`GraphTensorBundle` stores entries as `shared_ptr<ITensor>`,
   not `unique_ptr<ITensor>`.** Required so a `ragged_offset` aux
   can be threaded into a `RaggedTensor`'s ctor as a shared
   reference.
2. **Iteration visits every position in each batch's
   `[ragged_offset[b], ragged_offset[b+1])` range.** The
   iterator has no reference to `seq_lens` and does not skip the
   per-batch padding tails. Ops that need to skip padding query
   `seq_lens` directly from the variant pack.
3. **Sentinel-skip validation requires per-CPU-reference
   guarantees.** The sentinel must be unproducible by the
   reference under test. NaN is safe for floats by default, but
   edge cases (e.g. SDPA with `seq_lens_q[b] == 0` producing NaN
   from a fully masked row) may require a non-NaN magic-number
   sentinel. Low-precision floats (bf16, fp16, fp8) have a
   smaller sentinel space and harder unproducibility guarantees.
4. **Test authors must construct `ragged_offset` tensors by
   hand.** Every integration test of a ragged-consuming op must
   build a deliberate `ragged_offset` (and, when the op consumes
   `seq_lens`, supply `seq_lens` too) and put it in the
   pre-supplied input bundle. No harness-level helper for "a
   plausible random ragged layout" is in this RFC.
5. **`isPacked()` is overloaded.** It returns `false` for these
   types because `elementCount() != elementSpace()` when
   `alignment > 1`, but a packed-with-`alignment=1` `RaggedTensor`
   is colloquially "packed". See
   [§4.5](#45-data-sdk-shared-elements) item 7 and
   [§7](#future-work) for a possible `hasRegularDims()` split.

---

## Alternatives considered

### 6.1 `IrregularTensor<T>` + `RaggedView<T, IndexT>` split

An alternative shape introduces two SDK types instead of one:

- **`IrregularTensor<T>`** — owning storage with
  `physicalElementCount ≠ prod(dims)`, no aux refs.
- **`RaggedView<T, IndexT>`** — non-owning ragged-aware wrapper
  that takes the underlying storage by `shared_ptr` plus
  `ragged_offset` (and optionally `seq_lens`) at construction,
  providing the ragged iteration.

The plan layer assembles a view at execute time from the
variant-pack pointer (wrapped in a `ShallowTensor<T>`-style
underlying) and the resolved aux refs. The CPU reference receives
a `TensorBase<T>&` referring to the view.

**Why this might be appealing.** The split keeps "I have a buffer
that is not `prod(dims)` elements" (a storage concern) separate
from "I want ragged-aware iteration with `seq_lens` bounding"
(a presentation concern). In particular, if `seq_lens` were
attached to the view, the bundle could store entries as
`unique_ptr<ITensor>` without ever needing to thread aux refs
into a primary's constructor — the view would attach the aux
indirectly at the plan layer.

**Why this RFC does not take it.** Once `seq_lens` is excluded
from the SDK abstraction (per
[§6.2](#62-keeping-seq_lens-in-the-ragged-tensor-sdk-abstraction)),
the only aux the SDK still cares about is `ragged_offset`, which
must be available at primary-allocation time to size the buffer.
The bundle therefore still has to thread `ragged_offset` into a
primary's ctor — exactly the `shared_ptr` shift that the split
design was trying to avoid. The split's main advantage
disappears, and what remains is pure cost:

- **Two new SDK types instead of one** (`IrregularTensor<T>` +
  `RaggedView<T, IndexT>`, plus a `ShallowTensor<T>` underlying
  pattern at the plan layer that the single-type design folds
  into `ShallowRaggedTensor<T>` directly).
- **A storage type that derives from `TensorBase<T>` but whose
  `getHostValue` / `setHostValue` have no meaningful semantics on
  the ragged primary's physical buffer** — those methods would
  need to either throw or address into "raw physical positions",
  motivating a further hierarchy refactor (introducing a
  `TypedTensor<T>` between `ITensor` and `TensorBase<T>`) just to
  clean up.
- **Two distinct `elementCount()` contracts** (one for the
  storage type, one for the view) that callers have to keep
  straight.

The single-type design (`RaggedTensor<T, IndexT>` +
`ShallowRaggedTensor<T, IndexT>`) avoids all three of these
costs while incurring the same `shared_ptr`-typed bundle storage
either way.

### 6.2 Keeping `seq_lens` in the ragged-tensor SDK abstraction

A variant of this design has the ragged primary's type hold a
`shared_ptr` to `seq_lens` alongside `ragged_offset`, provide a
`validSeqLen(b)` API, and iterate only the valid prefix of each
batch's range.

**The case for keeping it:**

- Ragged iteration is "fully ragged-aware" — callers using
  `begin()/end()` on the primary automatically skip per-batch
  padding without having to look up `seq_lens` separately.
- CPU reference bodies don't need to take `seq_lens` as a
  separate parameter; they call `primary.validSeqLen(b)` and an
  early-out-on-`sq >= validSeqLen(b)` check is all that's needed.

**The case against (taken by this design):**

- **`seq_lens` is never used during input fill or output
  validation.** Randomization walks the physical buffer
  regardless (padding values are harmless because nobody reads
  them). Sentinel-skip validation doesn't consult `seq_lens`
  either. The only consumer is the CPU reference body — a single,
  narrow site.
- **CPU references already look up arbitrary inputs from the
  variant pack.** Querying a `seq_lens` tensor that's just
  another input is no more awkward than querying any other input.
  There is no concrete ergonomic gain from a `validSeqLen(b)` API
  over `seqLens.getHostValue({b})`.
- **Indexing is always semantically correct without `seq_lens`.**
  Walking each batch's full `ragged_offset` range only ever
  visits positions that belong to the right batch and to the
  tensor's owned memory. The cost of carrying `seq_lens` in the
  SDK is paid in every layer (frontend bundle setup, view ctor
  signatures, pre-supplied bundle strict checks,
  alternatives-section analysis) to buy what amounts to a
  per-batch loop bound in one CPU reference function body.
- **`seq_lens` does not bind 1:1 to a primary.** In SDPA a single
  `seq_lens_q` is shared across the Q primary and a single
  `seq_lens_kv` across K and V; putting `seq_lens` on the
  primary's type would either store the same `shared_ptr`
  redundantly on multiple primaries or require a per-op binding
  rule that the node-level location expresses naturally.
- **cuDNN frontend-API compatibility.** Existing cuDNN frontend
  consumers configure `seq_lens` on op-level attributes; mirroring
  it onto a primary's runtime type would create two ways to
  express the same thing.
- **Significant complexity savings.** Dropping `seq_lens` from
  the SDK removes the storage-vs-view split, the
  post-construction-attach-seq_lens problem, the `validSeqLen`
  virtual on `TensorBase`, a seq_lens-aware output-validation
  fallback, and the strict-requirement check on `seq_lens` UIDs
  in the pre-supplied bundle.

The deciding factor is the cost/benefit ratio: a tiny amount of
work shifted into per-op CPU references in exchange for a much
simpler SDK and harness.

---

## Future work

### 7.1 Runtime-side `TensorBundle` helper

A `UID → owning_runtime_tensor` helper on the user side, with a
factory that for non-ragged attrs builds a `Tensor<T>` and for
ragged attrs handles the pattern of constructing `ragged_offset`
from user-supplied values and then constructing a
`RaggedTensor<T>` sized from `(ragged_offset, alignment)`.
Collapses the user's API to one call per `TensorAttributes`.

A companion helper on the harness side could generate plausible
random ragged layouts and emit them as pre-supplied input bundles
(reducing boilerplate for conformance tests where the specific
ragged layout doesn't matter as long as it is well-formed).

### 7.2 Unify `RaggedTensor` and `ShallowRaggedTensor` as one templated class

The two types share their `RaggedCompositeIndex` implementation,
their constructor-time structural validation, and (after the
`seq_lens` decoupling) their entire surface. Only memory
ownership differs. They could plausibly be expressed as one class
template parameterized by the memory carrier (owning
`MigratableMemory<T, …>` vs borrowed `void*`). The split kept here
is the conservative choice that mirrors the existing
`Tensor<T>` / `ShallowTensor<T>` split; unifying them is a
mechanical refactor left as follow-up.

### 7.3 Split `isPacked()` into orthogonal predicates

The existing `isPacked()` predicate conflates "elementCount
equals elementSpace" with "buffer is `prod(dims)` elements with
regular strides" (see
[§4.5](#45-data-sdk-shared-elements) item 7). A follow-up could
introduce a separate `hasRegularDims()` (or similarly-named)
predicate to let callers distinguish "I can treat this as a flat
`prod(dims)`-sized dense buffer" from "iteration visits every
allocated element". `RaggedTensor` / `ShallowRaggedTensor` would
report `hasRegularDims() == false` always, and `isPacked()` would
mean strictly `elementCount() == elementSpace()`.

### 7.4 Sentinel selection per CPU reference

Document the sentinel choice and the "the reference cannot
produce this value" guarantee in each CPU reference header. NaN
is the default safe choice for floats, but watch for edge cases —
e.g. SDPA forward with `seq_lens_q[b] == 0` can produce NaN
naturally (`log(0)` from a fully masked row), in which case a
non-NaN magic-number sentinel is required. For low-precision
floating-point types (bf16, fp16, fp8) the sentinel space is
smaller and "unproducible by reference" is harder to guarantee. A
future RFC could codify the sentinel-per-dtype-per-op table.
