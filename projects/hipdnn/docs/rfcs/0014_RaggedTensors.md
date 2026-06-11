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
   - 4.6 [Data SDK: `RaggedTensor<T>` (owning, ragged-aware)](#46-data-sdk-raggedtensort-owning-ragged-aware)
     - 4.6.1 [User-side construction pattern](#461-user-side-construction-pattern)
   - 4.7 [Data SDK: `ShallowRaggedTensor<T>` (non-owning peer)](#47-data-sdk-shallowraggedtensort-non-owning-peer)
   - 4.8 [Tiered API: which type each role uses](#48-tiered-api-which-type-each-role-uses)
   - 4.9 [Wiring sources of truth](#49-wiring-sources-of-truth)
   - 4.10 [Plan-layer construction and CPU reference impact](#410-plan-layer-construction-and-cpu-reference-impact)
     - 4.10.1 [Reuse at the executor virtual-tensor pass](#4101-reuse-at-the-executor-virtual-tensor-pass)
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
   `alignment` is a needed addition in its own right (and is
   cuDNN-compatible); it is bundled into the same schema change so
   plugins update once to support both it and ragged tensors,
   though no logic in this RFC consumes it.
3. **Backend** propagates the new fields through existing
   get/set-attribute paths; no new C-API entry points; variant pack
   unchanged.
4. **Data SDK** adds a single owning type `RaggedTensor<T>`
   (memory + `ragged_offset` aux ref) and its non-owning peer
   `ShallowRaggedTensor<T>`. Both expose ragged-aware
   iteration over `ragged_offset` ranges. The `ragged_offset`
   aux is held type-erased as `std::shared_ptr<ITensor>` and read
   through a runtime element-size branch (int32 or int64) that
   widens to `int64_t` at the read site, so neither ragged-tensor
   type carries an `IndexT` template parameter.
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

A tensor with a physical memory layout of `[B, X, …]` (referred to here as the primary tensor) where `X` (typically the sequence
dimension) varies per batch. Batch `b` occupies the contiguous
range `[ragged_offset[b], ragged_offset[b+1])` of the physical
buffer.

Some ops additionally need to distinguish *valid* sequence
positions within each batch's range from *trailing padding*
positions kept for alignment. That information is carried by a
separate `seq_lens[B]` tensor referenced through the op's
node-level attributes (e.g. `SdpaAttributes::set_seq_len_q`). Each
entry in `seq_lens` specifies the number of valid elements in that batch. `seq_lens` is independent
of the ragged-tensor abstraction: it can be used in isolation, or with the `ragged_offset`
(and the `ragged_offset` similarly functions with or without it).
Ops that care about per-batch valid lengths read `seq_lens` from the variant pack directly.

Both `ragged_offset` and `seq_lens` aux tensors may be referenced
by more than one primary in the same graph — SDPA, for example,
typically shares a single `seq_lens_kv` across the K and V
primaries and may share a `ragged_offset_kv` between them as well.
The design supports this naturally: each primary's
`TensorAttributes` references the aux by UID; the runtime holds
one `ITensor` per UID; multiple `RaggedTensor<T>`s share the same
`ragged_offset` aux via `shared_ptr<ITensor>`.

This RFC does not introduce anything to address the way `seq_lens` are
handled, but discussion of their role in the process is included for context.

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
physical buffer is always exactly `ragged_offset[B]` elements:

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
  (`getHostValue` / `setHostValue`). These are **non-virtual**
  conveniences that funnel through `ITensor::getIndex`, so the
  override point for custom addressing is `getIndex`, not these
  methods.
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
   Declarative only; no new C-API entry points (see
   [§4.2](#42-frontend-tensorattributes-additions) for
   `alignment`).
2. **Data SDK** introduces a single new owning type
   `RaggedTensor<T>` and its non-owning peer
   `ShallowRaggedTensor<T>`. Both expose ragged-aware
   iteration over `ragged_offset` ranges. The `ragged_offset`
   aux is held type-erased as `std::shared_ptr<ITensor>` (no
   `IndexT` template parameter — see
   [§4.5](#45-data-sdk-shared-elements) item 1). `seq_lens` is
   not part of either type.
3. **`ITensor`** gains two complementary polymorphic hooks: a
   `getIndexImpl` override point behind `getIndex` so ragged
   *addressing* bases each batch at `ragged_offset[b]`, and an
   index-strategy hook so ragged *iteration* can supply a
   `RaggedCompositeIndex`.
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

auto set_alignment(int64_t alignmentInBytes) -> Tensor_attributes&;
auto get_alignment() const -> int64_t;   // default 16
```

`set_alignment` declares the required byte alignment of the
*pointer* to the tensor's physical buffer (default `16`). It
is introduced here only because doing so alongside the
ragged-tensor schema changes lets plugins add support for both
at once (see [§4.3](#43-flatbuffer-schema-additions)). Nothing in
this RFC consumes `alignment`; it is carried through the frontend,
flatbuffer, and backend but is otherwise unused by the
ragged-tensor SDK types.

**Frontend validation** (in `validate()`):

- The `ragged_offset` aux exists in the graph (by UID), has rank
  4, and its first dim equals `B + 1` where `B` is the primary's
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
alignment:                long = 16;
```

Wire-compatible per RFC 0005. Only `ragged_offset_tensor_uid` is
functionally required by *this RFC's* logic. `alignment` is a
needed addition in its own right (kept consistent with cuDNN, see
[§4.2](#42-frontend-tensorattributes-additions)); it is appended
in the same change so that plugins make a single update to support
both alignment and ragged tensors, rather than going through two
separate rounds of schema evolution and version bumps. No code in
this RFC reads it.

No `seq_lens_tensor_uid` is added
here: ops that consume `seq_lens` reference it through their own
per-op attribute tables (e.g. `SdpaAttributes` already carries
`seq_len_q_tensor_uid` / `seq_len_kv_tensor_uid`).

To ensure that only plugins that have appropriately handled
ragged tensors process graphs with them present, this also
requires an `is_ragged_tensor_enabled` boolean to be added to
the graph, and `computeMinimumPluginApiVersion` to be updated
to map that boolean to the appropriate version.

This also requires an update to the TensorAttributes and Graph
json serialization/deserialization to output these values, and
set appropriate defaults on older graphs where they aren't
present.


### 4.4 Backend wiring

Backend tensor descriptor mirrors the frontend additions: optional
`ragged_offset_tensor_uid` and `alignment` (default 16), exposed
through existing get/set-attribute paths under new enum values in
the hipDNN extension range. As on the frontend, `alignment` is
carried through but unused here (see
[§4.2](#42-frontend-tensorattributes-additions)). No new `hipdnnBackend*` entry points,
and the variant pack representation is unchanged: at execute time
the variant pack carries `UID → void*` for every tensor in the
graph, including ragged primaries, their `ragged_offset`, and any
`seq_lens` an op references.

### 4.5 Data SDK: shared elements

These apply to both `RaggedTensor<T>` and
`ShallowRaggedTensor<T>`:

1. **The `ragged_offset` aux is held type-erased as
   `std::shared_ptr<ITensor>`** — neither ragged-tensor type
   carries an `IndexT` template parameter. Both `int32_t` and
   `int64_t` element types are permitted; the actual element type
   is discovered at runtime from the aux's `elementSize()` (and
   widened to `int64_t` at the read site). A typical inline read
   helper looks like:

   ```cpp
   int64_t readOffset(size_t b) const {
       void* p = _raggedOffset->hostDataOffsetFromIndex(
           static_cast<int64_t>(b));
       switch (_raggedOffset->elementSize()) {
           case 4:  return *static_cast<const int32_t*>(p);
           case 8:  return *static_cast<const int64_t*>(p);
           default: throw std::runtime_error(
               "ragged_offset element size must be 4 or 8 bytes");
       }
   }
   ```

   The widen-to-`int64_t` cost is negligible relative to the per-
   element work the CPU reference does, and the structural
   invariants in item 5 (rank 4, packed, length `B + 1`, and the
   element-size check) bound the surface where type-erasure could
   go wrong to this one helper. See
   [§6.3](#63-templating-the-ragged-tensor-types-on-indext) for
   why this is preferred over a templated `IndexT` parameter.
2. **`dims()[1]` is required to be `S_max`** (the max padded
   sequence length). This is a convention the user / graph must
   uphold so the reported geometry matches kernel and
   overridable-shape expectations; the SDK types do not derive or
   verify it.
3. **Two complementary indexing hooks on `ITensor`** — one for
   addressing, one for traversal:

   - **Addressing (`getIndexImpl`).** `getIndex(const
     std::vector<int64_t>&)` is refactored to forward to a new
     `protected virtual int64_t getIndexImpl(const
     std::vector<int64_t>&) const` holding the default offset
     computation (`inner_product` of indices and strides); the
     argument-count check stays in the non-virtual forwarder. Both
     ragged types override `getIndexImpl` to base each batch's
     offset at `readOffset(b)` (see
     [§4.6](#46-data-sdk-raggedtensort-owning-ragged-aware)).
     Every addressing path funnels through `getIndex`
     (`getHostValue` / `setHostValue` / `operator()`,
     `CompositeIndex::getValue`, `TensorView`), so this single
     override makes direct addressing ragged-aware everywhere — at
     the cost of making `getIndex` a virtual call for all
     non-packed tensors (see [§5](#known-limitations) item 7).
   - **Traversal (`makeIndex` / `RaggedCompositeIndex`).**
     Introduce `virtual IndexType makeIndex(bool isEnd) const`
     (`isEnd` selects the begin vs end position, mirroring the
     existing `LinearIndex` / `CompositeIndex` pattern) and add a
     `RaggedCompositeIndex` to the `IndexType` variant that walks
     each batch's full `[ragged_offset[b], ragged_offset[b+1])`
     range in turn. This is **required in addition to**
     `getIndexImpl`: traversal bounds derive from `dims()` /
     `elementCount()`, so a plain `CompositeIndex` would visit
     `prod(paddedDims)` positions rather than the
     `ragged_offset[B]` that `elementCount()` reports (item 6).
     `RaggedCompositeIndex` emits `{b, within-batch…}` indices and
     delegates the per-element offset back to `getIndex`, keeping
     the offset math in one place.
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
   - `raggedOffset` has rank 4.
   - `raggedOffset->elementSize() == 4 || raggedOffset->elementSize() == 8`
     (int32 or int64 element type — checked once at construction
     so the type-erased `readOffset` helper from item 1 only ever
     encounters supported sizes).
6. **Element-count reporting.** `elementCount()` reports
   `ragged_offset[B]` — the number of addressable elements across
   all batches' per-batch ranges, which is what the iterator
   visits. `elementSpace()` reports `physicalElementCount`, the
   size of the allocated buffer. The buffer is sized to exactly
   `ragged_offset[B]` elements, so the two normally coincide.
   `alignment` does **not** enter either calculation: it
   constrains the buffer *pointer*, not the buffer *size* (see
   [§4.2](#42-frontend-tensorattributes-additions)).
   **`elementCount()` does not account for `seq_lens` limits** — it
   reports every position the iterator visits, including per-batch
   padding tails.
7. **`isPacked()` returns `false`** for both new types. Note this
   *diverges* from the literal `Tensor<T>` convention
   `_packed = (elementCount == elementSpace)`: for a ragged tensor
   those two are equal (both `ragged_offset[B]`, per item 6), yet
   the buffer is neither `prod(dims)` elements long nor regularly
   strided, so treating it as a flat dense buffer would be wrong.
   The current `isPacked()` predicate conflates two distinct
   properties — "elementCount equals elementSpace" vs "buffer is
   `prod(dims)` elements with regular strides" — and ragged
   tensors are the case that pulls them apart. A follow-up could
   split these, e.g. by introducing a separate `hasRegularDims()`
   predicate (see [§7](#future-work)).

### 4.6 Data SDK: `RaggedTensor<T>` (owning, ragged-aware)

A memory-owning ragged tensor whose physical element count is
independent of `prod(dims)` and which holds a `shared_ptr` to its
`ragged_offset` aux. Used as the runtime type for ragged primaries
in the bundle, as the type for ragged graph intermediates, and as
the user-facing type for samples.

```cpp
template <typename T,
          typename HostAlloc   = HostAllocator<T>,
          typename DeviceAlloc = DeviceAllocator<T>>
class RaggedTensor : public TensorBase<T>
{
public:
    // physicalElementCount is optional: when omitted it is inferred
    // as ragged_offset[B] by reading the aux at construction. Pass
    // it explicitly to skip that host read of the aux (and the
    // device->host synchronization it may incur when the aux lives
    // in device memory).
    RaggedTensor(std::vector<int64_t>     paddedDims,
                 std::vector<int64_t>     strides,
                 std::shared_ptr<ITensor> raggedOffset,
                 std::optional<size_t>    physicalElementCount = std::nullopt);

    // ITensor / TensorBase<T> overrides:
    //   dims()         -> paddedDims                    (dims()[1] == S_max)
    //   strides()      -> strides
    //   elementSpace() -> physicalElementCount           (allocation size == ragged_offset[B])
    //   elementCount() -> ragged_offset[B]               (iterated elements)
    //   isPacked()     -> false
    //   getIndexImpl() -> readOffset(b) + sq*stride_1 + ...  (ragged addressing)
    //   begin/end      -> RaggedCompositeIndex via makeIndex()
    //                      (walks each batch's ragged_offset range)
    //
    // Direct addressing (rawHostData / rawDeviceData) is supported.

    const ITensor* raggedOffset() const;

private:
    MigratableMemory<T, HostAlloc, DeviceAlloc> _memory;
    std::vector<int64_t>                        _paddedDims;
    std::vector<int64_t>                        _strides;
    size_t                                      _physicalElementCount;
    std::shared_ptr<ITensor>                    _raggedOffset;   // non-null

    // Type-erased read helper from §4.5 item 1; reads
    // ragged_offset[b] from _raggedOffset and widens to int64_t.
    int64_t readOffset(size_t b) const;
};
```

Structural validation at construction is shared with
`ShallowRaggedTensor` and listed in
[§4.5](#45-data-sdk-shared-elements). Crucially, the aux's
element type (int32 vs int64) is *not* a template parameter — it
is discovered at construction time and validated against the
permitted set (4 or 8 bytes), and individual reads dispatch
through `readOffset` (see [§4.5](#45-data-sdk-shared-elements)
item 1).

Pinned host memory is available via
`RaggedTensor<T, PinnedHostAllocator<T>, …>` — same pattern
`PinnedTensor<T>` uses for `Tensor<T>`.

**Immutability.** The `_raggedOffset` `shared_ptr` member is fixed
at construction and never reseated; the type exposes no setter for
it. The primary's T-typed buffer remains mutable as usual via
`rawHostData` / `rawDeviceData` / `MigratableMemory`.

**Ragged-aware multi-dim addressing.** `RaggedTensor<T>`
overrides `getIndexImpl` (the protected virtual from
[§4.5](#45-data-sdk-shared-elements) item 3) so that a multi-dim
index `{b, sq, …}` translates to a physical offset using
`readOffset(b)` (i.e. `ragged_offset[b]` widened to `int64_t`) as
the per-batch base:
`physical_offset = readOffset(b) + sq * stride_1 + …`. (The
default implementation uses only the padded strides, indexing
into `b * stride_0 + …` regardless of where batch `b`'s range
actually starts in the physical buffer.) Overriding `getIndexImpl`
rather than the non-virtual `getHostValue` / `setHostValue` is
what makes every addressing path ragged-aware at once (§4.5 item
3). Callers may index into batch `b` with `sq` ranging up to that
batch's per-batch extent (`readOffset(b+1) - readOffset(b)`);
indices outside that range are out-of-bounds for that batch and
behavior is unspecified.

#### 4.6.1 User-side construction pattern

Samples build ragged tensors by hand in two steps. There is no
`RaggedTensor::RaggedTensor(const TensorAttributes&)` convenience
constructor in this iteration — the aux runtime tensor is a
required ctor input and is not derivable from a single
`TensorAttributes`.

```cpp
// 1. Allocate the aux as an ordinary Tensor<IndexType> (either
//    int32_t or int64_t — the SDK accepts both per §4.5 item 1),
//    held by shared_ptr so it can be threaded into the ragged
//    primary's ctor and shared across multiple ragged primaries.
auto qRaggedOffset =
    std::make_shared<utilities::Tensor<int32_t>>(/*dims=*/{B + 1, 1, 1, 1});
qRaggedOffset->fillFromHost(myOffsetsHost);   // user-supplied values

// 2. Allocate the ragged primary. The buffer is sized to exactly
//    ragged_offset[B] elements; alignment plays no part (§4.2).
//    The aux is passed as shared_ptr<ITensor>; the ragged tensor
//    type-erases its element type at construction (§4.5 item 1).
//    Form (a): let the ctor infer the size by reading
//    ragged_offset[B] from the aux.
auto qTensor = std::make_shared<utilities::RaggedTensor<float>>(
    qAttr->get_dim(),
    qAttr->get_stride(),
    qRaggedOffset);     // implicit upcast Tensor<int32_t> -> ITensor

//    Form (b): pass physicalElementCount explicitly — the user
//    already knows ragged_offset[B] (they just wrote it) — to skip
//    the aux read and any device->host sync it would imply:
//
//    auto qTensor = std::make_shared<utilities::RaggedTensor<float>>(
//        qAttr->get_dim(), qAttr->get_stride(), qRaggedOffset,
//        static_cast<size_t>(myOffsetsHost.back()));

// 3. Wire into variantPack as today — one entry per primary, one
//    entry per aux. Nothing about variantPack assembly changes
//    relative to non-ragged tensors.
variantPack[qAttr      ->get_uid()] = qTensor      ->rawDeviceData();
variantPack[qRaggedAttr->get_uid()] = qRaggedOffset->rawDeviceData();
```

For non-ragged tensors in the same sample, the user continues to
construct an ordinary `Tensor<T>(attr->get_dim(), attr->get_stride())`
as today — no aux, no `physicalElementCount`, no `shared_ptr`
required.

### 4.7 Data SDK: `ShallowRaggedTensor<T>` (non-owning peer)

A non-owning peer to `RaggedTensor<T>`, used by the plan layer at
execute time when only a `void*` from the variant pack is
available.

```cpp
template <typename T>
class ShallowRaggedTensor : public TensorBase<T>
{
public:
    // As with RaggedTensor (§4.6), physicalElementCount is optional
    // and inferred as ragged_offset[B] when omitted.
    ShallowRaggedTensor(
        void*                    data,
        std::vector<int64_t>     paddedDims,
        std::vector<int64_t>     strides,
        std::shared_ptr<ITensor> raggedOffset,
        std::optional<size_t>    physicalElementCount = std::nullopt);

    // Same overrides as RaggedTensor:
    //   dims()/strides() as provided
    //   elementSpace() -> physicalElementCount  (allocation size)
    //   elementCount() -> ragged_offset[B]      (iterated elements)
    //   isPacked()     -> false
    //   begin/end      -> RaggedCompositeIndex via makeIndex()
    //
    // rawHostData() / rawDeviceData() return the borrowed pointer.

    const ITensor* raggedOffset() const;
};
```

`ShallowRaggedTensor` performs the same constructor-time
structural validation listed in
[§4.5](#45-data-sdk-shared-elements), including the int32/int64
element-size check on the aux. It shares its `getIndexImpl`
override, its `RaggedCompositeIndex` implementation, and its
type-erased `readOffset` helper with `RaggedTensor`; only memory
ownership differs.

Unlike `RaggedTensor`, `ShallowRaggedTensor` does not carry an
allocator template parameter — pinned-vs-pageable is determined
by the caller-supplied buffer being wrapped, not by the wrapper.

### 4.8 Tiered API: which type each role uses

| Role | Type | Iteration |
|---|---|---|
| Owning storage for a ragged primary (bundle / executor intermediate) | `RaggedTensor<T>` | Per-batch over `ragged_offset` ranges |
| Ragged I/O at the CPU reference plan layer | `ShallowRaggedTensor<T>` over variant-pack `void*` | Per-batch over `ragged_offset` ranges |
| Ragged tensor a user iterates directly (samples) | `RaggedTensor<T>` | Per-batch over `ragged_offset` ranges |
| Aux tensor (`ragged_offset`) | Plain `Tensor<int32_t>` or `Tensor<int64_t>`, held as `shared_ptr<ITensor>` | Dense |
| `seq_lens` (op input, not part of the ragged abstraction) | Plain `Tensor<int32_t>` or `Tensor<int64_t>` | Dense |
| Non-ragged tensors | Plain `Tensor<T>` | Dense |

### 4.9 Wiring sources of truth

- **Graph (`TensorAttributes`)** carries `ragged_offset` (pointer
  to the aux tensor's attributes) and `alignment`.
- **Node-level op attributes** (e.g. `SdpaAttributes`) carry
  `seq_lens` UIDs via existing accessors; these are referenced
  like any other tensor input — the SDK ragged types know
  nothing about them.
- **Runtime tensorMap (`UID → ITensor`)** carries:
  `RaggedTensor<T>` per ragged primary; ordinary
  `Tensor<int32_t>` or `Tensor<int64_t>` per `ragged_offset` aux
  and per `seq_lens` (the integer type is whatever the test /
  user supplied — the SDK does not constrain it beyond
  int32-or-int64); ordinary `Tensor<T>` per non-ragged tensor.
- **Plan layer (executor)**: when executing an op whose graph
  declares a given input/output as ragged, the plan wraps the
  variant-pack pointer in a `ShallowRaggedTensor<T>` whose
  `ragged_offset` ref is obtained from the variant pack by the
  UID stored in `TensorAttributes::ragged_offset_tensor_uid()`.
  The aux is passed as `shared_ptr<ITensor>` (its element type is
  discovered at construction by the type-erased helper from
  [§4.5](#45-data-sdk-shared-elements) item 1). `seq_lens`, if
  the op consumes it, is fetched as a separate input from the
  variant pack the same way any other input would be — typically
  as a `TensorBase<int32_t>&` or `TensorBase<int64_t>&` depending
  on what the op's per-node attribute declares.

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

The aux is wrapped via a small dispatched factory shared with the
executor's virtual-tensor pass (see
[§4.10.1](#4101-reuse-at-the-executor-virtual-tensor-pass)):

```cpp
// makeShallowITensor(dataType, ptr, dims, strides) -> shared_ptr<ITensor>
//
//   Runtime-dispatched factory. Returns a non-owning
//   ShallowTensor<T> wrapped as shared_ptr<ITensor>, with T
//   selected by `dataType`. This factory is the only site at
//   which the aux's element type appears statically; everywhere
//   else the aux flows as shared_ptr<ITensor> per §4.5 item 1.
//   For ragged_offset and seq_lens the supported element types
//   are int32_t and int64_t.
```

```cpp
// Resolve ragged_offset aux from the variant pack first — the
// view reads ragged_offset[B] from it to size itself.
//
// The aux's element type (int32 or int64) is cached in
// _params at plan-build time from the flatbuffer aux's
// data_type field; makeShallowITensor returns a
// ShallowTensor<int32_t> or ShallowTensor<int64_t> wrapped as
// shared_ptr<ITensor> based on that tag. The ragged-tensor
// types treat it type-erased from here on out (§4.5 item 1).
std::shared_ptr<ITensor> qRaggedOffset = makeShallowITensor(
    _params.qTensor.raggedOffsetDataType,
    variantPack.at(_params.qTensor.raggedOffsetUid),
    /* aux dims/strides: [B + 1, 1, 1, 1], packed */);

// The view's buffer is exactly ragged_offset[B] elements;
// alignment plays no part in sizing (§4.2). physicalElementCount
// is inferred from the aux here (see §4.6 for passing it
// explicitly).
auto qView = std::make_shared<ShallowRaggedTensor<QType>>(
    variantPack.at(_params.qTensor.uid),
    _params.qTensor.dims,
    _params.qTensor.strides,
    qRaggedOffset);

// If the op consumes seq_lens, fetch it as an ordinary input.
// seq_lens has its own data_type cached in _params; the same
// makeShallowITensor factory is used (it is not specific to
// ragged_offset). The CPU reference takes the result as
// TensorBase<int32_t>& or TensorBase<int64_t>& depending on
// what the op declared.
auto seqLenQ = _params.seqLenQTensor.has_value()
    ? makeShallowITensor(_params.seqLenQTensor->dataType,
                         variantPack.at(_params.seqLenQTensor->uid),
                         _params.seqLenQTensor->dims,
                         _params.seqLenQTensor->strides)
    : nullptr;

// CPU reference signature changes only by accepting seq_lens as
// an ordinary extra parameter (where it isn't already), since
// it's no longer attached to the ragged primary:
cpuFpReferenceSdpa.forward(*qView, /* ..., */
                           seqLenQ.get(),
                           /* ... */);
```

Note that neither `qView` nor any helper above is templated on
the aux's index type — `ShallowRaggedTensor<QType>` has a single
template parameter (the primary's data type), and the aux flows
through as `shared_ptr<ITensor>`. `makeShallowITensor` does a
small runtime switch on the data-type tag cached in `_params` and
returns an `ITensor`-typed result; per-op plan templates do not
need an `IndexT` parameter (see
[§6.3](#63-templating-the-ragged-tensor-types-on-indext)).

Resolving the buffer size from `ragged_offset` per execute (rather
than caching it at plan-build time) avoids pinning `ragged_offset`
values to the lifetime of the compiled plan: each execute is free
to use different `ragged_offset` contents.

**CPU reference body.** The ragged primary's iterator walks each
batch's full per-batch range. When the reference also takes a
`seq_lens`, the body queries `seq_lens[b]` directly to decide
whether to skip indices `>= seq_lens[b]` within batch `b` — the
same pattern as any other op input that bounds the work shape.
The reference does **not** need any new `validSeqLen(b)`-style API
on the ragged primary itself.

#### 4.10.1 Reuse at the executor virtual-tensor pass

The CPU executor's existing virtual-tensor pass walks `tensorMap`
from the deserialized flatbuffer graph, allocates one `ITensor` per
virtual attribute that is not already in the variant pack, and
patches `rawHostData()` of the new allocation back into the
variant pack. For ragged virtual intermediates this needs the
same per-execute aux-resolution dance as the plan layer, since
the aux is in the variant pack as a `void*` and the aux's
geometry (dims, `data_type`) lives in `tensorMap`.

To keep the executor's loop body single-shaped (no per-attribute
ragged branching at the call site) and to share the dispatch
logic with the plan layer, the flatbuffer-side factory grows a
second overload:

```cpp
// hipdnn_test_sdk::detail

// Existing, unchanged: single-dispatch on attribute.data_type();
// allocates Tensor<T>(dims, strides). Used by callers that do
// not need ragged support.
std::unique_ptr<ITensor> createTensorFromAttribute(
    const flatbuffers_sdk::data_objects::TensorAttributes& attribute);

// New: handles both ragged and non-ragged attributes uniformly.
// The variantPack/tensorMap arguments are only consulted when
// attribute.ragged_offset_tensor_uid() is set.
std::unique_ptr<ITensor> createTensorFromAttribute(
    const flatbuffers_sdk::data_objects::TensorAttributes& attribute,
    const std::unordered_map<int64_t, void*>&              variantPack,
    const TensorMap&                                       tensorMap);
```

The new overload's contract:

- **If `attribute.ragged_offset_tensor_uid()` is unset** —
  defers to the existing single-arg overload. Non-ragged
  attributes pay no overhead and the executor's loop body sees
  no behavioural change for them.
- **If `attribute.ragged_offset_tensor_uid()` is set** —
  1. Looks up the aux UID in `tensorMap` (for dims and
     `data_type`).
  2. Looks up the aux UID in `variantPack` (for `void*`).
  3. Wraps the aux pointer with `makeShallowITensor(auxDataType,
     auxPtr, auxDims, auxStrides)` to obtain a non-owning
     `shared_ptr<ITensor>` over the aux.
  4. Single-dispatches on the primary's `attribute.data_type()`
     and allocates
     `make_unique<RaggedTensor<T>>(dims, strides, auxSharedPtr)`,
     letting the ctor infer the buffer size as `ragged_offset[B]`
     from the aux. (`alignment` is not consulted — it does not
     affect buffer size; see
     [§4.2](#42-frontend-tensorattributes-additions).)
  5. Returns `unique_ptr<ITensor>`.

`populateVariantPackWithMissingVirtualTensors` switches to
calling this new overload for every virtual attribute. Because
non-ragged attributes fall through to existing behaviour, the
loop body's shape does not change:

```cpp
for (const auto& [id, attr] : tensorMap)
{
    if (attr->virtual_() && updatedVariantPack.find(id) == updatedVariantPack.end())
    {
        auto tensor = detail::createTensorFromAttribute(
            *attr, updatedVariantPack, tensorMap);
        tensor->fillWithSentinelValue();
        virtualTensors.push_back(std::move(tensor));
        updatedVariantPack[id] = virtualTensors.back()->rawHostData();
    }
}
```

The executor never synthesizes a fresh `ragged_offset` for an
intermediate — the aux comes from the pre-supplied bundle via
the variant pack ([§4.9](#49-wiring-sources-of-truth) and
[§4.11.1](#4111-pre-supplied-input-bundle)). A virtual ragged
intermediate's `TensorAttributes` must therefore declare an
existing aux UID that was already threaded through by the
harness; the strict requirement check in §4.11.1 ensures this is
the case.

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
  bundle to obtain its `shared_ptr<ITensor>` — the strict check
  in [§4.11.1](#4111-pre-supplied-input-bundle) guarantees its
  presence — and allocate
  `make_shared<RaggedTensor<T>>(dims, strides,
  raggedOffsetSharedPtr)`, letting the ctor infer the buffer size
  as `ragged_offset[B]`. The factory does not need to know the
  aux's element type statically — it is single-dispatched on the
  primary's `T` only.

A single-pass walk suffices because every `ragged_offset` aux a
primary could reference is, by the strict check, already held by
the pre-supplied bundle before allocation begins. No auxiliary
pre-pass over `TensorAttributes` is needed.

The buffer is sized to exactly `ragged_offset[B]` elements:

```
physicalElementCount = ragged_offset[B]
```

`alignment` does not enter this calculation (see
[§4.5](#45-data-sdk-shared-elements) item 6); `elementCount()`
on the resulting `RaggedTensor` is likewise `ragged_offset[B]`.

#### 4.11.3 Bundle init pass

The init pass walks every UID in the (per-path) bundle once and
applies the following rule:

- **UID is in the pre-supplied bundle:** skip. The pre-supplied
  `ITensor` already carries deliberate values (see
  [§4.11.1](#4111-pre-supplied-input-bundle)); both paths see
  the same `ITensor` via shared ownership.
- **UID is a ragged output:** fill the entire physical buffer
  (both bundles) with the per-dtype sentinel value rather than
  randomizing. The rationale and the validator change that
  consumes this are in
  [§4.11.4](#4114-output-validation-in-verifygraph).
- **UID is a ragged primary input** (non-pre-supplied):
  `bundle.randomizeTensor(uid)` walks the `RaggedTensor<T>`
  per-batch over its `ragged_offset` ranges, filling every
  position in every batch's range (including any trailing
  per-batch padding tails). Filling more positions than strictly
  necessary is harmless — neither GPU kernels nor CPU references
  read padding positions, and identical seeds keep padding
  values byte-equal between the two paths.
- **UID is a non-ragged tensor:** randomize linearly as today.

The same one-pass loop in `initializeBundle` handles all four
cases; the only new logic is the "is this UID a ragged output?"
branch that picks `fillWithSentinelValue()` over
`randomizeTensor()`. Whether a UID corresponds to an output is
already tracked by the harness in `outputTensorIds`.

#### 4.11.4 Output validation in `verifyGraph`

The CPU reference writes only valid sequence positions of an
output (those `< seq_lens[b]`); the GPU kernel makes no
contractual guarantees about output padding. A naive full-buffer
compare therefore fails on padding for reasons unrelated to
algorithmic correctness.

**Recommended approach: sentinel-skip.** Both bundles' ragged
outputs are seeded with a sentinel value at init time (see
[§4.11.3](#4113-bundle-init-pass)) rather than randomized. The
sentinel must be a value the CPU reference cannot legitimately
produce (NaN for floats; a magic number for ints). After
execution the validator iterates the full physical buffer;
positions where the CPU side is bit-equal to the sentinel are
skipped (the reference didn't write there); other positions use
the existing tolerance check. This needs one validator overload
(`allCloseSkipSentinel(cpu, gpu, sentinel, tolerance)`) plus
the init-time branch already described in §4.11.3.

Sentinel-skip is the only validation mechanism in this design.
Because `seq_lens` is not attached to the ragged primary, there
is no SDK-level "wrap with `seq_lens` for ragged-aware compare"
fallback to fall back to.

Even where no value is guaranteed to be unproducible by the reference,
a random sentinel has a very low likelihood of showing up in
the output (though this rises with very large tensors). The
consequence of the CPU reference producing it is also small: the
validator would only skip over this one value.

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
   types even though `elementCount() == elementSpace()` (both
   `ragged_offset[B]`), because the buffer is neither `prod(dims)`
   elements long nor regularly strided. This diverges from the
   literal `elementCount == elementSpace` predicate. See
   [§4.5](#45-data-sdk-shared-elements) item 7 and
   [§7](#future-work) for a possible `hasRegularDims()` split.
6. **`ragged_offset` element-type checking is runtime, not
   static.** The aux is held as `shared_ptr<ITensor>` and the
   int32-vs-int64 distinction is resolved by a small element-size
   switch inside the `readOffset` helper
   ([§4.5](#45-data-sdk-shared-elements) item 1). Element types
   other than int32 and int64 are rejected at construction time.
   The trade — runtime-checked aux type for no `IndexT` template
   parameter on `RaggedTensor` / `ShallowRaggedTensor` / per-op
   plan templates / test-SDK factories — is discussed in
   [§6.3](#63-templating-the-ragged-tensor-types-on-indext).
7. **`getIndex` becomes a virtual call for non-packed tensors.**
   Forwarding `getIndex` to the new `getIndexImpl` virtual
   ([§4.5](#45-data-sdk-shared-elements) item 3) adds one vtable
   dispatch to a previously inlinable function — on the per-element
   path of `CompositeIndex::getValue` and every `operator()` /
   `getHostValue` / `setHostValue` call, for *all* non-packed
   tensors, not only ragged ones. (Packed tensors iterate via
   `LinearIndex`, which never calls `getIndex`.) The cost is
   negligible for the correctness-oriented CPU-reference paths this
   RFC targets, consistent with the perf stance in
   [§6.3](#63-templating-the-ragged-tensor-types-on-indext).

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
  default addressing (`getIndex` / `getHostValue` /
  `setHostValue`) has no meaningful semantics on the ragged
  primary's physical buffer** — with no aux ref it cannot base
  addressing at `ragged_offset[b]`, so addressing would need to
  either throw or address into "raw physical positions",
  motivating a further hierarchy refactor (introducing a
  `TypedTensor<T>` between `ITensor` and `TensorBase<T>`) just to
  clean up.
- **Two distinct `elementCount()` contracts** (one for the
  storage type, one for the view) that callers have to keep
  straight.

The single-type design (`RaggedTensor<T>` +
`ShallowRaggedTensor<T>`) avoids all three of these costs while
incurring the same `shared_ptr`-typed bundle storage either way.

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
- **Associations between `seq_lens` and primaries are determined by the nodes through context.**
  Unlike the `ragged_offset`, which is directly specified in the
  `TensorAttributes`, the association between `seq_lens` and a primary
  is determined by the intended use of arguments for the node. This would
  require a per-node mapping that associates primaries with their intended
  sequence length parameters, and walking each of the nodes to find that mapping.
- **`seq_lens` does not bind 1:1 to a primary.** There is nothing
  that structurally guarantees that a single primary cannot have two
  different `seq_lens` associated with it in two different operations
  in the graph. Even if we took the previous approach of walking the nodes
  to find their associations, this mapping is ill-formed.
- **Significant complexity savings.** Dropping `seq_lens` from
  the SDK removes the storage-vs-view split, the
  post-construction-attach-seq_lens problem, the `validSeqLen`
  virtual on `TensorBase`, a seq_lens-aware output-validation
  fallback, and the strict-requirement check on `seq_lens` UIDs
  in the pre-supplied bundle.

The deciding factor is the cost/benefit ratio: a tiny amount of
work shifted into per-op CPU references in exchange for a much
simpler SDK and harness.

### 6.3 Templating the ragged-tensor types on `IndexT`

An earlier draft of this RFC parameterized both ragged-tensor
types on the aux element type — `RaggedTensor<T, IndexT>` and
`ShallowRaggedTensor<T, IndexT>` (defaulting to `int32_t`,
permitting `int64_t`). The aux was then held as
`std::shared_ptr<TensorBase<IndexT>>`, with reads going through
ordinary typed access.

This is statically type-safe on the aux, but it requires `IndexT`
to be known at every construction site:

- **`createTensorFromAttribute`** (test-SDK and flatbuffer-side)
  becomes a double-dispatch factory keyed on
  `(primary_data_type, aux_data_type)`. The aux's `data_type`
  must be discovered by a side lookup of the aux's own
  `TensorAttributes` (by UID) in the same graph, in addition to
  the primary's.
- **Per-op plan templates** (e.g.
  `SdpaFwdPlan<QType, KType, VType, OType>`) gain `IndexT` as an
  additional parameter, multiplying the plan-builder selection
  matrix in `buildPlanForNode`.
- **Every test-SDK helper that touches a ragged tensor** gains
  the same parameter.
- **`SdpaFwdParams`** (and the analogous per-op `params` struct)
  has to cache an `IndexT` tag at plan-build time so per-execute
  construction can pick the right specialization.

The alternative this RFC takes — holding the aux as
`shared_ptr<ITensor>` and reading through a runtime element-size
branch widening to `int64_t` (see
[§4.5](#45-data-sdk-shared-elements) item 1) — costs a single
runtime branch per offset read. The branch widens an `int32_t`
or `int64_t` aux element up to `int64_t`; on any modern CPU this
is essentially free relative to the per-element work the CPU
reference performs, and the structural invariants on the aux
(rank 4, packed, B+1 long, int32 or int64) bound the surface
where type-erasure could go wrong to one small helper. In
return:

- `RaggedTensor<T>` and `ShallowRaggedTensor<T>` have a single
  template parameter (the primary's data type only).
- `createTensorFromAttribute` stays single-dispatched on the
  primary's `T`.
- Per-op plan templates do not grow an extra parameter; the
  plan-builder matrix stays at its existing size.
- The CPU reference signature is unchanged — it still takes
  `TensorBase<T>&` for the primary, and the aux is hidden inside
  the ragged tensor's `getIndexImpl` override (reached via
  `getHostValue` / `setHostValue` and the iterator).
- The single sourced cost is the runtime element-size branch
  inside `readOffset`, paid on each offset read by the CPU
  reference.

This is the right trade for the in-scope use cases:
correctness-oriented CPU references are not perf-critical paths,
and the type-erasure stays well-contained. See
[§7.4](#74-revisit-typed-indext-propagation-if-needed) for the
escape hatch should a future code path force the static-type
question.

---

## Future work

### 7.1 Unify `RaggedTensor` and `ShallowRaggedTensor` as one templated class

The two types share their `RaggedCompositeIndex` implementation,
their constructor-time structural validation, and (after the
`seq_lens` decoupling) their entire surface. Only memory
ownership differs. They could plausibly be expressed as one class
template parameterized by the memory carrier (owning
`MigratableMemory<T, …>` vs borrowed `void*`). The split kept here
is the conservative choice that mirrors the existing
`Tensor<T>` / `ShallowTensor<T>` split; unifying them is a
mechanical refactor left as follow-up.

Depending on the work involved, it may make sense for this
to be included in the initial implementation.

### 7.2 Split `isPacked()` into orthogonal predicates

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

### 7.3 Improve CPU validation sentinel value handling

Document the sentinel choice and the "the reference cannot
produce this value" guarantee in each CPU reference header. NaN
is the default safe choice for floats, but watch for edge cases —
e.g. SDPA forward with `seq_lens_q[b] == 0` can produce NaN
naturally (`log(0)` from a fully masked row), in which case a
non-NaN magic-number sentinel is required. For low-precision
floating-point types (bf16, fp16, fp8) the sentinel space is
smaller and "unproducible by reference" is harder to guarantee. A
future RFC could codify the sentinel-per-dtype-per-op table.

Additionally, there are ways to reduce the chance of a naturally
produced output matching a sentinel value being overlooked. For
instance, it is necessary that there is a block of sentinel values
at the end of each row. If a sentinel value appears between valid values
in the same row, it must be genuine output. This greatly reduces the
likelihood of real output being overlooked.

There may also be a solution that does not rely on sentinel values
identified in the future, which could replace this strategy.

### 7.4 Revisit typed-`IndexT` propagation if needed

The current design type-erases the `ragged_offset` aux as
`shared_ptr<ITensor>` and reads through a runtime element-size
branch (see
[§4.5](#45-data-sdk-shared-elements) item 1 and
[§6.3](#63-templating-the-ragged-tensor-types-on-indext)). The
SDK therefore never knows the aux's element type statically.

If a future code path emerges where the aux's element type *must*
be known statically — for example, a templated SDK-level helper
that hands the aux directly to a GPU function specialized on the
index type, rather than passing it through the variant pack to
the kernel as a `void*` — the type-erased path can be replaced
with either:

- A `std::variant<shared_ptr<TensorBase<int32_t>>,
  shared_ptr<TensorBase<int64_t>>>`-typed aux handle (visit-on-use,
  static dispatch at each site), or
- Full `IndexT` propagation through `RaggedTensor`,
  `ShallowRaggedTensor`, the per-op plan templates, and the
  test-SDK factories (as analyzed in §6.3).

Neither change is needed for the in-scope use cases, where the
aux is only read inside the ragged tensor's `getIndexImpl`
override (reached via `getHostValue` / `setHostValue` and the
iterator) and where GPU kernels consume the aux as an opaque
`void*` from the variant pack.
