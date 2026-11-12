# hipDNN: Ragged Tensors Design Document

- Contributors: hipDNN maintainers
- **Status**: In progress
- **Status notes**: Frontend, flatbuffer, backend, plugin-version gate, and data SDK types (`RaggedTensor<T>`, `ShallowRaggedTensor<T>`) shipped (#8834, #11382, #12190). Plan-layer consumption was built first in the GPU reference (`GpuSdpaRaggedFwdPlan`, #11532) with ragged SDPA bundles (#12738), and now also in the CPU reference (`SdpaRaggedFwdPlan`). Both executors run the single-node ragged SDPA graphs the native plans decline, forward or backward, through a dense padded rewrite when it accepts them; see §4.10. Per-batch origins, lengths from `seq_len`, strided and head-major layouts, and the exact element-space footprint land with #15080.
- **Last verified**: 2026-11-04 against #15080 on `develop` (49a01e717d5)

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
     - 4.5.1 [Worked examples](#451-worked-examples)
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
`X` varies. Batch `b` starts at the origin
`ragged_offset[b] * multiplier` of a single buffer and has `len_b`
live rows, laid out with ordinary strides. The buffer holds exactly
the addressed range (`elementSpace` elements). Layouts whose token rows
overlap or that violate the sorted-stride rule are rejected; zero-stride
axes are broadcasts ([§4.5](#45-data-sdk-shared-elements) item 5).

This RFC adds end-to-end ragged-tensor support:

1. **Frontend (`TensorAttributes`)** gains `set_ragged_offset` /
   `get_ragged_offset` and `set_alignment` / `get_alignment`.
2. **Flatbuffer schema** gains defaulted `ragged_offset_tensor_uid`,
   `alignment`, and `ragged_offset_multiplier` fields (wire-compatible
   per RFC 0005). `alignment` is a needed addition in its own right (and
   is cuDNN-compatible); it is bundled into the same schema change so
   plugins update once to support both it and ragged tensors, though no
   logic in this RFC consumes it. `ragged_offset_multiplier`, by
   contrast, is consumed by SDK logic and version-gated (see
   [§4.3](#43-flatbuffer-schema-additions)).
3. **Backend** propagates the new fields through existing
   get/set-attribute paths; no new C-API entry points; variant pack
   unchanged.
4. **Data SDK** adds a single owning type `RaggedTensor<T>`
   (memory + `ragged_offset` aux ref) and its non-owning peer
   `ShallowRaggedTensor<T>`. Both expose ragged-aware
   iteration over each batch's origin and length. The `ragged_offset`
   aux is held type-erased as `std::shared_ptr<ITensor>` and read
   through a runtime element-size branch (int32 or int64) that
   widens to `int64_t` at the read site, so neither ragged-tensor
   type carries an `IndexT` template parameter. Each stored offset is
   scaled to element units at a single read boundary by a per-tensor
   `ragged_offset_multiplier` (default 1):
   `element_offset = stored_offset * ragged_offset_multiplier`.
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

`seq_lens` stays an ordinary graph tensor that ops reference through
their own node-level attribute APIs (e.g.
`SdpaAttributes::set_seq_len_q`). A ragged tensor additionally takes
it as an optional length tensor that bounds the live rows
([§4.5](#45-data-sdk-shared-elements) item 4).

The immediate consumer is SDPA on AITER FMHA kernels.

---

## Problem Statement

### 2.1 What is a ragged tensor

A ragged tensor (referred to here as the primary tensor) has logical
dims `[B, …]` with one axis `X` (typically the sequence dimension) whose
extent varies per batch; the logical dim gives its maximum. Batch `b`
starts at the origin `ragged_offset[b] * multiplier` of the physical
buffer and has `len_b` live rows. The strides place everything else, so
the batch need not be the outermost physical dimension: in token-major
(THD) layouts the tokens of all batches form the outer axis, `[T, H, D]`,
and in head-major layouts `H` is outermost, `[H, T, D]`, with each head
holding the tokens of every batch grouped by batch
([§4.5](#45-data-sdk-shared-elements) item 4).

Some ops additionally need to distinguish *valid* sequence
positions within each batch's range from *trailing padding*
positions kept for alignment. That information is carried by a
separate `seq_lens[B]` tensor referenced through the op's
node-level attributes (e.g. `SdpaAttributes::set_seq_len_q`). Each
entry in `seq_lens` specifies the number of valid elements in that
batch. `seq_lens` can be used in isolation, or with the
`ragged_offset` (and the `ragged_offset` similarly functions with or
without it). When both are present, `seq_lens` gives the batch lengths
and the offsets only give the origins.

Both `ragged_offset` and `seq_lens` aux tensors may be referenced
by more than one primary in the same graph — SDPA, for example,
typically shares a single `seq_lens_kv` across the K and V
primaries and may share a `ragged_offset_kv` between them as well.
The design supports this naturally: each primary's
`TensorAttributes` references the aux by UID; the runtime holds
one `ITensor` per UID; multiple `RaggedTensor<T>`s share the same
`ragged_offset` aux via `shared_ptr<ITensor>`.

A `ragged_offset` may be stored either in element units or in coarser
units than a primary's element stride — e.g. one *token* offset per
batch boundary. Each primary carries a `ragged_offset_multiplier`
recovering element units at the single read boundary
(`element_offset = stored_offset * multiplier`; default 1 means offsets
are already in element units). This lets one shared token-unit aux
serve primaries of differing `H*D`: AITER's SDPA implementation shares a
`ragged_offset` across Q/O and another across K/V, and each primary sets
its own `multiplier = H*D = seqStride` so the same stored offsets
address each buffer correctly.

Where a ragged tensor takes `seq_lens` is given in
[§4.5](#45-data-sdk-shared-elements) item 4.

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
random fill. Random values are meaningless for `ragged_offset`: its
values must be whole-token origins that give the primary a layout the
container accepts. The harness needs a way to consume deliberate
values for this aux tensor.

### 2.3 Configurations targeted in this iteration

Both AITER FMHA configurations are uniformly representable. The
physical buffer is always exactly `elementSpace` elements:

- **Packed only.** Each batch contributes
  `ragged_offset[b+1] - ragged_offset[b]` rows of valid data; no
  internal padding. `elementSpace` equals `ragged_offset[B]`.
- **Packed with per-batch trailing padding.** Each batch's range
  still goes from `ragged_offset[b]` to `ragged_offset[b+1]`, but
  only the first `seq_lens[b]` rows within that range are valid;
  the remaining
  `ragged_offset[b+1] - ragged_offset[b] - seq_lens[b]` rows are
  padding kept for alignment.

  Without a bound length tensor the ragged tensor iterates the full
  per-batch range and the op queries `seq_lens` to decide what to do
  with positions `>= seq_lens[b]`. With one, the ragged tensor
  iterates only the first `seq_lens[b]` rows.

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
   iteration over each batch's origin and length. The `ragged_offset`
   aux is held type-erased as `std::shared_ptr<ITensor>` (no
   `IndexT` template parameter — see
   [§4.5](#45-data-sdk-shared-elements) item 1). `seq_lens` enters
   as an optional length tensor with no SDPA knowledge in the types
   ([§4.5](#45-data-sdk-shared-elements) item 4).
3. **`ITensor`** gains two complementary polymorphic hooks: a
   `getIndexImpl` override point behind `getIndex` so ragged
   *addressing* bases each batch at its origin, and a
   `raggedIterationInfo()` data hook the iterator dispatches on so
   ragged *iteration* can supply a `RaggedCompositeIndex`.
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

auto set_ragged_offset_multiplier(int64_t value) -> Tensor_attributes&;
auto get_ragged_offset_multiplier() const -> int64_t;   // default 1
bool has_ragged_offset_multiplier() const;              // != default
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

- `get_alignment() >= 1`.
- `get_ragged_offset_multiplier() >= 1`.
- A non-default multiplier requires a `ragged_offset` to be set.

The frontend does **not** inspect the aux's rank or first dim: those
structural constraints (rank 4, first dim `B + 1`, int32/int64 element
type) are enforced at SDK construction in
`RaggedTensorBase::validateRaggedStructure`
([§4.5](#45-data-sdk-shared-elements) item 5).

`seq_lens` validation, where an op cares, lives on that op's
node-level `validate()`.

**Reported `dims()`.** The primary's sequence dim
(`dims()[seqAxis]`; `dims()[2]` for SDPA's logical `[B, H, S, D]`)
remains the max padded sequence length (`S_max`), matching
overridable-shape semantics without re-deriving `S_max` from
`ragged_offset` at runtime.

### 4.3 Flatbuffer schema additions

`tensor_attributes.fbs` gains three appended optional defaulted
fields:

```
ragged_offset_tensor_uid: long = null;
alignment:                long = 16;
ragged_offset_multiplier: long = 1;
```

Wire-compatible per RFC 0005. `alignment` is a
needed addition in its own right (kept consistent with cuDNN, see
[§4.2](#42-frontend-tensorattributes-additions)); it is appended
in the same change so that plugins make a single update to support
both alignment and ragged tensors, rather than going through two
separate rounds of schema evolution and version bumps. No code in
this RFC reads it.

`ragged_offset_multiplier` differs from `alignment` on both counts: the
SDK addressing path reads it ([§4.5](#45-data-sdk-shared-elements) item
1) and it is gated by its own plugin-API floor (below). It is folded
into the compiled-plan cache key **by value**, since it changes
execution.

No `seq_lens_tensor_uid` is added
here: ops that consume `seq_lens` reference it through their own
per-op attribute tables (e.g. `SdpaAttributes` already carries
`seq_len_q_tensor_uid` / `seq_len_kv_tensor_uid`).

To ensure that only plugins that have appropriately handled
ragged tensors process graphs with them present, this also
requires the backend to detect ragged tensors in the graph
(`GraphDescriptor::hasRaggedTensors()`) and
`computeMinimumEnginePluginApiVersion` to map that flag to
`K_RAGGED_TENSOR_MIN_API_VERSION` (`"1.3.0"`).

`ragged_offset_multiplier` carries its own floor
`K_RAGGED_OFFSET_MULTIPLIER_MIN_VERSION = "1.4.0"`, which **dominates**
the ragged-tensor floor; `K_MAX_SUPPORTED_API_VERSION` is now `"1.4.0"`.
It is surfaced by `GraphDescriptor::hasRaggedOffsetMultiplier()` (true
when any tensor carries a non-default multiplier), which feeds
`computeMinimumEnginePluginApiVersion` alongside the ragged flag.

This also requires an update to the TensorAttributes and Graph
json serialization/deserialization to output these values, and
set appropriate defaults on older graphs where they aren't
present.


### 4.4 Backend wiring

Backend tensor descriptor mirrors the frontend additions: optional
`ragged_offset_tensor_uid`, `alignment` (default 16), and
`ragged_offset_multiplier`
(`HIPDNN_ATTR_TENSOR_RAGGED_OFFSET_MULTIPLIER` = 1311, INT64, default 1),
exposed through existing get/set-attribute paths under new enum values
in the hipDNN extension range. As on the frontend, `alignment` is
carried through but unused here (see
[§4.2](#42-frontend-tensorattributes-additions)); unlike `alignment`,
`ragged_offset_multiplier` feeds the version gate
([§4.3](#43-flatbuffer-schema-additions)). No new `hipdnnBackend*` entry points,
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
   widened to `int64_t` at the read site). The constructor reads an
   aux through one type-erased helper:

   ```cpp
   // First `count` values of an index aux, widened to int64_t and
   // scaled by `scale`.
   static std::vector<int64_t> readIndexTensor(ITensor& aux, size_t count,
                                               int64_t scale, const char* name);
   ```

   It switches on `elementSize()` (4 or 8 bytes) and throws
   otherwise. Value `i` is read at physical element
   `i * strides()[0]` of the aux, so an aux is read at its batch
   stride (the stride must be `>= 1` when more than one value is read).
   The constructor reads the offsets (`B + 1` values,
   scaled by the per-tensor multiplier, default 1) and, when given,
   the lengths (`B` values, scale 1) once, and derives the batch
   `origins` and `lengths` (item 4) from them. This is the single
   boundary where the multiplier applies. Every downstream consumer
   (`getIndexImpl`, `raggedIterationInfo`) reads that snapshot, so
   the batch origins (item 4) are already in element units and no
   other addressing math changes.

   The widen-to-`int64_t` cost is negligible relative to the per-
   element work the CPU reference does, and the structural
   invariants in item 5 (shape `[B + 1, 1, 1, 1]`, and the
   element-size check) bound the surface where type-erasure could
   go wrong to this one helper. See
   [§6.3](#63-templating-the-ragged-tensor-types-on-indext) for
   why this is preferred over a templated `IndexT` parameter.
2. **`dims()[seqAxis]` is required to be `S_max`** (the max padded
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
     ragged types override `getIndexImpl` to base each batch at
     its origin (see
     [§4.6](#46-data-sdk-raggedtensort-owning-ragged-aware)).
     Every addressing path funnels through `getIndex`
     (`getHostValue` / `setHostValue` / `operator()`,
     `CompositeIndex::getValue`, `TensorView`), so this single
     override makes direct addressing ragged-aware everywhere — at
     the cost of making `getIndex` a virtual call for all
     non-packed tensors (see [§5](#known-limitations) item 7).
   - **Traversal (`raggedIterationInfo` / `RaggedCompositeIndex`).**
     Add `virtual std::optional<RaggedIterationInfo> raggedIterationInfo() const`
     to `ITensor` (default `std::nullopt`; ragged types return
     `{origins, lengths, seqAxis, seqStride}`, where `origins` and
     `lengths` hold one entry per batch (origin in element units,
     length in sequence rows; item 4), `seqAxis` is the
     caller-provided sequence axis, and `seqStride` is
     `strides()[seqAxis]`; `origins` and `lengths` are authoritative.
     The accessor `rowOffsets()` returns the `B + 1` offset table of
     a packed, in-order layout with no gaps: each non-empty batch
     contributes `origins[b]` and advances the running end by
     `lengths[b] * seqStride`, and the final entry is the end of the
     last live row. Batches with `lengths[b] == 0` own no rows: they
     are skipped by the contiguity check and their entry is the running
     end of the preceding batches (for leading empty batches, the origin
     of the first non-empty batch; when every batch is empty, every
     entry, including the last, is `origins[0]`), whatever their stored origin. The
     accessor throws
     `std::logic_error` naming the non-empty batch `a` that precedes
     the break, when the next non-empty batch's origin is not
     `origins[a] + lengths[a] * seqStride`, i.e. when the non-empty
     batches are not contiguous and in order. It is not part of
     `operator==`) and a `RaggedCompositeIndex` to the
     `IndexType` variant that walks each batch's live rows. The iterator's
     `makeIndex` dispatches on the hook, snapshotting the info once
     at `begin()` / `end()` so traversal does no per-step aux reads.
     This is **required in
     addition to** `getIndexImpl`: a plain `CompositeIndex` derives
     its bounds from `dims()` and would visit `prod(paddedDims)`
     positions rather than the live elements that
     `elementCount()` reports (item 6). `RaggedCompositeIndex`
     emits `{b, within-batch…}` indices and delegates the
     per-element offset back to `getIndex`.

     A data hook is used rather than a literal
     `virtual IndexType makeIndex(bool) const` because `IndexType`
     is scoped *inside* `ITensorIterator<IsConst>` and differs for
     the const/non-const iterators, so the non-templated `ITensor`
     base cannot return it. Returning plain offset data keeps every
     template-dependent index type iterator-internal.
4. **Each batch has an origin and a length; iteration visits the
   live rows.** With `seqStride = strides()[seqAxis]`, batch `b` has
   an origin `origin_b` (element units) and a length `len_b` (sequence
   rows):

   - **With a length tensor** (`seq_len`, `[B, 1, 1, 1]`, INT32 or INT64):
     `origin_b = ragged_offset[b] * multiplier` and
     `len_b = seq_len[b]`. Origins may be in any order.
     `ragged_offset` keeps shape `[B + 1, 1, 1, 1]`. `ragged_offset[B]`
     is read with the table (the shape is checked), but it is neither
     used nor validated:
     origins come from entries `0..B-1` and lengths from `seq_len`.
   - **Without one:** offsets are monotonic,
     `origin_b = ragged_offset[b] * multiplier` and
     `len_b = min((ragged_offset[b+1] - ragged_offset[b]) / seqStride, S_max)`.
     `ragged_offset[0]` may be non-zero.
   - **Aux shape:** a `ragged_offset` aux has shape `[B + 1, 1, 1, 1]`
     and a `seq_len` aux has shape `[B, 1, 1, 1]`; values are read at
     the batch-axis stride. Any other shape is rejected with
     `std::invalid_argument` before any read or device copy.

   Batch `b` owns the token rows `[origin_b / seqStride,
   origin_b / seqStride + len_b)`. Rows outside every batch's rows
   (gap rows) belong to no batch and are not iterated.
   `RaggedCompositeIndex` bounds the sequence axis by `len_b`
   (`RaggedIterationInfo::seqExtent(b)`) and every other non-batch
   axis by its full `dims()`; it emits `{b, within-batch…}` indices
   and delegates the per-element offset to `getIndex`. Every visited
   element belongs to the right batch and to the tensor's owned
   memory. An op that masks beyond the lengths queries `seq_lens`
   from the variant pack.

   **The length tensor.** The container takes the optional length
   tensor (`seq_len`, one value per batch) beside `ragged_offset`, as a
   `std::shared_ptr<ITensor>` supplied by the caller. It has no SDPA
   knowledge and exposes lengths only as `RaggedIterationInfo::lengths`.

   **Why the container takes lengths.** Origins plus lengths are the only
   way to represent the cuDNN layouts where a block is longer than its
   live rows or origins are not monotonic: padded THD, gap rows between
   batches, shifted and non-monotonic origins (§4.5.1, L5s, L7d, L9, L14).
   Without lengths, `len_b` derives from offset differences and none of
   those is expressible. Footprint and the non-aliasing check depend on
   the live rows, so iteration, element count and addressing must agree on
   them.

   **How lengths reach the container.** A `seq_len` belongs to the SDPA
   node, not to a tensor, and one tensor can have different `seq_len`s in
   different nodes. The test SDK maps each ragged tensor uid to its node's
   `seq_len` uid by role (`raggedLengthUids(graph)`):

   | `seq_len` | Forward | Backward |
   |---|---|---|
   | `seq_len_q` | Q, O, Stats, Max, Sum_exp | Q, O, dO, dQ, Stats |
   | `seq_len_kv` | K, V | K, V, dK, dV |

   A tensor whose node has no `seq_len` has no entry. Loaders pass the
   length tensor next to the offset.

   **Identifying the ragged (sequence) axis.** The sequence axis is
   caller-provided at construction (e.g. `SDPA_SEQ_AXIS`), not
   inferred from strides: with a singleton head or `S_max == 1` the
   strides alone cannot tell the sequence axis apart from its
   neighbours.

   Logical dim order is the caller's choice. SDPA tensors keep
   cuDNN's logical `[B, H, S, D]` dims and pass `SDPA_SEQ_AXIS`
   (`2`); `BSHD_SEQ_AXIS` (`1`) is for tensors whose dims are
   physically ordered `[B, S, H, D]`. Ragged offsets on graph tensors
   are an SDPA feature (as in cuDNN), so the graph-driven construction
   sites below (plan layer, executor, harness) pass `SDPA_SEQ_AXIS`.

   The *physical* layout is any layout that item 5 accepts: BSHD
   packing (SDPA strides `[S·H·D, D, H·D, 1]`), head-major
   `[H, T]` Stats and primaries, fused QKV, and padded THD. Layouts
   whose token rows overlap, or that violate the sorted-stride rule,
   are rejected; a zero-stride axis is a broadcast and is accepted.
5. **Constructor-time validation** (enforced by both types): a
   structural pass, then the layout rules (RFC 0014 §4.5). The structural pass runs
   first so the type-erased `readIndexTensor` helper from item 1 only ever
   encounters a supported element size. A violation throws
   `std::invalid_argument` with a message that names the violated
   rule (overlap, sorted-stride, length range, whole-token origin).

   *Structural:*
   - `raggedOffset != nullptr`.
   - `raggedOffsetMultiplier >= 1`.
   - `paddedDims` has rank `>= 2`.
   - `strides.size() == paddedDims.size()`.
   - `seqAxis ∈ [1, rank(paddedDims))`.
   - `raggedOffset->elementCount() == paddedDims[0] + 1`
     (i.e. `B + 1`).
   - `raggedOffset` has shape `[B + 1, 1, 1, 1]`.
   - `raggedOffset->elementSize() == 4 || raggedOffset->elementSize() == 8`
     (int32 or int64 element type).
   - a length tensor, when given, has shape `[B, 1, 1, 1]` and element
     size 4 or 8.
   - the sequence-axis stride is positive (`strides()[seqAxis] > 0`).
   - every other non-batch stride is non-negative (a zero stride
     broadcasts).

   *Origins and lengths* (item 4), read from the offset table
   snapshotted at construction:
   - every origin is a whole number of sequence rows
     (`origin_b % seqStride == 0`) and `>= 0`;
   - with lengths: `0 <= len_b <= S_max`;
   - without lengths: offsets are monotonic non-decreasing and each
     block `ragged_offset[b+1] - ragged_offset[b]` is a whole number of
     sequence rows.

   *Non-aliasing* (every logical element is visited once):
   1. The non-empty token-row intervals of the batches are pairwise
      disjoint.
   2. The sorted-stride rule holds on the dense view
      `[maxRow + 1, other non-batch dims]`: sort the axes with extent
      `> 1` and a non-zero stride by stride; each stride must be at
      least the span of the axes below it. The batch axis is
      excluded and zero-stride (broadcast) axes are exempt.

   Zero-stride axes are broadcasts, not aliasing. An input may carry
   one: every logical element along the axis reads the same storage
   (L12). `elementCount()` still counts every logical element, while
   `elementSpace()` counts the shared storage once. An output with a
   zero-stride axis writes shared storage from several logical
   elements; the GPU reference does not order those writes.
6. **Element-count reporting.** `elementCount()` reports
   `Σ len_b * prod(non-batch, non-sequence dims)`: the elements the
   iterator visits. `elementSpace()` reports `physicalElementCount`,
   the size of the allocated buffer, which is the exact addressed
   range:

   ```
   elementSpace = 0                                     if elementCount == 0
                = 1 + maxRow * seqStride + Σ (d_i - 1) * stride_i   otherwise
   ```

   where `maxRow` is the last live token row and `i` ranges over the
   non-batch, non-sequence axes. An explicit `physicalElementCount`
   must equal it. A ragged `.bin` file holds exactly `elementSpace`
   elements. For packed layouts `elementSpace` equals
   `ragged_offset[B] * multiplier`; for strided, head-major, shifted or
   padded layouts it can be smaller or larger
   ([§4.5.1](#451-worked-examples)).
   `alignment` does **not** enter either calculation: it
   constrains the buffer *pointer*, not the buffer *size* (see
   [§4.2](#42-frontend-tensorattributes-additions)).
7. **`isPacked()` returns `false`** for both new types. Note this
   *diverges* from the literal `Tensor<T>` convention
   `_packed = (elementCount == elementSpace)`: for a packed ragged
   layout those two are equal (item 6), yet
   the buffer is neither `prod(dims)` elements long nor regularly
   strided, so treating it as a flat dense buffer would be wrong.
   The current `isPacked()` predicate conflates two distinct
   properties — "elementCount equals elementSpace" vs "buffer is
   `prod(dims)` elements with regular strides" — and ragged
   tensors are the case that pulls them apart. A follow-up could
   split these, e.g. by introducing a separate `hasRegularDims()`
   predicate (see [§7](#future-work)).

#### 4.5.1 Worked examples

Dims `[B, H, S, D]`, sequence axis 2, offsets in element units,
multiplier 1 unless a cell shows `×m`. The batch stride (`-`) is unused.
Lengths come from the offsets unless a `seq_len` is given.

| ID | Dims | Strides | Offsets; `seq_len` | `len_b` | `elementCount` | `elementSpace` | Outcome |
|---|---|---|---|---|---|---|---|
| L1 | `[2,2,4,2]` | `[-,2,4,1]` | `[0,8,20]` | 2, 3 | 20 | 20 | packed; equals `offset[B]` |
| L2 | `[2,4,3,8]` | `[-,8,96,1]` | `[0,192,480]` | 2, 3 | 160 | 416 | fused-QKV V; span, not `offset[B]` (480) |
| L3 | `[2,4,3,8]` | `[-,8,96,1]` | `[0,0,0]` | 0, 0 | 0 | 0 | every batch empty |
| L4 | `[2,4,8,1]` | `[20,1,4,1]` | `[0,20,40]` | 5, 5 | 40 | 40 | `[T,H]` Stats |
| L5 | `[2,4,5,8]` | `[-,8,32,1]` | `[0,256,512]` | 5, 5 | 320 | 416 | padded THD: block of 8 rows capped at `S_max` 5 |
| L5s | `[2,4,5,8]` | `[-,8,32,1]` | `[0,256,512]`; `[3,5]` | 3, 5 | 256 | 416 | L5 with lengths |
| L6 | `[2,2,5,2]` | `[-,10,2,1]` | `[0,4,10]` | 2, 3 | 20 | 20 | head-major primary |
| L7a | `[2,4,3,1]` | `[-,5,1,1]` | `[0,2,5]` | 2, 3 | 20 | 20 | `[H,T]` Stats, `H = 4` |
| L7b | `[2,4,3,1]` | `[-,1,4,1]` | `[0,8,20]` | 2, 3 | 20 | 20 | `[T,H]` Stats, `H = 4`; same data as L7a |
| L7c | `[2,4,3,1]` | `[-,10,1,1]` | `[0,2,5]` | 2, 3 | 20 | 35 | L7a with a head stride of 10 (padded) |
| L7d | `[2,2,3,1]` | `[-,7,1,1]` | `[0,4,7]`; `[2,3]` | 2, 3 | 10 | 14 | `[H,T]` Stats with gap rows 2-3 between the batches |
| L8 | `[2,1,4,2]` | `[-,2,2,1]` | `[4,8,14]` | 2, 3 | 10 | 14 | shifted origin |
| L9 | `[2,1,4,2]` | `[-,2,2,1]` | `[10,0,0]`; `[2,3]` | 2, 3 | 10 | 14 | non-monotonic origins |
| L9m | `[2,1,4,2]` | `[-,2,2,1]` | `[5,0,0]` `×2`; `[2,3]` | 2, 3 | 10 | 14 | L9 with offsets in multiplier units |
| L7bm | `[2,4,3,1]` | `[-,1,4,1]` | `[0,2,5]` `×4` | 2, 3 | 20 | 20 | L7b with offsets in multiplier units |
| L10 | `[2,1,4,2]` | `[-,2,2,1]` | `[0,2,0]`; `[3,3]` | | | | rejected: rows `[0,3)` and `[1,4)` overlap |
| L11 | `[2,2,4,2]` | `[-,8,2,1]` | `[0,4,10]` | | | | rejected: sorted-stride (BHSD) |
| L12 | `[2,2,3,2]` | `[-,0,2,1]` | `[0,4,10]` | 2, 3 | 20 | 10 | zero-stride head axis (broadcast): both heads share storage |
| L13 | `[2,1,4,2]` | `[-,2,2,1]` | `[0,0,6]` | 0, 3 | 6 | 6 | empty first batch, then a batch at origin 0 |
| L14 | `[2,1,4,2]` | `[-,2,2,1]` | `[0,4,12]`; `[2,3]` | 2, 3 | 10 | 10 | trailing gap dropped; a 12-element file is rejected |
| L15 | `[2,1,4,2]` | `[-,2,2,1]` | `[6,0,0]`; `[0,3]` | 0, 3 | 6 | 6 | empty batch with origin 6, past the live rows |

- **L2.** Blocks are `192 / 96 = 2` and `288 / 96 = 3` rows at origins
  0 and 192, so the rows are `[0,2)` and `[2,5)` and `maxRow = 4`.
  `elementSpace = 1 + 4*96 + 3*8 + 7*1 = 416`; `offset[B]` over-claims
  by 64 elements.
- **L5.** Without `seq_len` each block is `256 / 32 = 8` rows, capped at
  `S_max = 5`. The origins are rows 0 and 8, so `maxRow = 12` and
  `elementSpace = 1 + 12*32 + 3*8 + 7*1 = 416`. L5s keeps the origins
  and shortens the lengths, so the count drops and the space stays.
- **L8.** The origins are rows 2 and 4 (`4/2`, `8/2`), the lengths 2 and
  3, so `maxRow = 6` and `elementSpace = 1 + 6*2 + 1*1 = 14`. The first
  two rows of the buffer are unused.
- **L6.** One head spans the dense view `[5, 2]`: `1 + 4*2 + 1 = 10`
  elements, which is the head stride, so the sorted-stride rule holds.
  L11 has the same view and a head stride of 8, below that span.

### 4.6 Data SDK: `RaggedTensor<T>` (owning, ragged-aware)

A memory-owning ragged tensor whose physical element count is
independent of `prod(dims)` and which holds a `shared_ptr` to its
`ragged_offset` aux. Used as the runtime type for ragged primaries
in the bundle, as the type for ragged graph intermediates, and as
the user-facing type for samples.

**Shared base `RaggedTensorBase<T>`.** Both ragged types derive
from a non-CRTP abstract intermediate
`RaggedTensorBase<T> : public TensorBase<T>` that carries
everything they share — the type-erased `readIndexTensor` /
`raggedIterationInfo`, the `getIndexImpl` ragged-addressing override,
geometry reporting, the caller-provided sequence axis, and all
constructor-time validation (§4.5 item 5).
The concrete types add only the memory carrier and the fill
operations. No CRTP is needed because `memory()` is already virtual
on `TensorBase<T>`. This realizes, at the shared-base level, the
unification anticipated in [§7.1](#71-unify-raggedtensor-and-shallowraggedtensor-as-one-templated-class)
(the two types are not collapsed into a single
memory-carrier-parameterized class).

```cpp
template <typename T>
class RaggedTensorBase : public TensorBase<T>
{
public:
    RaggedTensorBase(std::vector<int64_t>     paddedDims,
                     std::vector<int64_t>     strides,
                     int                      seqAxis,   // e.g. SDPA_SEQ_AXIS
                     std::shared_ptr<ITensor> raggedOffset,
                     std::optional<size_t>    physicalElementCount,
                     int64_t                  raggedOffsetMultiplier = 1,
                     const std::shared_ptr<ITensor>& raggedLength = nullptr); // seq_len [B, 1, 1, 1]; §4.5 item 4

    const std::vector<int64_t>& dims() const override;        // paddedDims
    const std::vector<int64_t>& strides() const override;     // strides
    size_t elementCount() const override;                     // Σ len_b * prod(other dims)
    size_t elementSpace() const override;                     // physicalElementCount == footprint
    bool   isPacked() const override { return false; }
    std::optional<RaggedIterationInfo> raggedIterationInfo() const override; // origins, lengths, axis, stride; rowOffsets()
    const ITensor*       raggedOffset() const;

protected:
    int64_t getIndexImpl(const std::vector<int64_t>& idx) const override;
    static std::vector<int64_t> readIndexTensor(ITensor& aux, size_t count, int64_t scale,
                                                const char* name); // §4.5 item 1
    // Constructor-time validation: §4.5 item 5.

    std::vector<int64_t>     _paddedDims;
    std::vector<int64_t>     _strides;
    int                      _seqAxis;               // caller-provided (e.g. SDPA_SEQ_AXIS)
    int64_t                  _seqStride;             // strides[_seqAxis]
    std::vector<int64_t>     _origins;               // per-batch origin, element units
    std::vector<int64_t>     _lengths;               // per-batch length, sequence rows
    size_t                   _iteratedElementCount;  // Σ len_b * prod(other dims)
    size_t                   _physicalElementCount;  // == elementSpace
    std::shared_ptr<ITensor> _raggedOffset;          // non-null, never reseated
    int64_t                  _raggedOffsetMultiplier; // stored_offset -> element scale
};
```

```cpp
template <typename T,
          typename HostAlloc   = HostAllocator<T>,
          typename DeviceAlloc = DeviceAllocator<T>>
class RaggedTensor : public RaggedTensorBase<T>
{
public:
    // physicalElementCount is optional: when omitted it is inferred
    // as elementSpace. The constructor always reads the offsets (and
    // lengths) at construction to derive the layout, host-side (a
    // device-resident aux incurs a device->host synchronization); an
    // explicit count is a confirmation and must equal elementSpace.
    RaggedTensor(std::vector<int64_t>     paddedDims,
                 std::vector<int64_t>     strides,
                 int                      seqAxis,   // e.g. SDPA_SEQ_AXIS
                 std::shared_ptr<ITensor> raggedOffset,
                 std::optional<size_t>    physicalElementCount = std::nullopt,
                 int64_t                  raggedOffsetMultiplier = 1,
                 const std::shared_ptr<ITensor>& raggedLength = nullptr);   // seq_len [B, 1, 1, 1]; §4.5 item 4

    // ITensor / TensorBase<T> overrides:
    //   dims()         -> paddedDims                    (dims()[seqAxis] == S_max)
    //   strides()      -> strides
    //   elementSpace() -> physicalElementCount           (allocation size == footprint)
    //   elementCount() -> Σ len_b * prod(other dims)     (iterated elements)
    //   isPacked()     -> false
    //   getIndexImpl() -> origin_b + sq*stride_seqAxis + ...  (ragged addressing)
    //   begin/end      -> RaggedCompositeIndex (selected by the
    //                      iterator from raggedIterationInfo(); walks
    //                      each batch's live rows)
    //
    // Direct addressing (rawHostData / rawDeviceData) is supported.

    const ITensor* raggedOffset() const;

private:
    // _paddedDims, _strides, _origins, _lengths, _physicalElementCount,
    // _raggedOffset, readIndexTensor(), getIndexImpl(), and the
    // validation helpers all live on RaggedTensorBase<T> above.
    MigratableMemory<T, HostAlloc, DeviceAlloc> _memory;
};
```

Validation at construction is shared with
`ShallowRaggedTensor` and listed in
[§4.5](#45-data-sdk-shared-elements). Crucially, the aux's
element type (int32 vs int64) is *not* a template parameter — it
is discovered at construction time and validated against the
permitted set (4 or 8 bytes), and individual reads dispatch
through `readIndexTensor` (see [§4.5](#45-data-sdk-shared-elements)
item 1).

Pinned host memory is available via
`RaggedTensor<T, PinnedHostAllocator<T>, …>` — same pattern
`PinnedTensor<T>` uses for `Tensor<T>`.

**Immutability.** The `_raggedOffset` `shared_ptr` is fixed at
construction and never reseated, and `_origins` and `_lengths` are the
snapshot taken then. The optional length tensor is read once at
construction and not retained. The type exposes no setter for any of
them. The primary's T-typed buffer remains mutable as
usual via
`rawHostData` / `rawDeviceData` / `MigratableMemory`.

**Ragged-aware multi-dim addressing.** `RaggedTensor<T>`
overrides `getIndexImpl` (the protected virtual from
[§4.5](#45-data-sdk-shared-elements) item 3) so that a multi-dim
index `{b, i_1, …}` (sequence index `sq = i_seqAxis`) translates to a
physical offset using
`origins[b]` (i.e. `ragged_offset[b] * multiplier` widened to
`int64_t`) as the per-batch base:
`physical_offset = origins[b] + inner_product({i_1, …}, strides()[1:])`.
(The default implementation uses only the padded strides, indexing
into `b * stride_0 + …` regardless of where batch `b`'s range
actually starts in the physical buffer.) A bare index `{b}` bases
at `origins[b]`, and an empty index `{}` returns offset `0`.
Overriding `getIndexImpl` rather than the non-virtual
`getHostValue` / `setHostValue` is what makes every addressing
path ragged-aware at once (§4.5 item 3). Callers may index into
batch `b` with `sq` ranging up to that batch's length
(`len_b = seqExtent(b)`, see §4.5 item 4); indices outside that
range are out-of-bounds for that batch and behavior is
unspecified.

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
//    elementSpace elements; alignment plays no part (§4.2).
//    The aux is passed as shared_ptr<ITensor>; the ragged tensor
//    type-erases its element type at construction (§4.5 item 1).
//    Form (a): let the ctor infer the size by reading
//    the offsets from the aux.
auto qTensor = std::make_shared<utilities::RaggedTensor<float>>(
    qAttr->get_dim(),
    qAttr->get_stride(),
    utilities::SDPA_SEQ_AXIS,
    qRaggedOffset);     // implicit upcast Tensor<int32_t> -> ITensor

//    Form (b): pass physicalElementCount explicitly as a confirmation —
//    the user already knows elementSpace (for a packed layout it is
//    ragged_offset[B]). The ctor still reads the offsets and rejects a
//    count that differs from elementSpace:
//
//    auto qTensor = std::make_shared<utilities::RaggedTensor<float>>(
//        qAttr->get_dim(), qAttr->get_stride(), utilities::SDPA_SEQ_AXIS,
//        qRaggedOffset, static_cast<size_t>(myOffsetsHost.back()));
//
//    Form (c): a shared token-unit offset aux. When ragged_offset
//    stores one token offset per batch boundary, each primary passes
//    its own multiplier = H*D so the shared aux addresses its buffer in
//    element units. AITER's SDPA shares one offset aux across Q and O,
//    each setting its own multiplier (Q and O may differ in H*D):
//
//    auto qTensor = std::make_shared<utilities::RaggedTensor<float>>(
//        qAttr->get_dim(), qAttr->get_stride(), utilities::SDPA_SEQ_AXIS,
//        qoRaggedOffset, std::nullopt, /*raggedOffsetMultiplier=*/qH * qD);
//    auto oTensor = std::make_shared<utilities::RaggedTensor<float>>(
//        oAttr->get_dim(), oAttr->get_stride(), utilities::SDPA_SEQ_AXIS,
//        qoRaggedOffset, std::nullopt, /*raggedOffsetMultiplier=*/oH * oD);
//
//    Form (d): a length tensor. seq_lens[b] rows of each batch are live
//    and origins may be in any order (§4.5 item 4):
//
//    auto qTensor = std::make_shared<utilities::RaggedTensor<float>>(
//        qAttr->get_dim(), qAttr->get_stride(), utilities::SDPA_SEQ_AXIS,
//        qRaggedOffset, std::nullopt, /*raggedOffsetMultiplier=*/1, seqLenQ);

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
class ShallowRaggedTensor : public RaggedTensorBase<T>
{
public:
    // As with RaggedTensor (§4.6), physicalElementCount is optional
    // and inferred as elementSpace when omitted.
    ShallowRaggedTensor(
        void*                    data,
        std::vector<int64_t>     paddedDims,
        std::vector<int64_t>     strides,
        int                      seqAxis,   // e.g. SDPA_SEQ_AXIS
        std::shared_ptr<ITensor> raggedOffset,
        std::optional<size_t>    physicalElementCount = std::nullopt,
        int64_t                  raggedOffsetMultiplier = 1,
        const std::shared_ptr<ITensor>& raggedLength = nullptr);   // seq_len [B, 1, 1, 1]; §4.5 item 4

    // Same overrides as RaggedTensor, inherited from
    // RaggedTensorBase<T>:
    //   dims()/strides() as provided
    //   elementSpace() -> physicalElementCount  (allocation size)
    //   elementCount() -> Σ len_b * prod(other dims)  (iterated elements)
    //   isPacked()     -> false
    //   begin/end      -> RaggedCompositeIndex (selected by the
    //                     iterator from raggedIterationInfo())
    //
    // Host-only, matching ShallowTensor (backed by
    // ShallowHostOnlyMigratableMemory<T>): rawHostData() returns
    // the borrowed pointer; rawDeviceData() throws. Only
    // fillWithValue() is supported — fillWithRandomValues() and
    // fillWithData() throw. A device-capable shallow carrier is
    // deferred to the ASM engine plan layer (workstream I4).

    const ITensor* raggedOffset() const;
};
```

`ShallowRaggedTensor` performs the same constructor-time
validation listed in
[§4.5](#45-data-sdk-shared-elements), including the int32/int64
element-size check on the aux. Its `getIndexImpl` override,
`RaggedCompositeIndex` traversal, and type-erased `readIndexTensor`
helper are inherited from the shared `RaggedTensorBase<T>`
([§4.6](#46-data-sdk-raggedtensort-owning-ragged-aware)); only the
memory carrier and fill operations differ. It is host-only
(`rawDeviceData()` throws), matching `ShallowTensor`; a
device-capable shallow carrier is deferred to workstream I4.

Unlike `RaggedTensor`, `ShallowRaggedTensor` does not carry an
allocator template parameter — pinned-vs-pageable is determined
by the caller-supplied buffer being wrapped, not by the wrapper.

### 4.8 Tiered API: which type each role uses

| Role | Type | Iteration |
|---|---|---|
| Owning storage for a ragged primary (bundle / executor intermediate) | `RaggedTensor<T>` | Per-batch over origins and lengths |
| Ragged I/O at the CPU reference plan layer | `ShallowRaggedTensor<T>` over variant-pack `void*` | Per-batch over origins and lengths |
| Ragged tensor a user iterates directly (samples) | `RaggedTensor<T>` | Per-batch over origins and lengths |
| Aux tensor (`ragged_offset`) | Plain `Tensor<int32_t>` or `Tensor<int64_t>`, held as `shared_ptr<ITensor>` | Dense |
| `seq_lens` (op input; optionally the length tensor of a ragged primary, §4.5 item 4) | Plain `Tensor<int32_t>` or `Tensor<int64_t>` | Dense |
| Non-ragged tensors | Plain `Tensor<T>` | Dense |

### 4.9 Wiring sources of truth

- **Graph (`TensorAttributes`)** carries `ragged_offset` (pointer
  to the aux tensor's attributes) and `alignment`.
- **Node-level op attributes** (e.g. `SdpaAttributes`) carry
  `seq_lens` UIDs via existing accessors; these are referenced
  like any other tensor input. The test SDK binds the role's
  `seq_len` to each ragged primary as its length tensor
  ([§4.5](#45-data-sdk-shared-elements) item 4).
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
  [§4.5](#45-data-sdk-shared-elements) item 1). The node's `seq_len`
  for the primary's role, if any, is passed as the length tensor.
  `seq_lens`, if the op also consumes it, is fetched as a separate
  input from the variant pack the same way any other input would be
  — typically as a `TensorBase<int32_t>&` or `TensorBase<int64_t>&`
  depending on what the op's per-node attribute declares.

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

> **Status.** The frontend → flatbuffer → backend → data-SDK plumbing
> above (including `ragged_offset_multiplier`) is landed. Plan-layer
> consumption is implemented for SDPA forward on both references:
> the CPU `SdpaRaggedFwdPlan.hpp` and the GPU `GpuSdpaRaggedFwdPlan.hpp`.
> These plans accept only an `INT32` `ragged_offset` aux, although the
> sketches below also allow `int64`. They accept every layout the
> container accepts ([§4.5](#45-data-sdk-shared-elements)) and the
> `seq_len_q` / `seq_len_kv` pair (both or neither); they decline
> `Max`, `Sum_exp` and ragged Q over paged K/V. Any other single-node
> ragged SDPA graph, forward or backward, that the rewrite accepts runs
> through a dense padded rewrite (`RaggedSdpaDensify.hpp`) that unpacks the ragged tensors,
> runs the dense plan and packs the outputs back. The rewrite accepts
> `INT32` and `INT64` offsets, serves ragged `Max` and `Sum_exp` and
> ragged Q over non-ragged paged K/V, and declines what it cannot
> rewrite, so the graph is unsupported: a `ragged_offset` on a paged K/V
> container; a ragged tensor that is not a Q/K/V/O/stats primary; a
> partly ragged Q-side or K/V-side group; Q or K that is not rank 4;
> ragged tensors whose batch differs from Q's, that are virtual, or whose
> strides alias; an offset table that is not `INT32` / `INT64` `[B + 1]`;
> a `seq_len` that is not `INT32` `[B, 1, 1, 1]`; and data types the
> dense plan does not register. No op other than SDPA consumes ragged tensors
> yet. The §4.11 pre-supplied input bundle is not implemented.
> The bundle harness loads ragged inputs from golden blobs instead.
> The sketches below show the intended shape.

The plan's `params` struct caches each ragged tensor's UID, that of
its `ragged_offset` aux UID, and the primary's
`ragged_offset_multiplier` (from the flatbuffer tensor field). At
execute time the plan resolves the tensors via the variant pack:

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
// view reads the batch origins from it.
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
    /* aux dims: [B + 1, 1, 1, 1] */);

// If the op consumes seq_lens, fetch it as an ordinary input.
// seq_lens has its own data_type cached in _params; the same
// makeShallowITensor factory is used (it is not specific to
// ragged_offset). The CPU reference takes the result as
// TensorBase<int32_t>& or TensorBase<int64_t>& depending on
// what the op declared. The same tensor is the view's length
// tensor.
auto seqLenQ = _params.seqLenQTensor.has_value()
    ? makeShallowITensor(_params.seqLenQTensor->dataType,
                         variantPack.at(_params.seqLenQTensor->uid),
                         _params.seqLenQTensor->dims,
                         _params.seqLenQTensor->strides)
    : nullptr;

// The view's buffer is exactly elementSpace elements;
// alignment plays no part in sizing (§4.2). physicalElementCount
// is inferred from the layout here (see §4.6 for passing it
// explicitly). The primary's ragged_offset_multiplier, cached in
// _params from the flatbuffer tensor field, scales the stored
// offsets to element units (§4.5 item 1).
auto qView = std::make_shared<ShallowRaggedTensor<QType>>(
    variantPack.at(_params.qTensor.uid),
    _params.qTensor.dims,
    _params.qTensor.strides,
    SDPA_SEQ_AXIS,
    qRaggedOffset,
    std::nullopt,                          // physicalElementCount inferred
    _params.qTensor.raggedOffsetMultiplier,
    seqLenQ);                              // length tensor (or null)

// CPU reference signature changes only by accepting seq_lens as
// an ordinary extra parameter (where it isn't already):
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
batch's live rows (`s < len_b`). With a bound length tensor those
are the first `seq_len[b]` rows of the batch's range, so the
reference needs no extra skip for padding. The reference reads
`seq_lens` for masking, like any other op input that bounds the work
shape, and needs no `validSeqLen(b)`-style API on the ragged primary.

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
     `make_unique<RaggedTensor<T>>(dims, strides, SDPA_SEQ_AXIS,
     auxSharedPtr, std::nullopt, attribute.ragged_offset_multiplier(),
     lengthsOrNull)`,
     letting the ctor infer the buffer size as `elementSpace`,
     forwarding the primary's multiplier and the node's `seq_len` (or
     null) as the length tensor (§4.5 item
     1). (`alignment` is not consulted — it does not
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

Not yet built. `PreSuppliedInputs` does not exist. Ragged SDPA
tests use integration-test bundles that carry `ragged_offset`
data, for example
`dnn-providers/integration-tests/integration-test-bundles/quick/SdpaFwd/bshd/bf16/hd128_causal_ragged/`.

#### 4.11.1 Pre-supplied input bundle

`ragged_offset` cannot be meaningfully randomized: its values
must be whole-token origins that give the primary a layout the
container accepts ([§4.5](#45-data-sdk-shared-elements) items 4-5).
The harness therefore accepts a pre-supplied input
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
treats them like any other override. When a ragged tensor's SDPA
node names a `seq_len`, that tensor loads first and binds as the
ragged tensor's length tensor ([§4.5](#45-data-sdk-shared-elements) item 4).

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
  `make_shared<RaggedTensor<T>>(dims, strides, SDPA_SEQ_AXIS,
  raggedOffsetSharedPtr, std::nullopt, attr.ragged_offset_multiplier(),
  lengthsOrNull)`,
  letting the ctor infer the buffer size as `elementSpace`, forwarding
  the primary's multiplier, and binding the node's `seq_len` (or null)
  as the length tensor. The factory does not need to know
  the aux's element type statically — it is single-dispatched on the
  primary's `T` only.

A single-pass walk suffices because every `ragged_offset` aux a
primary could reference is, by the strict check, already held by
the pre-supplied bundle before allocation begins. No auxiliary
pre-pass over `TensorAttributes` is needed.

The buffer is sized to exactly `elementSpace` elements (§4.5 item 6);
a ragged `.bin` file holds exactly that many:

```
physicalElementCount = elementSpace
```

`alignment` does not enter this calculation (see
[§4.5](#45-data-sdk-shared-elements) item 6); `elementCount()`
on the resulting `RaggedTensor` is the number of live elements.

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
  per batch over its live rows (`s < len_b`). Elements outside the
  live rows are never read by GPU kernels or CPU references, so
  they keep their allocation value.
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
produce; for floating-point outputs it is NaN. Golden reference
data marks the in-block pad rows it leaves undefined the same way.

For ragged tensors, the existing `allClose` validators (and the
tensor-diff report) apply these rules; dense tensors behave
exactly as before:

- **Matching ragged layout.** Both tensors must have the same
  ragged layout (equal `raggedIterationInfo()`: origins, lengths,
  sequence axis, and sequence stride). A layout mismatch, including
  ragged vs dense, fails the comparison.
- **Live positions only.** The comparison iterates only
  live logical positions — those whose sequence index is below
  `lengths[b]` (§4.5 item 4). Positions past a batch's length are
  not live: they lie in gap rows, alias another batch's rows or lie
  past the end of the buffer, so they are never read.
- **NaN-sentinel skip.** For floating-point ragged tensors,
  positions where the expected (reference) value is NaN, with any
  payload, are skipped (the reference didn't write there). Other
  positions use the existing tolerance check. A NaN in a dense
  reference is still treated as an unwritten element and fails.
- **Nothing compared is a failure.** If the tensor has in-block
  elements but none was compared (e.g. the reference wrote only
  NaN), the comparison fails. A ragged tensor whose per-batch
  extents are all 0 has no in-block elements and passes.

Skipped positions do not count toward error statistics: the RMS
validator and the diff report use the number of compared elements.
Integer ragged outputs get the layout and in-block rules but no
sentinel skip, so every in-block position is compared (see
[§7.3](#73-improve-cpu-validation-sentinel-value-handling)).

Ragged comparison is host-only. The device-side validators reject
ragged tensors, and a comparison that resolves to the device
validation site reports the ragged output as unsupported there
(rerun with the CPU validator) rather than comparing it.

Sentinel-skip is the only validation mechanism in this design. The
compared rows are the container's live rows (`s < len_b`), which follow
`seq_len` when it is bound ([§4.5](#45-data-sdk-shared-elements) item 4).

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
2. **Iteration visits the live rows `s < len_b` of each batch.**
   Without a length tensor, `len_b` follows the offset differences
   (capped at `S_max`), so per-batch padding tails are visited. With
   one, only `seq_len[b]` rows are. An op that masks on `seq_lens`
   reads it from the variant pack.
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
   types even for packed layouts, where `elementCount() ==
   elementSpace()`, because the buffer is neither `prod(dims)`
   elements long nor regularly strided. This diverges from the
   literal `elementCount == elementSpace` predicate. See
   [§4.5](#45-data-sdk-shared-elements) item 7 and
   [§7](#future-work) for a possible `hasRegularDims()` split.
6. **`ragged_offset` element-type checking is runtime, not
   static.** The aux is held as `shared_ptr<ITensor>` and the
   int32-vs-int64 distinction is resolved by a small element-size
   switch inside the `readIndexTensor` helper
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
8. **Aliasing layouts are rejected.** The container rejects
   overlapping token rows and sorted-stride violations (§4.5.1, L10 and
   L11); zero-stride axes are broadcasts and are accepted. It also
   requires whole-token origins, `0 <= len_b <= S_max` and strides
   `>= 0`. Applicability sees only graph attributes, not offset
   values; layouts that depend on the values are rejected when the
   tensor is built (bundle load), or with `std::invalid_argument` at
   execute for callers passing raw pointers. The cuDNN frontend does
   not validate ragged geometry; the divergence is recorded in
   `SdpaSemanticsRegister.md` SEM-026.
9. **Paged K/V and ragged Max/Sum_exp.** The native ragged forward
   gate declines ragged Q over paged K/V and declines `Max` and
   `Sum_exp`. The dense rewrite (`RaggedSdpaDensify.hpp`) serves
   ragged Q over non-ragged paged K/V and serves ragged `Max` and
   `Sum_exp` (bound by `seq_len_q`, like Stats). It declines a
   `ragged_offset` on a paged K/V container.
10. **INT64 offsets.** The SDPA gate declines INT64 `ragged_offset`
    (the dense rewrite accepts it). The dense rewrite accepts only an
    `INT32` `seq_len` and declines an `INT64` one.
11. **The forward golden generator authors only bshd ragged
    tensors, without Stats.** `generate_sdpa_fwd_golden.py` rejects
    head-major, shifted-origin and padded-THD ragged bundles, and
    ragged bundles with a Stats output. The container, loader and
    references accept those layouts; the backward generator authors
    them.
12. **Without `seq_len`, the `S_max` cap can hide offsets written in
    the wrong units** when no buffer size is known (raw device
    pointers). Wherever a size is known (`physicalElementCount`, a
    bundle file) it must equal the element space, which catches them.

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

**Why this RFC does not take it.** The aux that sizes the buffer is
`ragged_offset` (the optional length tensor of
[§4.5](#45-data-sdk-shared-elements) item 4 is
the second), which must be available at primary-allocation time.
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

### 6.2 Carrying the length tensor on `TensorAttributes`

Rejected: a `ragged_seq_len_tensor_uid` on `TensorAttributes`. It is a
schema and plugin-API change that duplicates the node's `seq_len`; the
node-level mapping in [§4.5](#45-data-sdk-shared-elements) item 4 needs
neither. An ordinary `seq_len` input stays a regular tensor that
references read for masking.

### 6.3 Templating the ragged-tensor types on `IndexT`

An alternative design parameterizes both ragged-tensor
types on the aux element type — `RaggedTensor<T, IndexT>` and
`ShallowRaggedTensor<T, IndexT>` (defaulting to `int32_t`,
permitting `int64_t`). The aux is then held as
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
(shape `[B + 1, 1, 1, 1]`, int32 or int64) bound the surface
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
  inside `readIndexTensor`, paid once per aux element at
  construction.

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
their constructor-time structural and content validation, and
(after the `seq_lens` decoupling) their entire surface apart from
the memory carrier and fill operations. The initial implementation
already factors that shared surface into a common abstract base,
`RaggedTensorBase<T>` (see [§4.6](#46-data-sdk-raggedtensort-owning-ragged-aware)),
so the two concrete types differ only by their memory carrier
(owning `MigratableMemory<T, …>` vs borrowed `void*`) and their
fill ops.

A further step could collapse them into a *single* class template
parameterized on the memory carrier, eliminating the two concrete
classes entirely. That remaining collapse is left as a mechanical
follow-up; the two-concrete-types-over-a-shared-base split kept
here mirrors the existing `Tensor<T>` / `ShallowTensor<T>` split.

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
