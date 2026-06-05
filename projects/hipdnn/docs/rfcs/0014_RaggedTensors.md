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
   - 4.5 [Data SDK: shared elements](#45-data-sdk-shared-elements)
   - 4.6 [Data SDK: `IrregularTensor<T>` (owning storage)](#46-data-sdk-irregulartensort-owning-storage)
   - 4.7 [Data SDK: `RaggedView<T, IndexT>` (non-owning ragged view)](#47-data-sdk-raggedviewt-indext-non-owning-ragged-view)
   - 4.8 [Tiered API: which type each role uses](#48-tiered-api-which-type-each-role-uses)
   - 4.9 [Wiring sources of truth](#49-wiring-sources-of-truth)
   - 4.10 [Plan-layer view construction and CPU reference impact](#410-plan-layer-view-construction-and-cpu-reference-impact)
   - 4.11 [Test-harness integration](#411-test-harness-integration)
5. [Known limitations](#known-limitations)
6. [Alternatives considered](#alternatives-considered)
7. [Future work](#future-work)

---

## Summary

A ragged tensor is logically `[B, X, …]` where the per-batch extent
`X` varies. Physical memory is a single contiguous buffer indexed by
a `ragged_offset[B+1]` aux tensor; an optional `seq_lens[B]` aux
tensor distinguishes valid rows from per-batch trailing padding.

This RFC adds end-to-end ragged-tensor support:

1. **Frontend (`TensorAttributes`)** gains
   `set_ragged_offset` / `get_ragged_offset` and `set_alignment` /
   `get_alignment`.
2. **Flatbuffer schema** gains defaulted `ragged_offset_tensor_uid`
   and `alignment` fields (wire-compatible per RFC 0005).
3. **Backend** propagates the new fields through existing
   get/set-attribute paths; no new C-API entry points; variant pack
   unchanged.
4. **Data SDK** adds `IrregularTensor<T>` (owning storage decoupled
   from `prod(dims)`) and `RaggedView<T, IndexT>` (non-owning ragged
   wrapper). Storage and ragged semantics are kept separate.
5. **Plan layer** assembles a `RaggedView` per execute from
   `TensorAttributes` (`ragged_offset`) and node-level op attributes
   (`seq_lens`), then passes it to the CPU reference as
   `TensorBase<T>&` — reference signatures are unchanged.
6. **Integration-test harness** accepts a *pre-supplied input
   bundle* — a single `unordered_map<int64_t, unique_ptr<ITensor>>`
   keyed by UID — so aux tensors carry deliberate, structurally
   valid values shared between the GPU and CPU paths via each
   `ITensor`'s built-in host/device `MigratableMemory`.

The immediate consumer is SDPA on AITER FMHA kernels (packed batches
addressed by `ragged_offset`, optional per-batch valid lengths).

---

## Problem Statement

### 2.1 What is a ragged tensor

A logically `[B, X, …]` tensor where `X` (typically the sequence
dimension) varies per batch. Batch `b` occupies the contiguous range
`[ragged_offset[b], ragged_offset[b+1])` of the physical buffer.
Optionally, `seq_lens[B]` carries the valid sequence length per
batch when each batch's reserved region may exceed its real valid
extent (e.g. for kernel alignment requirements).

### 2.2 hipDNN gap

hipDNN's tensor model assumes physical element count equals
`prod(dims)`. There is no way to express:

- A primary tensor whose physical buffer is not `prod(dims)`.
- A reference from one tensor (primary) to another (aux).
- Iteration that walks only valid positions of a ragged primary.

Separately, the integration-test harness initializes inputs by
random fill, which is meaningless for ragged-aux tensors: a random
`ragged_offset` will almost never be monotonically non-decreasing
with `ragged_offset[0] == 0`; a random `seq_lens` routinely exceeds
the per-batch reserved extent. The harness needs a way to consume
deliberate values for these tensors.

### 2.3 Configurations targeted in this iteration

Two configurations, both used by AITER FMHA:

- **Packed with `ragged_offset` only.** Physical buffer is exactly
  `ragged_offset[B]` elements; each batch contributes
  `ragged_offset[b+1] - ragged_offset[b]` rows; no internal padding.
- **Packed with `ragged_offset` + `seq_lens`.** Physical buffer is
  the sum of per-batch reserved regions (possibly aligned up); each
  batch's first `seq_lens[b]` rows are valid; trailing rows are
  padding.

A "padded, seq-lens-only" mode (no `ragged_offset`) is intentionally
deferred; see [§7.3](#73-padded-mode-seq-lens-only-no-ragged_offset).

### 2.4 Constraint: `seq_lens` is not on `TensorAttributes`

For cuDNN frontend-API compatibility, **`TensorAttributes` gains
`set_ragged_offset(...)` only — not `set_seq_len(...)`**. The
`seq_lens` tensor is referenced from *node-level* op attributes
(e.g. `SdpaAttributes::set_seq_len_q`). In SDPA a single `seq_lens_q`
is shared across the Q primary and a single `seq_lens_kv` across the
K and V primaries, so there is no 1:1 binding between a primary
`TensorAttributes` and a `seq_lens` tensor.

Consequences:

- Code that walks the graph one `TensorAttributes` at a time can
  discover a tensor's `ragged_offset` but not its `seq_lens`.
- The plan layer (which already has node attributes) is the natural
  place to attach `seq_lens` to a ragged view.
- The test-harness bundle layer holds dense storage only;
  ragged-aware views are assembled at the plan layer.

---

## Existing Infrastructure

### 3.1 Frontend: `TensorAttributes`

`hipdnn_frontend::graph::Tensor_attributes` declares per-tensor
metadata (dims, strides, dtype, UID, virtual flag, alignment, …)
with chainable setters/getters. No concept of per-tensor references
to other tensors exists today.

### 3.2 Flatbuffer: `tensor_attributes.fbs`

Persistent representation of `TensorAttributes` for graph
serialization (and Compiled-Plan Serialization). Schema evolution
rules (RFC 0005) require appending optional defaulted fields.

### 3.3 Backend: tensor descriptor and variant pack

Backend tensor descriptor mirrors `TensorAttributes` and is built
during graph lowering. The variant pack carries `UID → void*`
device-buffer bindings at execute time. Neither layer currently
carries ragged-specific information.

### 3.4 Data SDK: `ITensor` / `TensorBase<T>` / `Tensor<T>` / `ShallowTensor<T>`

Runtime tensor hierarchy in `hipdnn_data_sdk::utilities`:

- **`ITensor`** — type-erased base: `dims()`, `strides()`,
  `elementSpace()`, `elementCount()`, `isPacked()`, `rawHostData()`,
  iteration via `LinearIndex` / `CompositeIndex` strategies.
- **`TensorBase<T>`** — adds typed addressing
  (`getHostValue`/`setHostValue`).
- **`Tensor<T>`** — owning dense tensor backed by
  `MigratableMemory<T, HostAlloc, DeviceAlloc>`. Asserts
  `_packed = (elementCount == elementSpace)`.
- **`PinnedTensor<T>`** — `Tensor<T>` with `PinnedHostAllocator<T>`.
- **`ShallowTensor<T>`** — non-owning wrapper over a borrowed
  `void*` plus dims/strides; dense iteration over `prod(dims)`.

`ITensorIterator` is implemented via a `std::variant` over the known
index strategies.

### 3.5 Test harness: `GraphTensorBundle` and `CpuReferenceGraphExecutor`

`GraphTensorBundle` holds
`unordered_map<int64_t, unique_ptr<ITensor>>` keyed by UID. The
harness walks `TensorAttributes` and calls
`createTensorFromAttribute(attr)` to allocate each entry. Two
bundles (GPU plan, CPU reference) are created with identical random
seeds; outputs are compared element-by-element by `verifyGraph`.

`CpuReferenceGraphExecutor` walks the graph topologically, allocates
virtual intermediates as ordinary `Tensor<T>`, and dispatches each
node through its `<Op>Plan::execute(variantPack)`, which calls into
the CPU reference (e.g.
`CpuFpReferenceSdpa::forward(q, k, v, …)` typed as
`TensorBase<T>&`).

---

## Design

### 4.1 Overview

1. **Frontend / flatbuffer / backend** propagate
   `ragged_offset_tensor_uid` and `alignment` per tensor.
   Declarative only; no new C-API entry points.
2. **Data SDK** introduces `IrregularTensor<T>` (owning storage with
   `physicalElementCount ≠ prod(dims)`) and `RaggedView<T, IndexT>`
   (non-owning ragged wrapper).
3. **`ITensor`** gains a polymorphic index-strategy hook for ragged
   iteration; `TensorBase` gains optional `validSeqLen(b)`.
4. **Plan layer** builds a `RaggedView` per execute from
   `TensorAttributes` (`ragged_offset`) + node-level op attributes
   (`seq_lens`) and passes it to the CPU reference unchanged.
5. **Test harness** accepts a pre-supplied input bundle so aux
   tensors carry deliberate values rather than meaningless random
   fill.

### 4.2 Frontend: `TensorAttributes` additions

Chainable setters/getters for the new fields:

```cpp
auto set_ragged_offset(std::shared_ptr<Tensor_attributes> const& value)
    -> Tensor_attributes&;
auto get_ragged_offset() const -> std::shared_ptr<Tensor_attributes>;

auto set_alignment(int64_t alignmentInElements) -> Tensor_attributes&;
auto get_alignment() const -> int64_t;   // default 1
```

`set_alignment` declares a per-batch alignment of the physical
buffer, in elements. The default `1` (no alignment) matches AITER
FMHA's needs.

**Frontend validation** (in `validate()`):

- The `ragged_offset` aux exists in the graph (by UID), has rank 1,
  and its first dim equals `B + 1` where `B` is the primary's first
  dim.
- `get_alignment() >= 1`.

Per-batch valid-length validation against `seq_lens` is per-op
validation (e.g. `SdpaAttributes::validate()`), per
[§2.4](#24-constraint-seq_lens-is-not-on-tensorattributes).

**Reported `dims()`.** The primary's `dims()[1]` remains the max
padded sequence length (`S_max`), matching overridable-shape
semantics without re-deriving `S_max` from `ragged_offset` at
runtime.

### 4.3 Flatbuffer schema additions

`tensor_attributes.fbs` gains two appended defaulted fields:

```
ragged_offset_tensor_uid: long = null;
alignment:                long = 1;
```

Wire-compatible per RFC 0005. No `seq_lens_tensor_uid` is added
here — the per-op attribute tables (e.g. `SdpaAttributes`) already
carry `seq_len_q_tensor_uid` / `seq_len_kv_tensor_uid` per
[§2.4](#24-constraint-seq_lens-is-not-on-tensorattributes).

### 4.4 Backend wiring

Backend tensor descriptor gains optional `ragged_offset_tensor_uid`
and `alignment` (default 1), exposed through existing
get/set-attribute paths under new enum values in the hipDNN
extension range. No new `hipdnnBackend*` entry points. Variant pack
representation is unchanged: every tensor (ragged primary or aux)
still binds `UID → void*`.

### 4.5 Data SDK: shared elements

1. **Aux tensors are `std::shared_ptr<TensorBase<IndexT>>`**, with
   `IndexT` templated (`int32_t` default, `int64_t` allowed).
2. **Auxiliaries are constructor-only / immutable** on the view.
3. **`dims()[1]` is required to be `S_max`** (the max padded
   sequence length) for the underlying ragged storage. This is a
   convention the user / graph must uphold so the reported geometry
   matches kernel and overridable-shape expectations; the SDK types
   do not derive or verify it. `elementSpace()` equals the physical
   buffer size and is decoupled from `prod(dims)`.
4. **Polymorphic index-strategy hook on `ITensor`**:
   `virtual std::unique_ptr<ITensorIndex> makeIndex(bool isEnd) const`
   so ragged iteration can supply a `RaggedCompositeIndex` walking
   only valid elements. The existing `LinearIndex` / `CompositeIndex`
   strategies become implementations of this interface; non-ragged
   behaviour is unchanged.
5. **Optional `virtual int64_t validSeqLen(int64_t b) const` on
   `TensorBase`** (default `dims()[1]`) so CPU references can bound
   inner loops without `dynamic_cast`.

### 4.6 Data SDK: `IrregularTensor<T>` (owning storage)

Memory-owning buffer whose physical element count is independent of
`prod(dims)`. Used as underlying storage for ragged views and as the
natural type for graph intermediates that don't need iteration.

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

    // dims()         -> paddedDims                    (dims()[1] == S_max)
    // strides()      -> strides
    // elementSpace() -> physicalElementCount
    // elementCount() -> physicalElementCount          (NOT prod(dims))
    // isPacked()     -> false
    //
    // Direct addressing (rawHostData / rawDeviceData) is supported.
    // Iteration walks the physical buffer linearly.

private:
    MigratableMemory<T, HostAlloc, DeviceAlloc> _memory;
    std::vector<int64_t>                        _paddedDims;
    std::vector<int64_t>                        _strides;
    size_t                                      _physicalElementCount;
};
```

Pinned host memory is available via
`IrregularTensor<T, PinnedHostAllocator<T>, …>` — same pattern as
`PinnedTensor<T>`.

**Iteration policy: linear over the physical buffer.** Ragged-aware
iteration is the job of `RaggedView<T>`. Linear iteration on the
storage type is required by the test harness's `randomizeTensor`
and validation call sites, which have only a `TensorAttributes` and
therefore cannot look up `seq_lens` (per
[§2.4](#24-constraint-seq_lens-is-not-on-tensorattributes)) to
construct a `RaggedView`. A linear walk is correct for both call
sites (see [§4.11](#411-test-harness-integration)); the alternative
of throwing is recorded in
[§6.3](#63-throwing-iterators-on-irregulartensort).

### 4.7 Data SDK: `RaggedView<T, IndexT>` (non-owning ragged view)

Non-owning wrapper that turns any `TensorBase<T>`-derived underlying
storage into a fully iterable ragged tensor.

```cpp
template <typename T, typename IndexT = int32_t>
class RaggedView : public TensorBase<T>
{
public:
    // Shape/strides/physicalElementCount come authoritatively from the
    // underlying — not passed in — eliminating "view disagrees with
    // underlying" bugs by construction.
    //
    // Underlying is shared_ptr<TensorBase<T>> so the view can wrap
    // either an owning IrregularTensor<T> (bundle / user case) or a
    // non-owning ShallowTensor<T> over a variant-pack void* (plan
    // case). Widening from IrregularTensor<T> to TensorBase<T>
    // accommodates both without a third tensor type.
    RaggedView(std::shared_ptr<TensorBase<T>>      underlying,
               std::shared_ptr<TensorBase<IndexT>> raggedOffset,        // required
               std::shared_ptr<TensorBase<IndexT>> seqLens = nullptr);  // optional

    // dims()/strides()/elementSpace()/memory() forward to _underlying.
    // elementCount() -> sum-of-seqLens if _seqLens,
    //                   else _underlying->elementSpace()
    // isPacked()     -> false
    // begin/end      -> RaggedCompositeIndex via makeIndex()
    //                   (walks only valid elements)

    bool hasRaggedOffset() const;              // always true
    bool hasSeqLens()      const;
    const TensorBase<IndexT>* raggedOffset() const;
    const TensorBase<IndexT>* seqLens()      const;
    int64_t validSeqLen(int64_t b) const;      // dims()[1] if no seqLens

private:
    std::shared_ptr<TensorBase<T>>      _underlying;
    std::shared_ptr<TensorBase<IndexT>> _raggedOffset;   // non-null
    std::shared_ptr<TensorBase<IndexT>> _seqLens;        // nullable
};
```

**Ctor-time structural validation:**

- `_underlying != nullptr`, `_raggedOffset != nullptr`.
- `_raggedOffset->elementCount() == B + 1` where
  `B = _underlying->dims()[0]`.
- If `_seqLens != nullptr`: `_seqLens->elementCount() == B`.
- `_raggedOffset` and `_seqLens` (if present) have rank 1.

### 4.8 Tiered API: which type each role uses

| Role | Type | Iteration |
|---|---|---|
| Owning storage for a ragged primary (bundle / executor intermediate) | `IrregularTensor<T>` | Linear over physical buffer |
| Ragged I/O at the CPU reference plan layer | `RaggedView<T, IndexT>` over `ShallowTensor<T>` of variant-pack pointer | Full ragged |
| Ragged tensor a user iterates directly | `RaggedView<T, IndexT>` over the owning `IrregularTensor<T>` | Full ragged |
| Aux tensors (`ragged_offset`, `seq_lens`) | Plain `Tensor<IndexT>` via `shared_ptr` | Dense |
| Graph intermediates no one walks | `IrregularTensor<T>` directly | Linear |
| Non-ragged tensors | Plain `Tensor<T>` | Dense |

### 4.9 Wiring sources of truth

- **Graph (`TensorAttributes`)** — `ragged_offset` (pointer to aux
  attrs) and `alignment`.
- **Node-level op attributes** (e.g. `SdpaAttributes`) — `seq_lens`
  UIDs via existing `set_seq_len_q` / `set_seq_len_kv` accessors;
  matching `seq_len_*_tensor_uid` fields in the per-op attribute
  flatbuffer tables.
- **Runtime tensorMap (`UID → ITensor`)** — only dense buffers:
  `IrregularTensor<T>` per ragged primary, `Tensor<IndexT>` per
  aux, `Tensor<T>` per non-ragged. **No ragged-view objects.**
- **Plan layer** — constructs the view at the call site at execute
  time, reading `ragged_offset` from the primary's
  `TensorAttributes` and `seq_lens` from the node's op attribute.

### 4.10 Plan-layer view construction and CPU reference impact

The physical element count needed for the underlying
`ShallowTensor<T>` is computed at execute time from the current
variant-pack `ragged_offset` (see
[§4.11.2](#4112-bundle-allocation) for the formula). Computing it
per execute — rather than caching at plan-build time — avoids
pinning `ragged_offset` values to the lifetime of the compiled
plan:

```cpp
auto qRaggedOffsetView = std::make_shared<ShallowTensor<IndexT>>(/* ... */);

const auto qPhysicalElementCount =
    computePhysicalElementCount(*qRaggedOffsetView,
                                _params.qTensor.alignment);

auto qUnderlying = std::make_shared<ShallowTensor<QType>>(
    variantPack.at(_params.qTensor.uid),
    _params.qTensor.dims,
    _params.qTensor.strides,
    qPhysicalElementCount);

auto qSeqLens = _params.seqLenQTensor.has_value()
    ? std::make_shared<ShallowTensor<IndexT>>(/* ... */)
    : nullptr;

auto qView = std::make_shared<RaggedView<QType, IndexT>>(
    qUnderlying, qRaggedOffsetView, qSeqLens);

// pass *qView as TensorBase<QType>& into CpuFpReferenceSdpa::forward
```

CPU reference signatures are **unchanged**. Body changes are limited
to per-batch sequence-length bounding via `q.validSeqLen(b)`, with
an early-out for `(b, sq)` pairs where `sq >= q.validSeqLen(b)`.
The parallel decomposition continues to use `S_max` so the scheduler
need not know about ragged-ness.

### 4.11 Test-harness integration

Most of the harness flow is unaffected beyond the type substitutions
above. Four stages warrant discussion.

#### 4.11.1 Pre-supplied input bundle

Aux tensors (`ragged_offset`, `seq_lens`) cannot be meaningfully
randomized: their values must be structurally consistent
(`ragged_offset` monotonic with `ragged_offset[0] == 0`;
`seq_lens[b]` within the per-batch reserved extent). The harness
therefore accepts a pre-supplied input bundle at setup:

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

Each pre-supplied entry is an owning `Tensor<T>` (or
`Tensor<IndexT>`) populated with deliberately chosen host-side
values. Only one map is needed: each `ITensor` owns both a host and
a device buffer via `MigratableMemory`, so the same tensor serves
both execution paths — the GPU path reads `rawDeviceData()` (with
migration before kernel launch) and the CPU path reads
`rawHostData()` directly. The harness owns the map; the per-path
`GraphTensorBundle`s continue to own the *non*-pre-supplied tensors.

The harness's UID lookup becomes a pre-supplied-first fallthrough:
pre-supplied bundle, then the relevant `GraphTensorBundle`. When
the per-`TensorAttributes` allocation pass visits a UID present in
the pre-supplied bundle, it skips both allocation and randomization
for that UID in either `GraphTensorBundle`.

**Strict requirement check.** Before the walk, the harness verifies
that every aux UID required by the graph is in the pre-supplied
bundle — every `ragged_offset` referenced by a `TensorAttributes`
and every `seq_lens` referenced by a node-level op attribute.
Missing aux UIDs fail setup with a clear error rather than silently
random-initializing a structurally invalid aux tensor.

**Variant-pack construction.** UIDs resolved from the pre-supplied
bundle contribute the *same* `void*` to both bundles' variant packs
(via the same `ITensor`), automatically guaranteeing byte-equal
input values across the GPU and CPU paths.

**General override semantics.** The pre-supplied bundle is not
restricted to aux UIDs: a test author may supply non-aux inputs to
pin them to fixed values (useful for reproducing a failing case or
for golden-value testing). The strict check rejects only *missing*
aux entries; *extra* non-aux entries are accepted.

The mechanism is harness-layer only — the SDK types, plan layer,
CPU references, and variant pack are unaffected.

#### 4.11.2 Bundle allocation in `createTensorFromAttribute`

For each `TensorAttributes`:

- If the UID is in the pre-supplied bundle: skip.
- Else if `attr.has_ragged_offset() == false`: allocate
  `Tensor<T>` sized by `prod(dims)` as today.
- Else: look up the `ragged_offset` UID in the pre-supplied bundle
  (the strict check guarantees presence), read its host values,
  compute the physical element count, and allocate
  `make_unique<IrregularTensor<T>>(dims, strides, physicalElementCount)`.

The physical element count is:

```
physicalElementCount = align_up(ragged_offset[B], attr.get_alignment())
```

That is, `ragged_offset` already encodes any per-batch reserved
extents (per [§2.3](#23-configurations-targeted-in-this-iteration));
`alignment` only rounds the total buffer size up to satisfy a
trailing-alignment requirement. For `alignment = 1` this reduces to
`ragged_offset[B]` (packed valid size). The plan layer re-evaluates
this expression at execute time against the current variant-pack
`ragged_offset` (see
[§4.10](#410-plan-layer-view-construction-and-cpu-reference-impact))
rather than caching the value at plan-build time.

No structural change to `GraphTensorBundle`: storage stays
`unordered_map<int64_t, unique_ptr<ITensor>>`. Because the view is
not built here, the bundle does not need to know `seq_lens` to
allocate the primary.

#### 4.11.3 Input randomization at bundle init

UIDs in the pre-supplied bundle are skipped. For non-pre-supplied
ragged primaries, `bundle.randomizeTensor(uid)` walks the underlying
`IrregularTensor<T>` linearly — filling more elements than strictly
valid (per-batch padding tails are randomized too). This is
harmless: neither GPU kernels nor CPU references read padding
positions, and identical seeds keep padding values byte-equal
between the two paths. This is the principal reason
`IrregularTensor`'s iterators walk the physical buffer rather than
throwing (see [§6.3](#63-throwing-iterators-on-irregulartensort)).

#### 4.11.4 Output validation in `verifyGraph`

The CPU reference writes only valid sequence positions of an output,
leaving padding at its randomize-time value. The GPU kernel has no
contractual guarantee about output padding — AITER FMHA forward in
particular may leave it alone, zero it out, or write garbage. A
naive full-buffer compare therefore fails for reasons unrelated to
algorithmic correctness.

**Sentinel-skip.** At harness init, fill each ragged output's
physical buffer (both bundles) with a sentinel the CPU reference
cannot legitimately produce (NaN for floats; a magic number for
ints) instead of randomizing. After execution the validator
iterates the full physical buffer; positions where the CPU side is
bit-equal to the sentinel are skipped (the reference didn't write
there, so it's padding). Other positions use the existing tolerance
check. This needs one validator overload
(`allCloseSkipSentinel(cpu, gpu, sentinel, tolerance)`) and a small
init-time branch — no node-attribute walk, no per-output
`RaggedView` construction.

An alternative seq_lens-aware approach was considered but rejected;
see [§6.6](#66-seq_lens-aware-output-validation).

---

## Known limitations

1. **Packed-only.** Only the two AITER-driven configurations are in
   scope. A "padded, seq-lens-only" mode is not supported by
   `RaggedView`'s structural validation; see
   [§7.3](#73-padded-mode-seq-lens-only-no-ragged_offset).
2. **`seq_lens` is not per-tensor.** Per
   [§2.4](#24-constraint-seq_lens-is-not-on-tensorattributes), code
   that walks the graph one `TensorAttributes` at a time can
   identify ragged primaries and their `ragged_offset` but not their
   `seq_lens`.
3. **`IrregularTensor<T>`'s `getHostValue` / `setHostValue` are
   semantically inert.** They are inherited from `TensorBase<T>` but
   have no meaningful per-batch addressing on a ragged primary
   (per-batch addressing requires the `ragged_offset` math living in
   `RaggedView`). As an interim measure they are overridden to throw
   `std::logic_error` directing the caller at `RaggedView<T>`; the
   long-term fix is the hierarchy refactor in
   [§7.1](#71-typedtensort-intermediate-in-the-tensor-hierarchy),
   which removes these methods from `IrregularTensor`'s interface
   entirely.
4. **Sentinel-skip validation requires per-CPU-reference
   guarantees.** The sentinel must be unproducible by the reference
   under test. NaN is safe for floats by default, but edge cases
   (e.g. SDPA with `seq_lens_q[b] == 0` producing NaN from a fully
   masked row) may require a non-NaN magic-number sentinel.
   Low-precision floats (bf16, fp16, fp8) have a smaller sentinel
   space and harder unproducibility guarantees.
5. **`elementCount()` semantics differ between the two new types.**
   `RaggedView` (without `seq_lens`) returns
   `_underlying->elementSpace()`; `IrregularTensor` returns
   `physicalElementCount`. Callers must be aware of the
   "valid elements" vs "physical buffer" distinction.
6. **Test authors must construct aux tensors by hand.** Every
   integration test of a ragged-consuming op must build deliberate
   `ragged_offset` (and optionally `seq_lens`) tensors and supply
   them via the pre-supplied input bundle. No harness-level helper
   for "a plausible random ragged layout" is in this RFC; that
   would require a generator with its own structural-validity rules
   and is deferred.

---

## Alternatives considered

### 6.1 Self-owning `RaggedTensor<T, IndexT>` subclass

An alternate shape: a `RaggedTensor<T, IndexT> : TensorBase<T>` that
owns both its physical buffer and its aux refs, constructed with all
of these passed at once. The user / bundle would hold a single
owning value instead of an `(IrregularTensor, RaggedView)` pair.

This is not actually a one-type design: the plan layer at execute
time has only a `void*` from the variant pack — there is no owning
`RaggedTensor` to hand to the reference, and `ShallowTensor<T>` is
dense-only. A non-owning peer `ShallowRaggedTensor<T, IndexT>` would
be required, bringing the total new types to two — the same as the
recommended design's `IrregularTensor` + `RaggedView`. The
alternative of widening every CPU reference's signature to take aux
pointers + dims separately was rejected because it forces every
reference to take ragged-specific parameters regardless of whether
the op is ragged.

**Pros relative to recommended.**

- Single owning object at the user-facing (samples) layer.
- Pinned host memory expressible via the same template parameters
  pattern as `PinnedTensor<T>`.

**Cons relative to recommended.**

- Same new-type count (`RaggedTensor` + `ShallowRaggedTensor`).
- **Bundle-side ctor-time aux dependency.** Allocating the primary
  via this ctor requires threading the aux `shared_ptr` in at
  construction, which forces `GraphTensorBundle` to shift
  `unique_ptr → shared_ptr` (or expose an aliasing `share()`) and
  constrains visit order. The recommended design avoids this — the
  bundle simply reads the pre-supplied `ragged_offset`'s host values
  to size each `IrregularTensor<T>`.
- **Awkward fit for intermediates no one iterates.** Allocating
  these as "no-aux `RaggedTensor`" is semantically odd; introducing
  a separate type ad hoc pushes the type count higher.
- **Output validation requires new infrastructure on the class.**
  Attaching `seq_lens` post-construction (for seq_lens-aware
  compare) needs either a second ctor, a mutator (violating
  immutability), or a wider `allClose` signature. Not needed in the
  recommended design — `RaggedView` already accepts `seq_lens` at
  construction. (This cost vanishes for both designs under
  sentinel-skip.)
- **Aux relationship is declared twice** (on the graph and on the
  runtime ctor), requiring a reconciler in any flow that surfaces
  both declarations.

The deciding factor in favour of the recommended split is the
bundle / intermediate allocation simplicity, which holds
unconditionally. The single-object user-facing ergonomic can be
partly recovered by a runtime-side helper (see
[§7.2](#72-runtime-side-tensorbundle-helper)).

### 6.2 Wrapper class around an existing dense `Tensor<T>`

An earlier sketch proposed wrapping an existing dense `Tensor<T>`
and overriding indexing/iterators with the aux data. Rejected
because `Tensor<T>` asserts `_packed = (elementCount == elementSpace)`
and is sized by `prod(dims)`. Either way:

- Allocate `prod(dims) = B * S_max * …` (much larger than what
  kernels want), or
- Bypass the dense allocation entirely (in which case the wrapper
  is constructing the storage and `Tensor<T>` adds nothing).

`IrregularTensor<T>` is the minimal storage type supporting
`physicalElementCount ≠ prod(dims)`; the wrapper-over-`Tensor<T>`
approach re-derives that type under a different name.

### 6.3 Throwing iterators on `IrregularTensor<T>`

The discarded variant had `IrregularTensor<T>`'s iterators throw to
force every iteration site to first wrap the storage in a
`RaggedView<T>`. Rejected once the `seq_lens`-on-node-attributes
constraint became firm: the harness's `randomizeTensor` and
validation sites cannot look up `seq_lens` from a
`TensorAttributes` alone, so they cannot construct the required
`RaggedView`. Forcing them to walk node-level op attributes purely
to enable iteration is unnecessary — a linear walk over the
physical buffer is correct for both sites (see
[§4.11.3](#4113-input-randomization-at-bundle-init) and
[§4.11.4](#4114-output-validation-in-verifygraph)).

### 6.4 Adding `seq_lens` to `TensorAttributes`

A symmetry-oriented design putting `set_seq_len(...)` alongside
`set_ragged_offset(...)`. Rejected for two reasons:

- **cuDNN frontend-API compatibility.** Existing cuDNN frontend
  consumers configure `seq_lens` on op-level attributes; mirroring
  it onto `TensorAttributes` would create two ways to express the
  same thing.
- **Many-to-one binding.** A single `seq_lens_q` is shared across
  the Q primary and a single `seq_lens_kv` across K and V. Putting
  `seq_lens` on `TensorAttributes` either duplicates the
  `shared_ptr` across multiple tensors or invents a per-op binding
  rule the node-level location expresses naturally.

### 6.5 Random initialization of aux tensors with structural fixup

An earlier formulation had aux tensors randomly initialized then
"fixed up" — sort `ragged_offset` for monotonicity, clamp
`seq_lens[b]` to the per-batch extent, etc. — so setup needed no
extra arguments from the test author. Rejected for three reasons:

- **Loss of test determinism in a non-obvious way.** Small changes
  to seed handling or to the fixup heuristic silently change every
  test's per-batch sequence lengths, which can mask or surface bugs
  unrelated to the change.
- **Fixup heuristics encode implicit policy** — how much padding
  per batch, what fraction of `S_max` `seq_lens[b]` should average,
  whether degenerate cases like `seq_lens[b] == 0` are allowed.
  These are real decisions invisible to the test author.
- **No actual saving of work.** The author still has to reason
  about which sequence-length distribution to exercise; explicit
  aux values just make that reasoning visible at the test site.

A future helper generating plausible random ragged layouts on top
of the pre-supplied bundle interface is reasonable (Known
limitation #6); the harness should not silently apply one without
the author asking.

### 6.6 seq_lens-aware output validation

An alternative to the sentinel-skip output-validation approach
([§4.11.4](#4114-output-validation-in-verifygraph)) is to have the
harness walk node-level op attributes to find each ragged output's
`seq_lens` UID, look up the corresponding aux in the pre-supplied
bundle (or in the per-bundle `GraphTensorBundle`), and construct a
`RaggedView` over each output's `IrregularTensor` with `seq_lens`
attached. The validator would then iterate only the valid elements
via the ragged-aware iterator.

Rejected for two reasons:

- **Node-attribute walk at the validation site.** The bundle layer
  only sees `TensorAttributes`; discovering each output's
  `seq_lens` requires plumbing node-level attribute information
  into a layer that does not otherwise need it.
- **Per-output `RaggedView` construction at validation.** The
  validator must construct a (transient) `RaggedView` for each
  ragged output, mirroring work the plan layer already does at
  execute time — duplicated machinery purely for compare.

Sentinel-skip avoids both of these and needs only one validator
overload and an init-time fill. The seq_lens-aware wrap remains a
viable fallback if a future op turns out to have a CPU reference
that can naturally produce any reasonable sentinel value, in which
case `RaggedView` already accepts `seq_lens` at construction so the
machinery exists.

---

## Future work

### 7.1 `TypedTensor<T>` intermediate in the tensor hierarchy

`IrregularTensor<T>` currently derives from `TensorBase<T>`, which
carries the `getHostValue` / `setHostValue` multi-dim addressing
API. Those methods have no meaningful semantics on a ragged
primary's physical buffer (the strides describe padded geometry;
per-batch addressing requires the `ragged_offset` math living in
`RaggedView`). The type system does not currently enforce this.

A cleaner end state introduces `TypedTensor<T>` between `ITensor`
and `TensorBase<T>`:

```
ITensor                              // type-erased
   ↑
TypedTensor<T>                       // typed rawHostData() -> T*,
   ↑                                 // memory() -> MigratableMemory<T>&,
   |                                 // typed fill / sentinel ops
   +-- IrregularTensor<T>            // owning, no multi-dim indexing
   |
   +-- TensorBase<T>                 // adds getHostValue / setHostValue
          ↑
          +-- Tensor<T>, RaggedView<T>, ShallowTensor<T>, …
```

`RaggedView`'s underlying re-types as
`shared_ptr<TypedTensor<T>>` — accepting both the owning
`IrregularTensor<T>` and the non-owning `ShallowTensor<T>` at a
narrower interface that drops the meaningless multi-dim API.
Nothing changes for callers of `TensorBase<T>`. Worth doing on its
own merits rather than as part of the ragged-tensors patch.

### 7.2 Runtime-side `TensorBundle` helper

A `UID → owning_runtime_tensor` factory on the user side that for
non-ragged attrs builds a `Tensor<T>`, and for ragged attrs handles
the pattern of constructing `ragged_offset`, allocating an
`IrregularTensor<T>` sized from `(ragged_offset, alignment)`, and
optionally resolving a `RaggedView` against node attributes. This
collapses the user's API to one call per `TensorAttributes` and
hides the two-object cost of this design.

A companion harness-side helper could generate plausible random
ragged layouts and emit them as pre-supplied input bundles
(reducing boilerplate for conformance tests where the specific
layout doesn't matter as long as it is well-formed).

### 7.3 Padded mode (seq-lens only, no `ragged_offset`)

A "padded, seq-lens-only" mode (physical buffer is
`prod(dims) = B * S_max * …`, only `seq_lens` distinguishes valid
from padded) is straightforward to add later because
`RaggedView`'s underlying is already `shared_ptr<TensorBase<T>>` —
pass a `Tensor<T>` underlying with a null `ragged_offset` and relax
the structural validation. Non-breaking against the API above.

### 7.4 Sentinel selection per CPU reference

Document the sentinel choice and "the reference cannot produce this
value" guarantee in each CPU reference header. NaN is the safe
default for floats, but SDPA forward with `seq_lens_q[b] == 0` can
produce NaN naturally (`log(0)` from a fully masked row), requiring
a non-NaN magic-number sentinel. Low-precision floats (bf16, fp16,
fp8) have a smaller sentinel space and harder unproducibility
guarantees. A future RFC could codify a sentinel-per-dtype-per-op
table.


### Discussion

A significant factor in opting for the non-owning `RaggedTensorView` over the owning `RaggedTensor` is the fact that the connection between the optional `seq_lens` tensor and the ragged tensor is difficult to ascertain. However, it should be noted that in the context of the the `IntegrationGraphVerificationHarness`, we are _not_ considering the `seq_lens` tensor until we reach the stage of executing the Sdpa node. If we decide that it is okay to iterate over the padded elements, and have the `RaggedTensor` only consider the `ragged_offset` (ignoring the `seq_lens`), we lose the ability to distinguish in isolation which elements are padded, but the tensor can still always guarantee that an index maps to the correct spot in memory. With this change, the design would become simpler, and `RaggedTensor` would likely become the top choice.
