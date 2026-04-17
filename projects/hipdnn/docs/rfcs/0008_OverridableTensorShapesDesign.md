# hipDNN - Overridable Tensor Shapes Design Document

- Contributors: Brian Harrison
- **Status**: Draft

> [!NOTE]
> This document specifies plumbing-only changes; initial rollout
> produces no reference engine implementation.

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Current System Overview](#3-current-system-overview)
4. [Proposed Design](#4-proposed-design)
5. [Key Design Decisions](#5-key-design-decisions)
6. [Phase 2: Dynamic Tensors](#6-phase-2-dynamic-tensors)
   - 6.1 [Why Phase 2](#61-why-phase-2)
   - 6.2 [Goal](#62-goal)
   - 6.3 [Per-tensor `is_dynamic` flag (FBS, backend, frontend)](#63-per-tensor-is_dynamic-flag-fbs-backend-frontend)
   - 6.4 [Dynamic-tensor semantics (wildcard dims and stride-as-order)](#64-dynamic-tensor-semantics-wildcard-dims-and-stride-as-order)
   - 6.5 [Auto-flagging behavior](#65-auto-flagging-behavior)
   - 6.6 [Frontend validation relaxation](#66-frontend-validation-relaxation)
   - 6.7 [IsApplicable improvements](#67-isapplicable-improvements)
   - 6.8 [Major version bump strategy](#68-major-version-bump-strategy)
   - 6.9 [Migration and rollout](#69-migration-and-rollout)
   - 6.10 [Reference-surface alignment](#610-reference-surface-alignment)
   - 6.11 [Phase 2 key design decisions](#611-phase-2-key-design-decisions)
7. [Risks](#7-risks)
8. [Execution Plan](#8-execution-plan)
9. [Testing Plan](#9-testing-plan)
10. [Future Considerations](#10-future-considerations)
11. [Glossary](#11-glossary)

---

## 1. Executive Summary

This RFC proposes adding **overridable tensor shapes** and **dynamic
tensors** to hipDNN as two committed phases shipped under this single
RFC. Phase 1 introduces a graph-compile-time flag declaring max-shape
semantics plus an execute-time mechanism that re-supplies per-tensor
dims and strides without rebuilding the graph. Phase 2 builds on that
transport with a per-tensor `is_dynamic` flag that lets a graph
declare wildcard dims and stride ordering at build time, so that
applicability and validation decisions are settled before the first
execute call. The immediate consumer is scaled dot-product attention
(SDPA), which needs to serve many sequence-length variants from a
single compiled plan.

Initial rollout is plumbing-only: frontend, backend descriptors,
plugin SDK, and host-side dispatch all gain override-aware
machinery, but **no reference engine implementation lands**. The
new public surface is gated behind whatever build-define the SDPA
team introduces; this RFC introduces no new env var, CMake option,
or runtime knob of its own. Existing
plugins remain binary-compatible: plugins that do not implement the
new optional plugin SDK entry continue to work unchanged for
non-override graphs and are filtered out for override-enabled graphs.
The public backend C-API surface is preserved unchanged: there is one
`hipdnnBackendExecute` today and one after this RFC lands.

**Binary-compatibility scope.** Phase 1: existing plugins remain
binary-compatible (plugins missing the override entry continue to serve
non-override graphs unchanged). Phase 2: the joint 1.0 SDK bump requires
plugin rebuilds — see §6.9 Layer 1, which filters all pre-1.0 plugins out
of `getApplicableEngineIds` for any 1.0-schema graph, including
non-override graphs.

### 1.1 Phased rollout

Overridable-tensor support is delivered as a two-phase rollout, both
phases committed in this RFC:

- **Phase 1, Overridable tensor shapes (this RFC, §4).** A
  graph-level flag opts a graph into execute-time override
  semantics, and an execute-time mechanism re-supplies per-tensor
  dims and strides without rebuilding the graph. All shapes are
  concrete at graph build time; overrides must fit within the
  declared max-shape.
- **Phase 2, Dynamic tensors (this RFC, §6). Builds on Phase 1's
  variant-pack override transport by adding declarative dynamism at
  graph build time, which enables build-time applicability filtering
  and closes the failure modes Phase 1 alone cannot address. Ships
  under a major version bump; see §6.9 for the rollout.**

The remainder of this RFC (§2 through §5, plus §7 through §10)
covers both phases. §2 through §5 detail Phase 1; §6 details Phase 2;
§7 through §9 cover risks, execution, and testing for both phases
together; §10 enumerates remaining future work beyond what either
phase commits to.

---

## 2. Problem Statement

### 2.1 Reference surface

The desired end-user surface follows a well-established pattern for
dynamic-shape graph runtimes:

- At graph build time, a graph flag (e.g.
  `set_dynamic_shape_enabled(true)`) declares intent to override
  shapes at execute time.
- At execute time, an overload of the graph's execute method accepts
  three parallel vectors of override UIDs, shapes, and strides. The
  graph is not rebuilt; the same compiled plan is reused.
- The single backend execute entry is preserved: there is no sibling
  execute-with-overrides function. The reference surface specifies
  the frontend overload and the single-execute-entry constraint; how
  each runtime carries the override payload to its plugin layer is
  an implementation choice. hipDNN folds the payload into the
  variant pack via dedicated variant-pack attributes (see §4.3).

This RFC adopts the same shape so that user code translating between
graph runtimes does not need per-runtime branching at every call site.
SDPA workloads commonly run on multiple runtimes; divergence forces
them to fork at every override call site.

### 2.2 hipDNN gap

hipDNN bakes per-tensor dims and strides into the operation graph at
compile time. Serving N sequence-length variants in SDPA today
requires N distinct compiled graphs and N cached execution plans. For
the SDPA workloads this is prohibitive: every in-flight
token-length variant forces a fresh graph build and plan
finalization, and plan compilation dominates first-call latency.
Without an override mechanism, hipDNN cannot serve SDPA on a
per-call basis from a single compiled plan.

### 2.3 Constraints

The design must:

1. **Preserve binary compatibility with existing plugins**. Plugins
   that do not adopt the new mechanism must continue to load and
   serve graphs that do not opt in. RFC 0002 commits hipDNN to a
   stable plugin contract; this RFC must extend it without breaking
   that contract.
2. **Preserve the public backend C-API surface**. There is exactly
   one `hipdnnBackendExecute` today, and there will continue to be
   exactly one after this RFC lands. The reference surface (§2.1)
   keeps a single execute entry; this RFC follows that pattern.
3. **Keep the graph descriptor read-only after build**. Per-tensor
   shapes declared at graph time must not be mutated at execute time;
   overrides travel via the variant pack only.
4. **Stay off-by-default for end users** until at least one shipping
   plugin implements the override path. See §4.7 for how this is
   achieved without introducing a new hipDNN-owned gate.

---

## 3. Current System Overview

### 3.1 Graph compile / execute today

```text
                build()                     execute()
 Frontend ───► GraphDescriptor ─► finalize ─► VariantDescriptor
                                                    │
                                                    ▼
                                       hipdnnBackendExecute
                                                    │
                                                    ▼
              EnginePluginResourceManager::executeOpGraph
                                                    │
                                       extract { uids, ptrs }
                                       into hipdnnPluginDeviceBuffer_t[]
                                                    │
                                                    ▼
                  hipdnnEnginePluginExecuteOpGraph (in plugin)
```

`Graph::build()`
(`frontend/include/hipdnn_frontend/Graph.hpp:1578`) materializes a
backend `GraphDescriptor`, finalizes it, and finalizes an execution
plan. `Graph::execute()`
(`frontend/include/hipdnn_frontend/Graph.hpp:1643-1740`, where the
`tensorLookup` overload starts at 1643) translates
the caller's UID-to-pointer map and workspace into a
`VariantDescriptor` and calls `hipdnnBackendExecute`
(`backend/include/hipdnn_backend.h:178`). Frontend types and logging
come from the Data SDK; the frontend never speaks FlatBuffers
directly.

### 3.2 Backend descriptor model

`GraphDescriptor` (`backend/src/descriptors/GraphDescriptor.{hpp,cpp}`)
owns the operation list, finalization state, and the FlatBuffer
serialization round-trip. The relevant attribute plumbing entry
points are `setAttribute` (line 301), `getAttribute` (line 160),
`finalize` (line 26), `buildGraphFromOperations` (line 47), and
`deserializeGraph` (line 348). The serialized schema lives in
`flatbuffers_sdk/schemas/graph.fbs`, in `table Graph` (lines 37–45).
RFC 0005 §4.6.1 establishes that appending optional defaulted fields
to existing tables is forwards- and backwards-compatible within a
major version, which is what this RFC will rely on for the new
graph flag.

### 3.3 Variant pack model

`VariantDescriptor`
(`backend/src/descriptors/VariantDescriptor.hpp:14-43`) is the
runtime-only carrier of per-execution payload. It holds three fields
today: `_dataPointers`, `_uniqueIds`, and `_workspace`. The attribute
IDs already defined on the variant pack live in the **700–799
reserved range**. There is no FlatBuffer schema for the variant
pack: it is **runtime-only by design**, and this RFC does not change
that. Direct inspection confirms there is no
`flatbuffers_sdk/schemas/variant_pack.fbs` file in the tree.

New variant-pack attributes are added by extending `setAttribute` /
`getAttribute` only. This RFC adds three such attributes (see §4.3),
each landing in the existing 700–799 reserved range and following the
same in-process get/set discipline that already exists for variant
pack fields today. The next free slot in that range is **704**, not
703: `HIPDNN_ATTR_VARIANT_PACK_INTERMEDIATES = 702` already exists
in `HipdnnBackendAttributeName.h:222` (although currently unwired in
`VariantDescriptor`), so the §4.3 inventory step starts at 704.

### 3.4 Plugin C-API surface today

The plugin SDK's execute entry is `hipdnnEnginePluginExecuteOpGraph`
in `plugin_sdk/include/hipdnn_plugin_sdk/EnginePluginApi.h:239-244`
with signature:

```c
hipdnnPluginStatus_t
hipdnnEnginePluginExecuteOpGraph(
    hipdnnEnginePluginHandle_t handle,
    hipdnnEnginePluginExecutionContext_t execution_context,
    void* workspace,
    const hipdnnPluginDeviceBuffer_t* device_buffers,
    uint32_t num_device_buffers);
```

A key constraint that shapes the §4.4 plugin SDK extension: **plugins
never see the variant pack**. The host extracts the pack
into flat parallel arrays in
`EnginePluginResourceManager::executeOpGraph`
(`backend/src/plugin/EnginePluginResourceManager.cpp:569-614`),
walking `getTensorIds()` and `getDataPointers()` to pack each pair
into a `hipdnnPluginDeviceBuffer_t` before calling the plugin.
Whatever the plugin needs from the variant pack must reach it through
this flattening step. Any new payload traveling from the variant
pack to the plugin must therefore appear as additional flat
parameters on the plugin entry signature, not as a richer structure
the plugin can introspect.

### 3.5 Plugin versioning today (cf. RFC 0005 §4.6.3–4.6.4)

Plugins report their plugin-SDK API version through the optional
symbol `hipdnnPluginGetApiVersion`
(`plugin_sdk/include/hipdnn_plugin_sdk/PluginApi.h:79`). The backend
uses the `tryAssignSymbol` pattern at the canonical use site
`backend/src/plugin/PluginCore.cpp:35-46`, where it currently resolves
the base-plugin optional symbols (`hipdnnPluginGetApiVersion`,
`hipdnnPluginSetLoggingCallback`, `hipdnnPluginSetLogLevel`); missing
symbols are not load-fatal, and `apiVersion()` defaults to `"0.0.0"`
for plugins that do not export the symbol. Engine entry points in
`backend/src/plugin/EnginePlugin.cpp:29-84` currently use plain
`_lib.getSymbol<>()` rather than `tryAssignSymbol`, so the new
optional engine symbol introduced in §4.4 is the first engine entry
to adopt the optional-symbol pattern. This RFC extends the
symbol-resolution pattern (RFC 0002 precedent) to engine entries for
capability detection; see §4.4 for the per-symbol gating policy.

### 3.6 SDPA surface today

hipDNN's SDPA frontend nodes (`Graph::sdpa()`,
`Graph::sdpa_backward()`, `SdpaFwdNode`, `SdpaBwdNode`,
`SdpaAttributes`, `SdpaBackwardAttributes`,
`SdpaFwdOperationDescriptor`, `SdpaBwdOperationDescriptor`, and
schemas `sdpa_attributes.fbs` and `sdpa_backward_attributes.fbs`)
are all in place. However, **no SDPA feature flag exists**: there is
no env var, no CMake option, and no runtime knob today that scopes
the SDPA work. This RFC explicitly does not introduce one (see §4.7);
the gating discussion is deferred to the SDPA team's RFC.

### 3.7 Already-reserved scaffolding

Two pieces of scaffolding are already in the tree but **not yet
wired**, which materially shapes the design choices in §4 and §5:

- `HIPDNN_ATTR_OPERATIONGRAPH_IS_DYNAMIC_SHAPE_ENABLED = 603` is
  defined in `backend/include/HipdnnBackendAttributeName.h:190` and
  string-mapped at `backend/src/BackendEnumStringUtils.hpp:306-307`,
  but `GraphDescriptor::setAttribute` / `getAttribute` do not handle
  it yet. Reusing this reserved enum value avoids a fresh
  attribute-namespace allocation and a parallel naming debate
  (see §5.2 trade-off).
- The optional-symbol resolution pattern (`tryAssignSymbol`) is
  already in production use for the base-plugin optional symbols at
  `backend/src/plugin/PluginCore.cpp:35-46` (see §3.5). Engine entry
  points in `EnginePlugin.cpp:29-84` use plain `_lib.getSymbol<>()`
  today; this RFC extends the same `tryAssignSymbol` pattern to
  engine entries for the first time, for the new optional plugin SDK
  entry, without introducing any new resolution mechanism.

---

## 4. Proposed Design

### 4.1 Graph flag (covers reqs 1, 4, 16)

Add a graph-level boolean flag `_isDynamicShapeEnabled` to
`GraphAttributes`
(`frontend/include/hipdnn_frontend/attributes/GraphAttributes.hpp:42-170`)
with frontend setter / getter:

```cpp
class Graph {
public:
    Graph& set_dynamic_shape_enabled(bool enabled);
    bool   is_dynamic_shape_enabled() const;
};
```

The attribute is packed into the `GraphDescriptor` via
`GraphPacker`/`GraphUnpacker`
(`frontend/include/hipdnn_frontend/detail/GraphPacker.hpp:22-134`)
and lands on the backend through the already-reserved enum value
`HIPDNN_ATTR_OPERATIONGRAPH_IS_DYNAMIC_SHAPE_ENABLED = 603`. Wiring
extends `GraphDescriptor::setAttribute` (line 301), `getAttribute`
(line 160), `finalize` (line 26), `buildGraphFromOperations` (line
47), and `deserializeGraph` (line 348). `is_dynamic_shape_enabled` is the first
graph-level boolean attribute on `GraphDescriptor`; each of these
touchpoints follows the established attribute-handling pattern
already used for non-bool graph attributes.

The serialized graph schema gains one new field at the end of `table
Graph` in `flatbuffers_sdk/schemas/graph.fbs:37-45`:

```fbs
table Graph {
    // ...existing fields...
    is_dynamic_shape_enabled: bool = false;  // NEW
}
```

Per RFC 0005 §4.6.1, appending an optional defaulted field to an
existing table is forwards-/backwards-compatible within a major
version. No schema-version bump is required, and the already-reserved
enum value 603 means no attribute-namespace conflict can arise either
in the C-API enum range or in the `BackendEnumStringUtils` mapping
table. No `BackendEnumStringUtils` change is needed for 603; per
§3.7, the string mapping for this enum is already in place.

### 4.2 Frontend execute API (covers reqs 2, 3, 9, 15, 17)

`Graph::execute()` gains an overload (the existing signature is
preserved unchanged):

```cpp
error_t execute(hipdnnHandle_t handle,
                std::unordered_map<int64_t, void*>& tensor_uid_to_pointer_map,
                void* workspace,
                std::vector<int64_t> const& override_uids,
                std::vector<std::vector<int64_t>> const& override_shapes,
                std::vector<std::vector<int64_t>> const& override_strides) const;
```

The three trailing parallel vectors mirror the reference surface
(§2.1): positionally indexed, so for each `i`, `override_shapes[i]`
and `override_strides[i]` are the override dims and strides for the
tensor identified by `override_uids[i]`. The frontend does not invent
a struct-based map for this; matching the reference surface is the
constraint (see §5.3 and §5.8). The trade-off is documented in §5.3:
the three vectors must remain length-consistent, validated by the
frontend before any backend call.

**Error semantics** (req 3): the override overload returns a non-OK
`Error` (the frontend's status type; `error_t` at `Error.hpp:179` is a
typedef of `Error`, used interchangeably in code samples below) and
emits a log entry if `is_dynamic_shape_enabled()` was not set on the
graph at compile time. The C-API surface (`hipdnnBackendExecute`)
returns the underlying `hipdnnStatus_t`; the frontend overload returns
`Error` to its caller. The frontend checks this before
making any `setAttribute` calls on the variant pack, so a misuse
never crosses the backend boundary. Setting the flag without
supplying overrides at execute time is permitted (req 9): the graph
still runs through the standard non-override dispatch path, with
identical behavior to a graph that never set the flag.

**Override translation**: the frontend translates the three vectors
into `setAttribute` calls on the `VariantDescriptor`, populating the
five new attributes defined in §4.3 (three payload attributes plus
two per-UID lengths sidebands). The graph descriptor is **not**
mutated at execute time; per-tensor shapes declared at graph build
remain read-only, satisfying constraint 3 from §2.3. After the
attribute writes, the frontend calls the existing
`hipdnnBackendExecute` exactly as today; the override payload reaches
the host through the variant pack.

**Existing path unchanged**: callers that do not use the new overload
see no behavior change. The existing `Graph::execute()` signature
remains the entry point for non-override execution, and the bytes
sent through `hipdnnBackendExecute` for non-override callers are
identical to today.

#### 4.2.1 Override input validation (covers reqs 3, 16; supports glossary §11 max-shape)

Before issuing any `setAttribute` call on the variant pack, the frontend
validates the override payload. Each violation listed below results in a
`HIPDNN_STATUS_BAD_PARAM` return and a log entry; no backend call is
made and no partial variant-pack state is constructed. The frontend's
validation is the **single point of input policing** for the override
surface; the backend assumes the variant pack carries well-formed
override attributes once they are present.

The following violations are checked in order, and the first failure
returns:

1. **Length consistency.** `override_uids.size()`,
   `override_shapes.size()`, and `override_strides.size()` must be
   equal, the central invariant of the parallel-array design (§5.3).
2. **Unknown UID.** Each `override_uids[i]` must identify a tensor
   declared in the graph (present in the graph descriptor's tensor-id
   set built at finalize time). Unknown UIDs are not silently dropped;
   they signal a structural mismatch the caller must resolve.
   Workspace has no UID by construction
   (`VariantDescriptor.hpp:19` carries `_workspace` separately from
   `_uniqueIds`); validation enforces this by checking each
   `override_uids[i]` against the graph's tensor-id set, which never
   contains a workspace entry.
3. **Rank mismatch.** `override_shapes[i].size()` and
   `override_strides[i].size()` must equal the declared rank of the
   tensor identified by `override_uids[i]`.
4. **Max-shape exceeded.** Each `override_shapes[i][d]` must be `<=`
   the declared graph-time dim. The graph-time dim is the
   **max-shape** (see Glossary §11 "Max-shape semantics"), the
   central invariant the engine relies on for buffer sizing.
5. **Duplicate UIDs.** `override_uids` must contain no duplicates;
   each tensor may be overridden at most once per execute call.
6. **Positive dim values.** Each `override_shapes[i][d]` must be
   `> 0`. Zero or negative dims are never legal under max-shape
   semantics.
7. **Positive stride values.** Each `override_strides[i][d]` must be
   `> 0`. The frontend does not bound-check the implied buffer
   footprint; engines must treat strides as plugin-side input that
   may be invalid in pathological cases. The "single point of input
   policing" applies to value-validity only, not to buffer-footprint
   validity.

An override call where all three vectors are empty (and length-
consistent) is equivalent to a non-override execute call; the
frontend skips the variant-pack attribute writes in this case so the
§4.6 dispatch falls through to the existing entry.

This validation runs in the frontend overload, before the
`setAttribute` calls in §4.2 / §4.3. Backend descriptors do not
re-validate; once an override attribute is set on the variant pack it
is treated as authoritative.

### 4.3 Backend C-API and variant-pack attributes (covers reqs 5, 8, 16)

The single existing `hipdnnBackendExecute` is preserved (see §2.1);
override semantics fold into the variant pack via attributes, with
no new public C-API entry exported.

Five new variant-pack attribute enum values are added in the
reserved 700–799 range, starting at the next free slot 704 (see
§3.3) — three payload attributes plus two per-UID lengths sidebands:

```c
HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_UNIQUE_IDS     = 704  // array of int64 UIDs
HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_SHAPES         = 705  // flat int64 per-UID dims
HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_STRIDES        = 706  // flat int64 per-UID strides
HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_SHAPES_LENGTHS = 707  // per-UID rank sideband
HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_STRIDES_LENGTHS = 708 // per-UID rank sideband
```

**Storage convention for jagged arrays.** Attributes 705
(`OVERRIDE_SHAPES`) and 706 (`OVERRIDE_STRIDES`) are each set as a
flat `int64_t[]` buffer (the concatenation of per-UID inner vectors).
The parallel per-UID lengths attributes 707 and 708 (each an
`int64_t[]`) carry the rank of each tensor in the same order as
`OVERRIDE_UNIQUE_IDS`. This **flatten + per-UID lengths** convention
matches the parallel-array form chosen for the plugin SDK signature in
§4.4 (see `override_shapes_lengths` / `override_strides_lengths`),
giving a single representation across both the variant-pack attribute
surface and the plugin entry surface.

These attribute-IDs realize the reference surface's variant-pack
override semantics on the hipDNN side. The reference surface (§2.1)
specifies the frontend overload and the single-execute-entry
constraint; the choice to fold the override payload into dedicated
variant-pack attributes is hipDNN's, motivated by clean separation
from the existing variant-pack fields. End users porting between
runtimes write against the frontend overload, not against these
backend attribute-IDs directly.

The attributes are wired through `VariantDescriptor::setAttribute` /
`getAttribute` only; the variant pack remains runtime-only as
established in §3.3. The implementation adds string mappings for the
new enum values to `backend/src/BackendEnumStringUtils.hpp` alongside
the existing 700-range entries, so attribute names appear correctly
in error messages and logs.

The override propagation path therefore lives in two places: the
variant-pack attributes (frontend to backend descriptor) and the
plugin SDK layer (host to plugin), which are wired together in §4.4
and §4.6. Either layer in isolation is insufficient. The variant
pack carries the payload across the backend boundary, and the plugin
SDK entry conveys the same payload across the plugin boundary, with
the host-side flattening step in §4.6 bridging the two
representations.

### 4.4 Plugin SDK extension (covers reqs 5, 6, 8, 18)

A new optional entry is added to
`plugin_sdk/include/hipdnn_plugin_sdk/EnginePluginApi.h`:

```c
HIPDNN_PLUGIN_NODISCARD HIPDNN_PLUGIN_EXPORT hipdnnPluginStatus_t
hipdnnEnginePluginExecuteOpGraphWithOverrides(
    hipdnnEnginePluginHandle_t              handle,
    hipdnnEnginePluginExecutionContext_t    execution_context,
    void*                                   workspace,
    const hipdnnPluginDeviceBuffer_t*       device_buffers,
    uint32_t                                num_device_buffers,
    const int64_t*                          override_uids,
    uint32_t                                num_override_uids,
    const int64_t* const*                   override_shapes,
    const uint32_t*                         override_shapes_lengths,
    const int64_t* const*                   override_strides,
    const uint32_t*                         override_strides_lengths);
```

The signature extends the existing `hipdnnEnginePluginExecuteOpGraph`
(`EnginePluginApi.h:239-244`) with three additional flat parallel
arrays plus their per-UID length sidebands. The flat-array shape is
dictated by §3.4: plugins never see the variant pack, so the host
extracts the override payload and passes it flat alongside the
existing device-buffer array. For each `i` in `[0, num_override_uids)`,
`override_shapes[i]` is an int64 array of length
`override_shapes_lengths[i]` carrying the override dims for tensor
`override_uids[i]`, and similarly for strides.

The layout mirrors the existing non-override variant-pack API in
`EnginePluginApi.h` so plugin authors can adopt the override entry by
reusing their existing per-UID iteration code. A flattened-with-offsets
alternative was rejected as divergent from the established convention.
This commits the plugin ABI to the parallel-arrays form for the lifetime
of the symbol; a future flattened-layout variant would require a
separately versioned symbol.

**Pointer lifetime.** The `device_buffers`, `override_shapes`, and
`override_strides` pointer-of-pointers (and their inner arrays) are
valid for the duration of this call only; the host guarantees the
underlying storage is not mutated mid-call. The plugin must not
retain or dereference any of these pointers after returning,
matching the existing `hipdnnPluginDeviceBuffer_t[]` contract.

**Versioning.** The implementation bumps
`HIPDNN_PLUGIN_SDK_VERSION_MINOR` in `plugin_sdk/version.h.in` per
RFC 0005 §4.2 (minor bump for backwards-compatible API addition).
Plugins compiled against the older SDK do not export the new symbol
and continue to serve non-override graphs. **Capability detection is
per-symbol via `hasOverrideExecute()`, never per-version-string** (per
RFC 0002, with the `tryAssignSymbol` precedent at
`backend/src/plugin/PluginCore.cpp:35-46`): the symbol either resolves
or it does not. Any version string reported through
`hipdnnPluginGetApiVersion` is for human consumption; a mismatch is a
diagnostics issue, not a correctness issue for the dispatch path.

**Resolution and detection.** The new symbol resolves through the
optional-symbol pattern in `EnginePlugin::resolveSymbols`
(`backend/src/plugin/EnginePlugin.cpp:29-84`) using `tryAssignSymbol`,
the same machinery RFC 0002 already commits to for the base-plugin
optional symbols (see §3.5); no new resolution path is introduced,
but this is the first engine entry to use it. A new predicate
`EnginePlugin::hasOverrideExecute()` returns true iff the symbol
resolved. This predicate is the **authoritative capability gate**
for §4.5 applicability filtering. §4.6 dispatch consults it once
more as a fallback guard for the C-API bypass case (override
attributes constructed on a flag-unset graph via direct backend
C-API misuse). It is consulted nowhere else.

### 4.5 Applicability filtering (covers reqs 7, 10, 11)

Modify `EnginePluginResourceManager::getApplicableEngineIds`
(`backend/src/plugin/EnginePluginResourceManager.cpp:385-423`):

```cpp
if (graphDesc.is_dynamic_shape_enabled()) {
    // Skip plugins that lack the override symbol entirely;
    // do NOT call their isApplicable.
    for (plugin in plugins) {
        if (!plugin.hasOverrideExecute()) continue;
        if (plugin.isApplicable(graph)) result.push_back(plugin.engineId());
    }
} else {
    // Standard path: ask every plugin, regardless of override symbol.
    for (plugin in plugins) {
        if (plugin.isApplicable(graph)) result.push_back(plugin.engineId());
    }
}
```

This satisfies three requirements simultaneously:

- **req 7**: when the flag is set, plugins lacking the override entry
  are never asked about applicability. The optimization is exact:
  the `continue` precedes any `isApplicable` call.
- **req 10**: plugins implementing the override entry are still
  asked about applicability for **all** graphs. The override entry
  itself is only invoked when the variant pack carries override
  attributes (§4.6), so non-override graphs see zero behavioral
  change.
- **req 11**: when the flag is set, plugins lacking the override
  entry are filtered out before engine selection, so they cannot be
  selected even by a stale cached engine-id list.

The filter is the **authoritative pre-dispatch backstop**, layered
on top of §4.4's per-plugin capability gate, for the §4.6 dispatch
switch: as long as §4.5 runs before dispatch and the graph flag has
the same value at applicability time and execute time (which it
does, because the graph descriptor is read-only after build,
constraint 3 of §2.3), the dispatch path can never select a plugin
missing the symbol.

**Invariant**: applicability filtering runs on every `Graph::execute()`
call before host dispatch. The filter is not memoized across calls and
does not rely on any cached engine-id list; re-evaluating cheaply per
execute is the price paid for the §7.7 layered-detection guarantee.
The §4.6 dispatch path is permitted to assume a non-empty result from
§4.5 and a selected plugin that, when the graph flag is set, has
`hasOverrideExecute() == true`.

Plugin authors implementing the override entry must continue to
support non-override graphs in their `isApplicable` implementation;
the override entry is invoked only when the variant pack carries
override attributes, but `isApplicable` is invoked for both.

### 4.6 Host dispatch logic

Modify `EnginePluginResourceManager::executeOpGraph`
(`backend/src/plugin/EnginePluginResourceManager.cpp:569-614`).
After extracting the existing tensor IDs / pointers into
`hipdnnPluginDeviceBuffer_t[]`, inspect the variant pack for the new
`HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_*` attributes:

```cpp
extract { uids, ptrs } -> deviceBuffers[]
extract optional { override_uids, override_dims, override_strides }
  from variantPackDesc

if (override attributes present) {
    // §4.5 guarantees the selected plugin has the symbol when the
    // graph flag is set; this guard catches the bypass case (caller
    // constructed override attributes on a flag-unset graph via direct
    // C-API) before we dereference a null pointer.
    if (!plugin.hasOverrideExecute()) {
        return HIPDNN_STATUS_NOT_SUPPORTED;  // loud, recoverable failure
    }

    plugin.executeOpGraphWithOverrides(
        handle, execContext, workspace,
        deviceBuffers.data(), deviceBuffers.size(),
        override_uids.data(), override_uids.size(),
        override_dims_arr, dim_lens_arr,
        override_strides_arr, stride_lens_arr);
} else {
    // Dispatch to EXISTING entry; flow is identical to today.
    plugin.executeOpGraph(
        handle, execContext, workspace,
        deviceBuffers.data(), deviceBuffers.size());
}
```

This is the **single dispatch switch** that distinguishes override
from non-override execution at the plugin boundary. All
override-aware plugin-dispatch logic lives in this one host-side
extension; no other point in the host needs to branch on
override-attribute presence.

**Host-side conversion lifetime.** The temporary `int64_t*` arrays
passed to the plugin are stack-built `std::vector<const int64_t*>` /
`std::vector<uint32_t>` capturing `.data()` and `.size()` from the
variant pack's `std::vector<std::vector<int64_t>>` attribute storage.
The inner storage lives in the `VariantDescriptor` for the duration
of the execute call (caller-owned, not modified mid-call), so the
captured pointers remain valid for the plugin call. The plugin must
not retain any pointers beyond the call: same contract as the
existing `hipdnnPluginDeviceBuffer_t[]`. The lengths passed to the
plugin entry are `uint32_t` while the frontend's `std::vector` sizes
are `size_t`; the host inserts explicit `static_cast<uint32_t>` with
a runtime size check, matching the `-Wconversion` discipline already
in force.

**Independence from the graph flag.** The dispatch switch keys on
variant-pack-attribute presence, not on the graph flag. A graph
compiled with the flag set may execute with no overrides (for
example, a "default-shape" execution against an override-capable
plan), and dispatch falls through to the existing entry. This
satisfies req 9 and keeps req 8 ("flow is basically identical to
before") true at the plugin boundary: when no overrides are
supplied, an override-implementing plugin sees exactly the same
`hipdnnEnginePluginExecuteOpGraph` call it sees today.

**Edge case: override attributes present but no implementing
plugin.** Cannot occur in well-formed flow; §4.5 removes plugins
lacking the symbol when the graph flag is set. The bypass case is
the caller constructing override attributes on a flag-unset graph
via direct C-API misuse. The explicit `hasOverrideExecute()` guard
in the pseudocode above returns `HIPDNN_STATUS_NOT_SUPPORTED` before
any null function pointer is invoked, ensuring a loud actionable
failure rather than a segfault.

**Plugin authors note.** The non-override `executeOpGraph` entry must
handle the case where the graph descriptor reports
`is_dynamic_shape_enabled() == true` (caller bypassed §4.5 via
direct C-API). Recommended behavior: ignore the flag if no override
variant-pack attributes are present — the dispatch switch (§5.5) keys
on attribute presence, not on the graph flag, so a flag-set / no-
overrides call is well-formed and should be served identically to a
flag-unset call.

### 4.7 SDPA feature-flag mechanism (covers reqs 12, 13, 14)

This RFC **introduces no new env var, CMake option, or runtime
knob**. The SDPA team's build-define gates the override surface
(graph flag setter and the `Graph::execute()` overload). When that
build-define is unset, the new public surface is conditionally
compiled out and inaccessible from end-user code; when the SDPA
team's first plugin implementation lands, the gate is flipped in
the same release.

The "additional flag to disable the public API until an engine
implements override-execute" requirement is satisfied by reusing
the SDPA gate, not by introducing a parallel knob. Divergence risk
from the SDPA team's eventual choice is tracked in §7.5; the gate
name and semantics should be confirmed against the SDPA RFC before
the override plumbing ships. If the SDPA team selects a runtime gate
rather than a build-define, the override surface ships live but
§4.5 applicability filtering plus §7.4 mitigation ensure a clean "no
applicable engines" error response when no implementing plugin is
loaded; the §4.7 wording does not require a build-define
specifically.

---

## 5. Key Design Decisions

### 5.1 Graph flag rather than per-tensor opt-in

**Decision**: a single boolean on the graph gates the entire
override surface. Override-capability is graph-wide; individual
tensors are not annotated.

**Rationale**: SDPA opts in for the whole graph, so a graph-wide
flag is sufficient for the immediate consumer. The reference surface
(§2.1) uses the same graph flag, and matching it reduces
friction for users porting between runtimes. A graph flag also
keeps the attribute-namespace impact minimal: one new graph
attribute (already reserved at 603) versus one per overridable
tensor.

**Trade-off**: a future workload wanting override on only some
tensors must opt the whole graph in, expanding the §4.5 filter
surface and forcing users to reason about override semantics for
tensors they will not vary. §10.1 keeps the door open to a per-tensor
opt-in attribute that would store max-shape metadata on individual
tensor descriptors.

### 5.2 Reuse of `HIPDNN_ATTR_OPERATIONGRAPH_IS_DYNAMIC_SHAPE_ENABLED` (603)

**Decision**: wire the already-reserved enum value 603
(string-mapped at `BackendEnumStringUtils.hpp:306-307`) for the
flag; use the frontend method name `set_dynamic_shape_enabled(bool)`.

**Rationale**: the value is already reserved and the string already
mapped; reusing them avoids both a fresh enum-namespace allocation
and a naming-debate detour that would slow down the RFC. The
frontend method name matches the established pattern users will be
porting from.

**Trade-off**: `dynamic_shape_enabled` is broader than the semantic
this RFC implements. Two distinct semantics could live under that
umbrella: kernel-cache reuse across shape-variant builds (build-time
optimization) and single-plan execute-time override (per-call
payload). This RFC implements the second only; a future expansion to
cache-reuse would land awkwardly on the existing flag. §10.6 leaves
room for a second flag if cache-reuse semantics arrive, but the cost
of choosing differently now is nontrivial because enum 603 already
carries the "dynamic_shape" name.

### 5.3 Parallel-vector override payload

**Decision**: three trailing `std::vector` arguments on the
`execute()` overload (`override_uids`, `override_shapes`,
`override_strides`), indexed positionally so that
`override_shapes[i]` and `override_strides[i]` describe the tensor
named by `override_uids[i]`.

**Rationale**: match the reference surface (§2.1). Cross-runtime-
portable user code is a significant value of matching the
established surface in this RFC. SDPA workloads commonly run on
multiple runtimes, and a divergent hipDNN signature would force
per-runtime branches at every call site.

**Trade-off**: a `std::unordered_map<int64_t, OverrideEntry>` would
be more idiomatic and would eliminate the length-consistency
invariant. Positional indexing across three vectors is error-prone
for users constructing overrides manually; the frontend mitigates by
validating lengths before any `setAttribute` calls, but this is a
runtime check rather than a compile-time guarantee. §10.1 may
revisit this if generalization beyond SDPA also revisits the
frontend ergonomics.

### 5.4 Override transport: variant-pack attributes + plugin SDK entry

**Decision**: keep a single `hipdnnBackendExecute` (see §2.1); add
five new variant-pack attributes (`HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_*` —
three payload + two per-UID lengths sidebands) in the 700–799 range, AND add the optional plugin SDK entry
`hipdnnEnginePluginExecuteOpGraphWithOverrides`. Both surfaces are
required.

**Rationale**: the reference surface (§2.1) constrains the frontend
overload and the single-execute-entry shape; the dedicated variant-
pack attributes are hipDNN's clean realization of that surface, not
a documented reference mechanism. Two alternatives were rejected:
(b) a sibling
`hipdnnBackendExecute_with_overrides` entry diverges from the
reference surface, and (c) a plugin-SDK-only path forces the
frontend to bypass the backend descriptor model, breaking the
layering RFC 0004 establishes. The plugin SDK entry is additionally
required because of the §3.4 flattening constraint: plugins never
see the variant pack.

**Trade-off**: two surfaces must stay in sync across the host's
flattening step. The risk of drift is bounded by the §7.7
layered-detection coupling and by the §9.5 four-corner test matrix.

### 5.5 Host-side dispatch switch

**Decision**: variant-pack override-attribute presence drives
whether the host dispatches to
`hipdnnEnginePluginExecuteOpGraphWithOverrides` (present) or
`hipdnnEnginePluginExecuteOpGraph` (absent).

**Rationale**: data-driven per-call behavior. A graph compiled with
the flag set may still execute the non-override path when no
overrides are supplied at execute time (req 9), and the cleanest
way to express this is to key dispatch on the per-call payload
rather than on the per-build flag. The switch is independent of the
graph flag, simplifying the §9.5 four-corner matrix. The four cases
(flag absent / flag present, crossed with overrides absent /
overrides present) each have a single, well-defined dispatch outcome
that does not require the host to consult both signals
simultaneously.

**Trade-off**: every execute call pays a small attribute-presence
check on the variant pack (a small in-memory map lookup), even for
graphs that never opt in. The alternative (keying dispatch on the
graph flag) was rejected because it would force an override-
implementing plugin to receive the override entry even when the
caller supplied no overrides, violating req 8 for the non-override
case.

### 5.6 Optional plugin symbol vs major version bump

**Decision**: optional symbol with minor-version bump
(`HIPDNN_PLUGIN_SDK_VERSION_MINOR`).

**Rationale**: per RFC 0002, with the `tryAssignSymbol` precedent at
`backend/src/plugin/PluginCore.cpp:35-46`, missing optional symbols
don't break loading; that model is already in production use for the
base-plugin optional symbols. A major-version bump would force every
plugin to recompile against the new SDK before it could be loaded
again, which directly violates req 6 ("existing providers that don't
implement this API continue to work"). A minor bump signals the
backwards-compatible API addition without invalidating any existing
plugin binaries.

**Trade-off**: capability detection is per-symbol (see §4.4); a
plugin that implements the symbol but forgets to bump its reported
version still dispatches correctly; only diagnostics under-report.
This RFC inherits that caveat from the existing optional-symbol
machinery.

### 5.7 SDPA feature-flag mechanism

**Decision**: defer the gate to the SDPA team; reuse their
build-define. This RFC introduces no new env var, CMake option, or
runtime knob.

**Rationale**: avoids a hipDNN-specific gate that would need to be
retired or aligned later. The SDPA work owns end-user surfacing of
the feature; coupling the override-execute gate to the SDPA gate
means there is one knob, owned by one team, that controls
end-to-end visibility.

**Trade-off**: this RFC does not specify the gate name, so
divergence from the SDPA team's eventual choice is possible (§7.5).
The gate name and semantics should be confirmed against the SDPA
RFC; if the SDPA team chooses a build-define semantics that does not
cleanly compile-out the new public surface, additional wiring may be
needed.

### 5.8 Frontend overload rather than separate method

**Decision**: overload `Graph::execute()` rather than add
`Graph::execute_with_overrides()`.

**Rationale**: match the reference surface (§2.1). The reference
uses an overload, not a separate method, and porting code between
runtimes is much smoother when method names match. Precedent in
RFC 0004 §4.4
(`create_execution_plan` is overloaded for the knob-aware variant)
also establishes that frontend overloads are an accepted pattern in
the hipDNN frontend.

**Trade-off**: disambiguation is by argument list, not method name;
unfamiliar readers may miss that a call is the override variant.
Mitigated by the throw-on-misuse semantic in §4.2: a misuse returns
a non-OK error before any backend work, making the failure mode
loud.

### 5.9 No reference implementation in initial rollout

**Decision**: initial implementation ships plumbing only; no
provider adopts the override path in the same drop.

**Rationale**: the SDPA work owns the first real implementation;
plumbing must land first to unblock that work. Splitting the work
this way also lets the plumbing be reviewed and merged on a
schedule independent of provider readiness.

**Trade-off**: design validation comes only from fake plugins
(§9.1); a real engine may surface API gaps that fake plugins miss.
The SDPA gate (§4.7) keeps the public surface dormant until a
shipping provider exists, so users cannot stumble onto a
non-functional API in the interim.

### 5.10 Tensor rank is fixed at build time

**Decision**: tensor rank is declared at graph build time and cannot
vary per execute call. Wildcards (Phase 2, §6) apply to dim *values*
within a fixed rank. Dynamic rank is out of scope for both phases of
this RFC.

**Rationale**: §4.2.1 rule 3 already requires
`override_shapes[i].size() == declared rank`, but that is a validation
mechanic. Stating the commitment explicitly here makes the design
intent unambiguous: the override transport is for value-level
dynamism, not structural dynamism. Fixed-rank semantics also keep the
backend descriptor model unchanged — every existing op validates and
plans against a known operand rank — and bound the §6.6 / §7
discussions to a tractable surface.

**Trade-off**: workloads that need rank to vary per execute (rare in
deep-learning serving, but not unknown) cannot use this transport.
§10 records dynamic rank as an explicit out-of-scope future
consideration.

---

## 6. Phase 2: Dynamic Tensors

This section specifies the **Phase 2** design committed by this RFC.
Phase 2 builds on Phase 1's variant-pack override transport by adding
a per-tensor `is_dynamic` flag declared at graph build time, with
wildcard dim and stride-ordering semantics resolved at execute time
through the Phase 1 transport.

### 6.1 Why Phase 2

Phase 1 keeps a graph's per-tensor shapes concrete at build time and
accepts overrides only at execute time. This has two consequences the
author of an SDPA-style workload feels acutely:

- **Applicability is only decidable at execute time.** Whether a
  graph can actually be served via the override path depends on which
  tensors the user *intends* to vary, which Phase 1 has no way to
  know until the execute call is made. Plugins cannot pre-filter on
  this signal at build time, and a graph that looks override-eligible
  at build can still fail at execute when the user supplies an
  override the engine cannot consume.
- **User intent is implicit and recoverable only by inspection.**
  Phase 1's runtime cannot distinguish "this tensor is intentionally
  dynamic" from "this tensor was concrete and the user happened not
  to override it." The dispatch path treats both identically and
  discovers mismatches late.

Phase 2 closes both gaps (this section) by making **declared intent**
first-class in the schema: tensors marked dynamic are statements
about *what the user will control at execute time and how*. The graph
thereby carries enough information at build time for applicability
checks, plan-cache decisions, and validation policy to be settled
before the first execute call. The execute-time transport itself is
unchanged (Phase 2 reuses the §4.3 / §4.4 surface), but the question
"is this graph executable under override semantics?" becomes
answerable at build, not at run.

### 6.2 Goal

Phase 2 extends Phase 1's override transport with a per-tensor
`is_dynamic` flag that lets a graph declare wildcard dims and stride
ordering at build time, with concrete values resolved at execute time
through the Phase 1 transport. This pushes hipDNN toward a richer
dynamic-shape declaration model than the reference surface (§2.1)
exposes, while keeping the execute-time surface unchanged.

### 6.3 Per-tensor `is_dynamic` flag (FBS, backend, frontend)

The Phase 2 schema, backend descriptor, and frontend wiring add a
per-tensor `is_dynamic` flag that travels alongside the existing
`is_virtual` plumbing.

**FBS schema.** Append `dynamic: bool = false;` to the end of
`table TensorAttributes` in **both**
`data_sdk/schemas/tensor_attributes.fbs` and
`flatbuffers_sdk/schemas/tensor_attributes.fbs`. The bare-noun field
name matches the existing `virtual: bool;` convention at
`tensor_attributes.fbs:55`; the schema layer uses bare nouns while the
frontend layer adds the `is_` / `get_is_` prefixes (see §6.3 Frontend
below). Both schema files must update in lockstep — other paired
schema files in those trees are unaffected by this change. Per RFC 0005
§4.6.1, appending a defaulted bool to the table is wire-compatible, so
a Phase 1 graph deserialized in a Phase 2 runtime sees `dynamic == false`
on every tensor.

**Backend enum.** Add `HIPDNN_ATTR_TENSOR_IS_DYNAMIC = 1308` to
`backend/include/HipdnnBackendAttributeName.h`. Slot 1308 is the next
free entry in the 1300-range tensor-attribute block: 1305
(`IS_VIRTUAL`), 1306 (`VALUE_EXT`), and 1307 (`IS_BY_VALUE`) are
already allocated. Add the corresponding string mapping next to
`IS_VIRTUAL`'s in `BackendEnumStringUtils.hpp`. Wire `setAttribute`
and `getAttribute` for the new enum value in `TensorDescriptor.cpp`,
templated on the existing `is_virtual` plumbing.

`GraphDescriptor::setAttribute`/`getAttribute` does not currently
handle a `bool` attribute. Phase 1 introduces the first one (the
graph-level dynamic-shape-enabled flag, §4.1); the implementation
should add a typed helper following the existing per-type helper
pattern (e.g., `getString`, `getOptionalScalar`) so this
per-tensor `bool` and any future bool attributes reuse it.

**Frontend.** Add an `_isDynamic` member next to `_isVirtual` in
`TensorAttributes.hpp`. Add a `set_is_dynamic(bool)` setter and a
`get_is_dynamic()` getter, templated directly on `set_is_virtual` /
`get_is_virtual` (see `TensorAttributes.hpp:223` for the established
`get_` prefix convention). Add a chaining helper `mark_dynamic()` for
ergonomics so callers can write `graph.tensor(...)->mark_dynamic()`.
Pack and unpack the new flag via the same `DescriptorHelpers` and
`DescriptorUnpackHelpers` thread already used for `is_virtual`.

### 6.4 Dynamic-tensor semantics (wildcard dims and stride-as-order)

A **dynamic tensor** is a tensor declared with `is_dynamic == true`
at graph build time. Two semantic shifts apply, both scoped to
dynamic tensors only:

- **Wildcard dims.** A `dims[d] == -1` entry indicates a wildcard.
  The frontend exposes a constant `TensorAttributes::DYNAMIC_DIM` (= -1)
  so callers do not hard-code the sentinel. The constant lives on
  `TensorAttributes` (no `class Tensor` exists in
  `frontend/include/hipdnn_frontend/`; `TensorAttributes` is the
  natural home, alongside the `get_is_dynamic` peer member). Wildcard
  positions are declared at build; concrete values are supplied at
  execute time via the Phase 1 override transport (§4.2).
- **Stride-as-order.** When `is_dynamic == true`, the `strides`
  field is interpreted as **stride-order**: an axis-permutation in
  which lower index means inner (tighter packing) and higher index
  means outer. This reuses the existing helper
  `generateStrides(dim, strideOrder)` in
  `data_sdk/include/hipdnn_data_sdk/utilities/ShapeUtilities.hpp`
  and its inverse `extractStrideOrder(...)` in the same file. The
  runtime resolves actual element-strides at execute time from the
  resolved dims combined with the declared ordering.

Static tensors (`is_dynamic == false`) retain Phase 1 semantics
unchanged: `dims` are concrete positive values and `strides` are
explicit element-strides.

**Constraint.** A tensor with `is_dynamic == false` must not have
`-1` in any `dims[d]` and must have explicit element-strides. The
frontend enforces this at build time and rejects violations with a
non-OK status before the descriptor is finalized.

**Scalar tensors.** 0-D (scalar) tensors cannot be marked
`is_dynamic`. Build-time validation rejects this: dim/stride wildcards
have no meaning for empty `dims`/`strides` vectors.

**Illustrative example.** A 4-D input tensor `X` with concrete batch
and channel dims, wildcard spatial dims, and NCHW stride ordering is
declared as:

```yaml
X:
  is_dynamic = true
  dims       = { 1, 4, -1, -1 }   # N=1, C=4, H/W deferred
  strides    = { 3, 2, 1, 0 }     # axis ordering: N outermost, W innermost
```

The `strides` entries here are axis indices, not element-strides; the
runtime resolves element-strides from the dims supplied at execute
time combined with the declared ordering.

### 6.5 Auto-flagging behavior

During `lowerGraphToDescriptors()` (called from
`build_operation_graph_via_descriptors()` → `Graph::build()`), the
frontend scans the graph's tensor set; if any tensor has
`is_dynamic == true`, the graph's dynamic-shape-enabled flag (Phase 1
§4.1, enum 603) is auto-set before backend finalization. Users
authoring graphs through the Phase 2 dynamic-tensor API do not need
to call `set_dynamic_shape_enabled(true)` on the graph separately.

`is_dynamic_shape_enabled()` returns true only after
`Graph::build()` completes; before build, the auto-flag is unset.

The auto-set is **idempotent** with explicit
`set_dynamic_shape_enabled(true)` (Phase 1 path); a graph that uses
the explicit setter and also has dynamic tensors observes no
double-set effect.

The auto-set is **observable**: `is_dynamic_shape_enabled()` returns
true after `build()` regardless of which path set it. This keeps the
flag's runtime semantics single-valued and consistent across
inspection points.

### 6.6 Frontend validation relaxation

For dynamic tensors, the frontend's compile-time shape-validation
checks are relaxed by class. The boundary is bound to operand-level
properties so each per-op `validate()` carries a small, explicit
gate.

**Skipped for dynamic operands.** Any check whose evaluation requires
concrete dim values is skipped when at least one operand is dynamic.
Examples include:

- SDPA head-dim divisibility.
- Convolution input-and-filter spatial-dim relationships.
- Matmul inner-dim compatibility.
- Broadcast shape inference.

**Still enforced regardless of dynamic operands.**

- Rank consistency across operands of the same op.
- Dtype consistency and per-operation dtype constraints.
- Axis-ordering consistency across operands of the same op.
- Presence and absence of required tensor inputs for the op.

**Implementation boundary.** Each op's frontend `validate()` method
gains an "any-dynamic-operand" early-return for value-dependent
checks. The early-return is per-op and is added in the same change
that wires Phase 2 for that op; the per-op enumeration of which
checks are value-dependent lives in §8 and is not part of this RFC's
prose.

**Broadcast resolution at execute time.** When operands have wildcards
in the broadcast dim, the resolved override values are checked for
broadcast compatibility per the standard rules (equal, or one is 1).
Engines that do not support broadcasting can detect this via
`get_wildcard_axes()` overlap on broadcast inputs and report `not
applicable`. This partially closes the §6.1 applicability-decidability
gap for broadcast ops, but it is only a partial close: engines still
cannot express min/max ranges or correlated wildcards (deferred to
§10).

### 6.7 IsApplicable improvements

With per-tensor `is_dynamic` declared at build time, the serialized
FB Graph passed to `hipdnnEnginePluginGetApplicableEngineIds` carries
enough information for plugins to pre-filter at applicability time,
before the first execute call.

**data_sdk graph-handler helper queries.** The data_sdk graph-handler
interface gains three helper queries that plugins use inside their
`isApplicable`:

- `has_dynamic_tensors()` returns whether any tensor in the graph
  has `is_dynamic == true`.
- `tensor(uid).get_is_dynamic()` returns the per-tensor flag.
- `tensor(uid).get_wildcard_axes()` returns the axis indices of `-1`
  dims on the named tensor (an empty list for static tensors and
  for dynamic tensors with no wildcards).

**How plugins use them.** Plugins implementing the override entry
declare per-engine support inside `isApplicable`:

- An engine that supports only spatial wildcards rejects graphs
  with batch wildcards.
- An engine requiring NHWC stride ordering rejects graphs declaring
  NCHW.
- An engine that does not support dynamic tensors at all rejects
  any graph with `has_dynamic_tensors() == true`.

This closes the §6.1 "applicability decidable only at execute time"
gap: graphs that look override-eligible structurally but would fail
at execute time on a specific override now fail cleanly at
applicability time instead.

**Helper insufficiency.** Phase 2 helpers (`has_dynamic_tensors`,
`get_is_dynamic`, `get_wildcard_axes`) expose the *presence* and
*axes* of wildcards. They do not expose engine-side constraints
(min/max ranges per axis, alignment, correlated-wildcard pairs).
Engines must either over-accept and report `not applicable` at execute
time, or reject any wildcard graph in `isApplicable`. Richer
constraint expression is deferred to §10.

### 6.8 Major version bump strategy

Both `plugin_sdk` and `data_sdk` bump from `0.0.1` to `1.0.0` as part
of the Phase 2 ship. Files updated in lockstep: `plugin_sdk/version.json`,
`data_sdk/version.json`, `plugin_sdk/version.h.in`,
`data_sdk/version.h.in`.

Per RFC 0005 §4.2 (public-dependency rule), the `data_sdk` 1.0 bump
transitively forces the same major bump on `backend` and `frontend`.
All four `version.json` files (`plugin_sdk`, `data_sdk`, `backend`,
`frontend`) bump together.

**Why a major bump given the FBS change is wire-compatible.** Per RFC
0005 §4.6.1, appending a defaulted bool to `table TensorAttributes`
is wire-compatible. The bump is required because the **semantics** of
two existing fields (`dims`, `strides`) change for dynamic tensors:
`-1` becomes a valid `dims` entry, and `strides` becomes an
axis-permutation when `is_dynamic == true`. Old plugins that do not
understand these reinterpretations could mis-handle Phase-2 graphs
silently if they read the fields under their Phase-1 meaning.

**Phase-line alignment.** Phase 1 ships pre-1.0 (under the existing
0.x line); Phase 2 is the 1.0 milestone. The 1.0 commitment also
signals the broader plugin-SDK stability promise per RFC 0005 §4.2.

### 6.9 Migration and rollout

Two rollout options were considered:

- **Option A, breaking change with no version negotiation.** Land
  the schema reinterpretation under the existing 0.x line and
  require every provider to be rebuilt and updated in lockstep.
  Simpler runtime (no version checks in the hot path), but every
  out-of-tree consumer breaks silently the moment they pick up a
  1.0-style graph; failures surface as misinterpretation of
  `dims == -1` or strides-as-order rather than a clean rejection.
- **Option B, version bump with explicit disqualification (chosen).**
  Both `plugin_sdk` and `data_sdk` bump to 1.0.0 (§6.8). Plugins
  declare which graph-schema major version they consume via a new
  optional symbol; the backend disqualifies plugins that cannot
  consume the running graph's schema. Failure mode is a clean "no
  applicable engine" rather than silent misinterpretation, and
  pre-1.0 plugins keep working for pre-1.0 graphs during transition.

**Why Option B.** The codebase commits to plugin SDK stability per
RFC 0005 §4.2; silent misinterpretation of `dims` or `strides` in an
out-of-tree provider is the exact failure mode the versioning policy
exists to prevent. The cost (one new plugin-reported field plus a
one-line check at load time) is small relative to the safety it
provides.

**Chosen approach: three composing layers of disqualification.**

- **Layer 1, backend load-time pre-filter on supported graph-schema
  version.** At plugin load the backend reads the plugin's reported
  supported graph-schema version through a new optional symbol
  `hipdnnPluginGetSupportedGraphSchemaVersion(const char** version)`,
  resolved via the existing `tryAssignSymbol` precedent at
  `backend/src/plugin/PluginCore.cpp:35-46` (the same mechanism
  already used for `hipdnnPluginGetApiVersion`; see §3.5). The
  signature mirrors the existing `hipdnnPluginGetApiVersion(const char**
  version)` for consistency. The runtime owns the filter policy: a
  plugin reporting schema version V is compatible with graphs of major
  version `major(V)` (FlatBuffers additive forward-compat across
  minors). Pre-1.0 plugins do not export the new symbol →
  `tryAssignSymbol` resolves it to `nullptr` → such plugins are
  filtered out for any 1.0+ graph by `getApplicableEngineIds`. The
  existing `hipdnnPluginGetApiVersion` signature is unchanged, so
  introducing the new symbol does not constitute an ABI break on the
  existing one.
- **Layer 2, plugin self-check inside `isApplicable`.** Plugins
  reporting 1.0 support inspect the graph and return false if any
  tensor has `is_dynamic == true` and the engine cannot serve
  dynamic tensors. The data_sdk helpers (§6.7) make this a one-line
  check.
- **Layer 3, Phase 1 applicability filter (§4.5) still applies.**
  Plugins lacking the override-execute symbol are skipped when the
  graph flag is set, independent of dynamic-tensor status.

The three layers compose: load-time prefiltering removes pre-1.0
plugins; per-call self-check removes 1.0-but-no-dynamic-support
plugins; per-call applicability filter removes plugins lacking the
override-execute symbol.

**In-tree migration steps (no separate migration document).**

- Bump the two version files (`plugin_sdk/version.json`,
  `data_sdk/version.json`) and the matching `.h.in` pair, plus the
  transitively-bumped `backend/version.json` and
  `frontend/version.json` per §6.8.
- Implement the new optional symbol
  `hipdnnPluginGetSupportedGraphSchemaVersion` in each in-tree plugin.
- Rebuild and re-link the in-tree plugin set (test plugins,
  miopen-provider, hipblaslt-provider) against 1.0 headers.
- Out-of-tree consumers follow the same steps in their own trees;
  they remain functional against 0.x graphs in the meantime via
  Layer 1.

**Out-of-tree plugins.** The 1.0 bump is announced via the project's
release-note channel [TODO(implementation): pin the specific channel
URL or distribution list]. Migration steps for downstream consumers:
(1) rebuild against new `data_sdk` / `plugin_sdk` headers; (2)
implement `hipdnnPluginGetSupportedGraphSchemaVersion`; (3) optionally
implement the override entry. No deprecation window for 0.x plugins
is offered — clean break at 1.0.

### 6.10 Reference-surface alignment

Phase 1 deliberately matches the reference surface's existing
override API (§2.1), so that user code translating between graph
runtimes does not fork at the override call site. Phase 2 extends
beyond the reference surface to support per-tensor declarative
dynamism. Users writing Phase-1-only code see a portable surface;
users opting into Phase 2's dynamic-tensor declarations see an
hipDNN-extended surface that may not have a one-to-one analogue
elsewhere.

The §6.4 stride-as-order semantic specifically goes beyond the
reference surface: reusing the `strides` field as an axis-permutation
is documented here as an intentional hipDNN extension, not an
accidental divergence.

### 6.11 Phase 2 key design decisions

This section mirrors the §5 KDD structure used for Phase 1.

**Sentinel-vs-bool-array (decision: `dims[d] == -1`).** Wildcard
positions are encoded as `-1` entries inside the existing `dims`
field rather than as a parallel `is_wildcard: [bool]` array. The
sentinel form keeps the FBS additive work to a single bool field
(`is_dynamic`) without introducing a second per-tensor array that
would have to be kept in lockstep with `dims`. The cost is that
plugins reading raw FBS must check `is_dynamic` before treating any
`dims` entry as a non-negative size; the §6.4 frontend constraint
plus the §6.8 major bump together prevent silent misreads.

**Stride-reuse (decision: reinterpret existing `strides`).** When a
tensor is dynamic, the existing `strides` field is reinterpreted as
an axis-permutation rather than introducing a new
`stride_order: [int]` field. This avoids growing the schema with a
field that is meaningful only when `is_dynamic == true` and that
would be defaulted-empty for every static tensor. The reinterpretation
is gated on `is_dynamic`, the same gate that already changes `dims`
semantics, so reviewers see one semantic switch rather than two.

**Major-bump (decision: 1.0 rather than continuing in 0.x).** The
Phase 2 ship is the SDK's 1.0 milestone. The FBS additivity alone
does not require a major bump; the semantic reinterpretation of
`dims` and `strides` does. Continuing in 0.x would silently break
out-of-tree plugins that read the fields under Phase-1 meaning
(Option A in §6.9). The 1.0 commitment additionally signals the
plugin-SDK stability promise per RFC 0005 §4.2 and gives a single
recognizable line for downstream consumers to align against.

**Three-layer-compat (decision: load-time prefilter plus per-call
self-check plus per-call symbol filter).** Disqualification is
layered rather than collapsed into a single check because each layer
catches a distinct class of mismatch: Layer 1 catches version
mismatch (pre-1.0 plugin meets 1.0 graph), Layer 2 catches
capability mismatch (1.0 plugin lacking dynamic-tensor support),
Layer 3 catches symbol mismatch (1.0 plugin without override-execute).
Collapsing them into one check would either over-reject (e.g., a 1.0
plugin without dynamic-tensor support that could still serve static
graphs) or require a richer per-plugin capability-vector field that
both layers would have to consume.

---

## 7. Risks

### 7.1 Override overload called without flag

**Risk**: an end user calls the override `Graph::execute()` overload
on a graph that did not set the flag at compile time. This is the
req 3 / req 9 boundary: the user is allowed to set the flag and
not supply overrides, but is not allowed to supply overrides
without setting the flag.

**Mitigation**: the frontend overload returns a non-OK `error_t`
and emits a log entry before any `setAttribute` call on the variant
pack, so misuse never reaches the backend. Integration test §9.5
layer 4 covers this path.

### 7.2 Plugin reports inconsistent version

**Risk**: a plugin reports a version through
`hipdnnPluginGetApiVersion` that does not match the entry points it
actually implements (the per-symbol-vs-per-version-string caveat
inherited from the existing optional-symbol machinery; see RFC 0002
and the `tryAssignSymbol` precedent at
`backend/src/plugin/PluginCore.cpp:35-46`).

**Mitigation**: capability detection is per-symbol via
`hasOverrideExecute()`, never per version string; see §4.4.

### 7.3 Future execution-plan cache interaction

**Risk**: hipDNN has no execution-plan cache keyed on graph contents
today (`GraphDescriptor::invalidateCache()` is a per-descriptor
serialized-buffer cache, not a cross-graph plan cache). If a future
shape-variant-aware plan cache is added (§10.3), the new
`is_dynamic_shape_enabled` flag and the variant-pack override
attributes must be cache-key inputs. Otherwise a non-override graph
could silently inherit a plan compiled for an override-capable
variant.

**Mitigation**: open question for the future plan-cache work, not a
defect in the current design. The new field is part of the FlatBuffer
serialization, so any cache keying on the serialized graph naturally
distinguishes the two variants; future plan-cache implementation must
explicitly verify this rather than assume it.

### 7.4 End-user discovery before any plugin implements the path

**Risk**: end users discover the new flag and overload before any
plugin implements `hipdnnEnginePluginExecuteOpGraphWithOverrides`
(the req 14 / req 15 dependency). They could write code against
the surface, run it, and get nothing useful back.

**Mitigation**: the SDPA gate (§4.7) hides the public surface until
enabled. Coordination with the SDPA RFC owner is required so the
gate ships in the same release as this plumbing. Worst case (gate
slips, override path used with no implementing plugin): §4.5
filtering returns a clean "no applicable engines" error rather than
a silent incorrect result.

### 7.5 Hidden divergence from SDPA gate

**Risk**: the SDPA team's eventual gate name or semantics may not
match this RFC's wiring assumptions. For example, the SDPA team
may choose a runtime gate where this RFC assumed a build-define,
or may choose semantics that gate only the SDPA op rather than the
override surface as a whole.

**Mitigation**: cross-reference the SDPA RFC once it lands;
revisit §4.7 and align gate wiring before implementation starts.
The §4.7 wording is intentionally non-specific about the gate
mechanism so that implementation can pick up the SDPA team's
choice without amending this RFC.

### 7.6 Reused-name semantic mismatch

**Risk**: the `dynamic_shape_enabled` name is broader than the
single-plan override semantic this RFC implements (see §5.2 for the
trade-off rationale and §10.6 for the mitigation path via a future
second flag).

**Commitment**: approving this RFC commits the project to enum 603's
reuse semantics. A future Phase requiring a separate flag (e.g.,
distinct "dynamic shapes enabled" vs "overrides allowed") would need a
sibling enum, not a redefinition of 603. The string mapping at
`BackendEnumStringUtils.hpp:306-307` carries the same name forward.

### 7.7 Layered-detection coupling

**Risk**: the detection chain has four independent layers (graph
flag, applicability skip, variant-pack attributes, plugin entry dispatch)
that can drift. A change to one layer that does not land in the
others can produce a drift state. For example, the flag may be set
but the applicability skip is not invoked, or the attributes are
written but dispatch ignores them. (This is the runtime-execution order; the
§4 prose introduces the layers in definition order: §4.1 graph flag,
§4.3 variant-pack attributes, §4.4 plugin entry, §4.5 applicability skip,
§4.6 dispatch.)

**Mitigation**: applicability filtering (§4.5) is the authoritative
pre-dispatch backstop. The §9.5 four-corner matrix plus the §9.2
"C-API bypass path" test together form the structural mitigation
against future drift; reviewers modifying any of the four layers
should run both.

### 7.8 Additive-change wiring discipline

**Risk**: three minor wiring concerns share the same shape; each is
an additive change that needs disciplined enumeration to avoid silent
breakage.

- **FBS additivity.** Per RFC 0005 §4.6.1, appending a defaulted
  bool to `table Graph` is forwards/backwards compatible. The field
  defaults to `false`, so a graph that does not opt in is byte-
  identical to today; downstream consumers that explicitly enumerate
  fields rather than parse-and-ignore-unknowns may need a touch-up.
- **700–799 attribute-id inventory.** The reserved range is shared
  with other in-flight work; implementation must inventory current
  allocations in `HipdnnBackendAttributeName.h` and
  `BackendEnumStringUtils.hpp` before assigning the five new IDs,
  and must allocate at the next available slot.
- **`-Wswitch` ABI.** Adding five new variant-pack enum values may
  surface `-Wswitch` warnings in switches without a `default:` arm.
  In-tree switches in `VariantDescriptor::setAttribute` /
  `getAttribute` are part of the wiring work; out-of-tree consumers
  see the warning at their own build time, which is the intended
  signal to update.

**Mitigation**: code review of the enum/schema additions is the
structural mitigation; the existing entries are concentrated in a
small number of headers, so the inventory is straightforward.

### 7.9 Phase 2 backward-compatibility

**Risk**: Phase 2 (see §6) introduces additional schema fields and
an auto-flagging behavior that could in principle disturb the Phase 1
contract, for example by silently changing the meaning of the graph
flag for graphs authored against Phase 1.

**Mitigation**: the Phase 2 schema additions are FBS additive (a
per-tensor `is_dynamic` boolean defaulting to `false`), following the
same RFC 0005 §4.6.1 forwards/backwards-compatibility discipline this
RFC already relies on for the Phase 1 graph flag. A Phase 1 graph
deserialized in a Phase 2 runtime sees no per-tensor `is_dynamic`
flags set (FBS default `false`), so the auto-flag does not fire and
Phase 1 semantics are bit-preserved. Phase 2 additionally guarantees
that no new validation gated solely on the graph flag rejects
Phase 1-shaped graphs (which by definition have all-concrete tensors).

### 7.10 Pre-1.0 plugin breakage at major bump

**Risk**: every existing plugin must be recompiled against 1.0 SDK
headers and report 1.0 support before it can serve 1.0-schema graphs
(§6.8 / §6.9).

**Mitigation**: the in-tree plugin set is small (test plugins,
miopen-provider, hipblaslt-provider) and is bumped in lockstep with
the SDK version; out-of-tree plugins remain usable against 0.x graphs
via the §6.9 Layer 1 pre-filter until they pick up the new headers.
Migration steps live inline in §6.9.

### 7.11 Stride-as-order misinterpretation

**Risk**: a static tensor with `strides == {3, 2, 1, 0}` has the byte
pattern of a stride-order. A plugin reading `strides` without first
checking `is_dynamic` could mis-handle the tensor.

**Mitigation**: three layers prevent this. The FBS `is_dynamic` field
gates the reinterpretation; the §6.4 frontend validation constraint
rejects static tensors with malformed strides at build time; and the
major version bump (§6.8) ensures no plugin reading a 1.0 graph
interprets a static tensor's strides as stride-order.

### 7.12 Per-execute overhead

**Risk**: the §4.5 applicability filter and the per-execute
attribute-presence check (§5.5) add cost on the SDPA hot path.

**Mitigation**: (a) commit to a microbenchmark in Step 4 measuring
`Graph::execute` overhead with and without the override path; (b)
target overhead < 1µs / call (placeholder — adjust after measurement);
(c) if exceeded, hoist the applicability check to plan-build time and
cache the filtered engine list per plan.

### 7.13 Auto-flag and late tensor mutation

**Risk**: if a caller marks a tensor dynamic after `Graph::build()`
has run, the §6.5 auto-flag does not re-fire. The graph would then
carry a dynamic tensor without the dynamic-shape-enabled flag set.

**Mitigation**: tensor descriptors are read-only after
`Graph::tensor()` returns the shared_ptr; the frontend rejects
setter calls on tensors already attached to a finalized graph. The
mutation path the risk describes is therefore not reachable in
well-formed code.

---

## 8. Execution Plan

These steps describe the implementation work that will follow this
RFC; they are **not** part of this RFC's scope. The step
boundaries are also the natural review boundaries for the
implementation PRs.

### Step 1: Schema + backend descriptor + enum wiring

- Add `is_dynamic_shape_enabled: bool = false;` to the end of `table
  Graph` in `flatbuffers_sdk/schemas/graph.fbs`.
- Wire attribute 603 in `GraphDescriptor` (set/get/finalize/pack/
  unpack).
- Add the five new `HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_*` attribute
  enum values at 704–708 (three payload + two per-UID lengths
  sidebands; the next free slots in the 700–799 range, per §3.3 /
  §4.3); add string mappings in `BackendEnumStringUtils.hpp`; wire
  them through `VariantDescriptor::setAttribute` / `getAttribute`.
  **No FBS schema** is added for the variant pack.
- Backend unit tests for round-trip on both the graph attribute and
  the variant-pack attributes.

### Step 2: Plugin SDK + optional symbol + applicability skip + host dispatch

- Add `hipdnnEnginePluginExecuteOpGraphWithOverrides` as a new
  optional symbol in
  `plugin_sdk/include/hipdnn_plugin_sdk/EnginePluginApi.h`.
- Add `tryAssignSymbol` resolution + `hasOverrideExecute()`
  predicate in `EnginePlugin`
  (`backend/src/plugin/EnginePlugin.cpp:29-84`).
- Modify `getApplicableEngineIds`
  (`EnginePluginResourceManager.cpp:385-423`) to skip
  non-implementers when the flag is set.
- Modify `EnginePluginResourceManager::executeOpGraph`
  (`EnginePluginResourceManager.cpp:569-614`) to inspect the
  variant pack for override attributes and dispatch to the new vs.
  existing plugin entry accordingly.
- Bump `HIPDNN_PLUGIN_SDK_VERSION_MINOR` in
  `plugin_sdk/version.h.in`.
- **No new public backend C-API entry** in this step or any other.

### Step 3: Frontend API

- Add `set_dynamic_shape_enabled` / `is_dynamic_shape_enabled` to
  `Graph` and the `_isDynamicShapeEnabled` member to
  `GraphAttributes`.
- Add the `Graph::execute()` overload accepting the three trailing
  parallel vectors.
- Frontend translates the override vectors into `setAttribute` calls
  on the `VariantDescriptor` (the new attributes from Step 1)
  before calling the existing `hipdnnBackendExecute`.
- Pack/unpack the flag in `GraphPacker` / `GraphUnpacker`.
- Wire the new public surface behind the SDPA team's build-define.
- Frontend integration tests (§9.3).

**SDPA-gate fallback.** Step 3 lands behind a hipDNN-owned
compile-time flag (default off). The SDPA team's gate flips it on
when ready. This decouples merge timing from external coordination;
if the SDPA gate slips, the new frontend surface stays internally
inert and `getApplicableEngineIds` filters all engines (loud failure
rather than silent misuse).

### Step 4: Cross-cutting tests

Per-step tests land with the code they exercise (so no step merges
untested code):

- **Step 2 carries**: fake plugins (override-implementing,
  override-omitting) under `tests/test_plugins/` (precedent:
  `TestIncompleteApiPlugin`); backend integration tests under
  `tests/backend/`, including the host-side dispatch-switch test.
- **Step 3 carries**: frontend integration tests under
  `tests/frontend/`.

Step 4 itself adds the tests that span both steps:

- The §9.5 four-corner matrix (flag × overrides) end-to-end.
- Phase 1 end-to-end via the SDPA-gate-enabled fake.
- Per RFC 0006: harness conventions for plugin-agnostic integration
  testing.

### Step 5: Phase 2 schema, backend descriptor, frontend wiring

- Append `dynamic: bool = false;` (bare-noun, matching `virtual:
  bool;` at `tensor_attributes.fbs:55`) to the end of `table
  TensorAttributes` in `data_sdk/schemas/tensor_attributes.fbs` and
  `flatbuffers_sdk/schemas/tensor_attributes.fbs` in lockstep.
- Add `HIPDNN_ATTR_TENSOR_IS_DYNAMIC = 1308` in
  `HipdnnBackendAttributeName.h` (next free slot in the 1300 range
  after 1305 `IS_VIRTUAL`, 1306 `VALUE_EXT`, 1307 `IS_BY_VALUE`); add
  the string mapping next to `IS_VIRTUAL`'s in
  `BackendEnumStringUtils.hpp`; wire `setAttribute` and `getAttribute`
  in `TensorDescriptor.cpp`.
- Add `_isDynamic`, `set_is_dynamic(bool)`, `get_is_dynamic()`, and
  the `mark_dynamic()` chaining helper to `TensorAttributes.hpp`.
  Pack and unpack via `DescriptorHelpers` and
  `DescriptorUnpackHelpers`.
- Add `TensorAttributes::DYNAMIC_DIM = -1` for callers that prefer
  the named constant over the literal sentinel.
- Bump `plugin_sdk/version.json`, `data_sdk/version.json`,
  `plugin_sdk/version.h.in`, and `data_sdk/version.h.in` to
  `1.0.0`.
- Backend and frontend round-trip tests for the new flag (§9.7).

### Step 6: data_sdk graph-handler helper queries

- Add `has_dynamic_tensors()`, `tensor(uid).get_is_dynamic()`, and
  `tensor(uid).get_wildcard_axes()` to the data_sdk graph-handler
  interface.
- Helper-query unit tests (§9.7).

### Step 7: Backend load-time pre-filter, auto-flag, validation relaxation

- Add the new optional plugin SDK symbol
  `hipdnnPluginGetSupportedGraphSchemaVersion(const char** version)`
  in `plugin_sdk/include/hipdnn_plugin_sdk/PluginApi.h`; resolve via
  `tryAssignSymbol` (precedent: `PluginCore.cpp:35-46`); the backend
  reads this at plugin load.
- Implement Layer 1 pre-filter in
  `EnginePluginResourceManager::getApplicableEngineIds`: plugins for
  which the new symbol resolved to nullptr (or that report a major
  version below the running graph's schema major) are excluded for
  1.0-schema graphs.
- Implement the §6.5 auto-flag in `lowerGraphToDescriptors()` (along
  the `Graph::build()` chain).
- Add the per-op `validate()` "any-dynamic-operand" early-return for
  value-dependent checks across the ops listed in §6.6.

> TODO(implementation): enumerate all op `validate()` methods needing
> the any-dynamic-operand early return. Audit
> `frontend/include/hipdnn_frontend/attributes/` and `Graph.hpp`
> build-time validation paths to produce the exhaustive list. This is
> implementation-time work; the RFC commits to the per-op pattern but
> not to the per-op file list.

### Step 8: Phase 2 tests

See §9.7 for the full test list. Phase 2 testing is the
synchronization point for Steps 5 through 7.

### Step X (deferred)

Real engine implementation is **out of scope** for the initial
implementation and will be tracked separately under the SDPA work.

### Step dependencies and parallelism

Step 1 establishes the Phase 1 descriptor and enum surface that
Steps 2 and 3 both depend on; it must land first. Once Step 1 is in,
**Steps 2 and 3 can proceed in parallel**: Step 2 (plugin SDK + host
dispatch) touches `plugin_sdk/` and `backend/src/plugin/`; Step 3
(frontend API) touches `frontend/include/` and `frontend/detail/`.
The two steps share no files. Step 4 fake plugins can be authored
once Step 2 merges; backend integration tests run against Step 2
alone; Step 4's frontend integration tests require Step 3. Step 4 is
the synchronization point where the parallel Phase 1 work
re-converges.

Step 5 must land before Step 6 (the data_sdk helpers query the new
`is_dynamic` field) and before Step 7 (the load-time filter checks
plugin-reported version against the 1.0 schema, and the auto-flag
reads the per-tensor field). Once Step 5 is in, Steps 6 and 7 may
proceed in parallel; Step 8 is the synchronization point for the
Phase 2 work.

---

## 9. Testing Plan

All tests are plumbing / API tests; there is no shape-correctness
validation in initial rollout (req 19). Test conventions follow
RFC 0006. Phase 1 tests are §9.1 through §9.6. Phase 2 tests are
§9.7.

### 9.1 Fake plugins

- `TestOverrideExecutePlugin`: implements
  `hipdnnEnginePluginExecuteOpGraphWithOverrides` and reports the
  new minor API version. Verifies dispatch lands on the new entry
  and captures the override payload for assertions.
- `TestNoOverrideExecutePlugin`: omits the new symbol. Verifies the
  §4.5 applicability skip and confirms non-implementers continue to
  function for non-override graphs. Modeled on
  `TestIncompleteApiPlugin`.
- Both share the existing `TestPluginCommon.hpp:40+` base, so the
  fake plugins differ only in the optional-symbol exposure.
- `TestNoOverrideExecutePlugin` is built with the new SDK headers but
  with the override symbol export deliberately omitted, simulating
  the binary-compat scenario where a plugin compiled against the
  previous SDK is loaded under the new backend.

### 9.2 Backend C-API + integration tests

Files: `tests/backend/IntegrationBackendExecuteApi.cpp`,
`IntegrationVariantPackApi.cpp`,
`IntegrationGraphDescriptorApi.cpp`.

- Set/get of attribute 603 round-trip on `GraphDescriptor`.
- Set/get of the five new `HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_*`
  attributes (704–708) round-trip on `VariantDescriptor`, including
  the per-UID lengths sidebands.
- Schema serialization round-trip including the new graph field:
  asserts a graph serialized with the flag set deserializes with
  it set, and a graph serialized without the flag deserializes
  with the default `false` value.
- End-to-end happy path with `TestOverrideExecutePlugin`: the host
  extracts override attributes and dispatches to
  `hipdnnEnginePluginExecuteOpGraphWithOverrides`; the fake plugin
  captures the override UIDs/shapes/strides and the test asserts
  they match.
- **Host-side dispatch-switch test**: with
  `TestOverrideExecutePlugin` loaded, dispatch goes to the new entry
  when the variant pack carries override attributes and to
  `hipdnnEnginePluginExecuteOpGraph` when it does not: the direct
  verification of req 8 for the non-override path.
- Skip-path coverage: with the override flag set,
  `TestNoOverrideExecutePlugin` never receives the execute call,
  asserted via fake-plugin instrumentation that `isApplicable` was
  never invoked.
- Negative test: a graph without the flag continues to behave
  exactly as before, regardless of the loaded plugin's override
  capability: the regression guard for the §2.3 binary-compat
  constraint.
- **No new public backend C-API entry is exercised**: there is
  no `hipdnnBackendExecuteWithOverrides_ext` or similar.
- **Plugin returns non-OK from new entry**: status propagates
  through `EnginePluginResourceManager::executeOpGraph`, through
  `hipdnnBackendExecute`, and surfaces as a non-OK `error_t` to the
  frontend caller; the per-call `VariantDescriptor` is discarded
  after execute, so non-OK status from the plugin does not leak
  override-attribute state to subsequent execute calls.
- **Mismatched override-vector lengths**: invoke the frontend
  overload with mismatched sizes; assert `HIPDNN_STATUS_BAD_PARAM`
  per §4.2.1 rule 1 and that no `setAttribute` call is made.
- **Malformed override payload**: negative dim values return
  `HIPDNN_STATUS_BAD_PARAM` per §4.2.1 rule 6, and empty inner
  vectors (rank mismatch when declared rank > 0) return
  `HIPDNN_STATUS_BAD_PARAM` per §4.2.1 rule 3.
- **C-API bypass path**: construct override variant-pack attributes
  directly via `hipdnnBackendSetAttribute` without setting the
  graph flag; verify the §4.6 `hasOverrideExecute()` guard fires
  when the selected plugin lacks the symbol and dispatches to the
  new entry when it has it.

### 9.3 Plugin loading + applicability tests

Files: `tests/backend/IntegrationPluginLoading.cpp`,
`IntegrationGraphEngineFiltering.cpp`.

- `hasOverrideExecute()` returns true for `TestOverrideExecutePlugin`
  and false for `TestNoOverrideExecutePlugin`. This is the
  fundamental capability-detection test; if it fails, every other
  override test will fail in confusing ways.
- `getApplicableEngineIds` skips non-implementers when the flag is
  set; asks both plugins when the flag is unset. Asserted via
  fake-plugin instrumentation that records `isApplicable` calls,
  not just via the returned engine-id list.

### 9.4 Frontend lift / lower tests

Files: `tests/frontend/IntegrationGraphLifting.cpp`,
`IntegrationSdpaFwdDescriptorLifting.cpp`,
`IntegrationGraphEngineFiltering.cpp`.

- Round-trip the flag through the frontend pack/unpack path:
  `set_dynamic_shape_enabled(true)` on a `Graph`, lift to
  `GraphDescriptor`, lower back, assert
  `is_dynamic_shape_enabled()` is preserved.
- SDPA-specific lift/lower with the flag set, using the existing
  SDPA descriptor lifting harness, to confirm SDPA graphs
  participate correctly.
- The `Graph::execute()` override overload returns an error when
  called on a graph without the flag set. Asserts both the error
  return and the absence of any backend call.
- Round-trip of override vectors through the frontend → variant-pack
  `setAttribute` translation: supply override vectors at the
  frontend, read back the variant-pack attributes, assert match.

### 9.5 Layered-detection coverage (mitigates §7.7)

End-to-end test asserting the four layers cooperate as designed:

1. **Flag absent.** Both fakes consulted by `isApplicable`; no
   override-attribute writes; dispatch lands on the existing plugin entry.
2. **Flag present, no override args.** `TestNoOverrideExecutePlugin`
   never asked about applicability; `TestOverrideExecutePlugin`
   receives the existing-entry call (not the new entry). Verifies
   req 9.
3. **Flag present, override args supplied.** Non-implementers
   skipped; override attributes written to the variant pack;
   `TestOverrideExecutePlugin` receives the new entry with the
   correct payload.
4. **Override args without flag.** Frontend returns non-OK before
   any backend call. Asserts complete absence of backend-side
   activity: no `setAttribute`, no `hipdnnBackendExecute`, no
   plugin call.

The four-corner matrix is the structural mitigation for the §7.7
drift risk; running it should be a precondition for landing any
change to §4.5 or §4.6.

### 9.6 SDPA gate interaction

The plumbing exercised above runs **regardless** of the SDPA gate
: the gate is the SDPA team's responsibility, not this RFC's.
**No env var is introduced or tested by this RFC.** When the SDPA
gate is wired, additional tests verify the public API surface
(graph flag setter and `Graph::execute()` overload) is hidden
when the gate is disabled. Those tests are tracked under the SDPA
work, not under this RFC or the initial implementation.

### 9.7 Phase 2 test categories

- **FBS round-trip on `is_dynamic`.** The new field round-trips
  through both `data_sdk/schemas/tensor_attributes.fbs` and
  `flatbuffers_sdk/schemas/tensor_attributes.fbs`.
- **Backend descriptor round-trip on
  `HIPDNN_ATTR_TENSOR_IS_DYNAMIC`.** Set/get returns the same value;
  default is `false` for tensors that never set it.
- **Frontend round-trip.** `set_is_dynamic(true)`, lift to backend,
  lower back, then `get_is_dynamic()` returns true. Mirror test for
  `mark_dynamic()`.
- **Auto-flag end-to-end.** A graph with one dynamic tensor sees
  `is_dynamic_shape_enabled()` return true after `build()` without
  any explicit graph-flag call (§6.5).
- **Validation relaxation.** For each per-op `validate()` that
  gains a dynamic-tensor early-return, a test constructs an op with
  a dynamic operand (containing a `-1` dim) and confirms validation
  passes; the same op with an all-concrete operand and an
  intentionally invalid value still fails Phase 1's check.
- **SDK helper queries.** `has_dynamic_tensors()` returns true on a
  graph with at least one dynamic tensor and false on an
  all-static graph; `wildcard_axes()` returns the correct axis
  index list for a tensor with `dims == {1, 4, -1, -1}`.
- **Load-time pre-filter.** A fake plugin reporting "0.x only"
  support is filtered out for a 1.0-schema graph, even when it
  implements the override entry. Asserted via fake-plugin
  instrumentation that the plugin's `isApplicable` is never invoked
  for the 1.0 graph.
- **Self-check fake plugin.** A fake plugin reporting "1.0"
  support but rejecting dynamic tensors via its own `isApplicable`
  is correctly filtered for a dynamic-tensor graph but not for a
  static-tensor graph.
- **Stride-as-order resolution.** A dynamic tensor with `strides
  == {3, 2, 1, 0}` resolves to NCHW element-strides correctly when
  execute supplies concrete dims, exercising the
  `generateStrides(dim, strideOrder)` helper through the dynamic
  path.
- **Negative test — static tensor with `-1`.** Building a static
  tensor (`is_dynamic == false`) with `-1` in any `dims[d]` is
  rejected at build time with a non-OK status; the descriptor is
  never finalized.
- **§7.11 stride-coincidence regression.** A static tensor whose
  strides happen to look like a permutation `{3, 2, 1, 0}` is
  interpreted as element-strides, not stride-order. Cross-check that
  `extractStrideOrder` is not invoked on static tensors.
- **Multi-plugin co-loaded selection.** With both
  `TestOverrideExecutePlugin` and `TestNoOverrideExecutePlugin`
  loaded, an override-flagged graph selects the implementing plugin
  and never dispatches to the non-implementing one (asserted via
  fake-plugin instrumentation).
- **Cross-version FBS round-trip.** A graph serialized by a Phase 0
  binary deserializes correctly in a Phase 1 binary, and vice versa
  for Phase 1 ↔ Phase 2; default values for newly-added fields land
  as expected.
- **`TensorAttributes::DYNAMIC_DIM` constant equivalence.** Callers
  using the named constant produce byte-identical descriptors to
  callers using the literal `-1`.
- **Layered-disqualification ordering.** A pre-1.0 plugin that
  *also* lacks the override symbol — assert which layer (Layer 1
  load-time pre-filter or Layer 3 §4.5 applicability skip) fires
  first and that the user-visible diagnostic is unambiguous about the
  reason for exclusion.
- **No-leak after non-OK override entry.** The §9.2 "plugin returns
  non-OK from new entry" scenario is extended to verify subsequent
  execute calls (with and without overrides) succeed: no override
  state leaks across calls through the per-call `VariantDescriptor`
  lifecycle.

---

## 10. Future Considerations

### 10.1 Further generalization of declarative dynamism

Phase 2 (§6) introduces per-tensor `is_dynamic` declarations and so
partially addresses the original "generalization beyond SDPA"
question. A future RFC may extend the declarative-dynamism surface
further: storing max-shape metadata on individual tensor descriptors
(distinct from `is_dynamic`'s wildcard-dim semantic), or replacing
the §4.2 parallel-vector override payload with a struct-keyed map
once the frontend ergonomics are revisited per §5.3.

### 10.2 Reference implementation in real providers

Once shipping, the next steps are to wire override-execute into
MIOpen and hipBLASLt providers as those provider teams adopt the
surface. Each provider will report its support via the optional
symbol and gradually broaden the set of graphs for which the
override path is applicable.

### 10.3 Execution-plan caching strategy

§7.3 raises the cache-key concern. A deeper plan-cache strategy
for shape-variant plans (including whether to share plans across
override-shape variants of the same graph or to key on the
override-tensor metadata) is deferred to a future RFC. The
current design ensures the override flag distinguishes cache keys
correctly but does not attempt smart sharing across variants.

Phase 2 makes the cache-key story slightly more involved: the
per-tensor `is_dynamic` flag and, for dynamic tensors, the
wildcard-axis set must both be cache-key inputs so that two graphs
that differ only in which axes are wildcards do not collide on a
shared plan.

### 10.4 Removal of the SDPA feature-flag gate

Once override-execute is in general use across providers, the SDPA
build-define gate becomes a no-op and can be retired. Removal is
expected to be a one-line change to whichever build-define the
SDPA team chose, plus deletion of the conditional-compilation
guards in the frontend.

### 10.5 `data_sdk` schema impact beyond Phase 2

Phase 2 (§6.3) already changes the `data_sdk` schema: the
`is_dynamic: bool = false;` field is appended to `table
TensorAttributes` in both schema directories in lockstep. This is
part of the RFC, not deferred future work.

Future schema impact remains possible for unrelated needs. If
override semantics ever need to be reflected in serialized execution
plans or engine configs (e.g. for plan caching as in §10.3), the
`data_sdk` FlatBuffer schemas would need a further additive update.
The variant pack itself remains runtime-only.

### 10.6 Possible future kernel-cache-reuse flag

A future RFC may introduce a second graph flag to surface
kernel-cache-reuse semantics: sharing one set of compiled kernels
across shape-variant graph builds. That flag would be distinct from
the override-execute path this RFC introduces, and adding it would
partially recover the §5.2 trade-off, where the reused
`dynamic_shape_enabled` name is broader than the override-only
semantic actually implemented here.

### 10.7 Additional deferred work

- **Dynamic rank.** Tensors whose rank varies per execute. Out of
  scope for Phase 2; requires a variant-pack rank attribute,
  descriptor changes, and an SDK helper. §5.10 records the fixed-rank
  commitment for this RFC.
- **Partial-packing escape hatch for dynamic strides.** A third
  stride-mode where some inner-axis strides are declarative and outer-
  axis stride is element-wise. Currently the dynamic-tensor stride
  field is binary (axis-order or element-strides; see §6.4).
- **Engine-side constraint expression.** SDK extensions letting
  plugins declare per-axis min/max ranges, alignment requirements, and
  correlated-wildcard pairs. Closes the remaining "applicability
  decidable only at execute time" gap that §6.7's helper queries
  cannot address.
- **Plan-cache key includes auto-flag.** Explicit cache-key contract
  for the build-time-derived `is_dynamic_shape_enabled` flag (already
  noted as deferred in §7.3 — linked here for discoverability).

---

## 11. Glossary

- **Overridable tensor**: a graph tensor whose dims and strides
  may be re-supplied at execute time via the override surface.
- **Variant pack**: the `VariantDescriptor`-backed runtime-only
  carrier of per-execution payload (`_dataPointers`, `_uniqueIds`,
  `_workspace`, plus per-execute attributes set via
  `setAttribute`). New in this RFC: the new variant-pack attributes
  in the 700–799 range that travel via the same set/get path. The
  variant pack is constructed and destroyed per `Graph::execute()`
  call; it has no FlatBuffer schema and is never serialized.
- **Max-shape semantics**: the convention that graph-time tensor
  dims represent the maximum allowed shape; execute-time overrides
  must fit within these.
- **Override transport**: the mechanism by which override values
  reach the plugin: frontend translation into variant-pack
  `HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_*` attributes (IDs 704, 705,
  706), host extraction into flat parallel arrays, and dispatch to
  the optional plugin symbol
  `hipdnnEnginePluginExecuteOpGraphWithOverrides`.
- **Optional symbol**: a plugin entry point that may be absent
  without causing plugin-load failure. The backend uses
  `tryAssignSymbol` to resolve them and per-symbol predicates
  (e.g. `hasOverrideExecute()`) to gate behavior. Precedent: RFC
  0002 plus the `tryAssignSymbol` use at `PluginCore.cpp:35-46`.
- **Applicability skip**: the optimization in
  `getApplicableEngineIds` that bypasses `isApplicable` for plugins
  that cannot possibly support the requested feature, used here
  for plugins lacking the override-execute symbol when the graph
  flag is set.
- **Host dispatch switch**: the inspection in
  `EnginePluginResourceManager::executeOpGraph` of variant-pack
  override-attribute presence to decide which plugin entry to call.
- **SDPA gate**: the build-define (mechanism TBD, owned by the
  SDPA team) that controls whether the override API surface is
  exposed to end users in initial rollout. This RFC does not
  introduce the gate.
- **Dynamic tensor**: a tensor declared with `is_dynamic == true`
  at graph build time, whose wildcard dims and stride ordering are
  resolved at execute time via the Phase 1 override transport.
- **Wildcard dim**: a `dims[d] == -1` entry on a dynamic tensor,
  indicating the value is supplied at execute time. Callers may use
  the `TensorAttributes::DYNAMIC_DIM` named constant in place of the
  literal sentinel.
- **Stride order**: the axis-permutation interpretation of the
  `strides` field for dynamic tensors; reuses the existing
  `generateStrides(dim, strideOrder)` semantic.
- **Auto-flag**: the build-time mechanism that sets the graph's
  dynamic-shape-enabled flag if any tensor in the graph is declared
  dynamic.
- **Supported graph-schema version**: a per-plugin declaration of the
  single serialized-graph schema version the plugin can consume,
  reported via the optional symbol
  `hipdnnPluginGetSupportedGraphSchemaVersion`. The runtime owns the
  filter policy: a plugin reporting version V is compatible with
  graphs of major version `major(V)` (FlatBuffers additive forward-
  compat across minors). Consulted by the backend's load-time
  pre-filter (§6.9 Layer 1).
