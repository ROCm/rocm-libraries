# hipDNN: Overridable Tensor Shapes Design Document

- Contributors: Brian Harrison
- **Status**: Draft

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Current System Overview](#3-current-system-overview)
4. [Proposed Design](#4-proposed-design)
5. [Key Design Decisions](#5-key-design-decisions)
6. [Phase 2: Dynamic Tensors](#6-phase-2-dynamic-tensors)
   - 6.1 [Why Phase 2](#61-why-phase-2)
   - 6.2 [Per-tensor `is_dynamic` flag](#62-per-tensor-is_dynamic-flag)
   - 6.3 [Dynamic-tensor semantics](#63-dynamic-tensor-semantics)
   - 6.4 [Auto-flagging behavior](#64-auto-flagging-behavior)
   - 6.5 [Frontend validation relaxation](#65-frontend-validation-relaxation)
   - 6.6 [IsApplicable improvements](#66-isapplicable-improvements)
   - 6.7 [Versioning and plugin-version filtering](#67-versioning-and-plugin-version-filtering)
   - 6.8 [Migration and rollout](#68-migration-and-rollout)
   - 6.9 [Cross-runtime API surface alignment](#69-cross-runtime-api-surface-alignment)
   - 6.10 [Phase 2 key design decisions](#610-phase-2-key-design-decisions)
7. [Risks](#7-risks)
8. [Execution Plan](#8-execution-plan)
9. [Testing Plan](#9-testing-plan)
10. [Future Considerations](#10-future-considerations)
    - 10.1 [Per-tensor declarative metadata](#101-per-tensor-declarative-metadata)
    - 10.2 [Execution-plan caching strategy](#102-execution-plan-caching-strategy)
    - 10.3 [Removal of the SDPA feature-flag gate](#103-removal-of-the-sdpa-feature-flag-gate)
    - 10.4 [Override-transport surface evolution](#104-override-transport-surface-evolution)
    - 10.5 [Additional deferred work](#105-additional-deferred-work)
11. [Glossary](#11-glossary)

---

## 1. Executive Summary

This RFC proposes adding **overridable tensor shapes** and **dynamic
tensors** to hipDNN as two committed phases. Phase 1 introduces a
graph-compile-time flag declaring max-shape
semantics plus an execute-time mechanism that re-supplies per-tensor
dims and strides without rebuilding the graph. Phase 2 builds on the
Phase 1 transport with per-tensor `is_dynamic` declarations that move
applicability and validation decisions to build time. The immediate
consumer is scaled dot-product attention (SDPA); see
[§2.2](#22-hipdnn-gap) for the gap this closes.

Initial rollout is plumbing-only: frontend, backend descriptors,
plugin SDK, and host-side dispatch all gain override-aware
machinery, but **no engine implementation lands**. The
new public surface is gated behind the SDPA feature flag introduced
in PR [#6493](https://github.com/ROCm/rocm-libraries/pull/6493); this
RFC introduces no new env var, CMake option, or runtime knob of its
own. Existing
plugins remain binary-compatible: plugins that do not implement the
new optional plugin SDK entry continue to work unchanged for
non-override graphs and are filtered out for override-enabled graphs.
The public backend C-API surface is preserved unchanged: there is one
`hipdnnBackendExecute` today and one after this RFC lands.

**Binary-compatibility scope.** The backend computes the minimum
plugin API version each graph requires based on the features the
graph uses. Plugins report a single supported API version through the
existing `hipdnnPluginGetApiVersion()` symbol. Graphs that do not opt
into overrides impose no override-feature version requirement and
continue to be served by existing plugins unchanged. Graphs using the
Phase 1 override flag require plugins reporting
`>= 1.1.0`; graphs using the Phase 2 per-tensor dynamic feature
require `>= 1.2.0` (placeholders; see [§4.5](#45-applicability-filtering)
for the full rollout table). See
[§4.4](#44-plugin-sdk-extension) for the symbol-fallback behavior and
[§4.5](#45-applicability-filtering) /
[§6.7](#67-versioning-and-plugin-version-filtering) for the full
versioning model.

---

## 2. Problem Statement

### 2.1 End-user API surface

The desired end-user surface keeps the same execute API. Override values
are specified on the variant pack descriptor and transported to plugins
through a new optional plugin C-API used to filter plugin engines on
dynamic graphs.

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
   exactly one after this RFC lands. The single-execute-entry shape
   matches existing graph runtimes and makes porting easier.
3. **Keep the graph descriptor read-only after build**. Per-tensor
   shapes declared at graph time must not be mutated at execute time;
   overrides travel via the variant pack only.
4. **Stay off-by-default for end users** until at least one shipping
   plugin implements the override path. The new surface is gated behind
   the SDPA feature flag; see PR
   [#6493](https://github.com/ROCm/rocm-libraries/pull/6493).

---

## 3. Current System Overview

The hipDNN graph pipeline has four steps:

1. **Create graph.** Frontend builds a graph describing tensors,
   operations, and graph-level attributes.
2. **Validate, finalize, lower.** Frontend validates the graph and
   lowers it into the backend for plugin consumption.
3. **Plugins asked for applicability.** Backend asks each loaded
   plugin which of its engines can execute the finalized graph.
4. **Execute.** A variant pack carries per-execution payload
   (`tensorId → devicePtr`, workspace) into the chosen plugin engine.

---

## 4. Proposed Design

### 4.1 Graph flag

A graph-level boolean flag declares that the graph may receive
execute-time overrides. The flatbuffer schema gains one defaulted bool, and
the flag is wired through the existing C-API get/set attributes path
for lowering and lifting.

```cpp
class Graph {
public:
    Graph& set_dynamic_shape_enabled(bool enabled);
    bool   is_dynamic_shape_enabled() const;
};
```

Per RFC 0005, appending an optional defaulted field to an existing
table is wire-compatible. The attribute reuses the already-reserved
backend enum value `HIPDNN_ATTR_OPERATIONGRAPH_IS_DYNAMIC_SHAPE_ENABLED`
(ID 603).

Note: when Phase 2 lands, `set_dynamic_shape_enabled(false)` may be
overridden at `build()` time if any tensor in the graph is declared
dynamic; see [§6.4](#64-auto-flagging-behavior).

### 4.2 Frontend execute API

`Graph::execute` gains an overload that takes override shapes and
strides. Variant packs are runtime; the additions plumb a
`tensorId → override dims` and `tensorId → override strides` mapping
through the existing variant-pack channel.

Two overload shapes are provided. The parallel-array form aligns with
existing graph-runtime conventions:

```cpp
error_t execute(hipdnnHandle_t handle,
                std::unordered_map<int64_t, void*>& tensor_uid_to_pointer_map,
                void* workspace,
                std::vector<int64_t> const& override_uids,
                std::vector<std::vector<int64_t>> const& override_shapes,
                std::vector<std::vector<int64_t>> const& override_strides) const;
```

A map-keyed convenience overload trades the length-consistency
invariant for an idiomatic C++ surface:

```cpp
struct OverrideEntry {
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
};

error_t execute(hipdnnHandle_t handle,
                std::unordered_map<int64_t, void*>& tensor_uid_to_pointer_map,
                void* workspace,
                std::unordered_map<int64_t, OverrideEntry> const& overrides) const;
```

The map overload is pure sugar over the parallel-array form: it
internally lowers to the parallel-array form before any validation
runs, so both share identical validation, variant-pack translation,
and dispatch. There is no separate code path for the map form.

#### 4.2.1 Override input validation

The frontend is the single point of input policing. Each violation
returns `HIPDNN_STATUS_BAD_PARAM`; the first failure returns
immediately:

1. **Length consistency.** For the parallel-array overload,
   `override_uids.size()`, `override_shapes.size()`, and
   `override_strides.size()` must be equal. (The map overload has no
   such constraint by construction.)
2. **Unknown UID.** Each override key must identify a tensor declared
   in the graph. Workspace is never overridable.
3. **Rank mismatch.** Each entry's shape and stride vectors must equal
   the declared rank of the named tensor.
4. **Max-shape exceeded.** For non-wildcard dimensions, each override
   dim must be `<=` the declared graph-time dim (the max-shape).
   Wildcard dimensions (Phase 2, build-time `dims[d] == -1`) have no
   upper bound, and any positive override value is permitted.
5. **Duplicate UIDs.** The parallel-array overload's `override_uids`
   must contain no duplicates.
6. **Positive dim values.** Each override dim must be `> 0`.
7. **Positive stride values.** Each override stride must be `> 0`.
8. **Stride ordering preserved.** Overrides that reorder the layout
   are rejected at execution time; see
   [§6.3](#63-dynamic-tensor-semantics) for the canonical static /
   dynamic stride distinction.

A call with no overrides at all is equivalent to a non-override
execute call; the frontend skips the variant-pack attribute writes so
dispatch falls through to the existing entry.

### 4.3 Backend C-API and variant-pack attributes

The single existing `hipdnnBackendExecute` is preserved; override
semantics fold into the variant pack via four new attributes in the
reserved 700–799 range. Three are payload attributes; the fourth is
a single per-UID rank sideband:

```c
HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_UNIQUE_IDS = 704  // array of int64 UIDs
HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_SHAPES     = 705  // flat int64 dims, concatenated
HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_STRIDES    = 706  // flat int64 strides, concatenated
HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_LENGTHS    = 707  // per-UID rank sideband
```

**Single per-UID rank.** Shapes and strides for a given tensor share
the same rank, so a single `OVERRIDE_LENGTHS` sideband suffices to
slice both `OVERRIDE_SHAPES` and `OVERRIDE_STRIDES`.

**Storage convention.** `OVERRIDE_SHAPES` and `OVERRIDE_STRIDES` are
flat `int64_t[]` buffers (the concatenation of per-UID inner vectors).
`OVERRIDE_LENGTHS[i]` carries the rank of the tensor identified by
`OVERRIDE_UNIQUE_IDS[i]`, in the same order.

**Worked example.** A graph that overrides three tensors of ranks
3, 4, and 4:

```text
override_uids    = [42,  17,  99]                            // 3 entries
override_lengths = [ 3,   4,   4]                            // 3 entries (one rank per uid)
override_shapes  = [s0,s1,s2,  s3,s4,s5,s6,  s7,s8,s9,s10]  // 11 entries (3+4+4)
override_strides = [t0,t1,t2,  t3,t4,t5,t6,  t7,t8,t9,t10]  // 11 entries (3+4+4)
```

The flatten-plus-per-UID-rank form matches the plugin SDK signature
so both layers share a single representation.

### 4.4 Plugin SDK extension

A new optional entry is added to the engine-plugin C API:

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
    const int64_t* const*                   override_strides,
    const uint32_t*                         override_lengths);
```

The signature extends the existing `hipdnnEnginePluginExecuteOpGraph`
with three flat parallel arrays plus a single per-UID rank sideband.
Plugins never see the variant pack, so the host extracts the override
payload and passes it flat alongside the existing device-buffer array.
For each `i`, `override_shapes[i]` and `override_strides[i]` are int64
arrays of length `override_lengths[i]` carrying the override dims and
strides for tensor `override_uids[i]`.

**Pointer lifetime.** All pointer-of-pointer parameters (and their
inner arrays) are valid for the duration of this call only; the
plugin must not retain or dereference them after returning, matching
the existing `hipdnnPluginDeviceBuffer_t[]` contract.

**Versioning.** The implementation bumps the plugin SDK API version
to `1.1.0` (backwards-compatible addition; see
[§4.5](#45-applicability-filtering) for the two-bump rollout across
Phase 1 and Phase 2). Plugins compiled against the older
SDK do not export the new symbol and continue to serve non-override
graphs. Capability detection is layered: the per-graph plugin-version
filter ([§4.5](#45-applicability-filtering),
[§6.7](#67-versioning-and-plugin-version-filtering)) is the primary
applicability check, backed by the existing
`hipdnnPluginGetApiVersion()` query and parsed via
`hipdnn_data_sdk::utilities::Version` (from
`data_sdk/include/hipdnn_data_sdk/utilities/VersionUtils.hpp`). The
filter rejects non-conforming plugins before they are asked about a
graph that requires the override feature; the per-symbol
`hasOverrideExecute()` predicate then acts as a runtime safety net at
host dispatch ([§4.6](#46-host-dispatch-logic)).

**Resolution.** The new symbol resolves through the existing
`tryAssignSymbol` optional-symbol pattern (RFC 0002 precedent). A new
predicate `EnginePlugin::hasOverrideExecute()` returns true iff the
symbol resolved.

### 4.5 Applicability filtering

The backend maps each graph to the minimum plugin API version it
requires (see [§6.7](#67-versioning-and-plugin-version-filtering))
and filters plugins against that mapping. The minimum is computed from
the features the graph uses: a graph that does not opt into overrides
imposes no override-feature requirement; a graph with the Phase 1
override flag set requires `>= 1.1.0`; a graph that uses Phase 2
per-tensor dynamic features requires `>= 1.2.0`.

The backend reads each plugin's `MAJOR.MINOR.PATCH` via the existing
`hipdnnPluginGetApiVersion(const char**)` symbol and compares it
using `hipdnn_data_sdk::utilities::Version`:

```cpp
auto requiredVersion = computeMinimumPluginApiVersion(graph);
for (plugin in plugins) {
    if (Version{plugin.apiVersion()} < requiredVersion) continue;
    if (plugin.isApplicable(graph)) result.push_back(plugin.engineId());
}
```

Plugins that do not export `hipdnnPluginGetApiVersion` fall back to
`"0.0.0"` (the existing default in `PluginCore.cpp`), so they are
treated as not implementing any of the override-feature contracts and
are filtered out for any graph whose required version is `> 0.0.0`.

**Versioning rollout.** Plugins are about to undergo a planned
baseline bump to `1.0.0` (separate from this RFC); any plugin that
does not declare an API version is treated as `1.0.0` thereafter.
The plugin SDK API version reported by an override-capable provider
then tracks the override feature in two further bumps:

| State | `apiVersion()` | Graphs eligible to be served |
|-------|----------------|------------------------------|
| Plugin not exporting symbol | `0.0.0` (fallback) | Non-override graphs only |
| Today (symbol exported)     | `0.0.1` | Non-override graphs only |
| After baseline bump  | `1.0.0` | Non-override graphs only (existing plugins, no override capability) |
| After Phase 1 ships  | `1.1.0` | Graphs with the override flag set |
| After Phase 2 ships  | `1.2.0` | Graphs using `is_dynamic` / wildcard / stride-as-order |

Each ship of an override-capable provider must bump
`HIPDNN_PLUGIN_API_VERSION` to the value corresponding to the feature
set it implements. The `1.1.0 → 1.2.0` step is a backwards-compatible
minor addition; pre-existing 1.1.0 plugins continue to serve Phase 1
graphs unchanged.

The version numbers above (`1.0.0` / `1.1.0` / `1.2.0`) are
placeholders chosen to communicate the staging — the actual numbers
are TBD and will be finalized when the baseline bump lands.

Graphs that do not opt in see no plugin-eligibility change; existing
plugins continue to serve them. Graphs that opt in see only plugins
reporting an API version at or above the required minimum. Plugins
that meet the version requirement are still asked about applicability
for all graphs that meet that requirement; the override execute entry
itself is invoked only when the variant pack carries override attributes.

Two filters compose: (a) the per-graph plugin-version filter runs at
applicability time and is the primary gate; (b) `Graph::execute()`
performs a light check that the override execute symbol is present on
the chosen plugin before dispatch as a safety net. Caching
applicability results across compiles is a likely future enhancement
but is out of scope for this RFC.

### 4.6 Host dispatch logic

The host dispatch path inspects the variant pack for the new
`HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_*` attributes and dispatches
accordingly:

```cpp
extract { uids, ptrs } -> deviceBuffers[]
extract optional { override_uids, override_shapes, override_strides, override_lengths }

if (override attributes present) {
    // hasOverrideExecute() is the dispatch-time safety net. The per-graph
    // version filter at applicability time (§4.5) is the primary gate; this
    // guard catches the edge case where a plugin reports the right version
    // but is missing the symbol.
    if (!plugin.hasOverrideExecute()) {
        return HIPDNN_STATUS_NOT_SUPPORTED;
    }
    plugin.executeOpGraphWithOverrides(...);
} else {
    plugin.executeOpGraph(...);  // existing entry
}
```

**Override-graph / non-override-graph contract.** Override graphs
(those with the graph flag set) call the override execute API.
Non-override graphs call the non-override API. It is **not** an error
to take the override path with no overrides supplied; dispatch
correctly falls through to the existing entry. It **is** an error to
call the override execute API with overrides on a graph that did not
set the flag at build time; the frontend overload rejects this with
`HIPDNN_STATUS_BAD_PARAM` before any backend call.

**Lifetime.** Temporary arrays passed to the plugin point into the
variant pack's storage and live for the duration of the execute
call; the plugin must not retain them.

---

## 5. Key Design Decisions

### 5.1 Graph flag rather than per-tensor opt-in

**Decision**: a single boolean on the graph gates the entire
override surface. Override-capability is graph-wide; individual
tensors are not annotated.

**Rationale**: SDPA opts in for the whole graph, so a graph-wide
flag is sufficient for the immediate consumer. Matching the existing
graph-flag convention used by other graph runtimes makes it easy for
consumers to port existing code. A graph flag also keeps the
attribute-namespace impact minimal: one new graph attribute
(see [§4.1](#41-graph-flag)) versus one per overridable tensor.

**Trade-off**: a future workload wanting override on only some
tensors must opt the whole graph in, expanding the
applicability-filter surface and forcing users to reason about
override semantics for tensors they will not vary. A per-tensor opt-in attribute
(storing max-shape metadata on individual descriptors) remains a future option.

### 5.2 Parallel-vector override payload

**Decision**: three trailing `std::vector` arguments on the
`execute()` overload (`override_uids`, `override_shapes`,
`override_strides`), indexed positionally so that
`override_shapes[i]` and `override_strides[i]` describe the tensor
named by `override_uids[i]`.

**Rationale**: match the API surface established by other graph
runtimes so users can port existing code with minimal changes.
SDPA workloads commonly run on multiple runtimes, and a divergent
hipDNN signature would force per-runtime branches at every call site.

**Trade-off**: positional indexing across three vectors is error-prone
for users constructing overrides manually. The map-keyed convenience
overload ([§4.2](#42-frontend-execute-api)) addresses this without
forcing the parallel-array form on the public C-side API.

**Stride-ordering invariant.** Overrides cannot reorder the layout;
see [§6.3](#63-dynamic-tensor-semantics).

### 5.3 Override transport: variant-pack attributes + plugin SDK entry

**Decision**: keep a single `hipdnnBackendExecute`; add four new
variant-pack attributes (`HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_*`:
three payload plus one per-UID rank sideband) in the 700–799 range,
AND add the optional plugin SDK entry
`hipdnnEnginePluginExecuteOpGraphWithOverrides`. Both surfaces are
required.

**Rationale**: matching the established frontend overload and
single-execute-entry shape makes it easy for consumers to port
existing code; the dedicated variant-pack attributes are hipDNN's
clean realization of that API surface. Two alternatives were rejected:
a sibling `hipdnnBackendExecute_with_overrides` entry diverges from
the established surface, and a plugin-SDK-only path forces the
frontend to bypass the backend descriptor model, breaking the
layering RFC 0004 establishes. The plugin SDK entry is required
because of the host's flattening step; plugins never see the variant
pack directly.

**Trade-off**: two surfaces must stay in sync across the host's
flattening step. Drift is covered by testing.

### 5.4 Host-side dispatch switch

**Decision**: variant-pack override-attribute presence drives whether
the host dispatches to
`hipdnnEnginePluginExecuteOpGraphWithOverrides` (present) or
`hipdnnEnginePluginExecuteOpGraph` (absent).

**Rationale**: the graph flag declares user intent to override; that
flag is always present for fast applicability checks and API parity
across runtimes. But individual `execute()` calls may or may not
supply overrides, and the cleanest way to express this is to key
dispatch on the per-call payload rather than on the per-build flag.
The switch is independent of the graph flag, simplifying the
four-corner matrix of flag absent/present × overrides absent/present
(see [§9.4](#94-layered-detection-coverage)); each case has a single
well-defined dispatch outcome. The alternative (keying dispatch on the graph flag) was
rejected because it would force an override-implementing plugin to
receive the override entry even when the caller supplied no
overrides, violating the non-override-call contract.

Phase 2 layers per-tensor flagging on top of the graph flag to give
plugins enough information to make applicability decisions before
execute time; per-tensor flagging is not available in Phase 1 and
becomes the recommended path once Phase 2 ships (see
[§6](#6-phase-2-dynamic-tensors)).

**Trade-off**: every execute call pays an attribute-presence check on
the variant pack (a small in-memory map lookup), even for graphs that
never opt in.

### 5.5 Optional plugin symbol with minor-version bump

**Decision**: optional symbol with minor-version bump.

**Rationale**: per RFC 0002, missing optional symbols (resolved via
`tryAssignSymbol`) don't break plugin loading; this is the
already-established pattern for the base-plugin optional symbols. A
major-version bump would force every plugin to recompile against the
new SDK before it could be loaded again, breaking the constraint that
existing providers that don't implement this API continue to work.
A minor bump signals the backwards-compatible API addition without
invalidating any existing plugin binaries. The full per-graph
plugin-version filtering model is detailed in
[§6.7](#67-versioning-and-plugin-version-filtering).

**Trade-off**: defense-in-depth via two asymmetric layers. The
per-graph version filter at applicability time (see
[§6.7](#67-versioning-and-plugin-version-filtering)) is the primary
gate; the dispatch-time `hasOverrideExecute()` check is the safety
net for the rarer case where a plugin bumps its reported version but
forgets to implement the override symbol, returning
`HIPDNN_STATUS_NOT_SUPPORTED` instead of crashing. The version filter
is conservative: a symbol-implementing plugin that forgot to bump its
version is filtered out, but without a version bump the backend has
no way to know it.

### 5.6 SDPA feature-flag mechanism

**Decision**: defer the gate to the SDPA team; this RFC introduces no
new env var, CMake option, or runtime knob.

**Rationale**: SDPA is the immediate consumer and owns end-user
surfacing of the feature. Coupling the override-execute gate to the
SDPA gate means there is one knob, owned by one team, controlling
end-to-end visibility, with no hipDNN-specific gate to retire later.

**Trade-off**: tying override visibility to the SDPA gate means SDPA
could ship before the override feature is ready. If that happens,
hipDNN can introduce a dedicated override gate at that point; until
then the shared gate keeps the surface area small. See
[§10.3](#103-removal-of-the-sdpa-feature-flag-gate) for the removal
path.

### 5.7 Frontend overload rather than separate method

**Decision**: overload `Graph::execute()` rather than add a separate
override method.

**Rationale**: cross-runtime alignment
([§5.2](#52-parallel-vector-override-payload)) is the primary driver;
RFC 0004's overloaded `create_execution_plan` sets precedent that
overloads are accepted in the hipDNN frontend.

**Trade-off**: disambiguation is by argument list, not method name;
unfamiliar readers may miss that a call is the override variant.
Mitigated by the frontend overload returning a non-OK error before
any backend work on misuse, making the failure mode loud.

### 5.8 No engine implementation in initial rollout

**Decision**: initial implementation ships plumbing only; no
provider adopts the override path in the same drop.

**Rationale**: the SDPA work owns the first real implementation;
plumbing must land first to unblock that work. Splitting the work
this way also lets the plumbing be reviewed and merged on a
schedule independent of provider readiness.

**Trade-off**: design validation comes only from fake plugins; a real
engine may surface API gaps that fake plugins miss. The SDPA gate
(PR [#6493](https://github.com/ROCm/rocm-libraries/pull/6493)) keeps
the public surface dormant until a shipping provider exists, so
users cannot stumble onto a non-functional API in the interim.

### 5.9 Tensor rank is fixed at build time

**Decision**: tensor rank is declared at graph build time and cannot
vary per execute call. Phase 2 wildcards apply to dim *values* within
a fixed rank; the override transport is for value-level dynamism, not
structural dynamism.

**Rationale**: keeps the backend descriptor model unchanged and bounds
the validation-relaxation surface.

**Trade-off**: dynamic rank is explicitly out of scope, recorded as a
future consideration ([§10.5](#105-additional-deferred-work)).

---

## 6. Phase 2: Dynamic Tensors

This section specifies the **Phase 2** design committed by this RFC.
Phase 2 builds on Phase 1's variant-pack override transport by adding
a per-tensor `is_dynamic` flag declared at graph build time, with
wildcard dim and stride-ordering semantics resolved at execute time
through the Phase 1 transport.

### 6.1 Why Phase 2

Phase 1 keeps a graph's per-tensor shapes concrete at build time and
accepts overrides only at execute time. Phase 1 has two consequences
for SDPA-style workloads:

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

Phase 2 closes both gaps by making **declared intent** first-class in
the schema: tensors marked dynamic are statements about *what the
user will control at execute time and how*. The graph thereby
carries enough information at build time for applicability checks,
plan-cache decisions, and validation policy to be settled before the
first execute call. The execute-time transport itself is unchanged,
since Phase 2 reuses the Phase-1 variant-pack and plugin-SDK surface.

### 6.2 Per-tensor `is_dynamic` flag

Phase 2 adds a `bool is_dynamic` flag to the tensor schema and updates
validation checks to honor it. The flag mirrors the existing
`is_virtual` plumbing. Packing, unpacking, backend enum wiring, and
frontend setter/getter all follow that precedent.

The schema field is appended (defaulted false; wire-compatible, see
[§4.1](#41-graph-flag)): a Phase 1 graph deserialized in a Phase 2
runtime sees `is_dynamic == false` on every tensor.

The frontend exposes a chaining helper `mark_dynamic()` so callers can
write `graph.tensor(...)->mark_dynamic()`.

### 6.3 Dynamic-tensor semantics

A **dynamic tensor** is a tensor declared with `is_dynamic == true`
at graph build time. Two semantic shifts apply, both scoped to
dynamic tensors only:

- **Wildcard dims.** `-1` in any dim slot means "deferred until
  execute." The frontend exposes a `DYNAMIC_DIM` constant so callers
  do not hard-code the sentinel.
- **Stride-as-order.** When `is_dynamic == true`, the `strides`
  field carries an axis ordering (lower index = inner, higher = outer)
  rather than element-strides. The user supplies actual element-strides
  at execute time via the override transport; the declared ordering
  constrains which permutations are valid.

Static tensors retain Phase 1 semantics: concrete positive `dims` and
explicit element-strides.

**Build-time constraints.** A static tensor must not contain `-1` in
`dims`. Violations are rejected at build time.

**Illustrative example.** A 4-D input with concrete batch and channel
dims, wildcard spatial dims, and NCHW stride ordering:

```yaml
X:
  is_dynamic = true
  dims       = { 1, 4, -1, -1 }   # N=1, C=4, H/W deferred
  strides    = { 3, 2, 1, 0 }     # N outermost, W innermost
```

### 6.4 Auto-flagging behavior

If any tensor in a graph is marked dynamic, the graph's
dynamic-shape-enabled flag is set automatically during
`Graph::build()` to prevent user error. Users authoring graphs through
the Phase 2 dynamic-tensor API do not need to also call
`set_dynamic_shape_enabled(true)`; any prior explicit
`set_dynamic_shape_enabled(false)` is overridden if dynamic tensors
are present.

### 6.5 Frontend validation relaxation

If a tensor is flagged dynamic, frontend compile-time shape-validation
checks that depend on concrete dim values are skipped for that
operand; providers handle their own constraints at execute time.

**Skipped for dynamic operands**: any value-dependent check (SDPA
head-dim divisibility, convolution spatial-dim relationships, matmul
inner-dim compatibility, broadcast shape inference, etc.).

**Still enforced regardless**: rank consistency, dtype consistency
and per-op dtype constraints, axis-ordering consistency, presence/
absence of required tensor inputs.

**Broadcast carve-out.** Broadcast compatibility is checked at execute
time once override values are supplied (equal, or one is 1). Providers
that do not support broadcasting may reject the call at that point.

### 6.6 IsApplicable improvements

With per-tensor `is_dynamic` declared at build time, plugins can
pre-filter at applicability time instead of discovering mismatches at
execute time. Plugins implementing the override entry consult the
per-tensor flag and the wildcard-axis layout when deciding whether
their engines apply: an engine that supports only spatial wildcards
rejects graphs with batch wildcards, an engine requiring NHWC
ordering rejects graphs declaring NCHW, and so on.

This closes the "applicability decidable only at execute time" gap
for the cases the per-tensor flag exposes. Richer engine-side
constraints (min/max ranges per axis, alignment, correlated wildcards)
are out of scope for this RFC and can be added in a future iteration;
until then engines that need them must either over-accept and report
`not applicable` at execute time, or reject any wildcard graph in
`isApplicable`.

### 6.7 Versioning and plugin-version filtering

Phase 2's frontend/backend API additions are backwards-compatible
extensions, so they ship as a **minor** plugin-SDK version bump (from
1.1.0 to 1.2.0). The new schema field (`is_dynamic`) is additive
(see [§4.1](#41-graph-flag)); the new optional execute symbol is
resolved per-symbol via `tryAssignSymbol`. There is no ABI break.

**Per-graph plugin-version filtering.** The substantive change is how
the backend selects plugins for graphs that use the new Phase 2
features. Each graph's required minimum plugin version is computed
from the features it uses; the rollout table in
[§4.5](#45-applicability-filtering) gives the mapping.

**Composes with the override-symbol filter.** See
[§5.5](#55-optional-plugin-symbol-with-minor-version-bump) for the
two-layer (version filter + per-symbol check) model.

**Why minor and not major.** The minor-vs-major rationale is in
[§5.5](#55-optional-plugin-symbol-with-minor-version-bump).

### 6.8 Migration and rollout

Because the change is non-breaking, migration is straightforward:

- **Existing plugins (in-tree and out-of-tree).** Continue to function
  unchanged for Phase-1-shaped graphs. They are filtered out of the
  applicable set only for graphs that actually use Phase-2 features.
- **In-tree plugins adopting Phase 2.** Bump their reported version to
  1.2.0 once they are updated to recognize `is_dynamic` and handle
  the override execute API.
- **Out-of-tree plugins adopting Phase 2.** Same path: rebuild
  against 1.2.0 headers, implement the override execute symbol,
  bump the reported version.

No coordinated version bump across plugin / data / backend / frontend
SDKs is needed for this work. Each plugin migrates on its own
schedule, and the per-graph plugin-version filter ensures graphs
always reach a plugin capable of serving them or fail cleanly with
"no applicable engines."

### 6.9 Cross-runtime API surface alignment

Phase 2 extends beyond the cross-runtime-aligned Phase 1 surface
([§5.2](#52-parallel-vector-override-payload)) to support per-tensor
declarative dynamism. The stride-as-order semantic in particular is
an intentional hipDNN extension, not an accidental divergence:
reusing the `strides` field as an axis-permutation has no one-to-one
analogue in other graph runtimes.

### 6.10 Phase 2 key design decisions

**Sentinel rather than parallel bool array.**

**Decision**: encode wildcard positions as `-1` entries inside the
existing `dims` field rather than as a parallel `is_wildcard: [bool]`
array.

**Rationale**: keeps the flatbuffer-schema additive work to a single
bool field (`is_dynamic`) without introducing a second per-tensor
array that would have to be kept in lockstep.

**Trade-off**: plugins reading the raw flatbuffer schema must check
`is_dynamic` before treating any `dims` entry as a non-negative size;
the per-graph plugin-version filter ensures older plugins never see
such a graph.

**Stride reuse rather than new stride-order field.**

**Decision**: when a tensor is dynamic, reinterpret the existing
`strides` field as an axis-permutation rather than introduce a new
`stride_order: [int]` field.

**Rationale**: avoids growing the schema with a field that is
meaningful only when `is_dynamic == true` and is defaulted-empty for
every static tensor. The reinterpretation is gated on `is_dynamic`,
the same gate that already changes `dims` semantics.

**Trade-off**: a field's semantic meaning depends on a sibling flag,
which readers must remember to check.

---

## 7. Risks

### 7.1 Override overload called without flag

**Risk**: an end user calls the override `Graph::execute()` overload
on a graph that did not set the flag at compile time. Setting the
flag without supplying overrides is allowed; supplying overrides
without the flag is not.

**Mitigation**: the frontend overload returns a non-OK `error_t`
and emits a log entry before any `setAttribute` call on the variant
pack, so misuse never reaches the backend. Integration tests cover
this path.

### 7.2 Plugin reports inconsistent version

**Risk**: a plugin reports a version through
`hipdnnPluginGetApiVersion()` that does not match the entry points
it actually implements.

**Mitigation**: capability detection is layered. The per-graph version
filter at applicability time (see [§4.5](#45-applicability-filtering))
is the primary gate, and the per-symbol `hasOverrideExecute()` check
at dispatch is the safety net. The integration test suite catches
version-vs-symbol mismatches in CI.

### 7.3 End-user discovery before any plugin implements the path

**Risk**: end users discover the new flag and overload before any
plugin implements `hipdnnEnginePluginExecuteOpGraphWithOverrides`.
They could write code against the surface, run it, and get nothing
useful back.

**Mitigation**: the SDPA feature-flag gate
(PR [#6493](https://github.com/ROCm/rocm-libraries/pull/6493)) hides
the public surface until enabled. Worst case (gate slips, override
path used with no implementing plugin): applicability filtering
returns a clean "no applicable engines" error rather than a silent
incorrect result.

### 7.4 Stride-as-order misinterpretation

**Risk**: a static tensor with `strides == {3, 2, 1, 0}` has the value
pattern of a stride-order. A plugin reading `strides` without first
checking `is_dynamic` could mis-handle the tensor.

**Mitigation**: the flatbuffer-schema `is_dynamic` field gates the reinterpretation;
the frontend validation constraint rejects static tensors with
malformed strides at build time; and the per-graph plugin-version
filter ([§6.7](#67-versioning-and-plugin-version-filtering)) ensures
older plugins never see a graph that uses stride-as-order.

---

## 8. Execution Plan

Implementation plan for the work this RFC enables:

### Step 1: Phase 1 backend wiring

Schema field, graph attribute, variant-pack override attributes, and
their backend round-trip tests. No frontend changes.

### Step 2: Plugin SDK extension and host dispatch

Optional override execute symbol, `hasOverrideExecute()` predicate,
per-graph plugin-version filter at applicability time (graphs requiring
the override feature skip plugins reporting an older `apiVersion()`),
and the dispatch-time symbol check as a safety net. Bump plugin SDK
minor version to `1.1.0`.

### Step 3: Frontend API

`set_dynamic_shape_enabled` setter/getter on `Graph`, `Graph::execute`
overload (parallel-array and map-keyed forms), variant-pack
translation, and pack/unpack of the graph flag.

### Step 4: Phase 1 cross-cutting tests

Fake plugins (one implementing the override symbol, one omitting it)
and the four-corner test matrix verify the end-to-end flow (see
[§9.4](#94-layered-detection-coverage)).

### Step 5: Phase 2 backend + frontend wiring

Per-tensor `is_dynamic` schema field, backend enum, frontend
setter/getter, `mark_dynamic()` helper, `DYNAMIC_DIM` constant, and
their round-trip tests.

### Step 6: Phase 2 version-filter extension, auto-flag, validation relaxation

Extend the per-graph plugin-version filter (introduced in Step 2) to
recognize the `1.2.0` requirement for graphs using `is_dynamic`
tensors, wildcard dims, or stride-as-order; auto-flag during
`Graph::build()`; per-op `validate()` "any-dynamic-operand"
early-return for value-dependent checks.

### Step 7: Phase 2 tests

Full coverage per [§9.5](#95-phase-2-test-categories).

---

## 9. Testing Plan

All tests are plumbing / API tests; no shape-correctness validation
lands with this RFC. Test conventions follow RFC 0006.

### 9.1 Test types

The plan exercises three categories:

- **Unit tests** covering the new graph attribute, variant-pack
  override attributes, per-tensor `is_dynamic` flag, version-parsing
  helpers, and other component-level logic introduced by this work.
- **Integration tests** that exercise set/get round-trip on the new
  graph attribute, the new variant-pack attributes, and the
  per-tensor `is_dynamic` flag (descriptor → serialize → deserialize
  → read-back returns the same value, including the default-`false`
  case for graphs and tensors that never opt in); the host dispatch
  switch; applicability filtering; override-payload translation from
  frontend to variant pack to plugin; and rejection paths
  (`HIPDNN_STATUS_BAD_PARAM` for malformed overrides, "no applicable
  engines" for graphs no plugin can serve).
- **API tests** verifying the frontend overload contract: empty
  overrides equivalent to non-override execute; missing graph flag +
  override payload returns an error before any backend call;
  override-graph / non-override-graph dispatch contracts hold.

### 9.2 Fake plugins enable end-to-end without real engines

Two fake plugins (one implementing
`hipdnnEnginePluginExecuteOpGraphWithOverrides`, one omitting it) are
loaded by the integration suite. They capture call sites and payload
contents for assertion, making it possible to test the full plumbing
path (frontend overload through host dispatch into plugin entry)
without any real engine implementation. The override-omitting fake
doubles as the binary-compatibility regression: a plugin built without
the new symbol must continue to serve non-override graphs and must be
filtered out of override-eligible applicability sets.

### 9.3 SDPA gate

Only the public frontend API for overrides is gated by the SDPA
feature flag
(PR [#6493](https://github.com/ROCm/rocm-libraries/pull/6493)).
Internal backend plumbing (the schema field, variant-pack attributes,
host dispatch switch, applicability filtering, plugin SDK predicate)
is not gated, so the unit and integration tests that exercise that
plumbing run unconditionally. Keeping the internal-API tests
ungated is deliberate; they catch regressions in the underlying
machinery regardless of feature-flag state. Only the tests that drive
the gated public frontend API are themselves protected by the flag.

### 9.4 Layered-detection coverage

The four-corner matrix must run for any change to applicability
filtering or host dispatch:

1. **Flag absent.** Both fake plugins consulted by `isApplicable`;
   dispatch lands on the existing plugin entry.
2. **Flag present, no override args.** Non-implementers skipped from
   applicability; the override-implementing plugin receives the
   existing entry, not the new entry.
3. **Flag present, override args supplied.** Non-implementers
   skipped; override attributes written to the variant pack; the
   override-implementing plugin receives the new entry with the
   correct payload.
4. **Override args without flag.** Frontend returns non-OK before
   any backend call. No `setAttribute`, no `hipdnnBackendExecute`,
   no plugin call.

### 9.5 Phase 2 test categories

- **Round-trip** for `is_dynamic` through the flatbuffer schema, backend descriptor,
  and frontend pack/unpack.
- **Auto-flag** end-to-end: a graph with one dynamic tensor sees
  `is_dynamic_shape_enabled()` return true after `build()` without
  any explicit setter call.
- **Validation relaxation**: rolled out per-op as providers gain
  dynamic dim/stride support. SDPA is the first op to receive the
  dynamic-operand early-return in `validate()` and accompanying
  tests; other ops are updated only when a provider adds dynamic
  support for them. The test, when added, confirms a dynamic operand
  bypasses value-dependent checks while a concrete-operand bad value
  still fails.
- **Per-graph plugin-version filter**: a fake plugin reporting an
  older version is filtered out for graphs requiring 1.2.0+, but
  remains eligible for graphs that use no Phase-2 features.
- **Stride-as-order validation**: a dynamic tensor declaring
  `strides == {3, 2, 1, 0}` (NCHW ordering) accepts execute-time
  override strides consistent with that permutation and rejects
  strides implying a different axis order. Cross-check that the
  stride-order helper is never invoked on static tensors (a static
  tensor whose strides coincidentally look like a permutation must
  be interpreted as element-strides).
- **Build-time rejection** for static tensors containing `-1` dims.
- **Cross-version flatbuffer-schema round-trip**: graphs serialized by older
  binaries deserialize cleanly in newer ones with new fields landing
  at their defaults.

---

## 10. Future Considerations

### 10.1 Per-tensor declarative metadata

Phase 2 introduces per-tensor `is_dynamic` declarations and so
partially addresses the original "generalization beyond SDPA"
question. The initial scope here is intentionally minimal; future
RFCs can expand the declarative-dynamism surface as use cases
appear. Examples of richer per-tensor metadata that could be added
later: max-shape bounds (an upper limit on a wildcard dim),
alignment or multiple-of constraints (e.g. "any multiple of 4"), or
other shape descriptors the user wants to commit to up front so the
plugin can specialize.

### 10.2 Execution-plan caching strategy

Plan-cache strategy for shape-variant plans is deferred to a future
RFC.

### 10.3 Removal of the SDPA feature-flag gate

The SDPA feature flag gates this feature today. The gate is removed
once a provider implements override execute. If SDPA ships GA first,
hipDNN introduces a dedicated override gate that replaces the SDPA
gate so the public surface stays hidden until a real implementation
exists.

### 10.4 Override-transport surface evolution

Future ways the override transport may be extended include a
struct-keyed per-tensor map, wildcard-stride support, multiple layout
choices per tensor, and richer engine-side constraint expression. Each
would be additive on top of the current parallel-vector transport.

### 10.5 Additional deferred work

- **Dynamic rank.** Tensors whose rank varies per execute. Out of
  scope for Phase 2; requires a variant-pack rank attribute,
  descriptor changes, and an SDK helper. This RFC commits to a
  fixed-rank model.
- **Partial-packing escape hatch for dynamic strides.** A third
  stride-mode where some inner-axis strides are declarative and
  outer-axis stride is element-wise. The dynamic-tensor stride field
  is currently binary (axis-order or element-strides).

---

## 11. Glossary

- **Overridable tensor**: a graph tensor whose dims and strides
  may be re-supplied at execute time via the override surface.
- **Variant pack**: the runtime-only carrier of per-execution
  payload (data pointers, unique IDs, workspace, plus per-execute
  attributes set via `setAttribute`). New in this RFC: the variant-pack
  attributes in the 700–799 range that travel via the same set/get
  path. The variant pack is constructed and destroyed per
  `Graph::execute()` call; it has no flatbuffer schema and is never
  serialized.
- **Max-shape semantics**: the convention that graph-time tensor
  dims represent the maximum allowed shape; execute-time overrides
  must fit within these.
- **Override transport**: the mechanism by which override values
  reach the plugin: frontend translation into variant-pack
  `HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_*` attributes (IDs 704–707
  carrying the UID list, per-tensor dim arrays, per-tensor stride
  arrays, and per-tensor rank sideband respectively), host extraction
  into flat parallel arrays, and dispatch to the optional plugin
  symbol `hipdnnEnginePluginExecuteOpGraphWithOverrides`.
- **Optional symbol**: a plugin entry point that may be absent
  without causing plugin-load failure. The backend uses
  `tryAssignSymbol` to resolve them and per-symbol predicates
  (e.g. `hasOverrideExecute()`) to gate behavior. Precedent: RFC 0002.
- **Applicability skip**: the mechanism that bypasses non-conforming
  plugins for graphs that cannot use them. Two layers compose: the
  per-graph plugin-version filter at applicability time (the primary
  gate) and the per-symbol `hasOverrideExecute()` check at dispatch
  (a safety net). See [§4.5](#45-applicability-filtering) and
  [§4.6](#46-host-dispatch-logic).
- **Host dispatch switch**: the inspection of variant-pack
  override-attribute presence to decide which plugin entry to call
  (`hipdnnEnginePluginExecuteOpGraphWithOverrides` if present, the
  existing entry otherwise). The companion safety-net
  `hasOverrideExecute()` check is described under "Applicability
  skip" above.
- **SDPA gate**: the SDPA feature flag introduced in PR
  [#6493](https://github.com/ROCm/rocm-libraries/pull/6493), which
  controls whether the override public API surface is exposed to end
  users. This RFC does not introduce the gate.
- **Dynamic tensor**: a tensor declared with `is_dynamic == true`
  at graph build time, whose wildcard dims and stride ordering are
  resolved at execute time via the override transport.
- **Wildcard dim**: a `dims[d] == -1` entry on a dynamic tensor,
  indicating the value is supplied at execute time. Callers may use
  the `TensorAttributes::DYNAMIC_DIM` named constant in place of the
  literal sentinel.
- **Stride order**: the axis-permutation interpretation of the
  `strides` field for dynamic tensors, reusing the existing
  stride-generation semantic.
- **Auto-flag**: the build-time mechanism that sets the graph's
  dynamic-shape-enabled flag if any tensor in the graph is declared
  dynamic. See [§6.4](#64-auto-flagging-behavior).
- **Supported plugin SDK API version**: a per-plugin declaration of
  the Plugin SDK API version the plugin was built against, reported via
  the existing `hipdnnPluginGetApiVersion(const char**)` symbol as a
  plain `"MAJOR.MINOR.PATCH"` semver string. The backend parses and
  compares it with
  `hipdnn_data_sdk::utilities::Version` from `VersionUtils.hpp`.
  Plugins that do not export the symbol fall back to `"0.0.0"` and are
  treated as not implementing any of the override-feature contracts.
- **Required plugin SDK API version**: the per-graph minimum Plugin
  SDK API version the backend computes from the features the graph
  uses; see [§4.5](#45-applicability-filtering) for the mapping. The
  runtime keeps a plugin in the applicable set for a given graph only
  when the plugin's supported version is `>=` the graph's required
  version.
