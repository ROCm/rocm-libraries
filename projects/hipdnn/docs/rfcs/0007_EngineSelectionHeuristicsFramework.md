# RFC: Engine Selection and Heuristics Framework for hipDNN

| Field | Value |
|-------|--------|
| Status | First draft |
| Component | Backend, heuristic/selection plugins, frontend (§16) |

## Table of contents

1. [Executive summary](#1-executive-summary)
2. [Motivation](#2-motivation)
3. [Baseline: current behavior](#3-baseline-current-behavior)
4. [Goals and non-goals](#4-goals-and-non-goals)
5. [Design overview](#5-design-overview) (includes [5.4 Two-tier plugin objects](#54-two-tier-plugin-objects-handle-vs-policy-descriptor))
6. [Device properties](#6-device-properties)
7. [SelectionHeuristic interface](#7-selectionheuristic-interface)
8. [C ABI for heuristic plugins](#8-c-abi-for-heuristic-plugins)
9. [Policy plugins and the outer loop](#9-policy-plugins-and-the-outer-loop)
10. [HeuristicPluginManager and resource layer](#10-heuristicpluginmanager-and-resource-layer)
11. [Versioning and compatibility checks](#11-versioning-and-compatibility-checks)
12. [Logging](#12-logging)
13. [Serialized graph, device properties, and graph-level preferences](#13-serialized-graph-device-properties-and-graph-level-preferences)
14. [EngineHeuristicDescriptor integration](#14-engineheuristicdescriptor-integration)
15. [Public API notes](#15-public-api-notes)
16. [Frontend API flow (mirror engine selection)](#16-frontend-api-flow-mirror-engine-selection)
17. [Testing](#17-testing)
18. [Risks and open questions](#18-risks-and-open-questions)
19. [Glossary](#19-glossary)

---

## 1. Executive summary

This document proposes a **first-draft** design for an extensible **engine selection and heuristics** framework in hipDNN. The framework replaces the hard-coded ordering in `EngineHeuristicDescriptor::finalize()` with an **ordered outer loop** over **policy plugins**. Each policy is a distinct selection strategy (for example: machine-learning-based selector, rule-based selector, cache-driven selector, or a deterministic default selector).

Device capabilities enter the system **only as explicit data** (`DeviceProperties`), not by probing HIP from inside plugins. Callers may later supply overrides via backend attributes (for example `HIPDNN_ATTR_ENGINEHEUR_DEVICEPROP`), aligned with the broader direction of **explicit device facts at the selection boundary**. Across the **stable C ABI**, those facts are passed as a **FlatBuffer-serialized** payload in **`hipdnnPluginConstData_t`** (same tooling and buffer wrapper type as serialized graph data in **engine** plugins—see [§13.2](#132-serialized-device-properties-flatbuffer)) via **`hipdnnHeuristicHandleSetDeviceProperties`** on the plugin session handle ([§8](#8-c-abi-for-heuristic-plugins)); plugins **query** that handle state during **`Finalize`** (policy descriptors are **created with** the handle—[§8.7](#87-policy-descriptor-per-slot-graph--candidate-ids)), not via a separate device payload on the policy API.

Heuristic plugins implement a **two-tier** **stable C ABI** ([§8](#8-c-abi-for-heuristic-plugins)): a long-lived **plugin handle** (**`hipdnnHeuristicHandle_t`**, **createHandle** / **destroyHandle** / **setDeviceProperties**) created with the same timing and storage pattern as other hipDNN plugin handles, and a **policy descriptor** (**`hipdnnHeuristicPolicyDescriptor_t`**, **createPolicyDescriptor** / **destroyPolicyDescriptor** / **setEngineIds** / **Finalize** / **getSortedIds**) whose **lifecycle is owned by** **`EngineHeuristicDescriptor`**—one per slot in the ordered policy list, destroyed with the heuristic descriptor. That ABI is **separate from the engine plugin ABI**—a `.so` is one or the other, never both.

The **`SelectionHeuristic`** C++ type (or equivalent facade) wraps a **policy descriptor** and forwards to the C ABI; **stateful tracking** in the plugin lives behind the **plugin handle**. **HeuristicPluginManager** (and handle-scoped **HeuristicPluginResourceManager**) own discovery, loading, version validation, registration, and **plugin-handle** instances per **`hipdnnHandle`**. **`Graph.preferred_engine_id`** is honored by the **frontend** as a post-hoc reorder of the heuristic-ranked engine configs returned by **`HIPDNN_ATTR_ENGINEHEUR_RESULTS`** (see **`Graph::initializeEngineConfig`**); the backend heuristic loop is unaware of it. The previous **`HIPDNN_ENGINE_OVERRIDE_FILE`** knob has been renamed to **`HIPDNN_HEUR_CONFIG_PATH`** and is now implemented as a regular built-in policy, **`SelectionHeuristic::Config`**, that lives in the ordered policy list and **declines** when no rule matches so subsequent policies (typically **`SelectionHeuristic::StaticOrdering`**) take over ([§5.3.5](#535-the-config-built-in-policy)). The default deterministic ordering policy (**`SelectionHeuristic::StaticOrdering`**) is shipped as a **backend built-in**: it implements the same heuristic plugin C ABI shape (function-pointer table) but is **registered in-process** at **`HeuristicPluginManager`** construction time, with no separate `.so` to discover or dlopen ([§10.1](#101-heuristicpluginmanager)). **`EngineHeuristicDescriptor::finalize()`** walks the **ordered policy list** using **descriptor-owned** policy objects; the resolved order is **user-configurable** at the public/config surface ([§5.3](#53-ordered-policy-list-default-and-user-configuration)). The backend resolves user-supplied **policy name** strings to **`int64_t` policy IDs** with the **same deterministic hash as engine IDs** ([RFC 0003](0003_EngineIdDesign.md), `hipdnn_data_sdk::utilities::policyNameToId`) before matching loaded plugins (and built-ins) and running the outer loop.

There is **no** separate post-loop step that applies **`utilities::sortEngineIds`** (or any other ordering) inside the backend. Legacy deterministic ordering is available **only** when a policy in **`orderedPolicyIds`** implements it—for example **`SelectionHeuristic::StaticOrdering`**. If every policy declines or the list is misconfigured so that no policy succeeds, **`finalize()`** fails using the **same error path as the rest of the hipDNN backend** (for example **`THROW_IF_FALSE`** / **`HipdnnException`** with an appropriate **`hipdnnStatus_t`** such as **`HIPDNN_STATUS_INTERNAL_ERROR`**, matching other descriptor **`finalize()`** failures when a required step does not complete—exact status TBD).

---

## 2. Motivation

- **Extensibility:** Today, engine ordering is fixed in backend code (`utilities::sortEngineIds`). New strategies (performance models, user policy, caches) require editing the core library.
- **Explicit device context:** Heuristics should depend on **passed-in** device facts, matching hipDNN’s reserved device-property attributes and reducing reliance on global HIP state or opaque handle internals for device semantics.
- **Consistency:** Engine plugins already load from trusted paths and validate API versions. Heuristic policies should follow the **same class of patterns** (shared libraries, version gates) **without sharing symbols or types** with engine plugins.
- **Interop:** A **C ABI** for heuristic plugins enables non–C++ consumers, stable binary boundaries, and future P/Invoke-style bindings without dragging in the engine plugin surface.

---

## 3. Baseline: current behavior

`EngineHeuristicDescriptor` today:

1. Obtains candidate engine IDs from `EnginePluginResourceManager::getApplicableEngineIds(opGraph)`.
2. Sorts them with `utilities::sortEngineIds` inside `finalize()` (for example prioritizing MIOPEN engine IDs and deprioritizing deterministic variants).
3. Exposes the ordered list through `HIPDNN_ATTR_ENGINEHEUR_RESULTS` as engine configuration descriptors.

This RFC preserves **equivalent deterministic ordering** as one **policy** (the “StaticOrdering” selector), shipped as a **backend built-in** that implements the heuristic plugin C ABI shape (function-pointer table) without a separate `.so` ([§10.1](#101-heuristicpluginmanager), [§16.2](#162-loaded-heuristic-policy-enumeration)). It runs **only when that policy appears** in **`orderedPolicyIds`** and is **not** applied as unconditional backend logic after the outer loop. **`Graph.preferred_engine_id`** is now a **frontend** concern — **`Graph::initializeEngineConfig`** reorders the backend's heuristic-ranked engine configs after **`finalize()`**. The previous file-based override (formerly **`HIPDNN_ENGINE_OVERRIDE_FILE`**) is replaced by **`HIPDNN_HEUR_CONFIG_PATH`** and implemented as the built-in **`SelectionHeuristic::Config`** policy ([§5.3.5](#535-the-config-built-in-policy)) — it lives in the ordered policy list like any other selector and declines when no rule matches.

---

## 4. Goals and non-goals

**Goals**

- **Outer-loop-only** orchestration: try an ordered list of **policy** plugins until one succeeds.
- **Explicit device facts** at the selection boundary: C++ **`DeviceProperties`** in the backend, serialized to a FlatBuffer and carried in **`hipdnnPluginConstData_t`** across the heuristic C ABI ([§13.2](#132-serialized-device-properties-flatbuffer)); plugins do not call `hipGetDevice` / `hipGetDeviceProperties` themselves.
- **Pluggable policies** loaded through a **`HeuristicPluginManager`** / **`HeuristicPluginResourceManager`** layer that mirrors **engine** plugin *mechanics* (paths, discovery, version checks, optional static path configuration) but uses the **heuristic-only** C ABI in [§8](#8-c-abi-for-heuristic-plugins).
- **Stable C ABI** for heuristic `.so` files: module metadata, **plugin-handle** lifecycle, **policy-descriptor** lifecycle, and selection entry points as in [§8](#8-c-abi-for-heuristic-plugins) (to be codified in headers such as **`HeuristicsPluginApi.h`**).
- **Aligned lifetimes:** **`hipdnnHeuristicHandle_t`** per loaded heuristic module per **`hipdnnHandle`** (with engine-plugin handles); **`hipdnnHeuristicPolicyDescriptor_t`** instances **owned by** **`EngineHeuristicDescriptor`** (one per policy slot), created when the policy list for that descriptor is established and destroyed with the descriptor.
- **Same logging path** as the rest of the backend (see [§12 Logging](#12-logging)).
- **Version compatibility** at load time (see [§11](#11-versioning-and-compatibility-checks)).
- **User-configurable policy order** with a documented **default** list of **policy name** strings ([§5.3](#53-ordered-policy-list-default-and-user-configuration)); **internal** resolution uses **`int64_t`** policy IDs (hashed names), aligned with hipDNN **engine** ID practice.
- **No hidden fallback ordering:** if no policy in **`orderedPolicyIds`** reports success, **`EngineHeuristicDescriptor::finalize()`** fails using the **normal backend mechanism** (`THROW_IF_*` macros / **`HipdnnException`** and a **`hipdnnStatus_t`**—see [§5.1](#51-single-orchestration-model-outer-loop) and [§14.2](#142-pseudocode-for-finalize-first-draft)); there is no second pass that sorts candidates inside the core library.

**Non-goals (this draft)**

- Inner “stage chains” inside a single policy (for example PerEngine → Config → StaticOrdering as mandatory sub-steps). Policy composition is expressed by **ordering multiple policy plugins** in the outer list, not by nested pipelines inside one policy plugin implementation.
- Mandating async selection in v1; the API is only **shaped** to allow it later (including **`Finalize`** + **`GetSortedEngineIds`** as a two-phase C selection API in [§8.9](#89-finalize-and-sorted-results)).
- Tunning is not approached in this RFC. Including auto-tuning and exhaustive tuning (both of which will require device access).

---

## 5. Design overview

### 5.1 Single orchestration model: outer loop

There is **one** primary control flow:

1. Build the list of **candidate engine IDs** from existing **engine** plugins (unchanged).
2. Obtain **serialized device properties** and the **serialized operation graph** (see [§6](#6-device-properties) and [§13](#13-serialized-graph-device-properties-and-graph-level-preferences)).
3. Resolve the **ordered list of policy plugin IDs** (`int64_t`, from user **policy name** strings via the shared name→ID hash—[§5.3](#53-ordered-policy-list-default-and-user-configuration)).
4. Ensure **`EngineHeuristicDescriptor`** owns one **plugin policy descriptor** per slot (see [§5.4](#54-two-tier-plugin-objects-handle-vs-policy-descriptor)); each slot binds to the **`hipdnnHeuristicHandle_t`** for that policy’s loaded module **or built-in** (from **`HeuristicPluginResourceManager`**). The policy descriptor is **created with** that handle ([§8.7](#87-policy-descriptor-per-slot-graph--candidate-ids)); **Finalize** consumes device facts **from that handle’s session state**, not from separate per-call device arguments on the policy.
5. After resolving **serialized device properties** (FlatBuffer bytes from **`DeviceProperties`**—[§6](#6-device-properties), [§13.2](#132-serialized-device-properties-flatbuffer)), call **`hipdnnHeuristicHandleSetDeviceProperties`** **once per distinct** **`hipdnnHeuristicHandle_t`** that appears among the policy slots (not once per slot iteration). That establishes the device context on the **handle**; heuristic code **queries** it as needed during **Finalize** (and any later selection work on that handle).
6. For each slot in order: **setEngineIds** (current candidates) and **serialized graph** on that slot’s policy descriptor, call **Finalize**; if the policy **wins**, read **getSortedIds** and **stop**; otherwise continue. The default order puts **`SelectionHeuristic::Config`** ([§5.3.5](#535-the-config-built-in-policy)) ahead of **`SelectionHeuristic::StaticOrdering`** so user-supplied **`HIPDNN_HEUR_CONFIG_PATH`** rules win when matched and the deterministic fallback runs otherwise.
7. If no policy succeeds, **`finalize()`** **fails** using the same pattern as other backend descriptor logic errors (for example **`THROW_IF_FALSE(success, HIPDNN_STATUS_INTERNAL_ERROR, …)`** throwing **`HipdnnException`**, analogous to **`GraphDescriptor::finalize()`** when verification fails—exact message and status code are implementation details). There is **no** additional built-in sort after the loop.

There is **no** separate inner registry of sub-stages inside a single policy plugin in this design. If a team wants “config then static ordering” inside one deliverable, they ship **one** policy plugin that implements that sequence internally, or they register two entries in the **outer** list.

### 5.2 Policy examples (illustrative)

| Policy (illustrative name) | Role |
|----------------------------|------|
| **Cache selector** | Reuse a persisted ordering keyed by graph fingerprint and device properties. |
| **ML selector** | Score candidates using a model or cost function. |
| **Rule selector** | Deterministic rules over graph features and `DeviceProperties`. |
| **StaticOrdering-style selector** | Reproduce current `sortEngineIds` behavior when listed in **`orderedPolicyIds`**; users often place it **last** in the default list so behavior matches today unless they omit it. |

These are **examples**; each plugin chooses a **canonical UTF-8 policy name** (for documentation and user config). The **stable identifier** across the loader, registry, and C ABI is the **`int64_t`** produced by the **same `policyNameToId` hash** as computational engines, returned from **`hipdnnHeuristicGetPolicyId`** ([§8](#8-c-abi-for-heuristic-plugins)).

### 5.3 Ordered policy list: default and user configuration

Policy order has **two representations**:

- **User / public surface:** an ordered list of **UTF-8 policy name** strings (attributes, env vars, optional frontend helpers)—the same **human-readable names** vendors document for their plugins (for example **`SelectionHeuristic::StaticOrdering`**).
- **Internal / loader / C ABI:** each name maps to a deterministic **`int64_t`** using **`hipdnn_data_sdk::utilities::policyNameToId`** (`PolicyNames.hpp`). The hash is FNV-1a, the same hash family used for computational **engine** IDs ([RFC 0003](0003_EngineIdDesign.md), `EngineNames.hpp::engineNameToId`); the two functions are siblings rather than aliases — they share the algorithm but are kept as separate symbols so policy and engine name spaces stay textually distinct in call sites. The outer loop operates on **`std::vector<int64_t> orderedPolicyIds`**; each element must match **`hipdnnHeuristicGetPolicyId`** for a loaded heuristic plugin (or a **built-in** policy registered under the same **`int64_t`** without a separate `.so`).

**Collision note:** Policy IDs and engine IDs are both `int64_t` hashes of **different string namespaces** in normal usage (policy names like `SelectionHeuristic::…` vs engine names like `MIOPEN_PLUGIN`). A numeric collision is theoretically possible; **context** (policy registry vs engine registry) keeps them separate.

#### 5.3.1 Well-known policy names and IDs

This draft standardizes one logical policy using a **well-known UTF-8 name** (the `SelectionHeuristic::` prefix is a **naming convention** in the string itself, not C++ language linkage or a shared type with the C++ `SelectionHeuristic` class). Its **canonical `int64_t` ID** is **`policyNameToId(name)`** (a plugin or built-in implementing this policy **must** return that exact value from **`hipdnnHeuristicGetPolicyId`**):

| Policy name string | Role |
|--------------------|------|
| **`SelectionHeuristic::Config`** | Reads the JSON file pointed to by **`HIPDNN_HEUR_CONFIG_PATH`**, walks the graph’s convolution nodes against the rule set, and on a match returns the candidate list with the matched engine moved to the front. **Declines** when the env var is unset, the file is missing, the graph cannot be parsed, no rule matches, or the matched engine is not in the candidate set — so the policy loop falls through to the next slot. Shipped as a **backend built-in** ([§10.1](#101-heuristicpluginmanager), [§5.3.5](#535-the-config-built-in-policy)); no separate `.so`. |
| **`SelectionHeuristic::StaticOrdering`** | Deterministic ordering aligned with today’s **`utilities::sortEngineIds`** (for example MIOPEN preference, deterministic engine last). Shipped as a **backend built-in** ([§10.1](#101-heuristicpluginmanager)); registered through the heuristic plugin C ABI shape but with no separate `.so`. Invoked only when listed in the resolved policy order. |

**`Graph.preferred_engine_id`** is **not** a policy. The frontend reorders the heuristic-ranked engine configs in **`Graph::initializeEngineConfig`** after **`EngineHeuristicDescriptor::finalize()`** has returned, so it is honored independently of the policy list.

**Mandatory registration.** A shipped heuristic plugin (or built-in adapter) that implements **`SelectionHeuristic::StaticOrdering`** **must** expose **`hipdnnHeuristicGetPolicyId`** returning exactly **`policyNameToId("SelectionHeuristic::StaticOrdering")`** — there is **no** alternate ID for that behavior. Optional **`hipdnnHeuristicGetPolicyName`** ([§8.2](#82-plugin-module-metadata)) **should** report the same canonical UTF-8 string; when present, the host **validates at load time** that **`policyNameToId(hipdnnHeuristicGetPolicyName()) == hipdnnHeuristicGetPolicyId()`** and **rejects** the module on mismatch (same idea as keeping engine IDs and registered engine names consistent).

**Overlapping IDs at load.** **`HeuristicPluginManager`** **must** reject a heuristic module whose **`hipdnnHeuristicGetPolicyId`** duplicates an **`int64_t`** already claimed by another **accepted** heuristic plugin or built-in, using the **same** duplicate-ID tracking as **`EnginePluginManager::validateBeforeAdding`** does for computational engine IDs ([§10.1](#101-heuristicpluginmanager), [§11](#11-versioning-and-compatibility-checks)). The built-in **`StaticOrdering`** is registered first, so a third-party module trying to claim the same ID will be rejected.

#### 5.3.2 Default policy order (user strings)

If the user does **not** override the list, the **configured** fallback name list is:

```text
{ "SelectionHeuristic::Config", "SelectionHeuristic::StaticOrdering" }
```

That list is also the **effective default order** seen by **`finalize()`** — the backend does **not** prepend or append anything. **`Config`** runs first so user-supplied **`HIPDNN_HEUR_CONFIG_PATH`** rules win when matched; otherwise it declines and **`StaticOrdering`** takes over. **`Graph.preferred_engine_id`** is **not** part of this list — the frontend handles it after the heuristic returns ([§5.3.5](#535-the-config-built-in-policy)).

The backend **hashes** each entry to build the internal **`orderedPolicyIds`** used in **`finalize()`**. **StaticOrdering** is expected to succeed for valid candidate sets when implemented correctly; if **no** listed policy succeeds (for example the user supplies a policy order that omits **StaticOrdering** and every listed policy declines), **`EngineHeuristicDescriptor::finalize()`** fails per [§5.1](#51-single-orchestration-model-outer-loop) — there is **no** backend fallback sort.

#### 5.3.3 How the user sets orderedPolicyIds

The user surface is mixed: the env-var path accepts UTF-8 **policy name** strings (and raw IDs as an escape hatch); the backend attribute path takes pre-hashed **`int64_t`** IDs directly. **Resolution order (highest precedence first):**

1. **Process environment** — **`HIPDNN_HEUR_POLICY_ORDER`** as a comma-separated list. Each token is either a policy name (UTF-8) or a signed decimal **`int64_t`** policy ID. A token is treated as an ID only when `std::strtoll` consumes the entire trimmed token; anything else (including names that happen to start with digits) is hashed through **`policyNameToId`**. Highest precedence so operators can override application-supplied lists at deploy time.
2. **Engine-heuristic descriptor (per finalize)** — User sets an explicit ordered list via the **`HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT`** backend attribute, typed **`HIPDNN_TYPE_INT64`** (array of pre-hashed **`int64_t`** policy IDs). The C ABI takes IDs rather than name strings because a `void* arrayOfElements` + `int64_t elementCount` surface does not cleanly carry variable-length string arrays; callers (typically the frontend) are expected to hash names through **`policyNameToId`** before calling `setAttribute`. Applies only to that **`EngineHeuristicDescriptor`** instance.
3. **Built-in fallback** — **`{ "SelectionHeuristic::Config", "SelectionHeuristic::StaticOrdering" }`** when neither env nor descriptor supplied a list. The names are hashed in the backend via **`policyNameToId`** at the call site. **`Graph.preferred_engine_id`** is honored by the frontend after the heuristic returns and is independent of this list ([§5.3.5](#535-the-config-built-in-policy)).

A **handle-level** default (e.g. **`hipdnnSetHeuristicPolicyOrder_ext(handle, ...)`**) is **not implemented** in the current backend. It is retained as a candidate extension in [§11](#11-options-and-tradeoffs) but is not part of the resolution order today.

Unknown IDs that survive resolution simply produce a null `_policySlots` entry inside the per-finalize sync (the policy is treated as "not loaded") and the outer loop skips them without raising an error.

#### 5.3.4 Relationship to `finalize()`

`EngineHeuristicDescriptor::finalize()` resolves an ordered **`std::vector<int64_t> orderedPolicyIds`** from the precedence chain above (env tokens are name-or-ID hashed per token; the descriptor attribute is already a raw ID array; the default fallback is hashed from its built-in name list via **`policyNameToId`**). It then ensures **policy descriptor** objects exist **one-to-one** with that list (recreating them if the list changed since last setup) and runs the outer loop in [§14.2](#142-pseudocode-for-finalize-first-draft). If the loop completes without a successful policy, **`finalize()`** does not mark the descriptor finalized and exits via **`HipdnnException`** like other backend failures.

#### 5.3.5 The Config built-in policy

**`SelectionHeuristic::Config`** (in `backend/src/heuristics/config/`) is a regular built-in policy that wraps the JSON-driven engine-override behavior previously exposed through **`HIPDNN_ENGINE_OVERRIDE_FILE`**. The env var has been **renamed to `HIPDNN_HEUR_CONFIG_PATH`** and the resolver is no longer a precursor — it sits in **`orderedPolicyIds`** like any other policy and **declines** when nothing applies, letting the policy loop fall through to the next slot (typically **`SelectionHeuristic::StaticOrdering`**).

**Behavior on each `Finalize`:**

1. If the candidate list is empty, decline.
2. Read **`HIPDNN_HEUR_CONFIG_PATH`** from the environment on every invocation (no process cache). If the variable is unset or the file cannot be opened/parsed, decline.
3. Parse the serialized graph buffer with the data-SDK FlatBuffer verifier. If the buffer is malformed or the graph has no nodes, decline.
4. Walk the graph’s **`ConvolutionFwd`** / **`ConvolutionBwdData`** / **`ConvolutionBwdFilter`** nodes against the rule’s op + tensor-shape patterns (exact dim/stride matching with `-1` wildcards, exact-bucket-before-wildcard-bucket within declaration order). On the first match, take the matched engine ID.
5. If the matched engine ID is **not** in the candidate list, decline.
6. Otherwise return the candidates with the matched engine moved to the front (other candidates preserve their original order) and report success (`*outApplied = 1`).

**Default placement.** The default policy list is **`{ "SelectionHeuristic::Config", "SelectionHeuristic::StaticOrdering" }`** ([§5.3.2](#532-default-policy-order-user-strings)) so user-supplied JSON rules win over the deterministic fallback. Users may freely omit, reorder, or replace either built-in. There is no implicit guarantee that **`Config`** runs — if the user supplies a custom policy order without it, the JSON file is simply not consulted for that **`finalize()`**.

**Graph-level `preferred_engine_id` is no longer in scope of this policy.** **`Graph::set_preferred_engine_id_ext`** is honored by the **frontend** as a post-hoc reorder of the heuristic-ranked engine configs in **`Graph::initializeEngineConfig`**, after **`EngineHeuristicDescriptor::finalize()`** has returned. The backend heuristic loop is unaware of it, and the **`Config`** built-in does **not** consult it.

**`StaticOrdering`** retains no override guarantee — it is a normal default-ordered fallback ([§5.3.2](#532-default-policy-order-user-strings)) and may be omitted or reordered by the user. Both built-ins are shipped through **`HeuristicPlugin::createBuiltIn(populateFunctionTable(), label)`** at **`HeuristicPluginManager`** construction time ([§10.1](#101-heuristicpluginmanager)); neither has a separate `.so` and the user cannot unload them.

### 5.4 Two-tier plugin objects: handle vs policy descriptor

| Tier | C typedef (illustrative) | Lifetime | Holds |
|------|---------------------------|----------|--------|
| **Plugin handle** | **`hipdnnHeuristicHandle_t`** | Same pattern as other plugin handles on **`hipdnnHandle`**; created when the heuristic module is paired with the handle, destroyed with handle teardown | Plugin **session** state (caches, tuning, scratch, **parsed device facts** after **`SetDeviceProperties`**); receives **`SetDeviceProperties`** (serialized device-properties FlatBuffer in **`hipdnnPluginConstData_t`**), **queried** by heuristics during **`Finalize`** and related work |
| **Policy descriptor** | **`hipdnnHeuristicPolicyDescriptor_t`** | **Owned by** **`EngineHeuristicDescriptor`**; one per entry in **`orderedPolicyIds`**; **created** when the descriptor’s policy list is established (**implementation choice:** on attribute set, bind, or lazy at first **`finalize()`**) and **destroyed** with the heuristic descriptor | **Candidate engine IDs** + **serialized graph** for the current selection; **Finalize** / **GetSortedEngineIds** result. **Created with** a **`hipdnnHeuristicHandle_t`** ([§8.7](#87-policy-descriptor-per-slot-graph--candidate-ids)); **device properties** for selection are read from **that handle’s** session state (**`SetDeviceProperties`** on the handle—[§8.6](#86-plugin-handle-lifecycle)), not passed again on the policy descriptor. |

**Threading:** Plugin handles are **single-thread only** (**not** thread-safe). Parallelism uses **multiple** **`hipdnnHandle`** instances (each with its own heuristic plugin handles) or host-side serialization (policy TBD). All calls for a given **`hipdnnHeuristicHandle_t`** and its dependent policy descriptors on that thread must follow this contract.

**Design-note mapping:** **createHandle** / **destroyHandle** / **setDeviceProperties(Handle\*)** → [§8.6](#86-plugin-handle-lifecycle); **createPolicyDescriptor** / **destroyPolicyDescriptor** / **setEngineIds** / **Finalize** / **getSortedIds** → [§8.7](#87-policy-descriptor-per-slot-graph--candidate-ids)–[§8.9](#89-finalize-and-sorted-results). Documents that used the term **engineDescriptor** for the plugin-side object mean **policy descriptor** here, not hipDNN’s **`EngineDescriptor`** for computational engines.

---

## 6. Device properties

### 6.1 `DeviceProperties` struct

A single, plain structure carries the device facts needed for selection. It is filled by the backend (or by a future attribute override) and passed into every policy invocation.

```cpp
struct DeviceProperties
{
    int    deviceId            = -1;
    int    multiProcessorCount = 0;
    size_t totalGlobalMem      = 0;
    // Optional later: architecture name / ID, wavefront size, etc.
};
```

The **stable C ABI** does **not** expose a fixed-layout struct for device facts. The backend fills **`DeviceProperties`**, **serializes** it with the **device-properties FlatBuffer schema** ([§13.2](#132-serialized-device-properties-flatbuffer)), and passes it using the **existing** **`hipdnnPluginConstData_t`** struct from **`PluginApiDataTypes.h`** (the same type the plugin SDK already uses for serialized graph bytes and other const buffers in **engine** plugins—not a separate **`const uint8_t*`** + **`size_t`** pair on the heuristic API). The host passes **`const hipdnnPluginConstData_t*`** to **`hipdnnHeuristicHandleSetDeviceProperties`** ([§8.6](#86-plugin-handle-lifecycle)). Plugins **parse** that payload from **handle** session state (for example when servicing **`Finalize`** on a policy descriptor **created with** that handle) with the same data-SDK / codegen as the graph schema.

### 6.2 Default acquisition: `queryDeviceProperties()`

The backend may implement a helper (in an anonymous namespace or a small utility unit) that maps from HIP:

```cpp
namespace
{
DeviceProperties queryDeviceProperties()
{
    DeviceProperties out{};
    int device = 0;
    if(hipGetDevice(&device) != hipSuccess)
    {
        return out;
    }
    out.deviceId = device;

    hipDeviceProp_t hipProps{};
    if(hipGetDeviceProperties(&hipProps, device) != hipSuccess)
    {
        return out;
    }
    out.multiProcessorCount = hipProps.multiProcessorCount;
    out.totalGlobalMem      = static_cast<size_t>(hipProps.totalGlobalMem);
    return out;
}
} // namespace
```

**Important:** Plugins **must not** call this helper; they consume only the **serialized device-properties** buffer the host installs on the session handle via **`hipdnnHeuristicHandleSetDeviceProperties`** ([§8.6](#86-plugin-handle-lifecycle)), after **verifying** and **parsing** it per [§13.2](#132-serialized-device-properties-flatbuffer) when **Finalize** (or other selection logic) **reads** it from handle state.

### 6.3 Proposed override: descriptor-level device properties

**First-draft proposal:** Extend the engine-heuristic descriptor so callers can optionally supply device properties explicitly, using the already-reserved attribute space (for example `HIPDNN_ATTR_ENGINEHEUR_DEVICEPROP` and/or a dedicated `HIPDNN_BACKEND_DEVICEPROP_DESCRIPTOR` type when implemented).

**Semantics (proposal):**

- If the heuristic descriptor has **no** device-property override set, `EngineHeuristicDescriptor::finalize()` uses `queryDeviceProperties()` once.
- If the user sets device properties on the descriptor, **that** structure is used instead of querying HIP in finalize.

This keeps **one** logical pathway into heuristics (`DeviceProperties` in the backend, serialized at the plugin boundary) while allowing tests, multi-GPU hosts, and API parity with “device props, not handle” designs.

---

## 7. SelectionHeuristic interface

`SelectionHeuristic` is a **C++ facade** used by the backend for **one policy slot** on an **`EngineHeuristicDescriptor`**. It wraps an opaque **`hipdnnHeuristicPolicyDescriptor_t`** ([§8.7](#87-policy-descriptor-per-slot-graph--candidate-ids)) created with the **`hipdnnHeuristicHandle_t`** for that policy’s module. **Session state** (caches, etc.) lives in the plugin **behind the handle**, not in this wrapper.

Device properties are **not** set on the facade: the host **builds the serialized device-properties FlatBuffer** (from resolved `DeviceProperties`—[§6](#6-device-properties), [§13.2](#132-serialized-device-properties-flatbuffer)), wraps it in **`hipdnnPluginConstData_t`**, and calls **`hipdnnHeuristicHandleSetDeviceProperties`** on each **distinct** plugin handle that will participate in this **`finalize()`**, **before** invoking **`Finalize`** on any policy descriptor **created with** that handle. The plugin **stores** or **parses** that payload on the handle; **`Finalize`** (and internal heuristic logic) **queries** device facts **via the bound handle**, not via extra device arguments on the policy API. Ordering relative to **`PolicyDescriptorCreate(..., plugin_handle)`** is: handle exists → **`SetDeviceProperties`** on that handle for this finalize’s device context → policy inputs (**`SetEngineIds`** / **`SetSerializedGraph`**) → **`Finalize`**.

```cpp
class SelectionHeuristic
{
public:
    // Candidate engine IDs from EnginePluginResourceManager; mirrors setEngineIds ([§8.8](#88-policy-inputs-engine-ids-and-serialized-graph)).
    void setEngineIds(const std::vector<int64_t>& engineIds);

    // Serialized operation graph as hipdnnPluginConstData_t (FlatBuffer; [§13.1](#131-serialized-graph)).
    void setSerializedGraph(const SerializedGraph& serializedGraph);

    // Runs applicability + selection inside the plugin; true => policy won the outer loop ([§8.9](#89-finalize-and-sorted-results)).
    bool finalize();

    // Valid after finalize() returned true; mirrors GetSortedEngineIds.
    std::vector<int64_t> getSortedEngineIds();
};
```

**Notes**

- **Logging** matches **engine** plugins: not part of these method signatures; see [§12](#12-logging).
- **`finalize` + `getSortedEngineIds`** mirror the C ABI two-phase pattern so a future revision can perform async work in **`Finalize`** without changing names.
- **`EngineHeuristicDescriptor`** owns **`SelectionHeuristic`** (or equivalent) instances **one per** resolved policy slot; lifetimes match [§5.4](#54-two-tier-plugin-objects-handle-vs-policy-descriptor).

---

## 8. C ABI for heuristic plugins

This section defines the **stable C-language ABI** for heuristic (selection policy) shared libraries. It is **orthogonal** to the **engine** plugin ABI: a single `.so` is either an **engine** plugin or a **heuristic** plugin, **not both**. Heuristic libraries **do not** export `hipdnnPluginGetName`, `hipdnnPluginGetType`, or any **engine** entry points from `PluginApi.h` / `EnginePluginApi.h`—only the symbols below (final names live in **`HeuristicsPluginApi.h`** or equivalent).

### 8.1 Design principles

- All exported symbols use **C linkage** (`extern "C"` from C++ implementations). **`int64_t`** in signatures uses **`stdint.h`** / **`<cstdint>`** (same as engine plugin APIs).
- Opaque serialized payloads (**device properties**, **operation graph**) cross the heuristic C ABI only as **`const hipdnnPluginConstData_t*`**, using the **existing** **`hipdnnPluginConstData_t`** definition in **`PluginApiDataTypes.h`** (same **`ptr`** / **`size`** fields as **`EnginePluginApi.h`** graph and config arguments). The ABI does **not** introduce parallel raw-buffer parameters (for example ad-hoc **`const uint8_t*`** plus **`size_t`**) for these payloads.
- Versioning and rejection of incompatible plugins follow the same **ideas** as engine plugins (major API compatibility), but the **API version string** and **symbol set** are **heuristic-specific** ([§11](#11-versioning-and-compatibility-checks)).
- Plugins **must not** call HIP, mutate hipDNN graph descriptors, or execute engines; they only **read** inputs described here and **write** reordered engine IDs.
- The host (backend) owns output buffers unless the API explicitly transfers ownership.
- **Logging** is configured only via **`hipdnnHeuristicSetLoggingCallback`** / **`SetLogLevel`** at **module load time** ([§8.2](#82-plugin-module-metadata)); handle and policy-descriptor entry points do not take a logger argument ([§12](#12-logging)).

### 8.2 Plugin module metadata

Each heuristic `.so` exports the following (names are illustrative; implementations use the `hipdnnHeuristic` prefix in headers).

| Function | Purpose |
|----------|---------|
| `hipdnnHeuristicGetApiVersion(const char** version)` | Semantic version of **this C ABI** (for example `"1.0.0"`). Host rejects load on **major** mismatch. |
| `hipdnnHeuristicGetPolicyId(int64_t* policy_id)` | Stable **`int64_t`** policy identifier: **must** equal **`policyNameToId(canonical_utf8_name)`** for the plugin’s documented policy name (same hash as computational engine IDs—[§5.3](#53-ordered-policy-list-default-and-user-configuration), [RFC 0003](0003_EngineIdDesign.md)). The host matches this value against the resolved **`orderedPolicyIds`** after hashing user-supplied name strings. |
| `hipdnnHeuristicGetPolicyName(const char** policy_name)` | Optional. **NUL-terminated UTF-8** canonical name (same string the vendor tells users to put in **`HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT`**). For logging and enumeration; **not** required for matching if **`GetPolicyId`** is implemented correctly. When exported, the host **validates at load** that **`policyNameToId(*policy_name) ==`** the value from **`GetPolicyId`** ([§5.3.1](#531-well-known-policy-names-and-ids), [§11](#11-versioning-and-compatibility-checks)). Omit from minimal plugins if the host derives display names from a static registry. |
| `hipdnnHeuristicGetPluginVersion(const char** version)` | Plugin implementation version (informational). |
| `hipdnnHeuristicSetLoggingCallback(hipdnnCallback_t cb)` | Registers the consumer logging callback; optional `hipdnnHeuristicSetLogLevel(hipdnnSeverity_t)` mirroring engine plugin behavior. |
| `hipdnnHeuristicGetLastErrorString(const char** msg)` | Per-thread last error after a failed call; pointer valid only for immediate use (same contract as `hipdnnPluginGetLastErrorString`). |

The host identifies heuristic libraries by **which loader** opened them and by successful resolution of **`hipdnnHeuristicGetApiVersion`** (and related required symbols), not by reusing **`hipdnnPluginGetType`**.

### 8.3 Plugin handle (session object)

Each loaded heuristic module exposes a **plugin handle**: an opaque **session** object. The host creates **one** **`hipdnnHeuristicHandle_t`** per **loaded heuristic `.so`** associated with a given **`hipdnnHandle`**, using the **same timing and storage pattern** as other hipDNN plugin handles, so every future selection path can **forward that plugin-specific handle** and **stateful tracking** can live in the plugin behind it.

**Threading contract:** A plugin handle is **not** thread-safe (**single-thread only**). Concurrent use requires **separate** **`hipdnnHandle`** instances (and thus separate heuristic plugin handles) or host-side serialization (policy TBD).

```c
typedef struct hipdnnHeuristicHandle_opaque* hipdnnHeuristicHandle_t;
```

### 8.4 Serialized device properties (no POD struct)

Device facts cross the heuristic **C ABI** as **opaque serialized bytes** carried in the **existing** **`hipdnnPluginConstData_t`** wrapper (**`const hipdnnPluginConstData_t*`** at the API boundary), not as a public C struct and not as a standalone **`const uint8_t*`** + **`size_t`** pair. That matches the **serialized graph** path ([§8.8](#88-policy-inputs-engine-ids-and-serialized-graph), [§13](#13-serialized-graph-device-properties-and-graph-level-preferences)) and existing engine plugin APIs: the **function signatures** stay stable while the **FlatBuffer schema** evolves (new optional fields, same **`hipdnnPluginConstData_t`** entry point).

- **Schema:** A **device-properties table** (name TBD, for example `HeuristicDeviceProperties`) lives alongside the operation-graph schema in the **data-SDK**; exact `.fbs` definition and file identifier are implementation details. v1 fields align with the C++ **`DeviceProperties`** in [§6.1](#61-deviceproperties-struct) (for example `device_id`, `multi_processor_count`, `total_global_mem_bytes`); additional facts (architecture name, wavefront size, etc.) are **additive** in the schema without changing **`hipdnnHeuristicHandleSetDeviceProperties`**.
- **Contract:** The host supplies a **complete FlatBuffer** rooted at that table; plugins **must** reject buffers that fail **`Verifier`** checks or use an **incompatible schema version** (per data-SDK / file-identifier rules—same class of checks as for the graph buffer).
- **Tradeoff:** Plugins depend on FlatBuffer + generated types for **both** graph and device inputs; the win is **no struct-layout ABI breaks** when device facts grow.

### 8.5 Status codes

Heuristic C entry points return **`hipdnnPluginStatus_t`**, the same typedef as **engine** plugin APIs (**`PluginApiDataTypes.h`** in the plugin SDK). **`HeuristicsPluginApi.h`** includes (or re-exports) that header; heuristic **`.so`** files **must not** include **`HipdnnStatus.h`**, so the plugin ABI stays free of backend layering—the same rule as engine plugins.

Use the existing **`HIPDNN_PLUGIN_STATUS_*`** enumerators for ordinary outcomes—for example **`HIPDNN_PLUGIN_STATUS_SUCCESS`**, **`HIPDNN_PLUGIN_STATUS_BAD_PARAM`**, **`HIPDNN_PLUGIN_STATUS_INVALID_VALUE`**, **`HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR`**, and **`HIPDNN_PLUGIN_STATUS_ALLOC_FAILED`**. These are generic and align in purpose with backend **`HIPDNN_STATUS_*`** / **`hipdnnStatus_t`**; the host maps between plugin and backend status the same way it does for engine plugins.

**Extensions:** When a heuristic-only outcome needs a distinct code—for example **decline** / **not applicable** on **`Finalize`** when **`out_applied == 0`**—add a new **`HIPDNN_PLUGIN_STATUS_*`** value to **`hipdnnPluginStatus_t`** (exact name TBD, e.g. **`HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE`**). Append new ordinals **after** the existing **`HIPDNN_PLUGIN_STATUS_*`** values so the enum stays backward-compatible. Do **not** introduce a separate heuristic-only status typedef (for example **`hipdnnHeuristicStatus_t`**); extend **`hipdnnPluginStatus_t`** instead.

Each heuristic C entry point returns **`hipdnnPluginStatus_t`** unless noted.

**No buffer-size error for sorted IDs:** Do **not** add **`BUFFER_TOO_SMALL`** (or **`BAD_PARAM_SIZE_INSUFFICIENT`**) for **`hipdnnHeuristicPolicyGetSortedEngineIds`**. That call takes **`out_capacity`** and **`*out_count`**; the plugin writes **at most** **`out_capacity`** IDs and sets **`*out_count`** to how many were written. **`HIPDNN_PLUGIN_STATUS_SUCCESS`** is returned even when the caller’s buffer holds only a **prefix** of the full ordering ([§8.9](#89-finalize-and-sorted-results)). This differs from backend APIs that require one buffer to fit an entire payload and return **`HIPDNN_STATUS_BAD_PARAM_SIZE_INSUFFICIENT`** when the query size is too small.

### 8.6 Plugin handle lifecycle

Illustrative names (headers may shorten or alias symbols—exact spelling TBD):

```c
hipdnnPluginStatus_t hipdnnHeuristicHandleCreate(hipdnnHeuristicHandle_t* out_handle);
hipdnnPluginStatus_t hipdnnHeuristicHandleDestroy(hipdnnHeuristicHandle_t handle);

hipdnnPluginStatus_t hipdnnHeuristicHandleSetDeviceProperties(
    hipdnnHeuristicHandle_t handle,
    const hipdnnPluginConstData_t* device_props_serialized);
```

- **`Create`** / **`Destroy`** correspond to **createHandle** / **destroyHandle**: the host invokes them when binding a heuristic module to a **`hipdnnHandle`** (alongside other plugin-handle setup).
- **`SetDeviceProperties`** corresponds to **setDeviceProperties(Handle\*)**: the host supplies **serialized device-properties** only via **`const hipdnnPluginConstData_t*`** ([§8.4](#84-serialized-device-properties-no-pod-struct), [§13.2](#132-serialized-device-properties-flatbuffer)); plugins **must not** call HIP. The backend resolves **`DeviceProperties`** via **`queryDeviceProperties()`** or descriptor override ([§6](#6-device-properties)), **builds** the FlatBuffer, and fills the **`hipdnnPluginConstData_t`** pointed to by **`device_props_serialized`** (**`ptr`** and **`size`**); plugins verify and parse those bytes (FlatBuffer **`Verifier`** / accessors) exactly as for **`hipdnnHeuristicPolicySetSerializedGraph`**. The host then updates **each distinct** **`hipdnnHeuristicHandle_t`** used by the current policy slots **before** **`hipdnnHeuristicPolicyFinalize`** on any descriptor **created with** that handle ([§5.1](#51-single-orchestration-model-outer-loop), [§14.2](#142-pseudocode-for-finalize-first-draft)). The **`device_props_serialized`** view must remain valid for the duration of the call; the plugin may copy if it needs the payload after return. Heuristic implementations **read** this state from the handle during **`Finalize`** (and as needed elsewhere on that session)—the policy descriptor does **not** carry a parallel device-properties buffer in this design.

### 8.7 Policy descriptor (per-slot graph + candidate IDs)

A **policy descriptor** is a second opaque object. It holds **per–`EngineHeuristicDescriptor` slot** state: **candidate engine IDs**, **serialized graph** bytes, and internal state through **`Finalize`**.

**Naming:** This document uses **policy descriptor** in the C ABI to avoid confusion with hipDNN’s **`EngineDescriptor`** (computational engine). Informal design notes that say **engineDescriptor** for the plugin-side object refer to **policy descriptor** here.

```c
typedef struct hipdnnHeuristicPolicyDescriptor_opaque* hipdnnHeuristicPolicyDescriptor_t;
```

**Lifecycle:** The backend **`EngineHeuristicDescriptor` owns** one **`hipdnnHeuristicPolicyDescriptor_t`** per entry in **`orderedPolicyIds`**. Objects are **created** when the descriptor’s policy list is established (on attribute set, bind, or lazily at first **`finalize()`**—implementation choice) and **destroyed** when the **`EngineHeuristicDescriptor`** is destroyed (**plugin policy descriptors and the hipDNN heuristic descriptor die together**). **`hipdnnHeuristicPolicyDescriptorCreate`** takes the **`hipdnnHeuristicHandle_t`** for that slot’s module; that **binds** the policy to the handle **before** **`Finalize`**, so selection code can treat the handle as the source of **device-properties** session state (**`SetDeviceProperties`**—[§8.6](#86-plugin-handle-lifecycle)).

```c
hipdnnPluginStatus_t hipdnnHeuristicPolicyDescriptorCreate(
    hipdnnHeuristicHandle_t plugin_handle,
    hipdnnHeuristicPolicyDescriptor_t* out_desc);

hipdnnPluginStatus_t hipdnnHeuristicPolicyDescriptorDestroy(
    hipdnnHeuristicPolicyDescriptor_t desc);
```

### 8.8 Policy inputs: engine IDs and serialized graph

```c
hipdnnPluginStatus_t hipdnnHeuristicPolicySetEngineIds(
    hipdnnHeuristicPolicyDescriptor_t desc,
    const int64_t* engine_ids,
    size_t engine_id_count);

hipdnnPluginStatus_t hipdnnHeuristicPolicySetSerializedGraph(
    hipdnnHeuristicPolicyDescriptor_t desc,
    const hipdnnPluginConstData_t* serialized_graph);
```

These correspond to **setEngineIds** and to storing **graph details** on the plugin policy descriptor.

### 8.9 Finalize and sorted results

**Two-phase** selection (matches **`SelectionHeuristic::finalize`** / **`getSortedEngineIds`**; leaves room for future async **`Finalize`**):

```c
/* *out_applied == 1 => policy won; host then calls GetSortedEngineIds.
   *out_applied == 0 => not applicable or declined; host continues outer loop. */
hipdnnPluginStatus_t hipdnnHeuristicPolicyFinalize(
    hipdnnHeuristicPolicyDescriptor_t desc,
    int32_t* out_applied);

hipdnnPluginStatus_t hipdnnHeuristicPolicyGetSortedEngineIds(
    hipdnnHeuristicPolicyDescriptor_t desc,
    int64_t* out_ids,
    size_t out_capacity,
    size_t* out_count);
```

**Contract**

- Calls on a policy descriptor that hang off a given **`hipdnnHeuristicHandle_t`** must occur on a **thread consistent** with that handle’s **single-thread** contract ([§8.3](#83-plugin-handle-session-object)).
- **`Finalize`** assumes **current** device-properties bytes were applied to **that policy’s** **`hipdnnHeuristicHandle_t`** via **`hipdnnHeuristicHandleSetDeviceProperties`** earlier in the same **`EngineHeuristicDescriptor::finalize()`** (or whenever the host last updated device context for that handle). Plugins **query** that handle state as needed; the host does **not** pass device properties again on **`hipdnnHeuristicPolicyFinalize`**.
- **`SetEngineIds` / `SetSerializedGraph` / `Finalize`:** candidate IDs come from **`EnginePluginResourceManager`**; output IDs **must** be a **permutation or subset** of the **SetEngineIds** input (host validates).
- **`GetSortedEngineIds`:** valid only after **`Finalize`** with **`out_applied == 1`**. The plugin writes the first **`min(full_sorted_length, out_capacity)`** engine IDs into **`out_ids`**, sets **`*out_count`** to that count, and returns **`HIPDNN_PLUGIN_STATUS_SUCCESS`** (via **`hipdnnPluginStatus_t`**). A smaller **`out_capacity`** is **not** an error—only a prefix is returned; callers that need the full ordering must supply a large enough buffer (for example at least the candidate count from **`SetEngineIds`**).

### 8.10 Host integration (C++ backend)

**`HeuristicPlugin`** resolves **both** handle and policy symbols via `dlsym` (or equivalent).

**`HeuristicPluginResourceManager`** (per **`hipdnnHandle`**):

1. After loading each heuristic `.so`, calls **`hipdnnHeuristicSetLoggingCallback`** (and optionally **`SetLogLevel`**)—same timing as engine plugins ([§12](#12-logging)).
2. Creates **`hipdnnHeuristicHandle_t`** via **`hipdnnHeuristicHandleCreate`** for each accepted module, stores it **like other plugin handles**, and exposes lookup (for example **`getHeuristicHandleForPolicyId`**) for the backend.
3. Does **not** own **`EngineHeuristicDescriptor`**-scoped policy descriptors; those are created with **`hipdnnHeuristicPolicyDescriptorCreate(plugin_handle, …)`** when the heuristic descriptor (re)builds its policy slot table.

**`EngineHeuristicDescriptor`** holds **`std::vector<std::unique_ptr<SelectionHeuristic>>`** (or equivalent): each **`SelectionHeuristic`** wraps one **`hipdnnHeuristicPolicyDescriptor_t`**, bound to the **`hipdnnHeuristicHandle_t`** for that slot’s policy module. **`finalize()`** resolves **`orderedPolicyIds`**, ensures policy slots (and thus handle bindings) are in place, then **SetDeviceProperties** (serialized device-properties FlatBuffer in **`hipdnnPluginConstData_t`**) **once per distinct** **`hipdnnHeuristicHandle_t`** among those slots, then for each slot **SetEngineIds** / **SetSerializedGraph** / **Finalize** / **GetSortedEngineIds** as in [§14.2](#142-pseudocode-for-finalize-first-draft). If no slot succeeds, **`finalize()`** aborts via **`HipdnnException`** (normal backend error path); there is **no** post-loop **`utilities::sortEngineIds`**.

### 8.11 ABI evolution

- **Patch/minor:** additive optional C functions; **additive** fields in the **FlatBuffer** schemas for the **graph** and **device properties** ([§13](#13-serialized-graph-device-properties-and-graph-level-preferences)), with plugins and host agreeing via **schema / file-identifier** rules in the data-SDK (same pattern as today for the graph).
- **Major:** breaking changes to **required** C entry points or to the **meaning** of existing serialized fields without a compatibility story; incompatible plugins fail **`validateBeforeAdding`**-style checks at load time ([§11](#11-versioning-and-compatibility-checks)). The **C ABI** intentionally avoids a public **`hipdnnHeuristicDeviceProperties_t`** layout so **device facts** can evolve inside the FlatBuffer without a struct-layout break.

---

## 9. Policy plugins and the outer loop

Each **heuristic policy plugin** implements one selection strategy and is loaded by **`HeuristicPluginManager`** (see [§10](#10-heuristicpluginmanager-and-resource-layer)). The backend maintains an **ordered list** of **`int64_t` policy IDs** (resolved from user **name** strings—[§5.3](#53-ordered-policy-list-default-and-user-configuration)) and **owns** one **policy descriptor** + **`SelectionHeuristic`** wrapper **per slot** on **`EngineHeuristicDescriptor`** ([§5.4](#54-two-tier-plugin-objects-handle-vs-policy-descriptor)). **Before** the per-slot loop (after slots are synced and **`DeviceProperties`** is resolved), the backend calls **`hipdnnHeuristicHandleSetDeviceProperties`** **once per distinct** **`hipdnnHeuristicHandle_t`** referenced by those slots, with the same **serialized device-properties** FlatBuffer ([§6](#6-device-properties), [§13.2](#132-serialized-device-properties-flatbuffer)). Plugins keep that state on the **handle** and **query** it during **`Finalize`**. Then for each slot in order during **`finalize()`**, the backend:

1. Resolves the **`hipdnnHeuristicHandle_t`** for that slot’s module (from **`HeuristicPluginResourceManager`**); skips the slot if unknown / failed to load—policy TBD. (Device properties on that handle were already updated in the pre-loop step if the handle is non-null.)
2. Calls **`setEngineIds`** and **`setSerializedGraph`** on the slot’s **`SelectionHeuristic`** (policy descriptor).
3. Calls **`finalize()`**; if false, **continue** with the **original** candidate engine list unchanged for the next slot.
4. On success, replaces candidates with **`getSortedEngineIds()`** and **breaks**.
5. On failure, continues; the next policy always starts from the **original** candidate list (unless a future RFC defines chaining semantics).

**First-success wins:** The first applicable policy that reports success defines the final ordering. Later policies are not consulted.

**Exhausted list:** If every slot is skipped, declined, or errors without a successful policy, **`EngineHeuristicDescriptor::finalize()`** fails with **`HipdnnException`** (same **`THROW_IF_*` / status pattern as other backend descriptor `finalize()` paths**—see [§5.1](#51-single-orchestration-model-outer-loop)); the implementation does **not** fall back to **`utilities::sortEngineIds`** or any other ordering outside **`orderedPolicyIds`**.

**Read-only contract:** Heuristic plugins **must not** mutate hipDNN graph state, engine plugin handles, device memory, or global HIP device selection. They only **read** `SerializedGraph` bytes, **device-properties** data obtained from the **bound** **`hipdnnHeuristicHandle_t`** (installed via **`SetDeviceProperties`**—parsed per [§13.2](#132-serialized-device-properties-flatbuffer)), and the **candidate engine ID list** they receive, and they **output** a reordered subset or permutation of those candidates (subject to backend validation—see [§18](#18-risks-and-open-questions)).

---

## 10. HeuristicPluginManager and resource layer

This section mirrors the **two-layer** pattern used for engines today: a **manager** that loads and validates shared libraries, and a **resource manager** held by the handle that exposes high-level operations to the rest of the backend.

**Engine stack (today, for reference):**

- **`EnginePluginManager`** — searches paths, loads `.so` files, runs **`validateBeforeAdding`** (API major, duplicate engine IDs), owns **`EnginePlugin`** instances.
- **`EnginePluginResourceManager`** — constructed with `std::shared_ptr<EnginePluginManager>`; provides **`getApplicableEngineIds`**, execution, workspace queries, stream propagation, etc.; obtained from the handle via **`getPluginResourceManager()`**.

**Heuristic stack (proposal):**

### 10.1 `HeuristicPluginManager`

Analogous to **`EnginePluginManager`**:

- Extends the same **plugin manager base** pattern (shared library load, symbol resolution, lifecycle).
- Uses a **separate search path** from engine plugins (for example `hipdnn_plugins/heuristics/` and/or a dedicated env var such as `HIPDNN_HEURISTIC_PLUGIN_DIR`—exact names TBD).
- Resolves **heuristic-only** symbols from [§8](#8-c-abi-for-heuristic-plugins); does **not** use **`EnginePlugin`** symbol tables.
- Implements **`validateBeforeAdding`**-style checks before accepting a plugin—**parallel to `EnginePluginManager`**: **API major** match (via **`hipdnnHeuristicGetApiVersion`**); **unique** **`int64_t` policy ID** (via **`hipdnnHeuristicGetPolicyId`**, tracked in a set so a second module cannot register the same ID—same pattern as duplicate engine IDs in `EnginePluginManager`); optional **ID ↔ name** consistency when **`hipdnnHeuristicGetPolicyName`** is exported; and any additional rules from [§11](#11-versioning-and-compatibility-checks). One plugin may export **multiple** policies; each policy ID is checked individually for global uniqueness.
- Owns **`HeuristicPlugin`** wrappers that bind the C ABI (handle + policy symbols) and expose **`HandleCreate` / `HandleDestroy`** to the resource manager.
- **Registers backend built-ins at construction.** Before any external `.so` discovery, the manager calls **`registerBuiltIns()`** (in `backend/src/heuristics/BuiltInHeuristics.cpp`) to register every shipped built-in heuristic. Today there are two — **`SelectionHeuristic::Config`** (in `backend/src/heuristics/config/`, driven by **`HIPDNN_HEUR_CONFIG_PATH`** — see [§5.3.5](#535-the-config-built-in-policy)) and **`SelectionHeuristic::StaticOrdering`** (in `backend/src/heuristics/static_ordering/`). A built-in implements the **same heuristic plugin C ABI shape** (a populated **`HeuristicPluginFunctionTable`**) as a third-party `.so` and is wrapped via **`HeuristicPlugin::createBuiltIn(populateFunctionTable(), source_label)`**, so the rest of the manager (lookup by **`int64_t` policy ID**, version validation, duplicate-ID rejection) treats it identically to a loaded module. There is **no** separate **`BaseHeuristicsPlugin`** library to ship and **no** override of **`loadPlugins`** to inject default paths. Built-ins always exist regardless of the caller's **`hipdnnPluginLoadingMode_ext_t`**, including **`HIPDNN_PLUGIN_LOADING_ABSOLUTE`**.
- **`Graph.preferred_engine_id` is not a built-in concern.** It is honored by the **frontend** as a post-hoc reorder of the heuristic-ranked engine configs (see **`Graph::initializeEngineConfig`**), independent of any backend policy or built-in.

### 10.2 `HeuristicPluginResourceManager`

Analogous to **`EnginePluginResourceManager`**:

- Holds **`std::shared_ptr<HeuristicPluginManager> _pm`** (same structural idea as **`EnginePluginResourceManager::_pm`**).
- **Static path configuration (optional mirror of engine):** methods such as **`setHeuristicPluginPaths` / `getHeuristicPluginPaths`** with the same **loading mode** semantics as **`EnginePluginResourceManager::setPluginPaths`** (absolute vs additive paths, **no path change while handles are active** unless the same restrictions as engine plugins apply).
- **`static std::shared_ptr<HeuristicPluginResourceManager> create()`** (or equivalent factory) builds the resource manager after constructing the **`HeuristicPluginManager`**.
- **Instance API (read-only selection):** for example:
  - **`getHeuristicHandleForPolicyId(int64_t policyId)`** — returns the stored **`hipdnnHeuristicHandle_t`** (or null) for that loaded module; created at handle setup via [§8.6](#86-plugin-handle-lifecycle).
  - **`resolveHeuristicPolicyOrder(descriptor, handle)`** (free function or member) — implements the precedence in [§5.3.3](#533-how-the-user-sets-orderedpolicyids), merges **policy name** strings, **hashes** them to **`int64_t`**, and returns **`orderedPolicyIds`** for **`finalize()`**.
  - **`getHeuristicPluginInfos()`** — optional, parallel to **`getEngineInfos()`** (plugin version, **`int64_t` policy ID**, optional **policy name** string) for diagnostics.
  - **`getLoadedHeuristicPluginFiles(...)`** — optional, parallel to **`getLoadedPluginFiles`** on the engine resource manager.
- **Logging:** when heuristic `.so` files are loaded, call **`hipdnnHeuristicSetLoggingCallback`** (and optionally **`SetLogLevel`**) as defined in [§8.2](#82-plugin-module-metadata).

### 10.3 Handle integration

**Proposal:** **`hipdnnHandle`** exposes **`getHeuristicPluginResourceManager()`** alongside **`getPluginResourceManager()`**, returning a **`std::shared_ptr<HeuristicPluginResourceManager>`** created at handle construction (same era as the engine resource manager). At that time the manager creates and stores **`hipdnnHeuristicHandle_t`** values per loaded heuristic module ([§8.6](#86-plugin-handle-lifecycle)). If heuristic plugins are optional in early implementations, the pointer may refer to an empty manager that only supplies **built-in** handles/adapters without loading external `.so` files.

### 10.4 Relationship to `EnginePluginResourceManager`

- **Candidate engine IDs** always come from **`EnginePluginResourceManager::getApplicableEngineIds`** (unchanged).
- **Ordering** is applied by **`SelectionHeuristic`** instances **owned by** **`EngineHeuristicDescriptor`**, each wrapping a **`hipdnnHeuristicPolicyDescriptor_t`** bound to a **`hipdnnHeuristicHandle_t`** from **`HeuristicPluginResourceManager`**.
- The two subsystems stay **separate**: heuristic plugins **do not** register engine IDs and **do not** execute graphs; engine plugins **do not** implement the heuristic C ABI.

---

## 11. Versioning and compatibility checks

Follow the same **spirit** as `EnginePluginManager::validateBeforeAdding` in the backend: reject incompatible plugins at load time with a clear error.

**Proposed checks**

1. **Heuristic C ABI major:** Parse **`hipdnnHeuristicGetApiVersion`**; **major** must match the backend’s expected heuristic API major (analogous to engine plugins comparing `plugin.apiVersion()` major to `HIPDNN_BACKEND_VERSION_MAJOR`, but using the **heuristic** version string, not the engine plugin API version).
2. **Policy ID uniqueness:** Two loaded heuristic modules **must not** return the same **`int64_t`** from **`hipdnnHeuristicGetPolicyId`**. Enforce with the **same** “insert into a set, throw on duplicate” pattern as **`EnginePluginManager::validateBeforeAdding`** / **`actionAfterAdding`** for engine IDs ([§5.3.1](#531-well-known-policy-names-and-ids)).
3. **Policy ID ↔ optional policy name:** If **`hipdnnHeuristicGetPolicyName`** is provided, **`policyNameToId`** of the returned UTF-8 string **must** equal **`hipdnnHeuristicGetPolicyId`**; otherwise the loader **rejects** the module (catches mistaken or overlapping well-known implementations early—[§5.3.1](#531-well-known-policy-names-and-ids)).
4. **Binary compatibility:** Document minimum backend / data-SDK versions per heuristic plugin release (align with project-wide versioning RFCs under `docs/rfcs/`), including expectations for **graph** and **device-properties** FlatBuffer schemas ([§13](#13-serialized-graph-device-properties-and-graph-level-preferences)).

**On failure:** Do not register the plugin; log via the shared logging path ([§12](#12-logging)); continue loading other policies if policy loading is best-effort, or fail handle creation if strict mode is required (policy TBD).

**ABI evolution** is summarized in [§8.11](#811-abi-evolution).

---

## 12. Logging

### 12.1 Current state

Today, `hipdnnHandle` (**`struct hipdnnHandle`**) exposes stream and plugin resource manager functionality but **does not** expose a dedicated logger or `getLogger()` accessor. Backend code typically logs through **`hipdnn_backend::logging`** (`HIPDNN_BACKEND_LOG_*` macros in `backend/src/logging/Logging.hpp`), which ultimately dispatches via **`hipdnn_data_sdk::logging`** using the **global** user callback registered for the process.

Engine plugins may receive the logging callback through the existing plugin infrastructure (`PluginBase::setLoggingCallback`), keeping plugin logs on the **same** user-visible path as the backend.

### 12.2 How heuristic code obtains “the same logger” (same pattern as engine plugins)

Engine plugins do **not** take a logger on each API call. After the `.so` is loaded, **`PluginManagerBase::loadPluginFromFile`** calls **`plugin->setLoggingCallback(logging::backendLoggingCallback)`** once, then **`setLogLevel`**, so the plugin **stores** the callback and uses it internally (`backend/src/plugin/PluginCore.hpp`).

**Heuristic plugins should follow the same model:**

1. **Backend (C++) code** in the heuristic path uses **`HIPDNN_BACKEND_LOG_*`** / the same global SDK dispatch as today—no logger argument on **`SelectionHeuristic`** or handle methods ([§7](#7-selectionheuristic-interface)).
2. **Heuristic `.so` code:** Immediately after loading a heuristic library (in **`HeuristicPluginManager`** / resource manager, mirroring **`loadPluginFromFile`**), the host calls **`hipdnnHeuristicSetLoggingCallback`** ([§8.2](#82-plugin-module-metadata)) with the same **`logging::backendLoggingCallback`** (or an equivalent that forwards to the consumer’s registered path), then optionally **`hipdnnHeuristicSetLogLevel`**. Handle and policy-descriptor entry points in [§8.6](#86-plugin-handle-lifecycle)–[§8.9](#89-finalize-and-sorted-results) **do not** take a logging parameter.
3. **C ABI:** Only the module-level **`SetLoggingCallback`** / **`SetLogLevel`** symbols carry logging configuration—**not** the per-instance selection functions.

This matches engine plugins: **supplied and used outside the call signatures** of selection/engine operations, via one-time registration at load time.

---

## 13. Serialized graph, device properties, and graph-level preferences

Heuristic plugins receive **two** structured inputs as **FlatBuffer** payloads in **`hipdnnPluginConstData_t`** over the C ABI: the **operation graph** (per policy descriptor) and **device properties** (installed on the **plugin handle** via **`hipdnnHeuristicHandleSetDeviceProperties`** and **read** by the plugin during **`Finalize`** and other work on that session—[§8.6](#86-plugin-handle-lifecycle), [§8.9](#89-finalize-and-sorted-results)). Both use the **same** serialization stack (data-SDK schemas, `flatbuffers::Verifier`, generated accessors) so **payloads** can evolve without changing **function signatures** or adding public C structs beyond the existing const-buffer wrapper.

### 13.1 Serialized graph

`GraphDescriptor` already maintains a **FlatBuffer** serialized graph and exposes it via **`getSerializedGraph()`**, which returns **`hipdnnPluginConstData_t`** (see **`PluginApiDataTypes.h`**). The heuristic framework treats that buffer as the canonical **`SerializedGraph`** input to policies—**no alternate wire format** for the graph in v1.

Policies that need structured access may parse the FlatBuffer using existing data-SDK generated types, subject to version rules for the graph schema. The C ABI passes this buffer as **`const hipdnnPluginConstData_t*`** via **`hipdnnHeuristicPolicySetSerializedGraph`** ([§8.8](#88-policy-inputs-engine-ids-and-serialized-graph)), matching **`GraphDescriptor::getSerializedGraph()`** and engine plugin graph arguments.

### 13.2 Serialized device properties (FlatBuffer)

**Device facts** use a **second** FlatBuffer schema (root table name TBD, colocated with the graph schema in the data-SDK). The backend builds this buffer from the C++ **`DeviceProperties`** struct ([§6.1](#61-deviceproperties-struct)) after **`queryDeviceProperties()`** or a descriptor override ([§6.3](#63-proposed-override-descriptor-level-device-properties)).

- **C ABI:** **`hipdnnHeuristicHandleSetDeviceProperties(handle, const hipdnnPluginConstData_t*)`** ([§8.6](#86-plugin-handle-lifecycle)); same **`hipdnnPluginConstData_t`** pointer-lifetime pattern as **`hipdnnHeuristicPolicySetSerializedGraph`** for the duration of the call. The host applies this **to each distinct handle** used by the current policy slots **before** **`PolicyFinalize`** on descriptors **created with** that handle ([§5.1](#51-single-orchestration-model-outer-loop)); plugins **query** the stored device context from the handle as needed (including inside **`Finalize`**), rather than receiving device bytes again on the policy descriptor.
- **Plugins:** Verify and parse with the generated **device-properties** types; treat unknown or default-filled **optional** fields as “not provided” when the host runs an older backend.
- **Evolution:** New device fields are **additive** in the `.fbs` definition; heuristic C ABI **major** bumps are reserved for incompatible **API surface** changes, not for ordinary schema extension (aligned with graph-schema practice).

### 13.3 Graph-level preferences (for example `preferred_engine_id`)

The graph model already carries fields such as **`preferred_engine_id`** when built from operation descriptors. **Responsibility for honoring `preferred_engine_id` belongs to the frontend**, not to any backend policy or plugin. After **`EngineHeuristicDescriptor::finalize()`** returns the heuristic-ranked engine configs, **`Graph::initializeEngineConfig`** in **hipdnn_frontend** moves the preferred config (if its global engine index matches) to the front of the candidate list. Because the reorder happens outside the backend, the user's choice of **`HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT`** does not affect whether **`preferred_engine_id`** is honored.

The file-driven override formerly carried by **`HIPDNN_ENGINE_OVERRIDE_FILE`** has been **renamed to `HIPDNN_HEUR_CONFIG_PATH`** and is implemented as a regular built-in policy — **`SelectionHeuristic::Config`** ([§5.3.5](#535-the-config-built-in-policy)) — that lives in the policy list and operates on the serialized graph buffer like any other policy.

---

## 14. EngineHeuristicDescriptor integration

### 14.1 Responsibilities

`EngineHeuristicDescriptor` continues to:

- Own the operation graph reference and heuristic mode attributes (see [§15](#15-public-api-notes)).
- Discover **candidate** engines via **`EnginePluginResourceManager`**.
- Expose results through **`HIPDNN_ATTR_ENGINEHEUR_RESULTS`**.

Additionally it:

- Obtains **`HeuristicPluginResourceManager`** from the handle (proposal: **`getHeuristicPluginResourceManager()`**) for **`hipdnnHeuristicHandle_t`** lookup per policy module.
- **Owns** **`SelectionHeuristic`** (policy-descriptor) objects **one per** resolved policy slot ([§5.4](#54-two-tier-plugin-objects-handle-vs-policy-descriptor)); (re)creates them when **`orderedPolicyIds`** changes.
- Resolves **`DeviceProperties`** (override or `queryDeviceProperties()`) and **serializes** it for **`SetDeviceProperties`** on each **distinct** **`hipdnnHeuristicHandle_t`** used by its policy slots **before** the per-slot **`Finalize`** loop ([§13.2](#132-serialized-device-properties-flatbuffer)).
- Obtains serialized graph bytes from the finalized graph descriptor.
- Runs the **outer policy loop** described in [§5.1](#51-single-orchestration-model-outer-loop), using **`orderedPolicyIds`** from [§5.3](#53-ordered-policy-list-default-and-user-configuration).
- On success, stores the final ordered engine IDs for result construction; on **exhausted list without success**, aborts **`finalize()`** via **`HipdnnException`** (normal backend error path—[§14.2](#142-pseudocode-for-finalize-first-draft)).

**Frontend:** Callers that use **hipdnn_frontend** still **`backendFinalize`** the heuristic descriptor and read **`HIPDNN_ATTR_ENGINEHEUR_RESULTS`** (for example via **`getEngineConfigs`**) as today; how that lines up with **policy order** and **runtime discovery** of loaded policy IDs is in [§16](#16-frontend-api-flow-mirror-engine-selection).

### 14.2 Pseudocode for `finalize()` (first draft)

```text
finalize():
  validate descriptor state (graph set, mode set, not already finalized)

  handle    = graph.getHandle()
  engineRm  = handle.getPluginResourceManager()
  heurRm    = handle.getHeuristicPluginResourceManager()  // proposal: parallel to engineRm

  candidates = engineRm.getApplicableEngineIds(graph)

  serializedGraph = graph.getSerializedGraph()  // hipdnnPluginConstData_t; graph must be usable for heuristics

  devProps = userDeviceOverride if set else queryDeviceProperties()
  devicePropsSerialized = serializeDevicePropertiesFlatBuffer(devProps)  // §13.2; hipdnnPluginConstData_t; host-owned for this finalize()

  orderedPolicyIds = resolveHeuristicPolicyOrder(thisDescriptor)
    // §5.3.3 precedence: HIPDNN_HEUR_POLICY_ORDER env > HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT > built-in fallback
    // fallback names { "SelectionHeuristic::Config", "SelectionHeuristic::StaticOrdering" } if no override
    // implementation:
    //   1. pick the highest-precedence non-empty source
    //   2. env: each token is name OR signed-decimal int64 ID (name → policyNameToId, ID → as-is)
    //      descriptor: array is already int64 IDs (caller hashed names via policyNameToId)
    //      fallback: hash the built-in names via policyNameToId at the call site
    // No backend-injected entries — duplicates from the user-supplied list are preserved as-is.
    // Handle-level default (§11) is not implemented.

  syncPolicySlots(thisDescriptor, orderedPolicyIds, heurRm)
    // Ensure one SelectionHeuristic (hipdnnHeuristicPolicyDescriptor_t) per slot, each created with
    // the hipdnnHeuristicHandle_t for that policy's module; destroy/recreate if list changed.
    // PolicyFinalize reads device facts from that bound handle (§8.7 / §8.9).

  for each distinct non-null hipdnnHeuristicHandle_t h among policy slots for this finalize():
    hipdnnHeuristicHandleSetDeviceProperties(
        h,
        &devicePropsSerialized)  // hipdnnPluginConstData_t { ptr, size }; FlatBuffer; §8.6 / §13.2 — once per handle, not per slot

  success = false
  for each slot i aligned with orderedPolicyIds:
    pluginHandle = heurRm.getHeuristicHandleForPolicyId(orderedPolicyIds[i])
    if pluginHandle is null:
      continue

    selection = thisDescriptor.policySlot(i)  // wraps policy descriptor; §8.7
    selection.setEngineIds(candidates)
    selection.setSerializedGraph(serializedGraph)
    if not selection.finalize():  // §8.9; not applicable or declined
      continue
    candidates = selection.getSortedEngineIds()
    success = true
    break

  if not success:
    // Abort: same as other descriptor finalize() failures — THROW_IF_FALSE / HipdnnException
    // with hipdnnStatus_t (e.g. HIPDNN_STATUS_INTERNAL_ERROR). Do not call utilities::sortEngineIds.
    return via exception; descriptor stays not finalized

  store candidates as _engineIds
  mark finalized
```

---

## 15. Public API notes

- **`HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT`:** Ordered list of pre-hashed **`int64_t`** policy IDs for **`EngineHeuristicDescriptor`**, overriding the built-in default ([§5.3](#53-ordered-policy-list-default-and-user-configuration)). Attribute type: **`HIPDNN_TYPE_INT64`** (array of IDs). The backend stores IDs as-is — callers (typically the frontend) hash policy names via **`policyNameToId`** before calling `setAttribute` because the `void*` + `int64_t elementCount` ABI does not cleanly carry variable-length string arrays. When this list changes between finalizes (a fresh descriptor with a different list), the per-finalize sync step recreates the owned **policy descriptor** objects ([§5.4](#54-two-tier-plugin-objects-handle-vs-policy-descriptor)).
- **Handle-level override (deferred):** A per-handle default (e.g. **`hipdnnSetHeuristicPolicyOrder_ext(handle, ...)`**) was sketched in earlier drafts of this RFC but is **not implemented**. Today the resolution chain is env > descriptor attribute > built-in fallback only. Retained here as a candidate extension if a per-handle default surface is needed later.
- **`HIPDNN_HEUR_POLICY_ORDER` (optional env):** Comma-separated tokens; each token is a UTF-8 policy name **or** a signed decimal **`int64_t`** policy ID (per-token disambiguation by full-string `strtoll`). **Highest** precedence among user overrides so operators can override application-supplied lists at deploy time ([§5.3.3](#533-how-the-user-sets-orderedpolicyids)).
- **`HIPDNN_ATTR_ENGINEHEUR_MODE`:** Today the backend supports a narrow heuristic mode surface. This RFC does **not** remove the attribute; a future mapping might define default **policy order** per mode, or deprecate mode once **`HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT`** is sufficient. That decision is left open in this draft.
- **`HIPDNN_ATTR_ENGINEHEUR_DEVICEPROP`:** Proposed as the user-facing override for [§6.3](#63-proposed-override-descriptor-level-device-properties) when the descriptor type and setters are implemented.
- **No requirement for new enums** per new policy: adding a policy is **deployment + registry order** (names in env/config; **`int64_t`** IDs from **`policyNameToId`** at registration time), not necessarily a new public enum value. The well-known **names** **`SelectionHeuristic::Config`** and **`SelectionHeuristic::StaticOrdering`** are **strings** in env/config and docs; their **`int64_t`** values are **`policyNameToId(...)`**, not enum members. Both are shipped as **backend built-ins** ([§10.1](#101-heuristicpluginmanager)), so they are always available without loading any external `.so`; users may freely omit or reorder either in **`HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT`** or **`HIPDNN_HEUR_POLICY_ORDER`**. **`HIPDNN_HEUR_CONFIG_PATH`** is consulted by the **`Config`** built-in only when that policy is in the resolved order. **`Graph.preferred_engine_id`** is independent of this attribute — the frontend reorders engine configs after **`finalize()`** returns ([§5.3.5](#535-the-config-built-in-policy)).
- **Headers:** Publish **`HeuristicsPluginApi.h`** (name TBD) in **plugin_sdk** (or a sibling package) containing the types and declarations in [§8](#8-c-abi-for-heuristic-plugins), without including **engine** plugin API headers.

**Frontend:** End-to-end flow from application / **hipdnn_frontend** through heuristic **`finalize()`** and policy configuration is described in [§16](#16-frontend-api-flow-mirror-engine-selection).

---

## 16. Frontend API flow (mirror engine selection)

This section ties [§5.3](#53-ordered-policy-list-default-and-user-configuration), [§10](#10-heuristicpluginmanager-and-resource-layer), and [§15](#15-public-api-notes) to the **hipdnn_frontend** pattern used for **computational** engines today, so the path from user code to policy execution is explicit.

### 16.1 Engine configs after heuristic finalize today

The frontend already drives **engine configuration selection** through the **engine heuristic descriptor** and **`HIPDNN_ATTR_ENGINEHEUR_RESULTS`**:

1. The application builds the operation graph and obtains an **`EngineHeuristicDescriptor`** (backend descriptor) for that graph and **`hipdnnHandle`**.
2. The caller **`backendFinalize`**s the heuristic descriptor (or follows the equivalent backend sequence that produces heuristic results).
3. **`hipdnn_frontend::detail::getEngineConfigs`** (see `frontend/include/hipdnn_frontend/detail/GraphDetail.hpp`) queries **`HIPDNN_ATTR_ENGINEHEUR_RESULTS`**: first the **count** of engine configuration descriptors, then fills **`ScopedHipdnnBackendDescriptor`** instances and reads each engine’s **`HIPDNN_ATTR_ENGINE_GLOBAL_INDEX`** after finalizing each engine config.

So **which engines appear as options** for a given graph is **dynamic**: it comes from **loaded engine plugins** and **`EnginePluginResourceManager::getApplicableEngineIds`** (unchanged in this RFC), materialized in heuristic results only after **`EngineHeuristicDescriptor::finalize()`** succeeds (baseline in [§3](#3-baseline-current-behavior), integration in [§14](#14-engineheuristicdescriptor-integration)).

### 16.2 Loaded heuristic policy enumeration

**Policy IDs** (heuristic) and **engine IDs** are both **`int64_t`** name hashes but serve **different roles**: heuristic **`hipdnnHeuristicGetPolicyId`** identifies a **selection policy** module; engine plugins expose **engine** IDs for execution. Each loaded **heuristic** module — and each **backend built-in** registered through the same C ABI shape — contributes one stable **`int64_t`** from **`hipdnnHeuristicGetPolicyId`** ([§8.2](#82-plugin-module-metadata)). The **ordered list** the outer loop uses (**`orderedPolicyIds`**) is **`int64_t`**, derived from user-supplied **policy name** strings per [§5.3.3](#533-how-the-user-sets-orderedpolicyids). Applications and tools still need a **runtime view of which policies are actually available** for this **`hipdnnHandle`** (or process) — built-ins plus successfully loaded `.so` modules — analogous to **`getEngineInfos()`** (see [§10.2](#102-heuristicpluginresourcemanager)).

**Built-ins as the canonical adapter example.** **`SelectionHeuristic::StaticOrdering`** is the reference built-in (`backend/src/heuristics/static_ordering/`). Its source unit exports **`populateFunctionTable()`**, which fills a **`HeuristicPluginFunctionTable`** struct with function pointers matching the heuristic plugin C ABI surface from [§8](#8-c-abi-for-heuristic-plugins) (handle lifecycle, policy descriptor lifecycle, set-engine-ids, set-serialized-graph, finalize, get-sorted-ids). At **`HeuristicPluginManager`** construction time, **`registerBuiltIns()`** wraps that table via **`HeuristicPlugin::createBuiltIn(populateFunctionTable(), source_label)`** and inserts the resulting **`std::shared_ptr<HeuristicPlugin>`** into the same registry that `dlopen`-loaded modules use. Policy-ID lookup, version validation, duplicate-ID rejection, and per-handle plugin-handle creation all flow through that registry without distinguishing built-ins from external plugins. New built-in adapters follow the same recipe — add a `populateFunctionTable()` and a line in **`registerBuiltIns()`**.

**Proposal:**

- **Backend / handle scope:** **`HeuristicPluginResourceManager`** exposes a query (exact name TBD) that returns **available heuristic policy metadata** — both built-ins and accepted external modules: at minimum the **`int64_t` policy ID**; optionally the **canonical policy name** (from **`hipdnnHeuristicGetPolicyName`** if present, else from a host-maintained id→name map for well-known policies), **plugin implementation version**, install path (for `.so`-backed entries), and a flag distinguishing **built-in** from **loaded module** (mirroring **`getHeuristicPluginInfos()`** / **`getLoadedHeuristicPluginFiles`** in [§10.2](#102-heuristicpluginresourcemanager)). A thin **`hipdnn…_ext`** C wrapper is optional if the C API surface should stay symmetric with other handle queries.
- **Frontend:** Add a small helper (for example **`getLoadedHeuristicPolicyInfos(handle)`**) that forwards to that query, documented next to **`getEngineConfigs`** and **`Graph::initializeEngineConfig`** so policy **configuration** (**name** strings in **`HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT`**) can be validated or presented in UIs without hard-coding numeric IDs.

**How the list is determined dynamically:** Built-ins are present unconditionally as registered by **`registerBuiltIns()`** ([§10.1](#101-heuristicpluginmanager)). External entries are **only modules that passed load-time checks** ([§11](#11-versioning-and-compatibility-checks)): heuristic search paths ([§10.1](#101-heuristicpluginmanager)), **`validateBeforeAdding`**-style validation, and **unique `int64_t` policy ID** registration. Neither set depends on the operation graph. User-supplied entries — name tokens in **`HIPDNN_HEUR_POLICY_ORDER`** or pre-hashed IDs in **`HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT`** — are resolved against the combined registry when the list is resolved ([§5.3.3](#533-how-the-user-sets-orderedpolicyids)). Unknown IDs are not rejected up front; they produce a null `_policySlots` entry during the per-finalize sync and the outer loop skips them ([§9](#9-policy-plugins-and-the-outer-loop)).

### 16.3 End-to-end flow with policy order

1. **Optional:** The application calls the enumeration helper to list **available policies** (**`int64_t` IDs** and optional **names**) for loaded heuristic plugins plus any **built-in** adapters registered under the same scheme without a separate `.so`.
2. The application sets **`HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT`** on the **`EngineHeuristicDescriptor`** when overriding defaults, and/or relies on handle-level default or environment ([§5.3.3](#533-how-the-user-sets-orderedpolicyids)).
3. **Unchanged from today:** **`backendFinalize`** on **`EngineHeuristicDescriptor`** runs the outer policy loop ([§14.2](#142-pseudocode-for-finalize-first-draft)). **Candidate engine IDs** still come from **`EnginePluginResourceManager::getApplicableEngineIds`** ([§10.4](#104-relationship-to-enginepluginresourcemanager)).
4. **Unchanged from today:** The application reads **`HIPDNN_ATTR_ENGINEHEUR_RESULTS`** to obtain **ordered engine configuration descriptors**; **`Graph::initializeEngineConfig`** continues to pick among those configs using **preferred** or **default engine id** on the graph. **`Graph.preferred_engine_id`** is honored by the **frontend** as a post-hoc reorder of those engine configs in **`Graph::initializeEngineConfig`** — the backend heuristic loop does not see it ([§13.3](#133-graph-level-preferences-for-example-preferred_engine_id)).

In short: **policies** are **configured** by an ordered **policy name** string list (public surface) plus an **optional discovery API**; the backend uses **`int64_t`** policy IDs internally for matching. **Engines** remain **graph-dependent** and appear under **`HIPDNN_ATTR_ENGINEHEUR_RESULTS`** only after heuristic **`finalize()`** succeeds.

---

## 17. Testing

- **Unit tests** for each policy with **synthetic serialized device-properties** (FlatBuffer built from test `DeviceProperties`) and small **graph** FlatBuffers (no GPU required where possible).
- **Regression test** asserting that when **`SelectionHeuristic::StaticOrdering`** is in effect (for example via the default **policy name** order), ordering matches current `utilities::sortEngineIds` for a fixed candidate list.
- **Failure test** asserting that when **`orderedPolicyIds`** is empty, all IDs are unknown/skipped, or every policy declines, **`finalize()`** fails via **`HipdnnException`** / the same status path as other descriptor **`finalize()`** errors (no silent sort fallback).
- **Integration tests** with real graphs and devices when GPU is available.
- **ABI / loader tests** that load a minimal mock heuristic `.so`, verify **`hipdnnHeuristicGetApiVersion`**, **`hipdnnHeuristicGetPolicyId`** ( **`int64_t`** matches **`policyNameToId`** for the plugin’s documented name), **`HandleCreate` / `HandleDestroy`**, **`PolicyDescriptorCreate` / `Destroy`**, **`Finalize` / `GetSortedEngineIds`**, and reject wrong major versions; **negative tests** that a second module with a **duplicate** **`hipdnnHeuristicGetPolicyId`** and a module whose optional **`hipdnnHeuristicGetPolicyName`** does not hash to its policy ID are **rejected at load** (same spirit as **`EnginePluginManager`** duplicate engine IDs—[§5.3.1](#531-well-known-policy-names-and-ids), [§11](#11-versioning-and-compatibility-checks)).
- **Policy order tests** that assert the effective default **name** list is **`{ "SelectionHeuristic::Config", "SelectionHeuristic::StaticOrdering" }`** (the fallback in [§5.3.2](#532-default-policy-order-user-strings) — backend does not inject any other entries), descriptor override wins over handle which wins over env, and unknown **names** (after hashing) are handled per policy.
- **Built-in registration tests** that assert **`HeuristicPluginManager`** registers both **`SelectionHeuristic::Config`** and **`SelectionHeuristic::StaticOrdering`** as built-ins at construction (no `.so` loaded), the registered IDs equal **`policyNameToId("SelectionHeuristic::Config")`** and **`policyNameToId("SelectionHeuristic::StaticOrdering")`** respectively, and both are available regardless of **`hipdnnPluginLoadingMode_ext_t`** (including **`HIPDNN_PLUGIN_LOADING_ABSOLUTE`**) — see [§10.1](#101-heuristicpluginmanager).
- **`SelectionHeuristic::Config` built-in tests** ([§5.3.5](#535-the-config-built-in-policy)) covering: empty candidates → declines; **`HIPDNN_HEUR_CONFIG_PATH`** unset / file missing / unparsable → declines; serialized graph buffer invalid or graph has no nodes → declines; no rule matches → declines; matched engine ID not in candidates → declines; matched rule moves the matched engine to the front while preserving the order of other candidates; the JSON file is re-read on every invocation (no process cache).
- **Engine override config rule-matching tests** covering exact-dim and wildcard (`-1`) matching, exact-bucket-before-wildcard-bucket cross-partition ordering, declaration-order tiebreak within a bucket, stride patterns (exact, wildcard element, empty pattern matches any), op-name and tensor-count rejection, and JSON loading (valid, missing file, env-var unset).
- **Frontend `preferred_engine_id` tests** that assert **`Graph::initializeEngineConfig`** reorders the heuristic-ranked engine configs to put the requested engine first when its global index appears among the configs returned by **`HIPDNN_ATTR_ENGINEHEUR_RESULTS`**, and falls back to the heuristic order otherwise.
- **Lifetime tests** that assert destroying **`EngineHeuristicDescriptor`** invokes **`hipdnnHeuristicPolicyDescriptorDestroy`** for every owned slot (and does not leak plugin handles owned by **`hipdnnHandle`**).
- **Enumeration tests** that assert the handle-scoped **loaded heuristic policy** query ([§16.2](#162-loaded-heuristic-policy-enumeration)) matches modules that passed **`validateBeforeAdding`**-style checks, returns **`int64_t`** IDs consistent with **`hipdnnHeuristicGetPolicyId`**, and that optional frontend helpers return consistent metadata.

---

## 18. Risks and open questions

- **Policy list syntax:** Comma-separated env vars and string array attributes need clear rules for embedded commas and empty tokens; validate unknown **policy names** (no loaded module with matching **`hipdnnHeuristicGetPolicyId`**) strictly or leniently (skip vs error). If skipping unknown entries leaves no successful policy, **`finalize()`** fails (no backend fallback sort).
- **Duplicate policy IDs in `orderedPolicyIds`:** Two slots may reference the same loaded module (same **`hipdnnHeuristicHandle_t`**) but still require **distinct** **`hipdnnHeuristicPolicyDescriptor_t`** instances; whether duplicates are allowed, deduplicated, or rejected is policy TBD.
- **Failure modes:** If the cache or ML policy returns partial or invalid engine IDs, should the backend validate against `candidates` before accepting success?
- **Async selection:** When implemented, thread-safety of **`hipdnnHeuristicHandle_t`** / **`hipdnnHeuristicPolicyDescriptor_t`**, interaction with the **single-thread handle** rule ([§8.3](#83-plugin-handle-session-object)), and lifetime of **serialized graph** and **serialized device-properties** buffers, must be specified.
- **Invalid device-properties buffers:** If **`SetDeviceProperties`** receives malformed bytes, plugins should return **`HIPDNN_PLUGIN_STATUS_BAD_PARAM`** (or an equivalent **`HIPDNN_PLUGIN_STATUS_*`** code); whether the backend retries, fails **`finalize()`**, or skips the policy is implementation TBD.
- **`PluginApi.h` comment:** Today’s reference to heuristic plugins implementing `PluginApi.h` + `HeuristicsPluginApi.h` should be updated in code/docs to match **this** RFC: heuristic plugins use **only** the heuristic C ABI header, not **`PluginApi.h`**.

---

## 19. Glossary

| Term | Meaning |
|------|--------|
| **Engine plugin** | Shared library providing engines and execution; **distinct C ABI** from heuristic plugins. |
| **Heuristic / selection policy plugin** | Shared library implementing one outer-loop selection strategy via the C ABI in [§8](#8-c-abi-for-heuristic-plugins). |
| **Heuristic C ABI** | `extern "C"` symbol set: module metadata (**`int64_t` policy ID** via **`hipdnnHeuristicGetPolicyId`**), **`hipdnnHeuristicHandle_t`** and **`hipdnnHeuristicPolicyDescriptor_t`** lifecycle, selection functions ([§8](#8-c-abi-for-heuristic-plugins)). |
| **HeuristicPluginManager** | Loads and validates heuristic `.so` files **and** registers backend built-ins (via **`registerBuiltIns()`**) at construction time; analogous to **EnginePluginManager** but **heuristic-only** symbols. |
| **HeuristicPluginResourceManager** | Handle-scoped facade for heuristic plugins (built-ins and external modules); stores **`hipdnnHeuristicHandle_t`** per module, paths; analogous to **EnginePluginResourceManager**. |
| **Outer loop** | Ordered list of policies; first applicable successful policy wins. If none succeed, **`finalize()`** fails via **`HipdnnException`** (normal backend error path); no built-in sort after the loop. |
| **Policy name** | UTF-8 string in **`HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT`**, env, or handle defaults; hashed with **`policyNameToId`** to form the **`int64_t`** used in **`orderedPolicyIds`** and matched against **`hipdnnHeuristicGetPolicyId`**. |
| **Policy ID (heuristic)** | **`int64_t`** stable identifier for one heuristic policy (built-in or external); **must** equal **`policyNameToId(policy_name)`** for that policy’s canonical name ([§5.3](#53-ordered-policy-list-default-and-user-configuration)). |
| **orderedPolicyIds** | Resolved **`std::vector<int64_t>`** for one **`finalize()`**, each element the hash of a user-configured **policy name**. The backend does **not** inject any entries. Effective default when nothing is overridden: **`{ policyNameToId("SelectionHeuristic::Config"), policyNameToId("SelectionHeuristic::StaticOrdering") }`** ([§5.3](#53-ordered-policy-list-default-and-user-configuration), [§5.3.2](#532-default-policy-order-user-strings)). |
| **`SelectionHeuristic::Config`** | Backend built-in heuristic in `backend/src/heuristics/config/` that consults **`HIPDNN_HEUR_CONFIG_PATH`** for JSON rules matching the graph's convolution shapes to a preferred engine ID. On a match it returns the candidates with the matched engine moved to the front; on every miss path (env unset, file missing, no rule match, matched engine not in candidates) it **declines** so the policy loop falls through to the next slot ([§5.3.5](#535-the-config-built-in-policy)). |
| **`HIPDNN_HEUR_CONFIG_PATH`** | Environment variable consulted by **`SelectionHeuristic::Config`** (formerly **`HIPDNN_ENGINE_OVERRIDE_FILE`**) on every **`Finalize`**. Points at a JSON rule file that maps convolution op + tensor-shape patterns to a preferred engine ID. Re-read each invocation; no process cache. |
| **Built-in heuristic** | Backend-shipped policy implemented as a populated **`HeuristicPluginFunctionTable`** (the heuristic plugin C ABI shape) and wrapped via **`HeuristicPlugin::createBuiltIn(...)`** at **`HeuristicPluginManager`** construction time. No `.so` is loaded. Today **`SelectionHeuristic::Config`** and **`SelectionHeuristic::StaticOrdering`** ship this way ([§10.1](#101-heuristicpluginmanager), [§16.2](#162-loaded-heuristic-policy-enumeration)). |
| **DeviceProperties** | C++ struct of device facts in the backend; serialized to FlatBuffer and passed in **`hipdnnPluginConstData_t`** for the heuristic C ABI ([§13.2](#132-serialized-device-properties-flatbuffer)). Plugins do not call HIP. |
| **SelectionHeuristic** | C++ facade over **`hipdnnHeuristicPolicyDescriptor_t`** for one policy **slot** on **`EngineHeuristicDescriptor`**; session state stays on **`hipdnnHeuristicHandle_t`**. |
| **Plugin heuristic handle** | **`hipdnnHeuristicHandle_t`**: session object per heuristic module per **`hipdnnHandle`**; **SetDeviceProperties** (serialized device-properties FlatBuffer in **`hipdnnPluginConstData_t`**), applied **before** **`PolicyFinalize`** on descriptors bound to that handle; heuristics **read** device facts from the handle as needed; **single-thread** use ([§8.3](#83-plugin-handle-session-object)). |
| **Plugin policy descriptor** | **`hipdnnHeuristicPolicyDescriptor_t`**: per-slot graph + candidate IDs + finalize result; **owned by** **`EngineHeuristicDescriptor`**. |
| **SerializedGraph** | FlatBuffer from the operation graph descriptor, carried in **`hipdnnPluginConstData_t`** ([§13.1](#131-serialized-graph)). |
| **Loaded heuristic policy enumeration** | Handle-scoped query of **`int64_t` policy IDs** (and optional **name** / version metadata) for heuristic modules that passed load-time validation; see [§16.2](#162-loaded-heuristic-policy-enumeration). |
