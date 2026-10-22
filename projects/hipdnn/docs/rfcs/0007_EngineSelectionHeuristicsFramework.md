# RFC: Engine Selection and Heuristics Framework for hipDNN

- **Status**: Implemented
- **Status notes**: Shipped in four parts: plugin SDK and device properties (#6467), policy framework with the `Config` and `StaticOrdering` built-ins (#6605), policy enumeration (#6757), and the `ModeA` / `ModeB` prediction built-in with optional host-services exports at heuristic ABI 0.1.0 (#13028). One heuristic plugin can expose many policies. There is no descriptor-level device-properties override and no handle-level policy order.
- **Last verified**: 2026-10-15 against `develop` (2501d0b28cc)
- **Component**: Backend, heuristic/selection plugins, frontend (§16)

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

This document describes an extensible **engine selection and heuristics** framework in hipDNN. `EngineHeuristicDescriptor::finalize()` runs an **ordered outer loop** over **policy plugins**. Each policy is a distinct selection strategy (for example: machine-learning-based selector, rule-based selector, cache-driven selector, or a deterministic default selector).

Device capabilities enter the system **only as explicit data** (`DeviceProperties`), not by probing HIP from inside plugins. The backend queries them for the device of the handle's stream ([§6](#6-device-properties)). Across the **stable C ABI**, those facts are passed as a **FlatBuffer-serialized** payload in **`hipdnnPluginConstData_t`** (same tooling and buffer wrapper type as serialized graph data in **engine** plugins—see [§13.2](#132-serialized-device-properties-flatbuffer)) via **`hipdnnHeuristicHandleSetDeviceProperties`** on the plugin session handle ([§8](#8-c-abi-for-heuristic-plugins)); plugins **query** that handle state during **`Finalize`** (policy descriptors are **created with** the handle—[§8.7](#87-policy-descriptor-per-slot-graph--candidate-ids)), not via a separate device payload on the policy API.

Heuristic plugins implement a **two-tier** **stable C ABI** ([§8](#8-c-abi-for-heuristic-plugins)): a long-lived **plugin handle** (**`hipdnnHeuristicHandle_t`**, **createHandle** / **destroyHandle** / **setDeviceProperties**) created with the same timing and storage pattern as other hipDNN plugin handles, and a **policy descriptor** (**`hipdnnHeuristicPolicyDescriptor_t`**, **createPolicyDescriptor** / **destroyPolicyDescriptor** / **setEngineIds** / **Finalize** / **getSortedIds**) whose **lifecycle is owned by** **`EngineHeuristicDescriptor`**—one per slot in the ordered policy list, destroyed with the heuristic descriptor. That ABI is **separate from the engine plugin ABI**—a `.so` is one or the other, never both.

The **`SelectionHeuristic`** C++ type wraps a **policy descriptor** and forwards to the C ABI; **stateful tracking** in the plugin lives behind the **plugin handle**. **HeuristicPluginManager** (and handle-scoped **HeuristicPluginResourceManager**) own discovery, loading, version validation, registration, and **plugin-handle** instances per **`hipdnnHandle`**. **`Graph.preferred_engine_id`** is honored by the **frontend** as a post-hoc reorder of the heuristic-ranked engine configs returned by **`HIPDNN_ATTR_ENGINEHEUR_RESULTS`** (see **`Graph::initializeEngineConfig`**); the backend heuristic loop is unaware of it. The JSON engine-override file named by **`HIPDNN_HEUR_CONFIG_PATH`** is applied by a regular built-in policy, **`SelectionHeuristic::Config`**, that lives in the ordered policy list and **declines** when no rule matches so subsequent policies (typically **`SelectionHeuristic::StaticOrdering`**) take over ([§5.3.5](#535-the-config-built-in-policy)). The default deterministic ordering policy (**`SelectionHeuristic::StaticOrdering`**) is shipped as a **backend built-in**: it implements the same heuristic plugin C ABI shape (function-pointer table) but is **registered in-process** at **`HeuristicPluginManager`** construction time, with no separate `.so` to discover or dlopen ([§10.1](#101-heuristicpluginmanager)). **`EngineHeuristicDescriptor::finalize()`** walks the **ordered policy list** using **descriptor-owned** policy objects; the resolved order is **user-configurable** at the public/config surface ([§5.3](#53-ordered-policy-list-default-and-user-configuration)). The backend resolves user-supplied **policy name** strings to **`int64_t` policy IDs** with the **same deterministic hash as engine IDs** ([RFC 0003](0003_EngineIdDesign.md), `hipdnn_data_sdk::utilities::policyNameToId`) before matching loaded plugins (and built-ins) and running the outer loop.

There is **no** separate post-loop step that applies **`utilities::sortEngineIds`** (or any other ordering) inside the backend. Legacy deterministic ordering is available **only** when a policy in **`orderedPolicyIds`** implements it—for example **`SelectionHeuristic::StaticOrdering`**. If every policy declines or the list is misconfigured so that no policy succeeds, **`finalize()`** throws **`HipdnnException`** with **`HIPDNN_STATUS_INTERNAL_ERROR`**, the same error path as other descriptor **`finalize()`** failures.

---

## 2. Motivation

- **Extensibility:** A fixed engine ordering in backend code (`utilities::sortEngineIds`) forces an edit to the core library for each new strategy (performance models, user policy, caches). Policies are plugins instead.
- **Explicit device context:** Heuristics depend on **passed-in** device facts, not on global HIP state or opaque handle internals.
- **Consistency:** Engine plugins already load from trusted paths and validate API versions. Heuristic policies should follow the **same class of patterns** (shared libraries, version gates) **without sharing symbols or types** with engine plugins.
- **Interop:** A **C ABI** for heuristic plugins enables non–C++ consumers, stable binary boundaries, and future P/Invoke-style bindings without dragging in the engine plugin surface.

---

## 3. Baseline: current behavior

`EngineHeuristicDescriptor`:

1. Obtains candidate engine IDs from `EnginePluginResourceManager::getApplicableEngineIds(opGraph)`.
2. Orders them in `finalize()` with the outer policy loop ([§5.1](#51-single-orchestration-model-outer-loop)).
3. Exposes the ordered list through `HIPDNN_ATTR_ENGINEHEUR_RESULTS` as engine configuration descriptors.

The deterministic `utilities::sortEngineIds` ordering (MIOPEN engine IDs first, deterministic variants last) is one **policy** (the “StaticOrdering” selector), shipped as a **backend built-in** that implements the heuristic plugin C ABI shape (function-pointer table) without a separate `.so` ([§10.1](#101-heuristicpluginmanager), [§16.2](#162-loaded-heuristic-policy-enumeration)). It runs **only when that policy appears** in **`orderedPolicyIds`** and is **not** applied as unconditional backend logic after the outer loop. **`Graph.preferred_engine_id`** is a **frontend** concern — **`Graph::initializeEngineConfig`** reorders the backend's heuristic-ranked engine configs after **`finalize()`**. The JSON engine-override file named by **`HIPDNN_HEUR_CONFIG_PATH`** is applied by the built-in **`SelectionHeuristic::Config`** policy ([§5.3.5](#535-the-config-built-in-policy)) — it lives in the ordered policy list like any other selector and declines when no rule matches.

---

## 4. Goals and non-goals

**Goals**

- **Outer-loop-only** orchestration: try an ordered list of **policy** plugins until one succeeds.
- **Explicit device facts** at the selection boundary: C++ **`DeviceProperties`** in the backend, serialized to a FlatBuffer and carried in **`hipdnnPluginConstData_t`** across the heuristic C ABI ([§13.2](#132-serialized-device-properties-flatbuffer)); plugins do not call `hipGetDevice` / `hipGetDeviceProperties` themselves.
- **Pluggable policies** loaded through a **`HeuristicPluginManager`** / **`HeuristicPluginResourceManager`** layer that mirrors **engine** plugin *mechanics* (paths, discovery, version checks, optional static path configuration) but uses the **heuristic-only** C ABI in [§8](#8-c-abi-for-heuristic-plugins).
- **Stable C ABI** for heuristic `.so` files: module metadata, **plugin-handle** lifecycle, **policy-descriptor** lifecycle, and selection entry points as in [§8](#8-c-abi-for-heuristic-plugins), declared in **`HeuristicsPluginApi.h`**.
- **Aligned lifetimes:** **`hipdnnHeuristicHandle_t`** per loaded heuristic module per **`hipdnnHandle`** (with engine-plugin handles); **`hipdnnHeuristicPolicyDescriptor_t`** instances **owned by** **`EngineHeuristicDescriptor`** (one per policy slot), created when the policy list for that descriptor is established and destroyed with the descriptor.
- **Same logging path** as the rest of the backend (see [§12 Logging](#12-logging)).
- **Version compatibility** at load time (see [§11](#11-versioning-and-compatibility-checks)).
- **User-configurable policy order** with a documented **default** list of **policy name** strings ([§5.3](#53-ordered-policy-list-default-and-user-configuration)); **internal** resolution uses **`int64_t`** policy IDs (hashed names), aligned with hipDNN **engine** ID practice.
- **No hidden fallback ordering:** if no policy in **`orderedPolicyIds`** reports success, **`EngineHeuristicDescriptor::finalize()`** fails using the **normal backend mechanism** (**`HipdnnException`** and a **`hipdnnStatus_t`**—see [§5.1](#51-single-orchestration-model-outer-loop) and [§14.2](#142-pseudocode-for-finalize)); there is no second pass that sorts candidates inside the core library.

**Non-goals**

- Inner “stage chains” inside a single policy (for example PerEngine → Config → StaticOrdering as mandatory sub-steps). Policy composition is expressed by **ordering multiple policy plugins** in the outer list, not by nested pipelines inside one policy plugin implementation.
- Async selection; the API is only **shaped** to allow it later (including **`Finalize`** + **`GetSortedEngineIds`** as a two-phase C selection API in [§8.9](#89-finalize-and-sorted-results)).
- Tunning is not approached in this RFC. Including auto-tuning and exhaustive tuning (both of which will require device access).

---

## 5. Design overview

### 5.1 Single orchestration model: outer loop

There is **one** primary control flow:

1. Build the list of **candidate engine IDs** from existing **engine** plugins.
2. Obtain **serialized device properties** and the **serialized operation graph** (see [§6](#6-device-properties) and [§13](#13-serialized-graph-device-properties-and-graph-level-preferences)).
3. Resolve the **ordered list of policy plugin IDs** (`int64_t`, from user **policy name** strings via the shared name→ID hash—[§5.3](#53-ordered-policy-list-default-and-user-configuration)).
4. Ensure **`EngineHeuristicDescriptor`** owns one **plugin policy descriptor** per slot (see [§5.4](#54-two-tier-plugin-objects-handle-vs-policy-descriptor)); each slot binds to the **`hipdnnHeuristicHandle_t`** for that policy’s loaded module **or built-in** (from **`HeuristicPluginResourceManager`**). The policy descriptor is **created with** that handle ([§8.7](#87-policy-descriptor-per-slot-graph--candidate-ids)); **Finalize** consumes device facts **from that handle’s session state**, not from separate per-call device arguments on the policy.
5. After resolving **serialized device properties** (FlatBuffer bytes from **`DeviceProperties`**—[§6](#6-device-properties), [§13.2](#132-serialized-device-properties-flatbuffer)), call **`hipdnnHeuristicHandleSetDeviceProperties`** **once per distinct** **`hipdnnHeuristicHandle_t`** that appears among the policy slots (not once per slot iteration). That establishes the device context on the **handle**; heuristic code **queries** it as needed during **Finalize** (and any later selection work on that handle).
6. For each slot in order: **setEngineIds** (current candidates) and **serialized graph** on that slot’s policy descriptor, call **Finalize**; if the policy **wins**, read **getSortedIds** and **stop**; otherwise continue. The default order puts **`SelectionHeuristic::Config`** ([§5.3.5](#535-the-config-built-in-policy)) ahead of **`SelectionHeuristic::StaticOrdering`** so user-supplied **`HIPDNN_HEUR_CONFIG_PATH`** rules win when matched and the deterministic fallback runs otherwise.
7. If no policy succeeds, **`finalize()`** **fails**: it throws **`HipdnnException`** with **`HIPDNN_STATUS_INTERNAL_ERROR`**, like other backend descriptor logic errors. There is **no** additional built-in sort after the loop.

There is **no** separate inner registry of sub-stages inside a single policy plugin in this design. If a team wants “config then static ordering” inside one deliverable, they ship **one** policy plugin that implements that sequence internally, or they register two entries in the **outer** list.

### 5.2 Policy examples (illustrative)

| Policy (illustrative name) | Role |
|----------------------------|------|
| **Cache selector** | Reuse a persisted ordering keyed by graph fingerprint and device properties. |
| **ML selector** | Score candidates using a model or cost function. |
| **Rule selector** | Deterministic rules over graph features and `DeviceProperties`. |
| **StaticOrdering-style selector** | Reproduce the `sortEngineIds` order when listed in **`orderedPolicyIds`**; place it **last** so it acts as the fallback. |

These are **examples**; each policy has a **canonical UTF-8 policy name** (for documentation and user config). The **stable identifier** across the loader, registry, and C ABI is the **`int64_t`** produced by the **same `policyNameToId` hash** as computational engines, reported by **`hipdnnHeuristicPluginGetAllPolicyIds`** ([§8.2](#82-plugin-module-metadata)).

### 5.3 Ordered policy list: default and user configuration

Policy order has **two representations**:

- **User / public surface:** an ordered list of **UTF-8 policy name** strings (attributes, env vars, optional frontend helpers)—the same **human-readable names** vendors document for their plugins (for example **`SelectionHeuristic::StaticOrdering`**).
- **Internal / loader / C ABI:** each name maps to a deterministic **`int64_t`** using **`hipdnn_data_sdk::utilities::policyNameToId`** (`PolicyNames.hpp`). The hash is FNV-1a, the same hash family used for computational **engine** IDs ([RFC 0003](0003_EngineIdDesign.md), `EngineNames.hpp::engineNameToId`); the two functions are siblings rather than aliases — they share the algorithm but are kept as separate symbols so policy and engine name spaces stay textually distinct in call sites. The outer loop operates on **`std::vector<int64_t> orderedPolicyIds`**; each element must match an ID reported by **`hipdnnHeuristicPluginGetAllPolicyIds`** for a loaded heuristic plugin or a **built-in** policy.

**Collision note:** Policy IDs and engine IDs are both `int64_t` hashes of **different string namespaces** in normal usage (policy names like `SelectionHeuristic::…` vs engine names like `MIOPEN_PLUGIN`). A numeric collision is theoretically possible; **context** (policy registry vs engine registry) keeps them separate.

#### 5.3.1 Well-known policy names and IDs

The backend ships four policies with **well-known UTF-8 names** (the `SelectionHeuristic::` prefix is a **naming convention** in the string itself, not C++ language linkage or a shared type with the C++ `SelectionHeuristic` class). The **canonical `int64_t` ID** of each is **`policyNameToId(name)`**. A plugin or built-in that implements one of these policies **must** report that exact value from **`hipdnnHeuristicPluginGetAllPolicyIds`**:

| Policy name string | Role |
|--------------------|------|
| **`SelectionHeuristic::Config`** | First consults an exact-match autotune ranking cache (disable with **`HIPDNN_DISABLE_EXACT_ENGINE_CACHE`**). Then reads the JSON file pointed to by **`HIPDNN_HEUR_CONFIG_PATH`**, walks the graph’s supported primary nodes (convolution, SDPA, matmul) against the rule set, and on a match returns the candidate list with the matched engine moved to the front. **Declines** when the cache does not decide and the env var is unset, the file is missing, the graph cannot be parsed, no rule matches, or the matched engine is not in the candidate set — so the policy loop falls through to the next slot. Shipped as a **backend built-in** ([§10.1](#101-heuristicpluginmanager), [§5.3.5](#535-the-config-built-in-policy)); no separate `.so`. |
| **`SelectionHeuristic::StaticOrdering`** | Deterministic ordering from **`utilities::sortEngineIds`** (for example MIOPEN preference, deterministic engine last). **`HIPDNN_HEUR_FALLBACK_ENGINE_ORDER`** (comma-separated engine names or IDs) replaces that ordering when set. Shipped as a **backend built-in** ([§10.1](#101-heuristicpluginmanager)); registered through the heuristic plugin C ABI shape but with no separate `.so`. Invoked only when listed in the resolved policy order. |
| **`SelectionHeuristic::ModeA`** | The quick policy: ranks candidates on the engine-level (L1) prediction in the request's **ranking metric** alone, and never asks for a configuration-level (L2) prediction — not even for the winner, whose kernel is chosen at plan build by the same metric ([RFC 0019 §11.2](0019_UniversalHeuristicDescriptor.md#112-two-engine-selection-policies-rfc-0007), Open Question 21 option (b)). Shipped as a **backend built-in** (`backend/src/heuristics/prediction/`, [§10.1](#101-heuristicpluginmanager)). Scored engines are ordered by value in the metric's direction, ties broken by static order; unscored or invalid engines follow in static order with one warning naming them and the metric. Declines ("no engine serves `<metric>` at L1") when no candidate is scored. |
| **`SelectionHeuristic::ModeB`** | The thorough policy: per engine, the configuration-level (L2) prediction in the ranking metric, falling back to the engine-level (L1) one where L2 has no usable answer ([RFC 0019 §11.2](0019_UniversalHeuristicDescriptor.md#112-two-engine-selection-policies-rfc-0007)). Same built-in as **`ModeA`** — one function table exposing **both** policy IDs ([§8.2](#82-plugin-module-metadata)) — with the same ordering and tail rules; declines with "no engine serves `<metric>` at L1 or L2". |

**Ranking metric.** Both prediction policies compare engines in one registered **ranking metric** ([RFC 0019 §4.4](0019_UniversalHeuristicDescriptor.md#44-ranking-metrics)): `tflops` (higher is better, the default) or `time` (milliseconds, lower is better). The metric is resolved on every **`EngineHeuristicDescriptor::finalize()`** with the same precedence as the policy order: the **`HIPDNN_HEUR_RANKING_METRIC`** environment variable, then the **`HIPDNN_ATTR_ENGINEHEUR_RANKING_METRIC_EXT`** attribute, then `tflops`. An unregistered name is refused (`HIPDNN_STATUS_BAD_PARAM`) at set, while an unregistered environment value is logged and ignored. The host passes it to policies in the host services table ([§8.9](#89-finalize-and-sorted-results)) and stamps it into every result configuration's **`EngineConfig.ranking_metric`**, so the chosen engine ranks its own catalog by the same metric at plan build ([RFC 0019 §11.4](0019_UniversalHeuristicDescriptor.md#114-selecting-a-uhd-by-metric)). **`ModeA`** and **`ModeB`** are not in the default policy order ([§5.3.2](#532-default-policy-order-user-strings)). To use them, list them in the policy order.

**`Graph.preferred_engine_id`** is **not** a policy. The frontend reorders the heuristic-ranked engine configs in **`Graph::initializeEngineConfig`** after **`EngineHeuristicDescriptor::finalize()`** has returned, so it is honored independently of the policy list.

**Mandatory registration.** A heuristic plugin (or built-in adapter) that implements **`SelectionHeuristic::StaticOrdering`** **must** report exactly **`policyNameToId("SelectionHeuristic::StaticOrdering")`** from **`hipdnnHeuristicPluginGetAllPolicyIds`** — there is **no** alternate ID for that behavior. The same rule holds for every name in the table above. **`hipdnnHeuristicPluginGetPolicyName`** ([§8.2](#82-plugin-module-metadata)) returns the canonical UTF-8 name for each policy ID. At load time, **`HeuristicPlugin::validatePluginMetadata`** **rejects** the plugin if any name is empty or **`policyNameToId(name) != policy_id`** (same idea as keeping engine IDs and registered engine names consistent).

**Overlapping IDs at load.** **`HeuristicPluginManager`** rejects a heuristic module that reports a policy ID already claimed by another **accepted** heuristic plugin or built-in ([§10.1](#101-heuristicpluginmanager), [§11](#11-versioning-and-compatibility-checks)). The built-ins are registered first, so a third-party module that claims the same ID as a built-in is rejected. This is **stricter than the engine side**, which drops only the duplicate engine.

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

Unknown IDs that survive resolution simply produce a null `_policySlots` entry inside the per-finalize sync (the policy is treated as "not loaded") and the outer loop skips them without raising an error.

#### 5.3.4 Relationship to `finalize()`

`EngineHeuristicDescriptor::finalize()` resolves an ordered **`std::vector<int64_t> orderedPolicyIds`** from the precedence chain above (env tokens are name-or-ID hashed per token; the descriptor attribute is already a raw ID array; the default fallback is hashed from its built-in name list via **`policyNameToId`**). It then ensures **policy descriptor** objects exist **one-to-one** with that list (recreating them if the list changed since last setup) and runs the outer loop in [§14.2](#142-pseudocode-for-finalize). If the loop completes without a successful policy, **`finalize()`** does not mark the descriptor finalized and exits via **`HipdnnException`** like other backend failures.

#### 5.3.5 The Config built-in policy

**`SelectionHeuristic::Config`** (in `backend/src/heuristics/config/`) is a regular built-in policy. It applies JSON engine-override rules from the file named by **`HIPDNN_HEUR_CONFIG_PATH`**. It sits in **`orderedPolicyIds`** like any other policy and **declines** when nothing applies, so the policy loop falls through to the next slot (typically **`SelectionHeuristic::StaticOrdering`**). Code: `ConfigBuiltIn.cpp` and `EngineOverrideConfig.hpp`.

**Behavior on each `Finalize`:**

1. If the candidate list is empty, decline.
2. Unless **`HIPDNN_DISABLE_EXACT_ENGINE_CACHE`** is set to a truthy value, look up the exact-match autotune ranking cache. The key is derived from the serialized graph and the device properties on the handle. A usable hit wins: the policy returns the cached order and reports success. Any miss continues with step 3.
3. Read **`HIPDNN_HEUR_CONFIG_PATH`** from the environment on every invocation (no process cache). If the variable is unset or the file cannot be opened/parsed, decline.
4. Parse the serialized graph buffer with the data-SDK FlatBuffer verifier. If the buffer is malformed or the graph has no nodes, decline.
5. Walk the graph’s supported primary nodes (**`ConvolutionFwd`**, **`ConvolutionBwd`**, **`ConvolutionWrw`**, **`Sdpa`**, **`SdpaBackward`**, and **`Matmul`** attributes) against the rule’s op + tensor-shape patterns (exact dim/stride matching with `-1` wildcards, exact-bucket-before-wildcard-bucket within declaration order). A rule names its engine by engine name or ID. On the first match, take that engine ID.
6. If the matched engine ID is **not** in the candidate list, decline.
7. Otherwise return the candidates with the matched engine moved to the front (other candidates preserve their original order) and report success (`*outApplied = 1`).

**Default placement.** The default policy list is **`{ "SelectionHeuristic::Config", "SelectionHeuristic::StaticOrdering" }`** ([§5.3.2](#532-default-policy-order-user-strings)) so user-supplied JSON rules win over the deterministic fallback. Users may freely omit, reorder, or replace either built-in. There is no implicit guarantee that **`Config`** runs — if the user supplies a custom policy order without it, the JSON file is simply not consulted for that **`finalize()`**.

**`Config` does not use graph-level `preferred_engine_id`.** **`Graph::set_preferred_engine_id_ext`** is honored by the **frontend** as a post-hoc reorder of the heuristic-ranked engine configs in **`Graph::initializeEngineConfig`**, after **`EngineHeuristicDescriptor::finalize()`** has returned. The backend heuristic loop is unaware of it.

**`StaticOrdering`** is a normal default-ordered fallback ([§5.3.2](#532-default-policy-order-user-strings)); the user may omit or reorder it. Both are shipped through **`HeuristicPlugin::createBuiltIn(populateFunctionTable(), label)`** at **`HeuristicPluginManager`** construction time, as is the prediction built-in behind **`ModeA`** / **`ModeB`** ([§5.3.1](#531-well-known-policy-names-and-ids), [§10.1](#101-heuristicpluginmanager)); none has a separate `.so` and the user cannot unload them.

### 5.4 Two-tier plugin objects: handle vs policy descriptor

| Tier | C typedef | Lifetime | Holds |
|------|---------------------------|----------|--------|
| **Plugin handle** | **`hipdnnHeuristicHandle_t`** | Same pattern as other plugin handles on **`hipdnnHandle`**; one per heuristic library, shared by all policies of that library; created when the heuristic module is paired with the handle, destroyed with handle teardown | Plugin **session** state (caches, tuning, scratch, **parsed device facts** after **`SetDeviceProperties`**); receives **`SetDeviceProperties`** (serialized device-properties FlatBuffer in **`hipdnnPluginConstData_t`**), **queried** by heuristics during **`Finalize`** and related work |
| **Policy descriptor** | **`hipdnnHeuristicPolicyDescriptor_t`** | **Owned by** **`EngineHeuristicDescriptor`**; one per entry in **`orderedPolicyIds`**; **created** in **`finalize()`** when the resolved list differs from the stored list, and **destroyed** with the heuristic descriptor | **Candidate engine IDs** + **serialized graph** for the current selection; **Finalize** / **GetSortedEngineIds** result. **Created with** a **`hipdnnHeuristicHandle_t`** and a **`policy_id`** ([§8.7](#87-policy-descriptor-per-slot-graph--candidate-ids)); **device properties** for selection are read from **that handle’s** session state (**`SetDeviceProperties`** on the handle—[§8.6](#86-plugin-handle-lifecycle)), not passed again on the policy descriptor. |

**Threading:** Plugin handles are **single-thread only** (**not** thread-safe). Parallelism uses **multiple** **`hipdnnHandle`** instances (each with its own heuristic plugin handles). All calls for a given **`hipdnnHeuristicHandle_t`** and its dependent policy descriptors must follow this contract.

**Design-note mapping:** **createHandle** / **destroyHandle** / **setDeviceProperties(Handle\*)** → [§8.6](#86-plugin-handle-lifecycle); **createPolicyDescriptor** / **destroyPolicyDescriptor** / **setEngineIds** / **Finalize** / **getSortedIds** → [§8.7](#87-policy-descriptor-per-slot-graph--candidate-ids)–[§8.9](#89-finalize-and-sorted-results). Documents that used the term **engineDescriptor** for the plugin-side object mean **policy descriptor** here, not hipDNN’s **`EngineDescriptor`** for computational engines.

---

## 6. Device properties

### 6.1 `DeviceProperties` struct

The backend carries device facts in **`hipdnn_flatbuffers_sdk::data_objects::DevicePropertiesT`**, the object type generated from the `DeviceProperties` table in `flatbuffers_sdk/schemas/device_properties.fbs`. There is no hand-written C++ struct. Helpers live in `backend/src/heuristics/DeviceProperties.hpp`.

```fbs
table DeviceProperties {
    device_id: int = -1;
    multi_processor_count: int = 0;
    total_global_mem: ulong = 0;
    architecture_name: string;
}

root_type DeviceProperties;
file_identifier "HDDP";
```

The **stable C ABI** does **not** expose a fixed-layout struct for device facts. The backend fills **`DevicePropertiesT`**, **serializes** it with the **device-properties FlatBuffer schema** ([§13.2](#132-serialized-device-properties-flatbuffer)), and passes it using the **existing** **`hipdnnPluginConstData_t`** struct from **`PluginApiDataTypes.h`** (the same type the plugin SDK already uses for serialized graph bytes and other const buffers in **engine** plugins—not a separate **`const uint8_t*`** + **`size_t`** pair on the heuristic API). The host passes **`const hipdnnPluginConstData_t*`** to **`hipdnnHeuristicHandleSetDeviceProperties`** ([§8.6](#86-plugin-handle-lifecycle)). Plugins **parse** that payload from **handle** session state (for example when servicing **`Finalize`** on a policy descriptor **created with** that handle) with the same data-SDK / codegen as the graph schema.

### 6.2 Acquisition: `queryDeviceProperties()`

The backend helpers in namespace `hipdnn_backend::heuristics` map from HIP to the plugin payload:

```cpp
DevicePropertiesT queryDeviceProperties(int deviceId);
DevicePropertiesT queryDeviceProperties(hipdnnHandle_t handle);

std::vector<uint8_t> serializeDeviceProperties(const DevicePropertiesT& props);
hipdnnPluginConstData_t wrapSerializedDeviceProperties(const std::vector<uint8_t>& serializedBuffer);
```

- **`queryDeviceProperties(handle)`** resolves the device from the handle's stream (`hipStreamGetDevice`; a default stream uses `hipGetDevice`). It throws **`HipdnnException`** if a HIP call fails.
- **`serializeDeviceProperties`** builds the FlatBuffer bytes. **`wrapSerializedDeviceProperties`** returns a **`hipdnnPluginConstData_t`** that points into those bytes, so the byte vector must outlive the wrapper.
- **`EngineHeuristicDescriptor::finalize()`** always calls **`queryDeviceProperties(handle)`**. There is no caller-supplied override.

**Important:** Plugins **must not** call this helper; they consume only the **serialized device-properties** buffer the host installs on the session handle via **`hipdnnHeuristicHandleSetDeviceProperties`** ([§8.6](#86-plugin-handle-lifecycle)), after **verifying** and **parsing** it per [§13.2](#132-serialized-device-properties-flatbuffer) when **Finalize** (or other selection logic) **reads** it from handle state.

### 6.3 Reserved attribute: `HIPDNN_ATTR_ENGINEHEUR_DEVICEPROP`

The backend reserves **`HIPDNN_ATTR_ENGINEHEUR_DEVICEPROP`**, but **`EngineHeuristicDescriptor`** does not accept it: **`setAttribute()`** and **`getAttribute()`** return **`HIPDNN_STATUS_NOT_SUPPORTED`**.

---

## 7. SelectionHeuristic interface

`SelectionHeuristic` is a **C++ facade** used by the backend for **one policy slot** on an **`EngineHeuristicDescriptor`**. It wraps an opaque **`hipdnnHeuristicPolicyDescriptor_t`** ([§8.7](#87-policy-descriptor-per-slot-graph--candidate-ids)) created with the **`hipdnnHeuristicHandle_t`** for that policy’s module. **Session state** (caches, etc.) lives in the plugin **behind the handle**, not in this wrapper.

Device properties are **not** set on the facade: the host **builds the serialized device-properties FlatBuffer** (from resolved `DeviceProperties`—[§6](#6-device-properties), [§13.2](#132-serialized-device-properties-flatbuffer)), wraps it in **`hipdnnPluginConstData_t`**, and calls **`hipdnnHeuristicHandleSetDeviceProperties`** on each **distinct** plugin handle that will participate in this **`finalize()`**, **before** invoking **`Finalize`** on any policy descriptor **created with** that handle. The plugin **stores** or **parses** that payload on the handle; **`Finalize`** (and internal heuristic logic) **queries** device facts **via the bound handle**, not via extra device arguments on the policy API. Ordering relative to **`PolicyDescriptorCreate(..., plugin_handle)`** is: handle exists → **`SetDeviceProperties`** on that handle for this finalize’s device context → policy inputs (**`SetEngineIds`** / **`SetSerializedGraph`**) → **`Finalize`**.

```cpp
class SelectionHeuristic
{
public:
    // Creates the policy descriptor for policyId on that policy's plugin handle.
    SelectionHeuristic(std::shared_ptr<plugin::HeuristicPluginResourceManager> resourceManager,
                       int64_t policyId);

    // Candidate engine IDs from EnginePluginResourceManager; mirrors setEngineIds ([§8.8](#88-policy-inputs-engine-ids-and-serialized-graph)).
    void setEngineIds(const std::vector<int64_t>& engineIds);

    // Serialized operation graph as hipdnnPluginConstData_t (FlatBuffer; [§13.1](#131-serialized-graph)).
    void setSerializedGraph(const hipdnnPluginConstData_t* serializedGraph);

    // Runs applicability + selection inside the plugin; true => policy won the outer loop ([§8.9](#89-finalize-and-sorted-results)).
    bool finalize();

    // Valid after finalize() returned true; mirrors GetSortedEngineIds.
    std::vector<int64_t> getSortedEngineIds();
};
```

**Notes**

- **Logging** matches **engine** plugins: not part of these method signatures; see [§12](#12-logging).
- **`finalize` + `getSortedEngineIds`** mirror the C ABI two-phase pattern so a future revision can perform async work in **`Finalize`** without changing names.
- **`EngineHeuristicDescriptor`** owns **`SelectionHeuristic`** instances **one per** resolved policy slot; lifetimes match [§5.4](#54-two-tier-plugin-objects-handle-vs-policy-descriptor).

---

## 8. C ABI for heuristic plugins

This section defines the **stable C-language ABI** for heuristic (selection policy) shared libraries. The declarations are in **`plugin_sdk/include/hipdnn_plugin_sdk/HeuristicsPluginApi.h`**. The ABI version is in **`heuristic_api_version.h`** (**`HIPDNN_HEURISTIC_API_VERSION`**, currently `0.0.1`). The heuristic ABI is **separate** from the **engine** plugin ABI: a single `.so` is either an **engine** plugin or a **heuristic** plugin, **not both**. Heuristic libraries **do not** export any **engine** entry points from `EnginePluginApi.h`. They export the base `PluginApi.h` symbols (**`hipdnnPluginGetType`** returns **`HIPDNN_PLUGIN_TYPE_HEURISTIC`**) and the heuristic symbols below. One library can expose many policies. All policies of one library share one plugin handle; **`hipdnnHeuristicPolicyDescriptorCreate`** takes a **`policy_id`** to select the policy.

### 8.1 Design principles

- All exported symbols use **C linkage** (`extern "C"` from C++ implementations). **`int64_t`** in signatures uses **`stdint.h`** / **`<cstdint>`** (same as engine plugin APIs).
- Opaque serialized payloads (**device properties**, **operation graph**) cross the heuristic C ABI only as **`const hipdnnPluginConstData_t*`**, using the **existing** **`hipdnnPluginConstData_t`** definition in **`PluginApiDataTypes.h`** (same **`ptr`** / **`size`** fields as **`EnginePluginApi.h`** graph and config arguments). The ABI does **not** introduce parallel raw-buffer parameters (for example ad-hoc **`const uint8_t*`** plus **`size_t`**) for these payloads.
- Versioning and rejection of incompatible plugins follow the same **ideas** as engine plugins (major API compatibility), but the **API version string** and **symbol set** are **heuristic-specific** ([§11](#11-versioning-and-compatibility-checks)).
- Plugins **must not** call HIP, mutate hipDNN graph descriptors, or execute engines; they only **read** inputs described here and **write** reordered engine IDs.
- The host (backend) owns output buffers unless the API explicitly transfers ownership.
- **Logging** is configured only via **`hipdnnPluginSetLoggingCallback`** / **`hipdnnPluginSetLogLevel`** at **module load time** ([§8.2](#82-plugin-module-metadata)); handle and policy-descriptor entry points do not take a logger argument ([§12](#12-logging)).

### 8.2 Plugin module metadata

Each heuristic `.so` exports the base `PluginApi.h` symbols and the two policy-enumeration symbols from `HeuristicsPluginApi.h`:

| Function | Purpose |
|----------|---------|
| `hipdnnPluginGetApiVersion(const char** version)` | Semantic version of **the heuristic C ABI** (`HIPDNN_HEURISTIC_API_VERSION`). Host rejects load on **major** mismatch. |
| `hipdnnHeuristicPluginGetAllPolicyIds(int64_t* policy_ids, uint32_t max_policies, uint32_t* num_policies)` | Required. Two-pass query of the stable **`int64_t`** policy IDs the plugin exposes. Each **must** equal **`policyNameToId(canonical_utf8_name)`** for that policy (same hash family as computational engine IDs—[§5.3](#53-ordered-policy-list-default-and-user-configuration), [RFC 0003](0003_EngineIdDesign.md)). The host matches these values against the resolved **`orderedPolicyIds`**. |
| `hipdnnHeuristicPluginGetPolicyName(int64_t policy_id, const char** name)` | Required. **NUL-terminated UTF-8** canonical name for one policy (the string users put in **`HIPDNN_HEUR_POLICY_ORDER`** or hash for **`HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT`**). The host **validates at load** that **`policyNameToId(*name) == policy_id`** ([§5.3.1](#531-well-known-policy-names-and-ids), [§11](#11-versioning-and-compatibility-checks)). |
| `hipdnnPluginGetName(const char** name)` / `hipdnnPluginGetVersion(const char** version)` | Plugin library name (required, non-empty) and implementation version (informational). |
| `hipdnnPluginGetType(hipdnnPluginType_t* type)` | Returns **`HIPDNN_PLUGIN_TYPE_HEURISTIC`**. |
| `hipdnnPluginSetLoggingCallback(hipdnnCallback_t callback)` | Registers the consumer logging callback; optional `hipdnnPluginSetLogLevel(hipdnnSeverity_t level)` mirroring engine plugin behavior. |
| `hipdnnPluginGetLastErrorString(const char** error_str)` | Per-thread last error after a failed call; pointer valid only for immediate use. |

The host identifies heuristic libraries by **which loader** opened them, by **`hipdnnPluginGetType`** returning **`HIPDNN_PLUGIN_TYPE_HEURISTIC`**, and by successful resolution of all required symbols.

### 8.3 Plugin handle (session object)

Each loaded heuristic module exposes a **plugin handle**: an opaque **session** object. The host creates **one** **`hipdnnHeuristicHandle_t`** per **loaded heuristic library** associated with a given **`hipdnnHandle`**, using the **same timing and storage pattern** as other hipDNN plugin handles. All policies of that library share the handle, and **stateful tracking** lives in the plugin behind it.

**Threading contract:** A plugin handle is **not** thread-safe (**single-thread only**). Concurrent use requires **separate** **`hipdnnHandle`** instances (and thus separate heuristic plugin handles).

```c
typedef struct hipdnnHeuristicHandle_opaque* hipdnnHeuristicHandle_t;
```

### 8.4 Serialized device properties (no POD struct)

Device facts cross the heuristic **C ABI** as **opaque serialized bytes** carried in the **existing** **`hipdnnPluginConstData_t`** wrapper (**`const hipdnnPluginConstData_t*`** at the API boundary), not as a public C struct and not as a standalone **`const uint8_t*`** + **`size_t`** pair. That matches the **serialized graph** path ([§8.8](#88-policy-inputs-engine-ids-and-serialized-graph), [§13](#13-serialized-graph-device-properties-and-graph-level-preferences)) and existing engine plugin APIs: the **function signatures** stay stable while the **FlatBuffer schema** evolves (new optional fields, same **`hipdnnPluginConstData_t`** entry point).

- **Schema:** The device-properties table is **`DeviceProperties`** in **`flatbuffers_sdk/schemas/device_properties.fbs`** (file identifier `HDDP`). Fields: `device_id`, `multi_processor_count`, `total_global_mem`, and `architecture_name`. Additional facts (wavefront size, etc.) are **additive** in the schema without changing **`hipdnnHeuristicHandleSetDeviceProperties`**.
- **Contract:** The host supplies a **complete FlatBuffer** rooted at that table; plugins **must** reject buffers that fail **`Verifier`** checks or use an **incompatible schema version** (per data-SDK / file-identifier rules—same class of checks as for the graph buffer).
- **Tradeoff:** Plugins depend on FlatBuffer + generated types for **both** graph and device inputs; the win is **no struct-layout ABI breaks** when device facts grow.

### 8.5 Status codes

Heuristic C entry points return **`hipdnnPluginStatus_t`**, the same typedef as **engine** plugin APIs (**`PluginApiDataTypes.h`** in the plugin SDK). **`HeuristicsPluginApi.h`** includes (or re-exports) that header; heuristic **`.so`** files **must not** include **`HipdnnStatus.h`**, so the plugin ABI stays free of backend layering—the same rule as engine plugins.

Use the existing **`HIPDNN_PLUGIN_STATUS_*`** enumerators for ordinary outcomes—for example **`HIPDNN_PLUGIN_STATUS_SUCCESS`**, **`HIPDNN_PLUGIN_STATUS_BAD_PARAM`**, **`HIPDNN_PLUGIN_STATUS_INVALID_VALUE`**, **`HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR`**, and **`HIPDNN_PLUGIN_STATUS_ALLOC_FAILED`**. These are generic and align in purpose with backend **`HIPDNN_STATUS_*`** / **`hipdnnStatus_t`**; the host maps between plugin and backend status the same way it does for engine plugins.

**Heuristic-specific outcomes:** **`hipdnnPluginStatus_t`** also has **`HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE`** (5) and **`HIPDNN_PLUGIN_STATUS_NOT_INITIALIZED`** (6). A policy does **not** use **`NOT_APPLICABLE`** to decline: **`hipdnnHeuristicPolicyFinalize`** returns **`HIPDNN_PLUGIN_STATUS_SUCCESS`** with **`*out_applied = 0`**. **`hipdnnHeuristicPolicyGetSortedEngineIds`** returns **`NOT_INITIALIZED`** before a successful **`Finalize`**. New codes are appended **after** the existing **`HIPDNN_PLUGIN_STATUS_*`** values so the enum stays backward-compatible. There is no separate heuristic-only status typedef.

Each heuristic C entry point returns **`hipdnnPluginStatus_t`** unless noted.

**No buffer-size error for sorted IDs:** Do **not** add **`BUFFER_TOO_SMALL`** (or **`BAD_PARAM_SIZE_INSUFFICIENT`**) for **`hipdnnHeuristicPolicyGetSortedEngineIds`**. That call takes **`engine_ids`** and an in/out **`*num_engines`** (capacity in, count out); the plugin writes **at most** the given capacity and sets **`*num_engines`** to how many were written. **`HIPDNN_PLUGIN_STATUS_SUCCESS`** is returned even when the caller’s buffer holds only a **prefix** of the full ordering ([§8.9](#89-finalize-and-sorted-results)). This differs from backend APIs that require one buffer to fit an entire payload and return **`HIPDNN_STATUS_BAD_PARAM_SIZE_INSUFFICIENT`** when the query size is too small.

### 8.6 Plugin handle lifecycle

Declarations from **`HeuristicsPluginApi.h`** (export macros omitted):

```c
hipdnnPluginStatus_t hipdnnHeuristicHandleCreate(hipdnnHeuristicHandle_t* out_handle);
hipdnnPluginStatus_t hipdnnHeuristicHandleDestroy(hipdnnHeuristicHandle_t handle);

hipdnnPluginStatus_t hipdnnHeuristicHandleSetDeviceProperties(
    hipdnnHeuristicHandle_t handle,
    const hipdnnPluginConstData_t* device_props_serialized);
```

- **`Create`** / **`Destroy`** correspond to **createHandle** / **destroyHandle**: the host invokes them when binding a heuristic module to a **`hipdnnHandle`** (alongside other plugin-handle setup).
- **`SetDeviceProperties`** corresponds to **setDeviceProperties(Handle\*)**: the host supplies **serialized device-properties** only via **`const hipdnnPluginConstData_t*`** ([§8.4](#84-serialized-device-properties-no-pod-struct), [§13.2](#132-serialized-device-properties-flatbuffer)); plugins **must not** call HIP. The backend resolves **`DevicePropertiesT`** via **`queryDeviceProperties(handle)`** ([§6](#6-device-properties)), **builds** the FlatBuffer, and fills the **`hipdnnPluginConstData_t`** pointed to by **`device_props_serialized`** (**`ptr`** and **`size`**); plugins verify and parse those bytes (FlatBuffer **`Verifier`** / accessors) exactly as for **`hipdnnHeuristicPolicySetSerializedGraph`**. The host then updates **each distinct** **`hipdnnHeuristicHandle_t`** used by the current policy slots **before** **`hipdnnHeuristicPolicyFinalize`** on any descriptor **created with** that handle ([§5.1](#51-single-orchestration-model-outer-loop), [§14.2](#142-pseudocode-for-finalize)). The **`device_props_serialized`** view must remain valid for the duration of the call; the plugin may copy if it needs the payload after return. Heuristic implementations **read** this state from the handle during **`Finalize`** (and as needed elsewhere on that session)—the policy descriptor does **not** carry a parallel device-properties buffer.

### 8.7 Policy descriptor (per-slot graph + candidate IDs)

A **policy descriptor** is a second opaque object. It holds **per–`EngineHeuristicDescriptor` slot** state: **candidate engine IDs**, **serialized graph** bytes, and internal state through **`Finalize`**.

**Naming:** This document uses **policy descriptor** in the C ABI to avoid confusion with hipDNN’s **`EngineDescriptor`** (computational engine). Informal design notes that say **engineDescriptor** for the plugin-side object refer to **policy descriptor** here.

```c
typedef struct hipdnnHeuristicPolicyDescriptor_opaque* hipdnnHeuristicPolicyDescriptor_t;
```

**Lifecycle:** The backend **`EngineHeuristicDescriptor` owns** one **`hipdnnHeuristicPolicyDescriptor_t`** per entry in **`orderedPolicyIds`**. Objects are **created** in **`finalize()`** when the resolved policy list differs from the stored list, and **destroyed** when the **`EngineHeuristicDescriptor`** is destroyed (**plugin policy descriptors and the hipDNN heuristic descriptor die together**). **`hipdnnHeuristicPolicyDescriptorCreate`** takes the **`hipdnnHeuristicHandle_t`** for that slot’s module and the **`policy_id`** to run; that **binds** the policy to the handle **before** **`Finalize`**, so selection code can treat the handle as the source of **device-properties** session state (**`SetDeviceProperties`**—[§8.6](#86-plugin-handle-lifecycle)).

```c
hipdnnPluginStatus_t hipdnnHeuristicPolicyDescriptorCreate(
    hipdnnHeuristicHandle_t plugin_handle,
    int64_t policy_id,
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
    int64_t* engine_ids,     /* NULL => count query */
    size_t* num_engines);    /* in: capacity; out: count available or written */
```

**Contract**

- Calls on a policy descriptor that hang off a given **`hipdnnHeuristicHandle_t`** must occur on a **thread consistent** with that handle’s **single-thread** contract ([§8.3](#83-plugin-handle-session-object)).
- **`Finalize`** assumes **current** device-properties bytes were applied to **that policy’s** **`hipdnnHeuristicHandle_t`** via **`hipdnnHeuristicHandleSetDeviceProperties`** earlier in the same **`EngineHeuristicDescriptor::finalize()`** (or whenever the host last updated device context for that handle). Plugins **query** that handle state as needed; the host does **not** pass device properties again on **`hipdnnHeuristicPolicyFinalize`**.
- **`SetEngineIds` / `SetSerializedGraph` / `Finalize`:** candidate IDs come from **`EnginePluginResourceManager`**; output IDs **must** be a **permutation or subset** of the **SetEngineIds** input. The host checks this: an unknown or duplicate ID throws **`HipdnnException`** with **`HIPDNN_STATUS_PLUGIN_ERROR`**, and the outer loop logs it and skips the policy.
- **`GetSortedEngineIds`:** valid only after **`Finalize`** with **`out_applied == 1`**. Callers use a two-call pattern: pass **`engine_ids = NULL`** to get the full count in **`*num_engines`**, then call again with a buffer of that size. If the buffer is smaller, the plugin writes a prefix, sets **`*num_engines`** to the count written, and still returns **`HIPDNN_PLUGIN_STATUS_SUCCESS`**.

**Optional prediction-aware finalize and config read-back (heuristic ABI 0.1.0).** A policy that ranks on **engine predictions** needs two things the two calls above cannot carry: a way to **ask the host** for a candidate’s prediction while it is deciding, and a way to hand back the exact **engine configuration** it decided on instead of an ID alone. Both are **optional exports**; a module that omits them is a perfectly valid heuristic plugin, and the host uses plain **`Finalize`** and ID-only results for it.

```c
/* Scoped host services for prediction-aware policies (host services version 2). */
typedef struct
{
    uint32_t version;     /* host services version; currently 2 */
    size_t   struct_size; /* size in bytes of this table as the host built it */
    void*    context;     /* opaque, borrowed callback context */
    hipdnnPluginStatus_t (*get_prediction)(
        void* context,
        int64_t engine_id,
        hipdnnEnginePredictionKind_t kind,
        hipdnnPluginConstData_t* prediction); /* borrowed, host-verified EnginePrediction */
    const char* ranking_metric; /* version 2: registered metric this finalize ranks by */
} hipdnnHeuristicHostCallbacks_t;

hipdnnPluginStatus_t hipdnnHeuristicPolicyFinalizeWithHost(
    hipdnnHeuristicPolicyDescriptor_t desc,
    const hipdnnHeuristicHostCallbacks_t* host,
    int32_t* out_applied);

hipdnnPluginStatus_t hipdnnHeuristicPolicyGetEngineConfig(
    hipdnnHeuristicPolicyDescriptor_t desc,
    int64_t engine_id,
    hipdnnPluginConstData_t* engine_config);
```

- **Host services table.** The **host** builds **`hipdnnHeuristicHostCallbacks_t`** and the plugin only reads it. **`version`** and **`struct_size`** are checked **before** any callback field, so a later host may **append** fields and a module written against version 1 still runs against the larger table. The table, its **`context`**, and every buffer a callback hands back are **borrowed for the duration of the call**: a plugin **must not** retain them, invoke callbacks asynchronously, or free host buffers, and copies anything it still needs after **`FinalizeWithHost`** returns. Calls are synchronous on the finalize thread, under the same **single-thread** contract as the rest of this section.
- **`get_prediction`.** Borrows one candidate’s verified **`EnginePrediction`** FlatBuffer for the requested **`hipdnnEnginePredictionKind_t`** layer. Only engine IDs supplied through **`SetEngineIds`** may be queried. Every prediction is asked in the finalize's **`ranking_metric`**: its **`value`** is a calibrated physical quantity in that registered metric's units — an absolute figure of merit, not an ordering key ([RFC 0019 §11.3](0019_UniversalHeuristicDescriptor.md#113-cross-engine-comparison)) — and better is higher or lower as the registry says. The host has already turned an answer in any other metric, or an **`AVAILABLE`** value the metric rejects (non-finite, negative throughput, non-positive time), into **`INVALID`**. The per-engine surface behind the callback belongs to the **engine** plugin ABI, not this one ([RFC 0017 §3](0017_UniversalKernelDescriptor.md#3-how-it-works)); the metric reaches it as **`EngineConfig.ranking_metric`** and the heuristic ABI only forwards the result.
- **`ranking_metric`.** Version 2 appends the registered metric name the policy ranks by (never NULL, borrowed for the call). A plugin reads it only when **`version >= 2`** and **`struct_size`** covers the field; a version 1 table predates metrics and means `tflops`. A name the plugin's registry does not know is a **`BAD_PARAM`**: it has no direction to rank by.
- **`FinalizeWithHost`.** Same two-phase design and the same **`out_applied`** meaning as **`Finalize`**, plus the scoped table. The host calls it **instead of** **`Finalize`** when the module exports it. A **NULL** **`host`** means predictions are unavailable for this finalize; a policy that can rank only on predictions then **declines** (**`out_applied == 0`**) rather than inventing an order, and the outer loop continues.
- **`GetEngineConfig`.** Valid after a successful finalize, for an engine ID the policy returned. The buffer is owned by the **policy descriptor** and lives only until that descriptor’s next mutation or its destruction, so the host verifies and copies it before the next plugin call; the config’s own engine ID **must** match a returned input engine. **`HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE`** means this policy has no configuration for that engine — it returned the engine ID only — and is not an error. A configuration the policy **scored** must be returned with **every** scored knob setting preserved: this read-back is how a configuration-level decision reaches execution, and re-deriving or re-ranking it later would discard the prediction it was chosen on. The host rejects a configuration naming an unregistered **`ranking_metric`** and then stamps the finalize's metric into every result configuration, with or without a policy-supplied one.

### 8.10 Host integration (C++ backend)

**`HeuristicPlugin`** resolves **both** handle and policy symbols via `dlsym` (or equivalent).

**`HeuristicPluginResourceManager`** (per **`hipdnnHandle`**):

1. After loading each heuristic `.so`, calls **`hipdnnPluginSetLoggingCallback`** (and optionally **`hipdnnPluginSetLogLevel`**)—same timing as engine plugins ([§12](#12-logging)).
2. Creates **`hipdnnHeuristicHandle_t`** via **`hipdnnHeuristicHandleCreate`** for each accepted module, stores it **like other plugin handles**, and exposes lookup (for example **`getHeuristicHandleForPolicyId`**) for the backend.
3. Does **not** own **`EngineHeuristicDescriptor`**-scoped policy descriptors; those are created with **`hipdnnHeuristicPolicyDescriptorCreate(plugin_handle, …)`** when the heuristic descriptor (re)builds its policy slot table.

**`EngineHeuristicDescriptor`** holds **`std::vector<std::unique_ptr<SelectionHeuristic>>`**: each **`SelectionHeuristic`** wraps one **`hipdnnHeuristicPolicyDescriptor_t`**, bound to the **`hipdnnHeuristicHandle_t`** for that slot’s policy module. **`finalize()`** resolves **`orderedPolicyIds`**, ensures policy slots (and thus handle bindings) are in place, then **SetDeviceProperties** (serialized device-properties FlatBuffer in **`hipdnnPluginConstData_t`**) **once per distinct** **`hipdnnHeuristicHandle_t`** among those slots, then for each slot **SetEngineIds** / **SetSerializedGraph** / **Finalize** / **GetSortedEngineIds** as in [§14.2](#142-pseudocode-for-finalize). If no slot succeeds, **`finalize()`** aborts via **`HipdnnException`** (normal backend error path); there is **no** post-loop **`utilities::sortEngineIds`**.

### 8.11 ABI evolution

- **Patch/minor:** additive optional C functions; **additive** fields in the **FlatBuffer** schemas for the **graph** and **device properties** ([§13](#13-serialized-graph-device-properties-and-graph-level-preferences)), with plugins and host agreeing via **schema / file-identifier** rules in the data-SDK (same pattern as for the graph).
- **Optional exports are the additive mechanism.** **`hipdnnHeuristicPolicyFinalizeWithHost`** and **`hipdnnHeuristicPolicyGetEngineConfig`** ([§8.9](#89-finalize-and-sorted-results)) are optional exports of heuristic ABI **0.1.0** (`HIPDNN_HEURISTIC_API_VERSION`). The host resolves each symbol and uses it when present. When it is absent, the host uses **`Finalize`** and ID-only results. An **absent optional symbol is not a load failure**, so a plugin built against an older header loads unchanged. **`hipdnnHeuristicHostCallbacks_t`** is the one **host-built struct** in the ABI. Fields are **appended**, never reordered or retyped. A plugin checks its **`version`** / **`struct_size`** pair before it reads a field. Version 2 adds **`ranking_metric`**; a plugin given a version 1 table ranks by `tflops`.
- **Major:** breaking changes to **required** C entry points or to the **meaning** of existing serialized fields without a compatibility story; incompatible plugins fail **`validateBeforeAdding`**-style checks at load time ([§11](#11-versioning-and-compatibility-checks)). The **C ABI** intentionally avoids a public **`hipdnnHeuristicDeviceProperties_t`** layout so **device facts** can evolve inside the FlatBuffer without a struct-layout break.

---

## 9. Policy plugins and the outer loop

Each **heuristic policy** implements one selection strategy and is provided by a heuristic plugin or built-in loaded by **`HeuristicPluginManager`** (see [§10](#10-heuristicpluginmanager-and-resource-layer)). The backend maintains an **ordered list** of **`int64_t` policy IDs** (resolved from user **name** strings—[§5.3](#53-ordered-policy-list-default-and-user-configuration)) and **owns** one **policy descriptor** + **`SelectionHeuristic`** wrapper **per slot** on **`EngineHeuristicDescriptor`** ([§5.4](#54-two-tier-plugin-objects-handle-vs-policy-descriptor)). **Before** the per-slot loop (after slots are synced and **`DeviceProperties`** is resolved), the backend calls **`hipdnnHeuristicHandleSetDeviceProperties`** **once per distinct** **`hipdnnHeuristicHandle_t`** referenced by those slots, with the same **serialized device-properties** FlatBuffer ([§6](#6-device-properties), [§13.2](#132-serialized-device-properties-flatbuffer)). Plugins keep that state on the **handle** and **query** it during **`Finalize`**. Then for each slot in order during **`finalize()`**, the backend:

1. Resolves the **`hipdnnHeuristicHandle_t`** for that slot’s module (from **`HeuristicPluginResourceManager`**); skips the slot if the policy is unknown, failed to load, failed to create its policy descriptor, or its handle failed **`SetDeviceProperties`**. (Device properties on that handle were already updated in the pre-loop step if the handle is non-null.) A policy that throws is logged and skipped.
2. Calls **`setEngineIds`** and **`setSerializedGraph`** on the slot’s **`SelectionHeuristic`** (policy descriptor).
3. Calls **`finalize()`**; if false, **continue** with the **original** candidate engine list unchanged for the next slot.
4. On success, replaces candidates with **`getSortedEngineIds()`** and **breaks**.
5. On failure, continues; the next policy always starts from the **original** candidate list.

**First-success wins:** The first applicable policy that reports success defines the final ordering. Later policies are not consulted.

**Exhausted list:** If every slot is skipped, declined, or errors without a successful policy, **`EngineHeuristicDescriptor::finalize()`** fails with **`HipdnnException`** (same **`THROW_IF_*` / status pattern as other backend descriptor `finalize()` paths**—see [§5.1](#51-single-orchestration-model-outer-loop)); the implementation does **not** fall back to **`utilities::sortEngineIds`** or any other ordering outside **`orderedPolicyIds`**.

**Read-only contract:** Heuristic plugins **must not** mutate hipDNN graph state, engine plugin handles, device memory, or global HIP device selection. They only **read** `SerializedGraph` bytes, **device-properties** data obtained from the **bound** **`hipdnnHeuristicHandle_t`** (installed via **`SetDeviceProperties`**—parsed per [§13.2](#132-serialized-device-properties-flatbuffer)), and the **candidate engine ID list** they receive, and they **output** a reordered subset or permutation of those candidates (the backend validates this—[§8.9](#89-finalize-and-sorted-results)).

---

## 10. HeuristicPluginManager and resource layer

This section mirrors the **two-layer** pattern used for engines: a **manager** that loads and validates shared libraries, and a **resource manager** held by the handle that exposes high-level operations to the rest of the backend.

**Engine stack (for reference):**

- **`EnginePluginManager`** — searches paths, loads `.so` files, owns **`EnginePlugin`** instances. **`validateBeforeAdding`** checks the API major and rejects a plugin whole for no engines or a duplicate ID **within itself**; **`actionAfterAdding`** applies the per-engine rules (ID must equal the hash of the reported name; one ID admits one engine), dropping only the offending engine.
- **`EnginePluginResourceManager`** — constructed with `std::shared_ptr<EnginePluginManager>`; provides **`getApplicableEngineIds`**, execution, workspace queries, stream propagation, etc.; obtained from the handle via **`getPluginResourceManager()`**.

**Heuristic stack:**

### 10.1 `HeuristicPluginManager`

Analogous to **`EnginePluginManager`**:

- Extends the same **plugin manager base** pattern (shared library load, symbol resolution, lifecycle).
- Uses a **separate search path** from engine plugins: `hipdnn_plugins/heuristics/` by default, overridable with **`HIPDNN_HEURISTIC_PLUGIN_DIR`**.
- Resolves **heuristic-only** symbols from [§8](#8-c-abi-for-heuristic-plugins); does **not** use **`EnginePlugin`** symbol tables.
- Implements **`validateBeforeAdding`**-style checks before accepting a plugin—**parallel to `EnginePluginManager`**: **API major** match (via **`hipdnnPluginGetApiVersion`** against **`HIPDNN_HEURISTIC_API_VERSION_MAJOR`**); non-empty plugin name; **unique** **`int64_t` policy ID** across all loaded plugins and built-ins (tracked in a set so a second module cannot register the same ID; engines track the analogous map in `actionAfterAdding` but drop the engine rather than reject the plugin); and **ID ↔ name** consistency for every policy (checked in **`HeuristicPlugin::validatePluginMetadata`**). One plugin may export **multiple** policies; each policy ID is checked individually for global uniqueness.
- Owns **`HeuristicPlugin`** wrappers that bind the C ABI (handle + policy symbols) and expose **`HandleCreate` / `HandleDestroy`** to the resource manager.
- **Registers backend built-ins at construction.** Before any external `.so` discovery, the manager calls **`registerBuiltIns()`** (in `backend/src/heuristics/BuiltInHeuristics.cpp`) to register every shipped built-in heuristic. There are three registrations that cover four policies: **`SelectionHeuristic::Config`** (in `backend/src/heuristics/config/`, driven by **`HIPDNN_HEUR_CONFIG_PATH`** — see [§5.3.5](#535-the-config-built-in-policy)), **`SelectionHeuristic::StaticOrdering`** (in `backend/src/heuristics/static_ordering/`), and the prediction built-in (in `backend/src/heuristics/prediction/`). The prediction built-in has one function table that exposes **both** **`SelectionHeuristic::ModeA`** and **`SelectionHeuristic::ModeB`** through **`hipdnnHeuristicPluginGetAllPolicyIds`** ([§5.3.1](#531-well-known-policy-names-and-ids)). A built-in implements the **same heuristic plugin C ABI shape** (a populated **`HeuristicPluginFunctionTable`**) as a third-party `.so` and is wrapped via **`HeuristicPlugin::createBuiltIn(populateFunctionTable(), source_label)`**, so the rest of the manager (lookup by **`int64_t` policy ID**, version validation, duplicate-ID rejection) treats it identically to a loaded module. Built-ins always exist regardless of the caller's **`hipdnnPluginLoadingMode_ext_t`**, including **`HIPDNN_PLUGIN_LOADING_ABSOLUTE`**.
- **`Graph.preferred_engine_id` is not a built-in concern.** It is honored by the **frontend** as a post-hoc reorder of the heuristic-ranked engine configs (see **`Graph::initializeEngineConfig`**), independent of any backend policy or built-in.

### 10.2 `HeuristicPluginResourceManager`

Analogous to **`EnginePluginResourceManager`**:

- Holds **`std::shared_ptr<HeuristicPluginManager> _pm`** (same structural idea as **`EnginePluginResourceManager::_pm`**).
- **Static path configuration:** **`setHeuristicPluginPaths` / `getHeuristicPluginPaths`** use the same **loading mode** semantics as **`EnginePluginResourceManager::setPluginPaths`** (absolute vs additive paths).
- **`static std::shared_ptr<HeuristicPluginResourceManager> create()`** builds the resource manager after constructing the **`HeuristicPluginManager`**.
- **Instance API (read-only selection):**
  - **`getHeuristicHandleForPolicyId(int64_t policyId)`** — returns the stored **`hipdnnHeuristicHandle_t`** (or null) for that loaded module; created at handle setup via [§8.6](#86-plugin-handle-lifecycle).
  - **`getPluginForPolicyId(int64_t policyId)`** — returns the **`HeuristicPlugin`** (or null) that provides the policy.
  - **`setDevicePropertiesOnAllHandles(...)`** — calls **`SetDeviceProperties`** on every stored plugin handle.
  - **`getHeuristicPolicyInfos()`** — returns per-policy metadata (policy ID, policy name, plugin name, plugin version, API version) for diagnostics and enumeration ([§16.2](#162-loaded-heuristic-policy-enumeration)).
- **`EngineHeuristicDescriptor::resolveHeuristicPolicyOrder()`** (a descriptor member, not on the resource manager) implements the precedence in [§5.3.3](#533-how-the-user-sets-orderedpolicyids), hashes env name tokens to **`int64_t`**, and returns **`orderedPolicyIds`** for **`finalize()`**.
- **Logging:** when heuristic `.so` files are loaded, the host calls **`hipdnnPluginSetLoggingCallback`** (and optionally **`hipdnnPluginSetLogLevel`**) as defined in [§8.2](#82-plugin-module-metadata).

### 10.3 Handle integration

**`hipdnnHandle`** exposes **`getHeuristicPluginResourceManager()`** (`backend/src/handle/Handle.hpp`) alongside **`getPluginResourceManager()`**. It returns a **`std::shared_ptr<HeuristicPluginResourceManager>`** created at handle construction. At that time the manager creates and stores **`hipdnnHeuristicHandle_t`** values per loaded heuristic module ([§8.6](#86-plugin-handle-lifecycle)). Built-ins are always registered, so the manager is never empty.

### 10.4 Relationship to `EnginePluginResourceManager`

- **Candidate engine IDs** always come from **`EnginePluginResourceManager::getApplicableEngineIds`** (unchanged).
- **Ordering** is applied by **`SelectionHeuristic`** instances **owned by** **`EngineHeuristicDescriptor`**, each wrapping a **`hipdnnHeuristicPolicyDescriptor_t`** bound to a **`hipdnnHeuristicHandle_t`** from **`HeuristicPluginResourceManager`**.
- The two subsystems stay **separate**: heuristic plugins **do not** register engine IDs and **do not** execute graphs; engine plugins **do not** implement the heuristic C ABI.

---

## 11. Versioning and compatibility checks

Follow the same **spirit** as `EnginePluginManager::validateBeforeAdding` in the backend: reject incompatible plugins at load time with a clear error.

**Checks**

1. **Heuristic C ABI major:** Parse **`hipdnnPluginGetApiVersion`**; **major** must match **`HIPDNN_HEURISTIC_API_VERSION_MAJOR`** (from `heuristic_api_version.h`), not the engine plugin API version.
2. **Plugin name:** **`hipdnnPluginGetName`** must return a non-empty name.
3. **Policy ID uniqueness:** No two policies, from any loaded modules or built-ins, may report the same **`int64_t`** from **`hipdnnHeuristicPluginGetAllPolicyIds`**. The manager keeps a set of accepted IDs and rejects a module that reports an ID already in it. Engines use the same idea in **`actionAfterAdding`**, but log and drop the engine instead of rejecting the plugin ([§5.3.1](#531-well-known-policy-names-and-ids)).
4. **Policy ID ↔ policy name:** For every policy, **`policyNameToId`** of the name from **`hipdnnHeuristicPluginGetPolicyName`** **must** equal the policy ID; otherwise the loader **rejects** the module ([§5.3.1](#531-well-known-policy-names-and-ids)).

**On failure:** The loader throws **`HipdnnException`** with **`HIPDNN_STATUS_PLUGIN_ERROR`**. **`PluginManagerBase`** logs it, does not register the plugin, and continues loading other plugins ([§12](#12-logging)).

**ABI evolution** is summarized in [§8.11](#811-abi-evolution).

---

## 12. Logging

### 12.1 Backend logging

`hipdnnHandle` (**`struct hipdnnHandle`**) exposes stream and plugin resource manager functionality but **does not** expose a dedicated logger or `getLogger()` accessor. Backend code logs through **`hipdnn_backend::logging`** (`HIPDNN_BACKEND_LOG_*` macros in `backend/src/logging/Logging.hpp`), which dispatches via **`hipdnn_data_sdk::logging`** using the **global** user callback registered for the process.

Engine plugins may receive the logging callback through the existing plugin infrastructure (`PluginBase::setLoggingCallback`), keeping plugin logs on the **same** user-visible path as the backend.

### 12.2 How heuristic code obtains “the same logger” (same pattern as engine plugins)

Engine plugins do **not** take a logger on each API call. After the `.so` is loaded, **`PluginManagerBase::loadPluginFromFile`** calls **`plugin->setLoggingCallback(logging::backendLoggingCallback)`** once, then **`setLogLevel`**, so the plugin **stores** the callback and uses it internally (`backend/src/plugin/PluginCore.hpp`).

**Heuristic plugins follow the same model:**

1. **Backend (C++) code** in the heuristic path uses **`HIPDNN_BACKEND_LOG_*`** / the same global SDK dispatch—no logger argument on **`SelectionHeuristic`** or handle methods ([§7](#7-selectionheuristic-interface)).
2. **Heuristic `.so` code:** Immediately after loading a heuristic library (in **`HeuristicPluginManager`** / resource manager, mirroring **`loadPluginFromFile`**), the host calls **`hipdnnPluginSetLoggingCallback`** ([§8.2](#82-plugin-module-metadata)) with the same **`logging::backendLoggingCallback`** (or an equivalent that forwards to the consumer’s registered path), then optionally **`hipdnnPluginSetLogLevel`**. Built-ins get the same callback through **`PluginManagerBase::registerPlugin`**. Handle and policy-descriptor entry points in [§8.6](#86-plugin-handle-lifecycle)–[§8.9](#89-finalize-and-sorted-results) **do not** take a logging parameter.
3. **C ABI:** Only the module-level **`SetLoggingCallback`** / **`SetLogLevel`** symbols carry logging configuration—**not** the per-instance selection functions.

This matches engine plugins: **supplied and used outside the call signatures** of selection/engine operations, via one-time registration at load time.

---

## 13. Serialized graph, device properties, and graph-level preferences

Heuristic plugins receive **two** structured inputs as **FlatBuffer** payloads in **`hipdnnPluginConstData_t`** over the C ABI: the **operation graph** (per policy descriptor) and **device properties** (installed on the **plugin handle** via **`hipdnnHeuristicHandleSetDeviceProperties`** and **read** by the plugin during **`Finalize`** and other work on that session—[§8.6](#86-plugin-handle-lifecycle), [§8.9](#89-finalize-and-sorted-results)). Both use the **same** serialization stack (data-SDK schemas, `flatbuffers::Verifier`, generated accessors) so **payloads** can evolve without changing **function signatures** or adding public C structs beyond the existing const-buffer wrapper.

### 13.1 Serialized graph

`GraphDescriptor` maintains a **FlatBuffer** serialized graph and exposes it via **`getSerializedGraph()`**, which returns **`hipdnnPluginConstData_t`** (see **`PluginApiDataTypes.h`**). The heuristic framework treats that buffer as the canonical **`SerializedGraph`** input to policies—there is **no alternate wire format** for the graph.

Policies that need structured access may parse the FlatBuffer using existing data-SDK generated types, subject to version rules for the graph schema. The C ABI passes this buffer as **`const hipdnnPluginConstData_t*`** via **`hipdnnHeuristicPolicySetSerializedGraph`** ([§8.8](#88-policy-inputs-engine-ids-and-serialized-graph)), matching **`GraphDescriptor::getSerializedGraph()`** and engine plugin graph arguments.

### 13.2 Serialized device properties (FlatBuffer)

**Device facts** use a **second** FlatBuffer schema: root table **`DeviceProperties`** in **`flatbuffers_sdk/schemas/device_properties.fbs`** (file identifier `HDDP`). The backend builds this buffer from **`DevicePropertiesT`** ([§6.1](#61-deviceproperties-struct)) after **`queryDeviceProperties(handle)`**.

- **C ABI:** **`hipdnnHeuristicHandleSetDeviceProperties(handle, const hipdnnPluginConstData_t*)`** ([§8.6](#86-plugin-handle-lifecycle)); same **`hipdnnPluginConstData_t`** pointer-lifetime pattern as **`hipdnnHeuristicPolicySetSerializedGraph`** for the duration of the call. The host applies this **to each distinct handle** used by the current policy slots **before** **`PolicyFinalize`** on descriptors **created with** that handle ([§5.1](#51-single-orchestration-model-outer-loop)); plugins **query** the stored device context from the handle as needed (including inside **`Finalize`**), rather than receiving device bytes again on the policy descriptor.
- **Plugins:** Verify and parse with the generated **device-properties** types; treat unknown or default-filled **optional** fields as “not provided” when the host runs an older backend.
- **Evolution:** New device fields are **additive** in the `.fbs` definition; heuristic C ABI **major** bumps are reserved for incompatible **API surface** changes, not for ordinary schema extension (aligned with graph-schema practice).

### 13.3 Graph-level preferences (for example `preferred_engine_id`)

The graph model already carries fields such as **`preferred_engine_id`** when built from operation descriptors. **Responsibility for honoring `preferred_engine_id` belongs to the frontend**, not to any backend policy or plugin. After **`EngineHeuristicDescriptor::finalize()`** returns the heuristic-ranked engine configs, **`Graph::initializeEngineConfig`** in **hipdnn_frontend** moves the preferred config (if its global engine index matches) to the front of the candidate list. Because the reorder happens outside the backend, the user's choice of **`HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT`** does not affect whether **`preferred_engine_id`** is honored.

The file-driven engine override named by **`HIPDNN_HEUR_CONFIG_PATH`** is a regular built-in policy — **`SelectionHeuristic::Config`** ([§5.3.5](#535-the-config-built-in-policy)) — that lives in the policy list and operates on the serialized graph buffer like any other policy.

---

## 14. EngineHeuristicDescriptor integration

### 14.1 Responsibilities

`EngineHeuristicDescriptor`:

- Own the operation graph reference and heuristic mode attributes (see [§15](#15-public-api-notes)).
- Discover **candidate** engines via **`EnginePluginResourceManager`**.
- Expose results through **`HIPDNN_ATTR_ENGINEHEUR_RESULTS`**.

Additionally it:

- Obtains **`HeuristicPluginResourceManager`** from the handle (**`getHeuristicPluginResourceManager()`**) for **`hipdnnHeuristicHandle_t`** lookup per policy module.
- **Owns** **`SelectionHeuristic`** (policy-descriptor) objects **one per** resolved policy slot ([§5.4](#54-two-tier-plugin-objects-handle-vs-policy-descriptor)); (re)creates them when **`orderedPolicyIds`** changes.
- Resolves **`DevicePropertiesT`** via **`queryDeviceProperties(handle)`** and **serializes** it for **`SetDeviceProperties`** on each **distinct** **`hipdnnHeuristicHandle_t`** used by its policy slots **before** the per-slot **`Finalize`** loop ([§13.2](#132-serialized-device-properties-flatbuffer)).
- Obtains serialized graph bytes from the finalized graph descriptor.
- Runs the **outer policy loop** described in [§5.1](#51-single-orchestration-model-outer-loop), using **`orderedPolicyIds`** from [§5.3](#53-ordered-policy-list-default-and-user-configuration).
- On success, stores the final ordered engine IDs for result construction; on **exhausted list without success**, aborts **`finalize()`** via **`HipdnnException`** (normal backend error path—[§14.2](#142-pseudocode-for-finalize)).

**Frontend:** Callers that use **hipdnn_frontend** **`backendFinalize`** the heuristic descriptor and read **`HIPDNN_ATTR_ENGINEHEUR_RESULTS`** (for example via **`getEngineConfigs`**); how that lines up with **policy order** and **runtime discovery** of loaded policy IDs is in [§16](#16-frontend-api-flow-mirror-engine-selection).

### 14.2 Pseudocode for `finalize()`

The code is `EngineHeuristicDescriptor::finalize()` in `backend/src/descriptors/EngineHeuristicDescriptor.cpp`. **`HIPDNN_ATTR_ENGINEHEUR_MODE`** accepts only **`HIPDNN_HEUR_MODE_FALLBACK`**.

```text
finalize():
  validate descriptor state (not already finalized, graph set, mode set)

  handle    = graph.getHandle()
  engineRm  = handle.getPluginResourceManager()
  heurRm    = handle.getHeuristicPluginResourceManager()

  candidates = engineRm.getApplicableEngineIds(graph, findFirst)

  if candidates is empty:
    store empty _engineIds; mark finalized; return     // no policy runs

  if HIPDNN_ATTR_ENGINEHEUR_FIND_FIRST_EXT is set:
    store candidates as _engineIds; mark finalized; return   // applicability probe; no policy runs

  devProps = queryDeviceProperties(handle)
  devicePropsSerialized = serializeDeviceProperties(devProps)              // §13.2
  devicePropsWrapper = wrapSerializedDeviceProperties(devicePropsSerialized) // hipdnnPluginConstData_t

  serializedGraph = graph.getSerializedGraph()  // hipdnnPluginConstData_t

  orderedPolicyIds = resolveHeuristicPolicyOrder()
    // §5.3.3 precedence: HIPDNN_HEUR_POLICY_ORDER env > HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT > built-in fallback
    // env: each token is name OR signed-decimal int64 ID (name → policyNameToId, ID → as-is)
    // descriptor: array is already int64 IDs (caller hashed names via policyNameToId)
    // fallback: { "SelectionHeuristic::Config", "SelectionHeuristic::StaticOrdering" } hashed via policyNameToId
    // No backend-injected entries; duplicates in the user-supplied list are kept.

  syncPolicySlots(orderedPolicyIds)
    // One SelectionHeuristic (hipdnnHeuristicPolicyDescriptor_t) per slot, created with the
    // hipdnnHeuristicHandle_t and policy ID for that policy; recreated if the list changed.
    // A policy that is not loaded, or whose descriptor creation throws, gets a null slot.

  for each distinct non-null hipdnnHeuristicHandle_t h among policy slots:
    try:
      hipdnnHeuristicHandleSetDeviceProperties(h, &devicePropsWrapper)  // once per handle, not per slot
    on exception:
      log warning; set every slot that uses h to null

  success = false
  for each slot i aligned with orderedPolicyIds:
    if slot i is null:
      continue
    try:
      selection = slot i
      selection.setEngineIds(candidates)
      selection.setSerializedGraph(&serializedGraph)
      if not selection.finalize():  // §8.9; declined
        continue
      candidates = selection.getSortedEngineIds()  // validated as subset/permutation
      success = true
      break
    on exception:
      log warning; continue

  if not success:
    throw HipdnnException(HIPDNN_STATUS_INTERNAL_ERROR, ...)  // no utilities::sortEngineIds fallback

  store candidates as _engineIds
  mark finalized
```

---

## 15. Public API notes

- **`HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT`:** Ordered list of pre-hashed **`int64_t`** policy IDs for **`EngineHeuristicDescriptor`**, overriding the built-in default ([§5.3](#53-ordered-policy-list-default-and-user-configuration)). Attribute type: **`HIPDNN_TYPE_INT64`** (array of IDs). The backend stores IDs as-is — callers (typically the frontend) hash policy names via **`policyNameToId`** before calling `setAttribute` because the `void*` + `int64_t elementCount` ABI does not cleanly carry variable-length string arrays. When this list changes between finalizes (a fresh descriptor with a different list), the per-finalize sync step recreates the owned **policy descriptor** objects ([§5.4](#54-two-tier-plugin-objects-handle-vs-policy-descriptor)).
- **`HIPDNN_HEUR_POLICY_ORDER` (optional env):** Comma-separated tokens; each token is a UTF-8 policy name **or** a signed decimal **`int64_t`** policy ID (per-token disambiguation by full-string `strtoll`). **Highest** precedence among user overrides so operators can override application-supplied lists at deploy time ([§5.3.3](#533-how-the-user-sets-orderedpolicyids)).
- **`HIPDNN_ATTR_ENGINEHEUR_RANKING_METRIC_EXT`** / **`HIPDNN_HEUR_RANKING_METRIC`:** The registered ranking metric the prediction policies order engines by (**`HIPDNN_TYPE_CHAR`**; `tflops` by default, or `time`), resolved env > attribute > default on every finalize and readable back as the effective metric ([§5.3.1](#531-well-known-policy-names-and-ids)). An unregistered name is **`HIPDNN_STATUS_BAD_PARAM`** at set for the attribute; in the environment it is logged and ignored, so the attribute or default applies. The effective metric is stamped into every **`HIPDNN_ATTR_ENGINEHEUR_RESULTS`** configuration. The per-query counterparts are **`HIPDNN_ATTR_ENGINE_PREDICTION_METRIC_EXT`** (the metric **`HIPDNN_ATTR_ENGINE_PREDICTION_EXT`** answers in) and **`HIPDNN_ATTR_ENGINECFG_RANKING_METRIC_EXT`** (the configuration's **`EngineConfig.ranking_metric`**, read by its prediction and by plan build); the environment variable overrides neither.
- **`HIPDNN_ATTR_ENGINEHEUR_MODE`:** The backend accepts only **`HIPDNN_HEUR_MODE_FALLBACK`**. The mode does not change the policy order.
- **`HIPDNN_ATTR_ENGINEHEUR_DEVICEPROP`:** Reserved; the descriptor returns **`HIPDNN_STATUS_NOT_SUPPORTED`** for it ([§6.3](#63-reserved-attribute-hipdnn_attr_engineheur_deviceprop)).
- **No requirement for new enums** per new policy: adding a policy is **deployment + registry order** (names in env/config; **`int64_t`** IDs from **`policyNameToId`** at registration time), not necessarily a new public enum value. The well-known **names** **`SelectionHeuristic::Config`** and **`SelectionHeuristic::StaticOrdering`** are **strings** in env/config and docs; their **`int64_t`** values are **`policyNameToId(...)`**, not enum members. Both are shipped as **backend built-ins** ([§10.1](#101-heuristicpluginmanager)), so they are always available without loading any external `.so`; users may freely omit or reorder either in **`HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT`** or **`HIPDNN_HEUR_POLICY_ORDER`**. **`HIPDNN_HEUR_CONFIG_PATH`** is consulted by the **`Config`** built-in only when that policy is in the resolved order. **`Graph.preferred_engine_id`** is independent of this attribute — the frontend reorders engine configs after **`finalize()`** returns ([§5.3.5](#535-the-config-built-in-policy)).
- **Headers:** **`HeuristicsPluginApi.h`** and **`heuristic_api_version.h`** in **plugin_sdk** contain the types and declarations in [§8](#8-c-abi-for-heuristic-plugins). **`HeuristicsPluginApi.h`** includes **`PluginApi.h`** for the base plugin symbols, but not the **engine** plugin API header.

**Frontend:** End-to-end flow from application / **hipdnn_frontend** through heuristic **`finalize()`** and policy configuration is described in [§16](#16-frontend-api-flow-mirror-engine-selection).

---

## 16. Frontend API flow (mirror engine selection)

This section ties [§5.3](#53-ordered-policy-list-default-and-user-configuration), [§10](#10-heuristicpluginmanager-and-resource-layer), and [§15](#15-public-api-notes) to the **hipdnn_frontend** pattern used for **computational** engines, so the path from user code to policy execution is explicit.

### 16.1 Engine configs after heuristic finalize

The frontend already drives **engine configuration selection** through the **engine heuristic descriptor** and **`HIPDNN_ATTR_ENGINEHEUR_RESULTS`**:

1. The application builds the operation graph and obtains an **`EngineHeuristicDescriptor`** (backend descriptor) for that graph and **`hipdnnHandle`**.
2. The caller **`backendFinalize`**s the heuristic descriptor (or follows the equivalent backend sequence that produces heuristic results).
3. **`hipdnn_frontend::detail::getEngineConfigs`** (see `frontend/include/hipdnn_frontend/detail/GraphDetail.hpp`) queries **`HIPDNN_ATTR_ENGINEHEUR_RESULTS`**: first the **count** of engine configuration descriptors, then fills **`ScopedHipdnnBackendDescriptor`** instances and reads each engine’s **`HIPDNN_ATTR_ENGINE_GLOBAL_INDEX`** after finalizing each engine config.

So **which engines appear as options** for a given graph is **dynamic**: it comes from **loaded engine plugins** and **`EnginePluginResourceManager::getApplicableEngineIds`**, materialized in heuristic results only after **`EngineHeuristicDescriptor::finalize()`** succeeds (baseline in [§3](#3-baseline-current-behavior), integration in [§14](#14-engineheuristicdescriptor-integration)).

### 16.2 Loaded heuristic policy enumeration

**Policy IDs** (heuristic) and **engine IDs** are both **`int64_t`** name hashes but serve **different roles**: a heuristic policy ID identifies a **selection policy**; engine plugins expose **engine** IDs for execution. Each loaded **heuristic** module — and each **backend built-in** registered through the same C ABI shape — contributes one or more stable **`int64_t`** IDs from **`hipdnnHeuristicPluginGetAllPolicyIds`** ([§8.2](#82-plugin-module-metadata)). The **ordered list** the outer loop uses (**`orderedPolicyIds`**) is **`int64_t`**, derived from user-supplied **policy name** strings per [§5.3.3](#533-how-the-user-sets-orderedpolicyids). Applications and tools need a **runtime view of which policies are actually available** for this **`hipdnnHandle`** — built-ins plus successfully loaded `.so` modules — analogous to **`getEngineInfos()`** (see [§10.2](#102-heuristicpluginresourcemanager)).

**Built-ins as the canonical adapter example.** **`SelectionHeuristic::StaticOrdering`** is the reference built-in (`backend/src/heuristics/static_ordering/`). Its source unit exports **`populateFunctionTable()`**, which fills a **`HeuristicPluginFunctionTable`** struct with function pointers matching the heuristic plugin C ABI surface from [§8](#8-c-abi-for-heuristic-plugins) (handle lifecycle, policy descriptor lifecycle, set-engine-ids, set-serialized-graph, finalize, get-sorted-ids). At **`HeuristicPluginManager`** construction time, **`registerBuiltIns()`** wraps that table via **`HeuristicPlugin::createBuiltIn(populateFunctionTable(), source_label)`** and inserts the resulting **`std::shared_ptr<HeuristicPlugin>`** into the same registry that `dlopen`-loaded modules use. Policy-ID lookup, version validation, duplicate-ID rejection, and per-handle plugin-handle creation all flow through that registry without distinguishing built-ins from external plugins. New built-in adapters follow the same recipe — add a `populateFunctionTable()` and a line in **`registerBuiltIns()`**.

**Enumeration API:**

- **Backend / handle scope:** **`HeuristicPluginResourceManager::getHeuristicPolicyInfos()`** returns metadata for every available policy — built-ins and accepted external modules. The C API in `backend/include/hipdnn_backend.h` exposes it as **`hipdnnGetHeuristicPolicyCount_ext(handle, ...)`** and **`hipdnnGetHeuristicPolicyInfo_ext(handle, ...)`**.
- **Frontend:** **`getLoadedHeuristicPolicyInfos(handle)`** (alias **`get_loaded_heuristic_policy_infos`**) in `frontend/include/hipdnn_frontend/HeuristicPolicyInfo.hpp` forwards to that C API. Each **`HeuristicPolicyInfo`** has **`policyId`**, **`policyName`**, **`pluginName`**, **`pluginVersion`**, and **`apiVersion`**. Built-ins report the plugin name `hipdnn-builtin`.

**How the list is determined dynamically:** Built-ins are present unconditionally as registered by **`registerBuiltIns()`** ([§10.1](#101-heuristicpluginmanager)). External entries are **only modules that passed load-time checks** ([§11](#11-versioning-and-compatibility-checks)): heuristic search paths ([§10.1](#101-heuristicpluginmanager)), **`validateBeforeAdding`**-style validation, and **unique `int64_t` policy ID** registration. Neither set depends on the operation graph. User-supplied entries — name tokens in **`HIPDNN_HEUR_POLICY_ORDER`** or pre-hashed IDs in **`HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT`** — are resolved against the combined registry when the list is resolved ([§5.3.3](#533-how-the-user-sets-orderedpolicyids)). Unknown IDs are not rejected up front; they produce a null `_policySlots` entry during the per-finalize sync and the outer loop skips them ([§9](#9-policy-plugins-and-the-outer-loop)).

### 16.3 End-to-end flow with policy order

1. **Optional:** The application calls the enumeration helper to list **available policies** (**`int64_t` IDs** and optional **names**) for loaded heuristic plugins plus any **built-in** adapters registered under the same scheme without a separate `.so`.
2. The application sets **`HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT`** on the **`EngineHeuristicDescriptor`** when overriding defaults, and/or relies on the environment or the built-in default ([§5.3.3](#533-how-the-user-sets-orderedpolicyids)).
3. **`backendFinalize`** on **`EngineHeuristicDescriptor`** runs the outer policy loop ([§14.2](#142-pseudocode-for-finalize)). **Candidate engine IDs** come from **`EnginePluginResourceManager::getApplicableEngineIds`** ([§10.4](#104-relationship-to-enginepluginresourcemanager)).
4. The application reads **`HIPDNN_ATTR_ENGINEHEUR_RESULTS`** to obtain **ordered engine configuration descriptors**; **`Graph::initializeEngineConfig`** picks among those configs using **preferred** or **default engine id** on the graph. **`Graph.preferred_engine_id`** is honored by the **frontend** as a post-hoc reorder of those engine configs in **`Graph::initializeEngineConfig`** — the backend heuristic loop does not see it ([§13.3](#133-graph-level-preferences-for-example-preferred_engine_id)).

In short: **policies** are **configured** by an ordered **policy name** string list (public surface) plus a **discovery API**; the backend uses **`int64_t`** policy IDs internally for matching. **Engines** remain **graph-dependent** and appear under **`HIPDNN_ATTR_ENGINEHEUR_RESULTS`** only after heuristic **`finalize()`** succeeds.

---

## 17. Testing

- **Unit tests** for each policy with **synthetic serialized device-properties** (FlatBuffer built from test `DeviceProperties`) and small **graph** FlatBuffers (no GPU required where possible).
- **Regression test** asserting that when **`SelectionHeuristic::StaticOrdering`** is in effect (for example via the default **policy name** order), ordering matches current `utilities::sortEngineIds` for a fixed candidate list.
- **Failure test** asserting that when **`orderedPolicyIds`** is empty, all IDs are unknown/skipped, or every policy declines, **`finalize()`** fails via **`HipdnnException`** / the same status path as other descriptor **`finalize()`** errors (no silent sort fallback).
- **Integration tests** with real graphs and devices when GPU is available.
- **ABI / loader tests** that load a minimal mock heuristic `.so`, verify **`hipdnnPluginGetApiVersion`**, **`hipdnnHeuristicPluginGetAllPolicyIds`** (each **`int64_t`** matches **`policyNameToId`** of the name from **`hipdnnHeuristicPluginGetPolicyName`**), **`HandleCreate` / `HandleDestroy`**, **`PolicyDescriptorCreate` / `Destroy`**, **`Finalize` / `GetSortedEngineIds`**, and reject wrong major versions; **negative tests** that a second module with a **duplicate** policy ID and a module whose policy name does not hash to its policy ID are **rejected at load** (same spirit as **`EnginePluginManager`** duplicate engine IDs—[§5.3.1](#531-well-known-policy-names-and-ids), [§11](#11-versioning-and-compatibility-checks)).
- **Policy order tests** that assert the effective default **name** list is **`{ "SelectionHeuristic::Config", "SelectionHeuristic::StaticOrdering" }`** (the fallback in [§5.3.2](#532-default-policy-order-user-strings) — backend does not inject any other entries), env wins over the descriptor attribute which wins over the default, and unknown **names** (after hashing) are skipped.
- **Built-in registration tests** that assert **`HeuristicPluginManager`** registers both **`SelectionHeuristic::Config`** and **`SelectionHeuristic::StaticOrdering`** as built-ins at construction (no `.so` loaded), the registered IDs equal **`policyNameToId("SelectionHeuristic::Config")`** and **`policyNameToId("SelectionHeuristic::StaticOrdering")`** respectively, and both are available regardless of **`hipdnnPluginLoadingMode_ext_t`** (including **`HIPDNN_PLUGIN_LOADING_ABSOLUTE`**) — see [§10.1](#101-heuristicpluginmanager).
- **`SelectionHeuristic::Config` built-in tests** ([§5.3.5](#535-the-config-built-in-policy)) covering: empty candidates → declines; **`HIPDNN_HEUR_CONFIG_PATH`** unset / file missing / unparsable → declines; serialized graph buffer invalid or graph has no nodes → declines; no rule matches → declines; matched engine ID not in candidates → declines; matched rule moves the matched engine to the front while preserving the order of other candidates; the JSON file is re-read on every invocation (no process cache).
- **Engine override config rule-matching tests** covering exact-dim and wildcard (`-1`) matching, exact-bucket-before-wildcard-bucket cross-partition ordering, declaration-order tiebreak within a bucket, stride patterns (exact, wildcard element, empty pattern matches any), op-name and tensor-count rejection, and JSON loading (valid, missing file, env-var unset).
- **Frontend `preferred_engine_id` tests** that assert **`Graph::initializeEngineConfig`** reorders the heuristic-ranked engine configs to put the requested engine first when its global index appears among the configs returned by **`HIPDNN_ATTR_ENGINEHEUR_RESULTS`**, and falls back to the heuristic order otherwise.
- **Lifetime tests** that assert destroying **`EngineHeuristicDescriptor`** invokes **`hipdnnHeuristicPolicyDescriptorDestroy`** for every owned slot (and does not leak plugin handles owned by **`hipdnnHandle`**).
- **Enumeration tests** that assert the handle-scoped **loaded heuristic policy** query ([§16.2](#162-loaded-heuristic-policy-enumeration)) matches modules that passed **`validateBeforeAdding`**-style checks, returns **`int64_t`** IDs consistent with **`hipdnnHeuristicPluginGetAllPolicyIds`**, and that frontend helpers return consistent metadata.

---

## 18. Risks and open questions

- **Policy list syntax:** The env var is comma-separated; empty tokens are skipped. Unknown policy IDs are skipped, not rejected. If skipping unknown entries leaves no successful policy, **`finalize()`** fails (no backend fallback sort).
- **Duplicate policy IDs in `orderedPolicyIds`:** Duplicates are kept. Two slots may reference the same loaded module (same **`hipdnnHeuristicHandle_t`**); each slot gets its own **`hipdnnHeuristicPolicyDescriptor_t`**.
- **Async selection:** When implemented, thread-safety of **`hipdnnHeuristicHandle_t`** / **`hipdnnHeuristicPolicyDescriptor_t`**, interaction with the **single-thread handle** rule ([§8.3](#83-plugin-handle-session-object)), and lifetime of **serialized graph** and **serialized device-properties** buffers, must be specified.
- **Invalid device-properties buffers:** If **`SetDeviceProperties`** receives malformed bytes, plugins return **`HIPDNN_PLUGIN_STATUS_BAD_PARAM`** (or an equivalent **`HIPDNN_PLUGIN_STATUS_*`** code). The backend then disables every policy slot that uses that plugin handle for this **`finalize()`**.

---

## 19. Glossary

| Term | Meaning |
|------|--------|
| **Engine plugin** | Shared library providing engines and execution; **distinct C ABI** from heuristic plugins. |
| **Heuristic / selection policy plugin** | Shared library implementing one or more outer-loop selection policies via the C ABI in [§8](#8-c-abi-for-heuristic-plugins). |
| **Heuristic C ABI** | `extern "C"` symbol set: base **`PluginApi.h`** symbols, policy enumeration (**`hipdnnHeuristicPluginGetAllPolicyIds`**, **`hipdnnHeuristicPluginGetPolicyName`**), **`hipdnnHeuristicHandle_t`** and **`hipdnnHeuristicPolicyDescriptor_t`** lifecycle, selection functions ([§8](#8-c-abi-for-heuristic-plugins)). |
| **HeuristicPluginManager** | Loads and validates heuristic `.so` files **and** registers backend built-ins (via **`registerBuiltIns()`**) at construction time; analogous to **EnginePluginManager** but **heuristic-only** symbols. |
| **HeuristicPluginResourceManager** | Handle-scoped facade for heuristic plugins (built-ins and external modules); stores **`hipdnnHeuristicHandle_t`** per module, paths; analogous to **EnginePluginResourceManager**. |
| **Outer loop** | Ordered list of policies; first applicable successful policy wins. If none succeed, **`finalize()`** fails via **`HipdnnException`** (normal backend error path); no built-in sort after the loop. |
| **Policy name** | UTF-8 string in **`HIPDNN_HEUR_POLICY_ORDER`** or docs; hashed with **`policyNameToId`** to form the **`int64_t`** used in **`orderedPolicyIds`** (and in **`HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT`**) and matched against the IDs from **`hipdnnHeuristicPluginGetAllPolicyIds`**. |
| **Policy ID (heuristic)** | **`int64_t`** stable identifier for one heuristic policy (built-in or external); **must** equal **`policyNameToId(policy_name)`** for that policy’s canonical name ([§5.3](#53-ordered-policy-list-default-and-user-configuration)). |
| **orderedPolicyIds** | Resolved **`std::vector<int64_t>`** for one **`finalize()`**, each element the hash of a user-configured **policy name**. The backend does **not** inject any entries. Effective default when nothing is overridden: **`{ policyNameToId("SelectionHeuristic::Config"), policyNameToId("SelectionHeuristic::StaticOrdering") }`** ([§5.3](#53-ordered-policy-list-default-and-user-configuration), [§5.3.2](#532-default-policy-order-user-strings)). |
| **`SelectionHeuristic::Config`** | Backend built-in heuristic in `backend/src/heuristics/config/`. It consults an exact-match autotune ranking cache, then **`HIPDNN_HEUR_CONFIG_PATH`** for JSON rules that match the graph's convolution, SDPA, or matmul shapes to a preferred engine. On a match it returns the candidates with the matched engine moved to the front; on every miss path (env unset, file missing, no rule match, matched engine not in candidates) it **declines** so the policy loop falls through to the next slot ([§5.3.5](#535-the-config-built-in-policy)). |
| **`HIPDNN_HEUR_CONFIG_PATH`** | Environment variable consulted by **`SelectionHeuristic::Config`** on every **`Finalize`**. Points at a JSON rule file that maps op + tensor-shape patterns to a preferred engine name or ID. Re-read each invocation; no process cache. |
| **Built-in heuristic** | Backend-shipped policy implemented as a populated **`HeuristicPluginFunctionTable`** (the heuristic plugin C ABI shape) and wrapped via **`HeuristicPlugin::createBuiltIn(...)`** at **`HeuristicPluginManager`** construction time. No `.so` is loaded. One table may expose several policy IDs. **`SelectionHeuristic::Config`**, **`SelectionHeuristic::StaticOrdering`**, and the prediction pair **`SelectionHeuristic::ModeA`** / **`SelectionHeuristic::ModeB`** (one table, both IDs) ship this way ([§5.3.1](#531-well-known-policy-names-and-ids), [§10.1](#101-heuristicpluginmanager), [§16.2](#162-loaded-heuristic-policy-enumeration)). |
| **DeviceProperties** | Device facts in the backend, held in the generated FlatBuffer object type **`DevicePropertiesT`**; serialized to FlatBuffer and passed in **`hipdnnPluginConstData_t`** for the heuristic C ABI ([§13.2](#132-serialized-device-properties-flatbuffer)). Plugins do not call HIP. |
| **SelectionHeuristic** | C++ facade over **`hipdnnHeuristicPolicyDescriptor_t`** for one policy **slot** on **`EngineHeuristicDescriptor`**; session state stays on **`hipdnnHeuristicHandle_t`**. |
| **Plugin heuristic handle** | **`hipdnnHeuristicHandle_t`**: session object per heuristic module per **`hipdnnHandle`**; **SetDeviceProperties** (serialized device-properties FlatBuffer in **`hipdnnPluginConstData_t`**), applied **before** **`PolicyFinalize`** on descriptors bound to that handle; heuristics **read** device facts from the handle as needed; **single-thread** use ([§8.3](#83-plugin-handle-session-object)). |
| **Plugin policy descriptor** | **`hipdnnHeuristicPolicyDescriptor_t`**: per-slot graph + candidate IDs + finalize result; **owned by** **`EngineHeuristicDescriptor`**. |
| **SerializedGraph** | FlatBuffer from the operation graph descriptor, carried in **`hipdnnPluginConstData_t`** ([§13.1](#131-serialized-graph)). |
| **Loaded heuristic policy enumeration** | Handle-scoped query of **`int64_t` policy IDs** with name and version metadata for policies that passed load-time validation; see [§16.2](#162-loaded-heuristic-policy-enumeration). |
