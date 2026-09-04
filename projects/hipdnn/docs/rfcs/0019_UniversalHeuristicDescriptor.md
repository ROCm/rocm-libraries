# RFC 0019: Universal Heuristic Descriptor (UHD): Data-Driven Kernel Selection

- Contributors: Jason Campbell, Chris Erb
- Parent: [RFC 0017 Universal Kernel Descriptor](0017_UniversalKernelDescriptor.md) — the "UHD + kernel selection" follow-up named in [RFC 0017 §12.2](0017_UniversalKernelDescriptor.md#122-follow-up-rfcs).
- Siblings:
  - [RFC 0020 Universal Engine Descriptor](0020_UniversalEngineDescriptor.md) — owns engine identity, the UED's `nodes` pattern, and the **symbol table matching it publishes**. That table is the binding this RFC's `features_signature` reads, and the set every UHD is validated against ([Section 6.1](#61-feature-sources), [Section 6.3](#63-contract-enforcement)).
  - [RFC 0018 Universal Match Descriptor](0018_UniversalMatchDescriptor.md) — the UMD's criteria, applicability evaluated over that same table. A sibling consumer of the binding, not its owner.
  - The **descriptor expression language**, the shared syntax criteria, dispatch formulas, and feature entries are written in. Specified in its own follow-up; see [Open Question 15](#operational).
- Related: [RFC 0007 Engine Selection and Heuristics Framework](0007_EngineSelectionHeuristicsFramework.md), [RFC 0013 Autotune](0013_Autotune.md) (the benchmarking substrate for heuristic generation, [Section 13](#13-model-generation-pipeline))
- Series: **0018** is the UMD's criteria, **0019** is this document (UHD + kernel selection), **0020** is the UED, graph matching, and symbol binding.

Sections marked **OPEN** identify decisions deferred to review or to a named follow-up. The design is
grounded in the heuristic-generation tooling — a training-and-export pipeline any package author runs to
produce a heuristic for their own kernels — and in rocKE's selection path.

## Table of Contents

1. [Overview](#1-overview)
2. [Scope](#2-scope)
3. [Ownership Model](#3-ownership-model)
4. [UHD Schema](#4-uhd-schema)
5. [Selection Flow](#5-selection-flow)
6. [Feature Extraction](#6-feature-extraction)
7. [Model Adapters](#7-model-adapters)
8. [Versioning and Compatibility](#8-versioning-and-compatibility)
9. [Performance](#9-performance)
10. [Applicability Flow](#10-applicability-flow)
11. [Engine Selection Integration](#11-engine-selection-integration)
12. [Observability](#12-observability)
13. [Model Generation Pipeline](#13-model-generation-pipeline)
14. [Package-Creation-Time Selection: Knobs and AOT Kernels](#14-package-creation-time-selection-knobs-and-aot-kernels)
15. [Phased Delivery](#15-phased-delivery)
16. [Risks](#16-risks)
17. [Open Questions](#17-open-questions)
18. [Glossary](#18-glossary)

---

## 1. Overview

A **UHD (Universal Heuristic Descriptor)** is a kernel-selection model stored as data, an individual,
reusable descriptor referenced by ID (a GUID). Given multiple kernels that apply to a graph, the UHD
ranks them and picks the best one. It replaces hand-coded dispatch logic with a declarative, drop-in
artifact.

The **UED (engine) owns its UHDs**: up to three role-scoped heuristics, arch-keyed, shared by every
pack that joins the engine ([Section 3.1](#31-descriptor-relationships)). The central one,
`sort_kernel_catalog`, ranks the catalog; unqualified "the UHD" below refers to it. The UED also owns the
**KMD (Kernel Metadata Descriptor)**, the explicit declaration of compilation knobs (tile size, block
size, split-K, dtype, and the like, each with a type and optional default) that distinguish kernel
variants. The KMD is the feature space the ranker ranks over, so the two are coupled: a *breaking* KMD
change forces a retrain, while additive changes and dispatch-only fields do not
([Section 3.3](#33-coupling-rules)).

This RFC defines:

1. **UHD schema** — how a selection model is described as data ([Section 4](#4-uhd-schema))
2. **Ownership model** — the UED owns its role-scoped UHDs and one KMD; KDPs join the engine ([Section 3](#3-ownership-model))
3. **Selection flow** — how a UHD ranks matched kernels ([Section 5](#5-selection-flow))
4. **Engine integration** — matcher applicability bubbles up before engines are ranked, and the UHD runs
   only after that, on demand. The generation tooling produces an engine-level estimate
   (`predict_engine_tflops`) and the catalog ranker (`sort_kernel_catalog`); two engine-selection
   policies (quick and thorough) consume them, with a rank-ordering fallback
   ([Section 10–11](#10-applicability-flow)). The policies remain a
   [RFC 0007](0007_EngineSelectionHeuristicsFramework.md) follow-up; this RFC supplies the heuristics and
   keeps the schema from foreclosing cross-engine comparison.
5. **Generation pipeline** — automated benchmarking and model export ([Section 13](#13-model-generation-pipeline))

---

## 2. Scope

| Level | Question | Owner |
|-------|----------|-------|
| **Engine selection** | Which engine handles this graph? | [RFC 0007](0007_EngineSelectionHeuristicsFramework.md) |
| **Kernel selection** | Which kernel within the engine? | **UHD** (this RFC) |

A UHD is the kernel-selection heuristic. It is part of the generic provider that
[RFC 0017](0017_UniversalKernelDescriptor.md) introduces — not a new host interface, not a policy
plugin. The two levels are not cleanly one-after-the-other, but the interleaving is specifically between
the **matcher** and engine selection — *not* between the UHD and engine selection
([Section 10](#10-applicability-flow)).

- **Matcher applicability bubbles up first.** Before engine selection can rank anything, it must know
  which engines even apply. For a descriptor engine, "do I apply?" is the **matcher (UMD) pass** at the
  descriptor layer; that result **bubbles up** so non-viable engines are ruled out *before* the first
  plugin-policy layer ranks the survivors. It is the **matcher** that runs ahead of engine selection.
- **The UHD runs strictly after applicability, and only on demand.** A UHD is never loaded or evaluated
  to decide applicability, and never eagerly. It runs when — and only when — something asks for a result
  that requires the ranked catalog: a policy requesting a performance estimate, a knob query needing the
  top-ranked default, or kernel selection for a winning engine. A policy that never asks (explicit user
  selection, criteria-based selection) never loads it ([Section 9.2](#92-loading-and-caching)).
- **The UHD's predictions *can* feed engine selection.** The generation tooling produces a cheap
  engine-level **expected-performance** estimate and the fine-grained config UHD; engine-selection
  policies **may** consume those to rank engines *by predicted performance*
  ([Section 10](#10-applicability-flow)) — a possible input, not a requirement, and policies are free to
  ignore them entirely. The policies themselves remain
  [RFC 0007](0007_EngineSelectionHeuristicsFramework.md)'s territory; this RFC provides the heuristics
  they may consult.

**In scope:** the UHD schema; ranking semantics over a pack's matched UKDs; selection-group membership
(the UED owns the UHD and KMD; a KDP joins the engine) and how KMD fields, UKD metadata, and the knobs
an engine exposes relate; the feature contract; model formats and their adapter seam; dependency
constraints; load/eval performance; the model-generation pipeline.

**Out of scope (this RFC):** the engine-selection outer loop itself ([RFC 0007](0007_EngineSelectionHeuristicsFramework.md)
owns it); autotuning / exhaustive search (device-access tuning is [RFC 0013](0013_Autotune.md)); the
matcher and launch machinery ([RFC 0017 §5–6](0017_UniversalKernelDescriptor.md#5-matching-and-the-umd)).

---

## 3. Ownership Model

### 3.1 Descriptor Relationships

```
UED (engine)
 ├── sort_kernel_catalog:       {arch → UHD id}   ← kernel-selection heuristic (the main UHD)
 ├── predict_engine_tflops:     {arch → UHD id}   ← optional: cheap engine-level estimate
 ├── predict_applicable_kernels:{arch → UHD id}   ← optional, future: candidate generator
 ├── metadata:  KMD id          ← one metadata schema per engine
 └── knobs: [...]               ← user-facing runtime parameters (= sort_kernel_catalog's $kernel.* axes)

KDP (pack)
 ├── engine: UED id       ← joins an engine
 ├── arch: [...]          ← target architectures
 ├── matchers: [UMD ids]  ← applicability criteria
 ├── dispatch: UDD id     ← launch descriptor
 └── kernelDescriptors:   ← child UKDs
      └── UKD
           ├── kernel_source
           └── metadata: {...}  ← fills KMD fields
```

In JSON form:

```jsonc
// The UED owns the heuristics + metadata schema; the KDP joins the engine and adds kernels.
{
  "schema": "hipdnn.ued/v1",
  "id":       "efc9eae4-…",              // engine identity
  "sort_kernel_catalog": {              // UHD: ranks this engine's kernels  <-- membership
    "gfx950":  "ae896b07-…",
    "default": "c93e17aa-…"
  },
  "predict_engine_tflops": {"gfx950": "7b1e9c40-…"},  // optional (Section 11)
  "metadata": "9ae0b215-…",             // KMD: the variant-field schema this engine's kernels fill
  "knobs":    ["split_k", "tile_m"]     // KMD fields this engine exposes (Section 3.2)
}

{
  "schema": "hipdnn.kdp/v1",
  "arch":      ["gfx942"],
  "matchers":  ["968156a8-…"],      // shared matcher set (UMD ids)
  "engine":    "efc9eae4-…",        // UED: the engine these kernels join (carries the UHDs + KMD)
  "dispatch":  "625df14f-…",        // UDD: how they launch
  "kernelDescriptors": [            // the selection group = these child UKDs
    {"id": "15b02840-…", "kernel_source": {/* ... */}, "metadata": {"tile_m": 128, "split_k": 1}},  // fills the KMD
    {"id": "562e3777-…", "kernel_source": {/* ... */}, "metadata": {"tile_m": 256, "split_k": 1}}
    // ... the family, differing only in metadata (their compile-time build config)
  ]
}
```

**An engine names up to three role-scoped UHDs, each mapped by architecture** ([RFC 0020 §4.4](0020_UniversalEngineDescriptor.md)):

| UED field | Role | When it runs | This RFC |
|---|---|---|---|
| `sort_kernel_catalog` | Ranks the catalog, picks the winning kernel | Kernel selection, after applicability | the main subject — the "config UHD" throughout |
| `predict_engine_tflops` | Cheap `f(graph) → expected perf` estimate | Engine selection, before any catalog is built | [Section 11.1](#111-the-engine-estimate-and-the-kernel-catalog-ranker)'s engine estimate |
| `predict_applicable_kernels` | Generates the candidate set to be ranked | During applicability, combinatorial/JIT case | [Section 4.3](#43-future-predict_applicable_kernels-when-there-is-no-catalog-to-rank)'s candidate generator |

The pipeline is `predict_engine_tflops` (rank engines) → `predict_applicable_kernels` (produce
candidates, when present) → `sort_kernel_catalog` (rank and pick). Each role is **independently
optional**, and each value is an **arch → UHD id** map resolved by exact `gcnArchName`, then a `default`
entry, then unavailable ([Section 8.3](#83-out-of-distribution-inputs)). Almost everything in this RFC
concerns `sort_kernel_catalog`; where a statement is specific to another role it says so. `knobs` derives
from `sort_kernel_catalog`'s `$kernel.*` feature axes and no other ([Section 3.2](#32-kmd-fields-and-knobs-as-the-heuristics-feature-axes)).

Three named fields, rather than a single id or an ordered list, keep each role independently optional
and independently versioned, let a loader resolve exactly the role a request needs without walking a
list, and give the two secondary roles (`predict_engine_tflops`, `predict_applicable_kernels`) a defined
home. `predict_applicable_kernels` runs *during* applicability, ahead of the
[Section 10](#10-applicability-flow) rule that the UHD runs strictly after it; that section records it as
the one exception.

### 3.2 KMD fields, and knobs as the heuristic's feature axes

There is **one** space of variant fields, not two.
[RFC 0017](0017_UniversalKernelDescriptor.md) establishes it: the **KMD declares the engine's variant
fields** (name, type, optional default), each UKD's `metadata` fills them, and a **knob is a KMD field
the engine exposes to the user** — a *name*, nothing more.

**The exposed set is not arbitrary. `UED.knobs` equals the set of KMD fields the UHD ranks on.**

| Concept | Where | What it is |
|---|---|---|
| **KMD fields** | **KMD** `fields`, filled by each UKD's `metadata` | The engine's full variant space (`tile_m`, `warp_n`, `split_k`, `dtype`, plus any dispatch-only fields a UDD consumes). |
| **Knobs** | **UED** `knobs`: a list of **field names** | Exactly the KMD fields the UHD reads as `$kernel.*` — the model's **kernel-side** feature axes, surfaced to the user. |

**This governs `$kernel.*` only.** A UHD's feature vector also draws on the problem (`$q.*`,
`$<node>.*`) and the device (`$device.*`), and computes values over all three inline
([Section 6.1](#61-feature-sources)). None of those are knobs, and none should be: a knob is something a
caller can *set*, and nobody sets the sequence length or the CU count as a way of picking a kernel. The
rule concerns the one namespace where a feature axis and a user-settable choice are the same thing.

Within that namespace the knob list becomes a **derived** artifact rather than an independent authoring
choice: generation produces the model, and the kernel fields that model ended up using *are* the knobs.
The consequences:

- **The settable surface is exactly the set of kernel axes that provably affect performance.** A knob
  exists because the data showed it matters. There is no kernel field a caller can set that the
  heuristic is blind to, and none the heuristic uses that a caller cannot reach.
- **Load-time validation is exact and bidirectional.** Assert
  `set(UED.knobs) == set($kernel.* fields reachable from UHD.features_signature)`. Stronger than today's
  one-way "every knob names a KMD field" check, and it catches a UED and UHD regenerated out of step.
  Note *reachable*: `$kernel.*` references nested inside a computed feature count too, so
  `{"ceil_div": ["$q.dims[2]", "$kernel.tile_m0"]}` makes `tile_m0` a knob even though the signature
  never names it on its own ([Section 6.2](#62-the-features_signature)).
- **Non-knob KMD fields are dispatch-only, and invisible to selection.** `UHD features ⊆ KMD fields`
  still holds; what is new is that the *complement* has a defined role — launch geometry and workspace
  detail a UDD reads ([Section 3.3](#33-coupling-rules)). The heuristic cannot rank on them.
- **Two kernels differing only in non-knob fields are indistinguishable to the model,** so they score
  identically and the deterministic `priority`-then-`id` tie-break decides
  ([Section 5](#5-selection-flow)). The loader emits a **warning** in this case: a pack carrying kernels
  the heuristic cannot choose between indicates either a missing feature or a redundant variant.
- **Knob changes and model changes are the same event.** Adding, removing, or renaming a knob means the
  UHD's feature set changed, which means a regenerated model. The two descriptors move together and are
  version-checked together ([Section 8.1](#81-descriptor-versions-and-uhd-coupling)).

**Benchmark wide, expose what the model keeps.** Generation still sweeps the full space — every KMD
field is exposed on the *generation* UED so every kernel is individually addressable and timeable
([Section 13.2](#132-benchmarking-via-hipdnn-autotune)). Feature selection then prunes the axes that do
not earn their place, and the **emitted** UED exposes exactly the survivors. This is the same signal the
knob-reduction loop uses ([Section 14.4](#144-pipeline-knob-reduction-hipdnn-jit-case)): pruning a weak
feature and dropping a knob are one action rather than two kept consistent by hand.

**Consequence:** the public knob list is only as stable as the model. A retrain that drops `split_k` as a
feature also removes it as a knob, breaking a caller that was setting it. Knob removal is therefore a
**major** UED version bump ([Section 8.1](#81-descriptor-versions-and-uhd-coupling)). A "sticky" knob set —
a superset of the model's features, retained for compatibility — is a possible mitigation. **OPEN:**
whether that superset is needed, and what an engine with no UHD (and therefore, under this rule, no knobs)
exposes — see [Open Question 18](#structural).

- **A knob's legal values come from the *catalog*, not the schema** — the values the field actually takes
  among the kernels matching *this* graph, never the KMD's theoretical range. Critically, the API reports
  each knob's values **independently**, so **the cross-product of legal knob values is not the set of
  legal kernels.** Given kernels `(split_k, tile_m) ∈ {(0,0), (1,1)}`, both knobs report the range
  `{0,1}` and the cross-product suggests four combinations — but `(0,1)` and `(1,0)` match no kernel.
  Nothing in the API today expresses which *combinations* are valid, and for a sparse catalog over many
  knobs the valid set is a vanishing fraction of the cross-product. This is a real gap with consequences
  for enumeration and for the generation pipeline — see
  [Section 13.2](#132-benchmarking-via-hipdnn-autotune) and [Open Question 12](#operational).
- **A knob's default is the UHD's choice.** Whatever the heuristic ranks first is the reported default, so
  leaving knobs alone reproduces the out-of-the-box selection. **This is one of the demand triggers that
  loads the UHD:** answering a knob query means ranking that engine's catalog — see
  [Section 9.2](#92-loading-and-caching).

**Knob ranges need a step, not just bounds.** Where a knob is numeric and its values are regular, the
descriptor should carry `[min, max, step]` rather than bare bounds, so a consumer can enumerate the axis
without guessing the increment. At runtime the reported range narrows to what the applicable catalog
actually contains, but the step is forwarded unchanged. **OPEN:** whether the step lives on the KMD field
(a property of the variant axis) or on the UED knob (a property of the exposure) —
[Open Question 12](#operational).

Filtering and ranking **commute**: setting `split_k = 4` keeps only kernels whose `split_k` is 4 and the
UHD ranks those. That holds only because **a UHD scores each kernel on its own metadata and the problem,
never relative to the rest of the catalog** — a hard requirement on any UHD adapter
([Section 5](#5-selection-flow)), not an assumption. A scorer that normalizes across the candidate set
is out of scope.

**Knobs are dynamic choices, not a static parameter space.** A knob does not describe a dimension the
user may freely set; it names a field whose *currently selectable values* are whatever the applicable
catalog happens to contain for this graph. The same knob can report `{64, 128, 256}` for one graph and
`{128}` for another, because applicability differs per graph. The API therefore guarantees that each
reported value is achievable *on its own*, and nothing more — combining values across knobs may name a
kernel that does not exist, and setting knobs that jointly match nothing is a legitimate empty result,
not an internal error. The distinction that matters is **who sets the value**: the kernel's build (every
KMD field, as `metadata`) versus the user (the exposed subset — which is the heuristic's feature set).

### 3.3 Coupling Rules

**The KMD is the schema for `$kernel.*`, and the UED owns both it and the UHD.** The UED references one
KMD and one `sort_kernel_catalog` UHD, and every child UKD's `metadata` fills the KMD's fields, validated
at load. The KMD is the feature space the UHD ranks over, so the two are coupled. The coupling is
conditional, matching [RFC 0017](0017_UniversalKernelDescriptor.md):

- **Additive KMD change** — a new field, or new legal values on an existing field — **requires no
  retrain until that change is exposed to selection.** The old feature space is still valid, so the
  existing model keeps ranking correctly; a field the UHD never reads costs it nothing.
- **Breaking KMD change** — removing or reinterpreting an existing field's values — **must land its
  retrain in the same change**, because a field the model was not trained on is not selected against
  until it learns it.

**The KMD is a *superset* of the UHD's features, not equal to it.** All `$kernel.*` fields the UHD ranks
on must be in the KMD, but the KMD may also carry fields the UHD never reads — values a **UDD** formula
consumes for per-kernel dispatch detail (launch geometry, workspace). So the relationship is
`UHD features ⊆ KMD fields`, and an additive field added purely for dispatch never touches the heuristic.
This gives the UHD a firm, checkable contract:

- **A UKD is one point in the KMD-declared field space** — its `metadata`, which is its unique key
  engine-wide and is validated as such at load. The model, however, only sees that point's projection
  onto the **knob** axes ([Section 3.2](#32-kmd-fields-and-knobs-as-the-heuristics-feature-axes)); two kernels
  sharing a projection are indistinguishable to it and are separated by the deterministic tie-break.
- **The collection of those points is the KDP** (a pack joining one engine, adding a matcher set, a UDD,
  and the kernel vector); the **UHD and KMD belong to the UED (engine)**, shared by every pack that joins
  it. `arch` is a KDP property, so one engine — and its UHDs/KMD — spans arches, with per-arch model
  selection handled by the UED's arch-keyed heuristic maps ([Section 3.1](#31-descriptor-relationships)).
- **The UHD's `$kernel.*` references must be a subset of the KMD fields, and must equal the UED's
  knobs** — a two-part load-time check ([Section 6.3](#63-contract-enforcement)). The KMD is the
  authority on *what fields exist*; the UHD picks *which subset* it ranks on and how it derives from
  them; the UED's `knobs` must then name exactly that subset. The fields left over serve the UDD.

**Engine-scoped, never per-pack.** Many KDPs may join the same engine and share its heuristics. A UHD
is never inlined per kernel or per pack; the UED names one per role, and one per arch within a role
([Section 3.1](#31-descriptor-relationships)).

**Arch-aware via `$device.*`.** A UHD spans architectures; `arch` is a KDP property, not a UHD property.
A pack carries a list of GFX targets (empty meaning arch-independent), matched exactly against the device
and gated before any matcher runs. A mapped model takes `$device.*` features (CU count, LDS size, and the
like) and generalizes across the arches its engine serves. Two mechanisms handle arch differentiation:
the arch-keyed heuristic maps ([Section 3.1](#31-descriptor-relationships)) select a distinct model where
architectures diverge, and `$device.*` features cover within-model variation where they overlap. Where an
arch needs different metadata or materially different heuristic behavior, it is served by a separate
engine rather than folded into one model. **OPEN:** See [Open Question 2](#schema-and-training)
(arch-aware model scope).

---

## 4. UHD Schema

A UHD is a small, reusable scoring recipe. It names an `adapter` (ranking mechanism), a
`features_signature` (model inputs), an objective, and — for model adapters — a model artifact.

Because a UED spans arches (`arch` is a KDP property in [RFC 0017](0017_UniversalKernelDescriptor.md)),
a UHD is **per-engine and arch-aware**: one model taking `$device.*` features so it generalizes across
the arches its engine serves — not one model per arch.

A UHD has two parts: a universal header every UHD carries, and an adapter-scoped body. The loader reads
the header before it instantiates an adapter implementation; the body is opaque to everything except that
adapter. This separation keeps a new adapter from touching the header schema, and lets a loader validate
identity, versioning, and the feature contract without understanding the ranking mechanism.

```jsonc
// tree_data — the default; a GBDT tree table, shipped as data with the engine's descriptor set
{
  // ── universal header (every UHD, every adapter) ──────────────────────────────
  "schema":  "hipdnn.uhd/v1",
  "id":      "ae896b07-80cd-473c-b3f4-6a8892998519",   // GUID; referenced by the UED (one per engine)
  "name":    "rocKE FMHA fwd selector",                // per-engine, arch-aware — not per-arch
  "adapter": "tree_data",                              // discriminant: selects the body schema (Section 7)

  // ordered model inputs; order + form must match training (Section 6)
  "features_signature": [
    "$device.cu_count", "$device.lds_size",            // device props → arch-aware
    "$kernel.tile_m", "$kernel.split_k",               // KMD fields, exposed as knobs (Section 3.2)
    "$q.dims[3]", "$q.dims[2]",              // graph node attr + tensor dim
    {"/": ["$sdpa_fwd.flops", "$sdpa_fwd.bytes"]},     // computed inline — no $derived.* (Section 6.4)
    {"ceil_div": ["$q.dims[2]", "$kernel.tile_m"]}    // tile quantization, also inline
  ],
  "categorical_encoding": { … },                       // string → code maps, generated (Section 6.5)
  "features_hash": "sha256:…",                         // contract guard: signature + encoding

  // the descriptor versions this heuristic was generated against (Section 8.1)
  "trained_against": {"ued": "1.3", "umd": "1.0", "kmd": "2.1"},

  "objective": "max",                                  // higher predicted score wins
  "score": {"units": "tflops", "calibrated": true, "transform": "log1p"},  // recover TFLOPS → Section 12

  // ── adapter-scoped body: key MUST equal the `adapter` value ──────────────────
  "tree_data": {
    "artifact": "fmha_fwd/model.bin"                   // ships as data with the engine descriptors
  }
}
```

Other adapters keep the same header and swap the body:

```jsonc
// native — a compiled scorer resolved by symbol name; the first adapter to land (Section 7.1)
{ …, "adapter": "native",
  "native": {"symbol": "rocke_fmha_fwd_score"} }       // resolved from the engine's registered symbols

// onnx — dependency-gated (Section 7)
{ …, "adapter": "onnx",
  "onnx": {"artifact": "fmha_fwd/model.onnx"} }

// static_order — no features, no derived, no hash, no model
{ "schema": "hipdnn.uhd/v1", "id": "…", "name": "…", "adapter": "static_order",
  "static_order": {"order": ["priority", "id"]} }

// custom_library — author-shipped .so behind a C ABI; features_hash advisory if it self-features
{ …, "adapter": "custom_library",
  "custom_library": {"library": "vendor_scorer.so", "symbol": "vendor.fmha_scorer",
                     "config": { … }} }                // symbol + typed config, never inline code
```

**On `static_order.order`.** The entries are *ordering criteria* (`priority`, then `id`), not a literal
list of UKD ids — the same deterministic arbitration
[RFC 0017 §5](0017_UniversalKernelDescriptor.md#5-matching-and-the-umd) defines. An explicit id list is a
reasonable future extension for pinning a known-good order; if one is supplied, ids present in the list
rank first in the given order and **any catalog entry not named falls through to the default criteria**,
so a stale list degrades gracefully rather than hiding kernels.

### 4.1 Field Reference

The normative header. A loader can validate every row here without instantiating an adapter.

| Field | Required | Type | Meaning |
|---|---|---|---|
| `schema` | yes | string | Schema id + major version (`hipdnn.uhd/v1`). Unknown major → reject. |
| `id` | yes | GUID | Descriptor identity; what the UED's `heuristic` references. |
| `name` | yes | string | Human-readable; appears in the selection trace. |
| `adapter` | yes | enum | Ranking mechanism, and the **key of the body object** ([Section 7](#7-model-adapters)). |
| `features_signature` | if the adapter features | ordered list | Model inputs, in training order ([Section 6.2](#62-the-features_signature)). |
| `categorical_encoding` | if a feature reads a string field | field → (value → code) | Generated during training; makes string→number conversion explicit ([Section 6.5](#65-categorical-encoding)). |
| `features_hash` | if `features_signature` | `sha256:…` | Fingerprint of the **resolved feature contract** — the canonicalized signature *and* `categorical_encoding` ([Section 6.3](#63-contract-enforcement)). |
| `trained_against` | if the adapter features | `{ued, umd, kmd}` semvers | The descriptor versions this heuristic was generated against ([Section 8.1](#81-descriptor-versions-and-uhd-coupling)). |
| `objective` | if the adapter scores | `max` \| `min` | Direction of the winning score. *Deferred in implementation* — the current layout fixes higher-wins; adapters normalize until this lands. |
| `score` | no | object | `units`, `calibrated`, `transform` — lets a consumer recover real TFLOPS ([Section 11.3](#113-cross-engine-comparison)). |
| `<adapter>` | yes | object | Adapter-scoped body; its key **must** equal `adapter`. Keys inside it are the adapter's concern, not the loader's. |

Two header rules govern the split:

- **The body key equals the `adapter` value.** One discriminant selects one body, with no ambiguity about
  which schema applies. A document carrying a body for an adapter it does not name is a load error, not a
  silently ignored field.
- **The loader validates the header; the adapter validates the body.** A UHD naming an adapter the
  provider does not implement produces a diagnosable "unsupported adapter" error rather than a parse
  failure, so a newer pack landing next to an older provider degrades predictably.

**OPEN — formal schema artifact.** This section is a reference table, not a machine-checkable schema.
Consistent with the UED RFC, the UHD ships a schema definition (FlatBuffers table or JSON Schema, matching
whatever the descriptor family standardizes on) so
validation is generated rather than hand-written. Header first; adapter bodies as each adapter lands.
*(See [Open Question 13](#operational).)*

### 4.2 Adapter Summary

`adapter` is a **single discriminant** — it subsumes [RFC 0017](0017_UniversalKernelDescriptor.md)'s
illustrative `kind` + `model.framework` into one field (`tree_data` ≈ `kind:model, framework:lightgbm`
shipped as data), and the body is an adapter-keyed union. Adding a new ranker (a static list, ONNX, a
new model family) is one more `adapter` value — the single discriminant is what makes that additive.

| `adapter` | What it is | Ranking | Model artifact |
|-----------|------------|---------|----------------|
| `static_order` | A fixed precedence with no learned model | Declared criteria / UKD `priority` | none |
| `native` | A scorer compiled into the engine, resolved by symbol name — **first to land** | Whatever the function returns | none (code, not data) |
| `table` | A CSV/lookup keyed by coarse problem buckets | Table lookup, then tie-break | with engine |
| `tree_data` | A GBDT tree table (LightGBM/XGBoost), in-tree walker — **default shipping path** | Score each candidate, argmax | with engine |
| `onnx` | An ONNX graph via a gated runtime | Score each candidate, argmax | with engine |
| `custom_library` | An author-shipped `.so` behind a small C ABI (drop-in escape hatch) | Whatever the library returns | with engine |

See [Section 7](#7-model-adapters) for adapter details.

**The model ships as data with the engine, not linked into the provider.** For every *data* adapter, the
body's `artifact` is a path resolved relative to the engine's descriptor set (the UED + its UHD + KMD +
model), which is itself standalone-droppable. The model is per-engine (owned by the UED), so there is
no `(arch,dtype)→artifact` table — the single arch-aware model serves every pack that joins the engine.
The `native` adapter is the deliberate exception: it names a symbol the engine already compiled in, so it
is *not* droppable — which is exactly why it is the bootstrap path and not the destination
([Section 7.1](#71-first-native)).

**OPEN — regressor vs. ranker.** The tooling trains a *regressor* on TFLOPS and argmaxes. A
learning-to-rank objective (LambdaRank/NDCG) optimizes ordering directly and may pick better *within*
an engine without needing calibrated absolute values. But a calibrated TFLOPS regressor is what makes
the **absolute, cross-comparable metric** of [Section 11.3](#113-cross-engine-comparison) possible; a
pure ranker forecloses that and leaves only the rank-ordering fallback. The regressor preserves the
absolute option, keeping ranking as the fallback rather than the only mode; this is the recommended
default. Decide per-UHD via `objective` / `score`, or standardize. See [Open Question 1](#schema-and-training).

### 4.3 Future: `predict_applicable_kernels`, when there is no catalog to rank

`sort_kernel_catalog` assumes a **finite, enumerable catalog** — the engine's AOT-compiled kernels that
pass the matchers. It scores those candidates and picks a winner. That holds for every AOT case and is
what this RFC specifies in full.

**JIT breaks the assumption.** When kernels are generated rather than pre-compiled, the candidate space
is the *combinatorial product of the knobs* — tile sizes × warp layout × pipeline × padding × … — which
can be far too large to enumerate, let alone score one candidate at a time. In that regime there may be
**no catalog to rank at all**, so "score every candidate and argmax" has nothing to iterate over.

This is the role the UED reserves as **`predict_applicable_kernels`**
([Section 3.1](#31-descriptor-relationships)): a first-stage heuristic reduces the intractable knob space
to a small, plausible candidate set — generating or bounding it rather than filtering an existing list —
and `sort_kernel_catalog` then does the usual predicted-TFLOPS ranking over that set. The second stage
reuses everything in this RFC (adapters, `features_signature`, computed features, the contract checks);
what differs is the first stage's **output contract**.

**The schema already has the slot; the role is what's deferred.** Because the UED names its heuristics as
three independently-optional, role-scoped fields rather than one id, standing up
`predict_applicable_kernels` later is not a schema change — the field is specified now and simply left
unset until the JIT path is real. What is *not* settled, and is deferred to
[Open Question 8](#structural):

- **The stage's output contract.** Emitting candidate configurations that do not yet exist is not
  ranking; it produces new KMD tuples and feeds the build path, which is closer to catalog *synthesis*
  ([Section 14](#14-package-creation-time-selection-knobs-and-aot-kernels)) than to selection. Whether
  that output belongs in a UHD at all, or in the package-creation tooling, is the open question.
- **Where it runs.** `predict_applicable_kernels` runs *during* the applicability query — it answers
  "what could this engine build for this graph?" — which is the single case that breaks the "UHD runs
  strictly after applicability" rule of [Section 10](#10-applicability-flow). That rule is a property of
  the AOT v1 pipeline, not an invariant, and this role is the anticipated exception.

---

## 5. Selection Flow

The generic engine first builds the **catalog**: the set of the engine's kernels that pass every matcher
for this graph ([RFC 0017 §5](0017_UniversalKernelDescriptor.md#5-matching-and-the-umd)), each carrying
its build `metadata`. The catalog is engine-scoped — the union across every KDP that joins the engine —
and the engine owns the UHD that ranks it, so the candidates and their selector arrive together.

Kernels joining one engine are mutually substitutable for the graphs they co-match (same op family), and
the model is trained to rank exactly that catalog. Kernel selection then proceeds:

1. **Start from the catalog.** The applicable kernels for this graph are the candidate set.
2. **Extract the shared features once.** Problem and device features are identical for every candidate,
   so they are computed once per graph ([Section 6](#6-feature-extraction)); only each candidate's
   `$kernel.*` metadata varies.
3. **Score each candidate.** Invoke the UHD's scorer per candidate — for a model adapter, one inference
   call per candidate over its feature row.
4. **Choose by objective.** `max` (or `min`) over the scores; the winner is the selected kernel.
5. **Tie-break deterministically.** On equal scores (or when the UHD declines), fall through to explicit
   UKD `priority`, then stable `id` — the same deterministic arbitration
   [RFC 0017 §5](0017_UniversalKernelDescriptor.md#5-matching-and-the-umd) defines. Declaration order
   is never used.
6. **A UHD is optional.** An engine that names no heuristic is valid; it is the starting state
   ([Section 13.1](#131-two-stage-workflow)). The catalog is returned in deterministic
   `priority`-then-`id` order and a **warning** is logged once, naming the engine. No authoring is
   required for a working engine; generating a heuristic is a later step the package owner opts into.

   > "Unranked" is `priority`-then-`id` order, never *discovery* order. Descriptor sets are loaded by
   > scanning a directory, so discovery order varies by filesystem and would rank a package differently
   > on two machines. `priority`-then-`id` is stable across runs, load orders, and machines.

7. **A failure degrades the result; it never fails the request.** Descriptor sets are drop-in data from
   potentially third-party authors. A malformed one must not take down the system, and must not fail
   after the engine has already claimed applicability. Each failure mode resolves to a usable answer plus
   a diagnostic:
   - **No model, or the scorer errors** → rank by `static_order` (priority + id). No ranking information
     is available, and priority order is a valid answer.
   - **Broken feature contract** — `features_hash` disagrees, a `$kernel.*` reference is dangling, or the
     KMD pairing is incompatible ([Section 6.3](#63-contract-enforcement)) → the model is not used (its
     inputs are not the ones it was trained on, so its scores would be wrong), an **error** is logged,
     ranking falls back to `static_order`, and the engine reports an estimated throughput of 0 so any
     engine with a real estimate outranks it in engine selection. The engine still answers, still
     dispatches, and loses on merit rather than by exception.

   A contract mismatch is a **CI gate**, not only a log line: descriptor-set validation runs over shipped
   packs in CI and fails the build there rather than degrading silently in the field
   ([Section 12](#12-observability), [Open Question 14](#operational)).

**The output is the ranked catalog, not just a winner.** Selection returns the candidates in score order;
the winner is simply its first element. Callers need the ordering, not only the argmax — a knob query
reports the top-ranked value as the default, autotune walks the ranked list, and engine selection reads
the top score as the engine's figure of merit. Whatever ranks the list — model or fallback — the order
must be **deterministic run-to-run**, which is why every fallback path terminates in `priority` then
stable `id` rather than an arbitrary order.

The winner is a single UKD, which then dispatches through its pack's UDD
([RFC 0017 §6](0017_UniversalKernelDescriptor.md#6-dispatch-and-workspace)). A UHD only ranks; it never
launches, mutates the graph, or touches device memory.

---

## 6. Feature Extraction

The feature vector is the contract between training and inference. It is the most failure-prone part of
the design, and generalizing it is the central problem this RFC addresses.

### 6.1 Feature Sources

A feature row is assembled from three sources, all available at plan time. All are drawn from the symbol
table the **engine's pattern publishes**: the UED carries a `nodes` block, and matching it binds every
tensor, dim, stride, and attribute the pattern names
([RFC 0020 §6](0020_UniversalEngineDescriptor.md#6-symbol-binding-what-the-pattern-publishes)). The UHD's
`features_signature` is one of that table's consumers, alongside UMD criteria and UDD dispatch formulas,
so matching, launch, and selection read one binding.

| Source | Namespace | Examples | Scope |
|--------|-----------|----------|-------|
| **Problem** | pattern variables (`$q`, `$k`), `$graph.*`, `$<node>.*` | `$q.dims[2]`, `$q.dtype`, `$graph.node_count` | Shared across candidates |
| **Device** | `$device.*` | `$device.cu_count`, `$device.lds_size` | Shared across candidates |
| **Kernel** | `$kernel.*` | `$kernel.tile_m`, `$kernel.split_k` | Per-candidate (from UKD `metadata`) |

Computed features are not a fourth source. Quantization, ratios, and intensity are expressions over those
same three, written inline in the `features_signature`; there is no `$derived.*` namespace and no
named-value block ([Section 6.4](#64-computed-features)).

**Dims are positional, not named.** A bound tensor exposes each dim as `$q.dims[i]` and each stride as
`$q.strides[i]`, plus the derived facts `$q.rank`, `$q.dtype`, `$q.stride_order`, and `$q.packed`. Sizes
that read like attributes are dims: for a rank-4 SDPA tensor laid out as
`(batch, heads, sequence, head_dim)`, batch is `$q.dims[0]`, head count `$q.dims[1]`, sequence length
`$q.dims[2]`, and head size `$q.dims[3]`. A node's `$<node>.*` namespace carries **scalar attributes**
only — `$sdpa_fwd.dropout_probability` and the like — so a signature that reads a size reads a dim.

> **Implementation note — the bound set is op-specific.** Which tensors a pattern binds, and what each
> dim index means, differ per operation; the engine is op-scoped, so its UHD's references are valid for
> that op by construction. The extractor evaluates whatever the signature names, and the engine's
> published symbol set is the authority on what exists
> ([Section 6.3](#63-contract-enforcement)).

Device features come from the same device-facts path [RFC 0007 §6](0007_EngineSelectionHeuristicsFramework.md#6-device-properties)
defines. Kernel features are the compilation knobs the engine's KMD declares
([Section 3.2](#32-kmd-fields-and-knobs-as-the-heuristics-feature-axes)); these distinguish candidates
and are what make argmax meaningful.

> **Implementation note — `$device.*` fields.** The device namespace exposes what rocminfo (or equivalent
> HIP runtime queries) provides and what proves predictive for kernel selection. Expected fields include
> `cu_count`, `lds_size`, `warp_size`, `sgpr_count`, `vgpr_count`, `max_waves_per_cu`, `memory_clock_mhz`,
> `memory_bus_width`, and `peak_bandwidth_gbps`; the set is extended additively as the checks that need it
> land. **Architecture is not among them for an AOT engine:** `arch` is a KDP property gated at selection
> ([Section 3.3](#33-coupling-rules)), and per-arch model choice is the UED's arch-keyed heuristic maps
> ([Section 3.1](#31-descriptor-relationships)), so a `features_signature` does not read `$device.arch`.

### 6.2 The `features_signature`

The feature contract is the UHD's inline `features_signature` — an ordered list of model inputs, bound
the same way a UDD's `args_signature` binds kernel arguments. Each entry is either a direct field (a
bare `$`-prefixed reference) or an expression over those fields, written as a `{"op": [args]}` tree.
Order and form must match training exactly.

```jsonc
"features_signature": [
  // --- direct fields ---
  "$q.dims[2]",                                    // sequence length (positional dim)
  "$q.dims[3]",                                    // head size (positional dim)
  "$device.cu_count",                              // device property
  "$kernel.tile_m",                                // KMD field (per-candidate)

  // --- computed features, evaluated by the shared interpreter ---
  {"log2": ["$q.dims[2]"]},
  {"/": ["$q.dims[2]", "$k.dims[2]"]},                           // aspect ratio
  {"ceil_div": ["$q.dims[2]", "$kernel.tile_m"]},                // tile quantization
  {"/": ["$sdpa_fwd.flops", "$sdpa_fwd.bytes"]}                  // arithmetic intensity (Section 13.6)
]
```

**The expression language is the descriptor expression language**, shared with UMD criteria
(boolean-rooted) and UDD dispatch formulas (value-rooted), over the same symbol table; one parser,
validator, and interpreter serve all three. A feature entry is value-rooted like a dispatch formula.

The language is specified in its own RFC and is not restated here. It is JsonLogic-like but simpler —
most visibly, variable references are bare `$`-prefixed strings rather than JsonLogic's `{"var": …}`
wrapper, resolved through the engine's binding. Any JSON string beginning with `$` is a reference; every
other JSON scalar is a literal. No external JsonLogic specification is normative for this RFC.
**OPEN:** the expression-language RFC is deferred, so its section references cannot be cited yet; see
[Open Question 15](#operational).

The `features_signature` is a third consumer of that same language, alongside UMD criteria
(boolean-valued) and UDD dispatch formulas (value-valued); a feature entry is value-valued like a
dispatch formula. One parser, validator, and interpreter serve all three.

The operator set covers the computed features required here; no extension is needed:

> `and or !` · `== != < <= > >=` · `in` · `all` · `+ - * / %` · `shape rank divisible value_or_default` ·
> `if` · `ceil_div min max abs pow log2 rsqrt`

Log-scale sizes, aspect ratios, tile/wave quantization, and intensity are all expressible with this set.
Evaluation is a bounded interpreter: it fails closed on an unknown symbol, a type error, or an invalid
operation, and uses checked-width integers. A computation outside this closed set (for example a bespoke
occupancy model) uses a compiled scorer ([Section 7](#7-model-adapters)) rather than extending the
interpreter.

**Implementation status.** The shared implementation provides the arithmetic and comparison operators
this RFC's features need (`+ - * /`, `min`, `max`, `ceil_div`, `abs`, `pow`, `log2`, `rsqrt`,
`value_or_default`, `if`) as a compile-once / evaluate-many `Expression<DataT>` over any
`getData(path) → Value` source, which is the shape the feature extractor requires
([Section 9](#9-performance)). Two caveats carry into this RFC:

- **`shape` is a compiler short-hand, not a runtime op**; the compiler lowers it at compile time. A
  feature spec treats `shape`/`rank`-style helpers as lowered forms rather than runtime operators.
- **The custom-operation (native-predicate) hook is deferred.** Until it lands, a UHD needing a
  computation outside the closed operator set has no in-language path; that is the boundary at which a
  compiled scorer ([Section 7](#7-model-adapters)) takes over.

**The engine's published symbols are the feature source.** The `features_signature` draws its problem
features from the symbol table the engine's pattern binds, plus `$device.*` and the candidate's
`$kernel.*` ([Section 6.1](#61-feature-sources)). One binding pass feeds matching, dispatch, and
selection, with no parallel feature-extraction path.

Because the pattern is engine-wide and singular, the UHD — which is itself engine-wide — has an
engine-level binding to resolve against, and every token it names is declared before it is read. A
reference the engine's pattern does not publish is a **load error** at build and at drop-in load, not a
runtime decline ([RFC 0020 §6](0020_UniversalEngineDescriptor.md#6-symbol-binding-what-the-pattern-publishes)).
The UHD is validated against that published set exactly as a pack's UMD and UDD are, which is what makes
the [Section 6.3](#63-contract-enforcement) contract check mechanical rather than per-pack.

**Derived features commonly needed:**

- **Arithmetic / algorithmic intensity** (FLOPs ÷ bytes) — the single most important derived feature
  for predicting compute-bound vs. memory-bound behavior.
- **Tile/wave quantization** — `num_tiles_*`, `total_output_tiles`, `tile_efficiency` (problem-vs-grid
  remainder waste). In GEMM sweeps this family is as predictive as intensity.
- **Aspect ratios** — `M/N`, `M/K`, `N/K` (shape skew).
- **Occupancy proxy** — `lds_usage_ratio` (and register pressure if available) → waves/CU.
- **Padding-fit** — `needs_padding_*` / `has_padding_when_needed_*` (problem × kernel padding interaction).

**OPEN:** See [Open Question 4](#schema-and-training) (derived feature set).

### 6.3 Contract Enforcement

The tooling code-generates the C++ featurizer to guarantee training and inference vectors are identical.
This RFC keeps that principle but makes the `features_signature` the single source of truth for both sides:

- **Inference:** A generic feature extractor walks the signature against the bound-variable table and
  device facts, producing the row in declared order. A generic extractor is preferred over code-gen
  because the sources are declarative and finite, requiring no per-op C++. Code-gen remains an option for
  a hot path.
- **Training:** The same signature drives the offline featurizer, so dataset columns match the runtime
  row by construction.

A four-part, model-agnostic contract check replaces the tooling's bare feature-count guard, and
generalizes to any ranker (LightGBM, ONNX, a custom scorer):

1. **Signature → published symbols.** Every non-`$kernel.*` reference the `features_signature` names —
   pattern variables and their `dims[i]`/`strides[i]`/derived facts, `$graph.*`, `$<node>.*`, and
   `$device.*` — must resolve against the symbol set the engine's pattern publishes
   ([RFC 0020 §6](0020_UniversalEngineDescriptor.md#6-symbol-binding-what-the-pattern-publishes)). An
   unresolvable reference is a load error at build and at drop-in load, with both descriptors named. The
   UHD is checked against the same published set as the engine's UMDs and UDDs, so one publisher serves
   all three consumers.
2. **Signature → KMD → knobs.** Two assertions over the same set. Let `F` be the `$kernel.*` fields
   reachable from the `features_signature`, including those nested inside computed (expression) entries:
   - `F ⊆ KMD.fields` — a feature can never read a variant field the kernels don't carry.
   - `F == set(UED.knobs)` — the exposed knobs *are* the model's feature axes
     ([Section 3.2](#32-kmd-fields-and-knobs-as-the-heuristics-feature-axes)). An inequality either way is a load
     error: a knob the model ignores would let a caller tune something selection is blind to, and a
     feature with no knob would hide a performance-relevant axis from the caller.

   The UED owns the KMD and the UHD, so this is an intra-engine check the pipeline enforces when it
   emits the engine and the loader re-checks. Because the generation tool derives the knob list *from*
   the trained feature set, the equality holds by construction — the check exists to catch a UED and UHD
   that were regenerated out of step, or hand-edited.
3. **Signature → model.** The UHD carries `features_hash`; the model artifact embeds the hash it was
   trained against (tree-table metadata, ONNX `metadata_props`, or a sidecar). At load, assert
   `model.trained_hash == UHD.features_hash`. This check works for every model adapter because it
   fingerprints the *input contract*, not the model internals.

   `features_hash` covers the signature and the categorical encoding. Because computed features are
   written inline rather than referenced by name ([Section 6.4](#64-computed-features)), the signature is
   the computation, so hashing it fingerprints what each feature means rather than only what it is called.
   A named `$derived.*` layer would lack this property: editing a named expression changes a feature's
   meaning while every signature string stays byte-identical, and the hash would have to substitute the
   definitions back in to detect the change. The categorical encoding is folded in for the same reason,
   as [Section 6.5](#65-categorical-encoding) changes value semantics without changing feature names.
   The input is canonicalized before hashing so that whitespace and key order do not produce spurious
   mismatches.
4. **Vector → input.** Each adapter verifies its artifact accepts the resolved vector: `tree_data`
   checks feature count, `onnx` checks input arity/shape.

A failed check disables the model rather than failing the request. These run at load, so a violation
means a mis-built or mismatched descriptor set, and the model's scores would be wrong rather than missing.
The response is the one [Section 5](#5-selection-flow) step 7 defines: the model is not used, an error is
logged, ranking falls back to `static_order`, and the engine reports an estimated throughput of 0. Because
descriptor sets are drop-in and may be third-party, the loader handles this without taking down the
provider and without failing after the engine has claimed applicability. The error is logged and gated in
CI ([Open Question 14](#operational)); excluding the malformed engine from selection entirely is a
defensible stricter alternative, but failing the request is not. `features_hash` is optional for
feature-less adapters (`static_order`, and `native` when it self-features) and advisory for a
`custom_library` that self-features.

**Feature-vector portability.** This RFC specifies a per-model spec over a shared source vocabulary: the
spec is per-model, because models differ, but every spec draws from one fixed, versioned source
vocabulary and one extractor, so there is a single implementation to trust. Configurability comes from
which fields a signature selects and what expressions it builds over them — extensible without being
open-ended, because the set of extractable fields is fixed.

### 6.4 Computed Features

Many of the features a model ranks on are not raw fields but **computed** ones: tile/wave quantization
(`ceil_div($q.dims[2], $kernel.tile_m0)`), aspect ratios, arithmetic intensity, occupancy proxies. These
are written inline in the `features_signature`, as expressions over the three sources
([Section 6.2](#62-the-features_signature)). There is no separate namespace and no named-value block: a
computed feature is an entry that is an expression rather than a bare `$` token.

Computed features are inline rather than named because a named `$derived.*` layer would add a second
binding mechanism, a second thing to version, and a second place a feature's meaning could live, for
benefits the compiler supplies instead:

- **Repetition has no runtime cost.** Computed features layer: `total_tiles` is the product of two tile
  counts, tile efficiency builds on both, so the same subexpression appears in several entries. The
  signature is compiled once at load into an expression tree
  ([Section 9.3](#93-efficient-evaluation-expressive-spec-fast-hot-path)), and the compiler hash-conses
  identical subtrees, evaluating each distinct one once per row. Common-subexpression elimination is the
  evaluator's responsibility, not the schema's.
- **Hoisting is automatic.** A subexpression referencing only `$kernel.*` is graph-independent and is
  cached per kernel; one referencing only `$q.*`/`$device.*` is invariant across candidates and belongs
  in the shared prefix. The compiler classifies each subtree by the namespaces it touches; the author
  does not partition them.
- **A single hash target.** The `features_signature` is the computation, so `features_hash` fingerprints
  it directly ([Section 6.3](#63-contract-enforcement)). A named block would permit editing an expression
  to change a feature's meaning while every signature string stayed byte-identical — a silent divergence
  that does not arise when the expression is the entry.

The tradeoff is verbosity: a repeated subexpression is repeated textually, and a long signature is harder
to read than a set of named parts. Because the signature is normally generated
([Section 13.6](#136-auto-deriving-a-first-pass-features_signature)) rather than hand-written, this trades
author ergonomics, which tooling covers, for one fewer runtime mechanism.

The dim↔tile correspondence is engine-specific (the tile field names are the engine's), so it does not
belong on the shared op vocabulary. The FLOP and byte terms are op-intrinsic and belong there instead, as
precomputed fields the binding system provides for every op rather than anything the UHD declares
([Section 13.6](#136-auto-deriving-a-first-pass-features_signature)).

### 6.5 Categorical Encoding

A model consumes numbers, but a KMD field may be a **string** (`dtype`, a pipeline name), a **bool**, or
an **int list**. Turning those into feature values must be done identically at training and inference;
otherwise the model ranks on a different meaning than it learned, and `features_hash` does not catch it
because the signature text is unchanged. This is the same class of silent divergence the inline-expression
rule closes for computed features ([Section 6.4](#64-computed-features)), and it requires the same
treatment.

The tooling generates the encoding while it gathers the data and ships it in the UHD. Training and runtime
run against the same KDP, so the set of values a field takes is observable during generation. The tool
records it as an explicit map and emits it alongside the model:

```jsonc
"categorical_encoding": {
  "$kernel.dtype":    {"fp32": 0, "fp16": 1, "bf16": 2, "fp8_e4m3": 3},
  "$kernel.pipeline": {"intrawave": 0, "interwave": 1}
}
```

The rules that make this correct:

- **Derived from observation, never implicit.** The encoding is not an ordinal by declaration order, a
  hash of the string, or the underlying enum value; each of those changes silently when a schema is
  edited or a pack is rebuilt. An explicit table is the only form that can be diffed and version-checked.
- **Covered by `features_hash`.** The encoding is part of the resolved feature contract, like the
  signature itself, so editing it invalidates the model's contract check rather than passing silently.
- **An unseen value at runtime is out-of-distribution, not an error.** A string the map does not contain
  means the catalog moved outside what the model was trained on: the
  [Section 8.3](#83-out-of-distribution-inputs) path (log, degrade), not a load failure. It is also the
  cheapest out-of-distribution signal available, being an exact lookup rather than a range check.
- **`bool` needs no table** (0/1). **Int lists have no scalar form**: a `features_signature` cannot
  reference an `INT_LIST` field directly and must reduce one through an inline expression (a length, an
  element, a comparison) so the reduction is explicit and hashed.

**Ordering.** Integer codes imply an order a tree model can split on (`dtype < 2`), which is meaningless
for an unordered category. Two mitigations apply, selectable per field: assign codes in a meaningful order
where one exists (dtype by byte width, so splits are interpretable), or one-hot the field where none
exists. Codes assigned by first-seen order are not acceptable. **OPEN:** whether to standardize per-field
ordering semantics — see [Open Question 17](#operational).

### 6.6 Example: mapping the current rocKE SDPA features

> **Illustrative.** This maps the **existing** rocKE FMHA-forward feature engine
> (`FmhaFeatureEngine`, 69 features) onto this RFC's namespaces, to show the model in practice and to
> validate that a real, in-use feature set decomposes cleanly. It documents how rocKE sources each
> feature *today*; it does not prescribe the final set.

rocKE's featurizer takes two dicts — a **`problem`** (the graph/problem) and a **`kernel`** (the
compiled variant's config) — and emits a fixed 69-`double` row. Those two dicts map directly onto the
RFC's two per-plan sources: `problem` → `$q.* / $<node>.*`, `kernel` → `$kernel.*` (the KMD fields).
Everything else is either `$device.*` (hardware) or an inline computation over those
([Section 6.4](#64-computed-features)). The full decomposition:

**Problem — `$q.*` / `$<node>.*` (from the `problem` dict) — 8 raw + 7 log + 4 derived-shape**

| rocKE feature | RFC mapping |
|---|---|
| `batch, seqlen_q, seqlen_k, nhead_q, nhead_k, hdim_q, hdim_v` | `$q.dims[0]`, `$q.dims[2]`, `$k.dims[2]`, `$q.dims[1]`, `$k.dims[1]`, `$q.dims[3]`, `$v.dims[3]` |
| `dtype_enc` | `$q.dtype` (encoded) |
| `log2_batch … log2_hdim_v` (7) | computed: `{"log2": ["$q.<dim>"]}` |
| `gqa_ratio` = nhead_q/nhead_k | `{"/": ["$q.dims[1]", "$k.dims[1]"]}` |
| `aspect_sq_sk` = seqlen_q/seqlen_k | `{"/": ["$q.dims[2]", "$k.dims[2]"]}` |
| `log2_ops` | `{"log2": ["$sdpa_fwd.flops"]}` |
| `decode_flag` = (seqlen_q ≤ 1) | `{"<=": ["$q.dims[2]", 1]}` |

**Kernel — `$kernel.*` (from the `kernel` dict = KMD fields) — 20**

| rocKE feature | RFC mapping |
|---|---|
| `pipeline_code` | `$kernel.pipeline` (encoded) |
| `tile_m0, tile_n0, tile_k0, tile_n1, tile_k1, tile_k0max` | `$kernel.tile_m0 …` — the tiling of attention's two GEMMs |
| `pad_s, pad_sk, pad_d, pad_dv` | `$kernel.pad_*` |
| `num_warps` | `$kernel.num_warps` |
| `mask, bias, lse, dropout, logits, sink, skip, qscale, paged` | `$kernel.*` — **kernel capability flags** (whether the compiled variant supports that feature); note these are sourced from the `kernel` dict, i.e. they are build config, not problem attributes |

**Computed inline — the quantization / fit / occupancy families**

| rocKE feature | RFC inline expression (schematic) |
|---|---|
| `arithmetic_intensity` | `ops / mem`, where `ops = 2·batch·nhead_q·seqlen_q·seqlen_k·(hdim_q+hdim_v)` and `mem` sums Q/K/V/O bytes — the `{"/": ["$sdpa_fwd.flops", "$sdpa_fwd.bytes"]}` of [Section 13.6](#136-auto-deriving-a-first-pass-features_signature) |
| `num_tiles_m` = ⌈seqlen_q/tile_m0⌉ | `{"ceil_div": ["$q.dims[2]", "$kernel.tile_m0"]}` |
| `num_tiles_k` = ⌈seqlen_k/tile_n0⌉ | `{"ceil_div": ["$k.dims[2]", "$kernel.tile_n0"]}` |
| `total_tiles` = batch·nhead_q·num_tiles_m·num_tiles_k | product of the above with `$q.*` |
| `tile_eff_sq, tile_eff_sk, overall_tile_efficiency` | remainder-based tile-efficiency ratios |
| `cu_utilization` = total_tiles/num_cus | `total_tiles` expression ÷ `$device.cu_count` — spans computed × device |
| `tile_volume, tile_area` | products of `$kernel.tile_*` (graph-independent — cacheable per kernel) |
| `lds_usage_estimate` | `(tile_m0·tile_k0 + tile_n0·tile_k0)·dtype_bytes` |
| `lds_usage_ratio` | `lds_usage_estimate` expression ÷ `$device.lds_size` |
| `ratio_d_to_tk0, ratio_dv_to_tn1` | `$kernel` ratios (hdim vs. tile) |
| `sq_le_tm0, sk_le_tn0, d_eq_dv, gqa_flag` | boolean fit/shape flags |
| `total_q_elems, total_kv_elems` | element-count products |

**Device — `$device.*` — 8**

`hw_num_cus, hw_simds_per_cu, hw_total_simds, hw_shader_engines, hw_max_clock_mhz, hw_wavefront_size,
hw_lds_capacity, hw_num_xcd` → `$device.cu_count`, `$device.simds_per_cu`, … (from the device-facts path
of [Section 6.1](#61-feature-sources)).

*(`feature_count` is a bookkeeping constant, not a real feature.)*

**What this validates:**

- The 69 features partition exactly into the published namespaces — **no feature falls outside** the
  engine's bound tensors, `$kernel.*`, and `$device.*`, whether read raw or computed over. The published
  vocabulary is sufficient for a real model.
- Every rocKE "problem" quantity resolves to a **positional dim** of a bound tensor, not a node attribute
  ([Section 6.1](#61-feature-sources)): `batch`, `nhead_q`, `seqlen_q`, and `hdim_q` are `$q.dims[0..3]`,
  and `seqlen_k` / `nhead_k` are `$k.dims[2]` / `$k.dims[1]`. `dtype_enc` is the derived fact `$q.dtype`,
  encoded per [Section 6.5](#65-categorical-encoding).
- The split is roughly **~20 problem, ~20 kernel, ~20 computed, ~8 device** — about 30% of the vector is
  expressions rather than raw fields, which is the load the expression language carries
  ([Section 6.4](#64-computed-features)).
- rocKE's `mask/bias/lse/…` are **kernel capability flags** sourced from the kernel config, so they are
  `$kernel.*` (KMD fields) rather than problem attributes, even though the problem also has a mask. The
  KMD field records whether the compiled variant supports masking, which is what the model ranks on;
  problem-side mask presence is a matcher concern.
- `arithmetic_intensity` here is the **problem/ideal** intensity (identical across all candidates for a
  graph) — a shared-prefix context feature, not a per-candidate discriminator; the algorithm-level
  differences are carried by the `$kernel.*` tile fields and the quantization derivations, exactly as
  [Section 13.6](#136-auto-deriving-a-first-pass-features_signature) describes.

---

## 7. Model Adapters

The question "LightGBM, CSV, or a separate library?" is really about how model content reaches the
scorer, and it maps onto [RFC 0017 §8](0017_UniversalKernelDescriptor.md#8-adapters-and-extensibility)'s
adapter model. A UHD names an `adapter`; the adapter turns content into a scorer. Adding a new ranker
is one more adapter value. Adapters come in the same two delivery classes as kernel-source adapters.

**Design constraint:** The model travels as data with the engine, not linked into the provider. The
engine's descriptor set (UED + UHD + KMD + model) must be a standalone drop-in next to an already-shipped
provider, exactly like packs' `hsaco`/`kpack` code objects. This rules out statically linking the model
as the shipping mechanism — the model must be loadable data the running provider reads. The constraint is
on *linkage*, not compilation: a model may be compiled (e.g. a Treelite `.so`) provided it ships as a
loadable artifact rather than baked into `libhipdnn_provider.so`.

| Adapter | Runtime dependency | Standalone drop-in? | Notes |
|---------|-------------------|---------------------|-------|
| *(none named)* | none | n/a | **A UHD is optional.** No heuristic → deterministic `priority`/`id` order plus a warning ([Section 5](#5-selection-flow) step 6) |
| `static_order` | none | Yes (always available) | The same behavior, stated explicitly rather than inferred |
| `native` | none | **No — compiled into the engine** | Scorer function resolved by symbol name; the bootstrap path |
| `table` | none | Yes | CSV lookup by coarse problem buckets |
| `tree_data` | none | **Yes — default shipping path** | GBDT tree table + in-tree walker |
| `onnx` | ONNX runtime | Yes, if dep present | Opt-in, dependency-gated |
| `custom_library` | none on provider | Yes | Engine ships its own `.so`, `dlopen`'d through a tiny C ABI |

The ladder mirrors [RFC 0017](0017_UniversalKernelDescriptor.md)'s native → data → escape-hatch
progression, and the landing order ([Section 15](#15-phased-delivery)) follows the cost of each stage:
`native` first, because it needs no new file format, parser, or extractor and proves the UHD seam with a
function; `tree_data` next, as the shipping path; `custom_library` last, because it requires a trust and
dependency audit.

### 7.1 First: `native`

The scorer is a function compiled into the engine and named in the UHD by symbol, resolved through the
same symbol-registration mechanism the ingestor uses for matchers and dispatch handlers. The UHD still
carries `objective` and `score`, so a native scorer participates in ranking and cross-engine comparison
identically to a model; the ranking seam is exercised end to end while the scorer behind it is ordinary
C++.

This is the simplest implementation and the fastest route to a working UHD, so it lands before the
data-driven adapters. It is **not** a drop-in: changing the heuristic requires recompiling the engine, so
it does not serve the data-driven, independently-shippable goal. It is scaffolding for the seam, an escape
hatch for heuristics no model expresses, and a baseline against which `tree_data` is measured.

`features_signature` is optional here: a native scorer may consume the standard feature row, holding it to
the same contract as a model, or featurize from the bindings directly.

### 7.2 Default: `tree_data`

The provider ships one generic GBDT tree-walker; the engine ships the model as a data artifact (a tree
table of feature indices, thresholds, and leaf values) that the walker reads. GBDT trees are inexpensive
to evaluate (on the order of a few hundred lines), and the tooling dumps the model to a walkable
structure. The adapter has zero runtime dependency, is fully standalone, and is verifier-gated; it is the
hsaco-equivalent for heuristics. Its one constraint: the provider must already support the model *family*
(a new family is a provider change).

**Artifact format:** a FlatBuffer tree schema (recommended) — consistent with data-SDK serialization,
`Verifier`-gated, with additive evolution. The alternative is parsing LightGBM `model.txt` at load, which
lowers author friction but requires a bespoke parser to harden. **OPEN:** See
[Open Question 3](#schema-and-training).

### 7.3 Escape hatch: `custom_library`

For a model the in-tree walker does not cover, the engine ships its own compiled scorer `.so`, `dlopen`'d
through a small C ABI (`score(const double* feats, ...) -> double`). Treelite generates such a `.so` from
a tree model. Any model family is supported, under the author-native-code trust class of
[RFC 0017 §10](0017_UniversalKernelDescriptor.md#10-packaging-and-delivery).

The constraint is on linkage, not compilation. Compiling a model *into* the provider makes it
non-portable to third-party provider builds; a model may still be compiled (a Treelite `.so`) provided it
ships as a loadable artifact alongside the engine's descriptor set rather than statically linked into
`libhipdnn_provider.so`. `custom_library` mirrors
[RFC 0017](0017_UniversalKernelDescriptor.md)'s native-predicate and custom-plan escape hatches: the
author ships a `.so`, the provider `dlopen`s it. **OPEN:** See [Open Question 11](#operational)
(dependency + trust audit).

`custom_library` and `native` both run compiled code; they differ in who ships it and when. `native` is a
symbol already inside the engine binary — no loading, no ABI boundary, no trust question, and no
independent shipping. `custom_library` is an author-supplied `.so` the provider `dlopen`s, which restores
drop-in delivery at the cost of the dependency and trust questions [Open Question 11](#operational)
covers. They sit at opposite ends of the same tradeoff, which is why `native` lands first and
`custom_library` last.

### 7.4 Initial Support

The initial adapters are **`static_order`, then `native`, then `tree_data`**. `static_order` is trivial
and always available; `native` proves the seam with a compiled function and needs no new format;
`tree_data` is the data-driven shipping path. CSV `table` is a low-cost addition for coarse bucketed
heuristics. `onnx` and `custom_library` are added when a concrete need appears, the latter gated on the
trust audit.

---

## 8. Versioning and Compatibility

### 8.1 Descriptor Versions and UHD Coupling

Because the UED co-owns the KMD and UHD, a KMD change may invalidate the trained model. The obligation is
conditional, matching [RFC 0017](0017_UniversalKernelDescriptor.md) (see
[Section 3.3](#33-coupling-rules)):

- **Additive change (new field, or new legal values on a field):** no retrain is required until the
  change is exposed to selection. The old feature space is still valid, so the existing model keeps
  ranking correctly. A field added purely for the UDD (dispatch/launch geometry) does not affect the UHD,
  since `UHD features ⊆ KMD fields`.
  - *Caveat:* dropping in a KDP whose kernels vary along a field the model was **exposed to but not
    trained across** can degrade ranking, because the model ranks on values it never saw. This is a
    training-coverage gap, not a schema break; the fix is a retrain, not a load failure.
- **Breaking change (remove or reinterpret an existing field's values):** the retrain must land in the
  same change. A removed or reinterpreted field the model still references is caught at load.
- **Renaming a field:** treated as remove plus add — a breaking change on the old name.

Three descriptors can invalidate a heuristic, not one. The KMD defines the `$kernel.*` feature space; two
others change what a trained model reads:

| Descriptor | Why a change can invalidate the model |
|---|---|
| **KMD** | Defines the `$kernel.*` fields. Removing or reinterpreting one breaks the feature space directly. |
| **UED** | Its `knobs` **are** the model's `$kernel.*` feature axes ([Section 3.2](#32-kmd-fields-and-knobs-as-the-heuristics-feature-axes)), so a knob-set change and a model change are the same event. Engine identity and op scope changes land here too. |
| **UMD** | Determines the catalog and binds the `$graph.*` tokens the features read. A matcher change can alter which kernels are candidates, or change what a bound token *means* — both invisible to a feature-name check. |

The UHD records all three, as the versions it was generated against
([Section 4.1](#41-field-reference)):

```jsonc
"trained_against": {"ued": "1.3", "umd": "1.0", "kmd": "2.1"}
```

**Enforcement — the concrete rule.** Each of the three descriptors carries a semantic version. At load,
for every entry in `trained_against`:

```
compatible  ⇔  trained_against.<d>.major == <d>.version.major
           &&  trained_against.<d>.minor <= <d>.version.minor
```

Major bump = breaking, disable the model. Minor bump = additive, still compatible. A UHD newer than the
descriptor it names (`trained_against.minor > actual.minor`) is also incompatible — it may reference
something that descriptor does not yet declare, which catches a half-updated descriptor set.

For the KMD this maps onto the change classes above exactly as intended:

| KMD change | Version bump | Effect on an existing UHD |
|---|---|---|
| Add a field, or add legal values to one | **minor** | Still compatible — `UHD.minor <= KMD.minor` holds. The model keeps ranking on the fields it knows. |
| Remove, rename, or reinterpret a field | **major** | Incompatible — the model is disabled until retrained and restamped. |
| Change the UED's knob set at all | **major** | Under [Section 3.2](#32-kmd-fields-and-knobs-as-the-heuristics-feature-axes) the knobs *are* the model's feature axes, so any change to them means a different model. The two are regenerated together and versioned together; a UED and UHD that disagree on the knob set fail the [Section 6.3](#63-contract-enforcement) equality check. Knob **removal** is additionally a caller-visible API break. |

A mismatch is a contract violation and takes the [Section 6.3](#63-contract-enforcement) path — the model
is disabled, an error is logged, ranking falls back to `static_order`; the request never fails. The same
applies to any `features_signature` reference to a field the KMD no longer declares. The trace surfaces a
training-coverage warning when the catalog spans a field value outside what the model was trained on
([Section 8.3](#83-out-of-distribution-inputs)).

> **Not yet in the layout.** None of the three descriptors carries a version field today
> ([PR #10606](https://github.com/ROCm/rocm-libraries/pull/10606)); `EngineDescriptor::sdkVersion` is the
> *graph schema* axis and is a different thing. Adding them is cheap now and awkward once descriptor
> sets ship.

### 8.2 Model Updates

Models ship as data artifacts. Update path:

1. Drop new engine descriptor set (UED + UHD + model) alongside existing
2. Provider loads new descriptors on next initialization
3. Rollback: restore previous descriptor set

No provider recompile required.

### 8.3 Out-of-Distribution Inputs

Unknown architecture is the visible case of a general problem: the model has expectations about the values
it was trained on, and nothing in the schema prevents it from being handed others. A new GPU arch, a
dropped-in pack whose kernels use tile sizes the sweep never covered, and an additive KMD field newly
exposed to selection all produce the same failure — the model returns a value, with confidence and no
basis for it. Tree models are especially prone, extrapolating by returning the leaf value of the nearest
region seen in training, which can be arbitrarily wrong for a distant input.

This is an unmitigated gap in v1, and it is distinct from a contract violation: the contract holds, every
field resolves, and the hash matches. Only the values are new. Two approaches apply, with a real tradeoff:

- **(a) Accept it and manage it by versioning.** Treat out-of-distribution input as a known error class
  and prevent it procedurally: track which arches and value ranges a model was trained on, and require a
  retrain-and-republish when a pack or platform moves outside them. Zero runtime cost, but detection is
  entirely process discipline — nothing catches a violation in the field.
- **(b) Ship training-coverage metadata and check at runtime.** The model artifact carries the observed
  range (or value set) per feature; the loader compares the resolved row against it and downgrades to
  `static_order`, or keeps the model but flags the score as low-confidence, when a feature falls outside.
  Stronger detection, at the cost of per-evaluation checking latency and of publishing the training
  distribution inside a shipped artifact, which not every author will accept.

This RFC specifies **(a) for v1, with two exact-match exceptions.** Both are discrete lookups rather than
per-feature range checks, so they cost little and catch the highest-impact cases:

- **Arch** — compare the device's GFX name against the arch set in the model's training metadata. This is
  a **loader-side** check against a resolved device fact, not a `features_signature` read: `arch` is a KDP
  property gated at selection and is not a token an AOT signature may reference
  ([Section 6.1](#61-feature-sources)). The GFX name is already resolved for the pack-level arch gate and
  for selecting the arch-keyed heuristic ([Section 3.1](#31-descriptor-relationships)), so it costs
  nothing additional.
- **Categorical values** — a string outside the UHD's `categorical_encoding`
  ([Section 6.5](#65-categorical-encoding)) is an exact-lookup miss, and therefore free.

The artifact format keeps per-feature coverage metadata additive, so (b) remains available without a
format break. In every case the degradation is the [Section 5](#5-selection-flow) one: fall back to
`static_order`, log, and never fail. **OPEN:** See [Open Question 16](#operational).

---

## 9. Performance

Selection runs on the plan-build path, so its cost must be small and paid at most once per distinct need.

### 9.1 Dependencies

- **Zero new runtime dependency for the default path.** The `tree_data` adapter's evaluator is in-tree,
  so the default shipping path adds no runtime library. The provider cannot grow a hard `liblightgbm` link.
- **Build-time LightGBM is acceptable.** Training and tree-table conversion run offline in the pipeline
  ([Section 13](#13-model-generation-pipeline)), never in the shipped runtime.
- **A `custom_library` scorer `.so` carries its own inference** — no provider dependency; the engine
  owns whatever it linked (e.g. a Treelite-generated evaluator).
- **`liblightgbm` at runtime is opt-in only**, behind a `lightgbm_native` adapter, for environments
  that already have it. Never a default.
- **FlatBuffers / data-SDK are already in-tree** and are the natural carrier for the `features_signature`
  and any serialized model-table.

### 9.2 Loading and Caching

- **Load on demand, and never before applicability.** Two rules govern loading:
  1. The UHD is never consulted to decide applicability; that is the matcher's job
     ([Section 10](#10-applicability-flow)). No UHD is loaded, parsed, or evaluated until the engine has
     been found applicable for the graph.
  2. After that, it loads only when something asks for what it produces. There is no eager ranking and no
     speculative load. The demand triggers are a policy requesting an estimated throughput to rank
     engines, a **knob query** (the reported default is the UHD's top-ranked value,
     [Section 3.2](#32-kmd-fields-and-knobs-as-the-heuristics-feature-axes)), and kernel selection. A
     policy that ranks by explicit user choice or fixed criteria asks for none of these and loads no UHD.

  Ranking can be triggered for an engine that goes on to lose — a policy may ask every applicable engine
  for an estimate before picking one — so the model must be cheap to load and cheap to rank with. A
  provider that never sees FMHA never parses the FMHA model, and a policy that never asks for estimates
  never parses any.
- **Model cache — per engine.** After first load the parsed model / tree table / native handle is cached
  **on the engine**, matching the mechanism in the kernel-ingestor foundation
  ([PR #10606](https://github.com/ROCm/rocm-libraries/pull/10606)): lazily loaded, then held for the
  engine's lifetime. Since a UHD is owned by exactly one UED, an engine-scoped cache has no sharing to
  miss. It must not be **handle-scoped**: RFC 0017 moves caching off the handle, because a handle can be
  swapped between calls, rebound to another device, or destroyed while a plan
  built through it is still in use, so handle lifetime says nothing about cache validity. A
  provider-wide static cache only becomes interesting if UHDs are ever shared across engines, which the
  ownership model does not currently allow.
- **Result cache on the descriptor cache key.** Selection is a pure function of (feature vector,
  candidate set). Reuse RFC 0017's applicability cache key — **`(engine id, graph id, device id)` plus
  the inventory generation counter** — rather than a bespoke fingerprint. "Engine id" here is the
  **64-bit id** hipDNN keys on (the FNV-1a hash of the UED `name`), not the descriptor GUID;
  [RFC 0020 §3](0020_UniversalEngineDescriptor.md) keeps those two id spaces distinct. The device id
  because it is what `$device.*` resolved against, and the generation counter so a newly dropped-in pack
  invalidates. Where the cache already lives **on the engine** — as it does in the ingestor foundation —
  the engine id is *implicit* in the cache's location rather than absent from the key. In practice this
  is the **catalog cache** that foundation maintains: the ranked order rides along with the catalog it
  ranks rather than living in a second structure.

- **In-memory: the UHD needs no place in the key.** A heuristic can be replaced independently of the
  kernels it ranks ([Section 8.2](#82-model-updates)), which raises the obvious worry that a cached
  ranking outlives the model that produced it. In process, it cannot: the cache is a sibling of the
  heuristic on the engine, both fixed for that engine's lifetime, so **a swap means new descriptors,
  a new engine, and a new empty cache.** The UHD's identity is implicit in the cache's *location*,
  exactly as the engine id is. Should hipDNN ever re-scan descriptors in-process without rebuilding
  engines, the **inventory generation counter** already in the key covers that case too — a new UHD
  arrives as a new inventory. Nothing UHD-specific is needed.

- **Persistent: the UHD identity has to be in the path, because nothing else survives a restart.** The
  moment rankings outlive the process, the argument above evaporates — generation counters are
  process-local, and a restart happily reads entries written by a model that has since been replaced.
  The natural layout is a **directory per heuristic build**, with the same `(graph, device)` entry key
  inside it:

  ```
  <cache root>/<uhd id>/<uhd content hash>/<graph id>-<device>.rank
  ```

  Three details decide whether this actually works:
  - **Hash the content, don't trust the id or a version field.** A regenerated model normally keeps the
    same UHD id — it is the same logical heuristic for the engine — and a hand-maintained version can be
    forgotten. Hash the UHD document *and* the model artifact together. `features_hash` is not a
    substitute: it fingerprints the input contract, not the weights, so two models with identical
    features and different training both hash the same.
  - **Do not key the device on the HIP ordinal.** Device 0 is a different GPU on a different machine, and
    can be a different GPU after a reboot. Persisted entries need a stable device identity — arch name
    plus the `$device.*` facts the model actually consumed.
  - **Store kernel ids, not kernel definitions.** The ranked order is a list of UKD ids, re-resolved
    against the loaded descriptors on read. Persisting whole definitions duplicates descriptor state that
    can drift out from under the cache.

  The result is that invalidation becomes deleting a directory: a new model writes a new hash directory,
  the old one is unreachable and can be reaped by age, and no entry-by-entry staleness check is ever
  needed.

  **Worth measuring before building.** Ranking is a few hundred comparisons per candidate; the genuinely
  expensive work on this path is kernel compilation and matcher evaluation. A persistent cache for the
  *ranking* alone may not repay its complexity — the case for it is much stronger if it rides along with
  a persistent applicability or compiled-kernel cache. See [Open Question 9](#operational).

- **Two limits on caching.** [Section 9](#9-performance) states that selection cost is paid at most
  once per distinct need. Two conditions bound that guarantee:
  - **The cache is capacity-bounded** (an LRU in the current foundation), so a working set larger than
    the bound re-ranks on eviction. The bound is a tuning parameter, not a correctness issue, but the
    latency budget should be justified against the *uncached* path.
  - **Some graphs cannot be cached at all.** The key needs a stable graph identity; a graph that is
    unfinalized, legacy, or carries a non-v4 id has none, so **every call re-ranks**. That is the real
    worst case for selection overhead, and it is the one [Section 9.4](#94-latency-target) should be
    measured against.

  **OPEN:** in-process only vs. a persistent cross-run cache — see [Open Question 9](#operational).

### 9.3 Efficient evaluation (expressive spec, fast hot path)

Extensibility lives in the data contract; efficiency lives in a compiled core. The seam is the adapter,
and the extensibility cost is paid once per candidate (one indirect call), not per feature:

**Scoring is inherently per-candidate**, because `$kernel.*` features differ per candidate — that is what
makes argmax meaningful in the first place. So the goal is not to avoid the per-candidate loop but to
make each iteration as small as possible:

- **Lower the `features_signature` at load, never walk the expression tree per candidate.** The UMD PoC
  already compiles a rule to an `Expression<DataT>` once
  ([Section 6](#6-feature-extraction));
  the feature extractor reuses that so per-candidate scoring is a tight loop over a compiled expression
  and a flat tree table — no strings, no JSON, no map lookups.
- **Split the row into a shared prefix + per-candidate suffix.** Problem and device features
  (`$q.*`, `$device.*`) are identical across every candidate in the engine; only `$kernel.*` and the
  computed subexpressions that depend on it vary. Compute the invariant prefix **once per graph** and refill
  only the varying slots per candidate, turning O(N × full-featurize) into O(full-featurize + N × small)
  for N candidates. The kernel-dependent tail is the part that cannot be hoisted or cached across
  candidates, and keeping it small is the main lever on selection cost.
- **Reuse the engine's bound symbols; do not re-extract.** Matching the engine's pattern binds the
  problem tokens once per graph, and selection reads that table rather than re-featurizing
  ([Section 6.1](#61-feature-sources)). The scorer receives the bound symbol table alongside the
  candidate, so a heuristic never walks the raw graph — a second extraction path is what the single
  binding exists to prevent. Because the pattern is engine-wide, the binding is shared by every pack
  joining the engine and is computed once, not once per matcher.
- **Single-candidate short-circuit.** If only one UKD survives matching, skip the model and return it —
  common, and it makes the load-on-ranking case above nearly free.

### 9.4 Latency Target

The `native` adapter is the performance floor (a direct call, near-zero inference cost) and — because it
lands first ([Section 7.1](#71-first-native)) — it doubles as the measured baseline every later adapter
is compared against. The `tree_data` walker is expected to be close: a flat tree table over a
preallocated feature row is a few hundred comparisons per candidate. Target: overhead within 2× of the
`native` baseline.

**This needs measuring, not asserting.** Selection sits on the plan-build path and runs once per
candidate, so CPU overhead is the thing most likely to make a data-driven heuristic unshippable. The
components should be **wall-clocked separately** — descriptor load and model parse, feature extraction
(shared prefix vs. per-candidate tail), and scoring — so a regression is attributable and so the cost of
different adapters can be compared directly when choosing between heuristic options. The exact budget is
not fixed here; the requirement is that the numbers exist and are tracked, per
[RFC 0017 §12.1](0017_UniversalKernelDescriptor.md#121-testing-and-performance).

---

## 10. Applicability Flow

The two selection levels ([Section 2](#2-scope)) are not cleanly sequential; the part that interleaves is
the **matcher**, not the UHD. This distinction bounds when a model may be loaded.

**Applicability comes from the matcher, and it bubbles up.** Engine selection cannot rank engines it has
not ruled out. For a descriptor engine, "do I apply?" is the **matcher (UMD) pass** at the descriptor
layer, which uses no heuristic; that result bubbles up so non-viable engines are ruled out before the
first plugin-policy layer ranks the survivors.

**The UHD runs strictly after that, and only when asked.** Ranking is not part of deciding applicability
and is never performed speculatively. It happens when a caller needs something only the ranked catalog
can answer — an estimated throughput for a policy, a knob default, or the selected kernel:

```
Graph
  └─► Matcher pass (per engine)          ← no UHD involved
        └─► Applicable engines
              └─► Policy ranking          ← may request estimates ─┐
                    └─► Winner                                     │ on demand only
                          └─► Kernel selection ────────────────────┴─► UHD ranks the catalog
```

So there is no flow in which ranking happens before applicability is confirmed, and none in which it
happens without a request. Whether a UHD is loaded at all depends on the policy in play: one that ranks
by explicit user selection or fixed criteria never asks, and never triggers a load. The load-timing
rules, the per-engine model cache, and the shared `(engine id, graph id, device id)` +
generation-counter result key are specified once in [Section 9.2](#92-loading-and-caching) and are not
restated here.

**Where estimates fit.** Once the policy has its applicable set, it *may* ask each engine for a
predicted performance and order them by merit rather than by a static list. Today rocKE's `isApplicable`
is a yes/no gate; what changes is that a separate, subsequent query can return a score. The gate stays
free; the score costs a model evaluation and is only paid when a policy wants it.

> **One anticipated exception.** The future **`predict_applicable_kernels`** role — the JIT case where
> there is no enumerable catalog until something produces one
> ([Section 4.3](#43-future-predict_applicable_kernels-when-there-is-no-catalog-to-rank)) — would
> necessarily run *during* the applicability query, since "what can this engine even build for this
> graph?" is the question being answered. That inverts the rule above. It is noted here so the
> ordering is understood as a property of the v1 AOT design rather than an invariant of the descriptor
> model.

---

## 11. Engine Selection Integration

This section describes how kernel-level heuristics feed engine-level selection. The policies themselves
are [RFC 0007](0007_EngineSelectionHeuristicsFramework.md)'s territory; this RFC supplies the heuristics
they consult.

### 11.1 The engine estimate and the kernel-catalog ranker

Two of the engine's three UHD roles ([Section 3.1](#31-descriptor-relationships)) participate in
engine selection, both predicting absolute performance so they are comparable across engines:

| UED field | Signature | Cost | Role |
|-------|-----------|------|------|
| **`predict_engine_tflops`** | `f(graph) → expected perf` | Cheap (no catalog enumeration) | Quick-policy engine ranking |
| **`sort_kernel_catalog`** | `f(graph) → best kernel + perf` | Full per-candidate | Kernel selection + accurate cross-engine score |

`predict_engine_tflops` is the coarse proxy; `sort_kernel_catalog` both selects the kernel and yields the
better figure of merit. (The third role, `predict_applicable_kernels`, precedes both when present — it
produces the candidate set the ranker then sorts — but is not itself a performance estimate.)

**`predict_engine_tflops` is not needed for v1.** With a single descriptor engine there is nothing to
rank *against*, so the cheap estimate buys nothing — `sort_kernel_catalog`'s top score answers "how fast
would this engine be?" adequately. It becomes necessary when there are **competing engines** to order,
and for **opaque engines** that cannot expose a per-candidate catalog to rank at all. Until it exists,
**the full ranking is the stopgap**: an engine reports `sort_kernel_catalog`'s best predicted score as
its estimate, accepting the enumeration cost.

**Distinct model vs. derived from the ranker.** If `predict_engine_tflops` is a distinct trained model,
the quick policy can rank engines without enumerating candidates — cheaper at selection time. If it is
derived from `sort_kernel_catalog` (max predicted score over candidates), there's one fewer model to
train and maintain, but the quick policy must evaluate the full ranker to get the estimate. The tradeoff
is selection-time cost vs. training/maintenance complexity. **OPEN:** See [Open Question 6](#structural).

A distinct estimate model is what gives the quick policy its value: only the winning engine runs the full
`sort_kernel_catalog`, and losers never enumerate candidates. If the estimate is derived from the ranker,
every engine runs the ranker to produce it, collapsing the quick and thorough policies into one operation.
The "cheap (no catalog enumeration)" property depends on `predict_engine_tflops` being trained on
`f(graph)` alone, not `f(graph, candidates)`. See [Open Question 6](#structural).

### 11.2 Two Engine-Selection Policies (RFC 0007)

Shorthand in this section: **A** = `predict_engine_tflops`, **B** = `sort_kernel_catalog`
([Section 11.1](#111-the-engine-estimate-and-the-kernel-catalog-ranker)).

- **Quick policy.** Rank applicable engines by A (expected performance); pick the winner; if the winner
  has a config UHD (B), run it to pick the kernel. Only the winner drills down, so losers are never
  scored at the kernel level. Engines with no descriptor layer (e.g. MIOpen) contribute their high-level
  estimate for the ranking and, if they win, use their own internal kernel selection.
  **OPEN:** See [Open Question 7](#structural) (non-descriptor engine estimates).
- **Thorough policy.** Run B for every applicable engine that has it (best config + its predicted perf),
  fall back to A for engines that don't, then compare the predicted performance across engines and pick
  the global best (engine + config). More work, more accurate.

```
Quick:     applicable → rank by A → winner → (B? kernel : own selection) → dispatch
Thorough:  applicable → run B (or A) for each → compare perf → best (engine, config) → dispatch
```

**Both estimates are exposed through the plugin API, and both are optional.** The query surface must
carry A and B separately — they have different costs, and the quick policy's entire value is being able
to ask for A *without* paying for B. Neither is guaranteed to exist: an engine may have B but not A
(the v1 case), A but not B (an opaque engine like MIOpen reporting a coarse estimate), or neither. The
policy therefore has to treat a missing estimate as a normal outcome rather than an error:

| Engine has | Quick policy | Thorough policy |
|---|---|---|
| A and B | Rank by A; winner runs B | Run B; compare its top score |
| B only | Rank by B's top score (pays enumeration) | Run B; compare its top score |
| A only (opaque) | Rank by A; engine does its own kernel selection | Compare A against others' B |
| Neither | Falls back to static ordering; contributes no score | Same |

An engine that supplies no estimate is ordered by the existing static rules and simply does not
participate in performance-based comparison — the mixed case has to work, since it is the near-term
reality for every non-descriptor engine.

### 11.3 Cross-Engine Comparison

**OPEN:** See [Open Question 1](#schema-and-training) (regressor vs. ranker).

Cross-engine arbitration is hipDNN's and the user's. RFC 0017 does not constrain a UHD's scores to its
own engine, which leaves room for a policy that compares engines on UHD-estimated throughput. RFC 0017
fixes that arbitration is exercised through three mechanisms: **explicit selection** of an engine,
**policy configuration** (a resolved sequence of heuristic-policy plugins supplying the ranked engine
list, where a policy may itself be a heuristic), and **auto-tuning**, which measures engines and picks the
winner outright ([RFC 0013](0013_Autotune.md)). The two policies of
[Section 11.2](#112-two-engine-selection-policies-rfc-0007) are policy configuration — new policies in
that existing slot, not a new arbitration surface. Auto-tuning remains the ground truth these heuristics
approximate, and the substrate that trains them ([Section 13](#13-model-generation-pipeline)).

The mechanism is an absolute, cross-comparable figure of merit. Candidates are scored cardinally, on an
absolute metric (calibrated TFLOPS) rather than a within-group rank. When a UHD is a calibrated TFLOPS
regressor, its best-candidate score is a predicted figure of merit for what the engine would run,
expressed on a scale that means the same thing across engines. This lets each engine run its own heuristic
per package, independently, while the results remain comparable across engines: hipDNN compares engines by
predicted performance rather than a fixed order (for example, rocKE predicting 310 TFLOPS for its best
FMHA kernel against MIOpen's 240 selects rocKE). Local, per-package scoring composes into a global
comparison without a central ranker, and the UHD is where the number exists.

Absolute scoring is a harder problem than ranking. A per-package model need only be monotonic to pick
correctly among its own candidates; making it calibrated — accurate in absolute TFLOPS, so cross-engine
comparison is sound — is strictly harder. A miscalibrated absolute score is worse than a correct rank,
because it produces confident but wrong cross-engine picks.

**Fallback.** If the absolute method underperforms, engine selection degrades to rank-ordering at the
engine-policy level: it reverts to existing [RFC 0007](0007_EngineSelectionHeuristicsFramework.md)
static/rank ordering, and each UHD continues ranking within its engine without claiming a comparable
absolute score. Rank-ordering is the defined backstop; the design does not depend on calibration
succeeding.

**What this RFC commits to:** the UHD schema declares `score` (`units`/`calibrated`/`transform`,
[Section 4](#4-uhd-schema)) so a consumer can invert the training transform and recover real TFLOPS, and
supports a score-only evaluation mode (rank and return the best score without selecting for launch).

**What is deferred:** Delivering cross-engine comparison requires changes outside the UHD:

1. **A plugin-query surface** for the per-graph figure of merit — both the cheap **A** (engine
   performance estimate) and the accurate **B** (config UHD run in a "score only, don't launch" mode).
   This is an engine-plugin ABI addition, owned by the plugin SDK, not this RFC; it must also let a
   non-descriptor engine (MIOpen) report an A-level estimate through the same surface, or the policy
   falls back to today's static ordering for it.
2. **The two engine-selection policies** that consume it ([Section 11.2](#112-two-engine-selection-policies-rfc-0007)) —
   the quick policy (rank by A, drill into the winner's B) and the thorough policy (run every B, compare
   across engines). Both are squarely [RFC 0007](0007_EngineSelectionHeuristicsFramework.md)'s territory.
3. **Cross-engine calibration.** Comparing estimates across engines only works if the units are
   comparable and each model (A and B) is calibrated to real TFLOPS (not just monotonic for argmax).
   This is a real modeling requirement, not just plumbing — and the one most likely to force the
   rank-ordering fallback if it does not hold up.

This is a dedicated follow-up co-owned with [RFC 0007](0007_EngineSelectionHeuristicsFramework.md).

---

## 12. Observability

Because selection is data-driven, it must be inspectable — consistent with
[RFC 0017 §9](0017_UniversalKernelDescriptor.md#9-observability-and-diagnostics) and
[RFC 0007 §12](0007_EngineSelectionHeuristicsFramework.md#12-logging). The UHD path surfaces:

- **Selection trace:** Candidates, scores, the ranked order, winner, and whether the model or a fallback
  decided
- **Model provenance:** UHD id, adapter, model artifact version, `features_hash`,
  `trained_against` (UED/UMD/KMD versions), training provenance
- **Contract diagnostics:** A clear **error** (not a warning) naming which of the three checks failed and
  why, plus the fact that ranking degraded to `static_order` and the estimate was reported as 0
- **Coverage warnings:** When a scored candidate or device falls outside what the model was trained on
  ([Section 8.3](#83-out-of-distribution-inputs))

Logging alone is not sufficient. Because a broken contract degrades rather than fails
([Section 6.3](#63-contract-enforcement)), nothing at runtime stops a mis-built pack from shipping and
ranking by priority order. Descriptor-set validation therefore runs in **CI over shipped packs** —
checking hash agreement, KMD pairing, signature resolvability, and knob-tuple uniqueness — so the error
surfaces at build time. See [Open Question 14](#operational).

---

## 13. Model Generation Pipeline

The UHD is only useful if producing one is automated — by tooling any package author can run, not a
provider-specific service.

### 13.1 Two-Stage Workflow

1. **Ship a working pack with no heuristic.** The UED names no UHD. The catalog comes back in
   deterministic `priority`/`id` order with a warning that the engine has no heuristic
   ([Section 5](#5-selection-flow) step 6). Nothing to author, no benchmarking, no training — the pack is
   fully functional and model-free from day one. (An author who prefers the intent recorded rather than
   inferred can name an explicit `static_order` UHD; it is not required.)
2. **Generate a real heuristic from on-hardware timings.** A standalone generation tool loads the pack
   **through a UED exposing every KMD field** ([Section 13.2](#132-benchmarking-via-hipdnn-autotune)),
   times its kernels across a corpus of problem shapes, trains a model, and emits an updated UED/UHD —
   now `adapter: tree_data` pointing at an exported model, with the categorical encoding and
   `trained_against` versions it was built from. Dropping that updated engine descriptor set back in
   upgrades the pack in place. The emitted UED's `knobs` are **derived from the trained feature set** —
   the axes that survived feature selection ([Section 3.2](#32-kmd-fields-and-knobs-as-the-heuristics-feature-axes)),
   which is normally fewer than the generation UED exposed.
   **OPEN:** See [Open Question 10](#operational) (shape corpus location).

Because the shipped and generated heuristics are the same descriptor kind differing only in `adapter`
and fields, the tool only rewrites data; it never introduces a new interface. The tool runs over
hipDNN's public API — it adds no code to hipDNN and touches no provider internals — so it works for
any provider's pack.

### 13.2 Benchmarking via hipDNN Autotune

The timing substrate is hipDNN's own autotune ([RFC 0013](0013_Autotune.md)), not a bespoke sweep.
Autotune is provider-agnostic: it times whatever engine/kernel actually runs, so it exercises a rocKE
pack exactly as it would any other engine. **The UHD itself never runs kernels** — it is scored data.
The generation tool is a separate program that wraps hipDNN, drives the timing, logs the results, and
trains the model; it reaches the engine only through the public Graph API:

- `get_engine_configs()` — enumerate applicable engines and their knobs for a graph
- `add_engine_variants()` — enroll explicit `(engineId, knobSettings)` tuples as plan specs
- `add_engine_sweep()` — enroll the Cartesian product of knob axes plus fixed settings
- `autotune(mode = EXHAUSTIVE, strategy = RUN_UNTIL_STABLE)` — compile and time each enrolled spec,
  iterating until the trailing-window coefficient of variation converges
- `AutotuneResult[]` — per-spec `engineId`, `knobSettings`, `minTimeMs` / `avgTimeMs` / `stddevMs`,
  `iterationsRun`, `workspaceSize`, persisted to JSON. That JSON, joined with the feature row, is the
  training dataset.

The tool times the **shipped** kernels — it does not re-build a variant grid. The pack is the authority
on which variants exist; autotune is the authority on how fast each one runs.

**Generate against a fully-exposed UED; emit a UED exposing what the model kept.** These APIs address
kernels *only* through exposed knob settings, so a kernel the tool cannot name is a kernel it cannot
time, and therefore cannot train on. Hence the generation UED exposes **every** KMD field, making the
knob tuple equal to the metadata tuple and every catalog entry individually addressable. Feature
selection then prunes the axes that do not earn their place, and the emitted UED's `knobs` are exactly
the survivors — because knobs and the model's feature set are the same thing
([Section 3.2](#32-kmd-fields-and-knobs-as-the-heuristics-feature-axes)). Benchmark wide, expose what the model
keeps.

Full exposure makes each entry *addressable*. **It does not make the catalog efficiently enumerable**,
and the current API has no good answer for that — indeed full exposure makes it harder, since it
maximizes the number of knob axes:

- Knob ranges are reported **per knob, independently**, so a sweep can only be expressed as a Cartesian
  product. For a sparse catalog most of that product names no kernel — the tool enrolls combinations
  that cannot be satisfied and discovers this only by trying them.
- Cost scales with the product, not with the catalog. Many knobs over a handful of real kernels makes
  enumeration slow at best; for a large enough space with few valid points, **enumeration by sweeping is
  not viable at all**. (`add_engine_sweep()` caps the product at 10,000 combinations, which bounds the
  damage but does not solve the problem — it just turns an intractable sweep into a rejected one.)
- Nothing expresses *which combinations are valid*, so neither the tool nor a user can ask for "the
  catalog" directly.

This requires an API improvement, tracked outside this RFC: a way to enumerate the applicable catalog, or
the valid knob-tuple set, directly rather than by probing a cross-product. Until it exists, the generation
pipeline is practical for engines with small or dense knob spaces and impractical for large sparse ones.
This is a hipDNN API gap rather than a defect in the UHD design, recorded as
[Open Question 12](#operational); the pipeline of this section depends on it for the harder cases.

### 13.3 One source of truth, translated once

The tool guarantees that the runtime contract matches what was benchmarked. It emits the pack's
descriptors (updated UED/UHD, and the KMD/`features_signature` if not already present) from the same run
that produced the timings — generate-then-freeze — so descriptor, dataset, and runtime are consistent by
construction.

The tool freezes and emits two contracts:

1. **Feature contract.** Emit the UHD's `features_signature` ([Section 6.2](#62-the-features_signature))
   from the same feature definition training used (expressions over `$q.* / $kernel.* / $device.*`).
   Then **one generic extractor runs it on both sides** — offline for training, the in-tree evaluator
   for inference — both reading the *same* signature. No reimplementation, so no drift. (Caveat: the
   expression op set must cover the derived features — [Section 6.2](#62-the-features_signature)'s
   `log2`/`/`/`min`/`max` extension. A computation outside that set uses the `custom_library` featurizer
   escape hatch.)
2. **Kernel-identity contract.** The candidate autotune timed must map 1:1 to the **UKD** it stands for
   in the emitted pack, so the model's argmax over timed candidates maps exactly to argmax over UKDs at
   runtime. The join key is the knob tuple reported on each result; because generation runs against a
   fully-exposed UED, that tuple is the kernel's metadata tuple, which is unique engine-wide and validated
   as such at load. This drift risk carries two obligations:
   - **Verify that `knobSettings` round-trips.** The tool must confirm the settings reported back on a
     result select the same kernel they enrolled, rather than assuming it.
   - **A collision during generation is a dataset error, and means exposure is incomplete.** If two
     enrolled candidates share a tuple, timings cannot be attributed to a kernel. Under full exposure
     this cannot happen, so it indicates the generation UED did not expose every KMD field — fail loudly
     rather than train on an ambiguous join. (At *runtime*, once the emitted UED exposes only the model's
     feature axes, two kernels differing solely in dispatch-only fields will share a knob tuple; they
     score identically and the tie-break separates them, with a load-time warning —
     [Section 3.2](#32-kmd-fields-and-knobs-as-the-heuristics-feature-axes).)

### 13.4 New stage: package (Stage P)

From one timing run ([§13.2](#132-benchmarking-via-hipdnn-autotune)) the tool trains the catalog ranker
(`sort_kernel_catalog`) and, when needed, the engine estimate (`predict_engine_tflops`)
([Section 11.1](#111-the-engine-estimate-and-the-kernel-catalog-ranker)). A package stage then emits (or
updates) the engine's descriptor set:

- **`sort_kernel_catalog`** — rewritten from the shipped `static_order` (or absent) to
  `adapter: tree_data`, carrying the `features_signature` (referencing `$kernel.*` KMD fields and
  `$device.*` for arch-awareness), `features_hash`, `objective`/`score`, and the adapter-scoped
  `tree_data` body ([Section 4](#4-uhd-schema)); per engine and arch-keyed, so no artifact table;
- **`predict_engine_tflops`** — the coarse `f(graph) → expected perf` model the quick policy ranks engines
  by ([Section 11.2](#112-two-engine-selection-policies-rfc-0007)), emitted as data on the UED when the
  engine needs it (whether it is a distinct model or derived from the ranker is the OPEN in
  [Section 11.1](#111-the-engine-estimate-and-the-kernel-catalog-ranker));
- **the model files** — the trained boosters exported to the `tree_data` format read by the in-tree
  walker ([Section 7](#7-model-adapters)), each embedding the `features_hash` it was trained against,
  shipped with the engine descriptors rather than compiled in;
- **the KMD** — the compilation-knob schema (`fields`: `tile_m`, `warp_n`, `split_k`, `dtype`, …), if the
  pack does not already carry one; its fields are the `$kernel.*` metadata the UKDs fill
  ([Section 3.2](#32-kmd-fields-and-knobs-as-the-heuristics-feature-axes)) — one KMD per engine, owned by
  the UED;
- **the UED** — updated to reference the new UHDs, with `knobs` set to the ranker's `$kernel.*` feature
  axes ([Section 3.2](#32-kmd-fields-and-knobs-as-the-heuristics-feature-axes)).

The UMDs, UDD, and child UKDs are not regenerated; only the heuristic side changes. This is the intent of
the two-stage design: the expensive artifacts (compiled kernels) ship once, and the heuristics are
layered on afterward as data.

### 13.5 Sweep space: grid vs. constraint

The generation tool sweeps two things: the **problem-shape corpus** (batch, seqlen, heads, and the like —
supplied by the author as representative shapes, or a per-op default) and optionally the **exposed knobs**
(the KMD fields the engine lets a user set, via `add_engine_variants` knob settings). A knob setting only
*filters* the catalog ([Section 3.2](#32-kmd-fields-and-knobs-as-the-heuristics-feature-axes)), so
sweeping knobs explores user-visible restrictions, not new kernels. The variant space is fixed: it is the
pack's existing child UKDs, so the tool does not enumerate or build variants but enrolls the shipped ones
and times them ([Section 13.2](#132-benchmarking-via-hipdnn-autotune)). That holds within this pipeline;
deciding which variants exist, and pruning them once a heuristic exists, is the upstream
package-creation-time stage of
[Section 14](#14-package-creation-time-selection-knobs-and-aot-kernels).

One subtlety for anything that *drives* a sweep from a descriptor: a validity *constraint*
(`min:1, max:8`) expresses which values are **legal**, not which to **sample**. A `[min, max, step]`
triple ([Section 3.2](#32-kmd-fields-and-knobs-as-the-heuristics-feature-axes)) makes a regular axis enumerable
without guessing the increment, which covers most numeric knobs; an irregular axis still needs an
explicit `sweep_values` list. Neither addresses *cross-knob* validity — that is the enumeration gap of
[Section 13.2](#132-benchmarking-via-hipdnn-autotune). **OPEN**: standardize where the shape corpus and
any knob grid live (a tool-side config vs. a descriptor field), so a heuristic can be regenerated
reproducibly without out-of-band inputs.

### 13.6 Auto-deriving a first-pass `features_signature`

Most of a `features_signature` can be **derived from what a package already carries**, so the tool can
propose a first pass rather than requiring an author to hand-write the feature list. The key is that
"map the graph to features" spans **two layers**, and only the first is shared/derivable:

- **Layer 1 — op-intrinsic vocabulary (per op, shared).** The fields that *exist and are bindable*:
  tensor/dim bindings (`$q.dims[2]`), node attributes (`$q.dims[3]`), device properties
  (`$device.*`), and **precomputed fields the binding system supplies automatically**. These are facts
  about the op, identical for every engine that implements it
  ([Section 6](#6-feature-extraction)).
- **Layer 2 — the `features_signature` (per package/UHD, not shared).** *Which* of those fields a given
  model consumes, plus derived transforms. This is per-UHD: two packages of the same op may rank on
  different subsets.

The tool auto-derives Layer 1 and proposes a Layer-2 first pass from it, in three tiers:

| Tier | Feature kind | Derivable from | Author input needed |
|---|---|---|---|
| 1 | Raw fields (`$kernel.*` = KMD fields; tensor dims/attrs = the engine's published symbols; `$device.*`) | KMD schema + the UED's published symbol set + device vocab | **none** |
| 2 | Generic transforms (logs, ratios) and **tile/wave quantization** | Tier 1 + an expression pairing a problem dim with a `$kernel.*` tile axis | the **dim↔tile correspondence** — which dim goes with which tile field |
| 3 | **Physics** — arithmetic intensity, roofline bound | the op's FLOP and byte counts, divided inline | **none** — supplied as precomputed op fields (below) |

The tile/wave quantization (Tier 2) is not auto-inferable — the tool cannot guess that `seqlen_q` pairs
with `tile_m0`. It is the one genuine author input, and it is small: a list of (problem dim, tile field)
pairs. From that list the tool writes the quantization entries into the `features_signature` itself
([Section 6.4](#64-computed-features)); the author supplies correspondences, not expressions.

**Arithmetic intensity, and where the FLOP/byte counts come from.** Intensity is
`total_FLOPs / total_bytes_moved` (FLOP/byte) — the roofline x-axis that separates compute-bound from
memory-bound problems, which is exactly the split that decides which kernel wins. Both terms are
closed-form over the bound dims and dtype sizes, but they are **op-specific** and cannot be inferred
from the KMD field list, so something has to supply them per op.

**They are precomputed fields, declared in the hipDNN schema.** The binding layer already publishes
derived values that no descriptor declares: `$q.stride_order` and `$q.packed` stand in for
contiguous-stride arithmetic, and `$q.value_f32` coerces a tensor's compile-time value to a single typed
token. Each is declared in the schema like any other field and versioned with it, so adding one is an
additive schema change rather than a per-pack extension point
([RFC 0020 §6](0020_UniversalEngineDescriptor.md#6-symbol-binding-what-the-pattern-publishes)). FLOP and
byte counts fit that mechanism, as per-op precomputed fields:

```jsonc
// available wherever an sdpa_fwd node is bound — no descriptor declares these
"$sdpa_fwd.flops"        // 4·B·H·Sq·Sk·D for SDPA forward
"$sdpa_fwd.bytes"        // sum over Q/K/V/O of element_count × that tensor's dtype size

// so a features_signature entry just divides them
"arithmetic_intensity": {"/": ["$sdpa_fwd.flops", "$sdpa_fwd.bytes"]}
```

These are **not** authored as expression strings in table-level `.fbs` annotations. Embedding an
expression language inside the schema file is awkward, and changing an `.fbs` requires codegen plus a
recompile regardless — so a schema annotation buys none of the data-driven flexibility that would
justify it. What needs a rebuild to change belongs in code, as a precomputed binding field.

Because they are **op-intrinsic** (SDPA does `4·B·H·Sq·Sk·D` FLOPs regardless of engine or package),
these fields live at Layer 1 and are shared by every package of that op — defined once per op-family.
A `features_signature` then references intensity *identically* to a raw dim like `$q.dims[2]`, and the
Tier-3 "physics" distinction disappears at the point of use.

> **Coordinate with the UMD.** This says the op vocabulary should carry a *set* of useful precomputed
> fields — some universal across ops, some per-op — of which FLOPs and bytes are two. Where that set is
> defined and how per-op entries are registered is the UMD's to specify, not this RFC's; the requirement
> here is only that FLOP and byte counts be among them. See [Open Question 4](#schema-and-training).

Caveats:

- **Mixed-dtype ops** (e.g. fp8 in / fp16 accumulate, or differing I/O dtypes) make the byte count a sum
  over *per-tensor* dtype sizes, not one global `dtype_bytes`. Each tensor must contribute its own dtype
  — a single-dtype shortcut is wrong for quantized kernels.
- **Not every op has a clean closed form.** Ragged or data-dependent shapes (variable-length sequences,
  data-dependent masking) may make an exact count impossible; the field should then expose a documented
  upper bound or be absent rather than silently wrong, and a UHD that needs better can compute its own
  inline in the signature.
- **Auto-derivation yields a *superset*.** Deriving every raw field and generic transform produces a
  bloated, noisy vector that can hurt a small-data model; the sweep's feature-importance (or a curated
  per-op template) prunes it. Auto-derivation proposes; data or a template trims.

**A data-free structural first pass.** Tiers 1–2 alone (no benchmarking) already support a real
model-free heuristic better than `static_order`: prefer the kernel whose tile best divides the problem
(minimize quantization waste), tie-break on an occupancy proxy. That is a legitimate `table`/rule UHD
computable from the KMD, the engine's bound symbols, and device facts, and it is the zero-benchmark
starting point before
the autotune-trained `tree_data` model replaces it.

---

## 14. Package-Creation-Time Selection: Knobs and AOT Kernels

Everything above concerns *runtime* selection (which kernel wins for a graph) and *model* generation
(training a UHD for a fixed pack). This section covers the stage upstream of both: deciding **which knobs
a package exposes** and **which kernels it compiles AOT** — that is, generating the UED and the UKD set
itself, at **package creation time**.

It is a distinct heuristics area, and it closes a loop with
[Section 13](#13-model-generation-pipeline): a generated heuristic is one of the inputs used to prune
knobs, and the pruned knob set then changes which kernels are worth compiling.

> **Requirements, not a design — and likely a separate RFC.** This section exists to record what the
> UHD must *support* for package creation to work: that the same autotune substrate is reusable when the
> variant space itself is what varies, that a trained heuristic's feature importances are a usable
> pruning signal, and that a regenerated knob set feeds back into both the KMD and the feature space.
> The algorithms — prioritization objective, search-space refinement, pruning thresholds — are
> deliberately left open, and the package-creation pipeline should be specified in its own document
> rather than expanded here.

### 14.1 AOT selection depends on whether the engine can JIT

The goal of AOT differs sharply between two regimes.

**Engine can JIT in hipDNN — AOT is a cost optimization.** AOT exists to cut compile cost for the
kernels that actually get used. Coverage is *not* the objective, because the JIT path already covers the
space; AOT only needs the most-used, most-performant subset. All knobs are available, and compiling a
subset trades package size and build time against runtime compile latency.
**Goal: select the best kernels for a representative set of data.**

**Engine cannot JIT in hipDNN — AOT is the entire functional surface.** Here the compiled kernels are
everything the engine can do, so the set needs *both* good coverage and good performance. This is the
case when JIT has not been built (rocKE today), is never planned, or the kernel generator is not
JIT-able (CK). In this regime hipDNN **cannot inform** AOT selection — it can only **filter** an
already-generated set to reduce package size and runtime load. That filtering can feed back into knob
generation, but it **cannot reduce the up-front cost of variant explosion**, which happens in the
engine's own generator before hipDNN sees anything.
**Goal: filter the worst kernels to reduce size, without losing coverage.**

| | Engine can JIT | Engine cannot JIT |
|---|---|---|
| Role of AOT | compile-cost optimization | the entire functional surface |
| Coverage requirement | low — JIT is the fallback | high — nothing else fills the gap |
| Objective | **select the best** for the data | **filter the worst** to reduce size |
| hipDNN's leverage | can inform *what to build* | can only filter *what was already built* |

### 14.2 Pipeline: AOT selection

1. **Generate an unoptimized KDP** that exposes all possible knobs (for JIT or AOT), through tooling
   plus user generation.
2. **Benchmark that KDP through hipDNN** using the exposed knobs, over an *algorithmically refined*
   search space rather than the full cross-product — and, where relevant, over **targeted datasets for
   specific clients**.
3. **Post-hoc algorithmic evaluation** to prioritize the candidate list: frequency of kernel use, the
   time cost of a poor selection, and similar signals.
4. **Modify the KDP** to include the AOT kernels selected (or newly created) by that evaluation, on
   coverage / frequency / performance.

Step 2 uses the same autotune substrate as [Section 13.2](#132-benchmarking-via-hipdnn-autotune); what
differs is *what varies* (the variant space itself, not just the shape corpus) and *what the output
drives* (the pack's kernel set, not a model).

### 14.3 Knob selection: static vs. empirical

**Static analysis.** Largely package/project dependent, driven by the kernel author — increasingly with
LLM assistance — and limited tooling. This is likely **not** an area the heuristics team owns directly.
It answers two questions from the code: which knobs indicate that a **separate engine** is warranted
rather than one super-engine, and which knobs are **dead or low-impact** for performance.

**Empirical.** Requires a generation-and-execution loop. That loop can live **inside hipDNN** when the
engine supports both JIT and AOT; otherwise it must live **in the engine's own codebase**, wherever the
generation/execution loop can actually run.

### 14.4 Pipeline: knob reduction (hipDNN JIT case)

1. **Generate an unoptimized KDP** exposing all possible knobs for JIT.
2. **Benchmark through hipDNN** over an algorithmically refined search space.
3. **Generate heuristics** from the results ([Section 13](#13-model-generation-pipeline)).
4. **Backwards-evaluate the heuristics** to find the weakest knobs — the axes the trained model barely
   uses.
5. **Regenerate the UED** with the reduced knob set.
6. **Regenerate the AOT kernels** from the reduced knobs, or run the AOT-selection pipeline
   ([Section 14.2](#142-pipeline-aot-selection)).

Step 4 is the same signal as the feature-importance pruning noted in
[Section 13.6](#136-auto-deriving-a-first-pass-features_signature) — there it trims a bloated
`features_signature`; here it trims the **knob space itself**. A knob the model never splits on is a knob
whose variants are not earning their package size.

**Steps 4–5 collapse into one action.** Because `UED.knobs` *is* the model's feature set
([Section 3.2](#32-kmd-fields-and-knobs-as-the-heuristics-feature-axes)), pruning a weak feature and dropping a
knob are the same edit rather than two that have to be kept consistent by hand. Emitting the pruned UHD
emits the reduced UED with it, and the [Section 6.3](#63-contract-enforcement) equality check guarantees
they cannot drift apart.

Step 6 then closes the loop: a smaller knob set means fewer variant axes worth compiling, which shrinks
the AOT explosion, which changes what the next model sees.

**What a reduction does and does not cost.** Worth separating, because only one of these is expensive:

- **Pruning knobs** — the UHD is *regenerated* (the model is refit without the dropped features), but
  from the **timings already collected**. No new benchmarking. This is the cheap direction, and the
  purpose of the loop.
- **Dropping kernels** (step 6 emits a smaller AOT set) — costs nothing at all for the model. A UHD
  scores each candidate independently, so removing candidates cannot change the scores of those that
  remain; the model simply never sees the dropped ones. This is **filtering and ranking commute** doing
  the work.
- **Growing** either the knob set or the kernel set beyond what was benchmarked — this is the expensive
  direction, and it needs a new sweep, because there are no timings for the new axes or variants
  ([Section 8.3](#83-out-of-distribution-inputs)).

So the loop converges cheaply as long as it only ever narrows. **Benchmarking is the expensive step, and
reduction never re-triggers it.**

### 14.5 Relationship to the model-generation pipeline

[Section 13.5](#135-sweep-space-grid-vs-constraint) states that the variant space is *fixed* for model
generation — the tool enrolls the pack's existing UKDs and times them. That holds **within** that
pipeline. This section is the stage where the variant space is actually **decided**, and where a
generated heuristic feeds back to change it. The two run in sequence and iterate:

```
pack (all knobs) → benchmark → heuristic → prune knobs → regenerate pack → benchmark → heuristic → …
```

**OPEN items for this area:**

- The **prioritization objective** for AOT selection (step 3): how to weigh frequency of use against the
  time cost of a poor selection, and how client-specific targeted datasets factor in.
- How the **search-space refinement** algorithm works, and whether it is shared with the shape-corpus
  sweep of [Section 13.5](#135-sweep-space-grid-vs-constraint).
- The **pruning threshold** for a "weak" knob, and whether pruning is automatic or a proposal for author
  review.
- For the **non-JIT** case, whether anything can feed back into the *engine's own generator* to reduce
  variant explosion at the source, rather than only filtering after the fact.

---

## 15. Phased Delivery

Each phase is independently shippable and validated against the SDPA path and the reference tooling,
using the parity and overhead checks of [RFC 0017 §12.1](0017_UniversalKernelDescriptor.md#121-testing-and-performance).

The ordering establishes the ranking seam before the data-driven machinery. Phases 1–3 stand up the UHD
path — schema, wiring, ranking, load and cache, engine integration — using ranking mechanisms that need no
new file format, no parser, and no generic extractor. The data-driven adapters land only once the seam is
in place and measured, so a defect in the seam surfaces against a function rather than against a model
pipeline built on top of it.

| Phase | Deliverable | Notes |
|-------|-------------|-------|
| 1 | No-UHD default + `static_order` | UHD header schema + UED membership. An engine naming **no** UHD returns the catalog in deterministic `priority`/`id` order and warns — the zero-authoring starting state. `static_order` makes the same intent explicit. Proves UED→UHD→catalog wiring end to end. |
| 2 | `native` adapter | Scorer compiled into the engine, named by symbol ([Section 7.1](#71-first-native)). Exercises real ranking, `objective`/`score`, and the ranked-catalog output with no new format. Establishes the performance baseline everything later is measured against. |
| 3 | `tree_data` (escape-hatched) | The tree-table format and the **new in-tree GBDT walker written for this work** — a bounded parser and evaluator with no external dependency — behind a hand-written featurizer rather than the generic extractor. Lands the real FMHA-fwd model. Adds lazy load + per-engine model cache. |
| 4 | `features_signature` + generic extractor | Replaces the hand-written featurizer: inline signature with computed entries, one extractor over the shared namespaces, subexpression hash-consing, `features_hash` over signature + encoding, training↔runtime parity test. |
| 5 | Generation tool | Standalone tool wrapping hipDNN: drives autotune over a shape corpus and the knob space, logs results, trains, emits updated UED/UHD + model. Gated on the catalog-enumeration gap ([Open Question 12](#operational)) for sparse knob spaces. |
| 6 | `table` / CSV | Cheap bucketed heuristics for ops that don't warrant a model. |
| 7 | Engine-selection integration | Score-only mode, the A/B plugin-query surface, engine-selection policies. Introduces the engine estimate (A) — not needed before competing or opaque engines exist ([Section 11.1](#111-the-engine-estimate-and-the-kernel-catalog-ranker)). Co-owned with [RFC 0007](0007_EngineSelectionHeuristicsFramework.md). |
| 8 | `custom_library` | Author-shipped scorer `.so` for models the in-tree walker doesn't cover. Dependency + trust audit gated ([Open Question 11](#operational)). |
| 9 | AOT selection ([Section 14.2](#142-pipeline-aot-selection)) | Benchmark an all-knobs KDP, prioritize by frequency / cost-of-poor-selection, emit the AOT kernel set. Needed first for **non-JIT** engines (rocKE today, CK), where it is filtering rather than selection. Depends on phase 5. |
| 10 | Knob reduction loop ([Section 14.4](#144-pipeline-knob-reduction-hipdnn-jit-case)) | Backwards-evaluate a generated heuristic for weak knobs, regenerate the UED with a reduced knob set, then regenerate AOT kernels. Requires an engine that can **JIT in hipDNN**; depends on phases 5 and 9. |

Phases 3 and 4 split the model format from the generic feature path: shipping `tree_data` behind a
hand-written featurizer puts a trained model in place before the declarative extractor is ready, and makes
the extractor's parity test a comparison against a known-good implementation.

`onnx`, `lgbm_to_c` (build-time, in-tree perf optimization only), and `lightgbm_native` are
dependency-gated and land only when a concrete need appears.

---

## 16. Risks

| Risk | Description | Mitigation |
|------|-------------|------------|
| **Feature-contract drift** | Training and inference feature vectors diverge | Single `features_signature` drives both sides via one generic extractor; three-part load-time check ([Section 6.3](#63-contract-enforcement)); computed features are **inline**, so the signature *is* the computation and `features_hash` cannot miss a redefinition ([§6.4](#64-computed-features)) |
| **Catalog not enumerable** | Knobs are reported independently, so the valid set is a sparse subset of the cross-product; a large knob space cannot be swept, so no training data can be gathered | Generation exposes every KMD field, so every kernel is addressable; **needs an API to enumerate the valid catalog directly** — tracked as [Open Question 12](#operational). Pipeline is viable today only for small/dense knob spaces |
| **Knob-set churn** | A retrain that drops a feature also drops a knob, breaking a caller that was setting it | Knob removal is a **major** UED version bump; exact UED↔UHD knob-set equality checked at load so the two cannot drift; a "sticky" superset held open as an escape hatch ([Open Question 18](#operational)) |
| **Kernel-identity drift** | Timed candidate doesn't match emitted UKD | Generation runs fully exposed, so the join key is the full metadata tuple; verify `knobSettings` round-trips; a collision during generation fails loudly ([§13.3](#133-one-source-of-truth-translated-once)) |
| **KMD↔UHD coupling** | a *breaking* KMD change (removed/reinterpreted field) invalidates the trained model | Explicit semver rule at load (`major ==`, `minor <=`); additive changes need no retrain until exposed ([§8.1](#81-descriptor-versions-and-uhd-coupling)); model disabled (not request failed) on mismatch |
| **Out-of-distribution input** | New arch, or a dropped-in pack whose values the model never saw; the contract still passes, only the values are new | Arch check against training metadata; degrade to `static_order` and log; per-feature coverage metadata kept additive as a later option ([§8.3](#83-out-of-distribution-inputs)) |
| **Dependency creep** | Pressure to link `liblightgbm` at runtime | In-tree `tree_data` default; runtime deps stay opt-in only |
| **Bad/stale model** | Model picks worse than first-match | Degrade to `static_order`; parity gate against the `native` baseline; model provenance in trace |
| **Malformed drop-in pack** | Third-party descriptor set with a broken contract reaches a customer | Never fails the request — model disabled, error logged, estimate reported as 0; CI validation over shipped packs is the primary gate ([Open Question 14](#operational)) |
| **Miscalibrated cross-engine scores** | Absolute score misleads engine selection | Train calibratable TFLOPS from start; fall back to rank-ordering at policy level if calibration unreliable |
| **Selection CPU overhead** | Per-candidate scoring on the plan-build path costs more than it saves | Wall-clock load / extract / score separately against the `native` baseline; shared-prefix split; single-candidate short-circuit ([§9.4](#94-latency-target)) |
| **Cache key incompleteness** | Result cache returns wrong kernel | Fingerprint must include problem + candidate set + device |
| **Drop-in trust** | Model artifact is author-controlled input | Bounded loader/evaluator; inherit [RFC 0017 §10](0017_UniversalKernelDescriptor.md#10-packaging-and-delivery) trust rules |

---

## 17. Open Questions

### Schema and Training

1. **Regressor vs. ranker.** The tooling trains a *regressor* on TFLOPS and argmaxes. A
   learning-to-rank objective (LambdaRank/NDCG) optimizes ordering directly and may pick better
   *within* an engine without needing calibrated absolute values. But a calibrated TFLOPS *regressor*
   is what makes the absolute, cross-comparable metric of [Section 11.3](#113-cross-engine-comparison)
   possible; a pure ranker forecloses that and leaves only the rank-ordering path. Recommendation:
   regressor, preserving the absolute option, with ranking as the fallback rather than the only mode.
   *(Impacts [Section 4](#4-uhd-schema), [Section 11.3](#113-cross-engine-comparison).)*

2. **Arch-aware model scope.** The UHD is one per engine, and the engine spans arches (`arch` is a KDP
   property), so the model is arch-aware via `$device.*` features — one model generalizing across the
   engine's arches. A per-engine model assumes `$device.*` features capture the cross-arch differences
   well enough; where they do not, the arch-keyed heuristic maps ([Section 3.1](#31-descriptor-relationships))
   split the model per arch, and an engine may additionally be scoped more narrowly (a UED per
   arch/dtype). Decide the default split against real cross-arch accuracy data.
   *(Impacts [Section 3](#3-ownership-model).)*

3. **`tree_data` artifact format.** A data-SDK FlatBuffer tree schema (recommended) — consistent with
   graph/device-props serialization, `Verifier`-gated, additive evolution, needs a convert step at
   build. Alternative: parse LightGBM `model.txt` at load — lowest author friction (no conversion) but
   a bespoke parser to write and harden against hostile input. Decide alongside
   [Open Question 16](#operational) — whether coverage metadata rides in the artifact affects the format.
   *(Impacts [Section 7.2](#72-default-tree_data).)*

4. **Derived feature set.** Arithmetic intensity, tile quantization, aspect ratios, occupancy,
   padding-fit — are there others? Candidates: memory-footprint / working-set vs. cache and HBM
   capacity; a compute-vs-memory-bound flag from intensity vs. the device's roofline ridge point;
   wave-quantization *tail* (last-wave occupancy); K-splitting overhead for split-K variants.
   Enumerate the final set against real per-op sweeps before freezing.
   *(The expression-op question is resolved — the UMD's operator set already covers the derived
   features.)* *(Impacts [Section 6.2](#62-the-features_signature).)*
   **Auto-derivation dependency:** the physics features (arithmetic intensity, roofline bound) need
   per-op **FLOP and byte counts as precomputed fields**, declared in the hipDNN schema alongside the
   existing precomputed values such as `$q.stride_order` and `$q.packed`
   ([Section 13.6](#136-auto-deriving-a-first-pass-features_signature)). The mechanism and its home are
   settled; what remains open is which per-op counts to declare, and the mixed-dtype byte convention. Tier-2 quantization (the
   dim↔tile correspondence) is written inline in the signature
   ([Section 6.4](#64-computed-features)), from an author-supplied list of (problem dim, tile field)
   pairs.

### Structural

5. **Independently-authored packs joining one engine.** The catalog is engine-scoped and one UHD ranks
   the union across packs ([Section 5](#5-selection-flow)), so overlapping packs do *not* produce
   incomparable scores — but they do produce a catalog the model may never have seen. A UHD trained on
   pack A's kernels is asked to rank A ∪ B when B is dropped in later, and nothing in the load-time
   contract check catches it: the `features_signature` still resolves, because B's kernels fill the same
   KMD fields. The model silently extrapolates. Options: (a) adding a pack to an engine requires
   republishing that engine's UHD, retrained over the union — safe, but couples pack authors to the
   engine owner; (b) accept extrapolation, on the theory that a model over `$kernel.*` metadata
   generalizes to unseen kernels with in-distribution metadata, and add a training-coverage warning to
   the trace when a scored candidate falls outside the trained range; (c) restrict v1 to a single
   authoring owner per engine. Recommendation: (a) for v1, with (b)'s coverage warning as the detection
   mechanism — it is the only option that surfaces, at a drop-in pack's boundary, what the model was
   trained to rank.
   *(Impacts [Section 5](#5-selection-flow), [Section 8.2](#82-model-updates), [Section 16](#16-risks).)*

6. **Engine estimate (A) vs. config UHD (B).** Is A a distinct trained model, or derived from B as
   max predicted score over candidates? Distinct is cheaper for the quick policy (skips enumeration);
   derived is one fewer model to train.
   *(Impacts [Section 11.1](#111-the-engine-estimate-and-the-kernel-catalog-ranker).)*

7. **Non-descriptor engine estimates.** How does a non-descriptor engine (e.g. MIOpen) report an
   A-level estimate through the plugin-query surface? If it cannot, the quick policy falls back to
   static ordering for it — acceptable for v1, but limits performance-based engine ranking.
   *(Impacts [Section 11.2](#112-two-engine-selection-policies-rfc-0007).)*

8. **The `predict_applicable_kernels` output contract.** The UED already reserves the field
   ([Section 3.1](#31-descriptor-relationships)), so *whether* an engine can carry a candidate generator
   is settled — a schema slot, not an open question. What is open is what that stage **emits** and where
   it lives. Producing candidate configurations that do not yet exist is not ranking; it yields new KMD
   tuples that feed the build path, closer to catalog synthesis
   ([Section 14](#14-package-creation-time-selection-knobs-and-aot-kernels)) than to selection. So: does
   the generator's output belong in a UHD adapter at all, or does `predict_applicable_kernels` name a
   descriptor that hands off to the package-creation tooling? And it runs *during* applicability
   ([Section 4.3](#43-future-predict_applicable_kernels-when-there-is-no-catalog-to-rank),
   [Section 10](#10-applicability-flow)), unlike every other role. Deferred until the JIT path is real.

### Operational

9. **Caching scope.** The in-process cache keyed on `(engine id, graph id, device id)` + inventory
   generation counter is sufficient for repeated graphs within a session, and needs nothing
   UHD-specific ([Section 9.2](#92-loading-and-caching)). Persistent cross-run caching is a different
   design — it needs the heuristic's content hash in the storage path, a device identity that is not the
   HIP ordinal, and a garbage-collection story — and it interacts with a future
   [RFC 0007](0007_EngineSelectionHeuristicsFramework.md) "cache selector" policy. Two things to settle
   before building it: whether persisting *rankings* repays its complexity at all when compilation and
   matching dominate the cost, and whether it should instead ride along with a persistent applicability
   or compiled-kernel cache. Defer until that policy is designed.
   *(Impacts [Section 9.2](#92-loading-and-caching).)*

10. **Shape corpus location.** The variant space is fixed (the pack's UKDs, timed via autotune), but
   the **shape corpus** and any **runtime-knob grid** need a home. A validity *constraint*
   (`min:1, max:8`) expresses which values are *legal*, not which to *sample*, so a swept axis needs
   an explicit `sweep_values` / grid hint, not an inferred range. Standardize: tool-side config (less
   coupled, easier to iterate) or descriptor field (reproducible from pack alone)?
   *(Impacts [Section 13.5](#135-sweep-space-grid-vs-constraint).)*

11. **Dependency + trust audit.** Needs deeper investigation: the exact allowed dependency surface
    for a shipped provider (license, distro packaging, ROCm image contents), and for `custom_library`
    the trust/signing rules for dropping in author-compiled native code. The former decides whether
    the in-tree tree-walker must be fully first-party or may vendor a third-party evaluator; the
    latter gates the `custom_library` drop-in path.
    *(Impacts [Section 7](#7-model-adapters), [Section 9.1](#91-dependencies).)*

12. **Enumerating the valid catalog — the blocking API gap.** Knob values are reported per knob and
    independently, so the only way to address candidates today is the Cartesian product of knob ranges,
    of which a sparse catalog satisfies a vanishing fraction
    ([Section 13.2](#132-benchmarking-via-hipdnn-autotune)). This makes generation slow for moderate
    knob spaces and **infeasible for large sparse ones** — no data, so no model. Needed: a way to
    enumerate the applicable catalog (or the set of valid knob tuples) directly. Sub-questions: does
    the enumeration return knob tuples or opaque candidate handles; is it bounded/paged for large
    catalogs; is it public API or generation-tool-only; and does `[min, max, step]` live on the KMD
    field or the UED knob ([Section 3.2](#32-kmd-fields-and-knobs-as-the-heuristics-feature-axes))? Owned with the
    hipDNN API, not resolvable inside this RFC.
    *(Impacts [Section 13.2](#132-benchmarking-via-hipdnn-autotune), [Section 15](#15-phased-delivery) phase 5.)*

13. **Formal schema artifact.** [Section 4.1](#41-field-reference) is a reference table, not a
    machine-checkable schema. Ship a real definition — FlatBuffers table or JSON Schema, matching
    whatever the descriptor family standardizes on — so validation is generated rather than
    hand-written, and so the header/body split is enforced mechanically. Header first; adapter bodies
    as each adapter lands. Should align with the same decision for the UED.
    *(Impacts [Section 4](#4-uhd-schema).)*

14. **CI validation of shipped descriptor sets.** Because a broken feature contract degrades rather
    than fails ([Section 6.3](#63-contract-enforcement)), the runtime cannot be the thing that catches
    it — a mis-built pack would ship and silently rank by `static_order`. What validates descriptor
    sets in CI, what does it check (hash agreement, KMD pairing, signature resolvability, knob-tuple
    uniqueness), and does it run per-provider or centrally over all shipped packs?
    *(Impacts [Section 6.3](#63-contract-enforcement), [Section 12](#12-observability).)*

15. **The expression-language reference.** The descriptor expression language is specified in its own
    RFC, which is deferred and not yet written, so this document states the language's properties
    ([Section 6.2](#62-the-features_signature)) without being able to cite its sections. Two things
    settle when it lands: the normative operator list this RFC recaps, and whether the custom-operation
    hook arrives, which fixes the boundary at which a compiled scorer takes over
    ([Section 7](#7-model-adapters)). Until then, the operator set recapped here is the working contract.
    *(Impacts [Section 6.2](#62-the-features_signature), [Section 7](#7-model-adapters).)*

16. **Out-of-distribution detection.** Ship per-feature training-coverage metadata and check the
    resolved row at runtime, or manage it purely by versioning discipline
    ([Section 8.3](#83-out-of-distribution-inputs))? Recommendation: versioning plus the cheap discrete
    arch check for v1, with the artifact format leaving coverage metadata additive. Decide before the
    `tree_data` format freezes, since retrofitting it later is a format change.
    *(Impacts [Section 8.3](#83-out-of-distribution-inputs), [Section 7.2](#72-default-tree_data).)*

17. **Categorical ordering semantics.** Integer codes imply an order a tree model will split on
    (`dtype < 2`), which is meaningless for an unordered category
    ([Section 6.5](#65-categorical-encoding)). Standardize per-field: assign codes in a deliberately
    meaningful order where one exists (dtype by byte width), one-hot where none does, or let the author
    declare which. Decide before the first real model, since changing it later is a retrain.
    *(Impacts [Section 6.5](#65-categorical-encoding).)*

18. **Knob-set stability vs. model churn.** `UED.knobs` equals the UHD's `$kernel.*` feature set
    ([Section 3.2](#32-kmd-fields-and-knobs-as-the-heuristics-feature-axes)), which makes the public knob list only
    as stable as the model: a retrain that drops `split_k` as a feature removes it as a knob and breaks
    a caller that was setting it. Two sub-questions. (a) Is a **"sticky" knob set** needed — allowing
    `UED.knobs` to be a *superset* of the feature set, so a knob survives as an accepted-but-unmodelled
    filter after the model stops using it? That trades the exact-equality check
    ([Section 6.3](#63-contract-enforcement)) and the "every knob provably matters" property for API
    stability. (b) What does an engine with **no UHD** expose — under a strict reading, no knobs at all,
    which removes a caller's ability to pin a kernel on a pack that has not been benchmarked yet.
    Recommendation: strict equality for v1 (it is the property that makes the knob list meaningful),
    knob removal as a major UED version bump, and revisit (a) if a real consumer is broken by churn.
    *(Impacts [Section 3.2](#32-kmd-fields-and-knobs-as-the-heuristics-feature-axes), [Section 6.3](#63-contract-enforcement), [Section 8.1](#81-descriptor-versions-and-uhd-coupling).)*

---

## 18. Glossary

- **UHD (Universal Heuristic Descriptor):** A kernel-selection model. An engine names up to three by
  role, arch-keyed ([Section 3.1](#31-descriptor-relationships)): `sort_kernel_catalog` (ranks the
  catalog — the main one, and what unqualified "UHD" means here), `predict_engine_tflops` (a cheap
  engine-level estimate), and the future `predict_applicable_kernels` (candidate generator). Each is
  per-engine and arch-aware (takes `$device.*`), composed of a **universal header** (identity, feature
  contract, objective) and an **adapter-scoped body** ([Section 4.1](#41-field-reference)).

- **Catalog:** The set of an engine's kernels that pass every matcher for one graph — engine-scoped, the
  union across every KDP joining that engine ([RFC 0017](0017_UniversalKernelDescriptor.md)). The
  pipeline's initial candidate set. Determined by the **matcher**, before any UHD is loaded.

- **KDP (Kernel Descriptor Pack):** The pack that joins an engine and adds kernels; names one matcher
  set, one UED (which carries the UHD and KMD), and one UDD over a vector of child UKDs. The selection
  group is a pack's child kernels; the selector and metadata schema come from the engine.

- **KMD (Kernel Metadata Descriptor):** [RFC 0017](0017_UniversalKernelDescriptor.md)'s explicit,
  upfront declaration of the engine's **compilation knobs** — the variant `fields` (`tile_m`, `split_k`,
  `dtype`, …) every kernel carries, each with a type and optional default. **One KMD per engine, owned
  by the UED**; each UKD's `metadata` fills it. It is the authoritative schema for the `$kernel.*` fields
  the UHD ranks on and the `features_signature` references
  ([Section 3.2](#32-kmd-fields-and-knobs-as-the-heuristics-feature-axes)).

- **UED (Universal Engine Descriptor):** The UED names one UHD and one KMD. They are coupled — the KMD
  is the feature space the UHD ranks over — so the engine owns both; a *breaking* KMD change requires
  retraining the UHD, while additive changes and dispatch-only fields do not
  ([Section 3.3](#33-coupling-rules)).

- **`global.` knobs:** hipDNN's reserved knob namespace ([RFC 0004](0004_EngineConfigKnobs.md)), which a
  descriptor-backed engine implements like any other engine. Per
  [RFC 0020 §5](0020_UniversalEngineDescriptor.md) it is **separate from** the UED's `knobs` list and the
  two do not overlap — so `global.` knobs are not KMD fields and are not part of the UHD's feature space.

- **Knob:** A **KMD field the UHD ranks on**, surfaced to the user by name in the UED's `knobs` (the KMD
  already declares its type and default). `UED.knobs` equals the set of `$kernel.*` fields the UHD reads,
  so the knob list is *derived* from the trained model rather than chosen independently — the user-facing
  axes are exactly the performance-relevant ones. A knob's legal values come from the *catalog* for this
  graph, and its **default is whatever the UHD ranks first**. Knobs *filter* the catalog; the UHD then
  ranks what survives ([Section 3.2](#32-kmd-fields-and-knobs-as-the-heuristics-feature-axes)).

- **Knob tuple:** The values of an engine's knobs for one kernel — the kernel's projection onto the
  model's feature axes, and what an outside caller can name. Narrower than the **metadata tuple**, which
  covers every KMD field and is unique engine-wide. Two kernels differing only in **dispatch-only**
  fields (KMD fields that are not knobs, read by a UDD) share a knob tuple, score identically, and are
  separated by the deterministic tie-break — legal, but warned about at load
  ([Section 3.2](#32-kmd-fields-and-knobs-as-the-heuristics-feature-axes)). Note also that the *cross-product* of
  legal knob values is not the set of legal tuples: most combinations may match no kernel, which is the
  enumeration gap of [Open Question 12](#operational).

- **Benchmark wide, expose what the model keeps:** Generate against a UED exposing every KMD field — so
  every kernel is addressable and individually timeable — then let feature selection prune the axes that
  don't earn their place, and emit a UED whose `knobs` are exactly the survivors. Pruning a feature and
  dropping a knob become one action ([Section 3.2](#32-kmd-fields-and-knobs-as-the-heuristics-feature-axes),
  [Section 14.4](#144-pipeline-knob-reduction-hipdnn-jit-case)).

- **`categorical_encoding`:** The map from a string KMD field's values to numeric codes, generated by
  the tooling from the values observed during training and shipped in the UHD. Part of the resolved
  feature contract, so `features_hash` covers it; an unseen value at runtime is an out-of-distribution
  signal, not an error ([Section 6.5](#65-categorical-encoding)).

- **`trained_against`:** The UED, UMD, and KMD semantic versions a heuristic was generated against.
  Checked at load (`major ==`, `minor <=`) to disable a model whose descriptors have moved under it
  ([Section 8.1](#81-descriptor-versions-and-uhd-coupling)).

- **Dispatch-only field:** A KMD field that is *not* a knob, and therefore not in the UHD's feature set —
  launch geometry or workspace detail a UDD consumes. Invisible to selection
  ([Section 3.2](#32-kmd-fields-and-knobs-as-the-heuristics-feature-axes)).

- **KMD field space:** The engine's variant axes (`tile_m`, `warp_n`, `split_k`), filled per-kernel in
  UKD `metadata` — the space the UHD ranks over, read as `$kernel.*`. Each UKD is one point in it; the
  KDP is the collection. Knobs are a user-facing *subset* of these fields, not a separate category
  ([Section 3.2](#32-kmd-fields-and-knobs-as-the-heuristics-feature-axes)).

- **Kernel-selection heuristic vs. engine-selection heuristic:** The two levels; the UHD is the former
  (which kernel within an engine), [RFC 0007](0007_EngineSelectionHeuristicsFramework.md) owns the
  latter (which engine).

- **`features_signature`:** The UHD's ordered, versioned list of model inputs (bare `$`-prefixed fields
  or expressions) that both training and inference consume through one generic extractor; the contract
  that must stay bit-identical across the two. Fingerprinted by `features_hash` **together with the
  the `categorical_encoding` it uses** ([Section 6.3](#63-contract-enforcement)); because computed
  features are inline, the signature is itself the computation.

- **Computed feature:** A `features_signature` entry that is an expression rather than a bare `$` token.
  Written **inline**; there is no `$derived.*` namespace and no named-value block. Repetition is free
  because the signature compiles once and the evaluator hash-conses identical subtrees
  ([Section 6.4](#64-computed-features)).

- **`native`:** The bootstrap adapter — a scorer compiled into the engine and named in the UHD by symbol.
  Lands first because it needs no format, parser, or extractor, and serves as the performance baseline.
  Not a drop-in: changing it means recompiling the engine ([Section 7.1](#71-first-native)).

- **`tree_data`:** The default shipping path — a GBDT tree table shipped as data with the engine's
  descriptor set and evaluated by an in-tree GBDT walker written for this work; standalone drop-in, zero
  runtime dependency.

- **`custom_library`:** The drop-in escape hatch — a compiled scorer `.so` shipped with the engine and
  `dlopen`'d through a tiny C ABI; standalone, any model family, gated on the trust audit. Distinct from
  `native`, which is compiled *into* the engine and needs no loading or trust boundary.

- **Scorer / adapter:** The thing that turns a UHD's model content into a per-candidate score; reached
  through an adapter in build-and-runtime (default) or build-only delivery classes, mirroring
  [RFC 0017 §8](0017_UniversalKernelDescriptor.md#8-adapters-and-extensibility).

- **The tooling:** The heuristic-generation pipeline this RFC builds on — a sweep step that produces a
  training dataset (problem × kernel × measured TFLOPS), a training step that fits a LightGBM regressor on
  `log1p(tflops)`, an exporter, and a dispatcher path, first exercised on SDPA/FMHA forward and reusable
  by any package author ([Section 13](#13-model-generation-pipeline)). Two properties shape this design:
  shipping the model as data is what makes a heuristic drop-in, and the feature vector is the failure-prone
  contract, so generating it from one specification keeps training and inference consistent.

- **`lgbm_to_c`:** The tooling's build-only path that lowers a LightGBM booster to C linked into the
  provider. Kept only as an optional build-time perf optimization for in-tree AOT models — **not** a
  drop-in shipping mechanism.

- **Score-only mode:** Running a UHD to obtain the best predicted score without selecting for launch;
  the hook for surfacing estimated TFLOPS to engine selection.

- **Stage P (package):** The pipeline stage that emits the engine descriptor set (UED/UHD/KMD +
  tree-table) from the same sweep that trained the model, enforcing the feature and kernel-identity
  contracts ([Section 13](#13-model-generation-pipeline)).

- **Engine estimate (A):** Cheap `f(graph) → expected performance` model for quick-policy engine ranking.
  Not required for v1; it applies once there are competing engines to order, or opaque engines with no
  catalog to rank ([Section 11.1](#111-the-engine-estimate-and-the-kernel-catalog-ranker)).

- **Config UHD (B):** Full `f(graph) → best kernel + predicted performance` model for kernel selection
  and accurate cross-engine comparison. Doubles as the stopgap engine estimate until A exists.
