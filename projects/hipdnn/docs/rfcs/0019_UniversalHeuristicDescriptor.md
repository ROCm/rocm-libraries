# RFC 0019: Universal Heuristic Descriptor (UHD): Data-Driven Kernel Selection

- Contributors: (draft — jascampb, cderb)
- Status: First draft, for discussion
- Parent: [RFC 0017 Universal Kernel Descriptor](0017_UniversalKernelDescriptor.md) (this is the "UHD + kernel selection" follow-up named in [RFC 0017 §12.2](0017_UniversalKernelDescriptor.md#122-follow-up-rfcs))
- Sibling: **RFC 0018 Universal Match Descriptor (UMD) and the Graph Matcher** ([PR #10341](https://github.com/ROCm/rocm-libraries/pull/10341)) — the matcher follow-up from the same series, with a working implementation. It pins the **JsonLogic** expression language and the `$`-token namespaces (`$q.*`, `$kernel.*`, `$device.*`) that this RFC's `features_signature` consumes ([Section 7](#7-feature-extraction)). Its `JsonLogic.hpp` — a compile-once / evaluate-many implementation with the `$`-sigil convention and the operator set below — is the concrete evaluator this RFC's feature extractor reuses.
- Related: [RFC 0007 Engine Selection and Heuristics Framework](0007_EngineSelectionHeuristicsFramework.md), [RFC 0013 Autotune](0013_Autotune.md) (the benchmarking substrate for heuristic generation, [Section 14](#14-model-generation-pipeline))

> **Numbering note.** The UMD follow-up also drafted as "RFC 0018"; that number is taken by the matcher
> RFC, so this document is **0019**. Both are follow-ups of the same [RFC 0017 §12.2](0017_UniversalKernelDescriptor.md#122-follow-up-rfcs) series.

> **Draft note.** This is a first pass to frame the design and drive discussion, not a finished
> spec. Sections marked **OPEN** carry decisions we still need to make. It is grounded in the existing
> heuristic-generation tooling — a training-and-export pipeline any package author can run to produce a
> heuristic for their own kernels — and rocKE's current selection path, both summarized in
> [Section 3](#3-prior-art).

## Table of Contents

1. [Overview](#1-overview)
2. [Scope](#2-scope)
3. [Prior Art](#3-prior-art)
4. [Ownership Model](#4-ownership-model)
5. [UHD Schema](#5-uhd-schema)
6. [Selection Flow](#6-selection-flow)
7. [Feature Extraction](#7-feature-extraction)
8. [Model Adapters](#8-model-adapters)
9. [Versioning and Compatibility](#9-versioning-and-compatibility)
10. [Performance](#10-performance)
11. [Applicability Flow](#11-applicability-flow)
12. [Engine Selection Integration](#12-engine-selection-integration)
13. [Observability](#13-observability)
14. [Model Generation Pipeline](#14-model-generation-pipeline)
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

The **UED (engine) owns the UHD**: one heuristic per engine, shared by every pack that joins it. Many
KDPs may name the same engine, and thus share its one UHD. The UED also owns the **KMD (Kernel Metadata
Descriptor)** — the explicit declaration of compilation knobs (tile size, block size, split-K, dtype,
and the like, each with a type and optional default) that distinguish kernel variants. The KMD *is* the
feature space the UHD ranks over, so the two are coupled — though only a *breaking* KMD change forces a
retrain; additive changes and dispatch-only fields do not ([Section 4.3](#43-coupling-rules)).

This RFC defines:

1. **UHD schema** — how a selection model is described as data ([Section 5](#5-uhd-schema))
2. **Ownership model** — UED owns one UHD and one KMD; KDPs join the engine ([Section 4](#4-ownership-model))
3. **Selection flow** — how a UHD ranks matched kernels ([Section 6](#6-selection-flow))
4. **Engine integration** — applicability bubbles up before engines are ranked; the tooling produces
   two heuristics (a cheap engine estimate and the fine-grained config UHD); two policies (quick vs.
   thorough) consume them, with a rank-ordering fallback ([Section 11–12](#11-applicability-flow)).
   The policy changes stay a [RFC 0007](0007_EngineSelectionHeuristicsFramework.md) follow-up; this
   RFC supplies the heuristics and keeps the schema from foreclosing cross-engine comparison.
5. **Generation pipeline** — automated benchmarking and model export ([Section 14](#14-model-generation-pipeline))

---

## 2. Scope

| Level | Question | Owner |
|-------|----------|-------|
| **Engine selection** | Which engine handles this graph? | [RFC 0007](0007_EngineSelectionHeuristicsFramework.md) |
| **Kernel selection** | Which kernel within the engine? | **UHD** (this RFC) |

A UHD is the kernel-selection heuristic. It is part of the generic provider that
[RFC 0017](0017_UniversalKernelDescriptor.md) introduces — not a new host interface, not a policy
plugin. The two levels are not cleanly one-after-the-other: applicability bubbles up before engine
selection ranks anything, and UHD predictions can feed engine selection to rank by predicted
performance ([Section 11–12](#11-applicability-flow)).

- **Applicability bubbles up first.** Before engine selection can rank anything, it must know which
  engines even apply. For a descriptor engine, "do I apply?" is the **matcher (UMD) pass** at the
  descriptor layer; that result **bubbles up** so non-viable engines are ruled out *before* the first
  plugin-policy layer ranks the survivors. So the descriptor/UHD layer runs (at least for applicability)
  *ahead of* engine selection, not strictly after it.
- **The UHD's predictions feed engine selection.** The generation tooling produces a cheap engine-level
  **expected-performance** estimate and the fine-grained config UHD; engine-selection policies consume
  those to rank engines *by predicted performance* ([Section 11](#11-applicability-flow)).
  The policies themselves remain [RFC 0007](0007_EngineSelectionHeuristicsFramework.md)'s territory; this
  RFC provides the heuristics they consult.

**In scope:** the UHD schema; ranking semantics over a pack's matched UKDs; selection-group membership
(the UED owns the UHD and KMD; a KDP joins the engine) and how KMD fields, UKD metadata, and the knobs
an engine exposes relate; the feature contract; model formats and their adapter seam; dependency
constraints; load/eval performance; the model-generation pipeline.

**Out of scope (this RFC):** the engine-selection outer loop itself ([RFC 0007](0007_EngineSelectionHeuristicsFramework.md)
owns it); autotuning / exhaustive search (device-access tuning is [RFC 0013](0013_Autotune.md)); the
matcher and launch machinery ([RFC 0017 §5–6](0017_UniversalKernelDescriptor.md#5-matching-and-the-umd)).

---

## 3. Prior Art

Two concrete systems anchor this design: the existing heuristic-generation tooling and rocKE's
selection path. The UHD generalizes what that tooling already produces onto the descriptor model,
so that any package author can generate a heuristic for their own pack with the same tools.

### 3.1 Existing heuristic-generation tooling

An end-to-end pipeline for turning benchmarks into a LightGBM kernel-selection model already exists
as reusable tooling (training scripts, exporters, and a dispatcher path), first exercised on
SDPA/FMHA forward:

- **Offline training.** A sweep step produces a training dataset (problem × kernel × measured TFLOPS);
  a training step fits a LightGBM regressor on `log1p(tflops)` with grouped cross-validation.
- **Original inference.** AOT-compiled to plain C (an exporter lowers a trained booster to a
  dependency-free C scoring function, statically linked into the provider). This gave zero runtime
  dependency but is not drop-in — adding a model meant recompiling. This RFC replaces that shipping
  path with model-as-data ([Section 8](#8-model-adapters)).
- **Model registry** keyed by `(op, arch, dtype)`, generated from the trained models, mapping a problem
  to its scoring function.
- **Feature contract.** A fixed feature vector (problem dims, dtypes, tile and warp constants,
  hardware props) is generated from one feature specification so the Python training features and the
  C++ inference features are identical; a round-trip test gates drift.
- **Selection.** Featurize the problem, look up the model, score every satisfying candidate, argmax,
  stable-order tie-break, and fall back to first-match if there is no model or the feature count disagrees.

Two lessons carry directly into the UHD design: **(a)** shipping the model as data (not linked into
the provider) is what makes a heuristic drop-in — the compiled-C path gave zero dependency but forced
a recompile per model; **(b)** the feature vector is the fragile contract, and generating it from one
specification is what keeps training and inference honest.

### 3.2 rocKE selection today

rocKE (`dnn-providers/hip-kernel-provider/rocke/`) is a hipDNN engine plugin. Its selection path is
deterministic and catalog-driven: `AotCatalog::candidatesFor(op, arch)` → `satisfies(instance, problem,
attrs)` exact-match filter → **first match wins**, with an explicit `TODO(heuristics): tie-break with
trained per-arch FMHA model when >1 instances match`. That TODO is exactly the seam a UHD fills.

The normalized `SdpaProblem` (shape, dtype, layout, mask/dropout/alibi attributes, arch) is the feature
source already present at the selection point, and each `AotInstance` carries the `CompileSpec`
(tile/block/dtype/layout constants) that becomes a UKD's `metadata`. rocKE's selection is internal to
the engine and orthogonal to [RFC 0007](0007_EngineSelectionHeuristicsFramework.md): the UHD ranks
*within* rocKE, after the catalog match — in [RFC 0017](0017_UniversalKernelDescriptor.md) terms,
after a pack's matcher set passes, the engine's UHD ranks its surviving child kernels.

---

## 4. Ownership Model

### 4.1 Descriptor Relationships

```
UED (engine)
 ├── heuristic: UHD id    ← one selector per engine
 ├── metadata:  KMD id    ← one metadata schema per engine
 └── knobs: [...]         ← user-facing runtime parameters

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
// The UED owns the selector + metadata schema; the KDP joins the engine and adds kernels.
{
  "schema": "hipdnn.ued/v1",
  "id":        "efc9eae4-…",        // engine identity
  "heuristic": "ae896b07-…",        // UHD: the selector for this engine's kernels   <-- membership
  "metadata":  "9ae0b215-…",        // KMD: the variant-field schema this engine's kernels fill
  "knobs":     ["split_k", "tile_m"] // names of KMD fields this engine exposes to the user (Section 6.1)
}

{
  "schema": "hipdnn.kdp/v1",
  "arch":      ["gfx942"],
  "matchers":  ["968156a8-…"],      // shared matcher set (UMD ids)
  "engine":    "efc9eae4-…",        // UED: the engine these kernels join (carries the UHD + KMD)
  "dispatch":  "625df14f-…",        // UDD: how they launch
  "kernelDescriptors": [            // the selection group = these child UKDs
    {"id": "15b02840-…", "kernel_source": {/* ... */}, "metadata": {"tile_m": 128, "split_k": 1}},  // fills the KMD
    {"id": "562e3777-…", "kernel_source": {/* ... */}, "metadata": {"tile_m": 256, "split_k": 1}}
    // ... the family, differing only in metadata (their compile-time build config)
  ]
}
```

### 4.2 KMD fields, and knobs as a view onto them

There is **one** space of variant fields, not two. The reworked
[RFC 0017](0017_UniversalKernelDescriptor.md) settles this: the **KMD declares the engine's variant
fields** (name, type, optional default), each UKD's `metadata` fills them, and a **knob is simply a KMD
field the engine chooses to expose to the user** — a *name*, nothing more.

| Concept | Where | What it is |
|---|---|---|
| **KMD fields** | **KMD** `fields`, filled by each UKD's `metadata` | The engine's variant axes (`tile_m`, `warp_n`, `split_k`, `dtype`). **This is the space the UHD ranks over**, read as `$kernel.*`. The field set must uniquely key every kernel variant. |
| **Knobs** | **UED** `knobs`: a list of **field names** | A *view* onto a subset of those same KMD fields, marking them user-controllable. Only KMD fields can be knobs; a name matching no field is a load error. |

Three consequences from [RFC 0017](0017_UniversalKernelDescriptor.md) that matter to the UHD:

- **No second source of truth.** The UED restates neither type nor default — the KMD already declares
  them and every kernel already carries a value. Exposing a field is additive and reversible.
- **A knob's legal values come from the *catalog*, not the schema** — the values the field actually takes
  among the kernels matching *this* graph, never the KMD's theoretical range.
- **A knob's default is the UHD's choice.** Whatever the heuristic ranks first is the reported default, so
  leaving knobs alone reproduces the out-of-the-box selection. **This couples the UHD to knob reporting:**
  answering a knob query means ranking that engine's catalog — see
  [Section 10](#10-performance).

Filtering and ranking **commute**: setting `split_k = 4` keeps only kernels whose `split_k` is 4 and the
UHD ranks those. That holds only because **a UHD scores each kernel on its own metadata and the problem,
never relative to the rest of the catalog** — a hard requirement on any UHD adapter
([Section 6](#6-selection-flow)), not an assumption. A scorer that normalizes across the candidate set
is out of scope.

This supersedes an earlier draft of this RFC that split "UED runtime knobs" from "KMD compilation knobs"
as two disjoint categories. They are not disjoint: `split_k` is a KMD field that an engine may *also*
expose as a knob. The real distinction that survives is **who sets the value** — the kernel's build (every
KMD field, as `metadata`) versus the user (the exposed subset) — not two different kinds of parameter.

### 4.3 Coupling Rules

**The KMD is the schema for `$kernel.*`, and the UED owns both it and the UHD.** The UED references one
KMD and one UHD, and every child UKD's `metadata` fills the KMD's fields, validated at load. Putting both
on the UED is deliberate — the KMD *is* the feature space the UHD ranks over, so they are coupled. But
the coupling is **conditional, not unconditional** (matching [RFC 0017](0017_UniversalKernelDescriptor.md)):

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

- **A UKD is one point in the KMD-declared field space** — its `metadata`, and its unique key in the catalog.
- **The collection of those points is the KDP** (a pack joining one engine, adding a matcher set, a UDD,
  and the kernel vector); the **UHD and KMD belong to the UED (engine)**, shared by every pack that joins
  it. `arch` is a KDP property, so one engine — and its one UHD/KMD — spans arches.
- **The UHD's `features_signature` `$kernel.*` references must be a subset of the KMD fields** — a
  load-time check ([Section 7.3](#73-contract-enforcement)). The KMD is the authority on *what fields
  exist*; the `features_signature` picks *which subset* it ranks on (the rest may serve the UDD), and how
  it derives from them.

**One UHD per engine.** Many KDPs may join the same engine and share its UHD. The UHD is never inlined
per kernel or per pack.

**Arch-aware via `$device.*`.** The UHD spans architectures; `arch` is a KDP property, not a UHD
property. The model takes `$device.*` features (CU count, LDS size, etc.) so it generalizes across
the arches its engine serves. **OPEN:** See [Open Question 2](#schema-and-training) (arch-aware model scope).

---

## 5. UHD Schema

A UHD is a small, reusable scoring recipe. It names an `adapter` (ranking mechanism), a
`features_signature` (model inputs), an objective, and — for model adapters — a model artifact.

Because a UED spans arches (`arch` is a KDP property in [RFC 0017](0017_UniversalKernelDescriptor.md)),
a UHD is **per-engine and arch-aware**: one model taking `$device.*` features so it generalizes across
the arches its engine serves — not one model per arch.

```jsonc
// tree_data — the default; a GBDT tree table, shipped as data with the engine's descriptor set
{
  "schema":  "hipdnn.uhd/v1",
  "id":      "ae896b07-80cd-473c-b3f4-6a8892998519",   // GUID; referenced by the UED (one per engine)
  "name":    "rocKE FMHA fwd selector",                // per-engine, arch-aware — not per-arch
  "adapter": "tree_data",                              // the ranking mechanism (Section 8)

  // ordered model inputs; order + form must match training (Section 7)
  "features_signature": [
    "$device.cu_count", "$device.lds_size",            // device props → arch-aware
    "$kernel.tile_m", "$kernel.split_k",               // KMD fields (compilation knobs)
    "$sdpa_fwd.head_size", "$q.seqlen_q",              // graph node attr + tensor dim
    {"*": ["$q.batch", "$q.num_heads"]}                // a derived feature (expression)
  ],
  "features_hash": "sha256:…",                         // fail-closed input-contract guard (Section 7.3)

  "objective": "max",                                  // higher predicted score wins
  "score": {"units": "tflops", "calibrated": true, "transform": "log1p"},  // recover TFLOPS → Section 12

  "model": {"artifact": "fmha_fwd/model.bin"}          // ships as data with the engine descriptors
}
```

Other adapters keep the same head and vary the body:

```jsonc
// onnx — same shape, different adapter; dependency-gated (Section 8)
{ …, "adapter": "onnx", "model": {"artifact": "fmha_fwd/model.onnx"} }

// static_order — no features, no model, no hash
{ "schema": "hipdnn.uhd/v1", "id": "…", "name": "…", "adapter": "static_order",
  "order": ["priority", "id"] }

// custom_library — native scorer; features_hash advisory if it self-features
{ …, "adapter": "custom_library",
  "features_signature": [ … ], "features_hash": "…",
  "model": {"symbol": "vendor.fmha_scorer", "config": { … }} }   // symbol + typed config, never inline code
```

### 5.1 Adapter Summary

`adapter` is a **single discriminant** — it subsumes [RFC 0017](0017_UniversalKernelDescriptor.md)'s
illustrative `kind` + `model.framework` into one field (`tree_data` ≈ `kind:model, framework:lightgbm`
shipped as data), and the body is an adapter-keyed union. Adding a new ranker (a static list, ONNX, a
new model family) is one more `adapter` value — the single discriminant is what makes that additive.

| `adapter` | What it is | Ranking | Model artifact |
|-----------|------------|---------|----------------|
| `static_order` | A fixed precedence with no learned model | Declared order / UKD `priority` | none |
| `table` | A CSV/lookup keyed by coarse problem buckets | Table lookup, then tie-break | with engine |
| `tree_data` | A GBDT tree table (LightGBM/XGBoost), in-tree walker — **default** | Score each candidate, argmax | with engine |
| `onnx` | An ONNX graph via a gated runtime | Score each candidate, argmax | with engine |
| `custom_library` | A registry-resolved native scorer behind a small C API (escape hatch) | Whatever the library returns | with engine |

See [Section 8](#8-model-adapters) for adapter details.

**The model ships as data with the engine, not linked into the provider.** For every model adapter,
`model.artifact` is a path resolved relative to the engine's descriptor set (the UED + its UHD + KMD +
model), which is itself standalone-droppable. The model is per-engine (owned by the UED), so there is
no `(arch,dtype)→artifact` table — the single arch-aware model serves every pack that joins the engine.

**OPEN — regressor vs. ranker.** The tooling trains a *regressor* on TFLOPS and argmaxes. A
learning-to-rank objective (LambdaRank/NDCG) optimizes ordering directly and may pick better *within*
an engine without needing calibrated absolute values. But a calibrated TFLOPS regressor is what makes
the **absolute, cross-comparable metric** of [Section 12.3](#123-cross-engine-comparison) possible; a
pure ranker forecloses that and leaves only the rank-ordering fallback. We likely want the regressor to
preserve the absolute option, keeping ranking as the fallback rather than the only mode. Decide per-UHD
via `objective` / `score`, or standardize. See [Open Question 1](#schema-and-training).

---

## 6. Selection Flow

The generic engine produces applicable candidates as follows: a KDP's shared matcher set passes for
the graph ([RFC 0017 §5](0017_UniversalKernelDescriptor.md#5-matching-and-the-umd)), and its child
UKDs are the candidates, each carrying its build `metadata`. The pack joins an engine (UED), and that
engine owns the one UHD, so the candidate set and its selector arrive together.

All child UKDs joining one engine should be mutually substitutable for the graphs they co-match (same
op family), because the model is trained to rank exactly that catalog. Kernel selection then:

1. **Take the pack's candidate set.** The matched pack yields its child UKDs, and its engine yields
   the one UHD. If two different packs match the same graph, see [Open Question 5](#structural).
2. **Extract features once.** Build the feature vector for the problem from the bound match variables
   and device properties ([Section 7](#7-feature-extraction)). Per-candidate features come from each
   UKD's `metadata` (its compile-time build config); problem/device features are shared across the set.
3. **Score each candidate.** Invoke the UHD's scorer per candidate. For a model adapter this is one
   inference call per candidate over its feature row.
4. **Choose by objective.** `max` (or `min`) over the scores; the winner is the selected kernel.
5. **Tie-break deterministically.** On equal scores (or when the UHD declines / is absent), fall
   through to explicit UKD `priority`, then stable `id` — the same deterministic arbitration
   [RFC 0017 §5](0017_UniversalKernelDescriptor.md#5-matching-and-the-umd) defines. Declaration order
   is never used.
6. **Fail open to a safe default.** If no model loads, the feature contract mismatches, or the scorer
   errors, selection degrades to `static_order` (priority + id). This mirrors the tooling's first-match
   fallback and keeps a bad/absent model from breaking execution.

The winner is a single UKD, which then dispatches through the pack's one UDD
([RFC 0017 §6](0017_UniversalKernelDescriptor.md#6-dispatch-and-workspace)). A UHD only ranks; it
never launches, mutates the graph, or touches device memory.

---

## 7. Feature Extraction

The feature vector is the contract between training and inference, and the fragile part of the whole
system. Generalizing it is the core hard problem of this RFC.

### 7.1 Feature Sources

A feature row is assembled from three sources, all already available at plan time, and all drawn from
the same field namespaces the [RFC 0017 §5](0017_UniversalKernelDescriptor.md#5-matching-and-the-umd)
criteria read (`$q.*`, `$graph.*`, `$<node>.*`, `$kernel.*`, `$device.*`, referenced bare, no `var`
wrapper). The UHD's `features_signature` is just another consumer of that vocabulary, so matching,
launch, and selection share one binding:

| Source | Namespace | Examples | Scope |
|--------|-----------|----------|-------|
| **Problem** | `$q.*`, `$<node>.*` | `$q.seqlen_q`, `$sdpa_fwd.head_size` | Shared across candidates |
| **Device** | `$device.*` | `$device.cu_count`, `$device.lds_size` | Shared across candidates |
| **Kernel** | `$kernel.*` | `$kernel.tile_m`, `$kernel.split_k` | Per-candidate (from UKD `metadata`) |
| **Derived** | `$derived.*` | `$derived.num_tiles_m`, `$derived.arithmetic_intensity` | Computed from the above by the UED's `derived` block ([Section 7.4](#74-derived-values-the-uhd-derived-block)) |

Problem features are dims, dtypes, stride order, and op attributes bound by the matcher set.

> **Implementation note — `$q.*` is op-specific.** The problem namespace differs per operation: SDPA
> exposes `batch`, `seqlen_q`, `seqlen_k`, `num_heads`, `head_dim`; convolution exposes `n`, `c`, `h`,
> `w`, `k`, `r`, `s`, `pad`, `stride`, `dilation`; MoE exposes `num_experts`, `top_k`, `hidden_dim`,
> etc. The engine is op-scoped, so its UHD's `$q.*` references are implicitly valid for that op. The
> extractor evaluates whatever fields the signature names; the caller provides them from the bound
> match variables.

Device features come from the same device-facts path [RFC 0007 §6](0007_EngineSelectionHeuristicsFramework.md#6-device-properties)
defines. Kernel features are the compilation knobs the pack's KMD declares ([Section 4.2](#42-kmd-fields-and-knobs-as-a-view-onto-them)) —
these distinguish candidates within a pack and are what make argmax meaningful.

> **Implementation note — `$device.*` fields.** The device namespace exposes what rocminfo (or
> equivalent HIP runtime queries) provides and what proves predictive for kernel selection. Expected
> fields include: `arch` (e.g., `gfx942`, `gfx950`), `cu_count`, `lds_size`, `sgpr_count`, `vgpr_count`,
> `max_waves_per_cu`, `memory_clock_mhz`, `memory_bus_width`, `peak_bandwidth_gbps`. The exact set will
> be finalized against rocKE's existing device-feature vocabulary and extended as real sweeps reveal
> additional predictive properties.

### 7.2 The `features_signature`

The feature contract is the UHD's inline `features_signature` — an ordered list of model inputs, bound
the same way a UDD's `args_signature` binds kernel arguments. Each entry is either a direct field (a
bare `$`-prefixed reference) or a derived expression over those fields, using the
[RFC 0017 §5](0017_UniversalKernelDescriptor.md#5-matching-and-the-umd) expression tree (`{"op": [args]}`).
Order and form must match training exactly.

```jsonc
"features_signature": [
  // --- direct fields (shared namespaces) ---
  "$q.seqlen_q",                                   // graph tensor dim
  "$sdpa_fwd.head_size",                           // graph node attribute
  "$device.cu_count",                              // device property (arch-aware)
  "$kernel.tile_m",                                // KMD field (per-candidate)

  // --- derived features, computed by the shared interpreter ---
  {"log2": ["$q.seqlen_q"]},
  {"/": ["$q.seqlen_q", "$k.seqlen_k"]},                          // aspect ratio
  {"ceil_div": ["$q.seqlen_q", "$kernel.tile_m"]},                // num_tiles_m
  {"/": [{"*": ["$q.seqlen_q", "$k.seqlen_k", "$sdpa_fwd.head_size"]}, "$q.bytes"]}  // ~arithmetic intensity
]
```

**The expression language is JsonLogic, pinned by the UMD RFC.** RFC 0018 (UMD) fixes the concrete form
that [RFC 0017 §5–6](0017_UniversalKernelDescriptor.md#5-matching-and-the-umd) left open: a **JsonLogic**
`{"op": [args]}` tree, where **any string beginning with `$` is a variable reference** and every other
scalar is a literal — no `{"var": …}` wrapper is used or accepted. The `features_signature` is a third
consumer of that same language, alongside UMD criteria (boolean-valued) and UDD dispatch formulas
(value-valued); a feature entry is value-valued like a dispatch formula. One parser, validator, and
interpreter serve all three.

**The operator set already covers our derived features.** An earlier draft of this RFC asked for
`log2` / `/` / `min` / `max` / `-` as required extensions. That ask is **resolved** — the UMD RFC's
operator list already includes them:

> `and or !` · `== != < <= > >=` · `in` · `all` · `+ - * / %` · `shape rank divisible value_or_default` ·
> `if` · `ceil_div min max abs pow log2 rsqrt`

So the derived features — log-scale sizes, aspect ratios, tile/wave quantization, intensity — are all
expressible today with no extension request. Evaluation is the same safe, bounded interpreter: it fails
closed on an unknown symbol, a type error, or an invalid operation, and uses checked-width integers.
Anything beyond this closed set (e.g. a bespoke occupancy model) uses the `custom_library` escape hatch
([Section 8](#8-model-adapters)) rather than growing the interpreter unbounded.

**Implementation status (from the UMD PoC, [PR #10341](https://github.com/ROCm/rocm-libraries/pull/10341)).**
The shared `JsonLogic.hpp` already implements the arithmetic and comparison operators this RFC's features
need (`+ - * /`, `min`, `max`, `ceil_div`, `abs`, `pow`, `log2`, `rsqrt`, `value_or_default`, `if`) as a
compile-once / evaluate-many `Expression<DataT>` over any `getData(path) → Value` source — which is
exactly the shape the feature extractor wants ([Section 10](#10-performance)). Two caveats carry into
this RFC:

- **`shape` is a compiler short-hand, not a JsonLogic op** — the UMD compiler lowers it at
  compile-time. A feature spec should treat `shape`/`rank`-style helpers as lowered forms, not runtime
  operators.
- **The custom-operation (native-predicate) hook is deferred** in the PoC. Until it lands, a UHD that
  needs a computation outside the closed operator set has no in-language path; that is the boundary at
  which [Section 8](#8-model-adapters)'s `custom_library` adapter (a compiled scorer) takes over.

**Answering the UMD RFC's open question on the feature source.** RFC 0018 (UMD) asks whether "the UMD's
bindings should be the canonical feature source for kernel selection," noting the bound symbol table
overlaps the feature vector a UHD consumes. **This RFC's answer is yes**: the `features_signature` draws
its problem features from exactly the symbols the matcher bound, plus `$device.*` and the candidate's
`$kernel.*` ([Section 7.1](#71-feature-sources)). One binding pass feeds matching, dispatch, *and*
selection — no parallel feature-extraction path, and by construction a feature can only reference values
the match actually produced.

**Derived features commonly needed:**

- **Arithmetic / algorithmic intensity** (FLOPs ÷ bytes) — the single most important derived feature
  for predicting compute-bound vs. memory-bound behavior.
- **Tile/wave quantization** — `num_tiles_*`, `total_output_tiles`, `tile_efficiency` (problem-vs-grid
  remainder waste). In GEMM sweeps this family is as predictive as intensity.
- **Aspect ratios** — `M/N`, `M/K`, `N/K` (shape skew).
- **Occupancy proxy** — `lds_usage_ratio` (and register pressure if available) → waves/CU.
- **Padding-fit** — `needs_padding_*` / `has_padding_when_needed_*` (problem × kernel padding interaction).

**OPEN:** See [Open Question 4](#schema-and-training) (derived feature set).

### 7.3 Contract Enforcement

The tooling code-generates the C++ featurizer to guarantee training and inference vectors are identical.
We keep that principle but make the `features_signature` the single source of truth for both sides:

- **Inference:** A generic feature extractor walks the signature against the bound-variable table and
  device facts, producing the row in declared order. Generic beats code-gen here because the sources
  are declarative and finite — no per-op C++. (Code-gen remains an option for a hot path.)
- **Training:** The same signature drives the offline featurizer, so dataset columns match the runtime
  row by construction.

A three-part, model-agnostic contract check replaces the tooling's bare feature-count guard, and
generalizes to any ranker (LightGBM, ONNX, a custom scorer):

1. **Signature → KMD.** Every `$kernel.*` field the `features_signature` references must be declared
   in the engine's KMD ([Section 4.2](#42-kmd-fields-and-knobs-as-a-view-onto-them)). Both the KMD and
   the UHD are owned by the UED, so this is an intra-engine check the pipeline enforces when it emits
   the engine and the loader can re-check — a feature can never read a compilation knob the kernels
   don't carry.
2. **Signature → model.** The UHD carries `features_hash`; the model artifact embeds the signature
   hash it was trained against (tree-table metadata, ONNX `metadata_props`, or a sidecar). At load,
   assert `model.trained_hash == UHD.features_hash` and fail closed on mismatch. This check works for
   every model adapter because it fingerprints the *input contract*, not the model internals.
3. **Vector → input.** Each adapter verifies its artifact accepts the resolved vector: `tree_data`
   checks feature count, `onnx` checks input arity/shape.

Fail-closed on any mismatch. `features_hash` is optional for feature-less adapters (`static_order`)
and advisory for `custom_library` that self-features.

**Feature-vector portability.** Do we standardize one graph/device feature extractor so models are
portable across UHDs, or keep it per-model via the spec? Recommendation: **per-model spec, shared source
vocabulary**. The spec is per-model (models differ), but every spec draws from one fixed, versioned
source vocabulary and one extractor, so there is exactly one implementation to trust.

### 7.4 Derived Values: the UHD `derived` block

> **DISCUSSION POINT (not settled).** Where derived features are computed and defined is an open team
> decision. This RFC places the `derived` block on the **UHD** (with the model it feeds); the main
> alternative — the UED (with the engine) — is noted at the end for the team to weigh.

Many of the features a model actually ranks on are not raw fields but **computed** ones: tile/wave
quantization (`num_tiles_m = ceil_div($q.seqlen_q, $kernel.tile_m0)`), aspect ratios, arithmetic
intensity, occupancy proxies. These are expressible today as inline `features_signature` entries
([Section 7.2](#72-the-features_signature)), but they **layer** — `total_tiles = num_tiles_m *
num_tiles_k`, `overall_tile_efficiency` builds on both — so repeating subexpressions inline is
error-prone. A named, reusable `derived` block fixes that.

**The `derived` block lives on the UHD.** The UHD carries a named, ordered set of derived values, each a
JsonLogic expression over `$q.*` / `$device.*` / `$kernel.*` (and earlier `$derived.*` entries). They
form a new `$derived.*` namespace the same UHD's `features_signature` references by name, exactly like a
raw field:

```jsonc
// on the UHD, alongside its features_signature and model artifact
"derived": {
  "num_tiles_m":  {"ceil_div": ["$q.seqlen_q", "$kernel.tile_m0"]},
  "num_tiles_k":  {"ceil_div": ["$k.seqlen_k", "$kernel.tile_n0"]},
  "total_tiles":  {"*": ["$derived.num_tiles_m", "$derived.num_tiles_k"]},
  "tile_eff_sq":  {"/": ["$q.seqlen_q", {"*": ["$derived.num_tiles_m", "$kernel.tile_m0"]}]},
  "tile_volume":  {"*": ["$kernel.tile_m0", "$kernel.tile_n0", "$kernel.tile_k0"]},
  "arithmetic_intensity": {"/": ["$sdpa_fwd.umd_flops", "$sdpa_fwd.umd_bytes"]}
},
"features_signature": ["$q.seqlen_q", "$kernel.tile_m0",
                       "$derived.num_tiles_m", "$derived.overall_tile_efficiency",
                       "$derived.arithmetic_intensity"]
```

Why the UHD:

- **Self-contained, drop-in.** A UHD ships with *everything it needs to featurize* — the block and the
  signature that names it, next to the model artifact. Dropping in a regenerated UHD "just works" with no
  dependency on a separately-versioned engine descriptor already declaring the right derived fields.
- **The tooling owns one artifact.** The generation pipeline emits the UHD anyway
  ([Section 14](#14-model-generation-pipeline)); if a new model version references a new derived feature,
  the tool adds it to the same UHD it regenerates — no second descriptor to edit in lockstep, no
  cross-descriptor contract to keep valid.
- **One evaluator, compile-once.** The block is a dependency-ordered DAG of the same JsonLogic the
  matcher and dispatch use; it lowers to a compiled expression at load and evaluates in order
  ([Section 10.3](#103-efficient-evaluation-expressive-spec-fast-hot-path)). Values referencing only
  `$kernel.*` are graph-independent and can be cached per kernel; values referencing `$q.*`/`$device.*`
  are per-graph. The evaluator classifies each automatically — the author need not partition them.
- **The KMD stays a flat value schema.** Correspondences and computed values live in the UHD `derived`
  block, not the KMD.
- **Not the custom-operation hatch.** Quantization/intensity are closed-form JsonLogic, so they belong
  in `derived`, not behind a native predicate. The `custom_library` / native-predicate escape hatch
  ([Section 8](#8-model-adapters)) stays reserved for genuinely non-closed-form computations.

**Cost of this choice — and the alternatives (for the discussion):**

- *On the UED instead (main alternative).* The dim↔tile correspondence is really *engine structure*
  (one tiling, stable across retrains) shared by both heuristics an engine carries — the cheap estimate
  (A) and the config UHD (B) — and by every model version. On the UHD, that structural block is
  **duplicated** across A, B, and each retrain, and re-emitted by the tooling even though it rarely
  changes. The UED keeps it in one place and out of the regenerated artifact, at the cost of the
  self-containment above (a dropped-in UHD then depends on the UED declaring the fields it references).
  **The RFC picks the UHD for drop-in simplicity; the UED is the alternative if duplication across A/B
  and versions becomes the bigger pain.**
- *Split across KMD (graph-independent) + UED/UHD (graph-dependent).* A KMD-derived value would also be
  visible to the matcher and dispatch, not just selection — but it splits one concept across descriptors
  and makes authors choose which half a value goes in. Deferred.
- *Per-op-schema annotation of the correspondence.* The correspondence is engine-specific (tile field
  *names* are the engine's), so it does not belong on the shared op schema. Only the FLOP/byte formulas
  (`umd_flops`/`umd_bytes`) are truly op-intrinsic and stay there
  ([Section 14.6](#146-auto-deriving-a-first-pass-features_signature)).

---

## 8. Model Adapters

The question "LightGBM, CSV, or a separate library?" is really about how model content reaches the
scorer, and it maps onto [RFC 0017 §8](0017_UniversalKernelDescriptor.md#8-adapters-and-extensibility)'s
adapter model. A UHD names an `adapter`; the adapter turns content into a scorer. Adding a new ranker
is one more adapter value. Adapters come in the same two delivery classes as kernel-source adapters.

**Design constraint:** The model travels as data with the engine, not linked into the provider. The
engine's descriptor set (UED + UHD + KMD + model) must be a standalone drop-in next to an already-shipped
provider, exactly like packs' `hsaco`/`kpack` code objects. This rules out statically linking the model
as the shipping mechanism — the model must be loadable data the running provider reads. (The reframe:
the problem was never "compiled," it was "linked into the provider." A model can still be *compiled* —
it just has to ship as a loadable artifact, not bake in.)

| Adapter | Runtime dependency | Standalone drop-in? | Notes |
|---------|-------------------|---------------------|-------|
| `static_order` | none | Yes (always available) | Trivial baseline; safe default when no model ships |
| `table` | none | Yes | CSV lookup by coarse problem buckets |
| `tree_data` | none | **Yes — default** | GBDT tree table + in-tree walker |
| `onnx` | ONNX runtime | Yes, if dep present | Opt-in, dependency-gated |
| `custom_library` | none on provider | Yes | Engine ships its own `.so`, `dlopen`'d through a tiny C ABI |

The **two-tier resolution** mirrors [RFC 0017](0017_UniversalKernelDescriptor.md)'s data → escape-hatch
→ native ladder:

### 8.1 Default: `tree_data`

The provider ships one small, generic GBDT tree-walker; the engine ships the model as a data artifact
(tree table: feature indices, thresholds, leaf values) that the walker reads. GBDT trees are trivial
to evaluate (~few hundred lines) and the tooling already dumps the model to a walkable structure. Zero
runtime dependency, fully standalone, verifier-gated. This is the hsaco-equivalent for heuristics.
**Limit:** the provider must already support the model *family* (a new family = a provider change).

**Artifact format:** FlatBuffer tree schema (recommended) — consistent with data-SDK serialization,
`Verifier`-gated, additive evolution. Alternative: parse LightGBM `model.txt` at load (lower author
friction but a bespoke parser to harden). **OPEN:** See [Open Question 3](#schema-and-training).

### 8.2 Escape hatch: `custom_library`

For a model the in-tree walker doesn't cover, the engine ships its own compiled scorer `.so`, `dlopen`'d
through a tiny C ABI (`score(const double* feats, ...) -> double`). Treelite generates such a `.so` from
a tree model. Any model family; author-native-code trust class per
[RFC 0017 §10](0017_UniversalKernelDescriptor.md#10-packaging-and-delivery).

**Why "compiled" is fine here but "linked into provider" was not.** The problem was never that a model
is compiled — it was that compiling it *into* the provider makes the model non-portable to third-party
provider builds. A model can still be compiled (Treelite `.so`), as long as it ships as a loadable
artifact *alongside* the engine's descriptor set, not statically linked into `libhipdnn_provider.so`.
`custom_library` mirrors [RFC 0017](0017_UniversalKernelDescriptor.md)'s native-predicate / custom-plan
escape hatches — same "author ships a `.so`, provider `dlopen`s it" pattern.
**OPEN:** See [Open Question 10](#operational) (dependency + trust audit).

### 8.3 Initial Support

`static_order` (trivial, always available) + `tree_data` (default shipping path) + `custom_library`
(escape hatch). CSV `table` is a cheap add for coarse bucketed heuristics. `onnx` is added when a
concrete need appears.

---

## 9. Versioning and Compatibility

### 9.1 KMD ↔ UHD Coupling

Because the UED co-owns the KMD and UHD, a KMD change *may* invalidate the trained model — but the
obligation is **conditional**, matching [RFC 0017](0017_UniversalKernelDescriptor.md) (see
[Section 4.3](#43-coupling-rules)):

- **Additive change (new field, or new legal values on a field):** **no retrain required until the
  change is exposed to selection.** The old feature space is still valid, so the existing model keeps
  ranking correctly. A field added purely for the UDD (dispatch/launch geometry) never affects the UHD
  at all, since `UHD features ⊆ KMD fields`.
  - *Caveat:* dropping in a KDP whose kernels vary along a field the model was **exposed to but not
    trained across** can still degrade ranking — the model ranks on values it never saw. This is a
    training-coverage gap, not a schema break; the fix is a retrain, not a load failure.
- **Breaking change (remove or reinterpret an existing field's values):** the retrain **must land in
  the same change**. A removed/reinterpreted field the model still references is caught at load.
- **Renaming a field:** treated as remove + add — a breaking change on the old name.

**Enforcement:** the generation pipeline emits a KMD version alongside the UHD. The loader fails closed
on an incompatible pairing and on any `features_signature` reference to a field the KMD no longer
declares. The trace surfaces a training-coverage warning when the catalog spans a field value outside
what the model was trained on.

### 9.2 Model Updates

Models ship as data artifacts. Update path:

1. Drop new engine descriptor set (UED + UHD + model) alongside existing
2. Provider loads new descriptors on next initialization
3. Rollback: restore previous descriptor set

No provider recompile required.

### 9.3 Unknown Architecture Handling

When a device's `$device.*` values fall outside the training distribution (e.g., a new GPU
architecture appears that the model never saw):

- **Model may extrapolate poorly.** Tree models extrapolate by returning the leaf value of the
  nearest seen region, which may be arbitrarily wrong for distant inputs. A model trained only on
  gfx942 has no reason to rank well on gfx950.
- **Detection:** Compare `$device.arch` against the set of arches present in training metadata
  (embedded in the model artifact).
- **Fallback:** Degrade to `static_order` when device arch is unrecognized. This is safe (first-match
  still works) and explicit (logged).
- **Long-term:** Retrain with new arch data once benchmarks are available.

---

## 10. Performance

Selection runs on the plan-build path, so its cost must be small and paid at most once per distinct need.

### 10.1 Dependencies

- **Zero new runtime dependency for the default path.** The `tree_data` adapter's evaluator is in-tree,
  so the default shipping path adds no runtime library. The provider cannot grow a hard `liblightgbm` link.
- **Build-time LightGBM is acceptable.** Training and tree-table conversion run offline in the pipeline
  ([Section 14](#14-model-generation-pipeline)), never in the shipped runtime.
- **A `custom_library` scorer `.so` carries its own inference** — no provider dependency; the engine
  owns whatever it linked (e.g. a Treelite-generated evaluator).
- **`liblightgbm` at runtime is opt-in only**, behind a `lightgbm_native` adapter, for environments
  that already have it. Never a default.
- **FlatBuffers / data-SDK are already in-tree** and are the natural carrier for the `features_signature`
  and any serialized model-table.

### 10.2 Loading and Caching

- **Lazy load, triggered by *ranking* not winning.** An engine's UHD model is loaded/parsed on first
  use, not at provider startup or descriptor discovery. But per merged RFC 0017, "first use" is **any
  request that ranks the engine's catalog — including one it goes on to lose**: answering a knob query
  reports the UHD's top-ranked value as the default ([Section 4.2](#42-kmd-fields-and-knobs-as-a-view-onto-them)),
  so the model must be cheap to load and cheap to rank with, not merely rare to touch. A provider that
  never sees FMHA still never parses the FMHA model; one that *enumerates* FMHA engines does.
- **Model cache — process-scoped, not on the handle.** After first load the parsed model / tree table /
  native handle is cached for the **process**. RFC 0017 moved caching off the handle deliberately: a
  handle can be swapped between calls, rebound to another device, or destroyed while a plan built
  through it is still in use, so handle lifetime says nothing about cache validity.
- **Result cache on the descriptor cache key.** Selection is a pure function of (feature vector,
  candidate set). Reuse RFC 0017's applicability cache key — **`(engine id, graph id, device id)` plus
  the inventory generation counter** — rather than a bespoke fingerprint: the engine id because the
  catalog is per-engine, the device id because it is what `$device.*` resolved against, and the
  generation counter so a newly dropped-in pack invalidates. **OPEN:** in-process only vs. a persistent
  cross-run cache — see [Open Question 8](#operational).

### 10.3 Efficient evaluation (expressive spec, fast hot path)

Extensibility lives in the data contract; efficiency lives in a compiled core. The seam is the adapter,
and the extensibility cost is paid once per candidate (one indirect call), not per feature:

- **Lower the `features_signature` at load, never walk JsonLogic per candidate.** The UMD PoC's
  `JsonLogic.hpp` already compiles a rule to an `Expression<DataT>` once
  ([Section 7](#7-feature-extraction), [PR #10341](https://github.com/ROCm/rocm-libraries/pull/10341));
  the feature extractor reuses that so per-candidate scoring is a tight loop over a compiled expression
  and a flat tree table — no strings, no JSON, no map lookups.
- **Split the row into a shared prefix + per-candidate suffix.** Problem and device features
  (`$q.*`, `$device.*`) are identical across every candidate in the engine; only `$kernel.*` differs.
  Compute the invariant prefix **once per graph** and fill only the kernel-metadata slots per candidate,
  turning O(N × full-featurize) into O(full-featurize + N × small) for N candidates.
- **Reuse the matcher's bound symbols — don't re-extract.** The UMD matcher already bound `$q.*` /
  `$device.*` deciding applicability; selection reads that table rather than re-featurizing. (This is
  the mechanism behind the [Section 7](#7-feature-extraction) answer that the matcher bindings *are* the
  feature source.)
- **Single-candidate short-circuit.** If only one UKD survives matching, skip the model and return it —
  common, and it makes the load-on-ranking case above nearly free.

### 10.4 Latency Target

The compiled-C path is the performance floor (near-zero inference cost). The `tree_data` walker is
expected to be close — a flat tree table over a preallocated feature row is a few hundred comparisons
per candidate. Target: overhead within 2× of the compiled-C baseline. Validated per
[RFC 0017 §12.1](0017_UniversalKernelDescriptor.md#121-testing-and-performance).

---

## 11. Applicability Flow

The two selection levels ([Section 2](#2-scope)) are **not** cleanly one-after-the-other, and this is
a correction to intuition:

Engine selection cannot rank engines it hasn't ruled out. For a descriptor engine, "do I apply?" is
the **matcher (UMD) pass** at the descriptor layer; that result **bubbles up** so non-viable engines
are ruled out *before* the first plugin-policy layer ranks the survivors. So the descriptor/UHD layer
runs (at least for applicability) *ahead of* engine selection, not strictly after it.

- **Lazy load — but the trigger is *ranking*, not winning.** An engine's UHD model is loaded/parsed on
  first use rather than at provider startup or descriptor discovery (the UED itself loads eagerly per
  [RFC 0017 §3](0017_UniversalKernelDescriptor.md#3-how-it-works); its model artifact stays lazy). But
  "first use" is **any request that ranks the engine's catalog — including one the engine goes on to
  lose**: a caller enumerating its options queries knobs for every ranked engine, and answering a knob
  query means reporting the UHD's top-ranked value as the default
  ([Section 4.2](#42-kmd-fields-and-knobs-as-a-view-onto-them)). So the model must be cheap to load and
  cheap to rank with, not merely rare to touch. A provider that never sees FMHA still never parses the
  FMHA model; a provider that *enumerates* FMHA engines does.
- **Cache the loaded model — not on the handle.** After first load the parsed model / tree table /
  native handle is cached for the **process**, not per `hipdnnHandle`.
  [RFC 0017](0017_UniversalKernelDescriptor.md) moved caching off the handle deliberately: a handle can
  be swapped between calls, rebound to another device, or destroyed while a plan built through it is
  still in use, so handle lifetime has nothing to do with whether cached work is still valid.
- **Cache results where the problem repeats,** using the descriptor system's cache key rather than a
  bespoke one. [RFC 0017](0017_UniversalKernelDescriptor.md)'s applicability cache is keyed on
  **`(engine id, graph id, device id)`** plus an **inventory generation counter**: the engine id because
  the catalog is per-engine (without it one engine's catalog can answer for another in the same
  provider), the device id because it is what `$device.*` resolved against, and the generation counter so
  a newly dropped-in pack invalidates. Selection results should ride the same key — selection is a pure
  function of (feature vector, candidate set), and that key already identifies both. **OPEN**: is the
  in-process cache enough, or do we want a persistent cross-run cache (interacts with a future
  [RFC 0007](0007_EngineSelectionHeuristicsFramework.md) "cache selector" policy)?
- **Minimize init overhead.** Feature extraction is a fixed walk over the spec; inference is a handful
  of tree evaluations per candidate. Keep the feature row and any scratch preallocated per session, as
  [RFC 0017 §6](0017_UniversalKernelDescriptor.md#6-dispatch-and-workspace) does for launch. Overhead
  is validated against the compiled-C baseline in
  [RFC 0017 §12.1](0017_UniversalKernelDescriptor.md#121-testing-and-performance).

```
Graph → Matcher pass (per engine) → Applicable engines → Policy ranking → Winner → Kernel selection
```

Additionally, the UHD's predictions can feed engine selection: the same layer that gates applicability
can also report a *predicted performance*, so the policy can order engines by merit instead of a
static list. Today rocKE's `isApplicable` is that yes/no gate; what changes is that it can optionally
return a score.

---

## 12. Engine Selection Integration

This section describes how kernel-level heuristics feed engine-level selection. The policies themselves
are [RFC 0007](0007_EngineSelectionHeuristicsFramework.md)'s territory; this RFC supplies the heuristics
they consult.

### 12.1 Two Heuristics

Per engine, the generation tooling ([Section 14](#14-model-generation-pipeline)) emits two models, both
predicting absolute performance so they are comparable across engines:

| Model | Signature | Cost | Role |
|-------|-----------|------|------|
| **A: Engine estimate** | `f(graph) → expected perf` | Cheap (no config enumeration) | Quick policy engine ranking |
| **B: Config UHD** | `f(graph) → best kernel + perf` | Full per-candidate | Kernel selection + accurate cross-engine score |

A is the coarse proxy; B both selects the kernel and yields the better figure of merit. Both live on
the UED (engine-level).

**A as distinct model vs. derived from B.** If A is a distinct trained model, the quick policy can
rank engines without enumerating candidates — cheaper at selection time. If A is derived from B (max
predicted score over candidates), there's one fewer model to train and maintain, but the quick policy
must evaluate B to get A. The tradeoff is selection-time cost vs. training/maintenance complexity.
**OPEN:** See [Open Question 6](#structural).

> **Suggestion:** The two-policy design implicitly requires A to be distinct. The quick policy's value
> is that only the winning engine runs B — losers never enumerate candidates. If A is derived from B,
> every engine must run B to produce A, collapsing quick and thorough into the same operation. The
> "cheap (no config enumeration)" property of A ([Section 12.1](#121-two-heuristics) table) depends on
> A being a separate model trained on `f(graph)` alone, not `f(graph, candidates)`. See
> [Open Question 6](#structural).

### 12.2 Two Engine-Selection Policies (RFC 0007)

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

### 12.3 Cross-Engine Comparison

**OPEN:** See [Open Question 1](#schema-and-training) (regressor vs. ranker).

**RFC 0017 explicitly leaves this open, and arbitration stays hipDNN's.** An earlier draft of
[RFC 0017](0017_UniversalKernelDescriptor.md) asserted that a UHD's scores are meaningful only within
its own engine and are never compared across engines. That assertion was **removed** precisely to keep
this direction available — an early policy comparing engines on UHD-estimated throughput. What RFC 0017
does hold is that **cross-engine arbitration is hipDNN's and the user's**, exercised through three
existing mechanisms: **explicit selection** of an engine, **policy configuration** (a resolved sequence
of heuristic-policy plugins supplying the ranked engine list, where a policy may itself be a heuristic),
and **auto-tuning**, which measures engines and picks the winner outright
([RFC 0013](0013_Autotune.md)). The two policies of
[Section 12.2](#122-two-engine-selection-policies-rfc-0007) are *policy configuration*
— new policies in that existing slot, not a new arbitration surface. Auto-tuning remains the ground
truth these heuristics approximate (and the substrate that trains them,
[Section 14](#14-model-generation-pipeline)).

**The idea: an absolute, cross-comparable figure of merit.** The ambition is to score candidates
**cardinally** — an absolute metric (calibrated TFLOPS) rather than a within-group rank. If a UHD is a
calibrated TFLOPS regressor, its best-candidate score is a **predicted figure of merit for what the
engine would actually run**, expressed on a scale that means the same thing across engines. That is what
lets each engine **run its own heuristic per package, independently, and still have those results be
meaningfully comparable to every other package and engine**: hipDNN compares engines by predicted
performance instead of a fixed order — "rocKE predicts 310 TFLOPS for its best FMHA kernel; MIOpen's
estimate is 240" → pick rocKE. The value is precisely that local, per-package scoring composes into a
global comparison without a central ranker; the UHD is where the number naturally exists.

**Why absolute is harder than rank.** A per-package model only needs to be monotonic to pick correctly
among its own candidates. Making it calibrated — accurate in absolute TFLOPS so cross-engine comparison
is honest — is a strictly harder modeling problem. A miscalibrated absolute score is worse than an
honest rank because it yields confident-but-wrong cross-engine picks.

**Fallback.** If the absolute method underperforms, degrade to classic rank-ordering at the engine-policy
level — engine selection reverts to existing [RFC 0007](0007_EngineSelectionHeuristicsFramework.md)
static/rank ordering, and each UHD keeps ranking within its engine without claiming a comparable
absolute score. The design does not bet everything on calibration succeeding — rank-ordering is the
defined safe backstop.

**What this RFC commits to:** The UHD schema declares `score` (`units`/`calibrated`/`transform`,
[Section 5](#5-uhd-schema)) so a consumer can invert the training transform and recover real TFLOPS,
and supports a score-only evaluation mode (rank/return best score without selecting-for-launch).

**What is deferred:** Delivering cross-engine comparison requires changes outside the UHD:

1. **A plugin-query surface** for the per-graph figure of merit — both the cheap **A** (engine
   performance estimate) and the accurate **B** (config UHD run in a "score only, don't launch" mode).
   This is an engine-plugin ABI addition, owned by the plugin SDK, not this RFC; it must also let a
   non-descriptor engine (MIOpen) report an A-level estimate through the same surface, or the policy
   falls back to today's static ordering for it.
2. **The two engine-selection policies** that consume it ([Section 12.2](#122-two-engine-selection-policies-rfc-0007)) —
   the quick policy (rank by A, drill into the winner's B) and the thorough policy (run every B, compare
   across engines). Both are squarely [RFC 0007](0007_EngineSelectionHeuristicsFramework.md)'s territory.
3. **Cross-engine calibration.** Comparing estimates across engines only works if the units are
   comparable and each model (A and B) is calibrated to real TFLOPS (not just monotonic for argmax).
   This is a real modeling requirement, not just plumbing — and the one most likely to force the
   rank-ordering fallback if it does not hold up.

This is a dedicated follow-up co-owned with [RFC 0007](0007_EngineSelectionHeuristicsFramework.md).

---

## 13. Observability

Because selection is data-driven, it must be inspectable — consistent with
[RFC 0017 §9](0017_UniversalKernelDescriptor.md#9-observability-and-diagnostics) and
[RFC 0007 §12](0007_EngineSelectionHeuristicsFramework.md#12-logging). The UHD path surfaces:

- **Selection trace:** Candidates, scores, winner, whether model or fallback decided
- **Model provenance:** UHD id, model artifact version, training provenance
- **Contract diagnostics:** Clear failure message on feature/model mismatch

---

## 14. Model Generation Pipeline

The UHD is only useful if producing one is automated — by tooling any package author can run, not a
provider-specific service.

### 14.1 Two-Stage Workflow

1. **Ship a working pack with a trivial heuristic.** The pack's UED names a `static_order` UHD (rank
   by `priority`/`id`). The pack is fully functional and model-free from day one; no benchmarking or
   training is needed to use it.
2. **Generate a real heuristic from on-hardware timings.** A standalone generation tool loads the pack,
   times its kernels across a corpus of problem shapes, trains a model, and emits an updated UED/UHD —
   same descriptor kind, now `adapter: tree_data` pointing at an exported model. Dropping that updated
   engine descriptor set back in upgrades the pack from trivial ordering to a trained heuristic in place.
   **OPEN:** See [Open Question 9](#operational) (shape corpus location).

Because the shipped and generated heuristics are the same descriptor kind differing only in `adapter`
and fields, the tool only rewrites data; it never introduces a new interface. The tool runs over
hipDNN's public API — it adds no code to hipDNN and touches no provider internals — so it works for
any provider's pack.

### 14.2 Benchmarking via hipDNN Autotune

The timing substrate is hipDNN's own autotune ([RFC 0013](0013_Autotune.md)), not a bespoke sweep.
Autotune is provider-agnostic: it times whatever engine/kernel actually runs, so it exercises a rocKE
pack exactly as it would any other engine. The generation tool drives it through the public Graph API:

- `get_engine_configs()` — enumerate the applicable candidates for a graph (the pack's child UKDs)
- `add_engine_variants()` / `add_engine_sweep()` — enroll every candidate as plan specs
- `autotune(mode = EXHAUSTIVE, strategy = RUN_UNTIL_STABLE)` — compile and time each candidate
- `AutotuneResult[]` — per-candidate `engineId`/config, `minTimeMs`/`avgTimeMs`, `workspaceSize`,
  persisted to JSON. That JSON is the training dataset.

Because the pack already contains its variant kernels as UKDs, the tool times the shipped kernels —
it does not re-enumerate or re-build a variant grid. The pack is the authority on which variants exist;
autotune is the authority on how fast each one runs.

### 14.3 The principle: one source of truth, translate once

The tool must guarantee the runtime contract matches what was benchmarked. It emits the pack's
descriptors (updated UED/UHD, and the KMD/`features_signature` if not already present) from the same run
that produced the timings — **generate-then-freeze** — so descriptor ⟷ dataset ⟷ runtime are consistent
by construction.

"Translate once" is really **two contracts** the tool freezes and emits:

1. **Feature contract.** Emit the UHD's `features_signature` ([Section 7.2](#72-the-features_signature))
   from the same feature definition training used (expressions over `$q.* / $kernel.* / $device.*`).
   Then **one generic extractor runs it on both sides** — offline for training, the in-tree evaluator
   for inference — both reading the *same* signature. No reimplementation, so no drift. (Caveat: the
   expression op set must cover the derived features — [Section 7.2](#72-the-features_signature)'s
   `log2`/`/`/`min`/`max` extension. Anything gnarlier needs the `custom_library` featurizer escape
   hatch.)
2. **Kernel-identity contract.** The candidate autotune timed (its `engineId`/config, i.e. a specific
   UKD) must map 1:1 to the **child UKD's identity** (its `metadata`) in the emitted pack, so the
   model's argmax over timed candidates maps exactly to argmax over UKDs at runtime. Autotune already
   reports the config per result, so the tool keys the dataset on the UKD it enrolled — the quieter
   drift risk, made explicit.

### 14.4 New stage: package (Stage P)

From one timing run ([§14.2](#142-benchmarking-via-hipdnn-autotune)) the tool trains **two** models
([Section 12.1](#121-two-heuristics)): the fine-grained **config UHD (B)** and the cheap **engine
performance estimate (A)**. A **package stage** then emits (or updates) the engine's descriptor set:

- the **config UHD (B)** — rewritten from the shipped `static_order` to `adapter: tree_data`, carrying
  the `features_signature` (referencing `$kernel.*` KMD fields + `$device.*` for arch-awareness),
  `features_hash`, `objective`/`score`, and `model.artifact` ([Section 5](#5-uhd-schema)); one per
  engine, arch-aware, so no artifact table;
- the **engine performance estimate (A)** — the coarse `f(graph) → expected perf` model the quick policy
  ranks engines by ([Section 12.2](#122-two-engine-selection-policies-rfc-0007)), also emitted as data on the UED (whether A is
  its own model or derived from B is the OPEN in [Section 12.1](#121-two-heuristics));
- the **model files as data** — the trained boosters exported to their model files (the `tree_data`
  format, read by the in-tree walker; [Section 8](#8-model-adapters)), each embedding the `features_hash`
  it was trained against; shipped with the engine descriptors, *not* compiled in;
- the **KMD** — the compilation-knob schema (`fields`: `tile_m`, `warp_n`, `split_k`, `dtype`, …), if
  the pack does not already carry one; its fields are exactly the `$kernel.*` metadata the UKDs already
  fill ([Section 4.2](#42-kmd-fields-and-knobs-as-a-view-onto-them)) — **one KMD per engine, owned by the UED**;
- the **UED** — updated to reference the new UHD and the A model; its user runtime `knobs` are distinct
  from the KMD compilation knobs and untouched by generation.

The UMDs, UDD, and the child UKDs (kernels) are **not** regenerated — only the heuristic side changes.
That is the whole point of the two-stage design: the expensive artifacts (compiled kernels) ship once;
both heuristics are layered on afterward as data.

### 14.5 Sweep space: grid vs. constraint

The generation tool sweeps two things: the **problem-shape corpus** (batch, seqlen, heads, … — supplied
by the author as representative shapes, or a per-op default) and optionally the **exposed knobs** (the
KMD fields the engine lets a user set, via `add_engine_variants` knob settings). Note a knob setting only
*filters* the catalog ([Section 4.2](#42-kmd-fields-and-knobs-as-a-view-onto-them)), so sweeping knobs
explores user-visible restrictions, not new kernels. The **variant space itself is fixed** — it is the pack's
existing child UKDs, so the tool does not enumerate or build variants; it enrolls the shipped ones and
times them ([Section 14.2](#142-benchmarking-via-hipdnn-autotune)).

One subtlety for anything that *drives* a sweep from a descriptor: a validity *constraint*
(`min:1, max:8`) expresses which values are **legal**, not which to **sample**, so a swept axis needs an
explicit `sweep_values` / grid hint, not an inferred range. **OPEN**: standardize where the shape corpus
and any runtime-knob grid live (a tool-side config vs. a descriptor field), so a heuristic can be
regenerated reproducibly without out-of-band inputs.

### 14.6 Auto-deriving a first-pass `features_signature`

Most of a `features_signature` can be **derived from what a package already carries**, so the tool can
propose a first pass rather than requiring an author to hand-write the feature list. The key is that
"map the graph to features" spans **two layers**, and only the first is shared/derivable:

- **Layer 1 — op-intrinsic vocabulary (per op, shared).** The fields that *exist and are bindable*:
  tensor/dim bindings (`$q.seqlen_q`), node attributes (`$sdpa_fwd.head_size`), device properties
  (`$device.*`). These are facts about the op, identical for every engine that implements it, and the
  UMD op-schema registry already emits them by reflection from the `.fbs` annotations
  ([Section 7](#7-feature-extraction), [PR #10341](https://github.com/ROCm/rocm-libraries/pull/10341)).
- **Layer 2 — the `features_signature` (per package/UHD, not shared).** *Which* of those fields a given
  model consumes, plus derived transforms. This is per-UHD: two packages of the same op may rank on
  different subsets.

The tool auto-derives Layer 1 and proposes a Layer-2 first pass from it, in three tiers:

| Tier | Feature kind | Derivable from | Author input needed |
|---|---|---|---|
| 1 | Raw fields (`$kernel.*` = KMD fields; `$q.*`/attrs = UMD bindings; `$device.*`) | KMD schema + UMD op-schema registry + device vocab | **none** |
| 2 | Generic transforms (logs, ratios) and **tile/wave quantization** | Tier 1 + a UHD `derived` block that pairs problem dims with `$kernel.*` tile axes | the UHD's **`derived` block** ([Section 7.4](#74-derived-values-the-uhd-derived-block)) |
| 3 | **Physics** — arithmetic intensity, roofline bound | the op's FLOP and byte formulas, consumed by a `derived` entry | **`umd_flops` / `umd_bytes`** op annotations (below) |

The tile/wave quantization (Tier 2) is not auto-inferable — the tool cannot guess that `seqlen_q` pairs
with `tile_m0` — but it is not homeless either: it lives in the UHD's **`derived` block**
([Section 7.4](#74-derived-values-the-uhd-derived-block)). The tool can emit a **stub `derived` block**
(the raw fields plus placeholders for the correspondences) for the author to complete, then reference
the results from the proposed `features_signature` — and, being on the UHD, both are regenerated together.

**Arithmetic intensity, and what authors provide for it.** Intensity is
`total_FLOPs / total_bytes_moved` (FLOP/byte) — the roofline x-axis that separates compute-bound from
memory-bound problems, which is exactly the split that decides which kernel wins. Both terms are
closed-form over the bound dims and dtype sizes, but the *formulas* are op-specific and cannot be
inferred from the KMD field list. So they are authored **once per op, at Layer 1**, as two table-level
attributes on the op's `.fbs` schema — the same annotation channel and codegen as `umd_opcode`:

```fbs
// data_types.fbs — declare once, alongside the existing umd_* attributes
attribute "umd_flops";   // JsonLogic expr over bound dims -> FLOP count
attribute "umd_bytes";   // JsonLogic expr over bound dims + per-tensor dtype size -> bytes moved

// sdpa_attributes.fbs — applied table-level, next to umd_opcode
table SdpaAttributes (
    umd_opcode: "sdpa_fwd",
    umd_flops:  "{\"*\":[4,\"$q.batch\",\"$q.num_heads\",\"$q.seqlen_q\",\"$k.seqlen_k\",\"$q.head_size\"]}",
    umd_bytes:  "..."   // sum of per-tensor (element_count * dtype_bytes) for Q, K, V, O
) { /* ... */ }
```

Because these are **op-intrinsic** (SDPA does `4·B·H·Sq·Sk·D` FLOPs regardless of engine or package),
they live at Layer 1 and are shared by every package of that op — one annotation per op-family, authored
once. The codegen then promotes intensity into the bindable vocabulary as a derived field (e.g.
`$sdpa_fwd.arithmetic_intensity`), which a `features_signature` references *identically* to a raw dim
like `$q.seqlen_q`. The Tier-3 "physics" distinction thus disappears at the point of use.

Caveats:

- **Only the two `.fbs` attributes are a new author ask.** The FLOP/byte formulas are the sole required
  input; everything else (Tier 1, and Tier 2 given the correspondence hint) is derived. `dtype`→byte
  size is already a known schema mapping, not a new ask.
- **Mixed-dtype ops** (e.g. fp8 in / fp16 accumulate, or differing I/O dtypes) make `umd_bytes` a sum
  over *per-tensor* dtype sizes, not one global `dtype_bytes`. The single expression still handles it,
  but it must reference each tensor's own dtype — a single-dtype shortcut is wrong for quantized kernels.
- **A native-predicate fallback** covers ops whose FLOP/byte count is not a clean closed form (ragged,
  data-dependent masking): a registered function resolved by name instead of an inline formula, via the
  custom-operation escape hatch ([Section 8](#8-model-adapters)).
- **Auto-derivation yields a *superset*.** Deriving every raw field and generic transform produces a
  bloated, noisy vector that can hurt a small-data model; the sweep's feature-importance (or a curated
  per-op template) prunes it. Auto-derivation proposes; data or a template trims.

**A data-free structural first pass.** Tiers 1–2 alone (no benchmarking) already support a real
model-free heuristic better than `static_order`: prefer the kernel whose tile best divides the problem
(minimize quantization waste), tie-break on an occupancy proxy. That is a legitimate `table`/rule UHD
computable from KMD + matcher bindings + device facts, and it is the zero-benchmark starting point before
the autotune-trained `tree_data` model replaces it.

---

## 15. Phased Delivery

Each phase is independently shippable and validated against the SDPA path and the reference tooling,
using the parity and overhead checks of [RFC 0017 §12.1](0017_UniversalKernelDescriptor.md#121-testing-and-performance).

| Phase | Deliverable | Notes |
|-------|-------------|-------|
| 1 | `static_order` baseline | UHD schema + KDP membership + deterministic ranking. Every pack gets a working, model-free selector. Proves UED→UHD→child-UKD wiring end to end. |
| 2 | `features_signature` + generic extractor | Inline signature, single extractor over shared namespaces, training↔runtime parity test, expression-op extension. |
| 3 | `tree_data` + in-tree walker | Default shipping path. LightGBM model exported to data, evaluated by in-tree GBDT walker (reusing MIOpen's `LgbmForest`-style parser). Lands the real FMHA-fwd model. Adds lazy load + model cache. |
| 4 | Generation tool | Standalone tool that drives autotune across a shape corpus, trains a model, emits updated UED/UHD. |
| 5 | `table` / CSV | Cheap bucketed heuristics for ops that don't warrant a model. |
| 6 | `custom_library` | Compiled scorer `.so` for models the in-tree walker doesn't cover. Dependency + trust audit gated. |
| 7 | Engine-selection integration | Score-only mode, plugin-query surface, engine-selection policy. Co-owned with [RFC 0007](0007_EngineSelectionHeuristicsFramework.md). |

`lgbm_to_c` (build-time, in-tree perf optimization only) and `lightgbm_native` are dependency-gated and
land only when a concrete need appears.

---

## 16. Risks

| Risk | Description | Mitigation |
|------|-------------|------------|
| **Feature-contract drift** | Training and inference feature vectors diverge | Single `features_signature` drives both sides via one generic extractor; three-part load-time check ([Section 7.3](#73-contract-enforcement)); fail-closed on mismatch |
| **Kernel-identity drift** | Timed candidate doesn't match emitted UKD | Tool keys dataset on UKD identity (`engineId`/config → `metadata`); load-time validation |
| **KMD↔UHD coupling** | a *breaking* KMD change (removed/reinterpreted field) invalidates the trained model; a training-coverage gap degrades ranking silently | KMD-version check at load, fail closed on incompatible pairing or dangling `features_signature` ref; additive changes need no retrain until exposed ([§4.3](#43-coupling-rules)); training-coverage warning in the trace |
| **Dependency creep** | Pressure to link `liblightgbm` at runtime | In-tree `tree_data` default; runtime deps stay opt-in only |
| **Bad/stale model** | Model picks worse than first-match | Fail-open to `static_order`; generic-vs-baseline parity gate; model provenance in trace |
| **Miscalibrated cross-engine scores** | Absolute score misleads engine selection | Train calibratable TFLOPS from start; fall back to rank-ordering at policy level if calibration unreliable |
| **Cache key incompleteness** | Result cache returns wrong kernel | Fingerprint must include problem + candidate set + device |
| **Drop-in trust** | Model artifact is author-controlled input | Bounded loader/evaluator; inherit [RFC 0017 §10](0017_UniversalKernelDescriptor.md#10-packaging-and-delivery) trust rules |

---

## 17. Open Questions

### Schema and Training

1. **Regressor vs. ranker.** The tooling trains a *regressor* on TFLOPS and argmaxes. A
   learning-to-rank objective (LambdaRank/NDCG) optimizes ordering directly and may pick better
   *within* an engine without needing calibrated absolute values. But a calibrated TFLOPS *regressor*
   is what makes the absolute, cross-comparable metric of [Section 12.3](#123-cross-engine-comparison)
   possible; a pure ranker forecloses that and leaves only the rank-ordering path. Recommendation:
   regressor, preserving the absolute option, with ranking as the fallback rather than the only mode.
   *(Impacts [Section 5](#5-uhd-schema), [Section 12.3](#123-cross-engine-comparison).)*

2. **Arch-aware model scope.** The UHD is one per engine, and the engine spans arches (`arch` is a KDP
   property), so the model is arch-aware via `$device.*` features — one model generalizing across the
   engine's arches. But the tooling historically trained per-`(op, arch, dtype)`; consolidating to one
   model per engine assumes device features capture the cross-arch differences well enough. If they
   don't, an engine can be scoped more narrowly (a UED per arch/dtype) so its single UHD stays within
   one arch. Decide against real cross-arch accuracy data.
   *(Impacts [Section 4](#4-ownership-model).)*

3. **`tree_data` artifact format.** A data-SDK FlatBuffer tree schema (recommended) — consistent with
   graph/device-props serialization, `Verifier`-gated, additive evolution, needs a convert step at
   build. Alternative: parse LightGBM `model.txt` at load — lowest author friction (no conversion) but
   a bespoke parser to write and harden against hostile input.
   *(Impacts [Section 8.1](#81-default-tree_data).)*

4. **Derived feature set.** Arithmetic intensity, tile quantization, aspect ratios, occupancy,
   padding-fit — are there others? Candidates: memory-footprint / working-set vs. cache and HBM
   capacity; a compute-vs-memory-bound flag from intensity vs. the device's roofline ridge point;
   wave-quantization *tail* (last-wave occupancy); K-splitting overhead for split-K variants.
   Enumerate the final set against real per-op sweeps before freezing.
   *(The expression-op question is resolved — the UMD RFC's JsonLogic operator set already covers the
   derived features.)* *(Impacts [Section 7.2](#72-the-features_signature).)*
   **Auto-derivation dependency:** the physics features (arithmetic intensity, roofline bound) need
   per-op **`umd_flops` / `umd_bytes`** `.fbs` annotations to be derivable ([Section 14.6](#146-auto-deriving-a-first-pass-features_signature)).
   Open: settle the two attributes (declaration + per-op application) and the mixed-dtype byte-formula
   convention. Tier-2 quantization (the dim↔tile correspondence) lives in the **`derived` block**, whose
   placement is a **discussion point** ([Section 7.4](#74-derived-values-the-uhd-derived-block)): on the
   **UHD** (chosen — self-contained, regenerated with the model) vs. the **UED** (shared across the
   engine's A/B heuristics and versions, but a cross-descriptor dependency for a dropped-in UHD).

### Structural

5. **Overlapping packs.** Under the KDP model, two *packs* can match one graph. Their child kernels
   are ranked by *different* UHDs whose scores are not comparable. Options: (a) forbid overlapping
   packs for one engine — extend the deterministic-arbitration duplicate-match check in
   [RFC 0017 §5](0017_UniversalKernelDescriptor.md#5-matching-and-the-umd) to overlapping-but-not-identical;
   (b) rank each pack's group by its own UHD, then compare winners by `priority` only; (c) require
   comparable `score.units` to compare across packs. Recommendation: (a) for v1 — packs for one engine
   should partition the graph space, not overlap.
   *(Impacts [Section 6](#6-selection-flow).)*

6. **Engine estimate (A) vs. config UHD (B).** Is A a distinct trained model, or derived from B as
   max predicted score over candidates? Distinct is cheaper for the quick policy (skips enumeration);
   derived is one fewer model to train.
   *(Impacts [Section 12.1](#121-two-heuristics).)*

7. **Non-descriptor engine estimates.** How does a non-descriptor engine (e.g. MIOpen) report an
   A-level estimate through the plugin-query surface? If it cannot, the quick policy falls back to
   static ordering for it — acceptable for v1, but limits performance-based engine ranking.
   *(Impacts [Section 12.2](#122-two-engine-selection-policies-rfc-0007).)*

### Operational

8. **Caching scope.** The in-process cache keyed on `(engine id, graph id, device id)` + inventory
   generation counter is sufficient for repeated graphs within a session. Persistent cross-run caching
   interacts with a future [RFC 0007](0007_EngineSelectionHeuristicsFramework.md) "cache selector"
   policy — defer until that policy is designed.
   *(Impacts [Section 10.2](#102-loading-and-caching).)*

9. **Shape corpus location.** The variant space is fixed (the pack's UKDs, timed via autotune), but
   the **shape corpus** and any **runtime-knob grid** need a home. A validity *constraint*
   (`min:1, max:8`) expresses which values are *legal*, not which to *sample*, so a swept axis needs
   an explicit `sweep_values` / grid hint, not an inferred range. Standardize: tool-side config (less
   coupled, easier to iterate) or descriptor field (reproducible from pack alone)?
   *(Impacts [Section 14.5](#145-sweep-space-grid-vs-constraint).)*

10. **Dependency + trust audit.** Needs deeper investigation: the exact allowed dependency surface
    for a shipped provider (license, distro packaging, ROCm image contents), and for `custom_library`
    the trust/signing rules for dropping in author-compiled native code. The former decides whether
    the in-tree tree-walker must be fully first-party or may vendor a third-party evaluator; the
    latter gates the `custom_library` drop-in path.
    *(Impacts [Section 8](#8-model-adapters), [Section 10.1](#101-dependencies).)*

---

## 18. Glossary

- **UHD (Universal Heuristic Descriptor):** One kernel-selection model, owned by the UED (one per
  engine), that ranks the applicable child UKDs of every pack joining its engine and picks one.
  Per-engine and arch-aware (takes `$device.*`).

- **KDP (Kernel Descriptor Pack):** The pack that joins an engine and adds kernels; names one matcher
  set, one UED (which carries the UHD and KMD), and one UDD over a vector of child UKDs. The selection
  group is a pack's child kernels; the selector and metadata schema come from the engine.

- **KMD (Kernel Metadata Descriptor):** [RFC 0017](0017_UniversalKernelDescriptor.md)'s explicit,
  upfront declaration of the engine's **compilation knobs** — the variant `fields` (`tile_m`, `split_k`,
  `dtype`, …) every kernel carries, each with a type and optional default. **One KMD per engine, owned
  by the UED**; each UKD's `metadata` fills it. It is the authoritative schema for the `$kernel.*` fields
  the UHD ranks on and the `features_signature` references
  ([Section 4.2](#42-kmd-fields-and-knobs-as-a-view-onto-them)).

- **UED (Universal Engine Descriptor):** The UED names one UHD and one KMD. They are coupled — the KMD
  is the feature space the UHD ranks over — so the engine owns both; a *breaking* KMD change requires
  retraining the UHD, while additive changes and dispatch-only fields do not
  ([Section 4.3](#43-coupling-rules)).

- **Knob:** A **KMD field the engine chooses to expose** to the user — a name in the UED's `knobs`, and
  nothing more (the KMD already declares the field's type and default). Only KMD fields can be knobs; a
  knob's legal values come from the *catalog* for this graph, and its **default is whatever the UHD ranks
  first**. Knobs *filter* the catalog; the UHD then ranks what survives
  ([Section 4.2](#42-kmd-fields-and-knobs-as-a-view-onto-them)).

- **KMD field space:** The engine's variant axes (`tile_m`, `warp_n`, `split_k`), filled per-kernel in
  UKD `metadata` — the space the UHD ranks over, read as `$kernel.*`. Each UKD is one point in it (and
  its unique catalog key); the KDP is the collection. Knobs are a user-facing *subset* of these fields,
  not a separate category ([Section 4.2](#42-kmd-fields-and-knobs-as-a-view-onto-them)).

- **Kernel-selection heuristic vs. engine-selection heuristic:** The two levels; the UHD is the former
  (which kernel within an engine), [RFC 0007](0007_EngineSelectionHeuristicsFramework.md) owns the
  latter (which engine).

- **`features_signature`:** The UHD's ordered, versioned list of model inputs (bare `$`-prefixed fields
  or derived expressions) that both training and inference consume through one generic extractor; the
  contract that must stay bit-identical across the two.

- **`tree_data`:** The default shipping path — a GBDT tree table shipped as data with the engine's
  descriptor set and evaluated by an in-tree GBDT walker; standalone drop-in, zero runtime dependency.

- **`custom_library`:** The escape hatch — a compiled scorer `.so` shipped with the engine, `dlopen`'d
  through a tiny C ABI; standalone, any model family.

- **Scorer / adapter:** The thing that turns a UHD's model content into a per-candidate score; reached
  through an adapter in build-and-runtime (default) or build-only delivery classes, mirroring
  [RFC 0017 §8](0017_UniversalKernelDescriptor.md#8-adapters-and-extensibility).

- **`lgbm_to_c`:** The tooling's build-only path that lowers a LightGBM booster to C linked into the
  provider. Kept only as an optional build-time perf optimization for in-tree AOT models — **not** a
  drop-in shipping mechanism ([Section 3.1](#31-existing-heuristic-generation-tooling)).

- **Score-only mode:** Running a UHD to obtain the best predicted score without selecting for launch;
  the hook for surfacing estimated TFLOPS to engine selection.

- **Stage P (package):** The pipeline stage that emits the engine descriptor set (UED/UHD/KMD +
  tree-table) from the same sweep that trained the model, enforcing the feature and kernel-identity
  contracts ([Section 14](#14-model-generation-pipeline)).

- **Engine estimate (A):** Cheap `f(graph) → expected performance` model for quick policy engine ranking.

- **Config UHD (B):** Full `f(graph) → best kernel + predicted performance` model for kernel selection
  and accurate cross-engine comparison.
