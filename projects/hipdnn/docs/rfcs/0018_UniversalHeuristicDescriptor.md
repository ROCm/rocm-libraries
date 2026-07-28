# RFC 0018: Universal Heuristic Descriptor (UHD): Data-Driven Kernel Selection

- Contributors: (draft — jascampb)
- Status: First draft, for discussion
- Parent: [RFC 0017 Universal Kernel Descriptor](0017_UniversalKernelDescriptor.md) (this is the "UHD + kernel selection" follow-up named in [RFC 0017 §12.2](0017_UniversalKernelDescriptor.md#122-follow-up-rfcs))
- Related: [RFC 0007 Engine Selection and Heuristics Framework](0007_EngineSelectionHeuristicsFramework.md), [RFC 0013 Autotune](0013_Autotune.md) (the benchmarking substrate for heuristic generation, [Section 13](#13-model-generation-pipeline))

> **Draft note.** This is a first pass to frame the design and drive discussion, not a finished
> spec. Sections marked **OPEN** carry decisions we still need to make. It is grounded in the existing
> heuristic-generation tooling — a training-and-export pipeline any package author can run to produce a
> heuristic for their own kernels — and rocKE's current selection path, both summarized in
> [Section 3](#3-prior-art-what-we-already-have).

## Table of Contents

1. [Overview](#1-overview)
2. [Scope and the Two Selection Levels](#2-scope-and-the-two-selection-levels)
3. [Prior Art: What We Already Have](#3-prior-art-what-we-already-have)
4. [UHD Schema](#4-uhd-schema)
5. [How a UHD Ranks Matched Kernels](#5-how-a-uhd-ranks-matched-kernels)
6. [Selection-Group Membership: UED, KMD, UKD, and KDP](#6-selection-group-membership-ued-kmd-ukd-and-kdp)
7. [Feature Extraction and Binding](#7-feature-extraction-and-binding)
8. [Model Formats and the Adapter Seam](#8-model-formats-and-the-adapter-seam)
9. [Dependencies](#9-dependencies)
10. [Performance: Loading, Caching, Lazy Evaluation](#10-performance-loading-caching-lazy-evaluation)
11. [Engine Selection Interplay and Estimated TFLOPS](#11-engine-selection-interplay-and-estimated-tflops)
12. [Observability](#12-observability)
13. [Model Generation Pipeline](#13-model-generation-pipeline)
14. [Phased Delivery](#14-phased-delivery)
15. [Risks](#15-risks)
16. [Open Questions](#16-open-questions)
17. [Glossary](#17-glossary)

---

## 1. Overview

A **UHD (Universal Heuristic Descriptor)** is one kernel-selection model. Given many kernels that all
apply to a graph (the applicable child UKDs of a matched pack), the UHD ranks them and picks the one
predicted best for the concrete problem. It is the data form of the ranking logic that today lives,
when it exists at all, hand-coded inside an engine's dispatcher.

Like the other shared descriptors in [RFC 0017](0017_UniversalKernelDescriptor.md), a UHD is an
**individual, reusable descriptor referenced by ID** (a GUID). Under the reworked
[RFC 0017 §4](0017_UniversalKernelDescriptor.md#4-descriptor-formats), **the UED (engine) owns the
UHD**: a UED names **one UHD and one KMD** (its heuristic and its metadata schema). The **KDP** is the
cohesive pack that binds a matcher set, **one UED** (the engine — which carries the UHD and KMD), and
one UDD, over a *vector of child UKDs*; a child UKD is just its `kernel_source` + `metadata` and names
none of them. So the UHD is now **one per engine, owned by the UED, not referenced by the KDP or each
UKD** — it ranks the applicable child kernels of every pack that joins its engine. Many KDPs may name
the same engine, and thus share its one UHD.

The **KMD (Kernel Metadata Descriptor)** matters directly to the UHD, and that is exactly why
[RFC 0017](0017_UniversalKernelDescriptor.md) puts **both under the UED**. The KMD is the explicit,
upfront declaration of **every compilation knob (variant field) the engine's kernels carry** — tile
size, block size, split-K, dtype, and the like, each with a type and optional default. Each UKD fills in
concrete values (`metadata`), the UHD **ranks the catalog on those values**, and both matchers and the
UHD's feature signature read them as `$kernel.<field>`. Because the KMD *is* the feature space the UHD
ranks over, the two are coupled — mutate the KMD and you must retrain the UHD — so the engine owns both
([Section 6](#6-selection-group-membership-ued-kmd-ukd-and-kdp)).

This RFC covers three things and calls out a fourth:

1. The **UHD schema** — how a selection model is described as data ([Section 4](#4-uhd-schema)).
2. **How a UHD ranks and chooses** among a pack's matched kernels ([Section 5](#5-how-a-uhd-ranks-matched-kernels)).
3. **Selection-group membership** — how a kernel joins a selection group (via its KDP), and how UED
   knobs, the KMD variant fields, UKD metadata, and the KDP collection relate
   ([Section 6](#6-selection-group-membership-ued-kmd-ukd-and-kdp)).
4. **The engine-selection relationship** — how kernel-level selection fits under hipDNN's existing
   engine-selection heuristic ([RFC 0007](0007_EngineSelectionHeuristicsFramework.md)): applicability
   **bubbles up** before engines are ranked; the tooling produces **two** heuristics (a cheap engine-level
   performance estimate and the fine-grained config UHD); and **two** policies (quick vs. thorough)
   consume them, with a rank-ordering fallback ([Section 11](#11-engine-selection-interplay-and-estimated-tflops)).
   The policy/ABI changes stay a coordinated [RFC 0007](0007_EngineSelectionHeuristicsFramework.md)
   follow-up; this RFC just supplies the heuristics and keeps the schema from foreclosing them.

It also describes the **automated pipeline** that benchmarks the compile-time build-config (variant)
space and emits the model plus the pack's descriptors from one source of truth
([Section 13](#13-model-generation-pipeline)).

---

## 2. Scope and the Two Selection Levels

[RFC 0017 §2](0017_UniversalKernelDescriptor.md#2-the-descriptors) already names the two selection
levels, and this RFC lives entirely at the lower one. They must not be conflated:

| Level | Question it answers | Owned by | Descriptor |
|---|---|---|---|
| **Engine-selection heuristic** | *Which engine* handles this graph (rocKE vs. MIOpen vs. …)? | hipDNN backend, the outer policy loop of [RFC 0007](0007_EngineSelectionHeuristicsFramework.md) | (a policy plugin / built-in; not a UHD) |
| **Kernel-selection heuristic** | *Which kernel within the chosen engine* runs this problem? | The generic provider, inside one engine | **UHD** |

A UHD is the **kernel-selection heuristic**. It is part of the generic provider that
[RFC 0017](0017_UniversalKernelDescriptor.md) introduces; it is **not** a new host interface and
**not** a policy plugin under [RFC 0007](0007_EngineSelectionHeuristicsFramework.md). But the two levels
are **not** cleanly one-after-the-other, and this is a correction to earlier drafts:

- **Applicability bubbles up first.** Before engine selection can rank anything, it must know which
  engines even apply. For a descriptor engine, "do I apply?" is the **matcher (UMD) pass** at the
  descriptor layer; that result **bubbles up** so non-viable engines are ruled out *before* the first
  plugin-policy layer ranks the survivors. So the descriptor/UHD layer runs (at least for applicability)
  *ahead of* engine selection, not strictly after it.
- **The UHD's predictions feed engine selection.** The generation tooling produces a cheap engine-level
  **expected-performance** estimate and the fine-grained config UHD; engine-selection policies consume
  those to rank engines *by predicted performance* ([Section 11](#11-engine-selection-interplay-and-estimated-tflops)).
  The policies themselves remain [RFC 0007](0007_EngineSelectionHeuristicsFramework.md)'s territory; this
  RFC provides the heuristics they consult.

**In scope:** the UHD schema; ranking semantics over a pack's matched UKDs; selection-group membership
(the UED owns the UHD and KMD; a KDP joins the engine) and how UED runtime knobs / KMD compilation
knobs / UKD metadata relate; the feature
contract; model formats and their
adapter seam; dependency constraints; load/eval performance; the model-generation pipeline.

**Out of scope (this RFC):** the engine-selection outer loop itself ([RFC 0007](0007_EngineSelectionHeuristicsFramework.md)
owns it); autotuning / exhaustive search (device-access tuning is [RFC 0013](0013_Autotune.md)); the
matcher and launch machinery ([RFC 0017 §5–6](0017_UniversalKernelDescriptor.md#5-matching-and-the-umd)).

---

## 3. Prior Art: What We Already Have

Two concrete systems anchor this design: the existing **heuristic-generation tooling** and rocKE's
selection path. The UHD generalizes what that tooling already produces onto the descriptor model, so
that **any package author can generate a heuristic for their own pack** with the same tools rather than
hand-writing selection code.

### 3.1 The heuristic-generation tooling

An end-to-end pipeline for turning benchmarks into a LightGBM kernel-selection model already exists as
**reusable tooling** (training scripts, exporters, and a dispatcher path), first exercised on
SDPA/FMHA forward. It is not a one-off: the same tools are meant to be run by any package author to
produce a heuristic for their kernels. Its shape informs almost every decision below:

- **Offline training.** A sweep step produces a training dataset (problem × kernel × measured TFLOPS);
  a training step fits a LightGBM regressor on `log1p(tflops)` with grouped cross-validation.
- **Inference was originally AOT-compiled to plain C** (an exporter lowers a trained booster to a
  dependency-free C scoring function, statically linked into the provider). This gave zero runtime
  dependency but is **not drop-in** — adding a model meant recompiling. This RFC replaces that shipping
  path with model-as-data ([Section 8](#8-model-formats-and-the-adapter-seam)); the exporter survives
  only as an optional build-time optimization.
- **Model registry** keyed by `(op, arch, dtype)`, generated from the trained models, mapping a problem
  to its scoring function.
- **Feature contract is generated for bit-parity.** A fixed feature vector (problem dims, dtypes, tile
  and warp constants, hardware props) is generated from one feature specification so the Python
  training features and the C++ inference features are identical; a round-trip test gates drift.
- **Selection** in the dispatcher: featurize the problem, look up the model, **score every satisfying
  candidate, argmax**, stable-order tie-break, and **fall back to first-match** if there is no model or
  the feature count disagrees.

Two lessons carry directly into the UHD design: **(a)** shipping the model as data (not linked into the
provider) is what makes a heuristic drop-in — the compiled-C path gave zero dependency but forced a
recompile per model; **(b)** the feature vector is the fragile contract, and generating it from one
specification is what keeps training and inference honest. The pipeline that produces all this is
described in [Section 13](#13-model-generation-pipeline).

### 3.2 rocKE selection today

rocKE (`dnn-providers/hip-kernel-provider/rocke/`) is a hipDNN **engine plugin**. Its selection path
is deterministic and catalog-driven: `AotCatalog::candidatesFor(op, arch)` → `satisfies(instance,
problem, attrs)` exact-match filter → **first match wins**, with an explicit
`TODO(heuristics): tie-break with trained per-arch FMHA model when >1 instances match`. That TODO is
exactly the seam a UHD fills. The normalized `SdpaProblem` (shape, dtype, layout, mask/dropout/alibi
attributes, arch) is the feature source already present at the selection point, and each `AotInstance`
carries the `CompileSpec` (tile/block/dtype/layout constants) that becomes a UKD's `metadata`. rocKE's
selection is **internal to the engine** and orthogonal to
[RFC 0007](0007_EngineSelectionHeuristicsFramework.md): the UHD ranks *within* rocKE, after the
catalog match — in [RFC 0017](0017_UniversalKernelDescriptor.md) terms, after a pack's matcher set
passes, the engine's UHD ranks its surviving child kernels.

---

## 4. UHD Schema

A UHD is a small, reusable **scoring recipe**, **owned by the UED — one per engine**
([Section 6](#6-selection-group-membership-ued-kmd-ukd-and-kdp)). It names one **`adapter`** (how to
rank), an ordered **`features_signature`** (the model's inputs), an objective, and — for model adapters
— its model **artifact**. Because a UED spans arches (arch is a KDP property in
[RFC 0017](0017_UniversalKernelDescriptor.md)), a UHD is **per-engine and arch-aware**: one model,
taking `$device.*` features so it generalizes across the arches its engine serves — *not* one model per
arch. Examples elide the `schema`/`id`/`name` plumbing; `id`s are GUIDs.

`adapter` is a **single discriminant** — it subsumes [RFC 0017](0017_UniversalKernelDescriptor.md)'s
illustrative `kind` + `model.framework` into one field ( `tree_data` ≈ `kind:model, framework:lightgbm`
shipped as data) — and the body is an **adapter-keyed union**.

```jsonc
// tree_data — the default; a GBDT tree table, shipped as data with the engine's descriptor set
{
  "schema":  "hipdnn.uhd/v1",
  "id":      "ae896b07-80cd-473c-b3f4-6a8892998519",   // GUID; referenced by the UED (one per engine)
  "name":    "rocKE FMHA fwd selector",                // per-engine, arch-aware — not per-arch
  "adapter": "tree_data",         // the ranking mechanism (Section 8)

  // ordered model inputs, bound like a UDD args_signature; order + form must match training (Section 7)
  "features_signature": [
    "$device.cu_count", "$device.lds_size",            // device props → arch-aware
    "$kernel.tile_m", "$kernel.split_k",               // KMD fields (compilation knobs)
    "$sdpa_fwd.head_size", "$q.seqlen_q",              // graph node attr + tensor dim
    {"*": ["$q.batch", "$q.num_heads"]}                // a derived feature (expression)
  ],
  "features_hash": "sha256:…",     // fail-closed input-contract guard (Section 7.3)

  "objective": "max",              // higher predicted score wins
  "score":     {"units": "tflops", "calibrated": true, "transform": "log1p"},  // recover TFLOPS → Section 11

  "model":     {"artifact": "fmha_fwd/model.bin"}   // ships as data with the engine descriptors (Section 8)
}
```

Other adapters keep the same head and vary the body:

```jsonc
// onnx — same shape, different adapter; dependency-gated (Section 8/9)
{ …, "adapter": "onnx", "model": {"artifact": "fmha_fwd/model.onnx"} }

// static_order — no features, no model, no hash
{ "schema": "hipdnn.uhd/v1", "id": "…", "name": "…", "adapter": "static_order",
  "order": ["priority", "id"] }              // a fixed precedence, or an explicit metadata sort key

// custom_library — native scorer; features_hash advisory if it self-features
{ …, "adapter": "custom_library",
  "features_signature": [ … ], "features_hash": "…",
  "model": {"symbol": "vendor.fmha_scorer", "config": { … }} }   // symbol + typed config, never inline code
```

The `adapter` values, in rough order of increasing capability (full table with delivery classes and
dependencies in [Section 8](#8-model-formats-and-the-adapter-seam)):

| `adapter` | What it is | Ranking | Model artifact |
|---|---|---|---|
| **`static_order`** | A fixed precedence with no learned model | Declared order / UKD `priority` | none |
| **`table`** | A CSV/lookup keyed by coarse problem buckets | Table lookup, then tie-break | with engine |
| **`tree_data`** | A GBDT tree table (LightGBM/XGBoost), in-tree walker — **default** | Score each candidate, argmax | with engine |
| **`onnx`** | An ONNX graph via a gated runtime | Score each candidate, argmax | with engine |
| **`custom_library`** | A registry-resolved native scorer behind a small C API (escape hatch) | Whatever the library returns | `.so` with engine |

`static_order` is the trivial baseline (and a safe default when no model ships). `tree_data` is the
tooling's model generalized to ship-as-data. Adding a new ranker (a static list, ONNX, a new model
family) is **one more `adapter` value** — the single discriminant is what makes that additive.
`custom_library` is the escape hatch, mirroring the native-predicate / custom-plan escape hatches in
[RFC 0017](0017_UniversalKernelDescriptor.md#5-matching-and-the-umd): the descriptor carries only a
symbol name + typed config, never inline code, resolved from the provider-internal registry.

**The model ships as data with the engine, not linked into the provider.** For every model adapter,
`model.artifact` is a path resolved relative to the engine's descriptor set (the UED + its UHD + KMD +
model), which is itself standalone-droppable ([Section 8](#8-model-formats-and-the-adapter-seam)). The
model is **per-engine** (owned by the UED), so there is no `(arch,dtype)→artifact` table and no per-pack
model entry — the single arch-aware model serves every pack that joins the engine. This is a change from
the earlier per-arch / pack-supplied-entry draft, forced by the UED now owning the UHD.

**OPEN — regressor vs. ranker.** The tooling trains a *regressor* on TFLOPS and argmaxes. A
learning-to-rank objective (LambdaRank/NDCG) optimizes ordering directly and may pick better *within*
an engine without needing calibrated absolute values. But a calibrated TFLOPS *regressor* is what makes
the **absolute, cross-comparable metric** of [Section 11](#11-engine-selection-interplay-and-estimated-tflops)
possible; a pure ranker forecloses that and leaves only the rank-ordering path. We likely want the
regressor to preserve the absolute option, keeping ranking as the fallback rather than the only mode.
Decide per-UHD via `objective` / `score`, or standardize.

---

## 5. How a UHD Ranks Matched Kernels

The generic engine produces the applicable candidates for a graph as follows: a KDP's shared **matcher
set** passes for the graph ([RFC 0017 §5](0017_UniversalKernelDescriptor.md#5-matching-and-the-umd)),
and its **child UKDs** are the candidates, each carrying its build `metadata`. The pack joins an engine
(UED), and that engine owns the one UHD, so the candidate set and its selector arrive together. Kernel
selection then:

1. **Take the pack's candidate set.** The matched pack yields its child UKDs, and its engine yields the
   one UHD; there is no per-UKD heuristic to group by. If two *different packs* match the same graph,
   see the OPEN note below.
2. **Extract features once.** Build the feature vector for the problem from the bound match variables
   and device properties ([Section 7](#7-feature-extraction-and-binding)). Per-candidate features come
   from each UKD's `metadata` (its compile-time build config); problem/device features are shared across the set.
3. **Score each candidate.** Invoke the UHD's scorer per candidate. For a model adapter this is one
   inference call per candidate over its feature row.
4. **Choose by objective.** `max` (or `min`) over the scores; the winner is the selected kernel.
5. **Tie-break deterministically.** On equal scores (or when the UHD declines / is absent), fall
   through to explicit UKD `priority`, then stable `id` — the same deterministic arbitration
   [RFC 0017 §5](0017_UniversalKernelDescriptor.md#5-matching-and-the-umd) already defines. Declaration
   order is never used.
6. **Fail open to a safe default.** If no model loads, the feature contract mismatches, or the scorer
   errors, selection degrades to `static_order` (priority + id). This mirrors the tooling's
   first-match fallback and keeps a bad/absent model from breaking execution.

The winner is a single UKD, which then dispatches through the pack's one UDD
([RFC 0017 §6](0017_UniversalKernelDescriptor.md#6-dispatch-and-workspace)). A UHD **only ranks**;
it never launches, mutates the graph, or touches device memory (read-only, same contract spirit as
[RFC 0007 §9](0007_EngineSelectionHeuristicsFramework.md#9-policy-plugins-and-the-outer-loop)).

**OPEN — multiple matching packs (was: mixed-heuristic sets).** Under the KDP model the per-UKD
mixed-heuristic case disappears (a pack has exactly one UHD). What remains is two *packs* matching one
graph. Their child kernels are ranked by *different* UHDs whose scores are not comparable. Options:
(a) forbid overlapping packs for one engine — the deterministic-arbitration duplicate-match check in
[RFC 0017 §5](0017_UniversalKernelDescriptor.md#5-matching-and-the-umd) already flags identical
criteria at load; extend it to overlapping-but-not-identical; (b) rank each pack's group by its own
UHD, then compare winners by `priority` only; (c) require comparable `score.units` to compare across
packs (ties into the calibrated-TFLOPS discussion of
[Section 11](#11-engine-selection-interplay-and-estimated-tflops)). Recommend (a) for v1 — packs for
one engine should partition the graph space, not overlap.

---

## 6. Selection-Group Membership: UED, KMD, UKD, and KDP

Under the reworked [RFC 0017 §4](0017_UniversalKernelDescriptor.md#4-descriptor-formats), membership is
no longer a field on the UKD, and the **UHD and KMD now live on the UED (engine), not the KDP**. A child
UKD carries only `kernel_source` + `metadata`; the **UED** names its one UHD (selector) and one KMD
(metadata schema); and the **KDP** joins that engine and adds a matcher set, a UDD, and the kernel
vector. So a kernel joins a selection group **by being a child of a pack whose `engine` carries that
UHD**:

```jsonc
// The UED owns the selector + metadata schema; the KDP joins the engine and adds kernels.
{
  "schema": "hipdnn.ued/v1",
  "id":        "efc9eae4-…",        // engine identity
  "heuristic": "ae896b07-…",        // UHD: the selector for this engine's kernels   <-- membership
  "metadata":  "9ae0b215-…",        // KMD: the variant-field schema this engine's kernels fill
  "knobs":     [ /* user runtime knobs: split_k, use_atomics */ ]
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

### 6.1 Two distinct knob concepts: UED runtime knobs vs. KMD compilation knobs

There are **two separate things** that both get loosely called "knobs," and keeping them apart is the
crux of this section. [RFC 0017](0017_UniversalKernelDescriptor.md) now gives each its own descriptor,
so the distinction is explicit in the data, not just in prose:

| Concept | Declared in | Filled in | Meaning | Role in selection |
|---|---|---|---|---|
| **Runtime knobs** | **UED** `knobs` | set by the user at plan time | **User-exposed runtime** params (`split_k`, `use_atomics`), with `min/max`/`one_of` domains — the [RFC 0017 §4](0017_UniversalKernelDescriptor.md#4-descriptor-formats) / [RFC 0004](0004_EngineConfigKnobs.md) meaning | Largely **orthogonal**: user-facing tuning, not the ranking axes |
| **Compilation knobs (variant fields)** | **KMD** `fields` | each UKD's `metadata` | The constants a kernel was **built with** (`tile_m`, `warp_n`, `pipeline`, `dtype`, `head_size`), each with a type and optional default | **This is what the heuristic ranks over** — read as `$kernel.*` |

So the correction to the earlier draft: **UED knobs are *not* the heuristic's feature axes.** The
per-candidate features the UHD ranks on are the **compilation knobs** — now explicitly declared once by
the pack's **KMD** and filled per-kernel in each UKD's `metadata`, surfaced as `$kernel.*`
([Section 7.1](#71-feature-sources)). The UED's `knobs` remain what
[RFC 0017](0017_UniversalKernelDescriptor.md) always meant: a small set of user-controllable runtime
parameters, mostly beside the selection problem.

**The KMD is the explicit schema for `$kernel.*`, and the UED owns both it and the UHD.** Earlier drafts
said "there is no knob schema; the set of fields is only implied by which `$kernel.*` the features
signature references." That gap is now closed: the **KMD declares all the compilation knobs upfront** (name, type,
default), the **UED references one KMD and one UHD**, and every child UKD's `metadata` fills exactly those
fields, validated against the KMD at load. Putting both on the UED is deliberate — the KMD *is* the
feature space the UHD ranks over, so they are **coupled: mutate the KMD and you must retrain the UHD**
(a field the model was not trained on is not selected against until it learns it). Co-owning them on the
engine keeps that invariant in one place. This gives the UHD a firm, checkable contract for its
per-candidate inputs:

- **A UKD is one point in the KMD-declared compilation-knob space** — its `metadata`. ✅
- **The collection of those points is the KDP** (a pack joining one engine, adding a matcher set, a UDD,
  and the kernel vector); the **UHD and KMD belong to the UED (engine)**, shared by every pack that joins
  it. `arch` is a KDP property, so one engine — and its one UHD/KMD — spans arches
  ([Section 6.3](#63-membership-rules)).
- **The UHD's `features_signature` `$kernel.*` references must be a subset of the KMD fields** — a
  load-time check ([Section 7.3](#73-bit-parity-between-training-and-inference)). The KMD is the
  authority on *what fields exist*; the `features_signature` picks *which* it uses and how it derives from them.

**The `split_k` boundary case.** A parameter can legitimately be *either* kind depending on the kernel:
if a split-K variant is a **separate compiled kernel**, `split_k` is a **KMD field** (a compilation
knob, filled per-UKD, a distinct kernel); if the launcher applies it **at dispatch**, it is a **UED
runtime knob**. [RFC 0017](0017_UniversalKernelDescriptor.md)'s own examples show `split_k` in *both* a
KMD `fields` list and a UED `knobs` list, so the boundary is real — decide per-op, and don't let it be
silently both, or a value swept as a build variant but treated as a runtime knob (or vice-versa) is a
real drift bug ([Section 13.2](#132-the-principle-one-source-of-truth-translate-once)).

### 6.2 Derived features the heuristic needs

Beyond the raw build-config fields, the model needs **derived features** — computed from
`{problem dims, UKD metadata, device props}`, not stored anywhere. From the tooling's feature engine,
the load-bearing ones are:

- **Arithmetic / algorithmic intensity** (FLOPs ÷ bytes) — the single most important derived feature.
- **Tile/wave quantization** — `num_tiles_*`, `total_output_tiles`, `tile_efficiency`
  (problem-vs-grid remainder waste). In the GEMM sweep this family is as predictive as intensity.
- **Aspect ratios** — `M/N`, `M/K`, `N/K` (shape skew).
- **Occupancy proxy** — `lds_usage_ratio` (and register pressure if available) → waves/CU.
- **Padding-fit** — `needs_padding_*` / `has_padding_when_needed_*` (problem × kernel padding
  interaction).

These belong in the **UHD's `features_signature` as expressions** ([Section 7.2](#72-the-features_signature)),
computed by the shared interpreter — not on the UED and not as extra stored `metadata`. The feature
contract owns the derived set. **OPEN — are there others we need?** Candidates: memory-footprint /
working-set vs. cache and HBM capacity; a compute-vs-memory-bound flag from intensity vs. the device's
roofline ridge point; wave-quantization *tail* (last-wave occupancy); K-splitting overhead for split-K
variants. Enumerate the final set against real per-op sweeps before freezing
([Section 13](#13-model-generation-pipeline)).

### 6.3 Membership rules

- **Many UKDs → one UHD, via the engine.** A whole FMHA-forward family is the child vector of one or
  more packs that all join the same engine; the engine's one UHD ranks them. The UHD is owned by the UED
  and shared across those packs, never inlined per kernel.
- **A UHD is scoped to comparable kernels.** All child UKDs joining one engine should be mutually
  substitutable for the graphs they co-match (same op family), because the model is trained to rank
  exactly that catalog. Validated at build where possible.
- **No UHD ⇒ static ordering.** An engine may name a `static_order` UHD (or the provider supplies a
  default one); its kernels are then ranked by `priority`/`id` only. Useful before a model exists.
- **Model coverage: per-engine, arch-aware.** The UHD is **one per engine**, and because the engine
  spans arches (`arch` is a KDP property), the model is **arch-aware via `$device.*` features** — one
  model generalizing across the engine's arches, referenced by the UHD's `model.artifact` and shipped as
  data with the engine's descriptor set ([Section 4](#4-uhd-schema)). This *reverses* the earlier
  per-arch, pack-supplied-entry draft (and the "no table because the pack pins the arch" reasoning):
  with the UHD owned by the UED, there is one arch-aware model, not one per arch. **OPEN — scope of the
  arch-aware model:** the tooling trained per-`(op, arch, dtype)`; consolidating to one model per
  engine assumes device features capture the cross-arch differences well enough. If they don't, an
  engine can be scoped more narrowly (a UED per arch/dtype) so its single UHD stays within one arch —
  decide against real cross-arch accuracy ([Section 13](#13-model-generation-pipeline)).

---

## 7. Feature Extraction and Binding

The feature vector is the contract between training and inference, and the fragile part of the whole
system. Generalizing it is the core hard problem of this RFC.

### 7.1 Feature sources

A feature row is assembled from three sources, all already available at plan time, and — importantly —
all drawn from the **same field namespaces the reworked [RFC 0017 §5](0017_UniversalKernelDescriptor.md#5-matching-and-the-umd)
criteria already read** (`$q.*`, `$graph.*`, `$<node>.*`, `$kernel.*`, `$device.*`, referenced bare, no
`var` wrapper). The UHD's `features_signature` is just another consumer of that vocabulary, so matching,
launch, *and* selection share one binding:

- **Problem features** — dims, dtypes, stride order, and op attributes **bound by the matcher set**.
  The matcher binds `$q.seqlen_q`, `$sdpa_fwd.head_size`, etc.; the featurizer reads them by name.
- **Device features** — `$device.*`: arch, CU count, clock, LDS size, HBM. In the generic provider
  these come from the same device-facts path
  [RFC 0007 §6](0007_EngineSelectionHeuristicsFramework.md#6-device-properties) defines
  (`DeviceProperties`, serialized), rather than each heuristic probing HIP.
- **Per-candidate (kernel) features** — `$kernel.*`: a UKD's `metadata`, i.e. the **compilation knobs
  the pack's KMD declares** (tile sizes, `num_warps`, split factors — [Section 6.1](#61-two-distinct-knob-concepts-ued-runtime-knobs-vs-kmd-compilation-knobs)).
  This is exactly the [RFC 0017 §5](0017_UniversalKernelDescriptor.md#5-matching-and-the-umd)
  `$kernel.*` token set — the *same* metadata the criteria read and the heuristic ranks on. The KMD is
  the authoritative list of which `$kernel.*` fields exist; these distinguish candidates within a pack
  and are what make argmax meaningful.

### 7.2 The `features_signature`

The feature contract is the UHD's inline **`features_signature`** ([RFC 0017 §4](0017_UniversalKernelDescriptor.md#4-descriptor-formats),
[Section 4](#4-uhd-schema)) — the tooling's `feature_spec.json` folded into the descriptor. It is an
**ordered list of model inputs**, bound the same way a UDD's `args_signature` binds kernel arguments:
each entry is either a **direct field** (a bare `$`-prefixed reference) or a **derived expression** over
those fields, using the reworked [RFC 0017 §5–6](0017_UniversalKernelDescriptor.md#5-matching-and-the-umd)
expression tree (`{"op": [args]}`, leaves are literals or bare `$` refs — no `var` wrapper). Order and
form must match training exactly, so the vector the provider assembles is what the model expects.

```jsonc
"features_signature": [
  // --- direct fields (shared namespaces) ---
  "$q.seqlen_q",                                   // graph tensor dim
  "$sdpa_fwd.head_size",                           // graph node attribute
  "$device.cu_count",                              // device property (arch-aware)
  "$kernel.tile_m",                                // KMD field (per-candidate)

  // --- derived features (Section 6.2), computed by the shared interpreter ---
  {"log2": ["$q.seqlen_q"]},
  {"/":    ["$q.seqlen_q", "$k.seqlen_k"]},                          // aspect ratio
  {"ceil_div": ["$q.seqlen_q", "$kernel.tile_m"]},                  // num_tiles_m
  {"/": [{"*": ["$q.seqlen_q", "$k.seqlen_k", "$sdpa_fwd.head_size"]}, "$q.bytes"]}  // ~arithmetic intensity
]
```

The vocabulary reuses the [RFC 0017](0017_UniversalKernelDescriptor.md) field namespaces and expression
tree wholesale — no new evaluator, and by construction a feature can only reference fields the match
produces. `$device.*` and `$kernel.*` are already in the reworked §5 namespace list, so no new source
kinds are needed either.

**Required expression-op extensions.** The reworked [RFC 0017 §5](0017_UniversalKernelDescriptor.md#5-matching-and-the-umd)
interpreter currently lists `and/or/! · comparisons · in · all · + · * · %` plus custom `divisible` /
`ceil_div` / `rsqrt`. The derived-feature families of [Section 6.2](#62-derived-features-the-heuristic-needs)
need a few arithmetic ops the criteria interpreter does not yet include:

| Op | Needed for |
|---|---|
| `log2` / `log` | log-scale sizes (`log2_M/N/K`) — the tooling's core scale-invariance features |
| `/` (true division) | aspect ratios, intensity, efficiency ratios |
| `min` / `max` | efficiency clamps, last-wave/tail quantization, roofline ridge comparisons |
| `-` (subtraction) | remainders / padding-fit terms |

These are pure, total (with a defined divide-by-zero / log-of-nonpositive policy — fail-closed to a
sentinel, per the safe-interpreter contract) additions to the **same** interpreter, so criteria and the
`features_signature` stay on one evaluator. List them as a required extension in the UMD/expression
follow-up so the signature can express intensity / aspect-ratio / quantization without a native escape
hatch. Anything beyond this closed set (e.g. a bespoke occupancy model) uses the `custom_library` escape
hatch ([Section 8](#8-model-formats-and-the-adapter-seam)) rather than growing the interpreter unbounded.

### 7.3 Bit-parity between training and inference

The tooling **code-generates** the C++ featurizer to guarantee the training and inference vectors are
identical, and gates it with a round-trip test. We keep that principle but make the UHD's
`features_signature` the single source of truth for **both** sides:

- **Inference:** a *generic* feature extractor walks the signature against the bound-variable table and
  device facts, producing the row in declared order. (Generic beats code-gen here because the sources
  are declarative and finite — no per-op C++.) Code-gen remains an option for a hot path.
- **Training:** the same signature drives the offline featurizer, so the dataset columns match the
  runtime row by construction. A parity test (signature → both sides → assert identical on fixtures) stays.

**A three-part, model-agnostic contract check** replaces the tooling's bare feature-count guard, and
it generalizes to any ranker (LightGBM, ONNX, a custom scorer):

0. **Signature → KMD** (schema-level, adapter-agnostic). Every `$kernel.*` field the `features_signature`
   references **must be declared in the engine's KMD** ([Section 6.1](#61-two-distinct-knob-concepts-ued-runtime-knobs-vs-kmd-compilation-knobs)).
   Both the KMD and the UHD are owned by the UED, so this is an intra-engine check the pipeline enforces
   when it emits the engine and the loader can re-check — a feature can never read a compilation knob the
   kernels don't carry.
1. **Signature → feature-vector** (adapter-agnostic). The UHD carries `features_hash`
   ([Section 4](#4-uhd-schema)); the model artifact **embeds the signature hash it was trained against**
   (tree-table metadata, ONNX `metadata_props`, or a sidecar). At load the runtime asserts
   `model.trained_hash == UHD.features_hash` and fails closed on mismatch. This is the same check for
   every model adapter, because it fingerprints the *input contract*, not the model internals.
2. **Feature-vector → model input** (adapter-specific). Each model adapter additionally verifies its
   artifact accepts the resolved vector: `tree_data` checks the tree table's feature count (the
   tooling's guard), `onnx` checks the graph's input arity/shape equals `|features_signature|`, and so on.

`features_hash` is **optional for feature-less adapters** (`static_order` has no vector to hash) and
**advisory for a `custom_library` that self-features** (ignores the standard vector). For every adapter
that consumes the standard vector, checks 0–2 are mandatory.

**OPEN — feature-vector portability.** This is open question 5 in both
[RFC 0017 §14](0017_UniversalKernelDescriptor.md#14-risks) and the spirit of
[RFC 0007](0007_EngineSelectionHeuristicsFramework.md). Do we standardize one graph/device feature
extractor so models are portable across UHDs, or keep it per-model via the spec? Recommend: **per-model
spec, shared source vocabulary**. The spec is per-model (models differ), but every spec draws from one
fixed, versioned source vocabulary and one extractor, so there is exactly one implementation to trust.

---

## 8. Model Formats and the Adapter Seam

The initial question — "LightGBM, CSV, or a separate library following a defined API?" — is really a
question about **how model content reaches the scorer**, and it maps cleanly onto
[RFC 0017 §8](0017_UniversalKernelDescriptor.md#8-adapters-and-extensibility)'s **adapter** model. A
UHD names an `adapter` (its single discriminant, [Section 4](#4-uhd-schema)); the **adapter** turns
that content into a scorer. Adding a new ranker — a static list, ONNX, a new model family — is one more
adapter value, nothing else. Adapters come in the same two delivery classes as kernel-source adapters.

**Design constraint: the model travels as data with the *engine*, not linked into the provider.** The
engine's descriptor set (UED + its UHD + KMD + model) must be a **standalone drop-in next to an
already-shipped provider**, exactly like the packs' `hsaco`/`kpack` code objects. That rules out
**statically linking the model into the provider** as the *shipping* mechanism — the model must be
loadable data the running provider reads, not a symbol compiled into the provider binary. (This core
decision is unchanged by the UED now owning the UHD; only the *unit* the model ships with moved from the
pack to the engine.) The reframe: the problem was never "compiled," it was "linked
into the provider." A model can still be *compiled* — it just has to ship as a loadable artifact, not
bake in.

| UHD content | Adapter | Delivery class | Runtime dependency | Standalone drop-in? |
|---|---|---|---|---|
| **Model-as-data (GBDT tree table) + in-tree tree-walker** | `tree_data` | **build-and-runtime** | none (evaluator is in-tree) | **Yes — default** |
| **ONNX graph + gated ONNX runtime** | `onnx` | build-and-runtime | ONNX runtime | Yes, if dep present (opt-in) |
| **Compiled scorer `.so`** (e.g. Treelite output) | `custom_library` | build-and-runtime | none on provider; engine owns it | Yes (registered/dlopen handler) |
| **`liblightgbm` at runtime** | `lightgbm_native` | build-and-runtime | `liblightgbm` | Yes, if dep present (opt-in) |
| **CSV / lookup table** | `table` | build-and-runtime | none | Yes |
| **Static precedence** | `static_order` | built-in | none | Yes (always available) |
| **LightGBM → C linked into provider** | `lgbm_to_c` (original exporter) | **build-only** | none (static-linked) | **No** — not a shipping path |

The two-tier resolution (mirrors [RFC 0017](0017_UniversalKernelDescriptor.md)'s data → escape-hatch →
native ladder):

- **Default — model as data (`tree_data`).** The provider ships one small, generic **GBDT tree-walker**;
  the engine ships the model as a **data artifact** (tree table: feature indices, thresholds, leaf
  values) that the walker reads, referenced by the UHD's `model.artifact`. GBDT trees (LightGBM/XGBoost)
  are trivial to evaluate (~few hundred lines) and the tooling already dumps the model to a walkable
  structure. Zero runtime dependency, fully standalone, verifier-gated. This is the hsaco-equivalent for
  heuristics. Limit: the provider must already support the model *family* (a new family = a provider change).
- **Escape hatch — compiled scorer `.so` (`custom_library`).** For a model the in-tree walker does not
  cover, the engine ships its **own** compiled scorer `.so`, `dlopen`'d through a tiny C ABI
  (`score(const double* feats, ...) -> double`). Compiled but standalone-added — nothing links into the
  provider. **Treelite** generates such a `.so` from a tree model, so this is "`lgbm_to_c`, but the
  output is a standalone `.so` shipped with the engine rather than a provider TU." Any model family;
  author-native-code trust class (prebuilt-trust per
  [RFC 0017 §10](0017_UniversalKernelDescriptor.md#10-packaging-and-delivery)).

**Artifact-format sub-decision (OPEN).** For `tree_data`, the tree table can be either:
(a) a **data-SDK FlatBuffer tree schema** (recommended) — consistent with the graph/device-props
serialization, `Verifier`-gated, additive evolution, needs a convert step at build; or
(b) **parse LightGBM `model.txt` at load** — lowest author friction (no conversion) but a bespoke
parser to write and harden against hostile input. Lean FlatBuffer for consistency and safety.

All model adapters (`tree_data`, `onnx`, `lightgbm_native`) share the **artifact packaging** — the UHD
names `model.artifact`, shipped as data with the engine's descriptor set — and the **contract check** of
[Section 7.3](#73-bit-parity-between-training-and-inference). They differ only in evaluator and
dependency, so `onnx` is "`tree_data` with a gated runtime and its own input-arity check," not a new
code path.

**Recommended initial support:** `static_order` (trivial, always available) + **`tree_data`** (the
in-tree GBDT walker) as the default shipping path, with **`custom_library`** as the escape hatch. CSV
`table` is a cheap add for coarse bucketed heuristics. **`onnx`** and **`lightgbm_native`** are
additional model adapters, both opt-in / dependency-gated, added when a concrete need appears.
`lgbm_to_c` is kept **only** as an optional build-time perf optimization for in-tree AOT models, *not*
as a drop-in shipping mechanism.

---

## 9. Dependencies

Dependencies are the hard external constraint ("restricted by what we can support") and they steer the
format choice above. First-pass stance:

- **Zero new runtime dependency for the default path.** The `tree_data` adapter's evaluator is
  **in-tree**, so the default shipping path adds **no** runtime library and the model is standalone
  data shipped with the engine. This must remain true for anything we ship by default; the provider
  cannot grow a hard `liblightgbm` link.
- **Build-time LightGBM is acceptable** — training and the tree-table conversion run offline in the
  pipeline ([Section 13](#13-model-generation-pipeline)), never in the shipped runtime.
- **A `custom_library` scorer `.so` carries its own inference** — no provider dependency; the engine
  owns whatever it linked (e.g. a Treelite-generated evaluator).
- **`liblightgbm` at runtime is opt-in only**, behind the `lightgbm_native` adapter, for environments
  that already have it. Never a default.
- **FlatBuffers / data-SDK are already in-tree** and are the natural carrier for the `features_signature`
  and any serialized model-table, consistent with
  [RFC 0007 §13](0007_EngineSelectionHeuristicsFramework.md#13-serialized-graph-device-properties-and-graph-level-preferences)
  and [RFC 0017 §4](0017_UniversalKernelDescriptor.md#4-descriptor-formats).

**OPEN — needs deeper investigation:** the exact allowed dependency surface for a shipped provider
(license, distro packaging, ROCm image contents), and for a `custom_library` `.so` the trust/signing
rules for dropping in author-compiled native code. The former decides whether the in-tree tree-walker
must be fully first-party (recommended) or may vendor a third-party evaluator; the latter gates the
`custom_library` drop-in path. Flag for a dedicated dependency + trust audit before committing either.

---

## 10. Performance: Loading, Caching, Lazy Evaluation

Selection runs on the plan-build path, so its cost must be small and paid at most once per distinct
need. Three requirements:

- **Lazy load — don't pay if no one needs it.** An engine's UHD model is loaded/parsed only when a graph
  actually reaches kernel selection for that engine, not at provider startup or descriptor discovery
  (the UED itself loads eagerly per [RFC 0017 §3](0017_UniversalKernelDescriptor.md#3-how-it-works), but
  its model artifact stays lazy). A provider that never hits FMHA never parses the FMHA tree table. (The
  compiled-in path sidestepped this; a data/`.so` adapter must be explicitly lazy —
  parse-on-first-use.)
- **Cache the loaded model.** After first load, the parsed model / tree table / native handle is
  cached for the process (or per `hipdnnHandle`, matching the session-handle lifetime of
  [RFC 0007 §8.3](0007_EngineSelectionHeuristicsFramework.md#83-plugin-handle-session-object)). Loading
  is amortized to once.
- **Cache results where the problem repeats.** Selection is a pure function of (feature vector,
  candidate set). A small plan cache keyed by a problem fingerprint + candidate-set fingerprint can
  skip re-inference for repeated graphs, the same idea as MIGraphX's problem→solution cache noted in
  [RFC 0017 §15](0017_UniversalKernelDescriptor.md#15-references-and-prior-art). **OPEN**: is per-plan
  caching enough, or do we want a persistent cross-run cache (interacts with a future
  [RFC 0007](0007_EngineSelectionHeuristicsFramework.md) "cache selector" policy)?
- **Minimize init overhead.** Feature extraction is a fixed walk over the spec; inference is a handful
  of tree evaluations per candidate. Keep the feature row and any scratch preallocated per session, as
  [RFC 0017 §6](0017_UniversalKernelDescriptor.md#6-dispatch-and-workspace) does for launch. Overhead
  is validated against the compiled-C baseline in
  [RFC 0017 §12.1](0017_UniversalKernelDescriptor.md#121-testing-and-performance).

The compiled-C path is the **performance floor** for measurement (near-zero inference cost). The
`tree_data` walker is expected to be close — a flat tree table over a preallocated feature row is a few
hundred comparisons per candidate — but it is measured against that floor. If the gap is ever material
for a hot in-tree op, that op may *additionally* be built via the `lgbm_to_c` optimization; but the
**shipped drop-in path is always the data/`.so` model**, never a provider-linked symbol.

---

## 11. Engine Selection Interplay and Estimated TFLOPS

This is the fourth item from the brief and the one that reaches beyond the kernel level. It is also
where two refinements land: **applicability bubbles up before ranking**, and the tooling produces **two**
heuristics that engine-selection policies consume in **two** ways.

**Applicability bubbles up first.** Engine selection cannot rank engines it hasn't ruled out. For a
descriptor engine, applicability is the **matcher (UMD) pass**; that result bubbles up so the engine is
excluded *before* the first plugin-policy layer ranks the rest. So the descriptor/UHD layer runs (for
applicability, and optionally for scoring) *ahead of* engine selection — not strictly after it. Today
rocKE's `isApplicable` is that yes/no gate; what changes is that the same layer can also report a
*predicted performance*, so the policy can order engines by merit instead of a static list.

### 11.1 Two heuristics the tooling produces

Per engine, the generation tooling ([Section 13](#13-model-generation-pipeline)) emits **two** models,
both predicting **absolute** performance so they are comparable across engines:

| | Model | Signature | Cost | Role |
|---|---|---|---|---|
| **A** | **Engine performance estimate** | `f(graph) → expected performance of the package` | cheap (no config enumeration) | feeds the *quick* policy's engine ranking |
| **B** | **Config UHD** (this RFC's UHD) | `f(graph) → best config/kernel + its predicted performance` | full per-candidate | picks the kernel *and* gives the accurate cross-engine number |

A is the coarse proxy; B both selects the kernel and yields the better figure of merit. Both live on the
UED (engine-level). **OPEN:** is A a distinct trained model, or derived from B as the max predicted score
over its candidates? (Distinct is cheaper for the quick policy since it skips enumeration; derived is one
fewer model to train.) And what to name A — a working label is the *engine performance estimate*, kept
distinct from the *UHD* (config selector).

### 11.2 Two engine-selection policies (RFC 0007) that consume them

The policies live in [RFC 0007](0007_EngineSelectionHeuristicsFramework.md); this RFC only supplies what
they read. Two modes:

- **Quick policy.** Rank applicable engines by **A** (expected performance); pick the winner; if the
  winner has a config UHD (**B**), run it to pick the kernel. Only the winner drills down, so losers are
  never scored at the kernel level. Engines with no descriptor layer (e.g. MIOpen) contribute their
  **high-level** estimate for the ranking and, if they win, use their **own** internal kernel selection —
  they never touch B.
- **Thorough policy (longer-running).** Run **B** for every applicable engine that has it (best config +
  its predicted perf), fall back to **A** for engines that don't, then compare the predicted performance
  **across** engines and pick the global best (engine + config). More work, more accurate.

Both mix "engines with B" and "engines with only A" — the descriptor/UHD layer is **opt-in per engine**,
so the framework must compare a full config prediction against a coarser high-level estimate without
assuming every engine has both.

```
Quick:     applicable → rank by A → winner → (B? kernel : own selection) → dispatch
Thorough:  applicable → run B (or A) for every engine → compare perf → best (engine,config) → dispatch
```

### 11.3 The absolute metric and its fallback

**The idea: an absolute, cross-comparable figure of merit.** The ambition is to score candidates
**cardinally** — an absolute metric (calibrated TFLOPS) rather than a within-group rank. If a UHD is a
calibrated TFLOPS regressor, its best-candidate score is a **predicted figure of merit for what the
engine would actually run**, expressed on a scale that means the same thing across engines. That is what
lets each engine **run its own heuristic per package, independently, and still have those results be
meaningfully comparable to every other package and engine**: hipDNN compares engines by predicted
performance instead of a fixed order — "rocKE predicts 310 TFLOPS for its best FMHA kernel; MIOpen's
estimate is 240" → pick rocKE. The value is precisely that local, per-package scoring composes into a
global comparison without a central ranker; the UHD is where the number naturally exists.

**Why absolute is harder than rank — and the safety valve.** A per-package model only needs to be
*monotonic* to pick correctly among its own candidates (argmax is scale-free). Making it *calibrated* —
accurate in absolute TFLOPS so cross-engine comparison is honest — is a strictly harder modeling
problem, and a *miscalibrated* absolute score is worse than an honest rank, because it yields
confident-but-wrong cross-engine picks. So the absolute method is pursued as the goal, but it needs a
fallback for exactly the failure mode it introduces.

**Fallback if the absolute method underperforms.** One fallback is to **degrade to classic
rank-ordering at the engine-policy level** — engine selection reverts to the existing
[RFC 0007](0007_EngineSelectionHeuristicsFramework.md) static/rank ordering, and each UHD keeps ranking
*within* its engine (where only monotonicity is needed) without claiming a comparable absolute score.
This is one option, not the only one: if the absolute approach proves unreliable we may pursue other
cross-engine schemes in the future (for example a normalized/relative score, or a calibration layer
applied at the policy level). The point is that the design does not bet everything on calibration
succeeding — rank-ordering is the defined safe backstop, with room to explore alternatives.

**Why it's a coordinated follow-up, not committed here.** Delivering it requires changes *outside* the
UHD:

1. **A plugin-query surface** for the per-graph figure of merit — both the cheap **A** (engine
   performance estimate) and the accurate **B** (config UHD run in a "score only, don't launch" mode).
   This is an engine-plugin ABI addition, owned by the plugin SDK, not this RFC; it must also let a
   non-descriptor engine (MIOpen) report an A-level estimate through the same surface, or the policy
   falls back to today's static ordering for it.
2. **The two engine-selection policies** that consume it ([Section 11.2](#112-two-engine-selection-policies-rfc-0007-that-consume-them)) —
   the quick policy (rank by A, drill into the winner's B) and the thorough policy (run every B, compare
   across engines). Both are squarely [RFC 0007](0007_EngineSelectionHeuristicsFramework.md)'s territory.
3. **Cross-engine calibration.** Comparing estimates across engines only works if the units are
   comparable and each model (A and B) is calibrated to real TFLOPS (not just monotonic for argmax).
   This is a real modeling requirement, not just plumbing — and the one most likely to force the
   rank-ordering fallback above if it does not hold up.

**What this RFC commits to** so the door stays open: the UHD schema declares `score`
(`units`/`calibrated`/`transform`, [Section 4](#4-uhd-schema)) so a consumer can invert the training
transform and recover real TFLOPS, and supports a **score-only evaluation mode** (rank/return best
score without selecting-for-launch). The estimate-to-engine-selection wiring — the plugin query, the
[RFC 0007](0007_EngineSelectionHeuristicsFramework.md) policy, and calibration — is a **dedicated
follow-up co-owned with [RFC 0007](0007_EngineSelectionHeuristicsFramework.md)**, as is the
rank-ordering fallback (it lives at the engine-policy level, which is that RFC's territory). We should
write that follow-up's problem statement now and reference it, but not block UHD v1 on it.

**OPEN — how hard to commit to the absolute metric.** The ambition is a calibrated, cross-comparable
score. Training on real TFLOPS from the start is cheap insurance (the tooling already does), so v1
should train calibratable targets even while cross-engine comparison is validated separately. What is
*not* committed is that cross-engine comparison must succeed: if calibration proves unreliable, engine
selection falls back to classic rank-ordering at the [RFC 0007](0007_EngineSelectionHeuristicsFramework.md)
policy level (with other cross-engine schemes left open for the future). So the open decision is the
*degree of commitment* — build the absolute path and the rank-ordering backstop together, rather than
mandating that the absolute path be the only outcome.

---

## 12. Observability

Because selection is data-driven, it must be inspectable, consistent with
[RFC 0017 §9](0017_UniversalKernelDescriptor.md#9-observability-and-diagnostics) and
[RFC 0007 §12](0007_EngineSelectionHeuristicsFramework.md#12-logging). The UHD path surfaces:

- **A selection trace:** the candidate UKDs, the feature row (or its fingerprint), each candidate's
  score, the winner, and whether the decision came from the model or a tie-break/fallback.
- **Model provenance:** which UHD and which model artifact (id, version, training provenance) served
  the decision, and whether it loaded or fell back to `static_order`.
- **Feature-contract diagnostics:** a clear, single failure when the runtime feature count/spec
  disagrees with the model (the tooling's num-features guard), naming the mismatch rather than
  silently mis-scoring.

---

## 13. Model Generation Pipeline

The UHD is only useful if producing one is automated — by **tooling any package author can run**, not a
provider-specific service. The workflow is two-stage:

1. **Ship a working pack with a trivial heuristic.** The pack's UED names a **`static_order` UHD**
   (rank by `priority`/`id` — effectively "run the first applicable kernel"). The pack is fully
   functional and model-free from day one; no benchmarking or training is needed to use it.
2. **Generate a real heuristic from on-hardware timings.** A standalone generation tool loads the pack,
   times its kernels across a corpus of problem shapes, trains a model, and **emits an updated UED/UHD**
   — same descriptor kind, now `adapter: tree_data` pointing at an exported `model.txt`
   ([Section 8](#8-model-formats-and-the-adapter-seam)). Dropping that updated engine descriptor set
   back in upgrades the pack from trivial ordering to a trained heuristic **in place** — no hipDNN
   recompile, no provider-internal changes.

Because the shipped and generated heuristics are the *same descriptor kind* differing only in `adapter`
and fields, the tool only rewrites data; it never introduces a new interface. Critically, the tool runs
**over hipDNN's public API** — it adds no code to hipDNN and touches no provider internals — so it works
for any provider's pack, which is what makes it usable by anyone.

### 13.1 Benchmarking via hipDNN autotune (RFC 0013)

The timing substrate is hipDNN's own **autotune** ([RFC 0013](0013_Autotune.md)), not a bespoke sweep.
Autotune is **provider-agnostic**: it times whatever engine/kernel actually runs (through a backend
profiling descriptor and `backendExecute`), so it exercises a rocKE pack exactly as it would any other
engine. The generation tool drives it through the public frontend Graph API:

- **`get_engine_configs()`** — enumerate the applicable candidates for a graph (the pack's child UKDs);
- **`add_engine_variants()` / `add_engine_sweep()`** — enroll *every* candidate (and any runtime-knob
  combinations) as plan specs, not just the heuristic-picked one;
- **`autotune(mode = EXHAUSTIVE, strategy = RUN_UNTIL_STABLE)`** — compile and time each candidate
  (HIP-event timing, warmup + timed iterations, workspace filtering);
- **`AutotuneResult[]`** — per-candidate `engineId`/config, `minTimeMs`/`avgTimeMs`, `workspaceSize`,
  knobs, and rank, persisted to JSON. **That JSON is the training dataset** — no separate benchmarking
  format to define. `samples/autotune/AutotuneSample.cpp` is a reference driver.

Because the pack already contains its variant kernels as UKDs, the tool times the **shipped** kernels —
it does not re-enumerate or re-build a variant grid. The pack (its build step) is the authority on which
variants exist; autotune is the authority on how fast each one runs. The tool's own job is only to drive
autotune across the **shape corpus** ([Section 13.4](#134-sweep-space-grid-vs-constraint)); the per-point
timing is autotune's.

### 13.2 The principle: one source of truth, translate once

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

### 13.3 New stage: package (Stage P)

From one timing run (§13.1) the tool trains **two** models
([Section 11.1](#111-two-heuristics-the-tooling-produces)): the fine-grained **config UHD (B)** and the
cheap **engine performance estimate (A)**. A **package stage** then emits (or updates) the engine's
descriptor set:

- the **config UHD (B)** — rewritten from the shipped `static_order` to `adapter: tree_data`, carrying
  the `features_signature` (referencing `$kernel.*` KMD fields + `$device.*` for arch-awareness),
  `features_hash`, `objective`/`score`, and `model.artifact` ([Section 4](#4-uhd-schema)); one per
  engine, arch-aware, so no artifact table;
- the **engine performance estimate (A)** — the coarse `f(graph) → expected perf` model the quick policy
  ranks engines by ([Section 11.2](#112-two-engine-selection-policies-rfc-0007-that-consume-them)), also
  emitted as data on the UED (whether A is its own model or derived from B is the OPEN in
  [Section 11.1](#111-two-heuristics-the-tooling-produces));
- the **model files as data** — the trained boosters exported to their model files (the `tree_data`
  format, read by the in-tree walker; [Section 8](#8-model-formats-and-the-adapter-seam)), each embedding
  the `features_hash` it was trained against; shipped with the engine descriptors, *not* compiled in;
- the **KMD** — the compilation-knob schema (`fields`: `tile_m`, `warp_n`, `split_k`, `dtype`, …), if
  the pack does not already carry one; its fields are exactly the `$kernel.*` metadata the UKDs already
  fill ([Section 6.1](#61-two-distinct-knob-concepts-ued-runtime-knobs-vs-kmd-compilation-knobs)) —
  **one KMD per engine, owned by the UED**;
- the **UED** — updated to reference the new UHD and the A model; its user runtime `knobs` are distinct
  from the KMD compilation knobs and untouched by generation.

The UMDs, UDD, and the child UKDs (kernels) are **not** regenerated — only the heuristic side changes.
That is the whole point of the two-stage design: the expensive artifacts (compiled kernels) ship once;
both heuristics are layered on afterward as data.

### 13.4 Sweep space: grid vs. constraint

The generation tool sweeps two things: the **problem-shape corpus** (batch, seqlen, heads, … — supplied
by the author as representative shapes, or a per-op default) and optionally the **UED runtime knobs**
(via `add_engine_variants` knob settings). The **variant space itself is fixed** — it is the pack's
existing child UKDs, so the tool does not enumerate or build variants; it enrolls the shipped ones and
times them ([Section 13.1](#131-benchmarking-via-hipdnn-autotune-rfc-0013)).

One subtlety for anything that *drives* a sweep from a descriptor: a validity *constraint*
(`min:1, max:8`) expresses which values are **legal**, not which to **sample**, so a swept axis needs an
explicit `sweep_values` / grid hint, not an inferred range. **OPEN**: standardize where the shape corpus
and any runtime-knob grid live (a tool-side config vs. a descriptor field), so a heuristic can be
regenerated reproducibly without out-of-band inputs.

---

## 14. Phased Delivery

Each phase is independently shippable and validated against the SDPA path and the reference tooling, using the
parity and overhead checks of [RFC 0017 §12.1](0017_UniversalKernelDescriptor.md#121-testing-and-performance).
The estimated-TFLOPS item (phase 7) is a co-owned [RFC 0007](0007_EngineSelectionHeuristicsFramework.md)
follow-up.

1. **`static_order` baseline (the shippable trivial heuristic).** UHD schema + KDP membership +
   deterministic ranking (priority/id). Every pack gets a working, model-free selector — the first
   stage of the two-stage workflow ([Section 13](#13-model-generation-pipeline)). Proves the
   UED→UHD→child-UKD wiring end to end.
2. **`features_signature` + generic extractor.** The UHD's inline signature and the single generic extractor
   over the shared field namespaces ([Section 7](#7-feature-extraction-and-binding)), with a
   training↔runtime parity test, plus the expression-op extension
   ([Section 7.2](#72-the-features_signature)).
3. **`tree_data` model-as-data + in-tree walker.** The default shipping path: a LightGBM model exported
   to data, evaluated by the in-tree GBDT walker (reusing MIOpen's `LgbmForest`-style parser); lands the
   real FMHA-fwd model as a standalone drop-in. Adds lazy load + model cache
   ([Section 10](#10-performance-loading-caching-lazy-evaluation)).
4. **Generation tool over hipDNN autotune.** The standalone tool that drives `autotune` (RFC 0013)
   across a shape corpus, trains a model, and emits the updated UED/UHD — the second stage that turns a
   `static_order` pack into a `tree_data` one, enforcing the two contracts
   ([Section 13](#13-model-generation-pipeline)).
5. **`table` / CSV.** Cheap bucketed heuristics for ops that don't warrant a model.
6. **`custom_library` escape hatch.** Compiled scorer `.so` (e.g. Treelite) for models the in-tree
   walker doesn't cover; dependency + trust audit gated ([Section 9](#9-dependencies)).
7. **Estimated-TFLOPS follow-up (co-owned with [RFC 0007](0007_EngineSelectionHeuristicsFramework.md)).**
   Score-only mode, the plugin-query surface, and the engine-selection policy that consumes it
   ([Section 11](#11-engine-selection-interplay-and-estimated-tflops)). Separate RFC.

`lgbm_to_c` (build-time, in-tree perf optimization only) and `lightgbm_native` are dependency-gated and
land only when a concrete need appears.

---

## 15. Risks

- **Feature-contract drift.** The single largest risk, and the tooling's hardest-won lesson. One
  versioned `features_signature` driving both sides via one generic extractor
  ([Section 13.2](#132-the-principle-one-source-of-truth-translate-once)), a parity test, and the
  three-part load-time check — signature⊆KMD, `features_hash` (signature → vector), and the adapter's
  own arity check (vector → model input) — failing closed on any
  ([Section 7.3](#73-bit-parity-between-training-and-inference)).
- **Kernel-identity drift.** The candidate autotune timed (its `engineId`/config) must equal the emitted
  UKD `metadata`, or the model's argmax maps to the wrong kernel
  ([Section 13.2](#132-the-principle-one-source-of-truth-translate-once)).
- **KMD↔UHD coupling.** Because the UED co-owns the KMD and UHD, changing the KMD (adding/altering a
  compilation knob) invalidates the trained model — a new field is not selected against until the model
  learns it. Dropping in a KDP with kernels whose variants the engine's model never saw silently
  degrades ranking. Mitigation: gate on the coupling — retrain/re-emit the UHD whenever the KMD changes,
  and surface "model did not train on this field" in the trace ([Section 12](#12-observability)).
- **Dependency creep.** Pressure to just link `liblightgbm`. Held off by the in-tree `tree_data`
  default ([Section 9](#9-dependencies)); a runtime native dep stays opt-in.
- **Bad/stale model regresses selection.** A model can pick worse than first-match. Mitigations:
  fail-open to `static_order`; a generic-vs-baseline parity gate; model provenance in the trace.
- **Calibration for engine-level comparison.** The absolute cross-engine metric may not pan out — a
  *miscalibrated* score is worse than an honest rank. Mitigation is designed in, not hoped for: train
  calibratable TFLOPS from the start (cheap insurance, keeps the option open) and keep classic
  rank-ordering at the RFC-0007 policy level as the defined fallback, with other cross-engine schemes
  open for the future ([Section 11](#11-engine-selection-interplay-and-estimated-tflops)).
- **Cache correctness.** A result cache keyed on an incomplete fingerprint returns a wrong kernel.
  Fingerprint must include everything the feature row depends on (problem + candidate set + device).
- **Drop-in trust.** A runtime model artifact is author-controlled input; the loader/evaluator must be
  bounded and fail-closed, inheriting the trust rules of
  [RFC 0017 §10](0017_UniversalKernelDescriptor.md#10-packaging-and-delivery).

---

## 16. Open Questions

1. **Regressor vs. ranker, and degree of commitment to the absolute metric.** Train calibratable
   TFLOPS (keeps the cross-comparable option open) vs. a pure within-engine ranker; and how hard to
   commit to cross-engine cardinal comparison vs. keeping rank-ordering at the RFC-0007 policy level as
   the defined fallback ([Section 4](#4-uhd-schema), [Section 11](#11-engine-selection-interplay-and-estimated-tflops)).
2. **Per-engine arch-aware model vs. narrower engine scope.** The UHD is one per engine (owned by the
   UED), and the engine spans arches, so the model is arch-aware via `$device.*` — one model, no
   `(arch,dtype)→artifact` table. Open: does a single arch-aware model match per-arch accuracy, or should
   an engine be scoped per arch/dtype (a UED per arch) so its one UHD stays within one arch? Decide
   against real cross-arch data ([Section 6.3](#63-membership-rules), [Section 13](#13-model-generation-pipeline)).
3. **Multiple matching packs for one engine** — forbid overlap (recommended), rank-per-pack, or compare
   by units? ([Section 5](#5-how-a-uhd-ranks-matched-kernels)).
4. **Feature-source set** — the heuristic ranks over UKD `metadata` (compile-time build config), *not*
   UED knobs (user runtime knobs) — see the `split_k` boundary case
   ([Section 6.1](#61-two-distinct-knob-concepts-ued-runtime-knobs-vs-kmd-compilation-knobs)). Settle the full
   derived-feature set the `features_signature` must expose (arithmetic intensity, tile/wave quantization,
   aspect ratios, occupancy, padding-fit — are there others?)
   ([Section 6.2](#62-derived-features-the-heuristic-needs)).
5. **Model artifact format(s)** — for `tree_data`, a data-SDK FlatBuffer tree schema (recommended) vs.
   parsing LightGBM `model.txt` at load; and which model adapters to support beyond it (`onnx`,
   `lightgbm_native`) given their dependencies ([Section 8](#8-model-formats-and-the-adapter-seam)).
   Includes the three-part contract check (signature⊆KMD + `features_hash` + adapter arity —
   [Section 7.3](#73-bit-parity-between-training-and-inference)) and the dependency + trust audit for
   the in-tree walker and the `custom_library` `.so` path ([Section 9](#9-dependencies)).
6. **Feature portability** — one shared extractor + per-model spec (recommended) vs. a standardized
   fixed feature vector ([Section 7.3](#73-bit-parity-between-training-and-inference)); and the
   expression-op extension set ([Section 7.2](#72-the-features_signature)).
7. **Caching scope** — per-plan only vs. persistent cross-run, and its interaction with a future
   [RFC 0007](0007_EngineSelectionHeuristicsFramework.md) cache policy
   ([Section 10](#10-performance-loading-caching-lazy-evaluation)).
8. **Where the sweep inputs live** — the variant space is fixed (the pack's UKDs, timed via autotune),
   but the **shape corpus** and any **runtime-knob grid** need a home: a tool-side config vs. a
   descriptor field, so a heuristic regenerates reproducibly without out-of-band inputs
   ([Section 13.4](#134-sweep-space-grid-vs-constraint)).
9. **Where estimated-TFLOPS wiring lives** — confirm it is a co-owned
   [RFC 0007](0007_EngineSelectionHeuristicsFramework.md) follow-up, and draft that problem statement
   ([Section 11](#11-engine-selection-interplay-and-estimated-tflops)).
10. **The two heuristics (A vs. B).** Is the engine performance estimate (A) a distinct trained model or
    derived from the config UHD (B, max over candidates)? Where does A live and what is it named
    ([Section 11.1](#111-two-heuristics-the-tooling-produces))? And how does a non-descriptor engine
    (MIOpen) report an A-level estimate through the plugin-query surface — vs. the quick policy falling
    back to static ordering for it ([Section 11.2](#112-two-engine-selection-policies-rfc-0007-that-consume-them))?

---

## 17. Glossary

- **UHD (Universal Heuristic Descriptor):** one kernel-selection model, **owned by the UED (one per
  engine)** and referenced by the UED by ID, that ranks the applicable child UKDs of every pack joining
  its engine and picks one. Per-engine and arch-aware (takes `$device.*`).
- **KDP (Kernel Descriptor Pack):** the pack that joins an engine and adds kernels; names one matcher
  set, one UED (which carries the UHD and KMD), and one UDD over a vector of child UKDs. The selection
  group is a pack's child kernels; the selector and metadata schema come from the engine.
- **KMD (Kernel Metadata Descriptor):** [RFC 0017](0017_UniversalKernelDescriptor.md)'s explicit,
  upfront declaration of the engine's **compilation knobs** — the variant `fields` (`tile_m`, `split_k`,
  `dtype`, …) every kernel carries, each with a type and optional default. **One KMD per engine, owned
  by the UED**; each UKD's `metadata` fills it. It is the authoritative schema for the `$kernel.*` fields
  the UHD ranks on and the `features_signature` references
  ([Section 6.1](#61-two-distinct-knob-concepts-ued-runtime-knobs-vs-kmd-compilation-knobs)).
- **UED ownership of UHD + KMD:** the UED (engine) names one UHD and one KMD. They are coupled — the KMD
  is the feature space the UHD ranks over — so the engine owns both, and changing the KMD requires
  retraining the UHD ([Section 6.1](#61-two-distinct-knob-concepts-ued-runtime-knobs-vs-kmd-compilation-knobs)).
- **UED runtime knobs vs. KMD compilation knobs:** two distinct concepts in two descriptors. **UED
  knobs** are user-exposed *runtime* params (`split_k`, `use_atomics`), mostly orthogonal to selection.
  **KMD fields** are the *compile-time* variant axes each kernel was built with (`tile_m`, `warp_n`),
  filled per-kernel in UKD `metadata` — and *this* is what the UHD ranks over, as `$kernel.*`. Each UKD
  is one point in the KMD-declared space; the KDP is the collection
  ([Section 6.1](#61-two-distinct-knob-concepts-ued-runtime-knobs-vs-kmd-compilation-knobs)).
- **Kernel-selection heuristic vs. engine-selection heuristic:** the two levels; the UHD is the former
  (which kernel within an engine), [RFC 0007](0007_EngineSelectionHeuristicsFramework.md) owns the
  latter (which engine).
- **`features_signature`:** the UHD's ordered, versioned list of model inputs (bare `$`-prefixed fields
  or derived expressions over the shared [RFC 0017](0017_UniversalKernelDescriptor.md) namespaces) that
  both training and inference consume through one generic extractor; the contract that must stay
  bit-identical across the two ([Section 7](#7-feature-extraction-and-binding)).
- **Scorer / adapter:** the thing that turns a UHD's model content into a per-candidate score; reached
  through an adapter in build-and-runtime (default) or build-only delivery classes, mirroring
  [RFC 0017 §8](0017_UniversalKernelDescriptor.md#8-adapters-and-extensibility).
- **Score-only mode:** running a UHD to obtain the best predicted score without selecting for launch;
  the hook for surfacing estimated TFLOPS to engine selection
  ([Section 11](#11-engine-selection-interplay-and-estimated-tflops)).
- **`tree_data`:** the default shipping path — a GBDT tree table shipped as data *with the engine's
  descriptor set* and evaluated by an **in-tree GBDT walker**; standalone drop-in, zero runtime
  dependency ([Section 8](#8-model-formats-and-the-adapter-seam)).
- **`custom_library`:** the escape hatch — a compiled scorer `.so` (e.g. Treelite output) shipped with
  the engine, `dlopen`'d through a tiny C ABI; standalone, any model family, prebuilt-trust
  ([Section 8](#8-model-formats-and-the-adapter-seam)).
- **`lgbm_to_c`:** the tooling's build-only path that lowers a LightGBM booster to C linked **into the
  provider**. Kept only as an in-tree AOT perf optimization — **not** a drop-in shipping mechanism
  ([Section 3.1](#31-the-heuristic-generation-tooling)).
- **Stage P (package):** the pipeline stage that emits the engine descriptor set (UED/UHD/KMD +
  tree-table) and its KDP(s) from the same sweep that trained the model, enforcing the feature and
  kernel-identity contracts
  ([Section 13](#13-model-generation-pipeline)).
