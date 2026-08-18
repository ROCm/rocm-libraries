# RFC 0020: The Universal Engine Descriptor (UED)

- Contributors: TBD
- **Status**: Draft
- **Implementation Version**: TBD
- **Follows**: [RFC 0017](0017_UniversalKernelDescriptor.md) (Universal Kernel Descriptors). This is the "UED + engine registry" follow-up named in RFC 0017 § 14.2.

## Table of Contents

1. [Summary](#1-summary)
2. [Relationship to RFC 0017](#2-relationship-to-rfc-0017)
3. [Engine Identity](#3-engine-identity)
4. [The UED Schema](#4-the-ued-schema)
5. [Knobs](#5-knobs)
6. [Behavior and Numerical Notes](#6-behavior-and-numerical-notes)
7. [Engine Membership (UKD -> KDP -> UED)](#7-engine-membership)
8. [When a UED Is Loaded and Registered](#8-when-a-ued-is-loaded-and-registered)
9. [Engine Registration](#9-engine-registration)
10. [Validation](#10-validation)
11. [Versioning and Compatibility](#11-versioning-and-compatibility)
12. [Lifecycle and Operational Policy](#12-lifecycle-and-operational-policy)
13. [Testing](#13-testing)
14. [Glossary](#14-glossary)
15. [Appendix: Fully-Populated UED Examples](#15-appendix-fully-populated-ued-examples)

---

## 1. Summary

RFC 0017 established a family of declarative descriptors that one **generic engine** loads,
matches, selects, and launches with no new C++. It described each at a framing level and deferred
each descriptor's detailed format to its own follow-up. This RFC specifies the **Universal Engine
Descriptor (UED)**: one engine's identity, the KMD fields it exposes as knobs, and its behavior and
numerical notes, plus the registration that turns a UED into a selectable engine.

Concretely, this RFC delivers:

- The **UED field contract**: the normative field set, with a per-version JSON Schema file
  recommended (not mandated) as its single-source-of-truth expression, plus serialization
  (§ 4, § 11.3).
- The **engine-identity model**, including the two distinct id spaces a descriptor engine
  lives in: the descriptor-cross-reference UUID and the hipDNN 64-bit engine id (§ 3).
- **Engine registration**: the process that instantiates the generic engine from UED data and
  exposes it through the provider's engine list (§ 9), plus the ingestion paths and the
  registration-timing guarantee a loaded UED depends on (§ 8).
- The **validation contract**, structural (the field contract, build + runtime) and semantic
  (cross-descriptor, including drop-all duplicate detection), with guidance on the UED-vs-KDP
  boundary (§ 10).
- **Versioning & compatibility**: the accept rule keyed on the `version` field, the constrained
  meaning of `major`/`minor`, and the per-version schema-file mechanism (§ 11); plus
  **lifecycle/operational policy** (load-failure, concurrency, the `HIPDNN_DISABLE_ENGINES`
  opt-out) and **test scope** (§ 12-13).

**Out of scope:** Drop-in **trust and enablement** rules for untrusted descriptor files remain
out of scope, as in RFC 0017 § 14; this RFC adds no trust policy.

## 2. Relationship to RFC 0017

This RFC lives alongside RFC 0017; some sections restate 0017 material (identity, knobs, the
reference model) so the UED format reads standalone. **This RFC is the source of truth for UED
matters.** It follows 0017's descriptor conventions, including the two version-bearing fields
0017's KDP and UMD examples carry: a `schema` type tag (`hipdnn.ued/v1`) and a separate
`major.minor` `version` field that the accept rule gates on (§ 11). The relationships to 0017 recorded below are
**tightenings**: points 0017 defers or under-constrains (often to this very follow-up, per 0017
§ 14.2) that this RFC now pins down. As a follow-up, filling 0017's deferred scope is expected and
is not itself a divergence.

**Tightenings (this RFC pins down what 0017 deferred or left soft):**

- **Version field (§ 4.2).** 0017's UED example carries only the `schema` tag; this RFC adds the
  separate `major.minor` `version` field the accept rule gates on, matching 0017's KDP and UMD
  examples.
- **Compatibility mechanism (§ 11).** Absence-safe minor bumps, hard-reject unknown fields, and a
  runtime version from the provider's schema file(s).
- **Engine name format (§ 4.2).** A globally-unique, scoped `namespace:local` `name`, since it is
  hashed into the engine-id space and must not collide.
- **Duplicate detection (§ 10.2.1).** An independent descriptor-`id` check; drop all UEDs in a
  genuine collision, but accept content-identical `id` duplicates, loading them as one.
- **Role-scoped, arch-keyed heuristics (§ 4.4).** 0017 sketches a single `heuristic` id; this RFC
  replaces it with three optional, arch-keyed maps — `sort_kernel_catalog` (the kernel-selection
  heuristic 0017 describes), `predict_engine_tflops`, and `predict_applicable_kernels` — so an engine's
  distinct selection jobs, and per-architecture model choice, are expressible without a schema change
  later. `version` stays `1.0`: the UED is not yet consumed, so v1.0 is revised in place rather than
  bumped.

No other silent contradictions; this draft aims to formalize 0017. Any conflict surfaced during
review is recorded here.

## 3. Engine Identity

An engine lives in **two distinct id spaces**, which the UED keeps separate:

**(a) The descriptor UUID (`id`).** Every descriptor carries a stable UUID used only for
cross-references among descriptor files: a KDP names its UED by this id; a UED names its UHDs
and KMD by theirs (RFC 0017 § 4). It is internal to the descriptor graph and never crosses the
hipDNN library boundary.

**(b) The 64-bit engine id.** hipDNN identifies engines by a 64-bit id derived from a registered
engine **name**, an FNV-1a hash of the name (RFC 0017 § 4; [RFC 0003](0003_EngineIdDesign.md)).
A descriptor-backed engine hashes its UED `name` into this space exactly as a hand-written
engine does; this is the id the provider reports to the host, and what selection, diagnostics,
and support claims key on.

| Concern | Identifier |
|---|---|
| A KDP naming its UED; a UED naming its UHDs/KMD | descriptor UUID `id` |
| hipDNN selecting among engines; logs; support claims | 64-bit engine id (FNV-1a of `name`) |

The UED `name` is therefore load-bearing only where the engine surfaces outside the descriptor
graph (selection, logs, diagnostics, and the hash into the engine-id space); internally, the
UUID `id` binds. Names must be **globally unique** and should be scoped, e.g. `rocke:SDPA`.

## 4. The UED Schema

This section **defines** the UED schema (schema tag `hipdnn.ued/v1`): § 4.1 an example instance,
§ 4.2 the normative definition, § 4.3 serialization. Following RFC 0017's convention, a UED
carries two version-bearing fields: the `schema` type tag, constant across a major generation,
and a separate `version` field (`major.minor`) that the accept rule gates on (§ 11).

### 4.1 Example instance

```jsonc
{
  "schema":          "hipdnn.ued/v1",              // file-type tag (§ 4.2)
  "version":         "1.0",                        // major.minor; gated at load (§ 11)
  "id":              "efc9eae4-fe33-4cb0-a593-95d771dc13b2",  // UUID; referenced by KDPs (§ 3a)
  "name":            "rocke:attention_dense_fwd",  // globally-unique, scoped engine name (§ 3b)

  // Three role-scoped, arch-keyed UHD maps (§ 4.4). Each maps a gfx target — or the
  // "default" catch-all — to a UHD id. All three are optional; the common case sets
  // only sort_kernel_catalog.
  "sort_kernel_catalog": {                         // optional: kernel-selection heuristic, ranks the catalog
    "gfx950":  "ae896b07-80cd-473c-b3f4-6a8892998519",
    "gfx942":  "1f0c8d22-4b7e-49a1-9d3c-6e2a5f8b1074",
    "default": "c93e17aa-2d6b-4f10-8e75-3a9c04b6f2e1"
  },
  "predict_engine_tflops": {                       // optional: cheap engine-level estimate, used at engine selection
    "gfx950":  "7b1e9c40-5a2f-4d8b-91c6-0e3d7a2f6b58"
  },
  // "predict_applicable_kernels": { … }           // optional, future: generates the candidate set (§ 4.4)

  "metadata":        "9ae0b215-32a7-49d1-96df-e9b05e1927ea",  // one KMD id (required)
  "knobs":           ["split_k", "tile_m"],        // optional: KMD field names to expose (§ 5)
  "behavior_notes":  ["runtime_compilation"],      // optional (§ 6)
  "numerical_notes": ["tensor_core", "reduced_precision_reduction"]  // optional (§ 6)
}
```

### 4.2 Normative schema (version 1.0)

A conforming UED is a JSON object with exactly the members below. Members not listed are
rejected under the version rule (§ 11). The object has no logic; it is identity, one required
metadata reference, up to three role-scoped heuristic maps, and optional annotations.

**Field specification (normative).**

| Field | Req. | JSON type | Value constraints |
|---|---|---|---|
| `schema` | yes | string | The descriptor type tag; exact value `hipdnn.ued/v1`. Constant across the major-1 generation; names which descriptor kind this is. |
| `version` | yes | string | `<major>.<minor>` (both numeric), e.g. `1.0`. The compatibility field the accept rule gates on (§ 11). |
| `id` | yes | string | A UUID (RFC 4122) in canonical `8-4-4-4-12` hex form. Unique across all loaded descriptors, except that content-identical UEDs may share an `id` (§ 10.2.1). The cross-reference key a KDP's `engine` field uses (§ 3a). |
| `name` | yes | string | Globally-unique, scoped engine name matching `^[A-Za-z0-9_.-]+:[A-Za-z0-9_.-]+$` (a `namespace:local` form, e.g. `rocke:SDPA`). Hashed (FNV-1a, 64-bit) into the hipDNN engine-id space (§ 3b). Non-empty; unique by both literal name and by hash. |
| `sort_kernel_catalog` | no | object (arch → UUID) | The **kernel-selection heuristic** map (§ 4.4): each key is a gfx target (e.g. `gfx950`) or the literal `default`; each value a UHD UUID that must resolve at load (§ 10.2). This is the current main UHD — it ranks the catalog and picks the winning kernel. Absent => the engine has no kernel-selection model and falls back to deterministic ordering (RFC 0019 § 5). |
| `predict_engine_tflops` | no | object (arch → UUID) | Optional **engine-estimate** map (§ 4.4): a cheap `f(graph) → expected performance` UHD consulted at engine selection, before any catalog is ranked. Same arch-key form and resolution as above. |
| `predict_applicable_kernels` | no | object (arch → UUID) | Optional, **future** **candidate-generator** map (§ 4.4): a UHD that produces the applicable candidate set for a combinatorial/JIT space, which `sort_kernel_catalog` then ranks. Same arch-key form and resolution. |
| `metadata` | yes | string | UUID of this engine's one KMD. Must resolve to a loadable KMD at load (§ 10.2). |
| `knobs` | no | array of string | Each element is a field name declared in the referenced KMD (§ 5). No duplicates. Absent or `[]` => engine exposes no descriptor knobs. Every element must match a KMD field or it is a load error (§ 10.2). |
| `behavior_notes` | no | array of string | hipDNN behavior-note tags ([RFC 0010](0010_BehaviorNotes.md)). No duplicates. Absent => none. |
| `numerical_notes` | no | array of string | hipDNN numerical-note tags. No duplicates. Absent => none. |

All three heuristic maps and the three trailing optional fields may be omitted; a valid engine can
carry only its identity and `metadata`. A UED that names no `sort_kernel_catalog` orders its kernels
deterministically instead (RFC 0019 § 5), which is the state a pack ships in before a heuristic is
generated for it.

### 4.4 Heuristic roles and arch keying

A UED carries **three role-scoped heuristic maps**, because an engine has more than one distinct
selection job and they run at different points in the pipeline:

| Field | Role | When it runs | RFC 0019 |
|---|---|---|---|
| `predict_engine_tflops` | Cheap engine-level performance estimate | Engine selection, before any catalog is built | "engine estimate (A)" |
| `sort_kernel_catalog` | Ranks the catalog and picks the winning kernel | Kernel selection, after applicability | "config UHD / knob predictor (B)" — the current focus |
| `predict_applicable_kernels` | Generates the candidate set to be ranked | During applicability, for a combinatorial/JIT space | future candidate generator (C) |

The pipeline is therefore `predict_engine_tflops` (rank engines) → `predict_applicable_kernels`
(produce candidates, when present) → `sort_kernel_catalog` (rank and pick). Every role is **optional**,
including `sort_kernel_catalog`: an engine with none ranks deterministically (RFC 0019 § 5), and the
common trained AOT engine sets only `sort_kernel_catalog`. `predict_engine_tflops` earns its
place once engines compete or an opaque engine must report an estimate; `predict_applicable_kernels`
is reserved for the JIT case where no enumerable catalog exists until something produces one.

**Arch keying.** Each map's keys are gfx target names (matched exactly against the device's
`gcnArchName`), plus an optional `default` catch-all. Resolution for a role on a given device:

1. exact match on the device's gfx name, else
2. the `default` entry, else
3. the role is **unavailable** on that arch — `sort_kernel_catalog` unavailable means deterministic
   ordering (RFC 0019 § 5); an unavailable estimate or generator simply is not consulted.

The map chooses *which* model by architecture; it does not replace `$device.*` features. A single
mapped model still generalizes across a family (SKUs of one gfx) through its device features — the
map splits only where architectures genuinely diverge in metadata or heuristic behavior. One `default`
entry covering every arch is the degenerate, fully device-feature-driven case.

Each `major.minor` is a standalone JSON Schema **file** in the repository (§ 11.3); the inline copy
below mirrors the authoritative `ued/1.0.json`, and a CI check verifies the match.

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "ued/1.0.json",
  "title": "hipdnn.ued version 1.0",
  "type": "object",
  "additionalProperties": false,
  "required": ["schema", "version", "id", "name", "metadata"],
  "properties": {
    "schema": {
      "type": "string",
      "const": "hipdnn.ued/v1"
    },
    "version": {
      "type": "string",
      "const": "1.0"
    },
    "id": {
      "description": "This descriptor's own UUID. Unique across loaded UEDs except for content-identical duplicates (semantic; see RFC 0020 section 10.2.1).",
      "type": "string",
      "pattern": "^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
    },
    "name": {
      "type": "string",
      "pattern": "^[A-Za-z0-9_.-]+:[A-Za-z0-9_.-]+$"
    },
    "sort_kernel_catalog": {
      "description": "Arch -> UHD id. Kernel-selection heuristic; each value MUST resolve to a loadable UHD (semantic; see RFC 0020 section 10.2).",
      "type": "object",
      "propertyNames": { "type": "string", "minLength": 1 },
      "additionalProperties": {
        "type": "string",
        "pattern": "^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
      },
      "minProperties": 1
    },
    "predict_engine_tflops": {
      "description": "Arch -> UHD id. Optional engine-level estimate; each value MUST resolve to a loadable UHD (semantic; see RFC 0020 section 10.2).",
      "type": "object",
      "propertyNames": { "type": "string", "minLength": 1 },
      "additionalProperties": {
        "type": "string",
        "pattern": "^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
      },
      "minProperties": 1
    },
    "predict_applicable_kernels": {
      "description": "Arch -> UHD id. Optional, future candidate generator; each value MUST resolve to a loadable UHD (semantic; see RFC 0020 section 10.2).",
      "type": "object",
      "propertyNames": { "type": "string", "minLength": 1 },
      "additionalProperties": {
        "type": "string",
        "pattern": "^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
      },
      "minProperties": 1
    },
    "metadata": {
      "description": "Cross-reference: MUST resolve to a loadable KMD (semantic; see RFC 0020 section 10.2).",
      "type": "string",
      "pattern": "^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
    },
    "knobs": {
      "description": "Each entry MUST be a field name declared by the referenced KMD (semantic; see RFC 0020 section 10.2).",
      "type": "array",
      "items": { "type": "string", "minLength": 1 },
      "uniqueItems": true
    },
    "behavior_notes": {
      "type": "array",
      "items": { "type": "string", "minLength": 1 },
      "uniqueItems": true
    },
    "numerical_notes": {
      "type": "array",
      "items": { "type": "string", "minLength": 1 },
      "uniqueItems": true
    }
  }
}
```

The schema's `description` keywords flag which fields carry cross-references and what each must
resolve to. JSON Schema cannot *enforce* cross-file resolution, so these are recorded as
machine-readable annotations and checked semantically (§ 10.2); `additionalProperties: false`
makes any unknown field a hard rejection (§ 10.1).

The schema targets **Draft 7** (recommended, not required): off-the-shelf C++ JSON-Schema
validators target Draft 7, so one file can drive both the build-time and runtime checks without
a bespoke validator. The constructs used (`type`, `const`, `pattern`, `required`,
`additionalProperties`, `uniqueItems`) are common to Draft 7 and later dialects.

### 4.3 Serialization

The UED is authored and shipped as **JSONC** (JSON with comments), consistent with how RFC 0017
presents every descriptor. Comments are stripped before the UED is validated for its declared
version (§ 4.2, § 11.3).

## 5. Knobs

A knob is **a KMD field the engine chooses to expose**, a name and nothing more; the KMD
already declares the field's type and default (RFC 0017 § 4). The UED's contract:

- Only **KMD field names** may appear in `knobs`. A name no KMD field matches is a **load
  error** (§ 10.2).
- Exposing a field is **additive and reversible**: add a name to expose it, remove it to
  withdraw it.
- A knob's **legal values come from the catalog**, not the KMD's theoretical range; that is, the
  set of values the field takes among the kernels matching a given graph.
- A knob's **default is the kernel-selection heuristic's top-ranked choice** (`sort_kernel_catalog`,
  § 4.4), not a constant.
- `knobs` governs only what the UED *declares*. hipDNN's reserved `global.` knobs
  ([RFC 0004](0004_EngineConfigKnobs.md)) are a separate namespace a descriptor-backed engine
  implements like any other engine; the two do not overlap.

## 6. Behavior and Numerical Notes

`behavior_notes` and `numerical_notes` are hipDNN's existing per-engine annotations
([RFC 0010](0010_BehaviorNotes.md)) carried on the UED. They are optional string lists. This
RFC adds no new note vocabulary; it specifies only that the UED is where a descriptor-backed
engine declares them.

## 7. Engine Membership

A **UKD names no engine.** Its engine membership is determined by the **sibling UED referenced
by its KDP**: the KDP carries `"engine": "<UED id>"`, and every child UKD inherits it, along
with the pack's matchers and dispatch and the engine's heuristics and metadata schema (RFC 0017
§ 4). The membership chain is **UKD -> KDP -> UED**, bound by the descriptor UUID `id`; there is no
direct UKD->UED reference.

One UED is typically shared by many KDPs, and so serves many UKDs: one engine, one metadata
schema, and up to three role-scoped heuristics (§ 4.4), ranking a whole catalog of kernels over
one feature space.

## 8. When a UED Is Loaded and Registered

The loading *mechanism* (discovery, parsing, the shared load path, and when descriptor bodies are
parsed) is out of scope. What a loaded UED must satisfy to be valid is specified in **§ 10**
(structural checks, cross-descriptor resolution, and uniqueness). This section fixes only the
loading facts the UED's host contract depends on: the ingestion paths (§ 8.1), the one
registration-timing guarantee (§ 8.2), and concurrency (§ 8.3).

### 8.1 Ingestion paths

A UED may reach the provider by either RFC 0017 ingestion path, build-time (AOT) or runtime
drop-in (§ 12), and both converge on the same UED schema, validation (§ 10), and registration
(§ 9) defined here. A load always builds the **complete** descriptor set from scratch rather than
merging into an in-memory set, so every UED is validated against every other on each load
(§ 10.2.1); a drop-in descriptor is picked up by the next such load. What triggers a load, where
descriptors live, and how they are discovered are part of the loading mechanism and are out of
scope for this RFC.

### 8.2 Registration timing

The one timing guarantee this RFC fixes is that every valid UED is registered (name -> 64-bit id,
§ 9) before any graph is served, because the host must be able to enumerate the engines it may
select among. Registration of the loaded UED set therefore completes before engine enumeration or
selection begins. A KDP's `engine` reference correspondingly resolves against an
already-registered UED; how a provider orders UED and KDP loading to guarantee that is a mechanism
detail.

### 8.3 Concurrency

Engine registration occurs on the provider's plugin-load path, which is already serialized, so it
introduces no new concurrency model and requires no additional locking beyond that existing
serialization.

## 9. Engine Registration

**Registration** is the process that turns a validated UED into an engine the provider exposes
to hipDNN. The destination is the provider's existing engine list (the one that already holds
hand-written engines), and registration is the descriptor-driven equivalent of the provider's
hand-written engine-registration path. For each UED that passes validation (§ 10), registration:

1. **Derives the engine id**: the 64-bit hash of the UED `name` (§ 3).
2. **Instantiates one generic engine**: a single engine implementation that satisfies hipDNN's
   existing engine contract from descriptor data rather than hand-written code, one instance per
   UED, bound to that UED's descriptors: its heuristic maps (`sort_kernel_catalog` and any
   `predict_engine_tflops` / `predict_applicable_kernels`; § 4.4) and its `metadata` (KMD)
   reference, and the KDPs whose `engine` field names it.
3. **Adds the engine to the provider's engine list** and records the name -> id mapping, so the
   host can enumerate the engine and diagnostics / support claims
   ([RFC 0015](0015_EngineSupportClaims.md)) key on the real name rather than a hex id.

Nothing in the host-facing engine contract changes: a descriptor-backed engine is selected and
driven exactly as a hand-written one (RFC 0017 § 3, § 8). This RFC specifies registration (how
a validated UED becomes an exposed engine) and the generic engine's identity and descriptor
binding. Populating the generic engine's plan builder is registration's responsibility; the plan
builder's *internal* behavior over UDD and UKD data is defined by those descriptors' own
specifications, not this one.

## 10. Validation

Every check below is expected to run at **both build time and run time**. Build-time validation
catches errors before shipping; run-time validation ensures integrity of the loaded set and is
**required** for drop-in UEDs (§ 8), which never pass through the build. The checks divide into
**structural** (the field contract of § 4.2) and **semantic** (cross-descriptor); the
implementation may fold them into one pass.

### 10.1 Structural validation

The UED's fields are checked against the contract for its declared `version` (§ 4.2, § 11.3):
required fields present, each field's type and pattern, the `schema`/`version` constants, and
**any unknown field is a hard rejection** (not a forward-compat case, since an author stamps the
lowest version the UED needs, § 11.2). The contract applied is the one for the declared version,
not the runtime's latest (§ 11.3). The per-version JSON Schema file of § 4.2 is the recommended way
to express and run this check (`additionalProperties: false` gives the unknown-field rejection),
but the contract is normative independently of that mechanism.

### 10.2 Semantic validation (cross-descriptor)

These cannot be expressed in JSON Schema because they depend on other descriptors; each is
performed at build time and run time alike:

- **Reference resolution.** A UED's `metadata` (KMD) and every UHD id in its three heuristic maps
  (`sort_kernel_catalog`, `predict_engine_tflops`, `predict_applicable_kernels`; § 4.4) must each
  resolve to a loadable descriptor of the correct kind; a dangling reference is an error. This is an
  *existence* condition: the referent must be resolvable, not necessarily parsed. An empty map is a
  load error (a role is either absent or non-empty).
- **`knobs` must be a subset of KMD field names.** A knob name no KMD field matches is an error
  (RFC 0017 § 4). Unlike reference resolution, this reads the KMD's declared field set, so the
  referenced KMD must be resolvable **to its field set**, more than existence.
- **Uniqueness (§ 10.2.1).** No two loaded UEDs may share a descriptor `id`, and independently
  none may share a `name`, except for the content-identical case (§ 10.2.1).

The full cross-descriptor reference-integrity check (which references must resolve, and to what)
spans multiple descriptor types and is best specified at a higher level than the UED format.
This RFC fixes only that a UED's own heuristic-map and `metadata` references are subject to it, at
both build and run time.

#### 10.2.1 Duplicate detection (descriptor `id` and `name`)

Duplicate detection runs over the complete loaded set (§ 8.1), so a drop-in UED is checked against
every other UED exactly as any other.

**The version check precedes duplicate detection.** A UED whose `version` the runtime does not
support is logged (warning) and dropped by the accept rule (§ 11.1) *before* uniqueness is
evaluated, so it never participates in `id` or `name` detection. A UED with a duplicate `id` or
`name` but an unsupported `version` is therefore dropped for its version alone, and the UEDs it
would have conflicted with are retained (unless they separately conflict with other
supported-version duplicates).

Two independent uniqueness invariants hold across all loaded UEDs: no two share a descriptor
`id`, and no two share a `name` (equivalently its 64-bit hash). On collision under either
invariant, **all** UEDs involved are unloaded (logged, § 12), not keep-the-first, with one
exception below.

**Content-identical exception (`id` only).** UEDs that share a descriptor `id` are **not** dropped
if they are **semantically identical**: the same set of fields with the same values after
parsing, independent of comments, whitespace, and key order. Identical duplicates are one
definition expressed more than once, so which copy binds is immaterial; the provider keeps a single
instance and loads normally. If any field or value differs between UEDs sharing an `id`, they are
treated as a genuine collision and **all** are dropped. This exception applies only to the `id`
invariant; a `name` collision between non-identical UEDs is always a drop.

> Content-identical duplicates arise legitimately when a generic engine is shipped per
> architecture: the same UED is packaged with each architecture's descriptors and so appears more
> than once with one `id`.

Dropping all rather than keeping one (when they differ) is required because descriptor load order
is **not deterministic**: keep-the-first would leave it ambiguous which definition an engine bound
to. Dropping every participant makes the outcome unambiguous: the conflicting engine simply does
not load, and diagnostics name every descriptor involved. (RFC 0017 § 4 detects a duplicate
name/hash but specifies neither drop-all, the independent `id` check, nor the content-identical
exception.)

### 10.3 UED-load vs KDP-load boundary (guidance)

The engine-scoped checks above (§ 10.1-10.2) belong at **UED load**. For contrast, these are
**KDP-load** (pack-scoped) concerns, governed by the KDP's own specification and listed only to
place the boundary:

- the KDP's `engine`, `matchers`, and `dispatch` references resolve (the `engine` ref
  resolving to a registered UED is a KDP-load concern, not a UED concern);
- per-kernel `$kernel.*` matcher pins against the KMD (RFC 0017 § 5);
- **duplicate kernel keys**, engine-wide but triggered by loading a pack's UKDs, so detected
  at KDP/UKD load (RFC 0017 § 10).

This boundary is guidance, not a hard split; a shared loader may fold both into one pass.

## 11. Versioning and Compatibility

A UED's compatibility is carried by its `version` field (`major.minor`); the `schema` tag names
the descriptor kind and is constant within a major generation. Each descriptor file type versions
independently (a KMD and a UDD advance on their own schedules). This section defines the accept
rule, what `major` and `minor` are permitted to mean, and how per-version schema files back
validation.

> **Tightens RFC 0017 § 4 (compat mechanism).** RFC 0017 gives the accept/reject *policy* and a
> coarse field-evolution rule but leaves the unknown-field policy and the runtime's version source
> unspecified. This section keeps 0017's accept rule unchanged and pins down the rest; it does not
> override 0017, it makes the deferred detail concrete.

### 11.1 The accept rule

The runtime carries a supported `major.minor` for the UED type; concretely, the version(s) of
the UED schema file(s) embedded in the provider build (§ 11.3). A UED is accepted iff:

- **`file.major == provider.major`**, and
- **`file.minor <= provider.minor`**.

`major` and `minor` are compared as integers, not lexicographically or as a decimal fraction, so
`1.10` is newer than `1.9`. Otherwise the UED is rejected. A UED is refused, never silently
reinterpreted.

This yields **backward compatibility only**: an older-minor UED loads on a newer-minor provider;
a newer-minor UED is rejected on an older-minor provider (it may carry fields that runtime cannot
interpret). A **major mismatch is a hard break**: at this time the runtime supports **exactly one
major version**, and *every* UED of a different major (older or newer) is dropped (logged and
skipped, § 12). There is no multi-major support; a major bump orphans existing UEDs until they
are re-authored, which for a large descriptor set is expected to be a bulk re-emit from the
authoring toolchain rather than per-file hand-editing. Because major bumps should be rare
(§ 11.2), this is an accepted cost, revisited only if descriptor longevity across a break becomes
a requirement.

### 11.2 What `major` and `minor` are permitted to mean

- **Minor bump: additive, absence-safe changes only.** A minor may add a field **only if its
  absence is well-defined as "the behavior before the field existed"**: a UED at an earlier minor
  omits it, and the runtime reproduces prior behavior via the pre-addition code path (the
  semantics of absence *is* the old implementation, so no default table is needed). A minor may
  not remove, rename, retype, or change the meaning or permitted values of an existing field.
- **Major bump: everything else** (removing, renaming, retyping, making an optional field
  required, or changing a field's meaning/permitted values). These are the changes where an old
  reader would misinterpret a file, which the hard `major ==` break (§ 11.1) prevents.

**Authors stamp the lowest version their UED needs**, so it stays loadable on the oldest runtime
that can serve it and never carries a field its version does not define. The UED's version should
move rarely.

### 11.3 Per-version schema files (a supporting design, not a mandate)

The structural contract for each `major.minor` is expressed as a standalone JSON Schema file in
the repository (`ued/1.0.json` is authoritative; § 4.2 reproduces it), pinning that version's
`version` property with `const` (e.g. `"const": "1.0"`). A new version is a new file. This design
is **chosen to support**, not to require, a single source of truth that can drive validation in
two places from one artifact:

- **Build time**: the authoring toolchain can validate every authored UED.
- **Run time**: the provider can carry the schema file(s) it supports and run the same
  validation on ingested UEDs, since drop-ins (§ 8) bypass the build.

Because each file pins its own `version`, a UED is validated against the file for its declared
version, so a `1.0` UED carrying a `1.1`-only field is rejected even on a `1.1` runtime. This RFC
recommends the file-per-version approach so an off-the-shelf validator can enforce the structural
contract; it does **not** prescribe how the schema is carried in the provider, which validator is
used, or that an implementation must use JSON Schema at all.

### 11.4 Minimum-version checking (a supporting design, not a mandate)

§ 11.2 requires authors to stamp the **lowest** version their UED needs. This RFC does not require
automated enforcement of that rule, but the per-version file design (§ 11.3) supports a build-time
lint for implementations that want one.

Such a lint validates a UED declaring `<major>.<m>` against the schema files of the **same major**
from the oldest minor up to `m`; the oldest that accepts it is the true minimum, and a minimum
below the declared minor is an over-declaration. The search is confined to the declared major
because a major mismatch is a hard break (§ 11.1).

## 12. Lifecycle and Operational Policy

- **Load failure => log and skip.** A UED that fails validation is **logged as an error and
  skipped**; the provider does not hard-fail, and the skipped UED registers no engine. This
  matches RFC 0017's "reported in load diagnostics like any other exclusion" and the
  duplicate-key "logged and dropped" pattern.
- **Concurrency => the guarded plugin-load path** (§ 8.3), not a per-handle resource manager.
  Engine registration adds no new concurrency model.
- **`HIPDNN_DISABLE_ENGINES` => skip at load.** A disabled engine is skipped before registration:
  it never loads and never claims its name or id. A list entry may be any of three identifiers
  (the UED `name`, its 64-bit hash, or the UED UUID `id`), and the matcher checks all three.
  Because a skipped UED never claims its name or id, this can also serve as a recovery lever for a
  collision (§ 10.2.1): disabling the unwanted participant by an identifier that singles it out
  (for a same-`name` collision, its `id`) frees the contested name/id so the other loads. A
  collision whose participants share both `name` and `id` cannot be separated this way. The
  finer-grained `HIPDNN_DISABLE_KDPS` / `HIPDNN_DISABLE_UKDS` (RFC 0017 § 10) are governed by their
  own descriptors.
- **Drop-in trust => out of scope.** Trust and enablement for untrusted drop-in descriptors are
  out of scope (RFC 0017 § 14); this RFC adds no trust policy.

## 13. Testing

Reusing the existing tiers (RFC 0006 harness) and RFC 0017 § 14.1's descriptor-pipeline
fuzzing, this RFC adds UED-specific coverage.

**Unit tests:**

- **Structural validation** (§ 10.1): valid and invalid field sets; missing required fields
  (including `schema` and `version`); malformed `id`/`name` patterns; wrong `schema`/`version`
  constant; **unknown field is rejected**; optional fields absent.
- **Schema build/runtime parity** (if the per-version file design of § 11.3 is used): the schema
  file embedded in the provider parses to the same JSON as the repository's canonical file and as
  the inline copy in § 4.2 (a CI check comparing parsed structure, not bytes, so formatting and
  comments do not matter), so the build-time and runtime validators enforce the same contract.
- **Per-version validation** (§ 11.3): a UED whose `version` is `1.0` that carries a `1.1`-only
  field is **rejected** even on a `1.1` runtime (validated against its declared version, not the
  latest).
- **Version accept rule** (§ 11.1): matrix of `file` vs `provider` `major.minor` read from the
  `version` field: same major/older-or-equal minor loads; newer minor rejected; any major mismatch
  dropped.
- **Semantic checks** (§ 10.2): a dangling `metadata` or heuristic-map (`sort_kernel_catalog`,
  `predict_engine_tflops`, `predict_applicable_kernels`) reference; a `knobs` entry absent from
  the KMD.
- **Duplicate detection, drop-all** (§ 10.2.1): two UEDs differing in content but sharing an
  `id`; two sharing a `name`; two sharing both; a UED colliding by name with a built-in engine. In
  each case *every* colliding UED is dropped and named in diagnostics, and no engine is left bound
  to an arbitrarily-chosen definition.
- **Content-identical duplicates load** (§ 10.2.1): two UEDs with the same `id` and semantically
  identical fields/values (differing only in comments, whitespace, or key order) load as a single
  engine (not dropped); the same `id` with any differing field value is dropped as a collision.
- **Engine-id derivation**: the generic engine's `id()` equals `engineNameToId(name)` (FNV-1a)
  for representative names, including scoped names like `rocke:SDPA`.
- **`HIPDNN_DISABLE_ENGINES`**: an engine disabled by name, by id-hash, and by UUID is skipped
  before registration and frees its name.

**Integration tests:**

- A UED loads and the engine appears correctly registered in the hipDNN engine-id space at
  plugin load, reportable through `getAllEngineIds`.
- A KDP's `engine` reference resolves to a previously-registered UED.
- End-to-end: UED loads => engine is selectable through the `IEngine` lifecycle
  (`isApplicable` ... `initializeExecutionContext`).
- Disabling an engine that would collide by name lets the provider load (collision-recovery).
- A drop-in UED is validated at runtime by the same version contract as the build-time check
  (same accept/reject outcome for a matched pair of good/bad descriptors).

The descriptor pipeline parses untrusted input on the drop-in path, so the loader, parser, and
(future) validators run under the existing ASAN build with a seed corpus, per RFC 0017 § 14.1.

## 14. Glossary

- **UED (Universal Engine Descriptor):** one engine, comprising a stable identity (`name` + UUID
  `id`), the KMD field names it exposes as knobs, and its behavior/numerical notes. Names its one KMD
  and up to three role-scoped, arch-keyed UHD maps (§ 4.4) by id. 1:1 with a hipDNN engine.
- **Engine id (64-bit):** the hipDNN-facing engine identifier, derived (FNV-1a) from the UED
  `name`; what the plugin reports to the backend and what selection/diagnostics key on.
- **Descriptor UUID `id`:** the cross-reference identifier a descriptor carries; how a KDP
  names its UED and a UED names its UHDs/KMD. Distinct from the engine id.
- **Generic engine:** the single C++ engine class that satisfies hipDNN's `IEngine` contract
  from descriptor data, one instance per UED.

---

## 15. Appendix: Fully-Populated UED Examples

The examples in § 4 keep optional fields minimal. This appendix shows UEDs that populate **all**
optional fields, which requires knowing the KMD they reference, since every `knobs` entry must
name a field the KMD declares. § A.1 sketches only as much of the KMD as the UED depends on; the
KMD format itself is outside this RFC's scope.

### A.1 What a UED needs from its KMD

A UED does not define fields. It references one KMD by id (its `metadata`) and exposes a subset
of that KMD's field **names** through `knobs`. For the purposes of a UED, a KMD is simply a
named list of field names an engine's kernels vary over:

```jsonc
{
  "schema":  "hipdnn.kmd/v1",                          // KMD's own tag form; defined by the KMD RFC
  "version": "1.0",
  "id":      "9c53b6b0-9a1e-4b1d-8b5c-7e2d9a6f3c40",   // the UED's "metadata" names this
  "name":    "attention_dense variant fields",
  "fields": [ /* ... field definitions ... */
    {"name": "block_n"},        // a UED may expose this as a knob
    {"name": "waves_per_eu"},   // ...and this
    {"name": "num_persistent"}  // ...and this
    // (plus the engine's other fields: head_size, dtype, seqlen_q, persistent, ...)
  ]
}
```

The only fact the UED depends on is the **set of field names** the KMD declares: a `knobs` entry
matching one is valid; one matching nothing is a load error (§ 10.2). Field types, defaults, and
catalog semantics are KMD concerns (RFC 0017 § 4-5), not properties the UED reads.

### A.2 Fully-populated UED: the `attention_dense` engine

A UED that populates **all** optional fields. Each name in `knobs` is a field the referenced KMD
declares (§ A.1); the notes are RFC 0010 annotations.

```jsonc
{
  "schema":          "hipdnn.ued/v1",
  "version":         "1.0",
  "id":              "7d4c2a9e-3b6f-4e1a-8c5d-9a2f7b0e6c14",   // UUID; KDPs name this via "engine"
  "name":            "rocke:attention_dense_fwd",              // globally-unique, scoped; hashed to the 64-bit engine id
  "sort_kernel_catalog": {                                     // this engine's kernel-selection heuristic (§ 4.4)
    "gfx950":  "2b7a4e1c-6f3d-4a8e-9c2b-5d1f0a7e8b93",
    "default": "8e0d5f31-7c2a-4b16-93e8-1a6f4d902c57"
  },
  "predict_engine_tflops": { "gfx950": "5c9a1b73-2e4f-48d0-86b1-9f3e7a0c62d4" },
  "metadata":        "9c53b6b0-9a1e-4b1d-8b5c-7e2d9a6f3c40",   // the KMD (§ A.1)
  "knobs":           ["block_n", "waves_per_eu", "num_persistent"],  // all are KMD field names
  "behavior_notes":  ["runtime_compilation"],
  "numerical_notes": ["tensor_core", "reduced_precision_reduction"]
}
```

This exposes three KMD fields as knobs and carries two note lists. A KDP joins the engine with
`"engine": "7d4c2a9e-3b6f-4e1a-8c5d-9a2f7b0e6c14"`.

### A.3 A minimal engine, for contrast (all optional fields omitted)

The same engine with only required fields, no knobs, no notes:

```jsonc
{
  "schema":    "hipdnn.ued/v1",
  "version":   "1.0",
  "id":        "7d4c2a9e-3b6f-4e1a-8c5d-9a2f7b0e6c14",
  "name":      "rocke:attention_dense_fwd",
  "sort_kernel_catalog": { "default": "2b7a4e1c-6f3d-4a8e-9c2b-5d1f0a7e8b93" },
  "metadata":  "9c53b6b0-9a1e-4b1d-8b5c-7e2d9a6f3c40"
}
```

Both load and register identically; § A.2 only surfaces the extra heuristic roles, knobs, and
notes that § A.3 leaves unset. A single `default` entry serves every arch through device features
(§ 4.4).
