# RFC 0020: The Universal Engine Descriptor (UED)

- Contributors: TBD
- **Status**: Draft
- **Implementation Version**: TBD
- **Follows**: [RFC 0017](0017_UniversalKernelDescriptor.md) (Universal Kernel Descriptors) -- this is the "UED + engine registry" follow-up named in RFC 0017 § 14.2.

## Table of Contents

1. [Summary](#1-summary)
2. [Relationship to RFC 0017](#2-relationship-to-rfc-0017)
3. [Engine Identity](#3-engine-identity)
4. [The UED Format](#4-the-ued-format)
5. [Knobs](#5-knobs)
6. [Behavior and Numerical Notes](#6-behavior-and-numerical-notes)
7. [Engine Membership (UKD -> KDP -> UED)](#7-engine-membership)
8. [When a UED Is Loaded and Registered](#8-when-a-ued-is-loaded-and-registered)
9. [The Engine Registry](#9-the-engine-registry)
10. [Validation](#10-validation)
11. [Versioning and Compatibility](#11-versioning-and-compatibility)
12. [Lifecycle and Operational Policy](#12-lifecycle-and-operational-policy)
13. [Testing](#13-testing)
14. [Glossary](#14-glossary)
15. [Appendix: Fully-Populated UED Examples](#15-appendix-fully-populated-ued-examples)

---

## 1. Summary

RFC 0017 established a family of declarative descriptors that one **generic engine** loads,
matches, selects, and launches with no new C++. It described each descriptor at a framing
level and deferred the detailed format of each to its own follow-up. This RFC specifies the
**Universal Engine Descriptor (UED)** -- the descriptor that defines one engine's identity,
the KMD fields it exposes as knobs, and its behavior and numerical notes -- together with the
registry that turns a UED into an engine the hipDNN library can select.

The **descriptor-loading pipeline** -- how descriptor files are discovered, parsed, and the
shared load path all descriptor kinds use -- is out of scope for this RFC, which specifies only
the loading *behavior* its own rules depend on: principally *when* a UED is loaded and registered
(§ 8), and what must hold at that time. The loading *mechanism* is left to the provider
implementation.

Concretely, this RFC delivers:

- The **UED field schema**, defined as a versioned JSON Schema **file** that is the single
  source of truth for both build-time and runtime validation, plus serialization (§ 4, § 11.3).
- The **engine-identity model**, including the two distinct id spaces a descriptor engine
  lives in -- the descriptor-cross-reference GUID and the hipDNN 64-bit engine id (§ 3).
- The **engine registry** that instantiates the generic engine from UED data (§ 9), and the
  load/registration **timing** the UED's rules hinge on (§ 8).
- The **validation contract** -- structural (schema file, build + runtime) and semantic
  (cross-descriptor, including drop-all duplicate detection) -- with guidance on the UED-vs-KDP
  boundary (§ 10).
- **Versioning & compatibility** -- the accept rule, the constrained meaning of `major`/`minor`,
  and the schema-file mechanism (§ 11); plus **lifecycle/operational policy** (load-failure,
  drop-in reload, concurrency, the `HIPDNN_DISABLE_ENGINES` opt-out) and **test scope**
  (§ 12-13).

The UED's substance -- identity, knobs, notes, membership -- was largely pre-decided in RFC
0017 § 2 and § 4; this RFC formalizes it. The new design is the UED format, the engine registry,
and the validation/versioning contract.

**Out of scope.** Tags (an item some early ticket text listed) are **deferred**: RFC 0017
defines no `tags` field, its intent is unclear, and it blocks nothing here. Drop-in **trust
and enablement** rules for untrusted descriptor files remain out of scope, as in RFC 0017 § 14;
this RFC adds no trust policy.

## 2. Relationship to RFC 0017

This RFC lives alongside RFC 0017; some sections restate 0017 material (identity, knobs, the
reference model) so the UED format reads standalone. Where a decision here diverges from 0017,
**this RFC is the source of truth for UED matters**, and the divergence is called out with an
`Overrides RFC 0017 § X` note. The running list, for a later editor to fold back into 0017:

- **Schema tag format (§ 4.2).** 0017 stamps descriptor tags as `hipdnn.<type>/v1`, but its
  `major.minor` versioning rule can't evaluate a tag with no minor. This RFC fixes the UED tag
  to `hipdnn.ued/1.0` and recommends the same `major.minor` form for the other descriptors.
- **Compatibility mechanism (§ 11).** 0017 § 4 states the accept/reject *policy* but leaves the
  field-evolution contract, the unknown-field policy, and the runtime's version source
  unspecified. This RFC supplies all three (constrained minor bumps, hard-reject unknown fields,
  schema-file-backed runtime version) without changing 0017's accept rule.
- **Concurrency framing (§ 8.3).** Clarifies (does not contradict) that registration is guarded
  by the plugin-load path, not a per-handle resource manager.
- No other silent contradictions; this draft aims to formalize 0017. Any conflict surfaced
  during review is recorded here.

## 3. Engine Identity

An engine lives in **two distinct id spaces**, which the UED keeps separate:

**(a) The descriptor GUID (`id`).** Every descriptor carries a stable GUID used only for
cross-references among descriptor files -- a KDP names its UED by this id; a UED names its UHD
and KMD by theirs (RFC 0017 § 4). It is internal to the descriptor graph and never crosses the
hipDNN library boundary.

**(b) The 64-bit engine id.** hipDNN identifies engines by a 64-bit id derived from a registered
engine **name** -- an FNV-1a hash of the name (RFC 0017 § 4; [RFC 0003](0003_EngineIdDesign.md)).
A descriptor-backed engine hashes its UED `name` into this space exactly as a hand-written
engine does; this is the id the provider reports to the host, and what selection, diagnostics,
and support claims key on.

| Concern | Identifier |
|---|---|
| A KDP naming its UED; a UED naming its UHD/KMD | descriptor GUID `id` |
| hipDNN selecting among engines; logs; support claims | 64-bit engine id (FNV-1a of `name`) |

The UED `name` is therefore load-bearing only where the engine surfaces outside the descriptor
graph (selection, logs, diagnostics, and the hash into the engine-id space); internally, the
GUID `id` binds. Names must be **globally unique** and should be scoped, e.g. `rocke:SDPA`.

## 4. The UED Format

This section **defines** the `hipdnn.ued/1.0` schema: § 4.1 an example instance, § 4.2 the
normative definition, § 4.3 serialization.

> **Overrides RFC 0017 § 4 (tag format).** 0017's examples stamp the tag as `hipdnn.ued/v1`, but
> its versioning rule is `major.minor` and `v1` has no minor for the accept rule (§ 11) to
> evaluate. This RFC uses the explicit `hipdnn.<type>/<major>.<minor>` form; read `/v1` in 0017
> as `/1.0`, and apply the same form to the other descriptor tags in their follow-ups.

### 4.1 Example instance

```jsonc
{
  "schema":          "hipdnn.ued/1.0",             // file-type + version tag (§ 4.2, § 11)
  "id":              "efc9eae4-fe33-4cb0-a593-95d771dc13b2",  // GUID; referenced by KDPs (§ 3a)
  "name":            "rocke:attention_dense_fwd",  // globally-unique, scoped engine name (§ 3b)
  "heuristic":       "ae896b07-80cd-473c-b3f4-6a8892998519",  // one UHD id (required)
  "metadata":        "9ae0b215-32a7-49d1-96df-e9b05e1927ea",  // one KMD id (required)
  "knobs":           ["split_k", "tile_m"],        // optional: KMD field names to expose (§ 5)
  "behavior_notes":  ["runtime_compilation"],      // optional (§ 6)
  "numerical_notes": ["tensor_core", "reduced_precision_reduction"]  // optional (§ 6)
}
```

### 4.2 Normative schema: `hipdnn.ued/1.0`

A conforming UED is a JSON object with exactly the members below. Members not listed are
rejected under the version rule (§ 11). The object has no logic -- it is identity, two required
references, and optional annotations.

**Field specification (normative).**

| Field | Req. | JSON type | Value constraints |
|---|---|---|---|
| `schema` | yes | string | Exact form `hipdnn.ued/<major>.<minor>`; for this version, `major`=1. Governs accept/reject (§ 11). |
| `id` | yes | string | A UUID (RFC 4122) in canonical `8-4-4-4-12` hex form. Globally unique across all loaded descriptors; the cross-reference key a KDP's `engine` field uses (§ 3a). |
| `name` | yes | string | Globally-unique, scoped engine name matching `^[A-Za-z0-9_.-]+:[A-Za-z0-9_.-]+$` (a `namespace:local` form, e.g. `rocke:SDPA`). Hashed (FNV-1a, 64-bit) into the hipDNN engine-id space (§ 3b). Non-empty; unique by both literal name and by hash. |
| `heuristic` | yes | string | UUID of this engine's one UHD. Must resolve to a loadable UHD at load (§ 10.2). |
| `metadata` | yes | string | UUID of this engine's one KMD. Must resolve to a loadable KMD at load (§ 10.2). |
| `knobs` | no | array of string | Each element is a field name declared in the referenced KMD (§ 5). No duplicates. Absent or `[]` => engine exposes no descriptor knobs. Every element must match a KMD field or it is a load error (§ 10.2). |
| `behavior_notes` | no | array of string | hipDNN behavior-note tags ([RFC 0010](0010_BehaviorNotes.md)). No duplicates. Absent => none. |
| `numerical_notes` | no | array of string | hipDNN numerical-note tags. No duplicates. Absent => none. |

All three optional fields may be omitted; a valid engine can expose no knobs and carry no notes.

Each `major.minor` of the schema is a standalone JSON Schema **file** in the repository -- the
authoritative, machine-checkable definition of that version and the single source of truth. The
inline copy below mirrors it (CI verifies the match); the same file validates UEDs at both build
time and runtime (§ 11.3). Authoritative file: `ued/1.0.json`.

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "hipdnn.ued/1.0",
  "title": "hipdnn.ued/1.0",
  "type": "object",
  "additionalProperties": false,
  "required": ["schema", "id", "name", "heuristic", "metadata"],
  "properties": {
    "schema": {
      "type": "string",
      "pattern": "^hipdnn\\.ued/1\\.[0-9]+$"
    },
    "id": {
      "description": "This descriptor's own GUID. Must be unique across loaded UEDs (semantic; see RFC 0020 section 10.2).",
      "type": "string",
      "pattern": "^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
    },
    "name": {
      "type": "string",
      "pattern": "^[A-Za-z0-9_.-]+:[A-Za-z0-9_.-]+$"
    },
    "heuristic": {
      "description": "Cross-reference: MUST resolve to a loadable UHD (semantic; see RFC 0020 section 10.2).",
      "type": "string",
      "pattern": "^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
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
a bespoke validator. The constructs used (`type`, `pattern`, `required`, `additionalProperties`,
`uniqueItems`) are common to Draft 7 and later dialects.

### 4.3 Serialization

The UED is authored and shipped as **JSONC** (JSON with comments), consistent with how RFC 0017
presents every descriptor. Comments are stripped before validation against the schema file for
its declared version (§ 4.2, § 11.3). The AOT path may additionally pack descriptors into a
binary form carrying the identical fields; the binary form is a loader detail (§ 8) that
validates against the same schema.

## 5. Knobs

A knob is **a KMD field the engine chooses to expose** -- a name and nothing more; the KMD
already declares the field's type and default (RFC 0017 § 4). The UED's contract:

- Only **KMD field names** may appear in `knobs`. A name no KMD field matches is a **load
  error** (§ 10.2).
- Exposing a field is **additive and reversible**: add a name to expose it, remove it to
  withdraw it.
- A knob's **legal values come from the catalog**, not the KMD's theoretical range: the set
  of values the field takes among the kernels matching a given graph.
- A knob's **default is the heuristic's top-ranked choice**, not a constant.
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
with the pack's matchers and dispatch and the engine's heuristic and metadata schema (RFC 0017
§ 4). The membership chain is **UKD -> KDP -> UED**, bound by the descriptor GUID `id`; there is no
direct UKD->UED reference.

One UED is typically shared by many KDPs, and so serves many UKDs -- one engine, one UHD, one
KMD, ranking a whole catalog of kernels over one feature space.

## 8. When a UED Is Loaded and Registered

This section specifies the loading *behavior* the UED's rules depend on -- engine-id
registration, KDP `engine` resolution, `HIPDNN_DISABLE_ENGINES` skip-at-load, and reference
validation all hinge on **when** a UED is loaded relative to other descriptors. The loading
*mechanism* (discovery, parsing, the shared load path) is a provider implementation concern.

### 8.1 Two-phase timing: engine identity eager, descriptor bodies lazy

Loading is on demand and cached (RFC 0017 § 3): descriptor *bodies* are not bulk-parsed in
dependency order. Engine *identity* is the exception, because the host must know what engines
exist before it can select among them:

- **At plugin load -- eager, for every UED.** Each UED is parsed, validated (§ 10), and its
  engine registered (name -> 64-bit id, § 9), so the host can enumerate it. This completes for
  **all** UEDs before any graph is served.
- **Later -- lazy, per graph.** A KDP's body (matchers and UKD vector), the UHD, the UDD, and
  kernel sources load only when a graph needs that engine's catalog (RFC 0017 § 8.1).

**A UED's reference validation (§ 10.2) requires only that its referents *exist*, not that they
be fully parsed.** Confirming that a UED's `heuristic` (UHD) and `metadata` (KMD) resolve needs
just the set of loadable UHD and KMD ids -- the descriptor inventory, not the descriptor bodies.
So if loading makes that inventory of UHD and KMD ids available before a UED is validated, **the
UED owns its own reference check**: it verifies its `heuristic` and `metadata` ids appear in the
inventory. This is the load-ordering requirement the RFC fixes -- the referent inventory precedes
UED validation; the full ordering across all descriptor kinds is a provider implementation
concern. Likewise, a KDP's `engine` reference resolves against an already-registered UED, which
holds because engine registration is eager (above) and KDP bodies load later.

### 8.2 Discovery locations

Descriptors are discovered from two roots, both feeding the same parse/validate path:

- **AOT-shipped descriptors** install in a subfolder beneath the plugin library's directory, and
- **Drop-in descriptors** are placed in their own folder and picked up on the next plugin
  (re)load (§ 12).

Both roots must be supported. Exact folder names and the discovery mechanism are a provider
implementation concern.

### 8.3 Concurrency

Descriptor discovery and engine registration occur on the provider's plugin-load path, which is
already serialized; UED loading introduces no new concurrency model and requires no additional
locking beyond that existing serialization. Registration of all UEDs completes within that
guarded load before any engine is enumerated or selected.

## 9. The Engine Registry

The **engine registry** turns UED data into a live, registered engine -- the descriptor-driven
equivalent of the provider's hand-written engine-registration table. For each UED that passes
validation (§ 10), the registry:

1. **Instantiates one generic engine** -- a single engine implementation that satisfies hipDNN's
   existing engine contract from descriptor data rather than hand-written code, one instance per
   UED. Its engine id is the 64-bit hash of the UED `name` (§ 3).
2. **Binds the engine's descriptors** -- resolves the UED's `heuristic` (UHD) and `metadata`
   (KMD) references and associates the KDPs whose `engine` field names this UED. The KMD is
   available with the UED; the UHD resolves lazily at first knob/selection query (RFC 0017
   § 8.3).
3. **Registers the name -> id mapping** so the host can enumerate the engine and diagnostics /
   support claims ([RFC 0015](0015_EngineSupportClaims.md)) key on the real name rather than a
   hex id.

Nothing in the host-facing engine contract changes: a descriptor-backed engine is selected and
driven exactly as a hand-written one (RFC 0017 § 3, § 8). This RFC specifies only the registry
that stands an engine up from a UED; the generic engine's *internal* plan-building over UDD and
UKD data is governed by those descriptors' own specifications, not this one.

## 10. Validation

Every check below is expected to run at **both build time and run time**. Build-time validation
catches errors before shipping; run-time validation ensures integrity of the loaded set and is
**required** for drop-in UEDs (§ 8), which never pass through the build. The checks divide into
**structural** (expressible in the schema file) and **semantic** (cross-descriptor); the
implementation may fold them into one pass.

### 10.1 Structural validation (against the schema file)

The UED is validated against the JSON Schema file for its declared version (§ 4.2, § 11.3). This
covers required fields, each field's type and pattern, and -- via `additionalProperties: false`
-- **any unknown field is a hard rejection.** An unknown field is always an error (not a
forward-compat case): because an author stamps the lowest version the UED needs (§ 11.2), a
conforming UED never carries a field its version does not define. The body is validated against
the schema for its declared version, not the runtime's latest (§ 11.3).

### 10.2 Semantic validation (cross-descriptor)

These cannot be expressed in JSON Schema because they depend on other descriptors; each is
performed at build time and run time alike:

- **Reference resolution.** A UED's `heuristic` (UHD) and `metadata` (KMD) must each resolve to a
  loadable descriptor of the correct kind; a dangling reference is an error. The referenced UHD
  and KMD must therefore be available when this check runs (§ 8.1).
- **`knobs` must be a subset of KMD field names.** A knob name no KMD field matches is an error
  (RFC 0017 § 4); it requires the referenced KMD to resolve first.
- **Uniqueness (§ 10.2.1).** No two loaded UEDs may share a descriptor `id`, and independently
  none may share a `name`.

The full cross-descriptor reference-integrity check -- which references must resolve, and to what
-- spans multiple descriptor types and is best specified at a higher level than the UED format.
This RFC fixes only that a UED's own `heuristic` and `metadata` references are subject to it, at
both build and run time.

#### 10.2.1 Duplicate detection (descriptor `id` and `name`)

Two independent uniqueness invariants hold across all loaded UEDs: no two share a descriptor
`id`, and no two share a `name` (equivalently its 64-bit hash). On **any** collision under either
invariant, **all** UEDs involved are unloaded (logged, § 12) -- not keep-the-first.

Dropping all rather than keeping one is required because descriptor load order is **not
deterministic**: keep-the-first would leave it ambiguous which definition an engine bound to.
Dropping every participant makes the outcome unambiguous -- the duplicated engine simply does not
load, and diagnostics name every descriptor involved. (RFC 0017 § 4 detects a duplicate name/hash
but specifies neither drop-all nor the independent `id` check.)

### 10.3 UED-load vs KDP-load boundary (guidance)

Better checked at **UED load** (engine-scoped facts):

- UED `id` and `name` uniqueness, drop-all on collision (§ 10.2.1);
- `heuristic`/`metadata` resolvable;
- `knobs` is a subset of KMD field names.

Better checked at **KDP load** (pack-scoped facts):

- the KDP's `engine`, `matchers`, and `dispatch` references resolve (the `engine` ref
  resolving to a registered UED is a KDP-load concern, not a UED concern);
- per-kernel `$kernel.*` matcher pins against the KMD (RFC 0017 § 5);
- **duplicate kernel keys** -- engine-wide, but triggered by loading a pack's UKDs, so detected
  at KDP/UKD load (RFC 0017 § 10).

This boundary is guidance, not a hard split; a shared loader may fold both into one pass. This
RFC defines the engine-scoped checks; the KDP-scoped checks are governed by the KDP's own
specification and are listed here only to place the boundary.

## 11. Versioning and Compatibility

The UED `schema` tag is `hipdnn.ued/<major>.<minor>`. Each descriptor file type versions
independently (a KMD and a UDD advance on their own schedules). This section defines the accept
rule, what `major` and `minor` are permitted to mean, and how the schema files back both
build-time and runtime validation.

> **Overrides RFC 0017 § 4 (compat mechanism).** RFC 0017 states the accept/reject *policy* but
> leaves the field-evolution contract and the runtime's version source unspecified. This section
> supplies both. It does not change 0017's accept rule; it makes the rule realizable.

### 11.1 The accept rule

The runtime carries a supported `major.minor` for the UED type -- concretely, the version(s) of
the UED schema file(s) embedded in the provider build (§ 11.3). A UED is accepted iff:

- **`file.major == provider.major`**, and
- **`file.minor <= provider.minor`**.

Otherwise the UED is rejected. A UED is refused, never silently reinterpreted.

This yields **backward compatibility only**: an older-minor UED loads on a newer-minor provider;
a newer-minor UED is rejected on an older-minor provider (it may carry fields that runtime cannot
interpret). A **major mismatch is a hard break**: at this time the runtime supports **exactly one
major version**, and *every* UED of a different major -- older or newer -- is dropped (logged and
skipped, § 12). There is no multi-major support; a major bump orphans existing UEDs until they
are re-authored. Because major bumps should be rare (§ 11.2), this is an accepted cost, revisited
only if descriptor longevity across a break becomes a requirement.

### 11.2 What `major` and `minor` are permitted to mean

- **Minor bump -- additive, absence-safe changes only.** A minor may add a field **only if its
  absence is well-defined as "the behavior before the field existed"**: a UED at an earlier minor
  omits it, and the runtime reproduces prior behavior via the pre-addition code path (the
  semantics of absence *is* the old implementation, so no default table is needed). A minor may
  not remove, rename, retype, or change the meaning or permitted values of an existing field.
- **Major bump -- everything else** (removing, renaming, retyping, making an optional field
  required, or changing a field's meaning/permitted values). These are the changes where an old
  reader would misinterpret a file, which the hard `major ==` break (§ 11.1) prevents.

**Authors stamp the lowest version their UED needs**, so it stays loadable on the oldest runtime
that can serve it and never carries a field its version does not define. The UED's version should
move rarely.

### 11.3 Schema files back both build-time and runtime validation

Each supported `major.minor` is a standalone JSON Schema file in the repository (`ued/1.0.json`
is authoritative; § 4.2 reproduces it). A new version is a new file. The file is the single
source of truth and drives validation in both places, so they never diverge:

- **Build time** -- the authoring toolchain validates every authored UED.
- **Run time** -- the provider carries the schema file(s) it supports and runs the same
  validation on ingested UEDs, since drop-ins (§ 8) bypass the build.

The provider retains a schema file per minor it supports and validates each UED against the file
for **its declared version**, not the runtime's latest -- so a `1.0` UED carrying a `1.1`-only
field is rejected even on a `1.1` runtime. This RFC does not prescribe how the schema file is
carried in the provider or which validator is used.

## 12. Lifecycle and Operational Policy

- **Load failure => log and skip.** A UED that fails validation is **logged as an error and
  skipped**; the provider does not hard-fail, and the skipped UED registers no engine. This
  matches RFC 0017's "reported in load diagnostics like any other exclusion" and the
  duplicate-key "logged and dropped" pattern.
- **Drop-in reload => plugin (re)load only.** New drop-in descriptors are recognized only when
  the plugin is (re)loaded. There is no live hot-reload: an application triggers a reload by
  closing all open hipDNN handles (and letting the next handle re-scan) or by restarting.
  Descriptor discovery and engine registration are bound to plugin load (§ 8).
- **Concurrency => the guarded plugin-load path** (§ 8.3), not a per-handle resource manager. The
  UED loader adds no new concurrency model.
- **`HIPDNN_DISABLE_ENGINES` => skip at load.** A disabled engine is skipped before registration:
  it never loads and never claims its name or id. A list entry may be any of three identifiers --
  the UED `name`, its 64-bit hash, or the UED GUID `id` -- and the matcher checks all three.
  Because the name is never claimed, disabling one of two same-name UEDs frees the name and lets
  the provider load (a collision-recovery lever, § 10.2.1). The finer-grained
  `HIPDNN_DISABLE_KDPS` / `HIPDNN_DISABLE_UKDS` (RFC 0017 § 10) are governed by their own
  descriptors.
- **Drop-in trust => out of scope.** Trust and enablement for untrusted drop-in descriptors are
  out of scope (RFC 0017 § 14); this RFC adds no trust policy.

## 13. Testing

Reusing the existing tiers (RFC 0006 harness) and RFC 0017 § 14.1's descriptor-pipeline
fuzzing, this RFC adds UED-specific coverage.

**Unit tests:**

- **Structural validation** (§ 10.1) -- valid and invalid field sets; missing required fields;
  malformed `id`/`name` patterns; **unknown field is rejected**; optional fields absent.
- **Schema build/runtime parity** -- the schema file embedded in the provider is byte-identical
  to the repository's canonical file and to the inline copy in § 4.2 (a CI check), so the
  build-time and runtime validators enforce the same contract.
- **Per-version validation** (§ 11.3) -- a UED stamped `1.0` that carries a `1.1`-only field is
  **rejected** even on a `1.1` runtime (validated against its declared version, not the latest).
- **Version accept rule** (§ 11.1) -- matrix of `file` vs `provider` `major.minor`: same
  major/older-or-equal minor loads; newer minor rejected; any major mismatch dropped.
- **Semantic checks** (§ 10.2) -- dangling `heuristic`/`metadata`; a `knobs` entry absent from
  the KMD.
- **Duplicate detection, drop-all** (§ 10.2.1) -- two UEDs sharing an `id`; two sharing a `name`;
  two sharing both; a UED colliding by name with a built-in engine. In each case *every*
  colliding UED is dropped and named in diagnostics, and no engine is left bound to an
  arbitrarily-chosen definition.
- **Engine-id derivation** -- the generic engine's `id()` equals `engineNameToId(name)` (FNV-1a)
  for representative names, including scoped names like `rocke:SDPA`.
- **`HIPDNN_DISABLE_ENGINES`** -- an engine disabled by name, by id-hash, and by GUID is skipped
  before registration and frees its name.

**Integration tests:**

- A UED loads and the engine appears correctly registered in the hipDNN engine-id space at
  plugin load, reportable through `getAllEngineIds`.
- A KDP's `engine` reference resolves to a previously-registered UED.
- End-to-end: UED loads => engine is selectable through the `IEngine` lifecycle
  (`isApplicable` ... `initializeExecutionContext`).
- Disabling an engine that would collide by name lets the provider load (collision-recovery).
- A drop-in UED is validated at runtime by the embedded schema exactly as the build-time check
  would validate it (same accept/reject outcome for a matched pair of good/bad descriptors).

The descriptor pipeline parses untrusted input on the drop-in path, so the loader, parser, and
(future) validators run under the existing ASAN build with a seed corpus, per RFC 0017 § 14.1.

## 14. Glossary

- **UED (Universal Engine Descriptor):** one engine -- a stable identity (`name` + GUID `id`),
  the KMD field names it exposes as knobs, and its behavior/numerical notes. Names its one UHD
  and one KMD by id. 1:1 with a hipDNN engine.
- **Engine id (64-bit):** the hipDNN-facing engine identifier, derived (FNV-1a) from the UED
  `name`; what the plugin reports to the backend and what selection/diagnostics key on.
- **Descriptor GUID `id`:** the cross-reference identifier a descriptor carries; how a KDP
  names its UED and a UED names its UHD/KMD. Distinct from the engine id.
- **Engine registry:** the runtime mechanism that instantiates the generic engine from a UED
  and registers its name->id mapping, replacing the hand-written engine-registration table.
- **Generic engine:** the single C++ engine class that satisfies hipDNN's `IEngine` contract
  from descriptor data, one instance per UED.
- **Descriptor inventory:** the up-front list of descriptor ids, kinds, and locations the
  provider builds at load before parsing any bodies.

---

## 15. Appendix: Fully-Populated UED Examples

The examples in § 4 keep optional fields minimal. This appendix shows UEDs that populate **all**
optional fields, which requires knowing the KMD they reference -- since every `knobs` entry must
name a field the KMD declares. § A.1 sketches only as much of the KMD as the UED depends on; the
KMD format itself is outside this RFC's scope.

### A.1 What a UED needs from its KMD

A UED does not define fields. It references one KMD by id (its `metadata`) and exposes a subset
of that KMD's field **names** through `knobs`. For the purposes of a UED, a KMD is simply a
named list of field names an engine's kernels vary over:

```jsonc
{
  "schema": "hipdnn.kmd/1.0",
  "id":     "9c53b6b0-9a1e-4b1d-8b5c-7e2d9a6f3c40",   // the UED's "metadata" names this
  "name":   "attention_dense variant fields",
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

### A.2 Fully-populated UED -- the `attention_dense` engine

A UED that populates **all** optional fields. Each name in `knobs` is a field the referenced KMD
declares (§ A.1); the notes are RFC 0010 annotations.

```jsonc
{
  "schema":          "hipdnn.ued/1.0",
  "id":              "7d4c2a9e-3b6f-4e1a-8c5d-9a2f7b0e6c14",   // GUID; KDPs name this via "engine"
  "name":            "rocke:attention_dense_fwd",              // globally-unique, scoped; hashed to the 64-bit engine id
  "heuristic":       "2b7a4e1c-6f3d-4a8e-9c2b-5d1f0a7e8b93",   // this engine's one UHD
  "metadata":        "9c53b6b0-9a1e-4b1d-8b5c-7e2d9a6f3c40",   // the KMD (§ A.1)
  "knobs":           ["block_n", "waves_per_eu", "num_persistent"],  // all are KMD field names
  "behavior_notes":  ["runtime_compilation"],
  "numerical_notes": ["tensor_core", "reduced_precision_reduction"]
}
```

This exposes three KMD fields as knobs and carries two note lists. A KDP joins the engine with
`"engine": "7d4c2a9e-3b6f-4e1a-8c5d-9a2f7b0e6c14"`.

### A.3 A minimal engine, for contrast (all optional fields omitted)

The same engine with only required fields -- no knobs, no notes:

```jsonc
{
  "schema":    "hipdnn.ued/1.0",
  "id":        "7d4c2a9e-3b6f-4e1a-8c5d-9a2f7b0e6c14",
  "name":      "rocke:attention_dense_fwd",
  "heuristic": "2b7a4e1c-6f3d-4a8e-9c2b-5d1f0a7e8b93",
  "metadata":  "9c53b6b0-9a1e-4b1d-8b5c-7e2d9a6f3c40"
}
```

Both load and register identically; § A.2 only surfaces knobs and notes that § A.3 leaves unset.
