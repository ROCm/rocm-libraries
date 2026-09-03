# RFC 0020: The Universal Engine Descriptor (UED)

- Contributors: TBD
- **Status**: Draft
- **Implementation Version**: TBD
- **Follows**: [RFC 0017](0017_UniversalKernelDescriptor.md) (Universal Kernel Descriptors). This is the "UED + graph matching" follow-up named in RFC 0017 § 14.2: the engine format together with the graph-matching specification: the `nodes` structural pattern, the symbols matching publishes, and the op-schema registry behind them.

## Table of Contents

1. [Summary](#1-summary)
2. [Relationship to RFC 0017](#2-relationship-to-rfc-0017)
3. [Engine Identity](#3-engine-identity)
4. [The UED Schema](#4-the-ued-schema)
5. [The Graph Model the Pattern Matches](#5-the-graph-model-the-pattern-matches)
6. [Symbol Binding: What the Pattern Publishes](#6-symbol-binding-what-the-pattern-publishes)
7. [Pattern Matching: Stage One](#7-pattern-matching-stage-one)
8. [Knobs](#8-knobs)
9. [Behavior and Numerical Notes](#9-behavior-and-numerical-notes)
10. [Engine Membership (UKD -> KDP -> UED)](#10-engine-membership)
11. [When a UED Is Loaded and Registered](#11-when-a-ued-is-loaded-and-registered)
12. [Engine Registration](#12-engine-registration)
13. [Validation](#13-validation)
14. [Versioning and Compatibility](#14-versioning-and-compatibility)
15. [Lifecycle and Operational Policy](#15-lifecycle-and-operational-policy)
16. [Testing](#16-testing)
17. [Open Questions](#17-open-questions)
18. [Glossary](#18-glossary)
19. [Appendix: Fully-Populated UED Examples](#19-appendix-fully-populated-ued-examples)
20. [Appendix B: Op-Schema Registry Generation](#appendix-b-op-schema-registry-generation)

---

## 1. Summary

RFC 0017 established a family of declarative descriptors that one **generic engine** loads,
matches, selects, and launches with no new C++. It described each at a framing level and deferred
each descriptor's detailed format to its own follow-up. This RFC specifies the **Universal Engine
Descriptor (UED)**: one engine's identity, the **structural pattern** stating the graph shape it
serves, the KMD fields it exposes as knobs, and its behavior and numerical notes, plus the
registration that turns a UED into a selectable engine and the graph matching its pattern drives.

Concretely, this RFC delivers:

- The **UED field contract**: the normative field set, with a single JSON Schema file
  recommended (not mandated) as its single-source-of-truth expression, plus serialization
  (§ 4, § 14.3).
- The **`nodes` structural pattern**, normatively: the node-object members, the opcode and
  optional-operand forms, how nodes connect through shared variables, and the well-formedness
  rules a load rejects on (§ 4.3). RFC 0017 § 4 shows the block by example and defers its format
  here. It is the declarative arm of `graph_match`; the **`native` escape hatch** beside it, which
  is what ships today, is § 4.5.
- The **graph model the pattern is matched against** and the **op-schema registry** generated from
  FlatBuffers annotations that makes a UID-centric graph readable by name (§ 5, Appendix B).
- The **symbol table matching publishes**: the five namespaces, the auto-binding formula, and the
  normative published field set every consumer's references are validated against (§ 6, § 6.1).
- **Stage one of matching**: pattern compilation, the root-opcode index over engines, and the bind
  step (§ 7).
- The **engine-identity model**, including the two distinct id spaces a descriptor engine
  lives in: the descriptor-cross-reference UUID and the hipDNN 64-bit engine id (§ 3).
- **Engine registration**: the process that instantiates the generic engine from UED data and
  exposes it through the provider's engine list (§ 12), plus the ingestion paths and the
  registration-timing guarantee a loaded UED depends on (§ 11).
- The **validation contract**, structural (the field contract, build + runtime) and semantic
  (cross-descriptor, including pattern-name resolution and drop-all duplicate detection), with
  guidance on the UED-vs-KDP boundary (§ 13).
- **Versioning & compatibility**: the accept rule keyed on the `version` field, the constrained
  meaning of `major`/`minor`, and the single-schema mechanism (§ 14); plus
  **lifecycle/operational policy** (load-failure, concurrency, the `HIPDNN_DISABLE_ENGINES`
  opt-out) and **test scope** (§ 15-16).

**Out of scope:** Drop-in **trust and enablement** rules for untrusted descriptor files remain
out of scope, as in RFC 0017 § 16 and § 17 Q1; this RFC adds no trust policy.

## 2. Relationship to RFC 0017

This RFC lives alongside RFC 0017; some sections restate 0017 material (identity, knobs, the
reference model) so the UED format reads standalone. **This RFC is the source of truth for UED
matters.** Most of what follows **tightens** points 0017 defers or under-constrains (often to this
very follow-up, per 0017 § 14.2); a few points **diverge** from 0017's descriptor conventions. As
a follow-up, filling 0017's deferred scope is expected and is not itself a divergence.

**Tightenings (this RFC pins down what 0017 deferred or left soft):**

- **Compatibility mechanism (§ 14).** Absence-safe minor bumps, hard-reject unknown fields, and a
  runtime version from the provider's schema.
- **Engine name format (§ 4.2).** A globally-unique, scoped `namespace:local` `name`, since it is
  hashed into the engine-id space and must not collide.
- **Duplicate detection (§ 13.2.1).** An independent descriptor-`id` check; drop all UEDs in a
  genuine collision, but accept content-identical `id` duplicates, loading them as one.
- **The `nodes` pattern grammar (§ 4.3).** RFC 0017 § 4 shows the block by example and defers its
  format to this follow-up; this RFC fixes the node-object members, the opcode-set form, the `?`
  optional-operand suffix, how nodes connect, and the well-formedness and registry-resolution rules
  a load rejects on. It also adds the **`native` arm** (§ 4.5) beside it, the registry-resolved
  escape hatch 0017 § 5 calls a "native predicate", scoped here to the engine's stage-one match.
- **Pattern compile at registration (§ 12).** 0017 § 8.1 step 3 defers the pattern's compile — and
  the native symbol's resolution — to registration and names this section as the authority for it;
  the mechanism is specified here, so an unresolvable op or operand name is a load error rather
  than a first-graph surprise.
- **The graph-schema floor on the engine (§ 4.2).** 0017 § 4 requires a UED to declare the hipDNN
  schema version its pattern was authored against; this RFC gives it a field (`sdk_version`), a
  default, and a comparison rule. It is also the sole such floor: RFC 0018 § 10 keeps no
  `sdk_version` on the UMD, so a matcher runs under the floor of the engine of each pack that
  lists it.

**Divergences (this RFC departs from an 0017 convention):**

- **No in-band type tag (§ 4).** 0017 introduced the descriptor kinds with an in-band `schema` tag
  naming each kind in the body; no descriptor carries one. The kind is tracked externally by the
  filename suffix (§ 4) and a `major.minor` `version` field carries compatibility (§ 14): a file
  whose name and body disagree has no correct reading, so the body does not restate the name.
  Rejecting the key is specified wherever the unknown-member rule is — a UED rejects `schema` as an
  unknown field (§ 4.2, § 13.1), and so does a UMD
  ([RFC 0018 § A.1](0018_UniversalMatchDescriptor.md#a1-the-umd-descriptor-object)). The remaining
  kinds inherit the same rule when their own follow-ups land.
- **Version-specific validation is not required (§ 13.1).** A single schema validates the
  structural superset across all supported versions. The schema carries `addedInVersion` data
  (§ 4.2) that makes validating a UED against its declared version's exact field set *possible*, but
  this RFC does not require any consumer to perform that check (§ 13.1).

No other silent contradictions. In particular the two 0017 § 4 UED fields an earlier draft of this
RFC omitted — the `nodes` pattern and `sdk_version` — are carried here (§ 4.2, § 4.3), so the
field contract is a superset of 0017's, not a subset of it. The pattern is reached through
`graph_match` rather than as a top-level member, which is what keeps it additive (§ 14.2). Any
conflict surfaced during review is recorded here. The converse ledger, what changed *within* 0017
when this RFC and RFC 0018 landed beside it, is
[RFC 0017 § 1.2](0017_UniversalKernelDescriptor.md#12-what-this-revision-changed).

## 3. Engine Identity

An engine lives in **two distinct id spaces**, which the UED keeps separate:

**(a) The descriptor UUID (`id`).** Every descriptor carries a stable UUID used only for
cross-references among descriptor files: a KDP names its UED by this id; a UED names its UHD
and KMD by theirs (RFC 0017 § 4). It is internal to the descriptor graph and never crosses the
hipDNN library boundary.

**(b) The 64-bit engine id.** hipDNN identifies engines by a 64-bit id derived from a registered
engine **name**, an FNV-1a hash of the name (RFC 0017 § 4; [RFC 0003](0003_EngineIdDesign.md)).
A descriptor-backed engine hashes its UED `name` into this space exactly as a hand-written
engine does; this is the id the provider reports to the host, and what selection, diagnostics,
and support claims key on.

| Concern | Identifier |
|---|---|
| A KDP naming its UED; a UED naming its UHD/KMD | descriptor UUID `id` |
| hipDNN selecting among engines; logs; support claims | 64-bit engine id (FNV-1a of `name`) |

The UED `name` is therefore load-bearing only where the engine surfaces outside the descriptor
graph (selection, logs, diagnostics, and the hash into the engine-id space); internally, the
UUID `id` binds. Names must be **globally unique** and should be scoped, e.g. `rocke:SDPA`.

## 4. The UED Schema

This section **defines** the UED schema: § 4.1 an example instance, § 4.2 the normative
definition, § 4.3 the `nodes` structural pattern, § 4.4 serialization, § 4.5 the native
escape hatch. A UED carries a `major.minor` `version` field that the accept rule gates on
(§ 14).

A UED does not carry an in-band type tag, so the descriptor kind is determined externally rather
than from the file contents (for example, by a filename suffix such as `<name>.ued.json`).
Confirming that a file presented as a UED is in fact one is therefore the external mechanism's
responsibility; the schema validates a UED's contents but cannot establish that the file was meant
to be a UED.

### 4.1 Example instance

```jsonc
{
  "version":         "1.0",                        // major.minor; gated at load (§ 14)
  "id":              "efc9eae4-fe33-4cb0-a593-95d771dc13b2",  // UUID; referenced by KDPs (§ 3a)
  "name":            "rocke:example_attention_fwd",  // globally-unique, scoped engine name (§ 3b)
  "sdk_version":     "1.0",                        // graph schema this pattern was authored against
  "heuristic":       "ae896b07-80cd-473c-b3f4-6a8892998519",  // optional: one UHD id (§ 4.2)
  "metadata":        "9ae0b215-32a7-49d1-96df-e9b05e1927ea",  // one KMD id (required)
  "graph_match": {                                 // stage one: how this engine matches and binds
    "nodes": [                                     // declarative arm: the graph shape served (§ 4.3)
      {"kind": "op", "id": "sdpa_fwd", "op": "sdpa_fwd",
       "operands": {"q": "$q", "k": "$k", "v": "$v", "attn_mask": "$attn_mask?"},
       "results":  {"o": "$o"}}
    ]
  },
  "knobs":           ["split_k", "tile_m"],        // optional: KMD field names to expose (§ 8)
  "behavior_notes":  ["runtime_compilation"],      // optional (§ 9)
  "numerical_notes": ["tensor_core", "reduced_precision_reduction"]  // optional (§ 9)
}
```

The same engine written against the **native arm** instead, the escape hatch for a match the
declarative form cannot yet state (§ 4.5). The two arms are mutually exclusive; everything
outside `graph_match` is unchanged:

```jsonc
{
  "version":     "1.0",
  "id":          "efc9eae4-fe33-4cb0-a593-95d771dc13b2",
  "name":        "rocke:example_attention_fwd",
  "metadata":    "9ae0b215-32a7-49d1-96df-e9b05e1927ea",
  "graph_match": { "native": "rocke.example_attention.graph_match" }   // one registered symbol (§ 4.5)
}
```

### 4.2 Normative schema

A conforming UED is a JSON object with the members below. Unknown members are rejected. Every
member but `graph_match` is inert data — a version, identity, two references, and optional
annotations; `graph_match` is the one member carrying structure, and it holds whichever of the
two arms (§ 4.3, § 4.5) this engine matches with.

**Field specification (normative).**

| Field | Req. | JSON type | Value constraints |
|---|---|---|---|
| `version` | yes | string | `<major>.<minor>`, both numeric, and one of the values the schema enumerates (§ 14.3), e.g. `1.0`. The compatibility field the accept rule gates on (§ 14). |
| `id` | yes | string | A UUID (RFC 4122) in canonical `8-4-4-4-12` hex form. Unique across all loaded descriptors, except that content-identical UEDs may share an `id` (§ 13.2.1). The cross-reference key a KDP's `engine` field uses (§ 3a). |
| `name` | yes | string | Globally-unique, scoped engine name matching `^[A-Za-z0-9_.-]+:[A-Za-z0-9_.-]+$` (a `namespace:local` form, e.g. `rocke:SDPA`). Hashed (FNV-1a, 64-bit) into the hipDNN engine-id space (§ 3b). Non-empty; unique by both literal name and by hash. |
| `sdk_version` | no | string | `<major>.<minor>`, the hipDNN graph schema version this engine's pattern was authored against (RFC 0017 § 4). Defaults to `1.0` when omitted. Compared numerically by `(major, minor)`: refused at load when newer than the runtime's own graph schema, and at match time the whole engine declines a graph whose reported floor is above it, before binding and taking every pack naming it ([RFC 0018 § 10](0018_UniversalMatchDescriptor.md#10-serialization-and-versioning)). This is the **only** graph-schema floor in the system: no UMD carries one, and a matcher runs under the floor of the engine of each pack that lists it, so raising this field is a review point for every matcher on the engine. Independent of `version`, which gates the UED *format*. |
| `heuristic` | no | string | UUID of this engine's one UHD. Must resolve to a loadable UHD at load (§ 13.2). Absent => the engine ships no heuristic and its catalog is ordered by the declared fallback, `priority` then descriptor `id` (§ 8, RFC 0017 § 5). A key present but naming nothing is still an error. |
| `metadata` | yes | string | UUID of this engine's one KMD. Must resolve to a loadable KMD at load (§ 13.2). |
| `graph_match` | no | object | Stage one: how this engine decides a graph and binds the tokens every later stage reads (§ 6, § 7). Exactly one arm, and they are mutually exclusive: **`nodes`**, the declarative pattern of **§ 4.3**, or **`native`**, the escape-hatch symbol of **§ 4.5**. Absent => the engine binds nothing, publishes an empty symbol table, and is admitted or declined by its packs' UMDs alone. |
| `knobs` | no | array of string | Each element is a field name declared in the referenced KMD (§ 8). No duplicates. Absent or `[]` => engine exposes no descriptor knobs. Every element must match a KMD field or it is a load error (§ 13.2). |
| `behavior_notes` | no | array of string | hipDNN behavior-note tags ([RFC 0010](0010_BehaviorNotes.md)). No duplicates. Absent => none. |
| `numerical_notes` | no | array of string | hipDNN numerical-note tags. No duplicates. Absent => none. |

Every field but `version`, `id`, `name`, and `metadata` may be omitted; a valid engine can expose
no knobs, carry no notes, default its graph-schema floor, and ship no heuristic. `graph_match` is
optional for the same reason the others are: its absence is well-defined as the behavior before
the member existed — an engine that states no graph shape binds nothing and leaves applicability
entirely to its packs' criteria. That is a narrow way to write an engine, not a broken one, and
keeping it absence-safe is what holds the format on the `1.0` line (§ 14.2).

A single JSON Schema **file** is delivered with the provider, and the inline copy below is that
file's content; a build check verifies the two match. The file reflects the latest version
the runtime supports and covers **all** supported versions at once: its `version` property
enumerates the accepted versions (here, only `1.0`), and each new version adds its value to the
`enum`. The runtime supports exactly one major version, so every enumerated version is of that
major. Every field is present, with fields introduced after the earliest version marked
**optional**, so the schema is a structural **superset**: it accepts any supported version
structurally but does not, on its own, enforce which fields are required or forbidden at a specific
version. Confirming whether a given field is permitted or required at the UED's declared version is
possible from the `addedInVersion` data but is not part of this schema's own validation (§ 13.1,
§ 14).

The schema is also the limit of what a structural check can do for `graph_match`: it fixes the
arms' *shape* — that exactly one is present, the node members, the binding-form spelling, the
opcode forms, that `native` is a non-empty string — but every rule that needs the op-schema
registry (does this `op` exist, does it declare this operand name, is that name optional) or the
provider's native registry (is this symbol registered) is semantic and runs in § 13.2. § 4.3 and
§ 4.5 state both sets and say which is which.

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "ued.json",
  "title": "hipDNN UED",
  "type": "object",
  "additionalProperties": false,
  "required": ["version", "id", "name", "metadata"],
  "properties": {
    "version": {
      "type": "string",
      "enum": ["1.0"],
      "addedInVersion": "1.0"
    },
    "id": {
      "description": "This descriptor's own UUID. Unique across loaded UEDs except for content-identical duplicates (semantic; see RFC 0020 section 13.2.1).",
      "type": "string",
      "pattern": "^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$",
      "addedInVersion": "1.0"
    },
    "name": {
      "type": "string",
      "pattern": "^[A-Za-z0-9_.-]+:[A-Za-z0-9_.-]+$",
      "addedInVersion": "1.0"
    },
    "sdk_version": {
      "description": "Graph schema version this pattern was authored against; defaults to 1.0 when absent (see RFC 0020 section 4.2).",
      "type": "string",
      "pattern": "^[0-9]+\\.[0-9]+$",
      "addedInVersion": "1.0"
    },
    "heuristic": {
      "description": "Cross-reference: MUST resolve to a loadable UHD (semantic; see RFC 0020 section 13.2).",
      "type": "string",
      "pattern": "^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$",
      "addedInVersion": "1.0"
    },
    "metadata": {
      "description": "Cross-reference: MUST resolve to a loadable KMD (semantic; see RFC 0020 section 13.2).",
      "type": "string",
      "pattern": "^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$",
      "addedInVersion": "1.0"
    },
    "graph_match": {
      "description": "Stage one: exactly one arm. 'nodes' is the declarative pattern (RFC 0020 section 4.3); 'native' is the escape-hatch symbol (section 4.5) and MUST be registered in the provider's graph-match registry (semantic; see section 13.2).",
      "type": "object",
      "additionalProperties": false,
      "oneOf": [
        { "required": ["nodes"] },
        { "required": ["native"] }
      ],
      "properties": {
        "native": {
          "type": "string",
          "minLength": 1,
          "addedInVersion": "1.0"
        },
        "nodes": {
          "description": "Structural pattern over the op DAG. Opcodes, operand and result names MUST resolve against the op-schema registry (semantic; see RFC 0020 section 13.2).",
          "type": "array",
          "minItems": 1,
          "items": {
            "type": "object",
            "additionalProperties": false,
            "required": ["kind", "id", "op"],
            "properties": {
              "kind": { "type": "string", "enum": ["op"] },
              "id":   { "type": "string", "pattern": "^[A-Za-z_][A-Za-z0-9_]*$" },
              "op": {
                "oneOf": [
                  { "type": "string", "minLength": 1 },
                  {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["one_of"],
                    "properties": {
                      "one_of": {
                        "type": "array",
                        "minItems": 2,
                        "uniqueItems": true,
                        "items": { "type": "string", "minLength": 1 }
                      }
                    }
                  }
                ]
              },
              "operands": {
                "type": "object",
                "additionalProperties": {
                  "type": "string",
                  "pattern": "^\\$[A-Za-z_][A-Za-z0-9_]*\\??$"
                }
              },
              "results": {
                "type": "object",
                "additionalProperties": {
                  "type": "string",
                  "pattern": "^\\$[A-Za-z_][A-Za-z0-9_]*$"
                }
              }
            }
          },
          "addedInVersion": "1.0"
        }
      },
      "addedInVersion": "1.0"
    },
    "knobs": {
      "description": "Each entry MUST be a field name declared by the referenced KMD (semantic; see RFC 0020 section 13.2).",
      "type": "array",
      "items": { "type": "string", "minLength": 1 },
      "uniqueItems": true,
      "addedInVersion": "1.0"
    },
    "behavior_notes": {
      "type": "array",
      "items": { "type": "string", "minLength": 1 },
      "uniqueItems": true,
      "addedInVersion": "1.0"
    },
    "numerical_notes": {
      "type": "array",
      "items": { "type": "string", "minLength": 1 },
      "uniqueItems": true,
      "addedInVersion": "1.0"
    }
  }
}
```

The schema's `description` keywords flag which fields carry cross-references and what each must
resolve to. JSON Schema cannot *enforce* cross-file resolution, so these are recorded as
machine-readable annotations and checked semantically (§ 13.2); `additionalProperties: false`
makes any unknown field a hard rejection (§ 13.1).

**`addedInVersion`.** Each property carries an `addedInVersion` keyword naming the `major.minor` at
which the field was introduced; its value must be one the `version` enum accepts. The schema must
include it on every property, since it is the machine-readable record that makes version-accurate
auditing and validation possible (§ 13.1). **A JSON Schema validator ignores the keyword**, so it
does not affect the superset check; it is consumed by tooling and follow-on checks instead.

Every field here reads `1.0`: the format has had no major bump, so every member the schema
declares has been present since the earliest version it enumerates. Should a major bump ever
happen, `addedInVersion` **re-bases** across the whole file onto the new major's earliest minor,
because the runtime supports exactly one major (§ 14.1) and the keyword's value must be one the
`version` enum accepts. Field history before such a break is git's to carry, not the schema's;
what the keyword must stay correct about is the minimum-version computation of § 13.1, which is
only ever asked within the supported major.

The schema targets **Draft 7** (recommended, not required): off-the-shelf C++ JSON-Schema
validators target Draft 7, so one file can drive both the build-time and runtime checks without
a bespoke validator. The constructs used (`type`, `enum`, `pattern`, `required`, `minItems`,
`additionalProperties`, `uniqueItems`, `oneOf`) are common to Draft 7 and later dialects.

### 4.3 The `nodes` pattern (normative)

`graph_match.nodes` is the graph shape the engine serves: a list of op nodes and the operand and
result edges connecting them to the graph's tensors, each edge mapped to a **pattern variable**.
It is the **declarative arm** of `graph_match` (§ 4.2) and the one part of a UED that carries
structure; matching it is what publishes the symbol table every consumer downstream reads (§ 6).
RFC 0017 § 4 introduces the block by example and defers its format here; this section is that
format. The other arm, `native`, is § 4.5.

Throughout this section `nodes` means the `nodes` key of `graph_match`. A UED carries **exactly
one** `nodes` block, so every pack naming the engine matches the same graph shape and differs only
in what its criteria constrain (§ 6). A family spanning two topologies is two engines (§ 17 Q1).

**This arm is specified, not yet implemented.** The escape hatch of § 4.5 is what ships today; a
UED naming `nodes` is rejected by the current loader as an unknown key. Everything below is the
contract the declarative implementation must meet, and the reason `native` is scoped as a hatch
rather than the format's steady state.

#### 4.3.1 Grammar

```ebnf
nodes        = "[" , node , { "," , node } , "]" ;   (* at least one *)
node         = "{" , kind , "," , node-id , "," , opcode ,
                     [ "," , operands ] , [ "," , results ] , "}" ;
kind         = '"kind"'     , ":" , '"op"' ;         (* the only node kind at this version *)
node-id      = '"id"'       , ":" , string ;         (* ident; names the Attributes namespace root *)
opcode       = '"op"'       , ":" , ( string | opcode-set ) ;
opcode-set   = "{" , '"one_of"' , ":" , "[" , string , "," , string , { "," , string } , "]" , "}" ;
operands     = '"operands"' , ":" , "{" , [ edge , { "," , edge } ] , "}" ;
results      = '"results"'  , ":" , "{" , [ edge , { "," , edge } ] , "}" ;
edge         = edge-name , ":" , binding ;
edge-name    = string ;                              (* an operand/result name the op declares *)
binding      = '"$' , ident , [ "?" ] , '"' ;        (* "?" only on an operand *)
ident        = ( letter | "_" ) , { letter | digit | "_" } ;
```

**Members.**

| Member | Req. | Meaning |
|---|---|---|
| `kind` | yes | The node kind. `"op"` is the only value at this version; the member exists so a future node kind (a tensor-shaped or control-shaped node) is an additive change rather than a reinterpretation of an existing object. |
| `id` | yes | Names this node within the pattern, and is the **Attributes namespace root** its scalar attributes bind under: `{"id": "sdpa_fwd"}` publishes `$sdpa_fwd.causal_mask` (§ 6.1). Unique within the block. Bare in the descriptor; the `$` appears only on the reference. |
| `op` | yes | The opcode this node matches, keyed against the op-schema registry (§ 5, Appendix B). A bare string matches exactly. An **opcode set**, `{"one_of": ["a", "b"]}`, matches any listed opcode; every member must declare the operand and result names this node binds, so one node covers a family of ops that agree on their edges. |
| `operands` | no | Input edges: an object mapping an operand **name the op declares** to a pattern variable. Absent or `{}` binds no operands, which is legal but publishes nothing for that node's inputs. |
| `results` | no | Output edges, same form. A result binding takes no `?`: an op's declared result is always produced. |

**Binding form.** A binding is `"$name"`, or `"$name?"` on an operand the op declares optional. The
`?` is a property of the *pattern*, not of the graph: it says the engine still matches when the
graph omits that operand, and `{"not_present": ["$name"]}` then answers true for it
(§ 6.1, Appendix B.5).
An operand with no `?` is required — a graph omitting it does not match, and the engine declines.
An operand the pattern does not name at all is not bound, and no criterion can ask about it, which
is a real authoring choice: declaring an optional operand and declining it in criteria is not the
same as omitting it from the pattern, because only the first produces a diagnosable decline.

**There is no wildcard opcode.** An earlier draft of the matcher RFC listed `any` alongside
`one_of`; it is not in this grammar. A node whose op is unknown at compile has no registry entry,
so none of its edge names can be resolved (§ 4.3.3) and the symbol table it publishes cannot be
laid out ahead of a live graph — the pattern would bind by guess, which is the failure mode the
registry exists to prevent. `one_of` gives the same reach with the resolution intact, because every
arm is named and checked. A genuinely opcode-agnostic pattern is the general-matching problem
deferred to the JIT follow-up ([RFC 0017 §
9.3](0017_UniversalKernelDescriptor.md#93-future-jit-and-normalized-providers)).

**Connectivity is implicit through shared variables.** Nodes do not reference each other by `id`.
Two nodes are connected when the same pattern variable appears as one's result and the other's
operand, which is exactly how the graph itself connects nodes — by a shared tensor UID (§ 5). A
three-node Conv-Bias-ReLU pattern is written as three independent node objects whose chaining is
carried entirely by `$conv_out` and `$bias_out` appearing twice each:

```jsonc
"nodes": [
  {"kind": "op", "id": "conv", "op": "convolution_fwd",
   "operands": {"X": "$x", "W": "$w"},              "results": {"Y": "$conv_out"}},
  {"kind": "op", "id": "bias", "op": "pointwise_add",
   "operands": {"A": "$conv_out", "B": "$bias"},    "results": {"Y": "$bias_out"}},
  {"kind": "op", "id": "act",  "op": "pointwise_relu",
   "operands": {"A": "$bias_out"},                  "results": {"Y": "$y"}}
]
```

A variable appearing as a result of one node and an operand of another is an **intermediate**;
whether it is legal to fuse across is not the pattern's to say, and a criterion asks with
`$conv_out.virtual` ([RFC 0018 § 3](0018_UniversalMatchDescriptor.md#3-criteria-vocabulary)). The
pattern binds a shape; the criteria decide whether that shape is servable.

**Match semantics are exact, not subgraph-containment.** The pattern matches the ops it names and
publishes what they bind; it does not by itself bound the rest of the graph. A pack that means
"this graph and nothing else" pins it on the criteria side with `$graph.node_count` (§ 6.1). This
split is deliberate: node count is a *constraint*, varies between packs on one engine, and so
belongs with the criteria rather than being baked into a pattern all those packs share.

#### 4.3.2 Well-formedness (structural)

These are checkable from the block alone and are part of the structural check (§ 13.1); each is a
load rejection, never a warning:

- **The block is non-empty.** `"nodes": []` is refused.
- **`kind` is `"op"`.** Any other value is refused rather than ignored.
- **Node `id` is unique within the block**, and is an `ident`. A duplicate `id` would make an
  Attributes reference ambiguous.
- **A node `id` is not a reserved root.** `graph`, `kernel`, and `device` are refused (§ 6.1).
- **A pattern variable is bound at most once.** The same `$name` may be *read* by several nodes —
  that is how connectivity is expressed — but it is **bound** by exactly one edge: either one
  node's result, or, for a graph input, one node's operand. A variable appearing as the result of
  two nodes, or as an operand of two nodes with no producing result among them, is refused.
- **A variable is not a reserved root**, and does not collide with a node `id`.
- **`?` appears only on an operand binding**, never on a result and never mid-identifier.
- **Edge names are unique within their `operands` or `results` object**, which JSON object
  semantics already imply but a lenient parser may not enforce.

#### 4.3.3 Registry resolution (semantic)

These need the op-schema registry (§ 5, Appendix B) and so run in semantic validation (§ 13.2),
at pattern compile (§ 12):

- **Every `op` resolves.** An opcode absent from the registry is refused, never guessed at. For an
  opcode set, *every* member must resolve.
- **Every edge name is declared by the op**, as an operand for an `operands` entry and as a result
  for a `results` entry. A name the registry does not declare for that op is refused; this is the
  check that catches an operand renamed out from under a pattern by a schema change (Appendix B.6).
- **`?` matches the registry's optionality.** A `?` on an operand the registry declares required is
  refused, since the pattern would be claiming a graph the op cannot produce. A required binding on
  an operand the registry declares optional is *permitted* — it is the pattern narrowing itself to
  graphs that supply the operand.
- **An opcode set agrees on its edges.** Every member of a `one_of` must declare every edge name
  the node binds, with the same optionality, or the published symbol table would depend on which
  arm matched.

A failure here is a load error naming the UED, the node `id`, and the unresolved name (§ 15).

### 4.4 Serialization

The UED is authored and shipped as **JSONC** (JSON with comments), consistent with how RFC 0017
presents every descriptor. Comments are stripped before validation (§ 4.2, § 14.3).

### 4.5 The `native` arm (normative)

`graph_match.native` is a symbol naming a function the provider ships, resolved through the
provider's graph-match registry. It is the **escape hatch** for a match the declarative arm cannot
yet state, and it is what ships today. Its signature:

```cpp
std::optional<BoundTokens> (*)(const MatchContext&);
```

`nullopt` **declines** the graph; a returned map **is** the binding. Match and bind are one answer
from one function, which makes "matched but bound nothing" unrepresentable — the state the two
arms would otherwise both have to define away.

`MatchContext` carries the graph, a device ordinal, and the device properties (§ 5). `BoundTokens`
is a flat map from token name to scalar value; the token names are the function's own choice, not
derived from the op-schema registry, and the descriptor set naming them must agree with the
provider's source by convention.

**One producer.** An engine's binding comes from this function alone. A pack's UMDs read it and
never add to it (§ 6), so two descriptors cannot write one token to different values, and the
format needs no reconciliation rule for that case.

**Structural check.** `native` is a non-empty string, and the enclosing object carries no other
key. **Semantic check.** The symbol must be registered in the loading provider; an engine naming
an unregistered symbol is dropped at read time, before its id is advertised (§ 13.2).

**What this arm does not provide.** Each of these is a guarantee the declarative arm gives and
this one cannot, and each is why `native` is a hatch rather than the format's steady state:

- **No load-time symbol set.** The tokens are whatever the function returns on a live graph, so
  the published set is unknown at load. The cross-descriptor validation of § 6 and § 13.2 — that
  every `$`-reference in a UMD, UDD, or UHD resolves against the engine's binding — cannot run.
  A stale reference fails closed at match time instead of being an error at load.
- **No registry-derived naming.** Token names are hand-authored and restated in both the provider
  source and the descriptors that read them; nothing checks the two agree.
- **No root-opcode index entry.** The function is opaque, so it cannot be indexed by root opcode
  (§ 7) and is not pruned before it runs.
- **No structural/criteria separation.** The declarative arm splits topology (§ 4.3) from value
  checks (the UMD's criteria); a native function may fuse both, and in the shipped engines it
  does. Checks that would be per-pack memoized criteria then prune the whole engine instead.

A UED SHOULD move to the declarative arm once the pattern it needs is expressible. Both arms fill
the same slot in the same object, so that migration is a change within `graph_match` and not a
format break.

## 5. The Graph Model the Pattern Matches

The pattern is matched against an immutable graph, read through the existing `IGraph` interface
(`projects/hipdnn/flatbuffers_sdk/include/hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp`)
plus the HIP stream for device properties. Three properties of that model drive the design.

**The graph is UID-centric, not edge-centric.** A `Node` carries only `{name, compute_data_type,
attributes_type, attributes}` (`graph_generated.h`, Node table). It has no input or output tensor
lists. A node's operands and results are UID fields inside its concrete attribute table, for example
`SdpaAttributes::q_tensor_uid()`, `k_tensor_uid()`, `o_tensor_uid()`
(`sdpa_attributes_generated.h`). Connectivity between nodes is implicit: two nodes are connected
when a result UID of one appears as an operand UID of the other. To resolve a node's edges, the
matcher must know the op type, cast via `attributesAs<T>()`, and read the named UID fields.

**Consequence: the pattern needs an op-schema registry.** For each op type, the registry declares
which attribute fields are operand UIDs, which are result UIDs, whether each is required or
optional, and the names of the op's scalar attributes. This registry is what lets a `nodes` pattern
reference operands and results by name (`q`, `k`, `v`, `o`) and what powers the auto-binding formula
of § 6.

**The registry is generated from schema annotations, not name conventions.** A table-level
FlatBuffers attribute names the op, and field-level attributes on each op's attribute table declare
the binding contract next to the field they govern. `umd_opcode` on the attribute table gives the
op's descriptor-facing opcode (`SdpaAttributes (umd_opcode: "sdpa_fwd")`); a pattern node's `op`
names it, and the registry keys on it (falling back to the table type name when the attribute is
absent), so the schema is the single source of truth for the opcode rather than an ad-hoc string.
`umd_input_tensor` / `umd_output_tensor` mark a UID field and `umd_name` names it, so SDPA's Q
operand is `q_tensor_uid: long (umd_input_tensor, umd_name: "q")` and its O result is `o_tensor_uid:
long (umd_output_tensor, umd_name: "o")`. Optionality is not re-annotated: a UID field's `= null`
default already encodes it (`attn_mask_tensor_uid: long = null`), and the `NodeAttributes` union
already maps each opcode to its table, so every unannotated *scalar* field is a scalar attribute by
elimination (unannotated non-scalar fields: vectors, sub-tables, are not bindable scalars and are
skipped). A build step emits the binary reflection schema (`graph.bfbs`, which transitively covers
every attribute table; custom attributes surface only through reflection, not the generated headers)
and a generator reads each field's attributes to emit the registry, so it stays in lockstep with the
graph definitions rather than being hand-maintained ([Appendix
B](#appendix-b-op-schema-registry-generation) specifies the annotation contract, the
field-classification rules, and the generation pipeline). Names are never inferred from the
`_tensor_uid` name suffix, which would misclassify non-UID fields such as
`PointwiseAttributes::axis_tensor_uid` (a plain axis index, not a tensor UID).

**Tensors expose dims, strides, dtype, and a virtual flag, but no layout enum and no rank field.**
`TensorAttributes` (`tensor_attributes_generated.h`) offers `dims()`, `strides()` (both nullable
vectors), `data_type()`, `uid()`, and `virtual_()`. Rank is `dims()->size()`. Layout is not stored;
it is derived from the strides, which is why the pattern publishes `stride_order`, an array indexed
by logical dimension holding that dimension's stride rank, and a criterion compares layout as one
([RFC 0018 §
5](0018_UniversalMatchDescriptor.md#5-layout-and-stride-order-criteria)). Quantities like head
size, batch, and head count are **not** attributes; they are specific tensor dims (for SDPA,
`q.dims[3]`, `q.dims[0]`, `q.dims[1]`). A criterion reaches them positionally as `$q.dims[i]`, never
as an attribute read (§ 6).

**Device and arch are out-of-band.** The graph carries no device identity. Arch comes from the
stream via `getDeviceString(handle.getStream())` (`HipDeviceUtils.hpp:48`); for AOT it gates *pack
selection*, not a match criterion. Other device properties resolve against the `Handle` the matcher
receives alongside the graph rather than against a graph field, and are the matcher's namespace, not
the pattern's ([RFC 0018 § 2](0018_UniversalMatchDescriptor.md#2-the-symbol-table-criteria-read)).

**Graph guarantees the pattern may rely on.** Per the `IGraph` contract the graph is topologically
sorted, acyclic, fully connected, and has unique tensor UIDs. The matcher builds its own
UID-to-producer and UID-to-consumers index once per graph to walk edges and reconstruct
connectivity, since no adjacency query is provided; fusion legality reads each intermediate's
`virtual` flag.

![The matcher reads a UID-centric graph via an op-schema registry that reconstructs
edges and auto-binds symbols](../images/umd_binding_model.svg)

## 6. Symbol Binding: What the Pattern Publishes

Matching does double duty: it decides applicability and it binds named variables. The binding half
is the **engine's**: the UED's `graph_match` (§ 4.2) is what runs, and matching publishes a symbol
table. That table is what every consumer downstream is written against, so it is specified here,
with the field it travels on.

**One producer per engine.** Whichever arm is populated, the engine's binding comes from exactly
one place, and a pack's UMDs are pure readers of it (§ 7). Two descriptors therefore cannot write
one token to different values, so the format states no reconciliation or conflict rule: the
conflict is unrepresentable rather than detected. An engine with no `graph_match` at all publishes
an empty table, and its packs' criteria may then read only the two namespaces the pattern never
bound, `$kernel.*` and `$device.*`.

A symbol is **declared** in the UED's pattern, **bound** when the graph matches, and **used** by a
UMD's criteria ([RFC 0018 § 2](0018_UniversalMatchDescriptor.md#2-the-symbol-table-criteria-read)),
by the UDD's dispatch and workspace formulas ([RFC 0017 §
6](0017_UniversalKernelDescriptor.md#6-dispatch-and-workspace)), and by the engine's UHD
`features_signature`. Every symbol any of them references must be bound by the pattern, so none of
them can read a value the match does not produce.

**The pattern is engine-wide and singular**, one per UED (§ 4.3), so every pack naming that engine matches
the same graph shape and differs only in what it constrains. Two consequences run through the rest
of this document: the matcher binds once per engine per graph rather than once per matcher (§ 7),
and the bound-symbol set has a single owner. A kernel family whose graph shape differs
*structurally* is therefore a different engine: most variation does not reach that bar, and a
genuinely different topology already needed its own metadata schema and heuristic anyway, so fused
and unfused counterparts are mutually exclusive by engine rather than by node count.

**The UED publishes the bound-symbol set, and it is the single source every consumer is checked
against.** A UMD listed by a pack, that pack's UDD, and the engine's own UHD are each validated
against the engine's published symbols at build and at drop-in load, and a reference that does not
resolve is rejected then rather than failing closed later on a live graph. One publisher rather than
three keeps the check mechanical: the UHD in particular is engine-wide, so before the pattern moved
onto the engine there was no engine-level binding for its feature tokens to resolve against, only
whatever matchers the packs naming that engine happened to carry.

**That check is the declarative arm's, and only its.** Everything from here to § 6.1 describes what
`graph_match.nodes` publishes: a set laid out at compile from the op-schema registry, and therefore
knowable before any graph arrives. Under `graph_match.native` (§ 4.5) the tokens are whatever the
function returns at runtime, so there is no set to validate against and a stale `$`-reference fails
closed on a live graph instead of erroring at load. An engine on the native arm gets the *binding*
that the rest of this section describes; it does not get the load-time guarantee, and § 13.2 says
which checks consequently do not run for it.

**Auto-binding is the default, and follows a standard formula.** When the pattern names an operand
or result variable, the matcher, using the op-schema registry (§ 5), automatically binds it and its
fields, so authors get a complete symbol table for free and never hand-declare each field. Every
field a criteria or dispatch expression may reference falls in one of **five namespaces**, three of
which the pattern binds and two of which come from elsewhere; the hipDNN schema declares them all so
the interpreter fails closed on anything undeclared:

- **Tensor** — a bound operand/result and its fields: `$q` is the whole tensor (the matched
  `TensorAttributes`) and `$q.uid` its graph UID; each dim positionally as `$q.dims[i]`;
  each stride as `$q.strides[i]`; and the derived facts `$q.rank`, `$q.dtype`, `$q.stride_order`
  ([RFC 0018 § 5](0018_UniversalMatchDescriptor.md#5-layout-and-stride-order-criteria)), `$q.packed`, `$q.virtual` (an
  internal intermediate between matched nodes, not a graph input or output),
  `$q.is_runtime_pass_by_value` (its value arrives per execution rather than being baked into the
  graph, [RFC 0016](0016_RuntimePassByValueTensors.md)), and the precomputed scalar `$q.value_f32`
  (below). Whether the graph supplied an optional operand at all is a question asked with the
  `present` / `not_present` operators, not a field read off the tensor (§ 6.1).
- **Graph** — structural facts and graph-level flags of the matched graph: `$graph.node_count`, which
  pins an exact match, and `$graph.is_override_shape_enabled`, the graph's own opt-in to execute-time
  override shapes. That flag is the graph's state and is distinct from a matcher's
  `allow_override_shape`, which is the matcher's opt-in to accepting such a graph at all
  ([RFC 0018 § A.1](0018_UniversalMatchDescriptor.md#a1-the-umd-descriptor-object)).
- **Attributes** — a matched node's scalar attributes, named by the node's pattern `id`: an
  `{"id": "sdpa_fwd"}` node exposes `$sdpa_fwd.dropout_probability`, a `{"id": "conv"}` node
  `$conv.dilation`. An optional attribute is asked about the same way, with those same operators.
- **Kernel metadata** — `$kernel.<field>`, the values a UKD supplies for the fields its KMD declares
  (tile and vector constants, the dtype it targets, [RFC 0017 § 4](0017_UniversalKernelDescriptor.md#4-descriptor-formats)).
  These are the one namespace the pattern does not bind: they come from the engine's KMD, and a
  `$kernel.*` field a matcher reads MUST exist in it, so the matcher publishes the set of `$kernel.*`
  fields it reads and the loader checks them against the engine that pack names (§ 13.2). A matcher
  that reads them is evaluated per kernel
  ([RFC 0018 § 8](0018_UniversalMatchDescriptor.md#8-the-matcher-compilation-indexing-and-caching)).
- **Device properties** — `$device.<field>` such as `$device.lds_size` or `$device.warp_size`, for a
  check like an LDS budget; also not pattern-bound, since they come from the `Handle` rather than the
  graph (§ 5). The device facts hipDNN carries today are narrower than this vocabulary needs, so the
  device-property set is extended additively as the checks that need it land. Architecture is **not**
  here for AOT: it is a pack property gated at selection; a JIT pack may reference `$device.arch`.

**Precomputed fields.** Some tokens above are not stored on the graph: the schema layer derives them
once and publishes them as ordinary fields, so a matcher compares a value instead of re-deriving it.
`$q.packed` and `$q.stride_order` are the layout examples, standing in for `inferLayout`'s
contiguous-stride arithmetic. `$q.value_f32` is the other kind: a tensor's compile-time `value` is a
tagged union over eight differently-typed arms, and the expression language has no discriminator
syntax to unwrap one, so the schema layer coerces whichever arm is set to `f32` once and publishes
it as a single typed token — present only when the tensor carries a compile-time value at all, so a
criterion over it declines on a tensor that does not. A precomputed field is declared in the hipDNN
schema like any other field and versioned with it, so adding one is an additive schema change rather
than a per-pack extension point. Precomputed fields sit between the built-in operators and the
native-matcher escape hatch ([RFC 0018 §
6](0018_UniversalMatchDescriptor.md#6-the-native-matcher-escape-hatch)): reach for one when a check
needs a derived fact, and for the hatch only when it needs real C++.

### 6.1 The published field set (normative)

The grammar and table below are the complete set of roots and fields a match publishes, and are what
every consumer's references are validated against (§ 13.2). The two namespaces the pattern does not
bind are listed with them, because a consumer resolves all five against one environment.

```ebnf
var-ref      = "$" , ( tensor-ref | graph-ref | attr-ref | kernel-ref | device-ref ) ;
tensor-ref   = tvar , [ "." , tensor-field ] ;
tvar         = ident ;                          (* a pattern variable bound to a Tensor *)
tensor-field = "uid" | "rank" | "dtype" | "stride_order" | "packed" | "virtual"
             | "is_runtime_pass_by_value" | "value_f32"
             | "dims"    , "[" , uint , "]"
             | "strides" , "[" , uint , "]" ;
graph-ref    = "graph" , "." , ( "node_count" | "is_override_shape_enabled" ) ;
attr-ref     = node-id , "." , attr-name ;
kernel-ref   = "kernel" , "." , ident ;
device-ref   = "device" , "." , ident ;
uint         = digit , { digit } ;
```

| Namespace | Root | Bound by | Fields | Type |
|---|---|---|---|---|
| Tensor | a pattern variable (`$q`) | the pattern | `uid`, `rank`, `dtype`, `stride_order`, `packed`, `virtual`, `is_runtime_pass_by_value`, `value_f32`, `dims[i]`, `strides[i]` | `Tensor` / `Int` / `Dtype` / `IntArray` / `Bool` / `Float` |
| Graph | `$graph` | the pattern | `node_count`, `is_override_shape_enabled` | `Int` / `Bool` |
| Attributes | a node `id` (`$sdpa_fwd`) | the pattern | `<attr-name>` | scalar |
| Kernel | `$kernel` | the UKD, per candidate | `<field>` a UKD supplies ([RFC 0017 § 4](0017_UniversalKernelDescriptor.md#4-descriptor-formats)) | scalar |
| Device | `$device` | the `Handle` | `<field>` (`lds_size`, `warp_size`, …) | scalar |

- `graph`, `kernel`, and `device` are **reserved** namespace roots: a `tvar` and a node `id` MUST NOT
  use them, and the registry generator fails the build on an operand name that collides with one
  (Appendix B.3).
- **Presence is a question, not a field.** Whether an optional operand or attribute was supplied is
  asked with the `present` / `not_present` operators, which always evaluate and so are the one read
  an absent operand answers definitely (RFC 0017 § 5). No root carries a `.present` member. Asking
  it of a required operand is refused at compile, the answer being fixed by the pattern.
- **A `$` marks a reference.** Every reference to a bound field carries a leading `$`: tensors and
  their fields (`$q`, `$q.uid`, `$q.dims[2]`, `$q.rank`), a node's attributes
  (`$sdpa_fwd.dropout_probability` — the node id `sdpa_fwd` is bare, the reference carries the `$`),
  `$graph.node_count`, `$kernel.tile_m`, and `$device.lds_size`. Tokens without a `$` are literals:
  numbers, enum values (`"BFLOAT16"`, `"gfx942"`), opcodes, and layout aliases.
- What a **read of an absent** optional operand or attribute yields, and how the criteria language
  propagates it, are the reader's rules
  ([RFC 0018 § A.2](0018_UniversalMatchDescriptor.md#a2-variable-references-and-resolution)).

**The two spellings must widen together.** The native arm (§ 4.5) reaches the same five namespaces
through positional operands rather than `$`-names: a stage takes `MatchContext` (the device and
graph), `BoundTokens` (what the match bound), then `KernelDefinition` (the candidate kernel) — the
same access this table spells `$device`, `$graph`, `$<token>`, and `$kernel`. Every native stage
takes that one prefix, truncated to what exists at its point in the pipeline: the graph match takes
`MatchContext` alone, since it is what *produces* the bindings; a graph-scoped criterion adds
`BoundTokens`; a kernel-scoped criterion and a scorer add `KernelDefinition`. This correspondence is
normative. Widening one spelling without the other — adding a namespace here without a
corresponding operand there, or reordering the operands — makes the two arms disagree about what is
knowable at a given stage, which is the drift the shared order exists to prevent.

![A live graph matched against a declarative pattern, auto-binding tensors, dims, strides, and attributes](../images/umd_symbol_binding.svg)

## 7. Pattern Matching: Stage One

Matching a graph is two stages, and the pattern is the first of them. This section specifies the
pattern's half: when it is compiled, how engines are pruned before any pattern runs, and what the
bind step does. The second stage — a pack's criteria
evaluated over the published table, its per-kernel memoization, and the applicability-time cache
both stages share — is [RFC 0018 §
8](0018_UniversalMatchDescriptor.md#8-the-matcher-compilation-indexing-and-caching).

**Compiled once, at registration.** A pattern is authored as text and **compiled once** into an
in-memory structure: compiling resolves op-schema names into typed accessors and lays out the symbol
table the pattern will publish (§ 6.1). That compiled form, not the text, is what runs against live
graphs, and it is built at engine registration (§ 12) so a resolution failure is a load error rather
than a first-graph surprise. The compiled pattern is shared across every graph the engine sees; only
the binding result is per-problem. The native arm has no compile step — its symbol is resolved at
registration instead (§ 12), which is the same load-time failure at a coarser grain.

**Root-opcode indexing, over engines.** The compiled patterns are indexed by their root node's
opcode, so match cost does not grow linearly with the number of descriptors: a graph whose root op
is `sdpa_fwd` only consults engines whose pattern is rooted at `sdpa_fwd`. This is the index RFC
0017 § 16 calls for, and putting the pattern on the engine makes it coarser and therefore cheaper —
a miss prunes an engine and every pack naming it in one step, without loading a single UMD. Only the
surviving engines pay for criteria at all. A native match carries no root opcode to index on
(§ 4.5), so an engine on that arm is not pruned this way and pays the arch gate below instead.

**Stage one: bind, at most once per engine per (graph, device).** The engine's match walks the
graph against the per-graph UID-to-producer and UID-to-consumers index (§ 5), and publishes the
bound symbol table (§ 6). A graph the match declines declines the engine outright: `isApplicable`
returns false, no criteria are evaluated, and the catalog is empty. Because the match is the
engine's, this cost is paid once no matter how many packs name that engine, which is the structural
saving the split buys.

**The arch gate runs first, and the match is lazy.** A pack whose `arch` list excludes the running
device is skipped before the match is attempted, and the match runs on the first pack that clears
that gate. An engine whose packs are *all* arch-excluded therefore never matches at all: there is no
device it could serve, so walking the graph could only reach the same answer more slowly. The
binding is computed at most once per (graph, device) and reused by every surviving pack, so
laziness changes when the work happens, never how often. This is RFC 0017 § 8.1's order rather than
a re-ordering of it: its step 4 resolves the packs and applies the arch gate as it goes, and its
step 5 runs the match lazily on the first pack that cleared that gate, reasoning the all-excluded
case the same way.

## 8. Knobs

A knob is **a KMD field the engine chooses to expose**, a name and nothing more; the KMD
already declares the field's type and default (RFC 0017 § 4). The UED's contract:

- Only **KMD field names** may appear in `knobs`. A name no KMD field matches is a **load
  error** (§ 13.2).
- Exposing a field is **additive and reversible**: add a name to expose it, remove it to
  withdraw it.
- A knob's **legal values come from the catalog**, not the KMD's theoretical range; that is, the
  set of values the field takes among the kernels matching a given graph.
- A knob's **default is the heuristic's top-ranked choice**, not a constant.
- `knobs` governs only what the UED *declares*. hipDNN's reserved `global.` knobs
  ([RFC 0004](0004_EngineConfigKnobs.md)) are a separate namespace a descriptor-backed engine
  implements like any other engine; the two do not overlap.

## 9. Behavior and Numerical Notes

`behavior_notes` and `numerical_notes` are hipDNN's existing per-engine annotations
([RFC 0010](0010_BehaviorNotes.md)) carried on the UED. They are optional string lists. This
RFC adds no new note vocabulary; it specifies only that the UED is where a descriptor-backed
engine declares them.

## 10. Engine Membership

A **UKD names no engine.** Its engine membership is determined by the **sibling UED referenced
by its KDP**: the KDP carries `"engine": "<UED id>"`, and every child UKD inherits it, along
with the pack's matchers and dispatch and the engine's heuristic and metadata schema (RFC 0017
§ 4). The membership chain is **UKD -> KDP -> UED**, bound by the descriptor UUID `id`; there is no
direct UKD->UED reference.

One UED is typically shared by many KDPs, and so serves many UKDs: one engine, one KMD, and at
most one UHD, ranking a whole catalog of kernels over one feature space.

Every pack naming the engine inherits its **one `graph_match`** (§ 4.2), so membership also fixes
the graph shape a pack can constrain: a KDP does not narrow the shape, only what its criteria
demand of the symbols the match published (§ 6). A pack under an engine with no `graph_match`
inherits an empty binding and its criteria carry the whole applicability decision.

## 11. When a UED Is Loaded and Registered

The loading *mechanism* (discovery, parsing, the shared load path, and when descriptor bodies are
parsed) is out of scope. What a loaded UED must satisfy to be valid is specified in **§ 13**
(structural checks, cross-descriptor resolution, and uniqueness). This section fixes only the
loading facts the UED's host contract depends on: the ingestion paths (§ 11.1), the one
registration-timing guarantee (§ 11.2), and concurrency (§ 11.3).

### 11.1 Ingestion paths

A UED may reach the provider by either RFC 0017 ingestion path, build-time (AOT) or runtime
drop-in (§ 15), and both converge on the same UED schema, validation (§ 13), and registration
(§ 12) defined here. A load always builds the **complete** descriptor set from scratch rather than
merging into an in-memory set, so every UED is validated against every other on each load
(§ 13.2.1); a drop-in descriptor is picked up by the next such load. What triggers a load, where
descriptors live, how they are discovered, and how a file is recognized as a UED (the descriptor
kind is tracked externally, not in-band, § 4) are part of the loading mechanism and are out of
scope for this RFC.

### 11.2 Registration timing

The one timing guarantee this RFC fixes is that every valid UED is registered (name -> 64-bit id,
§ 12) before any graph is served, because the host must be able to enumerate the engines it may
select among. Registration of the loaded UED set therefore completes before engine enumeration or
selection begins. A KDP's `engine` reference correspondingly resolves against an
already-registered UED; how a provider orders UED and KDP loading to guarantee that is a mechanism
detail.

### 11.3 Concurrency

Engine registration occurs on the provider's plugin-load path, which is already serialized, so it
introduces no new concurrency model and requires no additional locking beyond that existing
serialization.

## 12. Engine Registration

**Registration** is the process that turns a validated UED into an engine the provider exposes
to hipDNN. The destination is the provider's existing engine list (the one that already holds
hand-written engines), and registration is the descriptor-driven equivalent of the provider's
hand-written engine-registration path. For each UED that passes validation (§ 13), registration:

1. **Derives the engine id**: the 64-bit hash of the UED `name` (§ 3).
2. **Resolves `graph_match`** (§ 4.2), by arm. For **`nodes`**, the block is resolved against the
   op-schema registry (§ 5) and compiled once into the in-memory form that runs against live
   graphs, laying out the symbol table it will publish (§ 6.1, § 7); the § 4.3.3 checks are exactly
   the ones this step performs, and the compiled form is shared across every graph the engine sees
   and across every pack naming it. RFC 0017 § 8.1 step 3 defers the compile to registration and
   names this section as the authority for it, so a name the op-schema registry does not declare is
   a **load error naming the UED and the node**, not a first-graph surprise; the compiled form then
   sits in the descriptor cache that 0017 § 8.1 reads from. For **`native`**, there is nothing to
   compile: the symbol is looked up in the provider's graph-match registry, and an unregistered
   symbol is a load error naming the UED and the symbol (§ 13.2). An **absent** `graph_match`
   resolves to the empty binding and skips this step. Either way the failure is at registration,
   before the engine's id is advertised.
3. **Instantiates one generic engine**: a single engine implementation that satisfies hipDNN's
   existing engine contract from descriptor data rather than hand-written code, one instance per
   UED, bound to that UED's descriptors: its `heuristic` (UHD, when it ships one) and `metadata`
   (KMD) references, its resolved `graph_match`, and the KDPs whose `engine` field names it.
4. **Indexes the engine by its pattern's root opcode** (§ 7), so a graph whose root op no engine
   pattern is rooted at prunes without a match attempt. Only the declarative arm is indexable; an
   engine on the native arm is not entered here and is reached on every graph.
5. **Adds the engine to the provider's engine list** and records the name -> id mapping, so the
   host can enumerate the engine and diagnostics / support claims
   ([RFC 0015](0015_EngineSupportClaims.md)) key on the real name rather than a hex id.

A UED that fails step 2 registers no engine at all (§ 15): a match that does not resolve cannot
decline a graph honestly, so the engine is dropped rather than exposed as one that never matches.

Nothing in the host-facing engine contract changes: a descriptor-backed engine is selected and
driven exactly as a hand-written one (RFC 0017 § 3, § 8). This RFC specifies registration (how
a validated UED becomes an exposed engine) and the generic engine's identity and descriptor
binding. Populating the generic engine's plan builder is registration's responsibility; the plan
builder's *internal* behavior over UDD and UKD data is defined by those descriptors' own
specifications, not this one.

## 13. Validation

Every check below is expected to run at **both build time and run time**. Build-time validation
catches errors before shipping; run-time validation ensures integrity of the loaded set and is
**required** for drop-in UEDs (§ 11), which never pass through the build. The checks divide into
**structural** (the field contract of § 4.2) and **semantic** (cross-descriptor); the
implementation may fold them into one pass.

### 13.1 Structural validation

The **structural check** validates the UED against the schema (§ 4.2, § 14.3), confirming six
things:

- **`version` is one the schema accepts**: the schema lists the exact `major.minor` values it
  recognizes, and the UED's `version` must be one of them.
- **each present field is well-formed**: correct type and pattern.
- **no unknown field is present**: an unknown field is a hard rejection, at the top level, inside
  `graph_match`, and inside a `nodes` node object alike.
- **`graph_match`, when present, carries exactly one arm**: `nodes` or `native`, never both and
  never neither.
- **the `nodes` block is well-formed**, when that is the arm: the § 4.3.2 rules — non-empty, `kind`
  is `"op"`, node ids unique and non-reserved, each pattern variable bound exactly once, `?` only
  on an operand. These need no other descriptor, so they belong here; the rules that need the
  op-schema registry are § 4.3.3 and run in § 13.2.
- **`native` is a non-empty string**, when that is the arm. Whether the symbol exists is semantic
  and runs in § 13.2.

Because fields added after the earliest version are optional in this schema, passing this check
means the UED is structurally valid for *some* supported version, not necessarily for the one it
declares (version-accurate validation below closes that gap). The § 4.2 JSON Schema is the recommended
way to express and run the check (`additionalProperties: false` gives the unknown-field rejection),
but the contract is normative independently of that mechanism.

**Version-accurate validation (facilitated, not required).** The superset schema alone does not
check that a UED is valid for the specific version it declares, since a later version's field is
optional in the schema and so passes structurally. The schema carries the data to make a finer
check *possible*: each property's `addedInVersion` (§ 4.2) records the version it was introduced in,
and the schema's `required` set records the always-required fields. From these, the **minimum
version a UED needs** is computable: it is the newest `addedInVersion` among the fields the UED
carries. Comparing that minimum to the declared `version` flags an error in either direction: a
declared version *below* the minimum means the UED carries a field newer than it claims, and a
declared version *above* the minimum means the UED over-declares and will needlessly fail to load
on older runtimes that could serve it (§ 14.2). A consumer can also confirm the required fields are
present. This RFC describes the capability and guarantees the data exists to support it; it does not
require any consumer to perform it.

### 13.2 Semantic validation (cross-descriptor)

These cannot be expressed in JSON Schema because they depend on other descriptors; each is
performed at build time and run time alike:

- **Reference resolution.** A UED's `metadata` (KMD) and, when present, its `heuristic` (UHD) must
  each resolve to a loadable descriptor of the correct kind; a dangling reference is an error. This
  is an *existence* condition: the referent must be resolvable, not necessarily parsed.
- **`knobs` must be a subset of KMD field names.** A knob name no KMD field matches is an error
  (RFC 0017 § 4). Unlike reference resolution, this reads the KMD's declared field set, so the
  referenced KMD must be resolvable **to its field set**, more than existence.
- **Pattern names resolve against the op-schema registry** (declarative arm). Every `op`, operand
  name, and result name in `graph_match.nodes`, and every `?` against the registry's optionality —
  the § 4.3.3 rules. This runs at pattern compile (§ 12) and, like the checks above, at build time
  and run time alike. It is the check that catches a graph-schema change renaming an operand out
  from under a shipped pattern (Appendix B.6).
- **The native symbol is registered** (native arm). `graph_match.native` must name a function
  registered in the loading provider's graph-match registry. An engine naming an unregistered
  symbol is **dropped at read time**, before its id is advertised, so a descriptor set that
  half-ships never advertises an engine it cannot serve. The diagnostic names the UED and the
  symbol. A provider SHOULD pre-flight this alongside the other symbol checks rather than paying an
  exception when the match is first attempted.
- **The published symbol set is validated per consumer pair** (declarative arm only). A pattern
  publishes a set (§ 6.1), and every descriptor written against it — a UMD listed by a pack naming
  this engine, that pack's UDD, and the engine's own UHD `features_signature` — is checked against
  that set. This is the UED half of the check; the reader's half, including that a failure names
  the matcher, the engine, the reference, **and the pack that paired them**, is
  [RFC 0018 A.5](0018_UniversalMatchDescriptor.md#a5-compile-time-validation-normative). The
  pairing is cached on `(matcher, engine)` and re-run when either side changes, so a pattern edit
  that drops or renames a bound variable invalidates every consumer written against it, loudly, at
  load. **Under the native arm this check cannot run**: the token set is not known until the
  function returns on a live graph (§ 4.5), so a consumer's `$`-reference is unvalidated at load and
  fails closed at match time instead. That is the guarantee the hatch trades away, and the reason a
  UED should leave it once the declarative arm can express its pattern.
- **Uniqueness (§ 13.2.1).** No two loaded UEDs may share a descriptor `id`, and independently
  none may share a `name`, except for the content-identical case (§ 13.2.1).

The full cross-descriptor reference-integrity check (which references must resolve, and to what)
spans multiple descriptor types and is best specified at a higher level than the UED format.
This RFC fixes only that a UED's own `heuristic` and `metadata` references are subject to it, at
both build and run time.

#### 13.2.1 Duplicate detection (descriptor `id` and `name`)

Duplicate detection runs over the complete loaded set (§ 11.1), so a drop-in UED is checked against
every other UED exactly as any other.

**The version check precedes duplicate detection.** A UED whose `version` the runtime does not
support is logged (warning) and dropped by the accept rule (§ 14.1) *before* uniqueness is
evaluated, so it never participates in `id` or `name` detection. A UED with a duplicate `id` or
`name` but an unsupported `version` is therefore dropped for its version alone, and the UEDs it
would have conflicted with are retained (unless they separately conflict with other
supported-version duplicates).

Two independent uniqueness invariants hold across all loaded UEDs: no two share a descriptor
`id`, and no two share a `name` (equivalently its 64-bit hash). On collision under either
invariant, **all** UEDs involved are unloaded (logged, § 15), not keep-the-first, with one
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

### 13.3 UED-load vs KDP-load boundary (guidance)

The engine-scoped checks above (§ 13.1-13.2) belong at **UED load**. For contrast, these are
**KDP-load** (pack-scoped) concerns, governed by the KDP's own specification and listed only to
place the boundary:

- the KDP's `engine`, `matchers`, and `dispatch` references resolve (the `engine` ref
  resolving to a registered UED is a KDP-load concern, not a UED concern);
- per-kernel `$kernel.*` matcher pins against the KMD (RFC 0017 § 5);
- **duplicate kernel keys**, engine-wide but triggered by loading a pack's UKDs, so detected
  at KDP/UKD load (RFC 0017 § 10).

This boundary is guidance, not a hard split; a shared loader may fold both into one pass.

## 14. Versioning and Compatibility

A UED's compatibility is carried by its `version` field (`major.minor`). Each descriptor file type
versions independently (a KMD and a UDD advance on their own schedules). This section defines the
accept rule, what `major` and `minor` are permitted to mean, and how the schema backs validation.

> **Tightens RFC 0017 § 4 (compat mechanism).** RFC 0017 gives the accept/reject *policy* and a
> coarse field-evolution rule but leaves the unknown-field policy and the runtime's version source
> unspecified. This section keeps 0017's accept rule unchanged and pins down the rest; it does not
> override 0017, it makes the deferred detail concrete.

### 14.1 The accept rule

The runtime carries a supported `major.minor` for the UED type; concretely, the UED schema embedded
in the provider build, which enumerates the versions it supports (§ 14.3). A UED is accepted iff:

- **`file.major == provider.major`**, and
- **`file.minor <= provider.minor`**.

`major` and `minor` are compared as integers, not lexicographically or as a decimal fraction, so
`1.10` is newer than `1.9`. Otherwise the UED is rejected. A UED is refused, never silently
reinterpreted.

This yields **backward compatibility only**: an older-minor UED loads on a newer-minor provider;
a newer-minor UED is rejected on an older-minor provider (it may carry fields that runtime cannot
interpret). A **major mismatch is a hard break**: at this time the runtime supports **exactly one
major version**, and *every* UED of a different major (older or newer) is dropped (logged and
skipped, § 15). There is no multi-major support; a major bump orphans existing UEDs until they
are re-authored, which for a large descriptor set is expected to be a bulk re-emit from the
authoring toolchain rather than per-file hand-editing. Because major bumps should be rare
(§ 14.2), this is an accepted cost, revisited only if descriptor longevity across a break becomes
a requirement.

### 14.2 What `major` and `minor` are permitted to mean

- **Minor bump: additive, absence-safe changes only.** A minor may add a field **only if its
  absence is well-defined as "the behavior before the field existed"**: a UED at an earlier minor
  omits it, and the runtime reproduces prior behavior via the pre-addition code path (the
  semantics of absence *is* the old implementation, so no default table is needed). A minor may
  not remove, rename, retype, or change the meaning or permitted values of an existing field.
- **Major bump: everything else** (removing, renaming, retyping, making an optional field
  required, or changing a field's meaning/permitted values). These are the changes where an old
  reader would misinterpret a file, which the hard `major ==` break (§ 14.1) prevents.

**`ued/1.0` is this rule applied to this RFC's own change.** Both members this RFC adds are
**additive and absence-safe**, so the format stays on the `1.0` line and no shipped UED is
orphaned. `sdk_version` defaults to `1.0`. `graph_match` is optional, and its absence is
well-defined as the behavior before the member existed: an engine that states no graph shape binds
nothing and leaves applicability to its packs' criteria (§ 4.2). Neither member removes, renames,
retypes, or changes the meaning of an existing field, and neither makes an optional field required.

**An earlier draft made `nodes` a required top-level field and bumped to `ued/2.0` on that
ground.** That is why the rule is worth restating here: a required field fails the absence-safe
test outright, since there is no "behavior before the field existed" for an engine with no
pattern, so it would have forced a major and dropped every `1.0` UED rather than reinterpreting
it. Nesting the pattern inside an optional `graph_match` removes that forcing — the declarative
arm is fully specified (§ 4.3) without being mandatory — and the format's one supported major
stays `1`. That matters more now than when the draft was written: UEDs ship, and a major the
runtime does not accept would be a spec no descriptor could conform to.

**Authors should stamp the lowest version their UED needs**, so it stays loadable on the oldest
runtime that can serve it and never carries a field its version does not define. The structural
check (§ 13.1) does not enforce this, but the `addedInVersion` data makes the lowest version a UED
needs computable, so a consumer that wants it can flag an over-declared version (§ 13.1).

### 14.3 The schema's role in versioning (a supporting design, not a mandate)

The schema (§ 4.2) is a single file whose `version` enum lists the accepted `major.minor` values.
That enum is one expression of the accept rule (§ 14.1): because the runtime supports exactly one
major version, every enumerated version is of that major, so the enum and the `major` equality /
`minor <=` comparison describe the same policy. A runtime enforces that comparison directly whether
or not the schema is available to it.

Maintaining the contract as one schema file supports a single source of truth that can drive
validation in two places from one artifact:

- **Build time**: the authoring toolchain can validate every authored UED.
- **Run time**: the provider can carry the schema and run the same validation on ingested UEDs,
  since drop-ins (§ 11) bypass the build.

This RFC recommends the single-schema approach so an off-the-shelf validator can enforce the
structural superset (§ 4.2), and the `addedInVersion` data makes a version-accurate check possible
for a consumer that wants one (§ 13.1); but it does **not** prescribe how the schema is carried in
the provider, which validator is used, that an implementation must use JSON Schema at all, or that
any version-accurate check be performed.

## 15. Lifecycle and Operational Policy

- **Load failure => log and skip.** A UED that fails validation is **logged as an error and
  skipped**; the provider does not hard-fail, and the skipped UED registers no engine. This
  matches RFC 0017's "reported in load diagnostics like any other exclusion" and the
  duplicate-key "logged and dropped" pattern.
- **Concurrency => the guarded plugin-load path** (§ 11.3), not a per-handle resource manager.
  Engine registration adds no new concurrency model.
- **`HIPDNN_DISABLE_ENGINES` => skip at load.** A disabled engine is skipped before registration:
  it never loads and never claims its name or id. A list entry may be any of three identifiers
  (the UED `name`, its 64-bit hash, or the UED UUID `id`), and the matcher checks all three.
  Because a skipped UED never claims its name or id, this can also serve as a recovery lever for a
  collision (§ 13.2.1): disabling the unwanted participant by an identifier that singles it out
  (for a same-`name` collision, its `id`) frees the contested name/id so the other loads. A
  collision whose participants share both `name` and `id` cannot be separated this way. The
  finer-grained `HIPDNN_DISABLE_KDPS` / `HIPDNN_DISABLE_UKDS` (RFC 0017 § 10) are governed by their
  own descriptors.
- **Drop-in trust => out of scope.** Trust and enablement for untrusted drop-in descriptors are
  out of scope (RFC 0017 § 16 and § 17 Q1, which defer source-trust rules to the delivery
  follow-up); this RFC adds no trust policy.

## 16. Testing

Reusing the existing tiers (RFC 0006 harness) and RFC 0017 § 14.1's descriptor-pipeline
fuzzing, this RFC adds UED-specific coverage.

**Unit tests:**

- **Superset structural validation** (§ 13.1): valid and invalid field sets; missing required
  fields (including `version`); a `version` value the schema does not enumerate is rejected;
  malformed `id`/`name` patterns; **unknown field is rejected**; optional fields absent.
- **Schema `addedInVersion` completeness** (§ 4.2): every property in the schema carries an
  `addedInVersion` keyword whose value the `version` enum accepts, so version-accurate validation
  is possible.
- **Version-accurate validation, if performed** (§ 13.1): a UED whose `version` is the earliest the
  schema enumerates but that carries a field whose `addedInVersion` is later is rejected even
  though the superset schema accepts it.
- **Schema build/runtime parity** (if the single-schema design of § 14.3 is used): the schema
  embedded in the provider parses to the same JSON as the repository's canonical file and as
  the inline copy in § 4.2 (a CI check comparing parsed structure, not bytes, so formatting and
  comments do not matter), so the build-time and runtime validators enforce the same contract.
- **Version accept rule** (§ 14.1): matrix of `file` vs `provider` `major.minor` read from the
  `version` field: same major/older-or-equal minor loads; newer minor rejected; any major mismatch
  dropped.
- **Semantic checks** (§ 13.2): dangling `metadata`; a dangling `heuristic` when one is named; a
  `knobs` entry absent from the KMD.
- **`graph_match` arm selection** (§ 4.2, § 13.1): an object carrying both `nodes` and `native` is
  rejected; one carrying neither is rejected; an absent `graph_match` loads and the engine binds an
  empty table, admitted or declined by its packs' UMDs alone.
- **Native arm resolution** (§ 4.5, § 13.2): an engine naming an unregistered `graph_match.native`
  symbol is dropped at read time and never advertises its id; a registered one loads; the match
  returning `nullopt` declines the graph and yields an empty catalog while leaving other engines
  unaffected. The drop must be asserted against the advertised engine list, not merely a log line.
- **Match laziness and the arch gate** (§ 7): an engine whose packs are all arch-excluded never
  invokes its match; an engine with one surviving pack invokes it exactly once per (graph, device)
  however many packs then read the binding.
- **`nodes` structural well-formedness** (§ 4.3.2): empty `nodes`; a `kind` other than `"op"`; a
  duplicate node `id`; a node `id` or pattern variable colliding with `graph`/`kernel`/`device`; a
  variable bound by two nodes' results; a `?` on a result binding; a binding that is not `$ident`.
  Each is a load rejection naming the node.
- **Pattern registry resolution** (§ 4.3.3, § 12): an unknown `op`; an operand name the op does not
  declare; a result name the op does not declare; a `?` on an operand the registry declares
  required; an opcode set whose members disagree on an edge name or its optionality. Each fails
  pattern compile at registration, and the UED registers no engine.
- **Pattern compile happens once, at registration** (§ 12): a UED with an unresolvable pattern name
  fails at load rather than on the first graph, and a valid one compiles exactly once however many
  graphs and packs exercise it.
- **Published-symbol-set resolution** (§ 6.1, § 13.2): a UMD, a UDD, and a UHD
  `features_signature` each referencing a symbol the engine's pattern does not publish are rejected
  at load, naming the reference, both descriptors, and the pack that paired them; a pattern edit that renames a bound variable
  invalidates every consumer written against it. Optional-`?` binding is covered directly: a graph
  omitting the operand still matches and `not_present` answers true for it, while a graph
  omitting a **required** operand declines the engine.
- **Published-symbol-set resolution is declarative-only** (§ 4.5, § 13.2): the consumer-pair check
  above is skipped for an engine on the native arm, and a UMD naming a token that arm does not
  return fails closed at match time rather than at load. The test pins the *difference*, so the
  guarantee the hatch trades away stays visible rather than being assumed absent.
- **`sdk_version` floor** (§ 4.2): a UED below the graph's reported floor declines the whole engine
  before binding, taking every pack naming it; a UED above the runtime's own graph schema is
  refused at load; an omitted `sdk_version` reads as `1.0`.
- **Duplicate detection, drop-all** (§ 13.2.1): two UEDs differing in content but sharing an
  `id`; two sharing a `name`; two sharing both; a UED colliding by name with a built-in engine. In
  each case *every* colliding UED is dropped and named in diagnostics, and no engine is left bound
  to an arbitrarily-chosen definition.
- **Content-identical duplicates load** (§ 13.2.1): two UEDs with the same `id` and semantically
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

## 17. Open Questions

1. **Alternative patterns under one engine.** A UED carries exactly one `nodes` block (§ 4.3), so a
   family spanning two topologies splits into two engines. A `patterns[]` list matched in order would
   avoid the split, but needs an answer for which arm's binding is published, whether the published
   set is the union or the intersection, and what a criterion referencing a symbol only one arm binds
   means. Is the split acceptable, or is that design worth doing?
2. **`criteria` inside `graph_match`.** The declarative arm is expected to carry a `criteria` key
   beside `nodes`, but every criteria expression this series specifies lives on a UMD
   ([RFC 0018](0018_UniversalMatchDescriptor.md)). What an engine-scoped `criteria` would express
   that a UMD listed by every pack does not, and how the two interact when both are present, is
   unsettled. It is named here because the native arm fuses topology and value checks in one
   function (§ 4.5), so the question is already live in shipped code even though no format carries
   the key yet.

---

## 18. Glossary

- **UED (Universal Engine Descriptor):** one engine, comprising a stable identity (`name` + UUID
  `id`), the `graph_match` stating how it matches and what it binds, the KMD field names it exposes
  as knobs, and its behavior/numerical notes. Names its one KMD, and optionally one UHD, by id.
  1:1 with a hipDNN engine.
- **`graph_match`:** the UED member holding stage one, in exactly one of two arms: the declarative
  **structural pattern** (`nodes`, § 4.3) or the **native escape hatch** (`native`, § 4.5). Absent
  means the engine binds nothing.
- **Structural pattern:** a UED's `graph_match.nodes` block — the op nodes and named operand/result
  edges the engine matches against the graph, connected implicitly through shared pattern variables
  (§ 4.3). One per engine; it runs at most once per engine per graph (§ 7) and its binding is what a
  UMD's criteria, a UDD's formulas, and the UHD's features are all written over.
- **Native match:** a registry-resolved function named by `graph_match.native`, returning the
  binding or declining the graph (§ 4.5). The engine-scoped escape hatch, distinct from a UMD's
  pack-scoped native criterion, which reads the binding rather than producing it.
- **Pattern variable:** a `$name` in the `nodes` block, bound by exactly one edge and read by any
  number of others; the unit of both connectivity (§ 4.3) and publication (§ 6.1).
- **Engine id (64-bit):** the hipDNN-facing engine identifier, derived (FNV-1a) from the UED
  `name`; what the plugin reports to the backend and what selection/diagnostics key on.
- **Descriptor UUID `id`:** the cross-reference identifier a descriptor carries; how a KDP
  names its UED and a UED names its UHD/KMD. Distinct from the engine id.
- **Generic engine:** the single C++ engine class that satisfies hipDNN's `IEngine` contract
  from descriptor data, one instance per UED.

---

## 19. Appendix: Fully-Populated UED Examples

The examples in § 4 keep optional fields minimal. This appendix shows UEDs that populate **all**
optional fields, which requires knowing the KMD they reference, since every `knobs` entry must
name a field the KMD declares. § A.1 sketches only as much of the KMD as the UED depends on; the
KMD format itself is outside this RFC's scope. Both UEDs below reach their pattern through
`graph_match` (§ 4.2) on the declarative arm (§ 4.3); its operand and result names are the ones
`sdpa_attributes.fbs` declares through the op-schema registry (Appendix B), minus the
`_tensor_uid` suffix.

### A.1 What a UED needs from its KMD

A UED does not define fields. It references one KMD by id (its `metadata`) and exposes a subset
of that KMD's field **names** through `knobs`. For the purposes of a UED, a KMD is simply a
named list of field names an engine's kernels vary over:

```jsonc
{
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
matching one is valid; one matching nothing is a load error (§ 13.2). Field types, defaults, and
catalog semantics are KMD concerns (RFC 0017 § 4-5), not properties the UED reads.

### A.2 Fully-populated UED: the `attention_dense` engine

A UED that populates **all** optional fields. Each name in `knobs` is a field the referenced KMD
declares (§ A.1); the notes are RFC 0010 annotations.

```jsonc
{
  "version":         "1.0",
  "id":              "7d4c2a9e-3b6f-4e1a-8c5d-9a2f7b0e6c14",   // UUID; KDPs name this via "engine"
  "name":            "rocke:attention_dense_fwd",              // globally-unique, scoped; hashed to the 64-bit engine id
  "sdk_version":     "1.0",                                    // graph schema this pattern was authored against
  "heuristic":       "2b7a4e1c-6f3d-4a8e-9c2b-5d1f0a7e8b93",   // this engine's one UHD
  "metadata":        "9c53b6b0-9a1e-4b1d-8b5c-7e2d9a6f3c40",   // the KMD (§ A.1)
  "graph_match": {                                             // declarative arm (§ 4.3)
    "nodes": [                                                 // the graph shape
      {"kind": "op", "id": "sdpa_fwd", "op": "sdpa_fwd",
       "operands": {"q": "$q", "k": "$k", "v": "$v",
                    "attn_mask": "$attn_mask?",                // optional: bound, and declined by criteria
                    "scale":     "$scale?"},
       "results":  {"o": "$o"}}
    ]
  },
  "knobs":           ["block_n", "waves_per_eu", "num_persistent"],  // all are KMD field names
  "behavior_notes":  ["runtime_compilation"],
  "numerical_notes": ["tensor_core", "reduced_precision_reduction"]
}
```

This exposes three KMD fields as knobs, carries two note lists, and binds seven symbols' worth of
tensors — `$q`, `$k`, `$v`, `$o` unconditionally and `$attn_mask`, `$scale` optionally
(§ 6.1). A KDP joins the engine with `"engine": "7d4c2a9e-3b6f-4e1a-8c5d-9a2f7b0e6c14"`.

### A.3 A minimal engine, for contrast (all optional fields omitted)

The same engine with only required fields: no knobs, no notes, and no `sdk_version`, which then
defaults to `1.0`. `graph_match` is optional but populated here, since an engine that omits it
binds nothing at all (§ 4.2) and would make a poor contrast — the minimal form still states a
graph shape, here the required operands only, which narrows the engine to graphs supplying nothing
else the pattern would have to bind:

```jsonc
{
  "version":   "1.0",
  "id":        "7d4c2a9e-3b6f-4e1a-8c5d-9a2f7b0e6c14",
  "name":      "rocke:attention_dense_fwd",
  "heuristic": "2b7a4e1c-6f3d-4a8e-9c2b-5d1f0a7e8b93",
  "metadata":  "9c53b6b0-9a1e-4b1d-8b5c-7e2d9a6f3c40",
  "graph_match": {
    "nodes": [
      {"kind": "op", "id": "sdpa_fwd", "op": "sdpa_fwd",
       "operands": {"q": "$q", "k": "$k", "v": "$v"}, "results": {"o": "$o"}}
    ]
  }
}
```

Both load and register identically; § A.2 surfaces knobs and notes that § A.3 leaves unset, and
binds two optional operands § A.3's pattern does not name — so a criterion on `$attn_mask` resolves
under § A.2 and is a load error under § A.3 (§ 13.2).

---

## Appendix B: Op-Schema Registry Generation

The op-schema registry (§ 5) is the table the matcher
consults to reconstruct a UID-centric graph's edges and to auto-bind symbols
(§ 6, § 6.1).
It is **generated from FlatBuffers field annotations on the graph schema**, never hand-maintained and
never inferred from field-name conventions, so the binding contract for an operand lives in the same
`.fbs` edit that adds the operand and cannot silently drift from the graph definitions.

### B.1 Attribute declarations

Four custom attributes are declared once in the graph schema. FlatBuffers requires an `attribute`
declaration before an attribute may be used, and declared attributes — on a table or on a field — are
retained in the binary reflection schema (`.bfbs`), which is what the generator reads.

```fbs
attribute "umd_opcode";         // table: the op's UMD-facing opcode (e.g. "sdpa_fwd")
attribute "umd_input_tensor";   // field flag: this field is an input (operand) tensor UID
attribute "umd_output_tensor";  // field flag: this field is an output (result) tensor UID
attribute "umd_name";           // field string: the name the UMD binds it by (e.g. "q", "o")
```

`umd_opcode` is a **table-level** attribute (applied in parentheses after the table name); the other
three are field-level. No separate operand/result type attribute is needed: an operand or result always
binds a `Tensor` — a tensor UID is the only edge kind in a UID-centric graph
(§ 5) — and every scalar attribute is precisely an
unannotated scalar field (B.3). The `long` UID field type and the `umd_input_tensor`/`umd_output_tensor` flag
together already fix what the binding is.

### B.2 Annotated schema

Each op's attribute table annotates its UID fields next to the field they govern. Optionality is **not**
re-annotated: a UID field's `= null` default (an optional field) already encodes it, so the generator
derives required-vs-optional from the field's presence semantics rather than a fourth attribute.

```fbs
table SdpaAttributes (umd_opcode: "sdpa_fwd") {              // table-level opcode shorthand
  q_tensor_uid:long (umd_input_tensor, umd_name: "q");          // required input
  k_tensor_uid:long (umd_input_tensor, umd_name: "k");
  v_tensor_uid:long (umd_input_tensor, umd_name: "v");
  o_tensor_uid:long (umd_output_tensor, umd_name: "o");         // required output
  attn_mask_tensor_uid:long = null (umd_input_tensor, umd_name: "attn_mask");  // optional input
  // ... other optional UID operands, likewise annotated ...
  dropout_probability:float = null;                        // unannotated -> scalar attribute
  alibi_mask:bool = false;                                 // unannotated -> scalar attribute
  causal_mask:bool = false;
}
```

### B.3 Field classification (normative)

For every table reachable from the `NodeAttributes` union (which already maps each opcode to its
attribute table), the generator reads the table's `umd_opcode` (the entry's opcode key; when absent it
falls back to the table type name) and classifies each field by these rules; a violation **fails the
build** rather than emitting a wrong registry:

| Field carries | Classified as | Requirements |
|---|---|---|
| `umd_input_tensor` + `umd_name` | input (operand) edge for that name | field type MUST be an integer UID (`long`); `umd_name` MUST be non-empty |
| `umd_output_tensor` + `umd_name` | output (result) edge for that name | field type MUST be an integer UID (`long`); `umd_name` MUST be non-empty |
| neither flag, a **scalar** field | scalar attribute, named by the field name | — |
| neither flag, a **non-scalar** field (vector, sub-table, union, string) | skipped (not a UMD scalar) | — |

- **Optionality** is derived, not annotated: a field with a `= null` default (an optional UID or an
  optional scalar) is optional; it supplies the `?`-binding of § 4.3 and is what the
  `present` / `not_present` operators of § 6.1 report on.
- **Build errors (fail closed):** `umd_input_tensor` and `umd_output_tensor` on the same field; `umd_name` without
  either flag; `umd_input_tensor`/`umd_output_tensor` on a non-integer field; a duplicate `umd_name` within one op;
  an input/output tensor whose name collides with a reserved token
  (§ 6.1); or a duplicate `umd_opcode` across
  ops.
- **Scalar attribute value kind.** A scalar attribute carries its value kind for compile-time type
  checking ([RFC 0018 A.5](0018_UniversalMatchDescriptor.md#a5-compile-time-validation-normative)): integer fields bind as `Int`,
  float/double as `Float`, `bool` as `Bool`, and an **enum-typed** field as `Dtype`, carrying the
  enum-value name string (e.g. `diagonal_alignment` → `"TOP_LEFT"`). This mirrors the tensor `dtype`
  representation and lets a criterion compare an enum attribute against a literal enum name.
- **No name-suffix inference.** `PointwiseAttributes::axis_tensor_uid` is a plain axis index, not a
  tensor UID; because it carries no `umd_input_tensor`/`umd_output_tensor` it is classified a scalar attribute,
  exactly as intended — nothing keys off the `_tensor_uid` suffix.
- **Scalar attributes need no annotation, and are still fully bound.** An annotation carries the two
  facts that cannot be inferred for an *edge*: that a `long` field is a tensor UID rather than a plain
  integer (`q_tensor_uid` and `left_bound` are the same type, distinguishable only by the flag), and a
  bind name distinct from the field name (`"q"` vs `q_tensor_uid`). A scalar needs neither: it is a
  non-edge by elimination, and its bind name *is* its field name, which reflection already reports. So
  every unannotated field is auto-bound in the Attributes namespace as `$<node_id>.<field_name>`
  (§ 6.1) with its reflected type and its
  `= null`-derived optionality — `$sdpa_fwd.dropout_probability`, `$sdpa_fwd.alibi_mask`,
  `$sdpa_fwd.left_bound` bind with no annotation (B.5). The one consequence is that a scalar's bind
  name is coupled to its field name: renaming the field renames the symbol, whereas `umd_name`
  insulates an operand from a field rename. If a scalar ever needs that insulation, the additive
  extension is an optional `umd_name` on an unflagged field, used purely as a bind-name override; the
  flag remains the operand/result discriminator.

### B.4 Generation pipeline

1. `flatc` compiles the graph schema and emits the binary reflection schema `graph.bfbs`, which
   transitively covers every attribute table and carries the declared `umd_*` attributes (custom
   attributes surface through reflection, not through the generated C++ headers).
2. A build-time generator loads `graph.bfbs` through the FlatBuffers reflection API, enumerates the
   `NodeAttributes` union members (opcode → table), reads each table's `umd_opcode`, and applies the
   B.3 rules to each table's fields. Because `graph.fbs` pulls the per-op tables in via `include`, the
   `.bfbs` (and thus the registry) must regenerate when **any** included schema changes, not only when
   the top-level schema does.
3. The generator emits the op-schema registry as generated C++ the provider compiles (a header-only
   registry emitted into the schema-owning SDK, so it needs no dependency on the provider): a table
   keyed by the `umd_opcode` shorthand, each entry also carrying its attribute-table name and the
   integer `NodeAttributes` value, and listing its input and output tensors (name, optionality, and
   the typed accessor for the UID field) and its scalar attributes (name, optionality, value kind, and
   typed accessor).

Reflection is used **only at build time**; the generated registry holds typed accessors, so the runtime
match path reads UID and attribute fields via `attributesAs<T>()`
(§ 5) with no per-match reflection cost.

### B.5 Generated registry and how the matcher uses it

The emitted entry for one opcode is, conceptually:

```jsonc
// generated; illustrative shape, not a wire format
"sdpa_fwd": {                          // key: the table's umd_opcode (fallback: table name)
  "attributes_type": "SdpaAttributes", // NodeAttributes union member; matched via Node::attributes_type()
  "operands": [
    {"name": "q", "uid": "&SdpaAttributes::q_tensor_uid",         "optional": false},
    {"name": "k", "uid": "&SdpaAttributes::k_tensor_uid",         "optional": false},
    {"name": "v", "uid": "&SdpaAttributes::v_tensor_uid",         "optional": false},
    {"name": "attn_mask", "uid": "&SdpaAttributes::attn_mask_tensor_uid", "optional": true}
  ],
  "results":  [{"name": "o", "uid": "&SdpaAttributes::o_tensor_uid", "optional": false}],
  "attributes": [
    {"name": "dropout_probability", "get": "&SdpaAttributes::dropout_probability", "optional": true},
    {"name": "alibi_mask",          "get": "&SdpaAttributes::alibi_mask",          "optional": false}
  ]
}
```

At compile (§ 7) the matcher resolves each pattern
name against this entry. At match time, for a node of that opcode it reads each name's UID via the typed
accessor, resolves the UID against the per-graph UID→producer/consumer index
(§ 5) to bind the tensor, and auto-binds the tensor's fields
and the node's scalar attributes into the five namespaces
(§ 6.1). An optional operand absent from the
graph is reported by `not_present` and is read only through a guarded reference or `value_or_default`.

### B.6 Lockstep and failure modes

- **Lockstep.** Adding or renaming an operand is one `.fbs` edit that carries its `umd_input_tensor` /
  `umd_name` with it; the next build regenerates the registry, so a UMD naming that name resolves and a
  UMD referencing a name that no longer exists fails compile
  ([RFC 0018 A.5](0018_UniversalMatchDescriptor.md#a5-compile-time-validation-normative)).
- **Unknown op or name at match compile.** The matcher fails closed: a pattern node whose opcode or
  name is absent from the registry is refused, never bound to a guessed field
  ([RFC 0018 § 16](0018_UniversalMatchDescriptor.md#16-risks)).
- **Generation is deterministic and diffable.** The generated registry is a build artifact; a schema
  change that alters bindings shows up as a registry diff, which is the review surface for a binding
  change.
