# RFC 0018: The UMD's Criteria: Applicability over the Engine's Binding

- Contributors: Brian Harrison

> Follow-up to [RFC 0017 (Universal Kernel Descriptors)](0017_UniversalKernelDescriptor.md),
> the "UMD + applicability" row of its follow-up series ([RFC 0017 §14.2](0017_UniversalKernelDescriptor.md#142-follow-up-rfcs)).
> This RFC designs the UMD — one criteria expression deciding whether a kernel applies — and the
> stage of matching that evaluates it. The engine's `nodes` pattern, the symbol table matching it
> publishes, and stage one of the matcher belong to the "UED + graph matching" row and are specified
> in [RFC 0020](0020_UniversalEngineDescriptor.md); this RFC reads that table and does not define it. The sibling formats (UDD,
> UHD, KDP) and subsystems (packaging, drop-in, adapters) are designed in their own follow-ups and
> are referenced, not redesigned, here.

## Table of Contents

1. [Overview](#1-overview)
2. [The Symbol Table Criteria Read](#2-the-symbol-table-criteria-read)
3. [Criteria Vocabulary](#3-criteria-vocabulary)
4. [The Shared Expression Language](#4-the-shared-expression-language)
5. [Layout and Stride-Order Criteria](#5-layout-and-stride-order-criteria)
6. [The Native-Matcher Escape Hatch](#6-the-native-matcher-escape-hatch)
7. [Composite Criteria](#7-composite-criteria)
8. [The Matcher: Compilation, Indexing, and Caching](#8-the-matcher-compilation-indexing-and-caching)
9. [Arbitration](#9-arbitration)
10. [Serialization and Versioning](#10-serialization-and-versioning)
11. [Security and Hostile Input](#11-security-and-hostile-input)
12. [Observability and Diagnostics](#12-observability-and-diagnostics)
13. [Testing and Performance](#13-testing-and-performance)
14. [Migration](#14-migration)
15. [Worked Example: SDPA Forward](#15-worked-example-sdpa-forward)
16. [Risks](#16-risks)
17. [Open Questions](#17-open-questions)
18. [References and Prior Art](#18-references-and-prior-art)
19. [Glossary](#19-glossary)
20. [Appendix A: Schema Reference](#appendix-a-schema-reference)

---

## 1. Overview

Deciding whether a kernel applies to an incoming problem graph is two questions, and the series
gives each its own descriptor. **Does this engine serve graphs of this shape, and what are the
pieces called?** is the **UED (Universal Engine Descriptor)**, whose `graph_match` binds every
tensor and attribute the graph supplies — a structural pattern over the op DAG, or the native
escape hatch standing in for one — specified in [RFC
0020](0020_UniversalEngineDescriptor.md). **Given those pieces, can this kernel take the problem?**
is the **UMD (Universal Match Descriptor)**, one JsonLogic boolean over the symbols the pattern
bound, and that is what this RFC specifies. Together they replace a hand-coded
`IPlanBuilder::isApplicable` ([RFC 0017 §2](0017_UniversalKernelDescriptor.md#2-the-descriptors)).

Matching is therefore two stages over one graph. The **engine's pattern** runs first: it resolves op
and tensor names against the op-schema registry, walks the graph, and publishes the bound symbol
table — every operand and result tensor with its dims and strides, and every matched node's scalar
attributes. It runs **once per engine per graph**, and a graph its pattern does not match declines
the engine outright, before any pack is consulted ([RFC 0020 §
7](0020_UniversalEngineDescriptor.md#7-pattern-matching-stage-one)). The **criteria** run second:
each UMD a pack lists evaluates its single boolean over that table, and a kernel applies only when
every matcher in its pack passes ([RFC 0017
§5](0017_UniversalKernelDescriptor.md#5-matching-the-ueds-pattern-and-the-umds-criteria)), so a
family of near-identical kernels shares a handful of criteria sets rather than carrying a bespoke
C++ check each.

**The split follows the shape of the calls hipDNN actually makes.** `isApplicable` arrives per
engine ([RFC 0017 §8](0017_UniversalKernelDescriptor.md#8-end-to-end-flow)). Had every matcher
carried its own pattern, an engine would re-walk one graph once per matcher of every pack naming it,
structurally matching the same nodes again and again before any of them could disagree. One pattern
per engine collapses that to a single structural pass, and the root-opcode index then keys engines
rather than matchers, so an engine whose pattern is not rooted at the graph's opcode is pruned
before a single criterion is read.

**It also gives the bound-symbol set one owner.** A UED names one heuristic and one metadata schema
([RFC 0017 §2](0017_UniversalKernelDescriptor.md#2-the-descriptors)), and that heuristic's
`features_signature` reads graph tokens — `$q.dims[2]`, `$sdpa_fwd.dropout_probability`. Those
symbols have to be bound by something the engine owns, or an engine-wide model is written against
names only some pack happens to supply. The UED publishes the bound-symbol set, and it is the single
source every consumer is checked against: a UMD's criteria, its pack's UDD formulas, and the
engine's own UHD `features_signature`. A reference none of them can resolve is rejected at load
rather than failing closed later on a live graph.

This document turns the criteria half of that frame into a concrete format and a concrete evaluation
stage. It specifies the criteria schema, the layout representation as a stride-rank array,
the native-matcher escape hatch, deterministic arbitration, and how criteria are compiled, memoized,
and cached over the binding the engine publishes. The static (compile-time) matcher is sketched as
options, not fully designed, in this iteration.

**The expression language itself is not specified here either.** Criteria are written in the
descriptor expression language, a deferred follow-up that will own its
grammar, type system, operator set, three-valued semantics, and bounded interpreter. What this
document supplies is the *criteria* written in it and the reader's contract on the environment they
evaluate over ([§2](#2-the-symbol-table-criteria-read)).

### 1.1 What This RFC Specifies Versus Defers

| Capability | This RFC (day-one) | Deferred |
|---|---|---|
| The reader's contract on the engine's binding: which namespace roots criteria may name and what each yields | Yes ([§2](#2-the-symbol-table-criteria-read)) | The `nodes` format: [RFC 0020 § 4.3](0020_UniversalEngineDescriptor.md#43-the-nodes-pattern-normative); the published field set: [RFC 0020 § 6.1](0020_UniversalEngineDescriptor.md#61-the-published-field-set-normative) |
| Criteria (UMD) as one JsonLogic expression: dtype (exact/set/relation), rank, dim relations, divisibility, stride order, packed, attribute, `virtual`, cross-tensor relation, optional operand, device property, `$kernel.*` pins | Yes ([§3](#3-criteria-vocabulary)) | None |
| The expression language criteria are written in: grammar, operators, type system, semantics, interpreter | None: the descriptor expression language follow-up owns it, with [RFC 0017 §5](0017_UniversalKernelDescriptor.md#5-matching-the-ueds-pattern-and-the-umds-criteria) the interim authority and [§4](#4-the-shared-expression-language) the recap | Operator additions, to that follow-up |
| Layout as a stride-rank array, with named aliases | Yes ([§5](#5-layout-and-stride-order-criteria)) | None |
| Escape hatch for checks needing C++, beside the descriptor rather than inside the expression language | Yes ([§6](#6-the-native-matcher-escape-hatch)) | None |
| Composite criteria: `(A AND B) OR C` as one criteria expression, via JsonLogic `and`/`or`/`!`/`if` | Yes ([§7](#7-composite-criteria)) | None |
| Stage two of matching: criteria evaluated per pack, memoized per `$kernel.*` projection, cached at applicability | Yes ([§8](#8-the-matcher-compilation-indexing-and-caching)) | None |
| Stage one: pattern compilation, the root-opcode index over engines, and the bind step | None: [RFC 0020 § 7](0020_UniversalEngineDescriptor.md#7-pattern-matching-stage-one) | None |
| Static (compile-time / AOT-lowered) matcher | None | Full design, including the interpreted-versus-lowered parity contract |
| General N-ary commutative matching, unbounded variable-length chains | None | JIT follow-up ([RFC 0017 §9.3](0017_UniversalKernelDescriptor.md#93-future-jit-and-normalized-providers)) |

---

## 2. The Symbol Table Criteria Read

A UMD's criteria are evaluated over a symbol table it does not produce. The table is the engine's:
the UED carries a `graph_match` whose declarative arm is a `nodes` block, a structural pattern over
the op DAG, and matching it binds every tensor, dim, stride, and scalar attribute the pattern
names. The engine is the **only** producer, so a criterion reads the table and never adds to it.
The pattern's format is [RFC 0020 §
4.3](0020_UniversalEngineDescriptor.md#43-the-nodes-pattern-normative) and the field set matching
it publishes is [RFC 0020 §
6](0020_UniversalEngineDescriptor.md#6-symbol-binding-what-the-pattern-publishes); what this
section fixes is the **reader's** side of that contract — the roots a criteria expression may name,
what each yields, and what the criteria may assume about when they are bound.

**Five namespace roots, three of them pattern-bound.**

| Root | Yields | Bound by |
|---|---|---|
| a pattern variable (`$q`) | the matched tensor and its fields — dims and strides positionally, plus derived facts like rank, dtype, stride order, and packedness | the engine's pattern |
| `$graph` | structural and graph-level facts of the matched graph, such as its node count and its override-shape opt-in | the engine's pattern |
| a node `id` (`$sdpa_fwd`) | that matched node's scalar attributes, named as the schema declares them | the engine's pattern |
| `$kernel` | the values the candidate UKD supplies for the fields its KMD declares | the UKD, per candidate |
| `$device` | device properties read from the `Handle`, such as `$device.lds_size` | the runtime |

The exact fields under each root, their types, and the reserved-root rule are [RFC 0020 §
6.1](0020_UniversalEngineDescriptor.md#61-the-published-field-set-normative).
[Appendix A.2](#a2-variable-references-and-resolution) fixes the rules a reader
needs on top of it: reference syntax, and what a read yields when the thing read is absent.

**A `$` marks a reference.** Any JSON string beginning with `$` is a reference into that table;
every other JSON scalar is a literal — numbers, enum values (`"BFLOAT16"`), and layout aliases. The
node id in an attribute reference is bare and the reference carries the `$`
(`$sdpa_fwd.dropout_probability`).

**Three properties of the binding shape everything below.**

- **The pattern is engine-wide and singular**, one per UED, so criteria on one engine all read the
  same table, and the topology is checked once, structurally, before any criterion runs
  ([§8](#8-the-matcher-compilation-indexing-and-caching)). A matcher can only constrain what the
  pattern already bound.
- **`$kernel.*` is not pattern-bound.** Those values come from the candidate UKD, which is why a
  matcher reading them is re-evaluated per kernel rather than once per graph
  ([§8](#8-the-matcher-compilation-indexing-and-caching)), and why the matcher publishes the set of
  `$kernel.*` fields it reads so the loader can check them against the engine's KMD
  ([Appendix A.5](#a5-compile-time-validation-normative)).
- **A reference that the engine's pattern does not publish is a load error, not a runtime decline.**
  A UMD is validated against the engine of each pack that lists it, so an unresolvable reference is
  caught at pair-validation, naming both descriptors and the pack that paired them
  ([Appendix A.5](#a5-compile-time-validation-normative)).

Quantities like head size, batch, and head count are **not** attributes; they are specific tensor
dims (for SDPA, `q.dims[3]`, `q.dims[0]`, `q.dims[1]`), reached positionally as `$q.dims[i]` and
never as an attribute read ([RFC 0020 §
5](0020_UniversalEngineDescriptor.md#5-the-graph-model-the-pattern-matches)). Layout is likewise not
stored on a tensor: it is derived from the stride order and compared as a stride-rank array
([§5](#5-layout-and-stride-order-criteria)).

---

## 3. Criteria Vocabulary

The `criteria` field is a **single JsonLogic boolean expression** evaluated over the symbol table the
engine's pattern published ([§2](#2-the-symbol-table-criteria-read), [§4](#4-the-shared-expression-language)). It is
normally an `and` of the individual tests, and reaches for `or` / `!` / `if` wherever a real
disjunction is needed ([§7](#7-composite-criteria)). The table below is not a set of criterion
*kinds* (there are none); it is the set of hand-written checks and the
JsonLogic sub-expression that expresses each. The residue — the handful of checks that need real C++
— is not an operator at all: it is a **native matcher** the pack lists beside the descriptor
([§6](#6-the-native-matcher-escape-hatch)), so the expression language itself stays closed.

| Hand-written check | JsonLogic criterion | Lowers from |
|---|---|---|
| **Opcode** | in the engine's pattern, not a criterion ([RFC 0020 § 4.3](0020_UniversalEngineDescriptor.md#43-the-nodes-pattern-normative)) | node attribute-type gate |
| **Dtype (exact / set)** | `{"==": ["$q.dtype", "BFLOAT16"]}` / `{"in": ["$q.dtype", ["BFLOAT16", "FP8_E4M3"]]}`; pin against a kernel with `{"==": ["$q.dtype", "$kernel.dtype"]}` when the pack ships one binary per dtype (below) | `validateDataTypeIsSupported`, `validateFixedDataType` |
| **Dtype (relation)** | `{"==": ["$k.dtype", "$q.dtype"]}` | `validateConsistentDataTypes`, `q == k == v` |
| **Rank** | `{"==": ["$q.rank", 4]}` | `validateDimensionCount`, rank == 4 |
| **Dim (value / relation)** | `{"==": ["$q.dims[3]", 128]}`; relate with `{"==": ["$k.dims[3]", "$q.dims[3]"]}`; pin against a kernel with `{"==": ["$q.dims[3]", "$kernel.head_size"]}` where the value is one the kernel bakes (below) | dim reads and cross-tensor dim relations |
| **Divisibility** | `{"divisible": [{"*": ["$y.dims[0]", "$y.dims[2]", "$y.dims[3]"]}, "$kernel.MPerBlock"]}` | tile-fit / GEMM-dim gates |
| **Layout** | `{"==": ["$q.stride_order", [3, 2, 1, 0]]}` ([§5](#5-layout-and-stride-order-criteria)) | `validateSupportedLayout` |
| **Packing** | `"$q.packed"` (a bound boolean) | `validatePackedTensors` |
| **Cross-tensor layout** | `{"==": ["$x.stride_order", "$y.stride_order"]}` (per pair) | `validateConsistentLayouts` |
| **Attribute (value)** | `{"==": ["$sdpa_fwd.causal_mask", false]}`; absent-or `{"or": [{"not_present": ["$sdpa_fwd.dropout_probability"]}, {"==": ["$sdpa_fwd.dropout_probability", 0.0]}]}` | per-attr value gates |
| **Attribute (one_of)** | `{"in": ["$sdpa_fwd.diagonal_alignment", ["TOP_LEFT", "BOTTOM_RIGHT"]]}` | enum-attribute set gates |
| **Optional operand present/absent** | the operand is declared optional in the engine's pattern ([RFC 0020 § 4.3](0020_UniversalEngineDescriptor.md#43-the-nodes-pattern-normative)); `{"not_present": ["$attn_mask"]}` (absent) / `{"present": ["$bias"]}` (present); one call takes a list, so a pack declines every optional operand it cannot serve at once | `attn_mask_tensor_uid()` absent gate |
| **Graph structure (exact / fusion)** | `{"==": ["$graph.node_count", 3]}`, and each intermediate `"$conv_out.virtual"` | node-count gate, fusion legality |
| **Cross-tensor / arithmetic** | `{"==": ["$q.dims[1]", "$k.dims[2]"]}`, `{"<": ["$q.dims[3]", 129]}`, `{"==": [{"%": ["$q.dims[1]", "$k.dims[1]"]}, 0]}` | arithmetic and comparison gates |
| **Device property** | `{"<=": ["$kernel.lds_per_block", "$device.lds_size"]}` (arch is a pack property, not a criterion) | LDS/occupancy budgets; `getDeviceString` arch → pack `arch` |
| **Needs real C++** | not a criterion; a native matcher listed beside the descriptor ([§6](#6-the-native-matcher-escape-hatch)) | |

Architecture gates *applicability* at pack selection via the KDP `arch` property and the per-arch
`kpack` manifest ([RFC 0017 §5](0017_UniversalKernelDescriptor.md#5-matching-the-ueds-pattern-and-the-umds-criteria),
[§11](0017_UniversalKernelDescriptor.md#12-packaging-and-delivery)), not as a match-time criterion for
AOT; other device properties (`$device.lds_size`, `$device.warp_size`) are read directly in criteria.

**Exactness runs to the kernel, not just the graph.** `$graph.node_count` and the engine's pattern
pin the shape of the *graph*, but say nothing about whether a given candidate kernel can serve it. A
prebuilt kernel bakes quantities into its binary — a dtype, a head size, sometimes a sequence length —
and a graph that clears a pack's graph-level gates may still disagree with what one kernel baked.
**Every quantity a kernel bakes MUST therefore be a KMD field, and the pack's matcher MUST pin it
against the graph with a `$kernel.*` criterion.** Those are the clauses re-evaluated per candidate
([§8](#8-the-matcher-compilation-indexing-and-caching)), which is what turns one matcher plus a
kernel vector into a per-kernel applicability test. It is also why, although a KDP may list no
matchers at all and rest on the engine's pattern alone, a prebuilt pack in practice always lists one.

Getting this wrong fails silently rather than loudly: a matcher gating dtype only as
`{"in": ["$q.dtype", ["FLOAT16", "BFLOAT16"]]}` accepts an fp16 graph and may hand it to a bf16
binary, which returns wrong numbers instead of an error. A field missing from the KMD also cannot be
pinned, so two kernels differing only in an unmodelled baked constant collide on the catalog key.
One case is mechanical, so the loader performs it: a UKD whose source declares a baked constant with
no corresponding KMD field is a load error. That check is a KDP/KMD-loader responsibility and is
specified by [RFC 0017 §5](0017_UniversalKernelDescriptor.md#5-matching-the-ueds-pattern-and-the-umds-criteria); the UMD's part
is to publish the `$kernel.*` fields it reads ([§2](#2-the-symbol-table-criteria-read))
so the loader can perform it.

**That check narrows the gap; it does not close it.** It catches a baked constant with no field to
pin it against, and a catalog-key collision is loud when it happens. Neither covers the case where
the field exists, the matcher pins it, and the pack still ships a set of concrete instances whose
criteria leave a graph unserved or two kernels overlapping — that depends on which instances the
kernel pack actually contains, which no per-descriptor check sees. Authoring a matcher against the
set a pack ships remains the author's responsibility.

**The engine's pattern is the topology.** One pattern per engine means the topology is checked once,
structurally, before any criterion runs, and a matcher can only constrain what the pattern already
bound.

---

## 4. The Shared Expression Language

A UMD's `criteria` field is a single `Bool`-rooted expression in the **descriptor expression
language**. That language is a deferred follow-up, reserved as **RFC 0019**, which is why the series
numbering steps from this document to
[RFC 0020](0020_UniversalEngineDescriptor.md) with a gap. It is not restated here; until it is
written,
[RFC 0017 §5](0017_UniversalKernelDescriptor.md#5-matching-the-ueds-pattern-and-the-umds-criteria)
is the interim authority for its operator vocabulary, and the semantics this document leans on are
stated locally in [Appendix A.2](#a2-variable-references-and-resolution) and
[Appendix A.3](#a3-the-expression-language). What
this document supplies is the binding environment those expressions evaluate over
([§2](#2-the-symbol-table-criteria-read)).

**The `$`-variable rule.** Any JSON string beginning with `$` is a reference into the bound symbol
table of [§2](#2-the-symbol-table-criteria-read); every other JSON scalar is a literal.

**Criteria are boolean-rooted; the UDD's dispatch formulas are value-rooted.** Both are the same
language over the same symbol table — a criterion decides applicability, a formula yields a grid,
block, or workspace number
([RFC 0017 §6](0017_UniversalKernelDescriptor.md#6-dispatch-and-workspace)) — and the engine's UHD
writes its `features_signature` entries over that same table
([RFC 0017 §4](0017_UniversalKernelDescriptor.md#4-descriptor-formats)), so one parser,
validator, and interpreter serve all three subsystems.

**The operator set is closed.** A descriptor cannot introduce an operation, so a check that needs
real C++ is a native matcher listed beside the descriptor, never a nested extension point
([§6](#6-the-native-matcher-escape-hatch)).

```jsonc
// criteria (boolean): sub-expressions of the single top-level criteria expression
{"==": ["$q.rank", 4]}                                      // rank pin
{"==": ["$q.dims[3]", 128]}                                 // head dimension (last axis)
{"==": ["$k.dims[3]", "$q.dims[3]"]}                        // cross-tensor dim relation
{"in": ["$q.dtype", ["BFLOAT16", "FP8_E4M3"]]}              // dtype set
{"==": ["$q.stride_order", [3, 2, 1, 0]]}, "$q.packed"      // layout + packed
{"or": [{"not_present": ["$attn_mask"]},
        {"==": ["$attn_mask.dtype", "$q.dtype"]}]}          // composition (§8)
```

---

## 5. Layout and Stride-Order Criteria

hipDNN tensors store no layout enum; layout is implied by stride order
([RFC 0020 § 5](0020_UniversalEngineDescriptor.md#5-the-graph-model-the-pattern-matches)). The UMD represents layout the way
[RFC 0017 §5](0017_UniversalKernelDescriptor.md#5-matching-the-ueds-pattern-and-the-umds-criteria) writes it, which is the
encoding hipDNN already computes: an array indexed by **logical dimension**, entry `d` giving the
**stride rank** of logical dimension `d`, **lower meaning faster-varying**. Entry value `0` marks the
unit-stride dimension, so the array reads as the packing order it describes: `[3, 0, 2, 1]` over an
`(n, c, h, w)` logical dim order puts C fastest, then W, then H, then N — NHWC.

```jsonc
{"==": ["$q.stride_order", [3, 2, 1, 0]]}   // descending-stride packed (BHSD, rank-4)
{"==": ["$x.stride_order", [3, 0, 2, 1]]}   // NHWC over an NCHW logical dim order
```

- The array is a permutation of `0..rank-1`. Entry `d` is the stride rank of logical dimension `d`,
  counting up from the fastest-varying, so `[3,2,1,0]` is descending-stride packed and `[3,0,2,1]`
  gives the channel dim rank `0`, hence fastest-varying (NHWC). It is indexed the same way as
  `dims`, `strides`, and the `axis` of `args_signature` — all four select a logical dimension —
  so a tensor's layout is read off the same axis numbering the rest of a descriptor uses.
- This is the encoding `extractStrideOrder` returns
  (`projects/hipdnn/data_sdk/include/hipdnn_data_sdk/utilities/ShapeUtilities.hpp:146`, called from
  `ApplicabilityChecks.cpp:22`), so the binding layer publishes `$q.stride_order` as the data SDK
  already computes it. One spelling serves descriptors and the shipped code alike.
- **Named aliases** are provided for the common cases and expand to the array literal at compile time,
  so `{"==": ["$x.stride_order", "nhwc"]}` compiles to a comparison against `[3, 0, 2, 1]`
  (A.5). The array remains the single canonical form. The four convolution aliases are exactly the
  layouts `validateSupportedLayout` accepts today — NCHW/NHWC at rank 4, NCDHW/NDHWC at rank 5
  (`ApplicabilityChecks.cpp:76`); `bhsd` is an addition for the attention families, which that
  oracle never covered.
- **Cross-tensor consistency** is a JsonLogic equality between stride orders,
  `{"==": ["$x.stride_order", "$y.stride_order"]}` (one per pair, joined by the top-level `and`),
  lowering `validateConsistentLayouts`; layout-agnostic tensors (rank-1 scalars, pass-by-value) are
  skipped as they are today.
- **Packing** is the separate bound boolean `$q.packed` (written `"$q.packed"`), since a supported
  stride order does not imply the tensor is gap-free; it lowers `validatePackedTensors`.
- `$q.stride_order` is an ordinary bound value ([§2](#2-the-symbol-table-criteria-read)),
  so a `stride_order == [3,2,1,0]` gate is expressible directly.

---

## 6. The Native-Matcher Escape Hatch

Some checks cannot be stated with the built-in operators: they need real C++. **The expression
language has no extension point for them.** There is no custom operation, no namespaced operator key,
and no predicate registry the criteria tree resolves against; an operation key the vocabulary does
not list ([Appendix A.3](#a3-the-expression-language)) is simply unrecognized
and refused at compile time ([Appendix A.5](#a5-compile-time-validation-normative)).

Instead, such a check is a **native criterion**: an ordinary `GraphCriterionFn` registered in the
provider's `NativeRegistry` and named by a UMD's `match_symbol`
([RFC 0017 §5](0017_UniversalKernelDescriptor.md#5-matching-the-ueds-pattern-and-the-umds-criteria)). It stays a
**pack-level** listing beside the descriptor-backed matcher, because what it expresses is a gate on
the kernel family rather than a statement about the graph shape the engine serves. The ingestor
conjoins their verdicts: a pack's graph-scoped matchers all run and all must pass, so "this UMD's
criteria **and** this C++ predicate" is expressed by listing two matcher ids, not by nesting one
inside the other. The schema enforces that reading: a UMD carries a criteria expression or a native
symbol, never both ([Appendix A.1](#a1-the-umd-descriptor-object)), so a conjunction is always
visible in the pack's matcher list rather than buried inside one descriptor — and a UMD file stays
either pure data or a name, never a mixture of the two.

**This is the pack-scoped half of a two-part hatch.** The engine-scoped half is the UED's
`graph_match.native` ([RFC 0020 § 4.5](0020_UniversalEngineDescriptor.md#45-the-native-arm-normative)),
which *produces* the binding at stage one by declining the graph or returning the bound tokens. A
native criterion *reads* that binding and returns a verdict; it cannot add to it. The two are
distinct registries and distinct signatures, and conflating them is the mistake to avoid: one
decides what an engine serves, the other narrows which of its packs apply.

```jsonc
// pack.matcherIds: [ <the UMD above>, <"hipdnn.sdpa.mask_self_consistent">, ... ]
```

**Why the hatch sits beside the expression language rather than inside it.** An escape hatch that
nests inside `criteria` has to be resolved by the compiler, which means the criteria language grows a
registry, a signature table, and a per-argument type contract — a second extension mechanism running
parallel to the one RFC 0017 already defines, differing only in grain. One hatch, at the matcher
level, keeps the expression language closed: every operator it publishes
([Appendix A.3](#a3-the-expression-language)) is
total, statically typed against the op-schema registry, and means the same thing in every provider
that ships a UMD. A descriptor is then fully interpretable from the schema alone, which is what keeps
the drop-in path and any future static lowering tractable.

**What this costs, and what the split already fixed.** A nested custom operation would have received
*bound variables* (`["$q", "$k", "$v", "$o"]`). A `GraphCriterionFn` now receives
`(const MatchContext&, const BoundTokens&)`: the raw graph and device id, **plus** the binding the
engine's `graph_match` published before any pack's checks ran
([§2](#2-the-symbol-table-criteria-read)). So a criterion reads a resolved graph attribute instead of
rewalking the graph, and the structural search the engine already performed is not repeated in
hand-written code that could drift from the registry-driven binding.

What remains is a *representational* limit, not a plumbing one. `BoundTokens` is
`string -> MetadataValue`, a scalar variant: it carries a dim, a uid, or a dtype name, but not a
tensor object, so a criterion wanting whole-tensor access still reaches through `MatchContext` by
uid. Widening that variant is an additive change to one type, and the operand order every native
stage shares ([RFC 0020 § 6.1](0020_UniversalEngineDescriptor.md#61-the-published-field-set-normative))
is what keeps the declarative and native spellings in step when it happens.

**The registry a provider ships is part of its published contract**, unchanged from
[RFC 0017 §5](0017_UniversalKernelDescriptor.md#5-matching-the-ueds-pattern-and-the-umds-criteria): a pack naming a `match_symbol`
the running provider does not ship fails to resolve, and fails closed. What changes is that this is
the *only* place a name resolves to C++, so the drop-in story is simple — **a UMD file is pure data
and always loads identically; a pack that needs C++ is not a drop-in.**

Together with the UDD's custom-plan hatch
([RFC 0017 §6](0017_UniversalKernelDescriptor.md#6-dispatch-and-workspace)), these form the graded
ladder: fully declarative constraints, then a native matcher beside the descriptor for one gate that
needs C++, then a full provider.

---

## 7. Composite Criteria

Composition is native to the expression language, so `(A AND B) OR C` is stated directly in the one `criteria`
tree with no extra mechanism
([§4](#4-the-shared-expression-language)):

```jsonc
"criteria":
  {"or": [
    {"and": [{"==": ["$q.dims[3]", 64]},          // head dimension (last axis)
             {"==": ["$q.dims[1]", "$k.dims[1]"]}]},  // (A AND B): equal head counts, so not GQA
    {"==": ["$q.dims[3]", 128]}                   // OR C
  ]}
```

A UMD whose tests all conjoin simply makes the top-level expression an `and`, the common case.
General N-ary commutative matching and unbounded chains remain deferred to the JIT follow-up, as in
RFC 0017 §5.

---

## 8. The Matcher: Compilation, Indexing, and Caching

Both halves are authored as text and **compiled once** into in-memory structures: the pattern's
compilation, and the root-opcode index that prunes engines before any pattern runs, are the UED's
([RFC 0020 § 7](0020_UniversalEngineDescriptor.md#7-pattern-matching-stage-one)). Compiling a UMD's
criteria expands layout aliases and parses the expression to an AST, on demand: nothing is parsed
until a graph needs it, and the parsed result is cached and reused ([RFC 0017
§3](0017_UniversalKernelDescriptor.md#3-how-it-works)). Neither compiled form is complete on its
own: a UMD is **validated against the engine of each pack that lists it**, checking that every
`$`-reference resolves in that engine's published symbols and every `$kernel.*` exists in its KMD
([Appendix A.5](#a5-compile-time-validation-normative)); a failure names both descriptors and the
pack that paired them. That pair-validation is cached on
`(matcher, engine)`, so a matcher shared by several packs on one engine is checked once. The
compiled forms, not the text, are what run against live graphs.

**Stage one has already run.** By the time any criterion is evaluated, the engine's pattern has
matched the graph and published the bound symbol table ([RFC 0020 §
7](0020_UniversalEngineDescriptor.md#7-pattern-matching-stage-one)); a graph the pattern does not
match declines the engine outright, with no pack consulted, no UMD loaded, and no criteria
evaluated. That cost is paid once per engine per graph however many packs name the engine, which is
the structural saving the split buys, and it is why this section specifies only what stage two adds.

**Stage two: constrain, per pack and per kernel.** A KDP lists a set of matcher IDs and a kernel
applies only when all pass
([RFC 0017 §5](0017_UniversalKernelDescriptor.md#5-matching-the-ueds-pattern-and-the-umds-criteria)),
so matchers are the unit of sharing and of evaluation within an engine. A matcher reading only bound
graph fields (Tensor / Graph / Attributes / Device, [§2](#2-the-symbol-table-criteria-read)) declares
`scope: "graph"` and
runs **once per graph**; on failure it prunes every pack that lists it, so the most-shared checks
(dtype, layout, rank) evaluated first shrink the candidate set fast. A matcher that also reads
`$kernel.*` declares `scope: "kernel"` and is the **same** matcher re-evaluated **once per distinct
value of the `$kernel.*` fields
it reads**, memoized on those, pruning per kernel rather than per pack. The projection is what makes
this pay: a kernel's full metadata tuple is unique by construction, so memoizing on the whole tuple
would save nothing, while a matcher reading one field
collapses an engine's catalog to that field's handful of distinct values. The compiler already
computes which `$kernel.*` fields a matcher reads ([§2](#2-the-symbol-table-criteria-read)),
so the memoization key costs nothing extra, and the same read set is what
[Appendix A.5](#a5-compile-time-validation-normative) checks the declared `scope` against, in both
directions: the two can never disagree at match time because a descriptor where they disagree does
not compile. Results are cached across queries.

**Short-circuit evaluation.** The matcher relies on written-order short-circuit evaluation, a rule
this document states locally pending the language follow-up, so a non-match is
rejected as early as the author's structure allows; the compiler may additionally hoist a cheap,
highly selective sub-expression (a scalar attribute or dtype read) ahead of an expensive one (a wide
cross-tensor relation), which changes when a decision is reached and never what it is.

**Matching runs during applicability, and the provider owns the cache.** The order and the cache are
specified by [RFC 0017 §8](0017_UniversalKernelDescriptor.md#8-end-to-end-flow), which this RFC
follows rather than restates: both stages happen inside `IEngine::isApplicable`, and their two
products — the **catalog** (the kernels whose full matcher set passed) and the **bound token state**
(every `$`-prefixed value the pattern and the criteria resolved) — are cached together on the
provider's shared container, keyed on the engine, graph, and device that describe the problem, plus
the descriptor-inventory generation. The key already carries the engine, which is what makes a
per-engine binding the natural thing to cache under it. Later phases read that cache instead of
re-matching, so the
`isApplicable` / `getMaxWorkspaceSize` / `buildPlan` sequence that re-runs the loop today
(`AsmSdpaEngine.cpp:67,87`) matches a graph once. The compiled pattern and criteria are built once and
shared across every graph; only the binding result is per-problem.

**Accepting is a promise.** Because the catalog is settled during applicability, a non-empty catalog
commits the engine to producing a launchable kernel: a later failure surfaces as a failed plan build,
not a fallback to another engine. This is RFC 0017 §8.6's base-path invariant, and it is what makes
match semantics load-bearing — a pattern or criteria set that accepts a graph its kernel cannot serve
turns a decline into a user-visible error rather than a retry. It is the reason every quantity a
kernel bakes must be pinned by a `$kernel.*` criterion ([§3](#3-criteria-vocabulary)).

**Device properties are constant per stream.** A `$device.<field>` sub-expression (for example
`$device.lds_size`) is evaluated once per graph, since device properties do not vary across a stream.
Architecture is not a match-time criterion at all for AOT: it is a pack property gated at selection
([RFC 0017 §5](0017_UniversalKernelDescriptor.md#5-matching-the-ueds-pattern-and-the-umds-criteria)).

![Compile-once pipeline: text UMD to constraint IR to a root-opcode-indexed matcher, with an applicability-time bind cache](../images/umd_matcher_pipeline.svg)

---

## 9. Arbitration

Today the first applicable plan builder wins, which is a documented latent bug when more than one
matches (`HipMlopsEngine.cpp:34`). Descriptor-driven matching makes overlap explicit and resolves it
deterministically, reusing the rule from RFC 0017 §5:

1. When several UKDs match a graph, the engine's **heuristic** (the UHD) ranks them and the
   top-scored kernel wins.
2. Ties break by explicit **`priority`** on the UKD.
3. Remaining ties break by the descriptor's stable **`id`**, compared as raw bytes, and the conflict
   is logged to the warning log so an unintended overlap is visible. That byte order carries no
   meaning; it is chosen for being stable across runs, load orders, and machines, not because a lower
   id is better.

Arbitration is a property of the generic engine over the set of matching UKDs; a UMD shared by several
UKDs contributes each of them as a candidate. This closes the mutual-exclusion-by-construction
requirement that the current engines depend on: overlap is allowed and resolved, not a correctness
hazard. Overlap *between* engines — two engines whose patterns both accept one graph — is not
arbitrated here at all: that is ordinary engine selection, which hipDNN owns and the caller controls
([RFC 0017 §2](0017_UniversalKernelDescriptor.md#2-the-descriptors)).

---

## 10. Serialization and Versioning

- **Authoring form.** Human-readable, diffable JSONC (the examples here): the JsonLogic criteria
  expression of [§4](#4-the-shared-expression-language) with the `$`-variable convention.
- **Compiled form.** The compact binary the matcher runs ([§8](#8-the-matcher-compilation-indexing-and-caching)),
  whose concrete bytes are defined with the KDP/packaging follow-up
  ([RFC 0017 §14.2](0017_UniversalKernelDescriptor.md#142-follow-up-rfcs)); the schema those bytes
  encode is specified in [Appendix A](#appendix-a-schema-reference).
- **Type and identity.** Every UMD carries a stable `id` (a UUID) and a mandatory `name` for
  diagnostics; the pattern travels with the UED and is versioned with it. The descriptor **type**
  is carried by the `.umd.json` filename, not by an in-band `schema` member: the name already
  states the fact, and a file whose name and body disagree has no correct reading, so there is
  nothing to reconcile. A descriptor whose format version is newer than the runtime understands is
  refused with a clear error, never silently reinterpreted, matching
  [RFC 0017 §4](0017_UniversalKernelDescriptor.md#4-descriptor-formats).
- **`version` is a ceiling.** The format version of the descriptor itself: a differing `major`, or a
  `minor` newer than the runtime's, is
  refused, because the descriptor carries features that runtime cannot understand. An older minor
  within the same major always loads — a file stamped `1.0` loads on a `1.1` runtime — so an author
  stamps the lowest version their descriptor needs and it stays loadable on the oldest runtime that
  can serve it. This is RFC 0017 §4's rule for every format, applied here.
- **The graph-schema floor is the engine's alone; a UMD carries no `sdk_version`.** The hipDNN graph
  schema version a descriptor was authored against is declared once, on the UED
  ([RFC 0020 § 4.2](0020_UniversalEngineDescriptor.md#42-normative-schema)), and gates the
  whole engine: a graph reports the schema version its own contents require, and an engine declaring
  less declines it before binding, taking every pack that names it. A matcher inherits that floor
  from the engine of each pack that lists it — the same pairing that already validates its
  `$`-references ([§8](#8-the-matcher-compilation-indexing-and-caching)) — so it never carries a
  floor of its own to disagree with. One gate, at the level that owns the symbol table the criteria
  read.
- **What the single floor costs, and why it is the right trade.** An earlier draft gave the UMD its
  own floor, on the argument that binding is registry-driven
  ([RFC 0020 Appendix B](0020_UniversalEngineDescriptor.md#appendix-b-op-schema-registry-generation)):
  a newly added attribute is bound whether or not the pattern was touched, so what actually goes
  stale is a criteria set that never gates it, and a per-matcher floor could skip exactly that
  matcher while other packs on the same engine carried on. The finer grain is real and it is given
  up here. Two floors are two places to state the same fact and therefore a place for them to
  contradict each other — a matcher above its engine's floor is dead weight, one below it decides by
  a rule the engine never stated — and the partial-skip semantics leave a graph's candidate set
  depending on which matchers were quietly dropped. Raising an engine to a newer schema version is
  instead a review point for **every** matcher on it: the engine's author confirms the criteria gate
  whatever the revision added, because nothing narrower will decline them. That is a coarser and
  louder unit of change, and the engine is already the unit a matcher is validated against.
- **The graph's floor is an existing mechanism, not a new one.** hipDNN already computes the minimum
  engine-plugin API version a graph requires from the optional features it uses and stamps it into
  the serialized graph (`min_required_engine_api_version`); override shapes
  ([RFC 0008](0008_OverridableTensorShapesDesign.md)) raise it to `1.1` and runtime pass-by-value
  tensors ([RFC 0016](0016_RuntimePassByValueTensors.md)) to `1.2`. The engine reads that field
  rather than deriving a second floor of its own. A graph carrying no stamp reads as the `1.0`
  baseline.
- **Additive evolution.** New layout aliases and new bound fields are additive
  within `v1` where they do not change the meaning of an existing descriptor; anything that would
  reinterpret existing fields bumps the version. The expression language versions on its own axis,
  in its own follow-up.
- **Identity.** A UMD `id` is a **UUID**: globally unique with no central allocator, so descriptors
  authored independently — including third-party drop-in files — do not collide by construction.
  References are typed by field (a KDP's `matchers` versus `engine`), so a matcher id and an engine id
  are never confused. A duplicate `id` seen on the drop-in path is logged and ignored rather than
  taking down the provider ([RFC 0017 §16](0017_UniversalKernelDescriptor.md#16-risks)).
- **A UMD is versioned alone but validated in context.** Its `version` is its own — the only version
  it carries — but the check that its `$`-references resolve, and the graph-schema floor it runs
  under, are both the engine's, taken from each pack that lists it
  ([§8](#8-the-matcher-compilation-indexing-and-caching)), so a UED pattern edit that drops or
  renames a bound variable invalidates every matcher written against it. That is a load-time error
  naming both descriptors and the pack that paired them, not a silent behavior change
  ([§16](#16-risks)).

---

## 11. Security and Hostile Input

On the drop-in path the loader, the matcher, and the expression interpreter parse input that may be
untrusted or simply malformed, so they must be bounded and fail closed rather than crash
([RFC 0017 §16](0017_UniversalKernelDescriptor.md#16-risks)).

- **Bounded parsing.** Descriptor size, pattern node count, and pattern edge count are capped;
  exceeding a cap quarantines the descriptor, it does not abort the provider. The caps split with the
  descriptors: node and edge counts bound a UED, descriptor size bounds either.
- **A bounded, fail-closed interpreter.** Expression depth and step count, checked arithmetic, and
  declining on an unknown symbol, an unrecognized operator key, an out-of-range axis, or a type error
  are the language's contract, stated here pending its follow-up: evaluation is bounded and total,
  so a criteria expression terminates and cannot trap.
- **Quarantine, not cascade.** A bad descriptor is quarantined on load with a diagnostic; the rest load
  ([RFC 0017 §12](0017_UniversalKernelDescriptor.md#12-packaging-and-delivery)).
- **Fuzzing.** A seed corpus of patterns, criteria, and graphs plus a fuzzer over the loader and
  matcher run under the existing ASAN build ([§13](#13-testing-and-performance)), backing the
  fail-closed requirement.

---

## 12. Observability and Diagnostics

Because matching is data-driven, it is inspectable. The provider surfaces:

- **A why-not trace, in two stages.** Matching declines in one of two places and the trace says
  which. An engine whose **pattern** did not match reports the node or edge that failed to resolve,
  and that one line explains why every pack on that engine is absent from the answer. A **criterion**
  that evaluated false reports the sub-expression and the concrete values compared, naming the
  matcher and the pack, so an author sees exactly which test declined and which kernels it took with
  it. Conflating the two would be the common confusion in a two-stage matcher: an author whose
  criteria are never reached needs to be told the engine never claimed the graph. The
  per-sub-expression content of that second trace is the language's evaluation-trace contract,
  owed by its follow-up and
  rendered on the matcher's diagnostic surface.
- **A binding view.** For a successful pattern match, the full bound symbol table (tensors, dims,
  strides, attributes) as the criteria and the UDD will see it — the bound token state of
  [RFC 0017 §8](0017_UniversalKernelDescriptor.md#8-end-to-end-flow). It is the engine's, so one view
  serves every pack naming that engine.
- **An arbitration trace.** Which UKDs matched, how the heuristic scored them, and where a tie fell to
  `priority` or stable `id` ([§9](#9-arbitration)).
- **Load diagnostics.** Which patterns and criteria compiled, which were quarantined and why, which
  `(matcher, engine)` pairs failed symbol resolution and on which reference, and unresolved native
  predicates by name.

These reuse the diagnostic surface
[RFC 0017 §10](0017_UniversalKernelDescriptor.md#10-observability-and-diagnostics) defines rather than
adding a UMD-specific one. Authoring and validation tooling around the format is a separate,
first-class deliverable specified in
[RFC 0017 §11](0017_UniversalKernelDescriptor.md#11-tooling), where agentic authoring — agent-driven
skills that build and check descriptors from intent — is a committed first step. These descriptors are
a good fit for it: the schema of [Appendix A](#appendix-a-schema-reference) and the compile-time
checks of A.5 are exactly what such a tool validates against.

A matcher is also affected by the runtime opt-outs RFC 0017 §10 defines
(`HIPDNN_DISABLE_ENGINES`, `HIPDNN_DISABLE_KDPS`, `HIPDNN_DISABLE_UKDS`). Disabling a kernel carries a
risk specific to shared matchers: a matcher written around the kernel set it was meant to cover may no
longer be correct once one of those kernels is excluded, leaving the engine over-claiming
applicability for cases it no longer serves. The option is provided with that risk stated.

---

## 13. Testing and Performance

The split introduces no new testing strategy; it slots into hipDNN's existing tiers (`docs/Testing.md`,
`docs/testing/TestingStrategy.md`) as RFC 0017 §14.1 requires. A descriptor-backed kernel runs through
the
generic engine as an ordinary engine and produces the same graphs everything else consumes, so the
plugin-agnostic integration harness ([RFC 0006](0006_PluginAgnosticIntegrationTests.md)) validates it
against the CPU reference ([RFC 0001](0001_CpuGraphExecutorDesign.md)) with the golden-reference
tolerance chain ([RFC 0011](0011_GoldenReferenceValidation.md)).

Matcher-specific coverage:

- **Match-equivalence against hand-written `isApplicable`.** For each converted engine, a test drives a
  battery of graphs (accepting and rejecting) through both the hand-written builder and the
  pattern-plus-criteria pair and
  asserts identical accept/reject decisions and identical bound values. The SDPA-forward builder
  ([§15](#15-worked-example-sdpa-forward)) is the first target.
- **Symbol-resolution rejection.** A UMD referencing a symbol a given UED does not publish is rejected
  at pair-validation, naming the reference, both descriptors, and the pack that paired them, and a
  UED pattern edit that removes a
  bound variable is caught the same way
  ([§8](#8-the-matcher-compilation-indexing-and-caching)). This is the check the split exists to make
  possible, so it is tested directly rather than only through the descriptors that happen to be valid.
- **Static/runtime parity.** Should a matcher ever be lowered to a static form, the interpreted
  matcher is the oracle: the same UMD and graph must decide identically on both, across the engine's
  pattern and the criteria alike.
- **Expression-language conformance.** The shared conformance suite the language follow-up will
  own, which this subsystem runs as a consumer of the language rather than re-specifying.
- **Fuzzing.** The corpus and fuzzer of [§11](#11-security-and-hostile-input).
- **Match overhead.** Plan-time match cost is measured against the hand-written baseline as
  benchmarking matures (`tools/dnn-benchmarking`, [RFC 0013](0013_Autotune.md)); the compiled matcher,
  root-opcode index, and applicability-time cache ([§8](#8-the-matcher-compilation-indexing-and-caching))
  keep it minimal, and the cost is paid once per graph and device.

---

## 14. Migration

Migration follows RFC 0017 §14: no engine is converted until a descriptor-backed kernel runs end to
end, and a
hand-written engine and its descriptor-backed replacement coexist until the generic one reaches parity
on the graphs that engine covers, at which point the hand-written code is retired.

The **SDPA-forward** `isApplicable` (`SdpaFwdPlanBuilder.cpp:169`) is the first conversion, because it
exercises nearly the whole vocabulary (opcode, attribute gates, optional-operand absence, rank, dtype
relations, and cross-tensor dim relations) in one node, with its two non-declarative gates in the
paired native matcher. It splits cleanly: the single `sdpa_fwd` node and its operands are the engine's
pattern, and every gate is criteria. Its match-equivalence
test ([§13](#13-testing-and-performance)) gates the cutover. The mlops builders follow, reusing the
`IValidator` primitives (`dnn-providers/hip-kernel-provider/src/engines/hip_mlops_engine/plans/ApplicabilityChecks.cpp`) as the reference for their criteria
lowering. The kernel-table lookups dissolve into the KDP as described in
[§6](#6-the-native-matcher-escape-hatch).

---

## 15. Worked Example: SDPA Forward

The SDPA-forward check collapses into one UED pattern, one UMD, and one native matcher. Compared to
the hand-written
builder (`SdpaFwdPlanBuilder.cpp:169-317`), the node-shape gates become the engine's pattern, each
remaining C++ gate becomes a criteria sub-expression, and only the one
genuinely non-declarative gate (mask self-consistency) stays in C++ — as a native
matcher the pack lists beside the criteria, not as criteria themselves
([§6](#6-the-native-matcher-escape-hatch)). Note the head dimension is a dim of `$q`, read
positionally, not an attribute ([RFC 0020 § 5](0020_UniversalEngineDescriptor.md#5-the-graph-model-the-pattern-matches)).

This example is grounded on the asm-SDPA builder because that builder is this RFC's first migration
target ([§14](#14-migration)), so the mapping table below doubles as the cutover checklist. It is
deliberately a different example from
[RFC 0017 §13](0017_UniversalKernelDescriptor.md#13-worked-example-sdpa-as-a-ukd), which works the
`attention_dense` kernel family end to end across all seven descriptor kinds; that one shows the pair in
the context of a full UKD, this one shows one hand-written `isApplicable` becoming one pattern plus
one criteria set.

The engine's pattern is a single `sdpa_fwd` node binding `$q`, `$k`, `$v` and the result `$o`, plus
the optional operands this kernel intends to decline — `attn_mask`, `page_table_k`, `page_table_v`,
each declared optional ([RFC 0020 § 4.3](0020_UniversalEngineDescriptor.md#43-the-nodes-pattern-normative)), since an operand the pattern never binds cannot be asked
about at all. The engine's KMD declares `dtype` and `head_size`, the two graph quantities these
kernels bake ([§3](#3-criteria-vocabulary)), so the pack's matcher can pin them per candidate. The
pack's matcher then constrains what that pattern bound:

```jsonc
{
  "version": "1.0",
  "id":   "9c3f5b2a-7d41-4e88-b6a0-1f2e3d4c5b6a",
  "name": "SDPA forward (d128, bf16/fp8) criteria",
  "scope": "kernel",       // reads $kernel.*, so a failure prunes only that candidate
  "criteria": {"and": [
    {"==": ["$q.dtype", "$kernel.dtype"]},                         // the dtype this binary baked
    {"==": ["$k.dtype", "$q.dtype"]}, {"==": ["$v.dtype", "$q.dtype"]},  // q == k == v
    {"==": ["$q.rank", 4]}, {"==": ["$k.rank", 4]},                // (batch, heads, sequence, head dim)
    {"==": ["$v.rank", 4]}, {"==": ["$o.rank", 4]},
    {"==": ["$k.dims[3]", "$q.dims[3]"]},                          // same head dim across q/k/v
    {"==": ["$v.dims[3]", "$q.dims[3]"]},
    {"==": ["$v.dims[1]", "$k.dims[1]"]},                          // same KV head count on k and v
    {"==": ["$q.dims[3]", "$kernel.head_size"]},                   // the head dim this binary baked
    {"or": [{"not_present": ["$sdpa_fwd.dropout_probability"]}, {"==": ["$sdpa_fwd.dropout_probability", 0.0]}]},
    {"==": ["$sdpa_fwd.alibi_mask", false]},
    {"==": ["$sdpa_fwd.padding_mask", false]},
    {"or": [{"not_present": ["$sdpa_fwd.generate_stats"]}, {"==": ["$sdpa_fwd.generate_stats", false]}]},
    // unsupported optional operands declined together; `not_present` always evaluates,
    // unlike a field read on an absent operand
    {"not_present": ["$attn_mask", "$page_table_k", "$page_table_v"]},
    {"==": ["$graph.node_count", 1]}                               // exact: this kernel is the whole graph
  ]}
  // arch is a pack property (KDP.arch), not a match criterion
  // mask self-consistency is a native matcher the pack lists
  // alongside this descriptor (§6), conjoined with these criteria by the ingestor
}
```

Mapping to the hand-written code:

| Hand-written (`SdpaFwdPlanBuilder.cpp`) | Where it lands |
|---|---|
| `getDeviceString` gfx942/gfx950 (:188) | pack `arch` property (KDP), gated at selection |
| `nodeWrappers().size() != 1` (:199) | UMD `{"==": ["$graph.node_count", 1]}` |
| `attributesType() != SdpaAttributes` (:200) | UED pattern node matching `sdpa_fwd` |
| dropout / alibi / padding / stats gates (:208-235) | UMD `$sdpa_fwd.*` criteria |
| `attn_mask` / `page_table_*` absent (:214-220) | UED optional operands + one UMD `{"not_present": [...]}` over all three |
| rank == 4 (:252-263) | UMD `{"==": ["$q.rank", 4]}`, one per bound tensor |
| `q == k == v` dtype (:265-271) | UMD `{"==": ["$k.dtype", "$q.dtype"]}` |
| `k.dims[1] == v.dims[1]` head count (:272-276) | UMD `{"==": ["$v.dims[1]", "$k.dims[1]"]}` |
| head dim, enforced implicitly by a registry miss (`key.empty()`, :309) | UMD `{"==": ["$q.dims[3]", "$kernel.head_size"]}`, stated explicitly |
| `getMaskType` throw-on-contradiction (:293) | native matcher ([§6](#6-the-native-matcher-escape-hatch)) |
| `getKernelNameKey` table lookup (:301) | dissolves into the KDP's Launch ([§6](#6-the-native-matcher-escape-hatch)) |

The split is visible in the two columns: everything about *which graph* is the engine's, everything
about *whether this kernel takes it* is the pack's. The bound symbols the engine publishes
(`$q`..`$o` and every auto-bound dim, stride, and attribute) are what the criteria above read and
what the paired UDD's grid
and argument formulas reference ([RFC 0017 §6](0017_UniversalKernelDescriptor.md#6-dispatch-and-workspace)).

---

## 16. Risks

- **Op-schema registry coupling, inherited.** Every symbol a criterion reads was auto-bound from a
  registry generated off the flatbuffer op schema, so a registry that drifts from the graph
  definitions makes criteria compare the wrong field while still evaluating cleanly. The risk and its
  mitigation — schema-annotation generation and fail-closed resolution — are the UED's
  ([RFC 0020 § 5](0020_UniversalEngineDescriptor.md#5-the-graph-model-the-pattern-matches),
  [RFC 0020 Appendix B](0020_UniversalEngineDescriptor.md#appendix-b-op-schema-registry-generation)). What is this
  document's is the consequence: a criterion cannot detect the drift itself, which is why the
  published set is validated per `(matcher, engine)` pair at load rather than trusted at match time
  ([Appendix A.5](#a5-compile-time-validation-normative)).
- **Expression language sharing.** The expression language is shared with the UDD
  ([§4](#4-the-shared-expression-language)), so a change made for one subsystem can affect the
  other. That risk and its mitigation belong to the language follow-up.
- **Native-symbol contract, at two scopes.** A pack's `match_symbol` names C++ the provider must ship
  ([§6](#6-the-native-matcher-escape-hatch)), and an engine's `graph_match.native` does the same at
  engine scope ([RFC 0020 § 4.5](0020_UniversalEngineDescriptor.md#45-the-native-arm-normative)); a
  drop-in naming an unshipped symbol fails to resolve. Mitigation: version and document the shipped
  symbol set, fail closed with a clear diagnostic, and keep pure-UMD packs free of symbols so they
  remain true drop-ins. The engine-scoped case is the sharper one — an engine on the native arm
  publishes no load-time symbol set, so a UMD's `$`-reference against it is unvalidated until a
  graph arrives ([RFC 0020 § 13.2](0020_UniversalEngineDescriptor.md#132-semantic-validation-cross-descriptor)).
- **Match overhead.** Per-candidate evaluation of the criteria expression is unbounded by the
  root-opcode index ([§8](#8-the-matcher-compilation-indexing-and-caching)). Mitigation: short-circuit
  evaluation, applicability-time caching, and the overhead test of [§13](#13-testing-and-performance).
- **Static-matcher parity.** Should a matcher ever be lowered to a static form, one that diverges
  from the interpreter is a silent correctness bug. Mitigation: the interpreter is the oracle and the
  parity test gates any lowering.
- **Matcher reuse is narrower than pack-scoped sharing suggests.** A UMD's criteria read symbols a
  particular engine's pattern published, so a matcher over tensor and attribute names is reusable
  across packs whose engine publishes a compatible set — in practice, packs on the same engine
  ([§8](#8-the-matcher-compilation-indexing-and-caching)). Only criteria confined to
  `$kernel.*`, `$device.*`, and `$graph.*` are reusable anywhere. Mitigation: none is needed for
  correctness, since pair-validation rejects a mismatch loudly at load, but authors should expect
  matcher libraries to be organized per engine rather than globally, and the `conv.tile_fit` shape —
  a pure `$kernel.*` tile gate — is the pattern to reach for when portability matters.
- **A UED pattern edit invalidates the matchers written against it.** Because the pattern owns the
  symbol table, dropping or renaming a bound variable breaks every UMD, UDD, and the
  UHD that read it. Mitigation: the break is a load-time error naming both descriptors, the
  unresolved reference, and the pack that paired them
  ([§13](#13-testing-and-performance)), never a silent behavior change; a
  pattern is engine-wide and versioned like any descriptor, so a breaking edit is a coordinated
  change in the sense of [RFC 0017 §16](0017_UniversalKernelDescriptor.md#16-risks).
- **The graph-schema floor is engine-grained, so a stale matcher is not declined individually.**
  With `sdk_version` on the UED alone ([§10](#10-serialization-and-versioning)), raising an engine
  to a newer schema admits graphs that use the new field to *every* pack on it, including a matcher
  whose criteria never gate it — the case a per-matcher floor would have skipped. Mitigation: the
  raise is a deliberate, reviewable edit to one descriptor, and the engine's author owns the
  matchers on it; an engine left at its old floor declines those graphs wholesale, which is the safe
  default. The residual risk is an author who raises the engine without auditing its matchers, and
  no mechanism catches that — it is a review obligation, recorded here as one.
- **Engine granularity is now forced by graph shape.** One pattern per UED
  ([§2](#2-the-symbol-table-criteria-read)) means a family serving two structurally different topologies must
  split into two engines, each with its own KMD and UHD, even where the kernels are otherwise
  siblings. Mitigation: the pattern's opcode-set and optional-operand forms ([RFC 0020 § 4.3](0020_UniversalEngineDescriptor.md#43-the-nodes-pattern-normative)) absorb most variation
  without a split, and a genuinely different topology already implied a different metadata schema and
  heuristic. The residual cost is more engines in the id space, which
  [RFC 0017 §4](0017_UniversalKernelDescriptor.md#4-descriptor-formats) sizes for hundreds to low
  thousands.
- **A positional axis read is legible only by convention.** A criterion names an axis by index, not
  by name: `$q.dims[3]` is the head dimension because the pattern's operand order says so
  ([RFC 0020 § 5](0020_UniversalEngineDescriptor.md#5-the-graph-model-the-pattern-matches)). A wrong
  index therefore resolves and evaluates cleanly, so the pair-validation of
  [Appendix A.5](#a5-compile-time-validation-normative) proves the *tensor* is bound but never that
  the *axis* is the intended one, and the same index recurs across criteria, the UDD's grid formulas,
  and the UHD's `features_signature`. The `stride_order` encoding sharpens this: a stride-rank array
  is read against the logical axis order rather than spelling the layout out, so BSHD is `[3,1,2,0]`
  ([§5](#5-layout-and-stride-order-criteria)) and a transposed pair of entries is a legal
  permutation that silently names a different layout. Mitigation: pin `$x.rank` beside every
  positional read, comment the axis at each site, and rely on the match-equivalence tests of
  [§13](#13-testing-and-performance) against the hand-written builder to catch what static
  validation structurally cannot. Whether dims may be named at all is the shape-matching follow-up's
  ([RFC 0017 §14.2](0017_UniversalKernelDescriptor.md#142-follow-up-rfcs)).

---

## 17. Open Questions

1. **Native-criterion bindings. SETTLED — the contract was extended.** A native criterion now takes
   `(const MatchContext&, const BoundTokens&)`, so it reads the binding the engine's `graph_match`
   published rather than re-locating tensors by hand ([§6](#6-the-native-matcher-escape-hatch)).
   The same widening applies to the kernel-scoped matcher and the scorer, all three sharing one
   operand order with the engine's match
   ([RFC 0020 § 6.1](0020_UniversalEngineDescriptor.md#61-the-published-field-set-normative)). What
   the question weighed — extend the signature, or accept the duplication — is answered in favor of
   extending it. The residual is representational, not structural: `BoundTokens` carries scalars,
   so whole-tensor access still goes through `MatchContext` by uid ([§6](#6-the-native-matcher-escape-hatch)).
2. **Static-matcher form.** Should a matcher be pre-compiled into a static form that cuts runtime
   cost — interpreted IR, a serializable bytecode, or generated C++ — and can one form serve both the
   AOT and drop-in paths? Whatever is chosen must decide identically to the interpreted matcher.
3. **Feature-vector overlap.** Largely settled by the split: the UED's pattern is engine-wide and
   publishes the tensor, dim, and attribute symbols a UHD's `features_signature` reads, so an
   engine's binding is the natural canonical feature source
   ([RFC 0017 §17 Q4](0017_UniversalKernelDescriptor.md#17-open-questions)). What remains is whether
   a *portable* extractor across engines is still wanted for model reuse, or whether per-engine
   feature spaces are the right granularity.

---

## 18. References and Prior Art

The design borrows established ideas; none is a dependency. These informed the matcher specifically.

| System | Idea borrowed |
|---|---|
| **MLIR PDL / PDLL** | Two-layer design: a declarative pattern compiled once to a fast matcher; constraints inline on the binding; a named native-predicate escape hatch; pattern priority for arbitration |
| **TVM Relax DFPattern** | Constraint vocabulary (op, dtype, symbolic shape, wildcard); dataflow use-def constraints; cross-tensor same-shape relations |
| **XLA pattern matcher** | Exact-vs-compatible equality; a tensor virtual/internal flag gating fusion; layout as a distinct constraint; optional operands; capture-by-reference binding |
| **PyTorch Inductor / torch.library** | Node/edge pattern vocabulary; serialized precompiled patterns; duplicate-pattern detection |
| **LLVM ISel / discrimination nets** | Sharing common prefixes of many patterns rooted at one opcode into one decision structure |
| **ONNX Runtime** | First-claim arbitration as the anti-pattern this RFC replaces with deterministic ranking; single-node versus fused-subgraph capability |

---

## 19. Glossary

- **UMD (Universal Match Descriptor) / matcher:** one criteria expression that decides whether a
  kernel applies, evaluated over the symbols its engine's pattern bound. A KDP lists a set of matcher
  IDs; a kernel applies only when all pass. Reused across packs whose engine publishes the symbols it
  reads ([§3](#3-criteria-vocabulary)).
- **UED (Universal Engine Descriptor) / the engine's pattern:** the engine, which besides naming its
  one metadata schema and optionally one heuristic carries the `graph_match`: the graph shape it
  serves and the binding that shape produces, as either a declarative `nodes` block or a native
  symbol. One match per engine; specified in [RFC 0020](0020_UniversalEngineDescriptor.md).
- **Two-stage matching:** the pattern binds once per engine per graph
  ([RFC 0020 § 7](0020_UniversalEngineDescriptor.md#7-pattern-matching-stage-one)), then each pack's criteria
  evaluate over that binding ([§8](#8-the-matcher-compilation-indexing-and-caching)).
- **Criteria expression:** the single JsonLogic `{"op": [args]}` boolean a UMD evaluates over the
  engine's bound symbol table, typically an `and` of the individual tests ([§3](#3-criteria-vocabulary)).
- **Published symbol set:** what a UED's pattern binds and every consumer is validated against; a
  reference that does not resolve is a load error, not a runtime decline. The set itself is
  [RFC 0020 § 6.1](0020_UniversalEngineDescriptor.md#61-the-published-field-set-normative); the reader's view of it is
  [§2](#2-the-symbol-table-criteria-read).
- **Op-schema registry:** the generated table mapping each op type to its operand/result UID fields and
  attributes, letting the matcher reconstruct edges and auto-bind
  ([RFC 0020 § 5](0020_UniversalEngineDescriptor.md#5-the-graph-model-the-pattern-matches)).
- **JsonLogic:** the descriptor expression language, owned by a deferred follow-up with
  [RFC 0017 §5](0017_UniversalKernelDescriptor.md#5-matching-the-ueds-pattern-and-the-umds-criteria)
  as the interim authority; a UMD's criteria are its boolean-rooted form over
  the five namespaces of [§2](#2-the-symbol-table-criteria-read), a UDD's dispatch formulas
  and the engine's UHD `features_signature` entries its value-rooted form over the same table
  ([§4](#4-the-shared-expression-language)).
- **Stride-order layout:** layout represented as a per-logical-dimension stride-rank array, lower
  meaning faster-varying, since tensors carry no layout enum ([§5](#5-layout-and-stride-order-criteria)).
- **Native criterion:** the pack-scoped escape hatch; a `GraphCriterionFn` named by a UMD's
  `match_symbol` and conjoined with the pack's other matchers by the ingestor, for logic
  the built-in operators cannot state. It **reads** the engine's binding and returns a verdict. It
  lives beside the expression language, never inside it
  ([§6](#6-the-native-matcher-escape-hatch)).
- **Native match:** the engine-scoped counterpart, a `GraphMatchFn` named by the UED's
  `graph_match.native`, which **produces** the binding by declining the graph or returning the
  bound tokens ([RFC 0020 § 4.5](0020_UniversalEngineDescriptor.md#45-the-native-arm-normative)).
  Distinct registry, distinct signature, distinct scope.
- **Composite criteria:** any boolean combination of tests within the one `criteria` expression
  ([§7](#7-composite-criteria)).
- **Arbitration:** the deterministic resolution when several UKDs match: heuristic (UHD) score, then
  `priority`, then stable `id` compared as raw bytes ([§9](#9-arbitration)).
- **Catalog / bound token state:** the two products of matching a graph — the kernels whose full
  matcher set passed, and every `$`-prefixed value the matchers resolved. Both are cached by the
  provider during applicability and read by every later phase
  ([RFC 0017 §8](0017_UniversalKernelDescriptor.md#8-end-to-end-flow)).
- **Root-opcode index:** the index of compiled patterns by root opcode that keeps match cost sublinear
  in descriptor count; a miss prunes an engine and every pack naming it
  ([RFC 0020 § 7](0020_UniversalEngineDescriptor.md#7-pattern-matching-stage-one)).

---

## Appendix A: Schema Reference

This appendix is the normative schema for the UMD descriptor object. The engine's `graph_match` and the field
set matching it publishes are specified with the UED and not here
([RFC 0020 § 4.3](0020_UniversalEngineDescriptor.md#43-the-nodes-pattern-normative), [RFC 0020 § 6.1](0020_UniversalEngineDescriptor.md#61-the-published-field-set-normative)).
Where the prose sections above describe a construct by example, the grammar and tables here fix its exact form. A descriptor that violates a
**MUST** here is refused at compile ([§8](#8-the-matcher-compilation-indexing-and-caching)); it never
matches by default ([§11](#11-security-and-hostile-input)). Grammar is EBNF; quoted terminals are JSON
tokens. The expression language's own normative reference — grammar, operator table, type rules, and
static validation — belongs to the descriptor expression language follow-up; this
appendix fixes the descriptor object and the hipDNN environment its criteria are evaluated over.

### A.1 The UMD descriptor object

| Field | Type | Required | Default | Rule |
|---|---|---|---|---|
| `id` | string (UUID) | yes | — | A UUID; stable, globally unique identity ([§10](#10-serialization-and-versioning)) |
| `name` | string | yes | — | Diagnostics only; not semantic |
| `version` | string | no | `"1.0"` | Matcher format version, `<major>.<minor>`, gated at load as a **ceiling**: a differing `major`, or a `minor` newer than the runtime's, is refused; an older minor always loads ([§10](#10-serialization-and-versioning)) |
| `allow_override_shape` | bool | no | `false` | The matcher's opt-in to accepting a graph that enables execute-time override shapes. When `false`, such a graph is declined before the criteria run. This is the matcher's own gate and is distinct from `$graph.is_override_shape_enabled`, which is the graph's state ([§2](#2-the-symbol-table-criteria-read), [RFC 0020 § 6](0020_UniversalEngineDescriptor.md#6-symbol-binding-what-the-pattern-publishes)). A prebuilt kernel that bakes its shape leaves this at the default rather than restating the condition as a criterion |
| `criteria` | Expr | see below | — | A single expression whose static type is `Bool` (A.3) |
| `scope` | `"graph"` \| `"kernel"` | yes | — | Which inputs the criteria read, and so what a failure prunes: `graph` is evaluated once per `(graph, device)` and disqualifies **every** kernel in the pack; `kernel` also reads `$kernel.*` and disqualifies **only the candidate** ([§8](#8-the-matcher-compilation-indexing-and-caching)). It is declared rather than inferred so the pruning level is a stated contract, not a consequence of which tokens an expression happens to name |
| `match_symbol` | string | no | — | The **native criterion** escape hatch: a symbol naming a `GraphCriterionFn` the provider ships, resolved through its registry ([§6](#6-the-native-matcher-escape-hatch)) |

**A UMD carries exactly one of `criteria` and `match_symbol`.** One that declares neither states no
check; one that declares both hides a conjunction inside a descriptor that a pack states by listing
two matcher ids ([§6](#6-the-native-matcher-escape-hatch)). Either is refused. No other top-level
keys are permitted, and an unknown key is refused. In particular a UMD carries no
`schema` member — the `.umd.json` filename already states the type, and a file whose name and body
disagree has no correct reading, so the body does not restate it ([§10](#10-serialization-and-versioning)).
Nor does it carry `nodes`: the pattern is the engine's ([§2](#2-the-symbol-table-criteria-read)). Nor
`sdk_version`: the graph-schema floor is declared once, on the UED, and a matcher runs under the
floor of the engine of each pack that lists it ([§10](#10-serialization-and-versioning)). `version`
compares numerically by `(major, minor)`, so `1.10` is above `1.9`; a value that does not parse as
exactly two decimal components is refused.

A UMD names no engine. It is bound to one by the KDPs that list it
([RFC 0017 §4](0017_UniversalKernelDescriptor.md#4-descriptor-formats)), and its `$`-references are
resolved per `(matcher, engine)` pair rather than in isolation (A.5).

### A.2 Variable references and resolution

A `$`-reference is spelled and resolved as the descriptor expression language follow-up will
specify. The environment it resolves against — the five namespace roots, the fields each carries,
their types, and the reserved-root rule — is
[RFC 0020 § 6.1](0020_UniversalEngineDescriptor.md#61-the-published-field-set-normative), summarized for the reader in
[§2](#2-the-symbol-table-criteria-read). What follows is normative for a *criteria* expression on top
of it.

- A reference resolves against the published set of the engine of each pack that lists this UMD; one
  that does not resolve there is refused at pair-validation, never at match time (A.5).
- A field access on an **absent** optional operand or attribute (e.g. `$attn_mask.dtype` when
  `attn_mask` is absent) yields **unknown**, and unknown is resolved at two levels. *Per operand*, it
  **propagates** through the enclosing expression rather than short-circuiting it: an `or` with a
  definite-`true` arm is `true`, and an `and` with a definite-`false` arm is `false`, whichever way
  the unknown arm would have gone. *At the root*, a criteria expression whose `Bool` root still holds
  unknown fails closed and declines the match ([§11](#11-security-and-hostile-input)). The first
  level is what lets the "absent, or present and constrained" pair of
  [§3](#3-criteria-vocabulary) accept a graph without the operand; the second is what stops an
  undecided criterion from admitting one. This restates
  [RFC 0017 §5](0017_UniversalKernelDescriptor.md#5-matching-the-ueds-pattern-and-the-umds-criteria)
  for the reader's side and moves to the expression-language follow-up with the rest of the
  semantics.
- Presence itself is never a field read, so it never yields unknown: `{"present": ["$attn_mask"]}`
  and `{"not_present": ["$attn_mask"]}` always evaluate to a definite answer, which is what makes
  them usable as the guard arm above.
- An out-of-range `dims[i]`/`strides[i]` is unknown on the same terms, one outcome for every
  producer of it. `value_f32` resolves only when the tensor carries a compile-time value and is
  unknown on one that does not.

### A.3 The expression language

The criteria expression grammar, the operator table with arities and types, and the type rules
belong to the descriptor expression language follow-up. Until it lands,
[RFC 0017 §5](0017_UniversalKernelDescriptor.md#5-matching-the-ueds-pattern-and-the-umds-criteria)
is the interim authority for the operator vocabulary, and the semantics this document depends on
are stated locally: unknown propagation in A.2, short-circuit order in
[§8](#8-the-matcher-compilation-indexing-and-caching), and the interpreter's bounds in
[§11](#11-security-and-hostile-input). Each moves out when the follow-up is written.

The operator set is closed in the sense that matters to a descriptor: there is no registry,
namespace, dotted key, or provider hook by which one introduces an operator, so an unlisted
operation key — including a dotted one such as `{"hipdnn.mask_self_consistent": [...]}` — is refused at
compile at whatever revision is in force. The published vocabulary still grows additively across
revisions. A check the operators cannot express belongs to a native criterion listed beside this
descriptor in the pack ([§6](#6-the-native-matcher-escape-hatch)).

### A.4 `stride_order` values and layout aliases

A `stride_order` comparison accepts either an integer array or an alias string; aliases expand to the
array at compile time, and the array is the single canonical form ([§5](#5-layout-and-stride-order-criteria)).
An array MUST be a permutation of `0 .. rank-1` giving, for each logical dimension `d`, that
dimension's stride rank, `0` being the fastest-varying, matching
[RFC 0017 §5](0017_UniversalKernelDescriptor.md#5-matching-the-ueds-pattern-and-the-umds-criteria).

| Alias | Array | | Alias | Array |
|---|---|---|---|---|
| `nchw` | `[3,2,1,0]` | | `ndhwc` | `[4,0,3,2,1]` |
| `nhwc` | `[3,0,2,1]` | | `bhsd` | `[3,2,1,0]` |
| `ncdhw` | `[4,3,2,1,0]` | | | |

Every alias is fixed-rank, so an alias compared against a tensor the criteria pin to a different rank
is refused at compile rather than declining silently at match time.

### A.5 Compile-time validation (normative)

A UMD MUST pass every check below to compile; a failure refuses (and, on the drop-in path,
quarantines) the descriptor with a diagnostic ([§8](#8-the-matcher-compilation-indexing-and-caching),
[§11](#11-security-and-hostile-input)). They fall into two groups, because a UMD's references cannot
be resolved without an engine to resolve them against
([§8](#8-the-matcher-compilation-indexing-and-caching)). The pattern's own validation belongs to
the UED ([§2](#2-the-symbol-table-criteria-read)).

**The UMD alone:**

1. `id` is a well-formed UUID, only the keys of A.1 at the top level, and `version`, when present,
   is a well-formed `<major>.<minor>` string the runtime can honor: same `major`, and a `minor` no
   newer than the runtime's ([§10](#10-serialization-and-versioning)).
2. `scope` is `"graph"` or `"kernel"`, and exactly one of `criteria` and `match_symbol` is
   present (A.1).
3. `criteria`, when present, passes the expression language's static validation — operator
   recognition, arity, argument types, and the `Bool` root — and every layout alias in it
   resolves (A.4). Its `$kernel.*` reads must agree with the declared `scope`, in both
   directions. A `graph`-scoped UMD whose criteria read any `$kernel.*` is **refused**: a
   graph-scoped verdict is computed once and disqualifies every kernel in the pack
   ([A.1](#a1-the-umd-descriptor-object)), so reading a per-candidate field there would prune
   the whole pack on one arbitrary candidate's metadata. A `kernel`-scoped UMD whose criteria
   read no `$kernel.*` is the harmless converse: accepted, but diagnosed, since it pays
   per-candidate evaluation for a decision that cannot vary by candidate. The compiler already
   computes the `$kernel.*` read set to build the memoization projection
   ([§8](#8-the-matcher-compilation-indexing-and-caching)), so neither check costs a second walk.
4. `match_symbol`, when present, is registered in the provider's registry
   ([§6](#6-the-native-matcher-escape-hatch)); an unregistered symbol refuses the descriptor
   rather than deferring the failure to match time.

**The UMD against the engine of each pack that lists it:**

5. Every `$`-reference in `criteria` resolves to a symbol that engine's pattern published — a
   pattern variable, a node `id`'s attribute, or a reserved `$graph.*` / `$device.*` root (A.2).
6. Every `$kernel.*` field the criteria read is declared by that engine's KMD
   ([§3](#3-criteria-vocabulary)).

Checks 5 and 6 are cached on `(matcher, engine)` and re-run when either side changes; a failure
names the unresolved reference, both descriptors, **and the pack that paired them**, without which
the reader cannot tell which of an engine's packs to correct
([§12](#12-observability-and-diagnostics)). The
same two checks apply to a pack's UDD formulas and to the engine's UHD `features_signature`, which is
what makes the engine's published set the single contract
([§2](#2-the-symbol-table-criteria-read)).
