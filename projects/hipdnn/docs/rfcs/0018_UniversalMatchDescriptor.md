# RFC 0018: Universal Match Descriptor (UMD) and the Graph Matcher

- Contributors: Brian Harrison

> Follow-up to [RFC 0017 (Universal Kernel Descriptors)](0017_UniversalKernelDescriptor.md),
> the "UMD + graph matcher" row of its follow-up series ([RFC 0017 §12.2](0017_UniversalKernelDescriptor.md#122-follow-up-rfcs)).
> This RFC designs the match format and the matcher; the sibling formats (UDD, UED, UHD, KDP) and
> subsystems (packaging, drop-in, adapters) are designed in their own follow-ups and are referenced,
> not redesigned, here. The RFC number is provisional and is reconciled against the concurrent
> follow-up series at PR-open time.

## Table of Contents

1. [Overview](#1-overview)
2. [What the UMD Replaces](#2-what-the-umd-replaces)
3. [The Matcher's Input: hipDNN's Graph Model](#3-the-matchers-input-hipdnns-graph-model)
4. [Structural Pattern](#4-structural-pattern)
5. [Symbol Binding and the Auto-Binding Formula](#5-symbol-binding-and-the-auto-binding-formula)
6. [Constraint Vocabulary](#6-constraint-vocabulary)
7. [The Shared Expression Language](#7-the-shared-expression-language)
8. [Layout and Stride-Order Constraints](#8-layout-and-stride-order-constraints)
9. [Native-Predicate Escape Hatch](#9-native-predicate-escape-hatch)
10. [Composite Constraints (Deferred, Forward-Compatible)](#10-composite-constraints-deferred-forward-compatible)
11. [The Matcher: Compilation, Indexing, and Caching](#11-the-matcher-compilation-indexing-and-caching)
12. [Static Matcher (Sketch)](#12-static-matcher-sketch)
13. [Arbitration](#13-arbitration)
14. [Serialization and Versioning](#14-serialization-and-versioning)
15. [Security and Hostile Input](#15-security-and-hostile-input)
16. [Observability and Diagnostics](#16-observability-and-diagnostics)
17. [Testing and Performance](#17-testing-and-performance)
18. [Migration](#18-migration)
19. [Worked Example: SDPA Forward](#19-worked-example-sdpa-forward)
20. [Risks](#20-risks)
21. [Open Questions](#21-open-questions)
22. [References and Prior Art](#22-references-and-prior-art)
23. [Glossary](#23-glossary)

---

## 1. Overview

A **UMD (Universal Match Descriptor)** is the declarative data that decides whether a kernel applies to
an incoming problem graph, and, in the same pass, binds the named variables that the kernel's dispatch
and workspace formulas reference. It replaces the graph half of a hand-coded
`IPlanBuilder::isApplicable` check ([RFC 0017 §2](0017_UniversalKernelDescriptor.md#2-the-descriptors)).
One UMD is authored once and referenced by ID from many UKDs, so a family of near-identical kernels
shares a handful of matches rather than carrying a bespoke C++ check each.

A UMD has two parts, unchanged in intent from
[RFC 0017 §5](0017_UniversalKernelDescriptor.md#5-matching-and-the-umd):

1. A **structural pattern**: named operation nodes and their named operand and result edges. Because
   hipDNN op graphs are DAGs, the pattern is an explicit node-and-edge graph, not a nested expression.
2. A **constraint list** attached to those nodes and edges. Constraints implicitly AND together.

This document turns that frame into a concrete format and a concrete matcher. It specifies the pattern
and constraint schema, the standard formula that auto-binds every tensor and attribute of a matched op,
a shared expression language (used here for boolean constraints and reused by the UDD for dispatch
formulas), the layout representation as a stride-order index array, the native-predicate escape hatch,
deterministic arbitration, and the compile-once matcher with its indexing and caching. The static
(compile-time) matcher is sketched as options, not fully designed, in this iteration.

### 1.1 What This RFC Specifies Versus Defers

| Capability | This RFC (day-one) | Deferred |
|---|---|---|
| Structural pattern: op nodes, named operand/result edges, single-op and bounded fused subgraphs | Yes ([§4](#4-structural-pattern)) | None |
| Auto-binding of every operand/result tensor, its dims and strides, and every op attribute | Yes ([§5](#5-symbol-binding-and-the-auto-binding-formula)) | None |
| Constraints: opcode, dtype (exact/one_of/relation), shape/rank, symbolic-dim unification, layout, packing, attribute, use-count, cross-tensor relation, optional operand, device/arch | Yes ([§6](#6-constraint-vocabulary)) | None |
| Shared expression language: arithmetic value core (UDD-shared) plus a boolean predicate layer (UMD) | Yes ([§7](#7-the-shared-expression-language)) | New operators as needed |
| Layout as a stride-order index array, with named aliases | Yes ([§8](#8-layout-and-stride-order-constraints)) | None |
| Native-predicate escape hatch, registry-resolved | Yes ([§9](#9-native-predicate-escape-hatch)) | None |
| Composite constraints: `(A AND B) OR C` as one constraint | Forward-compatible schema only ([§10](#10-composite-constraints-deferred-forward-compatible)) | Evaluation semantics |
| Compile-once matcher, root-opcode index, per-plan match cache | Yes ([§11](#11-the-matcher-compilation-indexing-and-caching)) | None |
| Static (compile-time / AOT-lowered) matcher | Options sketched ([§12](#12-static-matcher-sketch)) | Full design |
| General N-ary commutative matching, unbounded variable-length chains | None | JIT follow-up ([RFC 0017 §8.3](0017_UniversalKernelDescriptor.md#83-future-jit-and-normalized-providers)) |

---

## 2. What the UMD Replaces

Every check the UMD must express already exists in the provider as hand-written C++. Two families show
the full range: the `asm_sdpa_engine` builders inline their gates, and the `hip_mlops_engine` builders
delegate tensor-level gates to reusable `IValidator` primitives
(`dnn-providers/hip-kernel-provider/src/engines/hip_mlops_engine/plans/ApplicabilityChecks.cpp`). Each
hand-coded check kind maps to a UMD construct or, where it needs real C++, to a named native predicate.

| Hand-written check (with a representative site) | UMD construct |
|---|---|
| Override-shape gate (`SdpaFwdPlanBuilder.cpp:177`) | implicit precondition; matcher declines override-shape graphs unless a UMD opts in ([§4](#4-structural-pattern)) |
| Device arch gate `gfx942`/`gfx950` (`SdpaFwdPlanBuilder.cpp:186`) | `device.arch` constraint ([§6](#6-constraint-vocabulary)) |
| Node-count and single-node gate (`SdpaFwdPlanBuilder.cpp:199`) | the pattern's node set is exact ([§4](#4-structural-pattern)) |
| Node attribute-type gate (`attributesType() == SdpaAttributes`) (`SdpaFwdPlanBuilder.cpp:200`) | node `op` opcode ([§4](#4-structural-pattern)) |
| Per-attribute value gates: dropout unset/zero, alibi false, padding false, `generate_stats` false (`SdpaFwdPlanBuilder.cpp:205-224`) | `attr` constraints ([§6](#6-constraint-vocabulary)) |
| Optional operand absent: `attn_mask`, `page_table_k/v` UIDs (`SdpaFwdPlanBuilder.cpp:209-215`) | optional-operand `absent` constraint ([§6](#6-constraint-vocabulary)) |
| Tensor rank == 4 (`SdpaFwdPlanBuilder.cpp:231-247`) | `rank` or an exact `shape` ([§6](#6-constraint-vocabulary)) |
| Cross-tensor dtype equality `q == k == v` (`SdpaFwdPlanBuilder.cpp:244`) | dtype relation ([§6](#6-constraint-vocabulary)) |
| Cross-tensor dim relation, head count `k.dims[1] == v.dims[1]` (`SdpaFwdPlanBuilder.cpp:251`) | symbolic-dim unification ([§5](#5-symbol-binding-and-the-auto-binding-formula)) |
| Packed-tensor and supported-layout gates (`ApplicabilityChecks.cpp:65,77`) | packing and layout constraints ([§8](#8-layout-and-stride-order-constraints)) |
| Consistent-layout across tensors (`ApplicabilityChecks.cpp:106`) | cross-tensor layout relation ([§8](#8-layout-and-stride-order-constraints)) |
| Head dim `one_of {64,128,192}` (`SdpaBwdPlanBuilder.cpp:597`) | `one_of` on a bound dim ([§7](#7-the-shared-expression-language)) |
| GQA divisibility `nhead_q % nhead_k == 0 && nhead_k != 0` (`SdpaBwdPlanBuilder.cpp:548`) | expression predicate, or a native predicate if fail-closed overflow matters ([§9](#9-native-predicate-escape-hatch)) |
| uint32 byte-stride-overflow guard (`SdpaFwdPlanBuilder.cpp:294`, `SdpaPlanUtils.hpp:159`) | **native predicate** ([§9](#9-native-predicate-escape-hatch)) |
| Kernel-name-key / CSV-registry table lookups (`SdpaFwdPlanBuilder.cpp:287`, three in `SdpaBwdPlanBuilder.cpp:660`) | **native predicate** (or removed once the KDP names the code object directly) ([§9](#9-native-predicate-escape-hatch)) |
| Mask classification `getMaskType` throwing on contradiction (`SdpaFwdPlanBuilder.cpp:276`) | attr constraints, with a native predicate for the contradiction check ([§9](#9-native-predicate-escape-hatch)) |
| NumPy-broadcast affine-shape compatibility (`BatchnormApplicabilityChecks.cpp:169`), normalized-dim reconciliation (`LayernormApplicabilityChecks.cpp:68`), inv_rms derived shape (`RMSnormApplicabilityChecks.cpp:106`) | **native predicate** (derived-quantity relations) ([§9](#9-native-predicate-escape-hatch)) |
| Fusion wiring: intermediate is virtual and privately used (`BatchnormPlanBuilder.cpp:184`) | `use_count` / exclusivity plus a virtual-tensor constraint ([§6](#6-constraint-vocabulary)) |

The provider today has **no arbitration**: each engine loops its plan builders and takes the first that
returns true (`AsmSdpaEngine.cpp:32`, `HipMlopsEngine.cpp:36`), with a standing comment that this
"is wrong if we ever have more than 1 plan builder thats applicable" (`HipMlopsEngine.cpp:34`).
Correctness depends on builders being mutually exclusive by construction. The UMD makes overlap
explicit and resolves it deterministically ([§13](#13-arbitration)).

---

## 3. The Matcher's Input: hipDNN's Graph Model

The matcher reads an immutable graph through the existing `IGraph` interface
(`projects/hipdnn/flatbuffers_sdk/include/hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp`)
plus the HIP stream for device properties. Three properties of that model drive the design.

**The graph is UID-centric, not edge-centric.** A `Node` carries only `{name, compute_data_type,
attributes_type, attributes}` (`graph_generated.h`, Node table). It has no input or output tensor
lists. A node's operands and results are UID fields inside its concrete attribute table, for example
`SdpaAttributes::q_tensor_uid()`, `k_tensor_uid()`, `o_tensor_uid()` (`sdpa_attributes_generated.h`).
Connectivity between nodes is implicit: two nodes are connected when a result UID of one appears as an
operand UID of the other. To resolve a node's edges, the matcher must know the op type, cast via
`attributesAs<T>()`, and read the per-role UID fields.

**Consequence: the matcher needs an op-schema registry.** For each op type, the registry declares which
attribute fields are operand UIDs, which are result UIDs, whether each is required or optional, and the
names of the op's scalar attributes. This registry is what lets a UMD name operands and results by
role (`Q`, `K`, `V`, `O`) and what powers the auto-binding formula of
[§5](#5-symbol-binding-and-the-auto-binding-formula). It is generated from the flatbuffer op schema,
so it stays in lockstep with the graph definitions rather than being hand-maintained.

**Tensors expose dims, strides, dtype, and a virtual flag, but no layout enum and no rank field.**
`TensorAttributes` (`tensor_attributes_generated.h`) offers `dims()`, `strides()` (both nullable
vectors), `data_type()`, `uid()`, and `virtual_()`. Rank is `dims()->size()`. Layout is not stored; it
is derived from the stride order, which is why the UMD represents layout as a stride-order index array
([§8](#8-layout-and-stride-order-constraints)). Quantities like `head_size`, `batch`, and `num_heads`
are **not** attributes; they are specific tensor dims (for SDPA, `q.dims[3]`, `q.dims[0]`,
`q.dims[1]`). The UMD binds them as shape symbols, not as `attr` reads. (RFC 0017's illustrative
`attr: {head_size: ...}` is imprecise on this point; this RFC uses the shape-symbol form.)

**Device and arch are out-of-band.** The graph carries no device identity. Arch comes from the stream
via `getDeviceString(handle.getStream())` (`HipDeviceUtils.hpp:48`), so the matcher receives the
`Handle` alongside the graph, and a `device` constraint resolves against it rather than against a graph
field.

**Graph guarantees the matcher may rely on.** Per the `IGraph` contract the graph is topologically
sorted, acyclic, fully connected, and has unique tensor UIDs. The matcher builds its own
UID-to-producer and UID-to-consumers index once per graph to walk edges and to evaluate use-count
constraints, since no adjacency query is provided.

![The matcher reads a UID-centric graph via an op-schema registry that reconstructs edges and auto-binds symbols](../images/umd_binding_model.svg)

---

## 4. Structural Pattern

The pattern is a set of op nodes and the named edges between them and the graph's tensors. Each node
declares its opcode and maps operand and result **roles** (from the op-schema registry) to pattern
variables (`$q`, `$conv_out`).

```jsonc
{
  "schema": "hipdnn.umd/v1",
  "id":   1180449020,
  "name": "SDPA forward (d128, bf16) match",
  "nodes": [
    {"kind": "op", "id": "$root", "op": "sdpa_fwd",
     "operands": {"Q": "$q", "K": "$k", "V": "$v"}, "results": {"O": "$o"}}
  ],
  "constraints": [ /* Section 6 */ ]
}
```

- **Node identity.** Each node has a pattern-local `id` (`$root`, `$conv`, `$add`) used to target
  constraints and diagnostics. It is distinct from the descriptor's global `id`.
- **Opcode.** `op` names one opcode, or `{"one_of": [...]}` for a small fixed set, or `"any"` for a
  wildcard node (used only inside a bounded fused pattern).
- **Roles.** Keys in `operands` and `results` are op-schema role names; values are pattern variables.
  A role the schema marks optional may be omitted from the map, or bound and constrained `absent`
  ([§6](#6-constraint-vocabulary)). Roles not named are ignored for matching but still auto-bound
  ([§5](#5-symbol-binding-and-the-auto-binding-formula)).
- **Edges are implicit through shared variables.** Two nodes are connected when the same variable
  appears as a result of one and an operand of another. In the fused example below, `$conv_out` is
  `$conv`'s result and `$add`'s operand, which is the edge.
- **Node set is exact by default.** A single-node pattern matches only a single-node graph, replacing
  the `nodeWrappers().size() != 1` gate. A multi-node pattern matches a connected subgraph of exactly
  those op nodes; whether the subgraph must be the whole graph or may be embedded is governed by the
  fusion legality constraints (`use_count`) below.
- **Override-shape graphs** are declined by default (mirroring today's gate); a UMD that supports them
  sets `"allow_override_shape": true` at the top level.

**Fusion is day-one.** A multi-node pattern binds the whole fused subgraph at once and hands it to one
UKD. Legality (that the fused intermediates have no consumer outside the pattern) is expressed with
`use_count` constraints:

```jsonc
{
  "schema": "hipdnn.umd/v1",
  "id":   6620551007,
  "name": "Conv-Bias-ReLU (NHWC, f16) match",
  "nodes": [
    {"kind": "op", "id": "$conv", "op": "convolution_fwd",
     "operands": {"X": "$x", "W": "$w"},          "results": {"Y": "$conv_out"}},
    {"kind": "op", "id": "$add", "op": "pointwise_add",
     "operands": {"A": "$conv_out", "B": "$bias"}, "results": {"Y": "$bias_out"}},
    {"kind": "op", "id": "$act",  "op": "pointwise_relu",
     "operands": {"A": "$bias_out"},               "results": {"Y": "$y"}}
  ],
  "constraints": [
    {"on": "$x",   "dtype": {"one_of": ["FLOAT16"]}, "layout": [0, 3, 1, 2]},
    {"on": "$y",   "dtype": {"one_of": ["FLOAT16"]}, "layout": [0, 3, 1, 2],
     "shape": ["$batch", "$out_h", "$out_w", "$out_channels"]},
    {"on": "$bias", "shape": ["$out_channels"]},
    {"on": "$conv_out", "use_count": 1},
    {"on": "$bias_out", "use_count": 1}
  ]
}
```

Out of scope for this iteration, as in [RFC 0017 §5](0017_UniversalKernelDescriptor.md#5-matching-and-the-umd):
general N-ary commutative matching and unbounded variable-length chains. Bounded commutative pairs and
bounded optional slots cover the fusion cases a prebuilt kernel needs.

---

## 5. Symbol Binding and the Auto-Binding Formula

Matching does double duty: it decides applicability and it binds named variables. A symbol is
**declared** in the UMD, **bound** when the graph matches, and **used** by the UDD's dispatch and
workspace formulas ([RFC 0017 §6](0017_UniversalKernelDescriptor.md#6-dispatch-and-workspace)). Every
symbol a formula references must be bound by the match, so a formula can only read values the match
actually produces. The UMD publishes its bound-symbol set; each UKD that pairs this UMD with a UDD is
checked at build and at drop-in load, and a UDD that references an unbound symbol is rejected then
rather than failing closed later on a live graph.

**Auto-binding is the default, and follows a standard formula (AICK-1698).** When a pattern names an
operand or result variable, the matcher, using the op-schema registry, automatically binds:

- The **tensor handle** itself: `$q` (a pointer/UID reference).
- Each **dim**, addressable positionally as `$q.dims[i]` and, as authoring sugar, as `$q<i>` (so `$q0`
  is `$q.dims[0]`, `$q3` is `$q.dims[3]`).
- Each **stride**, addressable as `$q.strides[i]`.
- Derived tensor facts: `$q.rank`, `$q.dtype`, `$q.stride_order` ([§8](#8-layout-and-stride-order-constraints)),
  `$q.is_virtual`.

When a pattern names a node, the matcher auto-binds every scalar attribute of that op, addressable as
`$root.<attr>` (for example `$root.causal_mask`, `$root.dropout_probability`). Optional attributes carry
a presence flag, `$root.<attr>.present`.

**A `$` marks a bound symbol.** Every reference to a bound symbol carries a leading `$`: tensor handles
(`$q`), their dims, strides, and derived facts (`$q0`, `$q.dims[2]`, `$q.rank`), named shape symbols
(`$head_size`), and the node handle and its attributes (`$root`, `$root.causal_mask`). Tokens without a
`$` are literals: numbers, enum values (`"BFLOAT16"`, `"gfx942"`), attribute-name keys, layout aliases,
opcodes, and keywords (`device`, `one_of`, `any`). The same token then reads identically wherever it
appears, in a `shape` list, an `expr` string, or a native-predicate argument.

Authors therefore get a complete, uniform symbol table for free and never hand-declare each field. Two
optional forms add friendly names and cross-tensor unification on top:

- **Named shape dims.** A `shape` list names a tensor's dims: `"shape": ["$batch", "$num_heads",
  "$seqlen_q", "$head_size"]` binds those four symbols to `$q`'s dims. Reusing a name across tensors
  **unifies** it: writing `$head_size` on both `$q` and `$k` requires `$q3 == $k3` and binds one symbol,
  replacing the hand-coded `k.dims[1] == v.dims[1]` relation. A dim position may be left anonymous with
  `"_"`.
- **Named attributes.** Rarely needed, since `$root.<attr>` already reads them, but a constraint may
  alias one for reuse in formulas.

Symbol names are unique within a UMD across tensors, nodes, and named dims; a match/dispatch pair is
validated so that unification is consistent (a name bound to two different concrete values fails the
match, not the load).

![A live graph matched against a declarative pattern, auto-binding tensors, dims, strides, and attributes](../images/umd_symbol_binding.svg)

---

## 6. Constraint Vocabulary

Constraints attach to a target (`on`) and implicitly AND together. The target is a bound tensor
variable (`$q`), a pattern node id (`root`), or the special `device`. The vocabulary is fixed to what
the hand-written checks of [§2](#2-what-the-umd-replaces) express, so no check needs code that a
constraint cannot state (the residue goes to a native predicate, [§9](#9-native-predicate-escape-hatch)).

| Constraint | Form | Lowers from |
|---|---|---|
| **Opcode** | node `op`: exact, `one_of`, `any` | node attribute-type gate |
| **Dtype (exact / set)** | `{"on": "$q", "dtype": {"one_of": ["BFLOAT16", "FP8_E4M3"]}}` | `validateDataTypeIsSupported`, `validateFixedDataType` |
| **Dtype (relation)** | `{"on": "$k", "dtype": {"same_as": "$q"}}` | `validateConsistentDataTypes`, `q == k == v` |
| **Rank** | `{"on": "$q", "rank": 4}` | `validateDimensionCount`, rank == 4 |
| **Shape (exact / symbolic)** | `{"on": "$q", "shape": [1, "$num_heads", "_", "$head_size"]}` (int = exact, name = bind/unify, `_` = any) | dim reads and cross-tensor dim relations |
| **Layout** | `{"on": "$q", "layout": [0, 1, 2, 3]}` (stride-order index array, [§8](#8-layout-and-stride-order-constraints)) | `validateSupportedLayout` |
| **Packing** | `{"on": "$q", "packed": true}` | `validatePackedTensors` |
| **Cross-tensor layout** | `{"consistent_layout": ["$x", "$y", "$scale"]}` | `validateConsistentLayouts` |
| **Attribute (value)** | `{"on": "$root", "attr": {"causal_mask": {"equals": false}, "dropout_probability": {"absent_or": {"equals": 0.0}}}}` | per-attr value gates |
| **Attribute (one_of / relation)** | `{"on": "$root", "attr": {"head_dim": {"one_of": [64, 128, 192]}}}` | `head_dim in {...}` |
| **Optional operand present/absent** | `{"on": "$attn_mask", "present": false}` | `attn_mask_tensor_uid()` absent gate |
| **Virtual tensor** | `{"on": "$conv_out", "virtual": true}` | fusion tensor-virtuality gates |
| **Use-count / exclusivity** | `{"on": "$conv_out", "use_count": 1}` or `{"on": "$bias_out", "no_external_consumer": true}` | fusion legality checks |
| **Cross-tensor / expression** | `{"expr": "$q1 == $k2"}`, `{"expr": "$q0 <= 128"}` ([§7](#7-the-shared-expression-language)) | arithmetic and comparison gates |
| **Device / arch** | `{"on": "device", "arch": {"one_of": ["gfx942", "gfx950"]}}` | `getDeviceString` arch gate |
| **Native predicate** | `{"kind": "native_predicate", "name": "hipdnn.strides_fit_u32", "args": ["$q", "$k", "$v", "$o"]}` ([§9](#9-native-predicate-escape-hatch)) | overflow guards, table lookups, derived-shape relations |

The `device` constraint gates *applicability* at match time; the per-architecture `kpack` manifest
([RFC 0017 §10](0017_UniversalKernelDescriptor.md#10-packaging-and-delivery)) gates *loadability* at
install and load. Both must agree for a kernel to run.

---

## 7. The Shared Expression Language

The UMD needs boolean predicates over bound values (`$q1 == $k2`, `$q0 <= 128`, `$q0 one_of {64,128,256}`,
`$nhead_q % $nhead_k == 0`). The UDD needs integer-valued formulas over the same bound values for grid,
block, shared memory, and workspace (`ceil_div($seqlen_q, 16)`,
[RFC 0017 §6](0017_UniversalKernelDescriptor.md#6-dispatch-and-workspace)). These are the same language
at two layers, and this RFC defines it once so the two subsystems cannot drift.

**Layering.**

- **Value core (shared with the UDD).** Literals (int, float, string/enum, array), symbol references,
  tensor and attribute accessors, and arithmetic. The value core is exactly what the UDD RFC consumes;
  that RFC references this section rather than redefining it.
- **Predicate layer (UMD-only).** Comparison, set membership, and, in a later iteration, boolean
  composition ([§10](#10-composite-constraints-deferred-forward-compatible)) over value-core
  expressions, producing a boolean. A UMD constraint's `expr` is a predicate-layer expression.

**Canonical AST.** The stored form is the nested object form already used in RFC 0017 §6, so the two
subsystems share one parser, one validator, and one interpreter:

```jsonc
// value core
{"op": "ceil_div", "args": [{"sym": "$seqlen_q"}, 16]}
{"op": "mod", "args": [{"dim": "$q", "axis": 1}, {"dim": "$k", "axis": 1}]}

// predicate layer (UMD)
{"op": "eq", "args": [{"dim": "$q", "axis": 1}, {"dim": "$k", "axis": 2}]}   // $q1 == $k2
{"op": "one_of", "args": [{"dim": "$q", "axis": 0}, [64, 128, 256]]}          // $q0 in {64,128,256}
```

**Authoring sugar.** A UMD `expr` may be written as a light infix string that compiles to the AST, so
the AICK-1698 examples are literal:

```
$q0 == 64
$q1 == $k2
$w1 == $x1
$q0 <= 128
$q0 one_of {64, 128, 256}
stride_order($q) == [0, 1, 2, 3]
$nhead_q % $nhead_k == 0
```

In a UMD these are `expr` constraints ([§6](#6-constraint-vocabulary)); the infix string and the AST
form are interchangeable and compile to the same tree:

```jsonc
"constraints": [
  {"expr": "$q0 == 64"},                          // dim equality
  {"expr": "$q1 == $k2"},                         // cross-tensor dim relation
  {"expr": "$q0 <= 128"},                         // range bound
  {"expr": "$q0 one_of {64, 128, 256}"},          // set membership
  {"expr": "$nhead_q % $nhead_k == 0"},           // divisibility
  {"expr": "stride_order($q) == [0, 1, 2, 3]"},   // layout via stride order

  // the infix string is sugar; the equivalent AST form is accepted too
  {"expr": {"op": "eq", "args": [{"dim": "$q", "axis": 1}, {"dim": "$k", "axis": 2}]}}  // == "$q1 == $k2"
]
```

Grammar (EBNF, value core plus predicate layer):

```
predicate   = comparison | membership
comparison  = value ("==" | "!=" | "<" | "<=" | ">" | ">=") value
membership  = value "one_of" "{" value ("," value)* "}"
value       = term (("+" | "-" | "*" | "/" | "%") term)*
term        = number | symbol | call | array | "(" value ")"
symbol      = "$" ident ("." ident | "[" number "]")*      // $q0, $q.dims[2], $root.causal_mask
call        = ident "(" (value ("," value)*)? ")"          // ceil_div($seqlen_q, 16), stride_order($q)
array       = "[" (value ("," value)*)? "]"
```

**Operators.** Value core: `+ - * / %`, `ceil_div`, `min`, `max`, `abs`, `pow`, `log2`, `rsqrt`.
Predicate layer: `== != < <= > >=`, `one_of`. Booleans combine only by the implicit top-level AND in
this iteration; explicit `and`/`or`/`not` are reserved for composite constraints
([§10](#10-composite-constraints-deferred-forward-compatible)).

**Evaluation is a safe, bounded interpreter.** It fails closed on an unknown symbol, an out-of-range
axis, a type error, or an invalid operation, and never executes arbitrary code. Integer arithmetic
uses checked-width integers and fails closed on overflow rather than wrapping
([§15](#15-security-and-hostile-input)). The interpreter is bounded in recursion depth and step count.
Because the language is dependency-free and hand-written (no third-party parser), it is small enough to
audit and to lower into the static matcher ([§12](#12-static-matcher-sketch)).

---

## 8. Layout and Stride-Order Constraints

hipDNN tensors store no layout enum; layout is implied by stride order
([§3](#3-the-matchers-input-hipdnns-graph-model)). The UMD therefore represents layout as an **array of
dimension indexes giving the stride order**, from the slowest-varying dimension to the
fastest-varying. This is the shape a matcher can check directly against `strides()` and matches how
`TensorDescriptor` already precomputes `strideOrder` (`ApplicabilityChecks.cpp:17`).

```jsonc
{"on": "$q", "layout": [0, 1, 2, 3]}   // fully packed, natural order (BHSD for a rank-4 SDPA tensor)
{"on": "$x", "layout": [0, 3, 1, 2]}   // NHWC over an NCHW logical dim order
```

- The array is a permutation of `0..rank-1`. Entry `k` names the logical dimension that occupies stride
  position `k`, so `[0,1,2,3]` is descending-stride packed and `[0,3,1,2]` places the channel dim last
  (NHWC). The `axis` used everywhere else (dims, strides, `args_signature`) indexes the logical
  dimension order, independent of this physical layout, consistent with RFC 0017 §6.
- **Named aliases** are provided for the common cases and expand to arrays at compile time:
  `"nchw" -> [0,1,2,3]`, `"nhwc" -> [0,3,1,2]`, `"ncdhw"`, `"ndhwc"`, `"bhsd" -> [0,1,2,3]`,
  `"contiguous"` (identity permutation for the tensor's rank). Aliases keep authoring readable while
  the array remains the single canonical form. The set of aliases matches the layouts
  `validateSupportedLayout` accepts today (`ApplicabilityChecks.cpp:77`).
- **Cross-tensor consistency** uses `consistent_layout` ([§6](#6-constraint-vocabulary)), lowering
  `validateConsistentLayouts`; layout-agnostic tensors (rank-1 scalars, pass-by-value) are skipped as
  they are today.
- **Packing** is a separate `packed` constraint, since a supported stride order does not imply the
  tensor is gap-free; it lowers `validatePackedTensors`.
- The stride-order array is also available to expressions as `stride_order($q)`, so
  `stride_order(q) == [0,1,2,3]` from AICK-1698 is expressible directly.

---

## 9. Native-Predicate Escape Hatch

Some checks cannot be stated declaratively: they need real C++. A UMD constraint may name a **native
predicate** resolved from a provider-internal registry.

```jsonc
{"kind": "native_predicate", "name": "hipdnn.strides_fit_u32", "args": ["$q", "$k", "$v", "$o"], "negated": false}
```

The descriptor carries only a symbol name and a typed argument list drawn from bound variables, never
inline code. Predicates take explicit arguments, not the whole graph, so they stay auditable and
reusable. Both the build-time and drop-in paths resolve predicates from the same registry: a file that
names only shipped predicates loads identically either way, and a file naming a predicate the running
provider does not ship fails to resolve on the drop-in path. The registry a provider ships is therefore
part of its published contract.

The grounded cases that need this hatch, from [§2](#2-what-the-umd-replaces):

- **Integer-overflow guards.** `wouldFwdByteStridesFitUint32` / `byteStrideFitsU32`
  (`SdpaFwdPlanBuilder.cpp:294`, `SdpaPlanUtils.hpp:159`): the kernarg struct stores byte strides as
  `uint32`, so the check must be exact and fail closed. A predicate `hipdnn.strides_fit_u32`.
- **Derived-quantity relations.** NumPy-broadcast affine-shape compatibility
  (`BatchnormApplicabilityChecks.cpp:169`), layernorm normalized-dim reconciliation
  (`LayernormApplicabilityChecks.cpp:68`), and RMSnorm `inv_rms` derived shape
  (`RMSnormApplicabilityChecks.cpp:106`) each compute a shape and compare it, beyond the constraint
  vocabulary.
- **Contradiction checks.** `getMaskType` throws when mask attributes contradict
  (`SdpaFwdPlanBuilder.cpp:276`); a predicate encodes "the mask attributes are self-consistent".
- **GQA divisibility.** `nhead_q % nhead_k == 0 && nhead_k != 0` (`SdpaBwdPlanBuilder.cpp:548`) is
  expressible with `%` in [§7](#7-the-shared-expression-language), but a native predicate is an option
  if the zero-guard and fail-closed semantics are better centralized.

**Kernel-table lookups are a migration artifact, not a lasting predicate.** The
`getKernelNameKey` / CSV-registry lookups (`SdpaFwdPlanBuilder.cpp:287`, three in
`SdpaBwdPlanBuilder.cpp:660`) exist because today the builder resolves which prebuilt code object serves
a shape. Under the UKD model the KDP names the code object directly and the heuristic ranks candidates,
so these lookups mostly dissolve into ordinary constraints plus the Launch's kernel source. Where a
residual "is there a row for this exact combination" gate remains during coexistence, it is a native
predicate.

Together with the UDD's custom-plan hatch
([RFC 0017 §6](0017_UniversalKernelDescriptor.md#6-dispatch-and-workspace)), these form the graded
ladder: fully declarative constraints, then a named predicate for one step that needs C++, then a full
provider.

---

## 10. Composite Constraints (Deferred, Forward-Compatible)

All constraints AND together, which covers every hand-written check surveyed
([§2](#2-what-the-umd-replaces)). AICK-1698 anticipates a later need for a single constraint that is a
boolean combination, for example `(A AND B) OR C`. This iteration does not evaluate boolean composition
beyond the top-level AND, but reserves a forward-compatible shape so adding it is additive and does not
change existing descriptors:

```jsonc
// reserved shape, not evaluated in v1
{"any_of": [                                  // OR
  {"all_of": [{"expr": "q0 == 64"}, {"expr": "q1 == k2"}]},   // (A AND B)
  {"expr": "q0 == 128"}                        // OR C
]}
```

A v1 matcher rejects a UMD that uses `any_of` / `all_of` with a clear "composite constraints not
supported by this runtime version" error ([§14](#14-serialization-and-versioning)), rather than
silently ignoring it. The predicate-layer grammar ([§7](#7-the-shared-expression-language)) already
reserves `and` / `or` / `not` for the same reason. General N-ary commutative matching and unbounded
chains remain deferred to the JIT follow-up, as in RFC 0017 §5.

---

## 11. The Matcher: Compilation, Indexing, and Caching

A UMD is authored as text and **compiled once** into an in-memory matcher structure at provider load
(or, for the drop-in path, when the bundle is scanned). Compilation resolves op-schema roles, expands
layout aliases, parses expressions to AST, validates that every referenced symbol is bound, and orders
the constraints for early-out. The compiled form, not the text, is what runs against live graphs.

**Root-opcode indexing.** The compiled matchers are indexed by the root node's opcode, so match cost
does not grow linearly with the number of descriptors: a graph whose root op is `sdpa_fwd` only
consults UMDs rooted at `sdpa_fwd`. This is the index RFC 0017 §14 calls for. Per-candidate cost
(constraint, predicate, and expression evaluation) is separate and is bounded by the ordering below,
not by the index.

**Prioritized, early-out constraint ordering (AICK-1698).** Within a candidate, constraints run
cheapest-and-most-selective first so a non-match is rejected as early as possible. The compiler assigns
a static cost/selectivity order, roughly:

1. Node count and opcode (already narrowed by the index, but confirmed).
2. Attribute equality and `one_of` (scalar reads).
3. Rank and dtype (single tensor-field reads).
4. Shape, symbolic-dim unification, and layout/stride-order (vector reads).
5. Cross-tensor relations and expressions.
6. Native predicates (opaque, potentially the most expensive), last.

Authors do not order constraints; the compiler does, from the constraint kind. The relative order is a
matcher-internal detail, not part of the descriptor contract, so it can be tuned without a schema
change.

**Per-plan caching (AICK-1698).** Matching runs at plan-build time. The result (the chosen UMD, the
bound symbol table, and the arbitration outcome) is cached on the compiled plan and reused for
workspace queries and execution, so the same graph is not re-matched across the
`isApplicable` / `getMaxWorkspaceSize` / `buildPlan` calls that re-run the loop today
(`AsmSdpaEngine.cpp:66,87`). The compiled matcher itself is built once and shared across plans; only
the per-graph binding result is per-plan.

**Device gating short-circuits.** The `device.arch` constraint is evaluated once per graph, before
per-candidate work, since arch is constant for the stream.

![Compile-once pipeline: text UMD to constraint IR to a root-opcode-indexed matcher, with a per-plan bind cache](../images/umd_matcher_pipeline.svg)

---

## 12. Static Matcher (Sketch)

AICK-1698 asks whether a UMD can be pre-compiled into a static matcher that further cuts the runtime
cost, while still supporting runtime (drop-in) matchers. This iteration does not commit to a design; it
records the options and the constraint they must satisfy.

**The parity constraint.** However a static matcher is produced, it must be behaviorally identical to
the runtime matcher on the same UMD and graph. Build-time and drop-in descriptors run through one
generic engine ([RFC 0017 §3](0017_UniversalKernelDescriptor.md#3-how-it-works)), so a kernel that is
AOT-packed today and dropped in tomorrow must match the same graphs either way. Parity is testable as a
cross-path equivalence check ([§17](#17-testing-and-performance)).

Options, from least to most build coupling:

- **Interpreted compiled IR (baseline).** The [§11](#11-the-matcher-compilation-indexing-and-caching)
  compiled form, interpreted. No codegen; identical on both paths by construction. This is the
  fallback and the parity oracle.
- **Bytecode / flattened decision program.** Lower the constraint IR to a compact bytecode (a linear
  program of typed ops over the bound symbol table) that a tiny VM executes. Serializable into the
  KDP, so drop-in gets the same artifact AOT does. Faster than tree-walking, still data.
- **Generated C++ predicate, AOT only.** For build-time descriptors, emit a specialized C++
  applicability function per UMD and compile it into the provider (or a packed object). Closest to a
  hand-written `isApplicable`, but unavailable to pure drop-in descriptors, which fall back to the
  interpreted or bytecode path. Needs the generated function proven equivalent to the interpreter.
- **Shared decision tree across UMDs.** Combine many UMDs rooted at the same opcode into one decision
  tree that tests shared constraints once (a discrimination net). An optimization layered on any of the
  above; it changes throughput, not per-UMD semantics.

Recommendation for a later iteration: make the interpreted compiled IR the contract and the parity
oracle, add the bytecode form as the shared AOT/drop-in fast path, and treat generated C++ and the
shared decision tree as opportunistic optimizations gated behind the parity test. The concrete choice
is deferred.

---

## 13. Arbitration

Today the first applicable plan builder wins, which is a documented latent bug when more than one
matches (`HipMlopsEngine.cpp:34`). The UMD makes overlap explicit and resolves it deterministically,
reusing the rule from RFC 0017 §5:

1. When several UKDs match a graph, the **UHD** (kernel-selection heuristic) ranks them and the
   top-scored kernel wins.
2. Ties break by explicit **`priority`** on the UKD.
3. Remaining ties break by the descriptor's stable **`id`**, and the conflict is logged to the warning
   log so an unintended overlap is visible.

Arbitration is a property of the generic engine over the set of matching UKDs; a UMD shared by several
UKDs contributes each of them as a candidate. This closes the mutual-exclusion-by-construction
requirement that the current engines depend on: overlap is allowed and resolved, not a correctness
hazard.

---

## 14. Serialization and Versioning

- **Authoring form.** Human-readable, diffable JSONC (the examples here), including the infix `expr`
  sugar of [§7](#7-the-shared-expression-language).
- **Compiled form.** The compact binary the matcher runs ([§11](#11-the-matcher-compilation-indexing-and-caching)),
  whose concrete bytes are defined with the KDP/packaging follow-up
  ([RFC 0017 §12.2](0017_UniversalKernelDescriptor.md#122-follow-up-rfcs)); this RFC defines the schema
  those bytes encode.
- **Schema and version.** Every UMD carries `schema: "hipdnn.umd/v1"`, a stable opaque `id`, and a
  mandatory `name` for diagnostics. A UMD whose schema version is newer than the runtime understands is
  refused with a clear error, never silently reinterpreted, matching
  [RFC 0017 §4](0017_UniversalKernelDescriptor.md#4-descriptor-formats).
- **Additive evolution.** New constraint kinds, operators, layout aliases, and the composite forms of
  [§10](#10-composite-constraints-deferred-forward-compatible) are additive within `v1` where they do
  not change the meaning of an existing descriptor; anything that would reinterpret existing fields
  bumps the version.
- **Identity.** UMD ids are unique within the UMD kind; a match id and an engine id may collide as
  strings because references are typed by field (a UKD's `match` versus `engine`). A colliding drop-in
  id is logged and ignored rather than taking down the provider ([RFC 0017 §14](0017_UniversalKernelDescriptor.md#14-risks)).

---

## 15. Security and Hostile Input

On the drop-in path the loader, the matcher, and the expression interpreter parse input that may be
untrusted or simply malformed, so they must be bounded and fail closed rather than crash
([RFC 0017 §14](0017_UniversalKernelDescriptor.md#14-risks)).

- **Bounded parsing and matching.** Recursion depth, expression step count, node/constraint counts, and
  descriptor size are capped; exceeding a cap quarantines the descriptor, it does not abort the
  provider.
- **Checked arithmetic.** Shape, stride, and workspace arithmetic uses checked-width integers and fails
  closed on overflow rather than under-allocating or wrapping. This is the same class of bug the
  `strides_fit_u32` predicate guards ([§9](#9-native-predicate-escape-hatch)).
- **Fail-closed evaluation.** An unknown symbol, unresolved native predicate, out-of-range axis, or
  type error declines the match; it never matches by default.
- **Quarantine, not cascade.** A bad descriptor is quarantined on load with a diagnostic; the rest load
  ([RFC 0017 §10](0017_UniversalKernelDescriptor.md#10-packaging-and-delivery)).
- **Fuzzing.** A seed corpus of UMDs and graphs plus a fuzzer over the loader, matcher, and interpreter
  run under the existing ASAN build ([§17](#17-testing-and-performance)), backing the fail-closed
  requirement.

---

## 16. Observability and Diagnostics

Because matching is data-driven, it is inspectable, and the tooling is a first-class deliverable
([RFC 0017 §9](0017_UniversalKernelDescriptor.md#9-observability-and-diagnostics)). For the UMD the
provider surfaces:

- **A why-not trace.** For a graph and a candidate UMD, the first constraint that failed and why (the
  concrete values compared), so an author can see exactly which gate declined.
- **A binding view.** For a successful match, the full bound symbol table (tensors, dims, strides,
  attributes) as the UDD will see it.
- **An arbitration trace.** Which UKDs matched, how the UHD scored them, and where a tie fell to
  `priority` or stable `id` ([§13](#13-arbitration)).
- **Load diagnostics.** Which UMDs compiled, which were quarantined and why, and unresolved native
  predicates by name.

These reuse the diagnostic surface RFC 0017 §9 defines rather than adding a UMD-specific one.

---

## 17. Testing and Performance

The UMD introduces no new testing strategy; it slots into hipDNN's existing tiers (`docs/Testing.md`,
`docs/testing/TestingStrategy.md`) as RFC 0017 §12.1 requires. A UMD-backed kernel runs through the
generic engine as an ordinary engine and produces the same graphs everything else consumes, so the
plugin-agnostic integration harness ([RFC 0006](0006_PluginAgnosticIntegrationTests.md)) validates it
against the CPU reference ([RFC 0001](0001_CpuGraphExecutorDesign.md)) with the golden-reference
tolerance chain ([RFC 0011](0011_GoldenReferenceValidation.md)).

UMD-specific coverage:

- **Match-equivalence against hand-written `isApplicable`.** For each converted engine, a test drives a
  battery of graphs (accepting and rejecting) through both the hand-written builder and the UMD and
  asserts identical accept/reject decisions and identical bound values. The SDPA-forward builder
  ([§19](#19-worked-example-sdpa-forward)) is the first target.
- **Static/runtime parity.** The parity oracle of [§12](#12-static-matcher-sketch): the same UMD and
  graph must decide identically on the interpreted and any lowered matcher.
- **Expression-language conformance.** A table-driven suite over the value core and predicate layer,
  including the AICK-1698 examples and the fail-closed cases (overflow, unknown symbol, bad axis),
  shared with the UDD RFC's expression tests since the language is shared.
- **Fuzzing.** The corpus and fuzzer of [§15](#15-security-and-hostile-input).
- **Match overhead.** Plan-time match cost is measured against the hand-written baseline as
  benchmarking matures (`tools/dnn-benchmarking`, [RFC 0013](0013_Autotune.md)); the compiled matcher,
  root-opcode index, and per-plan cache ([§11](#11-the-matcher-compilation-indexing-and-caching)) keep
  it minimal, and the cost is paid once at plan build.

---

## 18. Migration

Migration follows RFC 0017 §12: no engine is converted until a UMD-backed kernel runs end to end, and a
hand-written engine and its descriptor-backed replacement coexist until the generic one reaches parity
on the graphs that engine covers, at which point the hand-written code is retired.

The **SDPA-forward** `isApplicable` (`SdpaFwdPlanBuilder.cpp:167`) is the first conversion, because it
exercises nearly the whole vocabulary (opcode, attribute gates, optional-operand absence, rank, dtype
relations, symbolic-dim unification, and two native predicates) in one node. Its match-equivalence test
([§17](#17-testing-and-performance)) gates the cutover. The mlops builders follow, reusing the
`IValidator` primitives ([§2](#2-what-the-umd-replaces)) as the reference for their constraint
lowering. The kernel-table lookups dissolve into the KDP as described in
[§9](#9-native-predicate-escape-hatch).

---

## 19. Worked Example: SDPA Forward

The SDPA-forward check collapses into one UMD. Compared to the hand-written builder
(`SdpaFwdPlanBuilder.cpp:167-296`), each C++ gate becomes a field, and only the two genuinely
non-declarative gates (uint32 stride fit, mask self-consistency) remain as native predicates. Note
`$head_size` is bound from `$q`'s dim, not read as an attribute
([§3](#3-the-matchers-input-hipdnns-graph-model)).

```jsonc
{
  "schema": "hipdnn.umd/v1",
  "id":   1180449020,
  "name": "SDPA forward (d128, bf16/fp8) match",
  "nodes": [
    {"kind": "op", "id": "$root", "op": "sdpa_fwd",
     "operands": {"Q": "$q", "K": "$k", "V": "$v"}, "results": {"O": "$o"}}
  ],
  "constraints": [
    {"on": "device", "arch": {"one_of": ["gfx942", "gfx950"]}},   // getDeviceString gate

    // dtype: $q == $k == $v (relation), $o supported (set)
    {"on": "$q", "dtype": {"one_of": ["BFLOAT16", "FP8_E4M3"]}},
    {"on": "$k", "dtype": {"same_as": "$q"}},
    {"on": "$v", "dtype": {"same_as": "$q"}},

    // rank 4 and dim binding; head_size and head count unify across q/k/v
    {"on": "$q", "shape": ["$batch", "$num_heads", "$seqlen_q", "$head_size"]},
    {"on": "$k", "shape": ["$batch", "$kv_heads",  "$seqlen_k", "$head_size"]},
    {"on": "$v", "shape": ["$batch", "$kv_heads",  "$seqlen_k", "$head_size"]},  // $k1 == $v1 via $kv_heads
    {"on": "$o", "rank": 4},
    {"expr": "$head_size == 128"},

    // attribute gates: no dropout, no alibi/padding mask, no stats output
    {"on": "$root", "attr": {
      "dropout_probability": {"absent_or": {"equals": 0.0}},
      "alibi_mask":   {"equals": false},
      "padding_mask": {"equals": false},
      "generate_stats": {"absent_or": {"equals": false}}
    }},

    // optional operands that this kernel does not support must be absent
    {"on": "$attn_mask",   "present": false},
    {"on": "$page_table_k","present": false},
    {"on": "$page_table_v","present": false},

    // escape hatches: the two checks that need real C++
    {"kind": "native_predicate", "name": "hipdnn.sdpa_mask_consistent", "args": ["$root"]},
    {"kind": "native_predicate", "name": "hipdnn.strides_fit_u32",      "args": ["$q", "$k", "$v", "$o"]}
  ]
}
```

Mapping to the hand-written code:

| Hand-written (`SdpaFwdPlanBuilder.cpp`) | UMD field |
|---|---|
| `getDeviceString` gfx942/gfx950 (:186) | `device.arch` |
| `nodeWrappers().size() != 1` (:199) | single-node pattern |
| `attributesType() != SdpaAttributes` (:200) | node `op: sdpa_fwd` |
| dropout / alibi / padding / stats gates (:205-224) | `attr` constraints |
| `attn_mask` / `page_table_*` absent (:209-215) | optional-operand `present: false` |
| rank == 4 (:231-247) | `shape` (rank 4) |
| `q == k == v` dtype (:244) | `dtype.same_as` |
| `k.dims[1] == v.dims[1]` head count (:251) | `$kv_heads` shape unification |
| head dim == 128 | `expr: $head_size == 128` |
| `getMaskType` throw-on-contradiction (:276) | `hipdnn.sdpa_mask_consistent` predicate |
| `wouldFwdByteStridesFitUint32` (:294) | `hipdnn.strides_fit_u32` predicate |
| `getKernelNameKey` table lookup (:287) | dissolves into the KDP's Launch ([§9](#9-native-predicate-escape-hatch)) |

The bound symbols (`$q..$o`, `$batch`, `$num_heads`, `$kv_heads`, `$seqlen_q`, `$seqlen_k`, `$head_size`, and
every auto-bound dim/stride) are exactly what the paired UDD's grid and argument formulas reference
([RFC 0017 §6](0017_UniversalKernelDescriptor.md#6-dispatch-and-workspace)).

---

## 20. Risks

- **Op-schema registry coupling.** Auto-binding depends on a registry generated from the flatbuffer op
  schema ([§3](#3-the-matchers-input-hipdnns-graph-model)). If it drifts from the graph definitions,
  bindings are wrong. Mitigation: generate it from the same schema, and fail closed on an unknown op or
  role rather than binding a wrong field.
- **Expression language sharing.** The value core is shared with the UDD ([§7](#7-the-shared-expression-language)).
  A change made for one subsystem can affect the other. Mitigation: one parser/validator/interpreter,
  a shared conformance suite, and a clear layer split (UDD uses the value core only).
- **Predicate registry as contract.** Native predicates are part of the published provider contract
  ([§9](#9-native-predicate-escape-hatch)); a drop-in naming an unshipped predicate fails to resolve.
  Mitigation: version and document the shipped predicate set; fail closed with a clear diagnostic.
- **Match overhead.** Per-candidate constraint and predicate evaluation is unbounded by the root-opcode
  index ([§11](#11-the-matcher-compilation-indexing-and-caching)). Mitigation: early-out ordering,
  per-plan caching, and the overhead test of [§17](#17-testing-and-performance).
- **Static-matcher parity.** A lowered matcher that diverges from the interpreter is a silent
  correctness bug ([§12](#12-static-matcher-sketch)). Mitigation: the interpreter is the oracle and the
  parity test gates any lowering.

---

## 21. Open Questions

1. **Expression language home.** This RFC defines the shared value core; the UDD RFC references it.
   Confirm this split with the UDD RFC rather than each defining its own, and decide where the shared
   conformance suite lives.
2. **GQA divisibility.** Express `nhead_q % nhead_k == 0` with the `%` operator
   ([§7](#7-the-shared-expression-language)) or centralize it as a native predicate for uniform
   fail-closed zero-guarding ([§9](#9-native-predicate-escape-hatch))?
3. **Static-matcher form.** Which of the [§12](#12-static-matcher-sketch) options becomes the AOT fast
   path, and does it also serve drop-in via a serialized bytecode?
4. **Feature-vector overlap.** The bound symbol table overlaps the feature vector a UHD consumes
   ([RFC 0017 §15 Q4](0017_UniversalKernelDescriptor.md#15-open-questions)); should the UMD's bindings
   be the canonical feature source for kernel selection?
5. **Composite timing.** When composite constraints ([§10](#10-composite-constraints-deferred-forward-compatible))
   are needed, do they land before or after the first engines are migrated?

---

## 22. References and Prior Art

The design borrows established ideas; none is a dependency. These informed the UMD specifically.

| System | Idea borrowed |
|---|---|
| **MLIR PDL / PDLL** | Two-layer design: a declarative pattern compiled once to a fast matcher; constraints inline on the binding; a named native-predicate escape hatch; pattern priority for arbitration |
| **TVM Relax DFPattern** | Constraint vocabulary (op, dtype, symbolic shape, wildcard); dataflow use-def constraints; cross-tensor same-shape relations |
| **XLA pattern matcher** | Exact-vs-compatible equality; use-count versus user-count; layout as a distinct constraint; optional operands; capture-by-reference binding |
| **PyTorch Inductor / torch.library** | Node/edge pattern vocabulary; serialized precompiled patterns; duplicate-pattern detection |
| **LLVM ISel / discrimination nets** | Sharing common prefixes of many patterns rooted at one opcode into one decision structure ([§12](#12-static-matcher-sketch)) |
| **ONNX Runtime** | First-claim arbitration as the anti-pattern this RFC replaces with deterministic ranking; single-node versus fused-subgraph capability |

---

## 23. Glossary

- **UMD (Universal Match Descriptor):** the declarative pattern and constraints that decide whether a
  kernel applies to a graph and bind the variables its dispatch and workspace formulas use. Reused
  across kernels by ID.
- **Structural pattern:** the op nodes and the named operand/result edges of a UMD; edges are implicit
  through shared pattern variables ([§4](#4-structural-pattern)).
- **Constraint:** a condition attached to a bound tensor, a node, or `device`; constraints implicitly
  AND ([§6](#6-constraint-vocabulary)).
- **Symbol lifecycle:** a name is declared in the UMD, bound when the graph matches, and used by the UDD
  ([§5](#5-symbol-binding-and-the-auto-binding-formula)).
- **Auto-binding formula:** the standard scheme that binds every operand/result tensor, its dims and
  strides, and every op attribute of a matched node, without hand-declaration
  ([§5](#5-symbol-binding-and-the-auto-binding-formula)).
- **Op-schema registry:** the generated table mapping each op type to its operand/result UID fields and
  attributes, letting the matcher reconstruct edges and auto-bind
  ([§3](#3-the-matchers-input-hipdnns-graph-model)).
- **Value core / predicate layer:** the two layers of the shared expression language; the value core
  (arithmetic, shared with the UDD) yields numbers, the predicate layer (comparison, membership) yields
  booleans ([§7](#7-the-shared-expression-language)).
- **Stride-order layout:** layout represented as an array of dimension indexes giving stride order,
  since tensors carry no layout enum ([§8](#8-layout-and-stride-order-constraints)).
- **Native predicate:** the escape hatch; a registry-resolved check a UMD names for logic it cannot
  state declaratively, carried as a symbol name and typed arguments, never inline code
  ([§9](#9-native-predicate-escape-hatch)).
- **Composite constraint:** a single constraint that is a boolean combination of sub-constraints;
  schema reserved, evaluation deferred ([§10](#10-composite-constraints-deferred-forward-compatible)).
- **Arbitration:** the deterministic resolution when several UKDs match: UHD score, then `priority`,
  then stable `id` ([§13](#13-arbitration)).
- **Root-opcode index:** the index of compiled matchers by root opcode that keeps match cost sublinear
  in descriptor count ([§11](#11-the-matcher-compilation-indexing-and-caching)).
