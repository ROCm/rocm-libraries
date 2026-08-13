# RFC 0018: The Descriptor Expression Language

- Contributors: Brian Harrison

> Base layer of the UKD descriptor series ([RFC 0017 (Universal Kernel Descriptors)](0017_UniversalKernelDescriptor.md),
> the follow-up series of [RFC 0017 §14.2](0017_UniversalKernelDescriptor.md#142-follow-up-rfcs)).
> This RFC specifies the expression language every descriptor format embeds: its grammar, type
> system, operator set, evaluation semantics, and the bounded interpreter that runs it. It defines
> the language over an **abstract binding environment** and depends on no other follow-up. The
> consuming formats — the matcher ([RFC 0019](0019_UniversalMatchDescriptor.md)), the UDD, and the
> UHD — each supply their own environment and are referenced, not redesigned, here. The RFC number
> is provisional and is reconciled against the concurrent follow-up series at PR-open time.

## Table of Contents

1. [Overview](#1-overview)
2. [Two Uses, One Language](#2-two-uses-one-language)
3. [Syntax](#3-syntax)
4. [The Binding Environment](#4-the-binding-environment)
5. [The Type System](#5-the-type-system)
6. [Operators](#6-operators)
7. [Unknown Values and Three-Valued Logic](#7-unknown-values-and-three-valued-logic)
8. [Evaluation Semantics](#8-evaluation-semantics)
9. [The Interpreter: Safety and Bounds](#9-the-interpreter-safety-and-bounds)
10. [Compilation and Lowering](#10-compilation-and-lowering)
11. [Versioning and Evolution](#11-versioning-and-evolution)
12. [Diagnostics and Evaluation Traces](#12-diagnostics-and-evaluation-traces)
13. [Conformance and Testing](#13-conformance-and-testing)
14. [Risks](#14-risks)
15. [Open Questions](#15-open-questions)
16. [References and Prior Art](#16-references-and-prior-art)
17. [Glossary](#17-glossary)
18. [Appendix A: Normative Reference](#appendix-a-normative-reference)

---

## 1. Overview

Three subsystems of the UKD series need to evaluate a small expression over facts about a problem,
and none of them should get to invent its own way of doing it. A matcher decides whether a kernel
applies. A dispatch descriptor computes a grid, a block, a shared-memory size, a workspace size, and
individual kernel arguments. A heuristic derives the feature vector it ranks on. Those are the same
computation with different root types, so this RFC specifies **one dialect, one parser, one
validator, one interpreter** and lets each subsystem bring only its own facts.

The consumers are therefore fixed, and the language is not negotiated per consumer:

- **[RFC 0019](0019_UniversalMatchDescriptor.md)'s UMD criteria** — a boolean expression that decides
  applicability.
- **The UDD follow-up's dispatch and workspace formulas** — value expressions for grid, block, shared
  memory, workspace, and the `expr` form of an argument source
  ([RFC 0017 §6](0017_UniversalKernelDescriptor.md#6-dispatch-and-workspace)).
- **The UHD follow-up's `features_signature`** — value expressions producing a heuristic's derived
  features.

**Descriptors stay pure data.** An expression is JSON, it names no code, and it can be loaded,
diffed, validated, and reasoned about without linking anything. That property is what makes the
drop-in path viable at all: a descriptor is fully interpretable from the schema alone, so a file
authored outside the build tree loads on exactly the same terms as one packaged with it.

**This RFC defines the language over an abstract binding environment and never enumerates hipDNN
fields.** It specifies how a `$`-reference is spelled, how it resolves, what happens when it does
not resolve, what each operator does to the values it receives, and what an implementation must
guarantee about bounds and determinism. What the references *are* — which namespace roots exist,
which fields they carry, and what each field means — is supplied by the host subsystem:
[RFC 0019 §3](0019_UniversalMatchDescriptor.md#3-symbol-binding-what-the-engine-publishes) for the
matcher, and each follow-up for its own consumer. Where this document shows a hipDNN-flavoured
example, it says so and the example is illustrative, never normative.

### 1.1 What This RFC Specifies Versus Defers

| Capability | This RFC (day-one) | Deferred |
|---|---|---|
| The JsonLogic dialect and the `$`-variable convention (no `var` wrapper) | Yes ([§3](#3-syntax)) | None |
| Type domain and static typing of every expression against a declared environment | Yes ([§5](#5-the-type-system), [A.4](#a4-type-rules)) | None |
| The binding-environment contract: namespace roots, typed fields, optionality, reserved roots, precomputed fields | Yes ([§4](#4-the-binding-environment)) | The concrete environment of each consumer: [RFC 0019 §3](0019_UniversalMatchDescriptor.md#3-symbol-binding-what-the-engine-publishes) and the UDD/UHD follow-ups |
| The closed operator set, exhaustive with arities and types | Yes ([§6](#6-operators), [A.3](#a3-operator-reference)) | New operators as consumers motivate them, additively ([§11](#11-versioning-and-evolution)) |
| Unknown values and three-valued `and`/`or` | Yes ([§7](#7-unknown-values-and-three-valued-logic)) | None |
| Evaluation order, short-circuit, checked arithmetic, determinism | Yes ([§8](#8-evaluation-semantics)) | None |
| The bounded, fail-closed interpreter: depth and step caps, no arbitrary code, no reach outside the environment | Yes ([§9](#9-the-interpreter-safety-and-bounds)) | Descriptor size limits and quarantine policy: each consuming format's loader |
| Compiled in-memory typed AST, and lowering parity against the interpreter as the normative oracle | Yes ([§10](#10-compilation-and-lowering)) | The concrete compiled/bytecode encoding, and which lowering becomes the AOT fast path (KDP packaging follow-up) |
| A shared table-driven conformance suite and a parser/interpreter fuzzer | Yes ([§13](#13-conformance-and-testing)) | Where the suite physically lives ([Open Question 2](#15-open-questions)) |
| Language versioning, distinct from any descriptor's `schema` / `version` / `sdk_version` | Yes ([§11](#11-versioning-and-evolution)) | None |
| Shape matching and any dim-naming mechanism | None — dims are indexed positionally | A future shape-matching RFC |

---

## 2. Two Uses, One Language

An expression has a **rooted type**: the type its top-level node must produce. There are exactly two
rooted forms, and the root's expected type is the only thing that distinguishes them. The operators,
the syntax, the type rules, the unknown semantics, and the environment mechanism are identical.

- **A criteria expression** is `Bool`-rooted. It decides applicability: it answers yes, no, or — when
  it evaluates to unknown — declines. This is what a UMD carries
  ([RFC 0019 §4](0019_UniversalMatchDescriptor.md#4-constraint-vocabulary)). A root whose static type
  is not `Bool` is a compile error, and a root that evaluates to unknown fails closed
  ([§7](#7-unknown-values-and-three-valued-logic)).
- **A formula** is value-rooted. It yields a number: the UDD's grid, block, shared-memory, and
  workspace quantities and the `expr` form of an argument source
  ([RFC 0017 §6](0017_UniversalKernelDescriptor.md#6-dispatch-and-workspace)), and the UHD's derived
  features. A formula that evaluates to unknown is an evaluation error, not a decline, because
  nothing downstream of it can proceed on a missing number.

Both terms are used throughout this document with exactly these meanings. `Float` arises in practice
only in value-rooted formulas: a criterion compares integers, dtypes, booleans, and integer arrays,
while a scale factor or an occupancy ratio is a formula's business.

---

## 3. Syntax

The dialect is **JsonLogic**: a nested `{"op": [args]}` tree whose arguments are themselves
expressions or literals. An object used as an expression MUST have **exactly one key**, the operator;
a multi-key object is refused at compile rather than resolved by some ordering rule. A
one-argument operator accepts **unary sugar** — the bare operand in place of a one-element array — so
`{"!": "$t.packed"}` and `{"!": ["$t.packed"]}` are the same expression.

**The `$`-variable convention.** Stock JsonLogic reads a bound value with `{"var": "path"}`. This RFC
replaces that with a single rule: **any JSON string beginning with `$` is a variable reference**, and
**every other JSON scalar is a literal** — numbers (`128`, `1.4426950408889634`), booleans (`false`),
and non-`$` strings (an enum-value name such as `"BFLOAT16"`). There is no ambiguity to resolve, so
no `{"var": …}` wrapper is used or accepted; a descriptor containing one is refused as an
unrecognized operator key. A bare `$`-string is itself a valid criteria expression when it resolves
to `Bool`.

**Operators nest to any depth, and either operand of a comparison may itself be a computed
expression.** A leaf is a literal or a reference, but nothing forces a comparison's arguments to be
leaves, so no temporaries, no let-bindings, and no intermediate names are needed — and consequently
the language has no binding construct at all ([§8](#8-evaluation-semantics)).

The snippets below are illustrative, and their references are drawn from **the matcher environment
RFC 0019 defines** ([RFC 0019 §3](0019_UniversalMatchDescriptor.md#3-symbol-binding-what-the-engine-publishes));
this RFC attaches no meaning to the particular roots or fields they name.

```jsonc
// criteria (Bool-rooted), over RFC 0019's matcher environment
{"==": ["$q.rank", 4]}                                        // rank pinned explicitly
{"==": ["$q.dims[3]", 64]}                                    // positional dim equality
{"in": ["$q.dims[3]", [64, 128, 256]]}                        // set membership
{"divisible": [{"*": ["$y.dims[0]", "$y.dims[2]", "$y.dims[3]"]},
               "$kernel.MPerBlock"]}                          // computed operand, no temporary
{"or": [{"not_present": ["$attn_mask"]},
        {"==": ["$attn_mask.dtype", "$q.dtype"]}]}            // absent, or present and constrained
```

```jsonc
// formulas (value-rooted), over the same environment
{"ceil_div": ["$q.dims[2]", 16]}                              // grid extent over a sequence axis
{"*": [{"rsqrt": ["$q.dims[3]"]}, 1.4426950408889634]}        // log2(e) / sqrt(head extent)
```

---

## 4. The Binding Environment

The language evaluates against a **binding environment** supplied by the host subsystem: a set of
named, typed values addressed by `$`-references and grouped under **namespace roots**. This is the
seam that lets one language serve three subsystems, and it is specified here as a contract only —
this RFC names no root and no field.

**The contract a host declares.** A root is an identifier. A reference is a root optionally followed
by a dotted path, with `[i]` indexing where a field is an indexable sequence, spelled by the grammar
of [A.2](#a2-variable-references). The host declares:

- which namespace roots exist, and how a root's identifier comes to be (fixed, or introduced by the
  host's own matching or binding process);
- which fields each root carries;
- each field's type, drawn from the domain of [§5](#5-the-type-system);
- whether each field is required or optional — an optional field is one that may be absent for a
  given problem, and reading an absent one yields unknown
  ([§7](#7-unknown-values-and-three-valued-logic));
- optionally, a set of **reserved roots** that host-introduced identifiers MUST NOT shadow.

**Every reference is resolved and type-checked statically against that declaration.** An undeclared
root, an undeclared field, or a field whose declared type does not fit the operator consuming it is a
**compile error naming the reference**, not a runtime surprise. This is the property that makes an
expression checkable before it ever sees a live problem: a descriptor is validated against the
environment it will run in, at load, and a mismatch is loud there rather than a silent decline later.

**The language attaches no meaning to any particular root or field name.** `rank`, `dims`, `dtype`,
and everything else are the host's vocabulary. The only structural fact the language relies on is
that a reference resolves to a value of a declared type, or does not resolve at all.

**Precomputed fields.** A host MAY publish a **derived** value as an ordinary typed field, so an
expression compares it rather than re-deriving it. A quantity the host already computes — a
normalized form of a layout, a flag summarising several conditions, a scalar folded out of a
constant — becomes a field like any other, and the language sees no difference between it and a
value read straight off the problem. This is the first rung of the extension ladder
([§14](#14-risks)): a check that is awkward to express is often a field the environment should carry,
and adding one is an additive change to the host's declaration rather than a change to the language.

**RFC 0019 supplies the matcher environment** — five namespaces over the matched graph, the device,
and kernel metadata, published by the engine's pattern
([RFC 0019 §3](0019_UniversalMatchDescriptor.md#3-symbol-binding-what-the-engine-publishes)) — and
the UDD and UHD follow-ups reuse it rather than declaring their own, so one published set is the
single contract every consumer of a given engine is checked against.

---

## 5. The Type System

The type domain is small and closed:

| Type | Meaning |
|---|---|
| `Int` | A signed integer, evaluated with checked-width arithmetic ([§8](#8-evaluation-semantics)) |
| `Float` | A floating-point number |
| `Bool` | `true` or `false` |
| `Dtype` | An enum-value name, spelled as a non-`$` string such as `"BFLOAT16"` |
| `IntArray` | An ordered array of `Int` |
| `Tensor` | An opaque host-bound handle |

`Number` is the union of `Int` and `Float`. `Value` is any of the six. `Array` in an operator
signature means an array literal whose element type matches the operand compared against it.

**Every expression has a static type**, computed bottom-up from its operator and its arguments'
types, and checked at compile against the environment's declared field types
([§4](#4-the-binding-environment)). A mismatch is a compile error, not a coercion: there is no
implicit truthiness, no numeric-to-boolean conversion, and no string-to-enum widening. `Int` is
usable wherever `Number` is required, which is the only subtyping in the system.

**Comparison requires matching types on both sides.** `==` and `!=` compare two operands of the same
type; the ordered comparisons take `Number` on both sides. Comparing a `Dtype` to an `Int`, or an
`IntArray` to a scalar, is a compile error rather than a false result, because a type-confused
comparison in a criteria expression is a silent wrong answer.

**`Tensor` is opaque.** The language never introspects a host value: only the `rank` operator and
field access reach into a `Tensor`, and both go through the environment's declaration. There is no
way to enumerate a handle's contents, compare two handles structurally, or obtain anything from one
the host did not declare.

The exact per-operator typing rules are [A.4](#a4-type-rules), and the grammar the types are computed
over is [A.1](#a1-grammar).

---

## 6. Operators

The operator set is complete, closed, and small. Every operator is total (it produces a value,
unknown, or a fail-closed outcome for every input), side-effect-free, and means the same thing in
every host that embeds the language.

**Logical.** `and`, `or`, and `!` compose boolean sub-expressions, so a composite condition such as
`(A AND B) OR C` is stated directly in one tree with no extra mechanism. `and` and `or` are n-ary and
three-valued ([§7](#7-unknown-values-and-three-valued-logic)); both short-circuit
([§8](#8-evaluation-semantics)).

**Comparison.** `==` and `!=` over any two same-typed values, including `Dtype` and `IntArray`; `<`,
`<=`, `>`, `>=` over numbers.

**Membership.** `in` tests a value against an array literal, which is the compact spelling of a
disjunction of equalities and the natural form for a supported-set gate.

**Arithmetic.** `+`, `*` (n-ary), `-`, `/` (binary), and `%` over integers.

**Value core.** `min`, `max`, `abs`, `pow`, `ceil_div`, `log2`, and `rsqrt` are what make a formula
expressible. They are not speculative: every grid formula in the kernels this series targets is a
ceil-div over a sequence or spatial extent; `min` and `max` size a workspace that depends on a knob,
such as a split-K GEMM whose scratch is the larger of its partials and its reduction, or one floored
at a minimum; and `rsqrt` expresses the SDPA convention's implicit default scale, the reciprocal
square root of the head extent, which two kernel families in this repository compute today.

**Presence.** `present` and `not_present` answer "was this supplied?" rather than reading a value.
Both are n-ary over references, so one call answers a whole set of optional references at once —
"none of these unsupported optionals is present" is a single node, not a chain.

**Short-hands.** `rank` reads a `Tensor`'s rank. `divisible` exists so that a zero divisor yields
`false` rather than an error: it is `true` exactly when the divisor is non-zero and divides the
dividend, which gives uniform fail-closed zero-guarding without every author writing the guard by
hand beside a `%`. `value_or_default` reads a possibly-absent optional with a fallback that may
itself be an expression of the same type — usually a literal, but a second reference works — so "this
field, else that one" is one operator instead of a branch.

**Conditional.** `if` is an `if`/`elif`/`else` chain and is what encodes a precedence chain, such as
a mask classifier that selects among several mutually-ordered cases. All branch results MUST share a
type.

The complete set, with arities, argument types, and results:

| Operator | Arity | Argument types | Result |
|---|---|---|---|
| `and` | n-ary | `Bool…` | `Bool` |
| `or` | n-ary | `Bool…` | `Bool` |
| `!` | 1 | `Bool` | `Bool` |
| `==`, `!=` | 2 | `Value, Value` (same type) | `Bool` |
| `<`, `<=`, `>`, `>=` | 2 | `Number, Number` | `Bool` |
| `in` | 2 | `Value, Array` | `Bool` |
| `+`, `*` | n-ary | `Number…` | `Number` |
| `-`, `/` | 2 | `Number, Number` | `Number` |
| `%` | 2 | `Int, Int` | `Int` |
| `min`, `max` | n-ary | `Number…` | `Number` |
| `abs` | 1 | `Number` | `Number` |
| `pow` | 2 | `Number, Number` | `Number` |
| `ceil_div` | 2 | `Int, Int` | `Int` |
| `log2`, `rsqrt` | 1 | `Number` | `Float` |
| `rank` | 1 | `Tensor` | `Int` |
| `divisible` | 2 | `Int, Int` | `Bool` |
| `value_or_default` | 2 | `Ref, Value` | `Value` |
| `present`, `not_present` | n-ary | `Ref…` | `Bool` |
| `if` | 3 or 2n+1 | `Bool, Value [, Bool, Value]…, Value` | `Value` |

`n-ary` means two or more arguments. `Ref` is a syntactic position, not a value type: it requires a
`$`-reference written literally, because the operator asks about the reference rather than about a
value read through it. [A.3](#a3-operator-reference) restates this table as the normative reference.

**The set is closed.** The table above is exhaustive. An operation key that is not in it is refused
at compile time as an unrecognized operator, and a **dotted or namespaced key** — the form a custom
operation would naturally take — gets no special treatment: it is simply not listed, so it is
refused. There is no registry, no namespace, and no provider hook by which a descriptor introduces
one. [A.6](#a6-the-operator-set-is-closed) states this normatively.

**Why closed.** An in-language escape hatch has to be resolved by the compiler, which means the
language grows a registry, a signature table, and a per-argument type contract — a second extension
mechanism running parallel to the one the descriptor series already defines, differing only in grain.
Keeping the set closed is what makes an expression **fully interpretable from the schema alone**, and
that single property is what makes both drop-in loading and lowering tractable: a loader can validate
a file it has never seen without linking anything, and a lowering pass has a finite, fixed set of
node kinds to emit code for. A consumer that genuinely needs real C++ puts it **beside** the
descriptor, not inside the expression; the matcher's instance of that pattern is the native matcher
of [RFC 0019 §7](0019_UniversalMatchDescriptor.md#7-the-native-matcher-escape-hatch).

---

## 7. Unknown Values and Three-Valued Logic

A reference that does not resolve at evaluation time yields **unknown**, a distinguished result that
is neither true nor false and is not a value of any type. Three situations produce it: an absent
optional field, an out-of-range index, and a field whose value the host does not carry for this
particular problem.

**Unknown is never coerced.** It never reads as `false`, as `0`, or as "not equal". The reason is a
narrowing check: over RFC 0019's matcher environment, `{"!=": ["$attn_mask.dtype", "BFLOAT16"]}` says
"this operand is not bfloat16". If an unresolved reference compared as "not equal", that criterion
would silently **pass** on a problem that never supplied the operand at all — the check would appear
to hold precisely where it was never evaluated. Propagation instead of coercion is what makes an
absent optional unable to satisfy anything by accident.

**Every operator except `present`, `not_present`, and `value_or_default` propagates unknown.** Those
three answer "did this resolve?" rather than reading a value through the reference, so they always
yield a real result on a resolving and a non-resolving reference alike.

**`and` and `or` are three-valued.** A definite `false` decides an `and` and a definite `true`
decides an `or`, even beside an unknown argument; only an otherwise-undecided result stays unknown.

| `and` | `true` | `false` | unknown | | `or` | `true` | `false` | unknown |
|---|---|---|---|---|---|---|---|---|
| `true` | `true` | `false` | unknown | | `true` | `true` | `true` | `true` |
| `false` | `false` | `false` | `false` | | `false` | `true` | `false` | unknown |
| unknown | unknown | `false` | unknown | | unknown | `true` | unknown | unknown |

This is what lets an "absent, or present and constrained" pair accept a problem lacking the optional:
the first arm is definitely `true`, so the `or` is decided before the second arm's unresolvable field
reads matter.

**What an unknown root means depends on the rooted form.** A `Bool`-rooted criteria expression
evaluating to unknown **fails closed** — it declines, exactly as a `false` would, and never matches
by default. A value-rooted formula evaluating to unknown is an **evaluation error**: there is no
number to hand downstream, and substituting one would be a fabricated launch geometry or workspace
size.

---

## 8. Evaluation Semantics

**Written order, with short-circuit.** Evaluation proceeds strictly left to right in the order the
author wrote. `and` stops at its first `false` and `or` at its first `true`, so the author controls
when a decision is reached and can put the cheap, highly selective test first. Written order is
observable through the trace ([§12](#12-diagnostics-and-evaluation-traces)), which is why it is
specified rather than left to the implementation.

**A compiler MAY reorder or hoist sub-expressions as an internal optimization, but only where doing
so cannot change the result.** This is permitted because every operator is total and side-effect-free:
there are no binding operators, no assignment, no environment mutation, and no way for one
sub-expression to observe whether another ran. Reordering therefore changes *when* a decision is
reached, never *what* it is. An optimization that could change a result — including one that
evaluates an arm the short-circuit would have skipped, in a way an evaluation error or a trace would
expose — is not permitted.

**Arithmetic is checked and fails closed.**

- Integer arithmetic uses **checked-width integers** and fails closed on overflow rather than
  wrapping. A wrapped size computation under-allocates, which is the exact class of bug the language
  must not be able to produce silently.
- `/`, `%`, and `ceil_div` fail closed on a **zero divisor**. `divisible` is the deliberate exception:
  it yields `false`, giving uniform zero-guarding without an explicit guard beside every use
  ([§6](#6-operators)).
- `log2` and `rsqrt` fail closed on a **non-positive argument**.

Fail-closed means the same thing everywhere: a `Bool`-rooted expression declines, and a value-rooted
one reports an evaluation error. It never means a default value.

**Evaluation is deterministic.** The same expression over the same environment always yields the same
result — on any host, in any build, under any permitted internal reordering. There is no ambient
state, no clock, no allocation-order dependence, and no unspecified numeric behavior an
implementation is free to vary. Determinism is not a nicety here: it is precisely what makes lowering
parity testable ([§10](#10-compilation-and-lowering)), since a parity test compares two
implementations by their outputs alone.

---

## 9. The Interpreter: Safety and Bounds

On a drop-in path the interpreter parses input that may be untrusted, and on every path it parses
input that may simply be malformed. It is therefore **bounded and fails closed rather than crashing**
([RFC 0017 §16](0017_UniversalKernelDescriptor.md#16-risks)).

- **Bounded parsing and evaluation.** Recursion depth and expression step count are capped. Exceeding
  a cap **quarantines the expression** with a diagnostic; it never aborts the host.
- **Fail-closed evaluation.** An unknown symbol, an unrecognized operator key, an out-of-range index,
  a type error, a non-boolean criteria result, or an invalid operation declines. Nothing matches, and
  no value is produced, by default.
- **Checked arithmetic.** As [§8](#8-evaluation-semantics) specifies, including overflow, zero
  divisors, and non-positive `log2`/`rsqrt` arguments.
- **No arbitrary code, no reach outside the environment.** The interpreter executes only the operators
  of [A.3](#a3-operator-reference) and reads only what the binding environment declares
  ([§4](#4-the-binding-environment)). It opens no file, resolves no symbol, and calls into no
  registry.
- **Hand-written evaluator, no third-party parser.** The dialect is tiny, and keeping the evaluator
  in-tree is what keeps it small enough both to audit and to lower
  ([§10](#10-compilation-and-lowering)).

**Division of responsibility.** This RFC bounds the *interpreter*: expression depth, step count, and
the fail-closed set above. Descriptor size limits, per-descriptor node counts, and the quarantine
policy that decides what happens to the rest of a load when one file is bad belong to each consuming
format's loader — for the matcher, [RFC 0019 §13](0019_UniversalMatchDescriptor.md#13-security-and-hostile-input).
The two compose: a loader's cap bounds how much expression there can be, and the interpreter's cap
bounds what evaluating it can cost.

---

## 10. Compilation and Lowering

An expression is authored as text and **compiled once** into an in-memory typed AST: the parse
resolves every operator key, computes every node's static type, resolves every `$`-reference against
the declared environment, and rejects anything [A.5](#a5-static-validation) forbids. The compiled
form is cached and reused, and **the compiled form, not the text, is what evaluates**. Nothing is
parsed until something needs it, and nothing is re-parsed after that.

**The parity constraint (normative).** Any lowered form of an expression — a bytecode program,
generated C++, a shared decision tree, or anything else — **MUST be behaviorally identical to the
interpreter on the same expression and the same environment**, including its unknown results, its
fail-closed outcomes, and its short-circuit behavior. **The interpreter is the normative oracle**: a
disagreement is a bug in the lowering, never in the interpreter, and a lowering ships only behind a
parity test ([§13](#13-conformance-and-testing)). A lowering that agreed on the common cases while
diverging on an unknown or an overflow would be wrong in exactly the way that is hardest to see.

The lowering options, from least to most build coupling:

- **Interpreted typed AST (baseline).** The compiled form above, tree-walked. No codegen, identical on
  every path by construction. This is the fallback and the parity oracle.
- **Compact bytecode.** Lower the AST to a linear typed program executed by a tiny VM. Serializable,
  so a drop-in artifact gets the same treatment an AOT one does. Faster than tree-walking, still
  pure data.
- **Generated native code, AOT only.** Emit a specialized function per expression and compile it into
  the provider. Closest to hand-written code, but unavailable to a pure drop-in descriptor, which
  falls back to the interpreted or bytecode path.

The concrete choice is deferred ([Open Question 3](#15-open-questions)); the parity constraint is not.

**Consumers layer their own structure on top.** RFC 0019's root-opcode index over engines and its
memoization of criteria on the `$kernel.*` fields they read are *matcher* concerns: they decide which
expressions run and how often, not what an expression means
([RFC 0019 §10](0019_UniversalMatchDescriptor.md#10-static-matcher-sketch)). The language's contract
to them is that an expression is a pure function of its environment, which is what makes indexing,
memoizing, and caching sound in the first place.

---

## 11. Versioning and Evolution

**The language carries its own version**, distinct from any descriptor's `schema`, `version`, or
`sdk_version`. A descriptor format states which language version it requires, and the runtime honors
a requirement it can meet and refuses one it cannot. Versioning the language separately is what lets
three descriptor formats evolve on their own cadences over one shared dialect.

- **Adding an operator is additive** where it cannot change the meaning of an existing expression.
  Because the set is closed, an expression that did not name the new key is unaffected by definition,
  so the only additive question is whether the new operator changes the typing or the result of an
  existing form. Where it does not, it is a minor-version addition.
- **Changing an existing operator's semantics, arity, or argument types is breaking** and bumps the
  version. So is **removing** an operator.
- **An expression naming an operator the runtime does not know is refused with a clear error**
  ([§12](#12-diagnostics-and-evaluation-traces)). It is never silently reinterpreted, never treated
  as a no-op, and never skipped — a skipped criterion is an over-claim, and a skipped formula is a
  fabricated number.

`abs`, `pow`, `log2`, and `if` are in this RFC's set but absent from the vocabulary table of
[RFC 0017 §5](0017_UniversalKernelDescriptor.md#5-matching-the-ueds-pattern-and-the-umds-criteria),
which declares itself complete. They are needed for dispatch formulas and precedence chains, and they
are additive under the rule above; reconciling the two lists is
[Open Question 1](#15-open-questions).

---

## 12. Diagnostics and Evaluation Traces

Because expressions are data, they are inspectable, and an implementation is required to make use of
that rather than reporting a bare verdict.

**The evaluation-trace contract.** An expression that declines or fails MUST report:

- the **specific sub-expression** that decided, identified by its position in the tree;
- the **concrete values** of that sub-expression's arguments, as resolved from the environment;
- the **reason**: a comparison evaluated false, an unknown propagated (naming the reference that did
  not resolve), an integer overflow, a zero divisor, a non-positive `log2`/`rsqrt` argument, a type
  error, or a cap exceeded.

That is enough for a consumer to name exactly which test decided and why, which is the whole point:
"this kernel declined" is not actionable, and "this comparison of these two values was false"
is.

**Compile diagnostics** name the offending construct: the unrecognized operator key, the operator
whose arity or argument types were not satisfied, the multi-key object, the reference that did not
resolve against the declared environment (and the type it was declared with, where it resolved but
did not type-check), or the cap that was exceeded.

**Consumers render these into their own diagnostic surfaces.** The matcher's why-not trace, binding
view, and load diagnostics are
[RFC 0019 §14](0019_UniversalMatchDescriptor.md#14-observability-and-diagnostics)'s, and the UDD and
UHD follow-ups have their own. This RFC defines **what a trace must contain**, not how it is
presented, formatted, or plumbed.

---

## 13. Conformance and Testing

**A single table-driven conformance suite is a first-class deliverable of this RFC**, shared by every
consumer rather than reimplemented per subsystem. One language with three consumers and three
independent test suites would be three dialects with the same name inside a year; the shared suite is
the mechanism that keeps that from happening. It covers:

- **Every operator** at each of its declared arities and argument types, in **both rooted forms**.
- **The unknown-propagation matrix**: for every operator, that it propagates unknown, and for
  `present`, `not_present`, and `value_or_default`, that it does not.
- **The three-valued `and`/`or` truth tables** of [§7](#7-unknown-values-and-three-valued-logic),
  exhaustively.
- **The fail-closed cases**: integer overflow, a zero divisor on `/`, `%`, and `ceil_div`, `false` on
  a zero divisor for `divisible`, a non-positive `log2`/`rsqrt` argument, an unknown symbol, an
  out-of-range index, a type error, a non-boolean criteria root, and an unrecognized operator key
  (including a dotted/namespaced one).
- **Short-circuit and written-order observability**: that `and` stops at its first `false` and `or` at
  its first `true`, asserted through the trace rather than inferred.
- **Static-validation rejections**: each numbered check of [A.5](#a5-static-validation) is rejected
  with the diagnostic that names it, tested directly rather than only through expressions that happen
  to be valid.

Two further layers sit on top:

- **Lowering parity.** A cross-path equivalence check: every conformance case is run through the
  interpreter oracle and through each lowered form, and the results MUST agree
  ([§10](#10-compilation-and-lowering)). This is what gates a lowering into a release.
- **Fuzzing.** A fuzzer over the parser and the interpreter with a seed corpus of valid and
  deliberately-malformed expressions, run under the existing ASAN build, backing the fail-closed and
  bounded requirements of [§9](#9-the-interpreter-safety-and-bounds) against inputs no hand-written
  case would think of.

---

## 14. Risks

- **One language shared by three subsystems.** A change made for one consumer affects the others, and
  a subsystem can acquire an incidental dependency on a behavior another subsystem was relying on
  differently. Mitigation: one parser, one validator, one interpreter, and the shared conformance
  suite of [§13](#13-conformance-and-testing); the rooted-type split of
  [§2](#2-two-uses-one-language) is deliberately the *only* per-consumer distinction, so there is no
  per-subsystem dialect to drift.
- **A closed operator set adds friction by design.** A consumer needing a check the set cannot state
  must either motivate an additive operator or reach for a native hatch beside the descriptor, and
  both cost more than writing the check inline would. That friction is the point — it is what keeps
  every expression interpretable from the schema alone
  ([§6](#6-operators)) — but it is real. Mitigation: the graded ladder. First ask whether the host
  should publish a **precomputed field** ([§4](#4-the-binding-environment)), which is an additive
  environment change and usually the right answer; then whether the check generalizes into an
  **additive operator** ([§11](#11-versioning-and-evolution)); and only then the **native hatch**
  beside the descriptor ([RFC 0019 §7](0019_UniversalMatchDescriptor.md#7-the-native-matcher-escape-hatch)).
- **Lowering divergence is a silent correctness bug.** A lowered form that disagrees with the
  interpreter produces a wrong decision with no error anywhere. Mitigation: the interpreter is the
  normative oracle and the parity test gates any lowering
  ([§10](#10-compilation-and-lowering), [§13](#13-conformance-and-testing)).
- **No shape matching and no dim naming.** Where a host environment publishes an indexable sequence
  of dimensions, an expression indexes it positionally — over RFC 0019's matcher environment, an
  author writes `$q.dims[3]` where a named form would read better, and a reader needs the operand's
  dim order in hand to follow a criterion. It is less legible and more sensitive to a convention
  change than a named form would be. Mitigation: a comment beside the reference in the authoring
  form, and the future shape-matching RFC, which will specify shape matching and dim naming together
  rather than bolting a naming mechanism onto the language now.

---

## 15. Open Questions

1. **Operators RFC 0017 omits.** `abs`, `pow`, `log2`, and `if` are in this RFC's set but not in the
   vocabulary table of
   [RFC 0017 §5](0017_UniversalKernelDescriptor.md#5-matching-the-ueds-pattern-and-the-umds-criteria),
   which declares itself complete. Add them to RFC 0017's table, or scope them here as value-rooted
   only and keep criteria to RFC 0017's set?
2. **Where the conformance suite lives.** Three consuming subsystems and one interpreter: does the
   suite live with the interpreter implementation, as a standalone shared test target every consumer
   depends on, or as data that each consumer's test binary drives?
3. **Which lowering becomes the AOT fast path**, and does a serialized bytecode also serve the
   drop-in path, giving both paths one artifact form
   ([§10](#10-compilation-and-lowering))?
4. **`Float` precision.** Does the value core need explicit precision and rounding rules — a declared
   evaluation width, a rounding mode, a statement about associativity in n-ary `+`/`*` — or is IEEE
   double throughout sufficient for dispatch formulas, whose results are almost always immediately
   truncated to an integer extent?

---

## 16. References and Prior Art

The design borrows established ideas; none is a dependency. These informed the language specifically.

| System | Idea borrowed |
|---|---|
| **JsonLogic** | The base dialect: tiny data-only expression trees, JSON-native, one key per operation. Its truthiness coercion and its `var` wrapper are deliberately **not** borrowed ([§3](#3-syntax)) |
| **Google CEL** | A closed, statically-typed, non-Turing-complete expression language checked against a declared environment with bounded evaluation — the closest analogue to this design's overall shape |
| **SQL three-valued logic** | NULL propagation through operators and the `and`/`or` truth tables reproduced in [§7](#7-unknown-values-and-three-valued-logic) |
| **MLIR PDL native constraints** | A named escape hatch that sits **beside** the declarative language rather than inside it, keeping the declarative core closed ([§6](#6-operators)) |
| **WebAssembly / eBPF verifiers** | Bounded, statically-validated programs over untrusted input that fail closed and cannot reach outside their declared environment ([§9](#9-the-interpreter-safety-and-bounds)) |

---

## 17. Glossary

- **Expression:** a nested `{"op": [args]}` tree of operators, literals, and `$`-references, carried
  as pure data by a descriptor and evaluated over a binding environment ([§3](#3-syntax)).
- **Criteria expression (criteria):** a `Bool`-rooted expression that decides applicability; unknown
  fails closed. This is what a UMD carries ([§2](#2-two-uses-one-language)).
- **Formula:** a value-rooted expression that yields a number — a UDD grid, block, shared-memory,
  workspace, or `expr` argument source, or a UHD derived feature; unknown is an evaluation error
  ([§2](#2-two-uses-one-language)).
- **Binding environment:** the set of named, typed values a host subsystem publishes for expressions
  to read, grouped under namespace roots and declared with types and optionality
  ([§4](#4-the-binding-environment)).
- **Namespace root:** the leading identifier of a reference, under which a host groups a set of
  fields. A host MAY reserve roots against shadowing ([§4](#4-the-binding-environment)).
- **Reference:** a JSON string beginning with `$`, naming a root and an optional dotted path with
  `[i]` indexing; resolved and type-checked statically against the declared environment
  ([A.2](#a2-variable-references)).
- **Literal:** any JSON scalar not beginning with `$` — a number, a boolean, a non-`$` string such as
  an enum-value name — or a JSON array of them ([§3](#3-syntax)).
- **Unknown:** the result of a reference that does not resolve at evaluation time; neither true nor
  false, never coerced, propagated by every operator except `present`, `not_present`, and
  `value_or_default` ([§7](#7-unknown-values-and-three-valued-logic)).
- **Three-valued logic:** the `and`/`or` semantics under unknown — a definite `false` decides an
  `and`, a definite `true` decides an `or` ([§7](#7-unknown-values-and-three-valued-logic)).
- **Precomputed field:** a derived value a host publishes as an ordinary typed field so an expression
  compares it instead of re-deriving it; indistinguishable from any other field to the language
  ([§4](#4-the-binding-environment)).
- **Closed operator set:** the property that [A.3](#a3-operator-reference) is exhaustive and no
  registry, namespace, or provider hook can add to it ([A.6](#a6-the-operator-set-is-closed)).
- **Rooted type:** the type an expression's top-level node must produce, `Bool` for criteria and a
  value type for a formula; the only thing distinguishing the two uses
  ([§2](#2-two-uses-one-language)).
- **Compiled form:** the in-memory typed AST an expression is parsed into once and cached; what
  actually evaluates ([§10](#10-compilation-and-lowering)).
- **Parity oracle:** the interpreter, against which every lowered form MUST be behaviorally identical
  ([§10](#10-compilation-and-lowering)).
- **Conformance suite:** the shared table-driven test suite over operators, unknown propagation,
  three-valued logic, fail-closed cases, short-circuit order, and static validation, plus lowering
  parity and fuzzing ([§13](#13-conformance-and-testing)).
- **Fail closed:** the uniform response to an error or an undecidable result — a criteria expression
  declines, a formula reports an evaluation error, and neither substitutes a default
  ([§8](#8-evaluation-semantics), [§9](#9-the-interpreter-safety-and-bounds)).

---

## Appendix A: Normative Reference

This appendix is the normative specification of the language. Where the prose sections above describe
a construct by example, the grammar and tables here fix its exact form. An expression that violates a
**MUST** here is refused at compile ([§10](#10-compilation-and-lowering)); it never evaluates by
default ([§9](#9-the-interpreter-safety-and-bounds)). Grammar is EBNF; quoted terminals are JSON
tokens. The binding environment the grammar's roots and fields resolve against is the host's, not
this appendix's ([§4](#4-the-binding-environment)).

### A.1 Grammar

```ebnf
expr        = literal | var-ref-str | operation ;
operation   = "{" , op-key , ":" , operand , "}" ;   (* exactly one key *)
operand     = expr | arg-array ;                      (* unary sugar, or an argument list *)
arg-array   = "[" , [ expr , { "," , expr } ] , "]" ;
op-key      = '"' , builtin-op , '"' ;                (* builtin-op is a key of A.3 *)
literal     = number | boolean | non-dollar-string | json-array ;
```

- An object used as an expression MUST have **exactly one key**, the operator; a multi-key object is
  refused.
- `non-dollar-string` is any JSON string not beginning with `$`, and is a literal (an enum-value name,
  for example).
- A one-argument operator accepts **unary sugar**: the bare operand in place of a one-element
  `arg-array`.
- `op-key` admits only a key listed in [A.3](#a3-operator-reference); there is no `custom-op`
  production, because the set is closed ([A.6](#a6-the-operator-set-is-closed)).
- The root MUST be an `expr` whose static type is the required rooted type
  ([§2](#2-two-uses-one-language)): `Bool` for a criteria expression, a value type for a formula. A
  bare `var-ref-str` is a valid criteria expression only when it resolves to `Bool`.
- Evaluation is strictly in written order with short-circuit ([§8](#8-evaluation-semantics)).

### A.2 Variable references

Any JSON string beginning with `$` is a variable reference; every other JSON scalar is a literal. No
`{"var": …}` wrapper is used or accepted.

```ebnf
var-ref-str = '"' , var-ref , '"' ;
var-ref     = "$" , root , { "." , field | "[" , uint , "]" } ;
root        = ident ;              (* declared by the host binding environment *)
field       = ident ;              (* declared by the host binding environment *)
uint        = digit , { digit } ;
```

The set of valid `root` identifiers, the fields each root carries, each field's type, each field's
optionality, and which path segments are indexable are **supplied by the host binding environment**
([§4](#4-the-binding-environment)); this grammar fixes only the spelling. A host MAY declare reserved
roots that host-introduced identifiers MUST NOT shadow.

Resolution rules:

- Every reference MUST resolve, at compile time, against the declared environment, to a field whose
  declared type is compatible with the operator consuming it. A reference that does not is a compile
  error naming the reference ([A.5](#a5-static-validation)).
- A field access on an **absent optional** field resolves, at evaluation time, to **unknown**, which
  is neither true nor false ([§7](#7-unknown-values-and-three-valued-logic)). Unknown is not coerced:
  it never reads as `false`, `0`, or "not equal".
- An **out-of-range index**, or any other reference that does not resolve at evaluation time, is
  likewise unknown.
- **Every operator except `present`, `not_present`, and `value_or_default` propagates unknown.** Those
  three report whether a reference resolved and so always yield a real value, on a required and an
  optional reference alike.
- **`and` and `or` are three-valued.** A definite `false` decides an `and` and a definite `true`
  decides an `or`, even beside an unknown argument; only an otherwise-undecided result stays unknown.
- A `Bool`-rooted expression evaluating to unknown fails closed; a value-rooted one reports an
  evaluation error.

### A.3 Operator reference

Integer arithmetic uses checked-width integers and fails closed on overflow
([§9](#9-the-interpreter-safety-and-bounds)). `n-ary` means two or more arguments. `Ref` denotes a
syntactic position requiring a literally-written `$`-reference.

| Operator | Arity | Argument types | Result |
|---|---|---|---|
| `and` | n-ary | `Bool…` | `Bool` |
| `or` | n-ary | `Bool…` | `Bool` |
| `!` | 1 | `Bool` | `Bool` |
| `==`, `!=` | 2 | `Value, Value` (same type) | `Bool` |
| `<`, `<=`, `>`, `>=` | 2 | `Number, Number` | `Bool` |
| `in` | 2 | `Value, Array` | `Bool` |
| `+`, `*` | n-ary | `Number…` | `Number` |
| `-`, `/` | 2 | `Number, Number` | `Number` |
| `%` | 2 | `Int, Int` | `Int` |
| `min`, `max` | n-ary | `Number…` | `Number` |
| `abs` | 1 | `Number` | `Number` |
| `pow` | 2 | `Number, Number` | `Number` |
| `ceil_div` | 2 | `Int, Int` | `Int` |
| `log2`, `rsqrt` | 1 | `Number` | `Float` |
| `rank` | 1 | `Tensor` | `Int` |
| `divisible` | 2 | `Int, Int` | `Bool` |
| `value_or_default` | 2 | `Ref, Value` | `Value` |
| `present`, `not_present` | n-ary | `Ref…` | `Bool` |
| `if` | 3 or 2n+1 | `Bool, Value [, Bool, Value]…, Value` | `Value` |

Per-operator rules:

- `and` short-circuits at the first `false`, `or` at the first `true`; both are three-valued.
- `!` accepts unary sugar, as does every one-argument operator.
- `in` requires the array's element type to match the compared value's type.
- `/`, `%`, and `ceil_div` **fail closed on a zero divisor**. `divisible` yields **`false`** on a zero
  divisor rather than an error — it is `true` exactly when the divisor is non-zero and divides the
  dividend — which gives uniform fail-closed zero-guarding.
- `log2` and `rsqrt` **fail closed on a non-positive argument**.
- `rank` reads an opaque `Tensor`'s rank and is equal to that handle's declared `rank` field where the
  host declares one.
- `value_or_default` yields the referenced optional's value when it resolves, else the fallback. The
  fallback is usually a literal but MAY be any expression of the same type, including a second
  reference; both arms MUST be type-compatible against the environment's declared field types.
- `present` is `true` iff **every** referenced field resolves; `not_present` iff **every** one is
  absent. Both take a list, so one call answers a set of optional references.
- `if` is an `if`/`elif`/`else` chain; all branch results MUST share a type.

**Unknown propagation.** Every operator in this table except `present`, `not_present`, and
`value_or_default` propagates unknown rather than coercing it, so an unresolved reference can never
satisfy a criteria expression by accident ([A.2](#a2-variable-references)).

**Closedness.** This table is exhaustive; see [A.6](#a6-the-operator-set-is-closed).

### A.4 Type rules

The type domain is `Int`, `Float`, `Bool`, `Dtype`, `IntArray`, and `Tensor`. `Number` is `Int` or
`Float`; `Value` is any of the six; `Array` is an array literal whose element type matches the value
compared against it. `Ref` is a syntactic position, not a member of the domain.

1. Every expression has a **static type**, computed bottom-up from its operator and its arguments'
   static types per [A.3](#a3-operator-reference). A literal's type is its JSON type, with a non-`$`
   string typed as `Dtype` where the consuming position expects one. A reference's type is the field's
   declared type ([§4](#4-the-binding-environment)).
2. `Int` is usable wherever `Number` is required. There is no other subtyping, and there are **no
   implicit conversions** — no truthiness, no numeric-to-boolean, no widening of an unrelated type.
3. `==` and `!=` require both operands to have the **same** static type. `<`, `<=`, `>`, `>=` require
   `Number` on both sides.
4. An n-ary `+`, `*`, `min`, or `max` yields `Float` if any argument is `Float`, else `Int`. `-` and
   `/` follow the same rule. `%` and `ceil_div` are `Int`-only in both arguments and result. `log2`
   and `rsqrt` always yield `Float`.
5. `if` requires `Bool` in every condition position and a single shared type across every branch
   result, which is the expression's type.
6. `value_or_default` yields the declared type of its referenced field, and its fallback MUST have
   that same type.
7. `rank` requires `Tensor` and yields `Int`. `Tensor` is **opaque**: only `rank` and declared field
   access reach into it, and the language never introspects a host value.
8. Any violation of rules 1–7 is a **compile error**, never a runtime coercion and never a `false`
   result.

### A.5 Static validation

An expression MUST pass every check below to compile. A failure refuses the expression with a
diagnostic naming the offending construct ([§12](#12-diagnostics-and-evaluation-traces)) and, on a
drop-in path, quarantines the descriptor carrying it per that format's loader policy
([§9](#9-the-interpreter-safety-and-bounds)).

1. Every object used as an expression has **exactly one key** ([A.1](#a1-grammar)).
2. Every operation key appears in [A.3](#a3-operator-reference)
   ([A.6](#a6-the-operator-set-is-closed)).
3. Every operation's **arity and argument types** satisfy [A.3](#a3-operator-reference) and
   [A.4](#a4-type-rules), including the same-type requirement on comparisons and the shared-branch-type
   requirement on `if`.
4. The root's **static type matches the required rooted type**: `Bool` for a criteria expression, a
   value type for a formula ([§2](#2-two-uses-one-language)).
5. Every `$`-reference **resolves against the declared binding environment** to a field whose declared
   type is compatible with the operator consuming it, and every `Ref` position holds a literally-written
   reference ([§4](#4-the-binding-environment), [A.2](#a2-variable-references)).
6. **Recursion depth and node count are within the interpreter's caps**
   ([§9](#9-the-interpreter-safety-and-bounds)).

### A.6 The operator set is closed

[A.3](#a3-operator-reference) is exhaustive. An operation key not listed there is refused at compile
time as an unrecognized operator, and there is **no registry, namespace, or provider hook** by which a
descriptor can introduce one.

A **namespaced (dotted) key** is the form a custom operation would naturally take and gets no special
treatment: it is not in [A.3](#a3-operator-reference), so it is refused exactly as any other unknown
key is. Adding a built-in operator to [A.3](#a3-operator-reference) is additive under
[§11](#11-versioning-and-evolution); adding one out-of-band is not possible. That is what makes an
expression fully interpretable from the schema alone, which is in turn what makes both the drop-in
path and the lowerings of [§10](#10-compilation-and-lowering) tractable. A check that genuinely needs
real code lives **beside** the descriptor, not inside the expression — for the matcher, the native
matcher of [RFC 0019 §7](0019_UniversalMatchDescriptor.md#7-the-native-matcher-escape-hatch).
