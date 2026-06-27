# `ck.dsl.ir/v1` — IR Serialization Format

| | |
|---|---|
| **Status** | Implemented on both engines (Python and C++) |
| **Decision** | A separate, explicit machine format; `print_ir` stays human-only |
| **Implementation** | Python: `rocke/core/ir_serialize.py` (`serialize` / `parse`), `rocke/core/verify.py`. C++: `Cpp/core/ir/serialize.cpp` (`rocke_ir_parse` and the serializer), header `Cpp/include/rocke/ir_serialize.h`. |
| **Scope** | A fully round-trippable text encoding of a `KernelDef` (`rocke/core/ir.py`) |

---

## 0. Purpose

`ck.dsl.ir/v1` is the **interchange artifact** for the dual-backend path,
the seam between the front end that builds IR and the backend that lowers it.
It captures *everything* needed to reconstruct a
`KernelDef` **exactly**, most importantly the **explicit SSA value ids**. The
Python and C++ engines serialize the IR they built; the consumer parses what was
written rather than re-deriving names. This kills the recurring **SSA-numbering
drift** defect class: if both backends serialize the same ids, the
text is identical; if they diverge, the diff is surfaced upstream of `.ll`.

This is explicitly **not** `print_ir`. `print_ir`
(`rocke/core/ir_print.py`) is a lossy, human-facing MLIR-ish dump: it has no
versioned header, drops `loc`, renders attrs without type tags (an `int` `1` and
a `bool` `True`/`1` are ambiguous), and is not parseable. `ck.dsl.ir/v1` is the
opposite on every axis: versioned, typed, lossless, parseable, and canonical.

Non-goal: this is **not** a human authoring input. It is a machine
contract. Readability is a courtesy, not a requirement.

---

## 1. Lexical structure

- UTF-8 text, `\n`-terminated lines. The serializer always emits `\n`; the
  parser tolerates `\r\n`.
- Indentation is two spaces per nesting level and is **significant only as a
  courtesy** — the parser keys off the `{` / `}` block tokens and the line
  grammar, not the column. (Round-trip is still byte-identical because the
  serializer's indentation is deterministic.)
- Blank lines and lines whose first non-space character is `#` are comments and
  are ignored by the parser. The serializer emits no comments, so round-trip is
  unaffected.
- Tokens within a line are space-separated except inside the delimited
  `(...)` operand list, `[...]` type/result list, and `{...}` attr map, which
  use `, ` separators.

### 1.1 String literals

Strings (op names appear bare; *string-valued attrs*, `loc`, and the kernel
name's textual form) are encoded with a minimal, unambiguous escape:
`\\` → backslash, `\"` → quote, `\n` → newline, `\t` → tab. Every other byte is
literal. This matters because attrs like `tile.inline_asm`'s `template` carry
embedded newlines and quotes (e.g. MFMA asm). The escape set is closed under
round-trip and is the same on both backends.

---

## 2. Types

A `Type` (and its subclasses `VectorType`, `PtrType`, `SmemType`) is encoded as
its **canonical type string**, which is exactly the `Type.name` field the IR
already computes:

| Form | Encoding | Example |
|---|---|---|
| scalar | `<name>` | `i32`, `f16`, `fp8e4m3` |
| vector | `vec<<elem>x<count>>` | `vec<f32x16>` |
| pointer | `ptr<<pointee>,<space>>` | `ptr<f16,global>` |
| smem | `smem<<elem>, [<d0>x<d1>...]>` | `smem<f16, [128x32]>` |

Choosing `Type.name` as the wire form is deliberate: it is the value the IR
*already* uses for identity (`Type` is a frozen dataclass keyed on `name`), the
LLVM lowerer reads it, and `print_ir` shows it — so there is exactly one type
spelling across the whole stack. The parser (`_parse_type`) is the inverse: it
recognizes the `vec<...>`, `ptr<...>`, `smem<...>` prefixes and recursively
parses the element/pointee, falling back to a scalar `Type(name)` (which
reconstructs the module-level singletons `I32`, `F16`, … by value equality since
`Type` is frozen).

Round-trip note: `SmemType.name` renders the shape with a leading space after
the comma (`smem<f16, [128x32]>`); the parser accepts optional whitespace there.

---

## 3. Document grammar

```
<document>   ::= <header> <kernel>
<header>     ::= "ckdsl.ir" SP "v1" NL
<kernel>     ::= "kernel" SP "@" <name> SP "{" NL
                   <attrs-line>?          ; kernel-level attrs, if any
                   <params-block>
                   <region>              ; the entry/body region, label "entry"
                 "}" NL
```

### 3.1 Kernel header line

```
ckdsl.ir v1
kernel @vgemm {
```

The version token (`v1`) is mandatory and checked by the parser; an unknown
version is a hard error (forward-compat gate).

### 3.2 Kernel-level attrs

If `kernel.attrs` is non-empty, exactly one line:

```
  attrs <attr-map>
```

`<attr-map>` is the canonical attr-map syntax of §5. Omitted entirely when
empty (so the common `{"max_workgroup_size": 256}` case is the only thing
present, and a no-attr kernel adds no line).

### 3.3 Params block

```
  params {
    %<name> : <type> <attr-map>?
    ...
  }
```

Each param is one line: the `%`-prefixed param name, ` : `, the canonical type,
and (only if non-empty) a trailing attr-map. The order is the declaration order
in `kernel.params` — preserved verbatim because it *is* the kernel ABI
(argument order is load-bearing for kernarg packing). The block is always
emitted, even when empty (`params {` immediately followed by `}`), so the
grammar has a fixed shape.

The param's SSA value name as seen by op operands is `%<name>` (the builder
stores params as `Value(name="%"+name)`); the serializer writes the bare
`%<name>` here and the parser registers `%<name>` in the value table so operand
references resolve.

---

## 4. Regions and ops

```
<region>  ::= "region" SP "@" <label> SP "{" NL
                <op>*
              "}" NL
<op>      ::= <results>? <opname> <operands>? <attrs>? <loc>? NL <nested-regions>?
```

### 4.1 Op line

The canonical single-line op form is:

```
<results> = <opname> ( <operands> ) <attr-map> @loc "<loc>"
```

- **`<results>`** — `%id : type` entries joined by `, `, followed by ` = `.
  Omitted (with the `=`) when the op has no results. **The result ids are
  written explicitly** — this is the core of the format. Each result is its
  exact `Value.name` plus its `Value.type` canonical string.
- **`<opname>`** — the op's `name` (e.g. `arith.add`, `tile.mma`, `scf.for`),
  bare (op names are a closed vocabulary of `[a-z0-9._]`).
- **`<operands>`** — `( %id, %id, ... )`, each operand referenced **by its SSA
  id only** (its type is recovered from the value table). Always emitted, even
  when empty (rendered as `(  )` — the fixed `( ` … ` )` delimiters with an
  empty body), to keep the line shape fixed and parsing trivial.
- **`<attr-map>`** — §5; omitted when empty.
- **`@loc "<...>"`** — present only when `op.loc is not None`. `print_ir` drops
  this; we keep it because it is part of the `KernelDef` and round-trip must be
  total.

Example (from a universal GEMM body):

```
    %add12 : i32 = arith.add ( %v3, %c8 )
    %acc : vec<f32x16> = tile.mma ( %av8, %bv8, %cz16 ) { op_id = s:"mfma_f32_32x32x8_f16" }
    cf.return (  )
```

### 4.2 Nested regions

Region-carrying ops (`scf.for`, `scf.if`) write their attr-map on the op line,
then their regions as indented blocks immediately after, before the next
sibling op:

```
    %for1 : vec<f32x16>, ... = scf.for ( %lb, %ub, %step, %acc_init, ... ) { ... attrs ... }
      region @body {
        ... body ops ...
        scf.yield ( %next_acc, ... ) { num = i:1 }
      }
```

The region label is `region.label` (`entry`, `body`, `then`). Region nesting is
recursive; there is no depth limit.

### 4.3 Block-defined SSA values (the subtle part)

Three SSA values are **not** produced as an op `result` yet are referenced by
operands inside a region:

1. **Params** — defined by the `params` block (§3.3).
2. **`scf.for` induction variable** — the loop carries `attrs["iv"]` (the `%`
   name) and `attrs["iv_type"]`. The serializer writes these as ordinary attrs
   (`iv = s:"%k0"`, `iv_type = s:"i32"`); the parser, *when it enters the
   `scf.for` body region*, reconstructs the `Value(name=iv, type=iv_type)` and
   registers it in the value table so body operands resolve.
3. **`scf.for` iter-arg block values** — `attrs["iter_args"]` is a list of
   `{"name": "%acc_...", "type": "vec<f32x16>"}` dicts. These name the
   loop-carried values *visible inside the body*. They are serialized as a typed
   list attr (§5.3) and the parser registers each as a body-scope `Value`.

This is why the format does not need a separate "block argument" syntax: the
exact contract the builder uses (`iv` / `iter_args` attrs) is preserved
verbatim, and the parser replays the builder's own registration logic. The
`scf.for` *result* values (`%for1`, …) are ordinary results on the op line.

`scf.yield`'s `num` attr is preserved; its operands are the yielded values.

---

## 5. Attributes

The attr surface that appears in real kernels (measured across GEMM, conv,
attention, norm, elementwise families) is: `int`, `float`, `bool`, `str`, and a
single `list`-of-`dict` case (`scf.for`'s `iter_args`). The format encodes the
**type tag** of every attr value so parsing is unambiguous (unlike `print_ir`,
where `1`/`True` collide).

### 5.1 Attr map

```
{ <key> = <tagged-value>, <key> = <tagged-value>, ... }
```

Keys are bare identifiers (`[A-Za-z_][A-Za-z0-9_]*` — all real attr keys
qualify). **Keys are emitted sorted (ascending, by Unicode code point)** — this
is the canonical ordering rule (§7). Empty maps are omitted from the op line
entirely.

### 5.2 Tagged scalar values

| Python type | Tag | Encoding | Example |
|---|---|---|---|
| `bool` | `b:` | `true` / `false` | `b:true` |
| `int` | `i:` | decimal, optional `-` | `i:-1` |
| `float` | `f:` | `repr(float)` (round-trip-exact) | `f:0.0`, `f:1.5` |
| `str` | `s:` | quoted, escaped per §1.1 | `s:"compv4"` |

`bool` is tagged and checked **before** `int` (Python `bool` is an `int`
subclass) so `True` never serializes as `i:1`. `float` uses `repr`, which on
CPython is the shortest string that round-trips to the same IEEE-754 double —
critical for attrs like `value = f:...` on `arith.constant` f32 literals. (Note:
the float *attr* round-trip is exact; the separate question of how the *LLVM
lowerer* formats the constant into `.ll` text is the lowerer's concern and is
unchanged by serialization.)

### 5.3 Tagged list values

The only list attr is `iter_args`. It is encoded as:

```
l:[ <tagged-value>, <tagged-value>, ... ]
```

and each element (a dict) as a nested attr-map (§5.1). So:

```
iter_args = l:[ { name = s:"%acc_m0_n0", type = s:"vec<f32x16>" }, ... ]
```

The encoding is **general**: `l:[...]` holds any tagged values, and a dict
element is just a `{...}` attr-map; this covers the present `iter_args` shape and
any future list/dict attr without a format change. Dict keys inside list
elements are sorted by the same rule.

---

## 6. Why these choices

- **Explicit SSA ids on every value (result, param, iv, iter-arg).** The whole
  point: the consumer prints what was assigned, not what it
  re-derives. The id is written once at its definition (result/param/iv/iter
  declaration) and referenced by bare id everywhere else.
- **Operands by id only; types recovered from the value table.** Operand types
  are redundant (an SSA value has exactly one type at its definition). Writing
  them would be a second source of truth that could drift; omitting them makes
  the format tighter and makes the parser's value-table the single authority.
- **Type tags on attrs.** Eliminates the `1` vs `True` ambiguity that makes
  `print_ir` unparseable, with zero guessing in the parser.
- **`repr(float)` for floats.** Shortest round-tripping decimal; deterministic
  on CPython; the C++ engine uses the same shortest-round-trip algorithm, so the
  two engines agree on the float wire form.
- **Sorted attr keys.** Determinism independent of dict insertion order, so two
  builders that add the same attrs in different orders serialize identically.
- **`loc` retained.** Round-trip must be total; `loc` is part of `KernelDef`.
- **Fixed-shape grammar (always-emitted `()`, `params {}`).** Trivial,
  branch-light parser; no "is this token an operand or an attr?" lookahead.

---

## 7. Canonicalization (for semantic diff)

Two notions of equality:

1. **Byte equality** — `serialize(k)` of the same `KernelDef` is deterministic
   (sorted attrs, fixed grammar, `repr` floats), so `serialize(parse(serialize
   (k))) == serialize(k)` is byte-identical. This is the round-trip idempotence
   gate.

2. **Semantic (canonical) equality** — two kernels are *semantically equal* if
   they differ only in **incidental SSA id choices** (the numbering-gap drift
   class). `canonicalize(text)` produces a normalized string where:

   - **Stable id normalization.** Every SSA id is renamed to `%<N>` by
     *first-definition order* in a pre-order walk (params first in ABI order,
     then ops/results/ivs/iter-args in textual order, descending into regions).
     Two kernels with the same structure but `%add1251` vs `%add1252` gaps
     normalize to the same `%0,%1,...` stream. Operand references are renamed
     consistently via the same map.
   - **Attr ordering.** Already sorted by the serializer; canonicalize re-sorts
     defensively.
   - **`loc` stripping.** `@loc "..."` is dropped (authoring-site provenance is
     not semantic).

   `canonical_equal(a, b)` returns `canonicalize(serialize(a)) ==
   canonicalize(serialize(b))`. This is the comparator the differential harness
   uses to assert builder equivalence independent of id gaps, while
   the byte-identity path (which keeps ids) remains available for the strict
   "ids must match too" check.

The two are intentionally separate: byte-identity proves *the serialization is
faithful*; canonical-equality proves *two builds mean the same thing*. The design
wants both — byte-identity as the round-trip gate, canonical-equality as the
drift-tolerant semantic diff.

---

## 8. Versioning

The header `ckdsl.ir v1` is the contract version. A change that is not
backward-compatible (new required token, changed type/attr encoding) bumps to
`v2`; the parser switches on the header and refuses unknown versions. Additive,
ignorable extensions (e.g. a new optional op-line suffix) may ship under `v1`
provided old parsers can skip them — but the conservative default is to bump.

---

## 9. Worked example

A trivial kernel built as:

```python
b = IRBuilder("add_one")
p = b.param("X", PtrType(F32, "global"), noalias=True)
i = b.const_i32(0)
v = b.global_load_f32(p, i)
one = b.const_f32(1.0)
s = b.fadd(v, one)
b.global_store(p, i, s)
b.ret()
```

serializes to:

```
ckdsl.ir v1
kernel @add_one {
  params {
    %X : ptr<f32,global> { noalias = b:true }
  }
  region @entry {
    %c1 : i32 = arith.constant (  ) { ity = s:"i32", value = i:0 }
    %gl2 : f32 = memref.global_load_typed ( %X, %c1 ) { align = i:4, elem_type = s:"f32" }
    %c3 : f32 = arith.constant (  ) { ity = s:"f32", value = f:1.0 }
    %fadd4 : f32 = arith.fadd ( %gl2, %c3 )
    memref.global_store_typed ( %X, %c1, %fadd4 ) { align = i:1, elem_type = s:"f32" }
    cf.return (  )
  }
}
```

(`parse` of that text rebuilds a `KernelDef` whose `lower_kernel_to_llvm`
output is identical to the original's.)
