# Portable IR — `rocke.ir/v1`, `rocke.recipe/v1`, `rocke.bundle/v1`

| | |
|---|---|
| **Status** | Implemented; byte-identity gated in CI, numerics gated on device |
| **Decision** | Ship kernels as structured artifacts the C++ engine replays; no CPython at runtime |
| **Implementation** | Python: `rocke/core/ir_export.py`, `rocke/portable_ir/`. C++: `cpp/portable_ir/` (`json_dom`, `cbor_dom`, `ir_import_json`, `recipe_vm`, `online`), headers in `cpp/include/rocke/`. |
| **Scope** | Three wire schemas plus the import/replay path that consumes them |

---

## 0. Purpose

These schemas let a kernel authored in Python ship as an artifact and be lowered
at runtime by the C++ engine alone. Python stays the authoring surface; nothing
in the serving process imports it.

```text
author time:  spec -> builder -> KernelDef -> artifact
run time:     artifact -> C++ import/replay -> .ll -> comgr -> HSACO -> launch
```

This sits between the two existing extremes. A prebuilt `.hsaco` is fastest but
frozen to one shape and one arch; shipping `.ll` is nearly as rigid, since the
datalayout and intrinsic spellings are already baked to an LLVM flavor. Portable
IR stays above both, so one artifact can be lowered against a different gfx
target or ROCm vintage than the one it was authored on.

Distinct from [`ir_serialization_format.md`](ir_serialization_format.md): that
is `ck.dsl.ir/v1`, rocke's own round-trippable *text* encoding, used as the seam
between the two engines and parsed by `rocke_ir_parse`. The schemas here are
JSON/CBOR, are produced by the Python front end, and are consumed by a different
importer. The two formats do not interoperate and are not versions of each other.

Non-goals: parsing or transpiling Python source; encoding host-side selectors,
launchers, or dispatch logic. Only the built graph, or the builder program that
produces it, crosses the boundary.

---

## 1. Which schema to use

| schema | encodes | shape flexibility | entry point |
|---|---|---|---|
| `rocke.ir/v1` | one concrete SSA graph | none — one artifact per shape | `rocke_import_kernel_from_json` |
| `rocke.recipe/v1` | the builder *program*, with its compile-time control flow | one artifact covers a family; specialized at replay | `rocke_recipe_run_from_json` / `_cbor` |
| `rocke.bundle/v1` | many recipes, addressed by key | as above, one file for a whole library | `rocke_recipe_run_from_bundle_cbor` |

A recipe is the interesting case. A builder that unrolls over a head dimension
`D` normally bakes `D` at Python time, so every `D` needs its own artifact. A
recipe keeps the loop as a `static_for` whose bound is the spec value, and the
VM expands it in C at JIT time — one small artifact, every `D`.

CBOR is the shipping encoding and JSON the debug one. Both decode to the same
arena-owned DOM (`jd_val_t`), so every consumer runs unchanged on either; that
equivalence is pinned by `tests/portable_ir/dom_decoders.cpp`. CBOR runs about
3× smaller.

---

## 2. `rocke.ir/v1` — concrete graph

Produced by `export_kernel_ir` / `export_kernel_ir_json`. Top level:

```json
{
  "schema": "rocke.ir/v1",
  "producer": { "name": "rocke_python", "version": "0.1" },
  "requires": { "min_rocke_ir": 1, "opcodes": ["arith.add", "func.ret"] },
  "target": { "arch_hint": "gfx950", "llvm_flavor_hint": "llvm22" },
  "kernel": { "name": "...", "attrs": {}, "params": [], "body": {} }
}
```

`target` is advisory and omitted unless a hint was passed. The lowerer takes the
arch and flavor from its caller, not from the artifact.

**Values.** Referenced by their existing SSA names (`%v17`, `%A`, `%k0`) rather
than by re-derived ids, so operands resolve by lookup on import and the imported
graph prints identically to the original.

**Types.** Scalars serialize as their canonical name string; composites as
objects:

```json
"f32"
{ "kind": "vector", "elem": "f16", "count": 4 }
{ "kind": "ptr",    "pointee": "f16", "space": "global" }
{ "kind": "smem",   "elem": "f16", "shape": [64, 32] }
```

**Attrs.** Always typed as `{"t": kind, "v": value}`, with kinds `i` (int64),
`f` (double), `b` (bool), `s` (string), `l` (list of nested attr maps), mirroring
`rocke_attr_kind_t`. The importer never infers a type from JSON syntax — an
untagged `1` is ambiguous between an int and a bool, and that ambiguity is
exactly the kind of drift this path exists to prevent. Attr keys are sorted for
diff-stable artifacts.

**Ops and regions.** An op carries `opcode`, `operands`, `results`, `attrs`, and
nested `regions`; a region carries a `label` and its `ops`.

```json
{
  "opcode": "memref.global_load_typed",
  "operands": ["%A", "%i"],
  "results": [{ "id": "%v5", "type": "f16" }],
  "attrs": { "align": { "t": "i", "v": 2 } },
  "regions": []
}
```

Params live in `kernel.params`, not as body ops, which keeps the launch ABI
checkable against the manifest without walking the graph.

**`loc` is dropped.** Source spans never reach the lowered `.ll` — the parity
harness verifies this — so omitting them keeps artifacts small and free of host
paths while staying byte-identical after lowering.

---

## 3. `rocke.recipe/v1` — builder program

A recipe has a `spec` (the inputs it specializes on), a `kernel_name_fmt`
interpolated with those inputs, kernel `attrs`, and a `program` of instructions.

```json
{
  "schema": "rocke.recipe/v1",
  "kernel_name_fmt": "rocke_recipe_toy_d{D}_{dtype}",
  "spec": [{ "name": "D", "kind": "int" }, { "name": "dtype", "kind": "str" }],
  "attrs": { "max_workgroup_size": { "t": "i", "v": 64 } },
  "program": [ /* instructions */ ]
}
```

The VM runs three environments: the spec inputs, integer registers (loop
induction variables and spec-derived integers), and IR-value registers holding
`rocke_value_t*`.

**Instructions** (each an object with `"op"`):

| instruction | fields | effect |
|---|---|---|
| `param` | `name`, `type`, `bind?`, `attrs?` | `rocke_b_param` |
| `const_i32` | `bind`, `val: <intexpr>` | `rocke_b_const_i32` |
| `const_f32` | `bind`, `fval` | `rocke_b_const_f32` |
| `thread_id_x` | `bind` | `rocke_b_thread_id_x` |
| `emit` | `opcode`, `in`, `out?`/`outs?`, `attrs?` | generic `rocke_b_op` |
| `alias` | `bind`, `from` | rebind a register |
| `static_for` | `var`, `lo`, `hi`, `step?`, `body` | compile-time loop |
| `static_if` | `pred: <intexpr>`, `then`, `else?` | compile-time branch |
| `scf_for` | `iv`, `lo`, `hi`, `step`, `iter`, `results`, `body` | runtime loop |
| `scf_if` | `cond`, `then` | runtime branch |
| `ret` | — | `rocke_b_ret` |

`static_for` and `static_if` are the compile-time layer: they run in the VM and
leave no trace in the emitted IR. `scf_for` and `scf_if` emit real control flow.

**Integer expressions.** Anywhere a size or constant appears:

```text
<intexpr> := number
           | {"spec": NAME} | {"var": NAME} | {"spec_str_eq": [NAME, literal]}
           | {"<OP>": [e, e]}   for OP in add sub mul div mod eq ne lt le gt ge
           | {"<FN>": e}        for FN in magic_multiplier magic_shift
```

Note the arity difference: binary ops take a 2-element array, the unary functions
take the operand directly.

`magic_multiplier` / `magic_shift` return the two operands of a strength-reduced
unsigned division — `n // d` compiled as `(umul_hi(n, M) + n) >> s`, which is what
`helpers/transforms.py::do_magic_division` emits. They exist because those two
integers are *not* arithmetic on the divisor: the shift is `ceil(log2 d)` and the
multiplier depends on `d`'s odd part, so a recipe that stays parametric in a
divisor has to regenerate them rather than carry a formula. The divisor must
satisfy `1 <= d < 2^31`. Three implementations must agree — the DSL helper that
emits them, `recipe_expand.py::magic_division_constants`, and `recipe_vm.cpp` —
and tests pin all three.

These reach constant values, integer attr values, and *type size fields* —
vector counts and smem shapes — so a recipe can size its LDS from the spec.

Because the ops nest, a value that is affine in *several* spec variables needs no
new grammar: it is nested binary `add`s over one term per axis, e.g.
`{"add": [{"mul": [{"spec": "S"}, 2]}, {"mul": [{"spec": "N"}, 8]}]}`. This is why
multi-axis rolling (`src/roll_nd.py`) required no VM change — the VM already
evaluates the tree, and multiple axes simply bind more `spec` entries.

**Parametric recipes** add two more moves. Register names may contain
`{var}`/`{spec}` tokens substituted at expansion (`"acc_m{lane}_n0"`), and
`scf_for` `iter`/`results` plus `emit` `in` lists may hold a rolled group
`{"for": {...}, "name": ..., "init"?: ...}` that expands to a spec-derived
*count* of entries. Together these let the loop-carry fan itself scale with the
spec.

**Exact SSA naming.** When `spec` is empty the recipe is *concrete*, and the VM
names each value verbatim from its bind instead of minting `%vN`. This is what
makes the recipe path byte-identical rather than merely isomorphic. It also
matters beyond cosmetics: a `tile.smem_alloc` result name becomes the LDS global
symbol, so the naming reaches the HSACO.

---

## 4. `rocke.bundle/v1` — many recipes, one file

```json
{
  "schema": "rocke.bundle/v1",
  "entries": [
    { "key": "kernel_name", "arch": "gfx950", "family": "...", "recipe": { } }
  ]
}
```

Lookup is by `key`, optionally narrowed by `arch` (a null arch matches any). The
runtime loads one blob and serves every kernel in it, with no per-recipe files.
Built by `recipe_bundle.build_bundle` / `write_bundle`.

---

## 5. Recording

Recipes are not hand-written. `RecordingIRBuilder`
(`portable_ir/src/recording_builder.py`) subclasses `IRBuilder` and captures the
calls a production builder makes, so `record_kernel(thunk)` returns both the
`KernelDef` and the recipe that reproduces it. `roller.py` then folds several
concrete traces into one parametric recipe by finding the structure that varies
with the spec — over one axis (`src/roll.py`) or over several axes' cross product
(`src/roll_nd.py`).

This is why the recorder needs its own coverage gate
(`drivers/record_coverage.py`): a builder that reaches an `IRBuilder` method the
recorder does not intercept would otherwise surface much later as a confusing
parity failure.

---

## 6. Consuming an artifact

**In-process**, over ctypes to the C ABI — no subprocess, no pybind:

```python
from rocke.portable_ir.src import online
ll, timings = online.recipe_cbor_to_llvm(cbor, arch="gfx950",
                                         ints={"D": 128}, strs={"dtype": "f32"})
```

**Standalone**, no Python in the process at all:

```text
rocke_portable_ir_replay_cli --recipe r.cbor --cbor --int D=128 --str dtype=f32
```

Both drive the same C++ core, so they emit identical output. See
[`../../tests/portable_ir/README.md`](../../tests/portable_ir/README.md).

---

## 7. Validation policy

The importer is strict: schema major mismatch, unknown opcode, unknown type or
attr kind, missing required field, forward operand reference, and duplicate SSA
id are all rejected. Unknown *optional* metadata is ignored. Strictness is worth
the friction because a malformed artifact that slips through fails much later,
inside LLVM or comgr, with far worse diagnostics.

One deliberate leniency: opcode resolution is dtype-aware. Python emits some
dtype-generic opcodes carrying an `elem_type` attr (`tile.buffer_load`), while
the C++ registry keys them by dtype (`tile.buffer_load_f16`). The importer and
the VM both try the exact name, then `name_elem_type`, then `name_f16`.

---

## 8. Gates

| gate | what it proves | where |
|---|---|---|
| DOM decoders | CBOR decodes to the same DOM as JSON; truncated input fails cleanly | `tests/portable_ir/dom_decoders.cpp` (ctest) |
| recipe VM | the spec drives structure; replay is deterministic | `tests/portable_ir/recipe_vm_replay.cpp` (ctest) |
| recorder coverage | every buildable production kernel records faithfully | `drivers/record_coverage.py` |
| parity matrix | replayed `.ll` is **byte-identical** to the Python lowerer's, for every kernel × arch, on both the import and recipe paths | `drivers/parity_matrix.py` |
| standalone | the same byte-identity through a binary with no Python in it | `tests/portable_ir/test_portable_ir.py` |
| device | replayed kernels compile and compute correctly on hardware | `drivers/gpu_replay.py` |

Byte-identity is the load-bearing one, and it is a deliberately stronger claim
than "the kernels agree numerically". Equivalent-but-differently-named IR would
pass a numeric check while hiding SSA-numbering and ordering drift — the same
defect class `ck.dsl.ir/v1` exists to catch on the engine seam.

The device gate splits author time from run time across a real file on disk, so
the runtime half cannot quietly depend on the Python IR stack. Elementwise
linear and non-transcendental unary kernels are gated bit-exact against numpy;
only the transcendentals get slack, because the GPU's `v_exp`/`v_tanh` are not
correctly rounded while the reference is.
