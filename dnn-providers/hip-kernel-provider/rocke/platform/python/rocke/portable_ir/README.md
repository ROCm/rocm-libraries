# Portable IR — record · roll · replay

This package turns a **Python-authored** CK-DSL kernel into a **compact, portable
artifact** that a **pure-C runtime** can re-emit and lower to a byte-identical
HSACO — with no CPython at JIT/serve time. It is the "author in Python, ship and
run without Python" path.

New here? Read **[Start here](#start-here-the-problem-and-the-trick)** and
**[What the artifacts actually look like](#what-the-artifacts-actually-look-like)**
first; they assume no prior knowledge of this package. The rollout strategy
(operator tiers, arch families, phasing) lives in
[`portable_ir_scaling_plan.md`](portable_ir_scaling_plan.md).

---

## Start here: the problem, and the trick

**The problem.** A kernel here is not a static file — it is the *output of a
Python program*. `build_universal_gemm(spec)` runs loops, calls helpers, does
descriptor arithmetic, and emits a few thousand IR instructions. To get a kernel
for a new shape you normally re-run that Python. That is fine in a dev tree and a
problem in a shipping runtime: it drags CPython, the whole `rocke` package, and
its import graph into your serving process, and it means a C++ inference runtime
cannot compile a kernel without embedding an interpreter.

**The trick, in two steps.**

*Step 1 — record.* Run the Python builder **once, unmodified**, and write down
the ops it emitted. That log is a **recipe**. Now a small C interpreter can
re-emit those ops and lower them, and Python is no longer needed. This is cheap,
works for any kernel, and needs no per-kernel code — but the recipe is
*concrete*: it describes one shape only.

*Step 2 — roll.* Record the builder at **two different shapes** and diff the two
logs. Where the second is the first with a block repeated more times, that block
becomes a loop; where a constant grew with the shape, it becomes a small formula.
The result is one **parametric** recipe that covers the whole family — and,
critically, values of the shape axis that were **never recorded**. This is the
*roller*, and it is the part that makes the artifact a compiler input rather than
a recording.

```
        ONE recorded trace  →  concrete recipe   → replays exactly that shape
        TWO recorded traces →  parametric recipe → replays the whole family,
                                                   held-out shapes included
```

**Why it is trustworthy.** The C side is not a reimplementation of the Python
lowerer — it is *the same* lowerer, already used as the production backend. So
the claim we test is not "close enough" but **byte-identical**: replaying an
artifact produces the same `.ll` text, character for character, and therefore
the same SHA-256, as running the Python builder. See
[Equivalence model](#equivalence-model-what-correct-means).

### Vocabulary

Terms used throughout, in plain language:

| Term | What it means here |
|---|---|
| **IR** | Intermediate representation: the kernel as a list of typed instructions (`%mul14 = mul i32 %tid7, 4`), before it becomes machine code. |
| **SSA** | Static Single Assignment: every value is written exactly once, so each instruction result gets its own name (`%tid7`). Why "names" come up so often below. |
| **KernelDef** | The in-memory Python object holding that instruction list. What a `build_*` function returns. |
| **Recipe** | The recorded log of ops. *Concrete* = one shape. *Parametric* = has a `spec` (free variables) plus loops/formulas, so it covers many shapes. |
| **Roll** | Turning several concrete recipes into one parametric recipe by finding the repetition. |
| **Replay** | Re-running a recipe through the C VM to rebuild the IR, then lowering it. |
| **JSON** | Human-readable text wire format. Used for debugging and for the concrete portable-IR graph. |
| **CBOR** | *Concise Binary Object Representation* (RFC 8949) — a binary format with the same data model as JSON (maps, arrays, strings, ints, bools). Same content, smaller and faster to parse, not human-readable. This is the shipping form. |
| **DOM** | *Document Object Model* — the decoded in-memory tree (`jd_val_t`: a tagged union of map/array/string/int/bool). Both the JSON and the CBOR decoder produce **the same DOM**, which is why the VM has exactly one implementation and does not care which wire format it was handed. |
| **Bundle** | One CBOR blob holding many recipes, looked up by `(key, arch)` — so a runtime opens one file instead of hundreds. |
| **HSACO** | *HSA Code Object*: the final compiled GPU binary that gets loaded and launched. |
| **comgr** | AMD's Code Object Manager — the library that compiles `.ll` text into an HSACO. The slow step in a JIT. |
| **`static_for`** | A **compile-time** loop *in the recipe*. The VM unrolls it while building, so it leaves no trace in the kernel — it is how one recipe emits a different number of instructions per shape. Distinct from `scf.for`, which is a **real loop in the generated kernel**. |
| **intexpr** | A small integer expression tree (`{"mul": [{"var": "_r0"}, 512]}`) the VM evaluates during replay. How a constant can depend on the spec or the loop variable. |

### The shape of it

```
            author (unchanged production builder)
                         │
                         ▼
                   KernelDef (Python SSA IR)
        ┌────────────────┼─────────────────────────────┐
        │ RECORD                                        │ serialize (concrete)
        ▼                                               ▼
  recipe (rocke.recipe/v1)                    portable IR (rocke.ir/v1)
  concrete, per-shape                          1:1 graph, per-shape
        │ ROLL (multi-trace)                            │
        ▼                                               │
  recipe (parametric)                                   │
  one artifact covers a family (static_for / intexpr)   │
        │                                               │
        │ pack                                          │
        ▼                                               │
  bundle (rocke.bundle/v1, CBOR)                        │
  many recipes keyed by (key, arch)                      │
        └───────────────┬───────────────────────────────┘
                        ▼  REPLAY (pure C, no CPython)
        ┌───────────────────────────────────────────────┐
        │ recipe VM (recipe_vm.cpp) | IR import (ir_import_json.cpp)│
        │            → rocke_lower_kernel_to_llvm → comgr → HSACO     │
        └───────────────────────────────────────────────┘
```

**Record is universal and ~free** (it just serializes the emitted op stream).
**Roll is the optional win**: it buys shape coverage, and compresses storage as a
side effect. The C engine is the *same* lowerer the production engine uses, so
output is **byte-identical**.

## Three artifacts

| Artifact | Schema | Shape | Emitter (Python) | Consumer (C) |
|---|---|---|---|---|
| **Portable IR** | `rocke.ir/v1` | concrete 1:1 graph | `rocke.core.ir_export` | `ir_import_json.cpp` (`rocke_import_kernel_from_json`) |
| **Recipe** | `rocke.recipe/v1` | concrete *or* parametric | `src/recording_builder.py`, `src/roll.py` | `recipe_vm.cpp` (`rocke_recipe_run_from_json` / `_cbor`) |
| **Bundle** | `rocke.bundle/v1` | many recipes by `(key, arch)` | `src/recipe_bundle.py` | `recipe_vm.cpp` (`rocke_recipe_run_from_bundle_cbor`) |

Portable IR is a *graph* (what the kernel is). A recipe is a *program that
rebuilds the graph* (how the kernel was constructed) — which is what makes the
parametric form possible: you cannot parameterize a finished graph over shape,
but you can parameterize the builder that produced it.

## What the artifacts actually look like

All snippets below are real output, from
`fused_moe_gather` (`drivers/roll_hsaco_parity.py::_moe`).

### 1. Recording a kernel

```python
from rocke.portable_ir.src.recording_builder import record_kernel
from rocke.instances.common.fused_moe import FusedMoeSpec, build_moe_gather

kernel, recipe = record_kernel(lambda: build_moe_gather(spec, arch="gfx950"))
```

Note what is *not* there: no changes to `build_moe_gather`, no annotations, no
registration. `record_kernel` swaps in a recording subclass of `IRBuilder`, runs
the builder untouched, and hands back both the normal `KernelDef` and the recipe.

### 2. A concrete recipe (`rocke.recipe/v1`)

Header — `spec: []` is what makes it *concrete* (no free variables):

```json
{
  "schema": "rocke.recipe/v1",
  "kernel_name_fmt": "fused_moe_gather_gather_T32_E8_K2_H1024_I256_f16_b128_v4",
  "spec": [],
  "attrs": {"max_workgroup_size": {"t": "i", "v": 128}},
  "program": [ ... ]
}
```

The `program` is a flat list of instructions. Kernel arguments first:

```json
{
  "op": "param",
  "name": "X",
  "type": {"kind": "ptr", "pointee": "f16", "space": "global"},
  "bind": "X",
  "attrs": {"noalias": true, "readonly": true, "align": 16}
}
```

then the ops, in emission order:

```json
{
  "op": "emit",
  "opcode": "gpu.thread_id",
  "in": [],
  "out": {"bind": "tid2", "type": "i32", "pfx": "tid"},
  "attrs": {"axis": {"t": "s", "v": "x"}}
}
```

Reading that instruction: emit a `gpu.thread_id` op, no operands, one result of
type `i32`. Three fields carry naming:

- **`bind`** — the name later instructions use to refer to this result. In a
  concrete recipe it is also Python's actual SSA name (`%tid2`), so the VM can
  reproduce Python's IR text verbatim.
- **`pfx`** — the *prefix* Python used to mint that name. Needed only for rolled
  recipes: there, one instruction expands many times, so every expansion must
  draw a fresh name, and the VM regenerates `%tid<counter>` from the prefix
  rather than reusing the bind. Recording the prefix (instead of mirroring
  Python's ~38-entry prefix table in C++) is what keeps the two engines from
  drifting as ops are added.
- **`attrs`** — typed op attributes; `{"t": "i" | "s" | "b", "v": ...}` tags each
  value's type so the wire form is unambiguous in both JSON and CBOR.

### 3. A parametric recipe, after rolling

```python
from rocke.portable_ir.src.roll import roll

r = roll(build_at=lambda v: build_moe(hidden=v), axis="hidden",
         sample_points=[512, 1024])
assert r.ok           # else r.reason says why it declined
recipe = r.recipe
```

Two things changed. The header now declares a free variable, and the kernel name
became a format string:

```json
{
  "spec": [{"name": "hidden", "kind": "int"}],
  "kernel_name_fmt": "fused_moe_gather_gather_T32_E8_K2_H{hidden}_I256_f16_b128_v4"
}
```

And the repeated block became a compile-time loop whose trip count is a formula
in `hidden`:

```json
{
  "op": "static_for",
  "var": "_r0",
  "lo": 0,
  "hi": {"div": [{"spec": "hidden"}, 512]},
  "step": 1,
  "body": [
    {
      "op": "emit",
      "opcode": "arith.constant",
      "in": [],
      "out": {"bind": "c19", "type": "i32", "pfx": "c"},
      "attrs": {
        "ity": {"t": "s", "v": "i32"},
        "value": {"t": "i", "v": {"mul": [{"var": "_r0"}, 512]}}
      }
    },
    ...
  ]
}
```

That is the whole idea of rolling in one object. The roller observed the block
once at `hidden=512` and twice at `hidden=1024`, inferred the trip count
`hidden/512`, and noticed the constant inside was `0` then `0, 512` — so it wrote
the **intexpr** `_r0 * 512`. Replay at `hidden=4096` therefore runs the body 8
times with constants `0, 512, … 3584`, a case never recorded.

`static_for` **disappears** during replay — it is the VM's `for` loop, not the
kernel's. A loop you want in the finished kernel is `scf.for`, recorded as its
own instruction with a body region.

### 4. CBOR: the shipping form

CBOR is JSON's data model in binary. Encoding is one call, and it round-trips
exactly:

```python
from rocke.portable_ir.src import recipe_bundle

blob = recipe_bundle.cbor_encode(recipe)
assert recipe_bundle.cbor_decode(blob) == recipe    # exact round-trip
```

The first bytes of a bundle, with printable characters shown underneath — you can
see the structure is the same map-of-keys as the JSON, just with lengths in place
of punctuation:

```
a2 66 73 63 68 65 6d 61 6f 72 6f 63 6b 65 2e 62 75 6e 64 6c 65 2f 76 31 67 65 6e 74 72 69 65 73
.  f  s  c  h  e  m  a  o  r  o  c  k  e  .  b  u  n  d  l  e  /  v  1  g  e  n  t  r  i  e  s
```

`a2` = "map with 2 pairs"; `66` = "6-byte string" → `schema`; `6f` = "15-byte
string" → `rocke.bundle/v1`; `67` = "7-byte string" → `entries`. No quotes,
colons, or whitespace to scan.

Size, for the rolled MoE recipe:

| Form | Size | vs CBOR |
|---|---|---|
| CBOR | 2.5 KiB | — |
| JSON, compact | 3.5 KiB | 1.4× |
| JSON, indented | 8.3 KiB | 3.4× |

The size win is real but secondary; the reason CBOR is the shipping form is that
it parses with no allocator churn and no number/string re-parsing. **Both
decoders produce the same DOM**, so `recipe_vm.cpp` has one code path — you can
debug in JSON and ship in CBOR with no behavioral difference.

### 5. A bundle: many recipes, one file

```python
blob = recipe_bundle.cbor_encode(recipe_bundle.build_bundle([
    {"key": "fused_moe_gather", "arch": "gfx950", "family": "moe",  "recipe": moe_recipe},
    {"key": "gemm_universal",   "arch": "gfx950", "family": "gemm", "recipe": gemm_recipe},
]))
```

```json
{
  "schema": "rocke.bundle/v1",
  "entries": [
    {"key": "fused_moe_gather", "arch": "gfx950", "family": "moe",  "recipe": {...}},
    {"key": "gemm_universal",   "arch": "gfx950", "family": "gemm", "recipe": {...}}
  ]
}
```

The runtime maps this once and serves by `(key, arch)`, so adding a kernel does
not add a file to open.

### 6. Portable IR (`rocke.ir/v1`), for contrast

The concrete graph, exported straight from a `KernelDef`:

```python
from rocke.core import ir_export
open("k.ir.json", "w").write(ir_export.export_kernel_ir_json(kernel))
```

Same information as a concrete recipe, expressed as a finished graph rather than
a build program. Useful when you want a plain, inspectable dump of a single
kernel; it cannot be parameterized over shape.

## Using a parametric recipe from the native C stack (JIT)

This is the payoff: a C or C++ runtime compiles a kernel for a shape it has never
seen, with **no CPython in the process**. Python was needed to *author* the
bundle; nothing at run time links against it.

### The flow

```
  ship once:   bundle.cbor   (built offline by Python: record → roll → pack)

  at runtime, per request:
    (1) shape arrives                      e.g. hidden = 4096
    (2) cache lookup on (key, arch, spec)  hit  → launch, done
                                           miss ↓
    (3) rocke_recipe_run_from_bundle_cbor  CBOR → DOM → VM replays the builder,
                                           expanding static_for / intexpr at
                                           hidden=4096            → KernelDef
    (4) rocke_lower_kernel_to_llvm_ex      KernelDef              → .ll text
    (5) comgr                              .ll                    → HSACO
    (6) cache insert, then launch          kernel->name is the symbol to look up
```

Steps 3–4 are the C engine and are fast (single-digit milliseconds even for the
attention kernel). Step 5 dominates, which is why the artifact-level cache in
step 2 is the thing that matters for serving latency.

### The call

The whole replay is two calls. `spec` values arrive as plain
`{name, value}` pairs — this is where `hidden = 4096` enters:

```c
#include "rocke/recipe_vm.h"
#include "rocke/lower_llvm.h"

const rocke_recipe_spec_int_t ints[] = {{"hidden", 4096}};

rocke_ir_builder_t     b;
rocke_kernel_def_t*    kernel = NULL;
char                   err[ROCKE_ERR_MSG_CAP] = {0};

/* (3) pick the recipe out of the bundle and re-run the builder at this shape. */
rocke_status_t st = rocke_recipe_run_from_bundle_cbor(
    bundle_bytes, bundle_len,
    /* key  */ "fused_moe_gather",
    /* arch */ "gfx950",
    ints, 1, /* strs */ NULL, 0,
    &b, &kernel, err, sizeof err);
if (st != ROCKE_OK) { /* err holds a human-readable reason */ }

/* (4) same lowerer the production backend uses. */
char* ll = NULL;
st = rocke_lower_kernel_to_llvm_ex(kernel, ROCKE_LLVM_FLAVOR_LLVM22, "gfx950",
                                   &ll, err, sizeof err);

/* (5) hand `ll` to comgr; kernel->name is the resulting symbol name. */
/* ... */

free(ll);                      /* rocke_online_free() for the online.h wrappers */
rocke_ir_builder_free(&b);     /* frees the arena: every IR node at once */
```

Notes that matter in practice:

- **`kernel->name`** is the resolved `kernel_name_fmt` — `{hidden}` already
  substituted — so it is both your cache key component and the symbol to look up
  in the HSACO.
- **Lifetime is an arena.** Every node the VM allocated lives in `b`; one
  `rocke_ir_builder_free(&b)` releases the lot. There is nothing per-node to
  track.
- **Errors are strings, not aborts.** A malformed recipe, an unknown opcode, or a
  missing spec value returns non-`ROCKE_OK` and fills `err`.
- **Flavor must match.** The `.ll` datalayout is LLVM-generation specific; lower
  with the flavor matching the comgr you will compile with, or you get a
  mismatch on the first line.

If you just want `.ll` and would rather not manage the builder, `rocke/online.h`
collapses steps 3–4 into one call (`rocke_online_bundle_cbor_to_llvm`, plus
`_recipe_cbor_` and `_ir_json_` variants). That is also what `src/online.py`
binds over ctypes, and what the parity drivers use.

### Trying it without writing C

`tests/portable_ir/replay_cli.cpp` is exactly this flow as a standalone binary —
no Python linked, no interpreter initialized:

```bash
cmake --build <build> --target rocke_portable_ir_replay_cli

# replay a PARAMETRIC recipe at a shape it was never recorded at
./rocke_portable_ir_replay_cli --recipe gemm.recipe.cbor --cbor \
    --int tile_n=256 --arch gfx950 --flavor llvm22 > jit.ll

# and the same claim, from a bundle
./rocke_portable_ir_replay_cli --bundle bundle.cbor --key fused_moe_gather \
    --int hidden=4096 --arch gfx950 --flavor llvm22 > jit.ll
```

`tests/portable_ir/test_recipe_roller.py` runs that binary against the Python
lowerer and asserts the two `.ll` files have the same SHA-256, at sampled *and*
held-out axis values.

## Directory layout

```
portable_ir/
├── src/            core engine + runtime binding
│   ├── recording_builder.py   RecordingIRBuilder + record_kernel (the recorder)
│   ├── kerneldef_to_recipe.py KernelDef → concrete recipe (post-hoc walk)
│   ├── recipe_recorder.py     idiomatic parametric authoring surface
│   ├── roller.py              multi-trace structural roller
│   ├── roll.py                roll(build_at, axis, …) driver (records + verifies)
│   ├── recipe_bundle.py       CBOR codec + bundle (rocke.bundle/v1)
│   └── online.py              ctypes binding to the C backend (recipe/IR → .ll)
├── utils/
│   └── recipe_expand.py       pure-Python recipe expander + recipes_equiv (oracle)
├── examples/       runnable demo kernels (--emit recipe|ll|name)
│   ├── recipe_toy.py  mini_attn.py  qk_block.py
│   ├── export_mha.py  export_gemm_cshuffle.py  recipe_multi_result.py
├── drivers/        runnable harnesses / benchmarks
│   ├── record_coverage.py         recorder coverage over the parity emitter set
│   ├── roll_coverage.py           tiered rolling coverage
│   ├── verify_recording_production.py
│   ├── roll_recipe.py             land-#2 attention rolling demo
│   ├── bench_online.py            compile-timeline benchmark
│   ├── parity_matrix.py           concrete-path .ll parity gate (all kernels × arches)
│   ├── hsaco_parity.py            concrete-path HSACO byte-identity gate
│   └── roll_hsaco_parity.py       rolled-path .ll sha + HSACO gate, incl. held-out
├── tests/          unittest suites (recorder drift, roller, CBOR/bundle)
└── portable_ir_scaling_plan.md
```

The C++ side lives in `platform/cpp/portable_ir/` (C++20, part of
`librocke_core.a`; see that dir's `README.md`):
- `recipe_vm.cpp` (+ `rocke/recipe_vm.h`) — the recipe VM.
- `ir_import_json.cpp` (+ `rocke/ir_import.h`) — the portable-IR importer.
- `cbor_dom.cpp`, `json_dom.cpp` — the two DOM decoders.
- `online.cpp` (+ `rocke/online.h`) — one-call wrappers (recipe/bundle/IR → `.ll`).

Its ctests, the pytest harnesses, and the standalone `replay_cli` are in
`platform/tests/portable_ir/`. The wire schemas are specified in
`dsl_docs/architecture/portable_ir_schema.md`.

## Record architecture

`RecordingIRBuilder` subclasses `core.ir.IRBuilder` and intercepts the **single op
choke point** (`_emit`) plus `param`, `push_region`/`pop_region`, and `_op` (for
the result-name prefix), recording each op into a recipe *as the kernel is built*.
Because it rides `_emit` (not the public op-builder methods), **new ops are
captured automatically**.

`record_kernel(build_fn)` temporarily rebinds the `IRBuilder` name across every
imported `rocke` module, runs the **unmodified** production builder, and returns
`(kernel, recipe)`. Helpers/closures/dataclass/descriptor math just execute; only
emitted ops are captured. So any `build_*` records with **zero kernel changes**.

An independent post-hoc walk of the finished `KernelDef`
(`kerneldef_to_recipe.py`) must produce the same recipe as the live recording;
that comparison is a test, and it is what catches a recorder that silently drops
or reorders ops.

## Replay paths

1. **Python oracle** (`utils/recipe_expand.py`): `expand_recipe(recipe, spec)` +
   `recipes_equiv` — device-free structural check that a rolled recipe expands to
   the recorded concrete recipe at sampled *and* held-out points.
2. **Engine import** (`rocke_import_kernel_from_json`): concrete portable IR → C
   builder → C lower. Byte-identical `.ll` to the Python lowerer (name hints
   survive `ir_export`).
3. **Recipe VM** (`recipe_vm.cpp`): concrete or parametric recipe → C build (with
   `static_for`/intexpr expansion) → C lower. Runs on JSON or CBOR; serves from a
   bundle by `(key, arch)`.
4. **Online, in-process** (`src/online.py`): ctypes into `online.cpp` — hand a
   CBOR recipe/bundle or IR-JSON and get `.ll` back, no subprocess, no pybind.

## Equivalence model (what "correct" means)

- Both replay paths produce `.ll` **byte-identical** to the native Python lowerer
  (with one LLVM flavor pinned on every path), so a **SHA-256 of the `.ll` is a
  sufficient gate** and no compile is needed to compare.
- This holds for **rolled** recipes too, not just concrete ones. A concrete
  recipe replays Python's SSA names verbatim from its binds; a rolled recipe
  cannot (each instruction expands many times, so every expansion must draw a
  fresh name) but reproduces them anyway, because the recipe carries each op's
  name prefix and the roller keeps Python's positional naming for loop-carry
  fans. The one documented exception is a fan whose names are not simply the lane
  index, which stays alpha-equivalent (identical after renaming).
- **HSACO byte-identity** is also gated, as the stronger artifact-level check. It
  is always a **same-toolchain differential** (compare both paths compiled by the
  *same* comgr), never a stored golden across ROCm versions.
- Every gate includes **held-out** axis values, and the negative control is
  checked: replaying at the wrong spec value must differ. Otherwise an all-pass
  result would not distinguish a working roller from a vacuous comparison.

## Running things

Everything below runs from `platform/` with the engine importable:

```bash
export PYTHONPATH="$PWD/python:$PWD/../library${PYTHONPATH:+:$PYTHONPATH}"

# unit tests + recorder coverage (pure Python, no engine binary)
python3 -m unittest discover -s python/rocke/portable_ir/tests
python3 -m rocke.portable_ir.drivers.record_coverage

# concrete path: both replay paths vs the Python lowerer, byte-identical .ll,
# every kernel x arch. Device-free, needs a shared librocke.
export ROCKE_ONLINE_LIB=<path>/librocke.so
python3 -m rocke.portable_ir.drivers.parity_matrix [--arches gfx942,gfx950]
python3 -m rocke.portable_ir.drivers.hsaco_parity        # ... and their HSACO

# rolled path: record 2 traces -> roll -> replay. Same .ll sha at sampled AND
# held-out axis values. --no-hsaco stops at .ll (no comgr, ~2s).
python3 -m rocke.portable_ir.drivers.roll_hsaco_parity [--no-hsaco]

# on-device: record -> CBOR -> C replay -> comgr -> launch -> check numerics
python3 -m rocke.portable_ir.drivers.gpu_replay --device 0 --verbose

# online in-process lowering smoke
python3 -m rocke.portable_ir.src.online
```

Under pytest, the same gates run from `tests/portable_ir/`:
`test_portable_ir.py` (concrete path) and `test_recipe_roller.py` (rolled path,
including the standalone-binary lane). Both skip the engine-binary lanes with an
actionable reason — never a silent pass — when no `librocke.so`, replay CLI, or
comgr is available.

## When does a new kernel need code here?

- **Concrete / CPython-free path:** never — it records and lowers automatically.
- **Rolling (shape coverage):** usually just declare the structural spec axes; a
  genuinely new structural-variation pattern needs a one-time roller extension
  (then amortized across all future kernels of that pattern). When the roller
  cannot prove a pattern it **declines** and says why, and the concrete path
  still works — a refusal costs coverage, never correctness.
- **Brand-new IRBuilder op:** captured generically, but the C side must know how to
  lower it (work owed to any C-JIT backend; the byte-identical oracle catches a
  missing/wrong lowering). Region-bearing ops beyond `scf.for`/`scf.if` make the
  recorder raise a loud "extend me".

See [`portable_ir_scaling_plan.md`](portable_ir_scaling_plan.md) for the full
status, caveats, and rollout plan.
