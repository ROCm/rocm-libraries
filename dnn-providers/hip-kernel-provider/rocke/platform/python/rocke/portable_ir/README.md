# Portable IR — record · roll · replay

This package turns a **Python-authored** rocKE kernel into a **compact, portable
artifact** that a **pure-C runtime** can re-emit and lower to a byte-identical
HSACO — with no CPython at JIT/serve time. It is the "author in Python, ship and
run without Python" path.

New here? Read **[Start here](#start-here-the-problem-and-the-trick)** and
**[What the artifacts actually look like](#what-the-artifacts-actually-look-like)**
first; they assume no prior knowledge of this package. The rollout strategy
(operator tiers, arch families, phasing) lives in
[`portable_ir_scaling_plan.md`](portable_ir_scaling_plan.md). What a non-Python
caller still needs before it can drive this path — data-driven spec validity,
structured recipe keys, catalog pruning — is in
[`hipdnn_jit_integration.md`](hipdnn_jit_integration.md).

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
      1+Σ(nⱼ-1)     traces →  parametric recipe → replays the CROSS PRODUCT of
       (one probe per axis)                        several axes (§3b)
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
| **Roll** | Turning several concrete recipes into one parametric recipe by finding the repetition. Over one axis (`roll`) or several at once (`roll_nd`). |
| **Axis** | A shape or tuning dimension the recipe is parametric in (`hidden`, `tile_n`, `N`). A *structural* axis changes which instructions are emitted; a *constants-only* axis leaves the instruction sequence alone and moves only the numbers in it. |
| **Replay** | Re-running a recipe through the C VM to rebuild the IR, then lowering it. |
| **JSON** | Human-readable text wire format. Used for debugging and for the concrete portable-IR graph. |
| **CBOR** | *Concise Binary Object Representation* (RFC 8949) — a binary format with the same data model as JSON (maps, arrays, strings, ints, bools). Same content, smaller and faster to parse, not human-readable. This is the shipping form. |
| **DOM** | *Document Object Model* — the decoded in-memory tree (`jd_val_t`: a tagged union of map/array/string/int/bool). Both the JSON and the CBOR decoder produce **the same DOM**, which is why the VM has exactly one implementation and does not care which wire format it was handed. |
| **Bundle** | One CBOR blob holding many recipes, looked up by `(key, arch)` — so a runtime opens one file instead of hundreds. |
| **HSACO** | *HSA Code Object*: the final compiled GPU binary that gets loaded and launched. |
| **comgr** | AMD's Code Object Manager — the library that compiles `.ll` text into an HSACO. The slow step in a JIT. |
| **`static_for`** | A **compile-time** loop *in the recipe*. The VM unrolls it while building, so it leaves no trace in the kernel — it is how one recipe emits a different number of instructions per shape. Distinct from `scf.for`, which is a **real loop in the generated kernel**. |
| **intexpr** | A small integer expression tree (`{"mul": [{"var": "_r0"}, 512]}`) the VM evaluates during replay. How a constant can depend on the spec or the loop variable. |

And the rolling terms, which is the vocabulary the refusal messages are written in:

| Term | What it means here |
|---|---|
| **Trace** | One recording of the builder at one point on an axis. Rolling is fundamentally a diff of two traces; everything after that is verification. |
| **Probe** vs **holdout** | A probe is a point recorded in order to *infer* the model. A holdout is a point the finished model is *checked* at but was never fitted to — so only holdouts can catch a model that merely memorized its inputs. |
| **Level** | One nesting depth of the instruction list being compared: the top-level program, or the body of one loop inside it. Alignment works a level at a time and descends into each body, which is why refusals say "no run **at level**" — and why they quote `\|la\|` and `\|lb\|`, the lengths of the two op lists *at that level* rather than of the whole kernel. |
| **Signature** | An op's fingerprint with SSA names and integer *values* stripped out, so two ops match when they do the same thing to differently-named inputs. Runs are found by comparing signatures, which is exactly why a constant that changed between traces does not prevent the match — it gets fitted afterwards. |
| **Run** | A block of ops repeated several times in a row: the thing that becomes a loop. Found at the point where the two traces first disagree, by looking for a repeating signature pattern around it. |
| **Period** (`L`) | The length of one copy of a run's block, in ops. A run of period 7 repeated 3 times spans 21 ops. Several periods can fit the same stretch (a coincidental short one, and the true unroll body), so candidates are tried largest-first. |
| **Trip count** | How many times a run repeats. When it differs between the two traces it becomes a formula in the axis — which is the entire point, since that is what lets one recipe emit a different amount of code per shape. |
| **Loop carry**, **fan**, **lane** | The values a *real* kernel loop (`scf.for`) hands from each iteration to the next: accumulators, advancing addresses. The roller splits them into individually named **lanes** (collectively a **fan**) because each has to be matched up across the two traces separately. Attention's KV loop carries 10 of them at `block_n = 64` and 14 at 128, which is why that axis does not roll. |
| **Slot** | A single integer position somewhere in the recipe tree that might depend on the axis — an offset, a size, a shape entry. `fit_slot` takes that slot's recorded values and either returns a formula or refuses. |
| **Affine** | The straight-line model `c0 + m·x`: a constant plus a fixed step per unit of the axis. It covers most of what a kernel *computes* from a shape, because address arithmetic is affine, and it is the first candidate tried. |
| **Cross term** | A constant that scales with *two* axes multiplied together (`m·N·K`). Invisible to any probe that moves one axis at a time, so it needs a point where two axes move together. |
| **Regime** | A stretch of an axis over which the structure stays uniform, so one recipe can cover it. An axis that switches code path past a threshold has two regimes and needs two recipes; `roll_regimes` finds the boundaries by verification rather than being told them. |
| **Refusal** | Returning "no recipe" plus a reason, instead of a recipe that might be wrong. A refusal is a normal outcome and costs only compression — the caller keeps concrete per-point recipes — so the roller is built to refuse readily. |
| **Structural roller** vs **constant solver** | The two machines rolling is made of. The constant solver keeps the instruction sequence fixed and turns the integers inside it into formulas; the structural roller changes the sequence itself, folding runs into `static_for` so one recipe emits a different amount of code per shape. `roll` always runs both; `roll_nd` runs the constant solver on every axis and the structural roller only on the axis named by `structural_axis=`. [§3c](#3c-the-structural-roller-and-the-simpler-one-next-to-it) walks through its six steps and its refusals. |
| **`roll_two`** | The engine under everything: given two concrete traces and their axis values, return one parametric recipe, or `None` with `last_reason` set. `roll` and `roll_nd` are both thin drivers over it. |
| **Annotate-then-roll** | How `roll_nd` handles several axes: first replace every axis-dependent constant with a formula (**annotate**), then run the structural roller over traces that are already parametric. Doing it in that order keeps the two problems — which numbers moved, and which code repeated — from having to be solved at once. |
| **Oracle** | The check that decides correctness: expand the parametric recipe at a point and compare it against a fresh recording of the real builder (`recipes_equiv`). Byte-identity, not similarity. |

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

### Getting from a shipped bundle to a launch

The pure-C path used to stop at `.ll`. A client could take a CBOR bundle to a
correct kernel with no Python in the process, then be stuck: it had a HSACO and
no idea what to launch it with, because the grid was never in the bundle. It
lived in host Python, as expressions like `(n + tile_n - 1) // tile_n` inside a
dispatch function — so the last step of the chain was the one step that needed
an interpreter.

A grid is a function of the shape, which is what the recipe language already
exists to express, so geometry is carried as intexprs over the spec axes and
evaluated by the same evaluator as every loop bound the recipe emits:

```json
"launch": {
  "grid":  [{"div": [{"add": [{"spec": "N"}, 2047]}, 2048]}, 1, 1],
  "block": [256, 1, 1],
  "lds_bytes": 0
}
```

`rocke/recipe_launch.h` turns that plus the recipe's own `param` declarations
into everything a launch needs — name, kernarg offsets, grid, block, dynamic
LDS. Offsets follow the AMDGPU natural-alignment rule, which only becomes
visible once a signature mixes widths: `(ptr, i32, ptr)` puts its last pointer
at 16, not 12.

`drivers/launch_from_bundle.py` runs the whole chain on real hardware and is
deliberately forbidden from importing anything from the kernel family that
authored the recipe — if the bundle did not carry enough to launch, it could
not run:

```text
N=2049     grid=(2, 1, 1) block=(256, 1, 1) kernarg=28B  OK
N=100000   grid=(49, 1, 1) block=(256, 1, 1) kernarg=28B  OK
```

Still missing in C, and worth knowing before planning against this: `.ll` →
HSACO (comgr) and HSACO → launch exist only as the Python ctypes wrappers in
`rocke/runtime/`. Both are thin bindings over `libamd_comgr` and `libamdhip64`,
which a C++ JIT client such as hipDNN already links for itself. Note also that
`tests/instances/jit_demo.cpp` advertises a complete C++ chain but references a
`rocke::Compiler` that does not exist in this tree and is not built.

### Keeping the two engines compatible

A bundle is written by Python at build time and read by C inside hipDNN, which
may have been built at some other time in either direction. Two things can be
mismatched, and they get separate numbers (`cpp/include/rocke/abi.h`,
`src/abi.py`) because folding them together would mean a new recipe instruction
invalidates every hipDNN binary, and a struct change invalidates every bundle
on disk — neither of which is true:

| | Question | Where checked |
|---|---|---|
| `ROCKE_ABI_VERSION` | Does this header match this `.so`? Structs, enums, signatures. | Once at load — `online.load()`, and hipDNN's own loader |
| `ROCKE_RECIPE_ABI` | Can this engine read this CBOR artifact? | Per artifact, in both readers |

The wire check is **not** "artifact version == mine". Each artifact declares the
*oldest reader that can read it correctly*, and a reader refuses exactly when
`min_reader` exceeds its own level:

```json
"abi": {"min_reader": 1, "writer": 1, "engine": "1.0.0+20260812", "build_id": "6bc59f33fd11"}
```

`writer`, `engine` and `build_id` are provenance for tracing a bad artifact.
Nothing compares them; only `min_reader` decides. A plain monotonic version
compared for equality would reject newer artifacts wholesale, whether or not
they use anything new — turning a generator upgrade into a flag day for every
deployed engine, over recipes it has always been able to read. A **missing**
block means level 1, so bundles recorded before this existed still replay.

`min_reader` is *derived* from what the recipe uses, never hand-set: a declared
requirement is a second copy of the truth and drifts the first time someone
forgets. Note what that does and does not buy. Both VMs already fail loudly on
an unknown instruction op, opcode or intexpr node, so a new construct is
self-policing and the stamp only improves the error message. The bump exists for
changes an old engine would *accept and get wrong* — a changed default, a
reinterpreted field. Attribute **values** are passed through to the builder
uninterpreted, so their meaning is the lowerer's contract and is outside what
this number can police.

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

### 3b. Several axes at once

`roll` moves one axis, so covering two of them costs two recipes and neither moves
with the other. `roll_nd` covers the whole cross product with one recipe:

```python
from rocke.portable_ir.src.roll_nd import roll_nd, roll_nd_report

r = roll_nd(build_conv,                      # called as build_conv(N=…, K=…)
            axes={"N": [8, 16], "K": [64, 128]},
            holdout_points=[{"N": 64, "K": 512}])
print(roll_nd_report(r))
# rolled: 1 recipe (1074 ops) covers 5 points from 3 recorded traces
#         (concrete total=5370 ops; 5.0x)
```

The header now declares both axes, and each integer field is solved as an affine
function of all of them — here conv's address math yields `{"mul": [{"spec": "K"},
54]}` and `{"mul": [{"spec": "N"}, 2916]}` among 33 such expressions:

```json
{"spec": [{"name": "N", "kind": "int"}, {"name": "K", "kind": "int"}],
 "kernel_name_fmt": "conv_K{K}_N{N}_N{N}H56W56C64_K{K}Y3X3_t32x32x32_w1x1_a16x16x16_mem_default"}
```

Note what carries the cross product: in this kernel **no single constant depends on
both axes** — different constants track different axes, and one recipe covers the
grid because all of them are solved together. A field genuinely scaling with `N*K`
would be *refused*, not fitted (see below). Fields that do combine axes appear when
a constants-only axis meets a structural one, e.g.
`{"add": [{"mul": [{"spec": "S"}, 2]}, {"mul": [{"spec": "N"}, 8]}]}`.

Read the `5.0x` as an **artifact-count** win, not compression: the recipe has
exactly the concrete op count (1074), because these axes move only constants. One
file now serves five shapes; it is not smaller than one of them. Axes that
restructure (`static_for`) are what shrink the op count.

Replay binds every axis — `--int N=64 --int K=512`, or `spec_int` entries in C.
Nothing in the VM changed to support this: a multi-axis constant is still an
intexpr over spec values.

### 3c. The structural roller, and the simpler one next to it

The two examples above were produced by two different machines. Knowing which one
is running explains most of what the roller says, so it is worth separating them.

**The constant solver** leaves the instruction sequence alone. Same ops, in the
same order, in the same number, at every value of the axis — only the integers
inside them move, and each one becomes a formula. §3b is entirely this. It is fast,
it fails cleanly (a constant either matches a candidate model at every recorded
point or it does not), and it fundamentally cannot describe a kernel that emits
*more code* at a bigger shape.

**The structural roller** changes the sequence itself. It is what turned a repeated
block into the `static_for` in §3 — one recipe emitting one copy at `hidden=512`
and eight at `hidden=4096`. This is the part that makes a recipe a compiler input
rather than a recording, and it is where nearly all the difficulty lives.

What it does, in order:

1. **Align, one level at a time.** Compare the two traces' instruction lists by
   *signature* (names and integer values stripped). An identical prefix is copied
   through; the first disagreement is where the work starts. Each loop body is its
   own level and is handled the same way, recursively.
2. **Find a run.** Around the disagreement, look for a block whose signature
   pattern repeats. Its length in ops is the *period*; count how many copies each
   trace has. Several periods can fit the same stretch — a short coincidental one
   and the true unroll body — so candidates are tried largest first.
3. **Solve the trip count** as a formula in the axis. One copy at 512 and two at
   1024 gives `hidden/512`.
4. **Learn what each iteration hands to the next** by diffing copy 0 against copy
   1: the loop-carried values, and the constants that step per iteration. This is
   the step that makes two copies the minimum evidence, and it is why a trip count
   of the form `n-1` is a trap.
5. **Fan out real loop carries.** If the run contains an `scf.for`, its iter-args
   are split into individually named *lanes* so each can be matched across the two
   traces on its own.
6. **Recurse on the remainder**, so a kernel with ten separately growing regions
   ends up with ten `static_for`s rather than one confused attempt.

**How to ask for it.** `roll` always runs it — a single-axis roll is structural by
default. `roll_nd` is the other way round: it runs the constant solver on *every*
axis and the structural roller only on the one axis you name with
`structural_axis=`. That asymmetry matters in practice, because it means a
`not constants-only` refusal has not yet been offered to the structural roller at
all. Three axes in `kernels/gfx950` were recovered exactly that way (see
[the sweep](#measured-every-kernel-in-kernelsgfx950)).

**How it declines.** Every message below is a refusal, which is to say a safe
outcome — the caller keeps concrete per-shape recipes. They are worth telling apart
because they call for different responses:

| Message | What it means | Usual response |
|---|---|---|
| `not constants-only: program: A instructions at base vs B` | The op *count* moved, so the constant solver is the wrong tool | name the axis as `structural_axis=` |
| `no run at level (\|la\|=… \|lb\|=…)` | The lists diverge but no repeating block was found around the divergence | check the axis really repeats something |
| `shorter-at-larger-axis (\|la\| > \|lb\|)` | *Fewer* ops recorded at the larger value; rolling looks for more repetition, not less | reorder `sample_points`, or accept that the axis shrinks code and specialize |
| `no run candidate rolled at level` | Runs were found, but none survived being turned into a loop | usually a loop-carry gap |
| `unresolved register 'accNNN'` | A value crossing the run boundary was never matched to a loop carry | roller gap |
| `runtime scf.for iter-arg arity scales with axis (variable loop-carry fan)` | The kernel loop's *carried-value count* itself grows with the axis | needs parametric `scf_for` iter-args in schema + VM |

**What each buys.** The constant solver wins artifact count: one file serves many
shapes, at the same op count as any one of them. The structural roller wins both —
it is the only one that shrinks the recipe below the concrete op count, because a
block stored once stands in for however many copies a shape needs.

### Not every constant is affine

Affine covers what a kernel *computes* from a shape, because address arithmetic is
affine. It does not cover constants a code generator *chose* given a shape, so the
solver tries a short ladder of candidates, simplest-first, and takes the first one
that reproduces every recorded point exactly:

| Candidate | Shape | Where it shows up |
|---|---|---|
| affine | `c0 + Σ mⱼ·xⱼ` | almost everything: offsets, strides, sizes |
| magic-division operand | `magic_shift(x)`, `magic_multiplier(x)` | a kernel dividing by a spec value |
| reciprocal | `k div x`, `ceil(k/x)` | a block or tile **count** from a block **size** |
| cross term | `+ m·xⱼ·xₖ` | one constant scaling with two axes at once |

The middle two matter more than they look. Conv's `C` axis makes the kernel divide
by `C`, and the compiler strength-reduces that into `(umul_hi(n, M) + n) >> s`
where `s = ceil(log2 C)` and `M` depends on `C`'s odd part — one logarithmic, one
number-theoretic, so no polynomial of any degree fits either. The recipe carries
the *generating formula* instead, and the C VM regenerates both at replay.

Cross terms are last because they cost evidence: a product is invisible to
one-axis probes, so fitting one needs a point where two axes move together.
`roll_nd` only records those after a verification failure, and refuses to do it at
all unless you have passed a holdout — otherwise fitting would consume the very
grid points that verification depends on.

Choosing axes, in order of what actually matters:

- **Leave the reduction axis out.** GEMM `tile_k`, attention `head_size` — these
  drive the hot loop, so they change *structure*, not just constants. If one axis
  does restructure, name it `structural_axis=` and it gets the full structural
  roller while the rest stay constant models.
- **Put at least one holdout outside the range you sampled.** Your samples span a
  box: sample `N` at 8 and 16 and `K` at 64 and 128, and the box is that rectangle.
  A holdout *inside* it — say `N=12, K=96` — is a weak test, because it sits
  between values the model was already fitted to, and a wrong model has not had
  room to go far wrong yet. A holdout *outside* it — `N=64, K=512` — is where a
  wrong model separates from a right one, and it is also the case you actually
  care about, since the whole point of a parametric recipe is to serve shapes
  nobody recorded.

- **Compare `n_recorded` against `len(points)`: the gap is the safety margin.**
  These two numbers answer different questions. `n_recorded` is how many traces
  were recorded to *build* the model; `len(points)` is how many points the finished
  model was *checked* at. Verification is deliberately much wider than inference.

  Conv makes the numbers concrete. Sampling `N` at 2 values, `K` at 2 and `C` at 3
  costs only **5** recordings, because each axis is probed one at a time out from a
  shared starting point — `1 + (2−1) + (2−1) + (3−1)`. But the recipe is then
  checked against a fresh recording at all `2×2×3 = 12` combinations plus 3
  extrapolated holdouts, so **15** points are verified from 5 traces.

  That gap is not an accident, and here is what it buys. Suppose some constant is
  really `m·N·K`. Probe `N` on its own, with `K` pinned at 64, and it looks exactly
  like an ordinary slope of `64m` per unit of `N` — a perfect fit. Probe `K` on its
  own and the same thing happens. Every one-axis probe agrees, so nothing in the
  fitting stage can possibly notice, and the model comes out as a sum of two
  independent slopes with no product term. The only place the lie shows up is a
  point where **both** axes moved at once — an interior point of the grid, which
  inference never recorded and verification always checks.

- **For a structural axis, pick values where every loop runs at least twice.** To
  turn a repeated block of code back into a loop, the roller has to work out what
  each iteration hands to the next — which register the running sum lives in, how
  an address advances. It learns that by lining the first copy of the block up
  against the second and looking at what changed between them. So a loop that
  appears **once** in a trace gives it nothing to compare against, and a loop that
  appears **zero** times is not in the trace at all. Two copies is the minimum
  evidence, and whether you have it is decided by the values you sampled, not by
  how clever the detector is.

  The trap is trip counts of the form `n-1`, which are everywhere, because
  combining `k` values takes `k-1` steps (adding up four numbers takes three
  additions). attention's softmax over `block_n` is one of these: the loop runs
  **zero** times at `block_n = 32` and once at 64. So 32 and 64 — the obvious first
  pair to reach for — is exactly the pair that cannot work, no matter what the
  roller does. 64 and 128 give 1 and 3 copies, and those can. When this happens the
  refusal says so and names values that would work, rather than reporting a
  detector failure.

- **Pick values that make the axis show its effect.** Two samples can pin down a
  model with two unknowns and nothing more, and *which* two you pick decides what
  the roller is able to see at all. If some constant happens to hold the same value
  at both of your samples, nothing suggests it varies, so the roller freezes it —
  and the roll then "succeeds" while being wrong at every other value.

  Powers of two are the classic way to walk into this. Dividing by a power of two
  is just a bit shift, so the magic-division multiplier is `1` for every single one
  of them. Sample conv `C` at 64 and 128 and that constant reads as invariant;
  sample 64, 96, 128 and it moves, and the axis rolls.

  Two samples can also be too few to tell apart two models that *both* fit. The KV
  block count `512 div b` is 16 at `b = 32` and 8 at `b = 64` — and the straight
  line `24 − b/4` passes through both of those points exactly. Two points, two
  models, no evidence to choose between them. A third value settles it: the line
  predicts `24 − 128/4 = −8` at `b = 128`, where the real count is 4.

- **A refusal is a real answer.** `r.reason` names the field and axis, and the
  caller keeps concrete per-point recipes. Some axes are not affine in any
  regime — GEMM `tile_m` re-vectorizes its load path, so re-sampling will not help.

### When one recipe is the wrong goal

Some axes are not uniform, and no better fit will make them so: past a threshold
the kernel takes a different path and there are simply two programs. That is worth
recognising because a refusal is expensive — it costs the *whole* axis, dropping
every value back to its own concrete recipe.

`roll_regimes` covers such an axis with as few recipes as its structure allows:

```python
from rocke.portable_ir.src.roll_regimes import legal_values, roll_regimes, regime_report

vals = legal_values("block_n", range(16, 1025, 16), make_spec)   # ask the kernel
r = roll_regimes(build_at, axis="block_n", values=vals)
print(regime_report(r))
r.recipe_for(64)
```

It never guesses where to split. A regime rolls from its first two values, then
extends one value at a time while the recipe still reproduces the real recording
byte-for-byte; the first value that does not verify starts the next regime. So a
threshold moving in the kernel moves the split with no change here, a uniform axis
still comes back as one recipe, and every regime is verified at every value it
claims. If structure changes at *every* value (gemm `tile_m`) there is no threshold
to find and you get concrete recipes — regimes help when the change is piecewise.

**Ask before you roll.** `legal_values` runs candidates through the kernel's own
spec validation, which keeps the constraint in one place *and* tells you what the
axis is worth: attention `num_query_heads` has 128 legal values, `seqlen_kv` 32,
`block_n` 5 (it must divide `seqlen_kv`), `head_size` 2. A recipe saves one
concrete recipe per legal value, so a two-value axis cannot repay one however
neatly it might roll. `roll_nd_coverage` prints this table.

The [scaling plan](portable_ir_scaling_plan.md#pitfalls--sharp-edges) collects the
traps in full, including the one that survives the oracle: `recipes_equiv` compares
programs and **not** the kernel symbol, so a recipe whose name does not track its
axes can pass every oracle check and still emit the wrong symbol. Both `roll` and
`roll_nd` check the name at every point (`roll` is a thin wrapper over `roll_nd`
with a single axis), but a caller using `recipes_equiv` directly is not covered.

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
│   ├── roller.py              multi-trace structural roller + affine solver
│   ├── roll.py                roll(build_at, axis, …) driver (records + verifies)
│   ├── roll_nd.py             roll_nd(build_at, axes={…}) — one recipe, N axes
│   ├── recipe_bundle.py       CBOR codec + bundle (rocke.bundle/v1)
│   ├── guard.py               derive/verify a rolled recipe's admission guard
│   ├── abi.py                 wire/binary compatibility contract (mirrors rocke/abi.h)
│   ├── launch.py              attach/read launch geometry (mirrors rocke/recipe_launch.h)
│   └── online.py              ctypes binding to the C backend (recipe/IR → .ll)
├── utils/
│   └── recipe_expand.py       pure-Python recipe expander + recipes_equiv (oracle)
│                              + check_guard (mirror of the C guard evaluator)
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
│   ├── roll_hsaco_parity.py       rolled-path .ll sha + HSACO gate, incl. held-out
│   ├── roll_nd_coverage.py        multi-axis gate: one recipe per axis cross product
│   ├── derive_guards.py           guards for real families: derive, verify, bundle
│   └── launch_from_bundle.py      CBOR -> plan -> HSACO -> verified GPU launch
├── tests/          unittest suites (recorder drift, roller, multi-axis, CBOR/bundle,
│                   guards, ABI compatibility, launch plans)
└── portable_ir_scaling_plan.md
```

The C++ side lives in `platform/cpp/portable_ir/` (C++20, part of
`librocke_core.a`; see that dir's `README.md`):
- `recipe_vm.cpp` (+ `rocke/recipe_vm.h`) — the recipe VM. Also carries the guard
  evaluator (+ `rocke/recipe_guard.h`), so that guards and recipes share one
  intexpr evaluator rather than two.
- `recipe_vm.cpp` also carries the launch planner (+ `rocke/recipe_launch.h`):
  kernel name, kernarg layout and grid/block/LDS for a shape.
- `core/rocke_abi.cpp` (+ `rocke/abi.h`) — the two compatibility contracts, and
  the policy for bumping each. Distinct from `rocke/rocke_build_id.h`, which is
  provenance: build-ids change every commit, so they cannot gate anything
  without forcing a lockstep upgrade of the whole stack.
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
- The Python oracle (`recipes_equiv`) compares **programs, not the kernel symbol**.
  A recipe whose name does not track its axes therefore passes every oracle check
  while emitting the wrong symbol; the `.ll` gates catch it because the symbol is
  in the text, and `roll_nd` checks the name at each point directly. Treat the
  oracle as necessary but not sufficient (scaling plan, pitfall 4).

## Measured: every kernel in `kernels/gfx950`

The gates above check axes already known to work. This is the opposite exercise —
point the roller at all five build entry points in `kernels/gfx950`, every axis
they expose and every feature flag that changes what they emit, and report what
happened. Reproduce with `drivers/roll_gfx950_sweep.py`; a refusal there is a
finding, not a failure.

| Family | Entry point | Axes probed | Rolled | One recipe covers | Traces | Feature settings held |
|---|---|---|---|---|---|---|
| attention_dense | `build_attention_dense` | 10 | **7** | 65 pts, 6 axes | 22 | 6/10 |
| attention_tiled_2d | `build_unified_attention_2d_tiled` | 11 | 3 | 9 pts, 3 axes | 4 | 22/22 |
| attention_tiled_3d | `build_unified_attention_3d_tiled` | 7 | 3 | 5 pts, 2 axes | 3 | 10/10 |
| attention_reduce | `build_unified_attention_reduce_tiled` | 4 | 2 | 3 pts, 1 axis | 2 | 2/2 |
| fastkv_regp | `build_unified_attention_2d_fastkv_register_p` | 6 | 1 | 3 pts, 1 axis | 2 | 3/3 |

**The `roll_nd` payoff, at full resolution.** Those coverage numbers are from the
2-samples-per-axis default. Raise it to three and `attention_dense` becomes the
clearest demonstration in the tree: **one recipe covering 730 verified points from
28 recorded traces**, parametric in six axes at once —

```
6 axes  batch, seqlen_kv, num_query_heads, seqlen_q, num_persistent, waves_per_eu
729 grid + 1 held-out points verified from 28 traces  (5913 ops, 171s)
one recipe = 548.6KiB CBOR vs 390.6MiB for 730 concrete (547.9KiB each)  729x
holdout: batch=64 seqlen_kv=2048 num_query_heads=512 seqlen_q=4096
         num_persistent=1024 waves_per_eu=8      -- outside the box on every axis
```

Note what the 26x is *not*. It is not a storage figure; it is that 28 builds of a
Python kernel generator answer for 730 shapes, each of which was checked against
its own independent recording. One recipe per axis would have taken six recipes
and covered a line through the space rather than the volume.

The byte figure on the third line is the same recipe measured a different way:
**548.6 KiB of CBOR in place of 390.6 MiB**, at 770 bytes per shape served. It is
also byte-for-byte the *same* 548.6 KiB the 2-sample run produced for 65 points —
raising the sample count grew what the recipe is verified to cover by 11x and grew
the recipe itself by nothing at all, since more samples buy confidence in the
models rather than more models. [Sizes for every family are
below](#what-the-rolled-recipes-cost-on-disk).

### Which axes rolled

Constants-only (`roll_nd` with no structural axis) covers most shape parameters:

| Family | Rolls today |
|---|---|
| attention_dense | `batch`, `seqlen_q`, `seqlen_kv`, `num_query_heads`, `num_kv_heads`, `waves_per_eu`, `num_persistent` |
| attention_tiled_2d | `num_seqs`, `num_kv_heads`, `kq_lds_pad_halves` |
| attention_tiled_3d | `num_query_heads`, `num_kv_heads`, `num_seqs` |
| attention_reduce | `num_query_heads`, `num_kv_heads` |
| fastkv_regp | `num_seqs` |

Three more roll once the [structural
roller](#3c-the-structural-roller-and-the-simpler-one-next-to-it) is pointed at
them with `structural_axis=`, which the sweep does not do by default: `attention_reduce ::
num_segments`, `attention_tiled_3d :: num_segments`, and `attention_tiled_2d ::
sliding_window`. The last one only works when sampled away from zero — `0` means
*disabled*, a different program rather than a smaller one, so a grid containing it
is asking one recipe to span two regimes.

### Which axes did not, and why

The refusals sort into five causes, and only two of them are roller gaps.

**1. The axis barely exists.** `head_size` takes 2 legal values, `block_n` 5,
`kv_ring_depth` 1. `fastkv_regp` pins four of its six axes to a single value each,
because its support gate admits exactly one shape. Nothing to roll toward; a
concrete recipe per value is already the right answer.

**2. The axis shrinks the code as it grows.** `attention_tiled_2d :: num_warps`
emits 1387, 1234 and 1162 ops at 1, 2 and 4 warps, because more warps means less
work each. The roller reports `shorter-at-larger-axis`, the same class as gemm
`tile_m`: rolling looks for a block that repeats *more* often at the larger value,
and here the larger value repeats it less. Specialized recipes fit this better
than any model would.

**3. Structural roller gaps.** `attention_tiled_2d :: tile_size` gets past name
reconstruction and then fails inside the roll with an unresolved loop-carried
register; `attention_tiled_3d :: block_size` finds no repeated run to segment.
These are genuine mechanism gaps rather than properties of the kernels.

**4. The constant depends on a PRODUCT of axes, structurally.** This is the one
real finding in `attention_dense`, and it explains all four of its held-back
feature settings. In persistent mode every axis still rolls *alone*, but any pair
fails at an interior grid point, and with cross-term fitting allowed the emitted
*length* diverges (2387 vs 2404 instructions). Persistent mode spreads
`batch × num_query_heads × q_blocks` tiles over a fixed CTA count, so the work
decomposition is a reciprocal of a product — structural in two axes at once.
Cross-term fitting handles constants, not structure, so this is out of reach.

**5. The kernel name drops a token at its default value.** `num_warps=1` yields
`..._bf16` while `num_warps=2` yields `..._bf16_w2`; `tile_size=32` omits the `_t`
token that 64 and 128 carry. The format is therefore not reconstructible by
substitution and the roll is refused — correctly, since the alternative is
emitting a wrong symbol. Sampling away from the default, or passing `name_fmt=`,
gets past it.

### Feature flags: rolling survives almost all of them

Flags are not axes. They pick different code at build time, so the question is not
whether a flag rolls but whether rolling still works with it set. Across 47
settings that the kernels accept, **43 held**: dtype, sinks, softcap, ALiBi, QQ
bias, sliding window, FP8 KV cache, FP8 MFMA, V double-buffering, staggered waits,
LDS padding, 64-bit KV addressing, scheduling barriers, the whole MFMA 32x32 and
transposed-QK stack, `kv_ring_depth=3`, register-P, softmax interleave, causal off,
varlen, ragged, and lazy-rescale off. The 4 that did not are the persistent-mode
group above — one cause, not four.

This is the more reassuring half of the result. The op counts move a lot across
these settings (`attention_tiled_2d` goes from 1162 to 3514 ops), so the recipes
are describing genuinely different programs and still rolling in each.

**It holds at three samples too.** That mattered enough to re-run, because a wider
grid is a stricter test — a third sample is what catches a constant that looked
invariant across two, the powers-of-two trap especially — so a flag that held at two
samples and refused at three would mean the count above was optimistic. Re-running
the full flag pass at `--samples 3` returns the same **43 of 47**, with the same four
refusals and the same persistent-mode cause. Only the arithmetic in one message
moves (`affine fit ['-8/5', …]` becomes `['-3/2', …]`), which is the residual of a
different grid, not a different finding.

### Things the sweep found in the kernels

Two are worth fixing on the kernel side rather than here.

**`num_kv_heads` is not validated against `num_query_heads`** in the tiled 2D, 3D
and reduce kernels. Neither the spec's `__post_init__` nor `supports_*` rejects a
`num_kv_heads` that does not divide `num_query_heads` — the admission check bounds
`num_queries_per_kv`, which is computed with a floor division that swallows the
remainder — so the kernel builds and bakes in a group size matching no real
grouping. `attention_dense` rejects the same combination outright. The sweep has to
filter these points itself to keep its domains honest.

**`UnifiedAttentionReduceTiledSpec` has no validation at all** and no `supports_*`
function, so its legal domains are the least trustworthy in the table: `head_size`
appears to accept 4 values only because the build asserts on the rest.

### What it changed here

One roller gap, closed. All three tiled kernels size their sequence-lookup loop as
`ceil(log2(num_seqs + 1))` — a binary search over the batch — and `num_seqs`
refused because no candidate model was that shape. It turned out the recipe
*schema* could already express it: `magic_shift` computes `ceil(log2 x)` and both
evaluators recurse into its operand, so `magic_shift(num_seqs + 1)` was always
representable and merely never hypothesised. The candidate ladder now tries the
magic operands at offset 0 and 1, and `num_seqs` rolls in all three families. Two
offsets is deliberately a tiny hypothesis space — `TestBinarySearchTripCount` pins
both that it extrapolates (fitted on 16/32/48, correct at 100000) and that it still
refuses an unrelated logarithm.

### What the rolled recipes cost on disk

CBOR is the shipping form, so this is the size question answered in the currency
that actually gets deployed. Sizes below are `len(cbor_encode(recipe))` at the
sweep's default two samples per axis, so they line up with the coverage table
above. "All concrete" is the mean concrete recipe times the points covered — what
shipping them one per shape would cost instead.

Three things get compared throughout this section, and they differ in *what varies
with the parameter*:

- **Concrete** — one file per shape, every integer baked in, no free variables.
  Nothing varies inside it; you ship N files for N shapes and pick by name. This is
  the baseline, not a rolling result.
- **Rolled, constants-only** — one file replacing those N. The same instruction list
  as any one of them (5913 ops either way, for `attention_dense`) with intexpr
  formulas where the integers differed. The numbers vary with the parameter; the
  instruction sequence does not, so the file is the size of one concrete recipe.
- **Rolled, structural** — one file that also varies *how many instructions are
  emitted*, by storing a repeated block once inside a `static_for` with a formula
  trip count. Its size stops depending on the axis, which is why this is the only
  one of the three where bytes actually shrink.

The last two are **layers, not alternatives**, and it is worth being clear about it
because the labels suggest otherwise. Constant fitting is the floor that every roll
stands on; a structural roll adds instruction-count parameterization on top and in
doing so asks the constant solver for *more*, not less. One structural axis on gemm
produces all of this in one recipe:

```
static_for var=_r0 hi={"div": [{"spec": "tile_n"}, 16]}   <- trip count, a fitted formula
  arith.constant value={"mul": [{"var": "_r0"}, 16]}      <- per-iteration stepper, fitted
arith.constant value={"spec": "tile_n"}                   <- an ordinary constants-only fit
```

A run cannot become a loop unless the constants that step from one iteration to the
next are modelled first, so "constants-only" means the structural layer found nothing
to do or was never asked — never that the structural case skips constants. `roll_nd`
makes the order explicit (annotate-then-roll, in the [vocabulary](#vocabulary)): fit every
axis-dependent constant across all axes, *then* roll structure over traces that are
already parametric. A two-axis MoE recipe ends up with `hidden` in a `static_for`
trip count and `tokens` only in the name format, in the same file.

| Family | Rolled recipe | One concrete | Parametric costs | All concrete (points) | Saved | Rolled per point |
|---|---|---|---|---|---|---|
| attention_dense | **548.6 KiB** | 547.9 KiB | +0.1% | 34.8 MiB (65) | **65x** | 8.4 KiB |
| attention_tiled_2d | **107.3 KiB** | 106.4 KiB | +0.8% | 957.4 KiB (9) | **9x** | 11.9 KiB |
| attention_tiled_3d | **270.8 KiB** | 264.8 KiB | +2.3% | 1.3 MiB (5) | **5x** | 54.2 KiB |
| attention_reduce | **12.0 KiB** | 11.9 KiB | +1.0% | 35.6 KiB (3) | **3x** | 4.0 KiB |
| fastkv_regp | **459.5 KiB** | 459.5 KiB | +0.0% | 1.3 MiB (3) | **3x** | 153.2 KiB |

The saved column is the point count in every row, give or take the rounding, and
that is the whole finding for a constants-only roll. The recipe holds the same
instructions as any one concrete recipe, with intexpr trees where plain integers
used to be, so it costs one recipe's worth of bytes plus a rounding error — the
"parametric costs" column, +0.1% to +2.3%, never more than a few KiB. Nothing is
compressed. What changes is how many files exist: one 548 KiB artifact where 65
stood, and it is *this* recipe rather than a lucky one, since all 65 points were
verified against their own independent recordings. Read the saved column as a count
of artifacts rather than bytes on the wire — a long-window compressor closes most of
that gap on its own, which is measured below.

Because the recipe does not grow, that ratio is bounded only by how much of the
space you verify. Re-running the whole sweep at three samples per axis produces the
*byte-identical* recipe in all five families — 548.6, 107.3, 270.8, 12.0 and 459.5
KiB again — while what each is verified to cover goes up:

| Family | 2 samples | 3 samples | Rolled recipe | Saved at 3 |
|---|---|---|---|---|
| attention_dense | 65 pts | **730 pts** | 548.6 KiB (unchanged) | **729x**, 770 B/shape |
| attention_tiled_2d | 9 pts | 19 pts | 107.3 KiB (unchanged) | 19x |
| attention_tiled_3d | 5 pts | 10 pts | 270.8 KiB (unchanged) | 10x |
| attention_reduce | 3 pts | 4 pts | 12.0 KiB (unchanged) | 4x |
| fastkv_regp | 3 pts | 4 pts | 459.5 KiB (unchanged) | 4x |

That is the cleanest statement of what a sample is for. More samples buy confidence
in the models already there — a third point is what distinguishes two candidate
models that agree on two — and no additional models, so coverage rises 11x on
`attention_dense` for zero extra bytes. The cost is time, not size: the extra points
are verified against fresh recordings, which is where the 171 seconds go.

**Where bytes really do shrink.** Point the [structural
roller](#3c-the-structural-roller-and-the-simpler-one-next-to-it) at an axis and
the picture changes, because the recipe stores a repeated block once instead of
once per repetition. Its size then stops depending on the axis at all while the
concrete recipe keeps growing. The two rolls below are *other kernels* — none of the
five attention families rolls structurally by default, so measuring this at all
means leaving `kernels/gfx950`:

| Structural roll | Rolled | Concrete at the smallest sampled value | … and at the largest verified |
|---|---|---|---|
| `gemm_universal :: tile_n` | **17.6 KiB** | 20.4 KiB at 32 (0.86x) | 80.1 KiB at 256 (**0.22x**) |
| `fused_moe/gather :: hidden` | **2562 B** | 2385 B at 512 (1.07x) | 12606 B at 8192 (**0.20x**) |

Both rolled recipes are a fixed size — 195 and 21 ops — while the concrete ones go
from 228 to 886 and from 20 to 125 ops as the axis grows. So the ratio falls with
the axis, with no bound and no further recording: one 17.6 KiB gemm recipe is
already smaller than the narrowest concrete recipe it replaces and is 4.6x smaller
than the widest, while covering all four values. Against per-value concrete recipes
over just those four, it is 17.6 KiB versus 175.5 KiB, or 10x. The MoE recipe shows
the crossover instead — 7% *larger* than the concrete recipe at `hidden=512`, where
the loop runs once and a `static_for` is pure overhead, then ahead everywhere
above it.

**These are raw CBOR, and compression matters for one of the two claims.** Nothing
in `cbor_encode` or the C++ loader compresses, so every figure above is the byte
count of the file as written. That invites the obvious objection — the 65 concrete
recipes are nearly identical, so would a compressor not find that redundancy by
itself? For a constants-only roll: yes, completely. Measured on a 16-point
`attention_dense` grid:

| | Rolled | 16 concrete | Ratio |
|---|---|---|---|
| raw CBOR | 548.5 KiB | 8.6 MiB | **16.0x** |
| gzip -9, per file | 46.3 KiB | 739.0 KiB | **15.9x** |
| gzip -9, one archive | 46.3 KiB | 734.5 KiB | 15.9x |
| xz -6, one archive | 26.4 KiB | 27.9 KiB | **1.06x** |

Per-file compression preserves the ratio exactly, and concatenating the 16 before
gzipping saves 0.6% — not because the redundancy is not there but because gzip's
32 KiB window cannot see across a 548 KiB recipe. Give a compressor a dictionary
big enough to hold them all and it finds what rolling found: `xz -6` reduces the 16
to 27.9 KiB, within 6% of the rolled recipe.

The marginal cost of each extra concrete recipe is what makes that vivid. Under
`xz -6` the first costs 26.3 KiB and the fifteenth costs **90 bytes** — 2 recipes
are 26.5 KiB, 4 are 26.7 KiB, 8 are 27.2 KiB, 16 are 27.9 KiB — because copies 2
onward are encoded as back-references to the first. Read the two forms side by side
and the symmetry is the point: rolling and `xz` remove *the same* redundancy by
different means, one semantically and one syntactically. A single concrete recipe
xz's to 26.3 KiB and the rolled recipe to 26.4 KiB, so a compressor prices them as
what they both are — one program's worth of information. The extra ~100 bytes is
what it costs to say how that program varies, which is also roughly what a
back-reference to a near-identical sibling costs. Neither is compressing better than
the other; they encode the same fact in different places. So for a constants-only axis the byte
ratio is **not** a storage claim, and the README's older framing — an artifact-count
win, not compression — is the correct reading. What `xz` cannot do is produce a
recipe for a shape nobody built: you have to run the generator 16 times, or 730
times, before there is anything to compress, and the 731st shape is still missing.
That, not bytes, is what rolling buys here.

**So should a refused axis just be compressed instead?** For storage, yes, and it is
worth doing. As a substitute for rolling, no — and the reason is that the 27.9 KiB
above and independent addressability are mutually exclusive:

| Option | Size | To load one recipe |
|---|---|---|
| Per-file `xz`, 16 files | 420.6 KiB | 1.1 ms, independently addressable |
| Solid `xz` archive | 27.9 KiB | 3.0 ms, inflate all 8.6 MiB |
| Rolled recipe, `xz`'d | 26.4 KiB | 1.1 ms, and covers 17 points |
| Rolled recipe, raw CBOR | 548.5 KiB | 0 ms — what the VM loads today |

The archive is small *because of* cross-file back-references, which is exactly what
makes its members non-independent. Require per-recipe access, as any kernel cache
does, and compression costs 420.6 KiB — 16x worse than the rolled recipe. It reaches
parity only in the form where fetching one kernel inflates 8.6 MiB at 3.0 ms, in
front of a comgr compile that is itself only 1.5–2.4 ms.

Compression is therefore the right answer for the [first refusal
class](#which-axes-did-not-and-why) — the axis that barely exists, where `head_size`
has 2 legal values and `block_n` 5, and a parametric recipe would cover only the
points it recorded — and for shipping a kernel library, where you inflate once at
install. It is the wrong answer for an axis with range, because it needs every shape
built first: 730 recipes need 730 generator runs and still cannot answer for the
731st. Best used *with* rolling rather than instead of it — roll what rolls, compress
the concrete residue, and compress the rolled recipe too, which also gains 21x.
Note that nothing in the C++ loader decompresses today, so this is a new dependency
and a new stage in the load path, and it does not touch the bottleneck that the
missing HSACO cache does.

Structural rolling is the opposite case, and survives the same test — a different
kernel, since the attention families have no structural axis rolled here:

| `gemm_universal :: tile_n`, 4 values | Rolled | Concrete | Ratio |
|---|---|---|---|
| raw CBOR | 17.6 KiB | 175.5 KiB | 10.0x |
| gzip -9, one archive | 2.5 KiB | 15.1 KiB | **6.1x** |
| xz -6, one archive | 2.3 KiB | 8.8 KiB | **3.9x** |

It narrows — a compressor does find some of the repetition that `static_for`
removes — but it does not close, because the recipe stores one copy of a block where
the concrete recipes store up to 886 ops of expansions. Even against a *single*
concrete recipe the rolled one stays smaller after compression: 0.22x of the
`tile_n=256` recipe raw, 0.34x gzipped, 0.50x xz'd, while also covering the other
three values. Here the byte figure really is a size win.

**And what `roll_nd` adds over `roll`.** Same axes either way, so the comparison is
one recipe per axis against one recipe for the cross product:

| Family | One `roll` per axis | Those cover | One `roll_nd` | It covers |
|---|---|---|---|---|
| attention_dense | 7 recipes, 3.7 MiB | 15 points | 548.6 KiB | **65 points** |
| attention_tiled_2d | 3 recipes, 320.0 KiB | 7 points | 107.3 KiB | **9 points** |
| attention_tiled_3d | 3 recipes, 804.6 KiB | 7 points | 270.8 KiB | **5 points** |
| attention_reduce | 2 recipes, 23.9 KiB | 5 points | 12.0 KiB | **3 points** |
| fastkv_regp | 1 recipe, 459.5 KiB | 3 points | 459.5 KiB | 3 points |

Since each single-axis recipe is itself constants-only, seven of them cost seven
full recipes — 3.7 MiB — and buy fifteen points on a cross of lines through the
space. One `roll_nd` recipe covers 65 points in the volume for 548.6 KiB: 7x fewer
bytes for 4x the coverage, or 8.4 KiB per point against 256 KiB, 30x better per
point served. The last row is the honest floor — `fastkv_regp` has one rollable
axis, so there is nothing to cross and the two forms are the same recipe.

**Reproducing these outside the sweep.** The numbers come from
`roll_gfx950_sweep`, but the rolling is `roll_kernel`'s, so the same figures come
back from naming the kernel directly — given the sweep's base config as `fixed`
and the grid its `choose_grid` picked as `axes`. Four of the five reproduce
byte-for-byte from a module path alone:

```bash
python3 -m rocke.portable_ir.drivers.roll_kernel \
    --kernel kernels.gfx950.attention_dense --arch gfx950 \
    --fixed batch=1 --fixed seqlen_q=512 --fixed seqlen_kv=512 \
    --fixed num_query_heads=128 --fixed num_kv_heads=8 --fixed head_size=128 \
    --fixed causal=true --fixed dtype=bf16 --fixed block_n=64 --fixed waves_per_eu=2 \
    --axis batch=1,22 --axis seqlen_kv=64,704 --axis num_query_heads=16,176 \
    --axis seqlen_q=256,1536 --axis num_persistent=64,384 --axis waves_per_eu=1,3 \
    --holdout batch=64 --holdout seqlen_kv=2048 --holdout num_query_heads=512 \
    --holdout seqlen_q=4096 --holdout num_persistent=1024 --holdout waves_per_eu=8
```

```
recorded 22 trace(s), verified 65 point(s)
CBOR     : 548.6 KiB parametric vs 35616.6 KiB for the same points concrete
```

`attention_tiled_3d` and `attention_reduce` live in one module, so they need
`--spec`/`--build` to say which. `fastkv_regp` is the exception and cannot be
reached from the command line at all: its spec is built by passing *another*
kernel's spec through `make_fastkv_register_p_spec`, which no flag expresses. It
needs a `Kernel(make_spec=...)`, which reproduces its 3 points / 459.5 KiB
exactly — see [`drivers/README.md`](drivers/README.md#kernels-that-do-not-follow-the-conventions).

Two caveats about the grid. `choose_grid` picks values that are legal
*together*, which is why the axis lists look arbitrary (`batch=1,22` rather than
`1,2`) and why `num_kv_heads` is absent despite rolling on its own — only one of
its values stays legal beside the other five axes. And the sweep's per-axis
`_samples` spread is deliberate: adjacent powers of two make every magic
multiplier 1, so a frozen constant reads as correct.

## Running things

The commands below are the gates and surveys. For the full driver index — and
for pointing the pipeline at **your own** kernel rather than the pinned gfx950
families — see [`drivers/README.md`](drivers/README.md).

Everything below runs from `platform/` with the engine importable:

```bash
export PYTHONPATH="$PWD/python:$PWD/../library${PYTHONPATH:+:$PYTHONPATH}"

# unit tests + recorder coverage (pure Python, no engine binary)
python3 -m unittest discover -s python/rocke/portable_ir/tests
python3 -m rocke.portable_ir.drivers.record_coverage

# the shared engine the ctypes replay path loads
cmake -S . -B /tmp/rocke/core -DCMAKE_BUILD_TYPE=Release -DROCKE_BUILD_SHARED_ENGINE=ON
cmake --build /tmp/rocke/core --target rocke_shared -j"$(nproc)"
export ROCKE_ONLINE_LIB=/tmp/rocke/core/librocke.so

# concrete path: both replay paths vs the Python lowerer, byte-identical .ll,
# every kernel x arch. Device-free, needs a shared librocke.
python3 -m rocke.portable_ir.drivers.parity_matrix [--arches gfx942,gfx950]
python3 -m rocke.portable_ir.drivers.hsaco_parity        # ... and their HSACO

# rolled path: record 2 traces -> roll -> replay. Same .ll sha at sampled AND
# held-out axis values. --no-hsaco stops at .ll (no comgr, ~2s).
python3 -m rocke.portable_ir.drivers.roll_hsaco_parity [--no-hsaco]

# multi-axis: ONE recipe per family covering its axis cross product, verified at
# every grid point + holdout. Pure Python (~1s); --ll adds an .ll sha report,
# --slow re-probes the refusals whose structural search costs ~20s each.
python3 -m rocke.portable_ir.drivers.roll_nd_coverage [--ll] [--slow]

# survey (not a gate): every build entry point in kernels/gfx950, every axis they
# expose, every feature flag. Reports coverage AND CBOR bytes per rolled recipe.
# ~2.5 min at the default 2 samples per axis; --samples 3 multiplies the verified
# grid, so --phase 3 (skip the per-flag re-rolls) is how to afford it: the 730-point
# attention_dense figure above is one 3.5-min run of
#   ... --family attention_dense --samples 3 --phase 3
python3 -m rocke.portable_ir.drivers.roll_gfx950_sweep \
    [--family F] [--samples N] [--phase 1|2|3|4]

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

### Which of these gate a pull request

All of the gating ones, in one command:

```bash
python3 tools/run_portable_ir_gates.py       # from platform/, as everywhere above
```

It builds the shared engine, then runs the unit tests, `parity_matrix`,
`hsaco_parity` and `roll_hsaco_parity --expect-points 22`, and owns the pinned
expectations so the CI workflow and a local run cannot disagree.
`.github/workflows/rocke-portable-ir-ci.yml` calls that script on any PR touching
`rocke/**`; it takes about 90 seconds and needs no GPU. The surveys
(`roll_gfx950_sweep`, `roll_nd_coverage`) and the device gate (`gpu_replay`) are
not wired: the first two are exploratory, and the last needs a GPU runner.

Two of those gates carry a pinned expectation, because both can pass while
measuring nothing:

* `hsaco_parity` reads `drivers/hsaco_baseline.json`, the set of kernels that do
  not reach HSACO at all, by name and with the LLVM error for each. A **new** one
  fails CI; the known ones do not, which is what keeps the gate from being red on
  its first run and switched off. Names rather than counts, so that one kernel
  fixed and another broken — which leaves every total unchanged — is still
  caught. Regenerate with `--update-baseline` and read the diff.
* `roll_hsaco_parity --expect-points N` fails if fewer than N points were
  verified. Without it, an axis that stops rolling produces a shorter table and a
  green tick, since a refusal is safe behavior and there is nothing wrong to
  report. Raise N when coverage grows.

## When does a new kernel need code here?

- **Concrete / CPython-free path:** never — it records and lowers automatically.
- **Rolling (shape coverage):** usually just declare the spec axes and their sample
  values (naming which one, if any, is structural) — `drivers/roll_kernel.py`
  takes them on the command line, so no gate needs editing to try it; a
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
