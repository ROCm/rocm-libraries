# hipDNN JIT integration: spec validity, recipe keys, and catalog pruning

Status: **the narrow path (§3) is implemented**, and a no-CPython caller can now
take a shipped bundle all the way to a launch; the broader path (§5 onward)
remains a proposal. Written against the portable-IR record-and-replay work on
`users/yraparti/rocke-jit-compilation-prototype`, whose byte-identity gates now
run in CI (see `dsl_docs/architecture/portable_ir_production_readiness.md`).

What shipped, and where it lives:

| Piece | Where |
|---|---|
| Guard derivation, oracle, gate adapters | `src/guard.py` |
| Guard evaluation (Python mirror) + enforcement in `expand_recipe` | `utils/recipe_expand.py` |
| Guard evaluation in C + enforcement in the VM | `cpp/portable_ir/recipe_vm.cpp` |
| **C API for hipDNN: admission** | `cpp/include/rocke/recipe_guard.h` |
| **C API for hipDNN: launch (name, args, grid)** | `cpp/include/rocke/recipe_launch.h` |
| **Version compatibility, both engines** | `cpp/include/rocke/abi.h`, `src/abi.py` |
| Launch geometry authoring + Python mirror | `src/launch.py` |
| ctypes surface for the C API | `src/online.py` |
| Tests: guard, ABI, launch, incl. Python/C parity | `tests/test_{guard,abi,launch}.py` |
| Worked examples over real families | `drivers/derive_guards.py`, `drivers/launch_from_bundle.py` |

Measured on the five gfx950 attention families, guards derive in under 10ms
each, occupy under 1KiB of CBOR, and agree with the family's own gate on every
sampled point — no unsound admissions and, on four of the five, no lost
coverage either. §3.3b records the numbers and the one case that needed work.

Three things landed after the guard work, each closing a gap that only shows up
once a bundle is actually shipped rather than replayed in the tree:

- **§3.3c — launch metadata.** The pure-C path used to stop at `.ll`. The grid
  was never in the bundle; it lived in host Python, so the last step of the
  chain was the one step that needed an interpreter. Geometry is now carried as
  intexprs over the spec axes and read back through `recipe_launch.h`.
- **§3.3d — version skew.** A bundle outlives the engine that wrote it, in both
  directions. Two numbers now govern that, and they are deliberately not one.
- **§9 — what is still missing.** `.ll` → HSACO and HSACO → launch exist only as
  Python ctypes wrappers. That may be fine, since hipDNN links comgr and HIP for
  itself, but it should be a decision rather than a surprise.

This document answers a specific set of questions from the hipDNN side: how a
non-Python JIT path decides whether a kernel instance is valid, how it finds out
whether a recipe was shipped for that instance, and what the two together cost
at build time and at launch time. Each section separates **what is supported
today** from **what needs to be added**.

It describes two paths. §3 and §4 are the **narrow path**: prune the recipe set
at CBOR-generation time using each kernel's own validation gate, and give every
generated recipe a small guard over its free parameters. That meets the "never
compile an invalid configuration" requirement, is feasible with the machinery
that exists, and needs no new predicate language. §5 onwards is the **broader
path** — a general, data-driven `is_valid_spec` that C++ can evaluate for any
spec. It solves a strictly larger problem and is not a prerequisite for the
narrow one.

## Terminology

This document is written for a hipDNN reader with no portable-IR background, so
the vocabulary is defined here before it is used. The package README has a fuller
[Vocabulary](README.md#vocabulary) section, including the rolling terms that
refusal messages are written in; below is only what this document leans on.

| Term | What it means here |
|---|---|
| **spec** | The struct of compile-time choices that identifies one kernel instance: dtypes, tile and warp shape, pipeline and epilogue selection, head size, flags. Not the problem shape — a spec is what you would call an instance's configuration. |
| **gate** / **admission predicate** | The function that answers "is this spec buildable at all", returning `(ok, reason)`. Spelled `is_valid_spec(spec, arch)` in `instances/`, `supports_*(...)` in `library/kernels/`. This is the thing hipDNN wants callable from C++. |
| **build** / **emit** | Running the Python kernel author's function to produce the kernel's instruction list. The code that does this is the **emitter** or **builder**. |
| **`.ll`** | LLVM IR as text — the instruction list serialized, one step before machine code. |
| **comgr** | AMD's Code Object Manager: compiles `.ll` into a GPU binary. The slow step in a JIT, 1.5–2.4 ms here. |
| **HSACO** | The compiled GPU binary that actually gets loaded and launched. |
| **byte-identity** | The correctness standard used throughout: two paths must produce *the same bytes*, not merely equivalent kernels. It is what makes the C replay path trustworthy, and what CI enforces. |
| **recipe** | A recorded log of what the emitter did, replayable without Python. **Concrete** = covers exactly one spec, every number baked in. **Parametric** (or **rolled**) = declares some values free and carries formulas, so one recipe covers many specs. |
| **free** vs **baked** | A parametric recipe lists its free values in its `spec` field; those are supplied at replay. Everything else was fixed when the recording was made — baked. This split is the hinge of the whole design in §3. |
| **roll** / **the roller** | Turning several concrete recipes into one parametric recipe by finding the pattern that relates them. **Refusal** = the roller declining, with a reason, because it could not prove a pattern; the caller then keeps concrete recipes. A refusal costs compression, never correctness. |
| **verified point** | A spec value at which a rolled recipe was actually checked against a fresh Python recording. Points it was *fitted* to plus **held-out** points it was only *tested* at. Anything else is inference. |
| **replay** / **the recipe VM** | Re-running a recipe in C to rebuild the instruction list, then lowering it. The VM is family-agnostic: it replays whatever was recorded, so it needs no C++ port of the kernel's builder. |
| **CBOR** | A binary format with JSON's data model (maps, arrays, ints, strings). The shipping form of a recipe: same content as the JSON shown in examples, smaller and faster to parse. |
| **bundle** | One CBOR blob holding many recipes, looked up by `(key, arch)`, so a runtime opens one file instead of hundreds. |
| **guard** | A small predicate carried by a recipe, checked before replay, that rejects an invalid binding of its free values. Proposed in §3 of this document; now implemented. |
| **kernarg** | The flat buffer of kernel arguments handed to the GPU at launch. Each argument sits at a fixed byte offset; getting one offset wrong corrupts every argument after it. |
| **launch plan** | What a caller needs once it holds a compiled kernel: the mangled kernel name, the kernarg offsets, and the grid/block/LDS to launch with. §3.3c. |
| **arch** | The GPU target, e.g. `gfx950` (CDNA4), `gfx942` (CDNA3). Gates are arch-aware because the legal MMA instructions and LDS capacity differ. |

### What an `intexpr` is

This is the one piece of jargon the rest of the document genuinely depends on,
so it is worth building up rather than defining in a clause.

A parametric recipe has to emit a *different* kernel for a different shape, which
means some of the numbers inside its recorded instruction stream cannot be stored
as numbers — they depend on a value not known until replay. Say a buffer offset
was `512` when the kernel was recorded at `hidden = 512`, and `1024` when
recorded at `hidden = 1024`. The recipe cannot store either; it has to store the
*relationship*.

An **intexpr** is that relationship, written as a small expression tree in the
same JSON/CBOR the recipe is made of. For the case above:

```json
{"spec": "hidden"}
```

and if the offset had been twice the hidden size:

```json
{"mul": [{"spec": "hidden"}, 2]}
```

The whole language is a handful of node kinds — leaves that fetch a value, and
binary operators that combine them:

| Node | Meaning |
|---|---|
| `512` | a literal |
| `{"spec": "hidden"}` | the value supplied for `hidden` at replay |
| `{"var": "_r0"}` | the current index of a compile-time loop in the recipe |
| `{"spec_str_eq": ["dtype", "fp16"]}` | 1 if the string supplied for `dtype` is `"fp16"`, else 0 |
| `{"add": [a, b]}`, and `sub mul div mod` | arithmetic on two sub-expressions |
| `{"eq": [a, b]}`, and `ne lt le gt ge` | comparison, yielding 1 or 0 |
| `{"magic_shift": e}`, `{"magic_multiplier": e}` | regenerate the two constants of a strength-reduced division, which are not expressible as arithmetic on the axis |

Two things about it matter for this document.

**It is evaluated in two places, and CI proves they agree.** `eval_intexpr` in
`utils/recipe_expand.py` (Python) and `rv_int` in `cpp/portable_ir/recipe_vm.cpp`
(C99) implement the same grammar. Because a wrong constant changes the emitted
kernel, the byte-identity gates fail if the two ever disagree. So "an expression
the C runtime can evaluate, that Python agrees with" is not something to be
built — it is running in CI today.

**The comparison operators make it a predicate language.** `eq`, `lt`, `ge` and
friends yield 1 or 0, and the VM already uses an intexpr *as a truth value*: its
`static_if` instruction evaluates one to choose which branch of a recipe to emit.
A validity check is the same shape — an expression over the spec that comes out
1 or 0 — which is why §3 can propose a guard without inventing anything.

### The generator

**"The generator"** throughout this document means the *offline, build-time
Python program that decides which kernel configurations to record and emits the
shipped CBOR bundle*. It is the offline counterpart to the runtime replay path:
the generator runs once, on a build machine, with the full Python kernel library
importable; the replay path runs per launch, in C, with none of it. Nothing else
is meant — not a Python `yield` generator, and not the IR emitter (which this
document calls the emitter or the builder).

It only partly exists today. What is there is plumbing rather than policy:


| Piece                                            | What it does                                                                                                                |
| ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| `recipe_bundle.record_concrete_bundle(cases)`    | Records a caller-supplied list of `(build_callable, arch)` into concrete recipes                                            |
| `recipe_bundle.py bundle out.cbor a.json b.json` | Packs already-written recipe JSON into one CBOR bundle                                                                      |
| `recipe_bundle.py record-demo`                   | The above on a hardcoded three-case example set                                                                             |
| `drivers/roll_gfx950_sweep.py`                   | Enumerates every axis of every gfx950 kernel and rolls them — but as a survey, measuring sizes and discarding the artifacts |


So the *decision* of what to record is currently a hand-written case list, and
the *enumeration* logic exists only in a driver that throws its output away. The
generator this document proposes is the missing piece that joins them: enumerate
a declared axis space, prune it with the kernel's own gate, record what survives,
roll where rolling succeeds, derive each recipe's guard, and write the bundle.

---

## Short answers


| Question                                                        | Answer                                                                                                                                                                                                                                                                                                                                               |
| --------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Can each CBOR recipe carry its own `is_valid` gate?             | **Yes, and it is small.** The expression language, both evaluators, and the CI gate that proves they agree all exist. Adding a `guard` field is backward compatible, and enforcing it is ~15 lines in the VM. See [§3](#3-the-narrow-path-a-validity-gate-carried-by-each-recipe).                                                                   |
| Why is that so much cheaper than porting `is_valid_spec`?       | Because baked parameters are validated in Python at generation time, so the runtime gate only has to cover a recipe's **free axes** — 1–3 integers, not a whole spec. Concrete recipes (`"spec": []`) need no guard at all.                                                                                                                          |
| Can we prune at CBOR-generation time using the kernel's gate?   | **The primitives exist** (`legal_values`, `legal_point`, `choose_grid`) but **the generator does not** — today's bundle builder takes a hand-written list of build callables, so there is no enumeration step to prune. That driver is the main new work.                                                                                            |
| Does a pruned bundle remove the need for a C++ `is_valid_spec`? | **Largely, yes.** If invalid combinations were never recorded, an invalid instance simply has no recipe and the existence check drops it. Invalid and not-shipped become the same outcome: skip the candidate. This takes the [broader migration](#5-the-broader-path-validity-as-data-in-the-language-the-vm-already-speaks) off the critical path. |
| Anything needed from the `rocke/library` kernels?               | **Nothing strictly, but four things are worth fixing** before a generated catalog trusts them: a uniform admission entry point, declared axes and candidate domains, closing known holes in the gates, and hoisting builder-deep assertions. See [§4](#4-what-the-library-kernels-need-for-this-to-work).                                            |
| Do we still need a check at replay time?                        | **Yes** — the guard is exactly that check. Constraints coupling free and baked parameters can only be evaluated once the problem shape is known. See [§7](#7-validity-at-replay-time).                                                                                                                                                               |
| Can we ask "is there a recipe for this key?"                    | **Partly.** `rocke_bundle_contains` answers existence without lowering, so the query is no longer "run it and see". But the key is still a kernel-name string rather than a structured non-free parameter tuple, and there is no parse-once index. Tracked as gap 8 / item 4.                                                                        |
| Once we have a HSACO, do we know what to launch it with?        | **Yes, now.** The recipe carries grid/block/LDS as expressions over its spec axes, and `recipe_launch.h` returns those plus the kernel name and the kernarg offsets. Previously this was the one step that still needed Python. See [§3.3c](#33c-launch-metadata-what-the-caller-needs-after-the-guard-says-yes).                                    |
| What happens when the engine and the bundle disagree in age?    | Two numbers, checked separately: a binary ABI verified once at load, and a per-artifact `min_reader` that lets an old bundle keep working on a new engine. A too-new bundle is an error, not a guard refusal. See [§3.3d](#33d-version-skew-between-the-two-engines).                                                                               |
| How big is the non-free parameter space?                        | Not answerable as a single number, and the product of per-axis domains badly overstates it because axes are coupled. Build-time cost is driven by **roller refusals**, not by the size of the valid space.                                                                                                                                           |


---

## 1. What validity means today


### 1.1 It is three layers, not one

There is no single `is_valid_spec` for a kernel. Legality is enforced at three
different depths, and each catches things the previous one does not. The
clearest statement of this is in `src/roll_regimes.py::legal_values`, which is
the function the production sweep uses to ask a kernel how large an axis
actually is:

1. **The spec constructor.** `make_spec(**{axis: v})` runs the dataclass
  `__post_init__`, which raises for per-field violations.
2. **The admission predicate.** `supports_tiled_2d(...)` / `is_valid_spec(spec,
  arch)`returning`(ok, reason)`. This is where cross-field constraints live —`  num_query_heads % num_kv_heads == 0`, tile budgets, arch atom availability.
3. **The builder itself.** Assertions raised deep inside the emitter, well after
  the spec is constructed. `legal_values` calls this the "slowest and most
   truthful" layer, and it is the only one that catches, for example, the tiled
   kernels' head-size stripe alignment.

Two consequences matter for hipDNN. First, porting layer 2 alone does not
reproduce Python's verdict. Second — and this is a live defect rather than a
design choice — the layers have holes. `drivers/roll_gfx950_sweep.py` documents
one it had to work around:

> the tiled 2D and 3D kernels accept a `num_kv_heads` that does not divide
> `num_query_heads`. Neither the spec's `__post_init__` nor `supports_*` rejects
> it … and the kernel then builds and bakes in a group size that does not
> correspond to any real grouping.

The sweep carries its own `coherent` predicate to filter those points. A
data-driven predicate transcribed from today's `supports_*` would inherit the
hole, so migration is an opportunity to close it, not merely to re-encode it.

### 1.2 The predicates are already nearly data

The actual content of these predicates is far simpler than their form. Taking
`library/kernels/gfx950/attention_tiled_2d.py::supports_tiled_2d` as
representative, every rule is one of:

- **set membership** — `head_size in {64,128,256}`, `block_size in {16,32,64}`,
`num_warps in {1,2,4,8}`, `dtype in {fp16,bf16}`
- **range** — `1 <= num_queries_per_kv <= 16`
- **divisibility** — `head_size % 32 == 0`, `tile_size % block_size == 0`
- **cross-field implication** — `block_m_per_warp == 32` requires `num_warps in {1,2,4}`; `use_fp8` requires `kv_storage_dtype == "fp8e4m3"`
- **arch equality**, delegated to a shared helper

All of that is expressible as data with no escape hatch. The harder families are
harder only in places. `instances/common/gemm_universal.py::is_valid_spec` adds
two kinds of rule that need more than comparison:

- **arch-table lookups** — `target.mma.has_shape(family, a, b, c, m, n, k)`,
`target.fits_lds(bytes)`, `target.max_threads_per_block`, `target.wave_size`.
These are already reads from `core/arch/data/arch_specs.json`, so they are
data too; they just need to be reachable from the predicate.
- **a derived cost model** — the LDS budget, computed by `_ab_lds_plan(spec, arch)`, a helper deliberately shared with the emitter so the gate and the
emitted code cannot drift apart. It is still a pure integer function of the spec,
so it is expressible, but it is the one place where "just write the formula"
has a real maintenance cost.


### 1.3 A C++ path already exists — and nothing proves it is correct

This is the most important thing to know before designing anything new.

`is_valid_spec` has **already been ported to C99** for most families. Counting
entry points: **68** Python definitions (`is_valid_spec`* / `supports_*` across
`platform/python/rocke/instances/` and `library/kernels/`) against **58** C99
definitions in `platform/cpp/instances/`, plus a dozen `*_spec_validate`
helpers. They are annotated with the Python line ranges they mirror:

```c
/*  is_valid_spec(spec, arch) -> (ok, reason).
 *  A faithful, ordered port of the Python predicate: each rejection writes the
 *  same single-line reason. `arch` NULL => "gfx950". On accept writes "ok". */
bool rocke_gemm_universal_is_valid_spec(const rocke_gemm_universal_spec_t* spec,
                                        const char* arch, char* reason, size_t reason_cap)
```

They are reachable from the engine binding as well — `rocke_engine.gemm_is_valid(spec, arch) -> (ok, reason)` and siblings for `batched_gemm`, `grouped_gemm`, `mfma_gemm`.

So the literal request — "do what `is_valid_spec` does, but not in Python" — is
already satisfied for a large part of the surface. The problems are elsewhere:

**No drift gate.** Nothing compares the Python verdict to the C99 verdict. The
parity gates that do run compare *emitted IR* for specs that are valid on both
sides; a spec that Python rejects and C accepts is invisible to them, because no
IR is ever recorded for it. The `rocke_engine.*_is_valid` bindings make such a
test cheap — call both from Python over a candidate grid and diff `(ok, reason)`
— but it does not exist yet.

**Coverage holes exactly where the JIT wants them.** `attention_dense` has
**zero** C++ presence — no builder, no validator — yet it is one of the four
families in the CI-gated rolled HSACO parity, where the C VM replays it
byte-identically. Also missing: the gfx1250 attention gates
(`supports_tiled_2d`, `supports_tiled_3d`, `wmma_attention_fwd`),
`batched_contraction`, `batched_transpose`, `transpose_bc`.

That `attention_dense` case is the whole argument in one example:

> **The recipe VM is family-agnostic — replaying a family needs no C++ port of
> its builder. Validity is the only family-specific thing the JIT still needs
> written in C.**

Porting builders is not required for hipDNN. Porting 68 predicates by hand, and
then maintaining two copies of each forever, is the only remaining obligation —
and it is precisely the kind of obligation that should be discharged with data
instead of code.

**Even the arch database is transcribed by hand.** `cpp/core/arch/data.cpp`
describes itself as "a faithful, byte-identical translation of … the embedded
`core/arch/data/arch_specs.json`". The data exists in one place and is copied to
the other by a human. Whatever we do for predicates, arch facts should be
*generated* from `arch_specs.json`, not transcribed again.

---

## 2. What the recipe side supports today


### 2.1 The recipe already declares its free parameters

A recipe (`rocke.recipe/v1`) looks like this:

```json
{
  "schema": "rocke.recipe/v1",
  "kernel_name_fmt": "conv_K{K}_N{N}_N{N}H56W56C64_K{K}Y3X3_t32x32x32_w1x1_a16x16x16_mem_default",
  "spec": [{"name": "N", "kind": "int"}, {"name": "K", "kind": "int"}],
  "attrs": {"max_workgroup_size": {"t": "i", "v": 128}},
  "program": [ ... ]
}
```

`spec` is the list of values the VM binds at replay time — exactly your "free
params". Everything else was baked in when the trace was recorded. A concrete
(unrolled) recipe has `"spec": []`: every parameter is baked.

The gap is that **the baked values are not recorded anywhere machine-readable.**
They survive only inside `kernel_name_fmt`, where the tile shape, warp shape,
MMA atom, pipeline and epilogue are encoded positionally in a string
(`..._t32x32x32_w1x1_a16x16x16_mem_default`). Forming a lookup key from
non-free parameters therefore means string surgery on a kernel name. This is
gap 8 in the readiness doc, "Bundle key hygiene unenforced — names embed
parametrized spec values".

### 2.2 The bundle is a flat list keyed by a name string

A bundle (`rocke.bundle/v1`) is a list of entries `{key, arch, family?, recipe}`.
Lookup is an exact string match on `key`, plus optional exact `arch`, by linear
scan — `src/recipe_bundle.py::bundle_lookup`:

```python
def bundle_lookup(bundle, key, arch=None):
    for e in bundle.get("entries", []):
        if e.get("key") == key and (arch is None or e.get("arch") == arch):
            return e.get("recipe")
    return None
```

`key` defaults to the kernel name for concrete recordings and to
`kernel_name_fmt` for rolled ones.

The complete C API surface for bundles is two functions:

```c
rocke_status_t rocke_recipe_run_from_bundle_cbor(const unsigned char* data, size_t len,
                                                 const char* key, const char* arch, ...);
rocke_status_t rocke_online_bundle_cbor_to_llvm(const unsigned char* data, ...);
```

Both **run** a recipe. There is no "does a key exist", no way to enumerate keys,
no index, and no structured key. For a virtual catalog of a few hundred
candidates against a bundle of a few thousand entries, a linear scan of CBOR per
candidate is the wrong shape even before correctness is considered.

> This section describes the state that motivated the design, and the surface
> has since grown: `rocke_bundle_contains` answers existence without running
> anything (§3.3a), and guard and launch queries were added alongside it
> (§3.3a, §3.3c). What has *not* changed is the part that matters here — the key
> is still an opaque string and lookup is still a linear scan over a freshly
> parsed bundle. That is item 4, and §3.3's cost note is the argument for it.

### 2.3 Catalog pruning already exists — in Python

`drivers/roll_gfx950_sweep.py` already builds a pruned catalog, and its design
notes are directly applicable:

- `legal_values(axis, candidates, make_spec, admits=, probe=)` — the three-layer
domain probe from §1.1.
- `Family.legal_point(point, build=)` — whole-combination legality, because
"axes interact … a cross product built from two independently-legal axis lists
can contain points the kernel refuses to build". The `build=False` mode is the
cheap declarative-only form used when scanning many combinations.
- `choose_grid(...)` — incremental feasible cross-product construction: axes are
added one at a time, "each new axis keeping only the values that stay legal
against every combination chosen so far", rather than dropping an axis at the
first conflict.

This is the virtual-catalog filter you describe, already written and already
exercised. It runs in Python because `admits` is a Python callable. Make the
predicate data and the same algorithm serves both the build-time generator and
the hipDNN runtime.

### 2.4 Measured costs

Re-measured after guards, the ABI check and the launch planner landed, so these
are the paths as they now ship rather than as they were assessed. Medians at
three axis points per family, `drivers/bench_jit_validation.py --n 21`, gfx950
on ROCm 7.2. Front end means the whole `CBOR -> .ll` step: decode, admission
checks, VM expand, C lower.


|                              | Python front end | Recipe VM front end | comgr   | Cold JIT via recipe |
| ---------------------------- | ---------------- | ------------------- | ------- | ------------------- |
| GEMM                         | 1.34 ms          | 0.32 ms             | 1.64 ms | **1.96 ms**         |
| Convolution (implicit GEMM)  | 2.37 ms          | 0.75 ms             | 1.51 ms | **2.26 ms**         |
| Attention (dense)            | 19.05 ms         | 7.05 ms             | 2.36 ms | **9.41 ms**         |


The admission checks do not show up here: timing the same recipe bare and in its
shipping form (ABI-stamped, guard attached) differs by under 1% for all three,
and repeated runs do not settle on a sign, so the difference is indistinguishable
from zero rather than a speedup. Guards are one rule and 412–455 B for
each of these families, and padding one to 256 tautological rules — far past
anything derivation produces — adds only 0.044 ms, most of it the extra guard
bytes to parse rather than rules to evaluate.

**The check that is not free is the one made without lowering** — §3.3a's
standalone query, the one this section's cost asymmetry argument depends on:

| | Bundle size | Guard check | `bundle_contains` |
|---|---|---|---|
| GEMM | 18.1 KiB | 0.077 ms | 0.075 ms |
| Convolution | 58.9 KiB | 0.271 ms | 0.268 ms |
| Attention (dense) | 548.7 KiB | 2.575 ms | 2.564 ms |

The time tracks the bundle size — 18.1 → 58.9 → 548.7 KiB against 0.077 → 0.271
→ 2.575 ms — and `bundle_contains` evaluates no rules yet costs the same as the
full check. So this is CBOR parsing rather than guard work. Rule evaluation is
nanoseconds; *reaching* the rules is not, because every call re-parses the whole
artifact and frees its arena on return.

This qualifies §9's ordering rather than overturning it. A guard check is not
free per candidate: against a 549 KiB bundle it costs 2.6 ms, more than an entire
GEMM cold JIT, and guard-checking 200 candidates would spend ~0.5 s re-parsing
the same bytes. Lowering those 200 instead would cost 0.4–1.9 s, so filtering
first still wins — but only if you filter on the cheap key lookup and check
guards on the surviving winner, not on all 200. Sweeping a catalog properly needs
item 4's parse-once handle, an argument that until now rested on general grounds.

Artifact size favours the parametric form: attention is 548 KiB as one recipe
versus 1644 KiB as three concrete traces; GEMM is 17.5 KiB versus 175.5 KiB and
convolution 58.2 KiB versus 174.3 KiB, across three shapes each.

Measured domain sizes, from the kernels' own gating rather than assumption:
`num_query_heads` 128 legal values, `seqlen_kv` 32, `block_n` 5 (it must divide
`seqlen_kv`), `attention_dense::head_size` 2.

---

## 3. The narrow path: a validity gate carried by each recipe

This is the cheapest design that meets the actual requirement — *never compile
an invalid configuration at hipDNN runtime* — and it is feasible today. It is
worth stating separately from §5 because it does **not** require porting or
re-authoring any of the 68 predicates.

### 3.1 The insight that makes it small

Validity splits cleanly along the free/baked line:

- **Baked parameters were already validated in Python, at generation time, by
the authoritative predicate.** The generator runs in CPython. It can call
`supports_tiled_2d` directly. Anything that fails is never recorded, so no
recipe for it ever reaches the bundle.
- **Free parameters are the only thing bound at runtime**, and a recipe declares
exactly which they are, in its `spec` list.

So the runtime gate does not need to answer "is this spec valid" in general. It
needs to answer the residual question: *is this binding of my 1–3 free axes
legal for me?* That is a per-recipe predicate over a handful of integers, not a
family-wide predicate over a whole spec.

Two consequences follow, and the second is the more important:

**Concrete recipes need no guard at all.** A concrete recipe has `"spec": []` —
it binds nothing at replay, so its validity was fully decided in Python when it
was recorded. Everything today's bundle generator emits is in this category, so
the guard applies only to the rolled recipes added on top.

**A pruned bundle makes recipe lookup itself the validity filter.** If the
bundle was generated by enumerating and pruning with the real Python gate, then
an invalid instance simply has no recipe, and hipDNN's existing
`does_recipe_exist` check drops it. Invalid and not-shipped collapse into the
same outcome — skip this candidate, try the next — which is exactly the
behaviour you described wanting. **This is why the general data-driven predicate
is not on the critical path for hipDNN.** The residual need for a standalone C
validity check shrinks to two things: distinguishing "invalid" from "valid but
not shipped" for coverage diagnostics, and guarding free-parameter bindings —
which is what the per-recipe guard does.

### 3.2 Feasibility: what it costs to build

Every piece needed already exists or is a small addition.


| Piece                                 | Status                                                                 |
| ------------------------------------- | ---------------------------------------------------------------------- |
| Expression language for the guard     | **Exists.** No new operators strictly required — see below             |
| Evaluator in C99                      | **Exists** (`recipe_vm.cpp::rv_int`)                                   |
| Evaluator in Python                   | **Exists** (`recipe_expand.py::eval_intexpr`)                          |
| Proof the two agree                   | **Exists** — the HSACO byte-identity gates already cover the evaluator |
| Adding a field to the recipe          | **Backward compatible** — see below                                    |
| Enforcement point in the VM           | ~15 lines at the top of `rv_run_root`                                  |
| Deriving the guard at generation time | Small; reuses `legal_values` / `legal_point`                           |


**No new operators are strictly required.** The guard is a truthy `intexpr`, and
the existing grammar (`add sub mul div mod eq ne lt le gt ge`, plus
`spec_str_eq`) is already functionally complete for boolean logic over 0/1
values: `and(a,b)` is `{"mul":[a,b]}`, `or(a,b)` is `{"gt":[{"add":[a,b]},0]}`,
`not(a)` is `{"eq":[a,0]}`, and membership is an `or`-chain of `eq`. A generator
emitting desugared expressions needs zero VM changes. The `and`/`or`/`not`/`in`
sugar from §5.2 is worth adding for readability, but it is optional and can come
later without invalidating anything shipped.

**Adding the field is backward compatible.** `rv_run_root` reads only the keys
it knows (`spec`, `kernel_name_fmt`, `attrs`, `program`) via `rocke_jget`;
unknown top-level keys are ignored, and the Python CBOR codec is a generic
map encoder. So a `guard` field is invisible to an older engine and enforced by
a newer one. The rollout does not need to be atomic.

**The enforcement point is trivial.** The guard must be checked before any IR is
built. `rv_int` reads only `vm->ints` / `vm->strs`, and `rv_fail` writes only
into `vm->err` — neither needs the builder — so the check slots in after the vm
is populated and before `rocke_ir_builder_init`:

```c
const jd_val_t* guard = rocke_jget(root, "guard");
if(guard)
{
    long ok = rv_int(&vm, rocke_jget(guard, "pred"));
    /* vm.failed distinguishes a malformed guard (e.g. a spec name the caller
     * did not bind) from an honest rejection; both refuse, for different
     * reasons, and neither may be reported as the other. */
    if(vm.failed || ok == 0)
    {
        if(err && err_cap)
            snprintf(err, err_cap, "%s", vm.failed
                     ? vm.err
                     : rocke_jstr(rocke_jget(guard, "reason")));
        return ROCKE_ERR_VALUE;   /* nothing allocated yet, nothing to unwind */
    }
}
```

hipDNN also wants to ask the question *without* lowering, so expose the same
check on its own:

```c
/* Evaluate a recipe's guard against a candidate binding. No IR, no allocation. */
rocke_status_t rocke_recipe_check_guard(const unsigned char* bundle, size_t len,
                                        const char* key, const char* arch,
                                        const rocke_recipe_spec_int_t* ints, int n_ints,
                                        const rocke_recipe_spec_str_t* strs, int n_strs,
                                        char* reason, size_t reason_cap);
```


### 3.3 Designing the validation function over a rolled recipe's free axes

This is the concrete design. A rolled recipe declares free axes in its `spec`
field; the function decides whether a binding of those axes is admissible.

```
check(binding) -> (ok, reason)
    binding : {axis_name -> int | str}, one entry per free axis
```

#### The contract, and which way it must fail

The truth the function approximates is *"would the kernel's own gate accept the
full spec, with this recipe's baked values and this binding?"* — written
`gate(baked ∪ binding, arch)`. The function must satisfy one invariant:

> **Soundness: `check(b)` accepts ⟹ `gate(baked ∪ b, arch)` accepts.**

The converse is deliberately *not* required. The two errors are not symmetric:

| Error | Consequence |
|---|---|
| Accepting a binding the gate rejects | **Correctness bug** — the exact failure this feature exists to prevent |
| Rejecting a binding the gate accepts | **Coverage loss** — hipDNN falls through to the next candidate |

So every fallback in the derivation must move toward rejection, never toward
permission. This is the same discipline the roller already follows — a refusal
costs compression, never correctness — applied to validity instead of to
emission.

#### Why the function is small: the baked values partially evaluate the gate

A family gate is large because it must handle every spec. This one does not.
With the baked values fixed, most of the gate's rules are already *decided*: they
mention only baked fields, and they came out true, or the recording that produced
this recipe would not exist. What survives is only the rules that mention a free
axis.

For a recipe with `head_size=128`, `block_size=16`, `dtype=fp16`, `num_kv_heads=8`
baked and `seqlen_kv`, `num_query_heads` free, the nineteen-argument
`supports_tiled_2d` collapses to a handful of rules: the ones constraining
`seqlen_kv` and `num_query_heads`, plus any coupling between them. Everything
about dtypes, tiles, warps and flags is settled.

Python cannot be symbolically partially evaluated, so the collapse is performed
**empirically** — by calling the gate and observing where it changes its mind.

#### Derivation

Run at generation time, once per rolled recipe, with the whole Python library
importable.

1. **Measure each axis's marginal domain.** For free axis `a`, over a
   deliberately over-wide candidate list, with the other axes held at reference
   points. This is the existing three-layer probe (spec constructor, gate, real
   build), reached through `gate_from_spec`. An empty result is a
   generation-time error — the recipe could not have been recorded — and a
   single-element one means the axis should not have been rolled at all.

   Measuring against *one* reference is the obvious implementation and it is
   wrong in a way worth recording, because it cost real coverage before it was
   found. A single reference reports what is legal **alongside that reference**,
   which for coupled axes is a slice of the domain rather than the domain. On
   the grouped-query axes of `attention_tiled_2d`, where `num_kv_heads` must
   divide `num_query_heads`, a reference that happens to sit at 272 collapses
   *both* axes to the single value 272: nothing else pairs with it. Every later
   step then inherits the collapse, and the guard refuses 102 of 1024 sampled
   shapes the kernel supports. It stays sound throughout — this is a coverage
   failure, not a correctness one — but a guard that discards most of its own
   axis is not earning the rolled recipe it protects.

   So a value survives if **any** of a pool of references accepts it, and the
   pool is seeded from the whole candidate space rather than from the first
   round's result — a pool drawn from inside a collapsed domain cannot escape
   it. With that, the same axes measure their true domains, step 4 finds the
   divisibility rule, and the loss falls to 4 points in 1024 (`pool_cap=32`,
   0.11s). The knob buys coverage only; the result is sound at any setting.

2. **Compress each axis** into the most compact sound form, in order of
   preference: unconstrained; a stride-range `{min, max, stride}` when `S[a]` is
   an arithmetic progression; an explicit value list otherwise. String axes
   (`kind: "str"`, e.g. a free `dtype`) always compress to a value list.

3. **Test whether legality factorizes.** Ask whether legality is exactly the
   product of the per-axis sets by sampling the cross product and comparing
   against the gate:

```python
coupled = [p for p in sample(cross(S)) if not gate(make_spec(**{**baked, **p}))[0]]
```

   If `coupled` is empty on a good sample, the factored form stands. This step
   is not optional: `legal_point` exists precisely because "a cross product
   built from two independently-legal axis lists can contain points the kernel
   refuses to build".

4. **If coupled, discover the rule and verify it.** Try a small library of
   candidate relations and keep only those that reproduce the gate's verdict on
   every sampled point: divisibility between two free axes (`a % b == 0`),
   divisibility by a baked value (`a % 64 == 0`), ordering or equality
   (`a <= b`, `a == b`), and a product bound (`a * b <= K`). This mirrors how
   the roller fits constants — a small candidate set, accepted only on
   verification.

5. **Fall back conservatively if nothing fits.** Four sound forms, most
   permissive first; take the most permissive one that *verifies*:

   | Form | When |
   |---|---|
   | factored per-axis membership | legality factorizes |
   | factored membership + a verified coupling rule | step 4 found one |
   | factored membership + an enumerated blocklist of illegal combinations | the exceptions are few |
   | an enumerated allowlist of legal points, or the verified points alone | nothing else verifies |

6. **Run the oracle before shipping** — see below.

7. **Attach the verified points** the roller actually checked, kept separate from
   legality (§7).

#### The oracle: verify the derived function against the gate

The derivation must not be trusted because it looks right. It gets the same
treatment the roller's output gets: check it against the authority, on points it
was not fitted to, and fail generation if it disagrees in the unsafe direction.

```python
for b in fresh_sample(space, exclude=points_used_in_derivation):
    guard_ok, _  = check_guard(guard, b)
    gate_ok, _   = gate(make_spec(**{**baked, **b}), arch=arch)
    if guard_ok and not gate_ok:
        raise GuardUnsound(b)        # hard failure: narrow the guard, do not ship
    if gate_ok and not guard_ok:
        coverage_loss.append(b)      # sound but strict: report, do not fail
```

An unsound guard is a build break. A strict guard is a number in the generation
report, which is also the signal for when a derivation form is too blunt.

#### What gets emitted

Ordered rules, first failure reported, so a rejection names the axis and the
constraint rather than just returning false:

```json
"guard": {
  "schema": "rocke.guard/v1",
  "free": ["block_size", "head_size"],
  "rules": [
    {"reason": "head_size must be in [16, 256]",
     "pred": {"mul": [{"ge": [{"spec": "head_size"}, 16]},
                      {"le": [{"spec": "head_size"}, 256]}]}},
    {"reason": "head_size must be 16 plus a multiple of 16",
     "pred": {"eq": [{"mod": [{"sub": [{"spec": "head_size"}, 16]}, 16]}, 0]}},
    {"reason": "block_size must divide head_size",
     "pred": {"eq": [{"mod": [{"spec": "head_size"}, {"spec": "block_size"}]}, 0]}}
  ],
  "verified": [{"block_size": 16, "head_size": 64},
               {"block_size": 32, "head_size": 128}],
  "derivation": {"method": "coupled", "gate": "supports_tiled_2d", "arch": "gfx950",
                 "reference": {"block_size": 32, "head_size": 128}, "references": 4,
                 "measured": {"block_size": 3, "head_size": 16},
                 "probed": 48, "exhaustive": true,
                 "oracle": {"checked": 256, "agreed": 256, "unsound": 0, "strict": 0}}
}
```

Every `pred` is an ordinary intexpr, desugared, so **today's evaluators run it
unchanged** — `mul` for conjunction, `eq`/`ge`/`le` for the comparisons. No new
node kinds means the CI gate that already pins `rv_int` against
`eval_intexpr` covers guards without being extended.

Rule **order** is part of the contract. Evaluation stops at the first failure,
and the bounds rule on an axis is emitted ahead of the divisibility rule on the
same axis, because the two evaluators do not agree about `mod` with a negative
left operand (Python floors, C truncates). Today that ordering is belt and
braces rather than the only defence — the rules ask whether a remainder is
*zero*, which is the same question under either convention — but a rule that
divided, or compared a remainder against anything else, would depend on the
ordering alone.

The `derivation` block is provenance: which form was used, against which gate,
how much was probed, and what the oracle found. It is what makes a suspicious
guard reviewable a year later.

#### Evaluation

One implementation per language, both trivial, and the C one is what the VM calls
before replay (§3.2):

`utils/recipe_expand.py::check_guard(guard, spec_int, spec_str, *,
require_verified=False) -> (ok, reason)` is the Python side;
`rv_guard_eval` in `recipe_vm.cpp` is the C side. Both check that every axis in
`free` is bound before evaluating anything, so "the caller forgot an axis" is a
different answer from "this shape is unsupported".

Loop variables are passed as `{}` because a guard is evaluated before any
`static_for` exists — a guard referencing `{"var": ...}` is malformed.

An **unknown guard schema is a refusal, not an accept**. An engine older than
the bundle cannot know what a newer guard would have rejected, and the one thing
it must not do is admit a configuration on the strength of not understanding it.

#### Cost

Measured, on the five gfx950 attention families, with the declarative gate (spec
constructor plus `supports_*`):

| | |
|---|---|
| Derivation, per recipe | under 10 ms; 0.11 s for the hardest coupled pair |
| Guard size | 368–809 B of CBOR |
| Enforcement at JIT time | 0.077 ms on an 18.1 KiB bundle … 2.575 ms on a 548.7 KiB one |

Adding the build probe to the gate changes the first row by orders of magnitude
— it compiles a kernel per gate call — which is why it is off by default and why
`pool_cap`/`pool_scan` are tunable. The other two rows are unaffected.

That last row is worth reading correctly, because it scales with **bundle size,
not guard complexity**, and §2.4 now measures the split directly.
`rocke_bundle_contains` evaluates no rules at all and costs the same as the full
guard check — 0.075 against 0.077 ms on the small bundle, 2.564 against 2.575 ms
on the large one. So the cost is `rocke_cbor_parse` walking the artifact, and
everything guards add on top of "does this key exist" is 2–11 µs. Rule
evaluation is genuinely a few integer comparisons; reaching the rules is the
expense, because every call parses from scratch and frees its arena on return.

Checking before compiling is still clearly right for a small bundle — 0.077 ms
against ~2 ms to finish a GEMM. For a large one it is much closer than the
ordering argument in §9 assumes: 2.6 ms to check an attention candidate exceeds
a whole GEMM cold JIT. A caller filtering hundreds of candidates needs the
parse-once handle (item 4), not a faster evaluator.

#### Limits worth stating plainly

- **The guard is only as honest as the gate.** If `supports_*` fails to reject
  something (§1.1), the derived guard inherits the hole — soundness is relative
  to the gate, not to the hardware. That is why §4 asks for the gate holes to be
  closed.
- **Runtime kernel arguments are out of scope.** Values that are not spec values
  cannot appear in a guard at all (§4, fifth item).
- **Legality is not the same as verified.** The rules answer "would the gate
  accept this"; the `verified` set answers "did we ever check that the recipe
  emits the right code here". §7 keeps them separate deliberately.

### 3.3a The C API hipDNN calls

Declared in `cpp/include/rocke/recipe_guard.h`, implemented in the recipe VM's
translation unit so that it reuses `rv_int` — one intexpr evaluator, not two.

```c
rocke_status_t rocke_bundle_check_guard_cbor(
    const unsigned char* data, size_t len,
    const char* key, const char* arch,
    const rocke_recipe_spec_int_t* ints, int n_ints,
    const rocke_recipe_spec_str_t* strs, int n_strs,
    unsigned flags, rocke_guard_verdict_t* out_verdict,
    char* reason, size_t reason_cap);

rocke_status_t rocke_recipe_check_guard_cbor(/* same, for a standalone recipe */);

bool rocke_bundle_contains(const unsigned char* data, size_t len,
                           const char* key, const char* arch);
```

Three deliberate choices in that shape:

**The verdict is separate from the status.** "This shape is not supported" is a
normal result of a working call, not a failure of the call. Folding them
together would force a caller to distinguish a routing decision from a corrupt
bundle by parsing an error string. `ROCKE_OK` plus `ROCKE_GUARD_REFUSED` means
route elsewhere; a non-OK status means something is wrong with the bundle or the
engine. Passing `NULL` for `out_verdict` collapses a refusal into
`ROCKE_ERR_VALUE` for callers that only want pass/fail.

**Absence is reported, not assumed.** `ROCKE_GUARD_ABSENT` distinguishes "this
recipe is concrete, so its presence in the bundle is the validity statement"
from "this rolled recipe was generated without a guard" — an ungoverned bundle,
which a caller wanting enforcement should be able to notice rather than infer
safety from silence. Both return `ROCKE_OK`, so guards stay additive and every
recipe that replayed before them still replays.

**A missing key is `ROCKE_ERR_KEY`, not a refusal.** For a pruned bundle absence
*is* the rejection for concrete recipes (§3.1), so one call is the complete
admission test for both kinds of recipe — while still letting a caller tell "we
never built this" from "we built it and it will not serve this shape".

`ROCKE_GUARD_REQUIRE_VERIFIED` narrows acceptance to points the generator
actually built and compared, giving up the rolled interior for the strongest
available evidence. Reasonable during bring-up or a conformance run; expensive
as a steady state. §7 is where that policy choice belongs.

The same guard is enforced **inside** `rv_run_root`, before the builder is
created and before any op is emitted, so a caller who never calls the check
cannot compile an unsupported configuration anyway. The standalone API exists so
the answer can be had without paying for a failed replay.

### 3.3b What it does on the real families

`drivers/derive_guards.py` runs the whole thing against the gfx950 attention
families and their own gates. All five derive a guard that agrees with the gate
on every sampled point:

| Family | Axes | Method | Oracle |
|---|---|---|---|
| `attention_dense` | head_size | factored | 16/16, 0 unsound, 0 strict |
| `attention_tiled_2d` | head_size, block_size | factored | 256/256, 0, 0 |
| `attention_tiled_3d` | head_size, block_size | factored | 256/256, 0, 0 |
| `attention_reduce` | head_size | factored (stride) | 16/16, 0, 0 |
| `fastkv_regp` | head_size, block_size | factored | 256/256, 0, 0 |

The grouped-query axes are the interesting case, and the one that drove the
pooled measurement in step 1. On `attention_tiled_2d` over
`num_query_heads`/`num_kv_heads` the derivation lands on `blocklist` and
discovers, on its own, that **`num_kv_heads` must divide `num_query_heads`** —
which is exactly the constraint the sweep's module docstring records these
kernels as *failing to enforce themselves*. The guard is therefore stricter than
`supports_tiled_2d` there only because the sweep's `coherent` predicate was
supplied as part of the gate; without it, the derivation would faithfully
reproduce the kernel's hole. That is the §4.3 argument in miniature: **a guard
can only be as honest as its gate.**

`--roll` closes the loop on a real kernel: roll `attention_tiled_2d` over
`num_kv_heads`, attach a guard derived from `supports_tiled_2d`, write the
bundle, and ask the C API. `num_kv_heads=32` lowers to 1135 lines of LLVM IR;
`num_kv_heads=17` is refused, by the standalone check and by the VM, with
`num_kv_heads must be one of {16, 32, 64}`.

### 3.3c Launch metadata: what the caller needs after the guard says yes

The guard answers *may I compile this*. A JIT caller then has to answer *what do
I launch*, and until recently the bundle could not help it.

The gap was narrow and total. A client could take a CBOR bundle to a correct
`.ll` with no Python in the process, hand that to comgr, get a HSACO — and then
be stuck holding a compiled kernel with no idea what grid to launch it with. The
grid was never in the bundle. It lived in host Python, as expressions like
`(n + tile_n - 1) // tile_n` inside a dispatch function, alongside a
hand-written argument signature. So the final step of the chain was the one step
that could not be taken without an interpreter, which is a strange place for the
"can run without Python" path to end.

A grid is a function of the shape, and the recipe language already exists to say
exactly that. Geometry is therefore carried as intexprs over the spec axes and
evaluated by the same `rv_int` that computes every loop bound the recipe emits:

```json
"launch": {
  "grid":  [{"div": [{"add": [{"spec": "N"}, 2047]}, 2048]}, 1, 1],
  "block": [256, 1, 1],
  "lds_bytes": 0
}
```

Nothing has to be kept in sync by hand, because there is only one copy: the
geometry ships in the same artifact as the kernel it launches, is derived from
the same axes, and is covered by the same guard and ABI checks. The argument
signature is *not* carried, because it does not need to be — the recipe's own
`param` instructions already declare it in order, so the plan reports what the
recipe actually declared and cannot disagree with the kernel built from it.

```c
rocke_status_t rocke_bundle_plan_launch_cbor(
    const unsigned char* data, size_t len,
    const char* key, const char* arch,
    const rocke_recipe_spec_int_t* ints, int n_ints,
    const rocke_recipe_spec_str_t* strs, int n_strs,
    rocke_launch_plan_t** out_plan, char* err, size_t err_cap);

const char* rocke_launch_plan_kernel_name(const rocke_launch_plan_t*);
bool        rocke_launch_plan_geometry(const rocke_launch_plan_t*,
                                       rocke_launch_dims_t* grid,
                                       rocke_launch_dims_t* block,
                                       unsigned* lds_bytes);
int         rocke_launch_plan_num_args(const rocke_launch_plan_t*);
const rocke_arg_desc_t* rocke_launch_plan_arg(const rocke_launch_plan_t*, int i);
unsigned    rocke_launch_plan_kernarg_size(const rocke_launch_plan_t*);
```

Choices worth calling out, in the same spirit as §3.3a:

**Absence is reported, not defaulted.** A recipe with no `launch` block returns
`false` from `..._geometry` rather than a 1×1×1 grid. A recipe recorded before
geometry existed is not the same as a kernel that wants one workgroup, and
defaulting would convert missing metadata into a silently wrong launch at the
point where it is hardest to notice. Same reasoning as `ROCKE_GUARD_ABSENT`.

**A refused shape cannot be planned.** Building the plan replays the recipe, so
the guard applies; planning a launch for a shape the kernel will not serve is
not a meaningful question to answer.

**Kernarg offsets follow natural alignment, and the size deliberately does
not round up.** Each argument sits at an offset aligned to its own size — 8 for
pointers and `i64`, 4 for `i32` and `f32`. This is invisible until a signature
mixes widths and then wrong for everything after the mix: `(ptr, i32, ptr)` puts
its trailing pointer at 16, not 12. The *total*, though, is the end of the last
argument and is **not** rounded up to the widest alignment, even though the
AMDGPU metadata's kernarg segment size is. That matches `runtime/packing.py`,
which packs a GEMM's `(ptr,ptr,ptr,i32,i32,i32)` as 36 bytes rather than 40 and
has been running that way. Reporting 40 would have C callers size their buffer
differently from every Python caller for the same kernel — the kind of
divergence that surfaces as an intermittent fault rather than a test failure. If
the convention ever changes it has to change in both engines together.

**Cost.** Building a plan replays the recipe, which is the same work as lowering
it. A caller wanting both the `.ll` and the plan pays that twice — roughly 1ms,
against a JIT compile that costs far more and is cached afterwards. Keeping the
two calls independent was judged worth the millisecond.

`drivers/launch_from_bundle.py` closes the loop on real hardware, and is
deliberately forbidden from importing anything from the kernel family that
authored the recipe — no `elementwise_grid`, no `elementwise_signature`. If the
bundle did not carry enough to launch, it could not run:

```text
elems_per_block=2048   (grid must be ceil(N/2048))
N=2049     grid=(2, 1, 1) block=(256, 1, 1) kernarg=28B  OK
N=100000   grid=(49, 1, 1) block=(256, 1, 1) kernarg=28B  OK
```

The sizes are deliberately not multiples of the slab, so a grid that failed to
round up would leave a tail unwritten and the comparison against numpy would
catch it. `OK` means the output matched elementwise. The 28 bytes are
`(ptr,ptr,ptr,i32)` — three 8-byte pointers and an `i32` at offset 24, all of it
reported by the C engine rather than known in advance by the driver.

### 3.3d Version skew between the two engines

A bundle is a persisted artifact. It is written by Python at build time and read
by C inside hipDNN, which may have been built earlier or later. Compatibility is
therefore a property of the artifact, decided per artifact, not a property of
the process — and until this landed the two engines did not even agree on what
they would accept: the C VM checked the recipe `schema` and the Python expander
checked nothing, so the oracle would happily replay a recipe the engine it
mirrors refuses.

Two things can be mismatched, for different reasons, so they get two numbers
(`cpp/include/rocke/abi.h`). Folding them into one would mean a new recipe
instruction invalidates every hipDNN binary, and a struct change invalidates
every bundle on disk; neither is true.

| Number | Question it answers | Checked |
|---|---|---|
| `ROCKE_ABI_VERSION` | Does this header match this `.so`? Structs, enums, signatures. | Once at load |
| `ROCKE_RECIPE_ABI` | Can this engine read this CBOR artifact? | Per artifact, both readers |

The wire check is **not** "artifact version equals mine". Each artifact declares
the *oldest reader that can read it correctly*, and a reader refuses exactly
when `min_reader` exceeds its own level:

```json
"abi": {"min_reader": 1, "writer": 1, "engine": "1.0.0+20260812", "build_id": "6bc59f33fd11"}
```

`writer`, `engine` and `build_id` are provenance for tracing a bad artifact;
nothing compares them. A monotonic version compared for equality would reject
newer artifacts wholesale whether or not they use anything new, turning a
generator upgrade into a flag day for every deployed engine over recipes it has
always been able to read. A **missing** block means level 1, so bundles recorded
before this existed still replay — the same additive rule guards follow.

`min_reader` is *derived* from what the recipe uses, never hand-set: a declared
requirement is a second copy of the truth and drifts the first time someone
forgets, which is §1.3's problem in a new place.

Two limits worth stating plainly, because they bound what the number is worth.
Both VMs already fail loudly on an unknown instruction op, opcode or intexpr
node, so a genuinely new construct is self-policing and the stamp mostly
improves the error message; the bump exists for changes an old engine would
*accept and get wrong*. And attribute **values** are passed through to the
builder uninterpreted, so their meaning is the lowerer's contract — a lowerer
silently ignoring an attribute it does not know is not something this can catch.

For the guard API specifically: a bundle too new for the engine returns
`ROCKE_ERR_VALUE`, **not** `ROCKE_GUARD_REFUSED`. A refusal means route
elsewhere and carry on; this means the deployed engine and the shipped artifacts
do not match, which no amount of falling back will fix. Reporting it as a
refusal would file a deployment fault under "unsupported shape", where it shows
up as a quiet loss of coverage that nobody investigates.

### 3.4 Pruning at generation time

The pruning half of the question is the easier half, with one missing piece.

**The primitives exist.** `legal_values`, `legal_point` and `choose_grid` in the
sweep are precisely a "keep only combinations the kernel accepts" filter, and
`choose_grid` already does the non-obvious part — adding axes one at a time,
keeping only values that stay legal against every combination chosen so far,
rather than dropping a whole axis at the first conflict.

**The generator that would consume them does not exist.** Today's bundle
generator takes a hand-written list:

```python
def record_concrete_bundle(cases: List[Tuple[Any, str]]) -> List[Dict[str, Any]]:
    """cases: list of (build_callable, arch). Records each kernel's emitted IR
    into a concrete recipe (universal, byte-identical), keyed by kernel name."""
```

There is no enumeration step, so there is nothing to prune yet — the caller is
assumed to have chosen valid cases. The sweep enumerates, but only to *measure*;
it calls `cbor_encode` for byte counts and throws the result away.

So the work item is a generator that enumerates a declared axis space, prunes it
with the kernel's gate, records what survives, rolls where rolling succeeds, and
emits the guard alongside each recipe. That is one driver, and it composes
existing parts. Its cost is dominated by recording, which is why pruning first
matters: every combination rejected by the gate is a recording not taken and a
recipe not shipped.

---

## 4. What the library kernels need for this to work

Strictly, nothing — the design in §3 derives everything from gates that already
exist. But it inherits their defects, and four are worth fixing before a
generated catalog depends on them. These are the answer to "is there anything we
should add to `rocke/library` kernels".

**1. A uniform admission entry point.** The gates disagree on their own calling
convention: `supports_attention_dense(spec, *, arch)` takes a spec object,
`supports_tiled_2d(*, head_size, block_size, dtype, ...)` takes nineteen
keyword-only arguments, and `instances/common/*` uses `is_valid_spec(spec, arch)`.
The sweep pays for this in hand-written adapters (`admits_2d`, `admits_3d`,
one per family) whose only job is to unpack a spec into keywords. A generator
that walks every family needs to call the gate generically. Proposal: each
kernel module exports

```python
def supports(spec, *, arch: str) -> Tuple[bool, str]: ...
```

with the existing functions kept as thin wrappers so no caller breaks.

**2. Declared axes and their candidate domains.** The generator has to know what
to enumerate. Today that knowledge lives in the sweep driver as a hardcoded
`CANDIDATES` dict, "deliberately wider than any kernel accepts — the point is to
let the kernel do the rejecting". That is the right instinct in a survey and the
wrong place for it in a shipping generator: a kernel gains an axis and the
generator silently never explores it. Proposal: each kernel declares its tunable
axes and the values worth probing, next to the spec they belong to.

**3. Close the known holes in the gates, because pruning will trust them.** The
`num_kv_heads` divisibility hole is the live example: neither `__post_init__`
nor `supports_*` rejects a `num_kv_heads` that does not divide
`num_query_heads`, and the kernel then "builds and bakes in a group size that
does not correspond to any real grouping". The sweep compensates with a
driver-local `coherent` predicate. If a generator prunes with the kernel gate
alone, it will happily record and ship recipes for those combinations, and the
derived guard will admit them at runtime. This constraint belongs in the kernel.
It is worth auditing the other families for the same class of omission — the
generator is only as trustworthy as the gate it prunes with.

**4. Hoist builder-deep assertions into the gate.** `legal_values` needs its
third layer (`probe=build`) because some constraints are asserted inside the
emitter — the tiled kernels' head-size stripe alignment is the cited case. Every
such constraint hoisted into `supports_*` has three payoffs: generation-time
pruning gets much cheaper (no build probe per candidate), the derived guard
becomes complete rather than approximate, and the failure surfaces as a
structured reason instead of an exception from deep in a builder. Where a
constraint genuinely cannot be hoisted, the build probe stays as a
generation-time backstop — it just must not be the *only* thing that knows.

**A fifth item that is not about the kernels' gates but will bite anyway.**
Some constraints are on *runtime kernel arguments*, not on spec values, so no
recipe guard can ever see them. `gemm_universal::is_valid_spec` documents this
explicitly for split-K:

> We can only validate the K-slice divisibility at build time when K is a
> compile-time fact, which it is not in the universal body (K is a runtime arg)
> … The `K % split_k` and `ks % tile_k` divisibility are the caller's
> responsibility.

If hipDNN is now the caller, it needs those rules, and today they exist only as
a comment. They should be declared as launch-time constraints, separate from the
spec guard, so the runtime can enforce what the recipe structurally cannot.

---

## 5. The broader path: validity as data, in the language the VM already speaks

**Not on the critical path for hipDNN.** §3 meets the "never compile an invalid
configuration" requirement without any of this. What follows is what it would
take to give C++ a *standalone* validity answer for an arbitrary spec —
independent of whether a recipe exists for it. Reach for it when one of these
becomes the binding constraint:

- hipDNN needs to distinguish "this instance is invalid" from "this instance is
valid but no recipe was shipped", rather than treating both as skip.
- The virtual catalog is large enough that filtering by bundle lookup alone is
too coarse, and heuristics should be told *why* a candidate was rejected.
- The 58 hand-ported C99 validators (§1.3) become a maintenance problem in their
own right, independent of the JIT flow.

The first two are product questions for hipDNN (§11). The third is true already
and is the reason item 1 in §10 is worth doing regardless.

### 5.1 Why the recipe expression language is the right substrate

The argument is the one made in [What an `intexpr` is](#what-an-intexpr-is), so
only the conclusion is repeated here: the recipe's expression language already
has integer arithmetic, the six comparisons and enum equality; it already has two
evaluators (`recipe_expand.py::eval_intexpr` in Python, `recipe_vm.cpp::rv_int`
in C99); the byte-identity gates already fail if those two ever disagree; and the
VM already evaluates such an expression *as a truth value* in `static_if`.

So "a predicate over spec values that C can evaluate and Python agrees with" is
not a capability to be built. Expressing validity in this language means validity
inherits an existing gate rather than needing a new one — which is the entire
reason to prefer it over a bespoke rule format.

### 5.2 What to add to the language

Additive, no change to existing behaviour:


| Addition    | Form                                                                                           | Notes                                                                                                      |
| ----------- | ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| Boolean ops | `{"and": [e, ...]}`, `{"or": [e, ...]}`, `{"not": e}`                                          | Desugarable to `mul`/`max`/`eq(x,0)`, but explicit ops keep the data readable and the reasons attributable |
| Membership  | `{"in": [e, [64, 128, 256]]}`, `{"str_in": ["dtype", ["fp16","bf16"]]}`                        | Sugar over an `or` of `eq`; by far the most common rule shape                                              |
| `min`/`max` | `{"min": [a, b]}`                                                                              | Needed by the LDS aliasing model (`max(ab, c)` when C aliases A/B)                                         |
| Arch facts  | `{"arch": "lds_capacity_bytes"}`, `{"arch": "wave_size"}`, `{"arch": "max_threads_per_block"}` | Reads from the arch table, keyed by the arch passed to the evaluator                                       |
| MMA catalog | `{"arch_has_mma": [family, a_dtype, b_dtype, c_dtype, m, n, k]}`                               | Mirrors `target.mma.has_shape`; each operand may itself be an expression                                   |


### 5.3 The rules document

Ordered rules, first failure wins — which preserves exactly the contract the
hand-written C ports already promise ("a faithful, **ordered** port … each
rejection writes the same single-line reason"):

```json
{
  "schema": "rocke.valid/v1",
  "family": "attention_tiled_2d",
  "arches": ["gfx950"],
  "params": [
    {"name": "head_size", "kind": "int"},
    {"name": "block_size", "kind": "int"},
    {"name": "num_warps", "kind": "int", "default": 1},
    {"name": "block_m_per_warp", "kind": "int", "default": 16},
    {"name": "num_query_heads", "kind": "int"},
    {"name": "num_kv_heads", "kind": "int"},
    {"name": "dtype", "kind": "str"},
    {"name": "use_fp8", "kind": "int", "default": 0},
    {"name": "kv_storage_dtype", "kind": "str", "default": ""}
  ],
  "rules": [
    {"require": {"str_in": ["dtype", ["fp16", "bf16"]]},
     "reason": "tiled 2D kernel currently supports fp16/bf16"},
    {"require": {"in": [{"spec": "head_size"}, [64, 128, 256]]},
     "reason": "tiled 2D kernel only supports head_size in {64,128,256}"},
    {"require": {"eq": [{"mod": [{"spec": "head_size"}, 32]}, 0]},
     "reason": "tiled 2D kernel requires head_size divisible by 32"},
    {"require": {"in": [{"spec": "num_warps"}, [1, 2, 4, 8]]},
     "reason": "tiled 2D kernel requires num_warps in {1,2,4,8}"},
    {"require": {"or": [{"ne": [{"spec": "block_m_per_warp"}, 32]},
                        {"in": [{"spec": "num_warps"}, [1, 2, 4]]}]},
     "reason": "block_m_per_warp=32 requires num_warps in {1,2,4}; 8 warps of 32 rows exceeds the 1024-thread CTA cap"},
    {"require": {"or": [{"eq": [{"spec": "use_fp8"}, 0]},
                        {"spec_str_eq": ["kv_storage_dtype", "fp8e4m3"]}]},
     "reason": "use_fp8=True requires kv_storage_dtype='fp8e4m3'"},
    {"require": {"eq": [{"mod": [{"spec": "num_query_heads"}, {"spec": "num_kv_heads"}]}, 0]},
     "reason": "num_kv_heads must divide num_query_heads"}
  ]
}
```

Note the last rule: it is the constraint the kernels *fail* to enforce today
(§1.1). Writing the rules down is what makes that omission visible.

Evaluation contract, matching the existing `(ok, reason)` shape on both sides:

```c
bool rocke_spec_is_valid(const rocke_valid_rules_t* rules,
                         const rocke_recipe_spec_int_t* ints, int n_ints,
                         const rocke_recipe_spec_str_t* strs, int n_strs,
                         const char* arch, char* reason, size_t reason_cap);
```

```python
ok, reason = rocke.portable_ir.validity.check(rules, spec_int, spec_str, arch)
```


### 5.4 Three tiers, and being honest about the third

Not every predicate will reduce to rules, and pretending otherwise is how a
data-driven scheme quietly becomes a lying one. Classify up front:

- **Tier 1 — pure rules.** The attention `supports_`* family, and most of
`instances/common`. Expressible with §5.2 and nothing else.
- **Tier 2 — rules plus arch facts and a named derived quantity.** GEMM/conv LDS
budgets. The budget formula is a pure integer function, so it can be a named
expression in the document (`"derived": {"ab_lds": ...}`) referenced by a rule.
The risk is that `_ab_lds_plan` is *shared with the emitter* on purpose; moving
it to data creates a second copy unless the emitter reads the same expression.
Recommendation: make the emitter's helper read the rules document's derived
expression, so there is still exactly one definition.
- **Tier 3 — builder-deep assertions.** Not declarable without hoisting them.
Do not attempt to encode them; instead ensure the replay fails cleanly, and
treat any Tier 3 rejection discovered at build time as a bug to be hoisted
into Tier 1.


### 5.5 Migration, with a gate at every step

The existing Python predicates are the oracle. Do not delete them; shadow them.

1. Author `rocke.valid/v1` for one family (`attention_tiled_2d` is the right
  first target: Tier 1, JIT-relevant, and its C99 mirror already exists to
   compare against).
2. Add a differential test that sweeps a candidate grid — reuse `CANDIDATES`
  from `roll_gfx950_sweep.py` — and asserts three-way agreement between the
   Python predicate, the data-driven evaluator, and, where it exists, the
   hand-written C99 validator via the `rocke_engine.*_is_valid` binding. Compare
   `ok` strictly; compare `reason` as a warning at first, since the C ports
   already promise reason parity and we do not want to weaken that.
3. Once agreement holds, make the Python predicate a thin wrapper over the rules
  document, and delete the hand-written C99 body in favour of the shared
   evaluator. Two implementations become one document and one evaluator pair
   that CI already watches.
4. Generate the C++ arch tables from `arch_specs.json` instead of transcribing
  them, so `{"arch": ...}` lookups cannot drift either.

Sizing: the differential harness and evaluator are the real work; per-family
authoring is mechanical afterwards, and roughly 60 families is a long tail that
can be migrated in JIT-priority order rather than all at once.

---

## 6. Recipe keys and `does_recipe_exist`


### 6.1 What to add to the recipe

Make the baked/free split explicit instead of implicit in a name:

```json
{
  "schema": "rocke.recipe/v1",
  "family": "attention_tiled_2d",
  "arch": "gfx950",
  "spec":  [{"name": "seqlen_kv", "kind": "int"}, {"name": "num_query_heads", "kind": "int"}],
  "baked": {"head_size": 128, "block_size": 16, "dtype": "fp16", "num_warps": 4},
  "guard": { ... },
  "kernel_name_fmt": "...",
  "program": [ ... ]
}
```

- `spec` — free at replay (unchanged meaning).
- `baked` — the non-free parameters and the values they were recorded at. This
is the lookup key, in structured form.
- `guard` — the per-recipe validity gate over the free axes, derived at
generation time (§3.3) and enforced before replay (§7).

The key becomes a canonicalization of `(family, arch, baked)` — sorted by
parameter name, with a fixed scalar spelling, e.g.
`attention_tiled_2d/gfx950/block_size=16,dtype=fp16,head_size=128,num_warps=4`.
Both sides must produce the identical string from the identical map, so
canonicalization belongs in shared code with a round-trip test, not in each
caller. This also closes gap 8 on its own terms: the key stops being a kernel
name that happens to embed spec values.

### 6.2 What to add to the C API

```c
/* Parse once, query many. */
rocke_status_t rocke_bundle_open(const unsigned char* data, size_t len,
                                 rocke_bundle_t** out, char* err, size_t err_cap);
void           rocke_bundle_close(rocke_bundle_t*);

/* Existence + handle, no lowering. Returns ROCKE_ERR_NOT_FOUND cleanly. */
rocke_status_t rocke_bundle_find(const rocke_bundle_t*, const char* family, const char* arch,
                                 const rocke_recipe_spec_int_t* baked_i, int n_baked_i,
                                 const rocke_recipe_spec_str_t* baked_s, int n_baked_s,
                                 rocke_recipe_handle_t* out);

/* Run a handle we already found, binding the free params. */
rocke_status_t rocke_recipe_run_handle(rocke_recipe_handle_t, ...);
```

Three properties matter for a virtual catalog: parse the bundle **once** rather
than per candidate; answer existence **without** lowering; and index the keys
(hash on the canonical string) rather than scanning. "Not found" must be an
ordinary return value, since on your flow it is the common case that moves to
the next candidate.

---

## 7. Validity at replay time

### 7.0 Does recording and rolling settle validity?

It is tempting to conclude that because a recipe was produced by a generator that
ran the real Python gate, and verified against fresh Python builds, nothing is
left to check at JIT time. That is true for a concrete recipe and false for a
rolled one — and the asymmetry is the opposite of the intuition, because **rolling
is what creates the need for a runtime check, not what removes it.**

| | Concrete recipe (`"spec": []`) | Rolled recipe (free axes) |
|---|---|---|
| What the caller supplies at JIT time | nothing | a value per free axis |
| Can the caller express an invalid spec? | No — there is nothing to vary | **Yes** — any integer at all |
| Spec validation needed at JIT time? | **No.** The lookup succeeding *is* the validation | **Yes**, over the free axes |
| Constraints on runtime kernel arguments | **Yes, still** (§4, fifth item) | **Yes, still** |

The reason is that rolling generalizes *how the kernel is emitted*, not *which
values are legal*. A recipe rolled over `seqlen_kv` from recordings at 512 and
1024, verified at 2048, will replay just as willingly at 513, at 0, or at
100 000. Nothing in it knows those are illegal, because nothing in it was ever
asked to carry the legality domain — the roller's job was to prove the emission
pattern, and it did that only at the points it checked. Two further edges follow
from the same cause: where several axes are free, the legal region is usually
*not* the cross product of the per-axis domains (§8.1), and a caller can bind a
combination the roller never verified even if each value individually appears in
its recordings.

So the guard in §3 is not a redundant second opinion on work the generator
already did. It is the generator's knowledge — the legal domain it measured and
the points it verified — travelling with the recipe so that the runtime cannot
step outside it. That is also why the guard should be derived rather than
authored (§3.3): it is a *record* of what generation established, not a new
claim about validity.

### 7.1 Three questions worth separating

Beyond the concrete/rolled split, three questions get conflated, and only the
first is answered by the catalog-time check.

**Is the instance well-formed?** Answered at catalog time, over the non-free
parameters. Nothing changes it later.

**Is the instance well-formed *with this problem*?** Not answered at catalog
time. Constraints that couple free and baked parameters can only be checked once
the problem shape is known — `seqlen_kv % block_n == 0`, `tile_size % block_size == 0`, `num_query_heads % num_kv_heads == 0` when one side comes from
the problem. So the same rules document must be evaluated a second time with the
full binding (baked ∪ free). This is cheap, and it is the same artifact, so it
is not new work beyond calling it twice.

**Is this replay inside the envelope the roller actually verified?** This is the
one that has no answer today, and it is the sharpest risk in the JIT flow. A
rolled recipe is verified at its sampled and held-out points; the safety
property the readiness doc claims is that *refusals are refusals* — the roller
never emits a recipe it could not verify. That is a statement about the points
it checked. Replaying at an arbitrary unverified point is silent extrapolation,
and the failure mode is not a clean error but a plausible kernel. This is what
the `guard`'s `verified` set (§3.3) is for: the roller populates it from the
points it actually checked, and the VM can refuse — or at minimum report —
a binding outside it. Without this, hipDNN's ability to bind free parameters
from the problem space is precisely the ability to walk off the verified region.

Whether "outside verified" should be a hard refusal or a warning is a policy
call worth making deliberately. Hard refusal is the safe default and costs
coverage: the whole point of a rolled recipe is to serve shapes it was never
recorded at, and the roller's held-out verification exists to justify exactly
that extrapolation. A reasonable middle is to refuse outside the *legal* domain
always, and treat outside-`verified`-but-inside-legal as permitted with a
counter, so coverage gaps show up in telemetry rather than as silent risk.

There is also a fourth, residual case: Tier 3 builder assertions (§5.4). Those
ran at record time for the baked parameters, but a free parameter can violate
one. The VM must surface that as a clean failure so the caller falls through to
the next candidate rather than launching something wrong.

---

## 8. How large is the space, and what does it cost to ship


### 8.1 Per-axis domains overstate the space badly

The instinct to bound the catalog by multiplying per-axis legal-value counts
will mislead, because the axes are coupled. `choose_grid` exists precisely
because "naively crossing per-axis sample lists produces points the kernel
rejects". `block_n` is the clean illustration: 5 legal values, not because the
axis is small, but because it must divide `seqlen_kv`. Any estimate must come
from the incremental feasible construction, not the product.

Concretely, the measurement is cheap to run and should be run before committing
to a catalog strategy: `legal_values` over a deliberately over-wide candidate
list, then `legal_point` over the cross product in `build=False` mode. Once the
predicate is data, this costs microseconds per candidate instead of a Python
build probe, so the full space can be characterized rather than sampled.

### 8.2 Recipe count is driven by refusals, not by the valid space

This is the counter-intuitive part, and it is the one that governs build-time
cost. A rolled recipe covers a **cross product**, not a point: conv covers 15
points from 5 recordings in one recipe; attention is 506 KiB as one recipe
against 1518 KiB as three concrete traces. Where the roller succeeds, adding
legal values to an axis costs approximately nothing in shipped artifacts.

Where it refuses, you fall back to one concrete recipe per point, and cost grows
with the size of the space. Currently 7 axes roll across the four probed
families, with 8 refusals, and only 1 of the 7 is tile/warp geometry (gap 5).
The known refusals are specific and documented: `gemm_universal::tile_m`
(non-monotonic, needs opcode selection), `attention_dense::head_size` (only 2
legal values — genuinely not worth rolling), `attention_dense::block_n`
(parametric loop-carry fan, gap 12).

The planning consequence: **the axes hipDNN most wants in the virtual catalog
are tile/warp geometry, and those are exactly the axes that do not roll today.**
If the catalog varies geometry, the build-time recipe count is the number of
geometry combinations, times whatever rolls underneath. Budget from the refusal
list, not from the total space.

### 8.3 Pruning at generation time pays twice

Enumerating and pruning before recording (§3.4) is a strict improvement over
recording-then-discovering, and the saving compounds: a combination the gate
rejects is a recording not taken, a roll not attempted, and a recipe not
shipped. Recording dominates generation cost, so the prune is close to free
relative to what it avoids.

It pays a second time at runtime, for a reason that is easy to miss: a bundle
that contains *only* valid combinations turns the recipe lookup into a validity
filter (§3.1). The pruning step is therefore not merely a size optimization —
it is what lets hipDNN skip invalid candidates without evaluating a predicate
at all.

---

## 9. The proposed flow, annotated

Your pipeline, with today's status and cost per stage:

```
BUILD TIME (Python, authoritative gates available)
  declared axes -> enumerate -> prune with kernel gate      [NEEDS: generator driver]
    -> record -> roll -> derive guard -> attach geometry    [EXISTS: guard + launch]
    -> stamp abi -> CBOR bundle                             [EXISTS: min_reader derived]
       every pruned combination is a recording not taken

RUNTIME (hipDNN, no CPython)
  load librocke                                             [EXISTS]
  |     assert rocke_abi_version() == ROCKE_ABI_VERSION       once per process
  |     a mismatch here is UB, not a wrong answer
  |
  heuristics -> virtual catalog of instances                [hipDNN, exists]
  |
  |-- key = canon(family, arch, non-free params)            [NEEDS: structured key]
  |     recipe = bundle_find(key)                           [NEEDS: existence query + index]
  |     miss -> skip candidate                                ~µs per candidate
  |     (invalid instances miss here by construction: the
  |      generator never recorded them)
  |
  |-- heuristic TFLOP estimate over survivors               [hipDNN, exists]
  |     pick winner
  |
  |-- check_guard(winner, free params from problem)         [EXISTS: recipe_guard.h]
  |     rejects couplings only visible once the problem is   0.08 ms (18 KiB) ..
  |     known; no IR built                                   2.6 ms (549 KiB)
  |     cost is the bundle parse, not the rules -- so check  per call; see §2.4
  |     the winner, not all 200 candidates
  |     too-new bundle -> ERR_VALUE, not REFUSED (§3.3d)
  |
  |-- replay -> .ll -> comgr -> HSACO                       [.ll EXISTS, gated
  |     1.96 ms (GEMM) .. 9.41 ms (attention dense)          byte-identical;
  |                                                          comgr: see below]
  |
  |-- plan_launch(winner) -> name, args, grid, block, lds   [EXISTS: recipe_launch.h]
  |     pack kernargs at the reported offsets
  |
  `-> hipModuleGetFunction + hipModuleLaunchKernel          [hipDNN, exists]
```

Note what is *absent* from the runtime column: a general `is_valid_spec`. The
validity decision was made at build time by the real Python predicate, and its
result is carried by two things — the existence of a recipe under the key, and
the guard on that recipe.

The cost asymmetry still justifies the ordering, but by less than first assumed,
and §2.4 says why: a *key lookup* is microseconds, while a *guard check* costs a
full bundle parse — 0.08 ms for GEMM but 2.6 ms for attention, which is more than
a GEMM cold JIT. Lowering 200 candidates exhaustively costs 0.4–1.9 s, so
filtering first is still the right order by a wide margin; the caveat is that the
filter itself is not free at scale, and guard-checking all 200 against a large
bundle would cost ~0.5 s on its own. Two things follow: check keys first and
guards only on the surviving winner, and prefer item 4's parse-once handle if the
catalog is ever swept. Lower only the winner.

Two refinements worth considering. First, cache on `(family, baked, free, arch)`
— gap 6, currently unimplemented, and comgr dominates the JIT path for small
kernels (1.5–2.4 ms of a 1.96–9.41 ms cold JIT, 84% of a GEMM), so a hit removes
most of the remaining cost. Second, "no recipe shipped" is a signal worth recording: if
hipDNN's heuristics repeatedly select instances with no recipe, that is direct
feedback on what the build-time generator should record next.

### 9.1 The two links rocke does not provide in C

Everything rocke owns in that column now exists in C: admission, replay to `.ll`,
and the launch plan. Two links do not, and both sit at the boundary where rocke
stops being about IR:

- **`.ll` → HSACO.** Only a Python ctypes wrapper around `libamd_comgr`
  (`runtime/compile.py`). There is no C++ wrapper in this tree — a
  `rocke::Compiler` appears in `tests/instances/jit_demo.cpp` but no such class
  is defined anywhere, so that file does not build.
- **HSACO → launch.** Likewise `runtime/launcher.py` over `hipModule*`.

This is plausibly the right boundary rather than a gap: hipDNN already links
comgr and HIP and has its own module cache, and a rocke-shaped wrapper would be
a second one to keep in step. But the Python wrappers are the only executable
statement of what the compile flags and the code-object bundling must be, so a
C++ caller reimplementing them is copying from a script rather than calling an
API. Worth an explicit decision (§11, question 7) — and `jit_demo.cpp` should
either get its class or be deleted, because a non-building example is worse than
no example.

---

## 10. Work items

Ordered so that the narrow path (§3, §4) lands first and the broader one (§5)
stays optional.

**Narrow path — delivers "never compile an invalid configuration".**


| #   | Item                                                                                           | Status | Notes                                                                           |
| --- | ---------------------------------------------------------------------------------------------- | ---------- | ------------------------------------------------------------------------------- |
| 1   | Recipe `guard` field                                                                           | **done** | Backward compatible: a recipe without one replays exactly as before. `family`/`baked`/canonical key (gap 8) still open |
| 2   | Guard derivation over a rolled recipe's free axes (§3.3)                                       | **done** | `src/guard.py`. Derived, never hand-authored — otherwise §1.3's two-copies problem returns |
| 2b  | Guard oracle: derived guard vs the real gate on unfitted points                                | **done** | Unsound ⇒ `GuardDerivationError` at generation time; merely strict ⇒ reported   |
| 3   | VM enforces `guard` before replay; standalone check for the no-lowering query                  | **done** | `rocke_bundle_check_guard_cbor` + `rocke_recipe_check_guard_cbor` (§3.3a); mirrored in `recipe_expand.py`, parity-tested |
| 4   | Bundle open/find/run-handle C API with a key index                                             | partial | `rocke_bundle_contains` answers existence without lowering; the parse-once handle is still open, and §3.3's cost note is the argument for it |
| 5   | Enumerating generator: declared axes → prune with the kernel gate → record → roll → emit guard | open | `drivers/derive_guards.py --roll` does this for one family; generalizing it still replaces `record_concrete_bundle`'s hand-written case list |
| 5b  | Recipe `launch` block + C API for kernel name, kernarg layout and geometry (§3.3c)             | **done** | `recipe_launch.h`; geometry as intexpr over spec axes, so no host-side grid function survives. Verified on GPU by `drivers/launch_from_bundle.py` |
| 5c  | Two-number version compatibility, wire and binary (§3.3d)                                      | **done** | `abi.h` + `src/abi.py`. `min_reader` derived from recipe content, checked by both engines; binary ABI checked once at load |
| 5d  | Decide the `.ll` → HSACO → launch boundary in C++ (§9.1)                                       | open | Python ctypes wrappers are the only executable spec today. Either wrap them for C++ or state that hipDNN owns those links — and fix or delete `jit_demo.cpp` either way |


**Kernel-side prerequisites (§4).**


| #   | Item                                                                                              | Depends on | Notes                                                                                  |
| --- | ------------------------------------------------------------------------------------------------- | ---------- | -------------------------------------------------------------------------------------- |
| 6   | Uniform `supports(spec, *, arch) -> (ok, reason)` per family, existing functions kept as wrappers | —          | Removes the per-family adapters the sweep writes by hand                               |
| 7   | Kernels declare tunable axes + candidate domains                                                  | 6          | Moves `CANDIDATES` out of the sweep driver so a new axis cannot be silently unexplored |
| 8   | Close known gate holes, starting with `num_kv_heads \| num_query_heads`; audit others             | —          | A generated catalog trusts these gates; today one of them is known wrong               |
| 9   | Hoist builder-deep assertions into the gate where possible                                        | 6          | Makes pruning cheap (no build probe) and the derived guard complete                    |
| 10  | Declare launch-time constraints on runtime args (`K % split_k`, `ks % tile_k`)                    | —          | Today a comment in `gemm_universal`; no recipe guard can see these                     |


**Broader path — only if §5's triggers are hit.**


| #   | Item                                                                          | Depends on | Notes                                                                                                                                 |
| --- | ----------------------------------------------------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| 11  | Differential harness: Python predicate vs C99 validator over a candidate grid | —          | Uses the existing `rocke_engine.*_is_valid` bindings. **Worth doing regardless**: nothing proves the 58 ported validators still agree |
| 12  | `rocke.valid/v1` schema + evaluator, language sugar, per-family rules         | 11         | §5.2–§5.5                                                                                                                             |
| 13  | Generate `cpp/core/arch/data.cpp` from `arch_specs.json`                      | —          | Removes a hand-transcription that exists independently of this work                                                                   |
| 14  | HSACO cache on `(family, baked, free, arch)`                                  | —          | Gap 6; largest remaining JIT-latency win, orthogonal to validity                                                                      |


Items 1, 6 and 11 are independent and can start in parallel. Item 11 tests
something that is shipping today and unverified, so it earns its place whatever
is decided about the rest.

With 1–3 and 5b–5c done, a caller with no CPython can open a bundle, be told
whether a shape is admissible, lower it, and learn what to launch. The remaining
narrow-path work is item 5 — the generator that applies this to a whole catalog
rather than one family at a time — and item 8, which is what decides whether the
guards are worth trusting at all. Item 8 is now the highest-value one on the
list: §3.3b shows the derivation reproducing a kernel's known gate hole
faithfully, because that is what a sound derivation does with a dishonest gate.

Item 5 also grew a second output. The generator now has to attach geometry as
well as a guard, which means the axes a family declares (item 7) are the axes
both are written over — one more reason those declarations belong on the kernel
rather than in a sweep driver.

---

## 11. Open questions for hipDNN

1. **Which parameters are non-free in your model?** The split determines the key
  and therefore how many recipes get generated. If tile/warp geometry is
   non-free (a catalog axis), see §8.2 — those are the axes that do not roll,
   so the recipe count is the geometry count.
2. **How many candidates per virtual catalog, typically?** This sets whether the
  bundle index needs to be a hash map or whether a sorted scan suffices, and
   whether the bundle should be memory-mapped.
3. **Is "no recipe shipped" acceptable as a routine outcome**, or should the
  generator guarantee coverage of whatever the heuristics can produce? The
   former is cheap; the latter turns catalog size into build-time cost directly.
4. **Do you need to tell "invalid" apart from "valid but not shipped"?** If
  skipping the candidate is enough in both cases, the narrow path (§3) is
   sufficient and §5 stays optional. If the heuristics need the distinction — to
   report coverage, or to avoid re-proposing instances that can never work —
   that is the main thing that would pull the broader path onto the critical
   path.
5. **Outside the verified envelope: refuse or permit-and-count?** §7 argues for
  refusing outside the legal domain always, and treating outside-verified as
   permitted with telemetry. This is a policy call with a real coverage cost
   either way.
6. **Do you need the rejection reason**, or only the boolean? Reasons cost
  nothing to carry in the recipe, but reason *parity* with Python is a stronger
   contract than verdict parity, and it is worth deciding whether to promise it.
7. **Where should the compile-and-load boundary be?** §9.1: rocke stops at `.ll`
  plus a launch plan, and you already link comgr and HIP. If you want to own
   those links, the Python wrappers should be treated as the reference for the
   flags and code-object handling. If you would rather call rocke for them, that
   is item 5d and worth knowing now rather than after you have written it.
8. **How do you want a too-new bundle surfaced?** Today it is `ROCKE_ERR_VALUE`
  from the guard API, distinct from a refusal, on the argument in §3.3d that a
   version mismatch is a deployment fault rather than a routing decision. If
   your dispatch treats every non-OK status as "skip this candidate", that
   distinction is lost and the failure looks like missing coverage.
