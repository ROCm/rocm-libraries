<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# AOT catalog families — kernel-author guide

This is the hands-on guide for **authoring AOT (ahead-of-time) kernel families** for the
HIP kernel provider. It is written for the rocKE kernel-authoring (KA) team so you can
bring your own gfx1151 (and beyond) kernels into hipDNN and do end-to-end testing
**without writing or rebuilding any C++** — you produce a compiled code object (`.co`) and
edit a JSON file.

This subsystem (`rocke/aot_catalog/`) **owns kernel authoring and the family build**. The
kernels it produces are loaded at runtime by a separate, throwaway consumer — the **AOT
catalog engine** (`src/engines/aot_catalog_engine/`), which only *loads* `.co` described by
`family.json` and has **zero build- or link-time dependency on rocke**. Authoring lives
here; loading lives there. For the runtime/loader contract, catalog resolution, the
capability/limits map, and the "my kernel isn't selected" debug trace, see the **[engine
README](../../src/engines/aot_catalog_engine/README.md)**.

The design is a deliberately thin, throwaway bring-up path. Its whole point is that a
kernel author owns kernels *as data*: an ahead-of-time-compiled code object (`.co` / HSACO)
plus a `family.json` describing how to **select** and **launch** it. The C++ side is a small
set of fixed, reviewed *adapters* (one per op kind) that map a hipDNN op graph to a launch
ABI. You extend coverage and tuning **data-only**; you touch C++ only when the *ABI itself*
changes or you teach the engine a genuinely new op.

Three op families ship today, each proven end-to-end on gfx1151 (AMD Radeon 8060S /
Strix Halo, RDNA3.5):

| op kind    | adapter          | shipped families                                   | §  |
|------------|------------------|----------------------------------------------------|----|
| `matmul`   | `GemmAdapter`    | `gemm_wmma_gfx1151`, `gemm_wmma_universal_gfx1151`  | [§6](#6-gemm--matmul) |
| `rmsnorm`  | `RmsNormAdapter` | `rmsnorm2d_gfx1151`                                 | [§7](#7-rmsnorm) |
| `sdpa`     | `SdpaAdapter`    | `fmha_wmma_fwd_gfx1151`                             | [§8](#8-sdpa-flash-attention-forward--the-universal-forward-adapter) |

A **family is one algorithm**, not one dtype: every family carries its f16 *and* bf16
kernels in a single flat `kernels[]` list, each kernel naming its own `dtype`
constraint (§3). All kernels in a family share the algorithm's tunable knobs — those
knobs are baked into each `.co` at produce time, and their consequences surface as each
kernel's per-kernel `constraints`.

**Core vocabulary** — a one-line gloss of the nouns used throughout; each is
expanded where it first matters.

| term | what it is |
|------|------------|
| **op kind** | the operation class an adapter handles: `matmul`, `rmsnorm`, or `sdpa`. |
| **adapter** | the fixed, reviewed C++ for one op kind; maps a hipDNN op graph to a launch (decode → problem → bindings → grid). One per op kind. Lives in the engine (`ops/<Op>Adapter.*`). |
| **family** | one algorithm for one op kind on one arch, in `aot_catalog/<arch>/<family>/`; carries all its dtypes' kernels together. |
| **kernel** | one compiled variant: a `.co` plus a `kernels[]` entry (its `symbol`, ABI, grid, and `constraints`). |
| **`.co`** | the ahead-of-time-compiled code object (HSACO) a kernel launches — a build product, never in git. |
| **`family.json`** | a family's spec: its `kernels[]` list plus identity (name, op kind, arch). |
| **producer** | the co-located `produce_<family>_co.py` that compiles the family's `.co`(s) at build time (uses rocke as a library). |
| **problem** (shape / key) | what an adapter's `decode` extracts from a graph — dtype, dims, capability facts; the keys that are legal in `constraints` and `grid`. |
| **constraint** | a per-kernel rule (`equals` / `multiple_of` / …) on a problem key; **fail-closed** — all must hold or the kernel is skipped. |
| **candidate** | a kernel whose constraints all hold for the current problem; when several apply, they are timed against each other. |
| **catalog** | the on-disk tree of families the engine loads at runtime (`<arch>/<family>/`). |
| **tune cache** | the remembered fastest candidate per problem, so later executes skip re-measuring. |
| **decline** | when no kernel applies, the engine returns "not applicable" and another hipDNN engine serves the graph — never a wrong answer. |

> **Design principle — a family folder is a self-contained unit.** Each
> `aot_catalog/<arch>/<family>/` folder owns *everything* about one family and only
> that family — its `family.json`, its `CMakeLists.txt`, its co-located producer,
> and its tests — so adding a family is dropping in a folder (auto-discovered) and
> deleting one is `rm -rf`, with nothing left dangling anywhere else. The testing
> contract that makes this isolation hold (a family's tests depend only on that
> family) is spelled out in [§9](#9-how-to-test).

**New here? Jump to [§2 Quick start](#2-quick-start-the-self-serve-recipe)** — the 30-second
recipe plus a top-to-bottom KA checklist. §3–§5 are the authoring mechanics that apply to
**every** op; §6–§8 are the per-op ABI specifics; [§9](#9-how-to-test)–[§11](#11-file-map)
cover testing, adding a brand-new op, and the file map.
[§12](#12-authoring-a-forward-sdpa-family-on-a-new-arch-gfx942gfx950) is the step-by-step
handoff kit for authoring a **forward SDPA family on a new arch** as data.

---

## 1. Where a family plugs in at runtime

```
torch op  (F.linear / F.rms_norm / F.scaled_dot_product_attention / …)
        │  (optional) a torch-functional monkeypatch that routes to AOT (§9)
        ▼
hipDNN frontend  ──►  single-node op graph  (Matmul / RmsNorm / Sdpa attributes)
        ▼
CatalogEngine::matchGraph
        ├─ <Op>Adapter::decode(graph)      → ProblemShape {dtype, dims…}
        │     (fails closed on anything the kernel can't serve → declines the graph)
        ├─ Catalog::candidatesFor(op_kind, problem)
        │     → every family.json kernel whose constraints all hold for this problem
        ▼
CatalogPlan  (first execute: measure each candidate, cache the fastest by problem key)
        ├─ <Op>Adapter::buildBindings → LaunchBindings (pointer UIDs + baked scalars)
        ├─ <Op>Adapter::gridSymbols   → SymbolTable the grid DSL evaluates
        ▼
LaunchAbi  packs the arg list by name, evaluates grid + block, sharedMemBytes, workspace
        ▼
hipModuleLaunchKernel  →  your .co
```

Adding a kernel = adding a `kernels[]` entry (and its `.co`) that `candidatesFor` can
select. Everything downstream is already wired. If `decode` declines, another hipDNN
engine serves the graph and **the model never miscomputes** — a missing/inapplicable
kernel is a fallback, never a wrong answer. (The full runtime contract — catalog
resolution, loader fail-closed behavior, the decline stages — is in the
[engine README](../../src/engines/aot_catalog_engine/README.md).)

---

## 2. Quick start (the self-serve recipe)

**The 30-second version** — bring a kernel into hipDNN as data:

1. Compile your kernel to a code object for the target arch with the **exact ABI** for
   its op kind (§6–§8). For a rocke-AOT family the family's co-located producer script
   does this at build time (§9); for a prebuilt family you check the `.co` into the
   folder yourself.
2. Add a `kernels[]` entry to the family's `family.json` (schema in §3) pointing at the
   `.co` by name, with the `dtype` (and other) constraints your kernel requires and the
   grid it launches with. One folder per algorithm — `aot_catalog/<arch>/<family>/`, e.g.
   `aot_catalog/gfx1151/gemm_wmma/` — holds every dtype's kernels flat.
3. Build the provider (rocke-AOT families run their producer → emit `.co` + stage
   `family.json` into the build tree; see §9), **or** for a quick data-only iteration
   point `HIPDNN_AOT_CATALOG_DIR` at a populated `<arch>/<family>/` tree directly.
4. Verify with the substrate parity test (§9); real-model E2E additionally needs the
   hipDNN→PyTorch injection layer (a separate PR; §9).

To add a **tuning candidate** for a shape you already serve, you only do step 1+3:
append another `kernels[]` entry with overlapping constraints. `CatalogPlan` measures
all applicable candidates on the first execute and caches the fastest per problem (§5).

### KA checklist (the thorough version)

Work top to bottom; each rung says *why* and where it is spelled out.

- [ ] **Confirm an adapter already handles your op** — `matmul` / `rmsnorm` / `sdpa`
      (§6–§8). If not, you have a **genuinely new op**: that needs a small reviewed C++
      adapter first (§10) and is **not** data-only.
- [ ] **Compile the `.co` for the exact target arch, with the op's exact ABI** — arg
      names, types, and order (§6–§8). A rocke-AOT family's co-located producer does this
      at build time (§9).
- [ ] **Write / extend `family.json`** (§3): one `kernels[]` entry per `.co`, pointing at
      it by `co_file`, with `grid` / `block` / `args_signature` and a constraint for every
      problem key your kernel handles.
- [ ] **Constrain fail-closed** (⚠️ the one way this engine returns a *wrong* answer — see
      the warning in §3). For each key you left *unconstrained*, ask: *will my kernel be
      correct for every value this key can take?* If not, add the constraint. When in
      doubt, over-constrain — a too-narrow kernel simply isn't picked (safe fallback, §1);
      a too-broad one miscomputes.
- [ ] **Build the provider** (producer emits `.co`, stages `family.json`; §9) — or point
      `HIPDNN_AOT_CATALOG_DIR` at a populated tree for a data-only iteration.
- [ ] **Verify correctness** — the substrate parity test is the required, hermetic gate
      (§9); real-model E2E is the fuller check but needs the hipDNN→PyTorch injection
      layer (a separate PR; §9).
- [ ] **If your kernel isn't selected**, set `HIPDNN_AOT_DEBUG=1` and read the resolution
      / load / decline trace (see the engine README's debug section).
- [ ] **Know when you have left data-only and need reviewed C++**: a changed ABI, a new
      capability to *decode*, or a grid the DSL can't express (§4); or a brand-new op
      (§10). These require an adapter edit + rebuild + review — data alone won't do it.

---

## 3. `family.json` schema (shared by all ops)

One file per family directory. The top-level fields identify the family; `kernels[]`
holds one entry per compiled variant.

```json
{
    "family": "gemm_wmma_gfx1151",       // unique family name (algorithm, not dtype)
    "op_kind": "matmul",                  // "matmul" | "rmsnorm" | "sdpa"
    "arch":    "gfx1151",
    "dtype":   ["f16", "bf16"],           // dtypes this family covers (documentation;
                                          //   the per-kernel dtype constraint is what
                                          //   actually gates selection)
    "kernels": [ { …f16 variant… }, { …bf16 variant… }, … ]
}
```

The `kernels[]` list is **flat and dtype-mixed**: one family folder holds every dtype's
kernels (f16 + bf16 today; fp8/fp32 the same way), each entry carrying its own
`{"dtype": {"equals": …}}` constraint. The per-dtype sets may be **disjoint** (e.g.
`rmsnorm2d`'s f16 vs bf16 specializations). There is no `sections` grouping and no
`{dtype}` token — selection is per-kernel via the `dtype` constraint.

Each `kernels[]` entry:

| field             | meaning |
|-------------------|---------|
| `symbol`          | the kernel's exported symbol *inside* the `.co` (for `hipModuleGetFunction`); also the tune-cache key. **Unique per candidate.** |
| `co_file`         | `.co` filename, relative to the family dir. |
| `constraints`     | map `problem_key → rule`. **Required and non-empty** — a kernel that constrains nothing asserts it handles *every* problem shape (see the ⚠️ note below), so an omitted or empty map is a load-time error, not a wildcard. Rules (integer keys unless noted): `{"equals": v}` (int/string/bool), `{"not_equals": v}`, `{"one_of": [..]}`, `{"min": n}` / `{"max": n}` (inclusive bounds), and `{"multiple_of": n}`. **Fail-closed:** every constrained key must be present in the decoded problem *and* every rule must hold, or the candidate is skipped. This is how the `dtype` constraint selects the f16 vs bf16 kernels within one family. **Zero-extent guard:** `multiple_of` alone admits `0` (since `0 % n == 0`), which would select a kernel for a degenerate `M/N/K==0` (or `seqlen==0`) problem and launch a zero/garbage grid — so every `multiple_of` dim also carries `"min": 1` to reject non-positive extents. |
| `grid`            | per-axis `x`/`y`/`z`. A value is a constant, a problem-key string (`"M"`, `"H"`, `"B"`), or `{"ceil_div": ["<key>", n]}`. Evaluated from that op's `gridSymbols`. |
| `block`           | `[x,y,z]` constant workgroup size. |
| `shared_mem_bytes`| omit for static-LDS kernels (defaults to 0). |
| `workspace_bytes` | this kernel's scratch need — either a **constant** (an integer, e.g. `0` as on every shipped kernel) **or** a **data-driven expression** over the problem's `grid` keys plus `elem_size`, evaluated per-problem. A kernel that needs scratch sets this **and** names a `ptr` arg called `workspace` in its `args_signature`; the engine reserves `max(workspace_bytes)` over the applicable candidates, allocates one buffer, and binds it to that arg. The expression is a small JSON-AST (see [§4](#workspace-a-constant-or-a-data-driven-expression)); a bare integer is just the degenerate case. This makes shape-scaled scratch (conv im2col, split-K) authorable — but note conv still needs an adapter and a selection strategy ([§10](#10-adding-a-brand-new-op)). |
| `args_signature`  | the launch ABI as an ordered list of `{name,type}` (`ptr`/`i32`/`f32`). **Must match the op's ABI order exactly** (§6–§8); `LaunchAbi` packs by name in this order. |

The set of legal `constraints`/`grid` keys is exactly the **problem keys the adapter
emits** for that op — listed per op in §6–§8.

> ⚠️ **Unknown fields are rejected, not ignored.** The loader fails the file on any
> unrecognized key in a family, kernel, constraint rule, or `args_signature` entry. This
> closes a *fail-open* footgun: a misspelled **field name** (`"constraint"` for
> `"constraints"`, `"mutiple_of"` for `"multiple_of"`) would otherwise be silently dropped,
> quietly discarding the predicate it was meant to carry — the opposite failure mode from a
> typo *inside* a constraints map, which fails closed. To annotate a file, use the
> **`_`-prefixed comment convention** (any key starting with `_`, e.g. `_comment`) — those
> are the only non-schema keys accepted. Note also that a family's `arch` field (if present)
> must match the arch directory it lives under (`aot_catalog/<arch>/<family>/`); a mismatch
> is a load error, since a file copied into the wrong arch folder would load kernels built
> for the wrong GPU.

> ⚠️ **Under-constraining is the one way this engine returns a wrong answer.** Constraints
> are the *only* thing standing between a kernel and a problem it cannot actually serve. A
> key you leave unconstrained is an implicit claim that your kernel is correct for *every*
> value that key can take. Forget the `dtype` constraint and a bf16 problem can select your
> f16 kernel; forget a static kernel's `N` and it matches *every* `N` and computes garbage;
> for SDPA, omit a capability key (`causal`, `gqa_ratio`, a mask fact — §8) and a graph with
> that feature selects a kernel that silently ignores it. The adapters guarantee only
> **memory safety** (rank / dtype / shape agreement); **correctness of *applicability* is
> entirely your `family.json`.** When in doubt, over-constrain: a too-narrow kernel just
> isn't picked (a safe fallback that another engine serves, §1); a too-broad one
> miscomputes. Every shipped family constrains its keys explicitly — copy that discipline.

---

## 4. When you need a C++ change (vs data-only)

The self-serve path (data only) covers:

- **new kernels** for a shape the adapter already decodes;
- **new dtypes** — add more `kernels[]` entries to the same family, each with its own
  `dtype` constraint (proven: bf16 GEMM and bf16 RMSNorm were each added with *zero*
  C++ change);
- **tuning candidates** — more `kernels[]` entries with overlapping constraints (§5).

You must edit the adapter (`src/engines/aot_catalog_engine/ops/<Op>Adapter.{hpp,cpp}`) and
re-review/rebuild only when the **contract** changes:

- a **different ABI** (arg added/removed/reordered, or a different scalar convention) —
  `buildBindings` and the `args_signature` must agree;
- a **new capability to decode** (e.g. SDPA causal masking, GQA `H_kv != H`, a runtime
  scale tensor) — `decode` must stop declining it and emit any new problem keys;
- a **grid/launch shape** the grid DSL can't express.

For a genuinely new op, see §10. For the full capability/limits map (which SDPA and conv
extensions on gfx942/gfx950/gfx1250 are data, which are adapter C++, and which need a new
substrate capability), see the **capabilities-and-limits section of the
[engine README](../../src/engines/aot_catalog_engine/README.md)**.

### Workspace: a constant, or a data-driven expression

`workspace_bytes` (§3) sizes the scratch buffer the engine allocates and binds to a kernel's
`workspace` `ptr` arg. The framework must size it **before** tuning knows which candidate
wins, so at plan time the engine evaluates each applicable candidate's expression for this
problem and reserves the **max** across them (`plans/CatalogPlan.cpp`), allocates that one
buffer, and binds it. It accepts two forms:

- a **constant** — a bare integer (`0` on every shipped kernel), the value you know at
  authoring time; or
- a **JSON-AST expression** over the kernel's `grid` symbols (`M`, `N`, `K`, … — whatever the
  adapter publishes) plus the injected `elem_size` symbol (element width in bytes for the
  problem's dtype), evaluated per-problem. A constant is just a LITERAL node, so nothing about
  the old behavior changed.

The node vocabulary (v1) is arithmetic + clamp only:

| node | form | meaning |
|------|------|---------|
| literal | `256` | integer constant |
| symbol | `"M"` | a grid symbol, or `"elem_size"` |
| `mul` / `add` / `min` / `max` | `{"mul": [a, b, …]}` | variadic (≥ 1 operand) |
| `sub` | `{"sub": [a, b]}` | `a - b` (must be ≥ 0) |
| `ceil_div` / `floor_div` | `{"ceil_div": [a, b]}` | integer division |
| `align_up` | `{"align_up": [a, b]}` | round `a` up to a multiple of `b` |

Example — an im2col buffer of `align_up(M · N · elem_size, 256)`:

```json
"workspace_bytes": { "align_up": [ { "mul": ["M", "N", "elem_size"] }, 256 ] }
```

Evaluation is **fail-closed**: a symbol the problem doesn't publish, a divide/align by zero,
or a negative `sub` throws and the engine declines rather than launching with a wrong size.
Malformed expressions (unknown op key, wrong arity) are rejected at **load** time, skipping
just that family (§3 unknown-key discipline).

**Deliberately out of v1:** conditionals (`?:`/`if`) and infix syntax. An algorithm branch
whose scratch differs (split-K vs none, im2col vs Winograd) is modeled as **separate kernels**
with their own `constraints` + expression — the selector already branches, so no ternary is
needed. Overflow of the int64 product and exotic dtypes (`elem_size` is absent, so referencing
it fails closed) are known gaps.

**This removes one conv blocker, not all of them.** Shape-scaled scratch is now expressible,
but convolution still needs a `ConvAdapter` (to decode and publish `N,C,H,W,…`) **and** a
selection strategy beyond measure-and-cache for its unbounded shape space (engine README §4.1,
[§10](#10-adding-a-brand-new-op)). SDPA backward likewise needs a multi-kernel plan, not just a
workspace expression (engine README §4.3). Data-driven workspace ≠ conv works.

---

## 5. Measure-and-cache tuning

Multiple `kernels[]` entries whose constraints all hold for the same problem are all
**applicable candidates**. On the first execute for a given problem key, `CatalogPlan`
times each candidate (1 warmup + median of several `hipEvent`-timed launches, skipping
any that error) and caches the fastest, keyed on `family + canonicalized problem`.
Subsequent executes of that shape reuse the winner from the tune cache.

- Cache location: env `HIPDNN_AOT_TUNE_CACHE`, else a temp file. Delete it to re-measure.
- A single applicable candidate → launched directly (nothing to measure; the cache
  read-back shows `[None]`, which is expected and not an error).
- There is currently **no silent cap** on candidates — every applicable one is
  measured, so keep the per-problem candidate set small (a handful) to keep
  first-execute tuning cheap.

The winner is launched *last* during tuning, and every candidate produces the same
correct output, so timing on the real output buffer is safe.

---

## 6. GEMM / matmul

**op_kind `"matmul"` · adapter `GemmAdapter` · families `gemm_wmma_gfx1151`
(reference) + `gemm_wmma_universal_gfx1151` (tiled), each carrying its f16 + bf16
kernels.**

### The kernels
- **Reference** (`gemm_wmma_*`): rocke `wmma_gemm`, one wave32 per 16×16 output tile,
  no LDS staging — correctness-first, launch-overhead-cheap (wins on tiny shapes).
- **Tiled** (`gemm_wmma_universal_*`): rocke `build_universal_gemm`, LDS-staged,
  register-blocked (tile 64×64×32, warp 2×2, wt 16×16×16). 3–7× faster than the
  reference at large shapes; the tune cache picks per shape.

### Layout — RCR only (`y = x @ Wᵀ`, i.e. `nn.Linear`)
hipDNN `MatmulAttributes` carries only a/b/c UIDs and **no transpose flag** — the
transpose is expressed by **strides**. `GemmAdapter::decode` reads logical dims and
**gates on RCR strides**: A `[M,K]` row-major, B logical `[K,N]` with strides `{1,K}`
(physical `[N,K]` weight), C `[M,N]` row-major. Anything else declines. This is exactly
`nn.Linear`'s `x @ weightᵀ`. The kernel has **no epilogue** — bias/activation are the
caller's job (a model override adds bias natively post-matmul).

### ABI (6 args, exact order)

| # | name | type | meaning |
|---|------|------|---------|
| 0 | `A` | ptr | activations `[M,K]` row-major |
| 1 | `B` | ptr | weight, physical `[N,K]` (logical `[K,N]` RCR) |
| 2 | `C` | ptr | output `[M,N]` row-major |
| 3 | `M` | i32 | rows |
| 4 | `N` | i32 | output cols (weight rows) |
| 5 | `K` | i32 | inner / reduction dim |

### ⚠️ Grid-order gotcha — reference vs tiled are INVERTED
- Reference: `grid.x = ceil_div(M,16)`, `grid.y = ceil_div(N,16)`.
- Tiled universal: **`grid.x = ceil_div(N,64)`, `grid.y = ceil_div(M,64)`** (NM order).

Copy the grid block from the matching family; do not assume M-then-N.

### Problem keys (`decode` emits) — legal in `constraints`/`grid`
| key | type | source |
|-----|------|--------|
| `dtype` | string | `"f16"` / `"bf16"` |
| `M` | int | rows of A / C |
| `N` | int | cols of C |
| `K` | int | inner dim |

Constraints are `multiple_of` (16 for the reference; M/N `multiple_of 64`, K
`multiple_of 32` for the tiled path — sub-tile shapes correctly fall back).

### Example — a complete `kernels[]` entry (the shipped reference f16 kernel)
This is the simplest end-to-end shape of "a kernel instance with conditions" — copy it as
your starting point. (The bf16 kernel is the same entry with `"dtype": {"equals": "bf16"}`
and its own `.co`; both live flat in the one `gemm_wmma/family.json`.)

```json
{
    "symbol": "rocke_wmma_gemm_wmma16x16x16_fp16_rcr_xm",
    "co_file": "rocke_wmma_gemm_wmma16x16x16_fp16_rcr_xm.co",
    "constraints": {
        "dtype": { "equals": "f16" },
        "M": { "min": 1, "multiple_of": 16 },
        "N": { "min": 1, "multiple_of": 16 },
        "K": { "min": 1, "multiple_of": 16 }
    },
    "grid":  { "x": { "ceil_div": ["M", 16] }, "y": { "ceil_div": ["N", 16] }, "z": 1 },
    "block": [32, 1, 1],
    "args_signature": [
        { "name": "A", "type": "ptr" }, { "name": "B", "type": "ptr" },
        { "name": "C", "type": "ptr" },
        { "name": "M", "type": "i32" }, { "name": "N", "type": "i32" },
        { "name": "K", "type": "i32" }
    ],
    "workspace_bytes": 0
}
```

- **Add a dtype** → append a second entry with a different `dtype` constraint and its `.co`
  (zero C++, §4).
- **Add a tuning candidate** for shapes you already serve (e.g. a faster tiled kernel) →
  append an entry with *overlapping* constraints and a distinct `symbol`; `CatalogPlan`
  measures both and caches the winner per shape (§5). This is exactly how the tiled
  `gemm_wmma_universal` family coexists with this reference one.
- **Narrow a kernel** to only the shapes it's correct for → tighten its constraints
  (`{"equals": …}`, tighter `multiple_of`, `min`/`max` bounds). When unsure, over-constrain
  (§3 ⚠️).

### Files
Adapter `src/engines/aot_catalog_engine/ops/GemmAdapter.{hpp,cpp}`; data + co-located
producers + per-family parity tests
`aot_catalog/gfx1151/gemm_wmma/{family.json, produce_gemm_wmma_co.py, TestGemmWmmaNumericParity.cpp}`,
`aot_catalog/gfx1151/gemm_wmma_universal/{family.json, produce_gemm_universal_co.py, TestGemmUniversalNumericParity.cpp}`.
Real-model E2E additionally needs the hipDNN→PyTorch injection layer (separate PR; §9).

---

## 7. RMSNorm

**op_kind `"rmsnorm"` · adapter `RmsNormAdapter` · family `rmsnorm2d_gfx1151` (f16 +
bf16 kernels flat).**

### The kernels
rocke CK-Tile `10_rmsnorm2d`: per-row RMS over the last dim of a 2D `[M,N]` tensor with
a per-column weight `Gamma[N]` (Llama/Mistral RMSNorm). Two body shapes exist —
single-pass VGPR-cached vs two-pass streaming — selected by `elems_per_thread =
N/block_size`; both are perf-only (identical correct output), so they're tuning
candidates (§5). Higher-dimensional inputs are flattened to `[M,N]` by the override.

- **Static variants** bake `N` → constraint `{"N": {"equals": <n>}}` (exact-match
  shape tiers, e.g. N=2048/4096).
- **Runtime-N variants** (symbol suffix `_dyn_`, rocke `rmsnorm2d_dynamic.py`) read `N`
  as the runtime i32 arg → constraint `{"N": {"multiple_of": <vec>}}`, matching any
  vec-aligned N (e.g. Flux 3072, SD3.5 2432). Two binaries cover every real ComfyUI
  hidden size (all multiples of 8).

### ⚠️ Gotcha — `wave_size = 32` at compile time
The producer **must** set `wave_size=32`. The default 64 miscompiles the wave32
cross-lane reduction on gfx1151 → silent wrong results. (This is the single most
common way to get a plausible-but-wrong RMSNorm kernel.)

### ⚠️ Gotcha — epsilon is a baked scalar *tensor*
In hipDNN, `epsilon` arrives as a scalar **tensor** operand (not a node attribute). The
adapter bakes it at plan-build via `makeScalarOperand`/`toDouble` and packs it as the
f32 ABI arg. It therefore **fails closed on a pure runtime user-supplied epsilon** —
the value must be knowable at plan build (it always is in practice).

### ABI (6 args, exact order)

| # | name | type | meaning |
|---|------|------|---------|
| 0 | `X` | ptr | input `[M,N]` |
| 1 | `Gamma` | ptr | per-column weight `[N]` |
| 2 | `Y` | ptr | output `[M,N]` |
| 3 | `M` | i32 | rows |
| 4 | `N` | i32 | normalized dim |
| 5 | `eps` | f32 | epsilon (baked) |

Grid `(M,1,1)`; block `[256,1,1]` (or the variant's block). `Gamma` maps from the
graph's `scale_tensor_uid`. Weightless norms (LTX's `common_dit.rms_norm(x)`) are
served by the override synthesizing a cached ones-weight.

### Problem keys (`decode` emits)
| key | type | source |
|-----|------|--------|
| `dtype` | string | `"f16"` / `"bf16"` |
| `M` | int | rows |
| `N` | int | normalized dim |

### Files
Adapter `src/engines/aot_catalog_engine/ops/RmsNormAdapter.{hpp,cpp}`; data + co-located
producer + per-family tests
`aot_catalog/gfx1151/rmsnorm2d/{family.json, produce_rmsnorm2d_co.py, TestRmsNormNumericParity.cpp, TestRmsNormSelection.cpp}`;
rocke runtime-N instance `rocke/library/.../rmsnorm2d_dynamic.py`.
Real-model E2E additionally needs the hipDNN→PyTorch injection layer (separate PR; §9).

---

## 8. SDPA (flash-attention forward) — the universal forward adapter

**op_kind `"sdpa"` · adapter `SdpaAdapter` (universal forward) · family
`fmha_wmma_fwd_gfx1151` (f16 + bf16 kernels flat).**

`SdpaAdapter` is **one universal *forward* adapter**, not a gfx1151-shaped one. It decodes
a single-node `SdpaAttributes` graph arch-neutrally, publishes the full **capability
vocabulary** as problem-shape facts, and lets each kernel's `family.json` constraints
decide applicability. It marshals a **superset** of by-name arguments, so a forward
kernel's ABI on *any* arch is selected as data (its `args_signature` picks the subset it
takes). The gfx1151 WMMA kernel below is the first family it serves; a gfx942/gfx950
forward family is authored as data against the same adapter — see the handoff kit in
[§12](#12-authoring-a-forward-sdpa-family-on-a-new-arch-gfx942gfx950).

Only **universal, memory-safety invariants** remain hard C++ declines (single
`SdpaAttributes` node; rank-4 BHSD Q/K/V/O; K/V agree on `H_kv`/`S_kv`/`D`; O mirrors Q;
one supported dtype across Q/K/V/O; integer `gqa_ratio`). Every **feature** decision
(causal, GQA, masks, fp8, …) moved from an `if` in the adapter to a *where* in data.

### The kernel (the shipped reference)
rocke's `build_wmma_fmha_fwd` (gfx1151 WMMA flash-attention forward), a thin adapter
over the unified `mfma_attention_fwd_inner_body`:

- **`head_size` (D) and head counts (H, H_kv) are compile-time; seqlen is runtime.** So
  **one binary per dtype** serves both LTX self-attn (S_q = S_kv = 4096) and cross-attn
  (S_q = 4096, S_kv = 128). D and H are *exact-match* constraints; S_q/S_kv only need to
  be tile multiples.
- **Native bf16, no cast.** bf16 shares the f16 16×16×16 WMMA fragment layout on
  gfx1151, so the same inner body lowers to `wmma.f32.16x16x16.bf16` for bf16. f16 and
  bf16 are separate `.co`s within the one family, selected by the `dtype` constraint.
- **`mask_mode="none"`, non-causal, MHA (H_kv == H), contiguous-D, batch-foldable.** These
  are now expressed as **capability constraints in `family.json`** (`causal {equals
  false}`, `gqa_ratio {equals 1}`, `d_contiguous {equals true}`, `batch_foldable {equals
  true}`, and all the `has_* {equals false}`), not as adapter declines. A graph the kernel
  can't serve fails the constraints → no candidate → the engine declines (aggregate
  fail-closed), and another engine serves it.
- **Grid** `(ceil_div(S_q,16), H, B)`, **block** `(32,1,1)` — one wave32 per CTA, each
  CTA owns a 16-row Q tile of one (head, batch). **Static LDS** → `sharedMemBytes = 0`;
  **no workspace**.

This is a correctness-first reference (single wave per tile, no LDS K/V staging). It is
numerically correct at LTX shapes; on gfx1151 it also *beats* stock PyTorch SDPA, but
only because PyTorch has no fused flash backend there and falls to an unfused O(S²) math
path — so treat that as "not the bottleneck," not a win over a tuned flash kernel.
Tuning (LDS staging, multi-tile, larger Q tiles) grows data-only via §5.

### This kernel's ABI (15 args, exact order — a subset of the adapter vocabulary)

| # | name | type | meaning |
|---|------|------|---------|
| 0 | `Q` | ptr | query `[B,H,S_q,D]` |
| 1 | `K` | ptr | key `[B,H,S_kv,D]` |
| 2 | `V` | ptr | value `[B,H,S_kv,D]` |
| 3 | `O` | ptr | output `[B,H,S_q,D]` |
| 4 | `scale_log2` | f32 | **`attn_scale * log2(e)`** — see gotcha #1 |
| 5 | `seqlen_q` | i32 | S_q |
| 6 | `seqlen_k` | i32 | S_kv |
| 7 | `stride_q_token` | i32 | `q.stride(2)` (S axis) |
| 8 | `stride_q_head` | i32 | `q.stride(1)` (H axis) |
| 9 | `stride_k_token` | i32 | `k.stride(2)` |
|10 | `stride_k_head` | i32 | `k.stride(1)` |
|11 | `stride_v_token` | i32 | `v.stride(2)` |
|12 | `stride_v_head` | i32 | `v.stride(1)` |
|13 | `stride_o_token` | i32 | `o.stride(2)` |
|14 | `stride_o_head` | i32 | `o.stride(1)` |

These 15 names are the subset this kernel's `args_signature` selects from the adapter's
**full argument vocabulary** — the complete emitted set (byte-stride variants, batch
strides, raw scale, head/dim scalars, optional mask/stats pointers) is the arg-vocabulary
reference table in [§12](#arg-vocabulary-reference). A different forward kernel picks a
different subset; a name it needs that isn't emitted yet is one added line in
`buildBindings` (reviewed C++, §12).

#### ⚠️ Gotcha #1 — `scale_log2`, not the raw scale
The softmax is computed **base-2** (`exp2`), so the kernel takes
`scale_log2 = attn_scale * log2(e)` where `log2(e) = 1.4426950408889634`. The adapter
emits **both** `scale_log2` (already multiplied, from `attn_scale_value`, defaulting to
`1/sqrt(D)`) **and** `scale_raw` (unmultiplied); a kernel that wants the raw scale simply
names `scale_raw` in its `args_signature` — no C++ change.

#### ⚠️ Gotcha #2 — BHSD stride mapping; batch-stride is available but this kernel folds
Tensors are `[B,H,S,D]`: token stride is `stride(2)` (S axis), head stride is
`stride(1)` (H axis). This kernel takes **no batch-stride argument** — it folds batch into
grid `z` assuming `batch_stride == seqlen * stride_token` (trivially true for `B == 1`,
LTX). It therefore constrains `batch_foldable {equals true}`. The adapter *does* emit
`stride_{q,k,v,o}_batch` (element and byte), so a kernel with a real batch-stride arg on
another arch accepts the non-folded `B > 1` case as data.

### Problem keys (`decode` publishes — the capability vocabulary)
Numeric shape:

| key | type | source |
|-----|------|--------|
| `dtype` | string | `"f16"`/`"bf16"`/`"f8"`/`"bf8"`/`"f8fnuz"`/`"bf8fnuz"` |
| `B` | int | `q.dim(0)` |
| `H` | int | `q.dim(1)` (query heads) |
| `H_kv` | int | `k.dim(1)` (kv heads) |
| `S_q` | int | `q.dim(2)` |
| `S_kv` | int | `k.dim(2)` |
| `D` | int | `q.dim(3)` (head dim) |
| `gqa_ratio` | int | `H / H_kv` (1 = MHA, H = MQA) |

Capability facts (bool unless noted) — a kernel opts in/out via a constraint; see the
[capability-key reference](#capability-key-reference) in §12 for the fail-closed warning:

| key | true when |
|-----|-----------|
| `d_contiguous` | innermost (D) axis is unit-stride on Q/K/V/O |
| `batch_foldable` | `B == 1`, or batch stride packs as `seqlen * stride_token` on all operands |
| `causal` | mask resolves to top-left causal (deprecated `causal_mask`, or the `left_bound`/`right_bound`/`diagonal_alignment` trio) |
| `causal_bottom_right` | mask resolves to bottom-right causal (deprecated `causal_mask_bottom_right`, or the trio with `diagonal_alignment = BOTTOM_RIGHT`) |
| `has_diagonal_band` | mask resolves to a sliding window (a bounded `left_bound`/`right_bound` that is not plain causal) |
| `has_mma_core_mode` | `mma_core_mode` is set to a non-default (non-`UNSET`) compute dtype |
| `has_alibi` | `alibi_mask` set |
| `has_padding_mask` | `padding_mask` set |
| `has_attn_mask` | an `attn_mask` tensor is present |
| `has_block_mask` | a `block_mask` tensor is present |
| `has_sink` | a `sink_token` tensor is present |
| `has_dropout` | dropout prob ≠ 0, or any dropout-plumbing tensor |
| `paged` | a page-table (K or V) tensor is present |
| `varlen` | a `seq_len_q`/`seq_len_kv` (group-mode) tensor is present |
| `gen_stats` | `generate_stats` set, or a stats tensor is present |
| `fp8` | an fp8 element dtype, or any fp8 (de)scale tensor |
| `runtime_scale` | a runtime `scale` tensor is present (vs. baked `attn_scale_value`) |

### Decline boundary (now: universal safety only)
`SdpaAdapter::decode` returns "not applicable" (→ another engine serves it) **only** for
the memory-safety invariants above: not a single `SdpaAttributes` node; any of Q/K/V/O
not rank-4 (dims or strides); K/V disagreeing on `H_kv`/`S_kv`/`D`; O not mirroring
`[B,H,S_q,D]`; an unsupported or mismatched dtype across Q/K/V/O; a non-integer
`gqa_ratio` (`H_kv` that doesn't divide `H`). **Everything else decodes** — causal, GQA,
masks, fp8, varlen, paged — and applicability is decided by each kernel's `family.json`
constraints. Serving a new feature is therefore, in the common case, **new data** (a
family that constrains the fact appropriately), not adapter C++; it is only C++ if the
kernel needs an argument the vocabulary doesn't yet emit or a grid the DSL can't express
(§12).

### Files
Adapter `src/engines/aot_catalog_engine/ops/SdpaAdapter.{hpp,cpp}`; data + co-located
producer + per-family parity test
`aot_catalog/gfx1151/fmha_wmma_fwd/{family.json, produce_fmha_fwd_co.py, TestSdpaNumericParity.cpp}`;
engine-level decode/bindings unit test
`src/tests/engines/aot_catalog_engine/TestSdpaDecode.cpp` (host-only, no GPU, arch-neutral
— not tied to this family). Real-model E2E additionally needs the hipDNN→PyTorch injection
layer (separate PR; §9). Authoring a new-arch forward family:
[§12](#12-authoring-a-forward-sdpa-family-on-a-new-arch-gfx942gfx950).

---

## 9. How to test

Verification has three rungs. Rungs 1–2 live here and are what you run to land a family;
rung 3 (real-model E2E) is provided by the separate hipDNN→PyTorch injection layer.

### The build: how a family's `.co` gets produced

Two independent axes control the build; do not conflate them:

- **`ENABLE_AOT_CATALOG_ENGINE`** (default **ON**) — compiles the throwaway engine into
  `libhip_kernel_provider.so`. This is the *consumer* of the catalog. It ships in the normal
  build because it is **inert without a catalog** (loads nothing, declines every graph), so a
  catalog produced later is picked up with no provider rebuild.
- **`HIPKERNELPROVIDER_ENABLE_ROCKE`** (default **OFF**) — `add_subdirectory(rocke)`,
  which builds the rocke platform, the build-time Python env (`rocke-pyenv`), and — when
  `ENABLE_AOT_CATALOG_ENGINE` is *also* ON — **this `aot_catalog/` subsystem**. This is the
  real opt-in: it is the flag that actually *produces* kernels.

So **kernels are produced only when BOTH options are ON.** In the default build the engine is
present but the catalog is **empty**, so it stays inert and the GPU parity tests skip (see
§9.2). This subsystem is never configured unless rocke is on, so defaulting the engine ON
never drags the normal build into depending on rocke.

Each family's co-located `produce_<family>_co.py` emits **every** kernel (all dtypes) the
`family.json` lists, using rocke as a *library*. The build runs it automatically through
the **rocke build interpreter** (`${ROCKE_PYENV_PYTHON}`, the editable venv that puts the
`rocke` package on `sys.path` with no `PYTHONPATH` surgery); the `.co` are **build
products, never checked into git** (see `.gitignore`). The per-family `CMakeLists.txt`
calls `rocke_add_aot_family()`, which compiles them into the build/install tree at
`${HIPDNN_AOT_CATALOG_BUILD_DIR}/<arch>/<family>/` and stages `family.json` beside them.

The producer compiles `.co` **in-process via `libamd_comgr.so`** (not hipcc). Resolve it
with `-DROCKE_COMGR_LIB=<path/to/libamd_comgr.so>`, else `$ROCKE_COMGR_LIB` in the
environment, else the ROCm default. A box with the pyenv but **no comgr** SKIPS the
families (build still succeeds; parity tests skip) rather than hard-failing mid-build.

To run one producer by hand (using the build-local pyenv interpreter):
```
<build>/rocke-pyenv/bin/python \
    aot_catalog/gfx1151/fmha_wmma_fwd/produce_fmha_fwd_co.py /tmp/out
# /tmp/out now holds the family's <symbol>.co (pair with the checked-in family.json)
```

> **Multi-arch builds (TheRock).** A family compiles **only when its arch is in the
> build's `GPU_TARGETS`** (`AMDGPU_TARGETS` is honored as the legacy alias) — so a
> gfx942 build never compiles the gfx1151 kernels, and vice versa. The family arch and
> each target are matched on their `gfxNNN` base (feature suffixes like `:xnack-` are
> dropped), mirroring the runtime device match. If **neither** variable is set (a
> standalone dev build that never named targets) **all** families build. A requested
> arch with no family folder is simply absent at runtime (the engine declines) — never
> an error. This gate lives in `_aot_arch_requested()` in `aot_catalog/CMakeLists.txt`;
> future family adders should call it too.

**Dedicated build lane.** The engine defaults ON, but kernel *production* still needs rocke
turned on, so an explicit lane must set it or the families bitrot. That lane configures
`-DHIPKERNELPROVIDER_ENABLE_ROCKE=ON -DENABLE_AOT_CATALOG_ENGINE=ON -DGPU_TARGETS=gfx1151
-DROCKE_COMGR_LIB=<...>`, builds `hip_kernel_provider_tests` (the AOT GTests compile into
that shared binary), and runs `ctest`. TheRock's default lanes build the engine but leave
rocke OFF, so they ship the inert engine with an empty catalog until rocke is enabled there.

**1. Producer (build-time codegen)** — with rocke available and the arch requested, the
producer *must* succeed: it exits non-zero on any skipped/empty kernel so a partial family
fails the build rather than silently shipping an incomplete catalog.

**2. C++ substrate parity test** — drives the engine substrate directly and compares to
a CPU reference (`C=A@Bᵀ`, RMS over rows, or `softmax(scale·QKᵀ)·V`):
```
# configure with both options ON (above), build hip_kernel_provider_tests
ctest -R GemmNumericParity      # or RmsNorm* / SdpaNumericParity
```

**Empty-catalog behavior is deliberately asymmetric.** When rocke was unavailable at
configure time, no families were built and these tests **skip** on the empty catalog — the
expected state on TheRock CI today, which never has rocke. But when the build *did*
configure rocke-AOT families for its arch (the `AOT_CATALOG_FAMILY_TARGETS` global is
non-empty), CMake compiles the tests with `AOT_ROCKE_FAMILIES_EXPECTED=1` and an empty
catalog becomes a **hard failure** instead of a skip (see
`AOT_SKIP_OR_FAIL_ON_EMPTY_CATALOG` in
`src/tests/engines/aot_catalog_engine/AotCatalogTestSupport.hpp`). This closes the gap
where a producer or loader that silently dropped every kernel would otherwise leave CI
green over a total failure.

These parity/selection tests are **co-located in the family folder they test**
(`aot_catalog/<arch>/<family>/Test*.cpp`) — the family folder is a self-contained unit
(spec + build + tests), so removing a family removes its tests with it and touches no other
family. Each family's `CMakeLists.txt` registers its own test sources with
`rocke_add_aot_family_test(ARCH … SOURCES …)`, which appends them (as absolute paths) to
the `AOT_CATALOG_FAMILY_TEST_SOURCES` global property — **only when that arch is in the
build's `GPU_TARGETS`**, the same gate the kernel build uses. `src/tests/CMakeLists.txt`
then compiles those sources into `hip_kernel_provider_tests`: because the tests `#include`
the engine's private headers (`catalog/`, `launch/`, `plans/`), the **test binary** — not
this rocke subsystem — owns their compile (it adds the engine include dir and the
`AOT_CATALOG_TEST_DIR` compile def). This keeps rocke free of any build-graph dependency on
the throwaway engine.

The contract is that **a family's tests reference only that family's kernels**: e.g. the
reference (`gemm_wmma`) and tiled (`gemm_wmma_universal`) GEMM families both match the same
mult-of-16 shapes, so each of their tests selects its own kernel *by symbol* (`wmma_gemm`
vs `ugemm`) rather than taking `candidates.front()`, and neither depends on the other
existing. Tests that belong to **no single family** — the tune-cache and SDPA-decode
engine-substrate unit tests — stay at the engine test level
(`src/tests/engines/aot_catalog_engine/`).

**3. Real-model E2E — the hipDNN→PyTorch injection layer (a separate PR).** The hermetic
substrate parity test above proves a family's kernels are correct *in isolation*. Proving
the op is correct **and** actually selected inside a real PyTorch model needs a **second
piece**: a way to inject hipDNN into PyTorch. That is a thin layer which monkeypatches the
torch functionals (`F.linear` / `F.rms_norm` / `F.scaled_dot_product_attention`) so
supported calls route through a hard-pinned AOT graph (with a native fallback otherwise),
keeps an intercept census (AOT hits vs native fallbacks per shape/dtype), and A/Bs
`allclose` + timing against stock PyTorch — end to end, up to running a real model.

This injection layer is **general-purpose** — it is how you drive *any* hipDNN engine from
PyTorch, not specific to this throwaway catalog engine — so it will land as **its own PR
in the hipDNN project** (a reusable "inject hipDNN into PyTorch" example) rather than being
checked in here.

> **Honest-baseline caveat for SDPA A/Bs.** On gfx1151, vanilla
> `F.scaled_dot_product_attention` silently falls back to the **unfused math path**
> (~9× slower), which flatters any AOT comparison. A fair A/B must set
> `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` **before** importing torch and force each
> real fused backend (FLASH_ATTENTION / EFFICIENT_ATTENTION) explicitly.

### Debugging: "my kernel isn't being selected"

A missing/inapplicable kernel is a **silent fallback by design** — the engine declines and
another engine serves the graph, so the model never miscomputes (§1). The flip side is that
a kernel you *expected* to run just quietly doesn't, with no error. The engine exposes a
full resolution/load/decline trace behind `HIPDNN_AOT_DEBUG=1`; see the **debug section of
the [engine README](../../src/engines/aot_catalog_engine/README.md)** for exactly what it
prints. The KA-side checklist:

1. Is the catalog where the engine looked? By default the engine self-locates the catalog
   beside the loaded plugin `.so` — if you built locally but loaded an installed plugin,
   you're reading the install's catalog, not yours. Override with
   `HIPDNN_AOT_CATALOG_DIR=/path/to/<catalog>` (the dir containing the `<arch>/` subdirs).
2. Did the family load, or was it skipped? A skip line names the JSON/`.co` error.
3. Did your op decode and your kernel's constraints hold? The decline dump names the failing
   constraint key. Remember constraints are **fail-closed**: every constrained key must be
   present in the decoded problem *and* satisfied (§3).

> **Gotcha — stale `.co` after editing kernel internals.** The build re-runs a family's
> producer only when its `produce_<family>_co.py` or `family.json` changes (the
> `add_custom_command` DEPENDS on exactly those, plus the pyenv stamp). Editing rocke
> kernel *internals* that the producer pulls in — without touching the producer script or
> the JSON — will **not** re-trigger the `.co` rebuild, so you keep running the stale
> binary. Touch the producer (or delete the built `.co`) to force a rebuild.

---

## 10. Adding a brand-new op

For an op that isn't matmul/rmsnorm/sdpa, the pattern touches **both** the engine (a
reviewed C++ adapter) and this subsystem (a data-only family):

**In the engine (`src/engines/aot_catalog_engine/`, reviewed C++):**
1. New `ops/<Op>Adapter.{hpp,cpp}` implementing `IOpAdapter`: `opKind()`,
   `decode(graph) → optional<ProblemShape>` (gate on the FlatBuffers attributes union
   discriminant; fail closed on unsupported features), `buildBindings(graph, problem,
   kernel) → LaunchBindings`, `gridSymbols(problem, kernel) → SymbolTable`.
2. One `push_back` in `CatalogEngine.cpp`.
3. One source line in the engine's `CMakeLists.txt`.

**Here (data + tests):**
4. A self-contained family dir `aot_catalog/<arch>/<family>/` holding `family.json`, its
   `CMakeLists.txt` (`rocke_add_aot_family` + `rocke_add_aot_family_test`), the producer,
   and the parity test(s) — following §9.

No changes to `LaunchAbi`, the `Catalog` loader, `Selection`, `CatalogTypes`,
`CatalogPlan`, or the other adapters are needed — the substrate is op-agnostic **for an op
that fits it.**

> ⚠️ **A new adapter is necessary but not always sufficient.** The step list above holds for
> an op whose selection is measure-and-cache over a small, repeated shape space. An op that
> breaks that assumption needs **more than an adapter** — and the two most likely next ops
> both do:
> - **Convolution.** Its workspace *is* now expressible — im2col/split-K scratch as an
>   `elem_size`-scaled expression ([§4 workspace note](#workspace-a-constant-or-a-data-driven-expression))
>   — so that blocker is gone. What remains: its key space
>   `(N,C,H,W,K,R,S,stride,pad,dilation,dtype)` is effectively unbounded per model, so
>   measure-and-cache thrashes and never amortizes (engine README §4.1). A viable conv path
>   still needs a `ConvAdapter` **and** a selection strategy beyond measure-all — a larger lift
>   than "write a `ConvAdapter`," even with workspace solved.
> - **SDPA backward** (and any **multi-kernel plan**) breaks the
>   one-candidate = one-module = one-launch assumption baked into `CatalogPlan`: backward is a
>   3-stage pipeline with a shape-derived workspace. Also a substrate change.
>
> The full "data vs adapter-C++ vs new-substrate-capability" map is in the
> **[engine README §4](../../src/engines/aot_catalog_engine/README.md#4-capabilities-and-limits)**.
> If you're scoping one of these, budget for the substrate work, not just an adapter.

---

## 11. File map

| Thing | Path |
|-------|------|
| Family build glue (functions + discovery) | `aot_catalog/CMakeLists.txt` (auto-discovers `<arch>/<family>/`) |
| Per-family unit — self-contained (edit these) | `aot_catalog/<arch>/<family>/{family.json, CMakeLists.txt, produce_<family>_co.py, Test*.cpp}` |
| Built `.co` (not in git) | `${HIPDNN_AOT_CATALOG_BUILD_DIR}/<arch>/<family>/*.co` (emitted by the producer at build time) |
| Per-family parity/selection tests | `aot_catalog/<arch>/<family>/Test*.cpp` (registered via `rocke_add_aot_family_test`; deleted with the family) |
| Adapters (engine, reviewed C++) | `src/engines/aot_catalog_engine/ops/{Gemm,RmsNorm,Sdpa}Adapter.{hpp,cpp}` |
| Engine runtime (loader / selection / launch / tuning) | `src/engines/aot_catalog_engine/{catalog,plans,launch}/` |
| Engine-substrate tests (not family-specific) | `src/tests/engines/aot_catalog_engine/{TestTuneCache,TestSdpaDecode}.cpp` |
| Real-model E2E (hipDNN→PyTorch injection) | separate PR in the hipDNN project (not checked in here; §9) |

(`src/…` paths are relative to the hip-kernel-provider root; `aot_catalog/…` are relative
to this directory. The `.co` kernel binaries are **churning rocke build products** and are
compiled at build time, not vendored into git.)

---

## 12. Authoring a forward SDPA family on a new arch (gfx942/gfx950)

The universal forward adapter (§8) means bringing SDPA *forward* to gfx942/gfx950 is
**mostly data**: produce a `.co`, map its ABI to the by-name vocabulary below, write a
`family.json` whose capability constraints match exactly what the kernel handles, and copy
the parity test. You touch C++ only if the kernel needs an argument the vocabulary doesn't
emit yet (one reviewed line, see the arg table) or a grid the DSL can't express.

### Step checklist

1. **Produce the `.co`.** Instantiate the forward kernel from rocke's gfx942/gfx950 MFMA
   attention instances (not the gfx1151 WMMA ones — different tile/occupancy) in a
   co-located `produce_<family>_co.py`, mirroring
   `aot_catalog/gfx1151/fmha_wmma_fwd/produce_fmha_fwd_co.py`. Pin the rocke ref you build
   against (the rocke instance API churns).
2. **Read the kernel's kernarg ABI and map each arg to a canonical vocabulary name**
   (table below). If a quantity the kernel needs isn't emitted, add one emission in
   `SdpaAdapter::buildBindings` — reviewed C++, the single explicit extension point.
3. **Author `family.json`** (schema §3): `op_kind: "sdpa"`, `arch: "gfx942"`, one
   `kernels[]` entry per variant with (a) numeric-shape constraints (`dtype`, `D`, `H`,
   `H_kv`, `S_q`/`S_kv` `{ "min": 1, "multiple_of": … }` — the `min: 1` rejects a
   zero-extent seqlen that `multiple_of` alone would admit), (b) **a capability constraint
   for every fact the kernel does *not* universally handle** (see the warning below),
   (c) `grid`/`block`, and (d) `args_signature` = the kernel's real ABI **in order**, drawn
   from the vocabulary.
4. **Drop it under `aot_catalog/gfx942/<family>/`** with a producer `CMakeLists.txt`
   (auto-discovered; mirror the gfx1151 family's). It compiles only when `gfx942` is in the
   build's `GPU_TARGETS` (see §9); a gfx1151-only build skips it.
5. **Copy the parity test into the new family folder.** `cp` the gfx1151 family's
   `TestSdpaNumericParity.cpp` → `aot_catalog/gfx942/<family>/`, change `kArch`, geometry
   (D/H/S), the dtype token, and the hand-built `LaunchBindings` to match the new
   `args_signature`. The CPU reference softmax and tolerances carry over. Register it from
   the family's own `CMakeLists.txt` with `rocke_add_aot_family_test(ARCH gfx942 SOURCES …)`
   — the test lives with the family, not under `src/tests/`. (`TestSdpaDecode.cpp` is
   arch-neutral, family-independent, and stays at the engine test level — no per-arch copy
   needed.)

### Arg-vocabulary reference

Every canonical name `buildBindings` emits. Pick the subset your kernel takes into
`args_signature` (in ABI order); `LaunchAbi::bindArgs` resolves by name and **fails closed
on a name the adapter didn't emit**. All strides follow the BHSD mapping (token = S axis =
`stride(2)`, head = H axis = `stride(1)`, batch = B axis = `stride(0)`).

Names in the **Optional pointers** block are bound **only when the graph carries that
tensor** — each pairs with a capability fact `decode` publishes (constrain the fact, then
name the pointer). Naming an optional pointer whose tensor is absent fails closed, so a
kernel and its `family.json` constraints must agree.

| name | type | meaning / derivation |
|------|------|----------------------|
| **Always emitted** | | |
| `Q` `K` `V` `O` | ptr | the four operands, by tensor uid |
| `scale_log2` | f32 | `attn_scale * log2(e)` (base-2 softmax; gotcha #1 §8) |
| `scale_raw` | f32 | the un-multiplied `attn_scale` (default `1/sqrt(D)`) |
| `seqlen_q` `seqlen_k` | i32/i64 | S_q, S_kv (fixed-length scalar values) |
| `stride_{q,k,v,o}_token` | i32/i64 | S-axis stride, **elements** |
| `stride_{q,k,v,o}_head` | i32/i64 | H-axis stride, **elements** |
| `stride_{q,k,v,o}_batch` | i32/i64 | B-axis stride, **elements** |
| `stride_{q,k,v,o}_token_bytes` | i32/i64 | S-axis stride × element size (for byte-stride ABIs) |
| `stride_{q,k,v,o}_head_bytes` | i32/i64 | H-axis stride × element size |
| `stride_{q,k,v,o}_batch_bytes` | i32/i64 | B-axis stride × element size |
| `H` `H_kv` `D` `B` | i32/i64 | head count, kv-head count, head dim, batch |
| `gqa_ratio` | i32/i64 | `H / H_kv` |
| **Optional pointers** (bound only when the tensor is present) | | pairs with fact |
| `attn_mask` | ptr | additive mask tensor — `has_attn_mask` |
| `block_mask` | ptr | block-sparse mask table — `has_block_mask` |
| `sink` | ptr | attention-sink token tensor — `has_sink` |
| `scale_tensor` | ptr | runtime scale tensor — `runtime_scale` |
| `seqlen_q_ptr` `seqlen_kv_ptr` | ptr | varlen cumulative-seqlen tables — `varlen` |
| `page_table_k` `page_table_v` | ptr | paged-KV block tables — `paged` |
| `dropout_mask` `dropout_scale` `dropout_seed` `dropout_offset` | ptr | dropout plumbing — `has_dropout` |
| `rng_dump` | ptr | optional RNG debug dump output — independently settable, does not gate `has_dropout` |
| `descale_q` `descale_k` `descale_v` `descale_s` | ptr | fp8 input/intermediate descales — `fp8` |
| `scale_s` `scale_o` | ptr | fp8 output scales — `fp8` |
| `amax_s` `amax_o` | ptr | fp8 output amax accumulators — `fp8` |
| `stats` / `lse` | ptr | log-sum-exp output — aliases — `gen_stats` |
| `max` `sum_exp` | ptr | split-form softmax stats outputs — `gen_stats` |

`type` in `args_signature` (`i32`/`i64`/`f32`) selects how the value is narrowed/packed
(§3); the same emitted quantity can be taken as `i32` or `i64`. A quantity not in this
table = one added `bindings.scalars.emplace(...)` (scalar) or `bindOptionalPtr(...)`
(pointer) in `buildBindings` (reviewed).

### Capability-key reference

Every fact `decode` publishes (full list with "true when" in §8). A kernel **opts in or
out via a constraint** in `family.json`:

- `{"causal": {"equals": false}}` — the kernel handles only the non-causal case.
- `{"gqa_ratio": {"equals": 1}}` — MHA only. Use `{"min": 1}` or omit-with-care for a
  GQA-capable kernel that reads `H_kv`/`gqa_ratio`.
- `{"d_contiguous": {"equals": true}}`, `{"batch_foldable": {"equals": true}}` — structural
  requirements of a kernel with no D-stride / no batch-stride arg.
- `{"D": {"equals": 128}}`, `{"S_q": {"multiple_of": 64}}` — numeric-shape tiers.

> ⚠️ **Omitting a capability constraint asserts the kernel handles that case.** This is the
> deliberate tradeoff of data-gated selection: the adapter no longer declines features in
> C++, so *a missing constraint is a claim*, not a safe default. If a kernel only does
> non-causal MHA, it **must** carry `causal {equals false}`, `has_attn_mask {equals
> false}`, `gqa_ratio {equals 1}`, etc. — otherwise a causal or GQA or masked graph will
> **select it and compute a wrong answer**. Fail-closed for features now lives in your
> `family.json`, not the adapter. (The universal memory-safety gates — rank-4, dtype match,
> integer gqa_ratio — remain in C++ and always hold; §8 decline boundary.) The shipped
> gfx1151 family constrains *every* capability key explicitly; copy that discipline.

### Copy-paste `family.json` template

Fill in `<arch>`, symbols/co_files, `D`/`H`/`H_kv`, the `multiple_of` tile sizes, and the
`args_signature` your kernel actually uses. Constrain **every** capability key the kernel
does not handle (the block below is the safe, fully-explicit default: non-causal MHA,
contiguous-D, batch-foldable, no masks/dropout/paged/varlen/stats/fp8/runtime-scale — the
same posture as the gfx1151 reference).

```json
{
    "family": "fmha_<algo>_fwd_<arch>",
    "op_kind": "sdpa",
    "arch": "<arch>",
    "dtype": ["f16", "bf16"],
    "kernels": [
        {
            "symbol": "<exported_symbol_in_co>",
            "co_file": "<symbol>.co",
            "constraints": {
                "dtype": { "equals": "f16" },
                "D":     { "equals": 128 },
                "H":     { "equals": 32 },
                "H_kv":  { "equals": 32 },
                "S_q":   { "min": 1, "multiple_of": 64 },
                "S_kv":  { "min": 1, "multiple_of": 64 },
                "gqa_ratio":           { "equals": 1 },
                "d_contiguous":        { "equals": true },
                "batch_foldable":      { "equals": true },
                "causal":              { "equals": false },
                "causal_bottom_right": { "equals": false },
                "has_diagonal_band":   { "equals": false },
                "has_mma_core_mode":   { "equals": false },
                "has_alibi":           { "equals": false },
                "has_padding_mask":    { "equals": false },
                "has_attn_mask":       { "equals": false },
                "has_block_mask":      { "equals": false },
                "has_sink":            { "equals": false },
                "has_dropout":         { "equals": false },
                "paged":               { "equals": false },
                "varlen":              { "equals": false },
                "gen_stats":           { "equals": false },
                "fp8":                 { "equals": false },
                "runtime_scale":       { "equals": false }
            },
            "grid": { "x": { "ceil_div": ["S_q", 64] }, "y": "H", "z": "B" },
            "block": [256, 1, 1],
            "args_signature": [
                { "name": "Q", "type": "ptr" },
                { "name": "K", "type": "ptr" },
                { "name": "V", "type": "ptr" },
                { "name": "O", "type": "ptr" },
                { "name": "scale_log2", "type": "f32" },
                { "name": "seqlen_q", "type": "i32" },
                { "name": "seqlen_k", "type": "i32" },
                { "name": "stride_q_token", "type": "i32" },
                { "name": "stride_q_head",  "type": "i32" },
                { "name": "stride_q_batch", "type": "i32" },
                { "name": "stride_k_token", "type": "i32" },
                { "name": "stride_k_head",  "type": "i32" },
                { "name": "stride_k_batch", "type": "i32" },
                { "name": "stride_v_token", "type": "i32" },
                { "name": "stride_v_head",  "type": "i32" },
                { "name": "stride_v_batch", "type": "i32" },
                { "name": "stride_o_token", "type": "i32" },
                { "name": "stride_o_head",  "type": "i32" },
                { "name": "stride_o_batch", "type": "i32" }
            ],
            "workspace_bytes": 0
        }
    ]
}
```

(The `args_signature` above shows a batch-stride-carrying ABI — a natural gfx942/gfx950
shape that accepts `B > 1` non-folded — as a contrast to the gfx1151 kernel's 15-arg,
no-batch-stride signature in §8. Take only the args your kernel really has.)
