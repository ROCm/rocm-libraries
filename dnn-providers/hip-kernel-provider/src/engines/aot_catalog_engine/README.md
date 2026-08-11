<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# The AOT catalog engine — runtime / loader

This is the **AOT (ahead-of-time) catalog engine** in the HIP kernel provider: a
deliberately thin, **throwaway** bring-up path that loads loose, rocke-authored code
objects (`.co` / HSACO) described by data-only `family.json` files, and launches them for
matmul / rmsnorm / sdpa graphs. It exists so kernels can be brought into hipDNN **as data**
— no C++ edit per kernel — for end-to-end experimentation on gfx1151 (and beyond).

This README covers the **engine's runtime behavior only**: what it loads, how it resolves
and validates the catalog, how it declines, its capability/limits, and how it is built.
**Kernel authoring lives elsewhere** — the families, producers, `family.json` schema, and
per-op ABIs are owned by the KA teams in the rocke subsystem. See
**[`rocke/aot_catalog/README.md`](../../../rocke/aot_catalog/README.md)** for the
authoring guide.

> **This engine is ON by default** (`ENABLE_AOT_CATALOG_ENGINE=ON`) but is **inert until a
> catalog is present** — with nothing dropped beside the plugin it loads zero kernels and
> declines every graph. It is compiled in by default precisely so a catalog that appears
> later (kernels dropped locally, or the rocke families enabled in TheRock) is picked up with
> **no provider rebuild**. Building the *kernels* is the separate opt-in (see [§6](#6-how-it-is-built)).

**Key property — the runtime has ZERO dependency on rocke.** The engine only *reads* the
catalog at runtime; it does not build kernels and does not link or import anything from
rocke. Adding or removing a kernel family is entirely a rocke-side change — this engine
never moves. The only contract between the two is the on-disk catalog layout and the
`family.json` format described below.

---

## 1. What happens at runtime

```
torch op  (F.linear / F.rms_norm / F.scaled_dot_product_attention / …)
        │  (optional) a torch-functional monkeypatch that routes to AOT
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
hipModuleLaunchKernel  →  the .co
```

Three op adapters ship (`GemmAdapter`, `RmsNormAdapter`, `SdpaAdapter`), each mapping one
op kind to a launch. If no adapter decodes the graph, or no kernel's constraints match, the
engine **declines** and another hipDNN engine serves the graph — a missing or inapplicable
kernel is always a fallback, **never a wrong answer**.

**Selection is measure-and-cache.** Constraints prune the family to the candidates that
*can* run a problem; `CatalogPlan` then times every survivor on the real hardware and caches
the fastest, keyed on `family + canonical problem` (cache file: env `HIPDNN_AOT_TUNE_CACHE`,
else a temp file). There is no heuristic or cost model. See
[§4](#4-capabilities-and-limits) for what this does and doesn't scale to.

---

## 2. What it loads — the catalog

The engine reads a catalog tree of the form:

```
<catalog-root>/<arch>/<family>/
    family.json      # the spec: kernels[] + constraints + ABI/grid + identity
    <symbol>.co      # one or more compiled code objects (HSACO)
```

`family.json` is the runtime source of truth: each `kernels[]` entry names its `.co`
(`co_file`), its exported `symbol`, its `constraints` (the applicability predicate), its
`grid`/`block`, and its `args_signature` (the by-name launch ABI). The full schema and the
per-op ABIs are in the [authoring guide](../../../rocke/aot_catalog/README.md#3-familyjson-schema-shared-by-all-ops).

### Loader is fail-closed and fail-loud

The loader enforces, at load time:

- **Unknown fields are rejected, not ignored** — a misspelled field name (`"constraint"`,
  `"mutiple_of"`) fails the file rather than silently dropping the predicate it carried.
  (Comment keys must be `_`-prefixed, e.g. `_comment`.)
- **Constraints are required, non-empty, and fail-closed** — a kernel with no constraints
  would claim it serves every problem shape, so an empty/omitted map is a load error. At
  match time every constrained key must be *present in the decoded problem* **and** satisfy
  its rule, or the candidate is skipped.
- **`arch` must match the directory** it lives under (`<arch>/<family>/`) — a file copied
  into the wrong arch folder (kernels built for the wrong GPU) is a load error.
- **A missing or empty `.co`** named by `co_file` skips that family with a named error
  rather than being catalogued as valid and failing later at `hipModuleLoad`.

The single way this design can return a *wrong* answer is an **under-constrained**
`family.json` (a kernel claiming applicability it doesn't have). The adapters guarantee
memory safety (rank / dtype / shape agreement); correctness of *applicability* is the
family author's responsibility. That risk lives entirely on the authoring side — see the
⚠️ warnings in the [authoring guide](../../../rocke/aot_catalog/README.md#3-familyjson-schema-shared-by-all-ops).

---

## 3. Catalog resolution and debugging

### Resolution order

The engine resolves the catalog root in this order:

1. **`HIPDNN_AOT_CATALOG_DIR`** env var, if set — explicit override, always wins. Point it
   at the dir that contains the `<arch>/` subdirs.
2. **Beside the loaded plugin `.so`** — `<plugin-dir>/arch_content/aot_catalog` (the reldir
   is baked as `HIPDNN_AOT_CATALOG_RELDIR`). Both the build tree and the install tree place
   the catalog at exactly this offset, so a single relative path serves both. Used **unconditionally**
   when the plugin's own directory can be determined (via `dladdr`), *even if that catalog
   is missing/empty*.
3. **Baked install path** (`HIPDNN_AOT_CATALOG_DIR` compile def) — only if the plugin's
   directory can't be resolved at all.

Step 2 is deliberate: a **locally-built or force-loaded plugin reads its OWN build-tree
catalog** and never silently crosses over to a system install's catalog (or vice-versa). If
you `LD_LIBRARY_PATH`-force your local `hip_kernel_provider.so` while a real hipDNN is
installed, you get *your* kernels. (The AITER ASM engine does **not** yet self-locate — it
resolves `HIPDNN_AITER_ASM_DIR` env → baked install path — so it has the same
cross-contamination footgun; the same fix could apply there if it becomes a problem.)

The location contract (the reldir and both tree offsets) is defined **once** in the
top-level `hip-kernel-provider/CMakeLists.txt` (`HIPDNN_AOT_CATALOG_RELDIR` /
`_BUILD_DIR` / `_INSTALL_DIR`) — the same constants the rocke producer emits into — so
producer and runtime can never drift.

> The offset lives under `arch_content/` for two reasons. (1) **Packaging:** TheRock ships
> this device library's non-`.so` files only via the recursive glob
> `**/engines/arch_content/**`; content placed elsewhere works in a local dev build but is
> silently dropped from the installed package (empty catalog). (2) **Loader safety:** the
> offset must **not** begin with `hip_kernel_provider` — the frontend loader, given the
> absolute basename `hip_kernel_provider`, would treat a directory of that name in `engines/`
> as a plugin dir (find no `.so` inside) and load zero engines. `arch_content` satisfies both,
> so the offset is `arch_content/aot_catalog`.

### Debugging: "a kernel I expected isn't selected"

A missing/inapplicable kernel is a **silent fallback by design** (§1). To see *why*, two
env vars make the engine's reasoning visible (the engine's own `INFO`/`WARN`/`ERROR`
breadcrumbs are off by default — `HIPDNN_LOG_LEVEL` defaults to `off`):

```
HIPDNN_AOT_DEBUG=1     # always-on stderr trace of catalog resolution + load + per-graph decline
HIPDNN_LOG_LEVEL=info  # the plugin-wide log level (also surfaces other engines' logs)
```

`HIPDNN_AOT_DEBUG=1` prints, independent of the log level:

- **Resolution:** the catalog root it chose and **how** (`env` / `self-located beside
  plugin .so` / `baked install path`).
- **Load:** the arch dir it scanned, each family loaded (name, op_kind, kernel count), and
  each family **skipped** with the parse/`co_file` error — so a malformed `family.json` or
  a missing `.co` is named, not swallowed.
- **Decline:** which stage bailed — arch dir missing, catalog empty, no adapter decoded the
  op, or an op decoded but **no kernel matched**. For the last it lists every kernel of that
  op_kind and the first constraint that filtered it, including the common trap of a
  constraint key **absent from the decoded problem shape** (a typo, or a key the adapter
  doesn't publish — fail-closed).

> **Future:** if these footguns recur, a small `catalog-doctor` tool could formalize this —
> dump the resolved root, validate every `family.json`, list families/kernels, and dry-run
> a problem shape against the constraints — instead of relying on the `HIPDNN_AOT_DEBUG`
> trace. Not built yet.

---

## 4. Capabilities and limits

This is the honest map of what the design does well and where it stops. None of the limits
are bugs — they are the deliberate edges of a thin bring-up engine. (Which specific SDPA and
conv extensions on gfx942/gfx950/gfx1250 are *data*, which are *adapter C++*, and which need
a *new substrate capability* is enumerated in the authoring guide's SDPA new-arch kit.)

### 4.1 Selection is "measure them all, cache the winner" — and nothing else

There is **no heuristic, no analytic cost model, and no shape-bucketed tuning database.**
Selection is exactly two steps: `constraints` prune the family to the candidates that *can*
run this problem, then `CatalogPlan` **times every survivor on the real hardware** and
caches the fastest. The winner is the measured winner — correct by construction, with no
model that can be wrong, and the kernel author ships candidates instead of hand-writing a
selector. That is the core new power (§4.2).

The cost of that simplicity is three structural scaling limits:

| # | Limit | Consequence | Bites hardest on |
|---|-------|-------------|------------------|
| 1 | **First-execute tax ∝ candidate count.** No pre-filter before timing — every candidate the constraints didn't prune is module-loaded and timed on the first execute of a shape. | Keep the per-problem candidate set to a handful. A family with a *large* flat kernel list (e.g. AITER's 290-entry `fmha_fwd.csv`) would time every survivor a shape leaves. | Large prebuilt/ASM families. |
| 2 | **The cache only amortizes when the problem-key space is small and repeated.** Every *new* key re-tunes from scratch. | Great for LLM decode and fixed model shapes (tune once, reuse forever). **Conv is the antithesis:** its key space `(N,C,H,W,K,R,S,stride,pad,dilation,dtype)` is effectively unbounded per model, so the cache thrashes and the tuning tax never amortizes. | **Conv**, dynamic-shape workloads. |
| 3 | **`constraints` are the only pruning lever.** Rules are `equals` / `multiple_of` / bounds. | Real selection intelligence lives entirely in how sharply the author writes constraints. Pruning that *isn't* a per-key equality/divisibility/bound — "pick the tile by M:N aspect ratio," "prefer split-K past this K" — cannot be expressed as a constraint at all. | Ops with many tile/algorithm variants. |

**Crossing this wall is a substrate change, not a data addition.** Making conv (or a big
ASM family) viable would need what we deliberately don't have: an analytic pre-selector, or
a shipped tuning DB keyed on shape buckets, to cut the candidate set *before* timing.

### 4.2 What grows as pure data (zero C++)

Along the axes the design was built for, coverage grows as data:

- **Ship N candidate kernels for one problem and get automatic best-pick** — no
  hand-written selector, no heuristic table. This is the genuine new capability.
- **Add a dtype / tile / shape-tier variant as data** — bf16 GEMM and bf16 RMSNorm each
  landed with *zero* C++ change, as `kernels[]` entries carrying a `dtype` constraint.
- **Add a whole arch as a folder** — the C++ loads `.co` by arch string and never learns
  the arch name; drop `<arch>/<family>/` and the loader picks it up.
- **Mix build backends in one catalog** — rocke-compiled and prebuilt-`.co` (AITER ASM)
  families coexist; the per-family `CMakeLists.txt` is the variation point.
- **SDPA forward across arches and features** (GQA, causal, masks, fp8, varlen) is now data
  too, gated by `family.json` against the universal forward adapter's fixed by-name
  vocabulary — once a kernel that serves the case exists.
- **Shape-scaled workspace as data** — `workspace_bytes` accepts a JSON-AST expression over
  the problem's grid symbols + `elem_size`, evaluated per-problem (a bare integer is the
  degenerate constant). So im2col/split-K-style scratch that grows with the problem is now
  authorable without a C++ change; the evaluator is arithmetic + clamp only (no conditionals —
  those are modeled as separate candidates). See the authoring guide's workspace note.

The load-bearing precondition on all of that: **it is only data-free when the op already
has an adapter and the kernel fits that adapter's fixed ABI.**

### 4.3 The walls that need C++ or a new substrate capability

- **A forward SDPA argument the vocabulary doesn't emit yet**, or a heuristic grid
  transform the DSL can't express — one reviewed line in the adapter / launch layer.
- **The 16-byte SGPR-slot kernarg padding** hand-written AITER ASM kernels need — a
  `packArgs` change in `LaunchAbi` (natural alignment today), deferred until a real ASM
  forward kernel lands.
- **SDPA backward** — breaks the model outright: `CatalogPlan` assumes one candidate = one
  module = one launch. Backward is a 3-stage pipeline (odo → dqdkdv → dq_convert); its
  workspace is now sizeable as an expression (§4.2), but the multi-kernel plan and the
  `accumulator_type` knob that drives selection are not. **New substrate capability**
  (multi-kernel plan).
- **Conv2d/Conv3d** — no `ConvAdapter` exists yet (a new op). Its workspace is now expressible
  as data (§4.2), so that is no longer the blocker; what remains is that measure-and-cache
  degrades on conv's unbounded shape space (§4.1 #2). Largest lift: a new adapter **plus** a
  selection strategy beyond measure-all.

**One line:** a correct, data-extensible best-pick engine that excels when the op's ABI is
fixed and the shape space is small and repeated — GEMM, norms, attention. It extends for
free along dtype / tile / arch-as-folder and (for SDPA forward) feature-as-data; it hits
real walls the moment work needs a novel launch arg / grid transform, a multi-kernel plan,
or selection over an unbounded shape space.

---

## 5. Runtime file map

| Thing | Path (relative to this engine dir) |
|-------|------|
| Engine + adapter registration | `CatalogEngine.{hpp,cpp}` (one `push_back` per adapter) |
| Adapter interface | `ops/IOpAdapter.hpp` |
| Adapters | `ops/{Gemm,RmsNorm,Sdpa}Adapter.{hpp,cpp}` |
| Catalog loader / selection | `catalog/{Catalog,Selection,CatalogTypes,ModulePath,AotDebug,TuneCache}.*` |
| Launch ABI (arg packing, grid/block eval) | `launch/` |
| Plan (measure-and-cache) | `plans/CatalogPlan.*` |
| Engine-substrate tests (not family-specific) | `../../tests/engines/aot_catalog_engine/{TestTuneCache,TestSdpaDecode}.cpp` |
| **Kernel families, producers, per-family tests (authoring)** | **`../../../rocke/aot_catalog/<arch>/<family>/` — see its [README](../../../rocke/aot_catalog/README.md)** |

The `.co` kernel binaries are **churning rocke build products**, compiled at build time into
`<plugin-dir>/arch_content/aot_catalog/<arch>/<family>/`, never vendored into git.

---

## 6. How it is built

Two independent CMake options separate *loading* the catalog from *producing* it — do not
conflate them:

- **`ENABLE_AOT_CATALOG_ENGINE`** (default **ON**) compiles this engine into
  `libhip_kernel_provider.so`. It is safe to default ON because the engine is **inert without
  a catalog** — an unpopulated engine loads zero kernels and declines every graph, so it can
  ship in every build and pick up a catalog that appears later with no provider rebuild.
- **`HIPKERNELPROVIDER_ENABLE_ROCKE`** (default **OFF**) builds the rocke subsystem — and,
  when `ENABLE_AOT_CATALOG_ENGINE` is *also* ON, the `rocke/aot_catalog/` families that
  produce this engine's `.co`. This is the real opt-in: producing kernels. rocke is a
  build-time **tool**; this engine has **no** build- or link-time reference to it.

The four combinations:

| `ENABLE_AOT_CATALOG_ENGINE` | `HIPKERNELPROVIDER_ENABLE_ROCKE` | result |
|---|---|---|
| ON *(default)* | OFF *(default)* | engine built, **empty catalog** → inert, declines every graph, GPU parity tests skip. This is the normal TheRock build. |
| ON | ON | engine built **and** families produced → full catalog, parity tests run |
| OFF | any | engine not built at all (opt out entirely) |
| OFF | ON | rocke built, but `aot_catalog/` not configured (gated on the engine option) |

The lane that produces the kernels + runs the family parity tests configures
`-DHIPKERNELPROVIDER_ENABLE_ROCKE=ON -DENABLE_AOT_CATALOG_ENGINE=ON -DGPU_TARGETS=gfx1151
-DROCKE_COMGR_LIB=<path/to/libamd_comgr.so>` and builds `hip_kernel_provider_tests`. The
family build, arch gating, producer invocation, and empty-catalog test semantics are all
documented in the [authoring guide](../../../rocke/aot_catalog/README.md#9-how-to-test).

When rocke eventually moves out of `hip-kernel-provider/rocke/` to its own home, this engine
is unchanged — only the rocke-side producer path moves.
