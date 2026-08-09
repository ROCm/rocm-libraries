<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# The AOT catalog engine — kernel-author guide

This is the hands-on guide for the **AOT (ahead-of-time) catalog engine** in the HIP
kernel provider. It is written for the rocKE kernel-authoring team so you can bring
your own gfx1151 (and beyond) kernels into hipDNN and do end-to-end testing **without
writing or rebuilding any C++** — you drop a compiled code object (`.co`) and edit a
JSON file.

The engine is a deliberately thin, throwaway bring-up path. Its whole point is that a
kernel author owns kernels *as data*: an ahead-of-time-compiled code object (`.co` /
HSACO) plus a `family.json` describing how to **select** and **launch** it. The C++
side is a small set of fixed, reviewed *adapters* (one per op kind) that map a hipDNN
op graph to a launch ABI. You extend coverage and tuning **data-only**; you touch C++
only when the *ABI itself* changes or you teach the engine a genuinely new op.

Three op families ship today, each proven end-to-end on gfx1151 (AMD Radeon 8060S /
Strix Halo, RDNA3.5):

| op kind    | adapter          | shipped families                                   | §  |
|------------|------------------|----------------------------------------------------|----|
| `matmul`   | `GemmAdapter`    | `gemm_wmma_gfx1151`, `gemm_wmma_universal_gfx1151`  | [§6](#6-gemm--matmul) |
| `rmsnorm`  | `RmsNormAdapter` | `rmsnorm2d_gfx1151`                                 | [§7](#7-rmsnorm) |
| `sdpa`     | `SdpaAdapter`    | `fmha_wmma_fwd_gfx1151`                             | [§8](#8-sdpa-dense-flash-attention-forward) |

A **family is one algorithm**, not one dtype: every family carries its f16 *and* bf16
kernels in a single flat `kernels[]` list, each kernel naming its own `dtype`
constraint (§3). All kernels in a family share the algorithm's tunable knobs — those
knobs are baked into each `.co` at produce time, and their consequences surface as each
kernel's per-kernel `constraints`.

Sections [§1](#1-the-end-to-end-path-what-happens-at-runtime)–[§5](#5-measure-and-cache-tuning)
are the engine mechanics that apply to **every** op. Sections §6–§8 are the per-op
specifics (ABI, gotchas, decline boundary). [§9](#9-how-to-test)–[§11](#11-file-map)
cover testing, adding a brand-new op, and the file map.
[§12](#12-capabilities-and-limits-and-the-walls-beyond-gfx1151) is the honest
capability map: what this design does well, and exactly where it stops — read it before
scoping SDPA/conv work on gfx942/gfx950/gfx1250.

---

## 1. The end-to-end path (what happens at runtime)

```
torch op  (F.linear / F.rms_norm / F.scaled_dot_product_attention / …)
        │  (optional) a tools/comfyui_hipdnn_*_override.py monkeypatch
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
kernel is a fallback, never a wrong answer.

---

## 2. Self-serve recipe (the 30-second version)

1. Compile your kernel to a code object for the target arch with the **exact ABI** for
   its op kind (§6–§8). For a rocKE-AOT family the family's co-located producer script
   does this at build time (§9); for a prebuilt family you check the `.co` into the
   folder yourself.
2. Add a `kernels[]` entry to the family's `family.json` (schema in §3) pointing at the
   `.co` by name, with the `dtype` (and other) constraints your kernel requires and the
   grid it launches with. One folder per algorithm — `library/<arch>/<family>/`, e.g.
   `library/gfx1151/gemm_wmma/` — holds every dtype's kernels flat.
3. Build the provider (rocKE-AOT families run their producer → emit `.co` + stage
   `family.json` into the build tree; see §9), **or** for a quick data-only iteration
   point `HIPDNN_AOT_CATALOG_DIR` at a populated `<arch>/<family>/` tree directly.
4. Verify with the A/B rig / parity test (§9).

To add a **tuning candidate** for a shape you already serve, you only do step 1+3:
append another `kernels[]` entry with overlapping constraints. `CatalogPlan` measures
all applicable candidates on the first execute and caches the fastest per problem (§5).

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
`rmsnorm2d`'s 10 f16 vs 8 bf16 specializations). There is no `sections` grouping and no
`{dtype}` token — selection is per-kernel via the `dtype` constraint.

Each `kernels[]` entry:

| field             | meaning |
|-------------------|---------|
| `symbol`          | the kernel's exported symbol *inside* the `.co` (for `hipModuleGetFunction`); also the tune-cache key. **Unique per candidate.** |
| `co_file`         | `.co` filename, relative to the family dir. |
| `constraints`     | map `problem_key → rule`. Rules: `{"equals": v}` (int/string/bool) and `{"multiple_of": n}`. **Fail-closed:** every constrained key must be present in the decoded problem *and* every rule must hold, or the candidate is skipped. This is how the `dtype` constraint selects the f16 vs bf16 kernels within one family. |
| `grid`            | per-axis `x`/`y`/`z`. A value is a constant, a problem-key string (`"M"`, `"H"`, `"B"`), or `{"ceil_div": ["<key>", n]}`. Evaluated from that op's `gridSymbols`. |
| `block`           | `[x,y,z]` constant workgroup size. |
| `shared_mem_bytes`| omit for static-LDS kernels (defaults to 0). |
| `workspace_bytes` | this kernel's scratch need (0 for all shipped reference kernels). |
| `args_signature`  | the launch ABI as an ordered list of `{name,type}` (`ptr`/`i32`/`f32`). **Must match the op's ABI order exactly** (§6–§8); `LaunchAbi` packs by name in this order. |

The set of legal `constraints`/`grid` keys is exactly the **problem keys the adapter
emits** for that op — listed per op in §6–§8.

---

## 4. When you need a C++ change (vs data-only)

The self-serve path (data only) covers:

- **new kernels** for a shape the adapter already decodes;
- **new dtypes** — add more `kernels[]` entries to the same family, each with its own
  `dtype` constraint (proven: bf16 GEMM and bf16 RMSNorm were each added with *zero*
  C++ change);
- **tuning candidates** — more `kernels[]` entries with overlapping constraints (§5).

You must edit the adapter (`ops/<Op>Adapter.{hpp,cpp}`) and re-review/rebuild only when
the **contract** changes:

- a **different ABI** (arg added/removed/reordered, or a different scalar convention) —
  `buildBindings` and the `args_signature` must agree;
- a **new capability to decode** (e.g. SDPA causal masking, GQA `H_kv != H`, a runtime
  scale tensor) — `decode` must stop declining it and emit any new problem keys;
- a **grid/launch shape** the grid DSL can't express.

For a genuinely new op, see §10. For how these boundaries compound when you take SDPA and
conv to gfx942/gfx950/gfx1250 — and which of them are data, which are adapter C++, and
which need a new substrate capability — see §12.

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
- **Reference** (`gemm_wmma_*`): rocKE `wmma_gemm`, one wave32 per 16×16 output tile,
  no LDS staging — correctness-first, launch-overhead-cheap (wins on tiny shapes).
- **Tiled** (`gemm_wmma_universal_*`): rocKE `build_universal_gemm`, LDS-staged,
  register-blocked (tile 64×64×32, warp 2×2, wt 16×16×16). 3–7× faster than the
  reference at large shapes; the tune cache picks per shape.

### Layout — RCR only (`y = x @ Wᵀ`, i.e. `nn.Linear`)
hipDNN `MatmulAttributes` carries only a/b/c UIDs and **no transpose flag** — the
transpose is expressed by **strides**. `GemmAdapter::decode` reads logical dims and
**gates on RCR strides**: A `[M,K]` row-major, B logical `[K,N]` with strides `{1,K}`
(physical `[N,K]` weight), C `[M,N]` row-major. Anything else declines. This is exactly
`nn.Linear`'s `x @ weightᵀ`. The kernel has **no epilogue** — bias/activation are the
caller's job (the `comfyui_hipdnn_linear_override` adds bias natively post-matmul).

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

### Files
Adapter `ops/GemmAdapter.{hpp,cpp}`; data + co-located producers
`library/gfx1151/gemm_wmma/{family.json, produce_gemm_wmma_co.py}`,
`library/gfx1151/gemm_wmma_universal/{family.json, produce_gemm_universal_co.py}`;
parity test `TestGemmNumericParity.cpp`; A/B rig `tools/gemm_aot_ab.py`; model override +
driver `tools/comfyui_hipdnn_linear_override.py`, `tools/ltx_linear_ab.py`.

---

## 7. RMSNorm

**op_kind `"rmsnorm"` · adapter `RmsNormAdapter` · family `rmsnorm2d_gfx1151` (f16 +
bf16 kernels flat).**

### The kernels
rocKE CK-Tile `10_rmsnorm2d`: per-row RMS over the last dim of a 2D `[M,N]` tensor with
a per-column weight `Gamma[N]` (Llama/Mistral RMSNorm). Two body shapes exist —
single-pass VGPR-cached vs two-pass streaming — selected by `elems_per_thread =
N/block_size`; both are perf-only (identical correct output), so they're tuning
candidates (§5). Higher-dimensional inputs are flattened to `[M,N]` by the override.

- **Static variants** bake `N` → constraint `{"N": {"equals": <n>}}` (exact-match
  shape tiers, e.g. N=2048/4096).
- **Runtime-N variants** (symbol suffix `_dyn_`, rocKE `rmsnorm2d_dynamic.py`) read `N`
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
Adapter `ops/RmsNormAdapter.{hpp,cpp}`; data + co-located producer
`library/gfx1151/rmsnorm2d/{family.json, produce_rmsnorm2d_co.py}`; rocKE runtime-N
instance `instances/common/rmsnorm2d_dynamic.py`; parity + selection tests
`TestRmsNormNumericParity.cpp`, `TestRmsNormSelection.cpp`; A/B rig
`tools/rmsnorm_aot_ab.py`; model override + driver
`tools/comfyui_hipdnn_rmsnorm_override.py`, `tools/ltx_rmsnorm_ab.py`.

---

## 8. SDPA (dense flash-attention forward)

**op_kind `"sdpa"` · adapter `SdpaAdapter` · family `fmha_wmma_fwd_gfx1151` (f16 + bf16
kernels flat).**

### The kernel (the shipped reference)
rocKE's `build_wmma_fmha_fwd` (gfx1151 WMMA flash-attention forward), a thin adapter
over the unified `mfma_attention_fwd_inner_body`:

- **`head_size` (D) and head counts (H, H_kv) are compile-time; seqlen is runtime.** So
  **one binary per dtype** serves both LTX self-attn (S_q = S_kv = 4096) and cross-attn
  (S_q = 4096, S_kv = 128). D and H are *exact-match* constraints; S_q/S_kv only need to
  be tile multiples.
- **Native bf16, no cast.** bf16 shares the f16 16×16×16 WMMA fragment layout on
  gfx1151, so the same inner body lowers to `wmma.f32.16x16x16.bf16` for bf16. f16 and
  bf16 are separate `.co`s within the one family, selected by the `dtype` constraint.
- **`mask_mode="none"`, non-causal, MHA (H_kv == H).** The adapter declines anything
  else (see the decline boundary below).
- **Grid** `(ceil_div(S_q,16), H, B)`, **block** `(32,1,1)` — one wave32 per CTA, each
  CTA owns a 16-row Q tile of one (head, batch). **Static LDS** → `sharedMemBytes = 0`;
  **no workspace**.

This is a correctness-first reference (single wave per tile, no LDS K/V staging). It is
numerically correct at LTX shapes; on gfx1151 it also *beats* stock PyTorch SDPA, but
only because PyTorch has no fused flash backend there and falls to an unfused O(S²) math
path — so treat that as "not the bottleneck," not a win over a tuned flash kernel.
Tuning (LDS staging, multi-tile, larger Q tiles) grows data-only via §5.

### ABI (15 args, exact order)

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

#### ⚠️ Gotcha #1 — `scale_log2`, not the raw scale
The softmax is computed **base-2** (`exp2`), so the kernel takes
`scale_log2 = attn_scale * log2(e)` where `log2(e) = 1.4426950408889634`. The adapter
does this multiply (from `attn_scale_value`, defaulting to `1/sqrt(D)`); your kernel
must consume the already-multiplied value. A kernel that expects the *raw* scale is an
ABI change (§4).

#### ⚠️ Gotcha #2 — BHSD stride mapping, no batch-stride arg
Tensors are `[B,H,S,D]`: token stride is `stride(2)` (S axis), head stride is
`stride(1)` (H axis). There is **no batch-stride argument** — the kernel folds batch
into grid `z` assuming `batch_stride == seqlen * stride_token`. This holds trivially for
`B == 1` (LTX). The adapter declines `B > 1` unless that packed relation holds on every
tensor.

### Problem keys (`decode` emits)
| key | type | source |
|-----|------|--------|
| `dtype` | string | `"f16"` / `"bf16"` |
| `B` | int | `q.dim(0)` |
| `H` | int | `q.dim(1)` (query heads) |
| `H_kv` | int | `k.dim(1)` (kv heads) |
| `S_q` | int | `q.dim(2)` |
| `S_kv` | int | `k.dim(2)` |
| `D` | int | `q.dim(3)` (head dim) |
| `causal` | bool | always `false` (masked graphs decline) |

### Decline boundary
`SdpaAdapter::decode` returns "not applicable" (→ another engine serves it) for **any**
of: causal / causal-bottom-right / alibi / padding masks; additive `attn_mask`,
`block_mask`, or `sink_token`; dropout ≠ 0 (or any dropout plumbing); paged-KV; varlen /
group batch; LSE / stats output; FP8 (de)scale tensors; a **runtime** scale tensor;
non-rank-4 tensors; mismatched dtype across Q/K/V/O; `D`/`H_kv` mismatch; `S_q % 16 != 0`
or `S_kv % 16 != 0`; non-foldable `B > 1`. Handling any of these is a new decode
capability → C++ change (§4).

### Files
Adapter `ops/SdpaAdapter.{hpp,cpp}`; data + co-located producer
`library/gfx1151/fmha_wmma_fwd/{family.json, produce_fmha_fwd_co.py}`; parity test
`TestSdpaNumericParity.cpp`; A/B rig `tools/sdpa_aot_ab.py`; model override + driver
`tools/comfyui_hipdnn_sdpa_override.py`, `tools/ltx_sdpa_ab.py`.

---

## 9. How to test

All Python runs use the provider-compatible torch venv + WSL shim (see the
`reference_pytorch_hipdnn_env` note); do not perturb ComfyUI's own venv. Each op has the
same four rungs:

**1. Producer (build-time codegen)** — each rocKE-AOT family owns a co-located
`produce_<family>_co.py` that emits **every** kernel (all dtypes) the `family.json`
lists. ck_dsl is used as a *library* (no rocKE edit, except RMSNorm's/SDPA's small
upstreamable instances). The build runs it automatically when `ROCKE_PYTHON_DIR` is
set (points at `<rocKE>/projects/composablekernel/python`; comgr via `ROCKE_COMGR_LIB`
or `/opt/rocm`) — the `.co` are **built products, never checked into git**; the
family's `library/CMakeLists.txt` compiles them into the build/install tree
(`${AOT_CATALOG_BUILD_DIR}/<arch>/<family>/`). With `ROCKE_PYTHON_DIR` unset the family
is skipped (empty catalog → engine declines, parity tests skip). To run one by hand:
```
PYTHONPATH=<rocKE>/projects/composablekernel/python \
    python3 library/gfx1151/fmha_wmma_fwd/produce_fmha_fwd_co.py /tmp/out
# /tmp/out now holds the family's <symbol>.co (pair with the checked-in family.json)
```

**2. C++ substrate parity test** — drives the engine substrate directly and compares to
a CPU reference (`C=A@Bᵀ`, RMS over rows, or `softmax(scale·QKᵀ)·V`):
```
# configure with -DENABLE_AOT_CATALOG_ENGINE=ON, build hip_kernel_provider_tests
ctest -R GemmNumericParity      # or RmsNorm* / SdpaNumericParity
```

**3. A/B rig** — builds a single-node graph through the real frontend, hard-pins the AOT
engine by hashed id, checks `allclose` vs the torch reference and times both:
```
LD_LIBRARY_PATH=$HOME/aot-ab-venv/wsl-shim \
    $HOME/aot-ab-venv/bin/python tools/sdpa_aot_ab.py   # or gemm_aot_ab / rmsnorm_aot_ab
```

**4. Model E2E** — `tools/comfyui_hipdnn_<op>_override.py` monkeypatches the torch
functional (`F.linear` / `F.rms_norm` / `F.scaled_dot_product_attention`), routing
supported calls through the AOT graph with native fallback otherwise and an intercept
census. The `tools/ltx_<op>_ab.py` drivers run the real `LTXVModel` (random weights)
native vs override, verify output parity, and print the census + a CUDA-event
device-time breakdown.

---

## 10. Adding a brand-new op

For an op that isn't matmul/rmsnorm/sdpa, the pattern (mirror any existing adapter):

1. New `ops/<Op>Adapter.{hpp,cpp}` implementing `IOpAdapter`: `opKind()`,
   `decode(graph) → optional<ProblemShape>` (gate on the FlatBuffers attributes union
   discriminant; fail closed on unsupported features), `buildBindings(graph, problem,
   kernel) → LaunchBindings`, `gridSymbols(problem, kernel) → SymbolTable`.
2. One `push_back` in `CatalogEngine.cpp`.
3. One source line in this engine's `CMakeLists.txt`.
4. A family dir `library/<arch>/<family>/` + producer + parity test, following §9.

No changes to `LaunchAbi`, the `Catalog` loader, `Selection`, `CatalogTypes`,
`CatalogPlan`, or the other adapters are needed — the substrate is op-agnostic.

---

## 11. File map

| Thing | Path |
|-------|------|
| Engine + registration | `CatalogEngine.{hpp,cpp}` (one `push_back` per adapter) |
| Adapter interface | `ops/IOpAdapter.hpp` |
| Adapters | `ops/{Gemm,RmsNorm,Sdpa}Adapter.{hpp,cpp}` |
| Catalog loader / selection / launch / tuning | `catalog/`, `plans/`, `launch/` |
| Family library (discovery) | `library/CMakeLists.txt` (auto-discovers `<arch>/<family>/`) |
| Per-family unit (edit these) | `library/<arch>/<family>/{family.json, CMakeLists.txt, produce_<family>_co.py}` |
| Built `.co` (not in git) | `${AOT_CATALOG_BUILD_DIR}/<arch>/<family>/*.co` (emitted by the producer at build time) |
| Substrate parity/selection tests | `src/tests/engines/aot_catalog_engine/Test*.cpp` |
| A/B rigs | `tools/{gemm,rmsnorm,sdpa}_aot_ab.py` |
| Model overrides + LTX drivers | `tools/comfyui_hipdnn_*_override.py`, `tools/ltx_*_ab.py` |

(`tools/` and `src/tests/` paths are relative to the repo root; the rest are relative to
this engine directory. The `.co` kernel binaries are **churning rocKE build products**
and are compiled at build time, not vendored into git.)

---

## 12. Capabilities and limits (and the walls beyond gfx1151)

This section is the honest map of what the design does well and where it stops. §4 tells
you *when* a change is data vs C++; this tells you *how far the current shape carries* and
what a rocKE author will hit taking SDPA and conv to **gfx942 / gfx950 / gfx1250**. None
of the limits below are bugs — they are the deliberate edges of a thin bring-up engine.

### 12.1 Selection is "measure them all, cache the winner" — and nothing else

There is **no heuristic, no analytic cost model, and no shape-bucketed tuning database.**
Selection is exactly two steps: `constraints` prune the family to the candidates that
*can* run this problem (§3, §5), then `CatalogPlan` **times every survivor on the real
hardware** and caches the fastest, keyed on `family + canonical problem` (§5). The winner
is the measured winner — correct by construction, with no model that can be wrong, and the
kernel author ships candidates instead of hand-writing a selector. That is the core new
power (§12.2).

The cost of that simplicity is three structural scaling limits:

| # | Limit | Consequence | Bites hardest on |
|---|-------|-------------|------------------|
| 1 | **First-execute tax ∝ candidate count.** There is no pre-filter before timing — every candidate the constraints didn't prune is module-loaded and timed on the first execute of a shape. | Keep the per-problem candidate set to a handful (§5 says so for a reason). A family with a *large* flat kernel list (e.g. AITER's 290-entry `fmha_fwd.csv`) would time every survivor a shape leaves. | Large prebuilt/ASM families. |
| 2 | **The cache only amortizes when the problem-key space is small and repeated.** Every *new* key re-tunes from scratch. | Great for LLM decode and fixed model shapes (tune once, reuse forever). **Conv is the antithesis:** its key space `(N,C,H,W,K,R,S,stride,pad,dilation,dtype)` is effectively unbounded per model, so the cache thrashes and the tuning tax never amortizes. This is where "time them all" is *least* appropriate. | **Conv**, dynamic-shape workloads. |
| 3 | **`constraints` are the only pruning lever.** Rules are `equals` / `multiple_of` (§3). | Real selection intelligence lives entirely in how sharply the author writes constraints. Genuinely useful pruning that *isn't* a per-key equality/divisibility test — "pick the tile by M:N aspect ratio," "prefer split-K past this K" — cannot be expressed as a constraint at all. | Ops with many tile/algorithm variants. |

**Crossing this wall is a substrate change, not a data addition.** Making conv (or a
big ASM family) viable would need what we deliberately don't have: an analytic
pre-selector, or a shipped tuning DB keyed on shape buckets, to cut the candidate set
*before* timing. That is a departure from the measure-everything philosophy — plan for it
explicitly rather than discovering it as cache thrash.

### 12.2 What a rocKE author can do today with zero C++

Along the axes the design was built for, coverage grows as pure data (§4):

- **Ship N candidate kernels for one problem and get automatic best-pick** — no
  hand-written selector, no heuristic table to maintain (§5). This is the genuine new
  capability; every prior engine baked its own selection logic.
- **Add a dtype / tile / shape-tier variant as data** — bf16 GEMM and bf16 RMSNorm each
  landed with *zero* C++ change, as `kernels[]` entries carrying a `dtype` constraint
  (§3, §4). New WMMA tiles and static-N tiers are the same story.
- **Add a whole arch as a folder** — the C++ loads `.co` by arch string and never learns
  the arch name; drop `library/gfx950/<family>/` and the loader picks it up.
- **Mix build backends in one catalog** — rocKE-compiled and prebuilt-`.co` (AITER ASM)
  families coexist; the per-family `CMakeLists.txt` is the variation point.

The load-bearing precondition on *all* of that: **it is only data-free when the op already
has an adapter and the kernel fits that adapter's fixed ABI.** The rest of §12 is where
that precondition fails.

### 12.3 The ABI is per-adapter — and for SDPA it is hardcoded, not data-driven

`family.json` carries an `args_signature` (§3), and `LaunchAbi` packs by name from it —
but the adapter is still what *produces* those names/values, and it may ignore the
signature entirely. `SdpaAdapter::buildBindings` does `(void)kernel;` and marshals a
**fixed 15-arg BHSD signature** (§8): the SDPA ABI is baked into C++, not read from data.
Consequences:

- The first SDPA kernel on another arch that exposes a **different ABI** — a real
  batch-stride arg, a D-stride, a GQA-ratio arg, or the 16-byte SGPR-slot kernarg padding
  the AITER ASM kernels need — is **new adapter C++**, not new JSON. "Add an arch as a
  folder" (§12.2) holds for GEMM/RMSNorm, where the ABI is stable across shapes; it does
  **not** hold for SDPA across arches.
- **Computed args are an adapter escape hatch, per feature.** `scale_log2` and the stride
  args are derived in `buildBindings` today (§8); that pattern generalizes to *any*
  computed scalar — but each new one is a line of C++, not a data field. Byte-stride
  conversion, a `tuneOpt` launch scalar, a GQA ratio: all adapter code.

### 12.4 The walls for SDPA and conv on gfx942 / gfx950 / gfx1250

Grouped by what each actually requires — data, adapter C++, or a new substrate capability:

| Wall | What it needs | Class |
|------|---------------|-------|
| **New arch `.co` (CDNA MFMA, gfx1250)** | The C++ is arch-transparent, so the *catalog* side is data — **but** a producer must exist. gfx1151 producers `import ck_dsl.instances.gfx1151.*`; gfx942/gfx950 use MFMA (not WMMA) and need their own rocKE instances (different tile/occupancy), pinned to a churning `ck_dsl` API. | rocKE kernel work + data |
| **GQA / MQA** (`H_kv != H`) | `SdpaAdapter::decode` declines it today (§8); modern LLM attention needs it. Decode gate + an ABI that carries the KV-head mapping. | adapter C++ (+ maybe ABI) |
| **Causal / additive / padding masks** | Declined today (§8). This is the **76%-of-device-time prize** (causal SDPA). Decode capability + kernel bias/mask input + ABI arg. | adapter C++ + kernel |
| **D=128, any-H** | Nominally data (a new `D equals 128` variant), *iff* the kernel compiles without VGPR spill on the single-wave body — the one place D=128 may force a kernel change. | data (with a kernel caveat) |
| **varlen / paged-KV / FP8 descale** | All declined today (§8); needed for real serving. Each is a decode capability + ABI/kernel work. | adapter C++ + kernel |
| **SDPA backward** | Breaks the model outright: `CatalogPlan` assumes **one candidate = one module = one launch**. Backward is a 3-stage pipeline (odo → dqdkdv → dq_convert) with a shape-derived workspace and an `accumulator_type` knob that drives *both* selection and workspace size. | **new substrate capability** (multi-kernel plan) |
| **Heuristic grid transforms** | The grid DSL does `ceil_div`/constants (§3). ASM kernels want conditional transforms (mask-halving, hd192 axis-swap), a launch-time `tuneOpt` scalar, sub-arch selection (MI300 vs MI308), a uint32 stride gate. | adapter C++ |
| **Conv2d/Conv3d** | No `ConvAdapter` exists yet (a new op, §10). Even once written, measure-and-cache degrades on conv's shape space (§12.1 #2) — the selection model, not just the adapter, is the limiter. Largest lift of all. | new adapter **+** a selection strategy beyond measure-all |

**The one-line version.** This is a correct, data-extensible best-pick engine that excels
when the op's ABI is fixed and the shape space is small and repeated — GEMM, norms,
decode-shape attention. It extends for free along dtype / tile / arch-as-folder. It hits
real walls the moment a new arch's SDPA or conv needs (a) a richer or different ABI
(adapter C++, §12.3), (b) a multi-kernel plan such as SDPA backward (new substrate
capability), or (c) selection over an unbounded shape space where timing-everything stops
amortizing (conv, §12.1). Those are the deliberate edges — and they are essentially the
Phase 4/5 roadmap.
