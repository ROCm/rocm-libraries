<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Forward convolution on the AOT catalog — KA enablement kit

This is the conv-specific companion to the master **[AOT catalog families — kernel-author
guide](../../README.md)** (read §1–§5 there first for the schema, fail-closed constraint
discipline, and measure-and-cache tuning). It documents the `conv_fprop` op that
`ConvFpropAdapter` handles, the runtime ABI the gfx1151 reference family launches with, and
**how a CDNA kernel-author (gfx942 / gfx950 / gfx1250) ships a forward-conv `.co` as pure
data — no C++ change.** The gfx1151 WMMA family in this folder is the reference the CDNA
implicit-GEMM efforts copy.

---

## 1. Why conv fits the catalog now (the runtime-generic model)

The master guide (§10) flags convolution as needing "a selection strategy beyond
measure-and-cache for its unbounded shape space." **This family is that strategy.** The key
move — the same one `rmsnorm2d_dynamic` / `layernorm2d_dynamic` make for the norm's `N` — is
that **all** conv geometry (`N, C, K, Hi, Wi, R, S, stride, pad, dilation`) is a **runtime
i32 argument**, and only the tile/perf config is baked into the `.co`. Consequences:

- **One shape-free `.co` per tile config serves ANY 2-D forward-conv shape.** Partial tiles
  at the implicit-GEMM `M` / `N_gemm` / `K_gemm` boundaries are **masked** (transform-DAG
  `valid` predicates + hardware buffer-OOB clamp + `m<M` / `n<N_gemm` store predication), not
  mis-addressed. A non-tile-aligned conv computes correctly.
- **The unbounded shape space no longer thrashes the tune cache.** Selection is shape-free:
  a handful of tile configs (this family ships 3 × 2 dtypes) are the *only* candidates for
  any shape, so measure-and-cache converges on a per-shape winner from a tiny fixed set
  instead of re-tuning a fresh key per `(N,C,H,W,K,R,S,…)` tuple.

So conv on the catalog is a `ConvFpropAdapter` (already merged, reviewed C++) **plus** this
data-only family model. Bringing conv to a new arch is now the same "produce a `.co`, write
`family.json`, copy the test" data exercise as SDPA (master guide §12).

---

## 2. The `conv_fprop` op

**op_kind `"conv_fprop"` · adapter `ConvFpropAdapter` · family `conv2d_fprop_gfx1151`
(f16 + bf16 kernels flat).**

### Semantics
2-D forward convolution as an **implicit GEMM**, cross-correlation (the DL "convolution"):

```
Y[n,ho,wo,k] = sum_{r,s,c}  X[n, ho*sH - pH + r*dH, wo*sW - pW + s*dW, c] * W[k,r,s,c]
```

with out-of-bounds input taps contributing 0 (zero padding). The implicit-GEMM extents are
`M = N*Ho*Wo`, `N_gemm = K/groups`, `K_gemm = R*S*(C/groups)`.

### Layout — channels-last packed only (NHWC / KRSC / NHWK)
The runtime kernel addresses **input NHWC, weight KRSC (== KYXC), output NHWK** — channels
at unit stride on every operand. `ConvFpropAdapter::decode` reads logical dims in canonical
NCHW order (`x=[N,C,Hi,Wi]`, `w=[K,C/groups,R,S]`, `y=[N,K,Ho,Wo]`) and **gates on
channels-last strides** (channel axis unit-stride, the exact strides the kernel addresses
with). Any other layout declines. Only **symmetric padding** is accepted (one pad per axis;
`pre_padding == post_padding`, else decline). `Ho`/`Wo` are verified against the conv
arithmetic — a graph whose declared output extent disagrees fails closed.

### The gfx1151 reference kernels
rocke `conv_implicit_gemm_dynamic.build_implicit_gemm_conv_dynamic` — a thin dynamic wrapper
over `build_implicit_gemm_conv`; all the pipeline / LDS / WMMA / epilogue machinery is reused
unchanged, only the descriptor addressing and the `M`/`N_gemm`/`K_gemm` bounds become runtime
SSA. gfx1151 WMMA subset: `wave_size=32`, warp grid 2×2 over the 16×16×16 WMMA atom
(→ block 128), `pipeline="mem"`, `epilogue="default"`, `groups=1`. This family ships 3 tile
configs (`64×64×64`, `64×64×32`, `128×64×32`) × {f16, bf16} = 6 shape-free `.co`.

> ⚠️ **`wave_size = 32` at compile time.** The WMMA atom is a wave32 primitive on gfx1151;
> the producer must set `wave_size=32`. (Unlike RMSNorm, the wave32 *cross-lane reduction*
> gotcha does **not** apply to conv — the WMMA accumulation path does not go through the
> Welford/shuffle helpers — but the atom width still must be 32.)

---

## 3. The runtime ABI (19 args, exact order)

The scalar geometry is **fully runtime**: the kernel derives `Ho, Wo, M, N_gemm, K_gemm` and
all tensor strides in-kernel from these i32 args. `ConvFpropAdapter::buildBindings` emits this
exact order; a family's `args_signature` must match it after the `A/B/D` pointers and the
`*_bytes` scalars (this is the ABI contract `DynamicConvGeometry.PARAM_ORDER` enforces on the
rocke side).

| #  | name      | type | meaning |
|----|-----------|------|---------|
| 0  | `A`       | ptr  | input, NHWC `[N,Hi,Wi,C]` |
| 1  | `B`       | ptr  | weight, KRSC `[K,R,S,C]` |
| 2  | `D`       | ptr  | output, NHWK `[N,Ho,Wo,K]` |
| 3  | `A_bytes` | i32  | input buffer size in bytes (hardware OOB clamp) |
| 4  | `B_bytes` | i32  | weight buffer size in bytes |
| 5  | `D_bytes` | i32  | output buffer size in bytes |
| 6  | `N`       | i32  | batch |
| 7  | `C`       | i32  | input channels |
| 8  | `K`       | i32  | output channels |
| 9  | `Hi`      | i32  | input height |
| 10 | `Wi`      | i32  | input width |
| 11 | `R`       | i32  | filter height (== Y) |
| 12 | `S`       | i32  | filter width (== X) |
| 13 | `sH`      | i32  | stride, height |
| 14 | `sW`      | i32  | stride, width |
| 15 | `pH`      | i32  | padding, height (symmetric) |
| 16 | `pW`      | i32  | padding, width (symmetric) |
| 17 | `dH`      | i32  | dilation, height |
| 18 | `dW`      | i32  | dilation, width |

Pointer uids: `A = x_tensor_uid`, `B = w_tensor_uid`, `D = y_tensor_uid`. The `*_bytes`
scalars back the buffer-resource OOB clamp; `decode` **declines** any tensor whose byte size
exceeds `INT32_MAX` (a larger tensor can't launch with a truncated i32 size).

### Grid
`grid.x = ceil_div(N_gemm, tile_n)`, `grid.y = ceil_div(M, tile_m)`, `z = 1`. The tile
literals are baked per `.co`; `N_gemm` and `M` come from `gridSymbols`.

---

## 4. Problem keys (`decode` publishes) — legal in `constraints` / `grid`

`ConvFpropAdapter::decode` publishes a **superset** of facts (SDPA-style fact-publishing), so
a CDNA kernel can gate on any of them without an adapter change. Grouped / split-K / MFMA
kernels opt in via their own constraints.

| key | type | source |
|-----|------|--------|
| `dtype` | string | `"f16"` / `"bf16"` |
| `N` `C` `K` | int | batch, input channels, output channels |
| `Hi` `Wi` | int | input spatial |
| `R` `S` | int | filter spatial |
| `Ho` `Wo` | int | output spatial (verified against conv arithmetic) |
| `sH` `sW` | int | stride |
| `pH` `pW` | int | padding (symmetric) |
| `dH` `dW` | int | dilation |
| `groups` | int | `C / (C/groups)` from the weight (1 = plain conv) |
| `conv_mode` | int | `ConvMode` enum (1 = CONVOLUTION, 2 = CROSS_CORRELATION) |
| `A_bytes` `B_bytes` `D_bytes` | int | i32-clamped buffer sizes |
| `M` | int | `N*Ho*Wo` (implicit-GEMM rows) |
| `N_gemm` | int | `K/groups` (implicit-GEMM cols) |
| `K_gemm` | int | `R*S*(C/groups)` (implicit-GEMM reduction) |

### What the gfx1151 family constrains
Minimal, per the runtime-generic model — **no shape `equals`, no spatial `multiple_of`**:

```json
"constraints": {
    "dtype":  { "equals": "f16" },
    "groups": { "equals": 1 },
    "C":      { "min": 1, "multiple_of": 8 },
    "K":      { "min": 1, "multiple_of": 8 }
}
```

`C`/`K` are `multiple_of` the vector width (vec = 8) for the channel-contiguous vectorized
loads; `min: 1` rejects a zero-extent channel that `multiple_of` alone would admit. That's
it — geometry is masked, not constrained.

> ⚠️ **Under-constraining miscomputes (master guide §3).** A key you leave unconstrained is a
> claim the kernel is correct for every value it can take. The gfx1151 kernel is groups==1,
> channels-last, symmetric-padding, f16/bf16 only — it constrains `dtype` and `groups`
> explicitly and relies on the adapter's **memory-safety** declines (layout, symmetric
> padding, `Ho/Wo` consistency, i32 buffer-size overflow) for the rest. A CDNA kernel with
> different capabilities must carry the matching constraints.

---

## 5. Shipping a CDNA forward-conv `.co` (zero C++ change)

The adapter is arch-neutral. To bring forward conv to gfx942 / gfx950 / gfx1250:

1. **Produce the `.co`.** Instantiate the forward implicit-GEMM conv from rocke's CDNA MFMA
   instances (not the gfx1151 WMMA subset — different tile/atom/occupancy) in a co-located
   `produce_<family>_co.py`, mirroring `produce_conv2d_fprop_co.py` here. Keep the kernel
   **fully dynamic** (geometry from the runtime args) so one `.co` per tile config still
   serves any shape. Pin the rocke ref you build against.
2. **Map the kernarg ABI to the vocabulary in §3.** If the kernel takes the same 19-arg ABI,
   you're done — `buildBindings` already emits it. A grouped or split-K kernel that needs a
   quantity the adapter doesn't emit yet (e.g. a per-group channel count, a split-K partial
   count) is **one added `bindings.scalars.emplace(...)`** in `ConvFpropAdapter::buildBindings`
   — reviewed C++, the single explicit extension point — after which it's data again.
3. **Author `family.json`** (schema in the master guide §3): `op_kind: "conv_fprop"`,
   `arch: "gfx942"`, one `kernels[]` entry per tile config with (a) `dtype`, (b) a `groups`
   constraint (`{equals: 1}` for plain conv, or `{min: 1}` / a specific value for a grouped
   kernel that reads `groups`), (c) `C`/`K` `multiple_of` your vec width, (d) the
   `ceil_div` grid, and (e) `args_signature` = your kernel's real ABI in order.
4. **Drop it under `aot_catalog/gfx942/<family>/`** with a producer `CMakeLists.txt`
   (`rocke_add_aot_family` + `rocke_add_aot_family_test`; mirror this folder's). It compiles
   only when `gfx942` is in the build's `GPU_TARGETS` (master guide §9).
5. **Copy the tests.** `cp` this folder's `TestConvNumericParity.cpp` (change `ARCH`,
   geometry, dtype token, and the hand-built bindings to match your `args_signature`) and
   `TestConvSelection.cpp` (host-only; change `ARCH` and the vec width). The CPU NHWC
   reference and tolerances carry over. Register them with
   `rocke_add_aot_family_test(ARCH gfx942 SOURCES …)` — tests live with the family.

### Grouped conv opt-in
The adapter publishes `groups`; the gfx1151 family pins it to 1. A CDNA grouped kernel simply
constrains `groups` differently (e.g. `{min: 1}` if it reads the runtime `groups` arg, or a
specific tier) — no C++ change. `C`/`K`/`K_gemm`/`N_gemm` already account for grouping in
`decode` (`N_gemm = K/groups`, `K_gemm = R*S*(C/groups)`).

---

## 6. Documented fast-follow — magic-division perf path

The one conv-specific runtime cost is **integer division in the coordinate unpack** (`m →
n,ho,wo` divides by `Ho*Wo`,`Wo`; `k → y,x,c` divides by `S*C`,`C`). v1 uses plain runtime
`sdiv`/`srem`, which is correct but slow in the inner load loop. The perf fast-follow is the
**host-precomputed magic-number path**: pass `(multiplier, shift)` pairs as extra scalar args
and use the SSA `do_magic_division` overload (`umul_hi_i32` already accepts SSA operands).

Because the ABI is **superset-friendly** (a kernel's `args_signature` names only the subset it
takes), this adds args **without an ABI break**: `buildBindings` emits the `(mult,shift)`
pairs, a magic-division kernel names them, the current kernels ignore them. This is the main
perf lever once the correct runtime path is proven on hardware — measure before assuming the
`sdiv`/`srem` path is the bottleneck.

---

## 7. Out of scope (adapter fail-closes; documented follow-ups)

- **wgrad / dgrad** — separate adapters (attrs union tags differ); a WMMA wgrad path exists
  in rocke for a fast follow.
- **3-D conv, NCHW layout, asymmetric padding** — the adapter fail-closes (declines) today.
- **Grouped conv on gfx1151** — the gfx1151 family is groups==1; grouped opt-in is already
  supported via the published `groups` key for a CDNA kernel that handles it.
- **Explicit im2col → GEMM** — two launches; declined until `CatalogPlan` gains a
  kernel-sequence substrate. Implicit-GEMM (one fused launch) is what fits the engine.

---

## 8. Files

Adapter `src/engines/aot_catalog_engine/ops/ConvFpropAdapter.{hpp,cpp}`; data + co-located
producer + per-family tests
`aot_catalog/gfx1151/conv2d_fprop/{family.json, produce_conv2d_fprop_co.py,
TestConvNumericParity.cpp, TestConvSelection.cpp, CMakeLists.txt}`; rocke runtime-generic
instance `rocke/platform/python/rocke/instances/common/conv_implicit_gemm_dynamic.py`.
Real-model E2E additionally needs the hipDNN→PyTorch injection layer (separate PR; master
guide §9).
