# The RDNA4 WMMA fragment ABI, from the lane map up

This directory is a small suite of **gfx1201 (RDNA4 / Navi 48) bring-up
examples**: a WMMA GEMM lane-map probe, a weight-only-quantized `MatMulNBits`
matmul, and a deep-fused conv + maxpool prototype. All three are built on a
single new primitive — the **gfx12 WMMA `16×16×16` f16 atom**
(`wmma_gfx12_f32_16x16x16_f16`) — and their shared purpose is to *prove that
primitive correct on silicon* and then exercise it through three real kernel
shapes.

This file is the specification: the math each kernel computes, the RDNA4
fragment layout that makes it different from RDNA3/3.5, and why a probe with
asymmetric inputs is the thing that pins the layout down. The
[`README.md`](README.md) is the runnable field guide — prerequisites, the file
map, exact commands, and arch notes.

> **New to the matrix unit on AMD GPUs?** CDNA parts (gfx9xx) issue `MFMA`
> instructions on wave64; RDNA parts (gfx11/gfx12) issue `WMMA` instructions on
> **wave32**. A WMMA instruction multiplies two `16×16` f16 fragments and
> accumulates into a `16×16` f32 result. The *only* thing that varies between
> RDNA generations — and the entire subject of this directory — is **how the 16
> lane-pairs of one wave32 hold the elements of those fragments**.

---

## 0. Notation

| symbol | meaning |
|---|---|
| `M, N, K` | GEMM dimensions; `C = A · Bᵀ`, with `A` row-major `M×K`, `B` row-major `N×K` (RCR) |
| lane `l` | one of the 32 lanes of a wave32, `l ∈ 0..31` |
| slot `i` | one of the per-lane fragment elements (A/B: 8 f16; C: 8 f32) |
| `m0, n0, k0` | the `(row, col, contraction)` origin of the `16×16×16` tile |
| WMMA | RDNA matrix instruction, computes `C += A · Bᵀ` over a `16×16×16` tile |

---

## 1. What the matrix unit computes (the contract)

The WMMA `16×16×16` instruction computes, for one `16×16` output tile,

```
C[m, n] += Σ_{k=0..15}  A[m, k] · B[n, k]          (i.e. C += A · Bᵀ)
```

with `A`, `B` in f16 and `C` accumulated in f32. **Note the transpose**: the
hardware contracts `A` and `B` along their *last* axis, so the natural layout is
`A` row-major `M×K` and `B` row-major `N×K` — the **RCR** convention all three
kernels here use. Larger GEMMs tile this `16×16×16` block over `M`, `N`, and a
`K`-loop; one wave32 owns one `16×16` output tile.

The contract above is fixed and generation-independent. What is *not* fixed is
the mapping from `(m, n, k)` to `(lane, slot)` — and that mapping changed in
RDNA4.

---

## 2. The RDNA4 fragment layout (what differs from RDNA3/3.5)

The gfx12 WMMA ABI differs from the gfx11 (RDNA3/3.5) one in three concrete
ways. The operand/accumulator lane maps (§2.1, §2.3) live in
`core/arch/target.py::_wmma_gfx12_*` and are walked by the GEMM operand loads
and epilogue in `instances/gfx1201/wmma_gemm.py`; the distinct intrinsic (§2.2)
is selected by op_id there but mapped to its builtin in `core/lower_hip.py`
(`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12`) and to the
`@llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16` intrinsic in
`core/lower_llvm.py`:

### 2.1 — No cross-half operand duplication

On gfx11, each lane carries a **`<16 × half>`** A/B fragment — the full 16 `K`
elements, *duplicated* across the two lane-halves. On gfx12 each lane carries a
**`<8 × half>`** fragment, and the 16 `K` elements of one WMMA are **split**
across the lane-halves:

```
lanes  0–15  carry  K = 0..7
lanes 16–31  carry  K = 8..15
```

So per `K`-step (stride 16) lane `l` loads **8 contiguous** elements starting at

```
K = k0 + (l // 16) · 8 .
```

This halves the per-lane operand register footprint relative to gfx11 — but it
means the operand load is a half-`K` gather keyed on `l // 16`, not a full-`K`
load.

### 2.2 — A distinct builtin / intrinsic

The instruction itself is a different opcode (`..._w32_gfx12`, lowering through
the `...v8f32.v8f16` intrinsic), selected by the op_id
`wmma_gfx12_f32_16x16x16_f16` rather than the gfx11
`wmma_f32_16x16x16_f16`. Picking the wrong op_id for the arch emits an
instruction the hardware does not implement the same way.

### 2.3 — A column-distributed accumulator

The f32 C-fragment is `<8 × float>` per lane (8 VGPR), and slot `i` of lane `l`
maps to output cell

```
( row = m0 + (l // 16) · 8 + i ,   col = n0 + l % 16 ) .
```

So a lane owns one **column** (`n0 + l%16`) and 8 consecutive **rows** of the
output tile, with the row block again selected by `l // 16`. The epilogue store
in `wmma_gemm.py` has to walk exactly this map to write `C` back, and the
`MatMulNBits` and conv-pool kernels inherit it.

---

## 3. Why an asymmetric-input probe is the proof (`wmma_probe.py`)

The lane maps in §2 are a **hypothesis** until they are run on a gfx1201 device
and checked against a reference. The subtle point is *what input distinguishes a
correct map from a transposed one.*

Consider a row↔col swap in the lane map: it transposes the result, computing
`A · B` where `Aᵀ · B` (or some permutation) was intended. If you verify with a
**symmetric** input — say `A = B` with identity-like or constant structure —
the transpose may be invisible: `C` and `Cᵀ` can coincide, so a wrong map still
passes. The probe therefore uses **random, asymmetric** `A` and `B`:

> With random asymmetric operands, `A · Bᵀ ≠ (A · Bᵀ)ᵀ` almost surely, so any
> row/col swap in the lane map shows up as a large `max_abs_diff` against the
> numpy reference `C = A @ B.T`. A **PASS at multiple tile counts**
> (`16×16×16`, `64×64×64`) uniquely confirms the mapping — an identity×constant
> input could not.

Because WMMA accumulates in f32 in a different order than numpy, the probe judges
the result **within a tolerance** (`--tol`, default `1e-2`), not bit-exact — a
real floating-point GEMM is correct, not identical. This is the foundation the
other two kernels stand on: the comment in `wmma_gemm.py` is explicit that the
`MatMulNBits` port should not be trusted until the probe matches.

---

## 4. `MatMulNBits` — weight-only-quantized matmul (`matmul_nbits_verify.py`)

`MatMulNBits` is the ONNX Runtime weight-only-quantized linear: the activation
`A` is f16, the weight `B` is **int4** (packed, with one f16/f32 **scale** per
`group_size` contraction elements), and the output `C` is f16. The math is the
ordinary GEMM of §1 with the weight dequantized on the fly:

```
B_deq[n, k] = (B_int4[n, k]) · scale[n, k // group_size]
C[m, n]     = Σ_k  A[m, k] · B_deq[n, k]
```

The kernel dequantizes `B` to f16 and feeds the gfx12 WMMA atom. This directory
covers the **full Qwen3.5-9B `MatMulNBits` set** with three families, selected by
`--family`, that differ in *output-tile shape and body*:

- **`large_n`** — the WMMA tiled body, for `N` a multiple of `tile_n = 128`
  (attention / FFN projections). Output tile `64×128`, `2×2` warps → block of
  128 threads.
- **`skinny_n`** — the same WMMA body with `tile_n = 32`, for the `N = 32`
  linear-attention `in_proj`. Output tile `64×32`, `2×1` warps → block of 64
  threads. (A narrow-N specialization rather than padding a wide-N tile.)
- **`decode_gemv`** — a **scalar** one-thread-per-output-column body for the
  `lm_head` GEMV (`N = 248320`, `M = 1`). It uses **no WMMA at all** (the atom
  list in the manifest is empty); `1×8` warps → block of 256 threads, one
  thread per output column. This family is arch-agnostic.

The two WMMA families share one source
(`instances/common/_matmul_nbits_large_n.py`) and **branch the fragment ABI per
arch** (the `_WmmaParams` table): the gfx1201 build emits a genuinely different
kernel — `<8 × half>` operands with the half-`K` split and the
column-distributed accumulator epilogue, versus gfx1151's `<16 × half>`
operands and row-distributed epilogue — selecting the `wmma_gfx12_..._f16`
op_id rather than `wmma_f32_16x16x16_f16`. The *driver script* is what differs
only in the manifest atom tag (`wmma_f32_16x16x16_f16` vs
`wmma_gfx12_f32_16x16x16_f16`); the shared body does the real per-arch work.

### 4.1 — The `--opt` combined-optimization body (`large_n` only)

`--opt` builds a combined-optimization `large_n` body that bundles three changes:

- **LDS-staged A** — stage the activation tile through LDS instead of
  re-reading it from global per `K`-step;
- **`tile_k = group_size` scale-on-accumulator** — set the `K`-tile to the
  quant group size (32) so the dequant scale is **constant across the whole
  `K`-tile**, letting the scale be applied once on the f32 accumulator rather
  than per element (the `_OPT_TILE` geometry: `64×128×32`, `2×2` warps);
- **LDS epilogue transpose** — reshape the WMMA C-distribution through LDS for a
  coalesced store.

`--opt` is only valid for `--family large_n` (the driver rejects it otherwise).

### 4.2 — Geometry preconditions

The WMMA families load full `tile_m` `A` rows in bounds (the store is
row-guarded), so they require:

- `K` a multiple of `tile_k`;
- `M` a multiple of `tile_m` (a partial-M tail is a follow-on, not implemented
  in this v1 body);
- `N` a multiple of `tile_n`.

The scalar `decode_gemv` body guards both `M` and `N` per element and only
requires `K` a multiple of `tile_k`.

---

## 5. Deep-fused conv + maxpool (`deep_fused_conv_pool_verify.py`)

This is the **same fused dataflow as the gfx950 prototype**, re-targeted to the
gfx12 WMMA atom on wave32. One kernel computes, with **no HBM round-trip for the
conv0 / conv1 intermediates**:

```
virtual-concat input
   → conv0 (3×3, pad 1, stride 1)  → ReLU         (implicit-GEMM, WMMA, staged through LDS)
   → conv1 (1×1)                    → ReLU
   → 2×2 stride-2 maxpool          → store to global
```

The two convolutions and the pool are fused into a single launch: conv0's output
stays in LDS, is consumed directly by the 1×1 conv1, and conv1's output is
maxpooled inline and only the **pooled** result is written to HBM. Both convs
run as **implicit GEMM** — the conv is reshaped into a matmul over the WMMA atom
— so this kernel is a third consumer of the §2 fragment layout.

The kernel *body* is the arch-parametric one in `instances/common`; the gfx1201
driver only pins the WMMA geometry (`warp_tile_m/n/k = atom.m/n/k`,
`wave_size`) and reuses the **gfx950 harness's numpy reference and verify/bench
helpers** verbatim, because those are spec-generic (they read only
`spec.problem`, `spec.block_size`, and the common grid).

### 5.1 — Tiling and shapes

A threadgroup owns a `pool_tile_h × pool_tile_w` block of **pooled** output. The
conv0/conv1 work tile it must compute is therefore the pre-pool region

```
conv_tile_h = pool_tile_h · pool_stride_h            (= pool_tile_h · 2)
conv_tile_w = pool_tile_w · pool_stride_w            (= pool_tile_w · 2)
tile_m      = conv_tile_h · conv_tile_w
```

The pool is fixed at `2×2` stride-2 (the spec validator rejects other pool
geometries). The output spatial dims follow from the conv0 output `Ho×Wo`:

```
pool_ho = (Ho - 2) // 2 + 1 ,   pool_wo = (Wo - 2) // 2 + 1 .
```

The kernel is marked **experimental** in its manifest.

---

## 6. Where the three examples meet

All three kernels are exercises of the *same* RDNA4 WMMA primitive at increasing
levels of composition:

| example | what it adds on top of the WMMA atom |
|---|---|
| `wmma_probe` | nothing — it **is** the atom, proving the §2 lane map on silicon |
| `matmul_nbits` | int4 weight dequant + three production tile shapes (incl. a non-WMMA scalar GEMV) |
| `deep_fused_conv_pool` | implicit-GEMM conv ×2 + ReLU + maxpool fused in one launch, no HBM intermediate |

The probe is the gate: the other two should only be trusted on gfx1201 once the
lane-map probe passes, because they all read and write through the layout it
verifies.

---

## 7. Where to go next

- [`README.md`](README.md) — prerequisites, the file map, the exact run
  commands (with the real CLI flags), and arch notes.
- `instances/gfx1201/wmma_gemm.py` — the WMMA GEMM and the encoded lane maps
  the probe checks.
- `instances/common/matmul_nbits.py` + `_matmul_nbits_large_n.py` — the shared
  `MatMulNBits` body and per-arch fragment branch. The `--opt` body is the
  separate `instances/common/_matmul_nbits_large_n_opt.py`; the scalar GEMV is
  `instances/common/_matmul_nbits_decode_gemv.py`.
- `instances/gfx1201/deep_fused_conv_pool.py` — the gfx1201 conv-pool spec.
