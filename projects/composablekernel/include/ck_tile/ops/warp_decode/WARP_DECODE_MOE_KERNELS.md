# MoE Warp-Decode Kernels — Reference & Reimplementation Guide

**Scope.** This document fully specifies the CK-Tile *warp-decode* Mixture-of-Experts
(MoE) MLP kernels as they currently stand: the algorithm, the exact GPU
instructions used for each part, the grid/block/warp/lane work decomposition, the
datatype conversions, the supported input datatypes and scale layouts, and the
tuning knobs (both used and not-yet-tried). It is intended to be detailed enough
to reimplement the kernels from scratch in another framework (e.g. FlyDSL,
Triton, raw HIP, or a different tile DSL) without reading the C++.

It also records what was **tried and dropped**, with the caveat that several of
those results are shape/regime-specific and not necessarily definitive.

**Source of truth.** Branch `users/samremes/ck/warp-decode` on
`git@github.com:ROCm/rocm-libraries` (the CK monorepo;
`projects/composablekernel`). Key files:

| File | Contents |
|---|---|
| `include/ck_tile/ops/warp_decode/kernel/warp_decode_gate_up_kernel.hpp` | `WarpDecodeGateUpKernel`, `WarpDecodeGateUpLdsXKernel` |
| `include/ck_tile/ops/warp_decode/kernel/warp_decode_down_reduce_kernel.hpp` | `WarpDecodeDownReduceKernel`, `WarpDecodeDownReduceLdsInterKernel` |
| `include/ck_tile/ops/warp_decode/kernel/warp_decode_numeric.hpp` | `WarpDecodeNumeric` — all dot/convert/reduce primitives |
| `include/ck_tile/ops/warp_decode/pipeline/warp_decode_problem.hpp` | `WarpDecodeGateUpProblem`, `WarpDecodeDownReduceProblem`, scale-layout tags |
| `include/ck_tile/ops/warp_decode/pipeline/warp_decode_policy.hpp` | `WarpDecodePolicy` — tile distributions |
| `include/ck_tile/ops/warp_decode.hpp` | `launch_warp_decode_gate_up`, `launch_warp_decode_down_reduce` |
| `test/ck_tile/warp_decode/test_warp_decode.cpp` | correctness tests (CPU reference) |
| `test/ck_tile/warp_decode/bench_warp_decode.cpp` | standalone C++ benchmark + all variant typedefs |

**Target hardware.** AMD Instinct MI350X / MI355X (**gfx950**, CDNA4), **wave64**,
256 CUs / 8 XCDs, HBM ≈ 5–8 TB/s, ~10 MB L2. The kernels assume `warp_size == 64`.
gfx942 (CDNA3) compiles but the FP4/FP8 packed-convert intrinsics are guarded by
`#if defined(__gfx950__)`.

---

## 1. Problem being solved

A decode-time MoE feed-forward block for one batch of `B` tokens. Per token, a
router has already chosen `TOPK` experts out of `E` and produced per-expert
weights. For each selected expert the MLP computes

```
inter[token, slot] = silu( gate_w[e] · x[token] ) * ( up_w[e] · x[token] )   # gate/up + activation
y[token]          += router_wt[token, slot] * ( down_w[e] · inter[token, slot] )   # down + top-k reduce
```

with

- `x`        : `[B, HIDDEN]`              activations
- `gate_w`,`up_w` : `[E, INTER, HIDDEN]`  (row = output neuron, contracted over HIDDEN)
- `down_w`   : `[E, HIDDEN, INTER]`       (row = output channel, contracted over INTER)
- `inter`    : `[B, TOPK, INTER]`         BF16 intermediate
- `y`        : `[B, HIDDEN]`              output
- routing    : `router_ids[B, TOPK]` (int32 expert id), `router_wts[B, TOPK]` (float)

**Decode regime / target.** Very small `M` (the number of tokens, `B` here):
**`B = 1` to `~4`**, i.e. single-token decode plus a few speculative/draft
tokens. This is the regime where a dense GEMM tiling is mostly wasted (an
`MFMA` 16-row tile is ≥75% empty), so the design uses **one wave per output
scalar** instead of a matrix-core tile. The kernels are correct and run for
larger `B` too, but the design choices (and the dispatch recommendations in §10)
are tuned for `B ≤ 4`. Above the `B ≈ 2–4` crossover, a fused MFMA single-launch
kernel (the AITER ASM path, out of scope here) starts to win.

**Reference shapes used throughout:**

| name | HIDDEN | INTER | TOPK | E |
|---|---:|---:|---:|---:|
| DeepSeek-V3 | 7168 | 2048 | 8 | 256 |
| MiniMax | 3072 | 1536 | 8 | 256 |
| Qwen3Next (TP=1) | 2048 | 512 | 10 | 512 |
| Qwen3Next-TP2 | 2048 | 256 | 10 | 512 |
| Qwen3Next-TP4 | 2048 | 128 | 10 | 512 |

---

## 2. High-level design: two-stage split, one wave per output scalar

The production design is **two separate kernel launches** with a small BF16
intermediate in HBM between them:

```
        gate_up_fused                       down_reduce
x ───────────────────────►  inter (BF16) ───────────────────────►  y
[B,HIDDEN]  fp8/bf16 W      [B,TOPK,INTER]   fp8/fp4 W            [B,HIDDEN]
```

Rationale (validated by profiling; see the companion optimization notes):

- The intermediate is **tiny** at decode `B` (e.g. DeepSeek `B=1`: `TOPK·INTER·2B
  = 32 KiB`), so the HBM round-trip between stages is negligible.
- Both kernels are **memory-bound on weight reads**. `gate_up` runs at ≈90% of
  achievable HBM, `down` at ≈74–79%. The only ≈2× lever is **fewer weight bytes**
  (fp4), not compute or cache tricks. Collapsing the two launches into one
  (persistent producer/consumer, or atomic-merge single-stage) was tried and
  loses below the `B≈4` crossover (see §11).

**Core decomposition principle (both kernels):** one **wave (64 lanes)** owns one
output scalar. The 64 lanes split the contraction dimension; a butterfly shuffle
reduces the 64 partial sums; **lane 0 writes the single output**. There is no
matrix core (`MFMA`) and no cross-wave reuse in the default path — each wave
streams the weights it needs and discards them. This maximizes the number of
independent waves (good occupancy / latency hiding) at tiny `M`.

The dot products use the **packed BF16 dot instruction `v_dot2_f32_bf16`** (2
MACs/lane/issue) rather than MFMA, because at `M ≤ 4` `dot2` matches MFMA's
useful throughput without the 15/16 tile-padding waste.

---

## 3. Supported datatypes & scale layouts

Datatypes are template parameters. `ComputeDataType` is always **`float`** (FP32
accumulation). `IntermediateDataType` is always **`bf16_t`**.

### gate_up (`WarpDecodeGateUpProblem`)

| operand | supported types | notes |
|---|---|---|
| `XDataType` (activation) | `bf16_t`, `fp8_t` | bf16 is the decode default; fp8 needs an activation scale |
| `WDataType` (gate/up weight) | `fp8_t` | `dot2` and `pkf` paths; **`pk_fp4_t` only on the slow scalar path** (no fast fp4 gate_up yet) |
| `IntermediateDataType` | `bf16_t` | written to HBM |
| compute | `float` | |

### down_reduce (`WarpDecodeDownReduceProblem`)

| operand | supported types | notes |
|---|---|---|
| `IntermediateDataType` (activation) | `bf16_t` | produced by gate_up |
| `WDataType` (down weight) | `fp8_t`, **`pk_fp4_t`** | fp4 fast path is implemented (see §6) |
| `YDataType` (output) | `bf16_t` | |
| compute | `float` | |

`pk_fp4_t` = OCP **MXFP4**: two FP4 (E2M1) values packed per byte, low nibble =
even index, high nibble = odd index. `INTER` FP4 elements occupy `INTER/2` bytes
per weight row; **`stride_w_down` is expressed in packed bytes (`INTER/2`)**.

### Scale layouts (`WarpDecodeScaleLayout`)

Both kernels accept independent scale layouts for activation (`XScaleLayout`) and
weights (`WScaleLayout`):

| tag | meaning | scale element type used in practice |
|---|---|---|
| `PerTensor` | one scalar for the whole tensor (read element 0) | `float` |
| `PerToken` | one scalar per row (per token for X, per weight-row for W) | `float` |
| `Block2D<Block_N, Block_K>` | 2-D blocked scale grid `[rows/Block_N, K/Block_K]` | `float` or `e8m0_t` |

Common instantiations:
- fp8 activations: `XScaleLayout = Block2D<1,128>` (per-128-along-HIDDEN per token).
- fp8 weights: `WScaleLayout = Block2D<128,128>`.
- MXFP4 down weights: `WScaleLayout = Block2D<1,32>` with `e8m0_t` scales (the MX
  microscaling block of 32).
- bf16 activations: `XScaleLayout = PerTensor` with scale `1.0` (i.e. unscaled).

`e8m0_t` is the OCP 8-bit power-of-two scale (shared exponent), the MX scale type.

---

## 4. Memory layout, strides, kernel arguments

All tensors are row-major. Strides are explicit args so callers can pass padded
rows.

**gate_up `Kargs`:** `p_x, p_x_scale, p_w_gate, p_w_gate_scale, p_w_up,
p_w_up_scale, p_router_ids, p_intermediate`, dims `b, hidden, inter, top_k, e`,
strides `stride_x, stride_w_gate, stride_w_up, stride_intermediate`.

- `x`: `[B, HIDDEN]`, row stride `stride_x ≥ HIDDEN`.
- `w_gate`,`w_up`: logically `[E, INTER, HIDDEN]` flattened to `[E*INTER, HIDDEN]`;
  row `w_row = e*INTER + neuron_j`, stride `stride_w_* ≥ HIDDEN`.
- `intermediate`: `[B, TOPK, INTER]` flattened to `[B*TOPK, INTER]`; row
  `token_b*TOPK + expert_k`, stride `stride_intermediate ≥ INTER`.

**down_reduce `Kargs`:** `p_intermediate, p_w_down, p_w_down_scale, p_router_ids,
p_router_wts, p_y`, dims `b, hidden, inter, top_k, e`, strides
`stride_intermediate, stride_w_down, stride_y`.

- `w_down`: logically `[E, HIDDEN, INTER]` flattened to `[E*HIDDEN, INTER]`; row
  `w_row = e*HIDDEN + out_j`. For fp8 `stride_w_down ≥ INTER`; for `pk_fp4_t`
  `stride_w_down ≥ INTER/2` (packed bytes).
- `y`: `[B, HIDDEN]`, stride `stride_y ≥ HIDDEN`.

**Alignment / divisibility requirements** (checked in `IsSupportedArgument`):
- gate_up: `HIDDEN % (warp_size * kVector) == 0`.
- down: `INTER % (warp_size * kVector) == 0` (full-wave path); plus per-variant
  constraints (e.g. the short-INTER subgroup path requires `INTER ==
  LanesPerOutput * kVector`).
- Block2D scales: the relevant dims must be divisible by `Block_N` / `Block_K`.

---

## 5. The `gate_up` kernel in detail

`WarpDecodeGateUpKernel::operator()`. Default config: 1 warp/block, 1 neuron/warp,
`kUseDot2 = true`.

### 5.1 Work assignment (grid → output)

```
GridSize  = B * TOPK * ceil(INTER / kNPerWarp)        # default kNPerWarp = 1
BlockSize = kWarpsPerBlock * 64                        # default 1 warp = 64 threads
```

Each **wave** computes one intermediate scalar `inter[token_b, expert_k,
neuron_j]`. With 1 warp/block, `block_id` decodes as:

```
neuron_j       = block_id % INTER
block_div      = block_id / INTER
expert_k       = block_div % TOPK
token_b        = block_div / TOPK
e              = router_ids[token_b*TOPK + expert_k]
w_row          = e*INTER + neuron_j         # row in w_gate / w_up
```

So at DeepSeek `B=1` the grid is `1*8*2048 = 16384` waves — plenty of parallelism
even at `M=1`. This is the key reason the warp-per-scalar design beats a dense
tiling at decode.

### 5.2 Contraction loop (over HIDDEN)

`kTileN = 64 * kVector`. The HIDDEN axis is walked in `num_iterations = HIDDEN /
kTileN` steps. In each step **lane `l` owns `kVector` contiguous K elements** at
`k_base = i*kTileN + l*kVector`. Three tiles are loaded per step:

- `x_tile`     : the `[token_b]` row of `x`, distributed so all lanes cover the
  `kTileN` slice (broadcast distribution, §8).
- `w_gate_tile`, `w_up_tile` : the `[w_row]` rows, output distribution (§8).

The loads use CK-Tile `make_tile_window` + `load_tile` with `kVector` as the
guaranteed vector width, which lowers to `global_load_dwordx{2,4}` (one 128-bit
transaction per lane when `kVector` bytes = 16).

### 5.3 Datatype conversion + dot (the hot inner loop)

The contraction operates on **BF16 pairs** packed into a `uint32_t` and uses
`v_dot2_f32_bf16` (FP32 accumulate of two BF16 products).

For **BF16 activations** (`XDataType == bf16_t`) the activation is already two
BF16 per `uint32_t`; only the **fp8 weight** is converted:

```
for ipair in [0, kVector/2):
    x_pair = x_tile.uint32(ipair)                         # 2 bf16 activations
    w_word = ipair/2; w_sel = ipair%2                     # 4 fp8 per uint32
    g_pair = fp8x2_to_bf16x2<w_sel>( w_gate_tile.uint32(w_word) )
    u_pair = fp8x2_to_bf16x2<w_sel>( w_up_tile.uint32(w_word) )
    gate_dot = dot2_bf16_packed_add(gate_dot, x_pair, g_pair)   # v_dot2_f32_bf16
    up_dot   = dot2_bf16_packed_add(up_dot,   x_pair, u_pair)
gate_acc += gate_dot * (x_scale * gate_scale)
up_acc   += up_dot   * (x_scale * up_scale)
```

For **FP8 activations** both `x` and `w` are fp8 and both are converted with
`fp8x2_to_bf16x2` before the dot.

Conversion primitive (`warp_decode_numeric.hpp`):
- `fp8x2_to_bf16x2<PairInWord>(uint32 fp8x4)` → `uint32` of two BF16. Lowers to
  **`__builtin_amdgcn_cvt_scalef32_pk_bf16_fp8`** (one instruction converts a
  packed fp8 pair to a bf16 pair; `PairInWord ∈ {0,1}` selects which half of the
  4-fp8 word).

Dot primitive:
- `dot2_bf16_packed_add(acc, a, b)` → `dot2_bf16_packed_raw`, which emits
  `v_dot2_f32_bf16 dst, a, b, dst` followed by **`s_nop 2`**. The `s_nop`
  covers the write→read latency of the accumulator across the dependent chain
  (the dot2 result feeds the next dot2's addend). See §7 for the s_nop-free
  variant used by the fp4 down path.

### 5.4 Scale handling

Three scale modes per operand, resolved with `if constexpr`:

- **PerTensor**: load once before the loop (`*p_scale`), reused every iteration.
- **PerToken**: load once per row (`p_scale[token_b]` for X; `p_scale[w_row]` for W).
- **Block2D<Block_N,Block_K>**: scale grid `[rows/Block_N, K/Block_K]`, indexed
  per K-block as `ptr[(row/Block_N)*(K/Block_K) + k_base/Block_K]`. This is read
  **inside** the loop because it varies along K.

**WD-OPT-18 LDS scale broadcast (landed for gate_up):** for Block2D scales, all
threads cooperatively pre-load the row's scale blocks (`≤ kMaxScaleBlocks = 128`)
into LDS (`x_scale_lds`, `w_gate_scale_lds`, `w_up_scale_lds`) with a
`block_sync_lds()`, then the inner loop reads `x_scale_lds[k_base/Block_K]` from
LDS instead of re-reading HBM. This removed 37–46% of VMEM read instructions for
gate_up. (It regressed the down half and is **not** used there.)

### 5.5 Reduction + epilogue

After the loop each lane holds a partial `gate_acc` / `up_acc`. Reduce across the
64 lanes with a **butterfly XOR shuffle** (`wavefront_reduce_sum`):

```
for stage in [0, log2(64)):              # 6 stages
    val += warp_shuffle(val, lane ^ (1<<stage))     # ds_bpermute / v_permlane
```

Then **lane 0 only**:

```
silu_gate = Silu(gate_acc)               # element_wise::Silu
out       = silu_gate * up_acc
intermediate[(token_b*TOPK + expert_k)*stride + neuron_j] = bf16(out)
```

`Silu(z) = z * sigmoid(z)`.

### 5.6 gate_up variants (all in the same file)

- **scalar default** (`kUseDot2=false, kUsePackedFp32=false`): a generic
  `sweep_tile_span` that converts each element with `type_convert` and does scalar
  FMA. Correct for all dtypes incl. `pk_fp4_t` (via `unpack_fp4_nibble` LUT) but
  VALU-bound. Used as the fp4 gate_up fallback and as the reference path.
- **dot2** (`kUseDot2=true`): the §5.3 path. **Recommended baseline.**
- **packed-FP32** (`kUsePackedFp32=true`): converts fp8→fp32 pairs
  (`fp8x2_to_f32x2` → `__builtin_amdgcn_cvt_pk_f32_fp8`) and uses
  `v_pk_fma_f32` (packed FP32 FMA) with a `horizontal_add` at the end. Same speed
  class as dot2 for gate_up; kept for fp8×fp8.
- **NPerWarp=2** (`kNPerWarp=2`): one wave computes **two adjacent neurons**
  (`neuron_j0, neuron_j1`) reusing the single loaded `x_pair` for both gate/up
  dots of both neurons (4 dots/iter). Grid divides INTER by 2. Activation reuse,
  but extra accumulators hurt scheduling and weights still dominate traffic — see
  §11.
- **LDS-X kernel** (`WarpDecodeGateUpLdsXKernel`): a multi-warp block (default 4
  warps) stages the shared `x` row into LDS with **double-buffered async copy**
  (`async_load_tile`, ping-pong `x_lds[2][...]`, `block_sync_lds_direct_load`),
  so the `kWarpsPerBlock` neurons in the block share one X load. Helps DeepSeek
  BF16 only; mixed elsewhere (§11). Constraints: `HIDDEN ≤ 8192`, `INTER %
  (kWarpsPerBlock*kNPerWarp) == 0`.

---

## 6. The `down_reduce` kernel in detail

`WarpDecodeDownReduceKernel::operator()`. The output `y[token_b, out_j]` sums over
**both** the contraction (`INTER`) and the **top-k** experts:

```
y[token_b, out_j] = Σ_k router_wt[k] · ( Σ_i inter[token_b,k,i] · down_w[e_k, out_j, i] · scale )
```

So each wave loops `k` over TOPK (accumulating into one `acc`) and within each `k`
loops over INTER. There are several layouts; the kernel picks one with
`if constexpr` on the Problem traits, in this order:

### 6.1 FP4 fast path, 1 output / wave  (`pk_fp4_t`, `kUseDot2`, `kHPerWarp==1`)

The current **fp8 best** is the H2 layout (§6.3); the **fp4 best is H2 too**
(§6.4). This 1-output fp4 path is the building block.

```
GridSize = B * HIDDEN          # (kHPerWarp = 1)
out_j    = block_id % HIDDEN
token_b  = block_id / HIDDEN
```

`kWordsPerLane = kVector / 8` (each `uint32` packs 8 FP4). Per `k` (expert) and
per INTER tile, lane `l` owns `kVector` FP4 at `k_elem = i*kTileN + l*kVector`:

```
# packed load: lane's FP4 chunk starts at k_elem/2 bytes, spans kWordsPerLane u32.
# Use memcpy (NOT reinterpret_cast) to avoid a strict-aliasing UB on the load.
w_bytes = (uint8*)w_base + (k_elem >> 1)
memcpy(w_words[0..kWordsPerLane], w_bytes, kWordsPerLane*4)

# 4 INDEPENDENT fp32 accumulators + s_nop-free v_dot2 to keep the pipe busy
dot0=dot1=dot2=dot3 = 0
for iw in [0, kWordsPerLane):
    ww = w_words[iw]; b = iw*8
    a0 = pack_bf16_pair(x[k_elem+b+0], x[k_elem+b+1])     # activation already bf16
    a1 = pack_bf16_pair(x[k_elem+b+2], x[k_elem+b+3])
    a2 = pack_bf16_pair(x[k_elem+b+4], x[k_elem+b+5])
    a3 = pack_bf16_pair(x[k_elem+b+6], x[k_elem+b+7])
    dot0 = dot2_nonop(dot0, a0, fp4x2_to_bf16x2<0>(ww))   # v_dot2_f32_bf16, NO s_nop
    dot1 = dot2_nonop(dot1, a1, fp4x2_to_bf16x2<1>(ww))
    dot2 = dot2_nonop(dot2, a2, fp4x2_to_bf16x2<2>(ww))
    dot3 = dot2_nonop(dot3, a3, fp4x2_to_bf16x2<3>(ww))
dot2_drain4(dot0,dot1,dot2,dot3)     # ONE s_nop 2 covering all 4 before the read
acc += ((dot0+dot1)+(dot2+dot3)) * (router_wt * scale)
```

Key instructions:
- `fp4x2_to_bf16x2<ByteSel>(uint32 fp4x8, scale=1)` → **`__builtin_amdgcn_cvt_scalef32_pk_bf16_fp4`** converts one byte (2 FP4) of the 8-FP4 word to a BF16 pair, optionally applying the MX scale during conversion. `ByteSel ∈ [0,4)`.
- The dot uses the **s_nop-free** `dot2_bf16_packed_raw_nonop` (no trailing
  `s_nop`). Correctness is preserved by having **4 independent accumulators** so
  the write→read hazard of any one is covered by the three others' issue latency,
  plus a single `dot2_drain4` (one `s_nop 2` tying all 4 accumulators) right
  before they are summed. This converts a serialized dependent dot2 chain into an
  ILP-friendly one and keeps the kernel bandwidth-bound rather than latency-bound.
  *(Caveat: dropping the drain is incorrect — verified — so at least one `s_nop`
  is required per accumulator group.)*

Why only `fp4→bf16` (and not `fp4→fp32` + `v_pk_fma_f32`): the activation stays
BF16, so the dot2 form needs **half** the conversion work of the pk_fma form,
which matters because the kernel is memory/latency-bound. (An `fp4x2_to_f32x2` +
`pk_fma_f32` variant exists in `warp_decode_numeric.hpp` and was measured;
dot2-nonop won.)

MX scale: for `Block2D<1,32>` with `block_k=32`, the scale is constant over a
lane's `kVector ∈ {8,16,32}` chunk, so it is applied **after** the dot
(`acc += dot * (wt * ds)`), not per element — this is why the e8m0 block scale is
cheap here.

### 6.2 FP8 path, 1 output / wave (the generic dot2 / pkf path)

For `WDataType == fp8_t`, the wave loads the BF16 `inter` tile and the fp8 weight
tile via CK-Tile windows, then:

- **dot2** (`kUseDot2`): convert fp8 weight pairs with `fp8x2_to_bf16x2`, dot with
  `dot2_bf16_packed_add` (with `s_nop 2`), `acc += dot*(wt*scale)`.
- **packed-FP32** (`kUsePackedFp32`): convert both to fp32 pairs, `v_pk_fma_f32`,
  `horizontal_add`.
- **scalar default**: `sweep_tile_span`, `unpack_fp4_nibble` for fp4, else
  `type_convert`.

### 6.3 FP8 H2 path — 2 outputs / wave (`kHPerWarp==2`, fp8, the current fp8 best)

```
GridSize  = B * ceil(HIDDEN/2)
out_j0    = (block_id % ceil(HIDDEN/2)) * 2 ;  out_j1 = out_j0+1
token_b   = block_id / ceil(HIDDEN/2)
```

One wave owns **two adjacent output rows**. It loads the shared `inter` tile
**once** and two weight tiles (`w_row0`, `w_row1`), runs two dot2 accumulators
(`dot0`, `dot1`), and writes both `y[out_j0]`, `y[out_j1]` from lane 0. The win is
**memory-level parallelism**: two independent weight loads in flight per wave plus
activation reuse. This recovers `down`'s HBM gap (≈74%→≈79% of peak at DeepSeek
`B=1`) and is the fastest fp8 `down` at `B=1/8/16/32`.

### 6.4 FP4 H2 path — 2 outputs / wave (`pk_fp4_t`, `kUseDot2`, `kHPerWarp==2`)

The fp4 analogue of §6.3 and the **current best fp4 `down`**. Each wave owns two
output rows, loads the BF16 activation row once, does two raw packed-FP4 loads
(`memcpy` per row), and runs **8 independent fp32 accumulators (4 per row)**
s_nop-free, with **one `dot2_drain4` per row** before summing:

```
d0r0..d3r0 = 0   # row 0
d0r1..d3r1 = 0   # row 1
for iw in [0, kWordsPerLane):
    w0=w_words0[iw]; w1=w_words1[iw]; b=iw*8
    a0..a3 = pack_bf16_pair(x[k_elem+b+...])           # shared activation
    d{0..3}r0 = dot2_nonop(d{0..3}r0, a{0..3}, fp4x2_to_bf16x2<{0..3}>(w0))
    d{0..3}r1 = dot2_nonop(d{0..3}r1, a{0..3}, fp4x2_to_bf16x2<{0..3}>(w1))
dot2_drain4(d0r0,d1r0,d2r0,d3r0); dot2_drain4(d0r1,d1r1,d2r1,d3r1)
acc0 += ((d0r0+d1r0)+(d2r0+d3r0)) * (wt*ds0)
acc1 += ((d0r1+d1r1)+(d2r1+d3r1)) * (wt*ds1)
```

**Measured (per-kernel C++ bench, gfx950, `down` ms, lower is better):**

| shape | B | fp8 best `down_h2_d2` | `down_fp4_h2` | fp4_h2 vs fp8 |
|---|---:|---:|---:|---:|
| DeepSeek-V3 | 1 | 0.0189 | 0.0212 | 0.89× (use `down_fp4_wide` at B=1) |
| DeepSeek-V3 | 2 | 0.0323 | **0.0266** | **1.21×** |
| DeepSeek-V3 | 4 | 0.0763 | **0.0515** | **1.48×** |
| DeepSeek-V3 | 8 | 0.1543 | **0.1267** | **1.22×** |
| DeepSeek-V3 | 32 | 0.6114 | **0.4714** | **1.30×** |
| Qwen3Next (INTER=512) | 4 | 0.0131 | **0.0086** | **1.52×** |
| Qwen3Next (INTER=512) | 8 | 0.0178 | **0.0155** | **1.15×** |

Resource usage (whole bench TU): **0 spills, 0 scratch, 8 waves/SIMD (max
occupancy)**, ≤65 VGPRs — the 8 accumulators are not a register problem. All four
fp4 scale layouts (per-tensor float, per-token float, `Block2D<1,32>` e8m0, on
both INTER=512 and 2048) × {1-output, H2} pass correctness vs a CPU reference.

### 6.5 FP4 "wide" 1-output (`kVector=32`)

`kVector=32` → 16 packed bytes/lane = a single 128-bit weight transaction, one
INTER iteration. Best fp4 `down` at **B=1** on long-INTER shapes (where H2's
extra outputs don't help because there's already enough parallelism). Requires
`INTER % (64*32) == 0` (e.g. INTER=2048). `down_fp4_h2_wide` (H2 + kVector=32)
was tried and is strictly dominated (single K-iteration kills the pipelining) —
**dropped**.

### 6.6 Short-INTER subgroup path (`kLanesPerOutput < 64`)

For very short INTER (Qwen 512/256/128) a wave can be split into `64/kLanesPerOutput`
logical workers, each computing a **different** output row while keeping 128-bit
(`kVector=16`) loads. Uses `subgroup_reduce_sum<LanesPerOutput>` (a partial
butterfly within each lane group). Requires `INTER == LanesPerOutput*kVector`,
`HPerWarp*LanesPerOutput == 64`, one warp, dot2, fp8. **Measured slower** than H2
on Qwen (it halves the wave count and Qwen down is latency-bound on a tiny
problem) — kept as an option, not the default. fp4 is not supported on this path.

### 6.7 LDS-intermediate down kernel (`WarpDecodeDownReduceLdsInterKernel`)

A multi-warp block stages the BF16 `inter` row into LDS (`kMaxInter=4096`) with
async copy, so `kWarpsPerBlock` output rows share it. Correct but **slower** than
the non-LDS dot2 at the 4-warp reuse factor (the down weights still dominate
traffic) — not a default (§11).

---

## 7. Instruction / primitive reference (`WarpDecodeNumeric`)

| primitive | emitted instruction(s) | purpose |
|---|---|---|
| `dot2_bf16_packed_raw(acc,a,b)` | `v_dot2_f32_bf16 d,a,b,d` + `s_nop 2` | dependent-chain BF16·BF16→FP32 dot of 2 elems; s_nop covers acc write→read |
| `dot2_bf16_packed_raw_nonop(acc,a,b)` | `v_dot2_f32_bf16 d,a,b,d` (no s_nop) | ILP dot; caller must cover the hazard with ≥2 independent accs + a drain |
| `dot2_drain4(a,b,c,d)` | `s_nop 2` (4 tied in/out operands) | one stall covering 4 independent accumulators before they're read |
| `dot2_bf16_packed_add(acc,a,b)` | wraps `..._raw` | typed convenience |
| `pack_bf16_pair(lo,hi)` | bit ops | pack two `bf16` into a `uint32` for `v_dot2` |
| `fp8x2_to_bf16x2<sel>(u32)` | `v_cvt_scalef32_pk_bf16_fp8` | 2×fp8 → 2×bf16 (sel picks half of the 4-fp8 word) |
| `fp4x2_to_bf16x2<bytesel>(u32,scale)` | `v_cvt_scalef32_pk_bf16_fp4` | 2×fp4 → 2×bf16, applies MX scale; bytesel∈[0,4) |
| `fp8x2_to_f32x2<sel>(u32)` | `v_cvt_pk_f32_fp8` | 2×fp8 → 2×f32 |
| `fp4x2_to_f32x2<bytesel>(u32,scale)` | `v_cvt_scalef32_pk_f32_fp4` | 2×fp4 → 2×f32, applies MX scale |
| `bf16x2_to_f32x2(u32)` | shifts | 2×bf16 → 2×f32 (for the pk_fma path) |
| `pk_fma_f32(acc,a,b)` | `v_pk_fma_f32` | packed FP32 FMA (2 lanes) |
| `horizontal_add(v2)` | add | reduce a `fp32x2` |
| `wavefront_reduce_sum(v)` | 6× `warp_shuffle` (XOR butterfly) | full 64-lane sum |
| `subgroup_reduce_sum<L>(v)` | `log2(L)` shuffles within lane groups | partial sum for the subgroup path |
| `unpack_fp4_nibble(byte,idx)` | LUT (`{0,±0.5,±1,±1.5,±2,±3,±4,±6}`) | scalar fp4 dequant (reference / slow path) |

**The s_nop hazard, precisely.** `v_dot2_f32_bf16` writes its FP32 result with a
fixed latency. If the *very next* instruction reads that register (the classic
`acc = dot2(acc, …)` dependent chain), the hardware needs `s_nop 2` to avoid a
stall/hazard. Two ways to handle it:
1. **Serialized**: `s_nop 2` after every dot2 (the `_raw` form). Simple, correct,
   but the chain is latency-bound.
2. **ILP**: keep ≥2 (here 4 or 8) independent accumulators so each dot2's result
   isn't read until several issues later, drop the per-op `s_nop`, and emit a
   single `s_nop 2` (`dot2_drain4`) right before the accumulators are read/summed.
   This is what the fp4 down paths use and is why they stay bandwidth-bound.

---

## 8. Tile distributions (`WarpDecodePolicy`)

CK-Tile encodes the lane/warp→data mapping as a `tile_distribution_encoding`.
Three are used:

- `MakeOutputTileDistribution`: P0 = `warp_id` → `WarpsPerBlock` output rows;
  P1 = (`lane_id`, `V`) → the K dimension split as 64 lanes × `kVector`. Used for
  weight tiles (each lane gets `kVector` contiguous K elements).
- `MakeXBroadcastTileDistribution`: replicate the single shared `x`/`inter` row
  across P0 (all warps), lanes still split K by `kVector`. Used for the activation
  that is shared by all output rows in a block.
- `MakeBlockCopyTileDistribution<CopyVector>`: all warps×lanes cover distinct
  vector segments — used for the cooperative LDS prefetch copies.

For a from-scratch reimplementation the takeaway is simpler than the encoding:
**lane `l` handles elements `[l*kVector, (l+1)*kVector)` of each `kTileN`-wide
slice; warps (when >1) handle adjacent output rows; the activation is broadcast.**

---

## 9. Tuning knobs

### 9.1 Knobs in use (template params on the Problem)

| knob | values | effect |
|---|---|---|
| `kVector` | 8 / 16 / 32 | per-lane elements = load width. 16 → one 128-bit fp8 transaction; 8 for fp4 fast path (needs `%8==0`); 32 = "wide" fp4 single transaction. Must divide the contraction tile. |
| `kUseDot2` | bool | use `v_dot2_f32_bf16` (recommended). |
| `kUsePackedFp32` | bool | use `v_pk_fma_f32` over fp32 pairs (fp8×fp8 alt). |
| `kHPerWarp` (down) | 1 / 2 | outputs per wave. **2 = current best** for both fp8 and fp4 at B≥2. |
| `kNPerWarp` (gate_up) | 1 / 2 | neurons per wave (activation reuse). Mixed results (§11). |
| `kWarpsPerBlock` | 1 / 4 | warps per block (only meaningful for the LDS staging kernels). |
| `kLanesPerOutput` (down) | 64 / 32 / 16 | subgroup split for very short INTER. |
| `XScaleLayout`/`WScaleLayout` | PerTensor / PerToken / Block2D | scale granularity. |
| `Activation` | `element_wise::Silu` | gate activation. |

### 9.2 Best-known configuration (this is the "best kernels" set)

- **gate_up:** `kUseDot2=true`, `kNPerWarp=1`, `kWarpsPerBlock=1`, `kVector=16`
  for aligned fp8 (`HIDDEN%1024==0`) else `kVector=8`. BF16 activations are the
  decode default (no host quant); fp8 activations use `Block2D<1,128>` + the
  LDS scale broadcast. This sits at ≈90% of achievable HBM — at the roofline.
- **down (fp8):** `kHPerWarp=2`, `kUseDot2=true`, `kVector=16`.
- **down (fp4 / MXFP4):** `kHPerWarp=2`, `kUseDot2=true`, `kVector=8`
  (`down_fp4_h2`) for **B≥2**; `kHPerWarp=1`, `kVector=32` (`down_fp4_wide`) for
  **B=1** on long-INTER shapes.

---

## 10. Dispatch recommendation (decode)

- **B = 1:** split, gate_up dot2 + down dot2-H2 (fp8) / down_fp4_wide (fp4 B=1).
- **B = 2–4:** split, gate_up dot2 + **down_fp4_h2** when fp4 weights are
  available (1.2–1.5× over fp8 down); else fp8 down-H2.
- **B ≥ 4 (Qwen-class):** a fused single-launch MFMA kernel (AITER ASM all-fused)
  starts to win on GPU time; out of scope for this doc but noted as the crossover.
- **Accuracy gating:** fp4 weights are an accuracy/product decision; gate behind a
  model-quality check before enabling MXFP4 weights in production.

---

## 11. Tried and dropped (with caveats)

These were measured and are **not** in the best set. Several are regime-specific;
treat "dropped" as "not a win at the reference shapes / decode B we tested," not
as a universal verdict.

| idea | outcome | caveat |
|---|---|---|
| **Persistent producer-owned single-stage (V2/V3/V4)** | 3.5–4× slower at B=1/4 | grid-barrier + ~256-CTA resident cap throttle parallelism; structural, unlikely to flip at decode |
| **ASM-fmoe single-launch atomic-merge** | loses B≤2, **wins B≥4** | not closed — it's the recommended B≥4 path; just not a warp-decode kernel |
| **LDS gate/up X-staging** | helps DeepSeek-BF16 only | mixed across shapes/dtypes; a single warp has little reuse to stage |
| **LDS down intermediate staging** | 1.1–1.45× slower | 4-warp reuse doesn't amortize the copy+barrier; **8-warp / double-buffered chunked retest is untried** |
| **NPerWarp=2 (gate_up)** | mixed / slight loss | halves occupancy; weights dominate so activation reuse is small. Diverges from the dense-GEMM result where it wins — possibly coupled to the (failed) swizzle and reg-alloc; worth a fresh look |
| **XCD / chiplet workgroup swizzle** | neutral on DeepSeek (large grid); regressed when packing | only plausibly useful on **small-grid Qwen INTER=512** (CU under-feeding); untested there |
| **Short-INTER wide/subgroup down (`down_short_d2`)** | 2.6–4.6× slower on Qwen | fewer lanes halves wave count; Qwen down is latency-bound on a ~10 MB problem |
| **down_fp4_h2_wide (kVector=32 + H2)** | strictly dominated | single K-iteration kills pipelining |
| **fp4→fp32 + v_pk_fma_f32 down** | slower than dot2-nonop | needs 2× the conversion work (activation is bf16) |
| **s_nop-free dot2 without a drain** | **incorrect** | at least one `s_nop`/drain per accumulator group is required |
| **MFMA-fp8 dots into the split kernels** | refuted | gate_up is at the memory roofline; compute offload can't cross it, and fewer/fatter MFMA waves drop below it |
| **Weight-reuse register tile (token-batched kMPerWarp)** | not feasible at decode | reuse ceiling ≈ B·TOPK/E ≈ 1.1× at ref shapes; the regime where it pays (large B / small E) is where MFMA-grouped GEMM wins anyway |
| **`slc=1` non-temporal weights, `__restrict__`** | neutral/regress | no measured benefit |

**Counter-measurement caveat:** the headline HBM-% numbers depend on a gfx950
counter correction — `FetchSize`/`TCC_MISS` undercount HBM by 2× (128 B L2 line
vs the 64 B the counters assume). Cross-check with the EA counter
(`TCC_EA*_RDREQ_DRAM_32B*32`) and the exact routing footprint, not `FetchSize`.

---

## 12. Remaining / untried optimization knobs

1. **FP4 gate_up (highest leverage, not yet built).** gate_up is at the HBM
   roofline reading fp8 weights; MXFP4 weights ≈halve its time — the single
   biggest remaining decode win. Apply the §6 recipe (raw packed loads +
   `cvt_scalef32_pk_bf16_fp4` + s_nop-free dot2, keep the consumer as warp-decode
   dot2, **no MFMA**). Currently fp4 gate_up only runs on the slow scalar path.
2. **B=1 fp4 down prefetch.** The 1-output fp4 path at B=1 long-INTER is
   ≈2.96 TB/s (MLP-bound). Software-pipelined / multi-buffered weight prefetch
   (borrow the ASM kernel's micro/macro pipelining) could approach the full 2×.
3. **Fuse input-side bf16→fp8 quant into gate_up** (when the fp8 path is selected)
   to drop the separate quant kernel (~2 µs at B=1).
4. **8-warp / double-buffered chunked LDS down** — the untried follow-up that
   would actually create the cross-wave reuse staging needs.
5. **Chunk-swept XCD swizzle on small-grid Qwen INTER=512** — the only swizzle
   regime not yet falsified.
6. **kVector autotuning** per shape/dtype (size to ~one 128-bit transaction;
   going wider only helps when issue-bound).
7. **Cooperative top-k / standalone topk producer** — deferred; topk is a small
   fixed ~3 µs.
8. **V5 single-stage** (per-CTA full-INTER + coarse atomic-merge + HIDDEN-half
   pipeline) — only build after a cost model shows it clears the CTA-count bar V4
   failed.

---

## 13. Build, test, benchmark

CK-Tile dev preset (gfx950), from `projects/composablekernel`:

```bash
# configure (dev preset) + build the two warp-decode targets
ninja -C build test_ck_tile_warp_decode bench_ck_tile_warp_decode

# correctness (CPU reference; all scale layouts incl. MXFP4)
./build/bin/test_ck_tile_warp_decode

# benchmark (ms / TFLOP/s / GB/s per kernel/shape/batch)
./build/bin/bench_ck_tile_warp_decode
# filters:
CK_WARP_DECODE_BENCH_SHAPES=deepseek-v3,qwen3next CK_WARP_DECODE_BENCH_BATCHES=1,2,4,8 \
  ./build/bin/bench_ck_tile_warp_decode
```

The bench prints variant rows like `down_fp4_h2`, `down_h2_d2`, `gate_fp8_d2`,
etc. Note the bench GB/s counts the activation re-read per output (cache-resident)
so it is a *logical* rate; the real HBM rate is the weight stream (compute it as
`weight_bytes / ms`).

---

## 14. From-scratch reimplementation checklist (e.g. FlyDSL)

1. **Two kernels**, BF16 intermediate in DRAM between them.
2. **gate_up**: grid `B*TOPK*INTER` waves; each wave = one neuron. Loop HIDDEN in
   `64*kVector` tiles, lane `l` owns `[l*kVector,(l+1)*kVector)`. Convert fp8 W
   (and fp8 X) to BF16 pairs with the packed scalef32 cvt; `v_dot2_f32_bf16`
   accumulate gate & up; apply `x_scale*w_scale` per K-block (broadcast Block2D
   scales via LDS); butterfly-reduce 64 lanes; lane 0 writes `silu(gate)*up`.
3. **down**: grid `B*ceil(HIDDEN/HPerWarp)` waves; each wave = one (HPerWarp=2:
   two) output channel(s). Loop TOPK then INTER. fp8 → `cvt_pk_bf16_fp8` + dot2;
   fp4 → raw packed 128-bit load + `cvt_scalef32_pk_bf16_fp4` + **s_nop-free dot2
   with 4/8 independent accumulators + one drain**. Apply MX block scale after the
   dot (block_k=32 ⊇ lane chunk). Accumulate `router_wt*scale`; butterfly-reduce;
   lane 0 writes `y`.
4. **Accumulate in FP32**; outputs/intermediate BF16.
5. **Scales**: support PerTensor / PerToken / Block2D (e8m0 for MXFP4).
6. **Hazard**: emulate the `v_dot2` write→read latency — either stall after each
   dependent dot, or keep independent accumulators and stall once before reading.
7. **Layouts**: weights row-major `[E*N, K]`; MXFP4 row = `K/2` packed bytes.
8. **Default config**: gate_up dot2 kVector=16 (fp8) / 8; down H2 dot2 (fp8
   kVector=16, fp4 kVector=8); B=1 long-INTER fp4 = wide kVector=32, 1 output.

---

*Companion strategy/profiling notes (local, not in this repo):
`docs/warp_decode_optimization_v3.md` and `docs/issues/warp_decode_profiling/`.*
