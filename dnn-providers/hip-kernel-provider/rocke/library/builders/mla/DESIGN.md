# MLA kernel family — design doc

> **Status:** Design spike (no kernel code). DoD = this doc approved + GLM-5 geometry
> confirmed. No perf numbers or impl in scope.

Covers DeepSeek V2/V3/V3.1/R1, GLM-5, and Kimi-K2. The kernel family splits into
two distinct kernels (prefill and decode-absorb) and a separate fp8 phase for
gfx950+. All sections below are specifications; nothing is a tuning history.

---

## 0. Notation

| symbol | shape | meaning |
|---|---|---|
| $Q$ | $S_q \times H_q \times (d_{\text{nope}} + d_{\text{rope}})$ | query; split into nope and rope slices |
| $c_{KV}$ | $S_k \times r_{KV}$ | compressed KV latent (the KV cache payload) |
| $K_{\text{rope}}$ | $S_k \times d_{\text{rope}}$ | separately-stored RoPE key component |
| $W_{UK}$ | $r_{KV} \times (d_{\text{nope}} + d_V)$ | up-projection: latent → K\_nope ‖ V |
| $W_{UV}$ | $r_{KV} \times d_V$ | V slice of $W_{UK}$ (column partition) |
| $W_{UK,K}$ | $r_{KV} \times d_{\text{nope}}$ | K\_nope slice of $W_{UK}$ (column partition) |
| $W_{UQ}$ | $H_q \times r_Q \times (d_{\text{nope}} + d_{\text{rope}})$ | query up-projection |
| $W_{\text{abs}}$ | $H_q \times r_Q \times (d_{\text{nope}} + d_{\text{rope}})$ | absorbed weight (decode only) |
| $d_{\text{nope}}$ | 128 | content head dimension (qk\_nope) |
| $d_{\text{rope}}$ | 64 | RoPE head dimension (qk\_rope) |
| $d_V$ | 128 | value head dimension |
| $r_{KV}$ | 512 | KV lora rank |
| $r_Q$ | 1536 | query lora rank |
| $H_q$ | 128 (DS/GLM-5), 64 (Kimi-K2) | query heads |
| $H_k$ | 1 | KV heads (MLA always has Hk=1) |

---

## 1. MLA geometry and model variants

| Model | $H_q$ | $d_{\text{nope}}$ | $d_{\text{rope}}$ | $d_V$ | $r_{KV}$ | $r_Q$ |
|---|---|---|---|---|---|---|
| DeepSeek V2/V3/V3.1/R1 | 128 | 128 | 64 | 128 | 512 | 1536 |
| GLM-5 | **TBD — confirm-config** | 128 | 64 | 128 | 512 | 1536 |
| Kimi-K2 | 64 | 128 | 64 | 128 | 512 | 1536 |

> **Action:** Confirm GLM-5 $H_q$ with the model owner before closing this
> spike. The design assumes 128 (same as DeepSeek); if Kimi-K2's 64 applies, update
> the tiling tables accordingly. The kernel logic is unchanged for either.

---

## 2. Math and data layout

### 2.1 KV cache layout

The KV cache stores **only the compressed latent** and the **decoupled RoPE keys**:

```
KV cache per token:
  c_KV[r_KV]          bf16   512 elements — the latent that expands to K_nope and V
  K_rope[d_rope]      bf16    64 elements — RoPE-rotated key, stored post-rotation
```

Stored weights (offline, not in the token cache):

```
W_UK[r_KV, d_nope + d_V]   bf16   512 × 256 — KV up-projection
W_UQ[r_Q, d_nope + d_rope] bf16  1536 × 192 — Q up-projection (prefill only)
```

Unlike standard MHA/GQA, the KV head count is 1: all query heads share one
compressed KV latent per token position. The KV cache HBM footprint per token is
$(r_{KV} + d_{\text{rope}}) \times \text{bytes} = (512 + 64) \times 2 = 1152$ bytes (bf16).
Compare to standard GQA-8 at $H_k \times 2d \times 2 = 8 \times 256 \times 2 = 4096$ bytes:
MLA uses ~3.6× less KV cache bandwidth.

### 2.2 Full attention score for one query head $h$, position $i$

$$
s_j^{(h)} = \underbrace{q_{\text{nope},i}^{(h)} \cdot K_{\text{nope},j}^{\top}}_{\text{content score}}
           + \underbrace{q_{\text{rope},i}^{(h)} \cdot K_{\text{rope},j}^{\top}}_{\text{positional score}}
$$

where $K_{\text{nope},j} = c_{KV,j} \cdot W_{UK,K}^{\top}$ is the expanded content key and
$K_{\text{rope},j}$ is read directly from the KV cache.

The two score components can be summed element-wise before softmax, so the
effective attention head dimension is $d_{\text{nope}} + d_{\text{rope}} = 192$.

### 2.3 Prefill: latent expansion path

At prefill, $W_{UQ}$ is available and is applied online inside the kernel:

```
q_latent[r_Q] = x_q · W_DQ^T           # compressed query (input)
q[d_nope + d_rope] = q_latent · W_UQ^T # expanded query
q_nope = q[:d_nope]
q_rope = q[d_nope:]                     # RoPE applied here

# Per KV tile:
K_nope = c_KV · W_UK_K^T               # [tile, d_nope]
V      = c_KV · W_UV^T                  # [tile, d_V]
score  = scale * (q_nope · K_nope^T + q_rope · K_rope^T)
# → online softmax → weighted sum of V → output
```

The latent expansion `c_KV · W_UK^T` is the dominant compute: it is a
`[tile_k × r_KV] × [r_KV × (d_nope + d_V)]` GEMM inside the flash loop.

### 2.4 Decode: weight absorption path

At decode ($S_q = 1$ per head), weight absorption collapses the two-step KV
expansion into a single dot-product in latent space, making the decode kernel
**structurally identical to MQA with a single large head dimension**.

#### Absorbed weights — pre-computed at model load time

The absorbed weights are computed **once, at model load**, and stored as constant
GPU tensors for the lifetime of the serving session. The kernel never sees the
original $W_{UQ}$ or $W_{UK}$:

$$
W_{\text{abs}}^{(h)} = W_{UQ}^{(h)} \cdot W_{UK,K}^{\top}
\quad \in \mathbb{R}^{r_Q \times r_{KV}}
\qquad \text{(content score absorbed weight)}
$$

$$
W_{UQ,\text{rope}}^{(h)} \in \mathbb{R}^{r_Q \times d_{\text{rope}}}
\qquad \text{(RoPE projection, also pre-computed)}
$$

$$
W_{UV} \in \mathbb{R}^{r_{KV} \times d_V}
\qquad \text{(value up-projection, stored separately)}
$$

#### Decode as MQA in latent space

After absorption the query for head $h$ is projected into the latent basis:

$$
q_{\text{abs}}^{(h)} = c_q^{(h)} \cdot W_{\text{abs}}^{(h)}
\quad \in \mathbb{R}^{r_{KV}}
\qquad
q_{\text{rope}}^{(h)} = c_q^{(h)} \cdot W_{UQ,\text{rope}}^{(h)}
\quad \in \mathbb{R}^{d_{\text{rope}}}
$$

The combined score against token $j$ is then:

$$
s_j^{(h)} = \underbrace{q_{\text{abs}}^{(h)} \cdot c_{KV,j}^{\top}}_{\text{latent dot}} +
            \underbrace{q_{\text{rope}}^{(h)} \cdot K_{\text{rope},j}^{\top}}_{\text{RoPE dot}}
$$

This is a **single `head_dim = r_{KV} + d_{\text{rope}} = 512 + 64 = 576` MQA
attention** against the concatenated KV cache `[c_KV ‖ K_rope]`. The decode kernel
requires no KV expansion and no per-token weight application — it is bandwidth-bound
on reading the 576-element latent from HBM, exactly like standard GQA decode is
bandwidth-bound on K/V reads. This architectural equivalence is the source of the
bulk of the performance gain (AITER reports 17× over non-absorbed naive MLA).

The effective hot-path KV cache layout at decode:

```
[num_blocks, block_size, r_KV + d_rope]   # = [*, *, 576]  bf16
```

`c_KV` and `K_rope` are stored concatenated. No per-token expansion is needed; the
kernel reads 576 elements per token, computes the dot product directly, then uses
$W_{UV}$ to reconstruct the value contribution from $c_{KV}$.

The effective hot-path KV read per token is **$c_{KV}$ + $K_{\text{rope}}$** — same cache
layout as prefill, no extra storage needed.

### 2.5 Prefill strategy: absorption vs materialize crossover

For large prefill batches the cost per KV token is dominated by the latent expansion
GEMM (`c_KV · W_UK^T`). At small $S_q$ the per-token overhead of the expansion
amortizes poorly; at large $S_q$ it becomes cheap enough that materializing full
$K, V$ once and running standard flash attention is cheaper than streaming $W_{UK}$
through every CTA in a tiled flash loop.

The crossover point (empirically ~171–228 query tokens, measured by SGLang on Hopper
and approximately hardware-independent) defines two prefill regimes:

| Regime | $S_q$ threshold | Strategy |
|---|---|---|
| Short prefill / chunked prefill | $S_q \lesssim 200$ | Latent expansion inside flash loop (§3 kernel) |
| Long prefill | $S_q \gtrsim 200$ | Materialize $K, V$ once → standard flash attention |

The materialization path (`c_KV · W_UK^T → K, V`, then standard `mla_prefill_fwd`
→ pass expanded K/V directly) reuses existing `UnifiedAttention` kernels and incurs
a separate GEMM launch; the in-loop path fuses expansion with attention but is
memory-bandwidth-bound on W_UK at large tile counts. The dispatch heuristic
must implement this threshold — it is not a kernel property but a launch-time
decision in `library/dispatch/attention.py`.

### 2.6 Online softmax

The streaming softmax recurrence (identical to the existing unified attention) runs
per query row. The only difference from standard SDPA is that the score is the sum
of two inner products (§2.2) and the value is `c_KV · W_UV^T` (not stored directly).
The flash-attention building blocks from `common/attention_unified.py` apply
without change; what changes is the score formation and the V expansion inside the
tile loop.

---

## 3. Prefill kernel specification

**Op:** `mla_prefill_fwd` — compressed-KV + decoupled RoPE, causal mask, bf16.

### Inputs

| tensor | shape | layout | notes |
|---|---|---|---|
| `q_latent` | `[total_q, r_Q]` | row-major | compressed queries, packed varlen |
| `c_kv` | `[num_blocks, block_size, r_KV]` | paged | compressed KV latent |
| `k_rope` | `[num_blocks, block_size, d_rope]` | paged | RoPE keys (post-rotation) |
| `W_UQ` | `[H_q, r_Q, d_nope + d_rope]` | row-major | query up-projection weight |
| `W_UK` | `[r_KV, d_nope + d_V]` | row-major | KV up-projection weight |
| `cu_seqlens_q` | `[B+1]` | int32 | prefix sums of query lengths |
| `block_table` | `[B, max_blocks]` | int32 | paged KV block pointers |

### Outputs

| tensor | shape | notes |
|---|---|---|
| `out` | `[total_q, H_q, d_V]` | bf16 output |
| `softmax_lse` | `[total_q, H_q]` | fp32 log-sum-exp (for chunked-prefill reduce) |

### Kernel structure

```
grid: (H_k, total_num_q_blocks, 1)   # H_k = 1 for MLA
block: (64 * num_warps, 1, 1)

for each q_block:
  1. load q_latent tile → LDS
  2. apply W_UQ GEMM: q_latent [Bq, r_Q] × W_UQ [r_Q, d_nope+d_rope] → q [Bq, d_nope+d_rope]
  3. split q → q_nope [Bq, d_nope], q_rope [Bq, d_rope]
  for each KV tile:
    4. load c_KV tile [Bk, r_KV] from paged cache
    5. load K_rope tile [Bk, d_rope] from paged cache
    6. expand K_nope: c_KV × W_UK_K^T → [Bk, d_nope]
    7. expand V:      c_KV × W_UV^T   → [Bk, d_V]
    8. score = scale * (q_nope · K_nope^T + q_rope · K_rope^T)  # [Bq, Bk]
    9. apply causal mask
   10. online softmax update (m, l, o_acc)
  end
  11. normalize and write out [Bq, H_q, d_V]
```

Step 2 (W\_UQ GEMM) is the main structural addition over standard flash attention.
It can be fused as a pre-pass in LDS or as a separate small kernel depending on LDS
pressure; see §5 (tiling/LDS budget).

> **Alternative implementation path:** CK FMHA already supports
> `(hdim_q=192, hdim_v=128)` natively for gfx942 and gfx950 (see §11.4). If
> the W_UQ and W_UK expansions are done as separate prior GEMMs (latent → expanded
> Q and latent → expanded K/V), the resulting tensors can be fed directly into the
> CK `fmha_fwd` kernel at (192,128) without a custom rocKE kernel. This path should
> be prototyped and measured against the in-loop fusion approach before
> committing to a fully custom kernel. The CK path's known limitation: LSE output is
> disabled for this pair, blocking chunked-prefill if that is in scope.

---

## 4. Decode-absorb kernel specification

**Op:** `mla_decode_absorb_fwd` — weight-absorbed decode, $S_q = 1$ per head, bf16.

This is structurally **MQA with `head_dim = r_{KV} + d_{\text{rope}} = 576`**: after
pre-projecting the query latent with the absorbed weights (§2.4), the kernel runs
standard flash decode against the concatenated KV cache `[c_KV ‖ K_rope]`. The
absorbed weights `W_abs`, `W_rope_proj`, and `W_UV` are **constant GPU tensors
loaded once at model startup** — they are not computed per-request.

### Inputs

| tensor | shape | layout | notes |
|---|---|---|---|
| `c_q` | `[B, H_q, r_Q]` | BHSD | compressed query latent (one token per seq) |
| `kv_cache` | `[num_blocks, block_size, r_KV + d_rope]` | paged | `c_KV ‖ K_rope` concatenated; 576 elem/token |
| `W_abs` | `[H_q, r_Q, r_KV]` | per-head | **model-load-time constant**: $W_{UQ} \cdot W_{UK,K}^{\top}$ |
| `W_rope_proj` | `[H_q, r_Q, d_rope]` | per-head | **model-load-time constant**: RoPE slice of $W_{UQ}$ |
| `W_UV` | `[r_KV, d_V]` | row-major | **model-load-time constant**: value up-projection |
| `block_table` | `[B, max_blocks]` | int32 | paged KV pointers |
| `seqused_k` | `[B]` | int32 | KV sequence lengths |

### Outputs

| tensor | shape | notes |
|---|---|---|
| `out` | `[B, H_q, d_V]` | bf16 |

### Kernel structure (3D split-KV, analogous to `attention_tiled_3d`)

```
# Pre-step (host-side, once per request):
q_abs[B, H_q, r_KV]   = c_q · W_abs^T       # project query into latent space
q_rope[B, H_q, d_rope] = c_q · W_rope_proj^T # RoPE query projection

grid: (B * H_q, NUM_SEGMENTS, 1)

per segment CTA:
  1. load q_abs [r_KV] and q_rope [d_rope] for this (batch, head)
  for each KV tile in segment:
    2. load kv_cache tile [Bk, r_KV + d_rope]   # c_KV and K_rope as one contiguous read
    3. split → c_KV [Bk, r_KV], K_rope [Bk, d_rope]
    4. score = scale * (q_abs · c_KV^T + q_rope · K_rope^T)   # [1, Bk]
    5. online softmax update
    6. v_acc += softmax_weight * (c_KV · W_UV^T)               # [1, d_V]
  end
  7. write partial (m, l, v_acc) to segment workspace

reduce_segments kernel (same as attention_tiled_3d's reduce):
  combine partials → final bf16 output
```

The pre-step projections `c_q · W_abs^T` and `c_q · W_rope_proj^T` are small
GEMMs (`[B·H_q, r_Q] × [r_Q, r_KV]` and `[B·H_q, r_Q] × [r_Q, d_rope]`) and can
be fused into a single batched GEMM kernel or executed as a pre-kernel. They are
**not inside the flash loop** — they run once per request.

The absorbed weight `W_abs[H_q, r_Q, r_KV]` is `H_q × 1536 × 512` elements = 1536
MB at bf16 — too large to live in LDS. It is used only in the pre-step GEMM, not
streamed per KV tile. The main scheduling constraint for the decode flash loop is
`W_UV` staging (see §5).

---

## 5. Per-arch tiling and LDS budget

### 5.1 gfx942

**MFMA atom:** `mfma_f32_16x16x16_bf16` (narrow default); fp16 flash option
`mfma_f32_32x32x8_f16` is not relevant for MLA (bf16 first).

**LDS per CU:** 64 KB. At typical 2–4 WG/CU occupancy, effective LDS budget per WG
is **16–32 KB**.

#### Prefill — gfx942

The dominant LDS pressure is the W_UQ pre-GEMM in step 2. With `Bq = 16` and
$r_Q = 1536$: loading the full W_UQ row for one query block is $16 \times 1536 \times 2
= 48$ KB — too large for a single LDS tile. Options:

| Strategy | Description | LDS cost |
|---|---|---|
| **Split-r Q-GEMM** | Stream W_UQ in $r$-slices of 64–128; accumulate q in registers | $\sim$ 2–4 KB per slice |
| **Separate pre-kernel** | Launch a small GEMM (q\_latent → q) before the flash loop | 0 (separate kernel) |

**Recommendation:** Separate pre-kernel for W_UQ application on gfx942.
The pre-kernel is a standard GEMM (`[total_q, r_Q] × [r_Q, d_nope+d_rope]`) and
can reuse existing GEMM infrastructure. The flash kernel then receives expanded `q`
directly.

Remaining LDS budget for the flash loop (single-kernel, separate pre-kernel assumed):

| Buffer | Size (Bk=64, bf16) | Notes |
|---|---|---|
| c_KV tile | 64 × 512 × 2 = 64 KB | **exceeds budget** — must use Bk=16 or stream r_KV |
| K_nope (expanded) | 64 × 128 × 2 = 16 KB | |
| K_rope tile | 64 × 64 × 2 = 8 KB | |
| V tile (expanded) | 64 × 128 × 2 = 16 KB | |
| o_acc (fp32) | 16 × 128 × 4 = 8 KB | per-Bq row |

The c_KV tile at `Bk=64` is 64 KB — the entire LDS. The expansion W_UK itself
(`r_KV × (d_nope + d_V) = 512 × 256 × 2 = 256 KB`) cannot fit in LDS at all. The
latent expansion (steps 6–7) must be tiled in the $r_{KV}$ dimension:

**Proposed gfx942 prefill tiling:**
- `Bq = 16`, `Bk = 16` (one paged-KV block per KV tile iteration)
- Stream $r_{KV}$ in slices of 64 per MFMA tile (`r_KV_tile = 64`)
- LDS layout per iteration:
  - `c_KV_slice[16, 64]` = 2 KB
  - `W_UK_slice[64, 256]` = 32 KB (K_nope+V cols, r-slice)
  - `K_rope[16, 64]` = 2 KB
  - `K_nope_acc[16, 128]` = 4 KB (accumulator for latent expansion)
  - `V_acc[16, 128]` = 4 KB
  - **Total ≈ 44 KB** — fits within 64 KB LDS with 1 WG/CU, or at reduced tile with 2 WG/CU

> **Open question:** Whether the W_UK weight tile fits LDS alongside
> the c_KV slice determines whether the latent expansion can be fused into one kernel
> or requires a two-pass approach. Both paths should be prototyped and measured for
> occupancy; the design does not mandate one path.

**Occupancy estimate gfx942 prefill:** 1–2 WG/CU at the proposed tile. This is below
the 4 WG/CU of standard attention; the W_UK streaming cost is the bottleneck. The
3D split-KV path is not applicable to prefill (each CTA already has Sq > 1).

#### Decode-absorb — gfx942

The pre-step GEMMs (`c_q · W_abs^T`, `c_q · W_rope_proj^T`) run as separate small
kernels before the flash loop — `W_abs` is **not streamed per KV tile**. The flash
loop itself operates like standard single-head-dim=576 MQA decode.

Per-KV-tile LDS pressure in the flash loop:

| Buffer | Size (Bk=16, bf16) | Notes |
|---|---|---|
| `kv_cache tile [Bk, 576]` | 16 × 576 × 2 = 18 KB | c_KV + K_rope concatenated |
| `W_UV slice [r_KV_tile, d_V]` | see below | streamed in r_KV-tiles |
| `q_abs + q_rope` in regs | 576 × 4 = 2.25 KB | fp32 accumulators, register-resident |
| `v_acc` in regs | 128 × 4 = 0.5 KB | fp32, register-resident |

`W_UV[r_KV=512, d_V=128]` = 128 KB total — does not fit. Stream in `r_KV`-tiles
of 64: `W_UV_slice[64, 128]` = 16 KB per slice. Full KV tile loop:

```
for r_kv_slice in range(0, r_KV, 64):
  load W_UV_slice [64, 128] → LDS   (16 KB)
  load kv_cache tile [Bk, 576]      (18 KB)
  score  += q_abs[r_kv_slice:+64] · c_KV_tile[:, r_kv_slice:+64]^T
  v_acc  += softmax_w * (c_KV_tile[:, r_kv_slice:+64] · W_UV_slice)
LDS peak ≈ 34 KB — fits within 64 KB LDS at 1 WG/CU, or with reduced Bk at 2 WG/CU
```

**Recommendation:** Use `register_pv` pattern (eliminating `P_lds`) — with
LDS already under pressure from the kv_cache and W_UV tiles, keeping the softmax
probability in registers (same technique as `attention_tiled_2d_fastkv_regp.py`) is
particularly valuable here. This is a **priority**, not a nice-to-have.

**Occupancy estimate gfx942 decode:** 1–2 WG/CU. The 3D split-KV pattern
(analogous to `attention_tiled_3d`) is needed to expose KV parallelism across CTAs.

---

### 5.2 gfx950

**MFMA atoms:** `mfma_f32_16x16x32_bf16` (default wide-K) and
`mfma_f32_32x32x16_bf16` (combo, `ds_read_tr`-enabled). Reference:
`library/kernels/gfx950/attention_tiled_2d.py`, `_fastkv_regp.py`.

**LDS per CU:** 128 KB. At 2–4 WG/CU: **32–64 KB per WG**.

#### Prefill — gfx950

The wider K-step (32 per MFMA vs 16 on gfx942) amortizes the c_KV streaming cost
better. Proposed tiling:

| Parameter | Value | Notes |
|---|---|---|
| `Bq` | 32 | 32×32 MFMA M-tile |
| `Bk` | 32 | one paged-KV block (block_size=32) |
| `r_KV_tile` | 128 | K-step for latent expansion (2× gfx942) |
| `num_warps` | 4 | 256 threads |

LDS layout per iteration:
- `c_KV_slice[32, 128]` = 8 KB (latent slice, one r_KV-tile)
- `W_UK_slice[128, 256]` = 64 KB — **at limit for 32 KB/WG budget**

At `r_KV_tile = 64` (half):
- `W_UK_slice[64, 256]` = 32 KB
- `K_rope[32, 64]` = 4 KB
- `K_nope_acc[32, 128]` = 8 KB
- `V_acc[32, 128]` = 8 KB
- **Total ≈ 52 KB** — 2 WG/CU at 128 KB LDS

> **`ds_read_tr` layout recommendation (gfx950 prefill):** Store `W_UK` in transposed-dimension
> alignment (column-major in the `r_KV` axis) so that `ds_read_tr16_b64` can deliver
> the MFMA B-operand for the latent expansion step without a separate transpose.
> This is the same technique that makes the gfx950 `attention_tiled_2d` V-stage fast
> (see `platform/cpp/instances/gfx950/attention_tiled_2d_kv_body_pv_epilogue.cpp`).
> This layout **must** be evaluated — it is a recommended direction, not optional.

**Occupancy estimate gfx950 prefill:** 2 WG/CU baseline; potentially 3 WG/CU with
`waves_per_eu` lever if VGPR pressure allows.

#### Decode-absorb — gfx950

Same pre-step + flash-loop split as gfx942: `W_abs` is applied offline; the flash
loop is MQA with `head_dim=576`. The wider MFMA (32-wide K-step) and 128 KB LDS
allow a larger KV tile.

**Proposed gfx950 decode tiling:**
- `Bk = 32` (one paged-KV block at `block_size=32`)
- `W_UV_slice` streamed in `r_KV`-tiles of 128: `W_UV_slice[128, 128]` = 32 KB
- `kv_cache tile [32, 576]` = 36 KB
- `LDS peak ≈ 68 KB` — 2 WG/CU at 128 KB LDS

**`ds_read_tr` layout recommendation (gfx950 decode):** Store `W_UV` in transposed-dimension
alignment (column-major in the `r_KV` axis) so that `ds_read_tr16_b64` can deliver
the MFMA B-operand for the PV step without a separate transpose step. This is the
same optimization that makes `ds_read_tr` profitable for V staging in the existing
gfx950 `attention_tiled_2d` kernel (see §11 references). This layout **must** be
evaluated — it is a recommended direction, not optional.

**`register_pv` recommendation (gfx950 decode):** Apply the `_fastkv_regp` register-P
technique (eliminate `P_lds` by keeping the softmax probability in registers,
`gfx950/attention_tiled_2d_fastkv_regp.py`). With LDS under pressure from
`kv_cache` + `W_UV` tiles, removing `P_lds` is a **priority**.

**Occupancy estimate gfx950 decode:** 2 WG/CU baseline; `waves_per_eu` lever
applicable if VGPR pressure allows a third WG.

---

## 6. Dtype plan

| Phase | Dtype | Arch |
|---|---|---|
| 1 | bf16 | gfx942 + gfx950 (prefill and decode-absorb) |
| 2 | fp8 e4m3 (KV cache) | gfx950+ only (fp8 prefill and fp8 decode-absorb) |

**fp8 approach:** Follow the existing sync-dequant pattern from
`library/kernels/common/fmha_fwd_fp8.py` and the gfx950 `ALGORITHM.md` §7.3:
store `c_KV` and `K_rope` as fp8 e4m3 with per-block scale factors; dequant to
bf16 in LDS before the MFMA. The W_UK and W_abs weights remain bf16 (quantizing
weights adds accuracy risk without the cache-size benefit — KV cache bandwidth is
the bottleneck, not weights).

**fp8 is excluded from gfx942.** The gfx942 decoder does not have the
`ds_read_tr` transposition facility that makes fp8 dequant efficient on gfx950.

---

## 7. hipDNN exposure plan

> **Note:** The hipDNN heuristics/dispatch layer is still being stood up. This section
> captures the proposed integration contract; the final field names and AotCatalog
> packaging format must be confirmed with the hipDNN team before implementation starts.

### 7.1 New op identifier

MLA prefill and decode-absorb are distinct ops from `sdpa_fwd`. Proposed op strings:

| Op string | Kernel |
|---|---|
| `mla_prefill_fwd` | Prefill (compressed-KV + decoupled RoPE) |
| `mla_decode_absorb_fwd` | Decode-absorb (weight-absorbed, q=1) |

### 7.2 SdpaProblem extensions

New fields on `SdpaProblem` (see `library/api/src/dispatcher/SdpaProblem.hpp`):

```cpp
// MLA geometry (zero-default = standard SDPA, not MLA)
std::int64_t kvLoraRank    = 0;   // r_KV (e.g. 512)
std::int64_t qLoraRank     = 0;   // r_Q  (e.g. 1536; 0 = standard SDPA)
std::int64_t qkNopeDim     = 0;   // d_nope (e.g. 128)
std::int64_t qkRopeDim     = 0;   // d_rope (e.g. 64)
std::string  mlaMode       = "none";  // "none" | "prefill" | "decode_absorb"
```

The `headSize` field retains its existing meaning (`d_nope + d_rope = 192` for
DeepSeek) for backward compatibility with the existing catalog match logic;
MLA-specific fields are additive.

### 7.3 AttentionRequest extensions

New fields on `AttentionRequest` in `library/dispatch/attention.py`:

```python
kv_lora_rank : int = 0     # r_KV; 0 = standard SDPA
q_lora_rank  : int = 0     # r_Q;  0 = standard SDPA
qk_nope_dim  : int = 0     # d_nope
qk_rope_dim  : int = 0     # d_rope
mla_mode     : str = "none"  # "none" | "prefill" | "decode_absorb"
```

### 7.4 AotInstance / SelectionConstraints

MLA instances in the AOT catalog use the new op strings and the new attribute
constraints. `SelectionConstraints::satisfies()` must check `mlaMode` equality
(MLA and non-MLA instances must not match each other).

The `AotCatalog.loadDefault()` currently returns empty (Phase 1 kpack TODO in
`library/api/src/dispatcher/AotCatalog.cpp`). MLA kernels will slot into the same
packaging mechanism once it is stood up; no special handling needed.

### 7.5 Open questions regarding hipDNN integration

- [ ] Confirm preferred op string naming convention (`mla_prefill_fwd` vs
      `sdpa_fwd_mla_prefill` vs other).
- [ ] **Absorbed weights as graph inputs (blocking):** `W_abs`, `W_rope_proj`, and
      `W_UV` are **model-load-time constant GPU tensors** (computed once at startup,
      not per-request). They must be represented as persistent graph inputs in the
      hipDNN graph, not as AOT compilation constants — their values are known only
      after the model is loaded, not at kernel compilation time. Confirm how
      `SdpaGraphAdapter` will expose them: as additional weight-type `IGraph` inputs,
      as opaque constant handles, or another mechanism. This is a blocking question
      for the prefill and decode-absorb implementations.
- [ ] Confirm whether the two-kernel decode structure (pre-step GEMM + flash loop)
      is expressed as a single fused op in the graph or as two separate ops with an
      intermediate tensor. The pre-step (`c_q · W_abs^T`) is a batched GEMM that
      produces `q_abs[B, H_q, r_KV]` — its lifetime is one decode step.
- [ ] Confirm whether the prefill crossover dispatch (§2.5: in-loop vs materialize,
      threshold ~200 tokens) is handled inside the kernel op or by the framework
      choosing between two ops.
- [ ] Confirm whether chunked-prefill (`softmax_lse` output) is in scope for the
      initial integration target.
- [ ] Confirm AotCatalog kpack timeline relative to implementation start.

---

## 8. Test and bench plan

### 8.1 Correctness reference

A Python reference in `library/builders/mla/ref_mla_attn.py` implementing the expanded-form attention:

```python
def ref_mla_prefill(q_latent, c_kv, k_rope, W_UQ, W_UK, cu_seqlens, causal=True):
    q = q_latent @ W_UQ.T                     # [total_q, d_nope+d_rope]
    q_nope, q_rope = q.split([d_nope, d_rope], dim=-1)
    # expand all KV positions
    K_nope = c_kv @ W_UK[:, :d_nope].T        # [S, d_nope]
    V      = c_kv @ W_UK[:, d_nope:].T        # [S, d_V]
    scores = scale * (q_nope @ K_nope.T + q_rope @ k_rope.T)
    # causal mask, softmax, weighted V
    return softmax(scores, causal) @ V

def ref_mla_decode_absorb(c_q, c_kv, k_rope, W_abs, W_rope_proj, W_UV):
    q_rope = c_q @ W_rope_proj.T              # [B, H_q, d_rope]
    q_abs  = torch.einsum("bhr,hrk->bhk", c_q, W_abs)           # [B, H_q, r_KV]
    scores = scale * (torch.einsum("bhk,sk->bhs", q_abs, c_kv) + (q_rope @ k_rope.T))
    # softmax(scores) @ (c_kv @ W_UV.T)
    V = c_kv @ W_UV.T
    return softmax(scores) @ V
```

Tolerance gate: `max_abs ≤ 4e-2` bf16 (matching the existing unified attention gate).

### 8.2 Benchmark shapes

See `mla_shapes.json` (sibling files: `library/benchmarks/gfx942/attention/decode/`
and `gfx950/`). Decode sweep (`seqlen_q=1`): kv_len in {512, 1024, 2048, 4096, 8192,
16384, 32768} for DeepSeek V3/R1, GLM-5 (pending), Kimi-K2. Prefill sweep:
seqlen_q = seqlen_k ∈ {512, 1024, 2048, 4096}.

### 8.3 Parity baselines

The parity harness in `library/builders/mla/` follows the same three-table
methodology as `library/builders/gfx950/attention/README.md` (`auto` / kernel-path
A vs B). Two external baselines should be included for meaningful comparison:

| Baseline | Source | What to compare |
|---|---|---|
| **AITER Triton MLA** (`ROCM_AITER_TRITON_MLA`) | `aiter.ops.mla` | Primary comparison; AITER reports this as best on gfx942 |
| **TileLang MLA** | `github.com/tile-ai/tilelang` | Open-source; achieved 95% of AITER ASM on gfx942 in ~80 lines; transparent tiling strategy |

AITER's own assembly decode kernel (`ROCM_AITER_MLA`) is the performance ceiling;
TileLang is the most transparent public reference for understanding tiling choices
that drive gfx942/gfx950 MLA decode performance.

---

## 9. Implementation scoping

| Kernel | Arch | Dtype | Spec |
|---|---|---|---|
| MLA prefill | gfx942 | bf16 | §3, §5.1 prefill tiling |
| MLA prefill | gfx950 | bf16 | §3, §5.2 prefill tiling |
| MLA decode-absorb | gfx942 | bf16 | §4, §5.1 decode tiling |
| MLA decode-absorb | gfx950 | bf16 | §4, §5.2 decode tiling |
| MLA prefill | gfx950+ | fp8 e4m3 | §6 fp8 plan, §3 struct |
| MLA decode-absorb | gfx950+ | fp8 e4m3 | §6 fp8 plan, §4 struct |

Each bf16 kernel delivers: kernel impl + parity gate + bench run against
`mla_shapes.json`. The fp8 kernels additionally deliver the fp8 KV cache dequant path.

---

## 11. State of the art — public MLA kernel implementations

This section documents the public implementations reviewed during the design spike.
They inform the architectural decisions in §2–§5; nothing here is prescriptive for
rocKE implementation.

### 11.1 AITER (AMD Inference Toolkit)

The canonical ROCm MLA reference. Lives at `github.com/ROCm/aiter`; integrated into
vLLM and SGLang as the AMD attention backend.

**Decode kernel** (`mla_decode_fwd`):
- Hand-written assembly (not open-sourced in detail).
- Implements MQA in latent space (head_dim=576) exactly as described in §2.4.
- Absorbed weights (`w_kc = W_abs`, `w_vc = W_UV`) pre-computed at model load.
- Paged KV cache with split-KV scheduling across SMs.
- Reported **17× speedup** over non-absorbed naive MLA on gfx942.
- On gfx942: `ROCM_AITER_TRITON_MLA` slightly outperforms the ASM backend (~2–3%).
- On gfx950: ASM backend matches or beats Triton MLA.

**Prefill**:
- `ROCM_AITER_MLA`: dispatches to CK or ASM; ASM prefill limited to $S_q < 160$
  (chunked-prefill regime).
- `ROCM_AITER_TRITON_MLA`: uses Triton MHA path for prefill (standard flash with
  expanded head_dim=192 for nope+rope).

**FlyDSL** (MLIR-based Python DSL, `github.com/ROCm/FlyDSL`):
- Used in AITER for MoE and GEMM kernels (Kimi-K2.5 MoE fused kernel).
- A `fused qk_norm_rope_quant` kernel for DeepSeek-V4 decode was delivered via
  FlyDSL (Q2 2026 roadmap) — relevant to MLA prefill preprocessing.
- No complete MLA attention kernel in FlyDSL yet; listed as future work.
- Architecture: CuTe-style layout algebra, compiles to HSACO via MLIR Fly dialect.
  Comparable to rocKE (Python DSL → GPU binary) but through a different compiler
  stack (MLIR vs LLVM IR).

### 11.2 FlashMLA (DeepSeek)

`github.com/deepseek-ai/FlashMLA` — the reference MLA decode kernel from DeepSeek
(CUDA/Hopper only, not ported to ROCm). Architecturally relevant:

- Confirms the MQA-in-latent-space approach (§2.4): Q shape `[Sq, N, r_KV+d_rope]`,
  KV cache `[Skv, 1, r_KV+d_rope]` with paged block size 64.
- Uses TMA on Hopper for async KV load; AMD equivalent would be `raw_ptr_buffer_load_lds`
  with `async_buffer_load_lds_addr` (already in `attention_tiled_2d.py`).
- Split-KV across SMs with partial-result merging (same structure as
  `attention_tiled_3d`).
- Fuses KV buffer writes with decode attention (+12% from removing PyTorch overhead).
- Performance ceiling: 3000 GB/s memory-bound / 660 TFLOPS compute-bound on H800.

### 11.3 TileLang MLA

`github.com/tile-ai/tilelang` — open-source composable tiled DSL with a complete,
readable MLA kernel for gfx942.

- Achieves **95% of AITER assembly performance** on gfx942; **1.98× over Triton**
  MLA; **3.76× over PyTorch** baseline.
- ~80 lines of Python, fully open-source. The tiling strategy is the most
  transparent public reference for MLA on gfx942.
- Handles 64 KB shared memory (vs Hopper 228 KB) explicitly; tile sizes not
  constrained to multiples of 64; swizzling for bank conflicts handled automatically.
- Recommended as a **parity baseline** in addition to AITER Triton MLA (§8.3).

### 11.4 CK (Composable Kernels)

CK has no dedicated MLA kernel (no `mla`, `kv_lora_rank`, `absorbed` code paths),
but it contains relevant MLA-adjacent infrastructure added explicitly for DeepSeek
V3:

**`(hdim_q=192, hdim_v=128)` — officially supported:**
- Commit `4399ad79029` (March 2025): "support hdim=192/128 pair for deepseekv3".
- Available in all forward variants: fp16, bf16, fp8, prefill, pagedkv (paged-KV),
  splitkv. Registered in `dispatcher/codegen/fmha/fmha_arch_specs.json`.
- This is absorbed-form MLA: Q/K head_dim = qk_nope(128) + qk_rope(64) = 192,
  V head_dim = v_head_dim = 128. The CK FMHA kernel with these dims is a direct
  viable substrate for the MLA prefill kernel without fusing the latent
  expansion — the expansion is a separate prior GEMM and CK handles the attention.
- LSE output is disabled for this pair (`fmha_fwd.py` line 822: skip if lse=="t"),
  which blocks chunked-prefill use. This should be verified before relying on CK
  if chunked-prefill is in scope.

**`MLA_H128xH576_Asymmetric` test case — planned but not instantiated:**
- `tile_engine/ops/fmha/ck_fmha_testing_matrix.yaml` (added 2026-03-16) contains
  a test entry: `hdim_q=128, hdim_v=576`, seqlen_q=4096, described as "Multi-latent
  attention fusion; asymmetric Q/KV (128 vs 576)." The 576-dim V corresponds to the
  MQA-in-latent-space approach (r_KV=512 + d_rope=64 = 576 from §2.4).
- **No kernel instance exists** for this pair in `fmha_arch_specs.json`. It is a
  future target, not a usable kernel.

**Partial RoPE via `rotary_dim`:**
- `include/ck_tile/ops/fmha/block/block_rotary_embedding.hpp` supports a
  `rotary_dim` parameter that applies RoPE only to the first `rotary_dim` elements
  of the head vector, leaving the nope slice unrotated. This directly models the MLA
  RoPE pattern (RoPE on `d_rope=64` of a 192-dim head). Already wired into
  `fmha_fwd_appendkv_kernel.hpp` at runtime.

**Hard cap `hdim_q <= 256`:**
- All CK FMHA pipelines contain `static_assert(kSubQKHeaddim <= 256)`. CK cannot
  directly handle the Q latent dimension (r_Q=1536) without structural changes.

**Bottom line for rocKE:** CK's (192,128) FMHA is a ready-made substrate for MLA
prefill in the "separate expansion" mode. The prefill implementation should explicitly
prototype wrapping this CK kernel rather than building a fully custom kernel, and
measure against the in-loop fusion approach.

### 11.5 FlashInfer ROCm MLA

`github.com/ROCm/flashinfer` — ROCm port of FlashInfer. Decode via
`trtllm_batch_decode_with_kv_cache_mla`:

- KV cache layout: `[num_pages, page_size, r_KV + d_rope]` — confirms the
  concatenated layout described in §4.
- Decode kernel: 128-head MQA in latent space, reusing `c_KV` as both K and V for
  the score + value step (same as §2.4).
- CK backend for gfx942 prefill; supports gfx942 and gfx950.

### 11.6 SGLang weight absorption

`github.com/sgl-project/sglang` — the weight absorption technique (§2.4) was first
deployed at scale in SGLang (PR #905, #1138). Key implementation details:

- Absorbed weights stored as `w_kc` and `w_vc` at model load; the SGLang MLA
  module never calls `W_UQ` or `W_UK` during serving.
- Prefill crossover threshold ~171–228 tokens (§2.5): below threshold → Triton
  absorbed decode; above → materialize K/V → standard flash prefill.
- Open issue (#4615): avoid materializing `w_kc`/`w_vc` to save GPU memory — still
  open, relevant to the hipDNN graph tensor representation question in §7.5.

---

## 12. References

**Internal (rocKE codebase):**
- `library/builders/gfx950/attention/ALGORITHM.md` — unified attention on gfx950; template for this doc's structure
- `library/builders/gfx942/attention/ALGORITHM.md` — gfx942 narrow/flash math
- `library/builders/gfx1250/attention/gfx1250_universal_attention_plan.md` — phased plan analog
- `library/kernels/common/attention_unified.py` — `UnifiedAttentionProblem`, dispatch heuristics, flash building blocks
- `library/kernels/common/fmha_fwd_fp8.py` — sync-dequant fp8 pattern (gfx950+ fp8 kernels)
- `library/dispatch/attention.py` — `AttentionRequest`, `ATTENTION_REGISTRY` (Python dispatch layer to extend)
- `library/api/src/dispatcher/SdpaProblem.hpp` — C++ normalized problem struct (to extend for hipDNN)
- `library/kernels/gfx942/attention_tiled_2d.py`, `attention_tiled_3d.py` — arch baselines
- `library/kernels/gfx950/attention_tiled_2d.py`, `attention_tiled_2d_fastkv_regp.py` — gfx950 baselines; `register_pv` and `ds_read_tr` patterns
- `library/kernels/common/attention_arch.py` — arch gating (`_NARROW_TILED_2D_ARCHES`, `validate_tiled_attention_arch`)
- `platform/cpp/instances/gfx950/attention_tiled_2d_kv_body_pv_epilogue.cpp` — `ds_read_tr16_b64` usage in V staging

**External (state of the art — see §11):**
- AITER MLA: `github.com/ROCm/aiter` / `rocm.blogs.amd.com/software-tools-optimization/aiter-mla/`
- FlashMLA (DeepSeek, CUDA): `github.com/deepseek-ai/FlashMLA`
- TileLang MLA on gfx942: `github.com/tile-ai/tilelang` / `tilelang.com/deeplearning_operators/deepseek_mla.html`
- FlashInfer ROCm: `github.com/ROCm/flashinfer`
- SGLang weight absorption: PR #905, #1138 at `github.com/sgl-project/sglang`
- FlyDSL: `github.com/ROCm/FlyDSL`
- CK FMHA (192,128): `github.com/ROCm/composable_kernel` — commit `4399ad79029`; `include/ck_tile/ops/fmha/`, `dispatcher/codegen/fmha/fmha_arch_specs.json`, `tile_engine/ops/fmha/ck_fmha_testing_matrix.yaml`
