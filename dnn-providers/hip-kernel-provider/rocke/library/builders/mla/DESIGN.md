# MLA kernel family — design doc

> **Status:** Design spike (no kernel code). DoD = this doc approved + GLM-5 geometry
> confirmed. No implementation in scope, and **nothing here is measured**: the µs
> figures in §4 and the WG/CU figures in §5 are analytical estimates, labelled as such
> at each use.

Covers DeepSeek V2/V3/V3.1/R1, GLM-5, and Kimi-K2. The kernel family splits into
two distinct kernels (prefill and decode-absorb) and a separate fp8 phase for
gfx950+. All sections below are specifications; nothing is a tuning history.

---

## Contents

- [0. Notation](#0-notation)
- [1. MLA geometry and model variants](#1-mla-geometry-and-model-variants)
- [2. Math and data layout](#2-math-and-data-layout)
  - [2.1 KV cache layout](#21-kv-cache-layout)
  - [2.2 Full attention score for one query head $h$, position $i$](#22-full-attention-score-for-one-query-head-h-position-i)
  - [2.3 Prefill: latent expansion path](#23-prefill-latent-expansion-path)
  - [2.4 Decode: weight absorption path](#24-decode-weight-absorption-path)
  - [2.5 Prefill strategy: absorption vs materialize crossover](#25-prefill-strategy-absorption-vs-materialize-crossover)
  - [2.6 Online softmax](#26-online-softmax)
- [3. Prefill kernel specification](#3-prefill-kernel-specification)
- [4. Decode-absorb kernel specification](#4-decode-absorb-kernel-specification)
- [5. Per-arch tiling and LDS budget](#5-per-arch-tiling-and-lds-budget)
  - [5.1 gfx942](#51-gfx942)
  - [5.2 gfx950](#52-gfx950)
- [6. Dtype plan](#6-dtype-plan)
- [7. hipDNN exposure plan](#7-hipdnn-exposure-plan)
  - [7.1 New op identifier](#71-new-op-identifier)
  - [7.2 SdpaProblem extensions](#72-sdpaproblem-extensions)
  - [7.3 AttentionRequest extensions](#73-attentionrequest-extensions)
  - [7.4 AotInstance / SelectionConstraints](#74-aotinstance--selectionconstraints)
  - [7.5 Open questions regarding hipDNN integration](#75-open-questions-regarding-hipdnn-integration)
- [8. Test and bench plan](#8-test-and-bench-plan)
  - [8.1 Correctness reference](#81-correctness-reference)
  - [8.2 Benchmark shapes](#82-benchmark-shapes)
  - [8.3 Parity baselines](#83-parity-baselines)
- [9. Implementation scoping](#9-implementation-scoping)
- [10. State of the art — public MLA kernel implementations](#10-state-of-the-art--public-mla-kernel-implementations)
  - [10.1 AITER (AMD Inference Toolkit)](#101-aiter-amd-inference-toolkit)
  - [10.2 FlashMLA (DeepSeek)](#102-flashmla-deepseek)
  - [10.3 TileLang MLA](#103-tilelang-mla)
  - [10.4 CK (Composable Kernels)](#104-ck-composable-kernels)
  - [10.5 FlashInfer ROCm MLA](#105-flashinfer-rocm-mla)
  - [10.6 SGLang weight absorption](#106-sglang-weight-absorption)
- [11. References](#11-references)

---

## 0. Notation

| symbol | shape | meaning |
|---|---|---|
| $Q$ | $S_q \times H_q \times (d_{\text{nope}} + d_{\text{rope}})$ | query; split into nope and rope slices |
| $c_q$ | $S_q \times r_Q$ | compressed query latent, **shared across heads** (called `q_latent` in the prefill spec, `c_q` in the decode spec — one tensor, two names) |
| $c_{KV}$ | $S_k \times r_{KV}$ | compressed KV latent (the KV cache payload) |
| $K_{\text{rope}}$ | $S_k \times d_{\text{rope}}$ | separately-stored RoPE key component |
| $W_{UK}$ | $H_q \times r_{KV} \times (d_{\text{nope}} + d_V)$ | up-projection: latent → K\_nope ‖ V. **Per head** — DeepSeek's `kv_b_proj` is $[r_{KV},\, H_q(d_{\text{nope}}{+}d_V)]$; the latent is shared across heads, its expansion is not. 128 × 512 × 256 × 2 B = **32 MiB** total; the **256 KB per-head slice** is what the §5 tiling streams |
| $W_{UV}$ | $H_q \times r_{KV} \times d_V$ | V slice of $W_{UK}$ (column partition), per head. 16 MiB total, 128 KB per head |
| $W_{UK,K}$ | $H_q \times r_{KV} \times d_{\text{nope}}$ | K\_nope slice of $W_{UK}$ (column partition), per head. 16 MiB total, 128 KB per head |
| $W_{UQ}$ | $H_q \times r_Q \times (d_{\text{nope}} + d_{\text{rope}})$ | query up-projection |
| $W_{UQ,\text{nope}}$ | $H_q \times r_Q \times d_{\text{nope}}$ | nope slice of $W_{UQ}$ (column partition, per head: $W_{UQ}^{(h)}[:, :d_{\text{nope}}]$) |
| $W_{UQ,\text{rope}}$ | $H_q \times r_Q \times d_{\text{rope}}$ | RoPE slice of $W_{UQ}$ (per head: $W_{UQ}^{(h)}[:, d_{\text{nope}}:]$); named `W_rope_proj` in the kernel specs |
| $W_{\text{abs}}$ | $H_q \times r_Q \times r_{KV}$ | absorbed weight (decode only) — $W_{UQ,\text{nope}} \cdot W_{UK,K}^{\top}$, see §2.4 |
| $\text{scale}$ | scalar | softmax scale $= 1/\sqrt{d_{\text{nope}} + d_{\text{rope}}} = 1/\sqrt{192}$ — see the note below |
| $d_{\text{nope}}$ | 128 | content head dimension (qk\_nope) |
| $d_{\text{rope}}$ | 64 | RoPE head dimension (qk\_rope) |
| $d_V$ | 128 | value head dimension |
| $r_{KV}$ | 512 | KV lora rank |
| $r_Q$ | 1536 | query lora rank |
| $H_q$ | 128 (DS/GLM-5), 64 (Kimi-K2) | query heads |
| $H_k$ | 1 | KV heads (MLA always has Hk=1) |
| `Bq`, `Bk` | tuning knobs (§5) | kernel query-tile and KV-**tile** sizes; `Bk` is independent of the paged-cache `block_size` — see the note in §5.2 |
| `r_KV_tile` | 64–128 (§5) | $r_{KV}$ K-step for the latent expansion (prefill) and for `W_UV` streaming (decode epilogue) |
| `BLOCK_H` | sized at impl (§9) | query **heads** per workgroup — the MFMA M-tile. §3 (prefill) and §4 (decode) each carry one and they are **not the same value**: raising it in decode amortizes the shared KV read (§2.4), raising it in prefill multiplies in-loop `W_UK` traffic (§3). Read every use as scoped to its section |
| `NUM_SEGMENTS` | sized at impl (§9) | split-KV segments per sequence in the 3D decode path (§4); coupled to `BLOCK_H` through grid parallelism, not independent of it |
| `num_warps` | sized at impl (§9) | wave64 waves per workgroup; workgroup size is `64 * num_warps` threads |

> **Weight layout convention.** Every weight matrix in this doc is stored `[in, out]`
> (row-major). An up-projection is therefore a plain `x · W` with **no transpose**:
> `K_nope = c_KV · W_UK_K`, `q = c_q · W_UQ[h]`, `V = c_KV · W_UV`. A `^T` appears in
> this doc only where two *activation* tensors are contracted to form a score matrix
> (`q_nope · K_nope^T`), or in the one weight-times-weight product that defines
> $W_{\text{abs}}$ (§2.4), where the shared $d_{\text{nope}}$ axis is contracted.

> **Softmax scale.** `scale` is $1/\sqrt{d_{\text{nope}} + d_{\text{rope}}} = 1/\sqrt{192}$
> for **both** kernels. It does **not** become $1/\sqrt{576}$ in the decode-absorb kernel:
> weight absorption is an exact algebraic rewrite of the same 192-wide content score
> (§2.4), not a widening of the head dimension, so the scale must not follow the
> `head_dim = 576` framing used to describe the decode kernel's *memory* structure.
> This is an easy mistake to make, and the existing tooling makes it easy: the decode
> harness derives `scale = shape.head_size**-0.5`
> (`benchmarks/gfx*/attention/decode/benchmark_decode_live.py`), so an MLA shape
> labelled `head_size = 576` would silently get $1/\sqrt{576}$ — one more reason the
> MLA shape files carry no `head_size` key today (§8.2). `scale` must therefore
> be a host-supplied runtime scalar, never derived from `head_dim` by the kernel or the
> harness. A second, independent reason: DeepSeek-V3 with YaRN rope scaling folds an
> extra `mscale²` factor into `scale` at the model level, so no geometry-derived value
> is correct for that model even at the right head dimension.

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
W_UK[H_q, r_KV, d_nope + d_V]   bf16  128 × 512 × 256  = 32 MiB — KV up-projection (per head)
W_UQ[H_q, r_Q, d_nope + d_rope] bf16  128 × 1536 × 192 = 72 MiB — Q up-projection (per head, prefill only)
```

Unlike standard MHA/GQA, the KV head count is 1: all query heads share one
compressed KV latent per token position. The KV cache HBM footprint per token is
$(r_{KV} + d_{\text{rope}}) \times \text{bytes} = (512 + 64) \times 2 = 1152$ bytes (bf16).
Compare to standard GQA-8 at $H_k \times 2d \times 2 = 8 \times 256 \times 2 = 4096$ bytes:
MLA's KV cache footprint per token is ~3.6× smaller. This is a footprint claim, not a
roofline claim — where decode actually sits is §2.4.

### 2.2 Full attention score for one query head $h$, position $i$

$$
s_j^{(h)} = \text{scale} \cdot \Big(
             \underbrace{q_{\text{nope},i}^{(h)} \cdot K_{\text{nope},j}^{\top}}_{\text{content score}}
           + \underbrace{q_{\text{rope},i}^{(h)} \cdot K_{\text{rope},j}^{\top}}_{\text{positional score}}
            \Big)
$$

where $K_{\text{nope},j} = c_{KV,j} \cdot W_{UK,K}$ is the expanded content key (no
transpose — weights are stored `[in, out]`, see §0) and $K_{\text{rope},j}$ is read
directly from the KV cache. The `^T` on $K_{\text{nope},j}$ and $K_{\text{rope},j}$ is
the score-forming contraction between two activation tensors, and is correct.

The two score components can be summed element-wise before softmax, so the
effective attention head dimension is $d_{\text{nope}} + d_{\text{rope}} = 192$ and
$\text{scale} = 1/\sqrt{192}$ (§0).

### 2.3 Prefill: latent expansion path

At prefill, $W_{UQ}$ is available and is applied online inside the kernel:

```
c_q[r_Q]                = x_q · W_DQ        # compressed query: hidden state x_q down-projected
                                            #   by W_DQ [d_model, r_Q]. Produced upstream —
                                            #   neither x_q nor W_DQ is seen by this kernel.
q[h][d_nope + d_rope]   = c_q · W_UQ[h]     # expanded query, per head h
q_nope[h] = q[h][:d_nope]
q_rope[h] = q[h][d_nope:]                   # RoPE rotation applied here, at position i

# Per KV tile:
K_nope = c_KV · W_UK_K                      # [tile, d_nope]
V      = c_KV · W_UV                        # [tile, d_V]
score  = scale * (q_nope[h] · K_nope^T + q_rope[h] · K_rope^T)   # scale = 1/sqrt(192), §0
# → online softmax → weighted sum of V → output
```

All three weight applications above are `x · W` with no transpose (§0); the two `^T`
on the `score` line contract activations to form the `[Bq, Bk]` score matrix and are
correct.

The latent expansion `c_KV · W_UK` is the dominant compute: it is a
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
W_{\text{abs}}^{(h)} = W_{UQ,\text{nope}}^{(h)} \cdot W_{UK,K}^{\top}
\quad \in \mathbb{R}^{r_Q \times r_{KV}}
\qquad \text{(content score absorbed weight)}
$$

Only the **nope slice** of $W_{UQ}$ participates: $W_{UQ,\text{nope}}^{(h)} =
W_{UQ}^{(h)}[:, :d_{\text{nope}}] \in \mathbb{R}^{r_Q \times d_{\text{nope}}} =
\mathbb{R}^{1536 \times 128}$, contracted with $W_{UK,K}^{\top} \in
\mathbb{R}^{d_{\text{nope}} \times r_{KV}} = \mathbb{R}^{128 \times 512}$, giving
$\mathbb{R}^{1536 \times 512}$. (The `^T` on $W_{UK,K}$ **is** correct here, unlike the
up-projections in §2.2/§2.3 — this product contracts the shared $d_{\text{nope}}$ axis
of two weight matrices.) With the slice in place the absorption identity holds exactly:

$$
q_{\text{abs}}^{(h)} \cdot c_{KV}^{\top}
= c_q \, W_{UQ,\text{nope}}^{(h)} \, W_{UK,K}^{\top} \, c_{KV}^{\top}
= q_{\text{nope}}^{(h)} \cdot K_{\text{nope}}^{\top}
$$

$$
W_{UQ,\text{rope}}^{(h)} \in \mathbb{R}^{r_Q \times d_{\text{rope}}}
\qquad \text{(RoPE projection } W_{UQ}^{(h)}[:, d_{\text{nope}}:] \text{, also pre-computed;}
$$
$$
\text{named } \texttt{W\_rope\_proj} \text{ in §4)}
$$

$$
W_{UV} \in \mathbb{R}^{r_{KV} \times d_V}
\qquad \text{(value up-projection, stored separately)}
$$

#### Decode as MQA in latent space

After absorption the query for head $h$ is projected into the latent basis. Note that
$c_q \in \mathbb{R}^{r_Q}$ has **no head axis** — the per-head expansion happens
entirely through $W_{\text{abs}}^{(h)}$ and $W_{UQ,\text{rope}}^{(h)}$:

$$
q_{\text{abs}}^{(h)} = c_q \cdot W_{\text{abs}}^{(h)}
\quad \in \mathbb{R}^{r_{KV}}
\qquad
q_{\text{rope}}^{(h)} = \text{RoPE}_i\!\big(c_q \cdot W_{UQ,\text{rope}}^{(h)}\big)
\quad \in \mathbb{R}^{d_{\text{rope}}}
$$

$K_{\text{rope}}$ is stored **already rotated** (§2.1), so the query side must be
rotated at its current position $i$ before the RoPE dot below — the $\text{RoPE}_i$
above is not optional.

The combined score against token $j$ is then:

$$
s_j^{(h)} = \text{scale} \cdot \Big(
            \underbrace{q_{\text{abs}}^{(h)} \cdot c_{KV,j}^{\top}}_{\text{latent dot}} +
            \underbrace{q_{\text{rope}}^{(h)} \cdot K_{\text{rope},j}^{\top}}_{\text{RoPE dot}}
            \Big)
$$

This is a **single `head_dim = r_{KV} + d_{\text{rope}} = 512 + 64 = 576` MQA
attention** against the concatenated KV cache `[c_KV ‖ K_rope]`. The decode kernel
requires no KV expansion and no per-token weight application, which is the source of
the bulk of the performance gain (AITER reports 17× over non-absorbed naive MLA,
§10.1).

> **Where decode sits on the roofline is a function of `BLOCK_H`.** Because
> $H_k = 1$, the `BLOCK_H` heads resident in a workgroup consume the *same* 1152-byte
> KV token, so the arithmetic intensity is
> $\texttt{BLOCK\_H} \cdot (r_{KV} + d_{\text{rope}} + r_{KV}) \cdot 2 / 1152
> = \texttt{BLOCK\_H} \cdot 1.89$ flop/byte:
>
> | `BLOCK_H` | flop/byte | vs MI300X HBM balance (~247) | vs MI300X MALL balance (~77) |
> |---|---|---|---|
> | 1 | 1.9 | 0.8% | 2.5% — deeply bandwidth-bound |
> | 16 | 30 | 12% | 39% |
> | 32 | 60 | 25% | 79% |
> | 128 (every head) | 242 | 98% — the HBM knee | past the MALL knee |
>
> Compare ~16 flop/byte for GQA-8 decode: MLA's geometry *permits* an intensity ceiling
> ~15× GQA's. Both balances are ≈1307 TFLOPS bf16 divided by the relevant bandwidth:
> 5.3 TB/s HBM gives 247 flop/byte, ~17 TB/s MALL gives 77.
>
> **Which balance applies is unmeasured, and it changes the conclusion.** §4's estimate
> assumes the KV working set is MALL-resident, so the redundant reads head-batching
> removes cost MALL bandwidth (~17 TB/s), not HBM. Against MALL the kernel approaches
> the knee near `BLOCK_H = 32`; against HBM it never does. Both columns are given
> because there is no measurement to choose between them, and sizing `BLOCK_H` must
> settle this first — the answer decides whether `BLOCK_H` is bounded by traffic (raise
> it until the knee) or only by occupancy.
>
> Consequence for §4 and §5: the first-order cost of an M=1 tile is that it multiplies
> KV traffic by $H_q$, and that shrinks as $1/\texttt{BLOCK\_H}$. The MFMA-lane waste
> (M=1 discards 15/16 of a `16x16x16` atom) is second-order under the HBM column and
> co-dominant under the MALL one. Either way the head axis is a requirement; only its
> size is deferred.

> **`head_dim = 576` is a memory-layout statement, not a scale statement.**
> $\text{scale}$ stays $1/\sqrt{192}$ (§0): the identity above shows the latent dot
> *is* the 192-wide content score, just evaluated in a different basis. Using
> $1/\sqrt{576}$ here is a silent accuracy bug.

The effective hot-path KV cache layout at decode:

```
[num_blocks, block_size, r_KV + d_rope]   # = [*, *, 576]  bf16
```

`c_KV` and `K_rope` are stored concatenated. No per-token expansion is needed; the
kernel reads 576 elements per token, computes the dot product directly, accumulates
the softmax-weighted latent in $r_{KV}$ space, and applies $W_{UV}$ **once** in the
epilogue to reconstruct the value contribution (§4).

The effective hot-path KV read per token is **$c_{KV}$ + $K_{\text{rope}}$** — same cache
layout as prefill, no extra storage needed.

### 2.5 Prefill strategy: absorption vs materialize crossover

For large prefill batches the cost per KV token is dominated by the latent expansion
GEMM (`c_KV · W_UK`). At small $S_q$ the per-token overhead of the expansion
amortizes poorly; at large $S_q$ it becomes cheap enough that materializing full
$K, V$ once and running standard flash attention is cheaper than streaming $W_{UK}$
through every workgroup in a tiled flash loop.

The crossover point (empirically ~171–228 query tokens, measured by SGLang on Hopper
and approximately hardware-independent) defines two prefill regimes:

| Regime | $S_q$ threshold | Strategy |
|---|---|---|
| Short prefill / chunked prefill | $S_q \lesssim 200$ | Latent expansion inside flash loop (§3 kernel) |
| Long prefill | $S_q \gtrsim 200$ | Materialize $K, V$ once → standard flash attention |

The materialization path (`c_KV · W_UK → K, V`, then standard `mla_prefill_fwd`
→ pass expanded K/V directly) incurs a separate GEMM launch; the in-loop path fuses
expansion with attention but is
memory-bandwidth-bound on W_UK at large tile counts. The dispatch heuristic
must implement this threshold — it is not a kernel property but a launch-time
decision in `library/dispatch/attention.py`.

> **The materialize path does not "just reuse" the existing kernels.** MLA's
> materialized form is asymmetric — `hdim_q = d_nope + d_rope = 192`,
> `hdim_v = d_V = 128` — and the current rocKE attention stack supports neither.
> `library/dispatch/attention.py` rejects the request outright (`hdim_q == hdim_v` is
> required), `UnifiedAttentionProblem` carries a single `head_size`, and
> `supports_tiled_2d` / `supports_tiled_3d` on both arches gate
> `head_size in {64, 128, 256}`, which excludes 192. Reusing `UnifiedAttention` here
> therefore needs (a) `head_size` split into `hdim_q` / `hdim_v` through the problem,
> spec and descriptor layers and (b) a widened `head_size` gate — dual-engine work
> (§9). The CK-FMHA `(192,128)` alternative in §3 sidesteps that gap for *measurement*
> only — it yields no shippable rocKE instance — so that prototype should come first
> but does not remove the dual-engine work.

### 2.6 Online softmax

The streaming softmax recurrence (identical to the existing unified attention) runs
per query row. The only difference from standard SDPA is that the score is the sum
of two inner products (§2.2) and the value is `c_KV · W_UV` (not stored directly).
The flash-attention building blocks from `common/attention_unified.py` apply
without change; what changes is the score formation and the V expansion.

The two kernels place that V expansion differently:

- **Prefill (§3)** expands `V = c_KV · W_UV` per KV tile, inside the loop, because
  the expanded `K_nope` is needed there anyway.
- **Decode-absorb (§4)** accumulates in latent space (`acc += p · c_KV`) and applies
  `W_UV` once in the epilogue. The rescaling step of the online-softmax recurrence is
  a scalar multiply, so it commutes with the linear map: `(α·acc) · W_UV = α·(acc · W_UV)`.
  The recurrence is therefore unchanged; only the accumulator's width changes (`r_KV`
  instead of `d_V` — 4× more accumulator registers per query row, see §5.1/§5.2).

---

## 3. Prefill kernel specification

**Op:** `mla_prefill_fwd` — compressed-KV + decoupled RoPE, causal mask, bf16.

#### Inputs

| tensor | shape | layout | notes |
|---|---|---|---|
| `q_latent` | `[total_q, r_Q]` | row-major | compressed queries, packed varlen |
| `c_kv` | `[num_blocks, block_size, r_KV]` | paged | compressed KV latent |
| `k_rope` | `[num_blocks, block_size, d_rope]` | paged | RoPE keys (post-rotation) |
| `W_UQ` | `[H_q, r_Q, d_nope + d_rope]` | row-major | query up-projection weight, per head |
| `W_UK` | `[H_q, r_KV, d_nope + d_V]` | head-major | KV up-projection weight, **per head** (§0). `W_UK_K` / `W_UV` in the pseudocode are its `[..., :d_nope]` / `[..., d_nope:]` column slices; the 256 KB the §5 tiling streams is the per-head slice |
| `cu_seqlens_q` | `[B+1]` | int32 | prefix sums of query lengths |
| `cu_seqlens_k` | `[B+1]` | int32 | prefix sums of KV lengths — with a paged cache and chunked prefill $S_k \neq S_q$, so the causal mask needs its own length |
| `block_table` | `[B, max_blocks]` | int32 | paged KV block pointers |
| `positions` | `[total_q]` | int32 | query token positions for the RoPE rotation in step 3; `k_rope` is stored already-rotated (§2.1) |
| `scale` | scalar | fp32 | softmax scale, **host-supplied** — never derived from any head dimension (§0) |

#### Outputs

| tensor | shape | notes |
|---|---|---|
| `out` | `[total_q, H_q, d_V]` | bf16 output |
| `softmax_lse` | `[total_q, H_q]` | fp32 log-sum-exp (for chunked-prefill reduce) |

#### Kernel structure

```
grid:      (H_q / BLOCK_H, total_num_q_blocks, 1)   # H_k = 1, so dim0 carries HEAD blocks
workgroup: (64 * num_warps, 1, 1)                   # BLOCK_H sized at implementation (§9)

for each (head block, q_block):
  1. load q_latent tile → LDS                    # [Bq, r_Q], shared by the block's heads
  for each head h in the block:
    2. apply W_UQ GEMM: q_latent [Bq, r_Q] × W_UQ[h] [r_Q, d_nope+d_rope]
                        → q[h] [Bq, d_nope+d_rope]
    3. split q[h] → q_nope[h] [Bq, d_nope], q_rope[h] [Bq, d_rope];
       rotate q_rope[h] at `positions`
  for each KV tile:
    4. load c_KV tile  [Bk, r_KV]  from paged cache   # shared by the block's heads
    5. load K_rope tile [Bk, d_rope] from paged cache # head-independent
    for each head h in the block:
      6. expand K_nope: c_KV × W_UK_K[h] → [Bk, d_nope]   # PER HEAD (§0)
      7. expand V:      c_KV × W_UV[h]   → [Bk, d_V]      # PER HEAD (§0)
      8. score = scale * (q_nope[h] · K_nope^T + q_rope[h] · K_rope^T)   # [Bq, Bk]
      9. apply causal mask (against cu_seqlens_k)
     10. online softmax update (m[h], l[h], o_acc[h])
    end                                          # head loop
  end                                            # KV tile loop
  11. normalize and write out [Bq, BLOCK_H, d_V]
```

Step 2 (W\_UQ GEMM) is the main structural addition over standard flash attention.
It can be fused as a pre-pass in LDS or as a separate small kernel depending on LDS
pressure; see §5 (tiling/LDS budget).

> **Prefill does not amortize over heads the way decode does.** The latent `c_KV` is
> shared across heads, but its *expansion* is not: steps 6–7 apply a different
> `W_UK[h]` per head, so both the expansion GEMM and the `W_UK` streaming scale with
> the heads resident in the workgroup. Batching heads here amortizes the `c_KV` and
> `K_rope` reads but multiplies in-loop weight traffic — the opposite trade from §4,
> where the shared latent is consumed directly and no per-head weight enters the loop.
> `BLOCK_H` for prefill is therefore a separate sizing question (§9); the prefill
> budgets in §5.1/§5.2 are stated for one head at a time.

> **Alternative implementation path:** CK FMHA already supports
> `(hdim_q=192, hdim_v=128)` natively for gfx942 and gfx950 (see §10.4). If
> the W_UQ and W_UK expansions are done as separate prior GEMMs (latent → expanded
> Q and latent → expanded K/V), the resulting tensors can be fed directly into the
> CK `fmha_fwd` kernel at (192,128) without a custom rocKE kernel. This path should
> be prototyped and measured against the in-loop fusion approach before
> committing to a fully custom kernel — as an external **measurement baseline**, not a
> shippable rocKE instance: `lower_cktile.py` is parity-only and accepts
> `UniversalGemmSpec` / `ImplicitGemmConvSpec`, never attention, so a wrapped CK kernel
> yields no `KernelDef`, no `ATTENTION_REGISTRY` candidate and no byte-identity
> coverage. What the pair costs at (192,128) differs by variant: plain `fmha_fwd` drops
> **bias** and **dropout** but does emit LSE (`fmha_fwd.py:820`, `check_hdim`) — and it
> is the *only* generator with a `(192,128)` tile today. `fmha_fwd_splitkv.py` has none,
> `fmha_pagedkv_prefill.py:587` has its `"192"` entry commented out, and
> `fmha_batch_prefill.py` carries only 128/256, so their LSE guards are unreachable.
> Independently of head dim, CK emits **no** paged-KV forward kernel with LSE:
> `get_pagedkv_pipelines` sets `lse="f"` for every `qr_pagedkv` spec
> (`dispatcher/codegen/fmha/instance_gen.py:1030`). Two consequences: chunked-prefill via
> `softmax_lse` is blocked on CK's paged path (§10.4), and the baseline that *is*
> buildable at (192,128) is the non-paged `fmha_fwd` — so the number measures a
> contiguous-KV kernel, not MLA's paged cache. State that alongside the measurement.

---

## 4. Decode-absorb kernel specification

**Op:** `mla_decode_absorb_fwd` — weight-absorbed decode, $S_q = 1$ per head, bf16.

This is structurally **MQA with `head_dim = r_{KV} + d_{\text{rope}} = 576`**: after
pre-projecting the query latent with the absorbed weights (§2.4), the kernel runs
standard flash decode against the concatenated KV cache `[c_KV ‖ K_rope]`. The
absorbed weights `W_abs`, `W_rope_proj`, and `W_UV` are **constant GPU tensors
loaded once at model startup** — they are not computed per-request.

#### Inputs

| tensor | shape | layout | notes |
|---|---|---|---|
| `c_q` | `[B, r_Q]` | row-major | compressed query latent (one token per seq); **no head axis** — the per-head expansion is in `W_abs` / `W_rope_proj` (§2.4). Same tensor as `q_latent` in §3 |
| `kv_cache` | `[num_blocks, block_size, r_KV + d_rope]` | paged | `c_KV ‖ K_rope` concatenated; 576 elem/token |
| `W_abs` | `[H_q, r_Q, r_KV]` | per-head | **model-load-time constant**: $W_{UQ,\text{nope}} \cdot W_{UK,K}^{\top}$ (§2.4) |
| `W_rope_proj` | `[H_q, r_Q, d_rope]` | per-head | **model-load-time constant**: RoPE slice of $W_{UQ}$, i.e. $W_{UQ,\text{rope}}$ |
| `W_UV` | `[H_q, r_KV, d_V]` | head-major | **model-load-time constant**: value up-projection, **per head** (§0). 16 MiB total; the epilogue streams a 128 KB slice **per head row of the block** (§4 step 7) in 16 KB `r_KV`-tiles |
| `block_table` | `[B, max_blocks]` | int32 | paged KV pointers |
| `seqused_k` | `[B]` | int32 | KV sequence lengths |
| `positions` | `[B]` | int32 | current token position per sequence; drives the query-side RoPE rotation in the pre-step (§2.4). `K_rope` is stored already-rotated, so this is required, not optional |
| `scale` | scalar | fp32 | softmax scale, **host-supplied** — never derived from any head dimension (§0) |

#### Outputs

| tensor | shape | notes |
|---|---|---|
| `out` | `[B, H_q, d_V]` | bf16 |

#### Kernel structure (3D split-KV, analogous to `attention_tiled_3d`)

```
# Pre-step (separate device kernel, once per decode step — not host compute):
q_abs[B, H_q, r_KV]    = c_q · W_abs        # project query into latent space (batched over h)
q_rope[B, H_q, d_rope] = RoPE_i(c_q · W_rope_proj)  # project, THEN rotate at position i

grid:      (B * (H_q / BLOCK_H), NUM_SEGMENTS, 1)   # BLOCK_H = MFMA M-tile of query HEADS
workgroup: (64 * num_warps, 1, 1)                   # BLOCK_H, num_warps sized at impl (§9)

per segment workgroup:   # owns BLOCK_H query heads of one sequence; they SHARE the KV tile
  1. load q_abs [BLOCK_H, r_KV] and q_rope [BLOCK_H, d_rope] for this (batch, head block)
  for each KV tile in segment:
    2. load kv_cache tile [Bk, r_KV + d_rope]   # ONE read, amortized over BLOCK_H heads
    3. split → c_KV [Bk, r_KV], K_rope [Bk, d_rope]
    4. score = scale * (q_abs · c_KV^T + q_rope · K_rope^T)  # [BLOCK_H, Bk], scale = 1/sqrt(192)
    5. online softmax update (m, l)             # independent per head row
    6. acc[BLOCK_H, r_KV] += p * c_KV           # accumulate in LATENT space, reusing the
  end                                           #   resident c_KV — but on the OTHER
                                                #   contraction axis (§5.2). No W_UV here.
  7. epilogue, per head row h of the block:
       out_partial[h, d_V] = acc[h, :] · W_UV[h]  # W_UV is PER HEAD, so the weight varies
                                                  #   along M: this is not a single MFMA.
                                                  #   Schedule + cost: implementation (§9)
  8. write partial (m, l, out_partial) to segment workspace

reduce_segments kernel (same as attention_tiled_3d's reduce, HD = d_V = 128):
  combine partials → final bf16 output
```

#### Query heads are the MFMA M dimension

`S_q = 1` makes the query *position* axis 1, but it does not make the MFMA M-tile 1.
In a decode kernel M is the **query-head** axis, sized by how many query heads share
one KV head. `attention_tiled_3d` already works this way (`NQK = num_query_heads //
num_kv_heads`, `BLOCK_M = 16`, row → `kv_head*NQK + row % NQK`).

MLA has $H_k = 1$, so **all** $H_q = 128$ heads share the same latent — the largest
head-sharing factor of any attention variant, not the smallest. One workgroup per
`(batch, head)` would therefore issue the same KV tile read $H_q$ times, and would run
every MFMA at M=1: the atom is a fixed shape, so a `16x16x16` MFMA costs the same
whether 1 row or 16 rows of M are populated.

**Requirement: the query-head axis is the MFMA M dimension.** A workgroup owns
`BLOCK_H` query heads of one sequence and they share one KV tile read. The *axis* is a
property of the grid, not a tuning knob — it cannot be retrofitted onto a
one-head-per-workgroup kernel, because the grid shape determines the spec. Its *size*,
`BLOCK_H`, is a tuning knob and is deferred below.

> **Order of magnitude — analytical, not measured.** The model counts one term only:
> the redundant KV traffic head-batching removes, which shrinks as
> $1/\texttt{BLOCK\_H}$.
>
> | config (`kv_len` 8192) | flash loop | `W_abs` pre-step | total |
> |---|---|---|---|
> | `BLOCK_H = 1`, `B = 1` | ~71 µs | ~38 µs | ~109 µs |
> | `BLOCK_H = 16`, `B = 1` | ~4 µs | ~38 µs | ~42 µs |
> | `BLOCK_H = 1`, `B = 32` | ~2270 µs | ~38 µs | ~2310 µs |
> | `BLOCK_H = 16`, `B = 32` | ~142 µs | ~38 µs | ~180 µs |
>
> ≈16× on the flash loop; 2.6× end-to-end at `B = 1`, ~13× at `B = 32`.
>
> **What the model is.** Redundant-read bandwidth only:
> `kv_len × 1152 B × (H_q / BLOCK_H) × B`, over ~17 TB/s on the assumption that the
> 9.4 MB KV working set is MALL-resident but does not fit one XCD's 4 MB L2. The 16× is
> therefore exactly `128 / 8`, the read-amplification ratio. **The M=1 MFMA lane waste
> is not in these numbers**, nor is any latency, launch or occupancy term.
>
> **Why the `B = 1` rows are the weakest.** A bandwidth limit is only reached if the
> machine is busy. At `B = 1, BLOCK_H = 16` the grid is `8 × NUM_SEGMENTS` workgroups,
> so filling 304 CUs needs `NUM_SEGMENTS ≳ 38` — about 13 KV tiles per segment at
> `kv_len = 8192, Bk = 16`, feasible but not free, and unreachable at `kv_len = 512`
> (32 tiles in total). Below that the `B = 1` figures are launch-bound rather than
> bandwidth-bound and the 2.6× overstates the gain. `NUM_SEGMENTS` is therefore an input
> to `BLOCK_H` sizing, not an independent knob.
>
> These are first-order estimates to justify the requirement, not performance targets.

> **The pre-step and `BLOCK_H` are coupled.** The pre-step is batch-invariant while KV
> work scales with batch, so which term dominates inverts with `B`. At `BLOCK_H = 16`
> (≈4.4 µs per sequence) the flash loop overtakes a materialized `W_abs` (~38 µs) at
> about `B = 9`, and a two-stage pre-step (~12.7 µs) at about `B = 3` — see the `W_abs`
> open question below. Two consequences: head-batching is worth ~2–3× end-to-end at
> `B = 1` and ~13× at `B = 32`, so the requirement holds at both but its *urgency* is a
> batch-size argument; and the pre-step form should be fixed before `BLOCK_H` is swept,
> since it sets how much of the step `BLOCK_H` can affect at all.

> **`attention_tiled_3d` cannot be reused as-is.** It validates
> `1 <= num_queries_per_kv <= 16` and `16 % num_queries_per_kv == 0`
> (`gfx942/attention_tiled_3d.py:240-246`; gfx950: 252-258). MLA's
> `NQK = H_q / H_k = 128` fails that gate, so the head→row remap (`Bq = 1` query
> position × `BLOCK_H` head rows) is a real spec change, not reuse. See also the engine
> gap below.

#### Sizing `BLOCK_H`

**`BLOCK_H` sizing is an implementation task (§9), not settled here.** The value trades
three things that cannot be resolved on paper:

- the latent accumulator's register cost, which sets the wave partition — and note the
  score contracts over `r_KV` while the accumulate produces it, so a partition chosen
  for one constrains the other;
- grid parallelism: batching divides the workgroup count by `BLOCK_H`, coupling it to
  `NUM_SEGMENTS` and to batch size;
- the epilogue, where per-head `W_UV` gives the projection an M-varying operand, so it
  does not amortize over the head block the way the KV read does.

Each of these changes the LDS and register budgets in §5, which are therefore stated at
`BLOCK_H = 1` and must be re-derived when `BLOCK_H` is chosen. Residual
cross-workgroup reuse (the `H_q / BLOCK_H` head blocks still sharing a segment) is an
L2/XCD locality question; rocKE precedent is the `chiplet_*` engine traits (GEMM and
implicit-GEMM-conv paths; the workgroup-ID remap itself is `chiplet_transform_chunked`
in `platform/python/rocke/helpers/grid.py`) and `use_q_major_grid` on gfx942
`attention_tiled_2d`. Fold the grid mapping into the new spec rather than retrofitting
it; as a new spec trait it is a dual-engine change (§9).

> **Engine gap: `attention_tiled_3d` cannot express this spec today.** Beyond the
> `NQK <= 16` gate above, the existing spec carries a single `head_size` that drives
> *both* the KV descriptor width and the workspace/output width, and
> `supports_tiled_3d` gates `head_size in {64, 128, 256}`. Decode-absorb needs a KV
> width of `r_KV + d_rope = 576` with an output width of `d_V = 128`. "The reduce is
> unchanged" holds only for the **reduce** kernel (it stays an `HD = 128` reduce); the
> **segment** kernel needs a new spec with decoupled `hdim_kv` / `hdim_out`, a relaxed
> `head_size` gate, and the `BLOCK_H` head remap. That belongs in the implementation
> scope (§9), not assumed away here.

The pre-step projections `c_q · W_abs` and `c_q · W_rope_proj` are small GEMMs
batched over the $H_q$ heads (`[B, r_Q] × [r_Q, r_KV]` and `[B, r_Q] × [r_Q, d_rope]`
per head) and can be fused into a single batched GEMM kernel or executed as a
pre-kernel. They are **not inside the flash loop** — they run once per decode step.
Neither projection is transposed: the weights are stored `[in, out]` (§0), matching
the form used in §2.4.

The absorbed weight `W_abs[H_q, r_Q, r_KV]` is `H_q × 1536 × 512` elements; for
`H_q = 128` this is ~192 MiB (~201 MB) at bf16 — too large to live in LDS. It is used
only in the pre-step GEMM, not streamed per KV tile.

> **Open question — materialize `W_abs`, or apply it in two stages?**
> "Too large for LDS" is not the constraint that matters: the materialized pre-step
> reads all 192 MiB on **every decode step**, ~38 µs at 5.3 TB/s, against ~4.4 µs for
> the flash loop at `kv_len = 8192, BLOCK_H = 16, B = 1` (the estimate table above).
> The two-stage alternative keeps $W_{UQ,\text{nope}}$ and $W_{UK,K}$ separate and
> applies them in sequence ($q_{\text{nope}} = c_q \cdot W_{UQ,\text{nope}}^{(h)}$,
> then $q_{\text{abs}} = q_{\text{nope}} \cdot W_{UK,K}^{(h)\top}$), producing an
> identical $q_{\text{abs}}$:
>
> | form | weights read / step | MAC / token | pre-step @ 5.3 TB/s |
> |---|---|---|---|
> | materialized `W_abs` | 192 MiB | 100.7 M | ~38 µs |
> | two-stage | 48 + 16 = **64 MiB** | **33.6 M** | ~12.7 µs |
>
> Only the nope slice of $W_{UQ}$ participates (§2.4), which is why the two-stage figure
> is 48 MiB and not the full 72 MiB. Both exclude `W_rope_proj` (24 MiB, 12.6 M MAC),
> which each form needs equally. SGLang issue #4615 (§10.6) is this exact trade-off.
>
> **What weighs the other way**, since the byte and MAC counts do not: the materialized
> form is one GEMM rather than two, so it is a simpler graph-input contract for §7.5
> (`W_abs` is one tensor, not a pair with an intermediate) and a simpler thing to fuse
> into the decode prologue; and because the pre-step is batch-invariant, its 3×
> disadvantage shrinks at serving batch, where the flash loop dominates anyway.
>
> **Not resolved here.** §2.4 specifies the materialized form. This note states both
> sides so the choice is explicit; it must be settled before §7.5 fixes the
> absorbed-weight graph-input contract, and §10.6's summary of what SGLang actually
> stores should be verified against source at the same time.

#### Why `W_UV` is applied once, not per tile

`W_UV` is a fixed linear map, so by linearity the per-tile V expansion can be hoisted
out of the flash loop entirely:

$$
\sum_j p_j \big(c_{KV,j} \cdot W_{UV}\big) = \Big(\sum_j p_j\, c_{KV,j}\Big) \cdot W_{UV}
$$

This survives the online-softmax rescaling because the rescale is a scalar
(`(α·acc) · W_UV = α·(acc · W_UV)`, §2.6). Step 6 above therefore reuses the `c_KV`
already resident in LDS for the score, and the projection becomes a
`[1, 512] × [512, 128]` epilogue on register-resident data — not a separate GEMM
launch and not an HBM round-trip. Over `N_tiles = S_kv / Bk` iterations:

| Variant | in-loop value work | `W_UV` residency |
|---|---|---|
| Per-tile expansion | $S_{kv} \cdot r_{KV} \cdot d_V$ | LDS-resident every iteration |
| Deferred (this spec) | $S_{kv} \cdot r_{KV}$, plus one $r_{KV} \cdot d_V$ at the end | not in the loop at all |

That is ~$d_V$× (~128×) less value-side *arithmetic* inside the loop — the value term
now costs the same order as the score term, since it reuses the same operand.
**That ratio is not a predicted speedup.** The reasons to prefer the deferred form are
structural, not arithmetic:

1. `W_UV` leaves the flash loop's LDS budget entirely. That is what the occupancy
   *estimates* in §5.1/§5.2 assume — those are LDS ceilings, not measurements; see the
   caveats stated there.
2. Per-tile staging re-fills a slice of the per-head 128 KB `W_UV` into LDS on every
   KV tile, on top of the 18–36 KB KV tile. How much of that L2 absorbs is unmeasured.

It is also what §2.4 means by "no KV expansion and no per-token weight application",
and what FlashInfer's decode kernel does when it reuses `c_KV` as both K and V (§10.5).

**Trade-off under 3D split-KV — and it is not free.** Placing the `W_UV` epilogue
*inside* the workgroup (step 7 above) keeps the segment workspace `d_V`-wide (128) and
applies `W_UV` once per (head row, segment) — `NUM_SEGMENTS` times per row instead of
`N_tiles` times. The alternative is to write the `r_KV`-wide accumulator to the
workspace and apply `W_UV` after the segment reduce: `W_UV` is then applied exactly
once per head row, but the reduction workspace grows ~4× (512 fp32 per partial
instead of 128).

Because `W_UV` is **per head** (§0), the flop count is not the deciding term — the
weight *traffic* is. The per-workgroup epilogue re-reads the whole 128 KB `W_UV[h]`
slice once per (head, segment): at `B = 1`, `H_q = 128` and, say, `NUM_SEGMENTS = 8` that is
`128 × 8 × 128 KB = 128 MiB` of requests per decode step, against `9.4 MB` of KV at
`kv_len = 8192`. The after-reduce variant divides that by `NUM_SEGMENTS` (16 MiB). The
16 MiB `W_UV` working set fits MI300X's 256 MB MALL, so most of the difference should
be absorbed below HBM — but it is L2/MALL request traffic contending with the KV
stream, not something "tiny next to the KV reads", and it grows linearly with the
split-KV factor.

**Step 7 (per-workgroup epilogue) is the spec** because it keeps the segment workspace
`d_V`-wide — the same `[*, NUM_QH, NUM_SEG, HD]` layout the existing
`attention_tiled_3d` reduce already consumes, at `HD = d_V = 128`. But the after-reduce
variant must be **measured**, not deferred to "if the epilogue shows up in a profile",
and the crossover between the two is a sizing question that moves with
`NUM_SEGMENTS` and `BLOCK_H` — settle it with them (§9), not here.

The flash loop's main remaining scheduling constraint is the `kv_cache` tile itself
(see §5).

---

## 5. Per-arch tiling and LDS budget

### 5.1 gfx942

**MFMA atom:** `mfma_f32_16x16x16_bf16` (narrow default); fp16 flash option
`mfma_f32_32x32x8_f16` is not relevant for MLA (bf16 first).

**LDS per CU:** 64 KB. There is no fixed per-WG budget: LDS per workgroup and
workgroups per CU are the same number seen twice (`65536 / lds_bytes`). 16–32 KB/WG is
what 2–4 WG/CU costs; each tiling below states the WG/CU it implies instead.

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
| c_KV tile | 64 × 512 × 2 = 64 KB | **the entire 64 KB LDS**, leaving nothing for anything else — must use Bk=16 or stream r_KV |
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
3D split-KV path is not applicable to prefill (each workgroup already has Sq > 1).

#### Decode-absorb — gfx942

The pre-step GEMMs (`c_q · W_abs`, `c_q · W_rope_proj`) run as separate small
kernels before the flash loop — `W_abs` is **not streamed per KV tile**. The flash
loop itself operates like standard single-head-dim=576 MQA decode.

> **These budgets are a `BLOCK_H`-independent floor, not a candidate configuration.**
> `BLOCK_H = 1` is *excluded* by §4 — the head axis being the MFMA M dimension is a
> requirement, not a knob — so nothing below describes a kernel this doc would ship.
> What the table gives is the part of the budget that does not move with `BLOCK_H`: the
> shared KV tile. Three terms are missing, and all three grow with `BLOCK_H`:
>
> 1. the latent accumulator, `BLOCK_H × r_KV` fp32 in registers;
> 2. the cross-wave reduction buffer that the accumulator's wave partition forces,
>    `BLOCK_H × d_V` per wave in LDS;
> 3. the epilogue's `W_UV` traffic — §4 step 7 applies a *different* `W_UV[h]` per head
>    row, so the 16 KB aliased slice below is a **per-head** figure. At `BLOCK_H` heads
>    the epilogue either serialises `BLOCK_H` slice streams (LDS unchanged, epilogue
>    time × `BLOCK_H`) or holds more than one live (LDS × the number held). This is the
>    term §4 flags as not amortizing over the head block.
>
> Read the LDS peak below as a lower bound and the occupancy as an upper bound; both
> must be re-derived once `BLOCK_H` and `num_warps` are chosen.

LDS and register footprint of the flash loop and its epilogue:

| Buffer | Size (at Bk = 16) | Notes |
|---|---|---|
| `kv_cache tile [Bk, 576]` | 16 × 576 × 2 = 18 KB | LDS. c_KV + K_rope concatenated; shared by every head in the workgroup (§4). The only in-loop LDS buffer **if `register_pv` eliminates `P_lds`** — see the recommendation below |
| `W_UV slice [r_KV_tile, d_V]` | 16 KB, **epilogue only** | LDS, aliased onto the KV tile's allocation — see below |
| `q_abs + q_rope` per head | 576 × 2 = 1.125 KB | registers, **bf16** — an MFMA A-operand, so it matches the atom's input type; the pre-step's fp32 result is rounded once on the way in |
| `acc` per head | 512 × 4 = 2 KB | registers, fp32 **latent-space** accumulator (§4); 4× the `d_V`-wide one it replaces |

Both register rows scale with `BLOCK_H` and drive the wave partition — see the sizing
note at the head of this subsection.

`W_UV[h][r_KV=512, d_V=128]` = 128 KB **per head** — does not fit in LDS, and does not
need to: per §4 it is applied once in the epilogue, streamed in `r_KV`-tiles of 64
(`W_UV_slice[64, 128]` = 16 KB per slice) into LDS **aliased onto the KV tile's
allocation**.

> **The epilogue slice must be an explicit LDS alias, not a second allocation.** LDS is
> a single group segment sized statically by the compiler at dispatch and live for the
> whole workgroup — it is never "released" mid-kernel. To get the 18 KB peak the
> builder must hand the epilogue the *same* `smem_alloc` handle as the KV tile. The
> source-level rocKE precedent is `Q_lds = K_lds` under `Q_ALIAS_K`
> (`library/kernels/gfx942/attention_tiled_2d.py:1922-1927`) — one `smem_alloc`, two
> names. `Acc_lds` is **not** that precedent: it is its own `smem_alloc` (same file,
> line 1938) that the *backend* was observed to overlap with the loop-dead K/V region —
> an empirical allocator result, not a contract. Do not plan around it: every
> `tile.smem_alloc` lowers to a distinct module-level LDS global
> (`platform/python/rocke/core/lower_llvm.py:1723`, `_op_tile_smem_alloc`), so the
> source-level alias is the
> only guaranteed way to reach the peak. Add an `s_barrier`
> after the last `c_KV` read and a second `s_barrier` + `s_waitcnt lgkmcnt(0)` before
> the epilogue reads. As two separate allocations the static segment is 18 + 16 = 34 KB
> → 1 WG/CU, exactly the number this deferral is meant to beat.

Full KV tile loop:

```
# flash loop — no W_UV, no V expansion:
for kv_tile in segment:
  load kv_cache tile [Bk, 576] → LDS            (18 KB)
  score = scale * (q_abs · c_KV_tile^T + q_rope · K_rope_tile^T)
  online softmax update (m, l)
  acc[r_KV] += p * c_KV_tile                    # reuses the c_KV already in LDS

# epilogue — once per query row, reusing the KV tile's LDS:
for r_kv_slice in range(0, r_KV, 64):
  load W_UV_slice [64, 128] → LDS               (16 KB)
  out_partial[d_V] += acc[r_kv_slice:+64] · W_UV_slice

LDS peak = max(18 KB flash loop, 16 KB epilogue) = 18 KB single-buffered
(36 KB with a double-buffered KV tile) — REQUIRES the two buffers to be aliased.
```

**Recommendation:** Evaluate the `register_pv` pattern (eliminating `P_lds`) — keeping
the softmax probability in registers (the gfx950 `attention_tiled_2d_fastkv_regp.py`
technique, which has no gfx942 precedent kernel) pairs naturally with the latent-space
accumulator, which consumes P immediately against the resident `c_KV` tile. This is
a **priority**, not a nice-to-have.

**Occupancy estimate gfx942 decode — LDS ceiling only, nothing measured.** LDS admits
3 workgroups/CU at the 18 KB aliased peak (65536 / 18432 = 3.5), 1 if the KV tile is
double-buffered (36 KB).

> **Workgroups/CU is not occupancy — the workgroup size has to be stated with it.** At
> the 64-thread (single-wave) workgroup `attention_tiled_3d` uses today, 3 WG/CU is
> 3 waves/CU = **0.75 waves/SIMD**: one SIMD idle, zero latency hiding. The figures
> below therefore assume `num_warps = 4` (256 threads) — the smallest non-degenerate
> value, not a settled one (§9) — at which 3 WG/CU means 12 waves/CU = **3 waves/SIMD**.
> AMD documents a 10 waves/SIMD (40/CU) cap for CDNA3, but rocKE's own
> `probe_occupancy.py` models a conservative 8
> (`platform/dsl_docs/optimization/arch/gfx942.md` §21.4), so a probe run reports
> against 8.

VGPRs are the more likely limiter once heads are batched, and must be sized per *wave*,
not per head row. Two structural consequences hold regardless of the value chosen:

- The `r_KV = 512` accumulator must be **split across the workgroup's waves**, never
  replicated — replication is an immediate occupancy wall. There are two axes to split
  on and neither is free, because the score contracts *over* `r_KV` while the accumulate
  *produces* it:
  - **Split on `r_KV`.** Each wave owns `512 / num_warps` latent columns and so holds
    only that slice of `q_abs`, making step 4 yield a *partial* score. Completing it
    needs a cross-wave reduction **inside the flash loop, once per KV tile**, with a
    barrier on the critical path of every iteration. This is the option that could
    disqualify the design, and it must be costed before `BLOCK_H` is chosen, not after.
  - **Split on the head axis.** The score stays wave-local, but every wave holds the
    full `r_KV` accumulator for its head rows, so register cost per wave does not fall
    with `num_warps` — this is the option that hits the occupancy wall first. It is
    available only when `BLOCK_H >= num_warps`.
- Either way the reduction needs a **cross-wave buffer in LDS plus a barrier**: per KV
  tile under the `r_KV` split, or once before the segment workspace write under the head
  split, where it is live alongside the epilogue's `W_UV` slice. Neither cost is in the
  table above.

So the 18 KB peak above is a lower bound and the 3 WG/CU an upper bound — neither is the
shipped configuration. Double-buffer-vs-occupancy is worth sweeping, but only after
`BLOCK_H`, `num_warps` and the accumulator partition are pinned (§9).

---

### 5.2 gfx950

**MFMA atoms:** `mfma_f32_16x16x32_bf16` (default wide-K) and
`mfma_f32_32x32x16_bf16` (combo, `ds_read_tr`-enabled). Reference:
`library/kernels/gfx950/attention_tiled_2d.py`, `_fastkv_regp.py`.

**LDS per CU:** 160 KB = 163840 B (CDNA4). Sources: `platform/cpp/core/arch/data.cpp`
(`k_target_gfx950.lds_capacity_bytes = 163840`), `platform/dsl_docs/optimization/arch/gfx950.md`
§21.2, and the compile-time gate in `library/kernels/common/attention_unified.py`
("over gfx950's 163840 B cap"). At 2–4 WG/CU: **40–80 KB per WG**.

#### Prefill — gfx950

The wider K-step (32 per MFMA vs 16 on gfx942) amortizes the c_KV streaming cost
better. Proposed tiling:

| Parameter | Value | Notes |
|---|---|---|
| `Bq` | 32 | 32×32 MFMA M-tile |
| `Bk` | 32 | KV **tile** = 2 paged-KV blocks at `block_size = 16` — see the note below |
| `r_KV_tile` | 64 | $r_{KV}$ K-step for the latent expansion; 128 (2× gfx942) is the target, but its `W_UK` slice alone is 64 KB — see the LDS budget below |
| `num_warps` | 4 | 256 threads |

> **`Bk` is the kernel's KV tile, not the paged-cache block size.** The two are
> decoupled in rocKE exactly as in `attention_tiled_2d.py`, which requires only
> `tile_size % block_size == 0` and walks `tile_size / block_size` paged blocks per
> iteration. All four bench shape files (§8.2) specify `block_size: 16` — the repo
> default for paged KV — so `Bk = 32` here means **two** blocks per tile, and the LDS
> figures below hold at that block size. `Bk` is a tuning knob to sweep; `block_size`
> is fixed by the cache allocator.

LDS layout per iteration:
- `c_KV_slice[32, 128]` = 8 KB (latent slice, one r_KV-tile)
- `W_UK_slice[128, 256]` = 64 KB — at the top of the 40–80 KB/WG budget; alone it would still admit 2 WG/CU (`163840 / 65536 = 2.5`), but with the rest of the tile it does not (see below)

At `r_KV_tile = 64` (half):
- `W_UK_slice[64, 256]` = 32 KB
- `c_KV_slice[32, 64]` = 4 KB
- `K_rope[32, 64]` = 4 KB
- `K_nope_stage[32, 128]` = 8 KB
- `V_stage[32, 128]` = 8 KB
- **Total ≈ 56 KB** — `163840 / 57344 = 2.9`, i.e. **2 WG/CU** at 160 KB LDS

(The `r_KV_tile = 128` variant totals ≈ 92 KB — `163840 / 94208 = 1.7`, i.e. 1 WG/CU.
That is why 64 is the proposed value.)

> **`K_nope_stage` / `V_stage` are bf16 staging buffers, not fp32 accumulators.** The
> expansion runs over `r_KV / r_KV_tile = 8` slices; that reduction must stay in the
> MFMA fp32 accumulator registers and be rounded to bf16 into LDS **once**, after the
> last slice — bf16 because the next MFMA consumes them as B-operands, which the
> hardware accepts only in bf16/fp16/fp8. Accumulating in LDS instead would make each
> buffer fp32 (16 KB, not 8 KB), push the total to 72 KB, and drop this config to
> 1 WG/CU. The same applies to `K_nope_acc` / `V_acc` in the gfx942 prefill tiling
> above (4 KB each as bf16 staging; 8 KB each if made fp32, taking 44 KB → 52 KB).

> **`ds_read_tr` layout recommendation (gfx950 prefill):** Store `W_UK` in transposed-dimension
> alignment (column-major in the `r_KV` axis) so that `ds_read_tr16_b64` can deliver
> the MFMA B-operand for the latent expansion step without a separate transpose.
> This is the same technique that makes the gfx950 `attention_tiled_2d` V-stage fast
> (see `platform/cpp/instances/gfx950/attention_tiled_2d_kv_body_pv_epilogue.cpp`).
> This layout **must** be evaluated — it is a recommended direction, not optional.

**Occupancy estimate gfx950 prefill:** 2 workgroups/CU at 56 KB (LDS-limited:
`163840 / 57344 = 2.9`; a third needs ≤ 54 KB). At the specified `num_warps = 4` /
256 threads that is 8 waves/CU = **2 waves/SIMD** — LDS is the binding limiter only if
the kernel stays under 512/2 = 256 registers per lane, which a 32×128 fp32 output
accumulator plus K/V staging will approach. gfx950 caps at 8 waves/SIMD; `waves_per_eu`
cannot raise occupancy above whichever of LDS/VGPR/AGPR binds first, it only lets the
compiler target a higher one at the cost of spills.

#### Decode-absorb — gfx950

Same pre-step + flash-loop split as gfx942: `W_abs` is applied offline; the flash
loop is MQA with `head_dim=576`. The wider MFMA (32-wide K-step) and 160 KB LDS allow
a larger KV tile.

> **As in §5.1, this is a `BLOCK_H`-independent floor, not a candidate configuration.**
> The `32x32x16` atom makes 32 the natural M-tile to evaluate first, but all three
> missing terms §5.1 lists — accumulator, cross-wave reduce buffer, per-head epilogue
> traffic — apply here too and are absent from the numbers below.

**Proposed gfx950 decode tiling:**
- `Bk = 32` (KV tile = 2 paged blocks at `block_size = 16`; same decoupling note as prefill above)
- `kv_cache tile [32, 576]` = 36 KB — the only in-loop LDS buffer, assuming `register_pv` eliminates `P_lds` (see below)
- `W_UV` is applied **once in the epilogue** (§4), streamed in `r_KV`-tiles of 128
  (`W_UV_slice[128, 128]` = 32 KB) into LDS **aliased onto the KV tile** — see below
- `LDS peak ≈ 36 KB` — 4 workgroups/CU at 160 KB LDS (163840 / 36864 = 4.4)

> **The epilogue slice must be an explicit LDS alias here too** (same mechanism and
> barriers as §5.1). Without the alias the static group segment is 36 + 32 = 68 KB →
> `163840 / 69632 = 2.35`, i.e. **2 WG/CU** — the epilogue buffer would cost exactly as
> much as staging `W_UV` per tile, and the deferral would buy nothing.
>
> Staging `W_UV` per tile keeps both buffers live for the whole loop: 68 KB → 2 WG/CU.
> Deferring per §4 **and** aliasing the epilogue slice is what takes this to 4 WG/CU.

**`ds_read_tr` (gfx950 decode) — an in-loop lever, not an epilogue one.** Deferring
`W_UV` does not remove the transpose problem, it *relocates* it onto `c_KV`. The
resident KV tile is read twice per iteration on **opposite** contraction axes:

| step | expression | MFMA K axis | wants contiguous per lane |
|---|---|---|---|
| 4 (score) | `q_abs[BLOCK_H, r_KV] · c_KV^T[r_KV, Bk]` | `r_KV` | `r_KV` — the natural `[Bk, 576]` layout |
| 6 (accumulate) | `p[BLOCK_H, Bk] · c_KV[Bk, r_KV]` | token | token — **stride 576 elems, the transpose** |

Both MFMA A- and B-operands need their K elements contiguous per lane, so step 6 cannot
use the step-4 layout as-is. `ds_read_tr16_b64` / `ds_read_tr16_b128` deliver exactly
this transposed gather from the natural layout, which makes them a **priority in-loop
lever on gfx950 decode**, ranked with `register_pv`. Point them at `c_KV`; the
epilogue's `W_UV[h]` projection runs once per (head row, segment) and is not worth a
bespoke layout. See `platform/cpp/instances/gfx950/attention_tiled_2d_kv_body_pv_epilogue.cpp`
(§11) for the existing V-staging use.

**Bank conflicts on the strided read.** The tile row stride is 576 bf16 = 1152 B = 288
dwords. `288 ≡ 0 (mod 32)` and `288 ≡ 32 (mod 64)`, so lanes walking the *token* axis
collide on 1–2 banks — a worst-case ~32-way conflict on both gfx942 (32 banks) and
gfx950 (64 banks; `ds_read_b128` has a 64-dword conflict period, every other
`ds_read`/`ds_write` opcode 32). Pad the LDS row stride off the conflict period or
XOR-swizzle it before measuring anything; per `platform/dsl_docs/optimization/arch/gfx950.md`
§21.2 the house preference is padding on gfx950 (abundant LDS) and XOR on gfx942
(capacity-constrained).

**gfx942 has no transpose read at all** (`has_ds_read_tr = false` in the arch catalog),
so a fully conflict-free transposed layout is not reachable there without a second copy.
The gfx942 options are (a) eat the strided read with padding, or (b) build a second
`[576, Bk]`-oriented copy via the `perm_b32` register transpose the existing gfx942 V
stage uses — which costs another 18 KB and takes decode to 36 KB / 1 WG/CU, the same
cost as double buffering. §5.1's 18 KB single-buffer peak assumes **(a)**; pick
explicitly before implementing.

**`register_pv` recommendation (gfx950 decode):** Apply the `_fastkv_regp` register-P
technique (eliminate `P_lds` by keeping the softmax probability in registers,
`gfx950/attention_tiled_2d_fastkv_regp.py`). The latent-space accumulator consumes P
immediately against the resident `c_KV` tile, so removing `P_lds` is a **priority**.

**Occupancy estimate gfx950 decode — LDS ceiling only, nothing measured.** LDS admits
4 workgroups/CU at the 36 KB aliased peak (163840 / 36864 = 4.4). At `num_warps = 4`
(256 threads) that is 16 waves/CU = **4 waves/SIMD**, against a gfx950 cap of
8 waves/SIMD (32/CU) — but this is an LDS division, so it is a ceiling and not a
prediction. As in §5.1 it omits the head-batched accumulator and the cross-wave
reduction buffer, and VGPRs are the more likely binding limiter once those are counted.
`waves_per_eu` only retargets the compiler's register budget; it cannot raise occupancy
past whichever of LDS/VGPR/AGPR binds first.

---

## 6. Dtype plan

| Phase | Dtype | Arch |
|---|---|---|
| 1 | bf16 | gfx942 + gfx950 (prefill and decode-absorb) |
| 2 | fp8 e4m3 (KV cache) | gfx950+ only (fp8 prefill and fp8 decode-absorb) |

**fp8 approach:** Follow the existing sync-dequant pattern from
`library/kernels/common/fmha_fwd_fp8.py` and the gfx950 `ALGORITHM.md` §7.3:
store `c_KV` and `K_rope` as fp8 e4m3 with per-block scale factors; dequant to
bf16 in LDS before the MFMA. The W_UK and W_abs weights remain bf16: quantizing them
adds accuracy risk, and the KV cache is what fp8 is being applied to here. Note this is
a scoping decision, not a claim that weight traffic is negligible — §4 records `W_abs`
at ~192 MiB per decode step, which dominates at low batch.

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

MLA instances in the AOT catalog use the new op strings.
`RockeClientDispatcher::select()` already indexes candidates by
`AotCatalog::candidatesFor(problem.op, problem.arch)` *before* running
`SelectionConstraints::satisfies()`, and `satisfies()` never compares `problem.op` — so
distinct op strings alone keep MLA and non-MLA instances (and prefill vs decode-absorb)
from matching each other. **No `satisfies()` change is needed for that.**

If a single shared op string is used instead, `mlaMode` must be made a real selection
key one of two ways: a new `CompileSpec` member that `satisfies()` compares exactly
(alongside `dtype` / `headSize` / `maskMode`), or a key published by
`SdpaProblem::attributes()` and constrained through `AotInstance::attributeConstraints`.
Simply adding the field to `SdpaProblem` has no effect — the free-form attribute map is
only consulted via those constraints, so an unconstrained new key is inert.

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
      intermediate tensor. The pre-step (`c_q · W_abs` — no transpose, §0) is a batched
      GEMM that produces `q_abs[B, H_q, r_KV]` — its lifetime is one decode step.
- [ ] Confirm whether the prefill crossover dispatch (§2.5: in-loop vs materialize,
      threshold ~200 tokens) is handled inside the kernel op or by the framework
      choosing between two ops.
- [ ] Confirm whether chunked-prefill (`softmax_lse` output) is in scope for the
      initial integration target.
- [ ] Confirm AotCatalog kpack timeline relative to implementation start.

---

## 8. Test and bench plan

### 8.1 Correctness reference

A Python reference in `library/builders/mla/ref_mla_attn.py` implementing the expanded-form attention.

> **Layout note.** `library/builders/` is a Python package and every existing
> subdirectory (`common/`, `gfx942/`, `gfx950/`, `gfx1151/`, `gfx1250/`) carries an
> `__init__.py`; `mla/` currently holds only this document, so the first code change
> under it must add one or `builders.mla.ref_mla_attn` will not import. `mla/` is also
> the first *family*-scoped rather than arch-scoped builders directory — deliberate,
> since the reference is arch-neutral — but the arch-specific parity and bench entry
> points still belong under `builders/gfx942/attention/` and
> `builders/gfx950/attention/` next to their siblings, not under `mla/`.

Weights are stored `[in, out]` (§0), so every up-projection below is a plain `@`
with no `.T`; the only transposes are the score-forming contractions, which are
written as `einsum` to keep the `H_q` axis explicit. `scale` is `1/sqrt(192)` for
both functions — see the note in §0.

```python
scale = (d_nope + d_rope) ** -0.5   # = 1/sqrt(192) — NOT 1/sqrt(576). See §0.

def ref_mla_prefill(q_latent, c_kv, k_rope, W_UQ, W_UK, cu_seqlens, positions, causal=True):
    # q_latent [total_q, r_Q] (no head axis); W_UQ [H_q, r_Q, d_nope + d_rope]
    q = torch.einsum("tr,hro->tho", q_latent, W_UQ)       # [total_q, H_q, d_nope+d_rope]
    q_nope, q_rope = q.split([d_nope, d_rope], dim=-1)    # [t,h,128], [t,h,64]
    q_rope = apply_rope(q_rope, positions)                # k_rope is stored post-rotation (§2.1)
    # expand all KV positions — W_UK is PER HEAD [H_q, r_KV, d_nope + d_V] (§0),
    # so the shared latent c_kv expands to a different K_nope/V for every head
    K_nope = torch.einsum("sr,hro->sho", c_kv, W_UK[:, :, :d_nope])   # [S, H_q, d_nope]
    V      = torch.einsum("sr,hro->sho", c_kv, W_UK[:, :, d_nope:])   # [S, H_q, d_V]
    scores = scale * (torch.einsum("thd,shd->ths", q_nope, K_nope)
                      + torch.einsum("thd,sd->ths", q_rope, k_rope))  # k_rope is head-shared
    p = softmax(causal_mask(scores, cu_seqlens) if causal else scores, dim=-1)
    return torch.einsum("ths,shv->thv", p, V)             # [total_q, H_q, d_V]

def ref_mla_decode_absorb(c_q, c_kv, k_rope, W_abs, W_rope_proj, W_UV, positions):
    # c_q [B, r_Q] (no head axis); W_abs [H_q, r_Q, r_KV]; W_rope_proj [H_q, r_Q, d_rope];
    # W_UV [H_q, r_KV, d_V] — per head (§0)
    q_abs  = torch.einsum("br,hrk->bhk", c_q, W_abs)          # [B, H_q, r_KV]
    q_rope = torch.einsum("br,hrd->bhd", c_q, W_rope_proj)    # [B, H_q, d_rope]
    q_rope = apply_rope(q_rope, positions)                    # rotate at the current position
    scores = scale * (torch.einsum("bhk,sk->bhs", q_abs, c_kv)
                      + torch.einsum("bhd,sd->bhs", q_rope, k_rope))
    p   = softmax(scores, dim=-1)                             # [B, H_q, S]
    acc = torch.einsum("bhs,sk->bhk", p, c_kv)                # latent-space accum [B, H_q, r_KV]
    return torch.einsum("bhk,hkv->bhv", acc, W_UV)            # [B, H_q, d_V]
```

The decode reference accumulates in latent space and applies `W_UV` last,
**mirroring the kernel's structure** (§4) rather than the mathematically equivalent
per-token expansion. Keeping the reference's contraction order aligned with the
kernel's is what makes the tolerance below meaningful: a parity gate that reduces in
a different order absorbs part of the error budget it is supposed to be measuring.

Tolerance gate: `max_abs ≤ 4e-2` bf16 (matching the existing unified attention gate).
The sweep should span the full `kv_len` range of §8.2 (512 … 32768), for two reasons
that are properties of this design rather than of the tolerance: long `kv_len` is what
exercises the 3D split-KV reduce with more than one segment per query row, and it is
where the fp32 latent accumulator (§4) has the deepest reduction chain. This gate does
**not** discriminate a wrong softmax `scale` (§0) by `kv_len` — a $1/\sqrt{576}$ vs
$1/\sqrt{192}$ error is a 1.73× temperature change that should fail at every length;
the defence against it is that reference and kernel take the same host-supplied
`scale`, not the sweep range.

### 8.2 Benchmark shapes

Four shape files: `mla_shapes.json` under
`library/benchmarks/{gfx942,gfx950}/attention/decode/` and `mla_prefill_shapes.json`
under the matching `attention/prefill/` directories. Decode sweep (`seqlen_q=1`):
kv_len in {512, 1024, 2048, 4096, 8192, 16384, 32768} for DeepSeek V3/R1, GLM-5
(pending), Kimi-K2. Prefill sweep: seqlen_q = seqlen_k ∈ {512, 1024, 2048, 4096,
8192}. All four files use
`block_size: 16` (the repo default for paged KV); the `Bk` values in §5 are kernel
tile sizes covering 1–2 blocks each, not block sizes (see the note in §5.2).

**Dtype coverage and harness limits.** The two gfx942 files are **bf16 only** —
Phase 1 (§6). The two gfx950 files additionally carry `dtype: "fp8_e4m3"` entries for
Phase 2; they are **specification, not runnable input for Phase 1**:

- The existing decode harness (`benchmarks/gfx*/attention/decode/benchmark_decode_live.py`)
  maps dtype with `torch.bfloat16 if shape.dtype == "bf16" else torch.float16`, so any
  non-`bf16` string — including `fp8_e4m3` — is **silently** benchmarked as fp16.
- The prefill harnesses (`benchmark_prefill2d_live.py`, both arches) behave differently:
  they never read a `dtype` string at all. They load ndjson `UAShape` records through
  `_ua_shape_utils.load_shapes` and key dtype off the torch dtype strings `q_dtype` /
  `k_dtype` — `filter_prefill_2d` matches `"torch.bfloat16"` / `"torch.float16"` and
  **drops** everything else rather than defaulting it — with
  `is_fp8 = "float8" in shape.k_dtype` on both arches (gfx942 then skips the shape,
  since fp8 is unsupported there). A `dtype: "fp8_e4m3"` key in the prefill JSON is
  therefore inert for those harnesses, not silently downgraded.
- Independently, none of the four files is loadable by the current harnesses.
  Decode: `load_decode_shapes` requires a `head_size` key per shape (no default), which
  MLA does not have a single value for (`d_nope + d_rope = 192` for the score,
  `r_KV + d_rope = 576` for the decode-absorb memory layout, `d_V = 128` for the
  output). Prefill: the blocker is the file format — `_ua_shape_utils.load_shapes`
  parses **ndjson** (one `UAShape` record per line, with `q_shape` / `k_shape` /
  `block_table_shape` / `softmax_scale` / `q_dtype`) and silently skips lines it cannot
  decode, so a nested-object `.json` yields zero shapes rather than an error. Wiring MLA
  shapes into a runner is part of the kernel deliverable (§9), not a prerequisite of
  this doc.

Consequence: a Phase-1 bf16 run must select the `bf16` entries explicitly. Do not run
the fp8 entries against a bf16/fp16 harness and read the result as an fp8 number. Both
constraints are repeated as `_dtype_note` / `_harness_note` inside the JSON files so
they travel with the data.

### 8.3 Parity baselines

The parity harness in `library/builders/mla/` follows the same three-table
methodology as `library/builders/gfx950/attention/README.md`: three apples-to-apples
lanes — `auto` vs `auto`, `2d` vs `2d`, `3d` vs `3d` — each comparing the reference
backend against the rocKE kernel *within* that lane, so a selector difference is never
read as a kernel difference. Two external baselines should be included for meaningful
comparison:

| Baseline | Source | What to compare |
|---|---|---|
| **AITER Triton MLA** (`ROCM_AITER_TRITON_MLA`) | `aiter.ops.mla` | Primary comparison; AITER reports this as best on gfx942 |
| **TileLang MLA** | `github.com/tile-ai/tilelang` | Open-source; upstream reports 95% of AITER ASM on gfx942 in ~80 lines (§10.3); transparent tiling strategy |

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

**Dual-engine definition of done.** Every row above is a *new instance*, so none is
complete on the Python side alone. Each kernel must land its Python builder under
`library/kernels/<arch>/` **and** its C++ engine mirror under `platform/cpp/` in the
same change, be wired into `ATTENTION_REGISTRY` and into the parity emit cases under
`library/tests/parity/`, and turn the differential `.ll` gate green at both flavors:

```bash
ROCKE_LLVM_FLAVOR=llvm20 python platform/tools/check_byte_identity.py
ROCKE_LLVM_FLAVOR=llvm22 python platform/tools/check_byte_identity.py
```

Pin the flavor on **both** runs. An unset `ROCKE_LLVM_FLAVOR` is not llvm20: it
auto-resolves comgr → `torch.version.hip` → `/opt/rocm/.info/version` → `llvm22`
(`platform/python/rocke/core/lower_llvm.py:200`, `_detect_llvm_flavor`), so on a
ROCm 7.2+ box an unset run
and the `llvm22` run are the same run and the llvm20 intrinsic declares are never
covered.

Any new IR op or spec trait required by the latent-space accumulator, the decoupled
`hdim_kv` / `hdim_out` descriptor, the decode `BLOCK_H` head remap (§4), the *separate*
prefill `BLOCK_H` head-block grid dim0 (§3 — an independently-sized second spec knob,
not the §4 one), or the XCD-aware grid mapping (§4) must be mirrored in
`platform/python/rocke/core/lower_llvm.py` and its C++ twin together, and
any intentional `.ll` change must be golden-re-blessed from a green state with the diff
reviewed. This docs-only spike carries no such obligation; the implementation PRs do.

---

## 10. State of the art — public MLA kernel implementations

This section documents the public implementations reviewed during the design spike.
They inform the architectural decisions in §2–§5; nothing here is prescriptive for
rocKE implementation.

### 10.1 AITER (AMD Inference Toolkit)

The canonical ROCm MLA reference. Lives at `github.com/ROCm/aiter`; integrated into
vLLM and SGLang as the AMD attention backend.

**Decode kernel** (`mla_decode_fwd`):
- Hand-written assembly (not open-sourced in detail).
- Implements MQA in latent space (head_dim=576) exactly as described in §2.4.
- Absorbed weights (`w_kc = W_abs`, `w_vc = W_UV`) pre-computed at model load.
- Paged KV cache with split-KV scheduling across CUs.
- Reported **17× speedup** over non-absorbed naive MLA on gfx942
  (`rocm.blogs.amd.com/software-tools-optimization/aiter-mla/`).
- On gfx942: `ROCM_AITER_TRITON_MLA` is reported to slightly outperform the ASM
  backend. Directional only — no public benchmark table to cite, so treat the two as
  comparable on gfx942 and measure locally rather than planning around a margin.
- On gfx950: the ASM backend is reported to match or beat Triton MLA — also
  directional, with no public benchmark table to cite.

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

### 10.2 FlashMLA (DeepSeek)

`github.com/deepseek-ai/FlashMLA` — the reference MLA decode kernel from DeepSeek
(CUDA/Hopper only, not ported to ROCm). Architecturally relevant:

- Confirms the MQA-in-latent-space approach (§2.4): Q shape `[Sq, N, r_KV+d_rope]`,
  KV cache `[Skv, 1, r_KV+d_rope]` with paged block size 64.
- Uses TMA on Hopper for async KV load; AMD equivalent would be `raw_ptr_buffer_load_lds`
  with `async_buffer_load_lds_addr` (already in `attention_tiled_2d.py`).
- Split-KV across SMs with partial-result merging (same structure as
  `attention_tiled_3d`).
- Fuses KV buffer writes with decode attention; upstream reports roughly +12% from
  removing the PyTorch-side overhead this eliminates.
- Upstream reports a ceiling of ~3000 GB/s memory-bound and ~660 TFLOPS
  compute-bound on H800.

> The two figures above are self-reported in the FlashMLA repo
> (`github.com/deepseek-ai/FlashMLA`), not independently verified, and are Hopper
> numbers on a different memory system. They are quoted for order of magnitude and
> for the *shape* of the optimization (fusing the KV write), not as targets for
> gfx942/gfx950.

### 10.3 TileLang MLA

`github.com/tile-ai/tilelang` — open-source composable tiled DSL with a complete,
readable MLA kernel for gfx942.

- Reported at **95% of AITER assembly performance** on gfx942, **1.98× over Triton**
  MLA and **3.76× over PyTorch** baseline — upstream's own figures
  (`tilelang.com/deeplearning_operators/deepseek_mla.html`), not independently
  reproduced.
- ~80 lines of Python, fully open-source. The tiling strategy is the most
  transparent public reference for MLA on gfx942.
- Handles gfx942's 64 KB LDS (vs Hopper's 228 KB shared memory) explicitly; tile sizes not
  constrained to multiples of 64; swizzling for bank conflicts handled automatically.
- Recommended as a **parity baseline** in addition to AITER Triton MLA (§8.3).

### 10.4 CK (Composable Kernels)

CK has no dedicated MLA kernel (no `mla`, `kv_lora_rank`, `absorbed` code paths),
but it contains relevant MLA-adjacent infrastructure added explicitly for DeepSeek
V3:

**`(hdim_q=192, hdim_v=128)` — officially supported:**
- Commit `4399ad79029` (March 2025): "support hdim=192/128 pair for deepseekv3".
- Registered as a supported `[hdim_q, hdim_v]` pair for fp16/bf16/fp8/fp8bf16/bf8 in
  `dispatcher/codegen/fmha/fmha_arch_specs.json` (`supported_hdims`). **Instantiation is
  narrower than registration:** in the `01_fmha` example codegen only plain
  `fmha_fwd.py` carries a `(192,128)` tile. `fmha_fwd_splitkv.py` has no 192 entry at
  all, `fmha_pagedkv_prefill.py:587` has its `"192"` tile **commented out**, and
  `fmha_batch_prefill.py` carries only 128/256 — so neither split-KV nor either prefill
  generator emits a `(192,128)` kernel today.
- This is absorbed-form MLA: Q/K head_dim = qk_nope(128) + qk_rope(64) = 192,
  V head_dim = v_head_dim = 128. The CK FMHA kernel with these dims is a direct
  viable substrate for the MLA prefill kernel without fusing the latent
  expansion — the expansion is a separate prior GEMM and CK handles the attention.
- Restrictions at this pair are per-variant, and the paged conclusion is stronger than
  a per-hdim gate. `fmha_fwd.py:820` (`check_hdim`) skips `(192,128)` only when
  `bias != "no"` or `dropout == "t"` — **LSE is available** there — but `fmha_fwd.py`
  has no paged-KV pipeline at all. The `(192,128)` guards in
  `fmha_pagedkv_prefill.py:703-706` (`bias != "no" or lse == "t"`) and
  `fmha_batch_prefill.py:846-853` (same, plus dropout) are **dead code today**, because
  neither generator has a 192 tile to reach them. On the dispatcher codegen the paged
  family *is* enumerated over `supported_hdims`, so `(192,128)` is reachable there — but
  `get_pagedkv_pipelines` (`dispatcher/codegen/fmha/instance_gen.py:1030`) passes
  `lse="f"` positionally for **every** `qr_pagedkv` spec. CK therefore emits no paged-KV
  forward kernel with LSE at any head dim: chunked-prefill via `softmax_lse` is blocked
  on the path MLA needs (§3), and bias is unavailable on every variant at this pair.

**`MLA_H128xH576_Asymmetric` test case — planned but not instantiated:**
- `tile_engine/ops/fmha/ck_fmha_testing_matrix.yaml` (added 2026-05-17, `a1834d2b22`) contains
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

**Bottom line for rocKE:** CK's (192,128) FMHA is a ready-made *measurement baseline*
for MLA prefill in the "separate expansion" mode — not a shippable rocKE instance: per
§3, `lower_cktile.py` accepts no attention spec, so a wrapped CK kernel yields no
`KernelDef` and no `ATTENTION_REGISTRY` candidate. The prefill implementation should
prototype wrapping it early and measure the in-loop fusion approach against it; the
custom rocKE kernel is still the deliverable (§9).

### 10.5 FlashInfer ROCm MLA

`github.com/ROCm/flashinfer` — ROCm port of FlashInfer. Decode via
`trtllm_batch_decode_with_kv_cache_mla`:

- KV cache layout: `[num_pages, page_size, r_KV + d_rope]` — confirms the
  concatenated layout described in §4.
- Decode kernel: 128-head MQA in latent space, reusing `c_KV` as both K and V for
  the score + value step (same as §2.4).
- CK backend for gfx942 prefill; supports gfx942 and gfx950.

### 10.6 SGLang weight absorption

`github.com/sgl-project/sglang` — the weight absorption technique (§2.4) was first
deployed at scale in SGLang (PR #905, #1138). Key implementation details:

- Absorbed weights stored as `w_kc` and `w_vc` at model load; the SGLang MLA
  module never calls `W_UQ` or `W_UK` during serving.
- Prefill crossover threshold ~171–228 tokens (§2.5): below threshold → Triton
  absorbed decode; above → materialize K/V → standard flash prefill.
- Open issue (#4615): avoid materializing `w_kc`/`w_vc` to save GPU memory — still
  open, relevant to the hipDNN graph tensor representation question in §7.5.

---

## 11. References

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

**External (state of the art — see §10):**
- AITER MLA: `github.com/ROCm/aiter` / `rocm.blogs.amd.com/software-tools-optimization/aiter-mla/`
- FlashMLA (DeepSeek, CUDA): `github.com/deepseek-ai/FlashMLA`
- TileLang MLA on gfx942: `github.com/tile-ai/tilelang` / `tilelang.com/deeplearning_operators/deepseek_mla.html`
- FlashInfer ROCm: `github.com/ROCm/flashinfer`
- SGLang weight absorption: PR #905, #1138 at `github.com/sgl-project/sglang`
- FlyDSL: `github.com/ROCm/FlyDSL`
- CK FMHA (192,128): `github.com/ROCm/composable_kernel` — commit `4399ad79029`; `include/ck_tile/ops/fmha/`, `dispatcher/codegen/fmha/fmha_arch_specs.json`, `tile_engine/ops/fmha/ck_fmha_testing_matrix.yaml`
