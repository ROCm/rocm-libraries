# HSTU Attention Backward — Algorithm

This document derives the HSTU (Hierarchical Sequential Transduction Unit)
attention **backward** pass from the math, independent of the kernel
implementation. Read [`ARCHITECTURE.md`](ARCHITECTURE.md) next for how the three
gradient kernels are tiled and scheduled on gfx950, and [`README.md`](README.md)
for the optimization case study and measured results.

The Rocke kernels are in
`library/kernels/common/hstu_attention_bwd.py`; the shared math helpers
(SiLU + derivative, HSTU mask) are in
`platform/python/rocke/helpers/hstu_attention.py`. The reference is the FlyDSL
kernel on AITER branch `dlejeune/flydsl_hsta_bwd`
(`aiter/ops/flydsl/hstu_attention_kernels.py`).

## 1. Forward recap

HSTU attention is **not** softmax attention. For a single sequence of length `N`
(jagged/packed across a batch via `seq_offsets`), per head:

```
S[q, kv]   = alpha * (Q[q, :] · K[kv, :])          # scaled score, alpha is a host scalar
P[q, kv]   = mask(q, kv) * silu(S[q, kv])          # SiLU gate, NOT softmax
O[q, d]    = (1/N) * sum_kv P[q, kv] * V[kv, d]     # 1/N normalization (no row-sum denom)
```

Key differences from FMHA/softmax attention:

- **No online softmax, no LSE, no row max/sum.** The nonlinearity is
  `silu(x) = x * sigmoid(x)`, applied elementwise to the scaled score. There is
  no per-row normalization; the output is scaled by the constant `1/N`.
- **`alpha` is folded into the score** before the gate; `1/N` is folded into the
  output (the forward hoists it to the O epilogue).
- The mask is HSTU-specific (see §4): causal on shifted logical ids, with
  optional sliding window, contextual prefix, and target-tail handling.
- Nothing is stashed by the forward — the backward **recomputes** `S` (and the
  sigmoid) from `Q, K`.

Dtype is `{f16, bf16}`; all matmuls accumulate in f32; `silu`/`sigmoid` use the
fast `exp2`/`rcp` path (not IEEE `exp`), matching the forward so numerics agree.

## 2. Gradients

Let `dO` be the upstream gradient of `O`. With `sc = alpha * S`,
`sigma = sigmoid(sc)`:

```
silu(sc)   = sc * sigma
silu'(sc)  = sigma * (1 + sc * (1 - sigma))          # d silu / d sc  (locked by a unit test)
```

The three gradients (each a separate kernel — see §5 for *why*):

```
# dV: reduce over the query index
dV[kv, d]  = (1/N) * sum_q  P[q, kv] * dO[q, d]
           = (1/N) * sum_q  ( mask * silu(sc) )[q, kv] * dO[q, d]

# intermediate: dA = dO · Vᵀ (contract the hidden dim), then the gated dS
dA[q, kv]  = sum_hd dO[q, hd] * V[kv, hd]
dS[q, kv]  = mask(q, kv) * (1/N) * silu'(sc[q, kv]) * dA[q, kv]

# dK: reduce over the query index
dK[kv, hc] = alpha * sum_q  dS[q, kv] * Q[q, hc]

# dQ: reduce over the key index
dQ[q, hc]  = alpha * sum_kv dS[q, kv] * K[kv, hc]
```

Scale placement (matches FlyDSL exactly):

- `alpha` lives inside the score for `silu`/`silu'`, and is applied **once more**
  in the `dK`/`dQ` epilogue (the `alpha` factor of `d sc / d Q,K`).
- `1/N` is applied in the `dV` epilogue (for `P`) and inside `dS` (for `silu'`).

## 3. Reductions and ownership

Each gradient reduces over a different index, which dictates data ownership:

| grad | formula | reduces over | one program owns | streams |
|------|---------|--------------|------------------|---------|
| dV   | `Aᵀ·dO`, `A = mask·silu/N` | query `q`  | a KV tile | Q tiles |
| dK   | `alpha·dSᵀ·Q`             | query `q`  | a KV tile | Q tiles |
| dQ   | `alpha·dS·K`              | key `kv`   | a Q tile  | KV tiles |

Because each output row is written by exactly one program (single-writer), the
backward needs **no atomics** — all stores are deferred to the epilogue.

## 4. HSTU mask

The mask is **not** plain causal: it compares shifted *logical ids* and keeps the
diagonal explicitly. With `to_id(x)` applying the contextual shift and target
clamp:

```
dist = to_id(q) - to_id(kv)
keep = (q == kv) | (dist > 0)                        # diagonal OR strict lower (causal)
if window:      keep &= (dist <= max_attn_len)       # sliding window
if contextual:  keep |= (to_id(q) == 0) & (to_id(kv) < max_id)   # prefix opener
keep &= (q in seq) & (kv in seq)                     # jagged bounds
```

- **contextual**: `to_id(x) = clamp(x - (contextual_seq_len - 1), >= 0)`; the
  prefix query (`id 0`) attends the whole contextual prefix above its diagonal.
- **targets**: `max_id = seq_len - contextual + 1`, clamped down by
  `num_targets`; ids above `max_id` clamp to `max_id` (target tokens share the
  tail id). Order matters: contextual shift **before** the target clamp.

`Rocke` factors these into `hstu_mask_keep` / `hstu_to_mask_id` /
`hstu_silu_and_grad` in `helpers/hstu_attention.py`, unit-agnostic so both the
scalar and MFMA kernels share one source of truth.

## 5. Why three single-family kernels

A fused kernel carrying both the `dV` (over hidden) and `dK` (over head) — or all
three — accumulator families needs ~48 MFMA accumulators (~336 AGPR-equivalent
at `block_m=192`). On CDNA4 the MFMA accumulators live in the **unified VGPR
file** (AGPR=0), so a fused kernel doubles VGPR pressure and halves occupancy,
pinning it at 1 wave/SIMD (on-chip-latency-bound). Splitting into three
single-family kernels keeps one accumulator family each (~8/16 accumulators),
reaching 2–4 waves/SIMD. The cost is recomputing `S` in each kernel — a
deliberate **recompute-over-occupancy** trade, consistent with FlyDSL.

## 6. Jagged (packed varlen) layout

Tensors are packed rank-3 `(total_tokens, num_heads, dim)`:

- `q, k`:  `(L, H, head_dim)`
- `v, dO, dV/dK/dQ`:  `(L, H, hidden_dim)`
- `seq_offsets`: `(B+1,)` int32 prefix sum; sequence `b` is `[seq_offsets[b],
  seq_offsets[b+1])`.
- `num_targets`: `(B,)` int32 (or a dummy `(1,)` when absent).
- `perm`: `(B,)` int32 optional group-aware sort-by-length remap (dummy `(1,)`
  when disabled).

Linear element index: `token * (H * dim) + head * dim + col`. Every token index
is offset by `seq_start`; bounds use the per-sequence `seq_len`, never the global
`N`.
