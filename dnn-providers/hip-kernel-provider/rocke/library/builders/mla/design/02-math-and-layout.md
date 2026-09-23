[← MLA design doc index](../DESIGN.md)

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
> 5.3 TB/s HBM gives 247 flop/byte, 17 TB/s Infinity Cache gives 77. Both bandwidths are
> AMD's published **theoretical peaks** for MI300X (192 GB HBM3 at 5.3 TB/s; 256 MB MALL
> at 17 TB/s). Measured Infinity Cache throughput is below peak — the subsystem is high
> latency (~218 ns) as well as high bandwidth — so 77 is an upper bound on the MALL
> balance and the real knee, if MALL is the tier that binds, arrives at a lower
> `BLOCK_H` than the table suggests.
>
> **Which balance applies is unmeasured, and it changes the conclusion.** §4's estimate
> prices the *compulsory* first read of each token against HBM and the redundant
> re-reads — the ones head-batching removes — against MALL (~17 TB/s), on the argument
> that the KV working set is MALL-resident but exceeds one XCD's L2. Against MALL the
> kernel approaches the knee near `BLOCK_H = 32`; against HBM it never does. Both columns are given
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

Two axes are in play here and this doc keeps them separate. The **regime** is a
scheduler mode, fixed by the shape ($S_k \gg S_q$ vs $S_k \approx S_q$); the
**strategy** is the kernel choice (in-loop expansion vs materialize), and it is what
the ~200-token crossover is about. The regimes are:

| Regime | $S_q$ | $S_k$ | Strategy |
|---|---|---|---|
| **Chunked prefill** | the scheduler's chunk (commonly 512–2048) | $S_k \gg S_q$; grows to the full context | Latent expansion inside flash loop (**§3 kernel**); **§3.1 is admissible only below the footprint bound below** — which side wins is **unmeasured**, see §8.2 family 2 |
| **Full-prompt prefill** | the whole prompt | $S_k \approx S_q$ | Materialize $K, V$ once → standard flash attention (**§3.1**) |

**Both regimes are in scope.** §9 carries deliverables for each; neither is optional,
and the dispatch heuristic between them is a launch-time decision in
`library/dispatch/attention/`, not a kernel property.

> **"Short prefill" would be a misnomer — read the first row as *chunked* prefill.**
> The regimes are separated by the $S_q : S_k$ relationship, not by $S_q$ alone, and
> $S_q$ is decoupled from context length: under chunked prefill the scheduler fixes
> $S_q$ at the chunk size — commonly 512–2048, i.e. *above* the ~200 strategy threshold —
> while $S_k$ grows to the full context. That combination is the *worst* case for
> materialization and the reason
> the first row exists. Materializing costs
> $O(S_k \cdot r_{KV} \cdot H_q \cdot (d_{\text{nope}} + d_V))$ of expansion to serve
> $O(S_q \cdot S_k)$ of attention, so the expansion overhead per query token scales as
> $1/S_q$ — at a 512-token chunk against 64 K of context you pay the full per-head
> expansion of 64 K tokens to attend with 512 queries. The $S_q = S_k$ shapes of §8.2
> are the *best* case for materialization by construction, because there the expansion
> amortizes over the largest query count the geometry permits. Do not read a
> full-prompt-prefill measurement as evidence about the chunked regime.
>
> Chunked prefill is the production-dominant path for long-context serving, and the
> regime is load-bearing enough that AITER ships hand-written assembly restricted to
> $S_q < 160$ for it (§10.1). §3's input table carries `cu_seqlens_k` separately from
> `cu_seqlens_q` for exactly this reason.

> **The threshold is a citation, not a measurement.** ~200 comes from SGLang on Hopper
> (§10.6) and is repeated here because it is the only published figure; nothing in this
> doc measures it on gfx942 or gfx950, and the "approximately hardware-independent"
> claim is SGLang's, not ours. The independent evidence is directional only: Yun et al.
> (§11) report the prefill attention block 2.02× *worse* with absorption at
> $B = 1, L = 4096$, and the decode block 119× *better* at $B = 256, L = 4096$ —
> confirming that a crossover exists and which way it runs, but fixing no token count.
> It is also a figure for one *shape family*, not for $S_q$ in isolation: SGLang measured
> it where $S_k \approx S_q$, and the $1/S_q$ argument above says the expansion cost
> scales with $S_k$ and amortizes as $1/S_q$ — so the crossover is a function of the
> ratio $S_q / S_k$, and importing a square-shape figure as an $S_q$-only bound is the
> same inference this doc warns against one paragraph up.
> **Measure it before it is baked into the dispatch heuristic**; §8.2's chunked shapes
> are the sweep that does so.

> **The dispatch needs a footprint bound, not just an $S_q$ threshold.** §3.1's
> materialized working set is
> $S_k \cdot H_q \cdot (d_{\text{nope}} + d_{\text{rope}} + d_V) \cdot
> \texttt{sizeof(dtype)}$ per layer per sequence — set by $S_k$ and $H_q$ alone, and
> **independent of $S_q$**. A dispatch keyed on $S_q$ alone therefore sends
> $(S_q, S_k) = (512, 32768)$ to materialize and asks for **2.5 GiB** of scratch to serve
> 512 query tokens (§3.1). Chunked shapes stay on the §3 in-loop path regardless of
> $S_q$ whenever that product exceeds the scratch budget; the $S_q$ threshold only
> arbitrates below it. The budget bound is a hard admissibility gate, the threshold is a
> tunable — §9 row 4 implements both.

The materialization path (`c_KV · W_UK → K, V`, then standard flash attention over the
expanded tensors) incurs a separate GEMM launch; the in-loop path fuses expansion with
attention but is memory-bandwidth-bound on `W_UK` at large tile counts.

> **The materialize path does not "just reuse" the existing kernels.** MLA's
> materialized form is asymmetric — `hdim_q = d_nope + d_rope = 192`,
> `hdim_v = d_V = 128` — and the current rocKE attention stack supports neither.
> `AttentionRequest` already carries `hdim_q` and `hdim_v` separately, but everything
> below it collapses or rejects them: `_request_errors` rejects `hdim_q != hdim_v`,
> `_problem()` collapses them (`head_size=int(req.hdim_q)`) into a
> `UnifiedAttentionProblem` that has a single `head_size`, `AttentionSpec` likewise
> carries one `head_size` (and bakes it into `kernel_name()` as `hd{head_size}`),
> `UNIFIED_HEAD_SIZES = (64, 128, 256)` excludes 192 by set membership, and — separately
> from that constant — the per-arch admission gates (`supports_tiled_2d` /
> `supports_tiled_3d` in `library/kernels/{gfx942,gfx950}/attention_tiled_{2d,3d}.py`)
> re-check `head_size not in (64, 128, 256)` against a *hardcoded literal*, so widening
> `UNIFIED_HEAD_SIZES` alone does not widen them
> (`library/dispatch/attention/common.py`,
> `library/kernels/common/attention_unified.py`). Reusing `UnifiedAttention` here
> therefore needs the split carried through the problem, spec and descriptor layers and
> a widened head-size gate in *both* places — spec-layer work (§9). The request layer is
> already shaped for it; the layers underneath are not. The CK-FMHA `(192,128)` alternative in §3 sidesteps that gap for *measurement*
> only — it yields no shippable rocKE instance — so that prototype should come first
> but does not remove the spec-layer work.

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

