[← MLA design doc index](../DESIGN.md)

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
# Pre-kernel (separate device kernel, once per prefill call — not in the flash loop),
# batched over heads exactly as §4's decode pre-step:
  2. apply W_UQ GEMM: q_latent [total_q, r_Q] × W_UQ[h] [r_Q, d_nope+d_rope]
                      → q[h] [total_q, d_nope+d_rope]
  3. split q[h] → q_nope[h] [total_q, d_nope], q_rope[h] [total_q, d_rope];
     rotate q_rope[h] at `positions`   # `positions` is a pre-kernel input, not a
                                       #   flash-kernel one

grid:      (H_q / BLOCK_H, total_num_q_blocks, 1)   # H_k = 1, so dim0 carries HEAD blocks
workgroup: (64 * num_warps, 1, 1)                   # BLOCK_H sized at implementation (§9)

for each (head block, q_block):
  1. load q_nope/q_rope tile → LDS               # [Bq, BLOCK_H, d_nope+d_rope] — the
                                                 #   pre-kernel's output. 6 KB per head at
                                                 #   Bq = 16, NOT the 48 KB [Bq, r_Q]
                                                 #   q_latent tile the fused form needed
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

**Step 2 runs as a separate pre-kernel, not inside the flash loop.** The W\_UQ GEMM is
the main structural addition over standard flash attention, and it is specified the same
way as the decode pre-step (§4): a standalone GEMM
(`[total_q, r_Q] × [r_Q, d_nope+d_rope]` per head) that reuses existing GEMM
infrastructure and hands the flash kernel an expanded `q` directly. The op's inputs stay
`q_latent` + `W_UQ` because the op spans both kernels; only the *internal* split is
fixed here.

> **The in-LDS fused alternative is rejected — on weight traffic, not on LDS.** LDS alone
> does not settle it: the 48 KB `q_latent` tile (§5.1) is large, but it is live only
> before the KV loop starts, so the smem pool's live-interval reuse hands its bytes to
> the KV tile for free (§5.1), and §5.1's split-r variant fits inside the budget at
> 26–52 KB per slice. The decisive term is `W_UQ` re-reads.
> Fused, every query block re-streams `W_UQ` (72 MiB across all heads), so a prefill call
> moves `(total_q / Bq) × 72 MiB` — ~36 GiB at `total_q = 8192, Bq = 16`. A pre-kernel
> GEMM reads `W_UQ` once and amortizes it over all `total_q`: ~72 MiB. Nothing
> recoverable in the LDS budget closes a ~500× gap.
>
> Steps 2–3 above are therefore the pre-kernel's body and steps 1 and 4–11 are the flash
> kernel's. The numbering runs across both because the two kernels are one *op* (§7.1),
> not because they are one dispatch — whether the graph sees one fused op or two is an
> open question, and the same one §7.4 already raises for the decode pre-step. It must be
> answered the same way for both.

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
> yields no `KernelDef`, no registry candidate and no golden-IR coverage. What the pair
> costs at (192,128) differs by variant: plain `fmha_fwd` drops
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

### 3.1 Full-prompt prefill: the materialize path

**Same op** (`mla_prefill_fwd`, §7.1), different internal strategy; §2.5's threshold
selects between this and the §3 flash loop at launch time. This path is **not** a new
attention kernel — it is two pre-GEMMs feeding the *existing* rocKE unified attention at
an asymmetric head dimension. Its cost is therefore concentrated in the spec layer, not
in a tiling.

```
# Pre-kernel 1 — the W_UQ GEMM, shared verbatim with §3 step 2.
  q[h] = q_latent · W_UQ[h]  → [total_q, H_q, d_nope + d_rope]; split, rotate q_rope

# Pre-kernel 2 — the KV expansion, PER HEAD (§0):
  K_nope[:, h, :] = c_KV · W_UK_K[h]              # [S_k, H_q, d_nope]
  V[:, h, :]      = c_KV · W_UV[h]                # [S_k, H_q, d_V]
  K_exp[:, h, :]  = concat(K_nope[:, h, :], K_rope)   # [S_k, H_q, d_nope + d_rope]
                                                  #   K_rope is head-shared (§2.1) and is
                                                  #   BROADCAST across h, not expanded

# Attention — standard rocKE unified attention, no MLA-specific code:
  hdim_q = d_nope + d_rope = 192,  hdim_v = d_V = 128,  H_k = H_q,  causal
```

> **After expansion this is MHA, not MQA.** The latent is shared across heads but its
> expansion is not (§0), so `K_exp` and `V` carry a full head axis and
> $H_k = H_q = 128$. Any design that reuses a single head-shared `[S_k, 1, ·]` K/V cache
> here is solving a different problem — see the trap list in §9.

**Materialized footprint, and why the path is regime-limited.** At
$S_k = 8192,\ H_q = 128$, bf16, per layer per sequence:

```
K_exp : 8192 x 128 x 192 x 2 B = 384 MiB
V     : 8192 x 128 x 128 x 2 B = 256 MiB
total                          = 640 MiB
```

and at the $S_k = 32768$ upper bound of the §8.2 sweep, 4× that:

```
K_exp : 32768 x 128 x 192 x 2 B = 1536 MiB
V     : 32768 x 128 x 128 x 2 B = 1024 MiB
total                           = 2560 MiB = 2.5 GiB
```

This is the concrete form of §2.5's $1/S_q$ argument: the footprint is set by $S_k$ and
$H_q$ alone and does not shrink as the query count shrinks. It is affordable when
$S_q \approx S_k$ and ruinous under chunked prefill — which is why §2.5's dispatch gates
this path on a scratch budget rather than on $S_q$.

**Enabling work — the spec layer, and it is the whole cost of this path.** MLA's
materialized form is asymmetric ($\texttt{hdim\_q} = 192$, $\texttt{hdim\_v} = 128$) and
the current stack collapses or rejects that at the five sites §7.2 enumerates. Those
five edits *are* this path's deliverable; there is no new tiling to write.

> **Do not pad the head dimension to 256 to dodge the gate.** Padding
> $192 \to 256$ on the score side and $128 \to 256$ on the value side makes the shape
> admissible to `UNIFIED_HEAD_SIZES` without any spec-layer change, and it is the
> obvious shortcut. It permanently discards **25% of the QK MFMA lanes and 50% of the
> PV lanes**, on a path whose entire justification is that it is cheaper than the flash
> loop. Admit 192 per §7.2 instead — the merit argument is there, and FlashInfer's MLA
> prefill mode runs natively at `(head_dim_k, head_dim_v) = (192, 128)` (§10.5), so the
> asymmetric shape is the one the ecosystem already targets. This shortcut was taken in
> an early implementation draft and measured well below the AITER Triton baseline; treat
> that as a recorded negative result, not as a starting point.

---

