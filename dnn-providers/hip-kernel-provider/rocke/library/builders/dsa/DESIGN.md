# DSA kernel family — design doc

> **Scope.** DeepSeek Sparse Attention (DSA) for gfx942 and gfx950. A lightning indexer scores earlier tokens, a large-k top-k keeps the best k, and MLA attention runs over only that selected subset, with the GLM IndexShare variant reusing a produced selection across layers. This document specifies what the kernels compute and why they are shaped the way they are. It is a design spike, not an implementation and not a tuning history, and it carries no measurements. Latency is recorded in the internal perf repository per repository compliance. The Definition of Done is this document approved, with the hipDNN integration developed together with the hipDNN team. It depends on the MLA design, not on MLA's kernels, which are not yet built.

---

## Contents

- [0. Notation](#0-notation)
- [1. DSA geometry and model variants](#1-dsa-geometry-and-model-variants)
  - [1.1 Model table](#11-model-table)
  - [1.2 Relationship to MLA geometry](#12-relationship-to-mla-geometry)
- [2. Math and data layout](#2-math-and-data-layout)
  - [2.1 The three-stage pipeline](#21-the-three-stage-pipeline)
  - [2.2 Lightning indexer scoring](#22-lightning-indexer-scoring)
  - [2.3 Top-k selection](#23-top-k-selection)
  - [2.4 MLA over the subset](#24-mla-over-the-subset)
  - [2.5 IndexShare](#25-indexshare)
  - [2.6 Online softmax over the selected set](#26-online-softmax-over-the-selected-set)
  - [2.7 Prefill vs decode forms](#27-prefill-vs-decode-forms)
- [3. Lightning indexer kernel specification](#3-lightning-indexer-kernel-specification)
  - [3.1 Inputs and outputs](#31-inputs-and-outputs)
  - [3.2 Kernel structure](#32-kernel-structure)
  - [3.3 Why it is a separate kernel from attention](#33-why-it-is-a-separate-kernel-from-attention)
- [4. Top-k selection specification](#4-top-k-selection-specification)
  - [4.1 The primitive](#41-the-primitive)
  - [4.2 Output contract](#42-output-contract)
- [5. Subset attention (MLA over subset) specification](#5-subset-attention-mla-over-subset-specification)
  - [5.1 block_table gather](#51-block_table-gather)
  - [5.2 Scatter-read vs gather-to-contiguous scratch](#52-scatter-read-vs-gather-to-contiguous-scratch)
  - [5.3 Prefill form](#53-prefill-form)
  - [5.4 Decode-absorb form](#54-decode-absorb-form)
- [6. IndexShare handling](#6-indexshare-handling)
- [7. Per-arch tiling and budget (gfx942 / gfx950)](#7-per-arch-tiling-and-budget-gfx942--gfx950)
  - [7.1 Indexer tiling and LDS/register budget](#71-indexer-tiling-and-ldsregister-budget)
  - [7.2 Top-k reduction budget](#72-top-k-reduction-budget)
  - [7.3 Gather cost and KV-traffic budget for subset attention](#73-gather-cost-and-kv-traffic-budget-for-subset-attention)
  - [7.4 gfx942](#74-gfx942)
  - [7.5 gfx950](#75-gfx950)
- [8. Dtype plan](#8-dtype-plan)
- [9. hipDNN exposure plan](#9-hipdnn-exposure-plan)
  - [9.1 Op identifiers](#91-op-identifiers)
  - [9.2 Request/spec extensions](#92-requestspec-extensions)
  - [9.3 Capability gating and candidate registration](#93-capability-gating-and-candidate-registration)
  - [9.4 Open questions](#94-open-questions)
- [10. Test and bench plan](#10-test-and-bench-plan)
  - [10.1 Correctness reference](#101-correctness-reference)
  - [10.2 Benchmark shapes](#102-benchmark-shapes)
  - [10.3 Parity baselines](#103-parity-baselines)
- [11. Implementation scoping](#11-implementation-scoping)
  - [11.1 Known implementation traps](#111-known-implementation-traps)
- [12. Public DSA / sparse-attention implementations](#12-public-dsa--sparse-attention-implementations)
  - [12.1 DeepSeek-V3.2 (reference DSA)](#121-deepseek-v32-reference-dsa)
  - [12.2 GLM-5 / GLM-5.2 IndexShare](#122-glm-5--glm-52-indexshare)
  - [12.3 Other sparse-attention kernels](#123-other-sparse-attention-kernels)
- [13. References](#13-references)
- [14. Known limits and follow-ups](#14-known-limits-and-follow-ups)

---

## 0. Notation

- Index-query, index-key, index_head_dim (per-model, currently 128), n_index_heads (per-model variable, value from the §1 table, not fixed here), k = index_topk (2048).
- scores I(t,s), topk_idx / selected indices S_t, block_table / sub_block_table, IndexShare.
- Softmax scale: scale = 1/sqrt(d_nope + d_rope), per-model, host-supplied, not 1/sqrt(576). DeepSeek = 1/sqrt(192), GLM = 1/sqrt(256). Never derive scale from the 576 memory layout.

## 1. DSA geometry and model variants

### 1.1 Model table

The table below gives the indexer geometry and MLA geometry for each target model. Values are taken from the published Hugging Face config.json files (deepseek-ai/DeepSeek-V3.2-Exp, zai-org/GLM-5, zai-org/GLM-5.2). GLM-5.2 is confirmed and is identical in geometry to GLM-5.

| Field | DeepSeek-V3.2-Exp | GLM-5 | GLM-5.2 |
|---|---|---|---|
| n_index_heads | 64 | 32 | 32 |
| index_head_dim | 128 | 128 | 128 |
| k (index_topk) | 2048 | 2048 | 2048 |
| H_q (query heads) | 128 | 64 | 64 |
| d_nope (qk_nope_head_dim) | 128 | 192 | 192 |
| d_rope (qk_rope_head_dim) | 64 | 64 | 64 |
| d_V (v_head_dim) | 128 | 256 | 256 |
| r_KV (kv_lora_rank) | 512 | 512 | 512 |
| r_Q (q_lora_rank) | 1536 | 2048 | 2048 |
| scale | 1/sqrt(192) × YaRN mscale² | 1/sqrt(256) | 1/sqrt(256) |
| num_hidden_layers | 61 | 78 | 78 |
| model_type | deepseek_v32 | glm_moe_dsa | glm_moe_dsa |

Notes on the table:

- The indexer width (n_index_heads) is per-model: 64 for DeepSeek-V3.2, 32 for GLM-5. The value "64" referenced in the ticket is the DeepSeek figure. This field must be carried as a per-model parameter, not a constant.
- scale is per-model and host-supplied. GLM-5 uses default RoPE, so scale is the plain 1/sqrt(d_nope + d_rope) = 1/sqrt(256). DeepSeek-V3.2 uses YaRN RoPE scaling (type yarn, factor 40, mscale 1.0, mscale_all_dim 1.0), which folds an additional mscale² factor into the softmax scale, so DeepSeek's effective scale is 1/sqrt(192) multiplied by that YaRN correction, not the plain 1/sqrt(192). scale must never be derived from the 576-wide decode memory layout, and the exact DeepSeek mscale formula should be confirmed against the model source.
- The config field num_key_value_heads reads 64 (GLM-5) / 128 (DeepSeek-V3.2), but MLA shares a single compressed latent per token (effective KV head count = 1).
- GLM-5 sets indexer_rope_interleave = true, i.e. the lightning indexer is position-aware and applies RoPE to its index-query and index-key. Confirm whether DeepSeek-V3.2's indexer does the same, the indexer kernel specification (§3) must apply RoPE if so.

### 1.2 Relationship to MLA geometry

DSA is a selection stage wrapped around MLA: the lightning indexer and top-k choose a subset of KV positions, and the existing MLA attention math then runs over that subset. This section records which geometry carries over unchanged from MLA and which is new or model-dependent, because the difference determines how much of the MLA design can be reused and how much must be extended.

Carries over (identical across both DSA models, and shared with the DeepSeek row the MLA design was written against):

- d_rope = 64 and r_KV = 512 are fixed. The decoupled-RoPE key width and the compressed-latent width do not vary, so the decode-absorb working width r_KV + d_rope = 576 is the same for every model.
- k = 2048 and index_head_dim = 128 are stable, so the selection size and the per-head indexer width can be treated as constants for tiling purposes even though the head count is not.

New or model-dependent:

- GLM-5 differs from DeepSeek in four MLA-geometry values: d_nope 192 (vs 128), d_V 256 (vs 128), r_Q 2048 (vs 1536), and H_q 64 (vs 128). The MLA design doc wrote its kernel specifications and per-arch budgets for the DeepSeek row (128 / 128 / 128 / 1536, H_q 128) and explicitly placed GLM out of scope. Reusing the MLA kernels for GLM therefore requires the MLA specification to be extended to GLM's dimensions. This is mentioned again in §5.
- The indexer width n_index_heads is per-model (64 vs 32) and must be a parameter of the indexer kernel and its dispatch.
- scale is per-model and, for DeepSeek, non-trivial: GLM-5 = 1/sqrt(256); DeepSeek-V3.2 = 1/sqrt(192) with a YaRN mscale correction folded in. Both are host-supplied.
- num_hidden_layers differs (DeepSeek 61, GLM 78), which is relevant to IndexShare (how many sparse layers reuse a shared shortlist) and to any per-layer weight-working-set argument.

## 2. Math and data layout

### 2.1 The three-stage pipeline

DSA replaces one dense attention pass with three stages that run in order. First the lightning indexer scores every candidate key position against the current query. Second, top-k selection keeps the k highest-scoring positions and discards the rest. Third, MLA attention runs over only that selected subset instead of the full context. The output of the third stage is the attention result for the token, which flows to the next layer exactly as dense attention would.

Selection is per-query-token. Each query position runs its own top-k and receives its own set of up to k selected key positions. In decode there is one query, so one selected set. In prefill there are many queries processed together, and each one selects a different set. This per-query property is the main structural driver of the subset-attention kernel and is revisited in Sections 4.2, 5.2, and 7.3.

The pipeline reads from two separate caches. The MLA latent cache holds the compressed KV latent and the decoupled RoPE key per token, in bf16, and is the same cache dense MLA uses. The indexer keeps its own index-key cache, one small vector per token, in fp8. Both caches are written when a new token arrives. The fp8 index-key cache is what makes the indexer cheap.

There is a dense fallback. When the key length Sk is less than or equal to k, top-k selects everything and the sparse path has no benefit, so the request routes to plain dense MLA. Sparsity is therefore a long-context feature that activates only once the context exceeds k. This routing condition belongs in dispatch (Section 9.3) and means the benchmark shapes must use Sk well above k to exercise the sparse path at all (Section 10.2).

Two kinds of decision run through the sections that follow, and separating them up front helps. Some are requirements fixed by the algorithm and not open. DSA is three separate kernels because top-k is a barrier (Section 2.3), selection is per query token (this section), the softmax scale is per-model and host-supplied (Section 0), and the fp8 index-key cache must be arch-native (Section 8). Others are deferred knobs, sized or chosen at implementation and marked as such where they appear, among them the kernel tile sizes (Section 7), the top-k algorithm (Section 4.1), the fused-versus-standalone fallback (Section 4.1), and the scatter-versus-contiguous gather (Section 5.2). Where a knob is deferred, the section that raises it names the working default and what decides it.

### 2.2 Lightning indexer scoring

For a query at position t and a key at position s, the indexer computes a single relevance score by combining a small number of index heads:

```
I(t, s) = sum over heads h of w_h * ReLU(q_index_t_h . k_index_s_h)
```

Here q_index and k_index are the indexer's own query and key projections, which are separate from the MLA query and key. The per-head weight w_h is learned, and the ReLU clamps each head's contribution to be non-negative, so the score is a weighted sum of non-negative per-head matches rather than a plain dot product.

The indexer is cheap by construction. It uses no softmax, produces no value output, uses only a few heads (n_index_heads, which is 64 for DeepSeek-V3.2 and 32 for GLM-5), uses a narrow index_head_dim of 128, and runs in fp8. Scoring is causal, so a query only scores key positions at or before its own position. The indexer is position aware. GLM-5 sets indexer_rope_interleave to true, meaning RoPE is applied to the index-query and index-key before the dot product. The indexer kernel must therefore apply RoPE, and whether DeepSeek-V3.2 does the same is yet to be confirmed (Section 1.1).

### 2.3 Top-k selection

Top-k keeps the k highest-scoring key positions per query and outputs their indices. This is a global reduction over the full set of scores for each query, because the k-th largest value cannot be known until all scores exist. That property is the reason why DSA is three kernels rather than one fused operation. A standard flash-attention kernel fuses scoring, softmax, and the value accumulation into a single streaming loop because each key can be processed and then dropped. Top-k breaks that streaming structure, because the selection cannot start until the indexer has produced every score, and the subset attention cannot start until selection has produced the index list. The top-k step is a hard barrier between the indexer and the attention, so those two stages are separate kernels with the top-k reduction between them. Section 3.3 expands on this.

### 2.4 MLA over the subset

The third stage is ordinary MLA attention with the key length capped at k. It reuses the MLA math directly, so the score, softmax, and value steps are the MLA ones, and the only change is that the loop over key positions visits the selected set instead of the full context.

Each query row attends over its own selected set (Section 2.1). The score for a selected key combines a content term and a position term into a single number before one softmax:

```
score_s = scale * (q_nope . K_nope_s + q_rope . k_rope_s)
```

The content term q_nope . K_nope_s is the meaning match, and the position term q_rope . k_rope_s is the RoPE match. The two are summed, not run as two separate attentions. The RoPE keys are stored already rotated, so only the query side is rotated at the current position at runtime, and skipping that rotation silently breaks positions. The value scale is per-model and host-supplied as defined in Section 0 and Section 1.1, and it is 1/sqrt(d_nope + d_rope), never derived from the 576-wide decode memory layout.

### 2.5 IndexShare

IndexShare reuses the selected indices from a prior dense (full) layer in the following sparse layers, so those layers skip the indexer and top-k and run only the subset attention. It is valid when the reused shortlist is a good enough approximation of what the current layer would have selected, which the model is trained or configured to rely on, and it is not valid across a boundary where the selection is expected to change, such as the dense layers themselves.

Two facts must be read from each model's configuration rather than assumed: which layers are dense and which are sparse, and how far a shared shortlist propagates before it is recomputed. The layer counts differ between models (61 for DeepSeek-V3.2, 78 for GLM-5), so the sharing pattern is per model. Reused indices are not identical to freshly computed ones, so correctness for IndexShare is checked as reused versus fresh selection staying within tolerance rather than as an exact match (Section 10.1).

### 2.6 Online softmax over the selected set

The streaming softmax recurrence is the same one dense MLA uses. It runs over the selected set instead of the full context, and because selection only changes which keys are visited and not the arithmetic of the recurrence, the running maximum, running sum, and accumulator update are unchanged. For the decode absorb form the accumulation happens in latent space, as in MLA, and the value projection is applied once in the epilogue. The design should confirm that no reordering introduced by the gather changes the reduction in a way that affects the tolerance in Section 10.1.

### 2.7 Prefill vs decode forms

The pipeline takes two forms that match the two MLA forms. The prefill form processes many query tokens at once and may expand the compressed latent back to full keys and values, so its per-token work is heavier. The decode absorb form processes one query token per sequence and attends in the compressed latent space, so its gather reads only the small latent and is light.

Sparsity pays most where the full-context cost was largest, which is long-context prefill and long-context decode. The per-query ragged selection is heaviest in prefill, because every query row selects a different set and the usual reuse of a loaded key tile across query rows no longer holds. The decode form gathers over the compressed latent and touches little data per selected token. These two cost profiles feed the fused-versus-split and single-versus-two-kernel decisions in Sections 4 and 5.

## 3. Lightning indexer kernel specification

The lightning indexer is a score-only kernel. It computes the relevance score I(t, s) of Section 2.2 for every query position t against every allowed key position s, and produces the scores that feed top-k selection (Section 4). It does not compute softmax and does not touch any value tensor, so it is closer to the first half of a flash-attention kernel than to a full attention kernel.

### 3.1 Inputs and outputs

The indexer uses its own geometry (n_index_heads and index_head_dim from Section 1.1), which is distinct from the MLA attention geometry, and its own fp8 index-key cache, which is separate from the MLA latent cache (Section 2.1). The index-query is produced by a pre-step projection from the hidden state, in the same way the MLA query is produced by a pre-kernel, and the index-keys are written to the index-key cache at token-append time.

Inputs:

| Tensor | Shape | Layout | Notes |
|---|---|---|---|
| index_q | [total_q, H_I, D_I] prefill, [B, H_I, D_I] decode | row-major | projected index-query, where H_I is n_index_heads and D_I is index_head_dim (128). Produced by the pre-step, RoPE already applied (Section 3.2). Matmul input dtype is fp8, confirm whether it is rounded from a bf16 projection |
| index_k_cache | [num_blocks, block_size, D_I] per key, paged | paged | the indexer index-key cache, fp8. Confirm whether index-keys are per-head (a full [.., H_I, D_I] cache) or shared across heads in an MQA style, because that decides the cache footprint and the read pattern |
| w | [total_q, H_I] or [H_I] | row-major | the per-head indexer weights of Section 2.2. Confirm whether they are per-query (produced by the pre-step) or a per-head constant |
| block_table | [B, max_blocks] | int32 | paged pointers into the index-key cache |
| cu_seqlens_q, seqused_k | [B+1], [B] | int32 | query and key lengths for the causal bound |
| positions | [total_q] | int32 | query token positions for the RoPE rotation in the pre-step |

Outputs:

| Tensor | Shape | Notes |
|---|---|---|
| scores | [total_q, Sk] logical | the per-query score row I(t, .) over allowed keys. See the note below on materialization |

The scores output is logically a [total_q, Sk] matrix, which is O(total_q times Sk) in size and therefore large at long context in prefill. Writing it to HBM and reading it back into top-k is the standalone form. Consuming it directly in a fused top-k tail, so that the score row never leaves the chip, is the fused form. This is the primitive choice in Section 4.1, and the size of the score matrix is the main reason to prefer fusing the indexer with top-k rather than materializing scores.

### 3.2 Kernel structure

The kernel is organized like a query-key score loop. Each workgroup owns a block of query positions and walks the allowed key range in tiles, and for each key tile it computes the per-head products, applies the ReLU, weights and sums across heads, and emits the score for those keys.

Pre-step (a separate device kernel, once per indexer call, not inside the key loop):

1. Project the hidden state to index_q of shape [total_q, H_I, D_I], using the index-query projection weight.
2. Apply RoPE to index_q at the query positions. The indexer is position aware, GLM-5 sets indexer_rope_interleave to true, and whether DeepSeek-V3.2 does the same is a confirm item (Section 1.1). If the index-keys are stored already rotated, only the query side is rotated here, matching the MLA convention.

Score loop (per query block, over key tiles):

1. Load the index_q tile for the block into registers or LDS. Its size is H_I times D_I per query row, which is small (for example 64 times 128 for DeepSeek, 32 times 128 for GLM).
2. Load an index-key tile from the fp8 index-key cache through the block_table.
3. For each head, compute the dot product of index_q and the index-key over D_I, apply ReLU, and multiply by the head weight w.
4. Sum the weighted per-head results across the H_I heads to form the scalar score for each key in the tile.
5. Apply the causal bound against seqused_k so a query only scores keys at or before its position.
6. Emit the scores for the tile, either to the score buffer (standalone form) or to the fused top-k tail (fused form, Section 4.1).

The tiling is sized to the index geometry, not to MLA's geometry. The contraction width is index_head_dim (128), the head count is n_index_heads (64 or 32), and there is no value dimension and no softmax, so the register and LDS pressure is much lower than an MLA attention tile. The score matmul consumes fp8 inputs, so on gfx950 it can use a native fp8 matmul atom, and on gfx942 the fp8 format question of Section 8 applies. The per-arch tiling and budget are in Section 7.1.

Two structural details must be settled at implementation and are flagged here. Whether the index-keys are per-head or shared (Section 3.1) changes the key read pattern and the cache footprint. Whether the score row is materialized or fused into top-k (Section 4.1) changes whether the O(total_q times Sk) score matrix ever exists in memory.

### 3.3 Why it is a separate kernel from attention

The indexer is a separate kernel from the subset attention because of the top-k barrier of Section 2.3. Top-k is a global reduction that needs every score for a query before it can select, and the subset attention needs the selected index list before it can start, so the selection sits as a hard barrier between scoring and attention and cannot be fused into a single streaming pass the way dense flash attention fuses scoring, softmax, and value accumulation.

Three further differences reinforce the split even setting the barrier aside. The indexer uses a different geometry (n_index_heads and index_head_dim rather than H_q and d_nope). It uses a different dtype (fp8 rather than bf16). It reads a different cache (the fp8 index-key cache rather than the MLA latent cache). For all of these reasons the indexer is specified and tuned as its own kernel.

The one fusion that does remain open is between the indexer and top-k, not between the indexer and attention. Because the score matrix is large in prefill, fusing the top-k tail onto the indexer so that scores never reach HBM is attractive, and that choice is specified in Section 4.1.

## 4. Top-k selection specification

Top-k selection turns the indexer scores into the per-query list of selected key positions that the subset attention reads. It is the barrier stage of Section 2.3, a global reduction that must see every score for a query before it can choose. This section specifies the primitive that performs the selection and the layout of the index list it produces.

### 4.1 The primitive

There are two forms of the primitive, and the choice between them is the same choice raised by the indexer output in Section 3.1.

For a fused tail on the indexer, top-k is computed as the indexer produces scores, so the running set of best candidates is maintained on chip and the full score row is never written to memory. This avoids materializing the score matrix, which is O(total_q times Sk) and large at long context in prefill (Section 3.1). Its cost is that the kernel must hold a per-query candidate structure of size up to k in registers or LDS while it streams, and k is 2048, so that working set is not small.

For a standalone reduction, the indexer writes the score row to HBM, and a separate top-k kernel reads it back and selects. This is simpler and lets the selection use a plain global reduction, but it pays the write and read of the O(total_q times Sk) score matrix, which is the traffic the fused form exists to avoid.

The fused form is the specified form, because the score matrix it avoids is largest exactly where DSA is used, in long-context prefill. The standalone form is kept only as a fallback, taken if the on-chip candidate structure of up to k entries proves too costly to hold within the gfx942 LDS cap (Sections 7.2 and 7.4). If a crossover between the two is ever needed it is by token count, in the same way the moe fused mega fp8 kernel selects between its fused and split variants (Section 11.1). The decode case is lighter (one query per sequence, Section 2.7), so it uses the fused form as well, and the standalone path is a gfx942-budget contingency rather than a decode-versus-prefill choice.

This is a large-k selection over the full context, and that is what makes it new. k is 2048 and the candidate set Sk can be tens of thousands of positions. The existing top-k primitives in the tree are small-k and MoE-scale (for example the CK warp-level block_topk_stream_2d and the topk_softmax path select a handful of experts), so they do not carry over directly. A DSA top-k must select 2048 of many thousands per query, which is a different regime.

A full sort is unnecessary and wasteful, because only the top k positions are needed and their internal order does not affect attention correctness (softmax is order independent up to the reduction tolerance of Section 2.6). The primitive is therefore a partial selection. The working default is a streaming argmax-k that maintains a running top set, because it composes with the fused form above, consuming scores as the indexer produces them without a second pass over the score matrix. The alternatives, a threshold pass that finds a score cutoff admitting about k positions and then filters, and a partial or bitonic selection, are kept as fallbacks worth measuring where the streaming form's on-chip working set is too large. The decider is the per-arch candidate-set budget of Section 7.2, since the trade is between the on-chip working-set size the streaming form needs and the extra passes a threshold form needs, and it is settled at implementation against the gfx942 cap.

Selection is per query and causal. Each query row selects independently over only the key positions at or before its own position. When Sk is less than or equal to k there is nothing to select and the request takes the dense fallback of Section 2.1, so the selection primitive is only reached when Sk is greater than k.

### 4.2 Output contract

The output is the per-query list of selected key positions. Its layout is the interface between selection and the gather in Section 5.1, and it is also the tensor that IndexShare caches and reuses across layers in Section 6, so the contract is specified once here and referred to from both.

| Tensor | Shape | Layout | Notes |
|---|---|---|---|
| topk_idx | [total_q, k] prefill, [B, k] decode | int32 | selected key positions per query row |
| topk_count | [total_q] or [B] | int32 | number of valid entries per query row, for rows with fewer than k valid keys |

Attention does not require the selected positions to be in any particular order, but the gather benefits from ordered indices, because sorted-ascending positions give the paged read a more contiguous access pattern, so the contract specifies sorted-ascending selected positions when the sort is cheap enough to be worth it. This is a point to measure rather than assume, because the sort adds cost to the selection in order to save cost in the gather. When a query has fewer than k valid keys, for example early positions under the causal bound or a sequence whose length sits just above the dense-fallback threshold, the list is shorter than k, so the contract carries a per-query topk_count that tells the gather and the attention loop how many entries are real, and any unused slots are filled with a sentinel value that the gather skips, which keeps the tensor a fixed shape of [., k] while remaining correct for short rows.

The selected positions are token positions in the sequence, which the gather in Section 5.1 turns into paged block and offset references through the block_table, and whether the list stores raw token positions or pre-resolved block and offset pairs is a gather-side decision in Section 5.1 that does not change this contract. Each row of topk_idx is an independent list, which is the per-query ragged-gather property of Section 2.1 expressed as a data layout, so in prefill the rows differ from one another, which is what breaks key-tile reuse across query rows in Section 5.2, while in decode there is one row per sequence, shared by all of that token's attention heads. Because topk_idx is exactly the shared object IndexShare reuses, its lifetime can extend beyond a single layer, so when IndexShare is active a layer reads a topk_idx produced by a prior layer instead of running the indexer and this primitive at all, and the representation of that cross-layer tensor as a persistent input is a hipDNN integration question covered in Section 9.4.

## 5. Subset attention (MLA over subset) specification

The subset attention is ordinary MLA attention with the key loop restricted to the selected positions produced by top-k. It reuses the MLA math and the MLA kernels directly, and the only structural change is that the loop over key positions visits the selected set instead of the full context. This reuse carries a dependency risk that must be stated plainly. The MLA kernels are design-only today, with no implementation in the tree, and the MLA design specifies its kernels and budgets for the DeepSeek and Kimi geometry while explicitly excluding GLM. Reusing the MLA kernels for GLM therefore requires the MLA specification to be extended to GLM's wider content and value dimensions and its different head count, as set out in Section 1.2, so subset attention for GLM depends on MLA scope growing, not only on MLA landing. The softmax scale used here is the per-model host-supplied value of Section 0 and Section 1.1, and it is never derived from the 576-wide decode memory layout.

### 5.1 block_table gather

The selected positions in topk_idx are token positions in the sequence, and the paged KV cache stores tokens in fixed-size blocks, so the gather resolves each selected token position into a block and offset reference through the block_table before the attention kernel reads it. The natural output of this resolution is a per-query sub_block_table, a compacted list that names only the blocks and offsets a given query selected, which the paged attention loop then walks in place of the full block_table. Because the cache is paged in blocks of block_size while selection is token-granular, several selected tokens can fall inside the same block, and when the selected positions are sorted ascending as recommended in Section 4.2 those in-block neighbours resolve to one block read rather than several, which is the main reason ordered indices help the gather. Whether the resolution from token position to block and offset is done inside the top-k output or as a first step of the gather is an implementation choice that does not change the output contract of Section 4.2.

### 5.2 Scatter-read vs gather-to-contiguous scratch

There are two ways to feed the selected tokens to the attention loop, and the choice is the per-query ragged-gather cost question, which must be measured rather than assumed. In the scatter-read approach the attention kernel reads the selected tokens directly from their scattered locations in the paged cache during the flash loop, which avoids any copy but has an irregular access pattern and, in prefill, breaks the reuse of a loaded key tile across query rows because neighbouring query rows select different tokens. In the gather-to-contiguous approach a separate pass first copies the selected tokens into a contiguous scratch buffer and the attention loop then runs over that buffer as if it were dense, which restores a regular access pattern and lets the attention kernel look like standard dense attention over k tokens, at the cost of the copy and the scratch memory it needs. The balance between the two differs by form. In prefill each query row selects a different set, so a per-row contiguous scratch would be sized as total_q times k times the latent width, which is large, and scatter-read is the more natural choice even though it is the one that loses tile reuse. In decode there is one selected set per sequence, so a contiguous scratch is only B times k times the latent width, which is small, and the gather-to-contiguous approach is more attractive. The design measures both per form and per arch, with the budget in Section 7.3.

### 5.3 Prefill form

The prefill form reuses the MLA prefill kernel, the latent-expansion path that expands the compressed latent back to keys and values inside the flash loop, with the key loop driven by the per-query sub_block_table from the gather instead of the full sequence. This is the form where the per-query ragged selection is heaviest, for the reason given in Section 5.2, so the scatter-read against the paged cache is the expected path and the cost of its irregular access is the main thing to measure here. The causal bound and the packed varlen handling of the MLA prefill kernel carry over unchanged, and if chunked prefill is in scope the softmax log-sum-exp output is still required from the kernel. Because this reuses the MLA prefill kernel, it inherits the dependency and the GLM-geometry gap noted at the head of this section.

### 5.4 Decode-absorb form

The decode-absorb form reuses the MLA decode-absorb kernel, the weight-absorbed path that runs as MQA in the compressed latent space at the 576-wide layout, with the key loop restricted to the selected set. The gather here is over the compressed latent, so it reads only the small latent and the decoupled RoPE key per selected token rather than expanded keys and values, which makes it cheap. Two properties make this form favourable. The selected set is shared across all the query token's attention heads, because selection is per query token and not per head, so the head-batching that makes MLA decode efficient is preserved and only the key list changes. The selected set is also bounded at k, so the kernel reads a fixed amount of KV regardless of how long the context is, which is the source of the long-context saving that decode DSA exists to capture. When the decode path uses the MLA 3D split-KV structure, the segments partition the selected set rather than the full context, and the segment reduce is otherwise unchanged. As with prefill, this reuses the MLA decode-absorb kernel and inherits the dependency and the GLM-geometry gap noted at the head of this section.

## 6. IndexShare handling

IndexShare is the GLM variant in which a layer reuses the selected indices produced by a prior layer instead of running the lightning indexer and top-k itself. The object it reuses is exactly the topk_idx list defined in Section 4.2, so an IndexShare layer skips stages one and two of the pipeline entirely and runs only the subset attention of Section 5 against the inherited list. The saving is the whole cost of the indexer scoring, its fp8 index-key cache reads, and the top-k selection for every layer that reuses rather than recomputes, which is why IndexShare is worth handling despite the extra plumbing it needs.

The behavior uses a layer index gate that decides, per layer, whether that layer computes a fresh selection or reuses an earlier one. Some layers produce a selection, and the following layers inherit it, and both which layers produce and how far a produced shortlist propagates before it is recomputed are properties of the model rather than choices this design makes. These must be read from each model's configuration and not assumed, and they can differ between models, since the layer counts differ (61 for DeepSeek-V3.2, 78 for GLM-5 and GLM-5.2) and the sharing pattern is expressed per model. The design specifies the gate as a per layer input that selects the compute fresh or reuse path, and it leaves the concrete pattern to the confirmed model configuration. For GLM-5.2 that configuration is explicit: the indexer_types array marks each layer full or shared, the first index_skip_topk_offset layers (3) are full, and thereafter a full layer occurs every index_topk_freq layers (4) with the intervening layers shared, so a shared list propagates at most index_topk_freq minus one layers before a full layer recomputes it, and index_share_for_mtp_iteration being true means the multi-token-prediction iteration also reuses the index.

Reuse has a correctness caveat, because the indices a layer inherits are not the indices it would have computed for itself, so an IndexShare layer attends to a slightly different set than a fresh selection would choose. This is expected behavior that the model is trained or configured to rely on, so the correctness gate for IndexShare is not an exact index match but a check that the end-to-end output under reused indices stays within the tolerance of Section 10.1 against a reference that reuses indices the same way, and separately that the propagation depth and the choice of which layers produce match the model configuration. A reference that recomputed fresh indices at every layer would not be a valid oracle for an IndexShare layer, since it would measure the approximation itself rather than the kernel.

Because the shared topk_idx must survive from the layer that produces it to the layers that consume it, its lifetime extends beyond a single layer and beyond a single kernel launch, which makes its representation a hipDNN integration question rather than a purely internal one. Whether it is carried as a persistent graph input, an opaque handle, or another mechanism is left to the coordination in Section 9.4. Finally, IndexShare is an optimization layered on top of a working sparse pipeline, so it can be scoped as a later phase than the base indexer, top-k, and subset attention, and the implementation scoping in Section 11 records whether it is in the initial target or deferred.

## 7. Per-arch tiling and budget (gfx942 / gfx950)

This section estimates the on-chip footprint and the memory traffic of the three DSA stages, per architecture. Every figure is an analytical estimate derived from the geometry of Section 1, not a measurement, and it is meant to size the design and expose the binding limits rather than to predict performance. The two architectures differ in the two ways that matter here. gfx942 has 64 KB of LDS per CU, the narrow 16x16x16 bf16 MFMA atom, no transposed LDS read, and it reads fp8 in the fnuz interpretation. gfx950 has 160 KB of LDS per CU, wider MFMA atoms, a transposed LDS read, and it reads fp8 in the OCP interpretation. The per-model geometry is carried as variables, so where a number depends on the model it is given for both DeepSeek-V3.2 and GLM-5.

### 7.1 Indexer tiling and LDS/register budget

The indexer is a score-only kernel, so its on-chip footprint is small compared with attention. Per query it needs the index-query resident, which is n_index_heads times index_head_dim values, so 64 times 128 for DeepSeek and 32 times 128 for GLM, a few kilobytes at most once cast to fp8. It streams the index-keys in tiles from the fp8 index-key cache, a tile of Bk keys costing Bk times index_head_dim fp8 bytes, and it holds a small running structure for the scores that feed top-k. There is no value tensor and no softmax state, so unlike attention the indexer is not LDS-bound, and it fits comfortably inside either architecture's LDS with room to spare. The tiling knobs are the query-block size and the key-tile size Bk, and both can be larger than in attention because the per-key work is a single narrow dot product plus a ReLU.

The load-bearing cost of the indexer is not its LDS, it is the index-key cache traffic. The indexer reads the whole index-key cache once per call, which is the compulsory O(Sk) read of the stage. The per-token footprint of that cache depends on an open question from Section 3.1, whether the index-keys are stored per head or shared across heads. If per head, the footprint is n_index_heads times index_head_dim fp8 bytes per token, which is 64 times 128 equals 8 KiB per token on DeepSeek and 32 times 128 equals 4 KiB per token on GLM, so a 32768-token context is on the order of 256 MiB and 128 MiB respectively. If shared, it is index_head_dim fp8 bytes per token, 128 bytes, roughly 64 times smaller. This difference is large enough that the storage form must be confirmed before the indexer budget is trusted, and it is the reason fp8 was chosen for this cache in the first place, since it halves that traffic against bf16. The index-query is negligible by comparison because it has no Sk axis.

The on-chip footprint and the index-key cache footprint are summarized below (analytical estimates, fp8, per Section 1.1 geometry):

| Item | Size | DeepSeek (H_I=64) | GLM (H_I=32) |
|---|---|---|---|
| index_q tile (on-chip) | Bq × H_I × D_I fp8 | small (a few KB per query block) | small |
| index_k tile (on-chip, streamed) | Bk × D_I fp8 | Bk × 128 B | Bk × 128 B |
| index-key cache, per token, per-head form | H_I × D_I fp8 | 8 KiB | 4 KiB |
| index-key cache, per token, shared form | D_I fp8 | 128 B | 128 B |
| index-key cache traffic per call | above × Sk (compulsory O(Sk) read) | e.g. ~256 MiB at Sk=32768 (per-head) | e.g. ~128 MiB at Sk=32768 (per-head) |

The on-chip rows fit either architecture's LDS easily; the cache-traffic row is the binding cost of the stage, and the per-head versus shared choice (Section 3.1) swings it by roughly 64x.

### 7.2 Top-k reduction budget

The top-k budget is dominated by the candidate structure, and the two forms of Section 4.1 price it differently. In the fused form the kernel maintains a running best-set of up to k entries per query while the indexer streams scores, so it holds on the order of k entries on-chip, each entry being a score plus an index. At k equal to 2048, and taking an entry as a small number of bytes, this is on the order of tens of kilobytes per query, which is the cost that competes for LDS and registers against the indexer tile that produced it. This is the figure to watch, because it is per query and it does not shrink with context, so it sets a floor under the fused kernel's occupancy. In the standalone form the on-chip candidate structure disappears, and the cost moves to HBM instead, where the indexer writes the O(total_q times Sk) score matrix and the top-k kernel reads it back. That matrix is small in decode, where total_q is the batch, and large in prefill, where total_q is the packed query count, which is why the fused form is preferred for prefill and either form is viable for decode.

The selection does not need a full sort, so the reduction budget is that of a partial selection rather than a sort of Sk element. The exact algorithm is deferred to implementation, but the budget consequence is that the working set is bounded by k, not by Sk, in the fused form, and by the score-matrix traffic, not by k, in the standalone form.

The two forms are summarized below (analytical estimates, k = 2048):

| Form | On-chip working set | HBM traffic | Best for |
|---|---|---|---|
| Fused (tail on the indexer) | ~k entries per query (score + index), on the order of tens of KB, does not shrink with context | none, scores never leave the chip | prefill, where the score matrix is largest |
| Standalone (separate kernel) | none | O(total_q × Sk) score matrix, written by the indexer and read back | decode, where total_q is the batch and the matrix is small |

The fused row's per-query working set is the item that competes with the indexer tile of Section 7.1 for LDS and registers, and it is the figure to check against the gfx942 cap in Section 7.4.

### 7.3 Gather cost and KV-traffic budget for subset attention

Once selection is done, the subset attention reads only the selected tokens, so its KV traffic is bounded by k rather than by Sk. In decode this is the clean case. There is one selected set per sequence, shared across all of the token's heads, and the read is over the compressed latent, so the traffic is k times the per-token latent width, which is 2048 times 576 times 2 bytes, on the order of 2.25 MiB per sequence per layer, and it is fixed regardless of how long the context is. That fixed bound is the source of the decode win, and it is why decode traffic does not grow with context the way dense attention does. The gather itself in decode is cheap, a copy of k latent tokens into a contiguous scratch is only k times the latent width per sequence, small enough to be worthwhile, which is why decode favours the contiguous-scratch path of Section 5.2.

Prefill is where the gather must be priced carefully, because selection is per query and the sets differ across query rows. The traffic floor is the same k-bounded read per query, but the structure of that read is ragged. In dense prefill a loaded KV tile is reused across neighbouring query rows, so the effective traffic is far below query-count times tile-size. In sparse prefill each query row selects a different set, so that reuse is lost, and in the worst case the traffic approaches total_q times k times the per-token width with no sharing between rows. The gather is therefore not free overhead on top of a cheaper attention, it is the dominant cost of prefill subset attention, and the scatter-read against the paged cache is the expected path because a per-row contiguous scratch would be total_q times k times the latent width, which is too large to materialize. The block-table resolution adds k index lookups per query to build the per-query sub-block-table, and sorting the selected indices reduces the read count by collapsing selected tokens that fall in the same paged block into a single block read. How much of the ragged read the cache absorbs is the main unknown, and it is the first thing to measure for prefill.

### 7.4 gfx942

On gfx942 the 64 KB LDS is the tight resource, and the subset attention inherits the MLA gfx942 tiling and its budgets directly, since the attention itself is unchanged, so the decode and prefill LDS peaks are the MLA ones plus the small sub-block-table the gather produces. The indexer fits easily within 64 KB because its tiles are small, and the top-k candidate structure of Section 7.2 is the item that competes for the remaining LDS, so on gfx942 the fused indexer-plus-top-k budget must be checked against the 64 KB cap with the candidate set included, not the indexer tile alone. The architecture has no transposed LDS read, so any place the subset attention needs the latent on two contraction axes inherits the same padding/copy choice the MLA design records for gfx942, and the copy option costs additional LDS that must be counted against the cap.

The fp8 index-key cache is the open risk on gfx942. gfx942 reads fp8 in the fnuz interpretation, so an index-key cache written in the OCP interpretation would be mis-decoded, which is the same format hazard the fp8 decode work hit. Two consequences follow. Either the indexer keys must be supplied in the fnuz format on gfx942, matching the arch, or the fp8 indexer is restricted to gfx950 and gfx942 runs the indexer in bf16 at twice the index-key cache traffic of Section 7.1. Which of these is chosen is a scoping decision that the dtype plan in Section 8 must settle, and until it is settled the gfx942 indexer budget should be quoted for the bf16 fallback as well as the fp8 path.

### 7.5 gfx950

On gfx950 the 160 KB LDS removes most of the pressure that binds gfx942. The larger cap lets the fused indexer-plus-top-k kernel hold the k-bounded candidate structure of Section 7.2 alongside a larger key tile without the cap becoming the limit, so the binding resource on gfx950 is more likely to be registers or the candidate structure than LDS. The wider MFMA atoms and the transposed LDS read benefit the subset attention exactly as they benefit MLA on gfx950, so the attention side again inherits the MLA gfx950 tiling and budgets, plus the sub-block-table.

gfx950 reads fp8 in the OCP interpretation, which is the format the indexer's fp8 path is naturally specified against, so gfx950 is the architecture where the fp8 indexer and the fp8 index-key cache of Section 7.1 are expected to run without the format hazard of Section 7.4. This is also why fp8 is a gfx950-first phase in the dtype plan, and why the fp8 index-key traffic estimates in Section 7.1 are the ones that apply on gfx950 while gfx942 may pay the bf16 figure instead.

## 8. Dtype plan

DSA is developed in two dtype phases, and the split is not uniform across the three stages, because the indexer and the subset attention have different precision needs and different fp8 readiness. The subset attention is MLA, so it follows the MLA dtype plan, bf16 first on both architectures and fp8 as a later gfx950 phase. The indexer is where DSA departs, because it is specified in fp8 from the start, and that is the stage where the architecture split must be handled explicitly.

Phase one is bf16 and covers both architectures. In this phase the subset attention runs the MLA bf16 kernels over the selected subset, the gather and the top-k are dtype-neutral, and the indexer runs in bf16 rather than fp8. Running the indexer in bf16 is not the intended production form, but it is the correct first target, because it lets the whole pipeline be brought up and made correct on both gfx942 and gfx950 without entangling the bring-up with the fp8 format hazard below. The cost of the bf16 indexer is traffic, since its index-key cache is twice the size of the fp8 one from Section 7.1, but the arithmetic and the selection are identical, so the bf16 indexer is a faithful functional stand-in for the fp8 one and a valid correctness baseline.

Phase two adds fp8, and it is a gfx950-first phase for two independent reasons. The first is the MLA fp8 plan, which already restricts fp8 to gfx950 and later, so the fp8 subset attention inherits that restriction with no new decision to make. The second is specific to the indexer, and it is the format hazard named below.

The fp8 format hazard is the load-bearing constraint of this section, and it must be named plainly. gfx942 is CDNA3 and gfx950 is CDNA4, and the two decode an fp8 e4m3 byte under different conventions. gfx942 reads the fnuz interpretation and gfx950 reads the OCP interpretation, so the same byte pattern is a different number on the two parts. A cache of index-keys written in one convention and read on the other is silently mis-decoded, which produces wrong scores rather than an error, and wrong scores select the wrong tokens. This is the same landmine the fp8 decode work encountered, and the resolution there is the precedent here, arch-native format with a validity gate rather than in-kernel conversion. The consequence for DSA is that the fp8 indexer is naturally specified against the OCP interpretation that gfx950 reads, so the fp8 indexer runs cleanly on gfx950, and gfx942 has two options and no free third one. Either gfx942 is fed index-keys in the fnuz format that matches the part, which makes the fp8 indexer gfx942-capable but requires the caller to supply arch-native fp8, or the fp8 indexer is excluded from gfx942 and gfx942 stays on the bf16 indexer of phase one, matching MLA's exclusion of fp8 from gfx942. The design does not have to choose in the abstract, but it does have to record that the gfx942 fp8 indexer is contingent on the arch-native-format contract, and that the safe default is the bf16 indexer on gfx942 with the fp8 indexer on gfx950.

The indexer's precision needs are what make the fp8 choice safe where it is used, and they are worth stating because they are gentler than attention's. The indexer produces a relevance score whose only downstream use is a ranking, the top-k selection, so what must be preserved is the order of the scores, not their exact values. A small absolute error in an fp8 score is harmless as long as it does not reorder the boundary between the 2048th and 2049th positions, and near that boundary the scores are by definition close, so occasional reordering there changes which of two near-equally-relevant tokens is attended and has negligible effect on the output. This is the same reason the indexer can use a ReLU instead of a softmax and can skip the value path, and it is why fp8 is an appropriate dtype for scoring even though it would be too coarse for the attention accumulation. The keys and the query are fp8, while the per-head weights and the small reductions can stay in higher precision at negligible cost, so the fp8 is confined to the operand where its traffic saving is largest and its accuracy cost is smallest.

The subset attention keeps bf16 for its accumulation in both phases, and the fp8 phase applies only to the KV-side storage on the MLA path exactly as the MLA fp8 plan specifies, so DSA introduces no new fp8 accumulation and no new fp8 hazard beyond the indexer's index-key cache. The net dtype picture is therefore that the selection front-end is fp8 where the architecture allows it and bf16 where it does not, and the attention behind it is bf16 first and fp8 later on gfx950, with the two tracks meeting only at the selected-index list, which is integer and dtype-neutral.

## 9. hipDNN exposure plan

This section is written against two moving surfaces, and both caveats apply throughout. The rocKE selection surface is Python under library/dispatch/attention/, and the fields and registration below are proposed against that path. The hipDNN op-descriptor schema itself lives in the external hipDNN SDK, not in this tree, so the descriptor field names and the eventual packaging are proposals to be confirmed with the hipDNN team, not decisions this document can fix on its own. DSA also has a wider surface than MLA, because it is three stages rather than two, and the central open question of this section is how many of those stages the graph sees as distinct ops.

### 9.1 Op identifiers

DSA is naturally described as three operations, a lightning_indexer_fwd that produces relevance scores, a topk_select that turns scores into a selected-index list, and the subset attention itself. Whether these are exposed to the graph as three ops with intermediate tensors, or fused behind a single dsa_attention wrapper op that runs the indexer, the selection, and the attention internally, is the hipDNN team's call and is not settled here. The two framings have different consequences. Three separate ops make the intermediate tensors, the score matrix and the selected-index list, visible to the graph, which is what IndexShare needs in order to reuse a prior layer's index list across layers, since the shared object has to be a graph tensor to be shared. A single wrapper op hides the intermediates and presents DSA as one attention variant, which is simpler for the framework but leaves no natural place to express the cross-layer index reuse. Because IndexShare pulls toward exposing the selected-index list as a first-class tensor while simplicity pulls toward a single op, the op-count decision cannot be made without the IndexShare requirement in hand, and it is raised again in Section 9.4. The subset attention op, however it is packaged, reuses the MLA op surface, since the attention is MLA, so the only genuinely new op strings DSA introduces are for the indexer and the selection.

Following the MLA precedent, a new op string must not be admitted by widening the shared request-error gate, because that would drop the op guard from every existing candidate at once. The DSA ops should get their own request-errors function that accepts only the DSA op strings, leaving the shared gate pinned to the standard attention op, so the DSA and non-DSA candidate sets stay mutually exclusive by construction.

### 9.2 Request/spec extensions

The normalized request needs additive fields to carry the DSA geometry, and as with the MLA additions, zero-defaults keep every existing caller on the standard path. The new fields are the indexer geometry, n_index_heads and index_head_dim, the selection size k, an IndexShare flag with the identity of the shared-index source when it is set, and the selected-indices tensor itself, which is an input when a layer reuses a prior layer's selection and an output when a layer produces one. The indexer's own projection weights and its fp8 index-key cache are further inputs, distinct from the MLA latent cache, and they are described in Section 3.1.

These fields are inert as selection keys until they are added in two places, the request's dims accessor and the registry's dim vocabulary, because the declarative capability prefilter reads dimension values from the request and rejects any capability that constrains a dimension outside the vocabulary. A shape range over k or over n_index_heads that is added to a capability without also being added to the vocabulary and the dims accessor will either fail at import or reject every request, so the field additions and the vocabulary additions are a single coordinated change, not two independent ones.

### 9.3 Capability gating and candidate registration

DSA candidates should be registered in their own registry with their own family and dim vocabulary, not in the unified attention registry, for the same reason MLA takes its own registry. The candidate registry rejects a candidate whose family differs from the registry's, so a DSA candidate cannot be added to the attention_unified registry, and it should not be, because a DSA geometry must not become selectable for a standard attention request as a side effect. The declarative capability has no op field, so the op never participates in the prefilter, and both directions of the DSA versus non-DSA exclusion are carried by the predicates, a DSA request rejected by every standard candidate because the standard request-errors rejects the DSA op, and a standard request rejected by every DSA candidate because the DSA predicate rejects the standard op. That is why Section 9.1 keeps the shared request-errors pinned and gives DSA its own.

The dense-fallback condition of Section 2.1 lives in this layer as a selection condition. When the key length Sk is less than or equal to k, there is nothing to select, and the request should route to the dense MLA path rather than to a DSA candidate, because the sparse pipeline has no benefit and the top-k is a no-op. This is a routing predicate on the DSA candidates, Sk greater than k as a necessary condition for a DSA candidate to admit, and it mirrors the way the existing dispatch special-cases short sequences. It also means the DSA candidates and the MLA candidates must coexist so that a request can fall through from the former to the latter as Sk crosses k, which is a further reason the subset attention reuses the MLA op rather than reimplementing it.

### 9.4 Open questions

Several integration questions cannot be closed inside this document and must be resolved with the hipDNN team. The first is the op-count decision of Section 9.1, whether DSA is one wrapper op or three ops with visible intermediates, and it should be answered together with the IndexShare requirement rather than on simplicity grounds alone. The second is the representation of the IndexShare shared-index tensor, which outlives a single layer and a single kernel launch, so it is not an ordinary per-op intermediate. It must be expressed as something the graph can carry across layers, a persistent graph input, an opaque handle, or another mechanism the graph adapter provides, and which of these is available is a hipDNN question of the same class the MLA design raises for its absorbed weights. The third is the exposure of the model-load-time constant weights, the indexer's own projection weights, which are known only after the model loads and not at kernel compilation time, so like the MLA absorbed weights they must be represented as persistent weight-type graph inputs rather than as compilation constants, and the mechanism for that must be confirmed. The fourth is the packaging and AOT question the MLA design also carries, what replaces the deleted prebuilt-instance catalog path for shipping DSA instances and on what timeline, since DSA adds new instances for the indexer and the selection on top of the MLA instances. These are the coordination items that make the hipDNN integration a joint deliverable with that team, which is why the definition of done requires it to be developed with them rather than specified here.

## 10. Test and bench plan

DSA is tested stage by stage and then end to end, and the guiding principle is that the correctness oracle must make the same approximation the kernel makes. DSA is a deliberate approximation of dense attention, so a reference that recomputes dense attention would measure the approximation, not the kernel. The oracle is therefore a sparse reference that scores, selects, and attends the same way the kernels do, and dense attention appears only as an upper-bound baseline in Section 10.3, never as the pass or fail gate.

### 10.1 Correctness reference

The reference is built in three pieces that match the three stages, and the two selection-side pieces are tested in isolation before the end-to-end gate, because a discrete selection error is easy to hide inside a tolerance band once it reaches attention. The indexer reference computes the scores of Section 2.2 directly, the per-head ReLU of the index-query and index-key dot products, weighted and summed, with RoPE applied, and the kernel's scores are checked against it. Because the scores feed only a ranking, the meaningful check on the indexer is not the exact score value but whether the induced order is preserved well enough for selection, so the indexer test is read together with the top-k test rather than in isolation.

The top-k reference is a plain reference selection over the scores, and it is tested by exact index-set match. This is a discrete result, integer indices, so the gate is exact rather than a tolerance, with one carve-out. When two scores tie at the k-th boundary, the kernel and the reference may break the tie differently, and that is not an error, so the gate is an exact set match except at exact ties, or the test uses scores constructed to have no ties at the boundary. Testing top-k in isolation this way is what keeps a selection bug from being absorbed into the attention tolerance later.

The end-to-end reference runs the whole pipeline, score then select then attend over the selected subset, and compares the output against the kernel within the bf16 tolerance band, the same max-abs gate the MLA and unified attention gates use. The reference selects with the same top-k the kernel uses, so the comparison isolates the attention arithmetic and the gather from the selection, which was already gated exactly in the previous step. For IndexShare the reference must reuse indices the same way the kernel does, producing at one layer and reusing at the following layers per the model configuration, because a reference that recomputed a fresh selection at every layer would measure the reuse approximation rather than the kernel. The reference lives under library/builders/ next to the MLA reference, and the tests live under library/tests/, following the layout the MLA design records.

### 10.2 Benchmark shapes

The benchmark shapes must make sparsity the point, which is long context, because DSA does nothing below the selection size. With k fixed at 2048, any shape whose key length Sk is at or below 2048 takes the dense fallback of Section 2.1 and measures dense MLA, not DSA, so a short-Sk shape is a test of the fallback routing and not of the sparse path. The benchmark set therefore sweeps Sk well above 2048, into the tens of thousands, where the selected 2048 is a small fraction of the context and the sparse win is real, and it holds k at 2048 as the models specify. The sweep covers both the decode form, one query per sequence with a batch axis, and the prefill form, many query tokens at once, because the two forms have different cost structures, decode bounded by the fixed k-token read and prefill dominated by the ragged gather of Section 7.3. The shapes are carried per model, since the geometry differs, and a short-Sk point is included deliberately as the dense-fallback case so that the routing boundary at Sk equal to k is exercised, not to measure sparse performance there. As with the MLA shapes, the numbers these shapes produce go to the protected performance record, not into this repository.

### 10.3 Parity baselines

Three baselines frame the results. The primary correctness parity is against a DeepSeek-V3.2 and GLM-5 reference implementation of the DSA pipeline, which is the specification the kernels must match and the source of the sparse oracle in Section 10.1. Dense MLA is included as an upper-bound baseline, not as a correctness oracle, because it answers a different question, how much the sparse path saves against reading the full context, and because at short context where the fallback engages the two must coincide exactly. Any existing ROCm, AITER, or Composable Kernel sparse-attention or DSA kernel is included as an external comparison where one exists, in the same way the MLA design uses AITER and TileLang as external baselines, and the survey in Section 12 is where those are identified. The three baselines answer three separate questions, the reference answers is it correct, dense MLA answers how much did sparsity save, and the external kernel answers how does it compare to what already exists, and they should not be conflated.

## 11. Implementation scoping

This section enumerates the DSA deliverables, maps them to architectures and dtypes, and makes the ticket-split recommendation the epic asks for. It also states the sequencing dependency plainly, because it governs when this work can start. This is a design spike, so nothing below is committed, and every kernel row carries the dual-engine definition of done that applies to a new instance, a Python builder and its C++ engine mirror in the same change, registration in the DSA registry, a parity emit case, and the byte-identity gate green at both LLVM flavors.

#### Deliverable table

| # | Deliverable | Arch | Dtype | Notes |
|---|---|---|---|---|
| 1 | Lightning indexer | gfx942, gfx950 | bf16 | new kernel, own fp8 index-key cache in phase two |
| 2 | Lightning indexer | gfx950 | fp8 | phase two, OCP-native, gfx942 contingent (Section 8) |
| 3 | Top-k selection (large-k, k=2048) | gfx942, gfx950 | dtype-neutral | new primitive, fused tail on the indexer is the target |
| 4 | Gather + subset attention, prefill | gfx942, gfx950 | bf16 | reuses MLA prefill, scatter-read path, ragged gather |
| 5 | Gather + subset attention, decode | gfx942, gfx950 | bf16 | reuses MLA decode-absorb, contiguous-scratch path |
| 6 | Gather + subset attention, prefill and decode | gfx950 | fp8 | phase two, inherits MLA fp8 plan |
| 7 | IndexShare | arch-neutral | dtype-neutral | cross-layer index reuse, later phase |
| 8 | Dispatch wrapper + dense fallback | arch-neutral | — | DSA registry, op strings, Sk greater than k routing |

Deliverables 4, 5, and 6 are the subset attention, and they are lighter than the indexer and top-k rows because the attention itself is MLA, so what is new in those rows is the gather that feeds it and the wiring of the selected subset into the key loop, not the attention arithmetic. Deliverables 1, 2, and 3 are the genuinely new compute. Deliverable 7, IndexShare, is an optimization on top of a working sparse pipeline and can follow the base rows. Deliverable 8 is the dispatch and capability work of Section 9 plus the dense-fallback routing.

#### Ticket split recommendation

The recommendation is to split the lightning indexer and the top-k, deliverables 1, 2, and 3, into their own implementation ticket, separate from the subset-attention compute, and to expand the corresponding stories accordingly. The case for the split is that the indexer and top-k are a different kind of work from the subset attention along every axis that matters for scheduling. They use a different geometry, the indexer geometry rather than the MLA geometry. They use a different dtype, fp8 with the format hazard of Section 8, where the subset attention is bf16 first. They have different architecture support, since the fp8 indexer is gfx950-first while the subset attention follows the MLA arch plan. And the large-k top-k is a new primitive with no in-tree predecessor, where the subset attention reuses MLA. Bundling the indexer and top-k with the subset attention would couple a new fp8 scoring kernel and a new large-k selection kernel to the MLA-reuse work and slow both, whereas splitting them lets the selection front-end and the attention back-end proceed in parallel and be verified independently, the front-end by the exact top-k gate of Section 10.1 and the back-end by the MLA parity harness. The split also matches the natural fusion boundary, since the indexer and top-k fuse together and the subset attention is a separate kernel behind the barrier.

#### Sequencing

The sequencing dependency is asymmetric and must be stated. This design depends only on the MLA design, which exists, so the DSA design work can proceed now. The DSA build, however, depends on the MLA build, because deliverables 4, 5, and 6 reuse the MLA prefill and decode-absorb kernels, and those kernels are design-only today with no implementation and no schedule. Two consequences follow. The subset-attention deliverables cannot complete before the MLA kernels they reuse exist, so they are gated on MLA implementation. The indexer and top-k deliverables, 1 through 3, are not gated on MLA in the same way, since they are new kernels that do not reuse MLA, so the recommended ticket split also front-loads the work that can start earliest. And because MLA excludes GLM geometry, serving GLM through DSA additionally depends on MLA being extended to GLM, which is a dependency on MLA scope beyond MLA implementation, and it should be tracked as such rather than assumed.

### 11.1 Known implementation traps

Several traps are known before implementation starts, and they are recorded here so they are designed around rather than discovered. The per-query ragged gather is the central one, because selection is per query and the sets differ across query rows in prefill, so the usual reuse of a loaded KV tile across rows is lost and the scatter-read against the paged cache is the expected path. This is the dominant cost of prefill subset attention, not overhead on top of it, and it is the first thing to measure. The per-model scale is a correctness trap, because the softmax scale is per model and host-supplied, 1 over the square root of the content plus position width, which is 192 for DeepSeek and 256 for GLM, and it must never be derived from the 576-wide decode layout, and DeepSeek folds a further factor in under its rope scaling, so no geometry-derived default is correct. The dense-fallback routing is a trap because a shape with Sk at or below k must route to dense MLA and not to a DSA candidate, and a benchmark or a test that forgets this will measure or verify the wrong path, so the Sk greater than k condition must be present in both the dispatch and the shape selection.

A further pair of traps comes from the fused-versus-split structure and has a concrete in-tree precedent. The top-k and, on the attention side, the split-KV reduce, are places where a fully fused form and a split form trade off by size, and the repository already has a kernel that manages exactly this, the moe fused mega fp8 kernel, which selects between a fully fused variant and a split-fused variant by token count at measured crossovers. Two lessons transfer. The crossover thresholds and the tuned configuration must be declared once beside the kernel and imported by both the dispatcher and the benchmark, so they cannot drift apart. And a crossover measured on one geometry must not be applied to another, which for DSA matters directly because the DeepSeek and GLM geometries differ, so any fused-versus-split threshold must be cohort-scoped to the geometry it was measured on, and out-of-cohort shapes must fall through to an untuned generic path rather than be served a crossover measured at a geometry they do not share. Treating that moe kernel as prior art for both the crossover-declaration discipline and the cohort-scoping discipline avoids reinventing them under DSA.

## 12. Public DSA / sparse-attention implementations

As with the MLA survey, the review was done early and informs the design in Sections 2 through 5 rather than dictating it, and nothing here is a rocKE implementation decision. Two of the three subsections are model references, DeepSeek-V3.2 and GLM, which are the specifications the kernels must match, and the third is a survey of existing sparse-attention and selection primitives that are candidate building blocks or baselines. Where a claim rests on a source I have not confirmed against the code, it is marked to verify, and the model geometry values are the ones in Section 1.1.

### 12.1 DeepSeek-V3.2 (reference DSA)

DeepSeek-V3.2 is the canonical DSA reference and the origin of the design. It introduces DeepSeek Sparse Attention on top of MLA, adding the lightning indexer and the top-k selection in front of the existing MLA attention, with the indexer geometry and the selection size k that Section 1.1 records, 64 indexer heads, index_head_dim 128, and k equal to 2048. It is the specification the rocKE kernels must match and the source of the sparse correctness oracle in Section 10.1, so any disagreement between this document and the DeepSeek reference is resolved in favour of the reference. Two properties of the reference matter for our work and should be confirmed against its source rather than assumed. The first is whether the reference indexer applies RoPE to its index-query and index-key, which the GLM configuration implies and which the indexer kernel in Section 3 depends on. The second is whether the reference stores its index-keys per head or shared across heads, which drives the index-key cache traffic estimate in Section 7.1 and is the largest single unknown in that budget. The reference is CUDA-oriented and targets a different memory system, so it is an architectural and correctness reference, not a performance target for gfx942 or gfx950, and whether DeepSeek has released standalone kernels for the indexer and the selection that we can read, as opposed to a framework integration, is a to-verify item that the implementation should check before reimplementing from the paper.

### 12.2 GLM-5 / GLM-5.2 IndexShare

GLM-5 is a DSA model in its own right, and it is the source of the IndexShare requirement. Its published configuration is model_type glm_moe_dsa with the GlmMoeDsaForCausalLM architecture, and it carries index_n_heads 32, index_head_dim 128, and index_topk 2048, so its indexer is narrower than DeepSeek's, which is the concrete evidence that the indexer width is a per-model variable rather than a constant. Its MLA geometry also differs from DeepSeek's, as Section 1.1 records, which is why the subset attention cannot assume DeepSeek dimensions and why the MLA dependency in Section 11 includes a GLM-geometry extension. IndexShare, the reuse of a produced selected-index list across subsequent layers, is a GLM feature, and GLM is the reason it is in scope at all. GLM-5.2, the named target in the epic, is confirmed from its published configuration and is identical in geometry to GLM-5, so its column in Section 1.1 is filled rather than assumed. Its configuration also specifies the IndexShare layer pattern explicitly through the indexer_types array and the index_topk_freq and index_skip_topk_offset fields, which layers produce a selection and how far a produced list propagates before a recompute, so that pattern is read from the model configuration rather than assumed, as Section 6 states.

### 12.3 Other sparse-attention kernels

Beyond the two model references, the survey looked for existing sparse-attention and selection primitives that DSA could build on or baseline against, and it found real substrate for the plumbing and a clear gap for the DSA-specific compute. On the selection-gated attention side, the repository already has a block-sparse path, the VSA and Jenga kernels in library/kernels/common/sparse_attention.py, which consume a per-block selected-index lookup table. This is the closest existing analog to attending over a selected subset, but it is block-granular and mask-based, meaning it loads the keys and forces the out-of-set ones to negative infinity rather than skipping them, so it saves no bandwidth and is a conceptual precedent rather than a drop-in for DSA's token-granular gather. The Composable Kernel tree carries the production jenga and VSA sparse FMHA the rocKE port derives from, under ck_tile ops sparse_attn, which is the same block-sparse pattern. On the paged-gather side, the block-table indirection that a selected-KV subset needs already exists, in library/kernels/common/fmha_paged_prefill.py and in the CK paged-KV forward kernel, so the mechanism for turning a selected-index list into paged reads is not new even though wiring it to a per-query selected set is. On the selection side, the existing top-k primitives are small-k and MoE-scale, the CK warp-level block_topk_stream_2d and the topk_softmax path that select a handful of experts, with a CPU reference in ck_tile host reference_topk, so they are the right pattern to mirror but do not cover the large-k selection over the full context that DSA needs, which is genuinely new work. Finally, the repository's moe fused mega fp8 kernel is prior art not for sparsity but for the fused-versus-split and cohort-scoped-crossover disciplines that Section 11.1 records, and it should be treated as such.

What is missing entirely, and therefore what the DSA implementation must build, is the lightning indexer scoring kernel, the large-k top-k over full context, the token-granular gather-based subset-attention loop, and IndexShare, none of which have an in-tree predecessor. As for an external kernel to baseline against, DeepSeek's earlier Native Sparse Attention work is related prior art on learned sparse selection and is worth reviewing for tiling ideas, but whether any ROCm, AITER, or Composable Kernel DSA or lightning-indexer kernel exists to compare against is a to-verify item, and if none exists then the parity baselines in Section 10.3 reduce to the model reference and dense MLA until one appears.

## 13. References

**Internal (rocKE codebase):**

- `library/builders/mla/DESIGN.md` — the MLA design doc this design depends on and mirrors; the subset attention (Section 5) reuses its prefill and decode-absorb kernels.
- `library/dispatch/attention/` — `AttentionRequest`, the candidate registry, and the arch candidate predicates; the dispatch layer Section 9 extends.
- `library/kernels/common/attention_unified.py` — unified attention building blocks reused by the subset attention.
- `library/kernels/common/sparse_attention.py` — the VSA / Jenga block-sparse path; the closest existing selection-gated attention precedent (Section 12.3).
- `library/kernels/common/fmha_paged_prefill.py` — paged block-table gather, the mechanism the Section 5.1 gather builds on.
- `library/kernels/common/fmha_fwd_fp8.py` — the fp8 fnuz vs OCP handling precedent for the format hazard in Section 8.
- `instances/common/moe_fused_mega_fp8_tuned.py` — the fused-versus-split and cohort-scoped-crossover disciplines cited in Section 11.1.
- `library/builders/` and `library/tests/` — the reference and test layout Section 10.1 follows.
- hipDNN plugin boundary: `dnn-providers/hip-kernel-provider/src/core/PluginPublic.cpp`, `src/core/Container.cpp`, and `src/engines/asm_sdpa_engine/` — the rocKE-to-hipDNN seam Section 9 targets.
- Composable Kernel: `projects/composablekernel/include/ck_tile/ops/topk/block/block_topk_stream_2d.hpp` (small-k top-k), `.../ops/sparse_attn.hpp` (jenga / VSA sparse FMHA), and `.../host/reference/reference_topk.hpp` (CPU top-k reference).

**External (models and prior art — where the doc and a source disagree, the source wins; arXiv ids to verify):**

- DeepSeek-V3.2-Exp — the reference DSA model. Config: `huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/config.json`; DeepSeek-V3.2 technical report.
- GLM-5 — config: `huggingface.co/zai-org/GLM-5/blob/main/config.json`.
- GLM-5.2 — config: `huggingface.co/zai-org/GLM-5.2/blob/main/config.json`.
- DeepSeek Native Sparse Attention (NSA), "Hardware-Aligned and Natively Trainable Sparse Attention" — related prior art on learned sparse selection (arXiv:2502.11089).
- DeepSeek-V2 (arXiv:2405.04434) and DeepSeek-V3 (arXiv:2412.19437) — the MLA specification DSA's subset attention rides on.

## 14. Known limits and follow-ups

This section collects the items that are deferred, gated, or to be confirmed, so a reviewer can see in one place what is not settled. Where an item is settled with a working default and a decider, it lives in the section that raises it (see the requirement-versus-knob note in Section 2.1); this list is the residue that is genuinely open.

**To confirm against the model reference, before the indexer budget and correctness gate are trusted:**

- Whether the index-keys are stored per head or shared across heads (Sections 3.1 and 7.1). This swings the index-key cache traffic by roughly 64x and must be read from the DeepSeek reference rather than assumed.
- Whether the DeepSeek-V3.2 indexer applies RoPE to its index-query and index-key, as both GLM models do (Sections 2.2 and 12.1). The indexer kernel applies RoPE if so.
- The exact DeepSeek-V3.2 YaRN mscale factor folded into the softmax scale (Section 1.1).
- Whether any ROCm, AITER, or Composable Kernel DSA or lightning-indexer kernel exists to baseline against (Section 12.3). If none exists the parity baselines reduce to the model reference and dense MLA.

**Dependencies, gating the build rather than this design:**

- The subset attention reuses the MLA prefill and decode-absorb kernels, which are design-only today with no schedule, so the subset-attention deliverables are gated on the MLA build (Sections 5 and 11).
- MLA excludes GLM geometry, so serving GLM through DSA additionally depends on MLA being extended to GLM's dimensions (Sections 1.2 and 11).
- The hipDNN integration is a joint deliverable with the hipDNN team, covering the op count, the IndexShare shared-index tensor persistence, the model-load-time constant weights, and the packaging path (Section 9.4).

**Deferred decisions, settled by measurement at implementation:**

- The fused-versus-standalone top-k fallback, and its crossover by token count if one is needed (Sections 4.1 and 11.1). The fused form is the specified default.
- The top-k selection algorithm, defaulting to streaming argmax-k, with the per-arch candidate-set budget as the decider (Sections 4.1 and 7.2).
- Scatter-read versus gather-to-contiguous per form, and whether both are kept (Section 5.2).
- The sort placement, with top-k or with the gather (Section 4.2).

**Follow-ups:**

1. The fp8 indexer on gfx942, contingent on an arch-native fnuz index-key contract. The safe default is the bf16 indexer on gfx942 with the fp8 indexer on gfx950 (Sections 7.4 and 8).
2. IndexShare, an optimization layered on the working sparse pipeline, scoped as a later phase than the base indexer, top-k, and subset attention (Sections 6 and 11).
3. Any per-model tuning of the indexer and top-k, once the kernels exist and the bottleneck of each stage is measured on device.
