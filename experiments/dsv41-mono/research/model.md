# DeepSeek-V4.1-Flash: model architecture, dtypes, and decode byte/FLOP budget

Research note R0 for the rocKE DSV4.1 decode work on MI355X (gfx950), served by ATOM.
Source is ATOM `origin/main`, read-only through `git show`. Paths below are relative to the ATOM repo root.
This file contains ratios and byte/FLOP counts derived from dims only. Measured numbers live in
`hackathon-prep/private-perf/dsv41-mono/research-numbers.md` (section "## model.md").

Notes on sources:
- Real config values come from `tests/models/deepseek_v41/fixtures/config.json`. This is the HF `config.json` of
  `deepseek-ai/DeepSeek-V4.1-Flash` at revision `dba1be0a...`, pinned by `tests/models/deepseek_v41/README.md`.
- The exact tensor set (names, shapes, dtypes, block sizes) comes from `checkpoint_schema()` in
  `atom/models/deepseek_v41/weights.py`. All parameter counts below come from that function and the config.
- The topology rules come from `build_attention_topology()` / `normalize_hf_config()` in `atom/models/deepseek_v41/config.py`.

---------------------------------------------------------------------------------------------------------

## 1. Headline dims (fixture `config.json`, `text_config`)

| Field | Value | Notes |
|---|---|---|
| `hidden_size` | 5120 | |
| `num_hidden_layers` | 40 | backbone |
| `num_nextn_predict_layers` | 3 | DSpark draft stages (layer ids 40..42 in topology, `mtp.0..2` in ckpt) |
| `vocab_size` | 129280 | embed and head are separate (`tie_word_embeddings=false`) |
| `num_attention_heads` | 64 | query heads |
| `num_key_value_heads` | 1 | MQA, enforced `== 1` in `normalize_hf_config` |
| `head_dim` | 512 | one shared K=V latent row per token (see Sec. 3.1) |
| `qk_rope_head_dim` | 64 | partial RoPE inside the 512-d row; must fit attention and index heads |
| `q_lora_rank` | 1280 | `wq_a` -> `q_norm` -> `wq_b` |
| `o_lora_rank` / `o_groups` | 1024 / 8 | grouped low-rank output (`wo_a` grouped, then `wo_b`) |
| `sliding_window` | 128 | per-layer SWA ring |
| `compress_ratios` | 43 entries: `[0,0, 2 x18, 1 x20, 0,0,0]` | 40 backbone + 3 draft; only 0/1/2 allowed ("CSA2") |
| `compress_rope_theta` | 160000 | YaRN on compressed (global) layers |
| `rope_theta` / `rope_scaling` | 10000 / yarn factor 16, orig 65536, beta 32/1 | window layers use plain theta 10000 |
| `max_position_embeddings` | 1048576 | |
| `kv_source_layer_ids` | [2, 8, 14, 20] | layers that own a compressed KV pool |
| `index_source_layer_ids` | [2, 8, 14, 20, 24, 28, 32, 36] | layers that run an indexer |
| `index_n_heads` / `index_head_dim` / `index_topk` | 32 / 128 / 512 | lightning indexer |
| `candidate_source_layer_id` | 20 | two-level top-k source |
| `candidate_topk_blocks` x `candidate_block_size` | 2048 x 8 | 16384 candidate rows |
| `n_routed_experts` / `num_experts_per_tok` | 384 / 6 | |
| `n_shared_experts` | 1 | enforced `== 1` |
| `moe_intermediate_size` | 2304 | routed and shared both use it (`mono/kernels/dims.py`: `MOE_INTER = SHARED_INTER = 2304`) |
| `scoring_func` / `topk_method` | `sqrtsoftplus` / `noaux_tc` | bias for selection only, `norm_topk_prob=true` |
| `routed_scaling_factor` | 1.5 | |
| `swiglu_limit` | 10.0 | clamped SwiGLU |
| `hc_mult` / `hc_sinkhorn_iters` / `hc_eps` | 4 / 20 / 1e-6 | mHC |
| `rms_norm_eps` | 1e-20 | |
| `engram_layer_ids` | [1, 14] | |
| `engram_num_embeddings` | [384006168, 384016682] | rows per Engram table |
| `engram_max_ngram_size` / `n_heads` / `head_dim` | 4 / 8 / 256 | Engram width = (4-1) x 8 x 256 = 6144 |
| `engram_vocab_size` / `engram_compressed_vocab_size` | 16000000 / 99092 | |
| `dspark_block_size` | 5 | five draft tokens |
| `dspark_target_layer_ids` | [37, 38, 39] | target layer inputs tapped for the drafter |
| `dspark_markov_rank` | 256 | |
| `dspark_n_routed_experts` / `dspark_num_experts_per_tok` | 128 / 3 | draft MoE |
| `dspark_noise_token_id` | 128799 | |
| `quantization_config` | fp8, dynamic acts, `weight_block_size [32,32]`, `scale_fmt ue8m0`, `expert_dtype fp4` | validated by `validate_native_quantization` |
| vision (`vision_config`) | 32 layers, hidden 1024, 16 heads, MLP 2816, patch 14 | not on the text decode path |

Not present in source: `num_hash_layers`. `DeepseekV4Args.from_hf_config` defaults it to 0
(`atom/models/deepseek_v4.py`), so V4.1 has **no hash-routed layers**. V4 had 3. There is also no
`index_topk_freq` or `index_topk_pattern`; V4.1 handles indexer reuse with the explicit owner topology instead.

---------------------------------------------------------------------------------------------------------

## 2. Layer structure

Each backbone layer (`Block` in `atom/models/deepseek_v41/model.py`) is:

```
[Engram (layers 1, 14 only)] -> mHC seam(attn) -> Attention -> mHC seam(ffn) -> MoE (384e top-6 + 1 shared) -> (post owed to next seam)
```

The residual is `[T, hc_mult=4, 5120]` BF16 throughout. `forward_hidden()` expands the embedding over 4 streams
(`SinglePassHCState.from_embeddings`, pre-mix = one-hot on stream 0). It collapses with the pre-mix at the end,
then applies the final `norm` and the BF16 `head`.

### 2.1 Attention topology ("CSA2"), from `build_attention_topology`

`AttentionMode` has four modes: `WINDOW`, `FULL`, `REINDEX`, `REUSE`.

| Layers | ratio | Mode | KV pool owner | top-k owner | candidates |
|---|---|---|---|---|---|
| 0, 1 | 0 | WINDOW (SWA only, 128) | - | - | - |
| 2 | 2 | FULL (compressor + indexer) | 2 | 2 | - |
| 3-7 | 2 | REUSE | 2 | 2 | - |
| 8 | 2 | FULL | 8 | 8 | - |
| 9-13 | 2 | REUSE | 8 | 8 | - |
| 14 | 2 | FULL | 14 | 14 | - |
| 15-19 | 2 | REUSE | 14 | 14 | - |
| 20 | 1 | FULL, candidate **producer** | 20 | 20 | produces 2048 blocks x 8 |
| 21-23 | 1 | REUSE | 20 | 20 | consumer |
| 24, 28, 32, 36 | 1 | REINDEX (indexer only, no compressor) | 20 | self | consumer: scores only inside layer-20's candidate blocks |
| 25-27, 29-31, 33-35, 37-39 | 1 | REUSE | 20 | preceding REINDEX | consumer |
| 40-42 (draft) | 0 | WINDOW | - | - | - (enforced: "V4.1 draft layers must use window-only attention") |

The key structural facts:
- **The compressed KV is shared across layers** (cross-layer KV sharing). Only owners 2, 8, 14, 20 have a
  `Compressor` and allocate global main/index storage (`docs/deepseek_v41_runtime.md`: "Only Full owners 2, 8, 14
  and 20 allocate global main/index storage"). Layers 3-7 attend into layer 2's pool, and so on. Ratio-2 owners
  hold one row per 2 tokens. Owner 20 (ratio 1) holds one row per token.
- **Selections are shared too.** REUSE layers take the top-k row ids of their `topk_owner`. So there are only 8
  distinct top-512 selections per token (the 8 index sources). `_tag_index_groups` groups the layers that share an
  index build into contiguous runs: {2-7}, {8-13}, {14-19}, {20-23}, {24-27}, {28-31}, {32-35}, {36-39}.
- **Every layer still has its own SWA window** of 128 rows. Each layer runs its own `wkv` projection
  (`wqkv_a` second half) and writes its own ring (`cache.rope_quant_window`, `write_window`). STATE holds
  40 rings plus 3 draft rings (`docs/deepseek_v41_dspark.md`: "43 request-owned windows"). With 5 spec tokens a
  physical ring has 133 rows (`V41PoolGeometry.ring_slots = window + speculative_tokens`).
- **Two-level index top-k.** `atom/model_ops/deepseek_v41/indexer.py` describes "level one of the two-level index
  top-k". Layer 20 scores the full ratio-1 context and keeps the top 2048 blocks of 8 rows (the newest visible block
  is pinned). Layers 24/28/32/36 then score only those 16384 candidate rows. They page over the kept list, which
  acts as their block table (`candidate_table.py`, `paged_scoring.score_topk_paged`). Ratio-2 owners 2/8/14 always
  score their full (L/2) context.
- Decode attention (`Attention.forward`, `attention.py`) is a 2-source sparse MQA: (a) the top-512 compressed rows
  from the owner pool, and (b) the layer's own 128-row window through `ring_view`. Kernel:
  `sparse_attn_v4_paged_2src` (BF16 pool) or `packed_decode` (FP4 pool), with a per-head `attn_sink` in the softmax
  denominator. Scale is `512^-0.5`.
- RoPE choice (`forward_hidden`): ratio>0 layers use `global_rope` (theta 160000 + YaRN). Window layers and the
  draft use `window_rope` (theta 10000). Inverse RoPE on the attention output is fused into the `wo_a` MXFP8 quant
  (`inverse_rope_group_quant`).

### 2.2 Per-layer attention math (`attention.py: Attention`)

1. `wqkv_a` (fused `wq_a`[1280x5120] + `wkv`[512x5120], `MergedReplicatedLinear`, FP8 32x32). Gives `q_lora`[1280] and `kv_pre`[512].
2. `qk_norm` = `DualRMSNormMXFP8(q_norm, kv_norm)`, one launch. Emits `(qr, qr_scale)` MXFP8 for both `wq_b`s.
3. `wq_b` [32768 x 1280], `ColumnParallelLinear`, gives 64 heads x 512.
4. `rope_quant_window` (`model_ops/deepseek_v41/rope_window.py`): rotates Q and KV, makes the window row, writes the
   ring (FP8 + E8M0 in the packed layout).
5. Indexer (FULL/REINDEX only, may run on a side stream): `indexer.wq_b` [4096 x 1280] FP8 (replicated) on `qr`.
   `weights_proj` [32 x 5120] BF16 on `hidden` gives per-head weights. Then score the paged index plane and take
   the top-512 (`score_topk_paged` FP8 / `score_topk_quantized` FP4).
6. Compressor (FULL only, may run on a side stream; `model_ops/deepseek_v41/compressor.py`). BF16
   `wkv_gate` [512 or 1024 x 5120], FP32 output for ratio 2. Ratio 2 does a softmax-gated, non-overlapping pool of
   2 tokens. Ratio 1 has no gate: just `wkv` + RMSNorm. There is no learned APE (zeros), unlike V4. At a boundary
   it writes the compressed latent (RoPE'd) to the owner pool, plus the index key
   `indexer.wk`[128x512] BF16 -> `k_norm` -> RoPE -> FP8/FP4 plane.
7. Sparse attention over (512 selected + 128 window) rows, K = V = the 512-d row
   (`tests/models/deepseek_v41/oracle_kernels.py: sparse_attn` uses `values` for both QK and PV).
8. `_project_out`: inverse RoPE + MXFP8 quant, then grouped `wo_a` (8 groups, each [1024 x 4096], FP8,
   `batched_gemm_a8w8_mxscale_bpreshuffle`), then `wo_b` [5120 x 8192] `RowParallelLinear`, all-reduce.

The indexer score follows `deepgemm_fp8_paged_mqa_logits` semantics: per key, the sum over the 32 heads of
`w_h * ReLU(q_h . k)`. `weights_scale = 128^-0.5 * 32^-0.5` (`Indexer.weights_scale`). Inferred from the V3.2/V4
DSA lineage and the kernel name; the kernel body was not read here. Ties go to the smaller position
(`recipes/DeepSeek-V4.1-Flash.md`, "Equal-score index selection"). The selection is returned in ascending order.

### 2.3 Index plane (`model_ops/deepseek_v41/index_plane.py`, `pool_layout/v41_pool_geometry.py`)

| Plane | Format | Bytes / row (128-d key) |
|---|---|---|
| FP8 (default `index_cache_dtype=fp8`) | E4M3, one power-of-two (ue8m0) FP32 scale per key, preshuffled 8/16-row tiles | 128 + 4 = 132 |
| FP4 (`--index-cache-dtype fp4`) | E2M1, E8M0 per 32 dims, page-8 row-group layout, values and scales in 2 planes | 64 + 4 = 68 |

- FP8 query: E4M3 with one FP32 scale per (token, head) row (`quantize_query_rows`). FP4 query: rotated and
  quantized in one launch (`quantize_query_fp4`, aiter `rope_rotate_activation`).
- The published model FP4-rounds both q and k (`docs/deepseek_v41_runtime.md`). The FP8 plane is ATOM's choice.
  The AgentX recipe runs the FP4 plane.
- The index block is 8 rows, so a candidate block equals an index block and the candidate list serves as the
  scorer's block table.

### 2.4 MoE (`atom/models/deepseek_v41/moe.py` -> V4 `MoE` in `atom/models/deepseek_v4.py`)

- Gate: `ReplicatedLinear(5120 -> 384)`, BF16, no quant. At decode widths (T <= 16 rows) it runs the fixed-order
  FlyDSL router GEMV `model_ops/deepseek_v41/router.py` (`EXPERTS=384, HIDDEN=5120, MAX_TOKENS=16`). This is
  bit-identical to the mono K2b router tile.
- Routing: `sqrtsoftplus(logits)`, plus `e_score_correction_bias` used for selection only, then top-6, renormalized,
  times `routed_scaling_factor=1.5` (`FusedMoE.select_experts(scoring_func="sqrtsoftplus", ...)`). There is no group
  top-k (`use_grouped_topk=False`) and no hash layers. A second bias, `gate.bias_vl`, applies to image tokens (`mm_topk`).
- Experts: SwiGLU with clamp (`swiglu_limit=10`: gate <= 10, up in [-10, 10]); `w1`/`w3` [2304 x 5120] and `w2` [5120 x 2304].
- Routed experts: MXFP4 weights (E2M1, E8M0 per 1x32). They take **MXFP8 activations**
  (`args.moe_fp8_activations = True`, i.e. W4A8). The moe.py docstring says aiter's default MXFP4 activations
  put the routed output about 12% off the reference. The recipe says the mono path is refused because its K2b
  still runs MXFP4 activations; that text is stale (K2b is A8W4 on `origin/main`, `atom-v41-mono.md` section 5).
- Shared expert: a separate module (not fused into the routed buffer), FP8 32x32 weights (MXFP8 activations), same 2304 width.
- TP options (`docs/deepseek_v41_runtime.md`): `enable_expert_parallel=True` gives whole experts per rank (primary
  TP4 deployment). Otherwise each expert's intermediate dimension is sharded across TP. The loader pads a rank's
  intermediate share to a multiple of 128 (`FusedMoE pad_align=128`, `dims.py MOE_PAD`):
  TP2 1152 (no pad), TP4 576 -> 640 (+11.1%), TP8 288 -> 384 (+33%). TP8 does not start today: the aiter FP8
  blockscale GEMM does not support K=288 for the shared `w2` (recipe).

### 2.5 mHC: manifold hyper-connections, "Single-Pass / delayed" (`model_ops/deepseek_v41/mhc.py`, `mhc_pre_delayed.py`)

- Residual: 4 streams x 5120, BF16. Per sublayer (attn, ffn) there are FP32 parameters `hc_{attn,ffn}_fn`
  [24 x 20480], `_base` [24], and `_scale` [3]. 24 = hc*(hc+2) = 4 pre + 4 post + 16 comb.
- `predict_mixes`: RMS-normalize the flattened residual (20480), project with `hc_fn` (FP32 GEMV), then
  `hc_split_sinkhorn`. That gives `pre = sigmoid(..)+eps` [4], `post = 2*sigmoid(..)` [4], and
  `comb` = softmax then 20 Sinkhorn row/column normalizations [4x4] (doubly stochastic).
  Reference: `oracle_kernels.hc_split_sinkhorn`.
- **Delayed / single-pass:** the sublayer input is `collapse(residual, pre_mix_carried_from_previous_seam)`. The
  pre-mix computed at this seam is carried to the next sublayer. So the gate projection and the sublayer input
  both read the same residual in one pass. The sublayer output is expanded back with
  `residual' = comb^T . residual + post * out` (`expand_residual` / `aiter.mhc_post`).
- In production one seam is a single fused op: `aiter ... mhc_fused_post_pre_delayed_rmsnorm`. It does the owed
  post of the previous sublayer, the `hc_fn` projection, Sinkhorn, collapse with the carried pre, and the sublayer
  input RMSNorm, as a split-K main kernel plus a per-token reduce. The FFN's post is "owed" and folded into the
  next block's attention seam (`Block.finish_ffn`). Engram layers must `settle()` the owed post first.
- Byte cost: the FP32 `hc_fn` is 2 x 24 x 20480 x 4 B = 3.93 MB per layer, read every step and replicated across
  TP. That is about the same bytes as the BF16 router gate.

### 2.6 Engram (layers 1 and 14; `atom/model_ops/engram/*`)

- An n-gram hashed memory. For each token, hashes of the (compressed-vocab) 2-, 3- and 4-gram suffixes index
  8 heads each. That gives 3 x 8 = 24 rows of 256 dims = 6144 per token per Engram layer
  (`Block.__init__` width = `(max_ngram-1)*n_heads*head_dim`).
- Tables: `layers.{1,14}.engram.embed.weight` [rows, 256] FP8 E4M3 plus `.scale` [rows, 8] E8M0, i.e. FP8 with
  1x32 scales (`checkpoint_schema`). Each table is about 101 GB (384M rows x 264 B). The pair is about 203 GB
  (about 189 GiB). They are **host-resident, mmap'd, gathered on the CPU, staged over UVA**
  (`engram/tables.py: HostEmbeddingTable`, `model_loader/deepseek_v41.py`). The runtime doc says "Host Engram
  lookup still reads final GPU IDs on the CPU; moving the lookup to HBM ... is future work". Under DPA each rank
  registers about 183 GiB of mapped host table.
- On-GPU op (`EngramOp`): `wkv` [(4+1) x 5120 = 25600 x 6144] FP8 32x32, replicated. It produces 4 per-stream keys
  plus 1 shared value. Then `engram_post_wkv` gates the value into each stream by a per-stream score of its key
  against the residual, with gate weights `q_weight * k_weight` (FP32). This happens before the attention seam,
  so the seam cannot fold.
- Per decode token: HBM reads 157 MB of `wkv` per Engram layer (2.4% of B=1 weight bytes). Host-to-device traffic is
  24 x 264 B = 6.3 KB per Engram layer per token.

### 2.7 Embedding / head

`embed` and `head` are both BF16 [129280 x 5120] (`quantized=False`, `tp_axis=0`, vocab-parallel). Each is 1.32 GB.
The head is read in full every step: 10.2% of B=1 weight bytes. It is also read again for the 5 DSpark draft rows
(shared with the drafter, `share_with_target`).

---------------------------------------------------------------------------------------------------------

## 3. Dtypes as ATOM loads them

Dense FP8 here means E4M3 weights with **32x32** E8M0 block scales. This is not V3/V4's 128x128.
`validate_native_quantization` rejects anything else. Activations are dynamic MXFP8 (1x32 E8M0)
(`layers.py: native_quant_config`: `QuantType.per_1x32`, act `float8_e4m3fn`). On gfx950 the weights are
(16,16)-preshuffled at load for aiter's FlyDSL group32 GEMM (`docs/deepseek_v41_performance.md`).

| Tensor | Shape | Weight dtype | Activation into it | TP |
|---|---|---|---|---|
| `attn.wq_a`+`attn.wkv` (fused `wqkv_a`) | [1280+512, 5120] | FP8 32x32 | MXFP8 (from mHC-seam norm) | replicated |
| `attn.wq_b` | [32768, 1280] | FP8 32x32 | MXFP8 (`qk_norm` fused quant) | column |
| `attn.wo_a` (grouped) | 8 x [1024, 4096] | FP8 32x32 (kept FP8; ckpt flag `dequantize=True` refers to the scale) | MXFP8 via fused inverse-RoPE quant | column (groups) |
| `attn.wo_b` | [5120, 8192] | FP8 32x32 | MXFP8 | row + all-reduce |
| `attn.attn_sink` | [64] | FP32 | - | split by head |
| `compressor.wkv` (+`wgate` ratio 2) | [512(+512), 5120] | BF16 | BF16, FP32 out | replicated |
| `indexer.wq_b` | [4096, 1280] | FP8 32x32 | MXFP8 `qr` (reused) | replicated |
| `indexer.weights_proj` | [32, 5120] | BF16 | BF16 | replicated |
| `indexer.wk` | [128, 512] | BF16 | BF16 latent | replicated |
| `ffn.gate` | [384, 5120] | BF16 (+ FP32 `bias`, `bias_vl`) | BF16 | replicated |
| `ffn.experts.{e}.w1/w3/w2` | [2304,5120] / [5120,2304] | **MXFP4** E2M1 + E8M0 1x32 | **MXFP8** (W4A8) | EP whole-expert or TP-sharded |
| `ffn.shared_experts.w1/w3/w2` | same | FP8 32x32 | MXFP8 | column/row |
| `hc_{attn,ffn}_{fn,base,scale}` | [24,20480],[24],[3] | FP32 | FP32 math on BF16 residual | replicated |
| norms | | BF16 | | replicated |
| `engram.wkv` | [25600, 6144] | FP8 32x32 | FP8 | replicated |
| `engram.embed` tables | [~384M, 256] x2 | FP8 + E8M0/32 | host | host |
| `embed`, `head` | [129280, 5120] | BF16 | BF16 | vocab-parallel |
| draft `main_proj` | [5120, 15360] | FP8 32x32 | MXFP8 | replicated |
| draft `markov_head.embed/head` | [129280, 256] each | BF16 | | vocab-parallel |
| draft `confidence_head.proj` | [1, 5120+256] | BF16 | | |

Caches (`docs/deepseek_v41_performance.md` table, `V41PoolGeometry`):

| Region | `kv_cache_dtype=bf16` | `kv_cache_dtype=fp4` (packed) |
|---|---|---|
| Main compressed KV (512-d) | BF16, 1024 B/row | E2M1 + E4M3 scale per 16 dims: 256+32 = 288 B/row |
| SWA ring (512-d) | BF16, 1024 B/row | FP8 E4M3 + E8M0/32: 512+16 = 528 B/row |
| Index keys (128-d) | FP8 132 B/row (default) or FP4 68 B/row, independent of main | same |
| Compressor tails | FP32 | FP32 |

DSpark requires `kv_cache_dtype=bf16` (`validate_speculative_config`).

Stored KV per cached token across all owners: 3 ratio-2 owners at 0.5 rows each plus 1 ratio-1 owner = 2.5 main
rows and 2.5 index rows per token.
- BF16 main + FP8 index: 2.5 x (1024+132) = 2890 B/token.
- FP4 main + FP4 index: 2.5 x (288+68) = 890 B/token.

Plus fixed STATE per request: 43 rings x 133 rows x 1024 B (BF16), about 5.9 MB.

---------------------------------------------------------------------------------------------------------

## 4. DSpark draft (`atom/models/deepseek_v41/dspark.py`, `model_ops/deepseek_v41/dspark.py`, `draft_block.py`, `docs/deepseek_v41_dspark.md`)

- **What it is:** a block-parallel drafter built from the checkpoint's 3 `mtp.*` stages. It is not autoregressive
  MTP. Each stage is a full `Block` (`DraftBlock`): mHC, window-only attention (ratio 0, own 128-window), and a MoE
  with **128 routed experts, top-3** + 1 shared, same 2304 width, MXFP4/FP8.
- **Context:** the target's layer **inputs** at layers 37, 38, 39, each taken as the mean of the 4 residual
  streams (`_target_layer_input`). They are concatenated [15360] and passed through `mtp.0.main_proj` (FP8) and
  `main_norm`. Then each stage's `wkv` makes context keys. These are written into that stage's draft window ring
  after every target forward (`write_context_kv`; the 3 stages' `wkv` fuse into one GEMM).
- **Block:** 5 rows per request: the anchor token's embedding in column 0, then the cached noise-token
  (`128799`) embedding (`build_block_state`), at positions anchor+1..anchor+5. Attention is **bidirectional inside
  the 5-row block** and causal into the 128-row context window (`draft_step`, `draft_attention`).
- **Sampling:** the shared target `head` (BF16, vocab-parallel) on the 5 rows, plus a low-rank first-order
  **Markov bias**. That bias is `W1[x_{k-1}] @ W2^T`, rank 256 (`DSparkMarkovHead`), applied sequentially
  across the 5 positions. A `DSparkConfidenceHead` (proj [1, 5376]) gives per-position survival probabilities
  for dynamic verification.
- **Chain, not tree:** 5 proposals, verified as a chain. Target verification runs anchor + 5 = **6 rows**. Greedy
  verification uses target argmax. Stochastic verification samples from the target per row and stops at the first
  mismatch (not ratio rejection sampling). Only the accepted prefix is committed. Ring slack is 5 rows (133-row rings).
- `num_speculative_tokens` must be 5 and `method="dspark"`, from the same checkpoint (`validate_speculative_config`).
  "Dynamic" verification (ragged widths, confidence schedule) needs a calibration profile.
- The decode step is two graph replays: the draft graph (keyed by request count) and the target graph (keyed by
  batch size and query bucket). ATOM's mono path covers target steps of up to 48 rows (8 requests x 6) and the draft
  backbone (up to 8 requests x 5 rows), as K1 + K2 (= K2a + K2b) persistent kernels per layer
  (`mono/config.py`: `MAX_ROWS = 48`, `MAX_REQUESTS = 8`). The recipe's "up to 12 rows / two requests" and
  "currently refused (MXFP4-activation K2b)" are stale: on `origin/main` K2b is A8W4 and the bind gate passes for the
  shipped MoE (see `atom-v41-mono.md` section 5).
- Draft weights: about 14.2 B params (7.2 GB routed MXFP4, 0.38 GB attention FP8, 0.13 GB Markov BF16, ...).

---------------------------------------------------------------------------------------------------------

## 5. Parameter and byte totals (derived from `checkpoint_schema` + config)

Bytes per parameter used here:
- FP8 32x32: 1 + 1/1024
- MXFP4 1x32: 0.5 + 1/32 = 0.53125
- BF16: 2
- FP32: 4

| Group | Params | Bytes (GPU) |
|---|---|---|
| Routed experts, 40 x 384 x 35.39 M | 543.6 B | 288.8 GB (MXFP4) |
| Attention projections, 40 x 126.6 M | 5.06 B | 5.07 GB (FP8) |
| Shared experts, 40 x 35.39 M | 1.42 B | 1.42 GB (FP8) |
| embed + head | 1.32 B | 2.65 GB (BF16) |
| Engram `wkv` x2 | 0.31 B | 0.31 GB (FP8) |
| gate / mHC / indexer / compressor | ~0.12 B | 0.40 GB |
| **Backbone total** | **551.9 B** | **298.6 GB** |
| DSpark (3 stages, 128e) | 14.2 B | 7.9 GB |
| **GPU-resident total** | **566 B** | **306.6 GB (285.5 GiB)** |
| Engram tables (host) | 196.6 B entries | 202.8 GB (188.8 GiB) |

This is consistent with the README's "475 GiB tensor payload" (285.5 + 188.8 = 474.3 GiB, plus vision).

**Active parameters per decoded token (backbone, B=1): about 16.1 B.** That is 40 x (126.6 M attention +
6 x 35.4 M routed + 35.4 M shared + 2.0 M gate + 1.0 M mHC), plus the head 0.66 B, Engram `wkv` 0.31 B, and
indexer/compressor about 0.03 B. Dense (non-routed) params are about 7.6 B of that, so routed are about 53% of active params. In bytes the routed
share is only 35%, because routed weights are MXFP4.

### Fit on MI355X (288 GB HBM per GPU)

| Config | Weights per GPU (approx.) | Fits? |
|---|---|---|
| TP1 | 306.6 GB (106% of HBM) | **No.** Weights alone exceed HBM even with the host Engram tables. |
| TP2 | ~154 GB (53%) | Yes, about 120 GB left for KV/state/workspace |
| TP4 | ~78 GB (27%) | Yes (primary deployment, whole-expert EP) |
| TP8 | ~40 GB (14%) | Fits, but does not start today (shared `w2` K=288 unsupported) |

The replicated part (about 1.08 GB) is counted per GPU and the rest is split 1/TP.

---------------------------------------------------------------------------------------------------------

## 6. Per-token decode budget at batch 1 (TP1 accounting, weights read once per step)

### 6.1 Per-layer components (one backbone layer, one token)

| Component | Weight bytes | FLOPs (2 x MACs) | Notes |
|---|---|---|---|
| `wq_a`+`wkv` (FP8) | 9.18 MB | 18.4 M | replicated under TP |
| `wq_b` (FP8) | 41.98 MB | 83.9 M | |
| `wo_a` grouped (FP8) | 33.59 MB | 67.1 M | |
| `wo_b` (FP8) | 41.98 MB | 83.9 M | + all-reduce of 5120 |
| **Attention projections subtotal** | **126.7 MB** | **253 M** | |
| Sparse attention, 640 rows (512 sel + 128 win) x 64 heads x (512 QK + 512 PV) | KV: 640 x 1024 B = 655 KB (BF16) | 83.9 M | layers 0-1: 128 rows only |
| Indexer `wq_b` + `weights_proj` (8 layers) | 5.58 MB | 10.8 M | |
| Indexer scoring (8 layers) | index plane rows x 132 B (FP8) | rows x 8192 | rows: L/2 (layers 2,8,14), L (layer 20), min(L,16384) (24..36) |
| Compressor (4 layers) | 10.5 MB (ratio 2) / 5.2 MB (ratio 1) BF16 | 10.5 / 5.2 M | `wk` 128x512 runs only at boundaries |
| Router gate (BF16) | 3.93 MB | 3.9 M | replicated |
| Routed experts, 6 x (w1,w3,w2) MXFP4 | 6 x 18.80 = 112.8 MB | 6 x 70.8 M = 425 M | + all-reduce of 5120 |
| Shared expert (FP8) | 35.42 MB | 70.8 M | |
| mHC 2 seams (`hc_fn` FP32) | 3.93 MB | 2 x 0.98 M + Sinkhorn | replicated; Sinkhorn is 20 iters on 4x4 |
| **Layer total (typical CSA layer)** | **~282.8 MB weights + ~0.66 MB KV** | **~837 M** | |

### 6.2 Whole step at B=1 (TP1 accounting)

Weight bytes are about **13.03 GB per token**. Share by component:

| Component | Share of B=1 weight bytes |
|---|---|
| Attention projections (FP8) | 38.9% |
| Routed experts (6 active, MXFP4) | 34.6% |
| Shared expert (FP8) | 10.9% |
| LM head (BF16) | 10.2% |
| Engram `wkv` (FP8) | 2.4% |
| Router gate (BF16) | 1.2% |
| mHC `hc_fn` (FP32) | 1.2% |
| Indexer + compressor weights | 0.6% |

KV / index bytes per token per step (not amortized by batch; L = context length):

| L | KV fmt / index fmt | Selected KV (38 layers x 512 rows) | Distinct selected (8 sets) | Windows (40 x 128) | Index scan | Total (naive) | Ratio to B=1 weight bytes |
|---|---|---|---|---|---|---|---|
| 8K | BF16 / FP8 | 19.9 MB | 4.2 MB | 5.2 MB | 7.0 MB | 32.2 MB | 0.25% |
| 8K | BF16 / FP4 | 19.9 MB | 4.2 MB | 5.2 MB | 3.6 MB | 28.8 MB | 0.22% |
| 8K | FP4 / FP4 | 5.6 MB | 1.2 MB | 2.7 MB | 3.6 MB | 11.9 MB | 0.09% |
| 32K | BF16 / FP8 | 19.9 MB | 4.2 MB | 5.2 MB | 19.5 MB | 44.6 MB | 0.34% |
| 32K | BF16 / FP4 | 19.9 MB | 4.2 MB | 5.2 MB | 10.0 MB | 35.2 MB | 0.27% |

Observations:
- Context barely matters to decode bytes. CSA2 caps attention at 640 rows per layer, and only the 8 indexer
  layers scan context. Above 16K, the four REINDEX layers are capped at 16384 candidate rows. Only layers
  2/8/14 (L/2) and 20 (L) grow with L.
- Within an index group the selected KV rows are **identical across layers** (same owner pool, same top-k), so
  38 x 512 row reads collapse to 8 distinct sets. A fused/persistent kernel can keep a group's 512 selected rows
  (512 KB BF16) resident in L2/MALL across the 4-6 layers of the group. This cuts selected-KV HBM traffic up to
  4.75x. It is an opportunity rather than something ATOM does today: ATOM's attention is per layer, while its
  index build is per group (`attention_indices` "one launch writes its index group's").

FLOPs per token:
- Weight GEMVs: about 32.3 GFLOP (2 x 16.13 B).
- Sparse attention: about 3.2 GFLOP (40 layers; 24576 row-visits x 131072).
- Indexer scoring: 0.44 GFLOP at 8K, 1.21 GFLOP at 32K.

Arithmetic intensity at B=1 is about 2.7 FLOP/B, which is deeply bandwidth-bound. The sparse attention has
intensity of about 128 FLOP per KV byte (BF16, 64 heads sharing one KV head). At 64 heads and B=1 it is still
small in absolute terms.

### 6.3 Batch scaling (TP1 accounting, uniform-routing estimate)

Expected distinct experts touched per layer: `U(B) = E * (1 - (1 - k/E)^B)`, with E=384, k=6. This assumes
independent uniform routing. Real routing is skewed and correlated across neighbouring tokens, so treat it as an
upper bound on unique experts:

| B | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|
| U(B) | 6.0 | 11.9 | 23.4 | 45.5 | 85.5 | 152.0 | 243.8 | 332.8 |
| Fraction of experts touched | 1.6% | 3.1% | 6.1% | 11.8% | 22% | 40% | 64% | 87% |
| Step bytes / step bytes(B=1) | 1.00 | - | 2.01 | 3.29 | 5.61 | 9.48 | 14.84 | 20.12 |
| Bytes per token / B=1 | 1.000 | - | 0.503 | 0.411 | 0.351 | 0.296 | 0.232 | 0.157 |
| KV+index share of step bytes (8K, BF16/FP8) | 0.2% | - | 0.5% | 0.6% | 0.7% | 0.8% | 1.1% | 1.6% |

(32K values are within 1% of 8K values.) Dense weights (about 8.5 GB, read once per step) amortize quickly. The
routed experts dominate from B of about 8 upward. At B=128 the step reads about 87% of the 289 GB routed set, so
a large-batch decode step costs about 20x a B=1 step. Each touched expert sees only about B*k/U(B) tokens
(1.0 at B=1, about 2.3 at B=128). Its intensity is about 3.8 FLOP per weight byte per token (MXFP4), so the MoE
stays deeply bandwidth-bound over the whole decode batch range.

---------------------------------------------------------------------------------------------------------

## 7. Bandwidth-floor ratios across TP (B=1, bytes per GPU per step)

Replicated on every rank (each rank reads the whole thing):
- `wqkv_a`, gate, mHC `hc_fn`, indexer, compressor, Engram `wkv`: about 1.08 GB per step.
- The selected/window KV and the index scan. MQA has one KV head and the indexer is replicated by design
  (`checkpoint_schema`: "Replicate the small indexer: scores must reduce all its heads before top-k").

Split 1/TP: `wq_b`, `wo_a`, `wo_b`, shared expert, head, and the routed experts (TP-sharded with padding, or EP).

| Config | Bytes/GPU/step, no KV | Ratio to TP1 | With 8K KV (BF16/FP8) ratio to TP1 | Ratio to TP2 (TP-sharded) |
|---|---|---|---|---|
| TP1 | 13.03 GB | 1.000 | 1.000 | 1.85 |
| TP2, experts TP-sharded | 7.06 GB | 0.541 | 0.543 | 1.000 |
| TP2, whole-expert EP | 7.76 GB | 0.595 | 0.596 | 1.10 |
| TP4, experts TP-sharded (576 to 640 pad) | 4.19 GB | 0.322 | 0.323 | 0.594 |
| TP4, whole-expert EP | 5.05 GB | 0.387 | 0.389 | 0.715 |
| TP8, TP-sharded (288 to 384 pad) | 2.76 GB | 0.212 | 0.214 | 0.391 |
| TP8, EP | 3.63 GB | 0.279 | 0.280 | 0.515 |

- TP scaling efficiency of the floor: TP2 gives 1.85x over TP1 and TP4 gives 3.10x (TP-sharded). The replicated
  part (about 1.1 GB plus KV) is 16% of per-GPU bytes at TP2, 26% at TP4 and 40% at TP8.
- EP at B=1 is load-imbalanced. With 6 picks over TP ranks of 96 experts each, the expected busiest rank holds
  E[max] = 3.93 (TP2), 2.80 (TP4) and 2.16 (TP8) experts. Compare 6/TP = 3, 1.5 and 0.75 for TP-sharding.
  So at B=1, TP-sharded experts are about 17% (TP4) below EP in bytes, even after the 11% TP4 padding. EP wins
  on collective volume and kernel shape at larger B.
- The per-GPU step at TP4 (TP-sharded) grows from 4.22 GB (B=1) to 21.3 GB (B=16) and 55.9 GB (B=64), ratios
  1 : 5.05 : 13.3. At TP2 the step goes 7.09, 37.5, 98.6 GB.
- There are 2 TP all-reduces of [T x 5120] per layer (attention `wo_b` and MoE), 80 per step plus the
  embedding/head collectives. At B=1 these are pure latency, which is one of the motivations for a persistent
  mono kernel with in-kernel CCL.
- An ideal floor in time is per-GPU bytes / HBM bandwidth. The MI355X spec bandwidth and the absolute floor
  values are in the private-perf file. Use the ratios above here.

### DSpark verify cost as a byte ratio (uniform routing, BF16 KV, 8K)

One spec step = draft (3 stages x 5 rows + head + Markov) + target verify (6 rows):

| | TP1 | TP2 | TP4 |
|---|---|---|---|
| Distinct target experts per layer (6 rows) | 34.6 | 34.6 | 34.6 |
| Spec-step bytes / non-spec step bytes | 2.88 | 2.75 | 2.64 |
| Draft share of spec step | 7.6% | 7.4% | 6.9% |
| Bytes per accepted token / non-spec, acceptance 3.0 | 0.96 | - | - |
| Bytes per accepted token / non-spec, acceptance 3.51 (recipe's fixed value) | 0.82 | 0.78 | 0.75 |

The 6 verify rows touch about 5.8x the experts of 1 row under uniform routing. That is why speculative decode only
breaks even in bytes at an acceptance of about 2.6-2.9. Correlated routing across the 6 consecutive rows lowers
this, so it is the pessimistic bound. The draft is cheap: 128 experts top-3, window-only attention.

---------------------------------------------------------------------------------------------------------

## 8. Implications for the rocKE decode design

1. Bytes are dominated by FP8 attention projections (39%) and MXFP4 routed experts (35%) at B=1. Experts dominate
   from B >= 8. Shared expert + BF16 head add another 21%. GEMV efficiency on FP8 32x32-block and MXFP4 1x32 is
   the main lever.
2. The 32x32 E8M0 weight block (not 128x128) means one scale byte per 1 KB. Activations are MXFP8 per 32.
   The routed experts must run **W4A8** (MXFP8 activations) to match reference accuracy. ATOM's mono K2b already
   does (the recipe's "refused" note is stale, `atom-v41-mono.md` section 5); any rocKE MoE must too.
3. Attention is a tiny-KV sparse MQA (640 rows x 512-d, K=V). The indexer scan is the only context-length-
   dependent cost. Group-level reuse of the selected rows (8 distinct sets for 38 layers) is a unique fusion
   opportunity.
4. The replicated parts (the `wqkv_a` + gate + FP32 `hc_fn` + Engram `wkv` + indexer, and all KV) cap TP scaling at
   about 3.1x for TP4. FP32 mHC weights are as large as the BF16 router, and both are re-read on every rank.
5. TP1 is infeasible on one MI355X (306.6 GB weights). TP2 is the smallest config, TP4 the primary, TP8 blocked
   by the K=288 GEMM.
6. Engram is host memory plus a large replicated FP8 `wkv`. Keep it out of the critical path (UVA prefetch) or
   move the tables to HBM only at TP>=4 (about 51 GB per rank if the 203 GB is sharded 4 ways; this is an option, not in ATOM).
