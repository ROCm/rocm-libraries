# Running Qwen3.8 with hipDNN

This page covers what hipDNN needs to run Qwen3.8 (Hugging Face `model_type: qwen3_5`):

- how each model op maps onto hipDNN graph operations;
- where a decoder layer has to be split into separate graphs;
- which engines already exist, and which FlyDSL kernels could serve the rest.

`test_sdk/tests/utilities/cpu_graph_executor/TestQwen38LayerGraphs.cpp` builds these exact graphs and checks them on the CPU reference executor against a direct transcription of the model. It covers both layer types, prefill followed by decode, embedding and greedy sampling, and the routed MoE block.

## Model

Values below are from `Qwen/Qwen3.8-27B` `config.json` (`text_config`).

| Part | 27B dense | 2.4T MoE flagship |
|---|---|---|
| Layers | 64 = 48 Gated DeltaNet + 16 gated full attention (`full_attention_interval: 4`) | same backbone |
| Gated DeltaNet | 16 key heads × 128, 48 value heads × 128, causal conv width 4 | same |
| Full attention | 24 q / 4 kv heads, head dim 256, QK RMSNorm, partial rotary 0.25 (leading 64 dims), interleaved M-RoPE (sections 11/11/10), output gate | same |
| MLP | SwiGLU, intermediate 17408 | softmax router, top-k, renormalised, expert SwiGLU, sigmoid-gated shared expert |
| Norms | RMSNorm with a zero-centred weight (`1 + w`, folded into the hipDNN scale) | same |
| Other | vocab 248320, untied lm_head, bf16; FP8 block-scaled checkpoints and FP8 KV cache are common | same |

The vision tower is out of scope here. Everything it uses is already in hipDNN: Conv3d patch embed, LayerNorm, GELU-tanh, 2-D RoPE via `rope` with per-patch `freqs`, and varlen non-causal SDPA via ragged tensors.

## Op inventory

| Model op | hipDNN graph op | Notes |
|---|---|---|
| Token embedding | `gather` (new) | `ids [1, T]` INT32/INT64 into `table [V, H]` → `[1, T, H]` |
| RMSNorm, QK-norm, gated-norm RMSNorm | `rmsnorm` | Normalised dims follow the scale shape: `[1,1,H]`, `[1,1,1,D]`, `[1,1,V]` |
| Linear projections, lm_head | `matmul`; FP8 via `block_scale_dequantize` → `matmul` | Packed QKV/QKVZ/q+gate projections are split host-side into one weight per output, so no slice is needed |
| Partial RoPE, M-RoPE | `rope` (new; cuDNN `ROPE_FWD` + `rope_offset_ext`) | `rope_dim = 64`, `rope_offset = 0`. `freqs` holds raw angles per token, so M-RoPE is the caller choosing the T/H/W position per frequency |
| KV-cache write | `paged_cache_store` (new) | `slot_mapping` into the SDPA paged layout `[blocks, H, block_size, D]`. Accepts `[T,H,D]` or SDPA-native `[B,H,S,D]`; optional FP8 `scale` |
| Causal GQA attention, prefill / chunked prefill / decode | `sdpa` with `seq_len_q/kv`, `page_table_k/v`, bottom-right causal | Already in the API |
| Output gate, SiLU, sigmoid, residual, SwiGLU | `pointwise` (`SIGMOID_FWD`, `SWISH_FWD` β=1, `MUL`, `ADD`) | |
| GDN causal conv + SiLU with conv state | `causal_conv1d` (new) | Token-packed with `cu_seqlens`. Optional initial/final state (may share a buffer), state pool via `state_indices` |
| Gated delta rule (QK L2 norm, softplus gate, sigmoid beta, recurrent state) | `gdn` | Existing schema, now with frontend, backend and CPU reference. `safe_gate` + `a_log` + `dt_bias` is exactly Qwen's `g = -exp(A_log)·softplus(a + dt_bias)` |
| Router softmax, sampling softmax | `softmax` (new; cuDNN `Softmax_attributes`) | Last dim, optional log-sum-exp stats |
| Router top-k, greedy argmax | `topk` (new) | Sorted best-first; ties go to the lower index |
| Top-k renormalisation, weighted combine | `reduction` ADD + `pointwise` DIV/MUL | |
| Expert offsets and row indices | `moe_routing` (new) | `topk_ids [S, k]` → `first_token_offset [E+1,1,1]`, `token_index`, `token_ks` |
| Expert GEMMs | `moe_grouped_matmul` GATHER (gate/up) and SCATTER (down) | Now accepts cuDNN's explicit `[G+1]` boundaries |
| Top-p / multinomial sampling | — | Not covered: done in the framework |

## Graph boundaries

hipDNN has no reshape node, and a tensor has one set of dims within a graph. A decoder layer is therefore split wherever a buffer changes rank. The caller binds the same device buffer to the next graph with new dims and strides.

| Layer | Graphs |
|---|---|
| Gated DeltaNet | (1) input RMSNorm + q/k/v/z/b/a projections `[1,T,·]` → (2) three `causal_conv1d` on `[T,C]` → (3) `gdn` on `[T,H,K]` + gated RMSNorm + `swish(z)·` → (4) out-proj + residual + MLP |
| Full attention | (1) input RMSNorm + q/gate/k/v projections → (2) QK-norm + `rope` + `paged_cache_store` ×2 + `sdpa`, all in `[B,H,S,D]` dims with BSHD strides over the projection buffers → (3) `sigmoid(gate)·` + out-proj + residual + MLP |
| MoE | (1) router matmul + `softmax` + `topk` + renormalise → (2) `moe_routing` on `[S,k]` → (3) GATHER gate/up + SwiGLU + SCATTER down → (4) weighted sum over k + shared expert → (5) add |

## Engine status

Every op above runs on the CPU reference executor. GPU engines today:

| Op | GPU engine |
|---|---|
| `matmul` (+bias/activation, MX FP8) | hipBLASLt provider |
| `rmsnorm` | HIP MLops; FlyDSL `flydsl_rmsnorm` (RDNA, gfx90a/942/950) |
| `sdpa` dense causal GQA | ASM SDPA, rocKE, FlyDSL. No shipped engine accepts paged KV or `seq_len_*` yet |
| `pointwise` binary (FP16/FP32) | MIOpen provider |
| `gather`, `rope`, `paged_cache_store`, `causal_conv1d`, `gdn`, `softmax`, `topk`, `moe_routing`, `moe_grouped_matmul`, paged/varlen `sdpa` | none yet |

## FlyDSL kernel coverage

Sources: FlyDSL `v0.5.0` (`226cde8d`) and AITER `main` (`a9918248`, `aiter/ops/flydsl/`). The provider currently pins FlyDSL 0.3.4 (`dnn-providers/hip-kernel-provider/flydsl/generators/_flydsl_env.py`), so vendoring any of these kernels first needs the pin bumped and the shipped objects regenerated (`flydsl/REGEN.md`).

| hipDNN op | Upstream kernel | Gap to hipDNN semantics | Archs (source evidence) | Effort |
|---|---|---|---|---|
| `causal_conv1d` | AITER `kernels/causal_conv1d_prefill.py`, `causal_conv1d_update.py` | Packed varlen, state pool and SiLU match; channels and width are compile-time | wave-size generic | vendor |
| `gdn` | AITER `kernels/gdr_prefill/*` (fused and chunked prefill), `kernels/gdr_decode.py` | Fused prefill is gfx950 and head dim 128 only; gate is precomputed outside the fused kernel; state dtype | gfx942/gfx950 | adapt |
| `rope` | FlyDSL `kernels/attention/fused_rope_cache_kernel.py` | No partial rotation (rejects `rotary_dim != head_dim`); takes cos/sin caches, not raw angles | generic | adapt |
| `paged_cache_store` | write half of `fused_rope_cache_kernel.py` | Flash `[nb,bs,H,D]` and ATOM layouts only, not `[nb,H,bs,D]`; per-tensor FP8 scale | FP8 path gfx942+ | adapt |
| `sdpa` paged / varlen | vendored CDNA `flash_attn_generic.py` (has `cu_seqlens` + block table, unused today); `pa_decode_tile.py` | Page size tied to the KV tile; per-batch `seq_len` must be turned into cumulative offsets | CDNA; decode gfx942/gfx950 | adapt (CDNA), new (RDNA) |
| `softmax` | FlyDSL `kernels/norm/softmax_kernel.py` | Row width is compile-time, so baked tiers are needed | generic | adapt |
| `softmax` + `topk` (router) | FlyDSL `kernels/moe/topk_gating_softmax_kernel.py` | Fused pattern only; `E / VPT` a power of two | generic | vendor as a fused engine |
| `topk` | AITER `kernels/topk/*` | FP32 input, INT32 indices | generic | adapt |
| `moe_routing` | FlyDSL `kernels/moe/moe_sorting_kernel.py` | Emits CK-style padded sorted ids, not `[E+1]` offsets plus row indices | generic | adapt |
| `moe_grouped_matmul` | FlyDSL `kernels/moe/moe_gemm_2stage/*` | FP8/INT8/INT4 only; no bf16 on `first_token_offset` | CDNA | new |
| `gather` | none (closest: AITER `moe_gather_reduce.py`) | Needs 64-bit row offsets for a 248320 × 5120 table | — | new |
| gated RMSNorm `rmsnorm(x)·silu(z)` | none (vendored `rmsnorm_kernel.py` has fused-add only) | — | — | adapt rmsnorm |
| standalone pointwise | none (device helpers only in `kernels/common/act.py`) | — | — | new |

FP8 KV and FP8 GEMM paths need `cvt_pk_fp8_f32` / FP8 MFMA, which gfx90a and gfx11 lack. Their matchers must decline FP8 there.

Each new FlyDSL engine follows the `flydsl_rmsnorm` template:

1. vendored source under `flydsl/kernels_src/`, with a provenance header;
2. `_ops.py`, `_instances.py` and `gen_<op>.py`;
3. descriptors from `gen_descriptors.py`, with objects in DVC;
4. `packs/Flydsl<Op>Native.cpp` with `graph_match`/`kernel_match`/`score`/`dispatch`;
5. registration in `IngestorPacks.{hpp,cpp}`;
6. engine, object and census tests;
7. `flydsl/README.md`, `COVERAGE.md` and `REGEN.md` sections.
