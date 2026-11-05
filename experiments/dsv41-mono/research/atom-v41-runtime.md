# ATOM DeepSeek-V4.1-Flash decode runtime (native + atom_vllm plugin)

Scope: how one V4.1-Flash decode step travels through ATOM `origin/main` on MI355X (gfx950). It covers the
scheduler, the runner, metadata, graph replay, the per-layer ops and which kernel library each op calls, the sampler,
DSpark propose/verify/commit, TP/EP/DPA options and host-bound risks. Every claim cites an ATOM path read through
`git show origin/main:<path>`. The doc has no absolute performance numbers; the absolute figures from ATOM docs are in
`hackathon-prep/private-perf/dsv41-mono/research-numbers.md` (section `## atom-v41-runtime.md`). The mono internals
(`atom/models/deepseek_v41/mono/kernels/*`) belong to another doc. Only the mono plug-in points are covered here (§8).

Legend: **[code]** means read in source. **[doc]** means stated in ATOM docs/recipes only. **[est]** means my own
estimate, not trace-verified.

---

## 0. Model facts that shape the runtime

From `tests/models/deepseek_v41/fixtures/config.json` (production-shaped fixture) and
`atom/models/deepseek_v41/config.py::build_attention_topology`:

| Item | Value | Runtime consequence |
|---|---|---|
| Layers | 40 target + 3 DSpark draft stages (`num_nextn_predict_layers=3`) | `compress_ratios` has 43 entries |
| hidden / heads / head_dim | 5120 / 64 / 512 (MQA latent, 1 KV head); q_lora 1280, o_lora 1024, o_groups 8 | `wo_a` is a grouped GEMM (8 groups) |
| mHC | `hc_mult=4`, Sinkhorn 20 iters | residual is 4 streams; two "seams" per layer |
| compress_ratios | L0-1: 0 (window only); L2-19: 2; L20-39: 1; draft: 0 | ratio-2 owners need FP32 compressor tails |
| KV owners (`kv_source_layer_ids`) | 2, 8, 14, 20 → `AttentionMode.FULL` (compressor + indexer) | only 4 layers run a compressor |
| Index owners | 2,8,14,20 (FULL) + 24,28,32,36 (`REINDEX`, scored over 2048 candidate blocks x 8 rows from L20) | 8 layers score; the other 28 are `REUSE` (read the owner's selection) |
| Indexer | 32 heads x 128, top-512, sliding window 128 | at most 1024 attention keys per token (`mono/config.py ATTENTION_KEYS_MAX`) |
| MoE | 384 routed (top-6, `sqrtsoftplus`, `noaux_tc`, scale 1.5) + 1 shared; routed MXFP4 W / MXFP8 A; shared FP8 | `models/deepseek_v41/moe.py` forces `moe_fp8_activations=True` |
| Engram | layers 1 and 14; 4-gram, 8 heads x 256; host hash tables (UVA) | host or UVA lookup on every step (§7 H3) |
| DSpark | block 5, draft taps at target layer **inputs** 37/38/39, Markov rank 256, draft MoE 128 experts top-3 | verify = anchor + 5 rows; 43 windows a request (`docs/deepseek_v41_dspark.md`) |

---

## 1. Decode step, end to end: native ATOM server

`atom.entrypoints.openai_server` → `EngineCore.busy_loop` → `_process_engine_step_inner`
(`atom/model_engine/engine_core.py`):

1. **Schedule** (host, engine-core process): `Scheduler.schedule()` (`atom/model_engine/scheduler.py:1871`) builds a
   `ScheduledBatch`. Under DSpark every decode reserves `mtp_k+1 = 6` token slots per request
   (`model_runner.capture_cudagraph` comment: "charging `mtp_k + 1` tokens per sequence"). Prefill and decode are not
   mixed in one batch (`prepare_input_ids`: "TODO: remove this when we support mixed prefill and decode").
2. **RPC to workers**: `runner_mgr.call_func("forward", batch, wait_out=True)`. The engine core blocks until every TP
   worker returns. The same synchronous IPC hop as Qwen3.8.
3. **Worker `ModelRunner.forward`** (`atom/model_engine/model_runner.py:3481`):
   1. `_advance_forward_vars()` / `_gate_staging_reuse()`: rotate or wait on the pinned staging ring, so the CPU does
      not overwrite buffers whose H2D copy is still in flight.
   2. `prepare_model(batch)` (`:2708`):
      - `_dspark_apply_q_bucket` (fixed-width DSpark: no-op);
      - `ForwardMode.decide(...)` picks eager vs graph, `running_bs`, `running_tokens` and `max_seqlen_q`, plus the DP
        all-reduce when DP>1;
      - `prepare_sample` (temperatures, top-k/p, `all_greedy` checked on CPU);
      - `attn_metadata_builder.publish_cu_seqlens_q`;
      - `tokenID_processor.prepare_input_ids` (`:511`). **Host sync**: `recv_mtp_status_async()` →
        `copy_done.synchronize()` waits on the previous step's D2H of `num_rejected` / `num_bonus`. The rest of
        prepare_input_ids:
        - fills `input_ids.np`, with a per-new-request Python loop for draft columns;
        - publishes the H2D group;
        - launches `fill_deferred_decode_ids` (device kernel). It scatters the previous step's sampled and draft ids,
          which never went to the host, into the padded graph width.
      - `prepare_inputs` → V4.1 `DeepseekV41MetadataBuilder.prepare_decode`
        (`atom/model_ops/attentions/deepseek_v41/backend.py:1012`). That calls `_prepare` → `_prepare_step`, which:
        - runs a Python loop over requests that builds `RequestSpan`s and block-table rows;
        - builds `_build_compress_plans` per ratio, at fixed capacity with sentinel tails;
        - stages `ENGRAM_ROWS`;
        - calls `cache.begin_step(...)` (positions, slots, visible, tentative staging);
        - calls `_assemble_metadata`.
        Then `prepare_model_inputs` (`:1093`) runs the Engram host lookup or the UVA batch, `cache.prepare_state`, and
        `_write_engram_cursor` / tentative prefix staging.
   3. `run_model` (`:3044`), graph branch:
      `self.graphs[(running_bs, max_seqlen_q)].replay()`. With DSpark, `max_seqlen_q = 6`.
      - `hidden = forward_vars["outputs"][:scheduled_tokens]`.
      - `logits = graph_logits[...]` only when `logits_in_graph`. That requires `world_size == 1` and no TBO (`:4088`).
        **At TP≥2 `compute_logits` (final `norm` + LM head + vocab gather) runs eagerly after the replay.**
   4. `drafter.compute_draft_kv(positions, hidden, ...)` (`atom/spec_decode/dspark_proposer.py:584`) writes the target
      context into the draft's windows (`DeepseekV41DSpark.write_context_kv`; fused Triton under
      `ATOM_DSPARK_FUSED_CTX_KV=1`). **Eager, every target forward.**
   5. `postprocess` (`:3276`), §3.
4. **Return**: the `ScheduledBatchOutput` carries *the previous step's* tokens (deferred output; `is_deferred_out`).
   The engine core runs `scheduler.postprocess` (appends tokens and placeholders, maps `dspark_ell`) and streams
   outputs.

Prefill differences:
- Runs eager (graphs capture only decode shapes; `docs/deepseek_v41_performance.md` §Graph ownership).
- Uses decoder SWA bounded replay (§6).
- The draft still runs `compute_draft_kv` and a draft block pass after prefill.

## 1b. Decode step: atom_vllm plugin path

`vllm serve ... --tensor-parallel-size 4 --tokenizer-mode deepseek_v4 --no-enable-prefix-caching`
(`recipes/atom_vllm/DeepSeek-V4.1-Flash.md`):

- **Runner**: vLLM's **V2 model runner** and vLLM scheduler. The validated configuration is TP4 with PP/DCP/PCP 1.
- **KV**: one fake proxy attention layer (`AtomDeepseekV41ProxyAttention`, `atom/plugin/vllm/deepseek_v41_bridge.py`)
  is declared as a uint8 byte arena. V4.1 carves its planes out of it.
  - `--block-size` is forced to 256, so a vLLM block id is the V4.1 PAGE id.
  - The STATE tail is reserved by `deepseek_v41_state_reserve_patch`, which withholds 2x`max_num_seqs` blocks: half
    for real requests, half for capture.
- **Per-step metadata**: `AtomDeepseekV41ProxyMetadataBuilder.build` → `snapshot_v41_batch` (`:962`).
  - It does **blocking D2H reads** of `seq_lens` (`common_attn_metadata.seq_lens[:n].cpu()` when `seq_lens_cpu` is
    absent) and of the PAGE table columns (`block_table_tensor[...].cpu().numpy()`) when the V2 `InputBatch` has no
    host table.
  - The doc says it outright: "This path is not synchronization-free" (§Host buffers).
  - Then `v41_stage_step` → ATOM's `VllmDeepseekV41MetadataBuilder._prepare` (`atom/plugin/vllm/v41_core_overrides.py`).
  - `_v41_stage_outside_forward` stages Engram, state reset and cursor advance *before* the model call, so nothing
    per-step sits inside a captured region.
- **Graphs**: off by default. They need both `ATOM_V41_EXPERIMENTAL_CUDAGRAPH=1` and
  `VLLM_USE_BREAKABLE_CUDAGRAPH=1` (`tests/plugin/test_vllm_deepseek_v41_cudagraph_mode.py` holds this shut).
  - PIECEWISE: attention runs in an `eager_break_during_capture` segment break, through a caller-owned
    `_attn_out_buffer`. Engram `_prefetch`, `_await_layer` and `_await_fork` are also eager breaks, which keeps
    Engram's side-stream overlap.
  - FULL: no breaks. Engram is joined before capture/replay, which loses that overlap.
  - The historical GSM8K validation ran FULL_AND_PIECEWISE.
- **DSpark**: vLLM's own DSpark drafter (`DSparkDeepseekV4ForCausalLM`), always 5 tokens.
  `atom/plugin/vllm/v41_speculative.py` adds three hooks:
  - the target LM-head view `V41TargetLMHeadView` (without it, acceptance length collapsed to 1.04);
  - `patch_v41_speculative_scheduling`, which never mixes a prefill into a verify step;
  - `patch_v41_speculative_commit`, which commits the tentative Engram cursor after `postprocess_sampled`.
- **Refused on the plugin**: image requests, prefix caching (the PAGE hit lacks the STATE rings/cursor), every KV
  connector, and mono. Mono's `config_refusal` → `common_refusals` lists "no plugin mode" (env doc,
  `ATOM_MONO_ENABLE` entry).

---

## 2. Graph modes

| Setting | Native default | Source |
|---|---|---|
| CLI `--level` | **3** | `atom/model_engine/arg_utils.py:337` |
| CLI `--cudagraph-mode` | **FULL** (choices NONE, PIECEWISE, FULL, FULL_AND_PIECEWISE, AF_PIECEWISE) | `arg_utils.py:340` |
| Config default when the mode is unset at level 3 | FULL for `deepseek_v41_text`, PIECEWISE for every other model. Reason given: "Engram stream fork/join currently requires a FULL graph" | `atom/config.py:~2412` |
| V4.1 admission | eager, FULL or PIECEWISE at level 0. **Level 3 requires FULL** (or eager). Levels 1/2 are refused | `models/deepseek_v41/config.py::validate_runtime_config` |
| Doc-stated acceptance baseline | level 0 (+FULL). Level-3 compiler correctness is "deferred" in the main recipe, but the AgentX benchmarks ran **level 3 FULL** | `recipes/DeepSeek-V4.1-Flash.md`, `...-Agentic.md` |
| Plugin | graphs off. Opt-in FULL or PIECEWISE (two env flags) | §1b |

**FULL** (manual capture in `ModelRunner.capture_cudagraph`, `model_runner.py:3991`):
- One `torch.cuda.CUDAGraph` per `(bs, max_q_len)` with a shared `graph_pool`. Under fixed-width DSpark `max_q_len = 6`.
  Ragged or confidence-scheduled DSpark adds q-buckets (`_dspark_capture_q_buckets`).
- The capture covers the **whole target forward**: embedding, 40 layers including Engram, the MoE collectives and
  `wo_b`'s all-reduce ("The FFN and its collective reductions are captured with the rest", runtime doc), the final mHC
  collapse, and the drafter aux-tap copies.
- `compute_logits` is in the graph only at TP1 (`logits_in_graph`).
- The draft has **its own graph**, keyed by request count (`DraftGraph`, `atom/spec_decode/draft_graph.py`;
  `DSparkProposer._declare_draft_graphs`). It holds the backbone plus `capture_epilogue=True`: LM head, Markov sampler
  and confidence head.
- **A decode step is two replays**: draft and target (`docs/deepseek_v41_dspark.md`).
- Capture sizes:
  - default `cuda_graph_sizes=[512]` → `[1..8] + range(16, 512+1, 16)` (`config.py::capture_sizes`);
  - then filtered to `≤ min(max_num_seqs, max_num_batched_tokens // (mtp_k+1))`;
  - the AgentX recipe passes `[1..8,16,32,48,64,128]`, or every size 1..32 at c32;
  - a batch between two sizes replays the next larger graph, and `fill_deferred_decode_ids` zero-fills the pad.
- What makes FULL possible for V4.1:
  - the scorer reads its bounds from the device;
  - compression plans use fixed capacity with sentinel tails;
  - padding rows use batch id −1 and zero-length `cu_seqlens_q`;
  - host cursor publication happens in input prep, not in the forward (`docs/deepseek_v41_performance.md`).
- What **forces eager**:
  - prefill;
  - `step_needs_eager`: the FP4 packed score workspace overflows on shared-prefix rows
    (`backend.py:380` → `score_workspace.fits`);
  - bs above the largest capture size;
  - a DP peer in prefill (`run_model` comment).

**PIECEWISE**:
- The `support_torch_compile` graphs: `_Backbone`, or `_EarlyLayers` + `_LateLayers` under bounded replay
  (`atom/models/deepseek_v41/runtime.py`).
- Split at the opaque custom ops `v41_attention`, `v41_engram`, `v41_begin_forward`, `v41_end_forward`,
  `v41_mhc_pre_delayed` and `v41_record_tbo_expert_output` (all `torch_compile_guard`).
- Dense pieces are captured per `num_tokens` bucket. **Attention (indexer, compressor, sparse attn, `wo_*`) stays
  eager between the pieces.**
- The runner also forces aiter `ca_comm` onto the "unreg" copy-in capture path (`_force_aiter_unreg_capture_for_piecewise`,
  `:3797`), because the registered direct-read path is only safe under one whole-forward graph.
- At level 3 V4.1 refuses PIECEWISE, so the native path effectively never runs it in production.

**torch.compile**:
- At level 3, `_Backbone` / `_EarlyLayers` / `_LateLayers` and the DSpark inner model are compiled
  (`ATOM_DSPARK_DISABLE_COMPILE=1` makes the draft eager).
- Under FULL, Inductor output is what gets captured. It only matters for the fused elementwise glue between custom
  ops. Every heavy op is an opaque aiter/FlyDSL/Triton call.
- Under bounded replay, layers 20 (decoder KV source) and 39 run *outside* the compiled graphs. In decode they are
  still inside the FULL capture (`runtime.py::DeepseekV41RuntimeModel.forward` → `run_decoder_stages`).

---

## 3. DSpark: propose, verify, accept

Components:
- `atom/spec_decode/dspark_proposer.py::DSparkProposer` (the V4.1 "inline draft" flavor, `_with_draft=False`);
- `atom/models/deepseek_v41/dspark.py::DeepseekV41DSpark`;
- `atom/spec_decode/dspark_verify.py::VerifyScheduler`;
- `atom/model_ops/rejection_sampler.py`;
- `atom/model_ops/attentions/deepseek_v41/speculative.py::TentativeState`.

**Shapes**:
- Draft width T = `num_speculative_tokens` = 5 = `dspark_block_size`.
- Verify rows per request = anchor + 5 = **6**. The batch is rectangular `[bs, 6]` by default.
- The ragged / confidence-scheduled verify (`DSparkConfig(confidence_schedule=True, ragged=True,
  calibration_profile=...)`) shrinks request i to `max(ell_i, num_bonus_i)+1` rows
  (`ModelRunner._dspark_apply_ragged`, `:2425`). **Under FULL graphs it stays rectangular** ("nothing flat is
  recorded ... the replay is rectangular either way", `:2501`). Ragged only pays under eager or PIECEWISE.
- V4.1 also refuses a dynamic verify without a calibrated profile.

**Draft pass** (`propose` → `self.block.run(running_bs, anchor_ids, anchor_positions)`):
1. `_stage_anchors` writes the anchor ids and positions into fixed staged inputs at `running_bs`.
2. `block_backbone` (`models/deepseek_v41/dspark.py:486`):
   - one `cache.read_windows` gather of the 3 draft context windows;
   - `draft_step_indices` (one kernel);
   - `draft_hidden`: embed the anchor only (the noise-token row is cached), then **3 draft blocks** over `[B,5]` rows.
     Each block is full V4.1-style: mHC seams, attention over its own window plus a bidirectional block, and an MoE
     with 128 experts top-3.
3. `head_and_sample`:
   - LM head `tgemm.mm` on the local vocab shard;
   - **5 sequential `dspark_markov_argmax_tp` calls**, each a fused Triton bias-GEMV + argmax plus an `(max,id)` TP
     exchange (`_head_and_sample_sharded`; `ATOM_DSPARK_FUSED_MARKOV_SAMPLE=1`);
   - `confidence_head`.
   - All of it is inside the draft graph (`capture_epilogue=True`).
4. With `confidence_schedule`, `VerifyScheduler.compute_ell` runs sync-free and keeps ell from a fixed generation 2
   steps back, so the D2H wait is free (`_ELL_GENERATION=2`).
5. `prepare_draft_ids` (deferred mode): returns the **previous** step's draft ids via `recv_async_output_draft()`
   (event sync), then queues this step's D2H.

**Verify and accept** (`postprocess`, `model_runner.py:3276`):
1. `index_select` the bonus and target logits rows.
2. If the batch is not all-greedy: `sampler.sample_verification_tokens` + **TP broadcast** of the target draws.
3. `sampler(bonus_logits)`, then `rejection_sampler.forward(...)`:
   - Triton `rejection_greedy_sample_kernel`;
   - or `rejection_synthetic_sample_kernel` when `--spec-decode-acceptance-length/-rate` is set;
   - output: `(num_bonus, num_reject, anchors)`.
4. **TP broadcast of `sampled_tokens`** (`get_tp_group().broadcast`) whenever a drafter exists.
5. `send_mtp_status_to_cpu_async` (D2H on `async_copy_stream`, no wait).
6. `next_token_ids = gather(sampled, next_token_locs)` stays on the GPU as `prev_token_ids` for the next
   `fill_deferred_decode_ids`.
7. `propose_draft_token_ids`:
   - `attn_metadata_builder.commit_speculative_state(metadata, anchors)` → `TentativeState.commit`, one Triton
     `_commit_cursors` launch (Engram cursor of the accepted prefix);
   - `drafter.propose(...)`.

**No KV rollback**:
- Rings have 133 physical rows against 128 logical, and compressor rings have the same slack. A rejected row only
  changes visibility, because later writes land past the window the next round reads (`speculative.py` docstring;
  `docs/deepseek_v41_dspark.md` "physical rings have 133 rows").
- Compress plans get `extra_write = speculative_tokens` (`backend.py::_prepare_step`).
- **`bounded_replay.py` is unrelated to speculation.** It is the prefill-time decoder/encoder SWA bounded replay (§6).

**Acceptance semantics**:
- Greedy: verify equals target argmax.
- Stochastic: per-row independent draws, accept-on-match, stop at the first mismatch. This is not probability-ratio
  rejection sampling.
- The AgentX benchmarks use **fixed synthetic acceptance length 3.51** (`--spec-decode-acceptance-length 3.51`). They
  measure runtime cost at an assumed acceptance, not real acceptance.
- The mono recipe reports GSM8K acceptance around 50% per draft token, mono and original alike
  (`recipes/DeepSeek-V4.1-Flash.md` table).

---

## 4. Per-decode-step kernel inventory (original, non-mono path, FULL graph, `ATOM_DSV41_SIDE_STREAMS=0`)

Layer body: `Block.forward` (`models/deepseek_v41/model.py`) → `prepare_attention` → `Attention.forward`
(`models/deepseek_v41/attention.py`) → `prepare_ffn` → `MoE` (`models/deepseek_v41/moe.py` over
`models/deepseek_v4.py::MoE`).

| # | Op | Implementation (library) | Selected by / env | Launches [est] |
|---|---|---|---|---|
| 1 | mHC attn seam (owed post + gate proj + Sinkhorn + collapse + attn RMSNorm) | `aiter.ops.triton.fusions.mhc_fused_post_pre_delayed_rmsnorm` (aiter **Triton**); split-K main + per-token reduce (`model_ops/deepseek_v41/mhc_pre_delayed.py`) | fixed | 2 |
| 1e | Engram (L1, L14 only): `state.settle()` then `EngramOp` | Triton/aiter device kernels + UVA gather (`model_ops/engram/device/*`); host lookup if `ATOM_ENGRAM_UVA=0` | `ATOM_ENGRAM_UVA` (default 1) | +4..8 a layer |
| 2 | `wqkv_a` (merged q_a + kv) FP8 32x32-block GEMM | aiter **FlyDSL** preshuffled group32 GEMM; input quant on the fly (hidden_scale None) | `ATOM_GROUP32_WEIGHT_PRESHUFFLE=1` (default); 0 → row-major/Triton `dot_scaled` | 1-2 |
| 3 | `qk_norm` (q RMSNorm + MXFP8 quant, kv RMSNorm) | `DualRMSNormMXFP8` (`atom/model_ops/layernorm.py`), one launch | fixed | 1 |
| 4 | `wq_b` (ColumnParallel FP8) | FlyDSL group32 GEMM (pre-quantized `qr`) | as 2 | 1 |
| 5 | Indexer `project` (8 FULL/REINDEX layers): `wq_b` (replicated), `weights_proj`, rope / FP4 query quant | FlyDSL GEMM ×2 + aiter `rope_rotate_activation` / `dynamic_per_token_scaled_quant` | `index_cache_dtype` fp8 / fp4 | +3..4 |
| 6 | Compressor (4 FULL layers): `wkv_gate` BF16 GEMM, `fused_compress_attn`, `update_compressor_states`, then index-key write (`wk` GEMM + k_norm + rope + `write_index`, or one aiter FP4 writer) | GEMM (tgemm / hipBLASLt) + ATOM **Triton** `v4_kernels/fused_compress.py`, `state_writes.py`, `index_write.py` | fixed | +5..8 |
| 7 | Indexer `score` (8 layers): `scale_indexer_weights`, logits, top-k | FP8: aiter **Triton** `deepgemm_fp8_paged_mqa_logits`; FP4: aiter **FlyDSL** `flydsl_pa_mqa_logits_fp4`; aiter `top_k_per_row_decode` (`model_ops/deepseek_v41/paged_scoring.py`) | `index_cache_dtype` | +3..4 (+1 candidate table on L20 / REINDEX) |
| 8 | `rope_quant_window` (q/kv RoPE, KV FP8 bytes, window write) | ATOM **Triton** `model_ops/deepseek_v41/rope_window.py` (aiter `rope_cached_positions_fwd_inplace` on other paths) | fixed | 1 |
| 9 | `attention_indices` + `write_window` | ATOM Triton (`attentions/deepseek_v41/indices.py`, `cache.py`) | fixed | 1-2 |
| 10 | Sparse decode attention, 2 sources (selection in pool + window ring) | aiter **OPUS** `pa_sparse_prefill_opus` (split-K across blocks for small token counts); Triton fallback | `ATOM_FORCE_ATTN_TRITON` | 1-2 |
| 11 | `inverse_rope_group_quant` (inverse RoPE + MXFP8 1x32) | aiter `aiter.ops.inverse_rope_group_quant` | fixed | 1 |
| 12 | `wo_a` grouped GEMM | aiter `batched_gemm_a8w8_mxscale_bpreshuffle` | fixed | 1 |
| 13 | `wo_b` RowParallel FP8 GEMM + **TP all-reduce** | FlyDSL group32 GEMM (+ quant) + aiter custom AR (`ca_comm`, registered-buffer capture under FULL) | `reduce_results=True` | 2-3 |
| 14 | mHC FFN seam (attn post folded in) | aiter Triton fused seam | fixed | 2 |
| 15 | Router GEMV (decode widths) | ATOM **FlyDSL** fixed-order router GEMV (`model_ops/deepseek_v41/router.py`, `router_supported`), bit-shared with mono | width check | 1 |
| 16 | Top-k select (sqrtsoftplus + bias + renorm + scale) | `FusedMoE.select_experts` → aiter biased top-k | fixed (image rows use `mm_topk`) | 1 |
| 17 | Routed experts A8W4 (MXFP4 W, MXFP8 A, gate/up interleaved) | aiter **`fused_moe`** (`aiter.fused_moe`; 2-stage CK/asm/FlyDSL picked by aiter's tuned table) via `Mxfp4MoEMethod` (`atom/model_ops/moe.py:1250`). On gfx950 Triton MoE is off unless `ATOM_USE_TRITON_MOE` / `ATOM_USE_TRITON_GEMM` | `ATOM_USE_TRITON_MOE`, `ATOM_MOE_GU_ITLV`, `--moe-backend` | 3-5 (sort, act-quant, stage1, stage2) |
| 18 | Shared expert (FP8, separate module) | FP8 GEMM gate_up + act/quant + down. Dual-stream on `alt_stream` when tokens ≤ `ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD` (1024) | `models/deepseek_v4.py::dual_stream_moe_forward` | 3-4 |
| 19 | combine (routed+shared) + **TP all-reduce** | `combine_outputs` → `tensor_model_parallel_all_reduce` (aiter custom AR) | — | 1-2 |

Non-layer work per step:

| Stage | What | Where | Launches [est] |
|---|---|---|---|
| Input prep | `fill_deferred_decode_ids`, H2D group copies (input_ids, cu_seqlens, V4.1 metadata, Engram rows), compress-plan uploads | `model_runner.py:511`, backend `_prepare_step` | 5-15 (copies + 1-3 kernels) |
| Embedding | `VocabParallelEmbedding` + TP all-reduce | in graph | 2 |
| Tail | `state.collapse()` + `norm` + LM head GEMM + vocab all-gather (eager at TP≥2) | `runtime.compute_logits` | 3-5 |
| Sampler / verify | 2× `index_select`, greedy argmax (`_apply_greedy_tokens`), rejection kernel, `gather`, TP broadcast(s), D2H copies | `postprocess` | 8-15 |
| Draft context KV | `write_context_kv` (fused Triton) + aux-tap copies (in the target graph) | `compute_draft_kv` | 1-3 |
| Commit | `_commit_cursors` | `speculative.py` | 1 |
| Draft graph | window gather + `draft_step_indices` + 3 draft blocks (~same op list as a REUSE layer, MoE 128/top-3) + head GEMM + 5×(Markov argmax + TP exchange) + confidence | `models/deepseek_v41/dspark.py` | ~100-130 |

**Launch-count estimate [est, from the op lists above, not a trace]**:
- REUSE layer (28 of them): ≈ 26-32.
- FULL layer (4): ≈ 40-48.
- REINDEX layer (4): ≈ 32-38.
- Engram layers: +4-8 each.
- **Target graph ≈ 1,150-1,400 kernel nodes**, plus draft graph ≈ 100-130, plus eager host-side kernels outside
  both graphs ≈ 15-30 at TP≥2 (logits, sampler, rejection, broadcasts, commit, context-KV).
- Collectives per step: target ≈ 81 all-reduces (embed + 2 per layer). The draft adds 3 blocks × 2 + embed + 5
  Markov exchanges. Two to three extra broadcasts in postprocess.
- **Host launches per step: 2 `hipGraphLaunch` plus ~15-30 eager launches plus the H2D copies.**
- Mono (§8) replaces the 40 layers' bodies with 2 persistent kernels a layer: K1 and K2 = K2a+K2b. That is 80 launches
  plus the compressor (4 layers) and Engram (2 layers) left on the original modules. Roughly a 10-15x cut in target
  graph nodes [est].

---

## 5. Parallelism: TP / EP / DPA, collectives, env

| Config | Status | Source |
|---|---|---|
| TP4 + whole-expert EP (`enable_expert_parallel=True`) | "primary deployment configuration"; quality baseline at level 0 | runtime doc §Current execution scope; `recipes/DeepSeek-V4.1-Flash.md` |
| TP4, experts TP-sharded (no EP) | supported; relative throughput is workload-dependent; **the only arrangement mono serves** | same; env doc `ATOM_MONO_ENABLE` |
| TP2 no EP | GPU benchmark at level 3 FULL (AgentX) | `recipes/DeepSeek-V4.1-Flash-Agentic.md` |
| TP1 | config tests only. Experts would have to fit one GPU | recipe table |
| TP8 | **does not start**: the shared-expert down-proj K = 288 is not a multiple of aiter's gfx950 FP8 blockscale GEMM K tile. Mono kernels compile-check only | recipe table and §Fused decode |
| DPA (`--tensor-parallel-size 4 --enable-dp-attention`) | four TP1-attention ranks. Experts TP-sharded via DP gather/scatter. Text only. Idle ranks join collectives with empty metadata | runtime doc §Data-parallel attention |
| DPA4 + MoRI EP4 (high-throughput), `MORI_SHMEM_HEAP_SIZE=16GiB` | validated with BF16 KV, FP8 index, DSpark5, prefill TBO | runtime doc §Prefill TBO |
| Prefill TBO | DPA only (>1 effective DP rank); `ATOM_TBO_PREFILL_MIN_TOKENS`, `ATOM_TBO_PREFILL_TOKEN_SPLIT`. Decode TBO refused | runtime doc; `validate_runtime_config` |
| PP / CP / DP-without-DPA / EPLB / RapidServe / online quant / non-lmcache KV transfer | refused before load | `validate_runtime_config` |

Collectives:
- **TP all-reduce**:
  - `tensor_model_parallel_all_reduce` (aiter dist) → aiter custom all-reduce (`ca_comm`), registered-buffer capture
    under FULL;
  - `wo_b` (`RowParallelLinear(reduce_results=True)`) and MoE `combine_outputs` each do one;
  - `ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION` (default 1) cannot fire here, because V4.1's norm sits inside the mHC seam
    and not right after the all-reduce [inference from `model.py` structure].
- **MoE all-to-all**: only with DP>1 *and* EP (`FusedMoEConfig.selected_all2all_backend`, `moe.py:213`). It is MoRI
  unless `ATOM_DISABLE_MORI_EP` or `--moe-all2all-backend rccl/none`. Plain TP4+EP therefore uses an expert mask plus
  the TP all-reduce, with no all-to-all.
- **Vocab**:
  - LM head all-gather (target, eager at TP≥2);
  - the draft uses per-position `(max,id)` exchanges instead of a full-vocab gather (`ATOM_USE_CUSTOM_ALL_GATHER`).
- **Engram**: an all-gather at the end of the lookup. The UVA fallback is group-wide (`ATOM_ENGRAM_UVA` doc).
- Recipe env:
  - AgentX: `OMP_NUM_THREADS=4 ATOM_NUMA_BIND=0 ATOM_DISABLE_MMAP=true AITER_LOG_LEVEL=WARNING`;
  - scheduler regression: `AITER_REUSE_IDENTICAL_COMM_GROUPS=1`.
- Recipe flags (AgentX): `--kv_cache_dtype bf16 --index-cache-dtype fp4 --max-num-seqs 128
  --max-num-batched-tokens 16384 --enable_prefix_caching --block-size 16 --state-checkpoint-interval-tokens 8192
  --level 3 --cudagraph-mode FULL --method dspark --num-speculative-tokens 5`.

**AgentX ratios** (from the recipe tables, TP2 vs TP4, no EP, synthetic acceptance 3.51):
- Per-user speed (X = 1000/P90 ITL), TP4/TP2: c2 1.11, c8 1.33, c16 1.35, c32 1.44, c64 1.63.
- Per-GPU throughput (Y), TP2/TP4: c8 1.86, c16 1.77, c32 1.74, c64 1.53.
- Per-user decay from low to high concurrency: TP2 c1→c64 0.086; TP4 c2→c64 0.142.
- At c1-2, TP4 barely beats TP2 per user (1.11). That is the signature of a **latency/launch-bound, not
  bandwidth-bound,** small-batch decode, where halving per-GPU weight bytes buys little. This is exactly mono's target
  cell.

**Mono vs original step time** (`recipes/DeepSeek-V4.1-Flash.md`, TP4, level 0 FULL, draft + verify, from the trace):
- 1 request: mono/original ≈ **0.34**;
- 2 requests: ≈ **0.45-0.56**;
- GSM8K and acceptance are within noise.
- The recipe text says mono is currently refused: the routed experts moved to MXFP8 activations, but K2b still uses
  MXFP4 activations, so every layer falls back.

**Correction (from the mono agent; see `research/atom-v41-mono.md`):** that recipe text is stale.
- `atom/models/deepseek_v41/mono/weights.py` `bind_layer` requires `is_guinterleave` and `fp8_activations`, which
  matches the A8W4 MXFP8-activation experts that `deepseek_v41/moe.py` sets up (`moe_fp8_activations=True`).
- ATOM commit `7fc0926c` reports GSM8K with every decode step running on mono.

Treat mono as active on eligible deployments (no EP, FULL or eager, `ATOM_DSV41_SIDE_STREAMS=0`, ≤48 rows a step), not
refused.

---

## 6. Bounded replay (prefill only) and why it touches decode

- `--decoder-swa-bounded-replay` (default on):
  - a final prefill chunk runs layers 20..38 on only the last W=128 rows, and layer 39's query on only the last row
    (`models/deepseek_v41/bounded_replay.py`, `runtime._replayed`);
  - the forward becomes two compiled graphs (`_EarlyLayers` 0..19, `_LateLayers` 21..38), with L20 and L39 in eager
    Python between and after them.
- **Decode, warmup, draft and padded steps run every layer on every row.** Inside a FULL decode capture the split does
  not matter.
- Under DPA each rank reports its decoder and last-layer token counts in the step's one packed DP sync
  (`dp_stage_report` → `sync_dp_metadata`).
- `--encoder-swa-bounded-replay` (default on) replaces state checkpoints with a one-PAGE rebuild on a prefix hit.
- These are accuracy-approximate modes, post-trained by DeepSeek. They are not speculative rollback.

---

## 7. Host-side cost centers (with Qwen3.8 cross-reference)

Qwen3.8 learnings (`experiments/qwen38-true-mono/D0.md` §6 item 7, `HANDOFF.md`, `research/q38-learnings.md`):
- The binding path at the round boundary was: round t verify → **host MTP-status wait in round t+1's
  `prepare_input_ids`** → about 0.11 of a round of host metadata prep → graph launch.
- The host waited about 0.27 of every round.
- The scheduler, IPC and eager drafter were *off* the critical path.
- GPU-tail savings reappeared as idle until `ATOM_Q38_RB_DEVMETA` prepared round t+1 at full acceptance and shifted the
  accept-dependent metadata on the device.
- A profiler-inflated trace overstated the idle (0.07-0.08 vs 0-0.03 unprofiled).

V4.1 runs the **same generic runner**, so that structure carries over unchanged. Below, ranked by expected exposure at
low concurrency, where mono is aimed:

| # | Risk | Where | Same as Qwen3.8? | Notes |
|---|---|---|---|---|
| H1 | **MTP-status D2H wait at the start of every step** (`recv_mtp_status_async` → `copy_done.synchronize()`) | `model_runner.py:322-355, 511-533` | yes, identical code | The prepared metadata depends on `num_rejected` (`prepare_decode`: `starts = context_lens - (num_spec_step+1) - rejected`). An RB_DEVMETA-style device-side shift is needed to take the host off the boundary. The V4.1 metadata is much richer than Qwen's (§H2), so the shift kernel is bigger. |
| H2 | **V4.1 metadata prep in Python** | `backend.py::_prepare` / `_prepare_step` / `cache.begin_step` / `_build_compress_plans` / `prepare_model_inputs` | heavier than Qwen | Per-request loops (`RequestSpan`, block rows, compress plans for 2 ratios, `extra_write` slack, Engram rows), plus `cache.prepare_state` and tentative staging. Expect a larger share of the round than Qwen's ~0.11 [est]. The prime candidate for device-side prep. |
| H3 | **Engram lookup** | `backend.prepare_model_inputs`, `EngramInputPreparer` | new for V4.1 | The UVA path (`ATOM_ENGRAM_UVA=1`, default) hashes on the device. The host path costs a very large per-step CPU time with the GPU idle (env doc). The runtime doc says "Host Engram lookup still reads final GPU IDs on the CPU" and "moving the lookup to HBM ... is future work". Under mono, Engram staging must `join()` before K1 because its side-stream all-gather would deadlock beside persistent kernels (`mono/runner.py:forward`). That is a serialization point. |
| H4 | **Eager LM head at TP≥2** (`logits_in_graph = world_size==1`) | `model_runner.py:4088, 3262` | Qwen's served path also kept head/sampler outside the graph | Norm + head GEMM + vocab all-gather run eagerly after the replay. Then come sampler/rejection/broadcast, the eager `compute_draft_kv`, the draft-graph replay and the commit. That is several host-launched segments between two graph replays. |
| H5 | **TP broadcasts in postprocess** (verification draws when not greedy; `sampled_tokens` always with a drafter) | `postprocess` | yes | Device collectives, not host syncs, but extra serialized launches on the critical path. |
| H6 | **Previous-step draft ids D2H** (`recv_async_output_draft` → `event.synchronize()`) | `prepare_draft_ids` | yes | Lagged by one step, so usually landed; still a host wait point. |
| H7 | **Synchronous engine-core ↔ worker IPC** (`call_func("forward", wait_out=True)`) and scheduler `postprocess` per step | `engine_core.py:467` | yes (Qwen: off critical path when the host leads) | The deferred-output design lets the worker return before the GPU finishes. The host leads by about one step. |
| H8 | **Graph padding to the next capture size** (bs between ladder rungs replays the larger graph; padded rows still run every kernel at a fixed width) | `prepare_input_ids` `fill_to`, `ForwardMode` | yes | Sparse ladders (AgentX: 8→16→32→48) waste GPU time at odd batch sizes. Capture every size up to the expected concurrency (the recipe does at c32). |
| H9 | **Eager fallback steps**: FP4 score workspace overflow (`step_needs_eager`), bs above the ladder, a DP peer in prefill | `backend.py:380`, `run_model` | partly | An eager V4.1 decode is ~1.2-1.4k host launches [est]. That is catastrophic for latency, so watch for it in traces. |
| H10 | **Plugin-path blocking D2H** of `seq_lens` and PAGE tables every step, plus PIECEWISE eager attention breaks | `deepseek_v41_bridge.py::snapshot_v41_batch`; `v41_core_overrides.py` | worse than native | The plugin path is unsuitable as the mono host; mono refuses plugin mode anyway. |
| H11 | **Ragged/confidence verify gives no FULL-graph benefit** (stays rectangular) and needs a calibration profile; SPS calibration replay is skipped under ragged | `_dspark_apply_ragged`, `_maybe_calibrate_dspark_sps` | n/a | A mono / true-mono design should keep fixed-width 6-row verifies (mono handles ≤48 rows a step). |
| H12 | **Side streams** (`ATOM_DSV41_SIDE_STREAMS` 1/2 put the compressor/indexer on alt streams; dual-stream shared expert) | `model.py`, `attention.py` | — | Mono requires `SIDE_STREAMS=0`. The FULL graph captures the stream forks as graph branches. Fine for the original path, but they contend with persistent kernels. |

Fixed-overhead outlook:
- At c1-2 the AgentX TP4/TP2 per-user ratio (1.11) and the mono step ratio (≈0.34 at 1 request) both say the original
  path is dominated by per-op fixed cost: ~1.3k launches, ~80+ all-reduces, a two-replay round.
- Once mono collapses the layers, the **non-layer and host share grows to dominate** (Qwen3.8's lesson: GPU-tail
  savings convert only after the round boundary leaves the host).
- Book H1+H2+H4 as the gating work for any true-monokernel round loop.

---

## 8. Where mono plugs in (interface only)

- `ModelRunner.__init__` looks up `mono_decode_installers["DeepseekV41ForCausalLM"]` →
  `atom.models.deepseek_v41.mono.dispatch.install_mono_decode(model, atom_config, drafter, metadata_builder)`
  (`model_runner.py:152, 902`).
- `config_refusal` (`mono/config.py`) refuses mono when any of the following holds:
  - `ATOM_MONO_ENABLE=0`;
  - `common_refusals`: TP outside {2,4,8}, EP, DP/PP/CP>1, TBO, plugin mode, a graph mode other than FULL or eager;
  - KV other than bf16;
  - an index cache other than fp8/fp4;
  - keys per token > 1024;
  - an indexer shape other than (512, 2048, 8, 32, 128);
  - a drafter other than DSpark or none;
  - a verify of more than `MAX_ROWS=48` rows;
  - `ATOM_DSV41_SIDE_STREAMS≠0`;
  - routed experts other than A8W4 gate/up interleaved, checked at bind.
- `MonoDecodeModel.forward` routes a step when it:
  - is decode (not prefill, no TBO ubatch, no images);
  - has `0 < rows ≤ 48`;
  - has `step.width == rows`.
  Otherwise it uses the original `self.model(...)`. The routing happens before compiled code, and **inside the FULL
  capture** (graph width, not scheduled count).
- `MonoDraftModel.block_backbone` routes the DSpark draft backbone for ≤ `MAX_REQUESTS=8` requests at full block
  width. The LM head and Markov sampler stay original. It binds on the first call outside a recording (the graph
  warmup).
- `V41MonoDecodeRunner.forward` (`mono/runner.py:413`) per step:
  1. `mailboxes.begin_step`, embed, then `begin_forward`;
  2. **Engram join**;
  3. `write_step_meta`;
  4. per layer: Engram (original) → K1 (`_front`) → original compressor (FULL layers) → K2 (`_layer_post` = K2a+K2b,
     or split under `ATOM_MONO_CHECK`) → candidate table on the producing layer;
  5. collapse, then the optional token-parallel row gather.
- FP4 index: `install_mono_decode` adds a `WalkPlanner` step planner to the metadata builder. Score task plans are then
  built during host metadata prep, which adds to H2.
- Debug and timeline knobs: `ATOM_MONO_CHECK`, `ATOM_MONO_TRACE`, `ATOM_MONO_DEBUG` (bounded mailbox waits),
  `ATOM_MONO_TIMELINE` (`s_memrealtime` stamps per K1/K2), all with `--enforce-eager`.

---

## 9. Takeaways for the rocKE / true-mono plan

1. **Baseline to beat**:
   - native ATOM, level 3, FULL, TP4 (or TP2) **without EP**, BF16 KV, FP8 or FP4 index, DSpark5 fixed width;
   - two graph replays per step plus eager logits/sampler/draft-KV;
   - mono needs no-EP, so the "primary" TP4+EP deployment is not the mono comparison point. Compare against TP4
     no-EP and TP2 no-EP, the AgentX configurations.
2. **The step is a draft graph plus a target graph, with eager glue between them.** A true monokernel round must fold:
   - the target 40 layers;
   - the eager tail: norm, head, vocab gather, argmax/rejection, broadcast;
   - `compute_draft_kv`;
   - the draft 3 blocks plus the 5-step Markov sampler;
   - `_commit_cursors`;
   - the next step's `fill_deferred_decode_ids`.
   Otherwise the Qwen3.8 lesson applies: kernel savings turn into host-boundary idle.
3. **Host prep is the V4.1-specific danger.** Compress plans, spans, tentative Engram staging, Engram hashing and FP4
   walk plans are all Python on every step. Plan a device-side prepare-ahead from day one (V4.1 analogue of
   `ATOM_Q38_RB_DEVMETA`). Note that `prepare_decode` subtracts `num_rejected` on the host.
4. **The no-rollback state contract (ring slack 133/128, compressor ring slack, cursor commit by anchor) is friendly to
   a persistent kernel.** Accept/reject is visibility plus one cursor commit. No state snapshot or restore, unlike
   Qwen's GDN state.
5. **Collectives**:
   - ≈81 TP all-reduces per target pass, plus vocab gathers and broadcasts;
   - mono already fuses the `wo_b` and MoE all-reduces into K2a/K2b (env doc);
   - the remaining host-visible collectives are the embedding all-reduce, the head gather, the sampler broadcasts and
     the Markov exchanges.
6. Watch eager-fallback triggers (FP4 score workspace overflow, ladder gaps) in every served trace. They silently
   multiply host launches.

Key files:
- `atom/model_engine/model_runner.py` (`prepare_model`, `prepare_input_ids`, `run_model`, `postprocess`,
  `capture_cudagraph`, `_dspark_apply_ragged`);
- `atom/model_engine/engine_core.py`;
- `atom/config.py` (graph defaults, `capture_sizes`);
- `atom/model_engine/arg_utils.py`;
- `atom/models/deepseek_v41/{model,runtime,attention,moe,dspark,bounded_replay,config}.py`;
- `atom/model_ops/deepseek_v41/{mhc_pre_delayed,paged_scoring,compressor,router,rope_window}.py`;
- `atom/model_ops/v4_kernels/paged_2src.py`;
- `atom/model_ops/moe.py` (`Mxfp4MoEMethod`, all2all selection);
- `atom/models/deepseek_v4.py::MoE`;
- `atom/model_ops/attentions/deepseek_v41/{backend,speculative}.py`;
- `atom/spec_decode/{dspark_proposer,dspark_verify,draft_graph}.py`;
- `atom/model_ops/rejection_sampler.py`;
- `atom/plugin/vllm/{deepseek_v41_bridge,v41_core_overrides,v41_speculative}.py`;
- `atom/models/deepseek_v41/mono/{dispatch,config,runner}.py`;
- docs: `docs/deepseek_v41_{runtime,performance,dspark}.md`, `docs/environment_variables.md`;
- recipes: `recipes/DeepSeek-V4.1-Flash{,-Agentic}.md`, `recipes/atom_vllm/DeepSeek-V4.1-Flash.md`.
