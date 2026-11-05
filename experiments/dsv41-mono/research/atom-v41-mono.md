# ATOM DeepSeek-V4.1 mono decode: framework, kernels, refusal, history

Source of truth: `dev/ATOM` at `origin/main` (head `df86c482` when read, 2026-11-02), read-only via `git show`.
Qwen comparison: branch `qwen38-megakernels` at `352229d6` (`atom/models/qwen4_exp_mono`).
Ratios only. Absolute numbers quoted in commits/recipes are in the private numbers file.

---

## 0. Summary

- **Launch structure.** Each layer runs as **two persistent FlyDSL kernels**: K1 `attn_pre` and K2 `layer_post`. K2 is
  K2a `attn_post` followed by K2b `moe` in the same launch. This is not one persistent kernel per step. A target
  step runs 40 layers, so it has **80 mono launches**. On top of those come one FlyDSL `step_begin` launch (zeroes
  the mailboxes and fences the peers), one Triton `write_step_meta` launch, and the original-module launches that
  stay outside the kernels:
  - embedding, `begin_forward` and `collapse` / `end_forward`
  - the compressor (`_compress_batch`) on selecting layers
  - Engram layers
  - `candidate_block_table` on the candidate-producing layer
  - a custom all-gather on token-parallel steps

  The DSpark draft backbone runs on the same K1/K2: 3 stages, so 6 mono launches.
- **DSL.** All device kernels are **FlyDSL** (`flydsl.compiler` / `flydsl.expr`, with aiter's FlyDSL `buffer_ops`).
  `step_meta.py` is Triton, one launch. There is no HIP C++ or rocKE. The kernels run on the shared `atom/mono`
  device library: tagged mailbox, MX, ops and stamps.
- **Grid.** 256 CTAs x 512 threads (one CTA per CU, all co-resident, no cooperative launch). Stages are task tables
  placed around the CTAs (`first_task`). Every cross-CTA and cross-GPU hand-off is a **tagged (word, tag) mailbox
  pair**. There are **no grid-wide barriers** inside the layer kernels, only `gpu.barrier` within a workgroup. TP
  all-reduce and reduce-scatter happen **in-kernel**: each rank pushes bf16 partials into every peer's uncached HIP-IPC
  buffer at system scope, and every rank sums them in rank order 0..TP-1.
- **"On by default but currently refused": the doc is stale, the code is not.** The recipe
  (`recipes/DeepSeek-V4.1-Flash.md:29` and the "Fused decode (mono)" section) says K2b still runs MXFP4 activations on
  separated gate/up rows, so mono refuses every layer. On `origin/main`:
  - The only MoE-precision gate is `atom/models/deepseek_v41/mono/weights.py:bind_layer` (lines 103-112). It **requires**
    `quant_method.is_guinterleave and quant_method.fp8_activations` (A8W4), otherwise it raises
    `MonoUnsupported("layer N: routed experts not A8W4 (interleaved gate/up rows, MXFP8 activations)")`.
  - V4.1's MoE hard-sets `args.moe_fp8_activations = True` (`atom/models/deepseek_v41/moe.py:65`). That flows to
    `Mxfp4MoEMethod.fp8_activations` and `is_guinterleave = ATOM_MOE_GU_ITLV or fp8_activations`
    (`atom/model_ops/moe.py:1258-1259`).
  - K2b's docstring and code are MXFP4 x MXFP8 with interleaved gate/up, and they already were at `08735170`.

  So on a standard deployment (TP2/4, no EP, bf16 KV, fp8/fp4 index, FULL graph or eager) the gate **passes**. The
  refusal text was written mid-PR in squash `#2479`, before K2b was moved to A8W4, and never updated. Commit `7fc0926c`
  (the next day) reports GSM8K "with every decode step on mono" for V4.1. Details and caveats are in §5.
- **Versus the Qwen FlyDSL mono.**

  | | Qwen mono | V4.1 mono |
  |---|---|---|
  | TP | TP1 | TP2/4/8 with in-kernel peer comms |
  | Launches per layer | 1 | 2 (K1 and K2 are separate launches so the compressor and Engram modules can run between them) |
  | Mailbox code | vendored from FlyDSL `monokernel/glm` | shared `atom/mono/device/sync.Mailbox` |
  | Tag | `step*64 + layer + 1`, read from a device step counter, so it never zeroes | `layer + 1` plus `POST_TAG0` / `MOE_TAG0` offsets, which needs a per-step zero and peer fence (`step_begin`) |
  | Rows a step | up to 4 rows / grouped verifies | up to 48 rows, one build per width |

  Both use 256 x 512, agent-scope `sc1` polls, A8W4 MoE read in place and bitwise parity with the original path.

---

## 1. Where mono is wired in and switched on

| What | Where |
|---|---|
| Installer registry | `atom/model_engine/model_runner.py:152-158` `mono_decode_installers = {"DeepseekV41ForCausalLM": "atom.models.deepseek_v41.mono.dispatch.install_mono_decode", "KimiK3ForConditionalGeneration": ...}`. Comment: "Architectures not listed here route mono decode inside their own model class instead" (GLM-5.2, MiniMax-M3). |
| Install call | `model_runner.py:902-909`: after TBO wrapping and after `drafter.arm_aux_capture(self.model)`, `self.model = installer(self.model, config, drafter, self.attn_metadata_builder)`, then `_maybe_warmup()`. |
| Master switch | `atom/utils/envs.py:489` `ATOM_MONO_ENABLE` default `"1"`, so **on by default**. `0` disables every mono runner. |
| Debug switches | `ATOM_MONO_CHECK` (K2a and K2b launched apart and compared with the original modules; eager), `ATOM_MONO_TRACE`, `ATOM_MONO_TIMELINE` (per-phase `s_memrealtime` stamps, K1 and K2 saved separately), `ATOM_MONO_DEBUG` (bounded 10 s mailbox waits that record and raise instead of hanging). |
| Related | `ATOM_DSV41_SIDE_STREAMS` (default 0, `envs.py:536`) must stay 0. |
| Docs | `docs/environment_variables.md:144-163` (the most accurate description of the V4.1 conditions); `recipes/DeepSeek-V4.1-Flash.md` §"Fused decode (mono)" (stale in parts, see §5.3). |

The wrapper sits **outside** the compiled model (`MonoDecodeModel`), so the mono-vs-native decision is Python at both
level 0 and level 3. Commit `08735170`'s third sub-commit was "mono decode at any compile level".

---

## 2. Framework: contract, dispatch, plans, runners

### 2.1 Deployment contract: `mono/config.py:config_refusal(atom_config)`

`config_refusal` returns the first failing reason (`first_refusal`) or None. Checks, in order:

1. `envs.ATOM_MONO_ENABLE`, else "switched off".
2. `common_refusals(atom_config, Dims, SUPPORTED_TP)` from `atom/mono/runtime/deployment.py`:
   - `shard_refusal(Dims, tp)`: every width must split (64 query heads, 8 wo_a groups, 2304 routed and shared
     intermediate).
   - `tp in (2, 4, 8)`.
   - No expert parallelism, DP == 1, PP == 1, decode CP == 1, prefill CP == 1, no TBO and no TBO decode.
   - Not plugin mode (vLLM / SGLang).
   - Not a runner that takes rollout weight updates (`_takes_weight_updates`: the runner class has `update_weights`).
   - Compilation level is not `DYNAMO_AS_IS` (fullgraph breaks on `data_ptr`).
   - Not a piecewise cudagraph (`requires_piecewise_compilation`). V4.1 keeps the default
     `piecewise_cudagraph_required=True`.
3. V4.1-specific:
   - `kv_cache_dtype == "bf16"` (an fp4 KV pool is "packed"; the runner also raises `MonoUnsupported("packed KV
     cache")` at `runner.forward`).
   - `index_cache_dtype in ("fp8", "fp4")`.
   - `sliding_window + index_topk <= ATTENTION_KEYS_MAX = 64*16 = 1024`, so each split is exactly one 16-key tile.
   - `(index_topk, candidate_topk_blocks, candidate_block_size, index_n_heads, index_head_dim) == (512, 2048, 8, 32, 128)`.
   - The speculative method is None or `"dspark"`. MTP or Eagle hooks would not fire under mono.
   - A DSpark verify fits: `num_speculative_tokens + 1 <= MAX_ROWS (48)`.
   - `ATOM_DSV41_SIDE_STREAMS == 0`: a side-stream compressor would contend for the CUs the persistent kernels need.

Constants: `SUPPORTED_TP = (2, 4, 8)` (TP8 kernels compile and fit, but the native path does not start at TP8 because
AITER's FP8 blockscale GEMM has no K=288). `MAX_ROWS = 48` (wider, TP2's shared expert exceeds one unit per CTA:
72 tasks x 4 token tiles > 256, and K1/K2 spill). `MAX_REQUESTS = 8` (draft blocks per pass).

V4.1's own runtime validation (`deepseek_v41/config.py:validate_speculative_config`) forces DSpark with exactly
5 draft tokens and a bf16 KV. So a target verify is 6 rows per request, and 8 requests fill the 48-row maximum.

**Gaps relative to siblings.** There is no gfx950 arch check (GLM-5.2 added `_is_gfx950`) and no CU-count check.
`atom/mono/plan/execution.py` says "the runner refuses a GPU with another CU count", but only MiniMax-M3
(`runner.py:220`) and Kimi-K3 (`config.py:335`) read `multi_processor_count`. V4.1 does not.

### 2.2 Bind-time contract: `mono/weights.py`, `runner._bind`, `consensus`

- `bind_model` calls `bind_layer(block, config, tp)` for every block. `bind_layer`:
  1. **MoE precision gate** (lines 103-112): requires `is_guinterleave and fp8_activations`, otherwise
     `MonoUnsupported(... "not A8W4 ...")`.
  2. Every parameter is checked against `expected(config, mode, tp)` for shape, dtype and contiguity. Nothing is copied
     or relaid: kernels read parameters in place. The table:
     - **FP8 e4m3 + E8M0 per 32x32** for `wqkv_a`, `wq_b`, `wo_a`, `wo_b`, shared `gate_up_proj` and `w2`, and (on
       FULL/REINDEX layers) `indexer.wq_b`.
     - **MXFP4 `float4_e2m1fn_x2`** for routed `w13_weight [E, 2*inter_pad, H/2]` and `w2_weight [E, H, inter_pad/2]`,
       with intermediates padded to 128 and scales in aiter `shuffle_scale` order.
     - bf16 for norms, the gate and `indexer.weights_proj`; f32 for `attn_sink`, `e_score_correction_bias` and the
       mHC `hc_{attn,ffn}_{fn,base,scale}`.
- `runner._bind` also requires bf16 RoPE tables of width `k1.HALF`, and decides token parallel
  (`_custom_gather_fits`). Token parallel is turned off, not refused, when aiter's custom all-gather is unavailable,
  because NCCL's cannot be captured.
- `consensus.bind_agreed(bind, cpu_group)`: every rank binds, then a gloo MIN all-reduce (`tp_agree`). One rank's
  refusal becomes everyone's before anything collective happens. A non-`MonoUnsupported` exception is converted into a
  refusal.
- `LazyRunner` (`atom/mono/runtime/route.py`) creates the runner on the first eligible step **outside graph capture**,
  which is a warmup. A `create()` refusal disables mono for good and logs `"V4.1 mono decode off: <why>"` at WARNING.
  A `prepare(width)` refusal only affects that width (`WidthBuilds._refused`).
- `WidthBuilds.prepare(width)` (`atom/mono/runtime/widths.py`) builds per width: it compiles K1 for each distinct
  `AttnPreBuild` plus one K2. A width whose kernel needs **scratch** (spills) is refused, because a persistent kernel
  with scratch can be partly resident and deadlock. The result is TP-agreed.
- Install-time refusal path: `dispatch.install_mono_decode` logs `logger.info("V4.1 mono decode off: %s", why)` and
  returns the unwrapped model.

### 2.3 Hand-off contract: `mono/contract.py`

`contract.py` is a static check, not a runtime gate. Each kernel declares its mailbox regions (`RegionDecl(name, space,
stage)`), and a compile-only trace (`atom.mono.plan.trace`, with `FLYDSL_RUNTIME_ENABLE_CACHE=0`) records every
put/poll, which `atom.mono.plan.check` holds against the table.

| Kernel | Regions |
|---|---|
| K1 | lin, pmix (slice); x8, x8s (norm); qkv (wqkv_a); qx8/qx8s (qkv), or in token parallel the peer qg/qgs/kg; nb, iq, iw or nbg/iwg (indexer layers) |
| K2a | the indexer's isf, irdy, i{sel,blk}{level}; am/al/ap (attn_score); xo/xos (irq); x8b/x8bs (wo_a); **attn (PEER, wo_b)**; lin, pmix (slice) |
| K2b | x8rdy (xq), logit (router), route, smid (shared, *exchange*), smq, smc, ugf (ug), **ffn (PEER, down)** |
| K2 merged | adds xrdy (K2a norm or K2b gather) and, in token parallel, **xg (PEER)** |

The three kernels of a layer share one scratch buffer. Same-named regions are separated by **tag ranges**: K1 uses
`layer + 1`, K2a uses `POST_TAG0 = 64`, K2b uses `MOE_TAG0 = 128` (`runner.py:68-70`). The `__main__` checks S in
{6, 12, 24, 48} across feeds_index/fp4/tok_par variants. Tests: `tests/test_deepseek_v41_mono_contract.py`
(target S=6 TP4 and draft S=10 E=128 top-3, plus negative controls).

### 2.4 Dispatch: `mono/dispatch.py`

- `step_supported(input_ids, inputs_embeds)` routes a step to mono only when all of these hold:
  - no `inputs_embeds`, and `0 < rows <= 48`
  - `fwd.context` is set, it is not a prefill, and there are no TBO ubatch slices
  - `attn_metadata.step` exists and there is no `image_mask`
  - `step.decode and step.width == rows`

  Routing is by **row count only** (a padded graph width included), however the rows split into requests.
  Inputs must be TP-uniform because the kernels spin on peers; the docstring warns that a per-rank flag such as
  `is_dummy_run` must never gate this.
- `MonoDecodeModel.forward`: `if step_supported(...) and self._mono.ready(rows): return runner.forward(...)`, otherwise
  the original `self.model(...)`. It wraps the runtime model the same way TBO's `UBatchWrapper` does (`__getattr__`
  passthrough).
- `MonoDraftModel.block_backbone`: routes the DSpark draft backbone when `0 < requests <= 8` and the width is the full
  `block_size`. It binds on the first call outside `DraftGraph` recording.
- `install_mono_decode`: runs `config_refusal`, then wraps the drafter if it has `block_backbone`. With an fp4 index it
  registers `WalkPlanner` as a step planner (`metadata_builder.add_step_planner`), and returns `MonoDecodeModel`.
- Tests: `tests/test_deepseek_v41_mono_dispatch.py` covers routing (verify, plain decode, ragged, width-only, draft by
  request count, embeddings and images to the original path), every config refusal reason, the weight table at TP
  2/4/8, and **`test_routed_experts_off_a8w4_are_refused`**, parametrized over (interleaved, fp8_act) in
  {(F,F), (T,F), (F,T)}, which expects `MonoUnsupported` matching "not A8W4". The `_Block` fixture's default is
  A8W4, the deployed state.

### 2.5 Plans: per-step mailbox cell decisions, "defined once" (DESIGN_v3 4.9, I3)

Each plan is plain Python that runs both on traced `Int32` inside the kernel and on Python ints in a CPU property test.
It uses only non-negative `//` and `%`, so floor and truncation agree.

| Plan | Decides |
|---|---|
| `attention_plan.py` | 64 splits x 16 keys (`SPLITS`, `BK`), 8 score waves. `kv_len = n_sel + window_count`; `written_splits = max(cdiv(n, 16), 1)` (a pad row writes split 0); `polled_split(i, last) = min(i, last)`, so PV polls the last written split past the end. |
| `index_plan.py` | CHUNK=128 columns per score task; SEG=8192 values per select task (LDS-resident); TOPK=512; TOPK_BLOCKS=2048; BLOCK_ROWS=8; FP4 walk STEP=64 keys and WALK_ROWS=6 query rows per key load. A token is `scored` only if `bound > 512`. Selection is a **tree of top-k levels** per kind (`sel`: rows, `blk`: candidate blocks). A token's list is final at `final_level(bound)`, which publishes IRDY. Also `fill_walk_plan` (host layout of FP4 score tasks) and `reindex_split`. |
| `moe_plan.py` | The union U of experts with picks; pick tiles of 16 (MFMA N); `ug_parts = cdiv(groups, ug_groups)` (parts that are all padding are skipped); `ug_flag(unit, part)`; `polled_tile`. |
| `walk_plan.py` | `WalkPlanner`, a host step planner: a `[BLOCKS, 4]` int32 plan per FULL compression ratio (`v41_mono_walk_plan_<ratio>`) in a pinned `CpuGpuBuffer`, published with the step metadata (only steps of 48 rows or fewer). |
| `step_meta.py` | **Triton**, one launch: `ring_rel[S]` (each token's window ring row, or -1 for a pad) and `key_meta[S, 4]` (first window position, count, block-table offset, slot row base). It replaced about 19 torch elementwise kernels. `write_draft_step_meta` is the draft-block variant. |

### 2.6 Target runner: `mono/runner.py:V41MonoDecodeRunner`

- **Builds per width.** `_layer_keys(s)` makes one `AttnPreBuild` per layer, keyed by tokens, `fold` (layer != 0 and not
  Engram), `feeds_index` (FULL/REINDEX), `index_fp4`, `aux` (DSpark tap layers), `tp`, timeline, `diag_off` and
  `tok_par`. Distinct keys are compiled once. `_post_key(s)` is one `LayerPostBuild` shared by all layers: tokens,
  `index_bound_max` (max_model_len rounded up to 128), `index_fp4`, experts/topk, tp, `ug_groups` (1 below 7 rows,
  else 2) and `tok_par`.
- **Token parallel** applies when S >= 12, S is a multiple of TP, and it is neither check mode nor split K2. The
  attention and FFN all-reduces become reduce-scatters to each token's owner (rank r owns rows r*S/TP onward), the
  per-token stages run on owned rows, and quantized rows are gathered through the peer buffer. The final hidden state
  and the DSpark taps are all-gathered once with aiter's custom all-gather (`_gather_rows`). Results are bit-identical
  because the rank order is the same.
- **forward(input_ids, positions)** for an S-row step:
  1. `_use_rows(S)` narrows every row buffer.
  2. `mailboxes.begin_step()`: the FlyDSL `step_begin` launch.
  3. `model.embed` and `model.begin_forward`; join the Engram side-stream all-gather (it spins on CUs and would
     deadlock next to a kernel that needs every CU).
  4. `write_step_meta` (Triton).
  5. Per layer: Engram if present (original module) -> **K1** (`_front`) -> on selecting layers
     `block.attn._compress_batch(normed, ...)` (original compressor) -> **K2** (`_layer_post`; `_attn_post` + `_moe`
     when split) -> `candidate_block_table(...)` if `produces_candidates`. The state becomes
     `SinglePassHCState(res_ffn, pre, ffn_out, post, comb)`.
  6. `state.collapse()`, `_gather_rows` if token parallel, `end_forward`, and a debug wait-report.
- Every launch passes `stream=torch.cuda.current_stream()`, because FlyDSL's default stream is NULL, which is outside
  graph capture.
- **Buffers.** One scratch buffer sized to the max of K1, K2a, K2b and K2 (plus DIAG in debug). Peer buffer:
  `peer_bytes(max_rows, tp)`. Indexer logits are `[S, index_bound_max]` f32 plus block maxima. Per selecting layer
  there are `isel` and `irow` `[S, 512]` buffers.

### 2.7 Draft runner: `mono/draft_runner.py:DraftMonoRunner`

- Subclasses the target runner. Rows are requests x `block_size` (`max_rows = width * 8`). It uses the draft's routing
  (`dspark_n_routed_experts`, `dspark_num_experts_per_tok`; the contract test uses 128/3) with `ug_groups=1`, has no
  indexer (`index_bound_max=0`), never uses token parallel, and has no check, timeline or split mode.
- The draft has 3 window-only stages (`draft.mtp`). K1 writes the block's keys into the stage's ring at anchor+1
  onward; the ring holds window + width slots. Every row of a block sees one span (its request's window up to the
  anchor plus the whole block).
- The key order is the paged decode's (window, then block), not the Triton `sparse_attn` order, so the draft is
  **not bit-identical** to the original draft. Acceptance is reported unchanged.
- The rest of the draft model is unchanged: embedding, noise embedding, `build_block_state`, LM head and Markov sampler.

---

## 3. Kernels

Common to all:
- Built with `@flyc.kernel(name=..., known_block_size=[512,1,1])` and launched as
  `.launch(grid=(256,), block=(512,), stream=...)`.
- The build key is `key_tuple(...)` plus `SOURCES = source_digest("mono", "models/deepseek_v41/mono",
  "model_ops/deepseek_v41/router.py")`, so any source change rekeys the JIT cache. The launcher is named after the
  kernel symbol (e.g. `launch_v41_attn_pre_s48_f1_ix0_i41_a0_tl0`, commit `fbb89df0`).
- Arguments are int64 device addresses packed by `KernelAbi` (`atom/mono/runtime/abi.py`).
- Stages are task tables placed from a base CTA (`first_task(bid, base)`, then every 256th). Each CTA walks its tasks
  stage by stage, with `gpu.barrier()` between stages inside the CTA. Cross-CTA ordering is entirely mailbox dataflow,
  so a consumer only waits for the producer cells it needs.
- LDS: 160 KiB per CTA (`LDS_BYTES`). K2's LDS is a union of K2a's and K2b's.

### 3.1 K1 `attn_pre.py` (about 57 KB): attention seam to rotated query

One launch per layer and rank. Stages, with TP4 task counts:

| Stage | Tasks | Work |
|---|---|---|
| slice | 160 | 32 hidden columns x 4 HC streams per task: fold the owed post into the residual (bf16 `res_out`), the collapsed layer input -> LIN, partial mix projection and sum of squares -> PMIX |
| norm | S | `attn_norm` of LIN -> bf16 -> `quantize_fp8` -> X8 |
| wqkv_a | 112, 16 rows each | FP8 GEMV -> q latent \| kv row -> QKV |
| gate | S | the 160 partials -> post/comb (Sinkhorn, 20 iterations) and the next `pre`. Placed after the stages it might otherwise block |
| qkv | S | `q_norm` -> MXFP8 (QX8); `kv_norm` -> RoPE -> FP8 QAT -> window ring row (in the KV pool) |
| wq_b | heads x 512 / 16 (512 at TP4) | FP8 GEMV -> bf16 -> RoPE on each head's tail -> `q` |
| indexer layers (via `index_query.py`) | iw: 32; iq: 256; iquant: S x 32 | see §3.5 |

- Fuses aiter `mhc_fused_post_pre_delayed_rmsnorm`, ATOM `quantize_fp8`, the q/kv projections and norms, RoPE, the
  window-ring KV write and the indexer query, following the original rounding exactly. Only the split-K reductions'
  split and order differ.
- Token-parallel variant: slice through qkv run on owned tokens. Each token's q latent is pushed to every rank's QG/QGS
  and its window row to every other rank's KG; the last stage writes KG into the ring. Indexer layers push iw to IWG
  and normed rows to NBG (for the compressor).
- Levers in the code:
  - weight prefetch ahead of mailbox waits (`slice_issued`, `batched_rounds`)
  - `SLICE_FN_EARLY_MAX = 24` (up to 24 rows a slice holds its mix weights through the wait; wider spills)
  - non-temporal loads (`CM_NT`) on weights read once per step
  - DPP butterflies for the Sinkhorn sums

### 3.2 K2a `attn_post.py` (about 30 KB): indexer, then attention, to the MoE input

| Stage | Tasks | Work |
|---|---|---|
| (selecting layers) indexer | | `index_score.py` / `index_score_fp4.py` plus `topk.py`, §3.5 |
| score | S x 8 per head tile | `attention.py` scores and softmax statistics -> AM, AL, AP |
| irq | S x 16 per head tile | PV and split combine, then inverse RoPE of each head's tail and group-32 FP8 (aiter `inverse_rope_group_quant` semantics) -> XO |
| wo_a | 64 | grouped FP8 GEMV (8 heads a group) -> bf16 -> `quantize_fp8` -> X8B, X8BS |
| wo_b | 320, 16 rows each | FP8 GEMV -> the rank's bf16 partial, **pushed to every rank's ATTN peer region** (system scope). In token parallel it goes only to the token's owner (reduce-scatter) |
| slice | 160 | sum the TP partials of 32 columns in rank order (fp32 -> bf16) = the all-reduce, fold into the residual, then the FFN seam as in K1 |
| norm, gate | | `ffn_norm` -> `normed` (the MoE input); FFN gates |

### 3.3 `attention.py`: K2a's decode sparse attention, bit-matched

- Reproduces the original Triton `_paged_decode_split_kernel` / `_paged_decode_reduce_kernel` (16-head tile, D=512,
  bf16 pool, 64 splits, 16-key tiles, D_CHUNK 128) **bit for bit**. Both orders were read off the ISA and LLVM IR and
  are pinned by harness emulators.
- The original picks splits from the step's rows (64 up to T=8, 32 at T=12). Mono keeps the order of a 6-row call
  (`ORDER_ROWS`) at every S and TP, so a row's result never depends on the other rows of its step. Check mode compares
  against a reference run 6 rows at a time.
- score: 16 chained 16x16x32 bf16 MFMAs over D, `exp2` softmax, row sums in a fixed lane order. pv: 16x16x16 bf16
  MFMAs, a DPP tree combine, the sink fold, bf16 into LDS `[16][32]`, which the same task's inverse-RoPE quant reads.
- K2 score loads key rows with `buffer_load_to_lds` (`5e8d9131`). A ragged head tile (TP8 has 8 heads per rank) reads
  the last live head and writes nothing for the dead ones.

### 3.4 K2b `moe.py` (about 67 KB): FFN to the reduced output, precision A8W4

Routing is the target's 384 experts / top-6 or the draft's 128 / top-3. The MoE is TP-sharded: the rank holds 1/TP of
every expert's intermediate and 1/TP of the shared expert.

| Stage | Tasks | Work |
|---|---|---|
| xq | S | row -> **MXFP8** for the routed experts (aiter `fused_mx_quant_moe_sort` semantics, amax seeded 1e-10) and FP8 for the shared expert (`quantize_fp8`, amax floored 1e-4) -> X8R, X8 plus per-token X8RDY |
| router | E/16 x 4 | bf16 GEMV over 16 rows x a quarter of K -> LOGIT. The device helpers are shared with `atom/model_ops/deepseek_v41/router.py` (the native decode uses the same fixed-order router GEMV, so both route identically bit for bit) |
| route | S | ordered partial sums -> bf16 logits -> aiter `topk_gating_kernel_opt` semantics (sqrtsoftplus + bias, top-k, renormalized x 1.5), one wave per token -> ROUTE (an LDS route table with token bitmaps, union U, slots and sorting networks) |
| shared | 36 (TP4) | FP8 GEMV -> bf16 -> silu_and_mul (clamped at 10) -> SMID. Odd and even tasks exchange through SMID |
| ug | \|U\| x 9 | 2 x (32 gate + 32 up) rows per task, **w13 gate/up interleaved per 16-row tile**. **MXFP4 x MXFP8 scaled-MFMA GEMV**; each B operand is read from LDS once for both gate and up. Epilogue: clamp, silu x up, bf16, stage-2 MXFP8 quant (ceil code) -> MID, then UGF flags |
| down | 320, 16 rows; two per CTA run together | per expert of U, MXFP4 x MXFP8 GEMV x routing weight -> bf16, summed in top-k order with bf16 rounding at every add. The shared expert's FP8 down is added in. The result is **pushed to every rank's FFN peer region** (in token parallel, only the owner's) |
| reduce | 160, 32 columns | TP partials in rank order -> bf16 `out` |

- K2b computes only experts in the union U of the step's picks, not all 384. The draft uses 1x(32+32) ug tasks with
  K split into quarters.
- `moe_shape.py` holds `MoeBuild` (tokens, experts, topk, tp, timeline, diag_off, `ug_groups`), `RouteShape` (task
  counts and placements), `SORT_NETS` and the LDS route-table readers (`MASK_WORDS = 2`, so at most 64 tokens).
  `ug_groups` = 1 versus 2 trades task count against tail balance: 1 is better at draft widths, 2 at 48-row target
  widths.

### 3.5 Indexer kernels: K1 projections, K2a scoring and selection

- **`index_query.py`** (in K1). The indexer is replicated: every rank computes all 32 heads. Stages:
  - **iw** (32 tasks): `weights_proj` GEMV emulating aiter `wv_splitk_small` lane and DPP order exactly.
  - **iq** (256 tasks): FP8 GEMV of the q latent -> bf16 -> GPT-J RoPE on dims 64-127.
  - **iquant** (S x 32): per-(token, head) e4m3 `dynamic_per_token_scaled_quant` semantics and the head-weight scale.
    On the FP4 plane, `stage_iquant_fp4` uses group-32 E8M0/E2M1 in the row-group scorer's packed scale order.
- **`index_score.py`** (in K2a, FULL and REINDEX layers). Reproduces the native `score_topk_paged` (gluon
  `deepgemm_fp8_paged_mqa_logits` + `top_k_per_row_decode(stable=True)` + `pick_candidate_blocks`) at **any context
  length**.
  - `iscore` tasks are runs of a request's tokens x up to `TASK_CHUNKS` chunks of 128 columns. All key loads are issued
    first, then 4 chained 16x16x32 fp8 MFMAs with relu x k-scale and a head sum, written to global `ilog`. A candidate
    producer also writes 8-row block maxima to `ibmax`. Then come the ISF flags.
  - `ilist` levels run `topk.lds_topk` over segments (level 0) or merge child lists. The final list goes to `isel`
    (lifted to pool rows on REINDEX) and `icout`, and IRDY is published for the attention.
  - Logits are too large for tagged pairs, so they go through **global memory at device scope (SC1, past the per-XCD
    L2)**: store, `s_waitcnt vmcnt(0)`, barrier, flag.
- **`index_score_fp4.py`** (`--index_cache_dtype fp4`). Reproduces aiter `pa_mqa_logits_fp4_rowgroup` (page 8): E2M1
  keys and queries with E8M0 per 32 dims.
  - A task is a "piece" (up to 6 adjacent tokens on one page table) x a run of chunks. FULL layers take it from the
    host walk plan (`iplan`, one descriptor per CTA); REINDEX layers derive it from the CTA index.
  - Waves walk 64-key steps with DEPTH steps in flight. Each key load is scored for every row of the piece (one
    16x16x128 scaled MFMA per row, head tile and key tile). Two `permlane32` swaps and a `permlane16` swap give a single
    dword store per row per step. Logits are bit-identical to the reference, and the output lands in the same `ilog`
    / `ibmax` cells as the FP8 path, so the selection tree is shared.
- **`topk.py`**: `lds_topk` is a CTA's exact top-k of up to 32768 fp32 keys (64 per thread). It ranks by the u32
  order key (-0 < +0, NaN > +inf), breaks ties toward the lowest index and emits in ascending index order with -1
  padding. It uses four 8-bit radix passes with integer counts, so the result is independent of arrival order. Commit
  `08735170` cut its fixed costs: quad loads, pass-0 run merging, no passes when n <= k, keys in registers.

### 3.6 Small kernel modules

| Module | Role |
|---|---|
| `layer_post.py` | **K2 = K2a + K2b in one launch.** Every CTA runs K2a's stages, then K2b's. K2a's norm publishes each token's row with device-scope stores, a vmcnt drain, then a per-token XRDY flag. xq and the router wait on XRDY and read at **device scope**, because "a plain load could hit this XCD's L2 copy of the previous layer's" buffer. In token parallel the norm also pushes rows to peers' XG and a K2b `gather` stage pulls them in. Each half keeps its own tag and its own scratch regions, with K2b's placed after K2a's (`moe_base`). |
| `token_par.py` | `own_count(s, tp) = s/tp`, `own_view(c, names)`: shifts per-token argument pointers by `t0 * TOKEN_ROW_BYTES[name]` so the per-token stages run unchanged on owned rows. |
| `peer_layout.py` | The single peer-buffer layout (8 B per (word, tag) pair): `attn` and `ffn` `[tp][tokens][H/2]` partials; `xg`, `qg`, `qgs`, `kg`, `nbg`, `iwg` token-parallel gathers. Shared by the kernels, the runner allocation and the debug report. |
| `mx.py` | V4.1-specific FP8 code rules. `code_round` (aiter Triton `_mxfp8_quant_op`: `amax + 0x200000`, exponent minus 8), versus the shared `code_ceil` (ATOM `quantize_fp8` / aiter `rmsnorm_quant` MXFP8). Each stage must use the rule its original kernel used. |
| `dims.py` | `Dims(tp)`: every per-rank width (heads 64/tp, wo_a groups 8/tp, inter 2304/tp padded to 128, shared 2304/tp, task counts). It refuses TP sizes that do not shard. |
| `debug.py` | `mailbox(...)` factory: a bounded-wait `Mailbox` with region ids (CRC-32 of the names) in debug builds. |

### 3.7 Shared device and runtime library: `atom/mono` (from `08735170`)

- **`device/sync.py`**:
  - `Mailbox.put` / `put_bf` / `put_words` store `(payload, tag)` pairs with one 8- or 16-byte buffer store. The tag is
    `layer + 1`.
  - `poll(specs)` batch-reloads up to `POLL_MAX = 12` pairs until every tag matches, with `s_sleep(1)` in the spin.
  - The scope comes from the region's space: SCRATCH uses `CM_DEV` (sc1, agent scope, visible on every XCD); PEER uses
    `CM_SYS` (sc0|sc1, system scope over XGMI).
  - `publish()`: `s_waitcnt vmcnt(0)`, CTA barrier, then one thread puts the flag, because LLVM's AMDGPU memory model
    omits vmcnt for a workgroup-scope release.
  - Debug builds use `_spin_bounded` (10 s at 100 MHz ticks) and write a diag record.
- **`device/ops.py`** (wave and math primitives: rsrc, xshfl, xred, butterfly, fp8_pack4, hw_exp2/rcp/rsq,
  `batched_rounds`, `rows_to_lds`, `kernel_symbol`, `launcher`, `traced`), **`device/mx.py`** (FP4/FP8 MX:
  `mfma_scaled`, `code_ceil`, `fp4_tile_load`, `scale_index`), **`device/ranks.py`** (`peer_bases`, and `sum_partials`
  in rank order), **`device/stamps.py`** (`s_memrealtime` timeline stamps), **`device/select.py`** (MiniMax-M3 exact
  top-k by counting 64-bit keys).
- **`runtime/step_begin.py`** (commit `7fc0926c`): one FlyDSL launch of 256 x 512.
  - Every CTA zeroes its share of the peer data at system scope and flags its epoch.
  - CTA 0, wave 0, waits for the flags, writes the epoch into every rank's fence slot (8 int64 slots at each peer
    buffer's tail, never zeroed) and polls its own slots. The other CTAs zero the scratch.
  - This replaced `_foreach_zero_` plus a Triton fence. Triton 3.7 on gfx950 compiled that fence's poll with its
    reloading lane masked off, which hung CI on the DSpark draft's first mono step.
  - A TP collective cannot serve as the fence, because aiter's custom all-reduce skips communication in a capture
    warmup.
- **`runtime/peer_memory.PeerBuffer`**: one **uncached** buffer per rank from aiter's `UncachedIpcHeap`, with HIP IPC
  handles exchanged over the CPU group and a table of every peer's address (`kernel_args()` gives the i64 address
  table). `SymmetricPeerBuffer` drives HIP directly through `hip_ipc.py` (ctypes: `hipIpcGetMemHandle`/`Open`,
  `hipDeviceMallocUncached` = 0x3). V4.1 uses `PeerBuffer` through `lifecycle.owned_peer_buffer`, freed by a weakref
  finalizer.
- **`runtime/mailboxes.StepMailboxes`** (owns the epoch and missing-rank state), **`widths.WidthBuilds`**,
  **`route.LazyRunner`**, **`consensus`** (`tp_agree`, `bind_agreed`, `MonoUnsupported`), **`compile`** (compile-only
  builds, `kernel_resources` from code-object metadata: VGPR, SGPR, spills, scratch, LDS), **`deployment`**
  (`common_refusals`), **`aux_taps`** (DSpark hidden-state taps), **`timeline`**, **`debug`**, **`compare`**.
- **`plan/`**: `execution.py` (BLOCKS = 256, THREADS = 512, LDS 160 KiB, `first_task`), `build_key.py`,
  `shard.py` (`Shard`, `Tiles`, `padded`, `shard_refusal`), `layout.pair_layout`, `trace.py` and `check.py` (the
  hand-off contract), `arith.sel`, `tiles.py`.
- **`weights/`** (config, formats, layer, packing): ATOM/AITER MXFP4 and MXFP8 storage views and packers. They are
  used by GLM-5.2, Kimi-K3 and MiniMax-M3. V4.1 does not use them; it reads parameters in place.

### 3.8 XCD awareness

- The design is **cache-scope aware only**. Polls and hand-offs use sc1 (agent scope) so they bypass the per-XCD L2.
  Large indexer logits use device-scope stores and loads. Layer-shared buffers (`normed`) are read at device scope to
  avoid stale per-XCD L2 lines.
- There is **no XCD-aware task placement**, **no NUMA-homed or dual-copy mailboxes** and **no contention "holds"**: a
  grep of `deepseek_v41/mono/kernels` finds none. The only scheduling levers are prefetch-before-wait,
  `SLICE_FN_EARLY_MAX` and the K2b `ug_groups` granularity. These are open levers (the Qwen learnings measured
  NUMA/holds as material).

---

## 4. Launch count per decode step

| Item | Launches | DSL |
|---|---|---|
| `step_begin` | 1 | FlyDSL |
| `write_step_meta` / `write_draft_step_meta` | 1 | Triton |
| K1 x 40 layers | 40 | FlyDSL |
| K2 x 40 layers (split into K2a + K2b under `ATOM_MONO_CHECK`) | 40 (80 split) | FlyDSL |
| Compressor (`_compress_batch`) on FULL/REINDEX layers | native, several | original modules |
| Engram layers | native | original |
| `candidate_block_table` on the candidate producer | native | original |
| embed, begin/end_forward, HC collapse | native | original |
| token-parallel `_gather_rows` | 1 custom all-gather | aiter |
| DSpark draft backbone (3 stages) | 2 x 3 + step_begin + draft meta | FlyDSL + Triton |

The whole thing is captured into the FULL decode CUDA graph; `LazyRunner` builds outside capture, during warmup. Per
the design notes there is no in-kernel layer-to-layer chaining: each layer's K1 waits on the previous K2 through
stream order.

---

## 5. "On by default but currently refused": precise status

### 5.1 Every refusal site

| Site | Condition | Message |
|---|---|---|
| `mono/config.py:config_refusal` -> `dispatch.install_mono_decode` | see §2.1 | INFO `"V4.1 mono decode off: <why>"`; the model is left unwrapped |
| `mono/weights.py:bind_layer` lines 103-112 | `not (method.is_guinterleave and method.fp8_activations)` | `MonoUnsupported("layer N: routed experts not A8W4 (interleaved gate/up rows, MXFP8 activations)")` -> `bind_agreed` -> `LazyRunner` WARNING `"V4.1 mono decode off: ..."`, mono off for good |
| `weights.py:bind_layer` lines 114-124 | missing, wrong shape or dtype, or non-contiguous tensor | `MonoUnsupported("layer N: <name> is ..., expected ...")` |
| `runner._bind` line 186 / `draft_runner` line 64 | RoPE table not bf16 or width != 32 | `MonoUnsupported("rope table ...")` |
| `runner.forward` line 418 / `draft_runner.backbone` line 104 | `cache.packed` (fp4 KV) | `MonoUnsupported("packed KV cache")`. Unreachable after `config_refusal`, which requires bf16 KV |
| `runner._front` line 523 | owed post does not match the build's `fold` | `MonoUnsupported("owed post is not as built")` |
| `WidthBuilds.prepare` | build exception, or a kernel needing scratch (spills) | WARNING, that width declined |

### 5.2 The MoE activation question, resolved from code

- **What mono requires:** A8W4. Routed experts must take **MXFP8 activations** on **interleaved** gate/up rows. K2b's
  `xq` quantizes to MXFP8, and `ug`/`down` are MXFP4 (weights) x MXFP8 (activations) scaled MFMAs.
- **What main runs natively:** `atom/models/deepseek_v41/moe.py:65` sets `args.moe_fp8_activations = True`
  unconditionally. The docstring says MXFP4 activations, aiter's gfx950 default, put the routed output about 12% off the
  reference and cost about 1.4 pp of GSM8K. This reaches `FusedMoE(mxfp4_fp8_activations=...)`
  (`deepseek_v4.py:3796`), `FusedMoEConfig.mxfp4_fp8_activations` and finally
  `Mxfp4MoEMethod.fp8_activations = True; is_guinterleave = ATOM_MOE_GU_ITLV or True` (`model_ops/moe.py:1258-1259`).
  `apply` then passes `quant_dtype_a = fp8` and `gate_mode = INTERLEAVE` (`moe.py:2046-2058`).
- **Conclusion:** on `origin/main` the gate passes for the shipped V4.1 MoE. The text "currently refused: its MoE
  kernel still runs MXFP4 activations" (`recipes/DeepSeek-V4.1-Flash.md:29`) and the paragraph "K2b still reads
  separated rows with MXFP4 activations, so mono refuses every layer until K2b follows" are **stale**:
  - `git log -S "K2b still reads separated rows"` shows both lines were introduced by squash `08735170` (#2479). That
    same squash already contains the A8W4 K2b (`git show 08735170:.../kernels/moe.py` header: MXFP8 xq, interleaved
    w13) and the A8W4 gate in `weights.py`. The doc was written at an intermediate PR state (the native MoE switched to
    A8W4 before K2b did) and never revised.
  - Corroboration: `7fc0926c` (one day later) reports "GSM8K with every decode step on mono: V4.1 ...".
    `docs/environment_variables.md:159` correctly lists "routed experts in A8W4 (MXFP4 weights, MXFP8 activations,
    gate / up interleaved: checked when the weights bind)" as a requirement, not a blocker.
  - The recipe is stale elsewhere too: it says "verify steps of up to 12 rows" and "draft backbone of up to two
    requests", while the code allows 48 rows (`5e8d9131`: "Mono routes any decode step up to 48 rows") and 8 draft
    requests.
  - The recipe's measurement table is dated 2026-09-29, before the A8W4 K2b. It was measured with the older MXFP4-act
    K2b against a native path that was presumably MXFP4-act too.
- **When it would refuse on MoE grounds:** only if `fp8_activations` were False or the rows separated. That cannot
  happen through the V4.1 `MoE` class on main (no env toggles `moe_fp8_activations`). A quant method without these
  attributes also refuses, through `getattr(..., False)`; an example would be a non-MXFP4 (FP8) expert checkpoint
  reaching `Fp8MoEMethod`. `moe_backend="mega"` (`MegaMxfp4MoEMethod`, which empties `w13_weight`) would fail the shape
  check, but mega requires EP, which mono already refuses.
- **Contrast:** Kimi-K3 mono requires the opposite (`kimi_k3/mono/weights.py:269-275`: refuse `is_guinterleave`,
  refuse `fp8_activations`, "the kernels feed the experts bf16"). GLM-5.2 requires `is_guinterleave` present and False
  (`glm52/mono/weights.py:231-238`).

### 5.3 What would make mono refuse in practice, and how to unblock

- **Most likely real-world blockers** (config, not MoE):
  - a piecewise cudagraph mode (use FULL or eager; V4.1's runtime validation already forbids non-FULL at level 3)
  - `--kv_cache_dtype fp4` (packed): use bf16
  - EP launches (`--enable-expert-parallel`): mono needs TP-sharded experts
  - `ATOM_DSV41_SIDE_STREAMS != 0`
  - plugin mode (vLLM/SGLang)
  - an RLHF runner (rollout weight updates)
  - `ATOM_USE_CUSTOM_ALL_GATHER` off: token parallel turns off, but mono still runs
- **If a deployment did run MXFP4 activations**, the unblock is either (a) set `moe_fp8_activations` (already the V4.1
  default), or (b) add an MXFP4-activation K2b variant: xq to MXFP4 with aiter's FP4 quant semantics, `ug`/`down` as
  FP4 x FP4 scaled MFMA, and the stage-2 quant to MXFP4. Then relax the gate in `weights.py:103-112` to admit
  `(is_guinterleave, not fp8_activations)` and select it by build key. The separated-rows layout would also need
  scale-reader changes, as the Kimi gate comment notes.
- **Tests encoding the refusal:** `tests/test_deepseek_v41_mono_dispatch.py::test_routed_experts_off_a8w4_are_refused`
  (the three non-A8W4 combinations), `test_a_weight_off_the_table_is_refused`, `test_unsupported_deployment_is_refused`
  (each `config_refusal` reason), `test_switches_refuse` ("switched off", "ATOM_DSV41_SIDE_STREAMS"),
  `test_plugin_mode_is_refused` and `test_refused_deployment_is_not_wrapped`. Framework tests:
  `tests/test_mono_runtime.py` (LazyRunner width refusal versus create refusal), `test_mono_plan.py`,
  `test_mono_shard.py`.
- **To verify on hardware** (we did not run anything): launch V4.1-Flash TP4 with DSpark at level 0 or 3 FULL and grep
  the log for `"V4.1 mono decode on"` / `"kernels built for width"` versus `"V4.1 mono decode off: ..."`.

---

## 6. Commit history

`git log --oneline origin/main -- atom/models/deepseek_v41/mono` has 6 commits, all from the same author (V4.1 is a
Lingpeng Jin line):

| Commit | Date | Content |
|---|---|---|
| `2b9f3412` (#2419, precursor; touches V4.1 but not `mono/`) | Sep | MiniMax-M3 mono with 1M context and indexer CP; V4.1 group32 FP8 and BF16 RoPE |
| **`08735170`** (#2479) | 2026-10-06 | **V4.1 mono decode, FP4 index plane, shared mono framework.** 177 files, about +19.5k / -4.5k. See §7 |
| `7fc0926c` (#2483) | 10-07 | `step_begin`: zero the mailboxes and fence the ranks in one FlyDSL launch (Triton 3.7 fence hang). About 0.82x step-begin time; stress test of 3000 steps clean (without the fence, wiped pushes appear) |
| `fbb89df0` (#2513) | 10-10 | Packed FP4 index logits (no `[rows, max_model_len]` plane; `ScorePlanner`, `step_needs_eager` for band overflow), two-source BF16 decode (`sparse_attn_v4_paged_2src`, OPUS split-K); mono launchers named after their kernel symbol for the FlyDSL cache directory. Needs ROCm/aiter#6334 |
| `355318c9` (#2507) | | Decoder SWA bounded replay for prefill (touches mono checkpoints) |
| **`5e8d9131`** (#2545) | 10-12 | **Token-parallel sublayers** (S >= 12, multiple of TP: reduce-scatter, own_view, quantized-row gathers; bit-identical); hand-off prefetch; fine ug below 7 rows; mHC seam reordering (norm before Sinkhorn gate, gate after wqkv_a in K1 and after the shared expert in K2; DPP Sinkhorn sums; early hc_fn loads up to 24 rows); K2 score through `buffer_load_to_lds`; **any decode step up to 48 rows**; one `peer_layout.py`; plus SWA bounded replay |
| `53cbd641` (#2535) | 10-16 | GLM-5.2 and Kimi-K3 mono on the shared framework (§8). Touches V4.1 only through framework fixes (LazyRunner per-width refusal, DYNAMO_AS_IS refusal, rollout-sync refusal, missing FlyDSL refused rather than crashing) |

The arc: one large PR landed the framework and V4.1 together, already bit-exact versus native, at 6/12-row widths.
After that came a hang fix in the step fence, the FP4 index plane and packed logits, then a scale-out to 48 rows with
token-parallel comms. There have been no V4.1 mono commits since 10-16; main continued with other models.

---

## 7. Shared-framework commit `08735170` (#2479)

`git show --stat` lists 177 files. Key groups:
- `atom/mono/**` (new):
  - **runtime**: TP consensus, peer memory and step fence, step mailboxes, lifecycle, widths, routing, debug reports,
    deployment refusals, check compare
  - **plan**: execution model, build keys, TP shard, mailbox layout, arithmetic, hand-off check
  - **device**: tagged mailbox publish, peer all-reduce, FP8/FP4 MX, wave and math primitives
- `atom/models/deepseek_v41/mono/**` (new): K1/K2a/K2b, DSpark draft runner, installed through
  `model_runner.mono_decode_installers`. `kernels/moe.py` is about 1.6k lines and `runner.py` about 680.
- MiniMax-M3 mono **moved onto** the shared framework. `peer_buffer.py` and its own timeline were removed; `contract`,
  `sources`, `dense_post` and `stages/*` (attention, dense_mlp, ffn_reduce, moe_common/narrow/wide, router, select,
  shared) were added; `post_attn.py` shrank by about 2.4k lines.
- The V4.1 FP4 index plane (`--index_cache_dtype fp4`) on both the original path and mono: page-8 value and scale
  planes, ATOM FP4 index writer, fused q RoPE + FP4 quant, FlyDSL row-group scorer, host step planner
  (`DeepseekV41MetadataBuilder.add_step_planner`).
- V4.1 MoE A8W4; mHC / Engram / DSpark / rejection-sampler fixes.
- MiniMax-M3 exact top-k (`atom/mono/device/select`); `lds_topk` cost cuts; `ops.batched_rounds`; CM_NT on weights read
  once per step.
- `runtime.compile.kernel_resources`; a default CUDA-graph capture ladder covering every batch size 1..8 (a padded
  batch used to run the next size's step and lost time).
- Tests: `tests/test_deepseek_v41_mono_{attention_plan,build_key,contract,dispatch,index_plan,moe_plan,tp_widths}.py`,
  `test_mono_{plan,runtime,shard}.py`, `test_minimax_m3_mono_{build_key,contract}.py`.

---

## 8. Sibling monos on the shared framework

| Model | Package | Kernels | Served shape | Requirements | Status on main |
|---|---|---|---|---|---|
| **GLM-5.2** | `atom/models/glm52/mono/` (kernel, layout, op, tp4_* and geometry; `kernels/support/` holds the enum vocabulary, launch geometry, weight ABI and symmetric peer buffer, vendored from FlyDSL) | "one persistent kernel launch a layer", the **MoE suffix** of each layer; optional fused indexer on the 18 FULL layers (`ATOM_GLM52_FUSED_INDEXER`, default 0) and indexer CP (`ATOM_GLM52_INDEXER_CP`, default 0) | 1 token, or an MTP verify of 4/5/6, batch 1 or 2 | TP4, ptpc-FP8 attention linears, fp8 KV, `ATOM_MLA_PAGE_SIZE=1`, gfx950, **MXFP4 experts with `ATOM_MOE_GU_ITLV=0`** (separated rows); no rollout sync | Enabled by default when eligible. Routes inside the **uncompiled** outer `GlmMoeDsaForCausalLM`, so `piecewise_cudagraph_required=False`. Refuses NeoX rope and unsupported attention scales |
| **Kimi-K3** | `atom/models/kimi_k3/mono/` (`kernels/attention/{kda,back}`, `kernels/layer`, `stages/{gemv,moe}`, plan, layouts) | **two kernels per KDA layer**: K1 (attention seam through the KDA recurrence) and K2 (`o_proj` + latent MoE). MLA layers stay native | uniform speculative verify of at most 8 rows (16 measured, not the default) | **TP8 only**, DSpark-only drafter, `ATOM_GDN_SSM_DTYPE` fp32, MXFP4 experts with **bf16 activations and separated rows**, preshuffled planes, CU count checked | Installed through `mono_decode_installers`. On by default. The commit notes an unsettled precision cost in the latent MoE and no task-level accuracy run. A dual-stream MoE refusal that disarmed mono by default was dropped. Refuses PIECEWISE (level 3's default for K3), so the documented launch must use FULL |
| **MiniMax-M3** | `atom/models/minimax_m3/mono/` (`pre_attn`, `post_attn`, `dense_post`, `index_score`, `stages/*`) | per-layer pre-attention and post-attention kernels (dense and sparse layers) | up to 16 tokens (requests x spec tokens), Eagle3 aux taps | TP4, ptpc-FP8 attention, fp8 KV and index cache, block 128, max_model_len up to 1M, FULL graph or eager, CU count checked; experts `not is_guinterleave` | First mono (#2419). Moved onto the shared framework in `08735170`. Accuracy reported with every decode step on mono. Has an SGLang bridge test, `tests/plugin/test_sglang_minimax_mono_refusal.py`, but plugin mode is refused by `common_refusals` |

What they share: `atom/mono/runtime` (consensus, `LazyRunner`, `WidthBuilds`, `common_refusals`, `step_begin` +
`StepMailboxes`, peer buffers, compile and trace), `atom/mono/plan` (execution model 256 x 512, shard, build key,
contract check), `atom/mono/device` (Mailbox, ops, mx, ranks, stamps). GLM's vendored `support/` keeps its own
symmetric peer buffer and primitives; duplicate copies of six primitives were folded into `atom/mono/device/ops.py` by
`53cbd641`, and `atom/model_ops/monokernel/` was deleted.

What differs: MoE activation precision (V4.1 MXFP8, interleaved; GLM, Kimi and MiniMax take bf16 or separated rows);
TP size (V4.1: 2/4/8; GLM and MiniMax: 4; Kimi: 8); kernels per layer (GLM and Qwen: 1; V4.1, Kimi and MiniMax: 2);
where routing lives (installer wrapper versus inside the model class).

---

## 9. Comparison with the Qwen FlyDSL mono (`qwen4_exp_mono` @ `352229d6`)

| Aspect | Qwen3.8-Flash-Next mono | DeepSeek-V4.1 mono |
|---|---|---|
| Launch granularity | **one FlyDSL launch per decoder layer**, 48 per step (`kernels/kernel.py:build_layer`); the stage subset is "prefix-closed" (`ATTN_HALF` = A1..C1, `DELTA`, etc.) | **two per layer** (K1, K2), 80 per step, so the native compressor and Engram can run between them |
| Persistent geometry | `GRID = 256`, `THREADS = 512` (`kernels/config.py`) | the same (`atom/mono/plan/execution.py`) |
| TP | TP1/EP1 only (`SUPPORTED_TP = (1,)`); no peer comms | TP2/4/8; in-kernel all-reduce or reduce-scatter through an uncached HIP-IPC peer buffer at system scope |
| Mailbox | vendored `kernels/mailbox.py`, adapted from FlyDSL `kernels/monokernel/glm/kernel.py` at v0.4.0.dev963; its own `buffer_ops.py` and `device_ops.py` | shared `atom/mono/device/sync.Mailbox` with a region and space type (`Addr`) recorded in the trace |
| Tag | `step * LAYER_SLOTS(64) + layer + 1`, from a **device step counter**: unique per launch, so no zeroing between steps | `layer + 1` (with K2a/K2b offsets 64/128); **zeroed every step** by `step_begin`, followed by a peer epoch fence |
| Scope rules | sc1 polls (plain and sc0 polls can spin on stale vL1D); agent scope everywhere; bulk data stored plainly, then vmcnt(0), barrier, one-thread `buffer_wbl2 sc1` release and `buffer_inv sc1` acquire | sc1 (CM_DEV) for scratch, sc0\|sc1 (CM_SYS) for peers; `publish` = vmcnt(0) + barrier + flag; bulk logits at device scope |
| CTA barriers | `layout_wide.barrier_before`: wide builds add a CTA barrier at phase changes | `gpu.barrier` between stages within a CTA; no grid barrier |
| MoE precision | **A8W4** (ATOM tensors in place; gate/up interleaved per 16-row block; MXFP8 activations; intermediate padded to 768) | **A8W4** in the same style (interleaved per 16 rows, MXFP8 act, padded to 128-multiples per rank) |
| MoE structure | M3 router, M4 routed+shared gate_up (110 tasks, wave-0 top-10 per task), M5 down + combine + MLP HC write | xq, router (shared fixed-order GEMV with the native path), route (LDS table, union U), shared, ug, down, reduce plus the TP push |
| Rows | up to 4 batched rows, or one request's k+1 verify (k <= 3), plus grouped verifies (`groups.py`, `group_runner.py`) | up to 48 rows, any split into requests, one build per width; token parallel from 12 rows |
| Shared framework use | only `consensus.MonoUnsupported`, `deployment.common_refusals`, `route.LazyRunner`, `build_key.source_digest`, `shard.Shard` | the full `atom/mono` stack (device, plan, runtime) |
| Contract check | none comparable | traced region tables (`contract.py`) |
| Out-of-kernel work | embedding, PLE n-gram embedding, `step_meta`, final HC mix (one per-row path) | embedding, `step_meta`, compressor, Engram, candidate table, collapse, gather |
| Parity | bitwise with native (Qwen DESIGN §7.8 gates) | bitwise for target layers and indexer selection (6-row reference order); the draft is not bitwise |

Implications for a rocKE port:
- V4.1's K1/K2 split is forced by the native compressor and Engram between them. A true single-launch-per-step mono
  would need those (and `step_meta` and the candidate table) moved in-kernel, as our Qwen DESIGN §3 "pass"
  granularity requires.
- The Qwen learnings (`q38-learnings.md` §1.6) found real two-layer stacking a small loss unless write-after-read
  hazards across layers are double-buffered and next-layer prefetch is held. V4.1's per-step zeroing and
  `layer + 1` tags already distinguish layers within a step, so cross-layer chaining would need layer-parity
  scratch.
- V4.1 has none of the NUMA dual copies or holds that our Qwen DESIGN §0.3 adopts. Those are candidate levers.

---

## 10. Files to read first for the port

1. `atom/models/deepseek_v41/mono/runner.py` (forward loop, argument packing, buffers)
2. `atom/models/deepseek_v41/mono/kernels/attn_pre.py` (K1 stages and task bases `GATE0`, `NORM0`, `WQKV0`, ...)
3. `atom/models/deepseek_v41/mono/kernels/attn_post.py` + `attention.py` (bit-matched decode attention)
4. `atom/models/deepseek_v41/mono/kernels/moe.py` + `moe_shape.py` + `moe_plan.py` (A8W4 MoE)
5. `atom/models/deepseek_v41/mono/kernels/index_score{,_fp4}.py` + `topk.py` + `index_plan.py` (indexer)
6. `atom/mono/device/sync.py`, `atom/mono/runtime/step_begin.py`, `atom/mono/runtime/peer_memory.py` (sync and comms)
7. `atom/models/deepseek_v41/mono/{config,weights,dispatch,contract}.py` (contract, refusals)
