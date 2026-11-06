# ML4 decoder layer as one persistent launch (gfx950)

One launch computes one Mistral Large 4 decoder layer for one TP rank: the
attention block (with the fused sparse indexer), the MoE block, and both
tensor-parallel reductions. 256 resident CTAs of 512 threads run every stage's
tasks in program order and hand values to each other through tagged scratch
mailboxes; nothing returns to the host between stages.

The kernel is a reimplementation of the FlyDSL ML4 mono layer in aiter
(`aiter/ops/flydsl/kernels/mono/mistral_large4/gfx950`). It takes that
launcher's arguments in that launcher's order, places every task on the CTA the
FlyDSL build would, and reads and writes the same scratch, symmetric buffer and
caches, so a host swaps one for the other without any other change. Every stage
is checked bit for bit against the FlyDSL kernel on the GPU.

The `draft` kind is one DSpark draft layer of the model's speculator (no
FlyDSL counterpart): see [Draft kind](#draft-kind).

## Files

| file | role |
|---|---|
| `spec.py` | `Ml4LayerSpec`: kind, rows, heads, TP size, the stages to emit, and the placement plan and mailbox offsets (data from the host that owns the buffers) |
| `ctx.py` | `Ml4Ctx`: kernel arguments, ids, launch tag, the LDS pool, and the shared device helpers (loads, mailboxes, reductions, bf16 / E4M3 conversions) |
| `gemv.py` | MFMA GEMV units (FP8 block-128, bf16, native MXFP4 on the scaled MFMA), the pipelined unit loop, the cross-wave row reduction, RMSNorm staging |
| `stage_*.py` | one module per stage group (below) |
| `kernel.py` | `build_ml4_layer(spec, *, arch)`: emits `spec.stages` in order |
| `build.py` | command line: JSON specs to code objects, in a fresh process |

## Stages

| stage | tasks | work |
|---|---|---|
| `qkv_a` | 36 | RMSNorm of the input, the 576-row kv latent / rope key projection (FP8); indexer kinds also the index key (FP8) and head weights (bf16) |
| `q_norm` | rows | RMSNorm for the query projection; indexer kinds also the index query LayerNorm |
| `cache` | 1 | kv latent RMSNorm, rope key RoPE, E4M3 rows into the latent cache; indexer kinds: index key LayerNorm, RoPE, E4M3 with a power-of-two row scale into the paged index plane |
| `q_b`, `index_q` | 128, 128 | query projection with the SSMax scale and RoPE; index query tiles |
| `uk` | 64 | absorbed query `W_UK^T q_nope` |
| `index_score` | rows x 512 | index scores: query x E4M3 keys on MFMA, ReLU, head weights, row scale |
| `index_sel*`, `index_select` | per level | exact top-2048 as a merge tree of 16K-key radix selects |
| `split` | rows x 32 | sparse MLA over 64-key chunks of the selected keys, split-local softmax |
| `uv`, `o` | 128, 192 | split merge, W_UV, W_o, TP reduce and residual |
| `router` | rows x 16 | post-attention RMSNorm, expert logits, MXFP8 activation blocks |
| `ug` | 768 (all CTAs) | routing, MXFP4 expert up/gate with SiLU; the shared expert for every row in one MFMA; with `moe_em` also each expert's down, per-expert partials |
| `down` | 256 | mid MXFP8, expert down weighted by the routes, TP reduce and residual; with `moe_em` the sum of a row's per-expert partials, TP reduce and residual |
| `mlp_ug` | 2 inter / 48 | `draft` kind: post-attention RMSNorm, three 16-row tiles of the dense [gate; up] projection (FP8 block-128), bf16 to `mid` |
| `mlp_down` | 192 | `draft` kind: SiLU(gate) * up, the dense down projection (FP8 block-128), TP reduce and residual into `x_out` |

## Draft kind

`kind="draft"` builds one layer of the DSpark drafter: the target's layer
type without the indexer and with a dense MLP (`inter`: this rank's rows),
over a bf16 latent cache. The stages are `qkv_a`, `q_norm`, `q_b`, `cache`,
`uk`, `split`, `uv`, `o`, `mlp_ug`, `mlp_down`; the attention stages run as
for the other kinds, except:

- `cache` stores bf16 rows (576 bf16 a slot, vLLM's MLA cache layout) in
  vLLM's draft-layer rounding: the projection rounded to bf16, then
  bf16(RMSNorm(kv_c) * g_kv), and the rope key rotated with bf16-rounded
  products (`MistralRotaryEmbedding.rotate_`);
- `split` gathers bf16 cache rows, and its keys are whatever `indices` holds:
  the host writes the draft block's window (the last positions up to the
  block's end, the block's own rows included), so attention is non-causal
  over the block;
- `q_b` runs with the parameter table's SSMax `s = 0`, `b = 1` (the drafter
  has no SSMax; the scale is then exactly 1) and the drafter's RoPE tables.

Its weights are the drafter's bf16 linears rounded through block FP8 at
load (128 x 128 blocks, `W_UK^T` 128 x 64, as the target's mono weight
build does for its bf16 `kv_a` / `kv_b`). The MoE and indexer stages are
not part of the kind; the dense MLP stages are not part of the others.

## Conventions

- **Bit-exact arithmetic.** Plain `fmul` / `fadd` (never contracted), the
  hardware `rsq` / `rcp` / `exp2`, butterfly wave reductions over lane offsets
  32 .. 1, cross-wave sums in wave order, bf16 rounding to nearest even. A
  stage keeps the reference kernel's accumulation order; schedule changes that
  move work between CTAs or waves keep every output's order.
- **Mailboxes.** `(payload, tag)` pairs, one 8- or 16-byte store each, polled
  with device-coherent loads (`rocke.helpers.mailbox`, the compiler-visible
  `ir` implementation from `qwen38_mono.mailbox_ir`). The inline-asm stores
  hide their store-data hazard from LLVM, which can then reuse the data
  registers of a 16-byte store one cycle early.
- **Per-task barrier.** A CTA runs several stages; each task starts with a
  barrier because the previous task may still read the LDS it overwrites.
- **Python engine only.** The family uses Python-engine-only opcodes
  (`int_to_ptr`, the cache-policy buffer ops, `tagged_poll`, the scaled MFMA,
  `cvt_scalef32_pk_bf16_fp8`, `log`); build with `backend="python"`.

## Schedule knobs

`Ml4LayerSpec` fields past the placement data change only the schedule (what
loads are in flight when, which CTA or wave runs what, which waits and fences
publish a result); every setting but `moe_em` produces the same bits.
`moe_em` (on by default for two or more rows) sums a row's experts as
per-expert partials, so the layer output matches the slot-major sum to
rounding instead (`mid`, `sel` and `prob` stay bitwise); `moe_em=false`
restores the slot-major MoE. The launcher takes
overrides as JSON (`AITER_ML4_ROCKE_SPEC`), so settings can be compared on one
GPU in one run. Most of them trade latency hiding against the 256-VGPR budget
of one 512-thread CTA per CU, so they interact: a setting that loses alone
can win next to one that frees registers, and the reverse.

| group | knobs |
|---|---|
| up/gate | `ug_batch`, `ug_prefetch` / `ug_prefetch_rows` (next round's tiles during this round's reduction, for some rows), `ug_shared_first`, `ug_scale_late` (scale byte extracted at the MFMA, not after its load), `ug_slot_order` (rounds finish slots in down's order) |
| down | `dn_pipe` (shared and routed units in one software pipeline), `dn_batch`, `dn_shared_early`, `mid_chunk`, `dn_reroute`, `dn_resid_early`, `dn_resid_with_mid` (the residual polled with the first `mid` chunk), `dn_scale_late`, `dn_slot_phases` / `dn_group_pre` (`mid` polled and multiplied a slot group at a time) |
| expert-major MoE (`stage_moe_em.py`) | `ug_em` (each distinct expert of the launch's routing gets a group of CTAs, consecutive in XCD-blocked order, that splits its up/gate chunks with the verify rows as the MFMA's B columns), `moe_em` (the group then runs the expert's down tiles once its own `mid` is in, partials to `dn_part`; `down` sums them in slot order), `em_shared_ctas` (a fixed shared-expert group that loads before routing), `em_dn_pre` (the first down tile's weights during the `mid` wait), `em_tiles` / `em_prefetch` (up/gate chunks per reduction, and of the next batch in flight) |
| mailboxes and placement | `mb_numa` / `numa_hdr` (tagged regions in IOD-homed dual copies, each poll reading the copy homed at its CTA's IOD; the host writes the copy mask and sizes the scratch), `stage_maps` (per-stage task map, e.g. `split:xcd_blocked`) |
| top-k | `select_radix`, `sel_pad` (conflict-free reads of the staged keys), `sel_lean` (no per-item flags live), `sel_release` (store-completion wait instead of a release fence), `sel_range` (one select over range-sized buckets, then a rank), `sel_tagged` (indices also as tagged pairs that split polls) |
| index scoring | `score_newk_lds` (this launch's index keys polled once per CTA), `score_prefetch`, `score_before_uk`, `score_hoist` (the request's first position loaded once per CTA, not per tile), `score_lazy_newk` (no `index_ready` wait; the new keys polled only before a tile that holds one) |
| cache | `cache_kv_late` (indexer kinds: the cache task publishes the index keys first, and its latent rows run after its CTA's score tiles) |
| query / sparse MLA | `index_q_first`, `iq_prefetch`, `iq_dual` (leftover index-query tiles as second tiles on the index_q CTAs), `idx_proj_on_qb`, `split_q_first` (off for the `draft` kind: its keys need no top-k, so the gather goes ahead of the query wait), `split_bt_lds` (block table in LDS), `split_newkv_lds`, `split_acquire` |
| dense MLP (`draft` kind) | `mlp_act` (the host interleaves `w_ug`'s rows, 8 gate and the same 8 up a tile, so `mlp_ug` publishes the activation as bf16 pairs and `mlp_down` stages a quarter of the bytes), `mlp_ug_pipe` (two of a task's three tiles in flight), `mlp_dn_batch` |
| timeline builds | `timeline`, `tl_inner` (the mid-task stamps; they perturb the stage they sit in) |

## Checking

`aiter/op_tests/flydsl_tests/check_rocke_ml4_layer.py` runs one FlyDSL launch
per kind, then replays the captured arguments into rocKE builds of single
stages, stage groups or the whole layer (`--groups full`), on the golden
scratch with the group's outputs cleared, and compares the outputs bit for bit
(scratch mailboxes, caches, layer output). `--npes 4` runs the TP4 shards with
the peer reduces live. With `mb_numa` it checks whole layers only: copy 0 of
each doubled region against the golden region, copy 1 against copy 0. With
`moe_em` the layer output may differ from FlyDSL's by at most 1 bf16 ulp,
and `dn_part`, which FlyDSL does not have, must carry the launch's tag in
every pair; a `down` build without `ug` is skipped.

`aiter/op_tests/flydsl_tests/check_rocke_ml4_draft.py` checks the `draft`
kind on one GPU (one rank's TP4 shard, its TP sums its own partials) against
a PyTorch model of vLLM's draft layer on the same dequantized weights: each
stage's outputs against the model applied to the kernel's own inputs of that
stage, the bf16 cache rows it writes, every mailbox pair's tag, and the
layer output against the model's chain from the layer input (tolerance:
relative L2 error). `--bench N` times cold-cache launches.
