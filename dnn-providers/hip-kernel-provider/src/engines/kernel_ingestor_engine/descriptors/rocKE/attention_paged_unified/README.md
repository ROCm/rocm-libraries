# hipkernel:AttentionPagedUnified

Paged (and ragged-query) SDPA forward for gfx942 and gfx950, served by rocKE's tiled 2D
unified-attention kernels (`rocke/library/kernels/{gfx942,gfx950}/attention_tiled_2d.py`,
`build_unified_attention_2d_tiled`) built with `runtime_layout=True`. The catalog source
of truth is `projects/hipdnn/tools/IngestorGenerator/configs/attention_paged_unified.yaml`;
the native half is `packs/AttentionPagedUnifiedNative.cpp`.

## Graphs served

cuDNN paged-attention semantics (`projects/hipdnn/docs/SdpaSemanticsRegister.md`, SEM-008
padding, SEM-009 paged K/V, SEM-010 empty rows), one SDPA-forward node, compute type FLOAT:

| Operand | Logical dims | Rules |
|---|---|---|
| Q, O | `[B, Hq, Sq, D]` | FP16 or BF16 (all four operands alike); dense with arbitrary strides, or both ragged through the same INT32 `[B+1,1,1,1]` `ragged_offset` (token units times each tensor's `ragged_offset_multiplier`) |
| K, V | `[num_blocks, Hkv, block_size, D]` | paged containers with equal, arbitrary strides (the interleaved K\|V views of one cache allocation included), a unit D stride, and one block size |
| page tables | `[B, 1, M, 1]` INT32 | `page_table_k` and `page_table_v`: one tensor for both, or two tensors with the same `M` and batch stride; unit stride along M; not packed |
| `seq_len_q`, `seq_len_kv` | `[B,1,1,1]` INT32 | both required, as for every paged graph |
| `max_seq_len_kv` | | set or unset; unset is inferred as `M * block_size` |
| `sink_token` | `[1, Hq, 1, 1]` | optional; FLOAT (cuDNN's form) or the Q/K/V/O data type, with a unit head stride (see [Attention sinks](#attention-sinks)) |

- Head size 64, 128 or 256; block size 16, 32 or 64; `Hq / Hkv` in {1, 2, 4, 5, 6, 7, 8, 16}.
- Masking: bottom-right causal (`right_bound` 0), optionally with a `left_bound` L, which
  keeps the L keys ending at the diagonal as in cuDNN: key k is visible to diagonal d iff
  `d - L < k <= d`, with `d = row + seq_len_kv - seq_len_q`. A vLLM sliding window W is
  `left_bound = W`. The kernel's `sliding_window` argument is L.
- Strides of non-unit axes and both buffer bases must keep 16-byte alignment.

A dense Q's rows past `seq_len_q[b]` (up to `Sq`) are padding and get O = 0. A ragged Q
writes only its live rows, the first `seq_len_q[b]` rows from its origin: rows of the offset
block past them belong to no sequence (RFC 0014), so the kernel leaves them untouched and
never writes past the last live row. A live row with no visible key, including every row
of a sequence with `seq_len_kv == 0`, also gets O = 0, with or without a sink. The kernel never
loads K/V past `seq_len_kv` nor page-table entries past `ceil(seq_len_kv / block_size)`, so
unused entries may hold anything (`-1`, a null block). K pages through `page_table_k` and V
through `page_table_v`, so with two tables V may sit in other blocks than K.

The engine does not clamp `seq_len_kv`: `0 <= seq_len_kv[b] <= max_seq_len_kv <= M *
block_size` is a precondition (with `max_seq_len_kv` unset, `M * block_size`), and a length
past it reads page-table entries past row `b`.

A ragged Q needs vLLM's `query_start_loc` layout: non-decreasing origins with
`ragged_offset[b+1] - ragged_offset[b] >= seq_len_q[b]` and `ragged_offset[B]` the end of
the last block. The first origin need not be 0. The kernel finds each query block's
sequence by a binary search over the offsets and sizes its grid from `ragged_offset[B]`.
RFC 0014 (#15080) also allows origins in any order and an unused `ragged_offset[B]` when
`seq_len_q` is bound; those are valid graphs whose offset values this engine cannot see at
match time, so it cannot decline them, and its output for them is wrong. The follow-up is
a packed query-block numbering driven by `seq_len_q` (a wave prefix sum over the
sequences' block counts), which reads each sequence's own origin and does not grow the
grid for a mixed batch.

Every other graph is declined, with the cause logged at INFO as
`hipkernel:AttentionPagedUnified declined the graph [<cause>]: <detail>`. Some are valid
hipDNN graphs this kernel cannot read:

| Cause | Graphs |
|---|---|
| `not_paged` | neither K nor V paged |
| `one_sided_paging` | only K or only V paged |
| `page_table` | K and V page tables of different widths `M` or batch strides, a packed table (with a ragged offset), a table that is not INT32 `[B, 1, M, 1]` with a unit entry stride |
| `alignment` | a declared tensor `alignment` (engine API 1.3.0) below what the kernels load: Q, K, V, O or `sink_token` not a multiple of 16 bytes (16-byte vectors), or a page table, `seq_len_q`, `seq_len_kv` or ragged offset not a multiple of 4 bytes (INT32 words) |
| `block_size` | K and V block sizes differ, or a block size outside 16/32/64 |
| `seq_len_q`, `seq_len_kv` | a missing or malformed length tensor |
| `ragged` | a ragged Q/O whose `ragged_offset_multiplier` is not its token (sequence-axis) stride, i.e. offsets stored in other units than tokens (RFC 0014 allows them; the kernel reads offsets as token counts); a ragged K or V; Q and O not ragged through one offset tensor; an offset tensor that is not INT32 `[B+1, 1, 1, 1]` |
| `mask` | no causal mask, top-left alignment, a right bound other than 0, `left_bound = 0` |
| `sinks` | a `sink_token` that is not `[1, Hq, 1, 1]`, one in a data type other than FLOAT or the Q data type, or a sink without a unit head stride |

Other causes: `alibi_mask`, `fp8`, `stats`, `dropout`, `attn_mask`,
`scale_tensor`, `block_mask`, `implementation`, `gqa`, `head_size`, `kv_layout`, `layout`,
`size_limit`, `operand`, `node`, `compute_data_type`, `shape`, `mma_core_mode`, `scale`.

## Execution

Workspace is zero. `execute()` performs one kernel launch with no host synchronization
and no allocation, so after a first (warm-up) execution on a device, which loads the
module, a plan can be captured into a HIP graph and replayed with new lengths, block ids
and data (`IntegrationGpuAttentionPagedUnified.HipGraphCaptureReplaysWithChangedLengthsAndBlocks`).

Each kernel is one object per (arch, dtype, head size, block size, GQA ratio, sinks): every
combination ships twice, as `paged_unified.<dtype>_d<D>_bs<bs>_g<ratio>.<arch>` and as
`..._g<ratio>_sinks.<arch>` (KMD `use_sinks` 1), which the matcher selects exactly when the
graph has a `sink_token`. The grid
is `(Hkv, q_block_grid)` with `num_warps * 64` threads; query blocks are walked by a
grid-stride loop, so a ragged batch whose total token count exceeds `Sq` is still covered.
K and V are read through one 64-bit buffer resource per page; K's page ids come from
`block_tables_ptr` and V's from `block_tables_v_ptr` (the same buffer for one table), with
one `block_table_stride`.
Split-KV (3D) decode needs kernel slots and is not part of this engine.

### Register pressure

The `runtime_layout` kernels take the Q/O/K/V strides, the head count, the window and the
query form as kernel arguments, which the fixed-layout kernels fold into constants. Those
values live in SGPRs, and most kernels run out: the compiler spills SGPRs into VGPR lanes
(`v_writelane`/`v_readlane`), never to scratch memory. Measured on the shipped catalog
(288 kernels per architecture, `runtime_layout_catalog_spec`), against the same specs with
`runtime_layout` off and no window:

| | gfx942 | gfx950 |
|---|---|---|
| Kernels that spill SGPRs | 264 of 288 | 232 of 288 |
| SGPRs spilled, per kernel (max) / in total | 57 / 4869 | 76 / 5596 |
| VGPR spills, scratch use | none | none |
| Lane moves added per kernel, median (max) | 30 (114) | 24 (152) |
| Lane moves as a share of the kernel's instructions, median (max) | 1.7% (7.4%) | 1.5% (6.3%) |
| Fixed-layout twins that spill | 0 of 192 that build | 4 of 288, 1 SGPR each |

The runtime head count, window and batch arguments of #15024 barely move the fixed layout
(gfx942: no spills before or after; gfx950: 2 kernels spilling 6 SGPRs before, 4 spilling 4
after), so the spills come from the runtime strides. The lane moves are a small part of
each kernel, but the lanes they hold count against VGPRs: the VGPR-limited occupancy
estimate is lower than the fixed twin's for 70 of 192 gfx942 kernels (higher for 20) and
for 209 of 288 gfx950 kernels. Removing the spills means fewer runtime arguments, and the
runtime arguments are what lets one plan serve every layout and step, so the cost is
accepted. The runtime cost has not been measured: no fixed-layout kernel serves the same
paged graph.

## One plan for every step

`B` in the graph is an upper bound. Batch size, lengths, offsets and block ids are read on
the device, so a plan built once serves every step of a serving loop: sequences past the
live batch carry `seq_len_q = seq_len_kv = 0` and their rows come back zero. For vLLM, bind
the cache's K and V halves as two containers over one allocation, its `block_table` as both
page tables (one tensor, `[B, 1, max_blocks, 1]`), `seq_lens` as `seq_len_kv`, the query lengths as
`seq_len_q`, and `query_start_loc` as the ragged offset of Q and O.

## Saved plans

The engine reports `SUPPORTS_EXECUTION_PLAN_SERIALIZATION`: a built plan saves with
`to_binary()` or `to_compiled_plan_binary()` and loads into a fresh handle without matching,
descriptor search or archive reads. The payload holds the kernel's code object and, under
the dispatch contract `hipkernel.attention_paged_unified.dispatch.v1`
(`packs/AttentionPagedUnifiedLaunchValues.hpp`), the launch values: the uids of Q, K, V, O,
both page tables, `seq_len_kv`, `seq_len_q`, the ragged offset and the sink (`-1` when
absent), the scale (any finite float, stored exactly as a double), every by-value kernel
argument (`num_seqs`, `block_table_stride`, the head counts, `sliding_window`,
`q_uniform_len` (`Sq` for a dense Q, 0 for a ragged one), the Q/O/KV strides,
`kv_block_span_bytes`, `q_block_grid`, `sink_is_f32` (1 for a FLOAT sink, else 0)) and
`block_threads`. The loaded plan
launches the same grid, `(num_kv_heads, q_block_grid)`, with `block_threads` threads. A load
refuses a value of another type, an extra or missing value, or one outside what the kernel
takes. Lengths, page tables, offsets and sinks stay device buffers bound at execute, so a
loaded plan serves every step just as the original does
(`IntegrationGpuAttentionPagedUnifiedSaveLoad`).

## Attention sinks

`sink_token` adds one logit per query head to the softmax max and denominator, with no value
row and no `attn_scale` (SdpaSemanticsRegister SEM-022); a row with no visible key still gets
O = 0. Both forms are served by the same `use_sinks` objects, and the launch argument
`sink_is_f32` tells the kernel which one `sink_ptr` holds:

- FLOAT `[1, Hq, 1, 1]`, cuDNN's form and the one the hipDNN frontend admits.
- The Q data type (FP16 or BF16), because vLLM keeps sinks in the model dtype
  (`layer.sinks`, a `[Hq]` tensor of raw logits): bind it without a copy as the view
  `[1, Hq, 1, 1]` with strides `[Hq, 1, 1, 1]`. A frontend that accepts only FLOAT sinks
  rejects such a graph before any engine sees it; the Q-dtype sink integration tests skip
  with that reason there.

The kernel reads exactly `Hq` logits of the declared type. The buffer must be 16-byte
aligned; a graph declaring a smaller `alignment` for it is declined with cause `alignment`.
