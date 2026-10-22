# hipkernel:AttentionPagedUnified

Paged (and ragged-query) SDPA forward for gfx942 and gfx950, served by rocKE's tiled 2D
unified-attention kernels (`rocke/library/kernels/{gfx942,gfx950}/attention_tiled_2d.py`,
`build_unified_attention_2d_tiled`) built with `runtime_layout=True`. The catalog source
of truth is `projects/hipdnn/tools/IngestorGenerator/configs/attention_paged_unified.yaml`;
the native half is `packs/AttentionPagedUnifiedNative.cpp`.

## Graphs served

cuDNN paged-attention semantics, one SDPA-forward node, compute type FLOAT:

| Operand | Logical dims | Rules |
|---|---|---|
| Q, O | `[B, Hq, Sq, D]` | FP16 or BF16 (all four operands alike); dense with arbitrary strides, or both ragged through the same INT32 `[B+1,1,1,1]` `ragged_offset` (token units times each tensor's `ragged_offset_multiplier`) |
| K, V | `[num_blocks, Hkv, block_size, D]` | paged containers with equal, arbitrary strides (the interleaved K\|V views of one cache allocation included) and a unit D stride |
| page table | `[B, 1, M, 1]` INT32 | one tensor for both `page_table_k` and `page_table_v`; unit stride along M |
| `seq_len_kv` | `[B,1,1,1]` INT32 | required |
| `seq_len_q` | `[B,1,1,1]` INT32 | optional; for a ragged Q without it, lengths come from the offsets |

- Head size 64, 128 or 256; block size 16, 32 or 64; `Hq / Hkv` in {1, 2, 4, 5, 6, 7, 8, 16}.
- Masking: bottom-right causal (`right_bound` 0), optionally with a `left_bound` L (key k is
  visible to diagonal d iff `d - L <= k <= d`, with `d = row + seq_len_kv - seq_len_q`).
- Strides of non-unit axes and both buffer bases must keep 16-byte alignment.

Sequences with `seq_len_kv == 0` write zeros. The kernel never loads K/V past
`seq_len_kv` nor page-table entries past `ceil(seq_len_kv / block_size)`. Batch size,
lengths, offsets and block ids are read on the device, so one plan serves every step
with `B` at most the planned batch (pad with zero-length sequences).

Every other graph is declined, with the cause logged at INFO as
`hipkernel:AttentionPagedUnified declined the graph [<cause>]: <detail>`. Notable causes:
`not_paged` (non-paged K/V), `page_table` (two page-table tensors, non-INT32), `mask`
(non-causal, top-left or a right bound other than 0), `sinks`, `alibi_mask`, `fp8`,
`stats`, `attn_mask`, `scale_tensor`, `gqa`, `head_size`, `block_size`, `kv_layout`,
`layout`, `ragged`, `seq_len_kv`, `seq_len_q`.

## Execution

Workspace is zero. `execute()` performs one kernel launch with no host synchronization
and no allocation, so after a first (warm-up) execution on a device, which loads the
module, a plan can be captured into a HIP graph and replayed with new lengths, block ids
and data (`IntegrationGpuAttentionPagedUnified.HipGraphCaptureReplaysWithChangedLengthsAndBlocks`).

Each kernel is one object per (arch, dtype, head size, block size, GQA ratio). The grid
is `(Hkv, q_block_grid)` with `num_warps * 64` threads; query blocks are walked by a
grid-stride loop, so a ragged batch whose total token count exceeds `Sq` is still covered.
Split-KV (3D) decode needs kernel slots and is not part of this engine.
