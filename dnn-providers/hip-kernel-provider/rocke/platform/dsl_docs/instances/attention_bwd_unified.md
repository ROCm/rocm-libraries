# Attention Backward, unified family: THD queries over a paged KV cache

Contract of the rocKE-native unified backward (dQ, dK, dV). It shares the
tile step, the statistics contract and the q-side aux kernels of the
layout-general family (`attention_bwd.md`); this page lists what differs.
Paged problems are not declared to hipDNN (its predicate keeps declining
`PAGED`); the family has its own declaration (version 2,
`ATTN_BWD_UNIFIED_DECLARATION_VERSION`).

Code: `library/kernels/common/attention_bwd_unified.py` (spec, main builder,
ABI), `attention_bwd_unified_plan.py` (request, predicate, workspace, plan,
declaration), `attention_bwd_unified_run.py` (launch glue),
`_attention_bwd_paged.py` (paged address generator).

## 1. Problem layout

| Tensor | Layout | Notes |
|---|---|---|
| Q, O, dO, dQ | `[total_q, h_q, D]`, `(token, head)` element strides, unit D stride | int32 `cu_seqlens_q[num_seqs + 1]` token offsets; dQ rows outside every span are untouched |
| LSE | natural-log fp32 `[total_q, h_q]` (optional trailing 1), `(token, head)` strides | same statistics contract as the general family |
| K, V caches | `[num_blocks, page, h_kv, D]`, `(block, token, head)` element strides | int32 `seq_lens_kv[num_seqs]`; int32 `block_table[num_seqs][max_blocks_per_seq]` with a row stride |
| dK, dV caches | the layout, strides and block table of K / V | written page-wise |

Masks: per-sequence bottom-right causal by default (`off = len_kv - len_q`),
top-left, sliding window, none. Heads: MHA, GQA, MQA (`h_q` a multiple of
`h_kv`). dtypes fp16 / bf16; head dims 32, 64, 128.

## 2. Page modes and output conventions

* **direct** (request field `block_table_injective = True`: distinct live
  (sequence, logical block) pairs map to distinct physical blocks): each CTA
  stores its tile's dK / dV into the physical slots of K / V. Deterministic.
* **atomic** (default; correct with shared prefix pages): fp32 atomics into a
  page workspace indexed by physical slot, zeroed by `page_zero` and written
  by `page_convert` (once for dK, once for dV). Non-deterministic in the last
  bits. Used whenever `g_split > 1` too.

Both modes give the same visible result on an injective table: every slot of
every page a live block references is written; slots of the last live page past
`len_kv` are exact zeros (in atomic mode unless a sequence sharing the page adds
to them); pages no live block references are untouched. No kernel reads
`block_table[s][j]` for `j >= ceil(len_kv / page)`.

Launch sequence: direct `prep -> main -> convert(dQ)`; atomic `prep -> page_zero
-> main -> convert(dQ) -> page_convert(dK) -> page_convert(dV)`. Prep and the dQ
convert are the general family's THD aux kernels (`rocke.attn_bwd.v3`,
`OFF_Q = cu_seqlens_q`).

## 3. Page classes and tiles

Every page size that is a positive multiple of 16 is served. The plan picks, in
order: the tuned configuration of `(arch, head size, page)` (`TUNED_PAGE_TILES`,
each entry measured) when legal for the page; the shipped `"multi"` tile with
the largest `kN0` dividing the page (one block-table lookup per kv tile, runtime
page); a shipped `"sub"` instance with `sub_page == page` (`kN0 / page` lookups
per kv tile, compile-time page; shipped sub pages 16, 32, 64 and 128, each where
it is below the default `kN0`); the one-wave `kN0 = 16` fallback (divides every
legal page; a performance cliff). Shipped CDNA tiles: the arch start point
(`"default"`), a four-wave `kN0 = 64` tile (`"k64"`), the sub instances of the
start point, the fallback; RDNA ships the one-wave tile only. In direct mode the grid covers `roundup(max_seqlen_kv, page)` keys,
so tiles in `[len_kv, roundup(len_kv, page))` store the zeros of the last live
page without running the q loop.

**Provenance of `TUNED_PAGE_TILES`** (qualitative; the measurements stay in the
private record). Every entry comes from a paired run of
`library/benchmarks/common/attention_bwd_unified_bench.py` on the cell named
in the entry's comment (an MHA, GQA or decode-like cell at that head size and
page) on gfx942 or gfx950: the planned unified route, every legal tile
configuration of the unified route (`attn_bwd_unified_tile_configs`) and the
general route on the gathered contiguous K / V, all gated against a float64
reference on the stored O and LSE before timing, then timed interleaved in the
same attempts. An entry records the tile that ran fastest on its cell where it
differs from what rules (2) - (4) choose: a sub instance for pages below the
default `kN0` (d32 / d64 on both targets, and d128 page 64 on gfx950; rule (2)
sent them to the smaller multi tile, which was slower), and the four-wave
`kN0 = 64` tile at d128 on the cells where it beat the default. A key that was not measured follows rules (2) - (4); no entry is
extrapolated to an unmeasured page or head size.

## 4. Spec (`AttnBwdUnifiedSpec`)

`head_size` is the only required field. Unified fields: `dtype` (`"bf16"`),
`page_mode` (`"atomic"` | `"direct"`), `mask_class` (`"band"` | `"none"`),
`sub_page` (`None` = the `"multi"` class, or the compile-time page of the
`"sub"` class), `stage_vec` (8 | 1). Every other field is a knob of
`AttnBwdSpec` with its default (derived from that spec, so a knob added to the
general family is a unified knob too); `None` knobs resolve to the general
family's THD start point, and legality is the general family's plus the page
rules.

## 5. Kernarg ABI `rocke.attn_bwd_unified.v1`

```
main         ptr : Q KC VC dO DKC DVC WS_LSE2 WS_DSUM WS_DQ WS_PDK WS_PDV CU_SEQLENS_Q SEQ_LENS_KV BLOCK_TABLE WORKLIST
             f32 : scale_log2 ds_mult dk_mult dq_mult
             i32 : q_t q_h do_t do_h kc_blk kc_tok kc_h vc_blk vc_tok vc_h bt_stride
                   h_q h_k G num_seqs max_seqlen_q max_seqlen_kv max_blocks_per_seq page
                   left right bottom_right ws_rows_q page_rows
                   g_split n_kv_tiles lb_order pair xcd_n xcd_chunk pack_heads use_worklist
page_zero    ptr : SEQ_LENS_KV BLOCK_TABLE WS_PDK WS_PDV
             i32 : h_kv num_seqs max_blocks_per_seq bt_stride page page_rows
page_convert ptr : SRC SEQ_LENS_KV BLOCK_TABLE DST
             f32 : mult
             i32 : h_kv num_seqs max_blocks_per_seq bt_stride page page_rows d_blk d_tok d_h
```

`page_rows` is the cache's physical slot count, `num_blocks * page`, in both page
modes (the workspace row bound in atomic mode and the block-id bound of the
guard, section 8).

Pointers are global, never `noalias`; inputs are `readonly`. Unused pointers are
0 (`DKC` / `DVC` in atomic mode, `WS_PDK` / `WS_PDV` in direct mode,
`WORKLIST`). `pair`, `xcd_n`, `xcd_chunk` and `use_worklist` must be 0 (mirror
pairing, the XCD remap and the work list are not built). Physical page rebase is
64-bit (`page_id * blk_stride` folded into the pointer); in-page indices are
32-bit.

## 6. Workspace

```
WS_DQ    A(4 * total_q * h_q * D)
WS_LSE2  A(4 * total_q * h_q)
WS_DSUM  A(4 * total_q * h_q)
WS_PDK   A(4 * num_blocks * page * h_kv * D)     atomic page mode only
WS_PDV   A(4 * num_blocks * page * h_kv * D)     atomic page mode only
```

`A(x)` rounds up to 256 bytes; sub-buffers are packed in this order. The caller
owns the workspace; nothing is allocated and no memset is issued.

## 7. Predicate

Ordered reason codes (first failure wins): `ARCH_UNSUPPORTED`,
`ARCH_NOT_VALIDATED`, `DROPOUT`, `BIAS_IN_BACKWARD`, `DBIAS`, `ALIBI`, `FP8`,
`DETERMINISTIC`, `UNSUPPORTED_FEATURE`, `SCALE_DEVICE_TENSOR`, `DTYPE`,
`OUT_DTYPE`, `LSE_FORMAT`, `DQK_NE_DV`, `HEAD_DIM_256_BWD`, `HEAD_DIM`,
`GQA_RATIO`, `BAND_BOUND`, `STRIDE_FORMAT`, `STRIDE_D`, `STRIDE_NEGATIVE`,
`OUTPUT_OVERLAP`, `STRIDE_ALIGN`, `INDEX_RANGE` (also `num_blocks * page >=
2^31` and `max_blocks_per_seq * page >= 2^30`), `GRID_LIMIT`, `SEQ_LEN_FORMAT`,
`RAGGED_FORMAT`, `KV_CACHE_LAYOUT` (cache dims not `[num_blocks, page, h_kv,
D]`, or dK / dV strides differ from K / V), `PAGE_SIZE` (not a multiple of 16),
`BLOCK_TABLE_FORMAT` (not int32 2-D, row stride too small, or `max_seqlen_kv`
beyond the table), `BLOCK_TABLE_NOT_INJECTIVE` (a non-injective table on an arch
where the atomic page mode is not declared), `SINKS`, `SOFTCAP`, `EMPTY`,
`INSTANCE_UNAVAILABLE`, `NOT_YET_DECLARED`.

Page sizes: every positive multiple of 16 (a power of two is not required);
anything else is declined `PAGE_SIZE` (kind `request_limit`), and the
declaration publishes the rule (`page_sizes`).

## 8. Host checks and device data

The run API is pointer-only and trusts its pointers (the hipDNN contract). A
caller that holds the tensors runs
`attention_bwd_unified_run.check_attn_bwd_unified_inputs(req, ...)` first. It
rejects, with `ValueError("<CODE>: detail")` (a code of
`attention_bwd_run.HOST_CHECK_CODES`): a `cu_seqlens_q` that is not monotone,
starts below 0, ends past `total_q` or has a span above `max_seqlen_q`; a
`seq_lens_kv` value above `max_seqlen_kv` or negative; a live block-table entry
outside `[0, num_blocks)` (`BLOCK_TABLE_RANGE`; entries past a sequence's live
blocks are never read and may hold anything); a physical block used twice
under `block_table_injective` (`BLOCK_TABLE_NOT_INJECTIVE`); and any tensor,
cache or table smaller than the request.

Device data that reaches a kernel without the check has a defined behaviour:

* a kv length is clamped into `[0, max_seqlen_kv]`;
* `page_zero` and `page_convert` range-check every block id they read: a bad
  id addresses block 0, its workspace slot moves past the workspace (to
  `page_rows`) and every store is predicated off, so no address leaves the
  cache or the workspace;
* the main kernel's paged address generator
  (`_attention_bwd_paged.PagedKv`) carries the same guard when it is built with
  the `page_rows` kernarg (a bad id addresses block 0, its K / V rows load as
  zeros, its direct-mode dK / dV stores are predicated off, its workspace slot
  moves past the workspace); the plan passes `page_rows = num_blocks * page` in
  both page modes, and the unified main kernel turns the guard on by passing
  that kernarg to its `PagedKv`.
