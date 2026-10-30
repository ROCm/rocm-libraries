# Attention 3D split-KV decode on gfx942 and gfx950 — case study

Measured scope: the split-KV ("3D") segment and reduce kernels for single-token
decode on gfx942 and gfx950, before and after the optimization series merged as
[ROCm/rocm-libraries#13492](https://github.com/ROCm/rocm-libraries/pull/13492). This
file records the mechanisms, which levers ship by default and which are conditional,
the honest losses and caveats, and how to replay the checks and the A/B. It publishes
no speed numbers: every outcome below is a qualitative statement bounded to the
measured configurations, not a general claim. Those configurations are the per-lever
A/Bs over decode shapes that the series' author ran during development (the exp2 one
over various shapes routed to the 3D path) and the series-wide comparison with default
dispatch on both arms (§5). The outcomes are transcribed from the commit messages and
the description of #13492 and from the author's development A/Bs and notes, which are
not in the repository (the gfx950 head-size-256 compiler note in §4 comes from the
docstring of `_gfx950_3d_decode_ctas_per_segment`), and were not re-measured for this
file. What was checked against the code or the compiler is listed in §5. The algorithm
is written up in the
[gfx950](../../gfx950/attention/ALGORITHM.md#51-inside-the-segment-kernel) and
[gfx942](../../gfx942/attention/ALGORITHM.md#8-the-decode-path-3d-split-kv)
`ALGORITHM.md` files; the fields are in the
[Knob reference](../README.md#knob-reference) and the runbook's
[§12.1.K2](../../../../platform/dsl_docs/optimization/optimization_runbook.md#121k2-attention-3d-split-kv-levers-ua-3d-specific).

## Contents

- [Terms](#terms)
- [1. Why decode needed work](#1-why-decode-needed-work)
- [2. Levers and outcomes](#2-levers-and-outcomes)
  - [2.1 Always on](#21-always-on)
  - [2.2 Spec fields and dispatch rules](#22-spec-fields-and-dispatch-rules)
- [3. Compatibility](#3-compatibility)
- [4. Caveats and honest losses](#4-caveats-and-honest-losses)
- [5. Method and correctness gate](#5-method-and-correctness-gate)
- [6. Reproducing](#6-reproducing)
- [7. Open items](#7-open-items)
- [8. Decision](#8-decision)

## Terms

- **Segment, "3D".** The split-KV path cuts each sequence's KV range into
  `num_segments` segments and gives each (query block, KV head, segment) its own
  workgroup, a three-dimensional grid; the reduce kernel merges the segments' partial
  results (§1).
- **Workgroup ("CTA").** Identifiers such as `target_ctas` and `*_ctas_per_segment`, and
  older prose in the `ALGORITHM.md` files, keep the CUDA name "CTA"; read it as
  workgroup.
- **Page, `block_size`, tile.** A paged K/V cache stores `block_size` tokens per page;
  the decode split rule needs 16. Strided K/V has no pages, and `kv_block_size`
  (default 16) plays that role. A *tile* is the KV tokens one loop iteration stages in
  LDS: one page by default, several pages (gfx950) or part of a page (gfx942) with
  `tile_size_override`.
- **All-decode** (`all_decode`). Every sequence in the batch has one query token.
- **KV tokens.** The longest sequence's KV length in the batch (`max_seqlen_k`).
- **`w` and CUs.** `w` is `num_seqs × num_kv_heads`, the workgroups in one segment's
  slice of the grid. CUs is `problem.num_cus`: the caller's value when it passes one,
  else the device's CU count when the dispatcher runs on a device of the requested
  architecture (at least 120 on gfx950), else 120
  ([gfx950 §6.1](../../gfx950/attention/ALGORITHM.md#61-how-many-segments-the-decode-split-rule)).
  Tuning without `num_cus` uses the architecture's full-part count instead (§3).
- **`target_ctas`.** An explicit workgroup-count target for routing and segmentation;
  `0` means auto (four per CU).
- **Wave page-id window** (`use_wave_page_id_load`). The wave holds 64 of the segment's
  page ids, one per lane. It has nothing to do with sliding-window attention.
- **`soffset`.** The scalar byte-offset operand of a buffer load; the wave-uniform page
  base rides in it.
- **Streaming K/V** (`kv_cache_policy="stream"`). Sets the non-temporal (NT) bit only;
  `"nt"` also sets SC0, a coherence-scope bit (§3), and is a different policy.
- **Wide feed, async copies.** How K/V reaches LDS. The async copies
  (`buffer_load … lds`) write K/V straight to LDS; gfx942's wide feed
  (`use_wide_kv_load`) loads 16 bytes per lane into registers and then stores them to
  LDS ([gfx942 §8.2](../../gfx942/attention/ALGORITHM.md#82-the-kv-feed)).
- **`BLOCK_Q`, XCD, "even x".** Explained in
  [gfx950 §5](../../gfx950/attention/ALGORITHM.md#5-the-3d-split-kv-path--many-ctas-share-a-query-block).

## 1. Why decode needed work

Decode sends one query token per sequence through the split-KV pipeline: a segment
kernel (one workgroup per query block, KV head and segment) writes a partial
$(m, \ell, \mathbf{o})$ to a workspace, and a reduce kernel merges the segments
([gfx950 ALGORITHM.md §5](../../gfx950/attention/ALGORITHM.md#5-the-3d-split-kv-path--many-ctas-share-a-query-block)).
With one token per sequence there is almost no math per KV byte, so the loop is bound
by latency and by how much of the device the grid fills. Before the series it paid
for:

- a prefill-shaped grid (`total_q // BLOCK_Q + num_seqs` q-blocks, each workgroup
  binary-searching `query_start_len`), which launched empty workgroups and, at 16
  query heads per KV head, put every useful workgroup on an even x and so on half of
  the XCDs (gfx950 `ALGORITHM.md` §5, linked above, explains why);
- a $P$ tile that went through LDS between the QK and PV MFMAs;
- a page-table load per KV tile whose wait drained the K/V copies already in flight;
- K/V, each byte read once per step, cached like reusable data;
- a segment count from a formula built for prefill, which distorted decode (q-blocks
  counted as for prefill, powers of two, at least 16 segments).

## 2. Levers and outcomes

Outcome words compare against the previous kernel on the same shapes: *faster*,
*neutral* or *slower* at the measured configurations. *Kept* means the lever ships on
its default path; *conditional* means it is on only where it measured well; *under
revision* means it overlaps open work (§7). *Alone* means the lever was timed without
the levers it ships with. *No separate outcome recorded* means the lever has no A/B of
its own, so only the series-wide comparison covers it. Most A/Bs ran on K/V that is
not cache-resident between calls (§4).

### 2.1 Always on

These have no spec field. Derivations: [gfx950 §5.1](../../gfx950/attention/ALGORITHM.md#51-inside-the-segment-kernel)
and [gfx942 §8](../../gfx942/attention/ALGORITHM.md#8-the-decode-path-3d-split-kv).

| Lever | Arch | Mechanism | Outcome |
|---|---|---|---|
| Transposed QK, register-resident $P$ | both | $S^{\top} = K Q^{\top}$: one $(m, \ell)$ per lane, no $P$ tile in LDS | Kept. gfx950: faster on long-KV, single-sequence and head-size-256 decode, neutral on short batched decode. gfx942: faster on batched, single-sequence and head-size-256 decode. Outputs match the previous kernel within rounding. |
| $Q^{\top}$ read once | both | loaded into registers before the KV loop instead of read from LDS every tile | Kept. gfx950: faster on long-KV and batched decode, neutral on short single-sequence decode; costs a few VGPRs and leaves occupancy unchanged. gfx942: faster alone on the 3D decode rows (it also landed with the transposed-QK change). |
| One V LDS slot | both | $V(i)$ is consumed in the iteration that loads it, so the second slot never held a live tile | Kept. gfx950: raises occupancy at the default tile, with broad gains on decode shapes. gfx942: head size 256 at block size 64 now builds (it had exceeded the 64 KiB LDS limit while reported as supported); neutral at the default decode tile. |
| Native `exp2` | both | `llvm.amdgcn.exp2` (one `v_exp_f32`) in the segment softmax and the reduce kernel | Kept. No regression, small gains on kernel-bound decode shapes, outputs matched on the measured shapes. |
| `16x16x32` PV | gfx950 | two 16-token blocks per MFMA on tiles of 32 or more tokens | Kept. Half as many PV MFMAs on those tiles; 16-token tiles unchanged. Paged K/V at 64-token tiles: slightly faster under LLVM 22 and LLVM 23. Strided K/V at 64-token tiles: slower under LLVM 22 (the series' author traced it to LLVM 22 issuing all QK$^{\top}$ MFMAs before the next tile's copies) and slightly faster under LLVM 23, a loss in a shipped default (§4). |
| Page ids a tile ahead | gfx950 | for paged fp16/bf16 K/V the next tile's page id is read during the current tile and passed as the copy's `soffset` (or folded into a 64-bit buffer base for caches above 2 GiB), instead of a block-table read issued with the K/V copies, whose wait drained the copies already in flight | Kept, under revision. Neutral to positive on decode shapes, largest on kernel-bound single-sequence and large-page shapes. Overlaps the open decoupled-KV-pages work (§7). |

### 2.2 Spec fields and dispatch rules

**What a decode problem gets by default.** The rows are for fp16/bf16 K/V on all-decode
problems; anything else keeps its previous defaults unless a cell says otherwise
("whatever the K/V dtype", "any 3D problem"). Strided K/V never takes the swizzle, the
page-id window or the wide feed by default. "The split rule" is the decode split rule
([gfx950 §6.1](../../gfx950/attention/ALGORITHM.md#61-how-many-segments-the-decode-split-rule),
[gfx942 §8.3](../../gfx942/attention/ALGORITHM.md#83-the-decode-split-rule)); $w$ and
CUs are defined in [Terms](#terms).

| Default | gfx950 | gfx942 |
|---|---|---|
| Decode grid, `use_decode_grid` | on, whatever the K/V dtype, sliding window or `target_ctas` | same |
| K/V cache policy, `kv_cache_policy` | `"stream"`; fp8 K/V keeps `"all"` | same |
| Split rule, `num_segments` | `block_size` 16, no sliding window, no `target_ctas`; head size 128, or 256 when $w \cdot 32 > \text{CUs}$ | the same, with head size 64 or 128, and either $w \ge \text{CUs}/16$ or fewer than 8K KV tokens |
| KV tile, `tile_size_override` | one page; 64 tokens in the rule's scope when $w \cdot 32 \le \text{CUs}$ | one page; half a page at head size 128 and above with `block_size` 32 and above (any 3D problem) |
| K/V feed (gfx942: `use_wide_kv_load`) | async copies | wide feed up to head size 128; async copies above 128 (any 3D problem), for strided K/V, and with `HIPDNN_GFX942_3D_WKV=0` |
| K/V LDS swizzle, `use_kv_lds_swizzle` | on with the 64-token tile, paged K/V only | on with the wide feed at head size 64 and 128; at 256 with page-sized tiles (`block_size` 16), always together with the page-id window |
| Wave page-id window, `use_wave_page_id_load` | on in the rule's scope, paged K/V only | on with the swizzle, for page-sized tiles |

To see what a given shape gets, build the spec the dispatcher builds, from a problem
whose `num_cus` is what the dispatcher would resolve (CUs in [Terms](#terms)): for paged
K/V, `_tiled_3d_spec_from_problem(problem, arch)` in
[`attention_spec_builder.py`](../attention_spec_builder.py); for strided K/V,
`_strided_3d_specs_from_problem(problem, arch=arch)` in
[`attention_unified.py`](../../../kernels/common/attention_unified.py). The
[split-rule test](../../../tests/test_attention_tiled_3d_decode_split.py) calls both (its
`_problem` helper builds the `UnifiedAttentionProblem`). For paged K/V, the
`dump_shape(label, req)` helper of
[`decode_dispatch_diagnose.py`](../../../benchmarks/gfx950/attention/decode/decode_dispatch_diagnose.py)
prints the selected path, segment count, tile, wide-feed flag and kernel name. The
script takes no command-line arguments and its own requests use `num_cus=120`, so call
`dump_shape` yourself with your own request and the device's CU count: the script's
`_req(**overrides)` builds one, and
`dump_shape("4 seqs, 16K", _req(batch=4, seqlen_k=16384, num_cus=256))` prints
`num_segments=32` and `tile=16`, the second worked example of gfx950 §6.1. It first
prints every registered candidate, thousands of lines for this example, so the summary
lines are the last ones. The kernel-name tokens in §3 show which levers are on.

Outcomes by lever (the conditions are in the table above):

| Lever | Arch | Outcome |
|---|---|---|
| Decode grid, `use_decode_grid` | both | Kept. Large gains at 16 query heads per KV head, neutral elsewhere. |
| Streaming K/V, `kv_cache_policy="stream"` | both | Kept. gfx950: faster on batched decode, neutral on single-sequence decode. gfx942: neutral to slightly faster. Both measured with K/V that is not cache-resident between calls (§4). |
| K/V LDS swizzle, `use_kv_lds_swizzle` | both | gfx950: conditional. Neutral at page-sized tiles, faster at multi-page tiles, most on single-sequence and head-size-256 decode; that A/B set the multi-page tiles by explicit spec, since default dispatch then selected none, and default dispatch now gives them only to head size 128 (table above). Strided K/V: slower in the segment-by-knob sweeps (the best arm with the swizzle was behind the best arm overall, clearly so on the worst shape), so default dispatch leaves it off; there is no gfx942 strided measurement. gfx942: kept for all-decode problems (conditions in the table above). Head size 128: faster alone, and faster still with the page-id window. Head size 64: no separate outcome recorded. Head size 256 (async copies): alone it is slightly faster under LLVM 22 but clearly slower under LLVM 23; the window alone is neutral under LLVM 22 and slightly faster under LLVM 23; the pair is slightly faster under LLVM 22 and faster under LLVM 23, which is why the two ship together. |
| Wave page-id window, `use_wave_page_id_load` | both | Kept. gfx950: only the pair with the retune to about 4 workgroups per CU (below) paid. The window alone, at the then-current about 8 workgroups per CU, was slightly slower on batched decode, and halving the segments without the window was slower too; the pair was faster on batched decode and unchanged on single-sequence decode and outside the rule's scope. gfx942, head size 128: alone neutral on batched decode and faster on a single-sequence row; with the swizzle faster than the swizzle alone. Head size 64: no separate outcome recorded; head size 256: see the swizzle row. |
| Multi-page KV tiles | gfx950 | Conditional, under revision. 64-token tiles with 64 segments and the swizzle off: faster than the previous kernel on single-sequence decode (the single-sequence tune, below). The swizzle adds a little more at 64-token tiles and is neutral at 16-token tiles. The tile was not separated from the 64 segments (§7). Overlaps the open decoupled-KV-pages work (§7). |
| Async K/V copies at head size 256 | gfx942 | Kept. A large win with LLVM 23, where the wide feed's load/store pairs serialize at head size 256; neutral with LLVM 22. |
| Decode split rule | both | Kept. History and attribution below. |
| Invariant hoist, `use_invariant_hoist` | both | Removed. The kernels compute the query-row invariants once per lane before the loop, so the field selected nothing. The environment toggle, dispatch helper, tuning axis and name token are gone; the spec field stays, inert, so saved specs still load. |

Decode split rule history:

- gfx950, single-sequence tune (one sequence, head size 128, paged K/V, KV of 4K tokens
  or more): 64-token tiles and 64 segments with the swizzle off, then the swizzle on
  top, as two A/Bs. Faster on every covered single-sequence shape, unchanged elsewhere.
- gfx950, first version, which generalized it (about 8 workgroups per CU, rounded down
  to a power of two between 4 and 64 segments; 64-token tiles when even 64 segments
  leave the device underfilled): faster on large batches and on small batches with few
  KV heads, and unchanged where the previous split already matched.
- gfx950, retune to about 4 workgroups per CU (still a power of two, 4 to 64): only
  together with the page-id window (above). A later 48-shape refit with the window on
  put about 4 per CU close to the best arm (geomean; the worst shape was clearly
  behind). The later extension added KV below 4K tokens, head size 256 and strided K/V,
  and it changed the counts to what ships: $\lfloor 4\,\text{CUs}/w \rfloor$ without the
  power-of-two rounding, at least 2 and at most 64 (128 from 8K KV tokens), and with few
  workgroups at most one segment per 64-token tile. Against the previous counts, on the
  shapes whose count changed, it was faster on the changed decode rows (no losses), on a
  synthetic paged sweep (a few shapes slightly slower) and on a synthetic strided sweep,
  and similarly under LLVM 23. The series-wide comparison covers the final rule.
- gfx942: scoped to where it measured well on net (enough workgroups per segment, or KV
  below 8K tokens); the previous counts stay elsewhere. Against the previous counts it
  was faster on the changed decode rows (no losses), on a synthetic paged sweep (a few
  shapes slower) and on a synthetic strided sweep, with small losses at short KV inside
  its scope (§4).

## 3. Compatibility

- **Tuning ids.** `kv_cache_policy`, `use_decode_grid`, `use_kv_lds_swizzle` and
  `use_wave_page_id_load` are new 3D spec fields and part of the 3D tuning ids, so
  every 3D tuning id changed: an id recorded before the series does not match. The
  kernel-name tokens `dg`, `kvcp<policy>` (a policy other than `"all"`), `kvswz` and
  `wpid` mark the new variants. `use_decode_grid` is derived from the problem; the
  others are also tuning axes.
- **Tuning space.** It can now express the kernel that default dispatch builds for
  decode: it adds segment candidates 2, 4 and 0 (the count `_num_segments` picks, the
  split rule's where it applies; recorded in the id) to the existing 8–128, explicit 3D
  configs take any count of at least 1, and without `num_cus` tuning uses the
  architecture's full-part CU count instead of 120.
- **Removed.** `HIPDNN_GFX942_3D_HOIST` and the hoist tuning axis. A saved spec that
  sets `use_invariant_hoist` still loads.
- **Cache-policy names.** `CACHE_POLICY_NAMES` and `cache_policy_from_name` (Python;
  `rocke_cache_policy_from_name` in C) replace per-kernel name maps. The gfx950 2D glue
  now rejects an unknown name instead of treating it as `"nt"`, and the cache-bit comments
  describe gfx942/gfx950 (SC0 selects the coherence scope, NT marks streaming data). A
  test pins the bits each name puts on a buffer LDS load.

## 4. Caveats and honest losses

- **K/V residency.** Real decode reads each K/V byte once per step, so most per-lever
  A/Bs used K/V that was not cache-resident between calls ("cold": every call read the
  next of several K/V copies in distinct memory, about 2 GB per shape in total, timed
  kernel-only). Cold were every split-rule step and the segment sweeps it was fitted
  from, the K/V swizzle and the wave page-id window on both architectures, the
  transposed QK and $Q^{\top}$ read once on both (gfx950 also warm, where the results agreed),
  the `16x16x32` PV, the non-temporal policy on both, and the gfx942 head-size-256
  copies. Four levers were measured warm, with host launch included, before the cold
  mode existed: the decode grid, native exp2, the one V slot and page ids a tile ahead.
  The series-wide comparison replayed the same inputs, so its K/V could stay warm; it is
  the only coverage of the levers with no A/B of their own (§2.2). Warm is the
  conservative direction for the non-temporal hint (replaying a small K/V makes it look
  slower, because the K/V no longer stays in the caches). For the four warm levers the
  direction is not known; they change instruction counts and waits rather than memory
  behavior, and the series' author expects them not to flip (§7).
- **Host launch.** A few small decode shapes are marginally slower once the host launch
  is included. Their kernels are faster on the device, but each call is bound by host
  launch overhead, which the series raises slightly. No measured shape on the 3D path
  was slower in device time. Reducing the per-call host overhead is a follow-up.
- **Compiler dependence.** The gfx942 wide K/V feed overlaps its loads with its stores
  only if the compiler schedules them that way (LLVM 22 does, LLVM 23 does not at head
  size 256), and the gfx942 head-size-256 swizzle alone is slower under some compilers.
  The gfx950 head-size-256 split rule is limited to grids with enough workgroups for
  page-sized tiles because its few-workgroup splits measured faster under one compiler
  and slower under another. The strided-K/V loss and the reduce-kernel slow zone below
  depend on the compiler too. The compiler is the COMGR the process loaded, not
  `ROCKE_LLVM_FLAVOR` (§6). Re-measure these on a compiler change.
- **Strided K/V at 64-token tiles.** gfx950 strided decode with few workgroups
  ($w \cdot 32 \le$ CUs) gets 64-token tiles by default and, with them, the unconditional
  `16x16x32` PV (§2.1). Under LLVM 22 that is slower than the previous kernel; under
  LLVM 23 it is slightly faster. It is a loss in a shipped default. The strided K/V
  swizzle is off because it measured slower (§2.2).
- **Small split-rule losses.** The split rule is faster in aggregate, not on every shape.
  gfx942 lost a little at 32 to 64 workgroups per segment with 1K to 2K KV tokens,
  inside its scope (the "KV below 8K tokens" clause). gfx950 lost a little at 16 to 32
  workgroups with 1K KV tokens, and at head size 256 with 64 KV heads and at least 64
  workgroups. Every changed decode row of the benchmark set improved. Not tested:
  non-power-of-two $w$ (the synthetic sweeps used powers of two), ragged batches and KV
  beyond 64K tokens.
- **Reduce kernel slow zone.** The reduce kernel loops over the segments with a
  compile-time trip count. Up to about 100 segments LLVM unrolls the loop completely, and
  for some counts the unrolled code waits for each load before issuing the next
  (`s_waitcnt vmcnt(0)` after almost every load), which makes the reduce markedly
  slower: the series' author timed it on gfx942 at head size 256 with 76 and 96 segments,
  against 64 or 128. Which counts serialize depends on the compiler and the head size,
  and they are not a clean range. Compiled offline (no device) and counted as described
  below, every count tried from 66 to 101 serialized at gfx942 head size 256 under
  LLVM 22 (48 to 101 under LLVM 20) while 112 and 128 stayed rolled, and none of 57 to 101
  did at gfx942 head size 64. On gfx950 under LLVM 22, head size 256 serialized at all but
  80 of the counts tried from 67 to 101 (only partly at 102), and head size 128 only at
  102. To check a count, compile a reduce spec with that `num_segments` for the target and
  compiler (`build_unified_attention_reduce_tiled`, `compile_kernel`), disassemble it with
  `llvm-objdump -d`, and compare the `s_waitcnt` lines that wait for `vmcnt(0)` with the
  `global_load` lines: about one per load is the serialized form. The shipped split rule
  reaches the zone on gfx950: with 256 CUs and 8K or more KV tokens it returns 113, 102,
  93, 85, 78, 73 and 68 segments at 9 to 15 workgroups per segment, so under LLVM 22 head
  size 256 serializes for 11 to 15 workgroups (partly for 10) and head size 128 for 10. On
  gfx942 the rule's counts above 64 occur only at head size 64, which does not serialize,
  and head size 256 is outside the rule. No shape of the series' benchmark set landed in
  the zone. A reduce redesign is a follow-up (§7).
- **Coverage.**
  - The series-wide comparison used one device per architecture and one compiler build
    (an LLVM 23 COMGR, per the series' author; the PR names only the benchmark's ROCm
    build). The per-lever A/Bs from the gfx950 K/V swizzle on, and all gfx942 ones, ran
    with the LLVM 22 COMGR bundled with torch, several also under LLVM 23; the earlier
    gfx950 A/Bs recorded only the node's ROCm with torch loaded. The gfx942 head-size-256
    loader was compared across LLVM 22 and 23 (§2.2). A GQA ratio that does not divide 16
    runs the tiled 2D kernel in both arms, not the 3D path (§7).
  - The kernel-body rewrites of §2.1 are not decode-gated: they run for every problem
    routed to the 3D path (prefill, speculative decode, mixed batches), subject only to
    their own conditions (paged fp16/bf16 K/V for page ids a tile ahead; tiles of 32 or
    more tokens for the `16x16x32` PV). Apart from exp2, their A/Bs used decode shapes,
    and the series-wide comparison had few other shapes (§5).
  - The split rule is narrower than the defaults around it (table in §2.2). It does not
    apply to fp8 K/V, sliding windows, an explicit `target_ctas` or non-decode problems,
    nor outside its head sizes (gfx950: 128, and 256 with enough workgroups; gfx942: 64
    and 128), its `block_size` of 16 and, on gfx942, few workgroups at long KV; those
    problems keep their previous segment counts and tiles.
  - Other defaults reach further: the decode grid and, for fp16/bf16 K/V, the
    non-temporal policy follow `all_decode` alone; on gfx942 the K/V swizzle and the
    page-id window follow `all_decode`, the head size and the K/V feed, so head size 256
    (page-sized tiles) gets them although the rule does not cover it; and gfx942 at head
    size 256 uses the async copies for every 3D problem, decode or not.

## 5. Method and correctness gate

**Provenance.** The outcomes in §2 come from the #13492 commit messages and PR
description, where the relative results live, and from the series author's development
A/Bs and notes (not in the repository); none was re-measured for this file, and the
outcome words are the series author's statements. The code facts (dispatch paths,
defaults, split-rule counts, harness scope, the cost of the per-launch cache key) were
checked against the tree, and the reduce-kernel slow zone of §4 against compiled code.

**Correctness.** The PR description says every shape's result was checked against a
reference before it was timed. The gates it lists:

- GPU numerics against an FP32 reference, reported on gfx942 and gfx950 with both
  engines: the new knob suites
  ([K/V swizzle](../../../tests/test_attention_tiled_3d_kv_lds_swizzle.py),
  [wave page-id window](../../../tests/test_attention_tiled_3d_wave_page_id_load.py);
  their GPU tests run on a gfx942 or gfx950 device, each on its own architecture, and
  skip elsewhere) with reversed and poisoned page tables, ragged and empty sequences
  and multi-page tiles, and the
  [strided-KV decode numerics](../../../tests/test_strided_kv_decode_numeric.py). The
  knob and strided suites run fp16 and bf16 and compare with absolute/relative
  tolerances of 0.003/0.005 (fp16) and 0.01/0.02 (bf16). The test plan also lists the
  forced-3D parity harness (§6) on both architectures; it ran on gfx950 only (below).
- CPU suites for the [split rule](../../../tests/test_attention_tiled_3d_decode_split.py)
  and the [cache policy](../../../tests/test_attention_tiled_3d_kv_cache_policy.py), and
  a platform [ISA test](../../../../platform/tests/core/test_buffer_lds_cache_policy_isa.py)
  of the cache-policy bits on gfx942 and gfx950.
- Python and C++ engines emit byte-identical LLVM IR at every flavor the gate runs
  (`llvm20`, `llvm22` and `llvm23` when the series landed;
  [`check_byte_identity.py`](../../../../platform/tools/check_byte_identity.py)),
  with the `.py`/`.c` emitter pairs
  [gfx942](../../../tests/parity/gfx942_attention_tiled_3d_emit.py),
  [gfx950](../../../tests/parity/gfx950_attention_tiled_3d_emit.py) and
  [strided KV](../../../tests/parity/attention_strided_kv_emit.py). The representative
  and strided-KV IR goldens were re-blessed where the 3D kernels changed.

The PR description does not say the following; it comes from the series author's notes:

- The check that gated the timed runs. In the per-lever A/Bs, each row compared the
  output of its first and its last sequence against an FP32 reference before timing,
  with a max-abs tolerance of 2e-2, in fp16 and bf16. In the series-wide comparison,
  each shape's whole output was compared by max-abs against an FP32 reference before
  timing; anything above 5e-2 was flagged, not dropped. Its decode rows are fp16/bf16.
  Both tolerances are absolute, so the checks catch less where the reference outputs are
  small, as they are for long-KV decode on random inputs (the outputs shrink with the KV
  length).
- The forced-3D parity run, on gfx950 only: `parity_unified_attention.py --paths 3d
  --set all`, once per scenario, without the nine `pr9233_*` scenarios, which fault when
  forced to 3D on develop too. That leaves 48 scenarios, 45 of which produce a 3D result
  (the three `gqa_nqk` scenarios with ratios 3, 5 and 7 fail the 3D gate and produce
  none). The harness has no pass/fail and exits 0 whenever it runs to the end, so the
  author judged each scenario's max-abs against the previous commit's run.
- gfx942 3D correctness. The gfx942 parity harness is 2D-only (§6), so no forced-3D
  parity ran there, which the #13492 test plan overstated. The 3D kernels rest on the
  knob, strided-KV and page-id suites above, on feature checks against the base tree
  (bit-identical for the cache policy, $Q^{\top}$ read once, the swizzle and the page
  ids; within rounding for the transposed QK) and on the per-row check above.

**Performance.** The per-lever A/Bs ran over decode shapes (the exp2 one over various
shapes routed to the 3D path), most of them interleaved. An arm was either an
environment toggle in one tree with per-arm kernel caches, interleaved row by row, or a
commit run against its parent in alternating processes. The series-wide comparison used
default dispatch on both sides (no tuned specs): the series with develop `a76c2460399`
merged in (`5b2dd750b61`, a commit of #13492) against `a76c2460399`, over a corpus of LLM
attention shapes (prefill and decode; 251 on gfx950, 248 on gfx942) on one gfx950 and one
gfx942 device. Each shape ran 5 rounds of 75 warm-up and 250 timed calls, reported as the
median, with two timings: wall time per call including the host launch, and device time
from HIP-graph replay. The cohorts on the 3D path were:

| Cohort | Shapes |
|---|---|
| Decode, 1 sequence | 14 |
| Decode, 2–16 sequences | 25 |
| Decode, 32 or more sequences | 16 |
| Decode, head size 256 | 7 |
| Prefill routed to 3D | 1 on gfx950, 3 on gfx942 |

Another 15 decode shapes have query heads per KV head that do not divide 16. They are not
on the 3D path: both arms ran the tiled 2D kernel on them, or the scalar kernels where 2D
rejects the shape (§7).

**Step 0.** The series' commit messages and PR description record no
[Step 0](../../../../platform/dsl_docs/optimization/optimization_runbook.md#step-0--exhaust-the-existing-lever-space-first-do-this-before-touching-the-algorithm-or-structure)
sweep of the existing lever space before the kernel-body rewrites of §2.1, but the
series' author did one on gfx950 and did not record it there. The then-existing lever
space was measured at its per-shape tuned ceiling (offline-tuned specs per shape), with
a leave-one-out and an add-one knob ablation and profiling, warm and with host launch
included. On the 3D shapes the segment count was the main lever and alone gave the full
tuned gain; `waves_per_eu`, staggered waits and `num_cus` were neutral. The remaining
tuned gains came from routing short-KV and large-batch shapes to tuned 2D configs and
from graph replay. Even the tuned ceiling trailed AITER from 8K KV tokens up, and from
4K for the largest-head and batch-64 shapes, and that gap motivated the kernel changes.
The split-rule constants then came from segment × tile × knob sweeps on both
architectures, all cold. gfx942 had a cold baseline against develop and AITER before its
lever ports, but no sweep of its existing lever space. Commit `9007caa2880` records a
separate limit: until then the 3D tuning space could not express what default dispatch
builds for decode (production sweeps never enabled the K/V swizzle or the page-id
window, and the segment catalog missed the split rule's counts); §3 describes what it
expresses now. So on gfx942 §2 rests on A/Bs against the previous kernel, not on a
sweep, and a sweep of the current space is an open item (§7).

## 6. Reproducing

Paths are relative to `dnn-providers/hip-kernel-provider/rocke/`; set the environment
up as in [BUILDING.md](../../../../BUILDING.md) and see
[TESTING.md](../../../../TESTING.md) for the tiers.

```bash
# Gates that need no GPU: relative-path guard, Python/C++ byte identity, the emitted-IR
# and arch-domain corpus gates, parity, pytest. The byte-identity gate runs every LLVM
# flavor the runner supports (the --flavor choices) unless --flavor <name> pins one.
python tools/run_checks.py --steps guard,gate,ir-validity,arch-domain,parity,pytest

# Representative IR goldens (prints "IR parity OK" with the flavors it checked).
PYTHONPATH=platform/python:library \
  python platform/tests/instances/rocke_ir_parity_harness.py \
    --check platform/tests/golden/rocke_representative_ir_sha256.json
```

The pytest stage picks up the new tests by itself, and the gate stage picks up the
emitter pairs; the GPU tests in the knob suites run on a gfx942 or gfx950 device and
skip elsewhere. `ir-validity` and `arch-domain` are stages of their own, which a
`--steps` list that omits them skips; the arch-domain gate flagged this series' kernels
while the series was in development. `--steps numeric` is a different lane (the
attention FMHA numeric harness `numeric_attention.py`, plus the gfx950 fp8 decode 3D
check), and it fails without a device when requested explicitly.

GPU numerics, on a gfx942 or gfx950 device. The C++-engine cases skip unless the
`rocke_engine` module is built (see BUILDING.md):

```bash
PYTHONPATH=platform/python:library python -m pytest \
    library/tests/test_attention_tiled_3d_kv_lds_swizzle.py \
    library/tests/test_attention_tiled_3d_wave_page_id_load.py \
    library/tests/test_strided_kv_decode_numeric.py \
    library/tests/test_attention_tiled_3d_running_max.py

# Forced-3D parity on gfx950 (RocKE lanes only; drop --skip-triton and set AITER_PATH
# to compare with AITER's Triton kernel as well).
(cd platform && PYTHONPATH=python:../library python -m \
    builders.gfx950.attention.prefill.parity_unified_attention --paths 3d --skip-triton)
```

The running-max test is the regression test of
[#14276](https://github.com/ROCm/rocm-libraries/pull/14276): its GPU tests are the
overflow cases that fix removed (each hit one of them before it), which is the hazard the
paragraph on replay and the overflow fixes below warns about. They are torch-free.

The PR description calls the harness the forced-3D parity harness; the series ran the
gfx950 one,
[`parity_unified_attention.py`](../../gfx950/attention/prefill/parity_unified_attention.py)
with `--paths 3d`. The command above runs its default set (27 scenarios, none of them a
`pr9233_*` one); §5 describes the set the series ran. It builds every problem with
`num_cus=120`, so its decode shapes get the split rule's choice for 120 CUs, not the
device's. The gfx942 parity harness
([`parity_unified_attention.py`](../../gfx942/attention/prefill/parity_unified_attention.py))
builds and launches only the 2D tiled kernel (explicit spec, `block_size` 64), so it
never runs the 3D path, and no forced-3D parity ran on gfx942 (§5).

For the A/B, build one checkout per arm. The series-wide comparison's own pair was
`5b2dd750b61` (the series with develop `a76c2460399` merged in, a commit of #13492 on
`refs/pull/13492/head`) against `a76c2460399`. To replay the series as it merged, check
out the squash commit `b53003d7bb6` and its first parent `503f5c82876`; that pair also
holds the twelve later commits of the PR (tests, docs, tuning ids, the tuning-space
change `9007caa2880` and two merges of develop). Both pairs predate the fixes below.
To replay a per-lever A/B, compare the series commit that introduced the lever with its
parent; the series' commits are on `refs/pull/13492/head`, in the order of the PR, and
the squash merge does not keep them. That measures the lever in the kernel as it was at
that commit, not in the final kernel. A lever with a spec field (segment count, tile,
swizzle, page-id window, cache policy) can instead be compared on one build, both specs
in one process on one stream, which gives the same-session ratios
[KERNEL_AUTHORING.md](../../../../KERNEL_AUTHORING.md#d-perf-optimization-no-new-feature)
asks for (on gfx950, `decode_table_sweep` times a shape's tuning variants in one
process).

Each arm must import its own tree. With one editable environment (BUILDING.md),
`python -m benchmarks…` run from either checkout imports `rocke`, `kernels`, `builders`,
`dispatch` and `benchmarks` from the environment's tree, so both arms would run the same
code. Use each checkout's own environment (`build/rocke-pyenv/bin/python`), or run from
its `rocke/` directory with `PYTHONPATH=platform/python:library` as the commands below
do, and check with `python -c 'import kernels; print(kernels.__file__)'`.

**Replay and the overflow fixes.** The series' commits predate later fixes to the 3D
kernels, among them [#14276](https://github.com/ROCm/rocm-libraries/pull/14276) (the
segment kernels' softmax update now carries the running maximum; with a tile-local
maximum, a tile that scores far below an earlier tile of its segment overflowed fp32 and
the row came out NaN) and [#14216](https://github.com/ROCm/rocm-libraries/pull/14216)
(the gfx942 K/V cache bound). The outcomes in §2 were measured before them, and a
commit-versus-parent A/B runs the old kernels on both sides. Replay each arm in its own
detached worktree of the commit (`git worktree add --detach <dir> <commit>`) and leave
the current tree alone. Do not cherry-pick or rebase the series' commits, and do not copy
their kernel files, onto a tree that has these fixes: the two transposed-QK commits,
`c686a6c5b7b` (gfx950) and `9aa3054bf77` (gfx942), wrote the update lines that #14276
rewrote, and `745416275ff` (native exp2) edits them too, in both engines, so resolving a
conflict (or copying a file) in the series' favor restores the overflow. A lever that has
a spec field needs no replay: compare it on one build of current develop, where both arms
have the fixes.

The in-repo decode harnesses can replay the comparison; they are not the harness the
series used, and the series' own shape corpus (prefill and decode attention shapes) is
not in the repo:

```bash
export PYTHONPATH=platform/python:library  # this checkout's own tree, for each arm
python -m benchmarks.gfx950.attention.decode.benchmark_decode_live \
    --shapes library/benchmarks/gfx950/attention/decode/decode_shapes.json \
    --num-cus-sweep <device CU count> --warmup 75 --iterations 250
python -m benchmarks.gfx942.attention.decode.benchmark_decode_live \
    --shapes library/benchmarks/gfx942/attention/decode/decode_shapes.json \
    --num-cus-sweep <device CU count> --warmup 75 --iterations 250
python -m benchmarks.gfx950.attention.decode.decode_table_sweep --list-only
```

[`benchmark_decode_live`](../../../benchmarks/gfx950/attention/decode/benchmark_decode_live.py)
compares the DSL kernel with AITER's Triton `unified_attention` and sweeps `num_cus` per
shape (see its module docstring; the
[gfx942](../../../benchmarks/gfx942/attention/decode/benchmark_decode_live.py) sibling
differs in its defaults).
[`decode_table_sweep`](../../../benchmarks/gfx950/attention/decode/decode_table_sweep.py)
walks every decode candidate the dispatcher registry offers. Follow the runbook's
[benchmark hygiene](../../../../platform/dsl_docs/optimization/optimization_runbook.md#23-benchmark-hygiene)
and note what this harness does and does not do:

- Alternate the arms (A, B, A, B, …); discard each arm's first run and keep at least 3
  more; compare medians and report the spread, (max − min) / median. Run the same
  harness file on both arms, and pin the clocks if you suspect throttling (runbook
  §2.3).
- Pass `--num-cus-sweep` with the count default dispatch resolves, alone: the device's
  CU count, but at least 120 on gfx950. Its default sweep (`30 60 80 120 152 304` on
  gfx950, `30 60 80 120` on gfx942) need not include it, and the harness's headline
  column takes the best over the sweep, which is a tuned arm; the series compared
  default dispatch.
- It reports the mean of back-to-back prepared 3D launches between two HIP events
  (`time_launches`), not the series' two timings: it leaves out the per-call dispatch
  that the wall time includes, so the host-launch caveat of §4 need not show, yet on
  host-bound shapes it is not pure device time either. On gfx942 the prepared launch is
  a HIP-graph replay for the shapes the
  [graph heuristic](../../gfx942/attention/ALGORITHM.md#9-host-overhead-and-the-graph-heuristic)
  picks; on gfx950 it is eager unless `HIPDNN_GFX950_3D_GRAPH=1`.
- It does not check outputs, and a failed run shows only as `ERR` in its column, without
  the error. The GPU numerics above do not run the timed shapes, and the new knob suites
  do not exist at `503f5c82876` or `a76c2460399`, so they cannot gate a baseline arm. On
  gfx950, `decode_table_sweep` checks each candidate against a reference (tolerance
  2e-2) before timing it by HIP-graph replay, over batched head-size-128 decode shapes
  (`--batch` overrides); the file is identical at `503f5c82876` and `b53003d7bb6`.
  gfx942 has no decode-specific harness that checks outputs, but the arch-generic
  [`attention_combo_sweep`](../../../benchmarks/common/attention_combo_sweep.py) checks
  each registry candidate against a reference (default tolerance 3e-2) before timing it,
  one shape grid per run (`--arch gfx942 --seqlen-q 1`); the file is identical at
  `503f5c82876` and `b53003d7bb6`. For `benchmark_decode_live` on either architecture,
  check each timed shape against a reference on both arms before timing, or publish no
  relative result from the run ([platform/AGENTS.md](../../../../platform/AGENTS.md)
  Compliance).
- Run both arms against the same compiler, the COMGR library the process loads, and
  record its LLVM version (`loaded_compiler_info().describe()` from
  `rocke.runtime.comgr`) with the ROCm version, the device (gfx id and CU count), both
  commits, every `HIPDNN_*` and `ROCKE_*` variable set, and per shape the correctness
  status and max error (runbook §2.4 and §15.1). Leave `ROCKE_LLVM_FLAVOR` unset so that
  each arm takes its flavor from that compiler; a set value changes only the emitted IR,
  not the compiler: an `llvm23` flavor on an LLVM 22 COMGR is still compiled by LLVM 22
  (only a flavor with a different datalayout, `llvm20` against LLVM 22 or 23, is
  rejected). The process loads `ROCKE_COMGR_LIB` if set, else the COMGR bundled with
  torch when torch is already imported (unless it is older than the newest ROCm
  install), else the newest ROCm install, so pin `ROCKE_COMGR_LIB` to make both arms use
  the same one. The compiler-dependent levers of §4 can flip between compiler versions:
  the series-wide comparison used an LLVM 23 COMGR and most per-lever A/Bs the LLVM 22
  one bundled with torch (§4).
- It builds each shape's inputs once and reuses them for every timed call, so it
  measures the warm case (§4). A non-temporal comparison needs K/V that is not
  cache-resident between calls, which no in-repo harness provides: allocate distinct K/V
  copies that together are larger than the caches (the L2 of every XCD plus the
  memory-side last-level cache), prepare one launch per copy and rotate through them
  between calls (the runbook's rotating buffers,
  [§6.6](../../../../platform/dsl_docs/optimization/optimization_runbook.md#66-caches)),
  so that a call does not find an earlier call's K/V in the caches.
- `decode_shapes.json` holds only single-sequence shapes (batch 1, head sizes 128 and
  256, `block_size` 16), while most outcomes in §2 are about batched decode. The gfx950
  harness covers them with `--batch <N>`, which overrides every shape's batch. The gfx942
  harness has no `--batch`: pass
  [`decode_batch_sweep_shapes.json`](../../../benchmarks/gfx942/attention/decode/decode_batch_sweep_shapes.json)
  to `--shapes` (bf16, head sizes 64 and 128, `block_size` 16, batches 8 to 256), or
  change the `batch` of every shape in a copy of `decode_shapes.json` (each shape sets
  its own `batch` of 1, which a model-level `batch` does not override); that is how to
  batch head size 256 there.

## 7. Open items

- **GQA ratios that do not divide 16.** On gfx942 and gfx950, a decode problem whose
  query heads per KV head is outside 1…16 or does not divide `BLOCK_M = 16` (for
  example 5) is rejected by `supports_tiled_3d`. `backend="3d"` raises with the gate's
  reason; `backend="auto"` falls back, without saying so, to the tiled 2D kernel, where
  one workgroup per sequence and KV head walks the whole KV range, and to the scalar
  kernels only where 2D rejects the problem too (a ratio above 16, or another 2D limit,
  for example gfx942 head size 256 with 64-token pages exceeding LDS). `select_path()`
  still labels these shapes "3d". None of the levers above reach them, as before the
  series. The open
  [#14367](https://github.com/ROCm/rocm-libraries/pull/14367) adds split-KV support for
  these ratios (with padded rows); when it lands, this item and the matching entry in
  [`limitations.md`](../../../../platform/dsl_docs/runtime/limitations.md) need updating.
- **Reduce kernel.** The series changes only its `exp2`. The follow-up is the slow zone
  of §4 plus the reduce's share of small-batch decode time, not only a faster reduce
  kernel.
- **Host launch overhead per call** (§4) is a follow-up. The 3D binary cache key is
  computed on every launch, and it now also recomputes the five derived settings the
  series added (decode grid, cache policy, gfx950 tile, swizzle and page-id window),
  besides the segment count it already recomputed. Caching the derived 3D dispatch per
  problem is the obvious first step.
- **Pre-PV LDS-DMA drain.** The gfx950 segment kernel still waits for all in-flight K/V
  copies (`vmcnt(0)`) before the PV-side LDS reads. Two causes stack: the partial wait
  is followed by `sync()`, which lowers to a `vmcnt(0)` wait plus a barrier, and LLVM
  cannot tell the pooled LDS buffers apart, so it must assume a read can alias any copy
  in flight. Prototypes on an earlier commit removed both, with an LDS-only barrier plus
  separate LDS globals or blanket alias scopes (an LLVM 23 async-mark variant measured
  the same), and gained clearly on single-sequence decode on top of 64-token tiles. They also moved the batched optimum to about 4 workgroups per CU then; the
  page-id window has since done that without them, so their batched gain on today's
  kernel is unmeasured. The opt-in `lds_alias_scopes` kernel attribute and
  `sync_lds_only` now exist, so this may need no platform change; the
  [gfx942 MLA prefill example](../../../examples/gfx942/mla_prefill/README.md) (levers
  9, 47 and 48) applies the same two.
- **Multi-page tiles and the page-id KV loader** are to be reconciled with the open
  decoupled-KV-pages work
  ([#13128](https://github.com/ROCm/rocm-libraries/pull/13128)), which reworks the same
  3D KV loading.
- **Compiler changes** can move the compiler-dependent levers in §4; re-measure the
  gfx942 wide feed and the head-size-256 swizzle pairing.
- **Non-resident K/V.** Re-check the decode grid, exp2, the one V slot and page ids a
  tile ahead on K/V that is not cache-resident between calls (§4); every other lever
  with an A/B of its own was measured that way. These four are structural and unlikely
  to flip, so this item could also be dropped. No in-repo harness has a cold-K/V mode.
- **Levers without an A/B of their own, and untested ranges** (§8). The gfx942 swizzle
  and page-id window at head size 64 have no A/B of their own; A/B each alone. The
  gfx950 64-token tile has not been separated from the 64 segments: A/B it against
  one-page tiles at the same segment count, with the swizzle off in both arms. Both are
  spec fields that an explicit tuning spec sets on the current tree. The split rule
  is also untested for non-power-of-two $w$ (where it returns counts in the reduce slow
  zone of §4, so check the reduce's ISA first), ragged batches and KV beyond 64K tokens.
- **Step 0 sweep** (§5). Sweep the 3D lever space on the decode cohorts and compare the
  swept best with the dispatched config, which the tuning space can now express (§3).
  Step 0 was done on gfx950 before the series but not recorded, so this repeats it on
  the current kernel there; gfx942 has no sweep at all.
  On gfx950, `decode_table_sweep --sweep-level full --tuning-sample 0` walks every legal
  tuning spec and checks each against a reference before timing it; tuning specs use the
  architecture's full-part CU count when `num_cus` is unset (§3), so run it on a part
  with that count. On gfx942, the arch-generic `attention_combo_sweep` does the same, one
  shape grid per run, with `--arch gfx942 --seqlen-q 1` and
  `--sweep-level full --tuning-sample 0`.
- **Series shape corpus.** The shape corpus of the series-wide comparison is not in the
  repo (§6).

## 8. Decision

Keep every lever in §2 on its default path as listed. The swizzle and the multi-page
tile path are conditional by construction, and the invariant hoist is removed. Almost
every lever has an A/B of its own (§2). The exceptions are the gfx942 swizzle and
page-id window at head size 64, which are kept on the series-wide comparison alone, and
the gfx950 64-token tile, which was measured together with 64 segments but not
separated from them. That comparison ran default dispatch on both arms, so each lever
ran only on the corpus shapes that meet its conditions, and which shapes those were is
not recorded; no shape on the 3D path was slower in device time. The losses of §4
(strided K/V at 64-token tiles under LLVM 22, the small split-rule losses at short KV)
and the reduce-kernel slow zone are documented, not fixed. §7 lists the open A/Bs, the
Step 0 sweep and the reduce redesign. Revisit when the decoupled-KV-pages work lands, on
a compiler upgrade, if the GQA gate is widened, and when those A/Bs, the sweep or the
redesign run.
