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
the description of #13492 (the gfx950 head-size-256 compiler note in §4 from the
docstring of `_gfx950_3d_decode_ctas_per_segment`) and were not re-measured for this
file. The algorithm is written up in the
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
revision* means it overlaps open work (§7). *No separate outcome recorded* means the
lever has no A/B of its own: either it was measured only together with other levers,
in a joint A/B that the row or the split-rule history names, or it had no A/B at all
and only the series-wide comparison covers it.

### 2.1 Always on

These have no spec field. Derivations: [gfx950 §5.1](../../gfx950/attention/ALGORITHM.md#51-inside-the-segment-kernel)
and [gfx942 §8](../../gfx942/attention/ALGORITHM.md#8-the-decode-path-3d-split-kv).

| Lever | Arch | Mechanism | Outcome |
|---|---|---|---|
| Transposed QK, register-resident $P$ | both | $S^{\top} = K Q^{\top}$: one $(m, \ell)$ per lane, no $P$ tile in LDS | Kept. gfx950: faster on long-KV, single-sequence and head-size-256 decode, neutral on short batched decode. gfx942: faster on batched, single-sequence and head-size-256 decode. Outputs match the previous kernel within rounding. |
| $Q^{\top}$ read once | both | loaded into registers before the KV loop instead of read from LDS every tile | Kept. gfx950: faster on long-KV and batched decode, neutral on short single-sequence decode; costs a few VGPRs and leaves occupancy unchanged. gfx942: landed with the transposed-QK change; no separate outcome recorded. |
| One V LDS slot | both | $V(i)$ is consumed in the iteration that loads it, so the second slot never held a live tile | Kept. gfx950: raises occupancy at the default tile, with broad gains on decode shapes. gfx942: head size 256 at block size 64 now builds (it had exceeded the 64 KiB LDS limit while reported as supported); neutral at the default decode tile. |
| Native `exp2` | both | `llvm.amdgcn.exp2` (one `v_exp_f32`) in the segment softmax and the reduce kernel | Kept. No regression, small gains on kernel-bound decode shapes, outputs matched on the measured shapes. |
| `16x16x32` PV | gfx950 | two 16-token blocks per MFMA on tiles of 32 or more tokens | Kept. Half as many PV MFMAs on those tiles; 16-token tiles unchanged. No separate timing outcome recorded. |
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
`num_segments=32` and `tile=16`, the second worked example of gfx950 §6.1. The
kernel-name tokens in §3 show which levers are on.

Outcomes by lever (the conditions are in the table above):

| Lever | Arch | Outcome |
|---|---|---|
| Decode grid, `use_decode_grid` | both | Kept. Large gains at 16 query heads per KV head, neutral elsewhere. |
| Streaming K/V, `kv_cache_policy="stream"` | both | Kept. gfx950: faster on batched decode, neutral on single-sequence decode. gfx942: neutral to slightly faster. Both measured with K/V that is not cache-resident between calls (§4). |
| K/V LDS swizzle, `use_kv_lds_swizzle` | both | gfx950: conditional. Neutral at page-sized tiles, faster at multi-page tiles, most on single-sequence and head-size-256 decode; that A/B set the multi-page tiles by explicit spec, since default dispatch then selected none, and default dispatch now gives them only to head size 128 (table above). gfx942: kept for all-decode problems (conditions in the table above). Head size 64 and 128: no separate outcome recorded. Head size 256: alone it is slower under some compilers, hence the pairing with the page-id window; the pair has no separate outcome recorded. |
| Wave page-id window, `use_wave_page_id_load` | both | Kept. gfx950: one A/B, together with the retune to about 4 workgroups per CU (below): faster on batched decode, unchanged on single-sequence decode and outside the rule's scope. gfx942: no separate outcome recorded. |
| Multi-page KV tiles | gfx950 | Conditional, under revision. No separate outcome recorded: measured only together with 64 segments and the swizzle (the single-sequence tune and the first split rule, below). Overlaps the open decoupled-KV-pages work (§7). |
| Async K/V copies at head size 256 | gfx942 | Kept. A large win with LLVM 23, where the wide feed's load/store pairs serialize at head size 256; neutral with LLVM 22. |
| Decode split rule | both | Kept. History and attribution below. |
| Invariant hoist, `use_invariant_hoist` | both | Removed. The kernels compute the query-row invariants once per lane before the loop, so the field selected nothing. The environment toggle, dispatch helper, tuning axis and name token are gone; the spec field stays, inert, so saved specs still load. |

Decode split rule history:

- gfx950, single-sequence tune (one sequence, head size 128, paged K/V, KV of 4K tokens
  or more): 64-token tiles, 64 segments and the swizzle, in one A/B. Faster on every
  covered single-sequence shape, unchanged elsewhere.
- gfx950, first version, which generalized it (about 8 workgroups per CU, rounded down
  to a power of two between 4 and 64 segments; 64-token tiles when even 64 segments
  leave the device underfilled): faster on large batches and on small batches with few
  KV heads, and unchanged where the previous split already matched.
- gfx950, retune to about 4 workgroups per CU (still a power of two, 4 to 64): measured
  only together with the page-id window (above). The later extension recorded no
  separate outcome. It added KV below 4K tokens, head size 256 and strided K/V, and it
  changed the counts to what ships: $\lfloor 4\,\text{CUs}/w \rfloor$ without the
  power-of-two rounding, at least 2 and at most 64 (128 from 8K KV tokens), and with few
  workgroups at most one segment per 64-token tile. The series-wide comparison covers
  the final rule.
- gfx942: scoped to where it measured well (enough workgroups per segment, or KV below
  8K tokens); the previous counts stay elsewhere.

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

- **K/V residency.** Real decode reads each K/V byte once per step. The A/Bs of the
  non-temporal hint on both architectures, the gfx942 transposed QK and the gfx942
  head-size-256 copies used K/V that was not cache-resident between calls. The other
  per-lever A/Bs do not record it, and the series-wide comparison replayed the same
  inputs, so the K/V could stay warm. That is the conservative direction for the
  non-temporal hint (replaying a small K/V makes it look slower, because the K/V no
  longer stays in the caches). For the other levers the direction is not known: the
  split rule, the decode grid, both page-id levers, the swizzle, the one V slot, exp2,
  the gfx950 transposed QK and $Q^{\top}$ read once, and the levers with no A/B of their
  own were judged on warm or unrecorded K/V. For the split rule in particular, shorter
  load latency can make fewer workgroups per CU look sufficient (§7).
- **Host launch.** A few small decode shapes are marginally slower once the host launch
  is included. Their kernels are faster on the device, but each call is bound by host
  launch overhead, which the series raises slightly. No measured shape on the 3D path
  was slower in device time. Reducing the per-call host overhead is a follow-up.
- **Compiler dependence.** The gfx942 wide K/V feed overlaps its loads with its stores
  only if the compiler schedules them that way (LLVM 22 does, LLVM 23 does not at head
  size 256), and the gfx942 head-size-256 swizzle alone is slower under some compilers.
  The gfx950 head-size-256 split rule is limited to grids with enough workgroups for
  page-sized tiles because its few-workgroup splits measured faster under one compiler
  and slower under another. Re-measure these on a compiler change.
- **Coverage.**
  - The series-wide comparison used one device per architecture and one compiler build,
    which the PR names only as the benchmark's ROCm build, so its LLVM version is not
    recorded; the gfx942 head-size-256 loader was also compared across LLVM 22 and 23
    (§2.2). A GQA ratio that does not divide 16 runs the scalar kernels in both arms
    (§7).
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
description, where the relative results live; none was re-measured for this file.

**Correctness.** The PR description says every shape's result was checked against a
reference before it was timed. The gates it lists:

- GPU numerics against an FP32 reference, reported on gfx942 and gfx950 with both
  engines: the new knob suites
  ([K/V swizzle](../../../tests/test_attention_tiled_3d_kv_lds_swizzle.py),
  [wave page-id window](../../../tests/test_attention_tiled_3d_wave_page_id_load.py);
  their GPU tests run on a gfx942 or gfx950 device, each on its own architecture, and
  skip elsewhere) with reversed and poisoned page tables, ragged and empty sequences
  and multi-page tiles, the
  [strided-KV decode numerics](../../../tests/test_strided_kv_decode_numeric.py), and
  the forced-3D parity harness (§6). The knob and strided suites run fp16 and bf16 and
  compare with absolute/relative tolerances of 0.003/0.005 (fp16) and 0.01/0.02 (bf16).
  The PR description gives no dtype or tolerance for the check that gated the timed
  runs, so the outcome words in §2 are the series author's statements; recording the
  tolerance is an open item (§7).
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

**Performance.** The per-lever A/Bs ran over decode shapes (the exp2 one over various
shapes routed to the 3D path), most of them interleaved; the commit messages do not say
how each arm was built (the commit against its parent, or a spec field set both ways on
one build). The series-wide comparison used default dispatch on both sides (no tuned
specs) against develop at `a76c2460399`, over a corpus of LLM attention shapes (prefill
and decode; 251 on gfx950, 248 on gfx942) on one gfx950 and one gfx942 device. Each
shape ran 5 rounds of 75 warm-up and 250 timed calls, reported as the median, with two
timings: wall time per call including the host launch, and device time from HIP-graph
replay. The cohorts on the 3D path were:

| Cohort | Shapes |
|---|---|
| Decode, 1 sequence | 14 |
| Decode, 2–16 sequences | 25 |
| Decode, 32 or more sequences | 16 |
| Decode, head size 256 | 7 |
| Prefill routed to 3D | 1 on gfx950, 3 on gfx942 |
| Decode with query heads per KV head not dividing 16 (scalar fallback on both sides) | 15 |

**Step 0.** The series' commit messages and PR description record no
[Step 0](../../../../platform/dsl_docs/optimization/optimization_runbook.md#step-0--exhaust-the-existing-lever-space-first-do-this-before-touching-the-algorithm-or-structure)
sweep of the existing lever space before the kernel-body rewrites of §2.1, and no
reason for skipping one. Commit `9007caa2880` records a separate limit: until then the
3D tuning space could not express what default dispatch builds for decode (production
sweeps never enabled the K/V swizzle or the page-id window, and the segment catalog
missed the split rule's counts); §3 describes what it expresses now. So §2 rests on
A/Bs against the previous kernel, not on a sweep; a sweep of the current space is an
open item (§7).

## 6. Reproducing

Paths are relative to `dnn-providers/hip-kernel-provider/rocke/`; set the environment
up as in [BUILDING.md](../../../../BUILDING.md) and see
[TESTING.md](../../../../TESTING.md) for the tiers.

```bash
# Gates that need no GPU: relative-path guard, Python/C++ byte identity, parity, pytest.
# The byte-identity gate runs every LLVM flavor the runner supports (the --flavor
# choices) unless --flavor <name> pins one of them.
python tools/run_checks.py --steps guard,gate,parity,pytest

# Representative IR goldens (prints "IR parity OK" with the flavors it checked).
PYTHONPATH=platform/python:library \
  python platform/tests/instances/rocke_ir_parity_harness.py \
    --check platform/tests/golden/rocke_representative_ir_sha256.json
```

The pytest stage picks up the new tests by itself, and the gate stage picks up the
emitter pairs; the GPU tests in the knob suites run on a gfx942 or gfx950 device and
skip elsewhere. `--steps numeric` is a different lane (the attention FMHA numeric
harness `numeric_attention.py`, plus the gfx950 fp8 decode 3D check), and it fails
without a device when requested explicitly.

GPU numerics, on a gfx942 or gfx950 device. The C++-engine cases skip unless the
`rocke_engine` module is built (see BUILDING.md):

```bash
PYTHONPATH=platform/python:library python -m pytest \
    library/tests/test_attention_tiled_3d_kv_lds_swizzle.py \
    library/tests/test_attention_tiled_3d_wave_page_id_load.py \
    library/tests/test_strided_kv_decode_numeric.py

# Forced-3D parity on gfx950 (RocKE lanes only; drop --skip-triton and set AITER_PATH
# to compare with AITER's Triton kernel as well).
(cd platform && PYTHONPATH=python:../library python -m \
    builders.gfx950.attention.prefill.parity_unified_attention --paths 3d --skip-triton)
```

The PR description does not name the harness it calls the forced-3D parity harness. On
gfx950 the in-repo harness that forces the 3D path is
[`parity_unified_attention.py`](../../gfx950/attention/prefill/parity_unified_attention.py)
with `--paths 3d`. It builds every problem with `num_cus=120`, so its decode shapes get
the split rule's choice for 120 CUs, not the device's. The gfx942 parity harness
([`parity_unified_attention.py`](../../gfx942/attention/prefill/parity_unified_attention.py))
builds and launches only the 2D tiled kernel (explicit spec, `block_size` 64), so it
never runs the 3D path; which harness the series ran on gfx942 is not recorded (§7).

For the A/B, build one checkout per arm. The series' own comparison used develop at
`a76c2460399` as the baseline; to replay the series as it merged, check out the squash
commit `b53003d7bb6` and its first parent `503f5c82876` (both predate the fixes below).
To replay a per-lever A/B, compare the series commit that introduced the lever with its
parent; the series' commits are on `refs/pull/13492/head`, in the order of the PR, and
the squash merge does not keep them. That measures the lever in the kernel as it was at
that commit, not in the final kernel. A lever with a spec field (segment count, tile,
swizzle, page-id window, cache policy) can instead be compared on one build, both specs
in one process on one stream, which gives the same-session ratios
[KERNEL_AUTHORING.md](../../../../KERNEL_AUTHORING.md#d-perf-optimization-no-new-feature)
asks for (on gfx950, `decode_table_sweep` times a shape's tuning variants in one
process).

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
  not the compiler. The compiler-dependent levers of §4 can flip between compiler
  versions, and the PR does not record which one the series used.
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
  example 5) is rejected by `supports_tiled_3d` and runs the scalar kernels, as before
  the series. None of the levers above reach it.
- **Reduce kernel.** The series changes only its `exp2`; a faster reduce kernel is a
  follow-up.
- **Host launch overhead per call** (§4) is a follow-up.
- **Multi-page tiles and the page-id KV loader** are to be reconciled with the open
  decoupled-KV-pages work
  ([#13128](https://github.com/ROCm/rocm-libraries/pull/13128)), which reworks the same
  3D KV loading.
- **Compiler changes** can move the compiler-dependent levers in §4; re-measure the
  gfx942 wide feed and the head-size-256 swizzle pairing.
- **Non-resident K/V.** Re-check the split rule, and ideally the decode grid, the
  page-id window and the swizzle, on K/V that is not cache-resident between calls (§4).
  No in-repo harness has a cold-K/V mode.
- **Isolating levers measured only together** (§8). A/B each alone: the 64-token tile
  against one-page tiles at the same segment count, with the swizzle off in both arms;
  about 4 against about 8 workgroups per CU with the page-id window on; the shipped
  segment counts (no power-of-two rounding, 2 to 128) against the retune's
  power-of-two counts; the gfx950 split rule's later extensions (KV below 4K tokens,
  head size 256, strided K/V) against the previous counts; the gfx942 swizzle and
  page-id window at head size 64 and 128, and their pairing at 256 against neither; the
  gfx942 $Q^{\top}$ read once; the gfx950 `16x16x32` PV. Most are spec fields or
  dispatch rules that an explicit tuning spec sets on the current tree; the two
  kernel-body items (the $Q^{\top}$ read and the PV) have no field, so compare their
  commits with their parents (§6).
- **Step 0 sweep** (§5). Sweep the 3D lever space on the decode cohorts and compare the
  swept best with the dispatched config, which the tuning space can now express (§3).
  On gfx950, `decode_table_sweep --sweep-level full --tuning-sample 0` walks every legal
  tuning spec and checks each against a reference before timing it; tuning specs use the
  architecture's full-part CU count when `num_cus` is unset (§3), so run it on a part
  with that count. On gfx942, the arch-generic `attention_combo_sweep` does the same, one
  shape grid per run, with `--arch gfx942 --seqlen-q 1` and
  `--sweep-level full --tuning-sample 0`.
- **Unrecorded conditions.** The input dtypes and tolerance of the correctness check in
  the timed runs, the K/V residency of the per-lever A/Bs that do not state it, the LLVM
  version of the compiler in the series-wide comparison and which harness the PR means
  by "the forced-3D parity harness" (§6) are not in the PR description; ask the series'
  author and record them here. The series' shape corpus is not in the repo either (§6).

## 8. Decision

Keep every lever in §2 on its default path as listed. The swizzle and the multi-page
tile path are conditional by construction, and the invariant hoist is removed. Several
levers were never measured alone. Three are kept on joint A/Bs: the multi-page tiles
(with 64 segments and the swizzle), the retune to about 4 workgroups per CU (with the
page-id window) and the gfx942 $Q^{\top}$ read once (with the transposed QK). The others
have no A/B of their own and are kept on the series-wide comparison alone: the later
split-rule extension (its scope and the shipped segment counts), the gfx942 swizzle and
page-id window at head size 64 and 128 and their pairing at 256, and the gfx950
`16x16x32` PV. That comparison ran default dispatch on both arms, so each lever ran only
on the corpus shapes that meet its conditions, and which shapes those were is not
recorded; no shape on the 3D path was slower in device time. §7 lists the isolating A/Bs
and the Step 0 sweep. Revisit when the decoupled-KV-pages work lands, on a compiler
upgrade, if the GQA gate is widened, and when those A/Bs or the sweep run.
