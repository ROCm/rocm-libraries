# Attention 3D split-KV decode on gfx942 and gfx950 — case study

Measured scope: the split-KV ("3D") segment and reduce kernels for single-token
decode on gfx942 and gfx950, before and after the optimization series merged as
[ROCm/rocm-libraries#13492](https://github.com/ROCm/rocm-libraries/pull/13492). This
file records the mechanisms, which levers ship by default and which are conditional,
the honest losses and caveats, and how to replay the checks and the A/B. It publishes
no speed numbers: every outcome below is a qualitative statement bounded to the
measured configurations (the per-lever interleaved A/Bs over decode shapes that the
author ran during development, the exp2 one over all shapes routed to the 3D path,
and the series-wide comparison with default dispatch on both arms, §5), not a general
claim. The algorithm is written up in the
[gfx950](../../gfx950/attention/ALGORITHM.md#51-inside-the-segment-kernel) and
[gfx942](../../gfx942/attention/ALGORITHM.md#8-the-decode-path-3d-split-kv)
`ALGORITHM.md` files; the fields are in the
[Knob reference](../README.md#knob-reference) and the runbook's
[§12.1.K2](../../../../platform/dsl_docs/optimization/optimization_runbook.md#121k2-attention-3d-split-kv-levers-ua-3d-specific).

## Contents

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
  the XCDs;
- a $P$ tile that went through LDS between the QK and PV MFMAs;
- a page-table load per KV tile whose wait drained the K/V copies already in flight;
- K/V, each byte read once per step, cached like reusable data;
- a segment count from a formula built for prefill, which distorted decode (q-blocks
  counted as for prefill, powers of two, at least 16 segments).

## 2. Levers and outcomes

Outcome words compare against the previous kernel on the same shapes: *faster*,
*neutral* or *slower* at the measured configurations. *Kept* means the lever ships on
its default path; *conditional* means it is on only where it measured well.

### 2.1 Always on

These have no spec field. Derivations: [gfx950 §5.1](../../gfx950/attention/ALGORITHM.md#51-inside-the-segment-kernel)
and [gfx942 §8](../../gfx942/attention/ALGORITHM.md#8-the-decode-path-3d-split-kv).

| Lever | Arch | Mechanism | Outcome |
|---|---|---|---|
| Transposed QK, register-resident $P$ | both | $S^{\top} = K Q^{\top}$: one $(m, \ell)$ per lane, no $P$ tile in LDS | Kept. gfx950: faster on long-KV, single-sequence and head-size-256 decode, neutral on short batched decode. gfx942: faster on batched, single-sequence and head-size-256 decode. Outputs match the previous kernel within rounding. |
| $Q^{\top}$ read once | both | loaded into registers before the KV loop instead of read from LDS every tile | Kept. gfx950: faster on long-KV and batched decode, neutral on short single-sequence decode; costs a few VGPRs and leaves occupancy unchanged. |
| One V LDS slot | both | $V(i)$ is consumed in the iteration that loads it, so the second slot never held a live tile | Kept. gfx950: raises occupancy at the default tile. gfx942: head size 256 at block size 64 now builds (it had exceeded the 64 KiB LDS limit while reported as supported); neutral at the default decode tile. |
| Native `exp2` | both | `llvm.amdgcn.exp2` (one `v_exp_f32`) in the segment softmax and the reduce kernel | Kept. No regression, small gains on kernel-bound decode shapes, outputs unchanged. |
| `16x16x32` PV | gfx950 | two 16-token blocks per MFMA on tiles of 32 or more tokens | Kept. Half as many PV MFMAs on those tiles; 16-token tiles unchanged. |
| Page ids a tile ahead | gfx950 | for paged fp16/bf16 K/V the next tile's page id is read during the current tile and passed as the copy's `soffset` (or folded into a 64-bit buffer base for caches above 2 GiB), instead of a block-table read issued with the K/V copies, whose wait drained the copies already in flight | Kept, under revision. Neutral to positive on decode shapes, largest on kernel-bound single-sequence and large-page shapes. Overlaps the open decoupled-KV-pages work (§7). |

### 2.2 Spec fields and dispatch rules

| Lever | Arch | Default dispatch | Outcome |
|---|---|---|---|
| Decode grid, `use_decode_grid` | both | Derived: on whenever every sequence decodes, whatever the K/V dtype, window or `target_ctas` | Kept. Large gains at 16 query heads per KV head, neutral elsewhere. |
| Non-temporal K/V, `kv_cache_policy="stream"` | both | On for all-decode fp16/bf16 K/V; fp8 K/V and mixed batches keep `"all"` | Kept. gfx950: faster on batched decode, neutral on single-sequence decode. gfx942: neutral to slightly faster. Both measured with K/V that is not cache-resident between calls (§4). |
| K/V LDS swizzle, `use_kv_lds_swizzle` | both | gfx950: with the 64-token tile, paged K/V only. gfx942: all-decode at head size 64 and 128 on the wide feed, so it follows `HIPDNN_GFX942_3D_WKV`; at 256 only together with the page-id window. Strided K/V: off | Conditional. gfx950: neutral at page-sized tiles, faster at multi-page tiles in the A/B (largest on single-sequence decode; the head-size-256 gain came from explicit specs, since default dispatch never gives head size 256 a multi-page tile). gfx942 head size 256: alone it is slower under some compilers, hence the pairing. |
| Wave page-id window, `use_wave_page_id_load` | both | gfx950: on in the split rule's scope, paged K/V only. gfx942: with the swizzle, for page-sized tiles | Kept. gfx950: faster on batched decode, unchanged on single-sequence decode and outside the rule's scope; it is what lets the split rule aim at about 4 workgroups per CU instead of about 8. |
| Multi-page KV tiles | gfx950 | Only the small-grid decode case: 64-token tiles (strided K/V without the swizzle) | Conditional, under revision. No separate outcome was recorded; see the swizzle row. Overlaps the open decoupled-KV-pages work (§7). |
| Async K/V copies at head size 256 | gfx942 | Head size above 128 | Kept. A large win with LLVM 23, where the wide feed's load/store pairs serialize at head size 256; neutral with LLVM 22. |
| Decode split rule | both | In scope: all-decode, fp16/bf16 K/V, no sliding window, no explicit `target_ctas` (details in the `ALGORITHM.md` files) | Kept. gfx950: the first version of the rule (about 8 workgroups per CU, head size 128, paged K/V, KV of 4K tokens or more) was faster on large batches and on small batches with few KV heads, and unchanged where the previous split already matched; its single-sequence long-KV branch (64-token tiles) was faster on every covered shape. The retune to about 4 workgroups per CU and the later extensions (KV below 4K tokens, head size 256, strided K/V) recorded no separate outcome; the series-wide comparison covers the final rule. gfx942: scoped to where it measured well (enough workgroups per segment, or KV below 8K tokens). |
| Invariant hoist, `use_invariant_hoist` | both | Removed | The kernels compute the query-row invariants once per lane before the loop, so the field selected nothing. The environment toggle, dispatch helper, tuning axis and name token are gone; the spec field stays, inert, so saved specs still load. |

## 3. Compatibility

- **Tuning ids.** `kv_cache_policy`, `use_decode_grid`, `use_kv_lds_swizzle` and
  `use_wave_page_id_load` are new 3D spec fields and part of the 3D tuning ids, so
  every 3D tuning id changed: an id recorded before the series does not match. The
  kernel-name tokens `dg`, `kvcp<policy>` (a policy other than `"all"`), `kvswz` and
  `wpid` mark the new variants. `use_decode_grid` is derived from the problem; the
  others are also tuning axes.
- **Tuning space.** It can now express the kernel default dispatch builds for decode:
  segment candidates 2, 4 and 0 (the split rule's count, recorded in the id), explicit
  3D configs take any count of at least 1, and without `num_cus` tuning uses the
  architecture's full-part CU count instead of 120.
- **Removed.** `HIPDNN_GFX942_3D_HOIST` and the hoist tuning axis. A saved spec that
  sets `use_invariant_hoist` still loads.
- **Cache-policy names.** `CACHE_POLICY_NAMES` and `cache_policy_from_name` (Python;
  `rocke_cache_policy_from_name` in C) replace per-kernel name maps. The gfx950 2D glue
  now rejects an unknown name instead of treating it as `"nt"`, and the cache-bit comments
  describe gfx942/gfx950 (SC0 selects the coherence scope, NT marks streaming data). A
  test pins the bits each name puts on a buffer LDS load.

## 4. Caveats and honest losses

- **Warm K/V flatters the baseline, not the hint.** Real decode reads each K/V byte once
  per step. Some per-lever A/Bs (the non-temporal hint on both architectures, the gfx942
  transposed QK and the gfx942 head-size-256 copies) used K/V that was not
  cache-resident between calls. The series-wide comparison replayed the same inputs, so
  the K/V could stay warm, which is the conservative direction for the non-temporal
  hint: replaying a small K/V makes it look slower because the K/V no longer stays in
  the caches.
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
- **Coverage.** The series-wide comparison used one device per architecture and one
  compiler build; the gfx942 head-size-256 loader was also compared across LLVM 22 and
  23 (§2.2). A GQA ratio that does not divide 16 runs the scalar kernels in both arms
  (§7). The split rule does not apply to fp8 K/V, sliding windows, an explicit
  `target_ctas` or non-decode problems, which keep their previous segment counts and
  tiles. Other defaults reach further: the decode grid and, for fp16/bf16 K/V, the
  non-temporal policy follow `all_decode` alone; on gfx942 the K/V swizzle and the
  page-id window follow `all_decode`, the head size and the K/V feed, so head size 256
  (page-sized tiles) gets them although the rule does not cover it; and gfx942 at head
  size 256 uses the async copies for every 3D problem, decode or not.

## 5. Method and correctness gate

**Correctness.** Every shape's result was checked against a reference before it was
timed. The gate:

- GPU numerics against an FP32 reference. The PR description reports them on gfx942 and
  gfx950 with both engines: the new knob suites
  ([K/V swizzle](../../../tests/test_attention_tiled_3d_kv_lds_swizzle.py),
  [wave page-id window](../../../tests/test_attention_tiled_3d_wave_page_id_load.py);
  their GPU tests run on a gfx942 or gfx950 device, each on its own architecture, and
  skip elsewhere) with reversed and poisoned page tables, ragged and empty sequences
  and multi-page tiles, the
  [strided-KV decode numerics](../../../tests/test_strided_kv_decode_numeric.py), and
  the forced-3D parity harness.
- CPU suites for the [split rule](../../../tests/test_attention_tiled_3d_decode_split.py)
  and the [cache policy](../../../tests/test_attention_tiled_3d_kv_cache_policy.py), and
  a platform [ISA test](../../../../platform/tests/core/test_buffer_lds_cache_policy_isa.py)
  of the cache-policy bits on gfx942 and gfx950.
- Python and C++ engines emit byte-identical LLVM IR at `llvm20`, `llvm22` and
  `llvm23` ([`check_byte_identity.py`](../../../../platform/tools/check_byte_identity.py)),
  with the `.py`/`.c` emitter pairs
  [gfx942](../../../tests/parity/gfx942_attention_tiled_3d_emit.py),
  [gfx950](../../../tests/parity/gfx950_attention_tiled_3d_emit.py) and
  [strided KV](../../../tests/parity/attention_strided_kv_emit.py). The representative
  and strided-KV IR goldens were re-blessed where the 3D kernels changed.

**Performance.** The per-lever A/Bs were interleaved over decode shapes (the exp2 one
over all shapes routed to the 3D path). The series-wide comparison used default
dispatch on both sides (no tuned specs) against develop at `a76c2460399`, the median of
several rounds after warm-up, with two timings: wall time per call including the host
launch, and device time from HIP-graph replay.

## 6. Reproducing

Paths are relative to `dnn-providers/hip-kernel-provider/rocke/`; set the environment
up as in [BUILDING.md](../../../../BUILDING.md) and see
[TESTING.md](../../../../TESTING.md) for the tiers.

```bash
# Gates that need no GPU: relative-path guard, Python/C++ byte identity, parity, pytest.
# --flavor llvm20 or llvm22 pins one flavor. The series also ran llvm23, which this
# runner does not take: ROCKE_LLVM_FLAVOR=llvm23 python platform/tools/check_byte_identity.py
python tools/run_checks.py --steps guard,gate,parity,pytest

# Representative IR goldens.
python platform/tests/instances/rocke_ir_parity_harness.py --check
```

The pytest stage picks up the new tests by itself, and the gate stage picks up the
emitter pairs; the GPU tests in the knob suites run on a gfx942 or gfx950 device and
skip elsewhere. `--steps numeric` is a different lane (the attention FMHA numeric
harness `numeric_attention.py`, plus the gfx950 fp8 decode 3D check), and it fails
without a device when requested explicitly.

For the A/B, build one checkout per arm. The series' own comparison used develop at
`a76c2460399` as the baseline; to replay on the merged tree use the squash commit
`b53003d7bb6` and its first parent `503f5c82876`. The in-repo decode harnesses can replay
the comparison; they are not the harness the series used, and the series' own shape
corpus (prefill and decode attention shapes) is not in the repo:

```bash
python -m benchmarks.gfx950.attention.decode.benchmark_decode_live \
    --shapes library/benchmarks/gfx950/attention/decode/decode_shapes.json
python -m benchmarks.gfx942.attention.decode.benchmark_decode_live \
    --shapes library/benchmarks/gfx942/attention/decode/decode_shapes.json
python -m benchmarks.gfx950.attention.decode.decode_table_sweep --list-only
```

[`benchmark_decode_live`](../../../benchmarks/gfx950/attention/decode/benchmark_decode_live.py)
compares the DSL kernel with AITER's Triton `unified_attention` and sweeps `num_cus` per
shape (see its module docstring; the
[gfx942](../../../benchmarks/gfx942/attention/decode/benchmark_decode_live.py) sibling
differs in its defaults). Its default sweep (`30 60 80 120 152 304` on gfx950,
`30 60 80 120` on gfx942) need not include the device's own CU count, so pass
`--num-cus-sweep` with it to see what default dispatch builds. It builds each shape's
inputs once and reuses them for every timed call, so it measures the warm case (§4); a
non-temporal comparison needs K/V that is not cache-resident between calls, which this
harness does not provide. Run it once per checkout and compare the DSL arm across the
runs.
[`decode_table_sweep`](../../../benchmarks/gfx950/attention/decode/decode_table_sweep.py)
walks every decode candidate the dispatcher registry offers.

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

## 8. Decision

Keep every lever in §2 on its default path as listed. The swizzle and the multi-page
tile path are conditional by construction, and the invariant hoist is removed. Revisit
when the decoupled-KV-pages work lands, on a compiler upgrade, and if the GQA gate is
widened.
