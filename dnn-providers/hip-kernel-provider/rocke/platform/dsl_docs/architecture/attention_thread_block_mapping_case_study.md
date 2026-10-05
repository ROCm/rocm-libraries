<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Dense attention thread-block mapping — gfx942/gfx950 case study

Measured scope: which thread block computes which work item in the dense attention
prefill kernel (`library/kernels/{gfx942,gfx950}/attention_dense.py`), and what that
costs. The kernel is self-attention (`Sq == Skv`) in bf16/fp16 with D 64/128, causal or
non-causal, on gfx942 and gfx950. It records the experiments behind PR #12714, which
added XCD- and L2-aware block orders. The previous version (the baseline) is `develop`
at `c81bca5ecf`, whose dense kernels are unchanged up to `5d7f9be53e`. The new version
is `1e905f9231`. The pair fold and the persistent `hq_minor_swz` (§2.3) came later, on
top of it.

Per [`platform/AGENTS.md`](../../AGENTS.md) §Compliance, this file records relative
results only: ratios between two code paths on the same device, with the correctness
gate, method and caveats stated. Absolute throughput is kept out of the repo. Results
marked **(experimental)** were measured on an unmerged experiments branch (based on
`develop` `7f09e16be8`) or on a prototype. The mapping each one uses is defined in
this document, and each passed the same correctness gate. Varlen, paged,
sliding-window and bottom-right-causal attention are not studied. They keep the
previous orders, apart from the automatic query-block fold described below.

## 1. Shape sets

| set | shapes | what it is for |
|---|---|---|
| end-to-end set | 83 (gfx942) / 105 (gfx950) dense-eligible shapes: MHA with 32–40 heads; GQA with 4 or 8 KV heads and 28–128 query heads; `S` 512–8K; nearly all `B = 1`; D128 (a few D64); causal, plus a few non-causal and sliding-window on gfx950 | real-model prefill shapes run end to end through the dispatcher, previous vs new (§6.1) |
| order-comparison grid | 48 prefill geometries plus a grid of MHA 8–128 heads and GQA `Hkv` ∈ {2, 4, 8, 16, 32} with `Hq` 32–128, `S` ∈ {8K, 32K}, `B` ∈ {1, 2, 4, 8}; causal and non-causal (about 250 shapes per arch) | the non-persistent orders compared with each other; the grid part is the Swizzled Head-first paper's grid, extended with more KV-head counts (§6.2) |
| grid-choice set | the 46 distinct geometries of the end-to-end set, each run causal and non-causal | whether auto picks the right grid, persistent or non-persistent (§6.4) |
| large-batch set | `B` ∈ {8, 16, 32} × MHA 64/64 and GQA 28/4, 32/8, 128/8 × `S` ∈ {2K, 8K}, causal (20 shapes within the kernel's 32-bit extents) | persistent orders and grid choice at large batch (§6.4) |
| profiling set | six representative shapes (§6.3) | every order and grid with L2 counters (§6.3) |

## 2. Background

### 2.1 How blocks reach the hardware

gfx942 and gfx950 have 8 XCDs (compute dies). Each XCD has its own 4 MB L2 and 38
(gfx942) or 32 (gfx950) CUs, and the XCDs share a last-level cache in front of HBM.
Workgroups are dispatched in linear-id order, x fastest, and **round-robin over the
XCDs: `xcd = linear_id % 8`**. So the low digits of the linear id decide which work
shares an L2.

The kernel has two grids:

- **Non-persistent:** one workgroup per work item. A CU takes the next workgroup when it
  frees up, so the assignment is *dynamic* (a queue).
- **Persistent:** `NP` workgroups (the CU count) grid-stride over a work index `wi`.
  Workgroup `c` runs `c, c+NP, c+2NP, …`, so the assignment is *static* (pinned).

### 2.2 Work items and their cost

A work item is `(query block, query head, batch)`. Its K/V is shared by every query
block and by the G query heads of one `(batch, kv head)`. Without a mask every item
costs the same. Under causal masking query block `qb` reads `qb+1` K/V tiles, so cost
grows linearly with `qb`.

The order therefore decides three things, which turned out to matter in this order:

1. **Causal balance and the tail.** What runs last, and how evenly the pinned
   workgroups are loaded. This was the largest effect measured.
2. **Which XCD serves a K/V stream.** If a stream's consumers span several XCDs, each
   XCD fetches the stream into its own L2.
3. **How many streams each XCD has live at once.** This matters mostly at long `S`
   with many heads.

### 2.3 Query-block traversal

- **Longest first** (`NQB-1-qb`) is right for the dynamic queue. It is list
  scheduling with the longest jobs first, and it leaves the cheapest blocks for the
  tail.
- **Fold** (`qb < NQB/2 ? qb : NQB-1-(qb-NQB/2)`) is right for the pinned grid. Along a
  workgroup's stride it pairs a cheap block with an expensive one, so per-workgroup
  totals equalize.
- **Pair fold** (persistent `hq_minor_swz`, `hkv_major`; even `NQB`) runs both blocks
  of `{p, NQB-1-p}` on one workgroup's consecutive steps, so every unit costs the
  same whatever the digit order. The fold balances only when a stride crosses both
  halves: automatic when the query block is the slowest digit (`qb_major`,
  `bt_hkv_minor`, where pairing had almost no effect, §7), not under XCD head bands
  or with the kv head slowest.
- **Alternating directions** (`interleave`: reverse on odd heads) is weaker than both.
  Every workgroup still sees the full cost range, and half of them end on the most
  expensive blocks.
- **No traversal can rescue an order whose fastest digit is the query block.** Every run
  of `NQB` consecutive items then holds each cost exactly once. A traversal only
  permutes costs inside a run, so the most expensive blocks recur up to the end of the
  dispatch.

Ordering stops mattering in two regimes. With a single wave (`W ≤` CUs) there is no
tail to shape. When every work item is very long (`S ≥ 16K`), the tail is a small
fraction of the runtime.

### 2.4 Capacity and temporal locality

K plus V of one `(batch, kv head)` is `512·S` bytes at D128, so a whole head fills one
XCD's L2 at about `S = 8K`. To confine a stream to one XCD, the stream also needs
enough work to fill it: `G·NQB` items against 38 or 32 CUs. MHA never fills an XCD with
one head's blocks. That is why MHA needs a head *band* per XCD (Swizzled Head-first,
§3) rather than one stream per XCD.

**Temporal locality beyond capacity.** A head that does not fit still gets L2 reuse.
Workgroups that consume the same stream and start together walk its K/V tiles from 0
upward at nearly the same pace. They are not synchronized, but every KV iteration
costs the same except the diagonal tile, so they drift apart slowly. A tile the
leader loads is still resident when the followers reach it. The live set per XCD is
then *streams in flight × drift window*, not whole heads.

- **It holds when:**
  - a stream's consumers are adjacent in dispatch order, so they start in the same
    wave on the same XCD;
  - per-iteration cost is uniform, which is true for causal and non-causal alike over
    the shared K/V prefix;
  - there are few streams per XCD.
- **It weakens when:**
  - starts are staggered (later waves, or the uneven finishes that longest-first
    creates on purpose);
  - a stream's consumers are spread over XCDs or waves;
  - an XCD interleaves many streams;
  - drift accumulates over very long K/V loops.

Measured (§6.3): at `S = 32K`, where one head's K/V is four times an XCD's L2, the
non-persistent auto order (`bt_hkv_minor`) keeps an 87% (gfx942) / 95% (gfx950) L2
hit rate.

### 2.5 Notation

Orders are written as digits, fastest first: `Q` query block, `B` batch, `V` kv head,
`G` query head within its kv group (`hq = hkv·G + g`).

## 3. Prior art

| work | mapping | relevance |
|---|---|---|
| [Swizzled Head-first](https://arxiv.org/abs/2511.02132) (arXiv 2511.02132) | each XCD owns a contiguous band of query heads and runs one head's blocks at a time | added here as `hq_minor_swz`. The paper has no causal traversal, its batch formula is inconsistent for `B > 1`, and its GQA tests are only at 8 KV heads |
| aiter `remap_xcd` (Triton MHA) | heads banded per XCD, query blocks ascending ("Swizzled Block-first") | the de-facto AMD convention |
| aiter `remap_workgroup_spatial` (opt-in) | KV head pinned to an XCD, block-first over its query heads; fewer than 8 KV heads are split over XCDs by contiguous block ranges | prototyped here (§7). Its split is unbalanced under causal masking, and its fallback for head counts not divisible by 8 is not a bijection |
| AOTriton causal forward | persistent, tiles claimed from an atomic counter, ascending | balances with a dynamic queue instead of an order |
| [FlashAttention-3/4](https://github.com/Dao-AILab/flash-attention/blob/main/hopper/tile_scheduler.hpp) tile scheduler | batch outermost, head sections sized to fit L2, longest-first query blocks, dynamic claiming | the closest analogue: L2-sized sections plus longest-first |
| [HipKittens](https://arxiv.org/abs/2511.08083) | notes that per-XCD L2 locality and shared-cache traffic trade off on 8-XCD parts | a caution against maximizing L2 locality alone |
| [Sawtooth](https://arxiv.org/abs/2601.16032) | alternates the K/V loop direction between a workgroup's consecutive items | only pays once K/V exceeds the cache |

None of the AMD attention libraries read here reverses or folds causal query blocks;
they balance dynamically or not at all. Versions read from source: aiter `c8325e00`
and `21ae719`, AOTriton 0.14b (`b5e8cfb`), FlashAttention `e9cf2c1`.

## 4. Previous status

- **Non-persistent:** one fixed order, grid `(NQB, Hq, B)` (now `qb_minor`, `QGVB`).
  The query block is the fastest digit and runs ascending, which is the worst class
  for causal attention (§2.3). Adjacent workgroups also split one K/V stream across
  XCDs by query block.
- **Persistent:** `qb_major` (ascending), `hkv_major` (folded), the `interleave` knob
  and, on gfx950, `gqa_pair` / `gqa_pair_2phase`. Auto chose `hkv_major` or
  `qb_major`, and on gfx950 it chose the pair decodes where their CTA-count condition
  happened to hold.
- **Consequence:** on causal shapes the non-persistent path was 26–44% slower than the
  persistent auto path at `S ≤ 8K`, and 14–20% slower at `S = 32K` (profiling set,
  §6.3). So gfx942 dispatch went persistent as soon as the grid filled one wave.

## 5. What this case study implemented

| path | value | order | used for (auto) |
|---|---|---|---|
| non-persistent | `qb_minor` | `QGVB`, the previous grid | non-causal, and everything that is not aligned causal |
| non-persistent | `bt_hkv_minor` (new) | `BVGQ`, longest-first | aligned causal (default) |
| non-persistent | `hq_minor_swz` (new) | Swizzled Head-first: XCD `a` owns head band `a`, one head's blocks at a time, longest-first | causal MHA with many heads × batch × blocks, where dispatch picks the non-persistent grid |
| persistent | `hq_minor_swz` (new) | Swizzled Head-first, pair fold | aligned causal MHA with more than 8 grid-stride rounds of work (`Hq` divisible by 8, even `NQB`) |
| persistent | `bt_hkv_minor` (new) | `BVGQ`, folded | aligned causal, batch below the XCD count |
| persistent | `hkv_major` | `BGQV`, now pair fold under causal | GQA with enough work per kv head (the previous rule), including aligned causal from `chiplet_num_xcds` batches |
| persistent | `qb_major` | `BGVQ`, now folded under causal | everything else on the persistent grid |
| persistent | `gqa_pair*` | unchanged | explicit only |

- **Knobs:** `nonpersist_decode` is new and mirrors `persist_decode`.
  `chiplet_num_xcds` defaults to 8; another value stays correct and only weakens
  locality.
- **Request field and builder flag:** `AttentionRequest.dense_nonpersist_decode`, and
  `--nonpersist-decode` on the builders.
- **Kernel name and cache key:** both include the resolved order.
- **Balance is automatic:** longest-first on the non-persistent grid, fold or pair
  fold (§2.3) on the persistent grid. There is no knob.
- **"Aligned causal"** means causal attention on the plain dense layout: no sliding
  window, ragged, varlen, paged, or moving bottom-right diagonal.
- **Dispatch (§6.4):**
  - gfx942 takes the persistent grid at D128 for every batch and mask, and the
    non-persistent grid at D64;
  - gfx950 keeps the previous rule: persistent once the work fills the grid.
- **The persistent grid stays ahead at D128 on both arches, for different reasons.**
  - gfx950's persistent builder is a different kernel body with extra optimizations:
    wide LDS DMA (128-bit buffer→LDS loads) and a different scheduling template. So
    non-persistent on gfx950 is still slower than persistent.
  - On gfx942 one body serves both grids, yet the persistent grid with the new
    decodes also measured ahead at D128, even when the work fits in one wave (§6.4).
    At D64 the non-persistent grid is ahead.
- **Correctness:** every order is a bijection onto the work space, so outputs are
  bit-identical across orders. Unit tests evaluate every order's decode on the host,
  and every measured cell was checked on the GPU.

## 6. Performance effect

**Method, for every table below:**
- **correctness:** bf16/fp16 inputs; outputs checked against a PyTorch SDPA reference
  (max abs error ≤ 2e-2) wherever the reference was affordable, and all orders of a
  shape verified bit-identical. On shapes too large for the reference, correctness
  rests on bit-identity with an order that passed SDPA at smaller shapes;
- **timing:** one process per kernel; median of 3–5 timed runs of back-to-back
  launches (warm caches); two independent passes;
- **noise:** the per-shape run-to-run difference has a median of 0.2–0.8% and a 90th
  percentile ≤ 1.7%.
- **configurations:** arms that pin an order or a grid are diagnostic configurations;
  only "auto" is what dispatch selects.

Replay a single point with the builders
`library/builders/{gfx942,gfx950}/attention/prefill/attention_dense_prefill.py`, which
take `--nonpersist-decode` / `--persist-decode` and check parity against SDPA.

### 6.1 End to end, previous → new, end-to-end set

Both versions ran through the dispatcher: `develop` at `5d7f9be53e` against the new
version `1e905f9231` (same tree). Each point is the median of 5 iterations of 250 timed
executions after 75 warmups. Every output was checked against a reference: none was
flagged, and error metrics were identical on both sides.

| | gfx942 (83 shapes) | gfx950 (105 shapes) |
|---|---|---|
| geomean | **+11.4%** | **+4.2%** |
| median | +9.9% | +2.5% |
| faster / slower by more than 1% | 77 / 2 | 67 / 9 |
| worst / best | −3.0% / +41% | −5.5% / +30% |
| `S` 2K–4K | +15.0% | +7.7% |
| `S ≤ 1K` | +9.5% | +0.9% |
| `S ≥ 8K` | +7.6% | +3.6% |

- **gfx942 gains at every sequence length.** Most shapes now run the persistent grid
  with the new decodes (§6.4). An earlier version of the rules, which kept gfx942 on
  the non-persistent grid, measured +7.8% here (§7).
- **gfx950 losses are almost all at `S ≤ 1K`,** where the work is about one wave or
  less. The sliding-window shapes are unchanged (−0.2%).
- The later pair fold and persistent `hq_minor_swz` (§2.3) change no kernel in this
  set.

### 6.2 Non-persistent orders, order-comparison grid

Non-persistent grid only. Geomean vs the previous non-persistent order (`qb_minor`),
order-comparison grid. "Auto" is the non-persistent auto order; which grid auto picks
is §6.4.

| regime | mask | gfx942: `bt_hkv_minor` / `hq_minor_swz` / auto | auto vs best order | gfx950: `bt_hkv_minor` / `hq_minor_swz` / auto | auto vs best |
|---|---|---|---|---|---|
| MHA | causal | +24.1% / +27.5% / **+28.5%** | −0.8% | +26.8% / +26.1% / **+31.2%** | −0.1% |
| GQA `Hkv < 8` | causal | +29.4% / +26.2% / **+29.4%** | −0.1% | +35.6% / +32.8% / **+35.6%** | −0.3% |
| GQA `Hkv = 8` | causal | +32.3% / +23.6% / **+32.3%** | −0.2% | +38.1% / +28.6% / **+38.1%** | −0.7% |
| GQA `Hkv > 8` | causal | +28.1% / +28.4% / **+28.1%** | −1.8% | +35.5% / +38.4% / **+35.5%** | −3.3% |
| all | non-causal | −8.4% … +0.2% / −4.1% … −0.9% / **0** | −0.0 … −0.7% | −7.5% … −1.0% / −3.0% … −1.2% / **0** | −0.4 … −1.1% |

- **Causal:** the new orders are worth about +24% to +38% over the previous order, and
  +45% to +67% at `S` 4K–8K. Auto is within 0.1–1.8% (gfx942) / 0.1–3.3% (gfx950) of
  the best order per shape.
- **Where auto loses most:** large GQA (many heads, `B ≥ 4`, long `S`). There
  `hq_minor_swz` beats `bt_hkv_minor` by up to 12% (gfx942) / 14% (gfx950). Auto
  selects it only for MHA (§8).
- **Non-causal:** the previous order is still the best non-persistent order, and step
  0a agrees (every alternative 0.2–3.2% behind), so auto keeps it.

### 6.3 Representative shapes: throughput and L2 counters

bf16, D128. "Auto" is the dispatcher's default choice: grid and order both automatic.
Throughput is relative to previous auto on the same arch. L2 hit rate is
`TCC_HIT / (TCC_HIT + TCC_MISS)`. "L2 misses" counts L2 read misses to the fabric
(`TCC_EA0_RDREQ`), relative to previous auto, median over 4 dispatches under
rocprofv3. The non-persistent orders were bit-identical on every shape. The SDPA
check ran on the `B = 1` shapes (the other two were too large).

**gfx942**

| shape | arm | throughput vs previous auto | L2 hit rate | L2 misses vs previous auto |
|---|---|---:|---:|---:|
| GQA 32/8, S=4K, B=1, causal | previous auto (persistent `qb_major`) | +0.0% | 54% | 1.00× |
|  | previous non-persistent `qb_minor` | -25.9% | 82% | 0.35× |
|  | new non-persistent `bt_hkv_minor` | +23.1% | 86% | 0.24× |
|  | new non-persistent `hq_minor_swz` | -8.0% | 81% | 0.35× |
|  | new auto (persistent `bt_hkv_minor`) | +29.3% | 81% | 0.38× |
| GQA 64/8, S=8K, B=1, causal | previous auto (persistent `qb_major`) | +0.0% | 27% | 1.00× |
|  | previous non-persistent `qb_minor` | -33.2% | 81% | 0.23× |
|  | new non-persistent `bt_hkv_minor` | +2.1% | 91% | 0.10× |
|  | new non-persistent `hq_minor_swz` | -0.8% | 77% | 0.29× |
|  | new auto (persistent `bt_hkv_minor`) | +5.8% | 73% | 0.35× |
| GQA 128/8, S=4K, B=16, causal | previous auto (persistent `hkv_major`) | +0.0% | 56% | 1.00× |
|  | previous non-persistent `qb_minor` | -35.5% | 87% | 0.24× |
|  | new non-persistent `bt_hkv_minor` | -6.6% | 65% | 0.75× |
|  | new non-persistent `hq_minor_swz` | -8.1% | 76% | 0.49× |
|  | new auto (persistent `hkv_major`) | +5.9% | 87% | 0.24× |
| MHA 64/64, S=8K, B=4, causal | previous auto (persistent `qb_major`) | +0.0% | 25% | 1.00× |
|  | previous non-persistent `qb_minor` | -28.7% | 63% | 0.48× |
|  | new non-persistent `bt_hkv_minor` | -3.4% | 28% | 0.95× |
|  | new non-persistent `hq_minor_swz` | +2.5% | 79% | 0.26× |
|  | new auto (persistent `hq_minor_swz`) | +10.4% | 51% | 0.65× |
| GQA 32/8, S=32K, B=1, causal | previous auto (persistent `qb_major`) | +0.0% | 9% | 1.00× |
|  | previous non-persistent `qb_minor` | -14.0% | 61% | 0.42× |
|  | new non-persistent `bt_hkv_minor` | +1.3% | 87% | 0.13× |
|  | new non-persistent `hq_minor_swz` | -0.1% | 79% | 0.22× |
|  | new auto (persistent `bt_hkv_minor`) | +6.3% | 58% | 0.45× |
| GQA 32/8, S=4K, B=1, non-causal | previous auto (persistent `qb_major`) | +0.0% | 70% | 1.00× |
|  | previous non-persistent `qb_minor` | -8.7% | 88% | 0.71× |
|  | new non-persistent `bt_hkv_minor` | -12.5% | 90% | 0.38× |
|  | new non-persistent `hq_minor_swz` | -11.8% | 83% | 0.63× |
|  | new auto (persistent `qb_major`) | -0.0% | 70% | 1.00× |

**gfx950**

| shape | arm | throughput vs previous auto | L2 hit rate | L2 misses vs previous auto |
|---|---|---:|---:|---:|
| GQA 32/8, S=4K, B=1, causal | previous auto (persistent `gqa_pair_2phase`) | +0.0% | 72% | 1.00× |
|  | previous non-persistent `qb_minor` | -38.1% | 80% | 0.67× |
|  | new non-persistent `bt_hkv_minor` | -2.6% | 90% | 0.25× |
|  | new non-persistent `hq_minor_swz` | -27.7% | 91% | 0.24× |
|  | new auto (persistent `bt_hkv_minor`) | +4.3% | 91% | 0.23× |
| GQA 64/8, S=8K, B=1, causal | previous auto (persistent `gqa_pair`) | +0.0% | 67% | 1.00× |
|  | previous non-persistent `qb_minor` | -43.7% | 88% | 0.30× |
|  | new non-persistent `bt_hkv_minor` | -3.9% | 93% | 0.16× |
|  | new non-persistent `hq_minor_swz` | -8.5% | 91% | 0.21× |
|  | new auto (persistent `bt_hkv_minor`) | +2.5% | 93% | 0.14× |
| GQA 128/8, S=4K, B=16, causal | previous auto (persistent `hkv_major`) | +0.0% | 89% | 1.00× |
|  | previous non-persistent `qb_minor` | -41.6% | 89% | 1.00× |
|  | new non-persistent `bt_hkv_minor` | -8.4% | 58% | 4.90× |
|  | new non-persistent `hq_minor_swz` | -9.6% | 90% | 0.82× |
|  | new auto (persistent `hkv_major`) | +0.5% | 89% | 1.00× |
| MHA 64/64, S=8K, B=4, causal | previous auto (persistent `qb_major`) | +0.0% | 17% | 1.00× |
|  | previous non-persistent `qb_minor` | -26.0% | 57% | 0.50× |
|  | new non-persistent `bt_hkv_minor` | -2.2% | 18% | 1.00× |
|  | new non-persistent `hq_minor_swz` | +7.9% | 86% | 0.14× |
|  | new auto (persistent `hq_minor_swz`) | +24.2% | 59% | 0.48× |
| GQA 32/8, S=32K, B=1, causal | previous auto (persistent `hkv_major`) | +0.0% | 73% | 1.00× |
|  | previous non-persistent `qb_minor` | -19.9% | 71% | 1.08× |
|  | new non-persistent `bt_hkv_minor` | -4.2% | 95% | 0.15× |
|  | new non-persistent `hq_minor_swz` | -5.3% | 93% | 0.23× |
|  | new auto (persistent `bt_hkv_minor`) | +1.3% | 91% | 0.32× |
| GQA 32/8, S=4K, B=1, non-causal | previous auto (persistent `qb_major`) | +0.0% | 86% | 1.00× |
|  | previous non-persistent `qb_minor` | -0.3% | 86% | 1.00× |
|  | new non-persistent `bt_hkv_minor` | +0.1% | 93% | 0.40× |
|  | new non-persistent `hq_minor_swz` | +1.1% | 93% | 0.40× |
|  | new auto (persistent `qb_major`) | +0.7% | 86% | 1.00× |

**Conclusions.**

- **Balance explains most of the speedup, not hit rate.** On GQA `B = 1` causal shapes,
  the previous non-persistent order already hit L2 80–88% of the time, yet was 26–44%
  slower than previous auto. The new non-persistent `bt_hkv_minor` raises the hit rate
  only to 86–93%, but removes the ascending tail.
- **The new auto cuts L2 misses substantially:**
  - GQA `B = 1` causal: 2.2–2.9× fewer on gfx942 and 3–7× fewer on gfx950 than
    previous auto, whose `qb_major` / `gqa_pair*` choices spread each K/V stream over
    many XCDs;
  - large MHA: the non-persistent `hq_minor_swz` needs 4–7× fewer (hit rate 79–86%
    vs 17–28% for the other orders). The persistent `hq_minor_swz` that auto now
    picks misses more (51–59% hit rate) but is the fastest, +10.4% (gfx942) / +24.2%
    (gfx950): balance again outweighs hit rate.
- **Hit rate does not rank the grids.** On gfx942 the new auto (persistent
  `bt_hkv_minor`) has a lower hit rate than the non-persistent `bt_hkv_minor` (58–81% vs
  86–91%) and is still 3–5% faster; see §6.4.
- **Temporal locality at `S = 32K`** (§2.4): the new non-persistent `bt_hkv_minor` keeps
  87% (gfx942) / 95% (gfx950) of reads in L2, against 9% / 73% for previous auto.
- **Unchanged where nothing better was found:**
  - non-causal auto is still the previous persistent `qb_major` (the non-persistent
    orders are 9–13% behind it on gfx942);
  - large-batch GQA auto is still `hkv_major`, now with the pair fold: +5.9% on
    gfx942, where its hit rate rises from 56% to 87% because each grid-stride phase
    spans fewer kv heads, and +0.5% on gfx950, where it was already 89%.

### 6.4 Grid choice (dispatch)

Auto vs previous auto, measured in the same session per arch. The shapes are the 46
prefill geometries (mostly `B = 1`, a few `B` 2–4; D128, plus five D64), each run
causal and non-causal, plus 20 large-batch shapes (`B` 8–32, causal, D128). The
large-batch shapes were re-measured with the pair fold (§2.3); outputs were
bit-identical to the previous version. Every
run with an affordable SDPA reference matched it (max abs error 3.9e-3 in bf16).

| slice | gfx942 | gfx950 |
|---|---|---|
| causal, `B ≤ 4` (46) | **+10.9%** (worst −3.5%) | **+6.5%** (worst −1.3%) |
| non-causal, `B ≤ 4` (46) | +5.5% (worst −3.1%) | +0.2% (worst −1.6%) |
| D64, both masks (10 of the 92 above) | +18.2% | +2.9% |
| causal, `B` 8–32 (20) | +4.3% (worst −1.0%) | +3.7% (worst −2.8%) |
| all (112) | **+7.4%** (worst −3.5%) | **+3.3%** (worst −2.8%) |

- **gfx942 at D128: the persistent grid wins at every batch and mask.** With the new
  decodes it is ahead of the non-persistent auto by about 4% on causal and 10% on
  non-causal shapes (`B = 1`). That holds even when the work fits in one wave, so part
  of the gap is the grid's code path rather than the order. The cause is not
  identified.
- **gfx942 at D64: the non-persistent grid wins,** by about 11% (causal) and 16%
  (non-causal).
- **Large batch:** MHA now runs the persistent `hq_minor_swz`: +10.5% (gfx942) /
  +20.0% (gfx950). GQA keeps the previous rule, `hkv_major` where it applied, now
  with the pair fold: +2.8% / +0.0%, with gfx950 up to 2.8% behind at `S = 8K`.
- **gfx950 at `S ≤ 1K`:** the unchanged previous rule picks the non-persistent grid
  when the work does not fill the persistent grid. There the persistent grid measured
  about 8% ahead on causal shapes (13 of 13) and 2% on non-causal (§8).
- **Caveats:**
  - the `B ≤ 4` set is mostly `B = 1`;
  - D64 covers one geometry at five sequence lengths;
  - sliding window was spot-checked on six gfx942 shapes only: persistent ahead on
    five, behind on a 128-token window.

## 7. What else was tried

| idea | result | why |
|---|---|---|
| all 24 digit orders × {ascending, longest-first, fold} **(experimental)** | only the batch/kv-head-fastest class generalizes. Orders with the query block fastest lose 22–31%; `BVGQ` was within 1.6% geomean of the per-shape best on every arch and grid | §2.3: nothing rescues query-block-fastest orders; balance is a hard gate |
| locality-maximizing orders (one K/V stream per XCD, e.g. `VGQB`) **(experimental)** | the best locality measured *worst*; `BVGQ` won with 8–32× poorer locality | pinning one stream per XCD gives each workgroup one query block, which ruins causal balance. Locality only separates orders once balance is equal |
| kv-phase split (split the fused `(batch, kv head)` identity across the fastest and slowest digits) **(experimental)** | +1.3% on gfx942 persistent at `Hkv = 8` (large batch only); ties elsewhere; about 12 extra integer ops per item | not worth a knob |
| `gqa_pair`, `gqa_pair_2phase` (gfx950) | very restrictive: GQA with even `G`, even `NQB`, a baked shape, and a CTA count that must equal `NQB·Hkv·B` (or half the work), so under the default CTA count they fire only by coincidence. At matched CTA count they are **2.2% (93 shapes) / 3.9% (64 shapes) behind** the new auto; at the default CTA count, far behind | the new persistent orders balance and localize at least as well without the constraints; now explicit only |
| pair fold for persistent `bt_hkv_minor` and `qb_major` (causal, 54 shapes) **(experimental)** | almost no effect | the query block is their slowest digit, so a workgroup's stride already crosses both halves of the fold |
| `interleave` vs fold (persistent `qb_major`, causal, 62 shapes) **(experimental)** | fold ahead by 4.5% (gfx942) / 4.9% (gfx950) geomean; interleave only 0.4–0.7% better than ascending | §2.3; interleave is kept only as an explicit knob |
| Swizzled Head-first on the persistent grid, as-is **(experimental)** | 10.5% (gfx942) / 27% (gfx950) behind auto on causal shapes, whatever the traversal | grid-stride advances the block digit by `NP/8`; when that is a multiple of `NQB`, every workgroup keeps one query block for its whole life. The pair fold fixes this (§2.3) |
| aiter's KV-head-first swizzle (`hkv_minor_swz`: grid `(Hq·NQB, 1, B)`, XCD `a` owns kv heads `a·Hkv/8 …` one at a time, block-first over their query heads) **(prototype on the new version)** | vs non-persistent auto: `Hkv = 8` ties (+0.1–2.8%); `Hkv > 8` causal **+1.6% (gfx942) / +4.4% (gfx950)**, but gfx950 auto uses the faster persistent grid there anyway; `Hkv < 8` causal **13–35% behind**. MHA is identical to `hq_minor_swz` | `bt_hkv_minor` already pins each `(batch, kv head)` to one XCD whenever `B·Hkv % 8 == 0`. The `Hkv < 8` split hands one XCD the late, expensive blocks |
| first version of the new rules, gfx942: persistent only from 16 batches | behind the previous auto on non-causal D128 (−3.3% geomean, worst −14.9%) and at `B = 8` (up to −7.3%) | it relied on the non-persistent grid beating the *old* persistent decodes; with the new ones the persistent grid wins at D128 (§6.4). The rule now keys on head size |
| first version of the new rules: folded `qb_major` for aligned causal from `chiplet_num_xcds` batches | behind the previous `hkv_major` on gfx950 GQA (−2.9% geomean, worst −10.7%) | the earlier study compared it only with `bt_hkv_minor`; auto now keeps the previous rule there |

**Measurement lessons.**
- **The launcher cache is keyed by kernel name.** An order missing from the name
  silently re-times another binary (it happened three times). Every order now has a
  name tag and is part of the cache key.
- **Re-baseline against the current auto.** Quoting against an order nothing selects
  overstates gains several-fold.
- **Re-check the grid choice after improving either grid.** Both first-version rule
  errors above came from comparing a new candidate against a stale alternative.
- **Median over runs, then two passes.** Single bad runs invent regressions.
- **Free noise probes.** Orders that coincide at `G = 1` or `B = 1` compile to the
  same mapping, so any gap between them is noise.

## 8. What can still be checked

- **Pair longest and shortest block in one unit.**
  - *Persistent:* done for even `NQB` (§2.3). Odd `NQB` still uses the fold; the
    `hq_minor_swz` docstring has a recipe (pair the middle blocks of two heads).
  - *Non-persistent:* one workgroup computes both blocks. That halves the grid,
    equalizes workgroup cost and shares the K/V prefix, but needs `W/2 ≥` CUs and
    makes each workgroup twice as long. Unlike `interleave`, it pairs within a
    workgroup rather than alternating across heads.
- **`hq_minor_swz` for large GQA** (§6.2): the largest remaining auto losses.
- **The prototype KV-head-first swizzle for `Hkv > 8`** (small gain on gfx942), and
  uneven head bands for head counts not divisible by 8.
- **Dynamic persistent scheduling:** an atomic work counter plus longest-first order,
  like AOTriton and FlashAttention.
- **`num_persistent = CUs × occupancy`** instead of the CU count. On gfx942 D64 the
  smaller tile runs two workgroups per CU while the grid has one per CU, which may be
  why the persistent grid loses there (§6.4).
- **gfx950 grid choice for small problems:** persistent measured about 8% ahead at
  `S ≤ 1K` (causal), where the unchanged rule picks the non-persistent grid (§6.4).
- **gfx942 GQA at large batch:** the folded `qb_major` measured 1.5% ahead of the
  folded `hkv_major` the rule picked. The pair fold has since gained `hkv_major`
  3.1% there, so recheck before adding a gfx942-only rule.
- **gfx942 D128 shapes where the non-persistent grid still wins:** 40 heads at `S ≥ 4K`
  (up to 18% ahead of persistent; end to end these shapes gave back 7–17% of the first
  version's gain, landing within 3% of the previous version), and a 128-token sliding
  window, where persistent is 11.5% slower. The first may be fold imbalance when the
  work does not divide the 304-workgroup grid.
- **Exploit temporal locality deliberately at long `S`:**
  - keep one stream's consumers in the same wave and XCD, and bound the streams per
    XCD (FlashAttention's L2-sized sections);
  - K/V-direction reversal (sawtooth) once one head exceeds L2 (`S > 8K`).
- **Coverage gaps,** each needing its own benchmarks:
  - `B ≥ 16` on the non-persistent grid;
  - `S ≥ 64K`, and non-power-of-two `S`;
  - non-causal at large batch;
  - varlen, paged, sliding window and bottom-right causal.

  On the order-comparison grid, auto is within about 1–2% of the best order on
  average up to `B = 8` and `S = 32K`, apart from the large-GQA case above.

## 9. Other possible improvements (not orderings)

- **Port wide LDS DMA and the persistent scheduling template to gfx950's non-persistent
  body.** Disabling `wide_lds_dma` on the persistent grid costs about 6%
  **(experimental)**, so porting it could give the non-persistent grid a similar gain.
  That would close most of the gap in §5 and reach varlen, which only has a
  non-persistent path.
- **Unify gfx950's persistent and non-persistent bodies if possible,** as gfx942 already does.
- **Runtime-shape persistent decodes.** Today they bake `NQB`.
- **GQA packing:** put a kv head's G query heads in one tile, so K/V reuse no longer
  depends on the cache.
- **Retire `interleave`, and consider retiring `gqa_pair*`** if broader benchmarks
  confirm they are not ahead of the other orders.

## 10. Where the code lives

| concern | file |
|---|---|
| decode classes (grid + decode per order) | `library/kernels/common/attention_dense_decode.py` |
| knobs, legality, auto rules, name tags, cache key | `library/kernels/common/attention_dense_spec.py` |
| call sites (grid, decode, persistent loop) | `library/kernels/{gfx942,gfx950}/attention_dense.py` |
| persistent vs non-persistent dispatch | `library/dispatch/attention/{gfx942,gfx950}.py` |
| builders (`--persist-decode`, `--nonpersist-decode`) | `library/builders/{gfx942,gfx950}/attention/prefill/attention_dense_prefill.py` |
| host tests of every decode | `library/tests/test_attention_dense_decode.py` |
