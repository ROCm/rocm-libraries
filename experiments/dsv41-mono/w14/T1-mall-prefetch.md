# W14 T1: routed-expert MALL prefetch for DeepSeek-V4.1-Flash decode (MI355X, TP2, "rocKE hybrid")

W14, 2026-11-03. gfx950 (MI355X), TP2 no-EP, DSpark 5, the B0 recipe (bf16 KV, fp8 index plane, `--level 0` + FULL
graphs). Ratios only; absolute numbers and raw logs: `hackathon-prep/private-perf/dsv41-mono/w14/t1/`. Harness:
`hackathon-prep/dsv41/w14/t1/`. Status: DONE (verdict in section 6).

## 1. MALL behaviour on MI355X (step 1, measured)

Probe: `rocke/library/kernels/gfx950/dsv41_mono/mall_probe.py` (a gathered byte streamer with any load path and
cache-policy bits, in-kernel `s_memrealtime` CTA stamps, and a K2-footprint occupant kernel), driver
`t1_mall.py`. Every number is the in-kernel span of the timed launch, median of 7-9 shuffled interleaved rounds.
Gate: the streamer's coverage is exact (XOR fold of every loaded byte == host fold, 3 knob sets x 2 unions), and the
must-fail twins (`skip`: a region dropped; a wrong expert id list; the `write` mutant changes the bytes, the plain
prefetch does not) all fail: PASS in every job (mb1-mb5).

Jobs: mb1 (n07-17), mb2 (n01-09), mb4 / mb5 (n07-17). mb1's "cold" cells were bimodal: a single 2 GiB streaming read
does NOT evict lines the MALL has seen re-referenced (see 1.5); from mb2 on the flush is a 192 MiB buffer read three
times plus 2 GiB once, and the cold cells are unimodal (spread under 10 %).

### 1.1 Which loads allocate into the MALL and which hit there

Flush, prefetch 128 MiB with a policy, then a floor-quality `nt` stream (256 x 512, depth 4, 4 x 16 B per lane: the
policy and shape of ATOM's expert stream and of W12 P4's bytes floor) over the same bytes; time / the cold time:

| prefetch load | n01-09 (mb2) | n07-17 (mb4) |
|---|---|---|
| `buffer_load` default (aux 0) | 0.865 | 0.861 |
| `sc0` (1) | 0.864 | 0.872 |
| `sc1` (16) | 0.865 | 0.879 |
| `sc0 sc1` (17) | 0.856 | 0.861 |
| any with `nt` (2, 3, 18, 19) | 0.989-1.008 | 1.011-1.018 |
| `buffer_load ... lds` (DMA) default / `sc0` | 0.938 / 0.944 | 0.943 / 0.944 |
| `buffer_load ... lds` with `nt` | 0.991-1.000 | 1.013-1.020 |
| `s_prefetch_data`, `global_prefetch_b8` | not available on gfx950 (the assembler refuses both: gfx1250 only) | |

- Only loads WITHOUT `nt` allocate. `nt` loads neither allocate (an `nt` stream read twice: second pass 0.98-1.02 of
  cold at every size, mb2 `cap`) nor evict (1.4), but they DO hit lines another load placed there. So the prefetcher
  must read without `nt`, and ATOM's `nt` expert stream needs no change to benefit.
- The LDS-DMA path allocates less and is slow as a prefetcher (1.33-1.62 of the `nt` stream's own time).

### 1.2 Bandwidth: MALL hit vs HBM

Same shape, prefetched (non-nt, then the stream) vs cold, MiB read once, TB/s ratio hit / cold (mb2, n01-09):

| stream (256 x 512) | 64 MiB | 128 MiB | 192 MiB |
|---|---|---|---|
| depth 4, 16 B a lane a step (latency-bound) | 1.41 | 1.18 | 1.31 |
| depth 4, 2 x 16 B | 1.22 | 1.19 | 1.21 |
| depth 4, 4 x 16 B (floor-quality, ATOM's MoE class) | 1.28 | 1.14 | 1.11 |
| depth 8, 4 x 16 B | 1.33 | 1.13 | 1.11 |

The MALL is a latency win, not a bandwidth one: a stream already at the HBM floor reads MALL-resident bytes only
1.11-1.14x faster at 128-192 MiB (more at small sizes, where the cold stream's ramp dominates).

### 1.3 Capacity, per XCD (L2) and per IOD

- Whole device, contiguous (both IODs): a default-policy re-read of F MiB is 0.91-0.94 of cold for F = 64-256 and
  0.99-1.01 from 288 MiB on: **about 256 MiB** usable (mb2; mb1's latency-bound stream: hits to 256, none at 288).
- Per XCD (L2, 4 MiB): 2 MiB prefetched by XCD 0, then read by XCD 0: 0.69 of cold (L2 hit); by XCD 1 or XCD 4:
  0.89-0.90 (MALL only). 64 MiB read by one XCD is XCD-bandwidth-bound (0.95-0.99 whatever was prefetched).
- MALL-only vs MALL + L2: 64 MiB read by all 8 XCDs after a prefetch by all 8 with the SAME unit-to-CTA mapping:
  0.78 of cold (part of it L2 hits); after a prefetch by a subset of XCDs (0-3, 4-7, or 0): 0.94-0.97. A real MoE
  maps units to CTAs its own way, so the prefetch it can use is MALL-only. Measured directly (mb8, n07-17: the
  prefetch's unit-to-CTA map rotated by r CTAs so no unit is prefetched by the XCD that reads it; r = 0 / 8 keep
  the XCD): contiguous 64 / 128 / 192 MiB 0.934 / 0.963 / 0.945 of cold (r = 1; r = 3: 0.943 / 0.955 / 0.942)
  vs 0.774 / 0.866 / 0.879 with the same XCD; the MoE-shaped unions of 1.7 at 6 / 12 / 24 rows 0.889 / 0.923 /
  0.942 MALL-only vs 0.835 / 0.889 / 0.920 same-XCD. **The usable MALL-only ceiling is 11 % of the MoE stream at
  6 rows, 8 % at 12, 6 % at 24.**
- Per IOD (data restricted to the 8 KiB chunks homed at one IOD; home = address bit 13 XOR a per-2-MiB-page flag,
  calibrated per page, two calibrations agreed): the MALL-hit saving grows with the size up to about 128 MiB of
  one-IOD data and then stays flat to 288 MiB: **about 128 MiB per IOD**, and lines past it are not cached while the
  resident ones stay (no LRU collapse; the retention behaviour of 1.5). Reading the other IOD's 128 MiB in between
  does not evict them (0.669-0.891 with, 0.677-0.892 without).
- Which XCDs read one IOD's MALL-resident lines fastest: one 4-XCD half reads them 1.70 / 1.44x faster than the other
  half (warm), 1.26 / 1.24x cold, consistently per half and node; the calibration's IOD label and this split
  disagree in sign, so only the asymmetry is claimed.

### 1.4 Eviction under concurrent traffic

Prefetch 128 / 192 MiB, then X MiB of other traffic, then the stream (mb2):

| traffic in between | X = 16 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|
| `nt` reads (ATOM's weight GEMVs) | 0.86 / 0.89 | 0.88 / 0.89 | 0.87 / 0.89 | 0.88 / 0.89 | 0.87 / 0.88 |
| `nt` writes | 0.91 / 0.90 | 0.86 / 0.89 | 0.87 / 0.90 | 0.88 / 0.90 | 0.90 / 0.90 |
| default reads | 0.94 | 0.96 | 0.96 | 1.00 | 1.00 |
| `sc1` reads (mailbox / device-scope) | 0.94 | 0.96 | 0.96 | 1.00 | 1.00 |
| default writes | 0.94 | 0.96 | 0.96 | 1.00 | 0.99 |

(no traffic: 0.86 / 0.89). ATOM's mono weight streams are `nt` (MONO-STAGES 2.3): they cannot evict a prefetch.
Device-scope mailbox traffic is small (kilobytes a stage).

### 1.5 Survival in time, and retention

- A prefetch survives a gap with no memory traffic of up to 430 us unchanged (0.88-0.89 / 0.93-0.94 at gaps of 0,
  27, 108, 430 us; mb2 / mb5).
- A single 2 GiB default-policy streaming read did not evict lines that had been read twice (mb1's bimodal cold
  cells); reading a 192 MiB buffer three times did. The MALL's replacement keeps re-referenced lines against streams.

### 1.6 Co-residence with a persistent kernel, and graph concurrency

Occupant: 256 x 512, 149,520 B of LDS (K2 at 6 rows), 224 VGPRs forced (K2: 228), spinning; prefetcher on a second
stream / as a captured graph branch (fork, join):

- A prefetcher with tiny LDS and VGPRs that fit beside the occupant (t256 / t512 x 1 x 16 B, 28 VGPRs) starts every
  CTA within 3-11 us of the occupant and finishes while it still runs (overlap 1.00), in streams AND in a captured
  graph (HIP runs the forked branch concurrently).
- One with 102 VGPRs (4 x 16 B a lane) does not fit beside 2 x 224-VGPR waves: its CTAs wait for the occupant to end
  (start 530 us later). The ATOM prefetcher is therefore built at 46 VGPRs (granule 48 = what K2 leaves free).
- Co-resident prefetch rate: 128 MiB in 39 us (t256 x 1 x 16 B, 512 CTAs), 25-31 us with 4 x 16 B (when it fits).
- Ordering (mb8 co3, MALL-hit of the floor-quality stream after a prefetch issued as): captured graph branch
  0.948, second stream + event 0.936, same stream with a host synchronize 0.969, same stream back to back 0.950
  (with the 1 x 16 B latency-bound reader: 0.75-0.81). The prefetch is hit whichever way it is ordered.
- Graph concurrency with a layer chain (mb8 co4: 40 layer kernels with a prefetch branch forked after each, the
  shape of the side-stream mechanism): the captured graph's replay is 1.23-1.44x the main chain alone, and 1.34x with
  the prefetch inline. HIP's graph replay starts the next main kernel only after the forked branch kernel ends (no
  overlap in the replay's own timeline), and none of `GPU_MAX_HW_QUEUES`, `DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING`,
  `DEBUG_HIP_GRAPH_CLASSIC_PATH`, `DEBUG_CLR_GRAPH_PACKET_CAPTURE=0`, a high-priority replay stream changed it.
- A prefetcher beside a K2-sized occupant at 232 / 240 VGPRs (K2's timeline build at 6 rows: 233; mb11 co5): the
  30-VGPR register path and the 26-VGPR LDS-DMA path (depth 8-32) both start every CTA beside the occupant and finish
  while it runs; its reader then reads at 0.94-0.98 of cold (128 MiB, the co-resident shape).

### 1.7 The MoE-shaped ceiling (perfect prediction, same unit mapping)

The four MXFP4 regions of the touched experts at ATOM's TP2 per-rank layout, union U from W12 P4's captured routing
(22 / 38 / 62 / 93 experts at 6 / 12 / 24 / 48 rows), floor-quality `nt` stream; full prefetch of U (or of its first
experts) just before:

| rows | prefetch all of U: n01-09 (mb2) | n07-17 (mb5) | co-resident shape (t256): mb2 / mb5 | first 14 experts | first 7 |
|---|---|---|---|---|---|
| 6 | 0.842 | 0.874 | 0.897 / 0.880 | 0.899 / 0.932 | 0.897 / 0.950 |
| 12 | 0.894 | 0.892 | 0.920 / 0.897 | 0.943 / 0.960 | 0.961 / 0.975 |
| 24 | 0.930 | 0.943 | 0.943 / 0.933 | 0.959 / 0.997 | 0.977 / 1.031 |
| 48 | 0.956 | 0.963 | 0.957 / 0.956 | 0.974 / 0.999 | 0.971 / 1.013 |

This is the most a prefetch can take off ATOM's MoE stream, with a perfect predictor, nothing else on the GPU, and
the prefetch's unit mapping equal to the reader's (part L2): 13-16 % at 6 rows, 4-7 % at 24-48 rows. ug + down are
0.44 (6 rows) to 0.61 (48 rows) of K2's critical path (WP4 2.3), so the K2 ceiling is about 0.06-0.07 (6 rows) and
0.03 (48 rows) of K2.

## 2. Routing prediction on real traffic (step 2, measured)

Capture: an eager TP2 DSpark server (native path: its router is the fixed-order GEMV the mono runs, `moe.py`), a
`sitecustomize` hook (no ATOM edit, `t1/rpred/`), rank 0, every decode MoE call; GSM8K at C1 / C4 / C8 and
long prompts (8K and 24K tokens of GSM8K worked examples) at C1 / C4 and C1 / C2; job rp3 (n07-17), 178,944 calls.
U = the union of a call's top-k picks (what the MoE streams); P = a predicted set. Recall = |P & U| / |U| (bytes
equal per expert), precision = |P & U| / |P|. Sanity / gates: `self_exact` (the selection rule on the call's own
logits) 0.998-1.000; `self_x` (fp32 GEMV of the call's own input) 0.994-0.998; the must-fail predictors (a random
set of the same size; layer L-1's input through layer L+1's router) at chance (0.06 at 6 rows, 0.17 at 24, 0.26 at
48) in every cell.

Predictors of layer L's union:
- `EA_k<k>` (option (b), the early router): layer L-1's FFN input (normed) through layer L's router, top-k per row.
  Available right after K2(L-1): a window of K1(L) + compressor / Engram + K2(L)'s attention half.
- `EAh_k<k>`: the same from this rank's own half of the rows (what a token-parallel mono step holds in `ffn_normed`).
- `T`: the union layer L had the previous step (temporal; no compute).
- `HOT`: the |U| experts layer L picked most often so far.
- `EB_k<k>`: layer L's own attention input through layer L's router (after K1(L); captured on the draft only).

Target, GSM8K (recall / precision; |P| / |U| in brackets):

| predictor | 6 rows (C1) | 24 rows (C4) | 48 rows (C8) |
|---|---|---|---|
| EA_k6 | 0.72 / 0.71 (1.01) | 0.75 / 0.74 (1.01) | 0.77 / 0.76 (1.01) |
| EA_k8 | 0.80 / 0.60 (1.33) | 0.83 / 0.64 (1.30) | 0.85 / 0.67 (1.27) |
| EA_k12 | 0.87 / 0.45 (1.95) | 0.90 / 0.50 (1.79) | 0.91 / 0.54 (1.68) |
| EA_k16 | 0.90 / 0.36 (2.52) | 0.93 / 0.42 (2.20) | 0.94 / 0.48 (1.98) |
| EAh_k6 / k12 | - | 0.48 / 0.63 recall | 0.54 / 0.70 recall |
| T | 0.48 | 0.58 | 0.64 |
| T + EA_k6 | 0.82 / 0.52 (1.58) | 0.85 / 0.58 (1.48) | 0.88 / 0.62 (1.41) |
| HOT | 0.41 | 0.58 | 0.67 |
| random (must-fail) | 0.06 | 0.17 | 0.26 |

- Long prompts (24K tokens, C1, 6 rows): EA_k6 0.72, EA_k8 0.80, EA_k12 0.87, T 0.47 -- the same as GSM8K.
- The DSpark draft (128 experts, top-3, 5 rows; U = 7): EA_k6 0.70-0.75 (precision 0.37-0.46), T 0.54-0.64,
  T + EA_k6 0.84-0.88, EB_k6 0.62-0.67.
- Window (a) (the router result known, U exact, before ug starts; graph-mode stamps, c4 / c5): 8.4 us at 6 rows
  (0.094 of K2's span), 7.6 us at 24, 10.7 us at 48. A prefetch of the exact U (about 200 MiB at 6 rows) cannot fit
  in it: the shadow kernel below streams 216 MiB in about 127 us. So option (a) is out on the measured window, and
  the predictors above are what a prefetch can use.

## 3. Mechanisms (steps 3-4: built, gated, timed in graph mode)

Knob `ATOM_DSV41_RK_PREFETCH=<mode>[,opt=v]` (ATOM-dsv41 `atom/models/deepseek_v41/rk/prefetch.py`, the in-kernel
hooks `mono/kernels/rk_pfk1.py`; rocKE kernels `dsv41_mono/prefetch.py`), default off. Knob-off identity: K1 (6 / 12
rows), K2 (6 / 48 rows) and the draft K2 (5 rows) `.text` and `.ll` equal to pristine df86c482 (GPU-less dump job
d1), and the traced MLIR of this tree with the knob off equal to the W13 base stage.

| mode | what runs | prediction |
|---|---|---|
| `t` | after every K2(L), on a side stream (a forked graph branch): one rocKE kernel records K2(L)'s routing into a table and prefetches layer L+1's previous-step union | T |
| `rec` | the same kernel, recording only (the side kernel's own cost) | - |
| `ids` | `t` with every predicted id + 1 (the no-gain twin) | wrong |
| `ea`, `ta` | after K2(L): a rocKE router GEMV of K2(L)'s FFN input through layer L+1's router (fp32 accumulate, rel. err 2e-7 vs torch), then the prefetch of its top-`kp` (default 8) union (`ta`: united with T) | EA / T+EA |
| `worst` | at the step's start, every layer's whole last-step union, no budget | T, unbounded |
| `k1` | inside ATOM's K1(L) (its end, every CTA), the predicted experts of layer L; K2 records its routing at its end | T |
| `k2` | inside K2(L), on the CTAs idle until ug, just before the MoE | T |
| `shadow` | ONE persistent rocKE kernel a decode step on a high-priority stream (no graph node), launched before the replay, 1 wave a SIMD (26-30 VGPRs, 8.5 KiB LDS): polls K1(L)'s mailbox tag, records K2(L-1)'s route pairs, prefetches table row L; every wait bounded | T |

Gate (eager, `_CHECK`): every K2 launch is run alone and with the prefetch beside it from the same state, all
43,204,736 bytes K2 can write compared (scratch, peers, every full buffer). `t` (g0 / g1 / r0: 6080 / 4080 / 9600
launches a rank), `ids`, `ta`, `ea` (2560), `worst` (1760), `k1` (4800: ATOM's K1 build vs the knob build), `k2`
(6960), `shadow` (2360-2440), `shadow` + ids: 0 differing launches. Must-fail: the `write` mutant (stores the
prefetched words back, low bit flipped) 1720 / 1760 differing launches; the in-kernel `k2!bit` mutant caught
(3760); the shadow `write` mutant 1720-1760 / 1800: all caught. GPU unit test (pf_build): record / predict exact
at 6-48 rows, EA counts exact, MALL ids-mutant 0.99-1.01 of cold vs prefetch 0.88-0.90, shadow record / prefetch /
bounded-timeout exact (a race in the test's first launch fixed by a warm launch, c7 -> c8).

Graph-mode timings (ATOM_MONO_TIMELINE builds, FULL graphs, p1 closed loop C1 / C2 / C4 [/ C8]; every arm a fresh
engine; ratios vs the mean of the same job's two A-vs-A baselines b0 / b0b, whose own spread is 0.97-1.03 at C1 and
up to 0.96-1.04 at C2 / C4; stage ratios at 6 rows, rank 0, median of 6 steps x 40 layers):

| arm (job) | C1 tok/s | MoE (ug+down) | ug | K2 span | where the time goes |
|---|---|---|---|---|---|
| `t` (c4) | 0.62 (C2 0.66, C4 0.76) | 0.888 | 0.842 | 0.989 | the side kernel serializes: K2(L) end -> K1(L+1) start 78 us vs 1.3 |
| `rec` (c4) | 0.92 (C2 0.91, C4 0.99) | 1.007 | 1.017 | 1.003 | the side kernel alone: the K2 -> K1 gap 11.9 us |
| `ids` (c4) | 0.61 | 0.973 | 0.953 | 1.018 | as `t`; most of the MoE gain goes with the right ids |
| `t` at 48 rows (c4) | - | 0.950 | 0.938 | 0.992 | (down 0.953) |
| `k1` be 24 + side `rec` (c4) | 0.73 | 0.893 | 0.852 | 0.967 | |
| `k2` (c4) | 0.82 | 1.031 | 1.087 | 1.260 | |
| `ea` g256 (c4, d2) | hang | 0.881 | 0.815 | 1.000 | see below |
| `ea` g64 (d2) | 0.25 | 0.895 | 0.852 | 1.012 | prefetch span 421 us, serialized |
| `worst` g64 (d2) | 0.29 | 1.030 | 1.018 | 1.053 | |
| `k1` be 24 (c5) | 0.78 | 0.888 | 0.855 | 0.966 | K1(L) end -> K2(L) start 45.2 us vs 1.8 |
| `k1` be 6 (c5) | 0.92 | 0.964 | 0.937 | 0.995 | K1(L) end -> K2(L) start 19.6 us |
| `k2` (c5) | 0.84 | 1.034 | 1.104 | 1.289 | the prefetching CTAs reach `xq` 35 us late |
| `shadow` (c8) | 0.64 | 0.990 | 0.980 | 0.998 | see below |
| `shadow` + ids (c8) | 0.63 | 0.997 | 1.004 | 1.003 | |
| `shadow` g128 (c9) | 0.53 | 0.999 | 0.920 | 1.381 | every K1 / K2 launch: 96-128 CTAs start 25-30 us late |
| `shadow` g64 (c9) | 0.45 | 1.001 | 0.998 | 1.336 | as g128 (prefetch 444 us a layer, mostly after ug) |
| `shadow` g128 b48 (c9) | 0.54 | 0.969 | 0.919 | 1.284 | 5 experts, all before ug: the MoE gain is real |
| `shadow` g128 + ids (c9) | 0.51 | 1.143 | 1.134 | 1.450 | wrong ids: extra traffic, no hits |

Diagnoses (from the stamps, `k1diag.py` / `pf_overlap.py`):

- Side stream (`t`, `rec`, `ea`, `ta`, `worst`): HIP's FULL-graph replay runs the forked branch kernel and the next
  main kernel one after the other (1.6 / co4), so every prefetch byte is on the critical path. The MoE does get the
  MALL hits (`t`: MoE 0.892; `ea` with the early router: MoE 0.881, ug 0.815 -- the measured in-situ ceiling at 6
  rows, the MALL-only bound of 1.3 plus some L2), but the step pays the prefetch's whole span.
- `ea` / `ta` / `worst` with a 256-CTA side grid hang in graph replay at the first C1 decode after the C16 warm
  (NCCL broadcast watchdog, both ranks); the eager gate runs them clean, and the same kernels with a 64-CTA grid
  run (d2). A full-grid side kernel next to the persistent K1 / K2 (which need all 256 CUs co-resident) is the
  trigger; with 64 CTAs the mechanism runs and loses (0.25-0.29).
- In-kernel (`k1`, `k2`): a wave cannot retire while its loads are in flight, and K1's / K2's next phase waits on
  the same `vmcnt`, so an in-kernel prefetch costs its whole span on the issuing CTAs: K2(L) starts 19.6 us (6
  experts) / 45 us (24 experts) after K1(L)'s last stage, for 1-3.4 us saved in ug. In K2 the prefetching CTAs
  hold back the stages before the MoE (`xq` front +35 us).
- `shadow` (a structurally different delivery: no graph node, own queue, co-resident at 1 wave a SIMD): it does
  start 5.5 us after K1(0) and serves layers 0-1 (24 experts, 0.94-1.00 of its span before ug), but at layer 2
  the step stalls 2.7 ms between K1(2) and K2(2): the indexer / compressor kernels the original modules run between
  K1 and K2 on every 6th layer (2, 8, 14, ...) cannot be dispatched next to the shadow's 256 CTAs and wait for its
  bounded poll to time out (the shadow then aborts the step). Its prefetch also runs at 216 MiB in about 127 us,
  longer than a layer. Restructure (c9): a 128 / 64-CTA shadow leaves CUs free for those kernels -- the stall is
  gone and the shadow serves all 40 layers (with a 48 MiB budget, 5 experts, every one before ug: MoE 0.969, ug
  0.919) -- but now every K1 and K2 launch has 96-128 of its 256 CTAs start 25-30 us after the first (CTA start
  spread 0.3-0.4 us without the shadow; 24-27 us with it at g = 64 / 128 / 128 + b48 alike), so the step is 1.95-2.06x.
  Residency alone is not the cause: in isolation (mb12 co6, mb13 co7) neither a shadow-shaped spinner (0 / 4 / 8.5
  KiB LDS, 8 / 26 VGPRs, 64-256 CTAs) nor the real shadow kernel polling a tag that never comes (LDS and register
  paths, with and without `s_sleep`, beside a stream-launched or a graph-replayed chain of K1 / K2-shaped persistent
  occupants) delays a single occupant CTA: start spread 0.21-0.27 us in every arm, 0.24-0.25 us without. In situ
  (c10), a 128-CTA shadow with a ZERO budget (polls and records, never prefetches) costs the same as one that
  prefetches: step 1.94x, K1 / K2 start spread 23-24 us (64-95 late CTAs), vs 1.96x / 1.99x with 1 / 5 experts; a
  256-CTA zero-budget shadow has no late CTAs (0.37 us) but stalls at layer 2 (step 1.60x). So the late CTAs come from
  128 shadow CTAs polling ATOM's own mailbox line (K1's x8 pair 0, which K1 / K2 CTAs write and poll themselves) with
  `sc1` loads -- not from residency (co6 / co7: a dummy word, no delay) and not from the prefetch traffic.
  Restructure built (rocKE `ShadowKnobs(relay=True)`, ATOM `sk=shadow_r` / `shadow_rp127`, ABI + per-launch token):
  CTA 0 alone polls ATOM's word and relays `2 token + aborted` into a word of the shadow's own buffer that the other
  CTAs poll; plus a 127-cycle-sleep variant. Its run (c11: GPU unit test, gate, gtl) was blocked: the Austin home
  hit its disk quota (`Disk quota exceeded` on any write) before the stage could be uploaded. PENDING_QUOTA.

## 4. The ceiling

All from measured ratios (6 rows = C1 with DSpark 5; the graph-mode stamps of the b0 arms; 1.3, 1.7, 2):

| | 6 rows | 24 rows | 48 rows |
|---|---|---|---|
| MoE (ug + down) share of a decode layer (K1 + K2 + gaps; c9 b0 / b0b) | 0.34 | 0.46 | 0.49 |
| MALL-only MoE ratio with the exact union prefetched (1.3) | 0.889 | 0.942 | ~0.96 (1.7: 0.956-0.963 incl. L2) |
| -> step ceiling, perfect prediction, free prefetch | 0.962 | 0.973 | 0.980 |
| best predictor's byte recall (EA_k8 / EA_k12) | 0.80 / 0.87 | 0.83 / 0.90 | 0.85 / 0.91 |
| -> step ceiling with EA_k8, free prefetch | 0.970 | 0.978 | 0.983 |
| measured in-situ MoE ratio, best arm | 0.881 (`ea`, 24 experts) | - | 0.950 (`t`) |

"Free prefetch" is the catch. A prefetch re-reads from HBM the very bytes the MoE then reads from the MALL (about
200 MiB a layer a rank at 6 rows, about 9 MiB an expert): the MALL saves the MoE about 11 % of its stream, and the
prefetch costs a full extra read of it. It only pays inside HBM-idle time, and a mono decode layer has little: the
shadow kernel, with the whole layer to work in, streamed 216 MiB in about 127 us (a layer is about 117 us).
Prefetching only what fits (5 experts, 48 MiB: c9 `s128b`, all done before ug) gave MoE 0.969 / ug 0.919 -- about
1 % of the step at most -- and the delivery cost of every mechanism tried is 4-55 % of the step.

Cross-step residency of the hottest experts (no per-step prefetch) is bounded the same way: the MALL holds about
256 MiB, against about 7.9 GiB of routed expert bytes a step a rank at 6 rows (40 layers x about 200 MiB). So at most
about 3 % of the MoE bytes can be MALL hits, at the 1.12x MALL-hit rate: under 0.4 % of the MoE, about 0.15 % of the
step.

## 5. Served C1 / C2 / C4 / C8 TP2, both nodes, and stacked with WP1

OpenAI server + the B0 client (GSM8K prompts, max 1024 tokens), stage t1-atom-a7, a fresh server for every arm-run,
two passes with the arm order rotated, ratios of the per-request decode tok/s median (in brackets: aggregate output
tok/s) vs the mean of the same job's b0 / b0b (ATOM mono, knob off). `w1` = WP1 alone
(`ATOM_DSV41_RK_PREFILL_GRAPH=40-1024`), `k1sw` = `k1` (be 6) stacked on WP1. Jobs srv-n01-09 / srv-n07-17.

| arm | node | C1 | C2 | C4 | C8 |
|---|---|---|---|---|---|
| b0b / b0 (A-vs-A) | n01-09 | 0.979 | 0.967 | 1.028 | 1.024 |
| | n07-17 | 0.990 | 1.025 | 0.953 | 1.007 |
| `t` (side stream) | n01-09 | 0.641 (0.682) | 0.713 (0.734) | 0.766 (0.776) | 0.859 (0.823) |
| | n07-17 | 0.641 (0.687) | 0.616 (0.677) | 0.776 (0.782) | 0.897 (0.824) |
| `k1` be 6 (in K1) | n01-09 | 0.886 (0.913) | 0.923 (0.967) | 0.983 (0.944) | 0.956 (0.971) |
| | n07-17 | 0.893 (0.912) | 0.952 (0.938) | 0.979 (0.950) | 0.977 (0.953) |
| `shadow` g128 b48 | n01-09 | 0.550 (0.607) | 0.965 (0.871) | 0.974 (0.905) | 1.011 (0.909) |
| | n07-17 | 0.551 (0.614) | 0.849 (0.838) | 0.993 (0.905) | 1.022 (0.918) |
| `w1` (WP1 alone) | n01-09 | 0.932 (1.115) | 1.041 (1.252) | 1.163 (1.232) | 1.299 (1.274) |
| | n07-17 | 0.951 (1.109) | 1.008 (1.191) | 1.158 (1.205) | 1.294 (1.252) |
| `k1sw` (k1 on WP1) | n01-09 | 0.854 (0.986) | 0.958 (1.112) | 1.085 (1.177) | 1.281 (1.204) |
| | n07-17 | 0.837 (0.985) | 0.928 (1.093) | 1.103 (1.132) | 1.289 (1.229) |

Every T1 arm is slower than ATOM mono at C1-C4 on both nodes, and slower than WP1 alone when stacked on it (C1 0.85
vs 0.94, C4 1.09-1.10 vs 1.16, C8 1.28-1.29 vs 1.30). The shadow's C8 1.01-1.02 is inside the A-vs-A spread and its
aggregate is 0.91-0.92 (at C8 the shadow is off: K2's VGPRs leave it no room, and its per-step launch and graph-replay
hook cost remain). Acceptance length is unchanged in every arm (3.6-3.9). No arm wins, so no accuracy run was made
(GSM8K / passkey are gated on a claimed win); the gates of section 3 show every arm's outputs bitwise equal to ATOM's.

## 6. Verdict

REJECTED on base t1-atom-a7 (ATOM main df86c482 + W13 stage a9). All mechanisms stay as default-off knobs
(`ATOM_DSV41_RK_PREFETCH=t|rec|ids|ea|ta|worst|k1|k2|shadow`, knob-off byte-identical).

- The MALL works as a prefetch target: about 256 MiB (128 MiB per IOD), only non-`nt` loads allocate, ATOM's `nt`
  streams hit without evicting, a prefetch survives hundreds of us. But a MALL hit is only 1.11-1.14x HBM bandwidth
  for a floor-quality stream: the MALL-only ceiling is 11 % of the MoE at 6 rows, 6 % at 24.
- The early router predicts 72-91 % of the union's bytes; temporal 48-64 %.
- In situ the bytes do land (MoE 0.881-0.89 at 6 rows), but every delivery costs more than it saves: HIP graph replay
  serializes a side branch; an in-kernel prefetch holds its waves; a persistent shadow kernel blocks the
  indexer / compressor kernels and coincides with 25-30 us late CTA starts of every K1 / K2.
- Even a free mechanism would gain at most 3-4 % of a C1 step and 2 % at C4-C8, since the prefetch re-reads the MoE's
  bytes from HBM in a layer that has little idle HBM time.
