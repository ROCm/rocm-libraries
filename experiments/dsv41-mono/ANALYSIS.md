# Where DeepSeek-V4.1-Flash TP2 decode can still go (analysis, 2026-11-03, ratios only)

Inputs: probes/P1.md (step buckets, bytes floor), reverse/MONO-STAGES.md (ATOM mono K1/K2 code), wide/P4.md, D0.md,
wps/WP1-3.md, wps/WP6.md. Names: "rocKE hybrid" = our work inside ATOM; "ATOM mono" = ATOM's FlyDSL per-layer fused
kernels; "ATOM native" = the regular path.

## 1. The physics: what 3x would require

- The best served step (ATOM mono at C1-C4, native at C8-C16) is 2.1-2.4x its bytes floor (measured routing).
  Ceiling with zero latency waste and zero prefill stall: 2.4x (C1) to 3.05x (C16); at a realistic 85% of the HBM
  stream: 2.0x (C1) to 2.6x (C16).
- So 3x at C1-C4 is physically impossible on today's weights and acceptance length. It needs BOTH every latency
  bucket at its floor AND 1.3-1.5x fewer bytes per accepted token. At C8-C16 it is at the edge.
- The bytes are dominated by the routed MoE (0.58 of the floor at C1, 0.84-0.88 at C8-C16), which is already MXFP4 and
  already streamed at about 1.0x (C1, mono) to 1.2x (C4+) of its floor. Byte cuts elsewhere are small (WP3: heads,
  projections and shared expert to MXFP4/FP8 pass accuracy but are worth 2-3% today).

## 2. Where the time actually goes (TP2 DSpark, shares of the step)

| | mono C1 | mono C4 | mono C8 | C16 (native) | x its bytes floor |
|---|---|---|---|---|---|
| routed MoE | 0.22 | 0.44 | 0.45 | 0.47 | 0.9-1.2 (at floor) |
| CCL / all-reduce (incl. rank-skew wait) | **0.19** | 0.03 | 0.06 | 0.11 | latency |
| attention pre + post | 0.16 | 0.17 | 0.16 | 0.13 | 1.5-8 |
| mHC seams | 0.09 | 0.07 | 0.08 | 0.06 | 39-107 |
| attention core | 0.08 | 0.09 | 0.09 | 0.04 | 22-190 |
| indexer + compressor | 0.05 | 0.04 | 0.04 | 0.03 | 30-70 |
| router + shared expert | 0.06 | 0.05 | 0.04 | 0.08 | 1.5-34 |
| draft (backbone, head, Markov, KV) | 0.10 | 0.07 | 0.06 | 0.05 | 1.4-5 |
| heads, Engram, meta, sampler | 0.06 | 0.03 | 0.03 | 0.02 | about 1 |

Plus the foreign-prefill stall at C2-C16 (0.12-0.28 of a request's decode window), which WP1 already cut.

Reading: latency-bound buckets are 0.45-0.60 of the step at every CONC. That, plus the prefill stall, is the
whole headroom. Byte-bound buckets are already near their floors.

## 3. What the measurements so far say

| WP | lever | served result | lesson |
|---|---|---|---|
| WP1 | prefill as exact-length graphs | x1.17 (C4), x1.23-1.25 (C8), x1.32 (C16); TTFT 0.30-0.48; accuracy pass | the stall was real and big; short prompts only (graphs 40-1024) |
| WP2 | eager tail into the graph | GPU step -1.0-1.6%, served tie | host is off the critical path (inter-step gap 0.001) |
| WP3 | byte cuts | tie | the step is not byte-bound at C1-C4 |
| WP6 | rocKE wide dense GEMM, router fold, quant seam, ladder | tie or loss | dense GEMMs are 0.07-0.09 of the step; the MoE port was not attempted |

Two things the work so far has NOT attacked, and they are the largest levers left:

### 3a. The C1 CCL bucket (0.19 +/- 0.1 of the step) is mostly rank-skew waiting, not data movement
- ATOM mono's K2 `slice` / `reduce` stages sum the peer partials after polling for them; any imbalance between the
  two ranks lands here. Native's all-reduce is 0.05 at C1, so the in-kernel exchange is not cheaper; it is waiting.
- Known skew sources in ATOM mono: `step_begin` zeroes every mailbox and fences the peers every step (a cross-GPU
  fence on the critical path); the indexer stages run only on selecting layers but their 790-1209 SGPR spills are paid
  by every K2 build; no NUMA-homed copies, no holds/prepoll/spread (Qwen's biggest levers); the poll pattern peels the
  first iteration and sleeps (`s_sleep 1`) between polls.
- The P1 split comes from eager stamps (skewed ranks), so the true size needs a graph-mode, per-rank no-profiler
  stamp of K2 `slice`/`reduce` wait vs work. If even half of 0.19 is recoverable, that is about x1.10 at C1, more
  than any WP so far at C1.
- Owner: WP4 item 5 (schedule levers, epoch tags) + WP5 (CCL). Ask both to measure the wait component first.

### 3b. Latency chains inside K2 (seams, indexer, router, attention core: 0.25-0.30 of the step)
- These are serial chains of small stages with device-scope publish/poll hand-offs: mHC Sinkhorn seams at 39-107x
  their bytes, the indexer's 4-pass LDS radix top-k (two levels) plus candidate-block top-k, the route stage, and a
  bit-matched split-K attention (64 splits x 16 keys) kept for 6-row reproducibility.
- The levers are structural: fewer hand-offs (fuse stage boundaries, as the Qwen WP2a HC port did), a single-pass
  top-k, overlapping independent chains across ranks/CTAs, and (tolerance-class) a cheaper attention split.
- K2 split by layer kind (WP4 item 1) removes the spills from non-selecting layers: a prerequisite, not the win.

## 4. Things not yet in any WP that matter

1. **The routed-MoE restructure (C4-C16).** It is 0.44-0.47 of the step at 1.2x its floor. Moving it to 1.05x is
   about x1.07-1.08 at C8-C16. WP6 skipped it on a paper argument (the Qwen moe_lb2 builds reach 0.72-0.78 of floor
   speed). The Qwen lesson is the opposite: start from ATOM's winning structure (its union-U streaming, A8W4 MFMAs),
   ATT-profile it, then search. Also XCD/L2 homing of experts: at C16 the target touches about 134 distinct experts
   per layer and the draft another 28-49% of its own set.
2. **TP4.** ATOM's best served C1 number is TP4 mono (about 1.24x TP2 mono per request). Per-GPU bytes halve and the
   latency buckets do not; the CCL and skew levers matter more there. A TP4 target track would show the per-request
   ceiling.
3. **Long prompts.** WP1's graphs cover 40-1024 tokens. Real traffic with 8K+ prompts brings the prefill stall back
   (chunked eager prefill). A MIX path (prefill chunks as extra rows of the decode step) is the structural fix.
4. **Acceptance length.** AL is 3.6-3.9 of a possible 6. Every +10% AL is about +10% tok/s at near-zero step cost.
   Levers: DSpark confidence-scheduled ragged verify (ATOM V4.1 runs fixed width), tree/branch verify using the
   spare rows at C1 (the C1 step is latency-bound, so extra verify rows are almost free there).
5. **rocKE mono (Path A).** Only worth it if a single persistent step kernel lets independent chains (draft vs target
   tail, attention vs MoE of the next layer, rank skew) overlap; launch savings are nil (inter-step gap 0.001).

## 5. Realistic outlook (TP2, per request, vs ATOM's best served path)

| | C1 | C4 | C8 | C16 |
|---|---|---|---|---|
| measured so far (WP1) | 1.0 | 1.17 | 1.23 | 1.32 |
| + skew/CCL + K2 latency chains (WP4/WP5 if they land) | 1.15-1.35 | 1.3-1.45 | 1.35-1.5 | 1.4-1.55 |
| + MoE restructure + byte cuts re-tested on that base | 1.2-1.4 | 1.4-1.6 | 1.45-1.65 | 1.5-1.7 |
| physics ceiling at 85% stream (no AL change) | 2.0 | 2.3 | 2.55 | 2.6 |

3x needs an acceptance-length or verify-structure gain on top of everything above.

## 6. Correction after WP4's graph-mode stage split (wps/WP4.md section 2.3)

- The C1 "CCL 0.19" bucket of P1 was an eager-mode artefact. In FULL graphs (job wp4g2), slice + reduce sit inside
  the 0.11-0.19 of K2 that all the small seam stages (gate, route, slice, norm, xq, router, reduce) take together.
  Section 3a's skew hypothesis is refuted; the NUMA ablation bounds the whole hop-locality term at 3-6% of the step.
- Graph-mode K2 critical path: routed MoE (ug + down) 0.44 (6 rows) to 0.61 (48 rows), at the byte floor for the
  experts touched; GEMVs (wo_a, wo_b, shared) 0.13-0.19; attention (irq + score) 0.12-0.15; indexer 0.02-0.04; seams
  0.11-0.19. K2 is 0.64-0.80 of the target forward's GPU time.
- So ATOM mono V4.1 is close to its structural optimum for its algorithm. The remaining big lever is overlap, not
  faster stages: **prefetch the next experts' weights into the 256 MiB MALL during the attention / seam window**,
  when the HBM stream is idle (needs an early routing estimate, e.g. the next layer's router on the current hidden
  state; accuracy-neutral because it only prefetches). Ceiling: the MoE's byte time that fits under the latency
  window, plausibly 0.15-0.30 of K2 at 6-24 rows.
- Other structural levers: fewer hops per layer (fuse slice -> norm -> xq -> router), acceptance length / tree or
  ragged verify, long-prompt MIX prefill, TP4.
