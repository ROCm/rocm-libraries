# Where rocKE can win on DeepSeek-V4.1-Flash decode (gfx950): opportunities, baselines, first probes

R0 synthesis, 2026-11-02. Inputs: `model.md`, `atom-v41-runtime.md`, `atom-v41-mono.md` (this directory), and the
Qwen3.8 record (`experiments/qwen38-true-mono/`: PLAN.md, D0.md, DESIGN.md, DESIGN-CCL.md section 10, DESIGN-LB.md,
XCD-SCHEDULING.md, HANDOFF.md, research/q38-learnings.md, research/flytape.md; FINAL-STACK.md does not exist yet).
Ratios only. Below 1 means faster for time. Every gain below is a prior (estimate), not a measurement; P1 replaces
the priors with a measured round decomposition.

---

## 0. Summary

1. **The competitor is ATOM's own FlyDSL mono, and it is probably on by default.** The recipe says the V4.1 mono is
   "refused (MXFP4 activations in K2b)". That text is stale: on `origin/main` K2b is A8W4 and the bind gate passes
   for the shipped MoE (`atom-v41-mono.md` section 5). Any no-EP TP2/TP4 deployment with bf16 KV, FULL graph (or
   eager), DSpark and at most 48 rows per step therefore runs the mono. That covers DSpark at CONC <= 8, and the
   AgentX recipe's flags pass `config_refusal`. Every baseline needs explicit `ATOM_MONO_ENABLE=0/1` arms and a log
   check (`"V4.1 mono decode on"` vs `"... off: <why>"`).
2. **There is a lot of headroom at small batch, and it is not bandwidth.** By the private-perf estimate, served
   TP2 CONC 1 runs at about 0.2 of the bytes floor, with a metric caveat. The recipe's mono step is about 0.34 of
   the original's at 1 request. TP4 beats TP2 by only 1.11x per user at CONC 2. So small-batch decode is bound by
   fixed cost and latency, as Qwen was (2.3x its issued-bytes floor, CTAs polling 0.60-0.75 of the time). The
   ordering that worked on Qwen should work here: schedule and sync levers first, then the round boundary and
   host, then restructured stages. Straight ports gave parity on Qwen.
3. **ATOM's V4.1 mono has none of the Qwen schedule levers.** It has no NUMA dual-copy mailboxes, no holds, no
   XCD placement and no prepare-ahead. It runs two launches per layer (80 per step) plus native compressor,
   Engram, candidate-table, step_begin and step_meta launches. The eager tail (head at TP >= 2, sampler, broadcasts,
   draft KV, commit) and the draft graph sit outside it. Each of these items had a measured Qwen analogue.
4. **Plan of record (proposed): Path B first, then Path A.**
   - Path B: rocKE per-layer kernels, one launch per layer once the compressor and candidate table move into K2,
     plus one persistent round-tail kernel (head, argmax, verify, commit, draft).
   - Path A: a persistent whole-pass kernel, gated by the D0.7-style switch rule re-measured on V4.1 bodies.
   - Persistent whole-run (FlyTape-style request loop): optional, last.
5. **TP is mandatory.** The weights do not fit one MI355X, so TP2 is the smallest cell. Our CCL (IPC peer
   mailboxes, lamport one-shot, rank-ordered f32 sums, epoch tags with no per-launch reset) applies directly.
   ATOM's mono already reduces in-kernel, so the CCL win is in the details: no per-step zero and fence, NUMA homing,
   and overlap. It is not a "kernel vs aiter" win.
6. **First probes, by gain/cost:**
   - P1: round decomposition of ATOM mono vs native.
   - P2: NUMA mailboxes and holds on ATOM's V4.1 mono.
   - P3: round boundary and host prepare-ahead, plus folding the eager tail into the graph.
   - P4: K2b null-bytes MoE build, then a fused expert-major rocKE MoE.
   - P5: rocKE head (vocab-parallel GEMV, fused argmax, CCL argmax merge) inside the graph at TP >= 2.

---

## 1. Facts that drive the ranking

| Fact | Consequence | Source |
|---|---|---|
| GPU weights are about 1.06x one GPU's HBM | TP1 is impossible. TP2 is the minimum and TP4 the primary. TP8 does not start natively (shared `w2` K=288). | `model.md` 5 |
| Bytes per B=1 token: attention projections 0.39, routed 0.35, shared 0.11, head 0.10, Engram `wkv` 0.024, router 0.012, mHC 0.012, indexer and compressor 0.006 | The byte levers are the FP8 32x32 GEMVs and the A8W4 MoE. KV bytes are negligible (0.25% at 8K). | `model.md` 6.2 |
| Per-GPU floor at B=1: TP2 0.54, TP4 0.32 of TP1 (TP-sharded experts). EP: TP2 0.60, TP4 0.39. | TP-sharded experts beat EP at small batch. EP's busiest rank holds 2.8 experts vs 1.5 at TP4. | `model.md` 7 |
| DSpark: 5-token chain, 6-row verify. Spec-step bytes 2.6-2.9x a plain step (uniform routing). Acceptance about 0.50 per draft token, i.e. about 3.5 tokens per step. | Bytes per token 0.75-0.82 of plain decode, but the latency win is large because fixed cost dominates. Correlated routing lowers verify bytes further. | `model.md` 7, `atom-v41-runtime.md` 3 |
| Mono domain: at most 48 rows (8 requests x 6) and at most 8 draft requests, no EP, TP 2/4/8 | Above CONC 8 with DSpark, native runs. Large batch is a separate front. | `atom-v41-mono.md` 2.1 |
| Native target graph: about 1.15-1.4k kernel nodes (estimate). Mono target: 80 persistent launches plus native ops. | The native path is launch-bound. The mono already removed most of that. | `atom-v41-runtime.md` 4 |
| At TP >= 2 the head, sampler, rejection, broadcasts, draft KV and commit run eagerly outside the graphs. The step is two graph replays plus that glue. Host waits on the previous step's MTP status in `prepare_input_ids`. | This is the same host boundary Qwen hit, with heavier V4.1 metadata (compress plans, spans, Engram, FP4 walk plans). | `atom-v41-runtime.md` 7 |
| No KV rollback: rings have 133 rows for 128 visible, and a commit is one cursor kernel | Accept/reject is device-friendly, unlike Qwen's GDN state snapshots. | `atom-v41-runtime.md` 3 |
| Cross-layer KV and selection sharing: only 8 distinct top-512 selections per token across 38 layers | A persistent kernel can keep a group's selected rows resident in L2/MALL. That is a latency lever more than a bytes lever. | `model.md` 2.1, 6.2 |

---

## 2. Stage-by-stage rocKE port candidates

Stage names follow ATOM's mono (`atom-v41-mono.md` section 3). "Byte share" is the B=1 weight-byte share. "Qwen
evidence" is the closest measured analogue (`q38-learnings.md`). Gains are priors on the stage's own time.

| # | Stage (ATOM mono) | Byte share / role | What rocKE brings | Qwen evidence | Prior gain on the stage | Cost | Risk |
|---|---|---|---|---|---|---|---|
| S1 | **K2b routed MoE** (xq, router, route, ug, down, reduce; A8W4, TP-sharded intermediate) | 0.35 at B=1; dominant from B >= 8 | Fused expert-major restructure: shared expert during the router window, balanced byte lines per IOD, down issued after the gate_up epilogue, rank-ordered combine. MXFP8 act quant with the V4.1 rounding rules (`mx.code_round` vs `code_ceil` per stage). | Fused MoE 0.78-0.85 of mono MoE. A straight port gave parity. Null-bytes build: fixed overhead was 0.625 (r4s) of mono's MoE. | 0.80-0.90 | High (about 1.6k lines in K2b; 384 experts; 6/12/48-row builds; TP2/TP4 intermediate 1152 / 640 padded) | Spills (K2b already at the 48-row limit), the router-window fixed cost, bitwise top-k order |
| S2 | **Attention projections** (`wqkv_a`, `wq_b` in K1; `wo_a`, `wo_b` in K2a) | 0.39 | Wave-owned FP8 32x32 GEMVs with all loads in flight, prefetch before the poll, `kernarg_preload`, NT loads, tail-balanced CTA counts. `wo_b` pushes its partial straight into the CCL. | Head GEMV 0.77-0.84 (mono fetched 1.23x bytes, 45-47% at `vmcnt(0)`). P-L3 out-proj at parity with FlyTape. | 0.85-0.95 if the mono over-fetches or stalls; parity if it streams | Medium (GEMV emitters exist: `mx_decode_gemv`, `large_batch/gemm.py`) | Must first measure issued vs unique bytes (P1) |
| S3 | **mHC seams** (K1 / K2a `slice` + `gate`: owed post, FP32 `hc_fn` GEMV, 20-iteration Sinkhorn, collapse, norm) | 0.012, but serial at the head of each sublayer | Spread the per-CTA norm and pack (Qwen HC spread), DPP sums, hold `hc_fn` in registers across the wait, critical-path island. Non-bitwise: bf16 `hc_fn` (accuracy-gated). | P3 HC spread 0.969-0.970 per r1 layer. HC A1 sat on the critical path (Coz entry slope 0.77-1.08). | 0.95-0.97 per layer | Medium | Sinkhorn order bitwise; 2 seams per layer |
| S4 | **Head plus argmax** (target: eager at TP >= 2; draft: inside the draft graph) | 0.10 (BF16 129280 x 5120) | Vocab-parallel GEMV with fused argmax, CCL argmax merge (pair form), inside the graph or the tail kernel. Non-bitwise: FP8/MXFP4 head (accuracy-gated, Qwen N8-style). | Head 0.77-0.84 of mono; one-launch argmax fold | 0.80 on the head, plus removal of eager launches | Low-medium | Logits needed for non-greedy sampling (Qwen WP8) |
| S5 | **Indexer** (K1 `iw`/`iq`/`iquant`; K2a `iscore`, `ilist` radix top-k, IRDY; 8 layers) | 0.006 bytes; latency chain grows with context | Wider score fan-out and an earlier top-k merge tree; keep logits in LDS/L2 instead of device-scope global; FP4 walk. Group-aware: one selection serves 4-6 layers. | Qwen QSA Q-chain 0.36 -> 0.58 of a QSA launch from 1K to max context (P8): a long-context latency term. | Unknown; grows with context | Medium-high (bitwise ties, two-level candidates) | Bitwise top-k with lowest-index ties |
| S6 | **Sparse attention** (K2a score + irq; 512 selected + 128 window rows, K=V, 64 heads) | KV less than 0.3% | Group-level residency of the 512 selected rows across the 4-6 layers of an index group (one XCD per group's attention tasks); fuse inverse RoPE + quant (already fused). | No direct analogue | Latency only | Medium | ATOM bit-matches a 6-row order; must keep row independence |
| S7 | **Out-of-kernel natives** (compressor on 4 FULL layers, `candidate_block_table` on L20, `step_meta`, `step_begin`, embed/collapse) | small bytes; extra launches, and the reason K1/K2 are split | Fold the compressor and candidate table into K2 so each layer is one launch. Epoch tags (`step*64 + layer + 1` or the CCL epoch word) remove `step_begin`'s zero and fence. Do step_meta on device inside the first kernel. | Launch boundary was about 0.02 of a Qwen round. step_begin zero plus fence is one more system-scope launch per step here. | Removes about 45 launches per step | Medium | Compressor FP32 tails and boundary semantics; Engram ordering |
| S8 | **Engram** (L1, L14: UVA host-table gather, 157 MB FP8 `wkv`, gating) | 0.024 | `wkv` GEMV in-kernel; keep the UVA gather off the critical path (prefetch for step t+1 during step t, since ids are known one step early under deferred ids). | None | Small | Medium | Host RAM per node (about 189 GiB of tables) |
| S9 | **Round tail** (norm, head, vocab gather, sampler/rejection, TP broadcasts, `compute_draft_kv`, commit, next-step `fill_deferred_decode_ids`) | launches plus host | One persistent tail kernel (Qwen Path B). Verify/accept compare on device, CCL broadcast replaced by rank-identical device compute, draft-KV write fused. | P9: restructured tail wins (verify chain 0.795, draft 0.848-0.890). A straight port lost. | Large share of non-layer time (P1 decides) | Medium | Non-greedy sampling parity |
| S10 | **DSpark draft** (3 window-only blocks over [B,5] rows with 128-expert top-3 MoE, head, 5 sequential Markov argmax steps each with a TP exchange, confidence head) | about 0.07 of spec-step bytes | Run inside the tail kernel. The Markov chain becomes 5 device-local steps with pair-form `(max,id)` CCL merges instead of 5 Triton launches plus exchanges. | Qwen draft x3 eager layers were part of the 0.11-0.14 non-layer share | 0.85-0.90 on the draft | Medium | The draft is already not bitwise in ATOM's mono (key order) |

Byte-only work cannot win at small batch (Qwen pitfall 1). S1, S2 and S4 pay only where P1 shows over-fetch or
stalls. S3, S7 and S9 are latency and structure levers and are the safer early bets.

---

## 3. Persistent true mono vs per-layer

| Option | What it is | For | Against (Qwen evidence) |
|---|---|---|---|
| ATOM today | 2 FlyDSL launches per layer (K1, K2) plus about 45 native launches, inside one FULL graph; draft graph; eager tail | Works, bitwise | 80+ launches; native ops between K1 and K2; tail and host outside |
| **Path B (recommend first)** | rocKE `layers=1` kernels, **one launch per layer** (compressor and candidate table folded into K2, Engram `wkv` folded into K1 on L1/L14), plus **one persistent tail launch** (head, argmax, verify, commit, draft, Markov, next-step meta) | D0.1 chose this for Qwen. The tail restructure won (P9). A launch boundary costs only about 0.02-0.08 of a layer. Graph-friendly with ATOM's FULL capture. | Each layer re-enters; the cross-layer prefetch lever is lost |
| Path A (`pass` / `round`) | One persistent kernel for the 40-layer pass, or the whole round including the draft | No boundaries; cross-layer prefetch; per-layer peer fence disappears | Qwen P1: the loop body cost 1.003-1.020 per layer, and a 2-layer stack lost 1.02-1.04 (fences, next-layer waits on all tiles, contention). Needs layer-parity scratch and epoch tags (V4.1's `layer + 1` tags are zeroed each step). |
| Whole-run (FlyTape) | Device request loop, host polls a cursor | No host in the loop | Qwen: worth about 0.01 of a round over `round` mode, never on the critical path. V4.1 Engram UVA and prefix-cache metadata make an on-device request loop hard. |

V4.1 specifics that shift the call toward A, so re-test it:
- At TP >= 2 every launch boundary also orders the peers, and ATOM pays a per-step `step_begin` zero plus fence.
- 40 layers give fewer boundaries than Qwen's 48, but each layer has 2 launches today.
- Index groups (4-6 layers sharing a selection) give a natural cross-layer residency lever that only Path A captures.

Switch rule, as D0.7: measure the loop-body cost x per layer on the rocKE bodies. Switch to A only if
x <= 0.99 at the 6-row verify width and x <= 1.00 at 1 row, on both nodes.

---

## 4. CCL reuse: TP2/TP4 all-reduce, vocab merges, EP all-to-all

**What exists (ours):** `qwen38_mono/ccl/` gives IPC peer mailboxes, three wire forms (pair, line, lamport),
one-shot and two-shot all-reduce, all-gather and argmax merge. Epochs are advanced by the kernel, with no host
reset, deadline polls and abort words. Host runtime: `rocke.runtime.peer`. Measured (DESIGN-CCL section 10):
- In situ at TP2 rows 5-128: lamport one-shot with a consumer pass at 0.83-0.93 of GEMM + aiter.
- At 1 row: a tie or a loss against aiter (the phase barrier dominates).
- TP4 (provisional): wins at rows 8-16 (0.87-0.94), loses at rows 32-128 (1.01-1.12). Two-shot wins at 64-128.

**What ATOM's V4.1 mono already does:** in-kernel push of bf16 partials to every peer's uncached IPC buffer at
system scope, summed in rank order. It reduce-scatters to token owners at S >= 12 (token parallel), with an aiter
custom all-gather at the end. The bar is therefore ATOM's in-kernel exchange, not aiter's separate kernel.

**Exchanges per step (V4.1):**
- 2 per layer: the `wo_b` and MoE-down partials, `rows x 5120 x bf16` = 10 KiB per row, 60 KiB at the 6-row
  verify, 480 KiB at 48 rows.
- The embedding all-reduce and the head vocab gather.
- Draft: 3 blocks x 2 plus 5 Markov `(max,id)` exchanges.
- 2-3 postprocess broadcasts.

| Lever | Expected effect | Basis |
|---|---|---|
| Epoch-tagged lamport / pair forms (no per-step zero, no `step_begin` fence launch) | Removes one system-scope launch plus a peer fence per step | Our CCL "no per-launch reset"; ATOM `step_begin` exists because its tags restart each step |
| NUMA homing of receive buffers and consumer-local polls | Up to the Qwen NUMA gain on the exchange hop (remote-homed round trip 1.48-1.50x) | DESIGN-CCL H5 is not verified on uncached memory (WP-C6 open) |
| Overlap: the next stage starts on the tiles whose exchange landed (tile flags), not the whole vector | Hides link time. This is DESIGN-CCL 10.3's restated target. | Exposed cost at 16 rows and up was 0.89-1.01 of link time |
| Pair-form argmax merges for the head and the 5 Markov steps; device-identical sampling instead of broadcasts | Removes about 8 small collectives per step from the critical path | CCL `ArgmaxSpec` exists |
| `ar_dtype=fp8` for the partials (accuracy-gated) | 0.53x exchange bytes | DESIGN-CCL 10.3 (not built) |
| Two-shot at TP4 for 48-row steps | 0.88-0.96 at 64-128 rows (TP4, provisional) | DESIGN-CCL 10.5 |

**EP all-to-all:** not needed for the mono domain.
- Mono refuses EP, and TP-sharded experts beat EP in bytes at small batch (`model.md` 7).
- EP matters for large batch (TP4 + EP is the docs' primary deployment; DPA4 + MoRI EP4 for throughput), with
  load imbalance and dispatch/combine payloads.
- It is new CCL work (a non-goal in DESIGN-CCL section 1): a dispatch of routed rows to expert owners and a
  rank-ordered combine. It reuses the line form and the `lay=unit` combine arena from DESIGN-CCL 10.4.
- Defer it to the large-batch phase. Do it only if a TP-sharded large-batch rocKE MoE loses to EP at B >= 64.

---

## 5. XCD scheduling (from `XCD-SCHEDULING.md`, applied to V4.1)

In order of evidence:

1. **NUMA dual-copy mailboxes** for all intra-GPU hand-offs. This was Qwen's biggest single lever (served round
   0.930, layer 0.89-0.95), and ATOM's V4.1 mono has none. It is calibrated by the ping-pong home-IOD test.
2. **Stage-entry holds.** Off-critical stages that start early steal HBM.
   - Candidates: the shared-expert stream versus the router window, indexer `iq` versus `wq_b`, next-sublayer
     `hc_fn` loads.
   - Find them with Coz entry slopes.
3. **Critical-path islands** for V4.1's serial chains, homed on one IOD: mHC gate (Sinkhorn), router -> route,
   `ilist` top-k levels, and the attention irq -> `wo_a` hop.
4. **Expert-to-XCD homing** (384 experts, 48 per XCD) by a balanced id hash. Then gate_up -> down of one expert stays
   in one XCD L2 and uses the intra-XCD protocol (no `wbl2 sc1`, per P5).
5. **Index-group residency.** Place a group's attention score tasks on fixed XCDs so the group's 512 selected rows
   (512 KB) stay L2-resident across its 4-6 layers. This is a V4.1-specific lever, and only useful under Path A or
   with a MALL hit under Path B.
6. Known losers. Keep them as knobs but do not lead with them: global dynamic tickets, simulator placement,
   prefetching last step's experts into MALL, and `xcd_blocked` placement.

---

## 6. DSpark ragged verify as a device-side scheduling problem

Today:
- ATOM verifies a rectangular [B, 6]. The ragged and confidence-scheduled width `max(ell_i, bonus_i) + 1` only pays
  under eager or PIECEWISE, because a FULL graph replays a fixed width (`atom-v41-runtime.md` 3, H11).
- It also needs a calibration profile.

In a persistent kernel the width is data, not a graph key:
- **Row skipping:** the tail kernel computes each request's verify width from the confidence head (or from
  `ell` kept two generations back, as `VerifyScheduler` does) and writes it into the next step's device meta.
  - Dense GEMVs at 6 rows or fewer are byte-bound, so skipping rows there saves little.
  - In the MoE, the union U of experts shrinks with the rows. Uniform routing touches 34.6 experts for 6 rows vs
    6 for 1. Correlated routing touches fewer, but it is still the largest byte term of the verify.
  - The indexer and attention also scale per row.
- **Width-aware routing union:** pick the active rows first, then route, then stream only the union U. This is the
  same structure as the Qwen fused MoE's slot list; inactive rows issue no loads (the Qwen "invalid chains issue no
  loads" win).
- **Draft/verify overlap** (XCD lever 9):
  - The draft for step t+1 needs only the target's layer 37-39 inputs and the accepted prefix.
  - The 3 draft blocks could start on a disjoint XCD set while the target's last layers and the head finish.
  - The cost is contention, so it needs holds.
- **Batch-level scheduling:** at CONC 2-8 the 48-row mono build already runs one width per step. A device-side
  planner can choose the build (6/12/24/48) from the live rows. This avoids padding to a larger ladder rung, which
  ATOM's FULL graph cannot avoid (H8).
- Bound:
  - Expected verify bytes fall roughly with U(rows).
  - The gain is capped by acceptance: a request accepting 5 needs all 6 rows.
  - At acceptance about 3.5, about 0.4-0.6 of verify rows are discardable in hindsight, but the confidence
    schedule has to predict them.
  - Treat this as an accuracy-neutral but acceptance-sensitive lever. Measure acceptance with real prompts
    (GSM8K), not the AgentX synthetic 3.51.

---

## 7. Risks

| Risk | Mitigation |
|---|---|
| **Hardware:** every experiment needs 2 or 4 gfx950 GPUs on one node; Qwen budgets assumed TP1 jobs | Pack 8-GPU rocke nodes with two TP4 or four TP2 jobs. Microbench stages at TP2 first; TP4 for gates. |
| **Host RAM and storage:** about 189 GiB of Engram tables in host memory per node (UVA), plus about 285 GiB of GPU weights; about 475 GiB checkpoint | B0 scout: node RAM, node-local `/tmp` capacity, staging time. Do not stage on NFS home (alola rule). |
| Stale docs (mono "refused", "12 rows") | Verify on hardware from logs before any baseline is believed |
| **Bitwise contract:** FP8 32x32 blocks, two MXFP8 rounding rules (`code_round` / `code_ceil`) chosen per stage, 6-row attention order, sorted top-k ties | Reverse-engineer K1/K2 `.ll`/ISA per stage (PLAN decision 16 method). Per-stage tolerance gates where non-bitwise is allowed. |
| Persistent kernels with scratch can deadlock (ATOM refuses widths that spill); K2b already limits rows to 48 | VGPR/spill gate on every build (Qwen pitfall 8) |
| TP4 CCL is the least mature part of ours (one node, provisional; loses at 32-128 rows) | Gate TP4 on two nodes. TP2 first. |
| Side streams and the Engram all-gather deadlock beside persistent kernels | Keep `ATOM_DSV41_SIDE_STREAMS=0`. Join Engram before the first persistent launch, or move it in-kernel. |
| ATOM main moves (V4.1 mono last touched 10-16; the shared framework is shared with GLM/Kimi/MiniMax) | Pin an ATOM main sha in the separate clone (PLAN decision 1). Work in that clone, never in dev/ATOM. |
| Accuracy: DSpark acceptance is the real metric multiplier; AgentX uses synthetic acceptance | Real-prompt acceptance (GSM8K), plus long-context (RULER 8K/32K/128K; the model supports 1M) for the indexer and candidate paths |
| FP4 index plane: eager fallback on score-workspace overflow (`step_needs_eager`) | Watch every trace. Both fp8 and fp4 index arms. |
| Large batch (more than 48 rows) is outside the mono and is a different problem (U(B) dominates: B=128 reads 87% of experts) | Separate phase, reusing DESIGN-LB; not in the first cells |

---

## 8. Recommended first baseline cells (B0 runs them; ratios reported within one job)

Common settings: ATOM main (pinned sha, separate clone), `--level 3 --cudagraph-mode FULL`, `--kv_cache_dtype bf16`,
no EP, `ATOM_DSV41_SIDE_STREAMS=0`, the capture ladder covering every size 1-8. For every mono-on arm, confirm in
the log that mono actually ran.

| Priority | TP | Index | DSpark | CONC | Arms | Why |
|---|---|---|---|---|---|---|
| 1 | TP4 | fp8 | on (5) | 1, 2, 4, 8 | mono off / mono on | Recipe's mono cell; the mono domain up to 48 rows |
| 1 | TP2 | fp8 | on (5) | 1, 2, 4, 8 | mono off / mono on | Smallest config; the AgentX TP2 c1 cell is the headroom case |
| 2 | TP4, TP2 | fp8 | off (K=0) | 1, 4 | mono off / mono on | Plain decode; isolates the DSpark tail and draft |
| 2 | TP4, TP2 | fp4 | on | 1, 8 | mono on | AgentX flags; FP4 walk planner and eager-fallback watch |
| 3 | TP4 + EP | fp8 | on | 1, 8, 32 | native only (mono refuses EP) | The docs' "primary deployment" reference |
| 3 | TP2 x 2 (DP2 on 4 GPUs) vs TP4 | fp8 | on | 8, 16, 32 | native and mono | PLAN decision 5 fairness (matched GPU count). DP replaces the impossible TP1. |
| 4 | TP4, TP2 | fp8 | on | 16, 32, 64 | native (out of the mono domain) | Large-batch context only |

Workloads:
- GSM8K, 32 prompts, for real acceptance and the quick metric (the Qwen `quick_tps.sh` pattern).
- The AgentX synthetic acceptance 3.51 for comparison with the recipe.

Metrics: per-request decode tok/s, step time from the graph trace, acceptance length, and GPU idle between replays.

---

## 9. First probes (ranked by expected gain / cost)

| Rank | Probe | What it measures or builds | Expected gain (prior) | Cost | Decides |
|---|---|---|---|---|---|
| **1** | **P1: round decomposition** of mono-on vs mono-off at TP2/TP4, 1 and 8 requests: `ATOM_MONO_TIMELINE` stamps per K1/K2 stage, a kineto/offline trace, PMC FETCH_SIZE per kernel vs the unique-bytes model of `model.md` | Shares of target K1/K2, native in-between ops, draft graph, eager tail, host idle at the boundary. Issued vs unique bytes per stage. CTA poll fraction. | None directly; it sets every other budget | Low (no kernel work; reuse the Qwen harness: `timeline.py`, `round_decomp.py`, `bytes_model.py`, `gap_table.py`) | Whether the next lever is schedule, host/tail or bytes |
| **2** | **P2: NUMA dual-copy mailboxes plus holds on ATOM's V4.1 mono** (patch `atom/mono/device/sync.py` and the V4.1 regions in the ATOM clone; ping-pong home-IOD calibration; Coz entry slopes for holds) | The Qwen schedule levers on V4.1's mono layers, bitwise | Qwen: 0.86-0.95 per layer, 0.93 served round. Prior for V4.1: 0.90-0.95 of the target pass. | Low-medium (the ports exist: `ATOM_Q38_SCHED`, calibration, `sched.BEST`) | How much of the in-layer gap is sync/placement. Also gives a stronger competitor to beat later. |
| **3** | **P3: round boundary and host prepare-ahead** (an `RB_DEVMETA` analogue: accept-dependent V4.1 metadata shifted on device; head + argmax + verify + commit + draft-KV moved into the graph at TP >= 2) | Host idle per step, eager launches between replays | Qwen: host idle 0.07-0.08 of a round, and kernel savings only converted after the boundary moved. V4.1 metadata is heavier, so the prior is 0.85-0.93 of the step at CONC 1. | Medium (ATOM runner work in the clone; Python only) | Whether in-kernel savings will show end to end |
| **4** | **P4: K2b MoE null-bytes build, then a fused expert-major rocKE MoE** at 6/12/48 rows, TP2/TP4 | The fixed overhead F of the router window and route table, then a restructured MoE (Qwen `moe_fused` design, V4.1 shapes) | Qwen fused MoE 0.78-0.85 of mono MoE. The MoE is about 0.35-0.45 of target-pass bytes at 6 rows, so the prior is 0.92-0.95 of the pass. | Medium for the null build; high for the port | Whether the MoE is bytes-bound or fixed-cost-bound here |
| **5** | **P5: rocKE head in the graph** (vocab-parallel BF16 GEMV + fused argmax + CCL pair argmax merge; also serves the draft head and the 5 Markov steps) | Head time vs aiter/tgemm; removal of the eager head at TP >= 2 | Qwen head 0.77-0.84. Prior 0.95-0.97 of the step. | Low-medium (`mx_decode_gemv` exists; BF16 variant and vocab sharding to add) | The first deployable rocKE piece, in the "early deployable wins" spirit of PLAN decision 14 |

Next after these:
- Fold the compressor and candidate table into K2, and use epoch tags to drop `step_begin` (S7).
- mHC seam spread (S3).
- Reverse-engineer K1/K2 `.ll`/ISA per stage for the attention-projection GEMVs (S2).
- The Path A loop-body test.
- Device-side ragged verify (section 6).
- Large batch and EP (sections 4 and 7) as a separate phase.

### Reusable rocKE and harness pieces (from `qwen38_mono/` and `hackathon-prep/qwen38/`)

- Framework and stages:
  - `framework.py` (StageSpec, TaskTable, layer kernels)
  - `persist/` (sync primitives, launch-fatal word, lints, ISA gate)
  - `stamps.py`, `mailbox_ir.py`, `device_function.py` / `stage_fn.py` (hybrid extern stages for in-situ A/B
    inside ATOM's FlyDSL kernels, the Qwen `ATOM_Q38_EXTERN_STAGES` pattern)
- Kernels:
  - `moe_fused.py` / `stage_moe_fused.py` (MoE design; V4.1 needs A8W4 shapes and 384 experts)
  - `kernels/gfx950/mx_decode_gemv.py` (head)
  - `large_batch/{gemm,insitu,insitu_moe,moe_lb2}.py`
- `ccl/` (collectives, wire forms, refs, lints) and `tune/` (tuner, lookback).
- Harness: `rig.py`, `timeline.py`, `round_decomp.py`, `bytes_model.py`, `gap_table.py`, `pmc_compare.py`;
  `sched/{blame,coz,ab_sched}.py`; `quick/quick_tps.sh`; and the cluster scripts `q38_job.sh` / `q38_sync.sh`,
  which need new multi-GPU stages and a V4.1 model path.
- Timing discipline: in-situ graph A/B, shuffled arm order, two controls, the win rule; bitwise or tolerance first
  (`q38-learnings.md` section 2).
