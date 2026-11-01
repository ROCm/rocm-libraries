# Qwen3.8-Flash-Next: a true rocKE monokernel (gfx950, TP1)

Goal: beat the hybrid stack of `Qwen3.8-Flash-Next-rocKE.md` (FlyDSL per-layer mono + rocKE MoE / B3 /
lm_head + NUMA / hold levers; verify round 0.856 of mono at MTP K=3, 0.868 per token at K=0) with a decode
path in which rocKE owns every kernel, using what FlyTape (`dev/flytape-main`, a FlyDSL whole-run
occupancy megakernel for the same model) does differently. Ratios only in this tree; absolute numbers live
in `hackathon-prep/private-perf/qwen38-true-mono/`.

## What the baselines say (measured, private-perf; ratios here)

- Hybrid stack round, K=3: layer kernels about 0.79 of a mono round, in-graph launch boundaries 0.023,
  everything outside the target layer kernels (head, fold, sampling, draft front / meta, 3 eager draft
  layers and heads, verify meta, prologue) about 0.19. Layer kernels are sync/latency bound: 2.3x their
  issued-bytes floor, CTAs poll 0.60-0.75 of the time, hand-off hops 0.26-0.55 of the critical path.
- FlyTape: one launch per closed batch, phases separated by a striped software grid sync (8 stripes per XCD,
  ballot poll: 1.14 us idle), wave-owned GEMVs with all loads in flight, prefetch before the sync, sampling,
  stop and the request loop on device. Its per-step time (random weights, no MTP) is being measured on the
  rocke partition (`flytape-rocke/notes`); its documented numbers put the step well above the mono's, so its
  in-step phase design is not the lever. Its whole-run persistence (no launches, no host in the loop, on-device
  head/sample/stop) is.

## Architecture

One persistent rocKE kernel, 256 CTAs x 512 threads (one per CU, residency ticket as in FlyTape), per MTP
verify round, then per request loop:

1. Inside a layer: the mono's tagged-mailbox dataflow (it beats phase-wide grid syncs), NUMA dual-copy
   mailboxes and holds from day one, every stage a rocKE emitter on `framework.py`'s `StageSpec`:
   existing `stage_b3`, `moe_fused` (M3+M4+M5), `stage_q2`; new HC (A1/A2/M1/M2), B1, C1, Q1, Q3-Q7, P1/P2.
2. Across layers: no launch boundary; layer-parity double-buffered scratch (`TaskTable.validate_stack`);
   next-layer weight issue only after a CTA's last own load of the current layer; stage holds.
3. Head and sampling in the kernel: `mx_decode_gemv` MXFP4 head with fused argmax, verify/accept compare,
   draft front, the drafter's QSA layer and head x3, verify meta: the whole K=3 round in one launch.
4. Request loop on device (FlyTape): host polls a mapped cursor; ATOM sees tokens only. OPTIONAL (about
   0.01 of a round over `round` mode; DESIGN.md 2.5, WP12), never on the critical path.

## Milestones (each gated: bitwise or tolerance numerics first, then shuffled paired A/B)

- M0 baselines: FlyTape TP1/TP2 on alola; reproduce mono vs hybrid ALL (`quick_tps.sh`) on the re-seeded node.
- M1 rocKE delta layer kernel (r1, r4s): all stages rocKE, bitwise vs the mono layer (rig gates), A/B vs
  the hybrid ALL layer.
- M2 qsa and delta_ple layer kernels (Q3-Q7 port).
- M3 48-layer target pass + head in one launch, swapped for ATOM's graph; pass A/B vs hybrid ALL.
- M4 whole round (draft in the kernel) and the ATOM runner hook; e2e tok/s vs hybrid ALL at K=3 and K=0.
- M5 RSI: WaveScope + stamps + Coz blame, hill climbing with lookback, losers kept as knobs.

## Goal restated (user, 2026-10-29)

Crush the hybrid rocKE work with a full rocKE monokernel that has full GPU control. The Rust direct
code-object compiler on `users/vanantha/rocke/rust-compiler` (rocKE IR -> HSACO, no LLVM; gfx950 + gfx1151)
is in scope where it buys control LLVM does not give (poll loops, exact waitcnt, cache-policy bits,
placement, register pinning, compile time of a whole-round kernel). That branch does not contain the
monokernel commit; bringing it in is a working-tree overlay (no git state changes). Analysis:
`research/rust-direct.md` (not written yet; DESIGN.md WP1 writes it from probe P6's compile-time numbers).

The work-package plan of record is DESIGN.md section 8 (revision 2); its D0 go/no-go decides between the
persistent `pass`/`round` path and the `layers=1` + tail-kernel fallback.

## User decisions (2026-10-30)

1. Numerics: non-bitwise restructuring is ALLOWED, gated by per-stage tolerance tests plus the accuracy
   suites (GSM8K, HumanEval, RULER long-context) within the pinned hybrid's own A-vs-A spread. Bitwise stays
   the default where it costs nothing. The design must be re-budgeted for this higher ceiling ("crush").
2. Rust direct compiler: a parallel PILOT track. Minimal overlay of the rust branch's own commits (no git
   state changes), the g950p crate from `copy-rocm-lib-agents` if it is a superset passing the compiler
   gates, IR-only direct builds (IR mailbox, intrinsic converts, int_to_ptr), pilot on the head GEMV and the
   toy-chain mailbox behind a default-off `code_object_backend` knob. Full 4-6 week commitment only if it wins.
3. Scope: RULER needle retrieval at 8K / 32K / 128K / max added as the long-context accuracy suite;
   non-greedy sampling and logprobs are IN scope (WP8).
4. (2026-10-30) Large batch is IN scope: CONC 8-128 (B5-128), TP1 and TP2, must beat FlyTape's tiled tape
   (same batch / TP / precision) and ATOM's native path. Multi-GPU is in scope: rocKE gets IPC peer
   mailboxes and an in-kernel CCL (all-reduce / all-gather) so TP2 (and up) works. Track: `large-batch/`,
   designs `DESIGN-LB.md`, `DESIGN-CCL.md`.
5. (2026-10-30) Fair comparisons: always at matched GPU count, batch, precision and metric. Every TP2 cell
   also gets a DP2 arm (two independent TP1 replicas on the same 2 GPUs, total tok/s and per-request
   latency); the target per cell is the better of TP2 and DP2 at equal per-request latency. Device-step
   numbers (FlyTape bench) and served end-to-end numbers are never mixed without saying so.
6. (2026-10-30) "Crush optimization all the way." The Rust direct compiler is COMMITTED as a full track
   (no longer a pilot), and it does NOT need to match COMGR bit for bit: any transform is allowed (FMA
   contraction, reassociation, own instruction selection / scheduling / waitcnt / register allocation,
   cross-stage scheduling) when the kernel passes per-op tolerance tests against an fp32 reference
   (runbook 1.4 ladder) and, once deployed, the model accuracy suites (GSM8K, HumanEval, RULER) within the
   pinned hybrid's A-vs-A spread. Determinism (same input, same bits run to run) stays required. The goal
   is to beat COMGR on every true-mono stage kernel and become the mono's default code-object backend.
7. (2026-10-30) Headline targets: K=3 CONC 1 >= 1000 tok/s per request on gsm8k (verify round <= 0.83 of the
   hybrid's; the hybrid serves 863), K=0 CONC 1 >= 550 tok/s (<= 0.78 of the hybrid's per-token time).
   Rough bytes floors (estimate): about 1.4-1.9 ms per K=3 round (depends on unique experts U) and about
   0.8 ms per K=0 token, so these need about 45-60% (K=3) and about 70% (K=0) of the floor end to end.
   D0 and every lever ledger rank work by the share of the gap to these targets it closes.
8. (2026-10-30) Full end-to-end control. Use the search-driven optimization method of
   `dev/rocKE-Search-Driven-Kernel-Optimization.md` and the tuning / search / selection code already in
   `dev/rocm-libraries` (read-only: never modify that tree; copy useful pieces into copy-rocm-lib,
   additive only). The mono tuner lives in `qwen38_mono/tune/`; inventory and guide in `search/`.
9. (2026-10-30) 1000 tok/s is a floor, not a finish line: after the targets are met, keep optimizing as far
   as the hardware allows (RSI loop until levers stop paying, with the bytes floors as the bound). No
   shortcuts: every gain verified on hardware, correctness first, nothing silently narrowed.
10. (2026-10-30) XCD-level extreme optimization is in scope: creative work scheduling, routing and dynamic
    optimization for peak fill and no bubbles, exploiting full compiler control. Lever catalogue:
    `XCD-SCHEDULING.md` (20 levers, each a default-off knob, gated and tuned; prior losses cited). D0 and
    WP1 must make per-edge protocol / placement / queue choices expressible in the framework.
11. (2026-10-30) Modifying compiler code itself is in scope for peak performance: the Rust direct compiler's
    core (instruction selection, scheduler, register allocator, waitcnt / hazard passes, encoders, code layout),
    rocKE's IR and Python lowering (new ops, attributes, scheduling hints; Python-only per the skip-C++ rule),
    and COMGR-path options. Gates unchanged: tolerance + mutants per kernel, determinism, accuracy suites,
    existing builds byte-identical with new knobs off (or changes confined to the direct backend).
12. (2026-10-30) Reference bar correction: the hybrid ALL arm measures about 875 tok/s per request at K=3
    CONC 1 (user's own measurement 875.1; e2ecomp2, 6 shuffled pairs, 96 prompts: 876). W1's re-baseline
    (863, fewer pairs, other job/GPU) is inside the hybrid's run-to-run / node spread and is NOT the bar.
    The headline target of decision 7 is therefore >= 1000 tok/s against a 875 tok/s hybrid, i.e. a verify
    round <= about 0.875 of the hybrid's at equal acceptance; every win is still decided by same-job paired
    arms, never against a remembered number.
13. (2026-10-31) Lookback done right: a lever's solo loss never removes it from the lookback. Ten levers can
    each lose alone while a combination of them wins big. Lookback pools contain EVERY retained knob,
    losers included; combination search screens interactions (pairs, triples, interaction-aware surrogates),
    and only measured combination data may prune. Winning combinations still pass the win rule on both nodes.
14. (2026-10-31) Quick end-to-end tok/s is the standing progress metric: `hackathon-prep/qwen38/truemono-quick/`
    (one job, mono / hybrid / candidate arms, K=3 and K=0, 32 gsm8k prompts). Every work package and every
    deployable lever ends with a quick e2e number in the same job as the hybrid; early deployable wins
    (tuner combination lookback over the hybrid's knobs, N8 draft vocab subset, proven probe levers) are
    measured end to end now, not after D0.
15. (2026-10-31) Never give up on an idea because it fails in isolation: order-dependent combinations work. A
    rejection is always "rejected on base X", never permanent. After every accepted change (lever or
    restructure) every previously rejected lever / idea is re-tested on the new base; climbs vary their order
    (several orders, restarts, revisiting rejects); where application order exists (compiler passes,
    transforms, restructure sequences) the order itself is a searchable knob. Ledgers record the base each
    verdict was measured on.
16. (2026-11-01) Porting method: reverse-engineer, then crush. For every FlyDSL mono stage, dump FlyDSL's final
    LLVM IR (.ll, the last MLIR->LLVM stage) and ISA, and diff them against the rocKE port's .ll / ISA per stage
    (instruction mix and order, waitcnt placement, registers / spills, LDS, MFMA interleave). Reach parity by
    construction (diff-proven), record where FlyDSL's own code leaves time (drains, spills, conservative waits),
    then turn every hard-coded choice into a knob and let the parametric search (tuner, losers and order-dependent
    combinations included) beat it. "1:1" is the floor of a port, never its goal.
17. (2026-11-01) Agents never delete files locally (no rm / del / Remove-Item of ISA dumps, temp files,
    __pycache__ or any artifact): deletions prompt the user for approval and stall the agent. Leave artifacts
    as they are and write to new paths instead. Never wait on a permission prompt.
