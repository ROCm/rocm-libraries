# DeepSeek-V4.1-Flash on rocKE: plan and user decisions

Started 2026-11-03 after the Qwen3.8 true-mono stack was frozen (experiments/qwen38-true-mono/FINAL-STACK.md).
Ratios only in this tree; absolute numbers go to hackathon-prep/private-perf/dsv41-mono/.

## User decisions
1. 2026-11-03: freeze Qwen3.8; start DeepSeek-V4.1-Flash. Work from a separate ATOM `main` clone (user clones it;
   until then ATOM main is read via `git -C dev/ATOM show origin/main:<path>`, read-only). dev/ATOM stays read-only.
2. Same method as Qwen: baseline first (ATOM native + its mono path), research the model and ATOM's V4.1 mono
   framework, then rocKE stage ports / reverse-engineering / search, persistent true mono, CCL for TP2/TP4.

## Phase 0 (now)
- R0 research: model architecture, ATOM V4.1 runtime + mono framework (atom/models/deepseek_v41/mono), DSpark,
  why the mono is refused (MXFP4 activations in its MoE), TP/EP configs, published numbers in ATOM docs.
- B0 cluster scout: container image for ATOM main, checkpoint availability / size / storage plan on alola,
  GPU needs (TP4 = 4 x gfx950 on the rocke partition), then a baseline run plan.
3. 2026-11-03: user asked for 1-2 dynamic workflows, "rocKE magic", target ~3x tok/s vs ATOM's best served path at
   TP2 at ALL concurrencies, and make TP1 run (weights ~1.06x one GPU's HBM -> needs compression/offload, accuracy-gated).
   W12 (run wf_74484644-c2e): probes P1 breakdown / P2 reverse-engineer ATOM V4.1 mono / P3 TP1 / P4 wide CONC ->
   D0 + ranked WPs -> build/verify/fix per WP -> integrate + lookback + TP2/TP1 matrix. GPU cap 8.
   ATOM edits go to dev/ATOM-dsv41 (git-archive export of origin/main df86c482, not a checkout) until the user's clone.

## Mandatory learnings carried over from Qwen3.8 (user, 2026-11-03: "take the learnings from the past, no shortcuts")
Every W12 agent applies these. Each one was learned the hard way.

### Method
1. Start from the competitor's winning structure, never a naive plan. Reverse-engineer ATOM's FlyDSL mono .ll/ISA per
   stage to bitwise parity FIRST, then crush with parametric search (rocKE's tuner), not hand tuning.
2. A loss is a diagnosis, not a verdict. Capture WaveScope/ATT (+PMC, ROCKE_DEBUG_LOC) before and after every lever.
   A losing kernel gets an ATT-guided restructure and a follow-up build right away (the Qwen MoE went from 1.06-1.32x
   slower to beating FlyTape this way).
3. Lookback: a solo loss NEVER removes a lever. Keep it as a default-off knob, recorded "rejected on base X".
   Re-test every reject after each accepted change. Climb order is a knob; search combinations (pairs/triples,
   interaction-aware surrogate). Ten levers can lose alone and win 50% together.
4. Intelligent sweeps (surrogate + uncertainty + boundary refinement), never brute-force grids.
5. In-isolation wins can vanish end to end: Qwen's schedule re-tune won 1.5-3% per layer and 0 in serving.
   ONLY end-to-end serving numbers count for claims. Do a quick e2e check early and often.
6. Profiler artifacts lie. The Qwen "5% host idle" came from rocprofv3 with the API trace slowing the host.
   Measure the host boundary with a no-profiler timeline. Kernel-only traces are fine for kernel shares.

### Measurement (DESIGN 9.1 discipline)
7. Comparator and arm in the SAME job, A-vs-A twins of both, shuffled/rotated passes, a fresh server per arm-run,
   medians (neighbour bursts move means), both nodes for any win. Match GPU counts in comparisons: DP2 vs TP2
   fairness bit us on FlyTape.
8. Verifiers re-derive every claim from raw logs. Past inflated claims, all caught by verification:
   - CCL vs RCCL 3x inflated (wrong RCCL protocol default);
   - fold-unmatched win counts;
   - tables built from miscompiled container builds;
   - a RULER "regression" that was a 256-token-cap artifact.
9. Accuracy: teacher-forced KL / T2 with enough runs, T4 (GSM8K, HumanEval, RULER) within the hybrid's A-vs-A spread,
   long-context caps of at least 4096. Mutants for every gate (an uncaught mutant means the gate is unproven).
   Determinism is required.

### GPU / platform facts
10. Coherence: `buffer_wbl2 sc1` on the writer, `buffer_inv sc1` on the reader; never scalar-reread in-launch writes.
    `s_nop` before asm atomics/polls after `v_readlane` (the hz knob).
11. Sync: tagged pairs beat graph boundaries; striped grid barriers (8 stripes per XCD); per-XCD tickets; abort-on-timeout.
    NUMA dual mailbox copies, holds, spread and prio were Qwen's biggest schedule levers.
12. Container COMGR waves=8 builds can miscompile (alola rule 219). Time with prebuilt code objects and verify
    bitwise after any toolchain change.
13. iglp_opt(1) codegen depends on prior compiles in the same process: compile fresh in a new process; keep the comgr cache job-local.
14. The in-kernel CCL beats RCCL/aiter on small messages, which is exactly the decode TP2 regime.
15. The Rust direct compiler wins on some stages (B3, MoE r1) and loses on others: auto-select per kernel; a loss is a defect to fix.

### Cluster
16. alola: login-01 (login-03 DNS dead); /tmp wipes on n07-17 (re-seed); n07-17 drops out around 05:45 UTC
    (requeue clobbers .out); heavy I/O in node-local /tmp; never pkill broadly (it killed another job's process);
    laptop suspends kill local pollers (resubmit, don't assume); never edit a script a job is running.
17. No local deletes; no state-changing git; no absolute numbers in repo files.
4. 2026-11-03: user: run more workflows. W13 (run wf_4ed9946d-c24) builds D0's WP4 (rocKE stage ports, K2 split),
   WP5 (in-kernel CCL), WP6 (wide path C>=8) in parallel, then WP7 (TP1 on stack), then integrates everything
   (W12's WP1-3 + W13's) incl. a CONC 10 cell and a TP4 reference. W12 keeps WP1-3 and is stopped before its second
   batch (WP4-7) so the work is not duplicated. GPU caps: W12 8, W13 6.
5. 2026-11-03 naming (user): our rocKE work = "rocKE hybrid"; only a pure persistent all-rocKE monokernel = "rocKE mono".
   ATOM's paths: "ATOM mono" (FlyDSL per-layer fused kernels in a graph) and "ATOM native". Use these in all new notes.
6. 2026-11-03: ANALYSIS.md (read it): 3x at C1-C4 exceeds the bytes ceiling (2.0-2.6x at 85% stream). The biggest
   untouched levers are the C1 CCL/rank-skew wait (0.19 +/- 0.1 of the step; measure wait vs work in graph mode
   first), the K2 latency chains (seams, indexer top-k, route, split-K attention), a routed-MoE restructure from ATOM's
   structure (WP6 skipped it on paper), TP4, long-prompt MIX prefill and acceptance length / ragged or tree verify.
7. 2026-11-03: user: "yes no shortcuts" / "be creative, work hard, make sure it works". W14 (run wf_1b026397-0d0):
   T1 MALL expert prefetch + T2 acceptance length (tree / ragged verify, lossless) in parallel, then T3 hop fusion,
   T4 long-prompt MIX prefill, T5 TP4; each build -> adversarial verify (3 rounds) -> fix; then stack on WP1 graph
   prefill + lookback + TP2 matrix (incl. C10) + long-prompt + TP4 cells. GPU cap 4 (W12 8, W13 6).
   Creativity mandate for every W14 agent: when the planned mechanism fails, try at least one structurally
   different mechanism before recording a reject (e.g. for T1: side-stream prefetch, poll-window prefetch, early
   router, L2/MALL residency of the hottest experts across steps; for T2: tree verify, ragged verify, draft
   recurrence, reuse of the target's own logits as an extra draft).
8. 2026-11-03 (user): "be creative, understand the best of what's in rocKE already, do hill climbing with lookback and
   other clever tricks from the md files in documents/dev, WaveScope it out." Mandatory for every agent from now on:
   - Read the rocKE methods before building: rocke/platform/dsl_docs/optimization/{optimization_runbook.md,
     runbook_mapping.md, measured_results.md, scheduler-policy.md}, dsl_docs/architecture/wavescope_integration.md,
     dev/rocKE-Search-Driven-Kernel-Optimization.md, dev/rocKE-Qwen38-TrueMono.md, qwen38-true-mono/search/{GUIDE,
     INVENTORY}.md and the tuner qwen38_mono/tune/ (surrogate + signed lookback that keeps losers, order as a knob).
   - WaveScope (ATT + PMC, source-correlated) before and after every lever on the real served kernel (ATOM mono K1/K2
     at 6/24/48 rows): bubble ledger per stage (idle waves, poll waits, waitcnt stalls, MFMA/VALU busy, LDS, HBM).
   - Expose every performance choice as a knob (default = today, byte-identical) and climb with lookback over ALL
     knobs from W12/W13/W14 plus new ones (losers included, orders varied, combinations screened), scoring on the
     graph-mode K2 critical path first and confirming on served cells.
   - W15 (queued; starts when W12 finishes) does exactly this as its whole job.
9. 2026-11-03 (user): "the rust compiler should help also". The Rust direct compiler (rocKE IR -> HSACO, no LLVM;
   branch users/vanantha/rocke/rust-compiler, worktree dev/copy-rocm-lib-rust, crate platform/rust/rocke-codeobj;
   qwen38-true-mono/research/rust-direct.md, rust-track/) is a full lever for every rocKE-built stage: per-kernel
   auto-select (direct only where it wins on both nodes), losses are compiler defects to fix, pass order is a knob.
   It gives exact control of waitcnt/poll placement, register allocation (K2 SGPR spills), scheduling and layout.
   For ATOM mono's FlyDSL stages: build bitwise rocKE ports of the high-share stages, then COMGR vs direct.
10. 2026-11-04 (user): "crush fly mono; what from Qwen helps significantly, even at CONC 1". The Qwen CONC-1 wins came
    mostly from the SCHEDULE / CTA-ALLOCATION knob surface that ATOM-truemono added to the FlyDSL mono
    (ATOM_Q38_SCHED: per-stage holds, prio, spread, dyn, head_sleep, pipe; per-stage CTA counts sh_ctas/sh_base,
    m3_ept; MoE issue order dn_issue, router_knobs=fused, line_order=slot, sh_join, route_pause; mailbox pause), searched
    PER LAYER KIND x WIDTH by the tuner with lookback (qwen38-true-mono/quick/README.md "tuned"/"final" tables):
    0.955-0.990 per layer from the picks, 0.929-0.964 with spread.a1=10, P10 hold drops x1.02, the N8 draft-vocab subset
    plus subset head 0.984 of the head. The V4.1 mono has NONE of this surface (MONO-STAGES 3: "schedule levers: none"),
    and WP4 built only NUMA, width constants and pause; holds, prio, spread, dyn, per-stage CTA counts and MoE issue
    order were never built. That is the most significant transferable lever. W15 Climb must:
    - port the ATOM_Q38_SCHED machinery into ATOM-dsv41's V4.1 mono kernels as ATOM_DSV41_RK_SCHED (holds/prio/spread/
      dyn/pipe per stage; CTA allocation per stage; issue order in ug/down/shared/route; poll pause per site);
    - tune it per layer kind (layer 0 / indexer-selecting / non-selecting / draft) x served width (5, 6, 12, 24, 48 rows)
      with the qwen38_mono/tune engine and signed lookback (losers kept, orders varied, re-test after every move);
    - then the Qwen draft-side wins: a draft vocab subset + fused subset head (V4.1 draft LM head + Markov = 0.03 of the
      C1 step with 5 sequential all-gathers), the lean draft round;
    - stack with the Rust direct compiler on every rocKE-built stage, and with WP1 graph prefill.
11. 2026-11-04 (user): prioritize what lands next, kill workflows going nowhere. Stopped W13 (WP4 fix round + WP7 +
    integration) and W14 (T2 fix round, then T3-T5); cancelled their cluster jobs. W15 is the single priority:
    WaveScope ledger -> climb with the Qwen schedule surface + Rust direct + lookback -> confirm (TP2 matrix incl. C10,
    TP4, long-prompt cell, accuracy). Parked follow-ups with measured bounds (resume later if W15 lands):
    same-row tree verify (+3-7% AL, needs CSA2 per-node stores), long-prompt MIX prefill (T4, never started),
    TP1 on the stack (WP7, never started), W13 integration (W15 Confirm replaces it).
    Closed by measurement: byte cuts (WP3), wide path (WP6), MALL prefetch (T1), CCL forms (WP5), in-graph tail (WP2),
    variable-width verify (T2), NUMA/K2SPLIT/pause (WP4). All kept as default-off knobs for W15's lookback.
12. 2026-11-04 (user): fast iteration. Measured: an arm-run is ~105-110 s fresh-server start + ~130 s generation;
    launches are microseconds. For W15 from now on: screen every candidate in an in-situ rig (graph-mode stage
    timeline / one running server, seconds per candidate) and send only rig winners to served runs; served cells use
    a fixed steady-state decode window (256 tokens a request, warm-up excluded); capture only needed graph buckets
    and trim server start. Win acceptance unchanged. (alola-cluster rule 262.)
13. 2026-11-04 (user): "why not WaveScope rocKE and improve that, why FlyDSL". W15 redirected (same run, playbook
    cached): instead of tracing ATOM's FlyDSL kernels, PORT the served K2 stage groups to rocKE bitwise (K2-moe: xq,
    route, ug, down, reduce + shared; K2-attn-gemv: attn_score, irq, wo_a, wo_b, slice, norm, gate, router + K1 wq_b),
    linked through the extern-stage mechanism; then WaveScope OUR code natively (ROCKE_DEBUG_LOC ATT + PMC), COMGR vs
    Rust direct per stage, and climb on the ports' knob surface + the decision-10 schedule surface, rig-screened first.
14. 2026-11-04 22:30 UTC: W15's two port agents died on a network drop (ECONNRESET) about 1.5 h in. Their partial
    work exists and must be CONTINUED, not redone: rocKE dsv41_mono/{stage_moe.py, stage_attn.py, stage_router.py,
    stage_norm.py, stage_gate.py, seam_common.py, README.md} and the harness hackathon-prep/dsv41/w15/{rig, moe,
    stages, ws}. Read them (and README.md) first, check what is gated, and pick up from there.
