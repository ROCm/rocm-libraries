# Handoff (written 2026-10-31 before a planned workstation restart)

Read PLAN.md (user decisions 1-14) first. Every file below is under
`C:/Users/vanantha/Documents/dev/copy-rocm-lib/experiments/qwen38-true-mono/` unless stated. Ratios only here;
absolute numbers in `hackathon-prep/private-perf/qwen38-true-mono/`.

## Where things stand

### W2: small-batch probes -> D0 (workflow run wf_590693fe-f24; dies with the restart)
- Done and independently verified: WP0a, P2, P4, P5, P6, P7, P8, P10, P1, P3 (`probes/*.md`).
- P9 done AND verified (`probes/P9.md`: restructured round tail wins: verify chain 0.795, draft 0.848 / 0.890,
  about 0.018 hu at K=3).
- Remaining: D0 (decision + per-cell budget + next WPs into DESIGN.md and
  a new D0.md), then an adversarial D0 check. The W2 script is
  `C:/Users/vanantha/.claude/projects/C--Users-vanantha-Documents-dev/31719099-769c-4a62-97c7-aea750cb70fe/workflows/scripts/q38-true-mono-w2-wf_590693fe-f24.js`
  (its D0 / D0-check prompts are reusable verbatim; feed them the probe results from probes/*.md).
- Key inputs for D0:
  - P1: gate NOT met: in-kernel hop is cheaper than a launch boundary (0.02-0.08 of a layer), but the loop body
    costs 1.012-1.023 (delta r1) / 0.999-1.013 (delta r4s) / 0.997-1.001 (qsa r4s) per layer -> Path B is the
    plan of record; re-test Path A on rocKE bodies (switch if x <= 0.99 at r4s and <= 1.00 at r1).
  - P3: HC spread V=10 = 0.969-0.970 per layer at r1 (delta, qsa), bitwise.
  - P10: router width no gain; dropping hold.m4 (empty M4 with the fused MoE) wins 0.963-0.975 per r1 layer.
  - P9: round tail in one launch wins after restructuring (straight port loses).
  - Rust compiler recommendations (rust-track/INTEGRATION.md): acqfence + bufoffany default on, B3 lever set.

### W4: large-batch probes (workflow run wf_066d87e8-865; dies with the restart)
- Done + verified: L0c-DP2 (ATOM DP2 beats ATOM TP2 in all 10 TP2 cells; FlyTape TP2 beats FlyTape DP2), CCL-LM
  (beats aiter custom all-reduce 80 KiB-2.5 MiB per rank 0.66-0.99, 5 MiB tie), P-L1 (GDN recurrence ties FlyTape
  at the bytes floor; bf16 state halves time).
- P-L3 done; W6 fix round restated it (fold-matched, 4 runs on 2 nodes): bf16 39 WIN / 13 TIE / 3 MIXED / 0 LOSS vs
  FlyTape's plan (out-proj and HC down at parity), mx 55/55, LB mx 0.07-0.44 of ATOM's own W4A8 linear; see P-L3.md.
- In flight at restart: P-L4 (MoE; null-bytes build first). Not started: WP-C5 (in-situ GEMM +
  all-reduce, needs CCL-LM), P-L2 (sync model), then synthesis into DESIGN-LB.md "Measured budget (W4)".
  Script: `.../workflows/scripts/q38-true-mono-w4-lb-probes-wf_066d87e8-865.js` (task prompts reusable).
  Check `large-batch/P-L3.md`, `P-L4.md` for partial results before relaunching them.

### Quick e2e track (standalone agent; dies with the restart)
- Harness: `hackathon-prep/qwen38/truemono-quick/` (`quick.sh`, `arms.sh`, `arms.d/`, `lookback/`, README in
  `quick/README.md`). Decide on per-request MEDIANS.
- Best measured stack (job 68229491, n01-09): tunedD lookback picks (`lookback/picksD.json`) + N8 96K draft vocab
  subset + subset-head pick + P10 hold drops: K=3 +4.7% vs the same-job hybrid, K=0 +3.0% (absolute in
  private-perf quick/README.md).
- TOP PRIORITY: RULER 8K regression on the best stack (qk-acc2 68231490): all 4 stack passes 0.634-0.650 below all
  4 hybrid passes 0.656-0.682 (mean -0.027 ~ hybrid spread 0.026; harness check passed it but treat as NOT
  cleared). Ablate lever by lever (tunedD picks, N8 96K, subset-head pick, P10 drops) with RULER 8K/32K; the 32K
  cell returned -1 (re-run; job qk-ruler 68231933 was running). HumanEval passed (+0.003 vs spread 0.055).
- Open: second-node confirmation, CONC 2/4 and K=1/2 cells,
  delta_ple lookback at budget >= 256, and the new P3 HC-spread arm (`ATOM_Q38_SCHED=spread.a1=10` at r1).

### Rust direct compiler track (standalone agent; dies with the restart)
- `rust-track/LEDGER.md`, `rust-track/INTEGRATION.md`; tree `C:/Users/vanantha/Documents/dev/copy-rocm-lib-rust`.
- Wins on both nodes: B3 all variants 0.948-0.990, fused MoE rows 1 0.970-0.976, toy d8w0 0.988; neutral: MoE
  rows 4, toy d4w8l2; small loss: head (1.001-1.016).
- Next: per-region lever scoping in the compiler core, head gap (cross-CU effects), MoE rows 4, timing
  harnesses for Q2 / framework / probe / CCL kernels, combination lookback over all levers incl. losers.

### Search engine / tuner (done)
- `search/INVENTORY.md`, `search/GUIDE.md`; code `.../qwen38_mono/tune/`; lookback keeps solo losers (decision 13).
- Open merge items M1-M12 (INVENTORY section 8) for the WP1 platform owner.

## How to resume after the restart
1. Check cluster jobs: `ssh vanantha@ctr2-alola-login-01 'squeue -u vanantha'` and the logs dir above; kill no
   job you do not own; old local pollers are gone with the restart.
2. Relaunch, at most two workflows at a time (user rule), roughly:
   - W5 = verify P9 (+fix) -> D0 -> D0 check (reuse W2's prompts and PRE preamble);
   - W6 = finish W4: P-L3/P-L4 (resume from their md files), WP-C5, P-L2, synthesis.
   - Standalone agents: quick e2e follow-ups, Rust track continuation.
3. Memory: `project_qwen38_true_mono.md` points here.

## Resumed 2026-10-31 (after the restart)
- D0 was completed before the restart (`D0.md`: Path B; budget 0.90 K=3 / 0.89 K=0-with-WP9 accuracy-gated).
- Relaunched: W5 (run wf_51c569d2-837): D0 adversarial check, then wave 1 (WP1, WP4h, P11, P13, P15, WP-R3 after
  WP1); results in `wave1/`. W6 (run wf_8728601c-c7a): P-L3 verify, P-L4, WP-C5, P-L2, LB synthesis.
- Quick-track agent resumed (stackF result, RULER two-mix resolution + ablation, second node, CONC/K cells).
- Rust track continues inside W5's WP-R3 (no separate Rust agent, to avoid conflicting edits).
- 2026-10-31 later: W6 completed (DESIGN-LB section 15 "Measured budget": GO conditional, MoE first; MoE is the
  only LB stage slower than FlyTape). W7 (run wf_631f72bb-d8a) launched: WP-L1, WP-L2a, WP-L2c, P-L5. Standalone
  agents: MoE rebuild (large-batch/P-L4b.md, moe_lb2.py), quick e2e (best: K=3 +4-4.7%, K=0 +6%, both nodes, accuracy
  clean). W5 (wf_51c569d2-837) still running wave 1.
- 2026-11-01: W8 (run wf_b09e8ceb-d67) launched: wave-2 stage ports WP2a/b/c/d/e/f + P3x, then WP3 layers=1
  kernels end to end and WP3a Path A switch test; results in `wave2/`. User allowed a third concurrent workflow.
  Running alongside: W5 (P11 fix round), W7 (P-L5, WP-L2c verify), MoE rebuild agent, consolidated-push agent
  (tail diagnosis). Best e2e so far: K=3 ~930, K=0 ~472 tok/s (hybrid ~870 / ~431).
- 2026-11-01: W5 completed (wave 1 all gates met; wave1/STATUS + D0 section 5 markers; P11 rev-3 T2 rule, canonical
  RULER mix v1, T4 tolerances). Round-boundary agent launched (host idle 0.07-0.08 of the round; FlyTape-style
  device loop / host prepare-ahead). W9 (run wf_380fde60-245) launched: P16 K=0 hop-cut probe, N4a accuracy gates,
  WP0b-acc remainder, re-verification of WP1 / P11 / WP-R3 fix rounds. Running: W7, W8, W9, MoE rebuild, round boundary.
