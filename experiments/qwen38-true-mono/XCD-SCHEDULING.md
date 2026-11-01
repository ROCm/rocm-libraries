# XCD-level scheduling, routing and dynamic work placement (lever catalogue)

User direction (2026-10-30, PLAN decision 10): consider XCD-level extreme optimizations; be creative with work
scheduling, routing and dynamic optimization so the GPU stays full with minimal or no bubbles; with full control
of the compiler (Rust direct track) such levers become cheap to express.

Every item below is a candidate lever, built as a default-off knob, numerics-gated, timed in situ against the
current best with the DESIGN 9.1 discipline, and fed to the mono tuner (`qwen38_mono/tune/`). Evidence that
already bounds an idea is cited: several "dynamic" ideas LOST on the hybrid, so each must state why it differs.

## Facts that shape the design (measured)

- gfx950: 8 XCDs x 32 CUs, an L2 per XCD, the 256 MiB MALL shared; two IODs (XCDs 0-3, 4-7); a mailbox homed
  on the remote IOD costs about 1.5x the round trip (NUMA lever, ledger/schedule.md).
- Coherence (P5): a writer must `buffer_wbl2 sc1` for readers on OTHER XCDs; same-XCD readers were correct
  without it. Readers need `buffer_inv sc1`. So an intra-XCD hand-off is a cheaper protocol than a cross-XCD one.
- Sync (P4): 8 stripes per XCD for counters; per-XCD residency ticket 0.25-0.40 of FlyTape's; tagged pairs beat
  a launch boundary; NUMA polarity must be calibrated in situ per edge.
- Lost on real layers (keep as knobs, do not repeat blindly): global dynamic ticket assignment (1.012-1.079,
  first arrivers start weight streams early = contention), simulator placement (1.041), `s_setprio` (no effect
  with one CTA per CU), poll pause, two polls in flight, MALL speculation of last step's experts (1.20-1.26:
  each CU's stream is already bandwidth bound), xcd_blocked/sequential placement (1.02-1.39).
- Bandwidth saturates near 192 CTAs; serial hand-off stages (A1, M1, C1, M3, Q-chain) use 9-26 tasks while the
  rest of the GPU waits.

## Levers

### Placement and routing
1. **XCD-affine dataflow.** Place each producer->consumer chain so the edge stays inside one XCD and use the
   intra-XCD protocol (no `wbl2 sc1`, L2-local polls). Candidates: B1->B3 per head group, Q-chain per KV head,
   C1 partials -> M1, MoE gate_up -> down of one expert. Requires per-edge protocol selection in the framework.
2. **Expert-to-XCD homing.** Fixed expert -> XCD affinity (by id hash balanced on measured routing
   frequencies) so an expert's gate_up and down stream through one XCD L2 and its combine partials stay local;
   the router broadcasts one compact list per XCD instead of a full all-to-all.
3. **Head / column ownership per XCD** for GDN/QSA heads and dense GEMV row blocks, with the layer output
   all-gathered once per XCD pair (hierarchical exchange: intra-XCD reduce, then one cross-XCD hop).
4. **Critical-path islands.** Run the latency-critical serial chain (A1/A2, M1/M2, M3, Q4/Q6) on a small
   XCD-local island of CUs whose mailboxes are homed on that IOD, while the remaining CUs stream bulk weights.
5. **Hierarchical reductions** (wave -> CU -> XCD -> GPU) everywhere a split-K or combine crosses CUs, with the
   last level on the consumer's XCD.

### Overlap and bubble popping
6. **Bubble filling with next-stage weight issue,** bounded by contention (holds won because early streams
   contend): issue next-stage weights only from CUs idle in the current stage, sized to leave the critical
   stage's bandwidth intact; the tuner climbs the issue budget per stage.
7. **Pre-decode while waiting.** CUs waiting on a mailbox decode/shuffle their next MXFP4 tiles into LDS or
   registers (weights do not depend on activations), so the post-poll tail is MFMA only.
8. **Cross-layer pipelining by XCD:** the CUs that finish layer l's MoE first start layer l+1's HC down
   partials on rows already combined (row-tile readiness instead of whole-vector readiness).
9. **Draft/verify overlap (MTP):** the draft layer of step t+1 and the tail of verify step t share the grid on
   disjoint XCD sets where the dependency allows (accepted-token readiness known per row).
10. **Wave specialization:** one poll wave per CU, loader waves, MFMA waves, a comm wave for CCL pushes;
    removes `vmcnt` coupling between pushes and weight loads (DESIGN-CCL `comm_wave`).

### Dynamic scheduling (must beat the measured losses)
11. **XCD-local work queues** instead of a global ticket: dynamic balancing only among CUs that share an L2,
    started in critical-path order and gated by the hold budget (the global ticket lost through contention,
    not through the idea of balancing).
12. **Routing-aware MoE balancing:** after the router, a deterministic per-XCD assignment of (expert, tile)
    units by measured bytes (line placement per XCD), recomputed on device each layer from the actual
    top-k, with fixed-order combines for determinism.
13. **Adaptive holds:** the hold per stage chosen on device from the last step's stamps (a closed loop on the
    measured critical path), bounded and deterministic in numerics.
14. **Online schedule selection:** a few precompiled schedules per variant; the kernel picks per step from
    cheap features (rows, accepted tokens, U unique experts, context length).

### Compiler-enabled (Rust direct track)
15. Per-XCD specialized code paths (xcd_id is uniform per CTA) with different unroll/issue orders.
16. Exact `vmcnt`/`lgkmcnt` that never drain in-flight weight loads at a poll or a CCL push.
17. Register-pinned state across stages (no reload of loop-invariant addresses / scales / LDS bases).
18. Instruction-level interleave of poll loads with weight issue and MFMA in the same wave.
19. Cache-policy bits per access class: NT streaming for experts, sc1 for mailboxes, MALL-resident small hot
    tensors (HC, router, norms, scales).

### Power and clocks
20. MI355X throttles at the power cap: fewer wasted bytes and fewer spinning waves raise the sustained clock;
    measure SCLK per arm (rule 70/79 samplers) and count spin-waits as power cost in the objective.

## How these are tested

Probe first where the effect is uncertain (a microbench can lie about placement: P4 polarity), then in situ on
real layers with both controls; the tuner owns the knob space; WaveScope + stamps + Coz before and after;
losers stay default-off knobs and enter the lookback combination sweeps.
