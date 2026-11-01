# W1 review: completeness and correctness of DESIGN.md

Reviewer: W1 completeness/correctness. Read-only review of `DESIGN.md` (557 lines) against PLAN.md, research/*,
`hackathon-prep/reports/qwen38-rocke/*`, the ATOM mono sources (`dev/ATOM/atom/models/qwen4_exp_mono`, read only),
ATOM's runner/sampler/proposer, the rocKE tree at 89209ac7665 and private-perf (cited as ratios only).
Line references are `DESIGN.md:<line>` unless a path is given. `A` = `dev/ATOM/atom/models/qwen4_exp_mono`,
`AT` = `dev/ATOM/atom`, `Q` = `rocke/library/kernels/gfx950/qwen38_mono`, `RP` = `hackathon-prep/reports/qwen38-rocke`.

Arithmetic and quoted shares were re-checked and hold: section 1.1 and 1.2 tables (GAP_TABLE 2.1/2.2, e2ecomp2),
the 0.856 / 0.868 hybrid ratios, FlyTape 1.70x / 1.59x (private flytape-baseline), the hop ratios 1 : 0.71 : 0.58 and
0.80 / 0.71 of the boundary gap (private mailbox-hop-latency vs t1-final-abs), BW 14 -> 80 CTAs 0.11 -> 0.58, the
stack losses (ledger/schedule.md lever 13), the budget sums 0.06-0.27 (K=3) and 0.05-0.27 (K=0).

Totals: 5 blockers, 11 majors, 17 minors.

---

## Blockers

### B1. The `xl` hand-off feeds only A1/A2, but C1 and P2 also read the layer input
- Evidence: C1's residual is "the layer input, prefetched from `h_in` before any poll" (`A/kernels/stages/out_proj.py:9-10, 87-96`).
  P2 reads `h_in` for the query norm and the residual (`A/kernels/stages/ple.py:415-440`). `stack.py:17-23` relies on exactly
  this: "every other stage of B reads `h_in` only after polling a (transitive) consumer of A1/A2 (C1's residual, P2's)".
  The mono already has the right template in the PLE layer: "P2 -> A1/A2/C1: the PLE output h' ... laid out as `w` and `whs`"
  (`A/kernels/layout.py:102-106`).
- Design: M5 publishes to A1/A2 only, and "Plain-memory `h_out` is written only by the pass's last layer" (`:36-39, 217-218`).
  As written, layer l+1's C1 (and P2 in layer 1) read a stale or never-written `h_in`.
- Fix: the `xl` consumer set is {A1, A2, C1, P2(PLE layer)}. C1 task t and M5 task t own the same 16 hidden columns
  (`moe.py:27-31`, `out_proj.py:3`), so C1's edge is 1:1 (co-place or poll one producer). Add C1/P2 to P1's scope, to
  `validate_stack` and to the bitwise gate. If `h_out` stays plain for any reader, that reader needs the bulk rule (`wbl2`/`inv`)
  and the fence saving claimed in 3.2 shrinks.

### B2. The ATOM hook is at the wrong level: head, sampling, acceptance and drafting are ModelRunner work
- Evidence: the mono runner never runs the head. It returns mixed hidden states (K=0) or unmixed HC streams (K>=1)
  (`A/runner.py:359-367`). `compute_logits` (`AT/models/qwen4_exp.py:1058-1063`), the Sampler, the RejectionSampler, logprobs
  and `propose_draft_token_ids` run in `AT/model_engine/model_runner.py:3136-3320` and `:3613`. The proposer loop is in
  `AT/spec_decode/eagle_proposer.py:612-830`: input scatter of the bonus token, positions + 1, `index_select` at
  `last_token_indices`, `_enter/_advance_decode_metadata`, `compute_draft_ids`. With deferred output, tokens and MTP status
  go through `prepare_sampled_ids`, `send_mtp_status_to_cpu_async` and `prev_token_ids`.
- Design: `pass` "replaces the runner's 48 layer launches plus the head and fold" (`:400-401`). `round` turns the drafter
  forwards into "reads of those results" (`:402-406`). The head and fold are not in the runner, and no hook point, output
  format or async-output path is specified.
- Fix: add an "ATOM call-path contract" section and a WP. Name the functions `pass`/`round` replace (compute_logits +
  sampler + rejection_sampler + propose). Give the exact outputs the kernel writes: `sampled_tokens [bs, k+1]`, the verdict
  `(num_bonus, num_reject, anchors)`, `draft_token_ids [k, bs]`, `next_token_ids`. Say who calls the deferred-output path.
  Route requests that need logprobs or full logits, which a fused-argmax head does not produce, to the hybrid.

### B3. Next-round words written by the kernel's tail have two owners, and the n-gram advance is not idempotent
- Evidence: block tables, spec slots and state indices are allocated by ATOM's scheduler every step. The mono stores only
  a pointer to the request's block-table row (`A/step_meta.py:278-280`), and slots come from `spec_state_indices_tensor`
  (`A/runner.py:313-321`). The prologue reads and rewrites the n-gram window in place, using `state_indices_in/out`,
  `has_initial_state` and `num_accepted_tokens` (`A/prologue.py:16-21, 276-291`).
- Design: the tail writes "step counter, positions, slots, block-table rows, ... n-gram window, embedding rows, PLE window"
  (`:229, 381-383`). The host re-runs `write_verify_meta` / `StepPrologue` "whenever the batch composition changes"
  (`:384-386`). If the tail already advanced the n-gram state, the host's StepPrologue advances it a second time and
  silently corrupts PLE. Block-table rows are not the kernel's to write. `:386` cites a "WP6a gate" that does not exist.
- Fix: run prologue and meta as in-kernel stages at the START of each launch. They read the per-step buffers ATOM has just
  prepared, so there is one owner per word and the result is bit-identical by construction. The tail writes only
  GPU-produced outputs (tokens, verdict, drafts) into ATOM's buffers. This also removes risk row `:556`. Replace "WP6a"
  with a real WP id.

### B4. No tag plan for one launch that holds 48 target layers, k draft steps and the round stages
- Evidence: the design pseudo-code runs every draft step as `body[qsa][variant_j](48)` (`:233`), with "one step load per
  launch; tags `step * 64 + layer + 1` computed in registers" (`:338`). Today the draft has its own counter, which
  `write_draft_meta` advances on every draft step (`A/draft_runner.py:134, 214-220`; `A/step_meta.py:392`), and its own
  scratch.
- In one launch the k draft steps would publish identical tags into the same regions, and a consumer can accept the
  previous draft step's payload. The design also gives the final mix, head, accept, front and meta stages no tags. The fused
  MoE's private `tag ^ 2^31` scheme (`Q/README.md` `moe_fused`) must stay unique per instance.
- Fix: write an explicit tag/region table for the whole round. Distinct layer ids 48..48+k-1, kept below
  `LAYER_SLOTS - 1` (`P/helpers/mailbox.py` `check_layer`), or a per-draft-step step increment. Ids for the round stages.
  Parity slots for every draft instance. Run `validate_stack` over the full round schedule, with a mutant that reuses one
  tag.

### B5. Words written during the launch are read through scalar or plain loads
- Evidence: state-table entries are scalar loads (`A/kernels/stages/common.py:79-87`, `is_scalar=True`). The cache length
  `pos` is a plain vector load behind them (`A/kernels/stages/qsa.py:182-194`). The design says scalar loads are legal
  because "nothing writes the table during the launch" (`:333-335`). In `round` mode the draft steps' positions follow
  from the in-kernel accept (`AT/spec_decode/eagle_proposer.py:812-830`, positions advance per step), so they are written
  and read inside one launch. The rule "scalar loads only for memory nothing writes during the launch" is a recorded
  pitfall (`research/q38-learnings.md:251`).
- Fix: carry per-draft-step `L` and positions in registers or a tagged mailbox, not in the scalar-loaded table. If a table
  must be rewritten in-launch, use the bulk rule plus a scalar-cache invalidate. Add this edge to the section 4 sync table
  and to P5, with a negative control on the scalar path.

---

## Major

### M1. The final HC mix's bitwise reference is wrong for K>=1 and for more than 8 rows
- Evidence: at K>=1 the target's final mix is `compute_logits -> mix_final` (`AT/models/qwen4_exp.py:1058-1063`). That is
  `hc_rows` only up to `HC_FINAL_ROWS_MAX = 8` rows, and the six-op `mix` GEMM path above that
  (`AT/model_ops/qwen4_exp/hyperconnection.py:44-47, 123-140`). `A/runner.py:18-22`, which the design cites, is the K=0 path.
- Design: "final HC mix: bitwise vs ATOM's per-row `hc_rows` path, rows 1..16" (`:301`). "The round bitwise vs the hybrid's
  round" (`:416`). At r9g3, r12g3, r12g4 and r16g4 a per-row mix cannot equal the hybrid's.
- Fix: either reproduce `mix`'s GEMM order above 8 rows, or declare the wide final mix tolerance-class with Tier A drift,
  acceptance and accuracy gates. State which one, per row count.

### M2. G2 (HC spread) is justified by bandwidth, but the data says A1 is tail-bound and M1 is wait-bound
- Evidence: A1 wait/span is 0.00 and tail/span 0.80-0.94 in every non-PLE cell. M1 wait/span is 0.48-0.86 (`RP/GAP_TABLE.md`
  section 6, e.g. lines 269, 274, 487, 492). The A1 time is the post-load norm of all 4 streams, the pack and the down chain
  (`A/kernels/stages/hc.py:7-12`), not per-CTA bandwidth. The design both cites "tail/span 0.84-0.88" (`:102`) and calls
  A1/M1 "bound by per-CTA bandwidth" (`:41-42`).
- The spread keeps each 20-k_iter chain the same length (`:292`). Today each A2 task polls the 10 groups' MXFP8 low-rank
  packs. With the redundant A2-side sum it polls every unit's f32 partials: 320 columns x 4 partials = 1280 pairs per row,
  times 80 A2 CTAs. That is no new hop, but a large poll payload that P3 must include.
- K=0 depends on G2. Without it, the K=0 terms in 2.2 sum to -0.004 .. +0.13. The low end is a loss, not the stated "thin
  margin 0.02-0.06" (`:176-177`).
- Fix: P3 times A1 + A2 (and M1 + M2) in situ on real-layer stamped builds, including the norm and A2's poll volume. Add
  alternative reductions as knobs: last arriver per n-block, two-level. Restate G2's evidence and the K=0 fallback range.

### M3. Most matrix cells have no hybrid baseline, and the budget's low ends are not floors
- Evidence: composed e2e exists only at CONC 1, K=3 and K=0 (`RP/E2E_COMPOSED.md:10-23, 180`). K=1, K=2, CONC 2-4, long ISL
  in-server and long-context accuracy were never measured (`RP/GAP_TABLE.md` section 7: "QSA at long context in-server ...
  not run"). NUMA has never run on wide builds (`A/kernels/sched.py:226-231, 382`).
- Design: expected ratios are given for all of these cells (`:159-166`). PLAN M0 (baselines on the re-seeded nodes) is not
  a WP. Low ends are called "mechanism mostly fails" (`:133-134`), but G1's low end is 0.00 even though the stack lost
  1.02-1.09 per layer, and G8 rests on the toy ledger (`:147`; `research/q38-learnings.md:240` pitfall 5).
- Fix: add WP0 = baselines for every cell (hybrid ALL and mono; K 0..3 x CONC 1..4; long ISL at 8K/32K/128K/max; the
  accuracy suites), each with its own A-vs-A spread. Pin the hybrid as the recipe's commits. Do not ship WP2d into the
  hybrid while comparing against it (`:488-490`), or re-baseline when it ships. Mark ranges that lack data as such.

### M4. The grouped/wide matrix is incompletely specified
- Variant mapping: "K>=1 CONC 2 (r4g2 narrow grouped)" (`:163`) holds only at K=1. K=2 CONC 2 is r6g3 and K=3 CONC 2 is
  r8g4, both wide. The full set is r4g2, r6g2, r8g2, r6g3, r9g3, r12g3, r8g4, r12g4, r16g4, odd-row r9g3 included
  (`AT/models/qwen4_exp_mono/dispatch.py:38-40`, `groups.py`, `kernels/sched.py:160-162`).
- Padding: under a captured graph the target verify is padded to the capture size, and a padded group runs on null state
  (`A/group_meta.py:12-18`, `A/groups.py:5-9`). Draft step 0 is "never padded" (n·S rows) (`A/groups.py:9-13`). A round
  kernel must handle target G != draft n and skip padded groups in accept, head and meta. The design never mentions
  padding.
- Sources: wide builds run different FlyDSL stages: `hc_wide.py`, `delta_group.py`, `delta_group_wide.py`, `moe_wide.py`,
  `out_proj_wide.py`, `ple_wide.py`, `qsa_wide.py`, `qsa_fast_wide.py`, `common_wide.py`. These are the bitwise references
  for WP2/WP6, but the emitter table cites only the narrow files (`:292-305`).
- Fused MoE: rows <= 4, with `NUMA_HDR` tied to the narrow scratch (`Q/moe_fused.py:68-72, 549-550`). It is reachable from
  the framework only through `stage_adapt` as self-placed m3/m4/m5 (`Q/stage_adapt.py:30-33, 62`). Wide is new design work,
  not "reuse" at 4G (`:452`).
- Fix: enumerate the variants per (K, CONC) cell. Add padding to the contract and to Tier G. List the wide sources. Re-cost
  WP2d/WP6. Add a NUMA-area sizing and calibration item for wide scratch and for parity slots (stack.py gave each layer
  "its own NUMA area", `A/kernels/stack.py:23-25`).

### M5. The `xhold` default does not reproduce the graph boundary's no-overlap profile
- Design: default "M4 done" counter, which "reproduces the graph boundary's no-overlap contention profile" (`:219, 257`).
- M5 streams the routed and shared down weights on 160 CTAs after M4. Its span is 5.6x its issued floor (`RP/GAP_TABLE.md:117`).
  CTAs without an M5 task would start next-layer B1/Q1 streams during M5: the losing run-ahead mechanism (ledger lever 13;
  `dyn` lever 14).
- Fix: default to "this layer's `xl` complete" (the event A1/A2 already wait for). Keep "M4 done" and `sleep:N` as knobs
  for Coz to climb.

### M6. The numerics gates are too weak for cross-layer races, and there are no mutants above the stage level
- Evidence: the toy LDS-parity race showed up in 3 of 3000 replays, and only under prefetch stress (`RP/ledger/layerkernel.md:63`).
- Design: "12 launches without zeroing" (`:412`). Gates 2-5 (`:414-419`) and WP1-WP10 name no mutant (`:448-484`). P4 has no
  gate at all (`:440`).
- Fix: for `pass` and `round` add delay-injection stress (random per-CTA `s_sleep` at stage entries and at the `xl` and
  `xhold` points) over thousands of replays. Add mutants that must fail:
  - parity slot swapped;
  - `xl` tag off by one layer;
  - row 0 reading `s[a]` instead of `s[a-1]`;
  - accept count off by one;
  - a planted tie at a Q4 chunk boundary;
  - a draft position off by one.
  WP1 needs negative controls: an oversize grid must time out, a dropped arrival must not release the grid sync.

### M7. The cpp-parity exemption is missing, and the WP1 helper placement violates the repo rules
- Evidence: `rocke/AGENTS.md:21-34` makes byte-identity the #1 invariant. `platform/AGENTS.md:448-473` says a new helper
  needs a "C++ mirror + byte-identity in the same PR" and gives "Do NOT add to helpers: emitters without a C++ port path".
  `Q/README.md` "cpp-parity exemption" requires a distinct name suffix, an explicit `backend="python"`, tests that assert
  the backend that ran, and nothing Python-only in `rocke.helpers`. The framework must keep
  `test_toy_cpp_engine_byte_identical`.
- Design: places the grid sync in `P/helpers/grid_sync.py` (`:440, 448`) and changes the framework (`layers="dynamic"`, args
  table, `xhold`, ticket). It never mentions the exemption. The user's rule is Python-only, no C++ work.
- Fix: put the grid sync, ticket and `xhold` in `Q/` (library family, like `mailbox_ir`). Or build them from mirrored
  inline-asm ops only and extend the byte-identity test. State the suffix and backend rule for every true-mono build. Keep
  the framework's mirrored path byte-identical. Treat the Rust direct backend as a third engine outside the gate: bitwise
  outputs plus ISA lint.

### M8. Prefill hand-off and step-type transitions are not designed
- Evidence: the word "prefill" does not appear in DESIGN.md.
  - Every prefill and every mixed step is ATOM's own (`A/dispatch.py:7-13, 53`).
  - The drafts for the first verify come from ATOM's proposer over the prefill chunk (`AT/spec_decode/eagle_proposer.py:697-705`).
  - A refused width turns mono off for good (`A/groups.py:15-17`; `AT/mono/runtime/route.py`, `LazyRunner`).
- Fix: specify the following.
  - The first round after prefill: inputs and state written by ATOM's prefill path.
  - Transitions round <-> hybrid <-> ATOM-eager when CONC changes, a prefill joins, or a request needs non-greedy sampling
    or logprobs: which state each path must leave behind, and that counters and tags stay disjoint.
  - That a true-mono refusal falls back to the hybrid runner for that width instead of turning mono off.
  - Gate it with Tier G chains that cross prefill and CONC changes.

### M9. The goal-level completeness criterion conflicts with per-cell routing
- Design: "A cell switches only after it wins, and the hybrid serves every other cell" (`:398-399`). Non-greedy requests
  are routed to the hybrid "until a sampler WP" (`:303, 557`), and no sampler WP exists in section 8.
- User goal: rocKE owns every kernel AND beats the hybrid on the FULL matrix.
- Fix: define done as every matrix cell routed to the true mono and passing the e2e WIN rule (state that rule for e2e: the
  median below the A-vs-A minimum, and a share of pairs). Add a sampler WP, or record non-greedy and logprobs as explicitly
  out of the matrix, with the user's sign-off.

### M10. Long ISL is under-specified
- The supported max is `MAX_CONTEXT - K`, not 262144 (`A/config.py:27-32, 94-97`). It is also bounded by the model's RoPE
  table and KV memory at the cell's CONC. The served recipe uses `--max-model-len 10240`.
- Long-ISL e2e tok/s is dominated by ATOM's prefill, which neither stack owns.
- The "long-context suite" (`:166, 419, 482`) is never named.
- Tier G must cover rollbacks that straddle page (16) and index-compression (4) block boundaries, and the draft's own step
  words (`write_draft_meta`, never exercised: `RP/GAP_TABLE.md` section 7).
- Fix: pin max_model_len per CONC, measure decode-only metrics (ms per verify round and per-token time at depth L), and
  name the suite and its thresholds.

### M11. Timing hygiene conflicts with running 6 GPUs in parallel
- Evidence: only one busy neighbour was measured neutral. "More than one busy neighbour ... not run" (`RP/GAP_TABLE.md`
  section 7).
- Design: allows up to 6 of our GPUs busy (`:426`) and "GPU neighbours ... only as in T0" (`:518-519`), but no WP plans
  for it.
- Fix: timed A/B and e2e jobs either take a node with at most one busy neighbour, or first validate the multi-neighbour
  case, as a WP0 item. Schedule builds and numerics on the other GPUs.

---

## Minor

1. The P1 control expects the fenced stack to "reproduce 1.02-1.05" (`:437`). On `best`, which is the arm P1 runs, it
   measured 1.034-1.090 (`:203-204`; ledger/schedule.md:53).
2. The FlyTape citation is wrong: `weights.py:37-40` (`:369`) is `flytape/models/qwen4/weights.py:37-40` (`EXPERT_GAP`).
3. Missing WPs:
   - creating `dev/ATOM-truemono` and its `layer` hook (P1 and WP3 need it; the copy does not exist);
   - a routing and refusal table;
   - graph capture and replay validation (padded sizes; compile at install, never during capture, as `rocke_head` does);
   - the `xl` producer and consumer emitters, which have no Phase 1 package and first appear in WP4 on the critical path.
4. `research/rust-direct.md` is referenced by PLAN.md:53 and DESIGN.md `:357` but does not exist.
5. The verify argmax reference is ATOM's Sampler (`topk_select(..., tie="low")`, temperature <= `SAMPLER_EPS` per row:
   `AT/model_ops/sampler.py:34-50, 137-150`) applied to the rocKE logits. It is not `_argmax_fold`. Gate 3 (`:415`)
   must compare against that path, including NaN/inf rows.
6. The rows-16 head build has an unexplained per-process bimodality at rows 5-16 (`RP/ledger/head.md:43`). It affects the
   wide cells' in-kernel head. Include rows 8 and 16 in P2.
7. `q4_chunks` "exact" (`:298`) needs a planted-tie gate at chunk boundaries. Q4 breaks ties "to the lower block"
   (`A/kernels/stages/qsa_fast.py:18-22`).
8. G1's evidence compares a one-pair-per-producer microbench with the boundary gap. The `xl` payload is every stream word
   plus its sums. P1 decides, but say so in 2.1.
9. PLAN.md:34 makes the on-device request loop part of the architecture, while the design makes it optional (WP10).
   Update PLAN.
10. `loop` mode (WP10) does not design on-device stop (EOS, max_tokens, stop strings) or the scalar-loaded tables that
    get rewritten across rounds.
11. The GPU-hour costs omit server-start compiles. Extern-linked kernels recompile on every start (recipe Troubleshooting),
    and a whole-round rocKE kernel per (K, G) variant has to compile at install. They also omit the e2e count: 16 or more
    cells x 6 pairs x arms + A-vs-A.
12. Section 0 item 3 says the grid sync gates next-layer weight streams. Section 4 uses a per-XCD counter. Make the
    wording consistent.
13. The code-size claim (`:321-323`) covers only K=3 CONC 1. P6 should build the widest round (r16g4 target x 3 kinds +
    draft r16g4 + r4 + heads at 16 and 4 rows) for its VGPR and scratch check.
14. G1's ramp bound uses the simulator (`:140`), while section 9.2 says the simulator never ranks levers. Fine as a bound;
    say so.
15. The cluster logistics left out of section 8:
    - login `ctr2-alola-login-01`;
    - `q38_sync.sh` stage packing;
    - the first `atom*` stage is mounted at `/atom`, so a hybrid-vs-copy job stages the hybrid under another name
      (`hackathon-prep/qwen38/README.md:64-67`);
    - the rocKE stage comes from copy-rocm-lib (`dev/rocm-libraries` lacks this work, `research/rocke.md:3-4`), with the
      CRLF rewrite;
    - node-local `/tmp`;
    - write new facts into the alola-cluster skill.
16. ATOM refusals the true mono inherits are not restated: ReplaySSM off, bf16 KV with block 16, f32 mamba state, text-only
    (`A/config.py:78-114`). The served line needs `ATOM_ENABLE_REPLAYSSM=0`.
17. The per-layer `h_out` debug knob (`:217`) makes per-layer gates run on a different binary. Keep gate 3 (pass output and
    logits) on the served binary, and say that gate 2 runs on the debug build.
