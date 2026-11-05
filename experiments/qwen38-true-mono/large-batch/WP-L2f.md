# WP-L2f: the large-batch head with the fused argmax (TP1 and vocab-parallel TP2), the PLE, the final mix and the embedding prologue, rows 5-128, gfx950

W10, 2026-11-02. `../DESIGN-LB.md` 4.2 (pass end, PLE layer), 6 (E10 / E11 / E12), 7.1 (head and PLE sharding), 15.6 /
15.7; `W7-STATUS.md` section 4 (entry conditions); `WP-L2a.md` (the E1 GEMM core the head and the PLE projection run on);
`WP-L2b.md` (the HC chain the final mix is one instance of). Ratios only here. Absolute times, job logs, WaveScope
output and raw JSON: `hackathon-prep/private-perf/qwen38-true-mono/large-batch/WP-L2f.md` and `.../wp-l2f-raw/`. Harness:
`hackathon-prep/qwen38/truemono-lb/wp-l2f/`. Below 1 is faster.

## Gate and verdict

**Gate: met on both rocke nodes.** Every family runs on MI355X with ATOM's numerics, passes its numerics suite with every
applicable must-fail mutant caught, and beats FlyTape's matching phases in the same job (FlyTape traced before and after
the LB timing; the LB arms load venv-built code objects strictly in the container). Final builds, one container bracket
per rocke node (`ct5a` n01-09, `ct5b` n07-17), ranges over the cells and over each job's FlyTape before / after traces:

| family (FlyTape phases) | TP1 (rows 5-64) | TP2 (rows 5-128) |
|---|---|---|
| **head + argmax** (`lm_head` + `sample_reduce` + `select_token`) | **0.23-0.28** | **0.23-0.33** |
| **final mix** (`final_mix_*`; one HC chain instance of WP-L2b) | **0.82-0.93** | **0.66-0.84** (FlyTape's combine includes its exchange wait) |
| **embed + PLE** (every FlyTape `embed` and `ple_*` phase; TP2 incl. `embed_sync` / `ple_reduce` / `ple_copy`) | **0.51-0.58** | **0.44-0.54** |

Rows 5 read FlyTape's B8 (it has no B5 configuration); FlyTape has no TP1 B128. Per cell: section 4.

- **Numerics** (15 jobs, both nodes): 4,118 per-arm head checks (the fused argmax's ids equal the argmax of the build's own
  logits twin bit for bit, both walks, determinism) with 0 failures; 516 logits checks against float64 (worst 0.06 of
  E1's bound) and 0 flips against ATOM's argmax of the float64 logits; 846 planted cases (ties inside a lane, across
  units, across ranks; an all-negative token where a padding row must not win; the winner in the last-arriving CTA) all
  correct; 1,381 of 1,384 applicable head mutant runs caught (the 3 misses: E1's `kskip` checked on planted one-hot data
  in round 1, which never reads the skipped K chunk; checked on random data since, caught every time); PLE / embed /
  prologue: 3,120 arm checks, 0 failures, 1,770 of 1,776 mutant runs caught (the 6: `mod` in round 1, an equivalent mutant
  at decode, section 2); the composed PLE chain G -> P -> A on device 65 of 66 (the miss: a relative tolerance near
  cancellation in round 2, replaced by an operand-scaled one; section 2). WP-L2b's chain (the final mix) passed its own
  numerics suite in both final brackets (9 of 9 cells each).
- **Climb** (4 rounds + a confirming round 5, both nodes, PLAN 13 / 15): every E1 lever on each cell's head pick, solo
  and paired with the merge form, on the PLE projection and on A; losers kept as arms; each round re-tests every lever on
  the new base. 7 head changes, 3 projection changes and A's `upc=2` at r128 WON on both nodes on the same base and are in
  the plan tables (`head_lb.HEAD_PLANS`, `ple_lb.PPROJ_PLANS`, `ple_lb.PAPPLY_PLANS`); section 5 is the ledger.
- **WaveScope before / after** (Markham MI350X, TP1 r16 / r128, the fused head against WP-L2a's logits-only head of the
  same plan): the fused argmax removes the logits store stall (13.6 % of the logits-only head's stall at r128) and its
  epilogue adds about 3 % (two LDS barriers); total stall 0.83 (r128) / 0.91 (r16) of the logits-only head's.
- **Budget** (DESIGN-LB 15.7's "head" and "other" rows, now measured): at TP2 B128 these families are 0.017 of FlyTape's
  traced step; LB's versions take 0.007-0.008 of it (head 0.0033, final mix 0.0015-0.0018, embed + PLE 0.0023-0.0025),
  inside the 0.006-0.009 W7 assumed (head 0.26-0.36 of 0.009, "other" 0.8-1.2 of 0.005). At TP1 B8: 0.035 -> 0.011.

## 1. What was built

| piece | file |
|---|---|
| E10 head: `HeadKnobs` (`e1` plan, `tp`, `akey` bf16 / f32, `lout`, `amrg` unit / cta, `xdelay`, mutants), `build_head_kernel`, `head_params`, `arena_bytes`, host key twins, the measured `HEAD_PLANS` + `plan(tp, rows)` | `dnn-providers/hip-kernel-provider/rocke/library/kernels/gfx950/qwen38_mono/large_batch/head_lb.py` |
| E11 PLE + E12 prologue: `WalkKnobs` (`kind` embed / pgather / prologue / papply, `threads`, `occ1`, `nt`, `upc`, `acq`, mutants), `build_embed_kernel`, `build_pgather_kernel`, `build_prologue_kernel`, `build_papply_kernel`, the measured `PPROJ_PLANS` + `pproj_plan(rows)` (P on E1) and `PAPPLY_PLANS` + `papply_knobs(rows)` | `.../large_batch/ple_lb.py` |
| CPU tests (32): E1 lowers to the same IR before and after head builds; markers of every head build (MFMAs, the umax / swap atomics, the TP2 push); every knob and mutant lowers; refusals; `plan` / `pproj_plan` valid for TP x rows 1..128; host key ordering; every embed / gather / prologue / apply build and mutant lowers | `.../rocke/library/tests/test_qwen38_lb_l2f.py` |
| bench: head (TP1 one process, TP2 torchrun + IPC arena), PLE / embed / prologue / projection / composed chain, strict prebuilt loading, code-object save, two-node verdicts, FlyTape-family reports, numerics census, static ISA counts, WaveScope launcher, job / submit / sync scripts | `hackathon-prep/qwen38/truemono-lb/wp-l2f/`: `l2f_head.py`, `l2f_ple.py`, `l2f_report.py`, `l2f_final.py`, `two_node.py`, `num_census.py`, `isa_stats.py`, `att_head.py`, `att_job.sh`, `job.sh`, `sub.sh`, `sub_ct.sh`, `sync.sh`, `dep/` (WP-L2a's `l2a_bench.py`, `isa_gate.py`, P-L4's `asm_hazard.py`, copied) |
| alola facts | `.claude/skills/alola-cluster/SKILL.md`, rule 242 |

No existing file changed: `e1.py`, `e1_plans.py`, `hc_lb.py`, the CCL (`ccl/wire.py`) and `persist/` are imported. Python
engine only; kernel names `q38lbhd_*` / `q38lbp_*`; nothing served imports the new modules. CPU tests: 32 pass (laptop,
lowering only).

E11 / E12 were gated in W7 on W5's small-batch WP2e / WP2f emitters, which have not been published. The PLE and the
prologue here are built from ATOM's model code directly (`qwen4_exp.py`, `model_ops/qwen4_exp/ple_layer.py`,
`ops/ple.py`, `hyperconnection.py`, `triton_gemma_rmsnorm.py`), with ATOM's rounding points; the small-batch emitters can
take the same semantics.

### 1.1 E10: the head with the fused argmax

The GEMM is WP-L2a's E1 (subclassed; only the epilogue changes) over the rank's vocabulary rows (TP1 248,320; TP2
124,160, rank r owns global rows [r M, (r + 1) M)).

- **Unit argmax.** Each lane folds its 4 mt rows of every token tile in increasing row order with ATOM's key: the
  signed-order key of the bf16-rounded logit, -0 = +0, the lowest row on a tie (`mx_decode_gemv`'s rule; `akey=f32` is
  FlyTape's f32 compare, a tolerance lever, a TIE in time everywhere). Rows past M (row-group padding) never win. The 4
  lane groups of a token fold by xor-16 / 32 shuffles (larger key, then lower row), the waves through LDS (same rule,
  order-free).
- **Cross-unit merge.** One 64-bit `global_atomic_umax_x2` per (unit, token) of `(key ^ 0x80000000) << 32 | ~row`: the
  unsigned maximum is "larger key, then lower row", ATOM's `tie="low"`, independent of arrival order (deterministic).
  The unit then arrives on its token block's counter; the last of the arrivals reads and clears the word with one
  returning `global_atomic_swap_x2` (zero at rest) and writes `(global id, key)`. `amrg=cta` first folds a CTA's units of
  the layer in LDS and issues one atomic per CTA and token (the counter then counts CTAs): fewer atomics on the hot token
  words (at TP2 r5 the per-unit form put 970 units x 5 tokens on 5 words).
- **TP2 merge** (the CCL argmax merge of DESIGN-CCL WP-C2, fused): the last arriver pushes the rank's `(key, global id)`
  of each token, tagged, into both ranks' IPC arenas (`ccl.wire.push_v4`, sc0 sc1), polls both ranks' pairs in its own
  arena (`ccl.wire.poll_pairs`, a 2 s deadline, an abort word, a timeout counted in the launch's error word) and merges
  in rank order (larger key, then the lower global id), so both ranks write the same token. Slots
  `[layer][launch parity][src rank][token]`; tag = epoch L + layer + 1 from the arena's epoch word, advanced by the
  launch's last CTA (graph-replay safe, no host reset). A rank needs its peer's pair of launch e + 1 before it can finish
  e + 1, so it is at most one launch ahead per slot parity: two parities suffice for both walks.
- Against WP-L2a's logits-only E1 head of the same plan (the "before", container jobs on both nodes): the fused head is
  0.85-1.04 of it at TP1 (faster from r16 on: it does not store the logits) and 0.92-1.12 at TP2, where it also does the
  in-kernel exchange (1.10-1.12 at r5). `lout` (the logits as well, for logprobs / sampling) costs 1.03-1.29 of the fused head.
- **Not built:** the logprobs / non-greedy sampling pieces (`DESIGN.md` WP8: LSE / top-K' merges).

### 1.2 E11: the PLE (ATOM semantics), and the E12 prologue and final mix

ATOM's PLE on layer 1: `out = gated + silu(conv(norm_conv(gated)))`, `gated = sigmoid(signed_sqrt(<norm_key(key_proj(emb)),
norm_query(h)> / sqrt(H))) value_proj(emb)` per HC stream, `emb` = 16 hashed n-gram rows (E4M3, one global scale) of the
token and its 2-token context, a dilated (3) causal depthwise conv of kernel 4 over a 9-token state per channel.

- **G** (`pgather`, FlyTape `ple_gather`): per token the 16 head ids (ATOM `compute_ngram_ids` at decode: int64
  wraparound products, xor, EOS cuts the n-gram, torch's non-negative remainder by the head's prime size, plus the head
  offset), the gather of 16 x 160 E4M3 bytes from the 47.7 GiB table in VRAM (replicated per rank: ATOM's `ngram=device`;
  no TP exchange, FlyTape pays `ple_reduce` / `ple_copy`), `bf16(bf16(e4m3) bf16(scale))` (ATOM `fp8_embedding_lookup`),
  its MXFP8 (the `mx_quant` rule: what ATOM's W4A8 linear quantizes) written in E1's op_sel activation layout, and the
  request's new context.
- **P**: key_proj and value_proj as ONE E1 GEMM (12,800 x 2,560, MXFP4 x MXFP8) reading G's output directly; plan table
  `pproj_plan` (WP-L2a's `qsa_in` TP1 picks, the nearest planned shape, plus measured overrides).
- **A** (`papply`, FlyTape `ple_stats` + `ple_apply`): one unit per (token, stream), 320 threads x 8 channels: the Gemma
  norms of key and residual, the gate with ATOM's bf16 rounding points (product, sum, scale, max, sqrt, sigmoid), the gated
  value, its Gemma norm, the conv (f32 products and sums in tap order, no FMA: ATOM's `enable_fp_fusion=False`), bf16
  before the SiLU, `ple = bf16(gated + silu)`, the residual `bf16(r' + ple)` and the state shift in ATOM's
  `[slots, channels, 9]` layout. The f32 sums are in this kernel's order (Triton's tree differs): tolerance class.
  `upc=2` runs two units per CTA at once (640 threads).
- **E** (`embed`): `r[token][stream] = embed_tokens[id]` for the 4 streams (ATOM's `repeat`); the bf16 table replicated
  per rank (ATOM's vocab-parallel lookup adds one nonzero row to zeros: the same bits, no exchange; FlyTape pays
  `embed_sync`).
- **E + G fused** (`prologue`, the default form): both depend on the token ids only, so one phase does both (one unit per
  token); it costs what G alone costs. P depends only on G: the pass can run G and P before the PLE layer, off its
  critical path (WP-L3 / L4 scheduling).
- **E12 final mix** is one instance of WP-L2b's HC chain (C combine + norm, D down GEMM, U up GEMM + mix) with the final
  mixer's weights, timed by WP-L2b's own bench and prebuilt objects inside this lane's container jobs (same-job FlyTape
  ratio). The final mixer has no injection: dropping C's 4 gate dots for it is a lever not built (`hc_lb.py` is WP-L2b's
  file). Its MXFP8 output (U's `xq_pass`, E1's op_sel X layout) is what the head reads.

Mutants (must fail): head `tie` (in-lane >=), `pad` (padding rows not masked), `ulo` (the packed low word is the row: a
cross-unit tie goes high), `early` (the merge one arrival early while CTA 0 starts every unit late), `xpeer` (the peer's
pair ignored), `xorder` (equal keys take rank 1's id), `xstale` (the poll accepts any non-zero tag and rank 1 pushes late:
the previous launch's pair is read; the deterministic stale-read twin), the `cta` merge's `tie` / `ulo` / `early`, and E1's
`kskip`; embed `row`, `s3`; gather `eos`, `mod`, `hash`, `scale`, `ctx`; prologue `row`, `s3`, `eos`, `hash`, `scale`, `ctx`;
apply `gate`, `knorm`, `conv`, `tap`, `state`, `stream` (also on the `upc=2` form); projection `kskip`.

## 2. Numerics (MI355X, both nodes)

Every timed code object is checked in the same process before its timing, on random data at the cell's shape, both walks
(grid-synced and free):

| check | result |
|---|---|
| head: ids of every arm = the argmax of its own `lout` twin's bf16 logits (ATOM's key and tie rule), both TP | bitwise, every arm, every cell, every job |
| head: `lout` logits vs float64 (E1's bound 2^-11 of sum |x w|) | worst 0.06 of the bound |
| head: ATOM's argmax of the float64 logits (bf16-rounded, merged over ranks) vs the kernel | 0 flips |
| head: TP2 ids identical on both ranks and = the rank-order merge of the two ranks' own argmaxes | every cell |
| head planted cases (lane-quad tie, cross-unit tie, cross-rank tie, all-negative token, the winner in CTA 0's last unit) | 551 / 551 |
| embed, gather, prologue: bitwise against ATOM's rules emulated on the host (ids, E4M3 gather, bf16 scale, MXFP8 codes and op_sel scales, the new context, the 4 stream copies) | bitwise, every arm, 8 layers, every rows value |
| projection P vs float64 | worst 0.04 of E1's bound |
| apply A vs the float32 reference with ATOM's rounding points (f64 sums) | r'' bitwise; ple / state worst 0.6-0.9 bf16 ulp |
| composed G -> P -> A on device (G's MXFP8 read by P, P's Y read by A with its row stride) | P worst 0.03 of the bound; A worst 1.9 ulp of the operands' scale |

- **Tolerance class:** A (its f32 sums differ from Triton's tree; a 1-ulp change of the bf16 score moves the gate by one
  bf16 ulp). In round 2 the composed chain at rows 64 failed a relative check at one element where `ple` = -258 cancels
  `r'` = 0.53: 1.5 bf16 ulps of 256. The check measures the error in ulps of the operands' scale since
  (|r'| + |gated| + |silu|); every A mutant lands 10^5 ulps or more away.
- **Equivalent mutant:** `mod` (the sign fix of the hash remainder). ATOM bounds the multipliers by (2^63 - 1) / vocab, so
  every product of a token id stays below 2^63 and their xor is never negative at decode: the fix never fires. Marked not
  applicable from round 2 on; the fix stays in the kernel (ATOM keeps it too).
- **Not applicable:** `pad` where the plan has no padding rows (M a multiple of the rows per unit: every TP1 plan and the
  TP2 plans with 256-row units); caught in every TP2 cell with 512-row units (r8, r32, r64).

## 3. Method

- Cells: TP1 and TP2 x rows 5, 8, 16, 32, 64, 128 (the head at TP2 on 2 ranks with the in-kernel merge; PLE / embed /
  prologue are per-rank work, identical at TP1 and TP2: replicated tables).
- Metrics (WP-L2a's): `phase` = one launch walks L instances (head L 4 / 7 at TP1 / TP2, 4x the MALL of weights; PLE /
  embed L 8; P L >= 4x the MALL) with W5's striped grid barrier between instances, per instance (the verdict metric);
  `stream` = the same walk without the barrier. TP2 rounds take the slower rank's time.
- Timing: 16 shuffled rounds, every arm with a second-graph and a recompiled control; DESIGN 9.1's win rule per job; a
  change is accepted only when it WINs on both rocke nodes on the same base (`two_node.py`); losers stay as arms (PLAN
  13 / 15); every round re-tests every lever on the new base.
- FlyTape bracket: container jobs with FlyTape's traced TP2 B8-B128 and TP1 B8-B64 (mxfp4, prompt 128) before and after
  the LB timing; the LB arms load the venv-built code objects strictly (alola rule 219); WP-L2b's bench runs in the same
  job for the final mix (its stages `w10-l2b-rk21` / `-h23` / `-hs23`, read only).
- Neighbours: the venv jobs log `squeue -w` of the node (1-4 of our jobs and 1-3 foreign jobs next to every run); the
  container jobs cannot (no Slurm client inside); all comparisons are in-process paired arms.
- Jobs (MI355X): venv `d1`, `s2a`, `r2a`, `r4a`, `r5a` (n01-09), `s1b2`, `r2b`, `r3b`, `r4b`, `r5b` (n07-17); container
  brackets `ct1b` (n07-17, round-1 builds), `ct2a` (n01-09, round 2), `ct4b` (n07-17, round 4), and the final pair `ct5a`
  (n01-09) / `ct5b` (n07-17) on the round-5 objects of `r5b`. `ct2b` was requeued when n07-17 went `down*` at 21:44 UTC
  (the node rebooted and lost its seeded `/tmp`; W11 reseeded it) and was cancelled; `r3a` / `ct4a` were cancelled while
  queued, superseded by `r4a` / `ct5a`. WaveScope: Markham MI350X `w10l2f-att1`.

## 4. Against FlyTape, per family and cell (same job, phase metric)

Ranges over the final brackets `ct5a` (n01-09) and `ct5b` (n07-17) and each job's FlyTape before / after traces
(FlyTape's own spread: up to 1.08 on its head, up to 1.12-1.20 on its small `embed` / `ple_gather` phases). head =
fused-argmax head (TP2 with the merge); final = one HC chain instance; prologue = E + G vs `embed` + `ple_gather`; P vs
`ple_proj`; A vs `ple_stats` + `ple_apply`; e+PLE = prologue + P + A vs every FlyTape embed and PLE phase incl. its TP
exchanges.

| cell | head | final | prologue (E+G) | P | A | e+PLE |
|---|---|---|---|---|---|---|
| 1:5 | 0.23 | - | 0.76-0.77 | 0.33 | 0.80-0.88 | 0.51-0.53 |
| 1:8 | 0.24 | 0.87-0.90 | 0.77-0.80 | 0.33-0.35 | 0.83-0.92 | 0.53-0.56 |
| 1:16 | 0.23 | 0.90-0.91 | 0.69-0.80 | 0.36-0.37 | 0.84-0.93 | 0.53-0.57 |
| 1:32 | 0.24-0.25 | 0.82-0.84 | 0.67-0.80 | 0.36-0.37 | 0.87-0.88 | 0.53-0.55 |
| 1:64 | 0.28 | 0.91-0.93 | 0.70-0.84 | 0.37-0.38 | 0.92-0.98 | 0.56-0.58 |
| 2:5 | 0.24 | - | 0.65-0.69 | 0.37-0.38 | 0.86-0.90 | 0.51-0.52 |
| 2:8 | 0.24-0.25 | 0.66-0.72 | 0.67-0.71 | 0.38-0.40 | 0.90-0.93 | 0.52-0.54 |
| 2:16 | 0.23 | 0.72-0.74 | 0.59-0.62 | 0.41-0.43 | 0.91-0.93 | 0.52-0.53 |
| 2:32 | 0.25 | 0.67-0.71 | 0.48-0.50 | 0.45 | 0.90-0.94 | 0.51-0.52 |
| 2:64 | 0.26-0.28 | 0.71-0.76 | 0.38 | 0.45-0.46 | 0.97-1.00 | 0.51-0.52 |
| 2:128 | 0.33 | 0.72-0.84 | 0.23-0.30 | 0.34-0.35 | 1.07-1.10 | 0.44-0.47 |

The earlier brackets agree: `ct2a` (n01-09, round-2 tables) and `ct4b` (n07-17, round 4) gave head 0.23-0.35, final mix
0.66-0.93 and e+PLE 0.47-0.58 over the same cells; A at TP2 B128 was 1.14-1.18 before `upc=2`.

Reading:
- **The head is 0.23-0.33 of FlyTape's head + sample + select** in every cell, on both nodes. At small rows it streams the
  weights near the copy rate; at r128 it is MFMA-bound (the f8f6f4 rate), 0.33 at TP2. Its static ISA (venv COMGR,
  `wp-l2f-raw/isa_head.json`): no spill, 184-444 VGPRs, the epilogue adds 350-1,950 instructions over the logits-only
  head, all VALU / LDS outside the K loop.
- **The final mix is 0.66-0.93.** At TP2 FlyTape's combine includes its exchange wait and LB's chain has none before
  WP-L5 (WP-L2b's caveat): read the TP2 column as LB's upper-bound advantage; the TP1 column (0.82-0.93) is like for like.
- **The prologue fuses two tiny phases.** E alone (1.19-1.41 at TP1, 0.27-0.82 at TP2) and G alone (1.10-1.26 at TP1,
  0.58-1.53 at TP2) are each about a grid barrier long and lose to FlyTape's tiny `embed` / `ple_gather` at small rows;
  E + G in one phase costs what G costs alone, 0.23-0.84 of FlyTape's two phases.
- **A at TP2 B128 is 1.07-1.10**, the one family member still slower than FlyTape (1.14-1.18 before `upc=2`, two units
  per CTA at once). Its 9-entry state shift moves 92 KB per (token, stream) unit (a ring buffer would cut that: section
  7). Everywhere else A is 0.80-1.00; the PLE family as a whole is 0.44-0.58 because P and the prologue are far ahead.
- **P is 0.33-0.46**: E1 on the WP-L2a `qsa_in` picks with the measured overrides; FlyTape's `ple_proj` runs its bf16 dense plan.
- The `ct1b` bracket (n07-17, round-1 builds with the per-barrier `buffer_inv sc1` that round 2 removed from A / E / G)
  gave the same head (0.23-0.34) and final mix (0.66-0.92) ranges; its PLE columns are the pre-fix builds (section 5).

## 5. What moved the numbers (ledger; base X = the build each verdict was measured on)

| change | base | n01-09 | n07-17 | verdict |
|---|---|---|---|---|
| no `buffer_inv sc1` after the walk barrier (`acq` default off; consumers in a pass read in-launch data with sc1 loads) | round-1 E / G / A | `acq` arm 1.33-3.09x | `acq` arm 1.33-3.11x | accepted (`acq` stays as a knob) |
| `waves=8` at TP2 r5 | WP-L2a pick + argmax | 0.930 | 0.923 | accepted (round 1) |
| `amrg=cta` + `map=xcd` at TP1 r5 | round 1 | 0.966 | 0.970 | accepted (round 2) |
| `amrg=cta` + `wcp=nt` at TP1 r128 | round 1 | 0.934 | 0.944 | accepted (round 2) |
| `amrg=cta` + `xu` at TP2 r5 | round 1 | 0.945 | 0.943 | accepted (round 2) |
| `amrg=cta` + `bpre=1` at TP2 r16 | round 1 | 0.960 | 0.961 | accepted (round 2) |
| `amrg=cta` + `xa=2` at TP1 r16 | round 1 (= round 2 for this cell) | 0.934 (round 2) | 0.939 (round 3) | accepted (round 3) |
| P `sb=False` at r8; `wcp=nt` at r32; `wcp=default` at r64 | WP-L2a qsa_in pick | 0.974; 0.971; 0.982 | 0.964; 0.991; 0.968 | accepted (rounds 2-3) |
| A `upc=2` (two units per CTA) at r128 | round 2 | 0.919 (round 4) | 0.925, 0.920, 0.921 (rounds 2-4) | accepted (round 4; `ple_lb.PAPPLY_PLANS`) |
| `amrg=cta` + `map=xcd` at TP2 r128 | round 1 | 0.962 (round 4) | 0.968 (round 4) | accepted (round 4) |
| `amrg=cta` + `bpre=1` at TP2 r128 | round 1 | 0.967 | 0.972 | WIN on both, behind `map=xcd`; kept as an arm |
| `amrg=cta` alone at TP2 r128 | round 1 | 0.967 WIN (round 2), 0.973 TIE (round 4) | 0.972 / 0.976 TIE, 0.965 WIN | not accepted; kept as an arm |
| `amrg=cta` + `waves=4` at TP2 r32 | round 1 | 0.979 / 0.973 TIE (rounds 2, 4) | 0.960, 0.965 WIN (rounds 3, 4) | not accepted (TIE on n01-09); kept as an arm |
| `amrg=cta` + `waves=4` at TP1 r32 | round 1 | 0.956, 0.952 WIN (rounds 2, 4) | 0.962, 0.955 TIE (rounds 3, 4) | not accepted (TIE on n07-17); kept as an arm |
| `amrg=cta` elsewhere | each round's pick | 0.95-1.01 | 0.96-1.01 | TIE: kept as arms |
| `akey=f32` (FlyTape's compare) | each pick | 0.98-1.00 TIE | 0.98-1.01 TIE | kept off (ATOM's bf16 key is the product) |
| E1 levers on the head (`mt=2`, `tb=2` / `4`, `wcp=default`, `sb`, `xu=False`, `bpre=0`) | each round's pick | 1.02-2.2 LOSS | 1.02-2.2 LOSS | kept as arms |
| `embed` 512 threads; `pgather` 256 threads; `nt` / no occupancy pad on E / G / A | round 1-4 bases | TIE-LOSS 1.0-1.65 | TIE-LOSS | kept as arms |

## 6. WaveScope and the reverse step

- WaveScope (`w10l2f-att1`, MI350X, CU 1, 4-head walks): head TP1 r16 / r128 against the logits-only E1 head of the same
  plan. r128: logits-only stall 59.0 % (vmem_store 13.6 %: the 63 MB logits store), fused 41.7 % (no store stall; s_barrier
  19.7 % incl. the epilogue's two LDS barriers 3.1 %); total stall 0.83 of the logits-only head's. r16: 0.91, dominated by
  the weight loads (`s_waitcnt vmcnt` + vmem_load 60-75 % of stall), as WP-L2a found for E1.
- Reverse (PLAN 16), FlyTape's tape ISA (`logs/q38/rv-ft1`, TP1 B16 / B64 builds): its `lm_head` carries 90-104 AGPR
  copies (`v_accvgpr` moves of a 253-VGPR tape-wide budget) and `select_token` 290 s_nop cycles; its head is three
  phases with two grid syncs (`lm_head`, `sample_reduce`, `select_token`); LB's is one phase with the merge in the
  epilogue (no grid sync between GEMM and argmax).
- Static counts of the PLE / embed stages (instructions without end padding; FlyTape's from the same tape split, LB's from
  the venv-built objects, `wp-l2f-raw/isa_ple.json`): FlyTape `embed` 182-197 instructions, `ple_gather` 386-393,
  `ple_stats` + `ple_apply` 1,732-1,774, every phase at the tape's 249-254 VGPR budget with 0-22 SGPR spill lanes
  (`v_writelane`) each; LB `embed` 240 (16 VGPRs), `pgather` 782 (62 VGPRs; the 16 heads' int64 hashes and the MXFP8
  pack are per thread), `prologue` 815 (85), `papply` 2,084 (158 VGPRs: two unrolled units) / 1,198 with `upc=2`
  (144), no spill, no scratch. The structural difference is the phase count, not the code: FlyTape runs gather, proj,
  stats, (reduce), apply, (copy) as 4-6 grid-synced phases; LB runs the prologue, P and A (3).

## 7. Open items

- Round 5 (both nodes, on the final tables) found no further change that WON on both nodes: the climb over these lever
  sets has converged; the one-node WINs it found (TP1 r8 `depth=2/3` + `amrg=cta`, TP1 r32 `waves=4` + `amrg=cta`) stay as
  arms for the next round on a new base (PLAN 15).
- Logprobs / non-greedy sampling pieces of the head (`DESIGN.md` WP8) not built; the final mixer without the injection
  dots (an hc_lb knob) not built.
- Integration (WP-L3 / L4 / L5): the PLE needs C's combine and C's norm split around A in the PLE layer (hc_lb fuses
  them); G and P can be scheduled before the PLE layer; the head reads the final mix's MXFP8 output directly.
- A conv state ring buffer (one entry written per step instead of the 9-entry shift) would cut A's traffic by about 40 %;
  it changes ATOM's `[slots, channels, 9]` state contract (prefill writes it), so it is a WP-L5 serving item.
- No served path runs these kernels yet, so no quick end-to-end number (PLAN 14): the pass (WP-L4 / L5) is where the
  head, the prologue and the PLE land.

## Verification (W10 independent verifier, 2026-11-02)

Verdict: **partial**. The gate holds: every family beats FlyTape's matching phases on both nodes, and the numerics suite
reproduces. Two claims are overstated: the PLE / embed / prologue timings were taken with warm caches, and "ATOM's
numerics" for the head covers the argmax rule but not the activation precision. Harness:
`hackathon-prep/qwen38/truemono-lb/w10-vWP-L2f/` (`ple_cold.py`, `job.sh`, `sync.sh`). Raw data (absolutes):
`hackathon-prep/private-perf/qwen38-true-mono/large-batch/vwp-l2f-raw/`.

What was checked:
- **Raw data reproduces the report.** `num_census.py` over the 15 raw job directories gives the same counts as section 2
  (4,118 / 0 head arm checks; 516 / 0 logits checks; 846 / 0 planted cases; 1,381 of 1,384 head mutants; 3,120 / 0 PLE
  arm checks; 1,770 of 1,776 PLE mutants; chain 65 of 66). `ct5a_report.json` and `ct5b_report.json` match the section 4
  table cell by cell. Both final brackets log `strict=True` for the head and load the PLE objects from `w10-l2f-hs13`
  (152 loaded, 0 refused). `xstale`, `xpeer` and `xorder` are caught in all 6 TP2 cells of both brackets.
- **The measured code is the delivered code.** The `head_lb.py` / `ple_lb.py` hashes logged by `ct5a`, `ct5b`, `r5a`
  and `r5b` equal the files in the tree, and `e1.py`, `e1_plans.py` and `hc_lb.py` are unchanged. The 32 CPU tests of
  `test_qwen38_lb_l2f.py` pass locally.
- **The final-mix comparison is fair or slightly favours FlyTape.** ATOM's final mixer (`hyper_connection_mixer`,
  `has_block_inject=False`) is combine + norm + low-rank down / up + gated mean. The LB HC instance also does the
  injection dots and writes the streams, so it does more work than the final mixer needs. The TP2 exchange caveat is
  already stated in the report.

**Re-run: the cheapest check that could flip the PLE claims (jobs `w10-vWP-L2f-ca` on n01-09 and `-cb` on n07-17, venv,
the final `hs13` objects loaded strictly).** `l2f_ple.py` replays each arm's graph back to back on the same ids, the same
state and the same inputs. In a served step the PLE runs once, after GBs of weights have streamed through, so the table
rows, the embed rows and A's conv state are cold. The wrapper reuses WP-L2f's bench unchanged and only swaps
`time_arms`. Each arm is timed in two modes, interleaved in the same 16 shuffled rounds:
- **warm:** WP-L2f's replay, unchanged.
- **cold:** before each replay, read 1 GiB and write 512 MiB of scratch to evict L2 / MALL. Then re-read only the inputs
  that are warm in a served step (A's y and r; the ids and context), and sync the device.

Numerics in both re-runs: 270 / 0 arm checks and 150 of 150 mutants.

| part (cold / warm, both nodes) | r5-r128 |
|---|---|
| embed | 1.05-1.11 |
| n-gram gather | 1.06-1.10 |
| prologue (E + G) | 1.06-1.10 |
| projection P (E1 walk, already rotates 4x the MALL: flush control) | 0.995-1.005 |
| apply A | 1.03-1.07 at r5-r32; 1.02 at r64; 1.005-1.01 at r128 (`upc=2`) |

The cold factors are carried onto each bracket's own ratios: each bracket's warm ratio times the cold / warm factor
measured on the same node. Venv and container times for the same objects differ by up to 1.24x, so the venv times are
never divided by FlyTape's container times directly.

| family | report (warm) | cold-cache |
|---|---|---|
| e + PLE, TP1 r5-64 | 0.51-0.58 | **0.54-0.59** |
| e + PLE, TP2 r5-128 | 0.44-0.54 | **0.45-0.57** |
| prologue (E + G), TP1 | 0.67-0.84 | 0.73-0.88 |
| prologue (E + G), TP2 | 0.23-0.71 | 0.25-0.77 |
| A, TP2 r64 | 0.97-1.00 | 0.99-1.01 (n07-17: 1.01, behind FlyTape) |
| A, TP2 r128 | 1.07-1.10 | 1.08-1.10 |
| A, elsewhere | 0.80-0.98 | 0.84-0.99 |

The head (L rotates 4x the MALL of weights) and the final mix (WP-L2b's `--time-layers 96`) were already cold and were
not re-run.

Issues:
1. **Warm-cache timing of E / G / prologue / A.** The section 4 per-phase ratios are 2-10 % optimistic. The family gate
   still holds, but section 4's "everywhere else A is 0.80-1.00" becomes 0.84-1.01. At TP2 r64 A is level with FlyTape,
   and behind it by 1 % on n07-17, so A trails FlyTape at r64 as well as at r128. Future PLE / embed timing should flush
   between replays, or rotate ids and state over more than the MALL.
2. **Head numerics are not ATOM's at the activations.** The fused head runs E1 MXFP4 x **MXFP8** (the final mix's
   `xq_pass`). ATOM's mono head (`qwen4_exp_mono/head.py`, `Mxfp4LMHead`) keeps the **bf16 activations**
   (`tl.dot_scaled` bf16 x e2m1, W4A16) up to 16 rows, and uses the bf16 GEMM on the same MXFP4-valued weights above
   that. Section 2's "0 flips against ATOM's argmax of the float64 logits" is computed from the MXFP8-dequantized
   activations (`l2a_bench.Cell._ref` uses `deq_x`), so it checks the argmax rule (key, ties, merge), not token parity
   with ATOM's head. The MXFP8 rounding of the final hidden state can flip near-tie tokens against ATOM, and that was
   never measured. The precision is inherited from WP-L2a's confirmed E1 head (DESIGN-LB "checkpoint precision"). It
   needs an accuracy gate (a real-checkpoint top-1 agreement against ATOM's head) or an `act=bf16` head knob before the
   head can be called ATOM-exact.
3. **The head ratio is mostly bytes.** FlyTape's `lm_head` is bf16 (DESIGN-LB section 2: "head bf16"), so 0.23-0.33
   is largely the 4x weight-byte cut of the MXFP4 head. That is legitimate under the checkpoint-precision regime, but
   section 4 does not say so.
