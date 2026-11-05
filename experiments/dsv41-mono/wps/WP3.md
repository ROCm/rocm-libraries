# WP3: fewer bytes per accepted token (DeepSeek-V4.1-Flash, TP2 DSpark, MI355X)

W12 work package 3 (D0 section 4). Ratios only. Absolute numbers and the raw logs are in
`hackathon-prep/private-perf/dsv41-mono/raw/wp3-*` (`*.abs.md`, `*.summ.json`).

What WP3 delivers:

- **Code** (all knobs default off; unset = never imported, byte-identical behaviour):
  - ATOM knobs: `dev/ATOM-dsv41/atom/models/deepseek_v41/rk/wp3.py` and
    `dev/ATOM-dsv41/atom/models/deepseek_v41/rk/wp3_quant.py`, plus one hook in
    `dev/ATOM-dsv41/atom/model_engine/model_runner.py`.
  - Mono MXFP4 GEMV stages: `dev/ATOM-dsv41/atom/models/deepseek_v41/mono/kernels/rk_w4.py`, plus additive hooks in
    the same directory's `attn_pre.py`, `attn_post.py`, `layer_post.py` and `moe.py`, and in
    `dev/ATOM-dsv41/atom/models/deepseek_v41/mono/weights.py`.
  - The ATOM-dsv41 tree is shared with W13 / W14. Their hooks have since landed in the same mono kernel files, so the
    knob-off identity recorded for WP3 (section 2.3) is re-checked on the current tree in job J7 (section 7).
- **Harness:** `hackathon-prep/dsv41/wp3/`.

RAGGED and the target-only MXFP4 scope: section 4.4 (jobs J5b, J6, J8b).

**Verifier round 1 (what changed in this revision):**
- Accuracy is now judged on the literal gate only. `QHEAD` alone and `QSHARED` alone **fail** it (1-4 questions under
  the reference minimum), and so does one of three runs each of the two J4 unions (section 3). The binomial rule is
  withdrawn.
- The K1 rig claim is restated (section 2.3). The FP4-vs-FP8 differences are deterministic (J7: each build bitwise
  over 8 repeats). The tolerance was set after a FAIL and is reported as an empirical bound. The mutants clear it by
  four orders of magnitude. The "seed-nondeterministic" rows-6 result was two different rig modes.
- The served-band wording is corrected: cells scatter beyond the twin band in both directions (sections 0, 4.1, 4.2).
- The AL cost was not resolved by J4 alone. Pooled with J6 and J8b it is about 1.5 % at C1-C4 (section 4.4); its
  cause is still a hypothesis.
- J6 (STS-calibrated RAGGED, n01-09) and J8b (n07-17: second-node W4 check, a recompiled no-op control, a sync-free
  STS apply) are done (section 4.4). They found a third ATOM defect, a per-step host sync in the STS apply, and added
  the default-off workaround `ATOM_DSV41_RK_RAGGED_STS_APPLY=nosync`.
- Knob-off identity is re-confirmed on the current shared tree, W13 / W14 hooks included, in the container (J7, J8,
  J8b; section 2.3). The file path of `rk_w4.py` is corrected above.

## 0. Verdict (read first)

**WP3 does not move the served per-request rate measurably at any CONC.** That holds for every lever and for their
union, at TP2 C1-C16 and at TP1. The predicted C1 gain was 1.18 (1.08-1.35). What was measured:

- Served tok/s/req is too noisy to resolve the effects. Lever cells scatter beyond the A-vs-A twin band in **both**
  directions (J3: twin 0.944-1.022, lever cells 0.855-1.083, 14 of 48 cells outside; J4: twin 0.969-1.051, quant and
  union cells 0.936-1.067, 11 of 30 outside), with no lever consistently on one side. The exception is DLEN 1-3 at
  C1-C2 in J4 (0.60-0.96): a real loss, matching the per-token table in 4.2.
- The low-noise metric is decode-step GPU time per accepted token (job J4: no-profiler host timeline, three rotated
  passes; its A-vs-A twin lands within 0.4 % on the step). It shows:
  - the best gated union (MXFP4 `wq_b` / `wo_a` / `wo_b` / shared + FP8 heads) cuts the decode step by **1.3-2.0 %**
    at C1-C4, 0.7 % at C8 and 0.3 % at C16 on n01-09 (J4; J6 repeats it). On n07-17 (J8b), against a recompiled
    no-op control, it is 0.8 / 1.4 / 1.6 / 1.0 / 0.0 % at C1 / C2 / C4 / C8 / C16. So the cut is real on both nodes
    and is not a recompile artefact;
  - in each single job its acceptance length overlaps the twin's spread. Pooled over the three jobs that ran it, 8 of
    its 9 C1-C4 cells lie below every twin cell (means 0.986 vs 1.003): it costs **about 1.5 % of AL** at C1-C4.
    Whether the target's or the draft's requantized linears cause that is a hypothesis (the target-only scope cannot
    run with the mono);
  - net per accepted token: **0.972-1.012 at C1-C16** across J4 / J6 / J8b, overlapping the twins' 0.984-1.011.
    No gain is resolved.

Why the byte cuts do not turn into time:

- **The C1-C4 step is not byte-bound** (P1: 2.1-2.4x the bytes floor). The mono MXFP4 GEMVs make K1 3.3-6 % faster
  in isolation (`wq_b` alone). K1 is about 0.15 of the step, and K2's FP8 GEMVs would take about 0.05 of it even at the full HBM stream (an estimate from their bytes). So even a
  perfect halving of those bytes is worth at most about 3 % of a C1 step.
- **K2 is routed-MoE-bound even at C1.** The six verify rows touch about 22 experts a layer, and those are already
  MXFP4. WP3's cut that reaches the MoE is the verify width (WP3c), and it costs more acceptance than it saves bytes.
- **The heads are the only purely byte-bound items the WP touches.** They are about 0.05 of a C1 step: FP8 halves them
  for a gain of about 2 %, inside the twin noise.

The 3x goal is not reachable through this WP. That agrees with D0 section 0: bytes cap the C1-C4 ceiling, and the
cuts that do not need an accuracy argument are the small ones.

Accuracy (section 3), on the literal gate (not below the lowest of the four reference runs, flexible and strict):
- **pass:** `QPROJ=mxfp4`, `QPROJ+QSHARED`, and the full union `QPROJ+QSHARED+QHEAD`;
- **fail:** `QHEAD=fp8` alone (1 question under on flexible, 4 under on strict) and `QSHARED=mxfp4` alone (4 under on
  strict); also one of three J4 runs each of `QPROJ=mxfp4a+QSHARED+QHEAD` (1 / 2 under) and of the union `u1`
  (3 under on flexible).

So "the gated byte cuts pass accuracy" holds only for the three passing arms. The passing cuts buy almost nothing in
time on today's latency-bound step. They become worth more once WP2/WP4/WP5 take the step toward the floor; the
integration stage re-tests them there (lookback).

## 1. Levers, knobs, status

| knob | what | class | built / served | status (base = ATOM df86c482 TP2 mono + DSpark) |
|---|---|---|---|---|
| `ATOM_DSV41_RK_DVOCAB=<k>` (+`_FILE`, `_BALANCE`, `_MUTANT`) | draft head and Markov `W2` over the k most frequent target tokens, split evenly over the TP ranks | exact (output); acceptance-sensitive | yes, TP1 and TP2 | **rejected on base B0**: AL falls by more than the step gains at every k tried (16k / 32k / 48k / 64k) |
| `ATOM_DSV41_RK_SHARD=engram:main_proj[:wqkv_a]` | column-shard replicated FP8 linears and all-gather; used only at row counts proven bitwise in-process | exact (bitwise) | Engram `wkv`, draft `main_proj`, `wqkv_a` (native path only: the mono K1 reads the full weight) | **rejected on base B0** (inside noise) |
| `SHARD` `router` / `mhc` | | | **not built** | decode router is ATOM's fixed-order 384-expert FlyDSL GEMV; `hc_fn` sits inside aiter's fused seam op. Each needs a new kernel (WP4 / WP5 territory) |
| `ATOM_DSV41_RK_QHEAD=fp8` (+`_SCOPE`, `_FREE`) | target and draft LM heads in E4M3 with 32x32 E8M0 (ATOM's dense format and GEMM) | accuracy-gated | yes, TP1 and TP2 | **fails the literal accuracy gate** (1 question under the 4-run reference minimum on flexible, 4 on strict); passkey 20/20. Step 0.993-1.000, per accepted token inside noise. **Kept default-off, rejected on base B0 (accuracy)**; it passes only as part of the union below |
| `ATOM_DSV41_RK_QPROJ=mxfp4` | `wq_b`, `wo_a`, `wo_b` in MXFP4 (W4A8), served by new mono GEMV stages (`rk_w4.py`); prefill stays FP8 | accuracy-gated | yes (mono decode, rows <= 48) | **passes the gate** (alone, with QSHARED, and in the union with QSHARED + QHEAD); step 0.988-0.994 at C1-C4 (with QSHARED), per accepted token inside noise: **kept default-off** |
| `ATOM_DSV41_RK_QPROJ=mxfp4a` | the same plus K1's replicated `wqkv_a` | accuracy-gated | yes | **fails the literal gate in 1 of 3 runs** (J4 pass 1: 1 question under on flexible, 2 on strict; passes 2-3 inside or above); no gain over `mxfp4` (K1 rig 0.958 vs 0.956 at 6 rows): **kept default-off, rejected on base B0** |
| `ATOM_DSV41_RK_QSHARED=mxfp4` | shared expert `gate_up` / `w2` in MXFP4 (mono K2b) | accuracy-gated | yes | **fails the literal gate alone** (strict-match 4 questions under the reference minimum); **passes** combined with QPROJ (and with QPROJ + QHEAD). Speed with QPROJ above. **Kept default-off; alone: rejected on base B0 (accuracy)** |
| `QPROJ` / `QSHARED` with `ATOM_MONO_ENABLE=0` (or `ATOM_DSV41_RK_QPROJ_NATIVE=1`) | the same weights through ATOM's Triton W4A8 GEMM, prefill included | accuracy-gated | yes | **rejected**: the Triton GEMM is 6-30x slower than the FP8 GEMM at decode widths; with prefill in FP4 too, GSM8K strict-match moves outside the range (section 3) |
| `ATOM_DSV41_RK_DLEN=<k>` | verify only the first k (1..5) draft tokens: ATOM's q-bucket path, constant k | acceptance-sensitive | yes, k = 1..5 | **rejected on base B0**: per accepted token 1.04 (k = 4) to 1.51 (k = 1) at C1. At C >= 4 the step does not shrink: on ATOM's q-bucket path a step is at least every request's (previous bonus + 1) wide |
| `DLEN` 6, 7 | | | **not buildable** | DSpark drafts a fixed block of 5 (the checkpoint's `dspark_block_size`; the noise-token block and `validate_speculative_config`); more than 5 needs a retrained drafter |
| `ATOM_DSV41_RK_RAGGED=bucket` (+`_BUCKETS`, `_PROFILE`, `_CALIB`) | per-step verify width from the DSpark confidence head (ATOM's prefix scheduler) on the batch-uniform q-bucket graphs | acceptance-sensitive | yes | **rejected on base B0** (section 4.4): with J4's measured SPS profile 1.000-1.016 per accepted token (J6); with fitted STS temperatures 1.10-1.17 (J6, ATOM's per-step sync in the STS apply) and 1.006-1.027 with the sync-free apply (J8b); can act only at C1-C2 (the width floor above) |
| per-request ragged verify | | | **not buildable on this base** | ATOM's `ragged` path is a no-op under `--cudagraph-mode FULL` (it needs PIECEWISE graphs, compile level 3, which stalls for V4.1); under FULL graphs it needs WP2's device-side step metadata (the D0 dependency) |

## 2. What was built

### 2.1 Draft vocabulary subset (DVOCAB, exact)

- **Head.** The draft head reads `k` rows of the BF16 head, gathered once at load. With `QHEAD` the subset is FP8 and
  padded to 512 rows.
- **Markov argmax.** A copy of ATOM's Markov argmax stage 1 publishes each winner's global id through a sorted id map.
  Stage 2 and the TP pair merge are ATOM's own, so a tie still goes to the lowest id.
- **Balanced subsets.** Frequent ids crowd the low end of the vocab, which is rank 0's head shard: a shard-local subset
  of 32k left rank 0 with 31.9k rows and rank 1 with 0.9k. The subset is therefore split evenly over the ranks, and
  each rank fetches the rows it does not own once at load.
- **Ranking.** Rows are ranked by how often the target emits each token, re-tokenized from the target's own replies to
  1024 GSM8K **train** prompts and 360 mixed-domain prompts. Tokenizer added / special tokens come first. The eval sets
  are disjoint from both.
- **Coverage of the eval replies** (job J3):

  | k | 16k | 32k | 64k | 98k |
  |---|---|---|---|---|
  | GSM8K test | 0.963-0.972 | 0.976-0.983 | 0.989-0.993 | 0.997-0.999 |
  | non-GSM8K eval | 0.861 | 0.928 | 0.971 | 0.990 |

  Every miss is a draft position that cannot be accepted. That is why AL falls faster than the draft head's bytes.

### 2.2 Column shards (SHARD, exact)

- **Mechanism.** The rank computes its N/2 columns of the native FP8 GEMM, then aiter's custom all-gather.
- **Exactness.** The GEMM's kernel choice (tile, split-K) is a function of (M, N, K), so the shard is bitwise only at
  some row counts:
  - Engram `wkv` and `wqkv_a`: every probed M;
  - `main_proj`: M <= 48 only (M = 96 and 512 differ).

  The served shard therefore probes M = 1..96, 112, 128, 192, 256 and 384 in-process at load, on the real weights.
  It keeps only the counts that are bitwise on **every** rank (the all-gather is collective), and uses the full GEMM
  elsewhere.
- **Gate.** The in-server check is bitwise at M = 1 / 6 / 48 on all 43 sharded linears, 129 of 129.
  `ATOM_DSV41_RK_SHARD_MUTANT=all_rows` is the must-fail twin.

### 2.3 MXFP4 GEMV stages in the mono kernels (QPROJ / QSHARED, gated)

The mono decode (K1 / K2, FlyDSL) reads FP8 32x32 weights in five GEMV stages. `rk_w4.py` gives
`gemv_fp8_loads` / `gemv_fp8_mfmas` an MXFP4 operand, chosen per stage at build time.

- **Weights.** The routed experts' layout: (16, 16)-shuffled packed E2M1, so a lane reads its row's 32 contiguous K
  values (the f8f6f4 FP4 A operand). Scales are row-major E8M0 [N, K/32]: a lane takes byte lane/16 of its row's step
  dword.
- **Unchanged.** The MXFP8 activation operand, its scale, the per-wave K order, the row sums and every epilogue and
  hand-off are the FP8 stage's.
- **Quantization.** The weights are the checkpoint FP8 weights, dequantized exactly and requantized with ATOM's own
  `quantize_fp4`, once at load. Prefill keeps the FP8 modules.
- **Build keys.** The kernel name and build key gain `_w4<stages>` only when a stage is on.
- **Knob off.** Two checks.
  - Local trace (Nov 3, the tree as WP3 left it): the traced MLIR of K1 and K2 at 6 rows was identical to pristine
    df86c482 (`cmp` of the dumps). The laptop FlyDSL can no longer trace these kernels ("Cannot select: intrinsic
    permlane32.swap"), so this could not be repeated locally.
  - **Job J7 (in the container, the served FlyDSL):** K1 and K2 at 6 and 12 rows, every `ATOM_DSV41_RK_*` unset,
    built from three trees: pristine df86c482 (`git archive`), the J1-J6 stage `atom-wp3-a6q2`, and the **current
    shared tree** (`atom-wp3-a10cur`, which now also holds the W13 / W14 hooks in `attn_pre`, `attn_post`,
    `layer_post` and `moe`). All 23 IR stages per kernel (MLIR passes, LLVM IR, final ISA) are **byte-identical**
    across the three trees, 4 kernels x 23 files. Jobs J8 / J8b repeated the trace on the final tree (`atom-wp3-a11`: the
    current tree plus the `RAGGED_STS_APPLY` hook of section 4.4): K1 and K2 at 6 rows, knobs off, are again
    byte-identical to pristine (46 of 46 files). The must-differ twin, the same trace with `QPROJ=mxfp4` and
    `QSHARED=mxfp4` on, builds differently named kernels (`..._w4wqb`, `..._w4sgusw2woawob`) with a different final
    ISA, so the check can fail. Scope: this shows the knob-off **source** generates the same IR and ISA. The served binary also carries
    the process-level compile randomness of these builds (alola rule 137: the dump path is deterministic, the normal
    path is not), which affects every tree alike.

Layer gate: `wp3_k1rig.py`, a rig of ATOM's real K1 (built on P2's rig).
- **Setup.** The weights are E2M1 codes with one E8M0 per 32x32 block, which FP8 represents exactly. The FP8 K1 and the
  MXFP4 K1 therefore multiply identical values.
- **Result (restated).** The MXFP4 K1 is **not** bitwise equal to the FP8 K1 on every input. What was measured:
  - `wq_b` only (`QPROJ=mxfp4`): rows 1 bitwise; rows 5 / 6 / 12 differ in **one** `q_out` element at 1 bf16 ulp, in
    one of the three seeds (seed 0), every time; all other outputs and the mailbox scratch bitwise.
  - `wq_b` + `wqkv_a` (`QPROJ=mxfp4a`): rows 6 bitwise; rows 12 differ in two seeds: 2 and 3 `q_out` elements at
    1 ulp, plus **one mailbox-scratch byte** in seed 0.
  - **The differences are deterministic.** Job J7 (`wp3_k1rig_aa.py`) ran each build 8 times per seed on the same
    inputs: FP8 vs FP8 and MXFP4 vs MXFP4 are bitwise in every repeat (6 configurations x 3 seeds x 7 pairs), and the
    FP4-vs-FP8 differences above appear in 8 of 8 repeats, always the same elements. The earlier report of a
    "seed-nondeterministic" rows-6 result compared two different modes: J3 ran `wq_b` only (seed 0 differs), J4 ran
    `wq_b` + `wqkv_a` for every width (rows 6 bitwise). J7 reproduces both.
- **Tolerance.** The gate's tolerance ("bitwise except `q_out` within 1 ulp on <= 1e-4 of its elements"; with
  `wqkv_a`, every buffer within 1e-3 of its bytes) was set **after** the first run failed bitwise (J2b rows 5 / 6).
  It is an empirical bound, not one derived in advance. What supports it independent of that failure:
  - each build is deterministic (above), so the difference belongs to the FP4-vs-FP8 operand pair, not to noise. The
    likely mechanism is that the scaled MFMA need not sum a block in the same order for an FP4 and an FP8 operand. That
    is an inference: no ISA document was found that states it;
  - the observed differences are at most 3 elements at 1 ulp (<= 1.5e-5 of `q_out`) and one scratch byte. That is two
    orders of magnitude inside the 1e-3 allowance, which is looser than it needs to be;
  - **the mutants clear the tolerance by four orders of magnitude** (J7): `q_out` max error 33,796-35,129 ulp, with
    0.30 (scale bump) to 0.999 (nibble swap, transpose) of the elements wrong. The old catch rule was "any byte
    differs", which the 1-ulp differences alone would also trip; J7 measures the margin over the tolerance instead.
- **Mutants.** All three must-fail mutants are caught at every width: one scale code bumped in every 7th row group,
  nibbles swapped, 16x16 byte blocks transposed.
- **K2 coverage.** The rig covers the per-step load and MFMA code that every stage shares. K2's `wo_a` (split-K,
  tiled) and the shared expert are covered by the accuracy gate. K2 needs a two-GPU rig, the same limit as P2.

### 2.4 Verify width (DLEN / RAGGED)

- **Mechanism.** Both run on ATOM's batch-uniform q-bucket verify path. This needs an extra captured graph set per
  bucket, plus mono builds for the extra widths. A WP3 `VerifyScheduler` subclass is installed on the drafter: DLEN
  returns a constant `ell`, RAGGED runs ATOM's prefix scheduler.
- **Three ATOM defects found and worked around:**
  - `_maybe_calibrate_dspark_sps` dies at startup on df86c482 (`PublicationError: owner is sealed`). The WP3 scheduler
    carries a profile tag, which skips it.
  - Replaying the captured (bs, q) graphs after capture to time them faults (`hipErrorIllegalAddress`): their step
    metadata is the last captured shape's. This is kept as the default-off `ATOM_DSV41_RK_RAGGED_CALIB=replay`,
    rejected.
  - With STS temperatures set, `calibrate_confidence` syncs device to host every decode step (`torch.any(t <= 0)`):
    J6 measured a ~1.10x step at every width. Worked around by the default-off
    `ATOM_DSV41_RK_RAGGED_STS_APPLY=nosync` (section 4.4).
- **Calibration.** RAGGED takes a measured SPS(B) profile, `wp3_profile.py`, built from the decode-step GPU times of
  the DLEN sweep (host timeline). Without a profile it uses ATOM's synthetic stub, which over-truncates at small batch.
  Fitted STS temperatures (`wp3_sts.py`, from a full-width run's logged confidence rows) are optional; J6 fitted
  them.

## 3. Accuracy gate (GSM8K 1319, 3-shot chat, flexible + strict; passkey 20 x 8K / 24K)

Same harness as B0 / P3: lm_eval at CONC 16, one fresh server per arm, jobs J2 + J2b on n07-17 (J4 on n01-09 for the
two three-run rows).

The reference spread comes from four runs of the unmodified model:
- two A-vs-A pairs, mono (`base`, `base_b`) and native (`nat`, `nat_b`);
- flexible-extract range 0.0046 wide;
- strict-match range 0.0099 wide.

**The gate is the literal one stated by the task: an arm passes when neither score is below the lowest reference
run.** The table's differences are against that lower end: negative means below every reference run, and the count
in brackets is the number of GSM8K questions (n = 1319). A score above the top of the range is not a failure on this
one-sided gate, but it is flagged ("above"). An earlier version of this report labelled `QHEAD` and `QSHARED` PASS
under a binomial 2-sigma rule (0.022 / 0.026 at n = 1319). That rule was chosen after the scores were seen and is not
the stated gate, so it is withdrawn.

| arm | served path | flexible vs ref min | strict vs ref min | passkey 8K / 24K | verdict (literal gate) |
|---|---|---|---|---|---|
| refs (4 runs) | mono / native | 0 .. +0.0046 | 0 .. +0.0099 | 20/20, 20/20 | spread |
| `QHEAD=fp8` | mono | **-0.0008** (1) | **-0.0030** (4) | 20/20, 20/20 | **FAIL** (1 / 4 questions under) |
| `QPROJ=mxfp4` | mono W4 decode | 0 | +0.0008 | 20/20, 20/20 | **PASS** (inside) |
| `QSHARED=mxfp4` | mono W4 decode | +0.0099 (above) | **-0.0030** (4) | 20/20, 20/20 | **FAIL** (strict 4 questions under) |
| `QPROJ+QSHARED` | mono W4 decode | +0.0091 (above) | +0.0076 | 20/20, 20/20 | **PASS** |
| `QPROJ+QSHARED+QHEAD` | mono W4 decode | +0.0053 (above by 1 question) | +0.0061 | 20/20, 20/20 | **PASS** |
| `QPROJ=mxfp4a+QSHARED+QHEAD`, 3 runs (J4) | mono W4 decode | **-0.0008** (1) / +0.0030 / +0.0091 (above) | **-0.0015** (2) / +0.0106 (above) / +0.0053 | 20/20, 20/20 (x3) | **FAIL** in run 1, PASS in runs 2-3 |
| union `u1` (above + SHARD + DVOCAB 64k), 3 runs (J4) | mono W4 decode | **-0.0023** (3) / +0.0030 / +0.0008 | +0.0023 / +0.0114 (above) / +0.0061 | 20/20, 20/20 (x3) | **FAIL** in run 1, PASS in runs 2-3 |
| `QPROJ=mxfp4` native (Triton, prefill FP4 too) | native | -0.0076 (10) | **+0.165** | 20/20, 20/20 | **FAIL**: the reply format moves (strict-match more than doubles), flexible 10 questions under |
| `QSHARED=mxfp4` native | native | -0.0076 (10) | **-0.057** | 20/20, 20/20 | **FAIL** |
| mutant: transposed blocks (native QPROJ) | native | -0.905 | -0.123 | 0/5, 0/5 | caught |
| mutant: scale code +1 in every 7th block (mono QPROJ+QSHARED) | mono | +0.025 (100 Q) | +0.107 (100 Q) | 5/5, 5/5 | **not caught end to end**; caught by the layer numerics gate (rel. error 0.40 vs bound 0.20) |
| mutant: scale code +1 (QHEAD) | mono | +0.005 (100 Q) | +0.117 (100 Q) | 5/5, 5/5 | **not caught end to end**; caught by the layer numerics gate (0.39 vs 0.08) |

Layer numerics gate (in-server, `ATOM_DSV41_RK_WP3_CHECK=1`, every layer-0 linear on random rows): this is the gate
for the mutants the end-to-end check cannot see. The P3 lesson again: GSM8K / passkey are insensitive to an error on
one block in seven.

- `QHEAD`: relative error 0.038-0.041, bound 0.08, top-1 agreement 0.875-1.0 on random rows. Mutants: 0.38 / 1.40.
- MXFP4 `wq_b` / `wo_a` / `wo_b` / shared vs their FP8 weights: 0.123-0.135, bound 0.20. Mutants: 0.40 / 1.49.

What the literal gate says, and what it does not:
- **Passing arms:** `QPROJ=mxfp4`, `QPROJ+QSHARED` and `QPROJ+QSHARED+QHEAD` (one run each). These are the only
  configurations for which "the gated byte cuts pass accuracy" is literally true.
- **Failing arms:** `QHEAD` alone and `QSHARED` alone, by 1-4 questions, and one of three runs of the two J4 unions.
  They stay default-off and are recorded "rejected on base B0 (accuracy, literal gate)".
- **The reference range is narrow for its sample.** The J4 unions ran three times each with identical settings, and
  their own flexible scores span 0.0053 (`u1`) and 0.0098 (`mxfp4a`), wider than the four-run reference range
  (0.0046). So a 1-4 question shortfall is weak evidence of real harm. It is still a failure on the stated gate.
  Settling it needs the failing arms re-run together with more reference runs **in one job**. That was not run: the
  workflow's GPU budget was fully used by J6 and the other packages. It is the first item for the lookback.

Decode-only FP4 (prefill FP8) keeps the reply distribution; FP4 in prefill too does not. The served configuration is
therefore the mono one, and the native QPROJ / QSHARED stay default-off and rejected.

## 4. Speed

### 4.1 Served per-request tok/s (jobs J1, J3, TP1; n01-09 / n07-17)

- **Twin band.** In J3 (two rotated passes, fresh server per arm-run, CONC 1/2/4/8/16 on GSM8K plus CONC 1 on the
  non-GSM8K set) the A-vs-A twin lands at 0.944-1.022 of `base` across the cells. The lever cells scatter beyond
  that band in **both directions**: 14 of 48 cells are outside it, 9 below and 5 above.
  - `QPROJ+QSHARED`: 0.929-1.017 (C4 0.929 below);
  - `QPROJ+QSHARED+QHEAD`: 0.925-1.020 (C2 0.925 below);
  - `QPROJ` alone: 0.924-1.053 (C2 below; C1 1.026 and C16 1.053 above);
  - `QSHARED` alone: 0.951-1.083 (C1 1.026 and C16 1.083 above);
  - `QHEAD`: 0.936-1.035 (C2 0.936 below, C1 1.035 above);
  - `SHARD`: 0.861-1.014 (C2 0.861 and C16 0.932 below);
  - `DVOCAB 64k`: 0.896-1.009 (C2 0.896 and C16 0.928 below);
  - `DLEN 4`: 0.855-1.012 (C2 0.855 below).

  No arm is outside on the same side in most of its cells, and the C2 cell is low for almost every arm (a base-run
  effect shared by the pass), so these excursions are read as noise, not as lever effects.
- **Noise check.** At CONC 16 the mono does not serve (96 rows), so the MXFP4 stages are not even in the step. There
  the four MXFP4 arms still read 1.001-1.083: a direct measure of the noise. The single-pass J1 smoke had the same picture.
- **Verdict.** The served metric cannot resolve effects of a few percent with this cell size. That is why J4 adds the
  host timeline.

Full ratio tables: `private-perf/dsv41-mono/raw/wp3-j3.ratio.md`, `wp3-j1.ratio.md`, `wp3-tp1a.ratio.md`.

### 4.2 Decode-step GPU time per accepted token (job J4, host timeline)

Job J4 ran on n01-09: 11 arms, three rotated passes, a fresh server per arm-run, and the B0 cells plus non-GSM8K C1.
Each server ran P1's no-profiler host timeline (`ATOM_DSV41_RK_HOSTTL`).

How the numbers are formed:
- **Step.** For each arm-run, the median GPU time of a pure-decode graph step at bs = CONC, on the slower rank, is taken
  over hundreds to thousands of steps per cell. Foreign prefills and client effects are excluded by construction.
- **Ratios** are per pass against the same pass's `base`, then the median over the passes.
- **AL** is pooled over the passes.
- **Per accepted token** = step ratio / AL ratio.

**Decode-step GPU time vs base**

| arm | C1 | C2 | C4 | C8 | C16 |
|---|---|---|---|---|---|
| A-vs-A twin | 0.998 | 1.003 | 0.998 | 1.001 | 0.996 |
| QHEAD | 0.994 | 0.998 | 0.993 | 0.998 | 1.000 |
| QPROJ + QSHARED (mono W4) | 0.994 | 0.988 | 0.992 | 0.998 | 0.997 |
| QPROJ + QSHARED + QHEAD | **0.987** | **0.981** | **0.983** | 0.993 | 0.997 |
| QPROJ=mxfp4a + QSHARED + QHEAD | 0.988 | 0.981 | 0.980 | 0.991 | 1.011 |
| union u1 (above + SHARD + DVOCAB 64k) | 0.989 | 0.980 | 0.978 | 0.996 | 1.016 |
| DLEN 4 | 0.979 | 0.947 | 0.964 | 0.994 | 0.996 |
| DLEN 3 | 0.920 | 0.855 | 0.966 | 0.998 | 0.999 |
| DLEN 2 | 0.851 | 0.801 | 0.992 | 1.000 | 0.991 |
| DLEN 1 | 0.807 | 0.739 | 0.999 | 0.998 | 0.996 |

**Acceptance length vs base**

| arm | C1 | C2 | C4 | C8 | C16 |
|---|---|---|---|---|---|
| A-vs-A twin | 0.996 | 1.005 | 1.003 | 1.014 | 1.009 |
| QHEAD | 0.983 | 0.998 | 1.013 | 1.012 | 1.013 |
| QPROJ + QSHARED | 0.973 | 1.020 | 0.977 | 0.995 | 1.009 |
| QPROJ + QSHARED + QHEAD | 0.987 | 1.009 | 0.982 | 0.989 | 1.013 |
| QPROJ=mxfp4a + QSHARED + QHEAD | 0.978 | 0.990 | 0.986 | 0.997 | 1.011 |
| u1 | 0.968 | 1.010 | 0.976 | 0.988 | 0.993 |
| DLEN 4 / 3 / 2 / 1 | 0.942 / 0.836 / 0.702 / 0.535 | 0.955 / 0.881 / 0.782 / 0.666 | 0.948 / 0.915 / 0.899 / 0.870 | 0.981 / 0.965 / 0.945 / 0.904 | 0.998 / 0.987 / 0.975 / 0.974 |

**Decode GPU time per accepted token vs base** (the WP3 metric: < 1 is better)

| arm | C1 | C2 | C4 | C8 | C16 |
|---|---|---|---|---|---|
| A-vs-A twin | 1.002 | 0.998 | 0.994 | 0.987 | 0.987 |
| QHEAD | 1.011 | 1.000 | 0.980 | 0.986 | 0.987 |
| QPROJ + QSHARED | 1.022 | 0.969 | 1.015 | 1.003 | 0.988 |
| QPROJ + QSHARED + QHEAD | 0.999 | 0.972 | 1.002 | 1.004 | 0.983 |
| QPROJ=mxfp4a + QSHARED + QHEAD | 1.011 | 0.991 | 0.994 | 0.994 | 1.000 |
| u1 | 1.022 | 0.971 | 1.002 | 1.008 | 1.024 |
| DLEN 4 | 1.039 | 0.992 | 1.016 | 1.013 | 0.998 |
| DLEN 3 | 1.100 | 0.971 | 1.056 | 1.033 | 1.012 |
| DLEN 2 | 1.212 | 1.025 | 1.103 | 1.058 | 1.016 |
| DLEN 1 | 1.508 | 1.110 | 1.148 | 1.105 | 1.023 |

Reading:

- **Real step cuts.** The MXFP4 GEMVs plus FP8 heads cut the step by 1.3-2.0 % at C1-C4. That is a measured effect
  well outside the twin's 0.2-0.4 %. Two in-situ views of the same lever:
  - **kineto, one C1 decode window each.** GPU busy time per step is 0.954 of base; by kernel, K1 is 0.946-0.965,
    target K2 0.957 and the head GEMM about 0.7.
  - **host timeline.** The step is 0.987. The kineto window is one request's 49 steps, whose MoE work depends on the
    tokens; the host-timeline median runs over thousands of steps.
- **An AL cost is not resolved in J4 alone.** The union's AL is 0.982-1.013 of base across J4's cells; the twin's own
  AL ratio ranges 0.996-1.014, and the ranges overlap. It **is** resolved once J6 and J8b are pooled in (section 4.4:
  8 of 9 C1-C4 cells below every twin cell, about 1.5 % of AL). The cause (target or draft requantization) remains a
  **hypothesis**: the test that would split them (`ATOM_DSV41_RK_QSCOPE=target`) cannot run with the mono.
- **Per accepted token every quant arm lands at 0.97-1.02**; the union at 0.972-1.004, against the twin's
  0.987-1.002. The ranges overlap: no lever and no union shows a resolved gain or loss per accepted token.
- **Served tok/s/req** (same job, three passes) is noisier still. The twin reads 0.969-1.051 (pass ranges
  0.936-1.054). The quant and union cells scatter beyond it in both directions (0.936-1.067, 11 of 30 outside; e.g.
  `mxfp4a` C16 0.952 below, the union non-GSM8K C1 1.067 above). Full table:
  `private-perf/dsv41-mono/raw/wp3-j4.report.md`.
- **Second node.** J4 ran on n01-09; J3 on n07-17 had no host timeline. J8b (n07-17, section 4.4) reproduces the
  step cut there, against both the twin and a recompiled no-op control.
- **DLEN shrinks the step only at C1-C2,** and there it loses more AL than it saves (per accepted token 1.04-1.51 at
  C1). At C >= 4 the step stays at full width: ATOM's q-bucket verify sizes a step to at least every request's
  (previous step's bonus + 1), and with four or more requests one of them almost always took the full bonus.

### 4.3 Bytes per accepted token (model, measured routing and AL)

P1's per-rank bytes model (`p1_bytes.py`, P4's measured distinct experts), with each lever applied and divided by the
arm's measured AL (`wp3_bytes.py`). The MXFP4 GEMVs count only where the mono serves (rows <= 48).

Model bytes per rank per accepted token vs base:

| arm | C1 | C2 | C4 | C8 | C16 |
|---|---|---|---|---|---|
| QHEAD | 0.971 | 0.970 | 0.965 | 0.973 | 0.975 |
| QPROJ + QSHARED | 0.919 | 0.909 | 0.972 | 0.968 | 0.991 |
| QPROJ + QSHARED + QHEAD | **0.860** | 0.887 | 0.945 | 0.959 | 0.975 |
| u1 | **0.846** | 0.865 | 0.935 | 0.949 | 0.988 |
| DLEN 4 (C1 only: see the note) | 0.960 | | | | |

- **What it costs in time.** The gated union reads **14-15 % fewer bytes per accepted token at C1**. That is close to
  D0's 1.30-1.37 "byte cut" when expressed per token at C1, minus the AL it costs. It realizes as a 1.3 % shorter
  decode step.
- **Why so little.** The step is latency-bound (P1: 2.4x the floor at C1), so cutting a stage's bytes mostly
  shortens a wait that is not on the critical path.
- **DLEN note.** The bytes model assumes the arm's verify width holds at every CONC. The host timeline shows that
  holds only at C1 (C2 partly), so DLEN's model numbers at C >= 2 are not shown.

### 4.4 RAGGED with a measured profile; target-only MXFP4 (job J5b)

**RAGGED, uncalibrated confidence (job J5b, n07-17; one pass before the job was stopped).** The setup: the prefix
scheduler with J4's measured SPS(B) profile and T = 1. At C1:
- the scheduler shortens the verify (step 0.948 of base);
- it loses 7 % of AL (0.930);
- per accepted token: **1.019, a loss**.

At C2-C8 the batch-uniform width floor of section 4.2 leaves it at full width (step 0.970-1.006). Single-pass served
ratios: 0.887 / 0.966 / 1.150 / 0.922 at C1 / 2 / 4 / 8, all noise-dominated. The raw confidence head under-states
survival, so the scheduler truncates verifies that would have been accepted.

**Target-only MXFP4 (`ATOM_DSV41_RK_QSCOPE=target`): not servable with the mono.**
- **What happened.** The J5b server hung in draft graph capture.
- **Why.** The target and the draft share K1 / K2 builds at equal widths (30 = 5 x 6 = 6 x 5), and `rk_w4` picks
  MXFP4 per build. A draft kept in FP8 is therefore read by MXFP4 GEMV stages.
- **Status.** The knob now refuses that combination. It is kept for the native path.
- **What it would take.** A target-only scope needs a draft field in the mono build keys; this WP did not add one. So
  the AL cost of the MXFP4 GEMVs could not be split into its target and draft parts.

**RAGGED with fitted confidence temperatures (job J6, `68262435`, n01-09, completed).**

How it ran:
1. a full-width calibration run (`DLEN=5`, GSM8K C1 / C4 and non-GSM8K C1) logged 9698 raw confidence rows;
2. `wp3_sts.py` fitted per-position temperatures by matching the mean calibrated survival to the measured
   P(accepted >= k). The fit is mild: the raw mean survival was already within 0.004-0.036 of the measured one, and
   T = 1.04 / 0.71 / 0.60 / 0.51 / 0.61 for positions 1-5;
3. seven arms, three rotated passes, a fresh server per arm-run, host timeline, cells GSM8K C1 / C2 / C4 and
   non-GSM8K C1: `base`, its twin `base_b`, RAGGED with the J4 profile and those temperatures (`rbsts`) plus its own
   twin (`rbsts_b`), RAGGED with the J4 profile and T = 1 (`rbkp`), RAGGED with ATOM's stub (`rbks`), and the gated
   MXFP4 union (`w4all`).

Ratios vs the same pass's `base` (step = decode-step GPU time on the slower rank; per token = step / AL):

| arm | step C1 / C2 / C4 | AL C1 / C2 / C4 / ng C1 | per accepted token C1 / C2 / C4 | served C1 / C2 / C4 / ng C1 |
|---|---|---|---|---|
| `base_b` (twin) | 0.999 / 1.000 / 1.000 | 1.005 / 1.002 / 1.009 / 1.021 | 0.994 / 0.998 / 0.992 | 1.014 / 1.035 / 0.990 / 1.034 |
| `rbsts` (STS) | 1.109 / 1.105 / 1.097 | 0.948 / 0.959 / 0.999 / 0.929 | **1.170 / 1.152 / 1.098** | 0.768 / 0.913 / 0.887 / 0.764 |
| `rbsts_b` (STS twin) | 1.116 / 1.110 / 1.096 | 0.962 / 0.977 / 1.003 / 0.940 | 1.160 / 1.137 / 1.093 | 0.782 / 0.864 / 0.871 / 0.780 |
| `rbkp` (profile, T = 1) | 0.943 / 0.967 / 1.006 | 0.943 / 0.952 / 0.994 / 0.934 | 1.000 / 1.016 / 1.012 | 0.953 / 0.920 / 0.997 / 0.980 |
| `rbks` (ATOM stub) | 0.862 / 0.861 / 1.004 | 0.684 / 0.817 / 0.947 / 0.721 | 1.261 / 1.053 / 1.061 | 0.757 / 1.024 / 1.007 / 0.843 |
| `w4all` (MXFP4 union) | 0.983 / 0.977 / 0.985 | 0.982 / 0.982 / 0.989 / 1.008 | 1.001 / 0.994 / 0.995 | 0.967 / 1.144 / 1.039 / 1.001 |

What it shows:
- **No RAGGED form gains.** The best is `rbkp` at 1.000-1.016 per accepted token: the verify shortens at C1-C2 (step
  0.943 / 0.967) and loses as much AL. ATOM's stub over-truncates (AL 0.68 at C1).
- **The STS arm is slower at every width, C4 included.** At C4 the width stays full (AL 0.999, as in 4.2), yet the step
  is 1.097. So the extra time is not the schedule; it is the STS apply itself. Its twin reproduces it within 0.007.
  **Cause (found by code reading, confirmed in J8b):** ATOM's `dspark_scheduler.calibrate_confidence` runs
  `if torch.any(t <= 0)` on the device temperature tensor every decode step. That is a device-to-host sync on a path
  documented as sync-free, and it only runs when STS temperatures are set. This is the third ATOM defect WP3 hit. The
  WP3 workaround `ATOM_DSV41_RK_RAGGED_STS_APPLY=nosync` (default off) validates the temperatures once on the host and
  swaps in the same math without the test. It is bitwise equal on CPU for 1-512 rows, the in-server check is
  `ATOM_DSV41_RK_WP3_CHECK=1`, and `ATOM_DSV41_RK_RAGGED_STS_MUTANT=1` is its must-fail twin. It is measured in J8b
  below.
- **Upper bound for the calibrated scheduler without the sync.** The fitted temperatures move the survival estimates
  by at most 0.036, so `rbkp` (same profile, T = 1) is a close proxy for it: 1.000-1.016 per accepted token, no gain.
  That is a proxy argument; J8b measured the sync-free arm directly and agrees.
- **MXFP4 union, same node, new job.** Step 0.983 / 0.977 / 0.985 at C1 / C2 / C4 reproduces J4's 0.987 / 0.981 /
  0.983. Per accepted token 0.994-1.001 against the twin's 0.992-0.998: still no resolved gain. This is a repeat on
  n01-09, **not** the second-node check (J6 was placed on n01-09 again); that check is J8b.

Raw: `private-perf/dsv41-mono/raw/wp3-j6/`, `wp3-j6.report.md`, `wp3-j6.htl.json`.

**Job J8b (`68265737`, n07-17, completed): the second node, a recompiled control and the sync-free STS apply.**
Seven arms, three rotated passes, a fresh server per arm-run, host timeline, all five GSM8K CONCs plus non-GSM8K C1:
`base`, `base_b`, the MXFP4 union `w4all`, a recompiled no-op control `nop` (`ATOM_DSV41_RK_LEV=nop=1`: the same K1 /
K2 code under new kernel names; alola rule 260 measured a ~1 % in-situ shift from recompiling alone), RAGGED with the
J4 profile (`rbkp`), with J6's STS temperatures through ATOM's apply (`rbsts`) and through the sync-free apply
(`rbstsn`). (The first submission, `68263911`, died in a NODE_FAIL on n07-17 after pass 1; J8b is the full rerun
after the node was re-seeded.)

| arm | step C1 / C2 / C4 / C8 / C16 | AL C1 / C2 / C4 / C8 / C16 | per accepted token C1 / C2 / C4 / C8 / C16 |
|---|---|---|---|
| `base_b` (twin) | 1.000 / 1.005 / 1.003 / 0.996 / 1.001 | 1.017 / 0.994 / 0.995 / 1.006 / 1.001 | 0.984 / 1.011 / 1.008 / 0.990 / 1.000 |
| `nop` (recompiled control) | 1.002 / 0.995 / 0.997 / 1.001 / 0.997 | 1.002 / 1.003 / 0.991 / 1.004 / 0.998 | 1.000 / 0.992 / 1.007 / 0.997 / 0.999 |
| `w4all` (MXFP4 union) | **0.994 / 0.981 / 0.981 / 0.991** / 0.998 | 0.982 / 0.978 / 0.986 / 0.979 / 0.997 | 1.011 / 1.004 / 0.995 / 1.012 / 1.001 |
| `rbkp` (profile, T = 1) | 0.944 / 0.975 / 1.005 / 1.006 / 1.004 | 0.939 / 0.956 / 0.977 / 0.983 / 0.996 | 1.006 / 1.020 / 1.029 / 1.023 / 1.008 |
| `rbsts` (STS, ATOM's apply) | 1.119 / 1.112 / 1.094 / 1.065 / 1.051 | 0.960 / 0.959 / 0.983 / 1.000 / 1.008 | 1.166 / 1.160 / 1.113 / 1.065 / 1.043 |
| `rbstsn` (STS, sync-free apply) | 0.952 / 0.989 / 1.009 / 1.008 / 1.007 | 0.946 / 0.963 / 0.986 / 0.987 / 0.987 | 1.006 / 1.027 / 1.023 / 1.021 / 1.020 |

What it shows:
- **The W4 step cut holds on the second node and against the recompiled control.** `w4all` vs `nop`: 0.992 /
  0.986 / 0.984 / 0.990 / 1.001 at C1-C16. So the cut is 1.4-1.6 % at C2-C4 and 1.0 % at C8 on both nodes; at C1 it is
  0.6-0.8 % on n07-17 against 1.3 % on n01-09 (J4; J6 1.7 %).
- **The AL cost is now resolved at C1-C4, pooled.** In each single job the union's AL overlaps its twin's spread. Over
  the three jobs that ran the union (J4, J6, J8b; two nodes), 8 of its 9 C1-C4 cells (0.978-0.989; the ninth is J4 C2
  at 1.009) lie below every twin cell at C1-C4 (0.994-1.017), and the pooled means are 0.986 vs 1.003. So the union
  costs about 1.5 % of AL at C1-C4. **Which** part causes it (the target's or the draft's requantized linears) is
  still a hypothesis: the target-only scope cannot run with the mono.
- **Net per accepted token, no gain:** `w4all` 0.995-1.012 against the twin's 0.984-1.011 and `nop`'s 0.992-1.007.
- **The STS overhead was the host sync.** `rbstsn` differs from `rbsts` only in dropping the per-step
  `torch.any(t <= 0)`. Its step falls from 1.119 to 0.952 at C1 and from 1.094 to 1.009 at C4. The in-server check
  passed in all three `rbstsn` servers (bitwise vs ATOM's `calibrate_confidence` at 1 / 2 / 4 / 8 / 16 rows), and the
  `ATOM_DSV41_RK_RAGGED_STS_MUTANT=1` server printed `FAIL (5 of 5 row counts differ)`.
- **Calibrated RAGGED, measured: no gain.** Without the sync, the STS-calibrated scheduler reads 1.006-1.027 per
  accepted token, the same as the T = 1 profile (`rbkp`, 1.006-1.029). It shortens the C1 verify (step 0.952) and
  loses as much AL (0.946). This confirms the J6 proxy argument with data.

Raw: `private-perf/dsv41-mono/raw/wp3-j8b/`, `wp3-j8b.report.md`, `wp3-j8b.htl.json`.

## 5. TP1

TP1 runs P3's configuration: native + DSpark, cold experts in pinned host memory, embedding on the host. The WP3 levers
were added (job tp1a, n07-17, two rotated passes, A-vs-A twin):

| arm | C1 | C4 | C16 | non-GSM8K C1 |
|---|---|---|---|---|
| twin | 1.009 [0.998, 1.019] | 0.978 | 0.987 | 1.013 |
| `QHEAD=fp8` + `QHEAD_FREE` (BF16 head released) | 1.033 [1.005, 1.060] | 0.957 | 0.988 | 0.996 |
| `QHEAD` + `DVOCAB 64k` | 1.016 | 0.980 | 0.981 | 0.981 |

- **The WP3 knobs serve at TP1.** Two hard parts: the unsharded draft head path, and freeing the BF16 head once both
  readers are patched.
- **No measured gain beyond the twin band.**
- **Finding for WP7.** P3's cold-expert plan was built from GSM8K routing. On the non-GSM8K prompts TP1 decodes at
  0.44x of its own GSM8K rate (TP2: 0.80-0.82x). Cold experts are hit far more off-distribution, so the plan should be
  built from a mixed workload.

## 6. Lookback / rejects (re-test on the integrated base)

Each of these stays a default-off knob, recorded with the base it lost on:

- **DVOCAB** (16k-64k): rejected on base B0.
- **SHARD** (`engram:main_proj`; `wqkv_a` native): rejected on base B0.
- **DLEN 1-4**: rejected on base B0 (per accepted token 1.04-1.51 at C1; no step change at C >= 4).
- **RAGGED=bucket**: rejected on base B0. Profile with T = 1: per accepted token 1.019 at C1 (J5b), 1.000-1.029 at
  C1-C16 (J6, J8b). STS-calibrated: 1.10-1.17 through ATOM's apply (a per-step host sync), 1.006-1.027 through the
  sync-free `ATOM_DSV41_RK_RAGGED_STS_APPLY=nosync` (J8b). Kept default-off with its sub-knobs.
- **Native QPROJ / QSHARED**: rejected on base B0 for both speed and accuracy.
- **QPROJ, QPROJ + QSHARED, QPROJ + QSHARED + QHEAD (mono)**: pass the literal accuracy gate. The full union cuts the
  decode step by 1.3-2.0 % at C1-C4 on n01-09 and 0.8-1.6 % on n07-17 (against a recompiled control), and costs about
  1.5 % of AL (pooled over three jobs), so there is no resolved change per accepted token. Kept default-off, "no gain on base B0". They are the first candidates to re-test once
  WP2 / WP4 / WP5 make the step byte-bound.
- **QHEAD alone, QSHARED alone (mono)**: fail the literal accuracy gate by 1-4 questions. Rejected on base B0
  (accuracy). Re-test together with more reference runs in one job (section 3).
- **QPROJ=mxfp4a + QSHARED + QHEAD and the union `u1`**: fail the literal gate in one of three runs each. Rejected on
  base B0 (accuracy, and no step gain over `mxfp4`).

All of them are worth re-testing once WP2 / WP4 / WP5 bring the step closer to the bytes floor. The MXFP4 GEMV stages
hook into the same `gemv_fp8_loads` / `gemv_fp8_mfmas` helpers WP4 ports, so a rocKE GEMV port can take an MXFP4
operand the same way.

## 7. Gates and mutants (index)

| gate | where | mutants (must fail) | result |
|---|---|---|---|
| DVOCAB kernel: whole vocab and a TP2 shard bitwise vs ATOM's `dspark_markov_argmax` / `_launch(packed)`; subset vs fp64 argmax | `wp3_unit.py` (J2: 79 PASS, 34 of 34 mutants caught) | id map shifted, W2 rows not subset, columns shifted | PASS (J1's first run missed the W2 mutant at B 1 / 5: the test bias was too small to matter; fixed data) |
| SHARD bitwise per row count, all ranks agreeing | in-server at load + `wp3_unit.py` | `SHARD_MUTANT=all_rows` | PASS on the probed counts |
| QHEAD numerics | `wp3_unit.py` + in-server | scale code +1 every 7th block; transposed 32x32 blocks | PASS / caught (0.38, 1.40) |
| MXFP4 GEMV numerics (Triton reference) | `wp3_unit.py` + in-server | scale; transposed 16x16 bytes | PASS 0.123-0.135 / caught |
| mono MXFP4 K1 vs FP8 K1 on FP4-representable weights | `wp3_k1rig.py` (J2b, J3, J4), `wp3_k1rig_aa.py` (J7) | scale, nibble swap, transpose | not bitwise: deterministic 1-ulp `q_out` differences (1-3 elements, <= 1.5e-5) and one scratch byte with `wqkv_a`; tolerance set after J2b's FAIL (section 2.3); mutants 33,796+ ulp on 0.30-0.999 of `q_out`, caught far outside the tolerance |
| run-to-run determinism of the K1 rig | `wp3_k1rig_aa.py` (J7) | - | FP8 and MXFP4 builds each bitwise over 8 repeats, 6 configurations x 3 seeds |
| knob-off identity of the mono kernels | local trace `cmp` vs pristine (K1, K2 at 6 rows, Nov 3); J7 in-container IR dumps (K1, K2 at 6 / 12 rows; pristine vs J1-J6 stage vs current shared tree); J8 (final tree, K1 / K2 at 6 rows) | knobs on (`QPROJ` + `QSHARED`) must differ: different kernels and ISA (J8, J8b) | identical: every IR stage byte-identical to pristine (J7: 4 kernels x 23 files x 2 trees; J8 and J8b: 46 of 46) |
| sync-free STS apply equals ATOM's `calibrate_confidence` | in-server, `ATOM_DSV41_RK_WP3_CHECK=1` (J8b, 3 servers) + CPU check (1-512 rows) | `ATOM_DSV41_RK_RAGGED_STS_MUTANT=1` (calibration applied twice) | PASS bitwise at 1 / 2 / 4 / 8 / 16 rows; mutant FAIL 5 of 5 |
| accuracy | J2 / J2b | section 3 | section 3 |

## 8. Reproduce

All jobs ran through `hackathon-prep/dsv41/wp3/` on the seeded rocke nodes:

1. **Stage.** `stage.sh <sfx>` stages the ATOM-dsv41 tree and the harness. `wp3_job.sh` mounts them; `arms.sh` runs
   them.
2. **Define the arms** in `defs/<job>.def`: one line per arm, `name|base arm|env knobs|extra server args`.
3. **Optional steps:**
   - `UNIT`: unit gates;
   - `K1RIG`: the K1 rig;
   - `VOCAB`: DVOCAB calibration;
   - `QPROBE`: the head GEMM probe;
   - `HTL`: the host timeline;
   - `TRACE_ARMS`: a kineto window;
   - `ACC_ARMS`: the accuracy gate, GSM8K 1319 plus passkey.
4. **Analysis:**
   - `wp3_summ.py`: served ratios and reply identity;
   - `wp3_htl.py`: decode-step GPU time;
   - `wp3_report.py`: everything per arm;
   - `wp3_bytes.py`: the bytes model;
   - `wp3_profile.py`: the RAGGED SPS profile.
5. **Verifier-round jobs** (new files only; `stage2.sh` = `stage.sh` that also skips the `lt7/` and `trees/` dirs):
   - `j7.sh` (1 GPU): `wp3_k1rig_aa.py` (K1 rig A-vs-A numerics and mutant margins) and the cross-tree knob-off IR
     identity (`lt_cluster.py`; trees as tars under the harness's `trees/`);
   - `j8.sh` with `defs/j8.def` (2 GPUs, n07-17): the knob-on / knob-off trace of the final tree, then seven arms with
     the host timeline, then the `RAGGED_STS_MUTANT` server;
   - `release_pair.sh`: releases held jobs in order under the GPU cap.
