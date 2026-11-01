# Large-batch baseline: FlyTape's tiled tape and ATOM's native path (gfx950, MXFP4, TP1 / TP2, CONC 8-128)

What the rocKE large-batch monokernel has to beat, cell by cell (user decision 4 in `../PLAN.md`): FlyTape's tiled
tape at the same batch / TP / precision, and ATOM's native path, which serves every step wider than the mono's 4 rows
today. Ratios only here; absolute step times, TPOT and tok/s live in
`hackathon-prep/private-perf/qwen38-true-mono/large-batch/baseline.md`.

Measured 2026-10-30 on the alola `rocke` partition (MI355X, 256 CUs), q38 ROCm 10.0 container. Jobs 68217648 (FlyTape
TP2), 68217675 (FlyTape TP1), 68217686 (ATOM TP1), 68217694 (ATOM TP2); node and GPU bus ids per job are in the private
file. Every job had its GPUs to itself (`amd-smi process`: no other process on them); other jobs ran on the other GPUs.

## What was run

- **FlyTape** (`dev/flytape-main/flytape-main`, stage `flytape`, FlyDSL 0.3.4.1): `scripts/bench_qwen4_full.py`, the
  full shape (48 layers, 512 experts, the 51 GB n-gram table), random weights, zero state at position prompt - 1,
  64 new tokens, median device step (`s_memrealtime` gaps, first step excluded). All configs of one precision in one
  process; 3 reps, each rep rotates the config order (so configs interleave). B > 4 runs the tiled tape.
  - TP2 `mxfp4`: B 8 / 16 / 32 / 64 / 128 at prompt 128 and 8192; TP2 `fp8`: B 32 / 128 at prompt 128.
  - TP1 `mxfp4`: B 8 / 16 / 32 / 64 at prompt 128 and 8192. **TP1 B128 is unsupported (confirmed)**: `make_tape_spec`
    refuses it before compiling (`gdn_state` needs 3.62e9 elements at batch 128, past the 2^31 i32 index limit).
  - Phase traces (a second, traced launch): TP2 B32 and B128, TP1 B32, prompt 128.
- **ATOM native** (stage `atom-rkrepro` = the hybrid ATOM tree; native path = upstream `352229d6`), the same MXFP4
  checkpoint as `../baseline/MATRIX.md`, `ATOM_MONO_ENABLE=0` (the mono serves at most 4 rows / 4 MTP groups; every
  wider step already runs this path), `ATOM_RUNNER_INLINE_SEND=1`, `--max-num-seqs 128 --max-model-len 10240`,
  `-tp 1` or `-tp 2`. Arms: K0 (no MTP), K1, K3 (native MTP, ATOM's default ReplaySSM, which is on with speculation),
  and at TP1 K3r0 (K3 with `ATOM_ENABLE_REPLAYSSM=0`, as the hybrid's serving line sets it). Two reps; per rep a
  seeded arm order (rotated) and a seeded cell order shared by every arm of the rep; one server per arm-run.
  - Cells: `benchmark_serving` random 1k/1k and 8k/1k at CONC 8, 16, 32, 64, 128, ignore-eos, 2 x CONC prompts (16 at
    CONC 8), same prompts for every arm of a rep, an untimed warmup at the cell's CONC first.
  - Accuracy per server: GSM8K `single_request_bench` (64 prompts at CONC 64, 2048 tokens, reasoning xhigh); server
    log checks (0 tracebacks, mono off, ReplaySSM state as the arm says). All 14 servers passed.
- **How ATOM serves TP2** in the q38 container: plain `-tp 2` on a 2-GPU job works for this checkpoint (ATOM's FP8
  recipe calls pure TP2 "not validated"; the MXFP4 MoE warmup did not fail). One model-runner process per GPU, torch
  ProcessGroupNCCL (RCCL) plus aiter's `tensor_model_parallel_all_reduce` at the attention output and MoE combine,
  CUDA graphs at sizes 1-8, 16, 32, 48, 64, 128, 256, 512. Per GPU the weights take about 0.59x of TP1's peak torch
  memory, leaving 3.75x TP1's KV block count.
- Harness: `hackathon-prep/qwen38/truemono-lb/` (`ft_lb.sh`, `atom_lb.sh`, `summarize_ft.py`, `summarize_atom.py`,
  `compare.py`, `sync.sh`, `submit.sh`).

## Metrics

Time per token of one request, decode only, in the same unit for both:
- **FT step**: FlyTape's median device step (closed batch: every row decodes every step; no MTP).
- **ATOM dec**: median inter-chunk latency / acceptance length. The median ITL is one decode step (K0) or one verify
  round (MTP) with the prefill-interrupted chunks dropped as outliers; acceptance from `/debug/mtp_stats` over the cell.
- **ATOM TPOT**: `benchmark_serving` mean TPOT, the serving number: it adds the prefill of other requests interleaved
  into the request's decode (open serving; MTP staggers finishes, so more admissions land mid-decode).
Throughput at batch B is B / time. The **reference** of a cell is the faster of the FT step and the best ATOM arm's dec.

## Reference to beat per cell (each column / the reference time; > 1 = slower than the reference)

TP2 rows: superseded by the DP2 section below (user decision 5 adds a DP2 arm to every TP2 cell; ATOM native DP2 K3
is the faster per-token decode at 1k B8 and 8k B8-B16, and within 3.2 % of FlyTape TP2 at 1k B16-B64 and 8k B32-B64).
The reference column compares FlyTape (K0 device step, random weights, W4A4) with ATOM (served, real weights, W4A8,
MTP): a cross-precision, cross-metric bar, not a matched-precision win for either stack.

| TP | ISL | B | FT step | ATOM K0 dec | ATOM best dec (arm) | ATOM best TPOT | FT fp8 step | reference |
|---|---|---|---|---|---|---|---|---|
| 1 | 1k | 8 | 1.097 | 1.716 | 1.000 (K3) | 1.064 | - | ATOM K3 |
| 1 | 1k | 16 | 1.094 | 1.602 | 1.000 (K3) | 1.124 | - | ATOM K3 |
| 1 | 1k | 32 | 1.047 | 1.568 | 1.000 (K3) | 1.191 | - | ATOM K3 |
| 1 | 1k | 64 | 1.013 | 1.429 | 1.000 (K3) | 1.268 | - | ATOM K3 (FlyTape tied) |
| 1 | 1k | 128 | unsupported | 1.327 | 1.000 (K3) | 1.365 | - | ATOM K3 |
| 1 | 8k | 8 | 1.174 | 1.784 | 1.000 (K3) | 1.395 | - | ATOM K3 |
| 1 | 8k | 16 | 1.084 | 1.532 | 1.000 (K3) | 1.696 | - | ATOM K3 |
| 1 | 8k | 32 | 1.052 | 1.526 | 1.000 (K3) | 2.124 | - | ATOM K3 |
| 1 | 8k | 64 | 1.021 | 1.373 | 1.000 (K3) | 2.316 | - | ATOM K3 (FlyTape tied) |
| 1 | 8k | 128 | unsupported | 1.307 | 1.000 (K1; K3 1.007) | 2.622 | - | ATOM K1 / K3 |
| 2 | 1k | 8 | 1.000 | 1.852 | 1.191 (K1) | 1.275 | - | FlyTape |
| 2 | 1k | 16 | 1.000 | 1.722 | 1.137 (K3) | 1.283 | - | FlyTape |
| 2 | 1k | 32 | 1.000 | 2.318 | 1.176 (K3) | 1.582 | 1.849 | FlyTape |
| 2 | 1k | 64 | 1.000 | 1.868 | 1.096 (K3) | 1.479 | - | FlyTape |
| 2 | 1k | 128 | 1.000 | 1.793 | 1.235 (K3) | 1.810 | 1.785 | FlyTape |
| 2 | 8k | 8 | 1.000 | 1.755 | 1.179 (K1) | 1.510 | - | FlyTape |
| 2 | 8k | 16 | 1.000 | 1.655 | 1.099 (K3) | 1.695 | - | FlyTape |
| 2 | 8k | 32 | 1.000 | 2.251 | 1.120 (K3) | 2.266 | - | FlyTape |
| 2 | 8k | 64 | 1.000 | 1.797 | 1.099 (K3) | 2.736 | - | FlyTape |
| 2 | 8k | 128 | 1.000 | 1.723 | 1.201 (K3) | 3.285 | - | FlyTape |

"FlyTape tied": within the noise of the two measurements (1-2 %). Reps agree to < 1.5 % (FlyTape) and, for ATOM dec,
median 0.9 %, 7 of 70 arm-cells 4-7.7 % (MTP arms at CONC 8-32, TP1); one TP1 K3r0 cell (8k CONC 128) is left out: its KV pool holds only ~64 rows at 9,216 tokens, so it
decoded a smaller batch.

Reading it:
- **TP2: FlyTape is the reference in every cell**, 1.10-1.24x faster per token than ATOM's best MTP arm and
  1.66-2.32x faster than ATOM without MTP, although it has no MTP. At B128 (the user's anchor) ATOM's best dec is
  1.235x (1k) / 1.201x (8k) the FlyTape step.
- **TP1: ATOM K3 is the reference in every cell** (B128 has no FlyTape at TP1), but only by 1-17 %: FlyTape (no MTP)
  is within 1.3-2.1 % at B64 and 5-17 % behind at B8-32. Without MTP, FlyTape is 1.34-1.56x faster than ATOM K0
  (FT / K0 dec 0.64-0.74).
- **The serving view is worse for ATOM**: its TPOT is 1.06-3.3x the reference; at 8k ISL the interleaved prefill
  doubles to triples it at CONC 64-128 (MTP more than K0). FlyTape measures no prefill at all.
- **FlyTape `fp8`** (W8A16 experts) is 1.79-1.85x its `mxfp4` step at TP2, slower than ATOM's best arm.

## FlyTape: how the step scales (ratios)

- Batch (step / B8 step): TP2 B16 1.20, B32 1.41-1.42, B64 1.99-2.00, B128 2.58; TP1 B16 1.21-1.22, B32 1.59-1.60,
  B64 2.22-2.23. Throughput (B / step) grows 6.2x from B8 to B128 at TP2.
- Context: prompt 8192 / prompt 128 = 1.031-1.043 at every batch and TP (QSA past its 2,048-token budget).
- TP: TP2 / TP1 step = 0.74-0.76 at B8-16, 0.67-0.69 at B32-64 (two GPUs buy 1.3-1.5x, not 2x).
- Against the 2026-10-29 run: TP2 B16 and B128 within 1.1 %.
- Phase families (traced, share of the per-step phase sum; "reduce/combine" = the split-K reduce and combine phases,
  which at TP2 hold the cross-rank exchange waits):

| config | MoE | GDN | reduce / combine | HC mixers | QSA | head | other |
|---|---|---|---|---|---|---|---|
| TP2 B128 | 0.409 | 0.276 | 0.151 | 0.109 | 0.041 | 0.009 | 0.005 |
| TP2 B32 | 0.425 | 0.205 | 0.159 | 0.136 | 0.055 | 0.015 | 0.005 |
| TP1 B32 | 0.496 | 0.268 | 0.064 | 0.086 | 0.064 | 0.019 | 0.003 |

  Top phases at TP2 B128: `gdn.moe_expert` 0.258, `gdn.gdn_recur` 0.173, `qsa.moe_expert` 0.087, `gdn.gdn_proj` 0.073,
  `gdn.mlp_mix_combine` 0.046, `gdn.moe_router` 0.042. At TP1 B32: `gdn.moe_up` 0.212, `gdn.gdn_recur` 0.118,
  `gdn.gdn_proj` 0.108, `gdn.moe_down` 0.107. Compared with B4 (`../research/flytape.md`: mixers 0.34, MoE 0.31), the
  large batch is MoE + GDN recurrence bound; the mixers shrink to 0.09-0.14.

## ATOM native: what the arms show (ratios)

- **MTP pays at every batch**: K0 dec / K3 dec = 1.30-1.78 at TP1, 1.41-2.01 at TP2 (shrinking with batch at TP1);
  K0 / K1 = 1.30-1.53 (TP1). K3 is best almost everywhere; K1 ties it at TP1 8k B128 and wins at TP2 B8 (by 1 % at 1k, 6 % at 8k).
  A K3 verify round costs 1.6-2.1x a K0 step at TP1 and 1.4-1.9x at TP2, against 2.7-2.95 tokens per round.
- **ReplaySSM** (ATOM's default with MTP) beats the hybrid's `ATOM_ENABLE_REPLAYSSM=0`: K3r0 / K3 dec = 1.00-1.13,
  worst at B64-128; with it off, TP1 holds 0.30x the KV blocks (8k CONC 128 does not fit).
- **TP2 / TP1** (K0 dec): 0.88-0.91 at B8-16 and B64, 0.78 at B128, but 1.04-1.05 at B32: ATOM's TP2 32-row step is
  an outlier (1.77-1.81x its B8 step against 1.52-1.53x at TP1); K1 at B16 and K3 at B8 (also 32 verify rows) show it
  too.
- **Batch scaling** (K0 dec / B8): TP1 B128 2.87-2.88, TP2 B128 2.50-2.53 (FlyTape TP2: 2.58).
- **Acceptance** on the random prompts' generations: K1 1.75-1.80, K3 2.72-2.95 (cell means); on GSM8K (the accuracy check)
  K1 1.83-1.85, K3 3.00-3.09.
- **GSM8K right of 64 per server**: TP1 K0 60 59, K1 62 56, K3 60 58, K3r0 60 59; TP2 K0 57 60, K1 58 58, K3 61 57;
  inside the 56-62 spread of the mono / hybrid runs in `../baseline/MATRIX.md`. A 64-prompt subset resolves only
  gross breakage; equal-accuracy claims for the rocKE kernel need the matrix's full suites.

## Caveats

- **Not the same numerics or bytes.** FlyTape's `mxfp4` preset quantizes the routed and shared experts only, and on
  the tiled tape runs them W4A4 (expert input and SiLU product quantized to MXFP4 per 32, RCEIL scale); dense
  projections, HC mixers, router, n-gram projections and the head stay bf16. ATOM's checkpoint runs dense linears W4A8
  (MXFP4 x MXFP8) and the MoE on MXFP8 activations; `lm_head` and the HC mix weights are bf16 (MXFP4-rounded).
  FlyTape reads more dense bytes and computes a cheaper, lower-precision MoE; ATOM's README measured FP4 activations
  3.3 GSM8K points below FP8 on this model. "Equal accuracy" against FlyTape therefore means ATOM's numerics (W4A8),
  unless a W4A4 MoE passes the accuracy suites (user decision 1).
- **Random weights, zero state, no serving.** FlyTape decodes random weights from zero state (near-uniform routing),
  a closed batch, no prefill, no scheduler or host work, greedy only, no MTP. ATOM serves real prompts with real
  routing, admits requests mid-decode, and pays host and scheduler time every step.
- **Context points differ for 1k.** FlyTape has no 1k point; its prompt-128 step stands in for ATOM's 1k-2k context
  (FlyTape's own context curve moves < 1 % from 64 to 2,048 tokens; prompt 8192 costs 3-4 %).
- **ATOM dec is a median.** It drops the prefill-interrupted steps and divides MTP rounds by the cell's mean
  acceptance. Random-prompt acceptance is 0.05-0.4 below GSM8K's (K3), 0.03-0.1 (K1).
- **Sharing.** Each job owned its GPUs, but FlyTape TP1 / TP2 and ATOM TP1 shared node `n01-09` (and W2 jobs ran on
  both nodes). FlyTape rep spreads stayed <= 1.5 %.
- **Arms across jobs.** FlyTape and ATOM ran in different jobs on different GPUs (cross-GPU spread about 1 % per
  FlyTape's notes), not interleaved with each other; ATOM's arms and FlyTape's configs are interleaved within their
  jobs. The comparison's margins (5-24 % except the two tied cells) are well above that.

## What the rocKE large-batch kernel must hit

Per cell, a per-token decode time below the reference above (the reference / the rocKE time > 1), at ATOM's numerics,
inside ATOM's serving (so its TPOT, with prefill interleaved, is what users see):
- TP2: below FlyTape's step at B8-128, both ISLs; with MTP that means a K3 verify round at B x 4 rows under about 2.8x
  FlyTape's B-row step (or K1 under about 1.8x).
- TP1: below ATOM K3 at B8-128 (and FlyTape at B8-64, which is within 1-17 % of it); B128 has no FlyTape bar.
- Without MTP (K0), the bar is FlyTape's step at every cell where it runs: 0.43-0.74x ATOM's K0 step.

The TP2 bullet is superseded by the next section: at TP2 cells the bar is the better of TP2 and DP2.

## DP2 arms of every TP2 cell (user decision 5; L0c-DP2, 2026-10-30, fix round 2026-10-31)

Every TP2 cell re-run against two independent TP1 replicas on the same two GPUs, interleaved in the same job.
- FlyTape: jobs 68220975 and 68225698 (fix round, another GPU pair). Arms TP2, DP2 = two concurrent `--tp 1` processes
  at B/2 with a per-sample start barrier, and their recompiled A-vs-A twins TP2s / DP2s, 4 shuffled rounds.
- ATOM native, fix round: jobs 68225658, 68225982, 68228847. The DESIGN.md 9.1 e2e WIN rule with 6 pairs per cell (7
  at B8). Arms TP2 K3, TP2s K3, DP2 K3 (= two `-tp 1` servers at CONC B/2 each), DP2s K3, and TP2 K1 at B8 and B128.
  A GSM8K score gate (>= 52 / 64 per server) ran with must-fail mutants caught on hardware.
- ATOM native, first round: jobs 68220965 and 68221379, K0 / K1 / K3, 2 reps. Superseded for the verdicts.

Details, gates and the per-arm tables: `L0c-DP2.md`; absolutes:
`hackathon-prep/private-perf/qwen38-true-mono/large-batch/L0c-DP2.md`. Same metrics as above (per-token decode time
of one request; FlyTape device step, ATOM dec). Each column is over the fastest of the four (FlyTape from job
68220975):

| ISL | B | FT TP2 | FT DP2 | ATOM TP2 best (arm) | ATOM DP2 best (arm) | FT: stronger mode (DP2/TP2, two jobs) | ATOM: stronger mode (DP2/TP2 median over pairs) | fastest (cross-stack) |
|---|---|---|---|---|---|---|---|---|
| 1k | 8 | 1.091 | 1.140 | 1.302 (K1) | 1.000 (K3) | TP2 (1.047, 1.053) | DP2 (0.761) | ATOM DP2 K3 |
| 1k | 16 | 1.027 | 1.142 | 1.191 (K3) | 1.000 (K3) | TP2 (1.114, 1.105) | DP2 (0.839) | within 3.1 % |
| 1k | 32 | 1.000 | 1.150 | 1.169 (K3) | 1.032 (K3) | TP2 (1.155, 1.140) | DP2 (0.891) | within 3.2 % |
| 1k | 64 | 1.007 | 1.059 | 1.107 (K3) | 1.000 (K3) | TP2 (1.052, 1.058) | DP2 (0.901) | within 1.5 % |
| 1k | 128 | 1.000 | 1.161 | 1.290 (K3) | 1.158 (K3) | TP2 (1.162, 1.156) | DP2 (0.897) | FlyTape TP2 |
| 8k | 8 | 1.123 | 1.182 | 1.328 (K1) | 1.000 (K3) | TP2 (1.055, 1.056) | DP2 (0.751) | ATOM DP2 K3 |
| 8k | 16 | 1.039 | 1.145 | 1.171 (K3) | 1.000 (K3) | TP2 (1.106, 1.096) | DP2 (0.855) | ATOM DP2 K3 |
| 8k | 32 | 1.000 | 1.137 | 1.163 (K3) | 1.007 (K3) | TP2 (1.138, 1.133) | DP2 (0.871) | within 0.7 % |
| 8k | 64 | 1.003 | 1.077 | 1.111 (K3) | 1.000 (K3) | TP2 (1.074, 1.078) | DP2 (0.896) | within 0.8 % |
| 8k | 128 | 1.000 | 1.106 | 1.239 (K3) | 1.107 (K3) | TP2 (1.106, 1.124) | DP2 (0.896) | FlyTape TP2 |

"Within x": FlyTape TP2 and ATOM DP2 K3 differ by at most x with either FlyTape job. A cell names one of them only
when the gap is above 3 % with both FlyTape jobs. Reading it:
- **FlyTape: TP2 is the stronger reference in every cell** (DP2 / TP2 1.047-1.162 in job 68220975 and 1.053-1.156
  in 68225698, 4 / 4 rounds, beyond both A-vs-A controls in each job).
  - Between jobs a cell moves by up to 0.028, so read the values as +-0.015.
  - At TP2's per-request time DP2 gives 0.61-0.87x its throughput. TP2's B8 latency is below DP2's measured range
    (B4 per replica and up).
  - FlyTape's TP2 steps reproduce the W3 run within 1.4 % and the second job within 1.5 %.
  - Tokens: DP2 replicas bitwise equal, and every config's tokens equal across all runs. The must-fail mutant was
    caught in both jobs.
- **ATOM native: DP2 is the stronger reference in all 10 cells** under the e2e WIN rule.
  - Best DP2 / best TP2 is 0.751-0.901, every valid pair below 1, every median below its A-vs-A band minimum
    (0.889-0.981). DP2 serves 1.23-1.41x the output tok/s.
  - The first round's 1k B16 tie is now 0.839 against a band of 0.893-1.035.
  - At equal K (first round, 2 reps), DP2 wins by 8-30 % except at B8-B16 without MTP and K1 at B8 (ties,
    0.965-1.016). The 0.70-0.71 cells are ATOM TP2's slow 32-row steps.
  - GSM8K 56-61 / 64 on every server, in both rounds.
- **Fastest per-token decode, across stacks:** ATOM DP2 K3 at 1k B8 and 8k B8-B16 (FlyTape TP2 1.039-1.123x it).
  FlyTape TP2 at B128 (ATOM DP2 K3 1.107-1.158x it). The two are within 3.2 % at 1k B16-B64 and 8k B32-B64.
  - This compares different precision (W4A4 random weights vs W4A8 real weights) and different metrics (K0 device
    step vs served MTP per-token time).
  - The TP2-cell bar for the rocKE kernel is the faster of FlyTape TP2 and ATOM DP2 K3. FlyTape's DP2 is never the
    bar.
- Bytes: one DP2 GPU (TP1 at B/2) reads 1.14-1.40x one TP2 rank's bytes (L0c-DP2 bytes model), so a TP2 LB kernel has
  the lower floor and must keep its exchange cost under that margin to beat its own DP2.

## How to rerun

From `hackathon-prep` (Git Bash), `CL_HOST=vanantha@ctr2-alola-login-01`, `Q38_MAX_GPUS=10`:

```bash
bash qwen38/truemono-lb/sync.sh                     # stage w3-lbh (common/ + truemono-lb/); flytape, atom-rkrepro exist
S=qwen38/truemono-lb/submit.sh                      # submits, then kills the sacct poller (alola-cluster skill, large-batch lane facts)
$S logs/q38/w3/ft-tp2.poll --mode container --gpus 2 --time 02:30:00 --tag w3-ft-tp2 --stage w3-lbh --stage flytape \
  --env TP=2 --env TMO=4000 --env "RUNS=tp2-mx:mxfp4:8x128,16x128,32x128,64x128,128x128,8x8192,16x8192,32x8192,64x8192,128x8192:3 tp2-fp8:fp8:32x128,128x128:3 tp2-trace:mxfp4:32x128,128x128:1:trace" \
  -- 'bash /job/stages/w3-lbh/truemono-lb/ft_lb.sh'
$S logs/q38/w3/ft-tp1.poll --mode container --gpus 1 --time 02:30:00 --tag w3-ft-tp1 --stage w3-lbh --stage flytape \
  --env TP=1 --env TMO=3000 --env "RUNS=tp1-mx:mxfp4:8x128,16x128,32x128,64x128,8x8192,16x8192,32x8192,64x8192:3 tp1-trace:mxfp4:32x128:1:trace tp1-b128:mxfp4:128x128:1" \
  -- 'bash /job/stages/w3-lbh/truemono-lb/ft_lb.sh'
C="bench:1024:1024:8:16 bench:8192:1024:8:16 bench:1024:1024:16:32 bench:8192:1024:16:32 bench:1024:1024:32:64 bench:8192:1024:32:64 bench:1024:1024:64:128 bench:8192:1024:64:128 bench:1024:1024:128:256 bench:8192:1024:128:256"
$S logs/q38/w3/atom-tp1.poll --mode container --gpus 1 --time 08:00:00 --tag w3-atom-tp1 --stage w3-lbh --stage atom-rkrepro \
  --env TP=1 --env REPS=2 --env "ARMS=K0 K1 K3 K3r0" --env "CELLS=$C" -- 'bash /job/stages/w3-lbh/truemono-lb/atom_lb.sh'
$S logs/q38/w3/atom-tp2.poll --mode container --gpus 2 --time 07:00:00 --tag w3-atom-tp2 --stage w3-lbh --stage atom-rkrepro \
  --env TP=2 --env REPS=2 --env "ARMS=K0 K1 K3" --env "CELLS=$C" -- 'bash /job/stages/w3-lbh/truemono-lb/atom_lb.sh'
```

Results land in `/home_aus/AMD/vanantha/rocke-sdpa-claude/q38/results/<tag>/` (`ft/summary.{txt,json}`,
`atom/abs.json`, `atom/summary.txt`); `compare.py --ft <ft summary.json>... --atom <atom abs.json>... --abs A --ratio R`
writes both tables. A rocKE arm is one more entry in `atom_lb.sh`'s arms (an ATOM stage with the rocKE large-batch
path) against the same cells, or a `bench`-style device-step script against FlyTape's configs.
