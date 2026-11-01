# L0c-DP2: large-batch bytes model, bandwidth curves, per-stage floors, and the DP2 arms (gfx950, MXFP4)

WP-L0c of `../DESIGN-LB.md` plus the DP2 arms of user decision 5 (`../PLAN.md`: every TP2 cell also gets two independent
TP1 replicas on the same two GPUs). Measured 2026-10-30 (fix round 2026-10-31) on the alola `rocke` partition
(MI355X, 256 CUs). Ratios, shares and modelled byte counts only; absolute step times, bandwidths, tok/s and latencies
are in `hackathon-prep/private-perf/qwen38-true-mono/large-batch/L0c-DP2.md`.

## 0. Summary

1. **DP2 vs TP2 per TP2 cell (both ISLs, B8-B128):**
   - **FlyTape: TP2 is the stronger reference in all 10 cells.** DP2 / TP2 per-request time 1.047-1.162 (4 of 4
     rounds above 1 in every cell, outside both A-vs-A controls: TP2s / TP2 0.984-1.010, DP2s / DP2 0.970-1.005). A
     second full-grid job (fix round, 68225698, GPU pair f5 + 85) gives 1.053-1.156 and the same verdict in every
     cell. Between jobs a cell's median moves by up to 0.028 (B128 p128: 1.134 / 1.156 / 1.162 in three jobs), more
     than the in-job A-vs-A band, so read the FlyTape DP2 / TP2 values as +-0.015 (per-cell ranges in the Fix round
     section). At the same B that is 0.86-0.96x TP2's decode throughput; at TP2's per-request time DP2 gives
     0.61-0.87x its throughput (B16-B128). At B8, TP2's per-request time is below every measured DP2 point (B4 per
     replica and up), so DP2 does not reach it within the measured range.
   - **ATOM native: DP2 is the stronger reference in all 10 cells under the e2e WIN rule (fix round).** The fix round
     ran 6 shuffled pairs per cell (7 at B8) of TP2 K3, TP2 K1 (B8, B128), DP2 K3 and the A-vs-A twins TP2s K3 / DP2s
     K3, after a GSM8K score gate with two must-fail mutants caught on hardware. Best DP2 arm (K3) / best TP2 arm (K1
     at B8, K3 elsewhere): median 0.751-0.901, every valid pair below 1, every median below its A-vs-A band
     (band minimum 0.889-0.981). Served output tok/s 1.23-1.41x (mean over pairs). The first round's 2-rep data
     (9 of 10, 1k B16 a tie) are superseded; 1k B16 is now 0.839 (0.776-0.898) against a 0.893-1.035 band. At
     equal K (2-rep data, K0 / K1): DP2 / TP2 is 0.70-0.92 except at B8-B16 without MTP (K0 0.99-1.02) and K1 at B8
     (0.97-1.00): ATOM's TP2 step is especially slow at 32 rows (K0 B32, K1 B16, K3 B8), and its all-reduce / host
     path costs more than TP2 saves.
   - **Cross-stack fastest per-token decode (not a matched comparison).** The cell's fastest of FlyTape TP2 (K0 device
     step, random weights, W4A4 experts) and ATOM DP2 K3 (served, real weights, W4A8, MTP) compares across precision
     and metric, as BASELINE.md's reference column does; it says which bar the rocKE kernel faces, not which stack
     wins at matched precision. With the fix-round ATOM data and both FlyTape jobs (a gap counts only when it is above
     3 % with both FlyTape jobs): ATOM DP2 K3 is faster at 1k B8, 8k B8 and 8k B16 (FlyTape TP2 1.039-1.123x it); the
     two are within 3.2 % at 1k B16-B64 and 8k B32-B64; FlyTape TP2 is faster at B128 (ATOM DP2 K3 1.107-1.158x it).
     Table in section 5 and `BASELINE.md`.
2. **Bytes model** (`rocke/library/kernels/gfx950/qwen38_mono/large_batch/bytes_model.py`, 19 CPU tests incl. 5
   must-fail mutants): per stage, per rank, rows 8-128, TP1 / TP2, three precision regimes (FlyTape-`matched`, LB at
   ATOM's `checkpoint` numerics, `atom` = what ATOM's native path reads). It reproduces every column of DESIGN-LB
   section 3 within 2 % and FlyTape's per-rank TP2 B128 estimate (48 GB) within 1.9 % on the same components.
3. **Bandwidth (Triton probe at 512 threads per CTA, cold, 3 jobs on 2 nodes):** the 1 GiB read stream at 256 CTAs is
   0.96-0.98 of the best CTA count (384); the curve is 0.83 at 128 CTAs and 0.48 at 64. A 1:1 read+write stream (the
   GDN state pattern) moves 0.87-0.94 of the read stream's bytes. Random vs adjacent expert slots cost at most 3.3 %.
   Small stages are latency-bound: a cold stage read after a grid barrier reaches 0.05-0.37 of the stream rate at
   0.25-2 MiB and 0.5-0.64 at 64 MiB; the probe's atomic grid barrier alone costs 0.25-0.36 of a 64 MiB stage read.
4. **Floors:** at TP2 B128 the streamed floor is 0.59-0.63 routed experts and 0.31-0.33 GDN state (MXFP4 checkpoint:
   0.93 of FlyTape's bytes); at TP2 B8 everything but experts and state is 0.20 (checkpoint) to 0.46 (matched). **FlyTape runs at
   0.25-0.49 of its own streamed floor at TP2 and 0.36-0.59 at TP1** (higher with batch); ATOM native K0 at 0.11-0.27
   (TP2) and 0.19-0.42 (TP1). The checkpoint-precision floor is 0.17-0.45 of FlyTape's TP2 step (B8-B128): the
   headroom is in efficiency at small B and in bytes + efficiency at large B. Serializing stages (no cross-stage
   overlap) costs 1.07-1.49x the streamed floor at TP1 and 1.13-1.81x at TP2, before any barrier.
5. **DP2 vs TP2 floors:** one DP2 GPU (TP1 at B/2) reads 1.14-1.40x the bytes of one TP2 rank (B), so TP2 has the
   lower bytes floor at every cell; the LB kernel's TP2 must keep its exchange cost below that 14-40 % to beat its own
   DP2, which FlyTape does (5-16 %) and ATOM native does not.

## 1. What was run

| job | what | node, GPUs (bus) | gates |
|---|---|---|---|
| 68220975 `w4-ft-dp2` | FlyTape `bench_qwen4_full.py` mxfp4, 4 rounds x arms {TP2, TP2s, DP2, DP2s} shuffled, 2 reps per arm-run, configs B 8-128 (DP2: B/2 per replica) x prompt 128 / 8192 | n01-09, 05 + 15 | numerics + mutant first; replica / run token agreement; rc, status; overlap |
| 68220965 `w4-atom-dp2-r1`, 68221379 `w4-atom-dp2-r2` | ATOM native (`atom-rkrepro`, `ATOM_MONO_ENABLE=0`), arms {TP2, DP2} x {K0, K1, K3} shuffled, the BASELINE cells, GSM8K 64 per server | n07-17, 05 + 65 / 15 + 85 | tracebacks 0, mono off, GSM8K, cells complete, DP2 overlap >= 0.8 |
| 68222460 `w4-bw7` (primary), 68221569 `w4-bw4`, 68221335 `w4-bw3` | `bw_lb.py`: CTA curve, bytes ramp, read+write, expert segments, in-kernel stage ramp; 5 rounds shuffled | n07-17 f5; n01-09 15; n01-09 f5 | numerics + 15 must-fail mutants first; every timed read checksum-verified |
| fix round: 68225658 `w4-atom-pairs-a` (pairs 1-3), 68225982 `w4-atom-pairs-b2` (pairs 4-6), 68228847 `w4-atom-pairs-c` (pairs 7-8, B8 cells only) | ATOM native, arms {TP2 K3, TP2s K3, DP2 K3, DP2s K3, TP2 K1 (B8, B128)} shuffled per pair, the BASELINE TP2 cells, GSM8K 64 per server | n01-09, 05 + e5 / 75 + 85 / 75 + 85 | GSM8K score >= 0.80 of 64 per server; must-fail mutants first (dummy-weight and sampled servers: b2, c; a's mutant crashed at load); tracebacks 0, mono off, cells complete, DP2 overlap >= 0.8 (failing samples excluded) |
| fix round: 68225698 `w4-ft-dp2-p3` | FlyTape, as 68220975, FT_SEED=11 | n01-09, f5 + 85 | as 68220975 (fixed summarizer in-job) |

All jobs had their GPUs to themselves (`amd-smi process`: no other process). Clock samplers (`amd-smi`, every 5-30 s):
no busy sample below 0.9 of the busy median gfx clock in any job, except one 10 s sample at 0.67 in 68225698 during
round 1's compile phase of DP2s (an idle-to-busy ramp: the samples around it are idle). Harness:
`hackathon-prep/qwen38/truemono-lb/l0c-dp2/` (section 8).

**Arms.** FlyTape TP2 = one `--tp 2` process (ranks on GPU 0 / 1). FlyTape DP2 = two `--tp 1` processes at B/2,
replica r on GPU r, each with its own FlyDSL cache, and a file barrier before every sample so the two timed decodes
overlap (median overlap 0.987, minimum 0.655: section 6). TP2s / DP2s = the same arms with fresh FlyDSL caches (recompiled) and, for DP2s, the
replicas on swapped GPUs: the A-vs-A controls. ATOM TP2 = one `-tp 2` server. ATOM DP2 = two `-tp 1` servers (one
per GPU, own ports); a cell at CONC c runs `benchmark_serving` on both at once, each at CONC c/2 with half the prompts
(different seeds, same ISL / OSL, ignore-eos); the cell's decode time is the mean of the two servers (server spread
median 1.7 %), its output tok/s the sum (window overlap median 0.987, min 0.84; fix round: median 0.96, 3 of 128
DP2 / DP2s samples below 0.8, all at B8, where the two servers' MTP acceptance differs by up to 22 % and the faster one
finishes first).

**Metrics** are BASELINE.md's: FlyTape = median device step (no MTP, random weights, closed batch); ATOM dec =
median inter-chunk latency / acceptance (served, real weights, MTP), TPOT = the serving number. At matched B a lower
per-request time is also the higher decode throughput, so "stronger" = faster per request.

## 2. Bytes model

`bytes_model.step_bytes(tp, rows, ctx, regime, state_bytes)` returns per stage (DESIGN-LB 4.2 stage list: HC down / up
x 2 per layer, `gdn_in` with the conv window, `gdn_recur`, `gdn_out`, `qsa_in`, `qsa_core` (KV of the attended tokens
up to the 2,048 budget, the compressed-key scan past it, the append), `qsa_out`, router, routed experts (U active of
512, `E (1 - (1 - 10/E)^B)` for random routing or a measured U), shared expert, PLE, final mix, `lm_head`, embed) the
weight, state, KV and scratch bytes per occurrence and the occurrence count. MXFP4 = 0.53125 B per weight; TP shards
GDN / QSA heads, the MoE intermediate (320 + 320) and the vocab, and replicates the rest.

Regimes: `matched` = FlyTape's `mxfp4` preset (experts MXFP4 W4A4, everything else bf16, conv state bf16 x 4 slots);
`checkpoint` = the LB kernel at ATOM's numerics (every dense linear MXFP4, HC mix weights and `lm_head` stored as
lossless MXFP4 because the checkpoint keeps them MXFP4-rounded, router bf16, conv window 3 slots); `atom` = ATOM's
native reads (as `checkpoint` but HC mix and `lm_head` bf16).

Validation (`validate()`, CPU test `library/tests/test_qwen38_lb_bytes_model.py`): all ten DESIGN-LB section 3 rows
within 2 % on every column (experts, GDN state, dense MXFP4, dense bf16, LB total, FlyTape total); FlyTape's TP2 B128
48 GB (state + experts + dense) at +1.9 %; FlyTape's per-tensor table (gdn_in 84.4 MB, gdn_out 31.5, HC down 6.6, up
6.55, router 2.6, qsa_in 71.4, lm_head 1.27 GB, ple 65.5) within 1 % (router 2 %). Must-fail mutants (each makes
`validate()` fail): MXFP4 without scales, MXFP4 counted as FP8, naive U = min(512, 10 B), 32 value heads, bf16 state as
the default.

GB per rank per step, p128 (the full table incl. p8192 and the per-family split, and every floor table, are in
`L0c-DP2-tables.md` next to this file):

| TP | B | U | matched | checkpoint | atom | checkpoint / matched | routed experts | GDN state r+w |
|---|---|---|---|---|---|---|---|---|
| 1 | 8 | 74.7 | 19.52 | 13.62 | 15.50 | 0.698 | 9.37 | 1.81 |
| 1 | 16 | 138.6 | 29.41 | 23.50 | 25.37 | 0.799 | 17.37 | 3.62 |
| 1 | 32 | 239.6 | 45.85 | 39.91 | 41.78 | 0.870 | 30.04 | 7.25 |
| 1 | 64 | 367.1 | 69.36 | 63.38 | 65.25 | 0.914 | 46.01 | 14.50 |
| 1 | 128 | 471.0 | 97.46 | 91.38 | 93.25 | 0.938 | 59.03 | 28.99 |
| 2 | 8 | 74.7 | 10.52 | 7.06 | 8.46 | 0.671 | 4.68 | 0.91 |
| 2 | 16 | 138.6 | 15.46 | 11.99 | 13.40 | 0.776 | 8.68 | 1.81 |
| 2 | 32 | 239.6 | 23.68 | 20.20 | 21.61 | 0.853 | 15.02 | 3.62 |
| 2 | 64 | 367.1 | 35.44 | 31.94 | 33.34 | 0.901 | 23.01 | 7.25 |
| 2 | 128 | 471.0 | 49.49 | 45.94 | 47.34 | 0.928 | 29.52 | 14.50 |

- p8192 adds 1.3-1.6 GB (TP1 B128: 3.1) of attended KV and key scan; `state=bf16` (N11) cuts the checkpoint total to
  0.84 (B128) - 0.93 (B8).
- One DP2 GPU (TP1 at B/2) vs one TP2 rank (B): 1.34 / 1.26 / 1.24 / 1.29 / 1.40 (matched, B8-B128), 1.16 / 1.14 /
  1.16 / 1.25 / 1.38 (checkpoint): DP2 replicates the dense weights and the hot experts (U at B/2 is more than half
  of U at B).

## 3. Bandwidth curves (`bw_lb.py`, ratios)

Cold 1 GiB read stream vs CTA count, ratio to 256 CTAs (best of 16 / 64 KiB per program iteration):

| CTAs | 8 | 16 | 32 | 64 | 96 | 128 | 160 | 192 | 224 | 256 | 320 | 384 | 512 | 768 | 1024 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bw7 (n07-17) | 0.068 | 0.134 | 0.259 | 0.482 | 0.672 | 0.831 | 0.951 | 1.013 | 1.011 | 1 | 0.996 | 1.023 | 1.017 | 1.014 | 0.917 |
| bw4 (n01-09) | 0.068 | 0.133 | 0.257 | 0.479 | 0.667 | 0.825 | 0.955 | 1.025 | 1.015 | 1 | 1.001 | 1.036 | 1.017 | 1.025 | 0.932 |
| bw3 (n01-09) | 0.068 | 0.133 | 0.256 | 0.477 | 0.669 | 0.830 | 0.949 | 1.020 | 1.030 | 1 | 1.015 | 1.045 | 1.028 | 1.031 | 0.915 |

- Controls at 256 CTAs: A-vs-A 0.986 / 0.995 / 1.004, recompiled kernel 0.993 / 0.990 / 1.004; the 256-CTA value
  across the three jobs (two nodes, three GPUs) within 0.9 %.
- 1:1 read+write (traffic counted both ways), ratio to the read stream at the same CTAs: 0.92-0.94 at 128 CTAs,
  0.87-0.94 at 256 (plain vs streaming `.cs` stores: no consistent difference), 0.77-0.89 at 512.
- Expert segments (U of 512 slots of one expert's MXFP4 bytes, 256 CTAs): random / adjacent slots in the same kernel
  0.967-1.006 (TP2 expert size) and 0.981-1.006 (TP1). The expert gather pattern is not a bandwidth problem; the
  probe kernel's dependent table load is (it reaches about half the stream rate in both arms).
- In-kernel stage ramp (R cold stages of b bytes, each followed by a grid barrier, in one launch; (T(R) - T(1)) /
  (R - 1); read time = stage time - the zero-byte stage), efficiency = stream time of b / read time:

  | stage MiB | 0.25 | 0.5 | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
  |---|---|---|---|---|---|---|---|---|---|
  | 128 CTAs | 0.08 | 0.17 | 0.26 | 0.37 | 0.25 | 0.34 | 0.45 | 0.55 | 0.64 |
  | 256 CTAs | 0.05 | 0.15 | 0.29 | 0.29 | 0.24 | 0.16 | 0.31 | 0.36 | 0.51 |

  The barrier-only stage (atomic counter, release / acquire, all programs) costs 0.25 (128 CTAs) and 0.36 (256) of
  the 64 MiB read. Event timing of single launches cannot resolve reads under about 32 MiB (launch overhead); this
  in-kernel ramp is the probe's (a Triton loop; one block load per program in the best case), so it bounds what an
  emitter that issues every load before the barrier can reach from below, not from above.

## 4. Per-stage floors (`floors.py`: bytes model x bw7 curves)

`F_stream` = every stage's bytes at its class's asymptotic rate at 256 CTAs (weights / KV: read stream; GDN state:
read+write stream; experts: read stream x the measured gather ratio): the floor of a kernel that overlaps all stages.
`F_serial` = every occurrence read cold and alone at the measured stage ramp (best of 128 / 256 CTAs), barriers
excluded. eta = F_stream / measured step. Floors from bw4 instead of bw7: F_stream 0.989-1.009, F_serial 0.971-1.002.

| TP | B | ckpt bytes / matched | F_serial / F_stream (matched, ckpt) | experts | GDN state | GDN proj | HC | QSA | eta FlyTape (matched) | ckpt F_stream / FlyTape step | eta ATOM K0 (atom) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 8 | 0.698 | 1.49, 1.45 | 0.685 | 0.139 | 0.082 | 0.024 | 0.025 | 0.367 | 0.258 | 0.187 |
| 1 | 16 | 0.799 | 1.33, 1.27 | 0.734 | 0.162 | 0.049 | 0.014 | 0.016 | 0.453 | 0.363 | 0.267 |
| 1 | 32 | 0.870 | 1.21, 1.16 | 0.742 | 0.193 | 0.031 | 0.008 | 0.011 | 0.534 | 0.466 | 0.325 |
| 1 | 64 | 0.914 | 1.14, 1.10 | 0.715 | 0.240 | 0.022 | 0.005 | 0.008 | 0.588 | 0.538 | 0.392 |
| 1 | 128 | 0.938 | 1.10, 1.07 | 0.632 | 0.332 | 0.018 | 0.004 | 0.008 | - | - | 0.390 |
| 2 | 8 | 0.671 | 1.80, 1.74 | 0.665 | 0.133 | 0.078 | 0.046 | 0.025 | 0.258 | 0.175 | 0.114 |
| 2 | 16 | 0.776 | 1.57, 1.47 | 0.718 | 0.159 | 0.048 | 0.028 | 0.016 | 0.317 | 0.247 | 0.161 |
| 2 | 32 | 0.853 | 1.38, 1.29 | 0.736 | 0.188 | 0.030 | 0.016 | 0.011 | 0.413 | 0.353 | 0.163 |
| 2 | 64 | 0.901 | 1.26, 1.18 | 0.711 | 0.238 | 0.021 | 0.010 | 0.008 | 0.440 | 0.397 | 0.222 |
| 2 | 128 | 0.928 | 1.18, 1.13 | 0.629 | 0.330 | 0.018 | 0.007 | 0.008 | 0.483 | 0.449 | 0.255 |

Context 128; family columns are shares of the checkpoint F_stream (router + shared, head + PLE: 0.004-0.026 each). At
p8192 eta changes by under 0.01 for FlyTape and rises for ATOM (0.118-0.273 at TP2, 0.190-0.416 at TP1); QSA takes
0.05-0.08 of the checkpoint floor.

Per stage at TP2 B128 p128 (share of F_stream, checkpoint; F_serial / F_stream of the stage; MB per occurrence):
routed experts 0.629 (1.06, 615 MB per layer), `gdn_recur` 0.330 (1.09, 403 MB), `lm_head` 0.004 (1.22, 169 MB),
`gdn_in` 0.015 (2.14, 19.1 MB), `qsa_core` 0.004 (2.22, 16.9 MB), `gdn_out` 0.003 (3.93, 4.2 MB), each HC
down / up 0.002 (2.9, 1.7-1.8 MB), router 0.003 (3.06, 2.6 MB), shared expert 0.001 (3.21, 1.3 MB). At TP2 B16 the
experts are 0.718 (1.20, 181 MB per layer), the GDN state 0.159 (1.66, 50 MB), `gdn_in` 0.036 (2.52), the four HC
stages 0.028 together (2.9 each). Small dense stages read at 0.25-0.35 of the stream rate when they run alone; they
are where cross-stage overlap (weights issued before the poll) pays, while the two big streams are already at
1.06-1.09 of their floor when serialized at TP2 B128 (B16: experts 1.20, state 1.66).

DP2 vs TP2 streamed floors (one DP2 GPU at B/2 over one TP2 rank at B; TP2 floor without its all-reduces):

| ctx | regime | B8 | B16 | B32 | B64 | B128 |
|---|---|---|---|---|---|---|
| 128 | matched | 1.326 | 1.265 | 1.243 | 1.279 | 1.395 |
| 128 | checkpoint | 1.146 | 1.141 | 1.166 | 1.234 | 1.373 |
| 128 | atom | 1.176 | 1.161 | 1.176 | 1.238 | 1.372 |
| 8192 | checkpoint | 1.140 | 1.134 | 1.156 | 1.219 | 1.343 |

Measured DP2 / TP2 per-request time: FlyTape 1.047-1.162 (it keeps 14-64 % of its floor advantage), ATOM 0.70-1.02
at equal K (its TP2 loses more to exchange and host time than the bytes save).

## 5. DP2 arms and the reference per TP2 cell

Every column is a per-token decode time over the fastest of the four arms (> 1 = slower), with FlyTape from job
68220975 and ATOM from the fix round's pairs (Fix round section). FlyTape's 1k column is its prompt-128 step (BASELINE caveat).
FT verdict: DESIGN.md 9.1 win rule over 4 rounds against both A-vs-A controls; the second FlyTape job's median
(68225698) is in the next column. ATOM verdict: the e2e WIN rule (median over the valid pairs below the A-vs-A band
minimum AND at least 5/6 of the pairs below 1); the column gives the median (min-max over the pairs, pairs below 1;
A-vs-A band = every pair's TP2s / TP2 and DP2s / DP2). ATOM out tok/s: mean over the pairs. Fastest: FlyTape TP2 vs
ATOM DP2 K3 across precision and metric (section 0); "within" = the gap is at most 3 % with one of the two FlyTape
jobs (the range covers both).

| ISL | B | FT TP2 | FT DP2 | FT DP2/TP2 (rounds) | 2nd FT job | FT | ATOM TP2 best (arm) | ATOM DP2 best (arm) | ATOM DP2/TP2 median (pairs; A-vs-A band) | ATOM | ATOM out tok/s DP2/TP2 | fastest (cross-stack) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1k | 8 | 1.091 | 1.140 | 1.047 (1.038-1.049) | 1.053 | TP2 | 1.302 (K1) | 1.000 (K3) | 0.761 (0.727-0.815, 7/7 < 1; 0.889-1.165) | DP2 | 1.258 | ATOM DP2 K3 (FT TP2 1.088-1.091) |
| 1k | 16 | 1.027 | 1.142 | 1.114 (1.100-1.120) | 1.105 | TP2 | 1.191 (K3) | 1.000 (K3) | 0.839 (0.776-0.898, 6/6 < 1; 0.893-1.035) | DP2 | 1.314 | within (FT TP2 1.027-1.031) |
| 1k | 32 | 1.000 | 1.150 | 1.155 (1.133-1.158) | 1.140 | TP2 | 1.169 (K3) | 1.032 (K3) | 0.891 (0.849-0.913, 6/6 < 1; 0.970-1.053) | DP2 | 1.234 | within (ATOM DP2 K3 1.024-1.032) |
| 1k | 64 | 1.007 | 1.059 | 1.052 (1.050-1.054) | 1.058 | TP2 | 1.107 (K3) | 1.000 (K3) | 0.901 (0.886-0.929, 6/6 < 1; 0.981-1.045) | DP2 | 1.253 | within (FT TP2 1.007-1.015) |
| 1k | 128 | 1.000 | 1.161 | 1.162 (1.158-1.163) | 1.156 | TP2 | 1.290 (K3) | 1.158 (K3) | 0.897 (0.895-0.903, 6/6 < 1; 0.977-1.025) | DP2 | 1.283 | FT TP2 (ATOM DP2 K3 1.141-1.158) |
| 8k | 8 | 1.123 | 1.182 | 1.055 (1.044-1.056) | 1.056 | TP2 | 1.328 (K1) | 1.000 (K3) | 0.751 (0.681-0.808, 7/7 < 1; 0.904-1.053) | DP2 | 1.407 | ATOM DP2 K3 (FT TP2 1.120-1.123) |
| 8k | 16 | 1.039 | 1.145 | 1.106 (1.091-1.107) | 1.096 | TP2 | 1.171 (K3) | 1.000 (K3) | 0.855 (0.834-0.872, 6/6 < 1; 0.937-1.054) | DP2 | 1.296 | ATOM DP2 K3 (FT TP2 1.039-1.040) |
| 8k | 32 | 1.000 | 1.137 | 1.138 (1.127-1.145) | 1.133 | TP2 | 1.163 (K3) | 1.007 (K3) | 0.871 (0.824-0.895, 6/6 < 1; 0.980-1.038) | DP2 | 1.306 | within (0.7 % either way) |
| 8k | 64 | 1.003 | 1.077 | 1.074 (1.068-1.077) | 1.078 | TP2 | 1.111 (K3) | 1.000 (K3) | 0.896 (0.878-0.935, 6/6 < 1; 0.961-1.041) | DP2 | 1.289 | within (FT TP2 1.003-1.008) |
| 8k | 128 | 1.000 | 1.106 | 1.106 (1.103-1.108) | 1.124 | TP2 | 1.239 (K3) | 1.107 (K3) | 0.896 (0.869-0.904, 6/6 < 1; 0.961-1.026) | DP2 | 1.328 | FT TP2 (ATOM DP2 K3 1.107-1.113) |

The first round's ATOM columns (2 reps, superseded): best DP2 / best TP2 0.753-0.920, 1k B16 a tie (0.851; reps
0.795, 0.909), the other nine DP2.

ATOM DP2 dec / TP2 dec at the same K (K0 / K1 / K3: first round, mean of 2 reps; K3 pairs: fix round, mean over the
valid pairs):

| ISL | B | K0 | K1 | K3 | K3 pairs |
|---|---|---|---|---|---|
| 1k | 8 | 1.008 | 1.003 | 0.708 | 0.764 |
| 1k | 16 | 0.989 | 0.696 | 0.851 | 0.840 |
| 1k | 32 | 0.708 | 0.841 | 0.852 | 0.883 |
| 1k | 64 | 0.838 | 0.884 | 0.895 | 0.903 |
| 1k | 128 | 0.887 | 0.905 | 0.915 | 0.898 |
| 8k | 8 | 1.016 | 0.965 | 0.741 | 0.724 |
| 8k | 16 | 0.990 | 0.704 | 0.829 | 0.854 |
| 8k | 32 | 0.706 | 0.843 | 0.853 | 0.866 |
| 8k | 64 | 0.849 | 0.878 | 0.895 | 0.900 |
| 8k | 128 | 0.884 | 0.904 | 0.920 | 0.893 |

The 0.70-0.71 cells are the 32-row steps (K0 B32, K1 B16, K3 B8) where ATOM's TP2 is an outlier (BASELINE.md). Without
them DP2 still wins by 8-16 % at B64-B128 and ties at B8-B16 without MTP.

Iso-latency (user decision 5: "at equal per-request latency"): DP2's decode throughput interpolated (log-log, over its
own batch curve) at the TP2 arm's per-request time, over TP2's throughput (ATOM: fix-round pairs; first round in
brackets):

| ISL | B | FlyTape | ATOM (best arm per mode) |
|---|---|---|---|
| 1k | 8 | below DP2's measured range | 2.202 (1.935) |
| 1k | 16 | 0.730 | 1.724 (2.013) |
| 1k | 32 | 0.612 | 1.325 (1.379) |
| 1k | 64 | 0.872 | 1.195 (1.221) |
| 1k | 128 | 0.738 | >= 1.114 (>= 1.093) |
| 8k | 8 | below DP2's measured range | 2.253 (2.514) |
| 8k | 16 | 0.733 | 1.718 (1.876) |
| 8k | 32 | 0.623 | 1.355 (1.392) |
| 8k | 64 | 0.840 | 1.214 (1.217) |
| 8k | 128 | 0.795 | >= 1.120 (>= 1.087) |

(">= x": TP2's time is above every DP2 point; x is DP2's B128 throughput over TP2's, a lower bound. "Below DP2's
measured range": TP2's B8 per-request time is shorter than DP2's at every measured batch, B4 per replica and up; a
smaller DP2 batch was not measured.)

**What this means for the targets** (user decision 5: beat the better of TP2 and DP2 per cell): at TP2 cells the rocKE
LB kernel must beat FlyTape TP2 at B128, ATOM DP2 K3 at 1k B8 and 8k B8-B16, and both (within 3.2 % of each other)
at 1k B16-B64 and 8k B32-B64. The ATOM DP2 K3 bar is a served MTP number at real weights: matching it with a K=0
kernel needs a step under 0.89-0.96x FlyTape's TP2 step at 1k B8 and 8k B8-B16 and about 0.97-1.00x at the "within"
cells; with MTP the LB round can use the per-token gain ATOM shows (acceptance 2.7-2.9). FlyTape's own DP2 is never
the stronger reference.

## 6. Numerics, discipline, controls

- **FlyTape:** before timing, DP2 at B4 x {128, 8192}: both replicas produced bitwise-identical tokens (sha256 of the
  64-token output); the must-fail mutant (replica 1 decodes a different last prompt token in row 0) was caught (digests
  differ). During timing: replicas agree in every sample; each config's tokens agree across all 8 TP2 / TP2s runs and
  all 16 DP2 / DP2s replica runs (one digest per config and mode), recompiled arms included. Every process rc 0 and
  status `max_tokens`; no barrier timeout. **The in-job gate of job 68220975 printed `TIMING GATES: FAIL`:** the
  summarizer staged with that job accepted only status `ok`, and FlyTape's completed launch reports `max_tokens`
  (`flytape.runtime.ExitStatus`). All 486 rows are `max_tokens`; the summarizer was fixed after the job
  (`OK_STATUS = {max_tokens, all_eos}`) and the local re-run on the same raw data passes. The fix-round FlyTape job
  (68225698) ran the fixed summarizer in-job and printed `NUMERICS GATE: PASS` and `TIMING GATES: PASS`.
- **FlyTape replica overlap:** median 0.987, minimum 0.655. 3 of 160 DP2 samples are below 0.8 (0.655-0.687), all
  at B64 per replica and p8192 (the 8k B128 cell); their device steps are 0.998 of the other samples of that config,
  so the metric (device step) is unaffected. The fix-round job shows the same pattern: 3 of 160 below 0.8
  (0.658-0.690), all in that config, device steps 0.996 of the rest; median 0.988.
- **FlyTape spread:** TP2 steps match the W3 baseline (other node and GPU pair) within 1.4 %. Within DP2 the second
  replica process is 0.3-2 % slower than the first whichever GPU it runs on (replica spread up to 3.9 %); DP2s / DP2 (swapped
  GPUs, recompiled) 0.970-1.005. Both inside the verdict margins (4.7-16 %).
- **ATOM, first round:** 12 server runs (6 TP2, 6 DP2 servers x 2): 0 tracebacks, mono off, no preemption, every cell
  complete (`completed` = prompts per server), DP2 window overlap >= 0.84; GSM8K 56-61 of 64 per server (DP2 servers
  56-61, TP2 56-58), inside the 56-62 spread of BASELINE.md. Every server's startup table holds 128 rows at all
  contexts. **Its accuracy gate was not automated:** the summarizer only required that GSM8K had run, no ATOM mutant
  ran, and the scores were checked by eye. The fix round's gate (below), re-run on the first round's raw data, passes
  every server, and the merged cells are unchanged.
- **ATOM GSM8K gate (fix round, `summarize_dp2.py`):** every server must answer all 64 GSM8K prompts and score at least
  0.80 of them (52 / 64; BASELINE's spread is 56-62), else the run fails. Must-fail mutants run first in the job, one
  `-tp 1` server each: `--load_dummy zero` (all weights 0) and real weights queried at temperature 3.0. Both served
  all 64 prompts and scored 0 / 64 (every reply at the 2,048-token cap), so both were caught, in jobs 68225982 and
  68228847. The gate passes when every served mutant fails the score gate and at least one mutant served. In job
  68225658 (the first fix-round job) the only mutant was `--load_dummy xavier`, which crashed at model load
  (`hipErrorLaunchFailure`, the server never came up) on a GPU that then served real-weight arms for 3.5 h; that
  job's own gate reports `NO MUTANT CAUGHT`, and its arms rely on the same gate's mutants caught in the two later
  jobs. All 56 fix-round arm servers scored 56-61 / 64 with 0 tracebacks, mono off, no preemption, every cell
  complete.
- **ATOM harness tests (CPU, `test_l0c_dp2_harness.py`, 15 tests):** each of these fails its gate: sequential DP2
  windows, a server serving all prompts, a traceback, a GSM8K score of 20 / 64, a GSM8K run that answered 10 prompts,
  51 / 64 (52 / 64 passes), a mutant scoring like a real server, a mutant that did not serve, a requested mutant that
  never ran, an uncaught mutant next to a caught one, a rep-index collision between merged jobs, a FlyTape replica
  disagreement, an uncaught FlyTape mutant, a bad status and non-overlapping replicas. 4 of 6 pairs below 1, a median
  inside the A-vs-A band and an overlap-failed pair (excluded: n = 5, 4 of 5 below 1) are ties. With the score
  threshold set to 0 the score tests fail, and with the win fraction lowered to 4 / 6 the win-rule test fails.
- **ATOM DP2 overlap in the fix round:** 3 of 128 DP2 / DP2s samples fell below the 0.8 overlap gate (0.71-0.77, all
  at B8: the two servers start together, and the one with the higher MTP acceptance finishes up to 1.5x earlier).
  They are excluded from the win rule with their pair. The supplementary job 68228847 added pairs 7-8 for the B8
  cells, so both still have 7 valid pairs (one more DP2s sample there failed and is left out of the A-vs-A band).
- **ATOM spread:** TP2 arms vs W3's TP2 baseline (other GPUs): K0 within 1 %, K1 / K3 within 6 % (random-prompt
  acceptance); first-round rep-to-rep spread of the best arms 1.3-9.5 % (largest at B16 K3). Fix round: the
  coefficient of variation over the pairs is 0.6-4.9 % for DP2 K3 (largest at B8) and 0.6-4.4 % for TP2 K3; the
  A-vs-A band is 0.889-1.165 at 1k B8 and within 0.937-1.054 elsewhere.
- **Bandwidth:** numerics gate first in every job (read / copy / streaming copy / segment / staged-barrier kernels
  against exact references, 12 (bw3) or 15 (bw4, bw7) mutants: skipped chunks, shifted stores, adjacent-slot reads, off-by-one-chunk stage
  reads), then every timed read and segment launch checksum-verified (0 failures). The stage barrier's arrival count
  equals stages x programs in every launch.
- All arms unstamped; shuffled orders per round / rep logged (`order.txt`); node and GPU bus ids per job in the
  private file.

## 7. Caveats

- FlyTape and ATOM still differ in numerics and workload (BASELINE.md caveats): FlyTape = random weights, zero state,
  W4A4 experts, bf16 dense, closed batch, device step; ATOM = real weights, W4A8, MTP, open serving with interleaved
  prefill. The reference column mixes the two per-token metrics exactly as BASELINE.md does.
- ATOM DP2 servers get different prompt sets (seeds) of the same shape; acceptance differs by up to 0.26 between
  servers / reps at K3, which is most of the ATOM rep spread.
- The first round used two ATOM reps (one per job) instead of the 6-pair e2e rule. The fix round ran the rule (6
  pairs per cell, 7 at B8, A-vs-A twins in every pair) on the arms that decide the verdict (DP2 K3, TP2 K3, and TP2
  K1 at B8 and B128). K0, K1 at B16-B64 and DP2 K1 keep the first round's 2-rep data, which shows them slower than
  K3 by more than the spread.
- The fix-round pairs come from three jobs on two GPU pairs of n01-09 (05 + e5, 75 + 85); each pair is compared
  within its job.
- Floors use a Triton probe's bandwidth (one CTA per CU at 512 threads); a hand-written rocKE stream may reach the
  higher 384-CTA rate (1.02-1.05x) and a deeper stage ramp. U is the random-routing value (real routing is correlated;
  measured U from the router is still open).
- `qsa_core` counts the KV once per row and layer and the conv window at 3 / 4 slots; partials and exchange lines
  (MALL-resident) are outside the floor.

## 8. How to rerun

From `hackathon-prep` (Git Bash), `CL_HOST=vanantha@ctr2-alola-login-01`, `Q38_MAX_GPUS=10`; run `submit.sh` as a
background task (never under a short `timeout`, alola-cluster rule 148):

```bash
bash qwen38/truemono-lb/l0c-dp2/sync_l0c.sh w4-l0c          # stage = common/ + truemono-lb/{lbcommon.sh,l0c-dp2/}
S=qwen38/truemono-lb/submit.sh
$S logs/q38/w4/ft-dp2.poll --mode container --gpus 2 --time 03:30:00 --tag w4-ft-dp2 --stage w4-l0c --stage flytape \
  --env Q38_H=w4-l0c --env TMO=5000 -- 'bash /job/stages/w4-l0c/truemono-lb/l0c-dp2/ft_dp2.sh'
C="bench:1024:1024:8:16 bench:8192:1024:8:16 bench:1024:1024:16:32 bench:8192:1024:16:32 bench:1024:1024:32:64 bench:8192:1024:32:64 bench:1024:1024:64:128 bench:8192:1024:64:128 bench:1024:1024:128:256 bench:8192:1024:128:256"
$S logs/q38/w4/atom-dp2-r1.poll --mode container --gpus 2 --mem 400 --time 04:30:00 --tag w4-atom-dp2-r1 --stage w4-l0c \
  --stage atom-rkrepro --env Q38_H=w4-l0c --env REPS=1 --env P0=0 --env SEED=0 --env "CELLS=$C" \
  -- 'bash /job/stages/w4-l0c/truemono-lb/l0c-dp2/atom_dp2.sh'                   # rep 2: P0=1 SEED=1
$S logs/q38/w4/bw.poll --mode container --gpus 1 --time 00:50:00 --tag w4-bw --stage w4-l0c --env Q38_H=w4-l0c \
  -- 'bash /job/stages/w4-l0c/truemono-lb/l0c-dp2/bw_job.sh'
# fix round, e2e pairs: pairs 1-3 (P0=0); 4-6: P0=3; B8-only pairs 7-8: P0=6 REPS=2 CELLS=<the two B8 cells>
$S logs/q38/w4/atom-pairs-a.poll --mode container --gpus 2 --mem 400 --time 05:00:00 --tag w4-atom-pairs-a \
  --stage w4-l0c --stage atom-rkrepro --env Q38_H=w4-l0c --env REPS=3 --env P0=0 --env SEED=0 --env "CELLS=$C" \
  --env "ARMS=TP2:K3 TP2s:K3 DP2:K3 DP2s:K3 TP2:K1" \
  --env "ARM_CELLS=TP2:K1=bench:1024:1024:8:16,bench:8192:1024:8:16,bench:1024:1024:128:256,bench:8192:1024:128:256" \
  --env MUTANT=1 -- 'bash /job/stages/w4-l0c/truemono-lb/l0c-dp2/atom_dp2.sh'
```

Then locally (results under `/home_aus/AMD/vanantha/rocke-sdpa-claude/q38/results/<tag>/`):
`summarize_ft_dp2.py [--numerics] <ft dir>`, `summarize_dp2.py --merge out.json <atom dir>...`,
`summarize_dp2.py --pairs out.json pairs.json` (e2e WIN rule),
`dp2_compare.py --ft <ft summary.json> --atom <merged.json> [--pairs pairs.json] --ratio R --abs A`,
`floors.py --bw <bw_lb.json> --ft ... --atom ... --ratio R --abs A`, `bw_tables.py`, `bytes_table.py`. CPU tests:
`library/tests/test_qwen38_lb_bytes_model.py` (rocKE) and `l0c-dp2/test_l0c_dp2_harness.py`.

## Verification (independent verifier, 2026-10-31)

**Verdict: partial.** The FlyTape and bandwidth gates ran on hardware and their must-fail mutants were caught, and
the FlyTape TP2 vs DP2 result reproduces on another GPU pair. Two things do not hold as reported: the ATOM arms had
no automated accuracy gate and no must-fail mutant on hardware, and the in-job FlyTape gate printed FAIL before a
summarizer fix that the report does not mention. The DP2 conclusions still stand.

**Re-run on hardware (`w4-vL0c-DP2-ft2`, job 68224810).** n01-09, GPU buses f5 + 85 (the builder used n01-09 05 + 15).
The stage `w4-vL0c-DP2-h` holds the local `l0c-dp2/` harness. The run covered FlyTape TP2 / TP2s / DP2 / DP2s at prompt
128, B8 and B128, with 4 rounds in a new shuffle (FT_SEED=7). Absolute numbers are in the private file, section
"Verification".
- Numerics gate first: PASS. At B4 x {128, 8192} the two DP2 replicas produced bitwise-equal tokens. The must-fail
  mutant was caught (digests differ). The digests are identical to the builder's run, and so are the TP1 B4 / B64 and
  TP2 B8 / B128 timing-run digests across both jobs, so tokens are bitwise reproducible across GPU pairs.
- Timing gates: PASS (rc, status, replica, tokens across runs; overlap min 0.95, median 0.989; no barrier timeout).
  - DP2 / TP2 at B8 is 1.049 (rounds 1.039-1.059, 0 of 4 below 1). The builder measured 1.047.
  - At B128 it is 1.134 (1.129-1.150). The builder measured 1.162.
  - A-vs-A controls: TP2s / TP2 0.992-1.016, DP2s / DP2 0.992-1.016. Verdict "TP2 stronger" in both cells.
  - Verifier / builder per-arm steps: TP2 0.995 (B8), 1.007 (B128); DP2 0.999, 0.986.
- First attempt, job 68224048 (n07-17, buses e5 + 95): it landed on the known bad GPU e5 (alola-cluster rules
  146-152). That GPU memory-faulted in torch `normal_` during weight init and the process hung, so I cancelled the job
  and none of its data is used. The fact is recorded as alola-cluster rule 154.

**CPU and desk checks.**
- `test_qwen38_lb_bytes_model.py`: 19 passed. `test_l0c_dp2_harness.py`: 9 passed. `bytes_model.validate()` returns
  no failures.
- Recomputed one DP2 GPU (TP1 at B/2) over one TP2 rank (B):
  - matched: 1.339 / 1.263 / 1.242 / 1.294 / 1.402
  - checkpoint: 1.160 / 1.136 / 1.163 / 1.250 / 1.380
  - Both match section 2.
- `summarize_ft_dp2.py` (both modes) and `summarize_dp2.py --merge` re-run on the raw data reproduce the private
  summaries exactly.
- Hand-checked from the raw tables, all consistent:
  - the FlyTape iso-latency ratio at 1k B16 (0.73) and the ATOM ratio at 1k B8 (1.935)
  - the reference ratios at 1k / 8k B8 (1.132, 1.124)
  - "keeps 14-64 % of its floor advantage"
- ATOM: 120 of 120 cells ok, 0 tracebacks, mono off, no preemption, GSM8K 56-61 / 64 on every server. Both GPU pairs
  were idle before the run, and the arm order was shuffled per rep.
- Bandwidth: every job's log shows `NUMERICS GATE: PASS` with 15 GPU mutants (12 in bw3), each caught, and
  `TIMED CHECKSUMS: PASS`.

**Fairness (PLAN decision 5).**
- Matched for both FlyTape and ATOM: GPU count (2 vs 2), total batch (DP2 = 2 x B/2), precision (same preset or
  checkpoint in both modes) and metric (FlyTape: device step; ATOM: dec per token).
- ATOM DP2 servers had the same `--max-num-seqs`, `--max-model-len` and GMU settings, and their own compile cache.
  Each mode got its own GSM8K run.
- The DP2 iso-latency comparison is computed as decision 5 asks.
- The "reference" column still mixes FlyTape's K0 device step (random weights, W4A4) with ATOM's served MTP
  per-token time (real weights, W4A8). That is BASELINE's convention and both md files say so, but "ATOM DP2 K3 is
  now the reference" compares across precision and metric. It is not a matched-precision win for either side.

**Issues found.**
1. **ATOM numerics gate has no threshold and no mutant.** The report says every GPU result "passed its numerics gate
   and a must-fail mutant". For ATOM it did not:
   - `summarize_dp2.py` only requires GSM8K to have been checked (`bool(gsm_checked)`), not a minimum score.
   - `test_l0c_dp2_harness.py` has no low-score mutant.
   - No GPU mutant ran for ATOM.

   A server producing garbage would pass the automated gate. The actual scores (56-61 / 64) are inside BASELINE's
   56-62 spread, so the data are fine; the gate was checked by eye.
2. **Undisclosed in-job gate FAIL.** Job 68220975's own `job.log` ends with `TIMING GATES: FAIL`: the then-staged
   summarizer rejected status `max_tokens`. The local summarizer (OK_STATUS = {max_tokens, all_eos}) was fixed after
   the job and re-run to PASS. The fix is correct: all 486 rows are `max_tokens`, a completed launch per
   `flytape.runtime.ExitStatus`, and this verifier's job passes in-job with the fixed summarizer. Section 6 should still
   say so.
3. **The DP2 / TP2 size depends on the GPU pair.** At B128 the verifier's pair gives 1.134, outside the builder's
   round range (1.158-1.163) and its A-vs-A controls. Single-pair ranges understate cross-pair spread by about 2.5 %.
   The direction (TP2 stronger) is unaffected; quote the FlyTape DP2 / TP2 values as about +-0.03.
4. **Minor.**
   - The FlyTape overlap minimum (0.655) is not reported; only the median is. This is harmless, because the metric is
     the device step.
   - "DP2 cannot reach TP2's B8 latency at all" extrapolates below the smallest measured DP2 point (B4 per replica).
     It should read "not within the measured range".
   - ATOM ran 2 reps with a cross-job A-vs-A instead of the 6-pair e2e rule (already in the caveats). The margins hold
     at every cell except 1k B16, which the report already calls a tie.

## Fix round (2026-10-31)

Every issue from the verification is fixed. Sections 0, 1, 5, 6, 7 and 8 above are updated in place; this section
lists what changed and why. Absolute numbers are in the private file, section "Fix round".

**Jobs** (all on n01-09, `rocke` partition, q38 container, GPUs idle before each run, no throttled busy sample except
one idle-to-busy ramp sample in 68225698's compile phase):

| job | what | GPUs (bus) | in-job gate |
|---|---|---|---|
| 68225698 `w4-ft-dp2-p3` | FlyTape full grid again (TP2 / TP2s / DP2 / DP2s, 4 rounds, FT_SEED=11) | f5 + 85 | NUMERICS PASS (mutant caught, digests equal to the first job's), TIMING PASS |
| 68225658 `w4-atom-pairs-a` | ATOM e2e pairs 1-3, all 10 cells | 05 + e5 | arms pass; `NO MUTANT CAUGHT` (the xavier mutant crashed at load) |
| 68225982 `w4-atom-pairs-b2` | ATOM e2e pairs 4-6, all 10 cells | 75 + 85 | mutants caught (0 / 64 each); 2 B8 DP2 samples below the 0.8 overlap gate (excluded) |
| 68228847 `w4-atom-pairs-c` | ATOM e2e pairs 7-8, the two B8 cells | 75 + 85 | mutants caught (0 / 64 each); 1 B8 DP2s sample below the overlap gate (excluded) |

68225684 `w4-atom-pairs-b` was cancelled about 30 s after it started (only its xavier mutant was running) and
resubmitted as 68225982 with working mutants; nothing from it is used. Bus e5 on n01-09 is a different GPU from the
known bad e5 on n07-17 (alola-cluster rules 154, 157); it ran the 3.5 h job without a fault.

**Issue 1: ATOM accuracy gate without a threshold or a mutant.** Fixed.
- `summarize_dp2.py` now fails a run unless every server answered all 64 GSM8K prompts and scored at least 0.80
  of them (52 / 64; first-round and fix-round servers scored 56-61).
- `atom_dp2.sh MUTANT=1` runs must-fail mutants first in the job: `--load_dummy zero` and real weights sampled at
  temperature 3.0. On hardware both served all 64 prompts and scored 0 / 64, so they were caught (jobs 68225982,
  68228847). `--load_dummy xavier`, the first choice, crashes this checkpoint at load (job 68225658; alola-cluster
  rule 158), so it is not in the default list.
- CPU tests: 6 new must-fail cases for the score gate and the mutant logic. Mutation-checked: with the threshold at
  0 the score tests fail.
- The first round's raw data re-gated with the new summarizer: every server passes, and the merged cells are
  identical.

**Issue 2: undisclosed in-job `TIMING GATES: FAIL`.** Disclosed in section 6. The new FlyTape job ran the fixed
summarizer in-job and passed.

**Issue 3: FlyTape DP2 / TP2 depends on the job.** Fixed by measurement: a second full-grid job (68225698), on the
verifier's GPU pair. Per-cell DP2 / TP2 medians:

| cell | 68220975 (05 + 15) | 68225698 (f5 + 85) | verifier 68224810 (f5 + 85) | range |
|---|---|---|---|---|
| p128 B8 | 1.047 | 1.053 | 1.049 | 1.047-1.053 |
| p8192 B8 | 1.055 | 1.056 | - | 1.055-1.056 |
| p128 B16 | 1.114 | 1.105 | - | 1.105-1.114 |
| p8192 B16 | 1.106 | 1.096 | - | 1.096-1.106 |
| p128 B32 | 1.155 | 1.140 | - | 1.140-1.155 |
| p8192 B32 | 1.138 | 1.133 | - | 1.133-1.138 |
| p128 B64 | 1.052 | 1.058 | - | 1.052-1.058 |
| p8192 B64 | 1.074 | 1.078 | - | 1.074-1.078 |
| p128 B128 | 1.162 | 1.156 | 1.134 | 1.134-1.162 |
| p8192 B128 | 1.106 | 1.124 | - | 1.106-1.124 |

- The verdict is "TP2 stronger" in every cell of every job, with 0 of 4 rounds below 1.
- Second-job controls: TP2s / TP2 0.973-1.007, DP2s / DP2 0.988-1.018. Per-arm steps match the first job within
  1.5 % (TP2) and 1.4 % (DP2).
- The spread is the job, not the GPU pair: the verifier's job and 68225698 ran on the same pair and differ by 0.022
  at B128 p128.
- Read FlyTape DP2 / TP2 as about +-0.015 (cross-job spread up to 0.028). That is smaller than every cell's margin
  over 1 (>= 0.047).

**Issue 4: the reference mixed precision and metric.** Reworded in section 0, the section 5 table and
`BASELINE.md`. The column is now "fastest (cross-stack)": FlyTape TP2 (K0, random weights, W4A4) against ATOM DP2 K3
(served, real weights, W4A8, MTP). It names the bar the rocKE kernel faces, not a matched-precision winner, and it
uses both FlyTape jobs with a 3 % tie band.

With the fix-round ATOM data, 8k B32 and 8k B64 move from "ATOM DP2 K3" to "within" (FlyTape TP2 / ATOM DP2 K3,
first round -> fix round):

| cell | first round | fix round (both FlyTape jobs) |
|---|---|---|
| 8k B32 | 1.023 | 0.993-1.001 |
| 8k B64 | 1.028 | 1.003-1.008 |
| 1k B16 | 0.998 | 1.027-1.031 |
| 1k B32 | 1.004 | 0.969-0.977 |
| 1k B64 | 1.004 | 1.007-1.015 |

1k B16-B64 stay "within". ATOM DP2 K3 stays faster at 1k B8 and 8k B8-B16, and FlyTape TP2 stays faster at B128.

**Issue 5 (minor): FlyTape overlap minimum.** Reported in sections 1 and 6:
- Minimum 0.655. 3 of 160 samples are below 0.8, all in one config, with device steps 0.998 of the rest.
- The second job is the same: minimum 0.658, steps 0.996.

**Issue 6 (minor): "DP2 cannot reach TP2's B8 latency at all".** Now "below DP2's measured range (B4 per replica and
up)" in section 0, the iso-latency table and its note.

**Issue 7: 2 ATOM reps instead of the 6-pair e2e rule.** Fixed by measurement: the DESIGN.md 9.1 e2e rule on the
deciding arms.
- Arms: TP2 K3, its A-vs-A twin TP2s K3, DP2 K3, its twin DP2s K3 (servers on swapped GPUs), and TP2 K1 at B8 and
  B128. They ran one server set at a time, in a seeded shuffle per pair.
- Pairs: 6 per cell, plus 2 more at B8 to replace pairs that failed the overlap gate. 7 valid pairs at B8, 6
  elsewhere.
- **Result: DP2 is stronger in all 10 cells.** The median of best DP2 / best TP2 is 0.751-0.901, every valid pair is
  below 1, and every median is below its A-vs-A band minimum (0.889-0.981).
- The first round's 1k B16 tie becomes DP2: 0.839 (0.776-0.898) against a band of 0.893-1.035.
- The best TP2 arm over the pairs is K1 at B8 and K3 elsewhere; DP2's is K3 everywhere. This matches the first round.
- Served output tok/s DP2 / TP2 is 1.23-1.41x.

**Files changed** (harness `hackathon-prep/qwen38/truemono-lb/l0c-dp2/`; no rocKE library file changed; existing
inputs give byte-identical outputs where the new options are off):
- `summarize_dp2.py`: GSM8K score gate, mutant gate, TP2s / DP2s controls, `arm_cells.txt`, gate-failed samples
  excluded, `--pairs` (e2e WIN rule), and a rep-collision check in `--merge`.
- `atom_dp2.sh`: TP2s / DP2s arms, `ARM_CELLS`, `MUTANT` / `MUT_MODES`. With the defaults, the arms and cells run
  as before.
- `dp2_compare.py`: `--pairs`, and gate-failed samples left out. On the first round's inputs without `--pairs` the
  output is byte-identical.
- `clk_summary.py`: tolerates amd-smi "N/A" power values.
- `test_l0c_dp2_harness.py`: 9 -> 15 tests, all passing.
