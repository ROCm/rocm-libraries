# WP6: the wide path for C >= 8 (rocKE dense GEMMs, routing fold, seams, capture ladder)

W13, 2026-11-03 (fix round 2026-11-04). MI355X (gfx950), TP2 no-EP, DSpark 5, bf16 KV, fp8 index plane, `--level 0`
+ FULL graphs (the B0 recipe flags). Ratios only. Absolute numbers: `hackathon-prep/private-perf/dsv41-mono/WP6-abs.md`
and `raw/wp6/<tag>/`.

## Verdict

**No WP6 lever wins end to end. The D0 expectation (C8 1.18, C16 1.22) is not met.** Each lever is built, gated and
measured, and kept as a default-off knob recorded "rejected on base B0":

| knob | what | numerics gate | accuracy gate (GSM8K 1319 + passkey vs native twins) | per-call ratio vs ATOM (same job) | served C8 / C16 / C32 vs ATOM native (median of per-pass ratios, both nodes) |
|---|---|---|---|---|---|
| `ATOM_DSV41_RK_WIDE=<buckets>` | rocKE FP8 dense GEMM, MXFP8 quantization fused or as its own rocKE launch | **PASS** (tolerance): float64 MXFP8 reference, quantization bitwise vs ATOM, 100 % of mutants caught; in situ (real weights and activations, 112 K calls, eager) max rel. L2 vs ATOM's GEMM 2.7e-4 | **NOT MET as defined**: 1 of 3 forced-pick runs below the twin range on both metrics; pooled over runs no difference (§2.1) | **0.94-1.19 at 48 rows (below 1 only shared w13 0.94, wqkv_a 0.97, wq_b 0.99), 1.05-1.43 at 96, 1.42-1.84 at 192** (best build per shape) | forced picks: 1.013 / 0.974 / **0.961** (loss only at C32 is beyond noise); winners only: 1.001 / 0.971 / 0.991 (ties) |
| `ATOM_DSV41_RK_FOLD=<buckets>` | router GEMM + sqrtsoftplus top-6 in one rocKE launch | **PASS** (tolerance, gate v2, every served row count 48-192, both nodes): 0 set mismatches outside the bf16-rounding envelope, 0.45-0.59x ATOM's pooled mismatches; gate v1 had misreported a FAIL cell (§2) | **NOT MET as defined**: 1 of 3 runs (with LADDER + SEAM) and the 1 run alone below the twin range on one metric; pooled no difference (§2.1) | **2.2-2.3x slower** | 1.017 / 1.018 / 1.024 (inside the A-vs-A band) |
| `ATOM_DSV41_RK_SEAM=quant` | ATOM's `quantize_fp8` as one rocKE launch | **PASS bitwise**: 10 shapes x 5 data classes, both mutants caught | alone: inside (1 run); with LADDER: 1 run below on both metrics (§2.1) | **0.90-0.94** at decode widths, **0.68** at prefill widths | with LADDER: 1.001 / 0.967 / 0.988 (band) |
| `ATOM_DSV41_RK_LADDER=1` | exact capture rungs 10 / 12 / 14 | runtime only (graphs captured, 15 sizes) | alone: inside (1 run) | - | native, lad2 only: 0.955-0.995 / 0.995-1.001 (band) |
| A-vs-A (native twin vs native) | | | twins: flexible 0.9052-0.9166, strict 0.1213-0.1304 (9 runs) | | 0.991 / 1.001 / 1.022; single passes 0.911-1.080 |

(Served: B0 client, GSM8K prompts, 1024-token cap, per-request median decode tok/s, fresh server per arm-run, arm
order shuffled per pass, ratio against the native arm of the same pass. Fix-round jobs s2n01 (n01-09) and s2n07b
(n07-17): 4 passes each, C8 64 / C16 96 / C32 128 prompts; pooled here with the earlier jobs e1 (2 passes, n07-17)
and lad2 (2 passes, n07-17) where they ran the same arm: 8-12 passes per cell. Tests in §3.4.)

**3x:** not reached at any cell by WP6. WP6's measured contribution to the 3x composition is 1.00: no lever moves a
served cell beyond the A-vs-A noise in the winning direction. The WIDE GEMM with forced picks (every bucket, including
the 96- and 192-row builds that lose per call) costs about 4 % at C32 (median 0.961 over 10 passes on both nodes,
permutation p = 0.003 against the twins, the only cell of 15 that survives a Holm correction). Its C16 loss in the
first report (0.963 / 0.946, job e1) did not replicate: n07-17 gave 0.971-1.026 and the 10-pass median is 0.974
(p = 0.44). Serving only the per-call winners ties at every cell.

**Noise floor.** One pass of one arm against the native arm of the same pass swings by about ±9 % on these cells (the
native twin ranges 0.911-1.080 over 34 single-pass ratios; lad2 alone has 0.911 at C16 and b1 / b0 0.910 at C8). The
first report's "0.97-1.08 band" was too narrow. All "inside the band" statements in this report refer to the
10-12-pass medians and the permutation tests of §3.4, not to single passes.

**Both nodes.** No win is claimed. The fix-round served cells ran on both rocke nodes with the same harness, arms and
seeds (s2n01 on n01-09, s2n07b on n07-17); the only beyond-noise effect (WIDE forced at C32) shows on both (median
of per-pass ratios 0.974 on n01-09, 0.959 on n07-17). The microbenches ran on both nodes (n07-17: s1c, d1, sc2, r3, r5; n01-09: g2, q1,
r2, r4), and the FOLD gate v2 on both (s2n01, s2n07b).

## 1. What was built

rocKE (`copy-rocm-lib/.../rocke/library/kernels/gfx950/dsv41_mono/`, new files, `qwen38_mono` untouched):

- `wide_gemm.py`: `y = x W^T` for ATOM's native group32 FP8 linears at 16-384 rows. Reads ATOM's (16, 16)-preshuffled
  E4M3 weight directly (the lane mapping of ATOM's own mono K1 GEMV, `stage_wqkv.py`) and its 32x32 E8M0 scales
  (repacked once on the host so one dword holds a 4-chunk group and `op_sel` picks the byte), scaled
  `v_mfma_scale_f32_16x16x128_f8f6f4`, bf16 out. Knobs (all built, all in the screens): `waves` 1/2/4/8, `mt` 1/2/4,
  `splits` (every divisor), `depth`, `bra` (B reads: per MFMA, per chunk, a chunk ahead), `gch` / `pipe` (activation
  staging groups, two LDS slots), `xin` (`bf16`: MXFP8 quantization fused into the staging, ATOM's rule bit for bit;
  `fp8`: ATOM's MXFP8 input), `xsrc` (`lds` | `glb`: no LDS, B operands from an operand-order buffer written by a
  separate rocKE quantization launch, `build_quant_kernel`), `order` / `fold` (`xcd` + `l2`: every split of a row group
  on one XCD, partials exchanged through that XCD's L2 with plain stores; `sc1` + `sfast` the write-through form),
  `wfirst`, `occ1`, `stamp` (phase stamps). Mutants: `kskip`, `scale`, `fold`, `xq`, `tok`.
  `build_quant_atom_kernel`: ATOM's `quantize_fp8` in ATOM's layout (the seam).
- `router_topk.py`: router GEMM (`v_mfma_f32_16x16x32_bf16`, K split over CTAs, f32 partials, last-arriver fold in
  split order) + bf16 logit rounding + sqrtsoftplus + biased top-6 (ties to the lowest expert index) + renorm +
  `routed_scaling_factor`. Knobs `splits`, `fast` (DPP reductions), `vbias`, `stamp`. Mutants `tie`, `bias`, `drop`,
  `renorm`.
- Tests `library/tests/test_dsv41_mono_wide.py` (CPU, 10 pass): knob checks, names, grid, the host scale packing byte
  for byte, every mutant changes the IR, the l2 fold drops `sc1`.

ATOM (`ATOM-dsv41`, additive, every lever default off; unset = the module is never imported):

- `atom/models/deepseek_v41/rk/wide.py`: ctypes HIP-module launcher on the runtime torch mapped (graph-capturable),
  `install` (WIDE: patches `blockscale.native_quant_linear` for registered weights; picks per `(K, tiles, call, N)`
  from `index.json`; scales packed and split-K scratch allocated before any capture), `install_fold`, `install_seam`,
  `ladder_rungs`, check modes (`ATOM_DSV41_RK_WIDE_CHECK=1` install-time smoke, `=2` per-call in-situ comparison with
  ATOM's GEMM; `ATOM_DSV41_RK_SEAM_CHECK=1` per-call bitwise), mutant selectors.
- `atom/model_engine/model_runner.py`: three env-guarded install hooks after WP3's. `atom/config.py`
  `capture_sizes`: LADDER rungs added only when `ATOM_DSV41_RK_LADDER` is set.

Harness (`hackathon-prep/dsv41/wp6/`, new): `wp6_build.py` (host-venv compile, fork pool, resource scan, plans `smoke`,
`diag`, `screen2`, `glb`, router and seam builds), `wp6_bench.py` / `wp6_rbench.py` / `wp6_qbench.py` (container
benches + gates), `wp6_served.sh` (served, eager check, accuracy phases), `wp6_job.sh`, `wp6_submit.sh`, `mkstage.py`,
`up.sh`, `wp6_merge.py`, `wp6_picks.py`, `wp6_chk.py`, `wp6_summ.py`; fix round (§6): `wp6_rbench2.py`, `wp6_served2.sh`,
`wp6_job2.sh`, `wp6_submit2.sh`, `wp6_release.sh`, `wp6_fixsumm.py`, `wp6_accsumm.py`.

## 2. Gates

- **WIDE GEMM, tolerance class.** Per cell: float64 reference of the MXFP8 GEMM (ATOM's own `quantize_fp8` codes, the
  dequantized 32x32 weight). ATOM's own relative L2 error is 1.65-1.66e-3 at every cell; a build passes at <= 1.25x
  ATOM's + 1e-4. Every timed build passed. Quantization parity: every `xin=fp8` build fed ATOM's quantized activations
  is bitwise equal to its `xin=bf16` twin (fused quantization), at every cell, so the fused quantization reproduces
  ATOM's `quantize_fp8` bit for bit. Mutants `kskip`, `scale`, `fold`, `xq`, `tok`: caught in 100 % of cells (4,024
  mutant builds over the screens).
- **WIDE in situ** (job e1, `ATOM_DSV41_RK_WIDE_CHECK=2`, eager server at C8 / C16 / C32, every routed call recomputed
  by ATOM's GEMM on the same real activations and weights): 112 K calls over 320 (projection, rows) groups, rows
  48-192 and the ladder widths between, target and draft (MTP) layers; max relative L2 2.7e-4, max relative max-norm
  error 6.1e-3, no non-finite output (threshold 4e-3 relative L2); 166 of the 320 groups are bitwise equal to ATOM (the
  same K order). Covered projections: `wqkv_a`, `wo_b`, shared `gate_up_proj` and `w2`. `wq_b` and the indexer's
  `wq_b` never reach the WIDE path in situ: ATOM hands them pre-quantized activations (`x_scale`), and only the
  `xin=bf16` call is picked (the served log says "no pick ... xin=fp8: ATOM's GEMM" for them). This is the layer-level
  numerics gate at 48 / 96 / 192 rows the spec asks for, per projection rather than per layer.
- **FOLD, tolerance class: gate v2 (fix round).**
  - *Correction.* Gate v1 (job r4, `wp6_rbench.py`) drew one random input per tile count and allowed "reference set
    mismatches <= ATOM's + 1". At 80 rows (the tiles-5 builds) every one of the 6 non-mutant builds had 5 mismatches
    against ATOM's 2: a FAIL that the first report summarised as "mismatches <= ATOM's ... PASS". The served buckets
    (48 / 64 / 96 / 192) do not run the tiles-5 builds, but the 96-row bucket build serves 65-95 rows and v1 never ran
    it there. The gate was unproven for those rows.
  - *Gate v2* (`wp6_rbench2.py`, rule written into the script before it ran): the served pick and every other build of
    a bucket, at **every row count the bucket serves** (48; 49-64; 65-96; 97-192), over 8 seeds (job s2n01, n01-09) and
    16 seeds (job s2n07b, n07-17), fresh weights, bias and activations per (seed, rows). G1: every row whose expert set
    differs from the float64 routing must be explainable by bf16 rounding of the router logits (moving each logit by at
    most the envelope, the chosen set must be a valid top 6); 0 unexplained rows allowed. The envelope is validated on
    ATOM: ATOM's own mismatches must all be explained. G2: pooled mismatches <= ATOM's pooled + 3 sqrt(ATOM's) + 2.
    G3: weight error on agreeing rows <= 1.5x ATOM's + 1e-3. G4: the exact tie test (experts 7 / 300 identical, five
    experts forced ahead) at the bucket's smallest and largest served rows.
  - *Results.* Every non-mutant build of every bucket, served picks included, has **0 unexplained rows** on both nodes,
    **0.45-0.59x ATOM's pooled set mismatches** (n01-09: 9 / 212 / 570 / 2,968 against ATOM's 16 / 434 / 974 / 5,449;
    n07-17: 24 / 392 / 1,125 / 5,833 against 43 / 863 / 1,979 / 10,650 for the 48 / 64 / 96 / 192 buckets), weight
    error 2.1-2.6e-3 against ATOM's 3.8-6.5e-3, and the tie test clean. The n01-09 run used a 1 bf16-ulp envelope,
    which ATOM itself violates on 7 / 2 / 18 rows of the 64 / 96 / 192 buckets, so by the pre-registered rule that
    run reports "envelope INVALID" and no PASS. A too-tight envelope only makes G1 harder (G1 is monotone in the
    envelope), and the rocKE builds pass it with 0 unexplained rows. The n07-17 run then used a 2-ulp envelope (fixed
    before its seeds 8-15 were drawn; seeds 0-7 repeat n01-09's data), under which ATOM has 0 unexplained rows: envelope
    valid, **every served pick PASS, all 16 build mutants caught (`bias`, `drop`: G1 / G2, 234-197,646 unexplained
    rows; `renorm`: G3; `tie`: G4), both checker mutants caught in every bucket** (`swap`: one chosen expert replaced
    by the reference's 20th; `tail`: the last row's ids replaced by experts 0-5, a tile-tail bug).
  - *The v1 FAIL cell, re-run.* The tiles-5 builds at 80 rows over the same seeds: 16 vs ATOM's 27 mismatches (8 seeds)
    and 35 vs 68 (16 seeds), all explained. The v1 cell was one unlucky draw (5 near-tie rows), not a defect.
- **SEAM quant, bitwise** with ATOM's `quantize_fp8` on every shape and data class, 6 build variants; `xq` / `skip`
  mutants differ on every cell.
- **P4 moe_floor coverage gate**: not re-run (no MoE kernel was ported, §4).
- **Spec mutants "skipped expert, wrong fnext segment order"**: not applicable (no MoE port); "wrong top-k tie-break" is
  the FOLD `tie` mutant above.

### 2.1 Accuracy (jobs e1, s2n01, s2n07b, s3n01)

GSM8K 1319 (3-shot chat, `max_gen_toks` 4096, `num_concurrent` 16, so the decode widths are the WIDE / FOLD buckets)
and passkey 20 per length at 8K and 24K, fresh server per run. The first report had one run per arm and relabelled
two below-range results as noise. The fix round re-ran every arm on both nodes, added native twins (9 in total) and
added single-factor arms to separate the levers from LADDER + SEAM:

| arm | runs | flexible per run | strict per run |
|---|---|---|---|
| native twins (b0, b0b, b0c, b0d) | 9 | 0.9052-0.9166 (mean 0.9104, sd 0.0036) | 0.1213-0.1304 (mean 0.1269, sd 0.0026) |
| WIDE forced + LADDER + SEAM (w0) | 3 | 0.9060, **0.8992**, 0.9098 | 0.1251, **0.1183**, 0.1266 |
| WIDE winners only + LADDER + SEAM (w2) | 2 | 0.9166, 0.9128 | **0.1183**, *0.1365* |
| FOLD + LADDER + SEAM (f0) | 3 | 0.9098, 0.9121, 0.9098 | **0.1205**, 0.1266, 0.1274 |
| LADDER + SEAM, no WIDE / FOLD (s0) | 1 | **0.9037** | **0.1190** |
| WIDE forced alone (w3) | 1 | 0.9113 | 0.1243 |
| FOLD alone (f3) | 1 | **0.9045** | 0.1236 |
| SEAM alone (q0) | 1 | 0.9060 | 0.1274 |
| LADDER alone (l0) | 1 | 0.9083 | 0.1281 |

Bold: below the twin range; italic: above it. Passkey: 20 / 20 at both lengths in every run of every arm (22 runs).

- **The gate as the task defines it ("within the A-vs-A spread of the baseline") is not met** by WIDE (1 of 3 forced
  runs below on both metrics; 1 of 2 winners-only runs below on strict) or by FOLD (1 of 3 runs with LADDER + SEAM below
  on strict; the one FOLD-alone run below on flexible).
- It is not met by the control either: the LADDER + SEAM arm without WIDE or FOLD (s0), whose SEAM is bitwise with ATOM
  and whose LADDER only adds capture sizes, is below the range on both metrics, and two runs of the same arm (w2) land
  0.018 apart on strict, more than the whole 9-twin range (0.009). A per-run range from 9 twins is narrower than the
  run-to-run spread of these arms, so this rule cannot separate a lever from served noise here.
- Pooled over runs no lever differs from the twins: flexible mean difference -0.0054 (WIDE forced, Welch t -1.6),
  +0.0043 (winners, t +1.9), +0.0002 (FOLD, t +0.1); strict -0.0035 (t -1.3), +0.0005 (t +0.1), -0.0020 (t -0.9).
  The in-situ numerics (WIDE 2.7e-4 relative L2 against ATOM's own GEMM; FOLD closer to the float64 routing than ATOM)
  give no mechanism for an accuracy loss.
- Record: **accuracy gate not met as defined; not attributable to WIDE or FOLD numerics on the evidence; both stay
  default off.** A gate that can decide needs paired per-question scoring or more twins per job (not run: no lever here
  is a speed win worth that cost).

## 3. Timing

### 3.1 WIDE dense GEMM per call (best passing build / ATOM's `quantize_fp8` + group32 GEMM, same process)

| shape (N x K per rank) | 48 rows | 96 rows | 192 rows | ATOM's GEMM alone / ATOM |
|---|---|---|---|---|
| wqkv_a 1792 x 5120 | 0.969 | 1.196 | 1.518 | 0.74-0.79 |
| wq_b 16384 x 1280 | 0.990 | 1.054 | 1.416 | 0.77-0.82 |
| wo_b 5120 x 4096 | 1.027 | 1.107 | 1.650 | 0.81-0.85 |
| shared w13 2304 x 5120 | 0.936 | 1.109 | 1.446 | 0.78 |
| shared w2 5120 x 1152 | 1.108 | 1.251 | 1.837 | 0.69-0.76 |
| indexer wq_b 4096 x 1280 | 1.191 | 1.426 | 1.804 | 0.70-0.73 |

A-vs-A (ATOM twin graph) 0.969-1.019 in every cell. Four kernel generations were screened (jobs s1c, d1, sc2, g2:
about 3,600 builds; the best of each generation per cell is in the private file); the table is the best over all.

**Why it loses** (phase stamps, job d1; ISA): the kernel time is dominated by the activation side, not the weights.
- With the quantization fused, every CTA quantizes its K slice of X again: the redundancy is the number of row groups
  (14-256), so at 96 rows the staging phase alone was 0.15-0.71 (median 0.41) of the kernel's span over 128 stamped
  builds. The VALU work and the bf16 X reads grow with rows; the weight bytes do not.
- Without LDS (`glb`, X quantized once by a separate rocKE launch), every wave re-reads the B operands of all token
  tiles: B traffic per chunk is (token tiles / mt) times the weight traffic (12x at 192 rows, mt 1).
- ATOM's group32 GEMM (aiter's FlyDSL `bmm_a8w8_mxscale`) uses a 4-deep LDS ring filled by `buffer_load ... lds`, a
  96-token x 32-row tile and an LDS-staged epilogue: each X tile is read once per CTA and quantized once, before the
  GEMM. Matching that structure is the next step (§5), and the separate quantization launch it needs would cost the
  0.15-0.31 of ATOM's time that the fused quantization was meant to save.
- What did help: the XCD-local L2 fold (every split of a row group on one XCD, plain partials, agent atomic) took the
  fold (partial stores + arrival) to 0.66x of the write-through `sc1` form's time (stamp medians over 64 + 64 builds);
  pipelined staging (two LDS slots, sc2) brought the best 48-row cells from 1.27-2.57 (first generation) to
  0.94-1.19.

### 3.2 FOLD (router + top-6) per call

| rows | 48 | 64 | 96 | 192 |
|---|---|---|---|---|
| rocKE / ATOM (gate GEMM + `topk_gating`) | 2.21 | 2.27 | 2.23 | 2.30 |

Phase stamps (job r5, `splits 16`, 96 rows): MFMA phase ends at about 0.37 of the kernel, partial stores + arrival
0.12, the last arriver's fold + scoring 0.30, the top-6 0.21. Every phase is latency bound (one CU per split streams
245 KB; the fold re-reads 16 partials per element with `sc1`; six dependent wave reductions per token). The DPP
reductions (`fast`) gained only 0.93x over the LDS ones. A version that wins needs the expert-sliced form (each CTA owns
16 experts over all of K and scores them, so the tail is a gather and the top-k only): estimated at about 0.7x of ATOM,
i.e. about 0.003 of the C16 step. Not built.

### 3.3 SEAM quant per call

| rows x K | 6x5120 | 48x5120 | 96x5120 | 192x5120 | 96x4096 | 96x1152 | 96x1280 | 96x2304 | 2048x5120 | 16384x5120 |
|---|---|---|---|---|---|---|---|---|---|---|
| rocKE / ATOM | 0.929 | 0.927 | 0.938 | 0.902 | 0.937 | 0.929 | 0.930 | 0.919 | 0.684 | 0.681 |

One block per 256 groups, one group per thread is the best variant at decode widths (2 or 4 groups per thread lose
1.05-2.0x). The quantization launches are about 0.04 of the C16 decode step (P4 trace), so 0.93x per launch is about
0.003 of the step: inside the served noise.

### 3.4 Served (jobs s2n01, s2n07b; earlier e1, lad2)

Every ratio is arm / native (b0) of the same pass. Fix-round jobs: 7 arms, 4 passes, arm order shuffled per pass
(seeded, so b0 and its twin are no longer adjacent), C8 64 / C16 96 / C32 128 prompts (8 / 6 / 4 waves), one job
per rocke node. Below: the median of all per-pass ratios with their range, and a two-sided permutation test of the
mean log ratio against the native twin's per-pass ratios of the same jobs (`wp6_fixsumm.py`, 20,000 relabelings).
With 5 arms x 3 cells = 15 tests, a Holm correction leaves only the smallest p (C32 WIDE forced, 0.003 x 15 = 0.045).

| arm | C8 | C16 | C32 (info) |
|---|---|---|---|
| native twin (A-vs-A) | 0.991 (0.953-1.018, n 12) | 1.001 (0.911-1.066, n 12) | 1.022 (0.963-1.080, n 10) |
| WIDE forced + LADDER + SEAM | 1.013 (0.970-1.043, n 10), p 0.03 | 0.974 (0.930-1.026, n 10), p 0.44 | **0.961 (0.915-1.011, n 10), p 0.003** |
| WIDE winners only + LADDER + SEAM | 1.001 (0.915-1.049, n 8), p 0.42 | 0.971 (0.937-1.067, n 8), p 0.66 | 0.991 (0.952-1.019, n 8), p 0.07 |
| FOLD + LADDER + SEAM | 1.017 (0.969-1.075, n 10), p 0.02 | 1.018 (0.949-1.050, n 10), p 0.32 | 1.024 (0.984-1.115, n 10), p 0.66 |
| LADDER + SEAM | 1.001 (0.953-1.035, n 10), p 0.29 | 0.967 (0.893-1.029, n 10), p 0.25 | 0.988 (0.778-1.085, n 10), p 0.33 |
| mono (ATOM's best at C1-C8) | 0.999 (0.886-1.074, n 12), p 0.91 | 0.964 (0.865-1.048, n 12), p 0.12 | 0.981 (0.933-1.164, n 10), p 0.53 |
| LADDER only, native (lad2) | 0.955, 0.995 | 0.995, 1.001 | - |

- **WIDE forced picks lose about 4 % at C32** on both nodes (n01-09 passes 0.935-1.011, n07-17 0.915-0.985), the
  one effect that survives the correction. At C16 the e1 loss (0.963, 0.946) did not replicate on n07-17 (0.971-1.026).
- **WIDE winners only** (the per-call winners: shared `gate_up_proj` 0.94, `wqkv_a` 0.97 at 48 rows, via
  `ATOM_DSV41_RK_WIDE_SCOPE`; `wq_b` 0.99 never engages because ATOM calls it with pre-quantized input) **ties at C8**
  (1.001; the WIDE path engages at exactly 48 rows, i.e. C8). Expected: the two winning projections are about a third of the dense-GEMM time (0.07-0.09
  of the step), so their 3-6 % per-call gains are worth under 0.002 of the step.
- **FOLD and WIDE forced at C8** sit at 1.013-1.017 with uncorrected p 0.02-0.03. After the correction this is not a
  claim, and a 2.2x slower router launch cannot speed the step up; it reads as the twin's median (0.991) being low.
- At C16 and C32 the mono arm serves natively (rows > 48), so mono / native there is in effect a second A-vs-A pair; its
  spread agrees with the twin's.

## 4. Not built, and why

- **Large-batch MoE port (`moe_lb2`)**: P4 measured ATOM's routed MoE at 1.13-1.31x of the rocKE bytes floor and the
  Qwen `moe_lb2` builds at 0.72-0.78 of floor speed, so a straight port is parity at best on the byte stream (D0 books
  no MoE floor). The "fold" half (top-k, sort and quantization folded into the MoE) was attacked from the cheap end
  first (router + top-k, §3.2) and loses 2.2x as one launch; folding aiter's two-phase `moe_sorting` and
  `fused_mx_quant_moe_sort` would mean reproducing aiter's sorted-id / padded-block / shuffled-scale layouts exactly for
  its FlyDSL `mfma_moe1` / `mfma_moe2`. The C16 trace (P4) puts top-k + sort + quant-sort + router at about 0.08 of the
  decode step; a fold that halves it is worth about 1.04. Open.
- **`fnext` combine, touched-expert streaming**: belong to the MoE port.
- **Width-specialised mono builds chosen on the device, the draft's MAX_REQUESTS lift**: ATOM's mono caps at 48 rows
  because K1 / K2 spill VGPRs and the shared expert outgrows one unit a CTA past it (`mono/config.py`); lifting it is a
  FlyDSL mono rework, and WP2 found no device-side build choice is needed (the host graph key already selects the
  build). At C16 the native path is within the A-vs-A band of mono at C8 (P1, B0), so a wider mono would have to beat
  native at 96 rows, where P1 measured mono attention 1.8x native and K1 2-3x. Not attempted.
- **Fused mHC / norm / RoPE seams**: only the quantization seam was built. mHC + norm (aiter Triton, 0.08 of the C16
  step) and RoPE / quant (0.07) are other teams' fused kernels; replacing them needs rocKE versions of the Sinkhorn mHC.

## 5. Lookback record (rejected on base B0, all default off)

| lever | status | re-test on the integrated base when |
|---|---|---|
| WIDE (every build family, every knob) | numerics PASS; accuracy gate not met as defined (§2.1); loses at 96 / 192 rows per call and about 4 % served at C32 with forced picks; winners-only ties | an LDS-ring (DMA) form with the X tile read once per CTA exists; or when the producers emit MXFP8 (then the `xin=fp8` builds compete with aiter's GEMM alone, 0.69-0.85 of ATOM's call) |
| FOLD router + top-6 | numerics PASS (gate v2, every served row, both nodes); accuracy gate not met as defined; 2.2x slower per call; served inside the band | the expert-sliced form; or folded with the sorting |
| SEAM quant | bitwise, 0.90-0.94 per launch (0.68 at prefill) | keep for the integrated stack (prefill TTFT); no served effect at decode |
| LADDER 10 / 12 / 14 | ties | intermediate concurrencies (C9-C15), and with WP1's admission policies that leave the batch below 16 longer |

## 6. Jobs and artefacts

- Jobs (all `rk-dsv41-wp6-*`, rocke partition): microbench s1c 68259583, d1 68259636, sc2 68259750, g2 68260094,
  q1 68260432, r2 68260553, r3 68260654, r4 68260700, r5 68260848; served lad2 68260326; served + in-situ check +
  accuracy e1 68260978. aiter source extraction (CPU) asrc 68259639, acsrc 68259902.
- Fix round: s2n01 68261708 (n01-09: FOLD gate v2 8 seeds, served 7 arms x 4 passes, accuracy 6 arms), s2n07b
  68262064 (n07-17: FOLD gate v2 16 seeds, the same served and accuracy arms), s3n01 68263037 (accuracy of the
  single-factor arms; it ran on n07-17 after its node pin was lifted while pending). 68261709 was cancelled before it
  started (replaced by s2n07b with the 2-ulp envelope). Same ATOM / rocKE stages and code objects as e1
  (`atom-wp6-a6`, `rk-wp6-a10`, hsaco of sc2 + g2 + r4, index `data/e2e_forced.json`); harness stage `h-wp6-a15`.
- Results: `~/rocke-sdpa-claude/dsv41/results/rk-dsv41-wp6-<tag>` (Austin home); copies with absolutes in
  `hackathon-prep/private-perf/dsv41-mono/raw/wp6/<tag>`; fix-round summaries `raw/wp6/fix_served_all.json`,
  `fix_served_new.json`, `fix_acc_summ.json`.
- Served index of the e1 and fix-round runs: `hackathon-prep/dsv41/wp6/data/e2e_forced.json` (forced picks,
  `wp6_merge.py --thresh 99`); the winners-only arm uses the same index with `ATOM_DSV41_RK_WIDE=48` and
  `ATOM_DSV41_RK_WIDE_SCOPE=attn.wqkv_a:shared_experts.gate_up_proj:attn.wq_b`.
- Fix-round harness (new files, `hackathon-prep/dsv41/wp6/`): `wp6_rbench2.py` (FOLD gate v2), `wp6_served2.sh`
  (shuffled arm order, WIDE scope field), `wp6_job2.sh`, `wp6_submit2.sh` (held submit), `wp6_release.sh` (login-node
  release under the W13 GPU cap), `wp6_fixsumm.py` (served tests), `wp6_accsumm.py` (accuracy table).
