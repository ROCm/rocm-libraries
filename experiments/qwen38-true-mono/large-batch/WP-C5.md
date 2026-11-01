# WP-C5: in-situ GEMM + all-reduce in one persistent launch (TP2), gfx950

W6, 2026-10-31. `../DESIGN-CCL.md` 6.3 / 8 (WP-C5), `../DESIGN-LB.md` (TP2 budgets). The large-batch dense GEMM of P-L3
(`large_batch/gemm.py`, its mx plan of record for the out-proj TP-rank shape) and the TP2 all-reduce of its bf16 partials
inside ONE persistent launch: the GEMM epilogue (the split-K fold's last arriver) pushes the partial, a consumer polls and
sums, no extra phase. Compared with the same GEMM followed by aiter's custom all-reduce (ATOM's TP path) or RCCL as
separate kernels in a HIP graph, and with FlyTape's exchange cost from its TP2 phase traces. Ratios only here; absolute
times are in `hackathon-prep/private-perf/qwen38-true-mono/large-batch/WP-C5.md` (raw JSON `wpc5-raw/`). Full tables:
`WP-C5-tables.md`.

> **Read the Fix round at the end of this file first. It supersedes parts of sections 1-8.**
> - **Status: partial; the DESIGN-CCL gate is not met** (`gate_met` false).
> - **The MoE-down exchange was measured in situ.** With its default arena layout it costs 0.84-4.3x the out-proj
>   exchange (1.3-8.3x with the first layout). Section 5's assumption that both exchanges cost the same is wrong.
> - **Section 5's shares are replaced by F5.** The share claim survives, with a smaller margin: the 96 measured
>   exchanges cost 0.048-0.061 of FlyTape's TP2 step, below FlyTape's own (0.060-0.082 skew floor, 0.078-0.107 old
>   proxy).
> - **End to end, the MoE + exchange in one launch still loses to MoE + aiter at 8-32 rows.**
> - **TP4 ran once.** Every in-situ value loses to GEMM + aiter at 32-128 rows there.
> - **Defaults are now chosen at TP2 for rows 5-128,** against `form=line` x `publish`, `ar=twoshot` and FlyTape's
>   `cons=phase`: lamport one-shot with the consumer pass.
> - **A correctness hazard was found in some of the builds below and fixed** (`hz`, Fix round F1): unpadded
>   inline-asm VMEM right after the compiler restored an SGPR base. The default `lam` builds were clean, so no
>   default-arm verdict changes.

## Gate and verdict

- **Numerics gate: met.** Bitwise on every rank against the host rank-order sum, over every layer and three launches
  without a reset; ranks bitwise equal; every stale-read twin and GEMM mutant fails; numerics ran before every timing,
  on the timed binaries (section 4).
- **Overhead: measured and reported** (section 5). At rows 16-128, the default in-situ arm's exposed exchange cost is
  0.24-0.63 of aiter's (in its separate-kernel graph) and 0.14-0.45 of RCCL's best protocol.
  - End to end per layer, the default in-situ arm beats GEMM + aiter at rows 4-128 on both nodes. The ratios are
    0.94-0.95 (rows 4), 0.84 (16), 0.87 (64) and 0.89 (128); every round is below 1.
  - It is not a win at rows 1: 1.021 LOSS on n01-09 and 1.008 TIE on n07-17.
  - It beats GEMM + RCCL everywhere, at 0.64-0.72.
- **Share of the FlyTape TP2 step: 0.018-0.064.** This is the cost of the 96 exchanges of one decode step (2 per layer
  x 48 layers), using the in-situ arm's exposed cost per exchange. FlyTape's own exchange share is 0.044-0.089.
  - Per layer (2 exchanges), the overhead is 0.0004-0.0013 of the FlyTape TP2 step.
- **DESIGN-CCL's WP-C5 gate: not met.** It asks for an exposed cost at most 0.25 of FlyTape's per-layer exchange at TP2
  B128.
  - Measured: 0.71 at B128 (B128 uses an extrapolated TP1 term, section 6). The other batches: 0.26-0.27 at B16, 0.53
    at B64, 0.40-0.44 at B4, 0.97 at B1.
  - From 16 rows up the exposed cost is 0.89-1.0 of the time the bf16 partial needs to cross the one xGMI link between
    the two GPUs (section 7). On this tile schedule only fewer bytes or overlap with the next layer can go lower.
- **TP4: not run.** It is built and CPU-tested at 4 ranks. The rocke nodes never had 4 idle GPUs within the 8-GPU budget
  during this WP; the q38 budget held at 7-9 of our GPUs.

Defaults chosen from the measurements (DESIGN 9.1 win rule; levers in section 6):
- **form=lamport** at every measured size, 5-640 KiB per rank. CCL-LM's standalone selection takes pull at 640 KiB. In
  situ, pull loses at every size (1.02-1.07).
- **cons=pass:** a consumer pass after the layer's last unit.
- **gb=gemm:** the GEMM's own grid barrier.
- **pause=s_nop.**
- Every other value stays a registered default-off knob.

## 1. What was built

| piece | path | notes |
|---|---|---|
| in-situ kernel | `rocke/library/kernels/gfx950/qwen38_mono/large_batch/insitu.py` | `InsituKnobs` (gemm plan, ranks, m, `form` lamport / pull / none, `cons` pass / fold, `gb` gemm / lite, `pause`, `mutant` noreset / slot, tag), `build_insitu_kernel`, `arena_map`, `poll_lamport_dw`; subclasses `gemm._Gemm` (gemm.py unchanged: sha256 39e27905..., the P-L3 fix round's measured file) |
| CPU tests | `rocke/library/tests/test_qwen38_lb_insitu.py` (51) | every form x cons x gb x mutant x plan (TP2 / TP4 shapes, splits 1 and 6) builds, lowers and passes the CCL peer-address lint; names unique; refusals; arena regions disjoint; the noreset twin drops exactly the resets; building in-situ kernels leaves the plain GEMM's IR unchanged |
| harness | `hackathon-prep/qwen38/truemono-lb/wp-c5/` | `wpc5_bench.py` (torchrun bench: numerics, twins, graphs, shuffled timing), `job.sh` (container: GPU pick + health check, bench per rows, FlyTape traces), `wpc5_report.py`, `lever_table.py`, `compile_diag.py` (compile-only COMGR check, no GPU), `sync_wpc5.sh` |

How the kernel works (module docstring has the protocol and the WAR argument):
- **Producer = the GEMM epilogue.** The unit that completes an output block (the split-K fold's last arriver; the unit
  itself at `splits=1`) rounds exactly as the plain GEMM does. Then, with no extra phase:
  - `lamport`: it pushes the bit complement of the canonicalised bf16 partial into every rank's slot with 8-byte
    system-scope stores through 64-bit peer bases. No drain, no flag.
  - `pull`: it stages the partial in its own arena, drains, and publishes a per-block flag on every rank.
- **Consumer (`cons=pass`).** After its last unit of the layer and before the layer's grid barrier, every wave of the
  grid, including CTAs without GEMM units, consumes 1 KiB slices of the exchange:
  - `lamport`: poll the own arena until no bf16 element is empty (every element checked), then reset the slot;
  - `pull`: poll the block flags, then read every rank's slice with system-scope loads.
  - Both add the ranks' bf16 partials in f32 in rank order and store the bf16 sum.
- **`cons=fold` (lever).** The fold's last arriver consumes its own block right after pushing it. It keeps its own
  partial in registers and polls the other ranks only, so a block's exchange overlaps the other CTAs' GEMM work. This is
  deadlock-free only while every CTA runs at most one unit per layer (`check` refuses otherwise).
- **Slots and tags.** Two slots and tags = `epoch (+) layer`, held in the arena control words and advanced by the last
  CTA. That makes the kernel graph-replay safe, with no host reset. Two slots suffice because a push of use u + 2 on
  rank P follows P's grid barrier of u + 1, which follows every consumer of use u on every rank.
- **Polls.** Every poll has a deadline and an abort word. A failed poll records the error in `ctrl` and pushes ABORT to
  every rank.

## 2. Method

- **Shape.** Out-proj at TP2 per rank: 2560 x 3072 MXFP4 weights x MXFP8 activations (W4A8, ATOM's numerics class).
  - Rows 1, 4, 16, 64, 128. Buckets 16 / 16 / 16 / 64 / 128.
  - P-L3's fix-round plan of record: 8 waves, 6 K splits, op_sel scales, tile layout. `noxu` at rows 1-8 / 32 / 64,
    `xa2` at 16 / 128, 4-tile token blocks at 128.
  - The MoE-down all-reduce moves the same rows x 2560 bf16 payload. Its GEMM is the expert-major MoE of P-L4, so it was
    not run in situ; its exchange cost is taken to equal the out-proj one (an assumption).
- **Layers.** L = 258 per launch (weights rotate over 4x MALL, P-L3's rule). Each rank has its own weights and
  activations.
- **Arms.** All are captured HIP graphs; times are per layer = graph time / L.
  - `gemm_ph`: the plain GEMM, one launch, P-L3's grid barrier between layers. This is the in-situ arms' floor: same tile
    schedule, same barrier.
  - `gemm_ln`: the plain GEMM, one launch per layer. This is the separate-kernel arms' floor.
  - In-situ arms: `lam`, `pull`, `lamF`, `pullF`, `lamL`, `pullL` (floor `gemmL` = the exchange-free twin with the lite
    barrier), and `lamS`.
  - `aiter`: per layer, GEMM launch + aiter `custom_all_reduce` of its Y (registered-input capture, ATOM's path).
  - `rccl_LL` / `rccl_LL128` / `rccl_Simple`: per layer, GEMM launch + `torch.distributed.all_reduce`, one communicator
    per protocol. "RCCL best" is the fastest protocol per size.
- **Exposed cost.** Arm minus its floor, per round.
- **Timing discipline (DESIGN 9.1).**
  - Every arm runs in one process per rank, with a gloo barrier before every replay. The order is shuffled every round
    with a fresh seed (recorded). 24 rounds.
  - The per-round MIN over the two ranks is kept: a later-launched rank does not wait for the launch skew. Rank-max is
    kept too and is within 0.3 % of it.
  - Controls: a second graph of `gemm_ph`, a recompiled twin of every build (`#rc`), and a second aiter capture. The
    win rule is applied against both arms' controls.
  - The binaries are unstamped. Node and bus ids are recorded per row, and amd-smi clock / power was sampled every 5 s
    for the whole job (`clocks.log`, private).
- **Runs.**
  - Final code on both rocke nodes: job 68234979 (n07-17, buses 75 / 15) and job 68235021 (n01-09, buses 65 / e5).
  - Earlier-code runs agree within 0.02 on every default-arm ratio: 68234271 / 68234546 (n07-17) and 68234926 (n01-09).
    Their in-situ builds computed the error-record tag inside a one-lane region; the hot path is unchanged.
- **FlyTape's exchange cost.** FlyTape has no exchange phase of its own: its waits and rank-ordered sums land in the HC
  mixer phases.
  - It is measured as the TP2 - TP1 difference of the replicated phases per traced step: HC mix down / combine / reduce
    / up, moe_combine and ple_reduce. These do identical work per rank at TP1 and TP2.
  - B8-B128 come from P-L3's traces (job 68231605). B1 / B4 come from this WP's traces (job 68234271); its TP2 B8 step
    agreed with 68231605's within 0.2 %.
  - TP1 B128 does not run (FlyTape's i32 index limit), so its replicated phases are extrapolated from B32 / B64. Treat
    the B128 column as an estimate.

## 3. Numerics (hardware, before every timing, on the timed binaries)

| check | result (final runs, both nodes, 5 rows each) |
|---|---|
| plain GEMM N1 vs float64 reference (layers 0 and L-1, P-L3's 2^-11 bound) | 0.01-0.04 of the bound; recompiled twin bitwise |
| every in-situ arm and its `#rc` twin (12 builds) x 3 launches (activation sets A, B, A; no reset) x 258 layers | bitwise vs the host rank-order f32 sum of every rank's bf16 Y on both ranks; error records clean; GEMM grid-barrier timeouts 0 |
| exchange-free twin (`gemmL`, `#rc`) | its Y bitwise equal to the plain GEMM's on every rank |
| each in-situ graph replayed once after capture | bitwise |
| must-fail twins: stale-read `noreset` (lamport pass / fold / pause / lite) and `slot` (pull pass / fold / lite), GEMM `kskip` inside lam / pull / lamF / pullF | 110 of 110 detected (lamport `noreset`: 256 of 258 layers wrong on the pass arms, i.e. every layer after the first slot reuse); 0 missed |
| separate-kernel arms (aiter x 2 captures, RCCL x 3 protocols), last layer's all-reduced Y | bitwise equal to the same host sum (share 1.0000) |

Grand totals over the final runs: 380 PASS lines, 0 FAIL, 0 dropped arms, 0 compile failures.

## 4. Results (final runs; per-round min over ranks; medians of 24 rounds; n01-09 / n07-17)

Default arm `lam` (lamport, consumer pass):

| rows (KiB / rank) | lam / gemm_ph | aiter / gemm_ln | lam / GEMM + aiter (9.1) | lam / GEMM + RCCL best (9.1) | exposed lam / exposed aiter | exposed lam / exposed RCCL best |
|---|---|---|---|---|---|---|
| 1 (5) | 1.160 / 1.160 | 1.464 / 1.469 | 1.021 LOSS / 1.008 TIE | 0.652 / 0.645 WIN | 0.444 / 0.437 | 0.160 / 0.158 |
| 4 (20) | 1.165 / 1.151 | 1.549 / 1.542 | 0.948 / 0.938 WIN | 0.693 / 0.678 WIN | 0.379 / 0.351 | 0.186 / 0.167 |
| 16 (80) | 1.143 / 1.136 | 1.696 / 1.696 | 0.837 / 0.837 WIN | 0.644 / 0.651 WIN | 0.255 / 0.244 | 0.147 / 0.144 |
| 64 (320) | 1.390 / 1.384 | 1.925 / 1.915 | 0.871 / 0.869 WIN | 0.648 / 0.646 WIN | 0.509 / 0.505 | 0.297 / 0.293 |
| 128 (640) | 1.670 / 1.664 | 2.285 / 2.275 | 0.888 / 0.889 WIN | 0.721 / 0.724 WIN | 0.634 / 0.634 | 0.449 / 0.450 |

The WINs above have every round below 1. Controls per cell were 0.98-1.02 (`WP-C5-tables.md`). The earlier-code runs
give the same verdicts:
- `lam` vs GEMM + aiter: 1.021 LOSS / 1.003 TIE at rows 1, then 0.942-0.965, 0.842, 0.860-0.862 and 0.895;
- `lam` vs GEMM + RCCL best: 0.63-0.73.

Why rows 1 is a tie at best:
- **The barrier, not the exchange.** The phase barrier between layers costs more than a graph's kernel boundary:
  `gemm_ph / gemm_ln` = 1.20-1.29.
  - At 1 row the in-situ exchange is cheaper than aiter's (exposed 0.44).
  - But the GEMM floor it sits on is 1.28x the separate kernels' floor.
- **Not an argument for separate kernels.** A layer kernel without a phase barrier (DESIGN-LB's tile-flag dataflow) is
  the fix.
- **Not addressed in this WP.** The lite barrier lever (below) did not cut the barrier.

## 5. In-situ exchange overhead against FlyTape's (TP2)

| batch | exposed lam / FlyTape's exchange (per exchange) | 96 in-situ exchanges / FlyTape TP2 step | per layer (2 exchanges) / FlyTape TP2 step | FlyTape's own exchanges / its TP2 step | GEMM + aiter's 96 exchanges / FlyTape TP2 step |
|---|---|---|---|---|---|
| 1 | 0.965 / 0.966 | 0.042 | 0.0009 | 0.044 | 0.095-0.096 |
| 4 | 0.438 / 0.400 | 0.023-0.025 | 0.0005 | 0.058 | 0.066-0.067 |
| 16 | 0.270 / 0.258 | 0.018-0.019 | 0.0004 | 0.071 | 0.075 |
| 64 | 0.537 / 0.530 | 0.041 | 0.0009 | 0.076 | 0.080-0.081 |
| 128 (TP1 term extrapolated) | 0.717 / 0.713 | 0.064 | 0.0013 | 0.089 | 0.100-0.101 |

So at every batch the in-situ exchange costs less of a FlyTape TP2 step than FlyTape's own exchange.
- The saving is 0.03-0.05 of the step at B4-B64 and 0.025 at B128. B1 is a tie.
- GEMM + aiter as separate kernels costs more than FlyTape's own exchange at B1-B128.
- The ratios are cross-job. FlyTape runs its whole step (random weights, no MTP), while the in-situ number comes from the
  out-proj microbenchmark.

## 6. Levers (default-off knobs; final runs, both nodes; ratio to the default arm, 9.1 verdict)

| lever | rows 1 | rows 4 | rows 16 | rows 64 | rows 128 | verdict |
|---|---|---|---|---|---|---|
| `pause=s_sleep1` (`lamS`) | 1.000 TIE | 1.002-1.003 TIE | 1.000 TIE | 1.000 TIE | 1.000 TIE | noise |
| `cons=fold`, lamport (`lamF`) | 1.086-1.092 LOSS | 1.095-1.100 LOSS | 1.137-1.145 LOSS | 1.044-1.048 LOSS | 0.995 TIE | loss (ties at 128) |
| `form=pull` (`pull`) | 1.018-1.042 LOSS | 1.033-1.035 LOSS | 1.068-1.071 LOSS | 1.056-1.066 LOSS | 1.045-1.059 LOSS | loss everywhere |
| `form=pull`, `cons=fold` (`pullF`) | 1.25-1.29 LOSS | 1.24 LOSS | 1.17 LOSS | 1.08-1.09 LOSS | 1.05-1.07 LOSS | loss |
| `gb=lite`, lamport (`lamL`) | 1.032-1.035 LOSS | 1.018-1.043 LOSS | 1.030-1.034 LOSS | 1.001-1.006 TIE | 1.010-1.013 TIE / LOSS | loss / tie |
| `gb=lite`, pull (`pullL`) | 1.05-1.07 LOSS | 1.05-1.07 LOSS | 1.07 LOSS | 1.04-1.06 LOSS | 1.05-1.06 LOSS | loss |
| `gb=lite` floor (`gemmL / gemm_ph`) | 1.049-1.074 LOSS | 1.061-1.082 LOSS | 0.990-0.995 TIE | 1.027-1.030 LOSS | 1.024-1.025 LOSS | loss |

- **`cons=fold` loses even though its exposed cost could overlap.** With one unit per CTA (120-240 units on 256 CTAs),
  every block finishes at the same moment, so nothing is left to overlap. The fold's synchronous poll then sits on the
  critical path. It ties only at 128 rows, the link-bound size.
- **`gb=lite` loses.** The lite barrier swaps the returning atomic for a target taken from the arena's launch count. Its
  exposed cost is lower at 1-4 rows (the lam - gemmL difference), but its floor is slower by about the same amount.
- **Pull loses in situ.** A pull pays a drain plus a flag before the consumer can read, and the lamport payload poll
  pays neither. CCL-LM's standalone pull win (640 KiB, a capped grid) came from link utilisation that this layout does
  not need.

## 7. Why the exposed cost is where it is

- **Link-bound from 16 rows.** Exposed `lam` divided by the payload's crossing time over the one xGMI link is 0.96-1.01
  (16 rows), 0.89-0.91 (64) and 0.90 (128). The link rate is CCL-LM's bidirectional push probe.
  - All 120-240 units of a layer finish within one unit's time, so the whole payload crosses after the GEMM's last fold.
  - The in-situ push already overlaps the folds and skips the standalone all-reduce's own grid sync. Against CCL-LM's
    standalone layer-shaped all-reduce at the same sizes (other jobs, indicative) the in-situ exposed cost is 0.36-0.37
    (16 rows), 0.60 (64) and 0.69-0.72 (128).
- **Latency-bound below 16 rows.** The cost is a fixed push-to-poll round trip of about one xGMI round trip.
  - It barely changes from 1 to 16 rows: rows 16 / rows 1 = 0.95-0.99.
  - aiter's separate kernel pays 2.3-2.9x it.
- **What could beat 0.25 of FlyTape's per-layer exchange at B128.** On this schedule the link floor alone is about 0.79
  of FlyTape's per-exchange cost, so this needs one of:
  - fewer bytes: the `ar_dtype=fp8` partials of DESIGN-CCL 4.4, about 0.53x the bytes; a tolerance class, not built;
  - overlap of the exchange with the next layer: DESIGN-LB's tile-flag dataflow, where the next layer's units start on
    the blocks whose exchange landed. That is WP-L3 / L5 work, not a GEMM-epilogue knob.

## 8. Caveats and what was not done

- **TP4: not run** (GPUs, see the gate). The TP4 plan (K 1536, 4 waves, 3 splits) is built, linted and lowered on the
  CPU only.
- **MoE-down GEMM: not run in situ.** Its payload equals the out-proj one; its GEMM is P-L4's expert-major kernel.
- **PLAN decision 5 (DP2 arm): does not apply.** DP2 has no all-reduce; this is a TP2 kernel-level comparison.
- **FlyTape B128: an estimate.** Its exchange cost needs TP1 phases that FlyTape cannot run at B128, so they are
  extrapolated. FlyTape's numbers come from other jobs.
- **Exposed costs are per microbenchmark layer.** In a model the consumer would be the next stage's prologue, not a
  store of the sum.
- **First jobs lost to harness bugs.** In job 68234271, rows 1 / 4 / 16 / 128 died on an NCCL int16 gather. Jobs
  68234869 / 68234879 skipped timing because a rank-agreement check is invalid for the exchange-free twin. The
  `cons=fold` lamport builds at buckets 64 / 128 did not compile until the tag computation moved out of a one-lane region
  (`s_add_u32` with a VGPR operand). All are fixed; none of these runs is used for a verdict.

## 9. Rerun

From `hackathon-prep` (Git Bash), `Q38_MAX_GPUS=8`:

```bash
bash qwen38/truemono-lb/wp-c5/sync_wpc5.sh both          # stages w6-wpc5-h, w6-wpc5-rocke
Q38_NODE=smci355-ccs-aus-n01-09 bash qwen38/truemono-lb/submit.sh logs/q38/w6/wpc5-g.poll --mode container --gpus 2 \
  --cpus 32 --mem 400 --time 01:00:00 --tag w6-wpc5-g --stage w6-wpc5-rocke --stage w6-wpc5-h \
  --env "WPC5_ARGS=--aiter+--rounds+24" -- 'bash $Q38_STAGES/w6-wpc5-h/job.sh pick:2 bench:2:16,128,4,64,1'
# FlyTape traces at B1 / B4 / B8: add --stage flytape --stage w3-lbh and the steps
#   ft:1:ft1:mxfp4:1x128,4x128,8x128:1:trace ft:2:ft2:mxfp4:1x128,4x128,8x128:1:trace
python qwen38/truemono-lb/wp-c5/wpc5_report.py --runs <f> <g> --ft-old <P-L3 D>/ft-pre1/ft/tp1-trace.jsonl \
  <P-L3 D>/ft-pre2/ft/tp2-trace.jsonl --ft-new <a>/ft-tp1/ft/ft1.jsonl <a>/ft-tp2/ft/ft2.jsonl --out-ratio R.md --out-abs A.md
```

A 5-row job (26 builds per row, numerics, twins, about 20 graphs x 24 rounds per row) runs in about 3 min on the node.

## Verification (independent verifier, W6, 2026-10-31)

**Verdict: partial.** The hardware gates ran and reproduce, and the timing method holds. The builder's status line
(`gate_met: true`) is refuted: DESIGN-CCL section 8 defines the WP-C5 gate as "exposed cost <= 0.25 of FlyTape's
per-layer exchange share at TP2 B128; defaults chosen". The exposed cost is 0.71-0.72 of that share (0.718 in the rerun),
and the defaults are only partly chosen. The builder's status line put the numerics gate in place of this one; this
report's own "Gate and verdict" section states the miss correctly. Count WP-C5 as **gate not met**.

What was checked:
- **Gate scope.** DESIGN-CCL 6.3 asks the in-situ benchmark to decide `form`, `publish` and the `ar=auto` thresholds
  per (rows, TP), over rows 8-128.
  - Built: `form` lamport / pull, plus `cons`, `gb` and `pause`.
  - Not built: `publish=unit_end / comm_wave / counted`, `form=line`, `ar=twoshot`, and a separate FlyTape-style arm
    with a grid sync before the consumer.
  - Not run: rows 8 and 32, and TP4. "Defaults chosen" is therefore partial.
  - At TP2, two-shot moves the same bytes per link as one-shot, so the link-bound argument in section 7 still holds.
- **Hardware numerics (builder's jobs 68234979, 68235021).**
  - Every rows log has 38 PASS lines, no FAIL, and 11 of 11 must-fail twins detected: 110 in total.
  - The reference is independent of the kernel under test. It is the host rank-order f32 sum of every rank's
    plain-GEMM Y, gathered over NCCL, so an arm that never reads the peer's partial cannot pass.
  - The lamport `noreset` twin fails on 256 of 258 layers in launch 1 (exactly the slot-reuse layers) and on every
    layer in launches 2-3. The pull `slot` twin and the `kskip` mutants fail on every layer.
  - The run digest `ecd9cbb8fa30` (gemm.py + insitu.py + wpc5_bench.py) equals the local files. gemm.py's sha256 is
    `39e27905`, the P-L3 fix round's.
- **Raw JSON recompute.** Every default-arm ratio in the tables reproduces from `per_rank_us`. Arms ran in 24 shuffled
  rounds with recorded seeds, and nothing was dropped. The rank-min and rank-max ratios differ by at most 0.001.
  Controls (`#rc`, `gemm_ph#b`, `aiter#2`) were 0.98-1.02.
- **Rerun (cheapest decisive check).** Job 68235311, stages `w6-vWP-C5-h` / `w6-vWP-C5-rocke`.
  - Setup: n07-17, buses 75 / 15 (GRES IDX 1,3, both randn health-checked). Same code digest, rows 128 / 1 / 16,
    16 shuffled rounds, aiter + RCCL x 3 protocols.
  - Numerics: 3 x 38 PASS, no FAIL, 33 of 33 must-fail detected.
  - Ratios (rows 128 / 16 / 1):
    - `lam` / GEMM + aiter: 0.897 WIN (16/16), 0.824 WIN (16/16), 1.000 TIE (9/16);
    - `lam` / GEMM + RCCL best: 0.729, 0.647, 0.646, all WIN;
    - exposed `lam` / exposed aiter: 0.641, 0.227, 0.391;
    - exposed `lam` / FlyTape's per-exchange cost: 0.718 at B128, 0.240 at B16.
  - Controls were 0.980-1.011. Every verdict matches the builder's. The only notable difference is the B1 ratio to
    FlyTape: 0.83 here against 0.97. At B1 the exposed cost is a small difference of two times, so that ratio is noisy.
  - Raw: `private-perf/.../large-batch/wpc5-raw/w6-vWP-C5-a/`.
- **Fairness.**
  - All arms use the same P-L3 GEMM plan, the same rows, the same bf16 payload and the same two GPUs.
  - aiter runs with registered-input graph capture (ATOM's path). "RCCL best" in the report is the fastest protocol's
    median.
  - PLAN decision 5's DP2 reference does not apply to a kernel-level exchange comparison, so I agree with section 8.
    Nothing here may be quoted as an end-to-end 2-GPU win.

Issues found (none overturns the measured verdicts):
1. **`cons=fold` at 128 rows is borderline, not a loss.** The rerun gives `lamF / lam` = 0.991, a WIN under 9.1. The
   builder's runs gave 0.995 TIE on both nodes. Keeping `cons=pass` at 128 rows is not robust: either re-decide it in
   the layer context (WP-C6 / WP-L5) or record it as within 1 %. At every other size the loss reproduces (1.11-1.14).
2. **The FlyTape exchange proxy favours the in-situ arm.**
   - At B16 / B64 the TP2 - TP1 delta of the replicated phases is mostly `*_mix_combine` waits. Those waits also absorb
     rank imbalance from the sharded phases before them: at B16, `gdn.moe_expert` grows by 0.24 of the TP2 step at
     TP2. The microbenchmark's ranks are balanced.
   - The replicated set also includes `ple_reduce` and the sample phases, yet the total is divided by 96.
   - So FlyTape's per-exchange cost is an upper bound. Section 5's "the in-situ exchange costs less of a step than
     FlyTape's own at every batch" is a comparison against that upper bound.
   - The non-monotone FlyTape shares (B8 0.093, B16 0.071, B32 0.082) fit this.
   - The gate miss (0.71 > 0.25) holds all the more.
3. **Console ratio bug in `wpc5_bench.py`.** Its `lam/rccl_best` line takes the minimum over protocols of the ratio,
   which compares against the slowest RCCL protocol: 0.598 at rows 128, where the report gives 0.721.
   `wpc5_report.py` (fastest protocol) is correct, and no verdict uses the console line.
4. **The MoE-down exchange was not measured.** The per-step shares (96 exchanges) assume it costs the same as the
   out-proj exchange.
5. **TP4 was not exercised on hardware.** It is CPU-only, as the report says.

## Fix round (W6, 2026-11-01)

**Status: partial. `gate_met` is false.** DESIGN-CCL section 8 sets the WP-C5 gate as an exposed cost of at most 0.25
of FlyTape's per-layer exchange share at TP2 B128, with defaults chosen. The defaults are now chosen at TP2. The gate
is not met under any estimate of FlyTape's cost, and it cannot be met on this schedule (F6).

This round answers every issue of the verification above. It also found and fixed one correctness hazard in the
in-situ builds (F1).

| verifier issue | what this round did | result |
|---|---|---|
| 1. `gate_met: true` is wrong | Restated the gate against a bracket of FlyTape's cost (F5, F6) | Not met: 0.58-0.77 at B128 for the out-proj exchange, 0.49-0.74 for the MoE-down exchange (default layout) |
| 2. Defaults only partly chosen | Built `form=line` x `publish=unit_end / comm_wave / counted`, `ar=twoshot`, `ar=auto`, and FlyTape's shape `cons=phase`. Ran rows 5, 8, 16, 32, 64, 128 at TP2 on both rocke nodes | Lamport one-shot with the consumer pass is the default at every TP2 bucket; every other value loses or ties on both nodes (F2). `ar=auto` holds the table. TP4 ran once, on n07-17 (F8): two-shot and the fold-side consumer win at 32-128 rows there, but every in-situ value loses to aiter at 32-128 |
| 3. `cons=fold` at 128 rows | Re-ran at 128 rows: 3 more runs on both nodes, 32-48 rounds each (F3) | TIE in every new run (0.993-0.996). `cons=pass` stays, recorded as "within 1 %" |
| 4. FlyTape proxy overstates | Traced both ranks of FlyTape TP2 and TP1 in one job; three estimates (F5) | FlyTape's exchange is 0.060-0.082 (skew floor) to 0.078-0.107 (old proxy) of its TP2 step. The 96 in-situ exchanges (both measured, default layouts) cost 0.048-0.061 of it |
| 5. Console ratio bug | Fixed in `wpc5_bench.py` and in the fix-round bench | No verdict used the console line |
| 6. MoE-down exchange not measured | Built it in situ (`insitu_moe.py`) with levers and measured it at rows 8-128 on both nodes (F4) | First layout: 1.3-8.3x the out-proj exchange. The combine-unit-major layout (now the default) cuts that to 0.84-4.3x. That is as cheap as aiter's exchange or cheaper, but the whole MoE + exchange still loses to MoE + aiter at 8-32 rows (1.05-1.08): the persistent MoE walk costs 1.04-1.08 |
| 7. Rows 1 is not a win | Rows 1-4 are outside the large-batch scope (5-128). Added rows 5, the in-scope lower edge (F7) | Rows 5: in situ beats GEMM + aiter (0.931 WIN) |

### F0. What was built

| piece | path | notes |
|---|---|---|
| in-situ GEMM knobs | `rocke/library/kernels/gfx950/qwen38_mono/large_batch/insitu.py` | `form=line` with `publish` (`unit_end`, `comm_wave`, `counted`); `ar` (`oneshot`, `twoshot` for lamport, `auto` through `resolve()` and the `AUTO` table); `cons=phase`; `hz` (F1). New arena regions sit after the old ones, so the old offsets do not move. `gemm.py` is unchanged (sha256 `39e27905`) |
| MoE-down in situ | `.../large_batch/insitu_moe.py` (new) | Subclasses P-L4's `moe_lb._Moe` (unchanged). The combine pushes its bf16 rows (lamport); a consumer pass runs after the combine. Knobs: `lay` (`auto` = `unit` under `combine=phase`: a combine-unit-major arena, so each push instruction writes 512 contiguous bytes; `tok` = token-major, the first layout measured), `cons=combine` (the combine unit consumes its own unit; no pass), `pause` |
| CPU tests | `rocke/library/tests/test_qwen38_lb_insitu.py` (111), `test_qwen38_lb_insitu_moe.py` (31) | Every new knob value builds, lowers and passes the CCL peer-address lint. Refusals, `ar=auto` resolution, arena maps, hz padding and the lay=unit address algebra are tested. The plain GEMM / MoE IR does not change |
| harness | `hackathon-prep/qwen38/truemono-lb/wp-c5-fix/` | `wpc5fix_bench.py` (made by `make_fix_bench.py` from `wp-c5/wpc5_bench.py`), `wpc5moe_bench.py`, `wpc5fix_specs.py`, `ft_ranks.py` (FlyTape bench with every rank's phase ticks dumped; FlyTape itself unedited), `ft_bracket.py`, `wpc5fix_report.py`, `hazard_scan.py`, `ir_hashes.py`, `job.sh`, `sync_fix.sh` |

How the new values work:
- **`form=line`.** The fold writes 8 rows x 1 token per thread, so one wave's 64 lanes write one 1 KiB line. Each line
  goes to every rank with `global_store_dwordx4 sc0 sc1`, and its flag follows the payload. The consumer polls the
  line's flag of every rank in its own arena, then reads the lines locally.
  - `unit_end`: every wave drains, the CTA meets at a barrier, then lanes store the block's flags.
  - `counted`: no CTA barrier. A wave publishes line i's flag after it has waited for fold iteration i + 1's partial
    loads. Those loads were issued after line i's stores, and the explicit `vmcnt(0)` at that point costs nothing.
    The last line drains.
  - `comm_wave`: the fold writes the block to LDS. The last wave pushes it, drains only its own stores and stores
    the flags. The other seven waves never drain.
- **`ar=twoshot` (lamport).** Block b goes only to its owner rank, b mod N. The owner's consumers sum in rank order
  and push the bf16 sum to every rank. Every rank then reads the owner's sum. A wave runs all its reduce-scatter units
  before any all-gather unit, so there is no cycle.
- **`cons=phase`.** FlyTape's shape: the consumer pass runs behind an extra grid barrier of the rank.
- **Byte identity.** With `hz=False`, all 158 pre-fix in-situ builds are byte-identical in LLVM IR to the pre-fix
  module (`wp-c5-fix/ir_hashes.py`, apart from the `_nohz` name). `hz=True` adds exactly one `s_nop 4` to the asm
  blocks that need it (CPU test).
- **Compile and hazard check.** All 268 fix-round GEMM builds and all 88 MoE in-situ builds compile with COMGR on the
  laptop, and the ISA hazard scan finds no hit in any of them (F1).
- **The MoE layout default changed after the measurements.** The runs named the token-major build without a suffix
  and the unit-major one `_lunit`. Now `lay=auto` resolves to `unit` (no suffix), and `tok` is named `_ltok`. The LLVM
  IR is the same apart from the name (CPU test).

### F1. A correctness hazard in the in-situ builds (new finding, fixed)

P-L4's ISA scanner (`p-l4/fix/asm_hazard.py`, alola rule 182) looks for an inline-asm VMEM instruction that reads an
SGPR fewer than 5 wait states after a VALU wrote it. The compiler restores spilled peer and arena bases with
`v_readlane`, and LLVM does not pad VMEM inside inline asm.

The scanner found unpadded hits in the in-situ builds, which WP-C5 had not scanned:
- **WP-C5's measured binaries.** 6 of 135 builds have hits:
  - the must-fail twins `mut_lam`, `mut_lamS` and `mut_lamL` at 16 rows;
  - the timed lever `lamL` and its `#rc` twin at 16 rows;
  - `mut_lam_kskip` at 128 rows.
  - The default `lam` builds at every size were clean, so no default-arm verdict is affected.
- **TP4.** Most TP4 builds have hits, including the default.
- **Effect on WP-C5.** WP-C5's numerics passed bitwise, with no grid-barrier timeouts, so the hazard never fired
  visibly. But a must-fail twin "detected" by a hazard proves nothing.

The fix is `pad_asm_vmem()`: knob `hz`, on by default, as in P-L4. It prefixes `s_nop 4` to every inline-asm block that
issues VMEM with an SGPR input. That covers the GEMM's grid-barrier atomic and poll, the CCL pushes, polls and
resets, and the error record.
- After the fix: 0 hits in all 268 GEMM and 88 MoE in-situ builds.
- `lamH0` (the default arm with `hz=False`, timed in every run) against `lam` is a TIE at every size and node
  (0.987-1.013). The one exception is a 1.015 LOSS at rows 32 on n01-09, where the padded build is the faster one. So
  the pad costs nothing measurable.
- This round re-ran `mut_lam` (16 rows), `mut_lam_kskip` (128 rows), `mut_lamL` and the `lamL` lever with `hz` on:
  every twin was detected, and `lamL` was re-measured (F2).
- `mut_lamS`, the twin of the poll-pause lever, was not re-run. That lever is noise and stays default-off.

### F2. Rows 5-128 at TP2: every DESIGN-CCL 6.3 value against the default (`lam`)

**Runs.**
- Job 68235814 (`w6-wpc5f-a`): n07-17, buses 05 / 65, rows 5 / 8 / 16 / 32 / 64 / 128, 32 rounds.
- Job 68235868 (`w6-wpc5f-c`): n07-17, buses 05 / 65, rows 16 / 128, 32 rounds, with the FlyTape traces.
- Job 68236144 (`w6-wpc5f-b`): n01-09, buses 05 / e5 (healthy on this node), rows 5 / 8 / 16 / 32 / 64 / 128, 48
  rounds.
- Each run was set up as in WP-C5:
  - every arm in one process per rank, shuffled every round, a gloo barrier before every replay;
  - numerics before timing, on the timed binaries;
  - controls: `#rc` recompiled twins, `gemm_ph#b`, `gemm_ph#rc`, `aiter#2`;
  - health-checked GPUs, unstamped binaries, clocks sampled.

**Numerics, every run.**
- Every arm (`lam`, `lamF`, `line`, `lineC`, `lineK`, `lam2`, `lamP`, `lineP`, `lamL`, `gemmL`, `lamH0`, each with a
  `#rc` twin) is bitwise equal to the host rank-order sum on both ranks: 258 layers x 3 launches (activation sets
  A, B, A), and again after graph replay.
- Error records are clean and there were no grid-barrier timeouts.
- Every must-fail twin was detected: 13 per rows value, 0 missed. The twins are:
  - `noreset` for every lamport arm;
  - `slot` for every line arm;
  - `kskip` for `lam`, `line`, `lam2` and `lineP`.
- The aiter and RCCL outputs are bitwise equal to the host sum.

Ratio to `lam` (median of per-round min over ranks), n07-17, DESIGN 9.1 verdict against the controls:

| arm | rows 5 | rows 8 | rows 16 | rows 32 | rows 64 | rows 128 |
|---|---|---|---|---|---|---|
| `line` (lines + flags, `unit_end`) | 1.108 LOSS | 1.121 LOSS | 1.107 / 1.110 LOSS | 1.074 LOSS | 1.131 LOSS | 1.114 / 1.112 LOSS |
| `lineC` (`comm_wave`) | 1.093 LOSS | 1.094 LOSS | 1.097 / 1.092 LOSS | 1.090 LOSS | 1.142 LOSS | 1.137 / 1.135 LOSS |
| `lineK` (`counted`) | 1.060 LOSS | 1.066 LOSS | 1.089 / 1.087 LOSS | 1.098 LOSS | 1.109 LOSS | 1.097 / 1.095 LOSS |
| `lam2` (`ar=twoshot`) | 1.060 LOSS | 1.063 LOSS | 1.053 / 1.052 LOSS | 1.052 LOSS | 1.025 LOSS | 1.014 / 1.015 LOSS |
| `lamP` (FlyTape shape, lamport) | 1.375 LOSS | 1.368 LOSS | 1.382 / 1.405 LOSS | 1.301 LOSS | 1.230 LOSS | 1.118 / 1.109 LOSS |
| `lineP` (FlyTape form and shape) | 1.454 LOSS | 1.445 LOSS | 1.513 / 1.538 LOSS | 1.395 LOSS | 1.352 LOSS | 1.203 / 1.191 LOSS |
| `lamF` (`cons=fold`) | 1.111 LOSS | 1.115 LOSS | 1.118 / 1.130 LOSS | 1.037 LOSS | 1.048 LOSS | 0.996 / 0.993 TIE |
| `lamL` (`gb=lite`, re-run with `hz`) | 1.057 LOSS | 1.041 LOSS | 1.023 / 1.026 LOSS | 1.017 TIE | 1.006 TIE | 1.006 / 1.011 TIE |
| `lamH0` (`hz=False`) | 1.008 TIE | 1.007 TIE | 0.989 / 0.987 TIE | 1.015 TIE | 1.000 TIE | 1.003 / 1.004 TIE |

**n01-09 (job 68236144, 48 rounds) gives the same verdicts:**
- Every `line`, `lineC`, `lineK`, `lam2`, `lamP` and `lineP` cell is a LOSS: `line` 1.08-1.14, `lineC` 1.09-1.16,
  `lineK` 1.06-1.12, `lam2` 1.02-1.07, `lamP` 1.11-1.39, `lineP` 1.19-1.52.
- `lamF` is a LOSS (1.04-1.14) at rows 5-64 and a TIE at 128 (0.995).
- `lamL` is a LOSS at 5-64 (1.02-1.06) and a TIE at 128.
- `lamH0` is a TIE (0.989-1.013) except rows 32, where it is a 1.015 LOSS. So the unpadded build is not faster; the
  pad is free.
- The per-cell tables are in `WP-C5-tables.md`.

Default arm `lam` end to end (n07-17 job 68235814 / 68235868; n01-09 job 68236144):

| rows (KiB / rank) | lam / GEMM + aiter (9.1) | lam / GEMM + RCCL best (9.1) | exposed lam / exposed aiter | exposed lam / exposed RCCL best |
|---|---|---|---|---|
| 5 (25) | 0.931; 0.933 WIN | 0.632; 0.624 WIN | 0.345; 0.352 | 0.148; 0.147 |
| 8 (40) | 0.905; 0.890 WIN | 0.676; 0.664 WIN | 0.332; 0.301 | 0.176; 0.159 |
| 16 (80) | 0.838 / 0.825; 0.834 WIN | 0.652 / 0.644; 0.637 WIN | 0.290 / 0.261; 0.283 | 0.171 / 0.155; 0.162 |
| 32 (160) | 0.888; 0.894 WIN | 0.642; 0.646 WIN | 0.405; 0.408 | 0.219; 0.220 |
| 64 (320) | 0.870; 0.862 WIN | 0.649; 0.634 WIN | 0.503; 0.499 | 0.294; 0.285 |
| 128 (640) | 0.884 / 0.890; 0.887 WIN | 0.719 / 0.726; 0.723 WIN | 0.622 / 0.631; 0.633 | 0.441 / 0.449; 0.450 |

Every WIN has every round below 1. Controls per cell were 0.983-1.020.

**Why the other values lose:**
- **`line`.** It must drain before its flags can go out, and the consumer pays a flag poll before it can read the
  payload. Lamport's payload poll pays neither.
  - `counted` and `comm_wave` shave the drain (by 0.01-0.06 below `unit_end` at 5-16 rows) but cannot remove the
    flag round trip.
  - At 64-128 rows `comm_wave` is the slowest line variant: one wave pushes the whole 4-16 KiB block.
- **`ar=twoshot`.** At TP2 it moves the same bytes over the one link and adds a hop. The loss shrinks with size,
  from 1.06 to 1.014, which matches CCL-LM's microbenchmark ("two-shot never wins at TP2").
- **`cons=phase` (FlyTape's shape).** The extra grid barrier before the consumer costs 0.11-0.41 of a layer with
  lamport (`lamP / lam`) and 0.07-0.39 with lines (`lineP / line`). It is worst at small rows, where the barrier
  dominates.
  - So FlyTape's structure, a consumer behind a grid sync, is itself a large part of why the in-situ exchange beats
    FlyTape's.

**Defaults** (`insitu.AUTO`, `ar=auto`): at TP2, `form=lamport`, `publish` not applicable, `ar=oneshot`, `cons=pass` at
buckets 16, 32, 64 and 128. Every other value stays a registered default-off knob.

### F3. `cons=fold` at 128 rows

| run | node | rounds | lamF / lam | rounds below 1 | 9.1 verdict |
|---|---|---|---|---|---|
| WP-C5 68234979 | n07-17 | 24 | 0.995 | - | TIE |
| WP-C5 68235021 | n01-09 | 24 | 0.995 | - | TIE |
| verifier 68235311 | n07-17 | 16 | 0.991 | - | WIN |
| fix 68235814 | n07-17 | 32 | 0.996 | 27/32 | TIE |
| fix 68235868 | n07-17 | 32 | 0.993 | 32/32 | TIE |
| fix 68236144 | n01-09 | 48 | 0.995 | 47/48 | TIE |

Decision: `cons=pass` stays the default at 128 rows. Recorded as within 1 % (`lamF` 0.991-0.996 of `lam`): a TIE in
every run except the verifier's single WIN.
- `cons=fold` also has a precondition the pass does not have: at most one unit per CTA per layer, which `check`
  enforces.
- Re-decide it in the layer context (WP-L5), where the consumer is the next stage's prologue.

### F4. The MoE-down all-reduce in situ (measured, not assumed)

**Setup.**
- The MoE is P-L4's default arm at the TP2 rank shape: the rank's I = 320 of 640, W4A8 (ATOM's numerics class),
  random router logits, 8 layers walked with P-L4's grid barrier.
- The router logits and MoE input are replicated on both ranks (same seed); each rank has its own expert weights.
- Arms:
  - `moe_ph` (one launch of 8 layers) is the in-situ arms' floor;
  - `moe_ln` (one launch per layer) is the separate-kernel floor;
  - `aiter` / `rccl_*` are the MoE launch plus the all-reduce as separate kernels in a graph.

**Runs.**
- n07-17: job 68235814 (buses 05 / 65, `lamM` only), job 68235868 (rows 128), job 68236240 (buses 65 / 15, every
  lever).
- n01-09: job 68236144 (buses 05 / e5, every lever).
- 32 rounds each.

**Numerics, every run.**
- The plain MoE is deterministic: 2 launches plus the recompiled twin are bitwise equal, with error words 0.
- Every in-situ arm and its `#rc` twin is bitwise equal to the host rank-order sum on both ranks, over 8 layers x 3
  launches (A, B, A), and again after graph replay. The MoE error words stayed 0 after timing.
- Every must-fail twin was detected at every size: `noreset` of every arm and the MoE's `slot` mutant; 0 missed.

**Levers against the first layout (`lamM`: token-major arena, consumer pass).** Ratio to `lamM`, 9.1 verdict, both
nodes:

| arm | rows 8 | rows 16 | rows 32 | rows 64 | rows 128 |
|---|---|---|---|---|---|
| `lamMu` (`lay=unit`) | 0.956 WIN; 0.969 TIE | 0.927 WIN; 0.922 WIN | 0.952 WIN; 0.954 WIN | 0.957 WIN; 0.963 WIN | 0.971 WIN; 0.969 WIN |
| `lamMcu` (`lay=unit`, `cons=combine`) | 0.963 WIN; 0.970 TIE | 0.937 WIN; 0.935 WIN | 0.954 WIN; 0.955 WIN | 0.968 WIN; 0.967 WIN | 0.963 WIN; 0.961 WIN |
| `lamMus16` (`lay=unit`, `pause=s_sleep16`) | 0.958 WIN; 0.966 TIE | 0.932 WIN; 0.921 WIN | 0.955 WIN; 0.953 WIN | 0.960 WIN; 0.963 WIN | 0.971 WIN; 0.969 WIN |
| `lamMc` (`cons=combine`) | 1.022 TIE; 1.041 LOSS | 1.013 TIE; 1.004 TIE | 1.000 TIE; 1.004 TIE | 1.006 TIE; 1.007 TIE | 0.991 TIE; 0.990 TIE |
| `lamMs1` / `lamMs16` (`pause`) | 0.999-1.007 TIE | 0.998-1.005 TIE | 1.000-1.002 TIE | 0.999-1.003 TIE | 0.998-1.000 TIE |

(n01-09; n07-17 65 / 15.)

**What the levers show.**
- **The arena layout was one cause.** The token-major layout scatters a combine unit's 8-byte pushes into 32-byte
  pieces of 16 token rows. The unit-major layout writes 512 contiguous bytes per instruction.
- **Poll pressure was not the problem.** The pause levers are noise.
- **`cons=combine` alone does not help.** On top of the unit layout, `lamMcu / lamMu` is a TIE at every size and node
  (0.991-1.013).
- **Decision.** `lay=unit` becomes the in-situ MoE default (`lay=auto`). `cons=pass` and `pause=s_nop` stay.

Default MoE in-situ arm (`lamMu`) end to end, n01-09; n07-17 (65 / 15):

| rows | lamMu / moe_ph | moe_ph / moe_ln | lamMu / MoE + aiter (9.1) | lamMu / MoE + RCCL best (9.1) | exposed lamMu / exposed aiter | exposed lamMu / exposed out-proj lam |
|---|---|---|---|---|---|---|
| 8 | 1.114; 1.123 | 1.082; 1.077 | 1.074 LOSS; 1.073 LOSS | 0.971 WIN; 0.971 WIN | 1.02; 1.05 | 3.4; 3.3 |
| 16 | 1.093; 1.105 | 1.083; 1.081 | 1.071 LOSS; 1.078 LOSS | 1.000 TIE; 1.008 TIE | 0.98; 1.05 | 3.6; 4.3 |
| 32 | 1.086; 1.096 | 1.061; 1.051 | 1.056 LOSS; 1.049 LOSS | 0.987 TIE; 0.982 TIE | 1.01; 1.05 | 2.7; 2.9 |
| 64 | 1.069; 1.072 | 1.047; 1.048 | 1.008 TIE; 1.018 LOSS | 0.955 WIN; 0.960 WIN | 0.66; 0.74 | 1.8; 1.8 |
| 128 | 1.061; 1.056 | 1.037; 1.043 | 0.985 TIE; 0.985 WIN | 0.944 WIN; 0.943 WIN | 0.54; 0.49 | 0.96; 0.84 |

With the first layout (`lamM`, both nodes, 4 runs) the same columns were:
- `lamM / MoE + aiter`: 1.10-1.17 LOSS at rows 8-32, 1.05 LOSS at 64, 1.013-1.016 LOSS / TIE at 128;
- `exposed lamM / exposed aiter`: 1.4-2.0 at 8-32, 0.78-0.83 at 128;
- `exposed lamM / exposed out-proj lam`: 4.2-8.3 at rows 8-32, 1.3-1.6 at 128.

**What this changes.**
- **The WP-C5 assumption was wrong.** WP-C5 took the MoE-down exchange to cost the same as the out-proj one.
  - With the default layout it costs 3.3-4.3x as much at 8-16 rows and 1.8-2.9x at 32-64 rows. Only at 128 rows is
    it about the same (0.84-0.96).
  - With the first layout it cost 1.3-8.3x as much.
- **Why it costs more: likely causes, not isolated by a measurement.**
  - The MoE's combine units finish at very different times. The out-proj GEMM's units all finish within one unit's
    time.
  - The exchange cannot start until a unit's combine has polled its down-projection arrivals.
  - What the levers do rule out: poll pressure (the pause levers are noise) and the separate consumer pass
    (`cons=combine` ties).
- **Against aiter.** The in-situ exchange alone is now as cheap as aiter's separate all-reduce at 8-32 rows
  (0.98-1.05 of its exposed cost) and cheaper at 64-128 (0.49-0.74).
  - End to end, MoE + exchange in one launch still loses to MoE + aiter at 8-32 rows (1.05-1.08). At 64 rows it is
    1.008-1.018 (TIE / LOSS), and at 128 rows 0.985 (TIE / WIN).
  - The persistent MoE walk (`moe_ph / moe_ln` = 1.04-1.08, P-L4's gs versus launch metric) accounts for most of
    that loss.
  - So the in-situ MoE exchange pays off only with a MoE whose layer walk costs no more than one launch per layer. That
    is WP-L2d's rebuilt MoE, not P-L4's.
- **Section 5's share table is superseded by F5.**

### F5. FlyTape's exchange cost, bracketed

**How it was traced.**
- Job 68235868 traced FlyTape TP2 (B8, B16, B32, B64, B128) and TP1 (B8, B16, B32, B64) on n07-17, buses 05 / 65.
- `ft_ranks.py` dumped every rank's per-step phase ticks; the stock bench prints rank 0 only.

**Three estimates of the exchange cost per decode step (`ft_bracket.py`):**
- **old:** WP-C5's proxy. The TP2 - TP1 difference of every replicated mixer, combine, reduce and up phase, plus
  `moe_combine`, `ple_reduce` and the sample mixers, on rank 0.
- **restricted:** only the phases that hold the 96 exchanges' waits or pushes: `(attn|mlp)_mix_combine` and
  `moe_combine` of every layer type, and `sample.final_mix_combine`. Computed per rank, then averaged over the ranks.
- **floor:** the same phases, but for each (instance, step) the minimum over the ranks is taken before summing. The
  later-arriving rank of a skewed pair waits only for the flight, so this removes the skew.
  - It is a lower estimate: when the skew exceeds the flight, it also hides part of the flight.

| batch | old (P-L3's traces, WP-C5) | old (this job) | restricted, rank mean (rank 0, rank 1) | skew floor | 96 in-situ exchanges, measured: out-proj lam + MoE lamMu (default) | same with the first MoE layout (lamM) |
|---|---|---|---|---|---|---|
| 8 | 0.093 | 0.098 | 0.086 (0.097, 0.074) | 0.070 | 0.048-0.051 | 0.064-0.067 |
| 16 | 0.071 | 0.078 | 0.070 (0.074, 0.066) | 0.060 | 0.049-0.052 | 0.084-0.089 |
| 32 | 0.082 | 0.107 | 0.084 (0.101, 0.068) | 0.064 | 0.058-0.061 | 0.085 |
| 64 | 0.076 | 0.103 | 0.083 (0.098, 0.069) | 0.064 | 0.055-0.058 | 0.074-0.080 |
| 128 (TP1 extrapolated) | 0.089 | 0.106 | 0.098 (0.106, 0.091) | 0.082 | 0.057-0.061 | 0.076-0.077 |

All columns are shares of FlyTape's TP2 step. The ranges are the two nodes; the in-situ costs come from their own
runs, divided by this job's FlyTape step.

**Reading the table.**
- **The verifier was right.**
  - Rank 0 waits more than rank 1 at every batch (rank 1 is the slower rank in the sharded phases).
  - The same proxy on two jobs differs by up to 0.027 of the step: the TP2 phases change with the GPU pair's skew.
  - The restriction to the exchange phases lowers the proxy by 0.008-0.023, and removing the skew by another
    0.010-0.020.
- **In situ against FlyTape's own exchanges, with both measured.** With the default MoE layout, the 96 in-situ
  exchanges cost 0.048-0.061 of FlyTape's TP2 step. That is below even FlyTape's skew floor (0.060-0.082) at every
  batch.
  - The margin is 0.003-0.025 of the step, smaller than WP-C5's table implied (0.025-0.05).
  - With the first MoE layout the in-situ total sat inside or below FlyTape's bracket, except at B16, where it was
    above even the old proxy.
  - So WP-C5's claim is confirmed only with the layout fix. It holds largely because the out-proj exchange is cheap:
    0.22-0.39 of FlyTape's per-exchange cost (old proxy) at B8-B64.

### F6. The gate

The in-situ exposed cost per exchange, divided by FlyTape's per-exchange cost, at TP2 B128 (two nodes):

| exchange | old (P-L3's traces; WP-C5) | old (this job) | restricted | skew floor |
|---|---|---|---|---|
| out-proj (`lam`) | 0.71-0.72 | 0.58-0.59 | 0.63-0.64 | 0.76-0.77 |
| MoE-down (`lamMu`, default) | - | 0.49-0.57 | 0.53-0.61 | 0.64-0.74 |
| MoE-down (`lamM`, first layout) | - | 0.84-0.87 | - | - |

The target is at most 0.25. It is not met under any estimate, for either exchange.

**Why 0.25 is out of reach on this schedule.**
- At 128 rows the out-proj exposed cost is 0.89-0.90 of the time its 640 KiB bf16 partial needs to cross the one xGMI
  link (WP-C5 section 7; `lam`'s exposed cost is unchanged in this round, within 0.02).
- So the link time alone is 0.65-0.87 of FlyTape's per-exchange cost under the three estimates.
- An exposed cost of 0.25 needs one of:
  - fewer bytes: `ar_dtype=fp8` would give about 0.53x the bytes, which is still 0.34-0.46 of FlyTape's cost;
  - overlap of the exchange with the next layer's work: DESIGN-LB's tile-flag dataflow, where the next layer's units
    start on blocks whose exchange has landed.
- DESIGN-CCL's 0.25 target therefore assumes overlap. Overlap is WP-L3 / WP-L5 work, not an epilogue knob.

### F7. Rows 1-4

- Rows 1-4 are outside the large-batch scope (rows 5-128, PLAN). WP-C5's rows 1 / 4 results stay as measured: at
  rows 1, a LOSS or TIE against GEMM + aiter, caused by the phase barrier.
- No large-batch claim rests on them. Small batch is W5's narrow path.
- The in-scope lower edge, rows 5, is a WIN: 0.931 against GEMM + aiter, 0.632 against RCCL best.

### F8. TP4 (one run)

**Setup.**
- Job 68236286 (`w6-wpc5f-e`): n07-17, 4 GPUs (buses 05 / 65 / 15 / 85, all health-checked).
- The budget allowed it once, near the end of the round. It ran only on this node.
- TP4 plan of record: K 1536, 4 waves, 3 splits. Rows 8 / 16 / 32 / 64 / 128, 24 rounds.
- Every TP4 build is hazard-free with `hz`. Before the fix most were not (F1), so a TP4 run without `hz` would have
  been unsafe.

**Numerics.**
- Every arm is bitwise equal to the host rank-order sum on all 4 ranks (3 launches x 258 layers, plus graph replay).
- All 65 must-fail twins were detected.
- aiter is bitwise. RCCL is within tolerance (0.67 bitwise share, different reduction order).

Results:

| rows | best in-situ value against `lam` (9.1) | best in-situ / GEMM + aiter (9.1) | best in-situ / GEMM + RCCL best | other values against `lam` |
|---|---|---|---|---|
| 8 | `lam` | 0.942 WIN | 0.574 WIN | line forms 1.04-1.07, `lam2` 1.055, `lamF` 1.087, phase 1.37-1.45: LOSS |
| 16 | `lam` | 0.874 WIN | 0.589 WIN | line forms 1.05-1.10, `lam2` 1.048, `lamF` 1.075, phase 1.33-1.45: LOSS |
| 32 | `lamF` 0.978 WIN | 1.010 LOSS | 0.640 WIN | `lam2` 1.034, line forms 1.03-1.08, phase 1.29-1.36: LOSS |
| 64 | `lamF` 0.955 WIN; `lam2` 0.959 WIN | 1.107 LOSS | 0.747 WIN | line forms 1.06-1.11, phase 1.13-1.25: LOSS |
| 128 | `lam2` 0.883 WIN; `lamF` 0.982 WIN | 1.117 LOSS | 0.732 WIN | line forms 1.08-1.18, phase 1.08-1.13: LOSS |

**What it shows.**
- **TP4 differs from TP2.**
  - Two-shot wins at 64-128 rows. That fits the bytes: one-shot sends the whole partial over each of the 3 links,
    and two-shot halves the bytes per link.
  - `cons=fold` wins at 32-128 rows. Why it wins at TP4 and not at TP2 is not isolated. The unit counts are similar
    (120 per layer), and the TP4 consumer pass polls 4 sources instead of 2.
- **The in-situ forms lose to aiter at 32-128 rows (1.01-1.12).** aiter's TP4 all-reduce beats every in-situ value
  built here. Two-shot together with the fold-side consumer is not built: `ar=twoshot` refuses `cons=fold`.
- **`insitu.AUTO` now holds this run's winners at TP4, marked provisional (one node):**
  - the pass at bucket 16;
  - `cons=fold` at 32 and 64;
  - `ar=twoshot` at 128.
- The large-batch plan is TP1 and TP2. TP4 here is the opportunistic DESIGN-CCL check, not a plan-of-record
  configuration.

### F9. Console ratio bug

`wpc5_bench.py` (and the fix-round bench) now print `arm / rccl_best` against the protocol with the fastest median.
That is what `wpc5_report.py` always used.

### F10. Reproduce

From `hackathon-prep` (Git Bash), with `Q38_MAX_GPUS=8`:

```bash
bash qwen38/truemono-lb/wp-c5-fix/sync_fix.sh both        # stages w6-wpc5f-h, w6-wpc5f-rocke
Q38_NODE=smci355-ccs-aus-n07-17 bash qwen38/truemono-lb/submit.sh logs/q38/w6/wpc5f-a.poll --tag w6-wpc5f-a \
  --mode container --gpus 3 --cpus 48 --mem 600 --time 01:45:00 --stage w6-wpc5f-rocke --stage w6-wpc5f-h \
  --env "WPC5_ARGS=--aiter+--rounds+32" --env "WPC5M_ARGS=--aiter+--rounds+32" \
  -- 'bash $Q38_STAGES/w6-wpc5f-h/job.sh pick:2 bench:2:128,5,8,16,32,64 moe:2:128,8,16,32,64'
# second node (n01-09, 48 rounds) and the MoE levers: Q38_NODE=smci355-ccs-aus-n01-09, --gpus 2, tag w6-wpc5f-b,
#   --env "WPC5M_ARGS=--aiter+--rounds+32+--arms+lamM,lamMu,lamMs1,lamMs16,lamMus16,lamMc,lamMcu", steps pick:2 moe:2:... bench:2:...
# TP4: --gpus 4, steps 'pick:4 bench:4:128,16,64,8,32' (tag w6-wpc5f-e)
# FlyTape bracket: add --stage flytape and the steps ftr:2:8x128,16x128,32x128,64x128,128x128 ftr:1:8x128,16x128,32x128,64x128
python qwen38/truemono-lb/wp-c5-fix/ft_bracket.py --tp2 <C>/ftr-tp2 --tp1 <C>/ftr-tp1 --steps-json <C>/ftr-tp2/ft.jsonl --json FB.json
python qwen38/truemono-lb/wp-c5-fix/wpc5fix_report.py --runs <A> <B> <C> <E> --moe <A> <B> <C> <D> --ft-bracket FB.json \
  --out-ratio R.md --out-abs A.md
```

Full tables: `WP-C5-tables.md` (section "Fix round"). Absolute times: `private-perf/.../large-batch/WP-C5.md` ("Fix
round") and `wpc5-raw/w6-wpc5f-*`.

### F11. Written

- This section and the banner at the top of this file.
- `WP-C5-tables.md`, section "Fix round" (every run, ratios only).
- `hackathon-prep/private-perf/qwen38-true-mono/large-batch/WP-C5.md`, section "Fix round" (absolute numbers and
  FlyTape's per-exchange costs), plus `wpc5-raw/w6-wpc5f-{a,b,c,d,e}/`, `fix_summary.json` and `ft_bracket.json`.
- Alola-cluster skill rule 191:
  - the in-situ asm hazard and the laptop-side ISA scan;
  - budget starvation under a shared 8-GPU cap;
  - FlyTape traced on both ranks;
  - RCCL at TP4 is not bitwise.
