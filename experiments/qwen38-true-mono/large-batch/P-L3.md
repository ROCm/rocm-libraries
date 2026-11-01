# P-L3: the large-batch tiled dense GEMM core (DESIGN-LB E1), rows 8-128, TP1 / TP2, bf16 and MXFP4 W4A8

Probe P-L3 of `../DESIGN-LB.md` (section 12, group L0; E1 in section 6). One rocKE kernel family for every dense
projection of a large-batch step: the weight rows are the MFMA A operand, the batch is the N dimension in 16-token
tiles, so every weight byte crosses HBM once per step whatever the batch. Built, proven on hardware (tolerance vs
a float64 torch reference, bitwise invariance over order-preserving knobs, determinism, must-fail mutants), then
measured per DESIGN.md 9.1 against FlyTape's tiled-tape plans (`flytape/blocks/tiled.py`, `bodies_tiled.py`,
`bodies_head.py`) at the same shapes, rows and bytes, and against a bytes-floor streamer. Ratios only here; absolute
times are in `hackathon-prep/private-perf/qwen38-true-mono/large-batch/P-L3.md`. Full tables: `P-L3-tables.md`.

Status: **done, gate met; verdicts restated in the fix round (W6)** (numerics on hardware with mutants, now including
the op_sel path; fraction of the bytes floor per cell; per-projection ratio vs FlyTape per cell with matched fold
accounting, on both rocke nodes, two runs per node; ATOM's W4A8 linear and aiter's MXFP4 / bf16 GEMMs as library arms).
The summary and sections 6-7 below are the fix-round numbers (section "Fix round (W6)" at the end; tables in
`P-L3-fix-tables.md`). First measured 2026-10-31, jobs 68231750 (n01-09, GPU bus 15) and 68231751 (n07-17, GPU bus 05),
FlyTape traces job 68231605 (n01-09, buses f5 + e5), earlier rounds 68231600-68231653, diagnosis jobs
68230098-68231540; fix round jobs 68232590, 68232959 (n01-09), 68232651, 68233016 (n07-17), 68232593, 68232821,
68233558 (container).

## 1. Summary

1. **The GEMM core works at both precisions, rows 8-128, every dense projection** (gdn_in, gdn_out = qsa_out's
   shape, qsa_in, HC down, HC up, router, head; TP1 and TP2 per-rank shapes; 55 cells). Numerics (fix round, four
   plan-of-record runs plus two container runs): 18,752 hardware records, 0 LB failures; 132 of 132 mutant runs caught
   in every run (`kskip` 65, `fold` 33, `scale` 21, and the new op_sel-path `sopsel` 13 at 580 eps and up; `kskip` now
   also runs on the op_sel arms); every knob that
   keeps the summation order gives bit-identical outputs (including the grid-synced multi-layer walk and the op_sel
   scale path). The `sopsel` mutant leaves the LLVM IR of every existing build unchanged (14 builds + 5 streamers vs
   the measured tree, in-job, both nodes).
2. **Checkpoint precision (MXFP4 W4A8, ATOM's) vs FlyTape's plan at bf16 (FlyTape's precision), same harness, both
   sides producing the complete output:** the LB mx best wins **55 of 55 cells in every run on both nodes** (phase
   metric); launch 49 WIN, 5 LOSS (HC up, every row count, 1.08-1.38), 1 MIXED; stream 50 WIN, 2 LOSS, 2 MIXED,
   1 TIE. Phase ratios 0.30-0.36 (head), 0.34-0.51 (gdn_in), 0.35-0.63 (qsa_in), 0.70-0.83 (out-proj), 0.65-0.75
   (HC down), 0.69-0.83 (HC up), 0.61-0.86 (router). This mixes precision (3.8x fewer weight bytes): it is the
   "checkpoint precision / FT" column of DESIGN-LB section 0, not a like-for-like kernel win.
   **At matched precision (W4A8) against ATOM's own dense linear** (`native_quant_linear`, the path ATOM serves
   gdn_in / qsa_in / out-proj with; one call per layer in a graph, launch metric): LB / ATOM 0.07-0.44, WIN in 55 of
   55 cells on both nodes. Against aiter's `gemm_a4w4` (W4A4: half the activation bytes, lower precision): 25 WIN
   (out-proj 0.56-0.95, HC down 0.33-0.41, router 0.68-0.78, head r8-32), 25 LOSS (gdn_in 1.08-1.66, qsa_in
   1.05-1.25, head r64-128, out-proj TP2 r128), 5 without a valid aiter arm (HC up, K = 320: aiter's output fails N1).
3. **Precision-matched (bf16 weights, FlyTape's regime), same harness, both sides producing the complete output
   (fold-matched):** over 4 runs (2 per node) the LB bf16 best wins in **39 cells** (WIN in every run), ties in 13,
   is mixed in 3, and loses in every run in none (phase metric). WIN: gdn_in 10 of 10 (0.69-0.83), qsa_in 10 of 10
   (0.65-0.93), head 10 of 10 (0.82-0.89), router 5 of 5 (0.86-0.93), HC up 4 of 5 (0.86-0.92). **Parity, not a
   win:** out-proj (gdn_out / qsa_out) at every row count (TIE 8, MIXED 2: 0.98-1.05; FlyTape's split-6 plan is at
   the optimum there), HC down at rows 8-64 (0.99-1.00, TIE) and 128 (MIXED), HC up at 64 rows. MIXED: gdn_out TP1
   r64 (LOSS in 3 of 4 runs, 1.05), gdn_out TP2 r8 (WIN in 1 of 4, 0.98), HC down r128 (WIN on n07 only, 0.97).
   Launch metric: 32 WIN, 10 LOSS (HC up 1.18-1.24, out-proj TP1 r8 / TP2 r8-16, HC down r64-128), 7 TIE, 6 MIXED or
   split. This replaces the first report's "44 of 55", which credited `fold=none` arms with not doing the split-K
   reduction FlyTape's plan pays (section "Fix round (W6)").
4. **Against FlyTape's in-situ phases (its traced 48-layer step, per layer, another job; 72 cells):** the LB mx best
   costs 0.18-1.36 of FlyTape's phase in the stream metric (no inter-layer sync; above 1.0 in 6 cells) and 0.32-2.03
   in this harness's phase metric (above 1.0 in 28). **The precision-matched LB bf16 best is slower than FlyTape's
   measured phase in 50 of 72 cells in the phase metric (0.58-2.66) and in 13 in the stream metric (0.21-1.70).**
   The phase metric's rocKE grid barrier costs more than FlyTape's striped sync: the FlyTape-plan replica measures
   0.63-2.71 of FlyTape's own phase in it (above 1.0 in 63 cells; TP2 out-proj and HC worst), and the replica's
   stream time, with no sync, still exceeds FlyTape's phase in 24 cells. Read the in-situ ratios with the replica
   columns next to them (section 6).
5. **Root cause found and fixed: a scaled MFMA whose scale register is rewritten for every MFMA runs about 8x
   slower on gfx950** (any producer: VALU shift, LDS read, global load, at any prefetch depth). The first mx kernel
   ran at 0.03 of the bytes floor at 128 rows for that reason. Packing four consecutive chunks' block scales into
   one dword and selecting the byte with MFMA `op_sel` (`sop=True`; one scale write per four K chunks) gave 8.5x
   on the same work with **bit-identical outputs**. Recorded in `rocke/platform/dsl_docs/optimization/arch/gfx950.md`.
6. **Fraction of the bytes floor** (fastest streamer of the same weight bytes over the arm, stream metric;
   complete-output arms, fix round runs 68232590 / 68232651): head 0.49-0.79 (mx) / 0.66-0.91 (bf16); gdn_in
   0.16-0.40 / 0.27-0.57 and qsa_in 0.14-0.35 / 0.28-0.79, falling with rows; out-proj 0.06-0.22 / 0.19-0.67;
   HC down 0.03-0.08 / 0.07-0.24, HC up 0.07-0.19 / 0.26-0.56, router 0.02-0.05 / 0.04-0.14: these are 0.7-7 MB of
   weights per layer, latency-bound, not bandwidth-bound. FlyTape's plan reaches 0.03-0.74 of the same floor. (The
   first report's higher mx out-proj / gdn_in fractions came from `fold=none` arms, whose consumer read is not timed.)
7. **What decides the GEMM core at large batch** (measured, section 5): (a) filling every CU with independent units
   (U >= 256): per-CU work beyond one unit chain does not overlap; (b) the scaled-MFMA scale writes (above); (c) the
   split-K fold is a real latency step in BOTH plans (FlyTape's plan spends 0.06-0.42 of its phase time on it:
   `ftfn / ft` 0.58-0.94), but leaving the partials to the consumer is not a measured win yet: with both sides leaving
   partials the LB `fold=none` plans win only 10 of 40 cells and lose 19 (they use more splits, S 2-80, so the
   consumer has more to read), and the consumer's read is not timed by any arm (section 7); (d) tile (MFMA operand
   order) weight layout (110 of 110 bf16 and 104 of 110 mx complete-output winners) and op_sel scales (104 of 110 mx
   winners); XCD-contiguous unit placement (3 of 220), deeper prefetch, non-temporal weight loads (7 of 220) and the
   row layout did not win. MFMA throughput becomes a co-bound for mx at 64-128 rows (8 tiles x 16x16x128 per chunk
   per wave).
8. **Against the libraries at bf16** (hipBLASLt via torch, launch metric, LB launch-best complete arm): LB / torch
   0.96-1.56 (gdn_in), 1.15-1.58 (qsa_in), 1.09-1.74 (router), 0.99-1.22 (head), 0.82-1.22 (out-proj), 0.79-1.19
   (HC down), 0.84-1.05 (HC up): 11 WIN, 36 LOSS. aiter's tuned bf16 GEMM times like hipBLASLt where it passes N1;
   it fails the bf16 tolerance at the out-proj shapes (K 6144 / 3072, 2.9-9.4x the bound) and is not timed there.
   The precision-matched LB bf16 kernel therefore trails hipBLASLt on most shapes per launch; its large-batch value is
   the persistent / phase form (no launches), the mx path, and the fusion it allows, not a faster standalone bf16 GEMM.

## 2. What was built

| file | content |
|---|---|
| `rocke/library/kernels/gfx950/qwen38_mono/large_batch/gemm.py` | E1: `GemmKnobs`, `build_gemm_kernel`, `build_stream_kernel` (bytes floor), Python engine only, names `q38lb_*` |
| `rocke/library/tests/test_qwen38_lb_gemm.py` | CPU checks: every bench knob set lowers, names, refusals, protocol text (op_sel, sc1 fold, no duplicate SSA names) |
| `rocke/platform/dsl_docs/optimization/arch/gfx950.md` | the scale-refresh fact (doc only) |
| `hackathon-prep/qwen38/truemono-lb/p-l3/` | `pl3_bench.py` (numerics suite, planner + screen + confirm, probe, timestamp diagnosis; fix round: fold-matched verdict arms, `ftfn`, op_sel mutants, `--aiter`), `pl3_ext.py` (fix round: ATOM / aiter library arms), `job.sh`, `sync_pl3.sh`, `isa_local.py`, `pl3_report.py`, `pl3_combine.py` |
| `hackathon-prep/qwen38/truemono-lb/p-l3fix/` | fix round: `job.sh` (per-GPU health check, IR identity), `sync_pl3f.sh`, `ir_identity.py`, `patch_bench.py` (the bench patch, applied), `pl3f_combine.py` (N runs, fold-matched tallies, library arms, in-situ), `summ.py` |
| `rocke/library/kernels/gfx950/qwen38_mono/large_batch/gemm.py` (fix round) | `sopsel` mutant (op_sel byte of the next chunk); existing builds' IR unchanged |

Kernel (one build per regime x K x rows bucket x knob set; M, live rows and layer count are runtime):
- `w="bf16"`: bf16 W x bf16 X on `v_mfma_f32_16x16x32_bf16` (FlyTape's operand map). `w="mx"`: MXFP4 W x MXFP8 X on
  `v_mfma_scale_f32_16x16x128_f8f6f4` (W4A8, ATOM's numerics class).
- Unit = (layer, row group of 16 x waves x mt rows, token block, K split); the persistent grid walks its units of each
  layer; weights (and scales) of chunk c + depth are issued before chunk c's MFMAs, across unit and layer boundaries;
  activations staged per CTA in LDS, double buffered, one LDS-only barrier per chunk (no vmcnt drain).
- Split-K: f32 partials with `sc1` stores, agent-scope arrival, last arriver sums in split order (`fold=last`) or
  the partials are left for the consumer (`fold=none`, DESIGN-LB `combine=fused_next` analogue).
- Runtime `gsync` flag: an 8-stripe grid barrier between layers with activation loads of a later layer gated to after
  it (the phase metric); off, layers overlap (the stream metric).
- Knobs (all default-off levers, each in the kernel name when set): `waves`, `mt`, `splits`, `tb`, `depth`, `xa`,
  `xsrc` (lds / global), `wcp` (nt), `map` (rr / xcd), `xu`, `grid`, `occ1`, `sb`, `bpre`, `fold`, `wlay` (row /
  tile), `xslay`, `xscale`, `xsb`, `wsb`, `sop`; diagnosis-only `diag`, `tstamp`, `stamp`; mutants.
- Existing builds untouched: new files only, plus one doc paragraph. The floor streamer's default name is unchanged by
  its new `vec` knob.

## 3. Method

- Cells: 7 projections x per-rank shapes (gdn_in 16480 / 8240 x 2560, out-proj 2560 x 6144 / 3072, qsa_in 13952 /
  7296 x 2560, HC down 324 x 10240, HC up 10240 x 320, router 513 x 2560, head 248320 / 124160 x 2560) x rows 8, 16,
  32, 64, 128. FlyTape's TP2 shapes from its `local_config` (qsa_in at TP2 is 7296 rows, not DESIGN-LB's 7040).
- Arms per cell: `ft` = FlyTape's plan on rocKE (bf16, 4 waves x 16 rows, its split counts per projection, depth 2,
  no cross-unit prefetch, compiler scheduling, row-major weights), `ftx` = the same plan at mx bytes (information),
  12-17 planner candidates per regime (cost-model best plans plus lever neighbours, and FlyTape's plan with LB levers),
  screened in 6 shuffled rounds; the best 2 per regime confirmed with 24 shuffled rounds next to `ft`, its second graph
  and a recompiled twin, the LB best's second graph and recompiled twin, torch bf16 (hipBLASLt) and five streamers.
- Metrics (us per layer, L layers rotating >= 4x MALL of weights, cold): **phase** (one launch, L layers, grid
  barrier between layers: phase semantics like FlyTape's), **launch** (one launch per layer in a HIP graph), **stream**
  (one launch, L layers, no barrier: throughput lower bound). Win rule of DESIGN.md 9.1 against the two controls of
  each side; verdicts on both nodes reported separately.
- Floor: the fastest of 5 read streamers (depth 4-16, 16-64 B per lane) over the same per-layer weight bytes, stream
  metric. It is a measured floor, not a datasheet one; for the head (1.3 GB bf16 per layer) the bf16 GEMM reaches up
  to 0.93 of it.
- Numerics before timing in every job, on the timed binaries; every arm of every cell checked on the cell's own data.

## 4. Numerics gate

- Reference: float64 torch matmul of the dequantized operands (MXFP4 codes x E8M0 scales; E4M3 x E8M0; bf16).
  Tolerance `max(|y - ref| - 2^-8 |ref|) / sum|x w| <= eps`, eps 2^-11 (mx: the f8f6f4 MFMA's documented 2^-12) and
  2^-14 (bf16). Observed: 0.009-0.14 eps (mx), under 0.001 eps (bf16); the float32 torch matmul's own error is
  printed next to each for scale.
- Suite (each job): 7 representative shapes x 2 regimes x 20 arms (splits 1 / mid / max, waves, mt, no prefetch,
  global X, B read-ahead, xa, no fences, xcd map, token blocks, fold none, tile layout, op_sel scales): 2-layer walk,
  partial rows (8, 16, 37, 64, 100), 3-run determinism, the grid-synced walk bitwise equal to the free one with no
  sync timeout, same-split arms bitwise equal. 91 mutant runs per job, all caught (errors 100-2,700 eps).
- Per cell: every candidate and both recompiled twins on the cell's data before timing; 0 failures in 5,096
  records over the two final runs.
- `fold=none` is checked by folding the partials on the host in split order: bitwise equal to `fold=last`.

## 5. Findings from the diagnosis rounds (ratios; absolute clocks in the private file)

1. **Grid fill dominates.** gdn_out TP2 at 128 rows (phase metric): the same mx kernel with 80 units per layer
   (2 splits) ran 4.4x the 240-unit one (6 splits); with 40 units (8 waves) 6.5x. A CU's second unit chain does not
   overlap the first.
2. **The scaled MFMA's scale register.** Per-step timestamps showed 128-row mx chunks costing 8-10x the bf16 MFMA
   work per chunk while the operand traffic was smaller. Bisection (each arm a full kernel, gdn_in TP2 128 rows, stream
   metric, over the base): constant X scale 0.11; 8 distinct loop-invariant scale registers 0.11; one scale address
   for every tile 0.85; wave-uniform scale 0.93; scales staged as 16-byte rows (`xslay=chunk`) 0.99; scales loaded per
   wave from global with the weights 1.28; pre-unpacked scale bytes from LDS (no VALU op) 1.33, from global (no VALU
   op) 1.39; weight scale constant 1.13; all scale values 127 (data only) no change. Only scale registers that are not
   rewritten between MFMAs ran fast. `sop` (op_sel bytes, one write per 4 chunks): 0.116-0.118 of the base, same bits.
3. **Weight layout.** Tile layout (16 rows x 128 K blocks in operand order): 0.96-0.99 of row-major on its own;
   the op_sel scale layout needs it.
4. **Fold.** The last-arriver fold costs about half of a short unit (timestamps: partial stores, drain, arrival,
   sc1 reads); `fold=none` removes it from the GEMM (the consumer reads S partials instead of one row).
5. **What lost or did not matter** (winners of the final rounds, 110 cell-runs per regime): XCD-contiguous unit
   ranges (`map=xcd`: 0 of 110 mx, 3 of 110 bf16 winners; neutral to slightly worse in the probes), deeper weight
   prefetch past 4 chunks (mx winners all depth 4), more activation chunks in flight (`xa=2`: 6 of 110), B-operand
   read-ahead for mx (0 of 110; bf16 winners read 1-2 tiles ahead), `nt` weight loads (6 of 110), row-major weights
   (5 of 110 mx).
6. **rocKE naming bug (worked around, not fixed):** a 1-element `smem_load_vN` (name hint `av1`) and a 16-element
   one (`av16`) can produce the same SSA name plus counter (`av1` + `6804` = `av16` + `804`) in large kernels:
   invalid IR at COMGR. `gemm.py` loads single LDS words through `smem_load_vN_f32`; the test asserts unique names.

## 6. Results (fix round: 4 runs, 2 per node; per-cell tables in `P-L3-fix-tables.md`, first round in `P-L3-tables.md`)

Same harness, phase metric unless stated. "complete" = both sides produce the complete bf16 output (`fold=last` or one
split); "fold-matched" = both sides leave their split-K partials (LB best `fold=none` arm vs FlyTape's plan with
`fold=none`, `ftfn`). Verdicts per cell: WIN / LOSS only when every run on both nodes agrees, MIXED when the nodes
disagree.

| projection | mx best / ft | bf16 best / ft (complete) | bf16 verdicts (cells) | LB fn / ftfn (fold-matched) | fold-matched verdicts | ftfn / ft |
|---|---|---|---|---|---|---|
| gdn_in (TP1, TP2) | 0.34-0.51 | 0.69-0.83 | WIN 10 | 0.78-1.02 | WIN 3, TIE 3, MIXED 4 | 0.75-0.87 |
| out-proj (gdn_out = qsa_out, TP1, TP2) | 0.70-0.83 | 0.98-1.05 | TIE 8, MIXED 2 | 1.10-1.59 | LOSS 10 | 0.75-0.94 |
| qsa_in (TP1, TP2) | 0.35-0.63 | 0.65-0.93 | WIN 10 | 0.90-1.16 | WIN 2, TIE 2, MIXED 1, LOSS 5 | 0.69-0.84 |
| HC down | 0.65-0.75 | 0.97-1.00 | TIE 4, MIXED 1 | 0.92-1.37 | WIN 2, TIE 1, LOSS 2 | 0.65-0.79 |
| HC up | 0.69-0.83 | 0.86-0.99 | WIN 4, TIE 1 | - (FlyTape's plan has one split) | - | - |
| router | 0.61-0.86 | 0.86-0.93 | WIN 5 | 0.87-1.56 | WIN 3, LOSS 2 | 0.58-0.82 |
| head (TP1, TP2) | 0.30-0.36 | 0.82-0.89 | WIN 10 | - (one split) | - | - |
| **total (55 cells)** | **WIN 55** | | **WIN 39, TIE 13, MIXED 3, LOSS 0** | | **WIN 10, TIE 6, MIXED 5, LOSS 19 (40 cells)** | |

Launch metric (complete): bf16 32 WIN, 10 LOSS, 7 TIE, 6 MIXED / split; mx 49 WIN, 5 LOSS (HC up), 1 MIXED. Stream
metric (complete): bf16 37 WIN, 1 LOSS, 8 TIE, 9 MIXED / split; mx 50 WIN, 2 LOSS, 2 MIXED, 1 TIE. Controls: the
FlyTape-plan A-vs-A band (second graph and recompiled twin) 0.975-1.027 (medians over cell-runs; extremes 0.93-1.09);
LB recompiled twin / LB arm 0.98-1.03. Fragile verdicts (median within 0.03 of the band edge) are listed per run in
`P-L3-fix-tables.md`; with two runs per node, 2 bf16 phase cells split between the runs of one node (both on n01:
gdn_out TP1 r64 LOSS / TIE, gdn_out TP2 r8 WIN / TIE), which is why they count as MIXED, not as a verdict.

Bytes floor (stream metric, complete-output arms, runs 68232590 / 68232651): floor / arm

| projection | mx best | bf16 best | ft |
|---|---|---|---|
| gdn_in | 0.16-0.40 | 0.27-0.57 | 0.18-0.45 |
| out-proj | 0.06-0.22 | 0.19-0.67 | 0.18-0.60 |
| qsa_in | 0.14-0.35 | 0.28-0.79 | 0.28-0.55 |
| HC down | 0.03-0.08 | 0.07-0.24 | 0.07-0.23 |
| HC up | 0.07-0.19 | 0.26-0.56 | 0.24-0.50 |
| router | 0.02-0.05 | 0.04-0.14 | 0.03-0.13 |
| head | 0.49-0.79 | 0.66-0.91 | 0.54-0.74 |

Libraries (one call per layer in a HIP / torch graph: launch metric; LB arm = the complete-output arm fastest in the
launch metric; ATOM / aiter arms in the q38 container, jobs 68232593 n01-09 and 68232821 n07-17, against the LB arm
compiled in the same job; torch in the plan-of-record runs). Every library arm passes N1 on the cell's own operands
before it is timed (layers 0 and L-1), else it is not timed:

| projection | LB mx / ATOM W4A8 (`native_quant_linear`) | LB mx / aiter `gemm_a4w4` (W4A4) | LB bf16 / torch bf16 (hipBLASLt) |
|---|---|---|---|
| gdn_in | 0.12-0.33, WIN 10 | 1.08-1.66, LOSS 10 | 0.96-1.56, LOSS 9, TIE 1 |
| out-proj | 0.10-0.44, WIN 10 | 0.56-1.05, WIN 9, LOSS 1 (TP2 r128) | 0.82-1.22, WIN 3, TIE 2, LOSS 2, MIXED 3 |
| qsa_in | 0.10-0.38, WIN 10 | 1.05-1.25, LOSS 10 | 1.15-1.58, LOSS 10 |
| HC down | 0.22-0.29, WIN 5 | 0.33-0.41, WIN 5 | 0.79-1.19, WIN 4, LOSS 1 |
| HC up | 0.33-0.41, WIN 5 | aiter output fails N1 (K = 320) | 0.84-1.05, WIN 4, LOSS 1 |
| router | 0.09-0.35, WIN 5 | 0.68-0.78, WIN 5 | 1.09-1.74, LOSS 5 |
| head | 0.07-0.13, WIN 10 | 0.80-1.31, WIN 6 (r8-32), LOSS 4 (r64-128) | 0.99-1.22, LOSS 8, TIE 2 |

- ATOM serves the dense linears (gdn_in, qsa_in, out-proj) W4A8 through `native_quant_linear` (MXFP4 weights unpacked
  to bf16 for a plain MFMA, Triton, its own split-K and reduce); that is the matched-precision baseline of the mx
  regime. ATOM's head and HC mix weights are bf16 (BASELINE caveats), so the ATOM W4A8 column there is ATOM's kernel at
  the LB shape, not what ATOM serves; the bf16 column is the ATOM-precision reference for those.
- aiter's `gemm_a4w4` reads half the activation bytes at a lower precision (FP4 activations; ATOM measured FP4
  activations 3.3 GSM8K points below FP8 on this model), so it is information, not a matched baseline. aiter's tuned
  bf16 GEMM (`tgemm.mm`) times like hipBLASLt where it passes N1 and fails the bf16 tolerance at the out-proj shapes
  (2.9-9.4x the bound; not timed there).

FlyTape in-situ (per layer from its traced step, medians of the 4 runs; complete-output arms; see the caveat in 1.4):

| projection | TP | mx phase | mx stream | bf16 phase | bf16 stream | replica phase | replica stream |
|---|---|---|---|---|---|---|---|
| gdn_in | 1 | 0.34-0.43 | 0.20-0.25 | 0.77-0.87 | 0.58-0.65 | 1.00-1.15 | 0.79-0.84 |
| gdn_in | 2 | 0.59-0.86 | 0.35-0.44 | 1.05-1.27 | 0.78-0.85 | 1.41-1.73 | 1.15-1.24 |
| out-proj | 1 | 0.78-0.97 | 0.41-0.53 | 1.08-1.35 | 0.52-0.68 | 1.09-1.39 | 0.57-0.69 |
| out-proj | 2 | 1.39-2.03 | 0.69-1.36 | 1.70-2.66 | 0.76-1.36 | 1.70-2.71 | 0.81-1.41 |
| qsa_in | 1 | 0.48-0.60 | 0.28-0.34 | 0.89-1.03 | 0.55-0.64 | 1.23-1.41 | 0.80-0.85 |
| qsa_in | 2 | 0.79-0.95 | 0.40-0.61 | 1.20-1.50 | 0.67-0.91 | 1.45-1.68 | 0.76-0.88 |
| HC down | 1, 2 | 1.22-1.30 | 0.53-1.03 | 1.62-1.93 | 0.65-1.70 | 1.64-1.92 | 0.69-1.70 |
| HC up | 1, 2 | 1.06-1.34 | 0.30-0.51 | 1.26-1.54 | 0.39-0.51 | 1.33-1.74 | 0.42-0.54 |
| router | 1, 2 | 0.53-0.60 | 0.18-0.30 | 0.58-0.82 | 0.21-0.66 | 0.63-0.95 | 0.21-0.95 |
| head | 1 | 0.32-0.39 | 0.28-0.35 | 0.93-1.08 | 0.88-1.04 | 1.08-1.21 | 1.06-1.16 |
| head | 2 | 0.36-0.45 | 0.25-0.35 | 0.89-1.07 | 0.81-0.96 | 1.03-1.27 | 0.94-1.18 |

Cells above 1.0 (slower than FlyTape's measured phase), of 72: mx phase 28, mx stream 6, **bf16 phase 50, bf16 stream
13**, replica phase 63, replica stream 24. FlyTape's phases include work the GEMM core does not do (GDN conv / gates,
the HC combine epilogue, router top-k, head argmax, TP pushes), and FlyTape prefetches across its phase boundary; the
stream columns have no inter-layer sync at all. The tile-flag dataflow of DESIGN-LB section 5 sits between the phase
and stream columns. The replica (FlyTape's plan in rocKE) is slower than FlyTape itself even in the stream metric in
24 cells (gdn_in TP2, out-proj TP2, HC down 32+ rows, head TP1), so the same-harness `ft` arm understates FlyTape there:
the same-harness bf16 wins are wins over the replica, and against FlyTape's own measured phases the precision-matched
LB kernel is slower in most cells under phase semantics.

## 7. What this means for DESIGN-LB

- E1 as designed (weights as A, batch as N, LDS-staged activations, weights ahead across units) works; the plan model
  of section 4.3 must put grid fill first (units >= CTAs per layer) and take K splits over token blocks.
- The MX path needs op_sel-packed scales for every scaled-MFMA stage with more than one accumulator per wave: E1,
  the MoE experts (E7), the head (E10). Their producers must write activation scales in the [chunk / 4][token][block]
  [4 chunks] layout and the checkpoint loader must repack weight scales once (bytes unchanged).
- `fold=none` (consumer-side fold, `combine=fused_next`) stays a default-off knob, NOT the default (restated in the fix
  round). The split-K fold costs FlyTape's plan 0.06-0.42 of its phase time, so moving it out of the GEMM is worth
  measuring, but with both sides leaving partials the LB `fold=none` plans lose 19 of 40 cells (out-proj all 10:
  1.10-1.59; they use S 2-80 splits against FlyTape's 2-40) and win 10. The consumer must then read S f32 partials per
  output (for HC down at S = 80 and 64 rows that is as many bytes as the layer's bf16 weights). Decide it in situ
  (P-L2 / WP-L3: the consumer's read + sum timed in the same arm, against `fold=last` and against fewer splits), not
  from this probe.
- The small GEMMs (HC down / up, router: 0.7-7 MB per layer) stay latency-bound at every row count (0.02-0.24 of the
  floor): their cost is hops and sync, which is P-L2's and the dataflow's business, not the GEMM core's.
- Precision-matched, the LB bf16 kernel ties FlyTape's plan on out-proj and HC down and trails hipBLASLt per launch on
  most shapes, and its phase time is above FlyTape's measured in-situ phase in 50 of 72 cells: the large-batch case for
  rocKE rests on the mx path (55 of 55 over FlyTape's plan; 0.07-0.44 of ATOM's own W4A8 linear), the persistent form
  and the fusion, not on a faster bf16 GEMM. A faster grid barrier (FlyTape's striped sync, cross-phase prefetch) is
  needed before phase-semantics comparisons can go our way at bf16.
- aiter's W4A4 `gemm_a4w4` beats the LB mx kernel on gdn_in / qsa_in (1.05-1.66) and the head at 64-128 rows: if a
  W4A4 path passes the accuracy suites (user decision 1), the `act=fp4` knob must beat it there.
- MFMA throughput is a co-bound for mx at 64-128 rows; the W4A4 knob (`act=fp4`, fp4 x fp4 MFMA rate) and 32x32x64
  shapes are the next E1 levers, together with a faster grid barrier for the phase-semantics fallback.

## 8. Caveats

- Same-harness ratios (section 6 first table) compare kernels on the same synthetic data, cold weights, hot
  activations; the in-situ table compares across jobs and against a whole-step trace (FlyTape's semantics).
- `mx / ft` mixes precision and bytes by design (checkpoint precision vs FlyTape's); the precision-matched claim is the
  bf16 column.
- The phase metric's grid barrier is this probe's (8 stripes, one returning atomic, all-lane poll with `s_sleep 1`);
  it costs several microseconds more per layer than FlyTape's striped sync, so phase ratios over FlyTape's in-situ
  phase are pessimistic for both the LB arms and the replica.
- The floor is a measured streamer at 256 x 512 threads; it is not a vendor peak.
- Neighbour load: each final job shared its node with 1-3 other jobs (logged per job); ratios agree between nodes
  within the A-vs-A bands. The same arm pair can move by up to 0.08 between jobs on one node (verification round 2),
  hence the fix round's rule: a verdict counts only when all 4 runs (2 per node) agree.
- `fold=none` arms are never compared with a `fold=last` arm in a verdict any more (fix round); their bytes-floor
  fractions are not reported, because the consumer's read of the partials is not timed.
- The library arms (ATOM, aiter) run in the q38 container, whose rocKE build uses the image's COMGR (ROCm 10), not the
  venv's llvm22 COMGR of the plan-of-record runs; their ratios use the LB arm compiled and timed in the same job. The
  container runs' own bf16 / ft verdicts agree with the venv runs (WIN 37, TIE 15, MIXED 2, LOSS 1 over 2 runs).

## 9. Rerun

From `hackathon-prep` (Git Bash), `CL_HOST=vanantha@ctr2-alola-login-01`, `Q38_MAX_GPUS=10`:

```bash
bash qwen38/truemono-lb/p-l3/sync_pl3.sh both
CELLS=$(for p in gdn_in:1 gdn_in:2 gdn_out:1 gdn_out:2 qsa_in:1 qsa_in:2 hc_down:0 hc_up:0 router:0 head:1 head:2; do for r in 8 16 32 64 128; do printf "%s:%s," $p $r; done; done)
Q38_NODE=smci355-ccs-aus-n01-09 bash qwen38/truemono-lb/submit.sh logs/q38/w4/pl3-R3a.poll --mode venv --gpus 1 --cpus 32 --mem 240 \
  --time 02:00:00 --tag w4-pl3-R3a --stage w4-pl3-rocke --stage w4-pl3-h \
  --env "PL3_ARGS=--phase numerics,timing --cells ${CELLS%,} --rounds 24 --screen-rounds 6 --top 2 --workers 24 --seed 7303" \
  -- 'bash $Q38_STAGES/w4-pl3-h/job.sh'
# FlyTape traces (container, 2 GPUs): --stage w4-pl3-h --stage flytape --stage w3-lbh --env PL3_ARGS=skip --env FT_WHEN=before \
#   --env "FT_RUNS_TP1=tp1-trace:mxfp4:8x128,16x128,32x128,64x128:1:trace" --env "FT_RUNS_TP2=tp2-trace:mxfp4:8x128,16x128,32x128,64x128,128x128:1:trace"
python qwen38/truemono-lb/p-l3/pl3_combine.py --a <R3a>/pl3.json --b <R3b>/pl3.json --ft <D>/ft-pre1/ft/tp1-trace.jsonl \
  <D>/ft-pre2/ft/tp2-trace.jsonl --out-ratio P-L3-tables.md --out-abs <private>.md
```

A full 55-cell job (numerics suite, about 900 kernels compiled in a 24-process pool, screening and confirmation) runs
in about 4 minutes of one GPU (about 6 minutes with the fix round's fold-matched arms).

Fix round (fold-matched verdicts, op_sel mutants, library arms; `Q38_MAX_GPUS=8`). `p-l3/pl3_bench.py` is the patched
bench (sha 25b6e252... in jobs 68232590 / 68232651 / 68233016 / 68232593 / 68232821; the later sha only adds the
library-arm wrong-layer control). On n07-17 ask for 2 GPUs: `job.sh` health-checks each GPU in its own process and runs
on the first healthy one (the bad GPU was in every n07-17 allocation of this round: bus e5 in all three 2-GPU jobs,
and both 1-GPU jobs failed the health check with its illegal-address signature).

```bash
bash qwen38/truemono-lb/p-l3fix/sync_pl3f.sh both     # stages w6-pl3f-h, w6-pl3f-rocke
# per node and run: venv (plan of record; PL3_IR=1 + --stage w4-pl3-rocke diffs the LLVM IR of existing builds)
Q38_NODE=smci355-ccs-aus-n07-17 bash qwen38/truemono-lb/submit.sh logs/q38/w6/pl3f-b2.poll --mode venv --gpus 2 --cpus 32 \
  --mem 240 --time 01:00:00 --tag w6-pl3f-b2 --stage w6-pl3f-rocke --stage w6-pl3f-h --stage w4-pl3-rocke --env PL3_IR=1 \
  --env "PL3_ARGS=--phase numerics,timing --cells ${CELLS%,} --rounds 24 --screen-rounds 6 --top 2 --workers 24 --seed 7312" \
  -- 'bash $Q38_STAGES/w6-pl3f-h/job.sh'
# library arms: --mode container --stage atom-rkrepro, and add --aiter to PL3_ARGS
python qwen38/truemono-lb/p-l3fix/pl3f_combine.py --runs <a>/run/pl3.json <c>/run/pl3.json <b2>/run/pl3.json <d>/run/pl3.json \
  --ext <xa>/run/pl3.json <xb2>/run/pl3.json --ft <D>/ft-pre1/ft/tp1-trace.jsonl <D>/ft-pre2/ft/tp2-trace.jsonl \
  --out-ratio P-L3-fix-tables.md --out-abs <private>/P-L3-fix.md
```

## Verification (independent verifier, W4 vP-L3, 2026-10-31)

Verdict: **partial**. The numerics gate is real and holds. The precision-matched (bf16) "44 of 55 wins vs FlyTape's
plan" headline does not hold as stated: in the cells whose best arm uses `fold=none`, the win comes from not charging
the split-K reduction that the baseline pays. Matched like for like, the LB arm ties or loses there. The mx
results and the scale finding stand as reported.

What was checked:
- **Numerics gate ran on hardware, with mutants that fail.** R3a and R3b `pl3.json`: 2 x 2,548 records, 0 failures.
  Each run's suite has 91 mutant runs (42 kskip, 28 fold, 21 scale), all caught, none failed to compile. The
  numerics suite runs before the first timed cell (log order). Every confirm arm and both recompiled twins are
  checked on the cell's own data. The worst error is 0.14 eps. The suite applies mutants only to the default-path
  arms, not to the op_sel (`sop`) path. N2 still covers that path: `s1_sop` must be bitwise equal to the mutant-tested
  `s1`.
- **Re-run, job 68231981 `w4-vP-L3-a`** (n01-09, GPU bus 65; the builder's stage `w4-pl3-rocke`, an unmodified
  `pl3_bench.py`, harness `hackathon-prep/qwen38/truemono-lb/v-pl3/`):
  - Numerics suite (`--quick`): 92 of 92 arms pass and 26 of 26 mutants are caught (kskip / scale / fold,
    340-2,378 eps).
  - Probe, all 4 cells: the 8 arms pass N1. `fold=none` folded on the host is bitwise equal to `fold=last`, for
    both the replica (`ftfn` = `ft`) and the LB plan.
  - Same-harness ratios reproduce the builder's within 0.02 (phase): LB `fold=none` / ft = 0.780 vs reported 0.781
    (hc_down r32); 0.866 vs 0.846 (r64); 0.864 vs 0.871 (gdn_out TP2 r8); 0.958 vs 0.950 (r16).
- **Interleaving and controls:** shuffled arm order and shuffled metric order per round. The second graph and the
  recompiled twin serve as controls on both sides. The win rule uses the most extreme per-round control ratio
  (strict). The best arm is picked from the 6 screening rounds and judged on the 24 independent confirm rounds, so
  there is no selection on the confirm data. The arms are unstamped. Node and bus are recorded. The current local
  `gemm.py` differs from the measured stage (sha 85404ab7...) only in the module docstring (diffed against the
  stage tar).

What is refuted or unsupported:
1. **`fold=none` accounting (decisive, measured).** With `fold=none` the GEMM leaves S f32 partials, and the
   consumer that sums them is not timed. The `ft` arm is always `fold=last`, so it pays its fold. 19 of 55 bf16 best
   arms (and 65 of 110 mx cell-runs) are `fold=none`, some with S = 80 (hc_down) or 24 (out-proj). For hc_down S80
   at 64 rows, the partials the consumer must read are the same size as the layer's bf16 weights. The re-run timed
   both sides with the same fold, phase metric, using the win rule with the replica's and the LB arm's controls:

   | cell | LB fnone / ft fnone | LB fold=last / ft fold=last | as reported: LB fnone / ft fold=last |
   |---|---|---|---|
   | hc_down r32 | 0.995 TIE | 1.717 LOSS | 0.780 WIN |
   | hc_down r64 | 1.296 LOSS | 1.990 LOSS | 0.866 WIN |
   | gdn_out TP2 r8 | 1.122 LOSS | 1.253 LOSS | 0.864 WIN |
   | gdn_out TP2 r16 | 1.208 LOSS | 1.286 LOSS | 0.958 WIN |

   - Stream metric, fnone vs fnone: 0.96-1.03 (TIE, with one WIN at gdn_out r8).
   - Launch metric, fnone vs fnone: 1.16-1.31 (LOSS).
   - The replica's own fold costs 21-33 % of its phase time (`ftfn / ft` 0.67-0.79). That fully explains the
     reported wins in these cells.
   - The builder's own screening data (6 rounds, both nodes) agrees. The best `fold=last` LB arm over ft is
     0.99-1.02 for hc_down r8-64 and 0.94-0.99 for gdn_out TP1 / TP2 r8-16, which is parity.
   - Other `fold=none` cells (gdn_in TP2 r32-128, qsa_in, router r8-64) keep a smaller `fold=last` margin over ft
     (0.65-0.98). They were not re-measured against `ft` with `fold=none`.
   - So the bf16 tally must be recounted with matched fold accounting. At least 4 of the 44 "wins" (likely
     8: hc_down r8-64, out-proj r8-16) are ties or losses.
2. **The replica is weaker than the real FlyTape in about a third of cells, and the report hides the precision-matched
   in-situ comparison.**
   - `P-L3-tables.md` (in-situ section): the replica's stream time, with no sync at all, exceeds FlyTape's full
     in-situ phase in 24 of 72 cells: gdn_in TP2, out-proj TP2, hc_down r32+, head TP1.
   - The LB bf16 best's phase time exceeds FlyTape's in-situ phase in 47 of 72 cells: gdn_in TP2 1.11-1.25, out-proj
     TP1 1.06-1.27, out-proj TP2 1.73-2.65, qsa_in TP2 1.15-1.38, HC down 1.18-1.62, HC up 1.25-1.50, head r64-128
     1.02-1.05.
   - Its stream time, also with no sync, exceeds FlyTape's phase in 7 cells.
   - The harness's slower grid barrier and FlyTape's cross-phase prefetch are real caveats.
   - Even so, the section 6 in-situ table above shows only the mx and replica columns. The summary claims
     precision-matched wins without saying that, against FlyTape's measured phases, the precision-matched LB kernel
     is slower in most cells.
3. **The DESIGN-LB P-L3 row asks for a comparison "vs hipBLASLt / aiter at the same shapes".** hipBLASLt bf16 is
   there; aiter is not. The mx (W4A8, ATOM's precision) regime therefore has no precision-matched external
   baseline: no aiter MXFP4 GEMM at these shapes. Every mx "win" compares different precisions, either against
   FlyTape's bf16 plan or against torch bf16. The report labels this correctly, but the gate's comparison set is
   incomplete.
4. **Not affected:**
   - The numerics gate.
   - Scaled-MFMA scale-register finding and the `sop` fix (bitwise-equal outputs, timed in diagnosis jobs).
   - The grid-fill finding.
   - The bytes-floor fractions, except for `fold=none` arms, where the uncharged consumer read makes them optimistic.
   - The hipBLASLt bf16 comparison (launch metric). It is unflattering and stands.
   - PLAN decision 5's DP2 arm is not applicable to a single-GPU kernel probe. The TP1 shapes cover the
     per-replica GEMMs.

Required to close: re-tally the bf16 (and mx) verdicts with matched fold accounting. Either time `ft` with
`fold=none` next to every `fold=none` LB arm, or charge the consumer fold. Restate summary item 3 accordingly. Add
the bf16 columns to the section 6 in-situ table. Add an aiter MXFP4 GEMM arm (container) for the mx regime, or mark
that gate item open.

## Verification round 2 (independent verifier, W6 vP-L3, 2026-10-31)

Verdict: **partial** (unchanged from round 1). The W6 builder re-reported `pass` without new work: this file is
unchanged since round 1 (summary item 3 still says 44 wins), no fix-round job ran (the W4 fix-round harness
`hackathon-prep/qwen38/truemono-lb/p-l3fix/` was written but never staged or submitted before the restart), and the
three items "required to close" above are still open. This round measured the open item 1 on both nodes.

Re-run: jobs 68232466 `w6-vP-L3-a` (n01-09, GRES idx 3, GPU bus 75) and 68232467 `w6-vP-L3-b` (n07-17, GRES idx 0,
GPU bus 05; not the bad e5 GPU), each with a health precheck (alloc + H2D + sync) that passed. Both used the builder's
measured stage `w4-pl3-rocke` (`gemm.py` sha 85404ab7...) and an unmodified copy of the measured `pl3_bench.py` (sha
8104b795...), `--phase probe`, 24 shuffled rounds. Harness: `hackathon-prep/qwen38/truemono-lb/v-pl3-w6/`
(`make_arms.py`, `job.sh`, `verdicts.py`). Absolute data: `private-perf/.../large-batch/vp-l3-raw/w6-vP-L3-{a,b}/`.

**1. Fold-matched bf16 re-tally (decisive, measured).** The 13 bf16 cells whose reported winner leaves `fold=none`
partials and whose margin looked thin were re-timed. Each cell has four arms, each with a second-graph control and a
recompiled twin: `ft` (FlyTape-plan replica, folds), `lbc`, `ftfn` (`ft` leaving its partials) and `lbfn` (the
reported winner). `lbc` is the fastest complete-output LB arm (`fold=last` or one split) of the builder's own 6-round
screen, so it is selected on the builder's data and judged on fresh rounds. Every arm passed N1 and the same-as checks
first. The win rule is `pl3_bench.win_rule`, with both sides' controls.

| cell | builder's confirm: winner / ft (phase, both runs) | **lbc / ft, phase (n01; n07)** | lbfn / ftfn, phase (n01; n07) | ftfn / ft |
|---|---|---|---|---|
| gdn_in TP2 r64 | 0.75-0.77 WIN | 0.858 WIN; 0.873 WIN | 0.89-0.90 WIN | 0.84 |
| gdn_out TP1 r8 | 0.89-0.91 WIN | 0.999 TIE; 1.007 TIE | 1.12 LOSS | 0.82 |
| gdn_out TP1 r16 | 0.91 WIN (n07: complete arm, 0.99 TIE) | 0.977 TIE; 0.980 TIE | 1.21-1.22 LOSS | 0.81 |
| gdn_out TP2 r8 | 0.87 WIN | 0.970 TIE; 0.969 WIN | 1.13-1.14 LOSS | 0.77 |
| gdn_out TP2 r16 | 0.92 WIN (n07: complete arm, 0.98 TIE) | 1.004 TIE; 0.999 TIE | 1.23-1.25 LOSS | 0.78 |
| qsa_in TP2 r32 | 0.82-0.92 WIN | 0.962 TIE; 0.967 TIE | 1.16 LOSS | 0.73 |
| qsa_in TP2 r64 | 0.96 WIN (n07: complete arm, 0.97 TIE) | 0.944 WIN; 0.940 WIN | 1.12 LOSS | 0.82-0.83 |
| hc_down r8 | 0.72 WIN | 0.992 TIE; 0.996 TIE | 0.91-0.92 WIN | 0.79 |
| hc_down r16 | 0.74-0.75 WIN | 0.992 TIE; 0.997 TIE | 0.95 WIN; TIE | 0.80 |
| hc_down r32 | 0.78 WIN | 0.987 TIE; 0.989 TIE | 1.00-1.02 TIE | 0.78 |
| hc_down r64 | 0.82-0.87 WIN | 1.013 TIE; 1.012 TIE | 1.27-1.30 LOSS | 0.67-0.69 |
| router r8 | 0.72-0.74 WIN | 0.947 WIN; 0.936 WIN | 0.89-0.90 WIN | 0.80-0.82 |
| router r64 | 0.74 WIN | 0.881 WIN; 0.880 WIN | 1.10-1.11 LOSS | 0.67 |

- The reported comparison mostly reproduces. `lbfn / ft` (same knobs as the builder's `bf16_fnone`) is within 0.03 of
  the builder's confirm median in 18 of 23 node-cell pairs. The other 5 drift by 0.04-0.08 between jobs:
  - gdn_out TP1 r16, gdn_out TP2 r16 and hc_down r64 on n01, all slower now;
  - qsa_in TP2 r64 on n01 and qsa_in TP2 r32 on n07, both faster now.
  So single-job verdicts near the band edge (gdn_out TP2 r8: 0.969 / 0.970) are fragile. FlyTape A-vs-A band in these
  jobs: 0.96-1.09.
- Corrected bf16 phase tally (55 cells, complete output on both sides):
  - **38 WIN, 13 TIE, 4 MIXED, 0 LOSS**, against the reported 44 / 5 / 6 / 0.
  - 6 cells move from WIN to TIE: gdn_out TP1 r8, hc_down r8-64 and qsa_in TP2 r32.
  - gdn_out TP2 r8 moves from WIN to MIXED.
  - 2 MIXED cells become TIE: gdn_out TP1 r16 and TP2 r16.
  - qsa_in TP2 r64 moves from MIXED to WIN.
  - 7 more cells have a `fold=none` winner on at least one node: gdn_in TP2 r32 / r128, qsa_in TP1 r8, qsa_in TP2
    r16 / r128 and router r16 / r32. They stay WIN on the builder's screen data: their best complete arm is 0.65-0.93
    of `ft`. They were not re-confirmed.
- The out-proj (gdn_out / qsa_out) column is at parity with FlyTape's plan at every row count, not "WIN 7". HC down
  is at parity at rows 8-64.
- With fold matched, the bf16 launch metric adds LOSSes: `lbc / ft` is 1.01-1.23 at gdn_out r8 / r16, hc_down r64 and
  router r8.
- `lbfn / ftfn`: when both sides leave partials, the LB `fold=none` plans lose in 8 of 13 cells (on both nodes). They use more splits
  (S 20-80 vs 6-40), so the consumer also has more to read. `fold=none` wins only by moving the fold out of the timed
  kernel. Section 7's "`fold=none` should be the default" needs the consumer's cost measured first. P-L2 or the
  dataflow is the place to measure it.
- mx: the best complete-output mx arm is 0.30-0.86 of `ft` in the builder's screen (both nodes, all 55 cells), far
  outside the control band. So "mx WIN in 55 of 55" survives fold matching (screen data, not re-confirmed). It is still
  precision-mixed, as the report says.

**2. op_sel path mutants and the unfinished W4 fix-round edit (measured).** The current local `gemm.py` adds a `sopsel`
mutant: the op_sel byte is moved to the next chunk's. It is the only change from the measured file. In the jobs it was
placed over a copy of the measured tree.
- LLVM IR of 14 existing builds (incl. sop / fold=none / mutant builds) and 5 streamers: **0 differing** (`ir_identity.py`).
- On hardware, at gdn_in TP1 r64, gdn_in TP2 r128, hc_down r37 and router r8, with one split and with 5 splits + mt 2:
  - the clean op_sel arms pass (0.011-0.040 eps);
  - `sopsel` fails N1 at 538-1,483 eps;
  - `kskip` on the op_sel path fails N1 at 101-780 eps.
  - Both nodes give the same values.
- This closes round 1's note that no mutant covered the op_sel path. The builder's suite still does not run these
  mutants: the patched `pl3_bench.py` is local only.

**Still open (required to close):**
- (a) Restate summary items 2-3, the section 6 verdict column and section 7's `fold=none` default with the corrected
  tally above. Better: rerun the confirm round with `fold=none` arms compared only to `ftfn`; that is what the local
  patched bench does.
- (b) Add the bf16 columns to the in-situ table. Round 1 found the bf16 best slower than FlyTape's in-situ phase in
  47 of 72 cells.
- (c) Add an aiter MXFP4 GEMM arm (container) for the mx regime, or mark that DESIGN-LB P-L3 comparison item open.

Not affected:
- the numerics gate (T0, the DESIGN-LB P-L3 gate), which is met;
- the scaled-MFMA scale-register finding and the `sop` fix;
- the grid-fill finding;
- the bytes-floor fractions of complete-output arms;
- the hipBLASLt comparison.

## Fix round (W6 builder, 2026-10-31)

Answers verification rounds 1 and 2 item by item. Everything above (status, summary, sections 6-9) is restated from
this round; the two verification sections are kept as written.

**Jobs.** Plan of record (venv, llvm22 COMGR, 55 cells each, numerics suite + per-cell numerics before timing, 6-round
screen, 24-round confirm): 68232590 `w6-pl3f-a` (n01-09, GPU bus e5 of n01-09, a healthy GPU), 68232959 `w6-pl3f-c`
(n01-09, bus 85), 68232651 `w6-pl3f-b2` (n07-17, bus f5), 68233016 `w6-pl3f-d` (n07-17, bus 75). Library arms (q38
container, ATOM stage `atom-rkrepro` = upstream 352229d6, aiter of the image): 68232593 `w6-pl3f-xa` (n01-09, bus 95),
68232821 `w6-pl3f-xb2` (n07-17, bus 15); library-arm check control 68233558 `w6-pl3f-xn1` (n01-09, bus 15). Every job health-checks each allocated GPU in its own process first and runs
on a healthy one; n07-17's bad GPU (bus e5) was in every n07-17 allocation and was skipped (68232591 and 68232601,
1-GPU, got only that GPU and stopped at the health check). Stages `w6-pl3f-rocke` (`gemm.py` sha 39e27905..., the
measured file plus the `sopsel` mutant) and `w6-pl3f-h` (harness). Harness: `hackathon-prep/qwen38/truemono-lb/p-l3fix/`
(`job.sh`, `sync_pl3f.sh`, `ir_identity.py`, `pl3f_combine.py`, `summ.py`) and the patched `p-l3/pl3_bench.py`,
`p-l3/pl3_ext.py`. Absolute data: `private-perf/qwen38-true-mono/large-batch/P-L3-fix.md` and `p-l3f-raw/`.

**1. Fold-matched verdicts (verification 1 item 1, round 2 items 1-3).** The bench now separates the two accountings
in every cell instead of re-timing a subset:
- The verdict arm of each regime is the fastest **complete-output** LB arm of the cell's screen (`fold=last` or one
  split), judged against `ft` (FlyTape's plan, which folds) with both sides' controls (second graph, recompiled twin).
- The fastest `fold=none` arm is judged only against `ftfn` (FlyTape's plan leaving its own partials, with its own
  second graph and recompiled twin), in the same shuffled confirm rounds. `ftfn` folded on the host is bitwise equal to
  `ft` (checked per cell before timing).
- A verdict counts only when all 4 runs (2 per node) agree; else MIXED (nodes disagree or a node's runs disagree).

Result (phase, complete output): **bf16 39 WIN, 13 TIE, 3 MIXED, 0 LOSS** (verification round 2's re-tally of the
first round's data: 38 / 13 / 4 / 0). Out-proj is at parity at every row count (TIE 8, MIXED 2) and HC down at rows
8-64 (TIE), as the verifier found. The verifier's re-timed cells reproduce within 0.03 (gdn_out TP1 r8 0.98-1.01, TP2
r16 1.00, hc_down r8-64 0.98-1.01, router r8 0.92-0.93, r64 0.88, gdn_in TP2 r64 0.83-0.84); qsa_in TP2 r32 is a WIN
here in all 4 runs (0.91-0.92, a 4-split complete arm this round's screen found) where the verifier's re-time of the
first round's pick was a TIE (0.96-0.97). Fold-matched (`fold=none` vs `ftfn`, 40 cells where FlyTape's plan splits):
10 WIN, 6 TIE, 5 MIXED, 19 LOSS; FlyTape's fold is 0.06-0.42 of its own phase time (`ftfn / ft` 0.58-0.94). Section 7
no longer recommends `fold=none` as the default. Launch metric with complete outputs: 10 LOSS cells (HC up 1.18-1.24,
out-proj TP1 r8 1.03, TP2 r8 1.22 and r16 1.06, HC down r64-128 1.06-1.08), confirming verification round 2's
"launch losses" item. mx: complete-output mx arms win 55 of 55 in every run (phase), now measured, not only screened.

**2. Bytes-floor fractions.** Recomputed from complete-output arms only (summary item 6, section 6): the first report's
upper ends for mx out-proj (0.62 -> 0.22) and gdn_in (0.64 -> 0.40) came from `fold=none` arms.

**3. bf16 columns in the in-situ table (verification 1 item 2, round 2 item (b)).** Added (section 6): the LB bf16 best
is slower than FlyTape's measured in-situ phase in 50 of 72 cells (phase metric, 0.58-2.66) and in 13 in the stream
metric; summary item 4 says so. The replica is slower than FlyTape even without any sync in 24 of 72 cells, so
same-harness wins are wins over the replica.

**4. aiter / ATOM arms (verification 1 item 3, round 2 item (c)).** `p-l3/pl3_ext.py` adds, in the container:
`atom_w4a8` (ATOM's `native_quant_linear`, the W4A8 path ATOM serves its dense linears with: the matched-precision
baseline of the mx regime), `aiter_a4w4` (aiter `gemm_a4w4`, preshuffled MXFP4 weights x MXFP4 activations, W4A4),
`aiter_bf16` (aiter `tgemm.mm`). Each runs one call per layer over the cell's L rotating layers in a torch graph (launch
metric), on the cell's own weights (MXFP4 codes and E8M0 scales; aiter's FP4 activations are its own, the reference
uses exactly them), checked against the float64 reference with the P-L3 metric on layers 0 and L-1 before timing, plus
a wrong-layer must-fail control of the check: job 68233558 `w6-pl3f-xn1` (container, n01-09, bus 15; 7 cells, one
per projection) judged layer 0's output against the last layer's operands for every library arm: 21 of 21 caught
(324-15,154 eps), while the same arms pass at 0.000-0.001 eps on their own layer where they pass at all (aiter bf16
at out-proj and aiter W4A4 at HC up failed again, as in the full runs). Results in section 6 and summary item 2: LB mx /
ATOM W4A8 0.07-0.44 (WIN 55 of 55 on both nodes); LB mx / aiter W4A4 25 WIN, 25 LOSS, 5 cells without a valid aiter
arm (HC up, K = 320: aiter's output fails N1 by 2e5-5e5 eps); aiter bf16 fails the bf16 tolerance at the out-proj
shapes (2.9-9.4 eps) and is not timed there. The DESIGN-LB P-L3 comparison item "vs hipBLASLt / aiter at the same
shapes" is now covered for both regimes.

**5. op_sel mutants in the builder's suite (round 2 item 2).** `sopsel` (the op_sel byte moved to the next chunk's) runs
on every `sop` suite arm (`s1_sop`, `smid_sop`) and `kskip` now runs on them too; the `fold` mutant runs only on arms
that fold. Every run: 132 of 132 mutant runs caught (`sopsel` 13 at 580 eps and up). LLVM IR of 14 existing builds and
5 streamers against the measured tree's `gemm.py` (sha 85404ab7..., stage `w4-pl3-rocke`), in-job on both nodes:
0 differing. CPU tests `library/tests/test_qwen38_lb_gemm.py`: 27 passed (includes the `sopsel` build, refusal and
body-differs checks).

**6. Fragility (round 2 item 8).** Two runs per node and the all-runs-agree rule: 2 bf16 phase cells split between the
runs of one node (gdn_out TP1 r64, gdn_out TP2 r8, both on n01) and are MIXED, not verdicts. The combine lists every
run verdict within 0.03 of its control-band edge (`P-L3-fix-tables.md`, "Fragile run verdicts").

**7. HANDOFF.md** line 30 restated to the corrected tally.

Still open (not P-L3's gate): the consumer-side cost of `fold=none` (P-L2 / WP-L3), a faster grid barrier for phase
semantics, the `act=fp4` knob against aiter's W4A4 on gdn_in / qsa_in / the head, and an end-to-end ATOM-precision
check of which projections ATOM runs in bf16 (head, HC mix; router not checked).
