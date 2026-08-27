# navi32 (gfx1101) catalog campaign

Supporting material for the navi32 changes on this branch. **This is campaign tooling and a
measurement record, not product documentation** — nothing here is wired into the Sphinx docs
under `projects/hipblaslt/docs/`.

## What changed, and why

navi32's TN GEMM catalogs were far thinner than navi31's on the same ProblemTypes — around
70 solutions against ~300, with **471-row** shape tables against **~9 700**. GridBased
selection is nearest-neighbour matching, so on a 471-row table most real shapes resolve to a
distant neighbour and run a badly-sized tile.

> **Measured: the pool is ~4x the lever the table is, and the table only pays once the pool is
> rich.** Two experiments, both at real 60 CUs:
> over a *thin* 64-solution pool, swapping in a different 471-row table moves **nothing**
> (99.79% vs a 99.70% A/A control — there is no well-sized tile to point at). Over the *shipped*
> 306-solution pool, sparsifying the table 9 692 → 471 rows costs **3.4 pt** (96.62% vs a 100.38%
> control), while the pool itself is worth **13.8 pt** — an ~80/20 split of the 17.2 pt total.
> **Widen the pool first; then the table fit is worth doing.**
>
> *(Resolution note, 2026-08-27: the pool figures are an order of magnitude above this setup's
> ~2-5% measurement floor and are solid. The **table** figures are at that floor — the "moves
> nothing" null means "no effect above a few percent", and an independent 70-shape replication of
> the 3.4 pt cost gives 1.029 with a 95% CI of [0.989, 1.074], i.e. consistent but not separable
> from zero. Quote the table half as "~3 pt, at the limit of what this setup resolves". See
> `MASKED_60CU_VALIDATION.md`.)*
> [`MASKED_60CU_VALIDATION.md`](MASKED_60CU_VALIDATION.md)

Widening all four affected TN ProblemTypes measured, on 996–997 stratified shapes each:

| ProblemType | solutions | wall-clock | A/A control |
|---|---|---|---|
| `Cijk_Alik_Bljk_HHS_BH_Bias_HAS_SAV` | 73 → 298 | **+23.9%** | 100.32% |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HAS_SAV` | 64 → 306 | **+22.2%** | 100.60% |
| `Cijk_Alik_Bljk_HHS_BH_Bias_AuxH_HAS_SAV` | 73 → 313 | **+20.4%** | 100.42% |
| `Cijk_Alik_Bljk_BBS_BH_Bias_AuxB_HAS_SAV` | 64 → 316 | **+18.8%** | 100.19% |

15 122 measurements in those sweeps and 22 400+ across the campaign, zero failures. Origami also gained gfx1101 support — it did not
recognise the architecture at all before (`arch_name_to_enum("gfx1101")` returned `Count`).

## Files

| file | what it is |
|---|---|
| `NAVI32_CATALOG_REPORT.md` | the measurement record: results, rejected hypotheses, caveats, audit |
| `NAVI32_RUNBOOK.md` | **read this before repeating any of it** — method and every trap hit |
| `bench_arms.py` | interleaved multi-arm benchmark; an arm is a (library, env) pair |
| `analyze.py` | geomean + flops-weighted wall-clock, jackknife, per-stratum breakdown |
| `arith_intensity.py` | splits results by flop/byte to bound the memory-bandwidth gap |
| `retarget_logic.py` | retarget a logic file between gfx1100 and gfx1101 (**two ISA sites**) |
| `to_prediction.py` | GridBased → Prediction (Origami-selected) conversion |
| `wgm_variant.py` | rewrite `WorkGroupMapping` across a whole logic file |
| `MASKED_60CU_VALIDATION.md` | all four catalogs re-validated at **genuine 60-CU execution**, plus the WGM, Prediction, oracle and hybrid-gate re-tests |
| `NAVI33_OCCUPANCY.md` | navi33's smaller register file costs only 10.1% of kernel time — LDS binds 88% of these kernels |
| `NAVI33_EVIDENCE.md` | navi33 ships an identical catalog; widening is worth ~+15% at its 32 CUs. Evidence, not a change |
| `regime_all.py` / `aa_floors.py` | 96-CU vs 60-CU comparison on matched shapes, and per-run A/A floors |
| `oracle60.py` | oracle over every library built, **with a cardinality-matched noise floor** |
| `hybrid_gate.py` | out-of-sample test of a size-gated GridBased/Prediction selector |
| `compare_masked.py` | the catalog win at 96-CU vs 60-CU execution on matched shapes |
| `pred_regime.py` | the Prediction-vs-GridBased gap in both execution regimes |
| `navi33_occupancy.py` / `occ_lib.py` | RDNA3 occupancy from a `.co`, honouring **both** the VGPR and LDS limits |

## Three things worth knowing before extending this

**It was developed on a gfx1100 part emulating navi32.** Selection is navi32-correct via
`--sm_count_target 60`; the main sweeps executed on all 96 CUs, on a belief that a real
per-stream CU mask hung ~37% of runs. **That rate was wrong — it is 2.0% over 1 242 runs** —
and the premise has since been checked directly: at **genuine 60-CU execution** the catalog
win is **+22.7% wall-clock / +25.2% geomean against a 0.11 pt A/A floor**, versus +25.7% on
the same shapes at 96 CUs ([`MASKED_60CU_VALIDATION.md`](MASKED_60CU_VALIDATION.md)). Arm
ratios were sound; absolute throughput is optimistic. ~73% of kernel time is compute-bound, so
the result transfers despite the bandwidth difference — see the report's arithmetic-intensity
section.

**Three hypotheses were tested and rejected.** Do not retry them without new evidence:
re-forking `WorkGroupMapping` for 60 CUs (**null, re-confirmed at real 30 WGPs** — WGM10 99.78%
/ WGM6 100.35% wall-clock against a 0.07 pt A/A floor; the first test ran at 48 WGPs and could
not have expressed the hypothesis, see [`MASKED_60CU_VALIDATION.md`](MASKED_60CU_VALIDATION.md)),
switching to an Origami `Prediction` library (**re-measured at matched 60 CUs: the 13 pt
gap is ~3.3 pt geomean / 1.3 pt wall-clock, near parity on wall-clock** — GridBased still wins,
but only on small shapes), and extending the catalog past ~300 solutions (**re-measured at 60 CUs
with a noise floor subtracted: the real oracle over everything built is +1.71 pt**).

**All three rejections have now been re-tested in the matched 60-CU regime**, because two of the
three were originally measured in a regime that could not express them. All three survive. The
oracle analysis relocated the remaining opportunity to the *selector* — **+1.47 of the +1.71 pt
comes from pairing GridBased with Origami Prediction** — and that lever has since been measured
too: a realizable size predicate captures **+0.47 pt out of sample** (a flops *band*; no
monotone threshold works at all, because Prediction's edge sits in a middle band rather than at
the top end). **~A quarter of the oracle is reachable; the rest is not separable by problem
geometry.** Against the C++ work of adding a row predicate to the selector, that is probably not
worth it — **so there is no known lever left on this catalog.**

**The thin catalog is one artefact shipped on five architectures.** navi32 (pre-fix), navi33,
gfx1103, gfx1150 and gfx1152 ship *identical* logic files for all four thin TN ProblemTypes —
a raw diff of any pair is 152 lines of 14 674, containing only the PCI device ID, the arch
name and 73 ISA tuples. Not one solution parameter differs. gfx1151 is the counter-example:
the only RDNA3 part with its own tuning campaign, and the only one with a full-size catalog.
navi33 is now measured too — at its 32 CUs widening is worth **+15.3% wall-clock / +25.2%
geomean against a 0.00 pt A/A floor** ([`NAVI33_EVIDENCE.md`](NAVI33_EVIDENCE.md)). Not
shipped: evidence for a decision. **This branch fixes one of the five.**

**The runbook exists because three separate checks returned a reassuring signal while doing
nothing** — a CU mask that reported 30 CUs but restricted nothing, a `--logic-filter` build
that exited 0 having compiled zero kernels, and a sweep that produced rows at the normal rate
which were all errors. Verify the artifact, not the exit status.
