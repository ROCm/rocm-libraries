# navi32 (gfx1101) catalog campaign

Supporting material for the navi32 changes on this branch. **This is campaign tooling and a
measurement record, not product documentation** — nothing here is wired into the Sphinx docs
under `projects/hipblaslt/docs/`.

## What changed, and why

navi32's TN GEMM catalogs were far thinner than navi31's on the same ProblemTypes — around
70 solutions against ~300, with **471-row** shape tables against **~9 700**. GridBased
selection is nearest-neighbour matching, so on a 471-row table most real shapes resolve to a
distant neighbour and run a badly-sized tile.

Widening all four affected TN ProblemTypes measured, on 996–997 stratified shapes each:

| ProblemType | solutions | wall-clock | A/A control |
|---|---|---|---|
| `Cijk_Alik_Bljk_HHS_BH_Bias_HAS_SAV` | 73 → 298 | **+23.9%** | 100.32% |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HAS_SAV` | 64 → 306 | **+22.2%** | 100.60% |
| `Cijk_Alik_Bljk_HHS_BH_Bias_AuxH_HAS_SAV` | 73 → 313 | **+20.4%** | 100.42% |
| `Cijk_Alik_Bljk_BBS_BH_Bias_AuxB_HAS_SAV` | 64 → 316 | **+18.8%** | 100.19% |

15 122 measurements, zero failures. Origami also gained gfx1101 support — it did not
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

**The runbook exists because three separate checks returned a reassuring signal while doing
nothing** — a CU mask that reported 30 CUs but restricted nothing, a `--logic-filter` build
that exited 0 having compiled zero kernels, and a sweep that produced rows at the normal rate
which were all errors. Verify the artifact, not the exit status.
