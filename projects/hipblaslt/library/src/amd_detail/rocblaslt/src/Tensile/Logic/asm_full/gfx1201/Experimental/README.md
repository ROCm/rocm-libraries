# gfx1201 (Navi48) HHS-TN experimental catalogs

Two tuned FP16 TN (`Cijk_Alik_Bljk_HHS_BH_Bias_SHB_HA_S_SAB_SCD_SAV_UserArgs`) catalogs from
the gfx1201 HHS-TN campaign, each in both selector forms. **Nothing here is loaded by a normal
build** — these live outside the arch logic directory and are consumed explicitly (see *Using
them* below). The shipped `gfx1201/GridBased/` logic is untouched.

| directory | catalog | selector (`logic[11]`) | kernels |
|---|---|---|---|
| `lean69_sk0_grid/` | lean-69 (SK0) | `GridBased` + 9,699-row table | 69, all StreamK=0 |
| `lean69_sk0_origami/` | lean-69 (SK0) | `Prediction` (Origami analytical) | same 69 |
| `sk3_bestof_grid/` | tuned-SK3 best-of | `GridBased` + the same 9,699-row table | 69: 38 StreamK=3, 31 SK0 |
| `sk3_bestof_origami/` | tuned-SK3 best-of | `Prediction` (Origami analytical) | same 69 |

The two catalogs are **index-aligned**: identical 69 tile geometries (MT0×MT1×DepthU) and an
identical routing table; best-of differs only in that 38 of the 69 solution slots hold a tuned
StreamK-3 kernel instead of the SK0 one, chosen per representative tile by whichever measured
faster. So the pair isolates *StreamK-3 vs SK0* with the catalog shape and routing held fixed,
and the Grid/Origami pair isolates *selector* with the kernel pool held fixed.

## Provenance

- **lean-69** — rep-per-tile distillation of the stock 778-kernel G0 catalog by usage + coverage
  (806 → 97 → 69 retained kernels). Roughly 1/11th the kernel count of G0.
- **tuned-SK3 best-of** — SK3 kernels tuned on 352 shapes, folded in only where they beat the
  SK0 rep. `eval-1500 ∩ 352-train = 7` shapes; those 7 were excluded from the tuned arms when
  scoring, so the numbers below are held-out.
- Measured on 8× gfx1201 (Navi48), rank-1 adaptive, rotation on (per-shape footprint-sized).

## Measured standing

**v6, geomean speedup vs the stock G0 778 × GridBased baseline = 100%** (200 curated shapes /
1500 fresh strata-balanced shapes, disjoint from the 200):

| catalog × selector | 200 | 1500 |
|---|--:|--:|
| lean-69 (SK0) × Grid | 100.9% | **109.3%** |
| lean-69 (SK0) × Origami | 92.1% | 101.0% |
| tuned-SK3 best-of × Grid | 98.0% | 107.5% |
| tuned-SK3 best-of × Origami | 96.0% | 102.3% |
| G0 778 (SK0) × Grid *(baseline)* | 100.0% | 100.0% |
| G0 778 (SK0) × Origami | 92.6% | 91.7% |

Reading: **lean-69 SK0 × GridBased is the best pair on both suites**, and its edge over the full
G0 *grows* with shape diversity (parity on the 200 → +9.3% on the fresh 1500) at 1/11th the
kernel count — fewer kernels means fewer chances to mis-route. Selective SK3 tracks it but stays
~2% behind on both suites; blanket SK3 (not included here) loses outright, worst on large tiles.

**Caveat on the Origami columns.** Those were measured against a runtime built *before* the
Origami source changes in this branch. Re-measured on a runtime built from this tree (v8, 160
stratified shapes, same kernel pool both arms), Origami-vs-Grid is roughly **parity**: geomean
per-shape **1.013**, FLOPs-weighted 0.980, 67 win / 60 lose / 33 tie. It now *wins* small
(+11.8%) and tiny; the remaining deficit is deep-K (K>512: 0.967) medium/large, where it still
picks a bigger MacroTile than Grid on 85% of large shapes. Treat the v6 Origami rows as a
lower bound on a stale binary, not as this runtime's standing.

## Using them

These are Tensile *logic* files, not device libraries. Point `TensileCreateLibrary` (or the
campaign's `make_devlib.sh`) at one of the four directories to build a device library from it,
then preload that library. The `*_origami` variants additionally need `TENSILE_PREDICTION_LIB=1`
at runtime so the `Prediction` library type is honoured.

Note that the stock 778-kernel G0 catalog faults with HIP error `719` under concurrent
benchmarking; the campaign used a crash-safe per-problem runner for G0 baselines. The 69-kernel
catalogs here were crash-free under the sharded persistent runner.
