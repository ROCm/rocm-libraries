# Which candidates to benchmark

The search ranks candidates by tier 0 and many tie at the best score: 25 of the MXF8 8×10
config's candidates score its 1377 floor.  `LoopModelAutoSelect: N` and the Expand CLI hand the
GPU a handful of them, so the question is which handful.  This note compares the current
selection against wider and stratified ones on measured kernels, and answers whether the
per-order cap drops good candidates.

## The designs

| Design | Pool | Pick |
|---|---|---|
| current | tier-0 ranking, at most 2 per order choice (`TIER2_PER_ORDER`), stopping at 8 plus ties | tier 3 on the first 16, the 8 best that emit |
| band, stratified, cap 2 | every candidate within 5% of the best tier-0 score, at most 48 lowered | the best tier 2, then round-robin over classes (inner × outer order, deeper-prefetched operand, copies ≤ 3), best tier 3 first |
| band, stratified, no cap | the same without `TIER2_PER_ORDER` or `PER_ORDER` | as above |
| band, tier 3, no cap | the same | the 8 best tier 3 |
| outside the band | tier-0 score 5–25% above the best | the 8 best tier 2 that emit |

An order choice is the inner and outer order together with the WMMA factors M, N and K.

## Measured

Six variants of `config/maf/mxf8_loop_model.yaml` (gfx1250, one card, 4096×5120×65536,
2026-11-20): the 8×10 / 8×8 variants with every band candidate that emits, twice (the faster
run kept), and the 4×4 variants once.  Each cell is the best kernel the design's set holds, as
a percentage of the best kernel measured for the variant:

| Variant | Best (PF) | current | stratified, cap 2 | stratified, no cap | tier 3, no cap | outside the band |
|---|---|---|---|---|---|---|
| MXF8 8×10 | 8.81 | 100 | 100 | 100 | 100 | 98.9 |
| F8 8×10 | 9.62 | 100 (4 emit) | 100 | 100 | 100 | 96.5 |
| MXF8 4×4 | 4.64 | 100 | 100 | 100 | 100 | 81.0 |
| F8 16×16×64 8×10 | 7.37 | 100 | 100 | 100 | 100 | 76.7 |
| BF16 8×8 | 2.29 | 100 | 99.9 | 99.9 | 99.9 | none |
| BF16 4×4 | 0.97 | **81.7** | 81.7 | 81.7 | 81.7 | **100** |

**The band is flat.**  Inside it the top kernels are within 1% (F8, MXF8), 0.4% (BF16 8×8) and
2.5% (F8 16×16×64) of each other, so any set drawn from it holds a kernel within 0.1% of the
best.  Stratifying only lowers the set's mean: its class-diversity picks are the slow ones
(BF16 8×8 mean 2.16 against 2.27 for the current set, from a two-copy `WFK2` kernel tier 3 puts
at 3144 cycles).

**Emit failures, not the ranking, shrink the set.**  Near the VGPR budget most of a tied band
does not emit: 16 of 22 MXF8 band candidates and 27 of 48 F8 ones go past 1024 VGPRs in
StinkyTofu's allocation, though tier 1 counts them under the budget.  The current set held only
4 kernels for F8.  `LoopModelAutoSelect: N` therefore emits each candidate and refills from the
ranking (`expand.REFILL`).

## The per-order cap

Removing both caps added one kernel in six variants (MXF8, a four-copy PLR 7 kernel at 7.67 PF)
and changed no set's best.  Across the six no-cap bands only one order choice had more than 2
tied candidates.  Two reasons:

- The cap is per order choice, not per loop order.  One inner × outer order reaches the band
  through each of its factor choices: 12 candidates of one loop order sat in the F8 band.
- An order choice's register variants rarely tie.  The best register pair sets the score; the
  others expose more latency or load more units and fall out of the band.

The cap bounds the lowerings and costs nothing measurable, so it stays.

## The miss: BF16 4×4

Every BF16 4×4 band kernel is a three-copy body at about 0.79 PF.  The fastest kernels are
one-copy bodies with both operands resident (CLR 1): 0.97 PF for `WIO4 WOO2 WFM1 WFN2` and
`WIO6 WOO1 WFM2 WFN1`, 0.83–0.84 for their `WFM2 WFN2` siblings.  Tier 0 scores them 1201
against 997, because `tdmExposed` charges a one-copy body the whole 800-cycle `tdmLatency` and a
multi-copy one 800 − 256 ([score.md](score.md)).  On a 256-cycle trip both drain the tensor loads,
and the one-copy body is the faster.  No band, however selected, reaches them.

Tier 2 sees it: the best tier 2 outside the band is 408 cycles against 832 inside it, a ratio
of 0.49.  On the other five variants that ratio is 0.94–1.03:

| Variant | Best tier 2 in the band | Best tier 2 outside | Ratio |
|---|---|---|---|
| MXF8 8×10 | 1409 | 1319 | 0.94 |
| F8 8×10 | 1323 | 1309 | 0.99 |
| MXF8 4×4 | 847 | 873 | 1.03 |
| F8 16×16×64 8×10 | 961 | 950 | 0.99 |
| BF16 4×4 | 832 | 408 | 0.49 |

Spending fixed slots outside the band does not work.  Four of eight cost MXF8 4×4 up to 10%,
and three reach only 86% on BF16 4×4, because the two 0.97 kernels tie on tier 2 with the
0.83 ones (422 against 408–414) and differ only in their WMMA factors.  The fix belongs in tier 0's
one-copy term.  Until then, a candidate whose tier 2 is below about 0.8 × the band's best can
be benchmarked alongside the band, which only BF16 4×4 triggers.  Neither change is made yet.

## Noise

The slow MXF8 tie (`WIO1 WOO3 WFM1 WFN2`, A PLR 4 / VG 4 / VA 5) measured 7.20 PF in one run
and 8.66 in the next; its cluster-barrier stall depends on timing.  Other kernels repeated
within 1.5%.
