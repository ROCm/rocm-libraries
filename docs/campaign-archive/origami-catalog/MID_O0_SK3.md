# Stock vs tuned Origami, SK3 vs SK0 catalog

Frozen evaluation set, 1500 shapes with complete data in every arm. All figures are the ratio to G0 (production GridBased) on the same shape.

Stock Origami = `eae132fefcf43743f1365c48db72ebd93454330c` (`fd85b319a36^`, the commit before "add gfx1100 Resource Edge prediction metadata").


## Summary (as % of G0)

| arm | n | geomean | mean | median | P10 | P5 | min | P90 | P95 | max |
|---|---|---|---|---|---|---|---|---|---|---|
| O0-SK3  stock Origami, 192 SK3 | 1500 | 98.04% | 100.69% | 98.48% | 76.53% | 66.77% | 37.63% | 121.89% | 141.79% | 243.95% |
| O3      tuned Origami, 192 SK3 | 1500 | 97.55% | 99.93% | 98.18% | 77.88% | 69.23% | 38.06% | 119.33% | 137.24% | 235.80% |
| G0      production GridBased, 298 SK0 | 1500 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |

**Pipeline check:** O3 restated here = 97.55% geomean (published 97.55%). Deviation 0.004 pp.


## Pairwise decomposition

"Material" = the two arms differ by more than 5% on that shape, which is well outside the measured run-to-run spread; smaller gaps are mostly noise and usually mean both arms picked the same or an equivalent kernel.

**Value of the origami commit (catalog fixed at SK3) — O0-SK3 / O3** — geomean 100.50%, median 100.23%. Material wins for stock O0: 380/1500 (25.3%); material wins for tuned O3: 320/1500 (21.3%); within +/-5%: 800/1500.

## Selection behaviour

| arm | distinct kernels | catalog buckets stock Origami can resolve | shapes whose pick was a tie |
|---|---|---|---|
| O0_SK3 | 63 | 130 of 192 (107 kernels share a bucket) | 966 of 1500 (64.4%) |

Within a tie group the kernel is chosen by enumeration order, so those picks are arbitrary by construction — a stock-model limitation, not a tuning result.


## Per-stratum geomean (% of G0)

| stratum | n | O0_SK3 | O3 |
|---|---|---|---|
| large | 260 | 95.47% | 95.10% |
| medium | 504 | 95.22% | 97.76% |
| small | 399 | 92.07% | 90.60% |
| tiny | 337 | 112.59% | 108.25% |

## Data quality

- **O0_SK3**: 1500 shapes measured, non-ok rows none, median run-to-run CV 1.58%, shapes with CV>5% 269.
