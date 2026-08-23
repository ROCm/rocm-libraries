# Was the Origami tuning needed? — four clean measured arms

1500 shapes with complete 3-rep data in every completed arm. All four arms are freshly built from pinned branches; selection is done by each build's own runtime Origami (no offline ranking, no index forcing).


## Provenance

| arm | code SHA | tuned weights in binary | kernels | bench sha256 |
|---|---|---|---|---|
| stock / SK3 (192) | `82580dfc726` | absent | 367 | `0f52668ff05a` |
| tuned / SK3 (192) | `3eb22b8a68b` | **present** | 367 | `830ceddc206e` |
| stock / Grid (298) | `82580dfc726` | absent | 476 | `0f52668ff05a` |
| tuned / Grid (298) | `3eb22b8a68b` | **present** | 476 | `830ceddc206e` |

Stock binaries must show the fitted constants *absent* and tuned *present*; that gate passed for every arm before it was benchmarked.


## Absolute throughput (geomean GFLOP/s over shared shapes)

| arm | geomean | median | P10 | min | distinct kernels |
|---|---|---|---|---|---|
| stock / SK3 (192) | 3241.1 | 8072.5 | 47.4 | 0.0 | 62 |
| tuned / SK3 (192) | 3238.8 | 7992.3 | 48.4 | 0.0 | 48 |
| stock / Grid (298) | 3108.7 | 7937.2 | 39.4 | 0.0 | 72 |
| tuned / Grid (298) | 3193.9 | 8319.0 | 39.6 | 0.0 | 71 |

## The three contrasts

Per-shape ratios; >100% means the first arm is faster.

| contrast | question | geomean | 95% CI | wins | verdict |
|---|---|---|---|---|---|
| stock_sk3 / tuned_sk3 | did tuning help? (catalog fixed = SK3) | 100.07% | [99.30%, 100.84%] | 632/1500 | no difference (CI spans 1.0) |
| stock_sk3 / stock_grid | did the SK3 catalog do the work? (selector fixed = stock) | 104.26% | [103.06%, 105.49%] | 687/1500 | first arm faster |
| stock_grid / tuned_grid | tuning off its fitted catalog (core refit only) | 97.33% | [96.61%, 98.07%] | 667/1500 | second arm faster |
| tuned_sk3 / tuned_grid | catalog effect under the tuned model | 101.40% | [100.26%, 102.56%] | 642/1500 | first arm faster |

CIs are 2,000-sample bootstraps over the per-shape paired ratios.


**Selection overlap on SK3:** stock and tuned chose the same kernel on 693/1500 shapes (46.2%). If this is near 100% the tuning is inert at runtime; if low, the models genuinely disagree and the throughput result is a real comparison.


## Per-stratum geomean ratio, tuned/SK3 vs stock/SK3

| stratum | n | tuned/stock |
|---|---|---|
| large | 260 | 99.28% |
| medium | 504 | 102.24% |
| small | 399 | 97.92% |
| tiny | 337 | 99.42% |

## Data quality

- **stock / SK3 (192)**: non-ok rows none, median CV 1.75%, CV>5% on 300 shapes, within-arm drift (first half / second half) 99.32%.
- **tuned / SK3 (192)**: non-ok rows none, median CV 1.61%, CV>5% on 279 shapes, within-arm drift (first half / second half) 99.42%.
- **stock / Grid (298)**: non-ok rows none, median CV 1.55%, CV>5% on 292 shapes, within-arm drift (first half / second half) 100.80%.
- **tuned / Grid (298)**: non-ok rows none, median CV 1.73%, CV>5% on 329 shapes, within-arm drift (first half / second half) 99.50%.

Arms were measured sequentially, so drift is the main confounder. Measured drift is under 1% in every arm — smaller than the material contrasts below it — so the +4.3% and −2.7% results are not drift artefacts.


## Against the production GridBased selector (G0)

G0 = the shipping GridBased selector on the 298-kernel catalog, from the earlier confirmed campaign. It is a *measured* per-shape lookup table (ExactLogic), not an analytical predictor, so it is the bar to beat.

**Cross-binary caveat.** G0/O3 were measured on a different `hipblaslt-bench` build. The one arm present in both campaigns — tuned+SK3 — reproduces to 99.28% (CI [98.97%, 99.61%]). So the old baseline is usable, but carries a systematic offset of roughly 0.7%; treat differences below that as noise.

| arm | % of G0 | 95% CI |
|---|---|---|
| stock / SK3 (192) | 96.92% | [95.78%, 98.04%] |
| tuned / SK3 (192) | 96.85% | [95.69%, 97.96%] |
| stock / Grid (298) | 92.96% | [91.99%, 93.91%] |
| tuned / Grid (298) | 95.51% | [94.65%, 96.37%] |
| *old O3 (tuned/SK3, previous campaign)* | 97.55% | published 97.55% |
| *G0 baseline* | 100.00% | — |

**No Origami configuration beats the production selector.** On the *same* 298-kernel catalog, Origami is far behind G0 — and that gap is where the tuning earns its keep.


## Conclusion

Ranked by geomean throughput, relative to the worst combination (stock selector on the production Grid catalog):

| combination | geomean GFLOP/s | vs stock/Grid |
|---|---|---|
| stock / SK3 (192) | 3241.1 | +4.26% |
| tuned / SK3 (192) | 3238.8 | +4.19% |
| tuned / Grid (298) | 3193.9 | +2.74% |
| stock / Grid (298) | 3108.7 | +0.00% |

**Tuning and catalog distillation are substitutes, not complements.** Either one alone closes most of the gap, and doing both adds nothing over doing one:

- Tuning on the distilled SK3 catalog buys **nothing** (-0.07%, CI spans 1.0) — even though the two models pick different kernels on 54% of shapes.
- The SK3 catalog alone, with the untuned selector, is worth **+4.26%**.
- Tuning alone, on the Grid catalog, is worth **+2.74%** — and note the resource/edge terms are inert there (StreamK gate), so that gain comes purely from the 11 refitted core weights.
- Under the tuned selector the catalog is worth only +1.40%, versus +4.26% under the stock one — the two mechanisms overlap.

**But none of it beats the shipping selector.** Measured against G0, every Origami arm is behind: the best (stock or tuned on the distilled SK3 catalog) reaches ~97%, and on the production Grid catalog Origami loses 4.5–7.0%. The honest reading is that the tuning is real and worth ~2.7% *where the selector must choose from a broad, undistilled catalog* — which is exactly the K=1 regime it was fitted for — but neither tuning nor catalog distillation closes the gap to a measured per-shape lookup table.
