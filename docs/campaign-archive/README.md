# Campaign archive — measurement records that existed only on one disk

Reports from gfx1100 tuning campaigns, committed because they lived nowhere else and are
referenced by ongoing work. **Research records, not product documentation.**

Only the *findings* are here. Raw measurement CSVs and build trees (several GB) are not
committed — they are reproducible from the methods each report describes.

## `origami-catalog/` — Origami ranking and catalog distillation, gfx1100 HHS-TN

16 reports from a campaign that measured ~159 000 GEMM configurations. The load-bearing ones:

| report | what it establishes |
|---|---|
| `FINAL_CATALOG_REPORT.md` | the headline: a distilled 58-kernel catalog reaches parity with the shipped selector from a **5x smaller** catalog |
| `STREAMK_CAMPAIGN.md` | 54 000 runs over 8 StreamK grid modes; the shipped predictor is vindicated **on geomean** |
| `BRANCH_MATRIX.md` | tuning vs catalog distillation are **substitutes**: tuning buys +2.74% on a broad catalog and ~nothing on a distilled one |
| `CATALOG_V2.md`, `FIXED_LINEAR_ARBITER_FINAL.md` | the distillation path, including the versions that failed |

**Three results here are protocol-dependent and easy to misquote.** Read
`FINAL_CATALOG_REPORT.md` §0a first: the same catalog scores 99.95% under one benchmark
iteration scheme and 94.86% under another, because a larger library pays more one-time
initialisation and a tiered harness charges it to whichever arm is measured shortest.

Note also that `STREAMK_CAMPAIGN.md`'s "predictor is vindicated" conclusion was later found to
be **metric-dependent** — see `../streamk-rdna/GATED_POLICY.md`, which shows the same data
gives 96.95% by per-shape geomean and **102.17% by wall-clock**.

## `ab1100-gfx1100-v6/` — v6 Prediction catalog vs develop GridBased, fp16 HHS-TN

`REPORT.md` (the measurement) and `PLAN.md` (the method). Headline **96.98% geomean /
106.51% throughput-weighted** against a measured ±5.72% deadband.

`PLAN.md` was reconstructed from a session transcript after the original was overwritten —
plan-mode scratch files are reused per session. That is precisely why this directory exists.
