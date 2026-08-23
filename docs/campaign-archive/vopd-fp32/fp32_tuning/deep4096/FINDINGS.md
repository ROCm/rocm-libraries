# Deep 4096³ TN Optimization — Findings (2026-06-04)

## Result
Found a **+5.2% improvement for TN 4096×4096×4096**: ~28,900 → **~30,400 GFlops** cold-cache,
by changing two parameters from the deployed kernel:
- **WorkGroupMapping: 1** (deployed used higher WGM) — the big lever (~+4%)
- **LdsPadB: 4** (deployed used 2) — small additional gain (~+1%)
Everything else in the deployed kernel was already optimal: MT128x128x16, TT8x16, WG16x8, DU16,
LdsPadA2, VW2, GRVW2, PGR1, PLR1, SIA1, GSU1, StaggerU0.

## Method
6 focused iterations, each cold-cache confirmed (hipblaslt-bench best-of-3, transA=T transB=N),
logged in research_diary.md. Tensile cache-warm numbers used only for ranking candidates.

## What did NOT help (ruled out for TN 4096³)
DU 8/32/64 (DU16 best), bigger tiles MT128x256/256x128 (also lost cold-cache in earlier work),
TransposeLDS=1, VectorWidth4 (much worse), WaveSeparateGlobalRead, SIA 2/3, PGR2, PLR2,
1LDSBuffer, odd LdsPad, GSU2, StaggerU>0.

## CRITICAL caveat — WGM=1 is shape-dependent, do NOT apply globally
Multi-shape check (iter6) showed WGM=1 only wins MID-large squares:
| shape | best WGM |
|---|---|
| 4096×4096×4096 | WGM1 (+3.3%) |
| 2048×8192×8192 | WGM1 (+5.1%) |
| 2048×2048×2048 | ~tie |
| 4096×8192×8192 | **WGM4** (+3.9%) |
| 8192×8192×8192 | **WGM4** (+6.8%) |
Bigger problems launch more workgroups → L2-locality remap (WGM>1) pays off; mid shapes don't have
enough workgroups so WGM1's lower overhead wins. A blanket WGM=1 would REGRESS 8192³ (just fixed!).

## Recommendation (pending user OK — nothing deployed/committed)
Capture the win PER-SHAPE: update the TN logic's 4096³ entry (and likely 2048×8192×8192) to the
WGM1+LPB4 kernel, leaving 8192³/4096×8192² on WGM4. The grid-tuned logic already does per-shape
selection, so this is a targeted re-tune + merge of the WGM1-favoring shapes.

TN's residual gap vs NT (NT 4096³ ~33.8 TF vs TN ~30.4 TF) appears intrinsic to the A^T data path;
not closable by the Tensile knobs swept. ~30.4 TF is the practical TN ceiling for this config family.

## State
Repo CLEAN at commit d6c855b6e91. TN/NT/NN all at committed (deployed) state — the deep run reverted
all trials. Best candidate logic saved: deep4096/iter4_merged/. Full trace: research_diary.md.

## UPDATE — retune+merge attempted, then REVERTED (result was noise)
Per user request, retuned WGM1-favoring shapes and merged. On rigorous PAIRED-INTERLEAVED A/B
(before/after benched back-to-back to cancel GPU drift), the apparent gains DISAPPEARED:
- Same-lib noise floor on 4096^3 = ~3.2% spread (27641-28528 on identical binary).
- The ~3-5% "improvement" is WITHIN that noise floor → not a real, deployable win.
- Earlier sequential before/after tests were contaminated by GPU thermal/clock drift over the session.
DECISION: NOT deployed. Repo reverted to clean committed state (d6c855b6e91). TN/NT/NN unchanged.
The WGM shape-dependence remains a valid qualitative insight but yields no noise-robust speedup here.
