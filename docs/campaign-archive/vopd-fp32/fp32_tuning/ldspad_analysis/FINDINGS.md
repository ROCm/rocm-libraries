# LdsPad SHAP analysis — findings for presentation

gfx1100 FP32 GEMM tuning. All images in `out/`. Interpreter: `rdna_gemm_bench/venv`.

## Headline: LdsPad dominates when it varies within a controlled config family

The **auto_engine ledger** (2,437 configs, each differing only in tuning params,
benched back-to-back on the same GPU) is the clean dataset. XGBoost model CV
R² = **0.994**.

- **LdsPadA + LdsPadB = 80.1% of total mean|SHAP|** — ranked #1 and #2, far above
  DepthU, tile, workgroup, etc.
- The effect is a **bank-conflict cliff**: an *odd* pad (LdsPad=1) forces LDS bank
  conflicts and craters throughput; pad = 2/4/8 removes them.

### Best slide-ready examples (strongest → supporting)

| Image | What it shows |
|---|---|
| `ledger_shap_bar.png` | **THE headline** — LdsPadA/B bars dwarf all other params |
| `ledger_dep_LdsPadA.png` | **THE cliff, SHAP form** — LdsPadA=1 → −15,000 GFLOPs; 2/4/8 → +3,000 |
| `showcase_ledger_cliff.png` | **THE cliff, measured** — 4096³ MT128×128 DU16: 5,800→30,500 GFLOPs (+425%) by pad alone |
| `showcase_ledger_heat.png` | LdsPadA×LdsPadB mean-GFLOPs grid — the LdsPad=1 row/col is red |
| `ledger_shap_beeswarm.png` | direction: low pad (blue) = negative, good pad (red) = positive |
| `showcase_ledger_boxdist.png` | GFLOPs distribution per (padA,padB) — all `*_B1`/`A1_*` combos at bottom |

Key numbers (from `showcase_ledger_table.csv`): top families all
**4096³, MT128×128, DepthU=16**, LdsPad-only swing **404–425%**.

## The "GFLOPs is mostly regular, few exceptions" story

From the wide per-kernel bench CSVs (shapes × kernels GFLOPs matrices):

| Image | What it shows |
|---|---|
| `reg_nn_campaign_heatmap.png` | smooth top-to-bottom gradient (shape-driven) with a few bright/dark vertical kernel stripes = the exceptions |
| `reg_nn_campaign_spread.png` | best/median/worst kernel per shape — tight for most, blows open for a few |
| `reg_<camp>_cv.png` | per-shape coefficient-of-variation histogram; low-CV bulk + long tail |

Campaigns rendered: `nn_campaign`, `tn_campaign`, `vopd_campaign`, `waves`.
`nn_campaign` (87 shapes × ~1000 kernels) reads best.

## Honest caveats (do NOT over-claim on slides)

1. **In the wide-CSV SHAP runs, LdsPad importance is SMALL (0.1–6%).** Reason:
   those campaigns sweep MANY tiles/shapes at once, so shape+tile dominate and
   swamp the padding signal. The clean signal only appears when you *hold
   everything else fixed* — which is exactly what the ledger does. Present the
   ledger, mention the campaigns as the "in the wild it's one of many knobs" foil.
2. The `*_ldspad_pairs.png` for the wide campaigns show absurd deltas
   (10,000%+) — these are **artifacts** (undecoded params collapse distinct
   kernels together + near-zero GFLOPs denominators). **Do not use them.** The
   ledger pairs (`ledger_ldspad_pairs.png`, max 425%) are trustworthy.
3. Controlled micro-experiment `lpb_investigation` (same shape, only LPB): real
   effect there was only **~3%** — because that config was *already* conflict-free
   at LPB=0. The cliff appears specifically where the base config would conflict.

## FP16 (HHS) analysis — gfx1201 retune campaign

Data: 200 Tensile sweep CSVs (`00_Final.csv`) extracted from
`retune/agent_hhs_gfx1201.zip` → `/tmp/hhs_all/`. 48,435 (shape,kernel) records,
HHS (half in / half in / fp32 accum), WMMA MI16x16x1. Model CV R² = 0.988.
Script: `shap_fp16.py`.

### Feature-importance bars (shape sizes removed — as requested)
| Image | What it shows |
|---|---|
| `fp16_shap_bar_noshape.png` | SHAP importance, tuning params only. Order: MT0, MT1, DepthU, MIWT0, **LDSB (#5)**, MIWT1, **LBSPPB (#7)**, WG0, **LBSPPA (#9)** |
| `fp16_importance_noshape.png` | XGBoost gain importance, LDS-padding family highlighted in orange |
| `fp16_shap_bar.png` | full bar incl. shape sizes (SizeI/SizeJ dominate, same as FP32) |

### IMPORTANT honesty caveat for FP16
**LdsPadA/LdsPadB themselves are pinned at 8 in every *valid* FP16 config** — the
only LPA0/LPB0 kernels live in `output_failed_degenerate`/`output_failed_rotbuf`
dirs (failed runs, no GFLOPs). So there is **no clean FP16 "LdsPad cliff" like the
FP32 ledger.** What DOES vary and rank highly is the *related LDS-padding family*:
- **LDSB** (LdsBlockSize / OneLDSBuffer) — SHAP rank #5
- **LBSPPA / LBSPPB** (LdsBlockSizePerPad) — ranks #7 / #9

Present FP16 as: "the LDS-padding family of knobs (LDSB, LBSPP) sits in the top
tier once shape size is factored out" — do NOT claim LdsPadA/B specifically, that
would be unsupported by this dataset. **For the strong, clean LdsPad=1 cliff story,
use the FP32 ledger plots.**

## "Shape sizes removed" bars (all datasets, the view you asked for)
`<dataset>_shap_bar_noshape.png` now exists for every dataset. Highlights:
- `wide_nn_campaign_shap_bar_noshape.png` — **LdsPadB jumps to #2** (behind DepthU)
- `ledger_shap_bar_noshape.png` — LdsPadA/B still #1/#2 (they already dominated)
- FP16: `fp16_shap_bar_noshape.png` (LDS family in top tier)

## Better-ranked LdsPad + banded heatmaps (2nd iteration)

### Per-shape normalized SHAP — LdsPad ranks MUCH higher (`shap_ranked.py`)
Normalizing the target per shape (`GFlops / max GFlops for that shape`) removes
shape-magnitude dominance, so LdsPad's true contribution surfaces:

| dataset | LdsPad best rank | LdsPad share |
|---|---|---|
| **nn_campaign** | **#1 (LdsPadB, ahead of DepthU)** | **29.8%** |
| tn_campaign | #3 (LdsPadB) | 20.1% |
| vopd_campaign | #3 (LdsPadB) | 15.3% |

Best slide: `out/ranked_nn_campaign_shap_bar.png` (LdsPadB #1, 29.8%).
Also `out/ranked_tn_campaign_shap_bar.png`, `out/ranked_*_dep_LdsPadB.png`.

### Kernel-name-sorted BANDED heatmaps (`banded_heatmap.py`)
Sorted so kernels sharing a pad value are contiguous → good/bad values form wide
green/red bands (NOT sorted by GFLOPs).
- **`out/banded_ledger_collapsed_LdsPadA.png`** — clearest: LdsPadA=1 column at
  **0.29** vs 0.78–0.84 for 0/2/4/8. Same for LdsPadB (=1 → 0.38).
- `out/banded_ledger_by_LdsPadA.png` / `_by_LdsPadB.png` — full shape×kernel bands.
- `out/banded_tn_campaign_by_LdsPadB.png` — 3 bands (LPB 0/2/4), subtler effect
  (0.64 / 0.67 / 0.67).
Run other params: `banded_heatmap.py --campaign ledger --by DepthU`.

### FP16/BF16: not possible from on-disk data — see `RUN_FP16_LDSPAD.md`
All valid FP16/BF16 configs pin LdsPad=8. `RUN_FP16_LDSPAD.md` is a portable how-to
to run a small LdsPad-forking sweep on a GPU box and feed the CSV straight into
`shap_ranked.py --glob` / `banded_heatmap.py --csv` for the same plots.

## Reproduce
```
cd ~/vopd_sgemm/fp32_tuning/ldspad_analysis
~/rdna_gemm_bench/venv/bin/python shap_ldspad.py        # SHAP across all datasets
~/rdna_gemm_bench/venv/bin/python ldspad_showcase.py    # clean cliff/heatmap slides
~/rdna_gemm_bench/venv/bin/python regularity_heatmap.py # regularity story
```
Interactive: Streamlit page `4_LdsPad_SHAP.py` added to the tensile-tuning app
(`~/.claude/skills/tensile-tuning/scripts/`, port 8023).
```
cd ~/.claude/skills/tensile-tuning/scripts
~/rdna_gemm_bench/venv/bin/streamlit run param_guide_st.py --server.port 8023
```
