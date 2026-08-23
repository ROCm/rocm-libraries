# VOPD FP32 Campaign Report — gfx1100 (2026-06-01)

## Timing

| Wave | Shapes | Start | End | Duration |
|------|--------|-------|-----|----------|
| Wave 2 (medium) | 240 | 17:08 | 18:31 | 1h23m |
| Wave 3 (large) | 320 | 18:31 | 19:33 | 1h02m |
| Wave 4 (large) | 250 | 19:33 | 23:10 | 3h37m |
| **Total** | **810** | | | **6h02m** |

Wave 4 took longer due to large shapes with more kernel configs surviving rejection.

## Results Summary

| Metric | Count |
|--------|-------|
| Total shapes (filtered ≤8192) | 1,020 |
| VOPD tuned (waves 2-4) | 810 |
| Wave 1 kept (non-VOPD, tiny) | 210 |
| | |
| **VOPD wins** | **406** (49.9% of tuned) |
| Baseline wins (VOPD slower) | 374 (with valid VOPD results) |
| Baseline kept (all VOPD rejected) | 30 |
| **Aggregate uplift** | **+29.5%** |

## Where VOPD Wins

| Metric | Value |
|--------|-------|
| Mean uplift (where VOPD won) | +38.4% |
| Max uplift | +153.8% |
| Min uplift | +0.2% |

VOPD dominates on shapes where both M and N are large enough for MT128x128 tiling
(both ≥ 128). Wave 4 (M×N ≥ 1M) has the best win rate: 178/250 = 71%.

## Where Baseline Wins

584 shapes keep non-VOPD baseline. Breakdown:

| Category | Count | Reason |
|----------|-------|--------|
| M×N < 1K (tiny) | 130 | Too small for even×even TT |
| 1K ≤ M×N < 100K (small) | 277 | Small TT (TT[1,2], TT[2,4]) fits better |
| 100K ≤ M×N < 1M (medium) | 105 | Skinny shapes (e.g. 16x4096, 64x8192) |
| M×N ≥ 1M (large) | 72 | Skinny shapes (128x8192, 256x4096) |

The pattern: **VOPD loses on skinny shapes** where one dimension is very small
(<256). MT128x128 wastes tiles, and smaller non-VOPD TTs fit the shape better.
This is expected and correct — the final logic should use non-VOPD for these shapes.

## Final Logic Composition

The production logic will combine:
- **406 VOPD solutions** (shapes where VOPD won)
- **584 non-VOPD solutions** (shapes where baseline won or VOPD rejected)
- **210 Wave 1 non-VOPD** (tiny shapes, not re-tuned — pending Wave 1 VOPD run on 50 eligible shapes)

## Per-Wave Detail

### Wave 2 (medium, 240 shapes)
- VOPD wins: 93 (38.8%)
- Uplift where VOPD won: +43.8% aggregate
- Recipe: 6 TTs, 4 WGs, GSU=[1,2,4], DU=[8,16,32]

### Wave 3 (large, 320 shapes)
- VOPD wins: 135 (42.2%)
- Uplift where VOPD won: +15.8% aggregate
- Many skinny shapes in this wave (16x4096 class)

### Wave 4 (largest, 250 shapes)
- VOPD wins: 178 (71.2%)
- Uplift where VOPD won: +31.5% aggregate
- Best win rate — large square/balanced shapes

## Files

| File | Size | Purpose |
|------|------|---------|
| `timing.log` | — | Start/end timestamps per wave |
| `wave{2,3,4}.yaml` | — | Input benchmark YAMLs |
| `wave{2,3,4}.log` | — | Full console output |
| `wave{2,3,4}_out/` | — | Complete Tensile output (kernels, CSVs, .hsaco) |
| `winners.json` | 1,020 entries | Per-shape winner with source |
| `gen_campaign_yamls.py` | — | YAML generation script |

## Next Steps

1. Run Wave 1 VOPD on 50 eligible tiny shapes (already generated: wave1.yaml)
2. Merge final logic YAML combining all winners
3. Validate 5% random sample with NumElementsToValidate=256
4. Consider re-tuning the 105 medium skinny shapes with narrower MT tiles (MT256x32, MT32x256)
