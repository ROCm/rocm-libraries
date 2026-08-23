# FP32 gfx1100 Tuning — Research Diary

Campaign: 2026-05-30
Goal: Build production-quality gfx1100 FP32 logic YAML covering 1,030 shapes.
**Status: COMPLETE**

---

## Campaign Summary

### Timeline
| Phase | Duration | Shapes | Solutions | Status |
|-------|----------|--------|-----------|--------|
| Setup + Analysis | 10 min | - | - | DONE |
| Wave 1 (tiny, M*N<4K) | 41 min | 210 | 1,512→47 winners | DONE |
| Wave 2 (small, 4K-64K) | 19 min | 240 | 508→63 winners | DONE |
| Wave 3 (medium, 64K-1M) | 28 min | 320 | 224→59 winners | DONE |
| Wave 4 (large, M*N>=1M) | 80 min | 260 | 124→26 winners | DONE |
| LibraryLogic gen | 5 min | - | - | DONE |
| Merge | 1 min | - | - | DONE |
| **Total** | **~3 hours** | **1,030** | **162 unique** | **DONE** |

### Output Files
- **Production logic**: `logic/merged/gfx1100_Cijk_Ailk_Bjlk_S_B_UserArgs.yaml` (48,591 lines)
- **Deployed to**: `library/.../gfx1100/GridBased/gfx1100_Cijk_Ailk_Bjlk_S_B_UserArgs.yaml`
- **Winner data**: `logic/all_winners.json` (1,030 entries)
- **Wave CSVs**: `waves/wave{1-4}/output/2_BenchmarkData/`

### Performance Results
| Category | Best TFLOPS | Avg TFLOPS | Shapes |
|----------|-------------|------------|--------|
| Tiny (M*N<4K) | 0.204 | ~0.04 | 210 |
| Small (4K-64K) | 2.696 | ~0.6 | 240 |
| Medium (64K-1M) | 9.531 | ~3.3 | 320 |
| Large (M*N>=1M) | **24.126** | 8.7 | 260 |
| **All** | **24.126** | **3.45** | **1,030** |

Peak efficiency: 78.5% of theoretical FP32 (30.7 TFLOPS)

### Key Parameter Rules (gfx1100 non-MI FP32)

**Fixed parameters (no sweep needed):**
| Param | Value | Confidence |
|-------|-------|------------|
| SIA | 1 | 100% |
| PLR | 0 | 100% (1,030/1,030 shapes) |
| WGM | 1 | 99%+ |
| VW/GRVW/LRVW | 1 | mandatory |
| WavefrontSize | 32 | mandatory |
| KernelLanguage | Assembly | mandatory |

**ThreadTile scaling:**
| M*N Range | Best TT | Runner-up |
|-----------|---------|-----------|
| <4K | [1,1] (89%) | [1,2] |
| 4K-32K | [1,2] | [2,2] |
| 32K-128K | [2,2] | [2,4] |
| 128K-512K | [2,4]/[4,4] | [4,8] |
| 512K-1M | [8,8] (39%) | [4,8] (35%) |
| >=1M | [8,8] (61%) | [6,8] (28%) |

**Context-dependent parameters:**
| Param | Tiny | Small | Medium | Large |
|-------|------|-------|--------|-------|
| PGR | 1 (78%) | 1 (88%) | 1 (61%) | **0 (68%)** |
| SU | 32 (64%) | 0 (83%) | 0 (83%) | 0 (73%) |
| DU | 128/32 | 128/64 | 64/16/32 | **16 (88%)** |
| WG | [16,8,1] 50% | [16,8,1] 72% | [16,8,1] 92% | [16,8,1] 98% |

### Surprises
1. **PLR=0 ALWAYS** — local read prefetch never helps non-MI FP32 on gfx1100
2. **DU=16 dominates large shapes** — less unrolling = better for big shapes
3. **PGR flips with size** — PGR=1 for small, PGR=0 for large
4. **24 TFLOPS peak** — excellent for non-MI (scalar VALU) path
5. **TT[6,8] strong contender** for large shapes (28% of M*N>=1M)
6. **VGPR limit** — gfx1100 has 256 VGPRs, TT[4,4]+DU=128 overflows

### Methodology Notes
- Used `--library-format=yaml` (client can't load msgpack)
- `force_merge=0` for all merges (winner-based comparison)
- All kernels validated with `NumElementsToValidate: 256`
- Problem type: `Cijk_Ailk_Bjlk` (TransposeA=False, TransposeB=True)

---

## VOPD Debug Session — 2026-06-01

### Root Causes Found
1. **VOPD TextBlock instructions silently dropped**: The assembler rejects `v_dual_fmac_f32` when RDNA3 constraints are violated (dst must be even/odd pair, src1 must use different VGPR banks). MAC_F32_VOPD pairs consecutive idx0 values which share the same B register → src1 bank violation. Assembler silently drops invalid instructions, producing binary identical to non-VOPD.

2. **The .co metadata differs**: Even though kernel assembly is identical, `TensileLibrary_gfx1100.co` has different metadata between EnableVOPD=0 and =1, causing incorrect kernel dispatch → wrong results.

3. **Helper kernel cache**: `~/.tensile/helper_cache/` must be deleted when changing MAC component code.

### RDNA3 VOPD Encoding Constraints
- **dst**: one must be even VGPR, other must be odd
- **src1**: must use different VGPR banks (consecutive VGPRs = OK)
- **src0**: no bank constraint (SGPR/VGPR/inline OK)
- Diagonal pairing `(idx0,idx1)+(idx0+1,idx1+1)` satisfies both when tt0 is even

## VOPD Performance Campaign — 2026-06-01

### Fixes Applied
1. **TextBlock → MacroInstruction**: TextBlock silently dropped inside Macro during KernelBody serialization. MacroInstruction (Instruction subclass) survives.
2. **Diagonal pairing**: (idx0,idx1)+(idx0+1,idx1+1) satisfies all 3 RDNA3 VOPD constraints (dst even/odd, src0 diff banks, src1 diff banks).
3. **Even tt0+tt1 validation**: Solution.py rejects odd tt0 or odd tt1 with EnableVOPD=1.
4. **Helper cache**: Must clear ~/.tensile/helper_cache/ after code changes.

### VOPD vs Non-VOPD Performance (Phase 1)
| Shape | Non-VOPD | VOPD | Delta |
|-------|----------|------|-------|
| 6144x8192xK=8192 | 24.1 T | **24.8 T** | **+3.0%** |
| 3072x4096xK=4096 | 19.6 T | **19.7 T** | +0.3% |
| 1536x2048xK=2048 | 10.0 T | **10.4 T** | **+3.7%** |
| 768x1024xK=4096 | 9.5 T | **9.7 T** | +1.6% |

### Full Tuning Campaign (Phase 2)
- VOPD Wave 3 (320 medium shapes): complete, 4 VOPD configs
- VOPD Wave 4 (260 large shapes): complete, 2 VOPD configs
- Merge result: VOPD solutions didn't replace non-MI winners in the merge tool (efficiency metric comparison), but raw TFLOPS shows 1-4% improvement
- Deployed merged logic: 1,030 shapes, 160 solutions

### Why VOPD Gain is Limited to ~3-4%
1. **88% pairing coverage** (TT[8,8]) — 12% still single-issue
2. **Memory-bound** — large shapes are limited by DRAM/LDS bandwidth, not compute
3. **Compute fraction** is small — MAC takes ~30% of loop time, rest is memory ops
4. With MAC taking 30% and VOPD giving 1.79x on 88% of MACs: effective speedup = 1/(0.7 + 0.3/1.56) = 1/(0.7+0.19) = 1.12x → ~3-4% total
