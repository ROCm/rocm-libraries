# VOPD FP32 Full Tuning Campaign Playbook — gfx1100

## Context

VOPD dual-issue gives +20-49% FP32 uplift on gfx1100 (proven across 12+ shapes, 7 iterations). The existing non-VOPD campaign covers 1,030 production shapes with 162 solutions. We now re-tune all shapes with VOPD to produce a new production logic YAML.

Key findings from pre-campaign iterations:
- **MT128x128x8** is the universal winner for medium/large shapes
- **MT256x64** wins narrow-N shapes, **MT64x256** wins narrow-M shapes
- **MT256x128x16** wins small shapes (2048x2048 class)
- **WGM=4** beats WGM=8 on most shapes
- **DU=4-8** wins balanced shapes, **DU=16** wins deep-K
- **GSU=4-8** massive win (+24%) on K-dominant shapes (K > 4×max(M,N))
- **LdsPadB>0 now works** with VDualFmacF32 fix — LPB=2 gives +3% (constraint removed)
- **PGR=1 + PLR=1** is the universal best pipeline config
- StaggerUStride=64 slightly better than 256 on GDDR6

## Infrastructure

**Tensile command:**
```bash
/home/vmijovic/TheRock/rocm-libraries/projects/hipblaslt/tensilelite/build_tmp/Tensile.sh <yaml> <output_dir>
```

**Existing scripts at `/home/vmijovic/vopd_sgemm/fp32_tuning/`:**
- `extract_shapes.py` — Extract shapes from reference logic YAML
- `gen_wave_yaml.py` — Generate per-wave benchmark YAMLs
- `gen_logic_yaml.py` — Convert wave CSVs to logic YAML
- `merge_logic.py` — Merge wave winners into production logic
- `parse_results.py` — Parse benchmark CSVs

**Shape source:** `/home/vmijovic/vopd_sgemm/fp32_tuning/logic/all_winners.json` (1,030 shapes)

---

## Wave Grouping

| Wave | Shapes | M×N Range | Strategy |
|------|--------|-----------|----------|
| **Wave 1** | 210 | M×N < 4K (tiny) | **Keep non-VOPD** — TT[1,1]/[1,2]/[2,1] can't do VOPD (needs even×even ≥ [4,4]) |
| **Wave 2** | 240 | 4K ≤ M×N < 64K | **Mixed** — VOPD with small tiles + non-VOPD fallback, GSU for deep-K |
| **Wave 3** | 320 | 64K ≤ M×N < 1M | **VOPD primary** — MT128x128 dominates |
| **Wave 4** | 260 | M×N ≥ 1M | **VOPD primary** — MT128x128 proven king |

---

## Parameter Recipes

### Recipe A: VOPD Large (Wave 3-4)
```yaml
EnableVOPD: [1]
VectorWidthA: [2]
VectorWidthB: [2]
PrefetchGlobalRead: [1]
PrefetchLocalRead: [1]
ThreadTile: [[8,16], [16,8], [8,8]]
WorkGroup: [[16,8,1], [8,8,1], [8,16,1]]
DepthU: [4, 8, 16]
WorkGroupMapping: [4, 8]
StaggerU: [0, 32]
LdsPadA: [0, 2]
LdsPadB: [0, 2, 4]
GlobalSplitU: [1]
```
Fork configs: 3×3×3×2×2×2×3×1 = **648** (many rejected, ~200 survive)

### Recipe A+ : Shape-Aware Tile Additions
Inject per-shape based on aspect ratio:
- If N < 0.5×M → add TT[16,8] WG[16,8,1] (→ MT256x64)
- If M < 0.5×N → add TT[8,16] WG[8,32,1] (→ MT64x256)
- If K > 4×max(M,N) → add GSU=[2,4] and DU=[16,32]

### Recipe B: VOPD Medium (Wave 2)
```yaml
EnableVOPD: [1]
VectorWidthA: [2]
VectorWidthB: [2]
PrefetchGlobalRead: [1]
PrefetchLocalRead: [1]
ThreadTile: [[8,16], [16,8], [8,8], [4,8], [8,4], [4,4]]
WorkGroup: [[16,8,1], [8,8,1], [8,16,1], [16,16,1]]
DepthU: [8, 16, 32]
WorkGroupMapping: [4, 8]
StaggerU: [0, 32]
LdsPadA: [0, 2]
LdsPadB: [0, 2]
GlobalSplitU: [1, 2, 4]  # for K-dominant shapes
```

### Recipe C: Non-VOPD (Wave 1 — no changes needed)
Keep existing non-VOPD winners from `all_winners.json`. No re-tuning.

---

## Execution Steps

### Step 1: Generate Campaign YAMLs
Modify `gen_wave_yaml.py` to emit VOPD recipes. Create:
- `vopd_campaign/wave2.yaml` — Recipe B, 240 shapes
- `vopd_campaign/wave3.yaml` — Recipe A, 320 shapes
- `vopd_campaign/wave4.yaml` — Recipe A, 260 shapes

Split large waves into sub-YAMLs if >150 shapes to keep runtime <1h per run.

### Step 2: Run Waves (timed, logged)
Each run is wrapped with `time` and timestamps logged to help optimize the process later.
```bash
TENSILE=~/TheRock/rocm-libraries/projects/hipblaslt/tensilelite/build_tmp/Tensile.sh
DIR=~/vopd_sgemm/fp32_tuning/vopd_campaign

for wave in wave2 wave3 wave4; do
  echo "=== START $wave: $(date -Iseconds) ===" | tee -a $DIR/timing.log
  time $TENSILE $DIR/$wave.yaml $DIR/${wave}_out 2>&1 | tee $DIR/$wave.log
  echo "=== END $wave: $(date -Iseconds) ===" | tee -a $DIR/timing.log
done
```
After each wave, record in `timing.log`:
- Start/end timestamps
- Wall-clock duration (from `time`)
- Number of shapes and kernels (parsed from CSV header)
- Seconds per shape (total_time / num_shapes)

**Time estimates (no validation):**
- Wave 2: ~240 shapes × ~300 configs × 0.5s ≈ 45 min
- Wave 3: ~320 shapes × ~200 configs × 0.5s ≈ 30 min
- Wave 4: ~260 shapes × ~200 configs × 0.5s ≈ 25 min
- **Total: ~1.5-2 hours**

### Step 3: Parse & Pick Winners
For each shape, compare VOPD best vs existing non-VOPD best from `all_winners.json`:
- VOPD wins → use VOPD solution
- Non-VOPD wins → keep existing (unlikely on Wave 3-4, possible on Wave 2 tiny shapes)

Output: `vopd_campaign/winners.json`

### Step 4: Spot-Check Refinement
Take bottom-10% shapes (lowest VOPD uplift) and re-run with wider search. These are likely edge cases where default recipes miss the optimal config.

### Step 5: Merge Logic YAML
Combine:
- Wave 1: existing non-VOPD winners (210 shapes)
- Wave 2-4: VOPD winners (820 shapes, or non-VOPD if VOPD lost)

Output: `vopd_campaign/gfx1100_Cijk_Ailk_Bjlk_S_B_UserArgs.yaml`

### Step 6: Validate
Run merged logic on 5% random sample (50 shapes) with `NumElementsToValidate: 256`.

---

## GlobalParameters for All Campaign YAMLs

```yaml
GlobalParameters:
  MinimumRequiredVersion: 5.0.0
  CMakeBuildType: Release
  EnqueuesPerSync: 4
  SyncsPerBenchmark: 4
  NumElementsToValidate: 0
  KernelTime: True
  PrintWinnersOnly: False
  PrintSolutionRejectionReason: True
  NumBenchmarks: 1
```

---

## Known Constraints (Hard Rules)

1. ThreadTile must be even×even for VOPD (2x2 block diagonal pairing)
2. SIA must be 1 (SIA≥2 needs MFMA infrastructure)
3. InnerUnroll must be 1
4. PLR=2 gives wrong results (pre-existing Tensile bug)
5. WavefrontSize=32 mandatory (gfx1100)
6. EnableMatrixInstruction=False (VOPD uses VALU, not matrix unit)
7. ~~LdsPadB must be 0~~ **REMOVED** — LPB works with VDualFmacF32 fix

---

## Quality Gates

1. Every shape has a winner with GFlops > 0
2. No regressions vs non-VOPD baseline on any shape
3. VOPD winners validated on ≥5% random sample
4. Logic YAML loads in Tensile client (`--library-format=yaml`)
5. Campaign report with per-wave uplift stats

---

## Outputs — ALL PRESERVED

**STRICT RULE: No overwriting, no deleting.** Every Tensile run gets a unique timestamped directory. If a run needs to be re-done, create a new output dir (e.g. `wave3_out_v2`), never `rm -rf` an existing one. All logs, CSVs, kernels, and build artifacts are permanent.

```
~/vopd_sgemm/fp32_tuning/vopd_campaign/
├── wave2.yaml                              # Input YAML
├── wave2.log                               # Full console output (tee'd)
├── wave2_out/                              # Complete Tensile output tree
│   ├── 1_BenchmarkProblems/
│   │   └── Cijk_Ailk_Bjlk_S_B_UserArgs_00/
│   │       ├── 00_Final/
│   │       │   ├── build/run.sh            # Benchmark run script
│   │       │   └── caches/<hash>/source/   # Built kernels, .hsaco, Kernels.cpp
│   │       └── Data/00_Final.csv           # Per-kernel per-shape GFlops
│   ├── 2_BenchmarkData/
│   │   └── *.csv                           # Aggregate benchmark data
│   └── 3_LibraryLogic/                     # Auto-generated logic (if enabled)
├── wave3.yaml / wave3.log / wave3_out/     # Same structure
├── wave4.yaml / wave4.log / wave4_out/     # Same structure
├── winners.json                            # Per-shape winner mapping
├── gfx1100_Cijk_Ailk_Bjlk_S_B_UserArgs.yaml  # Production logic YAML
└── campaign_report.md                      # Before/after per wave, total uplift
```

**What's in each `_out/` directory:**
- `Data/00_Final.csv` — raw benchmark results (GFlops per kernel per shape)
- `caches/*/source/` — compiled kernels (.hsaco), assembly source, Kernels.cpp
- `build/run.sh` — exact command to reproduce the benchmark run

**Shapes:** 1,020 (all ≤ 8192 per dimension, 10 M=12288 shapes removed)
