# VOPD FP32 GEMM Tuning — gfx1151 Handoff

> **SELF-CONTAINED:** every script/config below is embedded inline — save each to the
> indicated filename. Nothing outside the hipBLASLt clone is required.

Replicating the VOPD (dual-issue `v_dual_fmac_f32`) FP32 GEMM tuning work on a machine
with an **AMD gfx1151** GPU (Strix Halo APU, RDNA3.5, wave32). The original work was done
on **gfx1100 / RX 7900 XTX** (96 CU discrete, RDNA3). The VOPD feature code has **no
architecture gating** beyond `wave32 + FP32 + non-MatrixInstruction`, so it works on
gfx1151 unchanged — only the *tuned logic* (best tile per shape) must be regenerated for
your hardware, because the APU has different CU count / clocks / memory bandwidth.

This document assumes ONLY a fresh clone of `ROCm/rocm-libraries` (branch
`vmijovic/add_vopd`) exists. Every helper script and config the original work used has been
**embedded inline below** — create a working dir (e.g. `~/tune`) and save each block to the
indicated filename. All paths below use these roots:
- **Repo:** `~/TheRock/rocm-libraries` (hipBLASLt lives under `projects/hipblaslt`)
- **`$HB`** = `~/TheRock/rocm-libraries/projects/hipblaslt`
- **Work dir:** `~/tune` (where you save the embedded scripts/configs — pick anything)

---

## 0. Index of embedded files

Save each of these from the code block in the indicated section:

| Save as | Section | Purpose |
|---------|---------|---------|
| `create_library.yaml` | §4.3 | LibraryLogic arch config (gfx1151) |
| `wave_template.yaml` | §4.1 | Campaign config template (compact, extend ProblemSizes) |
| `seed.yaml` | §11 | **SEEDED FAST-TUNING config** — pruned fork grid (recommended start) |
| `build_merge.sh` | §4.5 | Per-wave LibraryLogic gen + sequential force_merge |
| `roundtrip.sh` | §7.1 | Generic-vs-tuned A/B verification harness (NT/TN/NN) |
| `bench_heur.sh` | §7.2 | Fast heuristic-only best-of-3 bench |
| `bench_stable.sh` | §7.2 | Stable best-of-N heur + algo_method=all bench |
| `shapes_big.txt` | §7.3 | 30-shape verification shape list |
| `extract_cluster.py` | §12 | Extract winning-config cluster from a wave CSV → build your own seed |

---

## 1. Quick Start

```bash
# 1) Get the branch with the VOPD feature + reference gfx1100 logic
git clone git@github.com:ROCm/rocm-libraries.git ~/TheRock/rocm-libraries
cd ~/TheRock/rocm-libraries/projects/hipblaslt
git checkout vmijovic/add_vopd        # contains all VOPD code + gfx1100 tuned logic
git log --oneline -3                  # expect f079c7fdce3 at HEAD

# 2) Build hipBLASLt + clients for gfx1151 (one time; includes the VOPD rocisa C++)
pip install --break-system-packages msgpack PyYAML   # needed by system python for codegen
invoke build -ca gfx1151 -d           # -d installs deps on first run; drops a build/ tree

# 3) Set up a work dir and save the embedded scripts/configs (see §0 index)
mkdir -p ~/tune && cd ~/tune
#   ... create seed.yaml, build_merge.sh, roundtrip.sh, bench_*.sh, shapes_big.txt,
#       extract_cluster.py, create_library.yaml from the code blocks below ...

# 4) Recommended fast path: run the SEEDED grid (§11), then merge+deploy+verify.
HB=~/TheRock/rocm-libraries/projects/hipblaslt
$HB/tensilelite/Tensile/bin/Tensile  ~/tune/seed.yaml  ~/tune/seed_out
# ...generate LibraryLogic, force_merge, deploy to gfx1151/ tree, rebuild device lib,
#    benchmark. All detailed below.
```

> The build directory created by `invoke build` is `build/` (this guide also refers to a
> `build/release/...` layout used by the original raw-cmake setup — adjust `build/release`
> to your actual build dir, e.g. plain `build`, if you used `invoke build`).

---

## 2. Branch & Build

| Item | Value |
|------|-------|
| Remote | `git@github.com:ROCm/rocm-libraries.git` |
| Branch | `vmijovic/add_vopd` |
| HEAD commit | `f079c7fdce3` *Add VOPD FP32 tuned logic: NT regression-free merge + TN orientation* |

Relevant commits (newest first):

| Hash | What it adds | Files |
|------|--------------|-------|
| `f079c7fdce3` | Updated NT (`Cijk_Ailk_Bjlk`) regression-free merged logic + new TN (`Cijk_Alik_Bljk`) logic, gfx1100. | `gfx1100_Cijk_Ailk_Bjlk_S_B_UserArgs.yaml`, `gfx1100_Cijk_Alik_Bljk_S_B_UserArgs.yaml` |
| `771965cb5df` | First gfx1100 NT tuned logic (830 shapes / 252 solutions). | `gfx1100_Cijk_Ailk_Bjlk_S_B_UserArgs.yaml` |
| `f93db5958c6` | **The VOPD feature code itself** (codegen + validation). | 11 files — see §3 |

The tuned-logic commits (`f079`, `7719`) are **gfx1100-specific reference data**. You will
generate your own `gfx1151_*.yaml` equivalents. The feature commit `f93db5958c6` is
architecture-agnostic and is what makes VOPD kernels generatable at all.

Build notes:
- `invoke build -ca gfx1151` builds host lib + device lib + clients. The VOPD C++ in
  `rocisa` is compiled as part of this — no separate rocisa step needed for a fresh build.
- The build is slow the first time (device library). To iterate on *just* the FP32 GridBased
  logic later, scope it: `invoke build -ca gfx1151 -f 'gfx1151/GridBased/*'`.
- System python (not just the venv) must have `msgpack` for the standalone
  `TensileCreateLibrary` device-lib rebuild used during tuning: `pip install
  --break-system-packages msgpack`.

---

## 3. Code Changes — what VOPD adds (commit `f93db5958c6`)

VOPD packs **two** FP32 FMA ops into one issue slot via the RDNA3+ `v_dual_fmac_f32`
encoding. Files changed (all under `$HB`):

| File | Change |
|------|--------|
| `tensilelite/rocisa/rocisa/include/instruction/common.hpp` | New `VDualFmacF32` C++ instruction struct (emits `v_dual_fmac_f32 ... :: v_dual_fmac_f32 ...`) with full register dependency tracking. |
| `tensilelite/rocisa/rocisa/src/instruction/common.cpp` | Nanobind registration so Python can construct `VDualFmacF32`. |
| `tensilelite/rocisa/rocisa/src/pass/insert_delay_alu.cpp` | Teaches the delay-ALU pass that `v_dual_fmac_f32` reads its dst as an accumulator (correct hazard insertion). |
| `tensilelite/rocisa/rocisa/include/hardware_caps.hpp` | Capability probe: assembles `v_dual_fmac_f32 v0,v4,v8 :: v_dual_fmac_f32 v1,v5,v9` (wave32) and records `asmCaps["v_dual_fmac_f32"]`. **Auto-detects per arch via the assembler** — no hardcoded gfx list. |
| `tensilelite/Tensile/Components/MAC_F32_VOPD.py` | **NEW** component. 2×2 block-diagonal pairing for 100% VOPD coverage on even×even ThreadTile. |
| `tensilelite/Tensile/Components/MAC_F32.py` | Excludes itself when `EnableVOPD` so the VOPD MAC is selected instead. |
| `tensilelite/Tensile/Components/__init__.py` | Registers `MAC_F32_VOPD`. |
| `tensilelite/Tensile/SolutionStructs/Solution.py` | VOPD validation block (§3.1) + auto `VectorWidthA/B = 2` defaults when `EnableVOPD` and FP32. |
| `tensilelite/Tensile/Common/ValidParameters.py` | `"EnableVOPD": [0, 1]`. |
| `tensilelite/Tensile/Common/GlobalParameters.py` | `EnableVOPD` default `[0]`. |
| `tensilelite/Tensile/KernelWriter.py` | Minor hookup for the VOPD path. |

### 3.1 The VOPD validation constraints (Solution.py ~line 693-711)

```python
# VOPD dual-issue validation (RDNA3+ FP32 non-MI only)
# Uses 2x2 block diagonal pairing for 100% coverage on even×even TT.
if state.get("EnableVOPD", 0) == 1:
  if state["EnableMatrixInstruction"]:
    reject(... "EnableVOPD requires EnableMatrixInstruction=False")
  if state["WavefrontSize"] != 32:
    reject(... "EnableVOPD requires WavefrontSize=32")
  if not (state["ProblemType"]["DataType"].isSingle()):
    reject(... "EnableVOPD requires FP32 data type")
  if state["ThreadTile0"] % 2 != 0:
    reject(... "EnableVOPD requires even ThreadTile0")
  if state["ThreadTile1"] % 2 != 0:
    reject(... "EnableVOPD requires even ThreadTile1")
```

And the auto-VW defaults (~line 1931 / 1953):

```python
if state.get("EnableVOPD", 0) == 1 and state["ProblemType"]["DataType"].isSingle():
    state["VectorWidthA"] = 2   # (same for VectorWidthB)
```

**Why gfx1151 is fine:** the requirements are only `non-MI + wave32 + FP32 + even
ThreadTile0/1`. gfx1151 is **RDNA3.5, wave32**, and `v_dual_fmac_f32` is an RDNA3+
instruction. There is **no architecture allowlist** — the `hardware_caps` probe asks the
assembler whether the ISA supports `v_dual_fmac_f32`. As long as your ROCm assembler
accepts that instruction for gfx1151 (it should), `asmCaps["v_dual_fmac_f32"]` is true and
VOPD kernels are generated. **If the probe returns false, VOPD solutions are silently
rejected** — see Gotchas §8.

---

## 4. Tuning Workflow (step by step)

The original 1030-shape campaign is split into 4 "waves" of YAML. You don't have those
files on this machine, so this doc gives you a **compact campaign template** (§4.1) you
extend, plus a **seeded fast config** (§11) that is the recommended starting point. Both
share the same header — only `ForkParameters` and the `ProblemSizes` list differ.

### 4.1 Campaign config TEMPLATE — save as `wave_template.yaml`

This is the full-grid NT template (the original waves used this header + ForkParameters).
It contains a **short representative ProblemSizes list (~18 shapes)** spanning
small / medium / large / skinny / GEMV. **Extend `ProblemSizes` to your workload set** (see
the note after the block). The original campaign had ~1030 such `Exact` entries.

```yaml
# save as: wave_template.yaml
# NT orientation (TransposeA:false, TransposeB:true). For other orientations change
# ONLY the TransposeA/TransposeB line per the Orientation Map (§5). Arch is NOT set here.
GlobalParameters: {MinimumRequiredVersion: 5.0.0, CMakeBuildType: Release, EnqueuesPerSync: 4, SyncsPerBenchmark: 4, NumElementsToValidate: 0,
  KernelTime: true, PrintWinnersOnly: false, PrintSolutionRejectionReason: true, NumBenchmarks: 1}
BenchmarkProblems:
- - {OperationType: GEMM, DataType: s, DestDataType: s, ComputeDataType: s, TransposeA: false, TransposeB: true, UseBeta: true,
    Batched: true}
  - InitialSolutionParameters: null
    BenchmarkCommonParameters:
    - KernelLanguage: [Assembly]
    - ScheduleIterAlg: [1]
    - ScheduleGlobalRead: [0]
    - ScheduleLocalWrite: [0]
    - EnableVOPD: [1]            # turn VOPD on
    - VectorWidthA: [2]          # VOPD wants VW=2
    - VectorWidthB: [2]
    - PrefetchGlobalRead: [1]
    - PrefetchLocalRead: [1]
    ForkParameters:             # the FULL search grid (3072 configs)
    - ThreadTile:
      - [4, 4]
      - [4, 8]
      - [8, 4]
      - [8, 8]                  # all even (VOPD requirement)
    - WorkGroup:
      - [16, 8, 1]
      - [8, 8, 1]
      - [8, 4, 1]
      - [4, 8, 1]
    - DepthU: [8, 16, 32, 64]
    - GlobalSplitU: [1, 2, 4]
    - WorkGroupMapping: [4, 8]
    - StaggerU: [0, 32]
    - LdsPadA: [0, 2]
    - LdsPadB: [0, 2]
    BenchmarkFinalParameters:
    - ProblemSizes:
      # [M, N, batch, K] — representative subset spanning the original 1030-shape set.
      # small / square:
      - Exact: [16, 16, 1, 16]
      - Exact: [64, 64, 1, 64]
      - Exact: [128, 128, 1, 128]
      - Exact: [256, 256, 1, 256]
      # medium / large square:
      - Exact: [512, 512, 1, 512]
      - Exact: [1024, 1024, 1, 1024]
      - Exact: [2048, 2048, 1, 2048]
      - Exact: [4096, 4096, 1, 4096]
      - Exact: [8192, 8192, 1, 8192]
      # large rectangular / deep-K:
      - Exact: [4096, 4096, 1, 8192]
      - Exact: [2048, 2048, 1, 8192]
      - Exact: [6144, 8192, 1, 8192]
      # skinny (small M, large N):
      - Exact: [16, 4096, 1, 4096]
      - Exact: [128, 8192, 1, 8192]
      - Exact: [768, 8192, 1, 8192]
      # GEMV / M=1 (memory-bound, use best-of-N when benching):
      - Exact: [1, 4096, 1, 4096]
      - Exact: [1, 8192, 1, 8192]
      - Exact: [6144, 1, 1, 4096]
```

**How to extend `ProblemSizes`:** the original campaign swept a grid of
`M ∈ {16,64,128,256,384,768,1536,3072,6144}` × `N ∈ {16,64,128,256,512,1024,2048,4096,8192}`
× `K ∈ {1,16,64,128,256,512,1024,2048,4096,8192}` plus GEMV rows (`M=1` / `N=1`) and a few
large squares (`4096³`, `8192³`). Each entry is one `- Exact: [M, N, 1, K]` line under
`ProblemSizes`. Generate them however you like; e.g. a one-liner:

```bash
python3 - <<'PY' >> ProblemSizes.txt
for M in (16,64,128,256,384,768,1536,3072,6144):
  for N in (16,64,128,256,512,1024,2048,4096,8192):
    for K in (1,16,64,128,256,512,1024,2048,4096,8192):
      print(f"      - Exact: [{M}, {N}, 1, {K}]")
PY
```

Splitting into "waves" is just for runtime/manageability — each wave is the same header
with a different `ProblemSizes` slice. The original split was roughly:
wave1 = small squares (~50), wave2 = GEMV/M=1 (~240), wave3 = skinny small-M (~320),
wave4 = large/rect (~250). You can run one big file instead if you have the time budget.

> **gfx1151 note:** the campaign YAML does *not* contain an arch string — arch comes from
> the GPU you run on. So you can run these unchanged on gfx1151. Only the
> `create_library.yaml` (§4.3) carries the arch name.

### 4.2 Run a wave (per orientation)

```bash
HB=~/TheRock/rocm-libraries/projects/hipblaslt
$HB/tensilelite/Tensile/bin/Tensile  ~/tune/wave_template.yaml  ~/tune/wave1_out
# produces wave1_out/{1_BenchmarkProblems, 2_BenchmarkData}
# 2_BenchmarkData/Cijk_<orient>_S_B_UserArgs_00.{csv,yaml} hold the per-shape winners
```

Repeat per wave file you create.

### 4.3 LibraryLogic arch config — save as `create_library.yaml`

This is the ONLY file carrying the architecture name. **Set it to gfx1151.**

```yaml
# save as: create_library.yaml
GlobalParameters:
  MinimumRequiredVersion: 5.0.0

LibraryLogic:
    ScheduleName: "gfx1151"
    ArchitectureName: "gfx1151"
    LibraryType: "GridBased"
```

### 4.4 Generate LibraryLogic per wave (manual)

For each wave, build a small logic dir, copy that wave's benchmark data into it, and run
Tensile on the `create_library.yaml`:

```bash
HB=~/TheRock/rocm-libraries/projects/hipblaslt
LG=~/tune/wave1_logic
rm -rf "$LG"; mkdir -p "$LG/2_BenchmarkData"
KNAME=Cijk_Ailk_Bjlk_S_B_UserArgs     # NT base name (see Orientation Map §5)
cp ~/tune/wave1_out/2_BenchmarkData/${KNAME}_00.csv  "$LG/2_BenchmarkData/"
cp ~/tune/wave1_out/2_BenchmarkData/${KNAME}_00.yaml "$LG/2_BenchmarkData/"
cp ~/tune/create_library.yaml "$LG/create_library.yaml"
$HB/tensilelite/Tensile/bin/Tensile "$LG/create_library.yaml" "$LG"
# -> $LG/3_LibraryLogic/*.yaml   (one GridBased logic file)
```

The `build_merge.sh` script in §4.5 loops this over all waves automatically.

### 4.5 Per-wave LibraryLogic + sequential force_merge — save as `build_merge.sh`

This is the LibraryLogic-gen + `force_merge` recipe, **generalized for gfx1151** and
parameterized by `KNAME`/orientation (set `KNAME` to the base name from the Orientation
Map §5). Adjust `NWAVES` to how many wave dirs you produced.

```bash
#!/bin/bash
# save as: build_merge.sh   (chmod +x build_merge.sh)
# Build production logic for gfx1151: per-wave LibraryLogic gen, then sequential
# force_merge of all waves into one logic dir.
# Usage:  ./build_merge.sh <KNAME> [NWAVES]
#   KNAME  = kernel/library base name for the orientation you tuned, e.g.:
#            NT -> Cijk_Ailk_Bjlk_S_B_UserArgs
#            TN -> Cijk_Alik_Bljk_S_B_UserArgs
#            NN -> Cijk_Ailk_Bljk_S_B_UserArgs   (see Orientation Map §5)
#   NWAVES = number of wave_out dirs you have (default 4). Expects ~/tune/wave${w}_out.
# Run AFTER all waves finish.
set -eu
HB=~/TheRock/rocm-libraries/projects/hipblaslt
TC=~/tune
TENSILE=$HB/tensilelite/Tensile/bin/Tensile
MERGE=$HB/tensilelite/Tensile/bin/TensileMergeLibrary
KNAME=${1:?usage: build_merge.sh <KNAME> [NWAVES]}
NWAVES=${2:-4}

# create_library.yaml carries the arch name -> gfx1151 (write it if missing)
cat > "$TC/create_library.yaml" <<'EOF'
GlobalParameters:
  MinimumRequiredVersion: 5.0.0

LibraryLogic:
    ScheduleName: "gfx1151"
    ArchitectureName: "gfx1151"
    LibraryType: "GridBased"
EOF

# 1) Per-wave LibraryLogic generation
for w in $(seq 1 "$NWAVES"); do
  LG=$TC/wave${w}_logic
  echo "=== gen LibraryLogic wave$w ==="
  rm -rf "$LG"; mkdir -p "$LG/2_BenchmarkData"
  cp "$TC/wave${w}_out/2_BenchmarkData/${KNAME}_00.csv"  "$LG/2_BenchmarkData/"
  cp "$TC/wave${w}_out/2_BenchmarkData/${KNAME}_00.yaml" "$LG/2_BenchmarkData/"
  cp "$TC/create_library.yaml" "$LG/create_library.yaml"
  "$TENSILE" "$LG/create_library.yaml" "$LG" > "$LG/gen.log" 2>&1
  echo "  -> $(find "$LG/3_LibraryLogic" -name '*.yaml' | head -1)"
done

# 2) Sequential force_merge of the waves (wave1 -> +wave2 -> ... -> +waveN).
# Untested shapes fall back to the existing navi3x generic kernels (already in the build).
# --force_merge 1 is ESSENTIAL: without it the merger drops all incremental entries on any
# efficiency-scale mismatch (you'd keep none of your new winners).
PREV=$TC/wave1_logic/3_LibraryLogic
for w in $(seq 2 "$NWAVES"); do
  OUT=$TC/merge_step$w
  rm -rf "$OUT"; mkdir -p "$OUT"
  echo "=== force_merge wave$w onto previous ==="
  "$MERGE" --force_merge 1 "$PREV" "$TC/wave${w}_logic/3_LibraryLogic" "$OUT" 2>&1 | tail -3
  PREV=$OUT
done

rm -rf "$TC/merged_logic"; cp -r "$PREV" "$TC/merged_logic"
echo "=== merged logic: $(find $TC/merged_logic -name '*.yaml' | head -1) ==="
find "$TC/merged_logic" -name '*.yaml' -exec wc -l {} \;
```

`TensileMergeLibrary` takes **directories** of logic YAML (original, incremental, output).

**Why `--force_merge 1` is essential:** without it, the merger compares the "efficiency
scale" of the two logic sets and, on any mismatch, **drops all incremental entries** — you
end up keeping none of your new winners. `force_merge=1` bypasses that gate and unions the
shapes. Untuned shapes fall back to the in-build navi3x generic FP32 kernels (see §6). The
original NT "merged_v3" was verified regression-free this way.

---

## 5. Orientation Map (CRITICAL)

Each transpose combo maps to a different library/kernel-family name. Only the `TransposeA/
TransposeB` line in the campaign YAML changes between orientations; everything else is
identical. **Tune and deploy each orientation separately.**

| Orientation | `TransposeA` | `TransposeB` | bench `--transA/--transB` | Library / kernel base name |
|-------------|:------------:|:------------:|:-------------------------:|----------------------------|
| **NN** | false | false | `N N` | `Cijk_Ailk_Bljk` |
| **NT** | false | true  | `N T` | `Cijk_Ailk_Bjlk` |
| **TN** | true  | false | `T N` | `Cijk_Alik_Bljk` |
| **TT** | true  | true  | `T T` | `Cijk_Alik_Bjlk` |

Full deployed YAML filename is `gfx1151_<base>_S_B_UserArgs.yaml`
(e.g. NT → `gfx1151_Cijk_Ailk_Bjlk_S_B_UserArgs.yaml`).

> The original work tuned **NT** (`Cijk_Ailk_Bjlk`) and **TN** (`Cijk_Alik_Bljk`) for
> gfx1100. NN was seeded with `seed.yaml` (§11). Pick the orientations your workloads use.

---

## 6. Merge & Deploy

Deploy the merged logic into the **gfx1151** tree (NOT gfx1100). The gfx1151 logic tree
already exists in the repo:

```
$HB/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx1151/
  ├── Equality/         (existing)
  └── GridBased/        (exists, but has NO FP32 *_S_B_UserArgs.yaml yet — you add it)
```

The corresponding gfx1100 dir contains the reference files you are reproducing:
`.../gfx1100/GridBased/gfx1100_Cijk_Ailk_Bjlk_S_B_UserArgs.yaml` and
`gfx1100_Cijk_Alik_Bljk_S_B_UserArgs.yaml`.

Deploy (NT example — repeat per orientation with the right base name):

```bash
HB=~/TheRock/rocm-libraries/projects/hipblaslt
GBDIR=$HB/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx1151/GridBased
MERGED=$(find ~/tune/merged_logic -name '*.yaml' | head -1)
cp "$MERGED" "$GBDIR/gfx1151_Cijk_Ailk_Bjlk_S_B_UserArgs.yaml"
```

Then rebuild **only the device library** (fast, ~40s — no full hipBLASLt rebuild). Change
the arch and the `build/release` path to match your build:

```bash
cd "$HB"
PYTHONPATH=build/release/tensilelite/rocisa:build/release/tensilelite:tensilelite \
python3 -m Tensile.TensileCreateLibrary \
  --architecture=gfx1151 \
  --cxx-compiler=/opt/rocm/bin/amdclang++ \
  --disable-asm-comments \
  library build/release/Tensile HIP
```

This regenerates `build/release/Tensile/library/gfx1151/` with your new logic baked in.
(Requires `msgpack` in system python — see §2.)

---

## 7. Benchmarking & Verification

Point the loader at your gfx1151 device library and bench the orientation you tuned.
**Use the transpose that matches the library you deployed** (see Orientation Map):

```bash
HB=~/TheRock/rocm-libraries/projects/hipblaslt
export HIPBLASLT_TENSILE_LIBPATH=$HB/build/release/Tensile/library/gfx1151

# warm the GPU first (clocks), then time:
$HB/build/release/clients/hipblaslt-bench --precision f32_r --compute_type s \
  --transA N --transB T -m 4096 -n 4096 -k 4096 --cold_iters 5 --iters 30 >/dev/null

$HB/build/release/clients/hipblaslt-bench --precision f32_r --compute_type s \
  --transA N --transB T -m 4096 -n 4096 -k 8192 \
  --cold_iters 20 --iters 80 --algo_method all --print_kernel_info
```

- `--print_kernel_info` shows the selected kernel name — confirm it is the
  `Cijk_Ailk_Bjlk...` (NT) family and a VOPD-style tile (TT up to 8×8), not a generic
  fallback kernel.

### 7.1 Generic-vs-tuned roundtrip harness — save as `roundtrip.sh`

The original verification A/B-tested generic kernels against tuned by *removing* the tuned
logic YAMLs, rebuilding, benching (generic), then *restoring* them, rebuilding, benching
(tuned). This version is adapted to **gfx1151** paths. It writes `ROUNDTRIP_RESULTS.md` +
6 CSVs. **Do not run while another tuning/bench job is using the GPU.**

```bash
#!/bin/bash
# save as: roundtrip.sh   (chmod +x roundtrip.sh)
# 3-way generic-vs-tuned roundtrip (NT, TN, NN) on gfx1151. Removes tuned logics -> rebuild
# -> bench generic; restore -> rebuild -> bench tuned. Writes ROUNDTRIP_RESULTS.md + 6 CSVs.
set -u
HB=~/TheRock/rocm-libraries/projects/hipblaslt
WORK=~/tune
GB=$HB/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx1151/GridBased
NT=$GB/gfx1151_Cijk_Ailk_Bjlk_S_B_UserArgs.yaml
TN=$GB/gfx1151_Cijk_Alik_Bljk_S_B_UserArgs.yaml
NN=$GB/gfx1151_Cijk_Ailk_Bljk_S_B_UserArgs.yaml   # NN shares the Ailk_Bljk base name
LIB=$HB/build/release/Tensile/library/gfx1151
BENCH=$HB/build/release/clients/hipblaslt-bench
SHAPES=$WORK/shapes_big.txt
OUT=$WORK
HOLD=$OUT/_held; mkdir -p "$HOLD"
log(){ echo "[$(date +%H:%M:%S)] $*"; }

rebuild(){ cd "$HB"; PYTHONPATH=build/release/tensilelite/rocisa:build/release/tensilelite:tensilelite \
  python3 -m Tensile.TensileCreateLibrary --architecture=gfx1151 \
  --cxx-compiler=/opt/rocm/bin/amdclang++ --disable-asm-comments \
  library build/release/Tensile HIP >"$OUT/_rebuild.log" 2>&1; log "rebuild rc=$? ($(tail -1 $OUT/_rebuild.log))"; }

bench(){ # $1=transA $2=transB $3=outcsv
  local ta=$1 tb=$2 out=$3
  export HIPBLASLT_TENSILE_LIBPATH=$LIB
  for p in $(ps aux|grep hipblaslt-bench|grep -v grep|awk '{print $2}'); do kill -9 $p 2>/dev/null; done
  $BENCH --precision f32_r --compute_type s --transA $ta --transB $tb -m 4096 -n 4096 -k 4096 --cold_iters 5 --iters 30 >/dev/null 2>&1
  echo "M,N,K,gflops,tile" > "$out"
  while IFS=, read -r M N K; do
    [ -z "$M" ] && continue
    local o gf mt best=0 bmt=""
    for r in 1 2 3; do
      o=$($BENCH --precision f32_r --compute_type s --transA $ta --transB $tb -m $M -n $N -k $K --cold_iters 20 --iters 80 --print_kernel_info 2>/dev/null)
      gf=$(echo "$o"|awk -F, '/^    [NT],/{print $(NF-2)}'|head -1)
      mt=$(echo "$o"|grep -m1 "kernel name"|grep -oE "MT[0-9]+x[0-9]+x[0-9]+")
      awk "BEGIN{exit !($gf>$best)}" && { best=$gf; bmt=$mt; }
    done
    echo "$M,$N,$K,$best,$bmt" >> "$out"
  done < "$SHAPES"
}

log "PHASE 1: remove tuned logics -> rebuild -> bench generic"
[ -f "$NT" ] && mv "$NT" "$HOLD/nt.yaml"
[ -f "$TN" ] && mv "$TN" "$HOLD/tn.yaml"
rebuild
bench N T "$OUT/rt_nt_generic.csv"; log "nt generic done"
bench T N "$OUT/rt_tn_generic.csv"; log "tn generic done"
bench N N "$OUT/rt_nn_generic.csv"; log "nn generic done"

log "PHASE 2: restore -> rebuild -> bench tuned"
[ -f "$HOLD/nt.yaml" ] && mv "$HOLD/nt.yaml" "$NT"
[ -f "$HOLD/tn.yaml" ] && mv "$HOLD/tn.yaml" "$TN"
rebuild
bench N T "$OUT/rt_nt_tuned.csv"; log "nt tuned done"
bench T N "$OUT/rt_tn_tuned.csv"; log "tn tuned done"
bench N N "$OUT/rt_nn_tuned.csv"; log "nn tuned done"
rm -rf "$HOLD"

log "writing ROUNDTRIP_RESULTS.md"
python3 - <<'PYEOF'
import csv,statistics,os
WORK=os.path.expanduser('~/tune')
def load(f):
    d={}
    if not os.path.exists(f): return d
    for r in csv.DictReader(open(f)):
        d[(r['M'],r['N'],r['K'])]=(float(r['gflops']), r.get('tile',''))
    return d
oris=[('NT','nt'),('TN','tn'),('NN','nn')]
lines=['# VOPD FP32 3-Way Roundtrip: generic vs tuned (gfx1151, cold-cache, best-of-3)','']
summ=[]
for name,key in oris:
    g=load(f'{WORK}/rt_{key}_generic.csv'); t=load(f'{WORK}/rt_{key}_tuned.csv')
    if not g or not t: continue
    lines+=[f'## {name}','', '| shape | generic GF | tuned GF | speedup | tuned tile |','|---|---:|---:|---:|---|']
    sps=[]
    for k in g:
        if k not in t: continue
        s=t[k][0]/g[k][0] if g[k][0] else 0; sps.append(s)
        lines.append(f"| {'x'.join(k)} | {g[k][0]:.0f} | {t[k][0]:.0f} | {s:.2f}x | {t[k][1]} |")
    gm=statistics.geometric_mean([x for x in sps if x>0]) if sps else 0
    pk=max(v[0] for v in t.values())/1000
    lines+=['',f'**{name}: geomean {gm:.2f}x, max {max(sps):.1f}x, peak {pk:.1f} TFLOPS**','']
    summ.append((name,gm,max(sps),pk))
lines=['# Summary','', '| orientation | geomean | max | peak TFLOPS |','|---|---:|---:|---:|']+[f'| {n} | {g:.2f}x | {m:.1f}x | {p:.1f} |' for n,g,m,p in summ]+[''] + lines
open(f'{WORK}/ROUNDTRIP_RESULTS.md','w').write('\n'.join(lines))
print('wrote ROUNDTRIP_RESULTS.md')
PYEOF
log "RESULTS_SAVED"
```

### 7.2 Fast heuristic bench harnesses — save as `bench_heur.sh` and `bench_stable.sh`

`bench_heur.sh` is a quick **heuristic-only** (what real users get, no `algo_method all`)
best-of-3 bench, GPU kept hot. `bench_stable.sh` additionally captures the
`--algo_method all` best per shape (the upper bound). Both are **NT-oriented**
(`--transA N --transB T`); change the transposes inside if you tuned a different
orientation. Lib path adapted to gfx1151.

```bash
#!/bin/bash
# save as: bench_heur.sh   (chmod +x bench_heur.sh)
# FAST heuristic-only FP32 NT bench (what real users get). best-of-3, GPU kept hot.
# Usage: bench_heur.sh <label> <shapes_file>
set -u
HB=~/TheRock/rocm-libraries/projects/hipblaslt
BENCH=$HB/build/release/clients/hipblaslt-bench
export HIPBLASLT_TENSILE_LIBPATH=$HB/build/release/Tensile/library/gfx1151
OUT=~/tune/heur_${1}.csv
SHAPES=${2:-~/tune/shapes_big.txt}
echo "M,N,K,heur_gflops" > "$OUT"
hot(){ $BENCH --precision f32_r --compute_type s --transA N --transB T -m 4096 -n 4096 -k 4096 --cold_iters 3 --iters 20 >/dev/null 2>&1; }
hot
while IFS=, read -r M N K; do
  [ -z "$M" ] && continue
  best=0
  for r in 1 2 3; do
    H=$($BENCH --precision f32_r --compute_type s --transA N --transB T \
          -m "$M" -n "$N" -k "$K" --cold_iters 30 --iters 100 2>/dev/null \
          | awk -F, '/^    [NT],/{print $(NF-2)}' | tail -1)
    awk "BEGIN{exit !($H>$best)}" && best=$H
  done
  echo "$M,$N,$K,$best" >> "$OUT"
  echo "  $M x $N x $K : heur=$best"
done < "$SHAPES"
echo "=> $OUT"
```

```bash
#!/bin/bash
# save as: bench_stable.sh   (chmod +x bench_stable.sh)
# Stable best-of-N FP32 NT bench. Keeps GPU hot, repeats each measurement, takes MAX
# (noise is downward from interference/clock-ramp, so max ~= true throughput).
# Usage: bench_stable.sh <label> <shapes_file>
set -u
HB=~/TheRock/rocm-libraries/projects/hipblaslt
BENCH=$HB/build/release/clients/hipblaslt-bench
export HIPBLASLT_TENSILE_LIBPATH=$HB/build/release/Tensile/library/gfx1151
OUT=~/tune/stable_${1}.csv
SHAPES=${2:-~/tune/shapes_big.txt}
REPEAT=3
echo "M,N,K,heur,best" > "$OUT"

hot() { # one big GEMM to keep clocks up
  $BENCH --precision f32_r --compute_type s --transA N --transB T -m 4096 -n 4096 -k 4096 \
    --cold_iters 5 --iters 30 >/dev/null 2>&1
}
hot; hot

while IFS=, read -r M N K; do
  [ -z "$M" ] && continue
  hbest=0; abest=0
  for r in $(seq $REPEAT); do
    hot
    H=$($BENCH --precision f32_r --compute_type s --transA N --transB T \
          -m "$M" -n "$N" -k "$K" --cold_iters 30 --iters 100 2>/dev/null \
          | awk -F, '/^    [NT],/{print $(NF-2)}' | tail -1)
    A=$($BENCH --precision f32_r --compute_type s --transA N --transB T \
          -m "$M" -n "$N" -k "$K" --cold_iters 10 --iters 50 --algo_method all 2>/dev/null \
          | awk -F, '/^    [NT],/{v=$(NF-2); if(v+0>m)m=v+0} END{print m}')
    awk "BEGIN{exit !($H>$hbest)}" && hbest=$H
    awk "BEGIN{exit !($A>$abest)}" && abest=$A
  done
  echo "$M,$N,$K,$hbest,$abest" >> "$OUT"
  echo "  $M x $N x $K : heur=$hbest best=$abest"
done < "$SHAPES"
echo "=> $OUT"
```

### 7.3 Verification shape list — save as `shapes_big.txt`

30 representative shapes (`M,N,K`, one per line) used by the roundtrip / bench scripts.

```text
1,4096,4096
1,8192,8192
512,512,512
1024,1024,1024
2048,2048,2048
4096,4096,4096
8192,8192,8192
32,4096,4096
64,4096,4096
128,4096,4096
256,4096,4096
512,4096,4096
1024,4096,4096
2048,4096,4096
4096,2048,2048
4096,8192,8192
256,8192,8192
512,8192,8192
1024,8192,8192
2048,8192,8192
1024,14336,4096
1024,4096,14336
128,14336,4096
3072,4096,4096
6144,4096,4096
2048,4096,8192
4096,4096,2048
512,2048,16384
2048,2048,8192
4096,1024,4096
```

---

## 8. Critical Gotchas (these cost us hours)

1. **Orientation / transpose mismatch.** The NT library `Cijk_Ailk_Bjlk` is exercised by
   `--transA N --transB T`, **NOT** TN. Benchmarking the wrong transpose makes VOPD look
   broken — the dispatcher falls to ~5 generic kernels and reports ~3-6 TFLOPS instead of
   30+. **Always** verify with `--print_kernel_info` that the kernel family matches the
   library you tuned.

2. **One bench process at a time.** Concurrent `hipblaslt-bench` on a single GPU corrupts
   small-shape timings. `pkill -9 hipblaslt-bench` (or kill stale PIDs) before each run.

3. **GEMV (M=1) shapes are memory-bound** and swing ~30% run-to-run. Use best-of-N
   (max of 3-5 runs), never a single sample. (The bench scripts above already do best-of-3.)

4. **GPU clocks.** Warm the GPU with a few large GEMMs before timing. Do **NOT** force
   `power_dpm_force_performance_level=high` — on RDNA3 it *hurt* throughput; leave it on
   `auto`. (Likely also true for the gfx1151 APU; leave clocks auto.)

5. **Tensile in-process numbers are cache-warm (optimistic).** The TFLOPS Tensile prints
   during a wave are higher than real cold-cache numbers. Trust `hipblaslt-bench` (cold
   iters) for the production figure. Expect Tensile-reported peak > hipblaslt-bench peak.

6. **`msgpack` for system python.** The standalone device-lib rebuild (`TensileCreate
   Library`) fails mid-way without it: `pip install --break-system-packages msgpack`
   (PyYAML too).

7. **rocisa must be rebuilt if the VOPD C++ changes.** For gfx1151 you build the whole
   hipBLASLt once (`invoke build -ca gfx1151`) and the VOPD C++ (in the branch) is compiled
   in — no separate step. Only if you *edit* the C++ in `rocisa/` do you need to rebuild
   it (`build/release/tensilelite/rocisa`).

8. **VOPD capability probe.** If the gfx1151 ROCm assembler does not accept
   `v_dual_fmac_f32`, `asmCaps["v_dual_fmac_f32"]` is false and **all VOPD solutions are
   silently rejected** (you'd only see them in rejection reasons). Confirm at the start of
   a wave: run a small smoke campaign with `PrintSolutionRejectionReason: true` and check no
   solution is rejected for missing `v_dual_fmac_f32`. The probe lives in
   `hardware_caps.hpp` ~line 297.

---

## 9. gfx1151 Adjustment Checklist

Concrete differences vs the gfx1100 reference — do all of these:

- [ ] **Arch string `gfx1100` → `gfx1151` everywhere:**
  - `Tensile --architecture=gfx1151` / `TensileCreateLibrary --architecture=gfx1151`
  - `create_library.yaml`: `ScheduleName: "gfx1151"` and `ArchitectureName: "gfx1151"`
  - Deploy dir: `.../Logic/asm_full/`**`gfx1151`**`/GridBased/gfx1151_Cijk_*_S_B_UserArgs.yaml`
  - Bench lib path: `build/release/Tensile/library/`**`gfx1151`**
  - (All embedded scripts above are already gfx1151.)
- [ ] **Build for gfx1151:** `invoke build -ca gfx1151 -d` (first run).
- [ ] **Confirm `v_dual_fmac_f32` is accepted** by your gfx1151 assembler (Gotcha §8).
- [ ] **Use the gfx1151 logic tree** (it already exists; the FP32 `GridBased/*_S_B_User
      Args.yaml` files do **not** yet — you create them). Do not write into `gfx1100/`.
- [ ] **Re-tune; do NOT copy gfx1100 logic.** gfx1151 is an APU (RDNA3.5) with fewer CUs,
      different clocks, and shared/lower memory bandwidth than the 96-CU 7900 XTX. Optimal
      tile sizes and DepthU will differ. Run your own waves (use the seeded config §11).
- [ ] **Use the seeded fork grid (§11) to go faster** — it's already pruned to the gfx1100
      winning cluster. Validate/re-derive on gfx1151 (§12) since the APU may prefer
      different tiles.
- [ ] **Tailor the shape set to your workloads** — extend `ProblemSizes` in
      `wave_template.yaml` / `seed.yaml` (§4.1).
- [ ] **Tune each orientation you need separately** (NN/NT/TN/TT — Orientation Map §5).

---

## 10. Seeded Fast-Tuning Strategy (recommended)

A blind full-grid sweep (3072 configs/shape × ~1030 shapes) is slow. On gfx1100 the winning
configs clustered tightly, so we built a **seeded narrow** config (§11) that prunes the fork
grid to that cluster (**~1728 vs ~3072 configs/shape**, ~44% fewer). Use this two-phase
approach on gfx1151:

**Phase 1 (optional, thorough — validate the cluster on gfx1151 hardware).**
Run ONE wave with the **seeded** fork grid (§11) on a representative shape subset (the ~18
shapes in `wave_template.yaml` are enough) to confirm the winning cluster holds on gfx1151.
gfx1151 is an **APU / RDNA3.5** with fewer CUs, different clocks, and shared/lower memory
bandwidth than the discrete 7900 XTX — the optimal tiles may differ, so don't assume the
gfx1100 cluster is final.

**Phase 2 (fast full pass).**
Build a gfx1151-specific `seed.yaml` from **gfx1151's own top configs** (use
`extract_cluster.py`, §12, on the Phase-1 wave's `2_BenchmarkData` CSV — it prints the top
16 configs). Drop those into the `ForkParameters` of your campaign, then run the full shape
set fast.

**Initial seed = our gfx1100 winning cluster** (already encoded in `seed.yaml` §11):
- MacroTile family: **128×128, 128×64, 64×64** (and smaller for skinny/GEMV)
- DepthU: **8–64** (DU8–16 favored large shapes)
- GlobalSplitU: **1** (GSU>1 rarely won)
- WorkGroupMapping: **4 or 8**
- ThreadTile: scales [1,1] → [8,8] with M·N (must be even for VOPD on the big tiles)
- VectorWidthA/B: **2** (VOPD default)
- PLR=0 favored large shapes.

> **Stress:** treat the gfx1100 cluster as a *starting hypothesis only*. gfx1151's different
> CU count, clocks, and memory bandwidth mean you should validate (Phase 1) and ideally
> re-derive (Phase 2) the cluster from gfx1151's own benchmark data.

---

## 11. SEEDED FAST-TUNING CONFIG — save as `seed.yaml`

This is the **key speed-up artifact**: the fork grid is pruned to the winning-config cluster
found on gfx1100 (MT128x128 / MT128x64 / MT64x64 family + skinny/GEMV tiles, DU8–64, GSU1,
WGM4/8, VW2). The fork grid is **~1728 configs/shape vs ~3072** in the full template (§4.1).
This is your **recommended starting point** — it's faster AND gfx1151-appropriate to re-tune
from. (Shown here for **NN**: `TransposeA:false, TransposeB:false`. For NT/TN/TT change ONLY
that line per the Orientation Map §5.)

The `ProblemSizes` block below is a representative subset; **extend it to your full shape set**
exactly as in §4.1. The original seeded NN campaign carried ~866 `Exact` entries.

```yaml
# save as: seed.yaml
# SEEDED NARROW grid (pruned to gfx1100 winning cluster). ~1728 configs/shape.
# NN orientation here; change TransposeA/TransposeB for other orientations (§5).
GlobalParameters: {MinimumRequiredVersion: 5.0.0, CMakeBuildType: Release, EnqueuesPerSync: 4, SyncsPerBenchmark: 4, NumElementsToValidate: 0,
  KernelTime: true, PrintWinnersOnly: false, PrintSolutionRejectionReason: true, NumBenchmarks: 1}
BenchmarkProblems:
- - {OperationType: GEMM, DataType: s, DestDataType: s, ComputeDataType: s, TransposeA: false, TransposeB: false, UseBeta: true,
    Batched: true}
  - InitialSolutionParameters: null
    BenchmarkCommonParameters:
    - KernelLanguage: [Assembly]
    - ScheduleIterAlg: [1]
    - ScheduleGlobalRead: [0]
    - ScheduleLocalWrite: [0]
    - EnableVOPD: [1]
    - VectorWidthA: [2]
    - VectorWidthB: [2]
    - PrefetchGlobalRead: [1]
    - PrefetchLocalRead: [1]
    ForkParameters:                # pruned to the winning cluster (1728 configs)
    - ThreadTile:                  # spans GEMV/skinny ([1,1]..[2,2]) up to big squares ([8,16],[16,8])
      - [8, 16]
      - [16, 8]
      - [8, 8]
      - [4, 8]
      - [8, 4]
      - [4, 4]
      - [2, 2]
      - [1, 2]
      - [1, 1]
    - WorkGroup:                   # only the 3 WGs that won on gfx1100
      - [16, 8, 1]
      - [8, 16, 1]
      - [8, 8, 1]
    - DepthU: [8, 16, 32, 64]
    - GlobalSplitU: [1]            # GSU>1 rarely won -> pinned to 1
    - WorkGroupMapping: [4, 8]
    - StaggerU: [0, 32]
    - LdsPadA: [0, 2]
    - LdsPadB: [0, 2]
    BenchmarkFinalParameters:
    - ProblemSizes:
      # representative subset — EXTEND to your full shape set (see §4.1).
      - Exact: [16, 16, 1, 16]
      - Exact: [64, 64, 1, 64]
      - Exact: [128, 128, 1, 128]
      - Exact: [256, 256, 1, 256]
      - Exact: [512, 512, 1, 512]
      - Exact: [1024, 1024, 1, 1024]
      - Exact: [2048, 2048, 1, 2048]
      - Exact: [4096, 4096, 1, 4096]
      - Exact: [8192, 8192, 1, 8192]
      - Exact: [4096, 4096, 1, 8192]
      - Exact: [6144, 8192, 1, 8192]
      - Exact: [16, 4096, 1, 4096]
      - Exact: [128, 8192, 1, 8192]
      - Exact: [768, 8192, 1, 8192]
      - Exact: [1, 4096, 1, 4096]
      - Exact: [1, 8192, 1, 8192]
      - Exact: [6144, 1, 1, 4096]
```

> **ThreadTile note:** the small odd tiles (`[1,1]`, `[1,2]`, `[2,2]`) are kept for
> GEMV/skinny shapes where VOPD doesn't apply (odd TT is rejected by the VOPD validator §3.1,
> so those shapes fall back to the scalar MAC). The even tiles ([4,4]…[16,8]) carry the
> VOPD wins. Tensile keeps whichever wins per shape; pruning the grid just removes the
> configs that never won on gfx1100.

Run it exactly like a wave:

```bash
HB=~/TheRock/rocm-libraries/projects/hipblaslt
$HB/tensilelite/Tensile/bin/Tensile  ~/tune/seed.yaml  ~/tune/seed_out
# then: build_merge.sh (§4.5) with the right KNAME, deploy (§6), verify (§7).
```

---

## 12. Build YOUR OWN seed from gfx1151 results — save as `extract_cluster.py`

After a first (full-ish or seeded) pass on gfx1151, extract the **winning-config cluster**
from the wave's `2_BenchmarkData` CSV so you can build a gfx1151-specific `seed.yaml`. Each
CSV row is one shape; columns 11+ are per-solution GFlops with the solution name in the
header. This picks the max per row, parses `MT{M}x{N}x{DepthU}`, `TT{m}_{n}`, `WGM{n}`,
`GSU{n}` from the winning column header, counts configs, and prints the top 16.

```python
#!/usr/bin/env python3
# save as: extract_cluster.py
# Usage: python3 extract_cluster.py <path-to>/2_BenchmarkData/Cijk_..._00.csv
# Prints the most-frequent winning configs (MacroTile / ThreadTile / WorkGroup / DepthU /
# GSU / WGM). Use the top entries to populate ForkParameters in your gfx1151 seed.yaml.
import sys, re, csv
from collections import Counter

path = sys.argv[1]
with open(path) as f:
    rows = list(csv.reader(f))
header = rows[0]
# First data column with a real solution name (skip GFlops, Size*, LD*, TotalFlops).
first = next(i for i, h in enumerate(header) if h.strip().startswith("Cijk_"))
sols = [h.strip() for h in header[first:]]

def parse(name):
    mt  = re.search(r'MT(\d+)x(\d+)x(\d+)', name)   # M x N x DepthU
    tt  = re.search(r'_TT(\d+)_(\d+)', name)
    wgm = re.search(r'_WGM(-?\d+)', name)
    gsu = re.search(r'_GSU(\d+)', name)
    if not (mt and tt): return None
    M, N, DU = mt.groups()
    t0, t1 = tt.groups()
    # WorkGroup = MacroTile / ThreadTile
    wg0 = int(M)//int(t0) if int(t0) else 0
    wg1 = int(N)//int(t1) if int(t1) else 0
    return (f"MT{M}x{N}", f"TT{t0}x{t1}", f"WG{wg0}x{wg1}", f"DU{DU}",
            f"GSU{gsu.group(1) if gsu else '?'}", f"WGM{wgm.group(1) if wgm else '?'}")

wins = Counter()
for r in rows[1:]:
    cells = r[first:]
    best_i, best_v = -1, -1.0
    for i, c in enumerate(cells):
        try: v = float(c)
        except ValueError: continue
        if v > best_v: best_v, best_i = v, i
    if best_i < 0: continue
    cfg = parse(sols[best_i])
    if cfg: wins[cfg] += 1

print(f"# shapes counted: {sum(wins.values())}   distinct winning configs: {len(wins)}")
print(f"# {'count':>5}  MacroTile  ThreadTile  WorkGroup  DepthU  GSU  WGM")
for cfg, n in wins.most_common(16):
    print(f"  {n:>5}  " + "  ".join(cfg))
```

Map the printed top configs back into `seed.yaml`'s `ForkParameters` (collect the distinct
`ThreadTile`, `WorkGroup`, `DepthU`, `GlobalSplitU`, `WorkGroupMapping` values that appear
in the top cluster). That gives you a gfx1151-derived narrow grid for the fast Phase-2 pass.

---

## 13. Reference Map (what each embedded file replaces)

| Original (gfx1100) external file | Embedded as | Section |
|----------------------------------|-------------|---------|
| `vopd_campaign/wave{1..4}.yaml` | `wave_template.yaml` (compact) | §4.1 |
| `tn_campaign/build_merge_tn.sh` | `build_merge.sh` (gfx1151, parameterized KNAME) | §4.5 |
| `nn_campaign/roundtrip3.sh` | `roundtrip.sh` (gfx1151 paths) | §7.1 |
| `regression_fix/bench_heur.sh` | `bench_heur.sh` (gfx1151 lib) | §7.2 |
| `regression_fix/bench_stable.sh` | `bench_stable.sh` (gfx1151 lib) | §7.2 |
| `regression_fix/shapes_big.txt` | `shapes_big.txt` | §7.3 |
| `nn_campaign/seed.yaml` | `seed.yaml` (the seeded narrow grid) | §11 |
| `create_library.yaml` snippet | `create_library.yaml` (gfx1151) | §4.3 |
| (new) cluster extractor | `extract_cluster.py` | §12 |

> Every former "see `<external file>`" reference in this doc has been replaced by an inline,
> copy-pasteable block above. Save them under `~/tune/` (or any work dir) and the workflow
> is fully reproducible from just the hipBLASLt clone.
