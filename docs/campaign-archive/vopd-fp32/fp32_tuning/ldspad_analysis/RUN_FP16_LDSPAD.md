# Producing an FP16 / BF16 LdsPad example on a GPU machine

## Why this doc exists

The SHAP + banded-heatmap analysis in this folder shows a strong LdsPad effect for
**FP32 (SGEMM)**. We could NOT produce the same for FP16/BF16 from data already on
this machine, because:

- Every **valid** FP16 (HHS) and BF16 (BBS) tuning result on disk **pins
  `LdsPadA = LdsPadB = 8`.** SHAP/heatmaps need the parameter to *vary* to show any
  effect — a constant column has zero importance by construction.
- The only `LdsPad=0` kernels that exist live in `output_failed_degenerate/` and
  `output_failed_rotbuf/` dirs (failed runs, no GFLOPs) — unusable.
- The winners-only sources (`retune/v2/verify_hhs.csv`, `retune/ship_final.csv`,
  the hipBLASLt grid logs in `rdna_gemm_bench/generated_bench_logs/`) don't carry a
  per-kernel LdsPad column at all.

So to get an FP16/BF16 LdsPad example you must **re-run a small Tensile tuning
sweep that forks LdsPadA/LdsPadB over several values**, on a machine with the
target GPU. This doc gives you a ready-to-run config and the exact commands to feed
the result back into the plotting scripts here.

## What you need on the GPU machine

- A working Tensile / hipBLASLt build for your GPU (e.g. gfx1201, gfx1100, gfx942).
  Use the `tensile-tuning` skill / your usual Tensile client. The command below
  assumes `Tensile/bin/Tensile` (or `tensilelite`) is on PATH.
- Python venv for the analysis (same as here):
  ```bash
  python3 -m venv venv && source venv/bin/activate
  pip install xgboost shap matplotlib scikit-learn pandas
  ```

## Step 1 — tuning config that FORKS LdsPad (FP16 / HHS)

This is based on a real HHS config from this repo
(`retune/_ref_gfx1201/agent_hhs_gfx1201/discrete/big/big_w7/config.yaml`). The ONLY
substantive change is forking `LdsPadA`/`LdsPadB` (were `[8]`) plus adding
`CSVExportWinner` so we get the full per-kernel matrix.

Save as `ldspad_hhs.yaml`:

```yaml
GlobalParameters: {MinimumRequiredVersion: 5.0.0, NumWarmups: 7, KernelTime: true,
  NumElementsToValidate: 0, CSVExportWinner: 1, CSVMergeSameProblemID: 1, Device: 0,
  PrintSolutionRejectionReason: true}
BenchmarkProblems:
- - OperationType: GEMM
    DataType: H            # H = fp16 in/out.  For BF16 use  DataType: B
    DestDataType: H        #                                  DestDataType: B
    ComputeDataType: S     # fp32 accumulate (HHS / BBS)
    HighPrecisionAccumulate: true
    TransposeA: true
    TransposeB: false
    UseBeta: true
    Batched: true
  - BenchmarkCommonParameters:
    - KernelLanguage: [Assembly]
    ForkParameters:
    - DepthU: [32, 64]
    - TransposeLDS: [1]
    - LdsPadA: [0, 2, 4, 8]      # <-- the sweep that makes LdsPad vary
    - LdsPadB: [0, 2, 4, 8]      # <-- (16 pad combos per tile)
    - ScheduleIterAlg: [3]
    - PrefetchGlobalRead: [2]
    - PrefetchLocalRead: [0]
    - WavefrontSize: [32]        # 32 for RDNA (gfx11/12); drop for CDNA
    - MIArchVgpr: [1]
    - Groups:
      - - MatrixInstruction: [16, 16, 16, 1, 1, 4, 4, 2, 2]
        - {DirectToLds: 0}
      - - MatrixInstruction: [16, 16, 16, 1, 1, 2, 2, 2, 2]
        - {DirectToLds: 0}
    BenchmarkFinalParameters:
    - ProblemSizes:
      - Exact: [4096, 4096, 1, 4096]
      - Exact: [8192, 8192, 1, 8192]
      - Exact: [3072, 8192, 1, 8192]
      - Exact: [1536, 8192, 1, 4096]
      - Exact: [8192, 2048, 1, 8192]
LibraryLogic:
  ScheduleName: gfx1201          # set to YOUR arch
  ArchitectureName: gfx1201      # set to YOUR arch
```

**Notes**
- `LdsPad=1` (odd pad) is the value that craters FP32 via LDS bank conflicts — if
  you want to reproduce the dramatic "cliff" band, add `1` to the lists:
  `LdsPadA: [0, 1, 2, 4, 8]`. Some archs reject odd pad for WMMA; that's fine, the
  rejected configs just won't appear.
- Keep the tile pool small (2 MIs × 2 DepthU × 16 pad combos ≈ 64 kernels) so the
  run is quick but LdsPad still varies richly.
- For **BF16**: change `DataType: B` and `DestDataType: B` (see comments above).
  Optionally also emit the classic HHS/BBS naming by matching your existing configs.

## Step 2 — run the tuning

```bash
Tensile ldspad_hhs.yaml ./ldspad_out
# (or your tensilelite / hipBLASLt tuning entrypoint pointed at the same yaml)
```

## Step 3 — collect the output CSV

Tensile writes a wide per-kernel CSV (rows = shapes, columns = kernels with
`_LPA#_LPB#` in each name, cells = GFLOPs). Grab either of:

```
ldspad_out/2_BenchmarkData/Cijk_*_00.csv                        # wide winner CSV
ldspad_out/1_BenchmarkProblems/Cijk_*_00/Data/00_Final.csv      # full per-kernel matrix (preferred)
```

Copy it back to this machine, e.g.:

```bash
scp gpubox:/path/to/ldspad_out/1_BenchmarkProblems/Cijk_*_00/Data/00_Final.csv \
    ~/vopd_sgemm/fp32_tuning/ldspad_analysis/fp16_ldspad_00.csv
```

## Step 4 — generate the SAME plots for FP16/BF16

The scripts here accept an arbitrary CSV. From this folder, with the venv active
(or use `~/rdna_gemm_bench/venv/bin/python`):

```bash
cd ~/vopd_sgemm/fp32_tuning/ldspad_analysis

# 1) SHAP importance (per-shape normalized) — LdsPad ranked among tuning params
python shap_ranked.py --glob 'fp16_ldspad_00.csv' --tag fp16
#   -> out/ranked_fp16_shap_bar.png, out/ranked_fp16_dep_LdsPadB.png

# 2) Kernel-name-sorted BANDED heatmap — LdsPad bands as green/red blocks
python banded_heatmap.py --csv fp16_ldspad_00.csv --by LdsPadB --tag fp16
python banded_heatmap.py --csv fp16_ldspad_00.csv --by LdsPadA --tag fp16
#   -> out/banded_fp16_by_LdsPadB.png, out/banded_fp16_collapsed_LdsPadB.png
```

If `shap_ranked.py --glob` finds the file but reports "LdsPad does not vary", your
sweep didn't actually fork the pad (check the ForkParameters lists) or all pad!=8
configs were rejected/failed on that arch.

## Sanity expectations

- The banded heatmap should show each `LdsPad=k` block as a distinct vertical band;
  a bad value (bank-conflict-prone) reads as a red/orange column, good values green.
- On the SHAP bar, LdsPadA/LdsPadB should appear in the mid-to-upper ranks once the
  per-shape normalization removes shape-size dominance (that's what happens for
  FP32 nn_campaign here: LdsPadB rank #1, ~30% share — see out/ranked_nn_campaign_shap_bar.png).

## Reference: what the FP32 result looks like (for comparison in your deck)
- `out/banded_ledger_collapsed_LdsPadA.png` — LdsPadA=1 column at 0.29 vs 0.78–0.84
- `out/ranked_nn_campaign_shap_bar.png` — LdsPadB ranked #1 after normalization
- `out/showcase_ledger_cliff.png` — 4096³ pad-only swing +425%
```
