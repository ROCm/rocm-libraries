# OOB BBS TN exact tuning, category 1 (gfx1250, BF16 in/out, FP32 compute, TN)

Three variants (`pure_LdaPadded/`, `fused_ldaPadded/`, `fused_ldaK/`), each in its own directory with a logic file and tuning configs. Both logic files hold only the 10 sizes of this
category, use the fused problem type (Bias + ScaleAlphaVec, ActivationFuncCall true on every solution), and key
ExactLogic on [M, N, 1, K] only (no strides).

## 2026-11-27 update: TailFolding retune (`pure_LdaPadded/` logic + tuning) and new `fused_ldaPadded/`

- **Requires** develop with TailFolding (PR #16312, merged as 0b95005e6af) **plus** the TDMStoreInst PR #15133 (head 4eb56fa3f5a, not merged yet).
  Checked on develop bec78032039 + #15133 4eb56fa3f5a. Under that PR head, TDMStoreInst 1 + AdaptiveGemm 1 is rejected for a pure GEMM, so the TDMStoreInst 1 winners use AdaptiveGemm 0.
- `fused_ldaK/` is unchanged.
- `pure_LdaPadded/logic`: the 10 winners of the 11/21–11/27 TailFolding retune. Fused problem type, ActivationFuncCall true, ExactLogic [M, N, 1, K], same treatment as before.
- `pure_LdaPadded/tuning`: the existing full-candidate file + the retune sets (all existing candidates kept, exact de-dup).
  - Sets: retune_1121 pass A/B/C, retune_tf1_1122, fixfold_1123, retune_1127 (WaveTileContiguousA × LDSSegmentInterleave, TPS None copies of persistent winners, TDMStoreInst 0 reference).
  - Group 1 = new winner, group 2 = previous (11/12) winner. The header lists the merged sets and the candidate count.
  - The retune trees' test knobs TailFolding / TailFoldingTestOff are removed (develop folds whenever legal).
  - Category 1 takes no StreamK candidates, as before.
- `fused_ldaPadded/` (new): the same winners built as fused GEMM kernels and called as the pure GEMM problem. See the section below.
- Checks (CPU, on that tree): TensileCreateLibrary --architecture=gfx1250 on the 4 logic files of each variant: **0 rejects, 0 codegen failures**.
  - Built kernel names = benched kernel names (except AFC).
  - Every tuning yaml parses and passes the Tensile parameter check; its group count matches the header.

### pure_LdaPadded results (Tensile client, GPU3, 2026-11-26, pure GEMM, LDA = LDB = ceil(K/128)*128, median of rounds 1+)

Category mean vs MAIDAS: **59.1%**.
The 11/12 us comes from the 11/12 table below (GPU3, older develop, different run), so it is a reference only, not a same-run gain.
The retune itself switched a winner only when it was >= 1% faster than the incumbent in the same client run on all 4 runs (fwd / rev × 2).

| size (MxNxK) | LDA=LDB | sol | MAIDAS us | 11/12 us | new us | vs 11/12 | new vs MAIDAS | winner | kernel |
|---|---|---|---|---|---|---|---|---|---|
| 96x224x240 | 256 | 0 | 1.70 | 2.88 | 2.51 | +14.6% | 67.6% | TF retune winner (fixfold 11/23) | MT32x16x256, MIWT1_1, MIWG2_1, CD4_4, TPS None, SS0, VWA1, VWB1, HPLR0, LSI0, TDMSI0, AG1, PGL0 |
| 256x224x144 | 256 | 1 | 1.70 | 3.37 | 2.67 | +26.4% | 63.6% | TF retune winner (fixfold 11/23) | MT32x16x128, MIWT1_1, MIWG2_1, CD4_1, TPS None, SS1, VWA1, VWB1, HPLR0, LSI0, TDMSI0, AG1, PGL0 |
| 832x896x16 | 128 | 2 | 1.75 | 3.84 | 3.63 | +5.8% | 48.2% | TF retune winner (fixfold 11/23) | MT16x128x32, MIWT1_4, MIWG1_2, CD4_1, TPS None, SS1, VWA1, VWB4, HPLR0, LSI0, TDMSI0, AG1, PGL0 |
| 688x768x912 | 1024 | 3 | 2.47 | 5.89 | 5.00 | +17.8% | 49.4% | TF retune winner (fixfold 11/23) | MT32x48x256, MIWT1_3, MIWG2_1, CD1_4, TPS None, SS1, VWA1, VWB1, HPLR0, LSI0, TDMSI0, AG1, PGL0 |
| 608x432x32 | 128 | 4 | 1.86 | 3.27 | 3.20 | +2.0% | 58.1% | TF retune winner (fixfold 11/23) | MT16x128x32, MIWT1_2, MIWG1_4, CD4_4, TPS None, SS1, VWA1, VWB2, HPLR0, LSI1, TDMSI0, AG1, PGL0 |
| 656x112x32 | 128 | 5 | 1.91 | 2.67 | 2.67 | -0.0% | 71.6% | TF retune winner (fixfold 11/23) | MT192x16x32, MIWT3_1, MIWG4_1, CD4_1, TPS None, SS1, VWA1, VWB1, HPLR0, LSI0, TDMSI0, AG1, PGL0 |
| 160x224x976 | 1024 | 6 | 1.99 | 4.54 | 3.12 | +45.3% | 63.8% | TF retune winner (fixfold 11/23) | MT32x16x512, MIWT1_1, MIWG2_1, CD4_1, TPS None, SS0, VWA1, VWB1, HPLR0, LSI0, TDMSI0, AG1, PGL0 |
| 720x432x352 | 384 | 7 | 1.98 | 4.51 | 3.70 | +21.9% | 53.6% | TF retune winner (fixfold 11/23) | MT64x16x256, MIWT2_1, MIWG2_1, CD4_1, TPS None, SS1, VWA1, VWB1, HPLR0, LSI0, TDMSI0, AG1, PGL0 |
| 576x208x736 | 768 | 8 | 2.05 | 4.59 | 3.83 | +19.7% | 53.6% | TF retune winner (fixfold 11/23) | MT32x32x512, MIWT1_1, MIWG2_2, CD4_2, TPS None, SS1, VWA1, VWB1, HPLR0, LSI1, TDMSI0, AG0, PGL0 |
| 224x464x608 | 640 | 9 | 2.06 | 4.05 | 3.37 | +20.1% | 61.1% | TF retune winner (fixfold 11/23) | MT64x16x512, MIWT2_1, MIWG2_1, CD4_2, TPS None, SS0, VWA2, VWB1, HPLR0, LSI0, TDMSI0, AG1, PGL0 |

- WaveTileContiguousA × LDSSegmentInterleave (11/27) did not win anywhere. With SourceSwap it needs VWA 1 and a native store width; best per size −0.2% .. −65%.

### fused_ldaPadded (new variant)

- Kernels: the `pure_LdaPadded` winners built as fused GEMM kernels: Bias [s, B] + ScaleAlphaVec + hipblaslt_all activation, ActivationFuncCall true.
  - CompactLoopStore 1, so the pure-gemm store pass (#15650) exists. TailFolding / TDMStoreInst as in the winner. No runtime pure bit.
  - Called as the **pure GEMM problem**: bias / ScaleAlphaVec null, activation none, beta 0. LDA = LDB padded to 256 B.
- `logic/`: ExactLogic [M, N, 1, K] for the 10 sizes, same file name and format as the other variants.
- `tuning/`: the `pure_LdaPadded` tuning file converted to the fused problem type.
  - Every candidate is kept, with ActivationFuncCall true + CompactLoopStore true (candidates that became identical are merged; the header gives the counts). Group 1 = fused winner.
  - NOEPI call: the problem type carries Bias + SAV, so after the Tensile build edit each ClientParameters.ini to `use-bias=0`, `use-scaleAlphaVec=0`, `activation-enum-args=None`, `init-beta=Zero`.
- Numbers: the 11/27 Excel column "20261127 Winner (g04-2 gpu0) - 256B alignment (Fused GEMM kernel + Pure GEMM problem)".
  - Tensile client, GPU0, NOEPI call, 4 runs (fwd / rev × 2), median of rounds 1+.
  - In those runs the fused kernel is −0.0 .. −0.7% vs the pure winner for most sizes of categories 3–4. Category 1 is −6 .. −22%, because without a runtime pure bit the store pass is picked in the epilogue.

Category mean vs MAIDAS: **50.0%**.

| size (MxNxK) | LDA=LDB | sol | MAIDAS us | us | vs MAIDAS | gflops | kernel |
|---|---|---|---|---|---|---|---|
| 96x224x240 | 256 | 0 | 1.70 | 3.110 | 54.6% | 3318 | retune winner + CompactLoopStore 1 |
| 256x224x144 | 256 | 1 | 1.70 | 3.346 | 50.7% | 4935 | retune winner + CompactLoopStore 1 |
| 832x896x16 | 128 | 2 | 1.75 | 4.209 | 41.6% | 5668 | retune winner + CompactLoopStore 1 |
| 688x768x912 | 1024 | 3 | 2.47 | 5.426 | 45.5% | 177631 | retune winner + CompactLoopStore 1 |
| 608x432x32 | 128 | 4 | 1.86 | 3.951 | 47.2% | 4255 | retune winner + CompactLoopStore 1 |
| 656x112x32 | 128 | 5 | 1.91 | 3.413 | 56.0% | 1378 | retune winner + CompactLoopStore 1 |
| 160x224x976 | 1024 | 6 | 1.99 | 3.637 | 54.8% | 19236 | retune winner + CompactLoopStore 1 |
| 720x432x352 | 384 | 7 | 1.98 | 4.175 | 47.5% | 52452 | retune winner + CompactLoopStore 1 |
| 576x208x736 | 768 | 8 | 2.05 | 4.056 | 50.6% | 43475 | retune winner + CompactLoopStore 1 |
| 224x464x608 | 640 | 9 | 2.06 | 3.980 | 51.8% | 31753 | retune winner + CompactLoopStore 1 |

### Pure GEMM, LDA/LDB padded (`pure_LdaPadded/`; 11/12 description, see the 2026-11-27 update for the current files)

- `pure_LdaPadded/logic/gfx1250_Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs.yaml`: ExactLogic for the 10 sizes below, 10 solutions.
  Winners tuned as pure GEMM (no bias / ScaleAlphaVec / activation) with LDA = LDB = ceil(K/128)*128.
- `pure_LdaPadded/tuning/M<m>_N<n>_K<k>.yaml`: one Tensile tuning config per size with the full candidate search for that size
  (all tuning rounds merged into one file, including a ClusterDim / ClusterAIDMapping / WorkGroupMapping sweep in which
  cluster candidates use ClusterAIDMapping -2; identical solutions removed; the header comment lists the merged sets
  and the candidate count). Group 1 = winner, group 2 = previous logic kernel (logic_1109a).
  Category 1 candidates contain no StreamK; where the previous logic kernel is StreamK it is included only as a
  comparison ruler in a separate second BenchmarkProblems section.
  Pure GEMM, `Exact: [M, N, 1, K, M, M, LDA, LDB]` with LDA = LDB = ceil(K/128)*128,
  EnqueuesPerSync 1000 / NumWarmups 300 / NumBenchmarks 5, RotatingBufferSize 384.

### Fused GEMM, LDA = LDB = K (`fused_ldaK/`)

- `fused_ldaK/logic/gfx1250_Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs.yaml`: ExactLogic for the 10 sizes below, 10 solutions.
  Winners tuned as fused GEMM (Bias (f32) + ScaleAlphaVec, activation none) with unpadded leading dimensions.
- `fused_ldaK/tuning/M<m>_N<n>_K<k>.yaml`: one Tensile tuning config per size with the full candidate search of the
  fused retune for that size (all candidate sets merged into one file, including a ClusterDim sweep; identical
  solutions removed; the header comment gives the candidate count). Group 1 = winner of the pure padded logic
  (previous winner), group 2 = previous logic kernel (logic_1109a).
  Fused GEMM, Bias (f32) + ScaleAlphaVec, activation none, ActivationFuncCall true,
  `Exact: [M, N, 1, K, M, M, K, K]` (LDA = LDB = K, LDC = LDD = M),
  EnqueuesPerSync 1000 / NumWarmups 300 / NumBenchmarks 5, RotatingBufferSize 384.
  Category 1 candidates contain no StreamK; where the previous logic kernel is StreamK it is included only as a
  comparison ruler in a separate second BenchmarkProblems section.

## Results, pure GEMM, LDA/LDB padded (11/12 logic, superseded by the 2026-11-27 update above; Tensile client, GPU3, median of rounds 1+)

Category mean: gain vs previous **+1.4%**, vs MAIDAS **50.2% -> 50.9%**
(logic_1109a: 42.6%; 6 of 10 sizes have a new winner; vs MAIDAS = MAIDAS us / measured us).
Previous = winner of the previous pure padded logic of this directory; logic_1109a = previous logic kernel. All three
kernels of a size are measured in the same client run; if the previous winner measured faster it is kept.

| size (MxNxK) | LDA=LDB | sol | MAIDAS us | logic_1109a us | previous us | new us | gain vs previous | previous vs MAIDAS | new vs MAIDAS | winner |
|---|---|---|---|---|---|---|---|---|---|---|
| 96x224x240 | 256 | 0 | 1.70 | 3.44 | 2.88 | 2.88 | +0.0% | 58.9% | 58.9% | previous kept |
| 160x224x976 | 1024 | 6 | 1.99 | 4.70 | 4.54 | 4.54 | +0.0% | 43.8% | 43.8% | same as previous |
| 224x464x608 | 640 | 9 | 2.06 | 5.02 | 4.05 | 4.05 | +0.0% | 50.9% | 50.9% | same as previous |
| 256x224x144 | 256 | 1 | 1.70 | 4.41 | 3.38 | 3.37 | +0.1% | 50.2% | 50.3% | new |
| 576x208x736 | 768 | 8 | 2.05 | 4.77 | 4.59 | 4.59 | +0.0% | 44.7% | 44.7% | same as previous |
| 608x432x32 | 128 | 4 | 1.86 | 3.94 | 3.38 | 3.27 | +3.4% | 55.1% | 57.0% | new |
| 656x112x32 | 128 | 5 | 1.91 | 3.20 | 2.68 | 2.67 | +0.6% | 71.4% | 71.8% | new |
| 688x768x912 | 1024 | 3 | 2.47 | 6.81 | 6.04 | 5.89 | +2.5% | 40.9% | 41.9% | new |
| 720x432x352 | 384 | 7 | 1.98 | 5.27 | 4.52 | 4.51 | +0.3% | 43.9% | 44.0% | new |
| 832x896x16 | 128 | 2 | 1.75 | 5.63 | 4.12 | 3.84 | +7.2% | 42.5% | 45.6% | new |

96x224x240 keeps the previous winner: its new candidate measured slower than the previous winner in a separate re-measurement.

## Results, fused GEMM, LDA = LDB = K (`fused_ldaK/`; Tensile client, GPU0 and GPU3 (column), median of rounds 1+)

Category mean: gain vs previous **+2.5%**, vs MAIDAS **40.7% -> 41.6%**
(3 of 10 sizes have a new winner; the others keep the previous winner).
Fused GEMM: Bias (f32) + ScaleAlphaVec, activation none, LDA = LDB = K, LDC = LDD = M. Previous = winner of the
earlier pure padded logic, measured in the same client run as the new winner; a size keeps the previous winner
when it measured faster. Rows marked GPU0 come from one bench of all sizes on GPU0; rows marked GPU3 are the sizes
whose winner was re-selected afterwards, re-measured on GPU3 (compare gains, not absolute us, across GPUs).

| size (MxNxK) | sol | GPU | MAIDAS us | previous us | new us | gain vs previous | previous vs MAIDAS | new vs MAIDAS | winner |
|---|---|---|---|---|---|---|---|---|---|
| 96x224x240 | 0 | GPU0 | 1.70 | 4.10 | 3.90 | +5.2% | 41.4% | 43.6% | new |
| 160x224x976 | 6 | GPU0 | 1.99 | 5.45 | 5.45 | +0.0% | 36.5% | 36.5% | kept |
| 224x464x608 | 9 | GPU0 | 2.06 | 4.95 | 4.95 | +0.0% | 41.6% | 41.6% | kept |
| 256x224x144 | 1 | GPU0 | 1.70 | 3.99 | 3.99 | +0.0% | 42.5% | 42.5% | kept |
| 576x208x736 | 8 | GPU3 | 2.05 | 5.45 | 5.45 | +0.0% | 37.7% | 37.7% | kept |
| 608x432x32 | 4 | GPU0 | 1.86 | 3.93 | 3.93 | +0.0% | 47.4% | 47.4% | kept |
| 656x112x32 | 5 | GPU0 | 1.91 | 3.98 | 3.68 | +8.2% | 48.0% | 52.0% | new |
| 688x768x912 | 3 | GPU0 | 2.47 | 7.68 | 6.90 | +11.3% | 32.1% | 35.8% | new |
| 720x432x352 | 7 | GPU0 | 1.98 | 4.95 | 4.95 | +0.0% | 40.0% | 40.0% | kept |
| 832x896x16 | 2 | GPU3 | 1.75 | 4.47 | 4.47 | +0.0% | 39.2% | 39.2% | kept |

## Winner key parameters, pure GEMM, LDA/LDB padded (11/12 logic, superseded)

| size | sol | kernel | winner |
|---|---|---|---|
| 96x224x240 | 0 | MT32x16x256, MIWT1_1, WG32_2_1, CD4_4, CAIDMn2, TPS None, SS1, VWA1, VWB1, SVW1, HPLR0, LDSSI0, TDMIM0, LPA8, LBSPPA512, PLR1, CLS0, AG1, PGL0, PAP0, WGM4 | previous kept |
| 160x224x976 | 6 | MT16x16x512, MIWT1_1, WG16_2_1, CD4_2, TPS None, SS0, VWA1, VWB1, SVW8, HPLR0, LDSSI0, TDMIM0, LPA8, LBSPPA1024, PLR1, CLS0, AG1, PGL0, PAP0, WGM8 | same as previous |
| 224x464x608 | 9 | MT64x16x256, MIWT2_1, WG32_2_1, CD4_2, TPS None, SS0, VWA2, VWB1, SVW8, HPLR0, LDSSI0, TDMIM0, LPA8, LBSPPA1024, PLR1, CLS0, AG1, PGL0, PAP0, WGM8 | same as previous |
| 256x224x144 | 1 | MT32x16x128, MIWT2_1, WG16_2_1, CD4_2, TPS None, SS1, VWA2, VWB1, SVW2, HPLR0, LDSSI0, TDMIM0, LPA8, LBSPPA512, PLR3, CLS0, AG1, PGL0, PAP0, WGM8 | new |
| 576x208x736 | 8 | MT32x32x256, MIWT1_1, WG32_4_1, CD4_2, TPS None, SS1, VWA1, VWB1, SVW1, HPLR0, LDSSI1, TDMIM0, LPA8, LBSPPA512, PLR3, CLS0, AG0, PGL0, PAP0, WGM8 | same as previous |
| 608x432x32 | 4 | MT16x128x32, MIWT1_2, WG16_8_1, CD4_4, TPS None, SS1, VWA1, VWB2, SVW1, HPLR0, LDSSI1, TDMIM0, LPA8, LBSPPA256, PLR0, CLS0, AG1, PGL0, PAP0, WGM8 | new |
| 656x112x32 | 5 | MT192x16x32, MIWT3_1, WG64_2_1, CD4_2, TPS None, SS1, VWA1, VWB1, SVW1, HPLR0, LDSSI0, TDMIM0, LPA8, LBSPPA256, PLR0, CLS0, AG1, PGL0, PAP0, WGM8 | new |
| 688x768x912 | 3 | MT32x48x256, MIWT1_3, WG32_2_1, CD2_2, CAIDMn2, TPS None, SS1, VWA1, VWB1, SVW1, HPLR0, LDSSI0, TDMIM0, LPA8, LBSPPA512, PLR1, CLS0, AG1, PGL0, PAP0, WGM1 | new |
| 720x432x352 | 7 | MT32x16x128, MIWT1_1, WG32_2_1, CD4_2, TPS None, SS1, VWA1, VWB1, SVW1, HPLR0, LDSSI0, TDMIM0, LPA8, LBSPPA256, PLR3, CLS0, AG1, PGL0, PAP0, WGM8 | new |
| 832x896x16 | 2 | MT16x128x32, MIWT1_4, WG16_4_1, CD4_2, TPS None, SS1, VWA1, VWB4, SVW1, HPLR0, LDSSI0, TDMIM0, LPA8, LBSPPA256, PLR0, CLS0, AG1, PGL0, PAP0, WGM8 | new |

## Winner key parameters, fused GEMM, LDA = LDB = K (`fused_ldaK/`)

| size | sol | kernel | winner |
|---|---|---|---|
| 96x224x240 | 0 | MT16x16x256, MIWT1_1, WG16_2_4, CD2_2, TPS None, SS1, VWA1, VWB1, SVW1, HPLR0, LDSSI0, TDMIM0, LPA8, LBSPPA512, PLR1, CLS0, AG0, PGL0, PAP0, WGM4 | new |
| 160x224x976 | 6 | MT16x16x512, MIWT1_1, WG16_2_1, CD4_2, TPS None, SS0, VWA1, VWB1, SVW8, HPLR0, LDSSI0, TDMIM0, LPA8, LBSPPA1024, PLR1, CLS0, AG1, PGL0, PAP0, WGM8 | kept |
| 224x464x608 | 9 | MT64x16x256, MIWT2_1, WG32_2_1, CD4_2, TPS None, SS0, VWA2, VWB1, SVW8, HPLR0, LDSSI0, TDMIM0, LPA8, LBSPPA1024, PLR1, CLS0, AG1, PGL0, PAP0, WGM8 | kept |
| 256x224x144 | 1 | MT32x16x128, MIWT2_1, WG16_2_1, CD4_2, TPS None, SS0, VWA2, VWB1, SVW8, HPLR0, LDSSI0, TDMIM0, LPA8, LBSPPA512, PLR1, CLS0, AG0, PGL0, PAP0, WGM4 | kept |
| 576x208x736 | 8 | MT32x32x256, MIWT1_1, WG32_4_1, CD4_2, TPS None, SS1, VWA1, VWB1, SVW1, HPLR0, LDSSI1, TDMIM0, LPA8, LBSPPA512, PLR3, CLS0, AG0, PGL0, PAP0, WGM8 | kept |
| 608x432x32 | 4 | MT32x16x32, MIWT1_1, WG32_2_1, CD2_2, TPS None, SS1, VWA1, VWB1, SVW1, HPLR0, LDSSI0, TDMIM0, LPA8, LBSPPA256, PLR0, CLS0, AG0, PGL0, PAP0, WGM8 | kept |
| 656x112x32 | 5 | MT32x16x32, MIWT2_1, WG16_2_1, CD1_4, TPS None, SS1, VWA2, VWB1, SVW2, HPLR0, LDSSI0, TDMIM0, LPA8, LBSPPA256, PLR0, CLS0, AG1, PGL0, PAP0, WGM4 | new |
| 688x768x912 | 3 | MT32x48x256, MIWT1_3, WG32_2_2, CD1_1, TPS None, SS1, VWA1, VWB1, SVW1, HPLR0, LDSSI0, TDMIM0, LPA8, LBSPPA512, PLR1, CLS0, AG0, PGL0, PAP0, WGM8 | new |
| 720x432x352 | 7 | MT32x16x128, MIWT1_1, WG32_2_1, CD4_2, TPS None, SS1, VWA1, VWB1, SVW1, HPLR0, LDSSI0, TDMIM0, LPA8, LBSPPA256, PLR2, CLS0, AG1, PGL0, PAP0, WGM8 | kept |
| 832x896x16 | 2 | MT16x64x32, MIWT1_4, WG16_2_1, CD2_2, TPS None, SS1, VWA1, VWB4, SVW1, HPLR0, LDSSI0, TDMIM0, LPA8, LBSPPA256, PLR0, CLS0, AG1, PGL0, PAP0, WGM8 | kept |

## Requirements / notes

- gfx1250 (`Device 73f0`), Tensile with TDM (TDMInst 3), TDMIterateMode, LDSSegmentInterleave, HalfPLR, ClusterDim and
  AdaptiveGemm support. TensileCreateLibrary --architecture=gfx1250 on each logic file: 0 rejects.
- The two tables differ in LDA alignment (with LDA = K, K*2 bytes not a multiple of 128 makes TDM loads go
  indirect) and in the fused epilogue, so the two variants can pick different kernels for the same size.
- The tables come from different GPUs and runs (pure: GPU3; fused: GPU0, plus GPU3 rows where marked); compare gains
  within a row, not absolute us across tables.
