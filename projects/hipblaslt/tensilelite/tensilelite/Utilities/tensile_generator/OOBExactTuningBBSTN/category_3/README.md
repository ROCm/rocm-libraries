# OOB BBS TN exact tuning, category 3 (gfx1250, BF16 in/out, FP32 compute, TN)

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

Category mean vs MAIDAS: **86.8%**.
The 11/12 us comes from the 11/12 table below (GPU3, older develop, different run), so it is a reference only, not a same-run gain.
The retune itself switched a winner only when it was >= 1% faster than the incumbent in the same client run on all 4 runs (fwd / rev × 2).

| size (MxNxK) | LDA=LDB | sol | MAIDAS us | 11/12 us | new us | vs 11/12 | new vs MAIDAS | winner | kernel |
|---|---|---|---|---|---|---|---|---|---|
| 4928x4848x7216 | 7296 | 0 | 105.99 | 133.13 | 131.03 | +1.6% | 80.9% | TF retune winner (fixfold 11/23) | MT192x256x128, MIWT12_4, MIWG1_4, CD1_1, TPS None, SS1, VWA4, VWB4, HPLR1, LSI1, TDMSI1, AG0, PGL0 |
| 7040x6912x6256 | 6272 | 1 | 184.96 | 214.93 | 214.17 | +0.4% | 86.4% | TF retune winner (fixfold 11/23) | MT256x256x128, MIWT8_8, MIWG2_2, CD2_1, TPS None, SS1, VWA8, VWB4, HPLR2, LSI1, TDMSI1, AG0, PGL0 |
| 6496x7056x5664 | 5760 | 2 | 160.19 | 203.07 | 197.14 | +3.0% | 81.3% | TF retune winner (fixfold 11/23) | MT384x160x128, MIWT6_10, MIWG4_1, CD1_1, TPS None, SS1, VWA2, VWB2, HPLR2, LSI0, TDMSI1, AG0, PGL0 |
| 6000x4928x7216 | 7296 | 3 | 133.61 | 150.60 | 146.41 | +2.9% | 91.3% | TF retune winner (fixfold 11/23) | MT192x320x128, MIWT12_5, MIWG1_4, CD4_4, TPS DataParallel, SS1, VWA4, VWB1, HPLR1, LSI0, TDMSI0, AG1, PGL2 |
| 7008x5808x6288 | 6400 | 4 | 156.05 | 189.06 | 181.93 | +3.9% | 85.8% | TF retune winner (fixfold 11/23) | MT320x256x128, MIWT10_8, MIWG2_2, CD2_1, TPS None, SS1, VWA2, VWB4, HPLR2, LSI0, TDMSI1, AG0, PGL0 |
| 7808x7088x5424 | 5504 | 5 | 186.47 | 205.98 | 196.91 | +4.6% | 94.7% | TF retune winner (fixfold 11/23) | MT256x320x128, MIWT8_10, MIWG2_2, CD4_4, TPS None, SS1, VWA8, VWB2, HPLR1, LSI1, TDMSI0, AG1, PGL2 |
| 4464x5968x6464 | 6528 | 6 | 107.79 | 131.79 | 123.88 | +6.4% | 87.0% | TF retune winner (fixfold 11/23) | MT224x256x128, MIWT14_4, MIWG1_4, CD2_2, TPS None, SS1, VWA2, VWB4, HPLR1, LSI1, TDMSI1, AG0, PGL2 |
| 7856x5120x7200 | 7296 | 7 | 175.45 | 189.09 | 192.29 | -1.7% | 91.2% | TF retune winner (fixfold 11/23) | MT256x320x128, MIWT8_10, MIWG2_2, CD4_4, TPS None, SS1, VWA8, VWB2, HPLR1, LSI1, TDMSI0, AG1, PGL2 |
| 7200x6304x5888 | 5888 | 8 | 161.62 | 203.58 | 201.07 | +1.2% | 80.4% | TF retune winner (fixfold 11/23) | MT240x256x128, MIWT15_4, MIWG1_4, CD2_1, TPS None, SS1, VWA1, VWB4, HPLR2, LSI1, TDMSI1, AG0, PGL0 |
| 5504x6096x5440 | 5504 | 9 | 111.83 | 129.62 | 125.30 | +3.4% | 89.2% | TF retune winner (fixfold 11/23) | MT192x384x128, MIWT12_6, MIWG1_4, CD4_4, TPS None, SS1, VWA4, VWB2, HPLR1, LSI0, TDMSI1, AG0, PGL2 |

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

Category mean vs MAIDAS: **84.5%**.

| size (MxNxK) | LDA=LDB | sol | MAIDAS us | us | vs MAIDAS | gflops | kernel |
|---|---|---|---|---|---|---|---|
| 4928x4848x7216 | 7296 | 0 | 105.99 | 133.718 | 79.3% | 2578522 | retune winner + CompactLoopStore 1 |
| 7040x6912x6256 | 6272 | 1 | 184.96 | 216.070 | 85.6% | 2817793 | retune winner + CompactLoopStore 1 |
| 6496x7056x5664 | 5760 | 2 | 160.19 | 201.018 | 79.7% | 2582991 | retune winner + CompactLoopStore 1 |
| 6000x4928x7216 | 7296 | 3 | 133.61 | 157.506 | 84.8% | 2709269 | retune winner + CompactLoopStore 1 |
| 7008x5808x6288 | 6400 | 4 | 156.05 | 184.288 | 84.7% | 2777577 | retune winner + CompactLoopStore 1 |
| 7808x7088x5424 | 5504 | 5 | 186.47 | 209.434 | 89.0% | 2866600 | retune winner + CompactLoopStore 1 |
| 4464x5968x6464 | 6528 | 6 | 107.79 | 128.034 | 84.2% | 2690047 | retune winner + CompactLoopStore 1 |
| 7856x5120x7200 | 7296 | 7 | 175.45 | 193.767 | 90.5% | 2989186 | retune winner + CompactLoopStore 1 |
| 7200x6304x5888 | 5888 | 8 | 161.62 | 201.467 | 80.2% | 2653036 | retune winner + CompactLoopStore 1 |
| 5504x6096x5440 | 5504 | 9 | 111.83 | 128.489 | 87.0% | 2841099 | retune winner + CompactLoopStore 1 |

### Pure GEMM, LDA/LDB padded (`pure_LdaPadded/`; 11/12 description, see the 2026-11-27 update for the current files)

- `pure_LdaPadded/logic/gfx1250_Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs.yaml`: ExactLogic for the 10 sizes below, 10 solutions.
  Winners tuned as pure GEMM (no bias / ScaleAlphaVec / activation) with LDA = LDB = ceil(K/128)*128.
- `pure_LdaPadded/tuning/M<m>_N<n>_K<k>.yaml`: one Tensile tuning config per size with the full candidate search for that size
  (all tuning rounds merged into one file, including a ClusterDim / ClusterAIDMapping / WorkGroupMapping sweep in which
  cluster candidates use ClusterAIDMapping -2; identical solutions removed; the header comment lists the merged sets
  and the candidate count). Group 1 = winner, group 2 = previous logic kernel (logic_1109a).
  Pure GEMM, `Exact: [M, N, 1, K, M, M, LDA, LDB]` with LDA = LDB = ceil(K/128)*128,
  EnqueuesPerSync 400 / NumWarmups 160 / NumBenchmarks 5, RotatingBufferSize 384.

### Fused GEMM, LDA = LDB = K (`fused_ldaK/`)

- `fused_ldaK/logic/gfx1250_Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs.yaml`: ExactLogic for the 10 sizes below, 10 solutions.
  Winners tuned as fused GEMM (Bias (f32) + ScaleAlphaVec, activation none) with unpadded leading dimensions.
- `fused_ldaK/tuning/M<m>_N<n>_K<k>.yaml`: one Tensile tuning config per size with the full candidate search of the
  fused retune for that size (all candidate sets merged into one file, including a ClusterDim sweep; identical
  solutions removed; the header comment gives the candidate count). Group 1 = winner of the pure padded logic
  (previous winner), group 2 = previous logic kernel (logic_1109a).
  Fused GEMM, Bias (f32) + ScaleAlphaVec, activation none, ActivationFuncCall true,
  `Exact: [M, N, 1, K, M, M, K, K]` (LDA = LDB = K, LDC = LDD = M),
  EnqueuesPerSync 400 / NumWarmups 160 / NumBenchmarks 3, RotatingBufferSize 384.

## Results, pure GEMM, LDA/LDB padded (11/12 logic, superseded by the 2026-11-27 update above; Tensile client, GPU3, median of rounds 1+)

Category mean: gain vs previous **+1.5%**, vs MAIDAS **83.4% -> 84.7%**
(logic_1109a: 76.8%; 7 of 10 sizes have a new winner; vs MAIDAS = MAIDAS us / measured us).
Previous = winner of the previous pure padded logic of this directory; logic_1109a = previous logic kernel. All three
kernels of a size are measured in the same client run; if the previous winner measured faster it is kept.

| size (MxNxK) | LDA=LDB | sol | MAIDAS us | logic_1109a us | previous us | new us | gain vs previous | previous vs MAIDAS | new vs MAIDAS | winner |
|---|---|---|---|---|---|---|---|---|---|---|
| 4464x5968x6464 | 6528 | 6 | 107.79 | 144.04 | 131.79 | 131.79 | +0.0% | 81.8% | 81.8% | previous kept |
| 4928x4848x7216 | 7296 | 0 | 105.99 | 144.75 | 134.28 | 133.13 | +0.9% | 78.9% | 79.6% | new |
| 5504x6096x5440 | 5504 | 9 | 111.83 | 134.35 | 129.71 | 129.62 | +0.1% | 86.2% | 86.3% | new |
| 6000x4928x7216 | 7296 | 3 | 133.61 | 173.76 | 162.93 | 150.60 | +8.2% | 82.0% | 88.7% | new |
| 6496x7056x5664 | 5760 | 2 | 160.19 | 225.71 | 205.01 | 203.07 | +1.0% | 78.1% | 78.9% | new |
| 7008x5808x6288 | 6400 | 4 | 156.05 | 197.10 | 189.06 | 189.06 | +0.0% | 82.5% | 82.5% | previous kept |
| 7040x6912x6256 | 6272 | 1 | 184.96 | 260.22 | 215.00 | 214.93 | +0.0% | 86.0% | 86.1% | new |
| 7200x6304x5888 | 5888 | 8 | 161.62 | 210.84 | 203.58 | 203.58 | +0.0% | 79.4% | 79.4% | previous kept |
| 7808x7088x5424 | 5504 | 5 | 186.47 | 227.91 | 214.87 | 205.98 | +4.3% | 86.8% | 90.5% | new |
| 7856x5120x7200 | 7296 | 7 | 175.45 | 218.48 | 189.69 | 189.09 | +0.3% | 92.5% | 92.8% | new |

## Results, fused GEMM, LDA = LDB = K (`fused_ldaK/`; Tensile client, GPU0 and GPU3 (column), median of rounds 1+)

Category mean: gain vs previous **+3.4%**, vs MAIDAS **66.0% -> 68.3%**
(9 of 10 sizes have a new winner; the others keep the previous winner).
Fused GEMM: Bias (f32) + ScaleAlphaVec, activation none, LDA = LDB = K, LDC = LDD = M. Previous = winner of the
earlier pure padded logic, measured in the same client run as the new winner; a size keeps the previous winner
when it measured faster. Rows marked GPU0 come from one bench of all sizes on GPU0; rows marked GPU3 are the sizes
whose winner was re-selected afterwards, re-measured on GPU3 (compare gains, not absolute us, across GPUs).

| size (MxNxK) | sol | GPU | MAIDAS us | previous us | new us | gain vs previous | previous vs MAIDAS | new vs MAIDAS | winner |
|---|---|---|---|---|---|---|---|---|---|
| 4464x5968x6464 | 6 | GPU0 | 107.79 | 155.92 | 146.95 | +6.1% | 69.1% | 73.4% | new |
| 4928x4848x7216 | 0 | GPU0 | 105.99 | 186.69 | 186.69 | +0.0% | 56.8% | 56.8% | kept |
| 5504x6096x5440 | 9 | GPU3 | 111.83 | 137.25 | 137.24 | +0.0% | 81.5% | 81.5% | new |
| 6000x4928x7216 | 3 | GPU3 | 133.61 | 216.40 | 215.03 | +0.6% | 61.7% | 62.1% | new |
| 6496x7056x5664 | 2 | GPU0 | 160.19 | 282.13 | 275.23 | +2.5% | 56.8% | 58.2% | new |
| 7008x5808x6288 | 4 | GPU0 | 156.05 | 241.94 | 231.95 | +4.3% | 64.5% | 67.3% | new |
| 7040x6912x6256 | 1 | GPU0 | 184.96 | 308.09 | 300.02 | +2.7% | 60.0% | 61.6% | new |
| 7200x6304x5888 | 8 | GPU0 | 161.62 | 232.90 | 216.01 | +7.8% | 69.4% | 74.8% | new |
| 7808x7088x5424 | 5 | GPU0 | 186.47 | 278.18 | 263.70 | +5.5% | 67.0% | 70.7% | new |
| 7856x5120x7200 | 7 | GPU0 | 175.45 | 239.26 | 228.15 | +4.9% | 73.3% | 76.9% | new |

## Winner key parameters, pure GEMM, LDA/LDB padded (11/12 logic, superseded)

| size | sol | kernel | winner |
|---|---|---|---|
| 4464x5968x6464 | 6 | MT224x256x128, MIWT14_4, WG16_8_1, CD2_2, CAIDMn2, TPS None, SS1, VWA2, VWB4, SVW2, HPLR2, LDSSI1, TDMIM0, LPA8, LBSPPA512, PLR1, CLS1, AG1, PGL0, PAP0, WGM8 | previous kept |
| 4928x4848x7216 | 0 | MT192x256x128, MIWT12_4, WG16_8_1, CD1_1, TPS None, SS1, VWA4, VWB4, SVW4, HPLR0, LDSSI1, TDMIM0, LPA8, LBSPPA1024, PLR1, CLS1, AG1, PGL0, PAP0, WGM16 | new |
| 5504x6096x5440 | 9 | MT192x384x128, MIWT12_6, WG16_8_1, CD4_4, TPS None, SS1, VWA4, VWB2, SVW4, HPLR1, LDSSI0, TDMIM0, LPA8, LBSPPA1024, PLR1, CLS1, AG1, PGL2, PAP0, WGM8 | new |
| 6000x4928x7216 | 3 | MT192x320x128, MIWT12_5, WG16_8_1, CD4_4, TPS None, SS1, VWA4, VWB1, SVW4, HPLR1, LDSSI0, TDMIM0, LPA8, LBSPPA1024, PLR1, CLS1, AG1, PGL0, PAP0, WGM8 | new |
| 6496x7056x5664 | 2 | MT384x160x128, MIWT6_10, WG64_2_1, CD1_1, TPS None, SS1, VWA2, VWB2, SVW2, HPLR2, LDSSI0, TDMIM0, LPA8, LBSPPA512, PLR1, CLS0, AG1, PGL0, PAP0, WGM32 | new |
| 7008x5808x6288 | 4 | MT320x256x128, MIWT10_8, WG32_4_1, CD1_1, TPS None, SS1, VWA2, VWB4, SVW2, HPLR2, LDSSI0, TDMIM0, LPA8, LBSPPA512, PLR1, CLS1, AG1, PGL2, PAP0, WGM16 | previous kept |
| 7040x6912x6256 | 1 | MT256x256x128, MIWT8_8, WG32_4_1, CD2_1, TPS None, SS1, VWA8, VWB4, SVW8, HPLR2, LDSSI1, TDMIM1, LPA8, LBSPPA2048, PLR1, CLS0, AG1, PGL2, PAP0, WGM8 | new |
| 7200x6304x5888 | 8 | MT160x384x128, MIWT10_6, WG16_8_1, CD1_1, TPS None, SS1, VWA2, VWB2, SVW2, HPLR1, LDSSI0, TDMIM0, LPA8, LBSPPA512, PLR1, CLS1, AG1, PGL0, PAP0, WGM8 | previous kept |
| 7808x7088x5424 | 5 | MT256x320x128, MIWT8_10, WG32_4_1, CD4_4, CAIDMn2, TPS None, SS1, VWA8, VWB2, SVW8, HPLR1, LDSSI1, TDMIM0, LPA8, LBSPPA1024, PLR1, CLS0, AG1, PGL2, PAP0, WGM8 | new |
| 7856x5120x7200 | 7 | MT256x320x128, MIWT8_10, WG32_4_1, CD4_4, CAIDMn2, TPS None, SS1, VWA8, VWB2, SVW8, HPLR2, LDSSI1, TDMIM1, LPA8, LBSPPA2048, PLR1, CLS1, AG1, PGL2, PAP0, WGM8 | new |

## Winner key parameters, fused GEMM, LDA = LDB = K (`fused_ldaK/`)

| size | sol | kernel | winner |
|---|---|---|---|
| 4464x5968x6464 | 6 | MT224x256x128, MIWT14_4, WG16_8_1, CD2_2, CAIDMn2, TPS None, SS0, VWA2, VWB4, SVW8, HPLR1, LDSSI0, TDMIM0, LPA8, LBSPPA512, PLR1, CLS0, AG1, PGL0, PAP0, WGM8 | new |
| 4928x4848x7216 | 0 | MT192x256x128, MIWT12_4, WG16_8_1, CD1_1, TPS None, SS1, VWA4, VWB4, SVW4, HPLR0, LDSSI1, TDMIM0, LPA8, LBSPPA1024, PLR1, CLS1, AG1, PGL0, PAP0, WGM8 | kept |
| 5504x6096x5440 | 9 | MT192x384x128, MIWT12_6, WG16_8_1, CD4_4, TPS None, SS1, VWA4, VWB2, SVW4, HPLR1, LDSSI0, TDMIM0, LPA8, LBSPPA1024, PLR1, CLS1, AG1, PGL2, PAP0, WGM8 | new |
| 6000x4928x7216 | 3 | MT192x320x128, MIWT12_5, WG16_8_1, CD1_1, TPS DataParallel, SS1, VWA4, VWB1, SVW4, HPLR0, LDSSI0, TDMIM0, LPA8, LBSPPA1024, PLR1, CLS1, AG1, PGL0, PAP0, WGM8 | new |
| 6496x7056x5664 | 2 | MT384x160x128, MIWT6_10, WG64_2_1, CD1_1, TPS None, SS0, VWA2, VWB2, SVW8, HPLR0, LDSSI0, TDMIM0, LPA8, LBSPPA512, PLR1, CLS1, AG1, PGL0, PAP0, WGM16 | new |
| 7008x5808x6288 | 4 | MT320x256x128, MIWT10_8, WG32_4_1, CD2_1, CAIDMn2, TPS DataParallel, SS0, VWA2, VWB4, SVW8, HPLR1, LDSSI0, TDMIM0, LPA8, LBSPPA512, PLR1, CLS1, AG1, PGL2, PAP0, WGM16 | new |
| 7040x6912x6256 | 1 | MT256x256x128, MIWT8_8, WG32_4_1, CD1_1, TPS None, SS1, VWA8, VWB4, SVW8, HPLR1, LDSSI1, TDMIM1, LPA8, LBSPPA2048, PLR1, CLS1, AG1, PGL2, PAP0, WGM8 | new |
| 7200x6304x5888 | 8 | MT240x256x128, MIWT15_4, WG16_8_1, CD2_1, TPS None, SS0, VWA1, VWB4, SVW8, HPLR2, LDSSI0, TDMIM0, LPA8, LBSPPA256, PLR1, CLS0, AG0, PGL0, PAP0, WGM8 | new |
| 7808x7088x5424 | 5 | MT256x320x128, MIWT8_10, WG32_4_1, CD2_1, TPS StreamK, SS1, VWA8, VWB2, SVW8, HPLR2, LDSSI0, TDMIM1, LPA8, LBSPPA2048, PLR1, CLS1, AG1, PGL2, PAP0, WGM16 | new |
| 7856x5120x7200 | 7 | MT256x320x128, MIWT8_10, WG32_4_1, CD2_1, TPS StreamK, SS1, VWA8, VWB2, SVW8, HPLR2, LDSSI1, TDMIM1, LPA8, LBSPPA2048, PLR1, CLS1, AG1, PGL2, PAP0, WGM8 | new |

## Requirements / notes

- gfx1250 (`Device 73f0`), Tensile with TDM (TDMInst 3), TDMIterateMode, LDSSegmentInterleave, HalfPLR, ClusterDim and
  AdaptiveGemm support. TensileCreateLibrary --architecture=gfx1250 on each logic file: 0 rejects.
- The two tables differ in LDA alignment (with LDA = K, K*2 bytes not a multiple of 128 makes TDM loads go
  indirect) and in the fused epilogue, so the two variants can pick different kernels for the same size.
- The tables come from different GPUs and runs (pure: GPU3; fused: GPU0, plus GPU3 rows where marked); compare gains
  within a row, not absolute us across tables.
- `pure_LdaPadded/logic/`: 2 solution(s) (2 size(s)) use TDMIterateMode 1, which sets
  AssertFree0ElementMultiple 8: they are only selected when M % 8 == 0 (true for these exact sizes).
- `fused_ldaK/logic/`: 3 solution(s) (3 size(s)) use TDMIterateMode 1, which sets
  AssertFree0ElementMultiple 8: they are only selected when M % 8 == 0 (true for these exact sizes).
