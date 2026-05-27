# okl_run vs hipblaslt-bench timing comparison

Same kernel by index in both runners, 500 hot iters after 500 cold iters, bf16 TN. CPU-timer bench is hipblaslt-bench's default; GPU-timer adds `--use_gpu_timer`.

- Bench   : `/opt/rocm-6.4.3/bin/hipblaslt-bench`
- Library : `/opt/rocm-6.4.3/lib/hipblaslt/library`

| Package | Shape | sol idx | okl_run | bench (CPU) | bench (GPU) | wrapper Δ | bench extra |
|---|---|---|---|---|---|---|---|
| | | | TFLOPS / µs | TFLOPS / µs | TFLOPS / µs | vs bench-GPU | µs/iter |
| `large_square` | 8192×8192×8192 | 45755 | 938.2 / 1171.95 | 667.7 / 1646.63 | 668.2 / 1645.54 | +40.4% | +473.59 |
| `medium_square` | 4096×4096×4096 | 45702 | 844.6 / 162.72 | 627.1 / 219.15 | 626.0 / 219.55 | +34.9% | +56.83 |
| `skinny_M` | 128×4096×4096 | 38136 | 286.0 / 15.02 | 227.7 / 18.86 | 228.1 / 18.83 | +25.4% | +3.81 |
| `skinny_N` | 4096×128×4096 | 38179 | 282.5 / 15.20 | 222.3 / 19.32 | 224.4 / 19.14 | +25.9% | +3.93 |
| `small_square` | 1024×1024×1024 | 45688 | 280.1 / 7.67 | 204.5 / 10.50 | 235.6 / 9.12 | +18.9% | +1.45 |

## Reading

- `okl_run` loads the same .co, packs the kernarg once, and launches in a tight loop via raw `hipExtModuleLaunchKernel`. No Tensile / hipBLASLt link.
- `hipblaslt-bench` wraps each launch in `hipblasLtMatmul`, which validates args, looks up the algo, manages workspace, and can launch additional helper kernels per call.
- **bench extra µs/iter** = `bench_gpu_us - okl_us`. For tiny shapes this looks like CPU-side API marshaling; for large shapes it scales with kernel size, which points at on-stream workspace/state management hipBLASLt does that the raw launch path skips.
- For comparing this kernel against a non-hipBLASLt implementation (cuBLAS, custom assembly), the **okl_run** number is fairer -- both sides pay only for kernel work. For predicting what a real hipBLASLt user observes, the **bench** number is.

## Per-package kernel symbols

- `large_square`: `Cijk_Alik_Bljk_BBS_BH_UserArgs_MT256x224x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA2_GRVWB2_GSUAMB_GLS0_ISA942_IU1_K1_LBSPPA256_LBSPPB128_LBSPPM0_LPA4_LPB4_LPM0_LRVW4_LWPMn1_MIAV0_MIWT8_7_MO40_NTn1_NTA0_NTB0_NTC4_NTD4_NTM0_NEPBS16_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO1_SRVW0_SSO0_SVW2_SK0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGROn1_VSn1_VWA2_VWB1_WSGRA0_WSGRB0_WS64_WG32_8_1`
- `medium_square`: `Cijk_Alik_Bljk_BBS_BH_UserArgs_MT256x224x64_MI16x16x1_SN_LDSB1_AFC0_AFEM8_AFEM8_ASEM32_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA4_GRVWB4_GSUAMB_GLS0_ISA942_IU1_K1_LBSPPA1024_LBSPPB128_LBSPPM0_LPA4_LPB4_LPMn1_LRVW4_LWPMn1_MIAV0_MIWT8_7_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW8_SK0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGROn1_VSn1_VWA8_VWB1_WSGRA0_WSGRB0_WS64_WG32_8_1`
- `skinny_M`: `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x64x256_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMBSK_GLS0_ISA942_IU1_K1_LBSPPA1024_LBSPPB2048_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT2_4_MO40_NTn1_NTA0_NTB0_NTC0_NTD4_NTM0_NEPBS16_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW2_SK0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGROn1_VSn1_VWA2_VWB4_WSGRA0_WSGRB0_WS64_WG16_4_4`
- `skinny_N`: `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x32x256_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMBSK_GLS0_ISA942_IU1_K1_LBSPPA2048_LBSPPB1024_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT4_2_MO40_NTn1_NTA0_NTB0_NTC0_NTD4_NTM0_NEPBS16_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW4_SK0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGROn1_VSn1_VWA4_VWB2_WSGRA0_WSGRB0_WS64_WG16_4_4`
- `small_square`: `Cijk_Alik_Bljk_BBS_BH_UserArgs_MT64x64x128_MI16x16x1_SN_LDSB1_AFC0_AFEM8_AFEM8_ASEM32_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_ISA942_IU1_K1_LBSPPA512_LBSPPB256_LBSPPM0_LPA16_LPB16_LPMn1_LRVW8_LWPMn1_MIAV0_MIWT2_2_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW2_SK0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGROn1_VSn1_VWA2_VWB1_WSGRA0_WSGRB0_WS64_WG32_8_1`
