# qr_tdm_cr forward performance

The tables report effective QK+PV throughput on gfx1250 B0 with 256 CUs.
All inputs are BF16, contiguous packed THD group mode with uniform lengths,
row-major V, no LSE, dropout, bias, soft cap, quantization scale, or sink.
Causal means bottom-right causal; SWA2048 has left=2047 and right=0.

TFLOPS uses the FMHA example's mask-aware convention:
`2 * B * Hq * unmasked_QK_area * (Dq + Dv) / (median_us * 1e6)`.
For each query row i, the valid key interval ends at `i + K - Q` (inclusive),
clamped to the key sequence; SWA additionally starts at that endpoint minus
2047. Thus masked-out pairs do not count as useful FLOPs. GQA counts query
heads. Softmax work is excluded. Gain is `(baseline_us / cr_us - 1) * 100`,
calculated before rounding. These are kernel-dispatch measurements.

## Representative comparisons

| Model (Hq/Hkv, Dq/Dv) | B | Q/K | Mask | Baseline TFLOPS | CR TFLOPS | Gain |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| d64 (16/2, 64/64) | 2 | 4096/4096 | causal | 492.74 | 1272.71 | +158.29% |
| d64 (16/2, 64/64) | 8 | 1024/33792 | causal | 532.05 | 1452.81 | +173.06% |
| d64 (16/2, 64/64) | 8 | 1024/33792 | swa2048 | 1129.04 | 1210.11 | +7.18% |
| d128tp8 (16/1, 128/128) | 2 | 4096/4096 | causal | 967.71 | 1655.35 | +71.06% |
| d128tp8 (16/1, 128/128) | 8 | 1024/33792 | causal | 1151.55 | 1869.64 | +62.36% |
| d128tp8 (16/1, 128/128) | 8 | 1024/33792 | swa2048 | 865.48 | 1608.06 | +85.80% |
| d128tp16 (8/1, 128/128) | 2 | 4096/4096 | causal | 1030.28 | 1169.62 | +13.52% |
| d128tp16 (8/1, 128/128) | 8 | 1024/33792 | causal | 1131.73 | 1840.79 | +62.65% |
| d128tp16 (8/1, 128/128) | 8 | 1024/33792 | swa2048 | 865.51 | 1572.27 | +81.66% |
| d192 (16/16, 192/128) | 2 | 4096/4096 | causal | 1336.73 | 1628.26 | +21.81% |
| d192 (16/16, 192/128) | 8 | 1024/33792 | causal | 1513.87 | 1729.15 | +14.22% |
| d192 (16/16, 192/128) | 8 | 1024/33792 | swa2048 | 1223.05 | 1439.43 | +17.69% |

## Full matrix

[All 504 paired measurements, TFLOPS, and selected kernel names](qr_tdm_cr_performance.csv).
Each model has 126 cases spanning B=1-64, Q=128-8192 and K=128-73728,
with both masks. The median below is the median of per-case TFLOPS ratios,
not the ratio of averaged TFLOPS or a model-throughput measurement.

| Model | Cases | Median CR/baseline throughput | >2% faster / >2% slower |
| --- | ---: | ---: | --- |
| d64 | 126 | 1.712x | 123 / 0 |
| d128tp8 | 126 | 1.707x | 126 / 0 |
| d128tp16 | 126 | 1.740x | 126 / 0 |
| d192 | 126 | 1.198x | 126 / 0 |
| all | 504 | 1.614x | 501 / 0 |

## Measurement provenance and source selection

The baseline is [PR #14826](https://github.com/ROCm/rocm-libraries/pull/14826)
at archived source `294903f0ec0869120f1bf9d5f34ea8a5aec18332`, not its latest
head. The public PR dispatcher chooses qr_tdm_sched for 480 cases and
qr_tdm_pingpong for 24 D128 cases. The CSV identifies the actual baseline.
The representative rows above all select qr_tdm_sched.

Both baseline and CR were compiled in the same ROCm 10.1 / PyTorch
2.14.0+rocm10.1.0 environment with AMD clang 24 revision
`d6f6cb691863e9258034ab48c1dfce21689d5606`. Each source tree uses its own
production per-source flags, generated files, headers and compile manifest;
shared libraries use -Bsymbolic to prevent cross-library symbol binding.
The 2026-11-24 measurements used GPU 3 on gfx1250 B0, 256 CUs.
No 96-CU partition results are included.

The public PR baseline was measured once and frozen. CR comparisons use two
seeds (17, 28), alternating variant order, five warm-up HIP graph replays,
seven timing samples and approximately 20 ms per sample. CR timing is the
median of the two per-seed medians. Baseline-versus-CR is a comparison across
runs on that server; clocks were not locked. It is not a fresh paired run of
the latest baseline PR and this PR head.

The archived 2026-11-24 tuning included D128 rules subsequently removed.
This report uses `cr_before_us` and `old_kernel` for D128, and `cr_after_us`
and `new_kernel` for D64/D192. All 504 kernel names match the final compiled
CR dispatcher at source commit `630cb0c20d9`. This selection excludes the
removed D128 tuning gains. The relevant device headers are unchanged by
the tile-tuning commits.

A 2026-11-25 recheck on GPU 0 of the currently reserved server rebuilt both
original and tuned CR. It reproduced D64 tuning: geometric mean +4.36%,
44/126 cases >2% faster, none >2% slower. The final D128 dispatcher restores
original CR. D192 was not remeasured on this new server, and PR14826 was not
rerun there. Absolute TFLOPS from these two servers must not be pooled.

## Numerical validation and limits

Archived runs use a sampled FP32 PyTorch reference with TF32 disabled,
checking four query heads per batch. All rows are checked for Q*K <= 4096^2;
larger shapes check first/middle/last query rows. The absolute-error limit is
0.02. Every timed variant also receives repeated bitwise output comparisons
with output buffers filled with NaNs between calls.

The 2026-11-25 full recheck passed all 756 rows (378 shapes, two seeds), with
maximum sampled absolute error 0.0099931 and 4,536 bitwise comparisons.
Final cleanup validation passed 60 rows and 2,880 bitwise comparisons,
with maximum sampled absolute error 0.0088577. Additional 2026-11-24 boundary
checks passed 28,672 bitwise comparisons and maximum sampled absolute error
0.0086818. These checks do not qualify backward/training numerical behavior.

Generated CR dispatch is inference-only: LSE and dropout must both be off.
Ragged packed groups and nonpositive scales using General tuning need
additional GPU coverage for the new geometries. The uniform packed benchmark
is not an exhaustive test of all supported features or head dimensions.
Unaligned D128 SWA boundaries outside this matrix showed approximately 2-4%
slower timing in the earlier tuning experiments. Full component regression,
framework end-to-end performance and other architectures are pending.
