# FP32 SGEMM Comparison: VOPD vs Non-VOPD on gfx1100 (7900 XTX)

## Hardware
- GPU: AMD Radeon RX 7900 XTX (gfx1100), 96 CUs, 2 SIMDs/CU
- Max sclk: 2431 MHz (spec), ~1955 MHz sustained under FP32 SGEMM load
- Power: 315W default, tested at 350W cap (drawing ~336-344W)
- CPU: AMD Ryzen 9 7950X
- ROCm: 7.2.0

## Theoretical Peak FP32 Performance
| Mode | Formula | At 2431 MHz | At 1955 MHz (sustained) |
|------|---------|-------------|------------------------|
| Single-issue (v_fmac_f32) | 96 CUs x 2 SIMDs x 32 lanes x 2 ops x clk | 29.87 TFLOPS | 24.01 TFLOPS |
| Dual-issue (v_dual_fmac_f32) | 2 x single-issue | 59.74 TFLOPS | 48.02 TFLOPS |

Note: 61.44 TFLOPS assumes 2500 MHz; actual spec max is 2431 MHz = 59.74 TFLOPS.

## Key Finding: Clock Speed is the Primary Bottleneck

The GPU sustains ~1955 MHz under heavy FP32 SGEMM load (power-limited at ~340W).
At this clock, the dual-issue peak is only 48.02 TFLOPS.

**Our kernel8 achieves 83.1% of the actual sustained clock peak.**
This is an excellent efficiency level; the blog's 50 TFLOPS claim corresponds to
~84% efficiency at ~2430 MHz, suggesting the blog author had a card running at
full boost clock (likely better power delivery or cooling).

## Results: HIP Baseline (Compiler-generated, partial VOPD)

This kernel uses 8x8 register tiling, 128x128 block, 256 threads, BK=8.
**Correction**: The compiler DOES emit `v_dual_fmac_f32` (256 VOPD + 64 single FMAs).
However, it achieves only ~17 TFLOPS due to:
- Bounds checking branches (83 branches vs 9 in kernel8)
- Inefficient memory patterns (66 global_loads vs 32)
- Higher occupancy (16 waves) but worse instruction mix

| M=N=K | TFLOPS | ms/iter | Efficiency (single-issue peak) |
|-------|--------|---------|-------------------------------|
| 256 | 0.46 | 0.073 | 1.9% |
| 512 | 1.98 | 0.136 | 8.2% |
| 1024 | 6.15 | 0.349 | 25.6% |
| 2048 | 8.48 | 2.026 | 35.3% |
| 4096 | 15.56 | 8.833 | 64.8% |
| 8192 | 16.65 | 66.035 | 69.3% |

## Results: VOPD Kernel8 (Hand-crafted v_dual_fmac_f32 assembly)

128x128 block, 128 threads, BK=8, 8x16 ThreadTile, Occupancy 7.
Hardcoded for N=4096 stride (supports N=4096 and 8192).

| M=N=K | TFLOPS | ms/iter | vs Non-VOPD | vs hipBLASLt | Eff (sustained peak) |
|-------|--------|---------|-------------|--------------|---------------------|
| 4096 | 38.50 | 3.570 | 2.48x | 11.1x | 80.2% |
| 8192 | 40.05 | 27.454 | 2.40x | 11.5x | 83.4% |

## Results: Kernel Variant Comparison (N=8192, 350W power cap)

| Variant | TFLOPS | ms/iter | vs Baseline | Description |
|---------|--------|---------|-------------|-------------|
| kernel8 (baseline) | 40.05 | 27.454 | 1.000x | Original blog kernel |
| variant_noprio | 39.67 | 27.714 | 0.991x | Removed all s_setprio |
| variant_prio2 | 39.93 | 27.538 | 0.997x | s_setprio 2 for compute |
| variant_clause8 | 39.96 | 27.514 | 0.998x | Larger s_clause for global loads |
| variant_lds_prefetch | 39.92 | 27.542 | 0.997x | Reordered ds_load/pointer ops |
| variant_wgp_mode | 39.92 | 27.542 | 0.997x | WGP mode instead of CU mode |

**Conclusion:** The original kernel8 is already near-optimal. All variants performed
within 1% of the baseline. The s_setprio mechanism provides a tiny (~1%) benefit.

## Performance Analysis

### Inner Loop Structure
```
Per K-element iteration:
  12 ds_load_b64 (LDS reads: 8 A values + 16 B values)
  s_waitcnt lgkmcnt(0)
  64 v_dual_fmac_f32 (128 FP32 FMA ops)
  2 global_load_b32 (prefetch next tile)
```

### Cycle Budget
- 64 VOPD FMACs: 64 cycles (compute)
- 12 ds_load + waitcnt: ~15-20 cycles stall
- 2 global_load + s_setprio: ~3 cycles
- Total: ~82-87 cycles per K-element
- VALU utilization: 64/85 = ~75% theoretical
- Measured: ~83% of sustained peak = 83% x 75% = ~62% of max-clock theoretical

### Why We Can't Reach 50 TFLOPS
1. **Clock speed**: GPU sustains ~1955 MHz, not 2431 MHz (power-limited at ~340W)
2. **At actual clock**: 83% efficiency is already excellent
3. **50 TFLOPS requires**: either ~2430 MHz sustained clock, OR >100% efficiency at 1955 MHz (impossible)
4. **Blog author likely had**: card running at full boost (better VRM/cooling/silicon lottery)

### What Would Help
- **Higher power limit / better cooling**: Could allow ~2200-2431 MHz sustained
- **UV/OC tuning**: Undervolt to reduce power at same clock = higher sustained frequency
- **Software pipelining**: Could improve inner loop efficiency from 75% to 85-90%, adding ~3-5 TFLOPS at current clock

## Generic VOPD Kernel (works for any N multiple of 128)

Created `kernel8_generic.s` - modified kernel8 to replace hardcoded N=4096 strides
with dynamic computation using the N parameter. Verified correct at N=256, 384, 512.

| N | TFLOPS | ms | Efficiency | vs Hardcoded |
|---|--------|-----|-----------|-------------|
| 256 | 1.13 | 0.030 | 1.8% | N/A |
| 384 | 2.89 | 0.039 | 4.7% | N/A |
| 512 | 5.19 | 0.052 | 8.4% | N/A |
| 768 | 9.68 | 0.094 | 15.8% | N/A |
| 1024 | 19.78 | 0.109 | 32.2% | N/A |
| 2048 | 36.24 | 0.474 | 59.0% | N/A |
| 4096 | 34.89 | 3.939 | 56.8% | 90.6% |
| 8192 | 39.54 | 27.805 | 64.4% | 98.7% |

Performance penalty vs hardcoded kernel: ~9% at 4096, ~1% at 8192.
The penalty comes from s_mul_i32 instructions replacing hardcoded shifts.

## Compiler VOPD Finding

**Correction**: The HIP compiler (hipcc/clang) DOES emit `v_dual_fmac_f32` instructions.
Our "non-VOPD" HIP baseline kernel generates 256 VOPD FMAs + 64 single FMAs.
The performance gap (17 vs 40 TFLOPS) comes from:
- 83 branches (bounds checking) vs 9 in the hand-crafted kernel
- 66 global_loads vs 32 (less efficient memory access)
- Higher occupancy (16 waves) but worse instruction mix
- Bounds checking prevents compiler from fully optimizing

## Tensile Non-MI ThreadTile Results (gfx1100)

Successfully ran Tensile FP32 non-MI assembly kernels on gfx1100 (after rebuilding
the client to fix TypesEqual array size mismatch). All validations PASSED.

| Shape | ThreadTile | MacroTile | DepthU | GFLOPS | TFLOPS |
|-------|-----------|-----------|--------|--------|--------|
| 256x256x256 | 4x4 | 32x32 | 32 | 2,481 | 2.48 |
| 1024x1024x1024 | 4x4 | 64x32 | 16 | 13,585 | 13.6 |
| 1024x1024x1024 | 4x8 | 64x64 | 16 | 15,732 | 15.7 |
| 1024x1024x1024 | **8x8** | **128x64** | 16 | **17,702** | **17.7** |
| 4096x4096x4096 | 4x4 | 64x32 | 16 | 14,869 | 14.9 |
| 4096x4096x4096 | 4x8 | 64x64 | 16 | 19,362 | 19.4 |
| 4096x4096x4096 | **8x8** | **128x64** | 16 | **23,712** | **23.7** |

Best Tensile non-MI: TT8_8 WG16_8_1 DU16 = 23.7 TFLOPS at 4096x4096
VOPD kernel8 at same shape: 38.5 TFLOPS → **VOPD is 1.62x faster**

## hipBLASLt FP32 Baseline (for reference)
| M=N=K | TFLOPS | Notes |
|-------|--------|-------|
| 1024 | 2.59 | Uses WMMA FP32, very suboptimal |
| 2048 | 3.32 | |
| 4096 | 3.47 | |
| 8192 | 3.09 | |

hipBLASLt uses WMMA for FP32 on gfx1100, which is fundamentally the wrong approach.
WMMA achieves only ~5.7% of peak vs our VOPD kernel at ~83% of sustained peak.

## Files

| File | Purpose |
|------|---------|
| `fp32_sgemm_amd/src/kernel8_batched_gmem.s` | Blog's VOPD kernel (baseline, BEST) |
| `fp32_sgemm_amd/src/variant_noprio.s` | No s_setprio instructions |
| `fp32_sgemm_amd/src/variant_prio2.s` | s_setprio 2 for compute |
| `fp32_sgemm_amd/src/variant_clause8.s` | Larger s_clause for global loads |
| `fp32_sgemm_amd/src/variant_lds_prefetch.s` | Reordered LDS loads |
| `fp32_sgemm_amd/src/variant_wgp_mode.s` | WGP mode enabled |
| `bench_compare.hip` | Comparison harness |
| `bench_compare` | Compiled harness binary |

## Clock Speed Measurements During Benchmarks
```
Time    sclk (MHz)  Notes
t=0     idle (0)    GPU idle
t=0.5   1110        Ramping up
t=1.5   1592        Still ramping
t=2.5   1958        Stable
t=3.5   1958        Stable
t=4.5   1955        Stable (power-limited)
t=5.5   1954        Stable
```

Sustained clock under FP32 SGEMM load: **~1955 MHz** at 340W draw, 350W cap.

## Efficiency Summary

| Metric | Value |
|--------|-------|
| VOPD SGEMM TFLOPS (8192x8192) | 40.05 |
| Non-VOPD SGEMM TFLOPS (8192x8192) | 16.65 |
| VOPD Speedup | 2.40x |
| hipBLASLt FP32 TFLOPS | 3.47 |
| VOPD vs hipBLASLt Speedup | 11.5x |
| Efficiency at spec peak (2431 MHz) | 67.1% |
| Efficiency at sustained peak (1955 MHz) | 83.4% |
| Inner loop VALU utilization | ~75% |
