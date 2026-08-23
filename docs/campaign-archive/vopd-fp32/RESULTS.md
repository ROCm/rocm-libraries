# VOPD Dual-Issue FP32 SGEMM Results

## Hardware
- GPU: Radeon RX 7900 XTX (gfx1100), 96 CUs, max sclk 2431 MHz
- CPU: AMD Ryzen 9 7950X
- ROCm: 7.2.0

## Theoretical Peak
- FP32 single-issue: 30.72 TFLOPS (96 CUs × 32 lanes × 2 ops × 2.5 GHz)
- FP32 dual-issue (VOPD): 61.44 TFLOPS

## Results Summary

### hipBLASLt FP32 Baseline (WMMA-based)
| Size | GFLOPS | TFLOPS |
|------|--------|--------|
| 1024 | 2,585 | 2.59 |
| 2048 | 3,319 | 3.32 |
| 4096 | 3,468 | 3.47 |
| 8192 | 3,090 | 3.09 |

### VOPD Kernel8 (v_dual_fmac_f32)
| Size | TFLOPS | Efficiency | Speedup vs hipBLASLt |
|------|--------|------------|---------------------|
| 4096 | 36.08 | 58.7% | 10.4x |
| 8192 | 36.54 | 59.5% | 11.8x |

### Naive HIP Kernel (compiler-generated, no VOPD)
| Size | TFLOPS |
|------|--------|
| 4096 | 14.63 |

## Key Findings

1. **hipBLASLt FP32 on gfx1100 is severely suboptimal**: Uses WMMA FP32, achieves only ~3.5 TFLOPS (5.7% of peak)
2. **VOPD dual-issue achieves 36 TFLOPS**: 10-12x faster than hipBLASLt FP32
3. **Compiler does NOT emit v_dual_fmac_f32**: Only hand-crafted assembly achieves VOPD
4. **Non-MI ThreadTile path not supported on gfx1100**: Crashes at runtime (exit 134)
5. **Tensile infrastructure added**: EnableVOPD parameter, MAC_F32_VOPD component, asmCap detection all implemented

## Files Modified in Tensile/hipBLASLt

| File | Change |
|------|--------|
| `rocisa/include/hardware_caps.hpp` | Added `v_dual_fmac_f32` assembler capability |
| `Tensile/Common/ValidParameters.py` | Added `EnableVOPD` parameter |
| `Tensile/Common/GlobalParameters.py` | Added `EnableVOPD` default |
| `Tensile/SolutionStructs/Solution.py` | Added VOPD validation + non-MI GRVW fixes |
| `Tensile/Components/MAC_F32_VOPD.py` | NEW: VOPD MAC component |
| `Tensile/Components/MAC_F32.py` | Added VOPD exclusion |
| `Tensile/Components/__init__.py` | Registered MAC_F32_VOPD |

## Standalone VOPD Kernel Location
- `/home/vmijovic/vopd_sgemm/fp32_sgemm_amd/` - Blog author's kernel (kernel8)
- `/home/vmijovic/vopd_sgemm/vopd_sgemm_v2.hip` - Our optimized HIP kernel (14.6 TFLOPS without VOPD)

## Next Steps
1. Fix non-MI ThreadTile crash on gfx1100 to enable Tensile-native VOPD
2. Integrate VOPD kernel as custom kernel in hipBLASLt device library
3. Add support for non-square and non-power-of-2 matrix sizes
4. Further optimize memory access patterns for different shapes
