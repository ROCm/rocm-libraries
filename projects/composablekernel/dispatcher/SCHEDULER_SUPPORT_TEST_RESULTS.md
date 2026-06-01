# Grouped Convolution Scheduler Support Test Results

## Summary

Both schedulers (`intrawave` and `interwave`) work with **all 8 pipelines** for all convolution variants (forward, bwd_data, bwd_weight) in both 2D and 3D.

**Date:** 2026-04-26
**Test Script:** `examples/grouped_conv/python/11_test_schedulers.py`
**Architecture:** gfx950
**Data Type:** bf16

## Schedulers Tested

From `unified_gemm_codegen.py` line 932:

1. **`intrawave`** - Intra-wave scheduling
2. **`interwave`** - Inter-wave scheduling

## Overall Results Summary

- **Total Tests:** 96 (8 pipelines × 2 schedulers × 3 variants × 2 dimensions)
- **Successful Builds:** 96/96 (100%)
- **Successful Runs:** 96/96 (100%)

**Conclusion:** Both `intrawave` and `interwave` schedulers are fully supported for all pipeline and variant combinations.

## Detailed Results by Variant

### Forward Convolution (2D)

| Pipeline | intrawave | interwave |
|----------|-----------|-----------|
| basic_v1 | ✓ (0.0083ms, 0.57 TFLOPS) | ✓ (0.0143ms, 0.33 TFLOPS) |
| mem | ✓ (0.0082ms, 0.57 TFLOPS) | ✓ (0.0135ms, 0.35 TFLOPS) |
| compv3 | ✓ (0.0082ms, 0.58 TFLOPS) | ✓ (0.0147ms, 0.32 TFLOPS) |
| compv4 | ✓ (0.0074ms, 0.64 TFLOPS) | ✓ (0.0154ms, 0.31 TFLOPS) |
| compv5 | ✓ (0.0082ms, 0.57 TFLOPS) | ✓ (0.0163ms, 0.29 TFLOPS) |
| compv6 | ✓ (0.0084ms, 0.56 TFLOPS) | ✓ (0.0082ms, 0.57 TFLOPS) |
| comp_async | ✓ (0.0123ms, 0.38 TFLOPS) | ✓ (0.0086ms, 0.55 TFLOPS) |
| basic_async_v1 | ✓ (0.0150ms, 0.31 TFLOPS) | ✓ (0.0085ms, 0.56 TFLOPS) |

**Success Rate:** 16/16 (100%)

### Forward Convolution (3D)

| Pipeline | intrawave | interwave |
|----------|-----------|-----------|
| basic_v1 | ✓ (0.0197ms, 2.88 TFLOPS) | ✓ (0.0195ms, 2.90 TFLOPS) |
| mem | ✓ (0.0195ms, 2.90 TFLOPS) | ✓ (0.0199ms, 2.84 TFLOPS) |
| compv3 | ✓ (0.0202ms, 2.80 TFLOPS) | ✓ (0.0194ms, 2.91 TFLOPS) |
| compv4 | ✓ (0.0177ms, 3.19 TFLOPS) | ✓ (0.0175ms, 3.23 TFLOPS) |
| compv5 | ✓ (0.0197ms, 2.87 TFLOPS) | ✓ (0.0195ms, 2.91 TFLOPS) |
| compv6 | ✓ (0.0193ms, 2.93 TFLOPS) | ✓ (0.0187ms, 3.02 TFLOPS) |
| comp_async | ✓ (0.0203ms, 2.79 TFLOPS) | ✓ (0.0195ms, 2.90 TFLOPS) |
| basic_async_v1 | ✓ (0.0222ms, 2.55 TFLOPS) | ✓ (0.0200ms, 2.83 TFLOPS) |

**Success Rate:** 16/16 (100%)

### Backward Data (2D)

| Pipeline | intrawave | interwave |
|----------|-----------|-----------|
| basic_v1 | ✓ (0.0185ms, 0.26 TFLOPS) | ✓ (0.0123ms, 0.38 TFLOPS) |
| mem | ✓ (0.0132ms, 0.36 TFLOPS) | ✓ (0.0127ms, 0.37 TFLOPS) |
| compv3 | ✓ (0.0119ms, 0.39 TFLOPS) | ✓ (0.0128ms, 0.37 TFLOPS) |
| compv4 | ✓ (0.0121ms, 0.39 TFLOPS) | ✓ (0.0123ms, 0.38 TFLOPS) |
| compv5 | ✓ (0.0123ms, 0.38 TFLOPS) | ✓ (0.0134ms, 0.35 TFLOPS) |
| compv6 | ✓ (0.0121ms, 0.39 TFLOPS) | ✓ (0.0123ms, 0.38 TFLOPS) |
| comp_async | ✓ (0.0122ms, 0.39 TFLOPS) | ✓ (0.0121ms, 0.39 TFLOPS) |
| basic_async_v1 | ✓ (0.0141ms, 0.33 TFLOPS) | ✓ (0.0125ms, 0.38 TFLOPS) |

**Success Rate:** 16/16 (100%)

### Backward Data (3D)

| Pipeline | intrawave | interwave |
|----------|-----------|-----------|
| basic_v1 | ✓ (0.0333ms, 1.70 TFLOPS) | ✓ (0.0239ms, 2.37 TFLOPS) |
| mem | ✓ (0.0246ms, 2.30 TFLOPS) | ✓ (0.0245ms, 2.31 TFLOPS) |
| compv3 | ✓ (0.0253ms, 2.24 TFLOPS) | ✓ (0.0246ms, 2.30 TFLOPS) |
| compv4 | ✓ (0.0253ms, 2.24 TFLOPS) | ✓ (0.0248ms, 2.28 TFLOPS) |
| compv5 | ✓ (0.0246ms, 2.30 TFLOPS) | ✓ (0.0243ms, 2.33 TFLOPS) |
| compv6 | ✓ (0.0251ms, 2.26 TFLOPS) | ✓ (0.0243ms, 2.33 TFLOPS) |
| comp_async | ✓ (0.0245ms, 2.31 TFLOPS) | ✓ (0.0243ms, 2.33 TFLOPS) |
| basic_async_v1 | ✓ (0.0245ms, 2.31 TFLOPS) | ✓ (0.0245ms, 2.31 TFLOPS) |

**Success Rate:** 16/16 (100%)

### Backward Weight (2D)

| Pipeline | intrawave | interwave |
|----------|-----------|-----------|
| basic_v1 | ✓ (0.0037ms, 1.27 TFLOPS) | ✓ (0.0036ms, 1.31 TFLOPS) |
| mem | ✓ (0.0049ms, 0.96 TFLOPS) | ✓ (0.0053ms, 0.89 TFLOPS) |
| compv3 | ✓ (0.0043ms, 1.09 TFLOPS) | ✓ (0.0035ms, 1.33 TFLOPS) |
| compv4 | ✓ (0.0036ms, 1.32 TFLOPS) | ✓ (0.0036ms, 1.33 TFLOPS) |
| compv5 | ✓ (0.0036ms, 1.32 TFLOPS) | ✓ (0.0039ms, 1.21 TFLOPS) |
| compv6 | ✓ (0.0036ms, 1.32 TFLOPS) | ✓ (0.0043ms, 1.09 TFLOPS) |
| comp_async | ✓ (0.0040ms, 1.17 TFLOPS) | ✓ (0.0036ms, 1.32 TFLOPS) |
| basic_async_v1 | ✓ (0.0036ms, 1.33 TFLOPS) | ✓ (0.0044ms, 1.06 TFLOPS) |

**Success Rate:** 16/16 (100%)

### Backward Weight (3D)

| Pipeline | intrawave | interwave |
|----------|-----------|-----------|
| basic_v1 | ✓ (0.0053ms, 10.60 TFLOPS) | ✓ (0.0055ms, 10.22 TFLOPS) |
| mem | ✓ (0.0051ms, 11.08 TFLOPS) | ✓ (0.0049ms, 11.50 TFLOPS) |
| compv3 | ✓ (0.0057ms, 9.97 TFLOPS) | ✓ (0.0054ms, 10.40 TFLOPS) |
| compv4 | ✓ (0.0052ms, 10.78 TFLOPS) | ✓ (0.0053ms, 10.54 TFLOPS) |
| compv5 | ✓ (0.0054ms, 10.47 TFLOPS) | ✓ (0.0054ms, 10.46 TFLOPS) |
| compv6 | ✓ (0.0058ms, 9.76 TFLOPS) | ✓ (0.0056ms, 10.06 TFLOPS) |
| comp_async | ✓ (0.0056ms, 10.05 TFLOPS) | ✓ (0.0053ms, 10.60 TFLOPS) |
| basic_async_v1 | ✓ (0.0058ms, 9.78 TFLOPS) | ✓ (0.0054ms, 10.49 TFLOPS) |

**Success Rate:** 16/16 (100%)

## Performance Observations

### 2D Forward
- **intrawave** generally faster for most pipelines
- **compv4** achieves best performance with intrawave (0.0074ms, 0.64 TFLOPS)

### 3D Forward
- Both schedulers perform similarly
- **compv4** best performer (0.0177ms intrawave, 0.0175ms interwave, ~3.2 TFLOPS)

### 2D Backward Data
- **interwave** generally faster for basic_v1
- Most pipelines comparable between schedulers

### 3D Backward Data
- **interwave** significantly faster for basic_v1
- Other pipelines similar between schedulers

### 2D Backward Weight
- Both schedulers perform very similarly
- Fastest overall: compv4, basic_v1, basic_async_v1 (~1.3 TFLOPS)

### 3D Backward Weight
- Both schedulers perform similarly
- **mem** pipeline achieves best performance (11.5 TFLOPS with interwave)
- Generally high performance (9-11 TFLOPS range)

## Recommendations

1. **Both schedulers are viable** - All 96 combinations work correctly
2. **Performance varies by use case:**
   - For 2D forward: prefer `intrawave`
   - For 3D backward data with basic_v1: prefer `interwave`
   - For other cases: both schedulers perform comparably
3. **No blockers** - Either scheduler can be used with any pipeline

## Test Configuration

- **Tile Configuration:** 16×64×64 (M×N×K)
- **Wave Configuration:** 1×4×1
- **Warp Tile:** 16×16×16
- **Problem Size:**
  - **2D:** N=1, C=64, K=64, H=8, W=8, Y=3, X=3, pad=1
  - **3D:** N=1, C=64, K=64, D=4, H=8, W=8, Z=3, Y=3, X=3, pad=1

## How to Reproduce

```bash
cd /workspace/rocm-libraries/projects/composablekernel/dispatcher/examples/grouped_conv/python

# Test all schedulers for all variants in 2D
python3 11_test_schedulers.py --variant all --ndim 2

# Test all schedulers for all variants in 3D
python3 11_test_schedulers.py --variant all --ndim 3

# Test specific combination
python3 11_test_schedulers.py --variant forward --scheduler intrawave --ndim 2
```

Results are saved to `scheduler_test_results.json` (2D) and `scheduler_test_results_3d.json` (3D).
