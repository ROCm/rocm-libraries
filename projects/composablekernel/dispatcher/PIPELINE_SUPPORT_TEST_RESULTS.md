# Grouped Convolution Pipeline Support Test Results

## Summary

All **8 pipelines** defined in `unified_grouped_conv_codegen.py` successfully build and run for all convolution variants (forward, bwd_data, bwd_weight) in both 2D and 3D.

**Date:** 2026-04-26
**Test Script:** `examples/grouped_conv/python/10_test_all_pipelines.py`
**Architecture:** gfx950
**Data Type:** bf16

## Pipelines Tested

From `unified_grouped_conv_codegen.py` lines 701-710:

1. `basic_v1` - BaseGemmPipelineAGmemBGmemCRegV1
2. `mem` - BaseGemmPipelineAgBgCrMem
3. `compv3` - BaseGemmPipelineAgBgCrCompV3
4. `compv4` - BaseGemmPipelineAgBgCrCompV4
5. `compv5` - BaseGemmPipelineAgBgCrCompV5
6. `compv6` - BaseGemmPipelineAgBgCrCompV6
7. `comp_async` - BaseGemmPipelineAgBgCrCompAsync
8. `basic_async_v1` - BaseGemmPipelineAGmemBGmemCRegV1

## Results

### Forward Convolution

#### 2D
| Pipeline | Build | Run | Time (ms) | TFLOPS |
|----------|-------|-----|-----------|--------|
| basic_v1 | ✓ | ✓ | 0.0081 | 0.58 |
| mem | ✓ | ✓ | 0.0138 | 0.34 |
| compv3 | ✓ | ✓ | 0.0081 | 0.58 |
| compv4 | ✓ | ✓ | 0.0076 | 0.62 |
| compv5 | ✓ | ✓ | 0.0081 | 0.58 |
| compv6 | ✓ | ✓ | 0.0145 | 0.32 |
| comp_async | ✓ | ✓ | 0.0080 | 0.59 |
| basic_async_v1 | ✓ | ✓ | 0.0155 | 0.30 |

**Success Rate:** 8/8 (100%)

#### 3D
| Pipeline | Build | Run | Time (ms) | TFLOPS |
|----------|-------|-----|-----------|--------|
| basic_v1 | ✓ | ✓ | 0.0195 | 2.91 |
| mem | ✓ | ✓ | 0.0194 | 2.92 |
| compv3 | ✓ | ✓ | 0.0194 | 2.92 |
| compv4 | ✓ | ✓ | 0.0171 | 3.32 |
| compv5 | ✓ | ✓ | 0.0194 | 2.92 |
| compv6 | ✓ | ✓ | 0.0194 | 2.92 |
| comp_async | ✓ | ✓ | 0.0196 | 2.89 |
| basic_async_v1 | ✓ | ✓ | 0.0197 | 2.87 |

**Success Rate:** 8/8 (100%)

### Backward Data Convolution

#### 2D
| Pipeline | Build | Run | Time (ms) | TFLOPS |
|----------|-------|-----|-----------|--------|
| basic_v1 | ✓ | ✓ | 0.0303 | 0.16 |
| mem | ✓ | ✓ | 0.0132 | 0.36 |
| compv3 | ✓ | ✓ | 0.0121 | 0.39 |
| compv4 | ✓ | ✓ | 0.0122 | 0.39 |
| compv5 | ✓ | ✓ | 0.0121 | 0.39 |
| compv6 | ✓ | ✓ | 0.0124 | 0.38 |
| comp_async | ✓ | ✓ | 0.0124 | 0.38 |
| basic_async_v1 | ✓ | ✓ | 0.0122 | 0.39 |

**Success Rate:** 8/8 (100%)

#### 3D
| Pipeline | Build | Run | Time (ms) | TFLOPS |
|----------|-------|-----|-----------|--------|
| basic_v1 | ✓ | ✓ | 0.0369 | 1.53 |
| mem | ✓ | ✓ | 0.0249 | 2.27 |
| compv3 | ✓ | ✓ | 0.0252 | 2.25 |
| compv4 | ✓ | ✓ | 0.0244 | 2.32 |
| compv5 | ✓ | ✓ | 0.0238 | 2.38 |
| compv6 | ✓ | ✓ | 0.0247 | 2.29 |
| comp_async | ✓ | ✓ | 0.0241 | 2.35 |
| basic_async_v1 | ✓ | ✓ | 0.0242 | 2.34 |

**Success Rate:** 8/8 (100%)

### Backward Weight Convolution

#### 2D
| Pipeline | Build | Run | Time (ms) | TFLOPS |
|----------|-------|-----|-----------|--------|
| basic_v1 | ✓ | ✓ | 0.0035 | 1.33 |
| mem | ✓ | ✓ | 0.0037 | 1.29 |
| compv3 | ✓ | ✓ | 0.0036 | 1.32 |
| compv4 | ✓ | ✓ | 0.0038 | 1.24 |
| compv5 | ✓ | ✓ | 0.0037 | 1.26 |
| compv6 | ✓ | ✓ | 0.0046 | 1.03 |
| comp_async | ✓ | ✓ | 0.0038 | 1.23 |
| basic_async_v1 | ✓ | ✓ | 0.0036 | 1.33 |

**Success Rate:** 8/8 (100%)

#### 3D
| Pipeline | Build | Run | Time (ms) | TFLOPS |
|----------|-------|-----|-----------|--------|
| basic_v1 | ✓ | ✓ | 0.0053 | 10.66 |
| mem | ✓ | ✓ | 0.0049 | 11.57 |
| compv3 | ✓ | ✓ | 0.0054 | 10.49 |
| compv4 | ✓ | ✓ | 0.0054 | 10.49 |
| compv5 | ✓ | ✓ | 0.0053 | 10.60 |
| compv6 | ✓ | ✓ | 0.0054 | 10.52 |
| comp_async | ✓ | ✓ | 0.0078 | 7.29 |
| basic_async_v1 | ✓ | ✓ | 0.0058 | 9.69 |

**Success Rate:** 8/8 (100%)

## Overall Summary

- **Total Tests:** 48 (8 pipelines × 3 variants × 2 dimensions)
- **Successful Builds:** 48/48 (100%)
- **Successful Runs:** 48/48 (100%)

## Updated Configuration

Based on these results, `grouped_config_rules.py` has been updated with:

```python
VARIANT_PIPELINES: Dict[str, List[str]] = {
    "forward": [
        "basic_v1",
        "mem",
        "compv3",
        "compv4",
        "compv5",
        "compv6",
        "comp_async",
        "basic_async_v1",
    ],
    "bwd_data": [
        "basic_v1",
        "mem",
        "compv3",
        "compv4",
        "compv5",
        "compv6",
        "comp_async",
        "basic_async_v1",
    ],
    "bwd_weight": [
        "basic_v1",
        "mem",
        "compv3",
        "compv4",
        "compv5",
        "compv6",
        "comp_async",
        "basic_async_v1",
    ],
}
```

## Test Configuration

- **Tile Configuration:** 16×64×64 (M×N×K)
- **Wave Configuration:** 1×4×1
- **Warp Tile:** 16×16×16
- **Problem Size:**
  - **2D:** N=1, C=64, K=64, H=8, W=8, Y=3, X=3, pad=1
  - **3D:** N=1, C=64, K=64, D=4, H=8, W=8, Z=3, Y=3, X=3, pad=1

## Notes

1. All pipelines successfully compiled and executed on gfx950
2. Performance varies by pipeline (compv4 generally fastest for forward)
3. The test used a single conservative tile configuration (16×64×64)
4. Future work could test larger tiles and different wave configurations
5. compv5/compv6 pipelines were previously thought to be problematic, but they build and run successfully when properly configured

## How to Reproduce

```bash
cd /workspace/rocm-libraries/projects/composablekernel/dispatcher/examples/grouped_conv/python

# Test all variants in 2D
python3 10_test_all_pipelines.py --variant all --ndim 2

# Test all variants in 3D
python3 10_test_all_pipelines.py --variant all --ndim 3

# Test specific variant
python3 10_test_all_pipelines.py --variant forward --ndim 2
```

Results are saved to `pipeline_test_results.json`.
