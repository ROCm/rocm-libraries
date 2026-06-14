# HipFlash2Engine — Flash-Attention 2 as a hipDNN SDPA Provider

## What This Is

This directory contains the implementation of our Flash-Attention 2 V7 kernel
as a **hipDNN engine plugin**, following the `hip-kernel-provider` architecture
from `ROCm/rocm-libraries`.

When integrated, calling `graph->sdpa(q, k, v, attrs)` with FP16 inputs on gfx942
automatically dispatches to our V7 kernel — no API change needed by your colleague.

## Architecture (from hipDNN PDF)

```
hipDNN Frontend:  graph->sdpa(q, k, v, {causal=true, sm_scale=...})
                  set_io_data_type(HALF)        ← FP16: our engine
                  set_io_data_type(BFLOAT16)    ← BF16: existing ASM engine
                  ↓
hipDNN Backend:   isApplicable() → selects HipFlash2Engine
                  buildPlan() → HipFlash2FwdPlan
                  execute() → launches flash2_v7_hipdnn kernel
                  ↓
hipDNN Provider:  HIP-kernel-provider plugin
```

## Performance (measured on real hardware)

| GPU | Config | TFLOPS | vs unfused | Job |
|-----|--------|:---:|:---:|-----|
| MI300X | Prefill causal seq=4096 D=128 | 71.27 | **+8.1×** | OCI 35381 |
| MI325X | Prefill causal seq=4096 D=128 | **78.98** | **+8.1×** | Alola 366655 |
| MI325X | Prefill causal GQA4 seq=4096 | 78.16 | **+8.1×** | Alola 366655 |
| MI325X | Prefill causal seq=2048 D=64 | 87.85 | **+7.1×** | Alola 366655 |

Correctness: 9/9 shapes PASS, MaxErr < 0.002 vs CPU FP32 reference.

## Files

```
hip_flash2_engine/
├── HipFlash2Engine.hpp          — IEngine implementation (isApplicable, etc.)
├── HipFlash2FwdPlanBuilder.hpp  — IPlanBuilder (dispatch heuristic, params)
├── HipFlash2FwdPlan.hip         — HIP kernel (V7: rocWMMA MFMA + causal skip)
└── Container_patch.cpp          — Diff to apply to Container.cpp in the repo

bench_hipdnn_sdpa.cpp            — Standalone benchmark (no full hipDNN build needed)
```

## Integration Steps (to PR into ROCm/rocm-libraries)

### 1. Add engine files
```
cp -r hip_flash2_engine/ \
    <rocm-libraries>/dnn-providers/hip-kernel-provider/src/engines/
```

### 2. Register in Container.cpp
Add the `#ifdef HIPDNN_ENGINE_HIP_FLASH2` block from `Container_patch.cpp`
to `dnn-providers/hip-kernel-provider/src/core/Container.cpp`.

### 3. Register engine ID in data SDK
In `projects/hipdnn/data_sdk/include/hipdnn_data_sdk/utilities/EngineNames.hpp`:
```cpp
constexpr int64_t HIP_FLASH2_ENGINE_ID = 0x4841544E32ULL;
constexpr const char* HIP_FLASH2_ENGINE_NAME = "HIP_FLASH2_SDPA_ENGINE";
```

### 4. Add CMake option
In `dnn-providers/hip-kernel-provider/CMakeLists.txt`:
```cmake
option(ENABLE_HIP_FLASH2_ENGINE "HIP Flash-Attention 2 SDPA engine (FP16, gfx942)" ON)
if(ENABLE_HIP_FLASH2_ENGINE)
    target_compile_definitions(hip_kernel_provider PRIVATE HIPDNN_ENGINE_HIP_FLASH2)
    target_sources(hip_kernel_provider PRIVATE
        src/engines/hip_flash2_engine/HipFlash2Engine.cpp
        src/engines/hip_flash2_engine/HipFlash2FwdPlanBuilder.cpp
        src/engines/hip_flash2_engine/HipFlash2FwdPlan.hip)
endif()
```

### 5. Build and test
```bash
cd dnn-providers/hip-kernel-provider/build
cmake -GNinja -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
      -DENABLE_HIP_FLASH2_ENGINE=ON ..
ninja

# Run standalone benchmark
hipcc -O3 --offload-arch=gfx942 -std=c++17 -I/opt/rocm/include \
    ../bench_hipdnn_sdpa.cpp ../hip_flash2_engine/HipFlash2FwdPlan.hip \
    -L/opt/rocm/lib -lhipblas -o bench_hipdnn_sdpa
./bench_hipdnn_sdpa
```

## Dispatch Logic

Our `isApplicable()` returns true when:
- Device: gfx942 or gfx950
- Input dtype: FP16 (BF16 → existing ASM engine)
- head_dim: 64 or 128
- No dropout, no ALiBi, no attention bias tensor
- seq_len_q > 1 AND seq_len_q * seq_len_kv > 6000 * ceil(seq_len_q/64)
  (Flash2 crossover heuristic — AMDable KB: `attention_prefill_decode_smart_dispatch`)

For decode (seq_len_q == 1): returns false → hipDNN selects a GEMM-based engine.

## Key Technical Details

- **Full rocWMMA MFMA**: both QK^T and P@V use `mma_sync` 16×16×16 (5.8× over scalar)
- **Causal tile skip**: `break` when `ki0 > qi_base+63` (1.39-1.50× at seq≥4096)
- **GQA native**: `hk = hq / (Hq/Hk)` — no K/V replication
- **AMD-specific**: `_Float16`, `__shfl_xor`, wavefront=64, `HIPBLAS_COMPUTE_32F`
- **9 AMDable KB entries** extracted from this optimization work (AMD-AIOSS/AMDable)

## PR Target

**ROCm/rocm-libraries** → `develop` branch  
The existing ASM SDPA engine (BF16) is unchanged. Our engine adds FP16 support
as an additional option selected by the hipDNN backend heuristics.
