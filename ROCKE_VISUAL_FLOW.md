# rocKE Visual Flow - Quick Reference

## Overview: Three Distinct Workflows

rocKE supports three separate workflows. **Production inference uses only C++; Python is for development and training.**

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. INFERENCE (Production Runtime) - C++ ONLY, no Python        │
│    hipDNN → hip-kernel-provider.so → C++ instances → GPU       │
│    Uses pre-trained ML models (if available)                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 2. ML TRAINING (Offline) - Python heuristics/ directory        │
│    Generate benchmark data → Extract features → Train models   │
│    Output: model_tflops.lgbm (loaded by C++ at runtime)        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 3. KERNEL DEVELOPMENT (Authoring) - Python → C++ porting       │
│    Write in Python instances → Test → Port to C++ → Verify     │
│    Both engines must emit byte-identical LLVM IR               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Workflow 1: Production Inference (C++ Only)

**This is what runs in production. Python is NOT involved.**

```
+------------------------------------------------------------------+
|  HIPDNN APPLICATION                                              |
|  projects/hipdnn/backend/                                        |
+------------------------------------------------------------------+
                            |
                            | Operation Request (SDPA, GEMM, etc.)
                            v
+------------------------------------------------------------------+
|  HIP-KERNEL-PROVIDER PLUGIN (hip_kernel_provider.so)             |
|  dnn-providers/hip-kernel-provider/src/                          |
|                                                                  |
|  +------------------------------------------------------------+  |
|  | Plugin Entry (core/PluginPublic.cpp)                       |  |
|  |   - hipdnnEnginePluginGetApplicableEngineIds()             |  |
|  |   - hipdnnEnginePluginCreateExecutionContext()             |  |
|  |   - hipdnnEnginePluginExecuteOpGraph()                     |  |
|  +------------------------------------------------------------+  |
|           |                                                      |
|           | Route to engine based on op graph                    |
|           v                                                      |
|  +------------------------------------------------------------+  |
|  | Engine Selection (engines/)                                |  |
|  |                                                            |  |
|  |  asm_sdpa_engine/                                          |  |
|  |    ├─ AsmSdpaEngine.cpp                                    |  |
|  |    └─ plans/SdpaFwdPlanBuilder.cpp                         |  |
|  |       (Assembly kernels, no rocKE)                         |  |
|  |                                                            |  |
|  |  hip_mlops_engine/                                         |  |
|  |    ├─ HipMlopsEngine.cpp                                   |  |
|  |    ├─ plans/ApplicabilityChecks.cpp                        |  |
|  |    └─ plans/*PlanBuilder.cpp                               |  |
|  |       (Calls rocKE for kernel generation)                  |  |
|  +------------------------------------------------------------+  |
|           |                                                      |
|           | Plan builder invokes rocKE                           |
|           v                                                      |
+------------------------------------------------------------------+
            |
            v
+------------------------------------------------------------------+
|  ROCKE C++ ENGINE (librocke_core.a)                              |
|  dnn-providers/hip-kernel-provider/rocKE/Cpp/                    |
|                                                                  |
|  +------------------------------------------------------------+  |
|  | C++ Instances (Cpp/instances/gfx950/)                      |  |
|  |   - attention_tiled_2d_ctx_prologue.cpp                    |  |
|  |   - attention_tiled_2d_kv_body_qk_softmax.cpp              |  |
|  |   - attention_tiled_2d_kv_body_pv_epilogue.cpp             |  |
|  |   - ... (13 C++ files for SDPA)                            |  |
|  +------------------------------------------------------------+  |
|           |                                                      |
|           | KernelDef (SSA IR)                                   |
|           v                                                      |
|  +------------------------------------------------------------+  |
|  | C++ Core Engine (Cpp/core/)                                |  |
|  |   - LLVM IR generation (mirrors Python engine)             |  |
|  |   - Type system & verification                             |  |
|  +------------------------------------------------------------+  |
|           |                                                      |
|           | LLVM IR (.ll file)                                   |
|           v                                                      |
|  +------------------------------------------------------------+  |
|  | Runtime (C++ wrappers over libamd_comgr)                   |  |
|  |   - Compile .ll → HSACO                                    |  |
|  |   - Launch kernel on GPU                                   |  |
|  +------------------------------------------------------------+  |
+------------------------------------------------------------------+
            |
            v
         AMD GPU (gfx942, gfx950, gfx1151, gfx1201)
```

**Key Point:** The C++ engine can optionally load pre-trained ML models (`model_tflops.lgbm`) for kernel selection, but the models are trained offline (Workflow 2).

---

## Workflow 2: ML Training (Offline, Python)

**Entirely in `Python/rocke/heuristics/` - does NOT involve hipDNN or kernel execution.**

```
+------------------------------------------------------------------+
| OFFLINE ML TRAINING PIPELINE (Python only)                       |
| Python/rocke/heuristics/                                         |
|                                                                  |
|  Step 1: Generate Benchmark Data                                 |
|  +------------------------------------------------------------+  |
|  | gen_gemm_sweep_data.py                                     |  |
|  |   - Enumerate (M, N, K, dtype, layout) problem shapes      |  |
|  |   - Enumerate kernel configs (tile sizes, warps, etc.)     |  |
|  |   - Run each (problem, kernel) pair on GPU                 |  |
|  |   - Measure TFLOPS/latency                                 |  |
|  |   - Output: benchmark_data.parquet                         |  |
|  +------------------------------------------------------------+  |
|           |                                                      |
|           v                                                      |
|  Step 2: Feature Extraction                                      |
|  +------------------------------------------------------------+  |
|  | feature_engine.py                                          |  |
|  |   - Extract 55 features per (problem, kernel) pair:        |  |
|  |     * Problem: M, N, K, aspect ratios (13 features)        |  |
|  |     * Kernel: tile sizes, warps, LDS usage (17 features)   |  |
|  |     * Interaction: tile efficiency, CU util (9 features)   |  |
|  |     * Hardware: CUs, cache, clock (12 features)            |  |
|  |   - Output: features.parquet                               |  |
|  +------------------------------------------------------------+  |
|           |                                                      |
|           v                                                      |
|  Step 3: Train LightGBM Models                                   |
|  +------------------------------------------------------------+  |
|  | train.py                                                   |  |
|  |   - Train regression models:                               |  |
|  |     * model_tflops.lgbm (primary)                          |  |
|  |     * model_latency.lgbm (latency-sensitive)               |  |
|  |     * model_bandwidth.lgbm (memory-bound)                  |  |
|  |   - Save feature spec: feature_spec.json                   |  |
|  +------------------------------------------------------------+  |
|           |                                                      |
|           v                                                      |
|  Output: Trained Models (stored externally)                      |
|    - model_tflops.lgbm      ◄─── Loaded by C++ engine at runtime |
|    - feature_spec.json                                           |
+------------------------------------------------------------------+

NOTE: No pre-trained models are included in the repository.
      Users must run this pipeline to generate their own models.
```

**Runtime Inference (from Workflow 1) loads `model_tflops.lgbm`:**
```
C++ Dispatch → Load model_tflops.lgbm → Predict TFLOPS for candidates → Select best kernel
```

---

## Workflow 3: Kernel Development (Python → C++ Porting)

**For developers adding or optimizing kernels.**

```
+------------------------------------------------------------------+
| KERNEL AUTHORING WORKFLOW (Python → C++)                         |
|                                                                  |
| Step 1: Write in Python                                          |
|  +------------------------------------------------------------+  |
|  | Python/rocke/instances/gfx950/attention_tiled_2d.py        |  |
|  |   - Define spec dataclass                                  |  |
|  |   - Implement build_*() → KernelDef                        |  |
|  |   - Test with Python engine                                |  |
|  |   - Validate correctness (pytest, torch comparisons)       |  |
|  +------------------------------------------------------------+  |
|           |                                                      |
|           v                                                      |
| Step 2: Performance Testing (Python)                             |
|  +------------------------------------------------------------+  |
|  | Python/rocke/benchmark/ or heuristics/gen_*_sweep_data.py  |  |
|  |   - Run kernel on GPU with various problem sizes           |  |
|  |   - Measure TFLOPS, latency, bandwidth                     |  |
|  |   - Compare against baseline (Triton, CK, vendor libs)     |  |
|  |   - Profile with rocprof (occupancy, LDS usage, cache)     |  |
|  |   - Tune hyperparameters (tile size, num_warps, etc.)      |  |
|  |   - Generate benchmark data for ML training (optional)     |  |
|  +------------------------------------------------------------+  |
|           |                                                      |
|           v                                                      |
| Step 3: Manually Port to C++                                     |
|  +------------------------------------------------------------+  |
|  | Cpp/instances/gfx950/attention_tiled_2d_*.cpp              |  |
|  |   - Split into compilation units (13 files for attention)  |  |
|  |   - Map Python closures → C++ context structs              |  |
|  |   - Handle arg evaluation order (Python left→right,        |  |
|  |     C++ right→left)                                        |  |
|  |   - Document line mappings in comments:                    |  |
|  |     "// (Python build body lines 711-1135)"                |  |
|  +------------------------------------------------------------+  |
|           |                                                      |
|           v                                                      |
| Step 4: Verify Byte-Identity                                     |
|  +------------------------------------------------------------+  |
|  | tools/check_byte_identity.py                               |  |
|  |   - Build both engines                                     |  |
|  |   - Lower same kernel spec to LLVM IR                      |  |
|  |   - Diff the .ll output byte-for-byte                      |  |
|  |   - MUST be GREEN (identical) to merge                     |  |
|  +------------------------------------------------------------+  |
|           |                                                      |
|           v                                                      |
| Step 5: Performance Validation (C++)                             |
|  +------------------------------------------------------------+  |
|  | Verify C++ performance matches Python                      |  |
|  |   - Run same benchmarks with C++ engine                    |  |
|  |   - Confirm TFLOPS/latency are identical                   |  |
|  |   - Any divergence indicates porting bug                   |  |
|  +------------------------------------------------------------+  |
|                                                                  |
|  Definition of "Done": Both engines pass differential gate       |
|                        AND performance matches                   |
+------------------------------------------------------------------+
```

**Python Instances vs C++ Instances:**
- **Python** (`Python/rocke/instances/common/attention_unified.py`):
  - 4312 lines, single file
  - High-level orchestration, dispatch logic, problem specs
  - Used during development, testing, and prototyping
- **C++** (`Cpp/instances/gfx950/attention_tiled_2d_*.cpp`):
  - 6387 lines total, split into 13 files
  - Same logic as Python, manually ported
  - Used in production (linked into `librocke_core.a`)

---

## Key File Index

### Production Runtime (C++)
```
C++ Engine Entry         Cpp/core/ → librocke_core.a
C++ SDPA Kernels         Cpp/instances/gfx950/attention_tiled_2d_*.cpp
C++ Dispatch             (Built into librocke_core.a, not separate)
Plugin Integration       dnn-providers/hip-kernel-provider/src/core/PluginPublic.cpp
hipDNN Plugin Interface  projects/hipdnn/backend/src/plugin/EnginePlugin.hpp
```

### ML Training (Python, Offline)
```
Data Generation          Python/rocke/heuristics/gen_gemm_sweep_data.py
Feature Engineering      Python/rocke/heuristics/feature_engine.py
Model Training           Python/rocke/heuristics/train.py
Trained Models           model_tflops.lgbm (not in repo, user-generated)
```

### Development (Python Authoring)
```
Python SDPA Kernels      Python/rocke/instances/gfx950/attention_tiled_2d.py
Python Dispatch          Python/rocke/dispatch/families/attention.py
Python High-Level API    Python/rocke/instances/common/attention_unified.py
Benchmarking             Python/rocke/benchmark/
Performance Sweeps       Python/rocke/heuristics/gen_*_sweep_data.py
Correctness Tests        Python/rocke/tests/ (pytest)
Byte-Identity Check      tools/check_byte_identity.py
```

---

## When is Python Used?

| Scenario | Python Used? | Which Workflow? |
|----------|--------------|-----------------|
| **Production inference in hipDNN** | ❌ NO (C++ only) | Workflow 1 |
| **Training ML heuristics models** | ✅ YES | Workflow 2 |
| **Developing new kernels** | ✅ YES (authoring) | Workflow 3 |
| **Performance testing/benchmarking kernels** | ✅ YES (Python engine) | Workflow 3 |
| **Tuning kernel hyperparameters** | ✅ YES (Python sweeps) | Workflow 3 |
| **Correctness testing/validation** | ✅ YES (pytest, torch) | Workflow 3 |
| **Porting Python kernel to C++** | ✅ YES (as reference) | Workflow 3 |
| **Verifying byte-identity** | ✅ YES (runs both engines) | Workflow 3 |
| **Verifying C++ perf matches Python** | ✅ YES (both engines) | Workflow 3 |

---

## Quick Commands

```bash
# ===== Workflow 1: Production Build (C++ only) =====
export ROCKE=$PWD/dnn-providers/hip-kernel-provider/rocKE
cmake -S $ROCKE -B /tmp/rocke -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/rocke --target rocke_core -j$(nproc)
# → /tmp/rocke/librocke_core.a (linked into hip_kernel_provider.so)

# ===== Workflow 2: ML Training =====
export PYTHONPATH=$ROCKE/Python
cd $ROCKE/Python/rocke/heuristics

# Step 1: Generate benchmark data
python gen_gemm_sweep_data.py --output benchmark_data.parquet

# Step 2: Extract features (automatic in train.py)
# Step 3: Train models
python train.py --input benchmark_data.parquet --output model_tflops.lgbm

# ===== Workflow 3: Kernel Development =====
export PYTHONPATH=$ROCKE/Python

# Step 1: Develop in Python
cd $ROCKE/Python/rocke/instances/gfx950
# Edit attention_tiled_2d.py

# Step 2: Performance test (Python engine)
cd $ROCKE/Python/rocke/benchmark
# Run benchmarks, measure TFLOPS/latency
python benchmark_attention.py --arch gfx950 --dtype fp16

# Or generate sweep data for analysis
cd $ROCKE/Python/rocke/heuristics
python gen_gemm_sweep_data.py --arch gfx950

# Step 3: Correctness test
python $ROCKE/tests/run_all.py --only attention

# Step 4: Port to C++
cd $ROCKE/Cpp/instances/gfx950
# Edit attention_tiled_2d_*.cpp

# Step 5: Verify byte-identity
python $ROCKE/tools/check_byte_identity.py

# Step 6: Verify C++ performance matches Python
# (Re-run benchmarks with C++ engine, compare results)
```

---

---

## Testing During Kernel Development - The Mechanics

**You're writing a new kernel variant in Python. How do you test it incrementally?**

### 1. Write a Standalone Test Script

**Create a minimal test for your kernel** (don't rely on full test suite initially):

```python
# test_my_new_attention_variant.py
import torch
from rocke.instances.gfx950.attention_tiled_2d import build_my_new_variant
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.runtime.launcher import compile_and_launch

# 1. Define problem
batch, seqlen, nheads, headdim = 2, 128, 8, 128
q = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float16, device='cuda')
k = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float16, device='cuda')
v = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float16, device='cuda')

# 2. Build your kernel
spec = MyAttentionSpec(tile_m=128, num_warps=4, ...)
kernel_def = build_my_new_variant(spec, arch='gfx950')

# 3. Compile and run
llvm_ir = lower_kernel_to_llvm(kernel_def, arch='gfx950')
output = compile_and_launch(llvm_ir, q, k, v)

# 4. Reference (torch SDPA or Triton)
ref_output = torch.nn.functional.scaled_dot_product_attention(q, k, v)

# 5. Validate
max_diff = (output - ref_output).abs().max().item()
print(f"Max diff vs torch: {max_diff}")
assert max_diff < 1e-3, f"Correctness failed! {max_diff=}"
print("✓ Correctness passed")
```

**Iterate fast:** Run just this script, not the full test suite.

```bash
export PYTHONPATH=$ROCKE/Python
python test_my_new_attention_variant.py  # ~10 seconds vs minutes for full suite
```

### 2. Benchmark Your Kernel

**Quick timing script** (measure just your kernel):

```python
# bench_my_kernel.py
import torch
import time
from rocke.runtime.launcher import compile_and_launch

# Setup (same as correctness test)
spec = MyAttentionSpec(tile_m=128, num_warps=4, ...)
kernel_def = build_my_new_variant(spec, arch='gfx950')
llvm_ir = lower_kernel_to_llvm(kernel_def, arch='gfx950')

# Warmup
for _ in range(10):
    output = compile_and_launch(llvm_ir, q, k, v)
torch.cuda.synchronize()

# Timed runs
start = time.perf_counter()
for _ in range(100):
    output = compile_and_launch(llvm_ir, q, k, v)
torch.cuda.synchronize()
elapsed = time.perf_counter() - start

# Calculate TFLOPS (for attention: 4 * batch * nheads * seqlen^2 * headdim)
flops = 4 * batch * nheads * seqlen * seqlen * headdim * 100
tflops = (flops / elapsed) / 1e12
print(f"Performance: {tflops:.2f} TFLOPS, {elapsed/100*1000:.3f} ms/iter")
```

**Compare against baseline:**

```python
# Benchmark Triton/torch reference
start = time.perf_counter()
for _ in range(100):
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
torch.cuda.synchronize()
ref_elapsed = time.perf_counter() - start
ref_tflops = (flops / ref_elapsed) / 1e12

print(f"Your kernel:  {tflops:.2f} TFLOPS")
print(f"Torch SDPA:   {ref_tflops:.2f} TFLOPS")
print(f"Speedup:      {tflops/ref_tflops:.2f}x")
```

### 3. Sweep Hyperparameters

**Find optimal tile sizes and warps:**

```python
# sweep_hyperparams.py
results = []
for tile_m in [64, 128, 256]:
    for num_warps in [1, 2, 4, 8]:
        try:
            spec = MyAttentionSpec(tile_m=tile_m, num_warps=num_warps, ...)
            kernel_def = build_my_new_variant(spec, arch='gfx950')
            
            # Quick benchmark
            tflops = benchmark_kernel(kernel_def, q, k, v, iters=50)
            results.append({
                'tile_m': tile_m,
                'num_warps': num_warps,
                'tflops': tflops
            })
            print(f"{tile_m=:3d}, {num_warps=}, TFLOPS={tflops:.2f}")
        except Exception as e:
            print(f"{tile_m=:3d}, {num_warps=}, FAILED: {e}")

# Find best config
best = max(results, key=lambda x: x['tflops'])
print(f"\nBest: tile_m={best['tile_m']}, num_warps={best['num_warps']}, {best['tflops']:.2f} TFLOPS")
```

**Output example:**
```
tile_m= 64, num_warps=1, TFLOPS=23.45
tile_m= 64, num_warps=2, TFLOPS=38.21
tile_m=128, num_warps=4, TFLOPS=52.78  ← Best
tile_m=256, num_warps=8, TFLOPS=48.12
```

### 4. Profile with rocprof

**Understand bottlenecks:**

```bash
rocprof --hip-trace python bench_my_kernel.py

# Generates:
# - results.csv: kernel timings
# - results.json: detailed metrics
```

**Analyze:**
```bash
# Check occupancy
grep "Occupancy" results.csv

# Check LDS usage
grep "LDS" results.csv

# Look for your kernel
grep "my_new_attention" results.csv
```

**Common metrics to check:**
- **Occupancy**: < 50% = too much LDS or too many registers
- **LDS usage**: Compare against hardware limit (64KB on CDNA3)
- **Memory bandwidth**: Compare against peak (5.3 TB/s for MI300X)
- **VMEM/SMEM stalls**: High stalls = memory bottleneck

### 5. Incremental Development Cycle

**Fast iteration loop:**

```bash
# Inner loop (repeat frequently):
vim Python/rocke/instances/gfx950/my_kernel.py       # Edit kernel
python test_my_kernel.py                             # Correctness check (~10s)
python bench_my_kernel.py                            # Quick perf (~30s)

# When performance looks good:
python sweep_hyperparams.py                          # Find optimal config (~5 min)

# When ready to validate:
rocprof --hip-trace python bench_my_kernel.py        # Profile bottlenecks
```

**Full validation (before porting to C++):**

```bash
# 1. Run full test suite for your operator
python tests/run_all.py --only attention             # ~2 min

# 2. Run structured benchmark (if available)
python -m rocke.benchmark.gemm.fp16_rcr_sweep --compile --run

# 3. Generate data for ML training (optional)
python heuristics/gen_gemm_sweep_data.py --arch gfx950
```

### 6. After Python Kernel is Ready

**Port to C++ and verify:**

```bash
# 1. Port Python → C++
vim Cpp/instances/gfx950/attention_tiled_2d_*.cpp

# 2. Byte-identity gate (MUST pass)
python tools/check_byte_identity.py
# → Must be GREEN for your kernel family

# 3. Verify C++ performance matches Python
ROCKE_BACKEND=cpp python bench_my_kernel.py
# → TFLOPS should match Python within ~1%
```

**Byte-identity gate:**
- Runs 65 `*_emit.py` / `*_emit.c` pairs
- Compares LLVM IR byte-for-byte
- **GREEN** = identical (ready to merge)
- **SKIP** = both reject arch (expected for unsupported configs)
- **DRIFT** = divergence (porting bug!)

---

### Summary: Testing Workflow

| Stage | Tool | Purpose | Time | GPU? |
|-------|------|---------|------|------|
| **Write** | vim | Author kernel | - | No |
| **Quick check** | `test_my_kernel.py` | Correctness | ~10s | Yes |
| **Quick perf** | `bench_my_kernel.py` | TFLOPS measurement | ~30s | Yes |
| **Tune** | `sweep_hyperparams.py` | Find optimal config | ~5min | Yes |
| **Profile** | `rocprof` | Bottleneck analysis | ~1min | Yes |
| **Full test** | `tests/run_all.py --only <op>` | Regression check | ~2min | Yes |
| **Port to C++** | vim | Manually port | - | No |
| **Verify** | `tools/check_byte_identity.py` | IR equivalence | ~5min | No |
| **Perf parity** | `ROCKE_BACKEND=cpp bench_my_kernel.py` | C++ matches Python | ~30s | Yes |

**Key insight:** Start with standalone scripts (`test_my_kernel.py`, `bench_my_kernel.py`) for fast iteration. Run full test suite only when ready to validate.

**Remember:**
- **Production = C++ engine only** (Python never loaded)
- **ML training = Python heuristics/** (offline, generates models)
- **Kernel authoring = Python instances → C++ instances** (development workflow)
- **Python instances are for development; C++ instances ship in production**
- **Testing is comprehensive:** Correctness (pytest) + Performance (benchmark/) + Byte-Identity (tools/)
