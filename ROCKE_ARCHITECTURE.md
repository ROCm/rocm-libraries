# rocKE Architecture Guide for New Developers

**rocKE** (ROCm Kernel Engine) is a dual-engine (Python + C++) kernel authoring system for AMD GPUs, integrated into hipDNN as the high-performance kernel provider.

## High-Level Architecture

```
+------------------------------------------------------------------+
|  PROJECTS/HIPDNN                                                 |
|  +------------------------------------------------------------+  |
|  |  hipDNN Backend (backend/src/)                             |  |
|  |  - Graph operations (SDPA, Conv, GEMM, MoE, Norm)          |  |
|  |  - Operator dispatch                                       |  |
|  |  - Memory management                                       |  |
|  +------------------------------------------------------------+  |
+------------------------------------------------------------------+
                           |
                           | Calls kernel provider APIs
                           v
+------------------------------------------------------------------+
|  DNN-PROVIDERS/HIP-KERNEL-PROVIDER/ROCKE                         |
|  +------------------------------------------------------------+  |
|  |  Python/rocke/                                             |  |
|  |  +------------------------------------------------------+  |  |
|  |  |  dispatch/families/                                  |  |  |
|  |  |    attention.py <-- SDPA dispatcher (MOST MATURE)    |  |  |
|  |  |    conv.py, gemm.py, moe.py, norm.py                 |  |  |
|  |  +------------------------------------------------------+  |  |
|  |  +------------------------------------------------------+  |  |
|  |  |  heuristics/ <-- ML Training Pipeline                |  |  |
|  |  |    train.py - LightGBM model training                |  |  |
|  |  |    predict.py - Kernel selection predictions         |  |  |
|  |  |    feature_engine.py - 55-feature extraction         |  |  |
|  |  |    gen_gemm_sweep_data.py - Generate training data   |  |  |
|  |  |    No pre-trained models - train your own            |  |  |
|  |  |    Output: model_tflops.lgbm, feature_spec.json      |  |  |
|  |  +------------------------------------------------------+  |  |
|  |  +------------------------------------------------------+  |  |
|  |  |  instances/common/                                   |  |  |
|  |  |    attention_unified.py <-- SDPA kernels (Python)    |  |  |
|  |  |    gemm_universal.py, conv, moe, norm                |  |  |
|  |  +------------------------------------------------------+  |  |
|  |  +------------------------------------------------------+  |  |
|  |  |  core/                                               |  |  |
|  |  |    lower_llvm.py - LLVM IR generation                |  |  |
|  |  |    Type system & verification                        |  |  |
|  |  +------------------------------------------------------+  |  |
|  |  +------------------------------------------------------+  |  |
|  |  |  runtime/                                            |  |  |
|  |  |    hip_module.py - HIP kernel launch                 |  |  |
|  |  |    comgr compilation - .ll -> HSACO                  |  |  |
|  |  +------------------------------------------------------+  |  |
|  +------------------------------------------------------------+  |
|  +------------------------------------------------------------+  |
|  |  Cpp/                                                      |  |
|  |  +------------------------------------------------------+  |  |
|  |  |  include/ckc/ - Public C ABI                         |  |  |
|  |  |    extern "C" kernel entry points                    |  |  |
|  |  +------------------------------------------------------+  |  |
|  |  +------------------------------------------------------+  |  |
|  |  |  bindings/                                           |  |  |
|  |  |    rocke_engine.cpp - Main pybind module             |  |  |
|  |  |    rocke_engine_attention.cpp <-- SDPA C++ bindings  |  |  |
|  |  |    rocke_engine_fmha_extra.cpp                       |  |  |
|  |  |    rocke_engine_moe_gemm_fused.cpp                   |  |  |
|  |  |    (Pybind11 Python/C++ bridge)                      |  |  |
|  |  +------------------------------------------------------+  |  |
|  |  +------------------------------------------------------+  |  |
|  |  |  instances/gfx950/ <-- Arch-specific kernels         |  |  |
|  |  |    attention_tiled_2d_*.cpp <-- SDPA kernels (C++)   |  |  |
|  |  |    attention_tiled_3d_*.cpp                          |  |  |
|  |  |    deep_fused_conv_pool_*.cpp                        |  |  |
|  |  |    gemm, moe variants                                |  |  |
|  |  |    (Mirrors Python, byte-identical LLVM IR)          |  |  |
|  |  +------------------------------------------------------+  |  |
|  |  +------------------------------------------------------+  |  |
|  |  |  core/helpers/support/                               |  |  |
|  |  |    C++ lowering engine (mirrors Python)              |  |  |
|  |  |    Byte-identical LLVM IR generation                 |  |  |
|  |  +------------------------------------------------------+  |  |
|  |                                                            |  |
|  |  Build Output:                                             |  |
|  |    librocke_core.a - Static C++ engine library             |  |
|  |    rocke_engine.so - Python binding (optional)             |  |
|  +------------------------------------------------------------+  |
+------------------------------------------------------------------+
                           |
                           | Generates
                           v
+------------------------------------------------------------------+
|  LLVM IR -> HSACO Pipeline                                       |
|  +------------------------------------------------------------+  |
|  |  KernelDef (SSA) -> lower_llvm -> .ll -> libamd_comgr      |  |
|  |                  -> HSACO -> GPU                           |  |
|  |                                                            |  |
|  |  GPU Execution (via HIP)                                   |  |
|  |  - Kernel launch with grid/block configuration             |  |
|  |  - Target archs: gfx942, gfx950, gfx1151, gfx1201          |  |
|  +------------------------------------------------------------+  |
+------------------------------------------------------------------+
```

## Component Deep Dive

### 1. **Dispatch Layer** (`Python/rocke/dispatch/`)

**Purpose:** Operator-to-kernel selection with ML-based heuristics

**Key Files:**
- `core.py` - Operator-agnostic dispatch contracts
- `families/attention.py` - **SDPA dispatcher** (most mature)
  - Path selection: 2D-tiled vs 3D split-KV kernels
  - Backend coverage gating
  - Request normalization
- `families/conv.py`, `gemm.py`, `moe.py`, `norm.py` - Other operators

**How it Works:**
```python
# Example SDPA dispatch flow:
AttentionRequest(batch, nhead_q, nhead_k, seqlen_q, seqlen_k, ...) 
  → select_path() → "2d" or "3d"
  → rank candidates by heuristics/ML
  → DispatchResult(kernel_id, grid, block)
```

### 2. **Heuristics Layer** (`Python/rocke/heuristics/`)

**Purpose:** ML-based kernel selection using LightGBM models

**🎯 This is where the ML-based selection happens!**

**Key Files:**
- `train.py` - Train LightGBM regressors
- `predict.py` - Kernel selection predictor
- `feature_engine.py` - Extract 55 features:
  - Problem: M, N, K, aspect ratios, arithmetic intensity (13)
  - Kernel: tile sizes, warps, pipeline, LDS usage (17)
  - Interaction: tile efficiency, CU utilization (9)
  - Hardware: CUs, cache, clock speed (12)
- `gen_gemm_sweep_data.py` - Generate training data

**Model Files (NOT included in repo):**
- `model_tflops.lgbm` - Primary ranking model
- `model_latency.lgbm` - Latency-sensitive workloads
- `model_bandwidth.lgbm` - Memory-bound analysis
- `feature_spec.json` - Feature layout contract

**NOTE:** No pre-trained models are provided. Users must:
1. Generate benchmark data via `gen_gemm_sweep_data.py`
2. Train models via `train.py`
3. The heuristics README documents evaluation results from internal training runs

### 3. **Instance Layer** (Kernel Implementations)

**Python:** `Python/rocke/instances/`
**C++:** `Cpp/instances/`

**Structure (Separate Directory Trees):**

Python Instance Tree:
```
Python/rocke/instances/
├── common/
│   ├── attention_unified.py         ◄─── SDPA (Python authoring)
│   ├── gemm_universal.py
│   └── ... (conv, moe, norm)
├── gfx942/
├── gfx950/
│   └── (arch-specific Python variants)
├── gfx1151/
└── gfx1201/
```

C++ Instance Tree (Manual Port):
```
Cpp/instances/
├── gfx942/
├── gfx950/
│   ├── attention_tiled_2d_*.cpp     ◄─── SDPA (C++ production)
│   ├── attention_tiled_3d_*.cpp
│   ├── deep_fused_conv_pool_*.cpp
│   └── ... (gemm, moe variants)
├── gfx1151/
└── gfx1201/
```

#### Python vs C++ Instances: The Dual-Engine Design

**Critical Concept:** rocKE has **two manually-maintained, byte-identical engines**:

**Python Instances** (`Python/rocke/instances/`):
- **Purpose:** Authoring frontend and differential oracle
- **Role:** Where you *write* and *prototype* kernels
- **Files:** 58 Python files (e.g., `attention_unified.py` = 4312 lines)
- **What they do:**
  - Define problem specs (`UnifiedAttentionProblem`)
  - Implement heuristics (e.g., `select_path()` → "2d" vs "3d")
  - Build SSA IR: `build_unified_attention_2d(spec) -> KernelDef`
  - High-level orchestration and dispatch logic
- **Output:** `KernelDef` object (SSA IR) that gets lowered to LLVM
- **Runtime:** Optional (only if `rocke_engine.so` is built)

**C++ Instances** (`Cpp/instances/`):
- **Purpose:** Production runtime engine (ships in hipDNN)
- **Role:** What *runs* in production with **zero Python dependency**
- **Files:** 16+ C++ files, often chunked (e.g., 13 files totaling 6387 lines for attention)
- **What they do:**
  - Manually hand-ported from Python (not code-generated!)
  - Same logic, but split into compilation units for build hygiene
  - Direct C/C++ implementation of the Python builders
- **Output:** Identical `KernelDef` SSA IR (byte-for-byte identical LLVM IR)
- **Runtime:** **Always used** (statically linked as `librocke_core.a`)

**🔑 Key Points:**

1. **Manual Hand-Porting (Not Generated):**
   - C++ code is explicitly documented as a "C99 port" of Python
   - Example: `attention_tiled_2d_ctx_prologue.cpp` header states:
     ```cpp
     // the BUILD-CONTEXT PROLOGUE bucket of the chunked C99 port of
     // rocke/instances/gfx950/attention_tiled_2d.py (arch gfx950, WIDE-K atoms).
     // (Python build body lines 711-1135)
     ```
   - No code generation tooling exists in the repo
   - Every Python change **must** be manually mirrored in C++

2. **Byte-Identity Requirement:**
   - Both engines **must emit identical LLVM IR** (verified by `tools/check_byte_identity.py`)
   - Any optimization in Python without C++ equivalent → incomplete change
   - Definition of "done" = both engines pass differential gate

3. **Why Two Engines?**
   - **Python advantages:** Fast iteration, rich ecosystem, easier debugging, ML integration
   - **C++ advantages:** No Python at runtime, ships as static lib, production-ready
   - **Together:** Prototype in Python, ship in C++, guarantee equivalence

4. **File Organization Differences:**
   - Python: One large file per operator (e.g., 4455 lines)
   - C++ : Chunked into phases for compilation units (e.g., 13 files)
     - `ctx_prologue.cpp` - Initialization (lines 711-1135 of Python)
     - `kv_body_qk_softmax.cpp` - QK computation
     - `kv_body_pv_epilogue.cpp` - Output
     - Plus arch-specific variants (fastkv, fp8, 3D split-KV, etc.)

5. **Common Pitfalls (from `invariants.md`):**
   - C++ evaluates function args **right-to-left**, Python **left-to-right**
   - Must bind operands explicitly to preserve SSA order
   - Closures in Python → explicit `ctx` structs in C++

**Workflow:**
1. Author kernel in **Python** (`instances/gfx950/attention_tiled_2d.py`)
2. Manually port to **C++** (create `attention_tiled_2d_*.cpp` files)
3. Run `tools/check_byte_identity.py` → verify identical LLVM IR
4. Iterate until byte-identical (fix divergences by hand)
5. Code review enforces: both engines change together

**SDPA Example Flow:**
1. `attention_unified.py` - High-level problem definition
2. Architecture selection: 2D vs 3D kernel path
3. `gfx950/attention_tiled_2d_*.cpp` - Actual kernel generation (manually ported)
4. Kernel variants:
   - `ctx_prologue` - Context initialization
   - `kv_body_qk_softmax` - QK^T + softmax computation
   - `kv_body_pv_epilogue` - Output projection
   - `fastkv_regp` - Fast KV cache handling
   - `fp8_kv` - FP8 quantized KV cache

### 4. **Core Engine** (`Python/rocke/core/` & `Cpp/core/`)

**Purpose:** Kernel definition → LLVM IR lowering

**Pipeline:**
```
Spec → build_*() → KernelDef (SSA) → lower_kernel_to_llvm() → .ll
```

**Key Components:**
- Type system & verification
- SSA construction
- LLVM IR emission
- **Byte-identical guarantee**: Python and C++ engines produce identical output

### 5. **Runtime Layer** (`Python/rocke/runtime/`)

**Purpose:** Compilation and GPU execution

**Components:**
- `hip_module.py` - HIP kernel launch
- `comgr` integration - LLVM IR → HSACO compilation
- Device query (`get_device_arch()`)

### 6. **C++ Bindings** (`Cpp/bindings/`)

**Purpose:** Expose C++ engine to Python (optional optimization)

**Key Files:**
- `rocke_engine.cpp` - Main pybind11 module
- `rocke_engine_attention.cpp` - **SDPA bindings**
- `rocke_engine_fmha_extra.cpp` - Additional FMHA variants

**Build Output:**
- `librocke_core.a` - Static library
- `rocke_engine.so` - Python module (if built)

**Fallback:** If `rocke_engine` not available, transparently uses Python engine

## SDPA (Scaled Dot-Product Attention)

### Architecture Selection

```
SDPA Problem
  ├─ head_size support check
  ├─ block_size validation
  └─ Path selection:
      ├─ 2D-tiled kernel (per-(kv_head, q_block) CTA)
      │   └─ gfx950/attention_tiled_2d_*.cpp
      │       ├─ ctx_prologue
      │       ├─ kv_body_qk_softmax
      │       ├─ kv_body_pv_epilogue
      │       └─ fastkv_regp (optimized KV cache)
      └─ 3D split-KV kernel
          └─ gfx950/attention_tiled_3d_*.cpp
              ├─ segment_prologue
              ├─ segment_loop_epilogue
              └─ reduce
```

### SDPA Dispatch Flow

1. **hipDNN** calls SDPA operation
2. **Dispatch** (`dispatch/families/attention.py`):
   - Normalize `AttentionRequest`
   - Call `select_path()` → "2d" or "3d"
   - Apply heuristics/ML model (if available)
3. **Instance** (`instances/common/attention_unified.py`):
   - Build `KernelDef` with selected path
   - Determine CTA geometry (num_warps, block_m, tile_size)
4. **Core** (`core/lower_llvm.py`):
   - Lower to LLVM IR
5. **Runtime**:
   - Compile via `libamd_comgr`
   - Launch on GPU

### SDPA Kernel Variants

| Variant | Purpose | File |
|---------|---------|------|
| 2D Tiled | Standard attention, one CTA per (kv_head, q_block) | `attention_tiled_2d_public_entry_glue.cpp` |
| 2D FastKV RegP | Optimized register-packed KV cache | `attention_tiled_2d_fastkv_regp.cpp` |
| 2D FP8 KV | FP8 quantized KV cache | `attention_tiled_2d_fp8_kv.cpp` |
| 3D Split-KV | Large sequence lengths, split over K dim | `attention_tiled_3d_public_glue.cpp` |
| 3D Reduce | Reduction kernel for 3D split | `attention_tiled_3d_reduce.cpp` |

## How hipDNN Calls rocKE

```
projects/hipdnn/backend/
  └─ (Operation execution)
      └─ Kernel provider API calls
          └─ rocKE dispatch interface
              ├─ Direct C++ path: include/ckc/*.h
              └─ Python path: import rocke_engine (optional)
```

**Note:** The exact integration point is in `projects/hipdnn/backend/src/` but uses the rocKE public APIs exposed through the kernel provider interface.

## Build & Test

### Build rocKE Engine

```bash
export ROCKE=$PWD/dnn-providers/hip-kernel-provider/rocKE
export PYTHONPATH=$ROCKE/Python

# Build C++ engine
cmake -S $ROCKE -B /tmp/rocke -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/rocke --target rocke_core -j$(nproc)
# → /tmp/rocke/librocke_core.a

# Build Python bindings (optional)
cmake -S $ROCKE/Cpp/bindings -B /tmp/rocke_pybind \
  -DROCKE_ENGINE_ARCHIVE=/tmp/rocke/librocke_core.a \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build /tmp/rocke_pybind -j$(nproc)
```

### Test

```bash
# All tests (guard + byte-identity + pytest + ctest)
python $ROCKE/tests/run_all.py

# Byte-identity check (Python ≡ C++ output)
python $ROCKE/tools/check_byte_identity.py

# Python tests only
python -m pytest $ROCKE/tests
```

## Key Concepts for New Developers

1. **Dual Engine**: Python for authoring/prototyping, C++ for production runtime (manually maintained, not code-generated)
2. **Byte-Identical**: Both engines must produce identical LLVM IR (verified by `tools/check_byte_identity.py`)
3. **Dispatch → Heuristics → Instance → Core → Runtime** flow
4. **ML Training Pipeline**: Framework exists to train LightGBM models for kernel selection (no pre-trained models included)
5. **SDPA is Most Mature**: Best starting point for understanding the system
6. **Arch-Specific Instances**: Per-GPU optimizations in `instances/gfx*/`

## Documentation References

- `ROCKE/README.md` - Main overview
- `ROCKE/AGENTS.md` - Agent/onboarding notes
- `ROCKE/BUILD.md` - Build & artifact hygiene
- `ROCKE/tests/README.md` - Test tree map
- `ROCKE/dsl_docs/` - Architecture, runtime, development docs
- `ROCKE/Python/rocke/heuristics/README.md` - ML model details
- `ROCKE/Python/rocke/dispatch/README.md` - Dispatch layer guide

---

**TL;DR for New Developers:**

1. **Start with SDPA** - Read `dispatch/families/attention.py` and `instances/common/attention_unified.py`
2. **Understand dispatch flow** - Request → Heuristics → Instance → Kernel
3. **ML training pipeline in `heuristics/`** - Train your own kernel selection models (none pre-trained)
4. **Python vs C++ instances** - NOT auto-generated; manually hand-ported with byte-identity requirement:
   - Python (`Python/rocke/instances/`) = authoring frontend, single large files
   - C++ (`Cpp/instances/`) = production runtime, chunked into compilation units
   - **Critical:** Every change needs both engines updated (`tools/check_byte_identity.py`)
5. **Instances in `instances/gfx950/attention_tiled_2d_*.cpp`** - Actual kernel code (C++ port)
6. **hipDNN calls rocKE** - Through kernel provider APIs in `backend/`, uses C++ engine (`librocke_core.a`)
