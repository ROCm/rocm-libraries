# CK/rocKE Training Approach Reference

**Source**: CK (Composable Kernel) dispatcher heuristics  
**Location**: `projects/composablekernel/dispatcher/heuristics/`  
**Status**: Active - shared by both CK (AOT) and rocKE (C-JIT) kernel frameworks  

---

## CK vs rocKE

| Aspect | CK (Composable Kernel) | rocKE (ROCm Kernel Engine) |
|--------|------------------------|----------------------------|
| **Kernel compilation** | AOT (ahead-of-time) | C-JIT (runtime) |
| **Kernel definition** | C++ tile engine templates | Python specs → JIT compiled |
| **Location** | `projects/composablekernel/` | `hip-kernel-provider` (dev branches) |
| **Training tooling** | `dispatcher/heuristics/` | **Same tooling** (shared) |
| **Descriptor examples** | RFC 0017 worked examples | RFC 0017 SDPA examples |
| **Status** | Production | Active development (gfx942/gfx950 SDPA) |

**Key insight**: rocKE uses the **same training pipeline** as CK. The `generate_benchmark_data.py`, `feature_engine.py`, and `train.py` scripts are **shared infrastructure**.

---

## Overview

CK's training pipeline has three main components:

1. **Corpus Generation** — Problem space exploration (hardcoded per operation)
2. **Benchmark Runner** — Kernel enumeration + timing on hardware
3. **Training** — Feature engineering + LightGBM model training

---

## 1. Corpus Generation (`generate_benchmark_data.py`)

### Problem Space Strategy

**Stratified sampling within bounded parameter ranges**, organized by use-case regimes:

```python
def generate_shape_list():
    """Generate comprehensive (M, N, K) shapes covering all corner cases.
    
    Categories:
      1. M=1 (single token inference) -- the hardest case
      2. Tiny M (2-16) -- small batch inference
      3. Small M (32-128) -- medium batch
      4. Medium M (256-2048) -- large batch / training
      5. Large M (4096-20480) -- very large batch
      6. Square shapes (powers of 2)
      7. Skinny M, tall N (M << N)
      8. Tall M, skinny N (M >> N)
      9. Deep K (K >> M, N) -- compute-bound
     10. Shallow K (K << M, N) -- memory-bound
     11. Prime dimensions -- worst-case for tiling
     12. LLM-specific shapes (DeepSeek, LLaMA, etc.)
     13. Non-power-of-2 common sizes
    """
    shapes = set()
    
    # --- 1. M=1 (single token) across various N, K ---
    for n in [512, 1024, 1536, 2048, 3072, 4096, 4608, 7168, 8192, 11008, 14336, 28672]:
        for k in [256, 512, 1024, 1536, 2048, 2304, 4096, 7168, 8192]:
            shapes.add((1, n, k))
    
    # --- 2. Tiny M (2-16) ---
    for m in [2, 4, 8, 16]:
        for n in [512, 1536, 4096, 7168]:
            for k in [256, 1024, 4096, 7168]:
                shapes.add((m, n, k))
    
    # --- 9. Deep K (K >> M, N) ---
    for m in [16, 64, 256]:
        for n in [16, 64, 256]:
            for k in [4096, 8192, 16384, 32768]:
                shapes.add((m, n, k))
    
    # --- 11. Prime dimensions ---
    primes = [17, 31, 37, 127, 251, 509, 1021, 2039, 4093]
    for p in primes:
        shapes.add((p, p, p))
    for p in primes[:5]:
        shapes.add((p, 4096, 4096))
        shapes.add((4096, p, 4096))
        shapes.add((4096, 4096, p))
    
    # --- 12. LLM-specific shapes ---
    llm_shapes = [
        # DeepSeek MoE
        (1, 1536, 7168), (1, 4608, 7168), (1, 7168, 2048),
        # LLaMA-7B
        (1, 4096, 4096), (32, 4096, 4096), (1, 4096, 11008),
        # LLaMA-70B
        (1, 8192, 28672), (32, 8192, 28672),
    ]
    for s in llm_shapes:
        shapes.add(s)
    
    return sorted(shapes)
```

**Key characteristics**:
- **Stratified discrete sampling**: Not exhaustive enumeration, but targeted samples per regime
- **Regime-based**: 13 categories covering inference/training, compute/memory-bound, tiling pathologies
- **Bounded parameter space**: Each dimension (M/N/K) sampled from a specific set per regime
- **Model-aware**: LLM shapes (DeepSeek, LLaMA) as explicit samples, not the only source
- **Combinatorial within regimes**: Cross-products within bounded lists (e.g., M=1 × 12 N values × 9 K values)
- **~thousands of shapes**, not millions (via `set()` deduplication)

### Kernel Space Enumeration

**Reads kernel list from tile engine build**:

```python
def load_kernel_list(build_dir: Path, dtype: str, layout: str) -> List[KernelConfig]:
    """Load kernel configurations from the tile engine build."""
    kernel_list_path = (
        build_dir / "tile_engine" / "ops" / "gemm" / "gemm_universal" /
        dtype / layout / "gemm_universal_kernel_list.txt"
    )
    
    # Format: kernel_name|tile_config|trait_combo
    # Parses: tile_m, tile_n, tile_k, warp_m, warp_n, warp_k, 
    #         warp_tile_m, warp_tile_n, warp_tile_k,
    #         pipeline, scheduler, epilogue, pad_m, pad_n, pad_k, persistent
```

**Key characteristics**:
- **Explicit enumeration**: Reads compiled kernel catalog from text file
- **Not runtime discovery**: Requires ahead-of-time build of all kernel variants
- **Metadata extraction**: Parses kernel name to extract config params

### Benchmark Execution

**Direct kernel invocation** (not through autotune API):

```python
def run_benchmark(build_dir: Path, kernel: KernelConfig, problem: ProblemSize,
                  warmup: int = 10, repeat: int = 50) -> BenchmarkResult:
    """Run benchmark for a single kernel and problem size."""
    exe_path = build_dir / "bin" / f"benchmark_{kernel.name}"
    
    # Runs standalone benchmark executable per kernel
    # Output: avg_time_ms, tflops, is_valid
```

**Key characteristics**:
- **Kernel-specific executables**: Each kernel has its own `benchmark_<kernel>` binary
- **Direct timing**: Calls kernel directly, not through a dispatcher
- **Parallel build**: Uses `ProcessPoolExecutor` to build kernels in parallel
- **Sequential timing**: Times one (problem, kernel) pair at a time

---

## 2. Feature Engineering (`feature_engine.py`)

### Three-Space Decomposition (Same as UHD)

**Device space** — Hardware properties:
```python
class GemmUniversalFeatureEngine(FeatureEngine):
    def __init__(self,
                 num_cus: int = 256,
                 lds_capacity: int = 65536,
                 max_clock_mhz: int = 2400,
                 simds_per_cu: int = 4,
                 shader_engines: int = 32,
                 max_waves_per_cu: int = 32,
                 wavefront_size: int = 64,
                 l1_cache_kb: int = 32,
                 l2_cache_kb: int = 4096,
                 l3_cache_kb: int = 262144,
                 num_xcd: int = 8):
        # These become features via derived calculations
```

**Problem space** — Query dimensions:
- Raw: `M`, `N`, `K`, `split_k`, `layout`
- Derived: `log2_M`, `log2_N`, `log2_K`, `log2_MNK`
- Physics: `arithmetic_intensity`, `aspect_ratio_mn`, `aspect_ratio_mk`, `aspect_ratio_nk`

**Kernel space** — Tile/warp config:
- Tile dims: `tile_m`, `tile_n`, `tile_k`
- Warp config: `warp_m`, `warp_n`, `warp_k`, `warp_tile_m`, `warp_tile_n`, `warp_tile_k`
- Pipeline: `pipeline`, `scheduler`, `epilogue`, `persistent`
- Padding: `pad_m`, `pad_n`, `pad_k`

### Derived Interaction Features

**Tile quantization** (problem-to-tile ratio):
```python
"num_tiles_m": ceil(M / tile_m),
"num_tiles_n": ceil(N / tile_n),
"num_tiles_k": ceil(K / tile_k),
"tile_eff_m": (M % tile_m) / tile_m,  # Remainder efficiency
"tile_eff_n": (N % tile_n) / tile_n,
"tile_eff_k": (K % tile_k) / tile_k,
"overall_tile_efficiency": (tile_eff_m * tile_eff_n * tile_eff_k),
```

**Utilization metrics**:
```python
"cu_utilization": num_output_tiles / (num_cus * max_waves_per_cu),
"lds_usage_ratio": lds_usage_estimate / lds_capacity,
```

**Padding interaction** (does kernel support padding when needed):
```python
"needs_padding_m": (M % tile_m != 0),
"has_padding_when_needed_m": needs_padding_m and pad_m,
```

**Key characteristics**:
- **~60 features** for GEMM (raw + derived)
- **Categorical encoding**: Maps pipeline/scheduler/epilogue to integers
- **Physics-aware**: Arithmetic intensity, aspect ratios
- **Interaction-heavy**: Tile efficiency, utilization, padding match

---

## 3. Training (`train.py`)

### LightGBM with log1p Transform

```python
# Log-transform target for scale invariance
y = np.log1p(df[target_col].values)  # target = "measured_tflops"

# GroupKFold CV (prevent problem leakage)
groups = df.groupby(['M', 'N', 'K']).ngroup().values
folds = GroupKFold(n_splits=5).split(X, y, groups)

# Train
model = lgb.LGBMRegressor(
    objective="regression",
    metric=["rmse", "mae"],
    num_leaves=255,
    max_depth=15,
    n_estimators=2000,
    learning_rate=0.02,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.05,
    reg_lambda=0.5,
)
```

**Key characteristics**:
- **log1p(TFLOPS)** target: "GEMM TFLOPS spans 5 orders of magnitude (0.02 for M=1 to 2230 for large shapes). Raw regression optimizes for absolute RMSE, which means the model spends all its capacity predicting large shapes accurately and ignores tiny shapes where TFLOPS is < 10."
- **GroupKFold**: Prevents same (M, N, K) appearing in train and validation
- **Warm-start**: Supports incremental training from previous model
- **Feature schema validation**: Checks feature compatibility when warm-starting

### Outputs

```python
# Model files
output_dir/
  ├── model.lgbm          # LightGBM booster (native format)
  ├── feature_spec.json   # Feature names, categorical features, encoding
  ├── training_metrics.json  # RMSE, MAE, CV scores
  └── model_params.json   # Hyperparameters
```

---

## 4. Key Differences from UHD Approach

| Aspect | CK/rocKE | hipDNN UHD (RFC 0019) |
|--------|----------|------------------------|
| **Corpus generation** | Hardcoded problem lists | Formalized per-op methodology (Section 9) |
| **Kernel enumeration** | Text file from build | Runtime via KMD schema + catalog |
| **Benchmark substrate** | Direct kernel executables | hipDNN autotune API (RFC 0013) |
| **Feature contract** | Python FeatureEngine class | FlatBuffer `features_signature` + `features_hash` |
| **Model format** | LightGBM native (.lgbm) | FlatBuffer GbdtModel (RFC 0019 §7) |
| **Descriptor** | Python JSON config | FlatBuffer UHD (RFC 0019 §4) |
| **Integration** | C++ dispatcher loads .lgbm | UHD loader + TreeDataAdapter |
| **Reproducibility** | Hardcoded corpus in script | Corpus provenance via JSON (proposed) |

---

## 5. What hipDNN Can Adopt from CK

### ✅ Already adopted:

1. **log1p(TFLOPS) target transform** — Implemented in `uhd_gen/train_uhd.py`
2. **GroupKFold CV** — Implemented in `uhd_gen/train_uhd.py`
3. **LightGBM regressor** — Same choice
4. **Feature engineering patterns** — Tile quantization, arithmetic intensity

### 🔄 Should consider adopting:

1. **Hardcoded model-specific problem sizes** as **defaults**
   - CK's LLaMA-2/70B dims could seed GEMM smoke tier
   - Easier starting point than "write your own corpus generator"
   
2. **Feature importance analysis** in training output
   - CK tracks which features matter for model inspection
   
3. **Warm-start incremental training**
   - Useful for adding data without full retrain

### ❌ Should NOT adopt:

1. **Stratified sampling WITHOUT parameterization**
   - CK's 13 regimes are hardcoded in Python (not parameterized by tier/config)
   - hipDNN needs **parameterized** corpus generation (Section 9) where tier/device/op drive the sampling
   
2. **Kernel-specific benchmark executables**
   - CK builds one binary per kernel (slow, inflexible)
   - hipDNN autotune API is more flexible
   
3. **Text-file kernel enumeration**
   - CK's `gemm_universal_kernel_list.txt` is build-time only
   - hipDNN UKD catalog is runtime-queryable

---

## 6. Lessons for UHD Tooling

### From CK's strengths:

1. **End-to-end script** — `generate_benchmark_data.py` does corpus + benchmark + output in one invocation
   - hipDNN should have: `uhd_gen generate-corpus` + `uhd_gen benchmark` + `uhd_gen train` (Section 8)

2. **Parallel kernel builds** — CK uses `ProcessPoolExecutor` for compilation
   - hipDNN autotune could parallelize graph compilation

3. **Feature engineering as a first-class module** — `feature_engine.py` is reusable
   - hipDNN `features_signature` is inline in UHD, but tooling could have a `FeatureBuilder` helper

### From CK's gaps:

1. **No parameterized corpus generation** — CK's 13 regimes are hardcoded per script, not driven by tier/config
   - hipDNN formalized parameterization in Section 9 (regimes, alignment, roofline controlled by tier/device/op)
   
2. **No multi-arch strategy** — CK scripts assume single device
   - hipDNN should clarify: separate training runs per arch → arch-keyed UHDs
   
3. **No feature contract enforcement** — CK's `feature_spec.json` is advisory
   - hipDNN's `features_hash` is mandatory (RFC §6.3)

---

## References

- CK dispatcher heuristics: `projects/composablekernel/dispatcher/heuristics/`
- CK benchmark data generation: `generate_benchmark_data.py`
- CK feature engine: `feature_engine.py`
- CK training script: `train.py`
- hipDNN UHD training methodology: [RFC 0019.13 Training Methodology](../rfcs/0019.13_TrainingMethodology.md)
