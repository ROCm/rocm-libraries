# Tutorial: Train Your First UHD

**Companion to:** RFC 0019.13 Training Methodology

**Purpose:** Concrete, soup-to-nuts walkthrough for training a Universal Heuristic Descriptor (UHD) for an existing hipDNN engine.

---

## Prerequisites

### Required Knowledge
- You have benchmark timing data in CSV format
- You know which hipDNN engine you want to train a heuristic for
- You understand the operation's parameter space (e.g., GEMM has M/N/K dims)

### Software Requirements
```bash
# 1. hipDNN built and installed
cd /path/to/rocm-libraries/projects/hipdnn
mkdir -p build && cd build
cmake -GNinja .. && ninja && ninja install

# 2. Install uhd_gen tooling
cd /path/to/rocm-libraries/projects/hipdnn/tools/uhd_gen
pip install -e .

# 3. Verify installation
python -m uhd_gen --help
```

---

## Step 1: Discover Available Engines

### Option A: Use ListEngines Tool

```bash
# From build directory
cd /path/to/rocm-libraries/projects/hipdnn/build
./bin/ListEngines
```

**Example output:**
```
Loaded engines:
  MIOpen Convolution Forward (0x00000001ABCD0001)
    Plugin:  miopen-provider
    Version: 1.0.0
    Type:    CONV_FWD

  hipBLASLt GEMM (0x00000002BEEF0001)
    Plugin:  hipblaslt-provider
    Version: 1.0.0
    Type:    MATMUL
```

### Option B: Check Provider Source

Engines are registered in provider plugins:
- MIOpen: `/path/to/dnn-providers/miopen-provider/engines/`
- hipBLASLt: `/path/to/dnn-providers/hipblaslt-provider/engines/`
- HipKernel: `/path/to/dnn-providers/hip-kernel-provider/engines/`

---

## Step 2: Understand the Missing Pieces

**Currently MISSING from hipDNN codebase:**

### 2.1 Benchmark Corpus Generation
- **Status:** NOT IMPLEMENTED
- **What's needed:** Per-operation corpus generators (RFC 0019.13 §7.1)
- **Workaround:** Manually create CSV with representative problem shapes

### 2.2 Autotune Integration
- **Status:** NOT IMPLEMENTED (RFC 0013 specifies the API, but orchestration tooling doesn't exist)
- **What's needed:** `uhd_gen benchmark` subcommand
- **Workaround:** Manually benchmark and create CSV

### 2.3 Applicability Predicates
- **Status:** Providers like MIOpen have `isApplicable()` but no generic interface exposed to tooling
- **What's needed:** Generic `KernelProvider.is_applicable()` interface (RFC 0019.13 §7.3)
- **Workaround:** Pre-filter problem shapes manually based on provider documentation

---

## Step 3: Prepare Benchmark Data (Manual Workaround)

Since autotune tooling doesn't exist yet, you need to create benchmark CSV manually.

### Required CSV Schema

```csv
q.M,q.N,q.K,kernel.tile_m,kernel.tile_n,kernel.tile_k,device.cu_count,tflops
1024,1024,1024,128,128,32,120,50.5
2048,2048,2048,256,128,32,120,75.2
4096,4096,4096,256,256,32,120,102.3
...
```

**Critical Requirements:**
1. **Namespace-qualified columns:** Every feature MUST start with `q.`, `kernel.`, or `device.`
2. **Target column:** Throughput (`tflops`) or cost (`latency_ms`)
3. **Problem features (`q.*`):** Operation-specific dims (M/N/K for GEMM, batch/heads/seqlen for SDPA)
4. **Kernel features (`kernel.*`):** Tile sizes, split factors, wavefront config from KMD metadata
5. **Device features (`device.*`):** GPU properties (cu_count, lds_size, etc.)

### Example: Generate GEMM Benchmark Corpus

```python
import pandas as pd
import numpy as np

# 1. Generate problem shapes (CK-style regimes, RFC 0019.13 §7.1)
problems = []

# Regime 1: Powers of 2
for p in [6, 8, 10, 12]:
    dim = 2**p
    problems.append({"q.M": dim, "q.N": dim, "q.K": dim})

# Regime 2: LLM shapes
problems.extend([
    {"q.M": 1, "q.N": 4096, "q.K": 4096},     # Single token
    {"q.M": 32, "q.N": 4096, "q.K": 4096},    # Batch
    {"q.M": 1, "q.N": 4096, "q.K": 11008},    # LLaMA MLP
])

# 2. Generate kernel variants (from KMD schema)
kernel_variants = [
    {"kernel.tile_m": 64, "kernel.tile_n": 64, "kernel.tile_k": 8},
    {"kernel.tile_m": 128, "kernel.tile_n": 128, "kernel.tile_k": 16},
    {"kernel.tile_m": 256, "kernel.tile_n": 128, "kernel.tile_k": 32},
]

# 3. Cartesian product
rows = []
for problem in problems:
    for variant in kernel_variants:
        row = {**problem, **variant}
        row["device.cu_count"] = 120  # Your GPU's CU count
        
        # 4. PLACEHOLDER: You need actual benchmark timing here
        # In real workflow, this would call hipdnnBackendAutotune()
        row["tflops"] = np.random.uniform(30, 120)  # FAKE DATA
        
        rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("gemm_benchmark.csv", index=False)
print(f"Generated {len(df)} benchmark rows")
```

**⚠️ CRITICAL:** The example above uses FAKE timing data. In production, you must:
1. Build graph for each problem
2. Enroll each kernel variant
3. Run autotune API to get actual timing
4. Extract `minTimeMs` or `tflops` from autotune results

---

## Step 4: Train the UHD Model

Once you have benchmark CSV, use `uhd_gen` to train:

```bash
python -m uhd_gen \
    --input gemm_benchmark.csv \
    --features q.M q.N q.K kernel.tile_m kernel.tile_n kernel.tile_k device.cu_count \
    --target tflops \
    --objective max \
    --group-by q.M q.N q.K \
    --output-dir ./gemm_uhd_output \
    --name "GEMM UHD gfx942" \
    --training-arches gfx942 \
    --model-version 1.0.0
```

### Arguments Explained

| Argument | Value | Why |
|----------|-------|-----|
| `--input` | `gemm_benchmark.csv` | Your benchmark data |
| `--features` | `q.M q.N q.K kernel.tile_m ...` | **MUST be namespace-qualified** |
| `--target` | `tflops` | Column to predict (throughput) |
| `--objective` | `max` | Higher tflops = better (use `min` for latency_ms) |
| `--group-by` | `q.M q.N q.K` | Prevents problem leakage in CV |
| `--output-dir` | `./gemm_uhd_output` | Where to write artifacts |
| `--name` | `"GEMM UHD gfx942"` | Human-readable name |
| `--training-arches` | `gfx942` | GPU arch (for OOD detection) |
| `--model-version` | `1.0.0` | Semantic version |

**Output:**
```
gemm_uhd_output/
├── uhd.fb              # FlatBuffer UHD descriptor (RUNTIME FORMAT)
├── uhd.json            # Human-readable UHD descriptor
├── model.bin           # FlatBuffer GbdtModel
└── train_manifest.json # Training metadata
```

---

## Step 5: Inspect the Trained Model

### Check uhd.json

```bash
cat gemm_uhd_output/uhd.json | jq .
```

**Example output:**
```json
{
  "schema": "hipdnn.uhd/v1",
  "id": "a1b2c3d4-...",
  "name": "GEMM UHD gfx942",
  "adapter": "tree_data",
  "features_signature": [
    "$q.M",
    "$q.N",
    "$q.K",
    "$kernel.tile_m",
    "$kernel.tile_n",
    "$kernel.tile_k",
    "$device.cu_count"
  ],
  "features_hash": "sha256:abc123...",
  "objective": "max",
  "score": {
    "units": "tflops",
    "calibrated": false,
    "transform": "log1p"
  },
  "model": {
    "artifact": "model.bin"
  }
}
```

### Check Training Manifest

```bash
cat gemm_uhd_output/train_manifest.json | jq .
```

**Shows:**
- Number of trees in model
- Number of training samples
- Features hash
- Training architectures

---

## Step 6: Deploy the UHD (NOT YET IMPLEMENTED)

**Currently MISSING:** Integration tooling to install UHD into engine descriptor set.

**What should exist (RFC 0019 §13.4):**
```bash
# HYPOTHETICAL — does not exist yet
python -m uhd_gen install \
    --uhd-dir ./gemm_uhd_output \
    --engine-id 0x00000002BEEF0001 \
    --target-dir /opt/rocm/share/hipdnn/engines/
```

**Manual workaround:**
1. Copy `uhd.fb` and `model.bin` to engine descriptor location
2. Update UED (Universal Engine Descriptor) to reference the new UHD
3. Restart application to reload descriptors

**Where engine descriptors live:**
- Installed: `/opt/rocm/share/hipdnn/engines/` (or `CMAKE_INSTALL_PREFIX/share/hipdnn/engines/`)
- Source: `dnn-providers/<provider-name>/descriptors/`

---

## Step 7: Test the UHD (NOT YET IMPLEMENTED)

**Currently MISSING:** Validation tooling.

**What should exist:**
```bash
# HYPOTHETICAL — does not exist yet
python -m uhd_gen validate \
    --uhd-dir ./gemm_uhd_output \
    --test-csv gemm_test_set.csv
```

**Manual workaround:**
1. Write C++ test using `UhdLoader::load("uhd.fb")`
2. Create feature vectors matching training schema
3. Call `TreeDataAdapter::score()` and verify predictions make sense

---

## What's Actually Implemented vs. Missing

| Component | Status | Location |
|-----------|--------|----------|
| **UHD Training Tool** | ✅ IMPLEMENTED | `tools/uhd_gen/__main__.py` |
| **LightGBM → FlatBuffer** | ✅ IMPLEMENTED | `tools/uhd_gen/lgbm_to_flatbuffer.py` |
| **UHD Loader (runtime)** | ✅ IMPLEMENTED | `backend/src/heuristics/uhd/UhdLoader.cpp` |
| **TreeDataAdapter (runtime)** | ✅ IMPLEMENTED | `backend/src/heuristics/uhd/adapters/TreeDataAdapter.cpp` |
| **Corpus Generation** | ❌ NOT IMPLEMENTED | RFC 0019.13 §7.1 specifies pattern |
| **Autotune Integration** | ❌ NOT IMPLEMENTED | RFC 0013 specifies API, tooling missing |
| **Applicability Interface** | ⚠️ PARTIAL | Providers have it, not exposed to tooling |
| **Install/Deploy Tooling** | ❌ NOT IMPLEMENTED | Manual copy workaround |
| **Validation Tooling** | ❌ NOT IMPLEMENTED | Manual C++ test workaround |

---

## Detailed Specifications for Missing Components

This section provides complete specifications so implementation is mechanical.

### Spec 1: `uhd_gen generate-corpus` Command

**Purpose:** Generate representative problem shapes for an operation using stratified sampling.

**CLI Signature:**
```bash
python -m uhd_gen generate-corpus \
    --operation <op_name> \
    --tier <smoke|standard|exhaustive> \
    [--config <yaml_path>] \
    [--output <path>] \
    [--format <json|csv>]
```

**Arguments:**

| Argument | Required | Type | Default | Description |
|----------|----------|------|---------|-------------|
| `--operation` | Yes | str | - | Operation name: `gemm`, `sdpa_forward`, `conv_fwd`, etc. |
| `--tier` | Yes | str | - | Sampling density: `smoke` (10-20 shapes), `standard` (100-500), `exhaustive` (1000+) |
| `--config` | No | path | - | YAML file with custom regime overrides |
| `--output` | No | path | `corpus.json` | Output file path |
| `--format` | No | str | `json` | Output format: `json` or `csv` |

**Output Schema (JSON):**
```json
{
  "metadata": {
    "operation": "gemm",
    "tier": "standard",
    "generator_version": "1.0.0",
    "generated_at": "2026-08-21T10:30:00Z",
    "count": 320,
    "regimes": {
      "powers_of_2": 8,
      "llm_shapes": 45,
      "compute_bound": 60,
      "memory_bound": 30,
      "primes": 12
    }
  },
  "problems": [
    {
      "q.M": 1024,
      "q.N": 1024,
      "q.K": 1024,
      "metadata": {
        "regime": "powers_of_2",
        "index": 0
      }
    },
    ...
  ]
}
```

**Output Schema (CSV):**
```csv
q.M,q.N,q.K,_regime,_index
1024,1024,1024,powers_of_2,0
2048,2048,2048,powers_of_2,1
...
```

**Config YAML Schema:**
```yaml
# Override default regimes for GEMM operation
operation: gemm
tier: standard  # Base tier to start from

# Override specific regimes
regimes:
  powers_of_2:
    enabled: true
    exponents: [6, 8, 10, 12, 13]  # 64, 256, 1024, 4096, 8192
  
  llm_shapes:
    enabled: true
    configs:
      - {M: 1, N: 4096, K: 4096, name: "llama_single_token"}
      - {M: 32, N: 4096, K: 4096, name: "llama_batch"}
      - {M: 1, N: 4096, K: 11008, name: "llama_mlp_up"}
  
  compute_bound:
    enabled: true
    m_values: [64, 256]
    n_values: [64, 256]
    k_values: [4096, 8192, 16384]  # Deep K
  
  memory_bound:
    enabled: true
    m_values: [4096, 8192]
    n_values: [4096, 8192]
    k_values: [128, 256]  # Shallow K
  
  primes:
    enabled: true
    values: [17, 31, 127, 251, 509, 1021, 2039]

# Add custom regime (extension point)
custom_regimes:
  my_special_shapes:
    - {M: 999, N: 888, K: 777}
```

**Implementation Contract:**

1. **Registry pattern** — operation generators registered by name:
   ```python
   from abc import ABC, abstractmethod
   
   class CorpusGenerator(ABC):
       @abstractmethod
       def generate(self, tier: str, config: dict | None) -> list[dict]:
           """Generate problem corpus. Returns list of problem dicts."""
           pass
   
   # Registry
   GENERATORS = {
       "gemm": GemmCorpusGenerator(),
       "sdpa_forward": SdpaCorpusGenerator(),
       "conv_fwd": ConvFwdCorpusGenerator(),
   }
   ```

2. **Namespace consistency** — all problem parameters MUST be `q.*` prefixed
3. **Metadata tracking** — each problem annotated with `regime` and `index` for provenance
4. **Deduplication** — use set-based dedup on tuple of all `q.*` values
5. **Deterministic** — same inputs produce same output (for reproducibility)

---

### Spec 2: `uhd_gen benchmark` Command

**Purpose:** Time kernel variants across problem corpus using hipDNN autotune API.

**CLI Signature:**
```bash
python -m uhd_gen benchmark \
    --corpus <corpus_path> \
    --engine-id <engine_id_hex> \
    [--variants <variants_json>] \
    [--device <device_idx>] \
    [--output <path>] \
    [--warmup <int>] \
    [--repeat <int>]
```

**Arguments:**

| Argument | Required | Type | Default | Description |
|----------|----------|------|---------|-------------|
| `--corpus` | Yes | path | - | Corpus JSON from `generate-corpus` |
| `--engine-id` | Yes | str | - | Engine ID in hex (e.g., `0x00000002BEEF0001`) |
| `--variants` | No | path | - | Explicit variant list JSON (XOR with auto-enumeration) |
| `--device` | No | int | `0` | HIP device index |
| `--output` | path | path | `benchmark_results.csv` | Output CSV path |
| `--warmup` | No | int | `10` | Warmup iterations per variant |
| `--repeat` | No | int | `50` | Timing iterations per variant |

**Variants JSON Schema** (optional explicit enumeration):
```json
{
  "variants": [
    {
      "kernel.tile_m": 128,
      "kernel.tile_n": 128,
      "kernel.tile_k": 8,
      "kernel.use_lds": true
    },
    ...
  ]
}
```

**Output CSV Schema:**
```csv
q.M,q.N,q.K,kernel.tile_m,kernel.tile_n,kernel.tile_k,device.cu_count,minTimeMs,avgTimeMs,stddevMs,tflops,is_valid,error
1024,1024,1024,128,128,8,120,0.123,0.125,0.002,50.5,True,
2048,2048,2048,256,128,16,120,0.456,0.460,0.005,75.2,True,
8192,8192,8192,128,128,8,120,0,0,0,0,False,timeout
```

**Required Columns:**
- All `q.*` from corpus
- All `kernel.*` from variant enumeration
- All `device.*` queried from HIP runtime
- `minTimeMs`, `avgTimeMs`, `stddevMs` — autotune timing results
- `tflops` — derived metric (operation-specific FLOP formula)
- `is_valid` — `True` if benchmark succeeded, `False` if failed/timeout
- `error` — error message if `is_valid=False`, empty otherwise

**Implementation Contract:**

1. **Graph construction** — operation-specific graph builders:
   ```python
   def build_gemm_graph(problem: dict) -> hipdnn_graph:
       """Build hipDNN graph from problem parameters."""
       # Create matmul node with dims from problem["q.M"], problem["q.N"], problem["q.K"]
       pass
   
   GRAPH_BUILDERS = {
       "gemm": build_gemm_graph,
       "sdpa_forward": build_sdpa_graph,
       "conv_fwd": build_conv_graph,
   }
   ```

2. **Variant enumeration** (when `--variants` not provided):
   ```python
   def enumerate_variants(engine_id: int) -> list[dict]:
       """Query engine KMD, extract kernel metadata, return variant dicts."""
       # Option A: Read UKD catalog metadata
       # Option B: Query engine knobs via hipdnnGetEngineConfigs_ext()
       pass
   ```

3. **Applicability filtering**:
   ```python
   def filter_applicable(
       engine_id: int,
       corpus: list[dict],
       variants: list[dict]
   ) -> list[tuple[dict, dict]]:
       """Return only (problem, variant) pairs where engine.is_applicable(problem)."""
       # Call provider's is_applicable() predicate
       # Skip entire problem if predicate returns False
       pass
   ```

4. **Benchmark loop**:
   ```python
   for problem in corpus:
       graph = build_graph(problem)
       
       if not is_applicable(engine_id, graph):
           continue  # Skip this problem
       
       for variant in variants:
           if not is_valid_config(engine_id, graph, variant):
               continue  # Skip this variant
           
           try:
               timing = autotune_single(graph, engine_id, variant)
               write_result(problem, variant, timing, is_valid=True)
           except Exception as e:
               write_result(problem, variant, None, is_valid=False, error=str(e))
   ```

5. **Device properties** — query once at startup:
   ```python
   device_props = query_device_properties(device_idx)
   # Returns: {"device.cu_count": 120, "device.lds_size": 65536, ...}
   ```

**Complete C++ Implementation Example**:

See `Spec 6: Graph Construction and Autotune API Examples` below for full working code.

---

### Spec 3: Generic Applicability Interface

**Purpose:** Provider-agnostic predicate for filtering inapplicable (problem, engine) pairs.

**Python Interface:**
```python
class ApplicabilityProvider(ABC):
    @abstractmethod
    def is_applicable(self, engine_id: int, graph: hipdnn_graph) -> bool:
        """Return True if engine can execute this problem graph."""
        pass
    
    def is_valid_config(
        self,
        engine_id: int,
        graph: hipdnn_graph,
        variant: dict
    ) -> bool:
        """Return True if this variant applies to this problem. Default: True."""
        return True  # Override for per-variant filtering
```

**Implementation Options:**

**Option A: Expose via ctypes/cffi** (call C++ provider code):
```python
# Load provider .so
lib = ctypes.CDLL("libmiopen_provider.so")

# Declare C API
# bool isApplicable(void* graph_ptr)
lib.isApplicable.argtypes = [ctypes.c_void_p]
lib.isApplicable.restype = ctypes.c_bool

class MiopenApplicability(ApplicabilityProvider):
    def is_applicable(self, engine_id, graph):
        return lib.isApplicable(graph._as_parameter_)
```

**Option B: Plugin SDK Extension** (add to `hipdnn_plugin_sdk`):
```cpp
// Add to plugin SDK header
typedef bool (*HipdnnPluginIsApplicableFn)(
    const HipdnnBackendDescriptor_t graph,
    int64_t engine_id
);

// Providers implement:
bool MiopenProvider_IsApplicable(
    const HipdnnBackendDescriptor_t graph,
    int64_t engine_id
) {
    size_t solutionCount = 0;
    miopenConvolutionForwardGetSolutionCount(..., &solutionCount);
    return solutionCount > 0;
}
```

**Option C: Python Fallback** (provider-specific logic in tooling):
```python
# Hardcode known provider constraints
class MiopenConvApplicability(ApplicabilityProvider):
    def is_applicable(self, engine_id, graph):
        # Extract graph attributes
        attrs = extract_conv_attrs(graph)
        
        # Known MIOpen constraints (from docs/source)
        if attrs["spatial_dims"] > 3:
            return False  # MIOpen only supports up to 3D conv
        if attrs["group"] > 1 and attrs["group"] != attrs["channels"]:
            return False  # Grouped conv not fully supported
        
        return True
```

**Decision Point:** Which option to use?
- **Option A** (ctypes) — fastest to implement, brittle (ABI coupling)
- **Option B** (SDK) — cleanest, requires SDK changes + provider updates
- **Option C** (Python) — no C++ changes, but duplicates provider logic

**Recommended:** Start with Option C for MVP, migrate to Option B when stable.

---

### Spec 4: `uhd_gen install` Command

**Purpose:** Deploy trained UHD to engine descriptor set.

**CLI Signature:**
```bash
python -m uhd_gen install \
    --uhd-dir <dir> \
    --engine-id <hex> \
    [--target-dir <path>] \
    [--dry-run]
```

**Arguments:**

| Argument | Required | Type | Default | Description |
|----------|----------|------|---------|-------------|
| `--uhd-dir` | Yes | path | - | Directory with `uhd.fb`, `model.bin` |
| `--engine-id` | Yes | str | - | Target engine ID in hex |
| `--target-dir` | No | path | Auto-detect | Engine descriptor install location |
| `--dry-run` | No | bool | `False` | Print actions without executing |

**Auto-Detection Logic for `--target-dir`:**
```python
def auto_detect_descriptor_dir() -> Path:
    """Find engine descriptor installation directory."""
    # 1. Check environment variable
    if "HIPDNN_DESCRIPTOR_PATH" in os.environ:
        return Path(os.environ["HIPDNN_DESCRIPTOR_PATH"])
    
    # 2. Check CMAKE_INSTALL_PREFIX
    if "CMAKE_INSTALL_PREFIX" in os.environ:
        return Path(os.environ["CMAKE_INSTALL_PREFIX"]) / "share/hipdnn/engines"
    
    # 3. Check standard ROCm install
    rocm_path = Path("/opt/rocm/share/hipdnn/engines")
    if rocm_path.exists():
        return rocm_path
    
    # 4. Fallback to source tree (for development)
    return Path.cwd() / "descriptors"
```

**Installation Steps:**

1. **Verify inputs**:
   ```python
   uhd_fb = uhd_dir / "uhd.fb"
   model_bin = uhd_dir / "model.bin"
   uhd_json = uhd_dir / "uhd.json"
   
   assert uhd_fb.exists(), "Missing uhd.fb"
   assert model_bin.exists(), "Missing model.bin"
   ```

2. **Compute install paths**:
   ```python
   # Structure: <target_dir>/<engine_id_short>/<arch>/
   engine_short = f"{engine_id:016x}"[:8]  # First 8 hex digits
   
   # Detect arch from train_manifest.json
   manifest = json.load(uhd_dir / "train_manifest.json")
   arches = manifest.get("training_arches", ["gfx942"])
   
   for arch in arches:
       install_path = target_dir / engine_short / arch
       install_path.mkdir(parents=True, exist_ok=True)
   ```

3. **Copy artifacts**:
   ```python
   import shutil
   
   shutil.copy(uhd_fb, install_path / "uhd.fb")
   shutil.copy(model_bin, install_path / "model.bin")
   shutil.copy(uhd_json, install_path / "uhd.json")  # For human reference
   ```

4. **Update manifest** (optional, for tracking):
   ```python
   install_manifest = install_path / "install_manifest.json"
   manifest_data = {
       "uhd_id": uhd_metadata["id"],
       "uhd_name": uhd_metadata["name"],
       "installed_at": datetime.now().isoformat(),
       "installed_by": os.getlogin(),
       "source_dir": str(uhd_dir.resolve()),
       "features_hash": uhd_metadata["features_hash"],
       "model_version": manifest.get("model_version"),
   }
   json.dump(manifest_data, open(install_manifest, "w"), indent=2)
   ```

5. **Verify installation**:
   ```python
   # Load UHD via runtime loader (requires C binding)
   # OR just verify files exist and are readable
   assert (install_path / "uhd.fb").exists()
   assert (install_path / "model.bin").exists()
   print(f"✓ Installed UHD to {install_path}")
   ```

**Uninstall Support:**
```bash
python -m uhd_gen uninstall \
    --engine-id <hex> \
    --arch <arch> \
    [--target-dir <path>]
```

---

### Spec 5: `uhd_gen validate` Command

**Purpose:** Test trained UHD against hold-out dataset, report accuracy metrics.

**CLI Signature:**
```bash
python -m uhd_gen validate \
    --uhd-dir <dir> \
    --test-csv <path> \
    [--top-k <int>] \
    [--output <path>]
```

**Arguments:**

| Argument | Required | Type | Default | Description |
|----------|----------|------|---------|-------------|
| `--uhd-dir` | Yes | path | - | Directory with `uhd.fb`, `model.bin` |
| `--test-csv` | Yes | path | - | Hold-out test set (same schema as training) |
| `--top-k` | No | int | `1` | Validate top-K accuracy (K=1 means best prediction) |
| `--output` | No | path | `validation_report.json` | Validation report path |

**Output Schema:**
```json
{
  "summary": {
    "test_samples": 1000,
    "test_problems": 50,
    "top_1_accuracy": 0.85,
    "top_3_accuracy": 0.95,
    "top_5_accuracy": 0.98,
    "mean_regret": 0.12,
    "median_regret": 0.05,
    "max_regret": 0.45
  },
  "per_problem_metrics": [
    {
      "problem": {"q.M": 1024, "q.N": 1024, "q.K": 1024},
      "num_candidates": 20,
      "predicted_best": {"kernel.tile_m": 128, ...},
      "actual_best": {"kernel.tile_m": 256, ...},
      "predicted_score": 50.5,
      "actual_best_score": 52.3,
      "regret": 0.034,
      "top_1_correct": false,
      "top_3_correct": true
    },
    ...
  ]
}
```

**Metrics Definitions:**
- **Top-K accuracy** — fraction of problems where true-best kernel is in predicted top-K
- **Regret** — `(actual_best_score - predicted_best_score) / actual_best_score`
- **Mean/Median/Max regret** — statistics over all test problems

**Implementation:**

```python
def validate_uhd(uhd_dir: Path, test_csv: Path, top_k: int) -> dict:
    # 1. Load UHD model
    uhd = load_uhd(uhd_dir / "uhd.fb")
    model = load_model(uhd_dir / "model.bin")
    
    # 2. Load test data
    df = pd.read_csv(test_csv)
    
    # 3. Group by problem
    problem_cols = [c for c in df.columns if c.startswith("q.")]
    groups = df.groupby(problem_cols)
    
    per_problem_metrics = []
    top_k_correct = []
    regrets = []
    
    for problem_key, group in groups:
        # 4. Extract feature vectors for all candidates
        features = extract_features(group, uhd.features_signature)
        
        # 5. Score all candidates
        scores = [model.score(f) for f in features]
        
        # 6. Rank predictions
        predicted_ranking = np.argsort(scores)[::-1]  # Descending
        
        # 7. True ranking (from actual benchmark)
        actual_scores = group["tflops"].values  # Or minTimeMs, depending on target
        actual_ranking = np.argsort(actual_scores)[::-1]
        
        # 8. Compute metrics
        predicted_best_idx = predicted_ranking[0]
        actual_best_idx = actual_ranking[0]
        
        top_k_correct.append(actual_best_idx in predicted_ranking[:top_k])
        
        regret = (actual_scores[actual_best_idx] - actual_scores[predicted_best_idx]) / \
                 actual_scores[actual_best_idx]
        regrets.append(max(0, regret))  # Clamp negative regret to 0 (model got lucky)
        
        per_problem_metrics.append({
            "problem": dict(zip(problem_cols, problem_key)),
            "num_candidates": len(group),
            "predicted_best": group.iloc[predicted_best_idx][kernel_cols].to_dict(),
            "actual_best": group.iloc[actual_best_idx][kernel_cols].to_dict(),
            "predicted_score": actual_scores[predicted_best_idx],
            "actual_best_score": actual_scores[actual_best_idx],
            "regret": regret,
            f"top_{top_k}_correct": top_k_correct[-1],
        })
    
    return {
        "summary": {
            "test_samples": len(df),
            "test_problems": len(groups),
            f"top_{top_k}_accuracy": np.mean(top_k_correct),
            "mean_regret": np.mean(regrets),
            "median_regret": np.median(regrets),
            "max_regret": np.max(regrets),
        },
        "per_problem_metrics": per_problem_metrics,
    }
```

---

### Spec 6: Graph Construction and Autotune API Examples

**Purpose:** Concrete C++ and Python code showing how to build graphs, invoke autotune, and extract timing results.

#### C++ GEMM Graph Construction

Extracted from `test_sdk/utilities/FlatbufferGraphTestUtils.hpp`:

```cpp
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include "hipdnn_backend.h"

// Build a GEMM graph for (M, N, K) problem dimensions
flatbuffers::FlatBufferBuilder buildGemmGraph(int64_t M, int64_t N, int64_t K) {
    namespace fbs = hipdnn_flatbuffers_sdk::data_objects;
    
    // Define tensor shapes (A: M×K, B: K×N, C: M×N)
    std::vector<int64_t> aDims = {M, K};
    std::vector<int64_t> aStrides = {K, 1};  // Row-major
    std::vector<int64_t> bDims = {K, N};
    std::vector<int64_t> bStrides = {N, 1};
    std::vector<int64_t> cDims = {M, N};
    std::vector<int64_t> cStrides = {N, 1};
    
    // Build FlatBuffer graph
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<fbs::TensorAttributes>> tensorAttributes;
    
    // Tensor UIDs
    const int64_t aUid = 1, bUid = 2, cUid = 3;
    
    // Create tensor descriptors
    tensorAttributes.push_back(fbs::CreateTensorAttributesDirect(
        builder, aUid, "A", fbs::DataType::FLOAT, &aStrides, &aDims));
    tensorAttributes.push_back(fbs::CreateTensorAttributesDirect(
        builder, bUid, "B", fbs::DataType::FLOAT, &bStrides, &bDims));
    tensorAttributes.push_back(fbs::CreateTensorAttributesDirect(
        builder, cUid, "C", fbs::DataType::FLOAT, &cStrides, &cDims));
    
    // Create matmul operation node
    auto matmulAttributes = fbs::CreateMatmulAttributes(builder, aUid, bUid, cUid);
    
    std::vector<flatbuffers::Offset<fbs::Node>> nodes;
    nodes.push_back(fbs::CreateNodeDirect(
        builder, "matmul", fbs::DataType::FLOAT,
        fbs::NodeAttributes::MatmulAttributes, matmulAttributes.Union()));
    
    // Finalize graph
    auto graphOffset = fbs::CreateGraphDirect(
        builder, "gemm_graph",
        fbs::DataType::FLOAT,  // Primary data type
        fbs::DataType::FLOAT,  // Secondary
        fbs::DataType::FLOAT,  // Tertiary
        &tensorAttributes, &nodes);
    builder.Finish(graphOffset);
    
    return builder;
}

// Deserialize FlatBuffer → backend descriptor
hipdnnBackendDescriptor_t deserializeGraph(
    const flatbuffers::DetachedBuffer& serialized,
    hipdnnHandle_t handle) {
    
    hipdnnBackendDescriptor_t graph;
    hipdnnBackendCreateAndDeserializeGraph_ext(
        &graph, serialized.data(), serialized.size());
    
    // Attach handle
    hipdnnBackendSetAttribute(graph, HIPDNN_ATTR_OPERATIONGRAPH_HANDLE,
                              HIPDNN_TYPE_HANDLE, 1, &handle);
    
    hipdnnBackendFinalize(graph);
    return graph;
}
```

#### Python Graph Construction (via ctypes)

```python
import ctypes
from ctypes import c_void_p, c_int64, c_size_t, POINTER

# Load libhipdnn_backend.so
lib = ctypes.CDLL("libhipdnn_backend.so")

# Declare C API signatures
lib.hipdnnBackendCreateAndDeserializeGraph_ext.argtypes = [
    POINTER(c_void_p), c_void_p, c_size_t]
lib.hipdnnBackendFinalize.argtypes = [c_void_p]
lib.hipdnnBackendDestroyDescriptor.argtypes = [c_void_p]

class HipdnnGraph:
    """Wrapper for hipDNN graph descriptor."""
    
    def __init__(self, serialized_bytes: bytes, handle_ptr: int):
        self.graph_ptr = c_void_p()
        status = lib.hipdnnBackendCreateAndDeserializeGraph_ext(
            ctypes.byref(self.graph_ptr),
            serialized_bytes, len(serialized_bytes))
        
        if status != 0:  # HIPDNN_STATUS_SUCCESS
            raise RuntimeError(f"Graph deserialization failed: {status}")
        
        # Attach handle
        handle = c_void_p(handle_ptr)
        lib.hipdnnBackendSetAttribute(
            self.graph_ptr,
            0x0201,  # HIPDNN_ATTR_OPERATIONGRAPH_HANDLE
            0x0003,  # HIPDNN_TYPE_HANDLE
            1, ctypes.byref(handle))
        
        lib.hipdnnBackendFinalize(self.graph_ptr)
    
    def __del__(self):
        if self.graph_ptr:
            lib.hipdnnBackendDestroyDescriptor(self.graph_ptr)

def build_gemm_graph_python(M: int, N: int, K: int) -> bytes:
    """
    Build GEMM graph using Python FlatBuffers codegen.
    
    Prerequisites:
      1. Run: flatc --python -o ./python_gen hipdnn_flatbuffers_sdk/schemas/*.fbs
      2. Import generated modules
    """
    import flatbuffers
    # TODO: Import from python_gen/ after running flatc --python
    # from python_gen.hipdnn_flatbuffers_sdk.data_objects import (
    #     Graph, Node, TensorAttributes, MatmulAttributes, DataType
    # )
    
    # For now, call C++ builder via ctypes wrapper
    # OR manually construct FlatBuffer bytes
    raise NotImplementedError("Python FlatBuffers codegen TBD")
```

#### Autotune Integration (C++)

```cpp
#include "hipdnn_backend.h"
#include <chrono>
#include <numeric>
#include <cmath>

struct BenchmarkResult {
    int64_t engineId;
    std::map<std::string, int64_t> knobs;
    double minTimeMs;
    double avgTimeMs;
    double stddevMs;
    bool isValid;
    std::string error;
};

BenchmarkResult benchmarkSingleVariant(
    hipdnnHandle_t handle,
    hipdnnBackendDescriptor_t graph,
    int64_t engineId,
    const std::map<std::string, int64_t>& knobs,
    int warmup = 10,
    int repeat = 50) {
    
    BenchmarkResult result;
    result.engineId = engineId;
    result.knobs = knobs;
    result.isValid = true;
    
    try {
        // 1. Create engine descriptor
        hipdnnBackendDescriptor_t engine;
        hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINE_DESCRIPTOR, &engine);
        hipdnnBackendSetAttribute(engine, HIPDNN_ATTR_ENGINE_OPERATION_GRAPH,
                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &graph);
        hipdnnBackendSetAttribute(engine, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX,
                                  HIPDNN_TYPE_INT64, 1, &engineId);
        hipdnnBackendFinalize(engine);
        
        // 2. Create engine config (variant-specific knobs would go here)
        // NOTE: Current backend API does not expose knob-setting directly
        // This is a gap — knobs must be set via engine heuristics or
        // extended API (hipdnnBackendSetEngineKnobs_ext)
        hipdnnBackendDescriptor_t engineConfig;
        hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, &engineConfig);
        hipdnnBackendSetAttribute(engineConfig, HIPDNN_ATTR_ENGINECFG_ENGINE,
                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &engine);
        hipdnnBackendFinalize(engineConfig);
        
        // 3. Create execution plan
        hipdnnBackendDescriptor_t executionPlan;
        hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &executionPlan);
        hipdnnBackendSetAttribute(executionPlan, HIPDNN_ATTR_EXECUTION_PLAN_HANDLE,
                                  HIPDNN_TYPE_HANDLE, 1, &handle);
        hipdnnBackendSetAttribute(executionPlan, HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &engineConfig);
        hipdnnBackendFinalize(executionPlan);
        
        // 4. Allocate tensors (extract from graph, simplified here)
        // TODO: Parse graph to get tensor UIDs/sizes, allocate via hipMalloc
        void* A = nullptr, *B = nullptr, *C = nullptr;
        // ... hipMalloc, initialize tensors ...
        
        // 5. Create variant pack (tensor bindings)
        hipdnnBackendDescriptor_t variantPack;
        hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &variantPack);
        
        std::vector<int64_t> tensorUids = {1, 2, 3};
        std::vector<void*> dataPtrs = {A, B, C};
        hipdnnBackendSetAttribute(variantPack, HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                                  HIPDNN_TYPE_INT64, tensorUids.size(), tensorUids.data());
        hipdnnBackendSetAttribute(variantPack, HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                  HIPDNN_TYPE_VOID_PTR, dataPtrs.size(), dataPtrs.data());
        hipdnnBackendFinalize(variantPack);
        
        // 6. Warmup + timing loop
        std::vector<double> times;
        for (int i = 0; i < warmup + repeat; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            
            hipdnnBackendExecute(handle, executionPlan, variantPack);
            hipDeviceSynchronize();
            
            auto end = std::chrono::high_resolution_clock::now();
            double timeMs = std::chrono::duration<double, std::milli>(end - start).count();
            
            if (i >= warmup) {
                times.push_back(timeMs);
            }
        }
        
        // Compute statistics
        result.minTimeMs = *std::min_element(times.begin(), times.end());
        result.avgTimeMs = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        double variance = 0.0;
        for (double t : times) {
            variance += (t - result.avgTimeMs) * (t - result.avgTimeMs);
        }
        result.stddevMs = std::sqrt(variance / times.size());
        
        // Cleanup
        hipdnnBackendDestroyDescriptor(variantPack);
        hipdnnBackendDestroyDescriptor(executionPlan);
        hipdnnBackendDestroyDescriptor(engineConfig);
        hipdnnBackendDestroyDescriptor(engine);
        // ... hipFree tensors ...
        
    } catch (const std::exception& e) {
        result.isValid = false;
        result.error = e.what();
    }
    
    return result;
}
```

#### Device Properties Query

```cpp
#include <hip/hip_runtime.h>

struct DeviceProperties {
    int cuCount;
    size_t ldsSizeBytes;
    int maxThreadsPerBlock;
    std::string archName;
};

DeviceProperties queryDeviceProperties(int deviceIdx = 0) {
    DeviceProperties props;
    
    hipSetDevice(deviceIdx);
    
    hipDeviceProp_t hipProps;
    hipGetDeviceProperties(&hipProps, deviceIdx);
    
    props.cuCount = hipProps.multiProcessorCount;
    props.ldsSizeBytes = hipProps.sharedMemPerBlock;
    props.maxThreadsPerBlock = hipProps.maxThreadsPerBlock;
    props.archName = hipProps.gcnArchName;  // e.g., "gfx942"
    
    return props;
}
```

#### Python Device Properties

```python
import ctypes

# Load HIP runtime
hip = ctypes.CDLL("libamdhip64.so")

class HipDeviceProp(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char * 256),
        ("totalGlobalMem", ctypes.c_size_t),
        ("sharedMemPerBlock", ctypes.c_size_t),
        ("regsPerBlock", ctypes.c_int),
        ("warpSize", ctypes.c_int),
        ("maxThreadsPerBlock", ctypes.c_int),
        ("multiProcessorCount", ctypes.c_int),
        ("gcnArchName", ctypes.c_char * 256),
        # ... (full struct has more fields)
    ]

def query_device_properties(device_idx: int = 0) -> dict:
    props = HipDeviceProp()
    hip.hipGetDeviceProperties(ctypes.byref(props), device_idx)
    
    return {
        "device.cu_count": props.multiProcessorCount,
        "device.lds_size": props.sharedMemPerBlock,
        "device.max_threads_per_block": props.maxThreadsPerBlock,
        "device.arch_name": props.gcnArchName.decode('utf-8'),
    }
```

#### Known Gaps and Workarounds

1. **Knob Setting API**: Current backend does not expose per-variant knob setting directly. Workarounds:
   - **Option A**: Enumerate engine configs via heuristics, filter by matching knobs
   - **Option B**: Extend backend with `hipdnnBackendSetEngineKnobs_ext()`
   - **Option C**: Build separate execution plans per variant (inefficient)

2. **Tensor Allocation**: Graph does not expose tensor sizes directly. Workarounds:
   - Parse graph FlatBuffer to extract dims
   - Use `hipdnnBackendGetAttribute()` to query tensor descriptors
   - Compute size from `dataType × product(dims)`

3. **Python FlatBuffers Codegen**: Requires running `flatc --python`. For now:
   - Call C++ builder via ctypes
   - OR manually construct FlatBuffer bytes (fragile)
   - OR generate Python bindings once and check into repo

---

## Next Steps to Complete the Pipeline

### High Priority (Blocks End-to-End Training)

1. **Implement `uhd_gen generate-corpus`** (RFC 0019.13 §7.1)
   ```bash
   python -m uhd_gen generate-corpus \
       --operation gemm \
       --tier standard \
       --output gemm_corpus.json
   ```

2. **Implement `uhd_gen benchmark`** (RFC 0019.13 §7.4)
   ```bash
   python -m uhd_gen benchmark \
       --corpus gemm_corpus.json \
       --engine-id 0x... \
       --output benchmark_results.csv
   ```

3. **Define Generic Applicability Interface** (RFC 0019.13 §7.3)
   - Expose `KernelProvider.is_applicable(graph)` to Python
   - Integrate into benchmark loop

### Medium Priority (Improves Usability)

4. **Implement `uhd_gen install`** — deploy UHD to engine descriptor set
5. **Implement `uhd_gen validate`** — test UHD against hold-out set
6. **Add derived features support** — auto-generate tile quantization, arithmetic intensity

### Low Priority (Nice to Have)

7. **Multi-arch training** — train one UHD across multiple GPUs
8. **Hyperparameter tuning** — auto-search LightGBM params
9. **Model comparison** — A/B test UHDs, report performance deltas

---

## Example: Complete Workflow (When Tooling Exists)

**Hypothetical future state:**

```bash
# Step 1: Generate corpus
python -m uhd_gen generate-corpus \
    --operation gemm \
    --tier standard \
    --output gemm_corpus.json

# Step 2: Benchmark
python -m uhd_gen benchmark \
    --corpus gemm_corpus.json \
    --engine-id 0x00000002BEEF0001 \
    --output gemm_benchmark.csv

# Step 3: Train
python -m uhd_gen \
    --input gemm_benchmark.csv \
    --features q.M q.N q.K kernel.tile_m kernel.tile_n device.cu_count \
    --target tflops \
    --output-dir gemm_uhd

# Step 4: Validate
python -m uhd_gen validate \
    --uhd-dir gemm_uhd \
    --test-csv gemm_test_set.csv

# Step 5: Install
python -m uhd_gen install \
    --uhd-dir gemm_uhd \
    --engine-id 0x00000002BEEF0001
```

**Current reality:** Only Step 3 (train) exists. Steps 1, 2, 4, 5 require manual workarounds or new implementation.
