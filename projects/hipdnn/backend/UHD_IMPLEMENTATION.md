# UHD (Universal Heuristic Descriptor) Implementation

This document describes the UHD POC implementation on branch `users/cderb/rfc-uhd`, covering RFC 0019 (Universal Heuristic Descriptor).

## Overview

The Universal Heuristic Descriptor (UHD) provides a data-driven kernel selection system that scores and ranks kernel candidates using ML models. This POC implements the complete UHD selection flow per RFC 0019 §6.

## Scope

**In Scope** (UHD POC):
- ✅ All adapter types (tree_data, table, onnx, custom_library, static_order)
- ✅ Feature extraction with derived values (RFC §6.4)
- ✅ Score transforms and objectives (RFC §5, §12.3)
- ✅ Hash validation (RFC §7.3, §9.2)
- ✅ Selection engine with full UHD integration
- ✅ Comprehensive test coverage

**Out of Scope** (other RFCs):
- ❌ Kernel ingestor (`HIPDNN_ENABLE_KERNEL_INGESTOR`)
- ❌ Real descriptor loading (UED, KMD, UKD, KDP)
- ❌ Graph matcher / graph_match
- ❌ UMD (Universal Match Descriptor) - FLOP/byte annotations are UMD fields
- ❌ UDD (Universal Dispatch Descriptor)

## Architecture

### Core Components

```
src/heuristics/uhd/
├── adapters/               # Model adapters (RFC §7)
│   ├── IUhdAdapter.hpp    # Adapter interface
│   ├── TreeDataAdapter.*  # GBDT tree walker (default)
│   ├── TableAdapter.*     # Bucket lookup
│   ├── OnnxAdapter.*      # ONNX Runtime (opt-in)
│   └── CustomLibraryAdapter.* # Shared library plugin
├── FeatureExtractor.*     # Feature extraction (RFC §6.1-6.4)
├── SelectionEngine.*      # Selection flow (RFC §6)
├── EngineRegistry.*       # Engine/candidate lookup (mock)
├── JsonLogicEvaluator.*   # Derived values expressions
├── ScoreTransform.hpp     # Score transforms (log1p, sqrt, etc.)
├── Sha256.*               # Hash validation
└── UhdLoader.*            # FlatBuffer UHD loading
```

### Data Flow

```
1. Register Engine
   ├─ EngineEntry (mock UED)
   │  ├─ UhdConfig (features, objective, adapter, model path)
   │  └─ KernelCandidate[] (mock UKD with KMD metadata)
   └─ EngineRegistry::registerEngine()

2. Selection (SelectionEngine::select)
   ├─ Load adapter (getOrCreateAdapter)
   ├─ Extract features (FeatureExtractor)
   │  ├─ Bind device vars ($device.*)
   │  ├─ Bind query vars ($q.*)
   │  ├─ Evaluate derived values ($derived.*)
   │  └─ Extract per-candidate ($kernel.*)
   ├─ Score candidates (IUhdAdapter::score)
   ├─ Apply score transforms (log1p, sqrt, etc.)
   ├─ Sort by objective (max/min)
   └─ Return ranked kernel IDs

3. Result
   └─ SelectionResult (bestKernelId, sortedKernelIds, applied, fallbackReason)
```

## Feature Extraction (RFC §6)

### Feature Sources

Features come from four namespaces:

| Namespace | Source | Example | Notes |
|-----------|--------|---------|-------|
| `$device.*` | Device properties | `$device.cu_count` | GPU hardware facts |
| `$q.*` | Query parameters | `$q.batch`, `$q.seqlen_q` | Problem attributes |
| `$kernel.*` | Kernel metadata (KMD) | `$kernel.tile_m`, `$kernel.num_warps` | Per-candidate config |
| `$derived.*` | Computed values | `$derived.num_tiles` | JsonLogic expressions |

### Derived Values (RFC §6.4)

Derived values are computed features defined as JsonLogic expressions. They can reference earlier derived values and are evaluated in declaration order.

**Example:**
```cpp
entry.uhdConfig.derived = {
    {"num_tiles_m", "{\"ceil_div\": [\"$q.seqlen_q\", \"$kernel.tile_m\"]}"},
    {"num_tiles_k", "{\"ceil_div\": [\"$q.seqlen_k\", \"$kernel.tile_n\"]}"},
    {"total_tiles", "{\"*\": [\"$derived.num_tiles_m\", \"$derived.num_tiles_k\"]}"}
};
```

**Kernel-dependent vs Independent:**
- **Independent**: References only `$device.*` and `$q.*` → evaluated once in `extractSharedRow()`
- **Dependent**: References `$kernel.*` or kernel-dependent `$derived.*` → re-evaluated per candidate

## Model Adapters (RFC §7)

### Adapter Types

| Adapter | Dependency | Use Case | File Format |
|---------|-----------|----------|-------------|
| `static_order` | None | Priority-based fallback | N/A (config only) |
| `tree_data` | None | **Default** - GBDT models | FlatBuffer `.gbdt` |
| `table` | None | Bucket lookup | FlatBuffer `.table` |
| `onnx` | ONNX Runtime | Neural networks | `.onnx` |
| `custom_library` | None | Custom scorers | `.so` (C ABI) |

### TreeDataAdapter (GBDT)

The default adapter for production use. Loads LightGBM GBDT models from FlatBuffer artifacts.

**Features:**
- ✅ Cycle detection (RFC §16 - bounded descent)
- ✅ NaN handling with `default_left`
- ✅ `decision_lte` support (≤ vs <)
- ✅ Model hash validation (SHA-256)
- ✅ Training arches metadata

**Example:**
```cpp
auto adapter = TreeDataAdapter::load("model.gbdt", expectedFeaturesHash, expectedModelHash);
double score = adapter->score(features);
```

### TableAdapter (Bucket Lookup)

Quantizes continuous features into buckets and looks up scores in a pre-computed table.

**Example:**
```cpp
auto adapter = TableAdapter::load("model.table", expectedFeaturesHash);
double score = adapter->score(features);
```

### OnnxAdapter

Dependency-gated ONNX Runtime integration. Only available when built with `-DHIPDNN_ENABLE_ONNX=ON`.

**Features:**
- ✅ ROCm execution provider with CPU fallback
- ✅ Single-threaded inference
- ✅ 2D input shape validation [1, num_features]

### CustomLibraryAdapter

Loads external scoring functions via `dlopen`. The engine ships its own `.so` with a C ABI scorer:

```c
extern "C" double my_scorer(const double* features, size_t count);
```

## Score Transforms (RFC §12.3)

Score transforms normalize model outputs for cross-engine comparison. The UHD declares a forward transform; the runtime applies the inverse to report scores in the declared units.

| Transform | Forward | Inverse | Use Case |
|-----------|---------|---------|----------|
| `identity` | x | x | Already in target units |
| `log1p` | log(1+x) | exp(x)-1 | TFLOPS (always positive) |
| `log` | log(x) | exp(x) | Ratios |
| `sqrt` | √x | x² | Magnitudes |

**Example:**
```cpp
entry.uhdConfig.scoreUnits = "tflops";
entry.uhdConfig.scoreTransform = "log1p";
entry.uhdConfig.scoreCalibrated = true; // Cross-engine comparable
```

At selection time, if the model returns `2.5`, the runtime computes `exp(2.5) - 1 ≈ 11.2 tflops`.

## Hash Validation (RFC §7.3, §9.2)

Two levels of hash validation ensure model/signature contract integrity:

1. **Features Hash** (§7.3): SHA-256 of features_signature
   - Model artifact embeds the hash it was trained on
   - Load-time check: descriptor hash must match model hash
   - Prevents signature/model mismatch

2. **Model Hash** (§9.2): SHA-256 of model artifact
   - Optional integrity check against corruption
   - Enables rollback (drop new descriptor, restore old)

**Example:**
```cpp
std::string featuresHash = FeatureExtractor::computeHash(signature);
entry.uhdConfig.featuresHash = featuresHash;
entry.uhdConfig.modelHash = sha256(modelBytes);
```

## Testing

### Test Coverage

| Test Suite | File | Coverage |
|------------|------|----------|
| Adapter tests | `TestTreeDataAdapter.cpp` | GBDT loading, scoring, hash validation |
| | `TestTableAdapter.cpp` | Bucket quantization, lookup |
| | `TestOnnxAdapter.cpp` | ONNX inference (dependency-gated) |
| | `TestCustomLibraryAdapter.cpp` | dlopen/dlsym, C ABI scoring |
| Feature extraction | `TestFeatureExtractor.cpp` | Namespace binding, signature parsing |
| | `TestDerivedValues.cpp` | Chained derived, kernel-dependent |
| JsonLogic | `TestJsonLogicEvaluator.cpp` | Operators, edge cases |
| Selection flow | `TestUhdSelectionFlow.cpp` | End-to-end: register → select → rank |
| Score transforms | (in `TestUhdSelectionFlow.cpp`) | log1p, sqrt, identity round-trips |
| UHD loader | `uhd/TestUhdLoader.cpp` | FlatBuffer UHD deserialization |

### Running Tests

```bash
cd <workspace>/projects/hipdnn/build
ninja hipdnn_backend_tests
./bin/hipdnn_backend_tests --gtest_filter="*Uhd*"
```

## Example Usage

A complete working example is in `examples/uhd_selection_example.cpp`. Build and run:

```bash
cd <workspace>/projects/hipdnn/build
ninja uhd_selection_example
./examples/uhd_selection_example
```

**Example output:**
```
=== UHD Selection Flow Example ===

[Step 1] Building GBDT model artifact...
  Model size: 512 bytes
  Trees: 2, Features: 2 (tile_m, batch)

[Step 2] Registering engine with UHD config...
  Engine ID: 1000
  Adapter: tree_data (GBDT)
  Candidates: 3 (tile_m = 64, 128, 256)
  Objective: max (higher score = better)
  Score transform: log1p

[Step 3] Setting up query context...
  Device: cu_count=120, total_mem=32GB
  Query: batch=32, seqlen_q=2048, heads=32, hdim=128

[Step 4] Running UHD selection...
  Selection applied: YES
  Candidates scored: 3

[Step 5] Selection Results:
  Best kernel: 101

  Ranked candidates (best to worst):
    1. Kernel 101 (tile_m=64)
    2. Kernel 102 (tile_m=128)
    3. Kernel 103 (tile_m=256)
```

## Mock Components

This POC uses mocks for components defined in other RFCs:

### EngineEntry (mock UED - RFC 0020)

Represents an engine with its UHD configuration and kernel candidates. In the full system, this comes from the UED (Universal Engine Descriptor).

```cpp
struct EngineEntry {
    int64_t engineId;
    std::string engineName;
    UhdConfig uhdConfig;        // From UED
    std::vector<KernelCandidate> candidates; // From KDPs/UKDs
};
```

### KernelCandidate (mock UKD - RFC 0017)

Represents a kernel variant with its KMD (Kernel Metadata Descriptor) field values. In the full system, this comes from UKDs.

```cpp
struct KernelCandidate {
    int64_t kernelId;
    int64_t priority = 0;
    std::unordered_map<std::string, double> metadata; // KMD fields
};
```

### EngineRegistry

Mock registry providing engine/candidate lookup. In the full system, this is replaced by real UED/UKD descriptor queries.

## Integration with Full System

When RFC 0017 (UKD) and RFC 0020 (UED) land:

1. **Remove mocks**: `EngineEntry`, `KernelCandidate`, `EngineRegistry`
2. **Replace with**: Real descriptor queries from the kernel ingestor
3. **Keep**: All adapter implementations, feature extraction, selection flow, score transforms

The UHD selection flow (`SelectionEngine::select`) is designed to work with both the mock registry and the real descriptor system - only the lookup mechanism changes.

## References

- **RFC 0019**: Universal Heuristic Descriptor (this implementation)
- **RFC 0017**: Universal Kernel Descriptor (UKD, KMD, adapters)
- **RFC 0020**: Universal Engine Descriptor (UED, arch-keyed UHD maps)

## Open Items

See `/home/cerb/.claude/projects/-home-cerb-ml-lobby-rocm-libraries/memory/rfc0019-uhd-open-items.md` for ranked remaining defects and future work.
