# RFC 0010: Golden Reference Validation

> Owner: Integration Test Team
> Last updated: 2026-05-04

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Problem Statement](#problem-statement)
3. [Pipeline Overview](#pipeline-overview)
4. [Step 1: construct -- Build the Graph](#step-1-construct----build-the-graph)
5. [Step 2: execute-reference -- Run the Reference Executor](#step-2-execute-reference----run-the-reference-executor)
6. [Step 3: serialize -- Save Inputs and Outputs to Disk](#step-3-serialize----save-inputs-and-outputs-to-disk)
7. [Step 4: deserialize -- Load Inputs Back](#step-4-deserialize----load-inputs-back)
8. [Step 5: execute-engine -- Run the Engine Under Test](#step-5-execute-engine----run-the-engine-under-test)
9. [Step 6: deserialize -- Load Saved Outputs](#step-6-deserialize----load-saved-outputs)
10. [Step 7: validate -- Compare Engine Output to Saved Output](#step-7-validate----compare-engine-output-to-saved-output)
11. [Verification Modes](#verification-modes)
12. [CLI and Configuration](#cli-and-configuration)
13. [Harness Integration](#harness-integration)
14. [Data Management](#data-management)
15. [CI Integration](#ci-integration)
16. [Adding New Golden Reference Tests](#adding-new-golden-reference-tests)
17. [Implementation Phases](#implementation-phases)
18. [Risk Register](#risk-register)
19. [Quality Principles](#quality-principles)
20. [Known Limitations](#known-limitations)
21. [Future Work](#future-work)

---

## Executive Summary

The integration test suite currently supports two verification modes. This RFC adds a third:

1. **CPU reference** (existing) -- compute reference on CPU at runtime
2. **GPU reference** (existing) -- compute reference on GPU at runtime
3. **Golden reference** (new) -- compare against pre-computed, version-controlled reference data from disk

### Key Benefits
- **Deterministic baselines and regression detection**: Reference data is frozen from an agreed-upon known-good source. Test outcomes depend only on the engine under test. Unlike computed mode -- where both the reference executor and the engine can drift together and the test still passes -- golden mode compares against a locked baseline, so any engine change is caught
- **Unblocks testing before C++ reference kernels exist**: Generate golden data from any trusted source (see [Reference Sources](#reference-sources)) and start validating immediately -- the C++ reference kernel can follow later without delaying test coverage
- **Faster execution**: Eliminates runtime reference computation for large tensors

### How It Works

At its core, the golden reference feature follows these steps across two pipelines -- generation (run once) and validation (run every CI):

| Step | Tag | What happens | Generation | Validation |
|------|-----|-------------|:---:|:---:|
| 1 | `construct` | Build the graph from the test fixture's `buildGraph()` | Y | Y |
| 2 | `execute-reference` | Run a trusted reference (CPU ref, GPU ref, or external) to produce truth | Y | -- |
| 3 | `serialize` | Save inputs + outputs to disk | Y | -- |
| 4 | `deserialize` | Load saved inputs from disk | -- | Y |
| 5 | `execute-engine` | Run MIOpen GPU (the thing being tested) | -- | Y |
| 6 | `deserialize` | Load saved reference outputs from disk | -- | Y |
| 7 | `validate` | Compare engine output to saved output | -- | Y |

No reference executor runs in CI. The truth was computed once and frozen to disk. The graph is always rebuilt from `buildGraph()` in the test fixture. A graph fingerprint (hash of graph properties) detects when the graph has changed since golden data was generated. The diagrams in [Pipeline Overview](#pipeline-overview) show both pipelines visually.

### Golden Data Is a Key-Value Store

Golden data is a **key-value store**:

- **Key** = the graph identity (what computation are we testing?)
- **Value** = the tensor data (what are the correct inputs and expected outputs?)

If the key is wrong — the graph has changed since generation but the fingerprint didn't catch it — every downstream step (deserialize, execute, compare) operates on stale data from a different computation. A pass is false confidence. A fail is a wild goose chase.

The graph fingerprint must be correct, deterministic, and under our control. See [Graph Fingerprint](#graph-fingerprint----catches-graph-drift-after-generation) for the design.

---

## Problem Statement

The integration test suite currently validates engine outputs by computing references at runtime via [`CpuReferenceGraphExecutor`](../../test_sdk/include/hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp) or [`GpuReferenceGraphExecutor`](../../../../dnn-providers/integration-tests/src/harness/gpu_graph_executor/GpuReferenceGraphExecutor.hpp). This creates several gaps:

1. **Circular dependency risk**: If the reference executor has a bug, both sides produce the same wrong answer and the test passes
2. **Coverage gap**: Operations not yet implemented in the reference executor cannot be tested (e.g., instead of writing a C++ SDPA reference kernel, golden data lets us use an external framework's output as truth via `--external-reference`)
3. **Non-determinism**: GPU reference results can vary across runs, making failure investigation harder
4. **Slowness**: CPU reference execution for large tensors is the bottleneck in full-tier tests

A prior effort established a golden reference pattern in the MIOpen plugin's test suite ([`GoldenReferenceGpu.hpp`](../../../../dnn-providers/miopen-provider/tests/common/GoldenReferenceGpu.hpp)) using serialized graph+tensor JSON files loaded from `hipdnn_reference_data/`. This RFC builds on that pattern, extends it for broader use, and integrates it into the shared integration test harness so it works across all plugins (MIOpen, Fusilli, and future ones) without plugin-specific code.

---

## Pipeline Overview

Two pipelines share the same set of steps:

### Generation Pipeline (runs once, produces golden data)

![Generation Pipeline](images/generation_pipeline.png)

### Validation Pipeline (runs every CI)

![Validation Pipeline](images/validation_pipeline.png)

---

## Step 1: construct -- Build the Graph

**What**: Construct the computation graph using the test fixture's `buildGraph()` method.

**Who calls it**: Both the generation pipeline and the validation pipeline.

**What exists today**: This step already works. Every test fixture (e.g., `IntegrationGpuConvForward.cpp`) implements `buildGraph()` which creates a `graph::Graph`, adds tensors with explicit **names** (e.g., `"x"`, `"w"`, `"y"`), creates operations, validates, and builds the operation graph.

**What changes**: Nothing. The graph construction code is untouched. The golden data system uses a [graph-property hash](#graph-fingerprint----catches-graph-drift-after-generation) built from graph properties (tensor names, dims, types, operation parameters) to detect when `buildGraph()` has changed since golden data was generated.

**Design constraint**: Tensor names set in `buildGraph()` are the **identity contract** for golden data. They are how golden files map to runtime tensors. A tensor named `"x"` in the golden file must correspond to a tensor named `"x"` in the graph.

### Tensor Identity Contract

**Invariant**: Golden data maps tensors by **name** (e.g., `"X"`, `"W"`, `"Y"`), never by UID.

UIDs are assigned during `buildGraph()` and can change across refactors. Names are set explicitly by test authors and are part of the test's semantic contract. Names must be unique within a graph -- if two tensors share a name, the manifest silently drops one and the name-to-UID map silently overwrites the other, causing wrong data to load with no error. The generator enforces this at write time. The golden data loader resolves names to UIDs at load time:

```cpp
// Resolve golden tensor names to runtime UIDs
void resolveGoldenTensors(
    const graph::Graph& graph,
    const GoldenManifest& manifest,
    GraphTensorBundle& bundle)
{
    std::unordered_map<std::string, int64_t> nameToUid;
    graph.visit([&](const graph::INode& node) {
        for(const auto& attr : node.getNodeInputTensorAttributes())
            nameToUid[attr->get_name()] = attr->get_uid();
        for(const auto& attr : node.getNodeOutputTensorAttributes())
            nameToUid[attr->get_name()] = attr->get_uid();
    });

    for(const auto& [name, tensorInfo] : manifest.inputs)
    {
        auto it = nameToUid.find(name);
        if(it == nameToUid.end())
        {
            FAIL() << "Golden data references tensor '" << name
                   << "' not found in runtime graph. "
                   << "Graph construction may have changed since golden data was generated.";
        }
        bundle.tensors.insert({it->second, loadTensorFromBin(tensorInfo)});
    }
}
```

If a golden file references a tensor name that does not exist in the runtime graph, the test **fails immediately** with a diagnostic message. This is a fail-fast guard against silent data staleness.

**Acceptance criteria**:
- [ ] `resolveGoldenTensors()` maps by `attr->get_name()`, never by UID
- [ ] Golden tensor name absent from runtime graph: **hard FAIL** with message naming the missing tensor
- [ ] Runtime output tensor absent from golden data: **hard FAIL** listing expected golden names
- [ ] Unit test: Rename a tensor UID, verify golden validation still works (name unchanged)
- [ ] Unit test: Rename a tensor name, verify golden validation fails with clear diagnostic
- [ ] Unit test: Add a new output tensor not in golden data, verify hard FAIL
- [ ] Generator refuses to write golden data if any tensor name is empty or duplicated within the graph

---

## Step 2: execute-reference -- Run the Reference Executor

**What**: Execute the CPU reference to produce trusted output values.

**Who calls it**: Only the generation pipeline. This step does NOT run during CI validation -- that's the whole point of golden data.

**What exists today**: `CpuReferenceGraphExecutor` serializes the graph to flatbuffer, then walks nodes in topological order using `PlanBuilderRegistry` to look up a CPU implementation for each operation. It supports: ConvFwd/Bwd/Wrw, BatchnormInf/Train/Bwd, Pointwise (all modes), Matmul, Layernorm, RMSNorm, SDPA, Reduction, BlockScaleDequantize.

**What changes**: Nothing to the executor itself. The generation pipeline calls it exactly as the existing `verifyGraph()` does:

```cpp
executeCpuGraph(graph, cpuBundle);

// Which is:
auto [serializedGraph, serErr] = graph.to_binary();
CpuReferenceGraphExecutor().execute(
    serializedGraph.data(), serializedGraph.size(),
    cpuBundle.toHostVariantPack());
```

**Two distinct executors -- do not confuse**:

| | execute-reference (Step 2) | execute-engine (Step 5) |
|---|---|---|
| **What** | A trusted reference source | MIOpen GPU engine |
| **When** | Once, during golden data generation | Every CI run, during validation |
| **Purpose** | Produce trusted truth | Produce output to validate |
| **Source** | CPU ref, GPU ref, or external (e.g., PyTorch) | `graph.execute(handle, ...)` |

### Reference Sources

The golden data format is **reference-source-agnostic**. The manifest records which source produced the truth (via `reference_executor` field), but the validation pipeline doesn't care -- it just loads tensors and compares. This means any trusted source can produce golden data:

| Category | Examples | Flag | When to use |
|----------|----------|------|------------|
| In-house CPU ref | `CpuReferenceGraphExecutor` | `--reference-executor cpu` (default) | Most ops -- already implemented, trusted |
| In-house GPU ref | `GpuReferenceGraphExecutor` | `--reference-executor gpu` | When CPU ref is too slow or unavailable |
| Python frameworks | PyTorch, TensorFlow, JAX | `--external-reference <dir>` | New ops with no C++ ref kernel yet; unblocks testing immediately |
| Third-party engines | Other vendor libraries, other hipDNN plugins | `--external-reference <dir>` | Cross-engine validation, competitive benchmarking |
| Other C++ implementations | Standalone C++ reference code, research prototypes | `--external-reference <dir>` | When a team has a verified implementation outside the test SDK |

The first two (CPU/GPU ref) are built-in -- the generator calls them directly. Everything else goes through `--external-reference`, which reads pre-computed tensor files from a directory. The golden data format doesn't distinguish between them; only the manifest metadata records the origin.

**Future-proofing**: As new engines are added to hipDNN (e.g., Fusilli, or a new plugin), their outputs can also serve as reference data for cross-engine comparison. The golden data format doesn't need to change -- just generate from one engine, validate against another.

### External Reference Pipeline

For operations where no C++ reference executor exists, **PyTorch (or any framework) can produce the truth instead**. This avoids writing and maintaining a C++ reference kernel for every operation -- PyTorch already has a correct, well-tested implementation.

```bash
# 1. Generate reference outputs with PyTorch
python scripts/generate_reference.py \
  --op sdpa_fwd \
  --cases "B1_H8_S128_D64,B2_H16_S256_D64" \
  --output-dir ./pytorch_outputs/

# 2. Feed them into the golden data generator
./hipdnn_integration_tests \
  --generate-golden \
  --external-reference ./pytorch_outputs/ \
  --golden-data-dir ./golden_data \
  --gtest_filter="*SdpaFwd*Smoke*"
```

#### What the Python script must produce

The `--external-reference` directory must contain one raw binary file per output tensor, named to match the tensor name in the graph (e.g., `Y.bin`). The generator reads these files instead of running a C++ reference executor, then packages them into the standard golden data format (manifest + binary blobs + SHA-256).

#### Where the scripts live

External reference scripts live in `scripts/reference_generators/` within the integration test project:

```
integration-tests/
  scripts/
    reference_generators/
      generate_sdpa_reference.py    # SDPA fwd/bwd
      generate_layernorm_reference.py
      common.py                     # Shared utilities (tensor I/O, shape parsing)
      requirements.txt              # torch, numpy
```

#### Graph-Level Correctness: Closing the Semantic Gap

The core risk with external references is: **how do we know the Python script ran the same computation as the C++ graph?** If the Python script hardcodes `padding=1` but the C++ graph says `padding=0`, the golden data is wrong and nothing catches it.

The following diagram shows how the graph definition flows through the system and where each correctness check happens:

![Graph-Level Correctness](images/graph_level_correctness.png)

Three checks, each catching a different kind of mismatch:

#### Graph Fingerprint -- catches graph drift after generation

The graph fingerprint is the **key** in the key-value store described in [Golden Data Is a Key-Value Store](#golden-data-is-a-key-value-store). If it fails, nothing else matters.

**Goal:** Detect when `buildGraph()` has changed since golden data was generated, *before* loading any tensors or running any engine.

**Why not hash `to_binary()`?** The obvious approach is to hash the output of `graph.to_binary()`. This delegates to the backend's `backendGetSerializedBinaryGraphExt()`, which uses FlatBuffers internally. FlatBuffers does **not** guarantee deterministic serialization — the official docs state *"two different implementations may produce different binaries given the same input values, and this is perfectly valid."* In practice, the same binary produces identical bytes, but FlatBuffers library upgrades, schema changes, or compiler changes can silently break every fingerprint without a single graph actually changing. Since the fingerprint is the foundation of the entire system (see [Golden Data Is a Key-Value Store](#golden-data-is-a-key-value-store)), we cannot build it on a format we don't control.

**Design: Graph-property hash.** Instead of hashing serialized bytes, hash the **graph properties that determine numerical output**:

```cpp
std::string computeGraphFingerprint(const graph::Graph& graph) {
    // Walk graph nodes in deterministic (sorted-by-name) order.
    // For each node, collect:
    //   - operation type (e.g., ConvFwd, Matmul, SDPA)
    //   - operation parameters (padding, stride, dilation, etc.)
    // For each tensor, collect:
    //   - name, dims, data type
    // Canonicalize into a deterministic string, then SHA-256.
}
```

**Reuses existing infrastructure.** The test SDK already has per-node signature keys for all 14 operation types (`ConvolutionFwdSignatureKey`, `MatmulSignatureKey`, `SdpaFwdSignatureKey`, etc. in `test_sdk/.../cpu_graph_executor/detail/`). Each key extracts the operation type and data types from a node and implements `hashSelf()`. `CpuReferenceGraphExecutor` already walks nodes in topological order and builds a tensor-UID map. `computeGraphFingerprint()` reuses this graph walk and per-node key pattern. **What it adds**: tensor names, dims, and operation parameters (padding, stride, dilation) to the per-node collection, then composes into a single graph-level SHA-256.

At validation time:

```cpp
auto fingerprint = computeGraphFingerprint(graph);
if(fingerprint != manifest.graph_fingerprint)
{
    FAIL() << "Graph definition has changed since golden data was generated."
           << "\n  Golden fingerprint: " << manifest.graph_fingerprint
           << "\n  Current fingerprint: " << fingerprint
           << "\n  Regenerate golden data with --generate-golden";
}
```

**Why this is the right design:**

| Property | `to_binary()` hash | Graph-property hash |
|----------|-------------------|-------------------|
| Deterministic by construction | No (FlatBuffers gives no guarantee) | Yes (we control sort order and canonicalization) |
| Survives FlatBuffers upgrades | No (all hashes break) | Yes (no FlatBuffers dependency) |
| Survives serialization format change | No | Yes |
| Catches real `buildGraph()` changes | Yes | Yes |
| Catches changes we care about only | No (false positives on format changes) | Yes (only mathematical semantics) |

The graph-property hash changes when — and only when — the computation changes. A FlatBuffers upgrade, a schema migration, or a backend swap does not affect it. This is the correct foundation for a system where every downstream step depends on the fingerprint being right.

**What if we miss a property?** If the hash omits a property that affects computation (e.g., a new convolution attribute), the fingerprint won't catch that change. But Step 7 (value comparison) will — the numerical output will differ. The fingerprint is the first gate, not the only one. Missing a property produces a confusing error message (numerical mismatch instead of "graph changed"), not a silent pass.

#### Graph Export -- eliminates dual-spec

Instead of the Python script independently defining operation parameters, it reads graph properties exported from `buildGraph()`. The exported graph is a **transient artifact** — it is consumed by the Python script during generation and discarded. It is not stored in the golden data directory.

```bash
# 1. Export the graph definition from C++ (transient, not stored)
./hipdnn_integration_tests \
  --export-graph \
  --output-dir /tmp/graph_exports/ \
  --gtest_filter="*SdpaFwd*Smoke*"

# 2. Python reads the exported graph, extracts params, runs PyTorch
python scripts/generate_reference.py \
  --graph-export /tmp/graph_exports/SdpaFwd_Smoke/ \
  --output-dir ./pytorch_outputs/

# 3. Feed outputs into golden data generator
./hipdnn_integration_tests \
  --generate-golden \
  --external-reference ./pytorch_outputs/ \
  --golden-data-dir ./golden_data \
  --gtest_filter="*SdpaFwd*Smoke*"
```

`buildGraph()` in C++ is the **single source of truth**. The Python script never independently defines parameters — it reads them from the export. The exact export format (FlatBuffers binary, JSON, or extending the manifest with a `graph_properties` section) is an implementation detail to decide when the first external reference script is written.

**Why the serialized graph is not stored in golden data.** The validation pipeline never reads it — it rebuilds the graph from `buildGraph()` and checks the fingerprint. Storing it would couple golden data to the FlatBuffers schema version: a schema upgrade would make every stored `graph.bin` unreadable, forcing regeneration even when the computation hasn't changed. The manifest and tensor files have no such coupling.

#### Cross-Validation -- catches implementation bugs

When a C++ reference executor IS available for an operation, a CI health check generates golden data from both sources and confirms they agree:

```bash
# Generate from C++ CPU reference
./hipdnn_integration_tests --generate-golden --reference-executor cpu \
  --golden-data-dir ./golden_cpu/ --gtest_filter="*SdpaFwd*Smoke*"

# Generate from Python external reference
./hipdnn_integration_tests --generate-golden --external-reference ./pytorch_outputs/ \
  --golden-data-dir ./golden_ext/ --gtest_filter="*SdpaFwd*Smoke*"

# Compare the two sets of golden outputs
python scripts/compare_golden_sets.py ./golden_cpu/ ./golden_ext/
```

If both produce the same outputs (within tolerance), the external reference is validated. If they disagree, investigation is needed before trusting either.

#### Summary

| Check | Phase | What it catches | How |
|-------|-------|----------------|-----|
| Graph fingerprint | Phase 1 | C++ graph changed after golden data was generated | Hash comparison at validation time |
| Graph export | Phase 2 | Python and C++ disagree on operation parameters | Single source of truth (flatbuffer) |
| Cross-validation | Phase 2 | Python implementation bug (correct params, wrong math) | Generate from both sources, compare outputs |

**Acceptance criteria**:
- [ ] `--reference-executor cpu` calls `CpuReferenceGraphExecutor` (default)
- [ ] `--reference-executor gpu` calls `GpuReferenceGraphExecutor`
- [ ] `--external-reference <dir>` reads raw binary files by tensor name, skips executor
- [ ] External reference with missing output tensor file (e.g., `Y.bin` absent): **hard FAIL** naming the missing file
- [ ] `reference_executor` field written to manifest for all three sources (`cpu`, `gpu`, `external`)
- [ ] Generator validates reference outputs contain no NaN/Inf before writing

---

## Step 3: serialize -- Save Inputs and Outputs to Disk

**What**: After the reference executor produces outputs, write all tensor data (inputs AND outputs) plus metadata to disk.

**Who calls it**: Only the generation pipeline.

**This is where format decisions live.** The question "JSON or binary?" is answered here.

### Golden Data Format

Golden data uses a **manifest + binary blobs** format. Each test case produces a directory:

```
a3f8c2e1/
  manifest.json          # Metadata, tensor map, checksums
  tensor_X.bin           # Raw binary tensor data (input)
  tensor_W.bin           # Raw binary tensor data (input)
  tensor_Y.bin           # Raw binary tensor data (reference output)
```

#### Manifest Format

```json
{
  "format_version": 1,
  "metadata": {
    "generator_version": "1.0.0",
    "created_at": "2026-05-04T18:00:00Z",
    "gpu_architecture": "gfx942",
    "rocm_version": "6.4.0",
    "reference_executor": "cpu",
    "reference_executor_hash": "a3f8c2e1",
    "operation": "conv_fwd",
    "seed": 42
  },
  "graph_fingerprint": "a3f8c2e1d4b7...",
  "inputs": {
    "X": {
      "file": "tensor_X.bin",
      "dims": [1, 16, 16, 16],
      "data_type": "FLOAT",
      "sha256": "a1b2c3d4e5f6..."
    },
    "W": {
      "file": "tensor_W.bin",
      "dims": [16, 16, 3, 3],
      "data_type": "FLOAT",
      "sha256": "f6e5d4c3b2a1..."
    }
  },
  "outputs": {
    "Y": {
      "file": "tensor_Y.bin",
      "dims": [1, 16, 16, 16],
      "data_type": "FLOAT",
      "sha256": "1a2b3c4d5e6f..."
    }
  }
}
```

#### Why Not JSON + Base64?

Jeremy's original pattern uses JSON with embedded tensor data. For small tensors this works, but a single `8x512x64x64` fp32 tensor is 64 MB raw. Base64 adds 33% overhead and JSON parsing becomes the bottleneck. The binary blob format:

- Stores tensors at raw size (no encoding overhead)
- Enables memory-mapped reads for large tensors
- Keeps the manifest human-readable for inspection and debugging
- Adds ~3 lines of I/O code compared to all-JSON

#### Extensibility to Binary

Start with JSON manifest for velocity. The reader/writer are behind abstract interfaces (`GoldenDataReader` / `GoldenDataWriter`), making a future binary manifest format a drop-in replacement:

```cpp
// Scaffold: abstract reader interface
class IGoldenDataReader
{
public:
    virtual ~IGoldenDataReader() = default;
    virtual GoldenManifest loadManifest(const std::filesystem::path& dir) = 0;
    virtual std::unique_ptr<ITensor> loadTensor(const GoldenTensorInfo& info,
                                                 const std::filesystem::path& dir) = 0;
};

// Phase 1: JSON implementation
class JsonGoldenDataReader : public IGoldenDataReader { ... };
// Future: binary implementation (if needed for velocity at scale)
```

#### Integrity Verification (SHA-256)

Every golden file is integrity-checked before use. This catches partial downloads, disk corruption, and storage-side bit flips:

```cpp
void verifyGoldenIntegrity(const GoldenManifest& manifest,
                           const std::filesystem::path& directory)
{
    auto verifyFile = [&](const std::string& file, const std::string& expectedHash) {
        auto path = directory / file;
        if(!std::filesystem::exists(path))
        {
            FAIL() << "Golden data file missing: " << path;
        }
        auto actualHash = computeSha256(path);
        if(actualHash != expectedHash)
        {
            FAIL() << "Golden data integrity check failed for " << path
                   << "\n  Expected SHA-256: " << expectedHash
                   << "\n  Actual SHA-256:   " << actualHash
                   << "\n  File may be corrupted. Re-fetch golden data from storage.";
        }
    };

    for(const auto& [name, info] : manifest.inputs)
        verifyFile(info.file, info.sha256);
    for(const auto& [name, info] : manifest.outputs)
        verifyFile(info.file, info.sha256);
}
```

`computeSha256()` takes either a byte buffer (for the graph fingerprint) or a file path (for integrity checks) and returns a lowercase hex SHA-256 string. Implementation uses OpenSSL's EVP interface, which is already a ROCm build dependency.

#### Versioning

The manifest stores `format_version`, `generator_version`, and `reference_executor_hash`. The loader refuses to read manifests with an unrecognized `format_version` -- this prevents silently misinterpreting a future format as the current one. `generator_version` and `reference_executor_hash` are for diagnostics and forensics only; no comparison logic is built on them. When the reference executor changes, regenerate golden data. "Truth should be truth."

#### Stride Safety

Strides are **not stored** in the manifest. `buildGraph()` produces strides from dims deterministically, so if golden dims match the graph's dims at load time, strides are guaranteed to match. Storing strides would create a second source of truth to maintain with no safety benefit. The primary safety mechanism is **dim validation at load time**: dims mismatch → hard FAIL before any comparison.

**Acceptance criteria** (for serialize):
- [ ] Tensor data stored as raw binary `.bin` files (no encoding overhead)
- [ ] Manifest is JSON (human-readable, < 1 KB per test case)
- [ ] Generator computes SHA-256 at write time and writes it to manifest
- [ ] `writeTensorBin()` writes raw bytes with no transformation
- [ ] Memory usage for loading a 64 MB tensor: < 128 MB (raw + GPU copy)
- [ ] Reader/writer behind abstract interfaces for future format extensibility
- [ ] Dims stored in manifest; dim mismatch at load time is hard FAIL
- [ ] Unrecognized `format_version` in manifest: hard FAIL (never silently misinterpret a future format)

**Acceptance criteria** (for integrity):
- [ ] Every `.bin` file has a `sha256` field in the manifest
- [ ] `verifyGoldenIntegrity()` runs before any tensor comparison
- [ ] Missing file: **hard FAIL** with file path and suggestion to re-fetch from storage
- [ ] Hash mismatch: **hard FAIL** with expected vs actual hash
- [ ] Unit test: Corrupt a tensor file (truncate by 1 byte), verify clear failure message
- [ ] Unit test: Delete a tensor file, verify clear failure message
- [ ] Unit test: Valid golden data, verify integrity check passes silently

---

## Step 4: deserialize -- Load Inputs Back

**What**: Read saved input tensors from golden data files and populate the GPU bundle with them.

**Who calls it**: The validation pipeline. During golden mode, inputs come from disk instead of being randomly generated.

**What this step replaces**: In computed mode, `initializeBundle()` fills tensors with random data using a seed. In golden mode, `resolveGoldenTensors()` fills tensors from binary files.

**How tensor matching works**: The golden manifest stores tensors by name (e.g., `"X"`, `"W"`). The graph has tensors with UIDs. Step 4 builds a name-to-UID map by visiting graph nodes, then loads each golden tensor into the bundle slot corresponding to its UID. (See [Tensor Identity Contract](#tensor-identity-contract) in Step 1.)

**Before loading tensors, verify the graph hasn't changed** (graph fingerprint check):

The first thing Step 4 does is compute the current graph's fingerprint (a hash of graph properties — tensor names, dims, types, operation parameters) and compare it to the fingerprint stored in the manifest. If they differ, the graph definition has changed since golden data was generated — the tensor data is stale and cannot be trusted. This is a hard FAIL with a message to regenerate. See [Graph Fingerprint](#graph-fingerprint----catches-graph-drift-after-generation) for the full design.

**What can go wrong** (from pen test):
- Graph definition changed since generation → hard FAIL on graph fingerprint mismatch
- Golden file references a tensor name not in the graph → hard FAIL (Step 1 contract)
- Binary file is shorter than expected (dims * element_size) → hard FAIL on short read
- Dims in manifest don't match dims from `buildGraph()` → hard FAIL with shape diagnostic
- Data type mismatch → hard FAIL before comparison
- All-zeros or all-NaN in golden data → detected by NaN/Inf checks in generator (Step 3)

**Acceptance criteria**:
- [ ] Graph fingerprint check runs before any tensor loading
- [ ] Graph fingerprint mismatch: hard FAIL with both hashes and regeneration suggestion
- [ ] `loadTensorFromBin()` reads raw bytes and casts to appropriate type per manifest `data_type`
- [ ] Short read: hard FAIL with expected vs actual byte count
- [ ] Dim mismatch between manifest and runtime graph: hard FAIL naming both shapes
- [ ] All golden input tensors loaded before execution begins (no partial loads)

---

## Step 5: execute-engine -- Run the Engine Under Test

**What**: Execute the MIOpen GPU engine on the inputs loaded from golden data.

**Who calls it**: The validation pipeline.

**What exists today**: `executeGpuGraph()` in `IntegrationGraphVerificationHarness.hpp`:

```cpp
void executeGpuGraph(hipdnnHandle_t handle,
                     graph::Graph& graph,
                     GraphTensorBundle& bundle)
{
    int64_t workspaceSize;
    auto result = graph.get_workspace_size(workspaceSize);
    ASSERT_EQ(result.code, ErrorCode::OK);
    Workspace workspace(static_cast<size_t>(workspaceSize));

    auto variantPack = bundle.toDeviceVariantPack();
    result = graph.execute(handle, variantPack, workspace.get());
    ASSERT_EQ(result.code, ErrorCode::OK);
}
```

**What changes**: Nothing to the execution code itself. The golden path calls it exactly as the computed path does. The only difference is where the inputs came from: random fill (computed) vs disk (golden).

**Design bug found during pen test**: Engine support check (engine ID lookup, `create_execution_plans()`, `check_support()`, `build_plans()`) currently lives inside `verifyGraphComputed()`. The golden path needs these same checks. Fix: extract engine support check into a shared method called by both the dispatcher and `verifyGraphGolden()`:

```cpp
// Shared engine setup -- called before BOTH computed and golden paths
void prepareEngine(graph::Graph& graph)
{
    // Engine support check (existing code from verifyGraph lines 96-137)
    // create_execution_plans + check_support + build_plans (lines 141-146)
}
```

### Architecture Guard

Golden data generated by a GPU reference executor is only valid on the architecture that generated it. Golden data generated by the CPU reference executor is architecture-independent.

```cpp
void checkArchitectureCompatibility(const GoldenMetadata& metadata)
{
    if(metadata.reference_executor == "cpu")
    {
        // CPU-generated golden data is architecture-independent
        return;
    }

    // GPU-generated golden data: architecture must match
    std::string currentArch = getCurrentGpuArchitecture();
    if(currentArch != metadata.gpu_architecture)
    {
        GTEST_SKIP() << "Golden data was generated on " << metadata.gpu_architecture
                     << " but current GPU is " << currentArch
                     << ". GPU-generated golden data is architecture-specific.";
    }
}
```

**Acceptance criteria**:
- [ ] GPU-generated golden data on architecture mismatch: **GTEST_SKIP** naming both architectures
- [ ] CPU-generated golden data: no architecture check, runs everywhere
- [ ] Engine support check shared between computed and golden paths
- [ ] `prepareEngine()` runs before any GPU execution in both paths

---

## Step 6: deserialize -- Load Saved Outputs

**What**: Read saved reference output tensors from golden data files for comparison.

**Who calls it**: The validation pipeline, after Step 5 completes.

**This is distinct from Step 4**: Step 4 loads inputs (to feed the engine). Step 6 loads outputs (to compare against the engine's results). They use the same `loadTensorFromBin()` function but at different points in the pipeline.

**What can go wrong**:
- Same failure modes as Step 4 (missing files, short reads, dim mismatches)
- Golden output shape doesn't match engine output shape → hard FAIL (detected by dim validation)

---

## Step 7: validate -- Compare Engine Output to Saved Output

**What**: Compare the engine's output (from Step 5) against the golden reference output (from Step 6) using the harness's tolerance framework.

**Who calls it**: The validation pipeline.

### Tolerance: Single Source of Truth

Tolerances are **always computed by the harness** via `getTolerance()` / `registerValidator()`. The golden manifest does NOT store tolerance values. This eliminates dual-source-of-truth bugs where the harness formula changes but golden data retains the old tolerance.

The validation step uses the same `_tensorIdToValidatorMap` as the computed path. `registerValidator()` is called during `runGraphTest()` before `verifyGraph()`, so the validators are available for both computed and golden paths.

**Diagnostic output on failure**:

```cpp
if(!valid)
{
    auto stats = computeMismatchStats(*goldenTensor, *gpuTensor);
    FAIL() << "Golden reference mismatch for tensor '" << name << "'"
           << "\n  Max absolute error: " << stats.maxAbsError
           << "\n  Max relative error: " << stats.maxRelError
           << "\n  Mismatched elements: " << stats.mismatchCount
           << " / " << stats.totalElements
           << "\n  Golden data from: " << manifest.metadata.created_at
           << "\n  Reference executor: " << manifest.metadata.reference_executor
           << "\n  Generator version: " << manifest.metadata.generator_version;
}
```

### Floating-Point Edge Cases

The comparison function must handle two edge cases that are mathematically correct but fail naive checks:

- **NaN == NaN**: If both golden and engine output NaN at the same position, treat them as matching (IEEE `NaN != NaN` would otherwise reject two equally-correct outputs). If only one side is NaN, that is a hard FAIL.
- **-0.0 vs +0.0**: Mathematically equal, bitwise different. The comparator uses value comparison, not bitwise comparison. Note that SHA-256 integrity checks use bitwise comparison on the raw file -- this is correct because integrity checks verify "same bytes on disk," not "same mathematical value."

**Acceptance criteria**:
- [ ] Golden manifest contains no tolerance fields
- [ ] `verifyGraphGolden()` uses `_tensorIdToValidatorMap` (same as computed path)
- [ ] `registerValidator()` is called before golden validation, same as computed path
- [ ] Changing `toleranceForNodeTyped()` takes effect immediately for both modes
- [ ] Failure message includes: tensor name, max errors, mismatch count, golden metadata
- [ ] Both golden and engine NaN at same position: treated as match
- [ ] One side NaN, other side finite: hard FAIL

---

## Verification Modes

The harness gains a `VerificationMode` that controls which pipeline runs:

```cpp
// src/harness/TestConfig.hpp

enum class VerificationMode
{
    COMPUTED,  // CPU/GPU reference executor (existing behavior, default)
    GOLDEN,   // Pre-computed golden data from disk
    BOTH,     // Run both; computed is authoritative for pass/fail
};
```

| Mode | Steps executed | Reference Source |
|------|---------------|-----------------|
| `computed` | 1, random fill, 5 (CPU ref), 5 (GPU engine), 7 | CPU/GPU reference executor at runtime |
| `golden` | 1, 4, 5, 6, 7 | Pre-computed data from disk |
| `both` | Both pipelines | Both; computed is authoritative |

### `both` Mode Failure Semantics

When `both` mode is active, the computed result is always authoritative. Golden comparison is advisory:

| Computed | Golden | Test Result | Action |
|----------|--------|-------------|--------|
| PASS | PASS | **PASS** | None |
| PASS | FAIL | **PASS** | Emit warning: golden data may be stale |
| FAIL | PASS | **FAIL** | Engine regression; golden data confirms old behavior worked |
| FAIL | FAIL | **FAIL** | Engine regression confirmed by both methods |

This ensures `both` mode never blocks merges due to stale golden data, while still providing signal when golden and computed disagree.

**Acceptance criteria**:
- [ ] Truth table implemented exactly as specified above
- [ ] Warning for computed-pass/golden-fail includes: golden data path, creation date, reference executor hash, and suggestion to regenerate
- [ ] `both` mode with missing golden data directory: **warning + continue** (does not fail, does not skip)
- [ ] Unit test: Each of the 4 truth table cells verified
- [ ] Unit test: `both` mode with nonexistent golden directory produces warning-only
- [ ] Integration test: Real graph in `both` mode with valid golden data produces PASS

---

## CLI and Configuration

New CLI flags added to `main.cpp`:

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--vm, --verification-mode` | `computed`, `golden`, `both` | `computed` | Selects verification strategy |
| `--gd, --golden-data-dir` | path | `<exe_dir>/../lib/hipdnn_golden_data` | Root directory for golden data |
| `--generate-golden` | flag | off | Generate golden data instead of running tests |
| `--golden-seed` | integer | 42 | Seed for golden data input generation |
| `--force-regenerate` | flag | off | Overwrite existing golden data (without this, generator refuses to clobber) |
| `--external-reference` | path | none | Directory with external reference outputs |

Environment variable fallbacks:
- `HIPDNN_TEST_VERIFICATION_MODE`
- `HIPDNN_TEST_GOLDEN_DATA_DIR`

TOML config integration (extends existing format):
```toml
[verification]
mode = "computed"                              # "computed" | "golden" | "both"
golden_data_dir = "/path/to/golden_data"       # overridden by CLI flag

[engines.MIOPEN_PLUGIN]
tolerance = "dynamic"
# expected_failures applies to all verification modes
expected_failures = [
    "IntegrationGpuConvFwd3dFp32/Smoke.Correctness/NCDHW_1x1x4x4x4_1x1x3x3x3",
]
```

`expected_failures` applies uniformly across verification modes. A test marked as expected-to-fail is expected to fail regardless of whether the reference comes from a computed executor or golden data.

### Staleness Detection

The `reference_executor_hash` field in the manifest (short git hash of the reference executor source) enables detection of stale golden data. When the current reference executor hash differs from the golden data's hash, the harness logs a warning. This is advisory -- it does not change pass/fail -- but provides a clear signal that regeneration may be needed.

**Acceptance criteria**:
- [ ] All 5 CLI flags parsed and stored in `TestConfig` singleton
- [ ] Environment variable fallbacks work when CLI flag is absent
- [ ] Generator writes `reference_executor_hash` to manifest
- [ ] `verifyGraphGolden()` logs a warning if hashes differ
- [ ] Warning does NOT change test pass/fail

---

## Harness Integration

The existing `verifyGraph()` is refactored into a dispatcher that routes to the appropriate pipeline. The public API is unchanged -- existing tests work without modification:

```cpp
template <typename DataType, typename TestCaseType>
class IntegrationGraphVerificationHarness : public ::testing::TestWithParam<TestCaseType>
{
protected:
    void verifyGraph(graph::Graph& graph, unsigned int seed)
    {
        // Generate mode: create golden data and return
        if(TestConfig::get().isGenerateGoldenMode())
        {
            generateGoldenData(graph, seed);
            return;
        }

        auto mode = TestConfig::get().getVerificationMode();

        if(mode == VerificationMode::COMPUTED || mode == VerificationMode::BOTH)
        {
            verifyGraphComputed(graph, seed);
        }

        if(mode == VerificationMode::GOLDEN || mode == VerificationMode::BOTH)
        {
            auto goldenPath = resolveGoldenPath();
            if(!std::filesystem::exists(goldenPath / "manifest.json"))
            {
                if(mode == VerificationMode::GOLDEN)
                {
                    FAIL() << "Golden data not found: " << goldenPath
                           << "\nFetch golden data from storage or generate with --generate-golden";
                }
                // BOTH mode: missing golden data is a warning, not a failure
                HIPDNN_PLUGIN_LOG_WARN(
                    "Golden data not found, skipping golden check: " << goldenPath);
                return;
            }
            verifyGraphGolden(graph, goldenPath);
        }
    }

private:
    // Existing computed verification flow, extracted verbatim from current verifyGraph()
    void verifyGraphComputed(graph::Graph& graph, unsigned int seed)
    {
        // ... existing code from IntegrationGraphVerificationHarness.hpp ...
    }

    void verifyGraphGolden(graph::Graph& graph, const std::filesystem::path& goldenDir)
    {
        // Step 4: deserialize inputs
        // Step 5: execute-engine
        // Step 6: deserialize outputs
        // Step 7: validate
    }

    void generateGoldenData(graph::Graph& graph, unsigned int seed)
    {
        // Step 1: construct (already done by caller)
        // Step 2: execute-reference
        // Step 3: serialize
    }
};
```

### Generator Flow (Steps 1-3)

```cpp
void generateGoldenData(graph::Graph& graph, unsigned int seed)
{
    auto goldenDir = resolveGoldenPath();

    // Overwrite protection: refuse to clobber existing golden data
    if(std::filesystem::exists(goldenDir / "manifest.json")
       && !TestConfig::get().forceRegenerate())
    {
        FAIL() << "Golden data already exists at: " << goldenDir
               << "\nUse --force-regenerate to overwrite.";
    }

    // Atomic write: generate into a temporary directory, then rename
    auto tmpDir = goldenDir.parent_path() / (goldenDir.filename().string() + ".tmp");
    std::filesystem::create_directories(tmpDir);

    GraphTensorBundle refBundle;
    std::vector<int64_t> outputTensorIds;
    generateBundles(graph, refBundle, refBundle, outputTensorIds);
    initializeBundle(graph, refBundle, seed);

    executeCpuGraph(graph, refBundle);  // Step 2: execute-reference

    GoldenDataWriter writer(tmpDir);
    writer.writeManifest(graph, refBundle, outputTensorIds, buildMetadata());
    writer.writeTensorBlobs(refBundle);  // Step 3: serialize

    // Atomic swap: rename only after all files are written
    std::filesystem::rename(tmpDir, goldenDir);

    std::cout << "Golden data written to: " << goldenDir << std::endl;
}
```

**Acceptance criteria**:
- [ ] `verifyGraph()` signature unchanged (zero changes to existing tests)
- [ ] `verifyGraphComputed()` is exact rename of existing `verifyGraph()` body
- [ ] `--generate-golden` calls `generateGoldenData()` which uses same `buildGraph()`, `generateBundles()`, `initializeBundle()`
- [ ] Engine support check shared between computed and golden paths (not duplicated, not missing from golden)
- [ ] Output directory structure matches manifest layout
- [ ] Generator refuses to overwrite existing golden data without `--force-regenerate`
- [ ] Generator writes to a temp directory and renames atomically (no partial golden data on disk if process crashes)
- [ ] Clang-tidy clean

---

## Data Management

### Repository Layout

```
integration-tests/
  golden_data/
    conv/
      fwd/
        a3f8c2e1/
          manifest.json
          tensor_X.bin
          tensor_W.bin
          tensor_Y.bin
        b7d4e9f0/
          manifest.json
          ...
      bwd/
        ...
    batchnorm/
      ...
    sdpa/
      fwd/
        c5a1b3d2/
          manifest.json
          ...
```

The case directory name is a short prefix of the graph fingerprint. The fingerprint uniquely identifies the graph configuration. The per-case `manifest.json` contains the full fingerprint and all graph properties in human-readable form for inspection.

**Path resolution.** `resolveGoldenPath()` computes the graph fingerprint and looks up `golden_data/<operation>/<direction>/<short_hash>/manifest.json`. A CI health check script scans the directory tree against the test executable's test list to detect orphaned or missing golden data.

**Test tiers are a selection concern.** The test framework selects which cases to run via `--gtest_filter`. If a smoke test and a full test use the same graph, they share the same golden data directory.

**Acceptance criteria**:
- [ ] `resolveGoldenPath()` uses graph fingerprint to resolve directory
- [ ] Missing `manifest.json` in resolved directory in golden mode: **hard FAIL** with suggestion to generate
- [ ] Missing `manifest.json` in resolved directory in both mode: **warning + skip golden check**
- [ ] CI health check script detects orphaned golden data (directories not matching any test) and missing golden data (tests with no directory)

### Storage Requirements

Golden data is too large for git (a single test case with large tensors can be tens of megabytes). It needs external storage that satisfies three requirements:

1. **CI-pullable**: CI jobs can fetch golden data before running tests
2. **Versioned**: Golden data versions are tied to code versions (regenerating golden data after a `buildGraph()` change should be trackable)
3. **Integrity-checked**: Partial or corrupted downloads are detected before comparison (SHA-256 in the manifest handles this regardless of storage tool)

### Storage Options

The choice of storage tool is an infrastructure decision, not an architectural one. The golden data format (manifest + binary blobs) is tool-agnostic. Any of the following work:

| Option | Pros | Cons |
|--------|------|------|
| DVC | Data versioned alongside code via `.dvc` files, backend-agnostic (S3/Azure/GCS) | Third-party tool, learning curve, CI integration |
| git-lfs | Built into git, no new tool | GitHub storage/bandwidth costs at scale |
| S3/Azure + script | No new dependency, simple | No automatic version linkage to code |
| CI artifact storage | Already in CI infrastructure | No cross-job versioning, harder to share |

This RFC does not prescribe a specific tool. The examples below use DVC for concreteness, but the workflow is the same with any tool that can push/pull files to remote storage.

### Example Workflow (DVC)

```bash
# Generate golden data (Steps 1-3)
./build/hipdnn_integration_tests \
  --generate-golden \
  --golden-data-dir ./golden_data \
  --gtest_filter="*ConvFwd*"

# Track and push
dvc add golden_data/
git add golden_data.dvc .gitignore
git commit -m "Add conv fwd golden data"
dvc push

# On another machine or in CI (Steps 4-7)
dvc pull
./hipdnn_integration_tests \
  --verification-mode golden \
  --golden-data-dir ./golden_data
```

**Acceptance criteria**:
- [ ] CI pipeline: golden data fetch step skipped entirely for `--verification-mode computed`
- [ ] CI pipeline: golden data fetch failure is **non-fatal warning** if `--verification-mode computed`
- [ ] CI pipeline: golden data fetch failure is **hard failure** if `--verification-mode golden`
- [ ] Runbook: Step-by-step for setting up storage credentials for CI and developer workstations

---

## CI Integration

### Recommended CI Strategy

| CI Stage | Verification Mode | Golden Data Required | Rationale |
|----------|-------------------|---------------------|-----------|
| Pre-submit (smoke) | `computed` | No | Fast feedback, no external storage dependency |
| Post-submit (full) | `both` | Yes | Cross-validates golden against computed |
| Nightly | `golden` | Yes | Regression gate against locked baselines |
| Weekly | `both` (all tiers) | Yes | Full cross-validation, staleness detection |

### CI Pipeline Integration

```yaml
# Excerpt from integration test CI job
- name: Pull golden reference data
  if: inputs.verification_mode != 'computed'
  run: |
    # Tool-specific fetch command (e.g., dvc pull, aws s3 sync, etc.)
    cd dnn-providers/integration-tests
    ./scripts/fetch_golden_data.sh

- name: Run integration tests
  run: |
    ./hipdnn_integration_tests \
      --verification-mode ${{ inputs.verification_mode }} \
      --golden-data-dir ./golden_data \
      --gtest_filter=${{ inputs.gtest_filter }}
```

Pre-submit jobs omit the golden data fetch step entirely, keeping them fast and independent of remote storage availability.

---

## Adding New Golden Reference Tests

### Step 1: Write the test (existing workflow)

Follow the test fixture convention. No golden-specific code is required in the test itself:

```cpp
template <typename DataType>
class MyOperation : public IntegrationGraphVerificationHarness<DataType, TestCaseType>
{
public:
    static std::pair<graph::Graph, GraphOutputs> buildGraph(
        hipdnnHandle_t handle, const TestCaseType& tc);

protected:
    void runGraphTest() override
    {
        auto [graphObj, outputs] = buildGraph(getSharedHandle(), this->GetParam());
        this->registerValidator(outputs.y, this->getTolerance(graphObj, outputs.y));
        this->verifyGraph(graphObj, seed);  // automatically routes to golden if configured
    }
};
```

### Step 2: Generate golden data (Steps 1-3 of the pipeline)

```bash
./hipdnn_integration_tests \
  --generate-golden \
  --golden-data-dir ./golden_data \
  --reference-executor cpu \
  --gtest_filter="*MyOperation*Smoke*"
```

### Step 3: Inspect the generated data

```bash
cat golden_data/myop/fwd/a3f8c2e1/manifest.json | python -m json.tool
```

Verify tensor shapes and value ranges match expectations.

### Step 4: Push to storage

```bash
# Push golden data to remote storage (tool-specific)
# Example with DVC:
dvc add golden_data/ && git add golden_data.dvc && dvc push
# Example with S3:
aws s3 sync golden_data/ s3://bucket/golden_data/
```

### Step 5: Validate with `both` mode (Steps 4-7 of the pipeline)

```bash
./hipdnn_integration_tests \
  --verification-mode both \
  --golden-data-dir ./golden_data \
  --gtest_filter="*MyOperation*Smoke*"
```

Both computed and golden validation must pass. This confirms the golden data is consistent with the current reference executor.

---

## Implementation Phases

### Phase 1: Foundation

**What ships**: The core 7-step pipeline end-to-end.

**Scope**:
1. `VerificationMode` enum and CLI flags in `TestConfig`
2. `GoldenManifest` struct with JSON parsing
3. `resolveGoldenTensors()` with name-based matching
4. `verifyGraphGolden()` in harness (steps 4-7)
5. `generateGoldenData()` in harness (steps 1-3)
6. Architecture guard
7. `both` mode truth table logic
8. Binary blob I/O (read/write) behind abstract interface
9. SHA-256 integrity verification
10. Unit tests for all acceptance criteria above

**Definition of done**:
- Round-trip test passes (generate + validate) for conv fwd fp32 smoke
- All acceptance criteria in Steps 1-7, Verification Modes, Harness Integration checked off
- Code review approved
- Clang-tidy clean
- Unit tests pass on CPU-only build
- Integration test passes on GPU machine

---

### Phase 2: Integration & Scale

**What ships**: Storage integration, CI integration, path resolution, staleness detection.

**Scope**:
1. `resolveGoldenPath()` implementation (fingerprint → directory path)
2. `reference_executor_hash` in metadata + staleness warnings
3. `--external-reference` flag for ops without reference executors
4. Storage setup and CI pipeline snippet (tool chosen by infrastructure team)
5. Storage credential runbook
6. Golden data generated for at least 10 test cases across conv fwd/bwd/wrw, batchnorm, fp32/fp16

**Definition of done**:
- All acceptance criteria in Data Management, CLI and Configuration checked off
- Storage round-trip documented and tested (generate → push → pull → validate)
- CI pipeline snippet reviewed by DevOps
- Corrupted golden data always produces a clear diagnostic

---

### Phase 3: Polish (ongoing)

**What ships**: Developer experience improvements as need arises.

**Scope**: Items from [Future Work](#future-work), picked up when the specific pain point emerges.

---

## Risk Register

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Remote storage becomes unavailable | Golden-mode CI fails | Low | Compute-mode CI is independent of storage; CI fallback to compute-only |
| Tensor naming conventions diverge across op families | Name-based matching breaks | Medium | Lint rule: all test tensors must have non-empty, unique names within a graph |
| Golden data regeneration cadence unclear | Stale data accumulates | Medium | `reference_executor_hash` provides signal; weekly `both`-mode CI catches drift |
| Large golden data sets slow down CI | CI feedback loop degrades | Low | Storage caching, selective fetch by test filter, compression (future) |
| Team unfamiliar with storage tool | Onboarding friction | Medium | Runbook in Phase 2, pair programming during first 3 onboardings |
| Engine support check missing from golden path | Silent failures or crashes | High | Design bug identified in pen test. Fix: extract `prepareEngine()` shared method |
| Shape/stride mismatch between golden data and runtime | Wrong comparison, subtle bugs | Medium | Dim validation at load time; hard FAIL on any mismatch |
| NaN/Inf in golden data goes undetected | All comparisons pass vacuously | Low | Generator validates outputs contain no NaN/Inf before writing |
| Partial download leaves truncated files | Integrity check catches it | Low | SHA-256 verification before any comparison |
| Reference executor bug frozen into golden data | Wrong truth accepted permanently | Medium | Cross-validation catches disagreements; future invariant checks provide code-independent anchors |
| Generator crash leaves partial golden data | Orphaned files, missing manifest | Low | Atomic write protocol: generate into `.tmp` dir, rename on success |
| CRLF line endings change SHA-256 on Windows | Integrity check fires on valid data | Low | `.gitattributes` forces LF for all golden data files; binary blobs are unaffected |
| Generator silently overwrites existing golden data | Good data clobbered without notice | Medium | `--force-regenerate` required to overwrite; default is hard FAIL |

---

## Quality Principles

1. **Fail loud, never fail silent**: Every failure mode produces an actionable error message. No silent passes on corrupted/stale data.
2. **Computed reference is always authoritative**: Golden is a second opinion, never the sole source of truth for pass/fail in `both` mode.
3. **Test authors should not think about golden data**: The harness handles everything. Writing a golden-validated test is identical to writing a computed-validated test.
4. **CI should work with zero golden data**: Compute-mode is always available. Golden mode is an overlay, not a dependency.
5. **Every golden file is integrity-checked**: SHA-256 before comparison, always. No exceptions.
6. **Three verbs, seven steps**: If a design decision doesn't serve serialize, deserialize, or validate, question whether it belongs.

---

## Known Limitations

This system adds a **comparison-based** verification layer. All comparison-based systems share a structural limitation: they can tell you that two things agree, not that either is correct. Three failure modes are irreducible within this RFC's scope:

1. **Correlated spec misunderstanding**: If both the reference executor and the engine under test implement the same wrong interpretation of an operation (e.g., both apply SDPA masking incorrectly in the same way), they agree, and the test passes. No amount of comparison infrastructure catches this — it requires an independent anchor derived from the mathematical definition, not from any code. Future work on invariant checks and hand-verified micro cases (computed from the spec by a human, not by any executor) addresses this.

2. **Spec ambiguity**: When the operation specification admits multiple valid interpretations (e.g., SDPA masking with `-inf` vs. a large negative number), the golden data captures one interpretation. A correct engine implementing the other interpretation will either fail (if tolerance is tight) or silently pass (if tolerance masks the difference). This requires a canonical spec document, not more verification code.

3. **Small-size coverage gap**: Hand-verified cases and smoke tests exercise small tensor sizes. Bugs that only appear at larger sizes (tile boundary conditions, padding edges, memory layout transitions) are not caught by small cases. Mitigated by ensuring test suites include boundary-straddling sizes, but never fully eliminated.

These are not flaws in the design — they are the boundary of what comparison testing can do. The RFC is designed so that future layers (invariant checking, hand-verified micro cases, cross-validation with structurally independent implementations) can plug these gaps without changing the golden data format or pipeline.

---

## Future Work

1. **Golden data inspection CLI**: A `--inspect-golden` mode that reads a golden directory and prints metadata, tensor shapes, and value statistics (min/max/mean/std) for debugging.

2. **Automated staleness detection**: A weekly CI job that compares reference executor hashes across all golden data and opens a tracking issue when mismatches are detected.

3. **Incremental generation**: Per-test-case generation that detects existing files and only generates missing ones, reducing regeneration overhead.

4. **Golden data garbage collection**: A CI job that scans the golden data directory tree against the test executable's test list to detect and clean up orphaned golden data from removed test cases.

5. **Fused graph naming convention**: The directory layout handles mixed-precision via the case directory name (e.g., `_fp16_fp32acc`). Fused graphs (conv+bias+relu) need a naming convention for the operation directory level — either a compound name (`conv_bias_relu/`) or a generic `fused/` directory with descriptive case names. Decide when these tests arrive.

6. **Compression**: Optional zstd compression of binary blobs for full-tier golden data with large tensors. The manifest would gain a `"compression": "zstd"` field.

7. **GPU non-determinism**: Some GPU operations (e.g., atomics in backward passes) are non-deterministic across runs. Golden validation for these operations may need a wider tolerance band or a deterministic execution mode flag.

8. **Binary manifest format**: If JSON parsing of manifests becomes a bottleneck at scale, swap the `IGoldenDataReader` implementation to a binary format (flatbuffer or protobuf) behind the same interface.

9. **Mathematical invariant checks**: Per-operation invariants (layernorm output mean ~0 / variance ~1, softmax rows sum to 1, batchnorm output statistics) that require no reference executor and catch bugs that comparison testing structurally cannot. These run in both pipelines — at generation time to guard what gets frozen, at validation time to gate before comparison. Separate RFC.

10. **Hand-verified micro cases**: At least one test case per operation family with expected outputs computed by hand from the mathematical definition, not by any executor. These serve as the external anchor for the entire chain — the only check that catches correlated spec misunderstandings between the reference executor and the engine. Separate RFC.

11. **Statistical masking detection**: In addition to per-element max error, check the percentage of elements outside a tighter inner tolerance. A tensor where 0.3% of elements are wrong by a small amount passes per-element max-error checks but accumulates errors in downstream fused operations.
