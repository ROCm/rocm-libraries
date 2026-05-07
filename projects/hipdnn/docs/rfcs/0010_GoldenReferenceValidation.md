# RFC 0010: Golden Reference Validation

> Owner: Integration Test Team
> Last updated: 2026-05-07

## Table of Contents
1. [Summary](#summary)
2. [Design Overview](#design-overview)
   - [Existing Infrastructure](#existing-infrastructure)
   - [The Graph IS the Key](#the-graph-is-the-key)
   - [Pipeline Overview](#pipeline-overview)
3. [Detailed Design](#detailed-design)
   - [Golden Data Format](#golden-data-format)
   - [Generic Test Runner](#generic-test-runner)
   - [Generation Pipeline (Python)](#generation-pipeline-python)
   - [Reference Sources](#reference-sources)
   - [Verification Modes](#verification-modes)
   - [Tolerance Framework](#tolerance-framework)
   - [Data Integrity](#data-integrity)
   - [Bundle Inspection Tool (v2)](#bundle-inspection-tool-v2)
   - [Harness Integration](#harness-integration)
4. [Folder Convention](#folder-convention)
5. [CLI and Configuration](#cli-and-configuration)
6. [Integration](#integration)
   - [CI Integration](#ci-integration)
   - [Adding New Golden Reference Tests](#adding-new-golden-reference-tests)
7. [Data Management](#data-management)
8. [Risk Register](#risk-register)
9. [Known Limitations](#known-limitations)
10. [Future Work](#future-work)

---

## Summary

The integration test suite validates engine outputs by computing references at runtime via [`CpuReferenceGraphExecutor`](../../test_sdk/include/hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp) or [`GpuReferenceGraphExecutor`](../../../../dnn-providers/integration-tests/src/harness/gpu_graph_executor/GpuReferenceGraphExecutor.hpp). This creates several gaps:

1. **Circular dependency risk**: If the reference executor has a bug, both sides produce the same wrong answer and the test passes
2. **Coverage gap**: Operations not yet implemented in the reference executor cannot be tested (e.g., SDPA has no C++ reference kernel)
3. **Non-determinism**: GPU reference results can vary across runs, making failure investigation harder
4. **Slowness**: CPU reference execution for large tensors is the bottleneck in full-tier tests

This RFC extends the existing golden reference pattern -- graph+tensor bundles loaded from disk and validated against engine outputs -- to cover all operation types and integrate with CI. It adds a third verification mode alongside the existing CPU and GPU reference modes:

- **Golden reference** (new) -- compare against pre-computed, version-controlled reference data from disk

A prior effort established this pattern and it is already partially working. The existing infrastructure includes graph+tensor JSON bundles at [`hipdnn_reference_data/BatchnormFwdInference/`](../../hipdnn_reference_data/BatchnormFwdInference/), CPU and GPU golden test runners ([`TestGoldenReferenceCpu`](../../test_sdk/tests/utilities/GoldenReferenceCpu.hpp), [`TestGoldenReferenceGpu`](../../../../dnn-providers/miopen-provider/tests/common/GoldenReferenceGpu.hpp)), and a Python generation framework ([`reference_data_scripts/utilities/`](../../reference_data_scripts/utilities/)). What's missing: only batchnorm has data; no CI integration; no formalized folder convention; the GPU runner has no tests; there is no coverage for convolution, matmul, SDPA, layernorm, RMS norm, reduction, or pointwise operations. This RFC fills those gaps.

### Key Benefits
- **Deterministic baselines**: Reference data is frozen from a known-good source. Unlike computed mode -- where both the reference executor and the engine can drift together -- golden mode compares against a locked baseline, so any engine change is caught
- **Unblocks testing before C++ reference kernels exist**: Generate golden data from any stable source -- PyTorch, AITER, AOTriton, Perf & benchmark team tools, or any tool that can produce the bundle format (see [Reference Sources](#reference-sources))
- **Minimal C++ to add a test case**: Adding a test case to an existing operation/layout/datatype requires zero C++ -- drop the bundle in the folder. Adding a **new** operation/layout/datatype requires a one-time `INSTANTIATE_TEST_SUITE_P` (see [Adding New Golden Reference Tests](#adding-new-golden-reference-tests))
- **Faster execution**: Eliminates runtime reference computation for large tensors

---

## Design Overview

### Existing Infrastructure

The core test-as-data infrastructure is already built and working for batchnorm. The table below summarizes each component:

| Component | File(s) | What it does |
|-----------|---------|-------------|
| Core loader | [`LoadGraphAndTensors.hpp`](../../test_sdk/include/hipdnn_test_sdk/utilities/LoadGraphAndTensors.hpp) | `loadGraphAndTensors()` reads `{Name}.json` + `{Name}.tensor{uid}.bin`, deserializes via FlatBuffers JSON parser. Returns `GraphAndTensorMap` with `extractAndClearOutputTensorData()`, `validateTensors()`, `hostBufferMap()`, `deviceBuffers()` |
| CPU golden runner | [`GoldenReferenceCpu.hpp`](../../test_sdk/tests/utilities/GoldenReferenceCpu.hpp) | gtest fixture parameterized by path. Loads graph+tensors, runs `CpuReferenceGraphExecutor`, validates. `getGoldenReferenceParams()` discovers `.json` files |
| GPU golden runner | [`GoldenReferenceGpu.hpp`](../../../../dnn-providers/miopen-provider/tests/common/GoldenReferenceGpu.hpp) | Same pattern but executes via `hipdnnEnginePluginExecuteOpGraphImpl` on GPU (defined but not yet exercised in tests) |
| Python framework | [`reference_data_scripts/utilities/`](../../reference_data_scripts/utilities/) | `Graph` (save/load JSON+bins), `TensorAttributes` (tensor I/O), `DTypeConverter`, per-operation node classes (e.g., `BatchnormInference`). Generator scripts use these to produce bundles |
| Golden data | [`hipdnn_reference_data/BatchnormFwdInference/`](../../hipdnn_reference_data/BatchnormFwdInference/) | 6 test cases across nchw/{fp32,fp16,bfp16} and ncdhw/fp32 with `Small`, `Large`, and `MIOpen` sizes |

### The Graph IS the Key

The graph JSON serialization captures **every property that determines the computation**:

- All 18 operation types have complete JSON serializers (ConvFwd/Bwd/Wrw, BatchnormInf/Train/Bwd, Pointwise all modes, Matmul, Layernorm, RMSNorm, SDPA Fwd/Bwd, Reduction, BlockScaleDequantize/Quantize)
- Per-tensor `data_type`, `dims`, `strides` are serialized
- All operation-specific parameters (padding, stride, dilation, epsilon, momentum, etc.) are serialized
- Graph-level `io_data_type`, `compute_data_type`, `intermediate_data_type` are serialized

Changing ANY computation-affecting property produces a different JSON. The saved graph JSON **IS** the unique key for the computation -- no separate fingerprint, manifest, or hash is needed. A golden data bundle (`{Name}.json` + `{Name}.tensor{uid}.bin`) is self-contained: it does not reference any C++ code, any `buildGraph()` function, or any test fixture. If the computation changes, generate a new bundle.

This is the architectural difference from a test-as-code approach: the graph definition lives on disk, not in C++. The test runner is generic -- it loads whatever graph JSON it finds and validates it. **To add a new test, you export the graph to the bundle format and drop it into the tests folder. No new C++ code is needed.** The runner discovers it automatically via `getGoldenReferenceParams()`.

### Pipeline Overview

Two pipelines operate on the same golden data format:

**Generation (Python, run once):**
1. Build graph in Python (define tensors, operation, parameters)
2. Create random input tensors
3. Run PyTorch (or another framework) to produce reference outputs
4. Call `Graph.save(base_filename)` to write `{Name}.json` + `{Name}.tensor{uid}.bin`

**Validation (C++, every CI run):**
1. `loadGraphAndTensors(path)` reads JSON + tensor bins into `GraphAndTensorMap`
2. `extractAndClearOutputTensorData()` separates reference outputs, zeros the output slots
3. Execute engine under test (CPU ref or GPU plugin)
4. `validateTensors(referenceOutputs, atol, rtol)` compares engine output to reference

#### Generation Pipeline (runs once, produces golden data)

![Generation Pipeline](images/generation_pipeline.png)

#### Validation Pipeline (runs every CI)

![Validation Pipeline](images/validation_pipeline.png)

---

## Detailed Design

### Golden Data Format

Golden data uses the existing format already established by `LoadGraphAndTensors.hpp` and `Graph.save()`. Each test case is a set of files with a shared base name:

```
{TestName}.json                    # Full graph definition (JSON)
{TestName}.tensor{uid}.bin         # Raw binary tensor data, one file per tensor UID
```

For example, the `Small` batchnorm test case in `nchw/fp32` consists of:

```
Small.json
Small.tensor0.bin    # x (input)
Small.tensor1.bin    # mean (input)
Small.tensor2.bin    # inv_variance (input)
Small.tensor3.bin    # scale (input)
Small.tensor4.bin    # bias (input)
Small.tensor5.bin    # y (output)
```

#### Graph JSON Structure

The JSON file contains the complete graph definition. Here is an example from an existing batchnorm test case:

```json
{
  "nodes": [
    {
      "inputs": {
        "x_tensor_uid": 0,
        "mean_tensor_uid": 1,
        "inv_variance_tensor_uid": 2,
        "scale_tensor_uid": 3,
        "bias_tensor_uid": 4
      },
      "outputs": { "y_tensor_uid": 5 },
      "type": "BatchnormInferenceAttributes",
      "compute_data_type": "float",
      "name": ""
    }
  ],
  "tensors": [
    { "name": "", "uid": 0, "strides": [60, 20, 5, 1],
      "dims": [2, 3, 4, 5], "data_type": "float", "virtual": false },
    { "name": "", "uid": 1, "strides": [3, 1, 1, 1],
      "dims": [1, 3, 1, 1], "data_type": "float", "virtual": false },
    ...
  ],
  "io_data_type": "float",
  "compute_data_type": "float",
  "intermediate_data_type": "float",
  "name": ""
}
```

#### Why This Format Is Sufficient

No separate manifest is needed. The graph JSON already contains:

- **Operation type** (`"type": "BatchnormInferenceAttributes"`) -- what operation to run
- **Tensor metadata** (dims, strides, data_type) -- shape and type of every tensor
- **Operation parameters** (encoded in the node's input/output UIDs and attributes) -- all computation-determining parameters
- **Graph-level types** (io_data_type, compute_data_type, intermediate_data_type) -- precision configuration
- **Tensor UIDs** -- the key linking each `.bin` file to its role in the graph

The tensor binary files are raw bytes with no encoding overhead. A `float32` tensor of shape `[2, 3, 4, 5]` is exactly `2*3*4*5*4 = 480` bytes. This avoids the 33% Base64 overhead that would be needed to embed tensor data in JSON.

#### Tensor Identity

Tensors are identified by **UID** (from the graph JSON), not by name. The file naming convention is `{TestName}.tensor{uid}.bin`. `loadGraphAndTensors()` iterates over `graph->tensors()`, reads each tensor's UID from the JSON, and loads the corresponding `.bin` file:

```cpp
for(auto attributes : *graph->tensors())
{
    auto tensorPath = basePath.string() + ".tensor"
                      + std::to_string(attributes->uid()) + ".bin";
    tensorMap[attributes->uid()] = tensorFromFileAndAttributes(tensorPath, *attributes);
}
```

UIDs are internal to each bundle -- there is no cross-bundle dependency on UID assignment. Each bundle is self-contained.

---

### Generic Test Runner

The existing test runners (`TestGoldenReferenceCpu` and `TestGoldenReferenceGpu`) already implement the generic pattern. Here is how `TestGoldenReferenceCpu` works:

```cpp
class TestGoldenReferenceCpu : public ::testing::TestWithParam<std::filesystem::path>
{
protected:
    GraphAndTensorMap _graphAndTensors;
    std::unordered_map<int64_t, std::unique_ptr<ITensor>> _referenceOutputTensors;

    void SetUp() override
    {
        const auto& path = GetParam();
        if(path.empty()) { GTEST_SKIP(); }

        // Step 1: Load graph definition + all tensor data from disk
        _graphAndTensors = loadGraphAndTensors(path);

        // Step 2: Separate output tensors (reference truth), zero the output slots
        _referenceOutputTensors = _graphAndTensors.extractAndClearOutputTensorData();
    }

    void goldenReferenceTestSuite(float absoluteTolerance, float relativeTolerance)
    {
        // Step 3: Execute CPU reference on the loaded graph + inputs
        auto tensorMap = _graphAndTensors.hostBufferMap();
        CpuReferenceGraphExecutor().execute(
            _graphAndTensors.graphBuffer.data(),
            _graphAndTensors.graphBuffer.size(), tensorMap);

        // Step 4: Compare CPU output against saved reference output
        EXPECT_TRUE(_graphAndTensors.validateTensors(
            _referenceOutputTensors, absoluteTolerance, relativeTolerance));
    }
};
```

`TestGoldenReferenceGpu` follows the same pattern but executes via `hipdnnEnginePluginExecuteOpGraphImpl` on the GPU.

Test cases are discovered automatically via `getGoldenReferenceParams()`:

```cpp
inline auto getGoldenReferenceParams(const std::filesystem::path& subDirectory)
{
    return testing::ValuesIn(filesInDirectoryWithExtReturnEmptyPathOnThrow(
        getCurrentExecutableDirectory() / "../lib/hipdnn_reference_data" / subDirectory,
        ".json"));
}
```

This scans a subdirectory of `hipdnn_reference_data/` for `.json` files and returns each file path as a gtest parameter. Adding a new test case is as simple as adding a new `.json` + `.bin` bundle to the directory.

#### What Needs to Change for Full Genericity

The current CPU runner has a hardcoded check that should be removed:

```cpp
// Current code in goldenReferenceTestSuite():
EXPECT_EQ(tensorMap.size(), 6);  // Only works for batchnorm (5 inputs + 1 output)
```

This must be removed so the runner works with any operation regardless of tensor count.

Additionally, tolerances are currently passed as arguments to `goldenReferenceTestSuite()`. For full genericity across all operations, tolerances should come from a per-operation configuration -- a lookup table keyed by operation type and data type. This can be a simple function:

```cpp
// Proposed: tolerance lookup by operation type + data type
std::pair<float, float> getToleranceForOperation(
    const std::string& operationType, const std::string& dataType);
```

#### `GraphAndTensorMap` Key Methods

The `GraphAndTensorMap` struct (defined in `LoadGraphAndTensors.hpp`) provides the core data manipulation methods used by both runners:

| Method | What it does |
|--------|-------------|
| `extractAndClearOutputTensorData()` | Moves output tensors out of the map (these are the reference truth), replaces them with zero-filled tensors (these will receive the engine's output) |
| `validateTensors(refTensors, atol, rtol)` | Per-element comparison of engine output against reference, using `CpuFpReferenceValidation::allClose()` |
| `hostBufferMap()` | Returns `{uid → raw_host_pointer}` map for CPU execution |
| `deviceBuffers()` | Returns `vector<hipdnnPluginDeviceBuffer_t>` for GPU execution |

---

### Generation Pipeline (Python)

Golden data is generated by Python scripts in [`reference_data_scripts/`](../../reference_data_scripts/), using PyTorch as the reference executor. The Python framework mirrors the C++ graph structure so that the generated JSON is directly loadable by `loadGraphAndTensors()`.

#### How It Works

1. A generator script creates `TensorAttributes` objects with random data for each input tensor
2. It constructs a node object (e.g., `BatchnormInference`) with those tensors
3. It calls `node.execute()` which runs the PyTorch equivalent operation
4. It wraps the node in a `Graph` and calls `graph.save(base_filename)`
5. `Graph.save()` writes the JSON (via `graph.as_dict()`) and binary tensor files (via `dump_data_as_binary()`)

Example from `generate_batchnorm_reference.py`:

```python
def save_batchnorm_inference_execution(x_size, io_data_type, ...):
    x = TensorAttributes.random(min_val, max_val, io_data_type, x_size)
    mean = TensorAttributes.random(min_val, max_val, intermediate_data_type, derived_sizes)
    # ... create all input tensors ...
    y = TensorAttributes.empty()

    node = BatchnormInference(x, mean, inv_variance, scale, bias, y)
    node.execute(using_gpu)  # Runs torch.nn.functional.batch_norm

    graph = Graph([node], io_data_type=io_data_type, ...)
    graph.save(base_filename)  # Writes {base_filename}.json + .tensor{uid}.bin
```

#### Writing a New Generator

To add golden data for a new operation (e.g., convolution forward):

1. **Create a node class** in `reference_data_scripts/utilities/`:
   ```python
   # reference_data_scripts/utilities/conv_forward.py
   @register_node
   class ConvForward:
       type_str = "ConvolutionForwardAttributes"

       class Input:
           def __init__(self, x: TensorAttributes, w: TensorAttributes): ...

       class Output:
           def __init__(self, y: TensorAttributes): ...

       def execute(self, using_gpu: bool):
           self.outputs.y.tensor = torch.nn.functional.conv2d(
               self.inputs.x.tensor, self.inputs.w.tensor,
               padding=self.padding, stride=self.stride, dilation=self.dilation)

       def as_dict(self): ...       # JSON serialization matching C++ schema
       @staticmethod
       def from_dict(d, tensors): ...  # Deserialization
   ```

2. **Create a generator script** in `reference_data_scripts/`:
   ```python
   # reference_data_scripts/generate_conv_reference.py
   def save_conv_forward(x_size, w_size, padding, stride, dilation, dtype, base_filename):
       x = TensorAttributes.random(-1, 1, dtype, x_size)
       w = TensorAttributes.random(-1, 1, dtype, w_size)
       y = TensorAttributes.empty()
       node = ConvForward(x, w, y, padding=padding, stride=stride, dilation=dilation)
       node.execute(using_gpu=False)
       graph = Graph([node], dtype=dtype)
       graph.save(base_filename)
   ```

3. **Run the generator** to produce data bundles:
   ```bash
   python generate_conv_reference.py \
     --base-filename ../hipdnn_reference_data/ConvFwd/nchw/fp32/Small \
     --io-type float --x-size 1 16 16 16 --w-size 16 16 3 3 \
     --padding 1 1 --stride 1 1 --dilation 1 1
   ```

4. **Commit** the generated `.json` + `.bin` files to `hipdnn_reference_data/`.

The `@register_node` decorator (from `common.py`) adds the node class to `NODE_REGISTRY`, keyed by `type_str`. This enables `Graph.from_file()` to deserialize any graph JSON back into executable Python objects.

#### Current Coverage

| Operation | Generator exists | Data bundles exist | Notes |
|-----------|:---:|:---:|-------|
| BatchnormFwdInference | Yes | Yes (6 cases) | nchw/{fp32,fp16,bfp16}, ncdhw/fp32 |
| ConvFwd / ConvBwd / ConvWrw | No | No | Needed |
| Matmul | No | No | Needed |
| SDPA Fwd / Bwd | No | No | Needed -- primary motivation for golden ref |
| Pointwise (all modes) | No | No | Needed |
| Layernorm / RMSNorm | No | No | Needed |
| Reduction | No | No | Needed |
| BatchnormTrain / BatchnormBwd | No | No | Needed |
| BlockScaleDequantize / Quantize | No | No | Needed |

---

### Reference Sources

The golden data format is **reference-source-agnostic**. Any tool that can produce a graph JSON matching the bundle schema and write the corresponding tensor `.bin` files is a valid reference source. The validation pipeline does not know or care what produced the data -- it loads tensors and compares.

| Category | Examples | When to use |
|----------|----------|------------|
| Python frameworks | PyTorch, TensorFlow, JAX | Primary path. Independent implementation, covers all ops, no C++ build required |
| In-house CPU ref | `CpuReferenceGraphExecutor` | Cross-validation against Python; fallback when no Python generator exists |
| In-house GPU ref | `GpuReferenceGraphExecutor` | When CPU ref is too slow or unavailable |
| AMD internal tools | AITER, AOTriton, Perf & benchmark team tools | Validated kernels from other AMD teams; especially useful for SDPA and fused operations |
| Third-party engines | cuDNN (via shim), oneDNN | Cross-vendor validation |

The key requirement is not **which** tool generates the data, but that the output matches the bundle schema. The canonical schema is the FlatBuffers definition at [`flatbuffers_sdk/schemas/graph.fbs`](../../flatbuffers_sdk/schemas/graph.fbs), which defines the `Graph`, `Node`, `TensorAttributes`, and `NodeAttributes` union types. The JSON must be parseable by the FlatBuffers JSON parser using this schema, and each `{Name}.tensor{uid}.bin` must contain raw bytes matching the tensor's declared `dims` and `data_type`. A generation script is a thin adapter: it translates between the source tool's API and this format.

Python/PyTorch is the **recommended starting point** because:
- It's independent of the C++ codebase (breaks the circular dependency in Problem #1)
- It covers all operation types via PyTorch's `torch.nn.functional` API
- Generation scripts are simple and auditable
- It doesn't require building the C++ project to generate test data

However, for operations where another source is more trusted (e.g., AITER's SDPA kernels, AOTriton's matmul), that source should be preferred. The format is the contract, not the tool.

---

### Verification Modes

Three modes control which test suites run:

| Mode | What runs | Reference source |
|------|-----------|-----------------|
| `computed` | Existing test-as-code tests (`IntegrationGraphVerificationHarness`) | CPU/GPU reference executor at runtime |
| `golden` | Data-driven tests (`TestGoldenReferenceCpu` / `TestGoldenReferenceGpu`) | Pre-computed data from disk |
| `both` | Both test suites | Both; both must pass |

**Key distinction**: `computed` and `golden` are **separate gtest suites**, not merged within a single test. They have different test fixtures, different parameterization, and different data sources.

- **COMPUTED** tests use `IntegrationGraphVerificationHarness`. The graph is built by `buildGraph()` in C++. Inputs are randomly generated. The CPU/GPU reference executor runs at test time. This is the existing behavior, unchanged.

- **GOLDEN** tests use `TestGoldenReferenceCpu` or `TestGoldenReferenceGpu`. The graph is loaded from JSON on disk. Inputs come from the data bundle. No `buildGraph()`. No runtime reference computation.

- **BOTH** means both test suites run in CI. They are independent -- both must pass. There is no complex truth table or advisory mode. If either suite fails, the CI fails.

#### Floating-Point Edge Cases

- **NaN in golden data is a generation error.** Golden reference tensors should never contain NaN. If the reference executor produces NaN, the generator script is wrong (bad input range, numerical overflow, or a bug in the reference operation). The [generation-time validation](#generation-time-check-output-validation-python) rejects NaN before writing. If the engine under test produces NaN where the golden reference is a finite value, that is a hard FAIL.
- **-0.0 vs +0.0**: Mathematically equal, bitwise different. The comparator uses value comparison, not bitwise comparison.

#### Architecture Note

All current golden data is generated from Python (PyTorch) or the CPU reference executor, which produce architecture-independent results. If GPU-generated golden data is introduced in the future, an architecture guard will be needed to skip tests when the current GPU does not match the generation architecture. See [Future Work](#future-work).

---

### Tolerance Framework

#### Single Source of Truth

Tolerances are **always defined in code**, never stored in the data bundle. The data bundle contains only the graph definition and tensor values. This eliminates dual-source-of-truth bugs where the code formula changes but golden data retains the old tolerance.

The current approach passes tolerances as arguments to `goldenReferenceTestSuite(atol, rtol)`, with values determined per test class:

```cpp
// Current pattern (from TestCpuFpReferenceBatchnorm.cpp):
class TestCpuBatchnormFwdInferenceGoldenReferenceNchwFp32 : public TestGoldenReferenceCpu
{
protected:
    void runTest() { goldenReferenceTestSuite(/*atol=*/1e-5, /*rtol=*/1e-5); }
};
```

**Proposed improvement**: tolerance lookup by operation type + data type, so a single generic test class can handle all operations:

```cpp
// Future: single test class for all operations
// Operation type extracted from the loaded FlatBuffers graph via node.attributes_type()
// (returns the NodeAttributes enum: PointwiseAttributes, ConvolutionFwdAttributes, etc.)
auto opType = graph->nodes()->Get(0)->attributes_type();
auto [atol, rtol] = getToleranceForOperation(opType, graph->io_data_type());
goldenReferenceTestSuite(atol, rtol);
```

The exact format of the tolerance configuration (lookup table, config file, or constexpr map) is an implementation detail. The operation type is available from the loaded graph via the FlatBuffers-generated `NodeAttributes` enum (see [`graph_generated.h`](../../flatbuffers_sdk/include/hipdnn_flatbuffers_sdk/data_objects/graph_generated.h)). The principle is: tolerances come from code keyed by operation type and data type, not from the data bundle.

**Acceptance criteria**:
- [ ] Golden data bundles contain no tolerance fields
- [ ] Changing tolerance values in code takes effect immediately for both computed and golden modes
- [ ] Failure message includes: tensor UID, max absolute error, max relative error, mismatch count

---

### Data Integrity

Key-value consistency is mostly guaranteed by construction: `Graph.save()` writes the JSON and `.bin` files from the same in-memory graph in a single call, and `loadGraphAndTensors()` reads UIDs from the JSON and loads the corresponding `.bin` files. The UIDs match because they come from the same source. Corruption can only happen after generation (partial downloads, disk errors, manual edits).

Two checks catch the real failure modes:

#### Load-Time Check: File Size Validation (C++)

Before loading tensor data, verify that the file size matches what the graph JSON declares:

```cpp
auto expectedBytes = product(attributes->dims()) * sizeOfDataType(attributes->data_type());
auto actualBytes = std::filesystem::file_size(tensorPath);
if(actualBytes != expectedBytes)
{
    FAIL() << "Tensor file size mismatch for UID " << attributes->uid()
           << "\n  Expected: " << expectedBytes << " bytes"
           << " (dims=" << formatDims(attributes->dims())
           << ", dtype=" << attributes->data_type() << ")"
           << "\n  Actual:   " << actualBytes << " bytes"
           << "\n  File:     " << tensorPath;
}
```

This catches truncated files (the most common corruption from partial downloads or crashed writes), oversized files (wrong tensor written to the wrong path), and complete mismatches (binary file from a different bundle). It's cheap — a single `stat()` call per tensor, no data reading.

`loadGraphAndTensors()` does not perform this check today. A truncated file silently produces garbage in the tail of the tensor. This must be added.

A missing `.bin` file for a UID in the JSON already causes `fillTensorFromFile()` to throw, but the error message should be improved to name the UID and suggest the bundle may be incomplete.

#### Generation-Time Check: Output Validation (Python)

`Graph.save()` must validate all output tensors before writing. This is not an opt-in per-script check — it is built into `Graph.save()` itself so that no generator can bypass it:

```python
class Graph:
    def save(self, base_filename):
        self._validate_outputs()       # Runs BEFORE any file I/O
        self._write_json(base_filename)
        self._write_tensors(base_filename)

    def _validate_outputs(self):
        for uid, tensor_attr in self.output_tensors.items():
            t = tensor_attr.tensor
            if torch.isnan(t).any():
                raise ValueError(f"Tensor UID {uid} contains NaN — check input ranges or reference op")
            if torch.isinf(t).any():
                raise ValueError(f"Tensor UID {uid} contains Inf — check input ranges or reference op")
            if t.numel() > 1 and t.std() == 0:
                raise ValueError(f"Tensor UID {uid} has zero variance (all-same values)")
```

A tensor of all NaN, all Inf, or all-same-value makes the test meaningless — everything passes within tolerance. Because the check is inside `Graph.save()`, it is impossible to write a bundle with degenerate outputs.

#### Load-Time Check: NaN/Inf Rejection (C++)

As a safety net (catches bundles generated before this check existed, or by external tools), `loadGraphAndTensors()` must also reject NaN/Inf in output tensors after loading:

```cpp
for(auto uid : outputTensorUids)
{
    auto* data = static_cast<const float*>(tensorMap.at(uid)->hostPtr());
    auto size = tensorMap.at(uid)->numElements();
    for(size_t i = 0; i < size; ++i)
    {
        if(std::isnan(data[i]) || std::isinf(data[i]))
        {
            FAIL() << "Golden output tensor UID " << uid
                   << " contains NaN/Inf at index " << i
                   << " — regenerate the bundle";
        }
    }
}
```

This is more expensive than the file-size check (it reads the data), but only runs on output tensors (not inputs) and catches the exact failure mode adickin identified: NaN in golden data means the reference was wrong.

**Acceptance criteria**:
- [ ] `loadGraphAndTensors()` validates file size before reading tensor data
- [ ] File size mismatch: hard FAIL with expected vs actual bytes, dims, data type, and file path
- [ ] Missing `.bin` file: hard FAIL naming the UID and file path
- [ ] `Graph.save()` calls `_validate_outputs()` internally — no generator can bypass it
- [ ] `loadGraphAndTensors()` rejects NaN/Inf in output tensors after loading (safety net for legacy/external bundles)
- [ ] Both checks produce actionable error messages naming the tensor UID

---

### Bundle Inspection Tool (v2)

A command-line tool that reads golden data bundles and reports their contents. Not required for v1 (the core loader and generator cover initial needs), but becomes essential as the number of operations and bundles grows.

```bash
# Inspect a single bundle
python inspect_bundle.py hipdnn_reference_data/ConvFwd/nchw/fp32/Small.json

# Output:
# Bundle: Small.json
# Operation: ConvolutionForwardAttributes
# io_data_type: float    compute_data_type: float
# Tensors:
#   UID 0  x       [1, 16, 16, 16]  float  1024 bytes  min=-0.98  max=0.99  mean=0.01  std=0.58
#   UID 1  w       [16, 16, 3, 3]   float  9216 bytes  min=-0.97  max=0.98  mean=0.00  std=0.57
#   UID 2  y (out) [1, 16, 14, 14]  float  12544 bytes min=-4.21  max=4.35  mean=0.02  std=1.23
# Integrity: OK (all file sizes match, no NaN/Inf)
```

```bash
# Validate all bundles in a directory
python inspect_bundle.py --validate hipdnn_reference_data/

# Output:
# Scanning hipdnn_reference_data/ ...
# BatchnormFwdInference/nchw/fp32/Small.json    OK
# BatchnormFwdInference/nchw/fp32/Large.json    OK
# BatchnormFwdInference/nchw/fp32/MIOpen.json   OK
# ...
# 6/6 bundles OK, 0 errors
```

The tool runs the same integrity checks described in [Data Integrity](#data-integrity) (file size validation, NaN/Inf detection, variance check) and adds human-readable metadata (tensor shapes, value statistics). It uses the existing `Graph.from_file()` and `TensorAttributes.load_data_from_binary()` from the Python framework.

**Acceptance criteria**:
- [ ] `inspect_bundle.py` reads any valid bundle and prints operation type, tensor metadata, and value statistics
- [ ] `--validate` mode checks all bundles in a directory tree, reports pass/fail per bundle
- [ ] Exit code non-zero if any bundle fails validation
- [ ] Tool reuses existing `reference_data_scripts/utilities/` framework (no duplication)

---

### Harness Integration

Two test patterns coexist in the codebase today. The long-term direction is convergence toward Pattern 2 (test-as-data) as the primary pattern, with Pattern 1 (test-as-code) maintained for backward compatibility:

#### Pattern 1: Test-as-Code (`IntegrationGraphVerificationHarness`)

Used for **computed** verification. The graph is built in C++ by `buildGraph()`. Inputs are randomly generated. The CPU/GPU reference executor runs at test time. This is the existing pattern, unchanged by this RFC.

```cpp
template <typename DataType, typename TestCaseType>
class IntegrationGraphVerificationHarness : public ::testing::TestWithParam<TestCaseType>
{
    void verifyGraph(graph::Graph& graph, unsigned int seed) { ... }
};
```

Every existing test (conv, matmul, SDPA, etc.) uses this pattern. This RFC does not modify it.

#### Pattern 2: Test-as-Data (`TestGoldenReferenceCpu` / `TestGoldenReferenceGpu`)

Used for **golden** verification. The graph is loaded from JSON on disk. No `buildGraph()`. No runtime reference computation. This is the pattern established by the existing golden reference infrastructure and extended by this RFC.

```cpp
class TestGoldenReferenceCpu : public ::testing::TestWithParam<std::filesystem::path>
{
    void SetUp() override
    {
        _graphAndTensors = loadGraphAndTensors(GetParam());
        _referenceOutputTensors = _graphAndTensors.extractAndClearOutputTensorData();
    }
    void goldenReferenceTestSuite(float atol, float rtol) { ... }
};
```

New golden tests should use Pattern 2. Existing computed tests continue using Pattern 1. Over time, as golden data coverage grows, Pattern 2 becomes the primary validation path and Pattern 1 serves as a cross-validation supplement.

#### Engine Setup for GPU Runner

The GPU runner (`TestGoldenReferenceGpu`) handles engine setup internally:

```cpp
void SetUp() override
{
    hipdnnEnginePluginCreateImpl(&_handle);
    _engineConfigBuffer = createValidEngineConfig(1).Release();
    _graphAndTensors = loadGraphAndTensors(path);
    _referenceOutputTensors = _graphAndTensors.extractAndClearOutputTensorData();
}
```

The execution creates a plugin execution context, executes the graph, marks device-modified output tensors, and validates:

```cpp
void goldenReferenceTestSuite(float atol, float rtol)
{
    hipdnnEnginePluginCreateExecutionContextImpl(_handle, &engineConfig, &opGraph, &ctx);
    hipdnnEnginePluginExecuteOpGraphImpl(_handle, ctx, nullptr, deviceBuffers.data(), ...);
    for(auto uid : _graphAndTensors.outputTensorUids)
        _graphAndTensors.tensorMap.at(uid)->markDeviceModified();
    EXPECT_TRUE(_graphAndTensors.validateTensors(_referenceOutputTensors, atol, rtol));
}
```

**Acceptance criteria**:
- [ ] Pattern 1 (computed) tests are unchanged -- zero modifications to existing test fixtures
- [ ] Pattern 2 (golden) tests work for any operation type (after removing `EXPECT_EQ(tensorMap.size(), 6)`)
- [ ] Both patterns can coexist in the same test binary
- [ ] `getGoldenReferenceParams()` discovers test cases by scanning for `.json` files

---

## Folder Convention

Golden data is organized under `hipdnn_reference_data/` with a three-level hierarchy:

```
hipdnn_reference_data/
  {Operation}/            # e.g., BatchnormFwdInference, ConvFwd, MatmulFwd
    {Layout}/             # e.g., nchw, nhwc, ncdhw
      {DataType}/         # e.g., fp32, fp16, bfp16
        {TestName}.json + {TestName}.tensor{uid}.bin
```

### Naming Conventions

| Level | Convention | Examples |
|-------|-----------|----------|
| Operation | PascalCase, direction suffix | `BatchnormFwdInference`, `ConvFwd`, `ConvBwd`, `MatmulFwd`, `SdpaFwd`, `PointwiseRelu` |
| Layout | Lowercase | `nchw`, `nhwc`, `ncdhw`, `ndhwc` |
| DataType | Lowercase abbreviation | `fp32`, `fp16`, `bfp16` |
| TestName | PascalCase, describes tensor size/source | `Small`, `Large`, `MIOpen`, `Smoke` |

### Example: Current Data

```
hipdnn_reference_data/
  BatchnormFwdInference/
    ncdhw/
      fp32/
        Small.json + Small.tensor{0..5}.bin
    nchw/
      bfp16/
        Small.json + Small.tensor{0..5}.bin
      fp16/
        Small.json + Small.tensor{0..5}.bin
      fp32/
        Small.json + Small.tensor{0..5}.bin
        Large.json + Large.tensor{0..5}.bin
        MIOpen.json + MIOpen.tensor{0..5}.bin
```

### How Test Discovery Works

`getGoldenReferenceParams("BatchnormFwdInference/nchw/fp32")` scans the directory for `.json` files and returns each file path as a gtest parameter. Each `.json` file becomes a separate test case.

To add a new test case to an existing operation/layout/datatype, drop a new `.json` + `.bin` bundle into the directory. The next test run picks it up automatically.

### gtest Filtering

Because test names include the file path, standard gtest filtering works:

```bash
# Run all batchnorm golden tests
./test_binary --gtest_filter="*BatchnormFwd*"

# Run only fp32 nchw batchnorm golden tests
./test_binary --gtest_filter="*nchw*fp32*"
```

### Adding a New Operation

1. Create the folder hierarchy: `hipdnn_reference_data/{Operation}/{Layout}/{DataType}/`
2. Generate data bundles using a Python script (see [Generation Pipeline](#generation-pipeline-python))
3. Add C++ test instantiation (see [Adding New Golden Reference Tests](#adding-new-golden-reference-tests))

---

## CLI and Configuration

### CLI Flags

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--vm, --verification-mode` | `computed`, `golden`, `both` | `computed` | Controls which test suites run |
| `--gd, --golden-data-dir` | path | `<exe_dir>/../lib/hipdnn_reference_data` | Root directory for golden data |

### Environment Variable Fallbacks

- `HIPDNN_TEST_VERIFICATION_MODE` -- overridden by `--verification-mode` CLI flag
- `HIPDNN_TEST_GOLDEN_DATA_DIR` -- overridden by `--golden-data-dir` CLI flag

Generation is performed by Python scripts (see [Generation Pipeline](#generation-pipeline-python)), not by the C++ test binary. The test binary is purely a consumer of golden data.

#### Implementation

Each golden test fixture checks the verification mode in `SetUp()` and skips itself if disabled:

```cpp
void SetUp() override
{
    if(TestConfig::instance().verificationMode() == VerificationMode::Computed)
    {
        GTEST_SKIP() << "Golden tests disabled (--verification-mode=computed)";
    }
    // ... load graph and tensors ...
}
```

Computed test fixtures use the same pattern in reverse, skipping when mode is `golden`. In `both` mode, neither fixture skips -- both suites run.

**Acceptance criteria**:
- [ ] Both CLI flags parsed and stored in `TestConfig` singleton
- [ ] Environment variable fallbacks work when CLI flag is absent
- [ ] `--verification-mode golden` with missing golden data directory: hard FAIL with path and suggestion
- [ ] `--verification-mode computed` ignores golden data entirely (no fetch, no directory check)

---

## Integration

### CI Integration

#### Recommended CI Strategy

| CI Stage | Verification Mode | Golden Data Required | Rationale |
|----------|-------------------|---------------------|-----------|
| Pre-submit (smoke) | `computed` | No | Fast feedback, no external storage dependency |
| Post-submit (full) | `both` | Yes | Cross-validates golden against computed |
| Nightly | `golden` | Yes | Regression gate against locked baselines |

#### CI Pipeline Integration

```yaml
# Excerpt from integration test CI job
- name: Pull golden reference data
  if: inputs.verification_mode != 'computed'
  run: |
    # Tool-specific fetch command (e.g., dvc pull, aws s3 sync, etc.)
    cd projects/hipdnn
    ./scripts/fetch_golden_data.sh

- name: Run integration tests
  run: |
    ./hipdnn_integration_tests \
      --verification-mode ${{ inputs.verification_mode }} \
      --golden-data-dir ${{ github.workspace }}/projects/hipdnn/hipdnn_reference_data \
      --gtest_filter=${{ inputs.gtest_filter }}
```

Pre-submit jobs omit the golden data fetch step entirely, keeping them fast and independent of remote storage availability.

### Adding New Golden Reference Tests

Adding a test case to an **existing** operation/layout/datatype requires zero C++ changes: generate the data bundle and drop it into the folder. The runner discovers it on the next run.

Adding a **new** operation requires a one-time C++ `INSTANTIATE_TEST_SUITE_P` (Step 4 below).

#### Step 1: Generate the data bundle

Use an existing generation script, or write a new one following the pattern in `generate_batchnorm_reference.py` (see [Writing a New Generator](#writing-a-new-generator)). Any tool that produces a valid bundle (matching the schema in [Golden Data Format](#golden-data-format)) works -- PyTorch, AITER, AOTriton, or any other stable reference source.

#### Step 2: Run the generator

```bash
cd reference_data_scripts/
python generate_conv_reference.py \
  --base-filename ../hipdnn_reference_data/ConvFwd/nchw/fp32/Small \
  --io-type float --x-size 1 16 16 16 --w-size 16 16 3 3 \
  --padding 1 1 --stride 1 1 --dilation 1 1
```

#### Step 3: Commit the bundle to `hipdnn_reference_data/`

```bash
git add hipdnn_reference_data/ConvFwd/nchw/fp32/Small.*
git commit -m "Add conv fwd golden reference data (nchw fp32 Small)"
```

For large tensor data, use git-lfs, DVC, or another storage solution (see [Data Management](#data-management)).

#### Step 4: Add C++ test instantiation (new operations only)

If this is the first golden test for a new operation, add a test instantiation. If the operation already has golden tests (e.g., adding a new size to `BatchnormFwdInference/nchw/fp32/`), skip this step -- the runner discovers the new bundle automatically.

```cpp
// In a test .cpp file (CPU runner):
using TestConvFwdGoldenFp32 = hipdnn_test_sdk::utilities::TestGoldenReferenceCpu;

TEST_P(TestConvFwdGoldenFp32, Correctness)
{
    goldenReferenceTestSuite(/*atol=*/1e-5, /*rtol=*/1e-5);
}

INSTANTIATE_TEST_SUITE_P(,
    TestConvFwdGoldenFp32,
    hipdnn_test_sdk::utilities::getGoldenReferenceParams("ConvFwd/nchw/fp32"));
```

For GPU golden tests, use `TestGoldenReferenceGpu` instead:

```cpp
// GPU runner:
using TestConvFwdGoldenGpuFp32 = test_helpers::TestGoldenReferenceGpu;

TEST_P(TestConvFwdGoldenGpuFp32, Correctness)
{
    goldenReferenceTestSuite(/*atol=*/1e-5, /*rtol=*/1e-5);
}

INSTANTIATE_TEST_SUITE_P(,
    TestConvFwdGoldenGpuFp32,
    test_helpers::getGoldenReferenceParams("ConvFwd/nchw/fp32"));
```

#### Step 5: Verify

```bash
# Run the new golden tests
./test_binary --gtest_filter="*ConvFwdGolden*"
```

---

## Data Management

### Repository Layout

Golden data lives in `hipdnn_reference_data/` at the project root, following the [Folder Convention](#folder-convention):

```
projects/hipdnn/
  hipdnn_reference_data/
    BatchnormFwdInference/
      nchw/
        fp32/
          Small.json + .tensor{0..5}.bin
          Large.json + .tensor{0..5}.bin
          MIOpen.json + .tensor{0..5}.bin
        fp16/
          Small.json + .tensor{0..5}.bin
        bfp16/
          Small.json + .tensor{0..5}.bin
      ncdhw/
        fp32/
          Small.json + .tensor{0..5}.bin
    ConvFwd/
      nchw/
        fp32/
          ...
    ...
```

At install time, golden data is placed at `<exe_dir>/../lib/hipdnn_reference_data/`, which is where `getGoldenReferenceParams()` looks by default.

### Storage Options

Golden data can grow large (a single test case with `8x512x64x64` fp32 tensors is ~64 MB). For large datasets, external storage is needed. The golden data format is tool-agnostic -- any of the following work:

| Option | Pros | Cons |
|--------|------|------|
| git (small data) | Simplest, no extra tools | Only practical for small tensors |
| git-lfs | Built into git, no new tool | GitHub storage/bandwidth costs at scale |
| DVC | Data versioned alongside code, backend-agnostic (S3/Azure/GCS) | Third-party tool, learning curve |
| S3/Azure + script | No new dependency, simple | No automatic version linkage to code |

This RFC does not prescribe a specific storage tool. The existing batchnorm data (small tensors) is committed directly to git.

**Acceptance criteria**:
- [ ] CI pipeline: golden data fetch step skipped entirely for `--verification-mode computed`
- [ ] CI pipeline: golden data fetch failure is **hard failure** if `--verification-mode golden`

---

## Risk Register

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| FlatBuffers schema change | Old JSON bundles unreadable by `loadGraphAndTensors()` | Medium | Regenerate from Python scripts (Python framework is schema-independent) |
| Reference script bug freezes wrong data | Silent incorrect baseline | Medium | Cross-validate against C++ CPU ref; review generated data before committing; [generation-time validation](#generation-time-check-output-validation-python) rejects degenerate outputs |
| PyTorch version drift | Different versions produce slightly different outputs | Low | Pin PyTorch version in `requirements.txt`; regenerate when upgrading |
| Large golden data sets slow CI | CI feedback loop degrades | Low | Storage caching, selective fetch by test filter, compression (future) |
| Remote storage unavailable | Golden-mode CI fails | Low | Computed-mode CI is independent of storage; CI fallback to computed-only |

---

## Known Limitations

Comparison testing can confirm that two implementations agree, not that either is correct. If the reference executor and the engine under test share the same bug, the test passes. Future work on mathematical invariant checks and hand-verified micro cases addresses this gap without changing the golden data format.

---

## Future Work

1. **Per-operation tolerance configuration**: A structured configuration mapping `(operation_type, data_type) → (atol, rtol)` so the generic test runner doesn't need per-operation test classes.

2. **Automatic test discovery**: Recursive scanning of `hipdnn_reference_data/` to auto-generate test instantiations, eliminating the need to manually write `INSTANTIATE_TEST_SUITE_P` for each operation/layout/datatype combination.

3. **Cross-validation within `IntegrationGraphVerificationHarness`**: Add a golden mode to Pattern 1 (test-as-code) that loads golden data for the same graph built by `buildGraph()` and compares both computed and golden results in a single test.

4. **C++ graph export**: A utility to export a graph built by `buildGraph()` in an existing test-as-code test to the golden data bundle format (`{Name}.json` + `{Name}.tensor{uid}.bin`). This would let teams convert existing computed tests into golden test cases without rewriting the graph in Python.

5. **Compression**: Optional zstd compression of binary blobs for full-tier golden data with large tensors.

6. **Mathematical invariant checks**: Per-operation invariants (layernorm output mean ~0 / variance ~1, softmax rows sum to 1, batchnorm output statistics) that require no reference executor and catch bugs that comparison testing structurally cannot. Separate RFC.

7. **Hand-verified micro cases**: At least one test case per operation family with expected outputs computed by hand from the mathematical definition, not by any executor. Separate RFC.
