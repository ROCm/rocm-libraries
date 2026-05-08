# RFC 0010: Golden Reference Validation

> Owner: Integration Test Team
> Last updated: 2026-05-08

## Table of Contents
1. [Summary](#summary)
2. [Design Overview](#design-overview)
3. [Detailed Design](#detailed-design)
   - [Existing Infrastructure](#existing-infrastructure)
   - [Self-Contained Bundles](#self-contained-bundles)
   - [Golden Data Format](#golden-data-format)
   - [Generic Test Runner](#generic-test-runner)
   - [Generation Pipeline](#generation-pipeline)
   - [Reference Sources](#reference-sources)
   - [Verification Modes](#verification-modes)
   - [Tolerance Framework](#tolerance-framework)
   - [Data Integrity](#data-integrity)
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

The integration test suite validates engine outputs by computing references at runtime. This creates several gaps:

1. **Circular dependency risk**: If the reference executor has a bug, both sides produce the same wrong answer and the test passes
2. **Coverage gap**: Operations not yet implemented in the reference executor cannot be tested (e.g., SDPA has no C++ reference kernel)
3. **Non-determinism**: GPU reference results can vary across runs, making failure investigation harder
4. **Slowness**: CPU reference execution for large tensors is the bottleneck in full-tier tests

A prior effort established a golden reference pattern -- golden data bundles (graph JSON + tensor `.bin` files) loaded from disk and validated against engine outputs. The initial infrastructure is in place for batchnorm. This RFC extends golden data coverage to all operation types, formalizes the folder convention, adds data integrity checks, and integrates with CI.

---

## Design Overview

Golden reference validation uses two pipelines -- [**generation**](#generation-pipeline) and [**validation**](#generic-test-runner) -- that share a common data format: the **golden data bundle** (`{Name}.json` + `{Name}.tensor{uid}.bin`). A bundle is a self-contained test case: the graph JSON defines the computation, the `.bin` files carry the tensor data (inputs and expected outputs).

The graph JSON is a complete computation description — operation type, tensor shapes, data types, and all operation parameters. Generation produces bundles. Validation loads a bundle — graph from `{Name}.json`, tensor data from `{Name}.tensor{uid}.bin` — executes the graph through the engine under test, and compares the result to the expected output. The test fixture determines which engine runs the graph (CPU reference, MIOpen GPU plugin, etc.); the bundle itself is engine-agnostic.

**[Generation](#generation-pipeline) (run once, any tool):**
1. Define graph and create input tensors
2. Run a reference source (PyTorch, CPU ref, etc.) to produce outputs
3. Write the bundle

**[Validation](#generic-test-runner) (C++, every CI run):**
1. Load bundle from disk, extract golden outputs
2. Execute engine under test
3. Compare engine output to golden output — PASS or FAIL

---

## Detailed Design

### Existing Infrastructure

The golden reference infrastructure is already built and working for batchnorm. What this RFC adds: golden data for operations beyond batchnorm; a formalized folder convention; data integrity checks; a CI strategy with explicit verification modes and golden data fetching; and GPU runner test instantiations. The table below summarizes each existing component:

| Component | File(s) | Role |
|-----------|---------|------|
| Core loader | [`LoadGraphAndTensors.hpp`](../../test_sdk/include/hipdnn_test_sdk/utilities/LoadGraphAndTensors.hpp) | Reads bundles from disk, separates inputs from expected outputs, validates results |
| CPU runner | [`GoldenReferenceCpu.hpp`](../../test_sdk/tests/utilities/GoldenReferenceCpu.hpp) | Discovers and runs golden tests using the CPU reference executor |
| GPU runner | [`GoldenReferenceGpu.hpp`](../../../../dnn-providers/miopen-provider/tests/common/GoldenReferenceGpu.hpp) | Same pattern, executes via MIOpen GPU plugin (defined, no tests yet) |
| Python framework | [`reference_data_scripts/utilities/`](../../reference_data_scripts/utilities/) | Generates bundles: defines graphs, runs PyTorch, writes output files |
| Golden data | [`hipdnn_reference_data/`](../../hipdnn_reference_data/BatchnormFwdInference/) | 6 batchnorm bundles across 4 layout/datatype combinations |

### Self-Contained Bundles

All of these components operate on a single shared artifact -- the golden data bundle. A bundle (`{Name}.json` + `{Name}.tensor{uid}.bin`) is self-contained. The graph JSON carries the full computation definition. The `.bin` files carry the raw tensor data (inputs and outputs). Together they are a complete test case. The bundle does not reference any C++ code, any `buildGraph()` function, or any test fixture. If the computation changes, generate a new bundle.

A bundle can be **full** or **graph-only**. A full bundle (`{Name}.json` + `.bin` files) carries pre-computed tensor data — the runner loads it and compares. A graph-only bundle (`{Name}.json` alone, no `.bin` files) carries only the computation definition, no tensor data. The runtime behavior is determined by the test fixture: it generates inputs, runs the engine under test and a reference source (e.g., CPU reference for GPU tests), compares their outputs, and optionally writes the resulting `.bin` files back to produce a full bundle. For reproducible inputs across runs, use fixed seeds.

### Golden Data Format

Golden data uses the existing format already established by `LoadGraphAndTensors.hpp` and `Graph.save()`. The folder path identifies the operation, layout, and data type. The filename identifies the test variant (e.g., tensor size). Each test case is a set of files with a shared base name:

```
{Operation}/{Layout}/{DataType}/{TestName}.json              # Graph definition (operation, tensor metadata, parameters)
{Operation}/{Layout}/{DataType}/{TestName}.tensor{uid}.bin   # Raw tensor data, one file per UID
```

For example, the file `Small.json` in `BatchnormFwdInference/nchw/fp32/` is a batchnorm inference test with small tensors in NCHW layout at fp32 precision:

```
BatchnormFwdInference/nchw/fp32/
  Small.json               # Graph: batchnorm inference, 6 tensors, fp32
  Small.tensor0.bin         # x (input)
  Small.tensor1.bin         # mean (input)
  Small.tensor2.bin         # inv_variance (input)
  Small.tensor3.bin         # scale (input)
  Small.tensor4.bin         # bias (input)
  Small.tensor5.bin         # y (output — golden reference)
```

---

### Generic Test Runner

![Validation Pipeline](images/validation_pipeline.png)

The CPU and GPU runners (`TestGoldenReferenceCpu`, `TestGoldenReferenceGpu`) follow the same three-step pattern:

1. **Load** — read the bundle from disk, separate golden outputs from inputs
2. **Execute** — run the engine under test (CPU reference or MIOpen GPU plugin)
3. **Compare** — check engine output against golden output — PASS or FAIL

Test cases are auto-discovered: `getGoldenReferenceParams()` scans a subdirectory for `.json` files and returns each as a gtest parameter. Adding a bundle to an existing folder is picked up on the next run. Golden tests are not special — the runner must respect the same test-filtering mechanism used by all other tests (e.g., a configuration matrix that determines which operations, shapes, or data types to run or skip). No hard-coded workarounds for skipping golden tests.

---

### Generation Pipeline

![Generation Pipeline](images/generation_pipeline.png)

Golden data is generated by Python scripts in [`reference_data_scripts/`](../../reference_data_scripts/). The reference source is configurable per operation (see [Reference Sources](#reference-sources)); the existing batchnorm generators use PyTorch. Each generator follows the same three-step pattern:

1. **Define** — create graph and input tensors
2. **Compute** — run a reference source (PyTorch, CPU ref, etc.)
3. **Write** — save the bundle (`.json` + `.tensor{uid}.bin`)

To add a new operation, create a node class and generator script following the existing [`generate_batchnorm_reference.py`](../../reference_data_scripts/generate_batchnorm_reference.py) pattern. See [Adding New Golden Reference Tests](#adding-new-golden-reference-tests) for the full workflow.

Only batchnorm has generators and data today. Generators for the remaining operations will be added incrementally.

---

### Reference Sources

The golden data format is **reference-source-agnostic**. Any tool that produces a valid bundle (graph JSON matching the [`graph.fbs`](../../flatbuffers_sdk/schemas/graph.fbs) schema + corresponding `.bin` files) is a valid reference source. The validation pipeline does not know or care what produced the data.

| Category | Examples |
|----------|----------|
| Python frameworks | PyTorch, TensorFlow, JAX |
| In-house references | `CpuReferenceGraphExecutor`, `GpuReferenceGraphExecutor` |
| AMD internal tools | AITER, AOTriton |

---

### Verification Modes

Three modes control which test suites run. In every mode, the **engine** is the thing being tested — the reference source (runtime executor or golden bundle) provides the expected output. `computed` and `golden` are separate gtest suites with different test fixtures, parameterization, and data sources.

| Mode | What runs | Reference source |
|------|-----------|-----------------|
| `computed` | Test-as-code tests (`IntegrationGraphVerificationHarness`) — graph built by `buildGraph()` in C++ | CPU/GPU reference executor at runtime |
| `golden` | Test-as-data tests (`TestGoldenReferenceCpu` / `TestGoldenReferenceGpu`) — graph loaded from disk | Pre-computed data from bundle |
| `both` | Both suites, independently — both must pass | Runtime executor + pre-computed data |

**Floating-point edge case**: `-0.0` vs `+0.0` uses value comparison, not bitwise. NaN handling is covered in [Data Integrity](#data-integrity).

**Architecture note**: all current golden data comes from Python or the CPU reference executor (architecture-independent). If GPU-generated data is added, an architecture guard will be needed. See [Future Work](#future-work).

---

### Tolerance Framework

Tolerances are **always defined in code**, never stored in the data bundle. The codebase already has per-operation dynamic tolerance functions in [`DynamicTolerances.hpp`](../../test_sdk/include/hipdnn_test_sdk/utilities/DynamicTolerances.hpp) (matmul, convolution, batchnorm, layernorm, RMS norm, pointwise). These are used in the computed tests today. The remaining work is to wire them into the golden test runner so a single generic test class handles all operations.

**Acceptance criteria**:
- [ ] Golden data bundles contain no tolerance fields
- [ ] Golden tests use the existing dynamic tolerance functions, not hardcoded `(atol, rtol)` pairs

---

### Data Integrity

Internal consistency is guaranteed by construction: `Graph.save()` writes the JSON and `.bin` files from the same in-memory graph in a single call, so UIDs and tensor data always correspond. `loadGraphAndTensors()` reads UIDs from the JSON and loads the matching `.bin` files. Corruption can only enter after generation — partial downloads, disk errors, or manual edits.

Three checks catch the real failure modes:

1. **File size validation (C++, load time)** — *proposed*. Before reading tensor data, verify that each `.bin` file's size matches `product(dims) * sizeOfDataType(data_type)` from the graph JSON. Catches truncated downloads, wrong-file swaps, and stale files. Cheap — one `stat()` call per tensor. `loadGraphAndTensors()` does not perform this check today; a truncated file silently produces garbage.

2. **Output validation (Python, generation time)** — *proposed*. `Graph.save()` should reject NaN, Inf, and zero-variance output tensors before writing any files. Built into `Graph.save()` itself so no generator can bypass it. A degenerate output (all NaN, all-same-value) makes the test meaningless — everything passes within tolerance.

3. **NaN/Inf rejection (C++, load time)** — *proposed*. Safety net for bundles generated by external tools or before check #2 is added. `loadGraphAndTensors()` should scan output tensors after loading and fail if any contain NaN or Inf.

A standalone CLI verifier will run these checks across a directory tree before bundles are committed, catching errors before they reach CI.

**Acceptance criteria**:
- [ ] All three checks implemented with actionable error messages naming the tensor UID
- [ ] File size mismatch and NaN/Inf in golden data are hard FAILs, not warnings
- [ ] CLI verifier validates full and graph-only bundles offline

---

## Folder Convention

Golden data lives in [`projects/hipdnn/hipdnn_reference_data/`](../../hipdnn_reference_data/). The folder path tells you what a test covers. The file inside tells the engine what to compute. Together they drive test discovery: `getGoldenReferenceParams()` takes a folder path and returns every `.json` file in it as a gtest parameter.

```
hipdnn_reference_data/
  {Operation}/            # e.g., BatchnormFwdInference, ConvFwd, MatmulFwd
    {Layout}/             # e.g., nchw, nhwc, ncdhw
      {DataType}/         # e.g., fp32, fp16, bfp16
        {TestName}.json + {TestName}.tensor{uid}.bin
```

The folder path is a human convention — the loader does not validate that it matches the graph content. The generator is responsible for placing files in the correct folder.

### Naming Conventions

| Level | Convention | Examples |
|-------|-----------|----------|
| Operation | PascalCase, direction suffix | `BatchnormFwdInference`, `ConvFwd`, `ConvBwd`, `MatmulFwd`, `SdpaFwd`, `PointwiseRelu` |
| Layout | Lowercase | `nchw`, `nhwc`, `ncdhw`, `ndhwc` |
| DataType | Lowercase abbreviation | `fp32`, `fp16`, `bfp16` |
| TestName | PascalCase, free-form label | `Small`, `Medium`, `Large` |

When TestName describes tensor size, align with integration test tiers: `Small` for smoke, `Medium` for standard, `Large` for comprehensive/full.

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

`getGoldenReferenceParams("BatchnormFwdInference/nchw/fp32")` resolves the full path as `<exe_dir>/../lib/hipdnn_reference_data/BatchnormFwdInference/nchw/fp32/` and scans it for `.json` files. Each `.json` file becomes a separate gtest parameter — one file, one test case.

To add a test case to an existing folder, drop in a new `.json` + `.bin` set — the next run picks it up automatically. To select tests, use `--gtest_filter` with the file path (operation, layout, data type, or test name).

---

## CLI and Configuration

The test binary will accept two flags (neither exists today). `--verification-mode` controls **what** runs. `--golden-data-dir` controls **where** golden data is read from (ignored when mode is `computed`).

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--vm, --verification-mode` | `computed`, `golden`, `both` | `computed` | Which test suites run |
| `--gd, --golden-data-dir` | path | `<exe_dir>/../lib/hipdnn_reference_data` | Where to find golden data (only used when mode includes golden tests) |

Each flag has an environment variable fallback. The CLI flag takes precedence when both are set.

| Flag | Environment variable |
|------|---------------------|
| `--verification-mode` | `HIPDNN_TEST_VERIFICATION_MODE` |
| `--golden-data-dir` | `HIPDNN_TEST_GOLDEN_DATA_DIR` |

**Acceptance criteria**:
- [ ] Both flags parsed and stored in `TestConfig` singleton
- [ ] Environment variable fallbacks work when CLI flag is absent
- [ ] `--verification-mode golden` with missing golden data directory: hard FAIL with actionable path in the error message
- [ ] `--verification-mode computed` ignores golden data entirely (no fetch, no directory check)

---

## Integration

### CI Integration

| CI Stage | Verification Mode | Golden Data Required |
|----------|-------------------|---------------------|
| Pre-submit (smoke) | `computed` | No |
| Post-submit (full) | `both` | Yes |
| Nightly | `golden` | Yes |

### Adding New Golden Reference Tests

1. **Generate** — write a generation script following the [`generate_batchnorm_reference.py`](../../reference_data_scripts/generate_batchnorm_reference.py) pattern, run it to produce a bundle
2. **Commit** — add the bundle to `hipdnn_reference_data/{Operation}/{Layout}/{DataType}/`. For large tensors, use git-lfs or DVC (see [Data Management](#data-management))
3. **Instantiate** — if the `{Operation}/{Layout}/{DataType}` folder is new, add a one-time `INSTANTIATE_TEST_SUITE_P` in a test `.cpp` file pointing `getGoldenReferenceParams()` at it. Existing folders already have this — no C++ change needed
   ```cpp
   INSTANTIATE_TEST_SUITE_P(, TestBnormFwdFp32, getGoldenReferenceParams("BatchnormFwdInference/nchw/fp32"));
   ```

---

## Data Management

Golden data lives in `projects/hipdnn/hipdnn_reference_data/` (see [Folder Convention](#folder-convention)). CMake copies it to the build tree at configure time and installs it to `<prefix>/lib/hipdnn_reference_data/`. At runtime, `getGoldenReferenceParams()` resolves the path as `<exe_dir>/../lib/hipdnn_reference_data/` — the standard ROCm layout where executables are in `bin/` and data is in `lib/`. The `--golden-data-dir` CLI flag and `HIPDNN_TEST_GOLDEN_DATA_DIR` env var override this default.

**Note**: because CMake uses `file(COPY)` at configure time, adding new golden data requires re-running CMake (not just rebuilding) to update the build tree.

The existing batchnorm data (small tensors) is committed directly to git. Golden data can grow large (a single test case with `8x512x64x64` fp32 tensors is ~64 MB), so large datasets will need external storage (git-lfs, DVC, or similar). The storage tool decision is out of scope for this RFC.

---

## Risk Register

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| FlatBuffers schema change | Old JSON bundles unreadable by `loadGraphAndTensors()` | Medium | Update Python scripts to match new schema, then regenerate all bundles |
| Reference script bug freezes wrong data | Silent incorrect baseline | Medium | Cross-validate against C++ CPU ref; review generated data before committing; [proposed generation-time validation](#data-integrity) will reject degenerate outputs |
| PyTorch version drift | Different versions produce slightly different outputs | Low | Pin PyTorch version in `requirements.txt`; regenerate when upgrading |
| Large golden data sets slow CI | CI feedback loop degrades | Low | Storage caching, selective fetch by test filter, compression (future) |
| Remote storage unavailable | Golden-mode CI fails | Low | Computed-mode CI is independent of storage; CI fallback to computed-only |

---

## Known Limitations

Comparison testing can confirm that two implementations agree, not that either is correct. If the reference executor and the engine under test share the same bug, the test passes. Future work on mathematical invariant checks and hand-verified micro cases addresses this gap without changing the golden data format.

---

## Future Work

1. **Dynamic tolerance integration**: Wire the existing `DynamicTolerances` functions (matmul, conv, batchnorm, layernorm, RMS norm, pointwise) into the golden test runner so a single generic test class handles all operations.
2. **Automatic test discovery**: Recursive scanning of `hipdnn_reference_data/` to auto-generate test instantiations, eliminating manual `INSTANTIATE_TEST_SUITE_P`.
3. **C++ graph export**: Utility to export a graph from an existing test-as-code `buildGraph()` to the bundle format, enabling conversion of computed tests to golden tests.
4. **Bundle inspection tool**: CLI that reads bundles and reports tensor metadata and statistics (integrity validation is covered in [Data Integrity](#data-integrity)).
5. **External data validation**: Because bundles are self-contained and tool-agnostic, external parties (customers, partner teams) could submit their own input+output tensor data for a given graph and validate it against golden references — or vice versa — without any C++ code. A Python-only comparison tool could load two bundles with the same graph and diff their output tensors.
6. **Multi-source consensus gate**: Before freezing golden data, run the same graph through multiple independent reference sources (PyTorch, CPU ref, GPU ref) and only save the bundle when all sources agree within tolerance. This catches single-source bugs that the current single-reference generation cannot detect.
7. **Bundle metadata**: Record provenance information (generator tool and version, reference source, ROCm version, generation timestamp) in the bundle for debugging and auditing. The runner does not need this to execute — it is for human use when investigating failures.
8. **Reproducible generation**: Fixed seeds for random input generation so that regenerating a bundle (after a schema change, PyTorch upgrade, or generator fix) produces the same inputs, isolating output differences to the reference source change.
9. **Mathematical invariant checks and hand-verified micro cases**: Per-operation invariants (e.g., softmax rows sum to 1) and hand-computed expected outputs that catch bugs comparison testing cannot. Separate RFC.
