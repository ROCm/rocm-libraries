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
   - [Workflows](#workflows)
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

**Design principle — test identity comes from the graph, not the filesystem.** The graph JSON contains everything that defines a test: operation type, tensor shapes, data types, parameters. The runner derives the test name from the graph content, not from the folder path where the bundle is stored. This decouples test identity from storage layout, which means: folders can be reorganized without breaking filters, TOML skip rules match stable graph-derived names, and customer-submitted bundles get meaningful test names regardless of where they're dropped.

**[Generation](#generation-pipeline) (run once, any tool):**
1. Define graph and create input tensors
2. Run a reference source (PyTorch, CPU ref, etc.) to produce outputs
3. Write the bundle

**[Validation](#generic-test-runner) (every CI run, any tool that can load bundles):**
1. Load bundle from disk, extract golden outputs
2. Execute engine under test
3. Compare engine output to golden output — PASS or FAIL

---

## Detailed Design

### Existing Infrastructure

The golden reference infrastructure is already built and working for batchnorm. The table below summarizes each existing component. This RFC **keeps** the bundle format, core loader, Python framework, test runner pattern, folder structure, and `INSTANTIATE_TEST_SUITE_P` discovery mechanism. It **adds** graph-derived test identity, a golden-first fallback chain, a bundle verifier, data integrity checks, per-engine tolerance overrides, and CI integration. It **changes** the data location (moves to the integration suite) and the default verification mode (golden-first instead of computed-first).

| Component | File(s) | Role |
|-----------|---------|------|
| Core loader | [`LoadGraphAndTensors.hpp`](../../test_sdk/include/hipdnn_test_sdk/utilities/LoadGraphAndTensors.hpp) | Reads bundles from disk, separates inputs from expected outputs, validates results |
| CPU runner | [`GoldenReferenceCpu.hpp`](../../test_sdk/tests/utilities/GoldenReferenceCpu.hpp) | Base fixture + `getGoldenReferenceParams(subDir)` — scans a directory for `.json` files, each becomes a gtest parameter |
| GPU runner | [`GoldenReferenceGpu.hpp`](../../../../dnn-providers/miopen-provider/tests/common/GoldenReferenceGpu.hpp) | Same pattern, executes via MIOpen GPU plugin (defined, no tests yet) |
| Test instantiation | [`TestCpuFpReferenceBatchnorm.cpp`](../../test_sdk/tests/utilities/TestCpuFpReferenceBatchnorm.cpp) | One class + `INSTANTIATE_TEST_SUITE_P` per operation/layout/datatype — e.g., `getGoldenReferenceParams("BatchnormFwdInference/nchw/fp32")` |
| Python framework | [`reference_data_scripts/utilities/`](../../reference_data_scripts/utilities/) | Generates bundles: defines graphs, runs PyTorch, writes output files |
| Golden data | [`hipdnn_reference_data/`](../../hipdnn_reference_data/BatchnormFwdInference/) | 6 batchnorm bundles across 4 layout/datatype combinations |

### Self-Contained Bundles

All of these components operate on a single shared artifact -- the golden data bundle. A bundle (`{Name}.json` + `{Name}.tensor{uid}.bin`) is self-contained. The graph JSON carries the full computation definition. The `.bin` files carry the raw tensor data (inputs and outputs). Together they are a complete test case. The bundle does not reference any C++ code, any `buildGraph()` function, or any test fixture. If the computation changes, generate a new bundle.

The bundle format is **deliberately independent of any test infrastructure**. The JSON follows the FlatBuffers `graph.fbs` schema; the `.bin` files are raw contiguous tensors. Any tool that can parse JSON and read binary can consume bundles — they are not tied to GTest, `TestGoldenReferenceCpu`, or the `getGoldenReferenceParams()` discovery mechanism. Examples of other consumers:

- **External test harnesses** — a partner's internal test framework loads the same bundles to validate their engine
- **Python validation scripts** — a PyTorch script loads a bundle, re-runs the computation, compares
- **Standalone CLI tools** — the proposed [bundle inspection tool](#generic-test-runner) and future [bundle-to-bundle comparison tool](#future-work) consume bundles directly
- **CI systems** — a non-GTest CI pipeline loads bundles, invokes an engine, checks results

A bundle can be **full** or **graph-only**. A full bundle (`{Name}.json` + `.bin` files) carries pre-computed tensor data — the runner loads it and compares. A graph-only bundle (`{Name}.json` alone, no `.bin` files) carries only the computation definition, no tensor data. The runtime behavior is determined by the test fixture: it generates inputs, runs the engine under test and a reference source (e.g., CPU reference for GPU tests), compares their outputs, and optionally writes the resulting `.bin` files back to produce a full bundle. For reproducible inputs across runs, use fixed seeds.

### Golden Data Format

The bundle format is a convention, not a library API. It uses the existing format already established by `LoadGraphAndTensors.hpp` (C++ reader) and `Graph.save()` (Python writer), but any tool that follows the same convention can produce or consume bundles. A bundle is a set of files with a shared base name: one `.json` file (graph definition conforming to the [`graph.fbs`](../../flatbuffers_sdk/schemas/graph.fbs) schema) and one `.tensor{uid}.bin` file per tensor (raw contiguous data matching the tensor's declared `data_type`, `dims`, and `strides`). All files sit in the same directory — no wrapper folder.

```
{Name}.json              # Graph definition (operation, tensor metadata, parameters)
{Name}.tensor{uid}.bin   # Raw tensor data, one file per UID
```

For example, the existing batchnorm inference bundles at `BatchnormFwdInference/nchw/fp32/`:

```
BatchnormFwdInference/nchw/fp32/
  Small.json               # Graph: batchnorm inference, 6 tensors, fp32
  Small.tensor0.bin         # x (input)
  Small.tensor1.bin         # mean (input)
  Small.tensor2.bin         # inv_variance (input)
  Small.tensor3.bin         # scale (input)
  Small.tensor4.bin         # bias (input)
  Small.tensor5.bin         # y (output — golden reference)
  Large.json               # Same operation, larger tensors
  Large.tensor0.bin
  ...
  MIOpen.json              # Same operation, MIOpen-specific shapes
  MIOpen.tensor0.bin
  ...
```

Multiple bundles coexist in the same directory, distinguished by their base name (`Small`, `Large`, `MIOpen`). See [Folder Convention](#folder-convention) for the full directory structure.

#### Binary tensor format

Each `.tensor{uid}.bin` file is a **raw dump of the tensor's underlying storage** — no header, no metadata, no framing. The graph JSON carries all the metadata needed to interpret the bytes:

| Property | Source | Details |
|----------|--------|---------|
| Element type | `data_type` field in JSON | `FLOAT` = 4 bytes, `HALF` = 2 bytes, `BFLOAT16` = 2 bytes, etc. |
| Dimensions | `dims` field in JSON | Element counts per dimension (e.g., `[1, 3, 224, 224]`) |
| Strides | `strides` field in JSON | **Element strides**, not byte strides (e.g., `[150528, 50176, 224, 1]`). Multiply by `sizeof(element_type)` for byte offsets. |
| Byte order | Native platform | Little-endian on x86-64. No endianness marker in the file. |

The file size in bytes equals `element_space × sizeof(element_type)`, where `element_space` is computed from `dims` and `strides` (the total storage footprint including any gaps for non-contiguous layouts). For a contiguous (packed) tensor, `element_space` equals the product of `dims`.

**To read a `.bin` file**: allocate `element_space × sizeof(T)` bytes, read the file into that buffer, then index using the strides from JSON. For contiguous tensors this is a straightforward row-major (C-order) array. The Python writer uses PyTorch's `untyped_storage()` and the C++ reader uses `memcpy` into a pre-allocated buffer — both operate on raw bytes with no interpretation.

---

### Generic Test Runner

![Validation Pipeline](images/validation_pipeline.png)

The CPU and GPU runners (`TestGoldenReferenceCpu`, `TestGoldenReferenceGpu`) follow the same three-step pattern:

1. **Load** — read the bundle from disk, separate golden outputs from inputs
2. **Execute** — run the engine under test (CPU reference or MIOpen GPU plugin)
3. **Compare** — check engine output against golden output — PASS or FAIL

Test discovery follows the same `INSTANTIATE_TEST_SUITE_P` + parameterized fixture pattern used by all other integration tests. Each operation/layout/datatype combination has a test class that inherits from `TestGoldenReferenceCpu` (or `TestGoldenReferenceGpu`) and an `INSTANTIATE_TEST_SUITE_P` call that points `getGoldenReferenceParams()` at the corresponding subdirectory. Within that directory, every `.json` file becomes a test case automatically — adding a new bundle to an existing directory requires no C++ changes.

```cpp
// Existing pattern — one class + instantiation per operation/layout/datatype
INSTANTIATE_TEST_SUITE_P(,
    TestCpuBatchnormFwdInferenceGoldenReferenceNchwFp32,
    getGoldenReferenceParams("BatchnormFwdInference/nchw/fp32"));
```

Tiers are assigned via the GTest prefix on `INSTANTIATE_TEST_SUITE_P`, exactly like all other integration tests — `Smoke` (or no prefix) for smoke, `Standard` for standard, etc. Golden tests are not special — they must respect the same test-filtering mechanism used by all other tests.

A **bundle inspection tool** will validate that bundles are well-formed and runnable. It reads bundles, reports tensor metadata and statistics, and can verify that the graph + tensors will load correctly before they reach CI.

---

### Generation Pipeline

![Generation Pipeline](images/generation_pipeline.png)

Golden data is generated by Python scripts in [`reference_data_scripts/`](../../reference_data_scripts/). The reference source is configurable per operation (see [Reference Sources](#reference-sources)); the existing batchnorm generators use PyTorch. Each generator follows the same three-step pattern:

1. **Define** — create graph and input tensors
2. **Compute** — run a reference source (PyTorch, CPU ref, etc.)
3. **Write** — save the bundle (`.json` + `.tensor{uid}.bin`)

To add a new operation, create a node class and generator script following the existing [`generate_batchnorm_reference.py`](../../reference_data_scripts/generate_batchnorm_reference.py) pattern. See [Workflows](#workflows) for the full workflow.

The generator should **auto-derive the output path** from the graph content. The graph JSON already contains the operation type, tensor layouts, and data types — exactly the `{Operation}/{Layout}/{DataType}/` structure used by the [folder convention](#folder-convention). The developer supplies only the **bundle name**; the generator computes the directory:

```bash
python generate_batchnorm_reference.py --name Small
# → BatchnormFwdInference/nchw/fp32/Small.json + .bin
```

Only batchnorm has generators and data today. Generators for the remaining operations will be added incrementally.

---

### Reference Sources

The golden data format is **reference-source-agnostic**. Any tool that produces a valid bundle (graph JSON matching the [`graph.fbs`](../../flatbuffers_sdk/schemas/graph.fbs) schema + corresponding `.bin` files) is a valid reference source. The validation pipeline does not know or care what produced the data.

| Category | Examples | Portability |
|----------|----------|-------------|
| Python frameworks | PyTorch, TensorFlow, JAX | Architecture-independent — bundles valid on any GPU |
| In-house CPU reference | `CpuReferenceGraphExecutor` | Architecture-independent |
| In-house GPU reference | `GpuReferenceGraphExecutor` | **Architecture-dependent** — bundles tied to the GPU that generated them |
| AMD internal tools | AITER, AOTriton | Architecture-dependent |

---

### Verification Modes

The default verification strategy is **golden-first with automatic fallback**. For each test, the runner selects the best available reference source in order:

1. **Golden data** — if a bundle exists for this test, use it
2. **GPU reference** — if no golden data but a GPU reference executor is available, use it
3. **CPU reference** — last resort, always available

The `--verification-mode` flag overrides this default. In golden mode, tests without golden data are **skipped** (not failed). In every mode, the **engine** is the thing being tested — the reference source provides the expected output.

| Mode | What runs | Reference source |
|------|-----------|-----------------|
| `auto` (default) | Per-test fallback: golden → GPU ref → CPU ref | Best available |
| `computed` | Test-as-code tests (`IntegrationGraphVerificationHarness`) — graph built by `buildGraph()` in C++ | CPU/GPU reference executor at runtime |
| `golden` | Test-as-data tests (`TestGoldenReferenceCpu` / `TestGoldenReferenceGpu`) — graph loaded from disk; tests without golden data are skipped | Pre-computed data from bundle |
| `both` | Both suites, independently — both must pass | Runtime executor + pre-computed data |

**Floating-point edge case**: `-0.0` vs `+0.0` uses value comparison, not bitwise. NaN handling is covered in [Data Integrity](#data-integrity).

---

### Tolerance Framework

Tolerances are **always defined in code and configuration**, never stored in the data bundle. The tolerance selection priority is:

1. **TOML per-engine overrides** (highest priority) — each engine's TOML config (e.g., [`MIOPEN_ENGINE.toml`](../../../dnn-providers/miopen-provider/config/MIOPEN_ENGINE.toml)) can declare `[[tolerance_overrides]]` entries with glob-pattern filters and atol/rtol values. The integration harness already queries these via `TestConfig::findToleranceOverride(testName)`. Golden tests must use the same mechanism so that per-engine tolerance exceptions apply uniformly. TOML overrides remain the highest priority at every stage — even after dynamic tolerances are adopted — because a specific engine may legitimately exceed the mathematical bound for certain operations (e.g., a fused kernel trading precision for throughput).

2. **Default tolerance** (fallback when no TOML override matches):
   - **Initially: fixed per-operation tolerances** — [`TestTolerances.hpp`](../../test_sdk/include/hipdnn_test_sdk/utilities/TestTolerances.hpp) defines compile-time tolerances per operation and data type (e.g., batchnorm inference fp32 = `2e-4`, conv fwd fp32 = `1e-5`). The integration harness selects these based on the root operation in the graph. Golden tests will initially use these same fixed values.
   - **Target: dynamic tolerances** — [`DynamicTolerances.hpp`](../../test_sdk/include/hipdnn_test_sdk/utilities/DynamicTolerances.hpp) computes tolerances from tensor dimensions using Higham error bounds. Functions exist for conv, matmul, batchnorm, layernorm, RMS norm, and pointwise but are not yet wired into the golden test runner. Dynamic tolerances will replace fixed values as the default, giving tighter bounds for small tensors and appropriately looser bounds for large ones.

The end-state priority chain: **TOML override → dynamic tolerance → fixed tolerance (legacy fallback)**. TOML overrides serve a fundamentally different purpose from dynamic tolerances: dynamic tolerances set the *mathematically expected* error bound for an operation; TOML overrides grant *engine-specific exceptions* when a particular engine exceeds that bound.

**Current gap**: The golden ref framework (`TestGoldenReferenceCpu`) takes tolerances as hard-coded function parameters — it does not go through the TOML override system or the operation-based tolerance selection in the integration harness. Golden tests must be connected to the same tolerance flow.

**Acceptance criteria**:
- [ ] Golden data bundles contain no tolerance fields
- [ ] Golden tests use the same tolerance selection as all other tests: TOML override → default tolerance (fixed initially, dynamic later)
- [ ] Per-engine TOML tolerance overrides apply to golden tests — no separate override mechanism

---

### Data Integrity

Internal consistency is guaranteed by construction: `Graph.save()` writes the JSON and `.bin` files from the same in-memory graph in a single call, so UIDs and tensor data always correspond. `loadGraphAndTensors()` reads UIDs from the JSON and loads the matching `.bin` files. Corruption can only enter after generation — partial downloads, disk errors, or manual edits.

Four checks catch the real failure modes:

1. **Per-tensor SHA-256 checksum (Python, generation time; C++, load time)** — *proposed*. `Graph.save()` computes a SHA-256 hash of each `.bin` file and stores it in the tensor's JSON entry. At load time, `loadGraphAndTensors()` recomputes the hash and rejects mismatches. This catches truncated downloads, wrong-file swaps, file corruption, and accidental mixing of `.bin` files from different bundles. It also solves the orphaned-file problem: given a `.bin` file separated from its JSON, compute its SHA-256 and search for a matching entry across available JSONs.

2. **Tensor size validation (C++, load time)** — *proposed*. After loading tensor data, verify that the file size in bytes equals `element_space × sizeof(element_type)` (see [binary tensor format](#binary-tensor-format)). Catches truncated downloads and wrong-type swaps. `loadGraphAndTensors()` does not perform this check today; a truncated file silently produces garbage.

3. **NaN/Inf rejection (Python, generation time)** — *proposed*. `Graph.save()` should reject output tensors containing NaN or Inf before writing any files. Built into `Graph.save()` itself so no generator can bypass it. All-same-value tensors are valid (e.g., a bias-only layer can produce uniform output).

4. **NaN/Inf rejection (bundle verifier, offline)** — *proposed*. Safety net for bundles generated by external tools or before check #3 is added. Run by the CLI verifier before commit, not at test load time — scanning every tensor at runtime adds overhead the test runner should not pay. The existing test validators already catch NaN/Inf from the engine's computation.

A standalone CLI verifier will run these checks across a directory tree before bundles are committed, catching errors before they reach CI.

**Acceptance criteria**:
- [ ] All four checks implemented with actionable error messages naming the tensor UID
- [ ] File size mismatch and NaN/Inf in golden data are hard FAILs, not warnings
- [ ] CLI verifier validates full and graph-only bundles offline

---

## Folder Convention

The existing folder structure organizes bundles by `{Operation}/{Layout}/{DataType}/`:

```
hipdnn_reference_data/
  {Operation}/
    {Layout}/
      {DataType}/
        {Name}.json + {Name}.tensor{uid}.bin
```

The root directory is `<exe_dir>/../lib/hipdnn_reference_data` (set by CMake `file(COPY ...)` at configure time and `install(DIRECTORY ...)` at install time). At runtime, `--golden-data-dir` or `HIPDNN_TEST_GOLDEN_DATA_DIR` can override it.

Each `INSTANTIATE_TEST_SUITE_P` call points `getGoldenReferenceParams()` at a specific subdirectory. The function does a **shallow scan** of that directory for `.json` files — each file becomes a gtest parameter. Multiple bundles coexist in the same directory, distinguished by base name (`Small`, `Large`, `MIOpen`).

### Naming Guidelines

| Level | Convention | Examples |
|-------|-----------|----------|
| Operation | PascalCase, direction suffix | `BatchnormFwdInference`, `ConvFwd`, `ConvBwd`, `MatmulFwd` |
| Layout | Lowercase | `nchw`, `nhwc`, `ncdhw`, `ndhwc` |
| DataType | Lowercase abbreviation | `fp32`, `fp16`, `bfp16` |
| BundleName | PascalCase, free-form label | `Small`, `Medium`, `Large`, `MIOpen` |

BundleName is a **human convenience label** — it is not stored in the graph JSON and cannot be derived from graph content. The graph itself (operation, tensor shapes, data types, parameters) is the true identity of the test case. BundleName exists only for human navigation and to group the `.json` + `.bin` file set on disk.

### Example (current layout)

```
hipdnn_reference_data/
  BatchnormFwdInference/
    nchw/
      fp32/
        Small.json + Small.tensor{0..5}.bin
        Large.json + Large.tensor{0..5}.bin
        MIOpen.json + MIOpen.tensor{0..5}.bin
      fp16/
        Small.json + Small.tensor{0..5}.bin
      bfp16/
        Small.json + Small.tensor{0..5}.bin
    ncdhw/
      fp32/
        Small.json + Small.tensor{0..5}.bin
```

### How Test Discovery Works

`getGoldenReferenceParams(subDir)` scans the given subdirectory for `.json` files and returns them as gtest parameters. Each `.json` file becomes a separate test case. Adding a new bundle to an existing directory (e.g., dropping `CustomerIssue42.json` + `.bin` files into `BatchnormFwdInference/nchw/fp32/`) requires no C++ changes — the next run picks it up automatically.

Adding a **new operation** requires a new C++ test class and `INSTANTIATE_TEST_SUITE_P` call, following the existing pattern in [`TestCpuFpReferenceBatchnorm.cpp`](../../test_sdk/tests/utilities/TestCpuFpReferenceBatchnorm.cpp).

**Tier assignment** follows the same mechanism as all other integration tests: the GTest prefix on `INSTANTIATE_TEST_SUITE_P`. See the [Test Tiers](../../../dnn-providers/integration-tests/README.md#test-tiers) section in the integration tests README.

Since `getGoldenReferenceParams()` scans an entire directory, **all bundles in a directory share the same tier**. To assign different tiers to different bundles, separate them into subdirectories:

```
BatchnormFwdInference/nchw/fp32/          # small bundles → smoke
BatchnormFwdInference/nchw/fp32/large/    # large bundles → standard
```

```cpp
// Smoke tier — no prefix, scans the base directory
INSTANTIATE_TEST_SUITE_P(,
    TestCpuBnFwdInfNchwFp32,
    getGoldenReferenceParams("BatchnormFwdInference/nchw/fp32"));

// Standard tier — Standard prefix, scans a subdirectory with larger bundles
INSTANTIATE_TEST_SUITE_P(Standard,
    TestCpuBnFwdInfNchwFp32Large,
    getGoldenReferenceParams("BatchnormFwdInference/nchw/fp32/large"));
```

Today all existing bundles (`Small`, `Large`, `MIOpen`) sit in the same directory with no prefix, so they all run in smoke.

To select tests, use `--gtest_filter`:

```
--gtest_filter=*BatchnormFwdInference*              # all batchnorm inference golden tests
--gtest_filter=*BatchnormFwdInference*fp32*          # batchnorm inference fp32 only
--gtest_filter=*Small*                               # all bundles named "Small"
```

---

## CLI and Configuration

The test binary will accept two flags (neither exists today). `--verification-mode` controls **what** runs. `--golden-data-dir` controls **where** golden data is read from (ignored when mode is `computed`).

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--vm, --verification-mode` | `auto`, `computed`, `golden`, `both` | `auto` | Which verification strategy to use (see [Verification Modes](#verification-modes)) |
| `--gd, --golden-data-dir` | path | `<exe_dir>/../lib/hipdnn_reference_data` | Where to find golden data (only used when mode includes golden tests) |

Each flag has an environment variable fallback. The CLI flag takes precedence when both are set.

| Flag | Environment variable |
|------|---------------------|
| `--verification-mode` | `HIPDNN_TEST_VERIFICATION_MODE` |
| `--golden-data-dir` | `HIPDNN_TEST_GOLDEN_DATA_DIR` |

**Acceptance criteria**:
- [ ] Both flags parsed and stored in `TestConfig` singleton
- [ ] Environment variable fallbacks work when CLI flag is absent
- [ ] `--verification-mode golden` skips tests that have no golden data
- [ ] `--verification-mode computed` ignores golden data entirely (no fetch, no directory check)

---

## Integration

### CI Integration

Golden tests follow the same [tier cascade](../../../dnn-providers/integration-tests/README.md#how-tiers-cascade) as all other integration tests. Tiers are assigned via GTest prefix on `INSTANTIATE_TEST_SUITE_P`, not by folder structure.

| CI Stage | ctest Command | Verification Mode | Notes |
|----------|--------------|-------------------|-------|
| Pre-submit (smoke) | `ctest -L quick` | `auto` | Golden tests with no prefix or `Smoke` prefix; tests without golden data fall back to computed |
| Post-submit | `ctest -L standard` | `both` | Smoke + `Standard`-prefixed golden tests; both suites must pass |
| Nightly | `ctest -L comprehensive` | `golden` | All tiers up to comprehensive; tests without golden data are skipped |
| Weekly | `ctest -L full` | `golden` | All tiers |

### Workflows

**Add a new bundle to an existing operation** (developer):
1. Write a generation script following the [`generate_batchnorm_reference.py`](../../reference_data_scripts/generate_batchnorm_reference.py) pattern, run it to produce a bundle
2. Run the bundle inspection tool to verify the bundle is well-formed
3. Place the `.json` + `.bin` files in the matching `{Operation}/{Layout}/{DataType}/` directory and commit (DVC for large tensors). No C++ changes needed — the existing `INSTANTIATE_TEST_SUITE_P` picks up new bundles automatically

**Add a new operation** (developer):
1. Generate bundles as above
2. Add a C++ test class inheriting from `TestGoldenReferenceCpu` (or `TestGoldenReferenceGpu`)
3. Add `INSTANTIATE_TEST_SUITE_P` calls pointing at the new directories, with appropriate tier prefixes

**Debug a customer issue** (support):
1. Receive the customer's bundle (`.json` + `.bin` files) — no source code or NDA required
2. Drop the files into the matching operation directory (or any directory already scanned by an `INSTANTIATE_TEST_SUITE_P`)
3. Run tests — the runner picks it up, executes the engine, compares against golden output
4. Inspect the diff: which tensors diverge, by how much, at which indices

**Reproduce a CI failure locally** (developer):
1. Pull the golden data via DVC
2. Run the failing test with `--gtest_filter=*OperationName*DataType*`
3. The bundle is self-contained — no environment-specific setup beyond the engine under test

---

## Data Management

Golden data will live with the integration test suite (exact path TBD — see open question below). At runtime, the `--golden-data-dir` CLI flag or `HIPDNN_TEST_GOLDEN_DATA_DIR` env var points the runner to the data location.

The existing batchnorm data (small tensors) is committed directly to git. Larger golden data will be stored in **DVC**, which the repo already uses for large binary assets. Golden data can grow large (a single test case with `8x512x64x64` fp32 tensors is ~64 MB), so DVC is the natural fit.

**Open question**: How to ship golden data to ROCm CI. The data must be available at test time without bloating the build tree. Options include DVC pull at CI time, a pre-staged CI cache, or a separate data artifact. Input from the broader team is needed here.

---

## Risk Register

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| FlatBuffers schema change | Old JSON bundles unreadable by `loadGraphAndTensors()` | Low | Schema changes must be backwards compatible — old graphs continue to work when new fields are added. FlatBuffers supports this natively via optional fields |
| Reference script bug freezes wrong data | Silent incorrect baseline | Medium | Cross-validate against C++ CPU ref; review generated data before committing; [proposed generation-time validation](#data-integrity) will reject degenerate outputs |
| PyTorch version drift | Different versions produce slightly different outputs | Low | Pin PyTorch version in `requirements.txt`; regenerate when upgrading |
| Large golden data sets slow CI | CI feedback loop degrades | Low | Storage caching, selective fetch by test filter, compression (future) |
| Remote storage unavailable | Golden-mode CI fails | Low | Computed-mode CI is independent of storage; CI fallback to computed-only |

---

## Known Limitations

Comparison testing can confirm that two implementations agree, not that either is correct. If the reference executor and the engine under test share the same bug, the test passes.

---

## Future Work

1. **Dynamic tolerance integration**: Wire the existing `DynamicTolerances` functions (matmul, conv, batchnorm, layernorm, RMS norm, pointwise) into the golden test runner so a single generic test class handles all operations.
2. **Engine-as-bundle-producer**: Any engine (MIOpen, CPU reference, future providers) can take a graph-only bundle as input, execute the graph, and write the results back as a full bundle. The codebase can already export graphs to JSON via `Graph.save()`; the remaining work is a produce-bundle mode in the test harness or a standalone CLI. Once every engine can produce bundles, validation becomes: feed the same graph-only bundle to two engines, compare the two output bundles. No reference source concept needed at comparison time — each engine is just a producer.
3. **Bundle-to-bundle comparison**: A standalone tool that loads two bundles and diffs their output tensors directly — no engine execution at comparison time. Matching is by **graph content** (operation, tensor shapes, data types, parameters), not by filename — bundle A called `Small.json` and bundle B called `Run42.json` are comparable if their graphs describe the same computation. The tool supports two modes: **directional** (`--expected A.json --actual B.json`) where one bundle is truth and tolerances apply to the deviation from it, and **symmetric** (`--bundle1 A.json --bundle2 B.json`) where neither is truth and the tool just reports the delta. Combined with #2, this completes the loop: generate a bundle from engine A, generate a bundle from engine B (or PyTorch, or a customer's framework), compare. Use cases include cross-engine validation, PyTorch version upgrades, and external parties submitting bundles without access to any engine or C++ toolchain.
4. **Reproducible generation**: Fixed seeds for random input generation so that regenerating a bundle (after a schema change, PyTorch upgrade, or generator fix) produces the same inputs, isolating output differences to the reference source change.
5. **Auto-tier classification**: The generator could suggest the appropriate GTest prefix (`Smoke`, `Standard`, `Comprehensive`, `Full`) based on tensor element counts — matching the existing `getSmall`/`getMedium`/`getLargeEdge`/`getLargeStress` convention. Needs formalized size thresholds and an override for cases where tensor size alone doesn't predict test duration.
